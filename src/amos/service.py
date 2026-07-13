"""High-level AMOS v1 service API."""

from __future__ import annotations

import json
import math
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import AccessDenied, CASConflict, IdempotencyConflict, ValidationError
from .maintenance import (
    EvidenceWindow,
    MaintenanceProcessor,
    SEMANTIC_RELATION_PROCESSOR_ID,
    SEMANTIC_RELATION_PROCESSOR_VERSION,
    default_processor_registry,
    load_maintenance_processor,
    proposal_is_auto_committable,
    semantic_relation_proposals_from_facets,
)
from .schemas import (
    EDGE_RELATIONS,
    SCHEMA_VERSION,
    canonical_json,
    confidence_score,
    digest,
    normalize_atom,
    normalize_evidence,
    normalize_relation,
    stable_id,
    utc_now,
)
from .smp import SemanticMaintenanceProcessor, cosine
from .store import SQLiteStore


LOW_HEALTH_STATES = {"confounding", "low_utility", "merged", "orphaned", "stale"}
CONFLICT_RELATIONS = {"rel:contradicts"}
HIGH_RISK_MAINTENANCE = {
    "delete",
    "hard_delete",
    "irreversible_delete",
    "merge_without_alias",
    "rewrite_confidence_policy",
}
DEFAULT_PACKET_PROFILES = {
    "reasoner": {"max_items": 24, "tokens": 6000, "include_conflicts": True},
    "planner": {"max_items": 20, "tokens": 4500, "include_conflicts": False},
    "executor": {"max_items": 16, "tokens": 3500, "include_conflicts": False},
    "critic": {"max_items": 32, "tokens": 8000, "include_conflicts": True},
    "steward": {"max_items": 64, "tokens": 12000, "include_conflicts": True},
    "self_awareness": {"max_items": 100, "tokens": 24000, "include_conflicts": True},
    "shared_coordination": {"max_items": 48, "tokens": 9000, "include_conflicts": True},
    "agentic_recall": {"max_items": 40, "tokens": 7000, "include_conflicts": True},
}
DEFAULT_MEMORY_POLICY = {
    "enabled": True,
    "schedule": {
        "every_graph_versions": 25,
        "every_seconds": 300,
        "run_on_pressure": True,
    },
    "maintenance": {
        "enabled": True,
        "run_smp": True,
        "max_smp_atoms": 128,
        "run_steward": True,
        "rebuild_indexes": True,
        "rebuild_lsa": True,
        "lsa_dimensions": 32,
        "lsa_max_terms": 300,
        "invalidate_packet_cache": True,
    },
    "distillation": {
        "enabled": True,
        "min_source_atoms": 6,
        "max_source_atoms": 10,
        "candidate_types": [
            "action_outcome",
            "agentic_trace",
            "belief",
            "episode",
            "preference",
        ],
        "distillation_type": "automatic_policy",
        "archive_sources": False,
        "approved_by": None,
    },
    "maintenance_distiller": {
        "enabled": True,
        "auto_commit_low_risk": True,
        "processor_ids": [],
        "domain": "generic",
        "max_atoms": 128,
        "max_events": 64,
        "max_retrieval_outcomes": 64,
        "reviewer": {
            "enabled": False,
            "authority": "draft_only",
        },
    },
    "decay": {
        "enabled": True,
        "max_atoms": 256,
        "require_atom_policy": True,
        "pressure_archive_policyless": True,
        "pressure_max_archives_per_run": 256,
        "pressure_protected_types": ["commitment", "policy", "self_model"],
        "capacity_assessment_targets": [256, 512, 768],
        "capacity_headroom_ratio": 0.2,
        "archive_superseded": True,
        "archive_superseded_after_seconds": 0,
        "mark_stale_after_seconds": None,
        "archive_after_seconds": None,
        "low_utility_threshold": None,
    },
    "storage_cleanup": {
        "enabled": True,
        "trigger": "idle",
        "idle_after_seconds": 300,
        "min_interval_seconds": 900,
        "max_deletions_per_tick": 256,
        "remove_archived_from_hot_index": True,
        "remove_stale_from_hot_index": True,
        "delete_archived_after_seconds": 604800,
        "delete_stale_after_seconds": 1209600,
        "protected_types": ["policy", "self_model", "commitment"],
        "compact_idempotency_after_seconds": 604800,
        "max_idempotency_compactions_per_tick": 512,
        "sqlite_compaction": {
            "checkpoint_wal": True,
            "checkpoint_mode": "TRUNCATE",
            "vacuum_enabled": True,
            "vacuum_idle_after_seconds": 1800,
            "vacuum_min_interval_seconds": 86400,
        },
    },
}
SEARCH_INDEX_REF = "amos.v1.search"
SEARCH_INDEX_SCHEMA = "amos.v1.search.payload_values.v2"
ATTENTION_POLICY_ID = "amos.v1.attention.default"
RETRIEVAL_RECENCY_HORIZON_SECONDS = 30 * 24 * 60 * 60
RETRIEVAL_WEIGHTS = {
    "direct_cue_match": 0.22,
    "semantic_similarity": 0.14,
    "edge_activation": 0.12,
    "recency": 0.08,
    "confidence": 0.12,
    "utility": 0.12,
    "salience": 0.08,
    "scope_specificity": 0.06,
    "goal_relevance": 0.08,
    "procedural_applicability": 0.04,
    "attention_focus": 0.14,
    "attention_type_boost": 0.08,
    "attention_counterevidence": 0.08,
    "attention_novelty": 0.05,
    "attention_suppression_penalty": -0.20,
    "contradiction_penalty": -0.30,
    "staleness_penalty": -0.18,
    "redundancy_penalty": -0.15,
    "superseded_penalty": -0.20,
}
SEMANTIC_MATCH_THRESHOLD = 0.22


def scope_visible(atom_scope: Mapping[str, Any], request_scope: Mapping[str, Any]) -> bool:
    if not atom_scope:
        return True
    for key, value in atom_scope.items():
        if value == "global":
            continue
        if request_scope.get(key) != value:
            return False
    return True


def maintenance_scope_visible(
    atom_scope: Mapping[str, Any], maintenance_scope: Mapping[str, Any]
) -> bool:
    if not maintenance_scope:
        return True
    # Maintenance scopes are compatible hierarchically in both directions: a
    # broad pass includes narrower evidence, while a run-specific pass still
    # includes tenant-level profiles. Only an explicit conflicting dimension
    # (for example a different run_id or tenant) makes an atom invisible.
    for key in set(atom_scope).intersection(maintenance_scope):
        atom_value = atom_scope.get(key)
        maintenance_value = maintenance_scope.get(key)
        if atom_value == "global" or maintenance_value == "global":
            continue
        if atom_value != maintenance_value:
            return False
    return True


def access_visible(
    access_policy: Mapping[str, Any], requester: str, target_processor: str
) -> bool:
    visibility = access_policy.get("visibility", ["all"])
    return (
        "all" in visibility
        or requester in visibility
        or target_processor in visibility
        or f"processor:{target_processor}" in visibility
    )


def payload_agent_id(payload: Mapping[str, Any]) -> Any:
    return payload.get("agent_id") or payload.get("subject_agent") or payload.get("agent")


def payload_capability_name(payload: Mapping[str, Any]) -> str:
    return str(payload.get("name") or payload.get("capability") or "")


def _structured_ref_list(value: Any) -> list[str]:
    refs: list[str] = []

    def add(ref: Any) -> None:
        text = str(ref or "").strip()
        if text and text not in refs:
            refs.append(text)

    if isinstance(value, str):
        add(value)
    elif isinstance(value, Mapping):
        for key in ("id", "atom_ref", "atom_id", "source_ref", "target_ref", "ref"):
            if value.get(key):
                add(value.get(key))
                break
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            refs.extend(ref for ref in _structured_ref_list(item) if ref not in refs)
    return refs


class Amos:
    """AMOS v1-local service facade."""

    def __init__(
        self,
        db_path: str | Path = "amos.sqlite3",
        *,
        store: Any | None = None,
        maintenance_processors: Sequence[MaintenanceProcessor] | None = None,
        maintenance_processor_paths: Sequence[str] | None = None,
    ):
        self.store = store or SQLiteStore(db_path)
        self.smp = SemanticMaintenanceProcessor()
        self._smp_vector_model_graph_version: int | None = None
        self._memory_policy_lock = threading.Lock()
        self._memory_policy_running = False
        self.maintenance_processors = default_processor_registry(
            self.smp,
            processors=maintenance_processors,
            processor_paths=maintenance_processor_paths,
        )

    def close(self) -> None:
        self.store.close()

    def register_maintenance_processor(
        self, processor: MaintenanceProcessor
    ) -> dict[str, Any]:
        self.maintenance_processors.register(processor)
        return {
            "status": "registered",
            "processor": {
                "processor_id": processor.processor_id,
                "processor_version": processor.processor_version,
            },
            "processors": self.list_maintenance_processors()["processors"],
        }

    def load_maintenance_processor(self, import_path: str) -> dict[str, Any]:
        processor = load_maintenance_processor(import_path)
        return self.register_maintenance_processor(processor)

    def list_maintenance_processors(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "processors": self.maintenance_processors.list(),
        }

    def configure_capacity_budget(
        self,
        *,
        hard_capacity_bytes: int,
        warning_ratio: float = 0.70,
        critical_ratio: float = 0.90,
    ) -> dict[str, Any]:
        if hard_capacity_bytes <= 0:
            raise ValidationError("hard_capacity_bytes must be positive")
        if not 0 < warning_ratio <= critical_ratio <= 1:
            raise ValidationError("capacity ratios must satisfy 0 < warning <= critical <= 1")
        budget = {
            "hard_capacity_bytes": int(hard_capacity_bytes),
            "warning_ratio": float(warning_ratio),
            "critical_ratio": float(critical_ratio),
        }
        self.store.set_meta("capacity_budget", canonical_json(budget))
        return {"status": "configured", "capacity_budget": budget}

    def configure_memory_policy(
        self,
        *,
        enabled: bool | None = None,
        schedule: Mapping[str, Any] | None = None,
        maintenance: Mapping[str, Any] | None = None,
        distillation: Mapping[str, Any] | None = None,
        maintenance_distiller: Mapping[str, Any] | None = None,
        decay: Mapping[str, Any] | None = None,
        storage_cleanup: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy = self.memory_policy()
        if enabled is not None:
            policy["enabled"] = bool(enabled)
        if schedule is not None:
            policy["schedule"] = {**policy["schedule"], **dict(schedule)}
        if maintenance is not None:
            policy["maintenance"] = {**policy["maintenance"], **dict(maintenance)}
        if distillation is not None:
            policy["distillation"] = {
                **policy["distillation"],
                **dict(distillation),
            }
        if maintenance_distiller is not None:
            policy["maintenance_distiller"] = {
                **policy["maintenance_distiller"],
                **dict(maintenance_distiller),
            }
        if decay is not None:
            policy["decay"] = {**policy["decay"], **dict(decay)}
        if storage_cleanup is not None:
            cleanup = dict(policy["storage_cleanup"])
            for key, value in dict(storage_cleanup).items():
                if (
                    key == "sqlite_compaction"
                    and isinstance(value, Mapping)
                    and isinstance(cleanup.get("sqlite_compaction"), Mapping)
                ):
                    cleanup["sqlite_compaction"] = {
                        **dict(cleanup["sqlite_compaction"]),
                        **dict(value),
                    }
                else:
                    cleanup[key] = value
            policy["storage_cleanup"] = cleanup
        policy = self._normalize_memory_policy(policy)
        self.store.set_meta("memory_policy", canonical_json(policy))
        return {
            "status": "configured",
            "policy": policy,
            "memory_policy": self.memory_policy_status(policy=policy),
        }

    def memory_policy(self) -> dict[str, Any]:
        raw = self.store.get_meta("memory_policy")
        if not raw:
            return self._normalize_memory_policy(DEFAULT_MEMORY_POLICY)
        try:
            configured = json.loads(raw)
        except json.JSONDecodeError:
            configured = {}
        return self._normalize_memory_policy(configured)

    def memory_policy_status(
        self, *, policy: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        current_policy = self._normalize_memory_policy(policy or self.memory_policy())
        state = self._memory_policy_state()
        due = self._memory_policy_due(current_policy, state)
        return {
            "policy": current_policy,
            "state": state,
            "due": due,
            "graph_version": self.store.graph_version(),
        }

    def _mark_foreground_activity(self, actor: str | None = None) -> None:
        if actor and str(actor).startswith("svc:"):
            return
        try:
            self.store.set_meta("last_foreground_activity_at", utc_now())
        except Exception:
            pass

    def _search_text_for_atom(self, atom: Mapping[str, Any]) -> str:
        return (
            str(atom.get("id") or "")
            + " "
            + str(atom.get("type") or "")
            + " "
            + self._search_text_for_value(atom.get("payload", {}))
        ).lower()

    def _search_text_for_value(self, value: Any) -> str:
        if value in (None, "", [], {}):
            return ""
        if isinstance(value, Mapping):
            return " ".join(
                part
                for item in value.values()
                if (part := self._search_text_for_value(item))
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(
                part
                for item in value
                if (part := self._search_text_for_value(item))
            )
        return str(value)

    def _search_index_for_atom(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        self._sync_smp_vector_model()
        text = self._search_text_for_atom(atom)
        raw_tokens = {token for token in re.findall(r"[a-z0-9_]+", text) if token}
        tokens = set(raw_tokens)
        for token in raw_tokens:
            tokens.update(part for part in token.split("_") if part)
        tokens = sorted(tokens)
        return {
            "text": text,
            "tokens": tokens,
            "vector": self.smp.encode(text),
            "processor_id": self.smp.processor_id,
            "processor_version": self.smp.processor_version,
            "search_schema": SEARCH_INDEX_SCHEMA,
            "vector_model": self.smp.vector_model_info(),
        }

    def _attach_search_index(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        indexed = dict(atom)
        index_refs = dict(indexed.get("index_refs") or {})
        index_refs[SEARCH_INDEX_REF] = self._search_index_for_atom(indexed)
        indexed["index_refs"] = index_refs
        return indexed

    def _atom_search_index(
        self, atom: Mapping[str, Any], *, allow_stale: bool = False
    ) -> dict[str, Any]:
        index_refs = atom.get("index_refs", {})
        if isinstance(index_refs, Mapping):
            stored = index_refs.get(SEARCH_INDEX_REF)
            if isinstance(stored, Mapping):
                text = stored.get("text")
                tokens = stored.get("tokens")
                vector = stored.get("vector")
                search_schema = stored.get("search_schema")
                if (
                    search_schema == SEARCH_INDEX_SCHEMA
                    and isinstance(text, str)
                    and isinstance(tokens, list)
                    and isinstance(vector, list)
                ):
                    index = {
                        "text": text,
                        "tokens": [str(token) for token in tokens],
                        "vector": [float(value) for value in vector],
                        "vector_model": dict(stored.get("vector_model") or {}),
                    }
                    stale = not self.smp._stored_vector_matches(stored)
                    if stale and not allow_stale:
                        return self._search_index_for_atom(atom)
                    if stale:
                        index["vector_stale"] = True
                    return index
        return self._search_index_for_atom(atom)

    def _prepare_committed_atom(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_atom(atom)
        now = utc_now()
        normalized["id"] = normalized["id"] or stable_id(
            "atom",
            {
                "type": normalized["type"],
                "payload": normalized["payload"],
                "scope": normalized["scope"],
                "evidence_refs": normalized["evidence_refs"],
            },
        )
        normalized["created_at"] = normalized["created_at"] or now
        normalized["observed_at"] = normalized["observed_at"] or now
        normalized["updated_at"] = normalized["updated_at"] or now
        normalized["version"] = 1
        return normalize_atom(self._attach_search_index(normalized), require_id=True)

    def _sync_smp_vector_model(
        self, *, graph_version: int | None = None, force: bool = False
    ) -> dict[str, Any]:
        graph_version = (
            self.store.graph_version() if graph_version is None else int(graph_version)
        )
        if not force and self._smp_vector_model_graph_version == graph_version:
            return self.smp.vector_model_info()
        document_count = self.store.atom_text_document_count()
        document_frequencies = self.store.token_document_frequencies()
        latent_vectors = self.store.list_token_latent_vectors(graph_version=graph_version)
        latent_dimensions = max(
            (len(vector) for vector in latent_vectors.values()),
            default=0,
        )
        self.smp.configure_vector_model(
            document_frequencies=document_frequencies,
            document_count=document_count,
            graph_version=graph_version,
            latent_vectors=latent_vectors,
            latent_dimensions=latent_dimensions,
        )
        self._smp_vector_model_graph_version = graph_version
        return self.smp.vector_model_info()

    def run_memory_policy(
        self,
        *,
        force: bool = False,
        trigger: str = "scheduler",
        scope: Mapping[str, Any] | None = None,
        actor: str = "svc:memory_policy",
    ) -> dict[str, Any]:
        policy = self.memory_policy()
        state = self._memory_policy_state()
        due = self._memory_policy_due(policy, state, force=force)
        if not due["due"]:
            return {
                "status": "skipped",
                "reason": "not_due",
                "trigger": trigger,
                "due": due,
                "graph_version": self.store.graph_version(),
            }
        if not policy["enabled"] and not force:
            return {
                "status": "skipped",
                "reason": "policy_disabled",
                "trigger": trigger,
                "due": due,
                "graph_version": self.store.graph_version(),
            }

        if not self._memory_policy_lock.acquire(blocking=False):
            return {
                "status": "skipped",
                "reason": "memory_policy_already_running",
                "trigger": trigger,
                "graph_version": self.store.graph_version(),
            }

        self._memory_policy_running = True
        started_graph_version = self.store.graph_version()
        scope = dict(scope or {})
        results: dict[str, Any] = {}
        target_refs: list[str] = []
        try:
            maintenance = policy["maintenance"]
            if maintenance["enabled"] and maintenance["run_smp"]:
                results["smp"] = self.run_smp_analysis(
                    scope=scope,
                    max_atoms=maintenance["max_smp_atoms"],
                )
            if maintenance["enabled"] and maintenance["run_steward"]:
                results["steward"] = self.run_steward(scope=scope, actor=actor)
                for action in results["steward"].get("actions", []):
                    target_refs.extend(
                        ref
                        for ref in action.get("atom_refs", [])
                        + [action.get("kept"), action.get("archived")]
                        if ref
                    )

            decay = policy["decay"]
            if decay["enabled"]:
                results["decay"] = self._run_decay_policy(
                    decay=decay,
                    scope=scope,
                    actor=actor,
                )
                target_refs.extend(
                    action["atom_ref"]
                    for action in results["decay"].get("actions", [])
                    if action.get("atom_ref")
                )

            if policy["distillation"]["enabled"]:
                results["distillation"] = self._run_policy_distillation(
                    policy=policy,
                    scope=scope,
                    actor=actor,
                )
                distilled = results["distillation"].get("distilled")
                if distilled:
                    target_refs.append(distilled["atom"]["id"])
                    target_refs.extend(distilled["source_refs"])

            maintenance_distiller = policy["maintenance_distiller"]
            if maintenance_distiller["enabled"]:
                results["maintenance_distiller"] = self.run_maintenance_distiller(
                    scope=scope,
                    actor=actor,
                    domain=maintenance_distiller["domain"],
                    processor_ids=maintenance_distiller["processor_ids"],
                    max_atoms=maintenance_distiller["max_atoms"],
                    max_events=maintenance_distiller["max_events"],
                    max_retrieval_outcomes=maintenance_distiller[
                        "max_retrieval_outcomes"
                    ],
                    auto_commit_low_risk=maintenance_distiller[
                        "auto_commit_low_risk"
                    ],
                    reviewer=maintenance_distiller["reviewer"],
                )
                for committed in results["maintenance_distiller"].get(
                    "committed", []
                ):
                    atom = committed.get("atom")
                    if atom:
                        target_refs.append(atom["id"])
                    target_refs.extend(committed.get("source_refs", []))

            storage_cleanup = policy["storage_cleanup"]
            if storage_cleanup["enabled"] and due.get("storage_cleanup", {}).get("due"):
                results["storage_cleanup"] = self._run_storage_cleanup(
                    cleanup=storage_cleanup,
                    due=due["storage_cleanup"],
                    scope=scope,
                    actor=actor,
                    state=state,
                    force=force,
                )
                target_refs.extend(results["storage_cleanup"].get("deleted_atom_refs", []))

            policy_event_graph_version = self.store.graph_version() + 1
            if maintenance["enabled"] and maintenance["rebuild_indexes"]:
                results["index"] = self._rebuild_derived_indexes(
                    graph_version=policy_event_graph_version
                )
            if maintenance["enabled"] and maintenance["invalidate_packet_cache"]:
                results["packet_cache"] = self._invalidate_packet_cache(
                    graph_version=policy_event_graph_version
                )

            completed_at = utc_now()
            event_payload = {
                "operation": "run_memory_policy",
                "trigger": trigger,
                "force": force,
                "due": due,
                "policy": policy,
                "started_graph_version": started_graph_version,
                "completed_graph_version": policy_event_graph_version,
                "results": self._memory_policy_journal_results(results),
            }
            with self.store.transaction() as conn:
                event = self.store.append_event(
                    conn,
                    event_type="memory_policy_run",
                    actor=actor,
                    payload=event_payload,
                    target_refs=sorted(set(target_refs)),
                )
                self.store._set_meta(
                    conn,
                    "memory_policy_state",
                    canonical_json(
                        {
                            "last_run_at": completed_at,
                            "last_graph_version": event["graph_version"],
                            "last_trigger": trigger,
                            "last_event_id": event["event_id"],
                            "last_due_reasons": due["reasons"],
                            "last_distilled_refs": [
                                results.get("distillation", {})
                                .get("distilled", {})
                                .get("atom", {})
                                .get("id")
                            ]
                            if results.get("distillation", {}).get("distilled")
                            else [],
                            "last_maintenance_distiller_refs": [
                                committed["atom"]["id"]
                                for committed in results.get(
                                    "maintenance_distiller", {}
                                ).get("committed", [])
                                if committed.get("atom")
                            ],
                            "last_storage_cleanup_at": self.store.get_meta(
                                "last_storage_cleanup_at"
                            ),
                            "last_vacuum_at": self.store.get_meta("last_vacuum_at"),
                            "last_foreground_activity_at": self.store.get_meta(
                                "last_foreground_activity_at"
                            ),
                        }
                    ),
                )
                if maintenance["enabled"] and maintenance["invalidate_packet_cache"]:
                    self.store.clear_packet_cache(conn)
            return {
                "status": "completed",
                "trigger": trigger,
                "due": due,
                "policy": policy,
                "results": results,
                "event": event,
                "graph_version": self.store.graph_version(),
            }
        finally:
            self._memory_policy_running = False
            self._memory_policy_lock.release()

    def capture_event(
        self,
        *,
        source_type: str,
        source_ref: str,
        payload: Any,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
        access_policy: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        request_payload = {
            "operation": "capture_event",
            "source_type": source_type,
            "source_ref": source_ref,
            "payload": payload,
            "scope": dict(scope or {}),
            "access_policy": dict(access_policy or {}),
        }
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, request_payload)
            if prior is not None:
                return prior
            evidence = normalize_evidence(
                {
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "payload": payload,
                    "scope": scope or {},
                    "access_policy": access_policy,
                }
            )
            op_payload = {"operation": "capture_event", "evidence": evidence}
            event = self.store.append_event(
                conn,
                event_type="evidence_captured",
                actor=actor,
                payload=op_payload,
                evidence_refs=[evidence["evidence_id"]],
                idempotency_key=idempotency_key,
            )
            self.store.insert_evidence(conn, evidence, event["event_id"])
            response = {"status": "captured", "evidence": evidence, "event": event}
            self._record_idempotency(
                conn, actor, idempotency_key, request_payload, event, response
            )
            return response

    def commit_atom(
        self,
        atom: Mapping[str, Any],
        *,
        actor: str = "system",
        idempotency_key: str | None = None,
        authorization_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        request_payload = {"operation": "commit_atom", "atom": dict(atom)}
        normalized = self._prepare_committed_atom(atom)
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, request_payload)
            if prior is not None:
                return prior
            projected_edges = self._intrinsic_edges_for_atom(normalized)
            op_payload = {
                "operation": "commit_atom",
                "atom": normalized,
                "projected_edges": projected_edges,
            }
            content_digest = self._memory_identity_digest(normalized)
            tombstone = self.store.get_tombstone(
                normalized["id"], content_digest=content_digest
            )
            if tombstone and tombstone["recreation_policy"] != "allow_recreate":
                raise ValidationError(
                    f"memory is tombstoned: {normalized['id']} / {content_digest}"
                )
            if self.store.get_atom(normalized["id"]) is not None:
                raise ValidationError(f"atom already exists: {normalized['id']}")
            event = self.store.append_event(
                conn,
                event_type="atom_committed",
                actor=actor,
                payload=op_payload,
                target_refs=[normalized["id"]],
                evidence_refs=normalized["evidence_refs"],
                idempotency_key=idempotency_key,
                authorization_context=authorization_context,
            )
            self.store.insert_atom(conn, normalized)
            for edge in projected_edges:
                self.store.insert_edge(conn, edge)
            self.store.clear_packet_cache(conn)
            response = {
                "status": "committed",
                "atom": normalized,
                "edges": projected_edges,
                "event": event,
            }
            self._record_idempotency(
                conn, actor, idempotency_key, request_payload, event, response
            )
            return response

    def propose_memory_atoms(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        proposals = []
        for candidate in candidates:
            atom = dict(candidate)
            if scope is not None:
                atom.setdefault("scope", dict(scope))
            atom["lifecycle_state"] = "proposed"
            atom.setdefault("confidence", {"level": "low-medium", "score": 0.35})
            proposals.append(self.commit_atom(atom, actor=actor))
        return {
            "status": "proposed",
            "proposals": proposals,
            "graph_version": self.store.graph_version(),
        }

    def commit_memory_atoms(
        self,
        atoms: Sequence[Mapping[str, Any]],
        *,
        actor: str = "system",
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        prepared = [self._prepare_committed_atom(atom) for atom in atoms]
        seen_ids: set[str] = set()
        for atom in prepared:
            if atom["id"] in seen_ids:
                raise ValidationError(f"duplicate atom in batch: {atom['id']}")
            seen_ids.add(atom["id"])
        committed = []
        with self.store.transaction() as conn:
            for normalized in prepared:
                projected_edges = self._intrinsic_edges_for_atom(normalized)
                op_payload = {
                    "operation": "commit_atom",
                    "atom": normalized,
                    "projected_edges": projected_edges,
                }
                content_digest = self._memory_identity_digest(normalized)
                tombstone = self.store.get_tombstone(
                    normalized["id"], content_digest=content_digest
                )
                if tombstone and tombstone["recreation_policy"] != "allow_recreate":
                    raise ValidationError(
                        f"memory is tombstoned: {normalized['id']} / {content_digest}"
                    )
                if self.store.get_atom(normalized["id"]) is not None:
                    raise ValidationError(f"atom already exists: {normalized['id']}")
                event = self.store.append_event(
                    conn,
                    event_type="atom_committed",
                    actor=actor,
                    payload=op_payload,
                    target_refs=[normalized["id"]],
                    evidence_refs=normalized["evidence_refs"],
                )
                self.store.insert_atom(conn, normalized)
                for edge in projected_edges:
                    self.store.insert_edge(conn, edge)
                committed.append(
                    {
                        "status": "committed",
                        "atom": normalized,
                        "edges": projected_edges,
                        "event": event,
                    }
                )
            if committed:
                self.store.clear_packet_cache(conn)
        return {
            "status": "committed",
            "committed": committed,
            "graph_version": self.store.graph_version(),
        }

    def update_atom(
        self,
        atom_id: str,
        *,
        payload_patch: Mapping[str, Any] | None = None,
        set_fields: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        op_payload = {
            "operation": "update_atom",
            "atom_id": atom_id,
            "payload_patch": dict(payload_patch or {}),
            "set_fields": dict(set_fields or {}),
            "expected_version": expected_version,
        }
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, op_payload)
            if prior is not None:
                return prior
            current = self.store.get_atom(atom_id)
            if current is None or current.get("deleted"):
                raise ValidationError(f"unknown atom: {atom_id}")
            self._assert_mutation_allowed(
                current, actor=actor, authorization_context=authorization_context
            )
            if expected_version is not None and current["version"] != expected_version:
                raise CASConflict(
                    f"expected {atom_id} version {expected_version}, "
                    f"found {current['version']}"
                )
            updated = dict(current)
            if payload_patch:
                updated_payload = dict(updated["payload"])
                updated_payload.update(dict(payload_patch))
                updated["payload"] = updated_payload
            for key, value in dict(set_fields or {}).items():
                if key in {"id", "type", "schema_version", "created_at", "version"}:
                    raise ValidationError(f"cannot update immutable atom field: {key}")
                if key == "payload":
                    updated["payload"] = dict(value)
                else:
                    updated[key] = value
            updated["revision_history"] = list(updated["revision_history"])
            updated["revision_history"].append(
                {
                    "version": current["version"],
                    "digest": digest(self._atom_projection(current)),
                    "changed_at": utc_now(),
                    "actor": actor,
                }
            )
            updated["version"] = int(current["version"]) + 1
            updated["updated_at"] = utc_now()
            updated = normalize_atom(
                self._attach_search_index(updated), require_id=True
            )
            projected_edges = []
            if (
                current.get("lifecycle_state") == "active"
                and updated.get("lifecycle_state") != "active"
            ):
                projected_edges = self.store.mark_edges_deleted_for_ref(conn, atom_id)
            event = self.store.append_event(
                conn,
                event_type="atom_updated",
                actor=actor,
                payload={
                    "operation": "update_atom",
                    "before": current,
                    "after": updated,
                    "projected_edges": projected_edges,
                },
                target_refs=[atom_id],
                evidence_refs=updated["evidence_refs"],
                idempotency_key=idempotency_key,
                expected_versions={atom_id: expected_version}
                if expected_version is not None
                else {},
                authorization_context=authorization_context,
            )
            self.store.replace_atom(conn, updated)
            self.store.clear_packet_cache(conn)
            response = {
                "status": "updated",
                "atom": updated,
                "event": event,
                "projected_edges": projected_edges,
            }
            self._record_idempotency(conn, actor, idempotency_key, op_payload, event, response)
            return response

    def archive_atom(
        self,
        atom_id: str,
        *,
        reason: str = "archived",
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.update_atom(
            atom_id,
            set_fields={
                "lifecycle_state": "archived",
                "health_status": "stale",
                "decay_policy": {"archive_reason": reason},
            },
            expected_version=expected_version,
            actor=actor,
            authorization_context=authorization_context,
        )

    def delete_atom(
        self,
        atom_id: str,
        *,
        reason: str,
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
        recreation_policy: str = "block_recreate",
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        op_payload = {
            "operation": "delete_atom",
            "atom_id": atom_id,
            "reason": reason,
            "expected_version": expected_version,
            "recreation_policy": recreation_policy,
        }
        with self.store.transaction() as conn:
            current = self.store.get_atom(atom_id)
            if current is None:
                raise ValidationError(f"unknown atom: {atom_id}")
            self._assert_mutation_allowed(
                current, actor=actor, authorization_context=authorization_context
            )
            if expected_version is not None and current["version"] != expected_version:
                raise CASConflict(
                    f"expected {atom_id} version {expected_version}, "
                    f"found {current['version']}"
                )
            updated = dict(current)
            updated["lifecycle_state"] = "deleted"
            updated["health_status"] = "deleted"
            updated["deleted"] = 1
            updated["version"] = int(current["version"]) + 1
            updated["updated_at"] = utc_now()
            updated["revision_history"] = list(updated["revision_history"])
            updated["revision_history"].append(
                {
                    "version": current["version"],
                    "digest": digest(self._atom_projection(current)),
                    "changed_at": utc_now(),
                    "actor": actor,
                    "reason": reason,
                }
            )
            updated = normalize_atom(
                self._attach_search_index(updated), require_id=True
            )
            updated["deleted"] = 1
            tombstone = self.store.insert_tombstone(
                conn,
                target_ref=atom_id,
                content_digest=self._memory_identity_digest(current),
                recreation_policy=recreation_policy,
                reason=reason,
            )
            deleted_edges = self.store.mark_edges_deleted_for_ref(conn, atom_id)
            event = self.store.append_event(
                conn,
                event_type="atom_deleted",
                actor=actor,
                payload={
                    **op_payload,
                    "before": current,
                    "tombstone": tombstone,
                    "projected_edges": deleted_edges,
                },
                target_refs=[atom_id],
                evidence_refs=current["evidence_refs"],
                expected_versions={atom_id: expected_version}
                if expected_version is not None
                else {},
                authorization_context=authorization_context,
            )
            self.store.replace_atom(conn, updated)
            self.store.clear_packet_cache(conn)
            return {
                "status": "deleted",
                "atom": updated,
                "tombstone": tombstone,
                "event": event,
            }

    def request_deletion(
        self,
        *,
        target_ref: str,
        reason: str,
        requested_by: str = "system",
        expected_version: int | None = None,
        authorization_context: Mapping[str, Any] | None = None,
        recreation_policy: str = "block_recreate",
    ) -> dict[str, Any]:
        self._mark_foreground_activity(requested_by)
        result = self.delete_atom(
            target_ref,
            reason=reason,
            expected_version=expected_version,
            actor=requested_by,
            authorization_context=authorization_context,
            recreation_policy=recreation_policy,
        )
        result["residual_retention"] = {
            "hot_database_payload": "suppressed",
            "packet_cache": "purged",
            "offline_backup_residual_window_days": 30,
            "evidence_archive": "retained_or_suppressed_by_policy",
        }
        return result

    def merge_atoms(
        self,
        *,
        source_refs: Sequence[str],
        merged_payload: Mapping[str, Any],
        merged_type: str = "semantic",
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        if not approved_by:
            return {
                "status": "review_required",
                "action": "merge_atoms",
                "source_refs": list(source_refs),
                "risk": "high",
                "reason": "active atom merge requires explicit review",
                "mutated": False,
            }
        with self.store.transaction() as conn:
            sources = []
            for ref in source_refs:
                atom = self.store.get_atom(ref)
                if atom is None or atom.get("deleted"):
                    raise ValidationError(f"unknown source atom: {ref}")
                self._assert_mutation_allowed(
                    atom,
                    actor=actor,
                    authorization_context={
                        "roles": ["owner"],
                        "trust_level": 10,
                        "capabilities": ["memory.write"],
                        "approved_by": approved_by,
                    },
                )
                sources.append(atom)
            now = utc_now()
            merged = normalize_atom(
                {
                    "type": merged_type,
                    "payload": dict(merged_payload),
                    "scope": dict(scope or {}),
                    "supersedes": list(source_refs),
                    "salience": max([float(atom["salience"]) for atom in sources] + [0.5]),
                    "utility": max([float(atom["utility"]) for atom in sources] + [0.5]),
                    "confidence": {"level": "medium-high", "score": 0.75},
                }
            )
            merged["id"] = stable_id(
                "atom",
                {
                    "operation": "merge_atoms",
                    "source_refs": list(source_refs),
                    "payload": merged_payload,
                    "scope": dict(scope or {}),
                },
            )
            merged["created_at"] = now
            merged["observed_at"] = now
            merged["updated_at"] = now
            merged = normalize_atom(self._attach_search_index(merged), require_id=True)
            self.store.insert_atom(conn, merged)
            projected_atoms = [merged]
            projected_edges = []
            for source in sources:
                edge = self._edge(
                    merged["id"], source["id"], "rel:derived_from", dict(scope or {})
                )
                self.store.insert_edge(conn, edge)
                projected_edges.append(edge)
                archived = dict(source)
                archived["lifecycle_state"] = "archived"
                archived["health_status"] = "merged"
                archived["version"] = int(source["version"]) + 1
                archived["updated_at"] = utc_now()
                archived["decay_policy"] = {
                    **dict(archived.get("decay_policy") or {}),
                    "merged_into": merged["id"],
                }
                archived = normalize_atom(
                    self._attach_search_index(archived), require_id=True
                )
                self.store.replace_atom(conn, archived)
                projected_atoms.append(archived)
            event = self.store.append_event(
                conn,
                event_type="atom_merged",
                actor=actor,
                payload={
                    "operation": "merge_atoms",
                    "merged_atom": merged,
                    "source_refs": list(source_refs),
                    "projected_atoms": projected_atoms,
                    "projected_edges": projected_edges,
                },
                target_refs=[merged["id"], *source_refs],
                authorization_context={"approved_by": approved_by},
            )
            self.store.clear_packet_cache(conn)
            return {
                "status": "merged",
                "atom": merged,
                "source_refs": list(source_refs),
                "edges": projected_edges,
                "event": event,
            }

    def retrieve_packet(
        self,
        *,
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        retrieval_mode: str = "general",
        max_items: int | None = None,
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        include_conflicts: bool | None = None,
        include_archived: bool = False,
        include_low_health: bool = False,
        include_superseded: bool = False,
        type_filter: Sequence[str] | None = None,
        attention_context: Mapping[str, Any] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(requester)
        if run_policy:
            self.run_memory_policy(trigger="retrieve_packet", scope=scope or {})
        profile = DEFAULT_PACKET_PROFILES.get(
            retrieval_mode, DEFAULT_PACKET_PROFILES.get(target_processor, {})
        )
        if max_items is None:
            max_items = int(profile.get("max_items", 8))
        if token_or_byte_budget is None and "tokens" in profile:
            token_or_byte_budget = {"tokens": int(profile["tokens"])}
        if include_conflicts is None:
            include_conflicts = bool(profile.get("include_conflicts", False))
        pressure_mode = self._capacity_pressure_mode()
        pressure_degraded = pressure_mode in {"orange", "red"}
        original_max_items = max_items
        if pressure_mode == "orange":
            max_items = max(1, max_items // 2)
        elif pressure_mode == "red":
            max_items = max(1, min(max_items, 3))
        attention_policy = self._attention_policy(attention_context)
        request = {
            "cues": list(cues or []),
            "scope": dict(scope or {}),
            "requester": requester,
            "target_processor": target_processor,
            "retrieval_mode": retrieval_mode,
            "max_items": max_items,
            "token_or_byte_budget": token_or_byte_budget,
            "include_conflicts": include_conflicts,
            "include_archived": include_archived,
            "include_low_health": include_low_health,
            "include_superseded": include_superseded,
            "type_filter": list(type_filter or []),
            "attention_context": attention_policy["context"],
            "pressure_mode": pressure_mode,
            "run_policy": bool(run_policy),
        }
        graph_version = self.store.graph_version()
        cached = self.store.get_cached_packet(
            request=request, graph_version=graph_version
        )
        if cached is not None:
            return cached
        self._sync_smp_vector_model(graph_version=graph_version)

        candidates: list[tuple[float, dict[str, Any]]] = []
        omissions: list[dict[str, Any]] = []
        allowed_types = set(type_filter or [])
        lifecycle_states = ["active", "proposed"]
        if include_archived:
            lifecycle_states.append("archived")
        cue_text = " ".join(request["cues"]).lower()
        cue_tokens = {token for token in re.findall(r"[a-z0-9_]+", cue_text) if token}
        candidate_atom_ids = self._indexed_retrieval_candidates(
            cue_tokens=cue_tokens,
            attention_policy=attention_policy,
        )
        atoms = self.store.list_atoms_filtered(
            types=sorted(allowed_types) if allowed_types else None,
            lifecycle_states=lifecycle_states,
            atom_ids=candidate_atom_ids,
        )
        atom_refs = [str(atom["id"]) for atom in atoms]
        edge_degrees = self._hot_graph_edge_degree_counts(atoms)
        cue_vector = self.smp.encode(cue_text) if cue_text else []
        superseded_refs = self._active_superseded_refs(atom_refs)
        edge_activation_scores = self._graph_activation_scores(
            atoms,
            cues=request["cues"],
            request_scope=request["scope"],
            requester=requester,
            target_processor=target_processor,
            include_conflicts=bool(include_conflicts),
            include_low_health=bool(include_low_health),
            cue_text=cue_text,
            cue_tokens=cue_tokens,
            attention_policy=attention_policy,
            superseded_refs=superseded_refs if not include_superseded else None,
        )
        for atom in atoms:
            atom_ref = atom["id"]
            if atom.get("deleted"):
                omissions.append({"atom_ref": atom_ref, "reason": "deleted"})
                continue
            if not scope_visible(atom["scope"], request["scope"]):
                omissions.append({"atom_ref": atom_ref, "reason": "scope_hidden"})
                continue
            if not access_visible(atom["access_policy"], requester, target_processor):
                omissions.append({"atom_ref": atom_ref, "reason": "access_hidden"})
                continue
            if atom["health_status"] == "contradicted" and not include_conflicts:
                omissions.append({"atom_ref": atom_ref, "reason": "contradicted"})
                continue
            if atom["health_status"] in LOW_HEALTH_STATES and not include_low_health:
                omissions.append(
                    {"atom_ref": atom_ref, "reason": f"health:{atom['health_status']}"}
                )
                continue
            if atom_ref in superseded_refs and not include_superseded:
                omissions.append(
                    {
                        "atom_ref": atom_ref,
                        "reason": "superseded",
                        "superseded_by": superseded_refs[atom_ref][:5],
                    }
                )
                continue
            score, matched, components = self._rank_atom(
                atom,
                request["cues"],
                request_scope=request["scope"],
                retrieval_mode=retrieval_mode,
                cue_text=cue_text,
                cue_tokens=cue_tokens,
                cue_vector=cue_vector,
                edge_degrees=edge_degrees,
                edge_activation_scores=edge_activation_scores,
                attention_policy=attention_policy,
                superseded_refs=superseded_refs,
            )
            if request["cues"] and not matched:
                omissions.append({"atom_ref": atom_ref, "reason": "low_relevance"})
                continue
            atom = {**atom, "_score_components": components}
            candidates.append((score, atom))

        candidates.sort(key=lambda item: item[0], reverse=True)
        byte_budget = self._byte_budget(token_or_byte_budget)
        used_bytes = 0
        items = []
        for score, atom in candidates:
            if len(items) >= max_items:
                omissions.append(
                    {
                        "atom_ref": atom["id"],
                        "reason": "pressure_degraded"
                        if pressure_degraded and len(items) >= max_items
                        else "budget_exhausted",
                    }
                )
                continue
            item, evidence_omissions = self._packet_item(
                atom, score, requester=requester, target_processor=target_processor
            )
            omissions.extend(evidence_omissions)
            rendered_size = len(canonical_json(item).encode("utf-8"))
            if used_bytes + rendered_size > byte_budget:
                omissions.append({"atom_ref": atom["id"], "reason": "budget_exhausted"})
                continue
            used_bytes += rendered_size
            items.append(item)
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank

        conflicts = []
        if include_conflicts and items:
            selected = {item["atom_ref"] for item in items}
            for edge in self.store.list_edges_for_refs(sorted(selected)):
                if edge["relation"] not in CONFLICT_RELATIONS:
                    continue
                if edge["source_ref"] in selected or edge["target_ref"] in selected:
                    conflicts.append(edge)

        packet = {
            "packet_id": stable_id(
                "pkt",
                {"request": request, "graph_version": graph_version, "items": items},
            ),
            "schema_version": SCHEMA_VERSION,
            "request": request,
            "graph_version": graph_version,
            "generated_at": utc_now(),
            "target_processor": target_processor,
            "retrieval_mode": retrieval_mode,
            "scope": dict(scope or {}),
            "pressure_mode": pressure_mode,
            "items": items,
            "omissions": omissions,
            "conflicts": conflicts,
            "degradation": {
                "mode": "smp-deterministic-local",
                "pressure_mode": pressure_mode,
                "reduced_recall_depth": pressure_degraded
                and max_items < original_max_items,
                "omitted_evidence_detail": any(
                    omission["reason"] == "evidence_access_denied"
                    for omission in omissions
                ),
                "index_freshness": {
                    "semantic_index": "inline_rebuildable",
                    "graph_version": graph_version,
                },
                "reason_codes": sorted({omission["reason"] for omission in omissions}),
                "vector_index_available": False,
                "byte_budget": byte_budget,
                "used_bytes": used_bytes,
            },
            "attention_trace": self._attention_trace(
                attention_policy=attention_policy,
                items=items,
                candidates=candidates,
                omissions=omissions,
            ),
            "provenance": {
                "store": getattr(self.store, "backend_name", "unknown"),
                "journal_head": self.store.last_event_hash(),
                "ranker_profile_id": "amos.v1.default",
                "smp_processor_id": self.smp.processor_id,
            },
            "cache_policy": {"cacheable": True, "keyed_by_graph_version": True},
        }
        with self.store.transaction() as conn:
            self.store.cache_packet(
                conn,
                packet_id=packet["packet_id"],
                request=request,
                response=packet,
                graph_version=graph_version,
            )
        return packet

    def record_retrieval_outcome(
        self,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._mark_foreground_activity(str(request.get("requester") or "system"))
        with self.store.transaction() as conn:
            record = self.store.insert_retrieval_outcome(
                conn, packet_id=packet_id, request=request, outcome=outcome
            )
            if record.get("status") != "recorded":
                return record
            feedback = self._apply_retrieval_outcome_feedback(
                conn,
                packet_id=packet_id,
                request=request,
                outcome=outcome,
            )
            if feedback["updated_atoms"]:
                event = self.store.append_event(
                    conn,
                    event_type="retrieval_outcome_recorded",
                    actor=str(request.get("requester") or "system"),
                    payload={
                        "operation": "record_retrieval_outcome",
                        "packet_id": packet_id,
                        "outcome_id": record["outcome_id"],
                        "feedback": feedback,
                        "projected_atoms": feedback["projected_atoms"],
                    },
                    target_refs=feedback["updated_atom_refs"],
                )
                self.store.clear_packet_cache(conn)
                record["event"] = event
            record["feedback"] = feedback
            return record

    def _apply_retrieval_outcome_feedback(
        self,
        conn: Any,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        del packet_id, request
        positive_refs, correction_refs = self._retrieval_outcome_atom_refs(outcome)
        label = str(outcome.get("label") or outcome.get("status") or "").lower()
        negative_label = label in {
            "bad",
            "wrong",
            "unused",
            "unhelpful",
            "misleading",
            "corrected",
            "correction",
            "failed",
            "failure",
        }
        now = utc_now()
        updated_atoms: list[dict[str, Any]] = []
        projected_atoms: list[dict[str, Any]] = []
        for atom_ref in sorted(positive_refs.union(correction_refs)):
            atom = self.store.get_atom(atom_ref)
            if atom is None or atom.get("deleted"):
                continue
            changed = dict(atom)
            telemetry = dict(changed.get("decay_policy") or {}).get(
                "retrieval_telemetry", {}
            )
            telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
            used_count = int(telemetry.get("used_count", 0) or 0)
            correction_count = int(telemetry.get("correction_count", 0) or 0)
            if atom_ref in positive_refs:
                used_count += 1
            if atom_ref in correction_refs or negative_label:
                correction_count += 1
            delta = 0.0
            if atom_ref in positive_refs and not negative_label:
                delta += 0.03
            if atom_ref in correction_refs or negative_label:
                delta -= 0.06
            changed["utility"] = max(0.0, min(1.0, float(changed["utility"]) + delta))
            if atom_ref in positive_refs and not negative_label:
                changed["salience"] = max(
                    0.0, min(1.0, float(changed["salience"]) + 0.02)
                )
            if changed["utility"] < 0.25 and changed["health_status"] == "healthy":
                changed["health_status"] = "low_utility"
            telemetry.update(
                {
                    "used_count": used_count,
                    "correction_count": correction_count,
                    "last_outcome_label": label or None,
                    "last_outcome_at": now,
                }
            )
            changed["decay_policy"] = {
                **dict(changed.get("decay_policy") or {}),
                "retrieval_telemetry": telemetry,
            }
            changed["last_accessed"] = now
            changed["updated_at"] = now
            changed["version"] = int(changed["version"]) + 1
            changed = normalize_atom(self._attach_search_index(changed), require_id=True)
            self.store.replace_atom(conn, changed)
            projected_atoms.append(changed)
            updated_atoms.append(
                {
                    "atom_ref": atom_ref,
                    "utility": changed["utility"],
                    "salience": changed["salience"],
                    "health_status": changed["health_status"],
                    "used_count": used_count,
                    "correction_count": correction_count,
                }
            )
        return {
            "updated_atom_refs": [item["atom_ref"] for item in updated_atoms],
            "updated_atoms": updated_atoms,
            "projected_atoms": projected_atoms,
            "positive_refs": sorted(positive_refs),
            "correction_refs": sorted(correction_refs),
        }

    def _retrieval_outcome_atom_refs(
        self, outcome: Mapping[str, Any]
    ) -> tuple[set[str], set[str]]:
        positive: set[str] = set()
        corrections: set[str] = set()

        def add_refs(target: set[str], value: Any) -> None:
            if isinstance(value, str):
                if value:
                    target.add(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    add_refs(target, item)

        for key in (
            "used_item_refs",
            "used_atom_refs",
            "cited_atom_refs",
            "selected_item_refs",
            "helpful_atom_refs",
        ):
            add_refs(positive, outcome.get(key))
        add_refs(positive, outcome.get("cited_atom_ref"))
        add_refs(positive, outcome.get("used_atom_ref"))
        for key in (
            "correction_refs",
            "corrected_atom_refs",
            "misleading_atom_refs",
            "unhelpful_atom_refs",
        ):
            add_refs(corrections, outcome.get(key))
        add_refs(corrections, outcome.get("corrected_atom_ref"))
        return positive, corrections

    def distill_memories(
        self,
        *,
        target_refs: Sequence[str],
        summary: str | Mapping[str, Any],
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        idempotency_key: str | None = None,
        distillation_type: str = "summary",
        archive_sources: bool = False,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        if archive_sources and not approved_by:
            return {
                "status": "review_required",
                "action": "distill_memories",
                "target_refs": list(target_refs),
                "risk": "high",
                "reason": "archiving source memories requires explicit approval",
                "mutated": False,
            }
        request_payload = {
            "operation": "distill_memories",
            "target_refs": list(target_refs),
            "summary": summary,
            "scope": dict(scope or {}),
            "distillation_type": distillation_type,
            "archive_sources": archive_sources,
            "approved_by": approved_by,
        }
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, request_payload)
            if prior is not None:
                return prior
            source_atoms = []
            for ref in target_refs:
                atom = self.store.get_atom(ref)
                if atom is None or atom.get("deleted"):
                    raise ValidationError(f"unknown source atom: {ref}")
                source_atoms.append(atom)
            now = utc_now()
            source_digests = [digest(self._atom_projection(atom)) for atom in source_atoms]
            distilled = normalize_atom(
                {
                    "type": "semantic",
                    "payload": {
                        "distillation_type": distillation_type,
                        "summary": summary,
                        "source_refs": list(target_refs),
                        "source_digests": source_digests,
                        "created_by": actor,
                    },
                    "scope": dict(scope or {}),
                    "layer": "consolidated_long_term",
                    "retention_class": "distilled",
                    "supersedes": list(target_refs) if archive_sources else [],
                    "salience": 0.8,
                    "utility": 0.85,
                    "confidence": {"level": "medium-high", "score": 0.75},
                }
            )
            distilled["id"] = stable_id(
                "atom",
                {
                    "type": "semantic",
                    "summary": summary,
                    "target_refs": list(target_refs),
                    "scope": dict(scope or {}),
                    "distillation_type": distillation_type,
                },
            )
            if self.store.get_atom(distilled["id"]) is not None:
                raise ValidationError(f"distilled atom already exists: {distilled['id']}")
            distilled["created_at"] = now
            distilled["observed_at"] = now
            distilled["updated_at"] = now
            distilled = normalize_atom(
                self._attach_search_index(distilled), require_id=True
            )
            edges = [
                self._edge(
                    distilled["id"],
                    source["id"],
                    "rel:derived_from",
                    dict(scope or {}),
                )
                for source in source_atoms
            ]
            event = self.store.append_event(
                conn,
                event_type="memories_distilled",
                actor=actor,
                payload={
                    "operation": "distill_memories",
                    "atom": distilled,
                    "projected_edges": edges,
                },
                target_refs=[distilled["id"], *target_refs],
                idempotency_key=idempotency_key,
                authorization_context={"approved_by": approved_by}
                if approved_by
                else {},
            )
            self.store.insert_atom(conn, distilled)
            for edge, source in zip(edges, source_atoms):
                self.store.insert_edge(conn, edge)
                if archive_sources:
                    changed = dict(source)
                    changed["lifecycle_state"] = "archived"
                    changed["health_status"] = "stale"
                    changed["version"] = int(source["version"]) + 1
                    changed["updated_at"] = utc_now()
                    changed["decay_policy"] = {
                        **dict(changed.get("decay_policy") or {}),
                        "archived_by_distillation": distilled["id"],
                    }
                    changed = normalize_atom(
                        self._attach_search_index(changed), require_id=True
                    )
                    self.store.replace_atom(conn, changed)
            self.store.clear_packet_cache(conn)
            response = {
                "status": "distilled",
                "atom": distilled,
                "source_refs": list(target_refs),
                "edges": edges,
                "archived_sources": archive_sources,
                "event": event,
            }
            self._record_idempotency(
                conn, actor, idempotency_key, request_payload, event, response
            )
            return response

    def record_runtime_state(
        self,
        *,
        agent_id: str,
        capabilities: Mapping[str, Any] | None = None,
        denied_capabilities: Sequence[str] | None = None,
        constraints: Sequence[str] | None = None,
        load: Mapping[str, Any] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "runtime_state",
                "payload": {
                    "agent_id": agent_id,
                    "capabilities": dict(capabilities or {}),
                    "denied_capabilities": list(denied_capabilities or []),
                    "constraints": list(constraints or []),
                    "load": dict(load or {}),
                },
                "scope": dict(scope or {}),
                "salience": 0.7,
                "utility": 0.8,
            },
            actor=actor,
        )

    def record_self_assessment(
        self,
        *,
        agent_id: str,
        claim: str,
        calibration: Mapping[str, Any],
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "self_assessment",
                "payload": {
                    "agent_id": agent_id,
                    "claim": claim,
                    "calibration": dict(calibration),
                },
                "scope": dict(scope or {}),
                "salience": 0.65,
                "utility": 0.75,
            },
            actor=actor,
        )

    def generate_self_narrative(
        self,
        *,
        agent_id: str,
        narrative: str,
        source_refs: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "self_narrative",
                "payload": {
                    "agent_id": agent_id,
                    "narrative": narrative,
                    "source_refs": list(source_refs or []),
                    "generated_from_graph_version": self.store.graph_version(),
                    "artifact": True,
                },
                "scope": dict(scope or {}),
                "salience": 0.55,
                "utility": 0.6,
                "confidence": {"level": "medium", "score": 0.5},
            },
            actor=actor,
        )

    def record_agentic_trace(
        self,
        *,
        agent_id: str,
        task: str,
        action: str,
        outcome: str,
        lesson: str | None = None,
        external_constraints: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "agentic_trace",
                "payload": {
                    "agent_id": agent_id,
                    "task": task,
                    "action": action,
                    "outcome": outcome,
                    "lesson": lesson,
                    "external_constraints": list(external_constraints or []),
                },
                "scope": dict(scope or {}),
                "salience": 0.8,
                "utility": 0.8,
            },
            actor=actor,
        )

    def record_action_outcome(
        self,
        *,
        agent_id: str,
        action_ref: str,
        status: str,
        evidence_refs: Sequence[str] | None = None,
        correction: str | None = None,
        limitation: str | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "action_outcome",
                "payload": {
                    "agent_id": agent_id,
                    "action_ref": action_ref,
                    "status": status,
                    "correction": correction,
                    "limitation": limitation,
                },
                "evidence_refs": list(evidence_refs or []),
                "scope": dict(scope or {}),
                "salience": 0.75,
                "utility": 0.8,
            },
            actor=actor,
        )

    def retrieve_self_awareness(
        self,
        *,
        agent_id: str,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "self-model",
    ) -> dict[str, Any]:
        by_type: dict[str, list[dict[str, Any]]] = {
            "capability": [],
            "commitment": [],
            "limitation": [],
            "runtime_state": [],
            "self_assessment": [],
            "self_model": [],
        }
        request_scope = dict(scope or {})
        omissions: list[dict[str, Any]] = []
        pressure_mode = self._capacity_pressure_mode()
        target_types = set(by_type)
        for atom in self.store.list_atoms():
            atom_ref = atom["id"]
            if atom.get("deleted"):
                omissions.append({"atom_ref": atom_ref, "reason": "deleted"})
                continue
            if atom["type"] not in target_types:
                continue
            if not scope_visible(atom["scope"], request_scope):
                omissions.append({"atom_ref": atom_ref, "reason": "scope_hidden"})
                continue
            if not access_visible(atom["access_policy"], requester, target_processor):
                omissions.append({"atom_ref": atom_ref, "reason": "access_hidden"})
                continue
            if atom["lifecycle_state"] == "archived":
                omissions.append({"atom_ref": atom_ref, "reason": "archived"})
                continue
            if atom["lifecycle_state"] not in {"active", "proposed"}:
                omissions.append(
                    {"atom_ref": atom_ref, "reason": f"lifecycle:{atom['lifecycle_state']}"}
                )
                continue
            payload = atom["payload"]
            if payload_agent_id(payload) not in {None, agent_id}:
                omissions.append({"atom_ref": atom_ref, "reason": "different_agent"})
                continue
            score, _matched, components = self._rank_atom(
                atom,
                [],
                request_scope=request_scope,
                retrieval_mode="self_awareness",
            )
            item, evidence_omissions = self._packet_item(
                {**atom, "_score_components": components},
                score,
                requester=requester,
                target_processor=target_processor,
            )
            omissions.extend(evidence_omissions)
            by_type[atom["type"]].append(item)

        for items in by_type.values():
            items.sort(
                key=lambda item: (
                    item.get("updated_at") or "",
                    item.get("score") or 0.0,
                ),
                reverse=True,
            )
        latest_runtime = None
        for item in by_type["runtime_state"]:
            if latest_runtime is None or item["updated_at"] > latest_runtime["updated_at"]:
                latest_runtime = item

        denied = set()
        capability_status: Mapping[str, Any] = {}
        if latest_runtime:
            runtime_payload = latest_runtime["payload"]
            denied = set(runtime_payload.get("denied_capabilities", []))
            capability_status = runtime_payload.get("capabilities", {})

        visible_capabilities = []
        for item in by_type["capability"]:
            name = payload_capability_name(item["payload"])
            if self._capability_unavailable(name, capability_status, denied):
                omissions.append(
                    {
                        "atom_ref": item["atom_ref"],
                        "reason": "capability_unavailable_in_runtime_state",
                    }
                )
                continue
            visible_capabilities.append(item)

        open_commitments = [
            item
            for item in by_type["commitment"]
            if str(item["payload"].get("status", "open")).lower()
            not in {"fulfilled", "cancelled", "canceled", "superseded"}
        ]
        response_items = [
            *by_type["self_model"],
            *visible_capabilities,
            *by_type["limitation"],
            *open_commitments,
            *by_type["self_assessment"],
        ]
        if latest_runtime:
            response_items.append(latest_runtime)
        for rank, item in enumerate(response_items, start=1):
            item["rank"] = rank
        selected = {item["atom_ref"] for item in response_items}
        conflicts = []
        if selected:
            for edge in self.store.list_edges():
                if edge["relation"] not in CONFLICT_RELATIONS:
                    continue
                if edge["source_ref"] in selected or edge["target_ref"] in selected:
                    conflicts.append(edge)
        graph_version = self.store.graph_version()
        request = {
            "scope": request_scope,
            "requester": requester,
            "target_processor": target_processor,
            "retrieval_mode": "self_awareness",
            "agent_id": agent_id,
            "structural": True,
            "budget_policy": "required_self_awareness_fields_not_budget_limited",
            "pressure_mode": pressure_mode,
        }
        packet_id = stable_id(
            "pkt",
            {"request": request, "graph_version": graph_version, "items": response_items},
        )
        used_bytes = len(canonical_json(response_items).encode("utf-8"))
        packet = {
            "packet_id": packet_id,
            "schema_version": SCHEMA_VERSION,
            "request": request,
            "graph_version": graph_version,
            "generated_at": utc_now(),
            "target_processor": target_processor,
            "retrieval_mode": "self_awareness",
            "scope": request_scope,
            "pressure_mode": pressure_mode,
            "items": response_items,
            "omissions": omissions,
            "conflicts": conflicts,
            "degradation": {
                "mode": "smp-deterministic-local",
                "pressure_mode": pressure_mode,
                "reduced_recall_depth": False,
                "omitted_evidence_detail": any(
                    omission["reason"] == "evidence_access_denied"
                    for omission in omissions
                ),
                "index_freshness": {
                    "semantic_index": "inline_rebuildable",
                    "graph_version": graph_version,
                },
                "reason_codes": sorted({omission["reason"] for omission in omissions}),
                "vector_index_available": False,
                "byte_budget": None,
                "used_bytes": used_bytes,
            },
            "provenance": {
                "store": getattr(self.store, "backend_name", "unknown"),
                "journal_head": self.store.last_event_hash(),
                "ranker_profile_id": "amos.v1.self_awareness_structural",
                "smp_processor_id": self.smp.processor_id,
            },
            "cache_policy": {"cacheable": True, "keyed_by_graph_version": True},
        }
        with self.store.transaction() as conn:
            self.store.cache_packet(
                conn,
                packet_id=packet_id,
                request=request,
                response=packet,
                graph_version=graph_version,
            )
        return {
            "view": "self_awareness",
            "agent_id": agent_id,
            "graph_version": graph_version,
            "generated_at": utc_now(),
            "self_model": by_type["self_model"],
            "capabilities": visible_capabilities,
            "limitations": by_type["limitation"],
            "open_commitments": open_commitments,
            "runtime_state": latest_runtime,
            "assessments": by_type["self_assessment"],
            "calibration": self.calibrate_self_model(
                agent_id=agent_id, scope=scope or {}, record=False
            )["calibration"],
            "omissions": omissions,
            "conflicts": conflicts,
            "source_packet_id": packet_id,
        }

    def calibrate_self_model(
        self,
        *,
        agent_id: str,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        record: bool = False,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        atoms = [
            atom
            for atom in self.store.list_atoms()
            if not atom.get("deleted")
            and scope_visible(atom["scope"], scope)
            and payload_agent_id(atom["payload"]) in {None, agent_id}
        ]
        capabilities = [atom for atom in atoms if atom["type"] == "capability"]
        outcomes = [atom for atom in atoms if atom["type"] == "action_outcome"]
        unverified = []
        for capability in capabilities:
            name = payload_capability_name(capability["payload"])
            has_evidence = bool(capability["evidence_refs"])
            has_success = any(
                name
                and name in canonical_json(outcome["payload"])
                and str(outcome["payload"].get("status", "")).lower()
                in {"success", "succeeded"}
                for outcome in outcomes
            )
            if not has_evidence and not has_success:
                unverified.append(name or capability["id"])
        rate = len(unverified) / len(capabilities) if capabilities else 0.0
        calibration = {
            "capability_claim_count": len(capabilities),
            "unverified_capability_count": len(unverified),
            "unverified_capabilities": unverified,
            "overconfident_claim_rate": round(rate, 4),
        }
        result = {"status": "calibrated", "agent_id": agent_id, "calibration": calibration}
        if record and capabilities:
            result["assessment"] = self.record_self_assessment(
                agent_id=agent_id,
                claim="capability self-report calibration",
                calibration=calibration,
                scope=scope,
                actor=actor,
            )
        return result

    def retrieve_agentic_recall(
        self,
        *,
        agent_id: str,
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "planner",
    ) -> dict[str, Any]:
        packet = self.retrieve_packet(
            cues=cues or [],
            scope=scope or {},
            requester=requester,
            target_processor=target_processor,
            retrieval_mode="agentic_recall",
            max_items=100,
            include_conflicts=True,
            include_low_health=True,
            type_filter=[
                "action_outcome",
                "agentic_trace",
                "limitation",
                "self_assessment",
                "self_narrative",
            ],
            run_policy=False,
        )
        recalls = []
        self_actions = []
        other_agent_actions = []
        shared_system_actions = []
        external_actions = []
        unknown_responsibility_actions = []
        omissions = list(packet["omissions"])
        active_narratives = []
        expired_narratives = []
        for item in packet["items"]:
            item_agent = payload_agent_id(item["payload"])
            responsibility = item["payload"].get("responsibility")
            item_kind = item["type"]
            if item_kind in {"action_outcome", "agentic_trace"}:
                responsibility_class = self._agentic_responsibility(
                    item, agent_id=agent_id
                )
                attributed = dict(item)
                attributed["responsibility"] = responsibility_class
                if responsibility_class == "other_agent":
                    other_agent_actions.append(attributed)
                    continue
                if responsibility_class == "shared_system":
                    shared_system_actions.append(attributed)
                    continue
                if responsibility_class == "external":
                    external_actions.append(attributed)
                    continue
                if responsibility_class == "unknown":
                    unknown_responsibility_actions.append(attributed)
                    continue
                self_actions.append(attributed)
            elif item_agent not in {None, agent_id}:
                attributed = dict(item)
                attributed["responsibility"] = (
                    "shared_system"
                    if responsibility == "shared_system"
                    else "other_agent"
                )
                if attributed["responsibility"] == "shared_system":
                    shared_system_actions.append(attributed)
                else:
                    other_agent_actions.append(attributed)
                continue
            if item["type"] == "self_narrative":
                if self._self_narrative_has_counterevidence(item, packet["items"]):
                    expired_narratives.append(item)
                    omissions.append(
                        {
                            "atom_ref": item["atom_ref"],
                            "reason": "self_narrative_drift",
                        }
                    )
                    continue
                active_narratives.append(item)
                continue
            recalls.append(item)
        material_counterevidence = [
            item
            for item in recalls
            if item["type"] in {"action_outcome", "agentic_trace", "limitation"}
            and (
                str(
                    item["payload"].get("status")
                    or item["payload"].get("outcome")
                    or item["payload"].get("result")
                    or ""
                ).lower()
                in {"blocked", "denied", "error", "failed", "failure"}
                or item["payload"].get("correction")
                or item["payload"].get("limitation")
                or item["type"] == "limitation"
            )
        ]
        external_constraints = [
            constraint
            for item in recalls + external_actions + shared_system_actions
            for constraint in item["payload"].get("external_constraints", [])
        ]
        return {
            "view": "agentic_recall",
            "agent_id": agent_id,
            "graph_version": packet["graph_version"],
            "generated_at": utc_now(),
            "successes": self._status_items(recalls, {"success", "succeeded"}),
            "failures": self._status_items(recalls, {"failure", "failed", "error"}),
            "blocked": self._status_items(recalls, {"blocked", "denied"}),
            "corrections": [
                item for item in recalls if item["payload"].get("correction")
            ],
            "traces": [item for item in recalls if item["type"] == "agentic_trace"],
            "self_actions": self_actions,
            "other_agent_actions": other_agent_actions,
            "shared_system_actions": shared_system_actions,
            "external_actions": external_actions,
            "unknown_responsibility_actions": unknown_responsibility_actions,
            "external_constraints": external_constraints,
            "material_counterevidence": material_counterevidence,
            "self_narratives": active_narratives,
            "expired_self_narratives": expired_narratives,
            "omissions": omissions,
            "conflicts": packet["conflicts"],
            "source_packet_id": packet["packet_id"],
        }

    def retrieve_shared_view(
        self,
        *,
        processor_ids: Sequence[str],
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        max_items: int = 20,
    ) -> dict[str, Any]:
        packets = {}
        union: dict[str, list[dict[str, Any]]] = {}
        overlays: dict[str, list[str]] = {}
        omissions_by_identity: dict[str, list[dict[str, Any]]] = {}
        graph_versions = []
        for processor_id in processor_ids:
            packet = self.retrieve_packet(
                cues=cues or [],
                scope=scope or {},
                requester=requester,
                target_processor=processor_id,
                retrieval_mode="shared_coordination",
                max_items=max_items,
                include_conflicts=True,
            )
            packets[processor_id] = packet["packet_id"]
            graph_versions.append(packet["graph_version"])
            overlays[processor_id] = []
            omissions_by_identity[processor_id] = list(packet["omissions"])
            for item in packet["items"]:
                union.setdefault(item["atom_ref"], []).append(item)
                overlays[processor_id].append(item["atom_ref"])
        common_items = [
            self._shared_common_item(atom_ref, items, processor_count=len(processor_ids))
            for atom_ref, items in union.items()
        ]
        return {
            "view": "shared_memory",
            "processor_ids": list(processor_ids),
            "common_graph_version": min(graph_versions) if graph_versions else self.store.graph_version(),
            "generated_at": utc_now(),
            "items": common_items,
            "per_processor_overlays": overlays,
            "omissions_by_identity": omissions_by_identity,
            "source_packets": packets,
        }

    def refresh_shared_view(
        self,
        *,
        processor_ids: Sequence[str],
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        max_items: int = 20,
    ) -> dict[str, Any]:
        view = self.retrieve_shared_view(
            processor_ids=processor_ids,
            cues=cues,
            scope=scope,
            requester=requester,
            max_items=max_items,
        )
        view["refresh_status"] = "refreshed"
        return view

    def evaluate_procedure_execution(
        self,
        *,
        procedure_ref: str,
        autonomous: bool = False,
        approved_by: str | None = None,
        tool_permission_binding: Mapping[str, Any] | None = None,
        preconditions_satisfied: bool = False,
        rollback_plan: Mapping[str, Any] | None = None,
        review_status: str | None = None,
    ) -> dict[str, Any]:
        procedure = self.store.get_atom(procedure_ref)
        if procedure is None or procedure.get("deleted"):
            raise ValidationError(f"unknown procedure atom: {procedure_ref}")
        if procedure["type"] != "procedure":
            raise ValidationError(f"atom is not a procedure: {procedure_ref}")
        if autonomous:
            return {
                "status": "denied",
                "procedure_ref": procedure_ref,
                "reason": "autonomous_external_state_execution_not_allowed_in_v1",
                "advisory_rendering": self._render_atom(procedure),
            }
        missing = []
        if not approved_by:
            missing.append("approved_by")
        if not tool_permission_binding:
            missing.append("tool_permission_binding")
        if not preconditions_satisfied:
            missing.append("preconditions_satisfied")
        if not rollback_plan:
            missing.append("rollback_plan")
        if review_status != "approved":
            missing.append("review_status:approved")
        if missing:
            return {
                "status": "review_required",
                "procedure_ref": procedure_ref,
                "missing": missing,
                "advisory_rendering": self._render_atom(procedure),
            }
        return {
            "status": "eligible_for_external_executor",
            "procedure_ref": procedure_ref,
            "approved_by": approved_by,
            "tool_permission_binding": dict(tool_permission_binding),
            "preconditions_satisfied": True,
            "rollback_plan": dict(rollback_plan),
            "review_status": review_status,
            "note": "AMOS evaluated policy only; execution remains outside AMOS.",
        }

    def llm_reviewer_policy(self) -> dict[str, Any]:
        return {
            "enabled_by_default": False,
            "allowed_when_enabled": [
                "ambiguous_atomization",
                "scope_refinement_suggestions",
                "contradiction_analysis_suggestions",
                "natural_language_explanation_drafting",
            ],
            "forbidden": [
                "direct_canonical_mutation",
                "deletion_approval",
                "access_policy_change",
                "autonomous_preference_alteration",
            ],
            "output_envelope": [
                "processor_id",
                "processor_version",
                "input_refs",
                "output_type",
                "confidence",
                "reason_code",
                "evidence_refs",
                "recommended_action",
                "risk_level",
            ],
        }

    def request_maintenance(
        self,
        *,
        action: str,
        target_refs: Sequence[str] | None = None,
        risk: str = "low",
        approved_by: str | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        if risk == "high" or action in HIGH_RISK_MAINTENANCE:
            if not approved_by:
                return {
                    "status": "review_required",
                    "action": action,
                    "target_refs": list(target_refs or []),
                    "risk": "high",
                    "reason": "high-risk maintenance requires explicit approval",
                    "mutated": False,
                }
        if action in {"cleanup", "steward", "deduplicate", "detect_contradictions"}:
            return self.run_steward(
                scope=scope or {}, actor=actor, approved_by=approved_by
            )
        if action == "delete":
            results = [
                self.delete_atom(ref, reason="approved maintenance", actor=actor)
                for ref in target_refs or []
            ]
            return {
                "status": "completed",
                "action": action,
                "approved_by": approved_by,
                "results": results,
            }
        return {
            "status": "preview",
            "action": action,
            "target_refs": list(target_refs or []),
            "risk": risk,
            "mutated": False,
        }

    def run_maintenance_distiller(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        actor: str = "svc:memory_policy",
        domain: str = "generic",
        processor_ids: Sequence[str] | None = None,
        max_atoms: int = 128,
        max_events: int = 64,
        max_retrieval_outcomes: int = 64,
        auto_commit_low_risk: bool = True,
        reviewer: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        processor_ids = list(processor_ids or [])
        window = self._maintenance_evidence_window(
            scope=scope,
            domain=domain,
            max_atoms=max_atoms,
            max_events=max_events,
            max_retrieval_outcomes=max_retrieval_outcomes,
        )
        processors = self.maintenance_processors.select(
            window, processor_ids=processor_ids
        )
        known_processor_ids = {
            item["processor_id"] for item in self.maintenance_processors.list()
        }
        missing_processors = sorted(set(processor_ids) - known_processor_ids)
        proposals = []
        semantic_facets = []
        for processor in processors:
            proposals.extend(proposal.to_dict() for proposal in processor.propose(window))
            extract_facets = getattr(processor, "extract_facets", None)
            if callable(extract_facets):
                semantic_facets.extend(extract_facets(window))
        relation_proposals = []
        if semantic_facets:
            relation_proposals = [
                proposal.to_dict()
                for proposal in semantic_relation_proposals_from_facets(
                    semantic_facets,
                    existing_edges=window.edges,
                )
            ]
            proposals.extend(relation_proposals)
        proposals.sort(
            key=lambda proposal: (
                proposal["risk_level"] != "low",
                proposal["processor_id"],
                proposal["proposal_id"],
            )
        )
        processor_records = [
            {
                "processor_id": processor.processor_id,
                "processor_version": processor.processor_version,
            }
            for processor in processors
        ]
        if relation_proposals:
            processor_records.append(
                {
                    "processor_id": SEMANTIC_RELATION_PROCESSOR_ID,
                    "processor_version": SEMANTIC_RELATION_PROCESSOR_VERSION,
                }
            )
        reviewer_status = self._maintenance_reviewer_status(reviewer)
        if not proposals and not missing_processors:
            return {
                "status": "skipped",
                "reason": "no_proposals",
                "scope": scope,
                "domain": domain,
                "window": window.to_dict(),
                "processors": processor_records,
                "missing_processors": missing_processors,
                "proposals": [],
                "committed": [],
                "deferred": [],
                "reviewer": reviewer_status,
                "event": None,
                "graph_version": self.store.graph_version(),
            }

        committed: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        for proposal in proposals:
            if auto_commit_low_risk and proposal_is_auto_committable(proposal):
                committed.append(
                    self._commit_maintenance_proposal(proposal, actor=actor)
                )
            else:
                deferred.append(
                    {
                        "proposal_id": proposal["proposal_id"],
                        "action": proposal["action"],
                        "risk_level": proposal["risk_level"],
                        "reason": "auto_commit_disabled"
                        if not auto_commit_low_risk
                        else "requires_review_or_unsupported_action",
                        "source_refs": proposal["source_refs"],
                        "proposal_digest": self._maintenance_proposal_fingerprint(
                            proposal
                        ),
                    }
                )

        committed_count = len(
            [item for item in committed if item.get("status") == "committed"]
        )
        already_committed_count = len(
            [item for item in committed if item.get("status") == "already_committed"]
        )
        blocked_fingerprint = self._maintenance_distiller_blocked_fingerprint(
            scope=scope,
            domain=domain,
            processor_ids=processor_ids,
            missing_processors=missing_processors,
            committed=committed,
            deferred=deferred,
            reviewer_status=reviewer_status,
            auto_commit_low_risk=auto_commit_low_risk,
        )
        if proposals and committed_count == 0 and not deferred:
            return {
                "status": "skipped",
                "reason": "all_proposals_already_committed",
                "scope": scope,
                "domain": domain,
                "window": window.to_dict(),
                "processors": processor_records,
                "missing_processors": missing_processors,
                "proposals": proposals,
                "committed": committed,
                "deferred": deferred,
                "reviewer": reviewer_status,
                "event": None,
                "graph_version": self.store.graph_version(),
            }
        if (
            proposals
            and committed_count == 0
            and deferred
            and self.store.get_meta(
                self._maintenance_distiller_blocked_state_key(
                    scope=scope, domain=domain, processor_ids=processor_ids
                )
            )
            == blocked_fingerprint
        ):
            return {
                "status": "skipped",
                "reason": "deferred_proposals_unchanged",
                "scope": scope,
                "domain": domain,
                "window": window.to_dict(),
                "processors": processor_records,
                "missing_processors": missing_processors,
                "proposals": proposals,
                "committed": committed,
                "deferred": deferred,
                "reviewer": reviewer_status,
                "deferred_fingerprint": blocked_fingerprint,
                "event": None,
                "graph_version": self.store.graph_version(),
            }

        target_refs = [
            ref
            for proposal in proposals
            for ref in proposal.get("source_refs", []) + proposal.get("target_refs", [])
        ]
        target_refs.extend(
            committed_item["atom"]["id"]
            for committed_item in committed
            if committed_item.get("atom")
        )
        target_refs.extend(
            ref
            for committed_item in committed
            if committed_item.get("edge")
            for ref in (
                committed_item["edge"].get("source_ref"),
                committed_item["edge"].get("target_ref"),
            )
            if ref
        )
        event_payload = {
            "operation": "run_maintenance_distiller",
            "scope": scope,
            "domain": domain,
            "window": window.to_dict(),
            "processors": processor_records,
            "missing_processors": missing_processors,
            "proposal_count": len(proposals),
            "committed_count": committed_count,
            "already_committed_count": already_committed_count,
            "deferred_count": len(deferred),
            "auto_commit_low_risk": auto_commit_low_risk,
            "reviewer": reviewer_status,
            "deferred_fingerprint": blocked_fingerprint if deferred else None,
            "deferred_proposal_ids": [
                item["proposal_id"] for item in deferred if item.get("proposal_id")
            ],
        }
        with self.store.transaction() as conn:
            event = self.store.append_event(
                conn,
                event_type="maintenance_distillation_run",
                actor=actor,
                payload=event_payload,
                target_refs=sorted(set(target_refs)),
                authorization_context={
                    "auto_commit_low_risk": auto_commit_low_risk,
                    "reviewer_authority": event_payload["reviewer"]["authority"],
                },
            )
            if deferred:
                self.store._set_meta(
                    conn,
                    self._maintenance_distiller_blocked_state_key(
                        scope=scope, domain=domain, processor_ids=processor_ids
                    ),
                    blocked_fingerprint,
                )
        return {
            "status": "completed",
            "scope": scope,
            "domain": domain,
            "window": window.to_dict(),
            "processors": event_payload["processors"],
            "missing_processors": missing_processors,
            "proposals": proposals,
            "committed": committed,
            "deferred": deferred,
            "reviewer": event_payload["reviewer"],
            "deferred_fingerprint": event_payload["deferred_fingerprint"],
            "event": event,
            "graph_version": self.store.graph_version(),
        }

    def run_smp_analysis(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        target_refs: Sequence[str] | None = None,
        max_atoms: int | None = None,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        self._sync_smp_vector_model()
        atoms = [
            atom
            for atom in self.store.list_atoms_filtered()
            if not atom.get("deleted") and scope_visible(atom["scope"], scope)
        ]
        if target_refs:
            allowed = set(target_refs)
            atoms = [atom for atom in atoms if atom["id"] in allowed]
        total_atom_count = len(atoms)
        if max_atoms is not None:
            atoms.sort(
                key=lambda atom: (
                    str(
                        atom.get("observed_at")
                        or atom.get("updated_at")
                        or atom.get("created_at")
                        or ""
                    ),
                    str(atom.get("id") or ""),
                ),
                reverse=True,
            )
            atoms = atoms[: max(1, int(max_atoms or 1))]
        shape_reports = [self.smp.validate_shape(atom) for atom in atoms]
        clusters = self.smp.cluster(atoms)
        conflicts = self.smp.detect_conflicts(atoms)
        health = [self.smp.propose_health(atom) for atom in atoms]
        links = []
        for index, atom in enumerate(atoms):
            links.extend(
                self.smp.propose_links(
                    atom,
                    self._smp_link_candidates(atom, atoms[index + 1 :]),
                )
            )
        outputs = shape_reports + clusters + conflicts + health + links
        return {
            "status": "completed",
            "processor_id": self.smp.processor_id,
            "processor_version": self.smp.processor_version,
            "graph_version": self.store.graph_version(),
            "scope": scope,
            "atom_count": total_atom_count,
            "analyzed_atom_count": len(atoms),
            "omitted_atom_count": max(0, total_atom_count - len(atoms)),
            "outputs": outputs,
            "review_required": [
                output
                for output in outputs
                if output["risk_level"] == "high"
                or output["recommended_action"].get("type") in HIGH_RISK_MAINTENANCE
            ],
        }

    def _smp_link_candidates(
        self,
        atom: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        *,
        limit: int = 24,
    ) -> list[Mapping[str, Any]]:
        if len(candidates) <= limit:
            return list(candidates)
        atom_index = self._atom_search_index(atom)
        atom_tokens = set(atom_index["tokens"])
        scored = []
        for candidate in candidates:
            candidate_tokens = set(self._atom_search_index(candidate)["tokens"])
            overlap = len(atom_tokens.intersection(candidate_tokens))
            same_type = 1 if candidate.get("type") == atom.get("type") else 0
            if overlap <= 0 and not same_type:
                continue
            scored.append((same_type, overlap, str(candidate.get("updated_at") or ""), candidate))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored[:limit]]

    def run_steward(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        actions: list[dict[str, Any]] = []
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        with self.store.transaction() as conn:
            atoms = [
                atom
                for atom in self.store.list_atoms_filtered(
                    lifecycle_states=["active"],
                )
                if not atom.get("deleted")
                and atom.get("lifecycle_state") == "active"
                and scope_visible(atom["scope"], scope)
            ]
            live_attachment_relations = {
                "rel:attributed_to",
                "rel:has_capability",
                "rel:has_limitation",
                "rel:made_commitment",
            }
            invalid_attachment_edge_ids: list[str] = []
            for edge in self.store.list_edges():
                if edge.get("relation") not in live_attachment_relations:
                    continue
                endpoints = (
                    self.store.get_atom(str(edge.get("source_ref") or "")),
                    self.store.get_atom(str(edge.get("target_ref") or "")),
                )
                if any(
                    atom is None
                    or atom.get("deleted")
                    or atom.get("lifecycle_state") != "active"
                    for atom in endpoints
                ):
                    invalid_attachment_edge_ids.append(str(edge.get("edge_id") or ""))
            invalid_attachment_edges = self.store.mark_edges_deleted(
                conn, invalid_attachment_edge_ids
            )
            if invalid_attachment_edges:
                projected_edges.extend(invalid_attachment_edges)
                actions.append(
                    {
                        "action": "prune_inactive_attachment_edges",
                        "edge_count": len(invalid_attachment_edges),
                        "relations": sorted(
                            {
                                str(edge.get("relation") or "")
                                for edge in invalid_attachment_edges
                            }
                        ),
                    }
                )
            smp_outputs = self.smp.cluster(atoms) + self.smp.detect_conflicts(atoms)
            existing_edge_ids = {
                edge["edge_id"] for edge in self.store.list_edges()
            }
            intrinsic_edge_count = 0
            for atom in atoms:
                for edge in self._intrinsic_edges_for_atom(atom):
                    if edge["edge_id"] in existing_edge_ids:
                        continue
                    self.store.insert_edge(conn, edge)
                    existing_edge_ids.add(edge["edge_id"])
                    projected_edges.append(edge)
                    intrinsic_edge_count += 1
            if intrinsic_edge_count:
                actions.append(
                    {
                        "action": "project_intrinsic_edges",
                        "edge_count": intrinsic_edge_count,
                        "policy": "deterministic_structured_atom_refs",
                    }
                )
            seen: dict[str, dict[str, Any]] = {}
            for atom in sorted(atoms, key=lambda row: row["created_at"]):
                key = digest(
                    {
                        "type": atom["type"],
                        "payload": atom["payload"],
                        "scope": atom["scope"],
                    }
                )
                existing = seen.get(key)
                if existing is None:
                    seen[key] = atom
                    continue
                duplicate, deleted_edges = self._archive_atom_projection(
                    conn,
                    atom,
                    reason="exact_duplicate",
                    superseded_by=existing["id"],
                    actor=actor,
                )
                projected_atoms.append(duplicate)
                projected_edges.extend(deleted_edges)
                actions.append(
                    {
                        "action": "deduplicate",
                        "kept": existing["id"],
                        "archived": duplicate["id"],
                        "smp_outputs": [
                            output
                            for output in smp_outputs
                            if output["reason_code"] == "near_duplicate"
                            and atom["id"] in output["input_refs"]
                        ],
                    }
                )
            structured_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            archived_ids = {atom["id"] for atom in projected_atoms}
            for atom in atoms:
                if atom["id"] in archived_ids:
                    continue
                key = self._structured_duplicate_key(atom)
                if key is None:
                    continue
                structured_groups.setdefault(key, []).append(atom)
            for key, group in structured_groups.items():
                active_group = [
                    atom
                    for atom in group
                    if atom["id"] not in archived_ids
                    and atom.get("lifecycle_state") == "active"
                    and not atom.get("deleted")
                ]
                if len(active_group) < 2:
                    continue
                kept = max(
                    active_group,
                    key=lambda atom: (
                        self._structured_duplicate_quality(atom),
                        str(atom.get("updated_at") or ""),
                        str(atom.get("id") or ""),
                    ),
                )
                for atom in active_group:
                    if atom["id"] == kept["id"]:
                        continue
                    duplicate, deleted_edges = self._archive_atom_projection(
                        conn,
                        atom,
                        reason=f"structured_duplicate:{key[0]}",
                        superseded_by=kept["id"],
                        actor=actor,
                    )
                    archived_ids.add(duplicate["id"])
                    projected_atoms.append(duplicate)
                    projected_edges.extend(deleted_edges)
                    actions.append(
                        {
                            "action": "archive_structured_duplicate",
                            "kind": key[0],
                            "kept": kept["id"],
                            "archived": duplicate["id"],
                            "deleted_edge_count": len(deleted_edges),
                        }
                    )

            contradiction_groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
            for atom in atoms:
                if atom["id"] in archived_ids:
                    continue
                signature = self._contradiction_signature(atom)
                if signature is None:
                    continue
                key, value = signature
                contradiction_groups.setdefault(key, {})[value] = atom
            for values in contradiction_groups.values():
                if len(values) < 2:
                    continue
                group_atoms = list(values.values())
                if approved_by:
                    for atom in group_atoms:
                        changed = dict(atom)
                        changed["health_status"] = "contradicted"
                        changed["version"] = int(changed["version"]) + 1
                        changed["updated_at"] = utc_now()
                        changed = normalize_atom(
                            self._attach_search_index(changed), require_id=True
                        )
                        self.store.replace_atom(conn, changed)
                        projected_atoms.append(changed)
                    for source in group_atoms:
                        for target in group_atoms:
                            if source["id"] >= target["id"]:
                                continue
                            edge = self._edge(
                                source["id"], target["id"], "rel:contradicts", scope
                            )
                            self.store.insert_edge(conn, edge)
                            projected_edges.append(edge)
                actions.append(
                    {
                        "action": "mark_contradiction"
                        if approved_by
                        else "propose_contradiction_review",
                        "atom_refs": [atom["id"] for atom in group_atoms],
                        "review_required": approved_by is None,
                        "approved_by": approved_by,
                        "smp_outputs": [
                            output
                            for output in smp_outputs
                            if output["reason_code"] == "contradiction_candidate"
                            and any(ref in output["input_refs"] for ref in [atom["id"] for atom in group_atoms])
                        ],
                    }
                )

            event = self.store.append_event(
                conn,
                event_type="steward_run",
                actor=actor,
                payload={
                    "operation": "run_steward",
                    "scope": scope,
                    "actions": actions,
                    "projected_atoms": projected_atoms,
                    "projected_edges": projected_edges,
                },
                target_refs=[
                    ref
                    for action in actions
                    for ref in action.get("atom_refs", [])
                    + [action.get("kept"), action.get("archived")]
                    if ref
                ],
                authorization_context={"approved_by": approved_by}
                if approved_by
                else {},
            )
            self.store.clear_packet_cache(conn)
        return {
            "status": "completed",
            "actions": actions,
            "event": event,
            "graph_version": self.store.graph_version(),
        }

    def health_memory(self, *, run_policy: bool = True) -> dict[str, Any]:
        policy_tick = (
            self.run_memory_policy(trigger="health_memory")
            if run_policy
            else {
                "status": "skipped",
                "reason": "policy_not_run_for_health",
                "trigger": "health_memory",
                "graph_version": self.store.graph_version(),
            }
        )
        indexes = self.store.list_derived_index_metadata()
        by_type = self.store.atom_counts_by("type")
        by_health = self.store.atom_counts_by("health_status")
        by_lifecycle = self.store.atom_counts_by("lifecycle_state")
        return {
            "graph_version": self.store.graph_version(),
            "journal_events": self.store.event_count(),
            "atoms": self.store.atom_count(),
            "by_type": by_type,
            "by_health": by_health,
            "by_lifecycle": by_lifecycle,
            "journal_head": self.store.last_event_hash(),
            "projection_lag": 0,
            "index_freshness": {
                index["index_name"]: {
                    "graph_version": index["graph_version"],
                    "freshness": index["freshness"],
                    "rebuilt_at": index["rebuilt_at"],
                }
                for index in indexes
            },
            "retrieval_outcomes": self.store.retrieval_outcome_count(),
            "deletion_residuals": {
                "offline_backup_residual_window_days": 30,
                "hot_packet_cache_policy": "purged_on_canonical_mutation",
            },
            "quality": self._memory_quality_diagnostics(
                policy=self.memory_policy(),
                indexes=indexes,
            ),
            "memory_policy": self.memory_policy_status(),
            "last_policy_tick": policy_tick,
        }

    def health_capacity(self) -> dict[str, Any]:
        path = self.store.path
        size_bytes = path.stat().st_size if path.exists() and str(path) != ":memory:" else 0
        budget = self._capacity_budget()
        pressure_mode = self._capacity_pressure_mode(size_bytes=size_bytes, budget=budget)
        return {
            "store": getattr(self.store, "backend_name", "unknown"),
            "path": str(path),
            "size_bytes": size_bytes,
            "capacity_budget": budget,
            "pressure_mode": pressure_mode,
            "graph_version": self.store.graph_version(),
            "degradation": {
                "vector_index_available": False,
                "external_object_store_available": False,
                "pressure_degraded": pressure_mode in {"orange", "red"},
            },
        }

    def verify_journal_chain(self) -> dict[str, Any]:
        events = self.store.list_events()
        previous = "genesis"
        failures = []
        for event in events:
            if event["previous_event_hash"] != previous:
                failures.append(
                    {
                        "event_id": event["event_id"],
                        "reason": "previous_event_hash_mismatch",
                        "expected": previous,
                        "actual": event["previous_event_hash"],
                    }
                )
            event_without_checksum = dict(event)
            checksum = event_without_checksum.pop("checksum")
            if digest(event_without_checksum) != checksum:
                failures.append(
                    {
                        "event_id": event["event_id"],
                        "reason": "checksum_mismatch",
                    }
                )
            previous = event["checksum"]
        return {
            "status": "ok" if not failures else "failed",
            "event_count": len(events),
            "graph_version": self.store.graph_version(),
            "journal_head": self.store.last_event_hash(),
            "failures": failures,
        }

    def replay_graph(self) -> dict[str, Any]:
        atoms: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        tombstones: dict[str, dict[str, Any]] = {}
        for event in self.store.list_events():
            payload = event["payload"]
            event_type = event["event_type"]
            if event_type == "atom_committed":
                atom = payload["atom"]
                atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = edge
            elif event_type == "atom_updated":
                atom = payload["after"]
                atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = edge
            elif event_type == "atom_deleted":
                before = payload["before"]
                atom_id = before["id"]
                atoms.pop(atom_id, None)
                tombstone = payload["tombstone"]
                tombstones[tombstone["target_ref"]] = tombstone
                for edge in payload.get("projected_edges", []):
                    edges.pop(edge["edge_id"], None)
            elif event_type == "memories_distilled":
                atom = payload["atom"]
                atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = edge
            elif event_type == "edge_committed":
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = edge
            elif event_type in {
                "atom_merged",
                "steward_run",
                "retrieval_outcome_recorded",
                "decay_policy_applied",
                "storage_cleanup_run",
            }:
                for atom in payload.get("projected_atoms", []):
                    if atom.get("deleted"):
                        atoms.pop(atom["id"], None)
                    else:
                        atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = edge
                for tombstone in payload.get("tombstones", []):
                    tombstones[tombstone["target_ref"]] = tombstone
        return {
            "graph_version": self.store.graph_version(),
            "atoms": atoms,
            "edges": edges,
            "tombstones": tombstones,
        }

    def verify_replay(self) -> dict[str, Any]:
        replayed = self.replay_graph()
        stored_atoms = {
            atom["id"]: atom
            for atom in self.store.list_atoms()
            if not atom.get("deleted")
        }
        replayed_atoms = replayed["atoms"]
        replayed_edges = replayed["edges"]
        missing = sorted(set(stored_atoms) - set(replayed_atoms))
        unexpected = sorted(set(replayed_atoms) - set(stored_atoms))
        mismatched = []
        for atom_id in sorted(set(stored_atoms).intersection(replayed_atoms)):
            if digest(self._atom_projection(stored_atoms[atom_id])) != digest(
                self._atom_projection(replayed_atoms[atom_id])
            ):
                mismatched.append(atom_id)
        stored_edges = {
            edge["edge_id"]: edge
            for edge in self.store.list_edges()
            if not edge.get("deleted")
        }
        missing_edges = sorted(set(stored_edges) - set(replayed_edges))
        unexpected_edges = sorted(set(replayed_edges) - set(stored_edges))
        mismatched_edges = []
        for edge_id in sorted(set(stored_edges).intersection(replayed_edges)):
            if digest(stored_edges[edge_id]) != digest(replayed_edges[edge_id]):
                mismatched_edges.append(edge_id)
        return {
            "status": "ok"
            if not missing
            and not unexpected
            and not mismatched
            and not missing_edges
            and not unexpected_edges
            and not mismatched_edges
            else "failed",
            "graph_version": self.store.graph_version(),
            "missing_in_replay": missing,
            "unexpected_in_replay": unexpected,
            "mismatched_atoms": mismatched,
            "missing_edges_in_replay": missing_edges,
            "unexpected_edges_in_replay": unexpected_edges,
            "mismatched_edges": mismatched_edges,
            "replayed_atom_count": len(replayed_atoms),
            "stored_atom_count": len(stored_atoms),
            "replayed_edge_count": len(replayed_edges),
            "stored_edge_count": len(stored_edges),
        }

    def _maintenance_evidence_window(
        self,
        *,
        scope: Mapping[str, Any],
        domain: str,
        max_atoms: int,
        max_events: int,
        max_retrieval_outcomes: int,
    ) -> EvidenceWindow:
        # Maintenance scopes are hierarchical: a broad tenant/component pass
        # must see atoms in narrower run, asset, and agent scopes. Filter before
        # applying the window bound so unrelated hot atoms cannot crowd the
        # requested scope out of the evidence window.
        visible_atoms = [
            atom
            for atom in self.store.list_atoms_filtered(prioritize_hot=True)
            if not atom.get("deleted")
            and maintenance_scope_visible(atom["scope"], scope)
        ]
        atoms = visible_atoms[: max(1, int(max_atoms or 1))]
        atom_refs = {atom["id"] for atom in atoms}
        edges = [
            edge
            for edge in self.store.list_edges()
            if edge["source_ref"] in atom_refs or edge["target_ref"] in atom_refs
        ]
        evidence = [
            record
            for record in self.store.list_evidence()
            if scope_visible(record.get("scope", {}), scope)
        ]
        event_limit = max(0, int(max_events or 0))
        events = self.store.list_events(limit=event_limit) if event_limit else []
        list_outcomes = getattr(self.store, "list_retrieval_outcomes", None)
        retrieval_outcomes = (
            list_outcomes(limit=max(0, int(max_retrieval_outcomes or 0)))
            if list_outcomes
            else []
        )
        return EvidenceWindow(
            atoms=tuple(atoms),
            edges=tuple(edges),
            evidence=tuple(evidence),
            retrieval_outcomes=tuple(retrieval_outcomes),
            events=tuple(events),
            scope=scope,
            domain=str(domain or "generic"),
            graph_version=self.store.graph_version(),
        )

    def _commit_maintenance_proposal(
        self, proposal: Mapping[str, Any], *, actor: str
    ) -> dict[str, Any]:
        if proposal.get("action") == "add_edge":
            return self._commit_maintenance_edge_proposal(proposal, actor=actor)
        atom_payload = proposal.get("payload", {}).get("atom")
        if not isinstance(atom_payload, Mapping):
            return {
                "status": "skipped",
                "reason": "proposal_has_no_atom_payload",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        atom = dict(atom_payload)
        atom["id"] = atom.get("id") or stable_id(
            "atom", {"maintenance_proposal_id": proposal["proposal_id"]}
        )
        atom.setdefault("evidence_refs", list(proposal.get("evidence_refs", [])))
        atom_payload_body = dict(atom.get("payload", {}))
        atom_payload_body.setdefault(
            "maintenance_proposal_id", proposal["proposal_id"]
        )
        atom_payload_body.setdefault("maintenance_reason_code", proposal["reason_code"])
        atom_payload_body.setdefault("maintenance_source_refs", proposal["source_refs"])
        atom["payload"] = atom_payload_body
        existing = self.store.get_atom(str(atom["id"]))
        if existing is not None and not existing.get("deleted"):
            return {
                "status": "already_committed",
                "proposal_id": proposal["proposal_id"],
                "atom": existing,
                "source_refs": list(proposal.get("source_refs", [])),
            }
        committed = self.commit_atom(
            atom,
            actor=actor,
            idempotency_key=stable_id(
                "maint_commit", {"proposal_id": proposal["proposal_id"]}
            ),
            authorization_context={
                "maintenance_proposal_id": proposal["proposal_id"],
                "maintenance_processor_id": proposal["processor_id"],
                "risk_level": proposal["risk_level"],
                "auto_commit_gate": "low_risk_add_atom",
            },
        )
        return {
            "status": "committed",
            "proposal_id": proposal["proposal_id"],
            "atom": committed["atom"],
            "event": committed["event"],
            "source_refs": list(proposal.get("source_refs", [])),
        }

    def _commit_maintenance_edge_proposal(
        self, proposal: Mapping[str, Any], *, actor: str
    ) -> dict[str, Any]:
        edge_payload = proposal.get("payload", {}).get("edge")
        if not isinstance(edge_payload, Mapping):
            return {
                "status": "skipped",
                "reason": "proposal_has_no_edge_payload",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        source_ref = str(edge_payload.get("source_ref", ""))
        target_ref = str(edge_payload.get("target_ref", ""))
        relation = normalize_relation(str(edge_payload.get("relation", "")))
        if not source_ref or not target_ref or source_ref == target_ref:
            return {
                "status": "skipped",
                "reason": "invalid_edge_endpoints",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        source = self.store.get_atom(source_ref)
        target = self.store.get_atom(target_ref)
        if (
            source is None
            or target is None
            or source.get("deleted")
            or target.get("deleted")
            or source.get("lifecycle_state") != "active"
            or target.get("lifecycle_state") != "active"
        ):
            return {
                "status": "skipped",
                "reason": "edge_endpoint_not_active",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        edge = self._edge(
            source_ref,
            target_ref,
            relation,
            dict(edge_payload.get("scope") or {}),
        )
        edge["evidence_refs"] = [
            str(ref) for ref in edge_payload.get("evidence_refs", [])
        ]
        edge["confidence"] = dict(
            edge_payload.get("confidence")
            or {"level": "medium-high", "score": proposal.get("confidence", 0.75)}
        )
        if any(existing["edge_id"] == edge["edge_id"] for existing in self.store.list_edges()):
            return {
                "status": "already_committed",
                "proposal_id": proposal["proposal_id"],
                "edge": edge,
                "source_refs": list(proposal.get("source_refs", [])),
            }
        with self.store.transaction() as conn:
            inserted = self.store.insert_edge(conn, edge)
            if not inserted:
                return {
                    "status": "already_committed",
                    "proposal_id": proposal["proposal_id"],
                    "edge": edge,
                    "source_refs": list(proposal.get("source_refs", [])),
                }
            event = self.store.append_event(
                conn,
                event_type="edge_committed",
                actor=actor,
                payload={
                    "operation": "commit_maintenance_edge",
                    "edge": edge,
                    "projected_edges": [edge],
                    "maintenance_proposal_id": proposal["proposal_id"],
                    "maintenance_reason_code": proposal["reason_code"],
                },
                target_refs=[source_ref, target_ref],
                authorization_context={
                    "maintenance_proposal_id": proposal["proposal_id"],
                    "maintenance_processor_id": proposal["processor_id"],
                    "risk_level": proposal["risk_level"],
                    "auto_commit_gate": "low_risk_add_edge",
                },
            )
            self.store.clear_packet_cache(conn)
        return {
            "status": "committed",
            "proposal_id": proposal["proposal_id"],
            "edge": edge,
            "event": event,
            "source_refs": list(proposal.get("source_refs", [])),
        }

    def _maintenance_reviewer_status(
        self, reviewer: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        config = dict(reviewer or {})
        enabled = bool(config.get("enabled", False))
        return {
            "enabled": enabled,
            "authority": "draft_only",
            "status": "not_configured" if enabled else "disabled",
            "mutates_canonical_memory": False,
            "allowed_outputs": [
                "proposal_explanation",
                "ambiguous_atomization_note",
                "scope_refinement_suggestion",
                "contradiction_analysis_draft",
            ],
        }

    def _maintenance_distiller_blocked_state_key(
        self,
        *,
        scope: Mapping[str, Any],
        domain: str,
        processor_ids: Sequence[str],
    ) -> str:
        return "maintenance_distiller_blocked:" + stable_id(
            "mdblk",
            {
                "scope": dict(scope),
                "domain": domain,
                "processor_ids": sorted(str(item) for item in processor_ids),
            },
        )

    def _maintenance_proposal_fingerprint(self, proposal: Mapping[str, Any]) -> str:
        return digest(
            {
                "action": proposal.get("action"),
                "risk_level": proposal.get("risk_level"),
                "source_refs": sorted(str(ref) for ref in proposal.get("source_refs", [])),
                "target_refs": sorted(str(ref) for ref in proposal.get("target_refs", [])),
                "payload": proposal.get("payload", {}),
                "recommended_action": proposal.get("recommended_action"),
                "reason_code": proposal.get("reason_code"),
                "output_type": proposal.get("output_type"),
            }
        )

    def _maintenance_distiller_blocked_fingerprint(
        self,
        *,
        scope: Mapping[str, Any],
        domain: str,
        processor_ids: Sequence[str],
        missing_processors: Sequence[str],
        committed: Sequence[Mapping[str, Any]],
        deferred: Sequence[Mapping[str, Any]],
        reviewer_status: Mapping[str, Any],
        auto_commit_low_risk: bool,
    ) -> str:
        return digest(
            {
                "scope": dict(scope),
                "domain": domain,
                "processor_ids": sorted(str(item) for item in processor_ids),
                "missing_processors": sorted(str(item) for item in missing_processors),
                "commit_eligible": sorted(
                    str(item.get("proposal_id"))
                    for item in committed
                    if item.get("proposal_id")
                ),
                "deferred": sorted(
                    (
                        {
                            "proposal_id": str(item.get("proposal_id")),
                            "action": str(item.get("action")),
                            "risk_level": str(item.get("risk_level")),
                            "reason": str(item.get("reason")),
                            "proposal_digest": str(item.get("proposal_digest") or ""),
                            "source_refs": sorted(
                                str(ref) for ref in item.get("source_refs", [])
                            ),
                        }
                        for item in deferred
                    ),
                    key=lambda item: (
                        item["proposal_id"],
                        item["action"],
                        item["risk_level"],
                    ),
                ),
                "reviewer_status": {
                    "enabled": bool(reviewer_status.get("enabled", False)),
                    "authority": str(reviewer_status.get("authority", "")),
                    "status": str(reviewer_status.get("status", "")),
                },
                "auto_commit_low_risk": bool(auto_commit_low_risk),
            }
        )

    def _idempotency_hit(
        self,
        conn: Any,
        actor: str,
        idempotency_key: str | None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        payload_digest = digest(payload)
        existing = self.store.get_idempotency(conn, actor, idempotency_key)
        if existing is None:
            return None
        if existing["payload_digest"] != payload_digest:
            raise IdempotencyConflict(
                f"idempotency key reused with different payload: {idempotency_key}"
            )
        return existing["response"]

    def _record_idempotency(
        self,
        conn: Any,
        actor: str,
        idempotency_key: str | None,
        payload: Mapping[str, Any],
        event: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        if not idempotency_key:
            return
        self.store.put_idempotency(
            conn,
            actor=actor,
            key=idempotency_key,
            payload_digest=digest(payload),
            event_id=event["event_id"],
            response=response,
        )

    def _assert_mutation_allowed(
        self,
        atom: Mapping[str, Any],
        *,
        actor: str,
        authorization_context: Mapping[str, Any] | None = None,
    ) -> None:
        if actor in {"system", "svc:memory_steward", "owner", "admin"}:
            return
        context = dict(authorization_context or {})
        policy = atom.get("access_policy", {})
        mutable_by = set(policy.get("mutable_by", ["owner"]))
        actor_roles = set(context.get("roles", []))
        if actor in mutable_by or actor_roles.intersection(mutable_by):
            pass
        else:
            raise AccessDenied(f"{actor} cannot mutate atom {atom['id']}")
        min_trust = int(policy.get("min_trust_level", 0))
        actor_trust = int(context.get("trust_level", 0))
        if actor_trust < min_trust:
            raise AccessDenied(
                f"{actor} trust_level {actor_trust} is below required {min_trust}"
            )
        required_capability = policy.get("requires_capability")
        capabilities = set(context.get("capabilities", []))
        if required_capability and required_capability not in capabilities:
            raise AccessDenied(
                f"{actor} lacks required capability {required_capability}"
            )

    def _capacity_budget(self) -> dict[str, Any]:
        raw = self.store.get_meta("capacity_budget")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(data)

    def _capacity_pressure_mode(
        self,
        *,
        size_bytes: int | None = None,
        budget: Mapping[str, Any] | None = None,
    ) -> str:
        budget = dict(budget if budget is not None else self._capacity_budget())
        if size_bytes is None:
            path = self.store.path
            size_bytes = path.stat().st_size if path.exists() and str(path) != ":memory:" else 0
        hard_limit = int(budget.get("hard_capacity_bytes", 0) or 0)
        if hard_limit <= 0:
            return "green"
        ratio = size_bytes / hard_limit
        if ratio >= float(budget.get("critical_ratio", 0.90)):
            return "red"
        if ratio >= float(budget.get("warning_ratio", 0.70)):
            return "orange"
        return "green"

    def _normalize_memory_policy(self, policy: Mapping[str, Any]) -> dict[str, Any]:
        normalized = json.loads(canonical_json(DEFAULT_MEMORY_POLICY))
        policy = dict(policy or {})
        for key, value in policy.items():
            if key in {
                "schedule",
                "maintenance",
                "distillation",
                "maintenance_distiller",
                "decay",
                "storage_cleanup",
            } and isinstance(value, Mapping):
                if key == "storage_cleanup":
                    cleanup = dict(normalized[key])
                    for cleanup_key, cleanup_value in dict(value).items():
                        if (
                            cleanup_key == "sqlite_compaction"
                            and isinstance(cleanup_value, Mapping)
                            and isinstance(cleanup.get("sqlite_compaction"), Mapping)
                        ):
                            cleanup["sqlite_compaction"] = {
                                **dict(cleanup["sqlite_compaction"]),
                                **dict(cleanup_value),
                            }
                        else:
                            cleanup[cleanup_key] = cleanup_value
                    normalized[key] = cleanup
                else:
                    normalized[key].update(dict(value))
            else:
                normalized[key] = value
        normalized["enabled"] = bool(normalized.get("enabled", True))
        schedule = normalized["schedule"]
        schedule["every_graph_versions"] = max(
            1, int(schedule.get("every_graph_versions", 25) or 25)
        )
        schedule["every_seconds"] = max(
            0, int(schedule.get("every_seconds", 300) or 0)
        )
        schedule["run_on_pressure"] = bool(schedule.get("run_on_pressure", True))
        maintenance = normalized["maintenance"]
        for key in [
            "enabled",
            "run_smp",
            "run_steward",
            "rebuild_indexes",
            "rebuild_lsa",
            "invalidate_packet_cache",
        ]:
            maintenance[key] = bool(maintenance.get(key, True))
        maintenance["max_smp_atoms"] = max(
            1,
            int(maintenance.get("max_smp_atoms", 128) or 128),
        )
        maintenance["lsa_dimensions"] = max(
            0,
            min(
                self.smp.dimensions,
                int(maintenance.get("lsa_dimensions", 32) or 0),
            ),
        )
        maintenance["lsa_max_terms"] = max(
            maintenance["lsa_dimensions"],
            int(maintenance.get("lsa_max_terms", 300) or 300),
        )
        distillation = normalized["distillation"]
        distillation["enabled"] = bool(distillation.get("enabled", True))
        distillation["min_source_atoms"] = max(
            2, int(distillation.get("min_source_atoms", 6) or 6)
        )
        distillation["max_source_atoms"] = max(
            distillation["min_source_atoms"],
            int(distillation.get("max_source_atoms", 10) or 10),
        )
        distillation["candidate_types"] = [
            str(item) for item in distillation.get("candidate_types", [])
        ]
        distillation["archive_sources"] = bool(distillation.get("archive_sources", False))
        approved_by = distillation.get("approved_by")
        distillation["approved_by"] = str(approved_by) if approved_by else None
        distillation["distillation_type"] = str(
            distillation.get("distillation_type") or "automatic_policy"
        )
        distiller = normalized["maintenance_distiller"]
        distiller["enabled"] = bool(distiller.get("enabled", True))
        distiller["auto_commit_low_risk"] = bool(
            distiller.get("auto_commit_low_risk", True)
        )
        distiller["processor_ids"] = [
            str(item) for item in distiller.get("processor_ids", [])
        ]
        distiller["domain"] = str(distiller.get("domain") or "generic")
        distiller["max_atoms"] = max(1, int(distiller.get("max_atoms", 128) or 128))
        distiller["max_events"] = max(0, int(distiller.get("max_events", 64) or 0))
        distiller["max_retrieval_outcomes"] = max(
            0,
            int(distiller.get("max_retrieval_outcomes", 64) or 0),
        )
        reviewer = dict(distiller.get("reviewer") or {})
        distiller["reviewer"] = {
            "enabled": bool(reviewer.get("enabled", False)),
            "authority": "draft_only",
        }
        decay = normalized["decay"]
        decay["enabled"] = bool(decay.get("enabled", True))
        decay["max_atoms"] = max(1, int(decay.get("max_atoms", 256) or 256))
        decay["require_atom_policy"] = bool(decay.get("require_atom_policy", True))
        decay["pressure_archive_policyless"] = bool(
            decay.get("pressure_archive_policyless", True)
        )
        decay["pressure_max_archives_per_run"] = max(
            1,
            int(decay.get("pressure_max_archives_per_run", 256) or 256),
        )
        decay["pressure_protected_types"] = sorted(
            {
                str(item)
                for item in decay.get(
                    "pressure_protected_types",
                    ["commitment", "policy", "self_model"],
                )
                if str(item)
            }
        )
        decay["capacity_assessment_targets"] = sorted(
            {
                max(1, int(item))
                for item in decay.get("capacity_assessment_targets", [256, 512, 768])
                if item not in (None, "")
            }
            | {decay["max_atoms"]}
        )
        decay["capacity_headroom_ratio"] = max(
            0.0,
            min(0.9, float(decay.get("capacity_headroom_ratio", 0.2) or 0.0)),
        )
        decay["archive_superseded"] = bool(decay.get("archive_superseded", True))
        value = decay.get("archive_superseded_after_seconds", 0)
        decay["archive_superseded_after_seconds"] = (
            None if value in (None, "") else max(0, int(value))
        )
        for key in (
            "mark_stale_after_seconds",
            "archive_after_seconds",
            "low_utility_threshold",
        ):
            value = decay.get(key)
            if value in (None, ""):
                decay[key] = None
            elif key == "low_utility_threshold":
                decay[key] = max(0.0, min(1.0, float(value)))
            else:
                decay[key] = max(0, int(value))
        cleanup = normalized["storage_cleanup"]
        cleanup["enabled"] = bool(cleanup.get("enabled", True))
        cleanup["trigger"] = str(cleanup.get("trigger") or "idle")
        if cleanup["trigger"] != "idle":
            cleanup["trigger"] = "idle"
        for key, default in (
            ("idle_after_seconds", 300),
            ("min_interval_seconds", 900),
            ("max_deletions_per_tick", 256),
            ("max_idempotency_compactions_per_tick", 512),
        ):
            cleanup[key] = max(0, int(cleanup.get(key, default) or 0))
        for key, default in (
            ("delete_archived_after_seconds", 604800),
            ("delete_stale_after_seconds", 1209600),
            ("compact_idempotency_after_seconds", 604800),
        ):
            value = cleanup.get(key, default)
            cleanup[key] = None if value in (None, "") else max(0, int(value))
        cleanup["remove_archived_from_hot_index"] = bool(
            cleanup.get("remove_archived_from_hot_index", True)
        )
        cleanup["remove_stale_from_hot_index"] = bool(
            cleanup.get("remove_stale_from_hot_index", True)
        )
        cleanup["protected_types"] = sorted(
            {str(item) for item in cleanup.get("protected_types", [])}
        )
        sqlite_compaction = dict(cleanup.get("sqlite_compaction") or {})
        checkpoint_mode = str(sqlite_compaction.get("checkpoint_mode") or "TRUNCATE").upper()
        if checkpoint_mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            checkpoint_mode = "TRUNCATE"
        cleanup["sqlite_compaction"] = {
            "checkpoint_wal": bool(sqlite_compaction.get("checkpoint_wal", True)),
            "checkpoint_mode": checkpoint_mode,
            "vacuum_enabled": bool(sqlite_compaction.get("vacuum_enabled", True)),
            "vacuum_idle_after_seconds": max(
                0, int(sqlite_compaction.get("vacuum_idle_after_seconds", 1800) or 0)
            ),
            "vacuum_min_interval_seconds": max(
                0,
                int(sqlite_compaction.get("vacuum_min_interval_seconds", 86400) or 0),
            ),
        }
        return normalized

    def _memory_policy_state(self) -> dict[str, Any]:
        raw = self.store.get_meta("memory_policy_state")
        if not raw:
            return {
                "last_run_at": None,
                "last_graph_version": 0,
                "last_trigger": None,
                "last_event_id": None,
                "last_due_reasons": [],
                "last_distilled_refs": [],
                "last_maintenance_distiller_refs": [],
                "last_storage_cleanup_at": self.store.get_meta(
                    "last_storage_cleanup_at"
                ),
                "last_vacuum_at": self.store.get_meta("last_vacuum_at"),
                "last_foreground_activity_at": self.store.get_meta(
                    "last_foreground_activity_at"
                ),
            }
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return {
            "last_run_at": data.get("last_run_at"),
            "last_graph_version": int(data.get("last_graph_version", 0) or 0),
            "last_trigger": data.get("last_trigger"),
            "last_event_id": data.get("last_event_id"),
            "last_due_reasons": list(data.get("last_due_reasons", [])),
            "last_distilled_refs": list(data.get("last_distilled_refs", [])),
            "last_maintenance_distiller_refs": list(
                data.get("last_maintenance_distiller_refs", [])
            ),
            "last_storage_cleanup_at": self.store.get_meta("last_storage_cleanup_at")
            or data.get("last_storage_cleanup_at"),
            "last_vacuum_at": self.store.get_meta("last_vacuum_at")
            or data.get("last_vacuum_at"),
            "last_foreground_activity_at": self.store.get_meta(
                "last_foreground_activity_at"
            )
            or data.get("last_foreground_activity_at"),
        }

    def _memory_policy_journal_results(
        self, results: Mapping[str, Any]
    ) -> dict[str, Any]:
        journal: dict[str, Any] = {}
        for key, value in results.items():
            if key == "smp" and isinstance(value, Mapping):
                journal[key] = self._summarize_smp_result(value)
            elif key == "steward" and isinstance(value, Mapping):
                journal[key] = self._summarize_steward_result(value)
            elif key == "distillation" and isinstance(value, Mapping):
                journal[key] = self._summarize_policy_distillation_result(value)
            elif key == "maintenance_distiller" and isinstance(value, Mapping):
                journal[key] = self._summarize_maintenance_distiller_result(value)
            elif key == "storage_cleanup" and isinstance(value, Mapping):
                journal[key] = self._summarize_storage_cleanup_result(value)
            elif key in {"index", "packet_cache"} and isinstance(value, Mapping):
                journal[key] = dict(value)
            else:
                journal[key] = self._bounded_json_summary(value)
        return journal

    def _summarize_smp_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        outputs = [
            output
            for output in result.get("outputs", [])
            if isinstance(output, Mapping)
        ]
        review_required = [
            output
            for output in result.get("review_required", [])
            if isinstance(output, Mapping)
        ]
        return {
            "status": result.get("status"),
            "processor_id": result.get("processor_id"),
            "processor_version": result.get("processor_version"),
            "graph_version": result.get("graph_version"),
            "scope": dict(result.get("scope") or {}),
            "atom_count": result.get("atom_count", 0),
            "analyzed_atom_count": result.get("analyzed_atom_count", 0),
            "omitted_atom_count": result.get("omitted_atom_count", 0),
            "output_count": len(outputs),
            "review_required_count": len(review_required),
            "output_type_counts": self._count_mapping_values(outputs, "output_type"),
            "reason_code_counts": self._count_mapping_values(outputs, "reason_code"),
            "risk_level_counts": self._count_mapping_values(outputs, "risk_level"),
            "review_required_refs": self._bounded_refs(
                ref
                for output in review_required
                for ref in output.get("input_refs", [])
            ),
            "sample_output_ids": self._bounded_refs(
                output.get("output_id") for output in outputs
            ),
        }

    def _summarize_steward_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        actions = [
            action
            for action in result.get("actions", [])
            if isinstance(action, Mapping)
        ]
        event = result.get("event")
        event_ref = event.get("event_id") if isinstance(event, Mapping) else None
        return {
            "status": result.get("status"),
            "graph_version": result.get("graph_version"),
            "action_count": len(actions),
            "action_counts": self._count_mapping_values(actions, "action"),
            "target_refs": self._bounded_refs(
                ref
                for action in actions
                for ref in [
                    *list(action.get("atom_refs", [])),
                    action.get("kept"),
                    action.get("archived"),
                ]
            ),
            "event_id": event_ref,
        }

    def _summarize_policy_distillation_result(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        distilled = result.get("distilled")
        distilled_atom = (
            distilled.get("atom")
            if isinstance(distilled, Mapping)
            and isinstance(distilled.get("atom"), Mapping)
            else None
        )
        return {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "candidate_count": result.get("candidate_count"),
            "min_source_atoms": result.get("min_source_atoms"),
            "source_refs": self._bounded_refs(result.get("source_refs", [])),
            "distilled_atom_ref": distilled_atom.get("id") if distilled_atom else None,
        }

    def _summarize_maintenance_distiller_result(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        committed = [
            item for item in result.get("committed", []) if isinstance(item, Mapping)
        ]
        deferred = [
            item for item in result.get("deferred", []) if isinstance(item, Mapping)
        ]
        proposals = [
            item for item in result.get("proposals", []) if isinstance(item, Mapping)
        ]
        event = result.get("event")
        event_ref = event.get("event_id") if isinstance(event, Mapping) else None
        return {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "scope": dict(result.get("scope") or {}),
            "domain": result.get("domain"),
            "graph_version": result.get("graph_version"),
            "window": dict(result.get("window") or {}),
            "processors": list(result.get("processors", [])),
            "missing_processors": list(result.get("missing_processors", [])),
            "proposal_count": len(proposals),
            "committed_count": len(committed),
            "deferred_count": len(deferred),
            "proposal_action_counts": self._count_mapping_values(proposals, "action"),
            "committed_status_counts": self._count_mapping_values(committed, "status"),
            "deferred_reason_counts": self._count_mapping_values(deferred, "reason"),
            "committed_refs": self._bounded_refs(
                item.get("atom", {}).get("id")
                if isinstance(item.get("atom"), Mapping)
                else item.get("edge", {}).get("edge_id")
                if isinstance(item.get("edge"), Mapping)
                else None
                for item in committed
            ),
            "deferred_proposal_ids": self._bounded_refs(
                item.get("proposal_id") for item in deferred
            ),
            "reviewer": dict(result.get("reviewer") or {}),
            "event_id": event_ref,
        }

    def _summarize_storage_cleanup_result(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        event = result.get("event")
        event_ref = event.get("event_id") if isinstance(event, Mapping) else None
        index_prune = dict(result.get("index_prune") or {})
        idempotency = dict(result.get("idempotency") or {})
        checkpoint = dict(result.get("checkpoint") or {})
        vacuum = dict(result.get("vacuum") or {})
        return {
            "status": result.get("status"),
            "deleted_atom_count": int(result.get("deleted_atom_count", 0) or 0),
            "deleted_atom_refs": self._bounded_refs(result.get("deleted_atom_refs", [])),
            "index_pruned_rows": int(index_prune.get("rows", 0) or 0),
            "idempotency_compacted_rows": int(idempotency.get("rows", 0) or 0),
            "idempotency_saved_bytes": int(idempotency.get("saved_bytes", 0) or 0),
            "checkpoint_status": checkpoint.get("status"),
            "checkpoint_mode": checkpoint.get("mode"),
            "vacuum_status": vacuum.get("status"),
            "vacuum_reason": vacuum.get("reason"),
            "event_id": event_ref,
        }

    def _count_mapping_values(
        self, rows: Sequence[Mapping[str, Any]], key: str
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    def _bounded_refs(self, refs: Any, *, limit: int = 24) -> list[str]:
        output: list[str] = []
        for ref in refs or []:
            if ref in (None, "", [], {}):
                continue
            text = str(ref)
            if text in output:
                continue
            output.append(text)
            if len(output) >= limit:
                break
        return output

    def _bounded_json_summary(self, value: Any, *, max_bytes: int = 2048) -> Any:
        try:
            encoded = canonical_json(value)
        except Exception:
            return {"summary": str(value)[:max_bytes], "truncated": True}
        if len(encoded.encode("utf-8")) <= max_bytes:
            return value
        return {
            "summary_digest": digest(value),
            "summary_bytes": len(encoded.encode("utf-8")),
            "truncated": True,
        }

    def _memory_policy_due(
        self,
        policy: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        graph_version = self.store.graph_version()
        last_graph_version = int(state.get("last_graph_version", 0) or 0)
        graph_delta = max(0, graph_version - last_graph_version)
        schedule = dict(policy.get("schedule", {}))
        reasons = []
        if force:
            reasons.append("force")
        if graph_delta >= int(schedule.get("every_graph_versions", 25)):
            reasons.append("graph_version_interval")
        every_seconds = int(schedule.get("every_seconds", 300) or 0)
        elapsed_seconds = self._seconds_since(state.get("last_run_at"))
        if every_seconds > 0 and elapsed_seconds is not None and elapsed_seconds >= every_seconds:
            reasons.append("time_interval")
        pressure_mode = self._capacity_pressure_mode()
        if (
            schedule.get("run_on_pressure", True)
            and pressure_mode in {"orange", "red"}
            and graph_delta > 0
        ):
            reasons.append(f"capacity_pressure:{pressure_mode}")
        storage_cleanup = self._storage_cleanup_due(
            policy.get("storage_cleanup", {}), state, force=force
        )
        if storage_cleanup["due"] and "force" not in reasons:
            reasons.append("storage_cleanup_idle")
        return {
            "due": bool(reasons),
            "reasons": reasons,
            "graph_version": graph_version,
            "last_graph_version": last_graph_version,
            "graph_delta": graph_delta,
            "elapsed_seconds": elapsed_seconds,
            "pressure_mode": pressure_mode,
            "storage_cleanup": storage_cleanup,
        }

    def _seconds_since(self, timestamp: Any) -> int | None:
        if not timestamp:
            return None
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))

    def _iso_before_seconds(self, seconds: int) -> str:
        return (
            datetime.now(timezone.utc) - timedelta(seconds=max(0, int(seconds)))
        ).isoformat().replace("+00:00", "Z")

    def _active_superseded_refs(
        self, refs: Sequence[str] | None = None
    ) -> dict[str, list[str]]:
        active_refs = self.store.active_atom_ids(lifecycle_states=["active", "proposed"])
        scoped_refs = {str(ref) for ref in refs or [] if str(ref)}
        edges = (
            self.store.list_edges_for_refs(sorted(scoped_refs))
            if refs is not None
            else self.store.list_edges()
        )
        superseded: dict[str, list[str]] = {}
        for edge in edges:
            if edge.get("relation") != "rel:supersedes":
                continue
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source in active_refs and target in active_refs:
                superseded.setdefault(target, []).append(source)
        return {ref: sorted(set(sources)) for ref, sources in superseded.items()}

    def _memory_quality_diagnostics(
        self,
        *,
        policy: Mapping[str, Any],
        indexes: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        atoms = self.store.list_atoms_filtered(lifecycle_states=["active", "proposed"])
        by_ref = {str(atom["id"]): atom for atom in atoms}
        active_count = len(atoms)
        decay = dict(policy.get("decay") or {})
        max_atoms = int(decay.get("max_atoms", 256) or 256)
        edge_degrees = self.store.edge_degree_counts()
        isolated = [
            atom for atom in atoms if int(edge_degrees.get(str(atom["id"]), 0)) == 0
        ]
        isolated_by_type: dict[str, int] = {}
        for atom in isolated:
            atom_type = str(atom.get("type") or "unknown")
            isolated_by_type[atom_type] = isolated_by_type.get(atom_type, 0) + 1

        superseded_refs = self._active_superseded_refs()
        superseded_by_type: dict[str, int] = {}
        for atom_ref in superseded_refs:
            atom = by_ref.get(atom_ref)
            atom_type = str((atom or {}).get("type") or "unknown")
            superseded_by_type[atom_type] = superseded_by_type.get(atom_type, 0) + 1

        graph_version = self.store.graph_version()
        index_lag = {
            str(index["index_name"]): max(
                0, graph_version - int(index.get("graph_version", 0) or 0)
            )
            for index in indexes
        }
        max_index_lag = max(index_lag.values(), default=0)
        pressure_eligible = [
            atom
            for atom in atoms
            if self._pressure_archive_eligible(atom, decay=decay, scope={})
        ]
        pressure_eligible_by_type: dict[str, int] = {}
        for atom in pressure_eligible:
            atom_type = str(atom.get("type") or "unknown")
            pressure_eligible_by_type[atom_type] = (
                pressure_eligible_by_type.get(atom_type, 0) + 1
            )
        archives_needed = max(0, active_count - max_atoms)
        capacity_headroom_ratio = float(
            decay.get("capacity_headroom_ratio", 0.2) or 0.0
        )
        capacity_targets = sorted(
            {
                max(1, int(item))
                for item in decay.get("capacity_assessment_targets", [256, 512, 768])
                if item not in (None, "")
            }
            | {max_atoms}
        )
        required_with_headroom = int(
            math.ceil(active_count / max(0.1, 1.0 - capacity_headroom_ratio))
        )
        recommended_target = next(
            (target for target in capacity_targets if target >= required_with_headroom),
            capacity_targets[-1],
        )
        capacity_utilization = active_count / max(1, max_atoms)
        capacity_near_limit = capacity_utilization >= 1.0 - capacity_headroom_ratio

        warnings: list[str] = []
        if active_count > max_atoms:
            warnings.append("active_atom_count_exceeds_decay_max_atoms")
            if len(pressure_eligible) < archives_needed:
                warnings.append("active_atom_pressure_not_fully_enforceable")
        if capacity_near_limit:
            warnings.append("active_atom_capacity_headroom_low")
        if superseded_refs:
            warnings.append("active_superseded_atoms_present")
        if isolated:
            warnings.append("isolated_active_atoms_present")
        maintenance_every = int(
            dict(policy.get("schedule") or {}).get("every_graph_versions", 25) or 25
        )
        if max_index_lag >= maintenance_every:
            warnings.append("derived_index_lag_exceeds_schedule")

        return {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "active_atom_count": active_count,
            "active_atom_limit": max_atoms,
            "active_atom_pressure": "over_limit"
            if active_count > max_atoms
            else "within_limit",
            "pressure_cleanup": {
                "policyless_fallback_enabled": bool(
                    decay.get("pressure_archive_policyless", True)
                ),
                "archives_needed": archives_needed,
                "eligible_policyless_count": len(pressure_eligible),
                "eligible_policyless_by_type": pressure_eligible_by_type,
                "max_archives_per_run": int(
                    decay.get("pressure_max_archives_per_run", 256) or 256
                ),
                "protected_types": list(decay.get("pressure_protected_types", [])),
            },
            "capacity_assessment": {
                "configured_target": max_atoms,
                "active_count": active_count,
                "headroom_atoms": max(0, max_atoms - active_count),
                "utilization": round(capacity_utilization, 4),
                "headroom_ratio_target": capacity_headroom_ratio,
                "near_limit": capacity_near_limit,
                "recommended_target": recommended_target,
                "candidate_targets": [
                    {
                        "target": target,
                        "headroom_atoms": target - active_count,
                        "utilization": round(active_count / max(1, target), 4),
                        "meets_headroom_target": target >= required_with_headroom,
                    }
                    for target in capacity_targets
                ],
            },
            "active_superseded_atoms": {
                "count": len(superseded_refs),
                "by_type": superseded_by_type,
                "sample_refs": sorted(superseded_refs)[:10],
            },
            "isolated_active_atoms": {
                "count": len(isolated),
                "by_type": isolated_by_type,
                "sample_refs": sorted(str(atom["id"]) for atom in isolated)[:10],
            },
            "derived_index_lag": {
                "max_graph_delta": max_index_lag,
                "by_index": index_lag,
            },
        }

    def _storage_cleanup_due(
        self,
        cleanup: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if not cleanup.get("enabled", True) and not force:
            return {"due": False, "reason": "storage_cleanup_disabled"}
        last_foreground = (
            self.store.get_meta("last_foreground_activity_at")
            or state.get("last_foreground_activity_at")
        )
        idle_elapsed = self._seconds_since(last_foreground)
        idle_after = int(cleanup.get("idle_after_seconds", 300) or 0)
        if idle_elapsed is not None and idle_elapsed < idle_after and not force:
            return {
                "due": False,
                "reason": "foreground_activity_recent",
                "idle_elapsed_seconds": idle_elapsed,
                "idle_after_seconds": idle_after,
                "last_foreground_activity_at": last_foreground,
            }
        last_cleanup = (
            self.store.get_meta("last_storage_cleanup_at")
            or state.get("last_storage_cleanup_at")
        )
        cleanup_elapsed = self._seconds_since(last_cleanup)
        min_interval = int(cleanup.get("min_interval_seconds", 900) or 0)
        if cleanup_elapsed is not None and cleanup_elapsed < min_interval and not force:
            return {
                "due": False,
                "reason": "cleanup_interval_not_elapsed",
                "elapsed_since_cleanup_seconds": cleanup_elapsed,
                "min_interval_seconds": min_interval,
                "last_storage_cleanup_at": last_cleanup,
            }
        return {
            "due": True,
            "reason": "force" if force else "idle_interval_elapsed",
            "idle_elapsed_seconds": idle_elapsed,
            "idle_after_seconds": idle_after,
            "last_foreground_activity_at": last_foreground,
            "elapsed_since_cleanup_seconds": cleanup_elapsed,
            "min_interval_seconds": min_interval,
            "last_storage_cleanup_at": last_cleanup,
        }

    def _timestamp_elapsed(self, timestamp: Any) -> bool:
        if not timestamp:
            return False
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            return False
        return datetime.now(timezone.utc) >= parsed

    def _run_storage_cleanup(
        self,
        *,
        cleanup: Mapping[str, Any],
        due: Mapping[str, Any],
        scope: Mapping[str, Any],
        actor: str,
        state: Mapping[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        now = utc_now()
        protected_types = {str(item) for item in cleanup.get("protected_types", [])}
        max_deletions = max(0, int(cleanup.get("max_deletions_per_tick", 256) or 0))
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        tombstones: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        deleted_refs: list[str] = []
        index_lifecycle_states = (
            ["archived"] if cleanup.get("remove_archived_from_hot_index", True) else []
        )
        index_health_statuses = (
            ["stale"] if cleanup.get("remove_stale_from_hot_index", True) else []
        )
        compact_after = cleanup.get("compact_idempotency_after_seconds")
        with self.store.transaction() as conn:
            index_prune = self.store.prune_atom_text_index(
                conn,
                lifecycle_states=index_lifecycle_states,
                health_statuses=index_health_statuses,
            )
            atoms = self.store.list_atoms_filtered(
                include_deleted=False,
                lifecycle_states=["active", "archived", "proposed"],
            )
            for atom in atoms:
                if len(actions) >= max_deletions:
                    break
                if not maintenance_scope_visible(atom["scope"], scope):
                    continue
                if atom["type"] in protected_types:
                    continue
                reason = self._storage_deletion_reason(atom, cleanup)
                if reason is None:
                    continue
                updated = dict(atom)
                updated["lifecycle_state"] = "deleted"
                updated["health_status"] = "deleted"
                updated["deleted"] = 1
                updated["version"] = int(atom["version"]) + 1
                updated["updated_at"] = now
                updated["revision_history"] = list(updated["revision_history"])
                updated["revision_history"].append(
                    {
                        "version": atom["version"],
                        "digest": digest(self._atom_projection(atom)),
                        "changed_at": now,
                        "actor": actor,
                        "reason": reason,
                    }
                )
                updated = normalize_atom(
                    self._attach_search_index(updated), require_id=True
                )
                updated["deleted"] = 1
                tombstone = self.store.insert_tombstone(
                    conn,
                    target_ref=atom["id"],
                    content_digest=self._memory_identity_digest(atom),
                    recreation_policy="block_recreate",
                    reason=reason,
                )
                deleted_edges = self.store.mark_edges_deleted_for_ref(conn, atom["id"])
                self.store.replace_atom(conn, updated)
                projected_atoms.append(updated)
                projected_edges.extend(deleted_edges)
                tombstones.append(tombstone)
                deleted_refs.append(atom["id"])
                actions.append(
                    {
                        "atom_ref": atom["id"],
                        "action": "delete",
                        "reason": reason,
                        "lifecycle_state_before": atom["lifecycle_state"],
                        "health_status_before": atom["health_status"],
                    }
                )
            if compact_after is None:
                idempotency = {
                    "status": "skipped",
                    "reason": "idempotency_compaction_disabled",
                    "rows": 0,
                }
            else:
                idempotency = self.store.compact_idempotency_responses(
                    conn,
                    older_than=self._iso_before_seconds(int(compact_after)),
                    max_rows=int(
                        cleanup.get("max_idempotency_compactions_per_tick", 512) or 0
                    ),
                )
            if actions or index_prune.get("rows") or idempotency.get("rows"):
                event = self.store.append_event(
                    conn,
                    event_type="storage_cleanup_run",
                    actor=actor,
                    payload={
                        "operation": "run_storage_cleanup",
                        "policy": dict(cleanup),
                        "due": dict(due),
                        "actions": actions,
                        "index_prune": index_prune,
                        "idempotency": idempotency,
                        "projected_atoms": projected_atoms,
                        "projected_edges": projected_edges,
                        "tombstones": tombstones,
                    },
                    target_refs=deleted_refs,
                )
                self.store.clear_packet_cache(conn)
            else:
                event = None
            self.store._set_meta(conn, "last_storage_cleanup_at", now)
        sqlite_compaction = dict(cleanup.get("sqlite_compaction") or {})
        checkpoint = {"status": "skipped", "reason": "checkpoint_disabled"}
        if sqlite_compaction.get("checkpoint_wal", True):
            try:
                checkpoint = self.store.checkpoint_wal(
                    mode=str(sqlite_compaction.get("checkpoint_mode") or "TRUNCATE")
                )
            except Exception as exc:
                checkpoint = {"status": "error", "error": str(exc)}
        vacuum = self._maybe_vacuum_sqlite(
            sqlite_compaction=sqlite_compaction,
            state=state,
            force=force,
        )
        checkpoint_after_vacuum = {"status": "skipped", "reason": "vacuum_not_completed"}
        if (
            sqlite_compaction.get("checkpoint_wal", True)
            and vacuum.get("status") == "completed"
        ):
            try:
                checkpoint_after_vacuum = self.store.checkpoint_wal(
                    mode=str(sqlite_compaction.get("checkpoint_mode") or "TRUNCATE")
                )
            except Exception as exc:
                checkpoint_after_vacuum = {"status": "error", "error": str(exc)}
        return {
            "status": "completed",
            "due": dict(due),
            "index_prune": index_prune,
            "deleted_atom_count": len(actions),
            "deleted_atom_refs": deleted_refs,
            "idempotency": idempotency,
            "checkpoint": checkpoint,
            "vacuum": vacuum,
            "checkpoint_after_vacuum": checkpoint_after_vacuum,
            "event": event,
        }

    def _storage_deletion_reason(
        self, atom: Mapping[str, Any], cleanup: Mapping[str, Any]
    ) -> str | None:
        if atom.get("deleted"):
            return None
        updated_age = self._seconds_since(
            atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
        )
        archived_after = cleanup.get("delete_archived_after_seconds")
        if (
            archived_after is not None
            and atom.get("lifecycle_state") == "archived"
            and updated_age is not None
            and updated_age >= int(archived_after)
        ):
            return "storage_cleanup_archived_retention_elapsed"
        stale_after = cleanup.get("delete_stale_after_seconds")
        if (
            stale_after is not None
            and atom.get("health_status") == "stale"
            and updated_age is not None
            and updated_age >= int(stale_after)
        ):
            return "storage_cleanup_stale_retention_elapsed"
        return None

    def _maybe_vacuum_sqlite(
        self,
        *,
        sqlite_compaction: Mapping[str, Any],
        state: Mapping[str, Any],
        force: bool,
    ) -> dict[str, Any]:
        if not sqlite_compaction.get("vacuum_enabled", True):
            return {"status": "skipped", "reason": "vacuum_disabled"}
        idle_after = int(sqlite_compaction.get("vacuum_idle_after_seconds", 1800) or 0)
        last_foreground = (
            self.store.get_meta("last_foreground_activity_at")
            or state.get("last_foreground_activity_at")
        )
        idle_elapsed = self._seconds_since(last_foreground)
        if idle_elapsed is not None and idle_elapsed < idle_after and not force:
            return {
                "status": "skipped",
                "reason": "foreground_activity_recent",
                "idle_elapsed_seconds": idle_elapsed,
                "idle_after_seconds": idle_after,
            }
        min_interval = int(sqlite_compaction.get("vacuum_min_interval_seconds", 86400) or 0)
        last_vacuum = self.store.get_meta("last_vacuum_at") or state.get("last_vacuum_at")
        vacuum_elapsed = self._seconds_since(last_vacuum)
        if vacuum_elapsed is not None and vacuum_elapsed < min_interval and not force:
            return {
                "status": "skipped",
                "reason": "vacuum_interval_not_elapsed",
                "elapsed_since_vacuum_seconds": vacuum_elapsed,
                "min_interval_seconds": min_interval,
                "last_vacuum_at": last_vacuum,
            }
        try:
            result = self.store.vacuum()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        completed_at = utc_now()
        self.store.set_meta("last_vacuum_at", completed_at)
        return {**result, "completed_at": completed_at}

    def _run_decay_policy(
        self,
        *,
        decay: Mapping[str, Any],
        scope: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        max_atoms = max(1, int(decay.get("max_atoms", 256) or 256))
        require_atom_policy = bool(decay.get("require_atom_policy", True))
        actions: list[dict[str, Any]] = []
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        now = utc_now()
        superseded_refs = (
            self._active_superseded_refs()
            if decay.get("archive_superseded", True)
            else {}
        )
        atoms_by_ref = {
            atom["id"]: atom
            for atom in self.store.list_atoms_filtered(
                lifecycle_states=["active", "proposed"]
            )
        }
        for atom in self.store.list_atoms_filtered(
            lifecycle_states=["active", "proposed"],
            atom_ids=sorted(superseded_refs),
        ):
            atoms_by_ref[atom["id"]] = atom
        atoms = list(atoms_by_ref.values())
        planned: list[tuple[dict[str, Any], dict[str, Any]]] = []
        planned_archives: set[str] = set()
        for atom in atoms:
            if not maintenance_scope_visible(atom["scope"], scope):
                continue
            atom_policy = (
                dict(atom.get("decay_policy") or {})
                if isinstance(atom.get("decay_policy"), Mapping)
                else {}
            )
            explicit_atom_policy = self._has_explicit_atom_decay_policy(atom_policy)
            if atom_policy.get("enabled") is False:
                continue
            superseded_action = self._decay_action_for_superseded_atom(
                atom,
                superseded_by=superseded_refs.get(atom["id"], []),
                policy=decay,
            )
            if superseded_action is not None:
                action = superseded_action
            else:
                if require_atom_policy and not explicit_atom_policy:
                    continue
                if self._timestamp_elapsed(atom_policy.get("retain_until")):
                    pass
                elif atom_policy.get("retain_until"):
                    continue
                action = self._decay_action_for_atom(
                    atom,
                    atom_policy=atom_policy,
                    policy=decay,
                )
            if action is None:
                continue
            planned.append((atom, action))
            if action["action"] == "archive":
                planned_archives.add(str(atom["id"]))

        hot_count_before = sum(
            1 for atom in atoms if atom.get("lifecycle_state") in {"active", "proposed"}
        )
        hot_count_after_rules = hot_count_before - len(planned_archives)
        pressure_needed = max(0, hot_count_after_rules - max_atoms)
        pressure_limit = int(decay.get("pressure_max_archives_per_run", 256) or 256)
        pressure_candidates = [
            atom
            for atom in atoms
            if str(atom["id"]) not in planned_archives
            and self._pressure_archive_eligible(atom, decay=decay, scope=scope)
        ]
        edge_degrees = self.store.edge_degree_counts() if pressure_needed else {}
        pressure_candidates.sort(
            key=lambda atom: self._pressure_archive_sort_key(atom, edge_degrees)
        )
        pressure_archive_count = 0
        if decay.get("pressure_archive_policyless", True) and pressure_needed:
            for atom in pressure_candidates[: min(pressure_needed, pressure_limit)]:
                action = {
                    "action": "archive",
                    "reason": "active_atom_pressure_policyless_fallback",
                    "health_status": "stale",
                }
                planned.append((atom, action))
                planned_archives.add(str(atom["id"]))
                pressure_archive_count += 1

        pressure = {
            "enabled": bool(decay.get("pressure_archive_policyless", True)),
            "triggered": pressure_needed > 0,
            "max_atoms": max_atoms,
            "hot_count_before": hot_count_before,
            "hot_count_after_rules": hot_count_after_rules,
            "eligible_policyless_count": len(pressure_candidates),
            "archive_limit": pressure_limit,
            "archive_count": pressure_archive_count,
            "remaining_hot_count": hot_count_after_rules - pressure_archive_count,
            "remaining_over_limit": max(
                0,
                hot_count_after_rules - pressure_archive_count - max_atoms,
            ),
        }

        with self.store.transaction() as conn:
            for atom, action in planned:
                atom_policy = (
                    dict(atom.get("decay_policy") or {})
                    if isinstance(atom.get("decay_policy"), Mapping)
                    else {}
                )
                changed = dict(atom)
                changed["version"] = int(changed["version"]) + 1
                changed["updated_at"] = now
                if action["action"] == "archive":
                    changed["lifecycle_state"] = "archived"
                    changed["health_status"] = action.get("health_status", "stale")
                    projected_edges.extend(
                        self.store.mark_edges_deleted_for_ref(conn, str(atom["id"]))
                    )
                elif action["action"] == "mark_stale":
                    changed["health_status"] = "stale"
                elif action["action"] == "mark_low_utility":
                    changed["health_status"] = "low_utility"
                changed["decay_policy"] = {
                    **atom_policy,
                    "last_decay": {
                        "action": action["action"],
                        "reason": action["reason"],
                        "applied_at": now,
                    },
                }
                changed = normalize_atom(
                    self._attach_search_index(changed), require_id=True
                )
                self.store.replace_atom(conn, changed)
                projected_atoms.append(changed)
                actions.append(
                    {
                        "atom_ref": changed["id"],
                        "action": action["action"],
                        "reason": action["reason"],
                        **(
                            {"superseded_by": action["superseded_by"]}
                            if action.get("superseded_by")
                            else {}
                        ),
                        "health_status": changed["health_status"],
                        "lifecycle_state": changed["lifecycle_state"],
                    }
                )
            if actions:
                event = self.store.append_event(
                    conn,
                    event_type="decay_policy_applied",
                    actor=actor,
                    payload={
                        "operation": "run_decay_policy",
                        "policy": dict(decay),
                        "actions": actions,
                        "projected_atoms": projected_atoms,
                        "projected_edges": projected_edges,
                    },
                    target_refs=[action["atom_ref"] for action in actions],
                )
                self.store.clear_packet_cache(conn)
            else:
                event = None
        return {
            "status": "completed",
            "action_count": len(actions),
            "actions": actions,
            "projected_edges": projected_edges,
            "pressure": pressure,
            "event": event,
        }

    def _has_explicit_atom_decay_policy(self, atom_policy: Mapping[str, Any]) -> bool:
        return any(
            atom_policy.get(key) not in (None, "", [], {})
            for key in {
                "archive_after_seconds",
                "expires_at",
                "low_utility_threshold",
                "mark_stale_after_seconds",
                "retain_until",
            }
        )

    def _pressure_archive_eligible(
        self,
        atom: Mapping[str, Any],
        *,
        decay: Mapping[str, Any],
        scope: Mapping[str, Any],
    ) -> bool:
        if not decay.get("pressure_archive_policyless", True):
            return False
        if atom.get("lifecycle_state") != "active":
            return False
        if not maintenance_scope_visible(atom.get("scope", {}), scope):
            return False
        if str(atom.get("type") or "") in set(
            decay.get("pressure_protected_types", [])
        ):
            return False
        atom_policy = (
            dict(atom.get("decay_policy") or {})
            if isinstance(atom.get("decay_policy"), Mapping)
            else {}
        )
        if atom_policy.get("enabled") is False:
            return False
        if self._has_explicit_atom_decay_policy(atom_policy):
            return False
        retain_until = atom_policy.get("retain_until")
        if retain_until and not self._timestamp_elapsed(retain_until):
            return False
        return True

    def _pressure_archive_sort_key(
        self,
        atom: Mapping[str, Any],
        edge_degrees: Mapping[str, int],
    ) -> tuple[Any, ...]:
        health_rank = {
            "low_utility": 0,
            "orphaned": 0,
            "stale": 0,
            "confounding": 1,
            "contradicted": 1,
            "healthy": 2,
        }
        atom_ref = str(atom.get("id") or "")
        timestamp = str(
            atom.get("last_accessed")
            or atom.get("updated_at")
            or atom.get("observed_at")
            or atom.get("created_at")
            or ""
        )
        return (
            1 if int(edge_degrees.get(atom_ref, 0) or 0) > 0 else 0,
            health_rank.get(str(atom.get("health_status") or ""), 1),
            float(atom.get("utility", 0.0) or 0.0),
            float(atom.get("salience", 0.0) or 0.0),
            timestamp,
            atom_ref,
        )

    def _decay_action_for_superseded_atom(
        self,
        atom: Mapping[str, Any],
        *,
        superseded_by: Sequence[str],
        policy: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not superseded_by:
            return None
        after = policy.get("archive_superseded_after_seconds", 0)
        if after not in (None, ""):
            age = self._seconds_since(
                atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
            )
            if age is not None and age < int(after):
                return None
        return {
            "action": "archive",
            "reason": "superseded_by_active_atom",
            "health_status": "stale",
            "superseded_by": list(superseded_by),
        }

    def _decay_action_for_atom(
        self,
        atom: Mapping[str, Any],
        *,
        atom_policy: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> dict[str, str] | None:
        if self._timestamp_elapsed(atom_policy.get("expires_at")):
            return {"action": "archive", "reason": "expires_at_elapsed"}
        low_utility_threshold = atom_policy.get(
            "low_utility_threshold", policy.get("low_utility_threshold")
        )
        if low_utility_threshold not in (None, ""):
            try:
                threshold = max(0.0, min(1.0, float(low_utility_threshold)))
            except (TypeError, ValueError):
                threshold = None
            if threshold is not None and float(atom["utility"]) < threshold:
                return {
                    "action": "mark_low_utility",
                    "reason": "utility_below_threshold",
                }
        archive_after = atom_policy.get(
            "archive_after_seconds", policy.get("archive_after_seconds")
        )
        if archive_after not in (None, ""):
            age = self._seconds_since(
                atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
            )
            if age is not None and age >= int(archive_after):
                return {"action": "archive", "reason": "archive_after_elapsed"}
        stale_after = atom_policy.get(
            "mark_stale_after_seconds", policy.get("mark_stale_after_seconds")
        )
        if stale_after not in (None, "") and atom.get("health_status") == "healthy":
            age = self._seconds_since(
                atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
            )
            if age is not None and age >= int(stale_after):
                return {"action": "mark_stale", "reason": "stale_after_elapsed"}
        return None

    def _run_policy_distillation(
        self,
        *,
        policy: Mapping[str, Any],
        scope: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        distillation = dict(policy["distillation"])
        candidates = self._policy_distillation_candidates(
            policy=policy, scope=scope
        )
        min_sources = int(distillation["min_source_atoms"])
        if len(candidates) < min_sources:
            return {
                "status": "skipped",
                "reason": "insufficient_candidates",
                "candidate_count": len(candidates),
                "min_source_atoms": min_sources,
            }
        max_sources = int(distillation["max_source_atoms"])
        selected = candidates[:max_sources]
        target_refs = [atom["id"] for atom in selected]
        source_digests = [digest(self._atom_projection(atom)) for atom in selected]
        summary = self._policy_distillation_summary(selected)
        idempotency_key = stable_id(
            "policy_distill",
            {
                "distillation_type": distillation["distillation_type"],
                "target_refs": target_refs,
                "source_digests": source_digests,
                "summary_digest": digest(summary),
                "scope": scope,
            },
        )
        try:
            distilled = self.distill_memories(
                target_refs=target_refs,
                summary=summary,
                scope=scope,
                actor=actor,
                idempotency_key=idempotency_key,
                distillation_type=distillation["distillation_type"],
                archive_sources=distillation["archive_sources"],
                approved_by=distillation["approved_by"],
            )
        except ValidationError as exc:
            if "distilled atom already exists" in str(exc):
                return {
                    "status": "skipped",
                    "reason": "already_distilled",
                    "source_refs": target_refs,
                }
            raise
        return {
            "status": "completed"
            if distilled.get("status") == "distilled"
            else distilled.get("status", "completed"),
            "selected_source_count": len(selected),
            "source_refs": target_refs,
            "distilled": distilled if distilled.get("status") == "distilled" else None,
            "result": distilled,
        }

    def _policy_distillation_candidates(
        self,
        *,
        policy: Mapping[str, Any],
        scope: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        distillation = dict(policy["distillation"])
        candidate_types = set(distillation["candidate_types"])
        covered_sources: set[str] = set()
        for atom in self.store.list_atoms_filtered(
            types=["semantic"],
            lifecycle_states=["active"],
        ):
            if atom.get("deleted") or atom["type"] != "semantic":
                continue
            if atom.get("lifecycle_state") != "active":
                continue
            payload = atom.get("payload", {})
            if payload.get("created_by") != "svc:memory_policy":
                continue
            if payload.get("distillation_type") != distillation["distillation_type"]:
                continue
            covered_sources.update(str(ref) for ref in payload.get("source_refs", []))
        candidates = []
        for atom in self.store.list_atoms_filtered(
            types=sorted(candidate_types) if candidate_types else None,
            lifecycle_states=["active"],
            included_health=["healthy", "low_utility"],
        ):
            if atom.get("deleted"):
                continue
            if atom["id"] in covered_sources:
                continue
            if candidate_types and atom["type"] not in candidate_types:
                continue
            if atom["lifecycle_state"] != "active":
                continue
            if atom["health_status"] not in {"healthy", "low_utility"}:
                continue
            if scope and not scope_visible(atom["scope"], scope):
                continue
            candidates.append(atom)
        candidates.sort(
            key=lambda atom: (
                -self._policy_distillation_priority(atom),
                atom.get("observed_at") or atom["created_at"],
                atom["id"],
            )
        )
        return candidates

    def _policy_distillation_priority(self, atom: Mapping[str, Any]) -> int:
        payload = atom.get("payload", {})
        payload = payload if isinstance(payload, Mapping) else {}
        score = 0
        kind = str(payload.get("qandl_kind") or payload.get("kind") or "").lower()
        outcome = str(
            payload.get("outcome") or payload.get("status") or payload.get("result") or ""
        ).lower()
        if kind in {"reflection", "outcome", "evaluation"}:
            score += 6
        if outcome and outcome not in {"issued", "pending", "planned", "started"}:
            score += 4
        for key in (
            "directive_atom_ref",
            "source_directive_ref",
            "metric_deltas",
            "deltas",
            "lesson",
            "correction",
        ):
            if payload.get(key) not in (None, "", [], {}):
                score += 2
        if self._payload_delta_fields(payload):
            score += 2
        if payload.get("summary") or payload.get("claim"):
            score += 1
        if payload.get("applied_controls") or payload.get("requested_controls"):
            score += 1
        return score

    def _policy_distillation_summary(
        self, atoms: Sequence[Mapping[str, Any]]
    ) -> str:
        type_counts = self._counts(atoms, "type")
        type_phrase = ", ".join(
            f"{count} {atom_type}" for atom_type, count in sorted(type_counts.items())
        )
        highlights = [self._policy_distillation_highlight(atom) for atom in atoms[:6]]
        highlights = [highlight for highlight in highlights if highlight]
        if highlights:
            source_phrase = " Key memories: " + "; ".join(highlights) + "."
        else:
            source_phrase = ""
        return (
            "Automatic AMOS memory policy distilled "
            f"{len(atoms)} source atoms"
            f" ({type_phrase or 'mixed types'}) into a reusable memory packet."
            f"{source_phrase}"
        )

    def _policy_distillation_highlight(self, atom: Mapping[str, Any]) -> str:
        payload = atom.get("payload", {})
        atom_id = str(atom.get("id", "unknown"))
        atom_type = str(atom.get("type", "memory"))
        if not isinstance(payload, Mapping):
            return self._truncate_text(f"{atom_id}: {payload}", 180)
        if payload.get("summary"):
            return self._truncate_text(f"{atom_id}: {payload['summary']}", 180)
        if payload.get("claim"):
            return self._truncate_text(f"{atom_id}: {payload['claim']}", 180)
        chunk = payload.get("chunk", payload.get("target_chunk"))
        outcome = (
            payload.get("outcome") or payload.get("status") or payload.get("result")
        )
        deltas = self._payload_delta_fields(payload)
        controls = payload.get("applied_controls") or payload.get("requested_controls")
        prefix = f"{atom_id}"
        if chunk is not None:
            prefix += f" chunk {chunk}"
        if outcome:
            prefix += f" {outcome}"
        if controls:
            controls_text = canonical_json(controls)
            detail = f"controls {controls_text}"
            if deltas:
                detail = f"deltas {self._format_delta_fields(deltas)}; {detail}"
            return self._truncate_text(f"{prefix}: {detail}", 220)
        if deltas:
            return self._truncate_text(
                f"{prefix}: deltas {self._format_delta_fields(deltas)}",
                220,
            )
        task = payload.get("task")
        action = payload.get("action")
        if task or action or outcome:
            parts = [str(part) for part in (task, action, outcome) if part]
            return self._truncate_text(f"{atom_id}: {'; '.join(parts)}", 180)
        rendered = self._render_atom(atom)["text"]
        return self._truncate_text(f"{atom_id} {atom_type}: {rendered}", 180)

    def _payload_delta_fields(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        deltas: dict[str, Any] = {}
        for key in ("metric_deltas", "deltas"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                deltas.update(
                    {
                        str(delta_key): delta_value
                        for delta_key, delta_value in value.items()
                        if delta_value not in (None, "", [], {})
                    }
                )
        for key, value in payload.items():
            if str(key).startswith("delta_") and value not in (None, "", [], {}):
                deltas[str(key)] = value
        return deltas

    def _format_delta_fields(self, deltas: Mapping[str, Any]) -> str:
        formatted = []
        for key, value in sorted(deltas.items()):
            if isinstance(value, (int, float)):
                formatted.append(f"{key}={value:+.6g}")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)

    def _truncate_text(self, text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _rebuild_derived_indexes(
        self, *, graph_version: int | None = None
    ) -> dict[str, Any]:
        graph_version = (
            graph_version if graph_version is not None else self.store.graph_version()
        )
        policy = self.memory_policy()
        maintenance = policy.get("maintenance", {})
        with self.store.transaction() as conn:
            for atom in self.store.list_atoms_filtered(include_deleted=True):
                self.store.replace_atom_text_index(conn, atom)
            cleanup = policy.get("storage_cleanup", {})
            pruned_index = {"status": "skipped", "reason": "storage_cleanup_disabled"}
            if cleanup.get("enabled", True):
                pruned_index = self.store.prune_atom_text_index(
                    conn,
                    lifecycle_states=["archived"]
                    if cleanup.get("remove_archived_from_hot_index", True)
                    else [],
                    health_statuses=["stale"]
                    if cleanup.get("remove_stale_from_hot_index", True)
                    else [],
                )
            lsa = self._build_lsa_token_vectors(
                graph_version=graph_version,
                enabled=bool(maintenance.get("rebuild_lsa", True)),
                dimensions=int(maintenance.get("lsa_dimensions", 32) or 0),
                max_terms=int(maintenance.get("lsa_max_terms", 300) or 300),
            )
            latent_store = self.store.replace_token_latent_vectors(
                conn,
                graph_version=graph_version,
                dimensions=int(lsa.get("dimensions", 0) or 0),
                vectors=lsa.get("vectors", {})
                if isinstance(lsa.get("vectors"), Mapping)
                else {},
            )
            self._sync_smp_vector_model(graph_version=graph_version, force=True)
            lexical = self.store.upsert_derived_index_metadata(
                conn,
                index_name="semantic_lexical_vectors",
                graph_version=graph_version,
                freshness="fresh",
                details={
                    "atom_count": self.store.atom_count(),
                    "token_count": self.store.atom_text_index_count(),
                    "processor_id": self.smp.processor_id,
                    "processor_version": self.smp.processor_version,
                    "vector_model": self.smp.vector_model_info(),
                    "rebuildable_from_canonical": True,
                    "maintained_by": "memory_policy",
                    "hot_index_prune": pruned_index,
                },
            )
            lsa_index = self.store.upsert_derived_index_metadata(
                conn,
                index_name="semantic_lsa_vectors",
                graph_version=graph_version,
                freshness=lsa.get("freshness", "fresh"),
                details={
                    key: value
                    for key, value in lsa.items()
                    if key != "vectors"
                }
                | {
                    "stored_vectors": latent_store,
                    "rebuildable_from_canonical": True,
                    "maintained_by": "memory_policy",
                },
            )
            graph = self.store.upsert_derived_index_metadata(
                conn,
                index_name="graph_adjacency",
                graph_version=graph_version,
                freshness="fresh",
                details={
                    "edge_count": self.store.edge_count(),
                    "rebuildable_from_canonical": True,
                    "maintained_by": "memory_policy",
                },
            )
        return {
            "status": "rebuilt",
            "graph_version": graph_version,
            "indexes": [lexical, lsa_index, graph],
        }

    def _build_lsa_token_vectors(
        self,
        *,
        graph_version: int,
        enabled: bool,
        dimensions: int,
        max_terms: int,
    ) -> dict[str, Any]:
        if not enabled or dimensions <= 0:
            return {
                "status": "skipped",
                "freshness": "skipped",
                "reason": "lsa_disabled",
                "dimensions": 0,
                "vectors": {},
            }
        rows = self.store.token_atom_index_rows(max_terms=max_terms)
        if not rows:
            return {
                "status": "skipped",
                "freshness": "empty",
                "reason": "no_token_index_rows",
                "dimensions": 0,
                "vectors": {},
            }
        doc_terms: dict[str, set[str]] = defaultdict(set)
        token_docs: dict[str, set[str]] = defaultdict(set)
        for atom_id, token in rows:
            doc_terms[atom_id].add(token)
            token_docs[token].add(atom_id)
        terms = sorted(token_docs, key=lambda token: (-len(token_docs[token]), token))
        terms = terms[: max(0, int(max_terms))]
        if len(terms) < 2 or len(doc_terms) < 2:
            return {
                "status": "skipped",
                "freshness": "insufficient_data",
                "reason": "insufficient_terms_or_documents",
                "term_count": len(terms),
                "document_count": len(doc_terms),
                "dimensions": 0,
                "vectors": {},
            }
        term_index = {token: index for index, token in enumerate(terms)}
        n_terms = len(terms)
        n_docs = len(doc_terms)
        idf = {
            token: math.log((1.0 + n_docs) / (1.0 + len(token_docs[token]))) + 1.0
            for token in terms
        }
        matrix = [[0.0] * n_terms for _ in range(n_terms)]
        for tokens_in_doc in doc_terms.values():
            indexed = [
                (term_index[token], idf[token])
                for token in sorted(tokens_in_doc)
                if token in term_index
            ]
            for left_pos, (left, left_weight) in enumerate(indexed):
                matrix[left][left] += left_weight * left_weight
                for right, right_weight in indexed[left_pos + 1 :]:
                    value = left_weight * right_weight
                    matrix[left][right] += value
                    matrix[right][left] += value
        components = _top_symmetric_components(
            matrix,
            count=min(max(0, int(dimensions)), n_terms),
            labels=terms,
        )
        if not components:
            return {
                "status": "skipped",
                "freshness": "insufficient_signal",
                "reason": "no_positive_components",
                "term_count": n_terms,
                "document_count": n_docs,
                "dimensions": 0,
                "vectors": {},
            }
        vectors: dict[str, list[float]] = {}
        for term_offset, token in enumerate(terms):
            coords = [
                component[1][term_offset] * math.sqrt(max(component[0], 0.0))
                for component in components
            ]
            norm = math.sqrt(sum(value * value for value in coords))
            if norm <= 0.0:
                continue
            vectors[token] = [round(value / norm, 8) for value in coords]
        return {
            "status": "rebuilt",
            "freshness": "fresh",
            "graph_version": graph_version,
            "term_count": n_terms,
            "document_count": n_docs,
            "dimensions": len(components),
            "max_terms": max_terms,
            "vectors": vectors,
            "component_eigenvalues": [
                round(component[0], 8) for component in components
            ],
        }

    def _indexed_retrieval_candidates(
        self,
        *,
        cue_tokens: set[str],
        attention_policy: Mapping[str, Any],
    ) -> list[str] | None:
        tokens = set(cue_tokens)
        tokens.update(str(token) for token in attention_policy.get("focus_terms", []) or [])
        tokens.update(str(token) for token in attention_policy.get("suppress_terms", []) or [])
        normalized = sorted(
            token
            for token in {token.strip().lower() for token in tokens if token.strip()}
            if len(token) > 1
        )
        if not normalized or self.store.atom_text_index_count() == 0:
            return None
        direct = self.store.candidate_atom_ids_for_tokens(normalized, limit=512)
        if not direct:
            return None
        candidates = set(direct)
        candidates.update(self.store.neighbor_atom_ids(direct))
        return sorted(candidates)

    def _invalidate_packet_cache(
        self, *, graph_version: int | None = None
    ) -> dict[str, Any]:
        with self.store.transaction() as conn:
            self.store.clear_packet_cache(conn)
        return {
            "status": "invalidated",
            "graph_version": (
                graph_version
                if graph_version is not None
                else self.store.graph_version()
            ),
        }

    def _attention_policy(
        self, attention_context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        context = self._normalize_attention_context(attention_context)
        return {
            "policy_id": ATTENTION_POLICY_ID,
            "context": context,
            "focus_terms": context.get("focus_terms", []),
            "suppress_terms": context.get("suppress_terms", []),
            "boost_memory_types": context.get("boost_memory_types", []),
            "suppress_memory_types": context.get("suppress_memory_types", []),
            "counterevidence_required": bool(
                context.get("counterevidence_required", False)
            ),
            "weight_adjustments": {
                "attention_focus": RETRIEVAL_WEIGHTS["attention_focus"],
                "attention_type_boost": RETRIEVAL_WEIGHTS["attention_type_boost"],
                "attention_counterevidence": RETRIEVAL_WEIGHTS[
                    "attention_counterevidence"
                ],
                "attention_novelty": RETRIEVAL_WEIGHTS["attention_novelty"],
                "attention_suppression_penalty": RETRIEVAL_WEIGHTS[
                    "attention_suppression_penalty"
                ],
            },
        }

    def _normalize_attention_context(
        self, attention_context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if not isinstance(attention_context, Mapping):
            return {}
        context: dict[str, Any] = {}
        scalar_keys = (
            "active_task",
            "mission",
            "goal",
            "role",
            "risk_posture",
            "time_horizon",
        )
        for key in scalar_keys:
            value = attention_context.get(key)
            if value not in (None, "", [], {}):
                context[key] = value

        focus_terms = self._attention_terms(attention_context.get("focus_terms"))
        suppress_terms = self._attention_terms(attention_context.get("suppress_terms"))
        for key in ("active_task", "mission", "goal", "role", "task_context"):
            focus_terms.extend(self._attention_terms(attention_context.get(key)))

        context["focus_terms"] = sorted(set(focus_terms))
        context["suppress_terms"] = sorted(set(suppress_terms))
        context["boost_memory_types"] = sorted(
            set(self._attention_type_terms(attention_context.get("boost_memory_types")))
        )
        context["suppress_memory_types"] = sorted(
            set(
                self._attention_type_terms(
                    attention_context.get("suppress_memory_types")
                )
            )
        )
        risk_posture = str(context.get("risk_posture", "")).lower()
        context["counterevidence_required"] = bool(
            attention_context.get("counterevidence_required", False)
            or risk_posture in {"cautious", "high_risk", "high-risk", "critical"}
        )
        novelty = attention_context.get("novelty_preference")
        if novelty not in (None, ""):
            try:
                context["novelty_preference"] = max(0.0, min(1.0, float(novelty)))
            except (TypeError, ValueError):
                pass
        return {
            key: value
            for key, value in context.items()
            if value not in (None, "", [], {})
        }

    def _attention_terms(self, value: Any) -> list[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, Mapping):
            terms: list[str] = []
            for item in value.values():
                terms.extend(self._attention_terms(item))
            return terms
        if isinstance(value, (list, tuple, set)):
            terms = []
            for item in value:
                terms.extend(self._attention_terms(item))
            return terms
        text = str(value).lower()
        return [token for token in re.findall(r"[a-z0-9_]+", text) if token]

    def _attention_type_terms(self, value: Any) -> list[str]:
        known_types = {
            "belief",
            "preference",
            "goal",
            "commitment",
            "procedure",
            "capability",
            "limitation",
            "episode",
            "agentic_trace",
            "action_outcome",
            "self_model",
            "runtime_state",
            "self_assessment",
            "semantic",
            "policy",
        }
        return [
            token
            for token in self._attention_terms(value)
            if token in known_types
        ]

    def _attention_score_components(
        self,
        atom: Mapping[str, Any],
        *,
        text: str,
        text_tokens: set[str],
        edge_degree: int,
        attention_policy: Mapping[str, Any] | None,
        superseded_refs: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, float]:
        policy = attention_policy if isinstance(attention_policy, Mapping) else {}
        context = policy.get("context", {}) if isinstance(policy.get("context", {}), Mapping) else {}
        focus_terms = set(policy.get("focus_terms", []) or [])
        suppress_terms = set(policy.get("suppress_terms", []) or [])
        atom_type = str(atom.get("type", ""))
        focus_overlap = len(focus_terms.intersection(text_tokens))
        suppress_overlap = len(suppress_terms.intersection(text_tokens))
        direct_focus = any(term and term in text for term in focus_terms)
        direct_suppress = any(term and term in text for term in suppress_terms)
        attention_focus = 0.0
        if focus_terms:
            attention_focus = min(1.0, focus_overlap / max(1, len(focus_terms)))
            if direct_focus:
                attention_focus = max(attention_focus, 0.75)
        attention_suppression = 0.0
        if suppress_terms:
            attention_suppression = min(
                1.0, suppress_overlap / max(1, len(suppress_terms))
            )
            if direct_suppress:
                attention_suppression = max(attention_suppression, 0.75)
        attention_type_boost = (
            1.0 if atom_type in set(policy.get("boost_memory_types", []) or []) else 0.0
        )
        if atom_type in set(policy.get("suppress_memory_types", []) or []):
            attention_suppression = max(attention_suppression, 1.0)
        try:
            novelty_preference = max(
                0.0, min(1.0, float(context.get("novelty_preference", 0.0) or 0.0))
            )
        except (TypeError, ValueError):
            novelty_preference = 0.0
        novelty = 0.0
        if novelty_preference:
            graph_familiarity = min(1.0, max(0, int(edge_degree)) / 5.0)
            novelty = novelty_preference * (1.0 - graph_familiarity)
        counterevidence = 0.0
        if policy.get("counterevidence_required"):
            if atom.get("health_status") == "contradicted":
                counterevidence = 1.0
            elif atom_type in {"limitation", "self_assessment", "action_outcome"}:
                counterevidence = 0.6
            elif any(
                term in text_tokens
                for term in {
                    "failure",
                    "correction",
                    "blocked",
                    "risk",
                    "contradiction",
                }
            ):
                counterevidence = 0.5
        return {
            "attention_focus": attention_focus,
            "attention_type_boost": attention_type_boost,
            "attention_counterevidence": counterevidence,
            "attention_novelty": novelty,
            "attention_suppression_penalty": attention_suppression,
        }

    def _attention_trace(
        self,
        *,
        attention_policy: Mapping[str, Any],
        items: Sequence[Mapping[str, Any]],
        candidates: Sequence[tuple[float, Mapping[str, Any]]],
        omissions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        selected = {str(item.get("atom_ref")) for item in items if item.get("atom_ref")}
        inhibited = []
        for _, atom in candidates:
            atom_ref = str(atom.get("id", ""))
            if not atom_ref or atom_ref in selected:
                continue
            components = atom.get("_score_components", {})
            if float(components.get("attention_suppression_penalty", 0.0) or 0.0) > 0:
                inhibited.append(atom_ref)
        omitted_reasons: dict[str, int] = {}
        for omission in omissions:
            reason = str(omission.get("reason", "unknown"))
            omitted_reasons[reason] = omitted_reasons.get(reason, 0) + 1
        return {
            "policy_id": attention_policy.get("policy_id", ATTENTION_POLICY_ID),
            "context": dict(attention_policy.get("context", {})),
            "focus_terms": list(attention_policy.get("focus_terms", []) or []),
            "suppress_terms": list(attention_policy.get("suppress_terms", []) or []),
            "weight_adjustments": dict(
                attention_policy.get("weight_adjustments", {})
            ),
            "selected_item_refs": [item["atom_ref"] for item in items],
            "inhibited_refs": inhibited[:50],
            "omitted_reasons": omitted_reasons,
        }

    def _recency_score(self, atom: Mapping[str, Any]) -> float:
        seconds = self._seconds_since(atom.get("updated_at") or atom.get("observed_at"))
        if seconds is None:
            return 0.0
        return max(
            0.0,
            min(1.0, 1.0 - (float(seconds) / RETRIEVAL_RECENCY_HORIZON_SECONDS)),
        )

    def _graph_activation_scores(
        self,
        atoms: Sequence[Mapping[str, Any]],
        *,
        cues: Sequence[str],
        request_scope: Mapping[str, Any] | None,
        requester: str,
        target_processor: str,
        include_conflicts: bool,
        include_low_health: bool,
        cue_text: str,
        cue_tokens: set[str],
        attention_policy: Mapping[str, Any] | None,
        superseded_refs: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, float]:
        eligible_refs: set[str] = set()
        seed_strengths: dict[str, float] = {}
        for atom in atoms:
            atom_ref = str(atom.get("id") or "")
            if not atom_ref or atom.get("deleted"):
                continue
            if not scope_visible(atom["scope"], request_scope or {}):
                continue
            if not access_visible(atom["access_policy"], requester, target_processor):
                continue
            if atom["health_status"] == "contradicted" and not include_conflicts:
                continue
            if atom["health_status"] in LOW_HEALTH_STATES and not include_low_health:
                continue
            if superseded_refs and atom_ref in superseded_refs:
                continue
            eligible_refs.add(atom_ref)
            search_index = self._atom_search_index(atom, allow_stale=True)
            text = str(search_index["text"])
            text_tokens = set(str(token) for token in search_index["tokens"])
            direct = any(cue.lower() in text for cue in cues if cue)
            overlap = len(cue_tokens.intersection(text_tokens))
            cue_score = 1.0 if direct else min(1.0, overlap / max(1, len(cue_tokens)))
            attention = self._attention_score_components(
                atom,
                text=text,
                text_tokens=text_tokens,
                edge_degree=0,
                attention_policy=attention_policy,
            )
            seed = max(cue_score, float(attention.get("attention_focus", 0.0) or 0.0))
            if seed > 0:
                seed_strengths[atom_ref] = seed
        if not seed_strengths:
            return {}

        atoms_by_ref = {str(atom.get("id") or ""): atom for atom in atoms}
        activation: dict[str, float] = {}
        for edge in self.store.list_edges_for_refs(sorted(eligible_refs)):
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source not in eligible_refs or target not in eligible_refs:
                continue
            if not self._hot_graph_edge_visible(edge, atoms_by_ref):
                continue
            relation_weight = self._edge_relation_activation_weight(
                str(edge.get("relation") or "")
            )
            if source in seed_strengths:
                activation[target] = max(
                    activation.get(target, 0.0),
                    min(1.0, seed_strengths[source] * relation_weight),
                )
            if target in seed_strengths:
                activation[source] = max(
                    activation.get(source, 0.0),
                    min(1.0, seed_strengths[target] * relation_weight * 0.8),
                )
        return activation

    def _hot_graph_edge_degree_counts(
        self, atoms: Sequence[Mapping[str, Any]]
    ) -> dict[str, int]:
        atoms_by_ref = {
            str(atom.get("id") or ""): atom
            for atom in atoms
            if str(atom.get("id") or "")
        }
        refs = sorted(atoms_by_ref)
        if not refs:
            return {}
        counts: dict[str, int] = {}
        for edge in self.store.list_edges_for_refs(refs):
            if not self._hot_graph_edge_visible(edge, atoms_by_ref):
                continue
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source in atoms_by_ref:
                counts[source] = counts.get(source, 0) + 1
            if target in atoms_by_ref:
                counts[target] = counts.get(target, 0) + 1
        return counts

    def _hot_graph_edge_visible(
        self,
        edge: Mapping[str, Any],
        atoms_by_ref: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        relation = str(edge.get("relation") or "")
        source = atoms_by_ref.get(str(edge.get("source_ref") or ""))
        target = atoms_by_ref.get(str(edge.get("target_ref") or ""))
        if not source or not target:
            return False
        if relation in {"rel:derived_from", "rel:supersedes"}:
            return True
        return not (
            source.get("lifecycle_state") == "archived"
            or target.get("lifecycle_state") == "archived"
        )

    def _edge_relation_activation_weight(self, relation: str) -> float:
        if relation in {
            "rel:uses",
            "rel:supports",
            "rel:produced_outcome",
            "rel:made_commitment",
            "rel:has_capability",
            "rel:has_limitation",
        }:
            return 0.9
        if relation in {"rel:derived_from", "rel:supersedes"}:
            return 0.75
        if relation in {"rel:contradicts", "rel:similar_to"}:
            return 0.65
        return 0.5

    def _rank_atom(
        self,
        atom: Mapping[str, Any],
        cues: Sequence[str],
        *,
        request_scope: Mapping[str, Any] | None = None,
        retrieval_mode: str = "general",
        cue_text: str | None = None,
        cue_tokens: set[str] | None = None,
        cue_vector: Sequence[float] | None = None,
        edge_degrees: Mapping[str, int] | None = None,
        edge_activation_scores: Mapping[str, float] | None = None,
        attention_policy: Mapping[str, Any] | None = None,
        superseded_refs: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[float, bool, dict[str, float]]:
        search_index = self._atom_search_index(atom, allow_stale=True)
        text = str(search_index["text"])
        cue_text = " ".join(cues).lower() if cue_text is None else cue_text
        cue_tokens = (
            {token for token in re.findall(r"[a-z0-9_]+", cue_text) if token}
            if cue_tokens is None
            else cue_tokens
        )
        text_tokens = set(str(token) for token in search_index["tokens"])
        direct = any(cue.lower() in text for cue in cues if cue)
        overlap = len(cue_tokens.intersection(text_tokens))
        matched = direct or overlap > 0 or not cue_tokens
        semantic_similarity = 0.0
        if cue_text:
            cue_vector = self.smp.encode(cue_text) if cue_vector is None else cue_vector
            semantic_similarity = cosine(cue_vector, search_index["vector"])
            matched = matched or semantic_similarity >= SEMANTIC_MATCH_THRESHOLD
        direct_score = 1.0 if direct else min(1.0, overlap / max(1, len(cue_tokens)))
        edge_degree = int((edge_degrees or {}).get(atom["id"], 0))
        edge_activation = min(
            1.0, max(0.0, float((edge_activation_scores or {}).get(atom["id"], 0.0)))
        )
        matched = matched or edge_activation > 0.0
        recency = self._recency_score(atom)
        confidence = confidence_score(atom["confidence"])
        utility = min(1.0, float(atom["utility"]))
        salience = min(1.0, float(atom["salience"]))
        request_scope = dict(request_scope or {})
        scope_specificity = (
            min(1.0, len(atom["scope"]) / max(1, len(request_scope)))
            if request_scope
            else 0.0
        )
        attention_components = self._attention_score_components(
            atom,
            text=text,
            text_tokens=text_tokens,
            edge_degree=edge_degree,
            attention_policy=attention_policy,
        )
        relevance_signal = max(
            direct_score,
            float(attention_components.get("attention_focus", 0.0) or 0.0),
            edge_activation * 0.5,
        )
        goal_relevance = (
            relevance_signal if atom["type"] in {"goal", "commitment"} else 0.0
        )
        procedural_applicability = (
            relevance_signal if atom["type"] == "procedure" else 0.0
        )
        contradiction_penalty = (
            1.0 if atom["health_status"] == "contradicted" else 0.0
        )
        staleness_penalty = (
            1.0
            if atom["health_status"] == "stale" or atom["lifecycle_state"] == "archived"
            else 0.0
        )
        redundancy_penalty = 1.0 if atom["health_status"] == "merged" else 0.0
        superseded_penalty = 1.0 if atom["id"] in (superseded_refs or {}) else 0.0
        components = {
            "direct_cue_match": direct_score,
            "semantic_similarity": semantic_similarity,
            "edge_activation": edge_activation,
            "recency": recency,
            "confidence": confidence,
            "utility": utility,
            "salience": salience,
            "scope_specificity": scope_specificity,
            "goal_relevance": goal_relevance,
            "procedural_applicability": procedural_applicability,
            "contradiction_penalty": contradiction_penalty,
            "staleness_penalty": staleness_penalty,
            "redundancy_penalty": redundancy_penalty,
            "superseded_penalty": superseded_penalty,
        }
        if retrieval_mode == "agentic_recall":
            components.update(self._agentic_score_components(atom))
        components.update(attention_components)
        score = 0.0
        weights = dict(RETRIEVAL_WEIGHTS)
        if retrieval_mode == "agentic_recall":
            weights.update(
                {
                    "agency_match": 0.16,
                    "attribution_confidence": 0.12,
                    "correction_learning_relevance": 0.10,
                    "over_attribution_penalty": -0.25,
                    "omitted_counterevidence_penalty": -0.25,
                    "ignored_failure_penalty": -0.20,
                }
            )
        for name, component in components.items():
            score += weights.get(name, 0.0) * component
        score = max(0.0, min(1.0, score))
        return score, matched, {key: round(value, 4) for key, value in components.items()}

    def _packet_item(
        self,
        atom: Mapping[str, Any],
        score: float,
        *,
        requester: str,
        target_processor: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        evidence_refs, evidence_omissions = self._visible_evidence_refs(
            atom, requester=requester, target_processor=target_processor
        )
        item = {
            "item_ref": atom["id"],
            "item_kind": "atom",
            "atom_id": atom["id"],
            "atom_type": atom["type"],
            "atom_ref": atom["id"],
            "type": atom["type"],
            "payload": atom["payload"],
            "confidence": atom["confidence"],
            "rank": None,
            "activation_score": round(score, 4),
            "score": round(score, 4),
            "score_components": dict(atom.get("_score_components", {})),
            "salience": atom["salience"],
            "utility": atom["utility"],
            "rendered_content": self._render_atom(atom),
            "evidence_refs": evidence_refs,
            "access_decision": {
                "atom": "allowed",
                "evidence": "allowed" if evidence_refs == atom["evidence_refs"] else "denied",
            },
            "freshness": {
                "updated_at": atom["updated_at"],
                "health_status": atom["health_status"],
            },
            "scope": atom["scope"],
            "lifecycle_state": atom["lifecycle_state"],
            "health_status": atom["health_status"],
            "updated_at": atom["updated_at"],
            "version": atom["version"],
            "provenance": {
                "created_at": atom["created_at"],
                "observed_at": atom["observed_at"],
                "layer": atom["layer"],
                "retention_class": atom["retention_class"],
            },
        }
        return item, evidence_omissions

    def _visible_evidence_refs(
        self,
        atom: Mapping[str, Any],
        *,
        requester: str,
        target_processor: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        evidence_refs = list(atom["evidence_refs"])
        policy = atom["access_policy"]
        evidence_visibility = policy.get("evidence_visibility", policy.get("visibility", ["all"]))
        allowed = (
            "all" in evidence_visibility
            or requester in evidence_visibility
            or target_processor in evidence_visibility
            or f"processor:{target_processor}" in evidence_visibility
        )
        if allowed:
            return evidence_refs, []
        return [], [
            {
                "atom_ref": atom["id"],
                "reason": "evidence_access_denied",
                "omitted_refs": evidence_refs,
            }
        ]

    def _render_atom(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        payload = atom["payload"]
        if isinstance(payload, Mapping):
            content = (
                payload.get("claim")
                or payload.get("name")
                or payload.get("description")
                or payload.get("summary")
                or canonical_json(payload)
            )
        else:
            content = str(payload)
        return {
            "format": "compact_json",
            "text": str(content),
            "payload": payload,
        }

    def _agentic_score_components(self, atom: Mapping[str, Any]) -> dict[str, float]:
        payload = atom["payload"]
        status = str(
            payload.get("status")
            or payload.get("outcome")
            or payload.get("result")
            or ""
        ).lower()
        has_correction = bool(payload.get("correction") or payload.get("lesson"))
        has_failure = status in {"failure", "failed", "error", "blocked", "denied"}
        return {
            "agency_match": 1.0 if payload_agent_id(payload) else 0.5,
            "attribution_confidence": confidence_score(atom["confidence"]),
            "correction_learning_relevance": 1.0 if has_correction else 0.0,
            "over_attribution_penalty": 0.0,
            "omitted_counterevidence_penalty": 0.0,
            "ignored_failure_penalty": 0.0 if has_failure else 0.1,
        }

    def _byte_budget(self, token_or_byte_budget: int | Mapping[str, int] | None) -> int:
        if token_or_byte_budget is None:
            return 100_000
        if isinstance(token_or_byte_budget, int):
            return max(1, token_or_byte_budget)
        if "bytes" in token_or_byte_budget:
            return max(1, int(token_or_byte_budget["bytes"]))
        if "tokens" in token_or_byte_budget:
            return max(1, int(token_or_byte_budget["tokens"]) * 4)
        return 100_000

    def _capability_unavailable(
        self, name: str, capability_status: Mapping[str, Any], denied: set[str]
    ) -> bool:
        if name in denied:
            return True
        status = capability_status.get(name)
        if status is None:
            return False
        if status is False:
            return True
        if isinstance(status, str):
            return status.lower() in {"denied", "disabled", "false", "unavailable"}
        if isinstance(status, Mapping):
            if status.get("available") is False:
                return True
            if str(status.get("permission", "")).lower() == "denied":
                return True
        return False

    def _status_items(
        self, items: Sequence[Mapping[str, Any]], statuses: set[str]
    ) -> list[dict[str, Any]]:
        matched = []
        for item in items:
            payload = item["payload"]
            status = str(
                payload.get("status")
                or payload.get("outcome")
                or payload.get("result")
                or ""
            ).lower()
            if status in statuses:
                matched.append(dict(item))
        return matched

    def _agentic_responsibility(
        self, item: Mapping[str, Any], *, agent_id: str
    ) -> str:
        payload = item["payload"]
        responsibility = str(payload.get("responsibility", "")).lower()
        if responsibility == "shared_system":
            return "shared_system"
        if responsibility == "external":
            return "external"
        item_agent = payload_agent_id(payload)
        if item_agent == agent_id:
            return "self"
        if item_agent:
            return "other_agent"
        if payload.get("external_constraints") or payload.get("external_actor"):
            return "external"
        return "unknown"

    def _shared_common_item(
        self,
        atom_ref: str,
        processor_items: Sequence[Mapping[str, Any]],
        *,
        processor_count: int,
    ) -> dict[str, Any]:
        common = dict(processor_items[0])
        evidence_sets = [set(item.get("evidence_refs", [])) for item in processor_items]
        if evidence_sets:
            visible_to_all = set.intersection(*evidence_sets)
        else:
            visible_to_all = set()
        common["evidence_refs"] = sorted(visible_to_all)
        common["shared_visibility"] = {
            "visible_processor_count": len(processor_items),
            "requested_processor_count": processor_count,
            "evidence_policy": "least_common_denominator",
            "omitted_evidence_for_some_identities": any(
                set(item.get("evidence_refs", [])) != visible_to_all
                for item in processor_items
            ),
        }
        if len(processor_items) < processor_count:
            common["shared_visibility"]["omitted_for_some_identities"] = True
        return common

    def _self_narrative_has_counterevidence(
        self,
        narrative_item: Mapping[str, Any],
        items: Sequence[Mapping[str, Any]],
    ) -> bool:
        agent_id = payload_agent_id(narrative_item["payload"])
        generated_at = narrative_item.get("updated_at", "")
        for item in items:
            if item["type"] == "self_narrative":
                continue
            payload = item["payload"]
            if payload_agent_id(payload) not in {None, agent_id}:
                continue
            if item.get("updated_at", "") <= generated_at:
                continue
            status = str(
                payload.get("status")
                or payload.get("outcome")
                or payload.get("result")
                or ""
            ).lower()
            if status in {"blocked", "denied", "error", "failed", "failure"}:
                return True
            if payload.get("correction") or payload.get("limitation"):
                return True
            calibration = payload.get("calibration")
            if isinstance(calibration, Mapping):
                if calibration.get("overconfident") is True:
                    return True
                if float(calibration.get("overconfident_claim_rate", 0.0) or 0.0) > 0:
                    return True
        return False

    def _archive_atom_projection(
        self,
        conn: Any,
        atom: Mapping[str, Any],
        *,
        reason: str,
        superseded_by: str,
        actor: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        archived = dict(atom)
        archived["lifecycle_state"] = "archived"
        archived["health_status"] = "merged"
        archived["version"] = int(archived["version"]) + 1
        archived["updated_at"] = utc_now()
        archived["supersedes"] = list(archived.get("supersedes") or []) + [
            superseded_by
        ]
        archived["decay_policy"] = {
            **dict(archived.get("decay_policy") or {}),
            "archive_reason": reason,
            "superseded_by": superseded_by,
        }
        archived["revision_history"] = list(archived.get("revision_history") or [])
        archived["revision_history"].append(
            {
                "version": atom["version"],
                "digest": digest(self._atom_projection(atom)),
                "changed_at": utc_now(),
                "actor": actor,
                "reason": reason,
            }
        )
        archived = normalize_atom(
            self._attach_search_index(archived), require_id=True
        )
        self.store.replace_atom(conn, archived)
        deleted_edges = self.store.mark_edges_deleted_for_ref(conn, archived["id"])
        return archived, deleted_edges

    def _structured_duplicate_key(
        self, atom: Mapping[str, Any]
    ) -> tuple[Any, ...] | None:
        if atom.get("deleted") or atom.get("lifecycle_state") != "active":
            return None
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        scope = atom.get("scope")
        scope = scope if isinstance(scope, Mapping) else {}
        tenant = scope.get("tenant")
        component = scope.get("component")
        asset = scope.get("asset") or payload.get("asset")
        run_id = scope.get("run_id") or payload.get("run_id")
        agent_id = payload.get("agent_id")
        if atom.get("type") == "agentic_trace":
            kind = payload.get("qandl_kind") or payload.get("kind")
            chunk = payload.get("chunk")
            if kind == "reflection" and chunk not in (None, ""):
                return (
                    "agentic_trace.reflection",
                    tenant,
                    component,
                    asset,
                    run_id,
                    agent_id,
                    chunk,
                )
        if atom.get("type") == "runtime_state":
            role_key = payload.get("role_key") or payload.get("role")
            if agent_id:
                return (
                    "runtime_state.current",
                    tenant,
                    component,
                    asset,
                    run_id,
                    agent_id,
                    role_key,
                )
        return None

    def _structured_duplicate_quality(self, atom: Mapping[str, Any]) -> int:
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        score = 0
        for key in (
            "directive_atom_ref",
            "source_directive_ref",
            "control_signature",
            "metric_deltas",
            "tool_surface",
            "runtime_capabilities",
            "runtime_constraints",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                score += 1
        score += min(5, len(canonical_json(payload)) // 1000)
        return score

    def _intrinsic_edges_for_atom(self, atom: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Project deterministic graph edges encoded by structured atom fields."""

        atom_id = str(atom["id"])
        scope = dict(atom.get("scope") or {})
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def active_atom(ref: Any) -> dict[str, Any] | None:
            ref_id = str(ref or "")
            if not ref_id or ref_id == atom_id:
                return None
            existing = self.store.get_atom(ref_id)
            if existing is None or existing.get("deleted"):
                return None
            if existing.get("lifecycle_state") != "active":
                return None
            return existing

        def add(source_ref: Any, target_ref: Any, relation: str) -> None:
            source = str(source_ref or "")
            target = str(target_ref or "")
            if not source or not target or source == target:
                return
            if source != atom_id and active_atom(source) is None:
                return
            if target != atom_id and active_atom(target) is None:
                return
            key = (source, target, relation)
            if key in seen:
                return
            seen.add(key)
            edges.append(self._edge(source, target, relation, scope))

        for ref in _structured_ref_list(atom.get("supersedes")):
            add(atom_id, ref, "rel:supersedes")

        for ref in _structured_ref_list(payload.get("source_refs")):
            add(atom_id, ref, "rel:derived_from")

        for ref in _structured_ref_list(payload.get("memory_references")):
            add(atom_id, ref, "rel:uses")

        directive_ref = payload.get("directive_atom_ref") or payload.get(
            "source_directive_ref"
        )
        if atom.get("type") == "agentic_trace" and directive_ref:
            add(directive_ref, atom_id, "rel:produced_outcome")

        if atom.get("type") in {
            "capability",
            "limitation",
            "commitment",
            "runtime_state",
            "self_assessment",
        }:
            relation_by_type = {
                "capability": "rel:has_capability",
                "limitation": "rel:has_limitation",
                "commitment": "rel:made_commitment",
                "runtime_state": "rel:attributed_to",
                "self_assessment": "rel:attributed_to",
            }
            relation = relation_by_type[str(atom["type"])]
            for ref in _structured_ref_list(atom.get("evidence_refs")):
                source = active_atom(ref)
                if source and source.get("type") == "self_model":
                    add(source["id"], atom_id, relation)

        return edges

    def _edge(
        self,
        source_ref: str,
        target_ref: str,
        relation: str,
        scope: Mapping[str, Any],
    ) -> dict[str, Any]:
        relation = normalize_relation(relation)
        now = utc_now()
        return {
            "edge_id": stable_id(
                "edge",
                {
                    "source_ref": source_ref,
                    "target_ref": target_ref,
                    "relation": relation,
                    "scope": dict(scope),
                },
            ),
            "source_ref": source_ref,
            "target_ref": target_ref,
            "relation": relation,
            "schema_version": SCHEMA_VERSION,
            "evidence_refs": [],
            "scope": dict(scope),
            "confidence": {"level": "medium-high", "score": 0.75},
            "lifecycle_state": "active",
            "health_status": "healthy",
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "deleted": 0,
        }

    def _contradiction_signature(
        self, atom: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], str] | None:
        payload = atom["payload"]
        if {"subject", "predicate", "value"}.issubset(payload):
            key = (
                atom["type"],
                canonical_json(atom["scope"]),
                payload["subject"],
                payload["predicate"],
            )
            return key, canonical_json(payload["value"])
        if {"key", "value"}.issubset(payload):
            key = (atom["type"], canonical_json(atom["scope"]), payload["key"])
            return key, canonical_json(payload["value"])
        return None

    def _atom_projection(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in atom.items()
            if key not in {"deleted", "revision_history", "last_accessed"}
        }

    def _memory_identity_digest(self, atom: Mapping[str, Any]) -> str:
        return digest(
            {
                "type": atom["type"],
                "payload": atom["payload"],
                "scope": atom["scope"],
                "evidence_refs": atom["evidence_refs"],
            }
        )

    def _counts(self, rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key))
            counts[value] = counts.get(value, 0) + 1
        return counts


def _top_symmetric_components(
    matrix: list[list[float]],
    *,
    count: int,
    labels: Sequence[str],
    iterations: int = 80,
    tolerance: float = 1e-10,
) -> list[tuple[float, list[float]]]:
    """Return deterministic leading eigen-components for a small dense matrix."""

    size = len(matrix)
    if size == 0 or count <= 0:
        return []
    working = [row[:] for row in matrix]
    components: list[tuple[float, list[float]]] = []
    for component_index in range(min(count, size)):
        vector = _deterministic_unit_vector(labels, component_index)
        if not vector:
            break
        for _ in range(iterations):
            candidate = _matrix_vector_product(working, vector)
            norm = math.sqrt(sum(value * value for value in candidate))
            if norm <= tolerance:
                break
            candidate = [value / norm for value in candidate]
            delta = sum(abs(left - right) for left, right in zip(candidate, vector))
            vector = candidate
            if delta <= tolerance:
                break
        projected = _matrix_vector_product(working, vector)
        eigenvalue = sum(left * right for left, right in zip(vector, projected))
        if eigenvalue <= tolerance:
            break
        pivot = max(range(size), key=lambda index: abs(vector[index]))
        if vector[pivot] < 0:
            vector = [-value for value in vector]
        components.append((eigenvalue, vector))
        for row_index in range(size):
            row_value = vector[row_index]
            for column_index in range(size):
                working[row_index][column_index] -= (
                    eigenvalue * row_value * vector[column_index]
                )
    return components


def _matrix_vector_product(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    return [
        sum(row[index] * vector[index] for index in range(len(vector)))
        for row in matrix
    ]


def _deterministic_unit_vector(labels: Sequence[str], salt: int) -> list[float]:
    values: list[float] = []
    for label in labels:
        raw = int(digest({"label": label, "salt": salt})[:12], 16)
        values.append((raw % 2000000) / 1000000.0 - 1.0)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 and values:
        values[0] = 1.0
        norm = 1.0
    return [value / norm for value in values] if norm > 0 else []
