"""High-level AMOS v1 service API."""

from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import AccessDenied, CASConflict, IdempotencyConflict, ValidationError
from .maintenance import (
    EvidenceWindow,
    MaintenanceProcessor,
    default_processor_registry,
    load_maintenance_processor,
    proposal_is_auto_committable,
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
    "self_awareness": {"max_items": 32, "tokens": 6000, "include_conflicts": True},
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
        "run_steward": True,
        "rebuild_indexes": True,
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
}
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
    "contradiction_penalty": -0.30,
    "staleness_penalty": -0.18,
    "redundancy_penalty": -0.15,
}


def scope_visible(atom_scope: Mapping[str, Any], request_scope: Mapping[str, Any]) -> bool:
    if not atom_scope:
        return True
    for key, value in atom_scope.items():
        if value == "global":
            continue
        if request_scope.get(key) != value:
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

    def run_memory_policy(
        self,
        *,
        force: bool = False,
        trigger: str = "scheduler",
        scope: Mapping[str, Any] | None = None,
        actor: str = "svc:memory_policy",
    ) -> dict[str, Any]:
        if getattr(self, "_memory_policy_running", False):
            return {
                "status": "skipped",
                "reason": "memory_policy_already_running",
                "trigger": trigger,
                "graph_version": self.store.graph_version(),
            }
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

        self._memory_policy_running = True
        started_graph_version = self.store.graph_version()
        scope = dict(scope or {})
        results: dict[str, Any] = {}
        target_refs: list[str] = []
        try:
            maintenance = policy["maintenance"]
            if maintenance["enabled"] and maintenance["run_smp"]:
                results["smp"] = self.run_smp_analysis(scope=scope)
            if maintenance["enabled"] and maintenance["run_steward"]:
                results["steward"] = self.run_steward(scope=scope, actor=actor)
                for action in results["steward"].get("actions", []):
                    target_refs.extend(
                        ref
                        for ref in action.get("atom_refs", [])
                        + [action.get("kept"), action.get("archived")]
                        if ref
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
                "results": results,
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
        request_payload = {"operation": "commit_atom", "atom": dict(atom)}
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
        committed = [self.commit_atom(atom, actor=actor) for atom in atoms]
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
            updated = normalize_atom(updated, require_id=True)
            event = self.store.append_event(
                conn,
                event_type="atom_updated",
                actor=actor,
                payload={"operation": "update_atom", "before": current, "after": updated},
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
            response = {"status": "updated", "atom": updated, "event": event}
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
            updated = normalize_atom(updated, require_id=True)
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
                archived = normalize_atom(archived, require_id=True)
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
        type_filter: Sequence[str] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
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
            "type_filter": list(type_filter or []),
            "pressure_mode": pressure_mode,
            "run_policy": bool(run_policy),
        }
        candidates: list[tuple[float, dict[str, Any]]] = []
        omissions: list[dict[str, Any]] = []
        allowed_types = set(type_filter or [])
        for atom in self.store.list_atoms():
            atom_ref = atom["id"]
            if atom.get("deleted"):
                omissions.append({"atom_ref": atom_ref, "reason": "deleted"})
                continue
            if allowed_types and atom["type"] not in allowed_types:
                continue
            if not scope_visible(atom["scope"], request["scope"]):
                omissions.append({"atom_ref": atom_ref, "reason": "scope_hidden"})
                continue
            if not access_visible(atom["access_policy"], requester, target_processor):
                omissions.append({"atom_ref": atom_ref, "reason": "access_hidden"})
                continue
            if atom["lifecycle_state"] == "archived" and not include_archived:
                omissions.append({"atom_ref": atom_ref, "reason": "archived"})
                continue
            if atom["lifecycle_state"] not in {"active", "proposed", "archived"}:
                omissions.append(
                    {"atom_ref": atom_ref, "reason": f"lifecycle:{atom['lifecycle_state']}"}
                )
                continue
            if atom["health_status"] == "contradicted" and not include_conflicts:
                omissions.append({"atom_ref": atom_ref, "reason": "contradicted"})
                continue
            if atom["health_status"] in LOW_HEALTH_STATES and not include_low_health:
                omissions.append(
                    {"atom_ref": atom_ref, "reason": f"health:{atom['health_status']}"}
                )
                continue
            score, matched, components = self._rank_atom(
                atom,
                request["cues"],
                request_scope=request["scope"],
                retrieval_mode=retrieval_mode,
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
            for edge in self.store.list_edges():
                if edge["relation"] not in CONFLICT_RELATIONS:
                    continue
                if edge["source_ref"] in selected or edge["target_ref"] in selected:
                    conflicts.append(edge)

        graph_version = self.store.graph_version()
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
        with self.store.transaction() as conn:
            return self.store.insert_retrieval_outcome(
                conn, packet_id=packet_id, request=request, outcome=outcome
            )

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
                    changed = normalize_atom(changed, require_id=True)
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
        packet = self.retrieve_packet(
            scope=scope or {},
            requester=requester,
            target_processor=target_processor,
            retrieval_mode="self_awareness",
            max_items=100,
            include_conflicts=True,
            include_low_health=True,
            type_filter=[
                "capability",
                "commitment",
                "limitation",
                "runtime_state",
                "self_assessment",
                "self_model",
            ],
        )
        by_type: dict[str, list[dict[str, Any]]] = {
            "capability": [],
            "commitment": [],
            "limitation": [],
            "runtime_state": [],
            "self_assessment": [],
            "self_model": [],
        }
        omissions = list(packet["omissions"])
        latest_runtime = None
        for item in packet["items"]:
            payload = item["payload"]
            if payload_agent_id(payload) not in {None, agent_id}:
                omissions.append(
                    {"atom_ref": item["atom_ref"], "reason": "different_agent"}
                )
                continue
            if item["type"] == "runtime_state":
                if latest_runtime is None or item["updated_at"] > latest_runtime["updated_at"]:
                    latest_runtime = item
            by_type[item["type"]].append(item)

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

        return {
            "view": "self_awareness",
            "agent_id": agent_id,
            "graph_version": packet["graph_version"],
            "generated_at": utc_now(),
            "self_model": by_type["self_model"],
            "capabilities": visible_capabilities,
            "limitations": by_type["limitation"],
            "open_commitments": [
                item
                for item in by_type["commitment"]
                if str(item["payload"].get("status", "open")).lower()
                not in {"fulfilled", "cancelled", "canceled", "superseded"}
            ],
            "runtime_state": latest_runtime,
            "assessments": by_type["self_assessment"],
            "calibration": self.calibrate_self_model(
                agent_id=agent_id, scope=scope or {}, record=False
            )["calibration"],
            "omissions": omissions,
            "conflicts": packet["conflicts"],
            "source_packet_id": packet["packet_id"],
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
        for processor in processors:
            proposals.extend(proposal.to_dict() for proposal in processor.propose(window))
        proposals.sort(
            key=lambda proposal: (
                proposal["risk_level"] != "low",
                proposal["processor_id"],
                proposal["proposal_id"],
            )
        )

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
                    }
                )

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
        event_payload = {
            "operation": "run_maintenance_distiller",
            "scope": scope,
            "domain": domain,
            "window": window.to_dict(),
            "processors": [
                {
                    "processor_id": processor.processor_id,
                    "processor_version": processor.processor_version,
                }
                for processor in processors
            ],
            "missing_processors": missing_processors,
            "proposal_count": len(proposals),
            "committed_count": len(
                [item for item in committed if item.get("status") == "committed"]
            ),
            "already_committed_count": len(
                [
                    item
                    for item in committed
                    if item.get("status") == "already_committed"
                ]
            ),
            "deferred_count": len(deferred),
            "auto_commit_low_risk": auto_commit_low_risk,
            "reviewer": self._maintenance_reviewer_status(reviewer),
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
            "event": event,
            "graph_version": self.store.graph_version(),
        }

    def run_smp_analysis(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        target_refs: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        atoms = [
            atom
            for atom in self.store.list_atoms()
            if not atom.get("deleted") and scope_visible(atom["scope"], scope)
        ]
        if target_refs:
            allowed = set(target_refs)
            atoms = [atom for atom in atoms if atom["id"] in allowed]
        shape_reports = [self.smp.validate_shape(atom) for atom in atoms]
        clusters = self.smp.cluster(atoms)
        conflicts = self.smp.detect_conflicts(atoms)
        health = [self.smp.propose_health(atom) for atom in atoms]
        links = []
        for index, atom in enumerate(atoms):
            links.extend(self.smp.propose_links(atom, atoms[index + 1 :]))
        outputs = shape_reports + clusters + conflicts + health + links
        return {
            "status": "completed",
            "processor_id": self.smp.processor_id,
            "processor_version": self.smp.processor_version,
            "graph_version": self.store.graph_version(),
            "scope": scope,
            "outputs": outputs,
            "review_required": [
                output
                for output in outputs
                if output["risk_level"] == "high"
                or output["recommended_action"].get("type") in HIGH_RISK_MAINTENANCE
            ],
        }

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
                for atom in self.store.list_atoms()
                if not atom.get("deleted") and scope_visible(atom["scope"], scope)
            ]
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
                duplicate = dict(atom)
                duplicate["lifecycle_state"] = "archived"
                duplicate["health_status"] = "merged"
                duplicate["version"] = int(duplicate["version"]) + 1
                duplicate["updated_at"] = utc_now()
                duplicate["supersedes"] = list(duplicate["supersedes"]) + [existing["id"]]
                duplicate = normalize_atom(duplicate, require_id=True)
                self.store.replace_atom(conn, duplicate)
                projected_atoms.append(duplicate)
                edge = self._edge(
                    existing["id"], duplicate["id"], "rel:similar_to", scope
                )
                self.store.insert_edge(conn, edge)
                projected_edges.append(edge)
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

            contradiction_groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
            for atom in atoms:
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
                        changed = normalize_atom(changed, require_id=True)
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

    def health_memory(self) -> dict[str, Any]:
        policy_tick = self.run_memory_policy(trigger="health_memory")
        atoms = self.store.list_atoms()
        events = self.store.list_events()
        indexes = self.store.list_derived_index_metadata()
        by_type = self._counts(atoms, "type")
        by_health = self._counts(atoms, "health_status")
        by_lifecycle = self._counts(atoms, "lifecycle_state")
        return {
            "graph_version": self.store.graph_version(),
            "journal_events": len(events),
            "atoms": len(atoms),
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
            elif event_type in {"atom_merged", "steward_run"}:
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
        atoms = [
            atom
            for atom in self.store.list_atoms()
            if not atom.get("deleted") and scope_visible(atom["scope"], scope)
        ][: max(1, int(max_atoms or 1))]
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
        events = self.store.list_events()[-event_limit:] if event_limit else []
        list_outcomes = getattr(self.store, "list_retrieval_outcomes", None)
        retrieval_outcomes = (
            list_outcomes()[: max(0, int(max_retrieval_outcomes or 0))]
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
            } and isinstance(value, Mapping):
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
        for key in ["enabled", "run_smp", "run_steward", "rebuild_indexes", "invalidate_packet_cache"]:
            maintenance[key] = bool(maintenance.get(key, True))
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
        return {
            "due": bool(reasons),
            "reasons": reasons,
            "graph_version": graph_version,
            "last_graph_version": last_graph_version,
            "graph_delta": graph_delta,
            "elapsed_seconds": elapsed_seconds,
            "pressure_mode": pressure_mode,
        }

    def _seconds_since(self, timestamp: Any) -> int | None:
        if not timestamp:
            return None
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))

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
        for atom in self.store.list_atoms():
            if atom.get("deleted") or atom["type"] != "semantic":
                continue
            payload = atom.get("payload", {})
            if payload.get("created_by") != "svc:memory_policy":
                continue
            if payload.get("distillation_type") != distillation["distillation_type"]:
                continue
            covered_sources.update(str(ref) for ref in payload.get("source_refs", []))
        candidates = []
        for atom in self.store.list_atoms():
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
        candidates.sort(key=lambda atom: (atom.get("observed_at") or atom["created_at"], atom["id"]))
        return candidates

    def _policy_distillation_summary(
        self, atoms: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        type_counts = self._counts(atoms, "type")
        scopes = sorted({canonical_json(atom["scope"]) for atom in atoms})
        highlights = []
        for atom in atoms[:8]:
            rendered = self._render_atom(atom)
            highlights.append(
                {
                    "atom_ref": atom["id"],
                    "type": atom["type"],
                    "text": rendered["text"],
                }
            )
        return {
            "summary": (
                "Automatic AMOS memory policy distillation over "
                f"{len(atoms)} source atoms."
            ),
            "source_count": len(atoms),
            "source_types": type_counts,
            "scope_fingerprints": scopes,
            "highlights": highlights,
            "policy": "amos.v1.automatic_memory_policy",
        }

    def _rebuild_derived_indexes(
        self, *, graph_version: int | None = None
    ) -> dict[str, Any]:
        atoms = self.store.list_atoms()
        edges = self.store.list_edges()
        graph_version = (
            graph_version if graph_version is not None else self.store.graph_version()
        )
        with self.store.transaction() as conn:
            lexical = self.store.upsert_derived_index_metadata(
                conn,
                index_name="semantic_lexical_vectors",
                graph_version=graph_version,
                freshness="fresh",
                details={
                    "atom_count": len([atom for atom in atoms if not atom.get("deleted")]),
                    "processor_id": self.smp.processor_id,
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
                    "edge_count": len(edges),
                    "rebuildable_from_canonical": True,
                    "maintained_by": "memory_policy",
                },
            )
        return {
            "status": "rebuilt",
            "graph_version": graph_version,
            "indexes": [lexical, graph],
        }

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

    def _rank_atom(
        self,
        atom: Mapping[str, Any],
        cues: Sequence[str],
        *,
        request_scope: Mapping[str, Any] | None = None,
        retrieval_mode: str = "general",
    ) -> tuple[float, bool, dict[str, float]]:
        text = (
            atom["id"]
            + " "
            + atom["type"]
            + " "
            + canonical_json(atom["payload"])
        ).lower()
        cue_text = " ".join(cues).lower()
        cue_tokens = {token for token in re.findall(r"[a-z0-9_]+", cue_text) if token}
        text_tokens = set(re.findall(r"[a-z0-9_]+", text))
        direct = any(cue.lower() in text for cue in cues if cue)
        overlap = len(cue_tokens.intersection(text_tokens))
        matched = direct or overlap > 0 or not cue_tokens
        semantic_similarity = 0.0
        if cue_text:
            semantic_similarity = cosine(self.smp.encode(cue_text), self.smp.encode(atom))
        direct_score = 1.0 if direct else min(1.0, overlap / max(1, len(cue_tokens)))
        edge_activation = self._edge_activation(atom["id"])
        recency = 1.0 if atom.get("updated_at") else 0.5
        confidence = confidence_score(atom["confidence"])
        utility = min(1.0, float(atom["utility"]))
        salience = min(1.0, float(atom["salience"]))
        request_scope = dict(request_scope or {})
        scope_specificity = (
            min(1.0, len(atom["scope"]) / max(1, len(request_scope)))
            if request_scope
            else 0.0
        )
        goal_relevance = 1.0 if atom["type"] in {"goal", "commitment"} else 0.0
        procedural_applicability = 1.0 if atom["type"] == "procedure" else 0.0
        contradiction_penalty = (
            1.0 if atom["health_status"] == "contradicted" else 0.0
        )
        staleness_penalty = (
            1.0
            if atom["health_status"] == "stale" or atom["lifecycle_state"] == "archived"
            else 0.0
        )
        redundancy_penalty = 1.0 if atom["health_status"] == "merged" else 0.0
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
        }
        if retrieval_mode == "agentic_recall":
            components.update(self._agentic_score_components(atom))
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

    def _edge_activation(self, atom_id: str) -> float:
        count = 0
        for edge in self.store.list_edges():
            if edge["source_ref"] == atom_id or edge["target_ref"] == atom_id:
                count += 1
        return min(1.0, count / 5.0)

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
