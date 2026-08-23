"""Generic interaction continuity transactions and cognitive workspaces.

This module contains no application routing or semantic classification.  It
binds caller-authored typed records to canonical AMOS atoms, graph edges, and
journal-derived compare-and-swap heads, then compiles bounded generated views
from those canonical records.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ._service_support import access_visible, scope_visible
from .errors import (
    CASConflict,
    CognitiveWorkspaceBudgetExceeded,
    StaleFrameError,
    ValidationError,
)
from .governance_service import GovernanceService
from .maintenance import context_compaction_source_digest
from .schemas import (
    CONTEXT_COMPACTION_PROFILE,
    SCHEMA_VERSION,
    canonical_json,
    digest,
    ensure_jsonable,
    normalize_atom,
    normalize_evidence,
    normalize_scope,
    stable_id,
    utc_now,
)


class ContinuityService:
    """Application-neutral event/head mutation and context compilation."""

    TRANSACTION_PROFILE = "amos.memory-transaction.v1"
    WORKSPACE_PROFILE = "amos.cognitive-workspace.v1"
    INTERACTION_PROJECTION_PROFILE = "amos.interaction-projection.v2"
    MEMORY_HEAD_PROFILE = "amos.memory-head.v1"
    MEMORY_SERIES_VERSIONS_PROFILE = "amos.memory-series-versions.v1"
    MEMORY_TRANSACTION_OBSERVATION_PROFILE = (
        "amos.memory-transaction-observation.v1"
    )
    SUPPORTED_HEAD_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "assessment_qualification", "authority_record", "discourse_thread",
            "goal_work", "interaction_stream", "project_work",
        }
    )

    def __init__(
        self,
        store: Any,
        access: Any,
        indexes: Any,
        graph: Any,
        mutations: Any,
        reasoning: Any,
    ):
        self.store = store
        self.reasoning = reasoning
        self._mark_foreground_activity = access._mark_foreground_activity
        self._idempotency_hit = access._idempotency_hit
        self._record_idempotency = access._record_idempotency
        self._prepare_committed_atom = indexes._prepare_committed_atom
        self._attach_search_index = indexes._attach_search_index
        self._intrinsic_edges_for_atom = graph._intrinsic_edges_for_atom
        self._memory_identity_digest = graph._memory_identity_digest
        self._atom_projection = graph._atom_projection
        self._edge = graph._edge
        self._assert_direct_commit_governance_safe = (
            mutations._assert_direct_commit_governance_safe
        )

    @staticmethod
    def _required_text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _mapping(value: Any, name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValidationError(f"{name} must be an object")
        return dict(value)

    @staticmethod
    def _sequence(value: Any, name: str) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"{name} must be a list")
        return list(value)

    @staticmethod
    def _same_scope(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return digest(dict(left)) == digest(dict(right))

    def _scoped_atom(
        self, atom: Mapping[str, Any], scope: Mapping[str, Any]
    ) -> dict[str, Any]:
        value = dict(atom)
        supplied_scope = normalize_scope(value.get("scope"))
        if supplied_scope and not self._same_scope(supplied_scope, scope):
            raise ValidationError("transaction atom scope must match transaction scope")
        value["scope"] = dict(scope)
        return value

    def _scoped_evidence(
        self, evidence: Mapping[str, Any], scope: Mapping[str, Any]
    ) -> dict[str, Any]:
        value = dict(evidence)
        supplied_scope = normalize_scope(value.get("scope"))
        if supplied_scope and not self._same_scope(supplied_scope, scope):
            raise ValidationError(
                "transaction evidence scope must match transaction scope"
            )
        value["scope"] = dict(scope)
        return normalize_evidence(value)

    def _validate_head_update(
        self,
        raw: Mapping[str, Any],
        *,
        scope: Mapping[str, Any],
        prepared_by_id: Mapping[str, Mapping[str, Any]],
        seen: set[tuple[str, str]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        update = self._mapping(raw, "head_update")
        series_kind = self._required_text(
            update.get("series_kind"), "head_update.series_kind"
        )
        series_id = self._required_text(
            update.get("series_id"), "head_update.series_id"
        )
        new_head_ref = self._required_text(
            update.get("new_head_ref"), "head_update.new_head_ref"
        )
        if series_kind not in self.SUPPORTED_HEAD_KINDS:
            raise ValidationError(f"unsupported head series_kind: {series_kind}")
        key = (series_kind, series_id)
        if key in seen:
            raise ValidationError(
                "a memory transaction may advance a head series only once"
            )
        seen.add(key)
        expected_ref = update.get("expected_head_ref")
        if expected_ref is not None and (
            not isinstance(expected_ref, str) or not expected_ref
        ):
            raise ValidationError(
                "head_update.expected_head_ref must be null or a non-empty string"
            )
        expected_version = update.get("expected_head_version", 0)
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 0
        ):
            raise ValidationError(
                "head_update.expected_head_version must be a non-negative integer"
            )
        current = self.store.get_memory_head(
            scope=scope,
            series_kind=series_kind,
            series_id=series_id,
        )
        current_ref = None if current is None else str(current["head_ref"])
        current_version = 0 if current is None else int(current["head_version"])
        if expected_ref != current_ref or expected_version != current_version:
            raise CASConflict(
                "memory head compare-and-swap failed for "
                f"{series_kind}:{series_id}; expected "
                f"{expected_ref!r}@{expected_version}, current "
                f"{current_ref!r}@{current_version}"
            )
        legacy_predecessor_ref = update.get("legacy_predecessor_ref")
        if legacy_predecessor_ref is not None:
            if (
                series_kind != "authority_record"
                or current is not None
                or expected_ref is not None
                or expected_version != 0
                or not isinstance(legacy_predecessor_ref, str)
                or not legacy_predecessor_ref
            ):
                raise ValidationError(
                    "legacy_predecessor_ref is permitted only when establishing "
                    "an absent authority_record head"
                )
            predecessor = self.store.get_atom(legacy_predecessor_ref)
            if predecessor is None:
                raise ValidationError(
                    "authority_record legacy_predecessor_ref must identify an "
                    "existing atom"
                )
            if normalize_scope(predecessor.get("scope")) != normalize_scope(scope):
                raise ValidationError(
                    "authority_record legacy predecessor must share the head scope"
                )
            current = {
                "head_ref": legacy_predecessor_ref,
                "head_version": 0,
            }
        new_head = prepared_by_id.get(new_head_ref)
        if new_head is None:
            raise ValidationError(
                "head_update.new_head_ref must identify an atom in this transaction"
            )
        if series_kind == "discourse_thread":
            if new_head.get("type") != "discourse_state":
                raise ValidationError(
                    "a discourse_thread head must identify a discourse_state atom"
                )
            payload = new_head.get("payload") or {}
            if payload.get("thread_id") != series_id:
                raise ValidationError(
                    "discourse_state.thread_id must match head_update.series_id"
                )
            if int(payload.get("revision", 0)) != current_version + 1:
                raise ValidationError(
                    "discourse_state.revision must be the next head version"
                )
        elif series_kind == "interaction_stream":
            if new_head.get("type") != "interaction_event":
                raise ValidationError(
                    "an interaction_stream head must identify an "
                    "interaction_event atom"
                )
            payload = new_head.get("payload") or {}
            if payload.get("conversation_id") != series_id:
                raise ValidationError(
                    "interaction_event.conversation_id must match "
                    "head_update.series_id"
                )
            if int(payload.get("sequence", 0)) != current_version + 1:
                raise ValidationError(
                    "interaction_event.sequence must be the next stream "
                    "head version"
                )
            if payload.get("in_reply_to") != current_ref:
                raise ValidationError(
                    "interaction_event.in_reply_to must identify the current "
                    "interaction stream head"
                )
        elif series_kind == "project_work":
            if new_head.get("type") != "goal":
                raise ValidationError(
                    "a project_work head must identify a goal atom"
                )
            payload = new_head.get("payload") or {}
            if payload.get("project_ref") != series_id:
                raise ValidationError(
                    "goal.project_ref must match head_update.series_id"
                )
            if int(payload.get("revision", 0)) != current_version + 1:
                raise ValidationError(
                    "goal.revision must be the next project head version"
                )
        elif series_kind == "goal_work":
            if new_head.get("type") != "goal":
                raise ValidationError(
                    "a goal_work head must identify a goal atom"
                )
            payload = new_head.get("payload") or {}
            if payload.get("goal_ref") != series_id:
                raise ValidationError(
                    "goal.goal_ref must match head_update.series_id"
                )
            if int(payload.get("revision", 0)) != current_version + 1:
                raise ValidationError(
                    "goal.revision must be the next goal head version"
                )
        elif series_kind == "assessment_qualification":
            if new_head.get("type") != "self_assessment":
                raise ValidationError(
                    "an assessment_qualification head must identify a self_assessment atom"
                )
            payload = new_head.get("payload") or {}
            if payload.get("assessment_series_id") != series_id:
                raise ValidationError(
                    "self_assessment.assessment_series_id must match head_update.series_id"
                )
            if int(payload.get("revision", 0)) != current_version + 1:
                raise ValidationError(
                    "self_assessment.revision must be the next assessment head version"
                )
        elif series_kind == "authority_record":
            if new_head.get("type") != "procedure":
                raise ValidationError(
                    "an authority_record head must identify a procedure atom"
                )
            payload = new_head.get("payload") or {}
            if payload.get("authority_series_id") != series_id:
                raise ValidationError(
                    "procedure.authority_series_id must match head_update.series_id"
                )
            if int(payload.get("authority_revision", 0)) != current_version + 1:
                raise ValidationError(
                    "procedure.authority_revision must be the next authority head version"
                )
        projected = {
            "scope": dict(scope),
            "series_kind": series_kind,
            "series_id": series_id,
            "head_ref": new_head_ref,
            "head_version": current_version + 1,
        }
        return projected, current

    def _superseded_projection(
        self,
        atom: Mapping[str, Any],
        *,
        successor_ref: str,
        actor: str,
    ) -> dict[str, Any]:
        projected = dict(atom)
        projected["lifecycle_state"] = "superseded"
        projected["version"] = int(projected["version"]) + 1
        projected["updated_at"] = utc_now()
        history = list(projected.get("revision_history") or [])
        history.append(
            {
                "version": int(atom["version"]),
                "digest": digest(self._atom_projection(atom)),
                "changed_at": projected["updated_at"],
                "actor": actor,
                "reason": "memory_head_advanced",
                "successor_ref": successor_ref,
            }
        )
        projected["revision_history"] = history
        return normalize_atom(
            self._attach_search_index(projected), require_id=True
        )

    def _validate_event_refs(
        self,
        prepared: Sequence[Mapping[str, Any]],
        prepared_by_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        def resolve(ref: str) -> Mapping[str, Any] | None:
            return prepared_by_id.get(ref) or self.store.get_atom(ref)

        for atom in prepared:
            payload = atom.get("payload") or {}
            if atom.get("type") == "interaction_event":
                reply_ref = payload.get("in_reply_to")
                if reply_ref:
                    target = resolve(str(reply_ref))
                    if target is None or target.get("type") != "interaction_event":
                        raise ValidationError(
                            "interaction_event.in_reply_to must identify an "
                            "interaction_event"
                        )
            if atom.get("type") == "discourse_thread":
                target = resolve(str(payload.get("opened_by_event_ref") or ""))
                if target is None or target.get("type") != "interaction_event":
                    raise ValidationError(
                        "discourse_thread.opened_by_event_ref must identify an "
                        "interaction_event"
                    )
            if atom.get("type") == "discourse_state":
                for field in ("head_event_refs", "source_event_refs"):
                    for ref in payload.get(field, []):
                        target = resolve(str(ref))
                        if target is None or target.get("type") != "interaction_event":
                            raise ValidationError(
                                f"discourse_state.{field} must contain only "
                                "interaction_event references"
                            )
                private_state = payload.get("private_state") or []
                visibility = {
                    str(item)
                    for item in (atom.get("access_policy") or {}).get(
                        "visibility", ["all"]
                    )
                }
                if private_state and "all" in visibility:
                    raise ValidationError(
                        "discourse_state containing private_state must use a "
                        "restricted access_policy.visibility"
                    )

    def commit_memory_transaction(
        self,
        *,
        evidence: Sequence[Mapping[str, Any]] | None = None,
        atoms: Sequence[Mapping[str, Any]] | None = None,
        edges: Sequence[Mapping[str, Any]] | None = None,
        head_updates: Sequence[Mapping[str, Any]] | None = None,
        receipt_refs: Sequence[str] | None = None,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
        authorization_context: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Commit one source-bound memory evolution as one journal operation."""

        actor = self._required_text(actor, "actor")
        transaction_scope = normalize_scope(scope)
        evidence_input = self._sequence(evidence, "evidence")
        atom_input = self._sequence(atoms, "atoms")
        edge_input = self._sequence(edges, "edges")
        head_input = self._sequence(head_updates, "head_updates")
        receipts = [
            self._required_text(ref, "receipt_ref")
            for ref in self._sequence(receipt_refs, "receipt_refs")
        ]
        if not any((evidence_input, atom_input, edge_input, head_input, receipts)):
            raise ValidationError("memory transaction must not be empty")
        request_payload = {
            "profile": self.TRANSACTION_PROFILE,
            "evidence": [dict(item) for item in evidence_input],
            "atoms": [dict(item) for item in atom_input],
            "edges": [dict(item) for item in edge_input],
            "head_updates": [dict(item) for item in head_input],
            "receipt_refs": receipts,
            "scope": dict(transaction_scope),
        }
        ensure_jsonable(request_payload)
        self._mark_foreground_activity(actor)

        with self.store.transaction() as conn:
            prior = self._idempotency_hit(
                conn, actor, idempotency_key, request_payload
            )
            if prior is not None:
                return prior

            prepared_evidence = [
                self._scoped_evidence(item, transaction_scope)
                for item in evidence_input
            ]
            evidence_ids: set[str] = set()
            for item in prepared_evidence:
                evidence_id = str(item["evidence_id"])
                if evidence_id in evidence_ids:
                    raise ValidationError(
                        f"duplicate evidence in transaction: {evidence_id}"
                    )
                evidence_ids.add(evidence_id)

            prepared = [
                self._prepare_committed_atom(
                    self._scoped_atom(item, transaction_scope)
                )
                for item in atom_input
            ]
            prepared_by_id: dict[str, dict[str, Any]] = {}
            for atom in prepared:
                atom_id = str(atom["id"])
                if atom_id in prepared_by_id:
                    raise ValidationError(
                        f"duplicate atom in transaction: {atom_id}"
                    )
                if self.store.get_atom(atom_id) is not None:
                    raise ValidationError(f"atom already exists: {atom_id}")
                prepared_by_id[atom_id] = atom
            for atom in prepared:
                self._assert_direct_commit_governance_safe(
                    atom, candidate_atoms=prepared_by_id
                )
                GovernanceService.assert_constitutional_capability(
                    str(atom["type"]),
                    authorization_context,
                    operation="create",
                )
                GovernanceService.assert_adjudication_capability(
                    atom, authorization_context, actor=actor
                )
                GovernanceService.assert_immutable_primary_record_capability(
                    atom, authorization_context, actor=actor
                )
                content_digest = self._memory_identity_digest(atom)
                tombstone = self.store.get_tombstone(
                    str(atom["id"]), content_digest=content_digest
                )
                if tombstone and tombstone["recreation_policy"] != "allow_recreate":
                    raise ValidationError(
                        f"memory is tombstoned: {atom['id']} / {content_digest}"
                    )
            self._validate_event_refs(prepared, prepared_by_id)

            projected_heads: list[dict[str, Any]] = []
            current_heads: list[dict[str, Any] | None] = []
            seen_heads: set[tuple[str, str]] = set()
            for raw in head_input:
                projected, current = self._validate_head_update(
                    raw,
                    scope=transaction_scope,
                    prepared_by_id=prepared_by_id,
                    seen=seen_heads,
                )
                projected_heads.append(projected)
                current_heads.append(current)
            interaction_events: dict[str, list[str]] = {}
            for atom in prepared:
                if atom.get("type") != "interaction_event":
                    continue
                conversation_id = str(
                    (atom.get("payload") or {}).get("conversation_id") or ""
                )
                interaction_events.setdefault(conversation_id, []).append(
                    str(atom["id"])
                )
            stream_heads = {
                str(head["series_id"]): str(head["head_ref"])
                for head in projected_heads
                if head["series_kind"] == "interaction_stream"
            }
            for conversation_id, event_refs in interaction_events.items():
                if len(event_refs) != 1:
                    raise ValidationError(
                        "a memory transaction may append exactly one "
                        "interaction_event per conversation"
                    )
                if stream_heads.get(conversation_id) != event_refs[0]:
                    raise ValidationError(
                        "every interaction_event requires a matching "
                        "interaction_stream head update"
                    )
            if set(stream_heads) != set(interaction_events):
                raise ValidationError(
                    "an interaction_stream head update requires one matching "
                    "interaction_event in the transaction"
                )

            # A successful CAS mechanically creates the supersession lineage.
            superseded_atoms: list[dict[str, Any]] = []
            for head, current in zip(projected_heads, current_heads):
                if head["series_kind"] not in {
                    "assessment_qualification", "authority_record",
                    "discourse_thread", "goal_work", "project_work",
                }:
                    continue
                if current is None:
                    continue
                prior_ref = str(current["head_ref"])
                successor_ref = str(head["head_ref"])
                prior_atom = self.store.get_atom(prior_ref)
                if prior_atom is None:
                    raise CASConflict(
                        f"indexed memory head is missing canonical atom: {prior_ref}"
                    )
                successor = dict(prepared_by_id[successor_ref])
                successor["supersedes"] = list(
                    dict.fromkeys(
                        [
                            *successor.get("supersedes", []),
                            prior_ref,
                        ]
                    )
                )
                successor = normalize_atom(
                    self._attach_search_index(successor), require_id=True
                )
                prepared_by_id[successor_ref] = successor
                for index, atom in enumerate(prepared):
                    if atom["id"] == successor_ref:
                        prepared[index] = successor
                        break
                superseded_atoms.append(
                    self._superseded_projection(
                        prior_atom,
                        successor_ref=successor_ref,
                        actor=actor,
                    )
                )

            requested_edges: list[dict[str, Any]] = []
            historical_edge_matches_allowed: set[str] = set()
            seen_edges: set[str] = set()
            for atom in prepared:
                for edge in self._intrinsic_edges_for_atom(atom):
                    if edge["edge_id"] not in seen_edges:
                        requested_edges.append(edge)
                        seen_edges.add(str(edge["edge_id"]))
            for index, raw in enumerate(edge_input):
                item = self._mapping(raw, f"edges[{index}]")
                source_ref = self._required_text(
                    item.get("source_ref"), f"edges[{index}].source_ref"
                )
                target_ref = self._required_text(
                    item.get("target_ref"), f"edges[{index}].target_ref"
                )
                if source_ref == target_ref:
                    raise ValidationError("memory transaction edge endpoints must differ")
                for ref in (source_ref, target_ref):
                    atom = prepared_by_id.get(ref) or self.store.get_atom(ref)
                    if atom is None:
                        raise ValidationError(
                            f"memory transaction edge endpoint does not exist: {ref}"
                        )
                    if not self._same_scope(atom.get("scope") or {}, transaction_scope):
                        raise ValidationError(
                            "memory transaction edge endpoint scope must match "
                            "transaction scope"
                        )
                edge = self._edge(
                    source_ref,
                    target_ref,
                    self._required_text(
                        item.get("relation"), f"edges[{index}].relation"
                    ),
                    transaction_scope,
                    evidence_refs=item.get("evidence_refs"),
                    confidence=item.get("confidence"),
                    derivation={
                        "kind": "explicit_memory_transaction",
                        "actor": actor,
                        "source_refs": [
                            self._required_text(
                                ref, f"edges[{index}].source_refs"
                            )
                            for ref in self._sequence(
                                item.get("source_refs"),
                                f"edges[{index}].source_refs",
                            )
                        ],
                    },
                )
                supplied_edge_id = item.get("edge_id")
                if supplied_edge_id is not None:
                    edge["edge_id"] = self._required_text(
                        supplied_edge_id, f"edges[{index}].edge_id"
                    )
                if edge["edge_id"] not in seen_edges:
                    requested_edges.append(edge)
                    seen_edges.add(str(edge["edge_id"]))
                allow_historical_match = item.get(
                    "allow_historical_match", False
                )
                if not isinstance(allow_historical_match, bool):
                    raise ValidationError(
                        f"edges[{index}].allow_historical_match must be a boolean"
                    )
                if allow_historical_match:
                    historical_edge_matches_allowed.add(str(edge["edge_id"]))

            # ``insert_edge`` is deliberately insert-only.  A stable edge id may
            # therefore already identify the requested relationship (for
            # example when two source-bound transactions cite the same
            # relationship).  Journal only the projections that this
            # transaction will actually insert; otherwise replay would apply
            # the later metadata while SQLite correctly retained the first
            # projection.
            projected_edges: list[dict[str, Any]] = []
            resolved_edges: list[dict[str, Any]] = []
            reused_edge_refs: list[str] = []
            historical_edge_refs: list[str] = []
            identity_fields = ("source_ref", "target_ref", "relation", "scope")
            for edge in requested_edges:
                edge_id = str(edge["edge_id"])
                existing = self.store.get_edge(edge_id)
                if existing is None:
                    projected_edges.append(edge)
                    resolved_edges.append(edge)
                    continue
                if any(existing.get(field) != edge.get(field) for field in identity_fields):
                    raise ValidationError(
                        "memory transaction edge_id collides with a different "
                        f"relationship: {edge_id}"
                    )
                if existing.get("deleted"):
                    if edge_id not in historical_edge_matches_allowed:
                        raise ValidationError(
                            "memory transaction cannot silently reactivate a "
                            f"deleted edge: {edge_id}"
                        )
                    resolved_edges.append(existing)
                    historical_edge_refs.append(edge_id)
                    continue
                resolved_edges.append(existing)
                reused_edge_refs.append(edge_id)

            projected_atoms = [*superseded_atoms, *prepared]
            event_payload = {
                "profile": self.TRANSACTION_PROFILE,
                "operation": "commit_memory_transaction",
                "evidence": prepared_evidence,
                "projected_atoms": projected_atoms,
                "projected_edges": projected_edges,
                "reused_edge_refs": reused_edge_refs,
                "historical_edge_refs": historical_edge_refs,
                "projected_heads": projected_heads,
                "receipt_refs": receipts,
                "scope": dict(transaction_scope),
            }
            event = self.store.append_event(
                conn,
                event_type="memory_transaction_committed",
                actor=actor,
                payload=event_payload,
                target_refs=sorted(
                    {
                        *prepared_by_id,
                        *(str(head["head_ref"]) for head in projected_heads),
                    }
                ),
                payload_refs=receipts,
                evidence_refs=sorted(evidence_ids),
                idempotency_key=idempotency_key,
                expected_versions={
                    f"{head['series_kind']}:{head['series_id']}": int(
                        head["head_version"]
                    )
                    - 1
                    for head in projected_heads
                },
                authorization_context=authorization_context,
            )
            for item in prepared_evidence:
                self.store.insert_evidence(conn, item, event["event_id"])
            for atom in superseded_atoms:
                self.store.replace_atom(conn, atom)
            for atom in prepared:
                self.store.insert_atom(conn, atom)
            for edge in projected_edges:
                self.store.insert_edge(conn, edge)
            committed_heads: list[dict[str, Any]] = []
            for head in projected_heads:
                committed = {
                    **head,
                    "journal_event_id": event["event_id"],
                    "updated_at": event["accepted_at"],
                }
                self.store.put_memory_head(conn, committed)
                committed_heads.append(committed)
            self.store.retire_packet_cache(conn)
            response = {
                "status": "committed",
                "profile": self.TRANSACTION_PROFILE,
                "evidence": prepared_evidence,
                "atoms": prepared,
                "superseded_atoms": superseded_atoms,
                "edges": resolved_edges,
                "historical_edge_refs": historical_edge_refs,
                "heads": committed_heads,
                "receipt_refs": receipts,
                "event": event,
                "revision": {
                    "graph_version": int(event["graph_version"]),
                    "journal_head": str(event["checksum"]),
                },
            }
            self._record_idempotency(
                conn, actor, idempotency_key, request_payload, event, response
            )
            return response

    @staticmethod
    def _event_projection(
        atom: Mapping[str, Any], *, alias: str
    ) -> dict[str, Any]:
        payload = atom.get("payload") or {}
        return {
            "alias": alias,
            "atom_ref": atom["id"],
            "conversation_id": payload.get("conversation_id"),
            "sequence": payload.get("sequence"),
            "actor_ref": payload.get("actor_ref"),
            "role": payload.get("role"),
            "content": payload.get("content"),
            "occurred_at": payload.get("occurred_at"),
            "in_reply_to": payload.get("in_reply_to"),
            "visibility": payload.get("visibility"),
            "thread_refs": list(payload.get("thread_refs") or []),
            "source_ref": payload.get("source_ref"),
        }

    @staticmethod
    def _canonical_context_projection(
        atom: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project cognitive content and authority without derived indexes.

        Search indexes, vectors, decay bookkeeping, and revision-history
        internals are rebuildable storage metadata.  Including them in an
        explicitly bound cognitive record consumes context without adding
        source content or authority, and can make a protected workspace exceed
        its budget after only a few turns.
        """

        fields = (
            "id",
            "type",
            "schema_version",
            "payload",
            "evidence_refs",
            "scope",
            "confidence",
            "salience",
            "utility",
            "layer",
            "lifecycle_state",
            "health_status",
            "retention_class",
            "access_policy",
            "created_at",
            "observed_at",
            "updated_at",
            "version",
            "supersedes",
        )
        return {
            field: copy.deepcopy(atom[field])
            for field in fields
            if field in atom
        }

    def _interaction_atoms(
        self,
        *,
        conversation_id: str,
        scope: Mapping[str, Any],
        requester: str,
        target_processor: str,
    ) -> list[dict[str, Any]]:
        atoms: list[dict[str, Any]] = []
        for atom in self.store.list_interaction_atoms(
            conversation_id=conversation_id
        ):
            payload = atom.get("payload") or {}
            if (
                payload.get("conversation_id") != conversation_id
                or not scope_visible(atom.get("scope") or {}, scope)
                or not access_visible(
                    atom.get("access_policy") or {},
                    requester,
                    target_processor,
                )
            ):
                continue
            atoms.append(atom)
        atoms.sort(
            key=lambda atom: (
                int((atom.get("payload") or {}).get("sequence", 0)),
                str(atom.get("id") or ""),
            )
        )
        return atoms

    @staticmethod
    def _visible_interaction(
        atom: Mapping[str, Any] | None,
        *,
        conversation_id: str,
        scope: Mapping[str, Any],
        requester: str,
        target_processor: str,
    ) -> bool:
        if not isinstance(atom, Mapping):
            return False
        payload = atom.get("payload") or {}
        return bool(
            atom.get("type") == "interaction_event"
            and payload.get("conversation_id") == conversation_id
            and not atom.get("deleted")
            and scope_visible(atom.get("scope") or {}, scope)
            and access_visible(
                atom.get("access_policy") or {}, requester, target_processor
            )
        )

    def _reply_chain(
        self,
        current: Mapping[str, Any],
        *,
        conversation_id: str,
        scope: Mapping[str, Any],
        requester: str,
        target_processor: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [dict(current)]
        seen = {str(current["id"])}
        cursor = current
        while len(result) < limit:
            reply_ref = str((cursor.get("payload") or {}).get("in_reply_to") or "")
            if not reply_ref or reply_ref in seen:
                break
            prior = self.store.get_atom(reply_ref)
            if not self._visible_interaction(
                prior,
                conversation_id=conversation_id,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
            ):
                break
            assert prior is not None
            result.append(dict(prior))
            seen.add(reply_ref)
            cursor = prior
        return list(reversed(result))

    def _recent_interaction_window(
        self,
        *,
        conversation_id: str,
        through_sequence: int,
        scope: Mapping[str, Any],
        requester: str,
        target_processor: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Load a bounded visible tail when explicit reply lineage is absent."""

        selected: list[dict[str, Any]] = []
        cursor = max(0, int(through_sequence))
        batch_limit = max(16, int(limit) * 2)
        seen: set[str] = set()
        while cursor > 0 and len(selected) < limit:
            batch = self.store.list_interaction_atoms(
                conversation_id=conversation_id,
                through_sequence=cursor,
                limit=batch_limit,
                newest_first=True,
            )
            if not batch:
                break
            oldest_sequence = cursor
            for atom in batch:
                payload = atom.get("payload") or {}
                sequence = int(payload.get("sequence") or 0)
                oldest_sequence = min(oldest_sequence, sequence)
                atom_ref = str(atom.get("id") or "")
                if atom_ref in seen or not self._visible_interaction(
                    atom,
                    conversation_id=conversation_id,
                    scope=scope,
                    requester=requester,
                    target_processor=target_processor,
                ):
                    continue
                seen.add(atom_ref)
                selected.append(dict(atom))
                if len(selected) >= limit:
                    break
            if len(batch) < batch_limit or oldest_sequence <= 1:
                break
            next_cursor = oldest_sequence - 1
            if next_cursor >= cursor:
                break
            cursor = next_cursor
        selected.sort(
            key=lambda atom: (
                int((atom.get("payload") or {}).get("sequence", 0)),
                str(atom.get("id") or ""),
            )
        )
        return selected[-limit:]

    def _thread_roots(
        self,
        *,
        conversation_id: str,
        scope: Mapping[str, Any],
        requester: str,
        target_processor: str,
    ) -> dict[str, dict[str, Any]]:
        roots: dict[str, dict[str, Any]] = {}
        for atom in self.store.list_discourse_thread_atoms(
            conversation_id=conversation_id
        ):
            payload = atom.get("payload") or {}
            if (
                payload.get("conversation_id") == conversation_id
                and scope_visible(atom.get("scope") or {}, scope)
                and access_visible(
                    atom.get("access_policy") or {},
                    requester,
                    target_processor,
                )
            ):
                roots[str(payload.get("thread_id") or "")] = atom
        return roots

    @staticmethod
    def _workspace_budget(
        token_or_byte_budget: int | Mapping[str, int] | None,
    ) -> dict[str, int | None]:
        if token_or_byte_budget is None:
            return {"bytes": 48_000, "tokens": 12_000, "items": None}
        if isinstance(token_or_byte_budget, bool):
            raise ValidationError("token_or_byte_budget must be positive")
        if isinstance(token_or_byte_budget, int):
            if token_or_byte_budget <= 0:
                raise ValidationError("token_or_byte_budget must be positive")
            return {
                "bytes": int(token_or_byte_budget) * 4,
                "tokens": int(token_or_byte_budget),
                "items": None,
            }
        if not isinstance(token_or_byte_budget, Mapping):
            raise ValidationError("token_or_byte_budget must be an integer or object")
        tokens = token_or_byte_budget.get("tokens")
        byte_limit = token_or_byte_budget.get("bytes")
        item_limit = token_or_byte_budget.get("items")
        if tokens is not None and (
            isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0
        ):
            raise ValidationError("token_or_byte_budget.tokens must be positive")
        if byte_limit is not None and (
            isinstance(byte_limit, bool)
            or not isinstance(byte_limit, int)
            or byte_limit <= 0
        ):
            raise ValidationError("token_or_byte_budget.bytes must be positive")
        if item_limit is not None and (
            isinstance(item_limit, bool)
            or not isinstance(item_limit, int)
            or item_limit <= 0
        ):
            raise ValidationError("token_or_byte_budget.items must be positive")
        if tokens is None and byte_limit is None:
            raise ValidationError(
                "token_or_byte_budget requires tokens or bytes"
            )
        resolved_bytes = int(byte_limit or int(tokens) * 4)
        if tokens is not None:
            resolved_bytes = min(resolved_bytes, int(tokens) * 4)
        return {
            "bytes": resolved_bytes,
            "tokens": int(tokens) if tokens else None,
            "items": int(item_limit) if item_limit else None,
        }

    @staticmethod
    def _workspace_item_count(node: Any) -> int:
        """Count nested list members without interpreting their content."""

        if isinstance(node, Mapping):
            return sum(
                ContinuityService._workspace_item_count(item)
                for item in node.values()
            )
        if isinstance(node, list):
            return len(node) + sum(
                ContinuityService._workspace_item_count(item)
                for item in node
            )
        return 0

    def _context_compaction_projection(
        self,
        *,
        conversation_id: str,
        current_sequence: int,
        scope: Mapping[str, Any],
        requester: str,
        target_processor: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Select the newest verified rolling projection for one stream."""

        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        omissions: list[dict[str, Any]] = []
        for atom in self.store.list_context_compaction_atoms(
            partition_key=conversation_id,
            through_sequence=current_sequence,
            profile=CONTEXT_COMPACTION_PROFILE,
        ):
            if (
                not scope_visible(atom.get("scope") or {}, scope)
                or not access_visible(
                    atom.get("access_policy") or {},
                    requester,
                    target_processor,
                )
            ):
                continue
            payload = atom.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            compaction = payload.get("context_compaction")
            if not isinstance(compaction, Mapping):
                continue
            partition = compaction.get("partition")
            coverage = compaction.get("coverage")
            if (
                compaction.get("profile") != CONTEXT_COMPACTION_PROFILE
                or compaction.get("mode") != "rolling"
                or not isinstance(partition, Mapping)
                or partition.get("kind") != "interaction_stream"
                or partition.get("key") != conversation_id
                or not isinstance(coverage, Mapping)
            ):
                continue
            through_sequence = coverage.get("through_sequence")
            if (
                isinstance(through_sequence, bool)
                or not isinstance(through_sequence, int)
                or through_sequence <= 0
                or through_sequence > current_sequence
            ):
                continue
            source_refs = compaction.get("source_refs")
            if not isinstance(source_refs, list) or not source_refs:
                continue
            sources: list[dict[str, Any]] = []
            source_visible = True
            for ref in source_refs:
                source = self.store.get_atom(str(ref))
                if (
                    source is None
                    or source.get("deleted")
                    or not scope_visible(source.get("scope") or {}, scope)
                    or not access_visible(
                        source.get("access_policy") or {},
                        requester,
                        target_processor,
                    )
                ):
                    source_visible = False
                    break
                sources.append(source)
            atom_ref = str(atom.get("id") or "")
            if not source_visible:
                omissions.append(
                    {
                        "ref": atom_ref,
                        "reason": "compaction_source_unavailable",
                    }
                )
                continue
            through_ref = str(coverage.get("through_ref") or "")
            through_source = next(
                (
                    source
                    for source in sources
                    if str(source.get("id") or "") == through_ref
                ),
                None,
            )
            through_payload = (
                through_source.get("payload")
                if isinstance(through_source, Mapping)
                else {}
            )
            through_payload = (
                through_payload if isinstance(through_payload, Mapping) else {}
            )
            if (
                through_source is None
                or through_source.get("type") != "interaction_event"
                or through_payload.get("conversation_id") != conversation_id
                or through_payload.get("sequence") != through_sequence
            ):
                omissions.append(
                    {
                        "ref": atom_ref,
                        "reason": "compaction_frontier_invalid",
                    }
                )
                continue
            if coverage.get("source_digest") != context_compaction_source_digest(
                sources
            ):
                omissions.append(
                    {
                        "ref": atom_ref,
                        "reason": "compaction_source_digest_mismatch",
                    }
                )
                continue
            projection = {
                "alias": "compacted_context_1",
                "atom_ref": atom_ref,
                "atom_type": "semantic",
                "profile": CONTEXT_COMPACTION_PROFILE,
                "mode": "rolling",
                "partition": dict(partition),
                "summary": str(payload.get("summary") or ""),
                "facets": dict(compaction.get("facets") or {}),
                "coverage": dict(coverage),
                "source_refs": [str(ref) for ref in source_refs],
                "evidence_refs": list(atom.get("evidence_refs") or []),
                "confidence": dict(atom.get("confidence") or {}),
                "epistemic_status": str(
                    payload.get("epistemic_status")
                    or "derived_summary_not_adopted_truth"
                ),
            }
            candidates.append(
                (
                    (
                        through_sequence,
                        int(coverage.get("source_count") or 0),
                        str(atom.get("updated_at") or atom.get("created_at") or ""),
                        atom_ref,
                    ),
                    projection,
                )
            )
        if not candidates:
            return [], omissions
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [candidates[0][1]], omissions

    def compile_cognitive_workspace(
        self,
        *,
        current_event_ref: str,
        conversation_id: str,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        participant_refs: Sequence[str] | None = None,
        operation_refs: Sequence[str] | None = None,
        project_refs: Sequence[str] | None = None,
        context_refs: Sequence[str] | None = None,
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        temporal_limit: int = 12,
        thread_limit: int = 8,
        recent_event_floor: int = 4,
        prior_workspace_revision: Mapping[str, Any] | None = None,
        include_associative_memory: bool = True,
        run_policy: bool = False,
        _policy_already_run: bool = False,
    ) -> dict[str, Any]:
        """Compile a bounded generated view anchored on one canonical event."""

        current_event_ref = self._required_text(
            current_event_ref, "current_event_ref"
        )
        conversation_id = self._required_text(conversation_id, "conversation_id")
        requester = self._required_text(requester, "requester")
        target_processor = self._required_text(
            target_processor, "target_processor"
        )
        request_scope = normalize_scope(scope)
        budget = self._workspace_budget(token_or_byte_budget)
        temporal_limit = max(2, min(64, int(temporal_limit)))
        thread_limit = max(1, min(32, int(thread_limit)))
        recent_event_floor = max(
            2, min(temporal_limit, int(recent_event_floor))
        )
        participants = [
            self._required_text(ref, "participant_ref")
            for ref in self._sequence(participant_refs, "participant_refs")
        ]
        operation_refs = [
            self._required_text(ref, "operation_ref")
            for ref in self._sequence(operation_refs, "operation_refs")
        ]
        project_refs = [
            self._required_text(ref, "project_ref")
            for ref in self._sequence(project_refs, "project_refs")
        ]
        context_refs = [
            self._required_text(ref, "context_ref")
            for ref in self._sequence(context_refs, "context_refs")
        ]
        if prior_workspace_revision is not None:
            self._mapping(prior_workspace_revision, "prior_workspace_revision")
        if not isinstance(include_associative_memory, bool):
            raise ValidationError("include_associative_memory must be a boolean")
        self._mark_foreground_activity(requester)
        if run_policy and not _policy_already_run:
            self.reasoning.run_memory_policy(
                trigger="compile_cognitive_workspace", scope=request_scope
            )

        revision = self.store.memory_revision()
        current = self.store.get_atom(current_event_ref)
        if current is None or current.get("type") != "interaction_event":
            raise ValidationError(
                "current_event_ref must identify an interaction_event atom"
            )
        current_payload = current.get("payload") or {}
        current_sequence = int(current_payload.get("sequence") or 0)
        if current_payload.get("conversation_id") != conversation_id:
            raise ValidationError(
                "current event conversation_id does not match request"
            )
        if not scope_visible(current.get("scope") or {}, request_scope):
            raise ValidationError("current event is outside the requested scope")
        if not access_visible(
            current.get("access_policy") or {}, requester, target_processor
        ):
            raise ValidationError("current event is not visible to this processor")

        omissions: list[dict[str, Any]] = []
        canonical_context: list[dict[str, Any]] = []
        seen_context_refs: set[str] = set()
        requested_context = [
            *[("context", ref) for ref in context_refs],
            *[("operation", ref) for ref in operation_refs],
            *[("project", ref) for ref in project_refs],
        ]
        alias_ordinals = {"context": 0, "operation": 0, "project": 0}
        for context_kind, ref in requested_context:
            if ref in seen_context_refs:
                continue
            seen_context_refs.add(ref)
            alias_ordinals[context_kind] += 1
            atom = self.store.get_atom(ref)
            if (
                atom is None
                or atom.get("deleted")
                or not scope_visible(atom.get("scope") or {}, request_scope)
                or not access_visible(
                    atom.get("access_policy") or {},
                    requester,
                    target_processor,
                )
            ):
                omissions.append(
                    {
                        "ref": ref,
                        "reason": f"{context_kind}_ref_unavailable",
                    }
                )
                continue
            canonical_context.append(
                {
                    "alias": (
                        f"{context_kind}_{alias_ordinals[context_kind]}"
                    ),
                    "atom_ref": ref,
                    "context_kind": context_kind,
                    "record": self._canonical_context_projection(atom),
                }
            )

        temporal = self._reply_chain(
            current,
            conversation_id=conversation_id,
            scope=request_scope,
            requester=requester,
            target_processor=target_processor,
            limit=temporal_limit,
        )
        temporal_refs = {str(atom["id"]) for atom in temporal}
        # If an event has no explicit reply pointer, preserve the immediately
        # preceding exchange.  This is sequence closure, not semantic routing.
        if len(temporal) == 1:
            temporal = self._recent_interaction_window(
                conversation_id=conversation_id,
                through_sequence=current_sequence,
                scope=request_scope,
                requester=requester,
                target_processor=target_processor,
                limit=min(4, temporal_limit),
            )
            temporal_refs = {str(atom["id"]) for atom in temporal}

        compacted_context, compaction_omissions = (
            self._context_compaction_projection(
                conversation_id=conversation_id,
                current_sequence=current_sequence,
                scope=request_scope,
                requester=requester,
                target_processor=target_processor,
            )
        )
        omissions.extend(compaction_omissions)

        roots = self._thread_roots(
            conversation_id=conversation_id,
            scope=request_scope,
            requester=requester,
            target_processor=target_processor,
        )
        directly_linked_threads: set[str] = set()
        for atom in temporal:
            directly_linked_threads.update(
                str(ref)
                for ref in (atom.get("payload") or {}).get("thread_refs", [])
                if str(ref)
            )
        root_by_ref = {str(atom["id"]): thread_id for thread_id, atom in roots.items()}
        for edge in self.store.list_edges_for_refs(sorted(temporal_refs)):
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source in temporal_refs and target in root_by_ref:
                directly_linked_threads.add(root_by_ref[target])
            if target in temporal_refs and source in root_by_ref:
                directly_linked_threads.add(root_by_ref[source])

        thread_candidates: list[dict[str, Any]] = []
        for head in self.store.list_memory_heads(
            scope=request_scope,
            series_kind="discourse_thread",
            series_ids=tuple(roots),
        ):
            thread_id = str(head["series_id"])
            root = roots.get(thread_id)
            state = self.store.get_atom(str(head["head_ref"]))
            if root is None or state is None:
                omissions.append(
                    {
                        "ref": str(head.get("head_ref") or ""),
                        "reason": "head_or_root_missing",
                    }
                )
                continue
            if not access_visible(
                state.get("access_policy") or {}, requester, target_processor
            ):
                omissions.append(
                    {
                        "ref": str(state["id"]),
                        "reason": "access_hidden",
                    }
                )
                continue
            payload = state.get("payload") or {}
            source_refs = {
                str(ref) for ref in payload.get("source_event_refs", [])
            }
            head_refs = {
                str(ref) for ref in payload.get("head_event_refs", [])
            }
            direct = bool(
                thread_id in directly_linked_threads
                or temporal_refs.intersection(source_refs)
                or temporal_refs.intersection(head_refs)
            )
            thread_candidates.append(
                {
                    "direct": direct,
                    "thread_id": thread_id,
                    "root": root,
                    "state": state,
                    "head": head,
                }
            )
        thread_candidates.sort(
            key=lambda item: (
                0 if item["direct"] else 1,
                0
                if (item["state"].get("payload") or {}).get("lifecycle") == "open"
                else 1,
                {
                    "foreground": 0,
                    "background": 1,
                    "dormant": 2,
                }.get(
                    str(
                        (item["state"].get("payload") or {}).get(
                            "attention_state"
                        )
                    ),
                    3,
                ),
                -int(item["head"]["head_version"]),
                item["thread_id"],
            )
        )
        selected_threads = thread_candidates[:thread_limit]

        temporal_projection: list[dict[str, Any]] = []
        for index, atom in enumerate(temporal):
            alias = (
                "event_current"
                if atom["id"] == current_event_ref
                else f"event_recent_{index + 1}"
            )
            temporal_projection.append(self._event_projection(atom, alias=alias))
        thread_projection: list[dict[str, Any]] = []
        for index, item in enumerate(selected_threads):
            state = item["state"]
            payload = state.get("payload") or {}
            state_basis: list[dict[str, Any]] = []
            seen_basis_refs: set[str] = set()
            for private_state in payload.get("private_state") or []:
                if not isinstance(private_state, Mapping):
                    continue
                for raw_ref in private_state.get("basis_refs") or []:
                    basis_ref = str(raw_ref or "")
                    if not basis_ref or basis_ref in seen_basis_refs:
                        continue
                    seen_basis_refs.add(basis_ref)
                    basis = self.store.get_atom(basis_ref)
                    if (
                        basis is None
                        or basis.get("deleted")
                        or not scope_visible(basis.get("scope") or {}, request_scope)
                        or not access_visible(
                            basis.get("access_policy") or {},
                            requester,
                            target_processor,
                        )
                    ):
                        continue
                    basis_payload = basis.get("payload") or {}
                    state_basis.append({
                        "atom_ref": basis_ref,
                        "atom_type": str(basis.get("type") or ""),
                        "content": str(basis_payload.get("content") or "")[:1000],
                    })
            thread_projection.append(
                {
                    "alias": f"thread_{index + 1}",
                    "thread_id": item["thread_id"],
                    "root_ref": item["root"]["id"],
                    "state_ref": state["id"],
                    "head_version": int(item["head"]["head_version"]),
                    "directly_linked": bool(item["direct"]),
                    "lifecycle": payload.get("lifecycle"),
                    "attention_state": payload.get("attention_state"),
                    "summary": payload.get("summary"),
                    "participants": list(payload.get("participants") or []),
                    "head_event_refs": list(payload.get("head_event_refs") or []),
                    "source_event_refs": list(
                        payload.get("source_event_refs") or []
                    ),
                    "shared_state": list(payload.get("shared_state") or []),
                    "private_state": list(payload.get("private_state") or []),
                    "state_basis": state_basis,
                    "unresolved_items": list(
                        payload.get("unresolved_items") or []
                    ),
                }
            )

        associative: dict[str, Any] | None = None
        associative_error: str | None = None
        associative_budget = max(4_096, min(16_000, int(budget["bytes"]) // 3))
        associative_request_budget: dict[str, int] = {
            "bytes": associative_budget
        }
        if budget["items"] is not None:
            fixed_projection_items = self._workspace_item_count(
                {
                    "current_event": next(
                        item
                        for item in temporal_projection
                        if item["atom_ref"] == current_event_ref
                    ),
                    "temporal_context": temporal_projection,
                    "compacted_context": compacted_context,
                    "thread_heads": thread_projection,
                    "canonical_context": canonical_context,
                    "available_new_thread_aliases": [
                        f"new_thread_{index}" for index in range(1, 5)
                    ],
                    "bound_refs": {
                        "participant_refs": participants,
                        "operation_refs": operation_refs,
                        "project_refs": project_refs,
                        "context_refs": context_refs,
                    },
                    "protected_refs": [
                        *[item["atom_ref"] for item in temporal_projection],
                        *[item["atom_ref"] for item in compacted_context],
                        *[item["root_ref"] for item in thread_projection],
                        *[item["state_ref"] for item in thread_projection],
                        *[
                            basis["atom_ref"]
                            for item in thread_projection
                            for basis in item.get("state_basis") or []
                        ],
                        *[item["atom_ref"] for item in canonical_context],
                    ],
                    "omissions": omissions,
                }
            )
            associative_request_budget["items"] = max(
                64,
                int(budget["items"]) - fixed_projection_items - 16,
            )
        need = str(current_payload.get("content") or "").strip() or "current interaction"
        if include_associative_memory:
            try:
                associative = self.reasoning.compile_memory_frame(
                    need=need,
                    purpose="support current interaction from canonical memory",
                    depth="working_frame",
                    task_context={
                        "conversation_id": conversation_id,
                        "participant_refs": participants,
                        "operation_refs": operation_refs,
                        "project_refs": project_refs,
                        "context_refs": context_refs,
                        "protected_atom_refs": [
                            current_event_ref,
                            *[item["state_ref"] for item in thread_projection],
                            *[item["atom_ref"] for item in canonical_context],
                        ],
                    },
                    scope=request_scope,
                    requester=requester,
                    target_processor=target_processor,
                    memory_mode="operational_recall",
                    token_or_byte_budget=associative_request_budget,
                    run_policy=False,
                )
            except ValidationError as exc:
                associative_error = str(exc)

        request_binding = {
            "current_event_ref": current_event_ref,
            "conversation_id": conversation_id,
            "scope": dict(request_scope),
            "requester": requester,
            "target_processor": target_processor,
            "participant_refs": participants,
            "operation_refs": operation_refs,
            "project_refs": project_refs,
            "context_refs": context_refs,
            # Preserve canonical refs for the established full-workspace
            # request. The opt-out is material and participates in the binding;
            # the default must not perturb existing workspace identities.
            **(
                {"include_associative_memory": False}
                if not include_associative_memory
                else {}
            ),
            "revision": revision,
        }
        page_aliases = [
            {
                "alias": f"memory_page_{index + 1}",
                "page_id": str(page.get("page_id") or ""),
                "descriptor_digest": str(
                    page.get("descriptor_digest") or ""
                ),
            }
            for index, page in enumerate(
                (
                    associative.get("page_index", [])
                    if isinstance(associative, Mapping)
                    else []
                )[:8]
            )
            if isinstance(page, Mapping)
        ]
        workspace = {
            "status": "compiled",
            "profile": self.WORKSPACE_PROFILE,
            "schema_version": SCHEMA_VERSION,
            "workspace_id": stable_id("workspace", request_binding),
            "revision": revision,
            "generated_at": utc_now(),
            "request": {
                "request_digest": digest(request_binding),
                "current_event_ref": current_event_ref,
                "conversation_id": conversation_id,
                "requester": requester,
                "target_processor": target_processor,
                **(
                    {"include_associative_memory": False}
                    if not include_associative_memory
                    else {}
                ),
            },
            "current_event": next(
                item
                for item in temporal_projection
                if item["atom_ref"] == current_event_ref
            ),
            "temporal_context": temporal_projection,
            "compacted_context": compacted_context,
            "thread_heads": thread_projection,
            "canonical_context": canonical_context,
            "available_new_thread_aliases": [
                f"new_thread_{index}" for index in range(1, 5)
            ],
            "associative_memory": associative,
            "page_aliases": page_aliases,
            "bound_refs": {
                "participant_refs": participants,
                "operation_refs": operation_refs,
                "project_refs": project_refs,
                "context_refs": context_refs,
            },
            "protected_refs": [
                *[item["atom_ref"] for item in temporal_projection],
                *[item["atom_ref"] for item in compacted_context],
                *[item["root_ref"] for item in thread_projection],
                *[item["state_ref"] for item in thread_projection],
                *[
                    basis["atom_ref"]
                    for item in thread_projection
                    for basis in item.get("state_basis") or []
                ],
                *[item["atom_ref"] for item in canonical_context],
            ],
            "omissions": [
                *omissions,
                *(
                    [
                        {
                            "reason": "thread_limit",
                            "count": len(thread_candidates) - len(selected_threads),
                        }
                    ]
                    if len(thread_candidates) > len(selected_threads)
                    else []
                ),
                *(
                    [
                        {
                            "reason": "associative_compilation_unavailable",
                            "detail": associative_error,
                        }
                    ]
                    if associative_error
                    else []
                ),
                *(
                    [{"reason": "associative_memory_deferred"}]
                    if not include_associative_memory
                    else []
                ),
            ],
            "budget": {
                "limit_bytes": int(budget["bytes"]),
                "limit_tokens": budget["tokens"],
                "limit_items": budget["items"],
                "used_bytes": 0,
                "estimated_tokens": 0,
                "used_items": 0,
            },
        }

        def finalize(value: dict[str, Any]) -> dict[str, Any]:
            # Budget fields contribute to their own serialized size. Iterate
            # until the recorded projection size reaches a fixed point.
            for _ in range(8):
                encoded = canonical_json(value).encode("utf-8")
                used_bytes = len(encoded)
                estimated_tokens = (used_bytes + 3) // 4
                used_items = self._workspace_item_count(value)
                if (
                    value["budget"]["used_bytes"] == used_bytes
                    and value["budget"]["estimated_tokens"]
                    == estimated_tokens
                    and value["budget"]["used_items"] == used_items
                ):
                    break
                value["budget"]["used_bytes"] = used_bytes
                value["budget"]["estimated_tokens"] = estimated_tokens
                value["budget"]["used_items"] = used_items
            return value

        def exceeds_budget(value: Mapping[str, Any]) -> bool:
            return (
                int(value["budget"]["used_bytes"]) > int(budget["bytes"])
                or (
                    budget["items"] is not None
                    and int(value["budget"]["used_items"])
                    > int(budget["items"])
                )
            )

        def guaranteed_minimum_budget(value: Mapping[str, Any]) -> dict[str, int]:
            """Return retry budgets that include their own envelope growth.

            Budget metadata is part of the canonical workspace serialization.
            Replacing a four-digit limit with a five-digit minimum can therefore
            make an otherwise exact retry one byte too small.  Compute the byte
            and token alternatives independently to a fixed point so either
            advertised form is sufficient for an unchanged request.
            """

            minimum_items = max(1, int(value["budget"]["used_items"]))

            token_candidate = copy.deepcopy(dict(value))
            minimum_tokens = max(1, int(value["budget"]["estimated_tokens"]))
            for _ in range(16):
                token_candidate["budget"]["limit_tokens"] = minimum_tokens
                token_candidate["budget"]["limit_bytes"] = minimum_tokens * 4
                token_candidate["budget"]["limit_items"] = minimum_items
                token_candidate = finalize(token_candidate)
                required_tokens = max(
                    minimum_tokens,
                    int(token_candidate["budget"]["estimated_tokens"]),
                )
                required_items = max(
                    minimum_items,
                    int(token_candidate["budget"]["used_items"]),
                )
                if (
                    int(token_candidate["budget"]["used_bytes"])
                    <= required_tokens * 4
                    and int(token_candidate["budget"]["used_items"])
                    <= required_items
                    and required_tokens == minimum_tokens
                    and required_items == minimum_items
                ):
                    break
                minimum_tokens = required_tokens
                minimum_items = required_items

            byte_candidate = copy.deepcopy(dict(value))
            minimum_bytes = max(1, int(value["budget"]["used_bytes"]))
            for _ in range(16):
                byte_candidate["budget"]["limit_tokens"] = None
                byte_candidate["budget"]["limit_bytes"] = minimum_bytes
                byte_candidate["budget"]["limit_items"] = minimum_items
                byte_candidate = finalize(byte_candidate)
                required_bytes = max(
                    minimum_bytes,
                    int(byte_candidate["budget"]["used_bytes"]),
                )
                required_items = max(
                    minimum_items,
                    int(byte_candidate["budget"]["used_items"]),
                )
                if (
                    required_bytes == minimum_bytes
                    and required_items == minimum_items
                ):
                    break
                minimum_bytes = required_bytes
                minimum_items = required_items

            return {
                "bytes": minimum_bytes,
                "tokens": minimum_tokens,
                "items": minimum_items,
            }

        workspace = finalize(workspace)
        if exceeds_budget(workspace) and compacted_context:
            through_sequence = int(
                compacted_context[0]["coverage"]["through_sequence"]
            )
            preserved_temporal_refs = {
                str(item["atom_ref"])
                for item in temporal_projection[-recent_event_floor:]
            }
            preserved_temporal_refs.add(current_event_ref)
            removable = [
                item
                for item in workspace["temporal_context"]
                if str(item["atom_ref"]) not in preserved_temporal_refs
                and int(item.get("sequence") or 0) <= through_sequence
            ]
            if removable:
                removed_refs = {
                    str(item["atom_ref"]) for item in removable
                }
                workspace["temporal_context"] = [
                    item
                    for item in workspace["temporal_context"]
                    if str(item["atom_ref"]) not in removed_refs
                ]
                workspace["protected_refs"] = [
                    ref
                    for ref in workspace["protected_refs"]
                    if ref not in removed_refs
                ]
                workspace["omissions"].append(
                    {
                        "reason": "covered_temporal_context_compacted",
                        "count": len(removed_refs),
                        "compaction_ref": compacted_context[0]["atom_ref"],
                        "covered_through_sequence": through_sequence,
                    }
                )
                workspace = finalize(workspace)
        while (
            exceeds_budget(workspace)
            and workspace["thread_heads"]
            and not workspace["thread_heads"][-1]["directly_linked"]
        ):
            removed = workspace["thread_heads"].pop()
            removed_refs = {
                removed["root_ref"],
                removed["state_ref"],
                *(
                    str(item.get("atom_ref") or "")
                    for item in removed.get("state_basis") or []
                    if isinstance(item, Mapping)
                ),
            }
            workspace["protected_refs"] = [
                ref
                for ref in workspace["protected_refs"]
                if ref not in removed_refs
            ]
            workspace["omissions"].append(
                {
                    "reason": "workspace_budget_thread_omitted",
                    "ref": removed["state_ref"],
                }
            )
            workspace = finalize(workspace)
        if exceeds_budget(workspace) and associative is not None:
            workspace["associative_memory"] = None
            workspace["page_aliases"] = []
            associative = None
            workspace["omissions"].append(
                {"reason": "workspace_budget_associative_memory_omitted"}
            )
            workspace = finalize(workspace)
        if exceeds_budget(workspace):
            raise CognitiveWorkspaceBudgetExceeded(
                limit_bytes=workspace["budget"]["limit_bytes"],
                limit_tokens=workspace["budget"]["limit_tokens"],
                limit_items=workspace["budget"]["limit_items"],
                used_bytes=workspace["budget"]["used_bytes"],
                estimated_tokens=workspace["budget"]["estimated_tokens"],
                used_items=workspace["budget"]["used_items"],
                minimum_budget=guaranteed_minimum_budget(workspace),
            )
        current_revision = self.store.memory_revision()
        if current_revision != revision:
            raise StaleFrameError(revision, current_revision)
        return workspace

    def compile_interaction_projection(
        self,
        *,
        conversation_id: str,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "participant-ui",
        after_sequence: int = 0,
        limit: int = 1000,
        linked_atom_types: Sequence[str] | None = None,
        linked_depth: int = 0,
        linked_limit_per_event: int = 32,
    ) -> dict[str, Any]:
        """Project canonical interaction history for a disposable read model.

        This ordered, access-filtered view performs no associative ranking and
        makes no semantic selection. Clients can rebuild transcript caches
        without treating a cache as memory authority.
        """

        conversation_id = self._required_text(conversation_id, "conversation_id")
        requester = self._required_text(requester, "requester")
        target_processor = self._required_text(
            target_processor, "target_processor"
        )
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValidationError(
                "after_sequence must be a non-negative integer"
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 5000
        ):
            raise ValidationError("limit must be an integer in [1, 5000]")
        requested_linked_types: tuple[str, ...] = tuple(
            dict.fromkeys(
                self._required_text(value, "linked_atom_types item")
                for value in (linked_atom_types or ())
            )
        )
        if len(requested_linked_types) > 16:
            raise ValidationError("linked_atom_types may contain at most 16 items")
        if (
            isinstance(linked_depth, bool)
            or not isinstance(linked_depth, int)
            or not 0 <= linked_depth <= 2
        ):
            raise ValidationError("linked_depth must be an integer in [0, 2]")
        if bool(requested_linked_types) != bool(linked_depth):
            raise ValidationError(
                "linked_atom_types and linked_depth must be supplied together"
            )
        if (
            isinstance(linked_limit_per_event, bool)
            or not isinstance(linked_limit_per_event, int)
            or not 1 <= linked_limit_per_event <= 128
        ):
            raise ValidationError(
                "linked_limit_per_event must be an integer in [1, 128]"
            )
        request_scope = normalize_scope(scope)
        self._mark_foreground_activity(requester)
        selected = [
            atom
            for atom in self._interaction_atoms(
                conversation_id=conversation_id,
                scope=request_scope,
                requester=requester,
                target_processor=target_processor,
            )
            if int((atom.get("payload") or {}).get("sequence", 0))
            > after_sequence
        ][: limit + 1]
        has_more = len(selected) > limit
        events = selected[:limit]
        projected_events: list[dict[str, Any]] = []
        for index, atom in enumerate(events):
            projected = self._event_projection(
                atom, alias=f"event_{index + 1}"
            )
            projected["linked_records"] = self._linked_record_projection(
                event_ref=str(atom["id"]),
                scope=request_scope,
                requester=requester,
                target_processor=target_processor,
                atom_types=set(requested_linked_types),
                depth=linked_depth,
                limit=linked_limit_per_event,
            )
            projected_events.append(projected)
        return {
            "status": "compiled",
            "profile": self.INTERACTION_PROJECTION_PROFILE,
            "schema_version": SCHEMA_VERSION,
            "conversation_id": conversation_id,
            "events": projected_events,
            "after_sequence": after_sequence,
            "next_after_sequence": (
                int((events[-1].get("payload") or {}).get("sequence", 0))
                if events
                else after_sequence
            ),
            "has_more": has_more,
            "revision": self.store.memory_revision(),
            "generated_at": utc_now(),
        }

    def _linked_record_projection(
        self,
        *,
        event_ref: str,
        scope: Mapping[str, Any],
        requester: str,
        target_processor: str,
        atom_types: set[str],
        depth: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return explicitly requested, access-filtered graph descendants.

        The traversal follows only outgoing canonical edges and only through
        atom types named by the caller. It performs no ranking, semantic
        classification, or operation-specific interpretation.
        """

        if not atom_types or depth < 1:
            return []
        frontier = [event_ref]
        seen = {event_ref}
        projected: list[dict[str, Any]] = []
        for hop in range(1, depth + 1):
            next_frontier: list[str] = []
            edges = sorted(
                self.store.list_edges_for_refs(frontier),
                key=lambda item: (
                    str(item.get("source_ref") or ""),
                    str(item.get("relation") or ""),
                    str(item.get("target_ref") or ""),
                ),
            )
            frontier_set = set(frontier)
            for edge in edges:
                source_ref = str(edge.get("source_ref") or "")
                target_ref = str(edge.get("target_ref") or "")
                if (
                    source_ref not in frontier_set
                    or not target_ref
                    or target_ref in seen
                ):
                    continue
                target = self.store.get_atom(target_ref)
                if (
                    target is None
                    or target.get("deleted")
                    or str(target.get("type") or "") not in atom_types
                    or not scope_visible(target.get("scope") or {}, scope)
                    or not access_visible(
                        target.get("access_policy") or {},
                        requester,
                        target_processor,
                    )
                ):
                    continue
                seen.add(target_ref)
                next_frontier.append(target_ref)
                projected.append(
                    {
                        "depth": hop,
                        "source_ref": source_ref,
                        "relation": str(edge.get("relation") or ""),
                        "record": self._canonical_context_projection(target),
                    }
                )
                if len(projected) >= limit:
                    return projected
            frontier = next_frontier
            if not frontier:
                break
        return projected

    def get_memory_head(
        self,
        *,
        scope: Mapping[str, Any] | None,
        series_kind: str,
        series_id: str,
        requester: str = "system",
        target_processor: str = "reasoner",
    ) -> dict[str, Any]:
        """Return an access-filtered canonical head pointer without atom content."""

        request_scope = normalize_scope(scope)
        requester = self._required_text(requester, "requester")
        target_processor = self._required_text(
            target_processor, "target_processor"
        )
        series_kind = self._required_text(series_kind, "series_kind")
        series_id = self._required_text(series_id, "series_id")
        if series_kind not in self.SUPPORTED_HEAD_KINDS:
            raise ValidationError(f"unsupported head series_kind: {series_kind}")
        self._mark_foreground_activity(requester)
        head = self.store.get_memory_head(
            scope=request_scope,
            series_kind=series_kind,
            series_id=series_id,
        )
        if head is None:
            return {
                "status": "absent",
                "profile": self.MEMORY_HEAD_PROFILE,
                "series_kind": series_kind,
                "series_id": series_id,
                "revision": self.store.memory_revision(),
            }
        atom = self.store.get_atom(str(head["head_ref"]))
        if (
            atom is None
            or atom.get("deleted")
            or not scope_visible(atom.get("scope") or {}, request_scope)
            or not access_visible(
                atom.get("access_policy") or {},
                requester,
                target_processor,
            )
        ):
            return {
                "status": "absent",
                "profile": self.MEMORY_HEAD_PROFILE,
                "series_kind": series_kind,
                "series_id": series_id,
                "revision": self.store.memory_revision(),
            }
        return {
            "status": "found",
            "profile": self.MEMORY_HEAD_PROFILE,
            "series_kind": series_kind,
            "series_id": series_id,
            "head_ref": str(head["head_ref"]),
            "head_version": int(head["head_version"]),
            "journal_event_id": str(head["journal_event_id"]),
            "updated_at": str(head["updated_at"]),
            "revision": self.store.memory_revision(),
        }

    def observe_memory_transaction(
        self,
        *,
        event_id: str,
        scope: Mapping[str, Any] | None,
        requester: str = "system",
        target_processor: str = "reasoner",
    ) -> dict[str, Any]:
        """Return a bounded, access-filtered projection of one journal commit.

        This is an audit read, not semantic retrieval.  It exposes the exact
        transaction boundary (record kinds, head updates, and receipt IDs)
        without returning the potentially very large canonical payloads.
        """

        event_id = self._required_text(event_id, "event_id")
        request_scope = normalize_scope(scope)
        requester = self._required_text(requester, "requester")
        target_processor = self._required_text(
            target_processor, "target_processor"
        )
        self._mark_foreground_activity(requester)
        event = self.store.get_event(event_id)
        absent = {
            "status": "absent",
            "profile": self.MEMORY_TRANSACTION_OBSERVATION_PROFILE,
            "event_id": event_id,
            "revision": self.store.memory_revision(),
        }
        if event is None or event.get("event_type") != "memory_transaction_committed":
            return absent
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        event_scope = normalize_scope(payload.get("scope"))
        if not scope_visible(event_scope, request_scope):
            return absent

        projected_atoms = [
            dict(item)
            for item in payload.get("projected_atoms") or ()
            if isinstance(item, Mapping)
        ]
        visible_atoms = [
            item
            for item in projected_atoms
            if scope_visible(item.get("scope") or {}, request_scope)
            and access_visible(
                item.get("access_policy") or {},
                requester,
                target_processor,
            )
        ]
        visible_atom_refs = {
            str(item.get("id") or "") for item in visible_atoms if item.get("id")
        }
        compacted_receipt = bool(event.get("storage_compacted"))
        projected_evidence = [
            dict(item)
            for item in payload.get(
                "projected_evidence" if compacted_receipt else "evidence"
            )
            or ()
            if isinstance(item, Mapping)
        ]
        visible_evidence = [
            item
            for item in projected_evidence
            if scope_visible(item.get("scope") or {}, request_scope)
            and access_visible(
                item.get("access_policy") or {},
                requester,
                target_processor,
            )
        ]
        visible_evidence_refs = {
            str(item.get("evidence_id") or "")
            for item in visible_evidence
            if item.get("evidence_id")
        }
        projected_heads = [
            dict(item)
            for item in payload.get("projected_heads") or ()
            if isinstance(item, Mapping)
        ]
        visible_heads = [
            item
            for item in projected_heads
            if scope_visible(item.get("scope") or {}, request_scope)
            and str(item.get("head_ref") or "") in visible_atom_refs
        ]

        integrity_body = {
            key: event.get(key)
            for key in (
                "event_id", "event_type", "schema_version", "actor",
                "target_refs", "payload", "payload_refs", "evidence_refs",
                "idempotency_key", "payload_digest", "causal_parent_ids",
                "expected_versions", "authorization_context", "occurred_at",
                "accepted_at", "result_status", "projection_status",
                "previous_event_hash", "graph_version",
            )
        }
        checksum_valid = (
            bool(event.get("compact_receipt_verified"))
            if compacted_receipt
            else (
                str(event.get("payload_digest") or "") == digest(payload)
                and str(event.get("checksum") or "") == digest(integrity_body)
            )
        )
        complete_visibility = (
            len(visible_atoms) == len(projected_atoms)
            and len(visible_evidence) == len(projected_evidence)
            and len(visible_heads) == len(projected_heads)
        )
        return {
            "status": "found",
            "profile": self.MEMORY_TRANSACTION_OBSERVATION_PROFILE,
            "verification_status": (
                "mechanically_verified" if checksum_valid else "invalid_checksum"
            ),
            "verification_basis": (
                "compacted_receipt_linked_to_verified_event_checksum"
                if compacted_receipt
                else "retained_event_payload_checksum"
            ),
            "storage_compacted": compacted_receipt,
            "complete_visibility": complete_visibility,
            "event_id": event_id,
            "event_type": str(event.get("event_type") or ""),
            "actor": str(event.get("actor") or ""),
            "idempotency_key": str(event.get("idempotency_key") or "") or None,
            "occurred_at": str(event.get("occurred_at") or ""),
            "accepted_at": str(event.get("accepted_at") or ""),
            "result_status": str(event.get("result_status") or ""),
            "projection_status": str(event.get("projection_status") or ""),
            "graph_version": int(event.get("graph_version") or 0),
            "payload_digest": str(event.get("payload_digest") or ""),
            "journal_checksum": str(event.get("checksum") or ""),
            "previous_event_hash": str(event.get("previous_event_hash") or ""),
            "operation": str(payload.get("operation") or ""),
            "projected_atoms": [
                {
                    "atom_ref": str(item.get("id") or ""),
                    "type": str(item.get("type") or ""),
                    "profile": str(
                        item.get("profile")
                        if compacted_receipt
                        else (item.get("payload") or {}).get("profile")
                        or ""
                    ),
                    "retention_class": str(item.get("retention_class") or ""),
                    "lifecycle_state": str(item.get("lifecycle_state") or ""),
                    "health_status": str(item.get("health_status") or ""),
                    "payload_digest": str(item.get("payload_digest") or "")
                    if compacted_receipt
                    else digest(item.get("payload") or {}),
                }
                for item in visible_atoms
            ],
            "projected_heads": [
                {
                    "series_kind": str(item.get("series_kind") or ""),
                    "series_id": str(item.get("series_id") or ""),
                    "head_ref": str(item.get("head_ref") or ""),
                    "head_version": int(item.get("head_version") or 0),
                }
                for item in visible_heads
            ],
            "projected_evidence": [
                {
                    "evidence_id": str(item.get("evidence_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "source_ref": str(item.get("source_ref") or ""),
                    "captured_at": str(item.get("captured_at") or ""),
                    "checksum": str(item.get("checksum") or ""),
                }
                for item in visible_evidence
            ],
            "receipt_refs": [
                str(ref)
                for ref in payload.get("receipt_refs") or ()
                if str(ref) in visible_evidence_refs
                or str(ref) in visible_atom_refs
            ],
            "counts": {
                "projected_atoms": len(projected_atoms),
                "visible_atoms": len(visible_atoms),
                "projected_edges": int(payload.get("projected_edge_count", 0) or 0)
                if compacted_receipt
                else len(payload.get("projected_edges") or ()),
                "projected_heads": len(projected_heads),
                "visible_heads": len(visible_heads),
                "projected_evidence": len(projected_evidence),
                "visible_evidence": len(visible_evidence),
            },
            "revision": self.store.memory_revision(),
        }

    def get_memory_series_versions(
        self,
        *,
        scope: Mapping[str, Any] | None,
        series_kind: str,
        series_id: str,
        versions: Sequence[int] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Resolve exact historical heads without associative retrieval.

        The append-only transaction journal remains authoritative.  The store's
        history table is a rebuildable projection of each committed head update.
        Historical atoms still pass the same scope and access checks as the
        canonical head, and no semantic claim in their payload is endorsed.
        """

        request_scope = normalize_scope(scope)
        requester = self._required_text(requester, "requester")
        target_processor = self._required_text(
            target_processor, "target_processor"
        )
        series_kind = self._required_text(series_kind, "series_kind")
        series_id = self._required_text(series_id, "series_id")
        if series_kind not in self.SUPPORTED_HEAD_KINDS:
            raise ValidationError(f"unsupported head series_kind: {series_kind}")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValidationError("limit must be an integer between 1 and 1000")
        requested_versions: list[int] = []
        if versions is not None:
            if isinstance(versions, (str, bytes)) or not isinstance(versions, Sequence):
                raise ValidationError("versions must be an array of positive integers")
            for value in versions:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValidationError(
                        "versions must contain only positive integers"
                    )
                if value not in requested_versions:
                    requested_versions.append(value)
            if len(requested_versions) > 100:
                raise ValidationError("versions may contain at most 100 entries")
        self._mark_foreground_activity(requester)
        rows = self.store.list_memory_head_history(
            scope=request_scope,
            series_kind=series_kind,
            series_id=series_id,
            versions=requested_versions or None,
            limit=max(limit, len(requested_versions)),
        )
        items: list[dict[str, Any]] = []
        for row in rows:
            atom = self.store.get_atom(str(row["head_ref"]))
            if (
                atom is None
                or atom.get("deleted")
                or not scope_visible(atom.get("scope") or {}, request_scope)
                or not access_visible(
                    atom.get("access_policy") or {},
                    requester,
                    target_processor,
                )
            ):
                continue
            items.append({
                "series_kind": series_kind,
                "series_id": series_id,
                "head_ref": str(row["head_ref"]),
                "head_version": int(row["head_version"]),
                "journal_event_id": str(row["journal_event_id"]),
                "updated_at": str(row["updated_at"]),
                "payload_digest": digest(atom.get("payload") or {}),
                "lifecycle_state": str(atom.get("lifecycle_state") or ""),
                "health_status": str(atom.get("health_status") or ""),
            })
        visible_versions = {int(item["head_version"]) for item in items}
        missing_versions = [
            version
            for version in requested_versions
            if version not in visible_versions
        ]
        return {
            "status": "found" if items else "absent",
            "profile": self.MEMORY_SERIES_VERSIONS_PROFILE,
            "series_kind": series_kind,
            "series_id": series_id,
            "requested_versions": requested_versions,
            "items": items,
            "missing_versions": missing_versions,
            "complete": bool(requested_versions) and not missing_versions,
            "revision": self.store.memory_revision(),
        }

    def rebuild_memory_heads(self) -> dict[str, Any]:
        heads = self.store.rebuild_memory_heads()
        return {
            "status": "rebuilt",
            "profile": "amos.memory-head-index.v1",
            "head_count": len(heads),
            "heads": heads,
            "revision": self.store.memory_revision(),
        }
