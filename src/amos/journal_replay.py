"""Deterministic AMOS journal replay and snapshot state helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .schemas import digest


EdgeMigrator = Callable[[str], dict[str, Any]]


def empty_replay_state() -> dict[str, Any]:
    return {
        "atoms": {},
        "edges": {},
        "heads": {},
        "head_history": {},
        "tombstones": {},
        "known_edge_ids": set(),
    }


def normalize_replay_state(
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = dict(value or {})
    return {
        "atoms": dict(value.get("atoms") or {}),
        "edges": dict(value.get("edges") or {}),
        "heads": dict(value.get("heads") or {}),
        "head_history": dict(value.get("head_history") or {}),
        "tombstones": dict(value.get("tombstones") or {}),
        "known_edge_ids": {
            str(item) for item in value.get("known_edge_ids", []) if str(item)
        },
    }


def serializable_replay_state(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_replay_state(state)
    return {
        "schema": "amos.journal-replay-state.v1",
        "atoms": normalized["atoms"],
        "edges": normalized["edges"],
        "heads": normalized["heads"],
        "head_history": normalized["head_history"],
        "tombstones": normalized["tombstones"],
        "known_edge_ids": sorted(normalized["known_edge_ids"]),
    }


def replay_events(
    events: Sequence[Mapping[str, Any]],
    *,
    initial_state: Mapping[str, Any] | None = None,
    migrated_edge_derivation: EdgeMigrator,
) -> dict[str, Any]:
    state = normalize_replay_state(initial_state)
    atoms: dict[str, dict[str, Any]] = state["atoms"]
    edges: dict[str, dict[str, Any]] = state["edges"]
    heads: dict[str, dict[str, Any]] = state["heads"]
    head_history: dict[str, dict[str, Any]] = state["head_history"]
    tombstones: dict[str, dict[str, Any]] = state["tombstones"]
    known_edge_ids: set[str] = state["known_edge_ids"]

    def replay_edge_projection(edge: Mapping[str, Any]) -> dict[str, Any]:
        projected = dict(edge)
        derivation = projected.get("derivation")
        if not isinstance(derivation, dict) or not derivation:
            projected["derivation"] = migrated_edge_derivation(
                str(projected.get("relation") or "")
            )
        return projected

    def replay_inserted_edge(edge: Mapping[str, Any]) -> None:
        edge_id = str(edge["edge_id"])
        if edge_id in known_edge_ids:
            return
        known_edge_ids.add(edge_id)
        if not edge.get("deleted"):
            edges[edge_id] = replay_edge_projection(edge)

    def replay_upserted_edge(edge: Mapping[str, Any]) -> None:
        edge_id = str(edge["edge_id"])
        known_edge_ids.add(edge_id)
        if edge.get("deleted"):
            edges.pop(edge_id, None)
        else:
            edges[edge_id] = replay_edge_projection(edge)

    def replay_legacy_retrieval_edge_feedback(payload: Mapping[str, Any]) -> None:
        if payload.get("projected_edges"):
            return
        feedback = payload.get("feedback") or {}
        summaries = feedback.get("updated_edges") or []
        if not summaries:
            return
        projected_atoms = payload.get("projected_atoms") or []
        timestamp = None
        label = None
        for atom in projected_atoms:
            telemetry = (atom.get("decay_policy") or {}).get(
                "retrieval_telemetry"
            ) or {}
            timestamp = telemetry.get("last_outcome_at") or atom.get("updated_at")
            label = telemetry.get("last_outcome_label")
            if timestamp:
                break
        for summary in summaries:
            edge_id = str(summary.get("edge_id") or "")
            prior = edges.get(edge_id)
            if not edge_id or prior is None:
                continue
            changed = dict(prior)
            derivation = dict(changed.get("derivation") or {})
            telemetry = dict(derivation.get("retrieval_telemetry") or {})
            telemetry.update(
                {
                    "used_count": int(summary.get("used_count", 0) or 0),
                    "correction_count": int(
                        summary.get("correction_count", 0) or 0
                    ),
                    "last_outcome_label": label,
                    "last_outcome_at": timestamp,
                }
            )
            derivation["retrieval_telemetry"] = telemetry
            changed["derivation"] = derivation
            if timestamp:
                changed["updated_at"] = timestamp
            changed["version"] = int(changed.get("version", 0) or 0) + 1
            edges[edge_id] = replay_edge_projection(changed)

    for raw_event in events:
        event = dict(raw_event)
        payload = dict(event.get("payload") or {})
        event_type = str(event.get("event_type") or "")
        if event_type == "atom_committed":
            atom = dict(payload["atom"])
            atoms[str(atom["id"])] = atom
            for edge in payload.get("projected_edges", []):
                replay_inserted_edge(edge)
        elif event_type == "atom_updated":
            atom = dict(payload["after"])
            atoms[str(atom["id"])] = atom
            for edge in payload.get("projected_edges", []):
                replay_upserted_edge(edge)
        elif event_type == "atom_deleted":
            before = dict(payload["before"])
            atoms.pop(str(before["id"]), None)
            tombstone = dict(payload["tombstone"])
            tombstones[str(tombstone["target_ref"])] = tombstone
            for edge in payload.get("projected_edges", []):
                replay_upserted_edge(edge)
        elif event_type == "memories_distilled":
            atom = dict(payload["atom"])
            atoms[str(atom["id"])] = atom
            for edge in payload.get("projected_edges", []):
                replay_inserted_edge(edge)
        elif event_type == "edge_committed":
            for edge in payload.get("projected_edges", []):
                replay_upserted_edge(edge)
        elif event_type == "memory_transaction_committed":
            for atom in payload.get("projected_atoms", []):
                atom = dict(atom)
                if atom.get("deleted"):
                    atoms.pop(str(atom["id"]), None)
                else:
                    atoms[str(atom["id"])] = atom
            for edge in payload.get("projected_edges", []):
                replay_inserted_edge(edge)
            for raw_head in payload.get("projected_heads", []):
                head = dict(raw_head)
                scope_digest = digest(head.get("scope") or {})
                series_key = (
                    f"{scope_digest}:{head.get('series_kind')}:"
                    f"{head.get('series_id')}"
                )
                projected_head = {
                    **head,
                    "scope_digest": scope_digest,
                    "journal_event_id": event["event_id"],
                    "updated_at": event["accepted_at"],
                }
                heads[series_key] = projected_head
                history_key = f"{series_key}:{int(head['head_version'])}"
                head_history[history_key] = projected_head
        elif event_type in {
            "atom_merged",
            "proposal_ratified",
            "proposal_resolved",
            "constitutional_record_replaced",
            "steward_run",
            "retrieval_outcome_recorded",
            "memory_reference_contract_repaired",
            "decay_policy_applied",
            "storage_cleanup_run",
        }:
            for atom in payload.get("projected_atoms", []):
                atom = dict(atom)
                if atom.get("deleted"):
                    atoms.pop(str(atom["id"]), None)
                else:
                    atoms[str(atom["id"])] = atom
            for edge in payload.get("projected_edges", []):
                if event_type == "atom_merged":
                    replay_inserted_edge(edge)
                else:
                    replay_upserted_edge(edge)
            if event_type == "retrieval_outcome_recorded":
                replay_legacy_retrieval_edge_feedback(payload)
            for tombstone in payload.get("tombstones", []):
                tombstone = dict(tombstone)
                tombstones[str(tombstone["target_ref"])] = tombstone

    return state
