from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from amos import (
    CASConflict,
    CognitiveWorkspaceBudgetExceeded,
    CONTEXT_COMPACTION_PROFILE,
    ValidationError,
    context_compaction_source_digest,
)
from amos.http_api import AmosHTTPServer

SCOPE = {"tenant": "continuity-test", "conversation": "main"}
PRIMARY_VISIBILITY = {
    "visibility": ["processor:primary-reasoner"],
    "mutable_by": ["owner"],
    "sensitivity": "private",
}


def interaction(
    atom_id: str,
    *,
    sequence: int,
    role: str,
    content: str,
    in_reply_to: str | None = None,
) -> dict:
    return {
        "id": atom_id,
        "type": "interaction_event",
        "payload": {
            "profile": "amos.interaction-event.v1",
            "conversation_id": "main",
            "sequence": sequence,
            "actor_ref": f"{role}:participant",
            "role": role,
            "content": content,
            "occurred_at": f"2026-07-29T00:00:{sequence:02d}Z",
            "in_reply_to": in_reply_to,
            "visibility": "shared",
            "source_ref": f"source:{atom_id}",
            "thread_refs": [],
        },
    }


def interaction_stream_head(
    new_head_ref: str,
    *,
    expected_head_ref: str | None,
    expected_head_version: int,
    conversation_id: str = "main",
) -> dict:
    return {
        "series_kind": "interaction_stream",
        "series_id": conversation_id,
        "new_head_ref": new_head_ref,
        "expected_head_ref": expected_head_ref,
        "expected_head_version": expected_head_version,
    }


def project_state(
    atom_id: str,
    *,
    project_ref: str,
    revision: int,
    status: str,
) -> dict:
    return {
        "id": atom_id,
        "type": "goal",
        "payload": {
            "profile": "example.project-work-object.v1",
            "project_ref": project_ref,
            "revision": revision,
            "objective": "Exercise canonical project continuity.",
            "goal_status": status,
            "owner": "agent:test",
        },
    }


def project_head(
    new_head_ref: str,
    *,
    project_ref: str,
    expected_head_ref: str | None,
    expected_head_version: int,
) -> dict:
    return {
        "series_kind": "project_work",
        "series_id": project_ref,
        "new_head_ref": new_head_ref,
        "expected_head_ref": expected_head_ref,
        "expected_head_version": expected_head_version,
    }


def goal_state(atom_id: str, *, goal_ref: str, revision: int) -> dict:
    return {
        "id": atom_id,
        "type": "goal",
        "payload": {
            "profile": "example.generic-goal-work.v1",
            "goal_ref": goal_ref,
            "revision": revision,
            "objective": "Exercise generic canonical goal continuity.",
            "status": "active",
        },
    }


def goal_head(
    new_head_ref: str,
    *,
    goal_ref: str,
    expected_head_ref: str | None,
    expected_head_version: int,
) -> dict:
    return {
        "series_kind": "goal_work",
        "series_id": goal_ref,
        "new_head_ref": new_head_ref,
        "expected_head_ref": expected_head_ref,
        "expected_head_version": expected_head_version,
    }


def authority_record(
    atom_id: str,
    *,
    series_id: str,
    revision: int,
    checksum: str,
    supersedes: tuple[str, ...] = (),
) -> dict:
    atom = {
        "id": atom_id,
        "type": "procedure",
        "scope": dict(SCOPE),
        "payload": {
            "profile": "example.authority-record.v1",
            "trigger_context": "reconcile an installed immutable package",
            "steps": ["verify checksum", "apply current standing"],
            "authority_series_id": series_id,
            "authority_revision": revision,
            "checksum": checksum,
        },
    }
    if supersedes:
        atom["supersedes"] = list(supersedes)
    return atom


def authority_head(
    new_head_ref: str,
    *,
    series_id: str,
    expected_head_ref: str | None,
    expected_head_version: int,
    legacy_predecessor_ref: str | None = None,
) -> dict:
    value = {
        "series_kind": "authority_record",
        "series_id": series_id,
        "new_head_ref": new_head_ref,
        "expected_head_ref": expected_head_ref,
        "expected_head_version": expected_head_version,
    }
    if legacy_predecessor_ref is not None:
        value["legacy_predecessor_ref"] = legacy_predecessor_ref
    return value


def test_authority_record_heads_preserve_superseded_checksum_revisions(amos):
    series_id = "plugin:example"
    amos.commit_atom(
        authority_record(
            "legacy_authority_state",
            series_id=series_id,
            revision=0,
            checksum="sha256:legacy",
        ),
        idempotency_key="legacy-authority-state",
    )
    first = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="authority-test",
        idempotency_key="authority-head-1",
        atoms=[authority_record(
            "authority_state_1",
            series_id=series_id,
            revision=1,
            checksum="sha256:old",
        )],
        head_updates=[authority_head(
            "authority_state_1",
            series_id=series_id,
            expected_head_ref=None,
            expected_head_version=0,
            legacy_predecessor_ref="legacy_authority_state",
        )],
    )
    assert first["heads"][0]["head_version"] == 1
    assert amos.store.get_atom("legacy_authority_state")["lifecycle_state"] == (
        "superseded"
    )

    second = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="authority-test",
        idempotency_key="authority-head-2",
        atoms=[authority_record(
            "authority_state_2",
            series_id=series_id,
            revision=2,
            checksum="sha256:new",
            supersedes=("authority_state_1",),
        )],
        head_updates=[authority_head(
            "authority_state_2",
            series_id=series_id,
            expected_head_ref="authority_state_1",
            expected_head_version=1,
        )],
    )

    assert second["heads"][0]["head_version"] == 2
    assert amos.get_memory_head(
        scope=SCOPE,
        series_kind="authority_record",
        series_id=series_id,
    )["head_ref"] == "authority_state_2"
    assert amos.store.get_atom("authority_state_1")["lifecycle_state"] == "superseded"
    assert amos.store.get_atom("authority_state_1")["payload"]["checksum"] == "sha256:old"
    assert amos.store.get_atom("authority_state_2")["payload"]["checksum"] == "sha256:new"


def test_canonical_record_batch_returns_coherent_head_atoms(amos):
    series_id = "plugin:batch-example"
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="authority-test",
        idempotency_key="authority-batch-head-1",
        atoms=[authority_record(
            "authority_batch_state_1",
            series_id=series_id,
            revision=1,
            checksum="sha256:batch",
        )],
        head_updates=[authority_head(
            "authority_batch_state_1",
            series_id=series_id,
            expected_head_ref=None,
            expected_head_version=0,
        )],
    )
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="authority-test",
        idempotency_key="authority-batch-private-1",
        atoms=[{
            "id": "authority_batch_private",
            "type": "belief",
            "payload": {"claim": "private canonical record"},
            "access_policy": PRIMARY_VISIBILITY,
        }],
    )

    batch = amos.get_canonical_records(
        atom_ids=["authority_batch_state_1", "missing_batch_atom"],
        heads=[
            {"series_kind": "authority_record", "series_id": series_id},
            {
                "series_kind": "authority_record",
                "series_id": "plugin:missing-batch-example",
            },
        ],
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        include_archived=True,
        include_low_health=True,
        include_superseded=True,
    )

    assert batch["status"] == "completed"
    assert batch["profile"] == "amos.canonical-record-batch.v1"
    assert batch["packet_id"].startswith("pkt_")
    assert batch["request"]["retrieval_mode"] == "exact_batch"
    assert batch["items_by_id"]["authority_batch_state_1"]["payload"][
        "checksum"
    ] == "sha256:batch"
    assert batch["atoms"][1] == {
        "status": "not_found",
        "atom_id": "missing_batch_atom",
        "reason": "not_found",
    }
    assert batch["heads"][0]["head_ref"] == "authority_batch_state_1"
    assert batch["items_by_id"][batch["heads"][0]["head_ref"]]["id"] == (
        "authority_batch_state_1"
    )
    assert batch["heads"][1]["status"] == "absent"

    outcome = amos.record_retrieval_outcome(
        packet_id=batch["packet_id"],
        request=batch["request"],
        outcome={
            "status": "used",
            "materially_used": True,
            "used_atom_refs": ["authority_batch_state_1"],
            "used_evidence_refs": [],
            "summary": "The exact batch record materially conditioned the result.",
        },
    )
    assert outcome["status"] == "recorded"
    assert outcome["feedback"]["ignored_non_packet_refs"] == []
    assert outcome["feedback"]["positive_refs"] == [
        "authority_batch_state_1"
    ]

    hidden = amos.get_canonical_records(
        atom_ids=["authority_batch_private"],
        heads=[],
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        include_archived=True,
        include_low_health=True,
        include_superseded=True,
    )
    assert hidden["items_by_id"] == {}
    assert hidden["atoms"][0]["reason"] == "access_hidden"


def test_generic_goal_work_heads_are_typed_versioned_and_superseding(amos):
    first = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="goal-test",
        idempotency_key="goal-head-1",
        atoms=[goal_state("goal_state_1", goal_ref="goal:self-directed", revision=1)],
        head_updates=[goal_head(
            "goal_state_1",
            goal_ref="goal:self-directed",
            expected_head_ref=None,
            expected_head_version=0,
        )],
    )
    assert first["heads"][0]["head_version"] == 1

    second = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="goal-test",
        idempotency_key="goal-head-2",
        atoms=[goal_state("goal_state_2", goal_ref="goal:self-directed", revision=2)],
        head_updates=[goal_head(
            "goal_state_2",
            goal_ref="goal:self-directed",
            expected_head_ref="goal_state_1",
            expected_head_version=1,
        )],
    )
    assert second["heads"][0]["head_version"] == 2
    assert amos.get_memory_head(
        scope=SCOPE,
        series_kind="goal_work",
        series_id="goal:self-directed",
    )["head_ref"] == "goal_state_2"
    assert amos.store.get_atom("goal_state_1")["lifecycle_state"] == "superseded"


def test_superseded_head_predecessor_leaves_hot_index_and_is_archived(amos):
    goal_ref = "goal:superseded-index-lifecycle"
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="goal-lifecycle-test",
        atoms=[goal_state("goal_lifecycle_1", goal_ref=goal_ref, revision=1)],
        head_updates=[goal_head(
            "goal_lifecycle_1",
            goal_ref=goal_ref,
            expected_head_ref=None,
            expected_head_version=0,
        )],
    )
    assert "goal_lifecycle_1" in amos.store.candidate_atom_ids_for_tokens(
        ["exercise"]
    )
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="goal-lifecycle-test",
        atoms=[goal_state("goal_lifecycle_2", goal_ref=goal_ref, revision=2)],
        head_updates=[goal_head(
            "goal_lifecycle_2",
            goal_ref=goal_ref,
            expected_head_ref="goal_lifecycle_1",
            expected_head_version=1,
        )],
    )

    assert amos.store.get_atom("goal_lifecycle_1")["lifecycle_state"] == (
        "superseded"
    )
    assert "goal_lifecycle_1" not in amos.store.candidate_atom_ids_for_tokens(
        ["exercise"]
    )
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={
            "enabled": True,
            "archive_superseded": True,
            "archive_superseded_after_seconds": 0,
        },
        storage_cleanup={"enabled": False},
    )

    result = amos.run_memory_policy(force=True, trigger="head_lifecycle_test")

    assert result["results"]["decay"]["actions"][0]["atom_ref"] == (
        "goal_lifecycle_1"
    )
    assert amos.store.get_atom("goal_lifecycle_1")["lifecycle_state"] == (
        "archived"
    )
    assert amos.store.get_atom("goal_lifecycle_2")["lifecycle_state"] == "active"
    assert amos.verify_replay()["status"] == "ok"


def test_project_work_heads_are_typed_versioned_and_cas_guarded(amos):
    first = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="project-test",
        idempotency_key="project-head-1",
        atoms=[project_state(
            "project_state_1",
            project_ref="project:continuity",
            revision=1,
            status="adopted_unplanned",
        )],
        head_updates=[project_head(
            "project_state_1",
            project_ref="project:continuity",
            expected_head_ref=None,
            expected_head_version=0,
        )],
    )
    assert first["heads"][0]["head_version"] == 1

    with pytest.raises(CASConflict):
        amos.commit_memory_transaction(
            scope=SCOPE,
            actor="project-test",
            idempotency_key="project-head-stale",
            atoms=[project_state(
                "project_state_stale",
                project_ref="project:continuity",
                revision=2,
                status="planned",
            )],
            head_updates=[project_head(
                "project_state_stale",
                project_ref="project:continuity",
                expected_head_ref=None,
                expected_head_version=0,
            )],
        )

    second = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="project-test",
        idempotency_key="project-head-2",
        atoms=[project_state(
            "project_state_2",
            project_ref="project:continuity",
            revision=2,
            status="planned",
        )],
        head_updates=[project_head(
            "project_state_2",
            project_ref="project:continuity",
            expected_head_ref="project_state_1",
            expected_head_version=1,
        )],
    )
    assert second["heads"][0]["head_version"] == 2
    head = amos.get_memory_head(
        scope=SCOPE,
        series_kind="project_work",
        series_id="project:continuity",
    )
    stored_head = amos.store.get_memory_head(
        scope=SCOPE,
        series_kind="project_work",
        series_id="project:continuity",
    )
    assert head["head_ref"] == "project_state_2"
    assert head["journal_event_id"] == stored_head["journal_event_id"]
    assert head["updated_at"] == stored_head["updated_at"]
    assert amos.store.get_atom("project_state_1")["lifecycle_state"] == "superseded"

    history = amos.get_memory_series_versions(
        scope=SCOPE,
        series_kind="project_work",
        series_id="project:continuity",
        versions=[1, 2, 3],
    )
    assert history["status"] == "found"
    assert history["complete"] is False
    assert history["missing_versions"] == [3]
    assert [item["head_version"] for item in history["items"]] == [1, 2]
    assert [item["head_ref"] for item in history["items"]] == [
        "project_state_1",
        "project_state_2",
    ]
    assert all(item["journal_event_id"] for item in history["items"])
    assert all(len(item["payload_digest"]) == 64 for item in history["items"])

    observed = amos.observe_memory_transaction(
        event_id=second["event"]["event_id"],
        scope=SCOPE,
        requester="project-auditor",
        target_processor="test",
    )
    assert observed["status"] == "found"
    assert observed["profile"] == "amos.memory-transaction-observation.v1"
    assert observed["verification_status"] == "mechanically_verified"
    assert observed["complete_visibility"] is True
    assert observed["projected_heads"] == [{
        "series_kind": "project_work",
        "series_id": "project:continuity",
        "head_ref": "project_state_2",
        "head_version": 2,
    }]
    assert {
        (item["atom_ref"], item["profile"])
        for item in observed["projected_atoms"]
    } >= {("project_state_2", "example.project-work-object.v1")}
    assert observed["counts"]["projected_heads"] == 1

    amos.store.conn.execute("DELETE FROM amos_memory_head_history")
    amos.rebuild_memory_heads()
    rebuilt = amos.get_memory_series_versions(
        scope=SCOPE,
        series_kind="project_work",
        series_id="project:continuity",
        versions=[1, 2],
    )
    assert rebuilt["complete"] is True
    assert [item["head_ref"] for item in rebuilt["items"]] == [
        "project_state_1",
        "project_state_2",
    ]


def test_memory_transaction_reuses_existing_edge_without_rejournal_projection(amos):
    amos.commit_memory_atoms(
        [
            {
                "id": "edge_reuse_source",
                "type": "belief",
                "payload": {"claim": "a"},
                "scope": dict(SCOPE),
            },
            {
                "id": "edge_reuse_target",
                "type": "belief",
                "payload": {"claim": "b"},
                "scope": dict(SCOPE),
            },
        ],
        actor="continuity-test",
        idempotency_key="edge-reuse-atoms",
    )
    request_edge = {
        "source_ref": "edge_reuse_source",
        "target_ref": "edge_reuse_target",
        "relation": "rel:supports",
        "confidence": {"level": "medium", "score": 0.6},
    }
    first = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="continuity-test",
        idempotency_key="edge-reuse-first",
        edges=[request_edge],
    )
    stored = dict(first["edges"][0])

    second = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="continuity-test",
        idempotency_key="edge-reuse-second",
        edges=[
            {
                **request_edge,
                "confidence": {"level": "high", "score": 0.95},
            }
        ],
    )

    assert second["edges"] == [stored]
    assert second["event"]["payload"]["projected_edges"] == []
    assert second["event"]["payload"]["reused_edge_refs"] == [stored["edge_id"]]
    assert amos.verify_replay()["status"] == "ok"


def test_replay_mirrors_legacy_insert_only_duplicate_edge_projection(amos):
    amos.commit_memory_atoms(
        [
            {
                "id": "legacy_edge_source",
                "type": "belief",
                "payload": {"claim": "a"},
                "scope": dict(SCOPE),
            },
            {
                "id": "legacy_edge_target",
                "type": "belief",
                "payload": {"claim": "b"},
                "scope": dict(SCOPE),
            },
        ],
        actor="continuity-test",
        idempotency_key="legacy-edge-atoms",
    )
    first = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="continuity-test",
        idempotency_key="legacy-edge-first",
        edges=[{
            "source_ref": "legacy_edge_source",
            "target_ref": "legacy_edge_target",
            "relation": "rel:supports",
        }],
    )
    legacy_projection = {
        **first["edges"][0],
        "confidence": {"level": "high", "score": 0.99},
        "updated_at": "2099-01-01T00:00:00Z",
        "version": 99,
    }
    with amos.store.transaction() as conn:
        amos.store.append_event(
            conn,
            event_type="memory_transaction_committed",
            actor="legacy-writer",
            payload={
                "profile": "amos.memory-transaction.v1",
                "operation": "commit_memory_transaction",
                "evidence": [],
                "projected_atoms": [],
                "projected_edges": [legacy_projection],
                "projected_heads": [],
                "receipt_refs": [],
                "scope": dict(SCOPE),
            },
            target_refs=["legacy_edge_source", "legacy_edge_target"],
        )

    assert amos.store.get_edge(legacy_projection["edge_id"]) == first["edges"][0]
    assert amos.verify_replay()["status"] == "ok"


def thread_root(atom_id: str, *, opening_event_ref: str) -> dict:
    return {
        "id": atom_id,
        "type": "discourse_thread",
        "payload": {
            "profile": "amos.discourse-thread.v1",
            "thread_id": atom_id,
            "conversation_id": "main",
            "opened_by_event_ref": opening_event_ref,
            "participants": ["human:participant", "agent:participant"],
        },
    }


def thread_state(
    atom_id: str,
    *,
    thread_id: str,
    revision: int,
    event_refs: list[str],
    secret: str = "orchid",
) -> dict:
    return {
        "id": atom_id,
        "type": "discourse_state",
        "payload": {
            "profile": "amos.discourse-thread-state.v2",
            "thread_id": thread_id,
            "revision": revision,
            "lifecycle": "open",
            "attention_state": "foreground",
            "summary": "A cooperative open-vocabulary activity",
            "participants": ["human:participant", "agent:participant"],
            "head_event_refs": [event_refs[-1]],
            "source_event_refs": event_refs,
            "shared_state": [
                {
                    "key": "protocol",
                    "value": "The participant supplies bounded questions",
                    "state_class": "reported_fact",
                    "authority": "discourse",
                    "basis_refs": [event_refs[0]],
                }
            ],
            "private_state": [
                {
                    "key": "concealed_value",
                    "value": secret,
                    "state_class": "private_secret",
                    "authority": "discourse",
                    "basis_refs": [event_refs[-1]],
                }
            ],
            "unresolved_items": [
                {
                    "kind": "expected_move",
                    "description": "Await the next participant turn",
                    "basis_refs": [event_refs[-1]],
                }
            ],
        },
        "access_policy": PRIMARY_VISIBILITY,
    }


def test_atomic_interaction_thread_head_and_workspace_visibility(amos, monkeypatch):
    first = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="participant-ingress",
        idempotency_key="human-1",
        evidence=[
            {
                "source_type": "interaction",
                "source_ref": "transport:human-1",
                "payload": {"text": r"Let \(x \in \mathbb{R}\) and begin."},
            }
        ],
        atoms=[
            interaction(
                "event_human_1",
                sequence=1,
                role="human",
                content=r"Let \(x \in \mathbb{R}\) and begin.",
            )
        ],
        head_updates=[
            interaction_stream_head(
                "event_human_1",
                expected_head_ref=None,
                expected_head_version=0,
            )
        ],
    )
    assert first["status"] == "committed"
    assert first["atoms"][0]["payload"]["content"] == r"Let \(x \in \mathbb{R}\) and begin."

    response = interaction(
        "event_agent_2",
        sequence=2,
        role="agent",
        content="Ready.",
        in_reply_to="event_human_1",
    )
    response["payload"]["thread_refs"] = ["thread_open_1"]
    opened = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="primary-reasoner",
        idempotency_key="agent-2",
        atoms=[
            response,
            thread_root("thread_open_1", opening_event_ref="event_human_1"),
            thread_state(
                "thread_state_1",
                thread_id="thread_open_1",
                revision=1,
                event_refs=["event_human_1", "event_agent_2"],
            ),
        ],
        edges=[
            {
                "source_ref": "event_agent_2",
                "target_ref": "event_human_1",
                "relation": "rel:responds_to",
            },
            {
                "source_ref": "event_agent_2",
                "target_ref": "thread_open_1",
                "relation": "rel:opens",
            },
            {
                "source_ref": "thread_state_1",
                "target_ref": "thread_open_1",
                "relation": "rel:part_of",
            },
        ],
        head_updates=[
            {
                "series_kind": "discourse_thread",
                "series_id": "thread_open_1",
                "new_head_ref": "thread_state_1",
                "expected_head_ref": None,
                "expected_head_version": 0,
            },
            interaction_stream_head(
                "event_agent_2",
                expected_head_ref="event_human_1",
                expected_head_version=1,
            ),
        ],
    )
    assert opened["heads"][0]["head_version"] == 1

    human_followup = interaction(
        "event_human_3",
        sequence=3,
        role="human",
        content="Is it alive?",
        in_reply_to="event_agent_2",
    )
    human_followup["payload"]["thread_refs"] = ["thread_open_1"]
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="participant-ingress",
        idempotency_key="human-3",
        atoms=[human_followup],
        edges=[
            {
                "source_ref": "event_human_3",
                "target_ref": "event_agent_2",
                "relation": "rel:responds_to",
            },
            {
                "source_ref": "event_human_3",
                "target_ref": "thread_open_1",
                "relation": "rel:continues",
            },
        ],
        head_updates=[
            interaction_stream_head(
                "event_human_3",
                expected_head_ref="event_agent_2",
                expected_head_version=2,
            )
        ],
    )

    def reject_unbounded_atom_scan():
        raise AssertionError("cognitive workspace must not scan every AMOS atom")

    monkeypatch.setattr(amos.store, "list_atoms", reject_unbounded_atom_scan)

    workspace = amos.compile_cognitive_workspace(
        current_event_ref="event_human_3",
        conversation_id="main",
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        participant_refs=["human:participant", "agent:participant"],
        token_or_byte_budget={"bytes": 48_000, "items": 512},
    )
    assert workspace["profile"] == "amos.cognitive-workspace.v1"
    assert [item["atom_ref"] for item in workspace["temporal_context"]] == [
        "event_human_1",
        "event_agent_2",
        "event_human_3",
    ]
    assert workspace["thread_heads"][0]["thread_id"] == "thread_open_1"
    assert workspace["thread_heads"][0]["private_state"][0]["value"] == "orchid"
    assert workspace["thread_heads"][0]["state_basis"] == [
        {
            "atom_ref": "event_agent_2",
            "atom_type": "interaction_event",
            "content": "Ready.",
        }
    ]
    assert workspace["budget"]["limit_items"] == 512
    assert workspace["budget"]["used_items"] <= 512

    deferred = amos.compile_cognitive_workspace(
        current_event_ref="event_human_3",
        conversation_id="main",
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        participant_refs=["human:participant", "agent:participant"],
        token_or_byte_budget={"bytes": 48_000, "items": 512},
        include_associative_memory=False,
    )
    assert deferred["request"]["include_associative_memory"] is False
    assert deferred["associative_memory"] is None
    assert {item["reason"] for item in deferred["omissions"]} >= {
        "associative_memory_deferred"
    }

    with pytest.raises(
        CognitiveWorkspaceBudgetExceeded,
        match="too small for protected cognitive workspace context",
    ) as overflow:
        amos.compile_cognitive_workspace(
            current_event_ref="event_human_3",
            conversation_id="main",
            scope=SCOPE,
            requester="agent:participant",
            target_processor="primary-reasoner",
            participant_refs=["human:participant", "agent:participant"],
            token_or_byte_budget={"bytes": 48_000, "items": 1},
        )
    assert overflow.value.exceeded_dimensions == ["items"]
    assert overflow.value.budget["limit_items"] == 1
    assert overflow.value.minimum_budget["items"] > 1

    contextual = amos.compile_cognitive_workspace(
        current_event_ref="event_human_3",
        conversation_id="main",
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        context_refs=["event_agent_2"],
        token_or_byte_budget={"bytes": 24_000},
    )
    context_record = contextual["canonical_context"][0]["record"]
    assert context_record["payload"]["content"] == "Ready."
    assert "index_refs" not in context_record
    assert "revision_history" not in context_record
    assert "decay_policy" not in context_record

    shared_view = amos.compile_cognitive_workspace(
        current_event_ref="event_human_3",
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        token_or_byte_budget={"bytes": 24_000},
    )
    assert shared_view["thread_heads"] == []
    assert any(
        item["reason"] == "access_hidden" for item in shared_view["omissions"]
    )
    assert "orchid" not in json.dumps(shared_view)
    hidden_head = amos.get_memory_head(
        scope=SCOPE,
        series_kind="discourse_thread",
        series_id="thread_open_1",
        requester="human:participant",
        target_processor="participant-ui",
    )
    assert hidden_head["status"] == "absent"
    authorized_head = amos.get_memory_head(
        scope=SCOPE,
        series_kind="discourse_thread",
        series_id="thread_open_1",
        requester="agent:participant",
        target_processor="primary-reasoner",
    )
    assert authorized_head["status"] == "found"
    assert authorized_head["head_ref"] == "thread_state_1"

    projection = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
    )
    assert projection["profile"] == "amos.interaction-projection.v2"
    assert [item["content"] for item in projection["events"]] == [
        r"Let \(x \in \mathbb{R}\) and begin.",
        "Ready.",
        "Is it alive?",
    ]
    assert projection["next_after_sequence"] == 3

    tail = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        after_sequence=2,
    )
    assert [item["atom_ref"] for item in tail["events"]] == [
        "event_human_3"
    ]


def test_workspace_minimum_budget_is_sufficient_across_digit_boundary(amos):
    current = interaction(
        "budget_event_current",
        sequence=1,
        role="human",
        content="x" * 20_000,
    )
    context = {
        "id": "budget_context",
        "type": "belief",
        "payload": {"claim": "y"},
    }
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="participant-ingress",
        idempotency_key="budget-boundary",
        atoms=[current, context],
        head_updates=[
            interaction_stream_head(
                "budget_event_current",
                expected_head_ref=None,
                expected_head_version=0,
            )
        ],
    )
    request = {
        "current_event_ref": "budget_event_current",
        "conversation_id": "main",
        "scope": SCOPE,
        "requester": "agent:participant",
        "target_processor": "primary-reasoner",
        "participant_refs": ["human:participant", "agent:participant"],
        "context_refs": ["budget_context"],
        "temporal_limit": 2,
        "recent_event_floor": 2,
        "thread_limit": 1,
    }

    with pytest.raises(CognitiveWorkspaceBudgetExceeded) as overflow:
        amos.compile_cognitive_workspace(
            **request,
            token_or_byte_budget={"tokens": 9_999, "items": 99},
        )

    # The protected projection lands exactly on a four-byte token boundary.
    # Growing the token limit from four to five digits adds one serialized
    # byte, so a raw ceil(used_bytes / 4) receipt would fail when replayed.
    assert overflow.value.budget["used_bytes"] % 4 == 0
    assert overflow.value.minimum_budget["tokens"] > overflow.value.budget[
        "estimated_tokens"
    ]
    recovered = amos.compile_cognitive_workspace(
        **request,
        token_or_byte_budget={
            "tokens": overflow.value.minimum_budget["tokens"],
            "items": overflow.value.minimum_budget["items"],
        },
    )
    assert recovered["status"] == "compiled"
    assert recovered["budget"]["used_bytes"] <= recovered["budget"]["limit_bytes"]
    assert recovered["budget"]["used_items"] <= recovered["budget"]["limit_items"]
    recovered_by_bytes = amos.compile_cognitive_workspace(
        **request,
        token_or_byte_budget={
            "bytes": overflow.value.minimum_budget["bytes"],
            "items": overflow.value.minimum_budget["items"],
        },
    )
    assert recovered_by_bytes["status"] == "compiled"
    assert (
        recovered_by_bytes["budget"]["used_bytes"]
        <= recovered_by_bytes["budget"]["limit_bytes"]
    )


def test_workspace_uses_verified_rolling_compaction_before_optional_shedding(amos):
    interactions = []
    previous = None
    for sequence in range(1, 13):
        atom = interaction(
            f"compact_event_{sequence}",
            sequence=sequence,
            role="human" if sequence % 2 else "agent",
            content=f"Canonical interaction {sequence} " + ("detail " * 12),
            in_reply_to=previous,
        )
        atom["scope"] = dict(SCOPE)
        committed = amos.commit_atom(
            atom,
            actor="continuity-test",
            idempotency_key=f"compact-event-{sequence}",
        )["atom"]
        interactions.append(committed)
        previous = committed["id"]

    covered = interactions[:8]
    source_refs = [str(atom["id"]) for atom in covered]
    summary = amos.commit_atom(
        {
            "id": "compact_rolling_summary",
            "type": "semantic",
            "payload": {
                "summary": "The first eight exchanges established the durable context.",
                "epistemic_status": "derived_summary_not_adopted_truth",
                "maintenance_source_refs": source_refs,
                "context_compaction": {
                    "profile": CONTEXT_COMPACTION_PROFILE,
                    "mode": "rolling",
                    "partition": {
                        "kind": "interaction_stream",
                        "key": "main",
                    },
                    "coverage": {
                        "from_sequence": 1,
                        "through_sequence": 8,
                        "through_ref": "compact_event_8",
                        "source_count": 8,
                        "source_digest": context_compaction_source_digest(
                            covered
                        ),
                        "raw_sources_retained": True,
                    },
                    "source_refs": source_refs,
                    "facets": {
                        "open_questions": ["What follows from the established context?"]
                    },
                },
            },
            "scope": dict(SCOPE),
        },
        actor="continuity-test",
        idempotency_key="compact-rolling-summary",
    )["atom"]
    invalid_source_refs = [
        str(atom["id"]) for atom in interactions[:9]
    ]
    amos.commit_atom(
        {
            "id": "compact_rolling_summary_invalid",
            "type": "semantic",
            "payload": {
                "summary": "This newer projection has an invalid source binding.",
                "maintenance_source_refs": invalid_source_refs,
                "context_compaction": {
                    "profile": CONTEXT_COMPACTION_PROFILE,
                    "mode": "rolling",
                    "partition": {
                        "kind": "interaction_stream",
                        "key": "main",
                    },
                    "coverage": {
                        "from_sequence": 1,
                        "through_sequence": 9,
                        "through_ref": "compact_event_9",
                        "source_count": 9,
                        "source_digest": "invalid-source-digest",
                        "raw_sources_retained": True,
                    },
                    "source_refs": invalid_source_refs,
                    "facets": {},
                },
            },
            "scope": dict(SCOPE),
        },
        actor="continuity-test",
        idempotency_key="compact-rolling-summary-invalid",
    )

    full = amos.compile_cognitive_workspace(
        current_event_ref="compact_event_12",
        conversation_id="main",
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        token_or_byte_budget={"bytes": 48_000, "items": 512},
        temporal_limit=12,
        recent_event_floor=4,
    )
    assert full["compacted_context"][0]["atom_ref"] == summary["id"]
    assert full["compacted_context"][0]["epistemic_status"] == (
        "derived_summary_not_adopted_truth"
    )
    assert full["compacted_context"][0]["coverage"]["through_sequence"] == 8
    assert any(
        item["reason"] == "compaction_source_digest_mismatch"
        for item in full["omissions"]
    )

    tight = amos.compile_cognitive_workspace(
        current_event_ref="compact_event_12",
        conversation_id="main",
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        token_or_byte_budget={"bytes": 48_000, "items": 80},
        temporal_limit=12,
        recent_event_floor=4,
    )
    assert [item["atom_ref"] for item in tight["temporal_context"]] == [
        "compact_event_9",
        "compact_event_10",
        "compact_event_11",
        "compact_event_12",
    ]
    assert tight["current_event"]["atom_ref"] == "compact_event_12"
    assert tight["compacted_context"][0]["atom_ref"] == summary["id"]
    assert tight["budget"]["used_items"] <= 80
    assert any(
        item["reason"] == "covered_temporal_context_compacted"
        for item in tight["omissions"]
    )
    assert all(
        amos.store.get_atom(ref)["lifecycle_state"] == "active"
        for ref in source_refs
    )
    summarizes = [
        edge
        for edge in amos.store.list_edges_for_refs([summary["id"], *source_refs])
        if edge["source_ref"] == summary["id"]
        and edge["relation"] == "rel:summarizes"
    ]
    assert {edge["target_ref"] for edge in summarizes} == set(source_refs)


def test_head_cas_is_atomic_idempotent_and_rebuildable(amos):
    opening = interaction(
        "event_opening",
        sequence=1,
        role="human",
        content="Begin",
    )
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="ingress",
        idempotency_key="opening",
        atoms=[
            opening,
            thread_root("thread_atomic", opening_event_ref="event_opening"),
            thread_state(
                "state_atomic_1",
                thread_id="thread_atomic",
                revision=1,
                event_refs=["event_opening"],
            ),
        ],
        edges=[
            {
                "source_ref": "state_atomic_1",
                "target_ref": "thread_atomic",
                "relation": "rel:part_of",
            }
        ],
        head_updates=[
            {
                "series_kind": "discourse_thread",
                "series_id": "thread_atomic",
                "new_head_ref": "state_atomic_1",
                "expected_head_ref": None,
                "expected_head_version": 0,
            },
            interaction_stream_head(
                "event_opening",
                expected_head_ref=None,
                expected_head_version=0,
            ),
        ],
    )
    before = amos.store.memory_revision()
    with pytest.raises(CASConflict):
        amos.commit_memory_transaction(
            scope=SCOPE,
            actor="primary",
            idempotency_key="stale-update",
            atoms=[
                interaction(
                    "event_stale",
                    sequence=2,
                    role="agent",
                    content="This must roll back",
                    in_reply_to="event_opening",
                ),
                thread_state(
                    "state_stale",
                    thread_id="thread_atomic",
                    revision=2,
                    event_refs=["event_opening", "event_stale"],
                ),
            ],
            head_updates=[
                {
                    "series_kind": "discourse_thread",
                    "series_id": "thread_atomic",
                    "new_head_ref": "state_stale",
                    "expected_head_ref": "wrong-head",
                    "expected_head_version": 1,
                },
                interaction_stream_head(
                    "event_stale",
                    expected_head_ref="event_opening",
                    expected_head_version=1,
                ),
            ],
        )
    assert amos.store.get_atom("event_stale") is None
    assert amos.store.get_atom("state_stale") is None
    assert amos.store.memory_revision() == before

    next_event = interaction(
        "event_next",
        sequence=2,
        role="agent",
        content="Continued",
        in_reply_to="event_opening",
    )
    next_state = thread_state(
        "state_atomic_2",
        thread_id="thread_atomic",
        revision=2,
        event_refs=["event_opening", "event_next"],
        secret="cedar",
    )
    request = {
        "scope": SCOPE,
        "actor": "primary",
        "idempotency_key": "advance-2",
        "atoms": [next_event, next_state],
        "edges": [
            {
                "source_ref": "state_atomic_2",
                "target_ref": "thread_atomic",
                "relation": "rel:part_of",
            }
        ],
        "head_updates": [
            {
                "series_kind": "discourse_thread",
                "series_id": "thread_atomic",
                "new_head_ref": "state_atomic_2",
                "expected_head_ref": "state_atomic_1",
                "expected_head_version": 1,
            },
            interaction_stream_head(
                "event_next",
                expected_head_ref="event_opening",
                expected_head_version=1,
            ),
        ],
    }
    committed = amos.commit_memory_transaction(**request)
    replayed = amos.commit_memory_transaction(**request)
    assert replayed == committed
    assert amos.store.get_atom("state_atomic_1")["lifecycle_state"] == "superseded"
    assert amos.store.get_memory_head(
        scope=SCOPE,
        series_kind="discourse_thread",
        series_id="thread_atomic",
    )["head_ref"] == "state_atomic_2"
    assert amos.verify_replay()["status"] == "ok"

    rebuilt = amos.rebuild_memory_heads()
    assert rebuilt["head_count"] == 2
    assert {
        (item["series_kind"], item["head_ref"])
        for item in rebuilt["heads"]
    } == {
        ("discourse_thread", "state_atomic_2"),
        ("interaction_stream", "event_next"),
    }
    assert amos.verify_replay()["status"] == "ok"


def test_interaction_stream_head_enforces_total_order_without_supersession(amos):
    with pytest.raises(
        ValidationError, match="requires a matching interaction_stream"
    ):
        amos.commit_memory_transaction(
            scope=SCOPE,
            idempotency_key="missing-stream-head",
            atoms=[
                interaction(
                    "event_missing_head",
                    sequence=1,
                    role="human",
                    content="Not ordered",
                )
            ],
        )

    first = interaction(
        "event_ordered_1",
        sequence=1,
        role="human",
        content="First",
    )
    amos.commit_memory_transaction(
        scope=SCOPE,
        idempotency_key="ordered-1",
        atoms=[first],
        head_updates=[
            interaction_stream_head(
                "event_ordered_1",
                expected_head_ref=None,
                expected_head_version=0,
            )
        ],
    )
    with pytest.raises(CASConflict, match="compare-and-swap"):
        amos.commit_memory_transaction(
            scope=SCOPE,
            idempotency_key="ordered-stale",
            atoms=[
                interaction(
                    "event_ordered_stale",
                    sequence=1,
                    role="agent",
                    content="Stale",
                )
            ],
            head_updates=[
                interaction_stream_head(
                    "event_ordered_stale",
                    expected_head_ref=None,
                    expected_head_version=0,
                )
            ],
        )
    amos.commit_memory_transaction(
        scope=SCOPE,
        idempotency_key="ordered-2",
        atoms=[
            interaction(
                "event_ordered_2",
                sequence=2,
                role="agent",
                content="Second",
                in_reply_to="event_ordered_1",
            )
        ],
        head_updates=[
            interaction_stream_head(
                "event_ordered_2",
                expected_head_ref="event_ordered_1",
                expected_head_version=1,
            )
        ],
    )
    assert amos.store.get_atom("event_ordered_1")["lifecycle_state"] == "active"
    head = amos.get_memory_head(
        scope=SCOPE,
        series_kind="interaction_stream",
        series_id="main",
        requester="human:participant",
        target_processor="participant-ui",
    )
    assert head["head_ref"] == "event_ordered_2"
    assert head["head_version"] == 2
    first_page = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        limit=1,
    )
    assert first_page["has_more"] is True
    final_page = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        after_sequence=1,
        limit=1,
    )
    assert final_page["has_more"] is False


def test_interaction_projection_can_include_typed_linked_lineage(amos):
    response = interaction(
        "event_linked_response",
        sequence=1,
        role="agent",
        content="I am initiating the selected operation.",
    )
    commitment = {
        "id": "commitment_linked",
        "type": "commitment",
        "payload": {
            "agent_id": "agent:participant",
            "description": "A typed caller operation",
            "promised_action": {
                "profile": "example.operation.v1",
                "kind": "example",
            },
            "commitment_status": "pending_authorized_execution",
            "source_event_ref": "event_linked_response",
        },
    }
    outcome = {
        "id": "outcome_linked",
        "type": "action_outcome",
        "payload": {
            "agent_id": "agent:participant",
            "action_ref": "commitment_linked",
            "status": "completed",
            "correction": None,
            "limitation": None,
        },
    }
    amos.commit_memory_transaction(
        scope=SCOPE,
        idempotency_key="linked-operation-lineage",
        atoms=[response, commitment, outcome],
        edges=[
            {
                "source_ref": "event_linked_response",
                "target_ref": "commitment_linked",
                "relation": "rel:supports",
            },
            {
                "source_ref": "commitment_linked",
                "target_ref": "outcome_linked",
                "relation": "rel:produced_outcome",
            },
        ],
        head_updates=[
            interaction_stream_head(
                "event_linked_response",
                expected_head_ref=None,
                expected_head_version=0,
            )
        ],
    )

    projection = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        linked_atom_types=["commitment", "action_outcome"],
        linked_depth=2,
    )

    linked = projection["events"][0]["linked_records"]
    assert [
        (
            item["depth"],
            item["relation"],
            item["record"]["id"],
            item["record"]["type"],
        )
        for item in linked
    ] == [
        (1, "rel:supports", "commitment_linked", "commitment"),
        (
            2,
            "rel:produced_outcome",
            "outcome_linked",
            "action_outcome",
        ),
    ]


def test_interaction_projection_rejects_half_bound_link_requests(amos):
    with pytest.raises(
        ValidationError,
        match="linked_atom_types and linked_depth must be supplied together",
    ):
        amos.compile_interaction_projection(
            conversation_id="main",
            scope=SCOPE,
            linked_atom_types=["commitment"],
        )


def test_private_state_requires_restricted_visibility(amos):
    with pytest.raises(ValidationError, match="restricted access_policy"):
        amos.commit_memory_transaction(
            scope=SCOPE,
            atoms=[
                interaction(
                    "event_visibility",
                    sequence=1,
                    role="human",
                    content="Begin",
                ),
                thread_root(
                    "thread_visibility", opening_event_ref="event_visibility"
                ),
                {
                    **thread_state(
                        "state_visibility",
                        thread_id="thread_visibility",
                        revision=1,
                        event_refs=["event_visibility"],
                    ),
                    "access_policy": {
                        "visibility": ["all"],
                        "mutable_by": ["owner"],
                    },
                },
            ],
        )


def test_http_memory_transaction_workspace_and_cas_conflict(tmp_path):
    try:
        server = AmosHTTPServer(
            ("127.0.0.1", 0), str(tmp_path / "continuity-http.sqlite3")
        )
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    try:
        committed = post(
            "/v1/memory-transactions:commit",
            {
                "profile": "amos.memory-transaction.v1",
                "scope": SCOPE,
                "actor": "http-ingress",
                "idempotency_key": "http-event",
                "atoms": [
                    interaction(
                        "event_http_1",
                        sequence=1,
                        role="human",
                        content=r"Preserve **Markdown** and \(x^2\).",
                    )
                ],
                "head_updates": [
                    interaction_stream_head(
                        "event_http_1",
                        expected_head_ref=None,
                        expected_head_version=0,
                    )
                ],
            },
        )
        assert committed["status"] == "committed"
        workspace = post(
            "/v1/cognitive-workspaces:compile",
            {
                "profile": "amos.cognitive-workspace-request.v1",
                "current_event_ref": "event_http_1",
                "conversation_id": "main",
                "scope": SCOPE,
                "requester": "human:participant",
                "target_processor": "primary-reasoner",
                "token_or_byte_budget": {"bytes": 20_000},
            },
        )
        assert workspace["current_event"]["content"] == (
            r"Preserve **Markdown** and \(x^2\)."
        )
        projection = post(
            "/v1/interaction-projections:compile",
            {
                "profile": "amos.interaction-projection-request.v2",
                "conversation_id": "main",
                "scope": SCOPE,
                "requester": "human:participant",
                "target_processor": "participant-ui",
            },
        )
        assert projection["events"][0]["content"] == (
            r"Preserve **Markdown** and \(x^2\)."
        )
        head = post(
            "/v1/memory-heads:get",
            {
                "profile": "amos.memory-head-request.v1",
                "scope": SCOPE,
                "series_kind": "interaction_stream",
                "series_id": "main",
                "requester": "human:participant",
                "target_processor": "participant-ui",
            },
        )
        assert head["status"] == "found"
        assert head["head_ref"] == "event_http_1"
        assert head["head_version"] == 1
        assert head["journal_event_id"] == committed["event"]["event_id"]
        assert head["updated_at"]
        versions = post(
            "/v1/memory-series:versions:get",
            {
                "profile": "amos.memory-series-version-request.v1",
                "scope": SCOPE,
                "series_kind": "interaction_stream",
                "series_id": "main",
                "versions": [1, 2],
                "requester": "human:participant",
                "target_processor": "participant-ui",
            },
        )
        assert versions["profile"] == "amos.memory-series-versions.v1"
        assert [item["head_ref"] for item in versions["items"]] == [
            "event_http_1"
        ]
        assert versions["missing_versions"] == [2]
        observed = post(
            "/v1/memory-transactions:observe",
            {
                "profile": "amos.memory-transaction-observation-request.v1",
                "event_id": committed["event"]["event_id"],
                "scope": SCOPE,
                "requester": "human:participant",
                "target_processor": "participant-ui",
            },
        )
        assert observed["profile"] == "amos.memory-transaction-observation.v1"
        assert observed["verification_status"] == "mechanically_verified"
        assert observed["projected_heads"][0]["head_ref"] == "event_http_1"

        with pytest.raises(urllib.error.HTTPError) as captured:
            post(
                "/v1/memory-transactions:commit",
                {
                    "scope": SCOPE,
                    "atoms": [
                        thread_state(
                            "state_http_stale",
                            thread_id="missing-thread",
                            revision=2,
                            event_refs=["event_http_1"],
                        )
                    ],
                    "head_updates": [
                        {
                            "series_kind": "discourse_thread",
                            "series_id": "missing-thread",
                            "new_head_ref": "state_http_stale",
                            "expected_head_ref": "old",
                            "expected_head_version": 1,
                        }
                    ],
                },
            )
        assert captured.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
def test_assessment_qualification_head_is_canonical_and_revision_guarded(amos):
    scope = {"system": "test", "namespace": "assessment"}
    first = {
        "id": "assessment_qualification_1",
        "type": "self_assessment",
        "payload": {
            "profile": "test.assessment-qualification.v1",
            "agent_id": "agent:test",
            "claim": "The candidate remains developmental.",
            "calibration": {"runs": 1},
            "assessment_series_id": "suite:test",
            "revision": 1,
        },
        "scope": scope,
    }
    amos.commit_memory_transaction(
        atoms=[first],
        head_updates=[{
            "series_kind": "assessment_qualification",
            "series_id": "suite:test",
            "expected_head_ref": None,
            "expected_head_version": 0,
            "new_head_ref": first["id"],
        }],
        actor="assessment:test",
        scope=scope,
    )
    head = amos.get_memory_head(
        scope=scope,
        series_kind="assessment_qualification",
        series_id="suite:test",
    )
    assert head["head_ref"] == first["id"]
    assert head["head_version"] == 1
