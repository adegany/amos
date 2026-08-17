from __future__ import annotations

import json
import threading
import time
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from amos import (
    AccessDenied,
    AgenticRecallAuditor,
    Amos,
    BackgroundMemoryPolicyWorker,
    CASConflict,
    CapacityGovernor,
    DistillerMaintenanceWorker,
    IdempotencyConflict,
    IndexMaintainer,
    JournalProjector,
    MemoryPolicyWorker,
    MemorySteward,
    MaintenanceProposal,
    PacketCacheInvalidator,
    SMPWorker,
    SemanticFacet,
    SemanticMaintenanceProcessor,
    SelfModelCalibrator,
    ValidationError,
    ontology_snapshot,
    semantic_relation_proposals_from_facets,
)
from amos.cli import main as cli_main
from amos.http_api import AmosHTTPServer
from amos.smp import cosine

from .helpers import ExampleTrainingFlightProcessor, item_refs


def test_schema_rejects_payload_envelope_duplication(amos):
    with pytest.raises(ValidationError):
        amos.commit_atom(
            {
                "type": "belief",
                "payload": {
                    "claim": "payload must not carry envelope metadata",
                    "confidence": {"level": "high"},
                },
            }
        )


def test_schema_validates_canonical_graph_metadata(amos):
    with pytest.raises(ValidationError, match="semantic_facets.*subject"):
        amos.commit_atom(
            {
                "type": "semantic",
                "payload": {
                    "summary": "Malformed facet",
                    "semantic_facets": [{"intent": "missing subject"}],
                },
            }
        )
    with pytest.raises(ValidationError, match="graph_relations.*unsupported"):
        amos.commit_atom(
            {
                "type": "semantic",
                "payload": {
                    "summary": "Malformed relation",
                    "graph_relations": [
                        {
                            "target_ref": "other_atom",
                            "relation": "rel:not_in_the_ontology",
                        }
                    ],
                },
            }
        )


def test_seed_ontology_uses_v1_relation_ids_without_aliases():
    snapshot = ontology_snapshot()
    assert "agent" in snapshot["entity_types"]
    assert "rel:contradicts" in snapshot["relation_ids"]
    assert "contradicts" not in snapshot["relation_ids"]
    assert (
        snapshot["dictionary_update_policy"]["agent_defined_relation"]
        == "not_allowed_in_v1_propose_for_review_only"
    )


def test_core_typed_payload_schema_artifact_defines_required_payloads():
    schema = json.loads(Path("schemas/core_payloads.schema.json").read_text())
    defs = schema["$defs"]
    for name in [
        "SemanticFacet",
        "GraphRelation",
        "BeliefAtom",
        "PreferenceAtom",
        "Goal",
        "Commitment",
        "ProcedureAtom",
        "Episode",
    ]:
        assert name in defs
    assert defs["PreferenceAtom"]["required"] == [
        "holder",
        "polarity",
        "target",
        "applicability_scope",
        "strength",
    ]
    assert defs["ProcedureAtom"]["required"] == ["trigger_context", "steps"]


def test_runtime_enforces_typed_payload_contracts(amos):
    with pytest.raises(ValidationError):
        amos.commit_atom(
            {
                "type": "preference",
                "payload": {
                    "holder": "trainer",
                    "polarity": "prefer",
                    "applicability_scope": "qandl training",
                    "strength": "high",
                },
            }
        )
    with pytest.raises(ValidationError):
        amos.commit_atom(
            {
                "type": "preference",
                "payload": {
                    "holder": "trainer",
                    "polarity": "likes",
                    "target": "advisor fallback",
                    "applicability_scope": "qandl training",
                    "strength": "high",
                },
            }
        )
    with pytest.raises(ValidationError):
        amos.commit_atom(
            {
                "type": "procedure",
                "payload": {"name": "no trigger", "steps": ["inspect status"]},
            }
        )
    with pytest.raises(ValidationError):
        amos.commit_atom(
            {
                "type": "procedure",
                "payload": {"trigger_context": "optimizer stalled", "steps": []},
            }
        )

    valid = amos.commit_atom(
        {
            "type": "procedure",
            "payload": {
                "trigger_context": "optimizer stalled",
                "steps": ["inspect status"],
            },
        }
    )["atom"]
    with pytest.raises(ValidationError):
        amos.update_atom(
            valid["id"],
            set_fields={"payload": {"trigger_context": "optimizer stalled"}},
        )


@pytest.mark.parametrize(
    "atom",
    [
        {"type": "belief", "payload": {"claim": 123}},
        {"type": "self_model", "payload": {"agent_id": 7}},
        {
            "type": "action_outcome",
            "payload": {"agent_id": "trainer", "action_ref": 4, "status": "success"},
        },
        {"type": "belief", "payload": {"claim": "valid",}, "utility": 1.1},
    ],
)
def test_runtime_enforces_json_schema_property_types_and_score_bounds(amos, atom):
    with pytest.raises(ValidationError):
        amos.commit_atom(atom)


def test_capture_commit_retrieve_scope_and_journal_chain(amos):
    evidence = amos.capture_event(
        source_type="log",
        source_ref="run-1",
        payload={"message": "UPRO drawdown patience should be increased"},
        scope={"tenant": "qandl"},
        idempotency_key="capture-1",
    )["evidence"]
    target = amos.commit_atom(
        {
            "id": "atom_qandl_patience",
            "type": "belief",
            "payload": {
                "claim": "UPRO optimizer benefits from drawdown patience after restart",
                "subject": "UPRO",
                "predicate": "drawdown_patience",
                "value": "increase",
            },
            "scope": {"tenant": "qandl"},
            "evidence_refs": [evidence["evidence_id"]],
        }
    )["atom"]
    hidden = amos.commit_atom(
        {
            "id": "atom_other_patience",
            "type": "belief",
            "payload": {"claim": "unrelated tenant UPRO note"},
            "scope": {"tenant": "other"},
        }
    )["atom"]

    packet = amos.retrieve_packet(
        cues=["UPRO drawdown patience"],
        scope={"tenant": "qandl"},
        target_processor="trainer_agent",
    )

    assert target["id"] in item_refs(packet)
    assert hidden["id"] not in item_refs(packet)
    assert any(o["atom_ref"] == hidden["id"] and o["reason"] == "scope_hidden" for o in packet["omissions"])
    events = amos.store.list_events()
    assert len(events) == 3
    assert events[1]["previous_event_hash"] == events[0]["checksum"]
    assert events[2]["previous_event_hash"] == events[1]["checksum"]


def test_idempotency_returns_same_response_and_conflicts_on_changed_payload(amos):
    request = {
        "type": "belief",
        "payload": {"claim": "Codex outage should use advisor fallback"},
        "scope": {"tenant": "qandl"},
    }
    first = amos.commit_atom(request, idempotency_key="commit-fallback")
    second = amos.commit_atom(request, idempotency_key="commit-fallback")
    assert second["atom"]["id"] == first["atom"]["id"]
    assert second["event"]["event_id"] == first["event"]["event_id"]

    with pytest.raises(IdempotencyConflict):
        amos.commit_atom(
            {
                "type": "belief",
                "payload": {"claim": "Changed payload"},
                "scope": {"tenant": "qandl"},
            },
            idempotency_key="commit-fallback",
        )


def test_commit_atom_projects_supersedes_edges(amos):
    old = amos.commit_atom(
        {
            "id": "old_runtime_observation",
            "type": "semantic",
            "payload": {"summary": "Old runtime observation"},
            "scope": {"tenant": "qandl"},
        }
    )
    new = amos.commit_atom(
        {
            "id": "new_runtime_observation",
            "type": "semantic",
            "payload": {"summary": "New runtime observation"},
            "scope": {"tenant": "qandl"},
            "supersedes": [old["atom"]["id"]],
        }
    )

    assert any(
        edge["source_ref"] == new["atom"]["id"]
        and edge["target_ref"] == old["atom"]["id"]
        and edge["relation"] == "rel:supersedes"
        for edge in new["edges"]
    )


def test_propose_batch_commit_deletion_request_and_shared_refresh(amos):
    proposed = amos.propose_memory_atoms(
        [
            {
                "id": "proposed_atom",
                "type": "belief",
                "payload": {"claim": "proposal should stay proposed"},
            }
        ],
        scope={"tenant": "qandl"},
    )
    assert proposed["proposals"][0]["atom"]["lifecycle_state"] == "proposed"

    committed = amos.commit_memory_atoms(
        [
            {
                "id": "batch_goal",
                "type": "goal",
                "payload": {"description": "shared refresh goal"},
                "scope": {"tenant": "qandl"},
            }
        ]
    )
    assert committed["committed"][0]["atom"]["id"] == "batch_goal"

    view = amos.refresh_shared_view(
        processor_ids=["planner", "executor"],
        cues=["shared refresh goal"],
        scope={"tenant": "qandl"},
    )
    assert view["refresh_status"] == "refreshed"
    assert view["common_graph_version"] == amos.store.graph_version()

    deleted = amos.request_deletion(
        target_ref="batch_goal",
        reason="test deletion request",
        expected_version=1,
    )
    assert deleted["status"] == "deleted"
    assert deleted["residual_retention"]["packet_cache"] == "purged"


def test_proposed_intrinsic_links_are_isolated_from_generic_promotion(amos):
    source = amos.commit_atom(
        {
            "id": "active_source_for_proposal",
            "type": "semantic",
            "payload": {"summary": "Reviewed source"},
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    proposal = amos.propose_memory_atoms(
        [
            {
                "id": "proposed_reflection_episode",
                "type": "episode",
                "payload": {
                    "summary": "A reviewable reflection occurrence",
                    "source_refs": [source["id"]],
                },
                "evidence_refs": [source["id"]],
            }
        ],
        scope={"tenant": "qandl"},
    )["proposals"][0]

    assert proposal["atom"]["lifecycle_state"] == "proposed"
    assert proposal["edges"] == []
    assert amos.store.list_edges() == []

    with pytest.raises(ValidationError, match="ratify_proposal"):
        amos.update_atom(
            proposal["atom"]["id"],
            set_fields={"lifecycle_state": "active"},
            expected_version=proposal["atom"]["version"],
        )

    assert amos.store.get_atom(proposal["atom"]["id"])["lifecycle_state"] == "proposed"
    assert amos.store.list_edges() == []
    assert amos.verify_replay()["status"] == "ok"


def test_batch_commit_uses_single_transaction_and_rejects_duplicate_batch(amos):
    batch = [
        {
            "id": "batch_one",
            "type": "belief",
            "payload": {
                "claim": "batch one",
                "graph_relations": [
                    {
                        "source_ref": "$self",
                        "target_ref": "batch_two",
                        "relation": "rel:derived_from",
                    }
                ],
            },
        },
        {
            "id": "batch_two",
            "type": "belief",
            "payload": {"claim": "batch two"},
        },
    ]
    committed = amos.commit_memory_atoms(
        batch,
        idempotency_key="batch-commit",
    )

    assert [item["atom"]["id"] for item in committed["committed"]] == [
        "batch_one",
        "batch_two",
    ]
    assert amos.store.graph_version() == 2
    assert any(
        edge["source_ref"] == "batch_one"
        and edge["relation"] == "rel:derived_from"
        and edge["target_ref"] == "batch_two"
        for edge in amos.store.list_edges()
    )
    replayed = amos.commit_memory_atoms(
        batch,
        idempotency_key="batch-commit",
    )
    assert replayed == committed
    assert amos.store.graph_version() == 2
    with pytest.raises(IdempotencyConflict):
        amos.commit_memory_atoms(
            [
                {
                    **batch[0],
                    "payload": {"claim": "changed batch one"},
                },
                batch[1],
            ],
            idempotency_key="batch-commit",
        )
    with pytest.raises(ValidationError):
        amos.commit_memory_atoms(
            [
                {
                    "id": "batch_duplicate",
                    "type": "belief",
                    "payload": {"claim": "duplicate"},
                },
                {
                    "id": "batch_duplicate",
                    "type": "belief",
                    "payload": {"claim": "duplicate again"},
                },
            ]
        )
    assert amos.store.get_atom("batch_duplicate") is None


def test_batch_commit_rejects_empty_batch(amos):
    with pytest.raises(ValidationError, match="must not be empty"):
        amos.commit_memory_atoms([], idempotency_key="empty-batch")


def test_batch_preflight_rejects_active_memory_derived_from_proposed_batch_atom(
    amos,
):
    with pytest.raises(
        ValidationError,
        match="active derived memory cannot use proposed",
    ):
        amos.commit_memory_atoms(
            [
                {
                    "id": "batch_active_derivative",
                    "type": "semantic",
                    "payload": {
                        "summary": "Must not gain active standing.",
                        "source_refs": ["batch_proposed_source"],
                    },
                },
                {
                    "id": "batch_proposed_source",
                    "type": "belief",
                    "payload": {"claim": "Still only a candidate."},
                    "lifecycle_state": "proposed",
                },
            ],
            idempotency_key="batch-proposed-source",
        )

    assert amos.store.get_atom("batch_active_derivative") is None
    assert amos.store.get_atom("batch_proposed_source") is None
    assert amos.store.graph_version() == 0


@pytest.mark.parametrize("source_lifecycle", ["archived", "superseded"])
def test_batch_preflight_accepts_historical_source_atom(amos, source_lifecycle):
    committed = amos.commit_memory_atoms(
        [
            {
                "id": f"batch_{source_lifecycle}_source",
                "type": "action_outcome",
                "payload": {
                    "agent_id": "agent:test",
                    "action_ref": "action:test",
                    "status": "completed",
                    "summary": "A retained historical execution result.",
                },
                "lifecycle_state": source_lifecycle,
                "health_status": "stale" if source_lifecycle == "archived" else "healthy",
            },
            {
                "id": f"batch_{source_lifecycle}_derivative",
                "type": "semantic",
                "payload": {
                    "summary": "An active assessment of historical evidence.",
                    "source_refs": [f"batch_{source_lifecycle}_source"],
                },
            },
        ],
        idempotency_key=f"batch-{source_lifecycle}-source",
    )

    assert [item["atom"]["lifecycle_state"] for item in committed["committed"]] == [
        source_lifecycle,
        "active",
    ]


@pytest.mark.parametrize(
    ("atom_type", "payload"),
    [
        ("goal", {"objective": "Track a provisional reflection."}),
        ("episode", {"summary": "Examined a provisional reflection."}),
        (
            "action_outcome",
            {
                "agent_id": "agent:test",
                "action_ref": "action:reflect",
                "status": "completed",
            },
        ),
    ],
)
def test_process_provenance_accepts_proposed_contradicted_source(
    amos,
    atom_type,
    payload,
):
    source_ref = f"proposed_process_source_{atom_type}"
    derived_ref = f"active_process_record_{atom_type}"

    committed = amos.commit_memory_atoms(
        [
            {
                "id": source_ref,
                "type": "semantic",
                "payload": {"summary": "A provisional, contested reflection."},
                "lifecycle_state": "proposed",
                "health_status": "contradicted",
            },
            {
                "id": derived_ref,
                "type": atom_type,
                "payload": {**payload, "source_refs": [source_ref]},
            },
        ],
        idempotency_key=f"process-provenance-{atom_type}",
    )

    assert [item["atom"]["id"] for item in committed["committed"]] == [
        source_ref,
        derived_ref,
    ]
    assert amos.store.get_atom(derived_ref)["lifecycle_state"] == "active"


def test_batch_replay_precedes_later_source_lifecycle_validation(amos):
    source = amos.commit_atom(
        {
            "id": "batch_replay_source",
            "type": "belief",
            "payload": {"claim": "active during the original write"},
        }
    )["atom"]
    batch = [
        {
            "id": "batch_replay_derived",
            "type": "semantic",
            "payload": {
                "summary": "Derived while the source was active.",
                "source_refs": [source["id"]],
            },
        }
    ]
    committed = amos.commit_memory_atoms(
        batch,
        idempotency_key="batch-replay-after-source-change",
    )
    amos.archive_atom(source["id"], reason="exercise stable operation replay")

    replayed = amos.commit_memory_atoms(
        batch,
        idempotency_key="batch-replay-after-source-change",
    )

    assert replayed == committed
    assert amos.store.get_atom("batch_replay_derived")["lifecycle_state"] == "active"


def test_batch_commit_rolls_back_every_atom_after_mid_batch_storage_failure(
    amos, monkeypatch
):
    insert_atom = amos.store.insert_atom
    calls = 0

    def fail_second_insert(conn, atom):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected batch storage failure")
        return insert_atom(conn, atom)

    monkeypatch.setattr(amos.store, "insert_atom", fail_second_insert)
    with pytest.raises(RuntimeError, match="injected batch storage failure"):
        amos.commit_memory_atoms(
            [
                {
                    "id": "batch_rollback_one",
                    "type": "belief",
                    "payload": {"claim": "must roll back"},
                },
                {
                    "id": "batch_rollback_two",
                    "type": "belief",
                    "payload": {"claim": "also must roll back"},
                },
            ],
            idempotency_key="batch-rollback",
        )

    assert amos.store.get_atom("batch_rollback_one") is None
    assert amos.store.get_atom("batch_rollback_two") is None
    assert amos.store.graph_version() == 0


def test_compare_and_swap_update_conflict(amos):
    atom = amos.commit_atom(
        {
            "id": "atom_cas",
            "type": "belief",
            "payload": {"claim": "initial"},
        }
    )["atom"]
    updated = amos.update_atom(
        atom["id"],
        payload_patch={"claim": "updated"},
        expected_version=1,
    )["atom"]
    assert updated["version"] == 2
    with pytest.raises(CASConflict):
        amos.update_atom(atom["id"], payload_patch={"claim": "stale"}, expected_version=1)


def test_identical_update_is_a_noop_without_revision_or_event(amos):
    atom = amos.commit_atom(
        {
            "id": "atom_noop_update",
            "type": "belief",
            "payload": {"claim": "stable"},
        }
    )["atom"]
    event_count = len(amos.store.list_events())

    result = amos.update_atom(
        atom["id"],
        payload_patch={"claim": "stable"},
        expected_version=atom["version"],
    )

    assert result["status"] == "unchanged"
    assert result["atom"]["version"] == atom["version"]
    assert result["event"] is None
    assert len(amos.store.list_events()) == event_count


def test_merge_atoms_requires_review_and_replays_projection(amos):
    first = amos.commit_atom(
        {
            "id": "merge_a",
            "type": "belief",
            "payload": {"claim": "merge source a"},
        }
    )["atom"]
    second = amos.commit_atom(
        {
            "id": "merge_b",
            "type": "belief",
            "payload": {"claim": "merge source b"},
        }
    )["atom"]
    gated = amos.merge_atoms(
        source_refs=[first["id"], second["id"]],
        merged_payload={"summary": "merged source"},
    )
    assert gated["status"] == "review_required"

    merged = amos.merge_atoms(
        source_refs=[first["id"], second["id"]],
        merged_payload={"summary": "merged source"},
        approved_by="reviewer",
    )
    assert merged["status"] == "merged"
    assert {edge["relation"] for edge in merged["edges"]} == {"rel:derived_from"}
    packet = amos.retrieve_packet(cues=["merged source"], include_archived=True)
    assert merged["atom"]["id"] in item_refs(packet)
    assert amos.store.get_atom(first["id"])["health_status"] == "merged"
    assert amos.verify_replay()["status"] == "ok"

    amos.delete_atom(merged["atom"]["id"], reason="remove merged view")
    assert amos.store.list_edges() == []
    assert amos.verify_replay()["status"] == "ok"


def test_commit_atom_projects_structured_graph_edges_and_replay(amos):
    source = amos.commit_atom(
        {
            "id": "source_memory",
            "type": "agentic_trace",
            "payload": {
                "task": "source task",
                "action": "record source memory",
                "outcome": "observed",
            },
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    cited = amos.commit_atom(
        {
            "id": "cited_memory",
            "type": "semantic",
            "payload": {"summary": "cited memory"},
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    directive = amos.commit_atom(
        {
            "id": "directive_with_refs",
            "type": "agentic_trace",
            "payload": {
                "task": "directive task",
                "action": "use retrieved memory",
                "outcome": "issued",
                "memory_references": [{"id": cited["id"]}],
            },
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    distilled = amos.commit_atom(
        {
            "id": "distilled_with_source",
            "type": "semantic",
            "payload": {
                "summary": "distilled from source",
                "source_refs": [source["id"]],
            },
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    outcome = amos.commit_atom(
        {
            "id": "directive_outcome",
            "type": "agentic_trace",
            "payload": {
                "task": "directive task",
                "action": "evaluate directive outcome",
                "outcome": "supported",
                "directive_atom_ref": directive["id"],
            },
            "scope": {"tenant": "graph"},
        }
    )["atom"]

    triples = {
        (edge["source_ref"], edge["relation"], edge["target_ref"])
        for edge in amos.store.list_edges()
    }
    assert (distilled["id"], "rel:derived_from", source["id"]) in triples
    assert (directive["id"], "rel:uses", cited["id"]) in triples
    assert (directive["id"], "rel:produced_outcome", outcome["id"]) in triples
    assert amos.verify_replay()["status"] == "ok"


def test_commit_atom_projects_self_model_profile_edges(amos):
    self_model = amos.commit_atom(
        {
            "id": "agent_self_model",
            "type": "self_model",
            "payload": {"agent_id": "agent:demo", "role": "demo"},
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    capability = amos.commit_atom(
        {
            "id": "agent_capability",
            "type": "capability",
            "payload": {"agent_id": "agent:demo", "name": "plan"},
            "evidence_refs": [self_model["id"]],
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    limitation = amos.commit_atom(
        {
            "id": "agent_limitation",
            "type": "limitation",
            "payload": {"agent_id": "agent:demo", "name": "no_shell"},
            "evidence_refs": [self_model["id"]],
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    commitment = amos.commit_atom(
        {
            "id": "agent_commitment",
            "type": "commitment",
            "payload": {"agent_id": "agent:demo", "description": "cite memory"},
            "evidence_refs": [self_model["id"]],
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    runtime = amos.commit_atom(
        {
            "id": "agent_runtime",
            "type": "runtime_state",
            "payload": {"agent_id": "agent:demo", "status": "available"},
            "evidence_refs": [self_model["id"]],
            "scope": {"tenant": "graph"},
        }
    )["atom"]

    triples = {
        (edge["source_ref"], edge["relation"], edge["target_ref"])
        for edge in amos.store.list_edges()
    }
    assert (self_model["id"], "rel:has_capability", capability["id"]) in triples
    assert (self_model["id"], "rel:has_limitation", limitation["id"]) in triples
    assert (self_model["id"], "rel:made_commitment", commitment["id"]) in triples
    assert (self_model["id"], "rel:attributed_to", runtime["id"]) in triples
    assert amos.verify_replay()["status"] == "ok"


def test_deletion_tombstone_blocks_recreation(amos):
    atom = amos.commit_atom(
        {
            "id": "atom_delete_me",
            "type": "belief",
            "payload": {"claim": "temporary"},
        }
    )["atom"]
    amos.delete_atom(atom["id"], reason="operator requested", expected_version=1)
    packet = amos.retrieve_packet(cues=["temporary"], include_archived=True)
    assert atom["id"] not in item_refs(packet)
    with pytest.raises(ValidationError):
        amos.commit_atom(
            {
                "id": "atom_delete_me",
                "type": "belief",
                "payload": {"claim": "temporary"},
            }
        )
    with pytest.raises(ValidationError):
        amos.commit_atom(
            {
                "type": "belief",
                "payload": {"claim": "temporary"},
            }
        )


def test_journal_replay_and_cache_invalidation_after_delete(amos):
    first = amos.commit_atom(
        {
            "id": "replay_first",
            "type": "belief",
            "payload": {"claim": "first"},
        }
    )["atom"]
    second = amos.commit_atom(
        {
            "id": "replay_second",
            "type": "belief",
            "payload": {"claim": "second"},
        }
    )["atom"]
    amos.update_atom(first["id"], payload_patch={"claim": "first updated"})
    amos.archive_atom(second["id"], reason="test archive")
    amos.run_steward(approved_by="reviewer")
    packet = amos.retrieve_packet(cues=["first updated"], include_archived=True)
    assert amos.store.list_packet_cache()
    assert first["id"] in item_refs(packet)

    amos.delete_atom(first["id"], reason="test delete")
    assert amos.store.list_packet_cache() == []
    chain = amos.verify_journal_chain()
    replay = amos.verify_replay()
    assert chain["status"] == "ok"
    assert replay["status"] == "ok"
    assert first["id"] not in amos.replay_graph()["atoms"]


def test_integrity_verification_uses_one_snapshot_during_concurrent_write(
    amos,
    monkeypatch,
):
    amos.commit_atom(
        {
            "id": "snapshot_initial",
            "type": "belief",
            "payload": {"claim": "Present before verification."},
        }
    )
    writer = Amos(amos.store.path)
    original_list_live_events = amos.store.list_live_events
    wrote_concurrently = False

    def list_events_with_concurrent_write(**kwargs):
        nonlocal wrote_concurrently
        events = original_list_live_events(**kwargs)
        if not wrote_concurrently:
            wrote_concurrently = True
            writer.commit_atom(
                {
                    "id": "snapshot_concurrent",
                    "type": "belief",
                    "payload": {"claim": "Committed during verification."},
                }
            )
        return events

    monkeypatch.setattr(
        amos.store, "list_live_events", list_events_with_concurrent_write
    )
    try:
        verified = amos.verify_integrity()
    finally:
        writer.close()

    assert verified["journal"]["status"] == "ok"
    assert verified["replay"]["status"] == "ok"
    assert verified["journal"]["graph_version"] == 1
    assert verified["replay"]["graph_version"] == 1
    assert verified["replay"]["stored_atom_count"] == 1
    assert verified["replay"]["replayed_atom_count"] == 1
    assert amos.store.get_atom("snapshot_concurrent") is not None


def test_mutation_authorization_enforces_actor_trust_and_capability(amos):
    atom = amos.commit_atom(
        {
            "id": "atom_guarded",
            "type": "belief",
            "payload": {"claim": "guarded"},
            "access_policy": {
                "visibility": ["all"],
                "mutable_by": ["memory_admin"],
                "min_trust_level": 5,
                "requires_capability": "memory.write",
            },
        }
    )["atom"]
    with pytest.raises(AccessDenied):
        amos.update_atom(atom["id"], payload_patch={"claim": "bad"}, actor="intruder")
    with pytest.raises(AccessDenied):
        amos.update_atom(
            atom["id"],
            payload_patch={"claim": "still bad"},
            actor="operator",
            authorization_context={"roles": ["memory_admin"], "trust_level": 4},
        )
    updated = amos.update_atom(
        atom["id"],
        payload_patch={"claim": "authorized"},
        actor="operator",
        authorization_context={
            "roles": ["memory_admin"],
            "trust_level": 5,
            "capabilities": ["memory.write"],
        },
    )["atom"]
    assert updated["payload"]["claim"] == "authorized"
