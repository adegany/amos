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


def test_steward_removes_legacy_active_edges_touching_proposed_atoms(amos):
    source = amos.commit_atom(
        {
            "id": "legacy_edge_active_source",
            "type": "semantic",
            "payload": {"summary": "Reviewed source"},
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    proposed = amos.propose_memory_atoms(
        [
            {
                "id": "legacy_edge_proposed_atom",
                "type": "semantic",
                "payload": {"summary": "Still under review"},
            }
        ],
        scope={"tenant": "qandl"},
    )["proposals"][0]["atom"]
    legacy_edge = amos.graph._edge(
        proposed["id"], source["id"], "rel:derived_from", {"tenant": "qandl"}
    )
    with amos.store.transaction() as conn:
        amos.store.insert_edge(conn, legacy_edge)
    assert amos.store.list_edges()

    result = amos.run_steward(scope={"tenant": "qandl"})

    action = next(
        item for item in result["actions"]
        if item["action"] == "isolate_proposed_endpoint_edges"
    )
    assert action["edge_count"] == 1
    assert amos.store.list_edges() == []
    assert amos.verify_replay()["status"] == "ok"


def test_llm_reviewer_default_policy_is_disabled_and_non_authoritative(amos):
    policy = amos.llm_reviewer_policy()
    assert policy["enabled_by_default"] is False
    assert "direct_canonical_mutation" in policy["forbidden"]
    assert "deletion_approval" in policy["forbidden"]
    assert "recommended_action" in policy["output_envelope"]


def test_steward_marks_contradictions_and_retrieval_reports_conflicts(amos):
    amos.commit_atom(
        {
            "id": "belief_patience_up",
            "type": "belief",
            "payload": {
                "subject": "UPRO",
                "predicate": "patience",
                "value": "increase",
                "claim": "increase UPRO patience",
            },
            "scope": {"tenant": "qandl"},
        }
    )
    amos.commit_atom(
        {
            "id": "belief_patience_down",
            "type": "belief",
            "payload": {
                "subject": "UPRO",
                "predicate": "patience",
                "value": "decrease",
                "claim": "decrease UPRO patience",
            },
            "scope": {"tenant": "qandl"},
        }
    )

    result = amos.run_steward(scope={"tenant": "qandl"})
    assert any(
        action["action"] == "propose_contradiction_review"
        and action["review_required"] is True
        for action in result["actions"]
    )

    hidden = amos.retrieve_packet(
        cues=["UPRO patience"],
        scope={"tenant": "qandl"},
        include_conflicts=False,
    )
    assert {"belief_patience_up", "belief_patience_down"}.issubset(item_refs(hidden))

    approved = amos.run_steward(scope={"tenant": "qandl"}, approved_by="reviewer")
    assert any(action["action"] == "mark_contradiction" for action in approved["actions"])

    hidden_after_approval = amos.retrieve_packet(
        cues=["UPRO patience"],
        scope={"tenant": "qandl"},
        include_conflicts=False,
    )
    assert "belief_patience_up" not in item_refs(hidden_after_approval)
    assert any(o["reason"] == "contradicted" for o in hidden_after_approval["omissions"])

    visible = amos.retrieve_packet(
        cues=["UPRO patience"],
        scope={"tenant": "qandl"},
        include_conflicts=True,
    )
    assert {"belief_patience_up", "belief_patience_down"}.issubset(item_refs(visible))
    assert any(edge["relation"] == "rel:contradicts" for edge in visible["conflicts"])


def test_smp_interface_outputs_required_envelope_and_review_gate(amos):
    atom_a = amos.commit_atom(
        {
            "id": "smp_a",
            "type": "belief",
            "payload": {
                "subject": "codex",
                "predicate": "available",
                "value": True,
                "claim": "Codex is available",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    atom_b = amos.commit_atom(
        {
            "id": "smp_b",
            "type": "belief",
            "payload": {
                "subject": "codex",
                "predicate": "available",
                "value": False,
                "claim": "Codex is unavailable",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]

    smp = SemanticMaintenanceProcessor()
    comparison = smp.compare(atom_a, atom_b)
    assert {
        "processor_id",
        "processor_version",
        "input_refs",
        "output_type",
        "confidence",
        "reason_code",
        "evidence_refs",
        "recommended_action",
        "risk_level",
    }.issubset(comparison)
    assert comparison["reason_code"] == "contradiction_candidate"
    assert comparison["risk_level"] == "high"

    analysis = amos.run_smp_analysis(scope={"tenant": "qandl"})
    assert analysis["review_required"]
    assert any(output["risk_level"] == "high" for output in analysis["review_required"])


def test_smp_encoder_uses_idf_weighting_and_character_ngrams():
    smp = SemanticMaintenanceProcessor(dimensions=256)
    smp.configure_vector_model(
        document_count=10,
        document_frequencies={"common": 10, "distinctive": 1},
        graph_version=7,
    )

    mixed = smp.encode("common distinctive")
    assert cosine(mixed, smp.encode("distinctive")) > cosine(
        mixed, smp.encode("common")
    )

    assert cosine(smp.encode("retrieval"), smp.encode("retrieving")) > 0.1


def test_distillation_creates_provenance_linked_summary_and_gates_archival(amos):
    first = amos.commit_atom(
        {
            "id": "lesson_timeout",
            "type": "belief",
            "payload": {"claim": "Codex directive timeout wastes supervisor budget"},
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    second = amos.commit_atom(
        {
            "id": "lesson_fallback",
            "type": "belief",
            "payload": {"claim": "Local advisor fallback preserves progress"},
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    gated = amos.distill_memories(
        target_refs=[first["id"], second["id"]],
        summary="Directive outages should use local advisor fallback.",
        scope={"tenant": "qandl"},
        archive_sources=True,
    )
    assert gated["status"] == "review_required"

    distilled = amos.distill_memories(
        target_refs=[first["id"], second["id"]],
        summary="Directive outages should use local advisor fallback.",
        scope={"tenant": "qandl"},
        idempotency_key="distill-fallback",
    )
    replay = amos.distill_memories(
        target_refs=[first["id"], second["id"]],
        summary="Directive outages should use local advisor fallback.",
        scope={"tenant": "qandl"},
        idempotency_key="distill-fallback",
    )
    assert replay["atom"]["id"] == distilled["atom"]["id"]
    assert distilled["atom"]["payload"]["source_refs"] == [first["id"], second["id"]]
    assert {edge["relation"] for edge in distilled["edges"]} == {"rel:derived_from"}

    packet = amos.retrieve_packet(
        cues=["advisor fallback"],
        scope={"tenant": "qandl"},
    )
    assert distilled["atom"]["id"] in item_refs(packet)
    archived = amos.archive_atom(distilled["atom"]["id"], reason="obsolete distillation")
    assert archived["projected_edges"]
    assert amos.store.list_edges() == []
    assert amos.verify_replay()["status"] == "ok"


def test_steward_prunes_live_profile_edges_to_inactive_atoms(amos):
    self_model = amos.commit_atom(
        {
            "id": "stale_edge_self_model",
            "type": "self_model",
            "payload": {"agent_id": "agent.edge", "role_key": "edge"},
        }
    )["atom"]
    capability = amos.commit_atom(
        {
            "id": "stale_edge_capability",
            "type": "capability",
            "payload": {
                "agent_id": "agent.edge",
                "role_key": "edge",
                "name": "old capability",
            },
            "evidence_refs": [self_model["id"]],
        }
    )["atom"]
    assert amos.store.list_edges()

    # Reproduce a legacy projection in which the atom lifecycle changed
    # without cascading its live attachment edge.
    with amos.store.transaction() as conn:
        archived = dict(capability)
        archived["lifecycle_state"] = "archived"
        archived["health_status"] = "stale"
        archived["version"] = int(archived["version"]) + 1
        amos.store.replace_atom(conn, archived)

    result = amos.run_steward(actor="test")

    assert any(
        action.get("action") == "prune_inactive_attachment_edges"
        for action in result["actions"]
    )
    assert amos.store.list_edges() == []


def test_steward_backfills_intrinsic_edges_for_existing_atoms(amos):
    semantic = amos.commit_atom(
        {
            "id": "early_semantic",
            "type": "semantic",
            "payload": {
                "summary": "semantic arrived before source",
                "source_refs": ["late_source"],
            },
            "evidence_refs": ["evt_semantic_observation"],
            "confidence": {"level": "high", "score": 0.91},
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    source = amos.commit_atom(
        {
            "id": "late_source",
            "type": "agentic_trace",
            "payload": {
                "task": "late task",
                "action": "record late source",
                "outcome": "observed",
            },
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    assert amos.store.list_edges() == []

    result = amos.run_steward(scope={"tenant": "graph"})

    assert any(
        action["action"] == "project_intrinsic_edges" and action["edge_count"] == 1
        for action in result["actions"]
    )
    triples = {
        (edge["source_ref"], edge["relation"], edge["target_ref"])
        for edge in amos.store.list_edges()
    }
    assert (semantic["id"], "rel:derived_from", source["id"]) in triples
    edge = next(
        edge
        for edge in amos.store.list_edges()
        if edge["source_ref"] == semantic["id"]
        and edge["relation"] == "rel:derived_from"
        and edge["target_ref"] == source["id"]
    )
    assert edge["evidence_refs"] == ["evt_semantic_observation"]
    assert edge["confidence"] == {"level": "high", "score": 0.91}
    assert amos.verify_replay()["status"] == "ok"


def test_steward_refreshes_legacy_intrinsic_edge_provenance(amos):
    source = amos.commit_atom(
        {
            "id": "provenance_source",
            "type": "agentic_trace",
            "payload": {
                "task": "provenance",
                "action": "record source",
                "outcome": "observed",
            },
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    semantic = amos.commit_atom(
        {
            "id": "provenance_semantic",
            "type": "semantic",
            "payload": {
                "summary": "Semantic relation awaiting provenance",
                "source_refs": [source["id"]],
            },
            "scope": {"tenant": "graph"},
        }
    )["atom"]
    edge = next(
        edge
        for edge in amos.store.list_edges()
        if edge["source_ref"] == semantic["id"]
        and edge["target_ref"] == source["id"]
    )
    assert edge["evidence_refs"] == []

    updated = amos.update_atom(
        semantic["id"],
        set_fields={
            "evidence_refs": ["evt_later_provenance"],
            "confidence": {"level": "high", "score": 0.92},
        },
        expected_version=semantic["version"],
    )["atom"]
    unchanged_edge = amos.store.get_edge(edge["edge_id"])
    assert unchanged_edge["evidence_refs"] == []

    result = amos.run_steward(scope={"tenant": "graph"})

    assert any(
        action["action"] == "refresh_intrinsic_edges"
        and action["edge_count"] == 1
        for action in result["actions"]
    )
    refreshed = amos.store.get_edge(edge["edge_id"])
    assert refreshed["evidence_refs"] == ["evt_later_provenance"]
    assert refreshed["confidence"] == {"level": "high", "score": 0.92}
    assert refreshed["version"] == unchanged_edge["version"] + 1
    assert updated["version"] == semantic["version"] + 1
    assert amos.verify_replay()["status"] == "ok"


def test_steward_archives_structured_reflection_and_runtime_duplicates(amos):
    scope = {
        "tenant": "qandl",
        "component": "training",
        "asset": "UPRO",
        "run_id": "run1",
    }
    directive = amos.commit_atom(
        {
            "id": "directive_chunk_7",
            "type": "agentic_trace",
            "payload": {
                "agent_id": "qandl.training.pilot",
                "qandl_kind": "directive",
                "target_chunk": 7,
                "task": "UPRO training chunk 7",
                "action": "issue directive",
                "outcome": "issued",
            },
            "scope": scope,
        }
    )["atom"]
    old_reflection = amos.commit_atom(
        {
            "id": "reflection_chunk_7_old",
            "type": "agentic_trace",
            "payload": {
                "agent_id": "qandl.training.pilot",
                "qandl_kind": "reflection",
                "chunk": 7,
                "task": "UPRO training chunk 7",
                "action": "evaluate outcome",
                "outcome": "supported",
                "delta_multiple": 0.2,
            },
            "scope": scope,
        }
    )["atom"]
    new_reflection = amos.commit_atom(
        {
            "id": "reflection_chunk_7_new",
            "type": "agentic_trace",
            "payload": {
                "agent_id": "qandl.training.pilot",
                "qandl_kind": "reflection",
                "chunk": 7,
                "task": "UPRO training chunk 7",
                "action": "evaluate outcome",
                "outcome": "supported",
                "delta_multiple": 0.2,
                "directive_atom_ref": directive["id"],
            },
            "scope": scope,
        }
    )["atom"]
    runtime_old = amos.commit_atom(
        {
            "id": "runtime_old",
            "type": "runtime_state",
            "payload": {
                "agent_id": "qandl.training.pilot",
                "role_key": "pilot",
                "status": "available",
            },
            "scope": scope,
        }
    )["atom"]
    runtime_new = amos.commit_atom(
        {
            "id": "runtime_new",
            "type": "runtime_state",
            "payload": {
                "agent_id": "qandl.training.pilot",
                "role_key": "pilot",
                "status": "available",
                "runtime_capabilities": {
                    "choose_next_chunk_directive": {"available": True}
                },
            },
            "scope": scope,
        }
    )["atom"]

    result = amos.run_steward(scope=scope)

    archived_actions = [
        action
        for action in result["actions"]
        if action["action"] == "archive_structured_duplicate"
    ]
    assert {
        (action["kind"], action["kept"], action["archived"])
        for action in archived_actions
    } == {
        ("agentic_trace.reflection", new_reflection["id"], old_reflection["id"]),
        ("runtime_state.current", runtime_new["id"], runtime_old["id"]),
    }
    assert amos.store.get_atom(old_reflection["id"])["lifecycle_state"] == "archived"
    assert amos.store.get_atom(runtime_old["id"])["lifecycle_state"] == "archived"
    assert amos.store.get_atom(new_reflection["id"])["lifecycle_state"] == "active"
    assert amos.store.get_atom(runtime_new["id"])["lifecycle_state"] == "active"
    assert not any(
        edge["source_ref"] in {old_reflection["id"], runtime_old["id"]}
        or edge["target_ref"] in {old_reflection["id"], runtime_old["id"]}
        for edge in amos.store.list_edges()
    )
    assert amos.verify_replay()["status"] == "ok"


def test_steward_does_not_project_intrinsic_edges_for_archived_atoms(amos):
    scope = {"tenant": "qandl"}
    self_model = amos.commit_atom(
        {
            "id": "self_model_for_archive_projection",
            "type": "self_model",
            "payload": {"agent_id": "trainer"},
            "scope": scope,
        }
    )["atom"]
    runtime = amos.commit_atom(
        {
            "id": "runtime_for_archive_projection",
            "type": "runtime_state",
            "payload": {"agent_id": "trainer"},
            "evidence_refs": [self_model["id"]],
            "scope": scope,
        }
    )["atom"]
    assert amos.store.list_edges()

    amos.archive_atom(runtime["id"], reason="stale runtime")
    assert amos.store.list_edges() == []

    result = amos.run_steward(scope=scope)

    assert not any(
        action["action"] == "project_intrinsic_edges" for action in result["actions"]
    )
    assert amos.store.list_edges() == []
    assert amos.verify_replay()["status"] == "ok"


def test_high_risk_maintenance_requires_review(amos):
    atom = amos.commit_atom(
        {"id": "atom_review_gate", "type": "belief", "payload": {"claim": "keep"}}
    )["atom"]
    before = amos.store.graph_version()
    result = amos.request_maintenance(
        action="delete",
        target_refs=[atom["id"]],
        risk="high",
    )
    assert result["status"] == "review_required"
    assert result["mutated"] is False
    assert amos.store.graph_version() == before


def test_maintenance_edge_conflict_does_not_append_phantom_projection(amos, monkeypatch):
    source = amos.commit_atom(
        {"id": "edge_race_source", "type": "semantic", "payload": {"summary": "source"}}
    )["atom"]
    target = amos.commit_atom(
        {"id": "edge_race_target", "type": "semantic", "payload": {"summary": "target"}}
    )["atom"]
    stored_edge = amos.graph._edge(source["id"], target["id"], "rel:derived_from", {})
    with amos.store.transaction() as conn:
        assert amos.store.insert_edge(conn, stored_edge) is True
    proposal = MaintenanceProposal(
        processor_id="test.edge.race",
        processor_version="test.edge.race.v1",
        action="add_edge",
        risk_level="low",
        confidence=0.8,
        reason_code="test_edge_insert_race",
        source_refs=(source["id"], target["id"]),
        payload={
            "edge": {
                "source_ref": source["id"],
                "target_ref": target["id"],
                "relation": "rel:derived_from",
            }
        },
    ).to_dict()
    event_count = len(amos.store.list_events())
    monkeypatch.setattr(amos.store, "list_edges", lambda: [])

    result = amos.stewardship._commit_maintenance_edge_proposal(proposal, actor="test")

    assert result["status"] == "already_committed"
    assert len(amos.store.list_events()) == event_count
