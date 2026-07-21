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
    MaintenanceWindowRequest,
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


def test_external_processor_distills_supported_control_lesson(amos):
    amos.register_maintenance_processor(ExampleTrainingFlightProcessor())
    scope = {"tenant": "example", "component": "training"}
    control_signature = "trainable_roles=encoder; replay_ratio=0.3"
    directive = amos.commit_atom(
        {
            "id": "example_directive_chunk7",
            "type": "agentic_trace",
            "payload": {
                "agent_id": "trainer",
                "example_kind": "directive",
                "task": "example chunk 7",
                "action": "apply sampled control packet",
                "outcome": "issued",
                "target_chunk": 7,
                "control_signature": control_signature,
                "requested_controls": {
                    "trainable_roles": ["encoder"],
                    "replay_ratio": 0.3,
                },
                "applied_controls": {
                    "trainable_roles": ["encoder"],
                    "replay_ratio": 0.3,
                },
            },
            "scope": scope,
        }
    )["atom"]
    outcome = amos.commit_atom(
        {
            "id": "example_outcome_chunk7",
            "type": "agentic_trace",
            "payload": {
                "agent_id": "trainer",
                "example_kind": "reflection",
                "task": "example chunk 7",
                "action": "evaluate chunk outcome",
                "outcome": "supported",
                "chunk": 7,
                "control_signature": control_signature,
                "previous_score": 1.1,
                "score": 1.24,
            },
            "scope": scope,
        }
    )["atom"]

    result = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
        auto_commit_low_risk=True,
    )

    assert result["status"] == "completed"
    assert result["processors"][0]["processor_id"] == "example.training.flight.v1"
    assert any(
        proposal["reason_code"] == "example_supported_training_lesson"
        for proposal in result["proposals"]
    )
    committed = [
        item["atom"]
        for item in result["committed"]
        if item.get("atom", {}).get("payload", {}).get("distillation_type")
        == "example_training_lesson"
    ]
    assert committed
    atom = committed[0]
    assert atom["layer"] == "consolidated_long_term"
    assert atom["retention_class"] == "distilled"
    assert atom["payload"]["source_refs"] == [directive["id"], outcome["id"]]
    assert atom["payload"]["metric_deltas"]["score"] == pytest.approx(0.14)
    assert any(
        event["event_type"] == "maintenance_distillation_run"
        for event in amos.store.list_events()
    )

    event_count = len(amos.store.list_events())
    repeat = DistillerMaintenanceWorker(amos).tick(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
    )
    assert repeat["status"] == "skipped"
    assert repeat["reason"] == "all_proposals_already_committed"
    assert repeat["committed"][0]["status"] == "already_committed"
    assert len(amos.store.list_events()) == event_count


def test_external_processor_facets_project_generic_support_edge(amos):
    amos.register_maintenance_processor(ExampleTrainingFlightProcessor())
    scope = {"tenant": "example", "component": "training", "case": "support_edge"}
    signature = "trainable_roles=encoder; replay_ratio=0.3"
    for chunk in (1, 2):
        amos.commit_atom(
            {
                "id": f"facet_outcome_supported_{chunk}",
                "type": "agentic_trace",
                "payload": {
                    "example_kind": "reflection",
                    "task": f"example chunk {chunk}",
                    "action": "evaluate chunk outcome",
                    "outcome": "supported",
                    "chunk": chunk,
                    "control_signature": signature,
                    "score": 1.0 + chunk / 10.0,
                },
                "scope": scope,
                "confidence": {"level": "medium-high", "score": 0.78},
            }
        )

    result = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
        auto_commit_low_risk=True,
    )

    assert result["status"] == "completed"
    assert any(
        proposal["processor_id"] == "amos.semantic_relations.v1"
        and proposal["action"] == "add_edge"
        and proposal["payload"]["edge"]["relation"] == "rel:supports"
        for proposal in result["proposals"]
    )
    committed_edges = [item["edge"] for item in result["committed"] if item.get("edge")]
    assert {edge["relation"] for edge in committed_edges} == {"rel:supports"}
    assert any(event["event_type"] == "edge_committed" for event in amos.store.list_events())
    assert amos.verify_replay()["status"] == "ok"

    event_count = len(amos.store.list_events())
    repeat = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
        auto_commit_low_risk=True,
    )
    assert repeat["status"] == "skipped"
    assert repeat["reason"] == "no_proposals"
    assert len(amos.store.list_events()) == event_count


def test_external_processor_facets_auto_commit_generic_contradiction_edge(amos):
    amos.register_maintenance_processor(ExampleTrainingFlightProcessor())
    scope = {"tenant": "example", "component": "training", "case": "conflict_edge"}
    signature = "trainable_roles=head; replay_ratio=0.6"
    for atom_id, outcome, chunk in (
        ("facet_outcome_supported", "supported", 1),
        ("facet_outcome_failed", "failed", 2),
    ):
        amos.commit_atom(
            {
                "id": atom_id,
                "type": "agentic_trace",
                "payload": {
                    "example_kind": "reflection",
                    "task": f"example chunk {chunk}",
                    "action": "evaluate chunk outcome",
                    "outcome": outcome,
                    "chunk": chunk,
                    "control_signature": signature,
                    "score": 1.0,
                },
                "scope": scope,
                "confidence": {"level": "medium-high", "score": 0.8},
            }
        )

    result = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
        auto_commit_low_risk=True,
        reviewer={"enabled": True},
    )

    assert result["status"] == "completed"
    assert result["deferred"] == []
    assert result["proposals"][0]["action"] == "add_edge"
    assert result["proposals"][0]["risk_level"] == "low"
    assert result["proposals"][0]["payload"]["edge"]["relation"] == "rel:contradicts"
    committed_edges = [item["edge"] for item in result["committed"] if item.get("edge")]
    assert {edge["relation"] for edge in committed_edges} == {"rel:contradicts"}
    assert any(edge["relation"] == "rel:contradicts" for edge in amos.store.list_edges())
    assert amos.store.get_atom("facet_outcome_supported")["health_status"] == "healthy"
    assert amos.store.get_atom("facet_outcome_failed")["health_status"] == "healthy"
    assert amos.verify_replay()["status"] == "ok"


def test_generic_semantic_facets_propose_supersession_without_domain_rules():
    proposals = semantic_relation_proposals_from_facets(
        [
            SemanticFacet(
                atom_ref="old_lesson",
                subject="shared planner budget policy",
                intent="old budget interpretation",
                outcome="mixed",
                outcome_direction="mixed",
                confidence=0.55,
                time_index=1,
                scope={"tenant": "example"},
            ),
            SemanticFacet(
                atom_ref="new_lesson",
                subject="shared planner budget policy",
                intent="updated budget interpretation",
                outcome="supported",
                outcome_direction="positive",
                confidence=0.8,
                time_index=3,
                scope={"tenant": "example"},
            ),
        ]
    )

    assert len(proposals) == 1
    proposal = proposals[0].to_dict()
    assert proposal["action"] == "add_edge"
    assert proposal["risk_level"] == "low"
    assert proposal["payload"]["edge"]["relation"] == "rel:supersedes"
    assert proposal["payload"]["edge"]["confidence"]["score"] == 0.8
    assert proposal["payload"]["edge"]["source_ref"] == "new_lesson"
    assert proposal["payload"]["edge"]["target_ref"] == "old_lesson"


def test_external_processor_facets_auto_commit_generic_supersession_edge(amos):
    class SupersessionFacetProcessor:
        processor_id = "example.supersession_facets.v1"
        processor_version = "example.supersession_facets.v1"

        def supports(self, window):
            return window.domain == "example_supersession"

        def propose(self, window):
            return []

        def extract_facets(self, window):
            facets = []
            for atom in window.atoms:
                payload = atom["payload"]
                facets.append(
                    SemanticFacet(
                        atom_ref=atom["id"],
                        subject=payload["subject"],
                        intent=payload["intent"],
                        outcome=payload.get("outcome", "neutral"),
                        outcome_direction=payload.get("outcome_direction", "neutral"),
                        confidence=atom["confidence"]["score"],
                        time_index=payload["time_index"],
                        scope=atom["scope"],
                    )
                )
            return facets

    amos.register_maintenance_processor(SupersessionFacetProcessor())
    scope = {"tenant": "example", "case": "supersession_edge"}
    for atom_id, confidence, time_index in (
        ("old_policy_memory", 0.55, 1),
        ("new_policy_memory", 0.82, 3),
    ):
        amos.commit_atom(
            {
                "id": atom_id,
                "type": "semantic",
                "payload": {
                    "summary": f"{atom_id} for shared planner budget policy.",
                    "subject": "shared planner budget policy",
                    "intent": f"{atom_id} interpretation",
                    "outcome": "supported",
                    "outcome_direction": "positive",
                    "time_index": time_index,
                },
                "scope": scope,
                "confidence": {"level": "medium-high", "score": confidence},
            }
        )

    result = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_supersession",
        processor_ids=["example.supersession_facets.v1"],
        auto_commit_low_risk=True,
    )

    assert result["status"] == "completed"
    assert result["deferred"] == []
    committed_edges = [item["edge"] for item in result["committed"] if item.get("edge")]
    assert len(committed_edges) == 1
    edge = committed_edges[0]
    assert edge["relation"] == "rel:supersedes"
    assert edge["source_ref"] == "new_policy_memory"
    assert edge["target_ref"] == "old_policy_memory"
    assert edge["confidence"]["score"] == 0.82
    assert amos.store.get_atom("old_policy_memory")["lifecycle_state"] == "active"
    assert amos.verify_replay()["status"] == "ok"


def test_generic_semantic_facets_propose_similarity_for_neutral_matches():
    proposals = semantic_relation_proposals_from_facets(
        [
            SemanticFacet(
                atom_ref="first_note",
                subject="shared retrieval budget",
                intent="inspect packet budget",
                outcome="observed",
                outcome_direction="neutral",
                controls={"packet_budget": 5},
                confidence=0.7,
                scope={"tenant": "example"},
            ),
            SemanticFacet(
                atom_ref="second_note",
                subject="shared retrieval budget",
                intent="inspect packet budget",
                outcome="observed",
                outcome_direction="neutral",
                controls={"packet_budget": 5},
                confidence=0.72,
                scope={"tenant": "example"},
            ),
        ]
    )

    assert len(proposals) == 1
    proposal = proposals[0].to_dict()
    assert proposal["action"] == "add_edge"
    assert proposal["risk_level"] == "low"
    assert proposal["payload"]["edge"]["relation"] == "rel:similar_to"


def test_generic_semantic_facets_do_not_mix_context_domains():
    proposals = semantic_relation_proposals_from_facets(
        [
            SemanticFacet(
                atom_ref="pen_activity",
                subject="shared study",
                intent="practice bounded skill",
                outcome="observed",
                semantic_context_key="project:pen-art",
                scope={"tenant": "cogito"},
            ),
            SemanticFacet(
                atom_ref="trading_activity",
                subject="shared study",
                intent="practice bounded skill",
                outcome="observed",
                semantic_context_key="project:trading",
                scope={"tenant": "cogito"},
            ),
        ]
    )

    assert proposals == []


def test_generic_semantic_facets_do_not_infer_support_between_activities():
    proposals = semantic_relation_proposals_from_facets(
        [
            SemanticFacet(
                atom_ref="activity_one",
                subject="pen art",
                intent="practice hatching",
                outcome="improved",
                outcome_direction="positive",
                semantic_context_key="project:pen-art",
                attributes={"semantic_role": "project_activity"},
            ),
            SemanticFacet(
                atom_ref="activity_two",
                subject="pen art",
                intent="practice hatching",
                outcome="improved",
                outcome_direction="positive",
                semantic_context_key="project:pen-art",
                attributes={"semantic_role": "project_activity"},
            ),
        ]
    )

    assert proposals == []


def test_generic_semantic_graph_degree_is_bounded():
    proposals = semantic_relation_proposals_from_facets(
        [
            SemanticFacet(
                atom_ref=f"bounded_note_{index}",
                subject="shared bounded graph subject",
                intent="observe bounded graph construction",
                outcome="observed",
                outcome_direction="neutral",
                confidence=0.75,
                scope={"tenant": "bounded-graph"},
            )
            for index in range(10)
        ],
        max_new_relations_per_facet=3,
    )

    degree: dict[str, int] = {}
    for proposal in proposals:
        edge = proposal.to_dict()["payload"]["edge"]
        degree[edge["source_ref"]] = degree.get(edge["source_ref"], 0) + 1
        degree[edge["target_ref"]] = degree.get(edge["target_ref"], 0) + 1
    assert proposals
    assert max(degree.values()) <= 3


def test_canonical_atom_facets_build_graph_without_domain_processor(amos):
    scope = {"tenant": "canonical-facets"}
    for atom_id, time_index in (("canonical_note_one", 1), ("canonical_note_two", 2)):
        amos.commit_atom(
            {
                "id": atom_id,
                "type": "semantic",
                "payload": {
                    "summary": f"Canonical observation {time_index}",
                    "semantic_facets": [
                        {
                            "subject": "shared memory maintenance",
                            "intent": "observe graph construction",
                            "outcome": "supported",
                            "outcome_direction": "positive",
                            "time_index": time_index,
                        }
                    ],
                },
                "scope": scope,
                "confidence": {"level": "medium-high", "score": 0.8},
            }
        )

    result = amos.run_maintenance_distiller(
        scope=scope,
        domain="generic",
        processor_ids=["amos.maintenance.generic.v1"],
        auto_commit_low_risk=True,
    )

    assert any(
        proposal["processor_id"] == "amos.semantic_relations.v1"
        and proposal["payload"]["edge"]["relation"] == "rel:supports"
        for proposal in result["proposals"]
    )
    assert any(
        edge["relation"] == "rel:supports" for edge in amos.store.list_edges()
    )
    assert amos.verify_replay()["status"] == "ok"


def test_canonical_graph_relations_are_gated_and_reprocessable(amos):
    scope = {"tenant": "canonical-relations"}
    source = amos.commit_atom(
        {
            "id": "canonical_relation_source",
            "type": "semantic",
            "payload": {"summary": "Canonical source"},
            "scope": scope,
        }
    )["atom"]
    target = amos.commit_atom(
        {
            "id": "canonical_relation_target",
            "type": "semantic",
            "payload": {"summary": "Canonical target"},
            "scope": scope,
        }
    )["atom"]
    target = amos.update_atom(
        target["id"],
        payload_patch={
            "graph_relations": [
                {
                    "source_ref": "$self",
                    "target_ref": source["id"],
                    "relation": "rel:derived_from",
                },
                {
                    "source_ref": "$self",
                    "target_ref": source["id"],
                    "relation": "rel:caused_by",
                },
            ]
        },
        expected_version=target["version"],
    )["atom"]

    result = amos.run_maintenance_distiller(
        scope=scope,
        processor_ids=["amos.maintenance.generic.v1"],
        auto_commit_low_risk=True,
    )

    assert any(
        item.get("edge", {}).get("relation") == "rel:derived_from"
        for item in result["committed"]
    )
    assert any(
        item["action"] == "add_edge"
        and item["risk_level"] == "medium"
        and item["source_refs"] == [target["id"], source["id"]]
        for item in result["proposals"]
    )
    assert any(
        item["reason"] == "requires_review_or_unsupported_action"
        for item in result["deferred"]
    )

    committed_edge = next(
        item["edge"]
        for item in result["committed"]
        if item.get("edge", {}).get("relation") == "rel:derived_from"
    )
    amos.archive_atom(target["id"], reason="exercise graph reprocessing")
    archived_edge = amos.store.get_edge(committed_edge["edge_id"])
    assert bool(archived_edge["deleted"]) is True
    current = amos.store.get_atom(target["id"])
    promoted = amos.update_atom(
        target["id"],
        set_fields={"lifecycle_state": "active", "health_status": "healthy"},
        expected_version=current["version"],
    )
    assert any(
        edge["relation"] == "rel:derived_from"
        for edge in promoted["projected_edges"]
    )

    reprocessed = amos.run_maintenance_distiller(
        scope=scope,
        processor_ids=["amos.maintenance.generic.v1"],
        auto_commit_low_risk=True,
    )

    assert not any(
        item.get("edge", {}).get("relation") == "rel:derived_from"
        for item in reprocessed["committed"]
    )
    revived = amos.store.get_edge(committed_edge["edge_id"])
    assert bool(revived["deleted"]) is False
    assert revived["version"] > archived_edge["version"]


def test_proposed_canonical_facets_remain_dormant_until_promotion(amos):
    scope = {"tenant": "canonical-proposal"}
    active = amos.commit_atom(
        {
            "id": "canonical_active_facet",
            "type": "semantic",
            "payload": {
                "summary": "Reviewed graph observation",
                "semantic_facets": [
                    {
                        "subject": "proposal lifecycle",
                        "intent": "observe promotion",
                        "outcome_direction": "positive",
                    }
                ],
            },
            "scope": scope,
        }
    )["atom"]
    proposed = amos.propose_memory_atoms(
        [
            {
                "id": "canonical_proposed_facet",
                "type": "semantic",
                "payload": {
                    "summary": "Unreviewed graph observation",
                    "semantic_facets": [
                        {
                            "subject": "proposal lifecycle",
                            "intent": "observe promotion",
                            "outcome_direction": "positive",
                        }
                    ],
                },
            }
        ],
        scope=scope,
    )["proposals"][0]["atom"]

    amos.run_maintenance_distiller(
        scope=scope,
        processor_ids=["amos.maintenance.generic.v1"],
    )
    assert not any(
        {edge["source_ref"], edge["target_ref"]} == {active["id"], proposed["id"]}
        for edge in amos.store.list_edges()
    )

    amos.update_atom(
        proposed["id"],
        set_fields={"lifecycle_state": "active"},
        expected_version=proposed["version"],
    )
    amos.run_maintenance_distiller(
        scope=scope,
        processor_ids=["amos.maintenance.generic.v1"],
    )
    assert any(
        edge["relation"] == "rel:supports"
        and {edge["source_ref"], edge["target_ref"]}
        == {active["id"], proposed["id"]}
        for edge in amos.store.list_edges()
    )


def test_maintenance_distiller_skips_empty_windows_without_journal_event(amos):
    event_count = len(amos.store.list_events())

    result = amos.run_maintenance_distiller(scope={"tenant": "empty"})

    assert result["status"] == "skipped"
    assert result["reason"] == "no_proposals"
    assert result["event"] is None
    assert len(amos.store.list_events()) == event_count


def test_maintenance_evidence_window_prioritizes_hot_atoms_over_recent_archives(amos):
    scope = {"tenant": "window-priority"}
    active = amos.commit_atom(
        {
            "id": "window_active_anchor",
            "type": "self_model",
            "payload": {"agent_id": "window-agent", "name": "active anchor"},
            "scope": scope,
        }
    )["atom"]
    for index in range(3):
        atom_id = f"window_recent_archive_{index}"
        amos.commit_atom(
            {
                "id": atom_id,
                "type": "semantic",
                "payload": {"summary": f"recent archive {index}"},
                "scope": scope,
            }
        )
        amos.archive_atom(atom_id, reason="window priority regression")

    window = amos.stewardship._maintenance_evidence_window(
        scope=scope,
        domain="generic",
        max_atoms=2,
        max_events=0,
        max_retrieval_outcomes=0,
    )

    assert window.atoms[0]["id"] == active["id"]
    assert window.atoms[0]["lifecycle_state"] == "active"
    assert len(window.atoms) == 2


def test_processor_specific_workset_is_typed_bounded_and_reports_coverage(amos):
    class TypedProcessor:
        processor_id = "test.typed-workset.v1"
        processor_version = "test.typed-workset.v1"

        def window_request(self, *, scope, domain):
            assert domain == "typed-domain"
            return MaintenanceWindowRequest(
                lifecycle_states=("proposed",),
                atom_types=("episode",),
                max_atoms=1,
                include_evidence=True,
                include_events=False,
                include_retrieval_outcomes=False,
                max_evidence=4,
            )

        def supports(self, window):
            return bool(window.atoms)

        def propose(self, window):
            return []

    scope = {"tenant": "workset"}
    for index in range(2):
        captured = amos.capture_event(
            source_type="test",
            source_ref=f"workset-source-{index}",
            payload={"index": index},
            actor="test",
            scope={**scope, "run_id": f"run-{index}"},
            idempotency_key=f"workset-evidence-{index}",
        )
        amos.propose_memory_atoms(
            [
                {
                    "id": f"workset_episode_{index}",
                    "type": "episode",
                    "payload": {
                        "task": "typed workset",
                        "origin": "test",
                        "maintenance_hints": {"profile": "test.profile"},
                    },
                    "evidence_refs": [captured["evidence"]["evidence_id"]],
                    "scope": {**scope, "run_id": f"run-{index}"},
                }
            ],
            actor="test",
            scope=scope,
        )
    amos.commit_atom(
        {
            "id": "workset_active_semantic",
            "type": "semantic",
            "payload": {"summary": "Must be excluded by typed workset."},
            "scope": scope,
        }
    )
    processor = TypedProcessor()
    amos.register_maintenance_processor(processor)

    result = amos.run_maintenance_distiller(
        scope=scope,
        domain="typed-domain",
        processor_ids=[processor.processor_id],
        max_atoms=64,
        max_events=64,
        max_retrieval_outcomes=64,
    )

    window = result["processor_windows"][processor.processor_id]
    assert window["atom_count"] == 1
    assert window["evidence_count"] == 2
    assert window["event_count"] == 0
    assert window["coverage"]["candidate_atom_count"] == 2
    assert window["coverage"]["truncated_atom_count"] == 1
    assert window["coverage"]["missing_referenced_evidence"] == []


def test_maintenance_window_hierarchically_includes_narrow_evidence(amos):
    scope = {"tenant": "hierarchical-evidence"}
    captured = amos.capture_event(
        source_type="test",
        source_ref="narrow-evidence-source",
        payload={"value": 1},
        actor="test",
        scope={**scope, "run_id": "narrow-run"},
        idempotency_key="narrow-evidence",
    )
    amos.commit_atom(
        {
            "id": "narrow_evidence_atom",
            "type": "semantic",
            "payload": {"summary": "Narrow evidence remains visible to broad maintenance."},
            "evidence_refs": [captured["evidence"]["evidence_id"]],
            "scope": {**scope, "run_id": "narrow-run"},
        }
    )

    window = amos.stewardship._maintenance_evidence_window(
        scope=scope,
        domain="generic",
        max_atoms=8,
        max_events=0,
        max_retrieval_outcomes=0,
        request=MaintenanceWindowRequest(),
    )

    assert [item["evidence_id"] for item in window.evidence] == [
        captured["evidence"]["evidence_id"]
    ]


def test_maintenance_evidence_window_includes_narrower_scopes_before_limit(amos):
    for index in range(3):
        amos.commit_atom(
            {
                "id": f"unrelated_hot_{index}",
                "type": "semantic",
                "payload": {"summary": f"unrelated {index}"},
                "scope": {"tenant": "other", "component": "training"},
            }
        )
    broad = amos.commit_atom(
        {
            "id": "hierarchical_broad",
            "type": "semantic",
            "payload": {"summary": "broad evidence"},
            "scope": {"tenant": "hierarchical", "component": "training"},
        }
    )["atom"]
    narrow = amos.commit_atom(
        {
            "id": "hierarchical_narrow",
            "type": "semantic",
            "payload": {"summary": "run evidence"},
            "scope": {
                "tenant": "hierarchical",
                "component": "training",
                "asset": "UPRO",
                "run_id": "run-1",
            },
        }
    )["atom"]

    window = amos.stewardship._maintenance_evidence_window(
        scope={"tenant": "hierarchical", "component": "training"},
        domain="generic",
        max_atoms=2,
        max_events=0,
        max_retrieval_outcomes=0,
    )

    assert {atom["id"] for atom in window.atoms} == {broad["id"], narrow["id"]}

    exact_window = amos.stewardship._maintenance_evidence_window(
        scope={
            "tenant": "hierarchical",
            "component": "training",
            "asset": "UPRO",
            "run_id": "run-1",
        },
        domain="generic",
        max_atoms=2,
        max_events=0,
        max_retrieval_outcomes=0,
    )
    assert {atom["id"] for atom in exact_window.atoms} == {
        broad["id"],
        narrow["id"],
    }


def test_maintenance_distiller_skips_unchanged_deferred_proposals(amos):
    amos.register_maintenance_processor(ExampleTrainingFlightProcessor())
    scope = {"tenant": "example", "component": "training", "case": "deferred"}
    normal_signature = "trainable_roles=encoder; replay_ratio=0.3"
    sanitized_signature = "trainable_roles=head; replay_ratio=0.6"
    for atom in (
        {
            "id": "deferred_directive_normal",
            "type": "agentic_trace",
            "payload": {
                "example_kind": "directive",
                "task": "normal directive",
                "action": "issue directive",
                "outcome": "issued",
                "control_signature": normal_signature,
            },
            "scope": scope,
        },
        {
            "id": "deferred_outcome_normal",
            "type": "agentic_trace",
            "payload": {
                "example_kind": "reflection",
                "task": "normal reflection",
                "action": "evaluate outcome",
                "outcome": "supported",
                "control_signature": normal_signature,
                "previous_score": 1.0,
                "score": 1.2,
            },
            "scope": scope,
        },
        {
            "id": "deferred_directive_sanitized",
            "type": "agentic_trace",
            "payload": {
                "example_kind": "directive",
                "task": "sanitized directive",
                "action": "issue directive",
                "outcome": "issued",
                "control_signature": sanitized_signature,
                "sanitized_controls": True,
            },
            "scope": scope,
        },
        {
            "id": "deferred_outcome_sanitized",
            "type": "agentic_trace",
            "payload": {
                "example_kind": "reflection",
                "task": "sanitized reflection",
                "action": "evaluate outcome",
                "outcome": "failed",
                "control_signature": sanitized_signature,
                "previous_score": 1.0,
                "score": 0.8,
            },
            "scope": scope,
        },
    ):
        amos.commit_atom(atom)

    first = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
        auto_commit_low_risk=True,
        reviewer={"enabled": True},
    )
    event_count = len(amos.store.list_events())

    assert first["status"] == "completed"
    assert first["committed"][0]["status"] == "committed"
    assert len(first["deferred"]) == 1
    assert first["event"] is not None
    assert first["deferred_fingerprint"]

    repeat = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
        auto_commit_low_risk=True,
        reviewer={"enabled": True},
    )

    assert repeat["status"] == "skipped"
    assert repeat["reason"] == "deferred_proposals_unchanged"
    assert repeat["deferred_fingerprint"] == first["deferred_fingerprint"]
    assert repeat["event"] is None
    assert len(amos.store.list_events()) == event_count


def test_external_processor_defers_sanitized_control_claim(amos):
    amos.register_maintenance_processor(ExampleTrainingFlightProcessor())
    scope = {"tenant": "example", "component": "training"}
    control_signature = "candidate_id=stale; replay_ratio=0.3"
    amos.commit_atom(
        {
            "id": "example_directive_sanitized",
            "type": "agentic_trace",
            "payload": {
                "agent_id": "trainer",
                "example_kind": "directive",
                "task": "example chunk 8",
                "action": "apply sampled control packet",
                "outcome": "issued",
                "target_chunk": 8,
                "control_signature": control_signature,
                "requested_controls": {
                    "candidate_id": "stale",
                    "replay_ratio": 0.3,
                },
                "applied_controls": {"replay_ratio": 0.3},
                "sanitized_controls": {"dropped": ["candidate_id"]},
            },
            "scope": scope,
        }
    )
    amos.commit_atom(
        {
            "id": "example_outcome_sanitized",
            "type": "agentic_trace",
            "payload": {
                "agent_id": "trainer",
                "example_kind": "reflection",
                "task": "example chunk 8",
                "action": "evaluate chunk outcome",
                "outcome": "supported",
                "chunk": 8,
                "control_signature": control_signature,
                "previous_score": 1.0,
                "score": 1.07,
            },
            "scope": scope,
        }
    )

    result = amos.run_maintenance_distiller(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
        auto_commit_low_risk=True,
        reviewer={"enabled": True},
    )

    assert result["reviewer"]["authority"] == "draft_only"
    assert result["reviewer"]["mutates_canonical_memory"] is False
    assert result["committed"] == []
    assert result["deferred"]
    assert result["proposals"][0]["action"] == "review_required"
    assert "sanitized_controls_present" in result["proposals"][0]["payload"]["confounders"]


def test_external_processor_import_path_loading(tmp_path, monkeypatch):
    plugin = tmp_path / "demo_processor_plugin.py"
    plugin.write_text(
        "\n".join(
            [
                "class LoadedProcessor:",
                "    processor_id = 'loaded.processor.v1'",
                "    processor_version = 'loaded.processor.v1'",
                "    def supports(self, window):",
                "        return False",
                "    def propose(self, window):",
                "        return []",
                "",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    service = Amos(
        tmp_path / "amos.sqlite3",
        maintenance_processor_paths=["demo_processor_plugin:LoadedProcessor"],
    )
    try:
        processors = service.list_maintenance_processors()["processors"]
    finally:
        service.close()

    assert {
        processor["processor_id"] for processor in processors
    } == {"amos.maintenance.generic.v1", "loaded.processor.v1"}
