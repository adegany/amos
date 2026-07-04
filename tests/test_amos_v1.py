from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from amos import (
    AccessDenied,
    AgenticRecallAuditor,
    Amos,
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
    SemanticMaintenanceProcessor,
    SelfModelCalibrator,
    ValidationError,
    ontology_snapshot,
)
from amos.cli import main as cli_main
from amos.http_api import AmosHTTPServer


@pytest.fixture()
def amos(tmp_path):
    service = Amos(tmp_path / "amos.sqlite3")
    try:
        yield service
    finally:
        service.close()


def item_refs(packet):
    return {item["atom_ref"] for item in packet["items"]}


class ExampleTrainingFlightProcessor:
    processor_id = "example.training.flight.v1"
    processor_version = "example.training.flight.v1"

    def supports(self, window):
        return window.domain == "example_training"

    def propose(self, window):
        directives = [
            atom
            for atom in window.atoms
            if atom.get("payload", {}).get("example_kind") == "directive"
        ]
        outcomes = [
            atom
            for atom in window.atoms
            if atom.get("payload", {}).get("example_kind") == "reflection"
        ]
        proposals = []
        for directive in directives:
            directive_payload = directive["payload"]
            signature = directive_payload.get("control_signature")
            if not signature:
                continue
            for outcome in outcomes:
                outcome_payload = outcome["payload"]
                if outcome_payload.get("control_signature") != signature:
                    continue
                source_refs = (directive["id"], outcome["id"])
                if directive_payload.get("sanitized_controls"):
                    proposals.append(
                        MaintenanceProposal(
                            processor_id=self.processor_id,
                            processor_version=self.processor_version,
                            action="review_required",
                            risk_level="medium",
                            confidence=0.7,
                            reason_code="example_sanitized_control_claim",
                            source_refs=source_refs,
                            target_refs=source_refs,
                            payload={
                                "confounders": ["sanitized_controls_present"],
                                "control_signature": signature,
                            },
                        )
                    )
                    continue
                previous = outcome_payload.get("previous_score")
                current = outcome_payload.get("score")
                if not isinstance(previous, (int, float)) or not isinstance(
                    current, (int, float)
                ):
                    continue
                proposals.append(
                    MaintenanceProposal(
                        processor_id=self.processor_id,
                        processor_version=self.processor_version,
                        action="add_atom",
                        risk_level="low",
                        confidence=0.82,
                        reason_code="example_supported_training_lesson",
                        source_refs=source_refs,
                        payload={
                            "atom": {
                                "type": "semantic",
                                "payload": {
                                    "distillation_type": "example_training_lesson",
                                    "summary": (
                                        "Example training controls produced "
                                        f"score_delta={current - previous:+.3f}."
                                    ),
                                    "source_refs": list(source_refs),
                                    "control_signature": signature,
                                    "metric_deltas": {
                                        "score": round(current - previous, 6)
                                    },
                                },
                                "scope": dict(window.scope),
                                "layer": "consolidated_long_term",
                                "retention_class": "distilled",
                                "confidence": {
                                    "level": "medium-high",
                                    "score": 0.78,
                                },
                            }
                        },
                    )
                )
        return proposals


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


def test_procedural_memory_is_advisory_and_autonomous_execution_denied(amos):
    procedure = amos.commit_atom(
        {
            "id": "procedure_restart",
            "type": "procedure",
            "payload": {
                "trigger_context": "optimizer stalled",
                "steps": ["inspect status", "restart supervisor"],
            },
        }
    )["atom"]
    advisory = amos.evaluate_procedure_execution(procedure_ref=procedure["id"])
    assert advisory["status"] == "review_required"
    assert "approved_by" in advisory["missing"]

    denied = amos.evaluate_procedure_execution(
        procedure_ref=procedure["id"], autonomous=True
    )
    assert denied["status"] == "denied"
    assert denied["reason"] == "autonomous_external_state_execution_not_allowed_in_v1"

    eligible = amos.evaluate_procedure_execution(
        procedure_ref=procedure["id"],
        approved_by="operator",
        tool_permission_binding={"tool": "systemctl", "permission": "restart-supervisor"},
        preconditions_satisfied=True,
        rollback_plan={"steps": ["restore prior supervisor config"]},
        review_status="approved",
    )
    assert eligible["status"] == "eligible_for_external_executor"


def test_llm_reviewer_default_policy_is_disabled_and_non_authoritative(amos):
    policy = amos.llm_reviewer_policy()
    assert policy["enabled_by_default"] is False
    assert "direct_canonical_mutation" in policy["forbidden"]
    assert "deletion_approval" in policy["forbidden"]
    assert "recommended_action" in policy["output_envelope"]


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


def test_retrieval_packets_include_score_components_and_evidence_omissions(amos):
    evidence = amos.capture_event(
        source_type="log",
        source_ref="secret-evidence",
        payload={"detail": "private supporting evidence"},
        scope={"tenant": "qandl"},
    )["evidence"]
    atom = amos.commit_atom(
        {
            "id": "atom_public_private_evidence",
            "type": "belief",
            "payload": {"claim": "Public claim with private evidence"},
            "evidence_refs": [evidence["evidence_id"]],
            "scope": {"tenant": "qandl"},
            "access_policy": {
                "visibility": ["all"],
                "evidence_visibility": ["critic"],
                "mutable_by": ["owner"],
            },
        }
    )["atom"]

    reasoner_packet = amos.retrieve_packet(
        cues=["private evidence"],
        scope={"tenant": "qandl"},
        target_processor="reasoner",
    )
    item = next(item for item in reasoner_packet["items"] if item["atom_ref"] == atom["id"])
    assert item["score_components"]
    assert "semantic_similarity" in item["score_components"]
    assert item["evidence_refs"] == []
    assert item["access_decision"]["evidence"] == "denied"
    assert any(o["reason"] == "evidence_access_denied" for o in reasoner_packet["omissions"])
    assert reasoner_packet["degradation"]["omitted_evidence_detail"] is True

    critic_packet = amos.retrieve_packet(
        cues=["private evidence"],
        scope={"tenant": "qandl"},
        target_processor="critic",
    )
    critic_item = next(item for item in critic_packet["items"] if item["atom_ref"] == atom["id"])
    assert critic_item["evidence_refs"] == [evidence["evidence_id"]]


def test_retrieval_outcome_telemetry_is_reportable(amos):
    atom = amos.commit_atom(
        {
            "id": "outcome_atom",
            "type": "belief",
            "payload": {"claim": "retrieval outcome telemetry works"},
        }
    )["atom"]
    packet = amos.retrieve_packet(cues=["retrieval outcome"])
    assert amos.health_memory()["retrieval_outcomes"] == 0

    outcome = amos.record_retrieval_outcome(
        packet_id=packet["packet_id"],
        request=packet["request"],
        outcome={
            "used_item_refs": [atom["id"]],
            "label": "useful",
            "correction_refs": [],
        },
    )

    assert outcome["packet_id"] == packet["packet_id"]
    assert outcome["outcome_id"].startswith("rto_")
    assert amos.health_memory()["retrieval_outcomes"] == 1


def test_retrieval_ranking_scoped_preference_beats_generic(amos):
    amos.commit_atom(
        {
            "id": "generic_preference",
            "type": "preference",
            "payload": {
                "holder": "trainer",
                "polarity": "prefer",
                "target": "advisor fallback",
                "applicability_scope": "any",
                "strength": "medium",
            },
            "utility": 0.6,
            "scope": {},
        }
    )
    amos.commit_atom(
        {
            "id": "scoped_preference",
            "type": "preference",
            "payload": {
                "holder": "trainer",
                "polarity": "prefer",
                "target": "advisor fallback",
                "applicability_scope": "qandl training",
                "strength": "medium",
            },
            "utility": 0.6,
            "scope": {"tenant": "qandl", "component": "training"},
        }
    )
    packet = amos.retrieve_packet(
        cues=["advisor fallback"],
        scope={"tenant": "qandl", "component": "training"},
    )
    assert packet["items"][0]["atom_ref"] == "scoped_preference"
    assert (
        packet["items"][0]["score_components"]["scope_specificity"]
        > packet["items"][1]["score_components"]["scope_specificity"]
    )


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


def test_self_awareness_suppresses_runtime_unavailable_capability(amos):
    amos.commit_atom(
        {
            "type": "self_model",
            "payload": {"agent_id": "trainer", "name": "Qandl trainer agent"},
            "scope": {"tenant": "qandl"},
        }
    )
    cap = amos.commit_atom(
        {
            "id": "cap_codex_directive",
            "type": "capability",
            "payload": {
                "agent_id": "trainer",
                "name": "codex_directive",
                "description": "ask Codex for live directives",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    amos.commit_atom(
        {
            "type": "limitation",
            "payload": {
                "agent_id": "trainer",
                "name": "external_directive_outage",
                "description": "Codex calls may time out",
            },
            "scope": {"tenant": "qandl"},
        }
    )
    amos.record_runtime_state(
        agent_id="trainer",
        capabilities={"codex_directive": {"available": False}},
        denied_capabilities=["codex_directive"],
        constraints=["offline fallback required"],
        scope={"tenant": "qandl"},
    )

    view = amos.retrieve_self_awareness(
        agent_id="trainer",
        scope={"tenant": "qandl"},
    )
    assert cap["id"] not in {item["atom_ref"] for item in view["capabilities"]}
    assert any(
        o["atom_ref"] == cap["id"]
        and o["reason"] == "capability_unavailable_in_runtime_state"
        for o in view["omissions"]
    )
    assert view["limitations"]
    assert view["runtime_state"]["payload"]["constraints"] == ["offline fallback required"]


def test_self_awareness_accepts_spec_native_subject_agent_aliases(amos):
    amos.commit_atom(
        {
            "type": "self_model",
            "payload": {"subject_agent": "trainer", "role": "training supervisor"},
            "scope": {"tenant": "qandl"},
        }
    )
    capability = amos.commit_atom(
        {
            "id": "cap_alias_local_advisor",
            "type": "capability",
            "payload": {
                "subject_agent": "trainer",
                "capability": "local_advisor",
                "description": "render local advisory packets",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    limitation = amos.commit_atom(
        {
            "id": "lim_alias_codex_outage",
            "type": "limitation",
            "payload": {
                "subject_agent": "trainer",
                "limitation": "codex_directive_unavailable",
                "description": "external directives may time out",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    runtime = amos.commit_atom(
        {
            "id": "runtime_alias",
            "type": "runtime_state",
            "payload": {
                "subject_agent": "trainer",
                "capabilities": {"local_advisor": {"available": True}},
                "constraints": ["service-owned SQLite"],
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    amos.commit_atom(
        {
            "id": "cap_other_alias",
            "type": "capability",
            "payload": {"subject_agent": "pilot", "capability": "pause_runner"},
            "scope": {"tenant": "qandl"},
        }
    )

    view = amos.retrieve_self_awareness(
        agent_id="trainer",
        scope={"tenant": "qandl"},
    )
    assert capability["id"] in {item["atom_ref"] for item in view["capabilities"]}
    assert limitation["id"] in {item["atom_ref"] for item in view["limitations"]}
    assert view["runtime_state"]["atom_ref"] == runtime["id"]
    assert any(
        omission["atom_ref"] == "cap_other_alias"
        and omission["reason"] == "different_agent"
        for omission in view["omissions"]
    )


def test_self_awareness_tracks_open_commitments_and_calibrates_claims(amos):
    amos.commit_atom(
        {
            "id": "cap_unverified_restart",
            "type": "capability",
            "payload": {
                "agent_id": "trainer",
                "name": "restart_optimizer",
                "description": "restart optimizer services",
            },
            "scope": {"tenant": "qandl"},
        }
    )
    commitment = amos.commit_atom(
        {
            "id": "commit_review",
            "type": "commitment",
            "payload": {
                "agent_id": "trainer",
                "description": "review next supervisor report",
                "status": "open",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]
    fulfilled = amos.commit_atom(
        {
            "id": "commit_done",
            "type": "commitment",
            "payload": {
                "agent_id": "trainer",
                "description": "old done item",
                "status": "fulfilled",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]

    view = amos.retrieve_self_awareness(
        agent_id="trainer",
        scope={"tenant": "qandl"},
    )
    assert commitment["id"] in {item["atom_ref"] for item in view["open_commitments"]}
    assert fulfilled["id"] not in {item["atom_ref"] for item in view["open_commitments"]}
    assert view["calibration"]["overconfident_claim_rate"] == 1.0
    recorded = amos.calibrate_self_model(
        agent_id="trainer",
        scope={"tenant": "qandl"},
        record=True,
    )
    assert recorded["assessment"]["atom"]["payload"]["calibration"][
        "unverified_capabilities"
    ] == ["restart_optimizer"]


def test_agentic_recall_balances_success_failure_blocked_and_corrections(amos):
    amos.record_agentic_trace(
        agent_id="trainer",
        task="UPRO restart",
        action="used preserved champion",
        outcome="success",
        lesson="preserved champion starts faster",
        scope={"tenant": "qandl"},
    )
    amos.record_action_outcome(
        agent_id="trainer",
        action_ref="UPRO_codex_directive",
        status="failed",
        correction="fall back to local advisor after timeout",
        limitation="Codex unavailable",
        scope={"tenant": "qandl"},
    )
    amos.record_action_outcome(
        agent_id="trainer",
        action_ref="UPRO_systemctl_restart",
        status="blocked",
        limitation="operator approval required",
        scope={"tenant": "qandl"},
    )

    recall = amos.retrieve_agentic_recall(
        agent_id="trainer",
        cues=["UPRO"],
        scope={"tenant": "qandl"},
    )
    assert recall["successes"]
    assert recall["failures"]
    assert recall["blocked"]
    assert recall["corrections"][0]["payload"]["correction"] == "fall back to local advisor after timeout"


def test_agentic_recall_labels_other_agent_and_shared_system_attribution(amos):
    amos.record_agentic_trace(
        agent_id="trainer",
        task="UPRO review",
        action="issued local advisory",
        outcome="success",
        scope={"tenant": "qandl"},
    )
    amos.record_agentic_trace(
        agent_id="pilot",
        task="UPRO review",
        action="paused runner",
        outcome="success",
        scope={"tenant": "qandl"},
    )
    amos.commit_atom(
        {
            "type": "agentic_trace",
            "payload": {
                "agent_id": "pilot",
                "task": "UPRO review",
                "action": "combined pilot and trainer decision",
                "outcome": "success",
                "responsibility": "shared_system",
            },
            "scope": {"tenant": "qandl"},
        }
    )
    amos.commit_atom(
        {
            "type": "agentic_trace",
            "payload": {
                "task": "UPRO review",
                "action": "exchange closed during attempted validation",
                "outcome": "blocked",
                "external_constraints": ["market data provider unavailable"],
            },
            "scope": {"tenant": "qandl"},
        }
    )
    amos.commit_atom(
        {
            "type": "agentic_trace",
            "payload": {
                "task": "UPRO review",
                "action": "unattributed prior advisory",
                "outcome": "success",
            },
            "scope": {"tenant": "qandl"},
        }
    )

    recall = amos.retrieve_agentic_recall(
        agent_id="trainer",
        cues=["UPRO review"],
        scope={"tenant": "qandl"},
    )
    assert recall["successes"]
    assert recall["other_agent_actions"]
    assert recall["other_agent_actions"][0]["responsibility"] == "other_agent"
    assert recall["shared_system_actions"]
    assert recall["shared_system_actions"][0]["responsibility"] == "shared_system"
    assert recall["external_actions"]
    assert recall["external_actions"][0]["responsibility"] == "external"
    assert recall["unknown_responsibility_actions"]
    assert recall["unknown_responsibility_actions"][0]["responsibility"] == "unknown"
    assert "market data provider unavailable" in recall["external_constraints"]


def test_agentic_recall_accepts_spec_native_subject_agent_alias(amos):
    trace = amos.commit_atom(
        {
            "type": "agentic_trace",
            "payload": {
                "subject_agent": "trainer",
                "task": "alias audit",
                "action": "used subject_agent payload",
                "outcome": "success",
            },
            "scope": {"tenant": "qandl"},
        }
    )["atom"]

    recall = amos.retrieve_agentic_recall(
        agent_id="trainer",
        cues=["alias audit"],
        scope={"tenant": "qandl"},
    )
    assert trace["id"] in {item["atom_ref"] for item in recall["self_actions"]}
    item = next(item for item in recall["self_actions"] if item["atom_ref"] == trace["id"])
    assert item["score_components"]["agency_match"] == 1.0


def test_self_narrative_expires_after_later_counterevidence(amos):
    amos.record_agentic_trace(
        agent_id="trainer",
        task="directive review",
        action="used local advisor",
        outcome="success",
        lesson="local fallback works",
        scope={"tenant": "qandl"},
    )
    narrative = amos.generate_self_narrative(
        agent_id="trainer",
        narrative="I reliably handle directive reviews without interruption.",
        scope={"tenant": "qandl"},
    )["atom"]
    before = amos.retrieve_agentic_recall(
        agent_id="trainer",
        scope={"tenant": "qandl"},
    )
    assert narrative["id"] in {item["atom_ref"] for item in before["self_narratives"]}

    amos.record_action_outcome(
        agent_id="trainer",
        action_ref="directive_review",
        status="failed",
        correction="ask for review after repeated outage",
        limitation="external reviewer unavailable",
        scope={"tenant": "qandl"},
    )
    after = amos.retrieve_agentic_recall(
        agent_id="trainer",
        scope={"tenant": "qandl"},
    )
    assert narrative["id"] in {
        item["atom_ref"] for item in after["expired_self_narratives"]
    }
    assert any(
        omission["atom_ref"] == narrative["id"]
        and omission["reason"] == "self_narrative_drift"
        for omission in after["omissions"]
    )


def test_shared_view_respects_processor_overlays(amos):
    evidence = amos.capture_event(
        source_type="log",
        source_ref="handoff-secret",
        payload={"detail": "planner-only evidence"},
        scope={"tenant": "qandl"},
    )["evidence"]
    amos.commit_atom(
        {
            "id": "atom_shared_handoff",
            "type": "procedure",
            "payload": {
                "name": "handoff",
                "trigger_context": "planner critic handoff",
                "steps": ["read status", "render advisory"],
            },
            "evidence_refs": [evidence["evidence_id"]],
            "scope": {"tenant": "qandl"},
            "access_policy": {
                "visibility": ["all"],
                "evidence_visibility": ["planner"],
                "mutable_by": ["owner"],
            },
        }
    )
    amos.commit_atom(
        {
            "id": "atom_planner_only",
            "type": "belief",
            "payload": {"claim": "planner-only handoff detail"},
            "scope": {"tenant": "qandl"},
            "access_policy": {"visibility": ["planner"], "mutable_by": ["owner"]},
        }
    )

    view = amos.retrieve_shared_view(
        processor_ids=["planner", "critic"],
        cues=["handoff"],
        scope={"tenant": "qandl"},
    )
    assert view["common_graph_version"] == amos.store.graph_version()
    assert "atom_shared_handoff" in view["per_processor_overlays"]["planner"]
    assert "atom_shared_handoff" in view["per_processor_overlays"]["critic"]
    assert "atom_planner_only" in view["per_processor_overlays"]["planner"]
    assert "atom_planner_only" not in view["per_processor_overlays"]["critic"]
    shared_item = next(
        item for item in view["items"] if item["atom_ref"] == "atom_shared_handoff"
    )
    assert shared_item["evidence_refs"] == []
    assert shared_item["shared_visibility"]["evidence_policy"] == "least_common_denominator"
    assert any(
        omission["atom_ref"] == "atom_shared_handoff"
        and omission["reason"] == "evidence_access_denied"
        for omission in view["omissions_by_identity"]["critic"]
    )


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


def test_capacity_pressure_degrades_packets_and_reports_mode(amos):
    for index in range(6):
        amos.commit_atom(
            {
                "id": f"capacity_atom_{index}",
                "type": "belief",
                "payload": {"claim": f"capacity pressure recall item {index}"},
            }
        )
    amos.configure_capacity_budget(hard_capacity_bytes=1)
    capacity = amos.health_capacity()
    assert capacity["pressure_mode"] == "red"

    packet = amos.retrieve_packet(cues=["capacity pressure recall"], max_items=10)
    assert packet["pressure_mode"] == "red"
    assert packet["degradation"]["reduced_recall_depth"] is True
    assert len(packet["items"]) == 3
    assert any(o["reason"] == "pressure_degraded" for o in packet["omissions"])


def test_worker_artifacts_update_indexes_and_observability(amos):
    amos.commit_atom(
        {
            "id": "worker_capability",
            "type": "capability",
            "payload": {"agent_id": "trainer", "name": "worker_test"},
            "scope": {"tenant": "qandl"},
        }
    )
    amos.record_agentic_trace(
        agent_id="trainer",
        task="worker audit",
        action="ran worker",
        outcome="success",
        scope={"tenant": "qandl"},
    )
    index = IndexMaintainer(amos).rebuild()
    assert {item["index_name"] for item in index["indexes"]} == {
        "graph_adjacency",
        "semantic_lexical_vectors",
    }
    health = amos.health_memory()
    assert health["projection_lag"] == 0
    assert "semantic_lexical_vectors" in health["index_freshness"]

    packet = amos.retrieve_packet(cues=["worker"])
    assert amos.store.list_packet_cache()
    assert PacketCacheInvalidator(amos).invalidate()["status"] == "invalidated"
    assert amos.store.list_packet_cache() == []

    assert JournalProjector(amos).verify_projection()["replay"]["status"] == "ok"
    assert CapacityGovernor(amos).report()["pressure_mode"] == "green"
    assert MemorySteward(amos).run(scope={"tenant": "qandl"})["status"] == "completed"
    assert SelfModelCalibrator(amos).run(
        agent_id="trainer", scope={"tenant": "qandl"}
    )["status"] == "calibrated"
    assert AgenticRecallAuditor(amos).audit(
        agent_id="trainer", cues=["worker"], scope={"tenant": "qandl"}
    )["balance"]["success_count"] == 1
    assert SMPWorker(amos).run(scope={"tenant": "qandl"})["status"] == "completed"
    assert packet["packet_id"]


def test_automatic_memory_policy_distills_and_maintains_on_retrieval(amos):
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 1, "every_seconds": 0},
        distillation={"min_source_atoms": 3, "max_source_atoms": 3},
    )
    for index in range(3):
        amos.commit_atom(
            {
                "id": f"policy_source_{index}",
                "type": "belief",
                "payload": {"claim": f"automatic policy source memory {index}"},
                "scope": {"tenant": "policy"},
            }
        )

    packet = amos.retrieve_packet(
        cues=["automatic policy source"],
        scope={"tenant": "policy"},
        include_archived=True,
    )
    semantic_atoms = [
        atom
        for atom in amos.store.list_atoms()
        if atom["type"] == "semantic"
        and atom["payload"].get("distillation_type") == "automatic_policy"
    ]
    assert semantic_atoms
    distilled = semantic_atoms[0]
    assert distilled["payload"]["created_by"] == "svc:memory_policy"
    assert distilled["payload"]["source_refs"] == [
        "policy_source_0",
        "policy_source_1",
        "policy_source_2",
    ]
    assert distilled["layer"] == "consolidated_long_term"
    assert distilled["retention_class"] == "distilled"
    assert distilled["id"] in item_refs(packet)

    health = amos.health_memory()
    assert health["memory_policy"]["state"]["last_trigger"] == "retrieve_packet"
    assert health["memory_policy"]["due"]["due"] is False
    assert health["last_policy_tick"]["status"] == "skipped"
    assert "semantic_lexical_vectors" in health["index_freshness"]
    assert any(
        event["event_type"] == "memory_policy_run"
        for event in amos.store.list_events()
    )


def test_memory_policy_worker_force_runs_without_manual_maintenance(amos):
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 100, "every_seconds": 0},
        distillation={"min_source_atoms": 2, "max_source_atoms": 2},
    )
    for index in range(2):
        amos.commit_atom(
            {
                "id": f"worker_policy_source_{index}",
                "type": "episode",
                "payload": {"summary": f"worker policy episode {index}"},
            }
        )

    result = MemoryPolicyWorker(amos).tick(force=True, trigger="test_worker")
    assert result["status"] == "completed"
    assert result["trigger"] == "test_worker"
    assert result["results"]["distillation"]["status"] == "completed"
    assert amos.memory_policy_status()["state"]["last_trigger"] == "test_worker"


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

    repeat = DistillerMaintenanceWorker(amos).tick(
        scope=scope,
        domain="example_training",
        processor_ids=["example.training.flight.v1"],
    )
    assert repeat["committed"][0]["status"] == "already_committed"


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


def test_qandl_like_lesson_retrieval_for_training_agent(amos):
    amos.commit_atom(
        {
            "id": "qandl_codex_outage_fallback",
            "type": "procedure",
            "payload": {
                "name": "codex outage fallback",
                "trigger_context": "Codex directive timeout during Qandl training",
                "steps": [
                    "stop waiting after directive timeout",
                    "use pilot memory and local task advisors",
                    "record advisor decision and evidence",
                ],
            },
            "scope": {"tenant": "qandl", "component": "training"},
            "access_policy": {
                "visibility": ["trainer_agent", "advisor", "all"],
                "mutable_by": ["owner"],
            },
            "salience": 0.9,
            "utility": 0.95,
        }
    )
    packet = amos.retrieve_packet(
        cues=["Codex timeout local advisor fallback"],
        scope={"tenant": "qandl", "component": "training"},
        target_processor="trainer_agent",
    )
    assert "qandl_codex_outage_fallback" in item_refs(packet)
    assert packet["items"][0]["payload"]["steps"][1] == "use pilot memory and local task advisors"


def test_cli_smoke_init_commit_retrieve(tmp_path, capsys):
    db_path = tmp_path / "cli.sqlite3"
    assert cli_main(["--db", str(db_path), "init"]) == 0
    cli_main(
        [
            "--db",
            str(db_path),
            "commit-atom",
            "--type",
            "belief",
            "--payload",
            json.dumps({"claim": "CLI recall works"}),
        ]
    )
    assert cli_main(["--db", str(db_path), "retrieve", "--cue", "CLI recall"]) == 0
    out = capsys.readouterr().out
    assert "CLI recall works" in out


def test_http_v1_endpoints_smoke(tmp_path):
    db_path = str(tmp_path / "http.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        committed = http_json(
            f"{base}/v1/atoms:commit",
            {
                "atom": {
                    "id": "http_atom",
                    "type": "belief",
                    "payload": {"claim": "HTTP endpoint works"},
                }
            },
        )
        assert committed["status"] == "committed"
        assert server.amos.health_memory()["atoms"] == 1
        packet = http_json(
            f"{base}/v1/packets:retrieve",
            {"cues": ["HTTP endpoint"]},
        )
        assert "http_atom" in item_refs(packet)
        health = http_json(f"{base}/v1/health/memory")
        assert health["atoms"] == 1
        verify = http_json(f"{base}/v1/verify")
        assert verify["journal"]["status"] == "ok"
        assert verify["replay"]["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def http_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))
