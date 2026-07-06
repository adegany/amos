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

    def extract_facets(self, window):
        facets = []
        for atom in window.atoms:
            payload = atom.get("payload", {})
            if payload.get("example_kind") != "reflection":
                continue
            signature = payload.get("control_signature")
            if not signature:
                continue
            outcome = payload.get("outcome", "neutral")
            facets.append(
                SemanticFacet(
                    atom_ref=atom["id"],
                    subject=f"example training controls {signature}",
                    intent="evaluate sampled controls",
                    outcome=str(outcome),
                    outcome_direction=str(outcome),
                    confidence=float(atom.get("confidence", {}).get("score", 0.75)),
                    controls={"control_signature": signature},
                    metrics={"score": payload.get("score")},
                    time_index=payload.get("chunk"),
                    scope=dict(atom.get("scope", {})),
                    evidence_refs=tuple(atom.get("evidence_refs", [])),
                )
            )
        return facets


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


def test_batch_commit_uses_single_transaction_and_rejects_duplicate_batch(amos):
    committed = amos.commit_memory_atoms(
        [
            {
                "id": "batch_one",
                "type": "belief",
                "payload": {"claim": "batch one"},
            },
            {
                "id": "batch_two",
                "type": "belief",
                "payload": {"claim": "batch two"},
            },
        ]
    )

    assert [item["atom"]["id"] for item in committed["committed"]] == [
        "batch_one",
        "batch_two",
    ]
    assert amos.store.graph_version() == 2
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


def test_retrieve_packet_uses_graph_version_packet_cache(amos, monkeypatch):
    amos.commit_atom(
        {
            "id": "cache_hit_atom",
            "type": "belief",
            "payload": {"claim": "cache hit retrieval works"},
        }
    )
    first = amos.retrieve_packet(cues=["cache hit retrieval"], run_policy=False)
    assert "cache_hit_atom" in item_refs(first)

    def fail_list_atoms_filtered(**_kwargs):
        raise AssertionError("cache hit should not scan atoms")

    monkeypatch.setattr(amos.store, "list_atoms_filtered", fail_list_atoms_filtered)
    second = amos.retrieve_packet(cues=["cache hit retrieval"], run_policy=False)

    assert second["packet_id"] == first["packet_id"]
    assert "cache_hit_atom" in item_refs(second)


def test_retrieve_packet_attention_context_shapes_ranking_and_trace(amos):
    amos.commit_atom(
        {
            "id": "attention_mission_policy",
            "type": "policy",
            "payload": {
                "rule": "System policy for performance search mission routing",
            },
            "salience": 0.5,
            "utility": 0.5,
        }
    )
    amos.commit_atom(
        {
            "id": "attention_archive_policy",
            "type": "policy",
            "payload": {
                "rule": "System policy for archive cleanup and cold storage",
            },
            "salience": 0.5,
            "utility": 0.5,
        }
    )

    packet = amos.retrieve_packet(
        cues=["system policy"],
        max_items=2,
        attention_context={
            "active_task": "performance search",
            "mission": "mission routing",
            "focus_terms": ["performance", "routing"],
            "suppress_terms": ["archive"],
            "boost_memory_types": ["policy"],
        },
        run_policy=False,
    )

    assert packet["items"][0]["atom_ref"] == "attention_mission_policy"
    mission_components = packet["items"][0]["score_components"]
    archive_components = packet["items"][1]["score_components"]
    assert mission_components["attention_focus"] > 0
    assert mission_components["attention_type_boost"] == 1.0
    assert archive_components["attention_suppression_penalty"] > 0
    assert packet["attention_trace"]["policy_id"] == "amos.v1.attention.default"
    assert "performance" in packet["attention_trace"]["focus_terms"]
    assert packet["attention_trace"]["selected_item_refs"] == [
        item["atom_ref"] for item in packet["items"]
    ]
    assert packet["request"]["attention_context"]["boost_memory_types"] == ["policy"]


def test_attention_matching_ignores_payload_keys(amos):
    amos.commit_atom(
        {
            "id": "attention_payload_key_only",
            "type": "belief",
            "payload": {
                "claim": "Unrelated storage cleanup note",
            },
        }
    )
    amos.commit_atom(
        {
            "id": "attention_payload_value_match",
            "type": "belief",
            "payload": {
                "claim": "Production claim handling requires review",
            },
        }
    )

    packet = amos.retrieve_packet(
        cues=[],
        max_items=2,
        attention_context={"active_task": "claim production"},
        run_policy=False,
    )
    by_ref = {item["atom_ref"]: item for item in packet["items"]}

    assert by_ref["attention_payload_value_match"]["score_components"]["attention_focus"] > 0
    assert "attention_payload_key_only" not in by_ref
    assert packet["items"][0]["atom_ref"] == "attention_payload_value_match"


def test_empty_cue_retrieval_browses_by_attention_context(amos):
    amos.commit_atom(
        {
            "id": "attention_no_cue_generic",
            "type": "semantic",
            "payload": {"summary": "Generic unrelated operating note"},
        }
    )
    amos.commit_atom(
        {
            "id": "attention_no_cue_mission",
            "type": "semantic",
            "payload": {"summary": "Mission routing policy for performance search"},
        }
    )

    packet = amos.retrieve_packet(
        cues=[],
        max_items=2,
        attention_context={"mission": "performance search mission routing"},
        run_policy=False,
    )

    assert packet["items"][0]["atom_ref"] == "attention_no_cue_mission"
    assert packet["items"][0]["score_components"]["attention_focus"] > 0
    assert packet["attention_trace"]["selected_item_refs"] == [
        item["atom_ref"] for item in packet["items"]
    ]


def test_attention_novelty_preference_affects_score_components(amos):
    source_refs = []
    for idx in range(5):
        source_refs.append(
            amos.commit_atom(
                {
                    "id": f"novelty_source_{idx}",
                    "type": "semantic",
                    "payload": {"summary": f"Novelty source {idx}"},
                }
            )["atom"]["id"]
        )
    amos.commit_atom(
        {
            "id": "attention_familiar_atom",
            "type": "semantic",
            "payload": {
                "summary": "Novelty target familiar graph node",
                "source_refs": source_refs,
            },
        }
    )
    amos.commit_atom(
        {
            "id": "attention_novel_atom",
            "type": "semantic",
            "payload": {"summary": "Novelty target isolated graph node"},
        }
    )

    packet = amos.retrieve_packet(
        cues=["novelty target"],
        max_items=8,
        attention_context={"novelty_preference": 1.0},
        run_policy=False,
    )
    by_ref = {item["atom_ref"]: item for item in packet["items"]}

    assert by_ref["attention_novel_atom"]["score_components"]["attention_novelty"] == 1.0
    assert by_ref["attention_familiar_atom"]["score_components"]["attention_novelty"] == 0.0
    assert packet["request"]["attention_context"]["novelty_preference"] == 1.0


def test_retrieval_recency_decays_with_updated_at_age(amos):
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=45)).isoformat().replace("+00:00", "Z")
    recent = now.isoformat().replace("+00:00", "Z")
    amos.commit_atom(
        {
            "id": "recency_old_atom",
            "type": "semantic",
            "payload": {"summary": "Recency ranking target old"},
            "updated_at": old,
            "created_at": old,
            "observed_at": old,
            "salience": 0.5,
            "utility": 0.5,
        }
    )
    amos.commit_atom(
        {
            "id": "recency_recent_atom",
            "type": "semantic",
            "payload": {"summary": "Recency ranking target recent"},
            "updated_at": recent,
            "created_at": recent,
            "observed_at": recent,
            "salience": 0.5,
            "utility": 0.5,
        }
    )

    packet = amos.retrieve_packet(cues=["recency ranking target"], max_items=2, run_policy=False)
    by_ref = {item["atom_ref"]: item for item in packet["items"]}

    assert by_ref["recency_recent_atom"]["score_components"]["recency"] > 0.95
    assert by_ref["recency_old_atom"]["score_components"]["recency"] == 0.0
    assert by_ref["recency_recent_atom"]["score"] > by_ref["recency_old_atom"]["score"]


def test_goal_and_procedure_boosts_require_relevance(amos):
    amos.commit_atom(
        {
            "id": "irrelevant_goal_atom",
            "type": "goal",
            "payload": {"description": "Archive cleanup objective for another project"},
        }
    )
    amos.commit_atom(
        {
            "id": "relevant_goal_atom",
            "type": "goal",
            "payload": {"description": "Route search objective for current mission"},
        }
    )
    amos.commit_atom(
        {
            "id": "irrelevant_procedure_atom",
            "type": "procedure",
            "payload": {
                "trigger_context": "Archive cleanup",
                "steps": ["Collect cold storage candidates"],
            },
        }
    )

    packet = amos.retrieve_packet(
        cues=["route search"],
        max_items=3,
        include_low_health=True,
        run_policy=False,
    )
    by_ref = {item["atom_ref"]: item for item in packet["items"]}

    assert by_ref["relevant_goal_atom"]["score_components"]["goal_relevance"] > 0
    assert "irrelevant_goal_atom" not in by_ref
    assert "irrelevant_procedure_atom" not in by_ref


def test_edge_activation_spreads_from_cue_matched_atom(amos):
    source = amos.commit_atom(
        {
            "id": "edge_origin_atom",
            "type": "semantic",
            "payload": {"summary": "Seed phrase for graph activation"},
        }
    )["atom"]
    linked = amos.commit_atom(
        {
            "id": "edge_linked_atom",
            "type": "semantic",
            "payload": {
                "summary": "Associated downstream memory without query wording",
                "source_refs": [source["id"]],
            },
        }
    )["atom"]

    packet = amos.retrieve_packet(cues=["seed phrase"], max_items=4, run_policy=False)
    by_ref = {item["atom_ref"]: item for item in packet["items"]}

    assert source["id"] in by_ref
    assert linked["id"] in by_ref
    assert by_ref[linked["id"]]["score_components"]["direct_cue_match"] == 0.0
    assert by_ref[linked["id"]]["score_components"]["edge_activation"] > 0


def test_retrieve_packet_uses_sqlite_token_candidate_index(amos, monkeypatch):
    amos.commit_atom(
        {
            "id": "indexed_candidate_match",
            "type": "semantic",
            "payload": {"summary": "Indexed retrieval target phrase"},
        }
    )
    amos.commit_atom(
        {
            "id": "indexed_candidate_other",
            "type": "semantic",
            "payload": {"summary": "Unrelated background memory"},
        }
    )
    assert amos.store.atom_text_index_count() > 0

    calls = []
    original = amos.store.list_atoms_filtered

    def capture_list_atoms_filtered(**kwargs):
        calls.append(kwargs.get("atom_ids"))
        return original(**kwargs)

    monkeypatch.setattr(amos.store, "list_atoms_filtered", capture_list_atoms_filtered)
    packet = amos.retrieve_packet(cues=["target phrase"], max_items=4, run_policy=False)

    assert calls[-1] is not None
    assert "indexed_candidate_match" in calls[-1]
    assert "indexed_candidate_other" not in calls[-1]
    assert [item["atom_ref"] for item in packet["items"]] == ["indexed_candidate_match"]


def test_attention_context_is_part_of_packet_cache_key(amos):
    amos.commit_atom(
        {
            "id": "attention_cache_atom",
            "type": "belief",
            "payload": {"claim": "Cache attention context retrieval"},
        }
    )

    first = amos.retrieve_packet(
        cues=["attention context"],
        attention_context={"focus_terms": ["first"]},
        run_policy=False,
    )
    second = amos.retrieve_packet(
        cues=["attention context"],
        attention_context={"focus_terms": ["second"]},
        run_policy=False,
    )
    repeated = amos.retrieve_packet(
        cues=["attention context"],
        attention_context={"focus_terms": ["first"]},
        run_policy=False,
    )

    assert first["packet_id"] != second["packet_id"]
    assert repeated["packet_id"] == first["packet_id"]


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
    assert outcome["feedback"]["updated_atom_refs"] == [atom["id"]]
    updated = amos.store.get_atom(atom["id"])
    assert updated["utility"] > atom["utility"]
    assert updated["salience"] > atom["salience"]
    assert updated["last_accessed"]
    assert updated["decay_policy"]["retrieval_telemetry"]["used_count"] == 1
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
    archived = amos.archive_atom(distilled["atom"]["id"], reason="obsolete distillation")
    assert archived["projected_edges"]
    assert amos.store.list_edges() == []
    assert amos.verify_replay()["status"] == "ok"


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


def test_retrieval_outcome_accepts_stable_outcome_id(amos):
    first = amos.record_retrieval_outcome(
        packet_id="pkt_demo",
        request={"scope": {"tenant": "graph"}},
        outcome={"outcome_id": "rto_demo", "cited_atom_ref": "atom_a"},
    )
    second = amos.record_retrieval_outcome(
        packet_id="pkt_demo",
        request={"scope": {"tenant": "graph"}},
        outcome={"outcome_id": "rto_demo", "cited_atom_ref": "atom_a"},
    )

    assert first["status"] == "recorded"
    assert second["status"] == "already_recorded"
    assert amos.store.retrieval_outcome_count() == 1


def test_retrieval_outcome_corrections_demote_atom_utility(amos):
    atom = amos.commit_atom(
        {
            "id": "outcome_correction_atom",
            "type": "belief",
            "payload": {"claim": "Correction telemetry target"},
            "utility": 0.3,
        }
    )["atom"]
    packet = amos.retrieve_packet(cues=["correction telemetry"], run_policy=False)

    amos.record_retrieval_outcome(
        packet_id=packet["packet_id"],
        request=packet["request"],
        outcome={
            "correction_refs": [atom["id"]],
            "label": "corrected",
        },
    )

    updated = amos.store.get_atom(atom["id"])
    assert updated["utility"] < atom["utility"]
    assert updated["decay_policy"]["retrieval_telemetry"]["correction_count"] == 1


def test_steward_backfills_intrinsic_edges_for_existing_atoms(amos):
    semantic = amos.commit_atom(
        {
            "id": "early_semantic",
            "type": "semantic",
            "payload": {
                "summary": "semantic arrived before source",
                "source_refs": ["late_source"],
            },
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


def test_self_awareness_structurally_includes_role_facets_under_noisy_budget(amos):
    scope = {"tenant": "qandl", "component": "training", "run_id": "flight-1"}
    amos.commit_atom(
        {
            "id": "trainer_structural_self",
            "type": "self_model",
            "payload": {"agent_id": "trainer", "name": "Qandl trainer"},
            "scope": {"tenant": "qandl", "component": "training"},
        }
    )
    capability_refs = []
    for index in range(12):
        capability_refs.append(
            amos.commit_atom(
                {
                    "id": f"trainer_structural_cap_{index}",
                    "type": "capability",
                    "payload": {
                        "agent_id": "trainer",
                        "name": f"trainer_capability_{index}",
                        "description": "large capability detail " + ("x" * 1600),
                    },
                    "scope": {"tenant": "qandl", "component": "training"},
                }
            )["atom"]["id"]
        )
    limitation_refs = []
    for index in range(6):
        limitation_refs.append(
            amos.commit_atom(
                {
                    "id": f"trainer_structural_limit_{index}",
                    "type": "limitation",
                    "payload": {
                        "agent_id": "trainer",
                        "name": f"trainer_limitation_{index}",
                        "description": "large limitation detail " + ("y" * 1600),
                    },
                    "scope": {"tenant": "qandl", "component": "training"},
                }
            )["atom"]["id"]
        )
    commitment = amos.commit_atom(
        {
            "id": "trainer_structural_commitment",
            "type": "commitment",
            "payload": {
                "agent_id": "trainer",
                "description": "keep role self-awareness complete",
                "status": "open",
            },
            "scope": {"tenant": "qandl", "component": "training"},
        }
    )["atom"]
    runtime = amos.record_runtime_state(
        agent_id="trainer",
        capabilities={
            f"trainer_capability_{index}": {"available": True}
            for index in range(12)
        },
        constraints=["use structural self-awareness"],
        scope=scope,
    )["atom"]
    for index in range(40):
        amos.commit_atom(
            {
                "id": f"other_agent_noise_{index}",
                "type": "runtime_state",
                "payload": {
                    "agent_id": "other-agent",
                    "status": "available",
                    "description": "noise " + ("z" * 2400),
                },
                "scope": scope,
            }
        )

    view = amos.retrieve_self_awareness(agent_id="trainer", scope=scope)

    assert {item["atom_ref"] for item in view["capabilities"]} == set(capability_refs)
    assert {item["atom_ref"] for item in view["limitations"]} == set(limitation_refs)
    assert commitment["id"] in {
        item["atom_ref"] for item in view["open_commitments"]
    }
    assert view["runtime_state"]["atom_ref"] == runtime["id"]
    assert not any(
        omission["reason"] == "budget_exhausted"
        for omission in view["omissions"]
        if omission.get("atom_ref") in {*capability_refs, *limitation_refs, runtime["id"]}
    )
    assert view["source_packet_id"].startswith("pkt_")


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
    summary = distilled["payload"]["summary"]
    assert isinstance(summary, str)
    assert summary.startswith("Automatic AMOS memory policy distilled 3 source atoms")
    assert not summary.lstrip().startswith("{")
    assert "automatic policy source memory 0" in summary
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

    original_summary = distilled["payload"]["summary"]
    amos.archive_atom(distilled["id"], reason="replace obsolete policy summary")
    amos._policy_distillation_summary = (  # type: ignore[method-assign]
        lambda _atoms: f"{original_summary} Replacement renderer output."
    )
    rerun = amos.run_memory_policy(
        force=True,
        trigger="replace_archived_policy_summary",
        scope={"tenant": "policy"},
    )
    assert rerun["results"]["distillation"]["status"] == "completed"
    replacement = rerun["results"]["distillation"]["distilled"]["atom"]
    assert replacement["lifecycle_state"] == "active"
    assert replacement["payload"]["source_refs"] == distilled["payload"]["source_refs"]
    assert not replacement["payload"]["summary"].lstrip().startswith("{")
    assert "Replacement renderer output" in replacement["payload"]["summary"]


def test_memory_policy_executes_atom_decay_policy(amos):
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat().replace(
        "+00:00", "Z"
    )
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )
    stale_atom = amos.commit_atom(
        {
            "id": "decay_stale_atom",
            "type": "semantic",
            "payload": {"summary": "Decay stale target"},
            "updated_at": old,
            "observed_at": old,
            "created_at": old,
            "decay_policy": {"mark_stale_after_seconds": 1},
        }
    )["atom"]
    archive_atom = amos.commit_atom(
        {
            "id": "decay_archive_atom",
            "type": "semantic",
            "payload": {"summary": "Decay archive target"},
            "decay_policy": {"expires_at": expired},
        }
    )["atom"]
    ignored_atom = amos.commit_atom(
        {
            "id": "decay_ignored_atom",
            "type": "semantic",
            "payload": {"summary": "Decay ignored target"},
            "updated_at": old,
            "observed_at": old,
            "created_at": old,
        }
    )["atom"]

    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={"enabled": True, "require_atom_policy": True},
    )
    result = amos.run_memory_policy(force=True, trigger="decay_test")

    assert result["results"]["decay"]["action_count"] == 2
    assert amos.store.get_atom(stale_atom["id"])["health_status"] == "stale"
    assert amos.store.get_atom(archive_atom["id"])["lifecycle_state"] == "archived"
    assert amos.store.get_atom(ignored_atom["id"])["health_status"] == "healthy"
    assert any(
        event["event_type"] == "decay_policy_applied"
        for event in amos.store.list_events()
    )


def test_health_memory_can_skip_foreground_policy_tick(amos):
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 1, "every_seconds": 0},
        distillation={"min_source_atoms": 3, "max_source_atoms": 3},
    )
    for index in range(3):
        amos.commit_atom(
            {
                "id": f"health_policy_source_{index}",
                "type": "belief",
                "payload": {"claim": f"health policy source memory {index}"},
                "scope": {"tenant": "health-policy"},
            }
        )

    health = amos.health_memory(run_policy=False)

    assert health["last_policy_tick"]["status"] == "skipped"
    assert health["last_policy_tick"]["reason"] == "policy_not_run_for_health"
    assert not [
        atom
        for atom in amos.store.list_atoms()
        if atom["type"] == "semantic"
        and atom["payload"].get("distillation_type") == "automatic_policy"
    ]


def test_background_memory_policy_worker_runs_queued_tick(amos):
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 1, "every_seconds": 0},
        distillation={"min_source_atoms": 3, "max_source_atoms": 3},
    )
    for index in range(3):
        amos.commit_atom(
            {
                "id": f"background_policy_source_{index}",
                "type": "belief",
                "payload": {"claim": f"background policy source memory {index}"},
                "scope": {"tenant": "background-policy"},
            }
        )
    worker = BackgroundMemoryPolicyWorker(amos, interval_seconds=30)
    try:
        worker.start()
        queued = worker.request_tick(
            trigger="retrieve_packet",
            scope={"tenant": "background-policy"},
        )
        assert queued["status"] == "queued"
        deadline = time.time() + 5
        while time.time() < deadline:
            status = worker.status()
            if status["last_result"] and status["last_result"]["status"] == "completed":
                break
            time.sleep(0.02)
        else:
            pytest.fail(f"background policy worker did not complete: {worker.status()}")
    finally:
        worker.stop()

    semantic_atoms = [
        atom
        for atom in amos.store.list_atoms()
        if atom["type"] == "semantic"
        and atom["payload"].get("distillation_type") == "automatic_policy"
    ]
    assert semantic_atoms
    assert amos.memory_policy_status()["state"]["last_trigger"] == "retrieve_packet"


def test_automatic_memory_policy_prioritizes_outcome_evidence_over_directives(amos):
    scope = {"tenant": "policy-priority"}
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 100, "every_seconds": 0},
        distillation={"min_source_atoms": 3, "max_source_atoms": 3},
    )
    for index in range(3):
        amos.commit_atom(
            {
                "id": f"priority_directive_{index}",
                "type": "agentic_trace",
                "payload": {
                    "qandl_kind": "directive",
                    "task": f"chunk {index}",
                    "action": "issue directive",
                    "outcome": "issued",
                    "target_chunk": index,
                    "applied_controls": {"exploration_eps_floor": 0.05},
                },
                "scope": scope,
            }
        )
    for index in range(3):
        amos.commit_atom(
            {
                "id": f"priority_reflection_{index}",
                "type": "agentic_trace",
                "payload": {
                    "qandl_kind": "reflection",
                    "task": f"chunk {index}",
                    "action": "evaluate outcome",
                    "outcome": "supported",
                    "chunk": index,
                    "directive_atom_ref": f"priority_directive_{index}",
                    "delta_multiple": 0.1 + index,
                    "delta_sharpe": -0.05 + index,
                },
                "scope": scope,
            }
        )

    result = amos.run_memory_policy(
        force=True,
        trigger="test_priority",
        scope=scope,
    )

    assert result["results"]["distillation"]["status"] == "completed"
    source_refs = result["results"]["distillation"]["source_refs"]
    assert source_refs == [
        "priority_reflection_0",
        "priority_reflection_1",
        "priority_reflection_2",
    ]
    summary = result["results"]["distillation"]["distilled"]["atom"]["payload"][
        "summary"
    ]
    assert "delta_multiple=+0.1" in summary
    assert "delta_sharpe=-0.05" in summary


def test_memory_policy_worker_force_runs_without_manual_maintenance(amos):
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 100, "every_seconds": 0},
        maintenance={"max_smp_atoms": 2},
        distillation={"min_source_atoms": 2, "max_source_atoms": 2},
    )
    for index in range(5):
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
    assert result["results"]["smp"]["atom_count"] == 5
    assert result["results"]["smp"]["analyzed_atom_count"] == 2
    assert result["results"]["smp"]["omitted_atom_count"] == 3
    assert result["results"]["distillation"]["status"] == "completed"
    assert amos.memory_policy_status()["state"]["last_trigger"] == "test_worker"


def test_memory_policy_journal_summarizes_large_smp_results(amos):
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 100, "every_seconds": 0},
        maintenance={"max_smp_atoms": 12},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
    )
    for index in range(12):
        amos.commit_atom(
            {
                "id": f"journal_policy_source_{index}",
                "type": "belief",
                "payload": {"claim": f"journal policy claim {index % 3}"},
                "scope": {"tenant": "journal-policy"},
            }
        )

    result = amos.run_memory_policy(
        force=True,
        trigger="test_journal_summary",
        scope={"tenant": "journal-policy"},
    )

    assert result["results"]["smp"]["outputs"]
    event = result["event"]
    payload = event["payload"]
    smp = payload["results"]["smp"]
    assert "outputs" not in smp
    assert smp["output_count"] == len(result["results"]["smp"]["outputs"])
    assert smp["analyzed_atom_count"] == 12
    assert smp["sample_output_ids"]
    assert payload["results"]["steward"]["event_id"]
    assert len(json.dumps(payload)) < 20000
    assert amos.verify_replay()["status"] == "ok"


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


def test_maintenance_distiller_skips_empty_windows_without_journal_event(amos):
    event_count = len(amos.store.list_events())

    result = amos.run_maintenance_distiller(scope={"tenant": "empty"})

    assert result["status"] == "skipped"
    assert result["reason"] == "no_proposals"
    assert result["event"] is None
    assert len(amos.store.list_events()) == event_count


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


def test_http_sqlite_lock_returns_retryable_json(tmp_path):
    db_path = str(tmp_path / "http_locked.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def locked_commit(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    server.amos.commit_atom = locked_commit
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            http_json(
                f"{base}/v1/atoms:commit",
                {
                    "atom": {
                        "id": "http_locked_atom",
                        "type": "belief",
                        "payload": {"claim": "lock handling works"},
                    }
                },
            )
        assert excinfo.value.code == 503
        payload = json.loads(excinfo.value.read().decode("utf-8"))
        assert payload["status"] == "error"
        assert payload["retryable"] is True
        assert "database is locked" in payload["error"]
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
