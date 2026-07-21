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


def test_suppress_terms_inhibit_but_do_not_expand_candidate_generation(amos):
    amos.commit_atom(
        {
            "id": "candidate_focus_atom",
            "type": "semantic",
            "payload": {"summary": "rare mission focus target"},
        }
    )
    amos.commit_atom(
        {
            "id": "candidate_suppressed_only_atom",
            "type": "semantic",
            "payload": {"summary": "obsolete archive quarantine material"},
        }
    )

    packet = amos.retrieve_packet(
        cues=["rare mission"],
        attention_context={"suppress_terms": ["archive quarantine"]},
        run_policy=False,
    )

    assert item_refs(packet) == {"candidate_focus_atom"}
    assert packet["degradation"]["candidate_generation"]["lexical_count"] == 1


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


def test_retrieve_packet_scopes_edge_reads_to_candidate_refs(amos, monkeypatch):
    for index in range(3):
        old = amos.commit_atom(
            {
                "id": f"unrelated_edge_old_{index}",
                "type": "semantic",
                "payload": {"summary": f"Unrelated edge old memory {index}"},
            }
        )["atom"]
        amos.commit_atom(
            {
                "id": f"unrelated_edge_new_{index}",
                "type": "semantic",
                "payload": {"summary": f"Unrelated edge new memory {index}"},
                "supersedes": [old["id"]],
            }
        )
    target = amos.commit_atom(
        {
            "id": "scoped_edge_candidate",
            "type": "semantic",
            "payload": {"summary": "Scoped edge retrieval target phrase"},
        }
    )["atom"]
    linked = amos.commit_atom(
        {
            "id": "scoped_edge_neighbor",
            "type": "semantic",
            "payload": {
                "summary": "Neighbor reached through candidate-scoped edge activation",
                "source_refs": [target["id"]],
            },
        }
    )["atom"]

    original_scoped_edges = amos.store.list_edges_for_refs
    scoped_calls = []

    def capture_list_edges_for_refs(refs):
        scoped_calls.append(list(refs))
        return original_scoped_edges(refs)

    def fail_list_edges():
        raise AssertionError("retrieve_packet should not scan all edges")

    monkeypatch.setattr(amos.store, "list_edges_for_refs", capture_list_edges_for_refs)
    monkeypatch.setattr(amos.store, "list_edges", fail_list_edges)

    packet = amos.retrieve_packet(
        cues=["target phrase"],
        max_items=4,
        include_conflicts=True,
        run_policy=False,
    )

    assert target["id"] in item_refs(packet)
    assert linked["id"] in item_refs(packet)
    assert scoped_calls
    allowed_refs = {target["id"], linked["id"]}
    assert all(set(call) <= allowed_refs for call in scoped_calls)


def test_retrieve_packet_filters_archived_hot_activation_except_provenance(amos):
    seed = amos.commit_atom(
        {
            "id": "hot_activation_seed",
            "type": "semantic",
            "payload": {"summary": "zzseedalpha active retrieval seed"},
        }
    )["atom"]
    archived_uses = amos.commit_atom(
        {
            "id": "archived_uses_neighbor",
            "type": "semantic",
            "payload": {"summary": "qqarchivedomega ordinary archived neighbor"},
            "lifecycle_state": "archived",
            "health_status": "stale",
        }
    )["atom"]
    archived_source = amos.commit_atom(
        {
            "id": "archived_provenance_neighbor",
            "type": "semantic",
            "payload": {"summary": "rrprovenanceomega archived source neighbor"},
            "lifecycle_state": "archived",
            "health_status": "stale",
        }
    )["atom"]
    with amos.store.transaction() as conn:
        amos.store.insert_edge(
            conn,
            amos.graph._edge(seed["id"], archived_uses["id"], "rel:uses", {}),
        )
        amos.store.insert_edge(
            conn,
            amos.graph._edge(seed["id"], archived_source["id"], "rel:derived_from", {}),
        )

    packet = amos.retrieve_packet(
        cues=["zzseedalpha"],
        max_items=5,
        include_archived=True,
        include_low_health=True,
        run_policy=False,
    )

    refs = item_refs(packet)
    assert seed["id"] in refs
    assert archived_source["id"] in refs
    assert archived_uses["id"] not in refs
    provenance = next(
        item for item in packet["items"] if item["atom_ref"] == archived_source["id"]
    )
    assert provenance["score_components"]["edge_activation"] > 0
    assert any(
        omission["atom_ref"] == archived_uses["id"]
        and omission["reason"] == "low_relevance"
        for omission in packet["omissions"]
    )


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


def test_retrieval_falls_back_to_semantic_similarity_for_morphology(amos):
    amos.commit_atom(
        {
            "id": "morphology_retrieval_atom",
            "type": "semantic",
            "payload": {"summary": "Packet retrieval troubleshooting procedure"},
        }
    )
    amos.commit_atom(
        {
            "id": "morphology_unrelated_atom",
            "type": "semantic",
            "payload": {"summary": "Cooking inventory background note"},
        }
    )

    packet = amos.retrieve_packet(
        cues=["retrieving"],
        max_items=2,
        run_policy=False,
    )

    assert packet["items"]
    assert packet["items"][0]["atom_ref"] == "morphology_retrieval_atom"
    assert packet["items"][0]["score_components"]["semantic_similarity"] > 0


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


def test_retrieval_feedback_only_updates_members_of_the_exact_packet(amos):
    selected = amos.commit_atom(
        {
            "id": "packet_member",
            "type": "belief",
            "payload": {"claim": "exact packet feedback target"},
        }
    )["atom"]
    outside = amos.commit_atom(
        {
            "id": "outside_packet",
            "type": "belief",
            "payload": {"claim": "unrelated exterior record"},
        }
    )["atom"]
    packet = amos.retrieve_packet(cues=["exact packet feedback"], run_policy=False)

    result = amos.record_retrieval_outcome(
        packet_id=packet["packet_id"],
        request=packet["request"],
        outcome={
            "used_item_refs": [selected["id"], outside["id"], "evd_not_an_atom"],
            "evidence_refs": ["evd_reported_separately"],
            "label": "useful",
        },
    )

    assert result["feedback"]["updated_atom_refs"] == [selected["id"]]
    assert result["feedback"]["ignored_non_packet_refs"] == [
        "evd_not_an_atom",
        outside["id"],
    ]
    assert result["feedback"]["reported_evidence_refs"] == [
        "evd_reported_separately"
    ]
    assert amos.store.get_atom(outside["id"])["utility"] == outside["utility"]


def test_two_hop_association_trace_is_bounded_and_trains_used_edges(amos):
    atoms = [
        amos.commit_atom(
            {
                "id": atom_ref,
                "type": "semantic",
                "payload": {"summary": summary},
            }
        )["atom"]
        for atom_ref, summary in (
            ("association_seed", "zzseedalpha direct retrieval anchor"),
            ("association_mid", "intermediate graph bridge"),
            ("association_leaf", "distant associated conclusion"),
        )
    ]
    with amos.store.transaction() as conn:
        first = amos.graph._edge(
            atoms[0]["id"], atoms[1]["id"], "rel:derived_from", {}
        )
        second = amos.graph._edge(
            atoms[1]["id"], atoms[2]["id"], "rel:derived_from", {}
        )
        amos.store.insert_edge(conn, first)
        amos.store.insert_edge(conn, second)

    packet = amos.retrieve_packet(
        cues=["zzseedalpha"], max_items=5, run_policy=False
    )
    leaf = next(item for item in packet["items"] if item["atom_ref"] == "association_leaf")
    assert leaf["score_components"]["edge_activation"] > 0
    assert [step["depth"] for step in leaf["association_trace"]] == [1, 2]

    outcome = amos.record_retrieval_outcome(
        packet_id=packet["packet_id"],
        request=packet["request"],
        outcome={"used_item_refs": ["association_leaf"], "label": "useful"},
    )
    assert outcome["feedback"]["updated_edge_refs"] == sorted(
        [first["edge_id"], second["edge_id"]]
    )


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
