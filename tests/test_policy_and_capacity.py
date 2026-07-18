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
        "semantic_lsa_vectors",
        "semantic_lexical_vectors",
    }
    health = amos.health_memory()
    assert health["projection_lag"] == 0
    assert "semantic_lexical_vectors" in health["index_freshness"]
    assert "semantic_lsa_vectors" in health["index_freshness"]

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


def test_index_rebuild_persists_lsa_vectors_and_refreshes_atom_vectors(amos):
    for atom_id, summary in [
        ("lsa_retrieval_packet", "retrieval packet memory recall"),
        ("lsa_retrieve_context", "retrieve packet context recall"),
        ("lsa_optimizer_budget", "optimizer budget schedule"),
        ("lsa_training_schedule", "training optimizer schedule"),
    ]:
        amos.commit_atom(
            {
                "id": atom_id,
                "type": "semantic",
                "payload": {"summary": summary},
            }
        )
    amos.configure_memory_policy(
        maintenance={"lsa_dimensions": 4, "lsa_max_terms": 32},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={"enabled": False},
        storage_cleanup={"enabled": False},
    )

    result = IndexMaintainer(amos).rebuild()
    by_name = {item["index_name"]: item for item in result["indexes"]}

    assert by_name["semantic_lsa_vectors"]["details_json"]["status"] == "rebuilt"
    assert by_name["semantic_lsa_vectors"]["details_json"]["dimensions"] > 0
    assert amos.store.list_token_latent_vectors(graph_version=result["graph_version"])

    atom = amos.store.get_atom("lsa_retrieval_packet")
    model = amos.indexes._atom_search_index(atom)["vector_model"]
    assert model["idf_graph_version"] == result["graph_version"]
    assert model["latent_graph_version"] == result["graph_version"]
    assert model["latent_dimensions"] > 0


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
    amos.policy._policy_distillation_summary = (  # type: ignore[method-assign]
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


def test_automatic_memory_policy_selects_one_coherent_source_group(amos):
    scope = {"tenant": "coherent-policy"}
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 100, "every_seconds": 0},
        distillation={"min_source_atoms": 2, "max_source_atoms": 3},
        maintenance_distiller={"enabled": False},
    )
    for index in range(2):
        amos.commit_atom(
            {
                "id": f"coherent_project_{index}",
                "type": "agentic_trace",
                "payload": {
                    "task": "coherent project",
                    "action": "evaluate project step",
                    "outcome": "supported",
                    "lesson": f"project finding {index}",
                    "maintenance_hints": {
                        "profile": "example.project.v1",
                        "kind": "project_outcome",
                        "consolidation_key": "project-one",
                        "priority": 6,
                    },
                },
                "scope": scope,
            }
        )
    amos.commit_atom(
        {
            "id": "unrelated_relationship_belief",
            "type": "belief",
            "payload": {
                "claim": "An unrelated relationship observation.",
                "maintenance_hints": {
                    "profile": "example.relationship.v1",
                    "consolidation_key": "relationship-one",
                },
            },
            "scope": scope,
        }
    )

    result = amos.run_memory_policy(
        force=True,
        trigger="coherent_source_test",
        scope=scope,
    )["results"]["distillation"]

    assert result["status"] == "completed"
    assert result["source_refs"] == ["coherent_project_0", "coherent_project_1"]
    assert "unrelated_relationship_belief" not in result["source_refs"]
    assert result["coherent_candidate_count"] == 2


def test_automatic_memory_policy_respects_domain_lane_and_derived_coverage(amos):
    scope = {"tenant": "domain-lane-policy"}
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 100, "every_seconds": 0},
        distillation={"min_source_atoms": 2, "max_source_atoms": 4},
        maintenance_distiller={"enabled": False},
    )
    for index in range(2):
        amos.commit_atom(
            {
                "id": f"domain_owned_{index}",
                "type": "agentic_trace",
                "payload": {
                    "task": "domain-owned project",
                    "action": "domain-owned project step",
                    "outcome": "active",
                    "maintenance_hints": {
                        "profile": "example.domain.v1",
                        "consolidation_key": "domain-project",
                        "distillation_lane": "domain_processor",
                    },
                },
                "scope": scope,
            }
        )
    for index in range(2):
        amos.commit_atom(
            {
                "id": f"covered_source_{index}",
                "type": "belief",
                "payload": {"claim": f"covered source {index}"},
                "scope": scope,
            }
        )
    amos.commit_atom(
        {
            "id": "existing_domain_consolidation",
            "type": "semantic",
            "payload": {
                "summary": "The covered sources already have an active derived memory.",
                "created_by_processor": "example.domain.processor.v1",
                "distillation_type": "example_consolidation",
                "source_refs": ["covered_source_0", "covered_source_1"],
            },
            "scope": scope,
        }
    )

    result = amos.run_memory_policy(
        force=True,
        trigger="domain_lane_test",
        scope=scope,
    )["results"]["distillation"]

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_candidates"
    assert result["candidate_count"] == 0


def test_memory_policy_skips_when_another_tick_holds_execution_lock(amos):
    assert amos.policy._memory_policy_lock.acquire(blocking=False)
    try:
        result = amos.run_memory_policy(force=True, trigger="concurrent_tick")
    finally:
        amos.policy._memory_policy_lock.release()

    assert result["status"] == "skipped"
    assert result["reason"] == "memory_policy_already_running"


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
            "payload": {
                "summary": "Decay ignored target",
                "source_refs": [archive_atom["id"]],
            },
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
    assert result["results"]["decay"]["projected_edges"]
    assert amos.store.list_edges() == []
    assert any(
        event["event_type"] == "decay_policy_applied"
        for event in amos.store.list_events()
    )


def test_memory_policy_pressure_archives_policyless_atoms_to_limit(amos):
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat().replace(
        "+00:00", "Z"
    )
    protected = amos.commit_atom(
        {
            "id": "pressure_protected_policy",
            "type": "policy",
            "payload": {"rule": "Preserve governance memory under pressure."},
            "utility": 0.1,
        }
    )["atom"]
    opted_out = amos.commit_atom(
        {
            "id": "pressure_opted_out_trace",
            "type": "agentic_trace",
            "payload": {"task": "pressure", "action": "keep", "outcome": "opted out"},
            "utility": 0.0,
            "decay_policy": {"enabled": False},
        }
    )["atom"]
    low = amos.commit_atom(
        {
            "id": "pressure_low_trace",
            "type": "agentic_trace",
            "payload": {"task": "pressure", "action": "rank", "outcome": "low"},
            "created_at": old,
            "observed_at": old,
            "updated_at": old,
            "utility": 0.1,
            "decay_policy": {
                "retrieval_telemetry": {"used_count": 3, "correction_count": 1}
            },
        }
    )["atom"]
    middle = amos.commit_atom(
        {
            "id": "pressure_middle_trace",
            "type": "agentic_trace",
            "payload": {"task": "pressure", "action": "rank", "outcome": "middle"},
            "utility": 0.2,
        }
    )["atom"]
    high = amos.commit_atom(
        {
            "id": "pressure_high_trace",
            "type": "agentic_trace",
            "payload": {"task": "pressure", "action": "rank", "outcome": "high"},
            "utility": 0.9,
        }
    )["atom"]
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={
            "enabled": True,
            "max_atoms": 3,
            "require_atom_policy": True,
            "pressure_archive_policyless": True,
            "pressure_max_archives_per_run": 10,
        },
        storage_cleanup={"enabled": False},
    )

    result = amos.run_memory_policy(force=True, trigger="pressure_decay_test")
    pressure = result["results"]["decay"]["pressure"]

    assert pressure == {
        "enabled": True,
        "triggered": True,
        "max_atoms": 3,
        "max_active_atoms": 3,
        "max_proposed_atoms": 3,
        "hot_count_before": 5,
        "hot_count_after_rules": 5,
        "active_count_after_rules": 5,
        "proposed_count_after_rules": 0,
        "active_pressure_needed": 2,
        "proposed_pressure_needed": 0,
        "eligible_policyless_count": 3,
        "eligible_proposed_count": 0,
        "archive_limit": 10,
        "archive_count": 2,
        "proposal_archive_count": 0,
        "active_archive_count": 2,
        "remaining_hot_count": 3,
        "remaining_over_limit": 0,
        "remaining_active_over_limit": 0,
        "remaining_proposed_over_limit": 0,
    }
    pressure_actions = [
        action
        for action in result["results"]["decay"]["actions"]
        if action["reason"] == "active_atom_pressure_policyless_fallback"
    ]
    assert [action["atom_ref"] for action in pressure_actions] == [
        low["id"],
        middle["id"],
    ]
    assert amos.store.get_atom(low["id"])["lifecycle_state"] == "archived"
    assert amos.store.get_atom(middle["id"])["lifecycle_state"] == "archived"
    assert amos.store.get_atom(high["id"])["lifecycle_state"] == "active"
    assert amos.store.get_atom(opted_out["id"])["lifecycle_state"] == "active"
    assert amos.store.get_atom(protected["id"])["lifecycle_state"] == "active"
    health = amos.health_memory(run_policy=False)
    assert health["quality"]["active_atom_count"] == 3
    assert health["quality"]["active_atom_pressure"] == "within_limit"
    assert health["quality"]["pressure_cleanup"]["eligible_policyless_count"] == 1


def test_memory_policy_enforces_proposed_quota_separately_from_active_atoms(amos):
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat().replace(
        "+00:00", "Z"
    )
    for index in range(2):
        amos.commit_atom(
            {
                "id": f"separate_quota_active_{index}",
                "type": "semantic",
                "payload": {"summary": f"Canonical memory {index}"},
            }
        )
    for index in range(3):
        amos.propose_memory_atoms(
            [
                {
                    "id": f"separate_quota_proposed_{index}",
                    "type": "episode",
                    "payload": {
                        "summary": f"Review candidate {index}",
                        "task": "quota test",
                        "outcome": "pending",
                        "started_at": old,
                        "participants": ["test"],
                        "proposal_retention": {
                            "profile": "test.v1",
                            "deduplication_key": f"candidate-{index}",
                        },
                    },
                    "created_at": old,
                    "observed_at": old,
                    "updated_at": old,
                }
            ]
        )
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={
            "enabled": True,
            "max_atoms": 10,
            "max_active_atoms": 2,
            "max_proposed_atoms": 2,
            "proposal_pressure_min_age_seconds": 0,
            "pressure_archive_proposed": True,
        },
        storage_cleanup={"enabled": False},
    )

    result = amos.run_memory_policy(force=True, trigger="separate_quota_test")
    pressure = result["results"]["decay"]["pressure"]

    assert pressure["triggered"] is True
    assert pressure["active_pressure_needed"] == 0
    assert pressure["proposed_pressure_needed"] == 1
    assert pressure["archive_count"] == 1
    assert pressure["proposal_archive_count"] == 1
    assert pressure["remaining_proposed_over_limit"] == 0
    assert result["results"]["decay"]["actions"][0]["reason"] == (
        "proposed_atom_pressure_fallback"
    )
    health = amos.health_memory(run_policy=False)["quality"]
    assert health["lifecycle_active_atom_count"] == 2
    assert health["lifecycle_active_atom_limit"] == 2
    assert health["proposed_atom_count"] == 2
    assert health["proposed_atom_limit"] == 2
    assert health["hot_atom_count"] == 4
    assert health["hot_atom_limit"] == 10


def test_memory_policy_deduplicates_only_explicitly_keyed_proposals(amos):
    base = {
        "type": "semantic",
        "payload": {
            "summary": "Repeated bounded reflection",
            "proposal_retention": {
                "profile": "test.v1",
                "deduplication_key": "same-bounded-meaning",
            },
        },
    }
    first = amos.propose_memory_atoms(
        [{**base, "id": "explicit_duplicate_first", "evidence_refs": ["evt_a"]}]
    )["proposals"][0]["atom"]
    second = amos.propose_memory_atoms(
        [
            {
                **base,
                "id": "explicit_duplicate_second",
                "evidence_refs": ["evt_a", "evt_b"],
            }
        ]
    )["proposals"][0]["atom"]
    unkeyed = amos.propose_memory_atoms(
        [
            {
                "id": "similar_but_unkeyed",
                "type": "semantic",
                "payload": {"summary": "Repeated bounded reflection"},
            }
        ]
    )["proposals"][0]["atom"]
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={"enabled": True, "max_atoms": 10},
        storage_cleanup={"enabled": False},
    )

    result = amos.run_memory_policy(force=True, trigger="proposal_dedupe_test")

    actions = result["results"]["decay"]["actions"]
    assert actions == [
        {
            "atom_ref": first["id"],
            "action": "archive",
            "reason": "explicit_proposal_deduplication",
            "superseded_by": [second["id"]],
            "health_status": "merged",
            "lifecycle_state": "archived",
        }
    ]
    assert amos.store.get_atom(second["id"])["lifecycle_state"] == "proposed"
    assert amos.store.get_atom(unkeyed["id"])["lifecycle_state"] == "proposed"


def test_memory_policy_archives_proposal_after_explicit_retention_window(amos):
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat().replace(
        "+00:00", "Z"
    )
    proposal = amos.propose_memory_atoms(
        [
            {
                "id": "expired_proposal_retention",
                "type": "semantic",
                "payload": {
                    "summary": "Temporary review candidate",
                    "proposal_retention": {
                        "profile": "test.v1",
                        "deduplication_key": "temporary-review-candidate",
                        "archive_after_seconds": 60,
                    },
                },
                "created_at": old,
                "observed_at": old,
                "updated_at": old,
            }
        ]
    )["proposals"][0]["atom"]
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={"enabled": True, "max_atoms": 10},
        storage_cleanup={"enabled": False},
    )

    result = amos.run_memory_policy(force=True, trigger="proposal_retention_test")

    assert result["results"]["decay"]["actions"] == [
        {
            "atom_ref": proposal["id"],
            "action": "archive",
            "reason": "proposed_retention_elapsed",
            "health_status": "stale",
            "lifecycle_state": "archived",
        }
    ]


def test_memory_policy_pressure_reports_residual_protected_atoms(amos):
    for index in range(2):
        amos.commit_atom(
            {
                "id": f"pressure_protected_policy_{index}",
                "type": "policy",
                "payload": {"rule": f"Protected policy {index}"},
            }
        )
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={
            "enabled": True,
            "max_atoms": 1,
            "pressure_archive_policyless": True,
        },
        storage_cleanup={"enabled": False},
    )

    result = amos.run_memory_policy(force=True, trigger="protected_pressure_test")
    pressure = result["results"]["decay"]["pressure"]

    assert pressure["archive_count"] == 0
    assert pressure["remaining_hot_count"] == 2
    assert pressure["remaining_over_limit"] == 1
    health = amos.health_memory(run_policy=False)
    assert "active_atom_pressure_not_fully_enforceable" in health["quality"]["warnings"]
    assert health["quality"]["pressure_cleanup"]["eligible_policyless_count"] == 0
    assert health["quality"]["pressure_cleanup"]["archives_needed"] == 1


def test_memory_health_recommends_capacity_target_with_headroom(amos):
    for index in range(3):
        amos.commit_atom(
            {
                "id": f"capacity_atom_{index}",
                "type": "semantic",
                "payload": {"summary": f"Capacity observation {index}"},
            }
        )
    amos.configure_memory_policy(
        decay={
            "max_atoms": 3,
            "capacity_assessment_targets": [3, 6, 9],
            "capacity_headroom_ratio": 0.2,
        }
    )

    health = amos.health_memory(run_policy=False)
    capacity = health["quality"]["capacity_assessment"]

    assert capacity["configured_target"] == 3
    assert capacity["active_count"] == 3
    assert capacity["utilization"] == 1.0
    assert capacity["near_limit"] is True
    assert capacity["recommended_target"] == 6
    assert capacity["candidate_targets"] == [
        {
            "target": 3,
            "headroom_atoms": 0,
            "utilization": 1.0,
            "meets_headroom_target": False,
        },
        {
            "target": 6,
            "headroom_atoms": 3,
            "utilization": 0.5,
            "meets_headroom_target": True,
        },
        {
            "target": 9,
            "headroom_atoms": 6,
            "utilization": 0.3333,
            "meets_headroom_target": True,
        },
    ]
    assert "active_atom_capacity_headroom_low" in health["quality"]["warnings"]


def test_memory_policy_archives_superseded_atoms_and_retrieval_omits_them(amos):
    old = amos.commit_atom(
        {
            "id": "superseded_runtime_observation",
            "type": "semantic",
            "payload": {"summary": "terrain candidate alpha old snapshot"},
        }
    )["atom"]
    new = amos.commit_atom(
        {
            "id": "current_runtime_observation",
            "type": "semantic",
            "payload": {"summary": "terrain candidate alpha current snapshot"},
            "supersedes": [old["id"]],
        }
    )["atom"]

    before = amos.retrieve_packet(
        cues=["terrain candidate alpha old"],
        include_low_health=True,
        run_policy=False,
    )
    assert old["id"] not in [item["atom_id"] for item in before["items"]]
    assert any(
        omission["atom_ref"] == old["id"] and omission["reason"] == "superseded"
        for omission in before["omissions"]
    )

    included = amos.retrieve_packet(
        cues=["terrain candidate alpha old"],
        include_low_health=True,
        include_superseded=True,
        run_policy=False,
    )
    assert old["id"] in [item["atom_id"] for item in included["items"]]
    old_item = next(item for item in included["items"] if item["atom_id"] == old["id"])
    assert old_item["score_components"]["superseded_penalty"] == 1.0

    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={
            "enabled": True,
            "require_atom_policy": True,
            "archive_superseded": True,
            "archive_superseded_after_seconds": 0,
        },
        storage_cleanup={"enabled": False},
    )
    result = amos.run_memory_policy(force=True, trigger="superseded_decay_test")

    assert result["results"]["decay"]["action_count"] == 1
    assert result["results"]["decay"]["actions"][0]["reason"] == "superseded_by_active_atom"
    assert result["results"]["decay"]["actions"][0]["superseded_by"] == [new["id"]]
    archived = amos.store.get_atom(old["id"])
    assert archived["lifecycle_state"] == "archived"
    assert archived["health_status"] == "stale"
    assert amos.store.get_atom(new["id"])["lifecycle_state"] == "active"


def test_service_owned_decay_archives_scoped_superseded_atoms_with_empty_scope(amos):
    scope = {"tenant": "qandl", "component": "training", "run_id": "run-1"}
    old = amos.commit_atom(
        {
            "id": "scoped_superseded_runtime_observation",
            "type": "runtime_state",
            "payload": {"agent_id": "qandl.training.pilot", "summary": "old scoped runtime state"},
            "scope": scope,
        }
    )["atom"]
    new = amos.commit_atom(
        {
            "id": "scoped_current_runtime_observation",
            "type": "runtime_state",
            "payload": {
                "agent_id": "qandl.training.pilot",
                "summary": "current scoped runtime state",
            },
            "scope": scope,
            "supersedes": [old["id"]],
        }
    )["atom"]
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={
            "enabled": True,
            "require_atom_policy": True,
            "archive_superseded": True,
            "archive_superseded_after_seconds": 0,
        },
        storage_cleanup={"enabled": False},
    )

    result = amos.run_memory_policy(
        force=True,
        trigger="background_interval",
        scope={},
    )

    assert result["results"]["decay"]["action_count"] == 1
    assert result["results"]["decay"]["actions"][0]["atom_ref"] == old["id"]
    assert amos.store.get_atom(old["id"])["lifecycle_state"] == "archived"
    assert amos.store.get_atom(new["id"])["lifecycle_state"] == "active"


def test_health_memory_reports_quality_diagnostics(amos):
    amos.configure_memory_policy(
        decay={"max_atoms": 1},
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        storage_cleanup={"enabled": False},
    )
    old = amos.commit_atom(
        {
            "id": "quality_superseded",
            "type": "semantic",
            "payload": {"summary": "quality superseded"},
        }
    )["atom"]
    amos.commit_atom(
        {
            "id": "quality_current",
            "type": "semantic",
            "payload": {"summary": "quality current"},
            "supersedes": [old["id"]],
        }
    )
    isolated = amos.commit_atom(
        {
            "id": "quality_isolated",
            "type": "semantic",
            "payload": {"summary": "quality isolated"},
        }
    )["atom"]

    health = amos.health_memory(run_policy=False)

    assert health["quality"]["status"] == "warning"
    assert "active_atom_count_exceeds_decay_max_atoms" in health["quality"]["warnings"]
    assert "active_superseded_atoms_present" in health["quality"]["warnings"]
    assert "isolated_active_atoms_present" in health["quality"]["warnings"]
    assert health["quality"]["active_superseded_atoms"]["count"] == 1
    assert health["quality"]["active_superseded_atoms"]["sample_refs"] == [old["id"]]
    assert health["quality"]["isolated_active_atoms"]["count"] >= 1
    assert isolated["id"] in health["quality"]["isolated_active_atoms"]["sample_refs"]
    graph_quality = health["quality"]["graph_quality"]
    assert graph_quality["active_atom_type_distribution"]["semantic"] == 3
    assert graph_quality["active_relation_distribution"]["rel:supersedes"] == 1
    assert graph_quality["component_count"] == 2
    assert graph_quality["largest_component_size"] == 2
    assert graph_quality["edge_derivation_distribution"]["intrinsic_structural"] == 1
    assert "hub_concentration_top_five" in graph_quality
    assert "edge_confidence_histogram" in graph_quality
    assert "proposal_quality" in health["quality"]
    assert "maintenance_processor_effectiveness" in health["quality"]


def test_health_isolation_separates_active_graph_from_dormant_proposals(amos):
    active = amos.commit_atom(
        {
            "id": "quality_active_isolated",
            "type": "semantic",
            "payload": {"summary": "Active graph-quality subject"},
        }
    )["atom"]
    proposed = amos.propose_memory_atoms(
        [
            {
                "id": "quality_proposed_dormant",
                "type": "semantic",
                "payload": {"summary": "Dormant proposal"},
            }
        ]
    )["proposals"][0]["atom"]

    health = amos.health_memory(run_policy=False)
    quality = health["quality"]

    assert quality["lifecycle_counts"] == {
        "active": 1,
        "proposed": 1,
        "hot_total": 2,
    }
    assert quality["active_atom_count"] == 2
    assert quality["isolated_active_atoms"]["count"] == 1
    assert quality["isolated_active_atoms"]["sample_refs"] == [active["id"]]
    assert quality["isolated_proposed_atoms"]["count"] == 1
    assert quality["isolated_proposed_atoms"]["expected_dormant"] is True
    assert quality["isolated_proposed_atoms"]["sample_refs"] == [proposed["id"]]


def test_memory_policy_storage_cleanup_deletes_expired_archived_and_stale_atoms(amos):
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace(
        "+00:00", "Z"
    )
    archived = amos.commit_atom(
        {
            "id": "cleanup_archived_atom",
            "type": "semantic",
            "payload": {"summary": "Cleanup archived target"},
            "created_at": old,
            "observed_at": old,
            "updated_at": old,
            "lifecycle_state": "archived",
            "health_status": "stale",
        }
    )["atom"]
    stale = amos.commit_atom(
        {
            "id": "cleanup_stale_atom",
            "type": "semantic",
            "payload": {"summary": "Cleanup stale target"},
            "created_at": old,
            "observed_at": old,
            "updated_at": old,
            "health_status": "stale",
        }
    )["atom"]
    protected = amos.commit_atom(
        {
            "id": "cleanup_protected_policy",
            "type": "policy",
            "payload": {"rule": "Cleanup should preserve protected policy atoms"},
            "created_at": old,
            "observed_at": old,
            "updated_at": old,
            "lifecycle_state": "archived",
            "health_status": "stale",
        }
    )["atom"]
    assert archived["id"] in amos.store.candidate_atom_ids_for_tokens(["archived"])
    assert stale["id"] in amos.store.candidate_atom_ids_for_tokens(["stale"])

    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={"enabled": False},
        storage_cleanup={
            "enabled": True,
            "idle_after_seconds": 0,
            "min_interval_seconds": 0,
            "delete_archived_after_seconds": 0,
            "delete_stale_after_seconds": 0,
            "protected_types": ["policy"],
            "sqlite_compaction": {
                "checkpoint_wal": False,
                "vacuum_enabled": False,
            },
        },
    )
    result = amos.run_memory_policy(force=True, trigger="storage_cleanup_test")

    cleanup = result["results"]["storage_cleanup"]
    assert cleanup["deleted_atom_count"] == 2
    assert set(cleanup["deleted_atom_refs"]) == {archived["id"], stale["id"]}
    assert amos.store.get_atom(archived["id"])["deleted"] == 1
    assert amos.store.get_atom(stale["id"])["deleted"] == 1
    assert amos.store.get_atom(protected["id"])["deleted"] == 0
    assert archived["id"] not in amos.store.candidate_atom_ids_for_tokens(["archived"])
    assert stale["id"] not in amos.store.candidate_atom_ids_for_tokens(["stale"])
    assert any(
        event["event_type"] == "storage_cleanup_run"
        for event in amos.store.list_events()
    )
    assert amos.verify_replay()["status"] == "ok"


def test_memory_policy_rebuild_keeps_archived_stale_atoms_out_of_hot_index(amos):
    archived = amos.commit_atom(
        {
            "id": "cleanup_rebuild_archived",
            "type": "semantic",
            "payload": {"summary": "Cleanup rebuild archived target"},
            "lifecycle_state": "archived",
            "health_status": "stale",
        }
    )["atom"]
    stale = amos.commit_atom(
        {
            "id": "cleanup_rebuild_stale",
            "type": "semantic",
            "payload": {"summary": "Cleanup rebuild stale target"},
            "health_status": "stale",
        }
    )["atom"]
    assert archived["id"] in amos.store.candidate_atom_ids_for_tokens(["archived"])
    assert stale["id"] in amos.store.candidate_atom_ids_for_tokens(["stale"])

    amos.configure_memory_policy(
        schedule={"every_graph_versions": 1, "every_seconds": 0},
        maintenance={
            "enabled": True,
            "run_smp": False,
            "run_steward": False,
            "rebuild_indexes": True,
            "invalidate_packet_cache": False,
        },
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={"enabled": False},
        storage_cleanup={
            "enabled": True,
            "idle_after_seconds": 0,
            "min_interval_seconds": 0,
            "max_deletions_per_tick": 0,
            "compact_idempotency_after_seconds": None,
            "sqlite_compaction": {
                "checkpoint_wal": False,
                "vacuum_enabled": False,
            },
        },
    )

    result = amos.run_memory_policy(force=True, trigger="storage_cleanup_rebuild_test")

    assert result["results"]["storage_cleanup"]["deleted_atom_count"] == 0
    hot_prune = result["results"]["index"]["indexes"][0]["details_json"][
        "hot_index_prune"
    ]
    assert hot_prune["rows"] >= 2
    assert archived["id"] not in amos.store.candidate_atom_ids_for_tokens(["archived"])
    assert stale["id"] not in amos.store.candidate_atom_ids_for_tokens(["stale"])


def test_memory_policy_storage_cleanup_compacts_idempotency_and_sqlite(amos, monkeypatch):
    amos.commit_atom(
        {
            "id": "cleanup_idempotency_atom",
            "type": "semantic",
            "payload": {"summary": "Cleanup idempotency target", "blob": "x" * 2048},
        },
        idempotency_key="cleanup-idempotency-key",
    )
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace(
        "+00:00", "Z"
    )
    with amos.store.transaction() as conn:
        conn.execute("UPDATE amos_idempotency SET created_at = ?", (old,))

    calls = {"checkpoint": 0, "vacuum": 0}

    def fake_checkpoint(*, mode="TRUNCATE"):
        calls["checkpoint"] += 1
        return {"status": "completed", "mode": mode, "busy": 0}

    def fake_vacuum():
        calls["vacuum"] += 1
        return {
            "status": "completed",
            "page_count_before": 10,
            "page_count_after": 8,
            "freelist_count_before": 2,
            "freelist_count_after": 0,
        }

    monkeypatch.setattr(amos.store, "checkpoint_wal", fake_checkpoint)
    monkeypatch.setattr(amos.store, "vacuum", fake_vacuum)
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={"enabled": False},
        storage_cleanup={
            "enabled": True,
            "idle_after_seconds": 0,
            "min_interval_seconds": 0,
            "max_deletions_per_tick": 0,
            "compact_idempotency_after_seconds": 0,
            "max_idempotency_compactions_per_tick": 8,
            "sqlite_compaction": {
                "checkpoint_wal": True,
                "checkpoint_mode": "TRUNCATE",
                "vacuum_enabled": True,
                "vacuum_idle_after_seconds": 0,
                "vacuum_min_interval_seconds": 0,
            },
        },
    )

    result = amos.run_memory_policy(force=True, trigger="storage_cleanup_sqlite_test")

    cleanup = result["results"]["storage_cleanup"]
    assert cleanup["idempotency"]["rows"] == 1
    assert cleanup["idempotency"]["saved_bytes"] > 0
    assert cleanup["checkpoint"]["status"] == "completed"
    assert cleanup["vacuum"]["status"] == "completed"
    assert cleanup["checkpoint_after_vacuum"]["status"] == "completed"
    assert calls == {"checkpoint": 2, "vacuum": 1}
    row = amos.store.conn.execute(
        "SELECT response_json FROM amos_idempotency WHERE idempotency_key = ?",
        ("cleanup-idempotency-key",),
    ).fetchone()
    assert json.loads(row["response_json"])["storage_compacted"] is True


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
