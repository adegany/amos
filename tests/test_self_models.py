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


def test_agentic_recall_skips_foreground_memory_policy(amos):
    amos.configure_memory_policy(
        schedule={"every_graph_versions": 1, "every_seconds": 0},
        distillation={"min_source_atoms": 3, "max_source_atoms": 3},
    )
    for index in range(3):
        amos.record_agentic_trace(
            agent_id="trainer",
            task=f"latency sensitive recall {index}",
            action="read agent memory",
            outcome="success",
            scope={"tenant": "qandl"},
        )

    recall = amos.retrieve_agentic_recall(
        agent_id="trainer",
        cues=["latency sensitive recall"],
        scope={"tenant": "qandl"},
    )

    assert recall["self_actions"]
    assert amos.memory_policy_status()["due"]["due"] is True
    assert not [
        event
        for event in amos.store.list_events()
        if event["event_type"] == "memory_policy_run"
    ]


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
