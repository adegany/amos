from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from examples.mirror_agent_demo import run_demo
from examples.mirror_agent_ui import MirrorAgentUIServer


def test_mirror_agent_demo_runs_all_scenarios_and_emits_inspector_report(tmp_path):
    report = run_demo(tmp_path / "mirror.sqlite3")

    assert report["demo"] == "amos_mirror_agent"
    assert {
        "chat",
        "current_self_model",
        "memory_packet",
        "evidence",
        "maintenance_journal",
        "capacity",
        "graph",
        "service_views",
        "scenario_results",
        "verification",
    }.issubset(report)
    assert all(
        result["status"] == "passed"
        for result in report["scenario_results"].values()
    )

    self_types = {
        atom["type"] for atom in report["current_self_model"]["canonical_self_atoms"]
    }
    assert {"capability", "commitment", "goal", "limitation", "procedure", "self_model"}.issubset(
        self_types
    )
    assert report["memory_packet"]["packet_id"]
    assert "mirror_belief_capacity_governor" in {
        item["atom_ref"] for item in report["memory_packet"]["items"]
    }
    assert report["evidence"]["captured"]

    maintenance = report["maintenance_journal"]["latest"]
    assert maintenance["policy"]["trigger"] == "retrieve_packet"
    assert maintenance["policy"]["results"]["packet_cache"]["status"] == "invalidated"
    assert maintenance["smp"]["output_count"] > 0
    assert maintenance["maintenance_distiller"]["proposal_count"] > 0
    assert maintenance["maintenance_distiller"]["committed_refs"]
    assert maintenance["steward"]["action_counts"]["deduplicate"] > 0
    assert (
        report["capacity"]["health"]["pressure_mode"] in {"orange", "red"}
    )
    assert report["capacity"]["degraded_packet"]["reduced_recall_depth"] is True

    assert {
        "reasoner",
        "planner",
        "executor",
        "critic",
        "self_observer",
        "introspection",
    }.issubset(report["service_views"])
    assert report["verification"]["journal"]["status"] == "ok"
    assert report["verification"]["replay"]["status"] == "ok"
    assert report["verification"]["llm_reviewer_policy"]["enabled_by_default"] is False


def test_mirror_agent_ui_serves_report_chat_and_non_llm_maintenance(tmp_path):
    try:
        server = MirrorAgentUIServer(
            ("127.0.0.1", 0),
            tmp_path / "mirror-ui.sqlite3",
            lm_mode="offline",
        )
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        report = http_json(f"{base}/api/report")
        assert report["demo"] == "amos_mirror_agent"
        assert report["lm"]["provider"] == "offline_test_renderer"

        chat = http_json(
            f"{base}/api/chat",
            {"message": "Why did you not write code here?"},
        )
        assert chat["lm"]["maintenance_uses_llm"] is False
        assert chat["turn"]["lm_provider"] == "offline_test_renderer"
        assert chat["packet"]["items"]
        assert chat["report"]["memory_packet"]["packet_id"] == chat["turn"]["memory_packet_id"]
        assert chat["report"]["memory_packet_source"] == "interactive_chat"
        assert chat["report"]["evidence"]["captured"][0]["source_ref"].startswith("ui/chat/")
        repeated_chat = http_json(
            f"{base}/api/chat",
            {"message": "Why did you not write code here?"},
        )
        assert repeated_chat["turn"]["lm_provider"] == "offline_test_renderer"
        assert repeated_chat["turn"]["memory_packet_id"]
        assert (
            repeated_chat["report"]["memory_packet"]["packet_id"]
            == repeated_chat["turn"]["memory_packet_id"]
        )
        assert repeated_chat["report"]["memory_packets"][0]["source"] == "interactive_chat"

        maintenance = http_json(f"{base}/api/maintenance/run", {}, timeout=60)
        assert maintenance["lm_used"] is False
        assert maintenance["smp"]["status"] == "completed"
        assert "maintenance_distiller" in maintenance
        assert maintenance["maintenance_distiller"]["reviewer"]["authority"] == "draft_only"
        assert maintenance["report"]["maintenance_journal"]["latest"]["lm_used"] is False
        assert maintenance["report"]["maintenance_journal"]["latest"]["maintenance_distiller"]["proposals"]
        assert maintenance["report"]["graph"]["graph_version"] == maintenance["report"]["verification"]["memory"]["graph_version"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def http_json(url, payload=None, *, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
