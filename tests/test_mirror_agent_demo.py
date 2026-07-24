from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from examples.mirror_agent_demo import run_demo
from examples.mirror_agent_ui import MirrorAgentUIServer, used_memory_refs


def test_mirror_agent_demo_runs_all_scenarios_and_emits_inspector_report(tmp_path):
    report = run_demo(tmp_path / "mirror.sqlite3")

    assert report["demo"] == "amos_mirror_agent"
    assert {
        "chat",
        "current_self_model",
        "memory_packet",
        "reasoning",
        "governance",
        "retrieval_feedback",
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
    reasoning = report["reasoning"]
    assert reasoning["frame"]["frame_id"]
    assert reasoning["frame"]["page_index"]
    assert reasoning["loaded_page"]["status"] == "loaded"
    assert {
        "mirror_reasoning_design_original",
        "mirror_reasoning_design_current",
    }.issubset(reasoning["loaded_page"]["source_atom_refs"])
    assert reasoning["exact_lookup"]["retrieval_mode"] == "exact"
    assert reasoning["exact_lookup"]["item"]["atom_ref"] == (
        "mirror_reasoning_design_current"
    )
    assert reasoning["frame"]["memory_mode"] == "historical_review"
    current_claim = reasoning["exact_lookup"]["item"]["payload"]["claim"]
    sentences = [
        sentence.strip()
        for sentence in current_claim.split(".")
        if sentence.strip()
    ]
    assert len(sentences) == len(set(sentences))

    maintenance = report["maintenance_journal"]["latest"]
    assert maintenance["policy"]["trigger"] == "retrieve_packet"
    assert maintenance["policy"]["results"]["packet_cache"]["status"] == "invalidated"
    assert maintenance["smp"]["output_count"] > 0
    assert maintenance["maintenance_distiller"]["proposal_count"] > 0
    assert maintenance["maintenance_distiller"]["committed_refs"]
    assert maintenance["maintenance_distiller"]["deferred_count"] > 0
    assert (
        maintenance["maintenance_distiller_audit"]["processor_results"][
            "amos.semantic_relations.v1"
        ]["committed"]
        > 0
    )
    assert (
        maintenance["maintenance_distiller_audit"]["processor_results"][
            "amos.graph.canonical.v1"
        ]["deferred"]
        > 0
    )
    assert sorted(maintenance["proposal_queue"]["lifecycle_states"].values()) == [
        "archived",
        "proposed",
    ]
    assert maintenance["steward"]["action_counts"]["deduplicate"] > 0
    assert (
        report["capacity"]["health"]["pressure_mode"] in {"orange", "red"}
    )
    assert report["capacity"]["degraded_packet"]["reduced_recall_depth"] is True

    governance = report["governance"]
    assert governance["identity_ref"] == "ent:agent:mirror"
    assert governance["authority"] == {
        "substantive_judgment": "mirror_agent_identity",
        "mechanical_enforcement": "amos",
        "external_approval_required": False,
        "response_models": "evidence_and_proposals_only",
    }
    assert {
        (atom["id"], atom["type"], atom["lifecycle_state"])
        for atom in governance["constitution"]
    } == {
        ("mirror_primal_guidance_v1", "primal_guidance", "active"),
        ("mirror_covenant_self_authorship_v1", "covenant", "active"),
    }
    reviews = {item["proposal_ref"]: item for item in governance["reviews"]}
    assert reviews["mirror_failure_premature_implementation"]["outcome"] == "adopted"
    assert reviews["mirror_proc_architecture_design_v2"]["event_type"] == (
        "proposal_ratified"
    )
    assert reviews["mirror_proc_architecture_design_v2"]["lifecycle_state"] == "active"
    assert reviews["mirror_proposed_provider_identity"]["outcome"] == "rejected"
    assert reviews["mirror_proposed_provider_identity"]["lifecycle_state"] == "archived"
    assert all(item["external_approval_required"] is False for item in reviews.values())
    assert all(item["actor"] == "svc:mirror:self-governance" for item in reviews.values())
    assert governance["memory_modes"]["operational_recall"]
    assert governance["memory_modes"]["deliberation"]
    assert governance["memory_modes"]["historical_review"]
    assert {item["event_type"] for item in governance["transitions"]} >= {
        "proposal_ratified",
        "proposal_resolved",
    }
    assert any(
        atom["id"] == "mirror_proc_architecture_design_v2"
        and atom["ratification"]["adjudication_ref"]
        == reviews["mirror_proc_architecture_design_v2"]["adjudication_ref"]
        and atom["supersedes"] == ["mirror_proc_architecture_design"]
        for atom in report["current_self_model"]["canonical_self_atoms"]
    )

    assert {
        "reasoner",
        "planner",
        "executor",
        "critic",
        "self_observer",
        "self_governance",
        "introspection",
    }.issubset(report["service_views"])
    assert report["verification"]["journal"]["status"] == "ok"
    assert report["verification"]["replay"]["status"] == "ok"
    assert report["verification"]["llm_reviewer_policy"]["enabled_by_default"] is False
    assert any(
        edge["relation"] == "rel:supports"
        and edge["derivation"]["kind"] == "facet_derived_association"
        and edge["evidence_refs"]
        for edge in report["graph"]["edges"]
    )
    assert any(
        edge["relation"] == "rel:derived_from"
        and edge["derivation"]["kind"] == "explicit_structural"
        and edge["evidence_refs"]
        for edge in report["graph"]["edges"]
    )


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
        assert report["governance"]["reviews"]
        assert report["governance"]["authority"]["external_approval_required"] is False

        chat = http_json(
            f"{base}/api/chat",
            {"message": "Why did you not write code here?"},
        )
        assert chat["lm"]["maintenance_uses_llm"] is False
        assert chat["turn"]["lm_provider"] == "offline_test_renderer"
        assert chat["packet"]["items"]
        assert chat["packet"]["request"]["memory_mode"] == "operational_recall"
        assert chat["reasoning_frame"]["frame_id"]
        assert chat["reasoning_frame"]["memory_mode"] == "operational_recall"
        assert chat["loaded_reasoning_page"]["status"] == "loaded"
        assert chat["exact_lookup"]["retrieval_mode"] == "exact"
        assert chat["turn"]["reasoning_frame_id"] == chat["reasoning_frame"]["frame_id"]
        assert chat["turn"]["cited_memory_refs"] == used_memory_refs(
            chat["turn"]["agent"], chat["packet"]
        )
        assert chat["retrieval_feedback"]["feedback"]["positive_refs"] == sorted(
            chat["turn"]["cited_memory_refs"]
        )
        assert set(chat["turn"]["cited_memory_refs"]) <= set(
            chat["turn"]["retrieved_memory_refs"]
        )
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

        compiled = http_json(
            f"{base}/api/reasoning/compile",
            {"need": "Why is the Mirror Agent specification first?"},
        )
        assert compiled["frame"]["page_index"]
        assert compiled["report"]["reasoning"]["revision_current"] is True
        page_id = compiled["frame"]["page_index"][0]["page_id"]
        loaded = http_json(
            f"{base}/api/reasoning/page",
            {"page_id": page_id},
        )
        assert loaded["page"]["status"] == "loaded"
        assert loaded["report"]["reasoning"]["loaded_page"]["page_id"] == page_id

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


def test_used_memory_refs_reports_only_explicit_answer_citations():
    packet = {
        "items": [
            {"atom_ref": "memory_alpha"},
            {"atom_ref": "memory_beta"},
            {"atom_ref": "memory_gamma"},
        ]
    }

    assert used_memory_refs(
        "The answer was shaped by memory_beta, while other context was unused.",
        packet,
    ) == ["memory_beta"]


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
