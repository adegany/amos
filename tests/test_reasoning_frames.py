from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from amos import StaleFrameError, ValidationError


def _insert_relation(amos, source_ref: str, target_ref: str, relation: str, scope=None):
    edge = amos.graph._edge(source_ref, target_ref, relation, scope or {})
    with amos.store.transaction() as conn:
        assert amos.store.insert_edge(conn, edge) is True
    return edge


def _canonical_size(value) -> int:
    return len(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )


def test_frame_keeps_supersession_chain_as_one_coherent_unit(amos):
    old = amos.commit_atom(
        {
            "id": "frame_old_decision",
            "type": "belief",
            "payload": {"claim": "The memory architecture used fixed isolated slots."},
        }
    )["atom"]
    new = amos.commit_atom(
        {
            "id": "frame_new_decision",
            "type": "belief",
            "payload": {"claim": "The memory architecture uses coherent demand pages."},
            "supersedes": [old["id"]],
        }
    )["atom"]

    legacy = amos.retrieve_packet(
        cues=["coherent demand pages"], max_items=1, run_policy=False
    )
    assert old["id"] not in {item["atom_ref"] for item in legacy["items"]}

    frame = amos.compile_memory_frame(
        need="coherent demand pages",
        purpose="continue the memory architecture",
        run_policy=False,
    )
    unit = next(unit for unit in frame["units"] if new["id"] in unit["source_atom_refs"])
    assert unit["unit_type"] == "decision_chain"
    assert unit["source_atom_refs"] == [old["id"], new["id"]]
    assert unit["active_conclusion_refs"] == [new["id"]]
    assert [step["atom_ref"] for step in unit["sequence"]] == [old["id"], new["id"]]
    assert unit["compression"]["mode"] == "none"
    assert unit["unit_id"] not in {page["unit_ref"] for page in frame["page_index"]}


def test_frame_keeps_both_sides_of_a_conflict(amos):
    left = amos.commit_atom(
        {
            "id": "conflict_left",
            "type": "belief",
            "payload": {"claim": "Deploy the memory compiler immediately."},
        }
    )["atom"]
    right = amos.commit_atom(
        {
            "id": "conflict_right",
            "type": "belief",
            "payload": {"claim": "Do not deploy the memory compiler immediately."},
        }
    )["atom"]
    edge = _insert_relation(amos, left["id"], right["id"], "rel:contradicts")

    frame = amos.compile_memory_frame(
        need="deploy memory compiler immediately",
        purpose="resolve the deployment disagreement",
        run_policy=False,
    )
    unit = next(unit for unit in frame["units"] if edge["edge_id"] in unit["conflict_refs"])
    assert unit["unit_type"] == "conflict_set"
    assert set(unit["source_atom_refs"]) == {left["id"], right["id"]}
    assert {item["atom_ref"] for item in unit["items"]} == {
        left["id"],
        right["id"],
    }


def test_commitment_amendments_remain_temporally_ordered(amos):
    original = amos.commit_atom(
        {
            "id": "commitment_original",
            "type": "commitment",
            "payload": {
                "description": "Deliver the coherent memory change Friday.",
                "started_at": "2026-07-01T10:00:00Z",
            },
        }
    )["atom"]
    amended = amos.commit_atom(
        {
            "id": "commitment_amended",
            "type": "commitment",
            "payload": {
                "description": "Deliver the coherent memory change Monday.",
                "started_at": "2026-07-02T10:00:00Z",
            },
            "supersedes": [original["id"]],
        }
    )["atom"]

    frame = amos.compile_memory_frame(
        need="deliver coherent memory change",
        purpose="check the active commitment",
        run_policy=False,
    )
    unit = next(
        unit for unit in frame["units"] if amended["id"] in unit["source_atom_refs"]
    )
    assert [step["atom_ref"] for step in unit["sequence"]] == [
        original["id"],
        amended["id"],
    ]
    assert unit["active_conclusion_refs"] == [amended["id"]]
    assert set(unit["commitment_refs"]) == {original["id"], amended["id"]}


def test_commitment_history_does_not_absorb_the_actor_identity_graph(amos):
    actor = amos.commit_atom(
        {
            "id": "commitment_actor_hub",
            "type": "self_model",
            "payload": {
                "subject_agent": "ent:agent:cogito",
                "statement": "The actor identity hub owns many unrelated memories.",
            },
        }
    )["atom"]
    unrelated = amos.commit_atom(
        {
            "id": "commitment_unrelated_identity_context",
            "type": "belief",
            "payload": {"claim": "Unrelated identity context must remain independent."},
        }
    )["atom"]
    original = amos.commit_atom(
        {
            "id": "hub_commitment_original",
            "type": "commitment",
            "payload": {"description": "Roll out the paging change directly."},
        }
    )["atom"]
    amended = amos.commit_atom(
        {
            "id": "hub_commitment_amended",
            "type": "commitment",
            "payload": {
                "description": "Run isolated checks before the paging rollout."
            },
            "supersedes": [original["id"]],
        }
    )["atom"]
    _insert_relation(amos, actor["id"], unrelated["id"], "rel:part_of")
    _insert_relation(amos, actor["id"], amended["id"], "rel:made_commitment")

    frame = amos.compile_memory_frame(
        need="isolated checks before the paging rollout",
        purpose="inspect the current rollout commitment",
        run_policy=False,
    )
    unit = next(
        unit for unit in frame["units"] if amended["id"] in unit["source_atom_refs"]
    )

    assert set(unit["source_atom_refs"]) == {original["id"], amended["id"]}
    assert actor["id"] not in unit["source_atom_refs"]
    assert unrelated["id"] not in unit["source_atom_refs"]
    assert unit["active_conclusion_refs"] == [amended["id"]]


def test_supporting_rationale_is_advertised_without_forcing_initial_residency(amos):
    decision = amos.commit_atom(
        {
            "id": "paged_rationale_decision",
            "type": "belief",
            "payload": {"claim": "Use runtime-owned page authorization."},
        }
    )["atom"]
    rationale = amos.commit_atom(
        {
            "id": "paged_rationale_evidence",
            "type": "belief",
            "payload": {
                "claim": "Generated JSON cannot delegate protected authority."
            },
        }
    )["atom"]
    support_edge = _insert_relation(
        amos,
        rationale["id"],
        decision["id"],
        "rel:supports",
    )

    frame = amos.compile_memory_frame(
        need="runtime-owned page authorization",
        purpose="apply the current paging decision",
        token_or_byte_budget={"tokens": 4000},
        run_policy=False,
    )
    decision_unit = next(
        unit for unit in frame["units"] if decision["id"] in unit["source_atom_refs"]
    )
    rationale_unit = next(
        unit for unit in frame["units"] if rationale["id"] in unit["source_atom_refs"]
    )
    duplicate_projection_descriptors = [
        page
        for page in frame["page_index"]
        if set(page["source_atom_refs"]) == {decision["id"], rationale["id"]}
        and page["relationship_refs"] == [support_edge["edge_id"]]
    ]

    assert decision_unit["inclusion_reasons"][decision["id"]] == ["semantic_seed"]
    assert rationale_unit["inclusion_reasons"][rationale["id"]] == ["semantic_seed"]
    assert rationale["id"] not in decision_unit["source_atom_refs"]
    assert len(duplicate_projection_descriptors) == 1
    descriptor = duplicate_projection_descriptors[0]
    assert descriptor["unit_ref"] == decision_unit["unit_id"]
    assert decision["id"] in descriptor["focus_atom_refs"]
    assert descriptor["title"] == "Supporting context via rel:supports"
    assert descriptor["summary"] == (
        "Adds 1 atom (belief) and 1 relationship (rel:supports) beyond the "
        "resident unit."
    )
    assert descriptor["summary"] != decision_unit["summary"]
    assert descriptor["relevance"] == (
        "Adds typed supporting context that is not resident in the frame."
    )
    descriptor_copy = " ".join(
        str(descriptor[field]) for field in ("title", "summary", "relevance")
    )
    assert "Generated JSON cannot delegate protected authority" not in descriptor_copy
    page = amos.load_memory_page(
        frame_id=frame["frame_id"],
        revision=frame["revision"],
        page=descriptor,
        depth="supporting",
        token_or_byte_budget={"tokens": 1600},
        run_policy=False,
    )
    assert {decision["id"], rationale["id"]} <= set(page["source_atom_refs"])
    assert any(
        relationship["relation"] == "rel:supports"
        for paged_unit in page["units"]
        for relationship in paged_unit["relationships"]
    )

    bounded_frame = amos.compile_memory_frame(
        need="runtime-owned page authorization",
        purpose="apply the current paging decision",
        token_or_byte_budget={"tokens": 1600},
        run_policy=False,
    )
    bounded_rationale = next(
        unit
        for unit in bounded_frame["units"]
        if rationale["id"] in unit["source_atom_refs"]
    )
    assert bounded_rationale["compression"]["mode"] != "none"
    bounded_descriptor = next(
        page
        for page in bounded_frame["page_index"]
        if page["unit_ref"] == bounded_rationale["unit_id"]
    )
    assert bounded_descriptor["focus_atom_refs"] == [rationale["id"]]
    assert bounded_descriptor["source_atom_refs"] == [rationale["id"]]
    assert bounded_descriptor["relationship_refs"] == []


def test_active_conclusion_and_constraint_roles_follow_relation_direction(amos):
    historical_episode = amos.commit_atom(
        {
            "id": "role_historical_episode",
            "type": "episode",
            "payload": {"summary": "A stale page was silently combined."},
            "observed_at": "2026-07-01T10:00:00Z",
        }
    )["atom"]
    decision = amos.commit_atom(
        {
            "id": "role_corrective_decision",
            "type": "belief",
            "payload": {"claim": "Reject stale pages explicitly."},
            "observed_at": "2026-07-02T10:00:00Z",
        }
    )["atom"]
    governing_constraint = amos.commit_atom(
        {
            "id": "role_governing_constraint",
            "type": "belief",
            "payload": {"claim": "Never mix incompatible revisions."},
            "observed_at": "2026-07-03T10:00:00Z",
        }
    )["atom"]
    _insert_relation(
        amos,
        decision["id"],
        historical_episode["id"],
        "rel:caused_by",
    )
    _insert_relation(
        amos,
        decision["id"],
        governing_constraint["id"],
        "rel:constrained_by",
    )

    frame = amos.compile_memory_frame(
        need="reject stale pages incompatible revisions",
        purpose="apply the corrective decision",
        run_policy=False,
    )
    unit = next(
        unit
        for unit in frame["units"]
        if {historical_episode["id"], decision["id"], governing_constraint["id"]}
        <= set(unit["source_atom_refs"])
    )
    roles = {
        step["atom_ref"]: step["role"]
        for step in unit["sequence"]
    }

    assert unit["active_conclusion_refs"] == [decision["id"]]
    assert unit["constraint_refs"] == [governing_constraint["id"]]
    assert roles[decision["id"]] == "active_conclusion"
    assert roles[governing_constraint["id"]] == "constraint"
    assert roles[historical_episode["id"]] == "history"


def test_page_descriptor_is_bound_to_frame_revision_and_budget(amos):
    old = amos.commit_atom(
        {
            "id": "paged_old",
            "type": "belief",
            "payload": {"claim": "Paged memory history " + "old detail " * 180},
        }
    )["atom"]
    new = amos.commit_atom(
        {
            "id": "paged_new",
            "type": "belief",
            "payload": {"claim": "Paged memory history active conclusion " + "new detail " * 180},
            "supersedes": [old["id"]],
        }
    )["atom"]
    frame = amos.compile_memory_frame(
        need="paged memory history active conclusion",
        purpose="inspect the complete decision history",
        token_or_byte_budget={"tokens": 800},
        run_policy=False,
    )
    descriptor = next(page for page in frame["page_index"] if new["id"] in page["focus_atom_refs"])
    assert all(
        descriptor["unit_ref"] != unit["unit_id"] for unit in frame["units"]
    )
    assert frame["token_estimate"] == (frame["budget"]["used_bytes"] + 3) // 4
    assert frame["budget"]["used_bytes"] == _canonical_size(frame)
    assert frame["budget"]["used_bytes"] <= frame["budget"]["limit_bytes"]

    page = amos.load_memory_page(
        frame_id=frame["frame_id"],
        revision=frame["revision"],
        page=descriptor,
        need="complete decision history",
        purpose="resolve the active conclusion",
        depth="supporting",
        token_or_byte_budget={"tokens": 900},
        run_policy=False,
    )
    assert page["status"] == "loaded"
    assert set(page["source_atom_refs"]) == {old["id"], new["id"]}
    assert page["active_conclusion_refs"] == [new["id"]]
    assert page["budget"]["used_bytes"] == _canonical_size(page)
    assert page["token_estimate"] == (page["budget"]["used_bytes"] + 3) // 4
    assert page["budget"]["used_bytes"] <= page["budget"]["limit_bytes"]

    tampered = dict(descriptor)
    tampered["title"] = "untrusted replacement"
    with pytest.raises(ValidationError, match="digest mismatch"):
        amos.load_memory_page(
            frame_id=frame["frame_id"],
            revision=frame["revision"],
            page=tampered,
            run_policy=False,
        )

    amos.commit_atom(
        {
            "id": "revision_change",
            "type": "belief",
            "payload": {"claim": "Canonical memory changed after frame compilation."},
        }
    )
    with pytest.raises(StaleFrameError) as excinfo:
        amos.load_memory_page(
            frame_id=frame["frame_id"],
            revision=frame["revision"],
            page=descriptor,
            run_policy=False,
        )
    assert excinfo.value.expected_revision == frame["revision"]
    assert excinfo.value.current_revision != frame["revision"]


def test_frame_never_crosses_trusted_scope_during_required_closure(amos):
    visible = amos.commit_atom(
        {
            "id": "scope_visible",
            "type": "belief",
            "payload": {"claim": "Scoped coherent memory for project alpha."},
            "scope": {"project": "alpha", "human": "ada"},
        }
    )["atom"]
    other_human = amos.commit_atom(
        {
            "id": "scope_other_human",
            "type": "belief",
            "payload": {"claim": "Scoped coherent memory for project alpha and another human."},
            "scope": {"project": "alpha", "human": "grace"},
        }
    )["atom"]
    hidden = amos.commit_atom(
        {
            "id": "scope_hidden",
            "type": "belief",
            "payload": {"claim": "Scoped coherent memory for project beta."},
            "scope": {"project": "beta"},
        }
    )["atom"]
    _insert_relation(
        amos,
        visible["id"],
        hidden["id"],
        "rel:derived_from",
        scope={"project": "alpha", "human": "ada"},
    )
    _insert_relation(
        amos,
        visible["id"],
        other_human["id"],
        "rel:requires",
        scope={"project": "alpha", "human": "ada"},
    )

    frame = amos.compile_memory_frame(
        need="scoped coherent memory",
        purpose="continue project alpha",
        scope={"project": "alpha", "human": "ada"},
        run_policy=False,
    )
    assert visible["id"] in frame["source_atom_refs"]
    assert hidden["id"] not in frame["source_atom_refs"]
    assert other_human["id"] not in frame["source_atom_refs"]
    assert all(hidden["id"] not in page["source_atom_refs"] for page in frame["page_index"])
    assert all(
        other_human["id"] not in page["source_atom_refs"]
        for page in frame["page_index"]
    )
    assert (
        frame["compilation_trace"]["filtered_counts"]["semantic_scope_hidden"]
        >= 2
    )


def test_task_context_semantic_scope_filters_same_text_payload_tags(amos):
    common_claim = "Semantic scope continuation memory for the active work."
    matching = amos.commit_atom(
        {
            "id": "semantic_scope_matching",
            "type": "belief",
            "payload": {
                "claim": common_claim,
                "human_id": "human-1",
                "project_id": "project-1",
                "project_thread_id": "thread-1",
            },
        }
    )["atom"]
    broad = amos.commit_atom(
        {
            "id": "semantic_scope_global",
            "type": "belief",
            "payload": {"claim": common_claim},
        }
    )["atom"]
    tagged_global = amos.commit_atom(
        {
            "id": "semantic_scope_tagged_global",
            "type": "belief",
            "payload": {
                "claim": common_claim,
                "human_id": "global",
                "project_id": "global",
                "conversation_id": "global",
            },
        }
    )["atom"]
    mismatches = []
    for atom_id, field, value in (
        ("semantic_scope_wrong_human", "human_id", "human-2"),
        ("semantic_scope_wrong_project", "project_id", "project-2"),
        ("semantic_scope_wrong_thread", "conversation_id", "thread-2"),
    ):
        mismatches.append(
            amos.commit_atom(
                {
                    "id": atom_id,
                    "type": "belief",
                    "payload": {"claim": common_claim, field: value},
                }
            )["atom"]
        )

    frame = amos.compile_memory_frame(
        need="semantic scope continuation memory",
        purpose="continue the active work",
        task_context={
            "human_id": "human-1",
            "project_id": "project-1",
            "conversation_id": "thread-1",
            "phase": "implementation",
        },
        run_policy=False,
    )
    all_frame_refs = set(frame["source_atom_refs"])
    all_frame_refs.update(
        ref for page in frame["page_index"] for ref in page["source_atom_refs"]
    )
    assert {matching["id"], broad["id"], tagged_global["id"]} <= all_frame_refs
    assert not ({atom["id"] for atom in mismatches} & all_frame_refs)
    assert (
        frame["compilation_trace"]["filtered_counts"]["semantic_scope_hidden"]
        >= 3
    )


def test_semantic_scope_is_digest_bound_and_revalidated_on_page_load(amos):
    seed = amos.commit_atom(
        {
            "id": "semantic_page_seed",
            "type": "belief",
            "payload": {
                "claim": "Semantic page safety active conclusion " + "detail " * 240,
                "human_id": "human-1",
                "project_id": "project-1",
                "project_thread_id": "thread-1",
            },
        }
    )["atom"]
    broad_support = amos.commit_atom(
        {
            "id": "semantic_page_global_support",
            "type": "belief",
            "payload": {"claim": "Globally applicable supporting evidence."},
        }
    )["atom"]
    wrong_required = amos.commit_atom(
        {
            "id": "semantic_page_wrong_required",
            "type": "belief",
            "payload": {
                "claim": "Semantic page safety active conclusion",
                "project_id": "project-2",
            },
        }
    )["atom"]
    wrong_support = amos.commit_atom(
        {
            "id": "semantic_page_wrong_support",
            "type": "belief",
            "payload": {
                "claim": "Supporting evidence for another human.",
                "human_id": "human-2",
            },
        }
    )["atom"]
    _insert_relation(amos, seed["id"], wrong_required["id"], "rel:requires")
    _insert_relation(amos, seed["id"], broad_support["id"], "rel:uses")
    _insert_relation(amos, seed["id"], wrong_support["id"], "rel:uses")

    frame = amos.compile_memory_frame(
        need="semantic page safety active conclusion",
        purpose="load safe supporting context",
        task_context={
            "human_id": "human-1",
            "project_id": "project-1",
            "project_thread_id": "thread-1",
        },
        token_or_byte_budget={"tokens": 1000},
        run_policy=False,
    )
    descriptor = next(
        page for page in frame["page_index"] if seed["id"] in page["focus_atom_refs"]
    )
    assert descriptor["semantic_scope"] == {
        "human_id": "human-1",
        "project_id": "project-1",
        "project_thread_id": "thread-1",
    }
    assert broad_support["id"] in descriptor["source_atom_refs"]
    assert wrong_required["id"] not in descriptor["source_atom_refs"]
    assert wrong_support["id"] not in descriptor["source_atom_refs"]

    page = amos.load_memory_page(
        frame_id=frame["frame_id"],
        revision=frame["revision"],
        page=descriptor,
        depth="supporting",
        token_or_byte_budget={"tokens": 1000},
        run_policy=False,
    )
    assert seed["id"] in page["source_atom_refs"]
    assert broad_support["id"] in page["source_atom_refs"]
    assert wrong_required["id"] not in page["source_atom_refs"]
    assert wrong_support["id"] not in page["source_atom_refs"]

    with pytest.raises(ValidationError, match="conflicts with trusted project_id"):
        amos.load_memory_page(
            frame_id=frame["frame_id"],
            revision=frame["revision"],
            page=descriptor,
            scope={"project_id": "project-2"},
            run_policy=False,
        )


def test_task_context_semantic_identifiers_are_strict(amos):
    with pytest.raises(ValidationError, match="task_context.human_id"):
        amos.compile_memory_frame(
            need="strict semantic context",
            purpose="reject model-shaped identity",
            task_context={"human_id": ["human-1"]},
            run_policy=False,
        )
    with pytest.raises(ValidationError, match="conflict"):
        amos.compile_memory_frame(
            need="strict semantic context",
            purpose="reject conflicting trusted identity",
            task_context={
                "project_thread_id": "thread-1",
                "conversation_id": "thread-2",
            },
            run_policy=False,
        )


def test_budget_derived_closure_truncation_exposes_loadable_continuation(amos):
    atoms = []
    for index in range(45):
        claim = (
            "zzseedalpha"
            if index == 0
            else "q" + chr(97 + index % 26) * 20 + str(index)
        )
        atoms.append(
            amos.commit_atom(
                {
                    "id": f"bounded_chain_{index:02d}",
                    "type": "belief",
                    "payload": {"claim": claim},
                }
            )["atom"]
        )
    with amos.store.transaction() as conn:
        for source, target in zip(atoms, atoms[1:]):
            assert amos.store.insert_edge(
                conn,
                amos.graph._edge(
                    source["id"], target["id"], "rel:derived_from", {}
                ),
            )

    frame = amos.compile_memory_frame(
        need="zzseedalpha",
        purpose="zzpurposeomega",
        token_or_byte_budget={"tokens": 1200},
        run_policy=False,
    )
    closure_unknown = next(
        item
        for item in frame["unknowns"]
        if item["reason"] == "relationship_closure_truncated"
    )
    trace = frame["compilation_trace"]
    limits = trace["relationship_work_budget"]
    assert frame["truncated"] is True
    assert closure_unknown["continuation_atom_count"] > 0
    assert trace["relationship_truncation_reasons"]
    assert trace["relationship_expansion_count"] <= limits["mandatory_atoms"]
    assert trace["mandatory_edges_examined"] <= limits["mandatory_edges"]

    descriptor = next(
        page
        for page in frame["page_index"]
        if set(page["source_atom_refs"]) - set(page["focus_atom_refs"])
    )
    continuation_refs = set(descriptor["source_atom_refs"]) - set(
        descriptor["focus_atom_refs"]
    )
    page = amos.load_memory_page(
        frame_id=frame["frame_id"],
        revision=frame["revision"],
        page=descriptor,
        depth="supporting",
        token_or_byte_budget={"tokens": 1500},
        run_policy=False,
    )
    assert continuation_refs <= set(page["source_atom_refs"])
    assert page["budget"]["used_bytes"] <= page["budget"]["limit_bytes"]


def test_budget_derived_candidate_scan_reports_truncation(amos):
    for index in range(105):
        amos.commit_atom(
            {
                "id": f"candidate_noise_{index:03d}",
                "type": "belief",
                "payload": {"claim": f"noise-{index:03d}"},
            }
        )
    seed = amos.commit_atom(
        {
            "id": "candidate_budget_seed",
            "type": "belief",
            "payload": {"claim": "candidatebudgetseed"},
        }
    )["atom"]

    frame = amos.compile_memory_frame(
        need="candidatebudgetseed",
        purpose="candidatebudgetpurpose",
        token_or_byte_budget={"tokens": 800},
        run_policy=False,
    )
    available_refs = set(frame["source_atom_refs"])
    available_refs.update(
        ref for descriptor in frame["page_index"] for ref in descriptor["source_atom_refs"]
    )
    assert seed["id"] in available_refs
    assert frame["compilation_trace"]["candidate_generation_truncated"] is True
    assert any(
        unknown["reason"] == "candidate_generation_truncated"
        for unknown in frame["unknowns"]
    )


def test_all_compression_levels_preserve_governing_memory(amos):
    unit = {
        "unit_id": "compression_unit",
        "unit_type": "decision_chain",
        "title": "Compressed decision history",
        "summary": "The active conclusion remains authoritative.",
        "relevance_score": 1.0,
        "active_conclusion_refs": ["active"],
        "constraint_refs": ["constraint"],
        "commitment_refs": ["commitment"],
        "conflict_refs": ["edge_conflict"],
        "source_atom_refs": [
            "historical",
            "active",
            "constraint",
            "commitment",
            "counter",
        ],
        "relationship_refs": [
            "edge_supersedes",
            "edge_constraint",
            "edge_commitment",
            "edge_conflict",
        ],
        "items": [
            {
                "atom_ref": ref,
                "type": "belief",
                "rendered_content": {
                    "format": "text",
                    "text": ref + " detail " * 100,
                },
                "confidence": {"score": 0.9},
                "evidence_refs": [],
                "scope": {},
                "lifecycle_state": "active",
                "health_status": "healthy",
                "updated_at": "2026-07-22T00:00:00Z",
            }
            for ref in (
                "historical",
                "active",
                "constraint",
                "commitment",
                "counter",
            )
        ],
        "relationships": [
            {
                "edge_id": "edge_supersedes",
                "relation": "rel:supersedes",
                "source_ref": "active",
                "target_ref": "historical",
            },
            {
                "edge_id": "edge_constraint",
                "relation": "rel:constrained_by",
                "source_ref": "active",
                "target_ref": "constraint",
            },
            {
                "edge_id": "edge_commitment",
                "relation": "rel:requires",
                "source_ref": "active",
                "target_ref": "commitment",
            },
            {
                "edge_id": "edge_conflict",
                "relation": "rel:contradicts",
                "source_ref": "active",
                "target_ref": "counter",
            },
        ],
        "sequence": [
            {"atom_ref": ref, "observed_at": None, "role": "history"}
            for ref in (
                "historical",
                "active",
                "constraint",
                "commitment",
                "counter",
            )
        ],
        "inclusion_reasons": {},
        "compression": {"mode": "none"},
        "truncated": False,
    }
    required_refs = {"active", "constraint", "commitment", "counter"}
    required_relations = {
        "rel:supersedes",
        "rel:constrained_by",
        "rel:requires",
        "rel:contradicts",
    }

    projected = amos.reasoning._compress_unit(unit)
    summary = amos.reasoning._reference_unit(projected)
    reference = amos.reasoning._bare_reference_unit(summary)

    assert {item["atom_ref"] for item in projected["items"]} == required_refs
    assert {item["atom_ref"] for item in summary["items"]} == required_refs
    assert all(
        len(item["rendered_content"]["text"]) <= 180
        for item in summary["items"]
    )
    for compressed in (projected, summary, reference):
        assert compressed["active_conclusion_refs"] == ["active"]
        assert compressed["constraint_refs"] == ["constraint"]
        assert compressed["commitment_refs"] == ["commitment"]
        assert compressed["conflict_refs"] == ["edge_conflict"]
        assert compressed["sequence"] == unit["sequence"]
        assert required_relations <= {
            relation["relation"] for relation in compressed["relationships"]
        }
    assert projected["compression"]["mode"] == "essential_projection"
    assert summary["compression"]["mode"] == "reference_summary"
    assert reference["compression"]["mode"] == "reference_only"


def test_constrained_frame_compresses_resident_before_dropping_independent_pages(amos):
    decision_old = amos.commit_atom(
        {
            "id": "bounded_two_domain_decision_old",
            "type": "belief",
            "payload": {
                "claim": "Two-domain paging originally used the obsolete decision. "
                + "historical decision detail " * 90
            },
        }
    )["atom"]
    decision_new = amos.commit_atom(
        {
            "id": "bounded_two_domain_decision_new",
            "type": "belief",
            "payload": {
                "claim": "Two-domain paging uses the active decision. "
                + "current decision detail " * 90
            },
            "supersedes": [decision_old["id"]],
        }
    )["atom"]
    commitment_old = amos.commit_atom(
        {
            "id": "bounded_two_domain_commitment_old",
            "type": "commitment",
            "payload": {
                "description": "Two-domain paging originally committed to Friday. "
                + "historical commitment detail " * 90
            },
        }
    )["atom"]
    commitment_new = amos.commit_atom(
        {
            "id": "bounded_two_domain_commitment_new",
            "type": "commitment",
            "payload": {
                "description": "Two-domain paging now commits to Monday. "
                + "current commitment detail " * 90
            },
            "supersedes": [commitment_old["id"]],
        }
    )["atom"]
    task_context = {
        "human_id": "human-1",
        "project_id": "project-1",
        "task": "inspect two independent coherent domains",
        "request_lifecycle": {
            "explanation": "trusted runtime detail " * 80,
            "semantic_routes": {"historical_continuity": True},
        },
    }
    frame = amos.compile_memory_frame(
        need="two-domain paging active decision Monday commitment",
        purpose="inspect two independent coherent domains",
        task_context=task_context,
        token_or_byte_budget={"tokens": 1600},
        run_policy=False,
    )

    assert set(frame["request"]) == {
        "request_digest",
        "depth",
        "requester",
        "target_processor",
        "token_or_byte_budget",
    }
    assert "request_lifecycle" not in frame["orientation"]["task_context"]
    assert frame["budget"]["used_bytes"] <= frame["budget"]["limit_bytes"]
    assert frame["compilation_trace"]["compressed_unit_count"] >= 1
    assert any(
        (unit.get("compression") or {}).get("mode") != "none"
        for unit in frame["units"]
    )

    descriptors = frame["page_index"]
    compressed_unit_refs = {
        unit["unit_id"]
        for unit in frame["units"]
        if (unit.get("compression") or {}).get("mode") != "none"
    }
    assert compressed_unit_refs <= {
        descriptor["unit_ref"] for descriptor in descriptors
    }
    assert any(
        decision_new["id"] in descriptor["source_atom_refs"]
        for descriptor in descriptors
    )
    assert any(
        commitment_new["id"] in descriptor["source_atom_refs"]
        for descriptor in descriptors
    )
    architecture_descriptor = next(
        descriptor
        for descriptor in descriptors
        if decision_new["id"] in descriptor["source_atom_refs"]
    )
    commitment_descriptor = next(
        descriptor
        for descriptor in descriptors
        if commitment_new["id"] in descriptor["source_atom_refs"]
    )
    assert architecture_descriptor["page_id"] != commitment_descriptor["page_id"]
    assert any(
        (unit.get("compression") or {}).get("mode") == "reference_only"
        for unit in frame["units"]
    )


def test_reasoning_contract_has_no_fixed_atom_count_and_schemas_parse():
    parameters = inspect.signature(
        __import__("amos").Amos.compile_memory_frame
    ).parameters
    assert "max_items" not in parameters
    for name in (
        "memory_frame_request.schema.json",
        "memory_frame.schema.json",
        "memory_page_request.schema.json",
        "memory_page.schema.json",
    ):
        schema = json.loads((Path("schemas") / name).read_text())
        assert schema["$schema"].endswith("2020-12/schema")
    request_schema = json.loads(
        Path("schemas/memory_frame_request.schema.json").read_text()
    )
    assert "max_items" not in request_schema["properties"]
