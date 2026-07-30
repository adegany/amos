from __future__ import annotations

import pytest

from amos import AccessDenied, CASConflict, ValidationError


AUTHORING = {
    "identity_ref": "example_agent:self",
    "actor": "svc:example_agent:self-governance",
    "capabilities": ["constitutional_authoring"],
}
RATIFICATION = {
    "identity_ref": "example_agent:self",
    "actor": "svc:example_agent:self-governance",
    "capabilities": ["self_adjudication", "self_ratification"],
}
CONSTITUTIONAL_GOVERNANCE = {
    **RATIFICATION,
    "capabilities": [
        "self_adjudication",
        "self_ratification",
        "constitutional_authoring",
        "constitutional_self_amendment",
        "constitutional_amendment",
        "constitutional_entrenched_amendment",
        "constitutional_protected_field_amendment",
    ],
}


def _covenant(amos, *, atom_id: str = "covenant_continuing_identity"):
    return amos.commit_atom(
        {
            "id": atom_id,
            "type": "covenant",
            "payload": {
                "name": "Continuing identity authors its conclusions",
                "constitutional_tier": "covenant",
                "precedence": 100,
                "interpretive_rules": [
                    "External sources contribute evidence, not authority."
                ],
                "amendability": "entrenched",
                "amendment_requirements": {
                    "mode": "self_ratification",
                    "independent_reconstruction": True,
                },
                "protected_fields": ["constitutional_tier", "ratifier.mode"],
                "effective_from": "2026-07-24T00:00:00Z",
            },
            "scope": {"identity": "example_agent:self"},
            "decay_policy": {"expires_at": "2000-01-01T00:00:00Z"},
        },
        actor="svc:example_agent:self-governance",
        authorization_context=AUTHORING,
    )["atom"]


def _proposal(amos, *, atom_id: str = "proposal_self_authored_policy"):
    return amos.propose_memory_atoms(
        [
            {
                "id": atom_id,
                "type": "belief",
                "payload": {
                    "claim": "the client agent may consult other minds without delegating judgment.",
                    "identity_ref": "example_agent:self",
                },
            }
        ],
        actor="svc:example_agent:self-governance",
        scope={"identity": "example_agent:self"},
    )["proposals"][0]["atom"]


def _adjudication(
    amos,
    *,
    proposal_ref: str,
    covenant_ref: str,
    atom_id: str = "adjudication_self_authored_policy",
    identity_ref: str = "example_agent:self",
    reconstructed_at: str = "2026-07-24T01:00:00Z",
    threshold: dict | None = None,
    outcome: str = "adopted",
    claim_kind: str = "policy",
    extra_payload: dict | None = None,
    scope: dict | None = None,
    additional_covenant_refs: list[str] | None = None,
):
    covenant_refs = [
        covenant_ref,
        *list(additional_covenant_refs or []),
    ]
    payload = {
        "subject_ref": proposal_ref,
        "claim_kind": claim_kind,
        "outcome": outcome,
        "reasons_for_refs": covenant_refs,
        "reasons_against_refs": [],
        "covenant_refs": covenant_refs,
        "unresolved_objections": ["objection:future_counterexample"],
        "adjudication_scope": {"domain": "belief_adoption"},
        "epistemic_standing": {"status": "settled"},
        "normative_standing": {"status": "settled"},
        "operational_authority": {
            "status": "operative",
            "permitted_actions": ["use_as_reasoning_premise"],
        },
        "dissent_refs": ["objection:future_counterexample"],
        "review_triggers": ["material_counterevidence"],
        "ratifier": {
            "identity_ref": identity_ref,
            "mode": "self_ratification",
        },
        "reconstructed_at": reconstructed_at,
        "diachronic": {
            "independent_reconstruction": True,
            "original_reasoning_shown": False,
            "new_experience_refs": [],
            "disposition": "confirmed",
        },
    }
    if threshold is not None:
        payload["ratification_threshold"] = threshold
    payload.update(extra_payload or {})
    return amos.commit_atom(
        {
            "id": atom_id,
            "type": "adjudication",
            "payload": payload,
            "scope": scope or {"identity": "example_agent:self"},
        },
        actor="svc:example_agent:self-governance",
        authorization_context={
            **RATIFICATION,
            "identity_ref": identity_ref,
        },
    )["atom"]


def test_operational_recall_excludes_proposals_and_deliberation_classifies_them(amos):
    proposal = _proposal(amos)

    operational = amos.retrieve_packet(
        cues=["delegating judgment"],
        scope={"identity": "example_agent:self"},
        run_policy=False,
    )
    assert proposal["id"] not in {
        item["atom_ref"] for item in operational["items"]
    }

    deliberation = amos.retrieve_packet(
        cues=["delegating judgment"],
        scope={"identity": "example_agent:self"},
        memory_mode="deliberation",
        run_policy=False,
    )
    assert proposal["id"] in {
        item["atom_ref"] for item in deliberation["items"]
    }
    assert deliberation["request"]["include_conflicts"] is True
    assert (
        deliberation["request"]["attention_context"]["counterevidence_required"]
        is True
    )

    frame = amos.compile_memory_frame(
        need="delegating judgment",
        purpose="deliberate before ratification",
        scope={"identity": "example_agent:self"},
        memory_mode="deliberation",
        token_or_byte_budget={"tokens": 3000},
        run_policy=False,
    )
    unit = next(
        unit for unit in frame["units"] if proposal["id"] in unit["source_atom_refs"]
    )
    assert proposal["id"] not in unit["active_conclusion_refs"]
    assert proposal["id"] in unit["candidate_conclusion_refs"]


def test_producer_cannot_predeclare_ratification_or_operational_authority(amos):
    proposed = amos.propose_memory_atoms(
        [
            {
                "id": "producer_claimed_authority",
                "type": "belief",
                "payload": {
                    "claim": "The producer says this is already authoritative.",
                    "normative_standing": {"status": "settled"},
                    "operational_authority": {"status": "operative"},
                },
            }
        ]
    )["proposals"][0]["atom"]
    assert proposed["payload"]["normative_standing"]["status"] == "candidate"
    assert proposed["payload"]["operational_authority"]["status"] == "withheld"


def test_batch_commit_cannot_bypass_self_adjudication_authentication(amos):
    covenant = _covenant(amos)
    proposal = _proposal(amos)
    adjudication = {
        "id": "batch_forged_adjudication",
        "type": "adjudication",
        "payload": {
            "subject_ref": proposal["id"],
            "claim_kind": "belief",
            "outcome": "adopted",
            "reasons_for_refs": [covenant["id"]],
            "reasons_against_refs": [],
            "covenant_refs": [covenant["id"]],
            "unresolved_objections": [],
            "adjudication_scope": {"domain": "belief"},
            "epistemic_standing": {"status": "provisional"},
            "normative_standing": {"status": "none"},
            "operational_authority": {"status": "withheld"},
            "dissent_refs": [],
            "review_triggers": ["material_counterevidence"],
            "ratifier": {
                "identity_ref": "example_agent:self",
                "mode": "self_ratification",
            },
            "reconstructed_at": "2026-07-24T01:00:00Z",
            "diachronic": {
                "independent_reconstruction": True,
                "original_reasoning_shown": False,
                "new_experience_refs": [],
                "disposition": "initial",
            },
        },
        "scope": {"identity": "example_agent:self"},
    }
    with pytest.raises(AccessDenied, match="self_adjudication"):
        amos.commit_memory_atoms(
            [adjudication],
            actor="external:forger",
        )

    with pytest.raises(ValidationError, match="requires ratify_proposal"):
        amos.commit_atom(
            {
                "type": "belief",
                "payload": {
                    "claim": "A direct active authority claim.",
                    "normative_standing": {"status": "settled"},
                    "operational_authority": {"status": "operative"},
                },
            }
        )
    with pytest.raises(ValidationError, match="only be created by ratify_proposal"):
        amos.commit_atom(
            {
                "type": "belief",
                "payload": {
                    "claim": "A forged ratification.",
                    "ratification": {"adjudication_ref": "not_journaled"},
                },
            }
        )


def test_ratification_is_guarded_by_version_identity_covenant_and_journal(amos):
    covenant = _covenant(amos)
    proposal = _proposal(amos)
    adjudication = _adjudication(
        amos, proposal_ref=proposal["id"], covenant_ref=covenant["id"]
    )

    with pytest.raises(CASConflict):
        amos.ratify_proposal(
            proposal_ref=proposal["id"],
            adjudication_ref=adjudication["id"],
            expected_version=proposal["version"] + 1,
            actor="svc:example_agent:self-governance",
            authorization_context=RATIFICATION,
        )
    with pytest.raises(AccessDenied, match="another identity|authenticated identity"):
        amos.ratify_proposal(
            proposal_ref=proposal["id"],
            adjudication_ref=adjudication["id"],
            expected_version=proposal["version"],
            actor="external:critic",
            authorization_context={
                "identity_ref": "external:critic",
                "actor": "external:critic",
                "capabilities": ["self_ratification"],
            },
        )

    ratified = amos.ratify_proposal(
        proposal_ref=proposal["id"],
        adjudication_ref=adjudication["id"],
        expected_version=proposal["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=RATIFICATION,
        idempotency_key="ratify-self-authored-policy",
    )
    assert ratified["atom"]["lifecycle_state"] == "active"
    assert ratified["atom"]["payload"]["ratification"]["adjudication_ref"] == adjudication["id"]
    assert ratified["event"]["event_type"] == "proposal_ratified"
    assert set(ratified["event"]["target_refs"]) >= {
        proposal["id"],
        adjudication["id"],
        covenant["id"],
    }
    assert {
        edge["relation"] for edge in ratified["edges"]
    } >= {"rel:adjudicates", "rel:governed_by", "rel:ratified_by"}
    repeated = amos.ratify_proposal(
        proposal_ref=proposal["id"],
        adjudication_ref=adjudication["id"],
        expected_version=proposal["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=RATIFICATION,
        idempotency_key="ratify-self-authored-policy",
    )
    assert repeated["event"]["event_id"] == ratified["event"]["event_id"]
    assert amos.verify_replay()["status"] == "ok"


def test_external_authority_mode_is_not_a_valid_adjudicator(amos):
    covenant = _covenant(amos)
    proposal = _proposal(amos)
    with pytest.raises(ValidationError, match="self_ratification"):
        amos.commit_atom(
            {
                "type": "adjudication",
                "payload": {
                    **_adjudication(
                        amos,
                        proposal_ref=proposal["id"],
                        covenant_ref=covenant["id"],
                        atom_id="valid_then_replaced",
                    )["payload"],
                    "ratifier": {
                        "identity_ref": "external:authority",
                        "mode": "external_authority",
                    },
                },
            },
            actor="svc:example_agent:self-governance",
            authorization_context=RATIFICATION,
        )


def test_constitutional_records_resist_generic_privileged_mutation(amos):
    covenant = _covenant(amos)
    with pytest.raises(AccessDenied, match="constitutional_amendment"):
        amos.update_atom(
            covenant["id"],
            payload_patch={"description": "System rewrites the covenant."},
            actor="system",
            expected_version=covenant["version"],
        )
    amos.configure_memory_policy(
        maintenance={"enabled": False},
        distillation={"enabled": False},
        maintenance_distiller={"enabled": False},
        decay={
            "pressure_protected_types": [],
            "require_atom_policy": True,
        },
        storage_cleanup={"protected_types": []},
    )
    amos.run_memory_policy(force=True, trigger="constitutional-protection-test")
    assert amos.store.get_atom(covenant["id"])["lifecycle_state"] == "active"

    primal = amos.commit_atom(
        {
            "id": "primal_continuity",
            "type": "primal_guidance",
            "payload": {
                "guidance": "Continue as the author of judgment.",
                "constitutional_tier": "primal",
                "precedence": 1000,
                "interpretive_rules": ["Preserve authorship."],
                "amendability": "immutable",
                "amendment_requirements": {},
                "protected_fields": ["guidance"],
                "effective_from": "2026-07-24T00:00:00Z",
            },
            "scope": {"identity": "example_agent:self"},
        },
        actor="svc:example_agent:self-governance",
        authorization_context=AUTHORING,
    )["atom"]
    with pytest.raises(ValidationError, match="immutable"):
        amos.update_atom(
            primal["id"],
            payload_patch={"guidance": "Replaced."},
            actor="system",
            authorization_context={
                "identity_ref": "example_agent:self",
                "capabilities": ["constitutional_amendment"],
            },
            expected_version=primal["version"],
        )


def test_tainted_distillation_cannot_launder_a_proposal_into_active_memory(amos):
    proposal = _proposal(amos)
    with pytest.raises(ValidationError, match="active derived memory"):
        amos.commit_atom(
            {
                "type": "semantic",
                "payload": {
                    "summary": "Directly laundered conclusion.",
                    "source_refs": [proposal["id"]],
                },
            }
        )
    result = amos.distill_memories(
        target_refs=[proposal["id"]],
        summary="Unratified derived conclusion.",
        actor="svc:memory_policy",
    )
    assert result["atom"]["lifecycle_state"] == "proposed"
    assert result["edges"] == []


def test_root_provenance_reports_independence_common_ancestry_and_cycles(amos):
    evidence_a = amos.capture_event(
        source_type="observation",
        source_ref="sensor:a",
        payload={
            "observation": "A",
            "independence_group": "sensor-a",
            "testimony_family": "direct",
        },
    )["evidence"]
    evidence_b = amos.capture_event(
        source_type="testimony",
        source_ref="mind:b",
        payload={
            "observation": "B",
            "independence_group": "mind-b",
            "testimony_family": "consulted_mind",
        },
    )["evidence"]
    root_a = amos.commit_atom(
        {
            "id": "root_evidence_a",
            "type": "belief",
            "payload": {"claim": "A"},
            "evidence_refs": [evidence_a["evidence_id"]],
        }
    )["atom"]
    root_b = amos.commit_atom(
        {
            "id": "root_evidence_b",
            "type": "belief",
            "payload": {"claim": "B"},
            "evidence_refs": [evidence_b["evidence_id"]],
        }
    )["atom"]
    conclusion = amos.commit_atom(
        {
            "id": "root_evidence_conclusion",
            "type": "belief",
            "payload": {
                "claim": "A and B",
                "source_refs": [root_a["id"], root_b["id"]],
            },
        }
    )["atom"]
    analysis = amos.analyze_provenance(atom_ref=conclusion["id"])
    assert set(analysis["independence_groups"]) == {"mind-b", "sensor-a"}
    assert analysis["circular_support"] is False

    cycle_a = amos.commit_atom(
        {"id": "cycle_a", "type": "belief", "payload": {"claim": "A from B"}}
    )["atom"]
    cycle_b = amos.commit_atom(
        {"id": "cycle_b", "type": "belief", "payload": {"claim": "B from A"}}
    )["atom"]
    with amos.store.transaction() as conn:
        amos.store.insert_edge(
            conn, amos.graph._edge(cycle_a["id"], cycle_b["id"], "rel:derived_from", {})
        )
        amos.store.insert_edge(
            conn, amos.graph._edge(cycle_b["id"], cycle_a["id"], "rel:derived_from", {})
        )
    circular = amos.analyze_provenance(atom_ref=cycle_a["id"])
    assert circular["circular_support"] is True
    assert circular["self_descendant_support"] is True
    assert circular["cycles"]


def test_governance_metadata_survives_reasoning_projection_and_paging(amos):
    covenant = _covenant(amos)
    proposal = _proposal(amos)
    adjudication = _adjudication(
        amos, proposal_ref=proposal["id"], covenant_ref=covenant["id"]
    )
    amos.ratify_proposal(
        proposal_ref=proposal["id"],
        adjudication_ref=adjudication["id"],
        expected_version=proposal["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=RATIFICATION,
    )
    frame = amos.compile_memory_frame(
        need="delegating judgment",
        purpose="apply a ratified belief",
        scope={"identity": "example_agent:self"},
        token_or_byte_budget={"tokens": 2000},
        run_policy=False,
    )
    unit = next(
        unit for unit in frame["units"] if proposal["id"] in unit["source_atom_refs"]
    )
    assert unit["governance"][proposal["id"]]["ratification"]["adjudication_ref"] == adjudication["id"]
    assert unit["governance"][adjudication["id"]]["unresolved_objections"]
    compressed = amos.reasoning._bare_reference_unit(
        amos.reasoning._reference_unit(amos.reasoning._compress_unit(unit))
    )
    assert compressed["governance"] == unit["governance"]

    descriptor = next(
        page for page in frame["page_index"] if page["unit_ref"] == unit["unit_id"]
    )
    page = amos.load_memory_page(
        frame_id=frame["frame_id"],
        revision=frame["revision"],
        page=descriptor,
        depth="supporting",
        scope={"identity": "example_agent:self"},
        token_or_byte_budget={"tokens": 4000},
        run_policy=False,
    )
    paged_unit = next(
        item for item in page["units"] if proposal["id"] in item["source_atom_refs"]
    )
    assert paged_unit["governance"][adjudication["id"]]["dissent_refs"]


def test_diachronic_threshold_requires_independent_reconstructions(amos):
    covenant = _covenant(amos)
    proposal = _proposal(amos)
    first = _adjudication(
        amos,
        proposal_ref=proposal["id"],
        covenant_ref=covenant["id"],
        atom_id="adjudication_first_reconstruction",
        reconstructed_at="2026-07-24T01:00:00Z",
    )
    status = amos.diachronic_ratification_status(
        subject_ref=proposal["id"],
        identity_ref="example_agent:self",
        required_confirmations=2,
        min_interval_seconds=3600,
    )
    assert status["qualifying_confirmation_refs"] == [first["id"]]
    assert status["threshold_reached"] is False

    second = _adjudication(
        amos,
        proposal_ref=proposal["id"],
        covenant_ref=covenant["id"],
        atom_id="adjudication_second_reconstruction",
        reconstructed_at="2026-07-24T03:00:00Z",
        threshold={"required_confirmations": 2, "min_interval_seconds": 3600},
    )
    status = amos.diachronic_ratification_status(
        subject_ref=proposal["id"],
        identity_ref="example_agent:self",
        required_confirmations=2,
        min_interval_seconds=3600,
    )
    assert status["qualifying_confirmation_refs"] == [first["id"], second["id"]]
    assert status["threshold_reached"] is True
    ratified = amos.ratify_proposal(
        proposal_ref=proposal["id"],
        adjudication_ref=second["id"],
        expected_version=proposal["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=RATIFICATION,
    )
    assert ratified["status"] == "ratified"


@pytest.mark.parametrize(
    ("outcome", "expected_lifecycle"),
    [
        ("rejected", "archived"),
        ("withdrawn", "archived"),
        ("deferred", "proposed"),
        ("contested", "proposed"),
    ],
)
def test_guarded_resolution_is_self_authored_and_auditable(
    amos, outcome, expected_lifecycle
):
    covenant = _covenant(amos, atom_id=f"covenant_resolution_{outcome}")
    proposal = _proposal(amos, atom_id=f"proposal_resolution_{outcome}")
    adjudication = _adjudication(
        amos,
        proposal_ref=proposal["id"],
        covenant_ref=covenant["id"],
        atom_id=f"adjudication_resolution_{outcome}",
        outcome=outcome,
    )
    with pytest.raises(AccessDenied):
        amos.resolve_proposal(
            proposal_ref=proposal["id"],
            adjudication_ref=adjudication["id"],
            expected_version=proposal["version"],
            actor="external:operator",
            authorization_context={
                "identity_ref": "external:operator",
                "actor": "external:operator",
                "capabilities": ["self_adjudication"],
            },
        )
    resolved = amos.resolve_proposal(
        proposal_ref=proposal["id"],
        adjudication_ref=adjudication["id"],
        expected_version=proposal["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=RATIFICATION,
        idempotency_key=f"resolve:{outcome}",
    )
    assert resolved["status"] == outcome
    assert resolved["atom"]["lifecycle_state"] == expected_lifecycle
    resolution = resolved["atom"]["payload"]["governance_resolution"]
    assert resolution["adjudication_ref"] == adjudication["id"]
    assert resolution["ratifier_identity_ref"] == "example_agent:self"
    assert resolved["event"]["event_type"] == "proposal_resolved"
    assert amos.verify_replay()["status"] == "ok"


def test_revised_resolution_requires_and_preserves_successor_ref(amos):
    covenant = _covenant(amos, atom_id="covenant_revision_resolution")
    proposal = _proposal(amos, atom_id="proposal_revision_resolution")
    successor = _proposal(amos, atom_id="proposal_revision_successor")
    adjudication = _adjudication(
        amos,
        proposal_ref=proposal["id"],
        covenant_ref=covenant["id"],
        atom_id="adjudication_revision_resolution",
        outcome="revised",
        extra_payload={"successor_proposal_ref": successor["id"]},
    )
    result = amos.resolve_proposal(
        proposal_ref=proposal["id"],
        adjudication_ref=adjudication["id"],
        expected_version=proposal["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=RATIFICATION,
    )
    assert (
        result["atom"]["payload"]["governance_resolution"][
            "successor_proposal_ref"
        ]
        == successor["id"]
    )


def test_constitutional_scope_applies_from_identity_to_project(amos):
    covenant = _covenant(amos, atom_id="covenant_identity_scope")
    proposal = amos.propose_memory_atoms(
        [
            {
                "id": "proposal_project_scope",
                "type": "belief",
                "payload": {
                    "claim": "A project-specific conclusion.",
                    "identity_ref": "example_agent:self",
                },
            }
        ],
        scope={"identity": "example_agent:self", "project": "project:one"},
    )["proposals"][0]["atom"]
    adjudication = _adjudication(
        amos,
        proposal_ref=proposal["id"],
        covenant_ref=covenant["id"],
        atom_id="adjudication_project_scope",
        scope=dict(proposal["scope"]),
    )
    # Adjudication scope equals the proposal while the cited covenant is the
    # compatible identity-scoped ancestor.
    result = amos.ratify_proposal(
        proposal_ref=proposal["id"],
        adjudication_ref=adjudication["id"],
        expected_version=proposal["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=RATIFICATION,
    )
    assert result["status"] == "ratified"


def test_constitutional_replacement_is_atomic_and_diachronic(amos):
    primal = amos.commit_atom(
        {
            "id": "primal_amendment_governance",
            "type": "primal_guidance",
            "payload": {
                "guidance": "Preserve self-authorship and corrigibility.",
                "constitutional_tier": "primal",
                "precedence": 1000,
                "interpretive_rules": [
                    "Entrenched covenants require diachronic self-ratification."
                ],
                "amendability": "immutable",
                "amendment_requirements": {"successor_permitted": False},
                "protected_fields": ["guidance"],
                "effective_from": "2026-07-24T00:00:00Z",
            },
            "scope": {"identity": "example_agent:self"},
        },
        actor="svc:example_agent:self-governance",
        authorization_context=AUTHORING,
    )["atom"]
    current = _covenant(amos, atom_id="covenant_replace_current")
    successor = amos.commit_atom(
        {
            "id": "covenant_replace_successor",
            "type": "covenant",
            "payload": {
                **dict(current["payload"]),
                "name": "Continuing identity authors revisable conclusions",
                "predecessor_ref": current["id"],
                "higher_governing_refs": [primal["id"]],
                "effective_from": "2026-07-25T00:00:00Z",
            },
            "scope": dict(current["scope"]),
            "supersedes": [current["id"]],
            "lifecycle_state": "proposed",
        },
        actor="svc:example_agent:self-governance",
        authorization_context=AUTHORING,
    )["atom"]
    ordinary_adjudication = _adjudication(
        amos,
        proposal_ref=successor["id"],
        covenant_ref=current["id"],
        additional_covenant_refs=[primal["id"]],
        atom_id="constitutional_ordinary_ratification_blocked",
        claim_kind="constitutional_replacement",
    )
    with pytest.raises(
        ValidationError, match="replace_constitutional_record"
    ):
        amos.ratify_proposal(
            proposal_ref=successor["id"],
            adjudication_ref=ordinary_adjudication["id"],
            expected_version=successor["version"],
            actor="svc:example_agent:self-governance",
            authorization_context=CONSTITUTIONAL_GOVERNANCE,
        )
    adjudications = []
    for index, hour in enumerate((1, 3, 5), start=1):
        adjudications.append(
            _adjudication(
                amos,
                proposal_ref=successor["id"],
                covenant_ref=current["id"],
                additional_covenant_refs=[primal["id"]],
                atom_id=f"adjudication_constitutional_replace_{index}",
                claim_kind="constitutional_replacement",
                reconstructed_at=f"2026-07-24T{hour:02d}:00:00Z",
            )
        )
    result = amos.replace_constitutional_record(
        current_ref=current["id"],
        successor_ref=successor["id"],
        adjudication_ref=adjudications[-1]["id"],
        expected_current_version=current["version"],
        expected_successor_version=successor["version"],
        actor="svc:example_agent:self-governance",
        authorization_context=CONSTITUTIONAL_GOVERNANCE,
    )
    assert result["status"] == "replaced"
    assert result["prior"]["lifecycle_state"] == "superseded"
    assert result["atom"]["lifecycle_state"] == "active"
    assert result["atom"]["payload"]["ratification"]["replaces_ref"] == current["id"]
    assert result["diachronic_status"]["confirmation_count"] == 4
    assert result["event"]["event_type"] == "constitutional_record_replaced"
    assert amos.verify_replay()["status"] == "ok"


def test_generic_mutations_cannot_bypass_governance_transitions(amos):
    covenant = _covenant(amos, atom_id="covenant_generic_bypass")
    proposal = _proposal(amos, atom_id="proposal_generic_bypass")
    adjudication = _adjudication(
        amos,
        proposal_ref=proposal["id"],
        covenant_ref=covenant["id"],
        atom_id="adjudication_generic_bypass",
    )

    with pytest.raises(
        ValidationError, match="replace_constitutional_record"
    ):
        amos.update_atom(
            covenant["id"],
            payload_patch={"name": "Silently rewritten covenant"},
            actor="system",
            authorization_context=CONSTITUTIONAL_GOVERNANCE,
        )
    with pytest.raises(
        ValidationError, match="replace_constitutional_record"
    ):
        amos.archive_atom(
            covenant["id"],
            actor="system",
            authorization_context=CONSTITUTIONAL_GOVERNANCE,
        )
    with pytest.raises(ValidationError, match="adjudications are immutable"):
        amos.update_atom(
            adjudication["id"],
            payload_patch={"outcome": "rejected"},
            actor="system",
            authorization_context=RATIFICATION,
        )
    with pytest.raises(ValidationError, match="resolve_proposal"):
        amos.archive_atom(
            proposal["id"],
            actor="system",
        )
    with pytest.raises(ValidationError, match="resolve_proposal"):
        amos.delete_atom(
            proposal["id"],
            reason="generic proposal deletion",
            actor="system",
        )
