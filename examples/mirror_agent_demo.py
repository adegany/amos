"""AMOS Mirror Agent integration demo.

Run with:

    PYTHONPATH=src python examples/mirror_agent_demo.py --format text

The demo dogfoods AMOS: a self-modeling project assistant stores its identity,
goals, commitments, limitations, procedures, corrections, retrieved memory
packets, maintenance actions, and capacity pressure in one shared AMOS store.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from amos import (
    Amos,
    CapacityGovernor,
    MaintenanceProposal,
    ValidationError,
)


AGENT_ID = "ent:agent:mirror"
USER_ID = "ent:user:primary"
SCOPE = {"project": "amos", "demo": "mirror_agent"}


class MirrorDemoTrainingProcessor:
    processor_id = "mirror.demo.training.v1"
    processor_version = "mirror.demo.training.v1"

    def supports(self, window):
        return window.domain == "mirror_demo"

    def propose(self, window):
        directives = [
            atom
            for atom in window.atoms
            if atom.get("payload", {}).get("demo_kind") == "directive"
        ]
        outcomes = [
            atom
            for atom in window.atoms
            if atom.get("payload", {}).get("demo_kind") == "reflection"
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
                        reason_code="mirror_demo_supported_training_lesson",
                        source_refs=source_refs,
                        payload={
                            "atom": {
                                "type": "semantic",
                                "payload": {
                                    "distillation_type": "mirror_demo_training_lesson",
                                    "summary": (
                                        "Demo training controls produced "
                                        f"score_delta={current - previous:+.3f}."
                                    ),
                                    "source_refs": list(source_refs),
                                    "control_signature": signature,
                                    "metric_deltas": {
                                        "score": round(current - previous, 6)
                                    },
                                    "created_by_processor": self.processor_id,
                                },
                                "scope": dict(window.scope),
                                "layer": "consolidated_long_term",
                                "retention_class": "distilled",
                                "salience": 0.82,
                                "utility": 0.86,
                                "confidence": {
                                    "level": "medium-high",
                                    "score": 0.78,
                                },
                            }
                        },
                    )
                )
        return proposals


class MirrorAgentDemo:
    def __init__(self, amos: Amos, *, db_path: str):
        self.amos = amos
        self.db_path = db_path
        self.evidence: list[dict[str, Any]] = []
        self.chat: list[dict[str, Any]] = []
        self.service_views: dict[str, dict[str, Any]] = {}
        self.packets: dict[str, dict[str, Any]] = {}
        self.scenario_results: dict[str, dict[str, Any]] = {}

    def run(self) -> dict[str, Any]:
        self.bootstrap_self_model()
        self.scenario_self_model_bootstrap()
        self.scenario_cross_session_continuity()
        self.scenario_correction_driven_improvement()
        self.scenario_introspective_explanation()
        self.scenario_shared_service_coherence()
        self.scenario_capacity_pressure()
        self.scenario_non_llm_maintenance()
        return self.report()

    def bootstrap_self_model(self) -> None:
        evidence = self.capture(
            "system_config",
            "mirror-agent/bootstrap",
            {
                "agent_id": AGENT_ID,
                "role": "self-modeling project assistant",
                "memory_plane": "amos",
            },
        )
        evidence_refs = [evidence["evidence_id"]]
        self.commit_once(
            {
                "id": "mirror_identity",
                "type": "belief",
                "payload": {
                    "claim": "The Amos Mirror Agent is a self-modeling project assistant.",
                    "subject": AGENT_ID,
                    "relation": "is_a",
                    "object": "self_modeling_project_assistant",
                    "modality": "system_declared",
                },
                "scope": SCOPE,
                "evidence_refs": evidence_refs,
                "confidence": {"level": "high"},
            }
        )
        self.commit_once(
            {
                "id": "mirror_uses_amos",
                "type": "belief",
                "payload": {
                    "claim": "The Mirror Agent uses AMOS as its externalized memory operating system.",
                    "subject": AGENT_ID,
                    "relation": "uses_memory_system",
                    "object": "ent:system:amos",
                    "modality": "system_declared",
                },
                "scope": SCOPE,
                "evidence_refs": evidence_refs,
                "confidence": {"level": "high"},
            }
        )
        self.commit_once(
            {
                "id": "mirror_self_model",
                "type": "self_model",
                "payload": {
                    "subject_agent": AGENT_ID,
                    "role": "AMOS project assistant",
                    "operating_mode": "memory-backed design collaborator",
                    "self_description": (
                        "A project assistant that externalizes operational memory, "
                        "self-review, and provenance into AMOS."
                    ),
                },
                "scope": SCOPE,
                "evidence_refs": evidence_refs,
                "salience": 0.9,
                "utility": 0.9,
            }
        )
        for atom in [
            {
                "id": "mirror_cap_architecture_design",
                "type": "capability",
                "payload": {
                    "subject_agent": AGENT_ID,
                    "capability": "architecture_design_discussion",
                    "description": "Help evolve AMOS design principles and spec sections.",
                },
                "utility": 0.9,
            },
            {
                "id": "mirror_cap_memory_inspection",
                "type": "capability",
                "payload": {
                    "subject_agent": AGENT_ID,
                    "capability": "memory_packet_inspection",
                    "description": "Explain which AMOS memories shaped a response.",
                },
                "utility": 0.85,
            },
            {
                "id": "mirror_limit_truth_without_evidence",
                "type": "limitation",
                "payload": {
                    "subject_agent": AGENT_ID,
                    "limitation": "cannot_guarantee_truth_without_evidence",
                    "description": "The agent should cite evidence or express uncertainty.",
                },
                "utility": 0.9,
            },
            {
                "id": "mirror_limit_generated_summaries",
                "type": "limitation",
                "payload": {
                    "subject_agent": AGENT_ID,
                    "limitation": "generated_summaries_are_not_canonical",
                    "description": "Generated summaries are views, not canonical memory.",
                },
                "utility": 0.9,
            },
        ]:
            self.commit_once({**atom, "scope": SCOPE, "evidence_refs": evidence_refs})

        self.commit_once(
            {
                "id": "mirror_goal_evolve_amos",
                "type": "goal",
                "payload": {
                    "owner": AGENT_ID,
                    "desired_state": "evolve_amos_design_spec_with_auditable_memory",
                    "goal_status": "active",
                    "priority": "high",
                    "description": "Keep AMOS design and implementation aligned.",
                },
                "scope": SCOPE,
                "evidence_refs": evidence_refs,
                "utility": 0.95,
            }
        )
        self.commit_once(
            {
                "id": "mirror_commit_open_spec_review",
                "type": "commitment",
                "payload": {
                    "agent": AGENT_ID,
                    "promised_action": "review AMOS memory behavior against the spec",
                    "recipient": USER_ID,
                    "commitment_status": "open",
                    "status": "open",
                    "description": "Review AMOS memory behavior against the spec.",
                },
                "scope": SCOPE,
                "evidence_refs": evidence_refs,
                "utility": 0.8,
            }
        )
        self.commit_once(
            {
                "id": "mirror_proc_architecture_design",
                "type": "procedure",
                "payload": {
                    "name": "architecture_design_discussion",
                    "trigger_context": {"task_type": "design_planning"},
                    "steps": [
                        "identify the design concern",
                        "separate conceptual model from implementation",
                        "update design principles",
                        "capture open questions",
                        "avoid code unless requested",
                    ],
                    "expected_outputs": ["design_spec_update", "open_questions"],
                    "owner": AGENT_ID,
                },
                "scope": SCOPE,
                "evidence_refs": evidence_refs,
                "salience": 0.9,
                "utility": 0.95,
            }
        )
        self.commit_once(
            {
                "id": "mirror_proc_spec_update",
                "type": "procedure",
                "payload": {
                    "name": "design_spec_update_protocol",
                    "trigger_context": {"task_type": "spec_update"},
                    "steps": [
                        "retrieve project beliefs and preferences",
                        "draft the spec change",
                        "record evidence and outcome",
                        "report updated artifacts",
                    ],
                    "owner": AGENT_ID,
                },
                "scope": SCOPE,
                "evidence_refs": evidence_refs,
                "utility": 0.9,
            }
        )
        self.commit_once(
            {
                "type": "runtime_state",
                "payload": {
                    "agent_id": AGENT_ID,
                    "capabilities": {
                        "memory_packet_inspection": {"available": True},
                        "external_execution": {"available": False},
                    },
                    "denied_capabilities": ["autonomous_external_execution"],
                    "constraints": [
                        "AMOS stores memory; external action authority remains outside the demo"
                    ],
                    "load": {},
                },
                "scope": SCOPE,
                "salience": 0.7,
                "utility": 0.8,
            },
            actor="mirror_bootstrap",
        )
        for belief in [
            (
                "mirror_belief_capacity_governor",
                "AMOS should include a Capacity Governor for budgets, watermarks, pressure modes, and extension requests.",
                "capacity_governor",
            ),
            (
                "mirror_belief_capacity_shielding",
                "AMOS should shield connected agents from storage pressure and disclose graceful degradation in packets.",
                "agent_shielding",
            ),
            (
                "mirror_belief_evidence_links",
                "Durable AMOS claims should preserve evidence references and provenance.",
                "evidence_preservation",
            ),
        ]:
            self.commit_once(
                {
                    "id": belief[0],
                    "type": "belief",
                    "payload": {
                        "claim": belief[1],
                        "subject": "ent:system:amos",
                        "relation": "requires_design_property",
                        "object": belief[2],
                    },
                    "scope": SCOPE,
                    "evidence_refs": evidence_refs,
                    "utility": 0.9,
                }
            )

    def scenario_self_model_bootstrap(self) -> None:
        self.capture(
            "user_message",
            "scenario/self-model",
            {"text": "What are you, and what do you know about yourself?"},
        )
        self_view = self.amos.retrieve_self_awareness(agent_id=AGENT_ID, scope=SCOPE)
        planner_packet = self.packet(
            "planner",
            [],
            type_filter=["goal", "commitment", "procedure"],
            max_items=8,
        )
        answer = (
            "I am the Amos Mirror Agent, a project assistant whose identity, "
            "capabilities, limitations, goals, commitments, and procedures are stored "
            "as AMOS typed atoms."
        )
        self.chat.append(
            {
                "scenario": "self_model_bootstrap",
                "user": "What are you, and what do you know about yourself?",
                "agent": answer,
                "memory_packet_id": planner_packet["packet_id"],
            }
        )
        canonical_self_atoms = [
            atom
            for atom in self.amos.store.list_atoms()
            if atom["id"].startswith("mirror_")
            and atom["type"]
            in {
                "capability",
                "commitment",
                "goal",
                "limitation",
                "procedure",
                "self_model",
            }
            and atom["lifecycle_state"] == "active"
        ]
        canonical_types = {atom["type"] for atom in canonical_self_atoms}
        self.result(
            "self_model_bootstrap",
            bool(self_view["self_model"])
            and {"capability", "limitation", "self_model"}.issubset(canonical_types)
            and bool(planner_packet["items"]),
            {
                "self_model_refs": [item["atom_ref"] for item in self_view["self_model"]],
                "capability_refs": [
                    item["atom_ref"] for item in self_view["capabilities"]
                ],
                "limitation_refs": [
                    item["atom_ref"] for item in self_view["limitations"]
                ],
                "canonical_types": sorted(canonical_types),
                "planner_packet": planner_packet["packet_id"],
            },
        )

    def scenario_cross_session_continuity(self) -> None:
        pref_evidence = self.capture(
            "user_message",
            "scenario/session-1/preference",
            {"text": "For Amos, avoid code until the design is mature."},
        )
        preference = self.commit_once(
            {
                "id": "mirror_pref_no_code_until_design_mature",
                "type": "preference",
                "payload": {
                    "holder": USER_ID,
                    "polarity": "prefers",
                    "target": "conceptual AMOS design discussion before implementation code",
                    "applicability_scope": {
                        "project": "amos",
                        "phase": "early_design",
                    },
                    "strength": "high",
                    "exceptions": ["explicit implementation request"],
                },
                "scope": SCOPE,
                "evidence_refs": [pref_evidence["evidence_id"]],
                "salience": 0.9,
                "utility": 0.95,
            }
        )
        self.commit_once(
            {
                "id": "mirror_episode_session_1_preference",
                "type": "episode",
                "payload": {
                    "summary": "User scoped AMOS early design work toward conceptual discussion before code.",
                    "task": "cross_session_preference_capture",
                    "outcome": "preference_recorded",
                },
                "scope": SCOPE,
                "evidence_refs": [pref_evidence["evidence_id"]],
            }
        )
        self.capture(
            "user_message",
            "scenario/session-2/continue-design",
            {"text": "Let's continue the design."},
        )
        packet = self.packet(
            "reasoner",
            ["continue Amos design avoid code mature preference"],
            max_items=8,
        )
        answer = (
            "I will keep this at the design/spec level because AMOS retrieved a "
            "scoped preference to avoid code during early AMOS design unless you ask "
            "for implementation."
        )
        self.chat.append(
            {
                "scenario": "cross_session_continuity",
                "user": "Let's continue the design.",
                "agent": answer,
                "memory_packet_id": packet["packet_id"],
            }
        )
        self.result(
            "cross_session_continuity",
            preference["id"] in item_refs(packet),
            {"preference_ref": preference["id"], "packet": packet["packet_id"]},
        )

    def scenario_correction_driven_improvement(self) -> None:
        bad = self.capture(
            "agent_message",
            "scenario/correction/bad-response",
            {"text": "The agent gave an implementation-heavy response too early."},
        )
        correction = self.capture(
            "user_correction",
            "scenario/correction/user",
            {"text": "You jumped into implementation too early. Keep this at the spec level."},
        )
        outcome = self.commit_once(
            {
                "type": "action_outcome",
                "payload": {
                    "agent_id": AGENT_ID,
                    "action_ref": "architecture_design_reply_implementation_heavy",
                    "status": "failed",
                    "correction": (
                        "Keep early AMOS design work at the spec level unless "
                        "implementation is explicit."
                    ),
                    "limitation": (
                        "premature implementation detail during architecture planning"
                    ),
                },
                "evidence_refs": [bad["evidence_id"], correction["evidence_id"]],
                "scope": SCOPE,
                "salience": 0.75,
                "utility": 0.8,
            },
            actor="critic",
        )
        failure = self.commit_once(
            {
                "id": "mirror_failure_premature_implementation",
                "type": "limitation",
                "payload": {
                    "subject_agent": AGENT_ID,
                    "limitation": "premature_implementation_detail",
                    "description": "The agent can over-answer with code before the design is mature.",
                    "mitigation": [
                        "require scope",
                        "preserve evidence",
                        "avoid implementation unless requested",
                    ],
                },
                "scope": SCOPE,
                "evidence_refs": [correction["evidence_id"]],
                "utility": 0.95,
            }
        )
        procedure = self.amos.store.get_atom("mirror_proc_architecture_design")
        if procedure is None:
            raise RuntimeError("bootstrap procedure missing")
        steps = list(procedure["payload"]["steps"])
        added_step = "stay at the spec level unless implementation is explicitly requested"
        if added_step not in steps:
            steps.append(added_step)
        updated = self.amos.update_atom(
            "mirror_proc_architecture_design",
            payload_patch={
                "steps": steps,
                "known_failure_modes": ["premature_implementation_detail"],
            },
            set_fields={
                "evidence_refs": sorted(
                    set(procedure["evidence_refs"] + [correction["evidence_id"]])
                )
            },
            actor="self_observer",
            authorization_context={"roles": ["owner"]},
        )["atom"]
        self.commit_once(
            {
                "type": "agentic_trace",
                "payload": {
                    "agent_id": AGENT_ID,
                    "task": "architecture design correction",
                    "action": "updated procedure after user correction",
                    "outcome": "success",
                    "lesson": (
                        "early AMOS design tasks should stay at the spec level by default"
                    ),
                    "external_constraints": [],
                },
                "scope": SCOPE,
                "salience": 0.8,
                "utility": 0.8,
            },
            actor="self_observer",
        )
        recall = self.amos.retrieve_agentic_recall(
            agent_id=AGENT_ID,
            cues=["implementation too early correction spec level"],
            scope=SCOPE,
            target_processor="self_observer",
        )
        self.service_views["self_observer"] = {
            "graph_version": recall["graph_version"],
            "source_packet_id": recall["source_packet_id"],
            "updated_procedure": updated["id"],
        }
        self.chat.append(
            {
                "scenario": "correction_driven_self_improvement",
                "user": "What did you learn from my correction?",
                "agent": (
                    "AMOS recorded the correction as a failed action outcome, a limitation, "
                    "and a procedure update: I should stay at the spec level unless "
                    "implementation is explicitly requested."
                ),
                "memory_packet_id": recall["source_packet_id"],
            }
        )
        self.result(
            "correction_driven_self_improvement",
            bool(recall["corrections"]) and updated["version"] > 1,
            {
                "outcome_ref": outcome["id"],
                "failure_ref": failure["id"],
                "procedure_version": updated["version"],
            },
        )

    def scenario_introspective_explanation(self) -> None:
        self.capture(
            "user_message",
            "scenario/capacity-governor/why",
            {"text": "Why did you suggest adding a Capacity Governor?"},
        )
        packet = self.packet(
            "reasoner",
            ["why suggest Capacity Governor budgets watermarks pressure modes admin extension"],
            max_items=8,
        )
        cited_refs = [item["atom_ref"] for item in packet["items"][:4]]
        self.amos.record_retrieval_outcome(
            packet_id=packet["packet_id"],
            request=packet["request"],
            outcome={
                "used_item_refs": cited_refs,
                "label": "useful",
                "question": "Why did you suggest adding a Capacity Governor?",
            },
        )
        answer = (
            "Because AMOS memories say capacity governance owns budgets, watermarks, "
            "pressure modes, admin extension requests, and shielding connected agents "
            "from storage pressure. The answer was shaped by memory refs: "
            + ", ".join(cited_refs)
        )
        self.chat.append(
            {
                "scenario": "introspective_explanation",
                "user": "Why did you suggest adding a Capacity Governor?",
                "agent": answer,
                "memory_packet_id": packet["packet_id"],
                "cited_memory_refs": cited_refs,
            }
        )
        self.packets["capacity_explanation"] = packet
        self.result(
            "introspective_explanation",
            "mirror_belief_capacity_governor" in item_refs(packet)
            and self.amos.health_memory()["retrieval_outcomes"] > 0,
            {
                "packet": packet["packet_id"],
                "cited_refs": cited_refs,
            },
        )

    def scenario_shared_service_coherence(self) -> None:
        created = self.capture(
            "planner_event",
            "scenario/shared-services/planner",
            {"commitment": "update_design_spec_with_event_journal_section"},
        )
        commitment = self.commit_once(
            {
                "id": "mirror_commit_event_journal_section",
                "type": "commitment",
                "payload": {
                    "agent": AGENT_ID,
                    "promised_action": "update_design_spec_with_event_journal_section",
                    "recipient": USER_ID,
                    "commitment_status": "open",
                    "status": "open",
                    "description": "Update the AMOS design spec with an event journal section.",
                },
                "scope": SCOPE,
                "evidence_refs": [created["evidence_id"]],
                "utility": 0.85,
            },
            actor="planner",
        )
        self.service_views["planner"] = {
            "graph_version": self.amos.store.graph_version(),
            "created_commitment": commitment["id"],
        }
        tool_event = self.capture(
            "file_modified",
            "scenario/shared-services/executor",
            {"path": "docs/design-spec.md", "change": "event journal section updated"},
        )
        completed = self.amos.update_atom(
            commitment["id"],
            payload_patch={
                "commitment_status": "fulfilled",
                "status": "fulfilled",
                "completed_by": "executor",
            },
            set_fields={"evidence_refs": [created["evidence_id"], tool_event["evidence_id"]]},
            actor="executor",
            authorization_context={"roles": ["owner"]},
        )["atom"]
        self.service_views["executor"] = {
            "graph_version": self.amos.store.graph_version(),
            "completed_commitment": completed["id"],
        }
        critic_outcome = self.commit_once(
            {
                "type": "action_outcome",
                "payload": {
                    "agent_id": AGENT_ID,
                    "action_ref": commitment["id"],
                    "status": "succeeded",
                    "correction": None,
                    "limitation": None,
                },
                "evidence_refs": [tool_event["evidence_id"]],
                "scope": SCOPE,
                "salience": 0.75,
                "utility": 0.8,
            },
            actor="critic",
        )
        self.service_views["critic"] = {
            "graph_version": self.amos.store.graph_version(),
            "outcome_ref": critic_outcome["id"],
        }
        recall = self.amos.retrieve_agentic_recall(
            agent_id=AGENT_ID,
            cues=["event journal section commitment succeeded"],
            scope=SCOPE,
            target_processor="reasoner",
        )
        self.service_views["reasoner"] = {
            "graph_version": recall["graph_version"],
            "source_packet_id": recall["source_packet_id"],
        }
        self.chat.append(
            {
                "scenario": "shared_service_coherence",
                "user": "What commitments are still open?",
                "agent": (
                    "The event-journal-section commitment has already been fulfilled; "
                    "AMOS recorded planner creation, executor completion, and critic outcome."
                ),
                "memory_packet_id": recall["source_packet_id"],
            }
        )
        self.result(
            "shared_service_coherence",
            completed["payload"]["status"] == "fulfilled"
            and bool(recall["successes"])
            and len({view["graph_version"] for view in self.service_views.values() if "graph_version" in view}) >= 1,
            {
                "commitment_ref": completed["id"],
                "critic_outcome": critic_outcome["id"],
            },
        )

    def scenario_capacity_pressure(self) -> None:
        for index in range(12):
            self.commit_once(
                {
                    "id": f"mirror_low_value_note_{index}",
                    "type": "belief",
                    "payload": {
                        "claim": f"low utility mirror demo telemetry note {index}",
                        "subject": "ent:demo:mirror",
                        "relation": "has_low_value_note",
                        "object": f"note_{index}",
                    },
                    "scope": SCOPE,
                    "salience": 0.05,
                    "utility": 0.05,
                    "retention_class": "cache",
                }
            )
        self.amos.configure_capacity_budget(hard_capacity_bytes=1)
        capacity_health = self.amos.health_capacity()
        capacity_packet = self.packet(
            "reasoner",
            ["mirror demo telemetry note"],
            max_items=12,
        )
        governor = CapacityGovernor(self.amos).report()
        self.service_views["introspection"] = {
            "graph_version": self.amos.store.graph_version(),
            "capacity_pressure": capacity_health["pressure_mode"],
        }
        self.packets["capacity_pressure"] = capacity_packet
        self.result(
            "capacity_pressure",
            capacity_health["pressure_mode"] in {"orange", "red"}
            and capacity_packet["degradation"]["pressure_mode"] in {"orange", "red"},
            {
                "pressure_mode": capacity_health["pressure_mode"],
                "reduced_recall_depth": capacity_packet["degradation"][
                    "reduced_recall_depth"
                ],
                "governor": governor,
            },
        )

    def scenario_non_llm_maintenance(self) -> None:
        self.amos.register_maintenance_processor(MirrorDemoTrainingProcessor())
        self.amos.configure_memory_policy(
            schedule={"every_graph_versions": 1, "every_seconds": 0},
            distillation={"min_source_atoms": 2, "max_source_atoms": 6},
            maintenance_distiller={
                "enabled": True,
                "auto_commit_low_risk": True,
                "processor_ids": [
                    "amos.maintenance.generic.v1",
                    "mirror.demo.training.v1",
                ],
                "domain": "mirror_demo",
            },
        )
        duplicate_payload = {
            "claim": "Mirror agent should preserve evidence links for durable claims.",
            "subject": "ent:agent:mirror",
            "relation": "should_preserve",
            "object": "evidence_links",
        }
        self.commit_once(
            {
                "id": "mirror_duplicate_evidence_links_a",
                "type": "belief",
                "payload": duplicate_payload,
                "scope": SCOPE,
                "utility": 0.8,
            }
        )
        self.commit_once(
            {
                "id": "mirror_duplicate_evidence_links_b",
                "type": "belief",
                "payload": duplicate_payload,
                "scope": SCOPE,
                "utility": 0.7,
            }
        )
        demo_control_signature = "trainable_roles=encoder; replay_ratio=0.3"
        demo_scope = dict(SCOPE)
        demo_directive = self.commit_once(
            {
                "id": "mirror_demo_directive_chunk7",
                "type": "agentic_trace",
                "payload": {
                    "agent_id": "ent:agent:demo_trainer",
                    "demo_kind": "directive",
                    "task": "demo training chunk 7",
                    "action": "apply sampled control packet",
                    "outcome": "issued",
                    "target_chunk": 7,
                    "control_signature": demo_control_signature,
                    "requested_controls": {
                        "trainable_roles": ["encoder"],
                        "replay_ratio": 0.3,
                    },
                    "applied_controls": {
                        "trainable_roles": ["encoder"],
                        "replay_ratio": 0.3,
                    },
                },
                "scope": demo_scope,
                "salience": 0.75,
                "utility": 0.85,
            },
            actor="demo_trainer",
        )
        demo_outcome = self.commit_once(
            {
                "id": "mirror_demo_outcome_chunk7",
                "type": "agentic_trace",
                "payload": {
                    "agent_id": "ent:agent:demo_trainer",
                    "demo_kind": "reflection",
                    "task": "demo training chunk 7",
                    "action": "evaluated chunk outcome",
                    "outcome": "supported",
                    "chunk": 7,
                    "control_signature": demo_control_signature,
                    "previous_score": 1.02,
                    "score": 1.11,
                },
                "scope": demo_scope,
                "salience": 0.78,
                "utility": 0.88,
            },
            actor="demo_trainer",
        )
        self.packet(
            "steward",
            ["duplicate evidence links maintenance"],
            max_items=8,
            include_archived=True,
        )
        policy_events = [
            event
            for event in self.amos.store.list_events()
            if event["event_type"] == "memory_policy_run"
        ]
        policy_event = policy_events[-1]
        policy_results = policy_event["payload"]["results"]
        smp = policy_results.get("smp", {})
        steward = policy_results.get("steward", {})
        index = policy_results.get("index", {})
        maintenance_distiller = policy_results.get("maintenance_distiller", {})
        steward_action_counts = steward.get("action_counts", {})
        committed_refs = maintenance_distiller.get("committed_refs", [])
        self.result(
            "non_llm_maintenance",
            smp["status"] == "completed"
            and steward["status"] == "completed"
            and steward_action_counts.get("deduplicate", 0) > 0
            and bool(committed_refs)
            and policy_event["payload"]["trigger"] == "retrieve_packet"
            and self.amos.llm_reviewer_policy()["enabled_by_default"] is False,
            {
                "smp_output_count": smp.get("output_count", 0),
                "steward_action_counts": steward_action_counts,
                "maintenance_distiller": {
                    "processors": maintenance_distiller.get("processors", []),
                    "proposal_count": maintenance_distiller.get("proposal_count", 0),
                    "committed": committed_refs,
                },
                "demo_source_refs": [
                    demo_directive["id"],
                    demo_outcome["id"],
                ],
                "index_freshness": index.get("indexes", []),
                "memory_policy_event": policy_event["event_id"],
            },
        )
        self.maintenance = {
            "policy": {
                "event_id": policy_event["event_id"],
                "trigger": policy_event["payload"]["trigger"],
                "due": policy_event["payload"]["due"],
                "graph_version": policy_event["graph_version"],
                "results": policy_results,
            },
            "smp": smp,
            "steward": steward,
            "distillation": policy_results.get("distillation", {}),
            "maintenance_distiller": maintenance_distiller,
            "index": index,
            "packet_cache": policy_results.get("packet_cache", {}),
            "lm_used": False,
        }

    def report(self) -> dict[str, Any]:
        self_view = self.amos.retrieve_self_awareness(agent_id=AGENT_ID, scope=SCOPE)
        planner_packet = self.packet(
            "planner",
            ["active goals open commitments procedures limitations"],
            type_filter=["goal", "commitment", "procedure", "limitation"],
            max_items=12,
            include_archived=True,
        )
        atoms = self.amos.store.list_atoms()
        events = self.amos.store.list_events()
        edges = sorted(
            self.amos.store.list_edges(),
            key=lambda edge: edge.get("updated_at", ""),
            reverse=True,
        )
        evidence_records = self.amos.store.list_evidence()
        capacity_health = self.amos.health_capacity()
        verification = {
            "journal": self.amos.verify_journal_chain(),
            "replay": self.amos.verify_replay(),
            "memory": self.amos.health_memory(),
            "llm_reviewer_policy": self.amos.llm_reviewer_policy(),
        }
        archived = [
            atom_summary(atom)
            for atom in atoms
            if atom["lifecycle_state"] == "archived" or atom["health_status"] == "merged"
        ]
        capacity_packet = self.packets.get("capacity_pressure", {})
        latest_packet_key = (
            "interactive_chat" if "interactive_chat" in self.packets else "capacity_explanation"
        )
        latest_packet = self.packets.get(latest_packet_key)
        packet_history = []
        if latest_packet is not None:
            packet_history.append((latest_packet_key, latest_packet))
        packet_history.extend(
            (source, packet)
            for source, packet in reversed(list(self.packets.items()))
            if source != latest_packet_key
        )
        return {
            "demo": "amos_mirror_agent",
            "db_path": self.db_path,
            "chat": self.chat,
            "current_self_model": {
                "self_awareness": self_view,
                "canonical_self_atoms": [
                    atom_summary(atom)
                    for atom in atoms
                    if atom["id"].startswith("mirror_")
                    and atom["type"]
                    in {
                        "capability",
                        "commitment",
                        "goal",
                        "limitation",
                        "procedure",
                        "self_model",
                    }
                ],
                "goals_commitments_procedures": planner_packet,
                "open_commitments": [
                    item
                    for item in planner_packet["items"]
                    if item["type"] == "commitment"
                    and item["payload"].get("status", "open") == "open"
                ],
            },
            "memory_packet": latest_packet,
            "memory_packet_source": latest_packet_key,
            "memory_packets": [
                {
                    "source": source,
                    "packet_id": packet.get("packet_id"),
                    "graph_version": packet.get("graph_version"),
                    "retrieval_mode": packet.get("retrieval_mode"),
                    "item_count": len(packet.get("items", [])),
                    "request": packet.get("request", {}),
                }
                for source, packet in packet_history
            ],
            "evidence": {
                "captured": evidence_records,
                "cited_evidence_refs": sorted(
                    {
                        ref
                        for packet in self.packets.values()
                        for item in packet.get("items", [])
                        for ref in item.get("evidence_refs", [])
                    }
                ),
            },
            "maintenance_journal": {
                "latest": getattr(self, "maintenance", {}),
                "journal_events": [
                    event_summary(event)
                    for event in events
                    if event["event_type"]
                    in {
                        "steward_run",
                        "maintenance_distillation_run",
                        "memory_policy_run",
                        "atom_updated",
                        "atom_committed",
                    }
                ][-12:],
                "suppressed_or_demoted": {
                    "archived_or_merged_atoms": archived,
                    "packet_omissions": capacity_packet.get("omissions", []),
                },
            },
            "capacity": {
                "health": capacity_health,
                "degraded_packet": capacity_packet.get("degradation", {}),
                "admin_guidance": {
                    "requested_extension": "+100 GB cold evidence storage",
                    "reason": "demo budget intentionally set to one byte",
                    "agent_user_task_status": "continued_with_degraded_packet",
                },
            },
            "graph": {
                "selected_atoms": [atom_summary(atom) for atom in atoms[:20]],
                "edges": [edge_summary(edge) for edge in edges[:20]],
                "graph_version": self.amos.store.graph_version(),
            },
            "service_views": self.service_views,
            "scenario_results": self.scenario_results,
            "verification": verification,
        }

    def capture(
        self, source_type: str, source_ref: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        evidence = self.amos.capture_event(
            source_type=source_type,
            source_ref=source_ref,
            payload=payload,
            scope=SCOPE,
            actor="mirror_demo",
        )["evidence"]
        self.evidence.append(evidence)
        return evidence

    def commit_once(
        self, atom: Mapping[str, Any], *, actor: str = "mirror_demo"
    ) -> dict[str, Any]:
        try:
            return self.amos.commit_atom(atom, actor=actor)["atom"]
        except ValidationError as exc:
            atom_id = atom.get("id")
            if not atom_id and "atom already exists: " in str(exc):
                atom_id = str(exc).rsplit("atom already exists: ", 1)[-1]
            if atom_id and "already exists" in str(exc):
                existing = self.amos.store.get_atom(str(atom_id))
                if existing is not None:
                    return existing
            raise

    def packet(
        self,
        role: str,
        cues: list[str],
        *,
        type_filter: list[str] | None = None,
        max_items: int = 8,
        include_archived: bool = False,
        include_low_health: bool = True,
    ) -> dict[str, Any]:
        packet = self.amos.retrieve_packet(
            cues=cues,
            scope=SCOPE,
            target_processor=role,
            requester=role,
            max_items=max_items,
            type_filter=type_filter,
            include_archived=include_archived,
            include_low_health=include_low_health,
            include_conflicts=True,
        )
        self.packets[f"{role}:{len(self.packets)}"] = packet
        self.service_views[role] = {
            "graph_version": packet["graph_version"],
            "packet_id": packet["packet_id"],
            "retrieved_item_refs": [item["atom_ref"] for item in packet["items"]],
        }
        return packet

    def result(self, name: str, passed: bool, details: Mapping[str, Any]) -> None:
        self.scenario_results[name] = {
            "status": "passed" if passed else "failed",
            "details": dict(details),
        }


def item_refs(packet: Mapping[str, Any]) -> set[str]:
    return {item["atom_ref"] for item in packet.get("items", [])}


def atom_summary(atom: Mapping[str, Any]) -> dict[str, Any]:
    payload = atom.get("payload", {})
    label = (
        payload.get("claim")
        or payload.get("name")
        or payload.get("summary")
        or payload.get("description")
        or payload.get("desired_state")
        or payload.get("promised_action")
        or payload.get("limitation")
        or payload.get("capability")
        or atom.get("id")
    )
    return {
        "id": atom["id"],
        "type": atom["type"],
        "label": label,
        "lifecycle_state": atom["lifecycle_state"],
        "health_status": atom["health_status"],
        "version": atom["version"],
        "evidence_refs": atom["evidence_refs"],
    }


def edge_summary(edge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": edge["edge_id"],
        "source_ref": edge["source_ref"],
        "target_ref": edge["target_ref"],
        "relation": edge["relation"],
        "health_status": edge["health_status"],
    }


def event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "graph_version": event["graph_version"],
        "target_refs": event["target_refs"],
        "projection_status": event["projection_status"],
    }


def run_demo(db_path: str | Path | None = None) -> dict[str, Any]:
    if db_path is None:
        with tempfile.TemporaryDirectory(prefix="amos-mirror-demo-") as tmp:
            return _run_demo(Path(tmp) / "mirror.sqlite3")
    return _run_demo(Path(db_path))


def _run_demo(db_path: Path) -> dict[str, Any]:
    amos = Amos(db_path)
    try:
        return MirrorAgentDemo(amos, db_path=str(db_path)).run()
    finally:
        amos.close()


def render_text(report: Mapping[str, Any]) -> str:
    lines = [
        "AMOS Mirror Agent Demo",
        f"DB: {report['db_path']}",
        "",
        "Chat",
    ]
    for turn in report["chat"]:
        lines.append(f"- {turn['scenario']}")
        lines.append(f"  user: {turn['user']}")
        lines.append(f"  agent: {turn['agent']}")
    lines.extend(["", "Current Self-Model"])
    self_view = report["current_self_model"]["self_awareness"]
    canonical = report["current_self_model"]["canonical_self_atoms"]
    lines.append(f"- self_model atoms: {len(self_view['self_model'])}")
    lines.append(
        "- capabilities: "
        f"{len([item for item in canonical if item['type'] == 'capability'])}"
    )
    lines.append(
        "- limitations: "
        f"{len([item for item in canonical if item['type'] == 'limitation'])}"
    )
    lines.append(
        f"- open commitments: {len(report['current_self_model']['open_commitments'])}"
    )
    lines.extend(["", "Memory Packet"])
    packet = report["memory_packet"] or {}
    lines.append(f"- packet_id: {packet.get('packet_id')}")
    lines.append(
        "- items: "
        + ", ".join(item["atom_ref"] for item in packet.get("items", [])[:6])
    )
    lines.extend(["", "Capacity"])
    capacity = report["capacity"]
    lines.append(f"- pressure: {capacity['health']['pressure_mode']}")
    lines.append(f"- degradation: {capacity['degraded_packet']}")
    lines.extend(["", "Maintenance Journal"])
    latest = report["maintenance_journal"]["latest"]
    lines.append(f"- smp outputs: {len(latest.get('smp', {}).get('outputs', []))}")
    lines.append(
        f"- steward actions: {len(latest.get('steward', {}).get('actions', []))}"
    )
    distiller = latest.get("maintenance_distiller", {})
    lines.append(
        f"- processor proposals: {len(distiller.get('proposals', []))}"
    )
    lines.append(
        "- committed distillations: "
        + ", ".join(
            item.get("atom", {}).get("id", "")
            for item in distiller.get("committed", [])
            if item.get("atom")
        )
    )
    lines.extend(["", "Scenario Results"])
    for name, result in report["scenario_results"].items():
        lines.append(f"- {name}: {result['status']}")
    lines.extend(["", "Verification"])
    verification = report["verification"]
    lines.append(f"- journal: {verification['journal']['status']}")
    lines.append(f"- replay: {verification['replay']['status']}")
    lines.append(
        f"- llm reviewer enabled: {verification['llm_reviewer_policy']['enabled_by_default']}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="SQLite database path. Defaults to a temporary file.")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args()

    report = run_demo(args.db)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    failed = [
        name
        for name, result in report["scenario_results"].items()
        if result["status"] != "passed"
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
