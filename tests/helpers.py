from __future__ import annotations

from amos import MaintenanceProposal, SemanticFacet


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

