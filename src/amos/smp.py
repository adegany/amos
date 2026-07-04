"""Non-generative Semantic Maintenance Processor for AMOS v1."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .errors import ValidationError
from .schemas import (
    SCHEMA_VERSION,
    canonical_json,
    confidence_score,
    digest,
    normalize_atom,
    stable_id,
)

PROCESSOR_ID = "amos.smp.deterministic"
PROCESSOR_VERSION = "amos.smp.deterministic.v1"

SMP_REASON_CODES = {
    "capacity_pressure",
    "confounding_after_correction",
    "contradiction_candidate",
    "low_retrieval_utility",
    "near_duplicate",
    "policy_required",
    "privacy_risk",
    "scope_too_broad",
    "scope_too_narrow",
    "shape_invalid",
    "stale_by_age",
    "stale_by_external_change",
    "valid_shape",
}

HIGH_RISK_ACTIONS = {
    "change_access_policy",
    "change_retention_class",
    "delete",
    "destructive_merge",
    "mark_active_belief_contradicted",
    "promote_procedure_active",
    "tombstone",
}


class SemanticMaintenanceProcessor:
    """Bounded deterministic SMP implementation.

    The processor intentionally uses local lexical/vector heuristics instead of
    generation. Its outputs are recommendations with audit fields; callers must
    still route mutations through AMOS policy and journal gates.
    """

    def __init__(self, *, dimensions: int = 64):
        self.dimensions = dimensions
        self.processor_id = PROCESSOR_ID
        self.processor_version = PROCESSOR_VERSION

    def encode(self, atom_or_text_span: Mapping[str, Any] | str) -> list[float]:
        if isinstance(atom_or_text_span, Mapping):
            index_refs = atom_or_text_span.get("index_refs")
            if isinstance(index_refs, Mapping):
                search_index = index_refs.get("amos.v1.search")
                if isinstance(search_index, Mapping):
                    vector = search_index.get("vector")
                    if isinstance(vector, list):
                        try:
                            return [float(value) for value in vector]
                        except (TypeError, ValueError):
                            pass
        text = (
            atom_or_text_span
            if isinstance(atom_or_text_span, str)
            else atom_text(atom_or_text_span)
        )
        vector = [0.0] * self.dimensions
        for token, count in Counter(tokens(text)).items():
            slot = int(digest(token), 16) % self.dimensions
            vector[slot] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 8) for value in vector]

    def classify(self, memory_candidate: Mapping[str, Any] | str) -> dict[str, Any]:
        text = memory_candidate if isinstance(memory_candidate, str) else atom_text(memory_candidate)
        lower = text.lower()
        labels: dict[str, float] = {}
        if any(word in lower for word in ["prefer", "preference", "avoid", "likes"]):
            labels["preference"] = 0.74
        if any(word in lower for word in ["should", "claim", "belief", "is true"]):
            labels["belief"] = max(labels.get("belief", 0.0), 0.68)
        if any(word in lower for word in ["step", "procedure", "fallback", "rollback"]):
            labels["procedure"] = 0.72
        if any(word in lower for word in ["failed", "blocked", "success", "correction"]):
            labels["agentic_trace"] = 0.7
        if not labels:
            labels["semantic"] = 0.5
        top_label, top_score = max(labels.items(), key=lambda item: item[1])
        return self._output(
            input_refs=[input_ref(memory_candidate)],
            output_type="classification",
            confidence=top_score,
            reason_code="policy_required",
            recommended_action={"type": "label_candidate", "label": top_label},
            risk_level="low",
            details={"labels": labels},
        )

    def compare(
        self, atom_a: Mapping[str, Any], atom_b: Mapping[str, Any]
    ) -> dict[str, Any]:
        similarity = cosine(self.encode(atom_a), self.encode(atom_b))
        relation_guess = "rel:similar_to"
        reason_code = "policy_required"
        recommended_action: dict[str, Any] = {
            "type": "add_candidate_link",
            "relation": "rel:similar_to",
        }
        risk = "low"
        if same_payload_signature(atom_a, atom_b):
            relation_guess = "rel:similar_to"
            reason_code = "near_duplicate"
            recommended_action = {
                "type": "propose_duplicate_link",
                "relation": "rel:similar_to",
            }
            risk = "medium"
            similarity = max(similarity, 0.98)
        elif contradiction_signature(atom_a) and contradiction_signature(atom_b):
            key_a, value_a = contradiction_signature(atom_a)  # type: ignore[misc]
            key_b, value_b = contradiction_signature(atom_b)  # type: ignore[misc]
            if key_a == key_b and value_a != value_b:
                relation_guess = "rel:contradicts"
                reason_code = "contradiction_candidate"
                recommended_action = {
                    "type": "propose_conflict_link",
                    "relation": "rel:contradicts",
                }
                risk = "high"
                similarity = max(similarity, 0.85)
        return self._output(
            input_refs=[input_ref(atom_a), input_ref(atom_b)],
            output_type="comparison",
            confidence=round(similarity, 4),
            reason_code=reason_code,
            recommended_action={
                **recommended_action,
                "similarity": round(similarity, 4),
                "relation_guess": relation_guess,
            },
            risk_level=risk,
        )

    def cluster(self, atom_set: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for atom in atom_set:
            groups[cluster_key(atom)].append(atom)
        outputs = []
        for key, atoms in groups.items():
            if len(atoms) < 2:
                continue
            outputs.append(
                self._output(
                    input_refs=[input_ref(atom) for atom in atoms],
                    output_type="cluster",
                    confidence=0.8,
                    reason_code="near_duplicate",
                    recommended_action={
                        "type": "review_cluster",
                        "cluster_key": key,
                        "members": [input_ref(atom) for atom in atoms],
                    },
                    risk_level="medium",
                )
            )
        return outputs

    def validate_shape(self, atom_or_edge: Mapping[str, Any]) -> dict[str, Any]:
        input_refs = [input_ref(atom_or_edge)]
        try:
            atom = normalize_atom(atom_or_edge)
        except ValidationError as exc:
            return self._output(
                input_refs=input_refs,
                output_type="shape_validation",
                confidence=1.0,
                reason_code="shape_invalid",
                recommended_action={"type": "reject", "error": str(exc)},
                risk_level="low",
            )
        problems = []
        payload = atom["payload"]
        if atom["type"] in {"belief", "preference", "procedure"} and not atom["evidence_refs"]:
            problems.append("missing_evidence_refs")
        if atom["type"] == "preference":
            for field in ["holder", "polarity", "target", "applicability_scope", "strength"]:
                if field not in payload:
                    problems.append(f"missing_payload_field:{field}")
        if atom["type"] == "procedure":
            if "steps" not in payload:
                problems.append("missing_payload_field:steps")
            if "trigger_context" not in payload:
                problems.append("missing_payload_field:trigger_context")
        if problems:
            return self._output(
                input_refs=input_refs,
                output_type="shape_validation",
                confidence=0.95,
                reason_code="shape_invalid",
                recommended_action={
                    "type": "mark_proposed_underspecified",
                    "problems": problems,
                },
                risk_level="low",
            )
        return self._output(
            input_refs=input_refs,
            output_type="shape_validation",
            confidence=1.0,
            reason_code="valid_shape",
            recommended_action={"type": "accept_shape"},
            risk_level="low",
        )

    def detect_conflicts(
        self, atom_set: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        outputs = []
        by_key: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for atom in atom_set:
            signature = contradiction_signature(atom)
            if signature is None:
                continue
            key, value = signature
            by_key[key][value] = atom
        for values in by_key.values():
            if len(values) < 2:
                continue
            atoms = list(values.values())
            outputs.append(
                self._output(
                    input_refs=[input_ref(atom) for atom in atoms],
                    output_type="conflict_candidates",
                    confidence=0.9,
                    reason_code="contradiction_candidate",
                    recommended_action={
                        "type": "review_conflict",
                        "relation": "rel:contradicts",
                    },
                    risk_level="high",
                )
            )
        return outputs

    def score_utility(
        self,
        atom: Mapping[str, Any],
        telemetry: Mapping[str, Any] | None = None,
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        telemetry = dict(telemetry or {})
        reuse = float(telemetry.get("successful_retrieval_count", 0))
        corrections = float(telemetry.get("correction_after_use_count", 0))
        base = float(atom.get("utility", 0.5))
        score = clamp01(base + min(reuse, 10.0) * 0.03 - min(corrections, 10.0) * 0.05)
        reason = "low_retrieval_utility" if score < 0.25 else "policy_required"
        action = "mark_low_utility" if score < 0.25 else "update_utility_counter"
        return self._output(
            input_refs=[input_ref(atom)],
            output_type="utility_score",
            confidence=0.7,
            reason_code=reason,
            recommended_action={
                "type": action,
                "utility_score": round(score, 4),
                "scope": dict(scope or {}),
            },
            risk_level="low",
        )

    def propose_links(
        self, atom: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        outputs = []
        for candidate in candidates:
            comparison = self.compare(atom, candidate)
            similarity = comparison["recommended_action"].get("similarity", 0.0)
            if similarity >= 0.65:
                outputs.append(
                    self._output(
                        input_refs=[input_ref(atom), input_ref(candidate)],
                        output_type="edge_candidate",
                        confidence=similarity,
                        reason_code=comparison["reason_code"],
                        recommended_action={
                            "type": "add_candidate_link",
                            "relation": comparison["recommended_action"][
                                "relation_guess"
                            ],
                        },
                        risk_level="medium"
                        if comparison["reason_code"] == "near_duplicate"
                        else comparison["risk_level"],
                    )
                )
        return outputs

    def propose_health(
        self,
        atom: Mapping[str, Any],
        telemetry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        telemetry = dict(telemetry or {})
        correction_count = int(telemetry.get("correction_after_use_count", 0))
        if correction_count > 0:
            return self._output(
                input_refs=[input_ref(atom)],
                output_type="health_status_candidate",
                confidence=min(0.95, 0.6 + correction_count * 0.1),
                reason_code="confounding_after_correction",
                recommended_action={"type": "mark_health", "health_status": "confounding"},
                risk_level="medium",
            )
        if confidence_score(atom.get("confidence")) < 0.25:
            return self._output(
                input_refs=[input_ref(atom)],
                output_type="health_status_candidate",
                confidence=0.7,
                reason_code="low_retrieval_utility",
                recommended_action={"type": "mark_health", "health_status": "low_utility"},
                risk_level="low",
            )
        return self._output(
            input_refs=[input_ref(atom)],
            output_type="health_status_candidate",
            confidence=0.8,
            reason_code="valid_shape",
            recommended_action={"type": "mark_health", "health_status": "healthy"},
            risk_level="low",
        )

    def _output(
        self,
        *,
        input_refs: Sequence[str],
        output_type: str,
        confidence: float,
        reason_code: str,
        recommended_action: Mapping[str, Any],
        risk_level: str,
        evidence_refs: Sequence[str] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if reason_code not in SMP_REASON_CODES:
            reason_code = "policy_required"
        if recommended_action.get("type") in HIGH_RISK_ACTIONS:
            risk_level = "high"
        body = {
            "processor_id": self.processor_id,
            "processor_version": self.processor_version,
            "input_refs": list(input_refs),
            "output_type": output_type,
            "confidence": round(clamp01(confidence), 4),
            "reason_code": reason_code,
            "evidence_refs": list(evidence_refs or []),
            "recommended_action": dict(recommended_action),
            "risk_level": risk_level,
            "schema_version": SCHEMA_VERSION,
        }
        if details:
            body["details"] = dict(details)
        body["output_id"] = stable_id("smp", body)
        return body


def atom_text(atom_or_text: Mapping[str, Any] | str) -> str:
    if isinstance(atom_or_text, str):
        return atom_or_text
    return " ".join(
        [
            str(atom_or_text.get("id", "")),
            str(atom_or_text.get("type", "")),
            canonical_json(atom_or_text.get("payload", atom_or_text)),
        ]
    )


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def cosine(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if not vector_a or not vector_b:
        return 0.0
    numerator = sum(a * b for a, b in zip(vector_a, vector_b))
    denom_a = math.sqrt(sum(a * a for a in vector_a))
    denom_b = math.sqrt(sum(b * b for b in vector_b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return round(clamp01(numerator / (denom_a * denom_b)), 4)


def input_ref(value: Mapping[str, Any] | str) -> str:
    if isinstance(value, str):
        return stable_id("text", value)
    return str(value.get("id") or value.get("edge_id") or stable_id("input", value))


def same_payload_signature(atom_a: Mapping[str, Any], atom_b: Mapping[str, Any]) -> bool:
    return (
        atom_a.get("type") == atom_b.get("type")
        and atom_a.get("scope", {}) == atom_b.get("scope", {})
        and atom_a.get("payload") == atom_b.get("payload")
    )


def contradiction_signature(
    atom: Mapping[str, Any]
) -> tuple[tuple[Any, ...], str] | None:
    payload = atom.get("payload", {})
    if not isinstance(payload, Mapping):
        return None
    if {"subject", "predicate", "value"}.issubset(payload):
        key = (
            atom.get("type"),
            canonical_json(atom.get("scope", {})),
            payload["subject"],
            payload["predicate"],
        )
        return key, canonical_json(payload["value"])
    if {"key", "value"}.issubset(payload):
        key = (atom.get("type"), canonical_json(atom.get("scope", {})), payload["key"])
        return key, canonical_json(payload["value"])
    return None


def cluster_key(atom: Mapping[str, Any]) -> str:
    payload = atom.get("payload", {})
    if isinstance(payload, Mapping):
        if "subject" in payload and "predicate" in payload:
            return canonical_json(
                {
                    "type": atom.get("type"),
                    "scope": atom.get("scope", {}),
                    "subject": payload["subject"],
                    "predicate": payload["predicate"],
                }
            )
        if "name" in payload:
            return canonical_json(
                {
                    "type": atom.get("type"),
                    "scope": atom.get("scope", {}),
                    "name": payload["name"],
                }
            )
    return digest({"type": atom.get("type"), "tokens": tokens(atom_text(atom))[:8]})


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
