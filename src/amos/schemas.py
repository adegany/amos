"""Small JSON-compatible schema helpers for AMOS v1.

The design spec names JSON Schema 2020-12 as the authoritative wire contract.
This module keeps the implementation dependency-free by enforcing the same
core invariants in Python and exporting schema files for external validators.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from .errors import ValidationError
from .ontology import SEED_RELATION_IDS

SCHEMA_VERSION = "amos.v1"

ATOM_TYPES = {
    "action_outcome",
    "agentic_trace",
    "belief",
    "capability",
    "commitment",
    "episode",
    "goal",
    "limitation",
    "policy",
    "preference",
    "procedure",
    "runtime_state",
    "self_assessment",
    "self_model",
    "self_narrative",
    "semantic",
}

EDGE_RELATIONS = SEED_RELATION_IDS
LIFECYCLE_STATES = {
    "active",
    "archived",
    "deleted",
    "proposed",
    "superseded",
    "tombstoned",
}

HEALTH_STATES = {
    "confounding",
    "contradicted",
    "deleted",
    "healthy",
    "low_utility",
    "merged",
    "orphaned",
    "stale",
}

ENVELOPE_FIELDS = {
    "access_policy",
    "confidence",
    "created_at",
    "decay_policy",
    "evidence_refs",
    "health_status",
    "id",
    "index_refs",
    "last_accessed",
    "layer",
    "lifecycle_state",
    "observed_at",
    "payload",
    "retention_class",
    "revision_history",
    "salience",
    "schema_version",
    "scope",
    "supersedes",
    "type",
    "updated_at",
    "utility",
    "version",
}

PAYLOAD_FORBIDDEN_FIELDS = ENVELOPE_FIELDS - {"payload"}

CONFIDENCE_SCORE_BY_LEVEL = {
    "low": 0.2,
    "low-medium": 0.35,
    "medium": 0.5,
    "medium-high": 0.75,
    "high": 0.9,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    ensure_jsonable(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{digest(value)[:20]}"


def ensure_jsonable(value: Any) -> None:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"value is not JSON-compatible: {exc}") from exc


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be an object")
    return dict(value)


def _expect_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    return list(value)


def normalize_confidence(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {"level": "medium", "score": CONFIDENCE_SCORE_BY_LEVEL["medium"]}
    data = _expect_mapping(value, "confidence")
    level = str(data.get("level", "medium"))
    if level not in CONFIDENCE_SCORE_BY_LEVEL:
        raise ValidationError(f"unsupported confidence level: {level}")
    score = data.get("score", CONFIDENCE_SCORE_BY_LEVEL[level])
    if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
        raise ValidationError("confidence.score must be a number between 0 and 1")
    data["level"] = level
    data["score"] = float(score)
    return data


def normalize_scope(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    scope = _expect_mapping(value, "scope")
    for key, item in scope.items():
        if not isinstance(key, str) or not key:
            raise ValidationError("scope keys must be non-empty strings")
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise ValidationError("scope values must be scalar JSON values")
    return scope


def normalize_access_policy(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {"visibility": ["all"], "mutable_by": ["owner"], "sensitivity": "normal"}
    policy = _expect_mapping(value, "access_policy")
    visibility = policy.get("visibility", ["all"])
    if not isinstance(visibility, list) or not all(isinstance(v, str) for v in visibility):
        raise ValidationError("access_policy.visibility must be a list of strings")
    policy["visibility"] = visibility
    mutable_by = policy.get("mutable_by", ["owner"])
    if not isinstance(mutable_by, list) or not all(isinstance(v, str) for v in mutable_by):
        raise ValidationError("access_policy.mutable_by must be a list of strings")
    policy["mutable_by"] = mutable_by
    policy.setdefault("sensitivity", "normal")
    return policy


PREFERENCE_POLARITIES = {
    "avoid",
    "avoids",
    "forbid",
    "forbids",
    "prefer",
    "prefers",
    "require",
    "requires",
}


def _has_payload_field(payload: Mapping[str, Any], field: str) -> bool:
    return field in payload and payload[field] is not None


def _require_payload_fields(
    atom_type: str, payload: Mapping[str, Any], fields: tuple[str, ...]
) -> None:
    missing = [field for field in fields if not _has_payload_field(payload, field)]
    if missing:
        raise ValidationError(
            f"{atom_type} payload missing required field(s): {', '.join(missing)}"
        )


def _require_payload_alternative(
    atom_type: str, payload: Mapping[str, Any], alternatives: tuple[tuple[str, ...], ...]
) -> None:
    for fields in alternatives:
        if all(_has_payload_field(payload, field) for field in fields):
            return
    rendered = ["+".join(fields) for fields in alternatives]
    raise ValidationError(
        f"{atom_type} payload must include one of: {', '.join(rendered)}"
    )


def _require_payload_list(
    atom_type: str, payload: Mapping[str, Any], field: str, *, non_empty: bool = False
) -> None:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ValidationError(f"{atom_type} payload field {field} must be a list")
    if non_empty and not value:
        raise ValidationError(f"{atom_type} payload field {field} must not be empty")


def validate_atom_payload(atom_type: str, payload: Mapping[str, Any]) -> None:
    """Enforce the v1 typed payload contracts used by MemoryAtom.type."""

    if atom_type == "belief":
        _require_payload_alternative(
            atom_type,
            payload,
            (
                ("claim",),
                ("subject", "predicate", "value"),
                ("subject", "relation", "object"),
            ),
        )
        return
    if atom_type == "preference":
        _require_payload_fields(
            atom_type,
            payload,
            ("holder", "polarity", "target", "applicability_scope", "strength"),
        )
        polarity = str(payload["polarity"])
        if polarity not in PREFERENCE_POLARITIES:
            raise ValidationError(f"unsupported preference polarity: {polarity}")
        return
    if atom_type == "goal":
        _require_payload_alternative(
            atom_type,
            payload,
            (("description",), ("objective",), ("desired_state",)),
        )
        return
    if atom_type == "commitment":
        _require_payload_alternative(
            atom_type,
            payload,
            (("description",), ("promised_action",)),
        )
        return
    if atom_type == "procedure":
        _require_payload_fields(atom_type, payload, ("trigger_context", "steps"))
        _require_payload_list(atom_type, payload, "steps", non_empty=True)
        return
    if atom_type == "episode":
        _require_payload_alternative(atom_type, payload, (("summary",), ("task",)))
        return
    if atom_type == "self_model":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        return
    if atom_type == "capability":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_alternative(atom_type, payload, (("name",), ("capability",)))
        return
    if atom_type == "limitation":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_alternative(atom_type, payload, (("name",), ("limitation",)))
        return
    if atom_type == "runtime_state":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        return
    if atom_type == "self_assessment":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_fields(atom_type, payload, ("claim", "calibration"))
        if not isinstance(payload["calibration"], Mapping):
            raise ValidationError("self_assessment payload field calibration must be an object")
        return
    if atom_type == "agentic_trace":
        _require_payload_fields(atom_type, payload, ("task", "action", "outcome"))
        external_constraints = payload.get("external_constraints", [])
        if external_constraints is not None and not isinstance(external_constraints, list):
            raise ValidationError(
                "agentic_trace payload field external_constraints must be a list"
            )
        return
    if atom_type == "action_outcome":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_fields(atom_type, payload, ("action_ref", "status"))
        return
    if atom_type == "self_narrative":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_fields(atom_type, payload, ("narrative", "artifact"))
        if payload["artifact"] is not True:
            raise ValidationError("self_narrative payload field artifact must be true")
        return
    if atom_type == "semantic":
        _require_payload_alternative(
            atom_type,
            payload,
            (("summary",), ("source_refs",), ("distillation_type",)),
        )
        return
    if atom_type == "policy":
        _require_payload_alternative(
            atom_type,
            payload,
            (("name",), ("description",), ("rule",), ("rules",)),
        )


def normalize_atom(atom: Mapping[str, Any], *, require_id: bool = False) -> dict[str, Any]:
    data = _expect_mapping(atom, "atom")
    atom_type = str(data.get("type", ""))
    if atom_type not in ATOM_TYPES:
        raise ValidationError(f"unsupported atom type: {atom_type!r}")

    payload = _expect_mapping(data.get("payload"), "payload")
    forbidden = sorted(PAYLOAD_FORBIDDEN_FIELDS.intersection(payload))
    if forbidden:
        raise ValidationError(
            "payload must not duplicate envelope fields: " + ", ".join(forbidden)
        )
    ensure_jsonable(payload)
    validate_atom_payload(atom_type, payload)

    atom_id = data.get("id")
    if require_id and not atom_id:
        raise ValidationError("atom.id is required")
    if atom_id is not None and not isinstance(atom_id, str):
        raise ValidationError("atom.id must be a string")

    lifecycle_state = str(data.get("lifecycle_state", "active"))
    if lifecycle_state not in LIFECYCLE_STATES:
        raise ValidationError(f"unsupported lifecycle_state: {lifecycle_state}")

    health_status = str(data.get("health_status", "healthy"))
    if health_status not in HEALTH_STATES:
        raise ValidationError(f"unsupported health_status: {health_status}")

    normalized = {
        "id": atom_id,
        "type": atom_type,
        "schema_version": str(data.get("schema_version", SCHEMA_VERSION)),
        "payload": payload,
        "evidence_refs": _expect_list(data.get("evidence_refs", []), "evidence_refs"),
        "scope": normalize_scope(data.get("scope")),
        "confidence": normalize_confidence(data.get("confidence")),
        "salience": float(data.get("salience", 0.5)),
        "utility": float(data.get("utility", 0.5)),
        "layer": str(data.get("layer", "working")),
        "lifecycle_state": lifecycle_state,
        "health_status": health_status,
        "retention_class": str(data.get("retention_class", "standard")),
        "access_policy": normalize_access_policy(data.get("access_policy")),
        "decay_policy": _expect_mapping(data.get("decay_policy", {}), "decay_policy"),
        "supersedes": _expect_list(data.get("supersedes", []), "supersedes"),
        "revision_history": _expect_list(
            data.get("revision_history", []), "revision_history"
        ),
        "index_refs": _expect_mapping(data.get("index_refs", {}), "index_refs"),
        "observed_at": data.get("observed_at"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "last_accessed": data.get("last_accessed"),
        "version": int(data.get("version", 1)),
    }
    if normalized["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"unsupported schema_version: {normalized['schema_version']}")
    if normalized["salience"] < 0 or normalized["utility"] < 0:
        raise ValidationError("salience and utility must be non-negative")
    return normalized


def normalize_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    data = _expect_mapping(evidence, "evidence")
    source_type = str(data.get("source_type", ""))
    source_ref = str(data.get("source_ref", ""))
    if not source_type:
        raise ValidationError("evidence.source_type is required")
    if not source_ref:
        raise ValidationError("evidence.source_ref is required")
    payload = data.get("payload", {})
    ensure_jsonable(payload)
    captured_at = data.get("captured_at") or utc_now()
    scope = normalize_scope(data.get("scope"))
    access_policy = normalize_access_policy(data.get("access_policy"))
    evidence_id = data.get("evidence_id") or stable_id(
        "evd",
        {
            "source_type": source_type,
            "source_ref": source_ref,
            "payload": payload,
            "captured_at": captured_at,
            "scope": scope,
        },
    )
    return {
        "evidence_id": evidence_id,
        "schema_version": str(data.get("schema_version", SCHEMA_VERSION)),
        "source_type": source_type,
        "source_ref": source_ref,
        "payload": payload,
        "captured_at": captured_at,
        "checksum": data.get("checksum") or digest(payload),
        "scope": scope,
        "access_policy": access_policy,
    }


def normalize_relation(relation: str) -> str:
    relation = str(relation)
    if relation not in EDGE_RELATIONS:
        raise ValidationError(f"unsupported edge relation: {relation}")
    return relation


def confidence_score(confidence: Mapping[str, Any] | None) -> float:
    if not confidence:
        return CONFIDENCE_SCORE_BY_LEVEL["medium"]
    if "score" in confidence:
        score = confidence["score"]
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
    return CONFIDENCE_SCORE_BY_LEVEL.get(str(confidence.get("level", "medium")), 0.5)


def parse_json_arg(value: str) -> Any:
    if value.startswith("@"):
        with open(value[1:], "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)
