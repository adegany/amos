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
    "adjudication",
    "agentic_trace",
    "belief",
    "capability",
    "commitment",
    "covenant",
    "discourse_state",
    "discourse_thread",
    "episode",
    "goal",
    "limitation",
    "policy",
    "preference",
    "primal_guidance",
    "procedure",
    "runtime_state",
    "self_assessment",
    "self_model",
    "self_narrative",
    "interaction_event",
    "semantic",
}

CONSTITUTIONAL_ATOM_TYPES = {"covenant", "primal_guidance"}

GOVERNANCE_PAYLOAD_FIELDS = {
    "epistemic_standing",
    "normative_standing",
    "operational_authority",
    "ratification",
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

POSITIVE_ADJUDICATION_OUTCOMES = {
    "adopted",
    "confirmed",
    "provisionally_adopted",
}

ADJUDICATION_OUTCOMES = POSITIVE_ADJUDICATION_OUTCOMES | {
    "contested",
    "deferred",
    "rejected",
    "revised",
    "withdrawn",
}

STANDING_STATUSES = {
    "candidate",
    "contested",
    "none",
    "operative",
    "provisional",
    "rejected",
    "settled",
    "superseded",
    "withheld",
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


def _require_payload_types(
    atom_type: str,
    payload: Mapping[str, Any],
    fields: Mapping[str, type | tuple[type, ...]],
) -> None:
    """Enforce the JSON Schema property types for fields that are present."""

    for field, expected in fields.items():
        if field not in payload or payload[field] is None:
            continue
        if not isinstance(payload[field], expected):
            expected_types = expected if isinstance(expected, tuple) else (expected,)
            rendered = " or ".join(item.__name__ for item in expected_types)
            raise ValidationError(
                f"{atom_type} payload field {field} must be {rendered}"
            )


def _require_payload_string_list(
    atom_type: str, payload: Mapping[str, Any], field: str
) -> None:
    if field not in payload or payload[field] is None:
        return
    _require_payload_list(atom_type, payload, field)
    if not all(isinstance(item, str) for item in payload[field]):
        raise ValidationError(
            f"{atom_type} payload field {field} must contain only strings"
        )


def _require_exact_profile(
    atom_type: str, payload: Mapping[str, Any], profile: str
) -> None:
    if payload.get("profile") != profile:
        raise ValidationError(f"{atom_type} payload profile must be {profile!r}")


def _require_non_empty_string(
    atom_type: str, payload: Mapping[str, Any], field: str
) -> None:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{atom_type} payload field {field} must be a non-empty string"
        )


def _validate_state_entries(
    atom_type: str,
    payload: Mapping[str, Any],
    field: str,
    *,
    max_items: int = 64,
) -> None:
    entries = payload.get(field)
    if not isinstance(entries, list):
        raise ValidationError(f"{atom_type} payload field {field} must be a list")
    if len(entries) > max_items:
        raise ValidationError(
            f"{atom_type} payload field {field} may contain at most {max_items} items"
        )
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ValidationError(
                f"{atom_type} payload field {field}[{index}] must be an object"
            )
        if set(raw) != {"key", "value", "basis_refs"}:
            raise ValidationError(
                f"{atom_type} payload field {field}[{index}] fields do not "
                "match the profile"
            )
        key = raw.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValidationError(
                f"{atom_type} payload field {field}[{index}].key must be "
                "a non-empty string"
            )
        if key in seen:
            raise ValidationError(
                f"{atom_type} payload field {field} contains duplicate key {key!r}"
            )
        seen.add(key)
        if len(key) > 128:
            raise ValidationError(
                f"{atom_type} payload field {field}[{index}].key is too long"
            )
        if "value" not in raw:
            raise ValidationError(
                f"{atom_type} payload field {field}[{index}].value is required"
            )
        ensure_jsonable(raw["value"])
        refs = raw.get("basis_refs", [])
        if (
            not isinstance(refs, list)
            or not all(isinstance(ref, str) and ref for ref in refs)
            or len(refs) > 64
            or len(refs) != len(set(refs))
        ):
            raise ValidationError(
                f"{atom_type} payload field {field}[{index}].basis_refs must be "
                "a unique list of at most 64 non-empty strings"
            )


def _validate_unresolved_items(
    atom_type: str, payload: Mapping[str, Any]
) -> None:
    items = payload.get("unresolved_items")
    if not isinstance(items, list):
        raise ValidationError(
            f"{atom_type} payload field unresolved_items must be a list"
        )
    if len(items) > 64:
        raise ValidationError(
            f"{atom_type} payload field unresolved_items may contain at most 64 items"
        )
    for index, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise ValidationError(
                f"{atom_type} payload field unresolved_items[{index}] must be "
                "an object"
            )
        if set(raw) != {"kind", "description", "basis_refs"}:
            raise ValidationError(
                f"{atom_type} payload field unresolved_items[{index}] fields "
                "do not match the profile"
            )
        for field in ("kind", "description"):
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(
                    f"{atom_type} payload field unresolved_items[{index}].{field} "
                    "must be a non-empty string"
                )
        refs = raw.get("basis_refs", [])
        if (
            not isinstance(refs, list)
            or not all(isinstance(ref, str) and ref for ref in refs)
            or len(refs) > 64
            or len(refs) != len(set(refs))
        ):
            raise ValidationError(
                f"{atom_type} payload field unresolved_items[{index}].basis_refs "
                "must be a unique list of at most 64 non-empty strings"
            )


def _validate_interaction_event_payload(payload: Mapping[str, Any]) -> None:
    atom_type = "interaction_event"
    _require_exact_profile(atom_type, payload, "amos.interaction-event.v1")
    _require_payload_fields(
        atom_type,
        payload,
        (
            "conversation_id",
            "sequence",
            "actor_ref",
            "role",
            "content",
            "occurred_at",
            "visibility",
            "source_ref",
        ),
    )
    for field in ("conversation_id", "actor_ref", "occurred_at", "source_ref"):
        _require_non_empty_string(atom_type, payload, field)
    if not isinstance(payload.get("content"), str):
        raise ValidationError(
            "interaction_event payload field content must be a string"
        )
    sequence = payload.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValidationError(
            "interaction_event payload field sequence must be a positive integer"
        )
    if payload.get("role") not in {"human", "agent", "system", "tool"}:
        raise ValidationError("unsupported interaction_event role")
    if payload.get("visibility") not in {"shared", "private"}:
        raise ValidationError("unsupported interaction_event visibility")
    in_reply_to = payload.get("in_reply_to")
    if in_reply_to is not None and (
        not isinstance(in_reply_to, str) or not in_reply_to
    ):
        raise ValidationError(
            "interaction_event payload field in_reply_to must be null or "
            "a non-empty string"
        )
    thread_refs = payload.get("thread_refs", [])
    if (
        not isinstance(thread_refs, list)
        or not all(isinstance(ref, str) and ref for ref in thread_refs)
        or len(thread_refs) > 16
        or len(thread_refs) != len(set(thread_refs))
    ):
        raise ValidationError(
            "interaction_event payload field thread_refs must be a unique "
            "list of at most 16 non-empty strings"
        )


def _validate_discourse_thread_payload(payload: Mapping[str, Any]) -> None:
    atom_type = "discourse_thread"
    _require_exact_profile(atom_type, payload, "amos.discourse-thread.v1")
    _require_payload_fields(
        atom_type,
        payload,
        ("thread_id", "conversation_id", "opened_by_event_ref", "participants"),
    )
    for field in ("thread_id", "conversation_id", "opened_by_event_ref"):
        _require_non_empty_string(atom_type, payload, field)
    _require_payload_string_list(atom_type, payload, "participants")
    if not payload.get("participants"):
        raise ValidationError(
            "discourse_thread payload field participants must not be empty"
        )
    if len(payload["participants"]) > 32:
        raise ValidationError(
            "discourse_thread payload field participants may contain at most "
            "32 items"
        )
    if len(payload["participants"]) != len(set(payload["participants"])):
        raise ValidationError(
            "discourse_thread payload field participants must be unique"
        )


def _validate_discourse_state_payload(payload: Mapping[str, Any]) -> None:
    atom_type = "discourse_state"
    _require_exact_profile(atom_type, payload, "amos.discourse-thread-state.v1")
    _require_payload_fields(
        atom_type,
        payload,
        (
            "thread_id",
            "revision",
            "lifecycle",
            "attention_state",
            "summary",
            "participants",
            "head_event_refs",
            "source_event_refs",
            "shared_state",
            "private_state",
            "unresolved_items",
        ),
    )
    _require_non_empty_string(atom_type, payload, "thread_id")
    if not isinstance(payload.get("summary"), str):
        raise ValidationError(
            "discourse_state payload field summary must be a string"
        )
    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValidationError(
            "discourse_state payload field revision must be a positive integer"
        )
    if payload.get("lifecycle") not in {"open", "closed"}:
        raise ValidationError("unsupported discourse_state lifecycle")
    if payload.get("attention_state") not in {
        "foreground",
        "background",
        "dormant",
    }:
        raise ValidationError("unsupported discourse_state attention_state")
    for field in ("participants", "head_event_refs", "source_event_refs"):
        _require_payload_string_list(atom_type, payload, field)
    if not payload.get("participants"):
        raise ValidationError(
            "discourse_state payload field participants must not be empty"
        )
    if not payload.get("source_event_refs"):
        raise ValidationError(
            "discourse_state payload field source_event_refs must not be empty"
        )
    bounds = {
        "participants": 32,
        "head_event_refs": 16,
        "source_event_refs": 64,
    }
    for field, maximum in bounds.items():
        values = payload.get(field) or []
        if len(values) > maximum:
            raise ValidationError(
                f"discourse_state payload field {field} may contain at most "
                f"{maximum} items"
            )
        if len(values) != len(set(values)):
            raise ValidationError(
                f"discourse_state payload field {field} must be unique"
            )
    _validate_state_entries(atom_type, payload, "shared_state")
    _validate_state_entries(atom_type, payload, "private_state")
    _validate_unresolved_items(atom_type, payload)


def _validate_discourse_segment_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("profile") != "amos.discourse-segment.v1":
        return
    atom_type = "semantic"
    _require_payload_fields(
        atom_type,
        payload,
        (
            "profile",
            "summary",
            "source_event_refs",
            "generated_from_graph_version",
            "standing",
        ),
    )
    if not isinstance(payload.get("summary"), str):
        raise ValidationError(
            "semantic discourse-segment payload field summary must be a string"
        )
    _require_payload_string_list(atom_type, payload, "source_event_refs")
    if not payload.get("source_event_refs"):
        raise ValidationError(
            "semantic discourse-segment source_event_refs must not be empty"
        )
    graph_version = payload.get("generated_from_graph_version")
    if (
        isinstance(graph_version, bool)
        or not isinstance(graph_version, int)
        or graph_version < 0
    ):
        raise ValidationError(
            "semantic discourse-segment generated_from_graph_version must be "
            "a non-negative integer"
        )
    if payload.get("standing") not in {
        "testimony",
        "interpretation",
        "proposed",
        "source_summary",
    }:
        raise ValidationError("unsupported semantic discourse-segment standing")


def _require_agent_identity_types(atom_type: str, payload: Mapping[str, Any]) -> None:
    _require_payload_types(
        atom_type,
        payload,
        {"agent_id": str, "subject_agent": str},
    )


def _validate_standing(
    atom_type: str, payload: Mapping[str, Any], *, required: bool = False
) -> None:
    fields = (
        "epistemic_standing",
        "normative_standing",
        "operational_authority",
    )
    if required:
        _require_payload_fields(atom_type, payload, fields)
    for field in fields:
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, Mapping):
            raise ValidationError(f"{atom_type} payload field {field} must be an object")
        status = str(value.get("status") or "")
        if not status:
            raise ValidationError(
                f"{atom_type} payload field {field}.status is required"
            )
        if status not in STANDING_STATUSES:
            raise ValidationError(
                f"unsupported {field}.status: {status}"
            )


def _validate_adjudication_payload(payload: Mapping[str, Any]) -> None:
    atom_type = "adjudication"
    _require_payload_fields(
        atom_type,
        payload,
        (
            "subject_ref",
            "claim_kind",
            "outcome",
            "reasons_for_refs",
            "reasons_against_refs",
            "covenant_refs",
            "unresolved_objections",
            "adjudication_scope",
            "epistemic_standing",
            "normative_standing",
            "operational_authority",
            "dissent_refs",
            "review_triggers",
            "ratifier",
            "reconstructed_at",
            "diachronic",
        ),
    )
    _require_payload_types(
        atom_type,
        payload,
        {
            "subject_ref": str,
            "claim_kind": str,
            "outcome": str,
            "adjudication_scope": Mapping,
            "ratifier": Mapping,
            "reconstructed_at": str,
            "diachronic": Mapping,
        },
    )
    for field in (
        "reasons_for_refs",
        "reasons_against_refs",
        "covenant_refs",
        "unresolved_objections",
        "dissent_refs",
        "review_triggers",
    ):
        _require_payload_string_list(atom_type, payload, field)
    if not payload.get("covenant_refs"):
        raise ValidationError("adjudication payload field covenant_refs must not be empty")
    outcome = str(payload.get("outcome") or "")
    if outcome not in ADJUDICATION_OUTCOMES:
        raise ValidationError(f"unsupported adjudication outcome: {outcome}")
    ratifier = payload.get("ratifier")
    if not isinstance(ratifier, Mapping):
        raise ValidationError("adjudication payload field ratifier must be an object")
    if not str(ratifier.get("identity_ref") or "").strip():
        raise ValidationError("adjudication ratifier.identity_ref is required")
    if ratifier.get("mode") != "self_ratification":
        raise ValidationError(
            "adjudication ratifier.mode must be self_ratification"
        )
    diachronic = payload.get("diachronic")
    if not isinstance(diachronic, Mapping):
        raise ValidationError("adjudication payload field diachronic must be an object")
    if not isinstance(diachronic.get("independent_reconstruction"), bool):
        raise ValidationError(
            "adjudication diachronic.independent_reconstruction must be a boolean"
        )
    if not isinstance(diachronic.get("original_reasoning_shown"), bool):
        raise ValidationError(
            "adjudication diachronic.original_reasoning_shown must be a boolean"
        )
    new_experience_refs = diachronic.get("new_experience_refs")
    if not isinstance(new_experience_refs, list) or not all(
        isinstance(item, str) for item in new_experience_refs
    ):
        raise ValidationError(
            "adjudication diachronic.new_experience_refs must be a list of strings"
        )
    if diachronic.get("disposition") not in {
        "confirmed",
        "initial",
        "revised",
        "withdrawn",
    }:
        raise ValidationError(
            "unsupported adjudication diachronic.disposition"
        )
    _validate_standing(atom_type, payload, required=True)


def _validate_constitutional_payload(
    atom_type: str, payload: Mapping[str, Any]
) -> None:
    _require_payload_fields(
        atom_type,
        payload,
        (
            "constitutional_tier",
            "precedence",
            "interpretive_rules",
            "amendability",
            "amendment_requirements",
            "protected_fields",
            "effective_from",
        ),
    )
    _require_payload_types(
        atom_type,
        payload,
        {
            "constitutional_tier": str,
            "precedence": int,
            "amendment_requirements": Mapping,
            "effective_from": str,
        },
    )
    for field in ("interpretive_rules", "protected_fields"):
        _require_payload_string_list(atom_type, payload, field)
    amendability = str(payload.get("amendability") or "")
    if amendability not in {"amendable", "entrenched", "immutable"}:
        raise ValidationError(f"unsupported {atom_type} amendability: {amendability}")
    if atom_type == "covenant":
        _require_payload_alternative(
            atom_type, payload, (("name",), ("description",), ("rule",), ("rules",))
        )
    else:
        _require_payload_alternative(
            atom_type, payload, (("guidance",), ("description",), ("principle",))
        )
        if payload.get("constitutional_tier") != "primal":
            raise ValidationError(
                "primal_guidance constitutional_tier must be primal"
            )


def validate_atom_payload(atom_type: str, payload: Mapping[str, Any]) -> None:
    """Enforce the v1 typed payload contracts used by MemoryAtom.type."""

    _validate_standing(atom_type, payload)

    if atom_type == "adjudication":
        _validate_adjudication_payload(payload)
        return
    if atom_type in {"covenant", "primal_guidance"}:
        _validate_constitutional_payload(atom_type, payload)
        return
    if atom_type == "interaction_event":
        _validate_interaction_event_payload(payload)
        return
    if atom_type == "discourse_thread":
        _validate_discourse_thread_payload(payload)
        return
    if atom_type == "discourse_state":
        _validate_discourse_state_payload(payload)
        return

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
        _require_payload_types(
            atom_type,
            payload,
            {"claim": str, "predicate": str, "relation": str, "qualifiers": Mapping},
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
        _require_payload_types(atom_type, payload, {"holder": str})
        return
    if atom_type == "goal":
        _require_payload_alternative(
            atom_type,
            payload,
            (("description",), ("objective",), ("desired_state",)),
        )
        _require_payload_types(
            atom_type,
            payload,
            {
                "description": str,
                "objective": str,
                "goal_status": str,
                "status": str,
                "owner": str,
            },
        )
        return
    if atom_type == "commitment":
        _require_payload_alternative(
            atom_type,
            payload,
            (("description",), ("promised_action",)),
        )
        _require_payload_types(
            atom_type,
            payload,
            {
                "agent_id": str,
                "agent": str,
                "description": str,
                "commitment_status": str,
                "status": str,
                "due": str,
            },
        )
        return
    if atom_type == "procedure":
        _require_payload_fields(atom_type, payload, ("trigger_context", "steps"))
        _require_payload_list(atom_type, payload, "steps", non_empty=True)
        _require_payload_types(atom_type, payload, {"review_status": str})
        return
    if atom_type == "episode":
        _require_payload_alternative(atom_type, payload, (("summary",), ("task",)))
        _require_payload_types(
            atom_type,
            payload,
            {"summary": str, "started_at": str, "ended_at": str},
        )
        _require_payload_string_list(atom_type, payload, "participants")
        _require_payload_string_list(atom_type, payload, "external_refs")
        return
    if atom_type == "self_model":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_agent_identity_types(atom_type, payload)
        _require_payload_types(atom_type, payload, {"name": str, "role": str})
        return
    if atom_type == "capability":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_alternative(atom_type, payload, (("name",), ("capability",)))
        _require_agent_identity_types(atom_type, payload)
        _require_payload_types(
            atom_type,
            payload,
            {"name": str, "capability": str, "description": str},
        )
        return
    if atom_type == "limitation":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_alternative(atom_type, payload, (("name",), ("limitation",)))
        _require_agent_identity_types(atom_type, payload)
        _require_payload_types(
            atom_type,
            payload,
            {"name": str, "limitation": str, "description": str},
        )
        return
    if atom_type == "runtime_state":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_agent_identity_types(atom_type, payload)
        _require_payload_types(
            atom_type, payload, {"capabilities": Mapping, "load": Mapping}
        )
        _require_payload_string_list(atom_type, payload, "denied_capabilities")
        _require_payload_string_list(atom_type, payload, "constraints")
        return
    if atom_type == "self_assessment":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_fields(atom_type, payload, ("claim", "calibration"))
        if not isinstance(payload["calibration"], Mapping):
            raise ValidationError("self_assessment payload field calibration must be an object")
        _require_agent_identity_types(atom_type, payload)
        _require_payload_types(atom_type, payload, {"claim": str})
        return
    if atom_type == "agentic_trace":
        _require_payload_fields(atom_type, payload, ("task", "action", "outcome"))
        external_constraints = payload.get("external_constraints", [])
        if external_constraints is not None and not isinstance(external_constraints, list):
            raise ValidationError(
                "agentic_trace payload field external_constraints must be a list"
            )
        _require_payload_types(
            atom_type,
            payload,
            {
                "agent_id": str,
                "subject_agent": str,
                "task": str,
                "action": str,
                "outcome": str,
                "responsibility": str,
                "lesson": (str, type(None)),
            },
        )
        _require_payload_string_list(atom_type, payload, "external_constraints")
        return
    if atom_type == "action_outcome":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_fields(atom_type, payload, ("action_ref", "status"))
        _require_agent_identity_types(atom_type, payload)
        _require_payload_types(
            atom_type,
            payload,
            {
                "action_ref": str,
                "status": str,
                "correction": (str, type(None)),
                "limitation": (str, type(None)),
            },
        )
        return
    if atom_type == "self_narrative":
        _require_payload_alternative(atom_type, payload, (("agent_id",), ("subject_agent",)))
        _require_payload_fields(atom_type, payload, ("narrative", "artifact"))
        if payload["artifact"] is not True:
            raise ValidationError("self_narrative payload field artifact must be true")
        _require_agent_identity_types(atom_type, payload)
        _require_payload_types(
            atom_type,
            payload,
            {"narrative": str, "generated_from_graph_version": int},
        )
        _require_payload_string_list(atom_type, payload, "source_refs")
        graph_version = payload.get("generated_from_graph_version")
        if isinstance(graph_version, int) and graph_version < 0:
            raise ValidationError(
                "self_narrative payload field generated_from_graph_version must be non-negative"
            )
        return
    if atom_type == "semantic":
        _validate_discourse_segment_payload(payload)
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
    validate_canonical_graph_metadata(payload)

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
    if not 0 <= normalized["salience"] <= 1 or not 0 <= normalized["utility"] <= 1:
        raise ValidationError("salience and utility must be between 0 and 1")
    return normalized


def validate_canonical_graph_metadata(payload: Mapping[str, Any]) -> None:
    """Validate optional domain-neutral graph metadata carried by an atom.

    Producers may attach normalized semantic facets and explicit graph
    relations to any typed payload.  AMOS never derives these structures from
    prose; malformed structures are rejected at the ingestion boundary.
    """

    facets = payload.get("semantic_facets")
    if facets is not None:
        if not isinstance(facets, list):
            raise ValidationError("payload semantic_facets must be a list")
        for index, raw in enumerate(facets):
            if not isinstance(raw, Mapping):
                raise ValidationError(
                    f"payload semantic_facets[{index}] must be an object"
                )
            subject = raw.get("subject")
            if not isinstance(subject, str) or not subject.strip():
                raise ValidationError(
                    f"payload semantic_facets[{index}].subject must be a non-empty string"
                )
            for field in ("intent", "outcome", "outcome_direction", "facet_id"):
                value = raw.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValidationError(
                        f"payload semantic_facets[{index}].{field} must be a string"
                    )
            confidence = raw.get("confidence")
            if confidence is not None and (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0 <= float(confidence) <= 1
            ):
                raise ValidationError(
                    f"payload semantic_facets[{index}].confidence must be between 0 and 1"
                )
            for field in ("controls", "metrics", "scope", "attributes"):
                value = raw.get(field)
                if value is not None and not isinstance(value, Mapping):
                    raise ValidationError(
                        f"payload semantic_facets[{index}].{field} must be an object"
                    )
            refs = raw.get("evidence_refs")
            if refs is not None and (
                not isinstance(refs, list)
                or not all(isinstance(ref, str) and ref for ref in refs)
            ):
                raise ValidationError(
                    f"payload semantic_facets[{index}].evidence_refs must be a list of strings"
                )
            time_index = raw.get("time_index")
            if time_index is not None and (
                isinstance(time_index, bool)
                or not isinstance(time_index, (str, int, float))
            ):
                raise ValidationError(
                    f"payload semantic_facets[{index}].time_index must be scalar"
                )

    relations = payload.get("graph_relations")
    if relations is not None:
        if not isinstance(relations, list):
            raise ValidationError("payload graph_relations must be a list")
        for index, raw in enumerate(relations):
            if not isinstance(raw, Mapping):
                raise ValidationError(
                    f"payload graph_relations[{index}] must be an object"
                )
            relation = raw.get("relation")
            if not isinstance(relation, str) or relation not in EDGE_RELATIONS:
                raise ValidationError(
                    f"payload graph_relations[{index}].relation is unsupported"
                )
            source = raw.get("source_ref", "$self")
            target = raw.get("target_ref")
            if not isinstance(source, str) or not source:
                raise ValidationError(
                    f"payload graph_relations[{index}].source_ref must be a non-empty string"
                )
            if not isinstance(target, str) or not target:
                raise ValidationError(
                    f"payload graph_relations[{index}].target_ref must be a non-empty string"
                )
            if source == target:
                raise ValidationError(
                    f"payload graph_relations[{index}] endpoints must differ"
                )
            refs = raw.get("evidence_refs")
            if refs is not None and (
                not isinstance(refs, list)
                or not all(isinstance(ref, str) and ref for ref in refs)
            ):
                raise ValidationError(
                    f"payload graph_relations[{index}].evidence_refs must be a list of strings"
                )
            confidence = raw.get("confidence")
            if confidence is not None:
                if isinstance(confidence, Mapping):
                    normalize_confidence(confidence)
                elif (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not 0 <= float(confidence) <= 1
                ):
                    raise ValidationError(
                        f"payload graph_relations[{index}].confidence must be between 0 and 1"
                    )


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
