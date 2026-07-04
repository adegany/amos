"""AMOS v1 seed ontology dictionary."""

from __future__ import annotations

SEED_ENTITY_TYPES = {
    "action_outcome",
    "agent",
    "agentic_trace",
    "capability",
    "concept",
    "document",
    "environment",
    "file",
    "limitation",
    "organization",
    "policy",
    "procedure",
    "project",
    "runtime_state",
    "self_assessment",
    "self_narrative",
    "service",
    "task",
    "tool",
    "user",
    "workspace",
}

SEED_RELATION_IDS = {
    "rel:acted_on",
    "rel:applies_to",
    "rel:attributed_to",
    "rel:avoids",
    "rel:caused_by",
    "rel:constrained_by",
    "rel:contradicts",
    "rel:corrected_by",
    "rel:currently_available",
    "rel:currently_denied",
    "rel:decided",
    "rel:depends_on",
    "rel:derived_from",
    "rel:forbids",
    "rel:has_capability",
    "rel:has_limitation",
    "rel:made_commitment",
    "rel:miscalibrated_on",
    "rel:owns",
    "rel:part_of",
    "rel:prefers",
    "rel:produced_outcome",
    "rel:requires",
    "rel:satisfied_commitment",
    "rel:shared_responsibility_for",
    "rel:similar_to",
    "rel:supersedes",
    "rel:supports",
    "rel:uses",
    "rel:working_on",
}

DICTIONARY_UPDATE_POLICY = {
    "system_defined_relation": "schema_or_dictionary_migration_only",
    "tenant_defined_relation": "tenant_admin_only",
    "agent_defined_relation": "not_allowed_in_v1_propose_for_review_only",
    "deprecated_relation": "remains_resolvable_while_referenced",
}


def ontology_snapshot() -> dict[str, object]:
    return {
        "entity_types": sorted(SEED_ENTITY_TYPES),
        "relation_ids": sorted(SEED_RELATION_IDS),
        "dictionary_update_policy": dict(DICTIONARY_UPDATE_POLICY),
    }
