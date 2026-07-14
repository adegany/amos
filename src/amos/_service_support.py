"""High-level AMOS v1 service API."""

from __future__ import annotations

import json
import math
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import AccessDenied, CASConflict, IdempotencyConflict, ValidationError
from .maintenance import (
    EvidenceWindow,
    MaintenanceProcessor,
    SEMANTIC_RELATION_PROCESSOR_ID,
    SEMANTIC_RELATION_PROCESSOR_VERSION,
    default_processor_registry,
    load_maintenance_processor,
    proposal_is_auto_committable,
    semantic_relation_proposals_from_facets,
)
from .schemas import (
    EDGE_RELATIONS,
    SCHEMA_VERSION,
    canonical_json,
    confidence_score,
    digest,
    normalize_atom,
    normalize_evidence,
    normalize_relation,
    stable_id,
    utc_now,
)
from .smp import SemanticMaintenanceProcessor, cosine
from .store import SQLiteStore


LOW_HEALTH_STATES = {"confounding", "low_utility", "merged", "orphaned", "stale"}
CONFLICT_RELATIONS = {"rel:contradicts"}
HIGH_RISK_MAINTENANCE = {
    "delete",
    "hard_delete",
    "irreversible_delete",
    "merge_without_alias",
    "rewrite_confidence_policy",
}
DEFAULT_PACKET_PROFILES = {
    "reasoner": {"max_items": 24, "tokens": 6000, "include_conflicts": True},
    "planner": {"max_items": 20, "tokens": 4500, "include_conflicts": False},
    "executor": {"max_items": 16, "tokens": 3500, "include_conflicts": False},
    "critic": {"max_items": 32, "tokens": 8000, "include_conflicts": True},
    "steward": {"max_items": 64, "tokens": 12000, "include_conflicts": True},
    "self_awareness": {"max_items": 100, "tokens": 24000, "include_conflicts": True},
    "shared_coordination": {"max_items": 48, "tokens": 9000, "include_conflicts": True},
    "agentic_recall": {"max_items": 40, "tokens": 7000, "include_conflicts": True},
}
DEFAULT_MEMORY_POLICY = {
    "enabled": True,
    "schedule": {
        "every_graph_versions": 25,
        "every_seconds": 300,
        "run_on_pressure": True,
    },
    "maintenance": {
        "enabled": True,
        "run_smp": True,
        "max_smp_atoms": 128,
        "run_steward": True,
        "rebuild_indexes": True,
        "rebuild_lsa": True,
        "lsa_dimensions": 32,
        "lsa_max_terms": 300,
        "invalidate_packet_cache": True,
    },
    "distillation": {
        "enabled": True,
        "min_source_atoms": 6,
        "max_source_atoms": 10,
        "candidate_types": [
            "action_outcome",
            "agentic_trace",
            "belief",
            "episode",
            "preference",
        ],
        "distillation_type": "automatic_policy",
        "archive_sources": False,
        "approved_by": None,
    },
    "maintenance_distiller": {
        "enabled": True,
        "auto_commit_low_risk": True,
        "processor_ids": [],
        "domain": "generic",
        "max_atoms": 128,
        "max_events": 64,
        "max_retrieval_outcomes": 64,
        "reviewer": {
            "enabled": False,
            "authority": "draft_only",
        },
    },
    "decay": {
        "enabled": True,
        "max_atoms": 256,
        "require_atom_policy": True,
        "pressure_archive_policyless": True,
        "pressure_max_archives_per_run": 256,
        "pressure_protected_types": ["commitment", "policy", "self_model"],
        "capacity_assessment_targets": [256, 512, 768],
        "capacity_headroom_ratio": 0.2,
        "archive_superseded": True,
        "archive_superseded_after_seconds": 0,
        "mark_stale_after_seconds": None,
        "archive_after_seconds": None,
        "low_utility_threshold": None,
    },
    "storage_cleanup": {
        "enabled": True,
        "trigger": "idle",
        "idle_after_seconds": 300,
        "min_interval_seconds": 900,
        "max_deletions_per_tick": 256,
        "remove_archived_from_hot_index": True,
        "remove_stale_from_hot_index": True,
        "delete_archived_after_seconds": 604800,
        "delete_stale_after_seconds": 1209600,
        "protected_types": ["policy", "self_model", "commitment"],
        "compact_idempotency_after_seconds": 604800,
        "max_idempotency_compactions_per_tick": 512,
        "sqlite_compaction": {
            "checkpoint_wal": True,
            "checkpoint_mode": "TRUNCATE",
            "vacuum_enabled": True,
            "vacuum_idle_after_seconds": 1800,
            "vacuum_min_interval_seconds": 86400,
        },
    },
}
SEARCH_INDEX_REF = "amos.v1.search"
SEARCH_INDEX_SCHEMA = "amos.v1.search.payload_values.v2"
ATTENTION_POLICY_ID = "amos.v1.attention.default"
RETRIEVAL_RECENCY_HORIZON_SECONDS = 30 * 24 * 60 * 60
RETRIEVAL_WEIGHTS = {
    "direct_cue_match": 0.22,
    "semantic_similarity": 0.14,
    "edge_activation": 0.12,
    "recency": 0.08,
    "confidence": 0.12,
    "utility": 0.12,
    "salience": 0.08,
    "scope_specificity": 0.06,
    "goal_relevance": 0.08,
    "procedural_applicability": 0.04,
    "attention_focus": 0.14,
    "attention_type_boost": 0.08,
    "attention_counterevidence": 0.08,
    "attention_novelty": 0.05,
    "attention_suppression_penalty": -0.20,
    "contradiction_penalty": -0.30,
    "staleness_penalty": -0.18,
    "redundancy_penalty": -0.15,
    "superseded_penalty": -0.20,
}
SEMANTIC_MATCH_THRESHOLD = 0.22


def scope_visible(atom_scope: Mapping[str, Any], request_scope: Mapping[str, Any]) -> bool:
    if not atom_scope:
        return True
    for key, value in atom_scope.items():
        if value == "global":
            continue
        if request_scope.get(key) != value:
            return False
    return True


def maintenance_scope_visible(
    atom_scope: Mapping[str, Any], maintenance_scope: Mapping[str, Any]
) -> bool:
    if not maintenance_scope:
        return True
    # Maintenance scopes are compatible hierarchically in both directions: a
    # broad pass includes narrower evidence, while a run-specific pass still
    # includes tenant-level profiles. Only an explicit conflicting dimension
    # (for example a different run_id or tenant) makes an atom invisible.
    for key in set(atom_scope).intersection(maintenance_scope):
        atom_value = atom_scope.get(key)
        maintenance_value = maintenance_scope.get(key)
        if atom_value == "global" or maintenance_value == "global":
            continue
        if atom_value != maintenance_value:
            return False
    return True


def access_visible(
    access_policy: Mapping[str, Any], requester: str, target_processor: str
) -> bool:
    visibility = access_policy.get("visibility", ["all"])
    return (
        "all" in visibility
        or requester in visibility
        or target_processor in visibility
        or f"processor:{target_processor}" in visibility
    )


def payload_agent_id(payload: Mapping[str, Any]) -> Any:
    return payload.get("agent_id") or payload.get("subject_agent") or payload.get("agent")


def payload_capability_name(payload: Mapping[str, Any]) -> str:
    return str(payload.get("name") or payload.get("capability") or "")


def _structured_ref_list(value: Any) -> list[str]:
    refs: list[str] = []

    def add(ref: Any) -> None:
        text = str(ref or "").strip()
        if text and text not in refs:
            refs.append(text)

    if isinstance(value, str):
        add(value)
    elif isinstance(value, Mapping):
        for key in ("id", "atom_ref", "atom_id", "source_ref", "target_ref", "ref"):
            if value.get(key):
                add(value.get(key))
                break
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            refs.extend(ref for ref in _structured_ref_list(item) if ref not in refs)
    return refs






def _top_symmetric_components(
    matrix: list[list[float]],
    *,
    count: int,
    labels: Sequence[str],
    iterations: int = 80,
    tolerance: float = 1e-10,
) -> list[tuple[float, list[float]]]:
    """Return deterministic leading eigen-components for a small dense matrix."""

    size = len(matrix)
    if size == 0 or count <= 0:
        return []
    working = [row[:] for row in matrix]
    components: list[tuple[float, list[float]]] = []
    for component_index in range(min(count, size)):
        vector = _deterministic_unit_vector(labels, component_index)
        if not vector:
            break
        for _ in range(iterations):
            candidate = _matrix_vector_product(working, vector)
            norm = math.sqrt(sum(value * value for value in candidate))
            if norm <= tolerance:
                break
            candidate = [value / norm for value in candidate]
            delta = sum(abs(left - right) for left, right in zip(candidate, vector))
            vector = candidate
            if delta <= tolerance:
                break
        projected = _matrix_vector_product(working, vector)
        eigenvalue = sum(left * right for left, right in zip(vector, projected))
        if eigenvalue <= tolerance:
            break
        pivot = max(range(size), key=lambda index: abs(vector[index]))
        if vector[pivot] < 0:
            vector = [-value for value in vector]
        components.append((eigenvalue, vector))
        for row_index in range(size):
            row_value = vector[row_index]
            for column_index in range(size):
                working[row_index][column_index] -= (
                    eigenvalue * row_value * vector[column_index]
                )
    return components


def _matrix_vector_product(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    return [
        sum(row[index] * vector[index] for index in range(len(vector)))
        for row in matrix
    ]


def _deterministic_unit_vector(labels: Sequence[str], salt: int) -> list[float]:
    values: list[float] = []
    for label in labels:
        raw = int(digest({"label": label, "salt": salt})[:12], 16)
        values.append((raw % 2000000) / 1000000.0 - 1.0)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 and values:
        values[0] = 1.0
        norm = 1.0
    return [value / norm for value in values] if norm > 0 else []

