"""Generic maintenance proposal and processor registry for AMOS.

The classes here keep semantic maintenance processors deterministic and
side-effect free. Processors inspect an evidence window and return proposals;
the AMOS service decides whether any proposal is safe to commit.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .errors import ValidationError
from .schemas import SCHEMA_VERSION, digest, normalize_relation, stable_id, utc_now
from .smp import SemanticMaintenanceProcessor


LOW_RISK_PROPOSAL_ACTIONS = {"add_atom", "add_edge"}

SEMANTIC_RELATION_PROCESSOR_ID = "amos.semantic_relations.v1"
SEMANTIC_RELATION_PROCESSOR_VERSION = "amos.semantic_relations.v1"

GENERIC_GRAPH_PROCESSOR_ID = "amos.graph.canonical.v1"
GENERIC_GRAPH_PROCESSOR_VERSION = "amos.graph.canonical.v1"

LOW_RISK_EXPLICIT_RELATIONS = {
    "rel:attributed_to",
    "rel:constrained_by",
    "rel:corrected_by",
    "rel:derived_from",
    "rel:has_capability",
    "rel:has_limitation",
    "rel:made_commitment",
    "rel:part_of",
    "rel:produced_outcome",
    "rel:supersedes",
    "rel:uses",
}


@dataclass(frozen=True)
class EvidenceWindow:
    """Bounded immutable context presented to maintenance processors."""

    atoms: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    retrieval_outcomes: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    scope: Mapping[str, Any] = field(default_factory=dict)
    domain: str = "generic"
    graph_version: int = 0
    coverage: Mapping[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_count": len(self.atoms),
            "edge_count": len(self.edges),
            "evidence_count": len(self.evidence),
            "retrieval_outcome_count": len(self.retrieval_outcomes),
            "event_count": len(self.events),
            "scope": dict(self.scope),
            "domain": self.domain,
            "graph_version": self.graph_version,
            "coverage": dict(self.coverage),
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class MaintenanceWindowRequest:
    """Optional domain-neutral workset request for a processor.

    The service remains responsible for scope, lifecycle, and size bounds. A
    processor may ask for a narrower typed workset, but cannot use this object
    to widen the caller-authorized scope or acquire mutation authority.
    """

    lifecycle_states: tuple[str, ...] = ("active", "proposed")
    atom_types: tuple[str, ...] = ()
    graph_metadata_profiles: tuple[str, ...] = ()
    max_atoms: int | None = None
    include_graph_neighbors: bool = False
    include_evidence: bool = True
    include_events: bool = True
    include_retrieval_outcomes: bool = True
    max_evidence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycle_states": list(self.lifecycle_states),
            "atom_types": list(self.atom_types),
            "graph_metadata_profiles": list(self.graph_metadata_profiles),
            "max_atoms": self.max_atoms,
            "include_graph_neighbors": self.include_graph_neighbors,
            "include_evidence": self.include_evidence,
            "include_events": self.include_events,
            "include_retrieval_outcomes": self.include_retrieval_outcomes,
            "max_evidence": self.max_evidence,
        }


@dataclass(frozen=True)
class MaintenanceProposal:
    """A normalized processor recommendation for later policy gating."""

    processor_id: str
    processor_version: str
    action: str
    risk_level: str
    confidence: float
    reason_code: str
    source_refs: tuple[str, ...]
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...] = ()
    target_refs: tuple[str, ...] = ()
    title: str = ""
    proposal_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        body = {
            "processor_id": self.processor_id,
            "processor_version": self.processor_version,
            "action": self.action,
            "risk_level": self.risk_level,
            "confidence": round(max(0.0, min(1.0, float(self.confidence))), 4),
            "reason_code": self.reason_code,
            "source_refs": list(self.source_refs),
            "target_refs": list(self.target_refs),
            "evidence_refs": list(self.evidence_refs),
            "payload": dict(self.payload),
            "title": self.title,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        body["proposal_id"] = self.proposal_id or stable_id(
            "mprop",
            {
                "processor_id": self.processor_id,
                "action": self.action,
                "source_refs": body["source_refs"],
                "payload_digest": digest(body["payload"]),
                "reason_code": self.reason_code,
            },
        )
        return body


@dataclass(frozen=True)
class SemanticFacet:
    """Domain-normalized semantic shape used for generic edge proposals.

    Domain processors may extract these facets from arbitrary atom payloads.
    AMOS compares only the normalized fields here; it does not need to know the
    domain meaning of a control, metric, asset, task, or outcome label.
    """

    atom_ref: str
    subject: str
    intent: str = ""
    outcome: str = ""
    outcome_direction: str = "neutral"
    confidence: float = 0.5
    evidence_refs: tuple[str, ...] = ()
    controls: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    time_index: int | float | str | None = None
    semantic_context_key: str = ""
    scope: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    facet_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        body = {
            "atom_ref": str(self.atom_ref),
            "subject": _normalize_facet_text(self.subject),
            "intent": _normalize_facet_text(self.intent),
            "outcome": _normalize_facet_text(self.outcome),
            "outcome_direction": _normalize_outcome_direction(self.outcome_direction),
            "confidence": round(max(0.0, min(1.0, float(self.confidence))), 4),
            "evidence_refs": [str(ref) for ref in self.evidence_refs],
            "controls": dict(self.controls),
            "metrics": dict(self.metrics),
            "time_index": self.time_index,
            "semantic_context_key": _normalize_facet_text(
                self.semantic_context_key
            ),
            "scope": dict(self.scope),
            "attributes": dict(self.attributes),
            "schema_version": self.schema_version,
        }
        body["facet_id"] = self.facet_id or stable_id(
            "facet",
            {
                "atom_ref": body["atom_ref"],
                "subject": body["subject"],
                "intent": body["intent"],
                "outcome": body["outcome"],
                "outcome_direction": body["outcome_direction"],
                "controls": body["controls"],
                "metrics": body["metrics"],
                "time_index": body["time_index"],
                "semantic_context_key": body["semantic_context_key"],
            },
        )
        return body


class MaintenanceProcessor(Protocol):
    processor_id: str
    processor_version: str

    def supports(self, window: EvidenceWindow) -> bool:
        ...

    def propose(self, window: EvidenceWindow) -> list[MaintenanceProposal]:
        ...


def coerce_window_request(
    value: MaintenanceWindowRequest | Mapping[str, Any] | None,
) -> MaintenanceWindowRequest:
    if value is None:
        return MaintenanceWindowRequest()
    if isinstance(value, MaintenanceWindowRequest):
        return value
    if not isinstance(value, Mapping):
        raise ValidationError("maintenance window request must be a mapping")
    lifecycle_states = tuple(
        dict.fromkeys(str(item) for item in value.get("lifecycle_states", ()) if str(item))
    ) or ("active", "proposed")
    atom_types = tuple(
        dict.fromkeys(str(item) for item in value.get("atom_types", ()) if str(item))
    )
    profiles = tuple(
        dict.fromkeys(
            str(item)
            for item in value.get("graph_metadata_profiles", ())
            if str(item)
        )
    )
    max_atoms = value.get("max_atoms")
    max_evidence = value.get("max_evidence")
    return MaintenanceWindowRequest(
        lifecycle_states=lifecycle_states,
        atom_types=atom_types,
        graph_metadata_profiles=profiles,
        max_atoms=max(1, int(max_atoms)) if max_atoms not in (None, "") else None,
        include_graph_neighbors=bool(value.get("include_graph_neighbors", False)),
        include_evidence=bool(value.get("include_evidence", True)),
        include_events=bool(value.get("include_events", True)),
        include_retrieval_outcomes=bool(
            value.get("include_retrieval_outcomes", True)
        ),
        max_evidence=(
            max(0, int(max_evidence))
            if max_evidence not in (None, "")
            else None
        ),
    )


class ProcessorRegistry:
    """In-process registry for deterministic maintenance processor packs."""

    def __init__(self, processors: Sequence[MaintenanceProcessor] | None = None):
        self._processors: dict[str, MaintenanceProcessor] = {}
        for processor in processors or []:
            self.register(processor)

    def register(self, processor: MaintenanceProcessor) -> None:
        _validate_processor(processor)
        self._processors[processor.processor_id] = processor

    def get(self, processor_id: str) -> MaintenanceProcessor | None:
        return self._processors.get(processor_id)

    def list(self) -> list[dict[str, str]]:
        return [
            {
                "processor_id": processor.processor_id,
                "processor_version": processor.processor_version,
            }
            for processor in self._processors.values()
        ]

    def resolve(
        self, *, processor_ids: Sequence[str] | None = None
    ) -> list[MaintenanceProcessor]:
        """Resolve registered processors before an evidence window is built."""

        if processor_ids:
            return [
                processor
                for processor_id in processor_ids
                if (processor := self.get(processor_id)) is not None
            ]
        return list(self._processors.values())

    def select(
        self,
        window: EvidenceWindow,
        *,
        processor_ids: Sequence[str] | None = None,
    ) -> list[MaintenanceProcessor]:
        selected = self.resolve(processor_ids=processor_ids)
        return [processor for processor in selected if processor.supports(window)]


def maintenance_hints_from_atom(atom: Mapping[str, Any]) -> dict[str, Any]:
    """Return producer hints that are safe for generic workset selection.

    Hints are advisory metadata, never canonical claims or mutation authority.
    Unknown values are retained so domain processors can interpret their own
    profiles without adding domain branches to AMOS.
    """

    payload = atom.get("payload")
    if not isinstance(payload, Mapping):
        return {}
    hints = payload.get("maintenance_hints")
    return dict(hints) if isinstance(hints, Mapping) else {}


def maintenance_source_refs(atom: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect explicit source coverage refs without inferring from prose."""

    payload = atom.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    hints = maintenance_hints_from_atom(atom)
    refs: list[str] = []
    for value in (
        atom.get("evidence_refs"),
        payload.get("source_refs"),
        payload.get("maintenance_source_refs"),
        payload.get("reviewed_refs"),
        hints.get("source_refs"),
    ):
        refs.extend(_structured_refs(value))
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def covered_source_refs(atoms: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return source refs explicitly covered by active derived memories."""

    covered: set[str] = set()
    for atom in atoms:
        if atom.get("deleted") or atom.get("lifecycle_state") != "active":
            continue
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if not (
            payload.get("maintenance_proposal_id")
            or payload.get("created_by_processor")
            or payload.get("distillation_type")
        ):
            continue
        covered.update(maintenance_source_refs(atom))
    return covered


def group_maintenance_cohorts(
    atoms: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Group atoms only by explicit producer-supplied stable cohort keys."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for atom in atoms:
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        hints = maintenance_hints_from_atom(atom)
        retention = payload.get("proposal_retention")
        retention = retention if isinstance(retention, Mapping) else {}
        key = str(
            hints.get("cluster_key")
            or hints.get("cohort_key")
            or retention.get("deduplication_key")
            or ""
        ).strip()
        if key:
            grouped.setdefault(key, []).append(atom)
    return {
        key: tuple(sorted(values, key=lambda item: str(item.get("id") or "")))
        for key, values in sorted(grouped.items())
    }


def evidence_diversity(
    refs: Sequence[str], evidence: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Summarize how many independent evidence sources support a proposal."""

    wanted = {str(ref) for ref in refs if str(ref)}
    records = [item for item in evidence if str(item.get("evidence_id") or "") in wanted]
    source_refs = {str(item.get("source_ref") or "") for item in records if item.get("source_ref")}
    source_types = {str(item.get("source_type") or "") for item in records if item.get("source_type")}
    return {
        "referenced": len(wanted),
        "resolved": len(records),
        "independent_sources": len(source_refs),
        "source_types": sorted(source_types),
    }


def derived_memory_proposal(
    *,
    processor_id: str,
    processor_version: str,
    reason_code: str,
    source_refs: Sequence[str],
    evidence_refs: Sequence[str],
    atom_type: str,
    atom_payload: Mapping[str, Any],
    scope: Mapping[str, Any],
    confidence: float,
    title: str,
    risk_level: str = "low",
    supersedes: Sequence[str] = (),
) -> MaintenanceProposal:
    """Build an idempotent processor-derived atom proposal."""

    sources = tuple(dict.fromkeys(str(ref) for ref in source_refs if str(ref)))
    evidence = tuple(dict.fromkeys(str(ref) for ref in evidence_refs if str(ref)))
    superseded = tuple(dict.fromkeys(str(ref) for ref in supersedes if str(ref)))
    score = round(max(0.0, min(1.0, float(confidence))), 4)
    atom_id = stable_id(
        "atom",
        {
            "processor_id": processor_id,
            "reason_code": reason_code,
            "source_refs": sources,
            "supersedes": superseded,
            "payload": dict(atom_payload),
            "scope": dict(scope),
        },
    )
    payload = dict(atom_payload)
    payload.setdefault("created_by_processor", processor_id)
    payload.setdefault("maintenance_source_refs", list(sources))
    return MaintenanceProposal(
        processor_id=processor_id,
        processor_version=processor_version,
        action="add_atom",
        risk_level=risk_level,
        confidence=score,
        reason_code=reason_code,
        source_refs=sources,
        evidence_refs=evidence,
        target_refs=(atom_id,),
        payload={
            "atom": {
                "id": atom_id,
                "type": str(atom_type),
                "payload": payload,
                "evidence_refs": list(evidence),
                "scope": dict(scope),
                "confidence": {"level": _confidence_level(score), "score": score},
                "lifecycle_state": "active",
                "supersedes": list(superseded),
            }
        },
        title=title,
    )


class GenericMaintenanceProcessor:
    """Proposal adapter around the built-in deterministic SMP."""

    processor_id = "amos.maintenance.generic.v1"
    processor_version = "amos.maintenance.generic.v1"

    def __init__(self, smp: SemanticMaintenanceProcessor | None = None):
        self.smp = smp or SemanticMaintenanceProcessor()

    def supports(self, window: EvidenceWindow) -> bool:
        return bool(window.atoms)

    def propose(self, window: EvidenceWindow) -> list[MaintenanceProposal]:
        proposals: list[MaintenanceProposal] = []
        for atom in window.atoms:
            if atom.get("deleted") or atom.get("lifecycle_state") != "active":
                continue
            health = self.smp.propose_health(atom)
            action = health["recommended_action"]
            health_status = action.get("health_status")
            if action.get("type") != "mark_health" or health_status == atom.get("health_status"):
                continue
            proposals.append(
                MaintenanceProposal(
                    processor_id=self.processor_id,
                    processor_version=self.processor_version,
                    action="mark_health",
                    risk_level=health["risk_level"],
                    confidence=health["confidence"],
                    reason_code=health["reason_code"],
                    source_refs=(str(atom["id"]),),
                    target_refs=(str(atom["id"]),),
                    payload={
                        "health_status": health_status,
                        "smp_output_ref": health["output_id"],
                    },
                    title=f"Mark {atom['id']} as {health_status}",
                )
            )
        for cluster in self.smp.cluster(window.atoms):
            refs = tuple(str(ref) for ref in cluster["input_refs"])
            proposals.append(
                MaintenanceProposal(
                    processor_id=self.processor_id,
                    processor_version=self.processor_version,
                    action="review_cluster",
                    risk_level=cluster["risk_level"],
                    confidence=cluster["confidence"],
                    reason_code=cluster["reason_code"],
                    source_refs=refs,
                    target_refs=refs,
                    payload={
                        "recommended_action": dict(cluster["recommended_action"]),
                        "smp_output_ref": cluster["output_id"],
                    },
                    title="Review near-duplicate memory cluster",
                )
            )
        for conflict in self.smp.detect_conflicts(window.atoms):
            refs = tuple(str(ref) for ref in conflict["input_refs"])
            proposals.append(
                MaintenanceProposal(
                    processor_id=self.processor_id,
                    processor_version=self.processor_version,
                    action="review_conflict",
                    risk_level=conflict["risk_level"],
                    confidence=conflict["confidence"],
                    reason_code=conflict["reason_code"],
                    source_refs=refs,
                    target_refs=refs,
                    payload={
                        "recommended_action": dict(conflict["recommended_action"]),
                        "smp_output_ref": conflict["output_id"],
                    },
                    title="Review candidate contradiction",
                )
            )
        proposals.extend(
            canonical_relation_proposals_from_atoms(
                window.atoms,
                existing_edges=window.edges,
            )
        )
        return proposals

    def extract_facets(self, window: EvidenceWindow) -> list[SemanticFacet]:
        """Read the canonical facet contract without domain-specific code."""

        return semantic_facets_from_atoms(window.atoms)


def coerce_semantic_facet(value: SemanticFacet | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a processor-emitted facet into the stable wire shape."""

    if isinstance(value, SemanticFacet):
        facet = value.to_dict()
    elif isinstance(value, Mapping):
        facet = SemanticFacet(
            atom_ref=str(value.get("atom_ref", "")),
            subject=str(value.get("subject", "")),
            intent=str(value.get("intent", "")),
            outcome=str(value.get("outcome", "")),
            outcome_direction=str(value.get("outcome_direction", "neutral")),
            confidence=float(value.get("confidence", 0.5) or 0.5),
            evidence_refs=tuple(str(ref) for ref in value.get("evidence_refs", ())),
            controls=dict(value.get("controls") or {}),
            metrics=dict(value.get("metrics") or {}),
            time_index=value.get("time_index"),
            semantic_context_key=str(
                value.get("semantic_context_key")
                or value.get("domain_key")
                or value.get("context_key")
                or ""
            ),
            scope=dict(value.get("scope") or {}),
            attributes=dict(value.get("attributes") or {}),
            facet_id=value.get("facet_id"),
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
        ).to_dict()
    else:
        raise ValidationError("semantic facet must be a mapping or SemanticFacet")
    if not facet["atom_ref"]:
        raise ValidationError("semantic facet atom_ref is required")
    if not facet["subject"]:
        raise ValidationError("semantic facet subject is required")
    return facet


def semantic_facets_from_atoms(
    atoms: Sequence[Mapping[str, Any]],
) -> list[SemanticFacet]:
    """Extract producer-normalized facets from active canonical atoms.

    Proposed atoms deliberately remain isolated.  Their facet metadata is
    retained and becomes eligible on a later maintenance pass after an
    authorized lifecycle promotion.
    """

    facets: list[SemanticFacet] = []
    for atom in atoms:
        if atom.get("deleted") or atom.get("lifecycle_state") != "active":
            continue
        payload = atom.get("payload")
        if not isinstance(payload, Mapping):
            continue
        raw_facets = payload.get("semantic_facets")
        if not isinstance(raw_facets, list):
            continue
        atom_ref = str(atom.get("id") or "")
        if not atom_ref:
            continue
        atom_confidence = atom.get("confidence")
        if isinstance(atom_confidence, Mapping):
            default_confidence = float(atom_confidence.get("score", 0.5) or 0.5)
        else:
            default_confidence = 0.5
        default_evidence = tuple(
            str(ref) for ref in atom.get("evidence_refs", []) if str(ref)
        )
        for raw in raw_facets:
            if not isinstance(raw, Mapping):
                continue
            facets.append(
                SemanticFacet(
                    atom_ref=atom_ref,
                    subject=str(raw.get("subject") or ""),
                    intent=str(raw.get("intent") or ""),
                    outcome=str(raw.get("outcome") or ""),
                    outcome_direction=str(
                        raw.get("outcome_direction") or "neutral"
                    ),
                    confidence=float(
                        raw.get("confidence", default_confidence)
                        or default_confidence
                    ),
                    evidence_refs=tuple(
                        str(ref)
                        for ref in raw.get("evidence_refs", default_evidence)
                        if str(ref)
                    ),
                    controls=dict(raw.get("controls") or {}),
                    metrics=dict(raw.get("metrics") or {}),
                    time_index=raw.get("time_index"),
                    semantic_context_key=str(
                        raw.get("semantic_context_key")
                        or raw.get("domain_key")
                        or raw.get("context_key")
                        or (raw.get("attributes") or {}).get(
                            "semantic_context_key"
                        )
                        or (raw.get("attributes") or {}).get("domain_key")
                        or ""
                    ),
                    scope=dict(raw.get("scope") or atom.get("scope") or {}),
                    attributes=dict(raw.get("attributes") or {}),
                    facet_id=(str(raw["facet_id"]) if raw.get("facet_id") else None),
                )
            )
    return facets


def canonical_relation_proposals_from_atoms(
    atoms: Sequence[Mapping[str, Any]],
    *,
    existing_edges: Sequence[Mapping[str, Any]] = (),
) -> list[MaintenanceProposal]:
    """Build graph proposals from explicit, domain-neutral atom structure."""

    active_atoms = {
        str(atom.get("id") or ""): atom
        for atom in atoms
        if atom.get("id")
        and not atom.get("deleted")
        and atom.get("lifecycle_state") == "active"
    }
    existing_keys = {
        (
            str(edge.get("source_ref") or ""),
            str(edge.get("target_ref") or ""),
            str(edge.get("relation") or ""),
        )
        for edge in existing_edges
        if not edge.get("deleted") and edge.get("lifecycle_state", "active") == "active"
    }
    proposed_keys: set[tuple[str, str, str]] = set()
    proposals: list[MaintenanceProposal] = []

    def add(
        owner: Mapping[str, Any],
        source_ref: Any,
        target_ref: Any,
        relation: str,
        *,
        reason_code: str,
        evidence_refs: Sequence[str] = (),
        confidence: float | Mapping[str, Any] | None = None,
        explicit: bool = False,
    ) -> None:
        owner_ref = str(owner.get("id") or "")
        source = owner_ref if str(source_ref or "$self") == "$self" else str(source_ref or "")
        target = owner_ref if str(target_ref or "") == "$self" else str(target_ref or "")
        if (
            not source
            or not target
            or source == target
            or source not in active_atoms
            or target not in active_atoms
        ):
            return
        if explicit and owner_ref not in {source, target}:
            return
        normalized_relation = normalize_relation(relation)
        key = (source, target, normalized_relation)
        if key in existing_keys or key in proposed_keys:
            return
        proposed_keys.add(key)
        if isinstance(confidence, Mapping):
            score = float(confidence.get("score", 0.75) or 0.75)
        elif isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            score = float(confidence)
        else:
            score = min(
                float(active_atoms[source].get("confidence", {}).get("score", 0.75)),
                float(active_atoms[target].get("confidence", {}).get("score", 0.75)),
            )
        score = round(max(0.0, min(1.0, score)), 4)
        risk_level = (
            "low" if normalized_relation in LOW_RISK_EXPLICIT_RELATIONS else "medium"
        )
        refs = tuple(
            dict.fromkeys(
                [
                    *[str(ref) for ref in owner.get("evidence_refs", []) if str(ref)],
                    *[str(ref) for ref in evidence_refs if str(ref)],
                ]
            )
        )
        proposals.append(
            MaintenanceProposal(
                processor_id=GENERIC_GRAPH_PROCESSOR_ID,
                processor_version=GENERIC_GRAPH_PROCESSOR_VERSION,
                action="add_edge",
                risk_level=risk_level,
                confidence=score,
                reason_code=reason_code,
                source_refs=(source, target),
                target_refs=(source, target),
                evidence_refs=refs,
                payload={
                    "edge": {
                        "source_ref": source,
                        "target_ref": target,
                        "relation": normalized_relation,
                        "scope": dict(owner.get("scope") or {}),
                        "confidence": {
                            "level": _confidence_level(score),
                            "score": score,
                        },
                        "evidence_refs": list(refs),
                        "derivation": {
                            "kind": (
                                "explicit_structural"
                                if explicit
                                else "canonical_structural"
                            ),
                            "processor_id": GENERIC_GRAPH_PROCESSOR_ID,
                            "processor_version": GENERIC_GRAPH_PROCESSOR_VERSION,
                            "reason_code": reason_code,
                            "source_refs": [owner_ref],
                        },
                    },
                    "canonical_owner_ref": owner_ref,
                },
                title=f"Build canonical {normalized_relation} edge",
            )
        )

    for atom in active_atoms.values():
        atom_ref = str(atom["id"])
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        for ref in _structured_refs(atom.get("supersedes")):
            add(
                atom,
                atom_ref,
                ref,
                "rel:supersedes",
                reason_code="canonical_atom_supersedes",
            )
        for ref in _structured_refs(payload.get("source_refs")):
            add(
                atom,
                atom_ref,
                ref,
                "rel:derived_from",
                reason_code="canonical_payload_source_ref",
            )
        for ref in _structured_refs(payload.get("memory_references")):
            add(
                atom,
                atom_ref,
                ref,
                "rel:uses",
                reason_code="canonical_payload_memory_reference",
            )
        directive_ref = payload.get("directive_atom_ref") or payload.get(
            "source_directive_ref"
        )
        if atom.get("type") == "agentic_trace" and directive_ref:
            add(
                atom,
                directive_ref,
                atom_ref,
                "rel:produced_outcome",
                reason_code="canonical_directive_outcome",
            )
        for raw in payload.get("graph_relations", []):
            if not isinstance(raw, Mapping):
                continue
            add(
                atom,
                raw.get("source_ref", "$self"),
                raw.get("target_ref"),
                str(raw.get("relation") or ""),
                reason_code="canonical_explicit_relation",
                evidence_refs=tuple(raw.get("evidence_refs") or ()),
                confidence=raw.get("confidence"),
                explicit=True,
            )
    return proposals


def _structured_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        ref = value.get("atom_ref") or value.get("id") or value.get("ref")
        return [str(ref)] if ref else []
    if not isinstance(value, Sequence):
        return []
    refs: list[str] = []
    for item in value:
        refs.extend(_structured_refs(item))
    return list(dict.fromkeys(refs))


def semantic_relation_proposals_from_facets(
    facets: Sequence[Mapping[str, Any]],
    *,
    existing_edges: Sequence[Mapping[str, Any]] = (),
    processor_id: str = SEMANTIC_RELATION_PROCESSOR_ID,
    processor_version: str = SEMANTIC_RELATION_PROCESSOR_VERSION,
    max_new_relations_per_facet: int = 4,
) -> list[MaintenanceProposal]:
    """Convert facets into a bounded set of generic graph edge proposals."""

    normalized = [coerce_semantic_facet(facet) for facet in facets]
    normalized = sorted(
        {
            facet["facet_id"]: facet
            for facet in normalized
        }.values(),
        key=lambda facet: (facet["subject"], facet["intent"], facet["atom_ref"]),
    )
    existing_keys = {
        (
            str(edge.get("source_ref", "")),
            str(edge.get("target_ref", "")),
            str(edge.get("relation", "")),
        )
        for edge in existing_edges
        if not edge.get("deleted")
    }
    proposals: list[MaintenanceProposal] = []
    seen: set[tuple[str, str, str]] = set()
    new_degree: dict[str, int] = {}
    for source_ref, target_ref, _relation in existing_keys:
        new_degree[source_ref] = new_degree.get(source_ref, 0) + 1
        new_degree[target_ref] = new_degree.get(target_ref, 0) + 1
    degree_limit = max(1, int(max_new_relations_per_facet or 1))
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            if left["atom_ref"] == right["atom_ref"]:
                continue
            if (
                new_degree.get(str(left["atom_ref"]), 0) >= degree_limit
                or new_degree.get(str(right["atom_ref"]), 0) >= degree_limit
            ):
                continue
            pair = _semantic_relation_for_pair(left, right)
            if pair is None:
                continue
            relation, source_ref, target_ref, risk_level, reason_code, confidence = pair
            key = (source_ref, target_ref, relation)
            reverse_key = (target_ref, source_ref, relation)
            if key in existing_keys or key in seen:
                continue
            if relation in {"rel:supports", "rel:similar_to"} and reverse_key in existing_keys:
                continue
            seen.add(key)
            new_degree[source_ref] = new_degree.get(source_ref, 0) + 1
            new_degree[target_ref] = new_degree.get(target_ref, 0) + 1
            scope = _common_scope(left.get("scope", {}), right.get("scope", {}))
            evidence_refs = tuple(
                dict.fromkeys(
                    [
                        *[str(ref) for ref in left.get("evidence_refs", [])],
                        *[str(ref) for ref in right.get("evidence_refs", [])],
                    ]
                )
            )
            proposals.append(
                MaintenanceProposal(
                    processor_id=processor_id,
                    processor_version=processor_version,
                    action="add_edge",
                    risk_level=risk_level,
                    confidence=confidence,
                    reason_code=reason_code,
                    source_refs=(source_ref, target_ref),
                    target_refs=(source_ref, target_ref),
                    evidence_refs=evidence_refs,
                    payload={
                        "edge": {
                            "source_ref": source_ref,
                            "target_ref": target_ref,
                            "relation": relation,
                            "scope": scope,
                            "confidence": {
                                "level": _confidence_level(confidence),
                                "score": confidence,
                            },
                            "evidence_refs": list(evidence_refs),
                            "derivation": {
                                "kind": "facet_derived_association",
                                "processor_id": processor_id,
                                "processor_version": processor_version,
                                "reason_code": reason_code,
                                "source_refs": [
                                    str(left["atom_ref"]),
                                    str(right["atom_ref"]),
                                ],
                                "facet_refs": [
                                    str(left["facet_id"]),
                                    str(right["facet_id"]),
                                ],
                            },
                        },
                        "facets": [left, right],
                    },
                    title=f"Add {relation} edge between semantic facets",
                )
            )
    return proposals


def _semantic_relation_for_pair(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[str, str, str, str, str, float] | None:
    if left.get("subject") != right.get("subject"):
        return None
    left_context = str(left.get("semantic_context_key") or "")
    right_context = str(right.get("semantic_context_key") or "")
    if (left_context or right_context) and left_context != right_context:
        return None
    left_role = str((left.get("attributes") or {}).get("semantic_role") or "")
    right_role = str((right.get("attributes") or {}).get("semantic_role") or "")
    if (
        left_role in {"activity", "project_activity"}
        and right_role in {"activity", "project_activity"}
    ):
        # Sequential activity is temporal lineage, not evidence that one result
        # supports another. Producers should express exact causal/part-of links
        # explicitly; the generic associator must not invent support here.
        return None
    confidence = round(
        min(float(left.get("confidence", 0.5)), float(right.get("confidence", 0.5))), 4
    )
    same_intent = bool(left.get("intent")) and left.get("intent") == right.get("intent")
    same_controls = (
        bool(left.get("controls"))
        and bool(right.get("controls"))
        and _canonical_or_empty(left.get("controls"))
        == _canonical_or_empty(right.get("controls"))
    )
    if same_intent or same_controls:
        direction = _normalize_outcome_direction(left.get("outcome_direction"))
        other_direction = _normalize_outcome_direction(right.get("outcome_direction"))
        if direction in {"positive", "negative"} and other_direction in {
            "positive",
            "negative",
        }:
            if direction == other_direction:
                return (
                    "rel:supports",
                    str(left["atom_ref"]),
                    str(right["atom_ref"]),
                    "low",
                    "same_subject_same_outcome_direction",
                    confidence,
                )
            return (
                "rel:contradicts",
                str(left["atom_ref"]),
                str(right["atom_ref"]),
                "low",
                "same_subject_opposite_outcome_direction",
                confidence,
            )
        return (
            "rel:similar_to",
            str(left["atom_ref"]),
            str(right["atom_ref"]),
            "low",
            "same_subject_similar_intent_or_controls",
            confidence,
        )
    if _is_later_stronger(left, right):
        supersession_confidence = round(float(left.get("confidence", confidence)), 4)
        return (
            "rel:supersedes",
            str(left["atom_ref"]),
            str(right["atom_ref"]),
            "low",
            "newer_stronger_same_subject_evidence",
            supersession_confidence,
        )
    if _is_later_stronger(right, left):
        supersession_confidence = round(float(right.get("confidence", confidence)), 4)
        return (
            "rel:supersedes",
            str(right["atom_ref"]),
            str(left["atom_ref"]),
            "low",
            "newer_stronger_same_subject_evidence",
            supersession_confidence,
        )
    return None


def _normalize_facet_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_outcome_direction(value: Any) -> str:
    direction = str(value or "neutral").strip().lower()
    aliases = {
        "supported": "positive",
        "success": "positive",
        "improved": "positive",
        "positive": "positive",
        "failed": "negative",
        "failure": "negative",
        "regressed": "negative",
        "negative": "negative",
        "mixed": "mixed",
        "neutral": "neutral",
    }
    return aliases.get(direction, "neutral")


def _common_scope(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> dict[str, Any]:
    left = dict(left or {})
    right = dict(right or {})
    return {key: value for key, value in left.items() if right.get(key) == value}


def _canonical_or_empty(value: Any) -> str:
    if not value:
        return ""
    return digest(value)


def _is_later_stronger(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    candidate_time = candidate.get("time_index")
    baseline_time = baseline.get("time_index")
    if candidate_time is None or baseline_time is None:
        return False
    try:
        if float(candidate_time) <= float(baseline_time):
            return False
    except (TypeError, ValueError):
        if str(candidate_time) <= str(baseline_time):
            return False
    return float(candidate.get("confidence", 0.5)) >= float(
        baseline.get("confidence", 0.5)
    ) + 0.1


def _confidence_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.7:
        return "medium-high"
    if score >= 0.45:
        return "medium"
    if score >= 0.3:
        return "low-medium"
    return "low"


def default_processor_registry(
    smp: SemanticMaintenanceProcessor | None = None,
    *,
    processors: Sequence[MaintenanceProcessor] | None = None,
    processor_paths: Sequence[str] | None = None,
) -> ProcessorRegistry:
    registry = ProcessorRegistry([GenericMaintenanceProcessor(smp)])
    for processor in processors or []:
        registry.register(processor)
    for processor_path in processor_paths or []:
        registry.register(load_maintenance_processor(processor_path))
    return registry


def load_maintenance_processor(import_path: str) -> MaintenanceProcessor:
    """Load an external maintenance processor from ``module:attribute``.

    The imported attribute may be a processor instance, a processor class with a
    no-argument constructor, or a no-argument factory returning a processor.
    Domain packages can use this hook without adding domain code to AMOS.
    """

    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValidationError(
            "processor import path must use the form 'module:attribute'"
        )
    try:
        module = importlib.import_module(module_name)
        target = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise ValidationError(
            f"unable to load maintenance processor '{import_path}': {exc}"
        ) from exc
    processor = target() if inspect.isclass(target) else target
    if callable(processor) and not hasattr(processor, "processor_id"):
        processor = processor()
    return _validate_processor(processor)


def _validate_processor(processor: Any) -> MaintenanceProcessor:
    processor_id = getattr(processor, "processor_id", None)
    processor_version = getattr(processor, "processor_version", None)
    if not processor_id or not isinstance(processor_id, str):
        raise ValidationError("maintenance processor must define processor_id")
    if not processor_version or not isinstance(processor_version, str):
        raise ValidationError("maintenance processor must define processor_version")
    if not callable(getattr(processor, "supports", None)):
        raise ValidationError("maintenance processor must define supports(window)")
    if not callable(getattr(processor, "propose", None)):
        raise ValidationError("maintenance processor must define propose(window)")
    return processor


def proposal_is_auto_committable(proposal: Mapping[str, Any]) -> bool:
    return (
        proposal.get("action") in LOW_RISK_PROPOSAL_ACTIONS
        and proposal.get("risk_level") == "low"
    )
