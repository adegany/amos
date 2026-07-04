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
from .schemas import SCHEMA_VERSION, digest, stable_id, utc_now
from .smp import SemanticMaintenanceProcessor


LOW_RISK_PROPOSAL_ACTIONS = {"add_atom"}


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
            "generated_at": self.generated_at,
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


class MaintenanceProcessor(Protocol):
    processor_id: str
    processor_version: str

    def supports(self, window: EvidenceWindow) -> bool:
        ...

    def propose(self, window: EvidenceWindow) -> list[MaintenanceProposal]:
        ...


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

    def select(
        self,
        window: EvidenceWindow,
        *,
        processor_ids: Sequence[str] | None = None,
    ) -> list[MaintenanceProcessor]:
        if processor_ids:
            selected = [
                processor
                for processor_id in processor_ids
                if (processor := self.get(processor_id)) is not None
            ]
        else:
            selected = list(self._processors.values())
        return [processor for processor in selected if processor.supports(window)]


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
        return proposals


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
