"""Coherent, revision-bound reasoning frames and demand-paged memory views.

The objects produced here are generated views over canonical AMOS atoms and
edges.  They are deliberately not persisted as MemoryAtoms: a frame is scoped
to one reasoning request, while a page descriptor is a deterministic capability
that the trusted runtime retains and presents again when deeper detail is needed.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ._service_support import access_visible, scope_visible
from .errors import StaleFrameError, ValidationError
from .schemas import (
    SCHEMA_VERSION,
    canonical_json,
    digest,
    normalize_scope,
    stable_id,
    utc_now,
)


FRAME_DEPTH_ALIASES = {
    "orientation": "orientation",
    "working_frame": "working_frame",
    "focused": "working_frame",
    "evidence": "evidence",
    "supporting": "evidence",
}
PAGE_DEPTH_ALIASES = {
    "focused": "focused",
    "working_frame": "focused",
    "orientation": "focused",
    "supporting": "supporting",
    "evidence": "supporting",
}

# These relations change how an atom must be interpreted.  Their visible
# endpoints therefore travel together as one coherent unit rather than
# competing for independent ranking slots.
MANDATORY_CONTEXT_RELATIONS = {
    "rel:applies_to",
    "rel:caused_by",
    "rel:constrained_by",
    "rel:contradicts",
    "rel:corrected_by",
    "rel:decided",
    "rel:depends_on",
    "rel:derived_from",
    "rel:forbids",
    "rel:part_of",
    "rel:produced_outcome",
    "rel:requires",
    "rel:satisfied_commitment",
    "rel:supersedes",
}

# These relations are useful deeper context, but do not by themselves prove
# that two atoms form one indivisible conclusion/history unit.
SUPPORTING_CONTEXT_RELATIONS = {
    "rel:acted_on",
    "rel:attributed_to",
    "rel:avoids",
    "rel:currently_available",
    "rel:currently_denied",
    "rel:has_capability",
    "rel:has_limitation",
    "rel:miscalibrated_on",
    "rel:owns",
    "rel:prefers",
    "rel:shared_responsibility_for",
    "rel:similar_to",
    "rel:supports",
    "rel:uses",
    "rel:working_on",
}

DEFAULT_FRAME_TOKEN_BUDGET = 2_000
DEFAULT_PAGE_TOKEN_BUDGET = 2_500

# Only these top-level task-context fields may drive semantic visibility.  They
# are populated by the trusted runtime, not inferred from free-form model text.
TASK_CONTEXT_SEMANTIC_FIELDS = {
    "human_id": ("human_id",),
    "project_id": ("project_id",),
    "project_thread_id": ("project_thread_id", "conversation_id"),
}
ATOM_SEMANTIC_FIELDS = {
    "human_id": ("human_id", "human", "user_id"),
    "project_id": ("project_id", "project"),
    "project_thread_id": (
        "project_thread_id",
        "conversation_id",
        "thread_id",
        "thread",
    ),
}


class ReasoningFrameService:
    """Compile coherent memory units and reload their deeper supporting pages."""

    def __init__(
        self,
        store: Any,
        smp: Any,
        access: Any,
        indexes: Any,
        graph: Any,
        retrieval: Any,
        capacity: Any,
        policy_runner: Any,
    ) -> None:
        self.store = store
        self.smp = smp
        self.indexes = indexes
        self.graph = graph
        self.retrieval = retrieval
        self._mark_foreground_activity = access._mark_foreground_activity
        self._capacity_pressure_mode = capacity._capacity_pressure_mode
        self.run_memory_policy = policy_runner

    def compile_memory_frame(
        self,
        *,
        need: str,
        purpose: str,
        depth: str = "working_frame",
        task_context: Mapping[str, Any] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        need = self._required_text(need, "need")
        purpose = self._required_text(purpose, "purpose")
        requester = self._required_text(requester, "requester")
        target_processor = self._required_text(target_processor, "target_processor")
        canonical_depth = self._depth(depth, FRAME_DEPTH_ALIASES)
        task_context = self._mapping(task_context, "task_context")
        request_scope = normalize_scope(scope)
        semantic_scope = self._trusted_semantic_scope(task_context, request_scope)
        visibility_scope = self._visibility_scope(request_scope, semantic_scope)
        budget = self._budget(
            token_or_byte_budget,
            default_tokens=DEFAULT_FRAME_TOKEN_BUDGET,
        )
        self._mark_foreground_activity(requester)
        if run_policy:
            self.run_memory_policy(trigger="compile_memory_frame", scope=request_scope)

        last_start = self.store.memory_revision()
        last_end = last_start
        # A background policy worker may commit while the view is being read.
        # Rebuild once from the new head; never silently return a mixed revision.
        for _attempt in range(2):
            revision = self.store.memory_revision()
            frame = self._compile_at_revision(
                need=need,
                purpose=purpose,
                depth=canonical_depth,
                task_context=task_context,
                scope=request_scope,
                visibility_scope=visibility_scope,
                semantic_scope=semantic_scope,
                requester=requester,
                target_processor=target_processor,
                byte_budget=budget["bytes"],
                revision=revision,
            )
            current = self.store.memory_revision()
            if current == revision:
                return frame
            last_start, last_end = revision, current
        raise StaleFrameError(last_start, last_end)

    def load_memory_page(
        self,
        *,
        frame_id: str,
        revision: Mapping[str, Any],
        page: Mapping[str, Any],
        need: str | None = None,
        purpose: str | None = None,
        depth: str = "focused",
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        frame_id = self._required_text(frame_id, "frame_id")
        requester = self._required_text(requester, "requester")
        target_processor = self._required_text(target_processor, "target_processor")
        expected_revision = self._revision(revision)
        descriptor = self._mapping(page, "page")
        canonical_depth = self._depth(depth, PAGE_DEPTH_ALIASES)
        if need is not None:
            need = self._required_text(need, "need")
        if purpose is not None:
            purpose = self._required_text(purpose, "purpose")
        request_scope = normalize_scope(scope)
        budget = self._budget(
            token_or_byte_budget,
            default_tokens=DEFAULT_PAGE_TOKEN_BUDGET,
        )
        self._mark_foreground_activity(requester)
        if run_policy:
            self.run_memory_policy(trigger="load_memory_page", scope=request_scope)

        current = self.store.memory_revision()
        if current != expected_revision:
            raise StaleFrameError(expected_revision, current)
        self._validate_page_descriptor(
            descriptor,
            frame_id=frame_id,
            revision=expected_revision,
        )
        semantic_scope = self._descriptor_semantic_scope(descriptor)
        visibility_scope = self._visibility_scope(request_scope, semantic_scope)

        requested_refs = (
            descriptor.get("focus_atom_refs", [])
            if canonical_depth == "focused"
            else descriptor.get("source_atom_refs", [])
        )
        atom_refs = list(
            dict.fromkeys(str(ref) for ref in requested_refs if str(ref).strip())
        )
        atoms: dict[str, dict[str, Any]] = {}
        omissions: list[dict[str, Any]] = []
        for atom_ref in atom_refs:
            atom = self.store.get_atom(atom_ref)
            reason = self._atom_omission_reason(
                atom,
                scope=visibility_scope,
                semantic_scope=semantic_scope,
                requester=requester,
                target_processor=target_processor,
            )
            if reason is not None:
                omissions.append({"atom_ref": atom_ref, "reason": reason})
                continue
            atoms[atom_ref] = dict(atom)

        relationship_refs = {
            str(ref) for ref in descriptor.get("relationship_refs", []) if str(ref)
        }
        edges: dict[str, dict[str, Any]] = {}
        for edge_ref in sorted(relationship_refs):
            edge = self.store.get_edge(edge_ref)
            if edge is None or not self._edge_visible(edge, visibility_scope):
                continue
            if str(edge.get("source_ref")) not in atoms or str(edge.get("target_ref")) not in atoms:
                continue
            edges[edge_ref] = dict(edge)

        components = self._connected_components(
            atom_refs=sorted(atoms),
            edges=edges,
        )
        units: list[dict[str, Any]] = []
        item_omissions: list[dict[str, Any]] = []
        for refs in components:
            unit, unit_omissions = self._build_unit(
                atom_refs=refs,
                atoms=atoms,
                edges={
                    edge_id: edge
                    for edge_id, edge in edges.items()
                    if str(edge.get("source_ref")) in refs
                    and str(edge.get("target_ref")) in refs
                },
                relevance_score=float(descriptor.get("relevance_score", 1.0) or 1.0),
                inclusion_reasons={ref: ["page_descriptor"] for ref in refs},
                requester=requester,
                target_processor=target_processor,
            )
            unit["page_id"] = descriptor["page_id"]
            units.append(unit)
            item_omissions.extend(unit_omissions)
        omissions.extend(item_omissions)
        units.sort(key=lambda unit: (-float(unit["relevance_score"]), unit["unit_id"]))
        selected_units = [dict(unit) for unit in units]
        budget_omitted = 0

        def build_page() -> dict[str, Any]:
            return self._page_payload(
                descriptor=descriptor,
                frame_id=frame_id,
                revision=expected_revision,
                depth=canonical_depth,
                need=need,
                purpose=purpose,
                units=selected_units,
                omissions=omissions,
                budget_omitted=budget_omitted,
                byte_budget=budget["bytes"],
                token_limit=budget.get("tokens"),
            )

        empty_page = self._page_payload(
            descriptor=descriptor,
            frame_id=frame_id,
            revision=expected_revision,
            depth=canonical_depth,
            need=need,
            purpose=purpose,
            units=[],
            omissions=omissions,
            budget_omitted=len(units),
            byte_budget=budget["bytes"],
            token_limit=budget.get("tokens"),
        )
        if self._json_bytes(self._finalize_token_estimate(empty_page)) > budget["bytes"]:
            raise ValidationError(
                "token_or_byte_budget is too small for the reasoning page envelope"
            )

        page = self._finalize_token_estimate(build_page())
        while self._json_bytes(page) > budget["bytes"] and selected_units:
            current = selected_units[-1]
            mode = str((current.get("compression") or {}).get("mode") or "none")
            if mode == "none":
                selected_units[-1] = self._compress_unit(current)
            elif mode == "essential_projection":
                selected_units[-1] = self._reference_unit(current)
            elif mode == "reference_summary":
                selected_units[-1] = self._bare_reference_unit(current)
            else:
                selected_units.pop()
                budget_omitted += 1
            page = self._finalize_token_estimate(build_page())
        if self._json_bytes(page) > budget["bytes"]:
            raise ValidationError(
                "token_or_byte_budget is too small for the reasoning page envelope"
            )

        end_revision = self.store.memory_revision()
        if end_revision != expected_revision:
            raise StaleFrameError(expected_revision, end_revision)
        return page

    def _page_payload(
        self,
        *,
        descriptor: Mapping[str, Any],
        frame_id: str,
        revision: Mapping[str, Any],
        depth: str,
        need: str | None,
        purpose: str | None,
        units: Sequence[Mapping[str, Any]],
        omissions: Sequence[Mapping[str, Any]],
        budget_omitted: int,
        byte_budget: int,
        token_limit: int | None,
    ) -> dict[str, Any]:
        sequence = sorted(
            [step for unit in units for step in unit.get("sequence", [])],
            key=lambda step: (
                str(step.get("observed_at") or ""),
                str(step.get("atom_ref") or ""),
            ),
        )
        source_refs = sorted(
            {
                str(ref)
                for unit in units
                for ref in unit.get("source_atom_refs", [])
                if str(ref)
            }
        )
        detailed_omissions = [dict(item) for item in omissions[:16]]
        if len(omissions) > len(detailed_omissions):
            detailed_omissions.append(
                {
                    "reason": "omission_detail_budgeted",
                    "omitted_detail_count": len(omissions) - len(detailed_omissions),
                }
            )
        if budget_omitted:
            detailed_omissions.append(
                {
                    "reason": "page_budget_exhausted",
                    "coherent_unit_count": budget_omitted,
                }
            )
        return {
            "status": "loaded",
            "page_id": descriptor["page_id"],
            "frame_id": frame_id,
            "schema_version": SCHEMA_VERSION,
            "revision": dict(revision),
            "generated_at": utc_now(),
            "page_type": descriptor["page_type"],
            "title": descriptor["title"],
            "summary": descriptor["summary"],
            "depth": depth,
            "need": need,
            "purpose": purpose,
            "units": [dict(unit) for unit in units],
            "sequence": sequence,
            "active_conclusion_refs": self._unit_refs(
                units, "active_conclusion_refs"
            ),
            "constraint_refs": self._unit_refs(units, "constraint_refs"),
            "conflict_refs": self._unit_refs(units, "conflict_refs"),
            "related_pages": list(descriptor.get("related_pages", [])),
            "source_atom_refs": source_refs,
            "omissions": detailed_omissions,
            "truncated": bool(
                budget_omitted or any(unit.get("truncated") for unit in units)
            ),
            "token_estimate": 0,
            "budget": {
                "limit_bytes": byte_budget,
                "limit_tokens": token_limit,
                "used_bytes": 0,
                "estimated_tokens": 0,
            },
            "provenance": {
                "store": getattr(self.store, "backend_name", "unknown"),
                "journal_head": revision["journal_head"],
                "compiler_profile_id": "amos.v1.reasoning_page.coherent",
                "descriptor_digest": descriptor["descriptor_digest"],
            },
        }

    def _compile_at_revision(
        self,
        *,
        need: str,
        purpose: str,
        depth: str,
        task_context: Mapping[str, Any],
        scope: Mapping[str, Any],
        visibility_scope: Mapping[str, Any],
        semantic_scope: Mapping[str, str],
        requester: str,
        target_processor: str,
        byte_budget: int,
        revision: Mapping[str, Any],
    ) -> dict[str, Any]:
        work_budget = self._reasoning_work_budget(byte_budget)
        seeds, discovery = self._discover_seed_atoms(
            need=need,
            purpose=purpose,
            depth=depth,
            task_context=task_context,
            scope=visibility_scope,
            semantic_scope=semantic_scope,
            work_budget=work_budget,
            requester=requester,
            target_processor=target_processor,
        )
        closure = self._coherent_closure(
            seeds=seeds,
            scope=visibility_scope,
            semantic_scope=semantic_scope,
            work_budget=work_budget,
            requester=requester,
            target_processor=target_processor,
        )
        units: list[dict[str, Any]] = []
        unit_omissions: list[dict[str, Any]] = []
        for component in closure["components"]:
            component_edges = {
                edge_id: closure["edges"][edge_id]
                for edge_id in component["edge_refs"]
                if edge_id in closure["edges"]
            }
            unit, omissions = self._build_unit(
                atom_refs=component["atom_refs"],
                atoms=closure["atoms"],
                edges=component_edges,
                relevance_score=component["relevance_score"],
                inclusion_reasons=closure["inclusion_reasons"],
                requester=requester,
                target_processor=target_processor,
            )
            unit["_page_atom_refs"] = component["page_atom_refs"]
            unit["_page_edge_refs"] = component["page_edge_refs"]
            units.append(unit)
            unit_omissions.extend(omissions)
        units.sort(key=lambda unit: (-float(unit["relevance_score"]), unit["unit_id"]))

        request = {
            "need": need,
            "purpose": purpose,
            "depth": depth,
            "task_context": dict(task_context),
            "scope": dict(scope),
            "requester": requester,
            "target_processor": target_processor,
            "token_or_byte_budget": {"bytes": byte_budget},
        }
        # The complete request participates in the stable frame identity, but
        # echoing it into the response spends the caller's reasoning-context
        # budget on text and trusted runtime state it already owns.  Retain a
        # compact, auditable binding instead.  The model-facing projection does
        # not receive this metadata, and operators can correlate the digest
        # with the original HTTP request and trace.
        response_request = {
            "request_digest": digest(request),
            "depth": depth,
            "requester": requester,
            "target_processor": target_processor,
            "token_or_byte_budget": {"bytes": byte_budget},
        }
        frame_id = stable_id(
            "frame",
            {
                "request": request,
                "revision": dict(revision),
                "units": [
                    {
                        "unit_id": unit["unit_id"],
                        "source_atom_refs": unit["source_atom_refs"],
                    }
                    for unit in units
                ],
            },
        )
        descriptors: list[dict[str, Any]] = []
        for unit in units:
            descriptor = self._page_descriptor(
                frame_id=frame_id,
                revision=revision,
                unit=unit,
                atoms=closure["page_atoms"],
                semantic_scope=semantic_scope,
            )
            descriptors.append(descriptor)
            unit.pop("_page_atom_refs", None)
            unit.pop("_page_edge_refs", None)

        # The complete response is budgeted, not merely its atom payloads.
        # Coherent units remain atomic, but a unit may use the same
        # preservation-aware projections as a loaded page: active conclusions,
        # constraints, commitments, conflicts, sequence and source refs survive
        # before peripheral detail is removed.  A compressed resident always
        # retains its full page descriptor.
        selected_units: list[dict[str, Any]] = []
        page_index: list[dict[str, Any]] = []
        omitted_unit_count = 0
        omitted_descriptor_count = 0

        def build_frame() -> dict[str, Any]:
            return self._frame_payload(
                frame_id=frame_id,
                revision=revision,
                request=response_request,
                task_context=task_context,
                selected_units=selected_units,
                page_index=page_index,
                unit_omissions=unit_omissions,
                omitted_unit_count=omitted_unit_count,
                omitted_descriptor_count=omitted_descriptor_count,
                closure=closure,
                discovery=discovery,
                seed_count=len(seeds),
                coherent_unit_count=len(units),
                byte_budget=byte_budget,
            )

        if self._json_bytes(self._finalize_token_estimate(build_frame())) > byte_budget:
            raise ValidationError(
                "token_or_byte_budget is too small for the reasoning frame envelope"
            )
        for index, (unit, descriptor) in enumerate(zip(units, descriptors)):
            # Leave room for the next independently coherent descriptor when
            # the budget permits it.  This is a one-item look-ahead, not an
            # atom/page quota: the loop still admits as many resident units and
            # descriptors as the serialized response budget can hold.
            reserve_descriptor = (
                descriptors[index + 1] if index + 1 < len(descriptors) else None
            )
            variants = [unit]
            compressed = self._compress_unit(unit)
            summarized = self._reference_unit(compressed)
            referenced = self._bare_reference_unit(summarized)
            variants.extend((compressed, summarized, referenced))

            admitted = False
            reserve_options = (
                (True, False) if reserve_descriptor is not None else (False,)
            )
            for preserve_next_descriptor in reserve_options:
                for candidate in variants:
                    compression_mode = str(
                        (candidate.get("compression") or {}).get("mode") or "none"
                    )
                    has_deeper_detail = bool(
                        set(descriptor["source_atom_refs"])
                        - set(candidate["source_atom_refs"])
                    )
                    needs_descriptor = bool(
                        compression_mode != "none"
                        or candidate.get("truncated")
                        or has_deeper_detail
                    )
                    selected_units.append(candidate)
                    if needs_descriptor:
                        page_index.append(descriptor)
                    reserved = False
                    if (
                        preserve_next_descriptor
                        and reserve_descriptor is not None
                        and all(
                            str(item.get("page_id") or "")
                            != str(reserve_descriptor.get("page_id") or "")
                            for item in page_index
                        )
                    ):
                        page_index.append(reserve_descriptor)
                        reserved = True
                    fits = (
                        self._json_bytes(
                            self._finalize_token_estimate(build_frame())
                        )
                        <= byte_budget
                    )
                    if reserved:
                        page_index.pop()
                    if fits:
                        admitted = True
                        break
                    if needs_descriptor:
                        page_index.pop()
                    selected_units.pop()
                if admitted:
                    break
            if admitted:
                continue

            omitted_unit_count += 1
            page_index.append(descriptor)
            if self._json_bytes(self._finalize_token_estimate(build_frame())) <= byte_budget:
                continue
            page_index.pop()
            omitted_descriptor_count += 1

        frame = self._finalize_token_estimate(build_frame())
        # Counts and decimal widths can make the final envelope a few bytes
        # larger than an earlier trial.  A descriptor attached to a compressed
        # resident is part of that resident's lossless continuation contract;
        # never discard it merely to preserve a richer inline projection.
        # First step the lowest-relevance non-bare resident through the same
        # preservation-aware compression ladder used during admission.  An
        # independent descriptor is more valuable than richer inline wording
        # while the resident can still preserve its governing semantic refs in
        # a smaller form.  Once every resident is bare, discard the lowest-
        # relevance unbound descriptor before making a resident page-only.
        descriptor_by_unit = {
            str(descriptor.get("unit_ref") or ""): descriptor
            for descriptor in descriptors
        }
        while self._json_bytes(frame) > byte_budget:
            selected_unit_refs = {
                str(unit.get("unit_id") or "") for unit in selected_units
            }
            compressible_unit_index = next(
                (
                    index
                    for index in range(len(selected_units) - 1, -1, -1)
                    if str(
                        (selected_units[index].get("compression") or {}).get(
                            "mode"
                        )
                        or "none"
                    )
                    in {"none", "essential_projection", "reference_summary"}
                ),
                None,
            )
            if compressible_unit_index is not None:
                current_unit = selected_units[compressible_unit_index]
                unit_ref = str(current_unit.get("unit_id") or "")
                mode = str(
                    (current_unit.get("compression") or {}).get("mode") or "none"
                )
                if mode == "none":
                    selected_units[compressible_unit_index] = self._compress_unit(
                        current_unit
                    )
                elif mode == "essential_projection":
                    selected_units[compressible_unit_index] = self._reference_unit(
                        current_unit
                    )
                else:
                    selected_units[compressible_unit_index] = self._bare_reference_unit(
                        current_unit
                    )
                descriptor = descriptor_by_unit.get(unit_ref)
                if descriptor is not None and all(
                    str(item.get("unit_ref") or "") != unit_ref
                    for item in page_index
                ):
                    page_index.append(descriptor)
            else:
                removable_descriptors = [
                    index
                    for index, descriptor in enumerate(page_index)
                    if str(descriptor.get("unit_ref") or "")
                    not in selected_unit_refs
                ]
                if removable_descriptors:
                    removable_descriptor = min(
                        removable_descriptors,
                        key=lambda index: (
                            float(page_index[index].get("relevance_score") or 0.0),
                            -index,
                        ),
                    )
                    page_index.pop(removable_descriptor)
                    omitted_descriptor_count += 1
                elif selected_units:
                    current_unit = selected_units.pop()
                    unit_ref = str(current_unit.get("unit_id") or "")
                    omitted_unit_count += 1
                    descriptor = descriptor_by_unit.get(unit_ref)
                    if descriptor is not None and all(
                        str(item.get("unit_ref") or "") != unit_ref
                        for item in page_index
                    ):
                        page_index.append(descriptor)
                elif page_index:
                    removable_descriptor = min(
                        range(len(page_index)),
                        key=lambda index: (
                            float(page_index[index].get("relevance_score") or 0.0),
                            -index,
                        ),
                    )
                    page_index.pop(removable_descriptor)
                    omitted_descriptor_count += 1
                else:
                    raise ValidationError(
                        "token_or_byte_budget is too small for the reasoning frame envelope"
                    )
            frame = self._finalize_token_estimate(build_frame())
        return frame

    def _frame_payload(
        self,
        *,
        frame_id: str,
        revision: Mapping[str, Any],
        request: Mapping[str, Any],
        task_context: Mapping[str, Any],
        selected_units: Sequence[Mapping[str, Any]],
        page_index: Sequence[Mapping[str, Any]],
        unit_omissions: Sequence[Mapping[str, Any]],
        omitted_unit_count: int,
        omitted_descriptor_count: int,
        closure: Mapping[str, Any],
        discovery: Mapping[str, Any],
        seed_count: int,
        coherent_unit_count: int,
        byte_budget: int,
    ) -> dict[str, Any]:
        sections = self._sections(selected_units)
        source_refs = sorted(
            {
                str(ref)
                for unit in selected_units
                for ref in unit.get("source_atom_refs", [])
                if str(ref)
            }
        )
        unknowns: list[dict[str, Any]] = []
        if closure["truncated"]:
            unknowns.append(
                {
                    "reason": "relationship_closure_truncated",
                    "detail": "The budget-derived traversal allowance was reached.",
                    "truncation_reasons": list(closure["truncation_reasons"]),
                    "continuation_atom_count": len(
                        closure["continuation_atom_refs"]
                    ),
                }
            )
        if discovery["candidate_generation_truncated"]:
            unknowns.append(
                {
                    "reason": "candidate_generation_truncated",
                    "detail": "The budget-derived candidate allowance was reached.",
                }
            )
        if omitted_unit_count:
            unknowns.append(
                {
                    "reason": "frame_budget_exhausted",
                    "omitted_unit_count": omitted_unit_count,
                    "exposed_page_count": len(page_index),
                }
            )
        if omitted_descriptor_count:
            unknowns.append(
                {
                    "reason": "page_descriptor_budget_exhausted",
                    "omitted_descriptor_count": omitted_descriptor_count,
                }
            )
        detailed_omissions = [dict(item) for item in unit_omissions[:16]]
        if len(unit_omissions) > len(detailed_omissions):
            detailed_omissions.append(
                {
                    "reason": "omission_detail_budgeted",
                    "omitted_detail_count": len(unit_omissions) - len(detailed_omissions),
                }
            )
        if omitted_unit_count:
            detailed_omissions.append(
                {
                    "reason": "coherent_units_not_resident",
                    "unit_count": omitted_unit_count,
                    "page_ids": [str(page["page_id"]) for page in page_index],
                }
            )
        if omitted_descriptor_count:
            detailed_omissions.append(
                {
                    "reason": "page_descriptors_omitted",
                    "descriptor_count": omitted_descriptor_count,
                }
            )
        return {
            "status": "compiled",
            "frame_id": frame_id,
            "schema_version": SCHEMA_VERSION,
            "revision": dict(revision),
            "generated_at": utc_now(),
            "request": dict(request),
            "orientation": {
                "task_context": {
                    key: task_context[key]
                    for key in (
                        "human_id",
                        "conversation_id",
                        "project_id",
                        "project_thread_id",
                        "dialogue_frame_id",
                        "task",
                        "objective",
                        "phase",
                    )
                    if task_context.get(key) not in (None, "", [], {})
                },
                "time_scope": task_context.get("time_scope"),
            },
            "units": [dict(unit) for unit in selected_units],
            "sections": sections,
            "current_state": sections["current_state"],
            "decisions": sections["decisions"],
            "constraints": sections["constraints"],
            "commitments": sections["commitments"],
            "relevant_episodes": sections["episodes"],
            "conflicts": sections["conflicts"],
            "unknowns": unknowns,
            "page_index": [dict(page) for page in page_index],
            "source_atom_refs": source_refs,
            "omissions": detailed_omissions,
            "truncated": bool(
                closure["truncated"]
                or discovery["candidate_generation_truncated"]
                or omitted_unit_count
                or omitted_descriptor_count
                or any(unit.get("truncated") for unit in selected_units)
            ),
            "token_estimate": 0,
            "budget": {
                "limit_bytes": byte_budget,
                "limit_tokens": (byte_budget + 3) // 4,
                "used_bytes": 0,
                "estimated_tokens": 0,
            },
            "compilation_trace": {
                "eligible_atom_count": discovery["eligible_atom_count"],
                "ranked_seed_count": seed_count,
                "candidate_generation_truncated": discovery[
                    "candidate_generation_truncated"
                ],
                "candidate_scan_limit": discovery["candidate_scan_limit"],
                "filtered_counts": dict(discovery["filtered_counts"]),
                "closure_atom_count": len(closure["atoms"]),
                **(
                    {
                        "relationship_expansion_count": len(closure["expansions"]),
                        "relationship_truncation_reasons": list(
                            closure["truncation_reasons"]
                        ),
                        "continuation_atom_count": len(
                            closure["continuation_atom_refs"]
                        ),
                        "continuation_edge_count": len(
                            closure["continuation_edge_refs"]
                        ),
                        "mandatory_edges_examined": closure[
                            "mandatory_edges_examined"
                        ],
                        "supporting_edges_examined": closure[
                            "supporting_edges_examined"
                        ],
                        "relationship_work_budget": {
                            "mandatory_atoms": closure["work_budget"][
                                "mandatory_atom_expansions"
                            ],
                            "mandatory_edges": closure["work_budget"][
                                "mandatory_edges"
                            ],
                            "supporting_atoms": closure["work_budget"][
                                "supporting_atom_expansions"
                            ],
                            "supporting_edges": closure["work_budget"][
                                "supporting_edges"
                            ],
                        },
                    }
                    if closure["truncated"]
                    else {}
                ),
                "coherent_unit_count": coherent_unit_count,
                "selected_unit_count": len(selected_units),
                "compressed_unit_count": sum(
                    1
                    for unit in selected_units
                    if str((unit.get("compression") or {}).get("mode") or "none")
                    != "none"
                ),
                "pages_exposed_count": len(page_index),
            },
            "pressure_mode": self._capacity_pressure_mode(),
            "provenance": {
                "store": getattr(self.store, "backend_name", "unknown"),
                "journal_head": revision["journal_head"],
                "compiler_profile_id": "amos.v1.reasoning_frame.coherent",
                "ranker_profile_id": "amos.v1.default",
                "smp_processor_id": self.smp.processor_id,
            },
        }

    def _finalize_token_estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("budget", {})
        for _attempt in range(12):
            used_bytes = self._json_bytes(payload)
            token_estimate = (used_bytes + 3) // 4
            budget = payload["budget"]
            if (
                payload.get("token_estimate") == token_estimate
                and budget.get("used_bytes") == used_bytes
                and budget.get("estimated_tokens") == token_estimate
            ):
                return payload
            payload["token_estimate"] = token_estimate
            budget["used_bytes"] = used_bytes
            budget["estimated_tokens"] = token_estimate
        # Decimal widths converge very quickly; this final assignment keeps the
        # public values consistent even on an unusually shaped response.
        used_bytes = self._json_bytes(payload)
        payload["token_estimate"] = (used_bytes + 3) // 4
        payload["budget"]["used_bytes"] = used_bytes
        payload["budget"]["estimated_tokens"] = payload["token_estimate"]
        return payload

    def _reasoning_work_budget(self, byte_budget: int) -> dict[str, int]:
        """Derive internal traversal work from the caller's context budget.

        These are safety allowances, not output slots: coherent units remain
        atomic, and reaching an allowance produces explicit truncation plus
        loadable continuation references in page descriptors.
        """

        return {
            "candidate_atoms": max(32, byte_budget // 32),
            "ranking_edges": max(32, byte_budget // 32),
            "mandatory_atom_expansions": max(8, byte_budget // 256),
            "mandatory_edges": max(16, byte_budget // 128),
            "supporting_atom_expansions": max(8, byte_budget // 256),
            "supporting_edges": max(16, byte_budget // 128),
            "continuation_edges": max(4, byte_budget // 512),
        }

    def _discover_seed_atoms(
        self,
        *,
        need: str,
        purpose: str,
        depth: str,
        task_context: Mapping[str, Any],
        scope: Mapping[str, Any],
        semantic_scope: Mapping[str, str],
        work_budget: Mapping[str, int],
        requester: str,
        target_processor: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidate_scan_limit = int(work_budget["candidate_atoms"])
        scanned_atoms = self.store.list_atoms_filtered(
            lifecycle_states=["active", "proposed", "archived", "superseded"],
            limit=candidate_scan_limit + 1,
            prioritize_hot=True,
        )
        candidate_generation_truncated = len(scanned_atoms) > candidate_scan_limit
        all_atoms = scanned_atoms[:candidate_scan_limit]
        eligible = []
        hidden_counts: dict[str, int] = {}
        for atom in all_atoms:
            reason = self._atom_omission_reason(
                atom,
                scope=scope,
                semantic_scope=semantic_scope,
                requester=requester,
                target_processor=target_processor,
            )
            if reason is None:
                eligible.append(atom)
            else:
                hidden_counts[reason] = hidden_counts.get(reason, 0) + 1

        if depth == "orientation":
            orientation_types = {
                "commitment",
                "goal",
                "limitation",
                "policy",
                "preference",
                "runtime_state",
            }
            eligible = [atom for atom in eligible if atom.get("type") in orientation_types]
        eligible_ids = {str(atom["id"]) for atom in eligible}
        query_text = " ".join(
            part
            for part in (
                need,
                purpose,
                self.indexes._search_text_for_value(task_context),
            )
            if part
        ).lower()
        cues = [need, purpose]
        cue_tokens = {
            token for token in re.findall(r"[a-z0-9_]+", query_text) if token
        }
        attention_policy = self.retrieval._attention_policy(task_context)
        self.indexes._sync_smp_vector_model(graph_version=self.store.graph_version())
        cue_vector = self.smp.encode(query_text) if query_text else []
        indexed_ids = self.indexes._indexed_retrieval_candidates(
            cue_tokens=cue_tokens,
            attention_policy=attention_policy,
            eligible_atom_ids=eligible_ids,
            limit=candidate_scan_limit,
            neighbor_edge_limit=int(work_budget["ranking_edges"]),
        )
        latent_limit = max(1, len(eligible))
        latent_ids = self.retrieval._latent_retrieval_candidates(
            eligible,
            cue_vector=cue_vector,
            limit=latent_limit,
            minimum_similarity=0.22,
        )
        if indexed_ids is None and not latent_ids:
            candidate_ids = set(eligible_ids)
        else:
            candidate_ids = set(indexed_ids or [])
            candidate_ids.update(latent_ids)
        candidates = [atom for atom in eligible if str(atom["id"]) in candidate_ids]
        superseded_refs = self.graph._active_superseded_refs(
            [str(atom["id"]) for atom in candidates]
        )
        edge_degrees = self.graph._hot_graph_edge_degree_counts(
            candidates,
            edge_scan_limit=int(work_budget["ranking_edges"]),
        )
        edge_scores, traces = self.retrieval._graph_activation_scores(
            candidates,
            cues=cues,
            request_scope=scope,
            requester=requester,
            target_processor=target_processor,
            include_conflicts=True,
            include_low_health=True,
            cue_text=query_text,
            cue_tokens=cue_tokens,
            attention_policy=attention_policy,
            superseded_refs=None,
            edge_scan_limit=int(work_budget["ranking_edges"]),
        )
        ranked = []
        for atom in candidates:
            score, matched, components = self.retrieval._rank_atom(
                atom,
                cues,
                request_scope=scope,
                retrieval_mode="general",
                cue_text=query_text,
                cue_tokens=cue_tokens,
                cue_vector=cue_vector,
                edge_degrees=edge_degrees,
                edge_activation_scores=edge_scores,
                attention_policy=attention_policy,
                superseded_refs=superseded_refs,
            )
            if cues and not matched:
                continue
            ranked.append(
                {
                    **atom,
                    "_frame_relevance": score,
                    "_score_components": components,
                    "_association_trace": traces.get(str(atom["id"]), []),
                }
            )
        ranked.sort(
            key=lambda atom: (
                -float(atom.get("_frame_relevance", 0.0)),
                str(atom.get("updated_at") or ""),
                str(atom.get("id") or ""),
            )
        )
        return ranked, {
            "eligible_atom_count": len(eligible),
            "ranked_seed_count": len(ranked),
            "lexical_candidate_count": len(indexed_ids or []),
            "latent_candidate_count": len(latent_ids),
            "candidate_union_count": len(candidate_ids),
            "candidate_generation_truncated": candidate_generation_truncated,
            "candidate_scan_limit": candidate_scan_limit,
            "candidate_rows_scanned": len(scanned_atoms),
            "ranking_edge_scan_limit": int(work_budget["ranking_edges"]),
            "filtered_counts": hidden_counts,
        }

    def _coherent_closure(
        self,
        *,
        seeds: Sequence[Mapping[str, Any]],
        scope: Mapping[str, Any],
        semantic_scope: Mapping[str, str],
        work_budget: Mapping[str, int],
        requester: str,
        target_processor: str,
    ) -> dict[str, Any]:
        atoms = {str(atom["id"]): dict(atom) for atom in seeds}
        scores = {
            str(atom["id"]): float(atom.get("_frame_relevance", 0.0))
            for atom in seeds
        }
        reasons = {str(atom["id"]): ["semantic_seed"] for atom in seeds}
        edges: dict[str, dict[str, Any]] = {}
        expansions: list[dict[str, Any]] = []
        continuation_atoms: dict[str, dict[str, Any]] = {}
        continuation_edges: dict[str, dict[str, Any]] = {}
        truncation_reasons: set[str] = set()
        traversal_frontier_refs: set[str] = set()
        mandatory_atom_limit = int(work_budget["mandatory_atom_expansions"])
        mandatory_edge_limit = int(work_budget["mandatory_edges"])
        supporting_atom_limit = int(work_budget["supporting_atom_expansions"])
        supporting_edge_limit = int(work_budget["supporting_edges"])
        continuation_probe_limit = int(work_budget["continuation_edges"])
        mandatory_edges_examined = 0
        supporting_edges_examined = 0
        supporting_atoms_added = 0
        seen_mandatory_edges: set[str] = set()
        seen_supporting_edges: set[str] = set()

        def record_continuation(edge: Mapping[str, Any]) -> None:
            edge_id = str(edge.get("edge_id") or "")
            if (
                edge_id not in continuation_edges
                and len(continuation_edges) >= continuation_probe_limit
            ):
                return
            if not self._edge_visible(edge, scope):
                return
            endpoints = {
                str(edge.get("source_ref") or ""),
                str(edge.get("target_ref") or ""),
            }
            endpoints.discard("")
            resolved: dict[str, dict[str, Any]] = {}
            for atom_ref in endpoints:
                atom = atoms.get(atom_ref) or continuation_atoms.get(atom_ref)
                if atom is None:
                    fetched = self.store.get_atom(atom_ref)
                    reason = self._atom_omission_reason(
                        fetched,
                        scope=scope,
                        semantic_scope=semantic_scope,
                        requester=requester,
                        target_processor=target_processor,
                    )
                    if reason is not None:
                        return
                    atom = dict(fetched)
                resolved[atom_ref] = dict(atom)
            for atom_ref, atom in resolved.items():
                if atom_ref not in atoms:
                    continuation_atoms.setdefault(atom_ref, atom)
            continuation_edges[edge_id] = dict(edge)

        frontier = set(atoms)
        truncated = False
        while frontier:
            next_frontier: set[str] = set()
            remaining_edges = max(0, mandatory_edge_limit - mandatory_edges_examined)
            fetch_limit = remaining_edges + continuation_probe_limit + 1
            fetched_edges = self.store.list_edges_for_refs(
                sorted(frontier),
                relations=sorted(MANDATORY_CONTEXT_RELATIONS),
                limit=fetch_limit,
            )
            new_edges = [
                edge
                for edge in fetched_edges
                if str(edge.get("edge_id") or "") not in seen_mandatory_edges
            ]
            process_edges = new_edges[:remaining_edges]
            probe_edges = new_edges[
                remaining_edges : remaining_edges + continuation_probe_limit
            ]
            if len(fetched_edges) >= fetch_limit or probe_edges:
                truncated = True
                truncation_reasons.add("mandatory_edge_budget")
                traversal_frontier_refs.update(frontier)
            for edge in probe_edges:
                seen_mandatory_edges.add(str(edge.get("edge_id") or ""))
                record_continuation(edge)
            for edge in process_edges:
                edge_id = str(edge.get("edge_id") or "")
                seen_mandatory_edges.add(edge_id)
                mandatory_edges_examined += 1
                relation = str(edge.get("relation") or "")
                if not self._edge_visible(edge, scope):
                    continue
                source = str(edge.get("source_ref") or "")
                target = str(edge.get("target_ref") or "")
                visible_endpoints = True
                for atom_ref in (source, target):
                    if atom_ref in atoms:
                        continue
                    atom = self.store.get_atom(atom_ref)
                    reason = self._atom_omission_reason(
                        atom,
                        scope=scope,
                        semantic_scope=semantic_scope,
                        requester=requester,
                        target_processor=target_processor,
                    )
                    if reason is not None:
                        visible_endpoints = False
                        break
                    if len(expansions) >= mandatory_atom_limit:
                        visible_endpoints = False
                        truncated = True
                        truncation_reasons.add("mandatory_atom_budget")
                        traversal_frontier_refs.update(frontier)
                        record_continuation(edge)
                        break
                    atoms[atom_ref] = dict(atom)
                    scores.setdefault(atom_ref, 0.0)
                    reasons.setdefault(atom_ref, []).append(f"required_by:{relation}")
                    next_frontier.add(atom_ref)
                    expansions.append(
                        {
                            "atom_ref": atom_ref,
                            "edge_id": str(edge.get("edge_id") or ""),
                            "relation": relation,
                            "reason": "mandatory_context",
                        }
                    )
                if visible_endpoints and source in atoms and target in atoms:
                    edges[edge_id] = dict(edge)
            if (
                mandatory_edges_examined >= mandatory_edge_limit
                or len(expansions) >= mandatory_atom_limit
            ):
                if next_frontier:
                    traversal_frontier_refs.update(next_frontier)
                    final_probe = self.store.list_edges_for_refs(
                        sorted(next_frontier),
                        relations=sorted(MANDATORY_CONTEXT_RELATIONS),
                        limit=continuation_probe_limit,
                    )
                    for edge in final_probe:
                        edge_id = str(edge.get("edge_id") or "")
                        if edge_id not in seen_mandatory_edges:
                            record_continuation(edge)
                truncated = True
                break
            frontier = next_frontier

        components = []
        for refs in self._connected_components(atom_refs=sorted(atoms), edges=edges):
            component_edge_refs = sorted(
                edge_id
                for edge_id, edge in edges.items()
                if str(edge.get("source_ref")) in refs
                and str(edge.get("target_ref")) in refs
            )
            page_atom_refs = set(refs)
            page_edge_refs = set(component_edge_refs)
            remaining_edges = max(0, supporting_edge_limit - supporting_edges_examined)
            fetch_limit = remaining_edges + continuation_probe_limit + 1
            fetched_edges = self.store.list_edges_for_refs(
                sorted(refs),
                relations=sorted(SUPPORTING_CONTEXT_RELATIONS),
                limit=fetch_limit,
            )
            new_edges = [
                edge
                for edge in fetched_edges
                if str(edge.get("edge_id") or "") not in seen_supporting_edges
            ]
            process_edges = new_edges[:remaining_edges]
            probe_edges = new_edges[
                remaining_edges : remaining_edges + continuation_probe_limit
            ]
            if len(fetched_edges) >= fetch_limit or probe_edges:
                truncated = True
                truncation_reasons.add("supporting_edge_budget")
                traversal_frontier_refs.update(refs)
            for edge in probe_edges:
                seen_supporting_edges.add(str(edge.get("edge_id") or ""))
                record_continuation(edge)
            for edge in process_edges:
                edge_id = str(edge.get("edge_id") or "")
                seen_supporting_edges.add(edge_id)
                supporting_edges_examined += 1
                relation = str(edge.get("relation") or "")
                if not self._edge_visible(edge, scope):
                    continue
                source = str(edge.get("source_ref") or "")
                target = str(edge.get("target_ref") or "")
                other_refs = {source, target} - set(refs)
                visible = True
                for atom_ref in other_refs:
                    atom = self.store.get_atom(atom_ref)
                    reason = self._atom_omission_reason(
                        atom,
                        scope=scope,
                        semantic_scope=semantic_scope,
                        requester=requester,
                        target_processor=target_processor,
                    )
                    if reason is not None:
                        visible = False
                        break
                    if atom_ref not in atoms:
                        if supporting_atoms_added >= supporting_atom_limit:
                            visible = False
                            truncated = True
                            truncation_reasons.add("supporting_atom_budget")
                            traversal_frontier_refs.update(refs)
                            record_continuation(edge)
                            break
                        atoms[atom_ref] = dict(atom)
                        supporting_atoms_added += 1
                        scores.setdefault(atom_ref, 0.0)
                        reasons.setdefault(atom_ref, []).append(
                            f"supporting_page:{relation}"
                        )
                if visible and source in atoms and target in atoms:
                    page_atom_refs.update({source, target})
                    page_edge_refs.add(str(edge["edge_id"]))
                    edges.setdefault(str(edge["edge_id"]), dict(edge))
            for edge_id, edge in edges.items():
                if str(edge.get("relation") or "") not in SUPPORTING_CONTEXT_RELATIONS:
                    continue
                endpoints = {
                    str(edge.get("source_ref") or ""),
                    str(edge.get("target_ref") or ""),
                }
                if endpoints.intersection(refs):
                    page_atom_refs.update(endpoints)
                    page_edge_refs.add(edge_id)
            for edge_id, edge in continuation_edges.items():
                endpoints = {
                    str(edge.get("source_ref") or ""),
                    str(edge.get("target_ref") or ""),
                }
                if not endpoints.intersection(refs):
                    continue
                page_atom_refs.update(endpoints)
                page_edge_refs.add(edge_id)
            components.append(
                {
                    "atom_refs": refs,
                    "edge_refs": component_edge_refs,
                    "page_atom_refs": sorted(page_atom_refs),
                    "page_edge_refs": sorted(page_edge_refs),
                    "relevance_score": max((scores.get(ref, 0.0) for ref in refs), default=0.0),
                }
            )
        components.sort(
            key=lambda component: (
                -float(component["relevance_score"]),
                component["atom_refs"],
            )
        )
        # Supporting edges are discoverable from either endpoint.  When both
        # endpoints are independent semantic seeds, that symmetric discovery
        # can otherwise advertise two capabilities that load exactly the same
        # atoms and relationships.  Keep the expanded projection on the
        # highest-ranked coherent unit and restore later duplicates to their
        # resident projection.  A later unit that must be compressed will
        # still receive its own aligned descriptor during frame admission.
        expanded_page_projections: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for component in components:
            base_atom_refs = tuple(component["atom_refs"])
            base_edge_refs = tuple(component["edge_refs"])
            page_projection = (
                tuple(component["page_atom_refs"]),
                tuple(component["page_edge_refs"]),
            )
            if page_projection == (base_atom_refs, base_edge_refs):
                continue
            if page_projection in expanded_page_projections:
                component["page_atom_refs"] = list(base_atom_refs)
                component["page_edge_refs"] = list(base_edge_refs)
                continue
            expanded_page_projections.add(page_projection)
        return {
            "atoms": atoms,
            "page_atoms": {**atoms, **continuation_atoms},
            "edges": edges,
            "scores": scores,
            "inclusion_reasons": reasons,
            "components": components,
            "expansions": expansions,
            "truncated": truncated,
            "truncation_reasons": sorted(truncation_reasons),
            "continuation_atom_refs": sorted(continuation_atoms),
            "continuation_edge_refs": sorted(continuation_edges),
            "traversal_frontier_refs": sorted(traversal_frontier_refs),
            "work_budget": dict(work_budget),
            "mandatory_edges_examined": mandatory_edges_examined,
            "supporting_edges_examined": supporting_edges_examined,
        }

    def _build_unit(
        self,
        *,
        atom_refs: Sequence[str],
        atoms: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, Mapping[str, Any]],
        relevance_score: float,
        inclusion_reasons: Mapping[str, Sequence[str]],
        requester: str,
        target_processor: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ordered_refs = sorted(
            atom_refs,
            key=lambda ref: (
                self._atom_time(atoms[ref]),
                ref,
            ),
        )
        constraints = self._constraint_refs(ordered_refs, atoms, edges)
        active_refs = self._active_conclusions(
            ordered_refs,
            atoms,
            edges,
            constraint_refs=constraints,
        )
        commitments = [ref for ref in ordered_refs if atoms[ref].get("type") == "commitment"]
        conflict_edges = sorted(
            edge_id
            for edge_id, edge in edges.items()
            if edge.get("relation") == "rel:contradicts"
        )
        unit_type = self._unit_type(ordered_refs, atoms, edges)
        unit_id = stable_id(
            "unit",
            {
                "unit_type": unit_type,
                "atom_refs": ordered_refs,
                "edge_refs": sorted(edges),
            },
        )
        items = []
        omissions = []
        for ref in ordered_refs:
            atom = dict(atoms[ref])
            atom.setdefault("_score_components", {"coherent_closure": 1.0})
            atom.setdefault("_association_trace", [])
            item, evidence_omissions = self.retrieval._packet_item(
                atom,
                max(0.0, min(1.0, relevance_score)),
                requester=requester,
                target_processor=target_processor,
            )
            item["rank"] = len(items) + 1
            items.append(item)
            omissions.extend(evidence_omissions)
        conclusion_ref = active_refs[0] if active_refs else ordered_refs[-1]
        summary = str(self.graph._render_atom(atoms[conclusion_ref])["text"])
        title = self._truncate_text(summary, 120) or unit_type.replace("_", " ").title()
        sequence = [
            {
                "atom_ref": ref,
                "atom_type": str(atoms[ref].get("type") or ""),
                "observed_at": self._atom_time(atoms[ref]) or None,
                "role": self._sequence_role(ref, active_refs, constraints, commitments),
            }
            for ref in ordered_refs
        ]
        relationship_items = [
            {
                "edge_id": edge_id,
                "relation": str(edge.get("relation") or ""),
                "source_ref": str(edge.get("source_ref") or ""),
                "target_ref": str(edge.get("target_ref") or ""),
                "confidence": dict(edge.get("confidence") or {}),
                "evidence_refs": list(edge.get("evidence_refs") or []),
            }
            for edge_id, edge in sorted(edges.items())
        ]
        return (
            {
                "unit_id": unit_id,
                "unit_type": unit_type,
                "title": title,
                "summary": self._truncate_text(summary, 280),
                "relevance_score": round(max(0.0, min(1.0, relevance_score)), 4),
                "active_conclusion_refs": active_refs,
                "constraint_refs": constraints,
                "commitment_refs": commitments,
                "conflict_refs": conflict_edges,
                "source_atom_refs": ordered_refs,
                "relationship_refs": sorted(edges),
                "relationships": relationship_items,
                "items": items,
                "sequence": sequence,
                "inclusion_reasons": {
                    ref: list(inclusion_reasons.get(ref, ["coherent_closure"]))
                    for ref in ordered_refs
                },
                "compression": {"mode": "none"},
                "truncated": False,
            },
            omissions,
        )

    def _page_descriptor(
        self,
        *,
        frame_id: str,
        revision: Mapping[str, Any],
        unit: Mapping[str, Any],
        atoms: Mapping[str, Mapping[str, Any]],
        semantic_scope: Mapping[str, str],
    ) -> dict[str, Any]:
        page_atom_refs = list(unit.get("_page_atom_refs", unit["source_atom_refs"]))
        page_edge_refs = list(
            unit.get("_page_edge_refs", unit["relationship_refs"])
        )
        descriptor_text = self._page_descriptor_text(
            unit=unit,
            page_atom_refs=page_atom_refs,
            page_edge_refs=page_edge_refs,
            atoms=atoms,
        )
        times = [self._atom_time(atoms[ref]) for ref in page_atom_refs if ref in atoms]
        times = [value for value in times if value]
        core = {
            "descriptor_version": "amos.reasoning.page.v1",
            "frame_id": frame_id,
            "revision": dict(revision),
            "semantic_scope": dict(semantic_scope),
            "unit_ref": unit["unit_id"],
            "page_type": unit["unit_type"],
            "title": descriptor_text["title"],
            "summary": descriptor_text["summary"],
            "relevance": descriptor_text["relevance"],
            "relevance_score": unit["relevance_score"],
            "time_range": {
                "start": min(times) if times else None,
                "end": max(times) if times else None,
            },
            "focus_atom_refs": list(unit["source_atom_refs"]),
            "source_atom_refs": page_atom_refs,
            "relationship_refs": page_edge_refs,
            "supported_depths": ["focused", "supporting"],
            "related_pages": [],
        }
        descriptor_digest = digest(core)
        return {
            **core,
            "descriptor_digest": descriptor_digest,
            "page_id": stable_id(
                "page",
                {"frame_id": frame_id, "descriptor_digest": descriptor_digest},
            ),
        }

    def _page_descriptor_text(
        self,
        *,
        unit: Mapping[str, Any],
        page_atom_refs: Sequence[str],
        page_edge_refs: Sequence[str],
        atoms: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, str]:
        """Describe an expanded page's semantic delta without exposing detail.

        Source and relationship references are already part of the descriptor
        capability.  For a supporting expansion, expose only bounded counts,
        atom types, and relation types so the reasoner can distinguish the page
        from its resident unit.  The atom payloads remain available only after
        the trusted runtime validates and loads the page.
        """

        resident_atom_refs = {
            str(ref) for ref in unit.get("source_atom_refs", []) if str(ref)
        }
        resident_edge_refs = {
            str(ref) for ref in unit.get("relationship_refs", []) if str(ref)
        }
        added_atom_refs = sorted(
            {str(ref) for ref in page_atom_refs if str(ref)} - resident_atom_refs
        )
        added_edge_refs = sorted(
            {str(ref) for ref in page_edge_refs if str(ref)} - resident_edge_refs
        )
        if not added_atom_refs and not added_edge_refs:
            return {
                "title": str(unit["title"]),
                "summary": str(unit["summary"]),
                # The active request already carries the need. Repeating it in
                # every descriptor can crowd independent pages out of a bounded
                # frame without adding information for the reasoner.
                "relevance": "Supports the active frame memory need.",
            }

        atom_types = sorted(
            {
                str(atoms[ref].get("type") or "memory")
                for ref in added_atom_refs
                if ref in atoms
            }
        )
        relation_types = sorted(
            {
                str(edge.get("relation") or "")
                for edge_ref in added_edge_refs
                if (edge := self.store.get_edge(edge_ref)) is not None
                and str(edge.get("relation") or "")
            }
        )

        def counted(count: int, noun: str) -> str:
            return f"{count} {noun if count == 1 else noun + 's'}"

        parts = []
        if added_atom_refs:
            type_detail = ", ".join(atom_types) if atom_types else "memory"
            parts.append(f"{counted(len(added_atom_refs), 'atom')} ({type_detail})")
        if added_edge_refs:
            relation_detail = (
                ", ".join(relation_types) if relation_types else "typed"
            )
            parts.append(
                f"{counted(len(added_edge_refs), 'relationship')} "
                f"({relation_detail})"
            )
        title_detail = ", ".join(relation_types or atom_types or ["memory"])
        return {
            "title": self._truncate_text(
                f"Supporting context via {title_detail}", 120
            ),
            "summary": self._truncate_text(
                "Adds " + " and ".join(parts) + " beyond the resident unit.",
                280,
            ),
            "relevance": (
                "Adds typed supporting context that is not resident in the frame."
            ),
        }

    def _validate_page_descriptor(
        self,
        page: Mapping[str, Any],
        *,
        frame_id: str,
        revision: Mapping[str, Any],
    ) -> None:
        fields = (
            "descriptor_version",
            "frame_id",
            "revision",
            "semantic_scope",
            "unit_ref",
            "page_type",
            "title",
            "summary",
            "relevance",
            "relevance_score",
            "time_range",
            "focus_atom_refs",
            "source_atom_refs",
            "relationship_refs",
            "supported_depths",
            "related_pages",
            "descriptor_digest",
            "page_id",
        )
        for field in fields:
            if field not in page:
                raise ValidationError(f"page descriptor missing field: {field}")
        unknown = sorted(set(page) - set(fields))
        if unknown:
            raise ValidationError(
                "page descriptor contains unknown field(s): " + ", ".join(unknown)
            )
        if page.get("descriptor_version") != "amos.reasoning.page.v1":
            raise ValidationError("unsupported page descriptor version")
        if str(page.get("frame_id")) != frame_id:
            raise ValidationError("page descriptor does not belong to frame_id")
        if self._revision(page.get("revision")) != dict(revision):
            raise ValidationError("page descriptor revision does not match frame revision")
        core = {
            key: page[key]
            for key in (
                "descriptor_version",
                "frame_id",
                "revision",
                "semantic_scope",
                "unit_ref",
                "page_type",
                "title",
                "summary",
                "relevance",
                "relevance_score",
                "time_range",
                "focus_atom_refs",
                "source_atom_refs",
                "relationship_refs",
                "supported_depths",
                "related_pages",
            )
        }
        expected_digest = digest(core)
        if str(page.get("descriptor_digest")) != expected_digest:
            raise ValidationError("page descriptor digest mismatch")
        expected_page_id = stable_id(
            "page",
            {"frame_id": frame_id, "descriptor_digest": expected_digest},
        )
        if str(page.get("page_id")) != expected_page_id:
            raise ValidationError("page descriptor page_id mismatch")

    def _compress_unit(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        essential_refs = self._essential_unit_refs(unit)
        projected_items = []
        omitted_detail = []
        for item in unit.get("items", []):
            atom_ref = str(item.get("atom_ref") or "")
            if atom_ref not in essential_refs:
                omitted_detail.append(atom_ref)
                continue
            projected_items.append(
                {
                    "atom_ref": atom_ref,
                    "type": item.get("type"),
                    "rendered_content": {
                        "format": "text",
                        "text": str((item.get("rendered_content") or {}).get("text") or ""),
                    },
                    "confidence": item.get("confidence"),
                    "evidence_refs": list(item.get("evidence_refs", [])),
                    "scope": dict(item.get("scope") or {}),
                    "lifecycle_state": item.get("lifecycle_state"),
                    "health_status": item.get("health_status"),
                    "updated_at": item.get("updated_at"),
                }
            )
        compressed = {
            **dict(unit),
            "items": projected_items,
            "relationships": [
                {
                    "edge_id": item.get("edge_id"),
                    "relation": item.get("relation"),
                    "source_ref": item.get("source_ref"),
                    "target_ref": item.get("target_ref"),
                }
                for item in unit.get("relationships", [])
            ],
            "inclusion_reasons": {
                ref: reasons
                for ref, reasons in dict(unit.get("inclusion_reasons") or {}).items()
                if ref in essential_refs
            },
            "compression": {
                "mode": "essential_projection",
                "omitted_atom_detail_refs": omitted_detail,
                "preserved": [
                    "active_conclusions",
                    "constraints",
                    "commitments",
                    "conflicts",
                    "temporal_sequence",
                    "source_refs",
                ],
            },
            "truncated": bool(unit.get("truncated") or omitted_detail),
        }
        compressed["compression"]["original_bytes"] = self._json_bytes(unit)
        compressed["compression"]["rendered_bytes"] = self._json_bytes(compressed)
        return compressed

    def _reference_unit(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        essential_refs = self._essential_unit_refs(unit)
        essential_items = []
        for item in unit.get("items", []):
            atom_ref = str(item.get("atom_ref") or "")
            if atom_ref not in essential_refs:
                continue
            essential_items.append(
                {
                    "atom_ref": atom_ref,
                    "type": item.get("type"),
                    "rendered_content": {
                        "format": "text",
                        "text": self._truncate_text(
                            (item.get("rendered_content") or {}).get("text"), 180
                        ),
                    },
                    "confidence": item.get("confidence"),
                }
            )
        preserved_relations = self._essential_relationships(unit, essential_refs)
        original_bytes = int(
            (unit.get("compression") or {}).get("original_bytes")
            or self._json_bytes(unit)
        )
        reference = {
            "unit_id": unit["unit_id"],
            "unit_type": unit["unit_type"],
            "title": unit["title"],
            "summary": unit["summary"],
            "relevance_score": unit["relevance_score"],
            "active_conclusion_refs": list(unit.get("active_conclusion_refs", [])),
            "constraint_refs": list(unit.get("constraint_refs", [])),
            "commitment_refs": list(unit.get("commitment_refs", [])),
            "conflict_refs": list(unit.get("conflict_refs", [])),
            "source_atom_refs": list(unit.get("source_atom_refs", [])),
            "relationship_refs": list(unit.get("relationship_refs", [])),
            "items": essential_items,
            "relationships": preserved_relations,
            "sequence": list(unit.get("sequence", [])),
            "inclusion_reasons": {},
            "page_id": unit.get("page_id"),
            "compression": {
                "mode": "reference_summary",
                "original_bytes": original_bytes,
                "preserved": [
                    "active_conclusions",
                    "constraints",
                    "commitments",
                    "conflicts",
                    "temporal_sequence",
                    "source_refs",
                ],
            },
            "truncated": True,
        }
        reference["compression"]["rendered_bytes"] = self._json_bytes(reference)
        return reference

    def _bare_reference_unit(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        essential_refs = self._essential_unit_refs(unit)
        original_bytes = int(
            (unit.get("compression") or {}).get("original_bytes")
            or self._json_bytes(unit)
        )
        reference = {
            **dict(unit),
            "items": [],
            "relationships": self._essential_relationships(unit, essential_refs),
            "inclusion_reasons": {},
            "compression": {
                "mode": "reference_only",
                "original_bytes": original_bytes,
                "preserved": [
                    "active_conclusions",
                    "constraints",
                    "commitments",
                    "conflicts",
                    "temporal_sequence",
                    "source_refs",
                ],
            },
            "truncated": True,
        }
        reference["compression"]["rendered_bytes"] = self._json_bytes(reference)
        return reference

    def _essential_unit_refs(self, unit: Mapping[str, Any]) -> set[str]:
        essential_refs = {
            str(ref) for ref in unit.get("active_conclusion_refs", []) if str(ref)
        }
        essential_refs.update(
            str(ref) for ref in unit.get("constraint_refs", []) if str(ref)
        )
        essential_refs.update(
            str(ref) for ref in unit.get("commitment_refs", []) if str(ref)
        )
        conflict_edge_ids = {
            str(ref) for ref in unit.get("conflict_refs", []) if str(ref)
        }
        for relationship in unit.get("relationships", []):
            if str(relationship.get("edge_id") or "") in conflict_edge_ids:
                essential_refs.add(str(relationship.get("source_ref") or ""))
                essential_refs.add(str(relationship.get("target_ref") or ""))
        essential_refs.discard("")
        if not essential_refs:
            essential_refs.update(
                str(ref) for ref in unit.get("source_atom_refs", [])[:1] if str(ref)
            )
        return essential_refs

    def _essential_relationships(
        self, unit: Mapping[str, Any], essential_refs: set[str]
    ) -> list[dict[str, Any]]:
        governing_relations = {
            "rel:constrained_by",
            "rel:contradicts",
            "rel:corrected_by",
            "rel:forbids",
            "rel:made_commitment",
            "rel:requires",
            "rel:satisfied_commitment",
            "rel:supersedes",
        }
        preserved = []
        for item in unit.get("relationships", []):
            relation = str(item.get("relation") or "")
            source = str(item.get("source_ref") or "")
            target = str(item.get("target_ref") or "")
            if relation not in governing_relations and not {
                source,
                target,
            }.intersection(essential_refs):
                continue
            preserved.append(
                {
                    "edge_id": item.get("edge_id"),
                    "relation": relation,
                    "source_ref": source,
                    "target_ref": target,
                }
            )
        return preserved

    def _connected_components(
        self,
        *,
        atom_refs: Sequence[str],
        edges: Mapping[str, Mapping[str, Any]],
    ) -> list[list[str]]:
        adjacency = {str(ref): set() for ref in atom_refs}
        for edge in edges.values():
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source in adjacency and target in adjacency:
                adjacency[source].add(target)
                adjacency[target].add(source)
        components = []
        remaining = set(adjacency)
        while remaining:
            start = min(remaining)
            stack = [start]
            component = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(sorted(adjacency[current] - component, reverse=True))
            remaining.difference_update(component)
            components.append(sorted(component))
        return components

    def _active_conclusions(
        self,
        refs: Sequence[str],
        atoms: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, Mapping[str, Any]],
        *,
        constraint_refs: Sequence[str],
    ) -> list[str]:
        superseded_targets = {
            str(edge.get("target_ref") or "")
            for edge in edges.values()
            if edge.get("relation") == "rel:supersedes"
        }
        constraints = set(constraint_refs)
        active = [
            ref
            for ref in refs
            if ref not in superseded_targets
            and ref not in constraints
            and atoms[ref].get("type") != "episode"
            and atoms[ref].get("lifecycle_state") in {"active", "proposed"}
        ]
        return sorted(active, key=lambda ref: (self._atom_time(atoms[ref]), ref), reverse=True)

    def _constraint_refs(
        self,
        refs: Sequence[str],
        atoms: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, Mapping[str, Any]],
    ) -> list[str]:
        constraints = {
            ref
            for ref in refs
            if atoms[ref].get("type") in {"limitation", "policy"}
            or (
                atoms[ref].get("type") == "preference"
                and str((atoms[ref].get("payload") or {}).get("polarity") or "")
                in {"require", "requires", "forbid", "forbids"}
            )
        }
        ref_set = set(refs)
        for edge in edges.values():
            relation = str(edge.get("relation") or "")
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            # Relation direction carries the governing role.  A decision that
            # is constrained by or requires another memory is not itself a
            # constraint; a policy that forbids another memory is.
            constraint_ref = (
                target
                if relation in {"rel:constrained_by", "rel:requires"}
                else source
                if relation == "rel:forbids"
                else ""
            )
            if constraint_ref in ref_set:
                constraints.add(constraint_ref)
        constraints.discard("")
        return sorted(constraints)

    def _unit_type(
        self,
        refs: Sequence[str],
        atoms: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, Mapping[str, Any]],
    ) -> str:
        relations = {str(edge.get("relation") or "") for edge in edges.values()}
        types = {str(atoms[ref].get("type") or "") for ref in refs}
        if "rel:contradicts" in relations:
            return "conflict_set"
        if "rel:supersedes" in relations or "rel:decided" in relations:
            return "decision_chain"
        if "commitment" in types or "rel:made_commitment" in relations:
            return "commitment_history"
        if "episode" in types:
            return "episode"
        if "rel:corrected_by" in relations or "action_outcome" in types:
            return "failure_correction"
        if types.intersection({"goal", "policy", "runtime_state"}):
            return "current_state"
        return "related_memory"

    def _sections(self, units: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
        sections = {
            "current_state": [],
            "decisions": [],
            "constraints": [],
            "commitments": [],
            "episodes": [],
            "conflicts": [],
        }
        for unit in units:
            unit_id = str(unit["unit_id"])
            unit_type = str(unit.get("unit_type") or "")
            if unit_type == "current_state":
                sections["current_state"].append(unit_id)
            if unit_type == "decision_chain":
                sections["decisions"].append(unit_id)
            if unit.get("constraint_refs"):
                sections["constraints"].append(unit_id)
            if unit_type == "commitment_history" or unit.get("commitment_refs"):
                sections["commitments"].append(unit_id)
            if unit_type == "episode":
                sections["episodes"].append(unit_id)
            if unit_type == "conflict_set" or unit.get("conflict_refs"):
                sections["conflicts"].append(unit_id)
        return sections

    def _atom_omission_reason(
        self,
        atom: Mapping[str, Any] | None,
        *,
        scope: Mapping[str, Any],
        semantic_scope: Mapping[str, str],
        requester: str,
        target_processor: str,
    ) -> str | None:
        if atom is None:
            return "not_found"
        if atom.get("deleted") or atom.get("lifecycle_state") in {"deleted", "tombstoned"}:
            return "deleted"
        if atom.get("health_status") == "deleted":
            return "deleted"
        if not self._semantic_atom_visible(atom, semantic_scope):
            return "semantic_scope_hidden"
        if not scope_visible(atom.get("scope", {}), scope):
            return "scope_hidden"
        if not access_visible(atom.get("access_policy", {}), requester, target_processor):
            return "access_hidden"
        return None

    def _trusted_semantic_scope(
        self,
        task_context: Mapping[str, Any],
        request_scope: Mapping[str, Any],
    ) -> dict[str, str]:
        semantic_scope: dict[str, str] = {}
        for canonical, context_fields in TASK_CONTEXT_SEMANTIC_FIELDS.items():
            values = []
            for field in context_fields:
                if field not in task_context:
                    continue
                value = task_context[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValidationError(
                        f"task_context.{field} must be a non-empty string"
                    )
                values.append(value.strip())
            for field in ATOM_SEMANTIC_FIELDS[canonical]:
                if field not in request_scope:
                    continue
                raw = request_scope[field]
                if raw in (None, "global"):
                    continue
                if not isinstance(raw, (str, int)) or isinstance(raw, bool):
                    raise ValidationError(
                        f"scope.{field} must be a string or integer identifier"
                    )
                value = str(raw).strip()
                if not value:
                    raise ValidationError(f"scope.{field} must not be empty")
                values.append(value)
            distinct = set(values)
            if len(distinct) > 1:
                raise ValidationError(
                    f"trusted semantic scope fields conflict for {canonical}"
                )
            if values:
                semantic_scope[canonical] = values[0]
        return semantic_scope

    def _descriptor_semantic_scope(
        self, descriptor: Mapping[str, Any]
    ) -> dict[str, str]:
        raw = descriptor.get("semantic_scope")
        if not isinstance(raw, Mapping):
            raise ValidationError("page descriptor semantic_scope must be an object")
        unknown = sorted(set(raw) - set(TASK_CONTEXT_SEMANTIC_FIELDS))
        if unknown:
            raise ValidationError(
                "page descriptor semantic_scope contains unknown field(s): "
                + ", ".join(unknown)
            )
        semantic_scope: dict[str, str] = {}
        for field, value in raw.items():
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(
                    f"page descriptor semantic_scope.{field} must be a non-empty string"
                )
            semantic_scope[str(field)] = value.strip()
        return semantic_scope

    def _visibility_scope(
        self,
        request_scope: Mapping[str, Any],
        semantic_scope: Mapping[str, str],
    ) -> dict[str, Any]:
        visible = dict(request_scope)
        for canonical, expected in semantic_scope.items():
            for field in ATOM_SEMANTIC_FIELDS[canonical]:
                if field in visible and visible[field] not in (None, "global"):
                    if str(visible[field]).strip() != expected:
                        raise ValidationError(
                            f"scope.{field} conflicts with trusted {canonical}"
                        )
                else:
                    visible[field] = expected
        return visible

    def _semantic_atom_visible(
        self,
        atom: Mapping[str, Any],
        semantic_scope: Mapping[str, str],
    ) -> bool:
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        atom_scope = atom.get("scope")
        atom_scope = atom_scope if isinstance(atom_scope, Mapping) else {}
        for canonical, fields in ATOM_SEMANTIC_FIELDS.items():
            expected = semantic_scope.get(canonical)
            for container in (atom_scope, payload):
                for field in fields:
                    if field not in container or container[field] is None:
                        continue
                    raw = container[field]
                    if raw == "global":
                        continue
                    if not isinstance(raw, (str, int)) or isinstance(raw, bool):
                        return False
                    value = str(raw).strip()
                    if not value or expected is None or value != expected:
                        return False
        return True

    def _edge_visible(
        self, edge: Mapping[str, Any], scope: Mapping[str, Any]
    ) -> bool:
        return (
            not edge.get("deleted")
            and edge.get("lifecycle_state") == "active"
            and scope_visible(edge.get("scope", {}), scope)
        )

    def _unit_refs(
        self, units: Sequence[Mapping[str, Any]], field: str
    ) -> list[str]:
        return sorted(
            {
                str(ref)
                for unit in units
                for ref in unit.get(field, [])
                if str(ref)
            }
        )

    def _sequence_role(
        self,
        ref: str,
        active_refs: Sequence[str],
        constraints: Sequence[str],
        commitments: Sequence[str],
    ) -> str:
        if ref in active_refs:
            return "active_conclusion"
        if ref in constraints:
            return "constraint"
        if ref in commitments:
            return "commitment"
        return "history"

    def _atom_time(self, atom: Mapping[str, Any]) -> str:
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        return str(
            payload.get("started_at")
            or payload.get("time_index")
            or atom.get("observed_at")
            or atom.get("updated_at")
            or atom.get("created_at")
            or ""
        )

    def _budget(
        self,
        value: int | Mapping[str, int] | None,
        *,
        default_tokens: int,
    ) -> dict[str, int]:
        if value is None:
            return {"tokens": default_tokens, "bytes": default_tokens * 4}
        if isinstance(value, bool):
            raise ValidationError("token_or_byte_budget must not be boolean")
        if isinstance(value, int):
            if value <= 0:
                raise ValidationError("token_or_byte_budget must be positive")
            return {"bytes": value}
        if not isinstance(value, Mapping):
            raise ValidationError("token_or_byte_budget must be an integer or object")
        has_tokens = "tokens" in value
        has_bytes = "bytes" in value
        if has_tokens == has_bytes:
            raise ValidationError(
                "token_or_byte_budget must contain exactly one of tokens or bytes"
            )
        key = "tokens" if has_tokens else "bytes"
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValidationError(f"token_or_byte_budget.{key} must be a positive integer")
        return {key: raw, "bytes": raw * 4 if key == "tokens" else raw}

    def _revision(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValidationError("revision must be an object")
        unknown = sorted(set(value) - {"graph_version", "journal_head"})
        if unknown:
            raise ValidationError(
                "revision contains unknown field(s): " + ", ".join(unknown)
            )
        graph_version = value.get("graph_version")
        journal_head = value.get("journal_head")
        if isinstance(graph_version, bool) or not isinstance(graph_version, int) or graph_version < 0:
            raise ValidationError("revision.graph_version must be a non-negative integer")
        if not isinstance(journal_head, str) or not journal_head:
            raise ValidationError("revision.journal_head must be a non-empty string")
        return {"graph_version": graph_version, "journal_head": journal_head}

    def _mapping(self, value: Any, name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValidationError(f"{name} must be an object")
        return dict(value)

    def _required_text(self, value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name} must be a non-empty string")
        return value.strip()

    def _depth(self, value: Any, aliases: Mapping[str, str]) -> str:
        if not isinstance(value, str) or value not in aliases:
            raise ValidationError(
                "unsupported reasoning memory depth: " + str(value)
            )
        return aliases[value]

    def _truncate_text(self, value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    def _json_bytes(self, value: Any) -> int:
        return len(canonical_json(value).encode("utf-8"))
