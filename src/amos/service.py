"""High-level AMOS v1 service API.

`Amos` is the stable public API entry point. Domain behavior lives in explicit
subsystem services so storage, retrieval, maintenance, governance, and views
can evolve independently without turning the entry point into a God Object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ._service_support import (
    MaintenanceProcessor,
    SemanticMaintenanceProcessor,
    SQLiteStore,
    default_processor_registry,
)
from .access_service import AccessService
from .capacity_service import CapacityService
from .continuity_service import ContinuityService
from .diagnostics_service import DiagnosticsService
from .graph_service import GraphService
from .governance_service import GovernanceService
from .index_service import IndexService
from .mutations_service import MutationService
from .policy_service import PolicyService
from .retrieval_service import RetrievalService
from .reasoning_service import ReasoningFrameService
from .stewardship_service import StewardshipService
from .temporal_service import TemporalService
from .views_service import ViewService


class Amos:
    """Stable public facade over the AMOS v1-local subsystem services."""

    def __init__(
        self,
        db_path: str | Path = "amos.sqlite3",
        *,
        store: Any | None = None,
        maintenance_processors: Sequence[MaintenanceProcessor] | None = None,
        maintenance_processor_paths: Sequence[str] | None = None,
    ):
        self.store = store or SQLiteStore(db_path)
        self.smp = SemanticMaintenanceProcessor()
        self.maintenance_processors = default_processor_registry(
            self.smp,
            processors=maintenance_processors,
            processor_paths=maintenance_processor_paths,
        )
        self.access = AccessService(self.store)
        self.indexes = IndexService(self.store, self.smp)
        self.graph = GraphService(self.store, self.indexes)
        self.temporal = TemporalService()
        self.capacity = CapacityService(self.store)
        self.mutations = MutationService(
            self.store, self.access, self.indexes, self.graph
        )
        self.governance = GovernanceService(
            self.store, self.access, self.indexes, self.graph
        )
        self.stewardship = StewardshipService(
            self.store,
            self.smp,
            self.maintenance_processors,
            self.mutations,
            self.indexes,
            self.graph,
        )
        self.policy = PolicyService(
            self.store,
            self.smp,
            self.mutations,
            self.indexes,
            self.graph,
            self.capacity,
            self.temporal,
            self.stewardship,
        )
        self.indexes.set_policy_provider(self.policy.memory_policy)
        self.retrieval = RetrievalService(
            self.store,
            self.smp,
            self.access,
            self.indexes,
            self.graph,
            self.capacity,
            self.temporal,
            self.policy.run_memory_policy,
        )
        self.reasoning = ReasoningFrameService(
            self.store,
            self.smp,
            self.access,
            self.indexes,
            self.graph,
            self.retrieval,
            self.capacity,
            self.policy.run_memory_policy,
        )
        self.continuity = ContinuityService(
            self.store,
            self.access,
            self.indexes,
            self.graph,
            self.mutations,
            self.reasoning,
        )
        self.views = ViewService(
            self.store,
            self.smp,
            self.mutations,
            self.retrieval,
            self.graph,
            self.capacity,
        )
        self.diagnostics = DiagnosticsService(
            self.store, self.policy, self.capacity, self.graph
        )

    def close(self) -> None:
        self.store.close()

    def register_maintenance_processor(
        self, processor: MaintenanceProcessor
    ) -> dict[str, Any]:
        return self.stewardship.register_maintenance_processor(processor)

    def load_maintenance_processor(self, import_path: str) -> dict[str, Any]:
        return self.stewardship.load_maintenance_processor(import_path)

    def list_maintenance_processors(self) -> dict[str, Any]:
        return self.stewardship.list_maintenance_processors()

    def configure_capacity_budget(
        self,
        *,
        hard_capacity_bytes: int,
        warning_ratio: float = 0.70,
        critical_ratio: float = 0.90,
    ) -> dict[str, Any]:
        return self.capacity.configure_capacity_budget(
            hard_capacity_bytes=hard_capacity_bytes,
            warning_ratio=warning_ratio,
            critical_ratio=critical_ratio,
        )

    def configure_memory_policy(
        self,
        *,
        enabled: bool | None = None,
        schedule: Mapping[str, Any] | None = None,
        maintenance: Mapping[str, Any] | None = None,
        distillation: Mapping[str, Any] | None = None,
        maintenance_distiller: Mapping[str, Any] | None = None,
        decay: Mapping[str, Any] | None = None,
        storage_cleanup: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.policy.configure_memory_policy(
            enabled=enabled,
            schedule=schedule,
            maintenance=maintenance,
            distillation=distillation,
            maintenance_distiller=maintenance_distiller,
            decay=decay,
            storage_cleanup=storage_cleanup,
        )

    def memory_policy(self) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.policy.memory_policy()

    def memory_policy_status(
        self, *, policy: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.policy.memory_policy_status(policy=policy)

    def run_memory_policy(
        self,
        *,
        force: bool = False,
        trigger: str = "scheduler",
        scope: Mapping[str, Any] | None = None,
        actor: str = "svc:memory_policy",
    ) -> dict[str, Any]:
        return self.policy.run_memory_policy(force=force, trigger=trigger, scope=scope, actor=actor)

    def capture_event(
        self,
        *,
        source_type: str,
        source_ref: str,
        payload: Any,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
        access_policy: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.mutations.capture_event(
            source_type=source_type,
            source_ref=source_ref,
            payload=payload,
            actor=actor,
            scope=scope,
            access_policy=access_policy,
            idempotency_key=idempotency_key,
        )

    def classify_refs(
        self,
        refs: Sequence[str],
        *,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.retrieval.classify_refs(
                refs,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
            )

    def commit_atom(
        self,
        atom: Mapping[str, Any],
        *,
        actor: str = "system",
        idempotency_key: str | None = None,
        authorization_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.mutations.commit_atom(
            atom,
            actor=actor,
            idempotency_key=idempotency_key,
            authorization_context=authorization_context,
        )

    def propose_memory_atoms(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.mutations.propose_memory_atoms(candidates, actor=actor, scope=scope)

    def ratify_proposal(
        self,
        *,
        proposal_ref: str,
        adjudication_ref: str,
        expected_version: int,
        actor: str,
        authorization_context: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.governance.ratify_proposal(
            proposal_ref=proposal_ref,
            adjudication_ref=adjudication_ref,
            expected_version=expected_version,
            actor=actor,
            authorization_context=authorization_context,
            idempotency_key=idempotency_key,
        )

    def resolve_proposal(
        self,
        *,
        proposal_ref: str,
        adjudication_ref: str,
        expected_version: int,
        actor: str,
        authorization_context: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.governance.resolve_proposal(
            proposal_ref=proposal_ref,
            adjudication_ref=adjudication_ref,
            expected_version=expected_version,
            actor=actor,
            authorization_context=authorization_context,
            idempotency_key=idempotency_key,
        )

    def replace_constitutional_record(
        self,
        *,
        current_ref: str,
        successor_ref: str,
        adjudication_ref: str,
        expected_current_version: int,
        expected_successor_version: int,
        actor: str,
        authorization_context: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.governance.replace_constitutional_record(
            current_ref=current_ref,
            successor_ref=successor_ref,
            adjudication_ref=adjudication_ref,
            expected_current_version=expected_current_version,
            expected_successor_version=expected_successor_version,
            actor=actor,
            authorization_context=authorization_context,
            idempotency_key=idempotency_key,
        )

    def analyze_provenance(
        self, *, atom_ref: str, max_depth: int = 64
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.governance.analyze_provenance(
                atom_ref=atom_ref, max_depth=max_depth
            )

    def diachronic_ratification_status(
        self,
        *,
        subject_ref: str,
        identity_ref: str,
        required_confirmations: int = 2,
        min_interval_seconds: int = 0,
        require_distinct_evidence: bool = False,
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.governance.diachronic_ratification_status(
                subject_ref=subject_ref,
                identity_ref=identity_ref,
                required_confirmations=required_confirmations,
                min_interval_seconds=min_interval_seconds,
                require_distinct_evidence=require_distinct_evidence,
            )

    def commit_memory_atoms(
        self,
        atoms: Sequence[Mapping[str, Any]],
        *,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.mutations.commit_memory_atoms(
            atoms,
            actor=actor,
            authorization_context=authorization_context,
            idempotency_key=idempotency_key,
        )

    def commit_memory_transaction(
        self,
        *,
        evidence: Sequence[Mapping[str, Any]] | None = None,
        atoms: Sequence[Mapping[str, Any]] | None = None,
        edges: Sequence[Mapping[str, Any]] | None = None,
        head_updates: Sequence[Mapping[str, Any]] | None = None,
        receipt_refs: Sequence[str] | None = None,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
        authorization_context: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.continuity.commit_memory_transaction(
            evidence=evidence,
            atoms=atoms,
            edges=edges,
            head_updates=head_updates,
            receipt_refs=receipt_refs,
            actor=actor,
            scope=scope,
            authorization_context=authorization_context,
            idempotency_key=idempotency_key,
        )

    def update_atom(
        self,
        atom_id: str,
        *,
        payload_patch: Mapping[str, Any] | None = None,
        set_fields: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.mutations.update_atom(
            atom_id,
            payload_patch=payload_patch,
            set_fields=set_fields,
            expected_version=expected_version,
            actor=actor,
            authorization_context=authorization_context,
            idempotency_key=idempotency_key,
        )

    def archive_atom(
        self,
        atom_id: str,
        *,
        reason: str = "archived",
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.mutations.archive_atom(
            atom_id,
            reason=reason,
            expected_version=expected_version,
            actor=actor,
            authorization_context=authorization_context,
        )

    def delete_atom(
        self,
        atom_id: str,
        *,
        reason: str,
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
        recreation_policy: str = "block_recreate",
    ) -> dict[str, Any]:
        return self.mutations.delete_atom(
            atom_id,
            reason=reason,
            expected_version=expected_version,
            actor=actor,
            authorization_context=authorization_context,
            recreation_policy=recreation_policy,
        )

    def request_deletion(
        self,
        *,
        target_ref: str,
        reason: str,
        requested_by: str = "system",
        expected_version: int | None = None,
        authorization_context: Mapping[str, Any] | None = None,
        recreation_policy: str = "block_recreate",
    ) -> dict[str, Any]:
        return self.mutations.request_deletion(
            target_ref=target_ref,
            reason=reason,
            requested_by=requested_by,
            expected_version=expected_version,
            authorization_context=authorization_context,
            recreation_policy=recreation_policy,
        )

    def merge_atoms(
        self,
        *,
        source_refs: Sequence[str],
        merged_payload: Mapping[str, Any],
        merged_type: str = "semantic",
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        return self.mutations.merge_atoms(
            source_refs=source_refs,
            merged_payload=merged_payload,
            merged_type=merged_type,
            scope=scope,
            actor=actor,
            approved_by=approved_by,
        )

    def retrieve_packet(
        self,
        *,
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        retrieval_mode: str = "general",
        memory_mode: str = "operational_recall",
        max_items: int | None = None,
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        include_conflicts: bool | None = None,
        include_archived: bool = False,
        include_low_health: bool = False,
        include_superseded: bool = False,
        type_filter: Sequence[str] | None = None,
        payload_filter: Mapping[str, Sequence[Any]] | None = None,
        result_payload_filter: Mapping[str, Sequence[Any]] | None = None,
        attention_context: Mapping[str, Any] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        if run_policy:
            self.policy.run_memory_policy(
                trigger="retrieve_packet", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.retrieval.retrieve_packet(
                cues=cues,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
                retrieval_mode=retrieval_mode,
                memory_mode=memory_mode,
                max_items=max_items,
                token_or_byte_budget=token_or_byte_budget,
                include_conflicts=include_conflicts,
                include_archived=include_archived,
                include_low_health=include_low_health,
                include_superseded=include_superseded,
                type_filter=type_filter,
                payload_filter=payload_filter,
                result_payload_filter=result_payload_filter,
                attention_context=attention_context,
                run_policy=run_policy,
                _policy_already_run=True,
            )

    def retrieve_atom(
        self,
        atom_id: str,
        *,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        memory_mode: str = "operational_recall",
        include_conflicts: bool = False,
        include_archived: bool = False,
        include_low_health: bool = False,
        include_superseded: bool = False,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        """Resolve a known atom ID without invoking associative ranking."""

        if run_policy:
            self.policy.run_memory_policy(
                trigger="retrieve_atom", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.retrieval.retrieve_atom(
                atom_id,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
                memory_mode=memory_mode,
                include_conflicts=include_conflicts,
                include_archived=include_archived,
                include_low_health=include_low_health,
                include_superseded=include_superseded,
                run_policy=run_policy,
                _policy_already_run=True,
            )

    def retrieve_evidence(
        self,
        evidence_id: str,
        *,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        run_policy: bool = True,
    ) -> dict[str, Any]:
        """Resolve a known evidence ID without associative ranking."""

        if run_policy:
            self.policy.run_memory_policy(
                trigger="retrieve_evidence", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.retrieval.retrieve_evidence(
                evidence_id,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
                run_policy=run_policy,
                _policy_already_run=True,
            )

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
        memory_mode: str = "operational_recall",
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        if run_policy:
            self.policy.run_memory_policy(
                trigger="compile_memory_frame", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.reasoning.compile_memory_frame(
                need=need,
                purpose=purpose,
                depth=depth,
                task_context=task_context,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
                memory_mode=memory_mode,
                token_or_byte_budget=token_or_byte_budget,
                run_policy=run_policy,
                _policy_already_run=True,
            )

    def compile_cognitive_workspace(
        self,
        *,
        current_event_ref: str,
        conversation_id: str,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        participant_refs: Sequence[str] | None = None,
        operation_refs: Sequence[str] | None = None,
        project_refs: Sequence[str] | None = None,
        context_refs: Sequence[str] | None = None,
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        temporal_limit: int = 12,
        thread_limit: int = 8,
        recent_event_floor: int = 4,
        prior_workspace_revision: Mapping[str, Any] | None = None,
        run_policy: bool = False,
    ) -> dict[str, Any]:
        if run_policy:
            self.policy.run_memory_policy(
                trigger="compile_cognitive_workspace", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.continuity.compile_cognitive_workspace(
                current_event_ref=current_event_ref,
                conversation_id=conversation_id,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
                participant_refs=participant_refs,
                operation_refs=operation_refs,
                project_refs=project_refs,
                context_refs=context_refs,
                token_or_byte_budget=token_or_byte_budget,
                temporal_limit=temporal_limit,
                thread_limit=thread_limit,
                recent_event_floor=recent_event_floor,
                prior_workspace_revision=prior_workspace_revision,
                run_policy=run_policy,
                _policy_already_run=True,
            )

    def compile_interaction_projection(
        self,
        *,
        conversation_id: str,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "participant-ui",
        after_sequence: int = 0,
        limit: int = 1000,
        linked_atom_types: Sequence[str] | None = None,
        linked_depth: int = 0,
        linked_limit_per_event: int = 32,
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.continuity.compile_interaction_projection(
                conversation_id=conversation_id,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
                after_sequence=after_sequence,
                limit=limit,
                linked_atom_types=linked_atom_types,
                linked_depth=linked_depth,
                linked_limit_per_event=linked_limit_per_event,
            )

    def get_memory_head(
        self,
        *,
        scope: Mapping[str, Any] | None,
        series_kind: str,
        series_id: str,
        requester: str = "system",
        target_processor: str = "reasoner",
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.continuity.get_memory_head(
                scope=scope,
                series_kind=series_kind,
                series_id=series_id,
                requester=requester,
                target_processor=target_processor,
            )

    def get_memory_series_versions(
        self,
        *,
        scope: Mapping[str, Any] | None,
        series_kind: str,
        series_id: str,
        versions: Sequence[int] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        limit: int = 100,
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.continuity.get_memory_series_versions(
                scope=scope,
                series_kind=series_kind,
                series_id=series_id,
                versions=versions,
                requester=requester,
                target_processor=target_processor,
                limit=limit,
            )

    def observe_memory_transaction(
        self,
        *,
        event_id: str,
        scope: Mapping[str, Any] | None,
        requester: str = "system",
        target_processor: str = "reasoner",
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.continuity.observe_memory_transaction(
                event_id=event_id,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
            )

    def rebuild_memory_heads(self) -> dict[str, Any]:
        return self.continuity.rebuild_memory_heads()

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
        memory_mode: str = "operational_recall",
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        if run_policy:
            self.policy.run_memory_policy(
                trigger="load_memory_page", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.reasoning.load_memory_page(
                frame_id=frame_id,
                revision=revision,
                page=page,
                need=need,
                purpose=purpose,
                depth=depth,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
                memory_mode=memory_mode,
                token_or_byte_budget=token_or_byte_budget,
                run_policy=run_policy,
                _policy_already_run=True,
            )

    def record_retrieval_outcome(
        self,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.retrieval.record_retrieval_outcome(
            packet_id=packet_id,
            request=request,
            outcome=outcome,
        )

    def distill_memories(
        self,
        *,
        target_refs: Sequence[str],
        summary: str | Mapping[str, Any],
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        idempotency_key: str | None = None,
        distillation_type: str = "summary",
        archive_sources: bool = False,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        return self.mutations.distill_memories(
            target_refs=target_refs,
            summary=summary,
            scope=scope,
            actor=actor,
            idempotency_key=idempotency_key,
            distillation_type=distillation_type,
            archive_sources=archive_sources,
            approved_by=approved_by,
        )

    def record_runtime_state(
        self,
        *,
        agent_id: str,
        capabilities: Mapping[str, Any] | None = None,
        denied_capabilities: Sequence[str] | None = None,
        constraints: Sequence[str] | None = None,
        load: Mapping[str, Any] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.views.record_runtime_state(
            agent_id=agent_id,
            capabilities=capabilities,
            denied_capabilities=denied_capabilities,
            constraints=constraints,
            load=load,
            scope=scope,
            actor=actor,
        )

    def record_self_assessment(
        self,
        *,
        agent_id: str,
        claim: str,
        calibration: Mapping[str, Any],
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.views.record_self_assessment(
            agent_id=agent_id,
            claim=claim,
            calibration=calibration,
            scope=scope,
            actor=actor,
        )

    def generate_self_narrative(
        self,
        *,
        agent_id: str,
        narrative: str,
        source_refs: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.views.generate_self_narrative(
            agent_id=agent_id,
            narrative=narrative,
            source_refs=source_refs,
            scope=scope,
            actor=actor,
        )

    def record_agentic_trace(
        self,
        *,
        agent_id: str,
        task: str,
        action: str,
        outcome: str,
        lesson: str | None = None,
        external_constraints: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.views.record_agentic_trace(
            agent_id=agent_id,
            task=task,
            action=action,
            outcome=outcome,
            lesson=lesson,
            external_constraints=external_constraints,
            scope=scope,
            actor=actor,
        )

    def record_action_outcome(
        self,
        *,
        agent_id: str,
        action_ref: str,
        status: str,
        evidence_refs: Sequence[str] | None = None,
        correction: str | None = None,
        limitation: str | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.views.record_action_outcome(
            agent_id=agent_id,
            action_ref=action_ref,
            status=status,
            evidence_refs=evidence_refs,
            correction=correction,
            limitation=limitation,
            scope=scope,
            actor=actor,
        )

    def retrieve_self_awareness(
        self,
        *,
        agent_id: str,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "self-model",
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.views.retrieve_self_awareness(
                agent_id=agent_id,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
            )

    def calibrate_self_model(
        self,
        *,
        agent_id: str,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        record: bool = False,
    ) -> dict[str, Any]:
        if record:
            return self.views.calibrate_self_model(
                agent_id=agent_id,
                scope=scope,
                actor=actor,
                record=True,
            )
        with self.store.read_snapshot():
            return self.views.calibrate_self_model(
                agent_id=agent_id,
                scope=scope,
                actor=actor,
                record=False,
            )

    def retrieve_agentic_recall(
        self,
        *,
        agent_id: str,
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "planner",
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.views.retrieve_agentic_recall(
                agent_id=agent_id,
                cues=cues,
                scope=scope,
                requester=requester,
                target_processor=target_processor,
            )

    def retrieve_shared_view(
        self,
        *,
        processor_ids: Sequence[str],
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        max_items: int = 20,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        if run_policy:
            self.policy.run_memory_policy(
                trigger="retrieve_shared_view", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.views.retrieve_shared_view(
                processor_ids=processor_ids,
                cues=cues,
                scope=scope,
                requester=requester,
                max_items=max_items,
            )

    def refresh_shared_view(
        self,
        *,
        processor_ids: Sequence[str],
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        max_items: int = 20,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        if run_policy:
            self.policy.run_memory_policy(
                trigger="refresh_shared_view", scope=scope or {}
            )
        with self.store.read_snapshot():
            return self.views.refresh_shared_view(
                processor_ids=processor_ids,
                cues=cues,
                scope=scope,
                requester=requester,
                max_items=max_items,
            )

    def evaluate_procedure_execution(
        self,
        *,
        procedure_ref: str,
        autonomous: bool = False,
        approved_by: str | None = None,
        tool_permission_binding: Mapping[str, Any] | None = None,
        preconditions_satisfied: bool = False,
        rollback_plan: Mapping[str, Any] | None = None,
        review_status: str | None = None,
    ) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self.views.evaluate_procedure_execution(
                procedure_ref=procedure_ref,
                autonomous=autonomous,
                approved_by=approved_by,
                tool_permission_binding=tool_permission_binding,
                preconditions_satisfied=preconditions_satisfied,
                rollback_plan=rollback_plan,
                review_status=review_status,
            )

    def llm_reviewer_policy(self) -> dict[str, Any]:
        return self.stewardship.llm_reviewer_policy()

    def request_maintenance(
        self,
        *,
        action: str,
        target_refs: Sequence[str] | None = None,
        risk: str = "low",
        approved_by: str | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.stewardship.request_maintenance(
            action=action,
            target_refs=target_refs,
            risk=risk,
            approved_by=approved_by,
            scope=scope,
            actor=actor,
        )

    def run_maintenance_distiller(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        actor: str = "svc:memory_policy",
        domain: str = "generic",
        processor_ids: Sequence[str] | None = None,
        max_atoms: int = 128,
        max_events: int = 64,
        max_retrieval_outcomes: int = 64,
        auto_commit_low_risk: bool = True,
        reviewer: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.stewardship.run_maintenance_distiller(
            scope=scope,
            actor=actor,
            domain=domain,
            processor_ids=processor_ids,
            max_atoms=max_atoms,
            max_events=max_events,
            max_retrieval_outcomes=max_retrieval_outcomes,
            auto_commit_low_risk=auto_commit_low_risk,
            reviewer=reviewer,
        )

    def run_smp_analysis(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        target_refs: Sequence[str] | None = None,
        max_atoms: int | None = None,
    ) -> dict[str, Any]:
        return self.stewardship.run_smp_analysis(
            scope=scope,
            target_refs=target_refs,
            max_atoms=max_atoms,
        )

    def run_steward(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        return self.stewardship.run_steward(scope=scope, actor=actor, approved_by=approved_by)

    def health_memory(self, *, run_policy: bool = True) -> dict[str, Any]:
        return self.diagnostics.health_memory(run_policy=run_policy)

    def health_capacity(self) -> dict[str, Any]:
        return self.diagnostics.health_capacity()

    def verify_journal_chain(self) -> dict[str, Any]:
        return self.diagnostics.verify_journal_chain()

    def replay_graph(self) -> dict[str, Any]:
        return self.diagnostics.replay_graph()

    def verify_replay(self) -> dict[str, Any]:
        return self.diagnostics.verify_replay()

    def verify_integrity(self) -> dict[str, Any]:
        return self.diagnostics.verify_integrity()
