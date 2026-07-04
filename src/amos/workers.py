"""Worker artifacts for AMOS v1 operations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .service import Amos


class JournalProjector:
    def __init__(self, amos: Amos):
        self.amos = amos

    def verify_projection(self) -> dict[str, Any]:
        return {
            "journal": self.amos.verify_journal_chain(),
            "replay": self.amos.verify_replay(),
        }


class IndexMaintainer:
    def __init__(self, amos: Amos):
        self.amos = amos

    def rebuild(self) -> dict[str, Any]:
        atoms = self.amos.store.list_atoms()
        edges = self.amos.store.list_edges()
        graph_version = self.amos.store.graph_version()
        with self.amos.store.transaction() as conn:
            lexical = self.amos.store.upsert_derived_index_metadata(
                conn,
                index_name="semantic_lexical_vectors",
                graph_version=graph_version,
                freshness="fresh",
                details={
                    "atom_count": len([atom for atom in atoms if not atom.get("deleted")]),
                    "processor_id": self.amos.smp.processor_id,
                    "rebuildable_from_canonical": True,
                },
            )
            graph = self.amos.store.upsert_derived_index_metadata(
                conn,
                index_name="graph_adjacency",
                graph_version=graph_version,
                freshness="fresh",
                details={
                    "edge_count": len(edges),
                    "rebuildable_from_canonical": True,
                },
            )
        return {
            "status": "rebuilt",
            "graph_version": graph_version,
            "indexes": [lexical, graph],
        }


class PacketCacheInvalidator:
    def __init__(self, amos: Amos):
        self.amos = amos

    def invalidate(self) -> dict[str, Any]:
        with self.amos.store.transaction() as conn:
            self.amos.store.clear_packet_cache(conn)
        return {"status": "invalidated", "graph_version": self.amos.store.graph_version()}


class CapacityGovernor:
    def __init__(self, amos: Amos):
        self.amos = amos

    def configure(
        self,
        *,
        hard_capacity_bytes: int,
        warning_ratio: float = 0.70,
        critical_ratio: float = 0.90,
    ) -> dict[str, Any]:
        return self.amos.configure_capacity_budget(
            hard_capacity_bytes=hard_capacity_bytes,
            warning_ratio=warning_ratio,
            critical_ratio=critical_ratio,
        )

    def report(self) -> dict[str, Any]:
        return self.amos.health_capacity()


class MemorySteward:
    def __init__(self, amos: Amos):
        self.amos = amos

    def run(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        return self.amos.run_steward(scope=scope, approved_by=approved_by)


class MemoryPolicyWorker:
    def __init__(self, amos: Amos):
        self.amos = amos

    def tick(
        self,
        *,
        force: bool = False,
        trigger: str = "worker",
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.amos.run_memory_policy(
            force=force,
            trigger=trigger,
            scope=scope,
            actor="svc:memory_policy",
        )


class DistillerMaintenanceWorker:
    def __init__(self, amos: Amos):
        self.amos = amos

    def tick(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        domain: str = "generic",
        processor_ids: Sequence[str] | None = None,
        auto_commit_low_risk: bool = True,
    ) -> dict[str, Any]:
        return self.amos.run_maintenance_distiller(
            scope=scope,
            actor="svc:maintenance_distiller",
            domain=domain,
            processor_ids=processor_ids,
            auto_commit_low_risk=auto_commit_low_risk,
        )


class SelfModelCalibrator:
    def __init__(self, amos: Amos):
        self.amos = amos

    def run(
        self,
        *,
        agent_id: str,
        scope: Mapping[str, Any] | None = None,
        record: bool = True,
    ) -> dict[str, Any]:
        return self.amos.calibrate_self_model(
            agent_id=agent_id, scope=scope, record=record
        )


class AgenticRecallAuditor:
    def __init__(self, amos: Amos):
        self.amos = amos

    def audit(
        self,
        *,
        agent_id: str,
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        recall = self.amos.retrieve_agentic_recall(
            agent_id=agent_id, cues=cues, scope=scope
        )
        balance = {
            "success_count": len(recall["successes"]),
            "failure_count": len(recall["failures"]),
            "blocked_count": len(recall["blocked"]),
            "correction_count": len(recall["corrections"]),
            "other_agent_count": len(recall["other_agent_actions"]),
            "shared_system_count": len(recall["shared_system_actions"]),
            "external_count": len(recall["external_actions"]),
            "unknown_responsibility_count": len(
                recall["unknown_responsibility_actions"]
            ),
            "material_counterevidence_count": len(recall["material_counterevidence"]),
            "expired_self_narrative_count": len(recall["expired_self_narratives"]),
        }
        return {
            "status": "audited",
            "agent_id": agent_id,
            "graph_version": recall["graph_version"],
            "balance": balance,
            "source_packet_id": recall["source_packet_id"],
        }


class SMPWorker:
    def __init__(self, amos: Amos):
        self.amos = amos

    def run(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        target_refs: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return self.amos.run_smp_analysis(scope=scope, target_refs=target_refs)
