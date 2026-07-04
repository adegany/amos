"""Worker artifacts for AMOS v1 operations."""

from __future__ import annotations

import threading
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


class BackgroundMemoryPolicyWorker:
    """Daemon worker for service-owned automatic memory policy maintenance."""

    def __init__(
        self,
        amos: Amos,
        *,
        interval_seconds: float = 5.0,
        actor: str = "svc:memory_policy",
    ):
        self.amos = amos
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.actor = actor
        self._condition = threading.Condition()
        self._pending: list[dict[str, Any]] = []
        self._running = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._run_count = 0
        self._error_count = 0

    def start(self) -> dict[str, Any]:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._stop = False
            self._thread = threading.Thread(
                target=self._loop,
                name="amos-memory-policy-worker",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def stop(self, *, timeout: float = 5.0) -> dict[str, Any]:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        return self.status()

    def request_tick(
        self,
        *,
        trigger: str = "background_request",
        scope: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        request = {
            "trigger": str(trigger or "background_request"),
            "scope": dict(scope or {}),
            "force": bool(force),
        }
        with self._condition:
            if self._stop:
                return {
                    "status": "skipped",
                    "reason": "worker_stopping",
                    "trigger": request["trigger"],
                }
            if any(
                pending["trigger"] == request["trigger"]
                and pending["scope"] == request["scope"]
                and pending["force"] == request["force"]
                for pending in self._pending
            ):
                return {
                    "status": "queued",
                    "reason": "already_queued",
                    "trigger": request["trigger"],
                    "pending_count": len(self._pending),
                }
            self._pending.append(request)
            self._condition.notify_all()
            return {
                "status": "queued",
                "trigger": request["trigger"],
                "pending_count": len(self._pending),
            }

    def status(self) -> dict[str, Any]:
        with self._condition:
            thread_alive = self._thread is not None and self._thread.is_alive()
            return {
                "status": "active" if thread_alive and not self._stop else "stopped",
                "interval_seconds": self.interval_seconds,
                "running": self._running,
                "pending_count": len(self._pending),
                "run_count": self._run_count,
                "error_count": self._error_count,
                "last_result": self._last_result,
                "last_error": self._last_error,
            }

    def _loop(self) -> None:
        while True:
            with self._condition:
                if not self._pending and not self._stop:
                    self._condition.wait(timeout=self.interval_seconds)
                if self._stop:
                    return
                if self._pending:
                    request = self._pending.pop(0)
                else:
                    request = {
                        "trigger": "background_interval",
                        "scope": {},
                        "force": False,
                    }
                self._running = True
            try:
                result = self.amos.run_memory_policy(
                    force=bool(request["force"]),
                    trigger=str(request["trigger"]),
                    scope=dict(request["scope"]),
                    actor=self.actor,
                )
                compact = self._compact_result(result)
                with self._condition:
                    self._last_result = compact
                    self._last_error = None
                    self._run_count += 1
            except Exception as exc:  # pragma: no cover - defensive service guard
                with self._condition:
                    self._last_error = str(exc)
                    self._error_count += 1
            finally:
                with self._condition:
                    self._running = False

    def _compact_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        compact = {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "trigger": result.get("trigger"),
            "graph_version": result.get("graph_version"),
        }
        due = result.get("due")
        if isinstance(due, Mapping):
            compact["due"] = {
                "due": due.get("due"),
                "reasons": list(due.get("reasons", [])),
                "graph_delta": due.get("graph_delta"),
                "elapsed_seconds": due.get("elapsed_seconds"),
            }
        results = result.get("results")
        if isinstance(results, Mapping):
            compact["result_keys"] = sorted(str(key) for key in results)
        event = result.get("event")
        if isinstance(event, Mapping):
            compact["event_id"] = event.get("event_id")
        return compact


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
