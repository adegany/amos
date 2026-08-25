"""Worker artifacts for AMOS v1 operations."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from typing import Any, Mapping, Sequence

from .errors import ValidationError
from .service import Amos


class ExpiringMaintenanceLeaseGate:
    """Bounded, crash-safe admission gate for background maintenance.

    Leases intentionally live in the AMOS process rather than canonical memory:
    they coordinate transient foreground recovery and disappear on restart. A
    monotonic expiry makes a dead holder self-releasing without trusting wall
    clock adjustments.
    """

    DEFAULT_TTL_SECONDS = 180.0
    MAX_TTL_SECONDS = 900.0

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        epoch_time: Callable[[], float] = time.time,
    ):
        self._monotonic = monotonic
        self._epoch_time = epoch_time
        self._lock = threading.Lock()
        self._leases: dict[str, dict[str, Any]] = {}

    def acquire(
        self,
        *,
        owner_ref: str,
        reason: str,
        ttl_seconds: float | int | None = None,
    ) -> dict[str, Any]:
        owner_ref = str(owner_ref or "").strip()
        reason = str(reason or "").strip()
        if not owner_ref:
            raise ValidationError("maintenance lease owner_ref is required")
        if not reason:
            raise ValidationError("maintenance lease reason is required")
        ttl = self._normalize_ttl(ttl_seconds)
        now = self._monotonic()
        lease_id = "maintenance-lease:" + secrets.token_hex(16)
        with self._lock:
            self._prune_locked(now)
            replaced = [
                existing_id
                for existing_id, lease in self._leases.items()
                if lease["owner_ref"] == owner_ref
                and lease["reason"] == reason
            ]
            for existing_id in replaced:
                self._leases.pop(existing_id, None)
            self._leases[lease_id] = {
                "lease_id": lease_id,
                "owner_ref": owner_ref,
                "reason": reason,
                "acquired_at_epoch_seconds": self._epoch_time(),
                "expires_at_monotonic": now + ttl,
                "ttl_seconds": ttl,
            }
            return {
                **self._public_lease_locked(
                    self._leases[lease_id], now=now
                ),
                "replaced_lease_count": len(replaced),
            }

    def renew(
        self,
        *,
        lease_id: str,
        ttl_seconds: float | int | None = None,
    ) -> dict[str, Any]:
        lease_id = str(lease_id or "").strip()
        ttl = self._normalize_ttl(ttl_seconds)
        now = self._monotonic()
        with self._lock:
            self._prune_locked(now)
            lease = self._leases.get(lease_id)
            if lease is None:
                return {
                    "status": "not_found",
                    "lease_id": lease_id,
                    "active": False,
                }
            lease["expires_at_monotonic"] = now + ttl
            lease["ttl_seconds"] = ttl
            return {
                **self._public_lease_locked(lease, now=now),
                "status": "renewed",
            }

    def release(self, *, lease_id: str) -> dict[str, Any]:
        lease_id = str(lease_id or "").strip()
        now = self._monotonic()
        with self._lock:
            self._prune_locked(now)
            lease = self._leases.pop(lease_id, None)
            return {
                "status": "released" if lease is not None else "not_found",
                "lease_id": lease_id,
                "active": False,
            }

    def admission(self) -> dict[str, Any]:
        now = self._monotonic()
        with self._lock:
            self._prune_locked(now)
            leases = [
                self._public_lease_locked(lease, now=now)
                for lease in self._leases.values()
            ]
        retry_after = min(
            (float(lease["remaining_seconds"]) for lease in leases),
            default=0.0,
        )
        return {
            "allowed": not leases,
            "reason": None if not leases else "foreground_recovery_lease_active",
            "active_count": len(leases),
            "retry_after_seconds": round(retry_after, 3),
            "leases": sorted(leases, key=lambda lease: str(lease["lease_id"])),
        }

    def status(self) -> dict[str, Any]:
        admission = self.admission()
        return {
            "status": "open" if admission["allowed"] else "held",
            **admission,
            "default_ttl_seconds": self.DEFAULT_TTL_SECONDS,
            "max_ttl_seconds": self.MAX_TTL_SECONDS,
        }

    def _normalize_ttl(self, ttl_seconds: float | int | None) -> float:
        try:
            ttl = (
                self.DEFAULT_TTL_SECONDS
                if ttl_seconds is None
                else float(ttl_seconds)
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "maintenance lease ttl_seconds must be numeric"
            ) from exc
        if ttl <= 0:
            raise ValidationError(
                "maintenance lease ttl_seconds must be positive"
            )
        return min(ttl, self.MAX_TTL_SECONDS)

    def _prune_locked(self, now: float) -> None:
        expired = [
            lease_id
            for lease_id, lease in self._leases.items()
            if float(lease["expires_at_monotonic"]) <= now
        ]
        for lease_id in expired:
            self._leases.pop(lease_id, None)

    def _public_lease_locked(
        self, lease: Mapping[str, Any], *, now: float
    ) -> dict[str, Any]:
        remaining = max(0.0, float(lease["expires_at_monotonic"]) - now)
        return {
            "status": "acquired",
            "lease_id": str(lease["lease_id"]),
            "owner_ref": str(lease["owner_ref"]),
            "reason": str(lease["reason"]),
            "active": remaining > 0,
            "ttl_seconds": float(lease["ttl_seconds"]),
            "remaining_seconds": round(remaining, 3),
            "acquired_at_epoch_seconds": float(
                lease["acquired_at_epoch_seconds"]
            ),
            "expires_at_epoch_seconds": round(self._epoch_time() + remaining, 3),
        }


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
        return self.amos.indexes.rebuild(
            graph_version=self.amos.store.graph_version()
        )


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
        interval_seconds: float = 60.0,
        actor: str = "svc:memory_policy",
        execution_lock: threading.Lock | threading.RLock | None = None,
        maintenance_admission: Callable[[], Mapping[str, Any]] | None = None,
    ):
        self.amos = amos
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.actor = actor
        self.execution_lock = execution_lock
        self.maintenance_admission = maintenance_admission
        self._condition = threading.Condition()
        self._pending: list[dict[str, Any]] = []
        self._running = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._run_count = 0
        self._error_count = 0
        self._deferred_count = 0
        self._last_deferred: dict[str, Any] | None = None

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
                "deferred_count": self._deferred_count,
                "last_deferred": self._last_deferred,
                "last_result": self._last_result,
                "last_error": self._last_error,
                "maintenance_admission": (
                    dict(self.maintenance_admission())
                    if self.maintenance_admission is not None
                    else {"allowed": True, "reason": None}
                ),
            }

    def _loop(self) -> None:
        while True:
            deferred = False
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
                if self.execution_lock is None:
                    result = self._run_admitted_request(request)
                else:
                    with self.execution_lock:
                        result = self._run_admitted_request(request)
                compact = self._compact_result(result)
                with self._condition:
                    self._last_result = compact
                    self._last_error = None
                    if compact.get("reason") == (
                        "foreground_recovery_lease_active"
                    ):
                        deferred = True
                        self._deferred_count += 1
                        self._last_deferred = compact
                        if (
                            request.get("trigger") != "background_interval"
                            and request not in self._pending
                        ):
                            self._pending.insert(0, request)
                    else:
                        self._run_count += 1
            except Exception as exc:  # pragma: no cover - defensive service guard
                with self._condition:
                    self._last_error = str(exc)
                    self._error_count += 1
            finally:
                with self._condition:
                    self._running = False
                    if deferred and not self._stop:
                        self._condition.wait(timeout=self.interval_seconds)

    def _run_admitted_request(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        admission = (
            dict(self.maintenance_admission())
            if self.maintenance_admission is not None
            else {"allowed": True, "reason": None}
        )
        if not bool(admission.get("allowed", True)):
            return {
                "status": "deferred",
                "reason": str(
                    admission.get("reason")
                    or "maintenance_admission_deferred"
                ),
                "trigger": str(request["trigger"]),
                "maintenance_admission": admission,
                "graph_version": self.amos.store.graph_version(),
            }
        return self._run_request(request)

    def _run_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.amos.run_memory_policy(
            force=bool(request["force"]),
            trigger=str(request["trigger"]),
            scope=dict(request["scope"]),
            actor=self.actor,
        )

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
        admission = result.get("maintenance_admission")
        if isinstance(admission, Mapping):
            compact["maintenance_admission"] = dict(admission)
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
