"""Minimal stdlib HTTP adapter for the AMOS v1 API surface."""

from __future__ import annotations

import hmac
import json
import queue
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from .errors import (
    AccessDenied,
    AmosError,
    CASConflict,
    CognitiveWorkspaceBudgetExceeded,
    RequestDeadlineExceeded,
    StaleFrameError,
    ValidationError,
)
from .governance_service import GovernanceService
from .request_context import remaining_seconds, request_context
from .schemas import CONSTITUTIONAL_ATOM_TYPES
from .service import Amos
from .workers import BackgroundMemoryPolicyWorker


class _FairAdmission:
    """FIFO bounded admission with deadline-aware waiter cancellation."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._condition = threading.Condition()
        self._waiters: deque[object] = deque()
        self._active = 0

    def acquire(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        waiter = object()
        with self._condition:
            self._waiters.append(waiter)
            while self._waiters[0] is not waiter or self._active >= self.capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                    return False
                self._condition.wait(timeout=remaining)
            self._waiters.popleft()
            self._active += 1
            self._condition.notify_all()
            return True

    def release(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()

    def status(self) -> dict[str, int | str]:
        with self._condition:
            return {
                "policy": "fifo",
                "capacity": self.capacity,
                "active": self._active,
                "waiting": len(self._waiters),
            }


class _SingleFlight:
    """Coalesce identical reads until the leader publishes its cache entry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._flights: dict[str, threading.Event] = {}

    def enter(self, key: str) -> tuple[bool, threading.Event]:
        with self._lock:
            event = self._flights.get(key)
            if event is not None:
                return False, event
            event = threading.Event()
            self._flights[key] = event
            return True, event

    def finish(self, key: str, event: threading.Event) -> None:
        with self._lock:
            if self._flights.get(key) is event:
                self._flights.pop(key, None)
                event.set()

    def status(self) -> dict[str, int]:
        with self._lock:
            return {"in_flight": len(self._flights)}


class AmosHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64
    REQUEST_SERVICE_COUNT = 4
    HEAVY_REQUEST_CAPACITY = 2
    DEFAULT_REQUEST_DEADLINE_SECONDS = 30.0
    HEAVY_PATHS = frozenset({
        "/v1/packets:retrieve",
        "/v1/cognitive-workspaces:compile",
        "/v1/interaction-projections:compile",
        "/v1/reasoning-frames:compile",
        "/v1/reasoning-pages:load",
        "/v1/self-awareness:retrieve",
        "/v1/agentic-recall:retrieve",
        "/v1/shared-views:retrieve",
    })

    def __init__(
        self,
        server_address: tuple[str, int],
        db_path: str,
        *,
        maintenance_processor_paths: list[str] | None = None,
        governance_principals: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.db_path = db_path
        self.maintenance_processor_paths = list(maintenance_processor_paths or [])
        self.governance_principals = {
            str(token): dict(principal)
            for token, principal in dict(governance_principals or {}).items()
            if str(token)
        }
        self.amos = Amos(
            db_path,
            maintenance_processor_paths=self.maintenance_processor_paths,
        )
        self.request_services = [
            self.amos,
            *[
                Amos(
                    db_path,
                    maintenance_processor_paths=self.maintenance_processor_paths,
                )
                for _ in range(self.REQUEST_SERVICE_COUNT - 1)
            ],
        ]
        self.request_service_pool: queue.Queue[Amos] = queue.Queue(
            maxsize=self.REQUEST_SERVICE_COUNT
        )
        # Keep the public primary service first so ordinary sequential requests
        # retain the existing test/embedding seam while overlapping requests
        # receive isolated SQLite connections.
        self.request_service_pool.put(self.amos)
        for service in self.request_services[1:]:
            self.request_service_pool.put(service)
        self.heavy_admission = _FairAdmission(self.HEAVY_REQUEST_CAPACITY)
        self.retrieval_singleflight = _SingleFlight()
        self.health_amos = Amos(
            db_path,
            maintenance_processor_paths=self.maintenance_processor_paths,
        )
        self.inventory_amos = Amos(
            db_path,
            maintenance_processor_paths=self.maintenance_processor_paths,
        )
        self.ready_amos = Amos(
            db_path,
            maintenance_processor_paths=self.maintenance_processor_paths,
        )
        self.health_lock = threading.Lock()
        self.inventory_lock = threading.Lock()
        self.ready_lock = threading.Lock()
        self.maintenance_amos = Amos(
            db_path,
            maintenance_processor_paths=self.maintenance_processor_paths,
        )
        self.policy_worker_amos = Amos(
            db_path,
            maintenance_processor_paths=self.maintenance_processor_paths,
        )
        self.maintenance_lock = threading.Lock()
        self.memory_policy_worker = BackgroundMemoryPolicyWorker(
            self.policy_worker_amos,
            execution_lock=self.maintenance_lock,
        )
        self.memory_policy_worker.start()
        self.service_lock = threading.RLock()
        self.closing = False
        super().__init__(server_address, make_handler())

    def server_close(self) -> None:
        self.memory_policy_worker.stop(timeout=30.0)
        with self.service_lock:
            self.closing = True
        try:
            super().server_close()
        finally:
            with self.maintenance_lock:
                self.maintenance_amos.close()
            with self.service_lock:
                self.policy_worker_amos.close()
                self.health_amos.close()
                self.inventory_amos.close()
                self.ready_amos.close()
                for service in self.request_services:
                    service.close()


def make_handler() -> type[BaseHTTPRequestHandler]:
    class AmosHandler(BaseHTTPRequestHandler):
        server_version = "AmosHTTP/1.0"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def log_message(self, _format: str, *args: Any) -> None:
            return

        def _handle(self, method: str) -> None:
            deadline = self._request_deadline_epoch()
            request_id = str(self.headers.get("X-Request-ID") or "") or None
            with request_context(
                deadline_epoch_seconds=deadline,
                request_id=request_id,
            ):
                self._handle_with_context(method)

        def _handle_with_context(self, method: str) -> None:
            # A timed-out client must not leave a handler waiting for another
            # request on a half-closed keep-alive socket.
            self.close_connection = True
            try:
                body = self._read_json() if method == "POST" else {}
                server = cast(AmosHTTPServer, self.server)
                path = self.path.split("?", 1)[0]
                remaining = self._request_deadline_remaining()
                if remaining <= 0:
                    self._write_json(
                        {
                            "status": "error",
                            "error": "request deadline exhausted before dispatch",
                            "code": "request_deadline_exhausted",
                            "retryable": True,
                        },
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
                if method == "POST" and path in {
                    "/v1/maintenance-distiller:run",
                    "/v1/memory-policy:run",
                }:
                    with server.service_lock:
                        closing = server.closing
                    if closing:
                        self._write_json(
                            {
                                "status": "error",
                                "error": "server is shutting down",
                                "retryable": True,
                            },
                            status=HTTPStatus.SERVICE_UNAVAILABLE,
                        )
                        return
                    # A semantic processor may make bounded model calls. Keep
                    # that work out of the request-service pool and serialize
                    # it with the service-owned policy worker so duplicate
                    # distillations cannot contend for the same model substrate.
                    if not server.maintenance_lock.acquire(timeout=remaining):
                        self._write_json(
                            {
                                "status": "error",
                                "error": (
                                    "AMOS maintenance capacity was unavailable "
                                    "before deadline"
                                ),
                                "code": "maintenance_capacity_exhausted",
                                "retryable": True,
                            },
                            status=HTTPStatus.SERVICE_UNAVAILABLE,
                        )
                        return
                    try:
                        if path == "/v1/maintenance-distiller:run":
                            return self._write_json(
                                server.maintenance_amos.run_maintenance_distiller(
                                    **body
                                )
                            )
                        return self._write_json(
                            server.policy_worker_amos.run_memory_policy(**body)
                        )
                    finally:
                        server.maintenance_lock.release()
                if method == "GET" and path == "/v1/ready":
                    if not server.ready_lock.acquire(timeout=remaining):
                        return self._write_saturated()
                    try:
                        return self._dispatch(
                            server, method, body, amos=server.ready_amos
                        )
                    finally:
                        server.ready_lock.release()
                if method == "GET" and path in {
                    "/v1/health/memory", "/v1/health/capacity",
                }:
                    if not server.health_lock.acquire(timeout=remaining):
                        return self._write_saturated()
                    try:
                        return self._dispatch(
                            server, method, body, amos=server.health_amos
                        )
                    finally:
                        server.health_lock.release()
                if method == "GET" and path == "/v1/health/memory-inventory":
                    if not server.inventory_lock.acquire(timeout=remaining):
                        return self._write_saturated()
                    try:
                        return self._dispatch(
                            server, method, body, amos=server.inventory_amos
                        )
                    finally:
                        server.inventory_lock.release()
                request_amos: Amos | None = None
                heavy_acquired = False
                flight_key: str | None = None
                flight_event: threading.Event | None = None
                flight_leader = False
                try:
                    if method == "POST" and path == "/v1/packets:retrieve":
                        flight_key = json.dumps(
                            {"path": path, "body": body},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        flight_leader, flight_event = (
                            server.retrieval_singleflight.enter(flight_key)
                        )
                        if not flight_leader and not flight_event.wait(
                            timeout=self._request_deadline_remaining()
                        ):
                            return self._write_saturated(
                                stage="retrieval_singleflight_wait"
                            )
                    if method == "POST" and path in server.HEAVY_PATHS:
                        heavy_acquired = server.heavy_admission.acquire(
                            timeout=self._request_deadline_remaining()
                        )
                        if not heavy_acquired:
                            return self._write_saturated(
                                stage="heavy_request_admission"
                            )
                    try:
                        request_amos = server.request_service_pool.get(
                            timeout=self._request_deadline_remaining()
                        )
                    except queue.Empty:
                        return self._write_saturated(
                            stage="request_service_admission"
                        )
                    with server.service_lock:
                        closing = server.closing
                    if closing:
                        self._write_json(
                            {
                                "status": "error",
                                "error": "server is shutting down",
                                "retryable": True,
                            },
                            status=HTTPStatus.SERVICE_UNAVAILABLE,
                        )
                        return
                    self._dispatch(server, method, body, amos=request_amos)
                finally:
                    if request_amos is not None:
                        server.request_service_pool.put(request_amos)
                    if heavy_acquired:
                        server.heavy_admission.release()
                    if flight_leader and flight_key and flight_event is not None:
                        server.retrieval_singleflight.finish(
                            flight_key, flight_event
                        )
            except RequestDeadlineExceeded as exc:
                self._write_json(
                    {
                        "status": "error",
                        "error": str(exc),
                        "code": "request_deadline_exhausted",
                        "stage": exc.stage,
                        "request_id": exc.request_id,
                        "retryable": True,
                    },
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            except StaleFrameError as exc:
                self._write_json(
                    {
                        "status": "error",
                        "error": str(exc),
                        "code": "stale_revision",
                        "error_code": "stale_frame",
                        "expected_revision": exc.expected_revision,
                        "current_revision": exc.current_revision,
                        "retryable": False,
                    },
                    status=HTTPStatus.CONFLICT,
                )
            except CASConflict as exc:
                self._write_json(
                    {
                        "status": "error",
                        "error": str(exc),
                        "code": "compare_and_swap_conflict",
                        "retryable": False,
                    },
                    status=HTTPStatus.CONFLICT,
                )
            except AccessDenied as exc:
                self._write_json(
                    {"status": "error", "error": str(exc)},
                    status=HTTPStatus.FORBIDDEN,
                )
            except CognitiveWorkspaceBudgetExceeded as exc:
                self._write_json(
                    {
                        "status": "error",
                        "error": str(exc),
                        "code": "cognitive_workspace_budget_exceeded",
                        "retryable": False,
                        "budget": exc.budget,
                        "minimum_budget": exc.minimum_budget,
                        "exceeded_dimensions": exc.exceeded_dimensions,
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
            except AmosError as exc:
                self._write_json(
                    {"status": "error", "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except json.JSONDecodeError as exc:
                self._write_json(
                    {"status": "error", "error": f"invalid json: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except KeyError as exc:
                self._write_json(
                    {"status": "error", "error": f"missing field: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except sqlite3.OperationalError as exc:
                message = str(exc)
                if "database is locked" in message.lower() or "database is busy" in message.lower():
                    self._write_json(
                        {
                            "status": "error",
                            "error": message,
                            "retryable": True,
                        },
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
                self._write_json(
                    {"status": "error", "error": message},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except NotImplementedError:
                self._write_json(
                    {"status": "error", "error": "unknown endpoint"},
                    status=HTTPStatus.NOT_FOUND,
                )

        def _dispatch(
            self,
            server: AmosHTTPServer,
            method: str,
            body: dict[str, Any],
            *,
            amos: Amos | None = None,
        ) -> None:
            amos = amos or server.amos
            path = self.path.split("?", 1)[0]
            if method == "GET":
                if path == "/v1/ready":
                    # Sidecar readiness must remain constant-time as the memory
                    # graph grows. Full graph-quality diagnostics belong to
                    # /v1/health/memory and are intentionally not part of boot.
                    return self._write_json({
                        "status": "ready",
                        "graph_version": amos.store.graph_version(),
                    })
                if path == "/v1/health/memory":
                    payload = amos.health_memory(run_policy=False)
                    payload["background_policy_worker"] = (
                        server.memory_policy_worker.status()
                    )
                    return self._write_json(payload)
                if path == "/v1/health/memory-inventory":
                    payload = amos.health_memory_inventory()
                    payload["background_policy_worker"] = (
                        server.memory_policy_worker.status()
                    )
                    return self._write_json(payload)
                if path == "/v1/health/capacity":
                    payload = amos.health_capacity()
                    payload["request_admission"] = {
                        "service_capacity": server.REQUEST_SERVICE_COUNT,
                        "service_available": server.request_service_pool.qsize(),
                        "heavy": server.heavy_admission.status(),
                        "retrieval_singleflight": (
                            server.retrieval_singleflight.status()
                        ),
                        "wait_policy": "fifo_until_request_deadline",
                        "default_request_deadline_seconds": (
                            server.DEFAULT_REQUEST_DEADLINE_SECONDS
                        ),
                    }
                    return self._write_json(payload)
                if path == "/v1/llm-reviewer/policy":
                    return self._write_json(amos.llm_reviewer_policy())
                if path == "/v1/memory-policy":
                    payload = amos.memory_policy_status()
                    payload["background_policy_worker"] = (
                        server.memory_policy_worker.status()
                    )
                    return self._write_json(payload)
                if path == "/v1/maintenance-processors":
                    return self._write_json(amos.list_maintenance_processors())
                if path == "/v1/verify":
                    return self._write_json(amos.verify_integrity())
                raise NotImplementedError

            if path == "/v1/events:capture":
                return self._write_json(amos.capture_event(**body))
            if path == "/v1/refs:classify":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {None, "amos.reference-classification.v1"}:
                    raise ValidationError(
                        "reference classification profile must be "
                        "'amos.reference-classification.v1'"
                    )
                return self._write_json(amos.classify_refs(**request))
            if path == "/v1/reference-leases:sync":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {None, "amos.reference-lease-sync.v1"}:
                    raise ValidationError(
                        "reference lease profile must be "
                        "'amos.reference-lease-sync.v1'"
                    )
                return self._write_json(amos.sync_reference_leases(**request))
            if path == "/v1/memory-transactions:commit":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {None, "amos.memory-transaction.v1"}:
                    raise ValidationError(
                        "memory transaction profile must be "
                        "'amos.memory-transaction.v1'"
                    )
                atoms = list(request.get("atoms") or [])
                protected = any(
                    isinstance(atom, Mapping)
                    and (
                        str(atom.get("type") or "")
                        in {*CONSTITUTIONAL_ATOM_TYPES, "adjudication"}
                        or GovernanceService.is_immutable_primary_record(atom)
                    )
                    for atom in atoms
                )
                context = self._authorization_context(
                    server, request, required=protected
                )
                request["authorization_context"] = context
                if protected and context.get("actor"):
                    request["actor"] = str(context["actor"])
                return self._write_json(
                    amos.commit_memory_transaction(**request)
                )
            if path == "/v1/memory-transactions:observe":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {
                    None,
                    "amos.memory-transaction-observation-request.v1",
                }:
                    raise ValidationError(
                        "memory transaction observation request profile must be "
                        "'amos.memory-transaction-observation-request.v1'"
                    )
                return self._write_json(
                    amos.observe_memory_transaction(**request)
                )
            if path == "/v1/cognitive-workspaces:compile":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {None, "amos.cognitive-workspace-request.v1"}:
                    raise ValidationError(
                        "cognitive workspace request profile must be "
                        "'amos.cognitive-workspace-request.v1'"
                    )
                policy_schedule = None
                if bool(request.get("run_policy", False)):
                    policy_schedule = server.memory_policy_worker.request_tick(
                        trigger="compile_cognitive_workspace",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                workspace = amos.compile_cognitive_workspace(**request)
                if policy_schedule is not None:
                    workspace["policy_schedule"] = policy_schedule
                return self._write_json(workspace)
            if path == "/v1/interaction-projections:compile":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {
                    None,
                    "amos.interaction-projection-request.v2",
                }:
                    raise ValidationError(
                        "interaction projection request profile must be "
                        "'amos.interaction-projection-request.v2'"
                    )
                return self._write_json(
                    amos.compile_interaction_projection(**request)
                )
            if path == "/v1/memory-heads:get":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {None, "amos.memory-head-request.v1"}:
                    raise ValidationError(
                        "memory head request profile must be "
                        "'amos.memory-head-request.v1'"
                    )
                return self._write_json(amos.get_memory_head(**request))
            if path == "/v1/memory-series:versions:get":
                request = dict(body)
                profile = request.pop("profile", None)
                if profile not in {
                    None,
                    "amos.memory-series-version-request.v1",
                }:
                    raise ValidationError(
                        "memory series version request profile must be "
                        "'amos.memory-series-version-request.v1'"
                    )
                return self._write_json(
                    amos.get_memory_series_versions(**request)
                )
            if path == "/v1/memory-heads:rebuild":
                return self._write_json(amos.rebuild_memory_heads())
            if path == "/v1/atoms:propose":
                return self._write_json(
                    amos.propose_memory_atoms(
                        body["candidates"],
                        actor=body.get("actor", "http"),
                        scope=body.get("scope"),
                    )
                )
            if path == "/v1/atoms:commit":
                atoms = body.get("atoms")
                candidates = (
                    atoms
                    if isinstance(atoms, list)
                    else [body.get("atom")]
                )
                protected = any(
                    isinstance(atom, Mapping)
                    and (
                        str(atom.get("type") or "")
                        in {*CONSTITUTIONAL_ATOM_TYPES, "adjudication"}
                        or GovernanceService.is_immutable_primary_record(atom)
                    )
                    for atom in candidates
                )
                context = self._authorization_context(
                    server, body, required=protected
                )
                actor = (
                    str(context["actor"])
                    if protected and context.get("actor")
                    else body.get("actor", "http")
                )
                if atoms is not None:
                    return self._write_json(
                        amos.commit_memory_atoms(
                            atoms,
                            actor=actor,
                            authorization_context=context,
                            idempotency_key=body.get("idempotency_key"),
                        )
                    )
                return self._write_json(
                    amos.commit_atom(
                        body["atom"],
                        actor=actor,
                        idempotency_key=body.get("idempotency_key"),
                        authorization_context=context,
                    )
                )
            if path == "/v1/atoms:update":
                current = amos.store.get_atom(str(body.get("atom_id") or ""))
                protected = bool(
                    current
                    and (
                        current.get("type")
                        in {*CONSTITUTIONAL_ATOM_TYPES, "adjudication"}
                        or GovernanceService.is_immutable_primary_record(current)
                        or any(
                            key
                            in {
                                "ratification",
                                "epistemic_standing",
                                "normative_standing",
                                "operational_authority",
                                "constitutional_standing",
                            }
                            for key in dict(body.get("payload_patch") or {})
                        )
                    )
                )
                context = self._authorization_context(
                    server, body, required=protected
                )
                return self._write_json(
                    amos.update_atom(
                        body["atom_id"],
                        payload_patch=body.get("payload_patch"),
                        set_fields=body.get("set_fields"),
                        expected_version=body.get("expected_version"),
                        actor=(
                            str(context["actor"])
                            if protected and context.get("actor")
                            else body.get("actor", "http")
                        ),
                        authorization_context=context,
                        idempotency_key=body.get("idempotency_key"),
                    )
                )
            if path == "/v1/proposals:ratify":
                context = self._authorization_context(server, body, required=True)
                return self._write_json(
                    amos.ratify_proposal(
                        proposal_ref=body["proposal_ref"],
                        adjudication_ref=body["adjudication_ref"],
                        expected_version=body["expected_version"],
                        actor=str(context["actor"]),
                        authorization_context=context,
                        idempotency_key=body.get("idempotency_key"),
                    )
                )
            if path == "/v1/proposals:resolve":
                context = self._authorization_context(server, body, required=True)
                return self._write_json(
                    amos.resolve_proposal(
                        proposal_ref=body["proposal_ref"],
                        adjudication_ref=body["adjudication_ref"],
                        expected_version=body["expected_version"],
                        actor=str(context["actor"]),
                        authorization_context=context,
                        idempotency_key=body.get("idempotency_key"),
                    )
                )
            if path == "/v1/constitutional-records:replace":
                context = self._authorization_context(server, body, required=True)
                return self._write_json(
                    amos.replace_constitutional_record(
                        current_ref=body["current_ref"],
                        successor_ref=body["successor_ref"],
                        adjudication_ref=body["adjudication_ref"],
                        expected_current_version=body[
                            "expected_current_version"
                        ],
                        expected_successor_version=body[
                            "expected_successor_version"
                        ],
                        actor=str(context["actor"]),
                        authorization_context=context,
                        idempotency_key=body.get("idempotency_key"),
                    )
                )
            if path == "/v1/provenance:analyze":
                return self._write_json(amos.analyze_provenance(**body))
            if path == "/v1/ratifications:diachronic-status":
                return self._write_json(
                    amos.diachronic_ratification_status(**body)
                )
            if path == "/v1/atoms:archive":
                current = amos.store.get_atom(str(body.get("atom_id") or ""))
                protected = bool(
                    current
                    and (
                        current.get("type")
                        in {*CONSTITUTIONAL_ATOM_TYPES, "adjudication"}
                        or GovernanceService.is_immutable_primary_record(current)
                    )
                )
                context = self._authorization_context(
                    server, body, required=protected
                )
                return self._write_json(
                    amos.archive_atom(
                        body["atom_id"],
                        reason=body.get("reason", "archived"),
                        expected_version=body.get("expected_version"),
                        actor=(
                            str(context["actor"])
                            if protected and context.get("actor")
                            else body.get("actor", "http")
                        ),
                        authorization_context=context,
                    )
                )
            if path == "/v1/atoms:merge":
                return self._write_json(amos.merge_atoms(**body))
            if path == "/v1/atoms:get":
                request = dict(body)
                atom_id = request.pop("atom_id")
                policy_schedule = None
                if bool(request.get("run_policy", True)):
                    policy_schedule = server.memory_policy_worker.request_tick(
                        trigger="retrieve_atom",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                packet = amos.retrieve_atom(atom_id, **request)
                if policy_schedule is not None:
                    packet["policy_schedule"] = policy_schedule
                return self._write_json(packet)
            if path == "/v1/evidence:get":
                request = dict(body)
                evidence_id = request.pop("evidence_id")
                policy_schedule = None
                if bool(request.get("run_policy", True)):
                    policy_schedule = server.memory_policy_worker.request_tick(
                        trigger="retrieve_evidence",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                packet = amos.retrieve_evidence(evidence_id, **request)
                if policy_schedule is not None:
                    packet["policy_schedule"] = policy_schedule
                return self._write_json(packet)
            if path == "/v1/packets:retrieve":
                request = dict(body)
                policy_schedule = None
                if bool(request.get("run_policy", True)):
                    policy_schedule = server.memory_policy_worker.request_tick(
                        trigger="retrieve_packet",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                packet = amos.retrieve_packet(**request)
                if policy_schedule is not None:
                    packet["policy_schedule"] = policy_schedule
                return self._write_json(packet)
            if path == "/v1/reasoning-frames:compile":
                request = self._reasoning_request(body, page=False)
                if bool(request.get("run_policy", True)):
                    server.memory_policy_worker.request_tick(
                        trigger="compile_memory_frame",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                frame = amos.compile_memory_frame(**request)
                return self._write_json(frame)
            if path == "/v1/reasoning-pages:load":
                request = self._reasoning_request(body, page=True)
                if bool(request.get("run_policy", True)):
                    server.memory_policy_worker.request_tick(
                        trigger="load_memory_page",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                page = amos.load_memory_page(**request)
                return self._write_json(page)
            if path == "/v1/retrieval-outcomes":
                return self._write_json(amos.record_retrieval_outcome(**body))
            if path == "/v1/maintenance:request":
                return self._write_json(amos.request_maintenance(**body))
            if path == "/v1/deletion-requests":
                return self._write_json(amos.request_deletion(**body))
            if path == "/v1/runtime-state":
                return self._write_json(amos.record_runtime_state(**body))
            if path == "/v1/self-assessments":
                return self._write_json(amos.record_self_assessment(**body))
            if path == "/v1/self-awareness:retrieve":
                return self._write_json(amos.retrieve_self_awareness(**body))
            if path == "/v1/agentic-recall:retrieve":
                return self._write_json(amos.retrieve_agentic_recall(**body))
            if path == "/v1/shared-views:retrieve":
                request = dict(body)
                policy_schedule = None
                if bool(request.get("run_policy", True)):
                    policy_schedule = server.memory_policy_worker.request_tick(
                        trigger="retrieve_shared_view",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                view = amos.retrieve_shared_view(**request)
                if policy_schedule is not None:
                    view["policy_schedule"] = policy_schedule
                return self._write_json(view)
            if path == "/v1/shared-views:refresh":
                request = dict(body)
                policy_schedule = None
                if bool(request.get("run_policy", True)):
                    policy_schedule = server.memory_policy_worker.request_tick(
                        trigger="refresh_shared_view",
                        scope=request.get("scope") or {},
                    )
                    request["run_policy"] = False
                view = amos.refresh_shared_view(**request)
                if policy_schedule is not None:
                    view["policy_schedule"] = policy_schedule
                return self._write_json(view)
            if path == "/v1/procedures:execution-policy":
                return self._write_json(amos.evaluate_procedure_execution(**body))
            if path == "/v1/capacity:configure":
                return self._write_json(amos.configure_capacity_budget(**body))
            if path == "/v1/smp:analyze":
                return self._write_json(amos.run_smp_analysis(**body))
            if path == "/v1/memory-policy:configure":
                return self._write_json(amos.configure_memory_policy(**body))
            if path == "/v1/memory-policy:run":
                return self._write_json(amos.run_memory_policy(**body))
            if path == "/v1/maintenance-distiller:run":
                raise RuntimeError(
                    "maintenance distiller must use the isolated execution lane"
                )
            raise NotImplementedError

        def _reasoning_request(
            self, body: dict[str, Any], *, page: bool
        ) -> dict[str, Any]:
            common = {
                "need",
                "purpose",
                "depth",
                "scope",
                "requester",
                "target_processor",
                "memory_mode",
                "token_or_byte_budget",
                "run_policy",
            }
            allowed = common | (
                {"frame_id", "revision", "page"} if page else {"task_context"}
            )
            required = (
                {"frame_id", "revision", "page"} if page else {"need", "purpose"}
            )
            unknown = sorted(set(body) - allowed)
            missing = sorted(required - set(body))
            if unknown:
                raise ValidationError(
                    "unknown reasoning request field(s): " + ", ".join(unknown)
                )
            if missing:
                raise ValidationError(
                    "missing reasoning request field(s): " + ", ".join(missing)
                )
            request = dict(body)
            object_fields = {"scope"}
            if page:
                object_fields.update({"revision", "page"})
            else:
                object_fields.add("task_context")
            for field in sorted(object_fields):
                if (
                    field in request
                    and request[field] is not None
                    and not isinstance(request[field], dict)
                ):
                    raise ValidationError(f"{field} must be an object")
            text_fields = {"requester", "target_processor", "depth", "memory_mode"}
            if page:
                text_fields.add("frame_id")
            else:
                text_fields.update({"need", "purpose"})
            for field in sorted(text_fields):
                if field in request and (
                    not isinstance(request[field], str)
                    or not request[field].strip()
                ):
                    raise ValidationError(f"{field} must be a non-empty string")
            if page:
                for field in ("need", "purpose"):
                    if field in request and request[field] is not None and (
                        not isinstance(request[field], str)
                        or not request[field].strip()
                    ):
                        raise ValidationError(
                            f"{field} must be null or a non-empty string"
                        )
            if "run_policy" in request and not isinstance(request["run_policy"], bool):
                raise ValidationError("run_policy must be a boolean")
            budget = request.get("token_or_byte_budget")
            if budget is not None and (
                isinstance(budget, bool) or not isinstance(budget, (int, dict))
            ):
                raise ValidationError(
                    "token_or_byte_budget must be an integer or object"
                )
            return request

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise json.JSONDecodeError("expected JSON object", raw.decode("utf-8"), 0)
            return data

        def _request_deadline_epoch(self) -> float:
            raw = str(self.headers.get("X-Request-Deadline-Epoch-Ms") or "")
            if not raw:
                return time.time() + cast(
                    AmosHTTPServer, self.server
                ).DEFAULT_REQUEST_DEADLINE_SECONDS
            try:
                return int(raw) / 1000.0
            except ValueError:
                return 0.0

        def _request_deadline_remaining(self) -> float:
            remaining = remaining_seconds()
            if remaining is not None:
                return remaining
            return max(0.0, self._request_deadline_epoch() - time.time())

        def _write_saturated(
            self, *, stage: str = "request_capacity_admission"
        ) -> None:
            self._write_json(
                {
                    "status": "error",
                    "error": "AMOS request capacity was unavailable before deadline",
                    "code": "request_capacity_exhausted",
                    "stage": stage,
                    "retry_after_ms": 250,
                    "retryable": True,
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        def _authorization_context(
            self,
            server: AmosHTTPServer,
            body: Mapping[str, Any],
            *,
            required: bool,
        ) -> dict[str, Any]:
            supplied = dict(body.get("authorization_context") or {})
            forbidden = sorted(
                {"identity_ref", "actor", "capabilities"}.intersection(supplied)
            )
            if forbidden:
                raise AccessDenied(
                    "identity, actor, and capabilities are transport-authenticated; "
                    "caller JSON cannot supply: " + ", ".join(forbidden)
                )
            header = str(self.headers.get("Authorization") or "")
            principal: dict[str, Any] | None = None
            if header.startswith("Bearer "):
                candidate = header[7:]
                for token, configured in server.governance_principals.items():
                    if hmac.compare_digest(candidate, token):
                        principal = dict(configured)
                        break
            if required and principal is None:
                raise AccessDenied(
                    "cognitive governance requires an authenticated service principal"
                )
            return {**supplied, **(principal or {})}

        def _write_json(
            self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            request_id = self.headers.get("X-Request-ID")
            if (
                request_id
                and len(request_id) <= 256
                and "\r" not in request_id
                and "\n" not in request_id
            ):
                self.send_header("X-Request-ID", request_id)
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                return

    return AmosHandler


def serve(
    host: str,
    port: int,
    db_path: str,
    *,
    maintenance_processor_paths: list[str] | None = None,
    governance_principals: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    server = AmosHTTPServer(
        (host, port),
        db_path,
        maintenance_processor_paths=maintenance_processor_paths,
        governance_principals=governance_principals,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
