"""Minimal stdlib HTTP adapter for the AMOS v1 API surface."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from .errors import AmosError
from .service import Amos


class AmosHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        db_path: str,
        *,
        maintenance_processor_paths: list[str] | None = None,
    ):
        self.db_path = db_path
        self.amos = Amos(
            db_path,
            maintenance_processor_paths=maintenance_processor_paths,
        )
        self.service_lock = threading.RLock()
        super().__init__(server_address, make_handler())

    def server_close(self) -> None:
        try:
            self.amos.close()
        finally:
            super().server_close()


def make_handler() -> type[BaseHTTPRequestHandler]:
    class AmosHandler(BaseHTTPRequestHandler):
        server_version = "AmosHTTP/1.0"

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def log_message(self, _format: str, *args: Any) -> None:
            return

        def _handle(self, method: str) -> None:
            try:
                body = self._read_json() if method == "POST" else {}
                server = cast(AmosHTTPServer, self.server)
                with server.service_lock:
                    self._dispatch(server.amos, method, body)
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
            except NotImplementedError:
                self._write_json(
                    {"status": "error", "error": "unknown endpoint"},
                    status=HTTPStatus.NOT_FOUND,
                )

        def _dispatch(self, amos: Amos, method: str, body: dict[str, Any]) -> None:
            path = self.path.split("?", 1)[0]
            if method == "GET":
                if path == "/v1/health/memory":
                    return self._write_json(amos.health_memory())
                if path == "/v1/health/capacity":
                    return self._write_json(amos.health_capacity())
                if path == "/v1/llm-reviewer/policy":
                    return self._write_json(amos.llm_reviewer_policy())
                if path == "/v1/memory-policy":
                    return self._write_json(amos.memory_policy_status())
                if path == "/v1/maintenance-processors":
                    return self._write_json(amos.list_maintenance_processors())
                if path == "/v1/verify":
                    return self._write_json(
                        {
                            "journal": amos.verify_journal_chain(),
                            "replay": amos.verify_replay(),
                        }
                    )
                raise NotImplementedError

            if path == "/v1/events:capture":
                return self._write_json(amos.capture_event(**body))
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
                if atoms is not None:
                    return self._write_json(
                        amos.commit_memory_atoms(atoms, actor=body.get("actor", "http"))
                    )
                return self._write_json(
                    amos.commit_atom(
                        body["atom"],
                        actor=body.get("actor", "http"),
                        idempotency_key=body.get("idempotency_key"),
                        authorization_context=body.get("authorization_context"),
                    )
                )
            if path == "/v1/atoms:archive":
                return self._write_json(
                    amos.archive_atom(
                        body["atom_id"],
                        reason=body.get("reason", "archived"),
                        expected_version=body.get("expected_version"),
                        actor=body.get("actor", "http"),
                        authorization_context=body.get("authorization_context"),
                    )
                )
            if path == "/v1/atoms:merge":
                return self._write_json(amos.merge_atoms(**body))
            if path == "/v1/packets:retrieve":
                return self._write_json(amos.retrieve_packet(**body))
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
                return self._write_json(amos.retrieve_shared_view(**body))
            if path == "/v1/shared-views:refresh":
                return self._write_json(amos.refresh_shared_view(**body))
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
                return self._write_json(amos.run_maintenance_distiller(**body))
            raise NotImplementedError

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise json.JSONDecodeError("expected JSON object", raw.decode("utf-8"), 0)
            return data

        def _write_json(
            self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
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
) -> None:
    server = AmosHTTPServer(
        (host, port),
        db_path,
        maintenance_processor_paths=maintenance_processor_paths,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
