from __future__ import annotations

import json
import threading
import time
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from amos import (
    AccessDenied,
    AgenticRecallAuditor,
    Amos,
    BackgroundMemoryPolicyWorker,
    CASConflict,
    CapacityGovernor,
    DistillerMaintenanceWorker,
    IdempotencyConflict,
    IndexMaintainer,
    JournalProjector,
    MemoryPolicyWorker,
    MemorySteward,
    MaintenanceProposal,
    PacketCacheInvalidator,
    SMPWorker,
    SemanticFacet,
    SemanticMaintenanceProcessor,
    SelfModelCalibrator,
    ValidationError,
    ontology_snapshot,
    semantic_relation_proposals_from_facets,
)
from amos.cli import main as cli_main
from amos.http_api import AmosHTTPServer
from amos.smp import cosine

from .helpers import ExampleTrainingFlightProcessor, item_refs


def test_cli_smoke_init_commit_retrieve(tmp_path, capsys):
    db_path = tmp_path / "cli.sqlite3"
    assert cli_main(["--db", str(db_path), "init"]) == 0
    cli_main(
        [
            "--db",
            str(db_path),
            "commit-atom",
            "--type",
            "belief",
            "--payload",
            json.dumps({"claim": "CLI recall works"}),
        ]
    )
    assert cli_main(["--db", str(db_path), "retrieve", "--cue", "CLI recall"]) == 0
    out = capsys.readouterr().out
    assert "CLI recall works" in out


def test_http_v1_endpoints_smoke(tmp_path):
    db_path = str(tmp_path / "http.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        committed = http_json(
            f"{base}/v1/atoms:commit",
            {
                "atom": {
                    "id": "http_atom",
                    "type": "belief",
                    "payload": {"claim": "HTTP endpoint works"},
                }
            },
        )
        assert committed["status"] == "committed"
        updated = http_json(
            f"{base}/v1/atoms:update",
            {
                "atom_id": "http_atom",
                "payload_patch": {
                    "semantic_facets": [
                        {
                            "subject": "http endpoint",
                            "intent": "exercise update",
                            "outcome_direction": "positive",
                        }
                    ]
                },
                "actor": "system",
                "expected_version": committed["atom"]["version"],
            },
        )
        assert updated["status"] == "updated"
        assert updated["atom"]["payload"]["semantic_facets"][0]["subject"] == "http endpoint"
        assert server.amos.health_memory()["atoms"] == 1
        exact = http_json(
            f"{base}/v1/atoms:get",
            {
                "atom_id": "http_atom",
                "requester": "http-test",
                "target_processor": "reasoner",
                "run_policy": False,
            },
        )
        assert exact["status"] == "found"
        assert exact["retrieval_mode"] == "exact"
        assert exact["item"]["atom_ref"] == "http_atom"
        packet = http_json(
            f"{base}/v1/packets:retrieve",
            {"cues": ["HTTP endpoint"]},
        )
        assert "http_atom" in item_refs(packet)
        health = http_json(f"{base}/v1/health/memory")
        assert health["atoms"] == 1
        verify = http_json(f"{base}/v1/verify")
        assert verify["journal"]["status"] == "ok"
        assert verify["replay"]["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_sqlite_lock_returns_retryable_json(tmp_path):
    db_path = str(tmp_path / "http_locked.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def locked_commit(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    server.amos.commit_atom = locked_commit
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            http_json(
                f"{base}/v1/atoms:commit",
                {
                    "atom": {
                        "id": "http_locked_atom",
                        "type": "belief",
                        "payload": {"claim": "lock handling works"},
                    }
                },
            )
        assert excinfo.value.code == 503
        payload = json.loads(excinfo.value.read().decode("utf-8"))
        assert payload["status"] == "error"
        assert payload["retryable"] is True
        assert "database is locked" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_reasoning_frame_page_contract_and_stale_revision(tmp_path):
    db_path = str(tmp_path / "http_reasoning.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, payload, request_id="reasoning-http-test"):
        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=5)
        with response:
            return (
                json.loads(response.read().decode("utf-8")),
                response.headers,
                response.version,
            )

    try:
        old = http_json(
            f"{base}/v1/atoms:commit",
            {
                "atom": {
                    "id": "http_reasoning_old",
                    "type": "belief",
                    "payload": {"claim": "HTTP reasoning history " + "old " * 300},
                }
            },
        )["atom"]
        new = http_json(
            f"{base}/v1/atoms:commit",
            {
                "atom": {
                    "id": "http_reasoning_new",
                    "type": "belief",
                    "payload": {
                        "claim": "HTTP reasoning history active " + "new " * 300
                    },
                    "supersedes": [old["id"]],
                }
            },
        )["atom"]
        frame, headers, version = post(
            "/v1/reasoning-frames:compile",
            {
                "need": "HTTP reasoning history active",
                "purpose": "exercise coherent frame transport",
                "token_or_byte_budget": {"tokens": 800},
                "run_policy": False,
            },
            request_id="frame-req-1",
        )
        assert version == 11
        assert headers["X-Request-ID"] == "frame-req-1"
        assert frame["status"] == "compiled"
        descriptor = next(
            page
            for page in frame["page_index"]
            if new["id"] in page["focus_atom_refs"]
        )
        page, headers, _version = post(
            "/v1/reasoning-pages:load",
            {
                "frame_id": frame["frame_id"],
                "revision": frame["revision"],
                "page": descriptor,
                "depth": "focused",
                "run_policy": False,
            },
            request_id="page-req-1",
        )
        assert headers["X-Request-ID"] == "page-req-1"
        assert page["status"] == "loaded"

        invalid_requests = [
            ({"purpose": "missing need", "run_policy": False}, "missing"),
            (
                {
                    "need": "valid",
                    "purpose": "valid",
                    "unknown_field": True,
                    "run_policy": False,
                },
                "unknown",
            ),
            (
                {
                    "need": ["wrong type"],
                    "purpose": "valid",
                    "run_policy": False,
                },
                "non-empty string",
            ),
        ]
        for payload, expected in invalid_requests:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                post("/v1/reasoning-frames:compile", payload)
            assert excinfo.value.code == 400
            error_payload = json.loads(excinfo.value.read().decode("utf-8"))
            assert error_payload["status"] == "error"
            assert expected in error_payload["error"]

        http_json(
            f"{base}/v1/atoms:commit",
            {
                "atom": {
                    "id": "http_reasoning_revision_change",
                    "type": "belief",
                    "payload": {"claim": "Revision changed."},
                }
            },
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post(
                "/v1/reasoning-pages:load",
                {
                    "frame_id": frame["frame_id"],
                    "revision": frame["revision"],
                    "page": descriptor,
                    "run_policy": False,
                },
                request_id="stale-req-1",
            )
        assert excinfo.value.code == 409
        assert excinfo.value.headers["X-Request-ID"] == "stale-req-1"
        stale = json.loads(excinfo.value.read().decode("utf-8"))
        assert stale["code"] == "stale_revision"
        assert stale["error_code"] == "stale_frame"
        assert stale["expected_revision"] == frame["revision"]
        assert stale["current_revision"] != frame["revision"]
        assert stale["retryable"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def http_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))
