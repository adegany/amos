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
