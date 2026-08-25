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
    CognitiveWorkspaceBudgetExceeded,
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
from amos.http_api import AmosHTTPServer, _FairAdmission, _FairServicePool
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


def test_cli_constitutional_governance_commands(tmp_path, capsys):
    from .test_constitutional_governance import (
        RATIFICATION,
        _adjudication,
        _covenant,
        _proposal,
    )

    db_path = tmp_path / "cli_governance.sqlite3"
    amos = Amos(db_path)
    try:
        covenant = _covenant(amos)
        proposal = _proposal(amos)
        adjudication = _adjudication(
            amos,
            proposal_ref=proposal["id"],
            covenant_ref=covenant["id"],
        )
    finally:
        amos.close()

    assert (
        cli_main(
            [
                "--db",
                str(db_path),
                "ratify-proposal",
                "--proposal-ref",
                proposal["id"],
                "--adjudication-ref",
                adjudication["id"],
                "--expected-version",
                str(proposal["version"]),
                    "--actor",
                    "svc:example_agent:self-governance",
                "--authorization-context",
                json.dumps(RATIFICATION),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "--db",
                str(db_path),
                "provenance-analysis",
                "--atom-ref",
                proposal["id"],
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "--db",
                str(db_path),
                "diachronic-status",
                "--subject-ref",
                proposal["id"],
                "--identity-ref",
                "example_agent:self",
                "--required-confirmations",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"status": "ratified"' in output
    assert '"status": "analyzed"' in output
    assert '"threshold_reached": true' in output


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
        lease = http_json(
            f"{base}/v1/reference-leases:sync",
            {
                "profile": "amos.reference-lease-sync.v1",
                "owner_ref": "cogito:pending-work:http-smoke",
                "target_refs": ["http_atom", "http_atom"],
                "scope": {"system": "test"},
                "replace": True,
            },
        )
        assert lease["status"] == "committed"
        assert lease["published_count"] == 1
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
        batch_body = {
            "atoms": [
                {
                    "id": "http_batch_one",
                    "type": "semantic",
                    "payload": {"summary": "HTTP batch one"},
                },
                {
                    "id": "http_batch_two",
                    "type": "semantic",
                    "payload": {"summary": "HTTP batch two"},
                },
            ],
            "idempotency_key": "http-batch",
        }
        batch = http_json(f"{base}/v1/atoms:commit", batch_body)
        replayed_batch = http_json(f"{base}/v1/atoms:commit", batch_body)
        assert replayed_batch == batch
        assert [item["atom"]["id"] for item in batch["committed"]] == [
            "http_batch_one",
            "http_batch_two",
        ]
        assert server.amos.health_memory()["atoms"] == 3
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
        captured = server.amos.capture_event(
            source_type="http-test",
            source_ref="http-evidence-source",
            payload={"summary": "HTTP exact evidence retrieval works"},
        )["evidence"]
        exact_evidence = http_json(
            f"{base}/v1/evidence:get",
            {
                "evidence_id": captured["evidence_id"],
                "requester": "http-test",
                "target_processor": "reasoner",
                "run_policy": False,
            },
        )
        assert exact_evidence["status"] == "found"
        assert exact_evidence["record"]["evidence_id"] == captured["evidence_id"]
        classified = http_json(
            f"{base}/v1/refs:classify",
            {
                "refs": [
                    "http_atom",
                    captured["evidence_id"],
                    "http_missing_ref",
                ],
                "requester": "http-test",
            },
        )
        assert classified["atom_refs"] == ["http_atom"]
        assert classified["evidence_refs"] == [captured["evidence_id"]]
        assert classified["unknown_refs"] == ["http_missing_ref"]
        packet = http_json(
            f"{base}/v1/packets:retrieve",
            {"cues": ["HTTP endpoint"]},
        )
        assert "http_atom" in item_refs(packet)
        ready = http_json(f"{base}/v1/ready")
        assert ready == {
            "status": "ready",
            "graph_version": server.amos.store.graph_version(),
        }
        health = http_json(f"{base}/v1/health/memory")
        assert health["atoms"] == 3
        inventory = http_json(f"{base}/v1/health/memory-inventory")
        assert inventory["profile"] == "amos.memory-inventory-health.v1"
        assert inventory["atoms"] == 3
        assert "background_policy_worker" in inventory
        verify = http_json(f"{base}/v1/verify")
        assert verify["journal"]["status"] == "ok"
        assert verify["replay"]["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_maintenance_does_not_block_health_reads(tmp_path):
    db_path = str(tmp_path / "http_maintenance_lane.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    entered = threading.Event()
    release = threading.Event()

    def slow_maintenance(**_kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return {"status": "completed"}

    server.maintenance_amos.run_maintenance_distiller = slow_maintenance
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    maintenance_result: list[dict] = []
    maintenance_thread = threading.Thread(
        target=lambda: maintenance_result.append(
            http_json(f"{base}/v1/maintenance-distiller:run", {})
        ),
        daemon=True,
    )
    try:
        maintenance_thread.start()
        assert entered.wait(timeout=1)
        started = time.monotonic()
        health = http_json(f"{base}/v1/health/memory")
        assert time.monotonic() - started < 1
        assert health["atoms"] == 0
        release.set()
        maintenance_thread.join(timeout=2)
        assert maintenance_result == [{"status": "completed"}]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_full_health_does_not_block_inventory_health(tmp_path):
    db_path = str(tmp_path / "http_inventory_health_lane.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    entered = threading.Event()
    release = threading.Event()

    def slow_health(*, run_policy=True):
        entered.set()
        assert release.wait(timeout=2)
        return {"atoms": 0}

    server.health_amos.health_memory = slow_health
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    full_health_thread = threading.Thread(
        target=lambda: http_json(f"{base}/v1/health/memory"),
        daemon=True,
    )
    try:
        full_health_thread.start()
        assert entered.wait(timeout=1)
        started = time.monotonic()
        inventory = http_json(f"{base}/v1/health/memory-inventory")
        assert time.monotonic() - started < 1
        assert inventory["atoms"] == 0
        release.set()
        full_health_thread.join(timeout=2)
        assert full_health_thread.is_alive() is False
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_full_health_does_not_block_capacity_health(tmp_path):
    db_path = str(tmp_path / "http_capacity_health_lane.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    entered = threading.Event()
    release = threading.Event()

    def slow_health(*, run_policy=True):
        entered.set()
        assert release.wait(timeout=2)
        return {"atoms": 0}

    server.health_amos.health_memory = slow_health
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    full_health_thread = threading.Thread(
        target=lambda: http_json(f"{base}/v1/health/memory"),
        daemon=True,
    )
    try:
        full_health_thread.start()
        assert entered.wait(timeout=1)
        started = time.monotonic()
        capacity = http_json(f"{base}/v1/health/capacity")
        assert time.monotonic() - started < 1
        assert capacity["graph_version"] == 0
        release.set()
        full_health_thread.join(timeout=2)
        assert full_health_thread.is_alive() is False
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_inventory_stale_while_refresh_is_nonblocking_and_marked(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "http_inventory_cache.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    refresh_entered = threading.Event()
    release_refresh = threading.Event()
    original = Amos.health_memory_inventory

    def slow_inventory(service):
        refresh_entered.set()
        assert release_refresh.wait(timeout=2)
        return original(service)

    try:
        initial = http_json(f"{base}/v1/health/memory-inventory")
        assert initial["diagnostic_cache"]["consistency"] == "exact"
        with server.inventory_cache_lock:
            server.inventory_cache_at_monotonic = (
                time.monotonic() - server.INVENTORY_CACHE_TTL_SECONDS - 1
            )
        monkeypatch.setattr(Amos, "health_memory_inventory", slow_inventory)
        started = time.monotonic()
        cached = http_json(
            f"{base}/v1/health/memory-inventory",
            headers={
                "X-AMOS-Diagnostic-Consistency": "stale-while-refresh",
            },
        )
        assert time.monotonic() - started < 1
        assert cached["diagnostic_cache"]["consistency"] == "cached"
        assert cached["diagnostic_cache"]["freshness"] == "stale"
        assert cached["diagnostic_cache"]["refreshing"] is True
        assert refresh_entered.wait(timeout=1)
    finally:
        release_refresh.set()
        refresh = server.inventory_refresh_thread
        if refresh is not None:
            refresh.join(timeout=2)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_heavy_retrievals_reserve_two_general_request_lanes(tmp_path):
    db_path = str(tmp_path / "http_heavy_reservation.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    server.amos.commit_atom({
        "id": "reserved_exact_atom",
        "type": "semantic",
        "payload": {"summary": "Exact reads retain a request lane."},
    })
    entered = threading.Condition()
    active = 0
    release = threading.Event()

    def slow_retrieval(**kwargs):
        nonlocal active
        with entered:
            active += 1
            entered.notify_all()
        assert release.wait(timeout=5)
        return {
            "packet_id": "packet:" + str((kwargs.get("cues") or ["none"])[0]),
            "items": [],
            "omissions": [],
        }

    for service in server.request_services:
        service.retrieve_packet = slow_retrieval
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    outcomes: list[dict] = []
    retrieval_threads = [
        threading.Thread(
            target=lambda cue=cue: outcomes.append(http_json(
                f"{base}/v1/packets:retrieve",
                {"cues": [cue], "run_policy": False},
            )),
            daemon=True,
        )
        for cue in ("heavy-one", "heavy-two")
    ]
    try:
        for retrieval_thread in retrieval_threads:
            retrieval_thread.start()
        with entered:
            assert entered.wait_for(lambda: active == 2, timeout=2)
        started = time.monotonic()
        exact = http_json(
            f"{base}/v1/atoms:get",
            {
                "atom_id": "reserved_exact_atom",
                "run_policy": False,
            },
        )
        assert time.monotonic() - started < 1
        assert exact["status"] == "found"
        assert server.heavy_admission.status()["active"] == 2
    finally:
        release.set()
        for retrieval_thread in retrieval_threads:
            retrieval_thread.join(timeout=2)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(outcomes) == 2


def test_http_heavy_retrieval_waits_fifo_until_its_request_deadline(tmp_path):
    db_path = str(tmp_path / "http_heavy_deadline_fifo.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    entered = threading.Condition()
    active = 0
    release = threading.Event()

    def slow_retrieval(**kwargs):
        nonlocal active
        with entered:
            active += 1
            entered.notify_all()
        assert release.wait(timeout=2)
        return {
            "packet_id": "packet:" + str((kwargs.get("cues") or ["none"])[0]),
            "items": [],
            "omissions": [],
        }

    for service in server.request_services:
        service.retrieve_packet = slow_retrieval
    # A historical fixed admission cap would fail this waiter immediately.
    # The current contract ignores that process-local value and honors each
    # caller's explicit deadline instead.
    server.ADMISSION_WAIT_SECONDS = 0.01
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    outcomes: list[dict] = []
    errors: list[Exception] = []
    headers = {
        "X-Request-Deadline-Epoch-Ms": str(int((time.time() + 1.5) * 1000)),
    }

    def retrieve(cue: str) -> None:
        try:
            outcomes.append(http_json(
                f"{base}/v1/packets:retrieve",
                {"cues": [cue], "run_policy": False},
                headers=headers,
            ))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    request_threads = [
        threading.Thread(target=retrieve, args=(f"heavy-{index}",), daemon=True)
        for index in range(3)
    ]
    try:
        for request_thread in request_threads:
            request_thread.start()
        with entered:
            assert entered.wait_for(lambda: active == 2, timeout=1)
        deadline = time.monotonic() + 1
        while (
            server.heavy_admission.status()["waiting"] != 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert server.heavy_admission.status()["waiting"] == 1
        time.sleep(0.05)
        assert errors == []
        release.set()
        for request_thread in request_threads:
            request_thread.join(timeout=2)
            assert request_thread.is_alive() is False
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert errors == []
    assert len(outcomes) == 3


def test_heavy_admission_prioritizes_interactive_without_reordering_its_class():
    admission = _FairAdmission(1, interactive_burst_limit=2)
    assert admission.acquire(timeout=0.1, request_class="standard")
    order: list[str] = []

    def wait_for_slot(label: str, request_class: str) -> None:
        assert admission.acquire(
            timeout=2,
            request_class=request_class,
        )
        order.append(label)
        admission.release()

    standard = threading.Thread(
        target=wait_for_slot,
        args=("standard", "standard"),
        daemon=True,
    )
    interactive = threading.Thread(
        target=wait_for_slot,
        args=("interactive", "interactive"),
        daemon=True,
    )
    standard.start()
    deadline = time.monotonic() + 1
    while (
        admission.status()["waiting_standard"] != 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert admission.status()["waiting_standard"] == 1
    interactive.start()
    deadline = time.monotonic() + 1
    while (
        admission.status()["waiting_interactive"] != 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert admission.status()["waiting_interactive"] == 1

    admission.release()
    interactive.join(timeout=2)
    standard.join(timeout=2)

    assert interactive.is_alive() is False
    assert standard.is_alive() is False
    assert order == ["interactive", "standard"]
    assert admission.status()["policy"] == (
        "interactive_priority_bounded_burst"
    )


def test_request_service_pool_prioritizes_interactive_waiters() -> None:
    service = object()
    pool = _FairServicePool([service])
    holder = pool.acquire(timeout=0.1, request_class="standard")
    assert holder is service
    order: list[str] = []

    def wait_for_service(label: str, request_class: str) -> None:
        acquired = pool.acquire(timeout=2, request_class=request_class)
        assert acquired is service
        order.append(label)
        pool.release(acquired, request_class=request_class)

    standard = threading.Thread(
        target=wait_for_service,
        args=("standard", "standard"),
        daemon=True,
    )
    interactive = threading.Thread(
        target=wait_for_service,
        args=("interactive", "interactive"),
        daemon=True,
    )
    standard.start()
    deadline = time.monotonic() + 1
    while (
        pool.status()["waiting_standard"] != 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert pool.status()["waiting_standard"] == 1
    interactive.start()
    deadline = time.monotonic() + 1
    while (
        pool.status()["waiting_interactive"] != 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert pool.status()["waiting_interactive"] == 1

    pool.release(holder, request_class="standard")
    interactive.join(timeout=2)
    standard.join(timeout=2)

    assert interactive.is_alive() is False
    assert standard.is_alive() is False
    assert order == ["interactive", "standard"]
    assert pool.status() == {
        "policy": "interactive_priority_bounded_burst",
        "capacity": 1,
        "available": 1,
        "active": 0,
        "active_interactive": 0,
        "active_standard": 0,
        "waiting": 0,
        "waiting_interactive": 0,
        "waiting_standard": 0,
        "interactive_burst_limit": 4,
        "consecutive_interactive": 0,
    }


def test_request_service_pool_bounds_interactive_burst() -> None:
    service = object()
    pool = _FairServicePool([service], interactive_burst_limit=1)
    holder = pool.acquire(timeout=0.1, request_class="standard")
    assert holder is service
    order: list[str] = []

    def wait_for_service(label: str, request_class: str) -> None:
        acquired = pool.acquire(timeout=2, request_class=request_class)
        assert acquired is service
        order.append(label)
        pool.release(acquired, request_class=request_class)

    waiting = [
        threading.Thread(
            target=wait_for_service,
            args=("standard", "standard"),
            daemon=True,
        ),
        threading.Thread(
            target=wait_for_service,
            args=("interactive-1", "interactive"),
            daemon=True,
        ),
        threading.Thread(
            target=wait_for_service,
            args=("interactive-2", "interactive"),
            daemon=True,
        ),
    ]
    waiting[0].start()
    deadline = time.monotonic() + 1
    while (
        pool.status()["waiting_standard"] != 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert pool.status()["waiting_standard"] == 1
    waiting[1].start()
    deadline = time.monotonic() + 1
    while (
        pool.status()["waiting_interactive"] != 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert pool.status()["waiting_interactive"] == 1
    waiting[2].start()
    deadline = time.monotonic() + 1
    while (
        pool.status()["waiting_interactive"] != 2
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert pool.status()["waiting"] == 3

    pool.release(holder, request_class="standard")
    for thread in waiting:
        thread.join(timeout=2)

    assert all(thread.is_alive() is False for thread in waiting)
    assert order == ["interactive-1", "standard", "interactive-2"]


def test_http_shared_view_queues_policy_instead_of_running_it_inline(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "http_shared_view_policy_lane.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    scheduled: list[dict] = []

    def queue_tick(**request):
        scheduled.append(dict(request))
        return {"status": "queued", "trigger": request["trigger"]}

    monkeypatch.setattr(server.memory_policy_worker, "request_tick", queue_tick)

    def fail_inline_policy(**_kwargs):
        raise AssertionError("shared-view HTTP read ran maintenance inline")

    monkeypatch.setattr(server.amos.policy, "run_memory_policy", fail_inline_policy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        view = http_json(
            f"{base}/v1/shared-views:retrieve",
            {"processor_ids": ["planner"], "scope": {"tenant": "test"}},
        )
        assert view["view"] == "shared_memory"
        assert view["policy_schedule"] == {
            "status": "queued",
            "trigger": "retrieve_shared_view",
        }
        assert scheduled == [
            {
                "trigger": "retrieve_shared_view",
                "scope": {"tenant": "test"},
            }
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_long_reasoning_request_does_not_block_readiness(tmp_path):
    db_path = str(tmp_path / "http_reasoning_isolation.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    entered = threading.Event()
    release = threading.Event()

    def slow_reasoning(**_kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return {
            "status": "compiled",
            "frame_id": "frame:slow",
            "revision": {"graph_version": 0, "journal_head": "genesis"},
            "page_index": [],
        }

    server.amos.compile_memory_frame = slow_reasoning
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    reasoning_result: list[dict] = []
    reasoning_thread = threading.Thread(
        target=lambda: reasoning_result.append(http_json(
            f"{base}/v1/reasoning-frames:compile",
            {
                "need": "bounded isolation test",
                "purpose": "verify readiness isolation",
                "run_policy": False,
            },
        )),
        daemon=True,
    )
    try:
        reasoning_thread.start()
        assert entered.wait(timeout=1)
        started = time.monotonic()
        ready = http_json(f"{base}/v1/ready")
        assert time.monotonic() - started < 1
        assert ready["status"] == "ready"
        release.set()
        reasoning_thread.join(timeout=3)
        assert reasoning_result[0]["frame_id"] == "frame:slow"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_expired_request_deadline_fails_before_dispatch(tmp_path):
    db_path = str(tmp_path / "http_expired_deadline.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    request = urllib.request.Request(
        f"{base}/v1/atoms:get",
        data=json.dumps({"atom_id": "never-dispatched"}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Request-Deadline-Epoch-Ms": str(int((time.time() - 1) * 1000)),
        },
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=2)
        assert excinfo.value.code == 503
        payload = json.loads(excinfo.value.read().decode("utf-8"))
        assert payload["code"] == "request_deadline_exhausted"
        assert payload["retryable"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_maintenance_admission_respects_request_deadline(tmp_path):
    db_path = str(tmp_path / "http_maintenance_deadline.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    assert server.maintenance_lock.acquire(blocking=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    request = urllib.request.Request(
        f"{base}/v1/memory-policy:run",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "X-Request-Deadline-Epoch-Ms": str(int((time.time() + 1.0) * 1000)),
        },
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=3)
        assert excinfo.value.code == 503
        payload = json.loads(excinfo.value.read().decode("utf-8"))
        assert payload["code"] == "maintenance_capacity_exhausted"
        assert payload["retryable"] is True
    finally:
        server.maintenance_lock.release()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_constitutional_governance_endpoints(tmp_path):
    from .test_constitutional_governance import (
        RATIFICATION,
        _adjudication,
        _covenant,
        _proposal,
    )

    db_path = str(tmp_path / "http_governance.sqlite3")
    try:
        server = AmosHTTPServer(
            ("127.0.0.1", 0),
            db_path,
            governance_principals={"test-governance-token": RATIFICATION},
        )
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    covenant = _covenant(server.amos)
    proposal = _proposal(server.amos)
    adjudication = _adjudication(
        server.amos,
        proposal_ref=proposal["id"],
        covenant_ref=covenant["id"],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as unauthenticated:
            http_json(
                f"{base}/v1/proposals:ratify",
                {
                    "proposal_ref": proposal["id"],
                    "adjudication_ref": adjudication["id"],
                    "expected_version": proposal["version"],
                },
            )
        assert unauthenticated.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as forged:
            http_json(
                f"{base}/v1/atoms:commit",
                {
                    "atom": {
                        **adjudication,
                        "id": "http_forged_adjudication",
                    },
                    "actor": "svc:example_agent:self-governance",
                    "authorization_context": RATIFICATION,
                },
            )
        assert forged.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as forged_transaction:
            http_json(
                f"{base}/v1/memory-transactions:commit",
                {
                    "profile": "amos.memory-transaction.v1",
                    "atoms": [{
                        **adjudication,
                        "id": "http_transaction_forged_adjudication",
                    }],
                    "actor": "svc:example_agent:self-governance",
                    "scope": dict(adjudication["scope"]),
                    "authorization_context": RATIFICATION,
                    "idempotency_key": "http-forged-governance-transaction",
                },
            )
        assert forged_transaction.value.code == 403
        ratified = http_json(
            f"{base}/v1/proposals:ratify",
            {
                "proposal_ref": proposal["id"],
                "adjudication_ref": adjudication["id"],
                "expected_version": proposal["version"],
            },
            headers={"Authorization": "Bearer test-governance-token"},
        )
        assert ratified["status"] == "ratified"
        provenance = http_json(
            f"{base}/v1/provenance:analyze",
            {"atom_ref": proposal["id"]},
        )
        assert provenance["status"] == "analyzed"
        diachronic = http_json(
            f"{base}/v1/ratifications:diachronic-status",
            {
                "subject_ref": proposal["id"],
                "identity_ref": "example_agent:self",
                "required_confirmations": 1,
            },
        )
        assert diachronic["threshold_reached"] is True
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


def test_http_workspace_budget_error_is_structured_and_nonretryable(tmp_path):
    db_path = str(tmp_path / "http_workspace_budget.sqlite3")
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), db_path)
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def overflow(**_request):
        raise CognitiveWorkspaceBudgetExceeded(
            limit_bytes=48_000,
            limit_tokens=12_000,
            limit_items=768,
            used_bytes=52_004,
            estimated_tokens=13_001,
            used_items=312,
        )

    server.amos.compile_cognitive_workspace = overflow
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            http_json(
                f"{base}/v1/cognitive-workspaces:compile",
                {
                    "current_event_ref": "event-current",
                    "conversation_id": "main",
                    "token_or_byte_budget": {
                        "tokens": 12_000,
                        "items": 768,
                    },
                },
            )
        assert excinfo.value.code == 400
        payload = json.loads(excinfo.value.read().decode("utf-8"))
        assert payload["code"] == "cognitive_workspace_budget_exceeded"
        assert payload["retryable"] is False
        assert payload["exceeded_dimensions"] == ["bytes"]
        assert payload["minimum_budget"] == {
            "bytes": 52_004,
            "tokens": 13_001,
            "items": 312,
        }
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


def http_json(url, payload=None, *, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))
