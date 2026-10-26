"""Presentation reads must progress without competing for the feedback writer."""
import threading
import time

import pytest

from amos.http_api import AmosHTTPServer
from amos.request_context import request_context, record_read_effects
from amos import Amos
from datetime import datetime, timedelta, timezone
from .test_cli_http import http_json


def test_observation_read_progresses_with_all_normal_channels_waiting_for_writer(tmp_path):
    try:
        server = AmosHTTPServer(("127.0.0.1", 0), str(tmp_path / "amos.sqlite3"))
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable: {exc}")
    server.memory_policy_worker.stop(timeout=5)
    server.amos.commit_atom({"id": "visible", "type": "semantic",
                             "payload": {"summary": "current"},
                             "scope": {"namespace": "test"}})
    server.amos.commit_atom({"id": "other-scope", "type": "semantic",
                             "payload": {"summary": "private"},
                             "scope": {"namespace": "other"}})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/canonical-records:batch-get"
    body = {"atom_ids": ["visible", "other-scope"], "heads": [],
            "scope": {"namespace": "test"}, "requester": "system",
            "target_processor": "test"}
    results, errors = [], []

    def normal_read():
        try:
            results.append(http_json(url, body))
        except Exception as exc:
            errors.append(exc)

    readers = [threading.Thread(target=normal_read) for _ in range(4)]
    try:
        with server.maintenance_amos.store.transaction(lane="maintenance"):
            for reader in readers:
                reader.start()
            deadline = time.monotonic() + 5
            while server.amos.store.writer_status()["waiting"] < 4 and time.monotonic() < deadline:
                time.sleep(.01)
            assert server.amos.store.writer_status()["waiting"] == 4
            assert server.request_service_pool.status()["active"] == 4
            start = time.monotonic()
            observation = http_json(url, body, headers={"X-AMOS-Read-Purpose": "observation"})
            assert time.monotonic() - start < 1
            assert observation["observation_only"] is True
            assert observation["read_effects_recorded"] is False
            assert observation["items_by_id"]["visible"]["payload"]["summary"] == "current"
            assert "other-scope" not in observation["items_by_id"]
            assert not results
            assert server.observation_amos.store.conn.execute(
                "SELECT count(*) FROM amos_retrieval_packet_receipts"
            ).fetchone()[0] == 0
        for reader in readers:
            reader.join(timeout=5)
        assert not errors and len(results) == 4
        for result in results:
            assert result["items_by_id"] == observation["items_by_id"]
            assert "observation_only" not in result
            assert server.amos.store.conn.execute(
                "SELECT count(*) FROM amos_retrieval_packet_receipts WHERE packet_id=?",
                (result["packet_id"],),
            ).fetchone()[0] == 1
        # The header cannot turn associative recall or a mutation into an
        # observation, suppress its feedback, or gain admission to the lane.
        recall = http_json(url.replace("canonical-records:batch-get", "packets:retrieve"),
            {"cues": ["current"], "run_policy": False},
            headers={"X-AMOS-Read-Purpose": "observation"})
        assert "observation_only" not in recall
        assert server.amos.store.conn.execute(
            "SELECT count(*) FROM amos_retrieval_packet_receipts WHERE packet_id=?",
            (recall["packet_id"],),
        ).fetchone()[0] == 1
        updated = http_json(url.replace("canonical-records:batch-get", "atoms:update"),
            {"atom_id": "visible", "payload_patch": {"summary": "updated"}, "actor": "owner"},
            headers={"X-AMOS-Read-Purpose": "observation"})
        assert "observation_only" not in updated
        assert updated["atom"]["payload"]["summary"] == "updated"
    finally:
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_observation_context_is_nested_and_restored():
    assert record_read_effects()
    with request_context(deadline_epoch_seconds=None, request_id=None, record_read_effects=False):
        assert not record_read_effects()
        with request_context(deadline_epoch_seconds=None, request_id="executive"):
            assert record_read_effects()
        assert not record_read_effects()
    assert record_read_effects()


def test_online_cleanup_catches_up_despite_uninterrupted_foreground_activity(tmp_path):
    service = Amos(tmp_path / "busy.sqlite3")
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    try:
        for index in range(4):
            service.commit_atom({"id": f"expired-{index}", "type": "semantic",
                "payload": {"summary": "expired archive"}, "lifecycle_state": "archived",
                "created_at": old, "observed_at": old, "updated_at": old})
        service.commit_atom({"id": "retained-policy", "type": "policy",
            "payload": {"rule": "protected"}, "lifecycle_state": "archived",
            "created_at": old, "observed_at": old, "updated_at": old})
        service.configure_memory_policy(
            maintenance={"enabled": False}, distillation={"enabled": False},
            maintenance_distiller={"enabled": False}, decay={"enabled": False},
            storage_cleanup={"enabled": True, "idle_after_seconds": 0,
                "min_interval_seconds": 0, "pressure_catchup_interval_seconds": 0,
                "max_deletions_per_tick": 1, "write_batch_size": 1,
                "protected_types": ["policy"], "delete_archived_after_seconds": 604800,
                "journal_compaction": {"enabled": False},
                "sqlite_compaction": {"checkpoint_wal": False,
                    "incremental_vacuum": False, "vacuum_enabled": False}})
        for index in range(4):
            service.commit_atom({"id": f"fresh-{index}", "type": "semantic",
                "payload": {"summary": "foreground stays busy"}})
            service.retrieve_atom(f"fresh-{index}", run_policy=False)
            result = service.run_memory_policy(trigger="busy-online-worker")
            cleanup = result["results"]["storage_cleanup"]
            assert cleanup["deleted_atom_count"] == 1
            assert not result["due"]["storage_cleanup"]["pressure_triggered"]
        assert all(service.store.get_atom(f"expired-{i}") is None for i in range(4))
        assert service.store.get_atom("retained-policy") is not None
        assert all(service.store.get_atom(f"fresh-{i}") is not None for i in range(4))
    finally:
        service.close()
