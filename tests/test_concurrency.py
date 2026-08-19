from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from amos import Amos


def _wait_for_waiting_writers(service: Amos, expected: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if service.store.writer_status()["waiting"] == expected:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"expected {expected} waiting writers, got "
        f"{service.store.writer_status()}"
    )


def test_exact_retrieval_is_pinned_to_one_revision_during_concurrent_write(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "amos.sqlite3"
    reader = Amos(db_path)
    writer = Amos(db_path)
    try:
        original = reader.commit_atom(
            {
                "id": "snapshot_atom",
                "type": "belief",
                "payload": {"claim": "before"},
            }
        )["atom"]
        read_reached_atom = threading.Event()
        writer_finished = threading.Event()
        allow_read = threading.Event()
        original_get_atom = reader.store.get_atom

        def paused_get_atom(atom_id: str):
            if atom_id == original["id"]:
                read_reached_atom.set()
                assert allow_read.wait(timeout=5)
            return original_get_atom(atom_id)

        monkeypatch.setattr(reader.store, "get_atom", paused_get_atom)
        outcome: dict[str, object] = {}

        def retrieve() -> None:
            outcome["packet"] = reader.retrieve_atom(
                original["id"], run_policy=False
            )

        read_thread = threading.Thread(target=retrieve)
        read_thread.start()
        assert read_reached_atom.wait(timeout=5)
        updated = writer.update_atom(
            original["id"],
            payload_patch={"claim": "after"},
            expected_version=original["version"],
        )["atom"]
        writer_finished.set()
        allow_read.set()
        read_thread.join(timeout=5)

        assert writer_finished.is_set()
        assert not read_thread.is_alive()
        packet = outcome["packet"]
        assert isinstance(packet, dict)
        assert packet["graph_version"] == 1
        assert packet["item"]["payload"]["claim"] == "before"
        assert packet["item"]["version"] == original["version"]
        assert updated["version"] == original["version"] + 1
    finally:
        reader.close()
        writer.close()


def test_read_snapshots_overlap_across_connections(tmp_path, monkeypatch):
    db_path = tmp_path / "amos.sqlite3"
    first = Amos(db_path)
    second = Amos(db_path)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    try:
        first.commit_atom(
            {
                "id": "parallel_read_atom",
                "type": "belief",
                "payload": {"claim": "parallel snapshots do not share a read lock"},
            }
        )
        original_first_get = first.store.get_atom
        original_second_get = second.store.get_atom

        def paused_first_get(atom_id: str):
            first_entered.set()
            assert release.wait(timeout=5)
            return original_first_get(atom_id)

        def paused_second_get(atom_id: str):
            second_entered.set()
            assert release.wait(timeout=5)
            return original_second_get(atom_id)

        monkeypatch.setattr(first.store, "get_atom", paused_first_get)
        monkeypatch.setattr(second.store, "get_atom", paused_second_get)
        threads = [
            threading.Thread(
                target=lambda: first.retrieve_atom(
                    "parallel_read_atom", run_policy=False
                )
            ),
            threading.Thread(
                target=lambda: second.retrieve_atom(
                    "parallel_read_atom", run_policy=False
                )
            ),
        ]
        threads[0].start()
        assert first_entered.wait(timeout=5)
        threads[1].start()
        assert second_entered.wait(timeout=5)
        release.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
    finally:
        release.set()
        first.close()
        second.close()


def test_writer_admission_is_fifo_across_connections(tmp_path):
    db_path = tmp_path / "amos.sqlite3"
    holder = Amos(db_path)
    maintenance = Amos(db_path)
    foreground = Amos(db_path)
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []
    try:
        def hold_writer() -> None:
            with holder.store.transaction():
                entered.set()
                assert release.wait(timeout=5)

        def write(service: Amos, lane: str, name: str) -> None:
            with service.store.transaction(lane=lane):
                order.append(name)

        holder_thread = threading.Thread(target=hold_writer)
        holder_thread.start()
        assert entered.wait(timeout=5)

        maintenance_thread = threading.Thread(
            target=write, args=(maintenance, "maintenance", "maintenance")
        )
        foreground_thread = threading.Thread(
            target=write, args=(foreground, "foreground", "foreground")
        )
        maintenance_thread.start()
        _wait_for_waiting_writers(maintenance, 1)
        foreground_thread.start()
        _wait_for_waiting_writers(foreground, 2)
        release.set()

        holder_thread.join(timeout=5)
        maintenance_thread.join(timeout=5)
        foreground_thread.join(timeout=5)
        assert order == ["maintenance", "foreground"]
    finally:
        release.set()
        holder.close()
        maintenance.close()
        foreground.close()


def test_memory_policy_execution_lock_is_shared_by_database(tmp_path):
    db_path = tmp_path / "amos.sqlite3"
    first = Amos(db_path)
    second = Amos(db_path)
    try:
        assert first.policy._memory_policy_lock.acquire(blocking=False)
        try:
            result = second.run_memory_policy(
                force=True, trigger="concurrent_cross_connection_tick"
            )
        finally:
            first.policy._memory_policy_lock.release()

        assert result["status"] == "skipped"
        assert result["reason"] == "memory_policy_already_running"
    finally:
        first.close()
        second.close()


def test_policy_due_state_is_rechecked_after_shared_admission(tmp_path):
    db_path = tmp_path / "amos.sqlite3"
    first = Amos(db_path)
    second = Amos(db_path)
    try:
        first.configure_memory_policy(
            schedule={"every_graph_versions": 1, "every_seconds": 3600},
            maintenance={"enabled": False},
            distillation={"enabled": False},
            maintenance_distiller={"enabled": False},
            decay={"enabled": False},
            storage_cleanup={"enabled": False},
        )
        first.commit_atom(
            {
                "id": "policy_due_interleaving_seed",
                "type": "belief",
                "payload": {"claim": "make the optimistic due check true"},
            }
        )
        shared_lock = second.policy._memory_policy_lock

        class InterleavingLock:
            def acquire(self, blocking: bool = True) -> bool:
                completed = first.run_memory_policy(
                    force=True, trigger="winner_before_shared_admission"
                )
                assert completed["status"] == "completed"
                return shared_lock.acquire(blocking=blocking)

            def release(self) -> None:
                shared_lock.release()

        second.policy._memory_policy_lock = InterleavingLock()
        result = second.run_memory_policy(trigger="stale_due_check")

        assert result["status"] == "skipped"
        assert result["reason"] == "not_due"
        assert [
            event
            for event in first.store.list_events()
            if event["event_type"] == "memory_policy_run"
        ][0]["payload"]["trigger"] == "winner_before_shared_admission"
        assert sum(
            event["event_type"] == "memory_policy_run"
            for event in first.store.list_events()
        ) == 1
    finally:
        first.close()
        second.close()


def test_storage_cleanup_yields_to_queued_foreground_writer_between_batches(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "amos.sqlite3"
    maintenance = Amos(db_path)
    foreground = Amos(db_path)
    first_batch_entered = threading.Event()
    release_first_batch = threading.Event()
    paused_once = False
    outcome: dict[str, object] = {}
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace(
        "+00:00", "Z"
    )
    try:
        for index in range(3):
            maintenance.commit_atom(
                {
                    "id": f"cleanup_batch_{index}",
                    "type": "semantic",
                    "payload": {"summary": f"cleanup batch {index}"},
                    "created_at": old,
                    "observed_at": old,
                    "updated_at": old,
                    "lifecycle_state": "archived",
                    "health_status": "stale",
                }
            )
        maintenance.configure_memory_policy(
            maintenance={"enabled": False},
            distillation={"enabled": False},
            maintenance_distiller={"enabled": False},
            decay={"enabled": False},
            storage_cleanup={
                "enabled": True,
                "idle_after_seconds": 0,
                "min_interval_seconds": 0,
                "max_deletions_per_tick": 3,
                "write_batch_size": 1,
                "delete_archived_after_seconds": 0,
                "delete_stale_after_seconds": 0,
                "compact_idempotency_after_seconds": None,
                "sqlite_compaction": {
                    "checkpoint_wal": False,
                    "vacuum_enabled": False,
                },
            },
        )
        original_replace_atom = maintenance.store.replace_atom

        def pause_first_deleted_atom(conn, atom):
            nonlocal paused_once
            if atom.get("deleted") and not paused_once:
                paused_once = True
                first_batch_entered.set()
                assert release_first_batch.wait(timeout=5)
            return original_replace_atom(conn, atom)

        monkeypatch.setattr(
            maintenance.store, "replace_atom", pause_first_deleted_atom
        )

        def run_cleanup() -> None:
            outcome["cleanup"] = maintenance.run_memory_policy(
                force=True, trigger="batched_cleanup_concurrency"
            )

        cleanup_thread = threading.Thread(target=run_cleanup)
        cleanup_thread.start()
        assert first_batch_entered.wait(timeout=5)

        writer_thread = threading.Thread(
            target=lambda: foreground.commit_atom(
                {
                    "id": "foreground_between_batches",
                    "type": "belief",
                    "payload": {"claim": "foreground writer was admitted"},
                }
            )
        )
        writer_thread.start()
        _wait_for_waiting_writers(foreground, 1)
        release_first_batch.set()

        writer_thread.join(timeout=5)
        cleanup_thread.join(timeout=10)
        assert not writer_thread.is_alive()
        assert not cleanup_thread.is_alive()

        events = maintenance.store.list_events()
        cleanup_versions = sorted(
            event["graph_version"]
            for event in events
            if event["event_type"] == "storage_cleanup_run"
        )
        foreground_write = next(
            event["graph_version"]
            for event in events
            if "foreground_between_batches" in event["target_refs"]
        )
        assert cleanup_versions[0] < foreground_write < cleanup_versions[1]
        cleanup = outcome["cleanup"]
        assert isinstance(cleanup, dict)
        assert cleanup["results"]["storage_cleanup"]["write_batch_count"] == 3
        assert cleanup["event"]["payload"]["completed_graph_version"] == (
            cleanup["event"]["graph_version"]
        )
    finally:
        release_first_batch.set()
        maintenance.close()
        foreground.close()


def test_index_cpu_build_does_not_hold_writer_and_stale_plan_is_not_published_fresh(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "amos.sqlite3"
    maintenance = Amos(db_path)
    foreground = Amos(db_path)
    build_started = threading.Event()
    release_build = threading.Event()
    outcome: dict[str, object] = {}
    try:
        for index in range(3):
            maintenance.commit_atom(
                {
                    "id": f"index_seed_{index}",
                    "type": "semantic",
                    "payload": {"summary": f"index semantic seed {index}"},
                }
            )
        original_build = maintenance.indexes._build_lsa_token_vectors

        def paused_build(**kwargs):
            build_started.set()
            assert release_build.wait(timeout=5)
            return original_build(**kwargs)

        monkeypatch.setattr(
            maintenance.indexes, "_build_lsa_token_vectors", paused_build
        )

        def rebuild() -> None:
            outcome["index"] = maintenance.indexes.rebuild()

        rebuild_thread = threading.Thread(target=rebuild)
        rebuild_thread.start()
        assert build_started.wait(timeout=5)

        writer_thread = threading.Thread(
            target=lambda: foreground.commit_atom(
                {
                    "id": "writer_during_index_cpu",
                    "type": "belief",
                    "payload": {"claim": "writer completed during LSA build"},
                }
            )
        )
        writer_thread.start()
        writer_thread.join(timeout=5)
        assert not writer_thread.is_alive()
        release_build.set()
        rebuild_thread.join(timeout=5)
        assert not rebuild_thread.is_alive()

        result = outcome["index"]
        assert isinstance(result, dict)
        assert result["status"] == "stale"
        assert {
            item["freshness"] for item in result["indexes"]
        } == {"stale"}
        assert foreground.store.get_atom("writer_during_index_cpu") is not None
    finally:
        release_build.set()
        maintenance.close()
        foreground.close()


def test_steward_cpu_planning_does_not_hold_writer_and_fails_closed_when_stale(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "amos.sqlite3"
    maintenance = Amos(db_path)
    foreground = Amos(db_path)
    planning_started = threading.Event()
    release_planning = threading.Event()
    outcome: dict[str, object] = {}
    try:
        for index in range(3):
            maintenance.commit_atom(
                {
                    "id": f"steward_plan_seed_{index}",
                    "type": "semantic",
                    "payload": {"summary": f"steward plan seed {index}"},
                }
            )
        original_cluster = maintenance.smp.cluster

        def paused_cluster(atoms):
            planning_started.set()
            assert release_planning.wait(timeout=5)
            return original_cluster(atoms)

        monkeypatch.setattr(maintenance.smp, "cluster", paused_cluster)

        def run_steward() -> None:
            outcome["result"] = maintenance.run_steward()

        steward_thread = threading.Thread(target=run_steward)
        steward_thread.start()
        assert planning_started.wait(timeout=5)
        foreground.commit_atom(
            {
                "id": "foreground_during_steward_cpu",
                "type": "belief",
                "payload": {"claim": "writer is not blocked by steward CPU"},
            }
        )
        release_planning.set()
        steward_thread.join(timeout=5)
        assert not steward_thread.is_alive()

        result = outcome["result"]
        assert isinstance(result, dict)
        assert result["status"] == "stale"
        assert result["reason"] == (
            "canonical_revision_advanced_during_steward_planning"
        )
        assert foreground.store.get_atom("foreground_during_steward_cpu") is not None
        assert not any(
            event["event_type"] == "steward_run"
            for event in maintenance.store.list_events()
        )
        assert maintenance.verify_replay()["status"] == "ok"
    finally:
        release_planning.set()
        maintenance.close()
        foreground.close()


def test_steward_graph_planning_does_not_block_exact_read_receipt(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "amos.sqlite3"
    maintenance = Amos(db_path)
    reader = Amos(db_path)
    planning_started = threading.Event()
    release_planning = threading.Event()
    steward_outcome: dict[str, object] = {}
    read_outcome: dict[str, object] = {}
    try:
        maintenance.commit_atom(
            {
                "id": "steward_exact_read_seed",
                "type": "belief",
                "payload": {"claim": "exact reads remain responsive"},
            }
        )
        original_intrinsic = maintenance.stewardship._intrinsic_edges_for_atom

        def paused_intrinsic(atom):
            planning_started.set()
            assert release_planning.wait(timeout=5)
            return original_intrinsic(atom)

        monkeypatch.setattr(
            maintenance.stewardship,
            "_intrinsic_edges_for_atom",
            paused_intrinsic,
        )

        steward_thread = threading.Thread(
            target=lambda: steward_outcome.setdefault(
                "result", maintenance.run_steward()
            )
        )
        steward_thread.start()
        assert planning_started.wait(timeout=5)

        read_thread = threading.Thread(
            target=lambda: read_outcome.setdefault(
                "packet",
                reader.retrieve_atom(
                    "steward_exact_read_seed",
                    run_policy=False,
                ),
            )
        )
        read_thread.start()
        read_thread.join(timeout=2)

        assert not read_thread.is_alive()
        assert read_outcome["packet"]["status"] == "found"
        assert steward_thread.is_alive()

        release_planning.set()
        steward_thread.join(timeout=5)
        assert not steward_thread.is_alive()
        assert steward_outcome["result"]["status"] == "completed"
        assert maintenance.verify_replay()["status"] == "ok"
    finally:
        release_planning.set()
        maintenance.close()
        reader.close()


def test_decay_uses_replayable_bounded_write_batches(tmp_path):
    amos = Amos(tmp_path / "amos.sqlite3")
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )
    try:
        for index in range(3):
            amos.commit_atom(
                {
                    "id": f"decay_batch_{index}",
                    "type": "semantic",
                    "payload": {"summary": f"decay batch {index}"},
                    "decay_policy": {"expires_at": expired},
                }
            )
        amos.configure_memory_policy(
            maintenance={"enabled": False},
            distillation={"enabled": False},
            maintenance_distiller={"enabled": False},
            decay={"enabled": True, "write_batch_size": 1},
            storage_cleanup={"enabled": False},
        )

        result = amos.run_memory_policy(
            force=True, trigger="decay_batch_verification"
        )
        decay = result["results"]["decay"]

        assert decay["action_count"] == 3
        assert decay["write_batch_count"] == 3
        assert len(decay["events"]) == 3
        assert all(
            amos.store.get_atom(f"decay_batch_{index}")["lifecycle_state"]
            == "archived"
            for index in range(3)
        )
        assert amos.verify_replay()["status"] == "ok"
    finally:
        amos.close()


def test_decay_fails_closed_when_canonical_revision_advances_during_preparation(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "amos.sqlite3"
    maintenance = Amos(db_path)
    foreground = Amos(db_path)
    preparation_started = threading.Event()
    release_preparation = threading.Event()
    outcome: dict[str, object] = {}
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )
    try:
        maintenance.commit_atom(
            {
                "id": "decay_stale_plan_target",
                "type": "semantic",
                "payload": {"summary": "stale decay plans must not publish"},
                "decay_policy": {"expires_at": expired},
            }
        )
        maintenance.configure_memory_policy(
            maintenance={"enabled": False},
            distillation={"enabled": False},
            maintenance_distiller={"enabled": False},
            decay={"enabled": True, "write_batch_size": 1},
            storage_cleanup={"enabled": False},
        )
        original_attach = maintenance.policy._attach_search_index

        def paused_attach(atom):
            if atom.get("id") == "decay_stale_plan_target":
                preparation_started.set()
                assert release_preparation.wait(timeout=5)
            return original_attach(atom)

        monkeypatch.setattr(
            maintenance.policy, "_attach_search_index", paused_attach
        )

        def run_decay() -> None:
            outcome["result"] = maintenance.run_memory_policy(
                force=True, trigger="decay_revision_guard"
            )

        decay_thread = threading.Thread(target=run_decay)
        decay_thread.start()
        assert preparation_started.wait(timeout=5)
        foreground.commit_atom(
            {
                "id": "foreground_during_decay_preparation",
                "type": "belief",
                "payload": {"claim": "advance the canonical revision"},
            }
        )
        release_preparation.set()
        decay_thread.join(timeout=5)
        assert not decay_thread.is_alive()

        result = outcome["result"]
        assert isinstance(result, dict)
        decay = result["results"]["decay"]
        assert decay["action_count"] == 0
        assert decay["skipped_stale_plans"] == 1
        assert decay["stale_revision"]["reason"] == (
            "canonical_revision_advanced_before_decay_publish"
        )
        assert (
            maintenance.store.get_atom("decay_stale_plan_target")[
                "lifecycle_state"
            ]
            == "active"
        )
        assert maintenance.verify_replay()["status"] == "ok"
    finally:
        release_preparation.set()
        maintenance.close()
        foreground.close()


def test_large_retrieval_bounds_hot_scan_but_keeps_direct_lexical_candidates(
    tmp_path,
):
    amos = Amos(tmp_path / "amos.sqlite3")
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace(
        "+00:00", "Z"
    )
    try:
        atoms = [
            {
                "id": "bounded_scan_lexical_target",
                "type": "belief",
                "payload": {"claim": "uniquewordoutsidehotscan"},
                "created_at": old,
                "observed_at": old,
                "updated_at": old,
            }
        ]
        atoms.extend(
            {
                "id": f"bounded_scan_filler_{index}",
                "type": "belief",
                "payload": {"claim": f"recent filler memory {index}"},
            }
            for index in range(512)
        )
        amos.commit_memory_atoms(atoms)

        packet = amos.retrieve_packet(
            cues=["uniquewordoutsidehotscan"],
            max_items=1,
            run_policy=False,
        )
        generation = packet["degradation"]["candidate_generation"]

        assert packet["items"][0]["atom_ref"] == "bounded_scan_lexical_target"
        assert generation["filtered_total_count"] == 513
        assert generation["scanned_count"] == 512
        assert generation["scan_truncated"] is True
        assert "candidate_scan_truncated" in packet["degradation"]["reason_codes"]
    finally:
        amos.close()


def test_canonical_mutation_retires_packet_cache_in_bounded_batches(tmp_path):
    amos = Amos(tmp_path / "amos.sqlite3")
    try:
        graph_version = amos.store.graph_version()
        with amos.store.transaction() as conn:
            for index in range(200):
                request = {"cues": [f"cached request {index}"]}
                amos.store.cache_packet(
                    conn,
                    packet_id=f"cached_packet_{index}",
                    request=request,
                    response={"packet_id": f"cached_packet_{index}", "items": []},
                    graph_version=graph_version,
                )

        amos.commit_atom(
            {
                "id": "cache_retirement_revision",
                "type": "belief",
                "payload": {"claim": "advance canonical revision"},
            }
        )

        assert len(amos.store.list_packet_cache()) == 72
        assert (
            amos.store.get_cached_packet(
                request={"cues": ["cached request 199"]},
                graph_version=amos.store.graph_version(),
            )
            is None
        )
    finally:
        amos.close()
