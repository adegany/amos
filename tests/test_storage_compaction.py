from __future__ import annotations

import sqlite3

from amos import Amos


def test_journal_snapshot_compaction_bounds_retained_event_payloads(tmp_path):
    path = tmp_path / "amos.sqlite3"
    amos = Amos(path)
    try:
        first_event_id = None
        for index in range(12):
            committed = amos.commit_atom(
                {
                    "id": f"journal_compaction_{index}",
                    "type": "semantic",
                    "payload": {
                        "summary": f"journal compaction item {index}",
                        "blob": "x" * 1024,
                    },
                }
            )
            first_event_id = first_event_id or committed["event"]["event_id"]

        results = [
            amos.store.compact_journal_segment(
                max_events=4,
                min_events=1,
                retain_tail_events=2,
                retain_snapshots=1,
                retain_full_segments=1,
            )
            for _ in range(3)
        ]

        assert [result["status"] for result in results] == [
            "completed",
            "completed",
            "completed",
        ]
        status = amos.store.journal_storage_status()
        assert status["live_event_count"] == 2
        assert status["compacted_event_count"] == 10
        assert status["retained_segment_count"] == 1
        assert status["digest_only_segment_count"] == 2
        assert status["retained_segment_event_count"] == 2
        assert status["digest_only_event_count"] == 8
        assert status["snapshot_count"] == 1
        assert amos.store.event_count() == 12
        assert len(amos.store.list_events()) == 4
        compact_receipt = amos.store.get_event(str(first_event_id))
        assert compact_receipt["storage_compacted"] is True
        assert compact_receipt["compact_receipt_verified"] is True

        pruned = amos.store.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM amos_journal_segments
            WHERE payload_retained = 0
              AND length(events_blob) = 0
              AND compressed_bytes = 0
            """
        ).fetchone()
        assert int(pruned["count"]) == 2
        retained_event_id = amos.store.conn.execute(
            """
            SELECT first_event_id
            FROM amos_journal_segments
            WHERE payload_retained = 1
            """
        ).fetchone()["first_event_id"]
        assert amos.store.get_event(str(retained_event_id)) is not None

        chain = amos.verify_journal_chain()
        assert chain["status"] == "ok"
        assert chain["verification_scope"] == (
            "retained_payloads_plus_compacted_boundaries"
        )
        assert chain["digest_only_event_count"] == 8
        assert chain["compact_receipt_count"] == 8
        assert amos.verify_replay()["status"] == "ok"
        assert amos.store.sqlite_space_status()["auto_vacuum_mode"] == (
            "incremental"
        )
    finally:
        amos.close()

    reopened = Amos(path)
    try:
        assert reopened.verify_journal_chain()["status"] == "ok"
        assert reopened.verify_replay()["status"] == "ok"
        assert reopened.store.event_count() == 12
        with reopened.store.transaction() as conn:
            conn.execute(
                """
                UPDATE amos_journal_event_receipts
                SET compact_payload = '{"storage_compacted":false}'
                WHERE event_id = ?
                """,
                (str(first_event_id),),
            )
        corrupted = reopened.verify_journal_chain()
        assert corrupted["status"] == "failed"
        assert any(
            failure["reason"] == "segment_compact_receipt_invalid"
            for failure in corrupted["failures"]
        )
    finally:
        reopened.close()


def test_incremental_vacuum_is_available_for_new_databases(amos):
    status = amos.store.incremental_vacuum(max_pages=1)

    assert status["status"] == "completed"
    assert amos.store.sqlite_space_status()["auto_vacuum_mode"] == "incremental"


def test_existing_database_adopts_incremental_auto_vacuum_after_full_vacuum(
    tmp_path,
):
    path = tmp_path / "existing.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE legacy_record(value TEXT)")
    connection.commit()
    connection.close()

    amos = Amos(path)
    try:
        space = amos.store.sqlite_space_status()
        assert space["auto_vacuum_mode"] == "none"
        assert space["requires_full_vacuum_for_incremental"] is True
        skipped = amos.store.incremental_vacuum(max_pages=1)
        assert skipped["reason"] == "incremental_auto_vacuum_not_enabled"

        rebuilt = amos.store.vacuum()

        assert rebuilt["status"] == "completed"
        assert amos.store.sqlite_space_status()["auto_vacuum_mode"] == (
            "incremental"
        )
    finally:
        amos.close()


def test_memory_head_indexes_rebuild_from_compacted_snapshot(amos):
    scope = {"tenant": "snapshot-head-test"}
    goal_ref = "goal:snapshot-rebuild"

    def goal(atom_id: str, revision: int) -> dict:
        return {
            "id": atom_id,
            "type": "goal",
            "payload": {
                "profile": "example.generic-goal-work.v1",
                "goal_ref": goal_ref,
                "revision": revision,
                "objective": "Rebuild head indexes from a compacted snapshot.",
                "status": "active",
            },
        }

    def head(atom_id: str, prior: str | None, version: int) -> dict:
        return {
            "series_kind": "goal_work",
            "series_id": goal_ref,
            "new_head_ref": atom_id,
            "expected_head_ref": prior,
            "expected_head_version": version,
        }

    amos.commit_memory_transaction(
        scope=scope,
        actor="snapshot-head-test",
        atoms=[goal("snapshot_goal_1", 1)],
        head_updates=[head("snapshot_goal_1", None, 0)],
    )
    second = amos.commit_memory_transaction(
        scope=scope,
        actor="snapshot-head-test",
        atoms=[goal("snapshot_goal_2", 2)],
        head_updates=[head("snapshot_goal_2", "snapshot_goal_1", 1)],
    )
    compacted = amos.store.compact_journal_segment(
        max_events=8,
        min_events=1,
        retain_tail_events=0,
        retain_snapshots=1,
        retain_full_segments=0,
    )
    assert compacted["status"] == "completed"
    with amos.store.transaction() as conn:
        conn.execute("DELETE FROM amos_memory_heads")
        conn.execute("DELETE FROM amos_memory_head_history")

    rebuilt = amos.store.rebuild_memory_heads()

    assert rebuilt[0]["head_ref"] == "snapshot_goal_2"
    history = amos.store.list_memory_head_history(
        scope=scope,
        series_kind="goal_work",
        series_id=goal_ref,
    )
    assert [item["head_version"] for item in history] == [2, 1]
    observed = amos.observe_memory_transaction(
        event_id=second["event"]["event_id"],
        scope=scope,
        requester="snapshot-auditor",
        target_processor="test",
    )
    assert observed["status"] == "found"
    assert observed["storage_compacted"] is True
    assert observed["verification_status"] == "mechanically_verified"
    assert observed["projected_heads"][0]["head_ref"] == "snapshot_goal_2"
    assert amos.verify_replay()["status"] == "ok"
