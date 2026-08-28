"""SQLite persistence for the AMOS v1-local service implementation."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
import zlib
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .journal_replay import (
    empty_replay_state,
    replay_events,
    serializable_replay_state,
)
from .request_context import check_deadline, current_stage, remaining_seconds
from .schemas import SCHEMA_VERSION, canonical_json, digest, utc_now

JSON_COLUMNS = {
    "access_policy",
    "authorization_context",
    "causal_parent_ids",
    "confidence",
    "decay_policy",
    "evidence_refs",
    "feedback_json",
    "expected_versions",
    "index_refs",
    "outcome_json",
    "payload",
    "payload_refs",
    "request_json",
    "response_json",
    "revision_history",
    "scope",
    "supersedes",
    "target_refs",
    "vector_json",
    "details_json",
    "derivation",
}


LEGACY_STRUCTURAL_RELATIONS = {
    "rel:attributed_to",
    "rel:constrained_by",
    "rel:corrected_by",
    "rel:derived_from",
    "rel:has_capability",
    "rel:has_limitation",
    "rel:made_commitment",
    "rel:part_of",
    "rel:produced_outcome",
    "rel:supersedes",
    "rel:uses",
}


class _FairWriteCoordinator:
    """Database-scoped FIFO admission for SQLite write transactions.

    SQLite still provides the authoritative locking and durability semantics.
    This coordinator keeps in-process writers out of busy-spin/backoff races and
    gives maintenance and foreground work the same bounded place in line.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._waiting_by_lane: Counter[str] = Counter()
        self._active_lane: str | None = None

    @contextmanager
    def acquire(self, lane: str) -> Iterator[None]:
        lane = str(lane or "foreground")
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiting_by_lane[lane] += 1
            while ticket != self._serving_ticket:
                self._condition.wait()
            self._waiting_by_lane[lane] -= 1
            if self._waiting_by_lane[lane] <= 0:
                self._waiting_by_lane.pop(lane, None)
            self._active_lane = lane
        try:
            yield
        finally:
            with self._condition:
                self._active_lane = None
                self._serving_ticket += 1
                self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "policy": "fifo",
                "active_lane": self._active_lane,
                "waiting": sum(self._waiting_by_lane.values()),
                "waiting_by_lane": dict(self._waiting_by_lane),
                "next_ticket": self._next_ticket,
                "serving_ticket": self._serving_ticket,
            }


@dataclass
class _SharedSQLiteRuntime:
    writer: _FairWriteCoordinator = field(default_factory=_FairWriteCoordinator)
    schema_lock: threading.Lock = field(default_factory=threading.Lock)
    memory_policy_lock: threading.Lock = field(default_factory=threading.Lock)
    activity_lock: threading.Lock = field(default_factory=threading.Lock)
    last_foreground_activity_at: str = field(default_factory=utc_now)
    references: int = 0


_SQLITE_RUNTIME_REGISTRY_LOCK = threading.Lock()
_SQLITE_RUNTIME_REGISTRY: dict[str, _SharedSQLiteRuntime] = {}


def _sqlite_runtime(path: Path) -> tuple[str, _SharedSQLiteRuntime]:
    if path == Path(":memory:"):
        key = f":memory:{uuid.uuid4()}"
    else:
        key = str(path.expanduser().resolve())
    with _SQLITE_RUNTIME_REGISTRY_LOCK:
        runtime = _SQLITE_RUNTIME_REGISTRY.get(key)
        if runtime is None:
            runtime = _SharedSQLiteRuntime()
            _SQLITE_RUNTIME_REGISTRY[key] = runtime
        runtime.references += 1
    return key, runtime


def _release_sqlite_runtime(key: str, runtime: _SharedSQLiteRuntime) -> None:
    with _SQLITE_RUNTIME_REGISTRY_LOCK:
        runtime.references = max(0, runtime.references - 1)
        if runtime.references == 0 and _SQLITE_RUNTIME_REGISTRY.get(key) is runtime:
            _SQLITE_RUNTIME_REGISTRY.pop(key, None)


def migrated_edge_derivation(relation: str) -> dict[str, Any]:
    """Return the conservative provenance assigned to a legacy edge.

    Migration can classify the relation family, but it cannot reconstruct an
    exact historical producer that was never journaled.
    """

    return {
        "kind": "migrated_relation_classification",
        "relation_class": (
            "structural" if str(relation or "") in LEGACY_STRUCTURAL_RELATIONS
            else "associative"
        ),
        "exact_producer_unknown": True,
    }


class SQLiteStore:
    """Durable AMOS v1-local store.

    SQLite is intentionally used behind the AMOS service boundary for the
    first usable deployment profile. It preserves the canonical journal and
    graph semantics while Postgres remains a migration target behind the same
    API contract.
    """

    backend_name = "sqlite"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        new_database = (
            self.path == Path(":memory:")
            or not self.path.exists()
            or self.path.stat().st_size == 0
        )
        self._runtime_key, self._runtime = _sqlite_runtime(self.path)
        self._connection_lock = threading.RLock()
        self._local = threading.local()
        self._closed = False
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(
                str(self.path), isolation_level=None, check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA busy_timeout = 5000")
            with self._runtime.schema_lock:
                # Incremental auto-vacuum must be selected before the first
                # table is created. Existing databases adopt it the next time
                # an explicitly scheduled full VACUUM rebuilds the file.
                if new_database:
                    self.conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                if self.path != Path(":memory:"):
                    self.conn.execute("PRAGMA journal_mode = WAL")
                    self.conn.execute("PRAGMA synchronous = NORMAL")
                self.init_schema()
        except Exception:
            if hasattr(self, "conn"):
                self.conn.close()
            _release_sqlite_runtime(self._runtime_key, self._runtime)
            raise

    def close(self) -> None:
        with self._connection_lock:
            if self._closed:
                return
            self.conn.close()
            self._closed = True
        _release_sqlite_runtime(self._runtime_key, self._runtime)

    @property
    def memory_policy_lock(self) -> threading.Lock:
        return self._runtime.memory_policy_lock

    @property
    def in_read_snapshot(self) -> bool:
        return int(getattr(self._local, "read_depth", 0) or 0) > 0

    def writer_status(self) -> dict[str, Any]:
        return self._runtime.writer.status()

    def cooperative_maintenance_yield(self, *, max_seconds: float = 0.01) -> bool:
        """Yield between maintenance batches when foreground writers wait."""

        status = self.writer_status()
        waiting = dict(status.get("waiting_by_lane") or {})
        foreground_waiting = sum(
            int(count or 0)
            for lane, count in waiting.items()
            if lane not in {"maintenance", "read_effect"}
        )
        if foreground_waiting <= 0:
            return False
        time.sleep(max(0.0, min(float(max_seconds), 0.05)))
        return True

    def set_write_lane(self, lane: str) -> str:
        previous = str(getattr(self._local, "write_lane", "foreground"))
        self._local.write_lane = str(lane or "foreground")
        return previous

    @contextmanager
    def transaction(
        self, *, lane: str | None = None
    ) -> Iterator[sqlite3.Connection]:
        """Run one canonical write with fair database-scoped admission."""

        if self.in_read_snapshot:
            raise RuntimeError(
                "a read snapshot cannot be upgraded to a write transaction"
            )
        if int(getattr(self._local, "write_depth", 0) or 0) > 0:
            self._local.write_depth += 1
            try:
                yield self.conn
            finally:
                self._local.write_depth -= 1
            return
        selected_lane = str(
            lane or getattr(self._local, "write_lane", "foreground")
        )
        with self._connection_lock:
            with self._runtime.writer.acquire(selected_lane):
                self.conn.execute("BEGIN IMMEDIATE")
                self._local.write_depth = 1
                try:
                    yield self.conn
                    self.conn.commit()
                except Exception:
                    if self.conn.in_transaction:
                        self.conn.rollback()
                    raise
                finally:
                    self._local.write_depth = 0

    @contextmanager
    def read_snapshot(self) -> Iterator[sqlite3.Connection]:
        """Hold one revision-pinned snapshot across a complete logical read."""

        if int(getattr(self._local, "write_depth", 0) or 0) > 0:
            yield self.conn
            return
        if self.in_read_snapshot:
            self._local.read_depth += 1
            try:
                yield self.conn
            finally:
                self._local.read_depth -= 1
            return

        deferred_writes: list[tuple[str, dict[str, Any]]] = []
        completed = False
        deadline_interrupted = False
        deadline_reserve_seconds = 0.01

        def interrupt_at_request_deadline() -> int:
            nonlocal deadline_interrupted
            remaining = remaining_seconds()
            if (
                remaining is not None
                and remaining <= deadline_reserve_seconds
            ):
                deadline_interrupted = True
                return 1
            return 0

        with self._connection_lock:
            deadline_guard_installed = remaining_seconds() is not None
            if deadline_guard_installed:
                # SQLite otherwise cannot observe the cooperative checks that
                # surround one long statement. Abort its virtual machine close
                # to the caller deadline so the request slot is released.
                self.conn.set_progress_handler(
                    interrupt_at_request_deadline,
                    1000,
                )
            try:
                self.conn.execute("BEGIN")
            except Exception:
                if deadline_guard_installed:
                    self.conn.set_progress_handler(None, 0)
                raise
            self._local.read_depth = 1
            self._local.deferred_writes = deferred_writes
            try:
                # BEGIN is deferred in SQLite. This first read is what pins the
                # WAL snapshot before any service composes a multi-query view.
                self.memory_revision()
                yield self.conn
                completed = True
            except sqlite3.OperationalError:
                if deadline_interrupted:
                    check_deadline(
                        current_stage() or "sqlite_read_snapshot",
                        reserve_seconds=deadline_reserve_seconds,
                    )
                raise
            finally:
                if deadline_guard_installed:
                    self.conn.set_progress_handler(None, 0)
                self.conn.rollback()
                self._local.read_depth = 0
                self._local.deferred_writes = []
        if completed and deferred_writes:
            self._flush_deferred_writes(deferred_writes)

    def mark_foreground_activity(self, occurred_at: str | None = None) -> None:
        """Coalesce read telemetry in memory instead of turning reads into writes."""

        timestamp = str(occurred_at or utc_now())
        with self._runtime.activity_lock:
            if timestamp > self._runtime.last_foreground_activity_at:
                self._runtime.last_foreground_activity_at = timestamp

    def persist_packet_after_read(
        self,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        graph_version: int,
        feedback_only: bool = False,
    ) -> None:
        effect = (
            "feedback" if feedback_only else "packet",
            {
                "packet_id": str(packet_id),
                "request": json.loads(canonical_json(request)),
                "response": json.loads(canonical_json(response)),
                "graph_version": int(graph_version),
            },
        )
        deferred = getattr(self._local, "deferred_writes", None)
        if self.in_read_snapshot and isinstance(deferred, list):
            deferred.append(effect)
            return
        self._flush_deferred_writes([effect])

    def _flush_deferred_writes(
        self, effects: Sequence[tuple[str, Mapping[str, Any]]]
    ) -> None:
        """Persist all read effects in one short post-snapshot transaction."""

        optional_cache_only = all(kind == "packet" for kind, _ in effects)
        writer = self.writer_status()
        remaining = remaining_seconds()
        if optional_cache_only and (
            writer.get("active_lane") is not None
            or int(writer.get("waiting", 0) or 0) > 0
            or (remaining is not None and remaining <= 0.25)
        ):
            return
        with self.transaction(lane="read_effect") as conn:
            current_graph_version = self.graph_version()
            for effect_kind, payload in effects:
                graph_version = int(payload["graph_version"])
                if effect_kind == "packet" and graph_version == current_graph_version:
                    self.cache_packet(
                        conn,
                        packet_id=str(payload["packet_id"]),
                        request=payload["request"],
                        response=payload["response"],
                        graph_version=graph_version,
                    )
                else:
                    # A concurrent canonical write can advance the graph after
                    # the read snapshot. Keep delayed-feedback membership, but
                    # never reintroduce a stale hot-cache entry after mutation.
                    self.cache_packet_feedback_receipt(
                        conn,
                        packet_id=str(payload["packet_id"]),
                        response=payload["response"],
                        graph_version=graph_version,
                    )

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS amos_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS amos_evidence (
                evidence_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                payload TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                checksum TEXT NOT NULL,
                access_policy TEXT NOT NULL,
                scope TEXT NOT NULL,
                event_id TEXT
            );

            CREATE TABLE IF NOT EXISTS amos_atoms (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                payload TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                scope TEXT NOT NULL,
                confidence TEXT NOT NULL,
                salience REAL NOT NULL,
                utility REAL NOT NULL,
                layer TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                health_status TEXT NOT NULL,
                retention_class TEXT NOT NULL,
                access_policy TEXT NOT NULL,
                decay_policy TEXT NOT NULL,
                created_at TEXT NOT NULL,
                observed_at TEXT,
                updated_at TEXT NOT NULL,
                last_accessed TEXT,
                version INTEGER NOT NULL,
                supersedes TEXT NOT NULL,
                revision_history TEXT NOT NULL,
                index_refs TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_atoms_type ON amos_atoms(type);
            CREATE INDEX IF NOT EXISTS idx_atoms_lifecycle ON amos_atoms(lifecycle_state);
            CREATE INDEX IF NOT EXISTS idx_atoms_health ON amos_atoms(health_status);
            CREATE INDEX IF NOT EXISTS idx_atoms_deleted_updated
                ON amos_atoms(deleted, updated_at);
            CREATE INDEX IF NOT EXISTS idx_atoms_lifecycle_health_type
                ON amos_atoms(lifecycle_state, health_status, type);
            CREATE INDEX IF NOT EXISTS idx_atoms_skill_authority
                ON amos_atoms(
                    json_extract(payload, '$.profile'),
                    json_extract(payload, '$.plugin_digest')
                )
                WHERE deleted = 0;
            CREATE INDEX IF NOT EXISTS idx_atoms_interaction_stream_sequence
                ON amos_atoms(
                    json_extract(payload, '$.conversation_id'),
                    CAST(json_extract(payload, '$.sequence') AS INTEGER)
                )
                WHERE deleted = 0 AND type = 'interaction_event';
            CREATE INDEX IF NOT EXISTS
                idx_atoms_discourse_thread_conversation_updated
                ON amos_atoms(
                    json_extract(payload, '$.conversation_id'),
                    updated_at DESC
                )
                WHERE deleted = 0 AND type = 'discourse_thread';
            CREATE INDEX IF NOT EXISTS idx_atoms_context_compaction_stream
                ON amos_atoms(
                    json_extract(
                        payload,
                        '$.context_compaction.partition.key'
                    ),
                    CAST(json_extract(
                        payload,
                        '$.context_compaction.coverage.through_sequence'
                    ) AS INTEGER) DESC,
                    updated_at DESC
                )
                WHERE deleted = 0
                  AND type = 'semantic'
                  AND lifecycle_state = 'active';

            CREATE TABLE IF NOT EXISTS amos_atom_text_index (
                atom_id TEXT NOT NULL,
                token TEXT NOT NULL,
                PRIMARY KEY(atom_id, token)
            );
            CREATE INDEX IF NOT EXISTS idx_atom_text_index_token
                ON amos_atom_text_index(token);

            CREATE TABLE IF NOT EXISTS amos_edges (
                edge_id TEXT PRIMARY KEY,
                source_ref TEXT NOT NULL,
                target_ref TEXT NOT NULL,
                relation TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                scope TEXT NOT NULL,
                confidence TEXT NOT NULL,
                derivation TEXT NOT NULL DEFAULT '{}',
                lifecycle_state TEXT NOT NULL,
                health_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON amos_edges(source_ref);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON amos_edges(target_ref);

            CREATE TABLE IF NOT EXISTS amos_retired_edges (
                edge_id TEXT PRIMARY KEY,
                source_ref TEXT NOT NULL,
                target_ref TEXT NOT NULL,
                relation TEXT NOT NULL,
                scope TEXT NOT NULL,
                retired_at TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retired_edges_source
                ON amos_retired_edges(source_ref);
            CREATE INDEX IF NOT EXISTS idx_retired_edges_target
                ON amos_retired_edges(target_ref);

            CREATE TABLE IF NOT EXISTS amos_tombstones (
                tombstone_id TEXT PRIMARY KEY,
                target_ref TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                recreation_policy TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tombstones_target ON amos_tombstones(target_ref);
            CREATE INDEX IF NOT EXISTS idx_tombstones_content ON amos_tombstones(content_digest);

            CREATE TABLE IF NOT EXISTS amos_reference_leases (
                owner_ref TEXT NOT NULL,
                target_ref TEXT NOT NULL,
                scope TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(owner_ref, target_ref)
            );
            CREATE INDEX IF NOT EXISTS idx_reference_leases_target
                ON amos_reference_leases(target_ref);

            CREATE TABLE IF NOT EXISTS amos_event_journal (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                actor TEXT NOT NULL,
                target_refs TEXT NOT NULL,
                payload TEXT NOT NULL,
                payload_refs TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                idempotency_key TEXT,
                payload_digest TEXT NOT NULL,
                causal_parent_ids TEXT NOT NULL,
                expected_versions TEXT NOT NULL,
                authorization_context TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                accepted_at TEXT NOT NULL,
                result_status TEXT NOT NULL,
                projection_status TEXT NOT NULL,
                previous_event_hash TEXT NOT NULL,
                checksum TEXT NOT NULL,
                graph_version INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_event_graph_version
                ON amos_event_journal(graph_version);

            CREATE TABLE IF NOT EXISTS amos_journal_segments (
                segment_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                start_graph_version INTEGER NOT NULL,
                end_graph_version INTEGER NOT NULL,
                first_event_id TEXT NOT NULL,
                last_event_id TEXT NOT NULL,
                first_previous_event_hash TEXT NOT NULL,
                last_event_hash TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                codec TEXT NOT NULL,
                events_blob BLOB NOT NULL,
                events_digest TEXT NOT NULL,
                uncompressed_bytes INTEGER NOT NULL,
                compressed_bytes INTEGER NOT NULL,
                payload_retained INTEGER NOT NULL DEFAULT 1,
                payload_pruned_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_journal_segments_range
                ON amos_journal_segments(start_graph_version, end_graph_version);

            CREATE TABLE IF NOT EXISTS amos_journal_segment_events (
                event_id TEXT PRIMARY KEY,
                segment_id TEXT NOT NULL,
                graph_version INTEGER NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_journal_segment_events_segment
                ON amos_journal_segment_events(segment_id, graph_version);

            CREATE TABLE IF NOT EXISTS amos_journal_event_receipts (
                event_id TEXT PRIMARY KEY,
                segment_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                actor TEXT NOT NULL,
                idempotency_key TEXT,
                payload_digest TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                accepted_at TEXT NOT NULL,
                result_status TEXT NOT NULL,
                projection_status TEXT NOT NULL,
                previous_event_hash TEXT NOT NULL,
                checksum TEXT NOT NULL,
                graph_version INTEGER NOT NULL UNIQUE,
                compact_payload TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_journal_event_receipts_segment
                ON amos_journal_event_receipts(segment_id, graph_version);

            CREATE TABLE IF NOT EXISTS amos_journal_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                through_graph_version INTEGER NOT NULL UNIQUE,
                through_event_id TEXT NOT NULL,
                through_event_hash TEXT NOT NULL,
                codec TEXT NOT NULL,
                state_blob BLOB NOT NULL,
                state_digest TEXT NOT NULL,
                uncompressed_bytes INTEGER NOT NULL,
                compressed_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_journal_snapshots_version
                ON amos_journal_snapshots(through_graph_version DESC);

            CREATE TABLE IF NOT EXISTS amos_memory_heads (
                scope_digest TEXT NOT NULL,
                scope TEXT NOT NULL,
                series_kind TEXT NOT NULL,
                series_id TEXT NOT NULL,
                head_ref TEXT NOT NULL,
                head_version INTEGER NOT NULL,
                journal_event_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(scope_digest, series_kind, series_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_heads_ref
                ON amos_memory_heads(head_ref);

            CREATE TABLE IF NOT EXISTS amos_memory_head_history (
                scope_digest TEXT NOT NULL,
                scope TEXT NOT NULL,
                series_kind TEXT NOT NULL,
                series_id TEXT NOT NULL,
                head_version INTEGER NOT NULL,
                head_ref TEXT NOT NULL,
                journal_event_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(
                    scope_digest, series_kind, series_id, head_version
                )
            );
            CREATE INDEX IF NOT EXISTS idx_memory_head_history_ref
                ON amos_memory_head_history(head_ref);
            CREATE INDEX IF NOT EXISTS idx_memory_head_history_series
                ON amos_memory_head_history(
                    scope_digest, series_kind, series_id, head_version
                );

            CREATE TABLE IF NOT EXISTS amos_idempotency (
                actor TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                event_id TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(actor, idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS amos_packet_cache (
                packet_id TEXT PRIMARY KEY,
                request_digest TEXT NOT NULL,
                graph_version INTEGER NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_packet_cache_request_graph
                ON amos_packet_cache(request_digest, graph_version);

            CREATE TABLE IF NOT EXISTS amos_retrieval_packet_receipts (
                packet_id TEXT PRIMARY KEY,
                graph_version INTEGER NOT NULL,
                feedback_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS amos_retrieval_outcomes (
                outcome_id TEXT PRIMARY KEY,
                packet_id TEXT NOT NULL,
                request_json TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS amos_derived_index_metadata (
                index_name TEXT PRIMARY KEY,
                graph_version INTEGER NOT NULL,
                freshness TEXT NOT NULL,
                rebuilt_at TEXT NOT NULL,
                details_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS amos_token_latent_vectors (
                token TEXT PRIMARY KEY,
                graph_version INTEGER NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_token_latent_vectors_graph
                ON amos_token_latent_vectors(graph_version);
            """
        )
        with self.transaction() as conn:
            self._migrate_edge_derivation(conn)
            self._migrate_retired_edge_storage(conn)
            self._migrate_journal_segment_storage(conn)
            if self._get_meta(conn, "graph_version") is None:
                self._set_meta(conn, "graph_version", "0")
            if self._get_meta(conn, "last_event_hash") is None:
                self._set_meta(conn, "last_event_hash", "genesis")
            self._restore_memory_heads_if_needed(conn)
            self._restore_memory_head_history_if_needed(conn)
            self._backfill_atom_text_index(conn)

    def _migrate_retired_edge_storage(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(amos_retired_edges)"
            ).fetchall()
        }
        for name in ("source_ref", "target_ref", "relation", "scope"):
            if name in columns:
                continue
            default = "'{}'" if name == "scope" else "''"
            conn.execute(
                f"ALTER TABLE amos_retired_edges ADD COLUMN {name} "
                f"TEXT NOT NULL DEFAULT {default}"
            )

    def _migrate_journal_segment_storage(
        self, conn: sqlite3.Connection
    ) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(amos_journal_segments)"
            ).fetchall()
        }
        if "payload_retained" not in columns:
            conn.execute(
                "ALTER TABLE amos_journal_segments "
                "ADD COLUMN payload_retained INTEGER NOT NULL DEFAULT 1"
            )
        if "payload_pruned_at" not in columns:
            conn.execute(
                "ALTER TABLE amos_journal_segments "
                "ADD COLUMN payload_pruned_at TEXT"
            )

    def _migrate_edge_derivation(self, conn: sqlite3.Connection) -> None:
        """Add explicit edge provenance and classify legacy rows for migration."""

        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(amos_edges)").fetchall()
        }
        if "derivation" not in columns:
            conn.execute(
                "ALTER TABLE amos_edges ADD COLUMN derivation TEXT NOT NULL DEFAULT '{}'"
            )
        rows = conn.execute(
            "SELECT edge_id, relation, derivation FROM amos_edges"
        ).fetchall()
        for row in rows:
            raw = str(row["derivation"] or "").strip()
            if raw not in {"", "{}", "null"}:
                continue
            derivation = migrated_edge_derivation(str(row["relation"] or ""))
            conn.execute(
                "UPDATE amos_edges SET derivation = ? WHERE edge_id = ?",
                (canonical_json(derivation), str(row["edge_id"])),
            )

    def _get_meta(self, conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM amos_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            INSERT INTO amos_meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def graph_version(self) -> int:
        value = self._get_meta(self.conn, "graph_version")
        return int(value or 0)

    def last_event_hash(self) -> str:
        return self._get_meta(self.conn, "last_event_hash") or "genesis"

    def memory_revision(self) -> dict[str, Any]:
        """Return the current canonical-memory revision in one SQLite read."""

        rows = self.conn.execute(
            """
            SELECT key, value
            FROM amos_meta
            WHERE key IN ('graph_version', 'last_event_hash')
            """
        ).fetchall()
        values = {str(row["key"]): str(row["value"]) for row in rows}
        return {
            "graph_version": int(values.get("graph_version", "0") or 0),
            "journal_head": values.get("last_event_hash", "genesis") or "genesis",
        }

    def _restore_memory_heads_if_needed(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM amos_memory_heads"
        ).fetchone()
        if int(row["count"] or 0) > 0:
            return
        event = conn.execute(
            """
            SELECT 1
            FROM amos_event_journal
            WHERE event_type = 'memory_transaction_committed'
            LIMIT 1
            """
        ).fetchone()
        snapshot = conn.execute(
            "SELECT 1 FROM amos_journal_snapshots LIMIT 1"
        ).fetchone()
        if event is not None or snapshot is not None:
            self._rebuild_memory_heads(conn)

    def _restore_memory_head_history_if_needed(
        self, conn: sqlite3.Connection
    ) -> None:
        """Backfill the version index from the canonical transaction journal."""

        row = conn.execute(
            "SELECT COUNT(*) AS count FROM amos_memory_head_history"
        ).fetchone()
        if int(row["count"] or 0) > 0:
            return
        event = conn.execute(
            """
            SELECT 1
            FROM amos_event_journal
            WHERE event_type = 'memory_transaction_committed'
              AND json_array_length(json_extract(payload, '$.projected_heads')) > 0
            LIMIT 1
            """
        ).fetchone()
        snapshot = conn.execute(
            "SELECT 1 FROM amos_journal_snapshots LIMIT 1"
        ).fetchone()
        if event is not None or snapshot is not None:
            self._rebuild_memory_heads(conn)

    def get_memory_head(
        self,
        *,
        scope: Mapping[str, Any],
        series_kind: str,
        series_id: str,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        connection = conn or self.conn
        row = connection.execute(
            """
            SELECT *
            FROM amos_memory_heads
            WHERE scope_digest = ? AND series_kind = ? AND series_id = ?
            """,
            (digest(dict(scope)), str(series_kind), str(series_id)),
        ).fetchone()
        return None if row is None else self._row_dict(row)

    def list_memory_heads(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        series_kind: str | None = None,
        series_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope_digest = ?")
            params.append(digest(dict(scope)))
        if series_kind is not None:
            clauses.append("series_kind = ?")
            params.append(str(series_kind))
        if series_ids is not None:
            normalized_series_ids = list(dict.fromkeys(
                str(series_id) for series_id in series_ids if str(series_id)
            ))
            if not normalized_series_ids:
                return []
            clauses.append(
                "series_id IN ("
                + ",".join("?" for _ in normalized_series_ids)
                + ")"
            )
            params.extend(normalized_series_ids)
        query = "SELECT * FROM amos_memory_heads"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY series_kind ASC, series_id ASC"
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [self._row_dict(row) for row in rows]

    def list_memory_head_history(
        self,
        *,
        scope: Mapping[str, Any],
        series_kind: str,
        series_id: str,
        versions: Sequence[int] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return exact journal-projected versions for one memory series."""

        bounded_limit = max(1, min(1000, int(limit)))
        params: list[Any] = [
            digest(dict(scope)),
            str(series_kind),
            str(series_id),
        ]
        query = """
            SELECT *
            FROM amos_memory_head_history
            WHERE scope_digest = ? AND series_kind = ? AND series_id = ?
        """
        exact_versions = list(dict.fromkeys(
            int(version) for version in (versions or ())
        ))
        if exact_versions:
            placeholders = ", ".join("?" for _ in exact_versions)
            query += f" AND head_version IN ({placeholders})"
            params.extend(exact_versions)
            query += " ORDER BY head_version ASC"
        else:
            query += " ORDER BY head_version DESC"
        query += " LIMIT ?"
        params.append(bounded_limit)
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [self._row_dict(row) for row in rows]

    def put_memory_head(
        self, conn: sqlite3.Connection, head: Mapping[str, Any]
    ) -> None:
        scope = dict(head.get("scope") or {})
        conn.execute(
            """
            INSERT INTO amos_memory_heads(
                scope_digest, scope, series_kind, series_id, head_ref,
                head_version, journal_event_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_digest, series_kind, series_id) DO UPDATE SET
                scope = excluded.scope,
                head_ref = excluded.head_ref,
                head_version = excluded.head_version,
                journal_event_id = excluded.journal_event_id,
                updated_at = excluded.updated_at
            """,
            (
                digest(scope),
                canonical_json(scope),
                str(head["series_kind"]),
                str(head["series_id"]),
                str(head["head_ref"]),
                int(head["head_version"]),
                str(head["journal_event_id"]),
                str(head["updated_at"]),
            ),
        )
        conn.execute(
            """
            INSERT INTO amos_memory_head_history(
                scope_digest, scope, series_kind, series_id, head_version,
                head_ref, journal_event_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                scope_digest, series_kind, series_id, head_version
            ) DO UPDATE SET
                scope = excluded.scope,
                head_ref = excluded.head_ref,
                journal_event_id = excluded.journal_event_id,
                updated_at = excluded.updated_at
            """,
            (
                digest(scope),
                canonical_json(scope),
                str(head["series_kind"]),
                str(head["series_id"]),
                int(head["head_version"]),
                str(head["head_ref"]),
                str(head["journal_event_id"]),
                str(head["updated_at"]),
            ),
        )

    def _rebuild_memory_heads(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        conn.execute("DELETE FROM amos_memory_heads")
        conn.execute("DELETE FROM amos_memory_head_history")
        snapshot = self.latest_journal_snapshot()
        through_graph_version = int(
            (snapshot or {}).get("through_graph_version", 0) or 0
        )
        if snapshot is not None:
            state = dict(snapshot.get("state") or {})
            history = [
                dict(head)
                for head in dict(state.get("head_history") or {}).values()
                if isinstance(head, Mapping)
            ]
            if not history:
                history = [
                    dict(head)
                    for head in dict(state.get("heads") or {}).values()
                    if isinstance(head, Mapping)
                ]
            history.sort(
                key=lambda head: (
                    str(head.get("series_kind") or ""),
                    str(head.get("series_id") or ""),
                    int(head.get("head_version", 0) or 0),
                )
            )
            for head in history:
                self.put_memory_head(conn, head)
        rows = conn.execute(
            """
            SELECT event_id, accepted_at, payload
            FROM amos_event_journal
            WHERE event_type = 'memory_transaction_committed'
              AND graph_version > ?
            ORDER BY graph_version ASC
            """,
            (through_graph_version,),
        ).fetchall()
        for row in rows:
            payload = self._json(str(row["payload"]))
            for raw in payload.get("projected_heads", []):
                head = dict(raw)
                head["journal_event_id"] = str(row["event_id"])
                head["updated_at"] = str(
                    head.get("updated_at") or row["accepted_at"]
                )
                self.put_memory_head(conn, head)
        return self.list_memory_heads_from_connection(conn)

    def list_memory_heads_from_connection(
        self, conn: sqlite3.Connection
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT *
            FROM amos_memory_heads
            ORDER BY series_kind ASC, series_id ASC
            """
        ).fetchall()
        return [self._row_dict(row) for row in rows]

    def rebuild_memory_heads(self) -> list[dict[str, Any]]:
        with self.transaction() as conn:
            return self._rebuild_memory_heads(conn)

    def get_meta(self, key: str) -> str | None:
        persisted = self._get_meta(self.conn, key)
        if key != "last_foreground_activity_at":
            return persisted
        with self._runtime.activity_lock:
            recent = self._runtime.last_foreground_activity_at
        if persisted is None:
            return recent
        return max(str(persisted), recent)

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            self._set_meta(conn, key, value)

    def sync_reference_leases(
        self,
        *,
        owner_ref: str,
        target_refs: Sequence[str],
        scope: Mapping[str, Any],
        replace: bool,
    ) -> dict[str, Any]:
        """Publish one owner's exact cleanup-protection frontier."""

        refs = list(dict.fromkeys(str(ref) for ref in target_refs if str(ref)))
        now = utc_now()
        encoded_scope = canonical_json(scope)
        with self.transaction(lane="foreground") as conn:
            if replace:
                conn.execute(
                    "DELETE FROM amos_reference_leases WHERE owner_ref=?",
                    (str(owner_ref),),
                )
            conn.executemany(
                """INSERT INTO amos_reference_leases(
                       owner_ref,target_ref,scope,created_at,updated_at
                   ) VALUES(?,?,?,?,?)
                   ON CONFLICT(owner_ref,target_ref) DO UPDATE SET
                       scope=excluded.scope,updated_at=excluded.updated_at""",
                [
                    (str(owner_ref), ref, encoded_scope, now, now)
                    for ref in refs
                ],
            )
            retained = conn.execute(
                """SELECT COUNT(*) AS count FROM amos_reference_leases
                   WHERE owner_ref=?""",
                (str(owner_ref),),
            ).fetchone()
        return {
            "owner_ref": str(owner_ref),
            "published_count": len(refs),
            "retained_count": int(retained["count"] if retained else 0),
            "replace": bool(replace),
            "updated_at": now,
        }

    def reference_lease_refs_from_connection(
        self,
        conn: sqlite3.Connection,
    ) -> set[str]:
        return {
            str(row["target_ref"])
            for row in conn.execute(
                "SELECT DISTINCT target_ref FROM amos_reference_leases"
            ).fetchall()
            if str(row["target_ref"] or "")
        }

    def is_reference_leased(
        self,
        conn: sqlite3.Connection,
        target_ref: str,
    ) -> bool:
        return conn.execute(
            """SELECT 1 FROM amos_reference_leases
               WHERE target_ref=? LIMIT 1""",
            (str(target_ref),),
        ).fetchone() is not None

    def retention_protected_atom_refs_from_connection(
        self,
        conn: sqlite3.Connection,
        candidate_refs: Sequence[str],
    ) -> dict[str, Any]:
        """Resolve exact hot-state dependencies that cleanup must preserve.

        Retention protection is structural rather than name- or pattern-based:
        a payload value protects an atom only when the complete JSON string is
        an exact candidate atom ID. Live graph relations provide the second
        typed dependency source. Lifecycle supersession is intentionally not
        a retention dependency because the canonical successor and durable
        journal receipt are what make the predecessor safe to compact.
        """

        candidates = {
            str(ref).strip() for ref in candidate_refs if str(ref).strip()
        }
        by_reason: dict[str, set[str]] = {
            "current_head": set(),
            "reference_lease": set(),
            "hot_payload": set(),
            "hot_edge": set(),
        }
        if not candidates:
            return {"refs": set(), "by_reason": by_reason}

        head_refs = {
            str(row["head_ref"])
            for row in conn.execute(
                "SELECT DISTINCT head_ref FROM amos_memory_heads"
            ).fetchall()
            if str(row["head_ref"] or "")
        }
        by_reason["current_head"].update(candidates.intersection(head_refs))
        leased_refs = self.reference_lease_refs_from_connection(conn)
        by_reason["reference_lease"].update(
            candidates.intersection(leased_refs)
        )

        owner_rows = conn.execute(
            """SELECT atom.id,atom.payload,atom.evidence_refs
               FROM amos_atoms AS atom
               WHERE atom.deleted = 0
                 AND (
                   atom.lifecycle_state IN ('active','proposed')
                   OR EXISTS (
                     SELECT 1 FROM amos_memory_heads AS head
                     WHERE head.head_ref = atom.id
                   )
                 )"""
        ).fetchall()
        owner_refs = {str(row["id"]) for row in owner_rows}

        def collect_exact_refs(owner_ref: str, value: Any) -> None:
            stack = [value]
            while stack:
                item = stack.pop()
                if isinstance(item, str):
                    if item != owner_ref and item in candidates:
                        by_reason["hot_payload"].add(item)
                elif isinstance(item, Mapping):
                    stack.extend(item.values())
                elif isinstance(item, Sequence) and not isinstance(
                    item, (str, bytes, bytearray)
                ):
                    stack.extend(item)

        for row in owner_rows:
            owner_ref = str(row["id"])
            collect_exact_refs(owner_ref, self._json(row["payload"]))
            collect_exact_refs(owner_ref, self._json(row["evidence_refs"]))

        # Keep SQL parameter counts bounded even when an embedding configures a
        # large cleanup limit.
        ordered_candidates = sorted(candidates)
        for offset in range(0, len(ordered_candidates), 400):
            batch = ordered_candidates[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            edge_rows = conn.execute(
                f"""
                SELECT source_ref,target_ref,relation
                FROM amos_edges
                WHERE deleted = 0
                  AND lifecycle_state = 'active'
                  AND relation <> 'rel:supersedes'
                  AND (
                    source_ref IN ({placeholders})
                    OR target_ref IN ({placeholders})
                  )
                """,
                (*batch, *batch),
            ).fetchall()
            for row in edge_rows:
                source_ref = str(row["source_ref"])
                target_ref = str(row["target_ref"])
                if source_ref in candidates and target_ref in owner_refs:
                    by_reason["hot_edge"].add(source_ref)
                if target_ref in candidates and source_ref in owner_refs:
                    by_reason["hot_edge"].add(target_ref)

        protected_refs = set().union(*by_reason.values())
        return {"refs": protected_refs, "by_reason": by_reason}

    def append_event(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any],
        target_refs: list[str] | None = None,
        payload_refs: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        idempotency_key: str | None = None,
        causal_parent_ids: list[str] | None = None,
        expected_versions: Mapping[str, int] | None = None,
        authorization_context: Mapping[str, Any] | None = None,
        result_status: str = "accepted",
        projection_status: str = "projected",
    ) -> dict[str, Any]:
        previous_hash = self._get_meta(conn, "last_event_hash") or "genesis"
        graph_version = int(self._get_meta(conn, "graph_version") or 0) + 1
        occurred_at = utc_now()
        accepted_at = utc_now()
        payload_digest = digest(payload)
        event_id = f"evt_{uuid.uuid4().hex}"
        body = {
            "event_id": event_id,
            "event_type": event_type,
            "schema_version": SCHEMA_VERSION,
            "actor": actor,
            "target_refs": target_refs or [],
            "payload": payload,
            "payload_refs": payload_refs or [],
            "evidence_refs": evidence_refs or [],
            "idempotency_key": idempotency_key,
            "payload_digest": payload_digest,
            "causal_parent_ids": causal_parent_ids or [],
            "expected_versions": dict(expected_versions or {}),
            "authorization_context": dict(authorization_context or {}),
            "occurred_at": occurred_at,
            "accepted_at": accepted_at,
            "result_status": result_status,
            "projection_status": projection_status,
            "previous_event_hash": previous_hash,
            "graph_version": graph_version,
        }
        checksum = digest(body)
        body["checksum"] = checksum
        conn.execute(
            """
            INSERT INTO amos_event_journal(
                event_id, event_type, schema_version, actor, target_refs, payload,
                payload_refs, evidence_refs, idempotency_key, payload_digest,
                causal_parent_ids, expected_versions, authorization_context,
                occurred_at, accepted_at, result_status, projection_status,
                previous_event_hash, checksum, graph_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                SCHEMA_VERSION,
                actor,
                canonical_json(target_refs or []),
                canonical_json(payload),
                canonical_json(payload_refs or []),
                canonical_json(evidence_refs or []),
                idempotency_key,
                payload_digest,
                canonical_json(causal_parent_ids or []),
                canonical_json(dict(expected_versions or {})),
                canonical_json(dict(authorization_context or {})),
                occurred_at,
                accepted_at,
                result_status,
                projection_status,
                previous_hash,
                checksum,
                graph_version,
            ),
        )
        self._set_meta(conn, "graph_version", str(graph_version))
        self._set_meta(conn, "last_event_hash", checksum)
        return body

    def get_idempotency(
        self, conn: sqlite3.Connection, actor: str, key: str
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT actor, idempotency_key, payload_digest, event_id, response_json
            FROM amos_idempotency WHERE actor = ? AND idempotency_key = ?
            """,
            (actor, key),
        ).fetchone()
        if row is None:
            return None
        return {
            "actor": row["actor"],
            "idempotency_key": row["idempotency_key"],
            "payload_digest": row["payload_digest"],
            "event_id": row["event_id"],
            "response": self._json(row["response_json"]),
        }

    def put_idempotency(
        self,
        conn: sqlite3.Connection,
        *,
        actor: str,
        key: str,
        payload_digest: str,
        event_id: str,
        response: Mapping[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO amos_idempotency(
                actor, idempotency_key, payload_digest, event_id, response_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (actor, key, payload_digest, event_id, canonical_json(response), utc_now()),
        )

    def compact_idempotency_responses(
        self,
        conn: sqlite3.Connection,
        *,
        older_than: str,
        max_rows: int,
    ) -> dict[str, Any]:
        max_rows = max(0, int(max_rows))
        if max_rows <= 0:
            return {"status": "skipped", "reason": "max_rows_zero", "rows": 0}
        rows = conn.execute(
            """
            SELECT actor, idempotency_key, payload_digest, event_id, response_json
            FROM amos_idempotency
            WHERE created_at < ?
              AND response_json NOT LIKE '%"storage_compacted":true%'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (older_than, max_rows),
        ).fetchall()
        compacted = 0
        original_bytes = 0
        compacted_bytes = 0
        compacted_at = utc_now()
        for row in rows:
            response_json = row["response_json"] or ""
            original_bytes += len(response_json.encode("utf-8"))
            compact_response = {
                "status": "compacted",
                "storage_compacted": True,
                "event_id": row["event_id"],
                "payload_digest": row["payload_digest"],
                "original_response_bytes": len(response_json.encode("utf-8")),
                "compacted_at": compacted_at,
            }
            encoded = canonical_json(compact_response)
            compacted_bytes += len(encoded.encode("utf-8"))
            conn.execute(
                """
                UPDATE amos_idempotency
                SET response_json = ?
                WHERE actor = ? AND idempotency_key = ?
                """,
                (encoded, row["actor"], row["idempotency_key"]),
            )
            compacted += 1
        return {
            "status": "completed",
            "rows": compacted,
            "original_response_bytes": original_bytes,
            "compacted_response_bytes": compacted_bytes,
            "saved_bytes": max(0, original_bytes - compacted_bytes),
        }

    def insert_evidence(
        self, conn: sqlite3.Connection, evidence: Mapping[str, Any], event_id: str
    ) -> None:
        conn.execute(
            """
            INSERT INTO amos_evidence(
                evidence_id, schema_version, source_type, source_ref, payload,
                captured_at, checksum, access_policy, scope, event_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(evidence_id) DO NOTHING
            """,
            (
                evidence["evidence_id"],
                evidence["schema_version"],
                evidence["source_type"],
                evidence["source_ref"],
                canonical_json(evidence["payload"]),
                evidence["captured_at"],
                evidence["checksum"],
                canonical_json(evidence["access_policy"]),
                canonical_json(evidence["scope"]),
                event_id,
            ),
        )

    def list_evidence(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM amos_evidence ORDER BY captured_at DESC"
        ).fetchall()
        return [self._row_dict(row) for row in rows]

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM amos_evidence WHERE evidence_id = ?",
            (str(evidence_id),),
        ).fetchone()
        return None if row is None else self._row_dict(row)

    def get_atom(self, atom_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM amos_atoms WHERE id = ?", (atom_id,)).fetchone()
        return None if row is None else self._row_dict(row)

    def reference_records(
        self, refs: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Resolve reference namespaces in bulk without returning record payloads."""

        normalized = list(dict.fromkeys(str(ref) for ref in refs if str(ref)))
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        records: dict[str, list[dict[str, Any]]] = {
            ref: [] for ref in normalized
        }
        atom_rows = self.conn.execute(
            f"""
            SELECT id, scope, access_policy, deleted
            FROM amos_atoms
            WHERE id IN ({placeholders})
            """,
            tuple(normalized),
        ).fetchall()
        for row in atom_rows:
            records[str(row["id"])].append(
                {
                    "kind": "atom",
                    "scope": self._json(row["scope"]),
                    "access_policy": self._json(row["access_policy"]),
                    "deleted": bool(row["deleted"]),
                }
            )
        evidence_rows = self.conn.execute(
            f"""
            SELECT evidence_id, scope, access_policy
            FROM amos_evidence
            WHERE evidence_id IN ({placeholders})
            """,
            tuple(normalized),
        ).fetchall()
        for row in evidence_rows:
            records[str(row["evidence_id"])].append(
                {
                    "kind": "evidence",
                    "scope": self._json(row["scope"]),
                    "access_policy": self._json(row["access_policy"]),
                    "deleted": False,
                }
            )
        return records

    def insert_atom(self, conn: sqlite3.Connection, atom: Mapping[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO amos_atoms(
                id, type, schema_version, payload, evidence_refs, scope, confidence,
                salience, utility, layer, lifecycle_state, health_status,
                retention_class, access_policy, decay_policy, created_at, observed_at,
                updated_at, last_accessed, version, supersedes, revision_history,
                index_refs, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom["id"],
                atom["type"],
                atom["schema_version"],
                canonical_json(atom["payload"]),
                canonical_json(atom["evidence_refs"]),
                canonical_json(atom["scope"]),
                canonical_json(atom["confidence"]),
                atom["salience"],
                atom["utility"],
                atom["layer"],
                atom["lifecycle_state"],
                atom["health_status"],
                atom["retention_class"],
                canonical_json(atom["access_policy"]),
                canonical_json(atom["decay_policy"]),
                atom["created_at"],
                atom["observed_at"],
                atom["updated_at"],
                atom["last_accessed"],
                atom["version"],
                canonical_json(atom["supersedes"]),
                canonical_json(atom["revision_history"]),
                canonical_json(atom["index_refs"]),
                1 if atom.get("deleted") else 0,
            ),
        )
        self.replace_atom_text_index(conn, atom)

    def replace_atom(self, conn: sqlite3.Connection, atom: Mapping[str, Any]) -> None:
        conn.execute(
            """
            UPDATE amos_atoms SET
                payload = ?, evidence_refs = ?, scope = ?, confidence = ?,
                salience = ?, utility = ?, layer = ?, lifecycle_state = ?,
                health_status = ?, retention_class = ?, access_policy = ?,
                decay_policy = ?, observed_at = ?, updated_at = ?,
                last_accessed = ?, version = ?, supersedes = ?,
                revision_history = ?, index_refs = ?, deleted = ?
            WHERE id = ?
            """,
            (
                canonical_json(atom["payload"]),
                canonical_json(atom["evidence_refs"]),
                canonical_json(atom["scope"]),
                canonical_json(atom["confidence"]),
                atom["salience"],
                atom["utility"],
                atom["layer"],
                atom["lifecycle_state"],
                atom["health_status"],
                atom["retention_class"],
                canonical_json(atom["access_policy"]),
                canonical_json(atom["decay_policy"]),
                atom["observed_at"],
                atom["updated_at"],
                atom["last_accessed"],
                atom["version"],
                canonical_json(atom["supersedes"]),
                canonical_json(atom["revision_history"]),
                canonical_json(atom["index_refs"]),
                1 if atom.get("deleted") else 0,
                atom["id"],
            ),
        )
        self.replace_atom_text_index(conn, atom)

    def replace_atom_text_index(
        self, conn: sqlite3.Connection, atom: Mapping[str, Any]
    ) -> None:
        atom_id = str(atom.get("id") or "")
        if not atom_id:
            return
        conn.execute("DELETE FROM amos_atom_text_index WHERE atom_id = ?", (atom_id,))
        if atom.get("deleted") or atom.get("lifecycle_state") in {
            "archived",
            "superseded",
            "deleted",
        }:
            return
        tokens = sorted(self._atom_text_index_tokens(atom))
        if not tokens:
            return
        conn.executemany(
            """
            INSERT OR IGNORE INTO amos_atom_text_index(atom_id, token)
            VALUES (?, ?)
            """,
            [(atom_id, token) for token in tokens],
        )

    def delete_atom_text_index(self, conn: sqlite3.Connection, atom_id: str) -> int:
        cursor = conn.execute(
            "DELETE FROM amos_atom_text_index WHERE atom_id = ?",
            (str(atom_id),),
        )
        return int(cursor.rowcount or 0)

    def prune_atom_text_index(
        self,
        conn: sqlite3.Connection,
        *,
        lifecycle_states: list[str] | None = None,
        health_statuses: list[str] | None = None,
        max_atoms: int | None = None,
    ) -> dict[str, Any]:
        predicates = []
        params: list[Any] = []
        lifecycle_states = [str(item) for item in lifecycle_states or []]
        health_statuses = [str(item) for item in health_statuses or []]
        if lifecycle_states:
            placeholders = ",".join("?" for _ in lifecycle_states)
            predicates.append(f"a.lifecycle_state IN ({placeholders})")
            params.extend(lifecycle_states)
        if health_statuses:
            placeholders = ",".join("?" for _ in health_statuses)
            predicates.append(f"a.health_status IN ({placeholders})")
            params.extend(health_statuses)
        if not predicates:
            return {
                "status": "skipped",
                "reason": "no_prune_criteria",
                "rows": 0,
                "atom_count": 0,
            }
        query = f"""
            SELECT DISTINCT i.atom_id
            FROM amos_atom_text_index i
            JOIN amos_atoms a ON a.id = i.atom_id
            WHERE {' OR '.join(predicates)}
            ORDER BY i.atom_id ASC
            """
        if max_atoms is not None:
            query += " LIMIT ?"
            params.append(max(0, int(max_atoms)))
        atom_ids = [
            str(row["atom_id"])
            for row in conn.execute(query, tuple(params)).fetchall()
        ]
        if atom_ids:
            cursor = conn.execute(
                "DELETE FROM amos_atom_text_index WHERE atom_id IN ("
                + ",".join("?" for _ in atom_ids)
                + ")",
                tuple(atom_ids),
            )
            deleted_rows = int(cursor.rowcount or 0)
        else:
            deleted_rows = 0
        return {
            "status": "completed",
            "rows": deleted_rows,
            "atom_count": len(atom_ids),
            "lifecycle_states": lifecycle_states,
            "health_statuses": health_statuses,
            "max_atoms": max_atoms,
        }

    def _atom_text_index_tokens(self, atom: Mapping[str, Any]) -> set[str]:
        tokens: set[str] = set()
        index_refs = atom.get("index_refs", {})
        if isinstance(index_refs, Mapping):
            for index in index_refs.values():
                if not isinstance(index, Mapping):
                    continue
                for token in index.get("tokens", []) or []:
                    text = str(token or "").strip().lower()
                    if text:
                        tokens.add(text)
        return tokens

    def _backfill_atom_text_index(self, conn: sqlite3.Connection) -> None:
        atom_count = int(
            conn.execute(
                "SELECT COUNT(*) AS count FROM amos_atoms WHERE deleted = 0"
            ).fetchone()["count"]
        )
        if atom_count == 0:
            return
        indexed_count = int(
            conn.execute("SELECT COUNT(*) AS count FROM amos_atom_text_index").fetchone()[
                "count"
            ]
        )
        if indexed_count > 0:
            return
        rows = conn.execute("SELECT * FROM amos_atoms WHERE deleted = 0").fetchall()
        for row in rows:
            self.replace_atom_text_index(conn, self._row_dict(row))

    def list_atoms(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM amos_atoms ORDER BY updated_at DESC").fetchall()
        return [self._row_dict(row) for row in rows]

    def list_interaction_atoms(
        self,
        *,
        conversation_id: str,
        after_sequence: int | None = None,
        through_sequence: int | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        """Return one interaction stream without decoding unrelated atoms."""

        clauses = [
            "type = 'interaction_event'",
            "deleted = 0",
            "json_extract(payload, '$.conversation_id') = ?",
        ]
        params: list[Any] = [str(conversation_id)]
        sequence_sql = "CAST(json_extract(payload, '$.sequence') AS INTEGER)"
        if after_sequence is not None:
            clauses.append(f"{sequence_sql} > ?")
            params.append(int(after_sequence))
        if through_sequence is not None:
            clauses.append(f"{sequence_sql} <= ?")
            params.append(int(through_sequence))
        direction = "DESC" if newest_first else "ASC"
        query = (
            "SELECT * FROM amos_atoms WHERE "
            + " AND ".join(clauses)
            + f" ORDER BY {sequence_sql} {direction}, id {direction}"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [self._row_dict(row) for row in rows]

    def list_discourse_thread_atoms(
        self, *, conversation_id: str
    ) -> list[dict[str, Any]]:
        """Return roots belonging to one exact conversation."""

        rows = self.conn.execute(
            """
            SELECT *
            FROM amos_atoms
            WHERE type = 'discourse_thread'
              AND deleted = 0
              AND json_extract(payload, '$.conversation_id') = ?
            ORDER BY updated_at DESC
            """,
            (str(conversation_id),),
        ).fetchall()
        return [self._row_dict(row) for row in rows]

    def list_context_compaction_atoms(
        self,
        *,
        partition_key: str,
        through_sequence: int,
        profile: str,
    ) -> list[dict[str, Any]]:
        """Return viable rolling compactions for one interaction stream."""

        rows = self.conn.execute(
            """
            SELECT *
            FROM amos_atoms
            WHERE type = 'semantic'
              AND deleted = 0
              AND lifecycle_state = 'active'
              AND (
                    health_status IS NULL
                    OR health_status <> 'contradicted'
                  )
              AND json_extract(
                    payload,
                    '$.context_compaction.profile'
                  ) = ?
              AND json_extract(
                    payload,
                    '$.context_compaction.mode'
                  ) = 'rolling'
              AND json_extract(
                    payload,
                    '$.context_compaction.partition.kind'
                  ) = 'interaction_stream'
              AND json_extract(
                    payload,
                    '$.context_compaction.partition.key'
                  ) = ?
              AND CAST(json_extract(
                    payload,
                    '$.context_compaction.coverage.through_sequence'
                  ) AS INTEGER) <= ?
            ORDER BY CAST(json_extract(
                       payload,
                       '$.context_compaction.coverage.through_sequence'
                     ) AS INTEGER) DESC,
                     updated_at DESC,
                     id DESC
            """,
            (str(profile), str(partition_key), int(through_sequence)),
        ).fetchall()
        return [self._row_dict(row) for row in rows]

    def list_atoms_filtered(
        self,
        *,
        include_deleted: bool = False,
        types: list[str] | None = None,
        lifecycle_states: list[str] | None = None,
        excluded_health: list[str] | None = None,
        included_health: list[str] | None = None,
        atom_ids: list[str] | None = None,
        payload_filter: Mapping[str, Sequence[str]] | None = None,
        limit: int | None = None,
        prioritize_hot: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_deleted:
            clauses.append("deleted = 0")
        if types:
            clauses.append(f"type IN ({','.join('?' for _ in types)})")
            params.extend(types)
        if lifecycle_states:
            clauses.append(
                f"lifecycle_state IN ({','.join('?' for _ in lifecycle_states)})"
            )
            params.extend(lifecycle_states)
        if excluded_health:
            clauses.append(
                f"health_status NOT IN ({','.join('?' for _ in excluded_health)})"
            )
            params.extend(excluded_health)
        if included_health:
            clauses.append(
                f"health_status IN ({','.join('?' for _ in included_health)})"
            )
            params.extend(included_health)
        if atom_ids is not None:
            if not atom_ids:
                return []
            clauses.append(f"id IN ({','.join('?' for _ in atom_ids)})")
            params.extend(atom_ids)
        self._append_payload_filter_sql(
            clauses,
            params,
            payload_filter=payload_filter,
        )
        query = "SELECT * FROM amos_atoms"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        if prioritize_hot:
            query += (
                " ORDER BY CASE lifecycle_state "
                "WHEN 'active' THEN 0 WHEN 'proposed' THEN 1 ELSE 2 END, "
                "updated_at DESC"
            )
        else:
            query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [self._row_dict(row) for row in rows]

    def atom_text_index_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM amos_atom_text_index"
        ).fetchone()
        return int(row["count"])

    def atom_text_document_count(self) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(DISTINCT atom_id) AS count
            FROM amos_atom_text_index
            """
        ).fetchone()
        return int(row["count"])

    def token_document_frequencies(self) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT token, COUNT(DISTINCT atom_id) AS document_frequency
            FROM amos_atom_text_index
            GROUP BY token
            """
        ).fetchall()
        return {
            str(row["token"]): int(row["document_frequency"])
            for row in rows
        }

    def token_atom_index_rows(
        self, *, max_terms: int | None = None
    ) -> list[tuple[str, str]]:
        params: tuple[Any, ...] = ()
        term_filter = ""
        if max_terms is not None:
            term_filter = """
                WHERE token IN (
                    SELECT token
                    FROM amos_atom_text_index
                    GROUP BY token
                    ORDER BY COUNT(DISTINCT atom_id) DESC, token ASC
                    LIMIT ?
                )
            """
            params = (max(0, int(max_terms)),)
        rows = self.conn.execute(
            f"""
            SELECT atom_id, token
            FROM amos_atom_text_index
            {term_filter}
            ORDER BY atom_id ASC, token ASC
            """,
            params,
        ).fetchall()
        return [(str(row["atom_id"]), str(row["token"])) for row in rows]

    def candidate_atom_ids_for_tokens(
        self,
        tokens: list[str],
        *,
        limit: int | None = None,
        eligible_atom_ids: set[str] | None = None,
        payload_filter: Mapping[str, Sequence[str]] | None = None,
    ) -> list[str]:
        normalized = sorted(
            {
                str(token or "").strip().lower()
                for token in tokens
                if str(token or "").strip()
            }
        )
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        query = f"""
            SELECT i.atom_id, i.token
            FROM amos_atom_text_index AS i
            JOIN amos_atoms AS a ON a.id = i.atom_id
            WHERE i.token IN ({placeholders})
              AND a.deleted = 0
        """
        params: list[Any] = list(normalized)
        if eligible_atom_ids is not None:
            eligible = sorted({str(ref) for ref in eligible_atom_ids if str(ref)})
            if not eligible:
                return []
            query += f" AND i.atom_id IN ({','.join('?' for _ in eligible)})"
            params.extend(eligible)
        payload_clauses: list[str] = []
        self._append_payload_filter_sql(
            payload_clauses,
            params,
            payload_filter=payload_filter,
            payload_column="a.payload",
        )
        if payload_clauses:
            query += " AND " + " AND ".join(payload_clauses)
        rows = self.conn.execute(query, tuple(params)).fetchall()
        if not rows:
            return []
        document_count = max(1, self.atom_text_document_count())
        frequencies = self.token_document_frequencies()
        scores: dict[str, float] = {}
        for row in rows:
            atom_id = str(row["atom_id"])
            token = str(row["token"])
            inverse_frequency = math.log(
                (document_count + 1.0) / (frequencies.get(token, document_count) + 1.0)
            ) + 1.0
            scores[atom_id] = scores.get(atom_id, 0.0) + inverse_frequency
        ranked = sorted(scores, key=lambda atom_id: (-scores[atom_id], atom_id))
        return ranked if limit is None else ranked[: max(0, int(limit))]

    def neighbor_atom_ids(
        self, refs: list[str], *, edge_limit: int | None = None
    ) -> list[str]:
        neighbors: set[str] = set()
        edges = (
            self.list_edges_for_refs(refs)
            if edge_limit is None
            else self.list_edges_for_refs(refs, limit=edge_limit)
        )
        for edge in edges:
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source:
                neighbors.add(source)
            if target:
                neighbors.add(target)
        return sorted(neighbors)

    def atom_count(self, *, include_deleted: bool = False) -> int:
        query = "SELECT COUNT(*) AS count FROM amos_atoms"
        params: tuple[Any, ...] = ()
        if not include_deleted:
            query += " WHERE deleted = 0"
        row = self.conn.execute(query, params).fetchone()
        return int(row["count"])

    def atom_count_filtered(
        self,
        *,
        types: Sequence[str] | None = None,
        lifecycle_states: Sequence[str] | None = None,
        payload_filter: Mapping[str, Sequence[str]] | None = None,
        include_deleted: bool = False,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_deleted:
            clauses.append("deleted = 0")
        normalized_types = [str(item) for item in types or []]
        if normalized_types:
            clauses.append(
                f"type IN ({','.join('?' for _ in normalized_types)})"
            )
            params.extend(normalized_types)
        normalized_lifecycle = [str(item) for item in lifecycle_states or []]
        if normalized_lifecycle:
            clauses.append(
                "lifecycle_state IN ("
                + ",".join("?" for _ in normalized_lifecycle)
                + ")"
            )
            params.extend(normalized_lifecycle)
        self._append_payload_filter_sql(
            clauses,
            params,
            payload_filter=payload_filter,
        )
        query = "SELECT COUNT(*) AS count FROM amos_atoms"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        row = self.conn.execute(query, tuple(params)).fetchone()
        return int(row["count"])

    @staticmethod
    def _append_payload_filter_sql(
        clauses: list[str],
        params: list[Any],
        *,
        payload_filter: Mapping[str, Sequence[str]] | None,
        payload_column: str = "payload",
    ) -> None:
        for raw_field, raw_values in (payload_filter or {}).items():
            field = str(raw_field)
            values = tuple(dict.fromkeys(str(value) for value in raw_values))
            if not values:
                clauses.append("0 = 1")
                continue
            if field in {"profile", "plugin_digest"}:
                expression = f"json_extract({payload_column}, '$.{field}')"
            else:
                expression = f"json_extract({payload_column}, ?)"
                params.append(f'$."{field}"')
            clauses.append(
                f"CAST({expression} AS TEXT) IN ("
                "SELECT CAST(value AS TEXT) FROM json_each(?)"
                ")"
            )
            params.append(canonical_json(list(values)))

    def active_atom_ids(
        self, *, lifecycle_states: list[str] | None = None
    ) -> set[str]:
        lifecycle_states = lifecycle_states or ["active", "proposed"]
        placeholders = ",".join("?" for _ in lifecycle_states)
        rows = self.conn.execute(
            f"""
            SELECT id
            FROM amos_atoms
            WHERE deleted = 0
              AND lifecycle_state IN ({placeholders})
            """,
            tuple(lifecycle_states),
        ).fetchall()
        return {str(row["id"]) for row in rows}

    def atom_counts_by(self, column: str, *, include_deleted: bool = False) -> dict[str, int]:
        if column not in {"type", "health_status", "lifecycle_state"}:
            raise ValueError(f"unsupported atom count column: {column}")
        query = f"SELECT {column} AS key, COUNT(*) AS count FROM amos_atoms"
        if not include_deleted:
            query += " WHERE deleted = 0"
        query += f" GROUP BY {column}"
        rows = self.conn.execute(query).fetchall()
        return {str(row["key"]): int(row["count"]) for row in rows}

    def insert_edge(self, conn: sqlite3.Connection, edge: Mapping[str, Any]) -> bool:
        retired = conn.execute(
            "SELECT 1 FROM amos_retired_edges WHERE edge_id = ?",
            (str(edge["edge_id"]),),
        ).fetchone()
        if retired is not None:
            return False
        cursor = conn.execute(
            """
            INSERT INTO amos_edges(
                edge_id, source_ref, target_ref, relation, schema_version,
                evidence_refs, scope, confidence, derivation, lifecycle_state,
                health_status, created_at, updated_at, version, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO NOTHING
            """,
            (
                edge["edge_id"],
                edge["source_ref"],
                edge["target_ref"],
                edge["relation"],
                edge["schema_version"],
                canonical_json(edge["evidence_refs"]),
                canonical_json(edge["scope"]),
                canonical_json(edge["confidence"]),
                canonical_json(edge.get("derivation") or {}),
                edge["lifecycle_state"],
                edge["health_status"],
                edge["created_at"],
                edge["updated_at"],
                edge["version"],
                1 if edge.get("deleted") else 0,
            ),
        )
        return cursor.rowcount > 0

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM amos_edges WHERE edge_id = ?", (edge_id,)
        ).fetchone()
        if row is not None:
            return self._row_dict(row)
        retired = self.conn.execute(
            "SELECT * FROM amos_retired_edges WHERE edge_id = ?", (edge_id,)
        ).fetchone()
        if retired is None:
            return None
        return {
            "edge_id": str(retired["edge_id"]),
            "source_ref": str(retired["source_ref"]),
            "target_ref": str(retired["target_ref"]),
            "relation": str(retired["relation"]),
            "scope": self._json(str(retired["scope"])),
            "schema_version": SCHEMA_VERSION,
            "evidence_refs": [],
            "confidence": {},
            "derivation": {},
            "lifecycle_state": "deleted",
            "health_status": "deleted",
            "created_at": str(retired["retired_at"]),
            "updated_at": str(retired["retired_at"]),
            "version": 0,
            "deleted": 1,
            "storage_compacted": True,
        }

    def upsert_edge(self, conn: sqlite3.Connection, edge: Mapping[str, Any]) -> None:
        """Project an edge state, including lifecycle reactivation on promotion."""
        retired = conn.execute(
            "SELECT 1 FROM amos_retired_edges WHERE edge_id = ?",
            (str(edge["edge_id"]),),
        ).fetchone()
        if retired is not None:
            return
        conn.execute(
            """
            INSERT INTO amos_edges(
                edge_id, source_ref, target_ref, relation, schema_version,
                evidence_refs, scope, confidence, derivation, lifecycle_state,
                health_status, created_at, updated_at, version, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                source_ref = excluded.source_ref,
                target_ref = excluded.target_ref,
                relation = excluded.relation,
                schema_version = excluded.schema_version,
                evidence_refs = excluded.evidence_refs,
                scope = excluded.scope,
                confidence = excluded.confidence,
                derivation = excluded.derivation,
                lifecycle_state = excluded.lifecycle_state,
                health_status = excluded.health_status,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                version = excluded.version,
                deleted = excluded.deleted
            """,
            (
                edge["edge_id"],
                edge["source_ref"],
                edge["target_ref"],
                edge["relation"],
                edge["schema_version"],
                canonical_json(edge["evidence_refs"]),
                canonical_json(edge["scope"]),
                canonical_json(edge["confidence"]),
                canonical_json(edge.get("derivation") or {}),
                edge["lifecycle_state"],
                edge["health_status"],
                edge["created_at"],
                edge["updated_at"],
                edge["version"],
                1 if edge.get("deleted") else 0,
            ),
        )

    def list_edges(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM amos_edges"
        if not include_deleted:
            query += " WHERE deleted = 0"
        rows = self.conn.execute(query).fetchall()
        return [self._row_dict(row) for row in rows]

    def edge_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM amos_edges WHERE deleted = 0"
        ).fetchone()
        return int(row["count"])

    def memory_integrity_summary(self) -> dict[str, int]:
        """Return UI health counters without materializing the memory graph."""

        reference_row = self.conn.execute(
            """WITH refs AS (
                 SELECT CAST(item.value AS TEXT) AS ref
                 FROM amos_atoms AS atom,
                      json_each(atom.evidence_refs) AS item
                 WHERE atom.deleted=0
                   AND atom.lifecycle_state IN ('active','proposed')
                 UNION ALL
                 SELECT CAST(item.value AS TEXT) AS ref
                 FROM amos_edges AS edge,
                      json_each(edge.evidence_refs) AS item
                 WHERE edge.deleted=0
               )
               SELECT
                 SUM(CASE WHEN EXISTS (
                   SELECT 1 FROM amos_evidence AS evidence
                   WHERE evidence.evidence_id=refs.ref
                 ) THEN 1 ELSE 0 END) AS exact_evidence_refs,
                 SUM(CASE WHEN NOT EXISTS (
                   SELECT 1 FROM amos_evidence AS evidence
                   WHERE evidence.evidence_id=refs.ref
                 ) AND EXISTS (
                   SELECT 1 FROM amos_atoms AS target
                   WHERE target.id=refs.ref AND target.deleted=0
                 ) THEN 1 ELSE 0 END) AS mistyped_atom_refs,
                 SUM(CASE WHEN NOT EXISTS (
                   SELECT 1 FROM amos_evidence AS evidence
                   WHERE evidence.evidence_id=refs.ref
                 ) AND NOT EXISTS (
                   SELECT 1 FROM amos_atoms AS target
                   WHERE target.id=refs.ref AND target.deleted=0
                 ) THEN 1 ELSE 0 END) AS unresolved_refs
               FROM refs"""
        ).fetchone()
        isolated_row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM amos_atoms AS atom
               WHERE atom.deleted=0 AND atom.lifecycle_state='active'
                 AND NOT EXISTS (
                   SELECT 1 FROM amos_edges AS edge
                   WHERE edge.source_ref=atom.id OR edge.target_ref=atom.id
                 )
                 AND NOT EXISTS (
                   SELECT 1 FROM amos_retired_edges AS edge
                   WHERE edge.source_ref=atom.id OR edge.target_ref=atom.id
                 )
                 AND NOT EXISTS (
                   SELECT 1 FROM amos_memory_heads AS head
                   WHERE head.head_ref=atom.id
                 )"""
        ).fetchone()
        superseded_row = self.conn.execute(
            """SELECT COUNT(DISTINCT edge.target_ref) AS count
               FROM amos_edges AS edge
               JOIN amos_atoms AS source
                 ON source.id=edge.source_ref
                AND source.deleted=0
                AND source.lifecycle_state='active'
               JOIN amos_atoms AS target
                 ON target.id=edge.target_ref
                AND target.deleted=0
                AND target.lifecycle_state IN ('active','superseded')
               WHERE edge.deleted=0
                 AND edge.lifecycle_state='active'
                 AND edge.relation='rel:supersedes'"""
        ).fetchone()
        return {
            "exact_evidence_refs": int(
                reference_row["exact_evidence_refs"] or 0
            ),
            "mistyped_atom_refs": int(
                reference_row["mistyped_atom_refs"] or 0
            ),
            "unresolved_refs": int(reference_row["unresolved_refs"] or 0),
            "isolated_active_atoms": int(isolated_row["count"] or 0),
            "active_superseded_atoms": int(superseded_row["count"] or 0),
        }

    def edge_degree_counts(
        self,
        refs: list[str] | None = None,
        *,
        include_deleted: bool = False,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        if refs is not None:
            ref_set = {str(ref) for ref in refs if str(ref)}
            if not ref_set:
                return counts
            if include_deleted:
                placeholders = ",".join("?" for _ in ref_set)
                rows = self.conn.execute(
                    f"""
                    SELECT source_ref, target_ref
                    FROM amos_edges
                    WHERE source_ref IN ({placeholders})
                       OR target_ref IN ({placeholders})
                    UNION ALL
                    SELECT source_ref, target_ref
                    FROM amos_retired_edges
                    WHERE source_ref IN ({placeholders})
                       OR target_ref IN ({placeholders})
                    """,
                    (
                        *sorted(ref_set),
                        *sorted(ref_set),
                        *sorted(ref_set),
                        *sorted(ref_set),
                    ),
                ).fetchall()
            else:
                rows = self.list_edges_for_refs(sorted(ref_set))
            for edge in rows:
                source = str(edge["source_ref"])
                target = str(edge["target_ref"])
                if source in ref_set:
                    counts[source] = counts.get(source, 0) + 1
                if target in ref_set:
                    counts[target] = counts.get(target, 0) + 1
        else:
            if include_deleted:
                rows = self.conn.execute(
                    """
                    SELECT source_ref, target_ref FROM amos_edges
                    UNION ALL
                    SELECT source_ref, target_ref FROM amos_retired_edges
                    """
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT source_ref, target_ref
                    FROM amos_edges
                    WHERE deleted = 0
                    """
                ).fetchall()
            for row in rows:
                counts[str(row["source_ref"])] = counts.get(str(row["source_ref"]), 0) + 1
                counts[str(row["target_ref"])] = counts.get(str(row["target_ref"]), 0) + 1
        return counts

    def list_edges_for_refs(
        self,
        refs: list[str],
        *,
        relations: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        refs = sorted({str(ref) for ref in refs if str(ref)})
        if not refs:
            return []
        placeholders = ",".join("?" for _ in refs)
        query = f"""
            SELECT * FROM amos_edges
            WHERE deleted = 0
              AND (source_ref IN ({placeholders}) OR target_ref IN ({placeholders}))
        """
        params: list[Any] = [*refs, *refs]
        if relations is not None:
            normalized_relations = sorted(
                {str(relation) for relation in relations if str(relation)}
            )
            if not normalized_relations:
                return []
            query += (
                " AND relation IN ("
                + ",".join("?" for _ in normalized_relations)
                + ")"
            )
            params.extend(normalized_relations)
        query += " ORDER BY edge_id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [self._row_dict(row) for row in rows]

    def mark_edges_deleted_for_ref(
        self, conn: sqlite3.Connection, target_ref: str
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT * FROM amos_edges
            WHERE deleted = 0 AND (source_ref = ? OR target_ref = ?)
            """,
            (target_ref, target_ref),
        ).fetchall()
        edges = [self._row_dict(row) for row in rows]
        now = utc_now()
        for edge in edges:
            edge["deleted"] = 1
            edge["lifecycle_state"] = "deleted"
            edge["health_status"] = "deleted"
            edge["updated_at"] = now
            edge["version"] = int(edge["version"]) + 1
            conn.execute(
                """
                UPDATE amos_edges SET
                    lifecycle_state = ?, health_status = ?, updated_at = ?,
                    version = ?, deleted = 1
                WHERE edge_id = ?
                """,
                (
                    edge["lifecycle_state"],
                    edge["health_status"],
                    edge["updated_at"],
                    edge["version"],
                    edge["edge_id"],
                ),
            )
        return edges

    def retire_and_purge_edges_for_ref(
        self,
        conn: sqlite3.Connection,
        target_ref: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Replace full deleted edge rows with minimal identity markers."""

        rows = conn.execute(
            """
            SELECT edge_id, source_ref, target_ref, relation, scope
            FROM amos_edges
            WHERE source_ref = ? OR target_ref = ?
            """,
            (str(target_ref), str(target_ref)),
        ).fetchall()
        retired_at = utc_now()
        conn.executemany(
            """
            INSERT INTO amos_retired_edges(
                edge_id, source_ref, target_ref, relation, scope,
                retired_at, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO NOTHING
            """,
            [
                (
                    str(row["edge_id"]),
                    str(row["source_ref"]),
                    str(row["target_ref"]),
                    str(row["relation"]),
                    str(row["scope"]),
                    retired_at,
                    str(reason),
                )
                for row in rows
            ],
        )
        deleted = conn.execute(
            "DELETE FROM amos_edges WHERE source_ref = ? OR target_ref = ?",
            (str(target_ref), str(target_ref)),
        )
        return {
            "status": "completed",
            "rows": int(deleted.rowcount or 0),
            "retired_edge_ids": [str(row["edge_id"]) for row in rows],
        }

    def purge_atom_projection(
        self, conn: sqlite3.Connection, atom_id: str
    ) -> dict[str, Any]:
        """Physically remove an atom row after its tombstone is durable."""

        index_rows = self.delete_atom_text_index(conn, atom_id)
        deleted = conn.execute(
            "DELETE FROM amos_atoms WHERE id = ?", (str(atom_id),)
        )
        return {
            "status": "completed",
            "atom_rows": int(deleted.rowcount or 0),
            "index_rows": index_rows,
        }

    def restore_edges_for_ref(
        self, conn: sqlite3.Connection, target_ref: str
    ) -> list[dict[str, Any]]:
        """Reactivate archived edges from a restored current memory head.

        Only edges whose opposite endpoint is still active or proposed are
        restored. This avoids reviving an obsolete subgraph merely because a
        canonical head was incorrectly archived by an older pressure policy.
        """

        rows = conn.execute(
            """
            SELECT e.*
            FROM amos_edges e
            WHERE e.deleted = 1
              AND (e.source_ref = ? OR e.target_ref = ?)
              AND EXISTS (
                  SELECT 1
                  FROM amos_atoms a
                  WHERE a.id = CASE
                      WHEN e.source_ref = ? THEN e.target_ref
                      ELSE e.source_ref
                  END
                    AND a.deleted = 0
                    AND a.lifecycle_state IN ('active', 'proposed')
              )
            """,
            (target_ref, target_ref, target_ref),
        ).fetchall()
        edges = [self._row_dict(row) for row in rows]
        now = utc_now()
        for edge in edges:
            edge["deleted"] = 0
            edge["lifecycle_state"] = "active"
            edge["health_status"] = "healthy"
            edge["updated_at"] = now
            edge["version"] = int(edge["version"]) + 1
            self.upsert_edge(conn, edge)
        return edges

    def mark_edges_deleted(
        self, conn: sqlite3.Connection, edge_ids: list[str]
    ) -> list[dict[str, Any]]:
        edge_ids = sorted({str(edge_id) for edge_id in edge_ids if str(edge_id)})
        if not edge_ids:
            return []
        placeholders = ",".join("?" for _ in edge_ids)
        rows = conn.execute(
            f"SELECT * FROM amos_edges WHERE deleted = 0 AND edge_id IN ({placeholders})",
            tuple(edge_ids),
        ).fetchall()
        edges = [self._row_dict(row) for row in rows]
        now = utc_now()
        for edge in edges:
            edge["deleted"] = 1
            edge["lifecycle_state"] = "deleted"
            edge["health_status"] = "deleted"
            edge["updated_at"] = now
            edge["version"] = int(edge["version"]) + 1
            conn.execute(
                """
                UPDATE amos_edges SET
                    lifecycle_state = ?, health_status = ?, updated_at = ?,
                    version = ?, deleted = 1
                WHERE edge_id = ?
                """,
                (
                    edge["lifecycle_state"],
                    edge["health_status"],
                    edge["updated_at"],
                    edge["version"],
                    edge["edge_id"],
                ),
            )
        return edges

    def insert_tombstone(
        self,
        conn: sqlite3.Connection,
        *,
        target_ref: str,
        content_digest: str,
        recreation_policy: str,
        reason: str,
    ) -> dict[str, Any]:
        tombstone = {
            "tombstone_id": f"tmb_{uuid.uuid4().hex}",
            "target_ref": target_ref,
            "content_digest": content_digest,
            "recreation_policy": recreation_policy,
            "reason": reason,
            "created_at": utc_now(),
        }
        conn.execute(
            """
            INSERT INTO amos_tombstones(
                tombstone_id, target_ref, content_digest, recreation_policy, reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                tombstone["tombstone_id"],
                target_ref,
                content_digest,
                recreation_policy,
                reason,
                tombstone["created_at"],
            ),
        )
        return tombstone

    def get_tombstone(
        self, target_ref: str | None = None, content_digest: str | None = None
    ) -> dict[str, Any] | None:
        if target_ref is None and content_digest is None:
            return None
        clauses = []
        params = []
        if target_ref is not None:
            clauses.append("target_ref = ?")
            params.append(target_ref)
        if content_digest is not None:
            clauses.append("content_digest = ?")
            params.append(content_digest)
        row = self.conn.execute(
            f"""
            SELECT * FROM amos_tombstones WHERE {' OR '.join(clauses)}
            ORDER BY created_at DESC LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return None if row is None else self._row_dict(row)

    def cache_packet(
        self,
        conn: sqlite3.Connection,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        graph_version: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO amos_packet_cache(
                packet_id, request_digest, graph_version, request_json,
                response_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(packet_id) DO UPDATE SET
                request_digest = excluded.request_digest,
                graph_version = excluded.graph_version,
                request_json = excluded.request_json,
                response_json = excluded.response_json,
                created_at = excluded.created_at
            """,
            (
                packet_id,
                digest(request),
                graph_version,
                canonical_json(request),
                canonical_json(response),
                utc_now(),
            ),
        )
        self.cache_packet_feedback_receipt(
            conn,
            packet_id=packet_id,
            response=response,
            graph_version=graph_version,
        )

    @staticmethod
    def _packet_feedback_projection(response: Mapping[str, Any]) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        evidence_refs: set[str] = set()
        for raw in response.get("items", []) or []:
            if not isinstance(raw, Mapping):
                continue
            item_ref = str(
                raw.get("atom_ref")
                or raw.get("atom_id")
                or raw.get("item_ref")
                or ""
            )
            if not item_ref:
                continue
            trace = [
                {"edge_id": str(step.get("edge_id") or "")}
                for step in raw.get("association_trace", []) or []
                if isinstance(step, Mapping) and str(step.get("edge_id") or "")
            ]
            visible_evidence = sorted(
                {
                    str(ref)
                    for ref in raw.get("evidence_refs", []) or []
                    if str(ref)
                }
            )
            evidence_refs.update(visible_evidence)
            items.append(
                {
                    "item_ref": item_ref,
                    "association_trace": trace,
                    "evidence_refs": visible_evidence,
                }
            )
        record = response.get("record")
        if isinstance(record, Mapping) and str(record.get("evidence_id") or ""):
            evidence_refs.add(str(record["evidence_id"]))
        return {
            "retrieval_mode": str(response.get("retrieval_mode") or ""),
            "items": items,
            "evidence_refs": sorted(evidence_refs),
        }

    def cache_packet_feedback_receipt(
        self,
        conn: sqlite3.Connection,
        *,
        packet_id: str,
        response: Mapping[str, Any],
        graph_version: int,
    ) -> None:
        """Persist the compact packet membership needed by delayed feedback."""

        conn.execute(
            """
            INSERT INTO amos_retrieval_packet_receipts(
                packet_id, graph_version, feedback_json, created_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(packet_id) DO UPDATE SET
                graph_version = excluded.graph_version,
                feedback_json = excluded.feedback_json,
                created_at = excluded.created_at
            """,
            (
                str(packet_id),
                int(graph_version),
                canonical_json(self._packet_feedback_projection(response)),
                utc_now(),
            ),
        )

    def get_packet_feedback_receipt(self, packet_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT packet_id, graph_version, feedback_json, created_at
            FROM amos_retrieval_packet_receipts
            WHERE packet_id = ?
            """,
            (str(packet_id),),
        ).fetchone()
        return None if row is None else self._row_dict(row)

    def get_cached_packet(
        self, *, request: Mapping[str, Any], graph_version: int
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT response_json
            FROM amos_packet_cache
            WHERE request_digest = ? AND graph_version = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (digest(request), int(graph_version)),
        ).fetchone()
        return None if row is None else self._json(row["response_json"])

    def get_cached_packet_by_id(self, packet_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT response_json FROM amos_packet_cache WHERE packet_id = ?",
            (str(packet_id),),
        ).fetchone()
        return None if row is None else self._json(row["response_json"])

    def clear_packet_cache(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM amos_packet_cache")

    def retire_packet_cache(
        self, conn: sqlite3.Connection, *, max_rows: int = 128
    ) -> dict[str, Any]:
        """Logically invalidate by revision and physically prune a bounded tail."""

        max_rows = max(0, int(max_rows))
        if max_rows == 0:
            return {"status": "retired", "rows": 0, "max_rows": 0}
        current_graph_version = self.graph_version()
        rows = conn.execute(
            """
            SELECT packet_id
            FROM amos_packet_cache
            WHERE graph_version <> ?
            ORDER BY graph_version ASC, created_at ASC
            LIMIT ?
            """,
            (current_graph_version, max_rows),
        ).fetchall()
        packet_ids = [str(row["packet_id"]) for row in rows]
        if packet_ids:
            conn.execute(
                "DELETE FROM amos_packet_cache WHERE packet_id IN ("
                + ",".join("?" for _ in packet_ids)
                + ")",
                tuple(packet_ids),
            )
        return {
            "status": "retired",
            "rows": len(packet_ids),
            "max_rows": max_rows,
            "current_graph_version": current_graph_version,
            "logical_invalidation": "graph_version_key",
        }

    def list_packet_cache(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM amos_packet_cache").fetchall()
        return [self._row_dict(row) for row in rows]

    def insert_retrieval_outcome(
        self,
        conn: sqlite3.Connection,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        outcome_payload = dict(outcome)
        outcome_id = str(
            outcome_payload.get("outcome_id")
            or f"rto_{digest({'packet_id': packet_id, 'request': request, 'outcome': outcome_payload})[:32]}"
        )
        record = {
            "outcome_id": outcome_id,
            "packet_id": packet_id,
            "request": dict(request),
            "outcome": outcome_payload,
            "created_at": utc_now(),
        }
        existing = conn.execute(
            "SELECT * FROM amos_retrieval_outcomes WHERE outcome_id = ?",
            (outcome_id,),
        ).fetchone()
        if existing is not None:
            existing_record = self._row_dict(existing)
            existing_record["status"] = "already_recorded"
            return existing_record
        conn.execute(
            """
            INSERT INTO amos_retrieval_outcomes(
                outcome_id, packet_id, request_json, outcome_json, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record["outcome_id"],
                packet_id,
                canonical_json(request),
                canonical_json(outcome),
                record["created_at"],
            ),
        )
        record["status"] = "recorded"
        return record

    def upsert_derived_index_metadata(
        self,
        conn: sqlite3.Connection,
        *,
        index_name: str,
        graph_version: int,
        freshness: str,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = {
            "index_name": index_name,
            "graph_version": graph_version,
            "freshness": freshness,
            "rebuilt_at": utc_now(),
            "details_json": dict(details),
        }
        conn.execute(
            """
            INSERT INTO amos_derived_index_metadata(
                index_name, graph_version, freshness, rebuilt_at, details_json
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(index_name) DO UPDATE SET
                graph_version = excluded.graph_version,
                freshness = excluded.freshness,
                rebuilt_at = excluded.rebuilt_at,
                details_json = excluded.details_json
            """,
            (
                index_name,
                graph_version,
                freshness,
                record["rebuilt_at"],
                canonical_json(details),
            ),
        )
        return record

    def replace_token_latent_vectors(
        self,
        conn: sqlite3.Connection,
        *,
        graph_version: int,
        dimensions: int,
        vectors: Mapping[str, Sequence[float]],
    ) -> dict[str, Any]:
        conn.execute("DELETE FROM amos_token_latent_vectors")
        updated_at = utc_now()
        rows = [
            (
                str(token),
                int(graph_version),
                int(dimensions),
                canonical_json([round(float(value), 8) for value in vector]),
                updated_at,
            )
            for token, vector in sorted(vectors.items())
        ]
        if rows:
            conn.executemany(
                """
                INSERT INTO amos_token_latent_vectors(
                    token, graph_version, dimensions, vector_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        return {
            "status": "replaced",
            "graph_version": int(graph_version),
            "dimensions": int(dimensions),
            "token_count": len(rows),
            "updated_at": updated_at,
        }

    def list_token_latent_vectors(
        self, *, graph_version: int | None = None
    ) -> dict[str, list[float]]:
        params: tuple[Any, ...] = ()
        query = "SELECT token, vector_json FROM amos_token_latent_vectors"
        if graph_version is not None:
            query += " WHERE graph_version = ?"
            params = (int(graph_version),)
        rows = self.conn.execute(query, params).fetchall()
        return {
            str(row["token"]): [float(value) for value in self._json(row["vector_json"])]
            for row in rows
        }

    def list_derived_index_metadata(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM amos_derived_index_metadata ORDER BY index_name"
        ).fetchall()
        return [self._row_dict(row) for row in rows]

    def retrieval_outcome_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM amos_retrieval_outcomes"
        ).fetchone()
        return int(row["count"])

    def list_retrieval_outcomes(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM amos_retrieval_outcomes ORDER BY created_at DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (max(0, int(limit)),)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_dict(row) for row in rows]

    def checkpoint_wal(self, *, mode: str = "PASSIVE") -> dict[str, Any]:
        safe_mode = str(mode or "PASSIVE").upper()
        if safe_mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            safe_mode = "PASSIVE"
        if safe_mode == "PASSIVE":
            with self._connection_lock:
                row = self.conn.execute(
                    f"PRAGMA wal_checkpoint({safe_mode})"
                ).fetchone()
        else:
            with self._connection_lock:
                with self._runtime.writer.acquire("maintenance"):
                    row = self.conn.execute(
                        f"PRAGMA wal_checkpoint({safe_mode})"
                    ).fetchone()
        values = list(row) if row is not None else []
        busy = int(values[0]) if len(values) > 0 and values[0] is not None else None
        log_pages = int(values[1]) if len(values) > 1 and values[1] is not None else None
        checkpointed_pages = (
            int(values[2])
            if len(values) > 2 and values[2] is not None
            else None
        )
        fully_checkpointed = (
            busy in {None, 0}
            and (
                log_pages is None
                or checkpointed_pages is None
                or checkpointed_pages >= log_pages
            )
        )
        return {
            "status": "completed" if fully_checkpointed else "deferred",
            "mode": safe_mode,
            "busy": busy,
            "log_pages": log_pages,
            "checkpointed_pages": checkpointed_pages,
            "reason": None if fully_checkpointed else "wal_readers_active",
        }

    def storage_usage(self) -> dict[str, int]:
        """Return filesystem bytes owned by this SQLite database.

        Capacity pressure includes the WAL because it is durable database
        storage even though it lives beside, rather than inside, the main
        SQLite file. The shared-memory coordination file is reported
        separately and excluded from the managed total.
        """

        if self.path == Path(":memory:"):
            return {
                "main_size_bytes": 0,
                "wal_size_bytes": 0,
                "shm_size_bytes": 0,
                "managed_size_bytes": 0,
                "allocated_size_bytes": 0,
                "freelist_space_bytes": 0,
                "used_size_bytes": 0,
            }

        def file_size(path: Path) -> int:
            try:
                return int(path.stat().st_size)
            except FileNotFoundError:
                return 0

        main_size = file_size(self.path)
        wal_size = file_size(Path(f"{self.path}-wal"))
        shm_size = file_size(Path(f"{self.path}-shm"))
        sqlite_space = self.sqlite_space_status()
        freelist_space = min(
            main_size,
            max(0, int(sqlite_space.get("freelist_bytes", 0) or 0)),
        )
        allocated_size = main_size + wal_size
        used_size = max(0, main_size - freelist_space) + wal_size
        return {
            "main_size_bytes": main_size,
            "wal_size_bytes": wal_size,
            "shm_size_bytes": shm_size,
            # managed_size_bytes remains the physical allocation for operators
            # that need to reason about filesystem consumption.
            "managed_size_bytes": allocated_size,
            "allocated_size_bytes": allocated_size,
            # Freelist pages are reusable by SQLite without growing the file.
            # They are physical allocation, not live capacity consumption.
            "freelist_space_bytes": freelist_space,
            "used_size_bytes": used_size,
        }

    def sqlite_space_status(self) -> dict[str, int | str | bool]:
        with self._connection_lock:
            page_size = int(self.conn.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(self.conn.execute("PRAGMA page_count").fetchone()[0])
            freelist_count = int(
                self.conn.execute("PRAGMA freelist_count").fetchone()[0]
            )
            auto_vacuum = int(
                self.conn.execute("PRAGMA auto_vacuum").fetchone()[0]
            )
        auto_vacuum_name = {
            0: "none",
            1: "full",
            2: "incremental",
        }.get(auto_vacuum, f"unknown:{auto_vacuum}")
        return {
            "page_size_bytes": page_size,
            "page_count": page_count,
            "freelist_count": freelist_count,
            "freelist_bytes": freelist_count * page_size,
            "auto_vacuum_mode": auto_vacuum_name,
            "incremental_reclaim_available": auto_vacuum == 2,
            "requires_full_vacuum_for_incremental": auto_vacuum == 0,
        }

    def incremental_vacuum(self, *, max_pages: int = 4096) -> dict[str, Any]:
        max_pages = max(0, int(max_pages))
        with self._connection_lock:
            with self._runtime.writer.acquire("maintenance"):
                auto_vacuum = int(
                    self.conn.execute("PRAGMA auto_vacuum").fetchone()[0]
                )
                if auto_vacuum != 2:
                    return {
                        "status": "skipped",
                        "reason": "incremental_auto_vacuum_not_enabled",
                        "auto_vacuum_mode": auto_vacuum,
                    }
                before_page_count = int(
                    self.conn.execute("PRAGMA page_count").fetchone()[0]
                )
                before_freelist = int(
                    self.conn.execute("PRAGMA freelist_count").fetchone()[0]
                )
                if max_pages > 0 and before_freelist > 0:
                    self.conn.execute(
                        f"PRAGMA incremental_vacuum({min(max_pages, before_freelist)})"
                    )
                after_page_count = int(
                    self.conn.execute("PRAGMA page_count").fetchone()[0]
                )
                after_freelist = int(
                    self.conn.execute("PRAGMA freelist_count").fetchone()[0]
                )
        return {
            "status": "completed",
            "requested_pages": max_pages,
            "page_count_before": before_page_count,
            "page_count_after": after_page_count,
            "freelist_count_before": before_freelist,
            "freelist_count_after": after_freelist,
            "reclaimed_pages": max(0, before_page_count - after_page_count),
        }

    def vacuum(self) -> dict[str, Any]:
        with self._connection_lock:
            with self._runtime.writer.acquire("maintenance"):
                before_page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
                before_freelist = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
                # A full rebuild is the only way to enable incremental
                # auto-vacuum on an existing database without one.
                self.conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                self.conn.execute("VACUUM")
                after_page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
                after_freelist = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
                auto_vacuum = self.conn.execute("PRAGMA auto_vacuum").fetchone()[0]
        return {
            "status": "completed",
            "page_count_before": int(before_page_count),
            "page_count_after": int(after_page_count),
            "freelist_count_before": int(before_freelist),
            "freelist_count_after": int(after_freelist),
            "auto_vacuum_mode": int(auto_vacuum),
        }

    @staticmethod
    def _encode_journal_blob(value: Any) -> tuple[bytes, int, str]:
        raw = canonical_json(value).encode("utf-8")
        return zlib.compress(raw, level=6), len(raw), digest(value)

    @staticmethod
    def _decode_journal_blob(
        blob: bytes,
        *,
        codec: str,
        expected_digest: str,
    ) -> Any:
        if codec != "zlib-json-v1":
            raise ValueError(f"unsupported journal codec: {codec}")
        value = json.loads(zlib.decompress(bytes(blob)).decode("utf-8"))
        if digest(value) != str(expected_digest):
            raise ValueError("journal archive digest mismatch")
        return value

    @staticmethod
    def _validate_exact_journal_events(
        events: Sequence[Mapping[str, Any]],
        *,
        expected_previous: str,
    ) -> str:
        previous = str(expected_previous)
        for event in events:
            if str(event.get("previous_event_hash") or "") != previous:
                raise ValueError("journal event chain mismatch during compaction")
            body = dict(event)
            checksum = str(body.pop("checksum", ""))
            if digest(body) != checksum:
                raise ValueError("journal event checksum mismatch during compaction")
            previous = checksum
        return previous

    @staticmethod
    def _compact_journal_event_payload(
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        if str(event.get("event_type") or "") != "memory_transaction_committed":
            return {"storage_compacted": True}
        payload = event.get("payload")
        payload = dict(payload) if isinstance(payload, Mapping) else {}
        compact_atoms = []
        for raw in payload.get("projected_atoms") or ():
            if not isinstance(raw, Mapping):
                continue
            atom_payload = raw.get("payload")
            atom_payload = atom_payload if isinstance(atom_payload, Mapping) else {}
            compact_atoms.append(
                {
                    "id": str(raw.get("id") or ""),
                    "type": str(raw.get("type") or ""),
                    "profile": str(atom_payload.get("profile") or ""),
                    "payload_digest": digest(atom_payload),
                    "scope": dict(raw.get("scope") or {}),
                    "access_policy": dict(raw.get("access_policy") or {}),
                    "retention_class": str(raw.get("retention_class") or ""),
                    "lifecycle_state": str(raw.get("lifecycle_state") or ""),
                    "health_status": str(raw.get("health_status") or ""),
                }
            )
        compact_evidence = []
        for raw in payload.get("evidence") or ():
            if not isinstance(raw, Mapping):
                continue
            compact_evidence.append(
                {
                    "evidence_id": str(raw.get("evidence_id") or ""),
                    "source_type": str(raw.get("source_type") or ""),
                    "source_ref": str(raw.get("source_ref") or ""),
                    "captured_at": str(raw.get("captured_at") or ""),
                    "checksum": str(raw.get("checksum") or ""),
                    "scope": dict(raw.get("scope") or {}),
                    "access_policy": dict(raw.get("access_policy") or {}),
                }
            )
        return {
            "storage_compacted": True,
            "operation": str(payload.get("operation") or ""),
            "scope": dict(payload.get("scope") or {}),
            "projected_atoms": compact_atoms,
            "projected_heads": [
                dict(raw)
                for raw in payload.get("projected_heads") or ()
                if isinstance(raw, Mapping)
            ],
            "projected_evidence": compact_evidence,
            "receipt_refs": [
                str(ref) for ref in payload.get("receipt_refs") or () if str(ref)
            ],
            "projected_edge_count": len(payload.get("projected_edges") or ()),
        }

    def _journal_event_receipt(
        self,
        event: Mapping[str, Any],
        *,
        segment_id: str,
        created_at: str,
    ) -> dict[str, Any]:
        compact_payload = self._compact_journal_event_payload(event)
        receipt = {
            "event_id": str(event["event_id"]),
            "segment_id": str(segment_id),
            "event_type": str(event["event_type"]),
            "schema_version": str(event["schema_version"]),
            "actor": str(event["actor"]),
            "idempotency_key": event.get("idempotency_key"),
            "payload_digest": str(event["payload_digest"]),
            "occurred_at": str(event["occurred_at"]),
            "accepted_at": str(event["accepted_at"]),
            "result_status": str(event["result_status"]),
            "projection_status": str(event["projection_status"]),
            "previous_event_hash": str(event["previous_event_hash"]),
            "checksum": str(event["checksum"]),
            "graph_version": int(event["graph_version"]),
            "compact_payload": compact_payload,
            "created_at": str(created_at),
        }
        receipt["receipt_digest"] = digest(receipt)
        return receipt

    def _insert_journal_event_receipts(
        self,
        conn: sqlite3.Connection,
        receipts: Sequence[Mapping[str, Any]],
    ) -> None:
        conn.executemany(
            """
            INSERT INTO amos_journal_event_receipts(
                event_id, segment_id, event_type, schema_version, actor,
                idempotency_key, payload_digest, occurred_at, accepted_at,
                result_status, projection_status, previous_event_hash,
                checksum, graph_version, compact_payload, receipt_digest,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO NOTHING
            """,
            [
                (
                    receipt["event_id"],
                    receipt["segment_id"],
                    receipt["event_type"],
                    receipt["schema_version"],
                    receipt["actor"],
                    receipt.get("idempotency_key"),
                    receipt["payload_digest"],
                    receipt["occurred_at"],
                    receipt["accepted_at"],
                    receipt["result_status"],
                    receipt["projection_status"],
                    receipt["previous_event_hash"],
                    receipt["checksum"],
                    int(receipt["graph_version"]),
                    canonical_json(receipt["compact_payload"]),
                    receipt["receipt_digest"],
                    receipt["created_at"],
                )
                for receipt in receipts
            ],
        )

    def latest_journal_snapshot(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM amos_journal_snapshots
            ORDER BY through_graph_version DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["state"] = self._decode_journal_blob(
            data.pop("state_blob"),
            codec=str(data["codec"]),
            expected_digest=str(data["state_digest"]),
        )
        return data

    def list_journal_segments(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT segment_id, schema_version, start_graph_version,
                   end_graph_version, first_event_id, last_event_id,
                   first_previous_event_hash, last_event_hash, event_count,
                   codec, events_digest, uncompressed_bytes, compressed_bytes,
                   payload_retained, payload_pruned_at, created_at
            FROM amos_journal_segments
            ORDER BY start_graph_version ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def journal_storage_status(self) -> dict[str, Any]:
        live = self.conn.execute(
            "SELECT COUNT(*) AS count FROM amos_event_journal"
        ).fetchone()
        segments = self.conn.execute(
            """
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(event_count), 0) AS events,
                   COALESCE(SUM(CASE WHEN payload_retained = 1
                                     THEN event_count ELSE 0 END), 0)
                       AS retained_events,
                   COALESCE(SUM(CASE WHEN payload_retained = 0
                                     THEN event_count ELSE 0 END), 0)
                       AS digest_only_events,
                   COALESCE(SUM(CASE WHEN payload_retained = 1
                                     THEN 1 ELSE 0 END), 0)
                       AS retained_segments,
                   COALESCE(SUM(CASE WHEN payload_retained = 0
                                     THEN 1 ELSE 0 END), 0)
                       AS digest_only_segments,
                   COALESCE(SUM(uncompressed_bytes), 0) AS uncompressed_bytes,
                   COALESCE(SUM(compressed_bytes), 0) AS compressed_bytes,
                   COALESCE(MAX(end_graph_version), 0) AS through_graph_version
            FROM amos_journal_segments
            """
        ).fetchone()
        snapshots = self.conn.execute(
            """
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(compressed_bytes), 0) AS compressed_bytes,
                   COALESCE(MAX(through_graph_version), 0) AS through_graph_version
            FROM amos_journal_snapshots
            """
        ).fetchone()
        receipts = self.conn.execute(
            """
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(length(compact_payload)), 0) AS payload_bytes
            FROM amos_journal_event_receipts
            """
        ).fetchone()
        return {
            "live_event_count": int(live["count"] or 0),
            "compacted_event_count": int(segments["events"] or 0),
            "segment_count": int(segments["count"] or 0),
            "retained_segment_count": int(segments["retained_segments"] or 0),
            "digest_only_segment_count": int(
                segments["digest_only_segments"] or 0
            ),
            "retained_segment_event_count": int(
                segments["retained_events"] or 0
            ),
            "digest_only_event_count": int(
                segments["digest_only_events"] or 0
            ),
            "segment_uncompressed_bytes": int(
                segments["uncompressed_bytes"] or 0
            ),
            "segment_compressed_bytes": int(segments["compressed_bytes"] or 0),
            "snapshot_count": int(snapshots["count"] or 0),
            "snapshot_compressed_bytes": int(
                snapshots["compressed_bytes"] or 0
            ),
            "compacted_receipt_count": int(receipts["count"] or 0),
            "compacted_receipt_payload_bytes": int(
                receipts["payload_bytes"] or 0
            ),
            "through_graph_version": max(
                int(segments["through_graph_version"] or 0),
                int(snapshots["through_graph_version"] or 0),
            ),
        }

    def event_count(self) -> int:
        status = self.journal_storage_status()
        return int(status["live_event_count"]) + int(
            status["compacted_event_count"]
        )

    def _journal_receipt_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        compact_payload = self._json(str(row["compact_payload"]))
        receipt = {
            "event_id": str(row["event_id"]),
            "segment_id": str(row["segment_id"]),
            "event_type": str(row["event_type"]),
            "schema_version": str(row["schema_version"]),
            "actor": str(row["actor"]),
            "idempotency_key": row["idempotency_key"],
            "payload_digest": str(row["payload_digest"]),
            "occurred_at": str(row["occurred_at"]),
            "accepted_at": str(row["accepted_at"]),
            "result_status": str(row["result_status"]),
            "projection_status": str(row["projection_status"]),
            "previous_event_hash": str(row["previous_event_hash"]),
            "checksum": str(row["checksum"]),
            "graph_version": int(row["graph_version"]),
            "compact_payload": compact_payload,
            "created_at": str(row["created_at"]),
        }
        if digest(receipt) != str(row["receipt_digest"]):
            raise ValueError("journal compact receipt digest mismatch")
        return {
            **receipt,
            "payload": compact_payload,
            "target_refs": [],
            "payload_refs": [],
            "evidence_refs": [],
            "causal_parent_ids": [],
            "expected_versions": {},
            "authorization_context": {},
            "storage_compacted": True,
            "compact_receipt_verified": True,
            "compact_receipt_digest": str(row["receipt_digest"]),
        }

    def list_journal_event_receipts(
        self, segment_id: str
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM amos_journal_event_receipts
            WHERE segment_id = ?
            ORDER BY graph_version ASC
            """,
            (str(segment_id),),
        ).fetchall()
        return [self._journal_receipt_from_row(row) for row in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Return a retained event or its verified compact reference receipt."""

        row = self.conn.execute(
            "SELECT * FROM amos_event_journal WHERE event_id = ?",
            (str(event_id),),
        ).fetchone()
        if row is not None:
            return self._row_dict(row)
        segment = self.conn.execute(
            """
            SELECT s.*
            FROM amos_journal_segment_events e
            JOIN amos_journal_segments s ON s.segment_id = e.segment_id
            WHERE e.event_id = ?
              AND s.payload_retained = 1
            """,
            (str(event_id),),
        ).fetchone()
        if segment is None:
            receipt_row = self.conn.execute(
                """
                SELECT * FROM amos_journal_event_receipts
                WHERE event_id = ?
                """,
                (str(event_id),),
            ).fetchone()
            if receipt_row is None:
                return None
            return self._journal_receipt_from_row(receipt_row)
        events = self._decode_journal_blob(
            segment["events_blob"],
            codec=str(segment["codec"]),
            expected_digest=str(segment["events_digest"]),
        )
        return next(
            (dict(event) for event in events if event.get("event_id") == event_id),
            None,
        )

    def journal_segment_events(
        self, segment_id: str
    ) -> list[dict[str, Any]] | None:
        row = self.conn.execute(
            "SELECT * FROM amos_journal_segments WHERE segment_id = ?",
            (str(segment_id),),
        ).fetchone()
        if row is None or not int(row["payload_retained"] or 0):
            return None
        decoded = self._decode_journal_blob(
            row["events_blob"],
            codec=str(row["codec"]),
            expected_digest=str(row["events_digest"]),
        )
        return [dict(event) for event in decoded]

    def list_live_events(
        self,
        *,
        limit: int | None = None,
        after_graph_version: int | None = None,
        newest: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if after_graph_version is not None:
            clauses.append("graph_version > ?")
            params.append(int(after_graph_version))
        query = "SELECT * FROM amos_event_journal"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY graph_version " + ("DESC" if newest else "ASC")
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.conn.execute(query, tuple(params)).fetchall()
        events = [self._row_dict(row) for row in rows]
        if newest:
            events.reverse()
        return events

    def list_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """List exact event bodies still present in retained segments and tail.

        ``event_count`` includes older digest-only segments; those deliberately
        shredded payloads are represented by compact receipts through
        ``get_event`` rather than synthesized as full journal entries here.
        """

        if limit is None:
            events: list[dict[str, Any]] = []
            segments = self.conn.execute(
                """
                SELECT * FROM amos_journal_segments
                ORDER BY start_graph_version ASC
                """
            ).fetchall()
            for segment in segments:
                if not int(segment["payload_retained"] or 0):
                    continue
                decoded = self._decode_journal_blob(
                    segment["events_blob"],
                    codec=str(segment["codec"]),
                    expected_digest=str(segment["events_digest"]),
                )
                events.extend(dict(event) for event in decoded)
            events.extend(self.list_live_events())
            return events

        requested = max(0, int(limit))
        if requested == 0:
            return []
        events = self.list_live_events(limit=requested, newest=True)
        needed = requested - len(events)
        if needed <= 0:
            return events
        segments = self.conn.execute(
            """
            SELECT * FROM amos_journal_segments
            ORDER BY end_graph_version DESC
            """
        ).fetchall()
        archived: list[dict[str, Any]] = []
        for segment in segments:
            if not int(segment["payload_retained"] or 0):
                continue
            decoded = self._decode_journal_blob(
                segment["events_blob"],
                codec=str(segment["codec"]),
                expected_digest=str(segment["events_digest"]),
            )
            selected = [dict(event) for event in decoded[-needed:]]
            archived = selected + archived
            needed -= len(selected)
            if needed <= 0:
                break
        return archived + events

    def compact_journal_segment(
        self,
        *,
        max_events: int = 512,
        min_events: int = 128,
        retain_tail_events: int = 128,
        retain_snapshots: int = 1,
        retain_full_segments: int = 2,
    ) -> dict[str, Any]:
        max_events = max(1, int(max_events))
        min_events = max(1, min(max_events, int(min_events)))
        retain_tail_events = max(0, int(retain_tail_events))
        retain_snapshots = max(1, int(retain_snapshots))
        retain_full_segments = max(0, int(retain_full_segments))
        with self.read_snapshot():
            prior_snapshot = self.latest_journal_snapshot()
            retained_segment_events = {
                str(segment["segment_id"]): self.journal_segment_events(
                    str(segment["segment_id"])
                )
                or []
                for segment in self.list_journal_segments()
                if int(segment.get("payload_retained", 1) or 0)
            }
            for retained_events in retained_segment_events.values():
                if retained_events:
                    self._validate_exact_journal_events(
                        retained_events,
                        expected_previous=str(
                            retained_events[0]["previous_event_hash"]
                        ),
                    )
            prior_version = int(
                (prior_snapshot or {}).get("through_graph_version", 0) or 0
            )
            live_count_row = self.conn.execute(
                "SELECT COUNT(*) AS count FROM amos_event_journal"
            ).fetchone()
            eligible = max(
                0, int(live_count_row["count"] or 0) - retain_tail_events
            )
            if eligible < min_events:
                return {
                    "status": "skipped",
                    "reason": "journal_segment_below_minimum",
                    "compacted_event_count": 0,
                    "eligible_event_count": eligible,
                    "min_events": min_events,
                    "live_event_count": int(live_count_row["count"] or 0),
                    "retain_tail_events": retain_tail_events,
                }
            selected = self.list_live_events(limit=min(max_events, eligible))
            if not selected:
                return {
                    "status": "skipped",
                    "reason": "journal_tail_within_retention",
                    "compacted_event_count": 0,
                    "live_event_count": int(live_count_row["count"] or 0),
                    "retain_tail_events": retain_tail_events,
                }
            expected_start = prior_version + 1
            actual_start = int(selected[0]["graph_version"])
            if actual_start != expected_start:
                return {
                    "status": "error",
                    "reason": "journal_snapshot_tail_gap",
                    "expected_start_graph_version": expected_start,
                    "actual_start_graph_version": actual_start,
                }
            base_state = (
                prior_snapshot["state"]
                if prior_snapshot is not None
                else empty_replay_state()
            )

        expected_previous = (
            str(prior_snapshot["through_event_hash"])
            if prior_snapshot is not None
            else "genesis"
        )
        terminal_hash = self._validate_exact_journal_events(
            selected,
            expected_previous=expected_previous,
        )
        if terminal_hash != str(selected[-1]["checksum"]):
            raise ValueError("journal segment terminal checksum mismatch")

        replayed = replay_events(
            selected,
            initial_state=base_state,
            migrated_edge_derivation=migrated_edge_derivation,
        )
        snapshot_state = serializable_replay_state(replayed)
        events_blob, events_bytes, events_digest = self._encode_journal_blob(
            selected
        )
        state_blob, state_bytes, state_digest = self._encode_journal_blob(
            snapshot_state
        )
        start_version = int(selected[0]["graph_version"])
        end_version = int(selected[-1]["graph_version"])
        segment_id = (
            f"jseg_{start_version}_{end_version}_{events_digest[:12]}"
        )
        snapshot_id = f"jsnap_{end_version}_{state_digest[:12]}"
        created_at = utc_now()
        retained_segment_events[segment_id] = selected
        prepared_receipts_by_segment = {
            retained_segment_id: [
                self._journal_event_receipt(
                    event,
                    segment_id=retained_segment_id,
                    created_at=created_at,
                )
                for event in segment_events
            ]
            for retained_segment_id, segment_events in retained_segment_events.items()
        }
        expected_rows = [
            (str(event["event_id"]), str(event["checksum"])) for event in selected
        ]

        with self.transaction(lane="maintenance") as conn:
            current_snapshot = conn.execute(
                """
                SELECT through_graph_version
                FROM amos_journal_snapshots
                ORDER BY through_graph_version DESC
                LIMIT 1
                """
            ).fetchone()
            current_prior_version = int(
                current_snapshot["through_graph_version"]
                if current_snapshot is not None
                else 0
            )
            if current_prior_version != prior_version:
                return {
                    "status": "stale",
                    "reason": "journal_snapshot_advanced_before_publish",
                    "planned_through_graph_version": prior_version,
                    "current_through_graph_version": current_prior_version,
                }
            current_rows = conn.execute(
                """
                SELECT event_id, checksum
                FROM amos_event_journal
                WHERE graph_version BETWEEN ? AND ?
                ORDER BY graph_version ASC
                """,
                (start_version, end_version),
            ).fetchall()
            if [
                (str(row["event_id"]), str(row["checksum"]))
                for row in current_rows
            ] != expected_rows:
                return {
                    "status": "stale",
                    "reason": "journal_segment_changed_before_publish",
                    "start_graph_version": start_version,
                    "end_graph_version": end_version,
                }
            conn.execute(
                """
                INSERT INTO amos_journal_segments(
                    segment_id, schema_version, start_graph_version,
                    end_graph_version, first_event_id, last_event_id,
                    first_previous_event_hash, last_event_hash, event_count,
                    codec, events_blob, events_digest, uncompressed_bytes,
                    compressed_bytes, payload_retained, payload_pruned_at,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment_id,
                    SCHEMA_VERSION,
                    start_version,
                    end_version,
                    selected[0]["event_id"],
                    selected[-1]["event_id"],
                    selected[0]["previous_event_hash"],
                    selected[-1]["checksum"],
                    len(selected),
                    "zlib-json-v1",
                    sqlite3.Binary(events_blob),
                    events_digest,
                    events_bytes,
                    len(events_blob),
                    1,
                    None,
                    created_at,
                ),
            )
            conn.executemany(
                """
                INSERT INTO amos_journal_segment_events(
                    event_id, segment_id, graph_version
                ) VALUES (?, ?, ?)
                """,
                [
                    (event["event_id"], segment_id, int(event["graph_version"]))
                    for event in selected
                ],
            )
            conn.execute(
                """
                INSERT INTO amos_journal_snapshots(
                    snapshot_id, schema_version, through_graph_version,
                    through_event_id, through_event_hash, codec, state_blob,
                    state_digest, uncompressed_bytes, compressed_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    SCHEMA_VERSION,
                    end_version,
                    selected[-1]["event_id"],
                    selected[-1]["checksum"],
                    "zlib-json-v1",
                    sqlite3.Binary(state_blob),
                    state_digest,
                    state_bytes,
                    len(state_blob),
                    created_at,
                ),
            )
            deleted = conn.execute(
                """
                DELETE FROM amos_event_journal
                WHERE graph_version BETWEEN ? AND ?
                """,
                (start_version, end_version),
            )
            if int(deleted.rowcount or 0) != len(selected):
                raise RuntimeError("journal compaction deleted an unexpected event count")
            conn.execute(
                """
                DELETE FROM amos_journal_snapshots
                WHERE snapshot_id NOT IN (
                    SELECT snapshot_id
                    FROM amos_journal_snapshots
                    ORDER BY through_graph_version DESC
                    LIMIT ?
                )
                """,
                (retain_snapshots,),
            )
            retained = conn.execute(
                """
                SELECT segment_id, compressed_bytes
                FROM amos_journal_segments
                WHERE payload_retained = 1
                ORDER BY end_graph_version DESC
                """
            ).fetchall()
            pruned_segments = retained[retain_full_segments:]
            pruned_segment_ids = [
                str(row["segment_id"]) for row in pruned_segments
            ]
            pruned_payload_bytes = sum(
                int(row["compressed_bytes"] or 0) for row in pruned_segments
            )
            if pruned_segment_ids:
                if any(
                    segment_id not in prepared_receipts_by_segment
                    for segment_id in pruned_segment_ids
                ):
                    raise RuntimeError(
                        "journal segment changed before receipt compaction"
                    )
                self._insert_journal_event_receipts(
                    conn,
                    [
                        receipt
                        for pruned_segment_id in pruned_segment_ids
                        for receipt in prepared_receipts_by_segment[
                            pruned_segment_id
                        ]
                    ],
                )
                placeholders = ",".join("?" for _ in pruned_segment_ids)
                conn.execute(
                    f"""
                    UPDATE amos_journal_segments
                    SET events_blob = X'', compressed_bytes = 0,
                        payload_retained = 0, payload_pruned_at = ?
                    WHERE segment_id IN ({placeholders})
                    """,
                    (created_at, *pruned_segment_ids),
                )
                conn.execute(
                    f"""
                    DELETE FROM amos_journal_segment_events
                    WHERE segment_id IN ({placeholders})
                    """,
                    tuple(pruned_segment_ids),
                )
            self._set_meta(conn, "last_journal_compaction_at", created_at)

        return {
            "status": "completed",
            "segment_id": segment_id,
            "snapshot_id": snapshot_id,
            "start_graph_version": start_version,
            "end_graph_version": end_version,
            "compacted_event_count": len(selected),
            "segment_uncompressed_bytes": events_bytes,
            "segment_compressed_bytes": len(events_blob),
            "snapshot_uncompressed_bytes": state_bytes,
            "snapshot_compressed_bytes": len(state_blob),
            "pruned_segment_count": len(pruned_segment_ids),
            "pruned_segment_payload_bytes": pruned_payload_bytes,
            "saved_bytes_before_sqlite_reclaim": max(
                0,
                events_bytes
                + pruned_payload_bytes
                - len(events_blob)
                - len(state_blob),
            ),
            "min_events": min_events,
            "retain_tail_events": retain_tail_events,
            "retain_snapshots": retain_snapshots,
            "retain_full_segments": retain_full_segments,
        }

    def _row_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key, value in list(data.items()):
            if key in JSON_COLUMNS and isinstance(value, str):
                data[key] = self._json(value)
        return data

    def _json(self, value: str) -> Any:
        return json.loads(value)
