"""SQLite persistence for the AMOS v1-local service implementation."""

from __future__ import annotations

import sqlite3
import uuid
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .schemas import SCHEMA_VERSION, canonical_json, digest, utc_now


JSON_COLUMNS = {
    "access_policy",
    "authorization_context",
    "causal_parent_ids",
    "confidence",
    "decay_policy",
    "evidence_refs",
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
    "details_json",
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
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

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

            CREATE TABLE IF NOT EXISTS amos_edges (
                edge_id TEXT PRIMARY KEY,
                source_ref TEXT NOT NULL,
                target_ref TEXT NOT NULL,
                relation TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                scope TEXT NOT NULL,
                confidence TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                health_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON amos_edges(source_ref);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON amos_edges(target_ref);

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
            """
        )
        with self.transaction() as conn:
            if self._get_meta(conn, "graph_version") is None:
                self._set_meta(conn, "graph_version", "0")
            if self._get_meta(conn, "last_event_hash") is None:
                self._set_meta(conn, "last_event_hash", "genesis")

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

    def get_meta(self, key: str) -> str | None:
        return self._get_meta(self.conn, key)

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            self._set_meta(conn, key, value)

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

    def get_atom(self, atom_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM amos_atoms WHERE id = ?", (atom_id,)).fetchone()
        return None if row is None else self._row_dict(row)

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

    def list_atoms(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM amos_atoms ORDER BY updated_at DESC").fetchall()
        return [self._row_dict(row) for row in rows]

    def insert_edge(self, conn: sqlite3.Connection, edge: Mapping[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO amos_edges(
                edge_id, source_ref, target_ref, relation, schema_version,
                evidence_refs, scope, confidence, lifecycle_state, health_status,
                created_at, updated_at, version, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                edge["lifecycle_state"],
                edge["health_status"],
                edge["created_at"],
                edge["updated_at"],
                edge["version"],
                1 if edge.get("deleted") else 0,
            ),
        )

    def list_edges(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM amos_edges WHERE deleted = 0").fetchall()
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

    def clear_packet_cache(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM amos_packet_cache")

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

    def list_retrieval_outcomes(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM amos_retrieval_outcomes ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_dict(row) for row in rows]

    def list_events(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM amos_event_journal ORDER BY graph_version ASC"
        ).fetchall()
        return [self._row_dict(row) for row in rows]

    def _row_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key, value in list(data.items()):
            if key in JSON_COLUMNS and isinstance(value, str):
                data[key] = self._json(value)
        return data

    def _json(self, value: str) -> Any:
        return json.loads(value)
