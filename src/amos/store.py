"""SQLite persistence for the AMOS v1-local service implementation."""

from __future__ import annotations

import sqlite3
import uuid
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        if self.path != Path(":memory:"):
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
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
            CREATE INDEX IF NOT EXISTS idx_atoms_deleted_updated
                ON amos_atoms(deleted, updated_at);
            CREATE INDEX IF NOT EXISTS idx_atoms_lifecycle_health_type
                ON amos_atoms(lifecycle_state, health_status, type);

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
            CREATE INDEX IF NOT EXISTS idx_packet_cache_request_graph
                ON amos_packet_cache(request_digest, graph_version);

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
            if self._get_meta(conn, "graph_version") is None:
                self._set_meta(conn, "graph_version", "0")
            if self._get_meta(conn, "last_event_hash") is None:
                self._set_meta(conn, "last_event_hash", "genesis")
            self._backfill_atom_text_index(conn)

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
        if atom.get("deleted"):
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
            return {"status": "skipped", "reason": "no_prune_criteria", "rows": 0}
        cursor = conn.execute(
            f"""
            DELETE FROM amos_atom_text_index
            WHERE atom_id IN (
                SELECT i.atom_id
                FROM amos_atom_text_index i
                JOIN amos_atoms a ON a.id = i.atom_id
                WHERE {' OR '.join(predicates)}
            )
            """,
            tuple(params),
        )
        return {
            "status": "completed",
            "rows": int(cursor.rowcount or 0),
            "lifecycle_states": lifecycle_states,
            "health_statuses": health_statuses,
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

    def list_atoms_filtered(
        self,
        *,
        include_deleted: bool = False,
        types: list[str] | None = None,
        lifecycle_states: list[str] | None = None,
        excluded_health: list[str] | None = None,
        included_health: list[str] | None = None,
        atom_ids: list[str] | None = None,
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
        self, tokens: list[str], *, limit: int | None = None
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
            SELECT atom_id, COUNT(*) AS matches
            FROM amos_atom_text_index
            WHERE token IN ({placeholders})
            GROUP BY atom_id
            ORDER BY matches DESC, atom_id ASC
        """
        params: list[Any] = list(normalized)
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [str(row["atom_id"]) for row in rows]

    def neighbor_atom_ids(self, refs: list[str]) -> list[str]:
        neighbors: set[str] = set()
        for edge in self.list_edges_for_refs(refs):
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
        return self._row_dict(row) if row else None

    def upsert_edge(self, conn: sqlite3.Connection, edge: Mapping[str, Any]) -> None:
        """Project an edge state, including lifecycle reactivation on promotion."""
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

    def list_edges(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM amos_edges WHERE deleted = 0").fetchall()
        return [self._row_dict(row) for row in rows]

    def edge_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM amos_edges WHERE deleted = 0"
        ).fetchone()
        return int(row["count"])

    def edge_degree_counts(self, refs: list[str] | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        if refs is not None:
            ref_set = {str(ref) for ref in refs if str(ref)}
            for edge in self.list_edges_for_refs(sorted(ref_set)):
                source = str(edge["source_ref"])
                target = str(edge["target_ref"])
                if source in ref_set:
                    counts[source] = counts.get(source, 0) + 1
                if target in ref_set:
                    counts[target] = counts.get(target, 0) + 1
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

    def list_edges_for_refs(self, refs: list[str]) -> list[dict[str, Any]]:
        refs = sorted({str(ref) for ref in refs if str(ref)})
        if not refs:
            return []
        placeholders = ",".join("?" for _ in refs)
        rows = self.conn.execute(
            f"""
            SELECT * FROM amos_edges
            WHERE deleted = 0
              AND (source_ref IN ({placeholders}) OR target_ref IN ({placeholders}))
            """,
            tuple(refs + refs),
        ).fetchall()
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

    def checkpoint_wal(self, *, mode: str = "TRUNCATE") -> dict[str, Any]:
        safe_mode = str(mode or "TRUNCATE").upper()
        if safe_mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            safe_mode = "TRUNCATE"
        row = self.conn.execute(f"PRAGMA wal_checkpoint({safe_mode})").fetchone()
        values = list(row) if row is not None else []
        return {
            "status": "completed",
            "mode": safe_mode,
            "busy": int(values[0]) if len(values) > 0 and values[0] is not None else None,
            "log_pages": int(values[1]) if len(values) > 1 and values[1] is not None else None,
            "checkpointed_pages": int(values[2])
            if len(values) > 2 and values[2] is not None
            else None,
        }

    def vacuum(self) -> dict[str, Any]:
        before_page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
        before_freelist = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
        self.conn.execute("VACUUM")
        after_page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
        after_freelist = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
        return {
            "status": "completed",
            "page_count_before": int(before_page_count),
            "page_count_after": int(after_page_count),
            "freelist_count_before": int(before_freelist),
            "freelist_count_after": int(after_freelist),
        }

    def event_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM amos_event_journal"
        ).fetchone()
        return int(row["count"])

    def list_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None:
            rows = self.conn.execute(
                "SELECT * FROM amos_event_journal ORDER BY graph_version ASC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM amos_event_journal
                ORDER BY graph_version DESC
                LIMIT ?
                """,
                (max(0, int(limit)),),
            ).fetchall()
            rows = list(reversed(rows))
        return [self._row_dict(row) for row in rows]

    def _row_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key, value in list(data.items()):
            if key in JSON_COLUMNS and isinstance(value, str):
                data[key] = self._json(value)
        return data

    def _json(self, value: str) -> Any:
        return json.loads(value)
