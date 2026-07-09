-- AMOS v1 local/test SQLite schema.
-- The Python store applies the same shape at runtime; this file is the durable
-- migration artifact for inspection and external bootstrap.

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
CREATE INDEX IF NOT EXISTS idx_event_graph_version ON amos_event_journal(graph_version);

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

CREATE TABLE IF NOT EXISTS amos_token_latent_vectors (
    token TEXT PRIMARY KEY,
    graph_version INTEGER NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_token_latent_vectors_graph
    ON amos_token_latent_vectors(graph_version);
