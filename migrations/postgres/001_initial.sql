-- AMOS v1 Postgres target schema.
-- This DDL mirrors the local SQLite implementation while using JSONB columns
-- for canonical records and indexes appropriate for the v1 storage target.

CREATE TABLE IF NOT EXISTS amos_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS amos_evidence (
    evidence_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    payload JSONB NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    checksum TEXT NOT NULL,
    access_policy JSONB NOT NULL,
    scope JSONB NOT NULL,
    event_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_amos_evidence_scope ON amos_evidence USING GIN(scope);

CREATE TABLE IF NOT EXISTS amos_atoms (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    payload JSONB NOT NULL,
    evidence_refs JSONB NOT NULL,
    scope JSONB NOT NULL,
    confidence JSONB NOT NULL,
    salience DOUBLE PRECISION NOT NULL,
    utility DOUBLE PRECISION NOT NULL,
    layer TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL,
    health_status TEXT NOT NULL,
    retention_class TEXT NOT NULL,
    access_policy JSONB NOT NULL,
    decay_policy JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    observed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL,
    last_accessed TIMESTAMPTZ,
    version BIGINT NOT NULL,
    supersedes JSONB NOT NULL,
    revision_history JSONB NOT NULL,
    index_refs JSONB NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_amos_atoms_type ON amos_atoms(type);
CREATE INDEX IF NOT EXISTS idx_amos_atoms_lifecycle ON amos_atoms(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_amos_atoms_health ON amos_atoms(health_status);
CREATE INDEX IF NOT EXISTS idx_amos_atoms_scope ON amos_atoms USING GIN(scope);

CREATE TABLE IF NOT EXISTS amos_edges (
    edge_id TEXT PRIMARY KEY,
    source_ref TEXT NOT NULL,
    target_ref TEXT NOT NULL,
    relation TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    evidence_refs JSONB NOT NULL,
    scope JSONB NOT NULL,
    confidence JSONB NOT NULL,
    lifecycle_state TEXT NOT NULL,
    health_status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    version BIGINT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_amos_edges_source ON amos_edges(source_ref);
CREATE INDEX IF NOT EXISTS idx_amos_edges_target ON amos_edges(target_ref);
CREATE INDEX IF NOT EXISTS idx_amos_edges_scope ON amos_edges USING GIN(scope);

CREATE TABLE IF NOT EXISTS amos_tombstones (
    tombstone_id TEXT PRIMARY KEY,
    target_ref TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    recreation_policy TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_amos_tombstones_target ON amos_tombstones(target_ref);
CREATE INDEX IF NOT EXISTS idx_amos_tombstones_content ON amos_tombstones(content_digest);

CREATE TABLE IF NOT EXISTS amos_event_journal (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    actor TEXT NOT NULL,
    target_refs JSONB NOT NULL,
    payload JSONB NOT NULL,
    payload_refs JSONB NOT NULL,
    evidence_refs JSONB NOT NULL,
    idempotency_key TEXT,
    payload_digest TEXT NOT NULL,
    causal_parent_ids JSONB NOT NULL,
    expected_versions JSONB NOT NULL,
    authorization_context JSONB NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    accepted_at TIMESTAMPTZ NOT NULL,
    result_status TEXT NOT NULL,
    projection_status TEXT NOT NULL,
    previous_event_hash TEXT NOT NULL,
    checksum TEXT NOT NULL,
    graph_version BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_amos_event_graph_version ON amos_event_journal(graph_version);

CREATE TABLE IF NOT EXISTS amos_idempotency (
    actor TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    event_id TEXT NOT NULL,
    response_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(actor, idempotency_key)
);

CREATE TABLE IF NOT EXISTS amos_packet_cache (
    packet_id TEXT PRIMARY KEY,
    request_digest TEXT NOT NULL,
    graph_version BIGINT NOT NULL,
    request_json JSONB NOT NULL,
    response_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS amos_retrieval_outcomes (
    outcome_id TEXT PRIMARY KEY,
    packet_id TEXT NOT NULL,
    request_json JSONB NOT NULL,
    outcome_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS amos_derived_index_metadata (
    index_name TEXT PRIMARY KEY,
    graph_version BIGINT NOT NULL,
    freshness TEXT NOT NULL,
    rebuilt_at TIMESTAMPTZ NOT NULL,
    details_json JSONB NOT NULL
);
