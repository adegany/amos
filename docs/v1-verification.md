# AMOS V1-Local Verification Matrix

This document maps the verified implementation boundary in
[`v1-local-contract.md`](v1-local-contract.md) to code and tests. The checked-in
runtime profile is one AMOS HTTP service process with one service-owned SQLite
store. Postgres DDL is a future migration contract, not a verified backend.

Status terms:

- **Implemented:** checked-in behavior with automated verification.
- **Partial:** only the stated v1-local subset is implemented.
- **Evidence only:** reproducible evidence exists, but no automated acceptance
  threshold is enforced.

## Architecture and artifact coverage

| Area | Implementation | Primary tests |
| --- | --- | --- |
| Public API composition | `src/amos/service.py` thin `Amos` facade | `tests/test_architecture.py`, plus the entire suite through the stable facade |
| Mutations and lifecycle | `src/amos/mutations_service.py`, `src/amos/access_service.py`, `src/amos/graph_service.py` | `tests/test_schema_and_mutations.py` |
| Retrieval, attention, and ranking | `src/amos/retrieval_service.py`, `src/amos/index_service.py` | `tests/test_retrieval.py` |
| Coherent reasoning frames and demand pages | `src/amos/reasoning_service.py` | `tests/test_reasoning_frames.py`, `tests/test_mirror_agent_demo.py` |
| Self-model and shared views | `src/amos/views_service.py` | `tests/test_self_models.py` |
| Stewardship and SMP | `src/amos/stewardship_service.py`, `src/amos/maintenance.py`, `src/amos/smp.py` | `tests/test_maintenance.py` |
| Automatic policy and capacity | `src/amos/policy_service.py`, `src/amos/capacity_service.py`, `src/amos/workers.py` | `tests/test_policy_and_capacity.py` |
| External processor packs | `src/amos/maintenance.py`, `src/amos/stewardship_service.py` | `tests/test_processor_packs.py` |
| Processor worksets and hierarchical evidence coverage | `src/amos/maintenance.py`, `src/amos/stewardship_service.py` | `tests/test_processor_packs.py` |
| Edge derivation and legacy migration | `src/amos/graph_service.py`, `src/amos/store.py` | `tests/test_maintenance.py` |
| Graph/proposal/processor quality diagnostics | `src/amos/policy_service.py` | `tests/test_policy_and_capacity.py` |
| Journal and health verification | `src/amos/diagnostics_service.py`, `src/amos/store.py` | `tests/test_schema_and_mutations.py`, `tests/test_policy_and_capacity.py` |
| HTTP and CLI adapters | `src/amos/http_api.py`, `src/amos/cli.py` | `tests/test_cli_http.py` |
| Mirror Agent reference integration | `examples/mirror_agent_demo.py`, `examples/mirror_agent_ui.py` | `tests/test_mirror_agent_demo.py` (packets, frames/pages, exact lookup, canonical graph metadata, governed proposals, truthful feedback, browser endpoints) |
| JSON Schema artifacts | `schemas/*.schema.json`, dependency-free runtime mirror in `src/amos/schemas.py` | `tests/test_schema_and_mutations.py::test_runtime_enforces_typed_payload_contracts`, `tests/test_schema_and_mutations.py::test_runtime_enforces_json_schema_property_types_and_score_bounds` |
| SQLite migration | `migrations/sqlite/001_initial.sql` | Migration smoke command below |
| Postgres target contract | `migrations/postgres/001_initial.sql` | Artifact only; runtime verification is intentionally absent |

## Acceptance status

| Gate | Status | Current verification evidence and boundary |
| --- | --- | --- |
| Schema | Implemented | Envelope/payload separation, required fields, property types, enum constraints, and score bounds are tested. |
| Journal | Implemented | Events include authorization context, expected versions, checksums, and projection status. |
| Projection | Implemented | Canonical mutations append events and project graph changes transactionally. |
| Replay | Partial | `DiagnosticsService.verify_replay()` rebuilds from the full retained journal. Snapshot-plus-tail recovery and segment compaction are not implemented. |
| Retrieval and attention | Implemented | Graph-version cache keys, token candidate indexing, semantic fallback, scoped edge activation, score components, omissions, provenance, degradation, and attention traces are tested. |
| Self-awareness and agentic recall | Implemented | Runtime capability suppression, commitments, calibration, responsibility attribution, counterevidence, and self-narrative expiry are tested. |
| Shared memory | Implemented | Common graph-version views plus identity-specific overlays and evidence omissions are tested. |
| Authorization | Implemented | Scope, visibility, mutation roles, trust levels, capabilities, and evidence visibility are independently exercised. |
| Deletion | Partial | Atom/edge suppression, hot index/cache cleanup, tombstones, replay exclusion, and residual-retention disclosure are tested. External evidence archives, snapshots, key management, and backups are not owned by v1-local. |
| Capacity | Partial | One SQLite-file byte budget produces one pressure mode and packet degradation. Per-tier and external-store budgets are planned. |
| SMP and processor packs | Implemented | Deterministic proposal envelopes, review gates, low-risk auto-commit policy, and external processor loading are tested. |
| Memory policy | Implemented | Background and forced deterministic maintenance, decay, cleanup, distillation, index refresh, cache invalidation, and journal summaries are tested. |
| Performance | Evidence only | `benchmarks/benchmark_amos.py` is reproducible, but CI enforces no scale or latency threshold. |
| Observability | Implemented with declared constants | Health, capacity, index freshness, retrieval outcomes, and deletion residuals are reportable. Projection lag is fixed at zero for the transactional single-process profile. |
| Procedure policy | Implemented | Procedures are advisory; autonomous execution is denied by default. |
| LLM reviewer default | Implemented | Disabled and non-authoritative by default. |
| Durable-agent identity | Documentation contract | Agent identity is distinct from processor/model identity; dedicated runtime enforcement remains integration work. |

## Verification commands

```bash
python -m compileall src
python -m pytest -q
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 memory-policy
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("schemas").glob("*.json")):
    json.loads(path.read_text())
print("schemas ok")
PY
python - <<'PY'
import sqlite3
from pathlib import Path
conn = sqlite3.connect(":memory:")
conn.executescript(Path("migrations/sqlite/001_initial.sql").read_text())
for table in [
    "amos_event_journal",
    "amos_atoms",
    "amos_edges",
    "amos_evidence",
    "amos_tombstones",
    "amos_packet_cache",
    "amos_derived_index_metadata",
]:
    conn.execute(f"SELECT 1 FROM {table} LIMIT 0")
print("sqlite migration ok")
PY
```

The HTTP endpoint tests skip explicitly when a sandbox forbids loopback
sockets. Outside such sandboxes they exercise the real stdlib HTTP adapter.
