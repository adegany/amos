# AMOS V1-Local Verification Matrix

This document maps the V1 implementation baseline and acceptance gates from
`docs/design-spec.md` to repository artifacts and tests for the first usable
deployment profile: an AMOS HTTP service with one in-process SQLite store. The
Postgres DDL is included as the target migration contract, but the verified
runtime backend for this repository state is SQLite behind the service boundary.

## Artifact Coverage

| Spec area | Evidence |
| --- | --- |
| JSON Schema 2020-12 artifacts | `schemas/*.schema.json` |
| Atom envelope and payload boundary | `src/amos/schemas.py`, `tests/test_amos_v1.py::test_schema_rejects_payload_envelope_duplication` |
| Typed payload schemas and runtime validation | `schemas/core_payloads.schema.json`, `schemas/self_awareness_atoms.schema.json`, `schemas/agentic_recall.schema.json`, `src/amos/schemas.py::validate_atom_payload`, `tests/test_amos_v1.py::test_runtime_enforces_typed_payload_contracts` |
| Evidence, edge, packet, event, access, scope, deletion, capacity, SMP schemas | `schemas/` |
| SQLite service migration | `migrations/sqlite/001_initial.sql` |
| Postgres target migration contract | `migrations/postgres/001_initial.sql` |
| Canonical store and journal | `src/amos/store.py` |
| Service API surface | `src/amos/service.py` |
| HTTP V1 API adapter with service-owned store | `src/amos/http_api.py` |
| CLI | `src/amos/cli.py` |
| Seed ontology governance | `src/amos/ontology.py` |
| Worker artifacts | `src/amos/workers.py` |
| Non-generative SMP | `src/amos/smp.py` |
| Automatic memory policy | `src/amos/service.py`, `src/amos/workers.py`, `src/amos/http_api.py`, `src/amos/cli.py`, `tests/test_amos_v1.py::test_automatic_memory_policy_distills_and_maintains_on_retrieval`, `tests/test_amos_v1.py::test_health_memory_can_skip_foreground_policy_tick`, `tests/test_amos_v1.py::test_background_memory_policy_worker_runs_queued_tick`, `tests/test_amos_v1.py::test_memory_policy_worker_force_runs_without_manual_maintenance` |
| Generic maintenance distiller and external processor packs | `src/amos/maintenance.py`, `src/amos/service.py`, `src/amos/workers.py`, `tests/test_amos_v1.py::test_external_processor_distills_supported_control_lesson`, `tests/test_amos_v1.py::test_external_processor_defers_sanitized_control_claim`, `tests/test_amos_v1.py::test_external_processor_import_path_loading` |
| Mirror Agent integration demo spec | `docs/mirror-agent-demo-spec.md` |
| Mirror Agent integration demo | `examples/mirror_agent_demo.py` |
| Mirror Agent browser UI and LM-backed chat adapter | `examples/mirror_agent_ui.py`, `tests/test_mirror_agent_demo.py::test_mirror_agent_ui_serves_report_chat_and_non_llm_maintenance` |

## Acceptance Gates

| Gate | Current verification evidence |
| --- | --- |
| Schema gate | Payload envelope duplication rejection test; runtime typed payload validation; SMP shape validation for advisory maintenance reports |
| Journal gate | Event entries include idempotency, authorization context, expected versions, checksum, projection status |
| Projection gate | Mutations append events and project graph changes in one transaction |
| Replay gate | `verify_journal_chain`, `verify_replay`, replay/cache invalidation tests |
| Retrieval gate | Packet graph version, provenance, omissions, degradation, score components, budgets |
| Self-awareness gate | Capability suppression, runtime state, limitations, open commitments, calibration tests |
| Agentic recall gate | Success/failure/blocked/correction/limitation/external constraint, self/other/shared/external/unknown attribution, counterevidence, self-narrative drift tests |
| Shared-memory gate | Shared common graph version with per-processor overlays, identity-specific omissions, and least-common-denominator evidence in common items |
| Authorization gate | Read/evidence access filtering and mutation trust/capability tests |
| Deletion gate | Atom deletion, edge suppression, packet cache purge, tombstone content prevention, residual-retention report |
| Capacity gate | Configured pressure modes and degraded packet disclosure |
| SMP gate | Required SMP output envelope and review-required high-risk recommendations |
| Memory policy gate | Background worker ticks and explicit operator runs perform deterministic distillation, SMP/steward maintenance, processor-pack distillation, derived-index refresh, packet-cache invalidation, persisted policy state, and `memory_policy_run` journal events; HTTP health remains observational |
| Processor-pack policy gate | Externally registered processors emit side-effect-free proposals; supported low-risk add-atom lessons commit as derived semantic atoms; sanitized/confounded claims are deferred with draft-only reviewer status |
| Observability gate | Memory/capacity health, background policy worker status, projection lag, index freshness, retrieval outcomes, deletion residuals |
| Procedure policy | Advisory default, autonomous execution denied, external executor eligibility only after approvals |
| LLM reviewer default | Disabled by default; forbidden actions exposed by policy |

## Verification Commands

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

The HTTP endpoint smoke test is included in the pytest suite. In sandboxes that
forbid loopback sockets it is skipped with an explicit reason; when sockets are
available it verifies the real HTTP adapter.
