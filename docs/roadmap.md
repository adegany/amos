# AMOS Roadmap

This document tracks work beyond the verified v1-local profile. Checked-in
behavior and partial gates are defined by the
[`v1-local contract`](v1-local-contract.md) and mapped to evidence in the
[`verification matrix`](v1-verification.md). The
[`design spec`](design-spec.md) remains the long-term architecture.

## Shipped in v1-local

The original design and implementation iterations are now represented in the
checked-in service:

- Canonical typed atoms, evidence, associative edges, schemas, lifecycle
  operations, access policy, and an append-only checksum journal.
- A thin public facade over explicit mutation, retrieval, reasoning, view,
  graph, indexing, stewardship, policy, capacity, and diagnostic services.
- Exact atom lookup, associative packets, attention traces, retrieval feedback,
  self-awareness, agentic recall, and shared views.
- Revision-bound coherent reasoning frames and trusted demand-loaded pages.
- Deterministic distillation, SMP analysis, canonical `semantic_facets` and
  `graph_relations`, processor packs, governed proposals, and review gates.
- Background policy scheduling, capacity pressure, retention, storage cleanup,
  SQLite compaction, index refresh, and cache invalidation.
- A dependency-free HTTP adapter, CLI, Mirror Agent reference integration,
  schemas, migrations, tests, and a reproducible local benchmark.

## Current hardening

- Extend scenario and scale pressure tests beyond the current local benchmark.
- Establish CI-enforced latency, throughput, storage, and reasoning-budget
  acceptance thresholds.
- Continue retrieval, ranking, coherent-unit, and maintenance-policy evaluation
  against larger and more varied agent histories.
- Strengthen operational documentation for backup, restore, observability, and
  production deployment.

## Beyond v1-local

### Production storage and concurrency

- Implement and verify the Postgres runtime adapter described by the migration
  contract.
- Add a multi-process concurrency and consistency model, service-level cache
  invalidation, and production connection management.
- Define supported backup, restore, encryption-key, and external evidence-object
  ownership contracts.

### Journal recovery and retention

- Add canonical graph snapshots and snapshot-plus-tail recovery.
- Add journal segment compaction, checkpoint verification, and provenance
  rollups without weakening replay auditability.
- Define enforcement boundaries for external archives, backups, and
  crypto-shredding.

### Capacity and scale

- Replace the single SQLite-file budget with per-tier and external-store
  capacity accounting.
- Add scoped budgets, production watermarks, expansion requests, and
  multi-tenant shielding.
- Establish production-scale latency and storage acceptance gates.

### Distributed service model

- Specify multi-instance coordination, client capabilities, pub/sub, and
  consistency guarantees.
- Preserve scope isolation and deterministic conflict handling across replicas.
- Keep authoritative application control outside AMOS while exposing auditable
  memory and proposal contracts to integrations.
