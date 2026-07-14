# AMOS Roadmap

This document tracks planned work beyond the verified v1-local contract. It is
not a statement of current implementation status. See
[`v1-local-contract.md`](v1-local-contract.md) for checked-in behavior.

## Iteration 1: Design foundation

- Define the Amos concept and its memory-kernel responsibilities.
- Separate evidence, canonical memory, indexes, and rendered views.
- Identify memory types, timescales, and maintenance needs.

## Iteration 2: Canonical schema and atom storage format

Design the minimal logical schema, canonical interchange encoding, and first physical storage mapping for:

```text
MemoryAtom envelope
typed atom payloads
Entity
SourceEvent
BeliefAtom
PreferenceAtom
Goal
Commitment
ProcedureAtom
Episode
EvidenceRecord
AssociationEdge
EventJournalEntry
```

Deliverables:

```text
strict JSON-compatible schemas for each atom type
identifier conventions for entities, relations, concepts, and enums
common envelope versus payload field boundaries
example canonical records
MVP table or document mapping
serialization and versioning rules
```

## Iteration 3: Lifecycle and maintenance model

Specify:

```text
capture
atomize
normalize
reconcile
link
promote
demote
archive
forget
repair
```

## Iteration 4: Retrieval and rendering contracts

Define the MemoryPacket interface for:

```text
reasoner
planner
executor
critic
steward
```

## Iteration 5: Quality and health metrics

Define metrics, maintenance thresholds, and audit requirements.

## Iteration 6: Scenario pressure tests

Pressure-test against:

```text
personal assistant over five years
enterprise agent over millions of tasks
coding agent learning repository conventions
research agent maintaining hypotheses
multi-agent operations team
```

## Iteration 7: Distributed service model

Specify:

```text
shared Amos instance contract
client identities and capabilities
EventJournalEntry write path
concurrency and consistency model
pub/sub and cache invalidation
scope isolation
conflict resolution
```

## Iteration 8: Journal, compaction, and retention model

Specify:

```text
Event Journal format
snapshot/checkpoint strategy
segment compaction
provenance rollups
telemetry aggregation
edge pruning
forgetting, deletion, tombstones, crypto-shredding
```

## Iteration 9: Capacity governance

Specify:

```text
capacity contract
capacity governor
watermarks and pressure modes
admin capacity extension requests
survival policy under delayed/denied expansion
retention classes
scoped budgets
agent shielding
```

## Iteration 10: Non-LLM semantic maintenance

Specify:

```text
Semantic Maintenance Processor interface
shape validation
embedding/classifier/graph processor roles
reason codes
maintenance decision ladder
optional LLM escalation policy
```

## Iteration 11: Implementation planning

The v1-local repository now includes the first implementation slice. Remaining
planning should track gaps between the verified SQLite service profile and later
production deployment targets:

```text
SQLite service migration sequence
future Postgres migration sequence
JSON Schema artifact layout
generated validator/client layout
indexing strategy
service API surface
maintenance scheduler
capacity governor
semantic maintenance processor
retrieval/ranking experiments
external processor-pack packaging
multi-instance/Postgres operational plan
```

---
