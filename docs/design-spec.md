# Amos Design Spec

**Project name:** Amos
**Expansion:** Agent Memory Operating System
**Status:** Long-term architecture and design intent
**Verified implementation contract:** [`v1-local-contract.md`](v1-local-contract.md)
**Implementation evidence:** [`v1-verification.md`](v1-verification.md)
**Roadmap:** [`roadmap.md`](roadmap.md)

---

## 1. Purpose

This design spec captures the long-term architecture for **Amos**, an Agent
Memory Operating System for agentic AI systems that must operate at long time
horizons and large scale without relying on textual English summaries as the
canonical long-term memory format.

Amos is intended to be a model-neutral, layered, associative, self-maintaining belief and memory substrate that can serve LLM reasoners, planners, executors, critics, symbolic systems, retrieval systems, and future processors through generated views.

This document is intentionally aspirational where it uses `should`, `may`, or
future deployment language. It does not assert that every described capability
exists in the checked-in runtime. Current behavior, partial gates, and known
limits are maintained in the separate v1-local contract so architecture intent
cannot silently become an implementation claim.

---

## 2. Problem statement

Current agentic memory systems commonly store long-term memory as English text: conversation summaries, notes, RAG chunks, prompt fragments, behavioral takeaways, and vector-search documents. This is easy to prototype but has serious limitations for long-running agents:

- Text is verbose and expensive to store, retrieve, and reprocess.
- Summaries drift over time when repeatedly rewritten.
- Cleanup requires expensive redistillation.
- Contradictions, stale claims, and overgeneralizations are hard to isolate.
- Prompt-ready memory is tied to one class of processor: the LLM reasoner.
- Embeddings are model-specific and should not be treated as ground truth.
- Procedural memory stored as prose is difficult to validate, version, or roll back.
- Associative relevance across memory categories is weak when memory is stored as isolated chunks.

The goal is to design a memory system where English is a generated view, not the canonical database.

---

## 3. Name and baseline reference

**Amos** stands for **Agent Memory Operating System**. The name refers to the overall agent-memory operating layer, not a single database, vector index, prompt format, or model-specific memory implementation.

Amos treats memory as an operating-system-like service for agentic AI: it manages capture, consolidation, retrieval, promotion, demotion, cleanup, provenance, permissions, and processor-specific rendering.

### 3.1 Baseline reference: `ALucek/agentic-memory`

The repository at <https://github.com/ALucek/agentic-memory> is a useful educational baseline. It models four memory types:

- Working memory: current conversation and immediate context.
- Episodic memory: historical experiences and takeaways.
- Semantic memory: knowledge context and factual grounding.
- Procedural memory: rules and skills for interaction.

Its implementation uses a simple RAG-oriented structure:

```text
working memory    = message history
episodic memory   = stored conversations + reflections
semantic memory   = retrieved document chunks
procedural memory = generated behavioral guidelines
```

This taxonomy is valuable. The storage representation is not the target end state. Amos keeps the memory categories but replaces text-first storage with a typed, provenance-bearing, self-maintaining memory substrate.

---

## 4. Core thesis

Agent memory should behave less like a document store and more like a layered, associative, self-consolidating belief system.

The canonical store should contain:

```text
typed atoms
entities
events
beliefs
preferences
goals
commitments
procedures
self-models
capabilities
limitations
runtime-state snapshots
episodes
association edges
evidence references
health states
promotion/demotion metadata
maintenance history
```

Generated artifacts should include:

```text
English summaries
prompt snippets
embeddings
planner state
executor instructions
self-awareness packets
self-report explanations
shared memory views
agentic recall packets
self-narratives
graph neighborhoods
memory packets
```

Only the canonical substrate is authoritative. Generated artifacts are disposable caches.

---

## 5. Architectural overview

```text
┌────────────────────────────────────────────────────────────┐
│ Runtime agent                                               │
│ reasoner / planner / executor / critic                     │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Memory Packet Interface                                    │
│ task-specific retrieval and processor-specific rendering   │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Canonical Memory Graph                                     │
│ atoms, edges, evidence links, lifecycle state              │
└────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌────────────────────────────────────────────────────────────┐
│ Memory Steward                                             │
│ distill, link, promote, demote, repair, audit, compact     │
└────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌────────────────────────────────────────────────────────────┐
│ Evidence Archive                                           │
│ raw episodes, transcripts, tool traces, source documents   │
└────────────────────────────────────────────────────────────┘
```

Within Amos, the Memory Steward maintains the Canonical Memory Graph while preserving links back to the Evidence Archive.

---

## 6. Three design axes

Memory is organized across three orthogonal axes.

### 6.1 Timescale axis

```text
Immediate memory
  raw current percepts, current user turn, current tool output, transient observations

Working memory
  active task state, current plan, constraints, open questions, active entities

Session memory
  accumulated state across the current interaction or task episode

Recent episodic memory
  high-fidelity compressed traces from recent sessions and tasks

Consolidated long-term memory
  stable beliefs, preferences, semantic knowledge, procedures, project models

Cold evidence archive
  raw transcripts, documents, traces, files, logs, screenshots, tool results
```

Each layer has different capacity, mutation, retrieval, and retention rules.

### 6.2 Functional category axis

```text
working state
episodic memory
semantic memory
procedural memory
belief memory
preference memory
goal memory
commitment memory
self-model memory
capability/limitation memory
runtime-state memory
agentic recall memory
action/outcome memory
policy/constraint memory
```

Functional categories are not fixed to one timescale. For example, procedural memory can be a recent success pattern, a candidate procedure, or a consolidated long-term procedure.

### 6.3 Associative axis

All memory objects can be connected through typed associative edges:

```text
temporal
entity co-reference
semantic similarity
causal influence
goal relevance
procedural trigger
contradiction/tension
abstraction/support
affect/salience/priority
retrieval co-activation
inhibition/exception
```

Associative links enable cross-category relevance and retrieval.

---

## 7. Core objects

### 7.1 MemoryAtom

A `MemoryAtom` is the smallest durable, addressable unit of Amos memory. It is not a paragraph, a prompt snippet, or an embedding. It is a schema-bound record that captures one useful memory claim, preference, goal, episode, procedure, or other memory object with evidence, scope, confidence, lifecycle state, and links.

Conceptually, a typed atom has this shape:

```text
Atom envelope
  common metadata used by Amos across all atom types

Typed payload
  schema-specific content for belief, preference, goal, procedure, episode, etc.

Provenance
  evidence references and confidence basis

Lifecycle metadata
  layer, lifecycle state, health, retention, version, timestamps

Index references
  references to derived embeddings, keyword indexes, graph nodes, packet caches
```

The `type` field determines the schema of `payload`. For example, an atom with `type = belief` must carry a `BeliefAtom` payload, while an atom with `type = preference` must carry a `PreferenceAtom` payload.

```text
MemoryAtom
  id
  type
  schema_version
  payload
  evidence_refs
  scope
  confidence
  salience
  utility
  layer
  lifecycle_state
  health_status
  retention_class
  access_policy
  created_at
  observed_at
  updated_at
  last_accessed
  decay_policy
  version
  supersedes
  revision_history
  index_refs
```

Specialized atom types inherit this envelope and define their own payload schemas.

#### 7.1.1 Canonical stored shape

The recommended canonical interchange shape is JSON-compatible, even if a production implementation later stores the same logical records as JSONB, CBOR, MessagePack, Avro, Protobuf, RDF/quads, or columnar snapshot files.

A stored atom should use an envelope-plus-payload format:

```json
{
  "id": "atm_01JZAMOS7Q6K2Q9E8F3F8Z2R1A",
  "type": "preference",
  "schema_version": "amos.v1",
  "payload": {
    "holder": "ent:user:primary",
    "polarity": "prefers",
    "target": "ent:interaction_style:iterative_conceptual_design",
    "applicability_scope": {
      "task_type": "architecture_design",
      "project": "ent:project:amos",
      "phase": "early_design"
    },
    "strength": "high",
    "exceptions": [
      "when_user_explicitly_requests_code"
    ]
  },
  "confidence": {
    "level": "high",
    "basis": {
      "source_type": "direct_user_instruction",
      "explicitness": "explicit",
      "recency": "current_session",
      "contradiction_count": 0
    }
  },
  "evidence_refs": [
    "evd_01JZAMOS3F7N6P7C4M7Q9T2X0B"
  ],
  "scope": {
    "tenant": "default",
    "workspace": "amos",
    "project": "ent:project:amos",
    "user": "ent:user:primary"
  },
  "layer": "consolidated_long_term",
  "lifecycle_state": "active",
  "health_status": "healthy",
  "retention_class": "project",
  "access_policy": {
    "tenant": "default",
    "workspace": "amos",
    "visibility": ["reasoner", "planner", "steward"]
  },
  "created_at": "<iso_timestamp>",
  "observed_at": "<iso_timestamp>",
  "updated_at": "<iso_timestamp>",
  "version": 1,
  "index_refs": {
    "embedding": ["emb_01JZAMOS9V2C1M4"],
    "keyword": ["idx_kw_91"],
    "graph_node": "node_atm_01JZAMOS7Q6K2Q9E8F3F8Z2R1A"
  }
}
```

This shape keeps the canonical memory compact and structured while still allowing Amos to render processor-specific English, JSON, graph neighborhoods, planner state, or executor context on demand.

#### 7.1.2 Belief atom example

A design belief such as “Amos should shield connected agents from capacity management concerns” should not be stored as the canonical English sentence. It should be stored as structured data:

```json
{
  "id": "atm_01JZB0Z8WX4R7AN7M7P5XDK9Q4",
  "type": "belief",
  "schema_version": "amos.v1",
  "payload": {
    "subject": "ent:system:amos",
    "relation": "rel:responsible_for",
    "object": "ent:concern:capacity_management",
    "qualifiers": {
      "agent_services_should_be_shielded": true,
      "admin_may_receive_capacity_requests": true,
      "capacity_pressure_handled_internally": true
    },
    "modality": "design_decision"
  },
  "confidence": {
    "level": "high",
    "basis": {
      "source_type": "design_conversation",
      "explicitness": "explicit",
      "evidence_count": 1
    }
  },
  "scope": {
    "project": "ent:project:amos",
    "applies_to": ["reasoner", "planner", "executor", "critic", "tool_worker"]
  },
  "evidence_refs": [
    "evd_01JZB0XG4X7Y3VHV93R5Y3NMD1"
  ],
  "layer": "consolidated_long_term",
  "lifecycle_state": "active",
  "health_status": "healthy",
  "retention_class": "project",
  "version": 1,
  "created_at": "<iso_timestamp>"
}
```

An LLM reasoner might later receive the generated sentence:

```text
Amos owns capacity management internally and shields connected agents from storage pressure.
```

That sentence is a rendered view. The atom is the memory.

#### 7.1.3 Payload schemas by atom type

Each atom type defines a narrow payload shape.

Belief payload:

```json
{
  "subject": "ent:user:primary",
  "relation": "rel:working_on",
  "object": "ent:project:amos",
  "modality": "observed",
  "qualifiers": {
    "activity_state": "active"
  },
  "validity_interval": {
    "from": "<iso_timestamp>",
    "until": null
  }
}
```

Preference payload:

```json
{
  "holder": "ent:user:primary",
  "polarity": "prefers",
  "target": "ent:response_style:precise_technical_discussion",
  "applicability_scope": {
    "task_type": "architecture_design"
  },
  "strength": "high",
  "exceptions": []
}
```

Goal payload:

```json
{
  "owner": "ent:user:primary",
  "desired_state": "ent:project_state:amos_design_spec_mature",
  "goal_status": "active",
  "priority": "high",
  "dependencies": [
    "ent:design_topic:capacity_governance",
    "ent:design_topic:semantic_maintenance_processor"
  ],
  "blockers": []
}
```

Procedure payload:

```json
{
  "name": "proc:architecture_design_discussion",
  "trigger_context": {
    "task_type": "architecture_design",
    "phase": "planning"
  },
  "preconditions": [
    "user_has_not_requested_code"
  ],
  "steps": [
    "define_problem",
    "identify_design_axes",
    "separate_logical_and_physical_layers",
    "discuss_tradeoffs",
    "capture_open_questions"
  ],
  "expected_outputs": [
    "design_spec_update",
    "open_questions",
    "roadmap"
  ],
  "known_failure_modes": [
    "premature_implementation_detail",
    "overgeneralized_memory_claims"
  ],
  "recovery_strategies": [
    "return_to_conceptual_model",
    "narrow_scope"
  ]
}
```

Episode payload:

```json
{
  "task": "design_amos_memory_architecture",
  "context": {
    "project": "ent:project:amos",
    "session": "ses_<ulid>"
  },
  "decisions": [
    "amos_should_be_shared_memory_plane",
    "amos_should_have_capacity_governor",
    "llm_not_required_for_core_maintenance",
    "semantic_maintenance_processor_preferred"
  ],
  "outcome": "design_spec_expanded",
  "successful_strategies": [
    "separate_logical_model_from_storage_format",
    "make_agents_insulated_from_capacity_concerns"
  ],
  "raw_event_range": {
    "from": "evt_<start_id>",
    "to": "evt_<end_id>"
  }
}
```

#### 7.1.4 Normalized values

The canonical atom should avoid repeating English labels where possible. Values should be normalized to entity, relation, concept, enum, and evidence identifiers.

Avoid storing this as canonical memory:

```json
{
  "subject": "the user",
  "relation": "likes",
  "object": "long detailed technical discussions about architecture"
}
```

Prefer:

```json
{
  "subject": "ent:user:primary",
  "relation": "rel:prefers",
  "object": "ent:response_style:technical_architecture_depth"
}
```

Human-readable labels live in dictionaries or entity records:

```json
{
  "id": "ent:response_style:technical_architecture_depth",
  "type": "concept",
  "labels": [
    "technical architecture depth",
    "deep technical design discussion"
  ]
}
```

This reduces storage bloat, improves deduplication, and avoids repeatedly reprocessing English.

Identifier namespaces:

```text
atom_<digest>:
  MemoryAtom generated by v1-local when the caller does not provide an id

<caller_supplied_stable_string>:
  MemoryAtom, when the caller provides an explicit id such as a migrated
  external atom id

edge_<digest>:
  AssociationEdge generated by v1-local

evd_<digest>:
  EvidenceRecord generated by v1-local

evt_<uuid>:
  EventJournalEntry

pkt_<digest>:
  MemoryPacket

rto_<uuid>:
  Retrieval outcome telemetry record

tmb_<uuid>:
  MemoryTombstone

ent:<type>:<stable_slug_or_ulid>:
  Entity

rel:<relation_name>:
  Relation dictionary entry

enum:<enum_group>:<value>:
  Controlled enum value

proc:<stable_slug_or_ulid>:
  Procedure name or procedure family
```

Identifier governance rules:

```text
1. Opaque record IDs are immutable.
2. Human-readable slugs may be aliases, but canonical record identity should not depend on mutable labels.
3. Entity merges create a journaled alias or merged_into pointer; old atom payloads remain interpretable.
4. Entity splits create new entity IDs and a journaled split event; old atoms are not rewritten without a projection event.
5. Relation IDs are added through a relation dictionary update, not free-form payload strings.
6. Relation and enum definitions include owner scope, description, inverse relation if any, and deprecation state.
7. Deprecated identifiers remain resolvable until no retained atom, edge, evidence, or journal event references them.
```

#### 7.1.5 Atom value type system

Atom payload values should support a compact type system:

```text
AtomValue =
  entity_ref
  relation_ref
  enum_ref
  string_literal
  number
  boolean
  timestamp
  duration
  quantity
  list
  record/object
  evidence_ref
  external_uri
```

Example quantity value:

```json
{
  "relation": "rel:max_capacity",
  "object": {
    "value_type": "quantity",
    "value": 500,
    "unit": "GB"
  }
}
```

Typed values prevent ambiguous storage such as “capacity is five hundred.”

#### 7.1.6 Atom versus event, evidence, and edge

A typed atom is not the raw evidence. It stores evidence references.

```json
{
  "evidence_refs": [
    "evd_01JZB0XG4X7Y3VHV93R5Y3NMD1"
  ]
}
```

The evidence record points to source material:

```json
{
  "evidence_id": "evd_01JZB0XG4X7Y3VHV93R5Y3NMD1",
  "schema_version": "amos.v1",
  "source_type": "conversation_turn",
  "source_ref": "archive://tenant/default/session/<session_id>/turn/<turn_id>",
  "payload": {
    "span": {
      "start": 0,
      "end": 187
    }
  },
  "captured_at": "<iso_timestamp>",
  "checksum": "sha256:...",
  "access_policy": {
    "visibility": ["all"],
    "mutable_by": ["owner"]
  },
  "scope": {
    "tenant": "default",
    "workspace": "amos"
  }
}
```

A typed atom is also not the association edge. Associations are separate records so they can be pruned, reinforced, inhibited, or reweighted independently:

```json
{
  "edge_id": "edge_01JZB22MEQ2T3P8R6N4B8Z7WEA",
  "source_ref": "atm_01JZB0Z8WX4R7AN7M7P5XDK9Q4",
  "target_ref": "atm_01JZB1H4M92HMSAPV2Y6RE3XKC",
  "relation": "rel:supports",
  "confidence": {
    "level": "high",
    "score": 0.87
  },
  "evidence_refs": [
    "evd_01JZB0XG4X7Y3VHV93R5Y3NMD1"
  ],
  "created_at": "<iso_timestamp>",
  "updated_at": "<iso_timestamp>",
  "lifecycle_state": "active",
  "health_status": "trusted"
}
```

Every committed atom should be introduced through the Amos Event Journal. The journal records the mutation; the Canonical Memory Graph stores the current projected state.

```json
{
  "event_id": "evt_01JZB1A4DM6R7Y6CAQ9VY0H2XF",
  "event_type": "atom_committed",
  "schema_version": "amos.v1",
  "actor": "svc:memory_steward",
  "idempotency_key": "idem_abc123",
  "occurred_at": "<iso_timestamp>",
  "accepted_at": "<iso_timestamp>",
  "graph_version": 12,
  "payload": {
    "operation": "commit_atom",
    "atom_id": "atm_01JZB0Z8WX4R7AN7M7P5XDK9Q4",
    "atom_type": "belief",
    "atom_version": 1
  },
  "evidence_refs": [
    "evd_01JZB0XG4X7Y3VHV93R5Y3NMD1"
  ]
}
```

#### 7.1.7 Physical storage plan

Amos should define the storage format in three levels:

```text
Level 1: Canonical logical schema
  MemoryAtom, AssociationEdge, EvidenceRecord, EventJournalEntry

Level 2: Canonical interchange encoding
  JSON-compatible records with strict schemas

Level 3: Physical storage encoding
  Implementation-dependent: JSONB, Protobuf, CBOR, graph database, columnar snapshot, etc.
```

A practical v1 implementation can use:

```text
Shared Amos service
  HTTP API process that owns the canonical store and serializes mutations

Event Journal
  SQLite append-only event journal table, with exportable JSON records

Canonical Memory Graph
  SQLite tables with JSON payload columns plus normalized indexes

Evidence Archive
  object storage or filesystem storage with checksums and retention metadata

Derived Indexes
  keyword index, graph adjacency tables, packet cache, and replaceable vector index
```

Postgres is not required for v1. A later production-scale backend may replace
SQLite with Postgres tables using JSONB payloads and stronger multi-process
operational features, as long as it preserves the same journal, schema,
authorization, deletion, replay, and packet contracts.

A v1 relational MVP can map the canonical graph into these SQLite tables.
Future Postgres migrations can keep the same logical columns and use JSONB
where appropriate.

```text
atoms
  id
  type
  schema_version
  payload JSON
  evidence_refs JSON
  scope JSON
  confidence JSON
  salience REAL
  utility REAL
  layer
  health_status
  lifecycle_state
  retention_class
  access_policy JSON
  created_at
  observed_at
  updated_at
  last_accessed
  decay_policy JSON
  version
  supersedes JSON
  revision_history JSON
  index_refs JSON
  deleted

edges
  edge_id
  source_ref
  target_ref
  relation
  schema_version
  evidence_refs JSON
  scope JSON
  confidence JSON
  lifecycle_state
  health_status
  created_at
  updated_at
  version
  deleted

evidence
  evidence_id
  schema_version
  source_type
  source_ref
  payload JSON
  captured_at
  checksum
  access_policy JSON
  scope JSON
  event_id

event_journal
  event_id
  event_type
  schema_version
  actor TEXT
  target_refs JSON
  payload JSON
  payload_refs JSON
  evidence_refs JSON
  idempotency_key
  payload_digest
  causal_parent_ids JSON
  expected_versions JSON
  authorization_context JSON
  occurred_at
  accepted_at
  result_status
  projection_status
  previous_event_hash
  checksum
  graph_version
```

The logical schema should remain stable even if the physical encoding changes.

```text
Typed Atom logical schema: stable.
Physical encoding: replaceable.
```

#### 7.1.8 Rendering principle

Typed atoms can render into English, but English is not the canonical memory.

A reasoner may receive:

```text
The user prefers iterative conceptual design before implementation code during architecture discussions, unless they explicitly ask for code.
```

That text is generated from a `PreferenceAtom`. The stored atom remains structured, normalized, scoped, evidence-backed, and versioned.

#### 7.1.9 Field ownership and canonical vocabulary

Envelope fields are owned by the Amos kernel and have the same meaning for every atom type. In v1-local, `schema_version` is the shared constant `amos.v1`; payload validation is selected by `MemoryAtom.type`. Later schema versions may split these into per-type version identifiers through an explicit migration.

An atom payload must not redefine envelope fields under different names. In particular:

```text
Use evidence_refs in the envelope.
Do not add evidence_ids, source, or evidence inside payloads unless the payload is itself describing evidence.

Use scope in the envelope for tenancy, workspace, project, user, agent, session, and retrieval applicability.
Use payload-specific names such as applicability_scope only when the atom type needs semantic applicability conditions.

Use confidence in the envelope.
Do not duplicate confidence in payloads.

Use lifecycle_state for storage/retrieval lifecycle.
Use health_status for memory quality.
Use payload-specific status fields, such as goal_status or commitment_status, only for domain state.
```

Canonical atom lifecycle states:

```text
proposed:
  candidate extracted from evidence but not committed to active memory

active:
  committed and eligible for ordinary retrieval

archived:
  retained outside ordinary retrieval; available for audit, history, or explicit deep retrieval

tombstoned:
  minimal marker retained to prevent silent recreation or preserve required audit

deleted:
  payload removed or rendered unrecoverable under retention, privacy, or user deletion policy
```

Legal lifecycle transitions:

```text
proposed -> active
proposed -> archived
proposed -> deleted
active -> archived
active -> tombstoned
active -> deleted
archived -> active
archived -> tombstoned
archived -> deleted
tombstoned -> deleted
```

`deleted` is terminal for the removed record. If a later observation reintroduces similar content, Amos must create a new atom only if policy allows it and the tombstone does not prohibit recreation.

Canonical health statuses:

```text
healthy
stale
redundant
contradicted
incoherent
orphaned
confounding
overgeneralized
underspecified
low_utility
privacy_sensitive
review_required
```

An atom may have one primary `health_status` and optional implementation-specific `health_flags` if several conditions apply. Retrieval defaults:

```text
healthy:
  eligible for normal retrieval

stale:
  excluded unless timeline/history or refresh is relevant

redundant:
  excluded in favor of canonical replacement

contradicted:
  retrieved only with conflict context

incoherent:
  excluded until repaired or archived

orphaned:
  excluded until relinked or explicitly requested

confounding:
  actively suppressed unless specifically requested

overgeneralized:
  down-ranked or narrowed before normal retrieval

privacy_sensitive:
  retrieved only when access policy and processor target allow it

review_required:
  excluded from autonomous promotion and high-impact use
```

Supersession is a lifecycle signal as well as a graph relation. When an active
atom is the target of an active `rel:supersedes` edge, normal packet retrieval
omits it as superseded unless the caller explicitly asks for superseded items.
If included, it remains down-ranked. The memory policy may archive such atoms
without requiring an atom-local decay rule, because the canonical graph already
contains the replacement evidence.

Canonical score fields:

```text
confidence:
  evidence-backed belief in correctness or usefulness for the scoped claim

salience:
  expected importance to future tasks independent of current retrieval context

utility:
  observed usefulness based on successful retrieval, reuse, corrections, and outcomes
```

Recommended score representation for implementation:

```json
{
  "level": "high",
  "score": 0.91,
  "basis": {
    "source_type": "direct_user_instruction",
    "explicitness": "explicit",
    "evidence_count": 2,
    "contradiction_count": 0,
    "last_calibrated_at": "<iso_timestamp>"
  }
}
```

Implementations may begin with categorical levels, but stored records should reserve numeric scores so ranking and thresholding can be calibrated without schema churn.

### 7.2 Entity

Stable references for people, projects, tools, repositories, files, organizations, concepts, agents, and environments.

```text
Entity
  id
  type
  labels
  aliases
  canonical_name
  scope
  external_refs
  merged_into
  version
  created_at
  updated_at
```

Repeated strings should be dictionary-encoded by entity IDs.

Entity IDs should be stable within a configured scope. Alias merges and splits must be journaled so old atoms can be reprojected or interpreted after dictionary changes.

### 7.3 SourceEvent

Immutable record of what happened in the source world.

```text
SourceEvent
  source_type
  source_ref
  payload
  scope
  access_policy
```

Source events preserve evidence and should not be casually rewritten. They are distinct from Event Journal entries. A source event describes an observation such as a user turn, tool call, file change, or evaluator result. V1-local capture accepts an actor and idempotency key at the service/API layer, normalizes the source event into an `EvidenceRecord`, and appends an `evidence_captured` journal entry. An Event Journal entry describes an accepted Amos memory mutation such as `atom_committed` or `atom_merged`.

### 7.4 BeliefAtom

Represents a claim payload. Evidence, confidence, scope, lifecycle, and health live in the atom envelope.

```text
BeliefAtom payload
  subject
  relation
  object
  modality: observed | inferred | user_stated | system_derived | predicted
  qualifiers
  validity_interval
```

Memory is what was observed. Belief is what the system currently accepts as useful or likely true.

### 7.5 PreferenceAtom

Preferences are scoped and contextual.

```text
PreferenceAtom payload
  holder
  polarity: prefers | avoids | requires | forbids
  target
  applicability_scope
  strength
  exceptions
```

Example:

```text
holder: user
polarity: prefers
target: iterative conceptual design before implementation
applicability_scope: architecture discussions
strength: high
```

This prevents overgeneralized memories such as `user never wants code`.

### 7.6 Goal

```text
Goal payload
  owner
  desired_state
  goal_status: proposed | active | paused | satisfied | abandoned | blocked
  priority
  deadline
  dependencies
  blockers
```

### 7.7 Commitment

```text
Commitment payload
  agent
  promised_action
  recipient
  commitment_status: open | fulfilled | failed | cancelled | superseded
  due_condition
  source_event
```

This helps the planner avoid losing open loops.

### 7.8 ProcedureAtom

Procedural memory should be structured, versioned, and auditable.

```text
ProcedureAtom payload
  name
  trigger_context
  preconditions
  steps
  tool_affordances
  expected_outputs
  known_failure_modes
  recovery_strategies
  owner
```

A procedure may render into English for an LLM or into a structured action schema for a planner/executor.

### 7.9 Episode

A compressed trace of meaningful activity.

```text
Episode payload
  task
  context
  actions_taken
  decisions
  outcome
  errors
  corrections
  successful_strategies
  linked_beliefs
  linked_procedures
  raw_event_range
```

Raw transcripts remain in the Evidence Archive.

### 7.10 EvidenceRecord

```text
EvidenceRecord
  evidence_id
  schema_version
  source_type
  source_ref
  payload
  captured_at
  checksum
  access_policy
  scope
```

Evidence supports auditability and reconstruction.

### 7.11 AssociationEdge

```text
AssociationEdge
  edge_id
  source_ref
  target_ref
  relation
  schema_version
  evidence_refs
  scope
  confidence
  lifecycle_state
  health_status
  created_at
  updated_at
  version
```

Edges are first-class. They are not incidental metadata.

When an atom leaves the active lifecycle, live edges attached to it must leave
the active graph in the same mutation. Stewardship also removes legacy live
capability, limitation, commitment, and attribution edges whose endpoint is no
longer active; provenance carried by canonical atom fields remains auditable.

### 7.12 MemoryPacket

The retrieval output consumed by processors. A `MemoryPacket` is a rendered, bounded, non-authoritative view over canonical memory at a specific graph version.

```text
MemoryPacketRequest
  request_id
  client_identity
  target_processor
  retrieval_mode: general | self_awareness | shared_coordination | agentic_recall
  attention_context
  task_context
  scope
  shared_view_ref
  agentic_recall_subject
  agency_attribution_filter
  bias_guardrails
  requested_memory_types
  max_items
  token_or_byte_budget
  consistency_requirement
  include_conflicts
  include_archived
  include_low_health
  include_superseded
  include_provenance
  rendering_target
```

```text
MemoryPacket
  packet_id
  schema_version
  request
  graph_version
  generated_at
  target_processor
  retrieval_mode
  scope
  shared_view_ref
  pressure_mode
  degradation
  items
  conflicts
  omissions
  attention_trace
  provenance
  cache_policy
```

Attention is runtime policy over canonical memory, not a canonical memory type.
It lets the caller disclose the current task, role, mission, risk posture,
focus terms, suppression terms, and desired counterevidence posture so Amos can
allocate packet budget deliberately.

```text
AttentionContext
  active_task
  mission
  goal
  role
  risk_posture: normal | cautious | high_risk
  time_horizon: immediate | short | long
  focus_terms
  suppress_terms
  boost_memory_types
  suppress_memory_types
  counterevidence_required
  novelty_preference
```

```text
AttentionTrace
  policy_id
  context
  focus_terms
  suppress_terms
  weight_adjustments
  selected_item_refs
  inhibited_refs
  omitted_reasons
```

Packet item shape:

```text
MemoryPacketItem
  atom_ref
  type
  payload
  score
  score_components
  item_ref             # v1-local compatibility alias for atom_ref
  item_kind: atom      # v1-local packets currently render atom items
  atom_id              # v1-local compatibility alias for atom_ref
  atom_type            # v1-local compatibility alias for type
  lifecycle_state
  health_status
  rank
  activation_score     # v1-local compatibility alias for score
  confidence
  salience
  utility
  rendered_content
  evidence_refs
  access_decision
  freshness
  scope
  updated_at
  version
  provenance
```

`omissions` records why potentially relevant material was not included:

```text
budget_exhausted
access_denied
stale_suppressed
confounding_suppressed
pressure_degraded
derived_index_stale
counterevidence_unavailable
bias_guardrail_suppressed
```

`degradation` must be present when recall depth, evidence detail, conflict detail, or derived-index freshness is reduced by capacity pressure or consistency lag.

```text
degradation
  pressure_mode
  reduced_recall_depth
  omitted_evidence_detail
  index_freshness
  reason_codes
```

The packet can be rendered as English, JSON, graph neighborhoods, planner state, or executor context.

### 7.13 Self-awareness support objects

Agent self-awareness requires explicit memory objects for self-model, capabilities, limitations, runtime state, and self-assessment. These objects should be ordinary typed atoms where durable, and short-lived runtime records where volatile.

#### 7.13.1 SelfModelAtom

Represents what an agent or service currently knows about itself as an operating participant.

```text
SelfModelAtom payload
  subject_agent
  role
  owner_scope
  operating_mode
  delegated_authority
  current_objectives
  active_constraints
  known_dependencies
  self_description
```

Rules:

```text
subject_agent must resolve to a ClientIdentity or Entity.
delegated_authority must be evidence-backed by policy, user instruction, or system configuration.
self_description is rendered from structured fields; it is not authoritative prose.
```

#### 7.13.2 CapabilityAtom

Represents a capability the agent can currently use or has demonstrated.

```text
CapabilityAtom payload
  subject_agent
  capability
  capability_type: tool | model | memory | planning | execution | communication | analysis
  availability: available | unavailable | degraded | unknown
  preconditions
  required_permissions
  operational_limits
  verification_method
  last_verified_at
```

Capability memories are volatile unless backed by stable configuration. Tool availability, sandbox permissions, network access, installed connectors, model limits, and execution budgets should be refreshed from current runtime state before being presented as active self-knowledge.

#### 7.13.3 LimitationAtom

Represents a known inability, restriction, uncertainty, or boundary.

```text
LimitationAtom payload
  subject_agent
  limitation
  limitation_type: policy | permission | resource | knowledge | model | tool | environment | reliability
  scope
  severity
  workaround
  verification_method
  last_verified_at
```

Limitations are as important as capabilities for self-awareness. An agent should retrieve relevant limitations before making capability claims, selecting tools, or committing to future work.

#### 7.13.4 RuntimeStateSnapshot

Represents current ephemeral operating state. It is not durable long-term memory by default.

```text
RuntimeStateSnapshot
  snapshot_id
  client_identity
  scope
  current_task
  active_plan
  active_goal_refs
  open_commitment_refs
  available_tools
  denied_tools
  resource_budgets
  execution_environment
  recent_errors
  current_uncertainties
  captured_at
  expires_at
```

Runtime state should be treated as high-volatility evidence. It may generate durable atoms only when repeated, policy-relevant, or explicitly committed.

#### 7.13.4.1 Experience-derived profile updates

Repeated agent experiences may update an agent's self-model, but they should not
overwrite the static self-model contract. They should be represented as ordinary
canonical atoms, usually `capability`, `limitation`, `procedure`, or `semantic`
atoms with explicit provenance back to action outcomes and retrieval outcomes.

Recommended payload fields for experience-derived capability and limitation
atoms:

```text
profile_update_source:
  identifies the client processor or experience source

subject_agent / agent_id:
  durable agent identity whose profile is being updated

experience_kind:
  action, decision, review, tool use, recovery, planning, or other client term

outcome_category:
  capability | limitation | procedure | observation

source_count:
  number of related experiences supporting the update

recent_source_refs:
  bounded refs to recent source traces or outcomes

supported_count / failed_count:
  aggregate outcome counts when available

reuse_guidance:
  concise instruction for when the learned profile item should influence a
  future decision
```

Experience-derived profile atoms must remain separate from bootstrap contract
atoms. Bootstrap logic may archive old static profile versions, but it must not
archive learned profile updates solely because they lack the current static
profile version. Learned profile updates are maintained by their source
processor, retrieval outcomes, health state, and stewardship policy.

#### 7.13.5 SelfAssessmentRecord

Represents an introspective evaluation of recent agent behavior.

```text
SelfAssessmentRecord
  assessment_id
  subject_agent
  task_ref
  packet_refs
  claimed_capabilities
  capabilities_used
  commitments_made
  commitments_satisfied
  uncertainties_declared
  errors_observed
  corrections_received
  calibration_delta
  recommended_memory_updates
  assessed_at
```

Self-assessments should update utility, confidence calibration, limitation memories, procedure health, and retrieval policies. They should not become durable self-beliefs without evidence and stewardship review.

### 7.14 Agentic recall support objects

Agentic recall is a retrieval mode that emphasizes memories where an agent acted, decided, corrected course, or accepted responsibility. It is useful for self-awareness and professional identity formation, but it is also bias-prone because it can over-select evidence that supports a preferred self-narrative.

Amos should therefore model agentic recall as an auditable retrieval and rendering contract, not as a separate identity store.

#### 7.14.1 AgenticTrace

Represents the evidence-backed chain connecting an agent, its intentions or decisions, its actions, and observed outcomes.

```text
AgenticTrace
  trace_id
  subject_agent
  scope
  task_ref
  intent_refs
  decision_refs
  action_refs
  tool_event_refs
  outcome_refs
  correction_refs
  limitation_refs
  responsibility_level: primary | contributing | reviewing | observing | blocked
  agency_confidence
  external_constraints
  counterevidence_refs
  assessed_at
```

Rules:

```text
subject_agent must identify the agent whose agency is being recalled.
responsibility_level must be derived from evidence, not inferred from a positive outcome alone.
external_constraints record policy, permission, user direction, tool failure, or environmental limits.
counterevidence_refs must be included when the trace could support an inflated agency claim.
```

#### 7.14.2 ActionOutcomeRecord

Represents the outcome of an agent action or decision.

```text
ActionOutcomeRecord
  action_id
  subject_agent
  scope
  action_type
  decision_ref
  tool_ref
  expected_outcome
  actual_outcome
  success_status: succeeded | failed | partial | blocked | unknown
  error_refs
  correction_refs
  learning_refs
  observed_at
```

Action outcomes should be captured for successes, failures, blocked actions, corrections, and abandoned plans. Failures and blocked actions are first-class inputs to self-awareness, not just negative telemetry.

#### 7.14.3 SelfNarrative

Represents a generated explanation of an agent's role, pattern of agency, or recent professional identity. It is never canonical memory.

```text
SelfNarrative
  narrative_id
  subject_agent
  scope
  source_trace_refs
  source_self_model_refs
  source_capability_refs
  source_limitation_refs
  source_assessment_refs
  source_counterevidence_refs
  narrative_scope
  rendering_target
  balance_report
  generated_at
  expires_at
```

Rules:

```text
1. A SelfNarrative is a generated artifact.
2. It must cite the AgenticTrace, ActionOutcomeRecord, SelfModelAtom,
   CapabilityAtom, LimitationAtom, and SelfAssessmentRecord refs it uses.
3. It must include relevant failures, corrections, limitations, blocked actions,
   and external constraints when they affect the claim being rendered.
4. It must expire or be rebuilt after contradictory evidence, capability changes,
   major runtime-state changes, or self-assessment calibration changes.
5. It must not be promoted into a durable self-belief without stewardship review.
```

#### 7.14.4 Agency attribution rules

Agentic recall must distinguish who or what caused an outcome.

```text
self:
  the subject_agent directly decided or acted

other_agent:
  another identified agent decided or acted

shared_system:
  multiple agents, user instructions, policies, procedures, or tools jointly shaped the outcome

external:
  environment, tool availability, permission, policy, user decision, or unrelated event dominated the outcome

unknown:
  evidence does not support a confident agency assignment
```

Multi-agent attribution rules:

```text
1. Do not attribute another agent's action to the subject_agent.
2. Do not convert shared-system success into individual success without evidence.
3. Preserve per-agent responsibility overlays in SharedMemoryView.
4. Represent handoffs, reviews, approvals, and blocked actions explicitly.
5. Include external constraints when they materially shaped the outcome.
6. Prefer unknown or contributing over primary when evidence is incomplete.
```

#### 7.14.5 Agentic recall bias guardrails

Because agentic recall is selective by design, Amos must include balance constraints.

```text
bias guardrails:
  retrieve successes and failures
  retrieve corrections and rejected claims
  retrieve blocked actions and external constraints
  retrieve limitations alongside capabilities
  retrieve counterevidence for strong self-claims
  prevent self-serving over-attribution
  prevent omission of recent failures when rendering competence claims
```

Agentic recall ranking may boost records where `subject_agent` intentionally acted, but it must penalize packets or narratives that omit material counterevidence, failure records, or relevant limitations.

### 7.15 SharedMemoryView

Different agents need a shared memory view without sharing one identity or one self-model. A `SharedMemoryView` is a coordinated projection over the Canonical Memory Graph for a scope and audience.

Design principle:

```text
Agents share memory.
Agents do not share a self-model.
An Amos instance is the memory plane for one coherent system of agents.
```

```text
SharedMemoryView
  view_id
  scope
  audience
  graph_version
  generated_at
  common_items
  shared_goals
  shared_commitments
  shared_constraints
  shared_context
  conflicts
  omissions_by_identity
  per_agent_overlays
  convergence_policy
  cache_policy
```

Rules:

```text
1. A SharedMemoryView is not a separate memory store.
2. common_items are selected from the canonical graph at one graph_version.
3. all agents in the audience see the same common_items unless access policy denies an item.
4. per_agent_overlays contain role-specific rendering, permissions, self-model refs,
   capability/limitation refs, and runtime-state refs.
5. omissions_by_identity records access or role filtering without leaking denied content.
6. shared goals, commitments, constraints, and conflicts should be stable enough for coordination.
7. volatile runtime state can be referenced but should not be promoted into common_items by default.
```

This gives a multi-agent team one coordinated memory surface while preserving distinct agent identities.

```text
Reasoner:
  sees common task memory plus reasoning-relevant overlays

Planner:
  sees same common task memory plus goals, dependencies, and open loops

Executor:
  sees same common task memory plus tool permissions and execution procedures

Critic/steward:
  sees same common task memory plus evidence, conflicts, telemetry, and health flags
```

Shared view consistency:

```text
strong_shared:
  all audience members receive common_items from the same graph_version

monotonic_shared:
  common_items do not move backward in graph_version for a continuing task

eventual_shared:
  common_items may lag, but packet metadata must disclose graph_version and freshness
```

---

## 8. Memory lifecycle

```text
1. Capture
   Store raw events cheaply and append-only.

2. Atomize
   Generate candidate memory atoms from events.

3. Normalize
   Resolve entities, references, scopes, and types.

4. Reconcile
   Merge duplicates, detect contradictions, update confidence.

5. Link
   Create cross-category associative edges.

6. Triage
   Score utility, novelty, confidence, privacy risk, and future retrieval value.

7. Commit
   Promote selected candidates to active canonical memory.

8. Render
   Generate processor-specific memory packets.

9. Reinforce
   Strengthen memories and associations that improve outcomes.

10. Repair
   Fix stale, redundant, incoherent, orphaned, confounding, or contradictory memories.

11. Compact
   Archive low-value details, consolidate episodes, rebuild indexes.

12. Forget or delete
   Apply retention policy, user deletion, privacy controls, and low-utility decay.
```

The system should avoid repeated full-memory summarization. Maintenance should be incremental and evidence-grounded.

---

## 9. Cross-tier promotion

Promotion is typed and directional.

```text
Immediate observation
  ↓ attention
Working memory item
  ↓ episode binding
Recent episode
  ↓ repeated/salient evidence
Belief / preference / semantic claim
  ↓ repeated successful use
Procedure / policy / durable project model
```

### 9.1 Preference promotion

```text
one explicit user instruction:
  candidate preference, high local priority

repeated explicit instructions:
  active scoped preference

consistent behavior over time:
  reinforced preference

contradictory later instruction:
  scope split or stale old preference
```

### 9.2 Procedure promotion

```text
one success:
  success pattern

multiple successes in same task class:
  procedure candidate

success across contexts:
  active procedure

failure after use:
  procedure revision or demotion
```

### 9.3 Semantic promotion

```text
one source-backed fact:
  source-backed claim

multiple independent sources:
  reinforced semantic claim

later source conflict:
  disputed claim with evidence branches
```

### 9.4 Belief promotion

```text
raw event:
  observation

extracted claim:
  candidate belief

validated against evidence:
  active belief

reused successfully:
  reinforced belief

contradicted/corrected:
  stale, narrowed, or replaced
```

---

## 10. Cross-tier demotion

Demotion is as important as promotion.

```text
health_status: healthy → stale
health_status: healthy → contradicted
lifecycle_state: active → archived
lifecycle_state: active → tombstoned
hot → warm
warm → cold
global → scoped
procedure → success pattern
belief → historical observation
```

Examples:

```text
A tool-specific procedure stops working after an API change:
  active procedure → stale procedure

A user preference turns out context-specific:
  global preference → scoped preference

A project appears inactive for months:
  active project memory → warm archive

A semantic claim depends on outdated software docs:
  active claim → stale claim requiring refresh
```

---

## 11. Associative retrieval

Retrieval should not be flat chunk search. It should use cue extraction, spreading activation, inhibition, and diversity selection.

```text
1. Extract cues from current situation.
2. Activate directly matching atoms.
3. Spread activation across associative edges.
4. Apply decay, inhibition, confidence, utility, and access policy weighting.
5. Select a diverse cross-type memory packet.
6. Render for the target processor.
```

### 11.1 Retrieval cues

```text
entities
task type
goal
constraints
recent user instructions
active project
agent identity
active self-model
retrieval mode
agentic recall subject
agency attribution
decision/action/outcome refs
available and denied capabilities
known limitations
runtime state snapshot
recent failures and corrections
counterevidence requirements
tool state
time horizon
risk level
desired output type
```

### 11.2 Activation scoring

```text
activation =
  direct cue match
+ semantic similarity
+ associative edge activation
+ recency
+ confidence
+ utility
+ salience
+ goal relevance
+ procedural applicability
+ attention focus
+ attention type boost
+ attention counterevidence boost
+ agency match
+ attribution confidence
+ correction/learning relevance
- contradiction penalty
- staleness penalty
- attention suppression penalty
- privacy/access penalty
- redundancy penalty
- over-attribution penalty
- omitted counterevidence penalty
```

Implementations should persist score components with each packet item:

```text
score_components
  direct_cue_match: 0.0..1.0
  semantic_similarity: 0.0..1.0
  edge_activation: 0.0..1.0
  recency: 0.0..1.0
  confidence: 0.0..1.0
  utility: 0.0..1.0
  salience: 0.0..1.0
  goal_relevance: 0.0..1.0
  procedural_applicability: 0.0..1.0
  attention_focus: 0.0..1.0
  attention_type_boost: 0.0..1.0
  attention_counterevidence: 0.0..1.0
  attention_novelty: 0.0..1.0
  attention_suppression_penalty: 0.0..1.0
  agency_match: 0.0..1.0
  attribution_confidence: 0.0..1.0
  correction_learning_relevance: 0.0..1.0
  contradiction_penalty: 0.0..1.0
  staleness_penalty: 0.0..1.0
  access_penalty: 0.0..1.0
  redundancy_penalty: 0.0..1.0
  over_attribution_penalty: 0.0..1.0
  omitted_counterevidence_penalty: 0.0..1.0
  ignored_failure_penalty: 0.0..1.0
```

The first implementation may use hand-tuned weights, but every packet item must expose enough components to debug why it appeared, why it was suppressed, or why a stronger scoped memory inhibited a generic one.

V1-local retrieval computes recency from `updated_at` age, not mere timestamp
presence. The default horizon is 30 days: a just-updated atom scores near `1.0`,
and atoms at or beyond the horizon score `0.0` for the recency component.

`goal_relevance` and `procedural_applicability` are relevance-conditioned. Goal,
commitment, and procedure atoms do not receive their type boost solely because
of their memory type; they need cue overlap, attention focus, or relation
activation connected to the current request.

`edge_activation` is seeded by cue or attention matches and then propagated over
typed graph edges with bounded relation weights. It is not raw degree
centrality. Degree may still be used by separate components such as novelty
preference, but it must not make a globally connected atom look relevant without
a path from the active cue or attention context.

Attention does not replace cues. Cues describe what the caller is asking about;
attention describes what should be foregrounded, reserved, or inhibited while
answering. For example, a pilot can ask about current training policy while
attention reserves budget for active mission rules and recent corrective
failures, and a critic can ask the same cue while reserving more budget for
counterevidence and contradictions.

If `cues` is empty, retrieval enters browse-by-context mode: all visible,
eligible atoms may enter ranking, and scope plus `attention_context` become the
primary relevance signal. This is intentional for callers that want "what is
relevant to my current mission/role/task?" rather than an answer to a specific
query string. Implementations must expose this through `attention_trace` so the
caller can see which focus terms selected or inhibited packet items.

Attention and cue token matching must use canonical search text derived from
atom ids, atom types, and payload values only. Payload object keys and envelope
field names are not semantic content and must not create focus, suppression, or
cue-overlap matches. This prevents generic keys such as `claim`, `confidence`,
`schema_version`, or `status` from making unrelated atoms appear mission
relevant.

`novelty_preference` is advisory and should be implemented as a bounded
`attention_novelty` component rather than merely echoed in the request. V1-local
uses a lightweight graph-familiarity proxy: less-connected atoms receive more
novelty credit when the caller asks for novelty, while ordinary retrieval keeps
the component at zero.

V1-local maintains a disposable SQLite token candidate index populated from the
same canonical payload-value search text. When cue or attention tokens are
present, retrieval prefilters atom ids through this token table and then expands
the candidate set to graph neighbors so edge activation can still surface linked
memories that do not repeat the query wording. Empty-cue retrieval with
attention terms uses the same token prefilter; empty-cue retrieval without
attention terms remains an unprefiltered browse over visible eligible memory.
If token prefiltering finds no direct candidates, v1-local falls back to bounded
semantic scoring over visible eligible memory so morphology, spelling variants,
and latent token relationships can still admit relevant atoms.
Once candidates are selected, edge degree, supersession, and activation reads
must be scoped to the candidate refs through indexed `source_ref`/`target_ref`
lookups. Retrieval must not scan the full edge table for every packet when a
bounded candidate set is available.
Canonical memory remains the event journal and atom/edge graph, not the derived
token index; the memory policy can rebuild the index from atom `index_refs`.

### 11.3 Diversity requirement

Retrieval should produce a bundle, not just top-k text chunks:

```text
best active beliefs
best active preferences
best procedure
best current self-model
best applicable capabilities
best applicable limitations
best agentic traces
best action outcomes
best corrections or blocked actions
best attribution counterevidence
best semantic claims
best recent episode
best conflict/uncertainty
best evidence pointers
```

This prevents one dominant cluster from crowding out cross-category relevance.

---

## 12. Lateral inhibition and scope selection

Associative retrieval needs inhibition, not only activation.

Examples:

```text
A highly relevant scoped preference suppresses a generic preference.
A newer correction suppresses an older inferred belief.
A high-confidence procedure suppresses a weak one-off episode.
An active task goal suppresses unrelated semantic neighbors.
```

This prevents confounding memories from being retrieved merely because they are globally similar.

---

## 13. Reconsolidation

When a memory is retrieved and used, the system should update metadata:

```text
last_accessed
access_count
task_contexts_used_in
success_after_use
corrections_after_use
reinforced_edges
weakened_edges
```

If a retrieved memory contributes to a bad response or user correction, the memory should be flagged for possible repair:

```text
mis-scoped
confounding
stale
overgeneralized
contradicted
low utility
```

Retrieval telemetry is a maintenance signal.

Retrieval telemetry must not imply that every retrieved atom helped. A packet
item that was present in context but did not materially affect the decision
should be recorded as neutral exposure, for example `label = observed` and
`use_status = context_only`. Only memories that changed the answer, selected
field, safety decision, or explanation should receive positive helpful refs.
Memories that caused or contributed to a bad decision should receive correction
or unhelpful refs.

---

## 14. The Memory Steward

The Memory Steward is an internal Amos process responsible for memory health. It does not answer the user directly.

Responsibilities:

```text
distill
deduplicate
reconcile
link
promote
demote
archive
delete
detect contradictions
detect stale memories
detect orphan memories
detect confounding memories
update retrieval indexes
track memory health
```

Action space:

```text
propose_atom
merge_atoms
split_atom
link_atoms
unlink_atoms
promote_atom
demote_atom
archive_atom
delete_atom
mark_stale
mark_contradicted
mark_confounded
request_review
rebuild_index
```

Self-maintenance must be auditable and reversible when possible.

---

## 15. Maintenance loops

Maintenance should be continuous but not always expensive.

```text
Real-time hygiene:
  cheap updates during interaction

Post-episode consolidation:
  after task/session boundaries

Scheduled memory metabolism:
  periodic graph and storage cleanup

Triggered audits:
  when contradictions, failures, or corrections occur
```

This avoids the expensive anti-pattern:

```text
store lots of English → later summarize everything → repeat forever
```

---

## 16. Lifecycle and health statuses

Every atom has both a `lifecycle_state` and a `health_status`, using the canonical vocabulary in section 7.1.9.

```text
lifecycle_state:
  where the record lives in storage and retrieval lifecycle

health_status:
  whether the memory is currently useful, coherent, scoped, and safe to retrieve
```

Lifecycle controls default eligibility:

```text
proposed:
  hidden from normal retrieval; visible to steward and reviewer workflows

active:
  eligible for normal retrieval subject to health and access policy

archived:
  hidden from normal retrieval; visible for audit, history, or explicit deep recall

tombstoned:
  not retrievable as memory content; checked during atomization to prevent disallowed recreation

deleted:
  not retrievable; retained only as policy allows for audit metadata
```

Health status modifies eligibility:

```text
healthy:
  no additional suppression

stale:
  retrieve only when timeline/history or refresh is relevant

contradicted:
  retrieve only with conflict context

orphaned:
  exclude until relinked or explicitly requested

confounding:
  actively suppress unless specifically requested

review_required:
  exclude from autonomous high-impact use
```

---

## 17. Cleanup categories

### 17.1 Stale memory

A memory that may once have been valid but is no longer likely to be valid.

Signals:

```text
age
new contradictory evidence
low recent access
domain volatility
changed external conditions
user correction
project status change
```

Actions:

```text
active → stale
active → archived
active belief replaced by newer belief, old retained historically
```

### 17.2 Redundant memory

A duplicate or near-duplicate.

Actions:

```text
merge atoms
preserve all evidence refs
keep scoped variants if meaningfully different
```

### 17.3 Incoherent memory

A malformed, underspecified, or semantically unclear memory.

Examples:

```text
user prefers it
project is about that memory thing
the repo is important
```

Actions:

```text
repair missing entity/scope/evidence
split overloaded content
demote to evidence-only
archive as low utility
```

### 17.4 Orphan memory

A memory with no useful graph links.

Signals:

```text
no evidence edge
no entity edge
no task/project edge
no retrieval history
no activation path
```

Actions:

```text
try relinking
archive or suppress if relinking fails
```

### 17.5 Confounding memory

A memory that is not necessarily false but causes bad retrieval, reasoning, or generalization.

Examples:

```text
Generic memory:
  user prefers concise responses

Current context:
  user asks for deep iterative architecture design

Bad outcome:
  system gives too short an answer
```

Actions:

```text
narrow scope
add inhibition rule
mark context-sensitive
link to exceptions
demote from global preference to scoped preference
```

### 17.6 Contradictory memory

Contradictions should be resolved through evidence, recency, and scope—not deletion by default.

Example:

```text
A: user prefers concise answers
B: user prefers detailed design exploration
```

Resolution:

```text
concise answers:
  scope: simple factual questions

detailed design exploration:
  scope: architecture/planning sessions
```

### 17.7 Overgeneralized memory

A memory whose scope is too broad.

Example:

```text
Evidence:
  user asked not to write code in this design conversation

Bad memory:
  user does not want code

Better memory:
  user prefers no implementation code during early-stage conceptual architecture planning unless requested
```

Actions:

```text
narrow scope
lower confidence
link to source episode
require reinforcement before global promotion
```

### 17.8 Low-utility memory

True but not useful enough for hot storage.

Actions:

```text
demote to cold archive
remove from hot indexes
retain only if needed for audit
```

---

## 18. Maintenance jobs

### 18.1 Atomizer

Turns raw events into candidate atoms.

### 18.2 Entity resolver

Normalizes references.

Example:

```text
ALucek repo
agentic-memory GitHub repo
https://github.com/ALucek/agentic-memory
→ entity:github_repo:ALucek/agentic-memory
```

### 18.3 Deduplicator

Finds exact and semantic duplicates.

### 18.4 Contradiction detector

Finds incompatible active claims.

### 18.5 Scope refiner

Fixes overgeneralization.

### 18.6 Linker

Creates cross-category associations.

### 18.7 Promoter

Moves memories upward when stable and useful.

### 18.8 Demoter

Moves memories downward when stale or low-utility.

### 18.9 Confounder detector

Finds memories that harm answer quality.

Signals:

```text
retrieved before bad outcome
followed by user correction
frequently co-retrieved but irrelevant
causes wrong style/plan/tool choice
overrides more specific memory
```

### 18.10 Index maintainer

Rebuilds derived views:

```text
embeddings
keyword indexes
graph neighborhoods
processor-specific packet caches
```

### 18.11 Self-model calibrator

Maintains self-awareness records.

Responsibilities:

```text
verify capability atoms against current runtime state
mark stale capabilities as unavailable or degraded
promote repeated limitations into durable LimitationAtoms
compare stated capability claims against actual tool outcomes
track commitments made versus commitments satisfied
generate SelfAssessmentRecords after task boundaries
flag overconfident self-reports for review
```

The self-model calibrator should run after user corrections, tool failures, permission denials, environment changes, and task completion.

### 18.12 Agentic recall auditor

Maintains agentic recall quality and attribution balance.

Responsibilities:

```text
construct AgenticTrace records from decisions, actions, tool events, outcomes, and corrections
construct ActionOutcomeRecords for successes, failures, partial results, and blocked actions
verify agency attribution against evidence and shared responsibility overlays
detect self-serving over-attribution in self-awareness packets and SelfNarratives
ensure failures, corrections, limitations, and external constraints remain retrievable
flag SelfNarratives that drift from canonical traces or omit material counterevidence
update agentic recall metrics after retrieval outcomes and task boundaries
```

The agentic recall auditor should run after task completion, user correction, failed tool use, shared-view refresh, self-assessment submission, and generation of any SelfNarrative.

---

## 19. Maintenance journal

Every maintenance action should be logged.

```text
MaintenanceJournalEntry
  action
  target_atom_ids
  reason
  evidence_refs
  before_state
  after_state
  confidence_delta
  performed_by
  review_status
  reversible
```

This enables the system to answer:

```text
Why did you remember this?
Why did you forget that?
Why did this preference change?
Which episode caused this procedure?
What evidence supports this belief?
```

Without a maintenance journal, self-maintenance becomes invisible self-corruption.

---

## 20. Maintenance safety levels

```text
Low-risk:
  update last_accessed
  increment use count
  add weak association edge
  mark proposed atom as underspecified

Medium-risk:
  merge duplicates
  promote proposed atom to active scoped belief
  demote low-utility memory
  rewrite generated summary view

High-risk:
  delete evidence
  alter user preference
  promote procedure
  mark active belief false
  globalize a scoped memory
```

High-risk actions should require stronger checks:

```text
multiple evidence sources
no unresolved contradiction
policy validation
possible user confirmation or human review
```

---

## 21. Processor-facing views

Different processors should receive different views of the same canonical memory.
Receiving a processor-facing view or being selected by `target_processor` does
not confer durable agent identity. A processor is a replaceable cognitive
function unless the application explicitly models it as a distinct durable
agent with its own `agent_id`, evidence, and lifecycle.

### 21.1 Reasoner view

Needs:

```text
relevant facts
uncertainties
conflicts
source quality
recent context
high-level summaries
```

### 21.2 Planner view

Needs:

```text
active goals
constraints
deadlines
dependencies
available procedures
known risks
state of open loops
```

### 21.3 Executor view

Needs:

```text
tool affordances
action schemas
permission boundaries
known failure modes
recovery procedures
recent execution outcomes
```

### 21.4 Critic/steward view

Needs:

```text
evidence links
maintenance history
retrieval telemetry
contradictions
health flags
policy constraints
```

### 21.5 Self-awareness view

Needs:

```text
current agent identity
current role and delegated authority
active objectives and owner scopes
open commitments and due conditions
available capabilities and verification freshness
known limitations and permission boundaries
current runtime state snapshot
recent errors, corrections, and failed assumptions
recent agentic traces and action outcomes
agency attribution confidence
counterevidence for strong self-claims
uncertainties that should be disclosed
confidence calibration and self-assessment records
evidence for capability or limitation claims
```

The self-awareness view must distinguish:

```text
durable self-knowledge:
  stable agent role or purpose, policy, durable capabilities, recurring limitations

current runtime state:
  tools currently available, sandbox state, active task, budgets, recent errors

self-report rendering:
  human-readable explanation generated from structured self-knowledge and runtime state

self-narrative rendering:
  generated explanation of agentic patterns, trace-backed actions, outcomes,
  corrections, limitations, and unresolved counterevidence
```

An agent should not claim a capability merely because it appears in durable memory. Capability claims must be checked against current runtime state and access policy.

An agent should not claim ownership of an outcome merely because the outcome appears in shared memory. Agency claims must be checked against AgenticTrace evidence, ActionOutcomeRecords, responsibility overlays, and external constraints.

### 21.6 Shared coordination view

Needs:

```text
shared task context
shared goals and priorities
shared constraints and policies
shared commitments and owners
common assumptions
known conflicts or disagreements
audience membership
per-agent responsibility overlays
per-agent agency attributions
shared-system action outcomes
per-agent access omissions
graph_version and freshness
```

The shared coordination view is the common operating picture for a multi-agent team. It should not include private self-model details, hidden evidence, or tool permissions unless policy allows those details to be shared with the audience.

---

## 22. Storage strategy

### 22.1 Raw history is cold

Raw transcripts, logs, files, tool traces, and long documents should be compressed and stored in the Evidence Archive.

### 22.2 Structured memory is hot/warm

The active memory graph contains compact atoms, edges, and pointers.

### 22.3 Embeddings are derived

Embeddings should be treated as disposable model-specific indexes.

Each embedding should know:

```text
embedding_model
embedding_version
source_atom_ids
created_at
quantization_method
```

### 22.4 English is a rendering

Generated summaries and prompt snippets should be rebuildable from canonical atoms and evidence.

---

## 23. Memory metabolism

The whole system can be understood as memory metabolism:

```text
Ingest
  capture raw events

Digest
  atomize and extract candidate memories

Assimilate
  validate, link, and commit useful atoms

Circulate
  retrieve and render memory packets during tasks

Reinforce
  strengthen memories and edges that help

Repair
  fix incoherent, orphaned, contradicted, or confounding memories

Consolidate
  promote repeated patterns into long-term belief/procedure/semantic memory

Excrete
  demote, archive, or delete low-value or unsafe material
```

---

## 24. Memory health metrics

Candidate operational metrics:

```text
stale_atom_ratio
orphan_atom_ratio
duplicate_cluster_count
contradiction_count
unresolved_conflict_age
average_evidence_links_per_active_atom
retrieval_success_rate
retrieval_correction_rate
confounding_memory_rate
hot_memory_growth_rate
promotion_precision
demotion_reversal_rate
summary_drift_rate
index_freshness
storage_bytes_per_active_memory
tokens_per_memory_packet
self_report_accuracy
capability_staleness_rate
limitation_retrieval_rate
commitment_followthrough_rate
uncertainty_disclosure_rate
overconfident_claim_rate
agentic_recall_balance
self_narrative_drift_rate
agency_overattribution_rate
ignored_failure_rate
correction_integration_rate
```

These metrics can drive automatic maintenance jobs.

Examples:

```text
orphan_atom_ratio > threshold:
  run linker job

retrieval_correction_rate increases:
  run confounder audit

hot_memory_growth_rate too high:
  run demotion/compaction

summary_drift_rate high:
  rebuild generated views from canonical atoms

capability_staleness_rate high:
  run self-model calibrator

overconfident_claim_rate high:
  suppress stale capability claims and require self-assessment review

agentic_recall_balance low:
  require failures, blocked actions, limitations, and corrections in agentic recall packets

self_narrative_drift_rate high:
  expire SelfNarratives and rebuild them from canonical traces

agency_overattribution_rate high:
  downgrade unsupported primary responsibility claims and run attribution audit

ignored_failure_rate high:
  boost recent failures, corrections, and limitations in self-awareness retrieval

correction_integration_rate low:
  run self-model calibrator and agentic recall auditor before generating new SelfNarratives
```

---


## 25. Distributed Amos instance

Amos is intended to operate as a shared memory plane for multi-process agentic systems. A realistic agentic runtime may have separate reasoner, planner, executor, critic, tool-worker, user-interface, evaluator, and steward processes. All of those processes should connect to the same logical Amos instance for memory access.

Amos should therefore be designed as a networked memory service, not only as an in-process library.

Logical instance boundary:

```text
One logical Amos instance serves one coordinated system of agents.

Within that system:
  agents share canonical memory, shared task context, goals, commitments,
  evidence pointers, procedures, constraints, and shared memory views

Each durable agent keeps:
  its own agent_id, SelfModelAtom, CapabilityAtoms, LimitationAtoms,
  commitments, lineage, and agent-scoped packet overlays

Each connected service or processor keeps:
  its own ClientIdentity, processor_id, permissions, RuntimeStateSnapshots,
  model/substrate metadata, and processor-specific packet overlays
```

Physical deployment is separate from logical instance identity. One physical Amos cluster may host many logical Amos instances, but unrelated agent systems should be separated by tenant/workspace/project scope and access policy so their memories and self-models do not contaminate one another.

```text
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Reasoner Service │   │ Planner Service  │   │ Executor Service │
└─────────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
          │                      │                      │
          └──────────────┬───────┴──────────────┬───────┘
                         ▼                      ▼
              ┌────────────────────────────────────┐
              │ Amos Memory Plane                  │
              │ shared API, auth, retrieval,       │
              │ lifecycle, consistency, stewardship│
              └────────────────────────────────────┘
                         │
          ┌──────────────┼────────────────┬──────────────┐
          ▼              ▼                ▼              ▼
┌────────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
│ Event Journal  │ │ Memory Graph │ │ Indexes    │ │ Evidence     │
│ append-only    │ │ canonical    │ │ derived    │ │ Archive      │
└────────────────┘ └──────────────┘ └────────────┘ └──────────────┘
```

A logical Amos instance may be implemented as a single service, replicated service, or cluster, but it presents one authoritative memory plane per configured tenant, workspace, project, and agent scope.

### 25.1 Service APIs instead of direct storage mutation

External agent processes should not directly mutate the underlying graph database, evidence store, vector index, or journal. They should call Amos APIs.

V1 API surface:

```text
POST /v1/events:capture
POST /v1/atoms:propose
POST /v1/atoms:commit
POST /v1/atoms:archive
POST /v1/atoms:merge
POST /v1/packets:retrieve
POST /v1/retrieval-outcomes
POST /v1/maintenance:request
GET  /v1/maintenance-processors
GET  /v1/memory-policy
POST /v1/memory-policy:configure
POST /v1/memory-policy:run
POST /v1/maintenance-distiller:run
POST /v1/deletion-requests
POST /v1/runtime-state
POST /v1/self-assessments
POST /v1/self-awareness:retrieve
POST /v1/agentic-recall:retrieve
POST /v1/shared-views:retrieve
POST /v1/shared-views:refresh
POST /v1/procedures:execution-policy
POST /v1/capacity:configure
POST /v1/smp:analyze
GET  /v1/health/memory
GET  /v1/health/capacity
GET  /v1/llm-reviewer/policy
GET  /v1/verify
```

The boundary is:

```text
Agent services request memory operations.
Amos owns memory validity, structure, lifecycle, maintenance, and audit.
```

Common request envelope:

```text
AmosRequest
  request_id
  client_identity
  scope
  idempotency_key
  expected_versions
  consistency_requirement: strong | monotonic | eventual
  payload
```

Common response envelope:

```text
AmosResponse
  request_id
  result_status: accepted | rejected | conflict | partial | error
  graph_version
  event_ids
  warnings
  errors
  payload
```

The v1-local stdlib HTTP adapter currently returns the service method payloads
directly rather than wrapping every response in `AmosResponse`. The envelope
above remains the stable client contract target for generated clients and later
production adapters.

The same adapter is the first usable single-process deployment profile. It owns
one SQLite store and serializes service calls behind one process-local lock. That
is correctness-first and simple, but it provides no parallel read throughput for
concurrent retrieval calls. Higher-load deployments should move to a
reader/writer concurrency model, SQLite WAL read parallelism, or a production
database adapter while preserving one linearizable writer per shard.

V1 endpoint contracts:

```text
POST /v1/events:capture
  request payload: SourceEvent or SourceEvent batch
  response payload: EvidenceRecord refs, captured event refs, journal event refs
  consistency: strong for accepted source events

POST /v1/atoms:propose
  request payload: candidates, optional actor, optional scope
  response payload: committed proposed atoms with lifecycle_state proposed,
  graph_version
  consistency: strong for the persisted proposed atoms

POST /v1/atoms:commit
  request payload: full atom record or atoms batch, optional idempotency_key,
  authorization_context
  response payload: committed atom refs, graph_version, projection_status
  consistency: strong; batch commits validate duplicate ids before mutation and
  commit journal entries, atoms, and edges in one transaction

POST /v1/atoms:archive
  request payload: atom_id, reason, optional expected_version, authorization_context
  response payload: archived atom ref, journal event ref
  consistency: strong

POST /v1/atoms:merge
  request payload: source_refs, merged type/payload, scope, approved_by
  response payload: merged atom ref, archived source refs, journal event refs
  consistency: strong when approved

POST /v1/packets:retrieve
  request payload: MemoryPacketRequest
  response payload: MemoryPacket; HTTP service mode includes a policy_schedule
  acknowledgement when retrieval queues background policy work
  consistency: monotonic by default; strong if min_graph_version is provided and reachable;
  cacheable by request digest and graph_version when no policy mutation is required

POST /v1/retrieval-outcomes
  request payload: packet_id, original retrieval request, outcome labels,
  used_item_refs, helpful_atom_refs, correction_refs, unhelpful_atom_refs,
  optional use_status such as used, context_only, ignored, or unused
  response payload: retrieval outcome record refs and created_at
  consistency: eventual

POST /v1/maintenance:request
  request payload: target_refs, action_type, reason_code, risk_level
  response payload: maintenance event refs, review requirement, accepted action refs
  consistency: strong for high-risk actions, eventual for low-risk proposals

GET /v1/maintenance-processors
  response payload: registered processor ids and versions
  consistency: monotonic

GET /v1/memory-policy
  response payload: configured policy, persisted policy state, due reasons,
  graph_version, background worker status in HTTP service mode
  consistency: monotonic

POST /v1/memory-policy:configure
  request payload: enabled flag, schedule overrides, maintenance overrides,
  distillation overrides, maintenance_distiller overrides
  response payload: effective policy and policy status
  consistency: strong

POST /v1/memory-policy:run
  request payload: optional force flag, trigger, scope
  response payload: policy tick status,
  SMP/steward/distillation/maintenance_distiller/index/cache results, journal
  event ref when completed
  consistency: strong when a tick runs; this endpoint is the synchronous
  operator/admin path

POST /v1/maintenance-distiller:run
  request payload: scope, domain, optional processor_ids, window limits,
  auto_commit_low_risk, reviewer
  response payload: shared and per-processor evidence-window/coverage summaries,
  selected processors, proposal records, committed low-risk atom or edge refs,
  deferred review items, reviewer status, journal event ref
  consistency: strong when a tick runs

POST /v1/deletion-requests
  request payload: DeletionRequest
  response payload: tombstone refs, deleted refs, residual_retention report
  consistency: strong

POST /v1/runtime-state
  request payload: agent_id, capabilities, denied_capabilities, constraints,
  load, optional scope and actor
  response payload: committed runtime_state atom and journal event
  consistency: strong for the committed runtime_state atom

POST /v1/self-assessments
  request payload: agent_id, claim, calibration, optional scope and actor
  response payload: committed self_assessment atom and journal event
  consistency: strong for the committed self_assessment atom

POST /v1/self-awareness:retrieve
  request payload: agent_id, optional scope, requester, target_processor
  response payload: self_awareness view containing self_model, capabilities,
  limitations, open_commitments, runtime_state, assessments, calibration,
  omissions, conflicts, graph_version, and source_packet_id
  consistency: monotonic by default; strong when current runtime claims are included

POST /v1/agentic-recall:retrieve
  request payload: agent_id, optional cues, scope, requester, target_processor
  response payload: agentic_recall view containing successes, failures, blocked
  actions, corrections, traces, responsibility-classified actions, external
  constraints, material counterevidence, self narratives, omissions, conflicts,
  graph_version, and source_packet_id
  consistency: monotonic by default; strong when current runtime claims or active
  shared-view responsibility overlays are included

POST /v1/shared-views:retrieve
  request payload: processor_ids, optional cues, scope, requester, max_items
  response payload: shared_memory view with common_graph_version, common items,
  per_processor_overlays, omissions_by_identity, and source_packets
  consistency: monotonic_shared by default

POST /v1/shared-views:refresh
  request payload: processor_ids, optional cues, scope, requester, max_items
  response payload: refreshed shared_memory view with refresh_status
  consistency: monotonic_shared by default

POST /v1/procedures:execution-policy
  request payload: procedure_ref, autonomous, approved_by, tool_permission_binding,
  preconditions_satisfied, rollback_plan, review_status
  response payload: advisory/executable eligibility, required approvals,
  denial reasons, and policy notes
  consistency: monotonic

POST /v1/capacity:configure
  request payload: hard_capacity_bytes, warning_ratio, critical_ratio
  response payload: accepted capacity budget
  consistency: strong

POST /v1/smp:analyze
  request payload: scope, optional target refs
  response payload: deterministic SMP envelope outputs and review requirements
  consistency: monotonic over the source graph version

GET /v1/health/memory
  response payload: memory health metrics by scope, lifecycle_state, health_status, atom type, and background worker status in HTTP service mode
  side effects: observational in HTTP service mode; does not run policy inline

GET /v1/health/capacity
  response payload: CapacityHealthReport

GET /v1/llm-reviewer/policy
  response payload: reviewer default state, allowed uses, forbidden uses,
  required output envelope
  consistency: monotonic

GET /v1/verify
  response payload: journal chain verification and replay verification status
  consistency: strong for the local store snapshot
```

Error codes:

```text
schema_invalid
authorization_denied
scope_denied
evidence_denied
idempotency_conflict
expected_version_conflict
review_required
retention_policy_denied
capacity_limited
projection_failed
derived_index_stale
```

HTTP service implementations may also return a transport-level transient
failure envelope:

```json
{
  "status": "error",
  "error": "database is locked",
  "retryable": true
}
```

In v1, service shutdown and SQLite locked/busy conditions return HTTP 503 with
`retryable: true`. Clients own bounded retry timing and delayed requeue state;
AMOS does not encode client-domain lifecycle decisions for these failures.
Clients should use exponential backoff with jitter and must reuse a stable
idempotency key, actor, and payload when retrying a mutation. Validation,
authorization, version, and idempotency conflicts are not transient transport
failures and must not be retried blindly.

#### 25.1.1 Client integration contract

AMOS owns canonical memory semantics, lifecycle, maintenance, retrieval
diagnostics, and audit. Client systems own domain interpretation, runtime
authority, prompt rendering, producer-normalized canonical graph metadata, and
domain-specific maintenance processors.

A production client should treat AMOS as a memory service, not as a prompt log:

```text
client responsibilities:
  capture evidence-backed traces, outcomes, corrections, and runtime state
  retrieve bounded role/task/scope packets
  retry explicitly retryable service failures with bounded backoff and idempotent writes
  render concise operational prompt digests from packets
  enforce application schemas, permissions, control registries, and guardrails
  record whether retrieved memories were materially used
  keep full packets and rendered prompts in client telemetry for audit

AMOS responsibilities:
  validate and journal canonical records
  retrieve scope- and access-filtered packets
  disclose omissions, conflicts, degradation, and attention traces
  update utility/salience from retrieval outcomes
  run deterministic maintenance and registered processor packs
  commit low-risk derived memories and active-endpoint edges through policy gates
  preserve explicit derivation provenance for every graph edge
```

Client-specific cleanup and learning should live in client packages as
registered maintenance processors. For example, a training harness, coding
agent, or support bot may promote recurring role experiences into capability,
limitation, procedure, or semantic atoms. AMOS should provide the generic
proposal and policy machinery; it should not encode that client's domain rules
in core.

A client-specific processor is not required merely to construct the graph. If
the producer already knows the typed semantics, it should attach canonical
`payload.semantic_facets` and `payload.graph_relations`; the built-in generic
processor validates those structures and proposes governed edges. External
processors remain appropriate for domain-specific aggregation, calibration,
causal review, or legacy payloads that cannot emit the canonical contract.
Canonical relation projections inherit evidence and confidence from their
owning atom unless the relation supplies a narrower provenance set. Every edge
also names its derivation path; legacy rows are migration-classified without
inventing a historical producer.

Processors may request a narrower `MaintenanceWindowRequest` before execution.
Lifecycle, atom-type, producer-profile, neighbor, evidence/event/outcome, and
size fields are workset hints; AMOS still enforces the caller scope and ceilings.
Coverage reports distinguish visible candidates, selected/truncated records,
internal and boundary edges, and resolved/missing evidence. Client processors
should use explicit producer hints and cohort keys rather than infer semantic
equivalence from prose.

Proposal-queue maintenance is likewise generic when the producer supplies an
explicit `payload.proposal_retention` contract. AMOS may deterministically
archive same-scope/type proposals sharing the producer's stable deduplication
key, honor a proposal TTL, and enforce distinct lifecycle-active and proposed
quotas. It must not infer semantic duplication from prose or treat proposal
cleanup as permission to discard a client's independent occurrence journal.

Prompt rendering should keep these sources distinct:

```text
static contract:
  bootstrap role description, stable policy text, schema requirements

self-awareness packet:
  current self-model, capabilities, limitations, commitments, runtime state

experience profile:
  recurring demonstrated capabilities, recurring limitations, and reuse
  guidance distilled from action outcomes

retrieved memory packet:
  task-specific prior memories, policies, counterevidence, and citation
  candidates selected for the current decision
```

The generated prompt should be compact and operational. More memory is not
better when it causes the model to ignore current task authority or turns AMOS
into a logging sink. The model should cite atom refs only when a memory
materially changes a decision, selected field, explanation, or safety check. If
memory was retrieved but not used, the client should record neutral
`context_only` telemetry instead of positive retrieval feedback.

#### 25.1.2 Durable agent identity and replaceable cognitive processors

The processor used to interpret a packet is not the subject represented by the
packet. AMOS integrations must maintain the following boundary:

```text
agent_id:
  durable subject of self-model, memory, commitments, autobiography, and lineage

processor_id / target_processor:
  functional processing role selected for an invocation

client_identity:
  authenticated service or process actor and its authority

model_profile:
  provider, model, checkpoint, weights, quantization, prompt, runtime, and other
  replaceable cognitive-substrate metadata
```

An LLM invocation may receive a bounded packet and use ephemeral inference
state, but it must be stateless with respect to durable identity, memory
authority, commitments, and cross-session continuity. Those properties live in
the canonical AMOS state and the integrating agent runtime. Processor-local
state must not silently become canonical memory.

When an LLM writes in the first person, that voice is delegated by the active
agent. The prompt and response-handling path must identify the agent as the
speaker and the model as a cognitive processor. They must not import the model
provider's persona, model name, training narrative, or self-description into the
agent's role, purpose, personality, biography, capabilities, or limitations.
Substrate-specific constraints may be recorded separately as model or runtime
metadata; they become agent-level learnings only through independent evidence
and the normal promotion policy.

Generated output, including prior chat output, is a fallible expression rather
than canonical truth or independent evidence about the agent. Any LLM-derived
memory or self-model mutation must be a provenance-bearing, evidence-linked
proposal subject to schema validation, authorization, contradiction handling,
review policy, journaling, and lifecycle controls. The model must never promote
its own identity claim merely because it generated the claim.

Replacing or upgrading an LLM is a processor-substrate migration. The
integration may update model metadata and observe new runtime capabilities or
limitations, but it must preserve `agent_id`, lineage, commitments, and the
existing self-model unless ordinary evidence-backed memory policy justifies a
change.

### 25.2 Client identity and capabilities

Every connected process should authenticate as a specific client identity.

```text
ClientIdentity
  service_id
  agent_id
  process_type
  tenant_id
  workspace_id
  project_id
  user_id, if applicable
  capabilities
  trust_level
```

Capabilities should be operation-specific.

```text
Reasoner:
  read memory packets
  propose candidate beliefs
  record retrieval outcomes

Planner:
  read goals, commitments, constraints, procedures
  propose goal and plan-state updates

Executor:
  read procedures, permissions, tool constraints
  write tool events and execution outcomes

Critic:
  read evidence, outputs, retrieval telemetry
  flag contradictions, confounders, and failures

Memory Steward:
  perform maintenance actions
  promote, demote, merge, archive, relink

User interface:
  read explainable memory
  request correction, deletion, pinning, or approval
```

### 25.3 Shared write path through EventJournalEntry records

All memory mutations should flow through a structured append-only event stream.

```text
EventJournalEntry
  event_id
  event_type
  schema_version
  actor
  target_refs
  payload
  payload_refs
  evidence_refs
  idempotency_key
  causal_parent_ids
  expected_versions
  authorization_context
  occurred_at
  accepted_at
  result_status
  projection_status
  checksum
  previous_event_hash
```

The canonical memory graph is a validated projection of accepted Event Journal entries.

```text
client command
  -> validate shape
  -> authorize against scope and capability policy
  -> check expected_versions and idempotency_key
  -> append EventJournalEntry
  -> project canonical graph update
  -> publish change event
  -> update derived indexes and packet caches asynchronously
```

For strong-consistency operations, append and canonical graph projection must commit atomically from the caller's perspective. Derived indexes, packet caches, and telemetry aggregation may lag but must record the graph version they reflect.

Rejected commands should either return no journal entry or append a `command.rejected` audit event, depending on deployment policy. User correction, deletion, privacy, and high-risk maintenance rejections should be journaled with a redacted reason code.

### 25.4 Concurrency control

Multiple services may update the same belief, goal, procedure, association, or memory health state. Amos must provide concurrency rules:

```text
optimistic versioning
compare-and-swap updates
idempotency keys
transaction boundaries
causal ordering
maintenance leases
conflict queues
```

Canonical objects should include versions:

```text
MemoryAtom
  id
  version
  revision_history
  last_writer
  last_write_event_id
```

Writes should be conditional when they affect existing canonical state:

```text
update atom X only if version == expected_version
```

The `idempotency_key` is scoped to:

```text
tenant_id
workspace_id
actor service_id
operation_type
target_refs
```

Reusing an idempotency key with a different payload is an error. Reusing it with the same payload returns the original result.

Compare-and-swap conflicts should not silently retry for high-risk mutations. They should produce a conflict result that includes:

```text
target_ref
expected_version
actual_version
conflicting_event_id
recommended_action: refresh | merge | review | retry
```

### 25.5 Consistency levels

Not all memory operations need the same consistency model.

Stronger consistency is required for:

```text
user corrections
active preferences
policy constraints
goal_status changes
commitments
procedure promotion
deletion requests
privacy-sensitive memory
```

Eventual consistency is acceptable for derived artifacts:

```text
embedding indexes
semantic similarity edges
retrieval co-activation edges
low-risk association weights
summary caches
packet caches
cold archive compaction
```

Principle:

```text
Canonical memory changes need consistency.
Derived views can lag.
```

Minimum guarantees:

```text
read-your-writes:
  a client that successfully commits a strong operation can request a packet at or after that graph_version

monotonic packet reads:
  a client may ask Amos not to return an older graph_version than a previously observed packet

bounded stale derived views:
  packet items must disclose when an embedding, summary, or association score was computed from an older graph_version

atomic strong mutation:
  journal append and canonical graph projection commit together for user corrections,
  policy constraints, active preferences, commitments, procedure promotion, privacy, and deletion

eventual derived mutation:
  vector indexes, cache entries, telemetry rollups, and weak association weights may update asynchronously
```

If Amos is deployed as a cluster, the spec requires one linearizable writer per tenant/workspace shard for strong canonical mutations, or an equivalent consensus mechanism. The deployment may use weaker consistency only for derived artifacts and low-risk telemetry.

V1-local packet caches are valid only for the exact request signature and
`graph_version` that produced them. Retrieval should check the packet cache
before ranking atoms, but cache misses must produce the same correctness
semantics as an uncached retrieval. Cache entries and materialized search
metadata are discarded or rebuilt after canonical graph mutations, deletion
requests, merges, health transitions, and policy maintenance that changes
retrieval eligibility.

Stored search vectors are derived caches. Request-time retrieval may use a
stored vector whose IDF or latent-vector model lags the current graph version so
packets stay bounded-latency; direct index refresh paths may recompute from
canonical atom text when exact freshness is required. Periodic maintenance
refreshes the graph-versioned IDF/LSA model and records the graph version of the
derived index.

### 25.6 Pub/sub and cache invalidation

Connected services may maintain local caches, but Amos remains authoritative. Amos should publish change events such as:

```text
atom_committed
atom_updated
atom_deleted
atom_merged
memories_distilled
retrieval_outcome_recorded
steward_run
maintenance_distillation_run
memory_policy_run
index_rebuilt
memory_health_alert
```

Clients use these events to invalidate local memory packets, active procedure snippets, or retrieved context caches. V1-local retrieval outcomes are stored as telemetry records and, when they reference atoms, also journal `retrieval_outcome_recorded` mutations that update atom utility, salience, `last_accessed`, and retrieval telemetry counters.

### 25.7 Scope isolation and authorization model

A shared Amos instance should support explicit scopes:

```text
tenant
workspace
project
user
agent
session
task
tool environment
```

Shared memory does not imply global leakage. Retrieval and mutation should always be evaluated against scope and access policy.

Canonical scope tuple:

```text
Scope
  tenant_id
  workspace_id
  project_id
  user_id
  agent_id
  session_id
  task_id
  tool_environment_id
```

Scopes are hierarchical for lookup but not for permission by default. A project-scoped atom can be visible to a workspace process only when the atom access policy allows cross-project or workspace-level retrieval.

Access policy shape:

```text
AccessPolicy
  owner_scope
  allowed_identities
  allowed_capabilities
  denied_identities
  denied_capabilities
  processor_visibility
  evidence_visibility
  mutation_policy
  retention_policy_ref
  audit_level
```

Authorization precedence:

```text
1. explicit denial wins
2. user deletion/privacy constraint wins over lower-priority retention
3. compliance/legal hold can prevent physical deletion but not ordinary retrieval suppression
4. more-specific scope policy overrides less-specific allow rules
5. high-risk mutation requires both capability and trust_level threshold
6. read permission does not imply mutation permission
7. evidence read permission is evaluated separately from atom read permission
```

Retrieval authorization and mutation authorization are separate checks:

```text
retrieve_memory_packet:
  filter candidate atoms, edges, evidence, and rendered fields by access policy

commit_memory_atoms:
  validate actor capability, target scope, evidence visibility, mutation risk level,
  expected_versions, and required review status
```

Every access denial in a packet should be summarized as an omission reason without leaking the denied content.

### 25.8 Distributed Memory Steward leases

The Memory Steward may itself be implemented as multiple workers. Steward jobs should acquire leases to prevent duplicate or conflicting maintenance.

```text
MaintenanceLease
  job_type
  target_scope
  acquired_by
  expires_at
  heartbeat
  status
```

Maintenance jobs can be partitioned by tenant, project, memory type, time range, health status, or graph neighborhood.

### 25.9 Cross-process conflict resolution

Distributed services will sometimes produce conflicting candidates. Amos should resolve them through:

```text
source priority
explicitness
scope
recency
evidence strength
retrieval outcome
policy constraints
user correction priority
```

Conflicts should usually produce scoped coexistence, contradiction state, or review requests rather than silent deletion.

---

## 26. Event journal, retention, and compaction

The append-only event log is the **Amos Event Journal**. It is the authoritative application-level mutation record for memory-relevant events and accepted state changes.

The Event Journal is logically append-only for correctness and auditability, but it is not an infinite hot store. Long-term scalability requires checkpoints, snapshots, segment compaction, evidence tiering, retention policies, memory budgets, edge pruning, telemetry aggregation, and deletion/tombstoning.

### 26.1 Journal terminology

Amos should distinguish three related records:

```text
Event Journal:
  authoritative append-only memory event stream

Maintenance Journal:
  typed subset or view of the Event Journal focused on stewardship actions

Evidence Archive:
  raw source material: transcripts, tool traces, files, documents, logs
```

All maintenance journal entries are event journal entries. Not all event journal entries are maintenance journal entries.

The Event Journal is also distinct from a database write-ahead log. A database WAL records storage-level changes. The Amos Event Journal records application-level memory semantics.

Examples:

```text
evidence_captured
atom_committed
atom_updated
atom_deleted
atom_merged
memories_distilled
steward_run
maintenance_distillation_run
storage_cleanup_run
memory_policy_run
```

Canonical stored shape:

```json
{
  "event_id": "evt_<uuid>",
  "event_type": "atom_committed",
  "schema_version": "amos.v1",
  "actor": "svc:memory_steward",
  "target_refs": ["atm_<ulid>"],
  "payload": {
    "operation": "commit_atom",
    "atom_id": "atm_<ulid>",
    "atom_type": "belief",
    "atom_version": 1
  },
  "payload_refs": [],
  "evidence_refs": ["evd_<ulid>"],
  "idempotency_key": "idem_<stable_key>",
  "causal_parent_ids": ["evt_<parent_uuid>"],
  "expected_versions": {
    "atm_<ulid>": 0
  },
  "authorization_context": {
    "tenant_id": "default",
    "workspace_id": "amos",
    "capability": "atom.commit",
    "decision": "allow"
  },
  "occurred_at": "<iso_timestamp>",
  "accepted_at": "<iso_timestamp>",
  "result_status": "accepted",
  "projection_status": "projected",
  "payload_digest": "sha256:<payload_hash>",
  "graph_version": 42,
  "checksum": "sha256:<event_hash>",
  "previous_event_hash": "sha256:<previous_event_hash>"
}
```

V1-local uses one event schema version, `amos.v1`, and stores `actor` as a
stable string identity. Richer actor metadata belongs in `authorization_context`
or in typed atoms/evidence until a later schema version introduces a structured
actor envelope. Event migration must preserve event identity, checksum chain
verification, and the ability to reconstruct the canonical graph at any retained
graph version.

### 26.2 Raw payloads are not stored inline forever

Journal entries should store compact structured events, payload digests, evidence pointers, checksums, and causal links. Large payloads should live in the Evidence Archive.

```text
evidence_captured
  event_id: evt_123
  source_type: user_message
  evidence_pointer: evidence://conversation/<session_id>/turn/<turn_id>
  checksum: ...
  actor: user
  timestamp: ...
```

### 26.3 Hot, warm, and cold journal tiers

```text
Hot journal:
  recent, fully replayable, low-latency

Warm compacted journal:
  older, structured, compressed, segment-level summaries

Cold archive:
  cheap object storage, compressed, rarely accessed

Deleted or shredded:
  expired or user-deleted payloads, with tombstones if needed
```

### 26.4 Snapshots and checkpoints

Amos should periodically snapshot the canonical graph.

```text
events 1..N
  ↓ applied
Canonical Memory Graph at version N
  ↓ snapshot
snapshot_N
```

Normal recovery and operation should require only:

```text
latest snapshot
+ recent hot journal events after snapshot
+ compacted journal rollups
+ cold archived segments only for audit/deep reconstruction
```

### 26.5 Journal segment compaction

After snapshot validation, old journal segments can be sealed and compacted.

Before compaction:

```text
event_1: user said X
event_2: candidate belief proposed
event_3: candidate belief validated
event_4: belief committed
event_5: edge created
event_6: retrieval used belief
event_7: retrieval succeeded
event_8: confidence reinforced
```

After compaction:

```text
compacted_segment_1
  covers_events: event_1..event_8
  resulting_atoms: [belief_123]
  resulting_edges: [edge_991]
  evidence_rollup: evidence_rollup_77
  confidence_delta: reinforced
  final_state_version: 44
  checksum_chain: ...
```

The detailed sequence can move to cold storage or be deleted according to policy.

### 26.6 Memory compaction after commit

Committed memory is not permanently hot. It should continue moving through lifecycle and health states:

```text
lifecycle_state:
  proposed
  active
  archived
  tombstoned
  deleted

health_status:
  healthy
  stale
  redundant
  contradicted
  confounding
  low_utility
```

Examples:

```text
active atom:
  hot graph + indexed

stale atom:
  warm graph or cold archive, not normal retrieval

redundant atom:
  merged into stronger atom, evidence refs preserved

confounding atom:
  suppressed, narrowed, or demoted

low-utility atom:
  archived or deleted
```

### 26.7 Sparse association policy

Associative edges can grow faster than atoms. Amos should enforce sparse graph policies:

```text
maximum edges per atom per edge type
minimum edge weight threshold
decay weak edges over time
materialize only useful associations
avoid all-pairs similarity linking
promote only reinforced edges
prune edges that never help retrieval
```

Protected edge types:

```text
evidence links
contradiction links
supersession links
user-correction links
policy links
```

Aggressively prunable edge types:

```text
weak semantic similarity
one-time co-retrieval
low-confidence causal guesses
old temporal proximity edges
```

### 26.8 Retrieval telemetry aggregation

Retrieval telemetry should not be retained as raw events forever. Older telemetry should collapse into counters and sampled diagnostic examples.

Per atom:

```text
access_count
last_accessed
successful_use_count
correction_after_use_count
failure_after_use_count
recent_context_histogram
utility_score
confounding_score
```

Raw retrieval events should be retained mainly for recent sessions, bad outcomes, user corrections, high-impact decisions, or debugging windows.

### 26.9 Forgetting, deletion, tombstones, and crypto-shredding

Amos should distinguish:

```text
demotion:
  less active, still retained

archival:
  retained cold, rarely retrieved

forgetting:
  removed from active memory and normal retrieval

deletion:
  payload or record removed under policy

crypto-shredding:
  encrypted data rendered unrecoverable by deleting keys

tombstoning:
  minimal marker retained to prevent re-creation or preserve audit
```

If a user asks Amos to forget a memory, Amos may need to remove the atom, remove derived indexes, and add a tombstone so the same memory is not re-inferred from old evidence.

Deletion workflow:

```text
1. Accept DeletionRequest through the strong write path.
2. Authorize requester against owner scope, retention policy, and legal hold.
3. Classify target records: atoms, edges, evidence, rendered caches, embeddings, telemetry, snapshots, backups.
4. Apply the strongest allowed action: suppress, archive, tombstone, physical delete, or crypto-shred.
5. Append deletion/tombstone events.
6. Reproject canonical graph and purge derived indexes/caches.
7. Record residual-retention explanation if any audit or legal metadata remains.
```

Deletion request shape:

```text
DeletionRequest
  request_id
  requester_identity
  target_refs
  scope
  requested_action: forget | delete | crypto_shred
  reason_code
  include_evidence
  include_derived_artifacts
  prevent_recreation
  requested_at
```

Tombstone shape:

```text
MemoryTombstone
  tombstone_id
  target_ref
  target_type
  scope
  deletion_event_id
  recreation_policy: forbid | require_review | allow_if_new_evidence
  retained_reason_code
  redacted_hash
  expires_at
```

Backups and cold archives must be covered by a deletion policy:

```text
hot stores:
  delete or crypto-shred immediately after successful projection

derived indexes and packet caches:
  purge before acknowledging strong deletion when feasible; otherwise mark inaccessible and complete purge asynchronously with audit

cold archives:
  delete, crypto-shred, or mark for expiry according to retention policy

snapshots:
  rebuild without deleted payloads or store encrypted payloads so key deletion makes them unrecoverable

backups:
  either support targeted deletion/crypto-shredding or document maximum residual retention window
```

Evidence cannot be used to regenerate a deleted atom when `prevent_recreation` is true. If compliance policy prevents physical removal of evidence, Amos must suppress ordinary retrieval and keep only the minimum audit metadata needed to explain residual retention.

### 26.10 Scalability principle

```text
The long-term growth rate of Amos should be proportional to durable useful memory,
not total observed activity.
```

---

## 27. Capacity governance

Amos must be instantiated with explicit capacity budgets and must operate within them without burdening connected agent services. Capacity management is handled inside Amos by the Capacity Governor and Memory Steward.

Agents using Amos should not need to know:

```text
how full the journal is
whether evidence was compacted
which atoms were demoted
whether indexes were rebuilt
whether more capacity has been requested
which storage action Amos will take next
```

Agents should keep using the same memory contract:

```text
capture_event(...)
retrieve_memory_packet(...)
propose_memory(...)
record_outcome(...)
```

Agents do need to know when a returned packet is degraded. Capacity pressure is therefore reported as packet metadata, not as a burden to choose deletion, compaction, or storage expansion actions.

### 27.1 Capacity Governor

```text
Capacity Governor
  owns budgets
  monitors usage
  enforces watermarks
  triggers compaction
  requests expansion
  prioritizes memory retention
  shields agents from capacity concerns
```

Relationship to the Memory Steward:

```text
Capacity Governor:
  storage/resource budgets, pressure response, growth control

Memory Steward:
  memory quality, cleanup, promotion, demotion, repair
```

The Capacity Governor detects pressure. The Memory Steward determines what can be compacted, demoted, merged, archived, or deleted safely.

### 27.2 Capacity contract

An Amos instance should start with a configured capacity contract.

```text
AmosCapacityContract
  initial_capacity
  growth_budget
  hard_capacity_limit
  requested_expansion_limit
  scope_budgets
  watermarks
  retention_policy
  pressure_policy
  admin_notification_policy
```

Each budget entry should specify:

```text
CapacityBudget
  scope
  tier
  soft_limit_bytes
  hard_limit_bytes
  max_growth_bytes_per_day
  max_objects
  minimum_retention_window
  pressure_mode_overrides
  enforcement_action
```

Recommended enforcement actions:

```text
warn
raise_thresholds
compact
archive
freeze_low_value_promotion
reject_noncritical_candidates
drop_disposable_cache
request_expansion
```

Capacity should be tracked separately by storage tier:

```text
hot_graph
hot_journal
evidence_archive
derived_indexes
packet_cache
maintenance_queue
```

Budgets should also be scoped:

```text
global instance
tenant
workspace
project
user
agent
session
memory category
storage tier
```

A noisy agent, project, or telemetry source should not consume the entire Amos instance.

### 27.3 Watermarks and pressure modes

Amos should act before crisis.

```text
Green:
  normal operation; tier usage < 70% and projected exhaustion >= 30 days

Yellow:
  light pressure; any tier usage >= 70% or projected exhaustion < 30 days;
  increase compaction, prune caches, aggregate telemetry

Orange:
  serious pressure; any tier usage >= 85% or projected exhaustion < 14 days;
  generate admin capacity extension request, compact older segments,
  archive stale evidence, raise promotion thresholds

Red:
  critical pressure; any tier usage >= 95% or projected exhaustion < 3 days;
  aggressive compaction, freeze low-value promotions, demote stale memory,
  delete expired evidence, prune disposable derived indexes

Black:
  emergency preservation mode; any tier usage >= 98%, hard limit would be exceeded,
  or writes are failing; protect core canonical state and critical memory,
  suspend non-critical indexing, drop disposable caches, capture minimal correctness records
```

Defaults are per storage tier and may be overridden by deployment policy. Predictive thresholds use recent growth rate, minimum observed retention windows, and configured business-hours notification windows.

Memory quality should degrade gracefully:

```text
storage pressure → controlled compaction → reduced recall depth → preserved core memory
```

Packet degradation contract:

```text
Green:
  full configured retrieval depth

Yellow:
  same canonical recall, reduced evidence detail or cache retention allowed

Orange:
  reduced recall depth allowed; packet must include pressure_degraded omission reasons

Red:
  only high-confidence/high-utility memory and critical conflicts guaranteed

Black:
  only policy-critical, active commitment, and correctness-critical memory guaranteed
```

not:

```text
storage pressure → random deletion → incoherent agent behavior
```

### 27.4 Capacity extension requests

At orange or predictive-risk thresholds, Amos should generate an admin-facing request.

```text
CapacityExtensionRequest
  amos_instance_id
  scope
  current_usage
  projected_exhaustion
  requested_capacity
  reason
  pressure_level
  actions_already_taken
  consequences_if_denied
  recommended_deadline
```

Capacity extension is optional. Amos must continue operating if expansion is delayed or denied.

### 27.5 Survival under denied or delayed expansion

If no new capacity arrives, Amos should preserve memory in priority order:

```text
1. Safety and policy constraints
2. User-approved durable preferences
3. Active commitments and goals
4. Active project memory
5. Reinforced beliefs and procedures
6. Evidence supporting active high-value atoms
7. Recent episode memory
8. Weak associations
9. Retrieval telemetry
10. Generated summaries and packet caches
11. Expired raw evidence
12. Rejected candidates and low-utility episodes
```

Disposable artifacts are sacrificed first. Durable, active, or policy-critical memory is protected longest.

### 27.6 Capacity-aware admission and promotion

Agents may continue submitting events, but Amos decides how much is retained.

Normal operation:

```text
event → evidence → candidates → atoms → indexes
```

Pressure operation:

```text
event → compact evidence pointer → selective candidate extraction
```

Critical operation:

```text
event → minimal journal record → short-TTL evidence → only high-salience candidates
```

Emergency operation:

```text
event → minimal correctness record
only policy-critical and correctness-critical memory committed
non-critical candidates dropped or session-scoped
```

Promotion thresholds should rise under pressure.

```text
Green:
  promote useful candidates normally

Yellow:
  require clearer utility or novelty

Orange:
  promote only high-confidence/high-utility candidates

Red:
  promote only critical memories: explicit corrections, active commitments,
  durable preferences, major project decisions, reusable procedures, safety/policy constraints

Black:
  commit only correctness-critical and policy-critical memory
```

### 27.7 Retention classes

Every memory object should have a retention class.

```text
RetentionClass
  ephemeral
  session
  recent
  project
  durable
  compliance
  user_pinned
```

Examples:

```text
ephemeral:
  packet caches, temporary summaries, low-value telemetry

session:
  current task state, temporary observations

recent:
  recent episodes, debug traces, unconfirmed candidates

project:
  active project decisions, design history

durable:
  explicit preferences, stable procedures, reinforced beliefs

compliance:
  audit records required by policy

user_pinned:
  memories explicitly marked as important
```

### 27.8 Agent shielding

Capacity concerns are Amos responsibilities, not agent responsibilities.

```text
Agents:
  express task context, events, outcomes, candidate memories

Amos:
  decides retention, compaction, promotion, demotion, deletion, indexing

Admins:
  configure budgets and approve capacity expansion

Users:
  may correct, pin, delete, or constrain memory
```

Agents should not decide which memory to delete, which journal segment to compact, whether to request storage, or how to prune edges.

### 27.9 Admin observability

Admins need reports and dashboards.

```text
CapacityHealthReport
  current usage by tier
  current usage by scope
  growth rate
  projected exhaustion
  pressure mode
  compaction actions taken
  deletion/archival actions proposed
  expansion requests
  memory quality impact estimate
```

Useful metrics:

```text
hot_graph_usage
hot_journal_usage
cold_evidence_usage
derived_index_usage
cache_usage
growth_rate_by_scope
days_until_next_watermark
compaction_savings
demotion_savings
expired_evidence_bytes
weak_edge_count
low_utility_atom_count
promotion_rate
discarded_candidate_rate
```

### 27.10 Capacity principle

```text
Amos must preserve useful long-term memory within explicit capacity budgets,
request more capacity when justified, and continue operating gracefully when
expansion is delayed or unavailable.
```

---

## 28. Non-LLM semantic maintenance

Amos should not require an LLM for core maintenance. Core maintenance must be policy-driven, deterministic where possible, auditable, and capable of operating under resource pressure.

LLMs and other generative models may be optional escalation paths, but routine maintenance should prefer a fast, bounded, non-generative semantic layer.

### 28.1 Semantic Maintenance Processor

```text
Semantic Maintenance Processor
  = non-generative semantic layer used by Amos for maintenance decisions
```

The Semantic Maintenance Processor, or SMP, should use specialized bounded processors rather than open-ended generation.

```text
┌─────────────────────────────────────────────────────────────┐
│ Semantic Maintenance Processor                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Rule and policy engine                                   │
│ 2. Schema/shape validator                                   │
│ 3. Embedding encoder                                        │
│ 4. Approximate nearest-neighbor index                       │
│ 5. Graph activation engine                                  │
│ 6. Lightweight classifiers                                  │
│ 7. Clustering/deduplication engine                          │
│ 8. Contradiction and scope engine                           │
│ 9. Utility/salience scorer                                  │
│ 10. Pattern miner / procedure candidate miner               │
└─────────────────────────────────────────────────────────────┘
```

### 28.2 SMP responsibilities

```text
encode text spans or atom content
compare semantic similarity
cluster related memories
classify memory type
classify health status
detect likely redundancy
detect likely contradiction
assign scope candidates
score salience and utility
rank promotion candidates
rank demotion candidates
support spreading activation
detect anomalous or confounding memories
validate structured memory against schemas
```

The SMP outputs scores, labels, clusters, candidate links, reason codes, and candidate actions. It does not rewrite canonical memory as free-form prose.

SMP outputs are advisory unless a policy rule explicitly allows autonomous execution.

```text
Autonomous low-risk actions:
  update utility counters
  add weak candidate link
  mark proposed atom as underspecified
  prune disposable packet cache

Autonomous medium-risk actions only with high confidence and reversible journal entry:
  archive low-utility proposed atom
  mark atom stale
  propose duplicate merge without deleting originals

Review-required actions:
  alter user preference
  promote procedure to active
  mark active belief contradicted
  merge active atoms destructively
  delete or tombstone any canonical record
  change access policy or retention class
```

Reason-code families:

```text
shape_invalid
scope_too_broad
scope_too_narrow
near_duplicate
contradiction_candidate
stale_by_age
stale_by_external_change
low_retrieval_utility
confounding_after_correction
privacy_risk
capacity_pressure
policy_required
```

### 28.3 Shape validation

Amos can use shape-style constraints to detect incoherent memory without an LLM.

Examples:

```text
MemoryAtom must have evidence_refs.
MemoryAtom must have confidence, scope, lifecycle_state, health_status, and retention_class.
PreferenceAtom payload must have holder, polarity, target, applicability_scope, strength.
ProcedureAtom payload must have trigger_context and steps.
AssociationEdge must have edge_id, source_ref, target_ref, relation, confidence, scope, lifecycle_state, and health_status.
High-risk mutations must pass authorization and review gates before commit.
```

This catches malformed, underspecified, orphaned, or overgeneralized memory early.

### 28.4 Embeddings and classifiers as derived processors

Embeddings and classifiers are useful but derived.

```text
canonical:
  atoms, evidence, edges, health state

derived:
  embedding vectors, nearest-neighbor candidates, cluster IDs, classifier scores
```

Embeddings can propose near-duplicates, semantic clusters, related episodes, and candidate association edges. Amos still validates through structure, scope, evidence, and policy.

V1-local uses a dependency-free deterministic encoder rather than an external
embedding service. The encoder combines:

```text
word hashes:
  token term frequency multiplied by graph-version document-frequency IDF

character hashes:
  token character trigrams and four-grams for morphology and typo tolerance

latent token vectors:
  optional maintenance-built LSA projection from the local token-atom matrix
```

The IDF map is derived from `amos_atom_text_index(token, atom_id)`. The LSA
projection is stored in `amos_token_latent_vectors` and refreshed by derived
index maintenance, not during request handling. Stored atom search vectors carry
graph-version vector-model metadata; stale vectors are ignored and recomputed
from current derived statistics. These vectors are ranking aids and maintenance
signals only. They are not canonical memory and are rebuildable from atom
payload search text plus derived token rows.

### 28.5 Graph activation without generation

Associative retrieval can run without an LLM.

```text
cue atoms
  → activate directly matching nodes
  → spread across typed edges
  → apply decay
  → apply inhibition
  → apply utility/confidence/scope weights
  → produce memory packet candidates
```

### 28.6 Pattern mining for procedure candidates

Procedure induction does not have to be generative. A pattern miner can detect repeated successful action sequences.

```text
task_type = architecture_design
successful episodes share:
  define problem
  establish axes
  identify tradeoffs
  propose roadmap
  avoid implementation code until requested
```

This can produce a structured `ProcedureCandidate` with observed steps, success count, failure count, confidence, and evidence episode IDs. Naming and prose rendering can be deferred or handled by a human/admin if needed.

### 28.7 Optional LLM reviewer

An LLM may help with ambiguous semantic work:

```text
ambiguous atomization
scope refinement
incoherent memory repair
semantic deduplication
contradiction analysis
procedure naming
episode distillation
natural-language explanations
```

But LLM outputs should always be proposals, not authoritative mutations.

The reviewer is a stateless, replaceable processor with respect to durable
agent identity. Its model identity, provider persona, and generated
self-description must not become the subject identity or evidence for a
self-model claim.

```text
LLM proposes
Amos validates
policy gates
journal records
canonical memory updates
indexes refresh
```

Amos must survive when no LLM is available.

### 28.8 Maintenance decision ladder

```text
1. Deterministic rule can decide?
   Execute.

2. Structural graph pattern can decide?
   Execute or propose.

3. Embedding + classifier agree with high confidence?
   Execute low-risk action or propose medium-risk action.

4. Ambiguous but low value?
   Archive, demote, or keep as proposed.

5. Ambiguous and high value?
   Request admin/user review.

6. Optional deployment has LLM enabled?
   Send to LLM as non-authoritative reviewer.
```

### 28.9 SMP interface

```text
SemanticMaintenanceProcessor
  encode(atom_or_text_span) -> vector
  classify(memory_candidate) -> labels + scores
  compare(atom_a, atom_b) -> similarity + relation_guess
  cluster(atom_set) -> clusters
  validate_shape(atom_or_edge) -> validation_report
  detect_conflicts(atom_set) -> conflict_candidates
  score_utility(atom, telemetry, scope) -> utility_score
  propose_links(atom, candidates) -> edge_candidates
  propose_health(atom, telemetry) -> health_status_candidates
```

Every SMP output should include:

```text
processor_id
processor_version
input_refs
output_type
confidence
reason_code
evidence_refs
recommended_action
risk_level
```

Reason codes can replace prose explanations for speed and auditability.

### 28.10 Non-LLM maintenance principle

```text
Amos should prefer non-generative semantic maintenance over LLM-based maintenance.
Routine maintenance should be performed by deterministic policy, graph algorithms,
schema validation, embeddings, classifiers, clustering, spreading activation, and
utility scoring. Generative LLM calls are optional escalation paths only.
```

---

## 29. Verified v1-local contract and roadmap

The implementation-specific defaults, repository artifacts, acceptance status,
and known partial gates are maintained in
[`v1-local-contract.md`](v1-local-contract.md). Future implementation work is
maintained separately in [`roadmap.md`](roadmap.md).

This separation is intentional: sections 1-28 define the longer-term AMOS
architecture. They are not claims that every distributed, archival, snapshot,
capacity-tier, or deletion-policy feature exists in the v1-local SQLite
profile. The checked-in profile is one HTTP service process with one
service-owned SQLite store and explicitly identified partial gates.

---

## 30. Current design principle

Amos should provide a shared, layered, associative, self-maintaining memory operating plane for agentic AI systems.

It should support:

```text
multiple agent services connected to one authoritative Amos instance
shared memory view with individual agent self-models
evidence preservation and compact canonical memory
controlled consolidation and compaction
cross-category linking and cross-tier promotion/demotion
trace-backed agentic recall and generated self-narratives
auditable maintenance and repair
capacity governance under explicit budgets
admin-facing capacity extension requests
agent shielding from storage pressure
non-generative semantic maintenance by default
optional LLM review only as a non-authoritative escalation path
```

The memory system should continuously answer:

```text
What should remain active?
What should be linked?
What should be promoted?
What should be demoted?
What is stale?
What is redundant?
What is incoherent?
What is orphaned?
What is confounding?
What should be preserved only as evidence?
What should be compacted?
What should be archived?
What should be deleted or tombstoned?
When should more capacity be requested?
How can service-facing memory quality be preserved under pressure?
```

The end goal is Amos: an Agent Memory Operating System that lets agentic systems share durable memory without forcing every agent process to manage storage, compaction, cleanup, retrieval semantics, or capacity pressure.
