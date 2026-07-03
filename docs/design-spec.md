# Amos Design Spec

**Project name:** Amos  
**Expansion:** Agent Memory Operating System  
**Status:** Planning / design-spec phase  
**Implementation status:** No code yet

---

## 1. Purpose

This design spec captures the current plan for **Amos**, an Agent Memory Operating System for agentic AI systems that must operate at long time horizons and large scale without relying on textual English summaries as the canonical long-term memory format.

Amos is intended to be a model-neutral, layered, associative, self-maintaining belief and memory substrate that can serve LLM reasoners, planners, executors, critics, symbolic systems, retrieval systems, and future processors through generated views.

The design is intentionally implementation-free at this stage.

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
│ atoms, edges, evidence links, maturity states              │
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
  layer, maturity, health, retention, version, timestamps

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
  maturity
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
  "schema_version": "amos.atom.preference.v1",
  "payload": {
    "holder": "ent:user:primary",
    "polarity": "prefers",
    "target": "ent:interaction_style:iterative_conceptual_design",
    "scope": {
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
  "layer": "consolidated_long_term",
  "maturity": "active",
  "health_status": "active",
  "retention_class": "project",
  "access_policy": {
    "tenant": "default",
    "workspace": "amos",
    "visibility": ["reasoner", "planner", "steward"]
  },
  "created_at": "2026-07-02T00:00:00Z",
  "observed_at": "2026-07-02T00:00:00Z",
  "updated_at": "2026-07-02T00:00:00Z",
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
  "schema_version": "amos.atom.belief.v1",
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
  "maturity": "active",
  "health_status": "active",
  "retention_class": "project",
  "version": 1,
  "created_at": "2026-07-02T00:00:00Z"
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
    "status": "active"
  },
  "validity_interval": {
    "from": "2026-07-02T00:00:00Z",
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
  "scope": {
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
  "status": "active",
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
    "session": "ses_01JZAMOS"
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
    "from": "evt_100",
    "to": "evt_147"
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
  "id": "evd_01JZB0XG4X7Y3VHV93R5Y3NMD1",
  "source_type": "conversation_turn",
  "source_uri_or_pointer": "archive://tenant/default/session/amos-2026-07-02/turn/42",
  "span": {
    "start": 0,
    "end": 187
  },
  "checksum": "sha256:...",
  "timestamp": "2026-07-02T00:00:00Z",
  "retention_policy": "project"
}
```

A typed atom is also not the association edge. Associations are separate records so they can be pruned, reinforced, inhibited, or reweighted independently:

```json
{
  "id": "edge_01JZB22MEQ2T3P8R6N4B8Z7WEA",
  "source_id": "atm_01JZB0Z8WX4R7AN7M7P5XDK9Q4",
  "target_id": "atm_01JZB1H4M92HMSAPV2Y6RE3XKC",
  "edge_type": "supports",
  "weight": 0.87,
  "evidence_refs": [
    "evd_01JZB0XG4X7Y3VHV93R5Y3NMD1"
  ],
  "direction": "directed",
  "created_at": "2026-07-02T00:00:00Z",
  "last_reinforced": "2026-07-02T00:00:00Z",
  "decay_policy": "slow",
  "inhibition_policy": null
}
```

Every committed atom should be introduced through the Amos Event Journal. The journal records the mutation; the Canonical Memory Graph stores the current projected state.

```json
{
  "event_id": "mev_01JZB1A4DM6R7Y6CAQ9VY0H2XF",
  "event_type": "atom.committed",
  "actor": "svc:memory_steward",
  "idempotency_key": "idem_abc123",
  "timestamp": "2026-07-02T00:00:00Z",
  "payload": {
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

A practical first implementation can use:

```text
Event Journal
  JSON Lines or Protobuf events

Canonical Memory Graph
  Postgres tables with JSONB payloads plus normalized indexes

Evidence Archive
  object storage or filesystem storage with checksums and retention metadata

Derived Indexes
  vector index, keyword index, graph adjacency tables, packet cache
```

A relational MVP could map the canonical graph into these tables:

```text
atoms
  id
  type
  schema_version
  payload JSONB
  scope JSONB
  confidence JSONB
  health_status
  maturity
  retention_class
  created_at
  updated_at
  version

edges
  id
  source_id
  target_id
  edge_type
  weight
  evidence_refs JSONB
  created_at
  updated_at

evidence
  id
  source_type
  pointer
  checksum
  retention_policy

event_journal
  event_id
  event_type
  actor
  timestamp
  payload JSONB
  idempotency_key
  causal_parent_ids JSONB
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

### 7.2 Entity

Stable references for people, projects, tools, repositories, files, organizations, concepts, agents, and environments.

```text
Entity
  id
  type
  labels
  aliases
  canonical_name
  external_refs
  created_at
```

Repeated strings should be dictionary-encoded by entity IDs.

### 7.3 Event

Immutable record of what happened.

```text
Event
  id
  actor
  action
  object
  timestamp
  input_digest
  output_digest
  outcome
  linked_entities
  raw_pointer
```

Events preserve evidence and should not be casually rewritten.

### 7.4 BeliefAtom

Represents an active or candidate claim.

```text
BeliefAtom
  subject
  relation
  object
  modality: observed | inferred | user_stated | system_derived | predicted
  confidence
  evidence_ids
  contradiction_links
  status: candidate | active | stale | rejected | uncertain
  scope
  validity_interval
```

Memory is what was observed. Belief is what the system currently accepts as useful or likely true.

### 7.5 PreferenceAtom

Preferences are scoped and contextual.

```text
PreferenceAtom
  holder
  prefers_or_avoids
  target
  scope
  strength
  source
  exceptions
  confidence
```

Example:

```text
holder: user
prefers: iterative conceptual design before implementation
scope: architecture discussions
strength: high
```

This prevents overgeneralized memories such as `user never wants code`.

### 7.6 Goal

```text
Goal
  owner
  desired_state
  status
  priority
  deadline
  dependencies
  blockers
  evidence
```

### 7.7 Commitment

```text
Commitment
  agent
  promised_action
  recipient
  status
  due_condition
  source_event
```

This helps the planner avoid losing open loops.

### 7.8 ProcedureAtom

Procedural memory should be structured, versioned, and auditable.

```text
ProcedureAtom
  id
  name
  trigger_context
  preconditions
  steps
  tool_affordances
  expected_outputs
  known_failure_modes
  recovery_strategies
  evidence_episode_ids
  confidence
  owner
  version
  supersedes
  rollback_pointer
```

A procedure may render into English for an LLM or into a structured action schema for a planner/executor.

### 7.9 Episode

A compressed trace of meaningful activity.

```text
Episode
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
  id
  source_type
  source_uri_or_pointer
  span
  checksum
  timestamp
  access_policy
  retention_policy
```

Evidence supports auditability and reconstruction.

### 7.11 AssociationEdge

```text
AssociationEdge
  source_id
  target_id
  edge_type
  weight
  evidence_refs
  direction
  created_at
  last_reinforced
  decay_policy
  inhibition_policy
```

Edges are first-class. They are not incidental metadata.

### 7.12 MemoryPacket

The retrieval output consumed by processors.

```text
MemoryPacket
  task_relevance
  active_beliefs
  active_preferences
  relevant_goals
  useful_episodes
  applicable_procedures
  semantic_claims
  conflicts_or_uncertainties
  provenance
  rendering_target
```

The packet can be rendered as English, JSON, graph neighborhoods, planner state, or executor context.

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
active → stale
active → candidate
active → contradicted
active → archived
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
- contradiction penalty
- staleness penalty
- privacy/access penalty
- redundancy penalty
```

### 11.3 Diversity requirement

Retrieval should produce a bundle, not just top-k text chunks:

```text
best active beliefs
best active preferences
best procedure
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

## 16. Memory health statuses

Each atom should have a health state.

```text
active
candidate
reinforced
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
archived
deleted
```

Health state affects retrieval eligibility.

Examples:

```text
stale:
  retrievable only if timeline/history matters

contradicted:
  retrieved with warning or conflict context

orphaned:
  excluded from normal retrieval until relinked or archived

confounding:
  actively suppressed unless specifically requested

candidate:
  not yet long-term truth; may become active after reinforcement
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
  mark candidate

Medium-risk:
  merge duplicates
  promote candidate to active scoped belief
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
```

---


## 25. Distributed Amos instance

Amos is intended to operate as a shared memory plane for multi-process agentic systems. A realistic agentic runtime may have separate reasoner, planner, executor, critic, tool-worker, user-interface, evaluator, and steward processes. All of those processes should connect to the same logical Amos instance for memory access.

Amos should therefore be designed as a networked memory service, not only as an in-process library.

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

Conceptual API surface:

```text
capture_event(...)
propose_memory_atoms(...)
commit_memory_atoms(...)
retrieve_memory_packet(...)
record_retrieval_outcome(...)
link_memory(...)
promote_memory(...)
demote_memory(...)
request_maintenance(...)
get_memory_health(...)
```

The boundary is:

```text
Agent services request memory operations.
Amos owns memory validity, structure, lifecycle, maintenance, and audit.
```

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

### 25.3 Shared write path through MemoryEvents

All memory mutations should flow through a structured append-only event stream.

```text
MemoryEvent
  id
  actor_service
  actor_agent
  operation_type
  payload
  idempotency_key
  causal_parent_ids
  timestamp
  authorization_context
  result_status
```

The canonical memory graph is a validated projection of accepted memory events.

```text
MemoryEvents
  → validation / authorization / reconciliation
  → Canonical Memory Graph update
  → derived index update
  → packet-cache invalidation
  → publication of change events
```

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

### 25.5 Consistency levels

Not all memory operations need the same consistency model.

Stronger consistency is required for:

```text
user corrections
active preferences
policy constraints
goal status
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

### 25.6 Pub/sub and cache invalidation

Connected services may maintain local caches, but Amos remains authoritative. Amos should publish change events such as:

```text
memory.atom.created
memory.atom.updated
memory.atom.promoted
memory.atom.demoted
memory.atom.marked_stale
memory.preference.corrected
memory.procedure.promoted
memory.goal.status_changed
memory.index.rebuilt
memory.health.alert
```

Clients use these events to invalidate local memory packets, active procedure snippets, or retrieved context caches.

### 25.7 Scope isolation

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
event.captured
atom.proposed
atom.committed
atom.merged
edge.created
preference.corrected
retrieval.performed
retrieval.outcome_recorded
maintenance.action_proposed
maintenance.action_committed
index.rebuilt
```

### 26.2 Raw payloads are not stored inline forever

Journal entries should store compact structured events, payload digests, evidence pointers, checksums, and causal links. Large payloads should live in the Evidence Archive.

```text
event.captured
  event_id: evt_123
  source_type: user_message
  evidence_pointer: evidence://conversation/session_44/turn_42
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

Committed memory is not permanently hot. It should continue moving through lifecycle states:

```text
candidate
active
reinforced
stale
redundant
contradicted
confounding
archived
deleted
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
whether Amos is operating under pressure
```

Agents should keep using the same memory contract:

```text
capture_event(...)
retrieve_memory_packet(...)
propose_memory(...)
record_outcome(...)
```

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
  normal operation

Yellow:
  light pressure; increase compaction, prune caches, aggregate telemetry

Orange:
  serious pressure; generate admin capacity extension request, compact older segments,
  archive stale evidence, raise promotion thresholds

Red:
  critical pressure; aggressive compaction, freeze low-value promotions, demote stale
  memory, delete expired evidence, prune disposable derived indexes

Black:
  emergency preservation mode; protect core canonical state and critical memory,
  suspend non-critical indexing, drop disposable caches, capture minimal correctness records
```

Memory quality should degrade gracefully:

```text
storage pressure → controlled compaction → reduced recall depth → preserved core memory
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

### 28.3 Shape validation

Amos can use shape-style constraints to detect incoherent memory without an LLM.

Examples:

```text
MemoryAtom must have evidence_refs.
PreferenceAtom must have holder, target, scope, confidence.
ProcedureAtom must have trigger_context and version.
AssociationEdge must have source_id, target_id, type, weight.
High-risk memories must have review_status.
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
   Archive, demote, or keep candidate.

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

## 29. Open design questions

1. What is the minimum viable atom schema that avoids overengineering?
2. What exact JSON Schema, Protobuf, or equivalent schema definition should represent each atom payload type?
3. Which fields belong in the common atom envelope versus type-specific payloads?
4. How should normalized entity, relation, enum, and concept identifiers be minted, versioned, and compacted?
5. What is the first physical storage mapping: JSONB tables, graph database, RDF/quads, Protobuf event stream, or a hybrid?
6. How should confidence be represented: scalar, categorical, or evidence-based vector?
7. How should retrieval diversity be enforced across memory types?
8. What maintenance actions require user confirmation?
9. How should privacy and retention policies interact with evidence preservation?
10. How should contradictory memories be rendered to the reasoner?
11. How much procedural memory should be executable versus advisory?
12. What is the right promotion threshold for preferences inferred from behavior rather than explicit instruction?
13. How should memories be shared across multiple agents without contaminating individual agent context?
14. How should generated English summaries be tested for drift against canonical atoms?
15. What is the minimum viable service API for a distributed Amos instance?
16. Which memory operations require strong consistency versus eventual consistency?
17. How should scope budgets be allocated across tenants, workspaces, projects, agents, and storage tiers?
18. What should the default watermark thresholds be for normal, pressure, critical, and preservation modes?
19. Which retention classes should be user-configurable, admin-configurable, or system-defined?
20. What compact journal rollup format preserves enough auditability without retaining all raw events hot?
21. What subset of semantic maintenance can be handled by non-generative processors in the first implementation?
22. When should Amos escalate from the Semantic Maintenance Processor to admin/user review or an optional LLM reviewer?

---

## 30. Iterative roadmap

### Iteration 1: Design foundation

- Define the Amos concept and its memory-kernel responsibilities.
- Separate evidence, canonical memory, indexes, and rendered views.
- Identify memory types, timescales, and maintenance needs.

### Iteration 2: Canonical schema and atom storage format

Design the minimal logical schema, canonical interchange encoding, and first physical storage mapping for:

```text
MemoryAtom envelope
typed atom payloads
Entity
Event
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

### Iteration 3: Lifecycle and maintenance model

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

### Iteration 4: Retrieval and rendering contracts

Define the MemoryPacket interface for:

```text
reasoner
planner
executor
critic
steward
```

### Iteration 5: Quality and health metrics

Define metrics, maintenance thresholds, and audit requirements.

### Iteration 6: Scenario pressure tests

Pressure-test against:

```text
personal assistant over five years
enterprise agent over millions of tasks
coding agent learning repository conventions
research agent maintaining hypotheses
multi-agent operations team
```

### Iteration 7: Distributed service model

Specify:

```text
shared Amos instance contract
client identities and capabilities
MemoryEvent write path
concurrency and consistency model
pub/sub and cache invalidation
scope isolation
conflict resolution
```

### Iteration 8: Journal, compaction, and retention model

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

### Iteration 9: Capacity governance

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

### Iteration 10: Non-LLM semantic maintenance

Specify:

```text
Semantic Maintenance Processor interface
shape validation
embedding/classifier/graph processor roles
reason codes
maintenance decision ladder
optional LLM escalation policy
```

### Iteration 11: Implementation planning

Only after the design is stable, begin implementation planning:

```text
storage backend candidates
schema format
indexing strategy
service API surface
maintenance scheduler
capacity governor
semantic maintenance processor
retrieval/ranking experiments
```

---

## 31. Current design principle

Amos should provide a shared, layered, associative, self-maintaining memory operating plane for agentic AI systems.

It should support:

```text
multiple agent services connected to one authoritative Amos instance
evidence preservation and compact canonical memory
controlled consolidation and compaction
cross-category linking and cross-tier promotion/demotion
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
