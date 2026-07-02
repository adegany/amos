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

A generic canonical memory unit.

```text
MemoryAtom
  id
  type
  layer
  maturity
  content
  evidence_refs
  confidence
  salience
  utility
  scope
  created_at
  observed_at
  last_accessed
  decay_policy
  health_status
  revision_history
```

Specialized atom types inherit this shape.

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

## 25. Open design questions

1. What is the minimum viable atom schema that avoids overengineering?
2. How should confidence be represented: scalar, categorical, or evidence-based vector?
3. How should retrieval diversity be enforced across memory types?
4. What maintenance actions require user confirmation?
5. How should privacy and retention policies interact with evidence preservation?
6. How should contradictory memories be rendered to the reasoner?
7. How much procedural memory should be executable versus advisory?
8. What is the right promotion threshold for preferences inferred from behavior rather than explicit instruction?
9. How should memories be shared across multiple agents without contaminating individual agent context?
10. How should generated English summaries be tested for drift against canonical atoms?

---

## 26. Iterative roadmap

### Iteration 1: Design foundation

- Define the Amos concept and its memory-kernel responsibilities.
- Separate evidence, canonical memory, indexes, and rendered views.
- Identify memory types, timescales, and maintenance needs.

### Iteration 2: Canonical schema

Design the minimal schema for:

```text
MemoryAtom
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

### Iteration 7: Implementation planning

Only after the design is stable, begin implementation planning:

```text
storage backend candidates
schema format
indexing strategy
API surface
maintenance scheduler
retrieval/ranking experiments
```

---

## 27. Current design principle

Amos should provide a layered, associative, self-maintaining belief system with evidence preservation, controlled consolidation, cross-category linking, cross-tier promotion, demotion, and auditable repair.

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
What should be forgotten?
```

The end goal is Amos: an Agent Memory Operating System for agentic AI.
