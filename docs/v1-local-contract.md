# AMOS V1-Local Contract

**Status:** Verified implementation contract for the checked-in single-process
SQLite service profile.

This document contains implementation-specific defaults, artifacts, and
acceptance evidence. The longer-term architecture remains in
[`design-spec.md`](design-spec.md); planned work remains in
[`roadmap.md`](roadmap.md). Statements here use these status terms:

- **Implemented:** present in code and covered by automated tests or a checked
  repository artifact.
- **Partial:** the checked-in implementation covers only the explicitly named
  subset.
- **Planned:** architecture or deployment work not provided by v1-local.

## 1. V1 implementation baseline

The checked-in v1-local implementation uses the following concrete defaults.
These choices are intentionally conservative: they favor auditability, replay,
schema validation, and simple operations over maximum scale.

### 1.1 Authoritative schema format

V1 uses JSON Schema 2020-12 as the authoritative schema format.

```text
authoritative schemas:
  JSON Schema 2020-12

canonical interchange:
  JSON-compatible records

generated code:
  allowed for application models, validators, and API clients

schema compatibility:
  additive optional fields are minor-compatible
  required-field changes require a new schema_version
  semantic changes require a new schema_version
```

Generated Protobuf, Avro, or typed language models may be produced from the JSON Schemas later, but they are derived artifacts in v1.

The runtime remains dependency-free, so `src/amos/schemas.py` mirrors the
applicable envelope and per-type payload constraints in Python rather than
loading a third-party JSON Schema evaluator. Contract tests cover required
fields, property types, enum constraints, envelope separation, and canonical
score bounds. Changes to either representation must update those parity tests.

### 1.2 Storage backend

V1 storage target:

```text
service boundary:
  HTTP API server owns one in-process SQLite database and is the shared Amos instance

primary database:
  SQLite

canonical graph:
  normalized relational columns + JSON payloads

event journal:
  SQLite append-only event_journal table

evidence archive:
  filesystem or object storage referenced by EvidenceRecord.source_ref

derived vector index:
  replaceable derived lexical and LSA indexes; v1 does not require a bundled
  vector database

packet cache:
  SQLite table keyed by graph_version and request signature
```

The first implementation must keep vector indexes, packet caches, generated summaries, and telemetry rollups rebuildable from canonical records and retained evidence.

V1-local SQLite service profile:

```text
connection profile:
  foreign key enforcement enabled
  busy timeout configured for foreground/background contention
  WAL journal mode for file-backed databases
  synchronous=NORMAL for service-owned local durability/performance balance

canonical write profile:
  strong mutations use explicit transactions
  batch atom commits preflight duplicate ids before writing
  atom rows, edge rows, and journal entries commit atomically
  packet cache invalidation is performed once per successful mutation batch

read/query profile:
  filtered atom queries support type, lifecycle, health, deletion, limit, and
  ordering constraints
  SQL count helpers back health and derived-index status without loading the
  full graph
  graph-edge degree maps and ref-scoped edge reads are preferred over repeated
  full-edge scans
```

Derived search metadata is stored in each atom's `index_refs` under a processor
specific key such as `amos.v1.search`. This metadata may include normalized
search text, token lists, deterministic TF-IDF/character-hash vectors, and
graph-version vector-model metadata. The disposable `amos_atom_text_index` table
provides token document-frequency counts and candidate prefiltering. The
disposable `amos_token_latent_vectors` table stores the latest maintenance-built
token -> latent vector projection. All of this is rebuildable from canonical
atom payloads and must not become the canonical source of memory truth.

Postgres is a future TODO for scale-out deployments that need multiple API
instances, independent writer processes, database-managed role separation,
replication, point-in-time recovery, JSONB/GIN indexing, or stronger operational
tooling. Postgres migration artifacts may be kept as a target contract, but
they are not required to prove v1 correctness.

### 1.3 Deployment topology

V1 strong-writer topology:

```text
one HTTP API service process owns the SQLite database for a shared Amos instance
service-level locking plus SQLite transactions for strong canonical mutations
row-level expected_version checks for atoms and edges
unique constraints for idempotency keys
event journal as the transactional outbox for later projections
in-process or service-coordinated workers for derived indexes and packet cache refresh
```

Multiple API instances, direct multi-process writers, read replicas, cluster
consensus, multi-primary writes, and cross-region active-active replication are
post-v1 concerns. A v1 deployment should share AMOS by routing clients through
the API service rather than by allowing several processes to write to the same
SQLite file independently.

### 1.4 Initial ontology governance

V1 ships with a small relation and entity-type dictionary.

Seed entity types:

```text
user
agent
service
project
workspace
tool
file
document
concept
policy
procedure
task
environment
capability
limitation
runtime_state
self_assessment
agentic_trace
action_outcome
self_narrative
organization
```

Seed relation IDs:

```text
rel:prefers
rel:avoids
rel:requires
rel:forbids
rel:working_on
rel:owns
rel:uses
rel:depends_on
rel:supports
rel:contradicts
rel:supersedes
rel:derived_from
rel:applies_to
rel:caused_by
rel:similar_to
rel:part_of
rel:has_capability
rel:has_limitation
rel:currently_available
rel:currently_denied
rel:made_commitment
rel:satisfied_commitment
rel:miscalibrated_on
rel:decided
rel:acted_on
rel:produced_outcome
rel:corrected_by
rel:constrained_by
rel:attributed_to
rel:shared_responsibility_for
```

Dictionary update policy:

```text
system-defined relation:
  added only through schema/dictionary migration

tenant-defined relation:
  allowed only for tenant admins

agent-defined relation:
  not allowed in v1; agents may propose relation candidates for review

deprecated relation:
  remains resolvable while retained records reference it
```

### 1.5 Default retrieval weights

V1 activation score starts with transparent hand-tuned weights. Implementations should normalize each score component to `0.0..1.0`, multiply by the configured weight, subtract penalties, then clamp the final score to `0.0..1.0`.

```text
direct_cue_match: 0.22
semantic_similarity: 0.14
edge_activation: 0.12
recency: 0.08
confidence: 0.12
utility: 0.12
salience: 0.08
goal_relevance: 0.08
procedural_applicability: 0.04
attention_focus: 0.14
attention_type_boost: 0.08
attention_counterevidence: 0.08
attention_novelty: 0.05

contradiction_penalty: -0.30
staleness_penalty: -0.18
attention_suppression_penalty: -0.20
access_penalty: suppress
redundancy_penalty: -0.15
confounding_penalty: suppress unless explicitly requested
```

Agentic recall ranking profile additions:

```text
agency_match: 0.16
attribution_confidence: 0.12
correction_learning_relevance: 0.10

over_attribution_penalty: -0.25
omitted_counterevidence_penalty: -0.25
ignored_failure_penalty: -0.20
```

When the `agentic_recall` profile is active, positive weights must be renormalized so agency-related cues do not crowd out scope, evidence, contradiction, and current-runtime constraints.

Calibration loop:

```text
1. Start with default weights.
2. Log score_components and retrieval outcomes.
3. Review failures where correction_after_use_count increases.
4. Tune weights by offline replay fixtures before changing production defaults.
5. Version ranking profiles as ranker_profile_id.
```

### 1.6 Default packet budgets

V1 default packet budgets:

```text
reasoner:
  max_items: 24
  token_budget: 6000
  include_conflicts: true
  include_provenance: compact

planner:
  max_items: 20
  token_budget: 4500
  include goals, commitments, constraints, procedures

executor:
  max_items: 16
  token_budget: 3500
  include procedures, permissions, tool constraints, recovery notes

critic:
  max_items: 32
  token_budget: 8000
  include evidence refs, conflicts, retrieval telemetry, health flags

steward:
  max_items: 64
  token_budget: 12000
  include maintenance history, health candidates, graph neighborhoods

self_awareness:
  max_items: 100
  token_budget: 24000
  include self-model, current runtime state, capabilities, limitations,
  open commitments, uncertainties, recent errors, and evidence refs
  required role self-awareness fields are structural and are not dropped
  by generic packet budget ordering

shared_coordination:
  max_items: 48
  token_budget: 9000
  include shared goals, commitments, constraints, common assumptions,
  conflicts, audience, ownership, and per-agent overlay refs

agentic_recall:
  max_items: 40
  token_budget: 7000
  include agentic traces, action outcomes, corrections, limitations,
  external constraints, attribution counterevidence, and optional SelfNarrative refs
```

Packets must enforce both item and token/byte budgets. If budget limits suppress relevant material, the packet must include `budget_exhausted` in `omissions`.

### 1.7 Retention control defaults

Retention-class authority:

```text
ephemeral:
  system-controlled

session:
  system-controlled with user-visible deletion

recent:
  system-controlled with admin-configurable TTL

project:
  admin-configurable

durable:
  user-configurable or admin-configurable, depending on owner scope

compliance:
  compliance-controlled; cannot be shortened by agents

user_pinned:
  user-controlled unless compliance/legal policy forbids retention
```

Default retention windows:

```text
ephemeral:
  minutes to 24 hours

session:
  until session close plus configured grace period

recent:
  30 to 90 days

project:
  life of project plus configured archive period

durable:
  indefinite until corrected, deleted, or superseded

compliance:
  deployment policy

user_pinned:
  indefinite until user unpins or deletes
```

### 1.8 Backup and deletion default

**Status: Partial.** V1-local implements logical atom deletion, tombstones,
edge suppression, hot-index cleanup, and packet-cache invalidation. It reports
residual-retention policy but does not own an evidence object store, canonical
snapshots, encryption keys, or offline backups.

Deployments integrating those external stores must provide this policy:

```text
hot database:
  logical delete and tombstone immediately through the strong write path

evidence archive:
  encrypt evidence objects by tenant/workspace key where feasible

snapshots:
  store deleted-sensitive payloads encrypted or rebuild snapshots after deletion

offline backups:
  configure and disclose a maximum residual retention window

crypto-shredding:
  delete tenant/workspace/object key to make encrypted evidence unrecoverable
```

The v1-local response currently discloses a default 30-day offline-backup
residual window; that is a policy declaration, not backup-system enforcement.
An integrating deployment must prevent restored backups from reintroducing
deleted memories without replaying deletion events.

### 1.9 Journal rollup format

**Status: Planned.** V1-local replays the retained journal from genesis. It does
not create canonical graph snapshots or compacted journal segments. A future
compaction implementation should use this rollup shape:

```text
CompactedJournalSegment
  segment_id
  schema_version
  tenant_id
  workspace_id
  covers_event_ids
  covers_event_range
  start_graph_version
  end_graph_version
  resulting_atom_refs
  resulting_edge_refs
  tombstone_refs
  evidence_rollup_refs
  aggregate_counters
  policy_decisions
  checksum_chain_start
  checksum_chain_end
  created_at
  created_by
```

Rollups are audit accelerators, not replacements for deletion policy. If raw events are deleted or shredded, the rollup must not retain forbidden payload content.

### 1.10 V1 evaluation suite

Implementation must include deterministic fixtures for:

```text
schema validation:
  reject payloads that duplicate envelope fields

journal projection:
  replay atom commit, update, merge, archive, tombstone, delete

idempotency:
  same key and same payload returns same result; same key and different payload fails

CAS conflict:
  stale expected_version produces conflict result

scope isolation:
  project A cannot retrieve project B memory without explicit policy

evidence visibility:
  atom can be visible while evidence detail is denied

retrieval ranking:
  scoped preference beats generic preference

contradiction rendering:
  contradicted memories appear only with conflict context

self-awareness retrieval:
  stale capability is suppressed when runtime state says unavailable

agentic recall attribution:
  an agent retrieves its own traces and outcomes, while another agent's actions
  appear only as other_agent or shared_system responsibility

agentic recall balance:
  successes, failures, blocked actions, corrections, limitations, and external
  constraints are represented when relevant to the self-claim

self-narrative drift:
  generated SelfNarrative expires or rebuilds after contrary AgenticTrace,
  ActionOutcomeRecord, or SelfAssessmentRecord evidence

shared-view convergence:
  reasoner, planner, and executor receive the same common goal and constraint refs
  at the same graph_version, with different role overlays

shared-view access filtering:
  an agent without evidence permission sees an omission reason while another
  authorized agent sees the evidence ref in its overlay

self-report calibration:
  claimed capability without successful verification increases overconfident_claim_rate

commitment tracking:
  open commitment appears in self-awareness packet until fulfilled, cancelled, or superseded

deletion:
  deleted atom is absent from packets, indexes, caches, and replayed graph

tombstone prevention:
  deleted memory is not recreated from old evidence when recreation_policy forbids it

capacity degradation:
  orange/red pressure reduces recall and marks packet degradation

SMP authority:
  high-risk SMP recommendation requires review before mutation

automatic memory policy:
  background worker ticks and explicit operator runs perform deterministic
  maintenance, create provenance-linked distilled atoms, refresh derived
  indexes, remove expired cold atoms, compact SQLite storage when idle,
  invalidate packet cache, persist policy state, and journal memory_policy_run
  events without an LLM
```

### 1.11 V1 automatic memory policy

V1 memory maintenance is a service-owned policy, not an integration-specific
manual bridge call.

Default policy:

```text
enabled:
  true

schedule:
  every_graph_versions: 25
  every_seconds: 300
  run_on_pressure: true

maintenance:
  run_smp: true
  run_steward: true
  rebuild_indexes: true
  invalidate_packet_cache: true

distillation:
  enabled: true
  min_source_atoms: 6
  max_source_atoms: 10
  candidate_types:
    action_outcome
    agentic_trace
    belief
    episode
    preference
  distillation_type: automatic_policy
  archive_sources: false
  approved_by: null

maintenance_distiller:
  enabled: true
  auto_commit_low_risk: true
  processor_ids: []
  domain: generic
  max_atoms: 128
  max_events: 64
  max_retrieval_outcomes: 64
  reviewer:
    enabled: false
    authority: draft_only

decay:
  enabled: true
  max_atoms: 256
  max_active_atoms: 256
  max_proposed_atoms: 256
  require_atom_policy: true
  pressure_archive_policyless: true
  pressure_archive_proposed: true
  proposal_pressure_min_age_seconds: 3600
  archive_proposed_after_seconds: null
  pressure_max_archives_per_run: 256
  pressure_protected_types:
    - commitment
    - policy
    - self_model
  capacity_assessment_targets:
    - 256
    - 512
    - 768
  capacity_headroom_ratio: 0.2
  archive_superseded: true
  archive_superseded_after_seconds: 0
  mark_stale_after_seconds: null
  archive_after_seconds: null
  low_utility_threshold: null

storage_cleanup:
  enabled: true
  trigger: idle
  idle_after_seconds: 300
  min_interval_seconds: 900
  max_deletions_per_tick: 256
  remove_archived_from_hot_index: true
  remove_stale_from_hot_index: true
  delete_archived_after_seconds: 604800
  delete_stale_after_seconds: 1209600
  protected_types:
    - policy
    - self_model
    - commitment
  compact_idempotency_after_seconds: 604800
  max_idempotency_compactions_per_tick: 512
  sqlite_compaction:
    checkpoint_wal: true
    checkpoint_mode: TRUNCATE
    vacuum_enabled: true
    vacuum_idle_after_seconds: 1800
    vacuum_min_interval_seconds: 86400
```

The v1 HTTP service starts a background memory-policy worker. Foreground
service calls do not need to complete maintenance before responding:

```text
GET /v1/health/memory
  reports memory health and background worker status
  includes quality diagnostics for active atom pressure, active superseded
  atoms, isolated active atoms, and derived-index graph lag
  does not run a policy tick inline

POST /v1/packets:retrieve
  returns a packet from the current graph view
  queues a background policy tick when run_policy is true

POST /v1/memory-policy:run
  runs the policy synchronously as an explicit operator/admin action
```

The in-process service API still exposes `run_memory_policy()` and
`retrieve_packet(run_policy=True)` for tests, CLI use, and embedded deployments
that intentionally want a synchronous tick. The shared-service contract is that
connected agents do not own lifecycle maintenance themselves.

Policy scope is interpreted differently for retrieval and service-owned cleanup.
An empty retrieval scope sees only global/unscoped memory. An empty service-owned
decay or storage-cleanup scope means whole-store maintenance, so tenant- or
run-scoped superseded atoms are still archived by the background worker. Passing
an explicit scope narrows the maintenance pass to that scope.

When due, the policy should:

```text
run SMP analysis
run the memory steward for low-risk reversible cleanup
execute deterministic decay rules from atom decay_policy and configured global
  bounds
archive active atoms that are superseded by active replacement atoms when
  archive_superseded is enabled
create provenance-linked semantic distillations when enough eligible source
  atoms are available
build a bounded evidence window and run registered maintenance processor packs
read canonical `semantic_facets` and `graph_relations` directly from active
  atoms through the built-in generic processor
commit only low-risk, policy-allowed proposals such as add_atom distillations
  and explicit structural graph relations with active endpoints
defer medium/high-risk proposals, health changes, merges, archives, access
  policy changes, and ambiguous claims to explicit review
prune archived/stale atoms from hot retrieval indexes during idle cleanup
delete expired archived/stale atoms through normal tombstone and journal
  projection paths while preserving protected types
compact old idempotency responses so duplicate-response cache rows do not
  dominate the hot SQLite file
checkpoint the SQLite WAL and run VACUUM only after the configured idle window
  and compaction interval
refresh rebuildable derived-index metadata
refresh dependency-free lexical and LSA derived vector indexes
invalidate packet cache
persist memory_policy_state
append a memory_policy_run event
```

Canonical graph metadata is producer-normalized, deterministic input rather
than an AMOS inference from prose. `semantic_facets` declare subject, intent,
outcome direction, confidence, provenance, and optional time/metric/control
dimensions. `graph_relations` declare ontology relation ids and endpoints,
using `$self` for the owning atom. AMOS validates both structures at ingestion,
ignores them while the owning atom is proposed, and re-evaluates them on later
maintenance passes after authorized promotion. Structural relations may be
auto-committed; causal and other non-low-risk declarations remain reviewable.
Intrinsic and explicitly declared edges carry the owning atom's evidence and
confidence by default. A relation may supply narrower evidence/confidence
directly; graph replay preserves those values. Steward maintenance refreshes
legacy intrinsic edges by merging newly available provenance and retaining the
stronger confidence instead of leaving structurally current edges evidentially
empty.

Decay execution is deterministic and non-generative. By default, v1-local only
applies time, utility, and expiry rules to active atoms with explicit atom-level
`decay_policy` rules, except for active atoms superseded by active replacements
when `archive_superseded` is enabled. `max_atoms` bounds the complete hot set,
while `max_active_atoms` and `max_proposed_atoms` independently bound canonical
and proposal populations. When any bound is exceeded, the worker first archives
the minimum eligible proposal excess, then the minimum eligible policyless
active excess. Proposed atoms are pressure-eligible only when their producer
attached `payload.proposal_retention`; `proposal_pressure_min_age_seconds`
prevents newly generated review work from being removed immediately.

The same producer-owned retention object may declare a stable
`deduplication_key` and `archive_after_seconds`. Maintenance archives duplicate
proposals only when that explicit key, atom type, and scope match, retaining the
candidate with stronger evidence and then stable age/id ordering. It archives a
proposal after its explicit retention window; the global
`archive_proposed_after_seconds` is an optional fallback. AMOS does not infer
semantic equivalence from prose and does not delete the producer's separate
audit history. Active pressure cleanup preserves `decay_policy.enabled = false`,
future `retain_until` rules, recognized explicit atom decay rules, and configured
`pressure_protected_types`. Candidates are ordered deterministically: isolated
before connected, unhealthy before healthy, then lower utility, lower salience,
older access/update time, and atom id. `pressure_max_archives_per_run` bounds each
maintenance transaction; residual pressure remains visible for the next tick.
Memory health also reports a non-mutating `capacity_assessment` across the
configured candidate targets. It recommends the smallest target that preserves
`capacity_headroom_ratio` above the current active count and warns when the
configured target no longer has that headroom. Deployments should use this
signal with retrieval-quality and maintenance-latency measurements rather than
raising `max_atoms` solely to suppress pressure warnings.

Capacity diagnostics retain the v1-local hot-set definition: lifecycle-active
atoms plus dormant proposals. `quality.lifecycle_counts` makes that composition
explicit. `quality.hot_atom_*`, `quality.lifecycle_active_atom_*`, and
`quality.proposed_atom_*` expose each count and configured bound; the legacy
`quality.active_atom_count` field remains a hot-total compatibility alias.
Graph isolation is narrower: `isolated_active_atoms` contains only
lifecycle-active canonical atoms, while `isolated_proposed_atoms` reports
proposal isolation separately as expected dormant state and does not raise the
active-graph isolation warning.
`quality.graph_quality` additionally reports active connected components,
atom-type and relation distributions, top-degree concentration, confidence and
edge-derivation histograms, and unresolved reference samples. Proposal age,
explicit deduplication, covered-source counts, and recent per-processor proposal,
commit, deferral, and idempotent-replay totals are exposed separately. These are
diagnostics, not deterministic semantic acceptance rules.
Supported v1-local rules include `expires_at`, `retain_until`,
`mark_stale_after_seconds`, `archive_after_seconds`, and
`low_utility_threshold`; operator policy can relax `require_atom_policy` to
apply global stale/archive/low-utility thresholds.
Applied decay actions are journaled as `decay_policy_applied`, update atom
version/health/lifecycle state, refresh derived token rows, and invalidate packet
cache.

Storage cleanup is deterministic and idle-triggered, not size-triggered by
default. It removes archived/stale atoms from the hot token index immediately
when the cleanup tick is due, then deletes archived/stale atoms only after their
configured retention windows. Deletion uses the same deleted lifecycle,
tombstones, edge deletion, packet-cache invalidation, and replay projection model
as explicit `atom_deleted` operations. The event journal remains logically
append-only; v1-local physical compaction trims derived/cache storage with
idempotency-response slimming, `PRAGMA wal_checkpoint`, and SQLite `VACUUM`.

V1-local policy execution profile:

```text
SMP analysis:
  analyzes active eligible atoms
  bounds pairwise semantic link analysis to likely candidate pairs
  ranks candidates by type compatibility, lexical overlap, current semantic
  vector similarity, recency, and policy limits rather than comparing every atom
  to every other atom

maintenance evidence window:
  uses configured limits for atoms, events, and retrieval outcomes
  lets each processor request a narrower lifecycle/type/profile workset without
  widening caller scope or limits
  applies hierarchical maintenance scope to atoms and evidence, so whole-store
  or tenant passes include evidence retained at narrower run/project scopes
  may add bounded graph neighbors after typed candidates
  prioritizes directly referenced evidence before other visible evidence
  reports candidate, truncation, lifecycle/type, graph-boundary, and unresolved
  evidence coverage per processor
  keeps processor packs side-effect-free

retrieval policy scheduling:
  packet retrieval may enqueue policy work but should not block on SMP,
  stewardship, distillation, index refresh, or cache invalidation in HTTP mode

health reporting:
  reports graph size, event count, edge count, pressure, stale indexes,
  background worker status, and policy due state using bounded reads
```

Automatic distillation is non-LLM. It creates a canonical semantic memory with
source refs, policy metadata, `layer = consolidated_long_term`, and
`retention_class = distilled`. Source archival is disabled by default unless an
explicit approval policy enables it. Candidate selection excludes source atoms
already covered by active derived semantic memory and records assigned to the
`domain_processor` distillation lane. Remaining candidates are partitioned by
an explicit producer cohort when present, otherwise by the conservative
scope/type/profile/kind fallback; one packet never spans those coherence groups.

The memory policy is observable and tunable through the V1 HTTP API and CLI.
Manual `run` operations are operator overrides or worker ticks; they are not
the primary maintenance path.

#### 1.11.1 Processor-pack distiller worker

The V1 distiller worker generalizes SMP beyond the built-in AMOS cleanup
heuristics. A processor pack receives a bounded `EvidenceWindow` and returns
side-effect-free `MaintenanceProposal` records. The AMOS service, not the
processor, applies policy gates and journaled mutations.

Evidence window:

```text
atoms:
  processor-requested visible lifecycle/type/profile workset in scope
edges:
  visible graph neighborhood for selected atoms
evidence:
  visible supporting evidence records
retrieval_outcomes:
  recent packet-use telemetry
events:
  recent event-journal entries
scope:
  request scope
domain:
  generic or domain-specific label
graph_version:
  source graph version
coverage:
  candidate, selected, truncated, edge-boundary, and evidence-resolution counts
```

Maintenance proposal:

```text
proposal_id:
  stable id derived from processor, action, source refs, and payload digest
processor_id:
  registered deterministic processor pack supplied by AMOS or by an installed
  client package
processor_version:
  pack version
action:
  add_atom | mark_health | review_cluster | review_conflict | review_required | ...
risk_level:
  low | medium | high
confidence:
  bounded numeric confidence
reason_code:
  machine-readable reason
source_refs:
  canonical refs that support the proposal
target_refs:
  refs that would be changed, if any
evidence_refs:
  supporting evidence refs
payload:
  proposed mutation or review details
```

Only `risk_level = low` with `action = add_atom` or `action = add_edge` is
auto-committable in V1. A committed atom must be a derived canonical atom with
source refs, proposal id, processor id, and policy metadata. A committed edge
must have two active endpoints, allowed relation semantics, evidence/confidence,
and a derivation record naming the processor and proposal. Other actions are
returned as deferred proposals and journaled in `maintenance_distillation_run`;
they are not silently discarded.

The default registry in the AMOS package includes only generic AMOS
maintenance:

```text
amos.maintenance.generic.v1:
  adapter around the deterministic SMP for health, duplicate, and conflict
  proposals
```

Client packages may register additional processors in-process or load them at
service startup through import paths. Those processors must live in the client
package, implement the generic `MaintenanceProcessor` contract, and return
side-effect-free `MaintenanceProposal` records. This keeps AMOS reusable across
agents: integrations write typed atoms and evidence into shared AMOS; client
processors inspect those canonical records; low-risk derived memories are
committed through the shared service policy; high-risk or ambiguous changes
remain review items.

A processor may optionally implement `window_request(scope=..., domain=...)`
and return a `MaintenanceWindowRequest`. Lifecycle, atom-type, producer-profile,
neighbor, evidence/event/outcome, and size preferences narrow its workset; the
service still enforces the caller's scope and ceilings. Shared helpers expose
explicit producer hints, cohort keys, source coverage, evidence diversity, and
idempotent derived-memory proposal construction without interpreting free prose.

Every newly projected edge carries `derivation`. Intrinsic/canonical structural
edges, explicit producer relations, semantic-facet associations, and
processor-reviewed relations use distinct derivation kinds. Existing databases
receive an explicit `migrated_relation_classification` marker; migration does
not pretend to know an unavailable historical producer.

Processor packs should promote repeated, evidence-backed experiences into
compact reusable memories rather than copying every source event into a prompt
surface. Source atoms and evidence remain auditable; the promoted atom carries
the operational lesson.

V1 local tooling exposes this boundary through the CLI and HTTP service
constructor. Operators pass external processors as `module:attribute` import
paths, list registered processors before running a domain-specific distiller,
and select processors by stable `processor_id`. The AMOS package itself should
not claim to bundle client-domain packs such as training-flight processors
unless those processors live in the package and are registered by default.

### 1.12 Procedural memory execution policy

V1 procedural memory is advisory by default.

```text
advisory procedure:
  can be rendered to reasoner, planner, or executor as guidance

executable procedure:
  requires explicit approval, tool permission binding, precondition schema,
  rollback/recovery plan, and review_status approved

autonomous execution:
  not allowed in v1 for procedures that change external state
```

The executor may use procedures to choose actions only within its existing tool permissions and policy constraints.

### 1.13 LLM reviewer default

V1 LLM reviewer default:

```text
enabled_by_default:
  false

allowed_when_enabled:
  ambiguous atomization
  scope refinement suggestions
  contradiction analysis suggestions
  natural-language explanation drafting

forbidden:
  direct canonical mutation
  deletion approval
  access-policy change
  autonomous preference alteration
```

LLM reviewer output must use the same SMP output envelope: `processor_id`, `processor_version`, `input_refs`, `output_type`, `confidence`, `reason_code`, `evidence_refs`, `recommended_action`, and `risk_level`.

### 1.14 Post-v1 extension points

Post-v1 work may add a production Postgres backend, multi-instance service
deployment, direct worker processes, multi-primary replication, learned ranking,
generated Protobuf or Avro artifacts, executable procedures, and default
LLM-review workflows. Those changes must preserve the v1 journal, replay,
deletion, authorization, and packet contracts unless a versioned migration
explicitly replaces them.

---

## 2. Implementation status and acceptance evidence

The v1-local implementation currently includes these concrete repository artifacts from the contracts above:

```text
schema artifacts:
  MemoryAtom envelope schema
  typed payload schemas
  SourceEvent schema
  EvidenceRecord schema
  AssociationEdge schema
  EventJournalEntry schema
  MemoryPacketRequest and MemoryPacket schemas
  SharedMemoryView schema
  SelfModelAtom, CapabilityAtom, LimitationAtom schemas
  RuntimeStateSnapshot and SelfAssessmentRecord schemas
  AgenticTrace, ActionOutcomeRecord, and SelfNarrative schemas
  AccessPolicy and Scope schemas
  DeletionRequest and MemoryTombstone schemas
  CapacityBudget and CapacityHealthReport schemas
  SMP output schema

storage artifacts:
  SQLite migration for the v1 service-owned store
  optional/future Postgres migration contract
  event_journal migration
  atoms migration
  edges migration
  evidence metadata migration
  tombstones migration
  packet cache metadata migration
  derived index metadata migration
  rebuildable atom token candidate index
  retrieval outcome telemetry table

service artifacts:
  capture_event endpoint
  propose_memory_atoms endpoint
  commit_memory_atoms endpoint
  archive_atom and merge_atoms endpoints
  retrieve_memory_packet endpoint
  record_retrieval_outcome endpoint
  retrieval outcome utility/salience feedback loop
  request_maintenance endpoint
  memory-policy status/configure/run endpoints
  memory-policy decay executor
  maintenance-processor listing endpoint
  maintenance-distiller endpoint
  deletion endpoint
  runtime-state endpoint
  self-assessments endpoint
  self-awareness retrieval endpoint
  agentic-recall retrieval endpoint
  shared-view retrieval and refresh endpoints
  procedure execution-policy endpoint
  capacity configure endpoint
  capacity health endpoint
  memory health endpoint
  deterministic SMP analysis endpoint
  LLM reviewer policy endpoint
  journal/replay verification endpoint
  stdlib HTTP adapter
  CLI commands for init, capture, commit, retrieve, self-awareness,
  agentic-recall, steward, distill, merge, maintenance, memory policy,
  maintenance processors, capacity, SMP analysis, health, verify, and serve

active worker artifact:
  background memory policy worker

in-process operation adapters:
  journal verification adapter
  index maintenance adapter
  packet cache invalidation adapter
  capacity governor adapter
  memory steward adapter
  self-model calibrator
  agentic recall auditor
  SMP adapter
  synchronous memory policy adapter
  distiller maintenance adapter
```

Acceptance status:

| Gate | Status | Verified v1-local boundary |
| --- | --- | --- |
| Schema | Implemented | Envelope fields are excluded from payloads; required fields, JSON Schema property types, and score bounds are enforced by dependency-free runtime validators and tests. |
| Journal | Implemented | Canonical graph mutations append checksummed `EventJournalEntry` records with authorization and expected-version context. |
| Projection | Implemented | Strong mutations append their event and project graph changes in one SQLite transaction. |
| Replay | Partial | The graph is reconstructable from the full retained journal. Snapshot-plus-tail recovery and journal segment compaction are planned. |
| Retrieval and attention | Implemented | Packets expose graph version, provenance, omissions, degradation, score components, attention trace, bounded candidate selection, and scoped edge activation. |
| Self-awareness and agentic recall | Implemented | Structural self views, responsibility attribution, counterevidence, and generated self-narrative expiry are tested. |
| Shared memory | Implemented | Common graph-version views and identity-specific overlays/omissions are tested. |
| Authorization | Implemented | Scope, visibility, mutation roles, trust, capabilities, and evidence visibility are independently exercised. |
| Deletion | Partial | V1-local suppresses atoms and edges, clears hot derived state, and creates tombstones. External evidence archives, snapshots, encryption keys, and backups remain integration responsibilities. |
| Capacity | Partial | One SQLite-file byte budget drives one pressure mode and packet degradation. Per-tier budgets and external-object-store capacity are planned. |
| SMP and processor packs | Implemented | Deterministic processors produce advisory proposals; only policy-approved low-risk atoms auto-commit. |
| Memory policy | Implemented | Background and operator-triggered deterministic maintenance paths are tested without an LLM. |
| Performance | Evidence only | A reproducible local benchmark exists, but CI has no scale or latency acceptance threshold; this is not a production-scale guarantee. |
| Observability | Implemented with declared constants | Health, capacity, index freshness, retrieval outcomes, and deletion residuals are reportable. Projection lag is always zero in the single-process transactional profile. |

[`v1-verification.md`](v1-verification.md) maps implemented and partial gates to
the exact artifacts and tests. Planned portions are not represented as passing
acceptance gates.

---
