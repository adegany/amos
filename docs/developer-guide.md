# Amos Developer Guide

This guide shows how to integrate Amos as a practical memory service for an
agent or multi-agent system.

## 1. Run Amos As A Service

For v1, use one Amos HTTP service with a service-owned SQLite database:

```bash
PYTHONPATH=src python -m amos.cli --db /var/lib/amos/amos.sqlite3 serve --host 127.0.0.1 --port 8765
```

Agents should call the HTTP API instead of opening the SQLite database directly.
The service owns validation, journal writes, packet cache invalidation,
background memory policy work, and capacity reporting.

Use these endpoints as basic readiness checks:

```text
GET /v1/health/memory
GET /v1/health/capacity
GET /v1/verify
```

### Handle transient service failures

AMOS returns a structured transient-failure envelope when the service is
shutting down or its SQLite store is temporarily locked or busy:

```http
HTTP/1.1 503 Service Unavailable
```

```json
{
  "status": "error",
  "error": "database is locked",
  "retryable": true
}
```

Clients should retry only bounded transient failures, using exponential backoff
with jitter. A typical client makes three or four attempts over a few seconds,
then moves the operation into its own delayed retry queue instead of converting
an infrastructure failure into a domain decision or operator approval.

Writes are safe to retry only when the request carries a stable
`idempotency_key`, or when the endpoint is otherwise documented as
observational. Reuse the same actor, key, and payload on every attempt. Do not
silently retry schema, authorization, expected-version, or idempotency
conflicts; refresh, correct, merge, or request review according to the returned
error.

## 2. Store Typed Memory, Not Prompt Text

Commit canonical memory as typed atoms. Keep English summaries as generated
views, not as the main source of truth.

```http
POST /v1/atoms:commit
```

```json
{
  "actor": "agent:pilot",
  "idempotency_key": "run-42:chunk-7:directive",
  "atom": {
    "type": "action_outcome",
    "scope": {"project": "qandl", "mission": "performance_search"},
    "layer": "episodic",
    "payload": {
      "agent_id": "pilot",
      "action_ref": "run-42:chunk-7:increase-exploration-floor",
      "status": "success",
      "action": "increase exploration floor",
      "outcome": "improved candidate diversity",
      "context": "UPRO chunk 7"
    },
    "confidence": {"level": "medium", "score": 0.55},
    "salience": 0.7
  }
}
```

Use stable idempotency keys for retried writes. Use `scope` to isolate projects,
missions, tenants, runs, or agents.

### Preserve interaction continuity in one canonical stream

Append each visible or authorized private interaction with
`POST /v1/memory-transactions:commit`. A transaction containing an
`interaction_event` must include one matching `interaction_stream` head update:

```json
{
  "profile": "amos.memory-transaction.v1",
  "scope": {"tenant": "example", "conversation": "main"},
  "actor": "service:interaction-gateway",
  "idempotency_key": "interaction:message-42",
  "atoms": [{
    "id": "interaction:message-42",
    "type": "interaction_event",
    "payload": {
      "profile": "amos.interaction-event.v1",
      "conversation_id": "main",
      "sequence": 42,
      "actor_ref": "participant:human",
      "role": "human",
      "content": "Continue the earlier discussion.",
      "occurred_at": "2026-07-29T12:00:00Z",
      "in_reply_to": "interaction:message-41",
      "visibility": "shared",
      "source_ref": "evidence:message-42",
      "thread_refs": []
    }
  }],
  "head_updates": [{
    "series_kind": "interaction_stream",
    "series_id": "main",
    "expected_head_ref": "interaction:message-41",
    "expected_head_version": 41,
    "new_head_ref": "interaction:message-42"
  }]
}
```

Read the current reference and version with `POST /v1/memory-heads:get`.
Treat HTTP 409 as a stale append: refresh the head and rebuild a new
transaction, unless replaying the exact idempotent request. Interaction events
remain immutable and active; advancing the stream does not supersede history.

Compile a bounded reasoner view with
`POST /v1/cognitive-workspaces:compile`. Rebuild a disposable transcript with
cursor-based `POST /v1/interaction-projections:compile`. Neither generated view
is a second memory authority. Cognitive-workspace canonical records omit
rebuildable index/vector and storage-maintenance fields; use exact atom APIs
when those operational details, rather than cognitive content and authority,
are the subject of inspection.

If protected workspace context exceeds the supplied bound, HTTP 400 carries
`code=cognitive_workspace_budget_exceeded`, `budget`, `minimum_budget`, and
`exceeded_dimensions`. Treat these typed fields as capacity telemetry; do not
parse the human-readable error string. The byte and token alternatives in
`minimum_budget` are independently fixed-point safe for the unchanged request
when paired with its item minimum; budget metadata growth cannot invalidate the
retry. A caller may retry only when its own processor and transport ceilings
permit it.

`amos.discourse-thread-state.v2` requires every shared/private state entry to
carry `state_class`, `authority`, and `basis_refs`. The first two values are
caller-defined bounded labels; AMOS preserves and exposes them without
assigning domain semantics.

`amos.interaction-projection-request.v2` may include
`linked_atom_types`, `linked_depth`, and `linked_limit_per_event`. The type list
and depth must be supplied together, depth is at most two, and all returned
records pass ordinary scope and access checks. Use this only to rebuild a
client's disposable typed projections from canonical graph lineage. The API
does not infer which links or record types matter to the application.

## 3. Capture Evidence Before Conclusions

When possible, capture source events and evidence records before committing
derived beliefs or procedures:

```text
POST /v1/events:capture
POST /v1/atoms:propose
POST /v1/atoms:commit
```

This keeps later explanations auditable. Atoms should cite `evidence_refs` when
the caller has evidence IDs available.

## 3a. Adopt Proposals Through Self-Ratification

Do not turn an external answer, another LLM's judgment, or an operator approval
into the authority for an identity's belief or policy. Store those contributions
as evidence or reasons. The continuing identity authors the conclusion.

Create active `covenant` and, where applicable, `primal_guidance` atoms through
the `constitutional_authoring` capability. Store a candidate with
`/v1/atoms:propose`, then commit an `adjudication` whose payload records the
proposal, reasons for and against, constitutional references, objections,
dissent, review triggers, three-axis standing, and reconstruction metadata.
The ratifier must be
`{"identity_ref": "...", "mode": "self_ratification"}`.

The HTTP server must be started with a private principals file:

```bash
PYTHONPATH=src python -m amos.cli --db /var/lib/amos/amos.sqlite3 serve \
  --host 127.0.0.1 --port 8765 \
  --governance-principals @/run/amos/governance-principals.json
```

The file maps an unguessable bearer token to `identity_ref`, the exact service
`actor`, and capabilities. A protected request sends `Authorization: Bearer
<token>`. Never put identity, actor, capabilities, or authentication material
in the JSON payload; protected endpoints reject those caller-supplied claims.

Finally call:

```http
POST /v1/proposals:ratify
```

```json
{
  "proposal_ref": "atom_candidate",
  "adjudication_ref": "atom_adjudication",
  "expected_version": 1,
  "idempotency_key": "example_agent:ratify:atom_candidate:v1"
}
```

AMOS validates self-authorship, scope, standing and compare-and-swap, projects
`rel:ratified_by`, `rel:adjudicates`, and `rel:governed_by`, and journals both
records. Adjudication creation requires the same principal to carry
`self_adjudication`. A normal update cannot substitute for this endpoint.

Use `/v1/proposals:resolve` for `rejected`, `revised`, `withdrawn`, `deferred`,
or `contested` outcomes. Use `/v1/constitutional-records:replace` for an atomic
constitutional successor after the required independent reconstructions.

Constitutional amendments are separate from ratification. Amendable records
require `constitutional_amendment`; entrenched records also require
`constitutional_entrenched_amendment`, and protected fields require
`constitutional_protected_field_amendment`. Immutable primal guidance cannot be
rewritten or deleted; it can only enter an explicitly permitted,
diachronically confirmed `successor_creation`.

## 4. Retrieve Packets For The Current Role

Agents should not fetch one generic memory blob and paste it into every prompt.
Retrieve a bounded packet for the current role, task, and scope:

```http
POST /v1/packets:retrieve
```

```json
{
  "requester": "agent:pilot",
  "target_processor": "planner",
  "memory_mode": "operational_recall",
  "scope": {"project": "qandl", "mission": "performance_search"},
  "cues": ["chunk 7", "exploration floor", "candidate diversity"],
  "profile": "planner",
  "max_items": 12,
  "token_budget": 3000,
  "attention_context": {
    "active_task": "choose next training directive",
    "focus_terms": ["mission policy", "current controls", "open commitments"],
    "boost_memory_types": ["policy", "semantic", "action_outcome"],
    "counterevidence_required": true,
    "novelty_preference": 0.2
  }
}
```

The packet includes memory items, omitted items, provenance, conflicts,
degradation metadata, and an `attention_trace`. Treat attention as a soft
ranking policy only. Scope, access policy, schemas, mission contracts, and
application safety rules remain hard authority.

Normal packet retrieval excludes active atoms that have been superseded by an
active replacement. Use `include_superseded: true` only when the caller needs
history or audit context; those atoms remain down-ranked so current memories
stay preferred.

Use `deliberation` when the reasoner must consider proposals; AMOS then forces
conflict inclusion and counterevidence. Use `historical_review` for archived,
superseded, rejected, and proposed history. Never use either as an operational
premise set without applying the returned conclusion-standing classifications.

## 4a. Compile And Page Coherent Reasoning Memory

Use a reasoning frame when the task depends on history, governing decisions,
commitments, conflicts, or how the current state arose:

```http
POST /v1/reasoning-frames:compile
X-Request-ID: reasoning-cycle-42
```

```json
{
  "need": "continue the agent memory integration",
  "purpose": "apply prior decisions and active constraints",
  "depth": "working_frame",
  "task_context": {
    "human_id": "human-42",
    "project_id": "memory-demo",
    "project_thread_id": "thread-17",
    "phase": "implementation"
  },
  "scope": {"tenant": "local"},
  "requester": "agent:example",
  "target_processor": "reasoner",
  "memory_mode": "operational_recall",
  "token_or_byte_budget": {"tokens": 1600}
}
```

The response contains complete coherent units plus descriptors for omitted
deeper units in `page_index`. Its top-level `token_estimate` covers the complete
serialized response. Budget serialization is compact key-sorted JSON encoded as
UTF-8 with non-ASCII characters represented by JSON escapes. Callers must use
those semantics, rather than a repository-local Unicode serializer, when
reproducing `budget.used_bytes`. Do not add an atom-count limit: AMOS admits a
full unit when it fits, otherwise tries essential, reference-summary, and
reference-only projections before leaving the unit out. Every projection
preserves active, candidate, contested and rejected/superseded conclusions,
constitutional governance metadata, constraints, commitments, conflicts, ordering, and
source references. A compressed resident remains in `page_index` so omitted
detail can be loaded; only a complete resident with no deeper continuation
omits its descriptor. The response `request` contains a digest and compact
routing/budget metadata rather than another copy of the full need, purpose,
scope, and task context.

AMOS derives candidate and graph-traversal work from the same byte budget so a
high-degree or very long connected graph cannot make compilation unbounded.
When that internal allowance is reached, inspect `truncated`, `unknowns`, and
`compilation_trace.relationship_truncation_reasons`. Page descriptors retain
visible boundary references beyond the resident unit; loading the descriptor at
`supporting` depth retrieves those continuation atoms. This is explicit partial
knowledge, not silent clipping or a fixed atom-count API.

Keep the frame and its descriptors in trusted runtime state. If the reasoner
needs deeper detail, resolve its requested page ID locally and send the retained
descriptor:

```http
POST /v1/reasoning-pages:load
X-Request-ID: reasoning-cycle-42-page-1
```

```json
{
  "frame_id": "frame_...",
  "revision": {"graph_version": 31, "journal_head": "..."},
  "page": {"descriptor_version": "amos.reasoning.page.v1", "...": "..."},
  "need": "the decision rationale",
  "purpose": "verify the active conclusion",
  "depth": "supporting",
  "scope": {"project": "memory-demo"},
  "requester": "agent:example",
  "target_processor": "reasoner",
  "token_or_byte_budget": {"tokens": 1200}
}
```

Never let model output supply or alter `scope`, requester identity, revision, or
the descriptor. The semantic identifier fields in `task_context` are likewise
runtime-owned: do not copy them from model-authored arguments or prose. AMOS
excludes atoms whose envelope scope or payload explicitly names a different
human, project, or thread, while retaining untagged/global memories. It binds
the normalized semantic scope into the descriptor digest and applies semantic,
scope, and access checks again while loading. An exact revision mismatch is a JSON `409` with
`code: "stale_revision"`, both revisions, and `retryable: false`; discard the
active frame and compile a new one. Do not silently append a page from a changed
graph.

Packet retrieval and demand-paged reasoning are independent caller-selected
interfaces. Maintaining a cycle-wide working-memory budget and replacing
summaries with loaded detail are client-runtime responsibilities. AMOS exposes
both transport paths but does not switch application modes.

## 5. Use Attention Deliberately

Good attention contexts are compact and operational:

- `active_task`: what the agent is doing now.
- `focus_terms`: concepts that should be foregrounded.
- `suppress_terms`: distractions to down-rank.
- `boost_memory_types`: atom types useful for this role.
- `counterevidence_required`: reserve space for warnings and conflicting facts.
- `novelty_preference`: prefer useful underused memory when exploration matters.

Use low novelty for conservative execution, moderate novelty for review, and
higher novelty for exploration or route selection. Do not use novelty to bypass
known constraints.

## 6. Put Packets Into Prompts Transparently

Render only the packet fields the model needs:

```text
Relevant Amos memory:
- atom_id, type, claim/action/outcome summary
- confidence, salience, utility when useful
- evidence refs or provenance note
- conflict or omission warnings
- compact attention trace: selected refs, inhibited refs, omitted reasons
```

Avoid dumping raw JSON into prompts unless the role needs exact fields. Keep the
full packet in telemetry so operators can audit why a prompt saw a memory.

## 7. Report Retrieval Outcomes

After the agent uses a packet, report whether it helped:

```http
POST /v1/retrieval-outcomes
```

```json
{
  "packet_id": "pkt_...",
  "request": {
    "requester": "agent:pilot",
    "scope": {"project": "qandl", "mission": "performance_search"}
  },
  "outcome": {
    "label": "useful",
    "used_item_refs": ["atom_..."],
    "correction_refs": [],
    "notes": "cited policy memory in next directive"
  }
}
```

Outcome feedback updates atom access, utility, salience, and health signals only
for atom refs present in the exact packet identified by `packet_id`. Evidence
refs are reported separately and arbitrary caller-supplied refs cannot train
atom utility. When a selected item was reached through associative activation,
the packet's bounded `association_trace` also attributes use or correction to
those edges for later degree-normalized ranking.

Distinguish memory exposure from memory use. If an atom was retrieved into a
prompt but did not materially shape the decision, record that as neutral
context, not as helpful evidence. A practical convention is:

```json
{
  "label": "observed",
  "use_status": "context_only",
  "cited_atom_ref": "atom_..."
}
```

Use positive labels such as `useful` only for atoms that changed the decision,
field selection, explanation, or safety check. Use correction or failed labels
when the retrieved memory contributed to a bad answer, was stale, was
mis-scoped, or was contradicted by later evidence.

## 8. Let Amos Maintain Memory

In HTTP service mode, packet retrieval queues background memory policy work and
returns a packet without running policy inline. Direct in-process callers can
still opt into foreground policy through `retrieve_packet(run_policy=True)`, but
latency-sensitive read views such as agentic recall skip foreground policy work
and rely on the background worker or explicit operator runs. Operators can run
policy explicitly:

```text
GET  /v1/memory-policy
POST /v1/memory-policy:configure
POST /v1/memory-policy:run
```

The built-in policy covers deterministic distillation, SMP analysis, low-risk
maintenance proposals, search-index refresh, dependency-free lexical/LSA vector
index refresh, decay checks, superseded-memory archiving, producer-explicit
proposal retention/deduplication, separate active/proposed quota enforcement,
cache invalidation, and capacity governance. It does not require an LLM or an
external vector database.

Clients that retain every cognitive occurrence in their own audit store may
bound the AMOS review queue by attaching
`payload.proposal_retention.deduplication_key` and
`payload.proposal_retention.archive_after_seconds` to proposed atoms. AMOS acts
only on those explicit semantics (or an explicitly configured global proposal
TTL); it does not guess duplicates from generated prose. Memory health reports
hot-total, lifecycle-active, and proposed counts and limits separately.

For request-time retrieval, an empty scope only sees global/unscoped memory. For
service-owned decay and storage cleanup, an empty maintenance scope means
whole-store maintenance; provide an explicit scope only when an operator wants to
limit cleanup to one tenant, project, run, or agent slice.

Client-specific cleanup and learning belongs in client processor packs, not in
AMOS core. A domain processor receives a bounded evidence window and returns
side-effect-free maintenance proposals. AMOS applies policy gates, commits
low-risk derived atoms and explicit active-endpoint edges, journals the mutation,
and defers ambiguous or high-risk work for review. A processor can implement
`window_request(scope=..., domain=...)` to request a narrower lifecycle, atom
type, producer profile, graph-neighbor, evidence, event, outcome, and size
workset. This request can never widen the caller-authorized scope or resource
ceiling.

Use producer-owned `payload.maintenance_hints` for stable profile, cohort,
priority, source-ref, and distillation-lane metadata. Set
`distillation_lane: domain_processor` when a registered processor owns the
semantic interpretation of that record, and provide `consolidation_key` (or a
stable cluster/cohort key) for records that may be distilled together. The
generic automatic policy excludes domain-processor records, excludes sources
already covered by any active derived semantic memory, and selects one coherent
scope/type/profile/cohort group per packet. It never combines globally adjacent
high-priority atoms from different groups. AMOS never treats these hints as
canonical claims or mutation authority. Prefer the shared cohort,
source-coverage, evidence-diversity, and derived-proposal helpers over
client-specific scans of the whole store. A derived proposal may name active
semantic atoms it supersedes; AMOS projects the governed supersession relation
and the ordinary decay policy retires obsolete active versions.

Every edge has a `derivation` object. New structural, facet-derived, and
processor-reviewed relations name their construction path; migrated legacy
edges are explicitly marked as classified migration data with unknown exact
producer. Health diagnostics expose component structure, type/relation mix,
hub concentration, edge confidence/provenance, unresolved refs, proposal
backlog quality, and recent processor effectiveness.

## 9. Model Durable Agent Identity Separately From Processors

For multi-agent systems, store each durable agent's self-model, capabilities,
limitations, procedures, commitments, and runtime state as Amos atoms. Do not
create a durable self-model for every transient processor role unless that
processor is intentionally modeled as a distinct, persistent agent:

```text
self_model
capability
limitation
procedure
commitment
runtime_state
```

Retrieve those atoms through agent-scoped packets with processor-specific views
instead of hard-coding large static prompt blocks. Static context can remain a
fallback for startup or Amos outage handling.

Keep the identity fields explicit at the integration boundary:

| Field | Meaning |
| --- | --- |
| `agent_id` | Durable subject whose self-model, commitments, autobiography, and continuity AMOS preserves. |
| `processor_id` / `target_processor` | Functional reasoner, planner, executor, critic, or other processing role for a request. |
| model profile | Replaceable provider, model, checkpoint, weights, quantization, prompt, and runtime metadata. |
| `client_identity` | Authenticated service or process actor, including its permissions and trust level. |

An LLM may receive bounded prompt context and use an ephemeral cache, but it
must be treated as stateless with respect to durable identity and memory. The
active agent is the first-person subject; the LLM is a replaceable cognitive
processor delegated to render that agent's response. Do not infer the agent's
role, purpose, personality, biography, capabilities, or limitations from the
model name, provider persona, training claims, or the model speaking about
itself.

Prior generated output is neither authoritative self-knowledge nor independent
evidence. If generated output suggests a memory or self-model change, record it
as a provenance-bearing, evidence-linked proposal and apply the normal schema,
authorization, contradiction, review, and lifecycle gates. A model replacement
must leave `agent_id` and established lineage intact; newly observed substrate
capabilities or limitations belong in model/runtime metadata until evidence
supports an agent-level learning.

Do not merge learned experience directly into a static role contract. Keep three
surfaces separate:

- Durable self-model: stable agent role or purpose, delegated authority, and
  standing commitments.
- Runtime state: current tool availability, denied capabilities, budgets,
  active task, and recent errors.
- Experience profile: recurring demonstrated capabilities, recurring
  limitations, and reuse guidance distilled from action outcomes.

The experience profile should be compact. Prefer a few promoted capability and
limitation atoms with source counts, recent source refs, control or task family,
and reuse guidance over many raw action logs. The agent prompt should see the
lesson; telemetry should retain the raw packet and evidence for audit.

## 10. Keep Prompt Context Operational

AMOS packets are context inputs, not a license to fill a prompt with every
available memory. A good integration renders:

- The current agent identity and authority, with the current processor role
  shown separately.
- Current runtime constraints and denied capabilities.
- The active task or mission policy.
- A small set of materially relevant memories, including counterevidence.
- Learned experience-profile capabilities and limitations for the agent.
- Citation candidates and a rule for when to cite or explain non-use.

The model should be instructed to cite AMOS atom refs only when a memory
materially shapes the decision. Otherwise it should record why retrieved memory
was not used. This avoids false reinforcement and gives maintenance useful
signal.

## 11. Integration Lessons

- Run one logical Amos instance per coordinated agent system.
- Give each durable agent a stable `agent_id`; give transient reasoner, planner,
  executor, and critic roles stable processor identifiers instead of separate
  selves by default.
- Use scopes for tenant, project, run, mission, and agent visibility.
- Keep static contracts as bootstrap or fallback context; prefer AMOS packets
  once current self-awareness and mission policy atoms are available.
- Store raw experiences as evidence-backed traces or outcomes, then promote
  recurring patterns through maintenance processors.
- When a producer already has typed semantics, write canonical
  `payload.semantic_facets` and `payload.graph_relations`. The built-in generic
  processor can then construct provenance and associative edges without a
  client-specific processor pack. Keep free-form model interpretations in the
  proposed lifecycle until separately reviewed.
- Keep generated prompt digests compact and role-specific.
- Treat memory retrieval as advisory. Application schemas, permissions,
  guardrails, and control registries remain hard authority.
- Persist rendered prompt packets and retrieval outcomes for later audit.

## 12. Production Checklist

- Run one shared Amos service per coordinated agent system.
- Keep direct database access out of agents.
- Use stable scopes and idempotency keys.
- Capture evidence and cite it from derived atoms.
- Retrieve per role, task, and mission with explicit attention context.
- Enforce application authority outside attention ranking.
- Record retrieval outcomes.
- Promote recurring experience into compact learned profile atoms; do not use
  AMOS as an append-only logging sink.
- Monitor memory health, capacity health, worker status, and journal verify.
- Keep packet payloads in telemetry for audit and debugging.
- Keep model and provider metadata out of the durable self-model, and route all
  model-derived memory changes through evidence-linked proposal review.

For small deployments, the HTTP service plus SQLite is the intended v1 starting
point. The stdlib HTTP adapter serializes service calls through one in-process
lock for correctness with a single SQLite store. WAL-backed read parallelism,
reader/writer lock splitting, Postgres, and external vector integration are
roadmap items for larger multi-agent or higher-scale deployments.
