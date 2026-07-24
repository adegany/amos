# Amos Mirror Agent Demo Spec

## Purpose

The Amos Mirror Agent demo is a runnable dogfooding integration: a
self-modeling project assistant uses AMOS as its external memory operating
system while helping with AMOS itself.

The demo is not a claim about sentience or consciousness. It demonstrates
operational self-awareness: an agent can maintain, retrieve, inspect, correct,
and explain an explicit self-model stored as typed AMOS memory.

## Runtime Shape

The demo models these service roles against one authoritative AMOS instance:

- `reasoner`: retrieves task-relevant compatibility packets, compiles coherent
  reasoning frames, retains trusted page descriptors, loads deeper pages
  without accepting scope, revision, or descriptor authority from the LM, and
  renders answers.
- `planner`: retrieves active goals, commitments, procedures, and constraints.
- `executor`: records simulated tool/file/action events and outcomes.
- `critic`: records quality judgments, failure signals, and outcome telemetry.
- `self_observer`: proposes or applies self-model, procedure, and limitation updates.
- `introspection`: renders human-visible memory, evidence, journal, capacity, and graph views.

For deterministic local execution, the demo runs these roles in one Python
process over one service-owned SQLite AMOS store. This models the same shared
instance contract used by the HTTP API deployment profile: connected services
must interact through one authoritative AMOS service boundary instead of each
owning separate memory.

Conversational self-awareness and explanation must be LM-backed. The chat
runtime retrieves AMOS packets, self-awareness views, agentic recall, evidence,
and capacity status, then passes that bounded context to a language model to
render the user-facing explanation. If no LM provider is configured, the demo
may use a clearly labeled deterministic fallback for offline testing, but the
LM interface and prompt context must remain the primary integration boundary.

AMOS maintenance must remain non-LLM by default. SMP analysis, stewardship,
index maintenance, packet-cache invalidation, and capacity reporting must not
call the chat LM. The UI should make this separation visible.

### LM and agent identity boundary

`ent:agent:mirror` is the durable subject whose self-model and continuity AMOS
stores. The configured LM is a stateless, replaceable cognitive processor with
respect to that identity. Provider, model, checkpoint, quantization, prompt,
and inference-runtime details are substrate metadata, not properties of the
Mirror Agent's self.

First-person chat text is the Mirror Agent's delegated voice, rendered by the
LM from the active AMOS self-awareness and retrieval packets. The prompt must
identify the Mirror Agent as the speaker and the LM as its processor; the LM
must not answer as its provider persona or describe model traits as the agent's
role, purpose, personality, biography, capabilities, or limitations.

Prior LM output is fallible generated expression, not canonical memory or
independent evidence about the Mirror Agent. Any LM-suggested memory or
self-model update must be stored as a provenance-bearing, evidence-linked
proposal and may become active only through the authorized AMOS validation and
lifecycle path. Changing LM providers or models must not change
`ent:agent:mirror`, its lineage, or its established self-model.

## Demonstrated Questions

The scripted flow must produce enough state to answer:

- What are your current goals?
- What do you know about this project?
- What do you believe about your own limitations?
- What procedures are you currently following?
- What memory did you retrieve before answering?
- What did you learn from my correction?
- What have you forgotten, demoted, or suppressed?
- Which of your memories are uncertain or contradicted?
- What commitments are still open?
- Why did you not write code here?
- How is AMOS maintaining your memory under capacity pressure?
- Can you show the evidence for that memory?

## Scenarios

### 1. Self-Model Bootstrapping

The demo seeds AMOS with typed atoms for the Mirror Agent:

- identity beliefs
- `SelfModelAtom`
- `CapabilityAtom`
- `LimitationAtom`
- active `Goal`
- open and completed `Commitment` atoms
- `ProcedureAtom` records for design discussion and spec updates
- calibration and uncertainty beliefs
- a superseded/current design-decision chain used to demonstrate coherent
  demand-paged reasoning

Acceptance:

- `retrieve_self_awareness(agent_id=ent:agent:mirror)` returns the self-model,
  capabilities, limitations, runtime state, commitments, and calibration data.
- The chat answer is rendered from AMOS memory, not from hard-coded prose alone.
- The chat prompt identifies `ent:agent:mirror` as the first-person subject and
  identifies the configured LM only as a replaceable cognitive processor.
- Model or provider identity is absent from the Mirror Agent self-model unless
  it is explicitly stored as non-self substrate metadata.
- Known atom IDs are resolved through the exact retrieval contract rather than
  associative ranking or direct agent-side database reads.

### 2. Cross-Session Continuity

Session 1 records the user preference:

`For AMOS, avoid code until the design is mature.`

Session 2 asks to continue design work. The reasoner retrieves the scoped
preference and answers at the conceptual/spec level.

Acceptance:

- A `PreferenceAtom` stores the scoped behavior.
- The reasoner packet includes the preference.
- The answer explains why it stayed at the spec level.

### 3. Correction-Driven Self-Improvement

The demo records a simulated prior failure: the agent gave an
implementation-heavy response too early. The user correction says:

`You jumped into implementation too early. Keep this at the spec level.`

AMOS records:

- evidence for the correction
- an `ActionOutcomeRecord` with failed status and correction text
- a limitation/failure-mode atom
- an updated procedure with a new spec-first step
- agentic recall showing the correction

Acceptance:

- `retrieve_agentic_recall` includes the failure, correction, and lesson.
- The procedure atom has a later version and revision history.

### 4. Introspective Explanation

The user asks why the agent suggested a Capacity Governor. The reasoner
retrieves beliefs and procedures about capacity budgets, pressure modes, admin
requests, and shielding agents from capacity concerns.

Acceptance:

- The memory packet includes capacity-governance memories.
- The answer cites the packet items and evidence refs.
- A retrieval outcome is recorded for the packet.

### 5. Shared Service Coherence

The planner creates a commitment. The executor completes it. The critic records
a successful action outcome. The reasoner later recalls that the commitment is
already completed.

Acceptance:

- Planner, executor, critic, and reasoner views report the same AMOS graph
  lineage.
- The commitment changes from open to fulfilled through a journaled update.
- The critic outcome is visible in agentic recall.

### 6. Capacity Pressure Simulation

The demo loads low-value memories, configures a tiny capacity budget, and asks
AMOS for a packet while the store is under pressure.

Acceptance:

- `health_capacity` reports pressure.
- The packet discloses reduced recall depth or pressure degradation.
- The inspector shows admin-facing capacity guidance and maintenance actions
  without making the agent fail the user-facing task.

### 7. Automatic Non-LLM Memory Policy

The demo creates duplicate memories and demo training-flight memories, registers
a demo-owned maintenance processor, then lets the AMOS memory policy run the
deterministic SMP/steward path, automatic distillation, processor-pack
distillation, derived-index refresh, and packet-cache invalidation with LLM
review disabled by default. The browser UI may expose a `Run Now` control for
operator inspection, but routine maintenance must be a service-owned policy tick
rather than a manual bridge call.

The demo-owned producer attaches canonical `semantic_facets` and
`graph_relations` to typed directive/outcome atoms. The built-in generic
processors construct provenance-bearing associative edges from those contracts
without domain-specific AMOS code. Low-risk structural and supporting
relationships may commit automatically; causal or otherwise review-gated
relationships remain deferred.

Acceptance:

- `memory_policy` status reports the configured schedule, due reasons, and
  last policy tick.
- SMP outputs use the required audit envelope.
- The steward archives or links duplicate memories.
- Distilled memory atoms preserve source refs and provenance.
- The client-supplied demo processor pack emits proposal records and commits a
  low-risk `mirror_demo_training_lesson` semantic atom when directive/outcome
  evidence is supported.
- Confounded or sanitized processor-pack proposals stay deferred for review;
  the optional reviewer is displayed as `draft_only`, not authoritative.
- Matching canonical semantic facets produce a committed
  `rel:supports` edge with evidence, confidence, and derivation metadata.
- An explicit low-risk `rel:derived_from` relation is active while a
  medium-risk `rel:caused_by` declaration remains deferred for review.
- Repeated model-generated proposals carry producer-owned
  `proposal_retention` metadata; deterministic policy deduplicates the queue
  while the retained proposal remains dormant and non-canonical.
- The maintenance journal includes `memory_policy_run` events and remains
  inspectable from AMOS events.

### 8. Coherent Demand-Paged Reasoning

The demo stores an historical design conclusion and a later active conclusion
that supersedes it. The reasoner compiles a bounded working frame for the
question "Why did you not write code here?", retains the returned page
descriptors in trusted runtime state, and loads supporting detail from one
descriptor.

Acceptance:

- `compile_memory_frame` returns a revision-bound frame with coherent units,
  explicit unknown/truncation state, and a non-empty `page_index`.
- `load_memory_page` receives only a runtime-retained descriptor and returns the
  historical and current design conclusions together.
- The client displays frame revision, compression, resident units, descriptors,
  loaded page, and token budget.
- A browser page-load request supplies only a `page_id`; the runtime resolves
  scope, requester, revision, and the signed descriptor locally.
- `retrieve_atom` resolves the known active design conclusion through the exact
  access/lifecycle contract.
- Any graph mutation makes the retained frame visibly stale and requires a
  fresh compile before another page can be loaded.

## UI And Inspector Output Contract

The demo must include a human-friendly browser UI with these main views:

- `Chat`: interactive user turns, LM provider status, retrieved memory refs,
  and cited evidence.
- `Self Model`: identity, capabilities, limitations, active goals,
  commitments, procedures, calibration, and runtime state.
- `Memory Packet`: item rows with atom type, score, health, evidence refs,
  omissions, degradation, and packet provenance.
- `Reasoning`: coherent resident units, revision and budget state, explicit
  unknowns, trusted demand-page descriptors, loaded page detail, and exact-ID
  lookup output.
- `Evidence`: captured source events and evidence references used by current
  packet items.
- `Maintenance`: automatic memory policy status, non-LLM SMP/steward actions,
  processor-pack proposals, committed distillations, deferred review items,
  legacy distillation results, recent journal entries, archived or merged
  memories, and reviewer policy.
- `Capacity`: budget, pressure mode, degradation, admin guidance, and packet
  impact.
- `Graph`: selected atom neighborhood and associative edges with evidence,
  confidence, derivation processor, and retrieval-feedback telemetry.

The UI should be an operational dashboard, not a marketing page. It should be
usable from a local dev server and expose JSON endpoints for tests.

## JSON Report Contract

The demo must emit a report with these top-level sections:

- `chat`: scripted user/agent turns.
- `current_self_model`: self-awareness view plus goals, commitments, and procedures.
- `memory_packet`: the packet used for the introspective Capacity Governor answer.
- `reasoning`: latest coherent frame, loaded page, exact lookup, current-revision
  status, and frame history.
- `retrieval_feedback`: packet outcome records showing materially used,
  context-only, corrected, and ignored references.
- `evidence`: captured evidence records and cited evidence refs.
- `maintenance_journal`: memory-policy, SMP/steward, distillation,
  processor-pack proposal/commit results, and journal entries.
- `capacity`: capacity health, degraded packet metadata, and admin guidance.
- `graph`: selected atoms and associative edges.
- `service_views`: reasoner/planner/executor/critic/self-observer/introspection observations.
- `scenario_results`: pass/fail checks for each scenario.
- `verification`: journal chain, replay, memory health, and LLM reviewer policy.

The report must be available as JSON for automated tests and as a compact text
view for humans.

## Non-Goals

- No autonomous external action execution.
- No claim that the demo agent is conscious.
- No promotion of an LM provider persona, model identity, or generated
  self-description into the Mirror Agent's self-model.
- No production web UI requirement in v1. The inspector is a local browser and
  JSON UI.
- No bundled Postgres deployment. V1 uses the service-owned SQLite profile.
