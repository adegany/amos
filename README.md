# Amos

**Amos** stands for **Agent Memory Operating System**.

Amos is a model-neutral, layered, associative, self-maintaining memory substrate for agentic AI systems. It treats agent memory as an operating-system-like service: capture evidence, maintain typed memory, preserve provenance, perform cleanup, promote and demote memories across tiers, and render task-specific memory packets for reasoners, planners, executors, critics, and future processors.

The core thesis is that long-term agent memory should not be stored primarily as English summaries. English, embeddings, prompt snippets, and planner-specific payloads should be generated views over a canonical memory substrate composed of typed atoms, evidence links, associative edges, health states, and maintenance actions.

## Current status

This repository now includes a dependency-free AMOS v1-local implementation
alongside the design spec.

The first usable deployment profile is an AMOS HTTP service that owns one
in-process SQLite store and serializes access through the service boundary:

- Service-owned SQLite canonical store with an append-only event journal and
  checksum chain.
- Typed memory atoms, evidence records, associative edges, tombstones, packet cache,
  and retrieval outcomes.
- Schema validation for envelope/payload separation, typed payload contracts,
  and JSON-compatible atoms.
- Idempotent capture/commit operations and compare-and-swap update checks.
- Memory packets with scope isolation, access filtering, omissions, conflicts,
  provenance, and degradation metadata.
- Self-awareness and agentic-recall views for self models, capabilities,
  limitations, runtime state, self-assessments, traces, outcomes, corrections,
  and blocked actions.
- Provenance-linked deterministic memory distillation.
- Automatic memory policy scheduling for deterministic distillation, SMP,
  stewardship, processor-pack distillation, derived-index refresh, and
  packet-cache invalidation.
- Deterministic non-generative Semantic Maintenance Processor (SMP) outputs
  using the required audit envelope.
- Generic maintenance proposal records, a processor registry, and a policy gate
  that auto-commits only low-risk derived atoms while deferring review items.
- A generic maintenance processor registry. AMOS ships the built-in generic SMP
  adapter; domain-specific processors are loaded from client packages with
  explicit import paths.
- Advisory maintenance for deduplication and contradiction marking, with
  high-risk mutation requests gated behind explicit approval.
- Capacity pressure reporting and degraded packet disclosure.
- Journal chain and replay verification.
- Worker artifacts for projection checks, index maintenance, packet-cache
  invalidation, capacity governance, stewardship, self-model calibration,
  agentic-recall auditing, and SMP analysis.
- Dependency-free HTTP adapter for the V1 JSON API surface; connected agents
  call the service instead of embedding their own stores.
- CLI and tests.

Start here:

- [Amos Design Spec](docs/design-spec.md)
- [AMOS V1 Verification Matrix](docs/v1-verification.md)
- [Amos Mirror Agent Demo Spec](docs/mirror-agent-demo-spec.md)

## Quick start

Run the tests:

```bash
python -m pytest -q
```

Initialize a local store:

```bash
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 init
```

Commit and retrieve a memory atom:

```bash
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 commit-atom \
  --type belief \
  --payload '{"claim":"Codex outages should fall back to local advisors"}'

PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 retrieve \
  --cue "Codex outage fallback"
```

Run the Amos Mirror Agent integration demo:

```bash
PYTHONPATH=src python examples/mirror_agent_demo.py --format text
```

Run the browser UI for conversational self-awareness and introspection:

```bash
PYTHONPATH=src python examples/mirror_agent_ui.py --host 127.0.0.1 --port 8787 --lm codex
```

The UI chat path uses local `codex exec` as the LM provider by default. AMOS
memory policy maintenance remains deterministic and non-LLM: SMP analysis,
stewardship, automatic distillation, index rebuilds, packet-cache invalidation,
and capacity reporting do not call the chat LM.

Serve the V1 HTTP API:

```bash
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 serve --host 127.0.0.1 --port 8765
```

Verify journal replay:

```bash
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 verify
```

Inspect or tune the automatic memory policy:

```bash
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 memory-policy
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 memory-policy --configure --schedule '{"every_graph_versions": 10, "every_seconds": 300}'
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 memory-policy --run --force --trigger operator_check
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 maintenance-processors
PYTHONPATH=src python -m amos.cli --db /tmp/amos.sqlite3 maintenance-distiller --domain generic --processor-id amos.maintenance.generic.v1
```

Load an external maintenance processor pack from another package:

```bash
PYTHONPATH=src python -m amos.cli \
  --db /tmp/amos.sqlite3 \
  --maintenance-processor my_package.processors:training_flight_processor \
  maintenance-distiller \
  --domain training_flight \
  --processor-id my.training.flight.v1
```

## Integration boundary

AMOS owns canonical memory, recall, provenance, cleanup metadata, self-awareness
views, and advisory maintenance. It does not directly execute external actions.
Integrations such as the Mirror Agent demo should keep live control authority,
validation, approval checks, and runtime packet application outside AMOS.
Domain-specific maintenance packs should follow the same boundary: they inspect
bounded AMOS evidence windows and return side-effect-free proposals; the AMOS
service applies policy gates, journals accepted low-risk mutations, and defers
ambiguous or high-risk work for review.

## Design goals

- Reduce long-term storage and token cost.
- Avoid repeated expensive full-memory redistillation.
- Preserve provenance and auditability.
- Support reasoners, planners, executors, critics, and future non-LLM processors.
- Model memory as dynamic: layered, associative, promotable, demotable, and self-maintaining.
- Treat memory maintenance as a first-class internal system responsibility.

## Non-goals for this phase

- No vendor-specific vector database commitment.
- No prompt-only memory architecture.
- No autonomous external-state procedure execution.
- No irreversible autonomous deletion policy without audit controls.
- No bundled production Postgres service yet; Postgres DDL is included as the
  target migration contract, while v1-local uses SQLite behind the HTTP API.
