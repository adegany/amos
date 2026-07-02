# Amos

**Amos** stands for **Agent Memory Operating System**.

Amos is a design project for a model-neutral, layered, associative, self-maintaining memory substrate for agentic AI systems. It treats agent memory as an operating-system-like service: capture evidence, maintain typed memory, preserve provenance, perform cleanup, promote and demote memories across tiers, and render task-specific memory packets for reasoners, planners, executors, critics, and future processors.

The core thesis is that long-term agent memory should not be stored primarily as English summaries. English, embeddings, prompt snippets, and planner-specific payloads should be generated views over a canonical memory substrate composed of typed atoms, evidence links, associative edges, health states, and maintenance actions.

## Current status

This repository currently contains a design spec only. Implementation is intentionally deferred.

Start here:

- [Amos Design Spec](docs/design-spec.md)

## Design goals

- Reduce long-term storage and token cost.
- Avoid repeated expensive full-memory redistillation.
- Preserve provenance and auditability.
- Support reasoners, planners, executors, critics, and future non-LLM processors.
- Model memory as dynamic: layered, associative, promotable, demotable, and self-maintaining.
- Treat memory maintenance as a first-class internal system responsibility.

## Non-goals for this phase

- No implementation code yet.
- No vendor-specific vector database commitment.
- No prompt-only memory architecture.
- No irreversible autonomous deletion policy without audit controls.
