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
  "actor": "agent:pilot",
  "outcome": "used",
  "used_atom_ids": ["atom_..."],
  "notes": "cited policy memory in next directive"
}
```

Outcome feedback updates atom access, utility, salience, and health signals. It
also gives the maintenance worker better evidence for cleanup and ranking.

## 8. Let Amos Maintain Memory

In HTTP service mode, packet retrieval queues background memory policy work.
Operators can also run policy explicitly:

```text
GET  /v1/memory-policy
POST /v1/memory-policy:configure
POST /v1/memory-policy:run
```

The built-in policy covers deterministic distillation, SMP analysis, low-risk
maintenance proposals, search-index refresh, decay checks, cache invalidation,
and capacity governance. It does not require an LLM.

## 9. Model Agent Identity As Memory

For multi-agent systems, store each role's self-model, capabilities,
limitations, procedures, commitments, and runtime state as Amos atoms:

```text
self_model
capability
limitation
procedure
commitment
runtime_state
```

Retrieve those atoms through role-specific packets instead of hard-coding large
static prompt blocks. Static context can remain a fallback for startup or Amos
outage handling.

## 10. Production Checklist

- Run one shared Amos service per coordinated agent system.
- Keep direct database access out of agents.
- Use stable scopes and idempotency keys.
- Capture evidence and cite it from derived atoms.
- Retrieve per role, task, and mission with explicit attention context.
- Enforce application authority outside attention ranking.
- Record retrieval outcomes.
- Monitor memory health, capacity health, worker status, and journal verify.
- Keep packet payloads in telemetry for audit and debugging.

For small deployments, the HTTP service plus SQLite is the intended v1 starting
point. Postgres and external vector integration are roadmap items for larger
multi-writer or higher-scale deployments.
