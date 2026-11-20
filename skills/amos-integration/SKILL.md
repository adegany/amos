---
name: amos-integration
description: Assess whether the Agent Memory Operating System (AMOS) fits a project, or integrate AMOS when an application needs durable agent recall, conversation continuity, shared memory, or evidence-backed reasoning.
---

# Integrate AMOS into a project

Start in the application that would use memory. Find one question a later request should be able to answer from an earlier event. Trace the code that observes the event, the code that could use it later, and what the application does today. Identify who may see the information and how long the application needs it. If the project is still an idea, work from a representative user story and state your assumptions.

For a fit assessment, keep the project read-only. Name the producer and consumer, the data AMOS would own, the expected benefit, and one test that would demonstrate it. If the existing store already answers the question, show how to verify that behavior and explain what additional requirement would justify AMOS. For an implementation request, connect the chosen workflow through the project's normal entry point and test it.

## Choose the memory contract

| What the application needs | Where to start |
| --- | --- |
| Look up a record already in AMOS by ID | Keep the existing producer. Use `POST /v1/atoms:get` or `POST /v1/evidence:get`. For presentation-only reads, `POST /v1/canonical-records:batch-get` and `POST /v1/evidence:get` support `X-AMOS-Read-Purpose: observation` to avoid recording retrieval feedback. |
| Recall a fact, preference, or observed outcome on a later task | Capture the source with `POST /v1/events:capture`. Use `POST /v1/atoms:commit` for a typed observation citing the returned evidence ID. Use `POST /v1/packets:retrieve` for recall by cue. |
| Resume a conversation with exact turn order | Read the previous head with `POST /v1/memory-heads:get`. Append each immutable `interaction_event` and its matching `interaction_stream` head update in one `POST /v1/memory-transactions:commit`, binding the next sequence and reply reference to that head. Use `POST /v1/interaction-projections:compile` for a transcript or `POST /v1/cognitive-workspaces:compile` for bounded context anchored on a turn. Version interpreted thread state separately when needed. |
| Explain how a decision arose, including conflicting or superseded evidence | Compile a frame with `POST /v1/reasoning-frames:compile`. Load detail with `POST /v1/reasoning-pages:load` using the returned descriptor, frame ID, revision, and original access context. Recompile on HTTP 409 `stale_revision`. |
| Coordinate multiple agents or recall an agent's own experience | Share one logical service. Give each durable agent a stable `agent_id` across model changes. Set `target_processor` to the current role. Use `POST /v1/shared-views:retrieve` for shared coordination, `POST /v1/self-awareness:retrieve` for a self-model, or `POST /v1/agentic-recall:retrieve` for past actions and outcomes. |
| Learn from repeated observations or adopt a model-suggested belief | Producers that know the relationships can supply `payload.semantic_facets` or `payload.graph_relations` for built-in maintenance. Send inferred claims with evidence to `POST /v1/atoms:propose`. Adoption through `POST /v1/proposals:ratify` requires an adjudication from the same authenticated identity. [Governance contract](https://github.com/adegany/amos/blob/main/docs/developer-guide.md#3a-adopt-proposals-through-self-ratification) |

Use the AMOS revision deployed by the project for exact request and response shapes. These upstream links also name the paths in a local checkout. [Developer guide](https://github.com/adegany/amos/blob/main/docs/developer-guide.md) · [JSON schemas](https://github.com/adegany/amos/tree/main/schemas) · [HTTP examples and tests](https://github.com/adegany/amos/blob/main/tests/test_cli_http.py)

## Connect the application

- **Service ownership.** Reuse the project's AMOS service if it has one. The v1-local deployment is one HTTP process owning one SQLite database. Clients in any language can call it through a small adapter. An isolated Python experiment can also use `from amos import Amos` with its own database. Treat PostgreSQL and multiple writer processes as roadmap items when assessing fit. [Deployment contract](https://github.com/adegany/amos/blob/main/docs/v1-local-contract.md#13-deployment-topology)
- **Visibility.** Derive scope, actor, requester, and processor from trusted application state. Set scope and access policy on both evidence and atoms. An empty record scope is globally eligible and visibility defaults to `all`. Keep the service behind the application's authentication boundary. Protected governance calls use a server-configured bearer principal. Keep external actions, permissions, job queues, and delivery receipts in the host application.
- **Writes.** Capture what the source actually observed. On capture, atom commits, and memory transactions, derive an idempotency key from the source event and operation. If the response is lost, resend the same actor, key, and payload on those endpoints. For writes such as `atoms:propose` that lack this guarantee, reconcile the original result before resending. On a compare-and-swap conflict, reread the head and reconcile before constructing a new write. Acknowledge persistence only after AMOS confirms it.
- **Reads.** For associative packets, set task-specific cues, scope, processor role, `max_items`, and `token_or_byte_budget`. Use `operational_recall` for active memory, `deliberation` to consider candidates and counterevidence, and `historical_review` for past states. Show relevant provenance, conflicts, and limits on what was recalled. If a model selects a memory, give it request-local handles and resolve them against the retained AMOS response. Send `POST /v1/retrieval-outcomes` feedback for material use or correction. Mere exposure is neutral.
- **Existing data and failures.** Import history only when the chosen workflow needs it, preserving source IDs and visibility. If a business transaction must produce memory, use its durable event or outbox to deliver a replay-safe write. Decide what the host can still do if recall is unavailable and which operations require a confirmed write. For retryable HTTP 503s, use backoff with jitter within the caller's deadline and honor `retry_after_ms` when supplied.

## Try one evidence-backed round trip

For a disposable trial, use Python 3.10+ from the root of an AMOS checkout. Pick an unused loopback port. This example uses 8765. [AMOS repository](https://github.com/adegany/amos)

```sh
amos_trial_dir=$(mktemp -d)
PYTHONPATH=src python3 -m amos.cli --db "$amos_trial_dir/amos.sqlite3" serve --host 127.0.0.1 --port 8765
```

From a second terminal, `curl -fsS http://127.0.0.1:8765/v1/ready` should return `status` equal to `ready`. Run the following Python against that trial service. An importer records which delimiter worked. On its next run it knows only the feed name.

```python
import json
from urllib.request import Request, urlopen

def post(operation, body):
    request = Request(
        f"http://127.0.0.1:8765/v1/{operation}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)

scope = {"tenant": "trial", "project": "imports"}
actor = "service:importer"
access = {"visibility": [actor]}
observation = {"feed": "Harbor ledger", "delimiter": "tab"}
source = post("events:capture", {
    "actor": actor, "idempotency_key": "trial:imports:42:evidence",
    "scope": scope, "access_policy": access,
    "source_type": "import_result", "source_ref": "import:42",
    "payload": {**observation, "rows_accepted": 12},
})
evidence_id = source["evidence"]["evidence_id"]
receipt = post("atoms:commit", {
    "actor": actor, "idempotency_key": "trial:imports:42:outcome",
    "atom": {
        "type": "action_outcome", "scope": scope, "access_policy": access,
        "evidence_refs": [evidence_id],
        "payload": {"agent_id": "importer", "action_ref": "import:42",
                    "status": "success", **observation},
    },
})
packet = post("packets:retrieve", {
    "requester": actor, "target_processor": "planner", "scope": scope,
    "cues": ["Harbor ledger"], "memory_mode": "operational_recall",
    "max_items": 3, "token_or_byte_budget": {"tokens": 1200},
})
item = next(item for item in packet["items"] if item["atom_ref"] == receipt["atom"]["id"])
assert item["payload"]["delimiter"] == "tab" and evidence_id in item["evidence_refs"]
print(item["payload"]["delimiter"], evidence_id)
```

## Verify an implementation

For a writing integration, repeat the same source event through the host entry point and check that it yields a single logical memory. For a read-only integration, start with a known record from the existing producer. In a fresh request or session, recall the intended value and trace it back to the source. With scoped or private data, verify that another tenant or unauthorized role receives no protected content. For an interaction stream, also exercise competing appends. For paged reasoning, exercise a stale descriptor. `GET /v1/verify` checks journal integrity and replay on the trial database.

Report the application files and hooks used, the behavior observed on that later request, and any remaining limits that affect the user's decision.
