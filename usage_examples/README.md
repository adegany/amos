# Try one memory behavior

Run either command from the AMOS repository root with Python 3.10+. Each takes a few seconds, uses a fresh temporary SQLite file and a free loopback port, and cleans up on exit. The two sessions communicate only through persisted AMOS data. You can rerun either example from the beginning.

## Remember a user's preference

```sh
PYTHONPATH=src python3 -m usage_examples.preference_continuity
```

1. Ask the reminder agent how it would display a reminder for 2:30 in the afternoon. Its default reply is `Reminder at 02:30 PM.`
2. The user says “Use 24-hour time.” The host recognizes that explicit command and commits the exact words as evidence alongside a typed `preference`.
3. After a service restart, the same reminder request produces `Reminder at 14:30.` The example prints the atom ID, resolves its evidence ID, and checks the original words.

In your agent, call `set_preference` when the user states a supported rule. Call `reminder` when rendering the next reminder. The read needs the user's identity and the purpose of the preference. It receives the time format from AMOS. [Example code](preference_continuity.py)

## Reuse a successful action outcome

```sh
PYTHONPATH=src python3 -m usage_examples.successful_action_recall
```

1. The Harbor feed has a tab-separated header. The worker tries its comma default and accepts zero rows. A tab retry succeeds. A second feed, Delta, succeeds with a comma.
2. The worker commits an `action_outcome` for each attempt, with the delimiter, accepted row count, and source run.
3. After a restart, the worker asks for Harbor outcomes with status `success`. Exactly one result qualifies. The worker applies its delimiter to a new input, imports SKU `H2`, and prints the atom and evidence for the choice.

Follow `run_import` in your worker. At job setup, query AMOS with the feed name and status `success`. At job completion, save the outcome in `record_attempt`. Both the first and later job go through this same function, so you can see exactly where recall changes what the CSV parser does. [Example code](successful_action_recall.py)

## Copy the write and read

Copy the relevant host functions and `memory_http.py` into your project. Construct `Memory` with your AMOS service URL and the actor and scope supplied by your trusted runtime. The example's source reference is a globally unique message or run ID. Reusing it for an identical event makes delivery idempotent. [HTTP client](memory_http.py)

| Operation | What the examples send or receive |
| --- | --- |
| `POST /v1/memory-transactions:commit` | Store one source event and one typed atom together. The atom cites that event's evidence ID. |
| `POST /v1/packets:retrieve` | Get matching visible items by atom type and payload fields. Apply the returned payload and keep its `atom_ref` and `evidence_refs` for citation. |
| `POST /v1/evidence:get` | When someone asks why, resolve a cited evidence ID to the original message or import observation. |

## Explore Mirror Agent

After either example, run the full reference agent to see how role-specific recall, self-models, reasoning frames, and governed maintenance interact. [Mirror Agent walkthrough](mirror_agent/README.md)
