from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from amos import CASConflict, ValidationError
from amos.http_api import AmosHTTPServer

SCOPE = {"tenant": "continuity-test", "conversation": "main"}
PRIMARY_VISIBILITY = {
    "visibility": ["processor:primary-reasoner"],
    "mutable_by": ["owner"],
    "sensitivity": "private",
}


def interaction(
    atom_id: str,
    *,
    sequence: int,
    role: str,
    content: str,
    in_reply_to: str | None = None,
) -> dict:
    return {
        "id": atom_id,
        "type": "interaction_event",
        "payload": {
            "profile": "amos.interaction-event.v1",
            "conversation_id": "main",
            "sequence": sequence,
            "actor_ref": f"{role}:participant",
            "role": role,
            "content": content,
            "occurred_at": f"2026-07-29T00:00:{sequence:02d}Z",
            "in_reply_to": in_reply_to,
            "visibility": "shared",
            "source_ref": f"source:{atom_id}",
            "thread_refs": [],
        },
    }


def interaction_stream_head(
    new_head_ref: str,
    *,
    expected_head_ref: str | None,
    expected_head_version: int,
    conversation_id: str = "main",
) -> dict:
    return {
        "series_kind": "interaction_stream",
        "series_id": conversation_id,
        "new_head_ref": new_head_ref,
        "expected_head_ref": expected_head_ref,
        "expected_head_version": expected_head_version,
    }


def thread_root(atom_id: str, *, opening_event_ref: str) -> dict:
    return {
        "id": atom_id,
        "type": "discourse_thread",
        "payload": {
            "profile": "amos.discourse-thread.v1",
            "thread_id": atom_id,
            "conversation_id": "main",
            "opened_by_event_ref": opening_event_ref,
            "participants": ["human:participant", "agent:participant"],
        },
    }


def thread_state(
    atom_id: str,
    *,
    thread_id: str,
    revision: int,
    event_refs: list[str],
    secret: str = "orchid",
) -> dict:
    return {
        "id": atom_id,
        "type": "discourse_state",
        "payload": {
            "profile": "amos.discourse-thread-state.v2",
            "thread_id": thread_id,
            "revision": revision,
            "lifecycle": "open",
            "attention_state": "foreground",
            "summary": "A cooperative open-vocabulary activity",
            "participants": ["human:participant", "agent:participant"],
            "head_event_refs": [event_refs[-1]],
            "source_event_refs": event_refs,
            "shared_state": [
                {
                    "key": "protocol",
                    "value": "The participant supplies bounded questions",
                    "state_class": "reported_fact",
                    "authority": "discourse",
                    "basis_refs": [event_refs[0]],
                }
            ],
            "private_state": [
                {
                    "key": "concealed_value",
                    "value": secret,
                    "state_class": "private_secret",
                    "authority": "discourse",
                    "basis_refs": [event_refs[-1]],
                }
            ],
            "unresolved_items": [
                {
                    "kind": "expected_move",
                    "description": "Await the next participant turn",
                    "basis_refs": [event_refs[-1]],
                }
            ],
        },
        "access_policy": PRIMARY_VISIBILITY,
    }


def test_atomic_interaction_thread_head_and_workspace_visibility(amos):
    first = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="participant-ingress",
        idempotency_key="human-1",
        evidence=[
            {
                "source_type": "interaction",
                "source_ref": "transport:human-1",
                "payload": {"text": r"Let \(x \in \mathbb{R}\) and begin."},
            }
        ],
        atoms=[
            interaction(
                "event_human_1",
                sequence=1,
                role="human",
                content=r"Let \(x \in \mathbb{R}\) and begin.",
            )
        ],
        head_updates=[
            interaction_stream_head(
                "event_human_1",
                expected_head_ref=None,
                expected_head_version=0,
            )
        ],
    )
    assert first["status"] == "committed"
    assert first["atoms"][0]["payload"]["content"] == r"Let \(x \in \mathbb{R}\) and begin."

    response = interaction(
        "event_agent_2",
        sequence=2,
        role="agent",
        content="Ready.",
        in_reply_to="event_human_1",
    )
    response["payload"]["thread_refs"] = ["thread_open_1"]
    opened = amos.commit_memory_transaction(
        scope=SCOPE,
        actor="primary-reasoner",
        idempotency_key="agent-2",
        atoms=[
            response,
            thread_root("thread_open_1", opening_event_ref="event_human_1"),
            thread_state(
                "thread_state_1",
                thread_id="thread_open_1",
                revision=1,
                event_refs=["event_human_1", "event_agent_2"],
            ),
        ],
        edges=[
            {
                "source_ref": "event_agent_2",
                "target_ref": "event_human_1",
                "relation": "rel:responds_to",
            },
            {
                "source_ref": "event_agent_2",
                "target_ref": "thread_open_1",
                "relation": "rel:opens",
            },
            {
                "source_ref": "thread_state_1",
                "target_ref": "thread_open_1",
                "relation": "rel:part_of",
            },
        ],
        head_updates=[
            {
                "series_kind": "discourse_thread",
                "series_id": "thread_open_1",
                "new_head_ref": "thread_state_1",
                "expected_head_ref": None,
                "expected_head_version": 0,
            },
            interaction_stream_head(
                "event_agent_2",
                expected_head_ref="event_human_1",
                expected_head_version=1,
            ),
        ],
    )
    assert opened["heads"][0]["head_version"] == 1

    human_followup = interaction(
        "event_human_3",
        sequence=3,
        role="human",
        content="Is it alive?",
        in_reply_to="event_agent_2",
    )
    human_followup["payload"]["thread_refs"] = ["thread_open_1"]
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="participant-ingress",
        idempotency_key="human-3",
        atoms=[human_followup],
        edges=[
            {
                "source_ref": "event_human_3",
                "target_ref": "event_agent_2",
                "relation": "rel:responds_to",
            },
            {
                "source_ref": "event_human_3",
                "target_ref": "thread_open_1",
                "relation": "rel:continues",
            },
        ],
        head_updates=[
            interaction_stream_head(
                "event_human_3",
                expected_head_ref="event_agent_2",
                expected_head_version=2,
            )
        ],
    )

    workspace = amos.compile_cognitive_workspace(
        current_event_ref="event_human_3",
        conversation_id="main",
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        participant_refs=["human:participant", "agent:participant"],
        token_or_byte_budget={"bytes": 48_000},
    )
    assert workspace["profile"] == "amos.cognitive-workspace.v1"
    assert [item["atom_ref"] for item in workspace["temporal_context"]] == [
        "event_human_1",
        "event_agent_2",
        "event_human_3",
    ]
    assert workspace["thread_heads"][0]["thread_id"] == "thread_open_1"
    assert workspace["thread_heads"][0]["private_state"][0]["value"] == "orchid"

    contextual = amos.compile_cognitive_workspace(
        current_event_ref="event_human_3",
        conversation_id="main",
        scope=SCOPE,
        requester="agent:participant",
        target_processor="primary-reasoner",
        context_refs=["event_agent_2"],
        token_or_byte_budget={"bytes": 24_000},
    )
    context_record = contextual["canonical_context"][0]["record"]
    assert context_record["payload"]["content"] == "Ready."
    assert "index_refs" not in context_record
    assert "revision_history" not in context_record
    assert "decay_policy" not in context_record

    shared_view = amos.compile_cognitive_workspace(
        current_event_ref="event_human_3",
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        token_or_byte_budget={"bytes": 24_000},
    )
    assert shared_view["thread_heads"] == []
    assert any(
        item["reason"] == "access_hidden" for item in shared_view["omissions"]
    )
    assert "orchid" not in json.dumps(shared_view)
    hidden_head = amos.get_memory_head(
        scope=SCOPE,
        series_kind="discourse_thread",
        series_id="thread_open_1",
        requester="human:participant",
        target_processor="participant-ui",
    )
    assert hidden_head["status"] == "absent"
    authorized_head = amos.get_memory_head(
        scope=SCOPE,
        series_kind="discourse_thread",
        series_id="thread_open_1",
        requester="agent:participant",
        target_processor="primary-reasoner",
    )
    assert authorized_head["status"] == "found"
    assert authorized_head["head_ref"] == "thread_state_1"

    projection = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
    )
    assert projection["profile"] == "amos.interaction-projection.v2"
    assert [item["content"] for item in projection["events"]] == [
        r"Let \(x \in \mathbb{R}\) and begin.",
        "Ready.",
        "Is it alive?",
    ]
    assert projection["next_after_sequence"] == 3

    tail = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        after_sequence=2,
    )
    assert [item["atom_ref"] for item in tail["events"]] == [
        "event_human_3"
    ]


def test_head_cas_is_atomic_idempotent_and_rebuildable(amos):
    opening = interaction(
        "event_opening",
        sequence=1,
        role="human",
        content="Begin",
    )
    amos.commit_memory_transaction(
        scope=SCOPE,
        actor="ingress",
        idempotency_key="opening",
        atoms=[
            opening,
            thread_root("thread_atomic", opening_event_ref="event_opening"),
            thread_state(
                "state_atomic_1",
                thread_id="thread_atomic",
                revision=1,
                event_refs=["event_opening"],
            ),
        ],
        edges=[
            {
                "source_ref": "state_atomic_1",
                "target_ref": "thread_atomic",
                "relation": "rel:part_of",
            }
        ],
        head_updates=[
            {
                "series_kind": "discourse_thread",
                "series_id": "thread_atomic",
                "new_head_ref": "state_atomic_1",
                "expected_head_ref": None,
                "expected_head_version": 0,
            },
            interaction_stream_head(
                "event_opening",
                expected_head_ref=None,
                expected_head_version=0,
            ),
        ],
    )
    before = amos.store.memory_revision()
    with pytest.raises(CASConflict):
        amos.commit_memory_transaction(
            scope=SCOPE,
            actor="primary",
            idempotency_key="stale-update",
            atoms=[
                interaction(
                    "event_stale",
                    sequence=2,
                    role="agent",
                    content="This must roll back",
                    in_reply_to="event_opening",
                ),
                thread_state(
                    "state_stale",
                    thread_id="thread_atomic",
                    revision=2,
                    event_refs=["event_opening", "event_stale"],
                ),
            ],
            head_updates=[
                {
                    "series_kind": "discourse_thread",
                    "series_id": "thread_atomic",
                    "new_head_ref": "state_stale",
                    "expected_head_ref": "wrong-head",
                    "expected_head_version": 1,
                },
                interaction_stream_head(
                    "event_stale",
                    expected_head_ref="event_opening",
                    expected_head_version=1,
                ),
            ],
        )
    assert amos.store.get_atom("event_stale") is None
    assert amos.store.get_atom("state_stale") is None
    assert amos.store.memory_revision() == before

    next_event = interaction(
        "event_next",
        sequence=2,
        role="agent",
        content="Continued",
        in_reply_to="event_opening",
    )
    next_state = thread_state(
        "state_atomic_2",
        thread_id="thread_atomic",
        revision=2,
        event_refs=["event_opening", "event_next"],
        secret="cedar",
    )
    request = {
        "scope": SCOPE,
        "actor": "primary",
        "idempotency_key": "advance-2",
        "atoms": [next_event, next_state],
        "edges": [
            {
                "source_ref": "state_atomic_2",
                "target_ref": "thread_atomic",
                "relation": "rel:part_of",
            }
        ],
        "head_updates": [
            {
                "series_kind": "discourse_thread",
                "series_id": "thread_atomic",
                "new_head_ref": "state_atomic_2",
                "expected_head_ref": "state_atomic_1",
                "expected_head_version": 1,
            },
            interaction_stream_head(
                "event_next",
                expected_head_ref="event_opening",
                expected_head_version=1,
            ),
        ],
    }
    committed = amos.commit_memory_transaction(**request)
    replayed = amos.commit_memory_transaction(**request)
    assert replayed == committed
    assert amos.store.get_atom("state_atomic_1")["lifecycle_state"] == "superseded"
    assert amos.store.get_memory_head(
        scope=SCOPE,
        series_kind="discourse_thread",
        series_id="thread_atomic",
    )["head_ref"] == "state_atomic_2"
    assert amos.verify_replay()["status"] == "ok"

    rebuilt = amos.rebuild_memory_heads()
    assert rebuilt["head_count"] == 2
    assert {
        (item["series_kind"], item["head_ref"])
        for item in rebuilt["heads"]
    } == {
        ("discourse_thread", "state_atomic_2"),
        ("interaction_stream", "event_next"),
    }
    assert amos.verify_replay()["status"] == "ok"


def test_interaction_stream_head_enforces_total_order_without_supersession(amos):
    with pytest.raises(
        ValidationError, match="requires a matching interaction_stream"
    ):
        amos.commit_memory_transaction(
            scope=SCOPE,
            idempotency_key="missing-stream-head",
            atoms=[
                interaction(
                    "event_missing_head",
                    sequence=1,
                    role="human",
                    content="Not ordered",
                )
            ],
        )

    first = interaction(
        "event_ordered_1",
        sequence=1,
        role="human",
        content="First",
    )
    amos.commit_memory_transaction(
        scope=SCOPE,
        idempotency_key="ordered-1",
        atoms=[first],
        head_updates=[
            interaction_stream_head(
                "event_ordered_1",
                expected_head_ref=None,
                expected_head_version=0,
            )
        ],
    )
    with pytest.raises(CASConflict, match="compare-and-swap"):
        amos.commit_memory_transaction(
            scope=SCOPE,
            idempotency_key="ordered-stale",
            atoms=[
                interaction(
                    "event_ordered_stale",
                    sequence=1,
                    role="agent",
                    content="Stale",
                )
            ],
            head_updates=[
                interaction_stream_head(
                    "event_ordered_stale",
                    expected_head_ref=None,
                    expected_head_version=0,
                )
            ],
        )
    amos.commit_memory_transaction(
        scope=SCOPE,
        idempotency_key="ordered-2",
        atoms=[
            interaction(
                "event_ordered_2",
                sequence=2,
                role="agent",
                content="Second",
                in_reply_to="event_ordered_1",
            )
        ],
        head_updates=[
            interaction_stream_head(
                "event_ordered_2",
                expected_head_ref="event_ordered_1",
                expected_head_version=1,
            )
        ],
    )
    assert amos.store.get_atom("event_ordered_1")["lifecycle_state"] == "active"
    head = amos.get_memory_head(
        scope=SCOPE,
        series_kind="interaction_stream",
        series_id="main",
        requester="human:participant",
        target_processor="participant-ui",
    )
    assert head["head_ref"] == "event_ordered_2"
    assert head["head_version"] == 2
    first_page = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        limit=1,
    )
    assert first_page["has_more"] is True
    final_page = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        after_sequence=1,
        limit=1,
    )
    assert final_page["has_more"] is False


def test_interaction_projection_can_include_typed_linked_lineage(amos):
    response = interaction(
        "event_linked_response",
        sequence=1,
        role="agent",
        content="I am initiating the selected operation.",
    )
    commitment = {
        "id": "commitment_linked",
        "type": "commitment",
        "payload": {
            "agent_id": "agent:participant",
            "description": "A typed caller operation",
            "promised_action": {
                "profile": "example.operation.v1",
                "kind": "example",
            },
            "commitment_status": "pending_authorized_execution",
            "source_event_ref": "event_linked_response",
        },
    }
    outcome = {
        "id": "outcome_linked",
        "type": "action_outcome",
        "payload": {
            "agent_id": "agent:participant",
            "action_ref": "commitment_linked",
            "status": "completed",
            "correction": None,
            "limitation": None,
        },
    }
    amos.commit_memory_transaction(
        scope=SCOPE,
        idempotency_key="linked-operation-lineage",
        atoms=[response, commitment, outcome],
        edges=[
            {
                "source_ref": "event_linked_response",
                "target_ref": "commitment_linked",
                "relation": "rel:supports",
            },
            {
                "source_ref": "commitment_linked",
                "target_ref": "outcome_linked",
                "relation": "rel:produced_outcome",
            },
        ],
        head_updates=[
            interaction_stream_head(
                "event_linked_response",
                expected_head_ref=None,
                expected_head_version=0,
            )
        ],
    )

    projection = amos.compile_interaction_projection(
        conversation_id="main",
        scope=SCOPE,
        requester="human:participant",
        target_processor="participant-ui",
        linked_atom_types=["commitment", "action_outcome"],
        linked_depth=2,
    )

    linked = projection["events"][0]["linked_records"]
    assert [
        (
            item["depth"],
            item["relation"],
            item["record"]["id"],
            item["record"]["type"],
        )
        for item in linked
    ] == [
        (1, "rel:supports", "commitment_linked", "commitment"),
        (
            2,
            "rel:produced_outcome",
            "outcome_linked",
            "action_outcome",
        ),
    ]


def test_interaction_projection_rejects_half_bound_link_requests(amos):
    with pytest.raises(
        ValidationError,
        match="linked_atom_types and linked_depth must be supplied together",
    ):
        amos.compile_interaction_projection(
            conversation_id="main",
            scope=SCOPE,
            linked_atom_types=["commitment"],
        )


def test_private_state_requires_restricted_visibility(amos):
    with pytest.raises(ValidationError, match="restricted access_policy"):
        amos.commit_memory_transaction(
            scope=SCOPE,
            atoms=[
                interaction(
                    "event_visibility",
                    sequence=1,
                    role="human",
                    content="Begin",
                ),
                thread_root(
                    "thread_visibility", opening_event_ref="event_visibility"
                ),
                {
                    **thread_state(
                        "state_visibility",
                        thread_id="thread_visibility",
                        revision=1,
                        event_refs=["event_visibility"],
                    ),
                    "access_policy": {
                        "visibility": ["all"],
                        "mutable_by": ["owner"],
                    },
                },
            ],
        )


def test_http_memory_transaction_workspace_and_cas_conflict(tmp_path):
    try:
        server = AmosHTTPServer(
            ("127.0.0.1", 0), str(tmp_path / "continuity-http.sqlite3")
        )
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    try:
        committed = post(
            "/v1/memory-transactions:commit",
            {
                "profile": "amos.memory-transaction.v1",
                "scope": SCOPE,
                "actor": "http-ingress",
                "idempotency_key": "http-event",
                "atoms": [
                    interaction(
                        "event_http_1",
                        sequence=1,
                        role="human",
                        content=r"Preserve **Markdown** and \(x^2\).",
                    )
                ],
                "head_updates": [
                    interaction_stream_head(
                        "event_http_1",
                        expected_head_ref=None,
                        expected_head_version=0,
                    )
                ],
            },
        )
        assert committed["status"] == "committed"
        workspace = post(
            "/v1/cognitive-workspaces:compile",
            {
                "profile": "amos.cognitive-workspace-request.v1",
                "current_event_ref": "event_http_1",
                "conversation_id": "main",
                "scope": SCOPE,
                "requester": "human:participant",
                "target_processor": "primary-reasoner",
                "token_or_byte_budget": {"bytes": 20_000},
            },
        )
        assert workspace["current_event"]["content"] == (
            r"Preserve **Markdown** and \(x^2\)."
        )
        projection = post(
            "/v1/interaction-projections:compile",
            {
                "profile": "amos.interaction-projection-request.v2",
                "conversation_id": "main",
                "scope": SCOPE,
                "requester": "human:participant",
                "target_processor": "participant-ui",
            },
        )
        assert projection["events"][0]["content"] == (
            r"Preserve **Markdown** and \(x^2\)."
        )
        head = post(
            "/v1/memory-heads:get",
            {
                "profile": "amos.memory-head-request.v1",
                "scope": SCOPE,
                "series_kind": "interaction_stream",
                "series_id": "main",
                "requester": "human:participant",
                "target_processor": "participant-ui",
            },
        )
        assert head["status"] == "found"
        assert head["head_ref"] == "event_http_1"
        assert head["head_version"] == 1

        with pytest.raises(urllib.error.HTTPError) as captured:
            post(
                "/v1/memory-transactions:commit",
                {
                    "scope": SCOPE,
                    "atoms": [
                        thread_state(
                            "state_http_stale",
                            thread_id="missing-thread",
                            revision=2,
                            event_refs=["event_http_1"],
                        )
                    ],
                    "head_updates": [
                        {
                            "series_kind": "discourse_thread",
                            "series_id": "missing-thread",
                            "new_head_ref": "state_http_stale",
                            "expected_head_ref": "old",
                            "expected_head_version": 1,
                        }
                    ],
                },
            )
        assert captured.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
