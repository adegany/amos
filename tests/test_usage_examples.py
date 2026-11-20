import socket
from datetime import time

import pytest

from usage_examples import preference_continuity, successful_action_recall
from usage_examples.memory_http import Memory
from usage_examples.trial import run_trial, service


@pytest.fixture(autouse=True)
def require_loopback():
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError as exc:
        pytest.skip(f"loopback sockets unavailable: {exc}")


@pytest.mark.parametrize("example", [preference_continuity, successful_action_recall])
def test_example_verifies_behavior_and_source_after_restart(example):
    run_trial(example.first_session, example.next_session)


def test_repeated_message_is_one_private_preference(tmp_path):
    with service(tmp_path / "preferences.sqlite3") as url:
        maya = Memory(url, "service:reminders", {"user": "maya"})
        message = ("reminders:maya:message:17", "Use 24-hour time.")
        original = preference_continuity.set_preference(maya, *message)
        replay = preference_continuity.set_preference(maya, *message)
        assert original == replay
        item, = maya.recall("preference", holder="maya")
        assert item["atom_ref"] == original["atoms"][0]["id"]
        evidence_id, = item["evidence_refs"]
        assert maya.evidence(evidence_id)["source_ref"] == message[0]

        sam = Memory(url, "service:reminders", {"user": "sam"})
        assert sam.recall("preference") == []
        assert sam.evidence(evidence_id) is None
        assert preference_continuity.reminder(sam, time(14, 30)) == ("Reminder at 02:30 PM.", None)
