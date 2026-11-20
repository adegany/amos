"""A reminder agent honors a user's stated time format in a later session."""

from datetime import time

from .memory_http import Memory


def set_preference(memory, message_id, message):
    # Two explicit commands keep the user's own words as the source of the rule.
    formats = {"Use 24-hour time.": "24h", "Use 12-hour time.": "12h"}
    return memory.remember(
        source_ref=message_id,
        source_type="user_message",
        evidence={"text": message},
        atom_type="preference",
        payload={
            "holder": memory.scope["user"],
            "polarity": "prefer",
            "target": formats[message],
            "applicability_scope": "reminder time format",
            "strength": "explicit",
        },
    )


def reminder(memory, when):
    items = memory.recall(
        "preference",
        holder=memory.scope["user"],
        applicability_scope="reminder time format",
        polarity="prefer",
    )
    preference = items[0] if items else None
    clock = preference["payload"]["target"] if preference else "12h"
    display = when.strftime("%H:%M" if clock == "24h" else "%I:%M %p")
    return f"Reminder at {display}.", preference


def first_session(url):
    memory = Memory(url, "service:reminders", {"user": "maya"})
    reply, citation = reminder(memory, time(14, 30))
    assert reply == "Reminder at 02:30 PM." and citation is None
    print("Before  ", reply)
    message = "Use 24-hour time."
    set_preference(memory, "reminders:maya:message:17", message)
    print("User    ", message)


def next_session(url):
    memory = Memory(url, "service:reminders", {"user": "maya"})
    reply, citation = reminder(memory, time(14, 30))
    assert reply == "Reminder at 14:30."
    source = memory.evidence(citation["evidence_refs"][0])
    assert source["payload"]["text"] == "Use 24-hour time."
    print("After   ", reply)
    print("Cited   ", citation["atom_ref"], "→", source["evidence_id"], source["payload"])


if __name__ == "__main__":
    from .trial import run_trial

    run_trial(first_session, next_session)
