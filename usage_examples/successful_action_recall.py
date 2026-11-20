"""An import worker reuses a successful delimiter for the same feed."""

import csv
from io import StringIO

from .memory_http import Memory


def import_rows(text, delimiter):
    reader = csv.DictReader(StringIO(text), delimiter=delimiter)
    return list(reader) if reader.fieldnames == ["sku", "qty"] else []


def record_attempt(memory, feed, run_id, text, delimiter):
    rows = import_rows(text, delimiter)
    result = {"feed": feed, "delimiter": delimiter, "rows_accepted": len(rows)}
    memory.remember(
        source_ref=run_id,
        source_type="import_result",
        evidence=result,
        atom_type="action_outcome",
        payload={
            **result,
            "agent_id": "importer",
            "action_ref": run_id,
            "status": "success" if rows else "failed",
        },
    )
    return rows


def run_import(memory, feed, run_id, text):
    matches = memory.recall("action_outcome", feed=feed, status="success")
    delimiter = matches[0]["payload"]["delimiter"] if matches else ","
    rows = record_attempt(memory, feed, run_id, text, delimiter)
    return rows, matches


def first_session(url):
    memory = Memory(url, "service:imports", {"tenant": "demo"})
    text = "sku\tqty\nH1\t2\n"
    missed, matches = run_import(memory, "Harbor", "demo:harbor:run:1", text)
    assert missed == [] and matches == []
    print("Before   Default comma accepted 0 Harbor rows.")
    recovered = record_attempt(memory, "Harbor", "demo:harbor:run:2", text, "\t")
    assert recovered == [{"sku": "H1", "qty": "2"}]
    delta_rows, _ = run_import(memory, "Delta", "demo:delta:run:1", "sku,qty\nD1,4\n")
    assert delta_rows == [{"sku": "D1", "qty": "4"}]
    print("Recorded Harbor failed with comma, succeeded with tab. Delta succeeded with comma.")


def next_session(url):
    memory = Memory(url, "service:imports", {"tenant": "demo"})
    rows, matches = run_import(memory, "Harbor", "demo:harbor:run:3", "sku\tqty\nH2\t5\n")
    item, = matches
    assert rows == [{"sku": "H2", "qty": "5"}]
    source = memory.evidence(item["evidence_refs"][0])
    assert source["source_ref"] == "demo:harbor:run:2" and source["payload"]["rows_accepted"] == 1
    print("After    Recalled delimiter", repr(item["payload"]["delimiter"]), "and imported", rows)
    print("Cited   ", item["atom_ref"], "→", source["evidence_id"], source["payload"])


if __name__ == "__main__":
    from .trial import run_trial

    run_trial(first_session, next_session)
