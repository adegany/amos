from __future__ import annotations


def test_amos_facade_composes_subsystems_without_backreferences(amos):
    components = {
        "access": amos.access,
        "capacity": amos.capacity,
        "diagnostics": amos.diagnostics,
        "graph": amos.graph,
        "indexes": amos.indexes,
        "mutations": amos.mutations,
        "policy": amos.policy,
        "reasoning": amos.reasoning,
        "retrieval": amos.retrieval,
        "stewardship": amos.stewardship,
        "views": amos.views,
    }

    assert len({id(component) for component in components.values()}) == len(components)
    for component in components.values():
        assert not hasattr(component, "amos")
        assert not hasattr(component, "facade")

    assert amos.policy.run_steward.__self__ is amos.stewardship
    assert amos.retrieval.run_memory_policy.__self__ is amos.policy

    committed = amos.commit_atom(
        {"type": "belief", "payload": {"claim": "facade delegation works"}}
    )
    assert committed["status"] == "committed"
    assert committed["atom"]["id"] in {
        item["atom_ref"]
        for item in amos.retrieve_packet(
            cues=["facade delegation"], run_policy=False
        )["items"]
    }
