from __future__ import annotations

from argparse import Namespace

from benchmarks.benchmark_amos import _benchmark, _markdown


def test_current_benchmark_covers_reasoning_and_complete_storage_footprint(tmp_path):
    result = _benchmark(
        Namespace(
            atoms=24,
            retrievals=2,
            exact_lookups=2,
            reasoning_frames=1,
            max_items=8,
            frame_tokens=1600,
            page_tokens=1800,
            db=str(tmp_path / "benchmark.sqlite3"),
            keep_db=True,
            run_policy=False,
        )
    )

    assert result["schema_version"] == 2
    assert result["parameters"]["retrievals"] == 2
    metrics = result["results"]
    assert metrics["semantic_facet_atoms"] == 24
    assert metrics["graph_relation_atoms"] == 6
    assert metrics["exact_lookup_found"] == 2
    assert metrics["retrieve_cold_latency_ms_p50"] > 0
    assert metrics["retrieve_warm_latency_ms_p50"] > 0
    assert metrics["reasoning_frame_avg_page_descriptors"] >= 1
    assert metrics["reasoning_page_loads"] == 1
    assert metrics["verify_replay_status"] == "ok"
    assert metrics["sqlite_footprint_bytes"] == (
        metrics["db_size_bytes"]
        + metrics["wal_size_bytes"]
        + metrics["shm_size_bytes"]
    )

    markdown = _markdown(result)
    assert "Cold packet latency p50 / p95" in markdown
    assert "Demand-page loads" in markdown
    assert "Replay verification |" in markdown
