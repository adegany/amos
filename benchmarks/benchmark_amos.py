#!/usr/bin/env python3
"""Quick local AMOS benchmark.

This benchmark intentionally uses only the Python standard library and the
in-process AMOS service API. It measures the v1-local SQLite baseline rather
than HTTP or network overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amos import Amos  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _commit_atom(amos: Amos, index: int) -> None:
    kind = index % 4
    scope = {"tenant": "bench", "asset": "UPRO"}
    base_payload: dict[str, Any] = {
        "benchmark": True,
        "chunk": index,
        "asset": "UPRO",
        "summary": (
            f"Benchmark memory {index} about UPRO chunk {index % 50}, "
            f"control signature sig_{index % 12}, and outcome {index % 5}."
        ),
        "tags": ["benchmark", f"sig_{index % 12}", f"outcome_{index % 5}"],
    }
    if kind == 0:
        atom = {
            "id": f"bench_semantic_{index:06d}",
            "type": "semantic",
            "payload": base_payload,
            "scope": scope,
            "salience": 0.5 + ((index % 10) / 20),
            "utility": 0.5 + ((index % 8) / 20),
        }
    elif kind == 1:
        atom = {
            "id": f"bench_trace_{index:06d}",
            "type": "agentic_trace",
            "payload": {
                **base_payload,
                "task": "benchmark retrieval",
                "action": f"evaluate control signature sig_{index % 12}",
                "outcome": "supported" if index % 3 else "mixed",
            },
            "scope": scope,
            "confidence": {"level": "medium", "score": 0.55},
        }
    elif kind == 2:
        source_ref = f"bench_trace_{index - 1:06d}" if index > 1 else ""
        atom = {
            "id": f"bench_lesson_{index:06d}",
            "type": "semantic",
            "payload": {
                **base_payload,
                "distillation_type": "benchmark_lesson",
                "source_refs": [source_ref] if source_ref else [],
                "control_signature": f"sig_{index % 12}",
                "metric_deltas": {"score": round((index % 9) / 100.0, 4)},
            },
            "scope": scope,
            "layer": "consolidated_long_term",
            "retention_class": "distilled",
            "confidence": {"level": "medium-high", "score": 0.76},
        }
    else:
        memory_ref = f"bench_lesson_{index - 1:06d}" if index > 2 else ""
        atom = {
            "id": f"bench_directive_{index:06d}",
            "type": "agentic_trace",
            "payload": {
                **base_payload,
                "task": "benchmark directive",
                "action": f"use memory for sig_{index % 12}",
                "outcome": "issued",
                "memory_references": [{"id": memory_ref}] if memory_ref else [],
            },
            "scope": scope,
            "salience": 0.72,
            "utility": 0.8,
        }
    amos.commit_atom(atom, actor="benchmark")


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    db_path = Path(args.db).expanduser() if args.db else None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if db_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="amos-bench-")
        db_path = Path(temp_dir.name) / "amos.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not args.keep_db:
        db_path.unlink()

    amos = Amos(db_path)
    try:
        commit_latencies: list[float] = []
        started = time.perf_counter()
        for index in range(args.atoms):
            op_started = time.perf_counter()
            _commit_atom(amos, index)
            commit_latencies.append(_duration_ms(op_started))
        commit_total_ms = _duration_ms(started)

        retrieve_latencies: list[float] = []
        item_counts: list[int] = []
        cues = [
            "UPRO supported control signature sig_3",
            "benchmark lesson outcome",
            "chunk metric deltas",
            "agentic trace directive",
        ]
        for index in range(args.retrievals):
            op_started = time.perf_counter()
            packet = amos.retrieve_packet(
                cues=[cues[index % len(cues)]],
                scope={"tenant": "bench", "asset": "UPRO"},
                requester="benchmark",
                target_processor="planner",
                retrieval_mode="planner",
                max_items=args.max_items,
                run_policy=False,
            )
            retrieve_latencies.append(_duration_ms(op_started))
            item_counts.append(len(packet.get("items", [])))

        verify_started = time.perf_counter()
        verify = amos.verify_replay()
        verify_ms = _duration_ms(verify_started)

        policy: dict[str, Any] = {"status": "skipped"}
        policy_ms = 0.0
        if args.run_policy:
            policy_started = time.perf_counter()
            policy = amos.run_memory_policy(
                force=True,
                trigger="benchmark",
                scope={"tenant": "bench", "asset": "UPRO"},
            )
            policy_ms = _duration_ms(policy_started)

        db_size = db_path.stat().st_size if db_path.exists() else 0
        result = {
            "schema_version": 1,
            "benchmark": "amos_v1_local_sqlite",
            "parameters": {
                "atoms": args.atoms,
                "retrievals": args.retrievals,
                "max_items": args.max_items,
                "db": str(db_path),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
                "cpu_count": os.cpu_count(),
            },
            "results": {
                "commit_total_ms": round(commit_total_ms, 3),
                "commit_atoms_per_second": round(args.atoms / (commit_total_ms / 1000.0), 2),
                "commit_latency_ms_p50": round(statistics.median(commit_latencies), 3),
                "commit_latency_ms_p95": round(_percentile(commit_latencies, 0.95), 3),
                "retrieve_latency_ms_p50": round(statistics.median(retrieve_latencies), 3),
                "retrieve_latency_ms_p95": round(_percentile(retrieve_latencies, 0.95), 3),
                "retrieve_avg_items": round(statistics.mean(item_counts), 2)
                if item_counts
                else 0,
                "verify_replay_ms": round(verify_ms, 3),
                "memory_policy_ms": round(policy_ms, 3),
                "db_size_bytes": db_size,
                "verify_replay_status": verify.get("status"),
                "memory_policy_status": policy.get("status"),
                "graph_version": amos.store.graph_version(),
                "atom_count": len(amos.store.list_atoms()),
                "edge_count": len(amos.store.list_edges()),
            },
        }
    finally:
        amos.close()
        if temp_dir is not None and not args.keep_db:
            temp_dir.cleanup()
    return result


def _markdown(result: dict[str, Any]) -> str:
    params = result["parameters"]
    env = result["environment"]
    res = result["results"]
    return "\n".join(
        [
            "| Benchmark | Result |",
            "| --- | ---: |",
            f"| Atoms committed | {params['atoms']} |",
            f"| Retrievals | {params['retrievals']} |",
            f"| Commit throughput | {res['commit_atoms_per_second']} atoms/s |",
            f"| Commit latency p50 / p95 | {res['commit_latency_ms_p50']} ms / {res['commit_latency_ms_p95']} ms |",
            f"| Retrieval latency p50 / p95 | {res['retrieve_latency_ms_p50']} ms / {res['retrieve_latency_ms_p95']} ms |",
            f"| Average packet items | {res['retrieve_avg_items']} |",
            f"| Replay verification | {res['verify_replay_ms']} ms ({res['verify_replay_status']}) |",
            f"| Forced memory policy run | {res['memory_policy_ms']} ms ({res['memory_policy_status']}) |",
            f"| Final atoms / edges | {res['atom_count']} / {res['edge_count']} |",
            f"| SQLite DB size | {res['db_size_bytes']} bytes |",
            f"| Environment | Python {env['python']}; {env['cpu_count']} CPUs; {env['platform']} |",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=100)
    parser.add_argument("--retrievals", type=int, default=20)
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--db", default="")
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--run-policy", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    result = _benchmark(args)
    if args.markdown:
        print(_markdown(result))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
