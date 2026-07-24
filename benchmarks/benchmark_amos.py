#!/usr/bin/env python3
"""Current local AMOS v1 benchmark.

This benchmark intentionally uses only the Python standard library and the
in-process AMOS service API. It measures the v1-local SQLite baseline rather
than HTTP or network overhead. The workload covers canonical writes, exact and
associative retrieval, coherent reasoning frames, demand-loaded pages, governed
semantic/graph maintenance, and final-state replay verification.
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


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
    }


def _mean(values: list[int]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _atom_id(index: int) -> str:
    prefix = ("semantic", "trace", "lesson", "directive")[index % 4]
    return f"bench_{prefix}_{index:06d}"


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
        "semantic_facets": [
            {
                "subject": f"UPRO control signature sig_{index % 12}",
                "intent": "evaluate benchmark memory",
                "outcome": f"outcome_{index % 5}",
                "outcome_direction": "positive" if index % 3 else "neutral",
                "time_index": index,
                "semantic_context_key": f"benchmark:sig_{index % 12}",
            }
        ],
    }
    if kind == 0:
        atom = {
            "id": f"bench_semantic_{index:06d}",
            "type": "semantic",
            "payload": base_payload,
            "scope": scope,
            "salience": 0.5 + ((index % 10) / 20),
            "utility": 0.5 + ((index % 8) / 20),
            "supersedes": [f"bench_semantic_{index - 12:06d}"]
            if index >= 12
            else [],
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
                "graph_relations": [
                    {
                        "source_ref": "$self",
                        "target_ref": source_ref,
                        "relation": "rel:derived_from",
                    },
                    {
                        "source_ref": "$self",
                        "target_ref": source_ref,
                        "relation": "rel:caused_by",
                    },
                ]
                if source_ref
                else [],
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

        retrieve_cold_latencies: list[float] = []
        retrieve_warm_latencies: list[float] = []
        item_counts: list[int] = []
        for index in range(args.retrievals):
            atom_index = index % args.atoms
            request = {
                "cues": [
                    (
                        f"Benchmark memory {atom_index} UPRO chunk "
                        f"{atom_index % 50} sig_{atom_index % 12} "
                        f"outcome_{atom_index % 5}"
                    )
                ],
                "scope": {"tenant": "bench", "asset": "UPRO"},
                "requester": "benchmark",
                "target_processor": "planner",
                "retrieval_mode": "planner",
                "max_items": args.max_items,
                "run_policy": False,
            }
            op_started = time.perf_counter()
            packet = amos.retrieve_packet(**request)
            retrieve_cold_latencies.append(_duration_ms(op_started))
            item_counts.append(len(packet.get("items", [])))
            op_started = time.perf_counter()
            amos.retrieve_packet(**request)
            retrieve_warm_latencies.append(_duration_ms(op_started))

        exact_latencies: list[float] = []
        exact_found = 0
        for index in range(args.exact_lookups):
            atom_index = (index * 7) % args.atoms
            op_started = time.perf_counter()
            exact = amos.retrieve_atom(
                _atom_id(atom_index),
                scope={"tenant": "bench", "asset": "UPRO"},
                requester="benchmark",
                target_processor="reasoner",
                include_superseded=True,
                run_policy=False,
            )
            exact_latencies.append(_duration_ms(op_started))
            exact_found += int(exact.get("found") is True)

        frame_latencies: list[float] = []
        resident_unit_counts: list[int] = []
        descriptor_counts: list[int] = []
        frame_used_bytes: list[int] = []
        page_latencies: list[float] = []
        page_source_counts: list[int] = []
        reasoning_cues = (
            "UPRO control signature sig_0 outcome",
            "UPRO control signature sig_4 outcome",
            "UPRO control signature sig_8 outcome",
        )
        for index in range(args.reasoning_frames):
            need = reasoning_cues[index % len(reasoning_cues)]
            op_started = time.perf_counter()
            frame = amos.compile_memory_frame(
                need=need,
                purpose="benchmark coherent historical reasoning",
                depth="working_frame",
                task_context={
                    "project_id": "amos-benchmark",
                    "phase": "performance",
                },
                scope={"tenant": "bench", "asset": "UPRO"},
                requester="benchmark",
                target_processor="reasoner",
                token_or_byte_budget={"tokens": args.frame_tokens},
                run_policy=False,
            )
            frame_latencies.append(_duration_ms(op_started))
            resident_unit_counts.append(len(frame.get("units", [])))
            descriptor_counts.append(len(frame.get("page_index", [])))
            frame_used_bytes.append(int(frame.get("budget", {}).get("used_bytes", 0)))
            descriptors = frame.get("page_index", [])
            if not descriptors:
                continue
            page_started = time.perf_counter()
            page = amos.load_memory_page(
                frame_id=frame["frame_id"],
                revision=frame["revision"],
                page=descriptors[0],
                need=need,
                purpose="benchmark demand-loaded supporting detail",
                depth="supporting",
                scope={"tenant": "bench", "asset": "UPRO"},
                requester="benchmark",
                target_processor="reasoner",
                token_or_byte_budget={"tokens": args.page_tokens},
                run_policy=False,
            )
            page_latencies.append(_duration_ms(page_started))
            page_source_counts.append(len(page.get("source_atom_refs", [])))

        committed_atoms = amos.store.list_atoms()
        semantic_facet_atoms = sum(
            bool((atom.get("payload") or {}).get("semantic_facets"))
            for atom in committed_atoms
        )
        graph_relation_atoms = sum(
            bool((atom.get("payload") or {}).get("graph_relations"))
            for atom in committed_atoms
        )
        edge_count_before_policy = len(amos.store.list_edges())

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

        verify_started = time.perf_counter()
        verify = amos.verify_replay()
        verify_ms = _duration_ms(verify_started)

        db_size = db_path.stat().st_size if db_path.exists() else 0
        wal_path = Path(f"{db_path}-wal")
        shm_path = Path(f"{db_path}-shm")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        shm_size = shm_path.stat().st_size if shm_path.exists() else 0
        sqlite_footprint = db_size + wal_size + shm_size
        distiller = policy.get("results", {}).get("maintenance_distiller", {})
        exact_summary = _latency_summary(exact_latencies)
        retrieve_cold_summary = _latency_summary(retrieve_cold_latencies)
        retrieve_warm_summary = _latency_summary(retrieve_warm_latencies)
        frame_summary = _latency_summary(frame_latencies)
        page_summary = _latency_summary(page_latencies)
        result = {
            "schema_version": 2,
            "benchmark": "amos_v1_local_sqlite",
            "parameters": {
                "atoms": args.atoms,
                "retrievals": args.retrievals,
                "exact_lookups": args.exact_lookups,
                "reasoning_frames": args.reasoning_frames,
                "max_items": args.max_items,
                "frame_tokens": args.frame_tokens,
                "page_tokens": args.page_tokens,
                "run_policy": args.run_policy,
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
                "exact_lookup_latency_ms_p50": exact_summary["p50"],
                "exact_lookup_latency_ms_p95": exact_summary["p95"],
                "exact_lookup_found": exact_found,
                "retrieve_cold_latency_ms_p50": retrieve_cold_summary["p50"],
                "retrieve_cold_latency_ms_p95": retrieve_cold_summary["p95"],
                "retrieve_warm_latency_ms_p50": retrieve_warm_summary["p50"],
                "retrieve_warm_latency_ms_p95": retrieve_warm_summary["p95"],
                "retrieve_avg_items": _mean(item_counts),
                "reasoning_frame_latency_ms_p50": frame_summary["p50"],
                "reasoning_frame_latency_ms_p95": frame_summary["p95"],
                "reasoning_frame_avg_resident_units": _mean(resident_unit_counts),
                "reasoning_frame_avg_page_descriptors": _mean(descriptor_counts),
                "reasoning_frame_avg_used_bytes": _mean(frame_used_bytes),
                "reasoning_page_loads": len(page_latencies),
                "reasoning_page_latency_ms_p50": page_summary["p50"],
                "reasoning_page_latency_ms_p95": page_summary["p95"],
                "reasoning_page_avg_source_atoms": _mean(page_source_counts),
                "verify_replay_ms": round(verify_ms, 3),
                "memory_policy_ms": round(policy_ms, 3),
                "db_size_bytes": db_size,
                "wal_size_bytes": wal_size,
                "shm_size_bytes": shm_size,
                "sqlite_footprint_bytes": sqlite_footprint,
                "verify_replay_status": verify.get("status"),
                "memory_policy_status": policy.get("status"),
                "semantic_facet_atoms": semantic_facet_atoms,
                "graph_relation_atoms": graph_relation_atoms,
                "edge_count_before_policy": edge_count_before_policy,
                "maintenance_proposals": len(distiller.get("proposals", [])),
                "maintenance_committed": len(distiller.get("committed", [])),
                "maintenance_deferred": len(distiller.get("deferred", [])),
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
    replay_label = (
        "Replay verification after policy"
        if params["run_policy"]
        else "Replay verification"
    )
    return "\n".join(
        [
            "| Benchmark | Result |",
            "| --- | ---: |",
            f"| Atoms committed | {params['atoms']} |",
            f"| Atoms with semantic facets / graph relations | {res['semantic_facet_atoms']} / {res['graph_relation_atoms']} |",
            f"| Exact lookups | {params['exact_lookups']} ({res['exact_lookup_found']} found) |",
            f"| Exact lookup latency p50 / p95 | {res['exact_lookup_latency_ms_p50']} ms / {res['exact_lookup_latency_ms_p95']} ms |",
            f"| Packet retrievals | {params['retrievals']} cold + {params['retrievals']} warm |",
            f"| Commit throughput | {res['commit_atoms_per_second']} atoms/s |",
            f"| Commit latency p50 / p95 | {res['commit_latency_ms_p50']} ms / {res['commit_latency_ms_p95']} ms |",
            f"| Cold packet latency p50 / p95 | {res['retrieve_cold_latency_ms_p50']} ms / {res['retrieve_cold_latency_ms_p95']} ms |",
            f"| Warm packet latency p50 / p95 | {res['retrieve_warm_latency_ms_p50']} ms / {res['retrieve_warm_latency_ms_p95']} ms |",
            f"| Average packet items | {res['retrieve_avg_items']} |",
            f"| Reasoning frame compiles | {params['reasoning_frames']} at {params['frame_tokens']} tokens |",
            f"| Reasoning frame latency p50 / p95 | {res['reasoning_frame_latency_ms_p50']} ms / {res['reasoning_frame_latency_ms_p95']} ms |",
            f"| Average resident units / page descriptors | {res['reasoning_frame_avg_resident_units']} / {res['reasoning_frame_avg_page_descriptors']} |",
            f"| Demand-page loads | {res['reasoning_page_loads']} at {params['page_tokens']} tokens |",
            f"| Demand-page latency p50 / p95 | {res['reasoning_page_latency_ms_p50']} ms / {res['reasoning_page_latency_ms_p95']} ms |",
            f"| Forced memory policy run | {res['memory_policy_ms']} ms ({res['memory_policy_status']}) |",
            f"| Maintenance proposals / committed / deferred | {res['maintenance_proposals']} / {res['maintenance_committed']} / {res['maintenance_deferred']} |",
            f"| {replay_label} | {res['verify_replay_ms']} ms ({res['verify_replay_status']}) |",
            f"| Edges before policy / final | {res['edge_count_before_policy']} / {res['edge_count']} |",
            f"| Final atoms / edges | {res['atom_count']} / {res['edge_count']} |",
            f"| SQLite DB / WAL / SHM / total footprint | {res['db_size_bytes']} / {res['wal_size_bytes']} / {res['shm_size_bytes']} / {res['sqlite_footprint_bytes']} bytes |",
            f"| Environment | Python {env['python']}; {env['cpu_count']} CPUs; {env['platform']} |",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=100)
    parser.add_argument("--retrievals", type=int, default=20)
    parser.add_argument("--exact-lookups", type=int, default=20)
    parser.add_argument("--reasoning-frames", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--frame-tokens", type=int, default=1600)
    parser.add_argument("--page-tokens", type=int, default=1800)
    parser.add_argument("--db", default="")
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--run-policy", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "atoms",
        "retrievals",
        "exact_lookups",
        "reasoning_frames",
        "max_items",
        "frame_tokens",
        "page_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    result = _benchmark(args)
    if args.markdown:
        print(_markdown(result))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
