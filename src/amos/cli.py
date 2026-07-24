"""Command line interface for the AMOS v1 reference implementation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import AmosError
from .http_api import serve
from .schemas import parse_json_arg
from .service import Amos


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    amos = Amos(args.db, maintenance_processor_paths=args.maintenance_processor)
    try:
        result = args.func(amos, args)
    except AmosError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    finally:
        amos.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amos")
    parser.add_argument(
        "--db",
        default="amos.sqlite3",
        help="SQLite path owned by this AMOS service process",
    )
    parser.add_argument(
        "--maintenance-processor",
        action="append",
        default=[],
        help="External processor import path in module:attribute form",
    )
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="initialize a local AMOS store")
    init.set_defaults(func=lambda amos, _args: amos.health_memory())

    capture = sub.add_parser("capture-event", help="capture evidence into the journal")
    capture.add_argument("--source-type", required=True)
    capture.add_argument("--source-ref", required=True)
    capture.add_argument("--payload", required=True, help="JSON payload or @file")
    capture.add_argument("--scope", default="{}")
    capture.add_argument("--actor", default="cli")
    capture.add_argument("--idempotency-key")
    capture.set_defaults(func=_capture_event)

    commit = sub.add_parser("commit-atom", help="commit a typed memory atom")
    commit.add_argument("--atom", help="full atom JSON object or @file")
    commit.add_argument("--type", dest="atom_type", help="atom type when --atom is absent")
    commit.add_argument("--payload", help="atom payload JSON or @file")
    commit.add_argument("--scope", default="{}")
    commit.add_argument("--actor", default="cli")
    commit.add_argument("--authorization-context", default="{}")
    commit.add_argument("--idempotency-key")
    commit.set_defaults(func=_commit_atom)

    retrieve = sub.add_parser("retrieve", help="render a memory packet")
    retrieve.add_argument("--cue", action="append", default=[])
    retrieve.add_argument("--scope", default="{}")
    retrieve.add_argument("--requester", default="cli")
    retrieve.add_argument("--target-processor", default="reasoner")
    retrieve.add_argument(
        "--memory-mode",
        choices=["operational_recall", "deliberation", "historical_review"],
        default="operational_recall",
    )
    retrieve.add_argument("--max-items", type=int, default=8)
    retrieve.add_argument("--include-conflicts", action="store_true")
    retrieve.add_argument("--include-archived", action="store_true")
    retrieve.add_argument("--attention-context", default="{}")
    retrieve.set_defaults(func=_retrieve)

    ratify = sub.add_parser(
        "ratify-proposal",
        help="adopt a proposal through constitutional self-ratification",
    )
    ratify.add_argument("--proposal-ref", required=True)
    ratify.add_argument("--adjudication-ref", required=True)
    ratify.add_argument("--expected-version", type=int, required=True)
    ratify.add_argument("--actor", required=True)
    ratify.add_argument("--authorization-context", required=True)
    ratify.add_argument("--idempotency-key")
    ratify.set_defaults(func=_ratify_proposal)

    resolve = sub.add_parser(
        "resolve-proposal",
        help="apply a guarded non-positive or unresolved adjudication",
    )
    resolve.add_argument("--proposal-ref", required=True)
    resolve.add_argument("--adjudication-ref", required=True)
    resolve.add_argument("--expected-version", type=int, required=True)
    resolve.add_argument("--actor", required=True)
    resolve.add_argument("--authorization-context", required=True)
    resolve.add_argument("--idempotency-key")
    resolve.set_defaults(func=_resolve_proposal)

    replace = sub.add_parser(
        "replace-constitutional-record",
        help="atomically activate a self-ratified constitutional successor",
    )
    replace.add_argument("--current-ref", required=True)
    replace.add_argument("--successor-ref", required=True)
    replace.add_argument("--adjudication-ref", required=True)
    replace.add_argument("--expected-current-version", type=int, required=True)
    replace.add_argument("--expected-successor-version", type=int, required=True)
    replace.add_argument("--actor", required=True)
    replace.add_argument("--authorization-context", required=True)
    replace.add_argument("--idempotency-key")
    replace.set_defaults(func=_replace_constitutional_record)

    provenance = sub.add_parser(
        "provenance-analysis",
        help="analyze root evidence independence and circular support",
    )
    provenance.add_argument("--atom-ref", required=True)
    provenance.add_argument("--max-depth", type=int, default=64)
    provenance.set_defaults(func=_provenance_analysis)

    diachronic = sub.add_parser(
        "diachronic-status",
        help="evaluate independent self-ratification confirmations over time",
    )
    diachronic.add_argument("--subject-ref", required=True)
    diachronic.add_argument("--identity-ref", required=True)
    diachronic.add_argument("--required-confirmations", type=int, default=2)
    diachronic.add_argument("--min-interval-seconds", type=int, default=0)
    diachronic.set_defaults(func=_diachronic_status)

    self_view = sub.add_parser("self-awareness", help="render a self-awareness view")
    self_view.add_argument("--agent-id", required=True)
    self_view.add_argument("--scope", default="{}")
    self_view.add_argument("--requester", default="cli")
    self_view.set_defaults(func=_self_awareness)

    recall = sub.add_parser("agentic-recall", help="render action/outcome recall")
    recall.add_argument("--agent-id", required=True)
    recall.add_argument("--cue", action="append", default=[])
    recall.add_argument("--scope", default="{}")
    recall.add_argument("--requester", default="cli")
    recall.set_defaults(func=_agentic_recall)

    steward = sub.add_parser("steward", help="run advisory memory maintenance")
    steward.add_argument("--scope", default="{}")
    steward.add_argument("--actor", default="cli")
    steward.set_defaults(func=_steward)

    distill = sub.add_parser("distill", help="create a provenance-linked distillation")
    distill.add_argument("--target-ref", action="append", required=True)
    distill.add_argument("--summary", required=True, help="plain text, JSON, or @file")
    distill.add_argument("--scope", default="{}")
    distill.add_argument("--actor", default="cli")
    distill.add_argument("--idempotency-key")
    distill.add_argument("--distillation-type", default="summary")
    distill.add_argument("--archive-sources", action="store_true")
    distill.add_argument("--approved-by")
    distill.set_defaults(func=_distill)

    merge = sub.add_parser("merge-atoms", help="merge atoms after explicit review")
    merge.add_argument("--source-ref", action="append", required=True)
    merge.add_argument("--payload", required=True, help="merged payload JSON or @file")
    merge.add_argument("--type", dest="merged_type", default="semantic")
    merge.add_argument("--scope", default="{}")
    merge.add_argument("--actor", default="cli")
    merge.add_argument("--approved-by")
    merge.set_defaults(func=_merge_atoms)

    maintenance = sub.add_parser("maintenance", help="request a maintenance action")
    maintenance.add_argument("--action", required=True)
    maintenance.add_argument("--target-ref", action="append", default=[])
    maintenance.add_argument("--risk", default="low")
    maintenance.add_argument("--approved-by")
    maintenance.add_argument("--scope", default="{}")
    maintenance.add_argument("--actor", default="cli")
    maintenance.set_defaults(func=_maintenance)

    memory_policy = sub.add_parser("memory-policy", help="show or configure automatic memory policy")
    memory_policy.add_argument("--configure", action="store_true")
    memory_policy.add_argument("--enabled", choices=["true", "false"])
    memory_policy.add_argument("--schedule", default="{}")
    memory_policy.add_argument("--maintenance", default="{}")
    memory_policy.add_argument("--distillation", default="{}")
    memory_policy.add_argument("--maintenance-distiller", default="{}")
    memory_policy.add_argument("--decay", default="{}")
    memory_policy.add_argument("--storage-cleanup", default="{}")
    memory_policy.add_argument("--run", action="store_true")
    memory_policy.add_argument("--force", action="store_true")
    memory_policy.add_argument("--trigger", default="cli")
    memory_policy.add_argument("--scope", default="{}")
    memory_policy.set_defaults(func=_memory_policy)

    maintenance_distiller = sub.add_parser(
        "maintenance-distiller",
        help="run generic SMP processor-pack distillation",
    )
    maintenance_distiller.add_argument("--scope", default="{}")
    maintenance_distiller.add_argument("--domain", default="generic")
    maintenance_distiller.add_argument("--processor-id", action="append", default=[])
    maintenance_distiller.add_argument("--max-atoms", type=int, default=128)
    maintenance_distiller.add_argument("--max-events", type=int, default=64)
    maintenance_distiller.add_argument("--max-retrieval-outcomes", type=int, default=64)
    maintenance_distiller.add_argument("--no-auto-commit", action="store_true")
    maintenance_distiller.add_argument("--reviewer", default="{}")
    maintenance_distiller.add_argument("--actor", default="cli")
    maintenance_distiller.set_defaults(func=_maintenance_distiller)

    maintenance_processors = sub.add_parser(
        "maintenance-processors",
        help="list registered maintenance processors",
    )
    maintenance_processors.set_defaults(
        func=lambda amos, _args: amos.list_maintenance_processors()
    )

    capacity = sub.add_parser("configure-capacity", help="configure capacity pressure budget")
    capacity.add_argument("--hard-capacity-bytes", type=int, required=True)
    capacity.add_argument("--warning-ratio", type=float, default=0.70)
    capacity.add_argument("--critical-ratio", type=float, default=0.90)
    capacity.set_defaults(func=_configure_capacity)

    smp = sub.add_parser("smp-analysis", help="run deterministic SMP analysis")
    smp.add_argument("--scope", default="{}")
    smp.add_argument("--target-ref", action="append", default=[])
    smp.set_defaults(func=_smp_analysis)

    verify = sub.add_parser("verify", help="verify journal chain and replay projection")
    verify.set_defaults(
        func=lambda amos, _args: {
            "journal": amos.verify_journal_chain(),
            "replay": amos.verify_replay(),
        }
    )

    health = sub.add_parser("health", help="show memory and capacity health")
    health.set_defaults(
        func=lambda amos, _args: {
            "memory": amos.health_memory(),
            "capacity": amos.health_capacity(),
        }
    )

    http = sub.add_parser("serve", help="serve the AMOS v1 HTTP API")
    http.add_argument("--host", default="127.0.0.1")
    http.add_argument("--port", type=int, default=8765)
    http.add_argument(
        "--governance-principals",
        default="{}",
        help=(
            "JSON token-to-principal map or @file; identity and cognitive "
            "governance capabilities are never accepted from request JSON"
        ),
    )
    http.set_defaults(func=_serve)
    return parser


def _capture_event(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.capture_event(
        source_type=args.source_type,
        source_ref=args.source_ref,
        payload=parse_json_arg(args.payload),
        scope=parse_json_arg(args.scope),
        actor=args.actor,
        idempotency_key=args.idempotency_key,
    )


def _commit_atom(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    if args.atom:
        atom = parse_json_arg(args.atom)
    else:
        if not args.atom_type or not args.payload:
            raise SystemExit("commit-atom requires --atom or both --type and --payload")
        atom = {
            "type": args.atom_type,
            "payload": parse_json_arg(args.payload),
            "scope": parse_json_arg(args.scope),
        }
    return amos.commit_atom(
        atom,
        actor=args.actor,
        idempotency_key=args.idempotency_key,
        authorization_context=parse_json_arg(args.authorization_context),
    )


def _retrieve(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.retrieve_packet(
        cues=args.cue,
        scope=parse_json_arg(args.scope),
        requester=args.requester,
        target_processor=args.target_processor,
        memory_mode=args.memory_mode,
        max_items=args.max_items,
        include_conflicts=args.include_conflicts,
        include_archived=args.include_archived,
        attention_context=parse_json_arg(args.attention_context),
    )


def _ratify_proposal(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.ratify_proposal(
        proposal_ref=args.proposal_ref,
        adjudication_ref=args.adjudication_ref,
        expected_version=args.expected_version,
        actor=args.actor,
        authorization_context=parse_json_arg(args.authorization_context),
        idempotency_key=args.idempotency_key,
    )


def _resolve_proposal(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.resolve_proposal(
        proposal_ref=args.proposal_ref,
        adjudication_ref=args.adjudication_ref,
        expected_version=args.expected_version,
        actor=args.actor,
        authorization_context=parse_json_arg(args.authorization_context),
        idempotency_key=args.idempotency_key,
    )


def _replace_constitutional_record(
    amos: Amos, args: argparse.Namespace
) -> dict[str, Any]:
    return amos.replace_constitutional_record(
        current_ref=args.current_ref,
        successor_ref=args.successor_ref,
        adjudication_ref=args.adjudication_ref,
        expected_current_version=args.expected_current_version,
        expected_successor_version=args.expected_successor_version,
        actor=args.actor,
        authorization_context=parse_json_arg(args.authorization_context),
        idempotency_key=args.idempotency_key,
    )


def _provenance_analysis(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.analyze_provenance(
        atom_ref=args.atom_ref,
        max_depth=args.max_depth,
    )


def _diachronic_status(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.diachronic_ratification_status(
        subject_ref=args.subject_ref,
        identity_ref=args.identity_ref,
        required_confirmations=args.required_confirmations,
        min_interval_seconds=args.min_interval_seconds,
    )


def _self_awareness(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.retrieve_self_awareness(
        agent_id=args.agent_id,
        scope=parse_json_arg(args.scope),
        requester=args.requester,
    )


def _agentic_recall(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.retrieve_agentic_recall(
        agent_id=args.agent_id,
        cues=args.cue,
        scope=parse_json_arg(args.scope),
        requester=args.requester,
    )


def _steward(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.run_steward(scope=parse_json_arg(args.scope), actor=args.actor)


def _distill(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    try:
        summary = parse_json_arg(args.summary)
    except json.JSONDecodeError:
        summary = args.summary
    return amos.distill_memories(
        target_refs=args.target_ref,
        summary=summary,
        scope=parse_json_arg(args.scope),
        actor=args.actor,
        idempotency_key=args.idempotency_key,
        distillation_type=args.distillation_type,
        archive_sources=args.archive_sources,
        approved_by=args.approved_by,
    )


def _merge_atoms(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.merge_atoms(
        source_refs=args.source_ref,
        merged_payload=parse_json_arg(args.payload),
        merged_type=args.merged_type,
        scope=parse_json_arg(args.scope),
        actor=args.actor,
        approved_by=args.approved_by,
    )


def _maintenance(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.request_maintenance(
        action=args.action,
        target_refs=args.target_ref,
        risk=args.risk,
        approved_by=args.approved_by,
        scope=parse_json_arg(args.scope),
        actor=args.actor,
    )


def _memory_policy(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    if args.configure:
        return amos.configure_memory_policy(
            enabled=None if args.enabled is None else args.enabled == "true",
            schedule=parse_json_arg(args.schedule),
            maintenance=parse_json_arg(args.maintenance),
            distillation=parse_json_arg(args.distillation),
            maintenance_distiller=parse_json_arg(args.maintenance_distiller),
            decay=parse_json_arg(args.decay),
            storage_cleanup=parse_json_arg(args.storage_cleanup),
        )
    if args.run:
        return amos.run_memory_policy(
            force=args.force,
            trigger=args.trigger,
            scope=parse_json_arg(args.scope),
            actor="cli",
        )
    return amos.memory_policy_status()


def _maintenance_distiller(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.run_maintenance_distiller(
        scope=parse_json_arg(args.scope),
        actor=args.actor,
        domain=args.domain,
        processor_ids=args.processor_id,
        max_atoms=args.max_atoms,
        max_events=args.max_events,
        max_retrieval_outcomes=args.max_retrieval_outcomes,
        auto_commit_low_risk=not args.no_auto_commit,
        reviewer=parse_json_arg(args.reviewer),
    )


def _configure_capacity(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.configure_capacity_budget(
        hard_capacity_bytes=args.hard_capacity_bytes,
        warning_ratio=args.warning_ratio,
        critical_ratio=args.critical_ratio,
    )


def _smp_analysis(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    return amos.run_smp_analysis(
        scope=parse_json_arg(args.scope),
        target_refs=args.target_ref,
    )


def _serve(amos: Amos, args: argparse.Namespace) -> dict[str, Any]:
    db_path = str(amos.store.path)
    amos.close()
    serve(
        args.host,
        args.port,
        db_path,
        maintenance_processor_paths=args.maintenance_processor,
        governance_principals=parse_json_arg(args.governance_principals),
    )
    return {"status": "stopped"}


if __name__ == "__main__":
    raise SystemExit(main())
