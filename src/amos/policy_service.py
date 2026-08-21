"""PolicyService implementation for the AMOS service facade."""

from ._service_support import (
    Any,
    DEFAULT_MEMORY_POLICY,
    Mapping,
    Sequence,
    ValidationError,
    canonical_json,
    digest,
    json,
    maintenance_scope_visible,
    math,
    normalize_atom,
    scope_visible,
    stable_id,
    threading,
    utc_now,
)
from .maintenance import covered_source_refs, maintenance_hints_from_atom
from .schemas import CONSTITUTIONAL_ATOM_TYPES


GOVERNANCE_MAINTENANCE_PROTECTED_TYPES = {
    *CONSTITUTIONAL_ATOM_TYPES,
    "adjudication",
    "commitment",
    "self_model",
}

LEGACY_PROPOSAL_RETENTION_SECONDS = {
    "cogito.memory-recall-strategy.v1": 6 * 60 * 60,
}


class PolicyService:
    def __init__(
        self,
        store: Any,
        smp: Any,
        mutations: Any,
        indexes: Any,
        graph: Any,
        capacity: Any,
        temporal: Any,
        stewardship: Any,
    ):
        self.store = store
        self.smp = smp
        self._memory_policy_lock = getattr(
            store, "memory_policy_lock", threading.Lock()
        )
        self._memory_policy_running = False
        self.distill_memories = mutations.distill_memories
        self._attach_search_index = indexes._attach_search_index
        self._rebuild_derived_indexes = indexes._rebuild_derived_indexes
        self._invalidate_packet_cache = indexes._invalidate_packet_cache
        self._active_superseded_refs = graph._active_superseded_refs
        self._memory_identity_digest = graph._memory_identity_digest
        self._atom_projection = graph._atom_projection
        self._render_atom = graph._render_atom
        self._counts = graph._counts
        self._capacity_pressure_mode = capacity._capacity_pressure_mode
        self._seconds_since = temporal._seconds_since
        self._iso_before_seconds = temporal._iso_before_seconds
        self._timestamp_elapsed = temporal._timestamp_elapsed
        self.run_smp_analysis = stewardship.run_smp_analysis
        self.run_steward = stewardship.run_steward
        self.run_maintenance_distiller = stewardship.run_maintenance_distiller

    def configure_memory_policy(
        self,
        *,
        enabled: bool | None = None,
        schedule: Mapping[str, Any] | None = None,
        maintenance: Mapping[str, Any] | None = None,
        distillation: Mapping[str, Any] | None = None,
        maintenance_distiller: Mapping[str, Any] | None = None,
        decay: Mapping[str, Any] | None = None,
        storage_cleanup: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy = self.memory_policy()
        if enabled is not None:
            policy["enabled"] = bool(enabled)
        if schedule is not None:
            policy["schedule"] = {**policy["schedule"], **dict(schedule)}
        if maintenance is not None:
            policy["maintenance"] = {**policy["maintenance"], **dict(maintenance)}
        if distillation is not None:
            policy["distillation"] = {
                **policy["distillation"],
                **dict(distillation),
            }
        if maintenance_distiller is not None:
            policy["maintenance_distiller"] = {
                **policy["maintenance_distiller"],
                **dict(maintenance_distiller),
            }
        if decay is not None:
            decay_update = dict(decay)
            if "max_atoms" in decay_update:
                decay_update.setdefault("max_active_atoms", decay_update["max_atoms"])
                decay_update.setdefault("max_proposed_atoms", decay_update["max_atoms"])
            policy["decay"] = {**policy["decay"], **decay_update}
        if storage_cleanup is not None:
            cleanup = dict(policy["storage_cleanup"])
            for key, value in dict(storage_cleanup).items():
                if (
                    key in {"sqlite_compaction", "journal_compaction"}
                    and isinstance(value, Mapping)
                    and isinstance(cleanup.get(key), Mapping)
                ):
                    cleanup[key] = {
                        **dict(cleanup[key]),
                        **dict(value),
                    }
                else:
                    cleanup[key] = value
            policy["storage_cleanup"] = cleanup
        policy = self._normalize_memory_policy(policy)
        self.store.set_meta("memory_policy", canonical_json(policy))
        return {
            "status": "configured",
            "policy": policy,
            "memory_policy": self.memory_policy_status(policy=policy),
        }


    def memory_policy(self) -> dict[str, Any]:
        raw = self.store.get_meta("memory_policy")
        if not raw:
            return self._normalize_memory_policy(DEFAULT_MEMORY_POLICY)
        try:
            configured = json.loads(raw)
        except json.JSONDecodeError:
            configured = {}
        return self._normalize_memory_policy(configured)


    def memory_policy_status(
        self, *, policy: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        current_policy = self._normalize_memory_policy(policy or self.memory_policy())
        state = self._memory_policy_state()
        due = self._memory_policy_due(current_policy, state)
        return {
            "policy": current_policy,
            "state": state,
            "due": due,
            "graph_version": self.store.graph_version(),
        }


    def run_memory_policy(
        self,
        *,
        force: bool = False,
        trigger: str = "scheduler",
        scope: Mapping[str, Any] | None = None,
        actor: str = "svc:memory_policy",
    ) -> dict[str, Any]:
        policy = self.memory_policy()
        state = self._memory_policy_state()
        due = self._memory_policy_due(policy, state, force=force)
        if not due["due"]:
            return {
                "status": "skipped",
                "reason": "not_due",
                "trigger": trigger,
                "due": due,
                "graph_version": self.store.graph_version(),
            }
        if not policy["enabled"] and not force:
            return {
                "status": "skipped",
                "reason": "policy_disabled",
                "trigger": trigger,
                "due": due,
                "graph_version": self.store.graph_version(),
            }

        if not self._memory_policy_lock.acquire(blocking=False):
            return {
                "status": "skipped",
                "reason": "memory_policy_already_running",
                "trigger": trigger,
                "graph_version": self.store.graph_version(),
            }

        previous_write_lane = self.store.set_write_lane("maintenance")
        self._memory_policy_running = True
        try:
            # The due check above is an optimistic fast path. Another service
            # connection may finish a policy pass between that check and this
            # shared single-flight admission, so refresh the policy clock once
            # admitted and avoid a duplicate maintenance pass.
            policy = self.memory_policy()
            state = self._memory_policy_state()
            due = self._memory_policy_due(policy, state, force=force)
            if not due["due"]:
                return {
                    "status": "skipped",
                    "reason": "not_due",
                    "trigger": trigger,
                    "due": due,
                    "graph_version": self.store.graph_version(),
                }
            if not policy["enabled"] and not force:
                return {
                    "status": "skipped",
                    "reason": "policy_disabled",
                    "trigger": trigger,
                    "due": due,
                    "graph_version": self.store.graph_version(),
                }
            started_graph_version = self.store.graph_version()
            scope = dict(scope or {})
            results: dict[str, Any] = {}
            target_refs: list[str] = []
            maintenance = policy["maintenance"]
            if maintenance["enabled"] and maintenance["repair_reference_contracts"]:
                results["reference_contracts"] = (
                    self._run_reference_contract_repairs(
                        scope=scope,
                        actor=actor,
                        max_repairs=maintenance["max_reference_contract_repairs"],
                    )
                )
                target_refs.extend(
                    action["record_ref"]
                    for action in results["reference_contracts"].get("actions", [])
                    if action.get("record_kind") == "atom"
                )
            if maintenance["enabled"] and maintenance["run_smp"]:
                results["smp"] = self.run_smp_analysis(
                    scope=scope,
                    max_atoms=maintenance["max_smp_atoms"],
                )
            if maintenance["enabled"] and maintenance["run_steward"]:
                results["steward"] = self.run_steward(
                    scope=scope,
                    actor=actor,
                    max_atoms=maintenance["max_steward_atoms"],
                    max_edge_mutations=maintenance[
                        "max_steward_edge_mutations"
                    ],
                    start_after=str(state.get("steward_cursor") or "") or None,
                )
                for action in results["steward"].get("actions", []):
                    target_refs.extend(
                        ref
                        for ref in action.get("atom_refs", [])
                        + [action.get("kept"), action.get("archived")]
                        if ref
                    )

            if policy["distillation"]["enabled"]:
                results["distillation"] = self._run_policy_distillation(
                    policy=policy,
                    scope=scope,
                    actor=actor,
                )
                distilled = results["distillation"].get("distilled")
                if distilled:
                    target_refs.append(distilled["atom"]["id"])
                    target_refs.extend(distilled["source_refs"])

            maintenance_distiller = policy["maintenance_distiller"]
            if maintenance_distiller["enabled"]:
                results["maintenance_distiller"] = self.run_maintenance_distiller(
                    scope=scope,
                    actor=actor,
                    domain=maintenance_distiller["domain"],
                    processor_ids=maintenance_distiller["processor_ids"],
                    max_atoms=maintenance_distiller["max_atoms"],
                    max_events=maintenance_distiller["max_events"],
                    max_retrieval_outcomes=maintenance_distiller[
                        "max_retrieval_outcomes"
                    ],
                    auto_commit_low_risk=maintenance_distiller[
                        "auto_commit_low_risk"
                    ],
                    reviewer=maintenance_distiller["reviewer"],
                )
                for committed in results["maintenance_distiller"].get(
                    "committed", []
                ):
                    atom = committed.get("atom")
                    if atom:
                        target_refs.append(atom["id"])
                    target_refs.extend(committed.get("source_refs", []))

            # Distillation may commit an active successor. Apply lifecycle
            # policy after those commits so its superseded predecessor is
            # archived in this same policy pass rather than remaining an
            # active graph-quality warning until the next interval.
            decay = policy["decay"]
            if decay["enabled"]:
                results["decay"] = self._run_decay_policy(
                    decay=decay,
                    scope=scope,
                    actor=actor,
                )
                target_refs.extend(
                    action["atom_ref"]
                    for action in results["decay"].get("actions", [])
                    if action.get("atom_ref")
                )

            storage_cleanup = policy["storage_cleanup"]
            if storage_cleanup["enabled"] and due.get("storage_cleanup", {}).get("due"):
                results["storage_cleanup"] = self._run_storage_cleanup(
                    cleanup=storage_cleanup,
                    due=due["storage_cleanup"],
                    scope=scope,
                    actor=actor,
                    state=state,
                    force=force,
                )
                target_refs.extend(results["storage_cleanup"].get("deleted_atom_refs", []))

            policy_event_graph_version = self.store.graph_version() + 1
            if maintenance["enabled"] and maintenance["rebuild_indexes"]:
                results["index"] = self._rebuild_derived_indexes(
                    graph_version=policy_event_graph_version,
                    publish_revision_offset=1,
                )
            if maintenance["enabled"] and maintenance["invalidate_packet_cache"]:
                results["packet_cache"] = self._invalidate_packet_cache(
                    graph_version=policy_event_graph_version
                )

            completed_at = utc_now()
            with self.store.transaction() as conn:
                # Foreground commits may run between bounded maintenance
                # phases. Derive the final event revision only after this
                # transaction owns the writer so journal metadata never
                # reports the earlier optimistic prediction as completed.
                completed_graph_version = self.store.graph_version() + 1
                if isinstance(results.get("index"), Mapping):
                    index_result = dict(results["index"])
                    if int(index_result.get("graph_version", -1)) != int(
                        completed_graph_version
                    ):
                        index_result["status"] = "stale"
                        index_result["reason"] = (
                            "canonical_revision_advanced_after_index_publish"
                        )
                        index_result["indexes"] = [
                            {**dict(index), "freshness": "stale"}
                            for index in index_result.get("indexes", [])
                            if isinstance(index, Mapping)
                        ]
                    results["index"] = index_result
                if isinstance(results.get("packet_cache"), Mapping):
                    results["packet_cache"] = {
                        **dict(results["packet_cache"]),
                        "graph_version": completed_graph_version,
                    }
                event_payload = {
                    "operation": "run_memory_policy",
                    "trigger": trigger,
                    "force": force,
                    "due": due,
                    "policy": policy,
                    "started_graph_version": started_graph_version,
                    "completed_graph_version": completed_graph_version,
                    "results": self._memory_policy_journal_results(results),
                }
                event = self.store.append_event(
                    conn,
                    event_type="memory_policy_run",
                    actor=actor,
                    payload=event_payload,
                    target_refs=sorted(set(target_refs)),
                )
                steward_cursor = state.get("steward_cursor")
                if results.get("steward", {}).get("status") == "completed":
                    steward_cursor = (
                        results["steward"].get("window", {}).get("next_cursor")
                    )
                self.store._set_meta(
                    conn,
                    "memory_policy_state",
                    canonical_json(
                        {
                            "last_run_at": completed_at,
                            "last_graph_version": event["graph_version"],
                            "last_trigger": trigger,
                            "last_event_id": event["event_id"],
                            "last_due_reasons": due["reasons"],
                            "last_distilled_refs": [
                                results.get("distillation", {})
                                .get("distilled", {})
                                .get("atom", {})
                                .get("id")
                            ]
                            if results.get("distillation", {}).get("distilled")
                            else [],
                            "last_maintenance_distiller_refs": [
                                committed["atom"]["id"]
                                for committed in results.get(
                                    "maintenance_distiller", {}
                                ).get("committed", [])
                                if committed.get("atom")
                            ],
                            "steward_cursor": steward_cursor,
                            "last_storage_cleanup_at": self.store.get_meta(
                                "last_storage_cleanup_at"
                            ),
                            "last_vacuum_at": self.store.get_meta("last_vacuum_at"),
                            "last_foreground_activity_at": self.store.get_meta(
                                "last_foreground_activity_at"
                            ),
                        }
                    ),
                )
                if maintenance["enabled"] and maintenance["invalidate_packet_cache"]:
                    self.store.retire_packet_cache(conn)
            return {
                "status": "completed",
                "trigger": trigger,
                "due": due,
                "policy": policy,
                "results": results,
                "event": event,
                "graph_version": self.store.graph_version(),
            }
        finally:
            self._memory_policy_running = False
            self.store.set_write_lane(previous_write_lane)
            self._memory_policy_lock.release()


    def _normalize_memory_policy(self, policy: Mapping[str, Any]) -> dict[str, Any]:
        normalized = json.loads(canonical_json(DEFAULT_MEMORY_POLICY))
        policy = dict(policy or {})
        for key, value in policy.items():
            if key in {
                "schedule",
                "maintenance",
                "distillation",
                "maintenance_distiller",
                "decay",
                "storage_cleanup",
            } and isinstance(value, Mapping):
                if key == "storage_cleanup":
                    cleanup = dict(normalized[key])
                    for cleanup_key, cleanup_value in dict(value).items():
                        if (
                            cleanup_key
                            in {"sqlite_compaction", "journal_compaction"}
                            and isinstance(cleanup_value, Mapping)
                            and isinstance(cleanup.get(cleanup_key), Mapping)
                        ):
                            cleanup[cleanup_key] = {
                                **dict(cleanup[cleanup_key]),
                                **dict(cleanup_value),
                            }
                        else:
                            cleanup[cleanup_key] = cleanup_value
                    normalized[key] = cleanup
                else:
                    normalized[key].update(dict(value))
            else:
                normalized[key] = value
        normalized["enabled"] = bool(normalized.get("enabled", True))
        schedule = normalized["schedule"]
        schedule["every_graph_versions"] = max(
            1, int(schedule.get("every_graph_versions", 25) or 25)
        )
        schedule["every_seconds"] = max(
            0, int(schedule.get("every_seconds", 300) or 0)
        )
        schedule["run_on_pressure"] = bool(schedule.get("run_on_pressure", True))
        schedule["pressure_min_interval_seconds"] = max(
            0,
            int(schedule.get("pressure_min_interval_seconds", 300) or 0),
        )
        maintenance = normalized["maintenance"]
        for key in [
            "enabled",
            "repair_reference_contracts",
            "run_smp",
            "run_steward",
            "rebuild_indexes",
            "rebuild_lsa",
            "invalidate_packet_cache",
        ]:
            maintenance[key] = bool(maintenance.get(key, True))
        maintenance["max_reference_contract_repairs"] = max(
            1,
            int(maintenance.get("max_reference_contract_repairs", 2048) or 2048),
        )
        maintenance["max_smp_atoms"] = max(
            1,
            int(maintenance.get("max_smp_atoms", 128) or 128),
        )
        maintenance["max_steward_atoms"] = max(
            1,
            int(maintenance.get("max_steward_atoms", 128) or 128),
        )
        maintenance["max_steward_edge_mutations"] = max(
            1,
            int(
                maintenance.get("max_steward_edge_mutations", 128) or 128
            ),
        )
        maintenance["lsa_dimensions"] = max(
            0,
            min(
                self.smp.dimensions,
                int(maintenance.get("lsa_dimensions", 32) or 0),
            ),
        )
        maintenance["lsa_max_terms"] = max(
            maintenance["lsa_dimensions"],
            int(maintenance.get("lsa_max_terms", 300) or 300),
        )
        maintenance["index_write_batch_size"] = max(
            1,
            int(maintenance.get("index_write_batch_size", 64) or 64),
        )
        distillation = normalized["distillation"]
        distillation["enabled"] = bool(distillation.get("enabled", True))
        distillation["min_source_atoms"] = max(
            2, int(distillation.get("min_source_atoms", 6) or 6)
        )
        distillation["max_source_atoms"] = max(
            distillation["min_source_atoms"],
            int(distillation.get("max_source_atoms", 10) or 10),
        )
        distillation["candidate_types"] = [
            str(item) for item in distillation.get("candidate_types", [])
        ]
        distillation["archive_sources"] = bool(distillation.get("archive_sources", False))
        approved_by = distillation.get("approved_by")
        distillation["approved_by"] = str(approved_by) if approved_by else None
        distillation["distillation_type"] = str(
            distillation.get("distillation_type") or "automatic_policy"
        )
        distiller = normalized["maintenance_distiller"]
        distiller["enabled"] = bool(distiller.get("enabled", True))
        distiller["auto_commit_low_risk"] = bool(
            distiller.get("auto_commit_low_risk", True)
        )
        distiller["processor_ids"] = [
            str(item) for item in distiller.get("processor_ids", [])
        ]
        distiller["domain"] = str(distiller.get("domain") or "generic")
        distiller["max_atoms"] = max(1, int(distiller.get("max_atoms", 128) or 128))
        distiller["max_events"] = max(0, int(distiller.get("max_events", 64) or 0))
        distiller["max_retrieval_outcomes"] = max(
            0,
            int(distiller.get("max_retrieval_outcomes", 64) or 0),
        )
        reviewer = dict(distiller.get("reviewer") or {})
        distiller["reviewer"] = {
            "enabled": bool(reviewer.get("enabled", False)),
            "authority": "draft_only",
        }
        decay = normalized["decay"]
        decay["enabled"] = bool(decay.get("enabled", True))
        decay["max_atoms"] = max(1, int(decay.get("max_atoms", 256) or 256))
        decay["write_batch_size"] = max(
            1, int(decay.get("write_batch_size", 32) or 32)
        )
        decay["max_active_atoms"] = max(
            1,
            int(decay.get("max_active_atoms", decay["max_atoms"]) or decay["max_atoms"]),
        )
        decay["max_proposed_atoms"] = max(
            1,
            int(decay.get("max_proposed_atoms", decay["max_atoms"]) or decay["max_atoms"]),
        )
        decay["require_atom_policy"] = bool(decay.get("require_atom_policy", True))
        decay["pressure_archive_policyless"] = bool(
            decay.get("pressure_archive_policyless", True)
        )
        decay["pressure_archive_proposed"] = bool(
            decay.get("pressure_archive_proposed", True)
        )
        decay["proposal_pressure_min_age_seconds"] = max(
            0, int(decay.get("proposal_pressure_min_age_seconds", 3600) or 0)
        )
        proposed_ttl = decay.get("archive_proposed_after_seconds")
        decay["archive_proposed_after_seconds"] = (
            None if proposed_ttl in (None, "") else max(0, int(proposed_ttl))
        )
        decay["pressure_max_archives_per_run"] = max(
            1,
            int(decay.get("pressure_max_archives_per_run", 256) or 256),
        )
        decay["pressure_protected_types"] = sorted(
            {
                str(item)
                for item in decay.get(
                    "pressure_protected_types",
                    ["commitment", "policy", "self_model"],
                )
                if str(item)
            }
            | GOVERNANCE_MAINTENANCE_PROTECTED_TYPES
        )
        decay["capacity_assessment_targets"] = sorted(
            {
                max(1, int(item))
                for item in decay.get("capacity_assessment_targets", [256, 512, 768])
                if item not in (None, "")
            }
            | {decay["max_atoms"]}
        )
        decay["capacity_headroom_ratio"] = max(
            0.0,
            min(0.9, float(decay.get("capacity_headroom_ratio", 0.2) or 0.0)),
        )
        decay["archive_superseded"] = bool(decay.get("archive_superseded", True))
        value = decay.get("archive_superseded_after_seconds", 0)
        decay["archive_superseded_after_seconds"] = (
            None if value in (None, "") else max(0, int(value))
        )
        for key in (
            "mark_stale_after_seconds",
            "archive_after_seconds",
            "low_utility_threshold",
        ):
            value = decay.get(key)
            if value in (None, ""):
                decay[key] = None
            elif key == "low_utility_threshold":
                decay[key] = max(0.0, min(1.0, float(value)))
            else:
                decay[key] = max(0, int(value))
        cleanup = normalized["storage_cleanup"]
        cleanup["enabled"] = bool(cleanup.get("enabled", True))
        cleanup["trigger"] = str(
            cleanup.get("trigger") or "idle_or_pressure"
        )
        if cleanup["trigger"] not in {"idle", "idle_or_pressure"}:
            cleanup["trigger"] = "idle_or_pressure"
        cleanup["run_on_pressure"] = bool(
            cleanup.get("run_on_pressure", True)
        )
        cleanup["pressure_modes"] = sorted(
            {
                str(mode)
                for mode in cleanup.get("pressure_modes", ["orange", "red"])
                if str(mode) in {"orange", "red"}
            }
        ) or ["orange", "red"]
        for key, default in (
            ("idle_after_seconds", 300),
            ("min_interval_seconds", 900),
            ("pressure_min_interval_seconds", 300),
            ("max_deletions_per_tick", 256),
            ("max_idempotency_compactions_per_tick", 512),
            ("write_batch_size", 32),
            ("max_index_prune_atoms_per_tick", 512),
        ):
            cleanup[key] = max(0, int(cleanup.get(key, default) or 0))
        cleanup["write_batch_size"] = max(1, cleanup["write_batch_size"])
        for key, default in (
            ("delete_archived_after_seconds", 604800),
            ("delete_stale_after_seconds", 1209600),
            ("delete_superseded_after_seconds", 604800),
            ("purge_deleted_after_seconds", 3600),
            ("pressure_delete_archived_after_seconds", 86400),
            ("pressure_delete_stale_after_seconds", 604800),
            ("pressure_delete_superseded_after_seconds", 86400),
            ("pressure_purge_deleted_after_seconds", 0),
            ("compact_idempotency_after_seconds", 3600),
            ("pressure_compact_idempotency_after_seconds", 300),
        ):
            value = cleanup.get(key, default)
            cleanup[key] = None if value in (None, "") else max(0, int(value))
        cleanup["remove_archived_from_hot_index"] = bool(
            cleanup.get("remove_archived_from_hot_index", True)
        )
        cleanup["remove_superseded_from_hot_index"] = bool(
            cleanup.get("remove_superseded_from_hot_index", True)
        )
        cleanup["remove_stale_from_hot_index"] = bool(
            cleanup.get("remove_stale_from_hot_index", True)
        )
        cleanup["protected_types"] = sorted(
            {str(item) for item in cleanup.get("protected_types", [])}
            | GOVERNANCE_MAINTENANCE_PROTECTED_TYPES
        )
        journal_compaction = dict(cleanup.get("journal_compaction") or {})
        max_events_per_segment = max(
            1,
            int(journal_compaction.get("max_events_per_segment", 512) or 512),
        )
        cleanup["journal_compaction"] = {
            "enabled": bool(journal_compaction.get("enabled", True)),
            "max_events_per_segment": max_events_per_segment,
            "min_events_per_segment": max(
                1,
                min(
                    max_events_per_segment,
                    int(
                        journal_compaction.get("min_events_per_segment", 128)
                        or 128
                    ),
                ),
            ),
            "pressure_min_events_per_segment": max(
                1,
                min(
                    max_events_per_segment,
                    int(
                        journal_compaction.get(
                            "pressure_min_events_per_segment", 64
                        )
                        or 64
                    ),
                ),
            ),
            "retain_tail_events": max(
                0,
                int(journal_compaction.get("retain_tail_events", 128) or 0),
            ),
            "retain_snapshots": max(
                1,
                int(journal_compaction.get("retain_snapshots", 1) or 1),
            ),
            "retain_full_segments": max(
                0,
                int(journal_compaction.get("retain_full_segments", 2) or 0),
            ),
        }
        sqlite_compaction = dict(cleanup.get("sqlite_compaction") or {})
        checkpoint_mode = str(sqlite_compaction.get("checkpoint_mode") or "PASSIVE").upper()
        if checkpoint_mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            checkpoint_mode = "PASSIVE"
        pressure_checkpoint_mode = str(
            sqlite_compaction.get("pressure_checkpoint_mode") or "TRUNCATE"
        ).upper()
        if pressure_checkpoint_mode not in {
            "PASSIVE",
            "FULL",
            "RESTART",
            "TRUNCATE",
        }:
            pressure_checkpoint_mode = "TRUNCATE"
        cleanup["sqlite_compaction"] = {
            "checkpoint_wal": bool(sqlite_compaction.get("checkpoint_wal", True)),
            "checkpoint_mode": checkpoint_mode,
            "pressure_checkpoint_mode": pressure_checkpoint_mode,
            "incremental_vacuum": bool(
                sqlite_compaction.get("incremental_vacuum", True)
            ),
            "incremental_vacuum_pages": max(
                0,
                int(
                    sqlite_compaction.get("incremental_vacuum_pages", 4096)
                    or 0
                ),
            ),
            "vacuum_enabled": bool(sqlite_compaction.get("vacuum_enabled", False)),
            "vacuum_idle_after_seconds": max(
                0, int(sqlite_compaction.get("vacuum_idle_after_seconds", 1800) or 0)
            ),
            "vacuum_min_interval_seconds": max(
                0,
                int(sqlite_compaction.get("vacuum_min_interval_seconds", 86400) or 0),
            ),
        }
        return normalized


    def _memory_policy_state(self) -> dict[str, Any]:
        raw = self.store.get_meta("memory_policy_state")
        if not raw:
            return {
                "last_run_at": None,
                "last_graph_version": 0,
                "last_trigger": None,
                "last_event_id": None,
                "last_due_reasons": [],
                "last_distilled_refs": [],
                "last_maintenance_distiller_refs": [],
                "steward_cursor": None,
                "last_storage_cleanup_at": self.store.get_meta(
                    "last_storage_cleanup_at"
                ),
                "last_vacuum_at": self.store.get_meta("last_vacuum_at"),
                "last_foreground_activity_at": self.store.get_meta(
                    "last_foreground_activity_at"
                ),
            }
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return {
            "last_run_at": data.get("last_run_at"),
            "last_graph_version": int(data.get("last_graph_version", 0) or 0),
            "last_trigger": data.get("last_trigger"),
            "last_event_id": data.get("last_event_id"),
            "last_due_reasons": list(data.get("last_due_reasons", [])),
            "last_distilled_refs": list(data.get("last_distilled_refs", [])),
            "last_maintenance_distiller_refs": list(
                data.get("last_maintenance_distiller_refs", [])
            ),
            "steward_cursor": data.get("steward_cursor"),
            "last_storage_cleanup_at": self.store.get_meta("last_storage_cleanup_at")
            or data.get("last_storage_cleanup_at"),
            "last_vacuum_at": self.store.get_meta("last_vacuum_at")
            or data.get("last_vacuum_at"),
            "last_foreground_activity_at": self.store.get_meta(
                "last_foreground_activity_at"
            )
            or data.get("last_foreground_activity_at"),
        }


    def _memory_policy_journal_results(
        self, results: Mapping[str, Any]
    ) -> dict[str, Any]:
        journal: dict[str, Any] = {}
        for key, value in results.items():
            if key == "smp" and isinstance(value, Mapping):
                journal[key] = self._summarize_smp_result(value)
            elif key == "steward" and isinstance(value, Mapping):
                journal[key] = self._summarize_steward_result(value)
            elif key == "distillation" and isinstance(value, Mapping):
                journal[key] = self._summarize_policy_distillation_result(value)
            elif key == "maintenance_distiller" and isinstance(value, Mapping):
                journal[key] = self._summarize_maintenance_distiller_result(value)
            elif key == "storage_cleanup" and isinstance(value, Mapping):
                journal[key] = self._summarize_storage_cleanup_result(value)
            elif key in {"index", "packet_cache"} and isinstance(value, Mapping):
                journal[key] = dict(value)
            else:
                journal[key] = self._bounded_json_summary(value)
        return journal


    def _summarize_smp_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        outputs = [
            output
            for output in result.get("outputs", [])
            if isinstance(output, Mapping)
        ]
        review_required = [
            output
            for output in result.get("review_required", [])
            if isinstance(output, Mapping)
        ]
        return {
            "status": result.get("status"),
            "processor_id": result.get("processor_id"),
            "processor_version": result.get("processor_version"),
            "graph_version": result.get("graph_version"),
            "scope": dict(result.get("scope") or {}),
            "atom_count": result.get("atom_count", 0),
            "analyzed_atom_count": result.get("analyzed_atom_count", 0),
            "omitted_atom_count": result.get("omitted_atom_count", 0),
            "output_count": len(outputs),
            "review_required_count": len(review_required),
            "output_type_counts": self._count_mapping_values(outputs, "output_type"),
            "reason_code_counts": self._count_mapping_values(outputs, "reason_code"),
            "risk_level_counts": self._count_mapping_values(outputs, "risk_level"),
            "review_required_refs": self._bounded_refs(
                ref
                for output in review_required
                for ref in output.get("input_refs", [])
            ),
            "sample_output_ids": self._bounded_refs(
                output.get("output_id") for output in outputs
            ),
        }


    def _summarize_steward_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        actions = [
            action
            for action in result.get("actions", [])
            if isinstance(action, Mapping)
        ]
        event = result.get("event")
        event_ref = event.get("event_id") if isinstance(event, Mapping) else None
        return {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "graph_version": result.get("graph_version"),
            "window": dict(result.get("window") or {}),
            "action_count": len(actions),
            "action_counts": self._count_mapping_values(actions, "action"),
            "target_refs": self._bounded_refs(
                ref
                for action in actions
                for ref in [
                    *list(action.get("atom_refs", [])),
                    action.get("kept"),
                    action.get("archived"),
                ]
            ),
            "event_id": event_ref,
        }


    def _summarize_policy_distillation_result(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        distilled = result.get("distilled")
        distilled_atom = (
            distilled.get("atom")
            if isinstance(distilled, Mapping)
            and isinstance(distilled.get("atom"), Mapping)
            else None
        )
        return {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "candidate_count": result.get("candidate_count"),
            "min_source_atoms": result.get("min_source_atoms"),
            "source_refs": self._bounded_refs(result.get("source_refs", [])),
            "distilled_atom_ref": distilled_atom.get("id") if distilled_atom else None,
        }


    def _summarize_maintenance_distiller_result(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        committed = [
            item for item in result.get("committed", []) if isinstance(item, Mapping)
        ]
        deferred = [
            item for item in result.get("deferred", []) if isinstance(item, Mapping)
        ]
        proposals = [
            item for item in result.get("proposals", []) if isinstance(item, Mapping)
        ]
        event = result.get("event")
        event_ref = event.get("event_id") if isinstance(event, Mapping) else None
        return {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "scope": dict(result.get("scope") or {}),
            "domain": result.get("domain"),
            "graph_version": result.get("graph_version"),
            "window": dict(result.get("window") or {}),
            "processors": list(result.get("processors", [])),
            "missing_processors": list(result.get("missing_processors", [])),
            "proposal_count": len(proposals),
            "committed_count": len(committed),
            "deferred_count": len(deferred),
            "proposal_action_counts": self._count_mapping_values(proposals, "action"),
            "committed_status_counts": self._count_mapping_values(committed, "status"),
            "deferred_reason_counts": self._count_mapping_values(deferred, "reason"),
            "committed_refs": self._bounded_refs(
                item.get("atom", {}).get("id")
                if isinstance(item.get("atom"), Mapping)
                else item.get("edge", {}).get("edge_id")
                if isinstance(item.get("edge"), Mapping)
                else None
                for item in committed
            ),
            "deferred_proposal_ids": self._bounded_refs(
                item.get("proposal_id") for item in deferred
            ),
            "reviewer": dict(result.get("reviewer") or {}),
            "event_id": event_ref,
        }


    def _summarize_storage_cleanup_result(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        event = result.get("event")
        event_ref = event.get("event_id") if isinstance(event, Mapping) else None
        index_prune = dict(result.get("index_prune") or {})
        idempotency = dict(result.get("idempotency") or {})
        journal_compaction = dict(result.get("journal_compaction") or {})
        incremental_vacuum = dict(result.get("incremental_vacuum") or {})
        checkpoint = dict(result.get("checkpoint") or {})
        vacuum = dict(result.get("vacuum") or {})
        return {
            "status": result.get("status"),
            "deleted_atom_count": int(result.get("deleted_atom_count", 0) or 0),
            "physically_purged_atom_count": int(
                result.get("physically_purged_atom_count", 0) or 0
            ),
            "deleted_atom_refs": self._bounded_refs(result.get("deleted_atom_refs", [])),
            "index_pruned_rows": int(index_prune.get("rows", 0) or 0),
            "idempotency_compacted_rows": int(idempotency.get("rows", 0) or 0),
            "idempotency_saved_bytes": int(idempotency.get("saved_bytes", 0) or 0),
            "journal_compaction_status": journal_compaction.get("status"),
            "journal_compacted_events": int(
                journal_compaction.get("compacted_event_count", 0) or 0
            ),
            "journal_pruned_payload_bytes": int(
                journal_compaction.get("pruned_segment_payload_bytes", 0) or 0
            ),
            "incremental_vacuum_status": incremental_vacuum.get("status"),
            "incremental_vacuum_reclaimed_pages": int(
                incremental_vacuum.get("reclaimed_pages", 0) or 0
            ),
            "checkpoint_status": checkpoint.get("status"),
            "checkpoint_mode": checkpoint.get("mode"),
            "vacuum_status": vacuum.get("status"),
            "vacuum_reason": vacuum.get("reason"),
            "event_id": event_ref,
        }


    def _count_mapping_values(
        self, rows: Sequence[Mapping[str, Any]], key: str
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))


    def _bounded_refs(self, refs: Any, *, limit: int = 24) -> list[str]:
        output: list[str] = []
        for ref in refs or []:
            if ref in (None, "", [], {}):
                continue
            text = str(ref)
            if text in output:
                continue
            output.append(text)
            if len(output) >= limit:
                break
        return output


    def _bounded_json_summary(self, value: Any, *, max_bytes: int = 2048) -> Any:
        try:
            encoded = canonical_json(value)
        except Exception:
            return {"summary": str(value)[:max_bytes], "truncated": True}
        if len(encoded.encode("utf-8")) <= max_bytes:
            return value
        return {
            "summary_digest": digest(value),
            "summary_bytes": len(encoded.encode("utf-8")),
            "truncated": True,
        }


    def _memory_policy_due(
        self,
        policy: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        graph_version = self.store.graph_version()
        last_graph_version = int(state.get("last_graph_version", 0) or 0)
        graph_delta = max(0, graph_version - last_graph_version)
        schedule = dict(policy.get("schedule", {}))
        reasons = []
        if force:
            reasons.append("force")
        if graph_delta >= int(schedule.get("every_graph_versions", 25)):
            reasons.append("graph_version_interval")
        every_seconds = int(schedule.get("every_seconds", 300) or 0)
        elapsed_seconds = self._seconds_since(state.get("last_run_at"))
        if every_seconds > 0 and elapsed_seconds is not None and elapsed_seconds >= every_seconds:
            reasons.append("time_interval")
        pressure_min_interval_seconds = max(
            0,
            int(schedule.get("pressure_min_interval_seconds", 300) or 0),
        )
        pressure_interval_elapsed = (
            elapsed_seconds is None
            or elapsed_seconds >= pressure_min_interval_seconds
        )
        pressure_mode = self._capacity_pressure_mode()
        if (
            schedule.get("run_on_pressure", True)
            and pressure_interval_elapsed
            and pressure_mode in {"orange", "red"}
            and graph_delta > 0
        ):
            reasons.append(f"capacity_pressure:{pressure_mode}")
        if (
            schedule.get("run_on_pressure", True)
            and pressure_interval_elapsed
            and graph_delta > 0
        ):
            decay = dict(policy.get("decay") or {})
            max_atoms = max(1, int(decay.get("max_atoms", 256) or 256))
            max_active_atoms = max(
                1, int(decay.get("max_active_atoms", max_atoms) or max_atoms)
            )
            max_proposed_atoms = max(
                1, int(decay.get("max_proposed_atoms", max_atoms) or max_atoms)
            )
            hot_atoms = self.store.list_atoms_filtered(
                lifecycle_states=["active", "proposed"]
            )
            active_count = sum(
                atom.get("lifecycle_state") == "active" for atom in hot_atoms
            )
            proposed_count = len(hot_atoms) - active_count
            if len(hot_atoms) > max_atoms:
                reasons.append("memory_atom_pressure:hot")
            if active_count > max_active_atoms:
                reasons.append("memory_atom_pressure:active")
            if proposed_count > max_proposed_atoms:
                reasons.append("memory_atom_pressure:proposed")
        storage_cleanup = self._storage_cleanup_due(
            policy.get("storage_cleanup", {}),
            state,
            force=force,
            pressure_mode=pressure_mode,
        )
        if storage_cleanup["due"] and "force" not in reasons:
            reasons.append(
                "storage_cleanup_pressure"
                if storage_cleanup.get("pressure_triggered")
                else "storage_cleanup_idle"
            )
        return {
            "due": bool(reasons),
            "reasons": reasons,
            "graph_version": graph_version,
            "last_graph_version": last_graph_version,
            "graph_delta": graph_delta,
            "elapsed_seconds": elapsed_seconds,
            "pressure_min_interval_seconds": pressure_min_interval_seconds,
            "pressure_cooldown_remaining_seconds": (
                max(0, pressure_min_interval_seconds - elapsed_seconds)
                if elapsed_seconds is not None
                else 0
            ),
            "pressure_mode": pressure_mode,
            "storage_cleanup": storage_cleanup,
        }


    def _memory_quality_diagnostics(
        self,
        *,
        policy: Mapping[str, Any],
        indexes: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        atoms = self.store.list_atoms_filtered(lifecycle_states=["active", "proposed"])
        active_atoms = [
            atom for atom in atoms if atom.get("lifecycle_state") == "active"
        ]
        proposed_atoms = [
            atom for atom in atoms if atom.get("lifecycle_state") == "proposed"
        ]
        by_ref = {str(atom["id"]): atom for atom in atoms}
        # The capacity ceiling historically applies to the whole hot set:
        # active canonical atoms plus dormant proposals. Keep that contract
        # distinct from lifecycle-active graph quality below.
        hot_count = len(atoms)
        decay = dict(policy.get("decay") or {})
        max_atoms = int(decay.get("max_atoms", 256) or 256)
        max_active_atoms = int(decay.get("max_active_atoms", max_atoms) or max_atoms)
        max_proposed_atoms = int(decay.get("max_proposed_atoms", max_atoms) or max_atoms)
        edge_degrees = self.store.edge_degree_counts()
        # Archival deliberately retires hot graph edges, but those rows remain
        # durable provenance. An active consolidation whose only sources were
        # archived is historically connected, not an orphan that never had
        # graph lineage.
        historical_edge_refs = set(
            self.store.edge_degree_counts(include_deleted=True)
        )
        head_anchored_refs = {
            str(head.get("head_ref") or "")
            for head in self.store.list_memory_heads()
            if str(head.get("head_ref") or "")
        }
        isolated = [
            atom
            for atom in active_atoms
            if int(edge_degrees.get(str(atom["id"]), 0)) == 0
            and str(atom["id"]) not in historical_edge_refs
            and str(atom["id"]) not in head_anchored_refs
        ]
        isolated_proposed = [
            atom
            for atom in proposed_atoms
            if int(edge_degrees.get(str(atom["id"]), 0)) == 0
        ]
        isolated_by_type: dict[str, int] = {}
        for atom in isolated:
            atom_type = str(atom.get("type") or "unknown")
            isolated_by_type[atom_type] = isolated_by_type.get(atom_type, 0) + 1

        active_refs = {str(atom["id"]) for atom in active_atoms}
        edges = [
            edge
            for edge in self.store.list_edges()
            if not edge.get("deleted")
            and edge.get("lifecycle_state", "active") == "active"
        ]
        active_edges = [
            edge
            for edge in edges
            if str(edge.get("source_ref") or "") in active_refs
            and str(edge.get("target_ref") or "") in active_refs
        ]
        adjacency: dict[str, set[str]] = {ref: set() for ref in active_refs}
        relation_distribution: dict[str, int] = {}
        derivation_distribution: dict[str, int] = {}
        confidence_histogram = {
            "0.00-0.24": 0,
            "0.25-0.49": 0,
            "0.50-0.74": 0,
            "0.75-0.89": 0,
            "0.90-1.00": 0,
        }
        for edge in active_edges:
            source_ref = str(edge["source_ref"])
            target_ref = str(edge["target_ref"])
            adjacency[source_ref].add(target_ref)
            adjacency[target_ref].add(source_ref)
            relation = str(edge.get("relation") or "unknown")
            relation_distribution[relation] = relation_distribution.get(relation, 0) + 1
            derivation = edge.get("derivation")
            derivation = derivation if isinstance(derivation, Mapping) else {}
            derivation_kind = str(derivation.get("kind") or "unclassified")
            derivation_distribution[derivation_kind] = (
                derivation_distribution.get(derivation_kind, 0) + 1
            )
            confidence = edge.get("confidence")
            confidence = confidence if isinstance(confidence, Mapping) else {}
            score = max(0.0, min(1.0, float(confidence.get("score", 0.0) or 0.0)))
            bucket = (
                "0.00-0.24"
                if score < 0.25
                else "0.25-0.49"
                if score < 0.5
                else "0.50-0.74"
                if score < 0.75
                else "0.75-0.89"
                if score < 0.9
                else "0.90-1.00"
            )
            confidence_histogram[bucket] += 1
        components: list[list[str]] = []
        remaining = set(active_refs)
        while remaining:
            seed = min(remaining)
            stack = [seed]
            component: list[str] = []
            remaining.remove(seed)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency.get(current, set())):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))
        components.sort(key=lambda item: (-len(item), item[0] if item else ""))
        active_degrees = {ref: len(adjacency.get(ref, set())) for ref in active_refs}
        total_degree = sum(active_degrees.values())
        top_hubs = sorted(active_degrees.items(), key=lambda item: (-item[1], item[0]))[:10]
        top_five_degree = sum(degree for _ref, degree in top_hubs[:5])

        evidence_records = self.store.list_evidence()
        # Superseded and archived atoms remain valid historical lineage
        # endpoints. Excluding them made healthy references appear unresolved.
        historical_refs = {
            str(atom["id"])
            for atom in self.store.list_atoms_filtered()
            if str(atom.get("id") or "")
        }
        evidence_ids = {
            str(record.get("evidence_id") or "")
            for record in evidence_records
            if str(record.get("evidence_id") or "")
        }
        known_refs = historical_refs | evidence_ids
        reference_contract_counts = {
            "atom_evidence_refs": 0,
            "edge_evidence_refs": 0,
            "exact_evidence_refs": 0,
            "mistyped_atom_refs": 0,
            "unresolved_refs": 0,
        }
        mistyped_ref_samples: set[str] = set()
        contract_unresolved_samples: set[str] = set()
        for record_kind, records in (("atom", atoms), ("edge", edges)):
            count_key = f"{record_kind}_evidence_refs"
            for record in records:
                for raw_ref in record.get("evidence_refs", []):
                    ref = str(raw_ref or "")
                    if not ref:
                        continue
                    reference_contract_counts[count_key] += 1
                    if ref in evidence_ids:
                        reference_contract_counts["exact_evidence_refs"] += 1
                    elif ref in historical_refs:
                        reference_contract_counts["mistyped_atom_refs"] += 1
                        mistyped_ref_samples.add(ref)
                    else:
                        reference_contract_counts["unresolved_refs"] += 1
                        contract_unresolved_samples.add(ref)
        unresolved_refs: set[str] = set()
        for atom in atoms:
            unresolved_refs.update(
                str(ref)
                for ref in atom.get("evidence_refs", [])
                if str(ref) and str(ref) not in known_refs
            )
        for edge in edges:
            for ref in (
                edge.get("source_ref"),
                edge.get("target_ref"),
                *edge.get("evidence_refs", []),
            ):
                text = str(ref or "")
                if text and text not in known_refs:
                    unresolved_refs.add(text)

        proposal_age = {"under_24h": 0, "1d_to_7d": 0, "over_7d": 0, "unknown": 0}
        dedupe_groups: dict[str, int] = {}
        for atom in proposed_atoms:
            elapsed = self._seconds_since(atom.get("created_at"))
            if elapsed is None:
                proposal_age["unknown"] += 1
            elif elapsed < 86400:
                proposal_age["under_24h"] += 1
            elif elapsed < 7 * 86400:
                proposal_age["1d_to_7d"] += 1
            else:
                proposal_age["over_7d"] += 1
            payload = atom.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            retention = payload.get("proposal_retention")
            retention = retention if isinstance(retention, Mapping) else {}
            dedupe_key = str(retention.get("deduplication_key") or "").strip()
            if dedupe_key:
                dedupe_groups[dedupe_key] = dedupe_groups.get(dedupe_key, 0) + 1

        covered_sources: set[str] = set()
        for atom in active_atoms:
            payload = atom.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            if not (
                payload.get("maintenance_proposal_id")
                or payload.get("created_by_processor")
                or payload.get("distillation_type")
            ):
                continue
            for field in ("source_refs", "maintenance_source_refs", "reviewed_refs"):
                covered_sources.update(
                    str(ref) for ref in payload.get(field, []) if str(ref)
                )

        processor_effectiveness: dict[str, dict[str, int]] = {}
        maintenance_runs = 0
        for event in self.store.list_events(limit=200):
            if event.get("event_type") != "maintenance_distillation_run":
                continue
            maintenance_runs += 1
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            for processor_id, counters in dict(payload.get("processor_results") or {}).items():
                target = processor_effectiveness.setdefault(
                    str(processor_id),
                    {"runs": 0, "proposed": 0, "committed": 0, "already_committed": 0, "deferred": 0, "skipped": 0},
                )
                target["runs"] += 1
                if isinstance(counters, Mapping):
                    for key in ("proposed", "committed", "already_committed", "deferred", "skipped"):
                        target[key] += int(counters.get(key, 0) or 0)

        superseded_refs = self._active_superseded_refs()
        superseded_by_type: dict[str, int] = {}
        for atom_ref in superseded_refs:
            atom = by_ref.get(atom_ref)
            atom_type = str((atom or {}).get("type") or "unknown")
            superseded_by_type[atom_type] = superseded_by_type.get(atom_type, 0) + 1

        graph_version = self.store.graph_version()
        index_lag = {
            str(index["index_name"]): max(
                0, graph_version - int(index.get("graph_version", 0) or 0)
            )
            for index in indexes
        }
        max_index_lag = max(index_lag.values(), default=0)
        pressure_eligible = [
            atom
            for atom in atoms
            if self._pressure_archive_eligible(
                atom, decay=decay, scope={}, lifecycle_state="active"
            )
        ]
        proposed_pressure_eligible = [
            atom
            for atom in atoms
            if self._pressure_archive_eligible(
                atom, decay=decay, scope={}, lifecycle_state="proposed"
            )
        ]
        pressure_eligible_by_type: dict[str, int] = {}
        for atom in pressure_eligible:
            atom_type = str(atom.get("type") or "unknown")
            pressure_eligible_by_type[atom_type] = (
                pressure_eligible_by_type.get(atom_type, 0) + 1
            )
        archives_needed = max(0, hot_count - max_atoms)
        active_archives_needed = max(0, len(active_atoms) - max_active_atoms)
        proposed_archives_needed = max(0, len(proposed_atoms) - max_proposed_atoms)
        capacity_headroom_ratio = float(
            decay.get("capacity_headroom_ratio", 0.2) or 0.0
        )
        capacity_targets = sorted(
            {
                max(1, int(item))
                for item in decay.get("capacity_assessment_targets", [256, 512, 768])
                if item not in (None, "")
            }
            | {max_atoms}
        )
        required_with_headroom = max(1, int(
            math.ceil(hot_count / max(0.1, 1.0 - capacity_headroom_ratio))
        ))
        if capacity_targets[-1] < required_with_headroom:
            capacity_targets.append(required_with_headroom)
        recommended_target = next(
            target for target in capacity_targets if target >= required_with_headroom
        )
        capacity_utilization = hot_count / max(1, max_atoms)
        capacity_near_limit = capacity_utilization >= 1.0 - capacity_headroom_ratio

        warnings: list[str] = []
        if hot_count > max_atoms:
            warnings.append("active_atom_count_exceeds_decay_max_atoms")
            if (
                len(pressure_eligible) + len(proposed_pressure_eligible)
                < archives_needed
            ):
                warnings.append("active_atom_pressure_not_fully_enforceable")
        if capacity_near_limit:
            warnings.append("active_atom_capacity_headroom_low")
        if len(active_atoms) >= max_active_atoms:
            warnings.append("lifecycle_active_atom_limit_reached")
            if (
                len(active_atoms) > max_active_atoms
                and len(pressure_eligible) < active_archives_needed
            ):
                warnings.append("lifecycle_active_atom_limit_not_fully_enforceable")
        if len(proposed_atoms) >= max_proposed_atoms:
            warnings.append("proposed_atom_limit_reached")
            if (
                len(proposed_atoms) > max_proposed_atoms
                and len(proposed_pressure_eligible) < proposed_archives_needed
            ):
                warnings.append("proposed_atom_limit_not_fully_enforceable")
        if superseded_refs:
            warnings.append("active_superseded_atoms_present")
        if isolated:
            warnings.append("isolated_active_atoms_present")
        if reference_contract_counts["mistyped_atom_refs"]:
            warnings.append("atom_ids_present_in_evidence_refs")
        if reference_contract_counts["unresolved_refs"]:
            warnings.append("unresolved_evidence_refs_present")
        maintenance_every = int(
            dict(policy.get("schedule") or {}).get("every_graph_versions", 25) or 25
        )
        if max_index_lag >= maintenance_every:
            warnings.append("derived_index_lag_exceeds_schedule")

        return {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "lifecycle_counts": {
                "active": len(active_atoms),
                "proposed": len(proposed_atoms),
                "hot_total": hot_count,
            },
            # Compatibility aliases retain the historical hot-set meaning.
            "active_atom_count": hot_count,
            "active_atom_limit": max_atoms,
            "active_atom_count_semantics": "hot_total_legacy",
            "hot_atom_count": hot_count,
            "hot_atom_limit": max_atoms,
            "lifecycle_active_atom_count": len(active_atoms),
            "lifecycle_active_atom_limit": max_active_atoms,
            "proposed_atom_count": len(proposed_atoms),
            "proposed_atom_limit": max_proposed_atoms,
            "active_atom_pressure": "over_limit"
            if hot_count > max_atoms
            else "within_limit",
            "pressure_cleanup": {
                "policyless_fallback_enabled": bool(
                    decay.get("pressure_archive_policyless", True)
                ),
                "proposed_fallback_enabled": bool(
                    decay.get("pressure_archive_proposed", True)
                ),
                "archives_needed": archives_needed,
                "active_archives_needed": active_archives_needed,
                "proposed_archives_needed": proposed_archives_needed,
                "eligible_policyless_count": len(pressure_eligible),
                "eligible_policyless_by_type": pressure_eligible_by_type,
                "eligible_proposed_count": len(proposed_pressure_eligible),
                "max_archives_per_run": int(
                    decay.get("pressure_max_archives_per_run", 256) or 256
                ),
                "protected_types": list(decay.get("pressure_protected_types", [])),
            },
            "capacity_assessment": {
                "configured_target": max_atoms,
                "active_count": hot_count,
                "headroom_atoms": max(0, max_atoms - hot_count),
                "utilization": round(capacity_utilization, 4),
                "headroom_ratio_target": capacity_headroom_ratio,
                "near_limit": capacity_near_limit,
                "recommended_target": recommended_target,
                "candidate_targets": [
                    {
                        "target": target,
                        "headroom_atoms": target - hot_count,
                        "utilization": round(hot_count / max(1, target), 4),
                        "meets_headroom_target": target >= required_with_headroom,
                    }
                    for target in capacity_targets
                ],
            },
            "active_superseded_atoms": {
                "count": len(superseded_refs),
                "by_type": superseded_by_type,
                "sample_refs": sorted(superseded_refs)[:10],
            },
            "isolated_active_atoms": {
                "count": len(isolated),
                "by_type": isolated_by_type,
                "sample_refs": sorted(str(atom["id"]) for atom in isolated)[:10],
                "head_anchored_excluded_count": sum(
                    1
                    for atom in active_atoms
                    if int(edge_degrees.get(str(atom["id"]), 0)) == 0
                    and str(atom["id"]) in head_anchored_refs
                ),
                "historically_connected_excluded_count": sum(
                    1
                    for atom in active_atoms
                    if int(edge_degrees.get(str(atom["id"]), 0)) == 0
                    and str(atom["id"]) in historical_edge_refs
                    and str(atom["id"]) not in head_anchored_refs
                ),
            },
            "isolated_proposed_atoms": {
                "count": len(isolated_proposed),
                "expected_dormant": True,
                "sample_refs": sorted(
                    str(atom["id"]) for atom in isolated_proposed
                )[:10],
            },
            "reference_contract": {
                **reference_contract_counts,
                "mistyped_atom_ref_samples": sorted(mistyped_ref_samples)[:32],
                "unresolved_ref_samples": sorted(contract_unresolved_samples)[:32],
            },
            "derived_index_lag": {
                "max_graph_delta": max_index_lag,
                "by_index": index_lag,
            },
            "graph_quality": {
                "active_atom_type_distribution": self._counts(active_atoms, "type"),
                "active_relation_distribution": dict(sorted(relation_distribution.items())),
                "active_edge_count": len(active_edges),
                "component_count": len(components),
                "largest_component_size": len(components[0]) if components else 0,
                "component_sizes": [len(component) for component in components[:20]],
                "hub_concentration_top_five": round(
                    top_five_degree / max(1, total_degree), 4
                ),
                "top_hubs": [
                    {"atom_ref": ref, "degree": degree} for ref, degree in top_hubs
                ],
                "edge_confidence_histogram": confidence_histogram,
                "edge_derivation_distribution": dict(sorted(derivation_distribution.items())),
                "unresolved_ref_count": len(unresolved_refs),
                "unresolved_ref_samples": sorted(unresolved_refs)[:32],
            },
            "proposal_quality": {
                "age_distribution": proposal_age,
                "dedupe_key_count": len(dedupe_groups),
                "duplicate_dedupe_key_count": sum(
                    1 for count in dedupe_groups.values() if count > 1
                ),
                "duplicate_proposal_count": sum(
                    count - 1 for count in dedupe_groups.values() if count > 1
                ),
                "covered_source_count": len(covered_sources),
            },
            "maintenance_processor_effectiveness": {
                "recent_run_count": maintenance_runs,
                "by_processor": processor_effectiveness,
            },
        }


    def _storage_cleanup_due(
        self,
        cleanup: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        force: bool = False,
        pressure_mode: str | None = None,
    ) -> dict[str, Any]:
        if not cleanup.get("enabled", True) and not force:
            return {"due": False, "reason": "storage_cleanup_disabled"}
        pressure_mode = str(pressure_mode or self._capacity_pressure_mode())
        pressure_triggered = bool(
            cleanup.get("run_on_pressure", True)
            and pressure_mode in set(cleanup.get("pressure_modes", ["orange", "red"]))
        )
        last_foreground = (
            self.store.get_meta("last_foreground_activity_at")
            or state.get("last_foreground_activity_at")
        )
        idle_elapsed = self._seconds_since(last_foreground)
        idle_after = int(cleanup.get("idle_after_seconds", 300) or 0)
        if (
            not pressure_triggered
            and idle_elapsed is not None
            and idle_elapsed < idle_after
            and not force
        ):
            return {
                "due": False,
                "reason": "foreground_activity_recent",
                "pressure_mode": pressure_mode,
                "pressure_triggered": False,
                "idle_elapsed_seconds": idle_elapsed,
                "idle_after_seconds": idle_after,
                "last_foreground_activity_at": last_foreground,
            }
        last_cleanup = (
            self.store.get_meta("last_storage_cleanup_at")
            or state.get("last_storage_cleanup_at")
        )
        cleanup_elapsed = self._seconds_since(last_cleanup)
        min_interval = int(
            cleanup.get(
                "pressure_min_interval_seconds"
                if pressure_triggered
                else "min_interval_seconds",
                300 if pressure_triggered else 900,
            )
            or 0
        )
        if cleanup_elapsed is not None and cleanup_elapsed < min_interval and not force:
            return {
                "due": False,
                "reason": "cleanup_interval_not_elapsed",
                "pressure_mode": pressure_mode,
                "pressure_triggered": pressure_triggered,
                "elapsed_since_cleanup_seconds": cleanup_elapsed,
                "min_interval_seconds": min_interval,
                "last_storage_cleanup_at": last_cleanup,
            }
        return {
            "due": True,
            "reason": (
                "force"
                if force
                else "capacity_pressure"
                if pressure_triggered
                else "idle_interval_elapsed"
            ),
            "pressure_mode": pressure_mode,
            "pressure_triggered": pressure_triggered,
            "idle_elapsed_seconds": idle_elapsed,
            "idle_after_seconds": idle_after,
            "last_foreground_activity_at": last_foreground,
            "elapsed_since_cleanup_seconds": cleanup_elapsed,
            "min_interval_seconds": min_interval,
            "last_storage_cleanup_at": last_cleanup,
        }


    def _run_storage_cleanup(
        self,
        *,
        cleanup: Mapping[str, Any],
        due: Mapping[str, Any],
        scope: Mapping[str, Any],
        actor: str,
        state: Mapping[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        now = utc_now()
        pressure_triggered = bool(due.get("pressure_triggered"))
        protected_types = {str(item) for item in cleanup.get("protected_types", [])}
        max_deletions = max(0, int(cleanup.get("max_deletions_per_tick", 256) or 0))
        write_batch_size = max(1, int(cleanup.get("write_batch_size", 32) or 32))
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        tombstones: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        deleted_refs: list[str] = []
        events: list[dict[str, Any]] = []
        index_lifecycle_states: list[str] = []
        if cleanup.get("remove_archived_from_hot_index", True):
            index_lifecycle_states.append("archived")
        if cleanup.get("remove_superseded_from_hot_index", True):
            index_lifecycle_states.append("superseded")
        index_health_statuses = (
            ["stale"] if cleanup.get("remove_stale_from_hot_index", True) else []
        )
        compact_after = cleanup.get(
            "pressure_compact_idempotency_after_seconds"
            if pressure_triggered
            else "compact_idempotency_after_seconds"
        )

        # Plan from a coherent read, then revalidate each candidate immediately
        # before mutation. The plan never holds SQLite's single-writer slot.
        candidates: list[dict[str, Any]] = []
        if max_deletions:
            with self.store.read_snapshot():
                current_head_refs = {
                    str(head.get("head_ref") or "")
                    for head in self.store.list_memory_heads()
                    if str(head.get("head_ref") or "")
                }
                leased_refs = self.store.reference_lease_refs_from_connection(
                    self.store.conn
                )
                atoms = sorted(
                    self.store.list_atoms_filtered(include_deleted=True),
                    key=lambda atom: (
                        str(
                            atom.get("last_accessed")
                            or atom.get("updated_at")
                            or atom.get("observed_at")
                            or ""
                        ),
                        str(atom.get("id") or ""),
                    ),
                )
                for atom in atoms:
                    if len(candidates) >= max_deletions:
                        break
                    if not maintenance_scope_visible(atom["scope"], scope):
                        continue
                    if str(atom["id"]) in current_head_refs:
                        continue
                    if str(atom["id"]) in leased_refs:
                        continue
                    if atom["type"] in protected_types and not atom.get("deleted"):
                        continue
                    if self._storage_deletion_reason(
                        atom,
                        cleanup,
                        pressure_triggered=pressure_triggered,
                    ) is not None:
                        candidates.append(atom)

        # Derived-index pruning is bounded by atom and yields between batches.
        # Keep it separate from canonical deletion batches so clients can enter
        # the FIFO between every maintenance phase.
        index_prune = {
            "status": "completed",
            "rows": 0,
            "atom_count": 0,
            "write_batch_size": write_batch_size,
            "write_batch_count": 0,
            "lifecycle_states": index_lifecycle_states,
            "health_statuses": index_health_statuses,
        }
        remaining_prune_atoms = max(
            0,
            int(cleanup.get("max_index_prune_atoms_per_tick", 512) or 0),
        )
        while remaining_prune_atoms > 0:
            prune_batch_size = min(write_batch_size, remaining_prune_atoms)
            with self.store.transaction() as conn:
                pruned = self.store.prune_atom_text_index(
                    conn,
                    lifecycle_states=index_lifecycle_states,
                    health_statuses=index_health_statuses,
                    max_atoms=prune_batch_size,
                )
            if pruned.get("status") == "skipped":
                index_prune = {
                    **pruned,
                    "write_batch_size": write_batch_size,
                    "write_batch_count": 0,
                }
                break
            pruned_atoms = int(pruned.get("atom_count", 0) or 0)
            index_prune["rows"] += int(pruned.get("rows", 0) or 0)
            index_prune["atom_count"] += pruned_atoms
            if pruned_atoms:
                index_prune["write_batch_count"] += 1
            remaining_prune_atoms -= pruned_atoms
            if pruned_atoms < prune_batch_size:
                break
        index_prune["limit_reached"] = remaining_prune_atoms == 0 and bool(
            index_prune.get("atom_count")
        )

        if compact_after is None:
            idempotency = {
                "status": "skipped",
                "reason": "idempotency_compaction_disabled",
                "rows": 0,
            }
        else:
            remaining = max(
                0,
                int(cleanup.get("max_idempotency_compactions_per_tick", 512) or 0),
            )
            idempotency = {
                "status": "completed",
                "rows": 0,
                "original_response_bytes": 0,
                "compacted_response_bytes": 0,
                "saved_bytes": 0,
            }
            older_than = self._iso_before_seconds(int(compact_after))
            while remaining > 0:
                batch_limit = min(write_batch_size, remaining)
                with self.store.transaction() as conn:
                    compacted = self.store.compact_idempotency_responses(
                        conn,
                        older_than=older_than,
                        max_rows=batch_limit,
                    )
                compacted_rows = int(compacted.get("rows", 0) or 0)
                for key in (
                    "rows",
                    "original_response_bytes",
                    "compacted_response_bytes",
                    "saved_bytes",
                ):
                    idempotency[key] += int(compacted.get(key, 0) or 0)
                remaining -= compacted_rows
                if compacted_rows < batch_limit:
                    break

        for batch_number, batch in enumerate(
            (
                candidates[offset : offset + write_batch_size]
                for offset in range(0, len(candidates), write_batch_size)
            ),
            start=1,
        ):
            batch_actions: list[dict[str, Any]] = []
            batch_atoms: list[dict[str, Any]] = []
            batch_edges: list[dict[str, Any]] = []
            batch_tombstones: list[dict[str, Any]] = []
            batch_refs: list[str] = []
            with self.store.transaction() as conn:
                current_head_refs_for_batch = {
                    str(head.get("head_ref") or "")
                    for head in self.store.list_memory_heads_from_connection(conn)
                    if str(head.get("head_ref") or "")
                }
                for planned_atom in batch:
                    atom = self.store.get_atom(str(planned_atom["id"]))
                    if (
                        atom is None
                        or int(atom.get("version", 0))
                        != int(planned_atom.get("version", 0))
                        or str(atom.get("id") or "")
                        in current_head_refs_for_batch
                        or self.store.is_reference_leased(
                            conn, str(planned_atom["id"])
                        )
                        or not maintenance_scope_visible(atom["scope"], scope)
                        or (
                            atom["type"] in protected_types
                            and not atom.get("deleted")
                        )
                    ):
                        continue
                    reason = self._storage_deletion_reason(
                        atom,
                        cleanup,
                        pressure_triggered=pressure_triggered,
                    )
                    if reason is None:
                        continue
                    if atom.get("deleted"):
                        if self.store.get_tombstone(str(atom["id"])) is None:
                            batch_tombstones.append(
                                self.store.insert_tombstone(
                                    conn,
                                    target_ref=str(atom["id"]),
                                    content_digest=self._memory_identity_digest(atom),
                                    recreation_policy="block_recreate",
                                    reason=reason,
                                )
                            )
                        deleted_edges = self.store.mark_edges_deleted_for_ref(
                            conn, str(atom["id"])
                        )
                        batch_edges.extend(
                            {
                                "edge_id": edge["edge_id"],
                                "deleted": 1,
                            }
                            for edge in deleted_edges
                        )
                        action = {
                            "atom_ref": atom["id"],
                            "action": "purge_deleted_projection",
                            "reason": reason,
                            "lifecycle_state_before": atom["lifecycle_state"],
                            "health_status_before": atom["health_status"],
                        }
                        batch_refs.append(atom["id"])
                        batch_actions.append(action)
                        continue
                    updated = dict(atom)
                    updated["lifecycle_state"] = "deleted"
                    updated["health_status"] = "deleted"
                    updated["deleted"] = 1
                    updated["version"] = int(atom["version"]) + 1
                    updated["updated_at"] = now
                    updated["revision_history"] = list(updated["revision_history"])
                    updated["revision_history"].append(
                        {
                            "version": atom["version"],
                            "digest": digest(self._atom_projection(atom)),
                            "changed_at": now,
                            "actor": actor,
                            "reason": reason,
                        }
                    )
                    # Deleted atoms leave the hot index; rebuilding their
                    # semantic vector while holding the writer is pure waste.
                    updated["index_refs"] = {}
                    updated = normalize_atom(updated, require_id=True)
                    updated["deleted"] = 1
                    tombstone = self.store.insert_tombstone(
                        conn,
                        target_ref=atom["id"],
                        content_digest=self._memory_identity_digest(atom),
                        recreation_policy="block_recreate",
                        reason=reason,
                    )
                    deleted_edges = self.store.mark_edges_deleted_for_ref(
                        conn, atom["id"]
                    )
                    self.store.replace_atom(conn, updated)
                    action = {
                        "atom_ref": atom["id"],
                        "action": "delete",
                        "reason": reason,
                        "lifecycle_state_before": atom["lifecycle_state"],
                        "health_status_before": atom["health_status"],
                    }
                    # The tombstone and action hold the durable deletion
                    # receipt. Journal only minimal removal projections, not
                    # the payload being physically discarded.
                    batch_atoms.append({"id": atom["id"], "deleted": 1})
                    batch_edges.extend(
                        {
                            "edge_id": edge["edge_id"],
                            "deleted": 1,
                        }
                        for edge in deleted_edges
                    )
                    batch_tombstones.append(tombstone)
                    batch_refs.append(atom["id"])
                    batch_actions.append(action)
                if batch_actions:
                    report_housekeeping = not events
                    batch_event = self.store.append_event(
                        conn,
                        event_type="storage_cleanup_run",
                        actor=actor,
                        payload={
                            "operation": "run_storage_cleanup",
                            "policy": dict(cleanup),
                            "due": dict(due),
                            "batch": {
                                "number": batch_number,
                                "size": len(batch_actions),
                                "write_batch_size": write_batch_size,
                            },
                            "actions": batch_actions,
                            "index_prune": index_prune
                            if report_housekeeping
                            else {"status": "reported_in_first_batch"},
                            "idempotency": idempotency
                            if report_housekeeping
                            else {"status": "reported_in_first_batch"},
                            "projected_atoms": batch_atoms,
                            "projected_edges": batch_edges,
                            "tombstones": batch_tombstones,
                        },
                        target_refs=batch_refs,
                    )
                    reasons_by_ref = {
                        str(action["atom_ref"]): str(action["reason"])
                        for action in batch_actions
                    }
                    for atom_ref in batch_refs:
                        self.store.retire_and_purge_edges_for_ref(
                            conn,
                            atom_ref,
                            reason=reasons_by_ref[atom_ref],
                        )
                        self.store.purge_atom_projection(conn, atom_ref)
                    # Storage cleanup physically deletes canonical payloads,
                    # so retain the strong deletion contract and purge packet
                    # copies before committing each bounded delete batch.
                    self.store.clear_packet_cache(conn)
                    events.append(batch_event)
            actions.extend(batch_actions)
            projected_atoms.extend(batch_atoms)
            projected_edges.extend(batch_edges)
            tombstones.extend(batch_tombstones)
            deleted_refs.extend(batch_refs)
            self.store.cooperative_maintenance_yield()

        housekeeping_changed = bool(
            index_prune.get("rows") or idempotency.get("rows")
        )
        with self.store.transaction() as conn:
            if not events and housekeeping_changed:
                event = self.store.append_event(
                    conn,
                    event_type="storage_cleanup_run",
                    actor=actor,
                    payload={
                        "operation": "run_storage_cleanup",
                        "policy": dict(cleanup),
                        "due": dict(due),
                        "batch": {
                            "number": 1,
                            "size": 0,
                            "write_batch_size": write_batch_size,
                        },
                        "actions": [],
                        "index_prune": index_prune,
                        "idempotency": idempotency,
                        "projected_atoms": [],
                        "projected_edges": [],
                        "tombstones": [],
                    },
                    target_refs=[],
                )
                self.store.retire_packet_cache(conn)
                events.append(event)
            self.store._set_meta(conn, "last_storage_cleanup_at", now)
        event = events[-1] if events else None

        journal_policy = dict(cleanup.get("journal_compaction") or {})
        journal_compaction: dict[str, Any] = {
            "status": "skipped",
            "reason": "journal_compaction_disabled",
        }
        if journal_policy.get("enabled", True):
            try:
                journal_compaction = self.store.compact_journal_segment(
                    max_events=int(
                        journal_policy.get("max_events_per_segment", 512) or 512
                    ),
                    min_events=int(
                        journal_policy.get(
                            "pressure_min_events_per_segment"
                            if pressure_triggered
                            else "min_events_per_segment",
                            64 if pressure_triggered else 128,
                        )
                        or 1
                    ),
                    retain_tail_events=int(
                        journal_policy.get("retain_tail_events", 128) or 0
                    ),
                    retain_snapshots=int(
                        journal_policy.get("retain_snapshots", 1) or 1
                    ),
                    retain_full_segments=int(
                        journal_policy.get("retain_full_segments", 2) or 0
                    ),
                )
            except Exception as exc:
                journal_compaction = {"status": "error", "error": str(exc)}

        sqlite_compaction = dict(cleanup.get("sqlite_compaction") or {})
        incremental_vacuum = {
            "status": "skipped",
            "reason": "incremental_vacuum_disabled",
        }
        if sqlite_compaction.get("incremental_vacuum", True):
            try:
                incremental_vacuum = self.store.incremental_vacuum(
                    max_pages=int(
                        sqlite_compaction.get("incremental_vacuum_pages", 4096)
                        or 0
                    )
                )
            except Exception as exc:
                incremental_vacuum = {"status": "error", "error": str(exc)}
        checkpoint = {"status": "skipped", "reason": "checkpoint_disabled"}
        if sqlite_compaction.get("checkpoint_wal", True):
            try:
                checkpoint_mode = str(
                    sqlite_compaction.get(
                        "pressure_checkpoint_mode"
                        if pressure_triggered
                        else "checkpoint_mode",
                        "TRUNCATE" if pressure_triggered else "PASSIVE",
                    )
                )
                checkpoint = self.store.checkpoint_wal(
                    mode=checkpoint_mode
                )
            except Exception as exc:
                checkpoint = {"status": "error", "error": str(exc)}
        vacuum = self._maybe_vacuum_sqlite(
            sqlite_compaction=sqlite_compaction,
            state=state,
            force=force,
        )
        checkpoint_after_vacuum = {"status": "skipped", "reason": "vacuum_not_completed"}
        if (
            sqlite_compaction.get("checkpoint_wal", True)
            and vacuum.get("status") == "completed"
        ):
            try:
                checkpoint_after_vacuum = self.store.checkpoint_wal(
                    mode=str(
                        sqlite_compaction.get(
                            "pressure_checkpoint_mode"
                            if pressure_triggered
                            else "checkpoint_mode",
                            "TRUNCATE" if pressure_triggered else "PASSIVE",
                        )
                    )
                )
            except Exception as exc:
                checkpoint_after_vacuum = {"status": "error", "error": str(exc)}
        return {
            "status": "completed",
            "due": dict(due),
            "index_prune": index_prune,
            "deleted_atom_count": len(actions),
            "physically_purged_atom_count": len(actions),
            "deleted_atom_refs": deleted_refs,
            "write_batch_size": write_batch_size,
            "write_batch_count": len(events),
            "idempotency": idempotency,
            "journal_compaction": journal_compaction,
            "incremental_vacuum": incremental_vacuum,
            "checkpoint": checkpoint,
            "vacuum": vacuum,
            "checkpoint_after_vacuum": checkpoint_after_vacuum,
            "event": event,
            "events": events,
        }


    def _storage_deletion_reason(
        self,
        atom: Mapping[str, Any],
        cleanup: Mapping[str, Any],
        *,
        pressure_triggered: bool = False,
    ) -> str | None:
        observed_ages = [
            age
            for timestamp in (
                atom.get("last_accessed"),
                atom.get("updated_at"),
                atom.get("observed_at"),
            )
            if timestamp and (age := self._seconds_since(timestamp)) is not None
        ]
        updated_age = min(observed_ages) if observed_ages else None
        if atom.get("deleted"):
            purge_after = cleanup.get(
                "pressure_purge_deleted_after_seconds"
                if pressure_triggered
                else "purge_deleted_after_seconds"
            )
            if (
                purge_after is not None
                and updated_age is not None
                and updated_age >= int(purge_after)
            ):
                return "storage_cleanup_deleted_projection_retention_elapsed"
            return None
        archived_after = cleanup.get(
            "pressure_delete_archived_after_seconds"
            if pressure_triggered
            else "delete_archived_after_seconds"
        )
        if (
            archived_after is not None
            and atom.get("lifecycle_state") == "archived"
            and updated_age is not None
            and updated_age >= int(archived_after)
        ):
            return "storage_cleanup_archived_retention_elapsed"
        superseded_after = cleanup.get(
            "pressure_delete_superseded_after_seconds"
            if pressure_triggered
            else "delete_superseded_after_seconds"
        )
        if (
            superseded_after is not None
            and atom.get("lifecycle_state") == "superseded"
            and updated_age is not None
            and updated_age >= int(superseded_after)
        ):
            return "storage_cleanup_superseded_retention_elapsed"
        stale_after = cleanup.get(
            "pressure_delete_stale_after_seconds"
            if pressure_triggered
            else "delete_stale_after_seconds"
        )
        if (
            stale_after is not None
            and atom.get("health_status") == "stale"
            and updated_age is not None
            and updated_age >= int(stale_after)
        ):
            return "storage_cleanup_stale_retention_elapsed"
        return None


    def _maybe_vacuum_sqlite(
        self,
        *,
        sqlite_compaction: Mapping[str, Any],
        state: Mapping[str, Any],
        force: bool,
    ) -> dict[str, Any]:
        if not sqlite_compaction.get("vacuum_enabled", False):
            return {"status": "skipped", "reason": "vacuum_disabled"}
        idle_after = int(sqlite_compaction.get("vacuum_idle_after_seconds", 1800) or 0)
        last_foreground = (
            self.store.get_meta("last_foreground_activity_at")
            or state.get("last_foreground_activity_at")
        )
        idle_elapsed = self._seconds_since(last_foreground)
        if idle_elapsed is not None and idle_elapsed < idle_after and not force:
            return {
                "status": "skipped",
                "reason": "foreground_activity_recent",
                "idle_elapsed_seconds": idle_elapsed,
                "idle_after_seconds": idle_after,
            }
        min_interval = int(sqlite_compaction.get("vacuum_min_interval_seconds", 86400) or 0)
        last_vacuum = self.store.get_meta("last_vacuum_at") or state.get("last_vacuum_at")
        vacuum_elapsed = self._seconds_since(last_vacuum)
        if vacuum_elapsed is not None and vacuum_elapsed < min_interval and not force:
            return {
                "status": "skipped",
                "reason": "vacuum_interval_not_elapsed",
                "elapsed_since_vacuum_seconds": vacuum_elapsed,
                "min_interval_seconds": min_interval,
                "last_vacuum_at": last_vacuum,
            }
        try:
            result = self.store.vacuum()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        completed_at = utc_now()
        self.store.set_meta("last_vacuum_at", completed_at)
        return {**result, "completed_at": completed_at}


    @staticmethod
    def _partition_evidence_refs(
        refs: Sequence[Any],
        *,
        atom_refs: set[str],
        evidence_refs: set[str],
    ) -> tuple[list[str], list[str], list[str]]:
        normalized = list(
            dict.fromkeys(str(ref).strip() for ref in refs if str(ref).strip())
        )
        exact_evidence: list[str] = []
        source_atoms: list[str] = []
        unresolved: list[str] = []
        for ref in normalized:
            # In the unlikely event that legacy data reused an identifier in
            # both namespaces, retain it as evidence.  Evidence identifiers
            # are the narrower contract and must not be silently demoted.
            if ref in evidence_refs:
                exact_evidence.append(ref)
            elif ref in atom_refs:
                source_atoms.append(ref)
            else:
                unresolved.append(ref)
        return exact_evidence, source_atoms, unresolved


    def _run_reference_contract_repairs(
        self,
        *,
        scope: Mapping[str, Any],
        actor: str,
        max_repairs: int,
    ) -> dict[str, Any]:
        """Repair legacy provenance that mixed atom IDs into evidence_refs.

        Evidence references name captured evidence records only.  Older
        clients also placed atom lineage there.  Preserve that lineage in the
        generic source_refs metadata and preserve unresolvable identifiers in
        unresolved_source_refs so maintenance never discards provenance.
        """

        all_atoms = self.store.list_atoms_filtered(include_deleted=True)
        atom_refs = {
            str(atom.get("id") or "")
            for atom in all_atoms
            if str(atom.get("id") or "")
        }
        evidence_refs = {
            str(record.get("evidence_id") or "")
            for record in self.store.list_evidence()
            if str(record.get("evidence_id") or "")
        }
        candidate_atoms = [
            atom
            for atom in all_atoms
            if not atom.get("deleted")
            and atom.get("lifecycle_state") in {"active", "proposed"}
            and maintenance_scope_visible(atom.get("scope", {}), scope)
        ]
        candidate_edges = [
            edge
            for edge in self.store.list_edges()
            if edge.get("lifecycle_state", "active") == "active"
            and maintenance_scope_visible(edge.get("scope", {}), scope)
        ]
        actions: list[dict[str, Any]] = []
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        now = utc_now()

        with self.store.transaction() as conn:
            for atom in candidate_atoms:
                if len(actions) >= max_repairs:
                    break
                exact, source_atoms, unresolved = self._partition_evidence_refs(
                    atom.get("evidence_refs", []),
                    atom_refs=atom_refs,
                    evidence_refs=evidence_refs,
                )
                if (
                    exact == list(atom.get("evidence_refs", []))
                    and not source_atoms
                    and not unresolved
                ):
                    continue
                payload = dict(atom.get("payload") or {})
                prior_source_refs = payload.get("source_refs", [])
                if not isinstance(prior_source_refs, list):
                    prior_source_refs = []
                prior_unresolved_refs = payload.get("unresolved_source_refs", [])
                if not isinstance(prior_unresolved_refs, list):
                    prior_unresolved_refs = []
                payload["source_refs"] = list(
                    dict.fromkeys(
                        str(ref)
                        for ref in [*prior_source_refs, *source_atoms]
                        if str(ref)
                    )
                )
                if unresolved or prior_unresolved_refs:
                    payload["unresolved_source_refs"] = list(
                        dict.fromkeys(
                            str(ref)
                            for ref in [*prior_unresolved_refs, *unresolved]
                            if str(ref)
                        )
                    )
                changed = dict(atom)
                changed["payload"] = payload
                changed["evidence_refs"] = exact
                changed["version"] = int(changed["version"]) + 1
                changed["updated_at"] = now
                changed["revision_history"] = list(
                    changed.get("revision_history") or []
                )
                changed["revision_history"].append(
                    {
                        "version": atom["version"],
                        "digest": digest(self._atom_projection(atom)),
                        "changed_at": now,
                        "actor": actor,
                        "reason": "reference_contract_repair",
                    }
                )
                changed = normalize_atom(
                    self._attach_search_index(changed), require_id=True
                )
                self.store.replace_atom(conn, changed)
                projected_atoms.append(changed)
                actions.append(
                    {
                        "record_kind": "atom",
                        "record_ref": changed["id"],
                        "moved_source_refs": source_atoms,
                        "moved_unresolved_refs": unresolved,
                        "retained_evidence_refs": exact,
                    }
                )

            for edge in candidate_edges:
                if len(actions) >= max_repairs:
                    break
                exact, source_atoms, unresolved = self._partition_evidence_refs(
                    edge.get("evidence_refs", []),
                    atom_refs=atom_refs,
                    evidence_refs=evidence_refs,
                )
                if (
                    exact == list(edge.get("evidence_refs", []))
                    and not source_atoms
                    and not unresolved
                ):
                    continue
                derivation = dict(edge.get("derivation") or {})
                prior_source_refs = derivation.get("source_refs", [])
                if not isinstance(prior_source_refs, list):
                    prior_source_refs = []
                prior_unresolved_refs = derivation.get(
                    "unresolved_source_refs", []
                )
                if not isinstance(prior_unresolved_refs, list):
                    prior_unresolved_refs = []
                derivation["source_refs"] = list(
                    dict.fromkeys(
                        str(ref)
                        for ref in [*prior_source_refs, *source_atoms]
                        if str(ref)
                    )
                )
                if unresolved or prior_unresolved_refs:
                    derivation["unresolved_source_refs"] = list(
                        dict.fromkeys(
                            str(ref)
                            for ref in [*prior_unresolved_refs, *unresolved]
                            if str(ref)
                        )
                    )
                changed_edge = dict(edge)
                changed_edge["evidence_refs"] = exact
                changed_edge["derivation"] = derivation
                changed_edge["version"] = int(changed_edge.get("version", 1)) + 1
                changed_edge["updated_at"] = now
                self.store.upsert_edge(conn, changed_edge)
                projected_edges.append(changed_edge)
                actions.append(
                    {
                        "record_kind": "edge",
                        "record_ref": changed_edge["edge_id"],
                        "moved_source_refs": source_atoms,
                        "moved_unresolved_refs": unresolved,
                        "retained_evidence_refs": exact,
                    }
                )

            if actions:
                event = self.store.append_event(
                    conn,
                    event_type="memory_reference_contract_repaired",
                    actor=actor,
                    payload={
                        "operation": "repair_reference_contracts",
                        "actions": actions,
                        "projected_atoms": projected_atoms,
                        "projected_edges": projected_edges,
                    },
                    target_refs=[action["record_ref"] for action in actions],
                )
                self.store.retire_packet_cache(conn)
            else:
                event = None

        return {
            "status": "completed",
            "action_count": len(actions),
            "actions": actions,
            "projected_atoms": projected_atoms,
            "projected_edges": projected_edges,
            "truncated": len(actions) >= max_repairs,
            "event": event,
        }


    def _run_decay_policy(
        self,
        *,
        decay: Mapping[str, Any],
        scope: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        max_atoms = max(1, int(decay.get("max_atoms", 256) or 256))
        max_active_atoms = max(
            1, int(decay.get("max_active_atoms", max_atoms) or max_atoms)
        )
        max_proposed_atoms = max(
            1, int(decay.get("max_proposed_atoms", max_atoms) or max_atoms)
        )
        require_atom_policy = bool(decay.get("require_atom_policy", True))
        actions: list[dict[str, Any]] = []
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        now = utc_now()
        planning_revision = self.store.memory_revision()
        current_head_refs = {
            str(head.get("head_ref") or "")
            for head in self.store.list_memory_heads()
            if str(head.get("head_ref") or "")
            and maintenance_scope_visible(head.get("scope", {}), scope)
        }
        # Older pressure policies could archive the atom still named by a
        # canonical memory head. Plan that structural repair here, but publish
        # it only in the same version-revalidated transaction as its journal
        # event. The protection follows the current head dynamically and does
        # not pin predecessors.
        restoration_plans: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for head_ref in sorted(current_head_refs):
            atom = self.store.get_atom(head_ref)
            if atom is None or atom.get("deleted"):
                continue
            if atom.get("lifecycle_state") != "archived":
                continue
            restoration_plans.append(
                (
                    atom,
                    {
                        "action": "restore",
                        "reason": "current_memory_head_protection",
                        "health_status": "healthy",
                    },
                )
            )
        superseded_refs = (
            self._active_superseded_refs()
            if decay.get("archive_superseded", True)
            else {}
        )
        atoms_by_ref = {
            atom["id"]: atom
            for atom in self.store.list_atoms_filtered(
                lifecycle_states=["active", "proposed", "superseded"]
            )
        }
        for atom in self.store.list_atoms_filtered(
            lifecycle_states=["active", "proposed", "superseded"],
            atom_ids=sorted(superseded_refs),
        ):
            atoms_by_ref[atom["id"]] = atom
        atoms = list(atoms_by_ref.values())
        planned: list[tuple[dict[str, Any], dict[str, Any]]] = []
        planned_archives: set[str] = set()
        edge_degrees = self.store.edge_degree_counts()
        current_plugin_digests: dict[str, str] = {}
        for atom in atoms:
            if str(atom.get("id") or "") not in current_head_refs:
                continue
            payload = atom.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            if payload.get("profile") != "cogito.plugin-activation.v1":
                continue
            plugin_name = str(payload.get("plugin_name") or "")
            package_digest = str(payload.get("package_digest") or "")
            if plugin_name and package_digest:
                current_plugin_digests[plugin_name] = package_digest
        historically_satisfied_commitments = {
            str(edge.get("target_ref") or "")
            for edge in self.store.list_edges(include_deleted=True)
            if edge.get("relation") == "rel:satisfied_commitment"
            and str(edge.get("target_ref") or "")
        }
        duplicate_actions = self._proposed_duplicate_archive_actions(atoms)
        for atom in atoms:
            if not maintenance_scope_visible(atom["scope"], scope):
                continue
            if str(atom.get("id") or "") in current_head_refs:
                continue
            if (
                atom.get("lifecycle_state") == "active"
                and str(atom.get("type") or "") == "commitment"
                and str(atom.get("id") or "")
                in historically_satisfied_commitments
            ):
                action = {
                    "action": "archive",
                    "reason": "satisfied_commitment_recorded",
                    "health_status": "healthy",
                }
                planned.append((atom, action))
                planned_archives.add(str(atom["id"]))
                continue
            if str(atom.get("type") or "") in GOVERNANCE_MAINTENANCE_PROTECTED_TYPES:
                continue
            atom_policy = (
                dict(atom.get("decay_policy") or {})
                if isinstance(atom.get("decay_policy"), Mapping)
                else {}
            )
            payload = atom.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            plugin_name = str(payload.get("plugin_name") or "")
            current_plugin_digest = current_plugin_digests.get(plugin_name)
            obsolete_plugin_semantic_revision = (
                atom.get("lifecycle_state") == "active"
                and payload.get("profile") == "cogito.skill-semantic-node.v1"
                and atom_policy.get("protection_reason")
                == "authoritative_plugin_registry"
                and current_plugin_digest is not None
                and str(payload.get("plugin_digest") or "")
                != current_plugin_digest
            )
            if obsolete_plugin_semantic_revision:
                action = {
                    "action": "archive",
                    "reason": "obsolete_plugin_semantic_revision",
                    "health_status": "stale",
                }
                planned.append((atom, action))
                planned_archives.add(str(atom["id"]))
                continue
            orphaned_authority_revision = (
                atom.get("lifecycle_state") == "active"
                and str(atom.get("type") or "") == "procedure"
                and atom_policy.get("protection_reason")
                == "authoritative_plugin_registry"
                and int(edge_degrees.get(str(atom["id"]), 0) or 0) == 0
            )
            if orphaned_authority_revision:
                action = {
                    "action": "archive",
                    "reason": "orphaned_noncanonical_authority_revision",
                    "health_status": "stale",
                }
                planned.append((atom, action))
                planned_archives.add(str(atom["id"]))
                continue
            explicit_atom_policy = self._has_explicit_atom_decay_policy(atom_policy)
            superseded_action = duplicate_actions.get(str(atom["id"]))
            if superseded_action is None:
                superseded_action = self._decay_action_for_superseded_atom(
                    atom,
                    superseded_by=superseded_refs.get(atom["id"], []),
                    policy=decay,
                )
            if superseded_action is None and atom_policy.get("enabled") is False:
                continue
            if superseded_action is None:
                superseded_action = self._decay_action_for_proposed_atom(
                    atom, policy=decay
                )
            if superseded_action is not None:
                action = superseded_action
            else:
                if require_atom_policy and not explicit_atom_policy:
                    continue
                if self._timestamp_elapsed(atom_policy.get("retain_until")):
                    pass
                elif atom_policy.get("retain_until"):
                    continue
                action = self._decay_action_for_atom(
                    atom,
                    atom_policy=atom_policy,
                    policy=decay,
                )
            if action is None:
                continue
            planned.append((atom, action))
            if action["action"] == "archive":
                planned_archives.add(str(atom["id"]))

        hot_count_before = sum(
            1 for atom in atoms if atom.get("lifecycle_state") in {"active", "proposed"}
        )
        planned_hot_archives = {
            str(atom["id"])
            for atom in atoms
            if str(atom["id"]) in planned_archives
            and atom.get("lifecycle_state") in {"active", "proposed"}
        }
        hot_count_after_rules = hot_count_before - len(planned_hot_archives)
        active_count_after_rules = sum(
            1
            for atom in atoms
            if atom.get("lifecycle_state") == "active"
            and str(atom["id"]) not in planned_archives
        )
        proposed_count_after_rules = sum(
            1
            for atom in atoms
            if atom.get("lifecycle_state") == "proposed"
            and str(atom["id"]) not in planned_archives
        )
        total_pressure_needed = max(0, hot_count_after_rules - max_atoms)
        active_pressure_needed = max(0, active_count_after_rules - max_active_atoms)
        proposed_pressure_needed = max(
            0, proposed_count_after_rules - max_proposed_atoms
        )
        pressure_limit = int(decay.get("pressure_max_archives_per_run", 256) or 256)
        proposal_pressure_candidates = [
            atom
            for atom in atoms
            if str(atom["id"]) not in planned_archives
            and str(atom["id"]) not in current_head_refs
            and self._pressure_archive_eligible(
                atom, decay=decay, scope=scope, lifecycle_state="proposed"
            )
        ]
        active_pressure_candidates = [
            atom
            for atom in atoms
            if str(atom["id"]) not in planned_archives
            and str(atom["id"]) not in current_head_refs
            and self._pressure_archive_eligible(
                atom, decay=decay, scope=scope, lifecycle_state="active"
            )
        ]
        pressure_required = bool(
            total_pressure_needed or active_pressure_needed or proposed_pressure_needed
        )
        proposal_pressure_candidates.sort(
            key=lambda atom: self._pressure_archive_sort_key(atom, edge_degrees)
        )
        active_pressure_candidates.sort(
            key=lambda atom: self._pressure_archive_sort_key(atom, edge_degrees)
        )
        pressure_archive_count = 0
        proposal_archive_count = 0
        active_archive_count = 0
        proposal_archive_target = max(
            proposed_pressure_needed, total_pressure_needed
        )
        if decay.get("pressure_archive_proposed", True) and proposal_archive_target:
            for atom in proposal_pressure_candidates[
                : min(proposal_archive_target, pressure_limit)
            ]:
                action = {
                    "action": "archive",
                    "reason": "proposed_atom_pressure_fallback",
                    "health_status": "stale",
                }
                planned.append((atom, action))
                planned_archives.add(str(atom["id"]))
                pressure_archive_count += 1
                proposal_archive_count += 1
        remaining_total_pressure = max(
            0, total_pressure_needed - pressure_archive_count
        )
        active_archive_target = max(active_pressure_needed, remaining_total_pressure)
        remaining_archive_budget = max(0, pressure_limit - pressure_archive_count)
        if decay.get("pressure_archive_policyless", True) and active_archive_target:
            for atom in active_pressure_candidates[
                : min(active_archive_target, remaining_archive_budget)
            ]:
                action = {
                    "action": "archive",
                    "reason": "active_atom_pressure_policyless_fallback",
                    "health_status": "stale",
                }
                planned.append((atom, action))
                planned_archives.add(str(atom["id"]))
                pressure_archive_count += 1
                active_archive_count += 1

        pressure = {
            "enabled": bool(
                decay.get("pressure_archive_policyless", True)
                or decay.get("pressure_archive_proposed", True)
            ),
            "triggered": pressure_required,
            "max_atoms": max_atoms,
            "max_active_atoms": max_active_atoms,
            "max_proposed_atoms": max_proposed_atoms,
            "hot_count_before": hot_count_before,
            "hot_count_after_rules": hot_count_after_rules,
            "active_count_after_rules": active_count_after_rules,
            "proposed_count_after_rules": proposed_count_after_rules,
            "active_pressure_needed": active_pressure_needed,
            "proposed_pressure_needed": proposed_pressure_needed,
            "eligible_policyless_count": len(active_pressure_candidates),
            "eligible_proposed_count": len(proposal_pressure_candidates),
            "archive_limit": pressure_limit,
            "archive_count": pressure_archive_count,
            "proposal_archive_count": proposal_archive_count,
            "active_archive_count": active_archive_count,
            "remaining_hot_count": hot_count_after_rules - pressure_archive_count,
            "remaining_over_limit": max(
                0,
                hot_count_after_rules - pressure_archive_count - max_atoms,
            ),
            "remaining_active_over_limit": max(
                0,
                active_count_after_rules - active_archive_count - max_active_atoms,
            ),
            "remaining_proposed_over_limit": max(
                0,
                proposed_count_after_rules
                - proposal_archive_count
                - max_proposed_atoms,
            ),
        }

        write_batch_size = max(1, int(decay.get("write_batch_size", 32) or 32))
        # Vector preparation is deterministic read/CPU work. Keep it out of the
        # single-writer transaction, then use atom version as the publish CAS.
        prepared: list[
            tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        ] = []
        stale_revision: dict[str, Any] | None = None
        with self.store.read_snapshot():
            prepared_revision = self.store.memory_revision()
            if prepared_revision != planning_revision:
                stale_revision = {
                    "reason": "canonical_revision_advanced_during_decay_planning",
                    "planned_revision": planning_revision,
                    "current_revision": prepared_revision,
                }
            else:
                for atom, action in [*restoration_plans, *planned]:
                    atom_policy = (
                        dict(atom.get("decay_policy") or {})
                        if isinstance(atom.get("decay_policy"), Mapping)
                        else {}
                    )
                    changed = dict(atom)
                    changed["version"] = int(changed["version"]) + 1
                    changed["updated_at"] = now
                    if action["action"] == "restore":
                        changed["lifecycle_state"] = "active"
                        changed["health_status"] = "healthy"
                    elif action["action"] == "archive":
                        changed["lifecycle_state"] = "archived"
                        changed["health_status"] = action.get(
                            "health_status", "stale"
                        )
                    elif action["action"] == "mark_stale":
                        changed["health_status"] = "stale"
                    elif action["action"] == "mark_low_utility":
                        changed["health_status"] = "low_utility"
                    changed["decay_policy"] = {
                        **atom_policy,
                        "last_decay": {
                            "action": action["action"],
                            "reason": action["reason"],
                            "applied_at": now,
                        },
                    }
                    changed = normalize_atom(
                        self._attach_search_index(changed), require_id=True
                    )
                    prepared.append((atom, action, changed))

        events: list[dict[str, Any]] = []
        skipped_stale_plans = (
            len(restoration_plans) + len(planned) if stale_revision else 0
        )
        expected_revision = prepared_revision
        for batch_number, offset in enumerate(
            range(0, len(prepared), write_batch_size), start=1
        ):
            batch_actions: list[dict[str, Any]] = []
            batch_atoms: list[dict[str, Any]] = []
            batch_edges: list[dict[str, Any]] = []
            batch = prepared[offset : offset + write_batch_size]
            with self.store.transaction() as conn:
                current_revision = self.store.memory_revision()
                if current_revision != expected_revision:
                    skipped_stale_plans += len(prepared) - offset
                    stale_revision = {
                        "reason": "canonical_revision_advanced_before_decay_publish",
                        "planned_revision": expected_revision,
                        "current_revision": current_revision,
                    }
                    break
                current_head_refs_for_batch = {
                    str(head.get("head_ref") or "")
                    for head in self.store.list_memory_heads_from_connection(conn)
                    if str(head.get("head_ref") or "")
                }
                for atom, action, changed in batch:
                    current = self.store.get_atom(str(atom["id"]))
                    is_current_head = str(atom["id"]) in current_head_refs_for_batch
                    if (
                        current is None
                        or current.get("deleted")
                        or int(current.get("version", 0))
                        != int(atom.get("version", 0))
                        or (
                            action["action"] == "restore"
                            and (
                                not is_current_head
                                or current.get("lifecycle_state") != "archived"
                            )
                        )
                        or (action["action"] != "restore" and is_current_head)
                    ):
                        skipped_stale_plans += 1
                        continue
                    restored_edge_count = 0
                    if action["action"] == "restore":
                        restored_edges = self.store.restore_edges_for_ref(
                            conn, str(atom["id"])
                        )
                        restored_edge_count = len(restored_edges)
                        batch_edges.extend(restored_edges)
                    elif action["action"] == "archive":
                        batch_edges.extend(
                            self.store.mark_edges_deleted_for_ref(
                                conn, str(atom["id"])
                            )
                        )
                    self.store.replace_atom(conn, changed)
                    batch_atoms.append(changed)
                    batch_actions.append(
                        {
                            "atom_ref": changed["id"],
                            "action": action["action"],
                            "reason": action["reason"],
                            **(
                                {"superseded_by": action["superseded_by"]}
                                if action.get("superseded_by")
                                else {}
                            ),
                            **(
                                {"restored_edge_count": restored_edge_count}
                                if action["action"] == "restore"
                                else {}
                            ),
                            "health_status": changed["health_status"],
                            "lifecycle_state": changed["lifecycle_state"],
                        }
                    )
                if batch_actions:
                    event = self.store.append_event(
                        conn,
                        event_type="decay_policy_applied",
                        actor=actor,
                        payload={
                            "operation": "run_decay_policy",
                            "policy": dict(decay),
                            "batch": {
                                "number": batch_number,
                                "size": len(batch_actions),
                                "write_batch_size": write_batch_size,
                            },
                            "actions": batch_actions,
                            "projected_atoms": batch_atoms,
                            "projected_edges": batch_edges,
                        },
                        target_refs=[
                            action_item["atom_ref"]
                            for action_item in batch_actions
                        ],
                    )
                    self.store.retire_packet_cache(conn)
                    events.append(event)
                    expected_revision = {
                        "graph_version": int(event["graph_version"]),
                        "journal_head": str(event["checksum"]),
                    }
            actions.extend(batch_actions)
            projected_atoms.extend(batch_atoms)
            projected_edges.extend(batch_edges)
        event = events[-1] if events else None
        return {
            "status": "completed",
            "action_count": len(actions),
            "actions": actions,
            "projected_edges": projected_edges,
            "pressure": pressure,
            "write_batch_size": write_batch_size,
            "write_batch_count": len(events),
            "skipped_stale_plans": skipped_stale_plans,
            "stale_revision": stale_revision,
            "event": event,
            "events": events,
        }


    def _has_explicit_atom_decay_policy(self, atom_policy: Mapping[str, Any]) -> bool:
        return any(
            atom_policy.get(key) not in (None, "", [], {})
            for key in {
                "archive_after_seconds",
                "expires_at",
                "low_utility_threshold",
                "mark_stale_after_seconds",
                "retain_until",
            }
        )


    def _pressure_archive_eligible(
        self,
        atom: Mapping[str, Any],
        *,
        decay: Mapping[str, Any],
        scope: Mapping[str, Any],
        lifecycle_state: str = "active",
    ) -> bool:
        if lifecycle_state == "proposed":
            if not decay.get("pressure_archive_proposed", True):
                return False
        elif not decay.get("pressure_archive_policyless", True):
            return False
        if atom.get("lifecycle_state") != lifecycle_state:
            return False
        if not maintenance_scope_visible(atom.get("scope", {}), scope):
            return False
        if str(atom.get("type") or "") in set(
            decay.get("pressure_protected_types", [])
        ):
            return False
        if lifecycle_state == "proposed":
            if not self._proposal_retention(atom):
                return False
            age = self._seconds_since(atom.get("created_at"))
            if age is None or age < int(
                decay.get("proposal_pressure_min_age_seconds", 3600) or 0
            ):
                return False
            return True
        atom_policy = (
            dict(atom.get("decay_policy") or {})
            if isinstance(atom.get("decay_policy"), Mapping)
            else {}
        )
        if atom_policy.get("enabled") is False:
            return False
        if self._has_explicit_atom_decay_policy(atom_policy):
            return False
        retain_until = atom_policy.get("retain_until")
        if retain_until and not self._timestamp_elapsed(retain_until):
            return False
        return True


    @staticmethod
    def _proposal_retention(atom: Mapping[str, Any]) -> dict[str, Any]:
        payload = atom.get("payload")
        if not isinstance(payload, Mapping):
            return {}
        retention = payload.get("proposal_retention")
        if isinstance(retention, Mapping):
            return dict(retention)
        profile = str(payload.get("profile") or "")
        archive_after_seconds = LEGACY_PROPOSAL_RETENTION_SECONDS.get(profile)
        if archive_after_seconds is None:
            return {}
        return {
            "profile": "amos.legacy-proposal-retention.v1",
            "archive_after_seconds": archive_after_seconds,
        }


    def _decay_action_for_proposed_atom(
        self, atom: Mapping[str, Any], *, policy: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if atom.get("lifecycle_state") != "proposed":
            return None
        retention = self._proposal_retention(atom)
        value = retention.get(
            "archive_after_seconds", policy.get("archive_proposed_after_seconds")
        )
        if value in (None, ""):
            return None
        age = self._seconds_since(atom.get("created_at"))
        if age is None or age < max(0, int(value)):
            return None
        return {
            "action": "archive",
            "reason": "proposed_retention_elapsed",
            "health_status": "stale",
        }


    def _proposed_duplicate_archive_actions(
        self, atoms: Sequence[Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
        for atom in atoms:
            if atom.get("lifecycle_state") != "proposed":
                continue
            retention = self._proposal_retention(atom)
            key = str(retention.get("deduplication_key") or "")
            if not key:
                continue
            group_key = (
                str(atom.get("type") or ""),
                canonical_json(atom.get("scope") or {}),
                key,
            )
            groups.setdefault(group_key, []).append(atom)
        actions: dict[str, dict[str, Any]] = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            ranked = sorted(
                members,
                key=lambda atom: (
                    -len(atom.get("evidence_refs") or []),
                    str(atom.get("created_at") or ""),
                    str(atom.get("id") or ""),
                ),
            )
            keeper = str(ranked[0]["id"])
            for duplicate in ranked[1:]:
                actions[str(duplicate["id"])] = {
                    "action": "archive",
                    "reason": "explicit_proposal_deduplication",
                    "health_status": "merged",
                    "superseded_by": [keeper],
                }
        return actions


    def _pressure_archive_sort_key(
        self,
        atom: Mapping[str, Any],
        edge_degrees: Mapping[str, int],
    ) -> tuple[Any, ...]:
        health_rank = {
            "low_utility": 0,
            "orphaned": 0,
            "stale": 0,
            "confounding": 1,
            "contradicted": 1,
            "healthy": 2,
        }
        atom_ref = str(atom.get("id") or "")
        timestamp = str(
            atom.get("last_accessed")
            or atom.get("updated_at")
            or atom.get("observed_at")
            or atom.get("created_at")
            or ""
        )
        return (
            1 if int(edge_degrees.get(atom_ref, 0) or 0) > 0 else 0,
            health_rank.get(str(atom.get("health_status") or ""), 1),
            float(atom.get("utility", 0.0) or 0.0),
            float(atom.get("salience", 0.0) or 0.0),
            timestamp,
            atom_ref,
        )


    def _decay_action_for_superseded_atom(
        self,
        atom: Mapping[str, Any],
        *,
        superseded_by: Sequence[str],
        policy: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not superseded_by:
            return None
        after = policy.get("archive_superseded_after_seconds", 0)
        if after not in (None, ""):
            ages = [
                age
                for timestamp in (
                    atom.get("last_accessed"),
                    atom.get("updated_at"),
                    atom.get("observed_at"),
                )
                if timestamp
                and (age := self._seconds_since(timestamp)) is not None
            ]
            age = min(ages) if ages else None
            if age is not None and age < int(after):
                return None
        return {
            "action": "archive",
            "reason": "superseded_by_active_atom",
            "health_status": "stale",
            "superseded_by": list(superseded_by),
        }


    def _decay_action_for_atom(
        self,
        atom: Mapping[str, Any],
        *,
        atom_policy: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> dict[str, str] | None:
        if self._timestamp_elapsed(atom_policy.get("expires_at")):
            return {"action": "archive", "reason": "expires_at_elapsed"}
        low_utility_threshold = atom_policy.get(
            "low_utility_threshold", policy.get("low_utility_threshold")
        )
        if low_utility_threshold not in (None, ""):
            try:
                threshold = max(0.0, min(1.0, float(low_utility_threshold)))
            except (TypeError, ValueError):
                threshold = None
            if threshold is not None and float(atom["utility"]) < threshold:
                return {
                    "action": "mark_low_utility",
                    "reason": "utility_below_threshold",
                }
        archive_after = atom_policy.get(
            "archive_after_seconds", policy.get("archive_after_seconds")
        )
        if archive_after not in (None, ""):
            age = self._seconds_since(
                atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
            )
            if age is not None and age >= int(archive_after):
                return {"action": "archive", "reason": "archive_after_elapsed"}
        stale_after = atom_policy.get(
            "mark_stale_after_seconds", policy.get("mark_stale_after_seconds")
        )
        if stale_after not in (None, "") and atom.get("health_status") == "healthy":
            age = self._seconds_since(
                atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
            )
            if age is not None and age >= int(stale_after):
                return {"action": "mark_stale", "reason": "stale_after_elapsed"}
        return None


    def _run_policy_distillation(
        self,
        *,
        policy: Mapping[str, Any],
        scope: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        distillation = dict(policy["distillation"])
        candidates = self._policy_distillation_candidates(
            policy=policy, scope=scope
        )
        min_sources = int(distillation["min_source_atoms"])
        if len(candidates) < min_sources:
            return {
                "status": "skipped",
                "reason": "insufficient_candidates",
                "candidate_count": len(candidates),
                "min_source_atoms": min_sources,
            }
        max_sources = int(distillation["max_source_atoms"])
        cohorts: dict[str, list[dict[str, Any]]] = {}
        for atom in candidates:
            cohorts.setdefault(
                self._policy_distillation_coherence_key(atom), []
            ).append(atom)
        eligible_cohorts = {
            key: atoms for key, atoms in cohorts.items() if len(atoms) >= min_sources
        }
        if not eligible_cohorts:
            return {
                "status": "skipped",
                "reason": "no_coherent_candidate_group",
                "candidate_count": len(candidates),
                "min_source_atoms": min_sources,
                "cohort_counts": {
                    key: len(atoms) for key, atoms in sorted(cohorts.items())
                },
            }
        candidate_rank = {
            str(atom["id"]): index for index, atom in enumerate(candidates)
        }
        coherence_key, coherent_candidates = min(
            eligible_cohorts.items(),
            key=lambda item: (
                min(candidate_rank[str(atom["id"])] for atom in item[1]),
                -len(item[1]),
                item[0],
            ),
        )
        selected = coherent_candidates[:max_sources]
        target_refs = [atom["id"] for atom in selected]
        source_digests = [digest(self._atom_projection(atom)) for atom in selected]
        summary = self._policy_distillation_summary(selected)
        idempotency_key = stable_id(
            "policy_distill",
            {
                "distillation_type": distillation["distillation_type"],
                "target_refs": target_refs,
                "source_digests": source_digests,
                "summary_digest": digest(summary),
                "scope": scope,
            },
        )
        try:
            distilled = self.distill_memories(
                target_refs=target_refs,
                summary=summary,
                scope=scope,
                actor=actor,
                idempotency_key=idempotency_key,
                distillation_type=distillation["distillation_type"],
                archive_sources=distillation["archive_sources"],
                approved_by=distillation["approved_by"],
            )
        except ValidationError as exc:
            if "distilled atom already exists" in str(exc):
                return {
                    "status": "skipped",
                    "reason": "already_distilled",
                    "source_refs": target_refs,
                }
            raise
        return {
            "status": "completed"
            if distilled.get("status") == "distilled"
            else distilled.get("status", "completed"),
            "selected_source_count": len(selected),
            "source_refs": target_refs,
            "coherence_key": coherence_key,
            "coherent_candidate_count": len(coherent_candidates),
            "distilled": distilled if distilled.get("status") == "distilled" else None,
            "result": distilled,
        }


    def _policy_distillation_candidates(
        self,
        *,
        policy: Mapping[str, Any],
        scope: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        distillation = dict(policy["distillation"])
        candidate_types = set(distillation["candidate_types"])
        active_semantics = self.store.list_atoms_filtered(
            types=["semantic"],
            lifecycle_states=["active"],
        )
        covered_sources = covered_source_refs(active_semantics)
        candidates = []
        for atom in self.store.list_atoms_filtered(
            types=sorted(candidate_types) if candidate_types else None,
            lifecycle_states=["active"],
            included_health=["healthy", "low_utility"],
        ):
            if atom.get("deleted"):
                continue
            hints = maintenance_hints_from_atom(atom)
            if str(hints.get("distillation_lane") or "").strip() == "domain_processor":
                continue
            if atom["id"] in covered_sources:
                continue
            if candidate_types and atom["type"] not in candidate_types:
                continue
            if atom["lifecycle_state"] != "active":
                continue
            if atom["health_status"] not in {"healthy", "low_utility"}:
                continue
            if scope and not scope_visible(atom["scope"], scope):
                continue
            candidates.append(atom)
        candidates.sort(
            key=lambda atom: (
                -self._policy_distillation_priority(atom),
                atom.get("observed_at") or atom["created_at"],
                atom["id"],
            )
        )
        return candidates


    def _policy_distillation_coherence_key(
        self, atom: Mapping[str, Any]
    ) -> str:
        """Return a domain-neutral key that prevents mixed-source packets.

        Explicit producer cohorts take precedence. The conservative fallback
        keeps scope, atom type, producer profile, and producer-supplied kind
        together, so unrelated types or processor domains are never combined
        merely because they rank next to each other.
        """

        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        hints = maintenance_hints_from_atom(atom)
        profile = str(
            hints.get("profile")
            or payload.get("memory_profile")
            or payload.get("graph_metadata_profile")
            or ""
        ).strip()
        explicit = str(
            hints.get("consolidation_key")
            or hints.get("cluster_key")
            or hints.get("cohort_key")
            or ""
        ).strip()
        key_material = {
            "mode": "explicit" if explicit else "fallback",
            "scope": dict(atom.get("scope") or {}),
            "atom_type": str(atom.get("type") or ""),
            "profile": profile,
            "kind": str(hints.get("kind") or "").strip(),
            "cohort": explicit,
        }
        return stable_id("distill_cohort", key_material)


    def _policy_distillation_priority(self, atom: Mapping[str, Any]) -> int:
        payload = atom.get("payload", {})
        payload = payload if isinstance(payload, Mapping) else {}
        score = 0
        hints = payload.get("maintenance_hints")
        hints = hints if isinstance(hints, Mapping) else {}
        kind = str(hints.get("kind") or payload.get("kind") or "").lower()
        outcome = str(
            payload.get("outcome") or payload.get("status") or payload.get("result") or ""
        ).lower()
        if kind in {"reflection", "outcome", "evaluation"}:
            score += 6
        priority = hints.get("priority")
        if isinstance(priority, (int, float)) and not isinstance(priority, bool):
            score += max(-4, min(8, int(priority)))
        if hints.get("distill") is True:
            score += 4
        if outcome and outcome not in {"issued", "pending", "planned", "started"}:
            score += 4
        for key in (
            "directive_atom_ref",
            "source_directive_ref",
            "metric_deltas",
            "deltas",
            "lesson",
            "correction",
        ):
            if payload.get(key) not in (None, "", [], {}):
                score += 2
        if self._payload_delta_fields(payload):
            score += 2
        if payload.get("summary") or payload.get("claim"):
            score += 1
        if payload.get("applied_controls") or payload.get("requested_controls"):
            score += 1
        return score


    def _policy_distillation_summary(
        self, atoms: Sequence[Mapping[str, Any]]
    ) -> str:
        type_counts = self._counts(atoms, "type")
        type_phrase = ", ".join(
            f"{count} {atom_type}" for atom_type, count in sorted(type_counts.items())
        )
        highlights = [self._policy_distillation_highlight(atom) for atom in atoms[:6]]
        highlights = [highlight for highlight in highlights if highlight]
        if highlights:
            source_phrase = " Key memories: " + "; ".join(highlights) + "."
        else:
            source_phrase = ""
        return (
            "Automatic AMOS memory policy distilled "
            f"{len(atoms)} source atoms"
            f" ({type_phrase or 'mixed types'}) into a reusable memory packet."
            f"{source_phrase}"
        )


    def _policy_distillation_highlight(self, atom: Mapping[str, Any]) -> str:
        payload = atom.get("payload", {})
        atom_id = str(atom.get("id", "unknown"))
        atom_type = str(atom.get("type", "memory"))
        if not isinstance(payload, Mapping):
            return self._truncate_text(f"{atom_id}: {payload}", 180)
        if payload.get("summary"):
            return self._truncate_text(f"{atom_id}: {payload['summary']}", 180)
        if payload.get("claim"):
            return self._truncate_text(f"{atom_id}: {payload['claim']}", 180)
        chunk = payload.get("chunk", payload.get("target_chunk"))
        outcome = (
            payload.get("outcome") or payload.get("status") or payload.get("result")
        )
        deltas = self._payload_delta_fields(payload)
        controls = payload.get("applied_controls") or payload.get("requested_controls")
        prefix = f"{atom_id}"
        if chunk is not None:
            prefix += f" chunk {chunk}"
        if outcome:
            prefix += f" {outcome}"
        if controls:
            controls_text = canonical_json(controls)
            detail = f"controls {controls_text}"
            if deltas:
                detail = f"deltas {self._format_delta_fields(deltas)}; {detail}"
            return self._truncate_text(f"{prefix}: {detail}", 220)
        if deltas:
            return self._truncate_text(
                f"{prefix}: deltas {self._format_delta_fields(deltas)}",
                220,
            )
        task = payload.get("task")
        action = payload.get("action")
        if task or action or outcome:
            parts = [str(part) for part in (task, action, outcome) if part]
            return self._truncate_text(f"{atom_id}: {'; '.join(parts)}", 180)
        rendered = self._render_atom(atom)["text"]
        return self._truncate_text(f"{atom_id} {atom_type}: {rendered}", 180)


    def _payload_delta_fields(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        deltas: dict[str, Any] = {}
        for key in ("metric_deltas", "deltas"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                deltas.update(
                    {
                        str(delta_key): delta_value
                        for delta_key, delta_value in value.items()
                        if delta_value not in (None, "", [], {})
                    }
                )
        for key, value in payload.items():
            if str(key).startswith("delta_") and value not in (None, "", [], {}):
                deltas[str(key)] = value
        return deltas


    def _format_delta_fields(self, deltas: Mapping[str, Any]) -> str:
        formatted = []
        for key, value in sorted(deltas.items()):
            if isinstance(value, (int, float)):
                formatted.append(f"{key}={value:+.6g}")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)


    def _truncate_text(self, text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."
