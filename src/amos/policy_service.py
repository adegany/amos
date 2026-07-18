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
        self._memory_policy_lock = threading.Lock()
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
                    key == "sqlite_compaction"
                    and isinstance(value, Mapping)
                    and isinstance(cleanup.get("sqlite_compaction"), Mapping)
                ):
                    cleanup["sqlite_compaction"] = {
                        **dict(cleanup["sqlite_compaction"]),
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

        self._memory_policy_running = True
        started_graph_version = self.store.graph_version()
        scope = dict(scope or {})
        results: dict[str, Any] = {}
        target_refs: list[str] = []
        try:
            maintenance = policy["maintenance"]
            if maintenance["enabled"] and maintenance["run_smp"]:
                results["smp"] = self.run_smp_analysis(
                    scope=scope,
                    max_atoms=maintenance["max_smp_atoms"],
                )
            if maintenance["enabled"] and maintenance["run_steward"]:
                results["steward"] = self.run_steward(scope=scope, actor=actor)
                for action in results["steward"].get("actions", []):
                    target_refs.extend(
                        ref
                        for ref in action.get("atom_refs", [])
                        + [action.get("kept"), action.get("archived")]
                        if ref
                    )

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
                    graph_version=policy_event_graph_version
                )
            if maintenance["enabled"] and maintenance["invalidate_packet_cache"]:
                results["packet_cache"] = self._invalidate_packet_cache(
                    graph_version=policy_event_graph_version
                )

            completed_at = utc_now()
            event_payload = {
                "operation": "run_memory_policy",
                "trigger": trigger,
                "force": force,
                "due": due,
                "policy": policy,
                "started_graph_version": started_graph_version,
                "completed_graph_version": policy_event_graph_version,
                "results": self._memory_policy_journal_results(results),
            }
            with self.store.transaction() as conn:
                event = self.store.append_event(
                    conn,
                    event_type="memory_policy_run",
                    actor=actor,
                    payload=event_payload,
                    target_refs=sorted(set(target_refs)),
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
                    self.store.clear_packet_cache(conn)
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
                            cleanup_key == "sqlite_compaction"
                            and isinstance(cleanup_value, Mapping)
                            and isinstance(cleanup.get("sqlite_compaction"), Mapping)
                        ):
                            cleanup["sqlite_compaction"] = {
                                **dict(cleanup["sqlite_compaction"]),
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
        maintenance = normalized["maintenance"]
        for key in [
            "enabled",
            "run_smp",
            "run_steward",
            "rebuild_indexes",
            "rebuild_lsa",
            "invalidate_packet_cache",
        ]:
            maintenance[key] = bool(maintenance.get(key, True))
        maintenance["max_smp_atoms"] = max(
            1,
            int(maintenance.get("max_smp_atoms", 128) or 128),
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
        cleanup["trigger"] = str(cleanup.get("trigger") or "idle")
        if cleanup["trigger"] != "idle":
            cleanup["trigger"] = "idle"
        for key, default in (
            ("idle_after_seconds", 300),
            ("min_interval_seconds", 900),
            ("max_deletions_per_tick", 256),
            ("max_idempotency_compactions_per_tick", 512),
        ):
            cleanup[key] = max(0, int(cleanup.get(key, default) or 0))
        for key, default in (
            ("delete_archived_after_seconds", 604800),
            ("delete_stale_after_seconds", 1209600),
            ("compact_idempotency_after_seconds", 604800),
        ):
            value = cleanup.get(key, default)
            cleanup[key] = None if value in (None, "") else max(0, int(value))
        cleanup["remove_archived_from_hot_index"] = bool(
            cleanup.get("remove_archived_from_hot_index", True)
        )
        cleanup["remove_stale_from_hot_index"] = bool(
            cleanup.get("remove_stale_from_hot_index", True)
        )
        cleanup["protected_types"] = sorted(
            {str(item) for item in cleanup.get("protected_types", [])}
        )
        sqlite_compaction = dict(cleanup.get("sqlite_compaction") or {})
        checkpoint_mode = str(sqlite_compaction.get("checkpoint_mode") or "TRUNCATE").upper()
        if checkpoint_mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            checkpoint_mode = "TRUNCATE"
        cleanup["sqlite_compaction"] = {
            "checkpoint_wal": bool(sqlite_compaction.get("checkpoint_wal", True)),
            "checkpoint_mode": checkpoint_mode,
            "vacuum_enabled": bool(sqlite_compaction.get("vacuum_enabled", True)),
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
            "graph_version": result.get("graph_version"),
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
        checkpoint = dict(result.get("checkpoint") or {})
        vacuum = dict(result.get("vacuum") or {})
        return {
            "status": result.get("status"),
            "deleted_atom_count": int(result.get("deleted_atom_count", 0) or 0),
            "deleted_atom_refs": self._bounded_refs(result.get("deleted_atom_refs", [])),
            "index_pruned_rows": int(index_prune.get("rows", 0) or 0),
            "idempotency_compacted_rows": int(idempotency.get("rows", 0) or 0),
            "idempotency_saved_bytes": int(idempotency.get("saved_bytes", 0) or 0),
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
        pressure_mode = self._capacity_pressure_mode()
        if (
            schedule.get("run_on_pressure", True)
            and pressure_mode in {"orange", "red"}
            and graph_delta > 0
        ):
            reasons.append(f"capacity_pressure:{pressure_mode}")
        storage_cleanup = self._storage_cleanup_due(
            policy.get("storage_cleanup", {}), state, force=force
        )
        if storage_cleanup["due"] and "force" not in reasons:
            reasons.append("storage_cleanup_idle")
        return {
            "due": bool(reasons),
            "reasons": reasons,
            "graph_version": graph_version,
            "last_graph_version": last_graph_version,
            "graph_delta": graph_delta,
            "elapsed_seconds": elapsed_seconds,
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
        isolated = [
            atom
            for atom in active_atoms
            if int(edge_degrees.get(str(atom["id"]), 0)) == 0
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
        known_refs = active_refs | {str(atom["id"]) for atom in proposed_atoms} | {
            str(record.get("evidence_id") or "") for record in evidence_records
        }
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
        required_with_headroom = int(
            math.ceil(hot_count / max(0.1, 1.0 - capacity_headroom_ratio))
        )
        recommended_target = next(
            (target for target in capacity_targets if target >= required_with_headroom),
            capacity_targets[-1],
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
            },
            "isolated_proposed_atoms": {
                "count": len(isolated_proposed),
                "expected_dormant": True,
                "sample_refs": sorted(
                    str(atom["id"]) for atom in isolated_proposed
                )[:10],
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
    ) -> dict[str, Any]:
        if not cleanup.get("enabled", True) and not force:
            return {"due": False, "reason": "storage_cleanup_disabled"}
        last_foreground = (
            self.store.get_meta("last_foreground_activity_at")
            or state.get("last_foreground_activity_at")
        )
        idle_elapsed = self._seconds_since(last_foreground)
        idle_after = int(cleanup.get("idle_after_seconds", 300) or 0)
        if idle_elapsed is not None and idle_elapsed < idle_after and not force:
            return {
                "due": False,
                "reason": "foreground_activity_recent",
                "idle_elapsed_seconds": idle_elapsed,
                "idle_after_seconds": idle_after,
                "last_foreground_activity_at": last_foreground,
            }
        last_cleanup = (
            self.store.get_meta("last_storage_cleanup_at")
            or state.get("last_storage_cleanup_at")
        )
        cleanup_elapsed = self._seconds_since(last_cleanup)
        min_interval = int(cleanup.get("min_interval_seconds", 900) or 0)
        if cleanup_elapsed is not None and cleanup_elapsed < min_interval and not force:
            return {
                "due": False,
                "reason": "cleanup_interval_not_elapsed",
                "elapsed_since_cleanup_seconds": cleanup_elapsed,
                "min_interval_seconds": min_interval,
                "last_storage_cleanup_at": last_cleanup,
            }
        return {
            "due": True,
            "reason": "force" if force else "idle_interval_elapsed",
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
        protected_types = {str(item) for item in cleanup.get("protected_types", [])}
        max_deletions = max(0, int(cleanup.get("max_deletions_per_tick", 256) or 0))
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        tombstones: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        deleted_refs: list[str] = []
        index_lifecycle_states = (
            ["archived"] if cleanup.get("remove_archived_from_hot_index", True) else []
        )
        index_health_statuses = (
            ["stale"] if cleanup.get("remove_stale_from_hot_index", True) else []
        )
        compact_after = cleanup.get("compact_idempotency_after_seconds")
        with self.store.transaction() as conn:
            index_prune = self.store.prune_atom_text_index(
                conn,
                lifecycle_states=index_lifecycle_states,
                health_statuses=index_health_statuses,
            )
            atoms = self.store.list_atoms_filtered(
                include_deleted=False,
                lifecycle_states=["active", "archived", "proposed"],
            )
            for atom in atoms:
                if len(actions) >= max_deletions:
                    break
                if not maintenance_scope_visible(atom["scope"], scope):
                    continue
                if atom["type"] in protected_types:
                    continue
                reason = self._storage_deletion_reason(atom, cleanup)
                if reason is None:
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
                updated = normalize_atom(
                    self._attach_search_index(updated), require_id=True
                )
                updated["deleted"] = 1
                tombstone = self.store.insert_tombstone(
                    conn,
                    target_ref=atom["id"],
                    content_digest=self._memory_identity_digest(atom),
                    recreation_policy="block_recreate",
                    reason=reason,
                )
                deleted_edges = self.store.mark_edges_deleted_for_ref(conn, atom["id"])
                self.store.replace_atom(conn, updated)
                projected_atoms.append(updated)
                projected_edges.extend(deleted_edges)
                tombstones.append(tombstone)
                deleted_refs.append(atom["id"])
                actions.append(
                    {
                        "atom_ref": atom["id"],
                        "action": "delete",
                        "reason": reason,
                        "lifecycle_state_before": atom["lifecycle_state"],
                        "health_status_before": atom["health_status"],
                    }
                )
            if compact_after is None:
                idempotency = {
                    "status": "skipped",
                    "reason": "idempotency_compaction_disabled",
                    "rows": 0,
                }
            else:
                idempotency = self.store.compact_idempotency_responses(
                    conn,
                    older_than=self._iso_before_seconds(int(compact_after)),
                    max_rows=int(
                        cleanup.get("max_idempotency_compactions_per_tick", 512) or 0
                    ),
                )
            if actions or index_prune.get("rows") or idempotency.get("rows"):
                event = self.store.append_event(
                    conn,
                    event_type="storage_cleanup_run",
                    actor=actor,
                    payload={
                        "operation": "run_storage_cleanup",
                        "policy": dict(cleanup),
                        "due": dict(due),
                        "actions": actions,
                        "index_prune": index_prune,
                        "idempotency": idempotency,
                        "projected_atoms": projected_atoms,
                        "projected_edges": projected_edges,
                        "tombstones": tombstones,
                    },
                    target_refs=deleted_refs,
                )
                self.store.clear_packet_cache(conn)
            else:
                event = None
            self.store._set_meta(conn, "last_storage_cleanup_at", now)
        sqlite_compaction = dict(cleanup.get("sqlite_compaction") or {})
        checkpoint = {"status": "skipped", "reason": "checkpoint_disabled"}
        if sqlite_compaction.get("checkpoint_wal", True):
            try:
                checkpoint = self.store.checkpoint_wal(
                    mode=str(sqlite_compaction.get("checkpoint_mode") or "TRUNCATE")
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
                    mode=str(sqlite_compaction.get("checkpoint_mode") or "TRUNCATE")
                )
            except Exception as exc:
                checkpoint_after_vacuum = {"status": "error", "error": str(exc)}
        return {
            "status": "completed",
            "due": dict(due),
            "index_prune": index_prune,
            "deleted_atom_count": len(actions),
            "deleted_atom_refs": deleted_refs,
            "idempotency": idempotency,
            "checkpoint": checkpoint,
            "vacuum": vacuum,
            "checkpoint_after_vacuum": checkpoint_after_vacuum,
            "event": event,
        }


    def _storage_deletion_reason(
        self, atom: Mapping[str, Any], cleanup: Mapping[str, Any]
    ) -> str | None:
        if atom.get("deleted"):
            return None
        updated_age = self._seconds_since(
            atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
        )
        archived_after = cleanup.get("delete_archived_after_seconds")
        if (
            archived_after is not None
            and atom.get("lifecycle_state") == "archived"
            and updated_age is not None
            and updated_age >= int(archived_after)
        ):
            return "storage_cleanup_archived_retention_elapsed"
        stale_after = cleanup.get("delete_stale_after_seconds")
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
        if not sqlite_compaction.get("vacuum_enabled", True):
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
        superseded_refs = (
            self._active_superseded_refs()
            if decay.get("archive_superseded", True)
            else {}
        )
        atoms_by_ref = {
            atom["id"]: atom
            for atom in self.store.list_atoms_filtered(
                lifecycle_states=["active", "proposed"]
            )
        }
        for atom in self.store.list_atoms_filtered(
            lifecycle_states=["active", "proposed"],
            atom_ids=sorted(superseded_refs),
        ):
            atoms_by_ref[atom["id"]] = atom
        atoms = list(atoms_by_ref.values())
        planned: list[tuple[dict[str, Any], dict[str, Any]]] = []
        planned_archives: set[str] = set()
        duplicate_actions = self._proposed_duplicate_archive_actions(atoms)
        for atom in atoms:
            if not maintenance_scope_visible(atom["scope"], scope):
                continue
            atom_policy = (
                dict(atom.get("decay_policy") or {})
                if isinstance(atom.get("decay_policy"), Mapping)
                else {}
            )
            explicit_atom_policy = self._has_explicit_atom_decay_policy(atom_policy)
            if atom_policy.get("enabled") is False:
                continue
            superseded_action = duplicate_actions.get(str(atom["id"]))
            if superseded_action is None:
                superseded_action = self._decay_action_for_superseded_atom(
                    atom,
                    superseded_by=superseded_refs.get(atom["id"], []),
                    policy=decay,
                )
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
        hot_count_after_rules = hot_count_before - len(planned_archives)
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
            and self._pressure_archive_eligible(
                atom, decay=decay, scope=scope, lifecycle_state="proposed"
            )
        ]
        active_pressure_candidates = [
            atom
            for atom in atoms
            if str(atom["id"]) not in planned_archives
            and self._pressure_archive_eligible(
                atom, decay=decay, scope=scope, lifecycle_state="active"
            )
        ]
        pressure_required = bool(
            total_pressure_needed or active_pressure_needed or proposed_pressure_needed
        )
        edge_degrees = self.store.edge_degree_counts() if pressure_required else {}
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

        with self.store.transaction() as conn:
            for atom, action in planned:
                atom_policy = (
                    dict(atom.get("decay_policy") or {})
                    if isinstance(atom.get("decay_policy"), Mapping)
                    else {}
                )
                changed = dict(atom)
                changed["version"] = int(changed["version"]) + 1
                changed["updated_at"] = now
                if action["action"] == "archive":
                    changed["lifecycle_state"] = "archived"
                    changed["health_status"] = action.get("health_status", "stale")
                    projected_edges.extend(
                        self.store.mark_edges_deleted_for_ref(conn, str(atom["id"]))
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
                self.store.replace_atom(conn, changed)
                projected_atoms.append(changed)
                actions.append(
                    {
                        "atom_ref": changed["id"],
                        "action": action["action"],
                        "reason": action["reason"],
                        **(
                            {"superseded_by": action["superseded_by"]}
                            if action.get("superseded_by")
                            else {}
                        ),
                        "health_status": changed["health_status"],
                        "lifecycle_state": changed["lifecycle_state"],
                    }
                )
            if actions:
                event = self.store.append_event(
                    conn,
                    event_type="decay_policy_applied",
                    actor=actor,
                    payload={
                        "operation": "run_decay_policy",
                        "policy": dict(decay),
                        "actions": actions,
                        "projected_atoms": projected_atoms,
                        "projected_edges": projected_edges,
                    },
                    target_refs=[action["atom_ref"] for action in actions],
                )
                self.store.clear_packet_cache(conn)
            else:
                event = None
        return {
            "status": "completed",
            "action_count": len(actions),
            "actions": actions,
            "projected_edges": projected_edges,
            "pressure": pressure,
            "event": event,
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
        return dict(retention) if isinstance(retention, Mapping) else {}


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
            age = self._seconds_since(
                atom.get("last_accessed") or atom.get("updated_at") or atom.get("observed_at")
            )
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
        selected = candidates[:max_sources]
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
        covered_sources: set[str] = set()
        for atom in self.store.list_atoms_filtered(
            types=["semantic"],
            lifecycle_states=["active"],
        ):
            if atom.get("deleted") or atom["type"] != "semantic":
                continue
            if atom.get("lifecycle_state") != "active":
                continue
            payload = atom.get("payload", {})
            if payload.get("created_by") != "svc:memory_policy":
                continue
            if payload.get("distillation_type") != distillation["distillation_type"]:
                continue
            covered_sources.update(str(ref) for ref in payload.get("source_refs", []))
        candidates = []
        for atom in self.store.list_atoms_filtered(
            types=sorted(candidate_types) if candidate_types else None,
            lifecycle_states=["active"],
            included_health=["healthy", "low_utility"],
        ):
            if atom.get("deleted"):
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
