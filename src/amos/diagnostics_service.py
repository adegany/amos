"""DiagnosticsService implementation for the AMOS service facade."""

import math

from ._service_support import Any, digest
from .journal_replay import empty_replay_state, replay_events
from .store import migrated_edge_derivation


class DiagnosticsService:
    def __init__(self, store: Any, maintenance: Any, capacity: Any, graph: Any):
        self.store = store
        self.run_memory_policy = maintenance.run_memory_policy
        self.memory_policy = maintenance.memory_policy
        self.memory_policy_status = maintenance.memory_policy_status
        self._memory_quality_diagnostics = maintenance._memory_quality_diagnostics
        self._capacity_budget = capacity._capacity_budget
        self._capacity_pressure_mode = capacity._capacity_pressure_mode
        self._atom_projection = graph._atom_projection

    def health_memory(self, *, run_policy: bool = True) -> dict[str, Any]:
        policy_tick = (
            self.run_memory_policy(trigger="health_memory")
            if run_policy
            else {
                "status": "skipped",
                "reason": "policy_not_run_for_health",
                "trigger": "health_memory",
                "graph_version": self.store.graph_version(),
            }
        )
        with self.store.read_snapshot():
            graph_version = self.store.graph_version()
            indexes = self.store.list_derived_index_metadata()
            by_type = self.store.atom_counts_by("type")
            by_health = self.store.atom_counts_by("health_status")
            by_lifecycle = self.store.atom_counts_by("lifecycle_state")
            return {
                "graph_version": graph_version,
                "journal_events": self.store.event_count(),
                "memory_heads": len(self.store.list_memory_heads()),
                "atoms": self.store.atom_count(),
                "edges": self.store.edge_count(),
                "by_type": by_type,
                "by_health": by_health,
                "by_lifecycle": by_lifecycle,
                "journal_head": self.store.last_event_hash(),
                "projection_lag": 0,
                "index_freshness": {
                    index["index_name"]: {
                        "graph_version": index["graph_version"],
                        "freshness": (
                            index["freshness"]
                            if int(index["graph_version"]) == graph_version
                            else "stale"
                        ),
                        "rebuilt_at": index["rebuilt_at"],
                    }
                    for index in indexes
                },
                "retrieval_outcomes": self.store.retrieval_outcome_count(),
                "deletion_residuals": {
                    "offline_backup_residual_window_days": 30,
                    "hot_packet_cache_policy": (
                        "graph_version_keyed_bounded_retirement_with_delete_purge"
                    ),
                },
                "quality": self._memory_quality_diagnostics(
                    policy=self.memory_policy(),
                    indexes=indexes,
                ),
                "memory_policy": self.memory_policy_status(),
                "concurrency": {
                    "read_consistency": "revision_pinned_wal_snapshot",
                    "writer_admission": self.store.writer_status(),
                },
                "last_policy_tick": policy_tick,
            }

    def health_memory_inventory(
        self, *, include_integrity: bool = True
    ) -> dict[str, Any]:
        """Return bounded counters, optionally including the deep integrity scan."""

        policy = self.memory_policy()
        decay = dict(policy.get("decay") or {})
        schedule = dict(policy.get("schedule") or {})
        with self.store.read_snapshot():
            graph_version = self.store.graph_version()
            indexes = self.store.list_derived_index_metadata()
            by_type = self.store.atom_counts_by("type")
            by_lifecycle = self.store.atom_counts_by("lifecycle_state")
            integrity = (
                self.store.memory_integrity_summary()
                if include_integrity
                else {
                    "active_superseded_atoms": 0,
                    "isolated_active_atoms": 0,
                    "exact_evidence_refs": 0,
                    "mistyped_atom_refs": 0,
                    "unresolved_refs": 0,
                }
            )
            active_count = int(by_lifecycle.get("active") or 0)
            proposed_count = int(by_lifecycle.get("proposed") or 0)
            hot_count = active_count + proposed_count
            hot_limit = int(decay.get("max_atoms", 256) or 256)
            active_limit = int(
                decay.get("max_active_atoms", hot_limit) or hot_limit
            )
            proposed_limit = int(
                decay.get("max_proposed_atoms", hot_limit) or hot_limit
            )
            headroom_ratio = float(
                decay.get("capacity_headroom_ratio", 0.2) or 0.0
            )
            utilization = hot_count / max(1, hot_limit)
            near_limit = utilization >= 1.0 - headroom_ratio
            targets = sorted({
                max(1, int(item))
                for item in decay.get(
                    "capacity_assessment_targets",
                    [256, 512, 768],
                )
                if item not in (None, "")
            } | {hot_limit})
            required_with_headroom = max(
                1,
                int(math.ceil(hot_count / max(0.1, 1.0 - headroom_ratio))),
            )
            if not targets or targets[-1] < required_with_headroom:
                targets.append(required_with_headroom)
            recommended_target = next(
                target for target in targets if target >= required_with_headroom
            )
            index_lag = {
                str(index["index_name"]): max(
                    0,
                    graph_version - int(index.get("graph_version", 0) or 0),
                )
                for index in indexes
            }
            max_index_lag = max(index_lag.values(), default=0)
            maintenance_every = int(
                schedule.get("every_graph_versions", 25) or 25
            )
            warnings: list[str] = []
            if not include_integrity:
                warnings.append("integrity_diagnostics_refreshing")
            if hot_count > hot_limit:
                warnings.append("active_atom_count_exceeds_decay_max_atoms")
            if near_limit:
                warnings.append("active_atom_capacity_headroom_low")
            if active_count >= active_limit:
                warnings.append("lifecycle_active_atom_limit_reached")
            if proposed_count >= proposed_limit:
                warnings.append("proposed_atom_limit_reached")
            if integrity["active_superseded_atoms"]:
                warnings.append("active_superseded_atoms_present")
            if integrity["isolated_active_atoms"]:
                warnings.append("isolated_active_atoms_present")
            if integrity["mistyped_atom_refs"]:
                warnings.append("atom_ids_present_in_evidence_refs")
            if integrity["unresolved_refs"]:
                warnings.append("unresolved_evidence_refs_present")
            if max_index_lag >= maintenance_every:
                warnings.append("derived_index_lag_exceeds_schedule")
            return {
                "profile": "amos.memory-inventory-health.v1",
                "diagnostic_scope": (
                    "operational_inventory"
                    if include_integrity
                    else "operational_inventory_without_integrity"
                ),
                "graph_version": graph_version,
                "journal_events": self.store.event_count(),
                "memory_heads": len(self.store.list_memory_heads()),
                "atoms": self.store.atom_count(),
                "edges": self.store.edge_count(),
                "by_type": by_type,
                "by_lifecycle": by_lifecycle,
                "projection_lag": 0,
                "quality": {
                    "status": (
                        "refreshing"
                        if not include_integrity
                        else "warning"
                        if warnings
                        else "ok"
                    ),
                    "integrity_exact": bool(include_integrity),
                    "warnings": warnings,
                    "hot_atom_count": hot_count,
                    "hot_atom_limit": hot_limit,
                    "capacity_assessment": {
                        "configured_target": hot_limit,
                        "active_count": hot_count,
                        "headroom_atoms": max(0, hot_limit - hot_count),
                        "utilization": round(utilization, 4),
                        "headroom_ratio_target": headroom_ratio,
                        "near_limit": near_limit,
                        "recommended_target": recommended_target,
                    },
                    "active_superseded_atoms": {
                        "count": integrity["active_superseded_atoms"],
                    },
                    "isolated_active_atoms": {
                        "count": integrity["isolated_active_atoms"],
                    },
                    "reference_contract": {
                        "exact_evidence_refs": integrity[
                            "exact_evidence_refs"
                        ],
                        "mistyped_atom_refs": integrity["mistyped_atom_refs"],
                        "unresolved_refs": integrity["unresolved_refs"],
                    },
                    "derived_index_lag": {
                        "max_graph_delta": max_index_lag,
                        "by_index": index_lag,
                    },
                },
            }
    def health_capacity(self) -> dict[str, Any]:
        path = self.store.path
        usage = self.store.storage_usage()
        with self.store.read_snapshot():
            budget = self._capacity_budget()
            used_size_bytes = int(
                usage.get("used_size_bytes", usage["managed_size_bytes"])
            )
            pressure_mode = self._capacity_pressure_mode(
                size_bytes=used_size_bytes, budget=budget
            )
            return {
                "store": getattr(self.store, "backend_name", "unknown"),
                "path": str(path),
                # Capacity pressure follows live used bytes. Physical allocation
                # and reusable freelist space remain explicit alongside it.
                "size_bytes": used_size_bytes,
                **usage,
                "sqlite_space": self.store.sqlite_space_status(),
                "journal_storage": self.store.journal_storage_status(),
                "capacity_budget": budget,
                "pressure_mode": pressure_mode,
                "graph_version": self.store.graph_version(),
                "concurrency": {
                    "read_consistency": "revision_pinned_wal_snapshot",
                    "writer_admission": self.store.writer_status(),
                },
                "degradation": {
                    "vector_index_available": False,
                    "external_object_store_available": False,
                    "pressure_degraded": pressure_mode in {"orange", "red"},
                },
            }


    def verify_journal_chain(self) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self._verify_journal_chain()

    def _verify_journal_chain(self) -> dict[str, Any]:
        previous = "genesis"
        expected_graph_version = 1
        failures: list[dict[str, Any]] = []
        retained_event_count = 0
        digest_only_event_count = 0
        compact_receipt_count = 0

        def verify_event(event: dict[str, Any]) -> None:
            nonlocal previous, expected_graph_version
            graph_version = int(event["graph_version"])
            if graph_version != expected_graph_version:
                failures.append(
                    {
                        "event_id": event["event_id"],
                        "reason": "graph_version_gap",
                        "expected": expected_graph_version,
                        "actual": graph_version,
                    }
                )
            if event["previous_event_hash"] != previous:
                failures.append(
                    {
                        "event_id": event["event_id"],
                        "reason": "previous_event_hash_mismatch",
                        "expected": previous,
                        "actual": event["previous_event_hash"],
                    }
                )
            event_without_checksum = dict(event)
            checksum = event_without_checksum.pop("checksum")
            if digest(event_without_checksum) != checksum:
                failures.append(
                    {
                        "event_id": event["event_id"],
                        "reason": "checksum_mismatch",
                    }
                )
            previous = event["checksum"]
            expected_graph_version = graph_version + 1

        segments = self.store.list_journal_segments()
        for segment in segments:
            start = int(segment["start_graph_version"])
            end = int(segment["end_graph_version"])
            count = int(segment["event_count"])
            if start != expected_graph_version:
                failures.append(
                    {
                        "segment_id": segment["segment_id"],
                        "reason": "segment_graph_version_gap",
                        "expected": expected_graph_version,
                        "actual": start,
                    }
                )
            if str(segment["first_previous_event_hash"]) != previous:
                failures.append(
                    {
                        "segment_id": segment["segment_id"],
                        "reason": "segment_previous_hash_mismatch",
                        "expected": previous,
                        "actual": segment["first_previous_event_hash"],
                    }
                )
            if int(segment.get("payload_retained", 1) or 0):
                try:
                    events = self.store.journal_segment_events(
                        str(segment["segment_id"])
                    ) or []
                except Exception as exc:
                    failures.append(
                        {
                            "segment_id": segment["segment_id"],
                            "reason": "segment_payload_invalid",
                            "error": str(exc),
                        }
                    )
                    events = []
                if len(events) != count:
                    failures.append(
                        {
                            "segment_id": segment["segment_id"],
                            "reason": "segment_event_count_mismatch",
                            "expected": count,
                            "actual": len(events),
                        }
                    )
                for event in events:
                    verify_event(event)
                if events:
                    if str(events[0]["event_id"]) != str(segment["first_event_id"]):
                        failures.append(
                            {
                                "segment_id": segment["segment_id"],
                                "reason": "segment_first_event_mismatch",
                            }
                        )
                    if str(events[-1]["event_id"]) != str(segment["last_event_id"]):
                        failures.append(
                            {
                                "segment_id": segment["segment_id"],
                                "reason": "segment_last_event_mismatch",
                            }
                        )
                    if previous != str(segment["last_event_hash"]):
                        failures.append(
                            {
                                "segment_id": segment["segment_id"],
                                "reason": "segment_last_hash_mismatch",
                            }
                        )
                retained_event_count += count
            else:
                # The detailed payload was deliberately shredded after the
                # snapshot covered it. Receipts preserve exact event refs and
                # checksum links without pretending the discarded payload can
                # still be recomputed.
                digest_only_event_count += count
                try:
                    receipts = self.store.list_journal_event_receipts(
                        str(segment["segment_id"])
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "segment_id": segment["segment_id"],
                            "reason": "segment_compact_receipt_invalid",
                            "error": str(exc),
                        }
                    )
                    receipts = []
                if len(receipts) != count:
                    failures.append(
                        {
                            "segment_id": segment["segment_id"],
                            "reason": "segment_compact_receipt_count_mismatch",
                            "expected": count,
                            "actual": len(receipts),
                        }
                    )
                for receipt in receipts:
                    graph_version = int(receipt["graph_version"])
                    if graph_version != expected_graph_version:
                        failures.append(
                            {
                                "event_id": receipt["event_id"],
                                "reason": "compact_receipt_graph_version_gap",
                                "expected": expected_graph_version,
                                "actual": graph_version,
                            }
                        )
                    if str(receipt["previous_event_hash"]) != previous:
                        failures.append(
                            {
                                "event_id": receipt["event_id"],
                                "reason": "compact_receipt_chain_mismatch",
                                "expected": previous,
                                "actual": receipt["previous_event_hash"],
                            }
                        )
                    previous = str(receipt["checksum"])
                    expected_graph_version = graph_version + 1
                compact_receipt_count += len(receipts)
                if receipts:
                    if str(receipts[0]["event_id"]) != str(
                        segment["first_event_id"]
                    ) or str(receipts[-1]["event_id"]) != str(
                        segment["last_event_id"]
                    ):
                        failures.append(
                            {
                                "segment_id": segment["segment_id"],
                                "reason": "segment_compact_receipt_boundary_mismatch",
                            }
                        )
                    if previous != str(segment["last_event_hash"]):
                        failures.append(
                            {
                                "segment_id": segment["segment_id"],
                                "reason": "segment_compact_receipt_hash_mismatch",
                            }
                        )
                else:
                    previous = str(segment["last_event_hash"])
                    expected_graph_version = end + 1

        live_events = self.store.list_live_events()
        for event in live_events:
            verify_event(event)

        try:
            snapshot = self.store.latest_journal_snapshot()
        except Exception as exc:
            snapshot = None
            failures.append(
                {
                    "reason": "journal_snapshot_invalid",
                    "error": str(exc),
                }
            )
        if snapshot is not None:
            through = int(snapshot["through_graph_version"])
            covering = next(
                (
                    segment
                    for segment in segments
                    if int(segment["end_graph_version"]) == through
                ),
                None,
            )
            if (
                covering is None
                or str(covering["last_event_hash"])
                != str(snapshot["through_event_hash"])
                or str(covering["last_event_id"])
                != str(snapshot["through_event_id"])
            ):
                failures.append(
                    {
                        "snapshot_id": snapshot["snapshot_id"],
                        "reason": "snapshot_segment_boundary_mismatch",
                    }
                )

        journal_head = self.store.last_event_hash()
        if previous != journal_head:
            failures.append(
                {
                    "reason": "journal_head_mismatch",
                    "expected": previous,
                    "actual": journal_head,
                }
            )
        event_count = (
            retained_event_count
            + digest_only_event_count
            + len(live_events)
        )
        return {
            "status": "ok" if not failures else "failed",
            "event_count": event_count,
            "fully_verified_event_count": retained_event_count + len(live_events),
            "digest_only_event_count": digest_only_event_count,
            "compact_receipt_count": compact_receipt_count,
            "verification_scope": (
                "retained_payloads_plus_compacted_boundaries"
                if digest_only_event_count
                else "full_event_payloads"
            ),
            "graph_version": self.store.graph_version(),
            "journal_head": journal_head,
            "failures": failures,
        }


    def replay_graph(self) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self._replay_graph()

    def _replay_graph(self) -> dict[str, Any]:
        snapshot = self.store.latest_journal_snapshot()
        through_graph_version = int(
            (snapshot or {}).get("through_graph_version", 0) or 0
        )
        replayed = replay_events(
            self.store.list_live_events(
                after_graph_version=through_graph_version
            ),
            initial_state=(snapshot or {}).get("state") or empty_replay_state(),
            migrated_edge_derivation=migrated_edge_derivation,
        )
        return {
            "graph_version": self.store.graph_version(),
            "snapshot_graph_version": through_graph_version,
            "atoms": replayed["atoms"],
            "edges": replayed["edges"],
            "heads": replayed["heads"],
            "tombstones": replayed["tombstones"],
        }


    def verify_replay(self) -> dict[str, Any]:
        with self.store.read_snapshot():
            return self._verify_replay()

    def _verify_replay(self) -> dict[str, Any]:
        replayed = self._replay_graph()
        stored_atoms = {
            atom["id"]: atom
            for atom in self.store.list_atoms()
            if not atom.get("deleted")
        }
        replayed_atoms = replayed["atoms"]
        replayed_edges = replayed["edges"]
        replayed_heads = replayed["heads"]
        missing = sorted(set(stored_atoms) - set(replayed_atoms))
        unexpected = sorted(set(replayed_atoms) - set(stored_atoms))
        mismatched = []
        for atom_id in sorted(set(stored_atoms).intersection(replayed_atoms)):
            if digest(self._atom_projection(stored_atoms[atom_id])) != digest(
                self._atom_projection(replayed_atoms[atom_id])
            ):
                mismatched.append(atom_id)
        stored_edges = {
            edge["edge_id"]: edge
            for edge in self.store.list_edges()
            if not edge.get("deleted")
        }
        missing_edges = sorted(set(stored_edges) - set(replayed_edges))
        unexpected_edges = sorted(set(replayed_edges) - set(stored_edges))
        mismatched_edges = []
        for edge_id in sorted(set(stored_edges).intersection(replayed_edges)):
            if digest(stored_edges[edge_id]) != digest(replayed_edges[edge_id]):
                mismatched_edges.append(edge_id)
        stored_heads = {
            (
                f"{digest(head.get('scope') or {})}:"
                f"{head.get('series_kind')}:{head.get('series_id')}"
            ): head
            for head in self.store.list_memory_heads()
        }
        missing_heads = sorted(set(stored_heads) - set(replayed_heads))
        unexpected_heads = sorted(set(replayed_heads) - set(stored_heads))
        mismatched_heads = [
            key
            for key in sorted(set(stored_heads).intersection(replayed_heads))
            if digest(stored_heads[key]) != digest(replayed_heads[key])
        ]
        return {
            "status": "ok"
            if not missing
            and not unexpected
            and not mismatched
            and not missing_edges
            and not unexpected_edges
            and not mismatched_edges
            and not missing_heads
            and not unexpected_heads
            and not mismatched_heads
            else "failed",
            "graph_version": self.store.graph_version(),
            "missing_in_replay": missing,
            "unexpected_in_replay": unexpected,
            "mismatched_atoms": mismatched,
            "missing_edges_in_replay": missing_edges,
            "unexpected_edges_in_replay": unexpected_edges,
            "mismatched_edges": mismatched_edges,
            "missing_heads_in_replay": missing_heads,
            "unexpected_heads_in_replay": unexpected_heads,
            "mismatched_heads": mismatched_heads,
            "replayed_atom_count": len(replayed_atoms),
            "stored_atom_count": len(stored_atoms),
            "replayed_edge_count": len(replayed_edges),
            "stored_edge_count": len(stored_edges),
            "replayed_head_count": len(replayed_heads),
            "stored_head_count": len(stored_heads),
        }

    def verify_integrity(self) -> dict[str, Any]:
        """Verify journal and replay against the same canonical read snapshot."""

        with self.store.read_snapshot():
            return {
                "journal": self._verify_journal_chain(),
                "replay": self._verify_replay(),
            }
