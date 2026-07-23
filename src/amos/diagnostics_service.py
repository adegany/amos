"""DiagnosticsService implementation for the AMOS service facade."""

from ._service_support import Any, digest
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
        indexes = self.store.list_derived_index_metadata()
        by_type = self.store.atom_counts_by("type")
        by_health = self.store.atom_counts_by("health_status")
        by_lifecycle = self.store.atom_counts_by("lifecycle_state")
        return {
            "graph_version": self.store.graph_version(),
            "journal_events": self.store.event_count(),
            "atoms": self.store.atom_count(),
            "by_type": by_type,
            "by_health": by_health,
            "by_lifecycle": by_lifecycle,
            "journal_head": self.store.last_event_hash(),
            "projection_lag": 0,
            "index_freshness": {
                index["index_name"]: {
                    "graph_version": index["graph_version"],
                    "freshness": index["freshness"],
                    "rebuilt_at": index["rebuilt_at"],
                }
                for index in indexes
            },
            "retrieval_outcomes": self.store.retrieval_outcome_count(),
            "deletion_residuals": {
                "offline_backup_residual_window_days": 30,
                "hot_packet_cache_policy": "purged_on_canonical_mutation",
            },
            "quality": self._memory_quality_diagnostics(
                policy=self.memory_policy(),
                indexes=indexes,
            ),
            "memory_policy": self.memory_policy_status(),
            "last_policy_tick": policy_tick,
        }


    def health_capacity(self) -> dict[str, Any]:
        path = self.store.path
        size_bytes = path.stat().st_size if path.exists() and str(path) != ":memory:" else 0
        budget = self._capacity_budget()
        pressure_mode = self._capacity_pressure_mode(size_bytes=size_bytes, budget=budget)
        return {
            "store": getattr(self.store, "backend_name", "unknown"),
            "path": str(path),
            "size_bytes": size_bytes,
            "capacity_budget": budget,
            "pressure_mode": pressure_mode,
            "graph_version": self.store.graph_version(),
            "degradation": {
                "vector_index_available": False,
                "external_object_store_available": False,
                "pressure_degraded": pressure_mode in {"orange", "red"},
            },
        }


    def verify_journal_chain(self) -> dict[str, Any]:
        events = self.store.list_events()
        previous = "genesis"
        failures = []
        for event in events:
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
        return {
            "status": "ok" if not failures else "failed",
            "event_count": len(events),
            "graph_version": self.store.graph_version(),
            "journal_head": self.store.last_event_hash(),
            "failures": failures,
        }


    def replay_graph(self) -> dict[str, Any]:
        atoms: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        tombstones: dict[str, dict[str, Any]] = {}

        def replay_edge_projection(edge: dict[str, Any]) -> dict[str, Any]:
            projected = dict(edge)
            derivation = projected.get("derivation")
            if not isinstance(derivation, dict) or not derivation:
                projected["derivation"] = migrated_edge_derivation(
                    str(projected.get("relation") or "")
                )
            return projected

        def replay_legacy_retrieval_edge_feedback(
            payload: dict[str, Any],
        ) -> None:
            """Replay feedback events written before full edges were journaled."""

            if payload.get("projected_edges"):
                return
            feedback = payload.get("feedback") or {}
            summaries = feedback.get("updated_edges") or []
            if not summaries:
                return
            projected_atoms = payload.get("projected_atoms") or []
            timestamp = None
            label = None
            for atom in projected_atoms:
                telemetry = (atom.get("decay_policy") or {}).get(
                    "retrieval_telemetry"
                ) or {}
                timestamp = telemetry.get("last_outcome_at") or atom.get("updated_at")
                label = telemetry.get("last_outcome_label")
                if timestamp:
                    break
            for summary in summaries:
                edge_id = str(summary.get("edge_id") or "")
                prior = edges.get(edge_id)
                if not edge_id or prior is None:
                    continue
                changed = dict(prior)
                derivation = dict(changed.get("derivation") or {})
                telemetry = dict(derivation.get("retrieval_telemetry") or {})
                telemetry.update(
                    {
                        "used_count": int(summary.get("used_count", 0) or 0),
                        "correction_count": int(
                            summary.get("correction_count", 0) or 0
                        ),
                        "last_outcome_label": label,
                        "last_outcome_at": timestamp,
                    }
                )
                derivation["retrieval_telemetry"] = telemetry
                changed["derivation"] = derivation
                if timestamp:
                    changed["updated_at"] = timestamp
                changed["version"] = int(changed.get("version", 0) or 0) + 1
                edges[edge_id] = replay_edge_projection(changed)

        for event in self.store.list_events():
            payload = event["payload"]
            event_type = event["event_type"]
            if event_type == "atom_committed":
                atom = payload["atom"]
                atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = replay_edge_projection(edge)
            elif event_type == "atom_updated":
                atom = payload["after"]
                atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = replay_edge_projection(edge)
            elif event_type == "atom_deleted":
                before = payload["before"]
                atom_id = before["id"]
                atoms.pop(atom_id, None)
                tombstone = payload["tombstone"]
                tombstones[tombstone["target_ref"]] = tombstone
                for edge in payload.get("projected_edges", []):
                    edges.pop(edge["edge_id"], None)
            elif event_type == "memories_distilled":
                atom = payload["atom"]
                atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = replay_edge_projection(edge)
            elif event_type == "edge_committed":
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = replay_edge_projection(edge)
            elif event_type in {
                "atom_merged",
                "steward_run",
                "retrieval_outcome_recorded",
                "decay_policy_applied",
                "storage_cleanup_run",
            }:
                for atom in payload.get("projected_atoms", []):
                    if atom.get("deleted"):
                        atoms.pop(atom["id"], None)
                    else:
                        atoms[atom["id"]] = atom
                for edge in payload.get("projected_edges", []):
                    if edge.get("deleted"):
                        edges.pop(edge["edge_id"], None)
                    else:
                        edges[edge["edge_id"]] = replay_edge_projection(edge)
                if event_type == "retrieval_outcome_recorded":
                    replay_legacy_retrieval_edge_feedback(payload)
                for tombstone in payload.get("tombstones", []):
                    tombstones[tombstone["target_ref"]] = tombstone
        return {
            "graph_version": self.store.graph_version(),
            "atoms": atoms,
            "edges": edges,
            "tombstones": tombstones,
        }


    def verify_replay(self) -> dict[str, Any]:
        replayed = self.replay_graph()
        stored_atoms = {
            atom["id"]: atom
            for atom in self.store.list_atoms()
            if not atom.get("deleted")
        }
        replayed_atoms = replayed["atoms"]
        replayed_edges = replayed["edges"]
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
        return {
            "status": "ok"
            if not missing
            and not unexpected
            and not mismatched
            and not missing_edges
            and not unexpected_edges
            and not mismatched_edges
            else "failed",
            "graph_version": self.store.graph_version(),
            "missing_in_replay": missing,
            "unexpected_in_replay": unexpected,
            "mismatched_atoms": mismatched,
            "missing_edges_in_replay": missing_edges,
            "unexpected_edges_in_replay": unexpected_edges,
            "mismatched_edges": mismatched_edges,
            "replayed_atom_count": len(replayed_atoms),
            "stored_atom_count": len(stored_atoms),
            "replayed_edge_count": len(replayed_edges),
            "stored_edge_count": len(stored_edges),
        }
