"""MutationService implementation for the AMOS service facade."""

from ._service_support import (
    Any,
    CASConflict,
    Mapping,
    Sequence,
    ValidationError,
    digest,
    normalize_atom,
    normalize_evidence,
    stable_id,
    utc_now,
)


class MutationService:
    def __init__(self, store: Any, access: Any, indexes: Any, graph: Any):
        self.store = store
        self._mark_foreground_activity = access._mark_foreground_activity
        self._idempotency_hit = access._idempotency_hit
        self._record_idempotency = access._record_idempotency
        self._assert_mutation_allowed = access._assert_mutation_allowed
        self._prepare_committed_atom = indexes._prepare_committed_atom
        self._attach_search_index = indexes._attach_search_index
        self._intrinsic_edges_for_atom = graph._intrinsic_edges_for_atom
        self._memory_identity_digest = graph._memory_identity_digest
        self._atom_projection = graph._atom_projection
        self._edge = graph._edge

    def capture_event(
        self,
        *,
        source_type: str,
        source_ref: str,
        payload: Any,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
        access_policy: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        request_payload = {
            "operation": "capture_event",
            "source_type": source_type,
            "source_ref": source_ref,
            "payload": payload,
            "scope": dict(scope or {}),
            "access_policy": dict(access_policy or {}),
        }
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, request_payload)
            if prior is not None:
                return prior
            evidence = normalize_evidence(
                {
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "payload": payload,
                    "scope": scope or {},
                    "access_policy": access_policy,
                }
            )
            op_payload = {"operation": "capture_event", "evidence": evidence}
            event = self.store.append_event(
                conn,
                event_type="evidence_captured",
                actor=actor,
                payload=op_payload,
                evidence_refs=[evidence["evidence_id"]],
                idempotency_key=idempotency_key,
            )
            self.store.insert_evidence(conn, evidence, event["event_id"])
            response = {"status": "captured", "evidence": evidence, "event": event}
            self._record_idempotency(
                conn, actor, idempotency_key, request_payload, event, response
            )
            return response


    def commit_atom(
        self,
        atom: Mapping[str, Any],
        *,
        actor: str = "system",
        idempotency_key: str | None = None,
        authorization_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        request_payload = {"operation": "commit_atom", "atom": dict(atom)}
        normalized = self._prepare_committed_atom(atom)
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, request_payload)
            if prior is not None:
                return prior
            projected_edges = self._intrinsic_edges_for_atom(normalized)
            op_payload = {
                "operation": "commit_atom",
                "atom": normalized,
                "projected_edges": projected_edges,
            }
            content_digest = self._memory_identity_digest(normalized)
            tombstone = self.store.get_tombstone(
                normalized["id"], content_digest=content_digest
            )
            if tombstone and tombstone["recreation_policy"] != "allow_recreate":
                raise ValidationError(
                    f"memory is tombstoned: {normalized['id']} / {content_digest}"
                )
            if self.store.get_atom(normalized["id"]) is not None:
                raise ValidationError(f"atom already exists: {normalized['id']}")
            event = self.store.append_event(
                conn,
                event_type="atom_committed",
                actor=actor,
                payload=op_payload,
                target_refs=[normalized["id"]],
                evidence_refs=normalized["evidence_refs"],
                idempotency_key=idempotency_key,
                authorization_context=authorization_context,
            )
            self.store.insert_atom(conn, normalized)
            for edge in projected_edges:
                self.store.insert_edge(conn, edge)
            self.store.clear_packet_cache(conn)
            response = {
                "status": "committed",
                "atom": normalized,
                "edges": projected_edges,
                "event": event,
            }
            self._record_idempotency(
                conn, actor, idempotency_key, request_payload, event, response
            )
            return response


    def propose_memory_atoms(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        actor: str = "system",
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        proposals = []
        for candidate in candidates:
            atom = dict(candidate)
            if scope is not None:
                atom.setdefault("scope", dict(scope))
            atom["lifecycle_state"] = "proposed"
            atom.setdefault("confidence", {"level": "low-medium", "score": 0.35})
            proposals.append(self.commit_atom(atom, actor=actor))
        return {
            "status": "proposed",
            "proposals": proposals,
            "graph_version": self.store.graph_version(),
        }


    def commit_memory_atoms(
        self,
        atoms: Sequence[Mapping[str, Any]],
        *,
        actor: str = "system",
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        prepared = [self._prepare_committed_atom(atom) for atom in atoms]
        seen_ids: set[str] = set()
        for atom in prepared:
            if atom["id"] in seen_ids:
                raise ValidationError(f"duplicate atom in batch: {atom['id']}")
            seen_ids.add(atom["id"])
        committed = []
        with self.store.transaction() as conn:
            for normalized in prepared:
                projected_edges = self._intrinsic_edges_for_atom(normalized)
                op_payload = {
                    "operation": "commit_atom",
                    "atom": normalized,
                    "projected_edges": projected_edges,
                }
                content_digest = self._memory_identity_digest(normalized)
                tombstone = self.store.get_tombstone(
                    normalized["id"], content_digest=content_digest
                )
                if tombstone and tombstone["recreation_policy"] != "allow_recreate":
                    raise ValidationError(
                        f"memory is tombstoned: {normalized['id']} / {content_digest}"
                    )
                if self.store.get_atom(normalized["id"]) is not None:
                    raise ValidationError(f"atom already exists: {normalized['id']}")
                event = self.store.append_event(
                    conn,
                    event_type="atom_committed",
                    actor=actor,
                    payload=op_payload,
                    target_refs=[normalized["id"]],
                    evidence_refs=normalized["evidence_refs"],
                )
                self.store.insert_atom(conn, normalized)
                for edge in projected_edges:
                    self.store.insert_edge(conn, edge)
                committed.append(
                    {
                        "status": "committed",
                        "atom": normalized,
                        "edges": projected_edges,
                        "event": event,
                    }
                )
            if committed:
                self.store.clear_packet_cache(conn)
        return {
            "status": "committed",
            "committed": committed,
            "graph_version": self.store.graph_version(),
        }


    def update_atom(
        self,
        atom_id: str,
        *,
        payload_patch: Mapping[str, Any] | None = None,
        set_fields: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        op_payload = {
            "operation": "update_atom",
            "atom_id": atom_id,
            "payload_patch": dict(payload_patch or {}),
            "set_fields": dict(set_fields or {}),
            "expected_version": expected_version,
        }
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, op_payload)
            if prior is not None:
                return prior
            current = self.store.get_atom(atom_id)
            if current is None or current.get("deleted"):
                raise ValidationError(f"unknown atom: {atom_id}")
            self._assert_mutation_allowed(
                current, actor=actor, authorization_context=authorization_context
            )
            if expected_version is not None and current["version"] != expected_version:
                raise CASConflict(
                    f"expected {atom_id} version {expected_version}, "
                    f"found {current['version']}"
                )
            updated = dict(current)
            if payload_patch:
                updated_payload = dict(updated["payload"])
                updated_payload.update(dict(payload_patch))
                updated["payload"] = updated_payload
            for key, value in dict(set_fields or {}).items():
                if key in {"id", "type", "schema_version", "created_at", "version"}:
                    raise ValidationError(f"cannot update immutable atom field: {key}")
                if key == "payload":
                    updated["payload"] = dict(value)
                else:
                    updated[key] = value
            updated["revision_history"] = list(updated["revision_history"])
            updated["revision_history"].append(
                {
                    "version": current["version"],
                    "digest": digest(self._atom_projection(current)),
                    "changed_at": utc_now(),
                    "actor": actor,
                }
            )
            updated["version"] = int(current["version"]) + 1
            updated["updated_at"] = utc_now()
            updated = normalize_atom(
                self._attach_search_index(updated), require_id=True
            )
            projected_edges = []
            if (
                current.get("lifecycle_state") == "active"
                and updated.get("lifecycle_state") != "active"
            ):
                projected_edges = self.store.mark_edges_deleted_for_ref(conn, atom_id)
            elif (
                current.get("lifecycle_state") != "active"
                and updated.get("lifecycle_state") == "active"
            ):
                for edge in self._intrinsic_edges_for_atom(updated):
                    existing_edge = self.store.get_edge(str(edge["edge_id"]))
                    if existing_edge:
                        edge = {
                            **edge,
                            "created_at": existing_edge["created_at"],
                            "updated_at": utc_now(),
                            "version": int(existing_edge["version"]) + 1,
                        }
                    self.store.upsert_edge(conn, edge)
                    projected_edges.append(edge)
            event = self.store.append_event(
                conn,
                event_type="atom_updated",
                actor=actor,
                payload={
                    "operation": "update_atom",
                    "before": current,
                    "after": updated,
                    "projected_edges": projected_edges,
                },
                target_refs=[atom_id],
                evidence_refs=updated["evidence_refs"],
                idempotency_key=idempotency_key,
                expected_versions={atom_id: expected_version}
                if expected_version is not None
                else {},
                authorization_context=authorization_context,
            )
            self.store.replace_atom(conn, updated)
            self.store.clear_packet_cache(conn)
            response = {
                "status": "updated",
                "atom": updated,
                "event": event,
                "projected_edges": projected_edges,
            }
            self._record_idempotency(conn, actor, idempotency_key, op_payload, event, response)
            return response


    def archive_atom(
        self,
        atom_id: str,
        *,
        reason: str = "archived",
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.update_atom(
            atom_id,
            set_fields={
                "lifecycle_state": "archived",
                "health_status": "stale",
                "decay_policy": {"archive_reason": reason},
            },
            expected_version=expected_version,
            actor=actor,
            authorization_context=authorization_context,
        )


    def delete_atom(
        self,
        atom_id: str,
        *,
        reason: str,
        expected_version: int | None = None,
        actor: str = "system",
        authorization_context: Mapping[str, Any] | None = None,
        recreation_policy: str = "block_recreate",
    ) -> dict[str, Any]:
        self._mark_foreground_activity(actor)
        op_payload = {
            "operation": "delete_atom",
            "atom_id": atom_id,
            "reason": reason,
            "expected_version": expected_version,
            "recreation_policy": recreation_policy,
        }
        with self.store.transaction() as conn:
            current = self.store.get_atom(atom_id)
            if current is None:
                raise ValidationError(f"unknown atom: {atom_id}")
            self._assert_mutation_allowed(
                current, actor=actor, authorization_context=authorization_context
            )
            if expected_version is not None and current["version"] != expected_version:
                raise CASConflict(
                    f"expected {atom_id} version {expected_version}, "
                    f"found {current['version']}"
                )
            updated = dict(current)
            updated["lifecycle_state"] = "deleted"
            updated["health_status"] = "deleted"
            updated["deleted"] = 1
            updated["version"] = int(current["version"]) + 1
            updated["updated_at"] = utc_now()
            updated["revision_history"] = list(updated["revision_history"])
            updated["revision_history"].append(
                {
                    "version": current["version"],
                    "digest": digest(self._atom_projection(current)),
                    "changed_at": utc_now(),
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
                target_ref=atom_id,
                content_digest=self._memory_identity_digest(current),
                recreation_policy=recreation_policy,
                reason=reason,
            )
            deleted_edges = self.store.mark_edges_deleted_for_ref(conn, atom_id)
            event = self.store.append_event(
                conn,
                event_type="atom_deleted",
                actor=actor,
                payload={
                    **op_payload,
                    "before": current,
                    "tombstone": tombstone,
                    "projected_edges": deleted_edges,
                },
                target_refs=[atom_id],
                evidence_refs=current["evidence_refs"],
                expected_versions={atom_id: expected_version}
                if expected_version is not None
                else {},
                authorization_context=authorization_context,
            )
            self.store.replace_atom(conn, updated)
            self.store.clear_packet_cache(conn)
            return {
                "status": "deleted",
                "atom": updated,
                "tombstone": tombstone,
                "event": event,
            }


    def request_deletion(
        self,
        *,
        target_ref: str,
        reason: str,
        requested_by: str = "system",
        expected_version: int | None = None,
        authorization_context: Mapping[str, Any] | None = None,
        recreation_policy: str = "block_recreate",
    ) -> dict[str, Any]:
        self._mark_foreground_activity(requested_by)
        result = self.delete_atom(
            target_ref,
            reason=reason,
            expected_version=expected_version,
            actor=requested_by,
            authorization_context=authorization_context,
            recreation_policy=recreation_policy,
        )
        result["residual_retention"] = {
            "hot_database_payload": "suppressed",
            "packet_cache": "purged",
            "offline_backup_residual_window_days": 30,
            "evidence_archive": "retained_or_suppressed_by_policy",
        }
        return result


    def merge_atoms(
        self,
        *,
        source_refs: Sequence[str],
        merged_payload: Mapping[str, Any],
        merged_type: str = "semantic",
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        if not approved_by:
            return {
                "status": "review_required",
                "action": "merge_atoms",
                "source_refs": list(source_refs),
                "risk": "high",
                "reason": "active atom merge requires explicit review",
                "mutated": False,
            }
        with self.store.transaction() as conn:
            sources = []
            for ref in source_refs:
                atom = self.store.get_atom(ref)
                if atom is None or atom.get("deleted"):
                    raise ValidationError(f"unknown source atom: {ref}")
                self._assert_mutation_allowed(
                    atom,
                    actor=actor,
                    authorization_context={
                        "roles": ["owner"],
                        "trust_level": 10,
                        "capabilities": ["memory.write"],
                        "approved_by": approved_by,
                    },
                )
                sources.append(atom)
            now = utc_now()
            merged = normalize_atom(
                {
                    "type": merged_type,
                    "payload": dict(merged_payload),
                    "scope": dict(scope or {}),
                    "supersedes": list(source_refs),
                    "salience": max([float(atom["salience"]) for atom in sources] + [0.5]),
                    "utility": max([float(atom["utility"]) for atom in sources] + [0.5]),
                    "confidence": {"level": "medium-high", "score": 0.75},
                }
            )
            merged["id"] = stable_id(
                "atom",
                {
                    "operation": "merge_atoms",
                    "source_refs": list(source_refs),
                    "payload": merged_payload,
                    "scope": dict(scope or {}),
                },
            )
            merged["created_at"] = now
            merged["observed_at"] = now
            merged["updated_at"] = now
            merged = normalize_atom(self._attach_search_index(merged), require_id=True)
            self.store.insert_atom(conn, merged)
            projected_atoms = [merged]
            projected_edges = []
            for source in sources:
                edge = self._edge(
                    merged["id"], source["id"], "rel:derived_from", dict(scope or {})
                )
                self.store.insert_edge(conn, edge)
                projected_edges.append(edge)
                archived = dict(source)
                archived["lifecycle_state"] = "archived"
                archived["health_status"] = "merged"
                archived["version"] = int(source["version"]) + 1
                archived["updated_at"] = utc_now()
                archived["decay_policy"] = {
                    **dict(archived.get("decay_policy") or {}),
                    "merged_into": merged["id"],
                }
                archived = normalize_atom(
                    self._attach_search_index(archived), require_id=True
                )
                self.store.replace_atom(conn, archived)
                projected_atoms.append(archived)
            event = self.store.append_event(
                conn,
                event_type="atom_merged",
                actor=actor,
                payload={
                    "operation": "merge_atoms",
                    "merged_atom": merged,
                    "source_refs": list(source_refs),
                    "projected_atoms": projected_atoms,
                    "projected_edges": projected_edges,
                },
                target_refs=[merged["id"], *source_refs],
                authorization_context={"approved_by": approved_by},
            )
            self.store.clear_packet_cache(conn)
            return {
                "status": "merged",
                "atom": merged,
                "source_refs": list(source_refs),
                "edges": projected_edges,
                "event": event,
            }


    def distill_memories(
        self,
        *,
        target_refs: Sequence[str],
        summary: str | Mapping[str, Any],
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        idempotency_key: str | None = None,
        distillation_type: str = "summary",
        archive_sources: bool = False,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        if archive_sources and not approved_by:
            return {
                "status": "review_required",
                "action": "distill_memories",
                "target_refs": list(target_refs),
                "risk": "high",
                "reason": "archiving source memories requires explicit approval",
                "mutated": False,
            }
        request_payload = {
            "operation": "distill_memories",
            "target_refs": list(target_refs),
            "summary": summary,
            "scope": dict(scope or {}),
            "distillation_type": distillation_type,
            "archive_sources": archive_sources,
            "approved_by": approved_by,
        }
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(conn, actor, idempotency_key, request_payload)
            if prior is not None:
                return prior
            source_atoms = []
            for ref in target_refs:
                atom = self.store.get_atom(ref)
                if atom is None or atom.get("deleted"):
                    raise ValidationError(f"unknown source atom: {ref}")
                source_atoms.append(atom)
            now = utc_now()
            source_digests = [digest(self._atom_projection(atom)) for atom in source_atoms]
            distilled = normalize_atom(
                {
                    "type": "semantic",
                    "payload": {
                        "distillation_type": distillation_type,
                        "summary": summary,
                        "source_refs": list(target_refs),
                        "source_digests": source_digests,
                        "created_by": actor,
                    },
                    "scope": dict(scope or {}),
                    "layer": "consolidated_long_term",
                    "retention_class": "distilled",
                    "supersedes": list(target_refs) if archive_sources else [],
                    "salience": 0.8,
                    "utility": 0.85,
                    "confidence": {"level": "medium-high", "score": 0.75},
                }
            )
            distilled["id"] = stable_id(
                "atom",
                {
                    "type": "semantic",
                    "summary": summary,
                    "target_refs": list(target_refs),
                    "scope": dict(scope or {}),
                    "distillation_type": distillation_type,
                },
            )
            if self.store.get_atom(distilled["id"]) is not None:
                raise ValidationError(f"distilled atom already exists: {distilled['id']}")
            distilled["created_at"] = now
            distilled["observed_at"] = now
            distilled["updated_at"] = now
            distilled = normalize_atom(
                self._attach_search_index(distilled), require_id=True
            )
            edges = [
                self._edge(
                    distilled["id"],
                    source["id"],
                    "rel:derived_from",
                    dict(scope or {}),
                )
                for source in source_atoms
            ]
            event = self.store.append_event(
                conn,
                event_type="memories_distilled",
                actor=actor,
                payload={
                    "operation": "distill_memories",
                    "atom": distilled,
                    "projected_edges": edges,
                },
                target_refs=[distilled["id"], *target_refs],
                idempotency_key=idempotency_key,
                authorization_context={"approved_by": approved_by}
                if approved_by
                else {},
            )
            self.store.insert_atom(conn, distilled)
            for edge, source in zip(edges, source_atoms):
                self.store.insert_edge(conn, edge)
                if archive_sources:
                    changed = dict(source)
                    changed["lifecycle_state"] = "archived"
                    changed["health_status"] = "stale"
                    changed["version"] = int(source["version"]) + 1
                    changed["updated_at"] = utc_now()
                    changed["decay_policy"] = {
                        **dict(changed.get("decay_policy") or {}),
                        "archived_by_distillation": distilled["id"],
                    }
                    changed = normalize_atom(
                        self._attach_search_index(changed), require_id=True
                    )
                    self.store.replace_atom(conn, changed)
            self.store.clear_packet_cache(conn)
            response = {
                "status": "distilled",
                "atom": distilled,
                "source_refs": list(target_refs),
                "edges": edges,
                "archived_sources": archive_sources,
                "event": event,
            }
            self._record_idempotency(
                conn, actor, idempotency_key, request_payload, event, response
            )
            return response
