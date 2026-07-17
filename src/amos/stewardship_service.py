"""StewardshipService implementation for the AMOS service facade."""

from ._service_support import (
    Any,
    EvidenceWindow,
    GENERIC_GRAPH_PROCESSOR_ID,
    GENERIC_GRAPH_PROCESSOR_VERSION,
    HIGH_RISK_MAINTENANCE,
    MaintenanceProcessor,
    Mapping,
    SEMANTIC_RELATION_PROCESSOR_ID,
    SEMANTIC_RELATION_PROCESSOR_VERSION,
    Sequence,
    digest,
    load_maintenance_processor,
    maintenance_scope_visible,
    normalize_atom,
    normalize_relation,
    proposal_is_auto_committable,
    scope_visible,
    semantic_relation_proposals_from_facets,
    stable_id,
    utc_now,
)


class StewardshipService:
    def __init__(
        self,
        store: Any,
        smp: Any,
        maintenance_processors: Any,
        mutations: Any,
        indexes: Any,
        graph: Any,
    ):
        self.store = store
        self.smp = smp
        self.maintenance_processors = maintenance_processors
        self.commit_atom = mutations.commit_atom
        self.delete_atom = mutations.delete_atom
        self._attach_search_index = indexes._attach_search_index
        self._atom_search_index = indexes._atom_search_index
        self._sync_smp_vector_model = indexes._sync_smp_vector_model
        self._archive_atom_projection = graph._archive_atom_projection
        self._structured_duplicate_key = graph._structured_duplicate_key
        self._structured_duplicate_quality = graph._structured_duplicate_quality
        self._intrinsic_edges_for_atom = graph._intrinsic_edges_for_atom
        self._contradiction_signature = graph._contradiction_signature
        self._atom_projection = graph._atom_projection
        self._edge = graph._edge

    def register_maintenance_processor(
        self, processor: MaintenanceProcessor
    ) -> dict[str, Any]:
        self.maintenance_processors.register(processor)
        return {
            "status": "registered",
            "processor": {
                "processor_id": processor.processor_id,
                "processor_version": processor.processor_version,
            },
            "processors": self.list_maintenance_processors()["processors"],
        }


    def load_maintenance_processor(self, import_path: str) -> dict[str, Any]:
        processor = load_maintenance_processor(import_path)
        return self.register_maintenance_processor(processor)


    def list_maintenance_processors(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "processors": self.maintenance_processors.list(),
        }


    def llm_reviewer_policy(self) -> dict[str, Any]:
        return {
            "enabled_by_default": False,
            "allowed_when_enabled": [
                "ambiguous_atomization",
                "scope_refinement_suggestions",
                "contradiction_analysis_suggestions",
                "natural_language_explanation_drafting",
            ],
            "forbidden": [
                "direct_canonical_mutation",
                "deletion_approval",
                "access_policy_change",
                "autonomous_preference_alteration",
            ],
            "output_envelope": [
                "processor_id",
                "processor_version",
                "input_refs",
                "output_type",
                "confidence",
                "reason_code",
                "evidence_refs",
                "recommended_action",
                "risk_level",
            ],
        }


    def request_maintenance(
        self,
        *,
        action: str,
        target_refs: Sequence[str] | None = None,
        risk: str = "low",
        approved_by: str | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        if risk == "high" or action in HIGH_RISK_MAINTENANCE:
            if not approved_by:
                return {
                    "status": "review_required",
                    "action": action,
                    "target_refs": list(target_refs or []),
                    "risk": "high",
                    "reason": "high-risk maintenance requires explicit approval",
                    "mutated": False,
                }
        if action in {"cleanup", "steward", "deduplicate", "detect_contradictions"}:
            return self.run_steward(
                scope=scope or {}, actor=actor, approved_by=approved_by
            )
        if action == "delete":
            results = [
                self.delete_atom(ref, reason="approved maintenance", actor=actor)
                for ref in target_refs or []
            ]
            return {
                "status": "completed",
                "action": action,
                "approved_by": approved_by,
                "results": results,
            }
        return {
            "status": "preview",
            "action": action,
            "target_refs": list(target_refs or []),
            "risk": risk,
            "mutated": False,
        }


    def run_maintenance_distiller(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        actor: str = "svc:memory_policy",
        domain: str = "generic",
        processor_ids: Sequence[str] | None = None,
        max_atoms: int = 128,
        max_events: int = 64,
        max_retrieval_outcomes: int = 64,
        auto_commit_low_risk: bool = True,
        reviewer: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        processor_ids = list(processor_ids or [])
        window = self._maintenance_evidence_window(
            scope=scope,
            domain=domain,
            max_atoms=max_atoms,
            max_events=max_events,
            max_retrieval_outcomes=max_retrieval_outcomes,
        )
        processors = self.maintenance_processors.select(
            window, processor_ids=processor_ids
        )
        known_processor_ids = {
            item["processor_id"] for item in self.maintenance_processors.list()
        }
        missing_processors = sorted(set(processor_ids) - known_processor_ids)
        proposals = []
        semantic_facets = []
        for processor in processors:
            proposals.extend(proposal.to_dict() for proposal in processor.propose(window))
            extract_facets = getattr(processor, "extract_facets", None)
            if callable(extract_facets):
                semantic_facets.extend(extract_facets(window))
        relation_proposals = []
        if semantic_facets:
            relation_proposals = [
                proposal.to_dict()
                for proposal in semantic_relation_proposals_from_facets(
                    semantic_facets,
                    existing_edges=window.edges,
                )
            ]
            proposals.extend(relation_proposals)
        proposals.sort(
            key=lambda proposal: (
                proposal["risk_level"] != "low",
                proposal["processor_id"],
                proposal["proposal_id"],
            )
        )
        processor_records = [
            {
                "processor_id": processor.processor_id,
                "processor_version": processor.processor_version,
            }
            for processor in processors
        ]
        if relation_proposals:
            processor_records.append(
                {
                    "processor_id": SEMANTIC_RELATION_PROCESSOR_ID,
                    "processor_version": SEMANTIC_RELATION_PROCESSOR_VERSION,
                }
            )
        if any(
            proposal.get("processor_id") == GENERIC_GRAPH_PROCESSOR_ID
            for proposal in proposals
        ):
            processor_records.append(
                {
                    "processor_id": GENERIC_GRAPH_PROCESSOR_ID,
                    "processor_version": GENERIC_GRAPH_PROCESSOR_VERSION,
                }
            )
        reviewer_status = self._maintenance_reviewer_status(reviewer)
        if not proposals and not missing_processors:
            return {
                "status": "skipped",
                "reason": "no_proposals",
                "scope": scope,
                "domain": domain,
                "window": window.to_dict(),
                "processors": processor_records,
                "missing_processors": missing_processors,
                "proposals": [],
                "committed": [],
                "deferred": [],
                "reviewer": reviewer_status,
                "event": None,
                "graph_version": self.store.graph_version(),
            }

        committed: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        for proposal in proposals:
            if auto_commit_low_risk and proposal_is_auto_committable(proposal):
                committed.append(
                    self._commit_maintenance_proposal(proposal, actor=actor)
                )
            else:
                deferred.append(
                    {
                        "proposal_id": proposal["proposal_id"],
                        "action": proposal["action"],
                        "risk_level": proposal["risk_level"],
                        "reason": "auto_commit_disabled"
                        if not auto_commit_low_risk
                        else "requires_review_or_unsupported_action",
                        "source_refs": proposal["source_refs"],
                        "proposal_digest": self._maintenance_proposal_fingerprint(
                            proposal
                        ),
                    }
                )

        committed_count = len(
            [item for item in committed if item.get("status") == "committed"]
        )
        already_committed_count = len(
            [item for item in committed if item.get("status") == "already_committed"]
        )
        blocked_fingerprint = self._maintenance_distiller_blocked_fingerprint(
            scope=scope,
            domain=domain,
            processor_ids=processor_ids,
            missing_processors=missing_processors,
            committed=committed,
            deferred=deferred,
            reviewer_status=reviewer_status,
            auto_commit_low_risk=auto_commit_low_risk,
        )
        if proposals and committed_count == 0 and not deferred:
            return {
                "status": "skipped",
                "reason": "all_proposals_already_committed",
                "scope": scope,
                "domain": domain,
                "window": window.to_dict(),
                "processors": processor_records,
                "missing_processors": missing_processors,
                "proposals": proposals,
                "committed": committed,
                "deferred": deferred,
                "reviewer": reviewer_status,
                "event": None,
                "graph_version": self.store.graph_version(),
            }
        if (
            proposals
            and committed_count == 0
            and deferred
            and self.store.get_meta(
                self._maintenance_distiller_blocked_state_key(
                    scope=scope, domain=domain, processor_ids=processor_ids
                )
            )
            == blocked_fingerprint
        ):
            return {
                "status": "skipped",
                "reason": "deferred_proposals_unchanged",
                "scope": scope,
                "domain": domain,
                "window": window.to_dict(),
                "processors": processor_records,
                "missing_processors": missing_processors,
                "proposals": proposals,
                "committed": committed,
                "deferred": deferred,
                "reviewer": reviewer_status,
                "deferred_fingerprint": blocked_fingerprint,
                "event": None,
                "graph_version": self.store.graph_version(),
            }

        target_refs = [
            ref
            for proposal in proposals
            for ref in proposal.get("source_refs", []) + proposal.get("target_refs", [])
        ]
        target_refs.extend(
            committed_item["atom"]["id"]
            for committed_item in committed
            if committed_item.get("atom")
        )
        target_refs.extend(
            ref
            for committed_item in committed
            if committed_item.get("edge")
            for ref in (
                committed_item["edge"].get("source_ref"),
                committed_item["edge"].get("target_ref"),
            )
            if ref
        )
        event_payload = {
            "operation": "run_maintenance_distiller",
            "scope": scope,
            "domain": domain,
            "window": window.to_dict(),
            "processors": processor_records,
            "missing_processors": missing_processors,
            "proposal_count": len(proposals),
            "committed_count": committed_count,
            "already_committed_count": already_committed_count,
            "deferred_count": len(deferred),
            "auto_commit_low_risk": auto_commit_low_risk,
            "reviewer": reviewer_status,
            "deferred_fingerprint": blocked_fingerprint if deferred else None,
            "deferred_proposal_ids": [
                item["proposal_id"] for item in deferred if item.get("proposal_id")
            ],
        }
        with self.store.transaction() as conn:
            event = self.store.append_event(
                conn,
                event_type="maintenance_distillation_run",
                actor=actor,
                payload=event_payload,
                target_refs=sorted(set(target_refs)),
                authorization_context={
                    "auto_commit_low_risk": auto_commit_low_risk,
                    "reviewer_authority": event_payload["reviewer"]["authority"],
                },
            )
            if deferred:
                self.store._set_meta(
                    conn,
                    self._maintenance_distiller_blocked_state_key(
                        scope=scope, domain=domain, processor_ids=processor_ids
                    ),
                    blocked_fingerprint,
                )
        return {
            "status": "completed",
            "scope": scope,
            "domain": domain,
            "window": window.to_dict(),
            "processors": event_payload["processors"],
            "missing_processors": missing_processors,
            "proposals": proposals,
            "committed": committed,
            "deferred": deferred,
            "reviewer": event_payload["reviewer"],
            "deferred_fingerprint": event_payload["deferred_fingerprint"],
            "event": event,
            "graph_version": self.store.graph_version(),
        }


    def run_smp_analysis(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        target_refs: Sequence[str] | None = None,
        max_atoms: int | None = None,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        self._sync_smp_vector_model()
        atoms = [
            atom
            for atom in self.store.list_atoms_filtered()
            if not atom.get("deleted") and scope_visible(atom["scope"], scope)
        ]
        if target_refs:
            allowed = set(target_refs)
            atoms = [atom for atom in atoms if atom["id"] in allowed]
        total_atom_count = len(atoms)
        if max_atoms is not None:
            atoms.sort(
                key=lambda atom: (
                    str(
                        atom.get("observed_at")
                        or atom.get("updated_at")
                        or atom.get("created_at")
                        or ""
                    ),
                    str(atom.get("id") or ""),
                ),
                reverse=True,
            )
            atoms = atoms[: max(1, int(max_atoms or 1))]
        shape_reports = [self.smp.validate_shape(atom) for atom in atoms]
        clusters = self.smp.cluster(atoms)
        conflicts = self.smp.detect_conflicts(atoms)
        health = [self.smp.propose_health(atom) for atom in atoms]
        links = []
        for index, atom in enumerate(atoms):
            links.extend(
                self.smp.propose_links(
                    atom,
                    self._smp_link_candidates(atom, atoms[index + 1 :]),
                )
            )
        outputs = shape_reports + clusters + conflicts + health + links
        return {
            "status": "completed",
            "processor_id": self.smp.processor_id,
            "processor_version": self.smp.processor_version,
            "graph_version": self.store.graph_version(),
            "scope": scope,
            "atom_count": total_atom_count,
            "analyzed_atom_count": len(atoms),
            "omitted_atom_count": max(0, total_atom_count - len(atoms)),
            "outputs": outputs,
            "review_required": [
                output
                for output in outputs
                if output["risk_level"] == "high"
                or output["recommended_action"].get("type") in HIGH_RISK_MAINTENANCE
            ],
        }


    def _smp_link_candidates(
        self,
        atom: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        *,
        limit: int = 24,
    ) -> list[Mapping[str, Any]]:
        if len(candidates) <= limit:
            return list(candidates)
        atom_index = self._atom_search_index(atom)
        atom_tokens = set(atom_index["tokens"])
        scored = []
        for candidate in candidates:
            candidate_tokens = set(self._atom_search_index(candidate)["tokens"])
            overlap = len(atom_tokens.intersection(candidate_tokens))
            same_type = 1 if candidate.get("type") == atom.get("type") else 0
            if overlap <= 0 and not same_type:
                continue
            scored.append((same_type, overlap, str(candidate.get("updated_at") or ""), candidate))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored[:limit]]


    def run_steward(
        self,
        *,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        actions: list[dict[str, Any]] = []
        projected_atoms: list[dict[str, Any]] = []
        projected_edges: list[dict[str, Any]] = []
        with self.store.transaction() as conn:
            atoms = [
                atom
                for atom in self.store.list_atoms_filtered(
                    lifecycle_states=["active"],
                )
                if not atom.get("deleted")
                and atom.get("lifecycle_state") == "active"
                and scope_visible(atom["scope"], scope)
            ]
            live_attachment_relations = {
                "rel:attributed_to",
                "rel:has_capability",
                "rel:has_limitation",
                "rel:made_commitment",
            }
            invalid_attachment_edge_ids: list[str] = []
            for edge in self.store.list_edges():
                if edge.get("relation") not in live_attachment_relations:
                    continue
                endpoints = (
                    self.store.get_atom(str(edge.get("source_ref") or "")),
                    self.store.get_atom(str(edge.get("target_ref") or "")),
                )
                if any(
                    atom is None
                    or atom.get("deleted")
                    or atom.get("lifecycle_state") != "active"
                    for atom in endpoints
                ):
                    invalid_attachment_edge_ids.append(str(edge.get("edge_id") or ""))
            invalid_attachment_edges = self.store.mark_edges_deleted(
                conn, invalid_attachment_edge_ids
            )
            if invalid_attachment_edges:
                projected_edges.extend(invalid_attachment_edges)
                actions.append(
                    {
                        "action": "prune_inactive_attachment_edges",
                        "edge_count": len(invalid_attachment_edges),
                        "relations": sorted(
                            {
                                str(edge.get("relation") or "")
                                for edge in invalid_attachment_edges
                            }
                        ),
                    }
                )
            proposed_endpoint_edge_ids: list[str] = []
            for edge in self.store.list_edges():
                if edge.get("lifecycle_state") != "active":
                    continue
                endpoints = (
                    self.store.get_atom(str(edge.get("source_ref") or "")),
                    self.store.get_atom(str(edge.get("target_ref") or "")),
                )
                if any(
                    atom is not None
                    and not atom.get("deleted")
                    and atom.get("lifecycle_state") == "proposed"
                    for atom in endpoints
                ):
                    proposed_endpoint_edge_ids.append(str(edge.get("edge_id") or ""))
            proposed_endpoint_edges = self.store.mark_edges_deleted(
                conn, proposed_endpoint_edge_ids
            )
            if proposed_endpoint_edges:
                projected_edges.extend(proposed_endpoint_edges)
                actions.append(
                    {
                        "action": "isolate_proposed_endpoint_edges",
                        "edge_count": len(proposed_endpoint_edges),
                        "policy": "proposed_atoms_do_not_participate_in_active_graph",
                    }
                )
            smp_outputs = self.smp.cluster(atoms) + self.smp.detect_conflicts(atoms)
            existing_edges = {
                edge["edge_id"]: edge for edge in self.store.list_edges()
            }
            intrinsic_edge_count = 0
            refreshed_intrinsic_edge_count = 0
            for atom in atoms:
                for edge in self._intrinsic_edges_for_atom(atom):
                    existing_edge = existing_edges.get(edge["edge_id"])
                    if existing_edge is not None:
                        evidence_refs = sorted(
                            {
                                *(
                                    str(ref)
                                    for ref in existing_edge.get("evidence_refs", [])
                                    if str(ref)
                                ),
                                *(
                                    str(ref)
                                    for ref in edge.get("evidence_refs", [])
                                    if str(ref)
                                ),
                            }
                        )
                        existing_score = float(
                            (existing_edge.get("confidence") or {}).get(
                                "score", 0.0
                            )
                        )
                        projected_score = float(
                            (edge.get("confidence") or {}).get("score", 0.0)
                        )
                        score = max(existing_score, projected_score)
                        confidence = {
                            "level": (
                                "high"
                                if score >= 0.85
                                else "medium-high"
                                if score >= 0.65
                                else "medium"
                            ),
                            "score": score,
                        }
                        if (
                            not existing_edge.get("deleted")
                            and existing_edge.get("lifecycle_state") == "active"
                            and existing_edge.get("health_status") == "healthy"
                            and existing_edge.get("evidence_refs") == evidence_refs
                            and existing_edge.get("confidence") == confidence
                        ):
                            continue
                        edge = {
                            **edge,
                            "evidence_refs": evidence_refs,
                            "confidence": confidence,
                            "created_at": existing_edge["created_at"],
                            "updated_at": utc_now(),
                            "version": int(existing_edge.get("version", 1)) + 1,
                        }
                        self.store.upsert_edge(conn, edge)
                        existing_edges[edge["edge_id"]] = edge
                        projected_edges.append(edge)
                        refreshed_intrinsic_edge_count += 1
                        continue
                    self.store.insert_edge(conn, edge)
                    existing_edges[edge["edge_id"]] = edge
                    projected_edges.append(edge)
                    intrinsic_edge_count += 1
            if intrinsic_edge_count:
                actions.append(
                    {
                        "action": "project_intrinsic_edges",
                        "edge_count": intrinsic_edge_count,
                        "policy": "deterministic_structured_atom_refs",
                    }
                )
            if refreshed_intrinsic_edge_count:
                actions.append(
                    {
                        "action": "refresh_intrinsic_edges",
                        "edge_count": refreshed_intrinsic_edge_count,
                        "policy": "merge_structured_edge_provenance",
                    }
                )
            seen: dict[str, dict[str, Any]] = {}
            for atom in sorted(atoms, key=lambda row: row["created_at"]):
                key = digest(
                    {
                        "type": atom["type"],
                        "payload": atom["payload"],
                        "scope": atom["scope"],
                    }
                )
                existing = seen.get(key)
                if existing is None:
                    seen[key] = atom
                    continue
                duplicate, deleted_edges = self._archive_atom_projection(
                    conn,
                    atom,
                    reason="exact_duplicate",
                    superseded_by=existing["id"],
                    actor=actor,
                )
                projected_atoms.append(duplicate)
                projected_edges.extend(deleted_edges)
                actions.append(
                    {
                        "action": "deduplicate",
                        "kept": existing["id"],
                        "archived": duplicate["id"],
                        "smp_outputs": [
                            output
                            for output in smp_outputs
                            if output["reason_code"] == "near_duplicate"
                            and atom["id"] in output["input_refs"]
                        ],
                    }
                )
            structured_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            archived_ids = {atom["id"] for atom in projected_atoms}
            for atom in atoms:
                if atom["id"] in archived_ids:
                    continue
                key = self._structured_duplicate_key(atom)
                if key is None:
                    continue
                structured_groups.setdefault(key, []).append(atom)
            for key, group in structured_groups.items():
                active_group = [
                    atom
                    for atom in group
                    if atom["id"] not in archived_ids
                    and atom.get("lifecycle_state") == "active"
                    and not atom.get("deleted")
                ]
                if len(active_group) < 2:
                    continue
                kept = max(
                    active_group,
                    key=lambda atom: (
                        self._structured_duplicate_quality(atom),
                        str(atom.get("updated_at") or ""),
                        str(atom.get("id") or ""),
                    ),
                )
                for atom in active_group:
                    if atom["id"] == kept["id"]:
                        continue
                    duplicate, deleted_edges = self._archive_atom_projection(
                        conn,
                        atom,
                        reason=f"structured_duplicate:{key[0]}",
                        superseded_by=kept["id"],
                        actor=actor,
                    )
                    archived_ids.add(duplicate["id"])
                    projected_atoms.append(duplicate)
                    projected_edges.extend(deleted_edges)
                    actions.append(
                        {
                            "action": "archive_structured_duplicate",
                            "kind": key[0],
                            "kept": kept["id"],
                            "archived": duplicate["id"],
                            "deleted_edge_count": len(deleted_edges),
                        }
                    )

            contradiction_groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
            for atom in atoms:
                if atom["id"] in archived_ids:
                    continue
                signature = self._contradiction_signature(atom)
                if signature is None:
                    continue
                key, value = signature
                contradiction_groups.setdefault(key, {})[value] = atom
            for values in contradiction_groups.values():
                if len(values) < 2:
                    continue
                group_atoms = list(values.values())
                if approved_by:
                    for atom in group_atoms:
                        changed = dict(atom)
                        changed["health_status"] = "contradicted"
                        changed["version"] = int(changed["version"]) + 1
                        changed["updated_at"] = utc_now()
                        changed = normalize_atom(
                            self._attach_search_index(changed), require_id=True
                        )
                        self.store.replace_atom(conn, changed)
                        projected_atoms.append(changed)
                    for source in group_atoms:
                        for target in group_atoms:
                            if source["id"] >= target["id"]:
                                continue
                            edge = self._edge(
                                source["id"], target["id"], "rel:contradicts", scope
                            )
                            self.store.insert_edge(conn, edge)
                            projected_edges.append(edge)
                actions.append(
                    {
                        "action": "mark_contradiction"
                        if approved_by
                        else "propose_contradiction_review",
                        "atom_refs": [atom["id"] for atom in group_atoms],
                        "review_required": approved_by is None,
                        "approved_by": approved_by,
                        "smp_outputs": [
                            output
                            for output in smp_outputs
                            if output["reason_code"] == "contradiction_candidate"
                            and any(ref in output["input_refs"] for ref in [atom["id"] for atom in group_atoms])
                        ],
                    }
                )

            event = self.store.append_event(
                conn,
                event_type="steward_run",
                actor=actor,
                payload={
                    "operation": "run_steward",
                    "scope": scope,
                    "actions": actions,
                    "projected_atoms": projected_atoms,
                    "projected_edges": projected_edges,
                },
                target_refs=[
                    ref
                    for action in actions
                    for ref in action.get("atom_refs", [])
                    + [action.get("kept"), action.get("archived")]
                    if ref
                ],
                authorization_context={"approved_by": approved_by}
                if approved_by
                else {},
            )
            self.store.clear_packet_cache(conn)
        return {
            "status": "completed",
            "actions": actions,
            "event": event,
            "graph_version": self.store.graph_version(),
        }


    def _maintenance_evidence_window(
        self,
        *,
        scope: Mapping[str, Any],
        domain: str,
        max_atoms: int,
        max_events: int,
        max_retrieval_outcomes: int,
    ) -> EvidenceWindow:
        # Maintenance scopes are hierarchical: a broad tenant/component pass
        # must see atoms in narrower run, asset, and agent scopes. Filter before
        # applying the window bound so unrelated hot atoms cannot crowd the
        # requested scope out of the evidence window.
        visible_atoms = [
            atom
            for atom in self.store.list_atoms_filtered(prioritize_hot=True)
            if not atom.get("deleted")
            and maintenance_scope_visible(atom["scope"], scope)
        ]
        atoms = visible_atoms[: max(1, int(max_atoms or 1))]
        atom_refs = {atom["id"] for atom in atoms}
        edges = [
            edge
            for edge in self.store.list_edges()
            if edge["source_ref"] in atom_refs or edge["target_ref"] in atom_refs
        ]
        evidence = [
            record
            for record in self.store.list_evidence()
            if scope_visible(record.get("scope", {}), scope)
        ]
        event_limit = max(0, int(max_events or 0))
        events = self.store.list_events(limit=event_limit) if event_limit else []
        list_outcomes = getattr(self.store, "list_retrieval_outcomes", None)
        retrieval_outcomes = (
            list_outcomes(limit=max(0, int(max_retrieval_outcomes or 0)))
            if list_outcomes
            else []
        )
        return EvidenceWindow(
            atoms=tuple(atoms),
            edges=tuple(edges),
            evidence=tuple(evidence),
            retrieval_outcomes=tuple(retrieval_outcomes),
            events=tuple(events),
            scope=scope,
            domain=str(domain or "generic"),
            graph_version=self.store.graph_version(),
        )


    def _commit_maintenance_proposal(
        self, proposal: Mapping[str, Any], *, actor: str
    ) -> dict[str, Any]:
        if proposal.get("action") == "add_edge":
            return self._commit_maintenance_edge_proposal(proposal, actor=actor)
        atom_payload = proposal.get("payload", {}).get("atom")
        if not isinstance(atom_payload, Mapping):
            return {
                "status": "skipped",
                "reason": "proposal_has_no_atom_payload",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        atom = dict(atom_payload)
        atom["id"] = atom.get("id") or stable_id(
            "atom", {"maintenance_proposal_id": proposal["proposal_id"]}
        )
        atom.setdefault("evidence_refs", list(proposal.get("evidence_refs", [])))
        atom_payload_body = dict(atom.get("payload", {}))
        atom_payload_body.setdefault(
            "maintenance_proposal_id", proposal["proposal_id"]
        )
        atom_payload_body.setdefault("maintenance_reason_code", proposal["reason_code"])
        atom_payload_body.setdefault("maintenance_source_refs", proposal["source_refs"])
        atom["payload"] = atom_payload_body
        existing = self.store.get_atom(str(atom["id"]))
        if existing is not None and not existing.get("deleted"):
            return {
                "status": "already_committed",
                "proposal_id": proposal["proposal_id"],
                "atom": existing,
                "source_refs": list(proposal.get("source_refs", [])),
            }
        committed = self.commit_atom(
            atom,
            actor=actor,
            idempotency_key=stable_id(
                "maint_commit", {"proposal_id": proposal["proposal_id"]}
            ),
            authorization_context={
                "maintenance_proposal_id": proposal["proposal_id"],
                "maintenance_processor_id": proposal["processor_id"],
                "risk_level": proposal["risk_level"],
                "auto_commit_gate": "low_risk_add_atom",
            },
        )
        return {
            "status": "committed",
            "proposal_id": proposal["proposal_id"],
            "atom": committed["atom"],
            "event": committed["event"],
            "source_refs": list(proposal.get("source_refs", [])),
        }


    def _commit_maintenance_edge_proposal(
        self, proposal: Mapping[str, Any], *, actor: str
    ) -> dict[str, Any]:
        edge_payload = proposal.get("payload", {}).get("edge")
        if not isinstance(edge_payload, Mapping):
            return {
                "status": "skipped",
                "reason": "proposal_has_no_edge_payload",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        source_ref = str(edge_payload.get("source_ref", ""))
        target_ref = str(edge_payload.get("target_ref", ""))
        relation = normalize_relation(str(edge_payload.get("relation", "")))
        if not source_ref or not target_ref or source_ref == target_ref:
            return {
                "status": "skipped",
                "reason": "invalid_edge_endpoints",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        source = self.store.get_atom(source_ref)
        target = self.store.get_atom(target_ref)
        if (
            source is None
            or target is None
            or source.get("deleted")
            or target.get("deleted")
            or source.get("lifecycle_state") != "active"
            or target.get("lifecycle_state") != "active"
        ):
            return {
                "status": "skipped",
                "reason": "edge_endpoint_not_active",
                "proposal_id": proposal["proposal_id"],
                "source_refs": list(proposal.get("source_refs", [])),
            }
        edge = self._edge(
            source_ref,
            target_ref,
            relation,
            dict(edge_payload.get("scope") or {}),
        )
        edge["evidence_refs"] = [
            str(ref) for ref in edge_payload.get("evidence_refs", [])
        ]
        edge["confidence"] = dict(
            edge_payload.get("confidence")
            or {"level": "medium-high", "score": proposal.get("confidence", 0.75)}
        )
        existing_edge = self.store.get_edge(str(edge["edge_id"]))
        if existing_edge and not existing_edge.get("deleted") and existing_edge.get(
            "lifecycle_state", "active"
        ) == "active":
            return {
                "status": "already_committed",
                "proposal_id": proposal["proposal_id"],
                "edge": existing_edge,
                "source_refs": list(proposal.get("source_refs", [])),
            }
        with self.store.transaction() as conn:
            if existing_edge:
                edge = {
                    **edge,
                    "created_at": existing_edge["created_at"],
                    "updated_at": utc_now(),
                    "version": int(existing_edge.get("version", 1)) + 1,
                    "deleted": False,
                    "lifecycle_state": "active",
                }
                self.store.upsert_edge(conn, edge)
            else:
                inserted = self.store.insert_edge(conn, edge)
                if not inserted:
                    return {
                        "status": "already_committed",
                        "proposal_id": proposal["proposal_id"],
                        "edge": edge,
                        "source_refs": list(proposal.get("source_refs", [])),
                    }
            event = self.store.append_event(
                conn,
                event_type="edge_committed",
                actor=actor,
                payload={
                    "operation": "commit_maintenance_edge",
                    "edge": edge,
                    "projected_edges": [edge],
                    "maintenance_proposal_id": proposal["proposal_id"],
                    "maintenance_reason_code": proposal["reason_code"],
                },
                target_refs=[source_ref, target_ref],
                authorization_context={
                    "maintenance_proposal_id": proposal["proposal_id"],
                    "maintenance_processor_id": proposal["processor_id"],
                    "risk_level": proposal["risk_level"],
                    "auto_commit_gate": "low_risk_add_edge",
                },
            )
            self.store.clear_packet_cache(conn)
        return {
            "status": "committed",
            "proposal_id": proposal["proposal_id"],
            "edge": edge,
            "event": event,
            "source_refs": list(proposal.get("source_refs", [])),
        }


    def _maintenance_reviewer_status(
        self, reviewer: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        config = dict(reviewer or {})
        enabled = bool(config.get("enabled", False))
        return {
            "enabled": enabled,
            "authority": "draft_only",
            "status": "not_configured" if enabled else "disabled",
            "mutates_canonical_memory": False,
            "allowed_outputs": [
                "proposal_explanation",
                "ambiguous_atomization_note",
                "scope_refinement_suggestion",
                "contradiction_analysis_draft",
            ],
        }


    def _maintenance_distiller_blocked_state_key(
        self,
        *,
        scope: Mapping[str, Any],
        domain: str,
        processor_ids: Sequence[str],
    ) -> str:
        return "maintenance_distiller_blocked:" + stable_id(
            "mdblk",
            {
                "scope": dict(scope),
                "domain": domain,
                "processor_ids": sorted(str(item) for item in processor_ids),
            },
        )


    def _maintenance_proposal_fingerprint(self, proposal: Mapping[str, Any]) -> str:
        return digest(
            {
                "action": proposal.get("action"),
                "risk_level": proposal.get("risk_level"),
                "source_refs": sorted(str(ref) for ref in proposal.get("source_refs", [])),
                "target_refs": sorted(str(ref) for ref in proposal.get("target_refs", [])),
                "payload": proposal.get("payload", {}),
                "recommended_action": proposal.get("recommended_action"),
                "reason_code": proposal.get("reason_code"),
                "output_type": proposal.get("output_type"),
            }
        )


    def _maintenance_distiller_blocked_fingerprint(
        self,
        *,
        scope: Mapping[str, Any],
        domain: str,
        processor_ids: Sequence[str],
        missing_processors: Sequence[str],
        committed: Sequence[Mapping[str, Any]],
        deferred: Sequence[Mapping[str, Any]],
        reviewer_status: Mapping[str, Any],
        auto_commit_low_risk: bool,
    ) -> str:
        return digest(
            {
                "scope": dict(scope),
                "domain": domain,
                "processor_ids": sorted(str(item) for item in processor_ids),
                "missing_processors": sorted(str(item) for item in missing_processors),
                "commit_eligible": sorted(
                    str(item.get("proposal_id"))
                    for item in committed
                    if item.get("proposal_id")
                ),
                "deferred": sorted(
                    (
                        {
                            "proposal_id": str(item.get("proposal_id")),
                            "action": str(item.get("action")),
                            "risk_level": str(item.get("risk_level")),
                            "reason": str(item.get("reason")),
                            "proposal_digest": str(item.get("proposal_digest") or ""),
                            "source_refs": sorted(
                                str(ref) for ref in item.get("source_refs", [])
                            ),
                        }
                        for item in deferred
                    ),
                    key=lambda item: (
                        item["proposal_id"],
                        item["action"],
                        item["risk_level"],
                    ),
                ),
                "reviewer_status": {
                    "enabled": bool(reviewer_status.get("enabled", False)),
                    "authority": str(reviewer_status.get("authority", "")),
                    "status": str(reviewer_status.get("status", "")),
                },
                "auto_commit_low_risk": bool(auto_commit_low_risk),
            }
        )
