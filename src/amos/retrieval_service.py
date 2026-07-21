"""RetrievalService implementation for the AMOS service facade."""

from ._service_support import (
    ATTENTION_POLICY_ID,
    Any,
    CONFLICT_RELATIONS,
    DEFAULT_PACKET_PROFILES,
    LOW_HEALTH_STATES,
    Mapping,
    RETRIEVAL_RECENCY_HORIZON_SECONDS,
    RETRIEVAL_WEIGHTS,
    SCHEMA_VERSION,
    SEMANTIC_MATCH_THRESHOLD,
    Sequence,
    access_visible,
    canonical_json,
    confidence_score,
    cosine,
    normalize_atom,
    payload_agent_id,
    re,
    scope_visible,
    stable_id,
    utc_now,
)


class RetrievalService:
    def __init__(
        self,
        store: Any,
        smp: Any,
        access: Any,
        indexes: Any,
        graph: Any,
        capacity: Any,
        temporal: Any,
        policy_runner: Any,
    ):
        self.store = store
        self.smp = smp
        self._mark_foreground_activity = access._mark_foreground_activity
        self._attach_search_index = indexes._attach_search_index
        self._atom_search_index = indexes._atom_search_index
        self._sync_smp_vector_model = indexes._sync_smp_vector_model
        self._indexed_retrieval_candidates = indexes._indexed_retrieval_candidates
        self._active_superseded_refs = graph._active_superseded_refs
        self._hot_graph_edge_degree_counts = graph._hot_graph_edge_degree_counts
        self._hot_graph_edge_visible = graph._hot_graph_edge_visible
        self._edge_relation_activation_weight = graph._edge_relation_activation_weight
        self._render_atom = graph._render_atom
        self._capacity_pressure_mode = capacity._capacity_pressure_mode
        self._seconds_since = temporal._seconds_since
        self.run_memory_policy = policy_runner

    def retrieve_packet(
        self,
        *,
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "reasoner",
        retrieval_mode: str = "general",
        max_items: int | None = None,
        token_or_byte_budget: int | Mapping[str, int] | None = None,
        include_conflicts: bool | None = None,
        include_archived: bool = False,
        include_low_health: bool = False,
        include_superseded: bool = False,
        type_filter: Sequence[str] | None = None,
        attention_context: Mapping[str, Any] | None = None,
        run_policy: bool = True,
    ) -> dict[str, Any]:
        self._mark_foreground_activity(requester)
        if run_policy:
            self.run_memory_policy(trigger="retrieve_packet", scope=scope or {})
        profile = DEFAULT_PACKET_PROFILES.get(
            retrieval_mode, DEFAULT_PACKET_PROFILES.get(target_processor, {})
        )
        if max_items is None:
            max_items = int(profile.get("max_items", 8))
        if token_or_byte_budget is None and "tokens" in profile:
            token_or_byte_budget = {"tokens": int(profile["tokens"])}
        if include_conflicts is None:
            include_conflicts = bool(profile.get("include_conflicts", False))
        pressure_mode = self._capacity_pressure_mode()
        pressure_degraded = pressure_mode in {"orange", "red"}
        original_max_items = max_items
        if pressure_mode == "orange":
            max_items = max(1, max_items // 2)
        elif pressure_mode == "red":
            max_items = max(1, min(max_items, 3))
        attention_policy = self._attention_policy(attention_context)
        request = {
            "cues": list(cues or []),
            "scope": dict(scope or {}),
            "requester": requester,
            "target_processor": target_processor,
            "retrieval_mode": retrieval_mode,
            "max_items": max_items,
            "token_or_byte_budget": token_or_byte_budget,
            "include_conflicts": include_conflicts,
            "include_archived": include_archived,
            "include_low_health": include_low_health,
            "include_superseded": include_superseded,
            "type_filter": list(type_filter or []),
            "attention_context": attention_policy["context"],
            "pressure_mode": pressure_mode,
            "run_policy": bool(run_policy),
        }
        graph_version = self.store.graph_version()
        cached = self.store.get_cached_packet(
            request=request, graph_version=graph_version
        )
        if cached is not None:
            return cached
        self._sync_smp_vector_model(graph_version=graph_version)

        candidates: list[tuple[float, dict[str, Any]]] = []
        omissions: list[dict[str, Any]] = []
        allowed_types = set(type_filter or [])
        lifecycle_states = ["active", "proposed"]
        if include_archived:
            lifecycle_states.append("archived")
        cue_text = " ".join(request["cues"]).lower()
        cue_tokens = {token for token in re.findall(r"[a-z0-9_]+", cue_text) if token}
        all_atoms = self.store.list_atoms_filtered(
            types=sorted(allowed_types) if allowed_types else None,
            lifecycle_states=lifecycle_states,
        )
        eligible_atoms: list[dict[str, Any]] = []
        for atom in all_atoms:
            atom_ref = str(atom["id"])
            if not scope_visible(atom["scope"], request["scope"]):
                omissions.append({"atom_ref": atom_ref, "reason": "scope_hidden"})
            elif not access_visible(
                atom["access_policy"], requester, target_processor
            ):
                omissions.append({"atom_ref": atom_ref, "reason": "access_hidden"})
            elif atom["health_status"] == "contradicted" and not include_conflicts:
                omissions.append({"atom_ref": atom_ref, "reason": "contradicted"})
            elif atom["health_status"] in LOW_HEALTH_STATES and not include_low_health:
                omissions.append(
                    {
                        "atom_ref": atom_ref,
                        "reason": f"health:{atom['health_status']}",
                    }
                )
            else:
                eligible_atoms.append(atom)
        eligible_atom_ids = {str(atom["id"]) for atom in eligible_atoms}
        semantic_query_text = " ".join(
            [
                cue_text,
                *[
                    str(term)
                    for term in attention_policy.get("focus_terms", []) or []
                    if str(term).strip()
                ],
            ]
        ).strip()
        cue_vector = self.smp.encode(semantic_query_text) if semantic_query_text else []
        indexed_candidate_ids = self._indexed_retrieval_candidates(
            cue_tokens=cue_tokens,
            attention_policy=attention_policy,
            eligible_atom_ids=eligible_atom_ids,
        )
        latent_candidate_ids = self._latent_retrieval_candidates(
            eligible_atoms,
            cue_vector=cue_vector,
            limit=max(64, int(max_items) * 8),
            minimum_similarity=(
                0.55 if indexed_candidate_ids else SEMANTIC_MATCH_THRESHOLD
            ),
        )
        if indexed_candidate_ids is None and not latent_candidate_ids:
            atoms = eligible_atoms
        else:
            candidate_atom_ids = set(indexed_candidate_ids or [])
            candidate_atom_ids.update(latent_candidate_ids)
            atoms = self.store.list_atoms_filtered(
                types=sorted(allowed_types) if allowed_types else None,
                lifecycle_states=lifecycle_states,
                atom_ids=sorted(candidate_atom_ids.intersection(eligible_atom_ids)),
            )
        atom_refs = [str(atom["id"]) for atom in atoms]
        superseded_refs = self._active_superseded_refs(atom_refs)
        edge_degrees = self._hot_graph_edge_degree_counts(atoms)
        edge_activation_scores, association_traces = self._graph_activation_scores(
            atoms,
            cues=request["cues"],
            request_scope=request["scope"],
            requester=requester,
            target_processor=target_processor,
            include_conflicts=bool(include_conflicts),
            include_low_health=bool(include_low_health),
            cue_text=cue_text,
            cue_tokens=cue_tokens,
            attention_policy=attention_policy,
            superseded_refs=superseded_refs if not include_superseded else None,
        )
        for atom in atoms:
            atom_ref = atom["id"]
            if atom.get("deleted"):
                omissions.append({"atom_ref": atom_ref, "reason": "deleted"})
                continue
            if not scope_visible(atom["scope"], request["scope"]):
                omissions.append({"atom_ref": atom_ref, "reason": "scope_hidden"})
                continue
            if not access_visible(atom["access_policy"], requester, target_processor):
                omissions.append({"atom_ref": atom_ref, "reason": "access_hidden"})
                continue
            if atom["health_status"] == "contradicted" and not include_conflicts:
                omissions.append({"atom_ref": atom_ref, "reason": "contradicted"})
                continue
            if atom["health_status"] in LOW_HEALTH_STATES and not include_low_health:
                omissions.append(
                    {"atom_ref": atom_ref, "reason": f"health:{atom['health_status']}"}
                )
                continue
            if atom_ref in superseded_refs and not include_superseded:
                omissions.append(
                    {
                        "atom_ref": atom_ref,
                        "reason": "superseded",
                        "superseded_by": superseded_refs[atom_ref][:5],
                    }
                )
                continue
            score, matched, components = self._rank_atom(
                atom,
                request["cues"],
                request_scope=request["scope"],
                retrieval_mode=retrieval_mode,
                cue_text=cue_text,
                cue_tokens=cue_tokens,
                cue_vector=cue_vector,
                edge_degrees=edge_degrees,
                edge_activation_scores=edge_activation_scores,
                attention_policy=attention_policy,
                superseded_refs=superseded_refs,
            )
            if request["cues"] and not matched:
                omissions.append({"atom_ref": atom_ref, "reason": "low_relevance"})
                continue
            atom = {
                **atom,
                "_score_components": components,
                "_association_trace": association_traces.get(atom_ref, []),
            }
            candidates.append((score, atom))

        candidates.sort(key=lambda item: item[0], reverse=True)
        byte_budget = self._byte_budget(token_or_byte_budget)
        used_bytes = 0
        items = []
        for score, atom in candidates:
            if len(items) >= max_items:
                omissions.append(
                    {
                        "atom_ref": atom["id"],
                        "reason": "pressure_degraded"
                        if pressure_degraded and len(items) >= max_items
                        else "budget_exhausted",
                    }
                )
                continue
            item, evidence_omissions = self._packet_item(
                atom, score, requester=requester, target_processor=target_processor
            )
            omissions.extend(evidence_omissions)
            rendered_size = len(canonical_json(item).encode("utf-8"))
            if used_bytes + rendered_size > byte_budget:
                omissions.append({"atom_ref": atom["id"], "reason": "budget_exhausted"})
                continue
            used_bytes += rendered_size
            items.append(item)
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank

        conflicts = []
        if include_conflicts and items:
            selected = {item["atom_ref"] for item in items}
            for edge in self.store.list_edges_for_refs(sorted(selected)):
                if edge["relation"] not in CONFLICT_RELATIONS:
                    continue
                if edge["source_ref"] in selected or edge["target_ref"] in selected:
                    conflicts.append(edge)

        packet = {
            "packet_id": stable_id(
                "pkt",
                {"request": request, "graph_version": graph_version, "items": items},
            ),
            "schema_version": SCHEMA_VERSION,
            "request": request,
            "graph_version": graph_version,
            "generated_at": utc_now(),
            "target_processor": target_processor,
            "retrieval_mode": retrieval_mode,
            "scope": dict(scope or {}),
            "pressure_mode": pressure_mode,
            "items": items,
            "omissions": omissions,
            "conflicts": conflicts,
            "degradation": {
                "mode": "smp-deterministic-local",
                "pressure_mode": pressure_mode,
                "reduced_recall_depth": pressure_degraded
                and max_items < original_max_items,
                "omitted_evidence_detail": any(
                    omission["reason"] == "evidence_access_denied"
                    for omission in omissions
                ),
                "index_freshness": {
                    "semantic_index": "inline_rebuildable",
                    "graph_version": graph_version,
                },
                "reason_codes": sorted({omission["reason"] for omission in omissions}),
                "vector_index_available": bool(semantic_query_text),
                "candidate_generation": {
                    "eligible_count": len(eligible_atoms),
                    "lexical_count": len(indexed_candidate_ids or []),
                    "latent_count": len(latent_candidate_ids),
                    "union_count": len(atoms),
                    "lexical_profile": "content_only_idf_weighted",
                    "latent_profile": "independent_smp_candidate_pool",
                },
                "byte_budget": byte_budget,
                "used_bytes": used_bytes,
            },
            "attention_trace": self._attention_trace(
                attention_policy=attention_policy,
                items=items,
                candidates=candidates,
                omissions=omissions,
            ),
            "provenance": {
                "store": getattr(self.store, "backend_name", "unknown"),
                "journal_head": self.store.last_event_hash(),
                "ranker_profile_id": "amos.v1.default",
                "smp_processor_id": self.smp.processor_id,
            },
            "cache_policy": {"cacheable": True, "keyed_by_graph_version": True},
        }
        with self.store.transaction() as conn:
            self.store.cache_packet(
                conn,
                packet_id=packet["packet_id"],
                request=request,
                response=packet,
                graph_version=graph_version,
            )
        return packet


    def _latent_retrieval_candidates(
        self,
        atoms: Sequence[Mapping[str, Any]],
        *,
        cue_vector: Sequence[float],
        limit: int,
        minimum_similarity: float,
    ) -> list[str]:
        """Select an independent semantic pool before lexical capping."""

        if not cue_vector:
            return []
        scored: list[tuple[float, str]] = []
        for atom in atoms:
            search_index = self._atom_search_index(atom, allow_stale=True)
            similarity = cosine(cue_vector, search_index.get("vector") or [])
            if similarity >= float(minimum_similarity):
                scored.append((similarity, str(atom["id"])))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [atom_ref for _, atom_ref in scored[: max(1, int(limit))]]


    def record_retrieval_outcome(
        self,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._mark_foreground_activity(str(request.get("requester") or "system"))
        with self.store.transaction() as conn:
            record = self.store.insert_retrieval_outcome(
                conn, packet_id=packet_id, request=request, outcome=outcome
            )
            if record.get("status") != "recorded":
                return record
            feedback = self._apply_retrieval_outcome_feedback(
                conn,
                packet_id=packet_id,
                request=request,
                outcome=outcome,
            )
            if feedback["updated_atoms"] or feedback["updated_edges"]:
                event = self.store.append_event(
                    conn,
                    event_type="retrieval_outcome_recorded",
                    actor=str(request.get("requester") or "system"),
                    payload={
                        "operation": "record_retrieval_outcome",
                        "packet_id": packet_id,
                        "outcome_id": record["outcome_id"],
                        "feedback": feedback,
                        "projected_atoms": feedback["projected_atoms"],
                    },
                    target_refs=[
                        *feedback["updated_atom_refs"],
                        *feedback["updated_edge_refs"],
                    ],
                )
                self.store.clear_packet_cache(conn)
                record["event"] = event
            record["feedback"] = feedback
            return record


    def _apply_retrieval_outcome_feedback(
        self,
        conn: Any,
        *,
        packet_id: str,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        del request
        reported_positive, reported_corrections = self._retrieval_outcome_atom_refs(
            outcome
        )
        packet = self.store.get_cached_packet_by_id(packet_id)
        packet_items = {
            str(item.get("atom_ref") or item.get("atom_id") or item.get("item_ref")): item
            for item in (packet or {}).get("items", [])
            if isinstance(item, Mapping)
            and (item.get("atom_ref") or item.get("atom_id") or item.get("item_ref"))
        }
        packet_refs = set(packet_items)
        positive_refs = reported_positive.intersection(packet_refs)
        correction_refs = reported_corrections.intersection(packet_refs)
        ignored_refs = sorted(
            reported_positive.union(reported_corrections) - packet_refs
        )
        label = str(outcome.get("label") or outcome.get("status") or "").lower()
        negative_label = label in {
            "bad",
            "wrong",
            "unused",
            "unhelpful",
            "misleading",
            "corrected",
            "correction",
            "failed",
            "failure",
        }
        now = utc_now()
        updated_atoms: list[dict[str, Any]] = []
        updated_edges: list[dict[str, Any]] = []
        projected_atoms: list[dict[str, Any]] = []
        for atom_ref in sorted(positive_refs.union(correction_refs)):
            atom = self.store.get_atom(atom_ref)
            if atom is None or atom.get("deleted"):
                continue
            changed = dict(atom)
            telemetry = dict(changed.get("decay_policy") or {}).get(
                "retrieval_telemetry", {}
            )
            telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
            used_count = int(telemetry.get("used_count", 0) or 0)
            correction_count = int(telemetry.get("correction_count", 0) or 0)
            if atom_ref in positive_refs:
                used_count += 1
            if atom_ref in correction_refs or negative_label:
                correction_count += 1
            delta = 0.0
            if atom_ref in positive_refs and not negative_label:
                delta += 0.03
            if atom_ref in correction_refs or negative_label:
                delta -= 0.06
            changed["utility"] = max(0.0, min(1.0, float(changed["utility"]) + delta))
            if atom_ref in positive_refs and not negative_label:
                changed["salience"] = max(
                    0.0, min(1.0, float(changed["salience"]) + 0.02)
                )
            if changed["utility"] < 0.25 and changed["health_status"] == "healthy":
                changed["health_status"] = "low_utility"
            telemetry.update(
                {
                    "used_count": used_count,
                    "correction_count": correction_count,
                    "last_outcome_label": label or None,
                    "last_outcome_at": now,
                }
            )
            changed["decay_policy"] = {
                **dict(changed.get("decay_policy") or {}),
                "retrieval_telemetry": telemetry,
            }
            changed["last_accessed"] = now
            changed["updated_at"] = now
            changed["version"] = int(changed["version"]) + 1
            changed = normalize_atom(self._attach_search_index(changed), require_id=True)
            self.store.replace_atom(conn, changed)
            projected_atoms.append(changed)
            updated_atoms.append(
                {
                    "atom_ref": atom_ref,
                    "utility": changed["utility"],
                    "salience": changed["salience"],
                    "health_status": changed["health_status"],
                    "used_count": used_count,
                    "correction_count": correction_count,
                }
            )
        edge_feedback: dict[str, dict[str, bool]] = {}
        for atom_ref in sorted(positive_refs.union(correction_refs)):
            for step in packet_items.get(atom_ref, {}).get("association_trace", []) or []:
                if not isinstance(step, Mapping) or not step.get("edge_id"):
                    continue
                edge_id = str(step["edge_id"])
                state = edge_feedback.setdefault(
                    edge_id, {"used": False, "corrected": False}
                )
                state["used"] = state["used"] or atom_ref in positive_refs
                state["corrected"] = (
                    state["corrected"]
                    or atom_ref in correction_refs
                    or negative_label
                )
        for edge_id, edge_state in sorted(edge_feedback.items()):
            edge = self.store.get_edge(edge_id)
            if edge is None or edge.get("deleted"):
                continue
            changed_edge = dict(edge)
            derivation = dict(changed_edge.get("derivation") or {})
            telemetry = derivation.get("retrieval_telemetry", {})
            telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
            used_count = int(telemetry.get("used_count", 0) or 0)
            correction_count = int(telemetry.get("correction_count", 0) or 0)
            if edge_state["used"] and not negative_label:
                used_count += 1
            if edge_state["corrected"]:
                correction_count += 1
            telemetry.update(
                {
                    "used_count": used_count,
                    "correction_count": correction_count,
                    "last_outcome_label": label or None,
                    "last_outcome_at": now,
                }
            )
            derivation["retrieval_telemetry"] = telemetry
            changed_edge["derivation"] = derivation
            changed_edge["updated_at"] = now
            changed_edge["version"] = int(changed_edge["version"]) + 1
            self.store.upsert_edge(conn, changed_edge)
            updated_edges.append(
                {
                    "edge_id": edge_id,
                    "used_count": used_count,
                    "correction_count": correction_count,
                }
            )
        return {
            "updated_atom_refs": [item["atom_ref"] for item in updated_atoms],
            "updated_atoms": updated_atoms,
            "projected_atoms": projected_atoms,
            "positive_refs": sorted(positive_refs),
            "correction_refs": sorted(correction_refs),
            "updated_edge_refs": [item["edge_id"] for item in updated_edges],
            "updated_edges": updated_edges,
            "ignored_non_packet_refs": ignored_refs,
            "reported_evidence_refs": self._retrieval_outcome_evidence_refs(outcome),
            "feedback_contract": "packet_items_only",
        }


    def _retrieval_outcome_atom_refs(
        self, outcome: Mapping[str, Any]
    ) -> tuple[set[str], set[str]]:
        positive: set[str] = set()
        corrections: set[str] = set()

        def add_refs(target: set[str], value: Any) -> None:
            if isinstance(value, str):
                if value:
                    target.add(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    add_refs(target, item)

        for key in (
            "used_item_refs",
            "used_atom_refs",
            "cited_atom_refs",
            "selected_item_refs",
            "helpful_atom_refs",
        ):
            add_refs(positive, outcome.get(key))
        add_refs(positive, outcome.get("cited_atom_ref"))
        add_refs(positive, outcome.get("used_atom_ref"))
        for key in (
            "correction_refs",
            "corrected_atom_refs",
            "misleading_atom_refs",
            "unhelpful_atom_refs",
        ):
            add_refs(corrections, outcome.get(key))
        add_refs(corrections, outcome.get("corrected_atom_ref"))
        return positive, corrections


    def _retrieval_outcome_evidence_refs(
        self, outcome: Mapping[str, Any]
    ) -> list[str]:
        refs: set[str] = set()

        def add(value: Any) -> None:
            if isinstance(value, str) and value:
                refs.add(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    add(item)

        for key in ("used_evidence_refs", "cited_evidence_refs", "evidence_refs"):
            add(outcome.get(key))
        return sorted(refs)


    def _attention_policy(
        self, attention_context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        context = self._normalize_attention_context(attention_context)
        return {
            "policy_id": ATTENTION_POLICY_ID,
            "context": context,
            "focus_terms": context.get("focus_terms", []),
            "suppress_terms": context.get("suppress_terms", []),
            "boost_memory_types": context.get("boost_memory_types", []),
            "suppress_memory_types": context.get("suppress_memory_types", []),
            "counterevidence_required": bool(
                context.get("counterevidence_required", False)
            ),
            "weight_adjustments": {
                "attention_focus": RETRIEVAL_WEIGHTS["attention_focus"],
                "attention_type_boost": RETRIEVAL_WEIGHTS["attention_type_boost"],
                "attention_counterevidence": RETRIEVAL_WEIGHTS[
                    "attention_counterevidence"
                ],
                "attention_novelty": RETRIEVAL_WEIGHTS["attention_novelty"],
                "attention_suppression_penalty": RETRIEVAL_WEIGHTS[
                    "attention_suppression_penalty"
                ],
            },
        }


    def _normalize_attention_context(
        self, attention_context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if not isinstance(attention_context, Mapping):
            return {}
        context: dict[str, Any] = {}
        scalar_keys = (
            "active_task",
            "mission",
            "goal",
            "role",
            "risk_posture",
            "time_horizon",
        )
        for key in scalar_keys:
            value = attention_context.get(key)
            if value not in (None, "", [], {}):
                context[key] = value

        focus_terms = self._attention_terms(attention_context.get("focus_terms"))
        suppress_terms = self._attention_terms(attention_context.get("suppress_terms"))
        for key in ("active_task", "mission", "goal", "role", "task_context"):
            focus_terms.extend(self._attention_terms(attention_context.get(key)))

        context["focus_terms"] = sorted(set(focus_terms))
        context["suppress_terms"] = sorted(set(suppress_terms))
        context["boost_memory_types"] = sorted(
            set(self._attention_type_terms(attention_context.get("boost_memory_types")))
        )
        context["suppress_memory_types"] = sorted(
            set(
                self._attention_type_terms(
                    attention_context.get("suppress_memory_types")
                )
            )
        )
        risk_posture = str(context.get("risk_posture", "")).lower()
        context["counterevidence_required"] = bool(
            attention_context.get("counterevidence_required", False)
            or risk_posture in {"cautious", "high_risk", "high-risk", "critical"}
        )
        novelty = attention_context.get("novelty_preference")
        if novelty not in (None, ""):
            try:
                context["novelty_preference"] = max(0.0, min(1.0, float(novelty)))
            except (TypeError, ValueError):
                pass
        return {
            key: value
            for key, value in context.items()
            if value not in (None, "", [], {})
        }


    def _attention_terms(self, value: Any) -> list[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, Mapping):
            terms: list[str] = []
            for item in value.values():
                terms.extend(self._attention_terms(item))
            return terms
        if isinstance(value, (list, tuple, set)):
            terms = []
            for item in value:
                terms.extend(self._attention_terms(item))
            return terms
        text = str(value).lower()
        return [token for token in re.findall(r"[a-z0-9_]+", text) if token]


    def _attention_type_terms(self, value: Any) -> list[str]:
        known_types = {
            "belief",
            "preference",
            "goal",
            "commitment",
            "procedure",
            "capability",
            "limitation",
            "episode",
            "agentic_trace",
            "action_outcome",
            "self_model",
            "runtime_state",
            "self_assessment",
            "semantic",
            "policy",
        }
        return [
            token
            for token in self._attention_terms(value)
            if token in known_types
        ]


    def _attention_score_components(
        self,
        atom: Mapping[str, Any],
        *,
        text: str,
        text_tokens: set[str],
        edge_degree: int,
        attention_policy: Mapping[str, Any] | None,
        superseded_refs: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, float]:
        policy = attention_policy if isinstance(attention_policy, Mapping) else {}
        context = policy.get("context", {}) if isinstance(policy.get("context", {}), Mapping) else {}
        focus_terms = set(policy.get("focus_terms", []) or [])
        suppress_terms = set(policy.get("suppress_terms", []) or [])
        atom_type = str(atom.get("type", ""))
        focus_overlap = len(focus_terms.intersection(text_tokens))
        suppress_overlap = len(suppress_terms.intersection(text_tokens))
        direct_focus = any(term and term in text for term in focus_terms)
        direct_suppress = any(term and term in text for term in suppress_terms)
        attention_focus = 0.0
        if focus_terms:
            attention_focus = min(1.0, focus_overlap / max(1, len(focus_terms)))
            if direct_focus:
                attention_focus = max(attention_focus, 0.75)
        attention_suppression = 0.0
        if suppress_terms:
            attention_suppression = min(
                1.0, suppress_overlap / max(1, len(suppress_terms))
            )
            if direct_suppress:
                attention_suppression = max(attention_suppression, 0.75)
        attention_type_boost = (
            1.0 if atom_type in set(policy.get("boost_memory_types", []) or []) else 0.0
        )
        if atom_type in set(policy.get("suppress_memory_types", []) or []):
            attention_suppression = max(attention_suppression, 1.0)
        try:
            novelty_preference = max(
                0.0, min(1.0, float(context.get("novelty_preference", 0.0) or 0.0))
            )
        except (TypeError, ValueError):
            novelty_preference = 0.0
        novelty = 0.0
        if novelty_preference:
            graph_familiarity = min(1.0, max(0, int(edge_degree)) / 5.0)
            novelty = novelty_preference * (1.0 - graph_familiarity)
        counterevidence = 0.0
        if policy.get("counterevidence_required"):
            if atom.get("health_status") == "contradicted":
                counterevidence = 1.0
            elif atom_type in {"limitation", "self_assessment", "action_outcome"}:
                counterevidence = 0.6
            elif any(
                term in text_tokens
                for term in {
                    "failure",
                    "correction",
                    "blocked",
                    "risk",
                    "contradiction",
                }
            ):
                counterevidence = 0.5
        return {
            "attention_focus": attention_focus,
            "attention_type_boost": attention_type_boost,
            "attention_counterevidence": counterevidence,
            "attention_novelty": novelty,
            "attention_suppression_penalty": attention_suppression,
        }


    def _attention_trace(
        self,
        *,
        attention_policy: Mapping[str, Any],
        items: Sequence[Mapping[str, Any]],
        candidates: Sequence[tuple[float, Mapping[str, Any]]],
        omissions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        selected = {str(item.get("atom_ref")) for item in items if item.get("atom_ref")}
        inhibited = []
        for _, atom in candidates:
            atom_ref = str(atom.get("id", ""))
            if not atom_ref or atom_ref in selected:
                continue
            components = atom.get("_score_components", {})
            if float(components.get("attention_suppression_penalty", 0.0) or 0.0) > 0:
                inhibited.append(atom_ref)
        omitted_reasons: dict[str, int] = {}
        for omission in omissions:
            reason = str(omission.get("reason", "unknown"))
            omitted_reasons[reason] = omitted_reasons.get(reason, 0) + 1
        return {
            "policy_id": attention_policy.get("policy_id", ATTENTION_POLICY_ID),
            "context": dict(attention_policy.get("context", {})),
            "focus_terms": list(attention_policy.get("focus_terms", []) or []),
            "suppress_terms": list(attention_policy.get("suppress_terms", []) or []),
            "weight_adjustments": dict(
                attention_policy.get("weight_adjustments", {})
            ),
            "selected_item_refs": [item["atom_ref"] for item in items],
            "inhibited_refs": inhibited[:50],
            "omitted_reasons": omitted_reasons,
        }


    def _recency_score(self, atom: Mapping[str, Any]) -> float:
        seconds = self._seconds_since(atom.get("updated_at") or atom.get("observed_at"))
        if seconds is None:
            return 0.0
        return max(
            0.0,
            min(1.0, 1.0 - (float(seconds) / RETRIEVAL_RECENCY_HORIZON_SECONDS)),
        )


    def _graph_activation_scores(
        self,
        atoms: Sequence[Mapping[str, Any]],
        *,
        cues: Sequence[str],
        request_scope: Mapping[str, Any] | None,
        requester: str,
        target_processor: str,
        include_conflicts: bool,
        include_low_health: bool,
        cue_text: str,
        cue_tokens: set[str],
        attention_policy: Mapping[str, Any] | None,
        superseded_refs: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
        eligible_refs: set[str] = set()
        seed_strengths: dict[str, float] = {}
        for atom in atoms:
            atom_ref = str(atom.get("id") or "")
            if not atom_ref or atom.get("deleted"):
                continue
            if not scope_visible(atom["scope"], request_scope or {}):
                continue
            if not access_visible(atom["access_policy"], requester, target_processor):
                continue
            if atom["health_status"] == "contradicted" and not include_conflicts:
                continue
            if atom["health_status"] in LOW_HEALTH_STATES and not include_low_health:
                continue
            if superseded_refs and atom_ref in superseded_refs:
                continue
            eligible_refs.add(atom_ref)
            search_index = self._atom_search_index(atom, allow_stale=True)
            text = str(search_index["text"])
            text_tokens = set(str(token) for token in search_index["tokens"])
            direct = any(cue.lower() in text for cue in cues if cue)
            overlap = len(cue_tokens.intersection(text_tokens))
            cue_score = 1.0 if direct else min(1.0, overlap / max(1, len(cue_tokens)))
            attention = self._attention_score_components(
                atom,
                text=text,
                text_tokens=text_tokens,
                edge_degree=0,
                attention_policy=attention_policy,
            )
            seed = max(cue_score, float(attention.get("attention_focus", 0.0) or 0.0))
            if seed > 0:
                seed_strengths[atom_ref] = seed
        if not seed_strengths:
            return {}, {}

        atoms_by_ref = {str(atom.get("id") or ""): atom for atom in atoms}
        edges = []
        degree: dict[str, int] = {}
        for edge in self.store.list_edges_for_refs(sorted(eligible_refs)):
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source not in eligible_refs or target not in eligible_refs:
                continue
            if not self._hot_graph_edge_visible(edge, atoms_by_ref):
                continue
            edges.append(edge)
            degree[source] = degree.get(source, 0) + 1
            degree[target] = degree.get(target, 0) + 1

        adjacency: dict[str, list[tuple[str, float, Mapping[str, Any]]]] = {}
        for edge in edges:
            source = str(edge["source_ref"])
            target = str(edge["target_ref"])
            relation_weight = self._edge_relation_activation_weight(
                str(edge.get("relation") or "")
            )
            telemetry = (edge.get("derivation") or {}).get("retrieval_telemetry", {})
            telemetry = telemetry if isinstance(telemetry, Mapping) else {}
            used = int(telemetry.get("used_count", 0) or 0)
            corrected = int(telemetry.get("correction_count", 0) or 0)
            learned_weight = max(0.35, min(1.25, 1.0 + 0.03 * used - 0.08 * corrected))
            forward = relation_weight * learned_weight / max(1.0, degree[source] ** 0.5)
            reverse = relation_weight * learned_weight * 0.8 / max(1.0, degree[target] ** 0.5)
            adjacency.setdefault(source, []).append((target, forward, edge))
            adjacency.setdefault(target, []).append((source, reverse, edge))

        activation: dict[str, float] = {}
        traces: dict[str, list[dict[str, Any]]] = {}
        frontier: list[tuple[str, float, list[dict[str, Any]]]] = [
            (seed_ref, strength, [])
            for seed_ref, strength in seed_strengths.items()
        ]
        for depth in (1, 2):
            next_frontier: list[tuple[str, float, list[dict[str, Any]]]] = []
            depth_decay = 1.0 if depth == 1 else 0.55
            for source_ref, source_strength, source_trace in frontier:
                for target_ref, edge_weight, edge in adjacency.get(source_ref, []):
                    if any(
                        step.get("edge_id") == edge.get("edge_id")
                        for step in source_trace
                    ):
                        continue
                    strength = min(1.0, source_strength * edge_weight * depth_decay)
                    if strength <= activation.get(target_ref, 0.0):
                        continue
                    step = {
                        "edge_id": str(edge.get("edge_id") or ""),
                        "relation": str(edge.get("relation") or ""),
                        "source_ref": source_ref,
                        "target_ref": target_ref,
                        "depth": depth,
                    }
                    trace = [*source_trace, step]
                    activation[target_ref] = strength
                    traces[target_ref] = trace
                    next_frontier.append((target_ref, strength, trace))
            frontier = next_frontier
            if not frontier:
                break
        return activation, traces


    def _rank_atom(
        self,
        atom: Mapping[str, Any],
        cues: Sequence[str],
        *,
        request_scope: Mapping[str, Any] | None = None,
        retrieval_mode: str = "general",
        cue_text: str | None = None,
        cue_tokens: set[str] | None = None,
        cue_vector: Sequence[float] | None = None,
        edge_degrees: Mapping[str, int] | None = None,
        edge_activation_scores: Mapping[str, float] | None = None,
        attention_policy: Mapping[str, Any] | None = None,
        superseded_refs: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[float, bool, dict[str, float]]:
        search_index = self._atom_search_index(atom, allow_stale=True)
        text = str(search_index["text"])
        cue_text = " ".join(cues).lower() if cue_text is None else cue_text
        cue_tokens = (
            {token for token in re.findall(r"[a-z0-9_]+", cue_text) if token}
            if cue_tokens is None
            else cue_tokens
        )
        text_tokens = set(str(token) for token in search_index["tokens"])
        direct = any(cue.lower() in text for cue in cues if cue)
        overlap = len(cue_tokens.intersection(text_tokens))
        matched = direct or overlap > 0 or not cue_tokens
        semantic_similarity = 0.0
        if cue_text:
            cue_vector = self.smp.encode(cue_text) if cue_vector is None else cue_vector
            semantic_similarity = cosine(cue_vector, search_index["vector"])
            matched = matched or semantic_similarity >= SEMANTIC_MATCH_THRESHOLD
        direct_score = 1.0 if direct else min(1.0, overlap / max(1, len(cue_tokens)))
        edge_degree = int((edge_degrees or {}).get(atom["id"], 0))
        edge_activation = min(
            1.0, max(0.0, float((edge_activation_scores or {}).get(atom["id"], 0.0)))
        )
        matched = matched or edge_activation > 0.0
        recency = self._recency_score(atom)
        confidence = confidence_score(atom["confidence"])
        utility = min(1.0, float(atom["utility"]))
        salience = min(1.0, float(atom["salience"]))
        request_scope = dict(request_scope or {})
        scope_specificity = (
            min(1.0, len(atom["scope"]) / max(1, len(request_scope)))
            if request_scope
            else 0.0
        )
        attention_components = self._attention_score_components(
            atom,
            text=text,
            text_tokens=text_tokens,
            edge_degree=edge_degree,
            attention_policy=attention_policy,
        )
        relevance_signal = max(
            direct_score,
            float(attention_components.get("attention_focus", 0.0) or 0.0),
            edge_activation * 0.5,
        )
        goal_relevance = (
            relevance_signal if atom["type"] in {"goal", "commitment"} else 0.0
        )
        procedural_applicability = (
            relevance_signal if atom["type"] == "procedure" else 0.0
        )
        contradiction_penalty = (
            1.0 if atom["health_status"] == "contradicted" else 0.0
        )
        staleness_penalty = (
            1.0
            if atom["health_status"] == "stale" or atom["lifecycle_state"] == "archived"
            else 0.0
        )
        redundancy_penalty = 1.0 if atom["health_status"] == "merged" else 0.0
        superseded_penalty = 1.0 if atom["id"] in (superseded_refs or {}) else 0.0
        components = {
            "direct_cue_match": direct_score,
            "semantic_similarity": semantic_similarity,
            "edge_activation": edge_activation,
            "recency": recency,
            "confidence": confidence,
            "utility": utility,
            "salience": salience,
            "scope_specificity": scope_specificity,
            "goal_relevance": goal_relevance,
            "procedural_applicability": procedural_applicability,
            "contradiction_penalty": contradiction_penalty,
            "staleness_penalty": staleness_penalty,
            "redundancy_penalty": redundancy_penalty,
            "superseded_penalty": superseded_penalty,
        }
        if retrieval_mode == "agentic_recall":
            components.update(self._agentic_score_components(atom))
        components.update(attention_components)
        score = 0.0
        weights = dict(RETRIEVAL_WEIGHTS)
        if retrieval_mode == "agentic_recall":
            weights.update(
                {
                    "agency_match": 0.16,
                    "attribution_confidence": 0.12,
                    "correction_learning_relevance": 0.10,
                    "over_attribution_penalty": -0.25,
                    "omitted_counterevidence_penalty": -0.25,
                    "ignored_failure_penalty": -0.20,
                }
            )
        for name, component in components.items():
            score += weights.get(name, 0.0) * component
        score = max(0.0, min(1.0, score))
        return score, matched, {key: round(value, 4) for key, value in components.items()}


    def _packet_item(
        self,
        atom: Mapping[str, Any],
        score: float,
        *,
        requester: str,
        target_processor: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        evidence_refs, evidence_omissions = self._visible_evidence_refs(
            atom, requester=requester, target_processor=target_processor
        )
        item = {
            "item_ref": atom["id"],
            "item_kind": "atom",
            "atom_id": atom["id"],
            "atom_type": atom["type"],
            "atom_ref": atom["id"],
            "type": atom["type"],
            "payload": atom["payload"],
            "confidence": atom["confidence"],
            "rank": None,
            "activation_score": round(score, 4),
            "score": round(score, 4),
            "score_components": dict(atom.get("_score_components", {})),
            "association_trace": list(atom.get("_association_trace", [])),
            "salience": atom["salience"],
            "utility": atom["utility"],
            "rendered_content": self._render_atom(atom),
            "evidence_refs": evidence_refs,
            "access_decision": {
                "atom": "allowed",
                "evidence": "allowed" if evidence_refs == atom["evidence_refs"] else "denied",
            },
            "freshness": {
                "updated_at": atom["updated_at"],
                "health_status": atom["health_status"],
            },
            "scope": atom["scope"],
            "lifecycle_state": atom["lifecycle_state"],
            "health_status": atom["health_status"],
            "updated_at": atom["updated_at"],
            "version": atom["version"],
            "provenance": {
                "created_at": atom["created_at"],
                "observed_at": atom["observed_at"],
                "layer": atom["layer"],
                "retention_class": atom["retention_class"],
            },
        }
        return item, evidence_omissions


    def _visible_evidence_refs(
        self,
        atom: Mapping[str, Any],
        *,
        requester: str,
        target_processor: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        evidence_refs = list(atom["evidence_refs"])
        policy = atom["access_policy"]
        evidence_visibility = policy.get("evidence_visibility", policy.get("visibility", ["all"]))
        allowed = (
            "all" in evidence_visibility
            or requester in evidence_visibility
            or target_processor in evidence_visibility
            or f"processor:{target_processor}" in evidence_visibility
        )
        if allowed:
            return evidence_refs, []
        return [], [
            {
                "atom_ref": atom["id"],
                "reason": "evidence_access_denied",
                "omitted_refs": evidence_refs,
            }
        ]


    def _agentic_score_components(self, atom: Mapping[str, Any]) -> dict[str, float]:
        payload = atom["payload"]
        status = str(
            payload.get("status")
            or payload.get("outcome")
            or payload.get("result")
            or ""
        ).lower()
        has_correction = bool(payload.get("correction") or payload.get("lesson"))
        has_failure = status in {"failure", "failed", "error", "blocked", "denied"}
        return {
            "agency_match": 1.0 if payload_agent_id(payload) else 0.5,
            "attribution_confidence": confidence_score(atom["confidence"]),
            "correction_learning_relevance": 1.0 if has_correction else 0.0,
            "over_attribution_penalty": 0.0,
            "omitted_counterevidence_penalty": 0.0,
            "ignored_failure_penalty": 0.0 if has_failure else 0.1,
        }


    def _byte_budget(self, token_or_byte_budget: int | Mapping[str, int] | None) -> int:
        if token_or_byte_budget is None:
            return 100_000
        if isinstance(token_or_byte_budget, int):
            return max(1, token_or_byte_budget)
        if "bytes" in token_or_byte_budget:
            return max(1, int(token_or_byte_budget["bytes"]))
        if "tokens" in token_or_byte_budget:
            return max(1, int(token_or_byte_budget["tokens"]) * 4)
        return 100_000
