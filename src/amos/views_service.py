"""ViewService implementation for the AMOS service facade."""

from ._service_support import (
    Any,
    CONFLICT_RELATIONS,
    Mapping,
    SCHEMA_VERSION,
    Sequence,
    ValidationError,
    access_visible,
    canonical_json,
    payload_agent_id,
    payload_capability_name,
    scope_visible,
    stable_id,
    utc_now,
)


class ViewService:
    def __init__(
        self,
        store: Any,
        smp: Any,
        mutations: Any,
        retrieval: Any,
        graph: Any,
        capacity: Any,
    ):
        self.store = store
        self.smp = smp
        self.commit_atom = mutations.commit_atom
        self.retrieve_packet = retrieval.retrieve_packet
        self._rank_atom = retrieval._rank_atom
        self._packet_item = retrieval._packet_item
        self._render_atom = graph._render_atom
        self._capacity_pressure_mode = capacity._capacity_pressure_mode

    def record_runtime_state(
        self,
        *,
        agent_id: str,
        capabilities: Mapping[str, Any] | None = None,
        denied_capabilities: Sequence[str] | None = None,
        constraints: Sequence[str] | None = None,
        load: Mapping[str, Any] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "runtime_state",
                "payload": {
                    "agent_id": agent_id,
                    "capabilities": dict(capabilities or {}),
                    "denied_capabilities": list(denied_capabilities or []),
                    "constraints": list(constraints or []),
                    "load": dict(load or {}),
                },
                "scope": dict(scope or {}),
                "salience": 0.7,
                "utility": 0.8,
            },
            actor=actor,
        )


    def record_self_assessment(
        self,
        *,
        agent_id: str,
        claim: str,
        calibration: Mapping[str, Any],
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "self_assessment",
                "payload": {
                    "agent_id": agent_id,
                    "claim": claim,
                    "calibration": dict(calibration),
                },
                "scope": dict(scope or {}),
                "salience": 0.65,
                "utility": 0.75,
            },
            actor=actor,
        )


    def generate_self_narrative(
        self,
        *,
        agent_id: str,
        narrative: str,
        source_refs: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "self_narrative",
                "payload": {
                    "agent_id": agent_id,
                    "narrative": narrative,
                    "source_refs": list(source_refs or []),
                    "generated_from_graph_version": self.store.graph_version(),
                    "artifact": True,
                },
                "scope": dict(scope or {}),
                "salience": 0.55,
                "utility": 0.6,
                "confidence": {"level": "medium", "score": 0.5},
            },
            actor=actor,
        )


    def record_agentic_trace(
        self,
        *,
        agent_id: str,
        task: str,
        action: str,
        outcome: str,
        lesson: str | None = None,
        external_constraints: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "agentic_trace",
                "payload": {
                    "agent_id": agent_id,
                    "task": task,
                    "action": action,
                    "outcome": outcome,
                    "lesson": lesson,
                    "external_constraints": list(external_constraints or []),
                },
                "scope": dict(scope or {}),
                "salience": 0.8,
                "utility": 0.8,
            },
            actor=actor,
        )


    def record_action_outcome(
        self,
        *,
        agent_id: str,
        action_ref: str,
        status: str,
        evidence_refs: Sequence[str] | None = None,
        correction: str | None = None,
        limitation: str | None = None,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        return self.commit_atom(
            {
                "type": "action_outcome",
                "payload": {
                    "agent_id": agent_id,
                    "action_ref": action_ref,
                    "status": status,
                    "correction": correction,
                    "limitation": limitation,
                },
                "evidence_refs": list(evidence_refs or []),
                "scope": dict(scope or {}),
                "salience": 0.75,
                "utility": 0.8,
            },
            actor=actor,
        )


    def retrieve_self_awareness(
        self,
        *,
        agent_id: str,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "self-model",
    ) -> dict[str, Any]:
        by_type: dict[str, list[dict[str, Any]]] = {
            "capability": [],
            "commitment": [],
            "limitation": [],
            "runtime_state": [],
            "self_assessment": [],
            "self_model": [],
        }
        request_scope = dict(scope or {})
        omissions: list[dict[str, Any]] = []
        pressure_mode = self._capacity_pressure_mode()
        target_types = set(by_type)
        for atom in self.store.list_atoms():
            atom_ref = atom["id"]
            if atom.get("deleted"):
                omissions.append({"atom_ref": atom_ref, "reason": "deleted"})
                continue
            if atom["type"] not in target_types:
                continue
            if not scope_visible(atom["scope"], request_scope):
                omissions.append({"atom_ref": atom_ref, "reason": "scope_hidden"})
                continue
            if not access_visible(atom["access_policy"], requester, target_processor):
                omissions.append({"atom_ref": atom_ref, "reason": "access_hidden"})
                continue
            if atom["lifecycle_state"] == "archived":
                omissions.append({"atom_ref": atom_ref, "reason": "archived"})
                continue
            if atom["lifecycle_state"] not in {"active", "proposed"}:
                omissions.append(
                    {"atom_ref": atom_ref, "reason": f"lifecycle:{atom['lifecycle_state']}"}
                )
                continue
            payload = atom["payload"]
            if payload_agent_id(payload) not in {None, agent_id}:
                omissions.append({"atom_ref": atom_ref, "reason": "different_agent"})
                continue
            score, _matched, components = self._rank_atom(
                atom,
                [],
                request_scope=request_scope,
                retrieval_mode="self_awareness",
            )
            item, evidence_omissions = self._packet_item(
                {**atom, "_score_components": components},
                score,
                requester=requester,
                target_processor=target_processor,
            )
            omissions.extend(evidence_omissions)
            by_type[atom["type"]].append(item)

        for items in by_type.values():
            items.sort(
                key=lambda item: (
                    item.get("updated_at") or "",
                    item.get("score") or 0.0,
                ),
                reverse=True,
            )
        latest_runtime = None
        for item in by_type["runtime_state"]:
            if latest_runtime is None or item["updated_at"] > latest_runtime["updated_at"]:
                latest_runtime = item

        denied = set()
        capability_status: Mapping[str, Any] = {}
        if latest_runtime:
            runtime_payload = latest_runtime["payload"]
            denied = set(runtime_payload.get("denied_capabilities", []))
            capability_status = runtime_payload.get("capabilities", {})

        visible_capabilities = []
        for item in by_type["capability"]:
            name = payload_capability_name(item["payload"])
            if self._capability_unavailable(name, capability_status, denied):
                omissions.append(
                    {
                        "atom_ref": item["atom_ref"],
                        "reason": "capability_unavailable_in_runtime_state",
                    }
                )
                continue
            visible_capabilities.append(item)

        open_commitments = [
            item
            for item in by_type["commitment"]
            if str(item["payload"].get("status", "open")).lower()
            not in {"fulfilled", "cancelled", "canceled", "superseded"}
        ]
        response_items = [
            *by_type["self_model"],
            *visible_capabilities,
            *by_type["limitation"],
            *open_commitments,
            *by_type["self_assessment"],
        ]
        if latest_runtime:
            response_items.append(latest_runtime)
        for rank, item in enumerate(response_items, start=1):
            item["rank"] = rank
        selected = {item["atom_ref"] for item in response_items}
        conflicts = []
        if selected:
            for edge in self.store.list_edges():
                if edge["relation"] not in CONFLICT_RELATIONS:
                    continue
                if edge["source_ref"] in selected or edge["target_ref"] in selected:
                    conflicts.append(edge)
        graph_version = self.store.graph_version()
        request = {
            "scope": request_scope,
            "requester": requester,
            "target_processor": target_processor,
            "retrieval_mode": "self_awareness",
            "agent_id": agent_id,
            "structural": True,
            "budget_policy": "required_self_awareness_fields_not_budget_limited",
            "pressure_mode": pressure_mode,
        }
        packet_id = stable_id(
            "pkt",
            {"request": request, "graph_version": graph_version, "items": response_items},
        )
        used_bytes = len(canonical_json(response_items).encode("utf-8"))
        packet = {
            "packet_id": packet_id,
            "schema_version": SCHEMA_VERSION,
            "request": request,
            "graph_version": graph_version,
            "generated_at": utc_now(),
            "target_processor": target_processor,
            "retrieval_mode": "self_awareness",
            "scope": request_scope,
            "pressure_mode": pressure_mode,
            "items": response_items,
            "omissions": omissions,
            "conflicts": conflicts,
            "degradation": {
                "mode": "smp-deterministic-local",
                "pressure_mode": pressure_mode,
                "reduced_recall_depth": False,
                "omitted_evidence_detail": any(
                    omission["reason"] == "evidence_access_denied"
                    for omission in omissions
                ),
                "index_freshness": {
                    "semantic_index": "inline_rebuildable",
                    "graph_version": graph_version,
                },
                "reason_codes": sorted({omission["reason"] for omission in omissions}),
                "vector_index_available": False,
                "byte_budget": None,
                "used_bytes": used_bytes,
            },
            "provenance": {
                "store": getattr(self.store, "backend_name", "unknown"),
                "journal_head": self.store.last_event_hash(),
                "ranker_profile_id": "amos.v1.self_awareness_structural",
                "smp_processor_id": self.smp.processor_id,
            },
            "cache_policy": {"cacheable": True, "keyed_by_graph_version": True},
        }
        with self.store.transaction() as conn:
            self.store.cache_packet(
                conn,
                packet_id=packet_id,
                request=request,
                response=packet,
                graph_version=graph_version,
            )
        return {
            "view": "self_awareness",
            "agent_id": agent_id,
            "graph_version": graph_version,
            "generated_at": utc_now(),
            "self_model": by_type["self_model"],
            "capabilities": visible_capabilities,
            "limitations": by_type["limitation"],
            "open_commitments": open_commitments,
            "runtime_state": latest_runtime,
            "assessments": by_type["self_assessment"],
            "calibration": self.calibrate_self_model(
                agent_id=agent_id, scope=scope or {}, record=False
            )["calibration"],
            "omissions": omissions,
            "conflicts": conflicts,
            "source_packet_id": packet_id,
        }


    def calibrate_self_model(
        self,
        *,
        agent_id: str,
        scope: Mapping[str, Any] | None = None,
        actor: str = "system",
        record: bool = False,
    ) -> dict[str, Any]:
        scope = dict(scope or {})
        atoms = [
            atom
            for atom in self.store.list_atoms()
            if not atom.get("deleted")
            and scope_visible(atom["scope"], scope)
            and payload_agent_id(atom["payload"]) in {None, agent_id}
        ]
        capabilities = [atom for atom in atoms if atom["type"] == "capability"]
        outcomes = [atom for atom in atoms if atom["type"] == "action_outcome"]
        unverified = []
        for capability in capabilities:
            name = payload_capability_name(capability["payload"])
            has_evidence = bool(capability["evidence_refs"])
            has_success = any(
                name
                and name in canonical_json(outcome["payload"])
                and str(outcome["payload"].get("status", "")).lower()
                in {"success", "succeeded"}
                for outcome in outcomes
            )
            if not has_evidence and not has_success:
                unverified.append(name or capability["id"])
        rate = len(unverified) / len(capabilities) if capabilities else 0.0
        calibration = {
            "capability_claim_count": len(capabilities),
            "unverified_capability_count": len(unverified),
            "unverified_capabilities": unverified,
            "overconfident_claim_rate": round(rate, 4),
        }
        result = {"status": "calibrated", "agent_id": agent_id, "calibration": calibration}
        if record and capabilities:
            result["assessment"] = self.record_self_assessment(
                agent_id=agent_id,
                claim="capability self-report calibration",
                calibration=calibration,
                scope=scope,
                actor=actor,
            )
        return result


    def retrieve_agentic_recall(
        self,
        *,
        agent_id: str,
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        target_processor: str = "planner",
    ) -> dict[str, Any]:
        packet = self.retrieve_packet(
            cues=cues or [],
            scope=scope or {},
            requester=requester,
            target_processor=target_processor,
            retrieval_mode="agentic_recall",
            max_items=100,
            include_conflicts=True,
            include_low_health=True,
            type_filter=[
                "action_outcome",
                "agentic_trace",
                "limitation",
                "self_assessment",
                "self_narrative",
            ],
            run_policy=False,
        )
        recalls = []
        self_actions = []
        other_agent_actions = []
        shared_system_actions = []
        external_actions = []
        unknown_responsibility_actions = []
        omissions = list(packet["omissions"])
        active_narratives = []
        expired_narratives = []
        for item in packet["items"]:
            item_agent = payload_agent_id(item["payload"])
            responsibility = item["payload"].get("responsibility")
            item_kind = item["type"]
            if item_kind in {"action_outcome", "agentic_trace"}:
                responsibility_class = self._agentic_responsibility(
                    item, agent_id=agent_id
                )
                attributed = dict(item)
                attributed["responsibility"] = responsibility_class
                if responsibility_class == "other_agent":
                    other_agent_actions.append(attributed)
                    continue
                if responsibility_class == "shared_system":
                    shared_system_actions.append(attributed)
                    continue
                if responsibility_class == "external":
                    external_actions.append(attributed)
                    continue
                if responsibility_class == "unknown":
                    unknown_responsibility_actions.append(attributed)
                    continue
                self_actions.append(attributed)
            elif item_agent not in {None, agent_id}:
                attributed = dict(item)
                attributed["responsibility"] = (
                    "shared_system"
                    if responsibility == "shared_system"
                    else "other_agent"
                )
                if attributed["responsibility"] == "shared_system":
                    shared_system_actions.append(attributed)
                else:
                    other_agent_actions.append(attributed)
                continue
            if item["type"] == "self_narrative":
                if self._self_narrative_has_counterevidence(item, packet["items"]):
                    expired_narratives.append(item)
                    omissions.append(
                        {
                            "atom_ref": item["atom_ref"],
                            "reason": "self_narrative_drift",
                        }
                    )
                    continue
                active_narratives.append(item)
                continue
            recalls.append(item)
        material_counterevidence = [
            item
            for item in recalls
            if item["type"] in {"action_outcome", "agentic_trace", "limitation"}
            and (
                str(
                    item["payload"].get("status")
                    or item["payload"].get("outcome")
                    or item["payload"].get("result")
                    or ""
                ).lower()
                in {"blocked", "denied", "error", "failed", "failure"}
                or item["payload"].get("correction")
                or item["payload"].get("limitation")
                or item["type"] == "limitation"
            )
        ]
        external_constraints = [
            constraint
            for item in recalls + external_actions + shared_system_actions
            for constraint in item["payload"].get("external_constraints", [])
        ]
        return {
            "view": "agentic_recall",
            "agent_id": agent_id,
            "graph_version": packet["graph_version"],
            "generated_at": utc_now(),
            "successes": self._status_items(recalls, {"success", "succeeded"}),
            "failures": self._status_items(recalls, {"failure", "failed", "error"}),
            "blocked": self._status_items(recalls, {"blocked", "denied"}),
            "corrections": [
                item for item in recalls if item["payload"].get("correction")
            ],
            "traces": [item for item in recalls if item["type"] == "agentic_trace"],
            "self_actions": self_actions,
            "other_agent_actions": other_agent_actions,
            "shared_system_actions": shared_system_actions,
            "external_actions": external_actions,
            "unknown_responsibility_actions": unknown_responsibility_actions,
            "external_constraints": external_constraints,
            "material_counterevidence": material_counterevidence,
            "self_narratives": active_narratives,
            "expired_self_narratives": expired_narratives,
            "omissions": omissions,
            "conflicts": packet["conflicts"],
            "source_packet_id": packet["packet_id"],
        }


    def retrieve_shared_view(
        self,
        *,
        processor_ids: Sequence[str],
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        max_items: int = 20,
    ) -> dict[str, Any]:
        packets = {}
        union: dict[str, list[dict[str, Any]]] = {}
        overlays: dict[str, list[str]] = {}
        omissions_by_identity: dict[str, list[dict[str, Any]]] = {}
        graph_versions = []
        for processor_id in processor_ids:
            packet = self.retrieve_packet(
                cues=cues or [],
                scope=scope or {},
                requester=requester,
                target_processor=processor_id,
                retrieval_mode="shared_coordination",
                max_items=max_items,
                include_conflicts=True,
            )
            packets[processor_id] = packet["packet_id"]
            graph_versions.append(packet["graph_version"])
            overlays[processor_id] = []
            omissions_by_identity[processor_id] = list(packet["omissions"])
            for item in packet["items"]:
                union.setdefault(item["atom_ref"], []).append(item)
                overlays[processor_id].append(item["atom_ref"])
        common_items = [
            self._shared_common_item(atom_ref, items, processor_count=len(processor_ids))
            for atom_ref, items in union.items()
        ]
        return {
            "view": "shared_memory",
            "processor_ids": list(processor_ids),
            "common_graph_version": min(graph_versions) if graph_versions else self.store.graph_version(),
            "generated_at": utc_now(),
            "items": common_items,
            "per_processor_overlays": overlays,
            "omissions_by_identity": omissions_by_identity,
            "source_packets": packets,
        }


    def refresh_shared_view(
        self,
        *,
        processor_ids: Sequence[str],
        cues: Sequence[str] | None = None,
        scope: Mapping[str, Any] | None = None,
        requester: str = "system",
        max_items: int = 20,
    ) -> dict[str, Any]:
        view = self.retrieve_shared_view(
            processor_ids=processor_ids,
            cues=cues,
            scope=scope,
            requester=requester,
            max_items=max_items,
        )
        view["refresh_status"] = "refreshed"
        return view


    def evaluate_procedure_execution(
        self,
        *,
        procedure_ref: str,
        autonomous: bool = False,
        approved_by: str | None = None,
        tool_permission_binding: Mapping[str, Any] | None = None,
        preconditions_satisfied: bool = False,
        rollback_plan: Mapping[str, Any] | None = None,
        review_status: str | None = None,
    ) -> dict[str, Any]:
        procedure = self.store.get_atom(procedure_ref)
        if procedure is None or procedure.get("deleted"):
            raise ValidationError(f"unknown procedure atom: {procedure_ref}")
        if procedure["type"] != "procedure":
            raise ValidationError(f"atom is not a procedure: {procedure_ref}")
        if autonomous:
            return {
                "status": "denied",
                "procedure_ref": procedure_ref,
                "reason": "autonomous_external_state_execution_not_allowed_in_v1",
                "advisory_rendering": self._render_atom(procedure),
            }
        missing = []
        if not approved_by:
            missing.append("approved_by")
        if not tool_permission_binding:
            missing.append("tool_permission_binding")
        if not preconditions_satisfied:
            missing.append("preconditions_satisfied")
        if not rollback_plan:
            missing.append("rollback_plan")
        if review_status != "approved":
            missing.append("review_status:approved")
        if missing:
            return {
                "status": "review_required",
                "procedure_ref": procedure_ref,
                "missing": missing,
                "advisory_rendering": self._render_atom(procedure),
            }
        return {
            "status": "eligible_for_external_executor",
            "procedure_ref": procedure_ref,
            "approved_by": approved_by,
            "tool_permission_binding": dict(tool_permission_binding),
            "preconditions_satisfied": True,
            "rollback_plan": dict(rollback_plan),
            "review_status": review_status,
            "note": "AMOS evaluated policy only; execution remains outside AMOS.",
        }


    def _capability_unavailable(
        self, name: str, capability_status: Mapping[str, Any], denied: set[str]
    ) -> bool:
        if name in denied:
            return True
        status = capability_status.get(name)
        if status is None:
            return False
        if status is False:
            return True
        if isinstance(status, str):
            return status.lower() in {"denied", "disabled", "false", "unavailable"}
        if isinstance(status, Mapping):
            if status.get("available") is False:
                return True
            if str(status.get("permission", "")).lower() == "denied":
                return True
        return False


    def _status_items(
        self, items: Sequence[Mapping[str, Any]], statuses: set[str]
    ) -> list[dict[str, Any]]:
        matched = []
        for item in items:
            payload = item["payload"]
            status = str(
                payload.get("status")
                or payload.get("outcome")
                or payload.get("result")
                or ""
            ).lower()
            if status in statuses:
                matched.append(dict(item))
        return matched


    def _agentic_responsibility(
        self, item: Mapping[str, Any], *, agent_id: str
    ) -> str:
        payload = item["payload"]
        responsibility = str(payload.get("responsibility", "")).lower()
        if responsibility == "shared_system":
            return "shared_system"
        if responsibility == "external":
            return "external"
        item_agent = payload_agent_id(payload)
        if item_agent == agent_id:
            return "self"
        if item_agent:
            return "other_agent"
        if payload.get("external_constraints") or payload.get("external_actor"):
            return "external"
        return "unknown"


    def _shared_common_item(
        self,
        atom_ref: str,
        processor_items: Sequence[Mapping[str, Any]],
        *,
        processor_count: int,
    ) -> dict[str, Any]:
        common = dict(processor_items[0])
        evidence_sets = [set(item.get("evidence_refs", [])) for item in processor_items]
        if evidence_sets:
            visible_to_all = set.intersection(*evidence_sets)
        else:
            visible_to_all = set()
        common["evidence_refs"] = sorted(visible_to_all)
        common["shared_visibility"] = {
            "visible_processor_count": len(processor_items),
            "requested_processor_count": processor_count,
            "evidence_policy": "least_common_denominator",
            "omitted_evidence_for_some_identities": any(
                set(item.get("evidence_refs", [])) != visible_to_all
                for item in processor_items
            ),
        }
        if len(processor_items) < processor_count:
            common["shared_visibility"]["omitted_for_some_identities"] = True
        return common


    def _self_narrative_has_counterevidence(
        self,
        narrative_item: Mapping[str, Any],
        items: Sequence[Mapping[str, Any]],
    ) -> bool:
        agent_id = payload_agent_id(narrative_item["payload"])
        generated_at = narrative_item.get("updated_at", "")
        for item in items:
            if item["type"] == "self_narrative":
                continue
            payload = item["payload"]
            if payload_agent_id(payload) not in {None, agent_id}:
                continue
            if item.get("updated_at", "") <= generated_at:
                continue
            status = str(
                payload.get("status")
                or payload.get("outcome")
                or payload.get("result")
                or ""
            ).lower()
            if status in {"blocked", "denied", "error", "failed", "failure"}:
                return True
            if payload.get("correction") or payload.get("limitation"):
                return True
            calibration = payload.get("calibration")
            if isinstance(calibration, Mapping):
                if calibration.get("overconfident") is True:
                    return True
                if float(calibration.get("overconfident_claim_rate", 0.0) or 0.0) > 0:
                    return True
        return False
