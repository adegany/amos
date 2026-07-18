"""GraphService implementation for the AMOS service facade."""

from ._service_support import (
    Any,
    LOW_RISK_EXPLICIT_RELATIONS,
    Mapping,
    SCHEMA_VERSION,
    Sequence,
    _structured_ref_list,
    canonical_json,
    digest,
    normalize_atom,
    normalize_relation,
    stable_id,
    utc_now,
)


class GraphService:
    def __init__(self, store: Any, indexes: Any):
        self.store = store
        self._attach_search_index = indexes._attach_search_index

    def _active_superseded_refs(
        self, refs: Sequence[str] | None = None
    ) -> dict[str, list[str]]:
        active_refs = self.store.active_atom_ids(lifecycle_states=["active"])
        scoped_refs = {str(ref) for ref in refs or [] if str(ref)}
        edges = (
            self.store.list_edges_for_refs(sorted(scoped_refs))
            if refs is not None
            else self.store.list_edges()
        )
        superseded: dict[str, list[str]] = {}
        for edge in edges:
            if edge.get("lifecycle_state") != "active":
                continue
            if edge.get("relation") != "rel:supersedes":
                continue
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source in active_refs and target in active_refs:
                superseded.setdefault(target, []).append(source)
        return {ref: sorted(set(sources)) for ref, sources in superseded.items()}


    def _hot_graph_edge_degree_counts(
        self, atoms: Sequence[Mapping[str, Any]]
    ) -> dict[str, int]:
        atoms_by_ref = {
            str(atom.get("id") or ""): atom
            for atom in atoms
            if str(atom.get("id") or "")
        }
        refs = sorted(atoms_by_ref)
        if not refs:
            return {}
        counts: dict[str, int] = {}
        for edge in self.store.list_edges_for_refs(refs):
            if not self._hot_graph_edge_visible(edge, atoms_by_ref):
                continue
            source = str(edge.get("source_ref") or "")
            target = str(edge.get("target_ref") or "")
            if source in atoms_by_ref:
                counts[source] = counts.get(source, 0) + 1
            if target in atoms_by_ref:
                counts[target] = counts.get(target, 0) + 1
        return counts


    def _hot_graph_edge_visible(
        self,
        edge: Mapping[str, Any],
        atoms_by_ref: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        if edge.get("lifecycle_state") != "active":
            return False
        relation = str(edge.get("relation") or "")
        source = atoms_by_ref.get(str(edge.get("source_ref") or ""))
        target = atoms_by_ref.get(str(edge.get("target_ref") or ""))
        if not source or not target:
            return False
        if (
            source.get("lifecycle_state") == "proposed"
            or target.get("lifecycle_state") == "proposed"
        ):
            return False
        if relation in {"rel:derived_from", "rel:supersedes"}:
            return True
        return not (
            source.get("lifecycle_state") == "archived"
            or target.get("lifecycle_state") == "archived"
        )


    def _edge_relation_activation_weight(self, relation: str) -> float:
        if relation in {
            "rel:uses",
            "rel:supports",
            "rel:produced_outcome",
            "rel:made_commitment",
            "rel:has_capability",
            "rel:has_limitation",
        }:
            return 0.9
        if relation in {"rel:derived_from", "rel:supersedes"}:
            return 0.75
        if relation in {"rel:contradicts", "rel:similar_to"}:
            return 0.65
        return 0.5


    def _render_atom(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        payload = atom["payload"]
        if isinstance(payload, Mapping):
            content = (
                payload.get("claim")
                or payload.get("name")
                or payload.get("description")
                or payload.get("summary")
                or canonical_json(payload)
            )
        else:
            content = str(payload)
        return {
            "format": "compact_json",
            "text": str(content),
            "payload": payload,
        }


    def _archive_atom_projection(
        self,
        conn: Any,
        atom: Mapping[str, Any],
        *,
        reason: str,
        superseded_by: str,
        actor: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        archived = dict(atom)
        archived["lifecycle_state"] = "archived"
        archived["health_status"] = "merged"
        archived["version"] = int(archived["version"]) + 1
        archived["updated_at"] = utc_now()
        archived["supersedes"] = list(archived.get("supersedes") or []) + [
            superseded_by
        ]
        archived["decay_policy"] = {
            **dict(archived.get("decay_policy") or {}),
            "archive_reason": reason,
            "superseded_by": superseded_by,
        }
        archived["revision_history"] = list(archived.get("revision_history") or [])
        archived["revision_history"].append(
            {
                "version": atom["version"],
                "digest": digest(self._atom_projection(atom)),
                "changed_at": utc_now(),
                "actor": actor,
                "reason": reason,
            }
        )
        archived = normalize_atom(
            self._attach_search_index(archived), require_id=True
        )
        self.store.replace_atom(conn, archived)
        deleted_edges = self.store.mark_edges_deleted_for_ref(conn, archived["id"])
        return archived, deleted_edges


    def _structured_duplicate_key(
        self, atom: Mapping[str, Any]
    ) -> tuple[Any, ...] | None:
        if atom.get("deleted") or atom.get("lifecycle_state") != "active":
            return None
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        scope = atom.get("scope")
        scope = scope if isinstance(scope, Mapping) else {}
        tenant = scope.get("tenant")
        component = scope.get("component")
        asset = scope.get("asset") or payload.get("asset")
        run_id = scope.get("run_id") or payload.get("run_id")
        agent_id = payload.get("agent_id")
        if atom.get("type") == "agentic_trace":
            kind = payload.get("qandl_kind") or payload.get("kind")
            chunk = payload.get("chunk")
            if kind == "reflection" and chunk not in (None, ""):
                return (
                    "agentic_trace.reflection",
                    tenant,
                    component,
                    asset,
                    run_id,
                    agent_id,
                    chunk,
                )
        if atom.get("type") == "runtime_state":
            role_key = payload.get("role_key") or payload.get("role")
            if agent_id:
                return (
                    "runtime_state.current",
                    tenant,
                    component,
                    asset,
                    run_id,
                    agent_id,
                    role_key,
                )
        return None


    def _structured_duplicate_quality(self, atom: Mapping[str, Any]) -> int:
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        score = 0
        for key in (
            "directive_atom_ref",
            "source_directive_ref",
            "control_signature",
            "metric_deltas",
            "tool_surface",
            "runtime_capabilities",
            "runtime_constraints",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                score += 1
        score += min(5, len(canonical_json(payload)) // 1000)
        return score


    def _intrinsic_edges_for_atom(self, atom: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Project deterministic graph edges encoded by structured atom fields."""

        if atom.get("deleted") or atom.get("lifecycle_state") != "active":
            return []

        atom_id = str(atom["id"])
        scope = dict(atom.get("scope") or {})
        payload = atom.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def active_atom(ref: Any) -> dict[str, Any] | None:
            ref_id = str(ref or "")
            if not ref_id or ref_id == atom_id:
                return None
            existing = self.store.get_atom(ref_id)
            if existing is None or existing.get("deleted"):
                return None
            if existing.get("lifecycle_state") != "active":
                return None
            return existing

        def add(
            source_ref: Any,
            target_ref: Any,
            relation: str,
            *,
            evidence_refs: Sequence[str] | None = None,
            confidence: float | Mapping[str, Any] | None = None,
            derivation_kind: str = "intrinsic_structural",
        ) -> None:
            source = str(source_ref or "")
            target = str(target_ref or "")
            if not source or not target or source == target:
                return
            if source != atom_id and active_atom(source) is None:
                return
            if target != atom_id and active_atom(target) is None:
                return
            key = (source, target, relation)
            if key in seen:
                return
            seen.add(key)
            relation_evidence = list(
                dict.fromkeys(
                    str(ref)
                    for ref in (
                        evidence_refs
                        if evidence_refs is not None
                        else atom.get("evidence_refs", [])
                    )
                    if str(ref)
                )
            )
            edges.append(
                self._edge(
                    source,
                    target,
                    relation,
                    scope,
                    evidence_refs=relation_evidence,
                    confidence=(
                        confidence if confidence is not None else atom.get("confidence")
                    ),
                    derivation={
                        "kind": derivation_kind,
                        "processor_id": "amos.graph.intrinsic.v1",
                        "source_refs": [atom_id],
                    },
                )
            )

        for ref in _structured_ref_list(atom.get("supersedes")):
            add(atom_id, ref, "rel:supersedes")

        for ref in _structured_ref_list(payload.get("source_refs")):
            add(atom_id, ref, "rel:derived_from")

        for ref in _structured_ref_list(payload.get("memory_references")):
            add(atom_id, ref, "rel:uses")

        directive_ref = payload.get("directive_atom_ref") or payload.get(
            "source_directive_ref"
        )
        if atom.get("type") == "agentic_trace" and directive_ref:
            add(directive_ref, atom_id, "rel:produced_outcome")

        if atom.get("type") in {
            "capability",
            "limitation",
            "commitment",
            "runtime_state",
            "self_assessment",
        }:
            relation_by_type = {
                "capability": "rel:has_capability",
                "limitation": "rel:has_limitation",
                "commitment": "rel:made_commitment",
                "runtime_state": "rel:attributed_to",
                "self_assessment": "rel:attributed_to",
            }
            relation = relation_by_type[str(atom["type"])]
            for ref in _structured_ref_list(atom.get("evidence_refs")):
                source = active_atom(ref)
                if source and source.get("type") == "self_model":
                    add(source["id"], atom_id, relation)

        for raw in payload.get("graph_relations", []):
            if not isinstance(raw, Mapping):
                continue
            relation = str(raw.get("relation") or "")
            if relation not in LOW_RISK_EXPLICIT_RELATIONS:
                continue
            source_ref = raw.get("source_ref", "$self")
            target_ref = raw.get("target_ref")
            source_ref = atom_id if source_ref == "$self" else source_ref
            target_ref = atom_id if target_ref == "$self" else target_ref
            if atom_id not in {str(source_ref or ""), str(target_ref or "")}:
                continue
            add(
                source_ref,
                target_ref,
                relation,
                evidence_refs=raw.get("evidence_refs"),
                confidence=raw.get("confidence"),
                derivation_kind="explicit_structural",
            )

        return edges


    def _edge(
        self,
        source_ref: str,
        target_ref: str,
        relation: str,
        scope: Mapping[str, Any],
        *,
        evidence_refs: Sequence[str] | None = None,
        confidence: float | Mapping[str, Any] | None = None,
        derivation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        relation = normalize_relation(relation)
        now = utc_now()
        if isinstance(confidence, Mapping):
            score = float(confidence.get("score", 0.75) or 0.75)
        elif isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            score = float(confidence)
        else:
            score = 0.75
        score = max(0.0, min(1.0, score))
        level = (
            "high" if score >= 0.85 else "medium-high" if score >= 0.65 else "medium"
        )
        return {
            "edge_id": stable_id(
                "edge",
                {
                    "source_ref": source_ref,
                    "target_ref": target_ref,
                    "relation": relation,
                    "scope": dict(scope),
                },
            ),
            "source_ref": source_ref,
            "target_ref": target_ref,
            "relation": relation,
            "schema_version": SCHEMA_VERSION,
            "evidence_refs": list(
                dict.fromkeys(str(ref) for ref in evidence_refs or [] if str(ref))
            ),
            "scope": dict(scope),
            "confidence": {"level": level, "score": score},
            "derivation": dict(
                derivation
                or {
                    "kind": "direct_service_edge",
                    "exact_producer_unknown": True,
                }
            ),
            "lifecycle_state": "active",
            "health_status": "healthy",
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "deleted": 0,
        }


    def _contradiction_signature(
        self, atom: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], str] | None:
        payload = atom["payload"]
        if {"subject", "predicate", "value"}.issubset(payload):
            key = (
                atom["type"],
                canonical_json(atom["scope"]),
                payload["subject"],
                payload["predicate"],
            )
            return key, canonical_json(payload["value"])
        if {"key", "value"}.issubset(payload):
            key = (atom["type"], canonical_json(atom["scope"]), payload["key"])
            return key, canonical_json(payload["value"])
        return None


    def _atom_projection(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in atom.items()
            if key not in {"deleted", "revision_history", "last_accessed"}
        }


    def _memory_identity_digest(self, atom: Mapping[str, Any]) -> str:
        return digest(
            {
                "type": atom["type"],
                "payload": atom["payload"],
                "scope": atom["scope"],
                "evidence_refs": atom["evidence_refs"],
            }
        )


    def _counts(self, rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key))
            counts[value] = counts.get(value, 0) + 1
        return counts
