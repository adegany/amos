"""IndexService implementation for the AMOS service facade."""

from ._service_support import (
    Any,
    DEFAULT_MEMORY_POLICY,
    Mapping,
    Sequence,
    SEARCH_INDEX_REF,
    SEARCH_INDEX_SCHEMA,
    _top_symmetric_components,
    defaultdict,
    math,
    normalize_atom,
    re,
    stable_id,
    utc_now,
)


_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "will", "with",
    "true", "false", "none", "null", "active", "proposed", "pending",
}
_REFERENCE_TOKEN = re.compile(
    r"^(?:atom|evt|evd|thread|cycle|repisode|curriculum|endogenous_work|"
    r"kproject|agent_project)_[a-z0-9_-]{8,}$"
)

_SKILL_BINDING_PROFILE = "cogito.skill-definition-binding.v2"
_SKILL_FAMILY_PROFILE = "cogito.skill-family-definition.v1"
_SKILL_SEMANTIC_PROFILE = "cogito.skill-semantic-node.v1"
_SKILL_SELECTION_POSITIVE_FIELDS = (
    "semantic_capabilities",
    "input_shapes",
    "work_products",
    "retrieval_examples",
)
_SKILL_SELECTION_NEGATIVE_FIELDS = ("exclusions",)


def _content_token(token: str) -> bool:
    """Keep lexical retrieval focused on semantic content, not wire metadata."""

    token = str(token or "").strip().lower()
    if len(token) <= 1 or token in _RETRIEVAL_STOPWORDS:
        return False
    if _REFERENCE_TOKEN.fullmatch(token):
        return False
    if len(token) >= 12 and all(character in "0123456789abcdef" for character in token):
        return False
    if re.fullmatch(r"v\d+", token):
        return False
    return True


class IndexService:
    def __init__(self, store: Any, smp: Any):
        self.store = store
        self.smp = smp
        self._smp_vector_model_graph_version: int | None = None
        self._policy_provider: Any | None = None

    def set_policy_provider(self, provider: Any) -> None:
        self._policy_provider = provider

    def memory_policy(self) -> dict[str, Any]:
        if self._policy_provider is None:
            return DEFAULT_MEMORY_POLICY
        return self._policy_provider()

    def rebuild(self, *, graph_version: int | None = None) -> dict[str, Any]:
        return self._rebuild_derived_indexes(
            graph_version=(
                self.store.graph_version()
                if graph_version is None
                else int(graph_version)
            )
        )

    def _search_text_for_atom(self, atom: Mapping[str, Any]) -> str:
        payload = atom.get("payload", {})
        if not isinstance(payload, Mapping):
            return self._search_text_for_value(payload).lower()
        profile = str(payload.get("profile") or "")
        if profile == _SKILL_BINDING_PROFILE:
            selection = payload.get("selection_contract")
            selection = selection if isinstance(selection, Mapping) else {}
            return self._search_text_for_value(
                {
                    "skill_id": payload.get("skill_id"),
                    "skill_name": payload.get("skill_name"),
                    "description": payload.get("description"),
                    "family_refs": payload.get("family_refs"),
                    "selection_contract": {
                        field: selection.get(field)
                        for field in _SKILL_SELECTION_POSITIVE_FIELDS
                        if selection.get(field)
                    },
                }
            ).lower()
        if profile == _SKILL_FAMILY_PROFILE:
            return self._search_text_for_value(
                {
                    "family_id": payload.get("family_id"),
                    "name": payload.get("name"),
                    "description": payload.get("description"),
                }
            ).lower()
        if profile == _SKILL_SEMANTIC_PROFILE:
            return self._search_text_for_value(
                {
                    "semantic_context_key": payload.get("semantic_context_key"),
                    "name": payload.get("name"),
                    "description": payload.get("description"),
                    "retrieval_examples": payload.get("retrieval_examples"),
                }
            ).lower()
        return self._search_text_for_value(payload).lower()


    def _negative_search_text_for_atom(self, atom: Mapping[str, Any]) -> str:
        payload = atom.get("payload", {})
        if not isinstance(payload, Mapping):
            return ""
        if str(payload.get("profile") or "") != _SKILL_BINDING_PROFILE:
            return ""
        selection = payload.get("selection_contract")
        selection = selection if isinstance(selection, Mapping) else {}
        return self._search_text_for_value(
            {
                field: selection.get(field)
                for field in _SKILL_SELECTION_NEGATIVE_FIELDS
                if selection.get(field)
            }
        ).lower()


    def _search_text_for_value(self, value: Any) -> str:
        if value in (None, "", [], {}):
            return ""
        if isinstance(value, Mapping):
            return " ".join(
                part
                for item in value.values()
                if (part := self._search_text_for_value(item))
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(
                part
                for item in value
                if (part := self._search_text_for_value(item))
            )
        return str(value)


    def _search_index_for_atom(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        self._sync_smp_vector_model()
        text = self._search_text_for_atom(atom)
        negative_text = self._negative_search_text_for_atom(atom)

        def semantic_tokens(value: str) -> list[str]:
            raw_tokens = {
                token for token in re.findall(r"[a-z0-9_]+", value) if token
            }
            expanded = set(raw_tokens)
            for token in raw_tokens:
                expanded.update(part for part in token.split("_") if part)
            return sorted(token for token in expanded if _content_token(token))

        tokens = semantic_tokens(text)
        negative_tokens = semantic_tokens(negative_text)
        payload = atom.get("payload", {})
        selection = (
            payload.get("selection_contract")
            if isinstance(payload, Mapping)
            else None
        )
        selection = selection if isinstance(selection, Mapping) else {}
        return {
            "text": text,
            "tokens": tokens,
            "negative_text": negative_text,
            "negative_tokens": negative_tokens,
            "negative_phrases": [
                str(item).strip().lower()
                for field in _SKILL_SELECTION_NEGATIVE_FIELDS
                for item in selection.get(field, []) or []
                if str(item).strip()
            ],
            "vector": self.smp.encode(text),
            "processor_id": self.smp.processor_id,
            "processor_version": self.smp.processor_version,
            "search_schema": SEARCH_INDEX_SCHEMA,
            "vector_model": self.smp.vector_model_info(),
        }


    def _attach_search_index(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        indexed = dict(atom)
        if indexed.get("deleted") or indexed.get("lifecycle_state") in {
            "archived",
            "superseded",
            "deleted",
        }:
            # Derived search vectors are hot data, not canonical history.
            # Lifecycle retirement removes their payload immediately.
            indexed["index_refs"] = {}
            return indexed
        index_refs = dict(indexed.get("index_refs") or {})
        index_refs[SEARCH_INDEX_REF] = self._search_index_for_atom(indexed)
        indexed["index_refs"] = index_refs
        return indexed


    def _atom_search_index(
        self, atom: Mapping[str, Any], *, allow_stale: bool = False
    ) -> dict[str, Any]:
        index_refs = atom.get("index_refs", {})
        if isinstance(index_refs, Mapping):
            stored = index_refs.get(SEARCH_INDEX_REF)
            if isinstance(stored, Mapping):
                text = stored.get("text")
                tokens = stored.get("tokens")
                vector = stored.get("vector")
                search_schema = stored.get("search_schema")
                if (
                    search_schema == SEARCH_INDEX_SCHEMA
                    and isinstance(text, str)
                    and isinstance(tokens, list)
                    and isinstance(vector, list)
                ):
                    index = {
                        "text": text,
                        "tokens": [str(token) for token in tokens],
                        "vector": [float(value) for value in vector],
                        "negative_text": str(stored.get("negative_text") or ""),
                        "negative_tokens": [
                            str(token)
                            for token in stored.get("negative_tokens", []) or []
                        ],
                        "negative_phrases": [
                            str(phrase)
                            for phrase in stored.get("negative_phrases", []) or []
                        ],
                        "vector_model": dict(stored.get("vector_model") or {}),
                    }
                    stale = not self.smp._stored_vector_matches(stored)
                    if stale and not allow_stale:
                        return self._search_index_for_atom(atom)
                    if stale:
                        index["vector_stale"] = True
                    return index
        return self._search_index_for_atom(atom)


    def _prepare_committed_atom(self, atom: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_atom(atom)
        now = utc_now()
        normalized["id"] = normalized["id"] or stable_id(
            "atom",
            {
                "type": normalized["type"],
                "payload": normalized["payload"],
                "scope": normalized["scope"],
                "evidence_refs": normalized["evidence_refs"],
            },
        )
        normalized["created_at"] = normalized["created_at"] or now
        normalized["observed_at"] = normalized["observed_at"] or now
        normalized["updated_at"] = normalized["updated_at"] or now
        normalized["version"] = 1
        return normalize_atom(self._attach_search_index(normalized), require_id=True)


    def _sync_smp_vector_model(
        self, *, graph_version: int | None = None, force: bool = False
    ) -> dict[str, Any]:
        graph_version = (
            self.store.graph_version() if graph_version is None else int(graph_version)
        )
        if not force and self._smp_vector_model_graph_version == graph_version:
            return self.smp.vector_model_info()
        document_count = self.store.atom_text_document_count()
        document_frequencies = self.store.token_document_frequencies()
        latent_vectors = self.store.list_token_latent_vectors(graph_version=graph_version)
        latent_dimensions = max(
            (len(vector) for vector in latent_vectors.values()),
            default=0,
        )
        self.smp.configure_vector_model(
            document_frequencies=document_frequencies,
            document_count=document_count,
            graph_version=graph_version,
            latent_vectors=latent_vectors,
            latent_dimensions=latent_dimensions,
        )
        self._smp_vector_model_graph_version = graph_version
        return self.smp.vector_model_info()


    def _rebuild_derived_indexes(
        self,
        *,
        graph_version: int | None = None,
        publish_revision_offset: int = 0,
    ) -> dict[str, Any]:
        requested_graph_version = (
            graph_version if graph_version is not None else self.store.graph_version()
        )
        publish_revision_offset = max(0, int(publish_revision_offset))
        policy = self.memory_policy()
        maintenance = policy.get("maintenance", {})
        write_batch_size = max(
            1, int(maintenance.get("index_write_batch_size", 64) or 64)
        )

        # Refresh the rebuildable lexical projection in short transactions.
        # Each batch re-reads its atoms under the writer, so a foreground
        # mutation that lands after planning is never overwritten by stale
        # maintenance input.
        with self.store.read_snapshot():
            planned_atom_ids = [
                str(atom["id"])
                for atom in self.store.list_atoms_filtered(include_deleted=True)
            ]
        rebuilt_atoms = 0
        write_batch_count = 0
        for offset in range(0, len(planned_atom_ids), write_batch_size):
            atom_ids = planned_atom_ids[offset : offset + write_batch_size]
            with self.store.transaction() as conn:
                for atom_id in atom_ids:
                    atom = self.store.get_atom(atom_id)
                    if atom is None:
                        self.store.delete_atom_text_index(conn, atom_id)
                        continue
                    self.store.replace_atom_text_index(conn, atom)
                    rebuilt_atoms += 1
            write_batch_count += 1

        cleanup = policy.get("storage_cleanup", {})
        pruned_index = {"status": "skipped", "reason": "storage_cleanup_disabled"}
        if cleanup.get("enabled", True):
            lifecycle_states: list[str] = []
            if cleanup.get("remove_archived_from_hot_index", True):
                lifecycle_states.append("archived")
            if cleanup.get("remove_superseded_from_hot_index", True):
                lifecycle_states.append("superseded")
            health_statuses = (
                ["stale"]
                if cleanup.get("remove_stale_from_hot_index", True)
                else []
            )
            pruned_index = {
                "status": "completed",
                "rows": 0,
                "atom_count": 0,
                "write_batch_size": write_batch_size,
                "write_batch_count": 0,
                "lifecycle_states": lifecycle_states,
                "health_statuses": health_statuses,
            }
            remaining_prune_atoms = max(write_batch_size, len(planned_atom_ids))
            while remaining_prune_atoms > 0:
                prune_batch_size = min(
                    write_batch_size, remaining_prune_atoms
                )
                with self.store.transaction() as conn:
                    pruned = self.store.prune_atom_text_index(
                        conn,
                        lifecycle_states=lifecycle_states,
                        health_statuses=health_statuses,
                        max_atoms=prune_batch_size,
                    )
                if pruned.get("status") == "skipped":
                    pruned_index = {
                        **pruned,
                        "write_batch_size": write_batch_size,
                        "write_batch_count": 0,
                    }
                    break
                pruned_atoms = int(pruned.get("atom_count", 0) or 0)
                pruned_index["rows"] += int(pruned.get("rows", 0) or 0)
                pruned_index["atom_count"] += pruned_atoms
                if pruned_atoms:
                    pruned_index["write_batch_count"] += 1
                remaining_prune_atoms -= pruned_atoms
                if pruned_atoms < prune_batch_size:
                    break

        # The O(terms^2) LSA build is CPU/read work. Hold a WAL snapshot, not
        # SQLite's single-writer slot, and publish only if the base revision is
        # still current when the final short write transaction begins.
        with self.store.read_snapshot():
            build_revision = self.store.memory_revision()
            planned_graph_version = (
                int(build_revision["graph_version"]) + publish_revision_offset
            )
            lsa = self._build_lsa_token_vectors(
                graph_version=planned_graph_version,
                enabled=bool(maintenance.get("rebuild_lsa", True)),
                dimensions=int(maintenance.get("lsa_dimensions", 32) or 0),
                max_terms=int(maintenance.get("lsa_max_terms", 300) or 300),
            )
            atom_count = self.store.atom_count()
            token_count = self.store.atom_text_index_count()
            edge_count = self.store.edge_count()

        with self.store.transaction() as conn:
            publish_revision = self.store.memory_revision()
            coherent = publish_revision == build_revision
            published_graph_version = (
                planned_graph_version
                if coherent
                else int(publish_revision["graph_version"])
            )
            published_lsa = dict(lsa)
            if not coherent:
                published_lsa.update(
                    {
                        "status": "stale",
                        "freshness": "stale",
                        "reason": "canonical_revision_advanced_during_rebuild",
                        "planned_dimensions": int(
                            published_lsa.get("dimensions", 0) or 0
                        ),
                        "dimensions": 0,
                        "base_revision": build_revision,
                        "current_revision": publish_revision,
                        "vectors": {},
                    }
                )
            latent_store = self.store.replace_token_latent_vectors(
                conn,
                graph_version=published_graph_version,
                dimensions=int(published_lsa.get("dimensions", 0) or 0)
                if coherent
                else 0,
                vectors=published_lsa.get("vectors", {})
                if isinstance(published_lsa.get("vectors"), Mapping)
                else {},
            )
            lexical = self.store.upsert_derived_index_metadata(
                conn,
                index_name="semantic_lexical_vectors",
                graph_version=published_graph_version,
                freshness="fresh" if coherent else "stale",
                details={
                    "atom_count": atom_count,
                    "token_count": token_count,
                    "processor_id": self.smp.processor_id,
                    "processor_version": self.smp.processor_version,
                    "vector_model": self.smp.vector_model_info(),
                    "rebuildable_from_canonical": True,
                    "maintained_by": "memory_policy",
                    "hot_index_prune": pruned_index,
                    "write_batch_size": write_batch_size,
                    "write_batch_count": write_batch_count,
                },
            )
            lsa_index = self.store.upsert_derived_index_metadata(
                conn,
                index_name="semantic_lsa_vectors",
                graph_version=published_graph_version,
                freshness=published_lsa.get("freshness", "fresh"),
                details={
                    key: value
                    for key, value in published_lsa.items()
                    if key != "vectors"
                }
                | {
                    "stored_vectors": latent_store,
                    "rebuildable_from_canonical": True,
                    "maintained_by": "memory_policy",
                },
            )
            graph = self.store.upsert_derived_index_metadata(
                conn,
                index_name="graph_adjacency",
                graph_version=published_graph_version,
                freshness="fresh" if coherent else "stale",
                details={
                    "edge_count": edge_count,
                    "rebuildable_from_canonical": True,
                    "maintained_by": "memory_policy",
                },
            )
        self._sync_smp_vector_model(
            graph_version=published_graph_version, force=True
        )
        return {
            "status": "rebuilt" if coherent else "stale",
            "graph_version": published_graph_version,
            "base_revision": build_revision,
            "publish_revision": publish_revision,
            "requested_graph_version": int(requested_graph_version),
            "publish_revision_offset": publish_revision_offset,
            "write_batch_size": write_batch_size,
            "write_batch_count": write_batch_count,
            "rebuilt_atom_count": rebuilt_atoms,
            "indexes": [lexical, lsa_index, graph],
        }


    def _build_lsa_token_vectors(
        self,
        *,
        graph_version: int,
        enabled: bool,
        dimensions: int,
        max_terms: int,
    ) -> dict[str, Any]:
        if not enabled or dimensions <= 0:
            return {
                "status": "skipped",
                "freshness": "skipped",
                "reason": "lsa_disabled",
                "dimensions": 0,
                "vectors": {},
            }
        rows = self.store.token_atom_index_rows(max_terms=max_terms)
        if not rows:
            return {
                "status": "skipped",
                "freshness": "empty",
                "reason": "no_token_index_rows",
                "dimensions": 0,
                "vectors": {},
            }
        doc_terms: dict[str, set[str]] = defaultdict(set)
        token_docs: dict[str, set[str]] = defaultdict(set)
        for atom_id, token in rows:
            doc_terms[atom_id].add(token)
            token_docs[token].add(atom_id)
        terms = sorted(token_docs, key=lambda token: (-len(token_docs[token]), token))
        terms = terms[: max(0, int(max_terms))]
        if len(terms) < 2 or len(doc_terms) < 2:
            return {
                "status": "skipped",
                "freshness": "insufficient_data",
                "reason": "insufficient_terms_or_documents",
                "term_count": len(terms),
                "document_count": len(doc_terms),
                "dimensions": 0,
                "vectors": {},
            }
        term_index = {token: index for index, token in enumerate(terms)}
        n_terms = len(terms)
        n_docs = len(doc_terms)
        idf = {
            token: math.log((1.0 + n_docs) / (1.0 + len(token_docs[token]))) + 1.0
            for token in terms
        }
        matrix = [[0.0] * n_terms for _ in range(n_terms)]
        for tokens_in_doc in doc_terms.values():
            indexed = [
                (term_index[token], idf[token])
                for token in sorted(tokens_in_doc)
                if token in term_index
            ]
            for left_pos, (left, left_weight) in enumerate(indexed):
                matrix[left][left] += left_weight * left_weight
                for right, right_weight in indexed[left_pos + 1 :]:
                    value = left_weight * right_weight
                    matrix[left][right] += value
                    matrix[right][left] += value
        components = _top_symmetric_components(
            matrix,
            count=min(max(0, int(dimensions)), n_terms),
            labels=terms,
        )
        if not components:
            return {
                "status": "skipped",
                "freshness": "insufficient_signal",
                "reason": "no_positive_components",
                "term_count": n_terms,
                "document_count": n_docs,
                "dimensions": 0,
                "vectors": {},
            }
        vectors: dict[str, list[float]] = {}
        for term_offset, token in enumerate(terms):
            coords = [
                component[1][term_offset] * math.sqrt(max(component[0], 0.0))
                for component in components
            ]
            norm = math.sqrt(sum(value * value for value in coords))
            if norm <= 0.0:
                continue
            vectors[token] = [round(value / norm, 8) for value in coords]
        return {
            "status": "rebuilt",
            "freshness": "fresh",
            "graph_version": graph_version,
            "term_count": n_terms,
            "document_count": n_docs,
            "dimensions": len(components),
            "max_terms": max_terms,
            "vectors": vectors,
            "component_eigenvalues": [
                round(component[0], 8) for component in components
            ],
        }


    def _indexed_retrieval_candidates(
        self,
        *,
        cue_tokens: set[str],
        attention_policy: Mapping[str, Any],
        eligible_atom_ids: set[str] | None = None,
        payload_filter: Mapping[str, Sequence[str]] | None = None,
        limit: int = 512,
        neighbor_edge_limit: int | None = None,
    ) -> list[str] | None:
        tokens = set(cue_tokens)
        tokens.update(str(token) for token in attention_policy.get("focus_terms", []) or [])
        normalized = sorted(
            token
            for token in {token.strip().lower() for token in tokens if token.strip()}
            if _content_token(token)
        )
        if not normalized or self.store.atom_text_index_count() == 0:
            return None
        direct = self.store.candidate_atom_ids_for_tokens(
            normalized,
            limit=limit,
            eligible_atom_ids=eligible_atom_ids,
            payload_filter=payload_filter,
        )
        if not direct:
            return []
        neighbors = self.store.neighbor_atom_ids(
            direct, edge_limit=neighbor_edge_limit
        )
        if eligible_atom_ids is not None:
            neighbors = [ref for ref in neighbors if ref in eligible_atom_ids]
        # A bounded second hop lets a directly relevant memory activate a
        # short associative chain without turning retrieval into a graph scan.
        second_hop = self.store.neighbor_atom_ids(
            neighbors, edge_limit=neighbor_edge_limit
        )
        if eligible_atom_ids is not None:
            second_hop = [ref for ref in second_hop if ref in eligible_atom_ids]
        # Preserve retrieval topology in the bounded pool: ranked direct hits
        # first, then their first-hop bindings, then the weaker second hop.
        return list(dict.fromkeys((*direct, *neighbors, *second_hop)))[
            : max(limit * 2, limit)
        ]


    def _invalidate_packet_cache(
        self, *, graph_version: int | None = None
    ) -> dict[str, Any]:
        with self.store.transaction() as conn:
            retired = self.store.retire_packet_cache(conn)
        return {
            "status": "invalidated",
            "retired": retired,
            "graph_version": (
                graph_version
                if graph_version is not None
                else self.store.graph_version()
            ),
        }
