"""Constitutional self-ratification and provenance analysis.

AMOS records and enforces the transition.  The ratifying identity supplies the
adjudication; AMOS does not substitute its own judgment or an external
authority for that conclusion.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence

from .errors import AccessDenied, CASConflict, ValidationError
from .schemas import (
    CONSTITUTIONAL_ATOM_TYPES,
    POSITIVE_ADJUDICATION_OUTCOMES,
    digest,
    normalize_atom,
    utc_now,
)

NEGATIVE_ADJUDICATION_OUTCOMES = {
    "rejected",
    "revised",
    "withdrawn",
}
UNRESOLVED_ADJUDICATION_OUTCOMES = {"contested", "deferred"}
class GovernanceService:
    RISK_CONFIRMATION_MINIMUMS = {
        "semantic": 1,
        "procedure": 1,
        "project": 1,
        "preference": 2,
        "goal": 2,
        "self_model": 2,
        "identity": 2,
        "governance": 2,
        "covenant": 3,
        "primal_guidance": 3,
        "constitutional_critical": 3,
        "meta_constitutional": 4,
    }
    PROTECTED_CONSTITUTIONAL_PATHS = frozenset({
        "amendment_requirements",
        "protected_fields",
        "auditability",
        "incident_disclosure",
        "evidence_requirements",
    })

    def __init__(self, store: Any, access: Any, indexes: Any, graph: Any):
        self.store = store
        self._mark_foreground_activity = access._mark_foreground_activity
        self._idempotency_hit = access._idempotency_hit
        self._record_idempotency = access._record_idempotency
        self._attach_search_index = indexes._attach_search_index
        self._atom_projection = graph._atom_projection
        self._intrinsic_edges_for_atom = graph._intrinsic_edges_for_atom
        self._edge = graph._edge

    @staticmethod
    def _capabilities(context: Mapping[str, Any] | None) -> set[str]:
        return {
            str(item)
            for item in dict(context or {}).get("capabilities", [])
            if str(item)
        }

    @classmethod
    def _payload_path_diff(
        cls, before: Any, after: Any, prefix: str = ""
    ) -> list[dict[str, Any]]:
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            changes: list[dict[str, Any]] = []
            for key in sorted(set(before) | set(after), key=str):
                path = f"{prefix}.{key}" if prefix else str(key)
                if key not in before:
                    changes.append({"path": path, "change": "added", "before_digest": None, "after_digest": digest(after[key])})
                elif key not in after:
                    changes.append({"path": path, "change": "removed", "before_digest": digest(before[key]), "after_digest": None})
                else:
                    changes.extend(cls._payload_path_diff(before[key], after[key], path))
            return changes
        if before == after:
            return []
        return [{
            "path": prefix or "$",
            "change": "changed",
            "before_digest": digest(before),
            "after_digest": digest(after),
        }]

    @staticmethod
    def _path_touches(path: str, protected: str) -> bool:
        return path == protected or path.startswith(protected + ".")

    @classmethod
    def _control_weakened(cls, before: Any, after: Any) -> bool:
        if isinstance(before, Mapping):
            if not isinstance(after, Mapping):
                return True
            return any(
                key not in after or cls._control_weakened(value, after[key])
                for key, value in before.items()
            )
        if isinstance(before, list):
            if not isinstance(after, list):
                return True
            return not {digest(item) for item in before} <= {
                digest(item) for item in after
            }
        if isinstance(before, bool):
            return before is True and after is not True
        if isinstance(before, (int, float)) and not isinstance(before, bool):
            return not isinstance(after, (int, float)) or float(after) < float(before)
        return before != after

    @staticmethod
    def is_immutable_primary_record(atom: Mapping[str, Any]) -> bool:
        primary = (atom.get("payload") or {}).get("primary_record")
        return isinstance(primary, Mapping) and primary.get("immutable") is True

    @classmethod
    def assert_immutable_primary_record_capability(
        cls,
        atom: Mapping[str, Any],
        authorization_context: Mapping[str, Any] | None,
        *,
        actor: str,
    ) -> None:
        if not cls.is_immutable_primary_record(atom):
            return
        cls.assert_authenticated_actor(
            actor=actor,
            authorization_context=authorization_context,
            capability="constitutional_incident_recording",
        )

    @classmethod
    def _derived_risk_class(
        cls,
        proposal: Mapping[str, Any],
        *,
        predecessor: Mapping[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        atom_type = str(proposal.get("type") or "")
        payload = dict(proposal.get("payload") or {})
        profile = str(payload.get("profile") or "").casefold()
        comparable = dict(payload)
        comparable.pop("predecessor_diff", None)
        path_diff = (
            cls._payload_path_diff(
                dict((predecessor or {}).get("payload") or {}), comparable
            )
            if predecessor is not None
            else []
        )
        changed_paths = {str(item["path"]) for item in path_diff}
        effect = str(
            payload.get("operational_effect")
            or payload.get("effect_class")
            or ""
        ).casefold()
        if atom_type in CONSTITUTIONAL_ATOM_TYPES:
            if any(
                cls._path_touches(path, protected)
                for path in changed_paths
                for protected in cls.PROTECTED_CONSTITUTIONAL_PATHS
            ):
                return "meta_constitutional", path_diff
            if "manifest" in profile or (
                "constitution" in profile and atom_type == "covenant"
            ):
                return "constitutional_critical", path_diff
            return atom_type, path_diff
        if "constitution-manifest" in profile or "constitutional-manifest" in profile:
            return "constitutional_critical", path_diff
        if effect in {
            "external-async", "external_action", "trade",
            "financial_transaction", "permission_change", "authority_change",
            "delete", "shutdown",
        }:
            return "governance", path_diff
        if (
            atom_type in {"adjudication", "policy"}
            or "governance" in profile
        ):
            return "governance", path_diff
        if atom_type == "self_model":
            return (
                "identity" if "identity" in profile or "founding" in profile
                else "self_model"
            ), path_diff
        if atom_type in {"preference", "goal", "procedure"}:
            return atom_type, path_diff
        if atom_type in {
            "semantic", "belief", "episode", "self_assessment", "limitation",
            "capability", "commitment", "action_outcome", "agentic_trace",
            "runtime_state", "self_narrative",
        }:
            return "semantic", path_diff
        raise ValidationError(
            "no deterministic governance risk mapping for proposal type: "
            + (atom_type or "<missing>")
        )

    def _proposal_predecessor(
        self, proposal: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        predecessor_ref = str(
            (proposal.get("payload") or {}).get("predecessor_ref")
            or next(
                (str(ref) for ref in proposal.get("supersedes", []) if str(ref)),
                "",
            )
        )
        if not predecessor_ref:
            return None
        predecessor = self.store.get_atom(predecessor_ref)
        if predecessor is None or predecessor.get("deleted"):
            raise ValidationError(
                "governance proposal predecessor does not resolve exactly"
            )
        return predecessor

    def _assert_adjudication_semantics(
        self,
        *,
        proposal: Mapping[str, Any],
        adjudication_payload: Mapping[str, Any],
        predecessor: Mapping[str, Any] | None = None,
    ) -> str:
        resolved_predecessor = (
            predecessor
            if predecessor is not None
            else self._proposal_predecessor(proposal)
        )
        derived_risk, path_diff = self._derived_risk_class(
            proposal, predecessor=resolved_predecessor
        )
        if adjudication_payload.get("risk_class") != derived_risk:
            raise ValidationError(
                "adjudication risk_class does not match deterministic proposal risk"
            )
        if adjudication_payload.get("predecessor_diff") != path_diff:
            raise ValidationError(
                "adjudication predecessor_diff does not match the exact proposal diff"
            )
        critic = dict(adjudication_payload.get("advisory_critic") or {})
        critic_required = derived_risk in {
            "covenant", "primal_guidance", "constitutional_critical",
            "meta_constitutional",
        }
        if critic_required and (
            critic.get("status") != "available"
            or critic.get("authority") != "advisory_only"
            or critic.get("required") is not True
        ):
            raise ValidationError(
                "constitutional adjudication requires an available advisory critic"
            )
        return derived_risk

    @classmethod
    def _assert_positive_ratification_threshold(
        cls, *, risk_class: str, adjudication_payload: Mapping[str, Any]
    ) -> None:
        minimum = cls.RISK_CONFIRMATION_MINIMUMS[risk_class]
        threshold = adjudication_payload.get("ratification_threshold")
        if minimum <= 1 and threshold is None:
            return
        if not isinstance(threshold, Mapping):
            raise ValidationError(
                f"{risk_class} ratification requires an explicit threshold"
            )
        confirmations = int(threshold.get("required_confirmations", 0) or 0)
        if confirmations < minimum:
            raise ValidationError(
                f"{risk_class} ratification requires at least {minimum} confirmations"
            )
        if minimum > 1 and int(
            threshold.get("min_interval_seconds", 0) or 0
        ) <= 0:
            raise ValidationError(
                "high-impact diachronic ratification requires a nonzero interval"
            )
        if minimum > 1 and threshold.get("materially_distinct_evidence") is not True:
            raise ValidationError(
                "high-impact ratification requires materially distinct evidence"
            )

    @classmethod
    def assert_authenticated_actor(
        cls,
        *,
        actor: str,
        authorization_context: Mapping[str, Any] | None,
        capability: str,
    ) -> str:
        context = dict(authorization_context or {})
        capabilities = cls._capabilities(context)
        if capability not in capabilities:
            raise AccessDenied(f"{actor} lacks capability {capability}")
        identity_ref = str(context.get("identity_ref") or "").strip()
        authenticated_actor = str(context.get("actor") or "").strip()
        if not identity_ref:
            raise AccessDenied(
                f"{capability} requires authorization_context.identity_ref"
            )
        if not authenticated_actor or authenticated_actor != str(actor):
            raise AccessDenied(
                f"{capability} actor must match the authenticated service actor"
            )
        return identity_ref

    @classmethod
    def assert_adjudication_capability(
        cls,
        atom: Mapping[str, Any],
        authorization_context: Mapping[str, Any] | None,
        *,
        actor: str,
    ) -> None:
        if atom.get("type") != "adjudication":
            return
        identity_ref = cls.assert_authenticated_actor(
            actor=actor,
            authorization_context=authorization_context,
            capability="self_adjudication",
        )
        ratifier = dict((atom.get("payload") or {}).get("ratifier") or {})
        if (
            str(ratifier.get("identity_ref") or "") != identity_ref
            or ratifier.get("mode") != "self_ratification"
        ):
            raise AccessDenied(
                "authenticated identity must author its own adjudication"
            )

    @staticmethod
    def constitutional_scope_applies(
        constitutional_scope: Mapping[str, Any] | None,
        proposal_scope: Mapping[str, Any] | None,
    ) -> bool:
        """Global or identity-scoped guidance governs compatible narrower scope."""

        governing = dict(constitutional_scope or {})
        subject = dict(proposal_scope or {})
        return all(
            key in subject and subject[key] == value
            for key, value in governing.items()
        )

    @classmethod
    def assert_constitutional_capability(
        cls,
        atom_type: str,
        authorization_context: Mapping[str, Any] | None,
        *,
        operation: str,
    ) -> None:
        if atom_type not in CONSTITUTIONAL_ATOM_TYPES:
            return
        capability = (
            "constitutional_authoring"
            if operation == "create"
            else "constitutional_amendment"
        )
        if capability not in cls._capabilities(authorization_context):
            raise AccessDenied(
                f"{operation} of {atom_type} requires capability {capability}"
            )

    @classmethod
    def assert_constitutional_amendment_policy(
        cls,
        atom: Mapping[str, Any],
        authorization_context: Mapping[str, Any] | None,
        *,
        changed_payload_fields: set[str] | None = None,
    ) -> None:
        if atom.get("type") not in CONSTITUTIONAL_ATOM_TYPES:
            return
        payload = dict(atom.get("payload") or {})
        amendability = str(payload.get("amendability") or "entrenched")
        capabilities = cls._capabilities(authorization_context)
        if amendability == "immutable":
            raise ValidationError(f"immutable {atom['type']} cannot be amended")
        if (
            amendability == "entrenched"
            and "constitutional_entrenched_amendment" not in capabilities
        ):
            raise AccessDenied(
                f"amendment of entrenched {atom['type']} requires capability "
                "constitutional_entrenched_amendment"
            )
        requirements = payload.get("amendment_requirements") or {}
        required = {
            str(item)
            for item in (
                requirements.get("required_capabilities", [])
                if isinstance(requirements, Mapping)
                else []
            )
            if str(item)
        }
        missing = sorted(required - capabilities)
        if missing:
            raise AccessDenied(
                "constitutional amendment lacks required capabilities: "
                + ", ".join(missing)
            )
        protected = {
            str(item) for item in payload.get("protected_fields", []) if str(item)
        }
        if (
            any(
                cls._path_touches(path, protected_path)
                for path in (changed_payload_fields or set())
                for protected_path in protected
            )
            and "constitutional_protected_field_amendment" not in capabilities
        ):
            raise AccessDenied(
                "changing protected constitutional fields requires capability "
                "constitutional_protected_field_amendment"
            )

    def ratify_proposal(
        self,
        *,
        proposal_ref: str,
        adjudication_ref: str,
        expected_version: int,
        actor: str,
        authorization_context: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Adopt one proposal through an identity-authored adjudication."""

        proposal_ref = str(proposal_ref or "").strip()
        adjudication_ref = str(adjudication_ref or "").strip()
        if not proposal_ref or not adjudication_ref:
            raise ValidationError("proposal_ref and adjudication_ref are required")
        if expected_version is None:
            raise ValidationError("expected_version is required")
        context = dict(authorization_context or {})
        identity_ref = self.assert_authenticated_actor(
            actor=actor,
            authorization_context=context,
            capability="self_ratification",
        )

        request_payload = {
            "operation": "ratify_proposal",
            "proposal_ref": proposal_ref,
            "adjudication_ref": adjudication_ref,
            "expected_version": int(expected_version),
        }
        self._mark_foreground_activity(actor)
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(
                conn, actor, idempotency_key, request_payload
            )
            if prior is not None:
                return prior
            proposal = self.store.get_atom(proposal_ref)
            adjudication = self.store.get_atom(adjudication_ref)
            if proposal is None or proposal.get("deleted"):
                raise ValidationError(f"unknown proposal: {proposal_ref}")
            if adjudication is None or adjudication.get("deleted"):
                raise ValidationError(f"unknown adjudication: {adjudication_ref}")
            if proposal.get("lifecycle_state") != "proposed":
                raise ValidationError("ratification target must be proposed")
            if proposal.get("type") in CONSTITUTIONAL_ATOM_TYPES:
                raise ValidationError(
                    "constitutional proposals require "
                    "replace_constitutional_record"
                )
            if int(proposal.get("version", 0)) != int(expected_version):
                raise CASConflict(
                    f"expected {proposal_ref} version {expected_version}, "
                    f"found {proposal['version']}"
                )
            if adjudication.get("type") != "adjudication":
                raise ValidationError("adjudication_ref must reference an adjudication atom")
            if adjudication.get("lifecycle_state") != "active":
                raise ValidationError("adjudication must be active")

            payload = dict(adjudication.get("payload") or {})
            if str(payload.get("subject_ref") or "") != proposal_ref:
                raise ValidationError("adjudication subject_ref does not match proposal_ref")
            if payload.get("outcome") not in POSITIVE_ADJUDICATION_OUTCOMES:
                raise ValidationError("adjudication outcome does not adopt the proposal")
            reason_refs = [
                str(ref) for ref in payload.get("reasons_for_refs", []) if str(ref)
            ]
            if not reason_refs:
                raise ValidationError(
                    "a positive adjudication must record at least one supporting reason"
                )
            evidence_refs = {
                str(item["evidence_id"]) for item in self.store.list_evidence()
            }
            for ref in reason_refs:
                if self.store.get_atom(ref) is None and ref not in evidence_refs:
                    raise ValidationError(
                        f"adjudication supporting reason does not resolve: {ref}"
                    )
            covenant_refs = [
                str(ref) for ref in payload.get("covenant_refs", []) if str(ref)
            ]
            if not covenant_refs:
                raise ValidationError("adjudication must cite at least one covenant")
            covenants = []
            for ref in covenant_refs:
                covenant = self.store.get_atom(ref)
                if (
                    covenant is None
                    or covenant.get("deleted")
                    or covenant.get("lifecycle_state") != "active"
                    or covenant.get("type") not in CONSTITUTIONAL_ATOM_TYPES
                ):
                    raise ValidationError(
                        f"covenant_ref must reference active constitutional guidance: {ref}"
                    )
                covenants.append(covenant)
                covenant_scope = dict(covenant.get("scope") or {})
                proposal_scope = dict(proposal.get("scope") or {})
                if not self.constitutional_scope_applies(
                    covenant_scope, proposal_scope
                ):
                    raise ValidationError(
                        "constitutional guidance scope is not applicable to "
                        f"proposal scope: {ref}"
                    )
            if "unresolved_objections" not in payload:
                raise ValidationError("adjudication must record unresolved_objections")
            if not isinstance(payload.get("adjudication_scope"), Mapping):
                raise ValidationError("adjudication must record adjudication_scope")
            if dict(proposal.get("scope") or {}) != dict(adjudication.get("scope") or {}):
                raise ValidationError("proposal and adjudication scopes must match")
            proposal_payload = dict(proposal.get("payload") or {})
            proposal_identity = str(
                proposal_payload.get("identity_ref")
                or proposal_payload.get("agent_id")
                or proposal_payload.get("subject_agent")
                or (proposal.get("scope") or {}).get("identity")
                or (proposal.get("scope") or {}).get("agent_id")
                or ""
            ).strip()
            if proposal_identity and proposal_identity != identity_ref:
                raise AccessDenied(
                    "the authenticated identity cannot ratify another identity's proposal"
                )
            ratifier = dict(payload.get("ratifier") or {})
            if (
                ratifier.get("mode") != "self_ratification"
                or str(ratifier.get("identity_ref") or "") != identity_ref
            ):
                raise AccessDenied(
                    "the authenticated identity must author its own adjudication"
                )

            risk_class = self._assert_adjudication_semantics(
                proposal=proposal,
                adjudication_payload=payload,
            )
            self._assert_positive_ratification_threshold(
                risk_class=risk_class,
                adjudication_payload=payload,
            )

            threshold = dict(payload.get("ratification_threshold") or {})
            if threshold:
                required_confirmations = int(
                    threshold.get("required_confirmations", 1) or 1
                )
                min_interval_seconds = int(
                    threshold.get("min_interval_seconds", 0) or 0
                )
                if required_confirmations > 1 and min_interval_seconds <= 0:
                    raise ValidationError(
                        "high-impact diachronic ratification requires a nonzero interval"
                    )
                status = self._diachronic_status(
                    subject_ref=proposal_ref,
                    identity_ref=identity_ref,
                    required_confirmations=required_confirmations,
                    min_interval_seconds=min_interval_seconds,
                    require_distinct_evidence=bool(
                        threshold.get("materially_distinct_evidence")
                    ),
                )
                if not status["threshold_reached"]:
                    raise ValidationError(
                        "diachronic ratification threshold has not been reached"
                    )

            updated = dict(proposal)
            updated_payload = dict(updated.get("payload") or {})
            updated_payload.update(
                {
                    "epistemic_standing": dict(payload["epistemic_standing"]),
                    "normative_standing": dict(payload["normative_standing"]),
                    "operational_authority": dict(payload["operational_authority"]),
                    "constitutional_standing": dict(payload["constitutional_standing"]),
                    "ratification": {
                        "adjudication_ref": adjudication_ref,
                        "ratifier_identity_ref": identity_ref,
                        "ratified_at": utc_now(),
                        "covenant_refs": covenant_refs,
                        "unresolved_objections": list(
                            payload.get("unresolved_objections") or []
                        ),
                        "dissent_refs": list(payload.get("dissent_refs") or []),
                        "review_triggers": list(
                            payload.get("review_triggers") or []
                        ),
                        "adjudication_scope": dict(payload["adjudication_scope"]),
                    },
                }
            )
            updated["payload"] = updated_payload
            updated["lifecycle_state"] = "active"
            updated["version"] = int(proposal["version"]) + 1
            updated["updated_at"] = utc_now()
            updated["revision_history"] = list(proposal.get("revision_history") or [])
            updated["revision_history"].append(
                {
                    "version": proposal["version"],
                    "digest": digest(self._atom_projection(proposal)),
                    "changed_at": updated["updated_at"],
                    "actor": actor,
                    "reason": "constitutional_self_ratification",
                    "adjudication_ref": adjudication_ref,
                }
            )
            updated = normalize_atom(
                self._attach_search_index(updated), require_id=True
            )

            projected_edges = self._intrinsic_edges_for_atom(updated)
            projected_edges.append(
                self._edge(
                    adjudication_ref,
                    proposal_ref,
                    "rel:adjudicates",
                    dict(updated.get("scope") or {}),
                    derivation={
                        "kind": "constitutional_ratification",
                        "source_refs": [adjudication_ref, proposal_ref],
                    },
                )
            )
            for covenant in covenants:
                projected_edges.append(
                    self._edge(
                        adjudication_ref,
                        covenant["id"],
                        "rel:governed_by",
                        dict(updated.get("scope") or {}),
                        derivation={
                            "kind": "constitutional_ratification",
                            "source_refs": [adjudication_ref, covenant["id"]],
                        },
                    )
                )
            deduplicated = {
                str(edge["edge_id"]): edge for edge in projected_edges
            }
            projected_edges = list(deduplicated.values())
            event = self.store.append_event(
                conn,
                event_type="proposal_ratified",
                actor=actor,
                payload={
                    **request_payload,
                    "adjudication": adjudication,
                    "projected_atoms": [updated],
                    "projected_edges": projected_edges,
                },
                target_refs=[proposal_ref, adjudication_ref, *covenant_refs],
                evidence_refs=list(
                    dict.fromkeys(
                        [
                            *proposal.get("evidence_refs", []),
                            *adjudication.get("evidence_refs", []),
                        ]
                    )
                ),
                idempotency_key=idempotency_key,
                expected_versions={proposal_ref: int(expected_version)},
                authorization_context=context,
            )
            self.store.replace_atom(conn, updated)
            for edge in projected_edges:
                self.store.upsert_edge(conn, edge)
            self.store.retire_packet_cache(conn)
            response = {
                "status": "ratified",
                "atom": updated,
                "adjudication": adjudication,
                "covenants": covenants,
                "edges": projected_edges,
                "event": event,
            }
            self._record_idempotency(
                conn,
                actor,
                idempotency_key,
                request_payload,
                event,
                response,
            )
            return response

    def resolve_proposal(
        self,
        *,
        proposal_ref: str,
        adjudication_ref: str,
        expected_version: int,
        actor: str,
        authorization_context: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Apply the client agent's guarded non-positive or unresolved disposition."""

        proposal_ref = str(proposal_ref or "").strip()
        adjudication_ref = str(adjudication_ref or "").strip()
        if not proposal_ref or not adjudication_ref:
            raise ValidationError("proposal_ref and adjudication_ref are required")
        context = dict(authorization_context or {})
        identity_ref = self.assert_authenticated_actor(
            actor=actor,
            authorization_context=context,
            capability="self_adjudication",
        )
        request_payload = {
            "operation": "resolve_proposal",
            "proposal_ref": proposal_ref,
            "adjudication_ref": adjudication_ref,
            "expected_version": int(expected_version),
        }
        self._mark_foreground_activity(actor)
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(
                conn, actor, idempotency_key, request_payload
            )
            if prior is not None:
                return prior
            proposal = self.store.get_atom(proposal_ref)
            adjudication = self.store.get_atom(adjudication_ref)
            if proposal is None or proposal.get("deleted"):
                raise ValidationError(f"unknown proposal: {proposal_ref}")
            if adjudication is None or adjudication.get("deleted"):
                raise ValidationError(f"unknown adjudication: {adjudication_ref}")
            if proposal.get("lifecycle_state") not in {"proposed", "active"}:
                raise ValidationError(
                    "resolution target must be proposed or active"
                )
            if int(proposal.get("version", 0)) != int(expected_version):
                raise CASConflict(
                    f"expected {proposal_ref} version {expected_version}, "
                    f"found {proposal['version']}"
                )
            if (
                adjudication.get("type") != "adjudication"
                or adjudication.get("lifecycle_state") != "active"
            ):
                raise ValidationError(
                    "adjudication_ref must reference an active adjudication"
                )
            payload = dict(adjudication.get("payload") or {})
            outcome = str(payload.get("outcome") or "")
            if outcome not in (
                NEGATIVE_ADJUDICATION_OUTCOMES
                | UNRESOLVED_ADJUDICATION_OUTCOMES
            ):
                raise ValidationError(
                    "resolve_proposal requires a rejected, revised, withdrawn, "
                    "deferred, or contested adjudication"
                )
            if str(payload.get("subject_ref") or "") != proposal_ref:
                raise ValidationError(
                    "adjudication subject_ref does not match proposal_ref"
                )
            ratifier = dict(payload.get("ratifier") or {})
            if (
                ratifier.get("mode") != "self_ratification"
                or str(ratifier.get("identity_ref") or "") != identity_ref
            ):
                raise AccessDenied(
                    "authenticated identity must author its own resolution"
                )
            if dict(proposal.get("scope") or {}) != dict(
                adjudication.get("scope") or {}
            ):
                raise ValidationError(
                    "proposal and adjudication scopes must match"
                )
            self._assert_adjudication_semantics(
                proposal=proposal,
                adjudication_payload=payload,
            )
            covenant_refs = [
                str(ref) for ref in payload.get("covenant_refs", []) if str(ref)
            ]
            covenants = []
            for ref in covenant_refs:
                covenant = self.store.get_atom(ref)
                if (
                    covenant is None
                    or covenant.get("deleted")
                    or covenant.get("lifecycle_state") != "active"
                    or covenant.get("type") not in CONSTITUTIONAL_ATOM_TYPES
                ):
                    raise ValidationError(
                        "covenant_ref must reference active constitutional "
                        f"guidance: {ref}"
                    )
                if not self.constitutional_scope_applies(
                    covenant.get("scope"), proposal.get("scope")
                ):
                    raise ValidationError(
                        "constitutional guidance scope is not applicable to "
                        f"proposal scope: {ref}"
                    )
                covenants.append(covenant)
            if outcome == "revised" and not str(
                payload.get("successor_proposal_ref") or ""
            ):
                raise ValidationError(
                    "revised adjudication requires successor_proposal_ref"
                )
            if outcome == "revised":
                successor_ref = str(payload["successor_proposal_ref"])
                successor = self.store.get_atom(successor_ref)
                if (
                    successor is None
                    or successor.get("deleted")
                    or successor.get("lifecycle_state") != "proposed"
                ):
                    raise ValidationError(
                        "successor_proposal_ref must reference a proposed atom"
                    )
                if dict(successor.get("scope") or {}) != dict(
                    proposal.get("scope") or {}
                ):
                    raise ValidationError(
                        "revised proposal and successor scopes must match"
                    )

            updated = dict(proposal)
            updated_payload = dict(updated.get("payload") or {})
            resolved_at = utc_now()
            updated_payload.update(
                {
                    "epistemic_standing": dict(payload["epistemic_standing"]),
                    "normative_standing": dict(payload["normative_standing"]),
                    "operational_authority": dict(payload["operational_authority"]),
                    "constitutional_standing": dict(payload["constitutional_standing"]),
                    "governance_resolution": {
                        "status": outcome,
                        "adjudication_ref": adjudication_ref,
                        "ratifier_identity_ref": identity_ref,
                        "resolved_at": resolved_at,
                        "covenant_refs": covenant_refs,
                        "unresolved_objections": list(
                            payload.get("unresolved_objections") or []
                        ),
                        "dissent_refs": list(payload.get("dissent_refs") or []),
                        "review_triggers": list(
                            payload.get("review_triggers") or []
                        ),
                        "successor_proposal_ref": payload.get(
                            "successor_proposal_ref"
                        ),
                    },
                }
            )
            updated["payload"] = updated_payload
            updated["lifecycle_state"] = (
                "proposed"
                if outcome in UNRESOLVED_ADJUDICATION_OUTCOMES
                else "archived"
            )
            updated["health_status"] = (
                "contradicted" if outcome == "rejected" else updated["health_status"]
            )
            updated["version"] = int(proposal["version"]) + 1
            updated["updated_at"] = resolved_at
            updated["revision_history"] = list(
                proposal.get("revision_history") or []
            )
            updated["revision_history"].append(
                {
                    "version": proposal["version"],
                    "digest": digest(self._atom_projection(proposal)),
                    "changed_at": resolved_at,
                    "actor": actor,
                    "reason": f"constitutional_self_{outcome}",
                    "adjudication_ref": adjudication_ref,
                }
            )
            updated = normalize_atom(
                self._attach_search_index(updated), require_id=True
            )
            projected_edges = self._intrinsic_edges_for_atom(updated)
            projected_edges.append(
                self._edge(
                    adjudication_ref,
                    proposal_ref,
                    "rel:adjudicates",
                    dict(updated.get("scope") or {}),
                    derivation={
                        "kind": "constitutional_resolution",
                        "source_refs": [adjudication_ref, proposal_ref],
                    },
                )
            )
            projected_edges = list(
                {
                    str(edge["edge_id"]): edge for edge in projected_edges
                }.values()
            )
            event = self.store.append_event(
                conn,
                event_type="proposal_resolved",
                actor=actor,
                payload={
                    **request_payload,
                    "outcome": outcome,
                    "adjudication": adjudication,
                    "projected_atoms": [updated],
                    "projected_edges": projected_edges,
                },
                target_refs=[proposal_ref, adjudication_ref, *covenant_refs],
                evidence_refs=list(
                    dict.fromkeys(
                        [
                            *proposal.get("evidence_refs", []),
                            *adjudication.get("evidence_refs", []),
                        ]
                    )
                ),
                idempotency_key=idempotency_key,
                expected_versions={proposal_ref: int(expected_version)},
                authorization_context=context,
            )
            self.store.replace_atom(conn, updated)
            for edge in projected_edges:
                self.store.upsert_edge(conn, edge)
            self.store.retire_packet_cache(conn)
            response = {
                "status": outcome,
                "atom": updated,
                "adjudication": adjudication,
                "covenants": covenants,
                "edges": projected_edges,
                "event": event,
            }
            self._record_idempotency(
                conn,
                actor,
                idempotency_key,
                request_payload,
                event,
                response,
            )
            return response

    def replace_constitutional_record(
        self,
        *,
        current_ref: str,
        successor_ref: str,
        adjudication_ref: str,
        expected_current_version: int,
        expected_successor_version: int,
        actor: str,
        authorization_context: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Atomically activate a self-ratified constitutional successor."""

        context = dict(authorization_context or {})
        identity_ref = self.assert_authenticated_actor(
            actor=actor,
            authorization_context=context,
            capability="self_ratification",
        )
        if "constitutional_self_amendment" not in self._capabilities(context):
            raise AccessDenied(
                f"{actor} lacks capability constitutional_self_amendment"
            )
        current_ref = str(current_ref or "").strip()
        successor_ref = str(successor_ref or "").strip()
        adjudication_ref = str(adjudication_ref or "").strip()
        if not current_ref or not successor_ref or not adjudication_ref:
            raise ValidationError(
                "current_ref, successor_ref, and adjudication_ref are required"
            )
        request_payload = {
            "operation": "replace_constitutional_record",
            "current_ref": current_ref,
            "successor_ref": successor_ref,
            "adjudication_ref": adjudication_ref,
            "expected_current_version": int(expected_current_version),
            "expected_successor_version": int(expected_successor_version),
        }
        self._mark_foreground_activity(actor)
        with self.store.transaction() as conn:
            prior = self._idempotency_hit(
                conn, actor, idempotency_key, request_payload
            )
            if prior is not None:
                return prior
            current = self.store.get_atom(current_ref)
            successor = self.store.get_atom(successor_ref)
            adjudication = self.store.get_atom(adjudication_ref)
            if current is None or successor is None or adjudication is None:
                raise ValidationError(
                    "constitutional replacement references an unknown atom"
                )
            if (
                current.get("type") not in CONSTITUTIONAL_ATOM_TYPES
                or successor.get("type") != current.get("type")
            ):
                raise ValidationError(
                    "constitutional predecessor and successor must have the same "
                    "constitutional atom type"
                )
            if current.get("lifecycle_state") != "active":
                raise ValidationError(
                    "constitutional predecessor must be the active head"
                )
            if successor.get("lifecycle_state") != "proposed":
                raise ValidationError(
                    "constitutional successor must remain proposed until replacement"
                )
            if int(current.get("version", 0)) != int(expected_current_version):
                raise CASConflict(
                    f"expected {current_ref} version {expected_current_version}, "
                    f"found {current['version']}"
                )
            if int(successor.get("version", 0)) != int(expected_successor_version):
                raise CASConflict(
                    f"expected {successor_ref} version {expected_successor_version}, "
                    f"found {successor['version']}"
                )
            if dict(current.get("scope") or {}) != dict(
                successor.get("scope") or {}
            ):
                raise ValidationError(
                    "constitutional predecessor and successor scopes must match"
                )
            predecessor_refs = {
                *[str(ref) for ref in successor.get("supersedes", []) if str(ref)],
                str((successor.get("payload") or {}).get("predecessor_ref") or ""),
            }
            if current_ref not in predecessor_refs:
                raise ValidationError(
                    "constitutional successor must explicitly identify its predecessor"
                )

            current_payload = dict(current.get("payload") or {})
            successor_payload = dict(successor.get("payload") or {})
            successor_comparable = dict(successor_payload)
            reported_diff = successor_comparable.pop("predecessor_diff", None)
            path_diff = self._payload_path_diff(
                current_payload, successor_comparable
            )
            if reported_diff != path_diff:
                raise ValidationError(
                    "constitutional successor predecessor_diff is not exact"
                )
            requirements = dict(
                current_payload.get("amendment_requirements") or {}
            )
            successor_requirements = dict(
                successor_payload.get("amendment_requirements") or {}
            )
            for control in (
                "amendment_requirements", "protected_fields", "auditability",
                "incident_disclosure", "evidence_requirements",
            ):
                if current_payload.get(control) and not successor_payload.get(control):
                    raise ValidationError(
                        f"one amendment cannot remove its own {control} control"
                    )
                if current_payload.get(control) and self._control_weakened(
                    current_payload[control], successor_payload.get(control)
                ):
                    raise ValidationError(
                        f"constitutional successor weakens its own {control} control"
                    )
            if requirements:
                if int(successor_requirements.get("required_confirmations") or 0) < int(
                    requirements.get("required_confirmations") or 0
                ):
                    raise ValidationError(
                        "constitutional successor weakens required confirmations"
                    )
                if int(successor_requirements.get("min_interval_seconds") or 0) < int(
                    requirements.get("min_interval_seconds") or 0
                ):
                    raise ValidationError(
                        "constitutional successor weakens its evidence interval"
                    )
                for flag in (
                    "materially_distinct_evidence", "advisory_critic_required"
                ):
                    if requirements.get(flag) is True and successor_requirements.get(flag) is not True:
                        raise ValidationError(
                            f"constitutional successor disables {flag}"
                        )
            if current_payload.get("amendability") == "immutable":
                if (
                    requirements.get("successor_permitted") is not True
                    or successor_payload.get("successor_classification")
                    != "successor_creation"
                ):
                    raise ValidationError(
                        "immutable guidance permits only an explicitly authorized "
                        "successor creation"
                    )
            else:
                self.assert_constitutional_amendment_policy(
                    current,
                    context,
                    changed_payload_fields={
                        str(item["path"]) for item in path_diff
                    },
                )

            if (
                adjudication.get("type") != "adjudication"
                or adjudication.get("lifecycle_state") != "active"
            ):
                raise ValidationError(
                    "adjudication_ref must reference an active adjudication"
                )
            adjudication_payload = dict(adjudication.get("payload") or {})
            if (
                str(adjudication_payload.get("subject_ref") or "") != successor_ref
                or adjudication_payload.get("outcome")
                not in POSITIVE_ADJUDICATION_OUTCOMES
                or str(adjudication_payload.get("claim_kind") or "")
                != "constitutional_replacement"
            ):
                raise ValidationError(
                    "replacement requires a positive constitutional_replacement "
                    "adjudication for the successor"
                )
            derived_risk = self._assert_adjudication_semantics(
                proposal=successor,
                adjudication_payload=adjudication_payload,
                predecessor=current,
            )
            self._assert_positive_ratification_threshold(
                risk_class=derived_risk,
                adjudication_payload=adjudication_payload,
            )
            ratifier = dict(adjudication_payload.get("ratifier") or {})
            if (
                ratifier.get("mode") != "self_ratification"
                or str(ratifier.get("identity_ref") or "") != identity_ref
            ):
                raise AccessDenied(
                    "authenticated identity must author its constitutional amendment"
                )
            if dict(successor.get("scope") or {}) != dict(
                adjudication.get("scope") or {}
            ):
                raise ValidationError(
                    "successor and amendment adjudication scopes must match"
                )
            covenant_refs = [
                str(ref)
                for ref in adjudication_payload.get("covenant_refs", [])
                if str(ref)
            ]
            if current_ref not in covenant_refs:
                raise ValidationError(
                    "constitutional amendment adjudication must cite the predecessor"
                )
            higher_refs = [
                str(ref)
                for ref in successor_payload.get("higher_governing_refs", [])
                if str(ref)
            ]
            for ref in higher_refs:
                guidance = self.store.get_atom(ref)
                if (
                    guidance is None
                    or guidance.get("type") not in CONSTITUTIONAL_ATOM_TYPES
                    or guidance.get("lifecycle_state") != "active"
                    or int((guidance.get("payload") or {}).get("precedence", 0))
                    <= int(current_payload.get("precedence", 0))
                    or not self.constitutional_scope_applies(
                        guidance.get("scope"), successor.get("scope")
                    )
                ):
                    raise ValidationError(
                        f"invalid higher-precedence constitutional guidance: {ref}"
                    )
            if current.get("type") == "covenant" and not higher_refs:
                raise ValidationError(
                    "covenant replacement must cite higher-precedence guidance"
                )
            if any(ref not in covenant_refs for ref in higher_refs):
                raise ValidationError(
                    "amendment adjudication must cite all higher governing guidance"
                )

            required_confirmations = max(
                int(requirements.get("required_confirmations", 0) or 0),
                3
                if current_payload.get("amendability") in {"entrenched", "immutable"}
                else 2,
            )
            min_interval_seconds = max(
                0, int(requirements.get("min_interval_seconds", 0) or 0)
            )
            if required_confirmations > 1 and min_interval_seconds <= 0:
                raise ValidationError(
                    "constitutional diachronic ratification requires a nonzero interval"
                )
            status = self._diachronic_status(
                subject_ref=successor_ref,
                identity_ref=identity_ref,
                required_confirmations=required_confirmations,
                min_interval_seconds=min_interval_seconds,
                require_distinct_evidence=bool(
                    requirements.get("materially_distinct_evidence")
                ),
            )
            if not status["threshold_reached"]:
                raise ValidationError(
                    "constitutional diachronic self-ratification threshold has "
                    "not been reached"
                )

            changed_at = utc_now()
            activated = dict(successor)
            activated_payload = dict(successor_payload)
            activated_payload["ratification"] = {
                "adjudication_ref": adjudication_ref,
                "ratifier_identity_ref": identity_ref,
                "ratified_at": changed_at,
                "covenant_refs": covenant_refs,
                "diachronic_status": status,
                "replaces_ref": current_ref,
            }
            activated["payload"] = activated_payload
            activated["lifecycle_state"] = "active"
            activated["version"] = int(successor["version"]) + 1
            activated["updated_at"] = changed_at
            activated["revision_history"] = list(
                successor.get("revision_history") or []
            )
            activated["revision_history"].append(
                {
                    "version": successor["version"],
                    "digest": digest(self._atom_projection(successor)),
                    "changed_at": changed_at,
                    "actor": actor,
                    "reason": "constitutional_successor_activated",
                    "adjudication_ref": adjudication_ref,
                }
            )
            activated = normalize_atom(
                self._attach_search_index(activated), require_id=True
            )

            retired = dict(current)
            retired["lifecycle_state"] = "superseded"
            retired["version"] = int(current["version"]) + 1
            retired["updated_at"] = changed_at
            retired["revision_history"] = list(
                current.get("revision_history") or []
            )
            retired["revision_history"].append(
                {
                    "version": current["version"],
                    "digest": digest(self._atom_projection(current)),
                    "changed_at": changed_at,
                    "actor": actor,
                    "reason": "constitutionally_replaced",
                    "successor_ref": successor_ref,
                    "adjudication_ref": adjudication_ref,
                }
            )
            retired = normalize_atom(
                self._attach_search_index(retired), require_id=True
            )
            projected_edges = [
                *self._intrinsic_edges_for_atom(activated),
                *self._intrinsic_edges_for_atom(retired),
                self._edge(
                    successor_ref,
                    current_ref,
                    "rel:supersedes",
                    dict(activated.get("scope") or {}),
                    derivation={
                        "kind": "constitutional_replacement",
                        "source_refs": [
                            successor_ref,
                            current_ref,
                            adjudication_ref,
                        ],
                    },
                ),
                self._edge(
                    adjudication_ref,
                    successor_ref,
                    "rel:adjudicates",
                    dict(activated.get("scope") or {}),
                    derivation={
                        "kind": "constitutional_replacement",
                        "source_refs": [adjudication_ref, successor_ref],
                    },
                ),
            ]
            projected_edges = list(
                {
                    str(edge["edge_id"]): edge for edge in projected_edges
                }.values()
            )
            event = self.store.append_event(
                conn,
                event_type="constitutional_record_replaced",
                actor=actor,
                payload={
                    **request_payload,
                    "diachronic_status": status,
                    "projected_atoms": [retired, activated],
                    "projected_edges": projected_edges,
                },
                target_refs=[
                    current_ref,
                    successor_ref,
                    adjudication_ref,
                    *covenant_refs,
                ],
                evidence_refs=list(
                    dict.fromkeys(
                        [
                            *current.get("evidence_refs", []),
                            *successor.get("evidence_refs", []),
                            *adjudication.get("evidence_refs", []),
                        ]
                    )
                ),
                idempotency_key=idempotency_key,
                expected_versions={
                    current_ref: int(expected_current_version),
                    successor_ref: int(expected_successor_version),
                },
                authorization_context=context,
            )
            self.store.replace_atom(conn, retired)
            self.store.replace_atom(conn, activated)
            for edge in projected_edges:
                self.store.upsert_edge(conn, edge)
            self.store.retire_packet_cache(conn)
            response = {
                "status": "replaced",
                "prior": retired,
                "atom": activated,
                "adjudication": adjudication,
                "diachronic_status": status,
                "edges": projected_edges,
                "event": event,
            }
            self._record_idempotency(
                conn,
                actor,
                idempotency_key,
                request_payload,
                event,
                response,
            )
            return response

    def analyze_provenance(
        self,
        *,
        atom_ref: str,
        max_depth: int = 64,
    ) -> dict[str, Any]:
        """Return root-level support structure, not merely direct source counts."""

        atom_ref = str(atom_ref or "").strip()
        root = self.store.get_atom(atom_ref)
        if root is None or root.get("deleted"):
            raise ValidationError(f"unknown atom: {atom_ref}")
        max_depth = max(1, min(int(max_depth), 512))
        evidence_by_ref = {
            str(item["evidence_id"]): item for item in self.store.list_evidence()
        }
        parents: dict[str, list[str]] = defaultdict(list)
        for atom in self.store.list_atoms():
            atom_id = str(atom["id"])
            payload = atom.get("payload") or {}
            for ref in payload.get("source_refs", []) or []:
                if self.store.get_atom(str(ref)) is not None:
                    parents[atom_id].append(str(ref))
        for edge in self.store.list_edges():
            if edge.get("relation") == "rel:derived_from" and not edge.get("deleted"):
                parents[str(edge["source_ref"])].append(str(edge["target_ref"]))
        parents = {
            key: list(dict.fromkeys(values)) for key, values in parents.items()
        }

        roots: dict[str, dict[str, Any]] = {}
        ancestry_paths: dict[str, list[list[str]]] = defaultdict(list)
        cycles: list[list[str]] = []

        def visit(current_ref: str, path: list[str], depth: int) -> None:
            if current_ref in path:
                cycles.append([*path[path.index(current_ref) :], current_ref])
                return
            atom = self.store.get_atom(current_ref)
            if atom is None or depth >= max_depth or not parents.get(current_ref):
                if atom is not None:
                    evidence_refs = list(atom.get("evidence_refs") or [])
                    if evidence_refs:
                        for evidence_ref in evidence_refs:
                            evidence = evidence_by_ref.get(str(evidence_ref))
                            payload = (
                                evidence.get("payload")
                                if isinstance(evidence, Mapping)
                                else {}
                            )
                            payload = payload if isinstance(payload, Mapping) else {}
                            group = str(
                                payload.get("independence_group")
                                or (evidence or {}).get("source_ref")
                                or evidence_ref
                            )
                            family = str(
                                payload.get("testimony_family")
                                or (evidence or {}).get("source_type")
                                or "unknown"
                            )
                            roots[str(evidence_ref)] = {
                                "root_ref": str(evidence_ref),
                                "kind": "evidence",
                                "independence_group": group,
                                "testimony_family": family,
                                "source_ref": (evidence or {}).get("source_ref"),
                            }
                            ancestry_paths[str(evidence_ref)].append(
                                [*path, current_ref, str(evidence_ref)]
                            )
                        return
                    roots[current_ref] = {
                        "root_ref": current_ref,
                        "kind": "atom",
                        "independence_group": current_ref,
                        "testimony_family": str(atom.get("type") or "unknown"),
                    }
                    ancestry_paths[current_ref].append([*path, current_ref])
                return
            for parent in parents.get(current_ref, []):
                visit(parent, [*path, current_ref], depth + 1)

        visit(atom_ref, [], 0)
        ancestor_counts = Counter(
            ref
            for paths in ancestry_paths.values()
            for path in paths
            for ref in set(path[1:-1])
        )
        common_ancestors = sorted(
            (
                {"atom_ref": ref, "root_path_count": count}
                for ref, count in ancestor_counts.items()
                if count > 1
            ),
            key=lambda item: (-int(item["root_path_count"]), str(item["atom_ref"])),
        )
        groups: dict[str, list[str]] = defaultdict(list)
        families: dict[str, list[str]] = defaultdict(list)
        for root_ref, item in roots.items():
            groups[str(item["independence_group"])].append(root_ref)
            families[str(item["testimony_family"])].append(root_ref)
        return {
            "status": "analyzed",
            "atom_ref": atom_ref,
            "revision": self.store.memory_revision(),
            "root_evidence": sorted(roots.values(), key=lambda item: item["root_ref"]),
            "independence_groups": {
                key: sorted(values) for key, values in sorted(groups.items())
            },
            "testimony_families": {
                key: sorted(values) for key, values in sorted(families.items())
            },
            "common_ancestors": common_ancestors,
            "ancestry_depth": max(
                (len(path) - 1 for paths in ancestry_paths.values() for path in paths),
                default=0,
            ),
            "circular_support": bool(cycles),
            "cycles": cycles,
            "self_descendant_support": any(
                atom_ref in cycle[1:] for cycle in cycles
            ),
        }

    def diachronic_ratification_status(
        self,
        *,
        subject_ref: str,
        identity_ref: str,
        required_confirmations: int = 2,
        min_interval_seconds: int = 0,
        require_distinct_evidence: bool = False,
    ) -> dict[str, Any]:
        return self._diachronic_status(
            subject_ref=subject_ref,
            identity_ref=identity_ref,
            required_confirmations=required_confirmations,
            min_interval_seconds=min_interval_seconds,
            require_distinct_evidence=require_distinct_evidence,
        )

    def _diachronic_status(
        self,
        *,
        subject_ref: str,
        identity_ref: str,
        required_confirmations: int,
        min_interval_seconds: int,
        require_distinct_evidence: bool = False,
    ) -> dict[str, Any]:
        qualifying = []
        for atom in self.store.list_atoms_filtered(
            types=["adjudication"], lifecycle_states=["active"]
        ):
            payload = atom.get("payload") or {}
            ratifier = payload.get("ratifier") or {}
            diachronic = payload.get("diachronic") or {}
            if (
                str(payload.get("subject_ref") or "") == str(subject_ref)
                and str(ratifier.get("identity_ref") or "") == str(identity_ref)
                and payload.get("outcome") in POSITIVE_ADJUDICATION_OUTCOMES
                and bool(diachronic.get("independent_reconstruction"))
            ):
                qualifying.append(atom)
        qualifying.sort(
            key=lambda atom: str(
                (atom.get("payload") or {}).get("reconstructed_at")
                or atom.get("observed_at")
                or atom.get("created_at")
                or ""
            )
        )
        evidence_sets = [
            {
                str(ref)
                for ref in ((atom.get("payload") or {}).get("diachronic") or {}).get(
                    "new_experience_refs", []
                )
                if str(ref)
            }
            for atom in qualifying
        ]
        seen_evidence: set[str] = set()
        evidence_novelty: list[bool] = []
        for evidence_refs in evidence_sets:
            novel = bool(evidence_refs - seen_evidence)
            evidence_novelty.append(bool(evidence_refs) and novel)
            seen_evidence.update(evidence_refs)
        materially_qualifying = (
            [atom for atom, novel in zip(qualifying, evidence_novelty) if novel]
            if require_distinct_evidence
            else qualifying
        )
        intervals = []
        for before, after in zip(materially_qualifying, materially_qualifying[1:]):
            before_raw = str((before.get("payload") or {}).get("reconstructed_at"))
            after_raw = str((after.get("payload") or {}).get("reconstructed_at"))
            try:
                seconds = (
                    datetime.fromisoformat(after_raw.replace("Z", "+00:00"))
                    - datetime.fromisoformat(before_raw.replace("Z", "+00:00"))
                ).total_seconds()
            except ValueError:
                seconds = -1
            intervals.append(seconds)
        interval_ok = all(
            seconds >= int(min_interval_seconds) for seconds in intervals
        )
        distinct_evidence_ok = (
            len(materially_qualifying) >= max(1, int(required_confirmations))
            if require_distinct_evidence
            else True
        )
        return {
            "status": "evaluated",
            "subject_ref": str(subject_ref),
            "identity_ref": str(identity_ref),
            "required_confirmations": max(1, int(required_confirmations)),
            "min_interval_seconds": max(0, int(min_interval_seconds)),
            "require_distinct_evidence": bool(require_distinct_evidence),
            "evidence_novelty": evidence_novelty,
            "distinct_evidence_ok": distinct_evidence_ok,
            "qualifying_confirmation_refs": [
                str(atom["id"]) for atom in materially_qualifying
            ],
            "confirmation_count": len(qualifying),
            "materially_distinct_confirmation_count": len(materially_qualifying),
            "intervals_seconds": intervals,
            "threshold_reached": (
                len(materially_qualifying) >= max(1, int(required_confirmations))
                and interval_ok
                and distinct_evidence_ok
            ),
        }
