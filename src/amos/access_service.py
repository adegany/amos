"""AccessService implementation for the AMOS service facade."""

from ._service_support import (
    AccessDenied,
    Any,
    IdempotencyConflict,
    Mapping,
    digest,
    utc_now,
)


class AccessService:
    def __init__(self, store: Any):
        self.store = store

    def _mark_foreground_activity(self, actor: str | None = None) -> None:
        if actor and str(actor).startswith("svc:"):
            return
        self.store.mark_foreground_activity(utc_now())


    def _idempotency_hit(
        self,
        conn: Any,
        actor: str,
        idempotency_key: str | None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        payload_digest = digest(payload)
        existing = self.store.get_idempotency(conn, actor, idempotency_key)
        if existing is None:
            return None
        if existing["payload_digest"] != payload_digest:
            raise IdempotencyConflict(
                f"idempotency key reused with different payload: {idempotency_key}"
            )
        response = dict(existing["response"])
        # Storage cleanup deliberately compacts large idempotent responses, but
        # callers still need the canonical references created by the original
        # mutation. Recover them from the retained event for legacy compact
        # receipts that predate reference preservation in the compact payload.
        if response.get("storage_compacted") and not response.get(
            "evidence_refs"
        ):
            event_id = str(response.get("event_id") or "")
            event = self.store.get_event(event_id) if event_id else None
            event = event if isinstance(event, Mapping) else {}
            evidence_refs = [
                str(ref)
                for ref in event.get("evidence_refs") or ()
                if str(ref)
            ]
            if evidence_refs:
                response["evidence_refs"] = evidence_refs
        return response


    def _record_idempotency(
        self,
        conn: Any,
        actor: str,
        idempotency_key: str | None,
        payload: Mapping[str, Any],
        event: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        if not idempotency_key:
            return
        self.store.put_idempotency(
            conn,
            actor=actor,
            key=idempotency_key,
            payload_digest=digest(payload),
            event_id=event["event_id"],
            response=response,
        )


    def _assert_mutation_allowed(
        self,
        atom: Mapping[str, Any],
        *,
        actor: str,
        authorization_context: Mapping[str, Any] | None = None,
    ) -> None:
        if actor in {"system", "svc:memory_steward", "owner", "admin"}:
            return
        context = dict(authorization_context or {})
        policy = atom.get("access_policy", {})
        mutable_by = set(policy.get("mutable_by", ["owner"]))
        actor_roles = set(context.get("roles", []))
        if actor in mutable_by or actor_roles.intersection(mutable_by):
            pass
        else:
            raise AccessDenied(f"{actor} cannot mutate atom {atom['id']}")
        min_trust = int(policy.get("min_trust_level", 0))
        actor_trust = int(context.get("trust_level", 0))
        if actor_trust < min_trust:
            raise AccessDenied(
                f"{actor} trust_level {actor_trust} is below required {min_trust}"
            )
        required_capability = policy.get("requires_capability")
        capabilities = set(context.get("capabilities", []))
        if required_capability and required_capability not in capabilities:
            raise AccessDenied(
                f"{actor} lacks required capability {required_capability}"
            )
