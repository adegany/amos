"""AMOS domain exceptions."""


class AmosError(Exception):
    """Base class for AMOS errors."""


class ValidationError(AmosError):
    """Raised when a memory object violates the AMOS schema contract."""


class CognitiveWorkspaceBudgetExceeded(ValidationError):
    """A workspace's non-sheddable projection exceeds the caller's budget."""

    def __init__(
        self,
        *,
        limit_bytes: int,
        limit_tokens: int | None,
        limit_items: int | None,
        used_bytes: int,
        estimated_tokens: int,
        used_items: int,
    ) -> None:
        self.budget = {
            "limit_bytes": int(limit_bytes),
            "limit_tokens": (
                int(limit_tokens) if limit_tokens is not None else None
            ),
            "limit_items": (
                int(limit_items) if limit_items is not None else None
            ),
            "used_bytes": int(used_bytes),
            "estimated_tokens": int(estimated_tokens),
            "used_items": int(used_items),
        }
        self.minimum_budget = {
            "bytes": int(used_bytes),
            "tokens": int(estimated_tokens),
            "items": int(used_items),
        }
        self.exceeded_dimensions = [
            *(["bytes"] if int(used_bytes) > int(limit_bytes) else []),
            *(
                ["items"]
                if limit_items is not None
                and int(used_items) > int(limit_items)
                else []
            ),
        ]
        super().__init__(
            "token_or_byte_budget is too small for protected cognitive "
            "workspace context"
        )


class IdempotencyConflict(AmosError):
    """Raised when an idempotency key is reused with a different payload."""


class CASConflict(AmosError):
    """Raised when a compare-and-swap version check fails."""


class AccessDenied(AmosError):
    """Raised when the caller is not allowed to perform the requested action."""


class StaleFrameError(AmosError):
    """Raised when a reasoning frame no longer matches canonical memory."""

    def __init__(self, expected_revision, current_revision):
        self.expected_revision = dict(expected_revision)
        self.current_revision = dict(current_revision)
        super().__init__(
            "reasoning frame revision is stale: "
            f"expected {self.expected_revision!r}, current {self.current_revision!r}"
        )
