"""AMOS domain exceptions."""


class AmosError(Exception):
    """Base class for AMOS errors."""


class ValidationError(AmosError):
    """Raised when a memory object violates the AMOS schema contract."""


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
