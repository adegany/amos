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
