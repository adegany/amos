"""Thread-local request deadlines shared by AMOS subsystem services."""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager

from .errors import RequestDeadlineExceeded

_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "amos_request_deadline", default=None
)
_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "amos_request_id", default=None
)
_STAGE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "amos_request_stage", default=None
)


@contextmanager
def request_context(
    *, deadline_epoch_seconds: float | None, request_id: str | None
) -> Iterator[None]:
    deadline_token = _DEADLINE.set(deadline_epoch_seconds)
    request_token = _REQUEST_ID.set(str(request_id or "") or None)
    stage_token = _STAGE.set(None)
    try:
        yield
    finally:
        _STAGE.reset(stage_token)
        _REQUEST_ID.reset(request_token)
        _DEADLINE.reset(deadline_token)


def remaining_seconds() -> float | None:
    deadline = _DEADLINE.get()
    if deadline is None:
        return None
    return max(0.0, float(deadline) - time.time())


def current_stage() -> str | None:
    """Return the last typed cooperative stage entered by this request."""

    return _STAGE.get()


def check_deadline(stage: str, *, reserve_seconds: float = 0.0) -> None:
    normalized_stage = str(stage or "unknown")
    _STAGE.set(normalized_stage)
    remaining = remaining_seconds()
    if remaining is not None and remaining <= max(0.0, float(reserve_seconds)):
        raise RequestDeadlineExceeded(
            normalized_stage,
            request_id=_REQUEST_ID.get(),
        )
