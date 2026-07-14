"""TemporalService implementation for the AMOS service facade."""

from ._service_support import Any, datetime, timedelta, timezone


class TemporalService:
    def __init__(self) -> None:
        pass

    def _seconds_since(self, timestamp: Any) -> int | None:
        if not timestamp:
            return None
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))


    def _iso_before_seconds(self, seconds: int) -> str:
        return (
            datetime.now(timezone.utc) - timedelta(seconds=max(0, int(seconds)))
        ).isoformat().replace("+00:00", "Z")


    def _timestamp_elapsed(self, timestamp: Any) -> bool:
        if not timestamp:
            return False
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            return False
        return datetime.now(timezone.utc) >= parsed
