"""CapacityService implementation for the AMOS service facade."""

from ._service_support import Any, Mapping, ValidationError, canonical_json, json


class CapacityService:
    def __init__(self, store: Any):
        self.store = store

    def configure_capacity_budget(
        self,
        *,
        hard_capacity_bytes: int,
        warning_ratio: float = 0.70,
        critical_ratio: float = 0.90,
    ) -> dict[str, Any]:
        if hard_capacity_bytes <= 0:
            raise ValidationError("hard_capacity_bytes must be positive")
        if not 0 < warning_ratio <= critical_ratio <= 1:
            raise ValidationError("capacity ratios must satisfy 0 < warning <= critical <= 1")
        budget = {
            "hard_capacity_bytes": int(hard_capacity_bytes),
            "warning_ratio": float(warning_ratio),
            "critical_ratio": float(critical_ratio),
        }
        self.store.set_meta("capacity_budget", canonical_json(budget))
        return {"status": "configured", "capacity_budget": budget}


    def _capacity_budget(self) -> dict[str, Any]:
        raw = self.store.get_meta("capacity_budget")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(data)


    def _capacity_pressure_mode(
        self,
        *,
        size_bytes: int | None = None,
        budget: Mapping[str, Any] | None = None,
    ) -> str:
        budget = dict(budget if budget is not None else self._capacity_budget())
        if size_bytes is None:
            path = self.store.path
            size_bytes = path.stat().st_size if path.exists() and str(path) != ":memory:" else 0
        hard_limit = int(budget.get("hard_capacity_bytes", 0) or 0)
        if hard_limit <= 0:
            return "green"
        ratio = size_bytes / hard_limit
        if ratio >= float(budget.get("critical_ratio", 0.90)):
            return "red"
        if ratio >= float(budget.get("warning_ratio", 0.70)):
            return "orange"
        return "green"
