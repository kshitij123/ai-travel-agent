import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TripState:
    """Explicit facts the agent has gathered — not buried in chat text."""

    source: str | None = None
    destination: str | None = None
    nights: int | None = None
    budget: float | None = None
    transport_mode: str | None = None
    transport_options: list[dict[str, Any]] = field(default_factory=list)
    hotel_options: list[dict[str, Any]] = field(default_factory=list)
    budget_breakdown: dict[str, float] | None = None
    status: str = "gathering"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_block(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def apply_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        if tool_name in ("search_flights", "search_trains"):
            self.source = arguments.get("source")
            self.destination = arguments.get("destination")
            self.transport_mode = "flight" if tool_name == "search_flights" else "train"
            self.transport_options = result if isinstance(result, list) else []
            return

        if tool_name == "search_hotels":
            city = arguments.get("city")
            if city and not self.destination:
                self.destination = city
            self.hotel_options = result if isinstance(result, list) else []
            return

        if tool_name == "calculate_budget":
            self.nights = arguments.get("nights")
            if isinstance(result, dict):
                self.budget_breakdown = result
            return

        if tool_name == "finish_trip_planning":
            self.status = "complete"
