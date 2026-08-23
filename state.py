import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TripState:
    """
    Structured memory — explicit facts the agent has gathered.

    Unlike messages[] (free-text chat history), this holds typed fields
    that are serialized into the system prompt before every LLM call.

    Context compression shrinks old messages[] entries but does NOT touch
    TripState — structured facts always survive in messages[0].

    Updated by agent._agent_loop() after each tool execution via apply_tool_result().
    """

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

    def to_json(self) -> str:
        """Serialized into messages[0] (system prompt) on every LLM call."""
        return json.dumps(self.to_dict(), indent=2)

    def apply_tool_result(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        """
        Update structured state after a tool runs.

        Called from agent._agent_loop() immediately after TOOL_REGISTRY[name](**args).
        The updated state is then injected into the next LLM request via _sync_system().
        """
        if tool_name in ("search_flights", "search_trains"):
            self.source, self.destination = args.get("source"), args.get("destination")
            self.transport_mode = "flight" if tool_name == "search_flights" else "train"
            self.transport_options = result if isinstance(result, list) else []
        elif tool_name == "search_hotels":
            if args.get("city") and not self.destination:
                self.destination = args["city"]
            self.hotel_options = result if isinstance(result, list) else []
        elif tool_name == "calculate_budget":
            self.nights = args.get("nights")
            self.budget_breakdown = result if isinstance(result, dict) else None
        elif tool_name == "finish_trip_planning":
            self.status = "complete"
