"""
Long-term memory — facts that persist across sessions.

Short-term memory (messages[], TripState, compression) lives only for the
current run. Long-term memory stores:

  - User preferences (home city, transport style, budget tier, notes)
  - Completed trip history (destination, transport, hotel, cost)

Loaded at agent startup, injected into the system prompt, and saved whenever
preferences change during the conversation (not only at trip end).
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from state import TripState


def infer_budget_tier(total_cost: float, nights: int) -> str:
    per_night = total_cost / max(nights, 1)
    if per_night < 3000:
        return "budget"
    if per_night < 8000:
        return "mid-range"
    return "luxury"


@dataclass
class TripRecord:
    destination: str
    nights: int
    transport_type: str
    transport_name: str
    hotel_name: str
    total_cost: float
    within_budget: bool
    completed_at: str

    @classmethod
    def from_plan(cls, plan: dict[str, Any]) -> "TripRecord":
        transport = plan.get("transport", {})
        hotel = plan.get("hotel", {})
        return cls(
            destination=plan.get("destination", "unknown"),
            nights=int(plan.get("nights", 0)),
            transport_type=transport.get("type", "unknown"),
            transport_name=transport.get("name", "unknown"),
            hotel_name=hotel.get("name", "unknown"),
            total_cost=float(plan.get("total_cost", 0)),
            within_budget=bool(plan.get("within_budget", False)),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass
class UserMemory:
    """
    Durable user profile and trip history.

    Unlike TripState (current session only), this survives restarts.
    The agent reads it at startup and updates it as preferences are learned,
    syncing immediately to disk after each change.
    """

    home_city: str | None = None
    preferred_transport: str | None = None
    budget_tier: str | None = None  # e.g. "budget", "mid-range", "luxury"
    typical_budget: float | None = None
    hotel_preferences: list[str] = field(default_factory=list)
    dietary_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    past_trips: list[TripRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def is_empty(self) -> bool:
        return not any([
            self.home_city,
            self.preferred_transport,
            self.budget_tier,
            self.typical_budget,
            self.hotel_preferences,
            self.dietary_notes,
            self.notes,
            self.past_trips,
        ])

    @classmethod
    def load(cls, path: Path) -> "UserMemory":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        trips = [TripRecord(**t) for t in raw.pop("past_trips", [])]
        return cls(past_trips=trips, **raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def remember_preference(self, category: str, value: str) -> dict[str, str]:
        """Apply an explicit preference update from the remember_preference tool."""
        category = category.strip().lower()
        value = value.strip()
        if not value:
            return {"status": "ignored", "reason": "empty value"}

        if category == "home_city":
            self.home_city = value
        elif category == "preferred_transport":
            self.preferred_transport = value
        elif category == "budget_tier":
            self.budget_tier = value
        elif category == "budget":
            try:
                self.typical_budget = float(value.replace(",", "").replace("₹", "").strip())
            except ValueError:
                return {"status": "error", "reason": f"invalid budget: {value}"}
        elif category == "hotel_preference":
            if value not in self.hotel_preferences:
                self.hotel_preferences.append(value)
        elif category == "dietary":
            if value not in self.dietary_notes:
                self.dietary_notes.append(value)
        elif category == "note":
            if value not in self.notes:
                self.notes.append(value)
        else:
            return {"status": "error", "reason": f"unknown category: {category}"}

        return {"status": "saved", "category": category, "value": value}

    def sync_basic_preferences(self, state: "TripState") -> list[str]:
        """
        Copy durable preference signals from the active trip state.

        Called after TripState updates so basic preferences are persisted
        during the conversation, not only when a trip completes.
        """
        updated: list[str] = []

        if state.source and self.home_city is None:
            self.home_city = state.source
            updated.append("home_city")

        if state.transport_mode and state.transport_mode != self.preferred_transport:
            self.preferred_transport = state.transport_mode
            updated.append("preferred_transport")

        if state.budget is not None and state.budget != self.typical_budget:
            self.typical_budget = state.budget
            updated.append("typical_budget")
            tier = infer_budget_tier(state.budget, state.nights or 1)
            if tier != self.budget_tier:
                self.budget_tier = tier
                updated.append("budget_tier")

        return updated

    def record_completed_trip(self, plan: dict[str, Any]) -> TripRecord:
        """Append a completed trip and infer durable preferences from it."""
        record = TripRecord.from_plan(plan)
        self.past_trips.append(record)

        transport_type = plan.get("transport", {}).get("type")
        if transport_type:
            self.preferred_transport = transport_type

        total_cost = float(plan.get("total_cost", 0))
        nights = int(plan.get("nights", 1) or 1)
        self.typical_budget = total_cost
        self.budget_tier = infer_budget_tier(total_cost, nights)

        return record
