import json
from typing import Any

from config import DATA_PATH


def _load_data() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_route(items: list[dict], src: str, dst: str) -> list[dict]:
    return [i for i in items if i["from"].lower() == src.lower() and i["to"].lower() == dst.lower()]


def search_flights(source: str, destination: str) -> list[dict]:
    return _match_route(_load_data()["flights"], source, destination)


def search_trains(source: str, destination: str) -> list[dict]:
    return _match_route(_load_data()["trains"], source, destination)


def search_hotels(city: str) -> list[dict]:
    return [h for h in _load_data()["hotels"] if h["city"].lower() == city.lower()]


def calculate_budget(transport_cost: float, hotel_cost_per_night: float, nights: int) -> dict[str, float]:
    hotel_cost = hotel_cost_per_night * nights
    return {"transport_cost": transport_cost, "hotel_cost": hotel_cost, "total_cost": transport_cost + hotel_cost}


def remember_preference(category: str, value: str) -> dict[str, str]:
    """
    Placeholder — actual persistence is handled by agent.py via UserMemory.
    The agent intercepts this tool and writes to long-term memory.
    """
    return {"status": "pending", "category": category, "value": value}


TOOL_REGISTRY = {
    "search_flights": search_flights,
    "search_trains": search_trains,
    "search_hotels": search_hotels,
    "calculate_budget": calculate_budget,
    "remember_preference": remember_preference,
}


def _tool(name: str, desc: str, props: dict, required: list) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}}


_ROUTE = {"source": {"type": "string", "description": "Departure city"}, "destination": {"type": "string", "description": "Arrival city"}}

TOOLS = [
    _tool("search_flights", "Search flights between two cities", _ROUTE, ["source", "destination"]),
    _tool("search_trains", "Search trains between two cities", _ROUTE, ["source", "destination"]),
    _tool("search_hotels", "Search hotels in a city", {"city": {"type": "string", "description": "City where the hotel is located"}}, ["city"]),
    _tool("calculate_budget", "Calculate total travel cost using transport cost, hotel cost per night and number of nights", {
        "transport_cost": {"type": "number", "description": "Total transport cost"},
        "hotel_cost_per_night": {"type": "number", "description": "Hotel cost per night"},
        "nights": {"type": "integer", "description": "Number of hotel nights"},
    }, ["transport_cost", "hotel_cost_per_night", "nights"]),
    _tool("finish_trip_planning", "Indicate that enough information has been gathered and the trip planning is complete. Call this only when all required information for the final travel plan is available.", {}, []),
    _tool(
        "remember_preference",
        "Save a durable user preference for future trips. Use when the user shares facts like home city, transport style, budget tier, hotel needs, or dietary restrictions.",
        {
            "category": {
                "type": "string",
                "enum": ["home_city", "preferred_transport", "budget_tier", "budget", "hotel_preference", "dietary", "note"],
                "description": "Type of preference to remember",
            },
            "value": {"type": "string", "description": "The preference value to store"},
        },
        ["category", "value"],
    ),
]
