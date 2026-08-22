import json
from typing import Any

from config import DATA_PATH


def load_data() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _match_route(items: list[dict], source: str, destination: str) -> list[dict]:
    source_key = source.lower()
    destination_key = destination.lower()
    return [
        item
        for item in items
        if item["from"].lower() == source_key and item["to"].lower() == destination_key
    ]


def search_flights(source: str, destination: str) -> list[dict]:
    data = load_data()
    return _match_route(data["flights"], source, destination)


def search_trains(source: str, destination: str) -> list[dict]:
    data = load_data()
    return _match_route(data["trains"], source, destination)


def search_hotels(city: str) -> list[dict]:
    data = load_data()
    city_key = city.lower()
    return [hotel for hotel in data["hotels"] if hotel["city"].lower() == city_key]


def calculate_budget(
    transport_cost: float,
    hotel_cost_per_night: float,
    nights: int,
) -> dict[str, float]:
    total_hotel_cost = hotel_cost_per_night * nights
    return {
        "transport_cost": transport_cost,
        "hotel_cost": total_hotel_cost,
        "total_cost": transport_cost + total_hotel_cost,
    }


TOOL_REGISTRY = {
    "search_flights": search_flights,
    "search_trains": search_trains,
    "search_hotels": search_hotels,
    "calculate_budget": calculate_budget,
}
