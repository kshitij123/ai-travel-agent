def _function_tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_ROUTE_PARAMS = {
    "source": {"type": "string", "description": "Departure city"},
    "destination": {"type": "string", "description": "Arrival city"},
}

TOOLS = [
    _function_tool(
        "search_flights",
        "Search flights between two cities",
        _ROUTE_PARAMS,
        ["source", "destination"],
    ),
    _function_tool(
        "search_trains",
        "Search trains between two cities",
        _ROUTE_PARAMS,
        ["source", "destination"],
    ),
    _function_tool(
        "search_hotels",
        "Search hotels in a city",
        {"city": {"type": "string", "description": "City where the hotel is located"}},
        ["city"],
    ),
    _function_tool(
        "calculate_budget",
        (
            "Calculate total travel cost using "
            "transport cost, hotel cost per night "
            "and number of nights"
        ),
        {
            "transport_cost": {"type": "number", "description": "Total transport cost"},
            "hotel_cost_per_night": {
                "type": "number",
                "description": "Hotel cost per night",
            },
            "nights": {"type": "integer", "description": "Number of hotel nights"},
        },
        ["transport_cost", "hotel_cost_per_night", "nights"],
    ),
    _function_tool(
        "finish_trip_planning",
        (
            "Indicate that enough information has been "
            "gathered and the trip planning is complete. "
            "Call this only when all required information "
            "for the final travel plan is available."
        ),
        {},
        [],
    ),
]
