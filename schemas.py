TRAVEL_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "nights": {"type": "integer"},
        "transport": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["type", "name", "price"],
        },
        "hotel": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "rating": {"type": "number"},
                "price_per_night": {"type": "number"},
            },
            "required": ["name", "rating", "price_per_night"],
        },
        "total_cost": {"type": "number"},
        "within_budget": {"type": "boolean"},
        "recommendation": {"type": "string"},
    },
    "required": [
        "destination",
        "nights",
        "transport",
        "hotel",
        "total_cost",
        "within_budget",
        "recommendation",
    ],
}
