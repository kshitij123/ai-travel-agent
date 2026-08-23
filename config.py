from pathlib import Path

MODEL = "openai/gpt-oss-120b"
DATA_PATH = Path(__file__).resolve().parent / "data" / "travel_data.json"
MEMORY_PATH = Path(__file__).resolve().parent / "data" / "user_memory.json"
DOCUMENTS_DIR = Path(__file__).resolve().parent / "data" / "documents"
RAG_TOP_K = 5
RAG_MIN_SCORE = 0.05
TOOL_TEMPERATURE = 0.5
MAX_TOOL_RETRIES = 3

# ------------------------------------------------------------------
# Context window management
# ------------------------------------------------------------------
# Compression reduces token count by summarizing OLD messages while
# keeping RECENT messages verbatim. This saves cost and avoids
# hitting the model's context window limit.
#
# Formula: compress when messages > MIN_MESSAGES_FOR_COMPRESSION
# After compression: [system, summary, ...last KEEP_RECENT_MESSAGES...]
# ------------------------------------------------------------------

COMPRESS_EVERY_N_USER_TURNS = 5   # check compression trigger every N user inputs
MIN_MESSAGES_FOR_COMPRESSION = 10 # don't compress unless we have at least this many
KEEP_RECENT_MESSAGES = 6          # keep last N messages verbatim (should be enough for context)
COMPRESS_SUMMARY_TEMPERATURE = 0.3

FINAL_PLAN_PROMPT = """
Now produce the final travel plan based only on
the information gathered during this conversation.
Return the response using the required travel plan schema.
Do not invent any information.
"""

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
    "required": ["destination", "nights", "transport", "hotel", "total_cost", "within_budget", "recommendation"],
}
