from pathlib import Path

MODEL = "openai/gpt-oss-120b"
DATA_PATH = Path(__file__).resolve().parent / "data" / "travel_data.json"
TOOL_TEMPERATURE = 0.5
MAX_TOOL_RETRIES = 3
FINAL_PLAN_PROMPT = """
Now produce the final travel plan based only on
the information gathered during this conversation.

Return the response using the required travel plan schema.

Do not invent any information.
"""
