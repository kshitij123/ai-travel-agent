from memory import UserMemory
from state import TripState

# Base instructions — always the same. Sent as the first part of messages[0].
SYSTEM_PROMPT = """You are an AI travel planning assistant.
Your job is to help users plan trips using the available tools.

Rules:
1. Use tools whenever you need travel information.
2. Never invent flights, trains, hotels, prices or availability.
3. Use calculate_budget for arithmetic calculations.
4. Always respect the user's stated budget.
5. Compare available options before making recommendations.
6. Clearly explain why you recommend an option.
7. If required information is missing, ask the user for it.
8. Use the current trip state below as the source of truth for gathered facts.
9. Use remembered user profile facts when relevant (home city, transport style, budget tier).
10. Call remember_preference immediately when the user shares durable preferences (home city, budget, transport style, hotel needs, dietary restrictions). Do not wait until trip planning finishes.
"""


def build_system_prompt(state: TripState, memory: UserMemory) -> str:
    """
    Build the system message content sent to the LLM on every call.

    Called by agent._sync_system() before each LLM request.
    TripState is short-term structured memory for the active trip.
    UserMemory is long-term memory loaded from previous sessions.
    """
    parts = [SYSTEM_PROMPT.strip(), f"\nCurrent trip state:\n{state.to_json()}"]
    if not memory.is_empty():
        parts.append(f"\nRemembered user profile (from past sessions):\n{memory.to_json()}")
    return "\n".join(parts)


# Used by agent._summarize_messages() during context compression.
# This is a separate LLM call — not part of the main agent loop.
COMPRESS_SUMMARY_PROMPT = """Summarize the following conversation history for a travel planning agent.

Preserve all facts needed to continue planning:
- user preferences, budget, cities, dates, nights
- transport and hotel options discussed
- recommendations made and open questions

Be concise. Do not invent information."""
