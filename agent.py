"""
Travel Agent — LLM call/response flow
=====================================

High-level loop:

    User input
        → append to messages[]
        → inject TripState into system prompt
        → AGENT LOOP (may call LLM multiple times before returning)
            → build request JSON
            → call Groq API
            → handle response (text OR tool calls)
            → if tool calls: execute tools, append results, loop again
            → if text: show reply, wait for next user input
        → every 5 user turns: COMPRESS old messages[] into a short summary
        → if finish_trip_planning was called:
            → FINAL PLAN call (separate LLM request with JSON schema)
            → print structured travel plan

Key idea: Groq is stateless. Every call sends the full messages[] history.
TripState is re-injected into messages[0] before each call so the LLM
always sees structured facts at the top, not buried in chat text.

Context compression (every 5 user turns, if enough messages):
    messages[] grows with every user message, assistant reply, and tool result.
    Long histories cost more tokens and can exceed the context window.
    Compression replaces OLD messages with one short summary message,
    while keeping the last KEEP_RECENT_MESSAGES messages verbatim.
    TripState is NOT affected — structured facts stay in messages[0].

    Only runs if len(messages) >= MIN_MESSAGES_FOR_COMPRESSION to avoid
    pointless compression when history is still short.
"""

import json
import os
from typing import Any

from groq import BadRequestError, Groq

from config import (
    COMPRESS_EVERY_N_USER_TURNS,
    COMPRESS_SUMMARY_TEMPERATURE,
    FINAL_PLAN_PROMPT,
    KEEP_RECENT_MESSAGES,
    MIN_MESSAGES_FOR_COMPRESSION,
    MAX_TOOL_RETRIES,
    MODEL,
    TOOL_TEMPERATURE,
    TRAVEL_PLAN_SCHEMA,
)
from prompts import COMPRESS_SUMMARY_PROMPT, build_system_prompt
from state import TripState
from tools import TOOL_REGISTRY, TOOLS
import logger as log

COMPRESSED_HISTORY_PREFIX = "[Earlier conversation summary]"


def _is_tool_error(e: BadRequestError) -> bool:
    """Groq returns 400 with code 'tool_use_failed' when the model emits a malformed tool call."""
    return isinstance(e.body, dict) and e.body.get("error", {}).get("code") == "tool_use_failed"


def _clean_tool_name(name: str) -> str:
    """Some models append Harmony channel suffixes like 'search_trains<|channel|>commentary'."""
    return name.split("<|channel|>", maxsplit=1)[0]


def _clean_assistant_message(msg: Any) -> dict[str, Any]:
    """
    Convert a Groq message object to a clean dict with only essential fields.
    
    Groq responses include extra fields like 'reasoning', 'annotations',
    'function_call', etc. that bloat every subsequent request.
    We only need: role, content, tool_calls.
    
    IMPORTANT: 'content' must ALWAYS be present for assistant messages,
    even if null. Groq/OpenAI API requires it when tool_calls are present.
    
    This is called when appending assistant messages to self.messages[].
    """
    if isinstance(msg, dict):
        # Always include content (can be None, but must exist)
        clean: dict[str, Any] = {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content"),  # None if missing
        }
        if msg.get("tool_calls"):
            clean["tool_calls"] = msg["tool_calls"]
        return clean
    
    # Groq message object — always include content field
    clean: dict[str, Any] = {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None),  # None if missing
    }
    if getattr(msg, "tool_calls", None):
        # Clean tool_calls to only keep id, type, function.name, function.arguments
        clean["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return clean


class TravelAgent:
    def __init__(self) -> None:
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.state = TripState()          # structured facts, injected into system prompt
        self.messages: list[dict[str, Any]] = []  # full conversation history sent to LLM
        self.turn = 0                     # LLM call counter (for logging)
        self.user_turn = 0                # user input counter (triggers compression check)

    # ------------------------------------------------------------------
    # Memory helpers — run before every LLM call
    # ------------------------------------------------------------------

    def _sync_system(self) -> None:
        """
        Rebuild messages[0] with the latest TripState.

        The system message always contains:
          - base instructions (prompts.py)
          - current TripState as JSON

        This is called before every LLM request so the model always sees
        up-to-date structured facts without re-reading the entire chat.
        """
        msg = {"role": "system", "content": build_system_prompt(self.state)}
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = msg   # update existing system message
        else:
            self.messages.insert(0, msg)  # first call — insert at index 0

    def _request(self, temp: float = TOOL_TEMPERATURE) -> dict[str, Any]:
        """
        Build the JSON payload for a tool-calling LLM request.

        Shape sent to Groq:
        {
          "model": "...",
          "messages": [...],   ← full history including system + user + assistant + tool
          "tools": [...],      ← tells LLM which functions it can call
          "temperature": 0.5
        }
        """
        return {"model": MODEL, "messages": self.messages, "tools": TOOLS, "temperature": temp}

    # ------------------------------------------------------------------
    # Context window management — compress old messages
    # ------------------------------------------------------------------
    #
    # WHY COMPRESS?
    #   Every LLM call sends the FULL messages[] array. As conversation grows:
    #     - Token cost increases (you pay per token)
    #     - Latency increases (more tokens to process)
    #     - Risk hitting context window limit (model can't read everything)
    #
    # HOW IT WORKS:
    #   1. Check if we have enough messages (MIN_MESSAGES_FOR_COMPRESSION)
    #   2. Split: [system] + [old messages to compress] + [recent messages to keep]
    #   3. Call LLM to summarize the old messages into one short text
    #   4. Replace old messages with the summary
    #   5. Result: shorter messages[] = cheaper/faster future requests
    #
    # WHAT SURVIVES COMPRESSION:
    #   - TripState in messages[0] (always rebuilt fresh anyway)
    #   - Last KEEP_RECENT_MESSAGES (full detail for immediate context)
    #   - Summary of older conversation (key facts preserved)
    #
    # ------------------------------------------------------------------

    def _summarize_messages(self, messages_to_compress: list[dict]) -> str:
        """
        Call LLM to condense old messages into a short summary.

        Input: list of message dicts to summarize
        Output: short text summary preserving key facts

        This is NOT an agent turn — no tools, just summarization.
        """
        self.turn += 1
        req = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": COMPRESS_SUMMARY_PROMPT},
                {"role": "user", "content": json.dumps(messages_to_compress, ensure_ascii=False)},
            ],
            "temperature": COMPRESS_SUMMARY_TEMPERATURE,
        }
        log.log_request(self.turn, req)
        msg = self.client.chat.completions.create(**req).choices[0].message
        log.log_response(self.turn, msg)
        return msg.content or ""

    def _compress_history(self) -> None:
        """
        Replace old messages with one summary, keeping recent messages intact.

        Example with MIN=10, KEEP=6:
        
        Before (12 messages):
          [system, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11]
                   └─── compress these 5 ───┘  └── keep these 6 ──┘
        
        After (8 messages):
          [system, summary, m6, m7, m8, m9, m10, m11]
        
        Token savings: 5 verbose messages → 1 short summary
        """
        total = len(self.messages)
        
        # Guard: don't compress if not enough messages
        if total < MIN_MESSAGES_FOR_COMPRESSION:
            log.subsection("Compression skipped")
            print(f"{log.INDENT}Only {total} messages (need {MIN_MESSAGES_FOR_COMPRESSION}+)")
            return
        
        # Guard: need at least 1 old message to compress after keeping recent
        # messages[0] is system, so old = messages[1 : -KEEP_RECENT_MESSAGES]
        if total <= 1 + KEEP_RECENT_MESSAGES:
            log.subsection("Compression skipped")
            print(f"{log.INDENT}Not enough old messages to compress")
            return

        # Split messages
        recent = self.messages[-KEEP_RECENT_MESSAGES:]
        to_compress = self.messages[1:-KEEP_RECENT_MESSAGES]  # skip system at [0]
        
        if len(to_compress) < 2:
            log.subsection("Compression skipped")
            print(f"{log.INDENT}Only {len(to_compress)} old message(s) — not worth compressing")
            return

        before_count = total
        summary = self._summarize_messages(to_compress)

        # Rebuild: system will be added by _sync_system(), then summary, then recent
        self.messages = [
            {"role": "assistant", "content": f"{COMPRESSED_HISTORY_PREFIX}\n{summary}"},
            *recent,
        ]
        self._sync_system()  # adds fresh system message at [0]

        log.log_compression(before_count, len(self.messages), summary)

    def _maybe_compress_history(self) -> None:
        """
        Run compression check every COMPRESS_EVERY_N_USER_TURNS user inputs.
        
        Called at the END of each user turn (after agent loop completes).
        Compression only actually runs if we have enough messages.
        """
        if self.user_turn > 0 and self.user_turn % COMPRESS_EVERY_N_USER_TURNS == 0:
            self._compress_history()

    # ------------------------------------------------------------------
    # Outer loop — one iteration per user message
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            self.user_turn += 1

            # Step 1: store user message in conversation history
            log.log_user(user_input)
            self.messages.append({"role": "user", "content": user_input})
            self._sync_system()

            # Step 2: run agent loop (may call LLM 1..N times for tool use)
            # Returns True only when LLM called finish_trip_planning
            if self._agent_loop():
                # Step 3: one final LLM call to produce structured JSON plan
                if plan := self._final_plan():
                    log.log_final_plan(plan)

            # Step 4: periodically compress old chat history to save tokens
            self._maybe_compress_history()

    # ------------------------------------------------------------------
    # LLM call with retry — used during the agent loop
    # ------------------------------------------------------------------

    def _call_llm(self, req: dict[str, Any]):
        """
        Send request to Groq and return the raw completion response.

        On 'tool_use_failed' (malformed tool call from model), retry with
        lower temperature up to MAX_TOOL_RETRIES times.
        """
        temp = req["temperature"]
        for attempt in range(MAX_TOOL_RETRIES):
            req["temperature"] = temp
            try:
                return self.client.chat.completions.create(**req)
            except BadRequestError as e:
                if not _is_tool_error(e) or attempt == MAX_TOOL_RETRIES - 1:
                    raise
                temp = max(temp - 0.2, 0.2)
                log.log_retry(temp)
        raise RuntimeError("Tool completion failed")

    # ------------------------------------------------------------------
    # Agent loop — ReAct pattern: Reason → Act → Observe → repeat
    # ------------------------------------------------------------------

    def _agent_loop(self) -> bool:
        """
        Inner loop for one user turn. Calls LLM repeatedly until either:
          A) LLM replies with text (no tool calls) → return False, wait for user
          B) LLM calls finish_trip_planning → return True, trigger final plan

        Flow per iteration:
          1. sync system prompt (inject latest TripState)
          2. build + log request JSON
          3. call LLM
          4. log response
          5. branch on response type (see below)
        """
        while True:
            # --- Before each LLM call: refresh system message with latest state ---
            self._sync_system()
            self.turn += 1
            req = self._request()
            log.log_request(self.turn, req)

            # --- LLM call: send full messages[] + tool definitions ---
            msg = self._call_llm(req).choices[0].message
            log.log_response(self.turn, msg)

            # --- Branch A: LLM replied with text (no tools) ---
            # This means the agent is asking a question or giving a recommendation.
            # Clean the message (strip extra Groq fields) and save to history.
            if not msg.tool_calls:
                self.messages.append(_clean_assistant_message(msg))
                log.section("AGENT REPLY")
                print(log.format_agent_text(msg.content))
                return False

            # --- Branch B: LLM wants to call one or more tools ---
            # Clean and save the assistant message (contains tool_call metadata).
            self.messages.append(_clean_assistant_message(msg))

            for tc in msg.tool_calls:
                name = _clean_tool_name(tc.function.name)
                args = json.loads(tc.function.arguments)

                # --- Branch B1: planning complete signal ---
                # LLM decided it has enough info. No Python tool to run.
                # Update state, return True to trigger _final_plan().
                if name == "finish_trip_planning":
                    self.state.apply_tool_result(name, args, None)
                    log.subsection("Planning complete")
                    print(f"{log.INDENT}Agent decided trip planning is complete.")
                    log.log_state(self.state)
                    return True

                # --- Branch B2: normal tool execution ---
                # 1. Run Python function
                # 2. Update TripState with result
                # 3. Re-sync system prompt (state changed)
                # 4. Append tool result to messages[] so LLM can read it next turn
                result = TOOL_REGISTRY[name](**args)
                self.state.apply_tool_result(name, args, result)
                self._sync_system()
                log.log_tool(name, args, result)
                log.log_state(self.state)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,       # links back to assistant's tool_call
                    "content": json.dumps(result),
                })

            # Loop continues → LLM will process tool results on next iteration

    # ------------------------------------------------------------------
    # Final plan — separate LLM call with JSON schema enforcement
    # ------------------------------------------------------------------

    def _final_plan(self) -> dict[str, Any] | None:
        """
        Called only after finish_trip_planning.

        Differences from the agent loop call:
          - No tools parameter (LLM must return JSON, not call functions)
          - response_format enforces TRAVEL_PLAN_SCHEMA
          - Adds a one-off user message (FINAL_PLAN_PROMPT) — NOT saved to messages[]

        The extra user message is temporary; it tells the LLM to produce
        the final structured output without polluting conversation history.
        """
        log.section("GENERATING STRUCTURED PLAN")
        self._sync_system()
        self.turn += 1

        req = {
            "model": MODEL,
            # Full history + temporary instruction (not appended to self.messages)
            "messages": self.messages + [{"role": "user", "content": FINAL_PLAN_PROMPT}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "travel_plan", "schema": TRAVEL_PLAN_SCHEMA},
            },
        }
        log.log_request(self.turn, req)

        msg = self.client.chat.completions.create(**req).choices[0].message
        log.log_response(self.turn, msg)

        try:
            return json.loads(msg.content)
        except json.JSONDecodeError:
            log.subsection("Parse error")
            print(f"{log.INDENT}Could not parse structured response.\n{msg.content}")
            return None
