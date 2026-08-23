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
        → if finish_trip_planning was called:
            → FINAL PLAN call (separate LLM request with JSON schema)
            → print structured travel plan

Key idea: Groq is stateless. Every call sends the full messages[] history.
TripState is re-injected into messages[0] before each call so the LLM
always sees structured facts at the top, not buried in chat text.
"""

import json
import os
from typing import Any

from groq import BadRequestError, Groq

from config import FINAL_PLAN_PROMPT, MAX_TOOL_RETRIES, MODEL, TOOL_TEMPERATURE, TRAVEL_PLAN_SCHEMA
from prompts import build_system_prompt
from state import TripState
from tools import TOOL_REGISTRY, TOOLS
import logger as log


def _is_tool_error(e: BadRequestError) -> bool:
    """Groq returns 400 with code 'tool_use_failed' when the model emits a malformed tool call."""
    return isinstance(e.body, dict) and e.body.get("error", {}).get("code") == "tool_use_failed"


def _clean_tool_name(name: str) -> str:
    """Some models append Harmony channel suffixes like 'search_trains<|channel|>commentary'."""
    return name.split("<|channel|>", maxsplit=1)[0]


class TravelAgent:
    def __init__(self) -> None:
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.state = TripState()          # structured facts, injected into system prompt
        self.messages: list[dict[str, Any]] = []  # full conversation history sent to LLM
        self.turn = 0                     # LLM call counter (for logging)

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
    # Outer loop — one iteration per user message
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

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
            # Save the reply to history and return to wait for next user input.
            if not msg.tool_calls:
                self.messages.append(msg)
                log.section("AGENT REPLY")
                print(log.format_agent_text(msg.content))
                return False

            # --- Branch B: LLM wants to call one or more tools ---
            # Save the assistant message (contains tool_call metadata) to history.
            self.messages.append(msg)

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
