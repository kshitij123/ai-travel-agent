import json
import os
from typing import Any

from groq import BadRequestError, Groq

from config import (
    FINAL_PLAN_PROMPT,
    MAX_TOOL_RETRIES,
    MODEL,
    TOOL_TEMPERATURE,
)
from prompts import build_system_prompt
from schemas import TRAVEL_PLAN_SCHEMA
from state import TripState
from tool_definitions import TOOLS
from tools import TOOL_REGISTRY


def _tool_use_failed(error: BadRequestError) -> bool:
    body = error.body
    return isinstance(body, dict) and body.get("error", {}).get("code") == "tool_use_failed"


def _normalize_tool_name(name: str) -> str:
    return name.split("<|channel|>", maxsplit=1)[0]


class TravelAgent:
    def __init__(self) -> None:
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.trip_state = TripState()
        self.messages: list[dict[str, Any]] = []

    def _sync_system_message(self) -> None:
        system_message = {
            "role": "system",
            "content": build_system_prompt(self.trip_state),
        }
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = system_message
        else:
            self.messages.insert(0, system_message)

    def run(self) -> None:
        while True:
            user_request = input("\nYou: ")
            if user_request.lower() == "exit":
                print("Goodbye!")
                break

            self.messages.append({"role": "user", "content": user_request})
            self._sync_system_message()
            finished = self._run_agent_loop()

            if not finished:
                continue

            travel_plan = self._generate_structured_plan()
            if travel_plan is None:
                continue

            print("\nFinal structured response:")
            print(json.dumps(travel_plan, indent=2, ensure_ascii=False))

    def _create_tool_completion(self, temperature: float = TOOL_TEMPERATURE):
        last_error: BadRequestError | None = None

        for attempt in range(MAX_TOOL_RETRIES):
            try:
                return self.client.chat.completions.create(
                    model=MODEL,
                    messages=self.messages,
                    tools=TOOLS,
                    temperature=temperature,
                )
            except BadRequestError as error:
                if not _tool_use_failed(error) or attempt == MAX_TOOL_RETRIES - 1:
                    raise

                last_error = error
                temperature = max(temperature - 0.2, 0.2)
                print(
                    f"\nTool call failed, retrying with temperature {temperature}..."
                )

        if last_error is not None:
            raise last_error

        raise RuntimeError("Tool completion failed without an error.")

    def _run_agent_loop(self) -> bool:
        finished = False

        while True:
            self._sync_system_message()
            print("\nCalling LLM...")
            response = self._create_tool_completion()
            message = response.choices[0].message

            if not message.tool_calls:
                self.messages.append(message)
                print("\nAgent:")
                print(message.content)
                break

            self.messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = _normalize_tool_name(tool_call.function.name)
                arguments = json.loads(tool_call.function.arguments)

                print("\nTool requested:", tool_name)
                print("Arguments:", arguments)

                if tool_name == "finish_trip_planning":
                    self.trip_state.apply_tool_result(tool_name, arguments, None)
                    self._sync_system_message()
                    finished = True
                    print("\nAgent decided that trip planning is complete.")
                    break

                tool = TOOL_REGISTRY.get(tool_name)
                if tool is None:
                    raise ValueError(f"Unknown tool: {tool_name}")

                result = tool(**arguments)
                self.trip_state.apply_tool_result(tool_name, arguments, result)
                self._sync_system_message()
                print("\nTool result:")
                print(result)
                print("\n[State updated]")
                print(self.trip_state.to_prompt_block())

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

            if finished:
                break

        return finished

    def _generate_structured_plan(self) -> dict[str, Any] | None:
        print("\nGenerating structured response...")

        self._sync_system_message()
        final_messages = self.messages + [
            {"role": "user", "content": FINAL_PLAN_PROMPT}
        ]
        final_response = self.client.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "travel_plan",
                    "schema": TRAVEL_PLAN_SCHEMA,
                },
            },
        )
        final_message = final_response.choices[0].message

        try:
            return json.loads(final_message.content)
        except json.JSONDecodeError:
            print("\nCould not parse structured response.")
            print(final_message.content)
            return None
