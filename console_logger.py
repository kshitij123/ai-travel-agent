import json
import re
from typing import Any

from state import TripState

WIDTH = 88
INDENT = "  "


def section(title: str) -> None:
    bar = "═" * WIDTH
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title: str) -> None:
    print(f"\n── {title} " + "─" * max(1, WIDTH - len(title) - 5))


def _format_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return f"{INDENT}(no data)"

    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in str_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _row(cells: list[str]) -> str:
        padded = [cells[i].ljust(widths[i]) for i in range(len(headers))]
        return f"{INDENT}│ " + " │ ".join(padded) + " │"

    divider = f"{INDENT}├─" + "─┼─".join("─" * width for width in widths) + "─┤"
    top = f"{INDENT}┌─" + "─┬─".join("─" * width for width in widths) + "─┐"
    bottom = f"{INDENT}└─" + "─┴─".join("─" * width for width in widths) + "─┘"

    lines = [top, _row(headers), divider]
    lines.extend(_row(row) for row in str_rows)
    lines.append(bottom)
    return "\n".join(lines)


def _format_price(amount: Any) -> str:
    if isinstance(amount, (int, float)):
        return f"₹{amount:,.0f}"
    return str(amount)


def format_transport_options(options: list[dict[str, Any]], mode: str) -> str:
    carrier_label = "Airline" if mode == "flight" else "Train"
    rows = [
        [
            option.get("id", ""),
            f"{option.get('from', '')} → {option.get('to', '')}",
            option.get("airline") or option.get("train", ""),
            option.get("departure", ""),
            option.get("date", ""),
            _format_price(option.get("price")),
        ]
        for option in options
    ]
    return _format_table(
        ["ID", "Route", carrier_label, "Depart", "Date", "Price"],
        rows,
    )


def format_hotel_options(options: list[dict[str, Any]]) -> str:
    rows = [
        [
            option.get("id", ""),
            option.get("name", ""),
            option.get("city", ""),
            option.get("rating", ""),
            _format_price(option.get("price_per_night")),
        ]
        for option in options
    ]
    return _format_table(["ID", "Hotel", "City", "Rating", "Per Night"], rows)


def format_budget_breakdown(breakdown: dict[str, float]) -> str:
    rows = [
        ["Transport", _format_price(breakdown.get("transport_cost"))],
        ["Hotel", _format_price(breakdown.get("hotel_cost"))],
        ["Total", _format_price(breakdown.get("total_cost"))],
    ]
    return _format_table(["Item", "Amount"], rows)


def format_trip_state(state: TripState) -> str:
    summary_rows = [
        ["Source", state.source or "—"],
        ["Destination", state.destination or "—"],
        ["Nights", state.nights if state.nights is not None else "—"],
        ["Budget", _format_price(state.budget) if state.budget is not None else "—"],
        ["Transport mode", state.transport_mode or "—"],
        ["Status", state.status],
    ]
    parts = ["Summary", _format_table(["Field", "Value"], summary_rows)]

    if state.transport_options:
        parts.extend(
            [
                "",
                f"Transport options ({state.transport_mode or 'unknown'})",
                format_transport_options(
                    state.transport_options,
                    state.transport_mode or "train",
                ),
            ]
        )

    if state.hotel_options:
        parts.extend(["", "Hotel options", format_hotel_options(state.hotel_options)])

    if state.budget_breakdown:
        parts.extend(["", "Budget breakdown", format_budget_breakdown(state.budget_breakdown)])

    return "\n".join(parts)


def format_tool_result(tool_name: str, result: Any) -> str:
    if tool_name in ("search_flights", "search_trains") and isinstance(result, list):
        mode = "flight" if tool_name == "search_flights" else "train"
        return format_transport_options(result, mode)

    if tool_name == "search_hotels" and isinstance(result, list):
        return format_hotel_options(result)

    if tool_name == "calculate_budget" and isinstance(result, dict):
        return format_budget_breakdown(result)

    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2, ensure_ascii=False)

    return str(result)


def _markdown_table_to_ascii(table_lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in table_lines:
        stripped = line.strip()
        if re.fullmatch(r"\|[-| :]+\|", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return "\n".join(table_lines)

    return _format_table(rows[0], rows[1:])


def format_agent_text(content: str | None) -> str:
    if not content:
        return f"{INDENT}(empty response)"

    lines = content.splitlines()
    output: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            output.append(_markdown_table_to_ascii(table_lines))
            continue

        output.append(line)
        index += 1

    return "\n".join(output)


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "?"))
    return str(getattr(message, "role", "?"))


def _message_content(message: Any) -> str | None:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return content if content else None


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        return message.get("tool_calls") or []
    return getattr(message, "tool_calls", None) or []


def _format_message_for_log(message: Any, index: int) -> str:
    role = _message_role(message).upper()
    header = f"{INDENT}[{index}] {role}"

    if role == "SYSTEM":
        content = _message_content(message) or ""
        preview = content[:500] + ("…" if len(content) > 500 else "")
        return f"{header}\n{INDENT}  {preview.replace(chr(10), chr(10) + INDENT + '  ')}"

    if role == "USER":
        content = _message_content(message) or ""
        return f"{header}\n{INDENT}  {content}"

    if role == "ASSISTANT":
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            lines = [header, f"{INDENT}  (tool calls)"]
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                else:
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments
                lines.append(f"{INDENT}  • {name}({arguments})")
            content = _message_content(message)
            if content:
                lines.append(f"{INDENT}  text: {content}")
            return "\n".join(lines)

        content = _message_content(message) or ""
        return f"{header}\n{format_agent_text(content)}"

    if role == "TOOL":
        if isinstance(message, dict):
            content = message.get("content", "")
            tool_call_id = message.get("tool_call_id", "")
        else:
            content = getattr(message, "content", "")
            tool_call_id = getattr(message, "tool_call_id", "")
        preview = content[:300] + ("…" if len(content) > 300 else "")
        return f"{header}  id={tool_call_id}\n{INDENT}  {preview}"

    content = _message_content(message) or str(message)
    return f"{header}\n{INDENT}  {content}"


def log_llm_input(turn: int, messages: list[Any], model: str) -> None:
    section(f"LLM INPUT — Turn {turn}")
    print(f"{INDENT}Model: {model}")
    print(f"{INDENT}Messages sent: {len(messages)}")
    for index, message in enumerate(messages):
        print(_format_message_for_log(message, index))


def log_llm_output(turn: int, message: Any) -> None:
    section(f"LLM OUTPUT — Turn {turn}")
    tool_calls = _message_tool_calls(message)
    if tool_calls:
        print(f"{INDENT}Type: tool calls ({len(tool_calls)})")
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
            else:
                name = tool_call.function.name
                arguments = tool_call.function.arguments
            print(f"{INDENT}• {name}")
            print(f"{INDENT}  args: {arguments}")
        return

    print(f"{INDENT}Type: text response")
    print(format_agent_text(_message_content(message)))


def log_tool_execution(tool_name: str, arguments: dict[str, Any], result: Any) -> None:
    subsection(f"Tool: {tool_name}")
    print(f"{INDENT}Arguments: {json.dumps(arguments, ensure_ascii=False)}")
    print(f"{INDENT}Result:")
    print(format_tool_result(tool_name, result))


def log_state_update(state: TripState) -> None:
    subsection("State updated")
    print(format_trip_state(state))


def log_user_input(text: str) -> None:
    section("USER INPUT")
    print(f"{INDENT}{text}")


def log_retry(temperature: float) -> None:
    print(f"\n{INDENT}⚠ Tool call failed — retrying with temperature {temperature}")


def log_final_plan(plan: dict[str, Any]) -> None:
    section("FINAL TRAVEL PLAN")
    rows = [
        ["Destination", plan.get("destination", "—")],
        ["Nights", plan.get("nights", "—")],
        [
            "Transport",
            f"{plan.get('transport', {}).get('type', '—')} — "
            f"{plan.get('transport', {}).get('name', '—')} "
            f"({_format_price(plan.get('transport', {}).get('price'))})",
        ],
        [
            "Hotel",
            f"{plan.get('hotel', {}).get('name', '—')} "
            f"({_format_price(plan.get('hotel', {}).get('price_per_night'))}/night)",
        ],
        ["Total cost", _format_price(plan.get("total_cost"))],
        ["Within budget", plan.get("within_budget", "—")],
    ]
    print(_format_table(["Field", "Value"], rows))
    print(f"\n{INDENT}Recommendation:")
    print(f"{INDENT}  {plan.get('recommendation', '—')}")
