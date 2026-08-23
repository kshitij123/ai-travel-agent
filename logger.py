import json
import re
from typing import Any

from state import TripState

INDENT = "  "
BOLD, DIM, YELLOW, RESET = "\033[1m", "\033[2m", "\033[33m", "\033[0m"


def bold(t: str) -> str:
    return f"{BOLD}{t}{RESET}"


def section(title: str) -> None:
    print(f"\n{'═' * 88}\n  {bold(title)}\n{'═' * 88}")


def subsection(title: str) -> None:
    print(f"\n── {bold(title)} " + "─" * max(1, 83 - len(title)))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return f"{INDENT}(no data)"
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), max((len(r[i]) for r in str_rows), default=0)) for i, h in enumerate(headers)]
    line = lambda cells: f"{INDENT}│ " + " │ ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " │"
    div = lambda ch, sep: f"{INDENT}{ch}" + sep.join("─" * w for w in widths) + ch.replace("┌", "┐").replace("├", "┤").replace("└", "┘")
    return "\n".join([div("┌─", "─┬─"), line([bold(h) for h in headers]), div("├─", "─┼─")] + [line(r) for r in str_rows] + [div("└─", "─┴─")])


def _price(amt: Any) -> str:
    return f"₹{amt:,.0f}" if isinstance(amt, (int, float)) else str(amt)


def _transport_table(opts: list[dict], mode: str) -> str:
    label = "Airline" if mode == "flight" else "Train"
    rows = [[o.get("id", ""), f"{o.get('from', '')} → {o.get('to', '')}", o.get("airline") or o.get("train", ""), o.get("departure", ""), o.get("date", ""), _price(o.get("price"))] for o in opts]
    return _table(["ID", "Route", label, "Depart", "Date", "Price"], rows)


def _hotel_table(opts: list[dict]) -> str:
    return _table(["ID", "Hotel", "City", "Rating", "Per Night"], [[o.get("id", ""), o.get("name", ""), o.get("city", ""), o.get("rating", ""), _price(o.get("price_per_night"))] for o in opts])


def _budget_table(b: dict[str, float]) -> str:
    return _table(["Item", "Amount"], [["Transport", _price(b.get("transport_cost"))], ["Hotel", _price(b.get("hotel_cost"))], ["Total", _price(b.get("total_cost"))]])


def format_state(s: TripState) -> str:
    parts = [bold("Summary"), _table(["Field", "Value"], [["Source", s.source or "—"], ["Destination", s.destination or "—"], ["Nights", s.nights if s.nights is not None else "—"], ["Budget", _price(s.budget) if s.budget else "—"], ["Transport mode", s.transport_mode or "—"], ["Status", s.status]])]
    if s.transport_options:
        parts += ["", bold(f"Transport options ({s.transport_mode or 'unknown'})"), _transport_table(s.transport_options, s.transport_mode or "train")]
    if s.hotel_options:
        parts += ["", bold("Hotel options"), _hotel_table(s.hotel_options)]
    if s.budget_breakdown:
        parts += ["", bold("Budget breakdown"), _budget_table(s.budget_breakdown)]
    return "\n".join(parts)


def _serialize(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {k: _serialize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_serialize(x) for x in v]
    if hasattr(v, "model_dump"):
        return _serialize(v.model_dump())
    return _serialize(vars(v)) if hasattr(v, "__dict__") else str(v)


def _get(msg: Any, key: str, default=None):
    return msg.get(key, default) if isinstance(msg, dict) else getattr(msg, key, default)


def log_request(turn: int, req: dict[str, Any]) -> None:
    print(f"\n{bold(f'LLM REQUEST — Turn {turn}')}\n\n{json.dumps(_serialize(req), indent=2, ensure_ascii=False)}\n")


def log_response(turn: int, msg: Any) -> None:
    resp = {"role": (_get(msg, "role") or "?").lower(), "content": _get(msg, "content"), "tool_calls": _serialize(_get(msg, "tool_calls") or []) or None}
    print(f"\n{bold(f'LLM RESPONSE — Turn {turn}')}\n\n{json.dumps(resp, indent=2, ensure_ascii=False)}\n")


def _result_fmt(name: str, result: Any) -> str:
    if name in ("search_flights", "search_trains") and isinstance(result, list):
        return _transport_table(result, "flight" if name == "search_flights" else "train")
    if name == "search_hotels" and isinstance(result, list):
        return _hotel_table(result)
    if name == "calculate_budget" and isinstance(result, dict):
        return _budget_table(result)
    return json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)


def log_tool(name: str, args: dict[str, Any], result: Any) -> None:
    subsection(f"Tool: {name}")
    print(f"{INDENT}{bold('Arguments:')} {json.dumps(args, ensure_ascii=False)}\n{INDENT}{bold('Result:')}\n{_result_fmt(name, result)}")


def log_state(state: TripState) -> None:
    subsection("TripState updated")
    print(format_state(state))


def log_user(text: str) -> None:
    section("USER INPUT")
    print(f"{INDENT}{text}")


def log_retry(temp: float) -> None:
    print(f"\n{INDENT}{YELLOW}⚠{RESET} {bold('Tool call failed')} — retrying with temperature {temp}")


def log_compression(before: int, after: int, summary: str) -> None:
    subsection("Context compression")
    print(f"{INDENT}{bold('Messages before:')} {before}")
    print(f"{INDENT}{bold('Messages after:')} {after}")
    print(f"{INDENT}{bold('Compressed summary:')}\n{INDENT}  {summary}")


def _md_table_to_ascii(lines: list[str]) -> str:
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.fullmatch(r"\|[-| :]+\|", l.strip())]
    return _table(rows[0], rows[1:]) if len(rows) >= 2 else "\n".join(lines)


def format_agent_text(content: str | None) -> str:
    if not content:
        return f"{INDENT}(empty response)"
    out, lines, i = [], content.splitlines(), 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and "|" in lines[i].strip()[1:]:
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out.append(_md_table_to_ascii(tbl))
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def log_final_plan(plan: dict[str, Any]) -> None:
    section("FINAL TRAVEL PLAN")
    t, h = plan.get("transport", {}), plan.get("hotel", {})
    rows = [["Destination", plan.get("destination", "—")], ["Nights", plan.get("nights", "—")], ["Transport", f"{t.get('type', '—')} — {t.get('name', '—')} ({_price(t.get('price'))})"], ["Hotel", f"{h.get('name', '—')} ({_price(h.get('price_per_night'))}/night)"], ["Total cost", _price(plan.get("total_cost"))], ["Within budget", plan.get("within_budget", "—")]]
    print(_table(["Field", "Value"], rows))
    print(f"\n{INDENT}{bold('Recommendation:')}\n{INDENT}  {plan.get('recommendation', '—')}")
