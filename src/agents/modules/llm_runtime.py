import json


def assistant_message_and_finish_reason(response) -> tuple[dict, str | None]:
    """DashScope Generation uses response.output.choices[0], not .content / .stop_reason."""
    out = response.output
    if out is None:
        raise ValueError("DashScope response has no output")
    choices = out.get("choices")
    if choices:
        choice = choices[0]
        msg = choice.get("message")
        fr = choice.get("finish_reason")
        return dict(msg), fr
    text = out.get("text")
    if text is not None:
        return {"role": "assistant", "content": text}, out.get("finish_reason")
    raise ValueError(f"Unexpected DashScope output shape: {out!r}")


def as_dict(obj) -> dict:
    return dict(obj) if obj is not None and hasattr(obj, "keys") else obj


def tool_arguments_json_for_api(raw) -> str:
    """DashScope requires function.arguments to be a string of valid JSON (object)."""
    if raw is None:
        return "{}"
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return "{}"
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return "{}"
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        return "{}"
    return json.dumps(parsed, ensure_ascii=False)


def normalize_assistant_message_for_dashscope(msg: dict) -> dict:
    """
    Normalize tool_calls for replay: arguments must be non-empty valid JSON objects.
    Model output may be partial, Python-ish, or non-object JSON — coerce to "{}" or dumps(dict).
    """
    msg = dict(msg)
    tcs = msg.get("tool_calls")
    if not tcs:
        return msg
    fixed: list[dict] = []
    for tc in tcs:
        tc = as_dict(tc)
        if not isinstance(tc, dict):
            continue
        fn = as_dict(tc.get("function"))
        if not isinstance(fn, dict):
            fn = {"name": "", "arguments": "{}"}
        fn["arguments"] = tool_arguments_json_for_api(fn.get("arguments"))
        if "name" not in fn:
            fn["name"] = ""
        tc["function"] = fn
        fixed.append(tc)
    msg["tool_calls"] = fixed
    return msg


def execute_tool_calls(
    tool_calls,
    handlers: dict,
    events: list | None = None,
) -> list[dict]:
    """Qwen/DashScope uses OpenAI-style tool_calls on the assistant message."""
    results = []
    for tc in tool_calls:
        tc = as_dict(tc)
        fn = as_dict(tc.get("function"))
        name = fn.get("name", "")
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}

        handler = handlers.get(name)
        if handler is None:
            out = f"Error: unknown tool {name!r}"
        else:
            try:
                tid = tc.get("id")
                tool_use_id = str(tid) if tid is not None else ""
                out = handler(**{**args, "tool_use_id": tool_use_id})
            except TypeError as e:
                out = f"Error: invalid arguments for {name!r}: {e}"
            except ValueError as e:
                out = f"Error: {e}"
            except Exception as e:
                out = f"Error: {e}"

            if events is not None:
                ev: dict = {"type": "tool", "name": name}
                if name == "bash":
                    ev["command"] = args.get("command", "")
                elif name == "write_file":
                    ev["path"] = args.get("path", "")
                    ev["bytes"] = len(args.get("content") or "")
                elif name == "read_file":
                    ev["path"] = args.get("path", "")
                elif name == "edit_file":
                    ev["path"] = args.get("path", "")
                elif name == "todo":
                    ev["items"] = len(args.get("items") or [])
                elif name == "run_subtask":
                    ev["title"] = (args.get("title") or "")[:80]
                    ev["prompt_preview"] = (args.get("prompt") or "")[:200]
                elif name == "compact":
                    ev["focus"] = (args.get("focus") or "")[:80]
                ev["output_preview"] = (out or "")[:800]
                events.append(ev)
            else:
                if name == "bash":
                    print(f"\033[33m$ {args.get('command', '')}\033[0m")
                elif name == "write_file":
                    print(
                        f"\033[33mwrite_file\033[0m {args.get('path', '')!r} "
                        f"({len(args.get('content') or '')} bytes)"
                    )
                elif name == "read_file":
                    print(f"\033[33mread_file\033[0m {args.get('path', '')!r}")
                elif name == "edit_file":
                    print(f"\033[33medit_file\033[0m {args.get('path', '')!r}")
                elif name == "todo":
                    nitems = len(args.get("items") or [])
                    print(f"\033[33mtodo\033[0m ({nitems} items)")
                elif name == "run_subtask":
                    t = (args.get("title") or "").strip() or "subtask"
                    print(f"\033[36mrun_subtask\033[0m [{t}]")
                elif name == "compact":
                    print(f"\033[33mcompact\033[0m [{args.get('focus', '')}]")
                clip = (out or "")[:200]
                if clip:
                    print(clip)

        results.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": out if isinstance(out, str) else str(out),
            }
        )
    return results

