#!/usr/bin/env python3
# Harness: the loop -- keep feeding real tool results back into the model.
"""
s01-the-agent-loop.py - The Agent Loop.

This file teacher the smallest useful coding-agent pattern:

    user message
        -> model reply
        -> if tool_use: execute tools
        -> write tool_result back to messages
        -> repeat

It intentioanally is simple, but powerful.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import readline

    readline.parse_and_bind('set bind-tty-special-chars on');
    readline.parse_and_bind('set input-meta on');
    readline.parse_and_bind('set output-meta on');
    readline.parse_and_bind('set convert-meta off');
    readline.parse_and_bind('set enable-meta-keybindings on');

except ImportError:
    pass


import requests
import dashscope
from dashscope import Generation

# International API host (use if your key is from Model Studio outside China, or TLS to China fails).
_INTL_HTTP_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"

WORKDIR = Path.cwd()

# Auth
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

# Endpoint: dashscope reads DASHSCOPE_HTTP_BASE_URL at import; override USE_INTL after import.
if os.environ.get("DASHSCOPE_USE_INTL", "").strip().lower() in ("1", "true", "yes"):
    dashscope.base_http_api_url = _INTL_HTTP_BASE.rstrip("/")
elif _custom := os.environ.get("DASHSCOPE_HTTP_BASE_URL"):
    dashscope.base_http_api_url = _custom.rstrip("/")


def _dashscope_ssl_hint() -> None:
    print(
        "\033[31mDashScope TLS failed.\033[0m Use an endpoint that matches your API key, e.g.\n"
        f"  export DASHSCOPE_HTTP_BASE_URL={_INTL_HTTP_BASE}\n"
        "or  export DASHSCOPE_USE_INTL=1\n"
        "China-region keys should use the default host; fix VPN/firewall if TLS still breaks."
    )


def _message_content_as_str(message) -> str:
    if message is None:
        return ""
    m = dict(message) if hasattr(message, "keys") and not isinstance(message, dict) else message
    if not isinstance(m, dict):
        return ""
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for item in c:
            if isinstance(item, dict) and "text" in item:
                parts.append(item.get("text") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(c)


def _tls_retry_switch_to_intl(already_retried: bool) -> bool:
    """Return True if we switched endpoint and caller should retry. False = give up."""
    if already_retried:
        return False
    base = (dashscope.base_http_api_url or "").lower()
    if "intl" in base:
        return False
    if os.environ.get("DASHSCOPE_DISABLE_TLS_FALLBACK", "").strip().lower() in ("1", "true", "yes"):
        return False
    print(
        "\033[33mTLS to DashScope failed; retrying once on international endpoint "
        f"({_INTL_HTTP_BASE}).\033[0m",
        flush=True,
    )
    dashscope.base_http_api_url = _INTL_HTTP_BASE.rstrip("/")
    return True


def _stream_generation_to_stdout(**call_kwargs):
    """
    Stream assistant text to stdout. Uses stream=True and incremental_output=False so the
    SDK merges chunks into cumulative snapshots; we print only the new suffix each time.
    Returns the last GenerationResponse, or None on unrecoverable TLS / empty stream.
    """
    kwargs = dict(call_kwargs)
    kwargs["stream"] = True
    kwargs["incremental_output"] = True

    prev_full = ""
    last_rsp = None
    tls_retried = False

    while True:
        try:
            gen = Generation.call(**kwargs)
        except requests.exceptions.SSLError:
            if _tls_retry_switch_to_intl(tls_retried):
                tls_retried = True
                prev_full = ""
                continue
            _dashscope_ssl_hint()
            return None

        try:
            for rsp in gen:
                last_rsp = rsp
                if rsp.status_code != 200:
                    continue
                out = rsp.output
                if not out:
                    continue
                choices = out.get("choices")
                if not choices:
                    continue
                msg = choices[0].get("message")
                full = _message_content_as_str(msg)
                if full.startswith(prev_full):
                    delta = full[len(prev_full) :]
                else:
                    delta = full
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                prev_full = full
        except requests.exceptions.SSLError:
            if _tls_retry_switch_to_intl(tls_retried):
                tls_retried = True
                prev_full = ""
                continue
            _dashscope_ssl_hint()
            return None
        break

    sys.stdout.write("\n")
    sys.stdout.flush()
    return last_rsp


def _generation_to_stdout(**call_kwargs):
    """
    Print assistant text and return the final GenerationResponse.

    When ``tools`` are passed, uses ``stream=False``. DashScope streams tool calls as
    deltas merged by the SDK (see ``dashscope.utils.message_utils.merge_single_response``):
    ``function.name`` / ``arguments`` are string-concatenated across chunks, and fields
    with empty values are skipped—so you can see empty ``id``, missing ``name``, or
    truncated ``arguments``. A single non-stream response avoids that class of bugs.
    """
    kwargs = dict(call_kwargs)
    if kwargs.get("tools"):
        kwargs["stream"] = False
        kwargs.pop("incremental_output", None)
        tls_retried = False
        while True:
            try:
                rsp = Generation.call(**kwargs)
            except requests.exceptions.SSLError:
                if _tls_retry_switch_to_intl(tls_retried):
                    tls_retried = True
                    continue
                _dashscope_ssl_hint()
                return None
            if rsp.status_code == 200 and rsp.output is not None:
                out = rsp.output
                choices = (
                    out.get("choices")
                    if isinstance(out, dict)
                    else getattr(out, "choices", None)
                )
                if choices:
                    c0 = choices[0]
                    if hasattr(c0, "keys") and not isinstance(c0, dict):
                        c0 = dict(c0)
                    msg = c0.get("message") if isinstance(c0, dict) else None
                    full = _message_content_as_str(msg)
                    if full:
                        sys.stdout.write(full)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return rsp
    return _stream_generation_to_stdout(**kwargs)


SYSTEM = (
    f"You are a coding agent in the workspace directory: {os.getcwd()}. "
    "You have tools: bash, read_file, write_file, edit_file. "
    "To create or overwrite a file you MUST call write_file(path, content)—do not only say you will create it. "
    "Use read_file to read files, edit_file to replace a unique snippet, bash for commands. "
    "Call tools instead of narrating; after tool results, give a short summary."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run one shell command in the current workspace (cwd is the project root). "
            "Prefer write_file/read_file/edit_file for file content; use bash for builds, git, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A single bash command (e.g. create a file: printf 'print(1)\\n' > hello.py)",
                },
            },
            "required": ["command"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file and return the content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read.",
                },
                "limit": {
                    "type": "integer",
                    "description": "The maximum number of lines to read.",
                },
            },
            "required": ["path"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write to.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Edit a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "The text to replace.",  
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace with.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
},
]

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

@dataclass
class LoopState:
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Command is dangerous blocked."
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (120s)."
    except (FileNotFoundError, PermissionError, OSError) as e:
        return f"Error: {e}"

    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "No output."

def run_read(path: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]  # hard cap to avoid blowing up the context

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
}


def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def _assistant_message_and_finish_reason(response) -> tuple[dict, str | None]:
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


def _as_dict(obj) -> dict:
    return dict(obj) if obj is not None and hasattr(obj, "keys") else obj


def _tool_arguments_json_for_api(raw) -> str:
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


def _normalize_assistant_message_for_dashscope(msg: dict) -> dict:
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
        tc = _as_dict(tc)
        if not isinstance(tc, dict):
            continue
        fn = _as_dict(tc.get("function"))
        if not isinstance(fn, dict):
            fn = {"name": "", "arguments": "{}"}
        fn["arguments"] = _tool_arguments_json_for_api(fn.get("arguments"))
        if "name" not in fn:
            fn["name"] = ""
        tc["function"] = fn
        fixed.append(tc)
    msg["tool_calls"] = fixed
    return msg


def execute_tool_calls(tool_calls) -> list[dict]:
    """Qwen/DashScope uses OpenAI-style tool_calls on the assistant message."""
    results = []
    for tc in tool_calls:
        tc = _as_dict(tc)
        fn = _as_dict(tc.get("function"))
        name = fn.get("name", "")
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}

        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            out = f"Error: unknown tool {name!r}"
        else:
            try:
                out = handler(**args)
            except TypeError as e:
                out = f"Error: invalid arguments for {name!r}: {e}"
            except ValueError as e:
                out = f"Error: {e}"
            except Exception as e:
                out = f"Error: {e}"

            if name == "bash":
                print(f"\033[33m$ {args.get('command', '')}\033[0m")
            elif name == "write_file":
                print(f"\033[33mwrite_file\033[0m {args.get('path', '')!r} ({len(args.get('content') or '')} bytes)")
            elif name == "read_file":
                print(f"\033[33mread_file\033[0m {args.get('path', '')!r}")
            elif name == "edit_file":
                print(f"\033[33medit_file\033[0m {args.get('path', '')!r}")
            clip = (out or "")[:200]
            if clip:
                print(clip)

        results.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": name,
            "content": out if isinstance(out, str) else str(out),
        })
    return results


def run_one_turn(state: LoopState) -> bool:
    response = _generation_to_stdout(
        model="qwen-plus",  # or qwen-turbo, qwen-max, etc.
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
        result_format="message",
    )
    if response is None:
        state.transition_reason = None
        return False
    if response.status_code != 200:
        err = f"[DashScope {response.code}] {response.message}"
        print(err)
        state.messages.append({"role": "assistant", "content": err})
        state.transition_reason = None
        return False

    assistant_msg, _finish_reason = _assistant_message_and_finish_reason(response)
    assistant_msg = _normalize_assistant_message_for_dashscope(assistant_msg)
    state.messages.append(assistant_msg)

    tool_calls = assistant_msg.get("tool_calls")
    if not tool_calls:
        state.transition_reason = None
        return False

    tool_messages = execute_tool_calls(tool_calls)
    if not tool_messages:
        state.transition_reason = None
        return False

    state.messages.extend(tool_messages)
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True

MAX_TOOL_ROUNDS = 32


def agent_loop(state: LoopState) -> None:
    rounds = 0
    while run_one_turn(state):
        rounds += 1
        if rounds >= MAX_TOOL_ROUNDS:
            print("\033[33mAgent stopped: max tool rounds reached.\033[0m")
            break

if __name__ == "__main__":
    history: list = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)
        print()