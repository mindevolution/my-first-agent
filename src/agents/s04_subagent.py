#!/usr/bin/env python3
# Harness: the loop -- keep feeding real tool results back into the model.
"""
s04_subagent.py — Todo coding agent (CLI, HTTP API, or `python ... serve`).

    user message
        -> model reply with todo plan
        -> update plan state
        -> continue execution

Also exposes a FastAPI app (`app`) for a Vite React UI.
"""

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

# International API host (use if your key is from Model Studio outside China, or TLS to China fails).
_INTL_HTTP_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"

WORKDIR = Path.cwd()
PLAN_REMINDER_INTERVAL = 3

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


def _call_generation_nonstream(**call_kwargs):
    """
    Same contract as the tools branch of _generation_to_stdout, but no stdout.
    Used by run_one_turn (always non-stream when tools are present).
    """
    kwargs = dict(call_kwargs)
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
        return rsp


SYSTEM = f"""You are a coding agent at {WORKDIR}.

Planning (required for multi-step work):
- If the user asks for anything that needs more than one tool call or more than one file/command, you MUST call `todo` first in that turn. Build a short checklist: one item `in_progress`, others `pending`.
- After each substantive step (file written, command run, etc.), call `todo` again to mark completed items and move `in_progress` to the next pending item.
- Keep exactly one `in_progress` at a time when multiple steps remain.
- If the session plan block shows "No session plan yet", start by calling `todo` before bash/write_file/read_file/edit_file.

Prefer tools over long prose; use bash/write_file/read_file/edit_file to change the workspace."""

SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

TOOL_HANDLERS = {
        "bash":       lambda **kw: run_bash(kw["command"]),
        "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
        "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    }

class AgentTemplate:
    """
    Parse agent definition from markdown frontmatter.

    Real Claude Code loads agent definitions from .claude/agents/*.md.
    Frontmatter fields: name, tools, disallowedTools, skills, hooks,
    model, effort, permissionMode, maxTurns, memory, isolation, color,
    background, initialPrompt, mcpServers.
    3 sources: built-in, custom (.claude/agents/), plugin-provided.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.stem
        self.config = {}
        self.system_prompt = ""
        self._parse()

    def _parse(self):
        text = self.path.read_text()
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            self.system_prompt = text
            return
        for line in match.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                self.config[k.strip()] = v.strip()
        self.system_prompt = match.group(2).strip()
        self.name = self.config.get("name", self.name)

CHILD_TOOLS = [{
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

# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        response = Generation.call(
            model="qwen-plus", system=SUBAGENT_SYSTEM, messages=sub_messages,
            tools=CHILD_TOOLS, result_format="message",
        )
        sub_messages.append({"role": "assistant", "content": response.output})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.output:
            if block.get("type") == "tool_use":
                handler = TOOL_HANDLERS.get(block.get("name"))
                output = handler(**block.get("arguments")) if handler else f"Unknown tool: {block.get('name')}"
                results.append({"type": "tool_result", "tool_use_id": block.get("id"), "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    # Only the final text returns to the parent -- child context is discarded
    return "".join(b.get("content") for b in response.output if b.get("content")) or "(no summary)"

PARENT_TOOLS = CHILD_TOOLS + [{
    "type": "function",
    "function": {
        "name": "todo",
        "description": (
            "Update the session task list. Call at the start of multi-step work and after each major step. "
            "Pass the full list each time (replace prior plan). Exactly one item should be in_progress until all done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "A description of the task to perform.",
                            },
                            "status": {
                                "type": "string",
                                "description": "The status of the task.",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label for the in-progress item.",
                                "default": "",
                            },
                        },
                        "required": ["content", "status"],
                    },
                    "description": "Full list of tasks for this session turn (replaces previous plan).",
                }
            },
            "required": ["items"],
        },
    },
}]

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0

class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def reset(self) -> None:
        """Clear plan for a new user query."""
        self.state = PlanningState()

    def update(self, items: list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(PlanItem(
                content=content,
                status=status,
                active_form=active_form,
            ))

        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return (
            "<reminder>You have an active session plan but have not called `todo` for several tool rounds. "
            "Call `todo` now to refresh statuses (mark completed, set next in_progress) before other tools.</reminder>"
        )

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."

        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)

        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)

def _tool_names_in_assistant_calls(tool_calls) -> set[str]:
    """OpenAI-style assistant tool_calls carry the name under function.name, not top-level."""
    names: set[str] = set()
    if not tool_calls:
        return names
    for tc in tool_calls:
        tc = _as_dict(tc)
        if not isinstance(tc, dict):
            continue
        fn = _as_dict(tc.get("function"))
        n = (fn.get("name") or "").strip()
        if n:
            names.add(n)
    return names


def _messages_for_llm(messages: list, todo: TodoManager) -> list:
    """Copy messages and inject live plan after the system message (not stored in state.messages)."""
    if not messages:
        return []
    out: list = []
    inserted = False
    for m in messages:
        out.append(m)
        if not inserted and m.get("role") == "system":
            out.append({
                "role": "user",
                "content": f"<session_plan>\n{todo.render()}\n</session_plan>",
            })
            inserted = True
    if not inserted:
        out.insert(0, {
            "role": "user",
            "content": f"<session_plan>\n{todo.render()}\n</session_plan>",
        })
    return out


@dataclass
class LoopState:
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None
    todo: TodoManager = field(default_factory=TodoManager)

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

def build_tool_handlers(todo: TodoManager) -> dict:
    return {
        "bash":       lambda **kw: run_bash(kw["command"]),
        "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
        "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
        "todo":       lambda **kw: todo.update(kw["items"]),
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


def execute_tool_calls(
    tool_calls,
    handlers: dict,
    events: list | None = None,
) -> list[dict]:
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

        handler = handlers.get(name)
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


def _log_llm_response(response) -> None:
    """Log one DashScope Generation round (success, API error, or missing response)."""
    if response is None:
        logger.warning("LLM response: None (no response from Generation.call)")
        return

    record: dict = {
        "status_code": getattr(response, "status_code", None),
        "code": getattr(response, "code", None),
        "message": getattr(response, "message", None),
        "request_id": getattr(response, "request_id", None),
    }
    usage = getattr(response, "usage", None)
    if usage is not None:
        record["usage"] = (
            dict(usage) if hasattr(usage, "keys") and not isinstance(usage, dict) else usage
        )

    if getattr(response, "status_code", None) == 200 and response.output is not None:
        try:
            msg, finish_reason = _assistant_message_and_finish_reason(response)
            content = _message_content_as_str(msg)
            record["finish_reason"] = finish_reason
            record["assistant_content"] = content[:8000] + ("..." if len(content) > 8000 else "")
            record["tool_calls"] = msg.get("tool_calls")
        except Exception as e:
            record["assistant_parse_error"] = str(e)

    logger.info("LLM response: %s", json.dumps(record, ensure_ascii=False, default=str))


def run_one_turn(state: LoopState, events: list | None = None) -> bool:
    handlers = build_tool_handlers(state.todo)
    response = _call_generation_nonstream(
        model="qwen-plus",  # or qwen-turbo, qwen-max, etc.
        messages=_messages_for_llm(state.messages, state.todo),
        tools=PARENT_TOOLS,
        max_tokens=8000,
        result_format="message",
    )
    _log_llm_response(response)
    if response is None:
        state.transition_reason = None
        if events is not None:
            events.append({"type": "error", "message": "No response from model (TLS or empty)."})
        return False
    if response.status_code != 200:
        err = f"[DashScope {response.code}] {response.message}"
        if events is not None:
            events.append({"type": "error", "message": err})
        else:
            print(err)
        state.messages.append({"role": "assistant", "content": err})
        state.transition_reason = None
        return False

    assistant_msg, finish_reason = _assistant_message_and_finish_reason(response)
    assistant_msg = _normalize_assistant_message_for_dashscope(assistant_msg)
    text = _message_content_as_str(assistant_msg)
    if events is not None:
        events.append({
            "type": "assistant",
            "content": text,
            "finish_reason": finish_reason,
            "tool_calls": assistant_msg.get("tool_calls"),
        })
    elif text:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    state.messages.append(assistant_msg)

    tool_calls = assistant_msg.get("tool_calls")
    if not tool_calls:
        state.transition_reason = None
        return False

    tool_names = _tool_names_in_assistant_calls(tool_calls)
    used_todo = "todo" in tool_names
    if used_todo:
        state.todo.state.rounds_since_update = 0
    else:
        state.todo.note_round_without_update()

    tool_messages = execute_tool_calls(tool_calls, handlers, events=events)
    if not tool_messages:
        state.transition_reason = None
        return False

    state.messages.extend(tool_messages)
    if not used_todo:
        reminder = state.todo.reminder()
        if reminder:
            state.messages.append({"role": "user", "content": reminder})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True

MAX_TOOL_ROUNDS = 32


def agent_loop(state: LoopState, events: list | None = None) -> None:
    rounds = 0
    while run_one_turn(state, events=events):
        rounds += 1
        if rounds >= MAX_TOOL_ROUNDS:
            msg = "Agent stopped: max tool rounds reached."
            if events is not None:
                events.append({"type": "warning", "message": msg})
            else:
                print(f"\033[33m{msg}\033[0m")
            break

def _serialize_messages_for_api(messages: list, tool_cap: int = 4000) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        row = dict(m) if isinstance(m, dict) else dict(m)
        role = row.get("role")
        content = row.get("content")
        if role == "tool" and isinstance(content, str) and len(content) > tool_cap:
            row = {**row, "content": content[:tool_cap] + "...[truncated]"}
        if row.get("tool_calls"):
            row = {
                **row,
                "tool_calls": json.loads(json.dumps(row["tool_calls"], default=str)),
            }
        out.append(row)
    return out


# --- HTTP API (FastAPI; optional — pip install fastapi uvicorn) --------

app = None
_api_sessions: dict[str, list] = {}

try:
    import uuid as _uuid

    from fastapi import FastAPI as _FastAPI
    from fastapi import HTTPException as _HTTPException
    from fastapi.middleware.cors import CORSMiddleware as _CORSMiddleware
    from pydantic import BaseModel as _BaseModel
    from pydantic import Field as _Field

    class _ChatRequest(_BaseModel):
        message: str = _Field(..., min_length=1)
        session_id: str | None = None

    class _ChatResponse(_BaseModel):
        session_id: str
        events: list
        messages: list
        plan: str

    app = _FastAPI(title="s04_subagent", version="1.0.0")
    app.add_middleware(
        _CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def _api_health():
        return {"ok": True, "workspace": str(WORKDIR)}

    @app.post("/api/chat", response_model=_ChatResponse)
    def _api_chat(body: _ChatRequest):
        if not (os.environ.get("DASHSCOPE_API_KEY") or "").strip():
            raise _HTTPException(status_code=503, detail="DASHSCOPE_API_KEY is not set")
        sid = body.session_id or str(_uuid.uuid4())
        if sid not in _api_sessions:
            _api_sessions[sid] = [{"role": "system", "content": SYSTEM}]
        msgs = _api_sessions[sid]
        msgs.append({"role": "user", "content": body.message.strip()})
        state = LoopState(messages=msgs)
        events: list = []
        agent_loop(state, events=events)
        return _ChatResponse(
            session_id=sid,
            events=events,
            messages=_serialize_messages_for_api(msgs),
            plan=state.todo.render(),
        )

    @app.delete("/api/sessions/{session_id}")
    def _api_delete_session(session_id: str):
        _api_sessions.pop(session_id, None)
        return {"ok": True}

except ImportError:
    pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        if app is None:
            print(
                "FastAPI is not installed. Run: pip install fastapi 'uvicorn[standard]'",
                file=sys.stderr,
            )
            raise SystemExit(1)
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8765)
        raise SystemExit(0)

    _logs_dir = WORKDIR / "logs"
    _logs_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _logs_dir / "s04_subagent.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(_log_path, encoding="utf-8"),
        ],
        force=True,
    )
    history: list = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)
        print()