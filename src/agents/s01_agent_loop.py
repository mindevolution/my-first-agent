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
from dataclasses import dataclass

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


def _generation_call_with_tls_fallback(**call_kwargs):
    try:
        return Generation.call(**call_kwargs)
    except requests.exceptions.SSLError:
        base = (dashscope.base_http_api_url or "").lower()
        if "intl" in base or os.environ.get("DASHSCOPE_DISABLE_TLS_FALLBACK", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            _dashscope_ssl_hint()
            raise
        print(
            "\033[33mTLS to DashScope failed; retrying once on international endpoint "
            f"({_INTL_HTTP_BASE}).\033[0m",
            flush=True,
        )
        dashscope.base_http_api_url = _INTL_HTTP_BASE.rstrip("/")
        try:
            return Generation.call(**call_kwargs)
        except requests.exceptions.SSLError:
            _dashscope_ssl_hint()
            raise

SYSTEM = (
    f"You are a coding agent in the workspace directory: {os.getcwd()}. "
    "You have exactly one tool: `bash`. There is no write_file, edit_file, or apply_patch tool—do not mention or assume them. "
    "To create or overwrite a file, call `bash` with a shell command such as: "
    "`printf '%s\\n' 'line1' 'line2' > path.py`, or `cat <<'EOF' > path.py\\n...\\nEOF`. "
    "Prefer running real commands over giving manual instructions. Act first, then summarize what you did."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run one shell command in the current workspace (cwd is the project root). "
            "This is the only way to read/write files: use redirection, heredocs, tee, etc. "
            "Do not ask for a separate write_file tool—it does not exist."
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
}]

# response = Generation.call(
#     model="qwen-plus",  # or qwen-turbo, qwen-max, etc.
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello."},
#     ],
#     result_format="message",
# )
# print(response.output)

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

        if name != "bash":
            out = f"Error: unknown tool {name!r}"
        else:
            command = args.get("command", "")
            print(f"\033[33m$ {command}\033[0m")
            out = run_bash(command)
            print(out[:200])

        results.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": name,
            "content": out,
        })
    return results


def run_one_turn(state: LoopState) -> bool:
    try:
        response = _generation_call_with_tls_fallback(
            model="qwen-plus",  # or qwen-turbo, qwen-max, etc.
            messages=state.messages,
            tools=TOOLS,
            max_tokens=8000,
            result_format="message",
        )
    except requests.exceptions.SSLError:
        state.transition_reason = None
        return False
    if response.status_code != 200:
        err = f"[DashScope {response.code}] {response.message}"
        print(err)
        state.messages.append({"role": "assistant", "content": err})
        state.transition_reason = None
        return False
    print(response.output)

    assistant_msg, _finish_reason = _assistant_message_and_finish_reason(response)
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

def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass

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

        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()