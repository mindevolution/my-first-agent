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


import dashscope
from dashscope import Generation

# Auth (pick one)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}"
    "Use bash to inspect and change the workspace. Act first, then report clearly what you did."
)

TOOLS = [{
    "name": "bash",
    "description": "Run a bash command in current workspace.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
        },
        "required": ["command"],
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
    if not isinstance(content, list):
        return ""
    text = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            text.append(text)
    return "\n".join(text).strip()

def execute_tool_calls(response_content) -> list[dict]:
    results = []
    for block in response_content:
        if block.type != "tool_use":
            continue
        command = block.input["command"]
        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        print(output[:200])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
    return results

def run_one_turn(state: LoopState) -> bool:
    response = Generation.call(
        model="qwen-plus",  # or qwen-turbo, qwen-max, etc.
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )
    state.messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False

    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False

    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True

def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass

if __name__ == "__main__":
    history = []
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