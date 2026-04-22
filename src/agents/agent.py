#!/usr/bin/env python3
# Harness: persistent tasks -- goals that outlive any single conversation.
"""
s12_worktree_isolation.py - Worktree + Task Isolation

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task has a dependency graph (blockedBy).

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], ...}

    Dependency resolution:
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- completing task 1 removes it from task 2's blockedBy

Key insight: "State that survives compression -- because it's outside the conversation."
"""

import json
import logging
import os
import re
import time
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
import threading
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

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
try:
    from .modules.autonomy import (
        scan_unclaimed_tasks as _scan_unclaimed_tasks,
        claim_task as _claim_task_in_dir,
        make_identity_block,
    )
    from .modules.worktree import detect_repo_root, EventBus, WorktreeManager
    from .modules.llm_runtime import (
        assistant_message_and_finish_reason as _assistant_message_and_finish_reason_impl,
        as_dict as _as_dict_impl,
        tool_arguments_json_for_api as _tool_arguments_json_for_api_impl,
        normalize_assistant_message_for_dashscope as _normalize_assistant_message_for_dashscope_impl,
        execute_tool_calls as _execute_tool_calls_impl,
    )
    from .modules.tool_specs import build_tool_specs
    from .modules.task_manager import TaskManager
    from .modules.teammate_manager import TeammateManager
except ImportError:
    from modules.autonomy import (  # type: ignore
        scan_unclaimed_tasks as _scan_unclaimed_tasks,
        claim_task as _claim_task_in_dir,
        make_identity_block,
    )
    from modules.worktree import detect_repo_root, EventBus, WorktreeManager  # type: ignore
    from modules.llm_runtime import (  # type: ignore
        assistant_message_and_finish_reason as _assistant_message_and_finish_reason_impl,
        as_dict as _as_dict_impl,
        tool_arguments_json_for_api as _tool_arguments_json_for_api_impl,
        normalize_assistant_message_for_dashscope as _normalize_assistant_message_for_dashscope_impl,
        execute_tool_calls as _execute_tool_calls_impl,
    )
    from modules.tool_specs import build_tool_specs  # type: ignore
    from modules.task_manager import TaskManager  # type: ignore
    from modules.teammate_manager import TeammateManager  # type: ignore

logger = logging.getLogger(__name__)

# Human-facing progress for blocking LLM calls (stderr so it does not mix with tool/stdout traces).
_LLM_PROGRESS_LABELS = {
    "main_agent": "main agent",
    "subagent": "sub-agent",
    "compact_summary": "conversation summary",
}


def _stderr_llm_waiting(context: str | None) -> None:
    label = _LLM_PROGRESS_LABELS.get(context or "", context or "model")
    print(
        f"\033[2m[LLM]\033[0m Waiting for model response ({label})…",
        file=sys.stderr,
        flush=True,
    )


def _stderr_llm_done(context: str | None, response) -> None:
    label = _LLM_PROGRESS_LABELS.get(context or "", context or "model")
    if response is None:
        print(
            f"\033[2m[LLM]\033[0m \033[31mNo response\033[0m ({label})",
            file=sys.stderr,
            flush=True,
        )
        return
    if getattr(response, "status_code", None) == 200:
        print(
            f"\033[2m[LLM]\033[0m \033[32mResponse received\033[0m ({label})",
            file=sys.stderr,
            flush=True,
        )
    else:
        code = getattr(response, "code", "") or getattr(response, "status_code", "")
        print(
            f"\033[2m[LLM]\033[0m \033[33mAPI error {code}\033[0m ({label})",
            file=sys.stderr,
            flush=True,
        )


# International API host (use if your key is from Model Studio outside China, or TLS to China fails).
_INTL_HTTP_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"

LLM_MODEL = "qwen-plus"
WORKDIR = Path.cwd()
PLAN_REMINDER_INTERVAL = 3
SKILLS_DIR = WORKDIR / "src/skills"
CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3
PERSIST_THRESHOLD = 30000
PREVIEW_CHARS = 2000
TRANSCRIPT_DIR = WORKDIR / "src/.transcripts"
TOOL_RESULTS_DIR = WORKDIR / "src/.task_outputs" / "tool-results"
TASKS_DIR = WORKDIR / "src/.tasks"
TEAM_DIR = WORKDIR / "src/.team"
INBOX_DIR = TEAM_DIR / "inbox"
WORKTREES_DIR = WORKDIR / "src/.worktrees"

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

shutdown_requests: dict[str, dict] = {}
plan_requests: dict[str, dict] = {}
_tracker_lock = threading.Lock()


REPO_ROOT = detect_repo_root(WORKDIR)

try:
    LLM_REQUEST_TIMEOUT_SEC = max(5, int(os.environ.get("LLM_REQUEST_TIMEOUT_SEC", "45")))
except ValueError:
    LLM_REQUEST_TIMEOUT_SEC = 45

# Auth
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

# Endpoint: dashscope reads DASHSCOPE_HTTP_BASE_URL at import; override USE_INTL after import.
if os.environ.get("DASHSCOPE_USE_INTL", "").strip().lower() in ("1", "true", "yes"):
    dashscope.base_http_api_url = _INTL_HTTP_BASE.rstrip("/")
elif _custom := os.environ.get("DASHSCOPE_HTTP_BASE_URL"):
    dashscope.base_http_api_url = _custom.rstrip("/")

class BackgroundManager:
    def __init__(self):
        self.tasks = {}  # task_id -> {status, result, command}
        self._notification_queue = []  # completed task results
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """Start a background thread, return task_id immediately."""
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """Thread target: run subprocess, capture output, push to queue."""
        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
            })

    def check(self, task_id: str = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


BG = BackgroundManager()


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict | None = None,
    ) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {sorted(VALID_MSG_TYPES)}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with self._lock:
            with inbox_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        with self._lock:
            raw = inbox_path.read_text(encoding="utf-8")
            inbox_path.write_text("", encoding="utf-8")
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- Autonomous task claiming -------------------------------------------------
def scan_unclaimed_tasks() -> list[dict]:
    return _scan_unclaimed_tasks(TASKS_DIR)


def claim_task(task_id: int, owner: str) -> str:
    return _claim_task_in_dir(TASKS_DIR, task_id, owner)


TASKS = TaskManager(TASKS_DIR)


EVENTS = EventBus(WORKTREES_DIR / "events.jsonl")
WORKTREES = WorktreeManager(REPO_ROOT, WORKTREES_DIR, TASKS, EVENTS)


def handle_shutdown_request(teammate: str) -> str:
    if teammate not in TEAM.member_names():
        return f"Error: Unknown teammate '{teammate}'"
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {
            "target": teammate,
            "status": "pending",
            "created_at": time.time(),
        }
    BUS.send(
        "lead",
        teammate,
        "Please shut down gracefully.",
        "shutdown_request",
        {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        req = shutdown_requests.get(request_id)
    if not req:
        return f"Error: Unknown shutdown request_id '{request_id}'"
    return json.dumps(req, ensure_ascii=False)


def submit_plan_for_approval(sender: str, plan: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        plan_requests[req_id] = {
            "from": sender,
            "plan": plan,
            "status": "pending",
            "created_at": time.time(),
        }
    BUS.send(
        sender,
        "lead",
        plan,
        "plan_approval_response",
        {"request_id": req_id, "plan": plan},
    )
    return f"Plan submitted (request_id={req_id}). Waiting for lead approval."


def review_plan_request(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
        if req is None:
            return f"Error: Unknown plan request_id '{request_id}'"
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead",
        req["from"],
        feedback,
        "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _teammate_shutdown_response(sender: str, args: dict) -> str:
    req_id = (args.get("request_id") or "").strip()
    if not req_id:
        raise ValueError("shutdown_response requires request_id")
    if "approve" not in args:
        raise ValueError("shutdown_response requires approve for teammates")
    approve = bool(args.get("approve"))
    reason = args.get("reason", "")
    with _tracker_lock:
        if req_id in shutdown_requests:
            shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
    BUS.send(
        sender,
        "lead",
        reason,
        "shutdown_response",
        {"request_id": req_id, "approve": approve},
    )
    if approve:
        TEAM._set_member_state(sender, status="shutdown")
        evt = TEAM.stop_events.get(sender)
        if evt is not None:
            evt.set()
    return f"Shutdown {'approved' if approve else 'rejected'}"


def _teammate_plan_submission(sender: str, args: dict) -> str:
    plan = (args.get("plan") or "").strip()
    if not plan:
        raise ValueError("plan_approval requires plan text for teammates")
    return submit_plan_for_approval(sender, plan)

@dataclass
class SkillManifest:
    name: str
    description: str
    path: Path


@dataclass
class SkillDocument:
    manifest: SkillManifest
    body: str

class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return

        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            manifest = SkillManifest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(manifest=manifest, body=body.strip())

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text

        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)

    def describe_available(self) -> str:
        if not self.documents:
            return "(no skills available)"
        lines = []
        for name in sorted(self.documents):
            manifest = self.documents[name].manifest
            lines.append(f"- {manifest.name}: {manifest.description}")
        return "\n".join(lines)

    def load_full_text(self, name: str) -> str:
        document = self.documents.get(name)
        if not document:
            known = ", ".join(sorted(self.documents)) or "(none)"
            return f"Error: Unknown skill '{name}'. Available skills: {known}"

        return (
            f"<skill name=\"{document.manifest.name}\">\n"
            f"{document.body}\n"
            "</skill>"
        )


SKILL_REGISTRY = SkillRegistry(SKILLS_DIR)

@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)


@dataclass
class LocalLLMErrorResponse:
    status_code: int
    code: str
    message: str
    output: dict | None = None
    request_id: str | None = None
    usage: dict | None = None


def estimate_context_size(messages: list) -> int:
    return len(str(messages))


def track_recent_file(state: CompactState, path: str) -> None:
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]

def persist_large_output(tool_use_id: str, output: str) -> str:
    if len(output) <= PERSIST_THRESHOLD:
        return output

    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not stored_path.exists():
        stored_path.write_text(output)

    preview = output[:PREVIEW_CHARS]
    rel_path = stored_path.relative_to(WORKDIR)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"
    )

def collect_tool_result_blocks(messages: list) -> list[tuple[int, int, dict]]:
    blocks = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((message_index, block_index, block))
    return blocks


def micro_compact(messages: list) -> list:
    tool_results = collect_tool_result_blocks(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages

    for _, _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = block.get("content", "")
        if not isinstance(content, str) or len(content) <= 120:
            continue
        block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages


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


def _tool_names_for_llm_log(tools) -> list[str]:
    if not tools:
        return []
    names: list[str] = []
    for item in tools:
        d = item if isinstance(item, dict) else None
        if d is None and item is not None and hasattr(item, "keys"):
            try:
                d = dict(item)
            except Exception:
                continue
        if not isinstance(d, dict):
            continue
        fn = d.get("function")
        if not isinstance(fn, dict) and fn is not None and hasattr(fn, "keys"):
            try:
                fn = dict(fn)
            except Exception:
                fn = {}
        if isinstance(fn, dict):
            n = fn.get("name")
            if n:
                names.append(n)
    return names


def _messages_detail_for_llm_log(messages) -> list[dict]:
    rows: list[dict] = []
    if not messages:
        return rows
    cap = PREVIEW_CHARS
    for raw in messages:
        d = raw if isinstance(raw, dict) else None
        if d is None and raw is not None and hasattr(raw, "keys"):
            try:
                d = dict(raw)
            except Exception:
                rows.append({
                    "role": "?",
                    "content_chars": 0,
                    "content_preview": "(unserializable message)",
                })
                continue
        if not isinstance(d, dict):
            rows.append({
                "role": "?",
                "content_chars": 0,
                "content_preview": "(non-dict message)",
            })
            continue
        role = d.get("role", "")
        text = _message_content_as_str(d)
        prev = text[:cap] + ("..." if len(text) > cap else "")
        row: dict = {"role": role, "content_chars": len(text), "content_preview": prev}
        tcs = d.get("tool_calls")
        if tcs:
            tnames: list[str] = []
            for tc in tcs:
                tcd = tc if isinstance(tc, dict) else None
                if tcd is None and tc is not None and hasattr(tc, "keys"):
                    try:
                        tcd = dict(tc)
                    except Exception:
                        continue
                if not isinstance(tcd, dict):
                    continue
                fn = tcd.get("function")
                if not isinstance(fn, dict) and fn is not None and hasattr(fn, "keys"):
                    try:
                        fn = dict(fn)
                    except Exception:
                        fn = {}
                if isinstance(fn, dict):
                    n = fn.get("name")
                    if n:
                        tnames.append(n)
            if tnames:
                row["assistant_tool_names"] = tnames
        if role == "tool":
            row["tool_call_id"] = d.get("tool_call_id")
            row["tool_name"] = d.get("name")
        rows.append(row)
    return rows


def _log_llm_request(kwargs: dict, *, context: str | None = None) -> None:
    """Log outbound Generation.call: model, tools, and per-message content previews."""
    msgs = kwargs.get("messages") or []
    msg_rows = _messages_detail_for_llm_log(msgs)
    record = {
        "context": context,
        "model": kwargs.get("model"),
        "max_tokens": kwargs.get("max_tokens"),
        "result_format": kwargs.get("result_format"),
        "stream": kwargs.get("stream"),
        "incremental_output": kwargs.get("incremental_output"),
        "temperature": kwargs.get("temperature"),
        "tool_names": _tool_names_for_llm_log(kwargs.get("tools")),
        "message_count": len(msgs),
        "total_content_chars": sum(r.get("content_chars", 0) for r in msg_rows),
        "messages": msg_rows,
    }
    logger.info("LLM request: %s", json.dumps(record, ensure_ascii=False, default=str))


def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message, default=str) + "\n")
    return path


def summarize_history(messages: list, *, show_progress: bool = True) -> str:
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )
    _compact_kw = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "result_format": "message",
        "request_timeout": LLM_REQUEST_TIMEOUT_SEC,
    }
    _log_llm_request(_compact_kw, context="compact_summary")
    if show_progress:
        _stderr_llm_waiting("compact_summary")
    response = Generation.call(**_compact_kw)
    if show_progress:
        _stderr_llm_done("compact_summary", response)
    return response.output.choices[0].message.content.strip()


def compact_history(
    messages: list,
    state: CompactState,
    focus: str | None = None,
    *,
    llm_progress: bool = True,
) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")

    summary = summarize_history(messages, show_progress=llm_progress)
    if focus:
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files:
        recent_lines = "\n".join(f"- {path}" for path in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"

    state.has_compacted = True
    state.last_summary = summary

    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{summary}"
        ),
    }]



def _dashscope_ssl_hint() -> None:
    print(
        "\033[31mDashScope TLS failed.\033[0m Use an endpoint that matches your API key, e.g.\n"
        f"  export DASHSCOPE_HTTP_BASE_URL={_INTL_HTTP_BASE}\n"
        "or  export DASHSCOPE_USE_INTL=1\n"
        "China-region keys should use the default host; fix VPN/firewall if TLS still breaks."
    )


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
    kwargs.setdefault("request_timeout", LLM_REQUEST_TIMEOUT_SEC)
    log_ctx = kwargs.pop("_llm_log_context", None)
    show_progress = kwargs.pop("_llm_user_progress", False)

    prev_full = ""
    last_rsp = None
    tls_retried = False

    while True:
        try:
            _log_llm_request(kwargs, context=log_ctx)
            if show_progress:
                _stderr_llm_waiting(log_ctx)
            gen = Generation.call(**kwargs)
        except requests.exceptions.SSLError:
            if _tls_retry_switch_to_intl(tls_retried):
                tls_retried = True
                prev_full = ""
                continue
            if show_progress:
                _stderr_llm_done(log_ctx, None)
            _dashscope_ssl_hint()
            return None
        except requests.exceptions.Timeout:
            if show_progress:
                _stderr_llm_done(log_ctx, None)
            return LocalLLMErrorResponse(
                status_code=408,
                code="RequestTimeout",
                message=f"Request timed out after {kwargs.get('request_timeout')}s",
            )
        except requests.exceptions.RequestException as e:
            if show_progress:
                _stderr_llm_done(log_ctx, None)
            return LocalLLMErrorResponse(
                status_code=503,
                code="NetworkError",
                message=str(e),
            )

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
                # if delta:
                    # sys.stdout.write(delta)
                    # sys.stdout.flush()
                prev_full = full
        except requests.exceptions.SSLError:
            if _tls_retry_switch_to_intl(tls_retried):
                tls_retried = True
                prev_full = ""
                continue
            _dashscope_ssl_hint()
            if show_progress:
                _stderr_llm_done(log_ctx, None)
            return None
        break

    if show_progress:
        _stderr_llm_done(log_ctx, last_rsp)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return last_rsp


def _call_generation_nonstream(**call_kwargs):
    """
    Same contract as the tools branch of _generation_to_stdout, but no stdout.
    Used by run_one_turn (always non-stream when tools are present).
    """
    kwargs = dict(call_kwargs)
    kwargs["stream"] = False
    kwargs.pop("incremental_output", None)
    kwargs.setdefault("request_timeout", LLM_REQUEST_TIMEOUT_SEC)
    log_ctx = kwargs.pop("_llm_log_context", None)
    show_progress = kwargs.pop("_llm_user_progress", False)
    tls_retried = False
    while True:
        try:
            _log_llm_request(kwargs, context=log_ctx)
            if show_progress:
                _stderr_llm_waiting(log_ctx)
            rsp = Generation.call(**kwargs)
            if show_progress:
                _stderr_llm_done(log_ctx, rsp)
        except requests.exceptions.SSLError:
            if _tls_retry_switch_to_intl(tls_retried):
                tls_retried = True
                continue
            if show_progress:
                _stderr_llm_done(log_ctx, None)
            _dashscope_ssl_hint()
            return None
        except requests.exceptions.Timeout:
            if show_progress:
                _stderr_llm_done(log_ctx, None)
            return LocalLLMErrorResponse(
                status_code=408,
                code="RequestTimeout",
                message=f"Request timed out after {kwargs.get('request_timeout')}s",
            )
        except requests.exceptions.RequestException as e:
            if show_progress:
                _stderr_llm_done(log_ctx, None)
            return LocalLLMErrorResponse(
                status_code=503,
                code="NetworkError",
                message=str(e),
            )
        return rsp


SYSTEM = f"""You are a coding agent at {WORKDIR}.

Use load_skill when a task needs specialized instructions before you act.

Team coordination:
- You can spawn persistent teammates with `spawn_teammate`.
- Use `send_message`, `read_inbox`, and `broadcast` for coordination.
- Use `list_teammates` to check teammate status.
- Autonomous tools:
  - `idle` means teammate enters idle polling for inbox/new tasks.
  - `claim_task` claims a pending, unowned, unblocked task.
- Protocol tools:
  - `shutdown_request` / `shutdown_response` for graceful shutdown handshake.
  - `plan_approval` for teammate plan submission and lead approval/rejection.
- Worktree isolation tools:
  - Use `worktree_create` to allocate isolated git directory lanes.
  - Use `worktree_run`/`worktree_status` for lane-scoped execution.
  - Use `worktree_keep` or `worktree_remove` for closeout and lifecycle control.
  - Use `task_bind_worktree` to bind control-plane tasks to execution lanes.
  - Use `worktree_events` for lifecycle visibility.

Skills available:
{SKILL_REGISTRY.describe_available()}

Planning (required for multi-step work):
- If the user asks for anything that needs more than one tool call or more than one file/command, you MUST call `todo` first in that turn. Build a short checklist: one item `in_progress`, others `pending`.
- After each substantive step (file written, command run, etc.), call `todo` again to mark completed items and move `in_progress` to the next pending item.
- Keep exactly one `in_progress` at a time when multiple steps remain.
- If the session plan block shows "No session plan yet", start by calling `todo` before bash/write_file/read_file/edit_file.

For a **focused sub-investigation** in fresh context (e.g. "figure out which test framework we use"), call `run_subtask` with a clear `prompt`. The subagent only has bash/read_file/write_file/edit_file and returns a text summary.

Prefer tools over long prose; use bash/write_file/read_file/edit_file to change the workspace."""

SUBAGENT_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. "
    "You only have bash, read_file, write_file, and edit_file. "
    "Use tools to complete the task, then reply with a concise summary for the parent agent."
)

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


CHILD_TOOLS, PARENT_TOOLS = build_tool_specs(VALID_MSG_TYPES)

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


def _tool_calls_fingerprint(tool_calls) -> str:
    """Stable signature for assistant tool-call batches, used for loop detection."""
    normalized: list[dict] = []
    for tc in tool_calls or []:
        tc = _as_dict(tc)
        fn = _as_dict(tc.get("function"))
        name = (fn.get("name") or "").strip()
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args or {}
        if not isinstance(args, dict):
            args = {}
        normalized.append({"name": name, "args": args})
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


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
    compact: CompactState = field(default_factory=CompactState)
    last_tool_fingerprint: str | None = None
    repeated_tool_call_count: int = 0

def run_bash(command: str, tool_use_id: str) -> str:
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
    return persist_large_output(tool_use_id, output)

def run_read(path: str, tool_use_id: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return persist_large_output(tool_use_id, "\n".join(lines))

def run_write(path: str, content: str, tool_use_id: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return persist_large_output(tool_use_id, f"Wrote {len(content)} bytes to {path}")
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str, tool_use_id: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return persist_large_output(tool_use_id, f"Edited {path}")
    except Exception as e:
        return f"Error: {e}"


def build_actor_tool_handlers(actor: str) -> dict:
    return {
        "bash":       lambda **kw: run_bash(kw["command"], kw["tool_use_id"]),
        "read_file":  lambda **kw: run_read(kw["path"], kw["tool_use_id"], kw.get("limit")),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"], kw["tool_use_id"]),
        "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"], kw["tool_use_id"]),
        "load_skill": lambda **kw: SKILL_REGISTRY.load_full_text(kw["name"]),
        "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
        "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
        "task_list":   lambda **kw: TASKS.list_all(),
        "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
        "background_run": lambda **kw: BG.run(kw.get("command") or kw.get("cmd") or (_ for _ in ()).throw(
            ValueError("background_run requires 'command'")
        )),
        "check_background": lambda **kw: BG.check(),
        "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
        "list_teammates": lambda **kw: TEAM.list_all(),
        "send_message": lambda **kw: BUS.send(
            actor,
            kw["to"],
            kw["content"],
            kw.get("msg_type", "message"),
        ),
        "read_inbox": lambda **kw: json.dumps(BUS.read_inbox(actor), indent=2, ensure_ascii=False),
        "broadcast": lambda **kw: BUS.broadcast(actor, kw["content"], TEAM.member_names()),
        "shutdown_request": lambda **kw: (
            handle_shutdown_request(kw["teammate"])
            if actor == "lead"
            else "Error: shutdown_request is lead-only"
        ),
        "shutdown_response": lambda **kw: (
            check_shutdown_status(kw["request_id"])
            if actor == "lead"
            else _teammate_shutdown_response(actor, kw)
        ),
        "plan_approval": lambda **kw: (
            review_plan_request(
                kw["request_id"],
                bool(kw["approve"]),
                kw.get("feedback", ""),
            )
            if actor == "lead"
            else _teammate_plan_submission(actor, kw)
        ),
        "idle": lambda **kw: "Entering idle polling phase.",
        "claim_task": lambda **kw: claim_task(int(kw["task_id"]), actor),
        "task_bind_worktree": lambda **kw: TASKS.bind_worktree(
            kw["task_id"],
            kw["worktree"],
            kw.get("owner", actor if actor != "lead" else ""),
        ),
        "worktree_create": lambda **kw: WORKTREES.create(
            kw["name"],
            kw.get("task_id"),
            kw.get("base_ref", "HEAD"),
        ),
        "worktree_list": lambda **kw: WORKTREES.list_all(),
        "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
        "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
        "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
        "worktree_remove": lambda **kw: WORKTREES.remove(
            kw["name"],
            kw.get("force", False),
            kw.get("complete_task", False),
        ),
        "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
    }


def build_tool_handlers(todo: TodoManager) -> dict:
    return build_actor_tool_handlers("lead") | {
        "todo":       lambda **kw: todo.update(kw["items"]),
        "run_subtask": lambda **kw: invoke_run_subtask(**kw),
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
    return _assistant_message_and_finish_reason_impl(response)


def _as_dict(obj) -> dict:
    return _as_dict_impl(obj)


def _tool_arguments_json_for_api(raw) -> str:
    return _tool_arguments_json_for_api_impl(raw)


def _normalize_assistant_message_for_dashscope(msg: dict) -> dict:
    return _normalize_assistant_message_for_dashscope_impl(msg)


def execute_tool_calls(
    tool_calls,
    handlers: dict,
    events: list | None = None,
) -> list[dict]:
    return _execute_tool_calls_impl(tool_calls, handlers, events=events)


TEAM = TeammateManager(
    TEAM_DIR,
    bus=BUS,
    workdir=WORKDIR,
    child_tools=CHILD_TOOLS,
    llm_model=LLM_MODEL,
    call_generation_nonstream=_call_generation_nonstream,
    assistant_message_and_finish_reason=_assistant_message_and_finish_reason,
    normalize_assistant_message_for_dashscope=_normalize_assistant_message_for_dashscope,
    message_content_as_str=_message_content_as_str,
    tool_names_in_assistant_calls=_tool_names_in_assistant_calls,
    execute_tool_calls=execute_tool_calls,
    build_actor_tool_handlers=lambda actor: build_actor_tool_handlers(actor),
    scan_unclaimed_tasks=scan_unclaimed_tasks,
    claim_task=claim_task,
    make_identity_block=make_identity_block,
    poll_interval=POLL_INTERVAL,
    idle_timeout=IDLE_TIMEOUT,
)


SUBAGENT_MAX_TURNS = 24


def invoke_run_subtask(**kw) -> str:
    prompt = (kw.get("prompt") or "").strip()
    if not prompt:
        return "Error: run_subtask requires a non-empty prompt"
    title = (kw.get("title") or "").strip()
    return run_subagent(prompt, title=title)


def run_subagent(prompt: str, title: str = "") -> str:
    """
    Inner agent loop using DashScope message + tool_calls (same shape as the parent).
    Fresh messages; only CHILD_TOOLS; context is discarded after return.
    """
    label = title or "subtask"
    sub_messages: list = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    handlers = build_actor_tool_handlers("subagent")
    last_text = ""
    print(f"\033[36m--- subagent: {label} ---\033[0m")
    for turn in range(SUBAGENT_MAX_TURNS):
        response = _call_generation_nonstream(
            model=LLM_MODEL,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=6000,
            result_format="message",
            _llm_log_context="subagent",
            _llm_user_progress=True,
        )
        if response is None:
            print(f"\033[31m[{label}] no API response\033[0m")
            return f"[{label}] Subagent error: no response from API"
        if response.status_code != 200:
            err = f"{response.code} {getattr(response, 'message', '')}"
            print(f"\033[31m[{label}] {err}\033[0m")
            return f"[{label}] Subagent API error: {err}"

        assistant_msg, finish_reason = _assistant_message_and_finish_reason(response)
        assistant_msg = _normalize_assistant_message_for_dashscope(assistant_msg)
        last_text = _message_content_as_str(assistant_msg)
        logger.info("Subagent turn %s finish_reason=%s", turn, finish_reason)
        sub_messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            print(f"\033[36m--- end subagent: {label} ---\033[0m")
            return last_text if last_text.strip() else "(subagent returned no text)"

        tool_results = execute_tool_calls(tool_calls, handlers, events=None)
        sub_messages.extend(tool_results)

    print(f"\033[33m[{label}] subagent max turns ({SUBAGENT_MAX_TURNS})\033[0m")
    return (last_text or "") + f"\n...[subagent stopped after {SUBAGENT_MAX_TURNS} tool rounds]"


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
    notifs = BG.drain_notifications()
    if notifs:
        notif_text = "\n".join(
            f"[bg:{n['task_id']}] {n['result']}" for n in notifs)
        state.messages.append({"role": "user",
            "content": f"<background-results>\n{notif_text}\n"
                       f"</background-results>"})

    handlers = build_tool_handlers(state.todo)
    if estimate_context_size(state.messages) > CONTEXT_LIMIT:
        print("[auto compact]")
        state.messages[:] = compact_history(
            state.messages, state.compact, llm_progress=(events is None)
        )
    response = _call_generation_nonstream(
        model=LLM_MODEL,  # or qwen-turbo, qwen-max, etc.
        messages=_messages_for_llm(state.messages, state.todo),
        tools=PARENT_TOOLS,
        max_tokens=8000,
        result_format="message",
        _llm_log_context="main_agent",
        _llm_user_progress=(events is None),
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
    # elif text:
        # sys.stdout.write(text + "\n")
        # sys.stdout.flush()

    state.messages.append(assistant_msg)

    tool_calls = assistant_msg.get("tool_calls")
    if not tool_calls:
        state.last_tool_fingerprint = None
        state.repeated_tool_call_count = 0
        state.transition_reason = None
        return False

    fp = _tool_calls_fingerprint(tool_calls)
    if fp and fp == state.last_tool_fingerprint:
        state.repeated_tool_call_count += 1
    else:
        state.last_tool_fingerprint = fp
        state.repeated_tool_call_count = 1

    if state.repeated_tool_call_count >= 4:
        warn = (
            "Agent stopped: repeated identical tool calls detected "
            f"({state.repeated_tool_call_count}x)."
        )
        if events is not None:
            events.append({"type": "warning", "message": warn})
        else:
            print(f"\033[33m{warn}\033[0m")
        state.messages.append({"role": "assistant", "content": warn})
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

    manual_compact = False
    compact_focus = None
    for tool_message in tool_messages:
        if tool_message.get("name") == "compact":
            manual_compact = True
            compact_focus = tool_message.get("focus")
            break

    state.messages.extend(tool_messages)
    if not used_todo:
        reminder = state.todo.reminder()
        if reminder:
            state.messages.append({"role": "user", "content": reminder})
    state.turn_count += 1
    state.transition_reason = "tool_result"

    if manual_compact:
        print("[manual compact]")
        state.messages[:] = compact_history(
            state.messages,
            state.compact,
            focus=compact_focus,
            llm_progress=(events is None),
        )
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

    app = _FastAPI(title="s12_worktree_isolation", version="1.0.0")
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

    # Set log file name to match the current Python file name, but in logs dir
    _log_filename = os.path.splitext(os.path.basename(__file__))[0] + ".log"
    _log_path = _logs_dir / _log_filename
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            # logging.StreamHandler(sys.stderr),
            logging.FileHandler(_log_path, encoding="utf-8"),
        ],
        force=True,
    )
    history: list = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue
        if query.strip() == "/tasks":
            print(TASKS.list_all())
            continue
        if query.strip() == "/worktrees":
            print(WORKTREES.list_all())
            continue
        if query.strip() == "/events":
            print(EVENTS.list_recent(20))
            continue

        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)
        print()