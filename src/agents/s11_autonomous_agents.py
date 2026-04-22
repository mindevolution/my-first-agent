#!/usr/bin/env python3
# Harness: persistent tasks -- goals that outlive any single conversation.
"""
s11_autonomous_agents.py - Autonomous Agents

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
_claim_lock = threading.Lock()

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
    unclaimed: list[dict] = []
    for path in sorted(TASKS_DIR.glob("task_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            task.get("status") == "pending"
            and not task.get("owner")
            and not task.get("blockedBy")
        ):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text(encoding="utf-8"))
        if task.get("owner"):
            return f"Error: Task {task_id} already claimed by {task.get('owner')}"
        if task.get("status") != "pending":
            return f"Error: Task {task_id} status is '{task.get('status')}'"
        if task.get("blockedBy"):
            return f"Error: Task {task_id} is blocked by {task.get('blockedBy')}"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Claimed task #{task_id} for {owner}"


def make_identity_block(name: str, role: str, team_name: str) -> str:
    return (
        f"<identity>You are '{name}', role: {role}, team: {team_name}. "
        "Continue your autonomous work.</identity>"
    )


# -- TeammateManager: persistent named agents with config.json --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.dir / "config.json"
        self._lock = threading.Lock()
        self.threads: dict[str, threading.Thread] = {}
        self.stop_events: dict[str, threading.Event] = {}
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        return {"team_name": "default", "members": []}

    def _save_config_locked(self) -> None:
        self.config_path.write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _find_member_locked(self, name: str) -> dict | None:
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    def _set_member_state(self, name: str, *, role: str | None = None, status: str | None = None) -> None:
        with self._lock:
            member = self._find_member_locked(name)
            if member is None:
                member = {"name": name, "role": role or "teammate", "status": status or "idle"}
                self.config["members"].append(member)
            if role is not None:
                member["role"] = role
            if status is not None:
                member["status"] = status
            self._save_config_locked()

    def _ensure_thread(self, name: str, role: str, prompt: str) -> None:
        t = self.threads.get(name)
        if t and t.is_alive():
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt, stop_event),
            daemon=True,
        )
        self.stop_events[name] = stop_event
        self.threads[name] = thread
        thread.start()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        self._set_member_state(name, role=role, status="working")
        self._ensure_thread(name, role, prompt)
        BUS.send("lead", name, prompt, "message", extra={"source": "spawn"})
        return f"Spawned '{name}' (role: {role})"

    def _inject_identity_if_needed(self, messages: list, name: str, role: str) -> None:
        if len(messages) > 3:
            return
        team_name = self.config.get("team_name", "default")
        messages.insert(0, {"role": "user", "content": make_identity_block(name, role, team_name)})
        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing autonomous work."})

    def _try_claim_next_task(self, name: str) -> dict | None:
        for task in scan_unclaimed_tasks():
            result = claim_task(int(task["id"]), name)
            if result.startswith("Error:"):
                continue
            return task
        return None

    def _idle_poll(self, name: str, role: str, messages: list, stop_event: threading.Event) -> bool:
        polls = max(1, IDLE_TIMEOUT // max(POLL_INTERVAL, 1))
        for _ in range(polls):
            if stop_event.wait(POLL_INTERVAL):
                return False
            inbox = BUS.read_inbox(name)
            if inbox:
                self._inject_identity_if_needed(messages, name, role)
                messages.append({
                    "role": "user",
                    "content": f"<inbox>{json.dumps(inbox, ensure_ascii=False)}</inbox>",
                })
                return True
            task = self._try_claim_next_task(name)
            if task:
                self._inject_identity_if_needed(messages, name, role)
                messages.append({
                    "role": "user",
                    "content": (
                        f"<auto-claimed>Task #{task['id']}: {task.get('subject', '')}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    ),
                })
                return True
        return False

    def _teammate_loop(self, name: str, role: str, initial_prompt: str, stop_event: threading.Event) -> None:
        team_name = self.config.get("team_name", "default")
        sys_prompt = (
            f"You are teammate '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            "Use idle when you have no immediate work. "
            "You can claim_task to claim unowned pending tasks."
        )
        messages: list = [{"role": "user", "content": initial_prompt}]
        handlers = build_actor_tool_handlers(name)

        while not stop_event.is_set():
            self._set_member_state(name, role=role, status="working")
            idle_requested = False

            # -- WORK phase --
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})

                response = _call_generation_nonstream(
                    model=LLM_MODEL,
                    messages=[{"role": "system", "content": sys_prompt}, *messages],
                    tools=CHILD_TOOLS,
                    max_tokens=6000,
                    result_format="message",
                    _llm_log_context="subagent",
                    _llm_user_progress=False,
                )
                if response is None:
                    BUS.send(name, "lead", f"{name} got no API response", "message")
                    idle_requested = True
                    break
                if response.status_code != 200:
                    BUS.send(
                        name,
                        "lead",
                        f"{name} API error: {response.code} {getattr(response, 'message', '')}",
                        "message",
                    )
                    idle_requested = True
                    break

                assistant_msg, _ = _assistant_message_and_finish_reason(response)
                assistant_msg = _normalize_assistant_message_for_dashscope(assistant_msg)
                text = _message_content_as_str(assistant_msg).strip()
                messages.append(assistant_msg)
                if text:
                    BUS.send(name, "lead", text, "message")

                tool_calls = assistant_msg.get("tool_calls")
                if not tool_calls:
                    idle_requested = True
                    break

                tool_names = _tool_names_in_assistant_calls(tool_calls)
                tool_messages = execute_tool_calls(tool_calls, handlers, events=None)
                if tool_messages:
                    messages.extend(tool_messages)
                if "idle" in tool_names:
                    idle_requested = True
                    break

            # -- IDLE phase --
            self._set_member_state(name, role=role, status="idle")
            if not idle_requested:
                idle_requested = True
            if idle_requested and self._idle_poll(name, role, messages, stop_event):
                continue
            self._set_member_state(name, role=role, status="shutdown")
            return

        self._set_member_state(name, role=role, status="shutdown")

    def list_all(self) -> str:
        with self._lock:
            members = list(self.config.get("members", []))
            team_name = self.config.get("team_name", "default")
        if not members:
            return "No teammates."
        lines = [f"Team: {team_name}"]
        for m in members:
            lines.append(f"  {m.get('name', '?')} ({m.get('role', '?')}): {m.get('status', 'unknown')}")
        return "\n".join(lines)

    def member_names(self) -> list[str]:
        with self._lock:
            return [m.get("name", "") for m in self.config.get("members", []) if m.get("name")]


# -- TaskManager: CRUD with dependency graph, persisted as JSON files --
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id, "subject": subject, "description": description,
            "status": "pending", "blockedBy": [], "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if remove_blocked_by:
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int):
        """Remove completed_id from all other tasks' blockedBy lists."""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        tasks = []
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda f: int(f.stem.split("_")[1])
        )
        for f in files:
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            owner = f" @{t['owner']}" if t.get("owner") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{blocked}")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)
TEAM = TeammateManager(TEAM_DIR)


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
    {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "Load the full body of a named skill into the current context.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                },
            },
            "required": ["name"],
        },
    },
},
    {
    "type": "function",
    "function": {
        "name": "compact",
        "description": "Summarize earlier conversation so work can continue in a smaller context.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "The name of the skill to focus on.",
                },
            },
            "required": [],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "task_create",
        "description": "Create a new task.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The subject of the task.",
                },
                "description": {
                    "type": "string",
                    "description": "The description of the task.",
                },
            },
            "required": ["subject"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "task_update",
        "description": "Update a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The ID of the task to update.",
                },
                "status": {
                    "type": "string",
                    "description": "The status of the task.",
                    "enum": ["pending", "in_progress", "completed"],
                },
                "addBlockedBy": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "description": "The ID of the task to add to the blockedBy list.",
                    },
                    "description": "The IDs of the tasks to add to the blockedBy list.",
                },
                "removeBlockedBy": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "description": "The ID of the task to remove from the blockedBy list.",
                    },
                    "description": "The IDs of the tasks to remove from the blockedBy list.",
                },
            },
            "required": ["task_id", "status", "addBlockedBy", "removeBlockedBy"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "task_list",
        "description": "List all tasks.",
        "parameters": {
            "type": "object",
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "task_get",
        "description": "Get a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The ID of the task to get.",
                },
            },
            "required": ["task_id"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "background_run",
        "description": "Run a background task.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run.",
                },
            },
            "required": ["command"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "check_background",
        "description": "Check the status of a background task.",
        "parameters": {
            "type": "object",
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "spawn_teammate",
        "description": "Spawn a persistent teammate that runs in its own thread.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Teammate name, e.g. 'alice'.",
                },
                "role": {
                    "type": "string",
                    "description": "Teammate role, e.g. 'coder' or 'researcher'.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Initial task prompt for the teammate.",
                },
            },
            "required": ["name", "role", "prompt"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "list_teammates",
        "description": "List all teammates with role and status.",
        "parameters": {
            "type": "object",
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Send a message to a teammate inbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient teammate name.",
                },
                "content": {
                    "type": "string",
                    "description": "Message content.",
                },
                "msg_type": {
                    "type": "string",
                    "enum": sorted(VALID_MSG_TYPES),
                    "description": "Message type (default: message).",
                },
            },
            "required": ["to", "content"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "read_inbox",
        "description": "Read and drain the current agent inbox.",
        "parameters": {
            "type": "object",
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "broadcast",
        "description": "Broadcast a message to all teammates.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Message content.",
                },
            },
            "required": ["content"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "shutdown_request",
        "description": "Lead requests teammate graceful shutdown. Returns request_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "teammate": {
                    "type": "string",
                    "description": "Teammate name to request shutdown from.",
                },
            },
            "required": ["teammate"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "shutdown_response",
        "description": "Respond to shutdown request (teammate) or check request status (lead).",
        "parameters": {
            "type": "object",
            "properties": {
                "request_id": {
                    "type": "string",
                    "description": "Protocol request_id to correlate request/response.",
                },
                "approve": {
                    "type": "boolean",
                    "description": "Teammate decision for shutdown_request.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional response reason.",
                },
            },
            "required": ["request_id"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "plan_approval",
        "description": "Submit plan for approval (teammate) or review plan request (lead).",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "Plan text when teammate requests approval.",
                },
                "request_id": {
                    "type": "string",
                    "description": "Existing request_id when lead approves/rejects.",
                },
                "approve": {
                    "type": "boolean",
                    "description": "Lead approval decision for request_id.",
                },
                "feedback": {
                    "type": "string",
                    "description": "Optional lead feedback.",
                },
            },
            "required": [],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "idle",
        "description": "Signal no immediate work and enter idle polling.",
        "parameters": {
            "type": "object",
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "claim_task",
        "description": "Claim a task from the task board by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "Task ID to claim.",
                },
            },
            "required": ["task_id"],
        },
    },
}
]

PARENT_TOOLS = CHILD_TOOLS + [
{
    "type": "function",
    "function": {
        "name": "run_subtask",
        "description": (
            "Run a sub-investigation in a fresh context (no parent chat history). "
            "The subagent has bash, read_file, write_file, edit_file, task_create, task_update, task_list, task_get only—no. "
            "Use for isolated questions (e.g. detect test framework, scan one directory). "
            "Returns a text summary to incorporate into your answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Specific task for the subagent (what to find or do).",
                },
                "title": {
                    "type": "string",
                    "description": "Short label for logs (optional).",
                },
            },
            "required": ["prompt"],
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

        results.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": name,
            "content": out if isinstance(out, str) else str(out),
        })
    return results


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

    app = _FastAPI(title="s11_autonomous_agents", version="1.0.0")
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
            query = input("\033[36ms11 >> \033[0m")
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

        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)
        print()