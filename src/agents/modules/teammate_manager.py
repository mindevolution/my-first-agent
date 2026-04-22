import json
import threading
from pathlib import Path


class TeammateManager:
    def __init__(
        self,
        team_dir: Path,
        *,
        bus,
        workdir: Path,
        child_tools: list,
        llm_model: str,
        call_generation_nonstream,
        assistant_message_and_finish_reason,
        normalize_assistant_message_for_dashscope,
        message_content_as_str,
        tool_names_in_assistant_calls,
        execute_tool_calls,
        build_actor_tool_handlers,
        scan_unclaimed_tasks,
        claim_task,
        make_identity_block,
        poll_interval: int,
        idle_timeout: int,
    ):
        self.dir = team_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.dir / "config.json"
        self._lock = threading.Lock()
        self.threads: dict[str, threading.Thread] = {}
        self.stop_events: dict[str, threading.Event] = {}
        self.config = self._load_config()

        self.bus = bus
        self.workdir = workdir
        self.child_tools = child_tools
        self.llm_model = llm_model
        self.call_generation_nonstream = call_generation_nonstream
        self.assistant_message_and_finish_reason = assistant_message_and_finish_reason
        self.normalize_assistant_message_for_dashscope = normalize_assistant_message_for_dashscope
        self.message_content_as_str = message_content_as_str
        self.tool_names_in_assistant_calls = tool_names_in_assistant_calls
        self.execute_tool_calls = execute_tool_calls
        self.build_actor_tool_handlers = build_actor_tool_handlers
        self.scan_unclaimed_tasks = scan_unclaimed_tasks
        self.claim_task = claim_task
        self.make_identity_block = make_identity_block
        self.poll_interval = poll_interval
        self.idle_timeout = idle_timeout

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
        self.bus.send("lead", name, prompt, "message", extra={"source": "spawn"})
        return f"Spawned '{name}' (role: {role})"

    def _inject_identity_if_needed(self, messages: list, name: str, role: str) -> None:
        if len(messages) > 3:
            return
        team_name = self.config.get("team_name", "default")
        messages.insert(0, {"role": "user", "content": self.make_identity_block(name, role, team_name)})
        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing autonomous work."})

    def _try_claim_next_task(self, name: str) -> dict | None:
        for task in self.scan_unclaimed_tasks():
            result = self.claim_task(int(task["id"]), name)
            if result.startswith("Error:"):
                continue
            return task
        return None

    def _idle_poll(self, name: str, role: str, messages: list, stop_event: threading.Event) -> bool:
        polls = max(1, self.idle_timeout // max(self.poll_interval, 1))
        for _ in range(polls):
            if stop_event.wait(self.poll_interval):
                return False
            inbox = self.bus.read_inbox(name)
            if inbox:
                self._inject_identity_if_needed(messages, name, role)
                messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, ensure_ascii=False)}</inbox>"})
                return True
            task = self._try_claim_next_task(name)
            if task:
                self._inject_identity_if_needed(messages, name, role)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<auto-claimed>Task #{task['id']}: {task.get('subject', '')}\n"
                            f"{task.get('description', '')}</auto-claimed>"
                        ),
                    }
                )
                return True
        return False

    def _teammate_loop(self, name: str, role: str, initial_prompt: str, stop_event: threading.Event) -> None:
        team_name = self.config.get("team_name", "default")
        sys_prompt = (
            f"You are teammate '{name}', role: {role}, team: {team_name}, at {self.workdir}. "
            "Use idle when you have no immediate work. "
            "You can claim_task to claim unowned pending tasks."
        )
        messages: list = [{"role": "user", "content": initial_prompt}]
        handlers = self.build_actor_tool_handlers(name)

        while not stop_event.is_set():
            self._set_member_state(name, role=role, status="working")
            idle_requested = False

            for _ in range(50):
                inbox = self.bus.read_inbox(name)
                for msg in inbox:
                    messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})

                response = self.call_generation_nonstream(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": sys_prompt}, *messages],
                    tools=self.child_tools,
                    max_tokens=6000,
                    result_format="message",
                    _llm_log_context="subagent",
                    _llm_user_progress=False,
                )
                if response is None:
                    self.bus.send(name, "lead", f"{name} got no API response", "message")
                    idle_requested = True
                    break
                if response.status_code != 200:
                    self.bus.send(
                        name,
                        "lead",
                        f"{name} API error: {response.code} {getattr(response, 'message', '')}",
                        "message",
                    )
                    idle_requested = True
                    break

                assistant_msg, _ = self.assistant_message_and_finish_reason(response)
                assistant_msg = self.normalize_assistant_message_for_dashscope(assistant_msg)
                text = self.message_content_as_str(assistant_msg).strip()
                messages.append(assistant_msg)
                if text:
                    self.bus.send(name, "lead", text, "message")

                tool_calls = assistant_msg.get("tool_calls")
                if not tool_calls:
                    idle_requested = True
                    break

                tool_names = self.tool_names_in_assistant_calls(tool_calls)
                tool_messages = self.execute_tool_calls(tool_calls, handlers, events=None)
                if tool_messages:
                    messages.extend(tool_messages)
                if "idle" in tool_names:
                    idle_requested = True
                    break

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

