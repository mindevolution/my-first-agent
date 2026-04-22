import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


_WORKTREE_LOCK = threading.Lock()


def detect_repo_root(cwd: Path) -> Path:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            root = Path(r.stdout.strip())
            if root.exists():
                return root
    except Exception:
        pass
    return cwd


class EventBus:
    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def emit(
        self,
        event: str,
        task: dict | None = None,
        worktree: dict | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]
        items = []
        for line in recent:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2, ensure_ascii=False)


class WorktreeManager:
    def __init__(self, repo_root: Path, store_dir: Path, tasks: Any, events: EventBus):
        self.repo_root = repo_root
        self.store_dir = store_dir
        self.tasks = tasks
        self.events = events
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2), encoding="utf-8")

    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, data: dict) -> None:
        self.index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find(self, name: str) -> dict | None:
        for wt in self._load_index().get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _run_git(self, args: list[str]) -> str:
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError((r.stdout + r.stderr).strip() or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def create(self, name: str, task_id: int | None = None, base_ref: str = "HEAD") -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError("Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -")
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        wt_path = self.store_dir / name
        branch = f"wt/{name}"
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )
        with _WORKTREE_LOCK:
            try:
                self._run_git(["worktree", "add", "-b", branch, str(wt_path), base_ref])
                entry = {
                    "name": name,
                    "path": str(wt_path),
                    "branch": branch,
                    "task_id": task_id,
                    "status": "active",
                    "created_at": time.time(),
                }
                idx = self._load_index()
                idx["worktrees"].append(entry)
                self._save_index(idx)
                if task_id is not None:
                    self.tasks.bind_worktree(task_id, name)
                self.events.emit(
                    "worktree.create.after",
                    task={"id": task_id} if task_id is not None else {},
                    worktree={"name": name, "path": str(wt_path), "branch": branch, "status": "active"},
                )
                return json.dumps(entry, indent=2, ensure_ascii=False)
            except Exception as e:
                self.events.emit(
                    "worktree.create.failed",
                    task={"id": task_id} if task_id is not None else {},
                    worktree={"name": name, "base_ref": base_ref},
                    error=str(e),
                )
                raise

    def list_all(self) -> str:
        wts = self._load_index().get("worktrees", [])
        if not wts:
            return "No worktrees in index."
        lines = []
        for wt in wts:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") is not None else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt.get('name')} -> {wt.get('path')} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return (r.stdout + r.stderr).strip() or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def keep(self, name: str) -> str:
        idx = self._load_index()
        kept = None
        with _WORKTREE_LOCK:
            for wt in idx.get("worktrees", []):
                if wt.get("name") == name:
                    wt["status"] = "kept"
                    wt["kept_at"] = time.time()
                    kept = wt
                    break
            if kept is None:
                return f"Error: Unknown worktree '{name}'"
            self._save_index(idx)
        self.events.emit(
            "worktree.keep",
            task={"id": kept.get("task_id")} if kept.get("task_id") is not None else {},
            worktree={"name": name, "path": kept.get("path"), "status": "kept"},
        )
        return json.dumps(kept, indent=2, ensure_ascii=False)

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        try:
            with _WORKTREE_LOCK:
                args = ["worktree", "remove"]
                if force:
                    args.append("--force")
                args.append(wt["path"])
                self._run_git(args)

                if complete_task and wt.get("task_id") is not None:
                    task_id = int(wt["task_id"])
                    self.tasks.update(task_id, status="completed")
                    self.tasks.unbind_worktree(task_id)
                    self.events.emit(
                        "task.completed",
                        task={"id": task_id, "status": "completed"},
                        worktree={"name": name},
                    )

                idx = self._load_index()
                for item in idx.get("worktrees", []):
                    if item.get("name") == name:
                        item["status"] = "removed"
                        item["removed_at"] = time.time()
                        break
                self._save_index(idx)
            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

