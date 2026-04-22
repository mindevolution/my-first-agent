import json
import threading
from pathlib import Path


_CLAIM_LOCK = threading.Lock()


def scan_unclaimed_tasks(tasks_dir: Path) -> list[dict]:
    unclaimed: list[dict] = []
    for path in sorted(tasks_dir.glob("task_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            task.get("status") == "pending"
            and not task.get("owner")
            and not task.get("blockedBy")
            and not task.get("worktree")
        ):
            unclaimed.append(task)
    return unclaimed


def claim_task(tasks_dir: Path, task_id: int, owner: str) -> str:
    with _CLAIM_LOCK:
        path = tasks_dir / f"task_{task_id}.json"
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

