"""
树状任务追踪系统

任务 ID 格式：T1, T1.1, T1.2, T1.1.1 ...
支持状态管理、进度记录、父子关系。
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import MEMORY_DIR

TASKS_FILE = MEMORY_DIR / "tasks.json"
TASKS_DIR = MEMORY_DIR / "tasks"
TASKS_DIR.mkdir(exist_ok=True)


def _load_all() -> dict:
    if TASKS_FILE.exists():
        try:
            return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        except:
            return {}
    return {}


def _save_all(tasks: dict):
    TASKS_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_id(tasks: dict) -> str:
    """生成下一个顶级任务 ID"""
    existing = [int(k[1:].split(".")[0]) for k in tasks.keys() if k.startswith("T") and "." not in k]
    return f"T{max(existing) + 1}" if existing else "T1"


def create_task(description: str, parent_id: Optional[str] = None) -> str:
    """创建任务"""
    tasks = _load_all()
    if parent_id and parent_id in tasks:
        children = [k for k in tasks if k.startswith(f"{parent_id}.")]
        child_nums = [int(k.split(".")[-1]) for k in children if k.split(".")[-1].isdigit()]
        task_id = f"{parent_id}.{max(child_nums) + 1}" if child_nums else f"{parent_id}.1"
    else:
        task_id = _next_id(tasks)

    tasks[task_id] = {
        "id": task_id,
        "description": description,
        "status": "pending",
        "parent": parent_id or None,
        "children": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    if parent_id and parent_id in tasks:
        if task_id not in tasks[parent_id]["children"]:
            tasks[parent_id]["children"].append(task_id)

    _save_all(tasks)
    _write_progress(task_id, f"任务创建: {description}")
    return task_id


def update_task(task_id: str, status: Optional[str] = None, description: Optional[str] = None):
    """更新任务状态"""
    tasks = _load_all()
    if task_id not in tasks:
        return False
    if status:
        tasks[task_id]["status"] = status
    if description:
        tasks[task_id]["description"] = description
    tasks[task_id]["updated_at"] = datetime.now().isoformat()
    _save_all(tasks)
    if status:
        _write_progress(task_id, f"状态变更: {status}")
    return True


def get_task(task_id: str) -> Optional[dict]:
    tasks = _load_all()
    return tasks.get(task_id)


def list_tasks(status: Optional[str] = None) -> list[dict]:
    """列出任务，按创建时间排序"""
    tasks = _load_all()
    result = sorted(tasks.values(), key=lambda t: t["created_at"])
    if status:
        result = [t for t in result if t["status"] == status]
    return result


def get_active_tasks() -> list[dict]:
    """获取进行中的任务"""
    return list_tasks(status="in_progress")


def get_task_tree() -> str:
    """获取格式化的任务树文本"""
    tasks = _load_all()
    lines = []
    for tid in sorted(tasks.keys(), key=lambda x: [int(n) if n.isdigit() else n for n in x.replace("T", "").split(".")]):
        task = tasks[tid]
        depth = tid.count(".")
        indent = "  " * depth
        status_icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅", "failed": "❌"}.get(task["status"], "⬜")
        lines.append(f"{indent}{status_icon} {tid}: {task['description']}")
    return "\n".join(lines)


def _write_progress(task_id: str, content: str):
    task_file = TASKS_DIR / f"{task_id}.md"
    with open(task_file, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{content}\n")


def read_progress(task_id: str) -> str:
    task_file = TASKS_DIR / f"{task_id}.md"
    if task_file.exists():
        return task_file.read_text(encoding="utf-8")
    return ""
