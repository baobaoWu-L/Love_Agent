"""
会话检查点系统

保存/恢复会话状态，支持：
  - 当前任务上下文
  - 活跃的任务树
  - 关键记忆引用
  - 会话摘要
"""
import json
import time
from datetime import datetime

from config.agent_config import MEMORY_DIR

CHECKPOINT_PATH = MEMORY_DIR / "checkpoint.json"


def save_checkpoint(
    session_summary: str = "",
    active_tasks: list[dict] = None,
    context_refs: list[str] = None,
    metadata: dict = None,
):
    """保存检查点"""
    checkpoint = {
        "version": "0.2.0",
        "timestamp": datetime.now().isoformat(),
        "unix_time": time.time(),
        "session_summary": session_summary,
        "active_tasks": active_tasks or [],
        "context_refs": context_refs or [],
        "metadata": metadata or {},
    }
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint() -> dict | None:
    """加载最新检查点"""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except:
            return None
    return None


def get_session_context() -> str:
    """获取格式化的会话上下文文本"""
    ckpt = load_checkpoint()
    if not ckpt:
        return ""

    parts = []
    if ckpt.get("session_summary"):
        parts.append(f"【上次会话摘要】\n{ckpt['session_summary']}")

    if ckpt.get("active_tasks"):
        tasks_text = "\n".join(
            f"  {t['id']}: {t['description']} ({t.get('status', 'unknown')})"
            for t in ckpt["active_tasks"]
        )
        parts.append(f"【进行中的任务】\n{tasks_text}")

    return "\n\n".join(parts)


def clear_checkpoint():
    """清空检查点"""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
