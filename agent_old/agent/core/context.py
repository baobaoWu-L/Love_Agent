"""
上下文管理：预算化注入（增强版）

从不同源收集上下文，按 token budget 裁剪，按重要性排序注入。
支持模块上下文注入（感知数据、训练状态等）。
"""
from agent.core.memory import search_memory, list_memories
from agent.core.checkpoint import load_checkpoint, get_session_context
from agent.core.task_tracker import get_task_tree, get_active_tasks

_CHARS_PER_TOKEN = 2


def estimate_tokens(text: str) -> int:
    """估算文本 token 数"""
    return len(text) // _CHARS_PER_TOKEN


def truncate_by_budget(text: str, budget: int) -> str:
    """按 token budget 截断"""
    return text[:budget * _CHARS_PER_TOKEN]


def build_agent_context(
    user_input: str,
    budget: int = 2048,
    include_memory: bool = True,
    include_checkpoint: bool = True,
    include_tasks: bool = True,
    extra_contexts: list[tuple[str, int]] = None,
) -> tuple[str, dict]:
    """
    构建 Agent 上下文

    extra_contexts: 额外的上下文片段列表，每项为 (text, priority)
                    优先级越低越优先注入（0=最高）

    Returns:
        (context_text, stats)
    """
    parts = []
    stats = {"memory": 0, "checkpoint": 0, "tasks": 0, "extra": 0, "total_tokens": 0}
    remaining = budget

    # 1) 检查点（最高优先级）
    if include_checkpoint:
        ckpt_text = get_session_context()
        ckpt_tokens = estimate_tokens(ckpt_text)
        if ckpt_tokens <= remaining:
            if ckpt_text:
                parts.append(ckpt_text)
            remaining -= ckpt_tokens
            stats["checkpoint"] = ckpt_tokens

    # 2) 额外上下文（按优先级排序）
    if extra_contexts:
        sorted_extras = sorted(extra_contexts, key=lambda x: x[1])
        for extra_text, _ in sorted_extras:
            extra_tokens = estimate_tokens(extra_text)
            if extra_tokens <= remaining:
                parts.append(extra_text)
                remaining -= extra_tokens
                stats["extra"] += extra_tokens

    # 3) 记忆搜索（只在记忆/回忆相关指令时触发）
    if include_memory:
        memory_keywords = ["记得", "记忆", "回忆", "之前", "忘记", "忘了", "记住", "memor"]
        if any(kw in user_input.lower() for kw in memory_keywords):
            memory_results = search_memory(user_input, limit=5)
        else:
            memory_results = []
        memory_text = ""
        for r in memory_results:
            memory_text += f"● [{r['importance']}] {r['title']}: {r['content'][:200]}\n"
        memory_tokens = estimate_tokens(memory_text)
        if memory_tokens <= remaining:
            if memory_text:
                parts.append(f"【相关记忆】\n{memory_text}")
            remaining -= memory_tokens
            stats["memory"] = memory_tokens

    # 4) 任务列表
    if include_tasks:
        task_text = get_task_tree()
        if task_text:
            task_header = "【任务列表】\n"
            task_full = task_header + task_text
            task_tokens = estimate_tokens(task_full)
            if task_tokens <= remaining:
                parts.append(task_full)
                remaining -= task_tokens
                stats["tasks"] = task_tokens

    stats["total_tokens"] = budget - remaining
    return "\n\n".join(parts), stats


def get_system_prompt(extra_capabilities: list[str] = None) -> str:
    """获取系统提示词（可注入额外能力描述）"""
    base_prompt = """你是 LoveFlow Agent，一个基于 LoveFlow 模型的智能 AI 助手。

你有以下核心能力：
1. 记忆对话中的重要信息（用 [MEMORY: 内容] 标记保存）
2. 追踪任务进度（用 [TASK: 描述] 创建，[DONE: T1] 完成）
3. 管理项目知识（用 [NOTE: 内容] 保存笔记）
4. 感知系统环境（CPU、内存、磁盘、进程等）
5. 编排大模型训练（启动训练、查看进度、管理数据）

规则：
- 每次回答尽量简洁
- 涉及知识性的内容请记入长期记忆
- 使用动作标记与系统交互"""

    if extra_capabilities:
        caps_text = "\n".join(f"  - {cap}" for cap in extra_capabilities)
        base_prompt += f"\n\n当前额外可用能力:\n{caps_text}"

    return base_prompt
