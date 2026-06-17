"""
LoveFlow Agent 主循环（模块化重构）

流程：
  1. 用户输入
  2. 搜索记忆 → 注入模块上下文 → 构建 prompt
  3. 调用 LoveFlow 推理
  4. 解析响应中的动作标记
  5. 执行动作（记忆/任务/训练/监控/代码等）
  6. 保存状态
  7. 输出结果

动作标记系统可扩展——模块可注册自己的 handler。
"""
import importlib
import logging
import re
import sys
from datetime import datetime
from typing import Callable, Optional

from agent.loveflow_client import LoveFlowClient
from agent.core.context import build_agent_context
from agent.core.system_prompt import get_system_prompt
from agent.core.memory import add_memory, append_note, sync_to_markdown
from agent.core.task_tracker import create_task, update_task
from agent.core.checkpoint import save_checkpoint

logger = logging.getLogger("loveflow.engine")


class ActionRegistry:
    """动作标记注册器——支持模块扩展"""

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, tag: str, handler: Callable):
        """注册动作处理器"""
        self._handlers[tag] = handler
        logger.debug(f"动作已注册: [{tag}] -> {handler.__name__}")

    def unregister(self, tag: str):
        """取消注册"""
        self._handlers.pop(tag, None)

    def parse(self, text: str) -> list[dict]:
        """解析文本中的所有动作标记"""
        actions = []
        for tag, handler in self._handlers.items():
            pattern = self._build_pattern(tag)
            for match in re.finditer(pattern, text, re.IGNORECASE):
                parsed = handler.parse(match)
                if parsed:
                    actions.append(parsed)
        return actions

    def execute(self, actions: list[dict]) -> list[str]:
        """执行动作列表"""
        results = []
        for action in actions:
            tag = action.get("_tag", "")
            handler = self._handlers.get(tag)
            if handler:
                try:
                    result = handler.execute(action)
                    results.append(result)
                except Exception as e:
                    results.append(f"[{tag}] 动作失败: {e}")
                    logger.error(f"动作执行失败 [{tag}]: {e}")
        return results

    def _build_pattern(self, tag: str) -> str:
        return rf'\[{tag}:\s*(.*?)\]'


class BaseActionHandler:
    """动作处理器基类"""

    tag = ""

    @classmethod
    def parse(cls, match: re.Match) -> Optional[dict]:
        """从正则匹配中解析动作数据"""
        return None

    @classmethod
    def execute(cls, action: dict) -> str:
        """执行动作，返回结果描述"""
        return ""


# ==================== 内置动作处理器 ====================


class MemoryHandler(BaseActionHandler):
    """[MEMORY: 内容] → 保存记忆"""

    tag = "MEMORY"

    @classmethod
    def parse(cls, match):
        return {"_tag": "MEMORY", "content": match.group(1).strip()}

    @classmethod
    def execute(cls, action):
        add_memory(action["content"], "knowledge", action["content"][:50])
        return f"已记忆: {action['content'][:50]}..."


class TaskHandler(BaseActionHandler):
    """[TASK: 描述] → 创建任务"""

    tag = "TASK"

    @classmethod
    def parse(cls, match):
        return {"_tag": "TASK", "content": match.group(1).strip()}

    @classmethod
    def execute(cls, action):
        task_id = create_task(action["content"])
        return f"已创建任务 {task_id}: {action['content'][:50]}..."


class DoneHandler(BaseActionHandler):
    """[DONE: T1] → 完成任务"""

    tag = "DONE"

    @classmethod
    def parse(cls, match):
        return {"_tag": "DONE", "task_id": match.group(1).strip()}

    @classmethod
    def execute(cls, action):
        update_task(action["task_id"], status="completed")
        return f"已完成任务 {action['task_id']}"


class NoteHandler(BaseActionHandler):
    """[NOTE: 内容] → 保存笔记"""

    tag = "NOTE"

    @classmethod
    def parse(cls, match):
        return {"_tag": "NOTE", "content": match.group(1).strip()}

    @classmethod
    def execute(cls, action):
        append_note(action["content"])
        return "已保存笔记"


# ==================== 引擎主类 ====================


class LoveFlowAgent:
    """Agent 主引擎"""

    def __init__(self):
        self.client = LoveFlowClient()
        self.history: list[dict] = []
        self.action_registry = ActionRegistry()
        self.extra_capabilities: list[str] = []
        self._modules: dict[str, object] = {}

        # 注册内置动作处理器
        self._register_builtin_actions()

    def _register_builtin_actions(self):
        """注册内置动作"""
        self.action_registry.register("MEMORY", MemoryHandler)
        self.action_registry.register("TASK", TaskHandler)
        self.action_registry.register("DONE", DoneHandler)
        self.action_registry.register("NOTE", NoteHandler)

    def register_action(self, tag: str, handler: BaseActionHandler):
        """注册扩展动作"""
        self.action_registry.register(tag, handler)

    def register_module(self, name: str, instance: object, capabilities: list[str] = None):
        """
        注册外部模块

        Args:
            name: 模块名
            instance: 模块实例
            capabilities: 能力描述列表（注入系统提示词）
        """
        self._modules[name] = instance
        if capabilities:
            self.extra_capabilities.extend(capabilities)
            logger.info(f"模块已注册: {name}")

    def get_module(self, name: str) -> Optional[object]:
        """获取已注册的模块"""
        return self._modules.get(name)

    def interact(self, user_input: str, verbose: bool = False) -> str:
        """
        单次交互

        Args:
            user_input: 用户输入
            verbose: 是否显示详细信息

        Returns:
            Agent 响应文本
        """
        # 1) 收集模块上下文
        extra_contexts = []
        for mod_name, mod_instance in self._modules.items():
            if hasattr(mod_instance, "get_context"):
                try:
                    ctx = mod_instance.get_context()
                    if ctx:
                        extra_contexts.append((ctx, getattr(mod_instance, "context_priority", 10)))
                except Exception as e:
                    logger.warning(f"模块 [{mod_name}] 上下文获取失败: {e}")

        # 2) 构建上下文
        context_text, stats = build_agent_context(
            user_input,
            budget=2048,
            extra_contexts=extra_contexts,
        )
        if verbose:
            print(f"\n[上下文统计] 记忆:{stats['memory']} 检查点:{stats['checkpoint']} 任务:{stats['tasks']}", file=sys.stderr)

        # 3) 构建 prompt
        system_prompt = get_system_prompt(self.extra_capabilities if self.extra_capabilities else None)
        messages = [{"role": "system", "content": system_prompt}]

        if context_text:
            messages.append({"role": "system", "content": f"当前上下文:\n{context_text}"})

        messages.append({"role": "user", "content": user_input})

        # 4) 调用 LoveFlow 推理
        response = self.client.chat(messages, max_tokens=512, temperature=0.4)
        if not response or response.startswith("[推理错误"):
            return response or "[无响应]"

        # 5) 解析动作
        actions = self.action_registry.parse(response)
        action_results = self.action_registry.execute(actions)

        # 6) 清理响应
        clean = self._clean_response(response)

        # 7) 保存检查点（每 5 轮）
        self.history.append({"input": user_input, "response": clean})
        if len(self.history) % 5 == 0:
            active_tasks = [t for t in __import__('agent.core.task_tracker', fromlist=['']).task_tracker.list_tasks(status='in_progress')]
            save_checkpoint(
                session_summary=f"最近对话: {len(self.history)} 轮",
                active_tasks=active_tasks,
            )

        # 8) 同步记忆到磁盘（如果有记忆操作）
        if any(a.get("_tag") == "MEMORY" for a in actions):
            sync_to_markdown()

        result = clean
        if action_results and verbose:
            result += f"\n\n[动作: {'; '.join(action_results)}]"
        elif action_results:
            result += f"\n\n*{' | '.join(action_results)}*"

        return result

    def _clean_response(self, text: str) -> str:
        """移除动作标记，只保留文本"""
        # 构建所有已注册 tag 的正则
        tags = "|".join(self.action_registry._handlers.keys())
        text = re.sub(rf'\[(?:{tags}):.*?\]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
