"""
自我驱动调度器

基于 schedule 库，支持：
  - 定时任务（系统监控、训练检查、记忆整理）
  - 事件驱动（异常检测触发动作）
  - 任务持久化
"""
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("loveflow.scheduler")


class Scheduler:
    """轻量级任务调度器"""

    def __init__(self):
        self._tasks: list[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_queue: queue.Queue = queue.Queue()

    def add_task(self, name: str, interval: int, callback: Callable, unit: str = "seconds"):
        """
        添加定时任务

        Args:
            name: 任务名称
            interval: 间隔
            callback: 回调函数
            unit: 时间单位（seconds/minutes/hours）
        """
        self._tasks.append({
            "name": name,
            "interval": interval,
            "unit": unit,
            "callback": callback,
            "last_run": 0,
        })
        logger.info(f"调度任务已添加: {name} (每 {interval} {unit})")

    def trigger_event(self, event_type: str, data: dict = None):
        """触发事件"""
        self._event_queue.put({"type": event_type, "data": data or {}, "time": time.time()})

    def start(self):
        """启动调度器（后台线程）"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("调度器已启动")

    def stop(self):
        """停止调度器"""
        self._running = False
        logger.info("调度器已停止")

    def _run_loop(self):
        """调度主循环"""
        while self._running:
            now = time.time()

            # 执行定时任务
            for task in self._tasks:
                elapsed = now - task["last_run"]
                multiplier = {"seconds": 1, "minutes": 60, "hours": 3600}.get(task["unit"], 1)
                if elapsed >= task["interval"] * multiplier:
                    try:
                        task["callback"]()
                        task["last_run"] = now
                    except Exception as e:
                        logger.error(f"调度任务 [{task['name']}] 执行失败: {e}")

            # 处理事件队列
            try:
                while True:
                    event = self._event_queue.get_nowait()
                    self._handle_event(event)
            except queue.Empty:
                pass

            time.sleep(5)  # 每 5 秒检查一次

    def _handle_event(self, event: dict):
        """处理事件"""
        logger.debug(f"事件: {event['type']}")

    def list_tasks(self) -> list[dict]:
        """列出所有调度任务"""
        return [
            {
                "name": t["name"],
                "interval": t["interval"],
                "unit": t["unit"],
                "status": "running" if self._running else "stopped",
            }
            for t in self._tasks
        ]
