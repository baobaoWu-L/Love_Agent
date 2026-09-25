"""
训练编排器

管理和监控 LoveFlow 模型的训练过程：
  - 启动训练子进程（调用 run_loveflow_train.py）
  - 实时捕获标准输出/标准错误
  - 提供训练状态查询和控制（停止）
  - 发现最新检查点
"""
import logging
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import (
    TRAIN_SCRIPT,
    BACKEND_DIR,
    LOVEFLOW_DIR,
    MODEL_DIR,
    TRAIN_DEFAULT_EPOCHS,
    TRAIN_DEFAULT_LR,
    TRAIN_DEFAULT_LORA_R,
    TRAIN_DEFAULT_LORA_ALPHA,
    TRAIN_DEFAULT_BATCH_SIZE,
    TRAIN_DEFAULT_GRADIENT_ACCUM,
    TRAIN_DEFAULT_MAX_SEQ_LENGTH,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
)
from agent.core.task_tracker import create_task, update_task, read_progress

logger = logging.getLogger("loveflow.TrainingOrchestrator")


class TrainingOrchestrator:
    """训练编排器：管理 LoveFlow 训练生命周期"""

    def __init__(self):
        # 训练脚本路径
        self.train_script: Path = TRAIN_SCRIPT
        # 训练输出基础目录（LoRA 权重保存位置）
        self.output_base: Path = MODEL_DIR / "lora" / "training"
        # 当前运行的子进程（None 表示没有运行中的训练）
        self._process: Optional[subprocess.Popen] = None
        # 当前任务 ID
        self._task_id: Optional[str] = None
        # 标准输出/标准错误捕获线程
        self._log_thread: Optional[threading.Thread] = None
        # 捕获的日志行列表
        self._log_lines: list[str] = []
        # 训练参数快照
        self._params: dict = {}
        # 启动时间
        self._started_at: Optional[str] = None
        # 检查训练脚本是否存在
        self._script_exists = self.train_script.exists()
        if not self._script_exists:
            logger.warning(f"训练脚本不存在: {self.train_script}")

    def start_training(
        self,
        epochs: int = TRAIN_DEFAULT_EPOCHS,
        lr: float = TRAIN_DEFAULT_LR,
        lora_r: int = TRAIN_DEFAULT_LORA_R,
        lora_alpha: int = TRAIN_DEFAULT_LORA_ALPHA,
        batch_size: int = TRAIN_DEFAULT_BATCH_SIZE,
        gradient_accumulation: int = TRAIN_DEFAULT_GRADIENT_ACCUM,
        max_seq_length: int = TRAIN_DEFAULT_MAX_SEQ_LENGTH,
    ) -> dict:
        """
        启动训练任务

        参数：
            epochs: 训练轮数
            lr: 学习率
            lora_r: LoRA 秩
            lora_alpha: LoRA alpha
            batch_size: 批次大小（GPU 推荐 4-8）
            gradient_accumulation: 梯度累积步数
            max_seq_length: 最大序列长度

        返回：
            {"success": bool, "task_id": str, "process_id": int, "error": str}
        """
        # 检查是否已有训练在运行
        if self._process is not None and self._process.poll() is None:
            return {
                "success": False,
                "task_id": self._task_id or "",
                "process_id": self._process.pid,
                "error": "已有训练任务正在运行",
            }

        # 检查训练脚本
        if not self._script_exists:
            return {
                "success": False,
                "task_id": "",
                "process_id": 0,
                "error": f"训练脚本不存在: {self.train_script}",
            }

        # 在任务跟踪器中创建训练任务
        task_desc = f"LoveFlow 训练: epochs={epochs}, lr={lr}, lora_r={lora_r}, batch_size={batch_size}, grad_acc={gradient_accumulation}"
        task_id = create_task(task_desc)
        update_task(task_id, status="in_progress")

        # 保存参数快照
        self._params = {
            "epochs": epochs,
            "lr": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "batch_size": batch_size,
        }
        self._task_id = task_id
        self._log_lines = []
        self._started_at = datetime.now().isoformat()

        # 构建命令行参数（GPU 优化版）
        cmd = [
            sys.executable,
            str(self.train_script),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--lora_r", str(lora_r),
            "--batch_size", str(batch_size),
            "--gradient_accumulation", str(gradient_accumulation),
            "--max_seq_length", str(max_seq_length),
        ]

        # 确保输出目录存在
        self.output_base.mkdir(parents=True, exist_ok=True)

        # 设置环境变量（传递给子进程的 MySQL 连接信息）
        env = os.environ.copy()
        env["MYSQL_SERVER"] = MYSQL_HOST
        env["MYSQL_PORT"] = str(MYSQL_PORT)
        env["MYSQL_USER"] = MYSQL_USER
        env["MYSQL_PASSWORD"] = MYSQL_PASSWORD
        env["LOVEFLOW_DIR"] = str(LOVEFLOW_DIR)

        logger.info(f"启动训练: {' '.join(cmd)}")

        try:
            # 启动子进程
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(BACKEND_DIR.parent),  # LoveFlow 项目根目录
                universal_newlines=True,
                bufsize=1,  # 行缓冲
            )

            # 启动日志捕获线程（实时读取输出）
            self._log_thread = threading.Thread(
                target=self._capture_output,
                daemon=True,
                name=f"train-log-{self._process.pid}",
            )
            self._log_thread.start()

            logger.info(f"训练已启动，进程 PID: {self._process.pid}，任务 ID: {task_id}")
            return {
                "success": True,
                "task_id": task_id,
                "process_id": self._process.pid,
                "error": "",
            }

        except Exception as e:
            error_msg = f"启动训练失败: {e}"
            logger.error(error_msg)
            update_task(task_id, status="failed")
            self._process = None
            self._task_id = None
            return {
                "success": False,
                "task_id": task_id,
                "process_id": 0,
                "error": error_msg,
            }

    def get_status(self) -> dict:
        """
        获取当前训练状态

        返回：
            {"running": bool, "task_id": str, "process_id": int,
             "pid": int, "elapsed_seconds": float, "log_lines": list[str],
             "params": dict, "started_at": str}
        """
        running = False
        pid = 0
        elapsed = 0.0

        if self._process is not None:
            return_code = self._process.poll()
            running = return_code is None
            pid = self._process.pid
            if self._started_at:
                start = datetime.fromisoformat(self._started_at)
                elapsed = (datetime.now() - start).total_seconds()

        return {
            "running": running,
            "task_id": self._task_id or "",
            "process_id": pid,
            "pid": pid,
            "elapsed_seconds": elapsed,
            "log_lines": self._log_lines[-50:],  # 返回最近 50 行日志
            "params": self._params,
            "started_at": self._started_at or "",
        }

    def stop_training(self) -> bool:
        """
        停止训练进程

        先尝试优雅终止（SIGTERM），5 秒后强制杀死（SIGKILL）。

        返回：
            是否成功停止
        """
        if self._process is None:
            logger.warning("没有正在运行的训练进程")
            return False

        if self._process.poll() is not None:
            logger.info("训练进程已结束")
            self._cleanup()
            return True

        pid = self._process.pid
        logger.info(f"正在停止训练进程 (PID: {pid})")

        try:
            if sys.platform == "win32":
                # Windows 上使用 terminate()
                self._process.terminate()
            else:
                # Unix 上先发 SIGTERM
                os.kill(pid, signal.SIGTERM)

            # 等待进程退出（最长 5 秒）
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 超时后强制杀死
                logger.warning(f"训练进程未响应 SIGTERM，强制杀死 (PID: {pid})")
                if sys.platform == "win32":
                    self._process.kill()
                else:
                    os.kill(pid, signal.SIGKILL)
                self._process.wait(timeout=5)

            if self._task_id:
                update_task(self._task_id, status="stopped")

            logger.info(f"训练进程已停止 (PID: {pid})")
            self._cleanup()
            return True

        except Exception as e:
            logger.error(f"停止训练进程失败: {e}")
            return False

    def get_latest_checkpoint(self) -> Optional[dict]:
        """
        查找最新的 LoRA 检查点

        按 step_* 目录名的数字后缀排序，返回最新的那个。

        返回：
            {"name": str, "path": str, "step": int,
             "created_at": str, "files": list[str]} 或 None
        """
        if not self.output_base.exists():
            return None

        # 扫描所有 step_* 目录
        step_dirs = []
        for d in self.output_base.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step_num = int(d.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                mtime = d.stat().st_mtime
                step_dirs.append((step_num, mtime, d))

        if not step_dirs:
            return None

        # 按 step 数字降序排序，取最大的
        step_dirs.sort(key=lambda x: x[0], reverse=True)
        step_num, mtime, latest_dir = step_dirs[0]

        # 列出目录中的文件
        files = [f.name for f in latest_dir.iterdir() if f.is_file()]

        return {
            "name": latest_dir.name,
            "path": str(latest_dir),
            "step": step_num,
            "created_at": datetime.fromtimestamp(mtime).isoformat(),
            "files": files,
        }

    def get_context(self) -> Optional[str]:
        """
        获取训练状态文本，供 Agent 上下文注入

        返回格式化的训练状态摘要，或 None（无有效状态时）。
        """
        status = self.get_status()

        if not status["running"] and not status.get("task_id"):
            # 没有活跃训练，检查是否有最新检查点
            ckpt = self.get_latest_checkpoint()
            if ckpt:
                return (
                    f"【训练模块】\n"
                    f"  状态: 空闲\n"
                    f"  最新检查点: {ckpt['name']} (step {ckpt['step']})\n"
                    f"  保存路径: {ckpt['path']}\n"
                )
            return None

        lines = status.get("log_lines", [])
        recent_log = "\n".join(lines[-5:]) if lines else "暂无日志"

        return (
            f"【训练模块】\n"
            f"  状态: {'运行中' if status['running'] else '已停止'}\n"
            f"  任务 ID: {status['task_id']}\n"
            f"  进程 PID: {status['pid']}\n"
            f"  运行时间: {status['elapsed_seconds']:.0f} 秒\n"
            f"  参数: {status['params']}\n"
            f"  最近日志:\n{recent_log}\n"
        )

    def _capture_output(self, proc: subprocess.Popen):
        """
        实时捕获子进程的标准输出（内部线程）

        逐行读取输出并存储到 _log_lines 列表，
        同时通过 logger 输出到日志系统。
        """
        try:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n\r")
                if not line:
                    continue
                self._log_lines.append(line)
                logger.info(f"[训练] {line}")

                # 检测训练完成或失败标志，更新任务状态
                if "训练完成" in line or "✅" in line:
                    if self._task_id:
                        update_task(self._task_id, status="completed")
                elif "失败" in line or "❌" in line:
                    if self._task_id:
                        update_task(self._task_id, status="failed")

        except Exception as e:
            logger.error(f"捕获训练输出时出错: {e}")
        finally:
            proc.stdout.close()

    def _cleanup(self):
        """清理训练进程相关资源"""
        self._process = None
        self._task_id = None
        self._log_thread = None
