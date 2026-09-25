"""
训练监控器

监控训练进度并生成报告：
  - 解析训练日志提取 loss/step 数据
  - 绘制损失曲线
  - 查询 MySQL training_metrics 表获取训练指标
  - 对比检查点文件
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import (
    MODEL_DIR,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
)

logger = logging.getLogger("loveflow.TrainingMonitor")

# 尝试导入 matplotlib（可选依赖，用于绘制训练曲线）
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，避免 GUI 依赖
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib 未安装，训练曲线绘制功能不可用")


class TrainingMonitor:
    """训练监控器：解析日志、绘制曲线、查询指标、对比检查点"""

    def __init__(self):
        # 训练输出基础目录
        self.output_base: Path = MODEL_DIR / "lora" / "training"
        # 默认曲线保存目录
        self.plot_dir: Path = MODEL_DIR / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def parse_log(self, log_path: Path) -> list[dict]:
        """
        解析训练日志文件，提取 loss 和 step 数据

        支持常见的日志格式：
          - "Step XXX, loss=X.XXXX"
          - "Epoch X, Step XXX, loss=X.XXXX"
          - "{\"loss\": X.XX, \"step\": XXX}"

        参数：
            log_path: 日志文件路径

        返回：
            [{"step": int, "loss": float, "epoch": int, "timestamp": str}, ...]
        """
        if not log_path.exists():
            logger.warning(f"日志文件不存在: {log_path}")
            return []

        metrics = []
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue

                metric = self._parse_line(line)
                if metric:
                    metrics.append(metric)

            logger.info(f"日志解析完成: {len(metrics)} 条指标 ({log_path.name})")
            return metrics

        except Exception as e:
            logger.error(f"解析日志文件失败 {log_path}: {e}")
            return []

    def plot_curve(
        self,
        metric_data: list[dict],
        save_path: Optional[Path] = None,
    ) -> Optional[str]:
        """
        绘制训练损失曲线（使用 matplotlib）

        参数：
            metric_data: parse_log 返回的指标数据列表
            save_path: 图片保存路径（None 则自动生成）

        返回：
            图片文件路径字符串，或 None（失败时）
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib 不可用，无法绘制曲线")
            return None

        if not metric_data:
            logger.warning("没有指标数据可供绘制")
            return None

        try:
            # 提取数据
            steps = [m.get("step", i) for i, m in enumerate(metric_data)]
            losses = [m["loss"] for m in metric_data if "loss" in m]

            if not losses:
                logger.warning("指标数据中没有 loss 字段")
                return None

            # 如果 steps 长度与 losses 不匹配，使用序号
            if len(steps) != len(losses):
                steps = list(range(1, len(losses) + 1))

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制 loss 曲线
            ax.plot(steps, losses, "b-", linewidth=1.5, alpha=0.8, label="Loss")
            # 绘制平滑曲线（移动平均）
            if len(losses) >= 10:
                window = min(10, len(losses) // 5)
                smoothed = self._moving_average(losses, window)
                smooth_steps = steps[: len(smoothed)]
                ax.plot(
                    smooth_steps,
                    smoothed,
                    "r--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"平滑 (窗口={window})",
                )

            ax.set_xlabel("训练步数 (Step)", fontsize=12)
            ax.set_ylabel("损失值 (Loss)", fontsize=12)
            ax.set_title("LoveFlow 训练损失曲线", fontsize=14, fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            # 标注最低 loss
            min_loss_idx = losses.index(min(losses))
            ax.annotate(
                f"Min Loss: {losses[min_loss_idx]:.4f}",
                xy=(steps[min_loss_idx], losses[min_loss_idx]),
                xytext=(steps[min_loss_idx] + len(steps) * 0.05, losses[min_loss_idx] + 0.1),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10,
                color="green",
            )

            # 如果提供了保存路径，使用它；否则自动生成
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.plot_dir / f"loss_curve_{timestamp}.png"

            # 确保保存目录存在
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"损失曲线已保存: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"绘制损失曲线失败: {e}")
            return None

    def check_metrics(self) -> dict:
        """
        从 MySQL training_metrics 表查询最新的训练指标

        返回：
            {"latest_step": int, "latest_loss": float, "latest_accuracy": float,
             "total_records": int, "recent_metrics": list[dict], "error": str}
        """
        try:
            import pymysql

            conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.Cursor,
            )
            cursor = conn.cursor()

            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM training_metrics")
            total_records = cursor.fetchone()[0]

            # 最新一条记录
            cursor.execute(
                "SELECT id, job_id, step, epoch, loss, accuracy, learning_rate, grad_norm, created_at "
                "FROM training_metrics ORDER BY id DESC LIMIT 1"
            )
            latest_row = cursor.fetchone()
            latest = {}
            if latest_row:
                latest = {
                    "id": latest_row[0],
                    "job_id": latest_row[1],
                    "step": latest_row[2],
                    "epoch": latest_row[3],
                    "loss": latest_row[4],
                    "accuracy": latest_row[5],
                    "learning_rate": latest_row[6],
                    "grad_norm": latest_row[7],
                    "created_at": latest_row[8].isoformat() if hasattr(latest_row[8], "isoformat") else str(latest_row[8]),
                }

            # 最近 20 条记录（按 id 降序）
            cursor.execute(
                "SELECT id, job_id, step, epoch, loss, accuracy, learning_rate, grad_norm, created_at "
                "FROM training_metrics ORDER BY id DESC LIMIT 20"
            )
            recent = []
            for row in cursor.fetchall():
                recent.append({
                    "id": row[0],
                    "job_id": row[1],
                    "step": row[2],
                    "epoch": row[3],
                    "loss": row[4],
                    "accuracy": row[5],
                    "learning_rate": row[6],
                    "grad_norm": row[7],
                    "created_at": row[8].isoformat() if hasattr(row[8], "isoformat") else str(row[8]),
                })

            cursor.close()
            conn.close()

            return {
                "latest_step": latest.get("step", 0),
                "latest_loss": latest.get("loss", 0.0),
                "latest_accuracy": latest.get("accuracy", 0.0),
                "total_records": total_records,
                "latest": latest,
                "recent_metrics": recent,
                "error": "",
            }

        except Exception as e:
            logger.debug(f"查询训练指标失败: {e}")
            return {
                "latest_step": 0,
                "latest_loss": 0.0,
                "latest_accuracy": 0.0,
                "total_records": 0,
                "latest": {},
                "recent_metrics": [],
                "error": str(e),
            }

    def compare_checkpoints(self) -> list[dict]:
        """
        对比 LoRA 检查点目录

        扫描 output_base 下的 step_* 目录，按步骤编号和创建时间排序。

        返回：
            [{"name": str, "step": int, "path": str,
              "created_at": str, "file_count": int, "file_size_mb": float}, ...]
        """
        if not self.output_base.exists():
            return []

        checkpoints = []
        try:
            for d in self.output_base.iterdir():
                if not d.is_dir() or not d.name.startswith("step_"):
                    continue

                try:
                    step_num = int(d.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                # 统计文件数量和总大小
                files = list(d.iterdir())
                file_count = len(files)
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = round(total_size / (1024 * 1024), 2)

                # 获取目录修改时间
                mtime = datetime.fromtimestamp(d.stat().st_mtime)

                checkpoints.append({
                    "name": d.name,
                    "step": step_num,
                    "path": str(d),
                    "created_at": mtime.isoformat(),
                    "file_count": file_count,
                    "file_size_mb": size_mb,
                })

            # 按 step 编号降序排列（最新的在前）
            checkpoints.sort(key=lambda x: x["step"], reverse=True)
            return checkpoints

        except Exception as e:
            logger.error(f"对比检查点失败: {e}")
            return []

    def get_context(self) -> Optional[str]:
        """
        获取训练进度摘要，供 Agent 上下文注入

        返回格式化的训练进度文本，或 None（无可用信息时）。
        """
        # 获取最近指标
        metrics = self.check_metrics()
        metrics_info = ""
        if not metrics.get("error") and metrics["total_records"] > 0:
            latest = metrics.get("latest", {})
            metrics_info = (
                f"  最新步数: {metrics['latest_step']}\n"
                f"  最新 Loss: {metrics['latest_loss']:.4f}\n"
                f"  最新 Accuracy: {metrics['latest_accuracy']:.4f}\n"
                f"  总记录数: {metrics['total_records']}\n"
            )

        # 获取检查点信息
        ckpts = self.compare_checkpoints()
        ckpt_info = ""
        if ckpts:
            latest_ckpt = ckpts[0]
            ckpt_info = (
                f"  检查点数量: {len(ckpts)}\n"
                f"  最新检查点: {latest_ckpt['name']} (step {latest_ckpt['step']})\n"
                f"  检查点大小: {latest_ckpt['file_size_mb']} MB\n"
            )

        if not metrics_info and not ckpt_info:
            return None

        summary = "【训练进度】\n"
        if metrics_info:
            summary += metrics_info
        if ckpt_info:
            summary += ckpt_info

        return summary

    def _parse_line(self, line: str) -> Optional[dict]:
        """
        解析单行日志文本，提取训练指标

        支持的格式：
          - "Step XXX, loss=X.XXXX"
          - "Epoch X, Step XXX, loss=X.XXXX"
          - JSON 格式: {"loss": X.XX, "step": XXX}

        参数：
            line: 日志行文本

        返回：
            解析出的指标字典，或 None
        """
        metric = {}

        # 尝试 JSON 格式解析
        json_match = re.search(r'\{"loss":\s*([\d.]+)', line)
        if json_match:
            metric["loss"] = float(json_match.group(1))
            step_match = re.search(r'"step":\s*(\d+)', line)
            if step_match:
                metric["step"] = int(step_match.group(1))
            epoch_match = re.search(r'"epoch":\s*(\d+)', line)
            if epoch_match:
                metric["epoch"] = int(epoch_match.group(1))
            metric["timestamp"] = datetime.now().isoformat()
            return metric

        # 尝试 "Step XXX, loss=X.XXXX" 格式
        loss_match = re.search(r"loss[=:]\s*([\d.]+)", line, re.IGNORECASE)
        if loss_match:
            metric["loss"] = float(loss_match.group(1))
        else:
            return None

        # 提取 step 编号
        step_match = re.search(r"(?:Step|step|步)[=:\s]*(\d+)", line)
        if step_match:
            metric["step"] = int(step_match.group(1))

        # 提取 epoch 编号
        epoch_match = re.search(r"(?:Epoch|epoch|轮)[=:\s]*(\d+)", line)
        if epoch_match:
            metric["epoch"] = int(epoch_match.group(1))

        metric["timestamp"] = datetime.now().isoformat()
        return metric

    @staticmethod
    def _moving_average(data: list[float], window: int) -> list[float]:
        """
        计算移动平均（用于损失曲线平滑）

        参数：
            data: 原始数据序列
            window: 滑动窗口大小

        返回：
            平滑后的数据序列
        """
        if window <= 1:
            return data[:]
        return [
            sum(data[max(0, i - window + 1): i + 1]) / min(window, i + 1)
            for i in range(len(data))
        ]
