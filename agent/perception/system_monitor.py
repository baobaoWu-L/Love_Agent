"""
系统监控模块：采集 CPU、内存、磁盘、网络、进程等系统状态

为 Agent 提供实时的环境感知能力，帮助理解宿主机的运行状况。
所有可选依赖均使用懒加载，缺失时以降级模式运行。
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import MONITOR_INTERVAL

logger = logging.getLogger("loveflow.SYSTEM_MONITOR")

# 尝试导入 psutil
psutil = None
try:
    import psutil as _psutil

    psutil = _psutil
except ImportError:
    logger.warning("psutil 未安装，系统监控模块将以降级模式运行。请执行: pip install psutil")


class SystemMonitor:
    """系统监控器：采集宿主机 CPU、内存、磁盘、网络、进程等信息"""

    def __init__(self):
        """初始化系统监控器"""
        self._last_network = None
        self._last_network_time = time.time()
        self._loveflow_keywords = [
            "run_loveflow",
            "run_training",
            "run_inference",
            "loveflow",
            "uvicorn",
        ]
        logger.info("系统监控模块初始化完成")

    # ==================== CPU ====================

    def get_cpu(self) -> dict:
        """
        获取 CPU 使用率

        Returns:
            dict: {
                "percent": 整体使用率,
                "per_core": 每核心使用率列表,
                "count": 逻辑核心数,
                "physical_count": 物理核心数,
                "frequency_current": 当前频率(MHz),
                "frequency_max": 最大频率(MHz),
            }
        """
        if psutil is None:
            return {"error": "psutil 未安装，无法获取 CPU 信息", "available": False}

        try:
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            freq = psutil.cpu_freq()
            return {
                "percent": psutil.cpu_percent(interval=0.1),
                "per_core": per_core,
                "count": psutil.cpu_count(logical=True),
                "physical_count": psutil.cpu_count(logical=False),
                "frequency_current": freq.current if freq else None,
                "frequency_max": freq.max if freq else None,
                "available": True,
            }
        except Exception as e:
            logger.error(f"获取 CPU 信息失败: {e}")
            return {"error": str(e), "available": False}

    # ==================== 内存 ====================

    def get_memory(self) -> dict:
        """
        获取内存使用情况

        Returns:
            dict: {
                "total": 总内存(GB),
                "available": 可用内存(GB),
                "used": 已用内存(GB),
                "percent": 使用率(%),
                "swap_total": 交换分区总量(GB),
                "swap_used": 交换分区已用(GB),
                "swap_percent": 交换分区使用率(%),
            }
        """
        if psutil is None:
            return {"error": "psutil 未安装，无法获取内存信息", "available": False}

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "total": round(mem.total / (1024**3), 2),
                "available": round(mem.available / (1024**3), 2),
                "used": round(mem.used / (1024**3), 2),
                "percent": mem.percent,
                "swap_total": round(swap.total / (1024**3), 2),
                "swap_used": round(swap.used / (1024**3), 2),
                "swap_percent": swap.percent,
                "available": True,
            }
        except Exception as e:
            logger.error(f"获取内存信息失败: {e}")
            return {"error": str(e), "available": False}

    # ==================== 磁盘 ====================

    def get_disk(self, path: str = "C:/") -> dict:
        """
        获取磁盘使用情况

        Args:
            path: 要检查的路径，默认 C 盘

        Returns:
            dict: {
                "total": 总容量(GB),
                "used": 已用(GB),
                "free": 剩余(GB),
                "percent": 使用率(%),
            }
        """
        if psutil is None:
            return {"error": "psutil 未安装，无法获取磁盘信息", "available": False}

        try:
            disk = psutil.disk_usage(path)
            return {
                "total": round(disk.total / (1024**3), 2),
                "used": round(disk.used / (1024**3), 2),
                "free": round(disk.free / (1024**3), 2),
                "percent": disk.percent,
                "path": path,
                "available": True,
            }
        except Exception as e:
            logger.error(f"获取磁盘信息失败: {e}")
            return {"error": str(e), "available": False}

    def get_all_disks(self) -> list[dict]:
        """
        获取所有磁盘分区的使用情况

        Returns:
            list[dict]: 每个分区的使用信息列表
        """
        if psutil is None:
            return [{"error": "psutil 未安装，无法获取磁盘信息", "available": False}]

        try:
            results = []
            for part in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    results.append({
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "total": round(usage.total / (1024**3), 2),
                        "used": round(usage.used / (1024**3), 2),
                        "free": round(usage.free / (1024**3), 2),
                        "percent": usage.percent,
                    })
                except PermissionError:
                    # 某些挂载点可能无权限访问
                    results.append({
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "error": "无权限访问",
                    })
            return results
        except Exception as e:
            logger.error(f"获取所有磁盘信息失败: {e}")
            return [{"error": str(e), "available": False}]

    # ==================== 网络 ====================

    def get_network(self) -> dict:
        """
        获取网络 IO 统计

        Returns:
            dict: {
                "bytes_sent": 发送字节数,
                "bytes_recv": 接收字节数,
                "packets_sent": 发送包数,
                "packets_recv": 接收包数,
                "speed_sent": 发送速度(每秒),
                "speed_recv": 接收速度(每秒),
                "connections": 当前连接数,
            }
        """
        if psutil is None:
            return {"error": "psutil 未安装，无法获取网络信息", "available": False}

        try:
            net_io = psutil.net_io_counters()
            now = time.time()
            time_delta = now - self._last_network_time

            # 计算网络速度
            speed_sent = 0
            speed_recv = 0
            if self._last_network and time_delta > 0:
                speed_sent = (net_io.bytes_sent - self._last_network.bytes_sent) / time_delta
                speed_recv = (net_io.bytes_recv - self._last_network.bytes_recv) / time_delta

            self._last_network = net_io
            self._last_network_time = now

            # 获取连接数
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, PermissionError):
                connections = -1  # 权限不足

            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "speed_sent": round(speed_sent, 2),
                "speed_recv": round(speed_recv, 2),
                "connections": connections,
                "available": True,
            }
        except Exception as e:
            logger.error(f"获取网络信息失败: {e}")
            return {"error": str(e), "available": False}

    # ==================== 进程 ====================

    def get_processes(self, filter_keyword: Optional[str] = None) -> list[dict]:
        """
        获取运行中的进程列表

        Args:
            filter_keyword: 可选筛选关键字，只返回名称包含关键字的进程

        Returns:
            list[dict]: 进程信息列表，每项包含 name, pid, cpu_percent, memory_percent, status
        """
        if psutil is None:
            return [{"error": "psutil 未安装，无法获取进程信息"}]

        try:
            results = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
                try:
                    info = proc.info
                    name = info["name"] or ""

                    if filter_keyword and filter_keyword.lower() not in name.lower():
                        continue

                    results.append({
                        "name": name,
                        "pid": info["pid"],
                        "cpu_percent": info["cpu_percent"] or 0,
                        "memory_percent": info["memory_percent"] or 0,
                        "status": info["status"] or "unknown",
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # 按 CPU 使用率排序
            results.sort(key=lambda x: x["cpu_percent"], reverse=True)
            return results
        except Exception as e:
            logger.error(f"获取进程信息失败: {e}")
            return [{"error": str(e)}]

    # ==================== LoveFlow 状态 ====================

    def get_loveflow_status(self) -> dict:
        """
        检测与 LoveFlow 相关的进程是否在运行

        Returns:
            dict: {
                "running": bool,  # 是否有相关进程在运行
                "processes": list[dict],  # 相关进程列表
                "training_running": bool,  # 训练守护进程是否运行
                "inference_running": bool,  # 推理服务器是否运行
            }
        """
        if psutil is None:
            return {"running": False, "error": "psutil 未安装", "available": False}

        try:
            loveflow_procs = []
            for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    name = (info["name"] or "").lower()
                    cmdline = " ".join(info["cmdline"] or []).lower()

                    # 检查进程名或命令行是否匹配 LoveFlow 关键字
                    matched = any(kw in name or kw in cmdline for kw in self._loveflow_keywords)
                    if not matched:
                        continue

                    loveflow_procs.append({
                        "name": info["name"],
                        "pid": info["pid"],
                        "cmdline": " ".join(info["cmdline"] or [])[:200],
                        "cpu_percent": info["cpu_percent"] or 0,
                        "memory_percent": info["memory_percent"] or 0,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # 判断训练和推理进程
            training_running = any("run_training" in (p.get("cmdline", "") or "") for p in loveflow_procs)
            inference_running = any("run_inference" in (p.get("cmdline", "") or "") for p in loveflow_procs)

            return {
                "running": len(loveflow_procs) > 0,
                "processes": loveflow_procs,
                "count": len(loveflow_procs),
                "training_running": training_running,
                "inference_running": inference_running,
                "available": True,
            }
        except Exception as e:
            logger.error(f"检测 LoveFlow 状态失败: {e}")
            return {"running": False, "error": str(e), "available": False}

    # ==================== 汇总 ====================

    def get_summary(self) -> str:
        """
        获取格式化的完整系统状态文本

        Returns:
            str: 格式化的系统状态报告
        """
        lines = []
        lines.append("=" * 50)
        lines.append(f"  系统状态报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 50)

        # CPU
        cpu = self.get_cpu()
        if cpu.get("available"):
            lines.append(f"\nCPU: {cpu['percent']}% | {cpu['count']} 逻辑核 ({cpu['physical_count']} 物理核)")
            if cpu.get("per_core"):
                core_str = ", ".join(f"{p:.1f}%" for p in cpu["per_core"])
                lines.append(f"  每核心: [{core_str}]")
            if cpu.get("frequency_current"):
                lines.append(f"  频率: {cpu['frequency_current']:.0f} MHz")
        else:
            lines.append(f"\nCPU: 不可用 ({cpu.get('error', '未知错误')})")

        # 内存
        mem = self.get_memory()
        if mem.get("available"):
            lines.append(f"\n内存: {mem['percent']}% | {mem['used']}G / {mem['total']}G (可用: {mem['available']}G)")
            if mem.get("swap_total", 0) > 0:
                lines.append(f"  交换分区: {mem['swap_percent']}% | {mem['swap_used']}G / {mem['swap_total']}G")
        else:
            lines.append(f"\n内存: 不可用 ({mem.get('error', '未知错误')})")

        # 磁盘
        disk = self.get_disk()
        if disk.get("available"):
            # 判断使用率预警
            alert = " !! 磁盘即将满 !!" if disk["percent"] > 90 else ""
            lines.append(f"\n磁盘(C:): {disk['percent']}% | {disk['used']}G / {disk['total']}G (剩余: {disk['free']}G){alert}")
        else:
            lines.append(f"\n磁盘(C:): 不可用 ({disk.get('error', '未知错误')})")

        # 网络
        net = self.get_network()
        if net.get("available"):
            lines.append(f"\n网络:")
            lines.append(f"  发送: {self._format_bytes(net['bytes_sent'])} | 接收: {self._format_bytes(net['bytes_recv'])}")
            lines.append(f"  实时速度: ↑{self._format_bytes(net['speed_sent'])}/s ↓{self._format_bytes(net['speed_recv'])}/s")
            if net.get("connections", -1) >= 0:
                lines.append(f"  连接数: {net['connections']}")
        else:
            lines.append(f"\n网络: 不可用 ({net.get('error', '未知错误')})")

        # LoveFlow 进程
        lf = self.get_loveflow_status()
        lines.append(f"\nLoveFlow 服务:")
        if lf.get("available") and lf.get("running"):
            lines.append(f"  ● 运行中 ({lf['count']} 个进程)")
            lines.append(f"  - 训练守护进程: {'运行中' if lf.get('training_running') else '未运行'}")
            lines.append(f"  - 推理服务器: {'运行中' if lf.get('inference_running') else '未运行'}")
            for p in lf.get("processes", []):
                lines.append(f"    [{p['pid']}] {p['name']} (CPU: {p['cpu_percent']}% 内存: {p['memory_percent']}%)")
        elif lf.get("available"):
            lines.append("  ○ 未运行")
        else:
            lines.append(f"  ? 无法检测 ({lf.get('error', '未知错误')})")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    # ==================== 上下文注入 ====================

    def get_context(self) -> Optional[str]:
        """
        返回格式化的系统状态摘要，供 Agent 上下文注入

        Returns:
            Optional[str]: 系统状态文本，如果完全不可用则返回 None
        """
        if psutil is None:
            return None

        try:
            cpu = self.get_cpu()
            mem = self.get_memory()
            disk = self.get_disk()
            lf = self.get_loveflow_status()

            parts = ["【系统状态】"]

            if cpu.get("available"):
                parts.append(f"CPU: {cpu['percent']}% ({cpu['count']}核)")

            if mem.get("available"):
                parts.append(f"内存: {mem['percent']}% ({mem['used']}G/{mem['total']}G)")

            if disk.get("available"):
                parts.append(f"磁盘: {disk['percent']}% ({disk['free']}G 剩余)")

            if lf.get("available"):
                status_icon = "运行中" if lf.get("running") else "未运行"
                parts.append(f"LoveFlow: {status_icon}")
                if lf.get("running"):
                    parts.append(f"  训练: {'是' if lf.get('training_running') else '否'} | 推理: {'是' if lf.get('inference_running') else '否'}")

            return " | ".join(parts)
        except Exception as e:
            logger.error(f"生成系统状态上下文失败: {e}")
            return None

    # ==================== 辅助方法 ====================

    @staticmethod
    def _format_bytes(size_bytes: float) -> str:
        """将字节数格式化为可读形式"""
        if size_bytes < 0:
            return "N/A"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(size_bytes) < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
