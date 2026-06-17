"""
网络发现模块：局域网设备扫描、端口扫描、服务检测

为 Agent 提供网络环境感知能力，帮助发现局域网中的设备和服务。
所有可选依赖均使用懒加载，缺失时以降级模式运行。
"""
import logging
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.agent_config import NETWORK_SCAN_TIMEOUT, MYSQL_PORT, REDIS_PORT

logger = logging.getLogger("loveflow.NETWORK_DISCOVERY")

# 尝试导入三方库增强扫描能力
nmap = None
try:
    import nmap as _nmap

    nmap = _nmap
except ImportError:
    pass


@dataclass
class ServiceInfo:
    """服务信息数据类"""
    port: int
    name: str
    protocol: str = "tcp"


# 常见服务端口映射
COMMON_SERVICES: dict[str, ServiceInfo] = {
    "HTTP": ServiceInfo(80, "HTTP"),
    "HTTPS": ServiceInfo(443, "HTTPS"),
    "RTSP": ServiceInfo(554, "RTSP"),
    "MySQL": ServiceInfo(MYSQL_PORT, "MySQL"),
    "Redis": ServiceInfo(REDIS_PORT, "Redis"),
    "LoveFlow": ServiceInfo(8005, "LoveFlow Inference"),
    "SSH": ServiceInfo(22, "SSH"),
    "Telnet": ServiceInfo(23, "Telnet"),
    "FTP": ServiceInfo(21, "FTP"),
    "SMB": ServiceInfo(445, "SMB"),
    "RDP": ServiceInfo(3389, "RDP"),
}


class NetworkScanner:
    """网络扫描器：ARP 扫描、端口扫描、服务检测"""

    def __init__(self):
        """初始化网络扫描器"""
        self._last_scan_result: Optional[list[dict]] = None
        self._last_scan_time: float = 0
        logger.info("网络发现模块初始化完成")

    # ==================== ARP 扫描 ====================

    def scan_arp(self) -> list[dict]:
        """
        ARP 扫描局域网设备

        解析 'arp -a' 命令输出，发现同网段活跃设备。
        优先使用系统 arp 命令，兼容 Windows/Linux/macOS。

        Returns:
            list[dict]: 设备列表，每项包含 ip, mac, type (动态/静态)
        """
        try:
            # 执行系统 ARP 命令
            if self._is_windows():
                result = subprocess.run(
                    ["arp", "-a"],
                    capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
            else:
                result = subprocess.run(
                    ["arp", "-a"],
                    capture_output=True, text=True, timeout=10,
                )

            devices = self._parse_arp_output(result.stdout)
            self._last_scan_result = devices
            self._last_scan_time = time.time()

            logger.info(f"ARP 扫描完成，发现 {len(devices)} 个设备")
            return devices

        except subprocess.TimeoutExpired:
            logger.warning("ARP 扫描超时")
            return []
        except FileNotFoundError:
            logger.warning("arp 命令不可用，尝试使用 socket 扫描...")
            return self._fallback_scan()
        except Exception as e:
            logger.error(f"ARP 扫描失败: {e}")
            return []

    def _parse_arp_output(self, arp_output: str) -> list[dict]:
        """解析 arp -a 的输出"""
        devices = []
        seen_ips = set()

        # Windows 格式:
        #   Internet Address    Physical Address    Type
        #   192.168.1.1         00-11-22-33-44-55   dynamic
        #
        # Linux/macOS 格式:
        #   ? (192.168.1.1) at 00:11:22:33:44:55 [ether] on eth0

        for line in arp_output.splitlines():
            line = line.strip()
            if not line:
                continue

            # 跳过表头
            if "Internet Address" in line or "Address" in line and "Physical" in line:
                continue
            if line.startswith("接口") or line.startswith("Interface"):
                continue

            ip = None
            mac = None
            arp_type = "unknown"

            # Windows 格式尝试
            win_match = re.match(
                r"^\s*(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F-]{17,})\s+(\w+)",
                line,
            )
            if win_match:
                ip = win_match.group(1)
                mac = win_match.group(2).replace("-", ":").lower()
                arp_type = win_match.group(3).lower()

            # Linux/macOS 格式尝试
            if not ip:
                unix_match = re.match(
                    r".*\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([0-9a-fA-F:]{17,})",
                    line,
                )
                if unix_match:
                    ip = unix_match.group(1)
                    mac = unix_match.group(2).lower()
                    arp_type = "dynamic"

            # 还尝试简化的 IPv4 + MAC 匹配
            if not ip:
                simple_match = re.match(
                    r".*?(\d+\.\d+\.\d+\.\d+).*?([0-9a-fA-F:-]{17,})",
                    line,
                )
                if simple_match:
                    ip = simple_match.group(1)
                    mac_raw = simple_match.group(2)
                    mac = mac_raw.replace("-", ":").lower()
                    arp_type = "dynamic"

            if ip and mac and ip not in seen_ips:
                # 过滤无效 MAC
                if not mac.startswith("ff:ff:ff:ff") and mac != "00:00:00:00:00:00":
                    seen_ips.add(ip)
                    devices.append({
                        "ip": ip,
                        "mac": mac,
                        "type": arp_type,
                        "hostname": self._resolve_hostname(ip),
                    })

        return devices

    def _resolve_hostname(self, ip: str) -> str:
        """尝试反向解析主机名"""
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            return hostname
        except (socket.herror, socket.gaierror):
            return ""

    def _fallback_scan(self) -> list[dict]:
        """
        arp 命令不可用时的备用方案

        使用 socket 尝试扫描常见网段，仅检测本机所在子网。
        """
        devices = []

        try:
            # 获取本机 IP
            host_ip = self._get_local_ip()
            if not host_ip:
                return []

            # 推导子网
            subnet = ".".join(host_ip.split(".")[:3])
            logger.info(f"使用 socket 扫描子网 {subnet}.0/24")

            # 仅扫描少量地址（局域网常用）
            for last_octet in [1, 254, 255] + list(range(2, 10)) + [host_ip.split(".")[-1]]:
                ip = f"{subnet}.{last_octet}"
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.3)
                    result = sock.connect_ex((ip, 445))  # SMB 端口
                    sock.close()

                    if result == 0:
                        devices.append({
                            "ip": ip,
                            "mac": "",
                            "type": "socket_scan",
                            "hostname": self._resolve_hostname(ip),
                        })
                except:
                    continue

        except Exception as e:
            logger.error(f"备用扫描失败: {e}")

        return devices

    @staticmethod
    def _get_local_ip() -> Optional[str]:
        """获取本机局域网 IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            # 不需要真正建立连接
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            try:
                hostname = socket.gethostname()
                return socket.gethostbyname(hostname)
            except Exception:
                return None

    # ==================== 端口扫描 ====================

    def scan_ports(self, host: str, ports: Optional[list[int]] = None) -> list[dict]:
        """
        TCP 端口扫描

        对指定主机的指定端口列表进行 TCP 连接检测。

        Args:
            host: 目标主机 IP
            ports: 要扫描的端口列表，默认扫描常见端口

        Returns:
            list[dict]: 开放端口列表，每项包含 port, service, state
        """
        if ports is None:
            ports = self._get_default_ports()

        open_ports = []
        total = len(ports)

        logger.info(f"开始扫描 {host} 的 {total} 个端口...")

        for i, port in enumerate(ports):
            if i % 20 == 0 and i > 0:
                logger.debug(f"端口扫描进度: {i}/{total}")

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(min(NETWORK_SCAN_TIMEOUT / len(ports), 1.0))
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    service_name = self._get_service_name(port)
                    open_ports.append({
                        "port": port,
                        "service": service_name,
                        "state": "open",
                    })
                    logger.debug(f"  端口 {port} ({service_name}) 开放")
            except Exception as e:
                logger.debug(f"扫描端口 {port} 时出错: {e}")

        logger.info(f"{host} 端口扫描完成，发现 {len(open_ports)} 个开放端口")
        return open_ports

    # ==================== 服务检测 ====================

    def detect_services(self, host: str) -> dict:
        """
        检测主机上的常见服务

        快速检查一组预定义的服务端口是否开放。

        Args:
            host: 目标主机 IP

        Returns:
            dict: 服务检测结果
                {
                    "host": str,
                    "services": {
                        "HTTP": {"port": 80, "open": bool},
                        "HTTPS": {"port": 443, "open": bool},
                        ...
                    },
                    "open_count": int,
                    "total_checked": int,
                }
        """
        results = {}
        open_count = 0

        for svc_name, svc_info in COMMON_SERVICES.items():
            is_open = self._check_port(host, svc_info.port)
            results[svc_name] = {
                "port": svc_info.port,
                "open": is_open,
            }
            if is_open:
                open_count += 1

        # 额外检测：尝试 HTTP 页面
        if results.get("HTTP", {}).get("open"):
            try:
                import urllib.request
                req = urllib.request.Request(f"http://{host}/", method="GET")
                req.timeout = 3
                response = urllib.request.urlopen(req)
                results["HTTP"]["status_code"] = response.status
                results["HTTP"]["server"] = response.headers.get("Server", "")
            except Exception:
                results["HTTP"]["status_code"] = None

        return {
            "host": host,
            "services": results,
            "open_count": open_count,
            "total_checked": len(COMMON_SERVICES),
        }

    @staticmethod
    def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
        """快速检测单个端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    # ==================== 集成扫描 ====================

    def full_scan(self, target_ip: Optional[str] = None) -> dict:
        """
        执行完整网络扫描：ARP 扫描 + 服务检测

        先扫描局域网设备，然后对每个设备（或指定设备）检测服务。

        Args:
            target_ip: 指定目标 IP，不指定则扫描所有发现的设备

        Returns:
            dict: 完整扫描报告
        """
        report = {
            "timestamp": time.time(),
            "local_ip": self._get_local_ip(),
            "devices": [],
        }

        # ARP 扫描
        devices = self.scan_arp()
        if not devices:
            logger.warning("未发现局域网设备")
            return report

        # 过滤目标
        if target_ip:
            devices = [d for d in devices if d["ip"] == target_ip]

        # 服务检测
        for device in devices:
            logger.info(f"扫描设备服务: {device['ip']}")
            services = self.detect_services(device["ip"])
            device["services"] = services
            report["devices"].append(device)

        report["device_count"] = len(report["devices"])
        return report

    # ==================== 辅助方法 ====================

    @staticmethod
    def _is_windows() -> bool:
        """判断当前操作系统是否为 Windows"""
        import platform
        return platform.system().lower() == "windows"

    @staticmethod
    def _get_default_ports() -> list[int]:
        """获取默认扫描端口列表（常见端口）"""
        return [
            21, 22, 23, 25, 53, 80, 110, 135, 139, 143,
            389, 443, 445, 554, 631, 993, 995, 1433, 1521,
            3306, 3389, 5432, 5900, 6379, 8000, 8005, 8080,
            8443, 9090, 27017,
        ]

    @staticmethod
    def _get_service_name(port: int) -> str:
        """根据端口号返回常见服务名称"""
        service_map = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
            53: "DNS", 80: "HTTP", 110: "POP3", 135: "RPC",
            139: "NetBIOS", 143: "IMAP", 389: "LDAP", 443: "HTTPS",
            445: "SMB", 554: "RTSP", 631: "IPP", 993: "IMAPS",
            995: "POP3S", 1433: "MSSQL", 1521: "Oracle",
            3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
            5900: "VNC", 6379: "Redis", 8000: "HTTP-Alt",
            8005: "LoveFlow", 8080: "HTTP-Proxy", 8443: "HTTPS-Alt",
            9090: "HTTP-Alt2", 27017: "MongoDB",
        }
        return service_map.get(port, "Unknown")

    # ==================== 上下文注入 ====================

    def get_context(self) -> Optional[str]:
        """
        返回网络状态摘要，供 Agent 上下文注入

        快速扫描局域网并检测关键服务状态，格式化为简短摘要。

        Returns:
            Optional[str]: 网络状态文本
        """
        try:
            parts = ["【网络状态】"]

            # 本机 IP
            local_ip = self._get_local_ip()
            if local_ip:
                parts.append(f"本机: {local_ip}")
            else:
                parts.append("本机: 未知")

            # 局域网设备数
            devices = self.scan_arp()
            parts.append(f"局域网设备: {len(devices)} 台")

            if devices:
                # 检查 LoveFlow 服务是否在局域网中
                loveflow_hosts = []
                for d in devices:
                    svc_check = self.detect_services(d["ip"])
                    for svc in svc_check.get("services", {}).values():
                        if svc.get("port") == 8005 and svc.get("open"):
                            loveflow_hosts.append(d["ip"])
                            break

                if loveflow_hosts:
                    parts.append(f"LoveFlow 服务: {', '.join(loveflow_hosts)}")

                # 检查网关
                if local_ip:
                    subnet = ".".join(local_ip.split(".")[:3])
                    gateway_ips = [f"{subnet}.1", f"{subnet}.254"]
                    found_gateway = any(d["ip"] in gateway_ips for d in devices)
                    if found_gateway:
                        parts.append("网关: 在线")

            return " | ".join(parts)
        except Exception as e:
            logger.error(f"生成网络状态上下文失败: {e}")
            return None
