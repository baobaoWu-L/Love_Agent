"""
Webhook 管理器模块

提供 Webhook 接收服务器和本地事件系统，
支持注册事件处理器、触发事件以及在后台线程中
运行 HTTP 服务接收外部 Webhook 请求。
"""
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("loveflow.webhook")


class _WebhookRequestHandler(BaseHTTPRequestHandler):
    """Webhook HTTP 请求处理器

    接收 POST 请求并传递给 WebhookManager 实例处理。
    """

    # 保存外部引用，避免跨处理器传递
    manager_ref: "WebhookManager" = None

    def do_POST(self):
        """处理 POST 请求（Webhook 回调）"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Webhook 收到无效 JSON: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "无效的 JSON 数据"}).encode())
            return

        # 调用管理器的内部处理
        if self.manager_ref:
            self.manager_ref._handle_webhook_post(data)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "received"}).encode())

    def do_GET(self):
        """处理 GET 请求（健康检查）"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        status = {
            "status": "running",
            "event_handlers": list(self.manager_ref._handlers.keys()) if self.manager_ref else [],
        }
        self.wfile.write(json.dumps(status).encode())

    def log_message(self, format: str, *args):
        """用 logger 替代 stderr 输出"""
        logger.debug(f"Webhook HTTP: {format % args}")


class WebhookManager:
    """Webhook 接收器和事件系统管理器

    支持：
    - 注册事件处理器 (on)
    - 本地事件触发 (trigger)
    - 后台 HTTP 服务接收外部 Webhook (start_server / stop_server)
    """

    def __init__(self):
        """初始化 Webhook 事件处理器字典"""
        self._handlers: dict[str, list[Callable]] = {}
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        logger.info("WebhookManager 初始化完成")

    def on(self, event_type: str, handler: Callable):
        """注册事件处理器

        Args:
            event_type: 事件类型名称（如 "task_completed", "deploy_finished"）
            handler: 处理函数，接收 (event_type, data) 参数
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(f"已注册事件处理器: {event_type} (共 {len(self._handlers[event_type])} 个)")

    def trigger(self, event_type: str, data: dict):
        """触发本地事件，调用所有已注册的处理器

        Args:
            event_type: 事件类型
            data: 事件数据字典
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            logger.debug(f"未找到事件 '{event_type}' 的处理器")
            return

        logger.info(f"触发事件: {event_type} (通知 {len(handlers)} 个处理器)")
        for handler in handlers:
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error(f"事件处理器执行失败 ({event_type}): {e}")

        # 同时触发通配符事件 "*"
        wildcard_handlers = self._handlers.get("*", [])
        for handler in wildcard_handlers:
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error(f"通配符事件处理器执行失败: {e}")

    def start_server(self, host: str = "0.0.0.0", port: int = 9090) -> bool:
        """在后台线程中启动 Webhook HTTP 服务器

        Args:
            host: 监听地址
            port: 监听端口

        Returns:
            是否成功启动
        """
        if self._running:
            logger.warning("Webhook 服务器已在运行中")
            return True

        try:
            # 绑定请求处理器到管理器实例
            _WebhookRequestHandler.manager_ref = self

            self._server = HTTPServer((host, port), _WebhookRequestHandler)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                name="webhook-server",
                daemon=True,
            )
            self._server_thread.start()
            self._running = True
            logger.info(f"Webhook 服务器已启动: {host}:{port}")
            return True

        except OSError as e:
            logger.error(f"Webhook 服务器启动失败 (地址 {host}:{port} 可能已被占用): {e}")
            return False
        except Exception as e:
            logger.error(f"Webhook 服务器启动异常: {e}")
            return False

    def stop_server(self):
        """停止 Webhook 服务器"""
        if not self._running or not self._server:
            logger.warning("Webhook 服务器未运行")
            return

        try:
            self._server.shutdown()
            self._server.server_close()
            self._running = False
            self._server_thread = None
            self._server = None
            logger.info("Webhook 服务器已停止")
        except Exception as e:
            logger.error(f"停止 Webhook 服务器异常: {e}")

    def list_handlers(self) -> list[str]:
        """列出所有已注册的事件类型

        Returns:
            事件类型名称列表
        """
        return list(self._handlers.keys())

    def _handle_webhook_post(self, data: dict):
        """内部方法：解析接收到的 Webhook 数据并触发事件

        Webhook 数据格式支持两种模式：
        1. 标准模式: {"event": "event_type", "data": {...}}
        2. 直接模式: {"type": "event_type", "payload": {...}}
        3. 扁平模式: 整个 data 作为事件数据，事件类型从 "event" 字段获取

        Args:
            data: 解析后的 Webhook JSON 数据
        """
        logger.info(f"收到 Webhook 请求: {json.dumps(data, ensure_ascii=False)[:200]}")

        event_type = None
        event_data = None

        # 尝试多种常见 Webhook 数据格式
        if "event" in data and "data" in data:
            # 标准格式: {"event": "push", "data": {...}}
            event_type = data["event"]
            event_data = data["data"]
        elif "type" in data and "payload" in data:
            # GitHub/GitLab 风格: {"type": "push", "payload": {...}}
            event_type = data["type"]
            event_data = data["payload"]
        elif "action" in data:
            # GitHub Webhook 风格
            event_type = data.get("action", "unknown")
            event_data = data
        elif "event_type" in data:
            event_type = data["event_type"]
            event_data = data.get("data", data)
        else:
            # 无法识别事件类型，使用默认值
            event_type = "webhook_received"
            event_data = data

        # 触发事件
        self.trigger(event_type, event_data or data)

    def get_context(self) -> Optional[str]:
        """获取 Webhook 状态摘要信息

        Returns:
            格式化的状态字符串，无处理器且未运行时返回 None
        """
        parts = []
        if self._running:
            server_info = f"运行中"
            if self._server:
                addr = self._server.server_address
                server_info = f"运行中 ({addr[0]}:{addr[1]})"
            parts.append(f"  - 服务器: {server_info}")

        if self._handlers:
            handler_list = ", ".join(self.list_handlers())
            parts.append(f"  - 已注册事件: {handler_list}")
            total = sum(len(hs) for hs in self._handlers.values())
            parts.append(f"  - 处理器总数: {total}")

        if not parts:
            return None

        lines = ["【Webhook 状态】"]
        lines.extend(parts)
        return "\n".join(lines)
