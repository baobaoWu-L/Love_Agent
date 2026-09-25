"""
协议适配器模块

支持 MQTT、WebSocket、SSH、MySQL、Redis 等多种
外部协议连接管理。所有可选依赖使用 try/except 导入，
缺失时对应方法返回 False/None 并记录警告。
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loveflow.protocol_adapters")

# ==================== 可选依赖导入 ====================

# MQTT (paho-mqtt)
try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    logger.warning("paho-mqtt 库未安装，MQTT 功能不可用。请执行: pip install paho-mqtt")

# WebSocket (websocket-client)
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("websocket-client 库未安装，WebSocket 功能不可用。请执行: pip install websocket-client")

# SSH (paramiko)
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger.warning("paramiko 库未安装，SSH 功能不可用。请执行: pip install paramiko")

# MySQL (pymysql)
try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    logger.warning("pymysql 库未安装，MySQL 功能不可用。请执行: pip install pymysql")

# Redis
try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis 库未安装，Redis 功能不可用。请执行: pip install redis")

# 从配置读取数据库和 Redis 连接信息
from config.agent_config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB,
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB,
)


class ProtocolAdapters:
    """多协议适配器管理器

    管理 MQTT、WebSocket、SSH、MySQL、Redis 的连接生命周期。
    所有依赖库均为可选，缺失时相关操作返回错误值。
    """

    def __init__(self):
        """初始化协议适配器，各连接初始为 None"""
        # MQTT 客户端
        self._mqtt_client: Optional["mqtt.Client"] = None
        self._mqtt_connected = False

        # WebSocket 连接
        self._ws_connection: Optional["websocket.WebSocket"] = None
        self._ws_connected = False

        # SSH 客户端
        self._ssh_client: Optional["paramiko.SSHClient"] = None
        self._ssh_connected = False

        # MySQL 连接
        self._mysql_conn: Optional["pymysql.Connection"] = None
        self._mysql_connected = False

        # Redis 连接
        self._redis_client: Optional["redis_lib.Redis"] = None
        self._redis_connected = False

        logger.info("ProtocolAdapters 初始化完成")

    # ==================== MQTT 方法 ====================

    def mqtt_connect(
        self,
        broker: str,
        port: int = 1883,
        topic: str = "#",
        client_id: str = "loveflow_agent",
    ) -> bool:
        """连接到 MQTT 代理服务器

        Args:
            broker: 代理地址
            port: 端口号
            topic: 订阅主题
            client_id: 客户端 ID

        Returns:
            是否成功连接
        """
        if not PAHO_AVAILABLE:
            logger.error("paho-mqtt 库未安装，无法连接 MQTT")
            return False

        if self._mqtt_connected:
            logger.warning("MQTT 已连接")
            return True

        try:
            self._mqtt_client = mqtt.Client(
                client_id=client_id,
                protocol=mqtt.MQTTv311,
            )

            # 配置日志回调
            self._mqtt_client.on_connect = self._on_mqtt_connect
            self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self._mqtt_client.on_message = self._on_mqtt_message

            # 设置遗嘱消息（可选）
            self._mqtt_client.will_set(
                f"loveflow/agent/{client_id}/status",
                payload="offline",
                qos=1,
                retain=True,
            )

            self._mqtt_client.connect(broker, port, keepalive=60)
            self._mqtt_client.loop_start()  # 启动后台网络循环

            # 订阅主题
            self._mqtt_client.subscribe(topic, qos=1)
            # 发布上线状态
            self._mqtt_client.publish(
                f"loveflow/agent/{client_id}/status",
                payload="online",
                qos=1,
                retain=True,
            )

            self._mqtt_connected = True
            logger.info(f"MQTT 已连接到 {broker}:{port}，订阅主题: {topic}")
            return True

        except Exception as e:
            logger.error(f"MQTT 连接失败: {e}")
            self._mqtt_connected = False
            self._mqtt_client = None
            return False

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT 连接回调"""
        if rc == 0:
            logger.info("MQTT 连接成功 (rc=0)")
        else:
            logger.error(f"MQTT 连接失败，返回码: {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT 断开回调"""
        self._mqtt_connected = False
        if rc != 0:
            logger.warning(f"MQTT 意外断开 (rc={rc})")
        else:
            logger.info("MQTT 正常断开")

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT 消息接收回调"""
        try:
            payload = msg.payload.decode("utf-8")
            logger.debug(f"MQTT 收到消息 - 主题: {msg.topic}, 消息: {payload[:200]}")
        except Exception as e:
            logger.warning(f"MQTT 消息解析失败: {e}")

    def mqtt_publish(self, topic: str, message: str) -> bool:
        """发布消息到 MQTT 主题

        Args:
            topic: 目标主题
            message: 消息内容

        Returns:
            是否成功发布
        """
        if not PAHO_AVAILABLE:
            logger.error("paho-mqtt 库未安装")
            return False

        if not self._mqtt_connected or not self._mqtt_client:
            logger.error("MQTT 未连接")
            return False

        try:
            result = self._mqtt_client.publish(topic, payload=message, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"MQTT 消息已发布到 {topic}")
                return True
            else:
                logger.error(f"MQTT 发布失败，返回码: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"MQTT 发布异常: {e}")
            return False

    def mqtt_disconnect(self):
        """断开 MQTT 连接"""
        if not PAHO_AVAILABLE or not self._mqtt_client:
            return

        try:
            if self._mqtt_connected:
                # 发布离线状态
                client_id = self._mqtt_client._client_id.decode() if hasattr(
                    self._mqtt_client, '_client_id'
                ) and self._mqtt_client._client_id else "unknown"
                self._mqtt_client.publish(
                    f"loveflow/agent/{client_id}/status",
                    payload="offline",
                    qos=1,
                    retain=True,
                )
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            self._mqtt_connected = False
            self._mqtt_client = None
            logger.info("MQTT 已断开")
        except Exception as e:
            logger.error(f"MQTT 断开异常: {e}")

    # ==================== WebSocket 方法 ====================

    def websocket_connect(self, url: str) -> bool:
        """打开 WebSocket 连接

        Args:
            url: WebSocket 地址 (如 ws://host:port/path)

        Returns:
            是否成功连接
        """
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client 库未安装，无法连接 WebSocket")
            return False

        if self._ws_connected:
            logger.warning("WebSocket 已连接")
            return True

        try:
            self._ws_connection = websocket.create_connection(
                url=url,
                timeout=30,
                enable_multithread=True,
            )
            self._ws_connected = True
            logger.info(f"WebSocket 已连接到 {url}")
            return True

        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            self._ws_connection = None
            self._ws_connected = False
            return False

    def websocket_send(self, message: str) -> bool:
        """发送 WebSocket 消息

        Args:
            message: 消息内容（字符串或 JSON 字符串）

        Returns:
            是否成功发送
        """
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client 库未安装")
            return False

        if not self._ws_connected or not self._ws_connection:
            logger.error("WebSocket 未连接")
            return False

        try:
            if isinstance(message, dict):
                message = json.dumps(message, ensure_ascii=False)
            self._ws_connection.send(message)
            logger.debug(f"WebSocket 消息已发送: {message[:100]}")
            return True

        except Exception as e:
            logger.error(f"WebSocket 发送失败: {e}")
            return False

    def websocket_close(self):
        """关闭 WebSocket 连接"""
        if not WEBSOCKET_AVAILABLE or not self._ws_connection:
            return

        try:
            self._ws_connection.close()
            self._ws_connected = False
            self._ws_connection = None
            logger.info("WebSocket 已关闭")
        except Exception as e:
            logger.error(f"WebSocket 关闭异常: {e}")

    # ==================== SSH 方法 ====================

    def ssh_connect(
        self,
        host: str,
        port: int = 22,
        username: str = None,
        password: str = None,
        key_path: str = None,
    ) -> bool:
        """通过 SSH 连接到远程服务器

        Args:
            host: 服务器地址
            port: SSH 端口
            username: 用户名
            password: 密码（与 key_path 二选一）
            key_path: 私钥文件路径（与 password 二选一）

        Returns:
            是否成功连接
        """
        if not PARAMIKO_AVAILABLE:
            logger.error("paramiko 库未安装，无法连接 SSH")
            return False

        if self._ssh_connected:
            logger.warning("SSH 已连接")
            return True

        try:
            self._ssh_client = paramiko.SSHClient()
            # 自动接受未知主机密钥
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": host,
                "port": port,
                "username": username,
                "timeout": 30,
            }

            if password:
                connect_kwargs["password"] = password
            if key_path:
                connect_kwargs["key_filename"] = key_path

            self._ssh_client.connect(**connect_kwargs)
            self._ssh_connected = True
            logger.info(f"SSH 已连接到 {username}@{host}:{port}")
            return True

        except paramiko.AuthenticationException:
            logger.error("SSH 认证失败")
            self._ssh_client = None
            return False
        except paramiko.SSHException as e:
            logger.error(f"SSH 连接异常: {e}")
            self._ssh_client = None
            return False
        except Exception as e:
            logger.error(f"SSH 连接失败: {e}")
            self._ssh_client = None
            return False

    def ssh_execute(self, command: str) -> Optional[str]:
        """通过 SSH 执行远程命令

        Args:
            command: 要执行的命令

        Returns:
            命令输出（stdout + stderr），失败返回 None
        """
        if not PARAMIKO_AVAILABLE:
            logger.error("paramiko 库未安装")
            return None

        if not self._ssh_connected or not self._ssh_client:
            logger.error("SSH 未连接")
            return None

        try:
            stdin, stdout, stderr = self._ssh_client.exec_command(
                command,
                timeout=60,
            )
            exit_code = stdout.channel.recv_exit_status()
            output = stdout.read().decode("utf-8", errors="replace")
            error_output = stderr.read().decode("utf-8", errors="replace")

            if exit_code != 0:
                logger.warning(f"SSH 命令返回非零退出码: {exit_code}")
                if error_output:
                    output += f"\n[STDERR]\n{error_output}"

            logger.info(f"SSH 命令执行完成 (退出码: {exit_code})")
            return output

        except Exception as e:
            logger.error(f"SSH 命令执行失败: {e}")
            return None

    def ssh_disconnect(self):
        """关闭 SSH 连接"""
        if not PARAMIKO_AVAILABLE or not self._ssh_client:
            return

        try:
            self._ssh_client.close()
            self._ssh_connected = False
            self._ssh_client = None
            logger.info("SSH 已断开")
        except Exception as e:
            logger.error(f"SSH 断开异常: {e}")

    # ==================== MySQL 方法 ====================

    def mysql_query(self, sql: str) -> Optional[list[dict]]:
        """对 LoveFlow 数据库执行 MySQL 查询

        使用 agent_config 中配置的数据库连接信息。

        Args:
            sql: SQL 查询语句（仅支持 SELECT）

        Returns:
            查询结果列表（每行为字典），失败返回 None
        """
        if not PYMYSQL_AVAILABLE:
            logger.error("pymysql 库未安装，无法查询 MySQL")
            return None

        # 检查是否为 SELECT 查询
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith("SELECT"):
            logger.warning("仅支持 SELECT 查询，拒绝执行非查询语句")
            return None

        try:
            # 每次查询创建新连接，避免连接过期
            conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=10,
                read_timeout=30,
            )

            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    results = cursor.fetchall()
                    # DictCursor 返回的已经是字典列表
                    return [dict(row) for row in results] if results else []
            finally:
                conn.close()

        except pymysql.Error as e:
            logger.error(f"MySQL 查询失败: {e}")
            return None
        except Exception as e:
            logger.error(f"MySQL 查询异常: {e}")
            return None

    # ==================== Redis 方法 ====================

    def redis_get(self, key: str) -> Optional[str]:
        """获取 Redis 键值

        Args:
            key: 键名

        Returns:
            键值字符串，不存在或失败返回 None
        """
        if not REDIS_AVAILABLE:
            logger.error("redis 库未安装，无法操作 Redis")
            return None

        try:
            client = self._get_redis_client()
            if not client:
                return None
            value = client.get(key)
            if value is not None:
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)
            return None

        except Exception as e:
            logger.error(f"Redis GET 失败: {e}")
            return None

    def redis_set(self, key: str, value: str) -> bool:
        """设置 Redis 键值

        Args:
            key: 键名
            value: 值

        Returns:
            是否成功设置
        """
        if not REDIS_AVAILABLE:
            logger.error("redis 库未安装，无法操作 Redis")
            return False

        try:
            client = self._get_redis_client()
            if not client:
                return False
            client.set(key, value)
            logger.debug(f"Redis SET {key} = {value[:50]}")
            return True

        except Exception as e:
            logger.error(f"Redis SET 失败: {e}")
            return False

    def _get_redis_client(self):
        """获取或创建 Redis 连接"""
        if self._redis_client is not None:
            try:
                self._redis_client.ping()
                return self._redis_client
            except Exception:
                # 连接已断开，重新创建
                self._redis_client = None

        try:
            self._redis_client = redis_lib.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD if REDIS_PASSWORD else None,
                db=REDIS_DB,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self._redis_client.ping()
            self._redis_connected = True
            logger.info(f"Redis 已连接到 {REDIS_HOST}:{REDIS_PORT}")
            return self._redis_client
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            self._redis_connected = False
            return None

    # ==================== 状态查询 ====================

    def get_connections(self) -> dict:
        """返回所有活动连接的状态

        Returns:
            各协议连接状态字典
        """
        return {
            "mqtt": {
                "connected": self._mqtt_connected,
                "client_id": self._mqtt_client._client_id.decode() if (
                    self._mqtt_client and hasattr(self._mqtt_client, '_client_id')
                    and self._mqtt_client._client_id
                ) else None,
            } if PAHO_AVAILABLE else {"available": False},
            "websocket": {
                "connected": self._ws_connected,
            } if WEBSOCKET_AVAILABLE else {"available": False},
            "ssh": {
                "connected": self._ssh_connected,
            } if PARAMIKO_AVAILABLE else {"available": False},
            "mysql": {
                "available": PYMYSQL_AVAILABLE,
                "host": MYSQL_HOST,
                "database": MYSQL_DB,
            },
            "redis": {
                "connected": self._redis_connected,
                "host": REDIS_HOST,
            } if REDIS_AVAILABLE else {"available": False},
        }

    def get_context(self) -> Optional[str]:
        """获取活动连接摘要信息

        Returns:
            格式化的连接状态字符串，无活动连接时返回 None
        """
        lines = ["【外部协议连接状态】"]
        has_active = False

        # MQTT
        if PAHO_AVAILABLE:
            status = "已连接" if self._mqtt_connected else "未连接"
            lines.append(f"  - MQTT: {status}")
            has_active = has_active or self._mqtt_connected
        else:
            lines.append("  - MQTT: 未安装")

        # WebSocket
        if WEBSOCKET_AVAILABLE:
            status = "已连接" if self._ws_connected else "未连接"
            lines.append(f"  - WebSocket: {status}")
            has_active = has_active or self._ws_connected
        else:
            lines.append("  - WebSocket: 未安装")

        # SSH
        if PARAMIKO_AVAILABLE:
            status = "已连接" if self._ssh_connected else "未连接"
            lines.append(f"  - SSH: {status}")
            has_active = has_active or self._ssh_connected
        else:
            lines.append("  - SSH: 未安装")

        # MySQL
        if PYMYSQL_AVAILABLE:
            lines.append(f"  - MySQL: {MYSQL_HOST}/{MYSQL_DB}")
            has_active = True
        else:
            lines.append("  - MySQL: 未安装")

        # Redis
        if REDIS_AVAILABLE:
            status = "已连接" if self._redis_connected else "未连接"
            lines.append(f"  - Redis ({REDIS_HOST}): {status}")
            has_active = has_active or self._redis_connected
        else:
            lines.append("  - Redis: 未安装")

        if not has_active:
            return None

        return "\n".join(lines)

    def disconnect_all(self):
        """断开所有活动连接"""
        logger.info("正在断开所有外部连接...")
        self.mqtt_disconnect()
        self.websocket_close()
        self.ssh_disconnect()
        self._mysql_connected = False
        self._redis_connected = False
        self._redis_client = None
        logger.info("所有外部连接已断开")
