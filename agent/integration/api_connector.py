"""
API 连接器模块

管理外部 REST API 连接，支持多种认证方式，
提供 OpenAI API 兼容接口调用能力。
连接配置持久化存储在 Agent 记忆系统中。
"""
import json
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

logger = logging.getLogger("loveflow.api_connector")

# 可选导入 requests 库
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests 库未安装，API 请求功能不可用。请执行: pip install requests")

# 从 Agent 记忆系统导入持久化存储函数
from agent.core.memory import add_memory, search_memory, delete_memory


class ApiConnector:
    """外部 REST API 连接管理器

    支持保存多个 API 连接配置，发送 HTTP 请求，
    以及调用 OpenAI 兼容接口。
    """

    def __init__(self):
        """初始化，从记忆系统加载已保存的 API 连接"""
        self._connections: dict[str, dict] = {}
        self._load_connections()
        logger.info(f"ApiConnector 初始化完成，已加载 {len(self._connections)} 个连接")

    def _load_connections(self):
        """从记忆系统加载所有已保存的 API 连接"""
        results = search_memory(
            query="api_connection",
            doc_type="api_connection",
            limit=100,
        )
        for item in results:
            try:
                data = json.loads(item["content"])
                name = data.get("name", item["title"])
                self._connections[name] = {
                    "base_url": data.get("base_url", ""),
                    "auth_type": data.get("auth_type", "none"),
                    "auth_token": data.get("auth_token", ""),
                    "memory_id": item["id"],
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"解析 API 连接记忆失败 (id={item['id']}): {e}")

    def add_connection(
        self,
        name: str,
        base_url: str,
        auth_type: str = "none",
        auth_token: str = "",
    ) -> int:
        """保存 API 连接到记忆系统

        Args:
            name: 连接名称（唯一标识）
            base_url: API 基础地址
            auth_type: 认证类型，支持 "none", "bearer", "api_key", "basic"
            auth_token: 认证令牌/密钥

        Returns:
            记忆条目 ID
        """
        content = json.dumps({
            "name": name,
            "base_url": base_url,
            "auth_type": auth_type,
            "auth_token": auth_token,
        }, ensure_ascii=False)

        memory_id = add_memory(
            content=content,
            doc_type="api_connection",
            title=name,
            tags=["api", auth_type],
            source="api_connector",
            importance=3,
        )

        # 更新本地缓存
        self._connections[name] = {
            "base_url": base_url,
            "auth_type": auth_type,
            "auth_token": auth_token,
            "memory_id": memory_id,
        }
        logger.info(f"已保存 API 连接: {name} ({base_url})")
        return memory_id

    def remove_connection(self, name: str) -> bool:
        """删除已保存的 API 连接

        Args:
            name: 连接名称

        Returns:
            是否成功删除
        """
        conn = self._connections.get(name)
        if not conn:
            logger.warning(f"未找到 API 连接: {name}")
            return False

        memory_id = conn.get("memory_id")
        if memory_id:
            delete_memory(memory_id)
        del self._connections[name]
        logger.info(f"已删除 API 连接: {name}")
        return True

    def list_connections(self) -> list[dict]:
        """列出所有已保存的 API 连接

        Returns:
            连接配置列表（不包含 auth_token 敏感信息）
        """
        result = []
        for name, conn in self._connections.items():
            result.append({
                "name": name,
                "base_url": conn["base_url"],
                "auth_type": conn["auth_type"],
                # 不返回完整 token，仅显示前几位
                "auth_token_prefix": conn["auth_token"][:8] + "..." if conn["auth_token"] else "",
            })
        return result

    def request(
        self,
        name: str,
        method: str = "GET",
        endpoint: str = "",
        data: dict = None,
        headers: dict = None,
    ) -> Optional[dict]:
        """发送 HTTP 请求到已保存的 API 连接

        Args:
            name: 连接名称
            method: HTTP 方法（GET, POST, PUT, DELETE 等）
            endpoint: API 端点路径
            data: 请求体数据（字典）
            headers: 额外的请求头

        Returns:
            JSON 响应数据，失败返回 None
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests 库未安装，无法发送请求")
            return None

        conn = self._connections.get(name)
        if not conn:
            logger.error(f"未找到 API 连接: {name}")
            return None

        # 构建完整 URL
        base_url = conn["base_url"].rstrip("/")
        endpoint = endpoint.lstrip("/")
        full_url = urljoin(base_url + "/", endpoint)

        # 构建请求头
        request_headers = {}
        if headers:
            request_headers.update(headers)

        # 根据认证类型添加认证头
        auth_type = conn.get("auth_type", "none")
        auth_token = conn.get("auth_token", "")

        if auth_type == "bearer" and auth_token:
            request_headers["Authorization"] = f"Bearer {auth_token}"
        elif auth_type == "api_key" and auth_token:
            request_headers["X-API-Key"] = auth_token
            request_headers["Authorization"] = f"Bearer {auth_token}"
        elif auth_type == "basic" and auth_token:
            request_headers["Authorization"] = f"Basic {auth_token}"

        # 设置默认 Content-Type
        if data is not None and "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"

        logger.info(f"发送 {method} 请求到 {full_url} (连接: {name})")

        try:
            response = requests.request(
                method=method.upper(),
                url=full_url,
                json=data if data else None,
                headers=request_headers,
                timeout=30,
            )
            response.raise_for_status()

            # 尝试解析 JSON 响应
            try:
                return response.json()
            except (json.JSONDecodeError, ValueError):
                # 非 JSON 响应，返回文本包装
                return {"status_code": response.status_code, "text": response.text}

        except requests.exceptions.Timeout:
            logger.error(f"请求超时: {full_url}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"连接失败: {full_url} - {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP 错误: {e}")
            try:
                return {"error": str(e), "response": e.response.json()}
            except (json.JSONDecodeError, AttributeError):
                return {"error": str(e)}
        except Exception as e:
            logger.error(f"请求异常: {e}")
            return None

    def call_openai(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        api_key: str = None,
        base_url: str = None,
    ) -> Optional[str]:
        """调用 OpenAI API 兼容接口进行对话

        Args:
            messages: 对话消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称
            api_key: OpenAI API 密钥（如不提供则从环境变量读取）
            base_url: API 基础地址（如不提供则使用 OpenAI 官方地址）

        Returns:
            响应文本内容，失败返回 None
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests 库未安装，无法调用 OpenAI API")
            return None

        # 如果未提供 api_key，尝试从第一个 bearer 连接中获取
        if not api_key:
            for name, conn in self._connections.items():
                if conn["auth_type"] in ("bearer", "api_key") and conn["auth_token"]:
                    api_key = conn["auth_token"]
                    if not base_url:
                        base_url = conn["base_url"]
                    break

        if not api_key:
            logger.error("未提供 API 密钥，无法调用 OpenAI 接口")
            return None

        target_url = (base_url or "https://api.openai.com").rstrip("/")
        # 如果 base_url 不以 /v1 结尾，自动补全
        if not target_url.endswith("/v1"):
            # 检查是否已经包含 chat/completions 路径
            if "chat/completions" not in target_url:
                target_url += "/v1/chat/completions"
            else:
                target_url = target_url

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        logger.info(f"调用 OpenAI 兼容接口: model={model}")

        try:
            response = requests.post(
                url=target_url,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            # 提取响应文本
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                return content

            logger.warning("OpenAI 响应中未找到 choices")
            return None

        except requests.exceptions.Timeout:
            logger.error("OpenAI API 请求超时")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"OpenAI API 连接失败: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"OpenAI API HTTP 错误: {e}")
            try:
                error_detail = e.response.json()
                logger.error(f"错误详情: {error_detail}")
            except (json.JSONDecodeError, AttributeError):
                pass
            return None
        except Exception as e:
            logger.error(f"OpenAI API 调用异常: {e}")
            return None

    def get_context(self) -> Optional[str]:
        """获取已连接 API 的摘要信息，供 Agent 上下文使用

        Returns:
            格式化的连接摘要字符串，无连接时返回 None
        """
        if not self._connections:
            return None

        lines = ["【已连接的 API 服务】"]
        for name, conn in self._connections.items():
            auth_display = conn["auth_type"]
            if auth_display == "bearer":
                auth_display = "Bearer 令牌"
            elif auth_display == "api_key":
                auth_display = "API 密钥"
            elif auth_display == "basic":
                auth_display = "Basic 认证"
            elif auth_display == "none":
                auth_display = "无认证"

            lines.append(f"  - {name}: {conn['base_url']} (认证: {auth_display})")

        if len(lines) > 1:
            return "\n".join(lines)
        return None
