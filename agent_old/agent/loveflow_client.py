"""
LoveFlow 本地推理客户端（增强版）

通过 HTTP 调用本机推理服务器，支持：
  - 非流式对话
  - 流式对话（SSE）
  - 文本向量化
  - 健康检查
"""
import json
from typing import Optional

import httpx
import requests

from config.agent_config import INFER_BASE_URL


class LoveFlowClient:
    """LoveFlow 本地模型推理客户端"""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or INFER_BASE_URL).rstrip("/")

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.4) -> Optional[str]:
        """
        非流式对话

        messages: [{"role": "user/assistant/system", "content": "..."}, ...]
        """
        try:
            prompt = self._build_prompt(messages)
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            resp = requests.post(
                f"{self.base_url}/v1/chat",
                json=data,
                timeout=300,
            )
            result = resp.json()
            return result.get("response")
        except Exception as e:
            return f"[推理错误: {e}]"

    def chat_stream(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.4):
        """
        流式对话（SSE）

        返回生成器，逐 token 产出字符串
        """
        try:
            prompt = self._build_prompt(messages)
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            with httpx.stream("POST", f"{self.base_url}/v1/chat/stream", json=data, timeout=120) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        # 解析 JSON 格式的 SSE 数据
                        try:
                            obj = json.loads(payload)
                            if obj.get("done"):
                                break
                            if "content" in obj:
                                yield obj["content"]
                        except json.JSONDecodeError:
                            yield payload
        except Exception as e:
            yield f"[流式错误: {e}]"

    def embed(self, text: str) -> Optional[list[float]]:
        """获取文本向量"""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/embed",
                json={"text": text},
                timeout=30,
            )
            result = resp.json()
            return result.get("vector")
        except Exception as e:
            return None

    def health(self) -> bool:
        """检查推理服务器是否在线"""
        try:
            resp = requests.get(f"{self.base_url}/v1/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def _build_prompt(self, messages: list[dict]) -> str:
        """将消息列表构建为统一 prompt 文本"""
        prompt = ""
        for m in messages:
            if m["role"] == "system":
                prompt += f"{m['content']}\n"
            elif m["role"] == "user":
                prompt += f"用户: {m['content']}\n"
            elif m["role"] == "assistant":
                prompt += f"LoveFlow: {m['content']}\n"
        prompt += "LoveFlow: "
        return prompt
