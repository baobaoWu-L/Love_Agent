"""
摄像头管理模块：USB 摄像头发现、RTSP 连接、画面抓拍

为 Agent 提供视觉感知能力，支持本地 USB 摄像头和网络 RTSP 流。
所有可选依赖均使用懒加载，缺失时以降级模式运行。
"""
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import MEMORY_DIR, CAMERA_TIMEOUT

logger = logging.getLogger("loveflow.CAMERA")

# 尝试导入 OpenCV
cv2 = None
try:
    import cv2 as _cv2

    cv2 = _cv2
except ImportError:
    logger.warning("OpenCV (cv2) 未安装，摄像头模块将以降级模式运行。请执行: pip install opencv-python")

# 摄像头配置保存目录
CAMERA_CONFIG_DIR = MEMORY_DIR / "cameras"
CAMERA_SNAPSHOT_DIR = MEMORY_DIR / "snapshots"
CAMERA_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


class CameraManager:
    """摄像头管理器：发现、连接、抓拍"""

    def __init__(self):
        """初始化摄像头管理器"""
        self._cameras: dict[str, dict] = {}  # camera_id -> {name, type, url/idx, config}
        self._load_saved_cameras()
        logger.info("摄像头管理模块初始化完成")

    # ==================== 摄像头发现 ====================

    def discover_cameras(self, max_indices: int = 5) -> list[int]:
        """
        发现可用的 USB 摄像头

        依次尝试 /dev/video* (Linux) 和 0~max_indices (通用) 索引，
        返回所有可成功打开的摄像头索引。

        Args:
            max_indices: 最大检测索引数量

        Returns:
            list[int]: 可用的摄像头索引列表
        """
        if cv2 is None:
            logger.warning("OpenCV 未安装，无法发现摄像头")
            return []

        available = []
        for idx in range(max_indices):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows 下使用 DShow 加速
                if cap.isOpened():
                    # 尝试读取一帧验证
                    ret, _ = cap.read()
                    if ret:
                        available.append(idx)
                        logger.info(f"发现摄像头: 索引 {idx}")
                    else:
                        logger.debug(f"摄像头索引 {idx} 可打开但无法读取画面")
                cap.release()
            except Exception as e:
                logger.debug(f"检测摄像头索引 {idx} 失败: {e}")

        if not available:
            logger.info("未发现任何 USB 摄像头")

        return available

    # ==================== RTSP 连接 ====================

    def connect_rtsp(self, url: str, name: str = "") -> bool:
        """
        连接到 RTSP 网络摄像头

        验证 RTSP 地址可连接，保存配置到本地以便后续使用。

        Args:
            url: RTSP 地址，例如 rtsp://admin:pass@192.168.1.100:554/stream1
            name: 摄像头名称（可选）

        Returns:
            bool: 连接是否成功
        """
        if cv2 is None:
            logger.warning("OpenCV 未安装，无法连接 RTSP")
            return False

        if not url.startswith("rtsp://"):
            logger.error(f"无效的 RTSP 地址: {url}")
            return False

        cap = None
        try:
            cap = cv2.VideoCapture(url)
            # 设置超时
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAMERA_TIMEOUT * 1000)

            if not cap.isOpened():
                logger.warning(f"无法连接到 RTSP: {url}")
                return False

            # 尝试读取一帧验证
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"RTSP 连接成功但无法读取画面: {url}")
                return False

            camera_id = f"rtsp_{uuid.uuid4().hex[:8]}"
            display_name = name or f"RTSP Camera ({url[:40]}...)"

            config = {
                "id": camera_id,
                "name": display_name,
                "type": "rtsp",
                "url": url,
                "created_at": time.time(),
                "last_connected": time.time(),
            }

            self._cameras[camera_id] = config
            self._save_camera_config(camera_id, config)
            logger.info(f"RTSP 摄像头添加成功: {display_name} ({url})")
            return True

        except Exception as e:
            logger.error(f"连接 RTSP 失败: {e}")
            return False
        finally:
            if cap is not None:
                cap.release()

    # ==================== 抓拍 ====================

    def snapshot(self, camera_id: str = "0") -> Optional[str]:
        """
        从指定摄像头拍摄一张照片

        支持索引号 (如 "0")、RTSP ID (如 "rtsp_abc123") 或 RTSP URL。

        Args:
            camera_id: 摄像头标识，默认为 "0"（第一个 USB 摄像头）

        Returns:
            Optional[str]: 保存的图片文件路径，失败返回 None
        """
        if cv2 is None:
            logger.warning("OpenCV 未安装，无法拍照")
            return None

        cap = None
        try:
            # 判断摄像头类型
            source = None
            is_usb = False

            if camera_id in self._cameras:
                cam = self._cameras[camera_id]
                if cam["type"] == "rtsp":
                    source = cam["url"]
                else:
                    try:
                        source = int(camera_id)
                        is_usb = True
                    except ValueError:
                        logger.error(f"无效的摄像头 ID: {camera_id}")
                        return None
            else:
                # 视为索引号或 RTSP URL
                try:
                    source = int(camera_id)
                    is_usb = True
                except ValueError:
                    # 可能是 RTSP URL
                    if camera_id.startswith("rtsp://"):
                        source = camera_id
                    else:
                        logger.error(f"无效的摄像头标识: {camera_id}")
                        return None

            # 打开摄像头
            if is_usb:
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                logger.error(f"无法打开摄像头: {camera_id}")
                return None

            # 等待摄像头稳定
            time.sleep(0.5)

            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                logger.error(f"无法从摄像头 {camera_id} 读取画面")
                return None

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"snapshot_{camera_id.replace(':', '_').replace('/', '_')}_{timestamp}.jpg"
            filepath = CAMERA_SNAPSHOT_DIR / filename

            # 保存图片
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(filepath))

            logger.info(f"抓拍成功: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"抓拍失败: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()

    # ==================== 配置管理 ====================

    def list_saved_cameras(self) -> list[dict]:
        """
        列出所有已保存的摄像头配置

        Returns:
            list[dict]: 摄像头配置列表
        """
        if not self._cameras:
            self._load_saved_cameras()
        return list(self._cameras.values())

    def _load_saved_cameras(self):
        """从本地加载已保存的摄像头配置"""
        self._cameras = {}
        if not CAMERA_CONFIG_DIR.exists():
            return

        for config_file in CAMERA_CONFIG_DIR.glob("*.json"):
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                cid = config.get("id")
                if cid:
                    self._cameras[cid] = config
            except Exception as e:
                logger.warning(f"加载摄像头配置失败 {config_file.name}: {e}")

        if self._cameras:
            logger.info(f"已加载 {len(self._cameras)} 个摄像头配置")

    def _save_camera_config(self, camera_id: str, config: dict):
        """保存摄像头配置到本地"""
        config_file = CAMERA_CONFIG_DIR / f"{camera_id}.json"
        try:
            config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"保存摄像头配置失败: {e}")

    def delete_camera(self, camera_id: str) -> bool:
        """
        删除已保存的摄像头配置

        Args:
            camera_id: 摄像头标识

        Returns:
            bool: 是否删除成功
        """
        if camera_id in self._cameras:
            del self._cameras[camera_id]
            config_file = CAMERA_CONFIG_DIR / f"{camera_id}.json"
            if config_file.exists():
                config_file.unlink()
            logger.info(f"已删除摄像头配置: {camera_id}")
            return True
        return False

    # ==================== 上下文注入 ====================

    def get_context(self) -> Optional[str]:
        """
        返回摄像头状态摘要，供 Agent 上下文注入

        Returns:
            Optional[str]: 摄像头状态文本
        """
        if cv2 is None:
            return None

        try:
            # 快速扫描可用 USB 摄像头
            available_usb = self.discover_cameras(max_indices=5)
            saved = list_saved = self.list_saved_cameras()

            parts = ["【摄像头状态】"]

            if available_usb:
                parts.append(f"USB 摄像头: {len(available_usb)} 个可用 (索引: {available_usb})")
            else:
                parts.append("USB 摄像头: 未检测到")

            if saved:
                rtsp_count = sum(1 for c in saved if c.get("type") == "rtsp")
                parts.append(f"已配置: {len(saved)} 个 (RTSP: {rtsp_count})")
            else:
                parts.append("已配置: 无")

            return " | ".join(parts)
        except Exception as e:
            logger.error(f"生成摄像头上下文失败: {e}")
            return None
