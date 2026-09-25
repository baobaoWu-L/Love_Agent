"""
启动自检与自动依赖补全

每次 Agent 启动时：
  1. 扫描 requirements.txt 中的核心依赖
  2. 检测缺失的包
  3. 自动 pip install 缺失包
  4. 可选包延迟安装（用到时才装）
  5. 环境检测报告
"""
import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loveflow.deps")


# 核心依赖（必须存在）
CORE_REQUIREMENTS = [
    "requests",
    "httpx",
    "schedule",
    "yaml",
]

# 可选依赖（延迟安装）
OPTIONAL_REQUIREMENTS = {
    "psutil": "系统监控",
    "cv2": "摄像头/图像处理（opencv-python）",
    "face_recognition": "人像识别",
    "nmap": "网络扫描（python-nmap）",
    "paho.mqtt": "MQTT 协议",
    "websocket": "WebSocket",
    "paramiko": "SSH 连接",
    "pymysql": "MySQL 连接",
    "redis": "Redis 连接",
    "fastapi": "Webhook 服务器",
    "uvicorn": "Webhook 服务器",
    "matplotlib": "训练曲线绘制",
}

# import 名称 -> pip 包名映射
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "face_recognition": "face-recognition",
    "nmap": "python-nmap",
    "paho.mqtt": "paho-mqtt",
    "websocket": "websocket-client",
    "yaml": "pyyaml",
}


def check_dependencies() -> dict:
    """
    检查所有依赖状态

    Returns:
        {"core": {"installed": [...], "missing": [...]},
         "optional": {"installed": {...}, "missing": {...}}}
    """
    result = {"core": {"installed": [], "missing": []}, "optional": {"installed": {}, "missing": {}}}

    # 检查核心依赖
    for pkg in CORE_REQUIREMENTS:
        if _is_installed(pkg):
            result["core"]["installed"].append(pkg)
        else:
            result["core"]["missing"].append(pkg)

    # 检查可选依赖
    for import_name, description in OPTIONAL_REQUIREMENTS.items():
        if _is_installed(import_name):
            result["optional"]["installed"][import_name] = description
        else:
            result["optional"]["missing"][import_name] = description

    return result


def auto_install_core() -> list[str]:
    """自动安装缺失的核心依赖，返回已安装的包名列表"""
    status = check_dependencies()
    installed = []

    for pkg in status["core"]["missing"]:
        pip_name = IMPORT_TO_PACKAGE.get(pkg, pkg)
        logger.info(f"正在安装核心依赖: {pip_name}")
        if _pip_install(pip_name):
            installed.append(pkg)
            logger.info(f"核心依赖安装成功: {pip_name}")
        else:
            logger.error(f"核心依赖安装失败: {pip_name}")

    return installed


def ensure_dependency(import_name: str) -> bool:
    """
    确保某个依赖可用（缺失则尝试安装）

    Returns:
        是否可用
    """
    if _is_installed(import_name):
        return True

    pip_name = IMPORT_TO_PACKAGE.get(import_name, import_name)
    logger.info(f"正在安装依赖: {pip_name}")
    success = _pip_install(pip_name)
    if success:
        logger.info(f"依赖安装成功: {pip_name}")
    else:
        logger.error(f"依赖安装失败: {pip_name}")
    return success


def get_environment_report() -> dict:
    """
    环境检测报告

    Returns:
        {"python_version": "...", "platform": "...",
         "cuda_available": bool, "loveflow_infer_online": bool, ...}
    """
    report = {
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # CUDA 检查
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            report["cuda_device_count"] = torch.cuda.device_count()
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["cuda_available"] = False

    # LoveFlow 推理服务器检查
    try:
        import requests
        resp = requests.get("http://localhost:8005/v1/health", timeout=3)
        report["loveflow_infer_online"] = resp.status_code == 200
    except:
        report["loveflow_infer_online"] = False

    # 关键路径检查
    from config.agent_config import LOVEFLOW_DIR, MODEL_DIR, BASE_MODEL_PATH
    report["loveflow_dir_exists"] = Path(LOVEFLOW_DIR).exists()
    report["model_dir_exists"] = Path(MODEL_DIR).exists()
    report["base_model_exists"] = Path(BASE_MODEL_PATH).exists()

    return report


def _is_installed(module_name: str) -> bool:
    """检查 Python 模块是否已安装"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _pip_install(package_name: str) -> bool:
    """执行 pip install"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"pip install 失败 [{package_name}]: {e}")
        return False
