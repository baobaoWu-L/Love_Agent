"""
LoveFlow Agent 配置管理

所有路径和参数集中管理，支持环境变量覆盖。
"""
import os
from pathlib import Path

# ==================== 项目路径 ====================
# Agent 自身目录
AGENT_DIR = Path(__file__).resolve().parent.parent
# 记忆数据目录
MEMORY_DIR = AGENT_DIR / "memory"
# 扩展插件目录
EXTENSIONS_DIR = AGENT_DIR / "agent" / "extensions"

# ==================== LoveFlow 项目路径 ====================
# LoveFlow 项目根目录（默认为 D:\LoveFlow）
LOVEFLOW_DIR = Path(os.environ.get("LOVEFLOW_DIR", "D:/LoveFlow"))
# 模型目录
MODEL_DIR = Path(os.environ.get("LOVEFLOW_MODEL_DIR", "D:/LoveFlow/loveflow_model"))
# 后端目录
BACKEND_DIR = LOVEFLOW_DIR / "backend"
# 训练脚本路径
TRAIN_SCRIPT = BACKEND_DIR / "run_loveflow_train.py"
# 基础模型路径（Qwen3.5-9B）
BASE_MODEL_PATH = os.environ.get("LOVEFLOW_BASE_MODEL", "D:/LLM/LoveFlow")

# ==================== 服务地址 ====================
# LoveFlow 推理服务器
INFER_HOST = os.environ.get("LOVEFLOW_INFER_HOST", "localhost")
INFER_PORT = int(os.environ.get("LOVEFLOW_INFER_PORT", "8005"))
INFER_BASE_URL = f"http://{INFER_HOST}:{INFER_PORT}"

# MySQL 数据库（LoveFlow 训练数据源）
MYSQL_HOST = os.environ.get("MYSQL_SERVER", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "root")
MYSQL_DB = "loveflow"

# Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

# ==================== Agent 运行时 ====================
# 默认对话参数
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.4
# 上下文预算（token 数）
CONTEXT_BUDGET = 2048
# 检查点保存间隔（交互轮数）
CHECKPOINT_INTERVAL = 5

# ==================== 模块开关 ====================
# 是否启用各模块（默认全部启用）
ENABLE_SYSTEM_MONITOR = True
ENABLE_CAMERA = True
ENABLE_FACE_RECOGNITION = True
ENABLE_NETWORK_DISCOVERY = True
ENABLE_TRAINING = True
ENABLE_API_CONNECTOR = True
ENABLE_WEBHOOK = True
ENABLE_PROTOCOL_ADAPTERS = True
ENABLE_CODE_ENGINE = True
ENABLE_SCHEDULER = True

# ==================== 感知模块配置 ====================
# 系统监控间隔（秒）
MONITOR_INTERVAL = 1800  # 30 分钟
# 摄像头默认 RTSP 超时（秒）
CAMERA_TIMEOUT = 10
# 网络扫描超时（秒）
NETWORK_SCAN_TIMEOUT = 30

# ==================== 训练配置 ====================
# 默认训练参数
TRAIN_DEFAULT_EPOCHS = 5           # GPU 训练可以更多轮次
TRAIN_DEFAULT_LR = 2e-4
TRAIN_DEFAULT_LORA_R = 8
TRAIN_DEFAULT_LORA_ALPHA = 16
TRAIN_DEFAULT_BATCH_SIZE = 2        # GPU 批次大小（6GB 显存推荐 2，12GB+ 可用 4）
TRAIN_DEFAULT_GRADIENT_ACCUM = 2    # 梯度累积步数
TRAIN_DEFAULT_MAX_SEQ_LENGTH = 1024

# ==================== 数据采集配置 ====================
# 每个 URL 最大生成训练对
COLLECT_MAX_PAIRS_PER_URL = 5
# 搜索最大结果数
COLLECT_MAX_SEARCH_RESULTS = 5
# 采集请求超时（秒）
COLLECT_TIMEOUT = 30

# ==================== Webhook 配置 ====================
WEBHOOK_HOST = "0.0.0.0"
WEBHOOK_PORT = 9090
