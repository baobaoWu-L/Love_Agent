# LoveFlow Agent

自研大语言模型的智能驾驶舱——基于 Qwen3.5-9B 底座 + LoRA 微调路线。

## 功能

| 模块 | 能力 | 依赖 |
|------|------|------|
| **记忆系统** | SQLite FTS5 全文搜索、标签分类、持久化记忆 | 标准库 |
| **任务追踪** | 树状任务、进度管理、父子关系 | 标准库 |
| **系统监控** | CPU/内存/磁盘/进程/网络实时监控 | psutil |
| **摄像头** | RTSP 流连接、USB 摄像头抓拍 | opencv-python |
| **人像识别** | 人脸检测、特征提取、跨会话匹配 | face_recognition |
| **网络发现** | ARP 扫描、端口检测、服务识别 | python-nmap |
| **训练编排** | LoRA 微调启动/停止/监控、训练曲线 | pymysql/matplotlib |
| **外部 API** | REST 连接管理、OpenAI 兼容调用 | requests |
| **Webhook** | 事件订阅、HTTP 接收端 | fastapi/uvicorn |
| **协议适配** | MQTT/WebSocket/SSH/数据库直连 | 各协议库 |
| **代码引擎** | 沙箱代码生成与执行、自动保存 | 标准库 |
| **模块热加载** | 插件式扩展、文件变化自动重载 | 标准库 |
| **启动自检** | 自动补全缺失依赖、环境检测 | — |

## 快速开始

```bash
# 交互式对话
python -m agent.main

# 启动训练
python -m agent.main train --epochs 5

# 系统状态
python -m agent.main status

# 安装依赖
python -m agent.main install
```

## 与 LoveFlow 的关系

Agent 通过 HTTP 连接 LoveFlow 推理服务器（默认 `localhost:8005`）进行对话，通过子进程调用 `LoveFlow/backend/run_loveflow_train.py` 进行训练。

配置路径在 `config/agent_config.py`，可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LOVEFLOW_DIR` | `D:/LoveFlow` | LoveFlow 项目路径 |
| `LOVEFLOW_MODEL_DIR` | `D:/LoveFlow/loveflow_model` | 模型产出路径 |
| `LOVEFLOW_BASE_MODEL` | `D:/LLM/LoveFlow` | Qwen3.5 基础模型路径 |
| `LOVEFLOW_INFER_HOST` | `localhost` | 推理服务器地址 |
| `LOVEFLOW_INFER_PORT` | `8005` | 推理服务器端口 |
| `MYSQL_SERVER` | `127.0.0.1` | 数据库地址 |

## 项目结构

```
D:\LoveFlow_Agent\
├── pyproject.toml
├── requirements.txt
├── config/agent_config.py     # 集中配置
├── memory/                     # 持久化数据
├── agent/
│   ├── main.py                 # CLI 入口
│   ├── loveflow_client.py      # 推理客户端
│   ├── core/                   # 核心引擎
│   │   ├── engine.py           # Agent 主循环
│   │   ├── memory.py           # 记忆系统
│   │   ├── context.py          # 上下文构建
│   │   ├── checkpoint.py       # 检查点
│   │   ├── task_tracker.py     # 任务追踪
│   │   ├── scheduler.py        # 调度器
│   │   └── system_prompt.py    # 提示词模板
│   ├── perception/             # 感知模块
│   │   ├── system_monitor.py
│   │   ├── camera.py
│   │   ├── face_recognition.py
│   │   └── network_discovery.py
│   ├── training/               # 训练编排
│   │   ├── orchestrator.py
│   │   ├── data_manager.py
│   │   └── monitor.py
│   ├── integration/            # 外部集成
│   │   ├── api_connector.py
│   │   ├── webhook.py
│   │   └── protocol_adapters.py
│   ├── self_mod/               # 自我进化
│   │   ├── dependency_manager.py
│   │   ├── code_engine.py
│   │   └── module_loader.py
│   └── extensions/             # 扩展插件目录
└── README.md
```
