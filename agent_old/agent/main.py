"""
LoveFlow Agent CLI 主入口

启动流程：
  1. 解析命令行参数
  2. 自检依赖 → 自动补全缺失核心依赖
  3. 加载配置
  4. 检查 LoveFlow 服务状态
  5. 初始化各模块（惰性加载）
  6. 输出状态报告
  7. 执行子命令（chat/train/status/daemon/install）

子命令：
  chat     交互式对话（默认）
  train    训练 LoveFlow 模型
  status   系统状态报告
  daemon   后台守护模式
  install  安装依赖
"""
import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("loveflow.main")


def check_environment(verbose: bool = True) -> dict:
    """
    启动自检——检查依赖、关键路径、服务状态

    Returns:
        环境状态报告
    """
    from agent.self_mod.dependency_manager import check_dependencies, auto_install_core, get_environment_report

    report = {"deps": {}, "env": {}, "services": {}}

    # 依赖检查
    deps_status = check_dependencies()
    report["deps"] = deps_status

    if verbose:
        core_ok = len(deps_status["core"]["missing"]) == 0
        print(f"  核心依赖: {'✓' if core_ok else '✗'} {len(deps_status['core']['installed'])} installed / {len(deps_status['core']['missing'])} missing")
        print(f"  可选依赖: {len(deps_status['optional']['installed'])} installed / {len(deps_status['optional']['missing'])} optional")

    # 自动安装缺失核心依赖
    if deps_status["core"]["missing"]:
        print(f"  → 正在安装缺失的核心依赖: {', '.join(deps_status['core']['missing'])}")
        installed = auto_install_core()
        if installed:
            print(f"  → 已安装: {', '.join(installed)}")
        remaining = [p for p in deps_status["core"]["missing"] if p not in installed]
        if remaining:
            print(f"  ⚠ 以下核心依赖安装失败: {', '.join(remaining)}")

    # 环境检测
    env_report = get_environment_report()
    report["env"] = env_report

    if verbose:
        print(f"  Python: {env_report.get('python_version', 'N/A').split()[0]}")
        print(f"  CUDA: {'✓' if env_report.get('cuda_available') else '✗'}")
        print(f"  LoveFlow 推理服务: {'✓ 在线' if env_report.get('loveflow_infer_online') else '✗ 离线'}")
        print(f"  LoveFlow 项目目录: {'✓' if env_report.get('loveflow_dir_exists') else '✗ 未找到'}")
        print(f"  基础模型: {'✓' if env_report.get('base_model_exists') else '✗ 未找到'}")

    return report


def init_agent(report: dict = None) -> object:
    """
    初始化 Agent 核心引擎并加载所有模块

    Returns:
        LoveFlowAgent 实例
    """
    from agent.core.engine import LoveFlowAgent
    from agent.core.scheduler import Scheduler

    agent = LoveFlowAgent()

    # 初始化调度器
    scheduler = Scheduler()
    agent.register_module("scheduler", scheduler)

    # 惰性加载各模块（仅在可用时注册）
    _try_load_perception(agent)
    _try_load_training(agent)
    _try_load_integration(agent)
    _try_load_selfmod(agent)

    # 启动调度器
    scheduler.start()

    return agent


def _try_load_perception(agent):
    """加载感知模块"""
    try:
        from agent.perception.system_monitor import SystemMonitor
        monitor = SystemMonitor()
        agent.register_module("system_monitor", monitor, ["系统监控（CPU/内存/磁盘/进程）"])

        # 注册动作处理器
        from agent.core.engine import BaseActionHandler
        import re
        from typing import Optional

        class SystemStatusHandler(BaseActionHandler):
            tag = "STATUS"
            @classmethod
            def parse(cls, match):
                return {"_tag": "STATUS"}
            @classmethod
            def execute(cls, action):
                try:
                    mon = agent.get_module("system_monitor")
                    if mon:
                        return mon.get_summary()
                    return "系统监控模块未加载"
                except Exception as e:
                    return f"获取状态失败: {e}"

        agent.register_action("STATUS", SystemStatusHandler)
        logger.info("感知模块已加载: SystemMonitor")
    except Exception as e:
        logger.warning(f"感知模块加载失败: {e}")

    try:
        from agent.perception.camera import CameraManager
        cam = CameraManager()
        agent.register_module("camera", cam, ["摄像头连接（RTSP/USB 拍照）"])
        logger.info("感知模块已加载: CameraManager")
    except Exception as e:
        logger.debug(f"摄像头模块跳过: {e}")

    try:
        from agent.perception.face_recognition import FaceRecognizer
        fr = FaceRecognizer()
        agent.register_module("face_recognizer", fr, ["人像识别（检测/匹配/注册）"])
        logger.info("感知模块已加载: FaceRecognizer")
    except Exception as e:
        logger.debug(f"人像识别模块跳过: {e}")

    try:
        from agent.perception.network_discovery import NetworkScanner
        ns = NetworkScanner()
        agent.register_module("network_scanner", ns, ["网络发现（ARP 扫描/端口检测）"])
        logger.info("感知模块已加载: NetworkScanner")
    except Exception as e:
        logger.debug(f"网络发现模块跳过: {e}")


def _try_load_training(agent):
    """加载训练编排模块"""
    try:
        from agent.training.orchestrator import TrainingOrchestrator
        trainer = TrainingOrchestrator()
        agent.register_module("trainer", trainer, ["训练编排（启动训练/查看进度）"])

        # 注册训练动作
        from agent.core.engine import BaseActionHandler
        import re

        class TrainHandler(BaseActionHandler):
            tag = "TRAIN"
            @classmethod
            def parse(cls, match):
                content = match.group(1).strip()
                return {"_tag": "TRAIN", "content": content}
            @classmethod
            def execute(cls, action):
                try:
                    t = agent.get_module("trainer")
                    if t:
                        # 支持格式: [TRAIN: epochs=5 lr=1e-4]
                        params = {}
                        for part in action["content"].split():
                            if "=" in part:
                                k, v = part.split("=", 1)
                                try:
                                    params[k] = int(v) if v.isdigit() else float(v)
                                except:
                                    params[k] = v
                        result = t.start_training(**params)
                        if result["success"]:
                            return f"训练已启动 (任务: {result.get('task_id', 'N/A')})"
                        return f"训练启动失败: {result}"
                    return "训练模块未加载"
                except Exception as e:
                    return f"训练启动失败: {e}"

        agent.register_action("TRAIN", TrainHandler)
        logger.info("训练模块已加载: TrainingOrchestrator")
    except Exception as e:
        logger.warning(f"训练模块加载失败: {e}")

    try:
        from agent.training.data_manager import TrainingDataManager
        dm = TrainingDataManager()
        agent.register_module("data_manager", dm)
        logger.info("训练模块已加载: TrainingDataManager")
    except Exception as e:
        logger.debug(f"训练数据管理模块跳过: {e}")

    try:
        from agent.training.monitor import TrainingMonitor
        tm = TrainingMonitor()
        agent.register_module("training_monitor", tm)
        logger.info("训练模块已加载: TrainingMonitor")
    except Exception as e:
        logger.debug(f"训练监控模块跳过: {e}")

    # 加载联网数据采集器
    try:
        from agent.training.data_collector import DataCollector
        dc = DataCollector()
        agent.register_module("data_collector", dc, ["联网数据采集（爬取网页/搜索→训练数据）"])
        logger.info("训练模块已加载: DataCollector")
    except Exception as e:
        logger.debug(f"数据采集器跳过: {e}")


def _try_load_integration(agent):
    """加载集成模块"""
    try:
        from agent.integration.api_connector import ApiConnector
        api = ApiConnector()
        agent.register_module("api_connector", api, ["API 连接（外部 REST 服务）"])
        logger.info("集成模块已加载: ApiConnector")
    except Exception as e:
        logger.debug(f"API 连接器跳过: {e}")

    try:
        from agent.integration.webhook import WebhookManager
        wh = WebhookManager()
        agent.register_module("webhook", wh)
        logger.info("集成模块已加载: WebhookManager")
    except Exception as e:
        logger.debug(f"Webhook 模块跳过: {e}")

    try:
        from agent.integration.protocol_adapters import ProtocolAdapters
        pa = ProtocolAdapters()
        agent.register_module("protocol_adapters", pa, ["协议适配（MQTT/WebSocket/SSH/DB）"])
        logger.info("集成模块已加载: ProtocolAdapters")
    except Exception as e:
        logger.debug(f"协议适配器模块跳过: {e}")


def _try_load_selfmod(agent):
    """加载自我进化模块"""
    try:
        from agent.self_mod.code_engine import CodeEngine
        ce = CodeEngine()
        agent.register_module("code_engine", ce, ["代码引擎（生成并执行 Python 代码）"])
        logger.info("自我进化模块已加载: CodeEngine")
    except Exception as e:
        logger.debug(f"代码引擎跳过: {e}")

    try:
        from agent.self_mod.module_loader import ModuleLoader
        ml = ModuleLoader()
        ml.load_extensions()
        agent.register_module("module_loader", ml)
        logger.info("自我进化模块已加载: ModuleLoader")
    except Exception as e:
        logger.debug(f"模块加载器跳过: {e}")


# ==================== CLI 子命令 ====================


def cmd_chat(args):
    """交互式对话模式（默认）"""
    print("=" * 56)
    print("  LoveFlow Agent v0.2.0 — 自研大语言模型智能驾驶舱")
    print(" 输入 /help 查看命令，/exit 退出")
    print("=" * 56)

    # 启动自检
    print("\n🔍 系统自检...")
    report = check_environment(verbose=True)
    print()

    # 初始化 Agent
    agent = init_agent(report)

    # 检查推理服务器
    if not agent.client.health():
        print("⚠️  LoveFlow 推理服务器未启动！对话功能不可用。")
        print("   请先启动: python D:\\LoveFlow\\backend\\run_loveflow.py")
        print()

    # 注册模块命令
    modules_info = []
    for name, instance in agent._modules.items():
        modules_info.append(name)
    if modules_info:
        print(f"📦 已加载模块: {', '.join(modules_info)}")
    print()

    # REPL 主循环
    while True:
        try:
            user_input = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 处理内置命令
        if user_input == "/exit":
            print("👋 再见！")
            break
        elif user_input == "/help":
            _show_help()
            continue
        elif user_input == "/memory":
            _show_memory()
            continue
        elif user_input == "/tasks":
            _show_tasks()
            continue
        elif user_input == "/status":
            _show_status(agent)
            continue
        elif user_input == "/train":
            _cmd_train_in_repl(agent)
            continue
        elif user_input == "/modules":
            _show_modules(agent)
            continue
        elif user_input.startswith("/collect"):
            _cmd_collect(agent, user_input)
            continue
        elif user_input == "/save":
            _save_checkpoint()
            continue
        elif user_input.startswith("/"):
            print(f"未知命令: {user_input}  输入 /help 查看帮助")
            continue

        # 正常对话
        response = agent.interact(user_input, verbose=False)
        print(f"\n🤖 LoveFlow: {response}\n")


def _show_help():
    """显示帮助"""
    print("""命令:
  /exit      退出
  /help      帮助
  /memory    查看记忆列表
  /tasks     查看任务列表
  /status    系统状态报告
  /train     启动训练
  /collect   采集数据并训练（见下方示例）
  /modules   查看已加载模块
  /save      强制保存检查点

采集示例:
  /collect url https://example.com             采集单个网页
  /collect search 人工智能最新进展               搜索并采集
  /collect train https://example.com            采集后自动训练

对话技巧:
  - 说"检查系统状态"查看 CPU/内存/磁盘
  - 说"帮我记住..."自动保存到记忆
  - 说"创建一个任务..."自动创建任务
  - 说"搜索关于xxx的资料来训练"自动采集+训练
  - 说"写一个 Python 脚本..."生成并执行代码""")


def _show_memory():
    """查看记忆列表"""
    from agent.core.memory import list_memories
    mems = list_memories("knowledge", limit=20)
    print(f"\n📚 记忆 ({len(mems)} 条):")
    for m in mems:
        tags_str = f" [{', '.join(m['tags'])}]" if m.get('tags') else ""
        print(f"  [{m['importance']}]{tags_str} {m['title'][:60]}")
    if not mems:
        print("  （暂无记忆）")


def _show_tasks():
    """查看任务列表"""
    from agent.core.task_tracker import list_tasks, get_task_tree
    tasks = list_tasks()
    print(f"\n📋 任务 ({len(tasks)} 条):")
    for t in tasks:
        icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅", "failed": "❌"}.get(t["status"], "⬜")
        print(f"  {icon} {t['id']}: {t['description'][:60]} ({t['status']})")
    if not tasks:
        print("  （暂无任务）")


def _show_status(agent):
    """显示系统状态"""
    print("\n📊 系统状态报告:")
    print("=" * 40)

    # 系统状态
    monitor = agent.get_module("system_monitor")
    if monitor:
        print(monitor.get_summary())
    else:
        print("  系统监控模块未加载（需要 psutil）")

    # 训练状态
    trainer = agent.get_module("trainer")
    if trainer:
        status = trainer.get_status()
        if status.get("running"):
            print(f"  训练状态: 🔄 运行中 (PID: {status.get('pid', 'N/A')})")
        else:
            print(f"  训练状态: ⏸ 空闲")

    # 推理服务
    online = agent.client.health()
    print(f"  LoveFlow 推理: {'✓ 在线' if online else '✗ 离线'}")

    print("=" * 40)


def _show_modules(agent):
    """查看已加载模块"""
    print(f"\n📦 已加载模块 ({len(agent._modules)} 个):")
    for name, instance in agent._modules.items():
        print(f"  • {name}: {type(instance).__name__}")
    print(f"  注册动作: {list(agent.action_registry._handlers.keys())}")


def _cmd_train_in_repl(agent):
    """REPL 中启动训练"""
    trainer = agent.get_module("trainer")
    if not trainer:
        print("❌ 训练模块未加载")
        return
    print("🚀 正在启动训练（默认: 3 epochs, lr=2e-4, lora_r=8）...")
    result = trainer.start_training(epochs=3)
    if result["success"]:
        print(f"✅ 训练已启动！任务 ID: {result.get('task_id', 'N/A')}")
        print(f"   进程 PID: {result.get('process_id', 'N/A')}")
        print(f"   使用 /status 查看进度")
    else:
        print(f"❌ 训练启动失败: {result}")


def _save_checkpoint():
    """强制保存检查点"""
    from agent.core.checkpoint import save_checkpoint
    save_checkpoint(session_summary=f"手动保存 @ {datetime.now().isoformat()}")
    print("✅ 检查点已保存")


def _cmd_collect(agent, user_input: str):
    """REPL 中的数据采集命令"""
    from agent.training.data_collector import DataCollector

    collector = agent.get_module("data_collector")
    if not collector:
        print("❌ 数据采集器未加载（需要 bs4 库）")
        return

    parts = user_input.split()
    if len(parts) < 3:
        print("用法:")
        print("  /collect url <URL>            采集单个网页")
        print("  /collect search <关键词>       搜索并采集")
        print("  /collect train <URL>           采集后自动训练")
        return

    mode = parts[1]
    target = " ".join(parts[2:])
    auto_train = mode == "train"

    print(f"🌐 正在采集: {target}")
    print("   这可能需要一些时间...\n")

    if mode in ("url", "train"):
        result = collector.collect_from_urls(
            [target],
            auto_train=auto_train,
            max_pairs_per_url=5,
        )
    elif mode == "search":
        result = collector.collect_from_search(
            target,
            max_results=5,
            auto_train=False,
        )
    else:
        print(f"未知采集模式: {mode}")
        return

    print(f"\n✅ 采集完成:")
    print(f"   爬取: {result.get('fetched', 0)} 个页面")
    print(f"   生成: {result.get('generated', 0)} 个训练对")
    print(f"   入库: {result.get('stored', 0)} 条")
    if auto_train and result.get("train_result"):
        tr = result["train_result"]
        if tr.get("success"):
            print(f"   🚀 训练已自动启动 (PID: {tr.get('process_id', 'N/A')})")
        else:
            print(f"   ⚠️ 自动训练启动失败: {tr.get('error', '未知错误')}")


# ==================== 其他子命令 ====================


def cmd_train(args):
    """训练模式"""
    print("🚀 LoveFlow Agent 训练模式")
    print("=" * 40)
    check_environment(verbose=True)
    agent = init_agent()

    trainer = agent.get_module("trainer")
    if not trainer:
        print("❌ 训练模块未加载")
        return

    epochs = args.epochs or 3
    lr = args.lr or 2e-4
    lora_r = args.lora_r or 8

    print(f"参数: epochs={epochs}, lr={lr}, lora_r={lora_r}")
    result = trainer.start_training(epochs=epochs, lr=lr, lora_r=lora_r)

    if result["success"]:
        print(f"✅ 训练已启动！任务: {result.get('task_id', 'N/A')}")
        print("按 Ctrl+C 停止训练\n")
        try:
            while True:
                time.sleep(5)
                status = trainer.get_status()
                if not status.get("running"):
                    print("✅ 训练已完成")
                    break
                print(f"  ⏳ 训练进行中 (PID: {status.get('pid', 'N/A')})")
        except KeyboardInterrupt:
            print("\n⏹ 正在停止训练...")
            trainer.stop_training()
            print("训练已停止")
    else:
        print(f"❌ 训练启动失败: {result}")


def cmd_status(args):
    """状态报告模式"""
    print("📊 LoveFlow Agent 系统状态")
    print("=" * 40)
    report = check_environment(verbose=True)
    print()
    print("📦 已注册模块:")
    agent = init_agent(report)
    for name, instance in agent._modules.items():
        print(f"  • {name}")
    print()
    print("💡 使用 'python -m agent.main chat' 进入交互模式")


def cmd_install(args):
    """安装依赖"""
    print("📦 安装依赖...")
    from agent.self_mod.dependency_manager import check_dependencies, auto_install_core

    status = check_dependencies()
    if status["core"]["missing"]:
        print(f"需要安装核心依赖: {', '.join(status['core']['missing'])}")
        installed = auto_install_core()
        if installed:
            print(f"✅ 已安装: {', '.join(installed)}")
    else:
        print("✅ 所有核心依赖已就绪")

    optional_missing = list(status["optional"]["missing"].keys())
    if optional_missing:
        print(f"可选依赖未安装 ({len(optional_missing)} 个): {', '.join(optional_missing[:10])}")
        print("  需要时自动安装")

    env = __import__('agent.self_mod.dependency_manager', fromlist=['']).get_environment_report()
    print(f"\n环境: Python {env.get('python_version', '').split()[0]}")
    print(f"CUDA: {'可用' if env.get('cuda_available') else '不可用'}")
    print(f"LoveFlow 推理: {'在线' if env.get('loveflow_infer_online') else '离线'}")


def cmd_collect(args):
    """采集模式：从互联网采集数据并训练"""
    print("🌐 LoveFlow Agent 数据采集模式")
    print("=" * 40)
    check_environment(verbose=True)
    agent = init_agent()

    collector = agent.get_module("data_collector")
    if not collector:
        print("❌ 数据采集器未加载（需要 beautifulsoup4 库）")
        print("   请安装: pip install beautifulsoup4 lxml")
        return

    auto_train = args.auto_train

    if args.urls:
        urls = args.urls.split(",")
        print(f"目标: {len(urls)} 个 URL")
        result = collector.collect_from_urls(urls, auto_train=auto_train)
    elif args.search:
        print(f"搜索关键词: {args.search}")
        result = collector.collect_from_search(args.search, auto_train=auto_train)
    else:
        print("请指定 --url 或 --search")
        return

    print(f"\n✅ 采集完成:")
    print(f"   爬取: {result.get('fetched', 0)} 个页面")
    print(f"   生成: {result.get('generated', 0)} 个训练对")
    print(f"   入库: {result.get('stored', 0)} 条")
    if auto_train and result.get("train_result"):
        tr = result["train_result"]
        print(f"   训练: {'✅ 已启动' if tr.get('success') else '❌ ' + tr.get('error', '')}")


# ==================== 入口 ====================


def cli():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="LoveFlow Agent — 自研大语言模型智能驾驶舱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  chat             交互式对话（默认）
  train            训练 LoveFlow 模型（GPU 加速）
  collect          从互联网采集数据并训练
  status           系统状态报告
  daemon           后台守护模式
  install          安装依赖

示例:
  python -m agent.main
  python -m agent.main train --epochs 5
  python -m agent.main collect --urls https://example.com --auto-train
  python -m agent.main collect --search "人工智能" --auto-train
  python -m agent.main status
        """,
    )
    parser.add_argument("command", nargs="?", default="chat", help="子命令")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--lr", type=float, help="学习率")
    parser.add_argument("--lora-r", type=int, help="LoRA 秩")
    parser.add_argument("--urls", type=str, help="采集目标 URL（逗号分隔）")
    parser.add_argument("--search", type=str, help="搜索关键词")
    parser.add_argument("--auto-train", action="store_true", help="采集后自动训练")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    cmd_map = {
        "chat": cmd_chat,
        "train": cmd_train,
        "status": cmd_status,
        "daemon": cmd_chat,  # daemon 模式暂等同于 chat
        "install": cmd_install,
        "collect": cmd_collect,
    }

    cmd_func = cmd_map.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    cli()
