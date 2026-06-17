"""
动态模块加载器

支持：
  - 扫描 extensions/ 目录自动加载插件
  - 运行时热重载（检测文件变化）
  - 插件注册接口
"""
import importlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Callable, Optional

from config.agent_config import EXTENSIONS_DIR

logger = logging.getLogger("loveflow.module_loader")


class ModuleLoader:
    """动态模块加载器"""

    def __init__(self):
        self._loaded_modules: dict[str, object] = {}
        self._watch_thread: Optional[Thread] = None
        self._watching = False
        self._on_load_callbacks: list[Callable] = []

    def on_load(self, callback: Callable):
        """注册模块加载回调"""
        self._on_load_callbacks.append(callback)

    def load_extensions(self) -> list[str]:
        """
        扫描并加载所有扩展模块

        Returns:
            成功加载的模块名称列表
        """
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
        loaded = []

        for pyfile in sorted(EXTENSIONS_DIR.glob("*.py")):
            if pyfile.name == "__init__.py":
                continue

            mod_name = pyfile.stem
            if mod_name in self._loaded_modules:
                continue

            try:
                module = self._load_single(pyfile)
                if module:
                    self._loaded_modules[mod_name] = module
                    loaded.append(mod_name)
                    logger.info(f"扩展模块已加载: {mod_name}")

                    # 触发回调
                    for cb in self._on_load_callbacks:
                        try:
                            cb(mod_name, module)
                        except Exception as e:
                            logger.warning(f"加载回调失败 [{mod_name}]: {e}")
            except Exception as e:
                logger.error(f"扩展模块加载失败 [{mod_name}]: {e}")

        return loaded

    def reload_all(self) -> list[str]:
        """重新加载所有模块"""
        old_modules = set(self._loaded_modules.keys())
        self._loaded_modules.clear()
        newly_loaded = self.load_extensions()

        # 删除已被移除的模块缓存
        for mod_name in old_modules:
            if mod_name not in newly_loaded:
                full_name = f"agent.extensions.{mod_name}"
                if full_name in sys.modules:
                    del sys.modules[full_name]

        return newly_loaded

    def start_watching(self, interval: int = 5):
        """启动文件变化监听（后台线程）"""
        if self._watching:
            return

        self._watching = True
        self._watch_thread = Thread(target=self._watch_loop, args=(interval,), daemon=True)
        self._watch_thread.start()
        logger.info(f"模块热加载监听已启动（间隔 {interval}s）")

    def stop_watching(self):
        """停止文件监听"""
        self._watching = False

    def _load_single(self, pyfile: Path):
        """加载单个扩展文件"""
        mod_name = f"agent.extensions.{pyfile.stem}"

        # 如果已加载，先清除缓存
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        spec = importlib.util.spec_from_file_location(mod_name, pyfile)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

            # 如果模块有 register 函数，调用它返回实例
            if hasattr(module, "register"):
                return module.register()

            return module

        return None

    def _watch_loop(self, interval: int):
        """监听文件变化循环"""
        last_mtimes = {}

        while self._watching:
            for pyfile in EXTENSIONS_DIR.glob("*.py"):
                if pyfile.name == "__init__.py":
                    continue
                mtime = pyfile.stat().st_mtime
                if pyfile.stem in self._loaded_modules:
                    if last_mtimes.get(pyfile.name, 0) < mtime:
                        logger.info(f"检测到文件变化: {pyfile.name}")
                        try:
                            self.reload_all()
                        except Exception as e:
                            logger.error(f"热重载失败: {e}")
                last_mtimes[pyfile.name] = mtime

            time.sleep(interval)

    def get_loaded_modules(self) -> dict[str, object]:
        """获取已加载的模块"""
        return dict(self._loaded_modules)
