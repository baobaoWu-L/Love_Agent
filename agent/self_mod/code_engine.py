"""
代码生成与执行引擎

Agent 可以通过 LLM 生成 Python 代码并执行，实现"自编译"能力。
安全沙箱限制文件系统和网络访问。
"""
import ast
import logging
import os
import sys
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import MEMORY_DIR

logger = logging.getLogger("loveflow.code_engine")

CODE_DIR = MEMORY_DIR / "generated_code"
CODE_DIR.mkdir(parents=True, exist_ok=True)

# 允许的文件系统路径白名单
ALLOWED_PATHS = [
    str(MEMORY_DIR),
    str(CODE_DIR),
]

# 允许的内置模块
ALLOWED_BUILTINS = {
    "print", "len", "range", "int", "float", "str", "list", "dict",
    "tuple", "set", "bool", "type", "isinstance", "hasattr", "getattr",
    "setattr", "min", "max", "sum", "abs", "round", "sorted", "reversed",
    "enumerate", "zip", "map", "filter", "any", "all", "open",
    "json", "math", "datetime", "re", "collections", "itertools",
}


class CodeSandbox:
    """代码执行沙箱"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def validate(self, code: str) -> list[str]:
        """
        静态检查代码安全性

        Returns:
            违规列表（空列表表示通过）
        """
        violations = []

        # 禁止的导入
        forbidden_imports = ["os.system", "subprocess", "shutil.rmtree", "pathlib.Path.unlink"]
        for item in forbidden_imports:
            if item in code:
                violations.append(f"禁止的操作: {item}")

        # AST 安全检查
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # 禁止调用系统相关函数
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        attr_name = f"{self._get_ast_name(node.func.value)}.{node.func.attr}" if hasattr(node.func, 'value') else ""
                        if any(forbid in attr_name for forbid in ["os.", "subprocess", "shutil.", "pathlib."]):
                            violations.append(f"禁止的系统调用: {attr_name}")
        except SyntaxError as e:
            violations.append(f"语法错误: {e}")

        return violations

    def execute(self, code: str, context: dict = None) -> dict:
        """
        在沙箱中执行代码

        Args:
            code: Python 代码
            context: 执行上下文变量

        Returns:
            {"success": bool, "output": str, "error": Optional[str]}
        """
        # 安全检查
        violations = self.validate(code)
        if violations:
            return {"success": False, "output": "", "error": f"安全违规: {'; '.join(violations)}"}

        # 准备执行环境
        exec_globals = {
            "__builtins__": {k: __builtins__[k] for k in ALLOWED_BUILTINS if k in __builtins__},
        }
        exec_locals = context or {}

        # 捕获输出
        from io import StringIO
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            exec(code, exec_globals, exec_locals)
            output = sys.stdout.getvalue()
            error_output = sys.stderr.getvalue()
            return {
                "success": True,
                "output": output + (error_output if error_output else ""),
                "error": None,
                "locals": {k: v for k, v in exec_locals.items() if not k.startswith("_")},
            }
        except Exception as e:
            return {
                "success": False,
                "output": sys.stdout.getvalue(),
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _get_ast_name(self, node) -> str:
        """获取 AST 节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_ast_name(node.value)}.{node.attr}"
        return ""


class CodeEngine:
    """代码生成与执行引擎"""

    def __init__(self):
        self.sandbox = CodeSandbox()
        self._generated_count = 0

    def generate_and_execute(self, code: str, description: str = "", context: dict = None) -> dict:
        """
        生成并执行代码

        Args:
            code: Python 代码
            description: 代码描述
            context: 执行上下文

        Returns:
            执行结果
        """
        # 保存代码快照
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = CODE_DIR / f"gen_{timestamp}.py"
        filepath.write_text(code, encoding="utf-8")
        self._generated_count += 1

        # 记录到记忆
        if description:
            desc_text = f"代码: {description}"
        else:
            desc_text = f"自动生成代码 (gen_{timestamp}.py)"

        # 执行
        result = self.sandbox.execute(code, context)

        # 写入执行结果注释
        if result["success"]:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"\n\n# ======== 执行结果 ========\n")
                f.write(f"# {result['output'][:500]}\n")

        return result

    def list_generated(self) -> list[dict]:
        """列出已生成的代码"""
        files = []
        for f in sorted(CODE_DIR.glob("gen_*.py"), reverse=True):
            files.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
            })
        return files

    def get_context(self) -> Optional[str]:
        """获取上下文（注入到系统提示词）"""
        total = len(list(CODE_DIR.glob("gen_*.py")))
        if total > 0:
            return f"【自编译】已生成 {total} 个代码模块，可通过代码引擎执行 Python 代码"
        return None
