"""
训练数据管理器

管理 LoveFlow 训练数据，通过 MySQL 数据库进行 CRUD 操作：
  - 查询训练数据统计（总数、来源分布、平均长度）
  - 获取样本数据
  - 添加训练记录
  - 获取数据集列表
  - 连接测试
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
)

logger = logging.getLogger("loveflow.TrainingDataManager")


class TrainingDataManager:
    """训练数据管理器：操作 training_data 表"""

    def __init__(self):
        # MySQL 连接参数
        self.host: str = MYSQL_HOST
        self.port: int = MYSQL_PORT
        self.user: str = MYSQL_USER
        self.password: str = MYSQL_PASSWORD
        self.database: str = MYSQL_DB
        # pymysql 连接实例（懒加载，每次操作时创建新连接）
        self._connection = None

    def get_stats(self) -> dict:
        """
        查询 training_data 表统计信息

        返回：
            {"total": int, "source_distribution": dict, "status_distribution": dict,
             "avg_prompt_length": float, "avg_completion_length": float,
             "avg_prompt_words": float, "avg_completion_words": float}
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM training_data")
            total = cursor.fetchone()[0]

            # 来源分布
            cursor.execute(
                "SELECT source, COUNT(*) as cnt FROM training_data GROUP BY source ORDER BY cnt DESC"
            )
            source_dist = {row[0]: row[1] for row in cursor.fetchall()}

            # 状态分布
            cursor.execute(
                "SELECT status, COUNT(*) as cnt FROM training_data GROUP BY status ORDER BY status"
            )
            status_dist = {str(row[0]): row[1] for row in cursor.fetchall()}

            # 平均 prompt 和 completion 长度（字符数）
            cursor.execute(
                "SELECT AVG(LENGTH(prompt)), AVG(LENGTH(completion)) FROM training_data"
            )
            row = cursor.fetchone()
            avg_prompt_len = round(row[0], 1) if row and row[0] else 0.0
            avg_completion_len = round(row[1], 1) if row and row[1] else 0.0

            # 平均词数（按空格分词粗略估算）
            cursor.execute(
                "SELECT AVG(LENGTH(prompt) - LENGTH(REPLACE(prompt, ' ', '')) + 1), "
                "AVG(LENGTH(completion) - LENGTH(REPLACE(completion, ' ', '')) + 1) "
                "FROM training_data"
            )
            row = cursor.fetchone()
            avg_prompt_words = round(row[0], 1) if row and row[0] else 0.0
            avg_completion_words = round(row[1], 1) if row and row[1] else 0.0

            cursor.close()
            conn.close()

            return {
                "total": total,
                "source_distribution": source_dist,
                "status_distribution": status_dist,
                "avg_prompt_length": avg_prompt_len,
                "avg_completion_length": avg_completion_len,
                "avg_prompt_words": avg_prompt_words,
                "avg_completion_words": avg_completion_words,
            }

        except Exception as e:
            logger.error(f"获取训练数据统计失败: {e}")
            return {
                "total": 0,
                "source_distribution": {},
                "status_distribution": {},
                "avg_prompt_length": 0.0,
                "avg_completion_length": 0.0,
                "avg_prompt_words": 0.0,
                "avg_completion_words": 0.0,
                "error": str(e),
            }

    def get_samples(self, limit: int = 5, source: Optional[str] = None) -> list[dict]:
        """
        获取样本训练记录

        参数：
            limit: 返回记录数上限
            source: 按来源过滤（None 表示不过滤）

        返回：
            [{"id": int, "prompt": str, "completion": str,
              "source": str, "dataset_name": str, "status": int,
              "created_at": str}, ...]
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if source:
                cursor.execute(
                    "SELECT id, prompt, completion, source, dataset_name, status, created_at "
                    "FROM training_data WHERE source = %s ORDER BY id DESC LIMIT %s",
                    (source, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, prompt, completion, source, dataset_name, status, created_at "
                    "FROM training_data ORDER BY id DESC LIMIT %s",
                    (limit,),
                )

            samples = []
            for row in cursor.fetchall():
                samples.append({
                    "id": row[0],
                    "prompt": row[1],
                    "completion": row[2],
                    "source": row[3],
                    "dataset_name": row[4],
                    "status": row[5],
                    "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
                })

            cursor.close()
            conn.close()
            return samples

        except Exception as e:
            logger.error(f"获取训练样本失败: {e}")
            return []

    def add_data(
        self,
        prompt: str,
        completion: str,
        source: str = "agent",
        dataset_name: Optional[str] = None,
    ) -> bool:
        """
        添加一条训练记录

        参数：
            prompt: 用户输入（提示词）
            completion: 期望输出（完成）
            source: 数据来源
            dataset_name: 数据集名称（可选）

        返回：
            是否添加成功
        """
        if not prompt or not completion:
            logger.warning("添加训练数据失败：prompt 和 completion 不能为空")
            return False

        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO training_data (prompt, completion, source, dataset_name, status) "
                "VALUES (%s, %s, %s, %s, 1)",
                (prompt, completion, source, dataset_name),
            )
            conn.commit()

            inserted_id = cursor.lastrowid
            cursor.close()
            conn.close()

            logger.info(f"训练数据添加成功 (ID: {inserted_id}, source: {source})")
            return True

        except Exception as e:
            logger.error(f"添加训练数据失败: {e}")
            return False

    def get_dataset_names(self) -> list[str]:
        """
        获取所有不同的数据集名称

        返回：
            [str, ...] 数据集名称列表
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT DISTINCT dataset_name FROM training_data WHERE dataset_name IS NOT NULL AND dataset_name != ''"
            )
            names = [row[0] for row in cursor.fetchall()]

            cursor.close()
            conn.close()
            return names

        except Exception as e:
            logger.error(f"获取数据集名称列表失败: {e}")
            return []

    def test_connection(self) -> bool:
        """
        测试 MySQL 连接是否正常

        返回：
            连接是否成功
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            success = result is not None and result[0] == 1
            if success:
                logger.info("MySQL 连接测试成功")
            else:
                logger.warning("MySQL 连接测试返回异常")
            return success

        except Exception as e:
            logger.error(f"MySQL 连接测试失败: {e}")
            return False

    def get_context(self) -> Optional[str]:
        """
        获取训练数据统计信息，供 Agent 上下文注入

        返回格式化的数据统计摘要，或 None（查询失败时）。
        """
        stats = self.get_stats()

        if "error" in stats and stats["total"] == 0:
            logger.warning("获取训练数据上下文失败")
            return None

        # 格式化来源分布
        source_lines = "\n".join(
            f"    - {k}: {v} 条" for k, v in stats["source_distribution"].items()
        )

        return (
            f"【训练数据统计】\n"
            f"  总记录数: {stats['total']} 条\n"
            f"  来源分布:\n{source_lines if source_lines else '    （无数据）'}\n"
            f"  平均 prompt 长度: {stats['avg_prompt_length']} 字符 / {stats['avg_prompt_words']} 词\n"
            f"  平均 completion 长度: {stats['avg_completion_length']} 字符 / {stats['avg_completion_words']} 词\n"
        )

    def _get_conn(self):
        """
        创建并返回一个新的 pymysql 连接

        返回：
            pymysql.Connection 对象
        """
        import pymysql

        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.Cursor,
        )
