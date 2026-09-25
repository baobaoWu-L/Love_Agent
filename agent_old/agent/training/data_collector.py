"""
联网数据采集器

自动从互联网采集数据并转换为训练数据（prompt/completion 对）：
  1. 爬取网页内容
  2. 搜索并采集
  3. 用 LLM 将原始文本转化为 Q&A 训练对
  4. 存入 MySQL training_data 表

工作流程：
  采集 URL/关键词 → 下载+解析 → 文本分块 → LLM 生成 Q&A → 存入 MySQL → 可触发训练
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pymysql
import requests
from bs4 import BeautifulSoup

from config.agent_config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB,
    LOVEFLOW_DIR,
)
from agent.loveflow_client import LoveFlowClient
from agent.core.task_tracker import create_task, update_task

logger = logging.getLogger("loveflow.DataCollector")

# ==================== robots.txt 合规检查器 ====================

BOT_NAME = "LoveFlowAgent/1.0"
BOT_EMAIL = "loveflow-agent@lovebreaker.dev"


class RobotChecker:
    """
    robots.txt 合规检查器

    根据 robots.txt 判断是否允许爬取指定 URL，并遵循 Crawl-delay 指令。
    结果按域名缓存，避免重复请求。
    """

    def __init__(self, user_agent: str = BOT_NAME):
        self.user_agent = user_agent
        # 域名 -> RobotFileParser 缓存
        self._cache: dict[str, RobotFileParser] = {}
        # 域名 -> 上次请求时间（用于 Crawl-delay）
        self._last_access: dict[str, float] = {}
        # 域名 -> Crawl-delay 秒数
        self._crawl_delays: dict[str, float] = {}
        # 统计
        self._stats = {"checked": 0, "allowed": 0, "blocked": 0, "errors": 0}

    def can_fetch(self, url: str) -> bool:
        """
        检查指定 URL 是否允许爬取

        流程：
          1. 解析域名
          2. 获取该域名的 robots.txt 解析器（缓存）
          3. 调用 can_fetch() 判断

        Args:
            url: 目标 URL

        Returns:
            True=允许爬取，False=禁止爬取
        """
        self._stats["checked"] += 1
        try:
            parsed = urlparse(url)
            if not parsed.hostname:
                self._stats["errors"] += 1
                return False

            # 获取该域名的 robot parser
            parser = self._get_parser(parsed.scheme, parsed.hostname)
            if parser is None:
                # 无法获取 robots.txt（如网络错误），根据 RFC 9309 默认允许
                return True

            allowed = parser.can_fetch(self.user_agent, url)
            if allowed:
                self._stats["allowed"] += 1
            else:
                self._stats["blocked"] += 1
                logger.info(f"⛔ robots.txt 禁止爬取: {url}")

            return allowed

        except Exception as e:
            logger.debug(f"robots.txt 检查异常 [{url}]: {e}")
            self._stats["errors"] += 1
            return True  # 异常时默认允许

    def wait_if_needed(self, url: str):
        """
        根据 Crawl-delay 等待必要的时间

        在两次请求之间插入延迟，遵守网站的爬取频率限制。
        """
        parsed = urlparse(url)
        if not parsed.hostname:
            return

        delay = self._crawl_delays.get(parsed.hostname, 0)
        if delay <= 0:
            return

        last = self._last_access.get(parsed.hostname, 0)
        elapsed = time.time() - last
        if elapsed < delay:
            wait = delay - elapsed
            logger.debug(f"  等待 {wait:.1f}s（Crawl-delay: {delay}s）")
            time.sleep(wait)

        self._last_access[parsed.hostname] = time.time()

    def _get_parser(self, scheme: str, hostname: str) -> Optional[RobotFileParser]:
        """获取域名的 RobotFileParser（带缓存）"""
        cache_key = f"{scheme}://{hostname}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        robots_url = f"{scheme}://{hostname}/robots.txt"
        parser = RobotFileParser(robots_url)
        try:
            parser.read()
            self._cache[cache_key] = parser

            # 解析 Crawl-delay
            self._parse_crawl_delay(parser, hostname)

            # 记录规则摘要
            if parser.disallow_all:
                logger.info(f"  robots.txt: {robots_url} → 禁止所有爬虫")
            elif parser.allow_all:
                logger.debug(f"  robots.txt: {robots_url} → 允许所有爬虫")
            else:
                logger.info(f"  robots.txt: {robots_url} → 部分限制，已加载")

            return parser

        except Exception as e:
            logger.debug(f"  robots.txt 读取失败 [{robots_url}]: {e}")
            self._cache[cache_key] = None  # 缓存失败结果，避免重复请求
            return None

    def _parse_crawl_delay(self, parser: RobotFileParser, hostname: str):
        """
        从 RobotFileParser 中提取 Crawl-delay

        RobotFileParser 不直接暴露 Crawl-delay，需要从原始文件中解析。
        """
        try:
            # 解析原始 robots.txt 内容中的 Crawl-delay 指令
            if hasattr(parser, '_entry') or hasattr(parser, 'entries'):
                entries = getattr(parser, 'entries', []) or getattr(parser, '_entry', [])
                if not isinstance(entries, list):
                    entries = [entries]
                for entry in entries:
                    # 匹配 User-agent
                    if hasattr(entry, 'rules'):
                        for rule_line in str(dir(entry)).split(','):
                            pass

            # 通过原始文本直接解析 Crawl-delay
            import urllib.request
            try:
                resp = urllib.request.urlopen(f"https://{hostname}/robots.txt", timeout=10)
                content = resp.read().decode("utf-8", errors="ignore")
                for line in content.splitlines():
                    line = line.strip()
                    if line.lower().startswith("crawl-delay"):
                        delay_str = line.split(":", 1)[1].strip()
                        try:
                            delay = float(delay_str)
                            self._crawl_delays[hostname] = delay
                            if delay > 0:
                                logger.debug(f"  Crawl-delay: {delay}s（{hostname}）")
                        except ValueError:
                            pass
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"  Crawl-delay 解析失败: {e}")

    def get_stats(self) -> dict:
        """获取检查统计"""
        return dict(self._stats)

    def reset(self):
        """清空缓存和统计"""
        self._cache.clear()
        self._last_access.clear()
        self._crawl_delays.clear()
        self._stats = {"checked": 0, "allowed": 0, "blocked": 0, "errors": 0}


class DataCollector:
    """
    联网数据采集器

    支持：
      - 单个 URL 爬取
      - 批量 URL 爬取
      - 关键词搜索 + 内容采集
      - 文本 → Q&A 训练对（用 LoveFlow LLM 生成）

    规范：
      - ⚠️ 所有爬取前先检查 robots.txt，禁止爬取被拒绝的 URL
      - ⚠️ 遵循 Crawl-delay 指令，控制请求频率
      - ⚠️ 使用明确的 User-Agent 标识自己
    """

    def __init__(self):
        self.llm = LoveFlowClient()
        self.robot_checker = RobotChecker()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": BOT_NAME,
            "From": BOT_EMAIL,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        # 采集统计
        self._stats = {"fetched": 0, "generated": 0, "stored": 0}

    # ==================== 网页抓取 ====================

    def fetch_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """
        爬取单个 URL，提取正文文本

        流程：
          1. 检查 robots.txt 是否允许爬取（不允许则直接返回 None）
          2. 根据 Crawl-delay 等待
          3. 爬取并解析

        Args:
            url: 网页地址
            timeout: 超时秒数

        Returns:
            提取的纯文本，失败返回 None
        """
        # 1. robots.txt 合规检查
        if not self.robot_checker.can_fetch(url):
            logger.warning(f"⛔ robots.txt 禁止，跳过: {url}")
            return None

        # 2. 遵守 Crawl-delay
        self.robot_checker.wait_if_needed(url)

        # 3. 爬取
        try:
            resp = self.session.get(url, timeout=timeout)
            resp.encoding = resp.apparent_encoding or "utf-8"
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # 移除无用元素
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()

            # 提取文本
            text = soup.get_text(separator="\n", strip=True)

            # 清理空白行
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            # 过滤过短的行（可能是导航/广告）
            lines = [line for line in lines if len(line) > 8]
            text = "\n".join(lines)

            # 限制最大长度（取前 10000 字符）
            text = text[:10000]

            self._stats["fetched"] += 1
            logger.info(f"✅ 已爬取: {url} ({len(text)} 字符)")
            return text

        except requests.RequestException as e:
            logger.warning(f"❌ 爬取失败 [{url}]: {e}")
            return None

    def fetch_multiple(self, urls: list[str], max_concurrent: int = 3) -> list[dict]:
        """
        批量爬取多个 URL

        Args:
            urls: URL 列表
            max_concurrent: 最大并发数

        Returns:
            [{"url": str, "text": str, "success": bool}, ...]
        """
        results = []
        for url in urls:
            text = self.fetch_url(url)
            results.append({
                "url": url,
                "text": text or "",
                "success": text is not None,
            })
            time.sleep(1)  # 礼貌延迟
        return results

    def search_and_fetch(self, keyword: str, max_results: int = 5) -> list[dict]:
        """
        搜索关键词并爬取结果页内容

        使用 Bing 搜索获取结果列表，然后爬取每个结果的正文。

        Args:
            keyword: 搜索关键词
            max_results: 最大采集结果数

        Returns:
            [{"url": str, "title": str, "text": str, "success": bool}, ...]
        """
        # 先用 Bing 搜索
        search_url = f"https://www.bing.com/search?q={requests.utils.quote(keyword)}&count={max_results}"
        logger.info(f"🔍 搜索: {keyword}")

        search_text = self.fetch_url(search_url)
        if not search_text:
            logger.warning("搜索失败，无法获取搜索结果")
            return []

        # 用 BeautifulSoup 解析搜索结果链接
        try:
            resp = self.session.get(search_url, timeout=30)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")

            # 提取搜索结果链接
            links = []
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                # Bing 搜索结果的链接格式
                if href.startswith("https://") and not any(
                    d in href for d in ["bing.com", "microsoft.com", "scorecardresearch.com"]
                ):
                    if href not in [l["url"] for l in links]:
                        links.append({"url": href, "title": a.get_text(strip=True)[:100]})

            logger.info(f"  找到 {len(links)} 个结果链接")

            # robots.txt 过滤（筛掉禁止爬取的）
            allowed_links = [l for l in links if self.robot_checker.can_fetch(l["url"])]
            blocked_count = len(links) - len(allowed_links)
            if blocked_count:
                logger.info(f"  robots.txt 过滤: {blocked_count} 个链接被禁止")

            # 爬取每个允许的结果
            results = []
            for link in allowed_links[:max_results]:
                text = self.fetch_url(link["url"])
                results.append({
                    "url": link["url"],
                    "title": link["title"],
                    "text": text or "",
                    "success": text is not None,
                })
                time.sleep(1)

            return results

        except Exception as e:
            logger.error(f"解析搜索结果失败: {e}")
            return []

    # ==================== 文本 → 训练数据 ====================

    def text_to_training_pairs(self, text: str, source: str = "web",
                                max_pairs: int = 10) -> list[dict]:
        """
        用 LoveFlow LLM 将原始文本转化为 Q&A 训练对

        流程：
          1. 将文本分块（每块 ~1000 字）
          2. 对每块调用 LLM 生成 Q&A 对
          3. 返回 prompt/completion 列表

        Args:
            text: 原始文本
            source: 来源标记
            max_pairs: 最大生成数量

        Returns:
            [{"prompt": "...", "completion": "...", "source": "..."}, ...]
        """
        pairs = []

        # 将文本分块（~800 字一块，避免超过 LLM 上下文）
        chunks = self._split_text(text, chunk_size=800)

        for i, chunk in enumerate(chunks):
            if len(pairs) >= max_pairs:
                break

            # 调用 LoveFlow 生成 Q&A
            qa_text = self._generate_qa(chunk)
            if not qa_text:
                continue

            # 解析返回的 Q&A 对
            parsed = self._parse_qa_text(qa_text, source)
            pairs.extend(parsed)

            self._stats["generated"] += len(parsed)
            logger.info(f"  分块 {i+1}/{len(chunks)} → 生成 {len(parsed)} 个训练对")

        return pairs[:max_pairs]

    def _split_text(self, text: str, chunk_size: int = 800) -> list[str]:
        """将长文本按段落分割成块"""
        paragraphs = text.split("\n")
        chunks = []
        current = []

        for para in paragraphs:
            current.append(para)
            if sum(len(p) for p in current) >= chunk_size:
                chunks.append("\n".join(current))
                current = []

        if current:
            chunks.append("\n".join(current))

        return [c for c in chunks if len(c) > 50]  # 过滤过短的块

    def _generate_qa(self, chunk: str) -> Optional[str]:
        """
        调用 LoveFlow LLM 从文本块生成 Q&A 对

        Prompt 设计：
          请根据以下文本生成 2-3 个问答对，格式：
          Q: 问题
          A: 答案
        """
        prompt = (
            "你是一个训练数据生成器。请根据以下文本内容，生成 2-3 个问答对（Q&A）。\n"
            "要求：\n"
            "  - 问题要有价值，覆盖文本的核心知识点\n"
            "  - 答案要准确、完整，直接使用原文信息\n"
            "  - 格式：每对问答用空行分隔\n\n"
            f"文本内容:\n{chunk}\n\n"
            "请生成问答对:"
        )

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            if response and not response.startswith("[推理错误"):
                return response
            return None
        except Exception as e:
            logger.warning(f"LLM 生成 Q&A 失败: {e}")
            return None

    def _parse_qa_text(self, text: str, source: str) -> list[dict]:
        """解析 LLM 返回的 Q&A 文本"""
        pairs = []

        # 支持多种格式
        patterns = [
            # Q: ... A: ... 格式
            r'(?:^|\n)\s*[QQ][：:]\s*(.*?)\n\s*[AA][：:]\s*(.*?)(?=\n\s*[Qq]|\Z)',
            # 问题: ... 答案: ... 格式
            r'(?:^|\n)\s*问题[：:]\s*(.*?)\n\s*答案[：:]\s*(.*?)(?=\n\s*问题|\Z)',
            # 问: ... 答: ... 格式
            r'(?:^|\n)\s*问[：:]\s*(.*?)\n\s*答[：:]\s*(.*?)(?=\n\s*问|\Z)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for q, a in matches:
                q = q.strip()
                a = a.strip()
                if q and a and len(q) > 5 and len(a) > 5:
                    pairs.append({
                        "prompt": q,
                        "completion": a,
                        "source": f"web_{source}",
                        "dataset_name": f"web_{datetime.now().strftime('%Y%m%d')}",
                    })

        # 去重
        seen = set()
        unique_pairs = []
        for p in pairs:
            key = p["prompt"][:50]
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)

        return unique_pairs

    # ==================== MySQL 存储 ====================

    def store_to_mysql(self, pairs: list[dict]) -> int:
        """
        将训练数据对存入 MySQL training_data 表

        Args:
            pairs: [{"prompt": "...", "completion": "...", "source": "...", ...}]

        Returns:
            成功存储的数量
        """
        if not pairs:
            return 0

        try:
            conn = pymysql.connect(
                host=MYSQL_HOST, port=MYSQL_PORT,
                user=MYSQL_USER, password=MYSQL_PASSWORD,
                database=MYSQL_DB, charset="utf8mb4",
            )
            with conn.cursor() as cursor:
                sql = """INSERT INTO training_data
                         (prompt, completion, source, dataset_name, status)
                         VALUES (%s, %s, %s, %s, 1)"""
                count = 0
                for pair in pairs:
                    try:
                        cursor.execute(sql, (
                            pair["prompt"],
                            pair["completion"],
                            pair.get("source", "web_auto"),
                            pair.get("dataset_name", "web_auto"),
                        ))
                        count += 1
                    except Exception as e:
                        logger.debug(f"存储失败: {e}")
                        continue
                conn.commit()

            conn.close()
            self._stats["stored"] += count
            logger.info(f"💾 已存入 MySQL: {count} 条训练数据")
            return count

        except pymysql.Error as e:
            logger.error(f"MySQL 存储失败: {e}")
            return 0

    # ==================== 完整工作流 ====================

    def collect_from_urls(self, urls: list[str], auto_train: bool = False,
                          max_pairs_per_url: int = 5) -> dict:
        """
        完整流程：爬取 URL → 生成训练数据 → 存入 MySQL → 可选训练

        Args:
            urls: URL 列表
            auto_train: 采集完成后是否自动启动训练
            max_pairs_per_url: 每个 URL 最大生成训练对数量

        Returns:
            {"fetched": int, "generated": int, "stored": int, "task_id": str}
        """
        task_desc = f"联网采集: {len(urls)} 个 URL"
        task_id = create_task(task_desc)
        update_task(task_id, status="in_progress")

        all_pairs = []
        results = self.fetch_multiple(urls)

        for r in results:
            if r["success"] and r["text"]:
                pairs = self.text_to_training_pairs(r["text"], source=r["url"], max_pairs=max_pairs_per_url)
                all_pairs.extend(pairs)

        stored = self.store_to_mysql(all_pairs)

        result = {
            "fetched": self._stats["fetched"],
            "generated": len(all_pairs),
            "stored": stored,
            "task_id": task_id,
        }

        update_task(task_id, status="completed" if stored > 0 else "failed")

        # 自动触发训练
        if auto_train and stored > 0:
            logger.info("🔄 自动触发训练...")
            from agent.training.orchestrator import TrainingOrchestrator
            trainer = TrainingOrchestrator()
            train_result = trainer.start_training(epochs=3)
            result["train_result"] = train_result

        return result

    def collect_from_search(self, keyword: str, max_results: int = 5,
                             auto_train: bool = False, max_pairs_total: int = 20) -> dict:
        """
        完整流程：搜索关键词 → 爬取 → 生成训练数据 → 存入 MySQL → 可选训练

        Args:
            keyword: 搜索关键词
            max_results: 最大爬取结果数
            auto_train: 是否自动训练
            max_pairs_total: 最大生成训练对总数

        Returns:
            {"keyword": str, "fetched": int, "generated": int, "stored": int, ...}
        """
        task_desc = f"联网采集: 搜索「{keyword}」"
        task_id = create_task(task_desc)
        update_task(task_id, status="in_progress")

        # 搜索 + 爬取
        search_results = self.search_and_fetch(keyword, max_results)

        # 生成训练数据
        all_pairs = []
        for r in search_results:
            if r["success"] and r["text"]:
                pairs = self.text_to_training_pairs(
                    r["text"],
                    source=f"search:{keyword}",
                    max_pairs=max(1, max_pairs_total // len(search_results)),
                )
                all_pairs.extend(pairs)

        stored = self.store_to_mysql(all_pairs)

        result = {
            "keyword": keyword,
            "fetched": self._stats["fetched"],
            "generated": len(all_pairs),
            "stored": stored,
            "task_id": task_id,
        }

        update_task(task_id, status="completed" if stored > 0 else "failed")

        # 自动触发训练
        if auto_train and stored > 0:
            logger.info("🔄 自动触发训练...")
            from agent.training.orchestrator import TrainingOrchestrator
            trainer = TrainingOrchestrator()
            train_result = trainer.start_training(epochs=3)
            result["train_result"] = train_result

        return result

    # ==================== 上下文注入 ====================

    def get_context(self) -> Optional[str]:
        """供 Agent 上下文注入"""
        total = self._stats["stored"]
        robot_stats = self.robot_checker.get_stats()
        parts = ["【数据采集器】"]

        # 采集统计
        if total > 0:
            parts.append(
                f"  本会话已采集 {self._stats['fetched']} 个页面，"
                f"生成 {self._stats['generated']} 个训练对，入库 {total} 条"
            )
        else:
            parts.append("  可联网爬取数据并转为训练集")

        # robots.txt 合规统计
        if robot_stats["checked"] > 0:
            parts.append(
                f"  robots.txt 合规: 检查 {robot_stats['checked']} 次，"
                f"允许 {robot_stats['allowed']} 次，"
                f"拒绝 {robot_stats['blocked']} 次"
            )

        parts.append("  用法: 「帮我采集 https://... 来训练」或「搜索关于xxx的资料来训练」")
        return "\n".join(parts) + "\n"

    def get_stats(self) -> dict:
        """获取采集统计"""
        stats = dict(self._stats)
        stats["robots"] = self.robot_checker.get_stats()
        return stats
