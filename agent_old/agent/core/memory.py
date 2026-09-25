"""
SQLite FTS5 持久化记忆系统（增强版）

存储类型：
  - knowledge: 跨会话项目知识
  - session: 会话摘要/检查点
  - note: 临时笔记
  - task: 任务记录
  - face: 人像特征向量（BLOB）

新增功能：
  - 标签系统：为记忆打标签便于分类检索
  - 向量存储：存储人脸 128 维特征向量（BLOB）
"""
import json
import sqlite3
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.agent_config import MEMORY_DIR

MEMORY_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = MEMORY_DIR / "memory.db"
MEMORY_MD_PATH = MEMORY_DIR / "MEMORY.md"
NOTES_PATH = MEMORY_DIR / "notes.md"
TASKS_DIR = MEMORY_DIR / "tasks"
TASKS_DIR.mkdir(exist_ok=True)


def get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接（FTS5 已启用）"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection):
    """初始化表结构（含增强字段）"""
    # 主记忆表（增加 tags 和 vector 字段）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL DEFAULT 'knowledge',
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',            -- JSON 数组，例如 '["人物","技术"]'
            source TEXT DEFAULT '',
            importance INTEGER DEFAULT 1,
            vector BLOB DEFAULT NULL,        -- 特征向量（例如人脸 128 维 float32）
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    # FTS5 全文搜索
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            title, content, tags, content=memories, content_rowid=id
        )
    """)
    # 同步触发器
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, title, content, tags) VALUES (new.id, new.title, new.content, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, title, content, tags) VALUES('delete', old.id, old.title, old.content, old.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, title, content, tags) VALUES('delete', old.id, old.title, old.content, old.tags);
            INSERT INTO memories_fts(rowid, title, content, tags) VALUES (new.id, new.title, new.content, new.tags);
        END;
    """)
    # 标签索引
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)
    """)
    # 人脸向量匹配索引（按 doc_type 快速过滤）
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_doc_type ON memories(doc_type)
    """)


# ==================== CRUD 操作 ====================

def add_memory(
    content: str,
    doc_type: str = "knowledge",
    title: str = "",
    tags: Optional[list[str]] = None,
    source: str = "",
    importance: int = 1,
    vector: Optional[list[float]] = None,
) -> int:
    """
    添加记忆条目，返回 ID
    vector: 128 维 float32 特征向量列表
    """
    now = time.time()
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    vector_blob = _vector_to_blob(vector) if vector else None
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO memories (doc_type, title, content, tags, source, importance, vector, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_type, title, content, tags_json, source, importance, vector_blob, now, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _vector_to_blob(vector: list[float]) -> bytes:
    """将 float32 列表转为 BLOB"""
    return struct.pack(f"{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes) -> list[float]:
    """将 BLOB 转为 float32 列表"""
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _sanitize_fts_query(text: str) -> str:
    """清理 FTS5 查询文本，移除特殊语法字符，避免查询报错"""
    # FTS5 特殊字符: ^ * + - ~ ( ) { } [ ] " : < > |
    # 替换为空格保留分词，避免被解析为语法操作符
    import re
    text = re.sub(r'[+\-~(){}[\]":<>|*^]', ' ', text)
    # 压缩多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else ""


def search_memory(query: str, limit: int = 10, doc_type: Optional[str] = None, tags: Optional[list[str]] = None) -> list[dict]:
    """
    FTS5 全文搜索，按相关性排名
    支持按 doc_type 和标签过滤
    """
    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return []

    conn = get_conn()
    try:
        conditions = []
        params = [safe_query]

        if doc_type:
            conditions.append("m.doc_type = ?")
            params.append(doc_type)

        if tags:
            # 标签过滤：JSON 数组包含任一标签
            tag_conditions = " OR ".join(["m.tags LIKE ?" for _ in tags])
            conditions.append(f"({tag_conditions})")
            params.extend([f'%"{t}"%' for t in tags])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        rows = conn.execute(f"""
            SELECT m.id, m.doc_type, m.title, m.content, m.tags, m.source,
                   m.importance, m.created_at, m.vector,
                   snippet(memories_fts, 1, '<b>', '</b>', '...', 40) AS snip
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.id
            WHERE memories_fts MATCH ? AND {where_clause}
            ORDER BY rank
            LIMIT ?
        """, params + [limit]).fetchall()

        return [
            {
                "id": r[0], "doc_type": r[1], "title": r[2],
                "content": r[3], "tags": json.loads(r[4]) if r[4] else [],
                "source": r[5], "importance": r[6],
                "created_at": r[7],
                "vector": _blob_to_vector(r[8]) if r[8] else None,
                "snippet": r[9],
            }
            for r in rows
        ]
    finally:
        conn.close()


def search_by_vector(vector: list[float], threshold: float = 0.6, limit: int = 5) -> list[dict]:
    """
    按余弦相似度搜索特征向量
    用于人脸匹配
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, doc_type, title, content, tags, source, importance, created_at, vector FROM memories WHERE vector IS NOT NULL AND doc_type='face'"
        ).fetchall()

        results = []
        query_vec = _normalize(vector)

        for r in rows:
            db_vec = _blob_to_vector(r[8])
            if db_vec:
                similarity = _cosine_similarity(query_vec, _normalize(db_vec))
                if similarity >= threshold:
                    results.append({
                        "id": r[0], "doc_type": r[1], "title": r[2],
                        "content": r[3], "tags": json.loads(r[4]) if r[4] else [],
                        "source": r[5], "importance": r[6],
                        "created_at": r[7],
                        "similarity": similarity,
                    })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    finally:
        conn.close()


def _normalize(v: list[float]) -> list[float]:
    """L2 归一化"""
    import math
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度"""
    return sum(x * y for x, y in zip(a, b))


def list_memories(doc_type: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[dict]:
    """列出记忆（按更新时间倒序）"""
    conn = get_conn()
    try:
        if doc_type:
            rows = conn.execute(
                "SELECT id, doc_type, title, content, tags, source, importance, created_at, updated_at FROM memories WHERE doc_type=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (doc_type, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, doc_type, title, content, tags, source, importance, created_at, updated_at FROM memories ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {"id": r[0], "doc_type": r[1], "title": r[2], "content": r[3],
             "tags": json.loads(r[4]) if r[4] else [], "source": r[5],
             "importance": r[6], "created_at": r[7], "updated_at": r[8]}
            for r in rows
        ]
    finally:
        conn.close()


def update_memory(id: int, content: Optional[str] = None, title: Optional[str] = None,
                  tags: Optional[list[str]] = None, importance: Optional[int] = None) -> bool:
    """更新记忆"""
    conn = get_conn()
    try:
        updates = []
        params = []
        if content is not None:
            updates.append("content=?")
            params.append(content)
        if title is not None:
            updates.append("title=?")
            params.append(title)
        if tags is not None:
            updates.append("tags=?")
            params.append(json.dumps(tags, ensure_ascii=False))
        if importance is not None:
            updates.append("importance=?")
            params.append(importance)
        if not updates:
            return False
        updates.append("updated_at=?")
        params.append(time.time())
        params.append(id)
        conn.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id=?", params)
        conn.commit()
        return True
    finally:
        conn.close()


def delete_memory(id: int) -> bool:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM memories WHERE id=?", (id,))
        conn.commit()
        return True
    finally:
        conn.close()


# ==================== 标签操作 ====================

def get_all_tags() -> list[str]:
    """获取所有标签"""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT DISTINCT tags FROM memories WHERE tags != ''").fetchall()
        all_tags = set()
        for (tags_json,) in rows:
            try:
                for t in json.loads(tags_json):
                    all_tags.add(t)
            except:
                pass
        return sorted(all_tags)
    finally:
        conn.close()


# ==================== 文件同步 ====================

def sync_to_markdown():
    """将 SQLite 记忆同步到 MEMORY.md"""
    entries = list_memories("knowledge", limit=200)

    lines = [
        "# LoveFlow Agent 持久化记忆",
        "",
        f"*最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "---",
        "",
    ]
    for e in entries:
        tags_str = f" [{', '.join(e['tags'])}]" if e.get('tags') else ""
        lines.append(f"## [{e.get('importance', 1)}]{tags_str} {e.get('title', '无标题')}")
        lines.append(f"*来源: {e.get('source', '未知')} | ID: {e['id']}*")
        lines.append("")
        lines.append(e.get("content", ""))
        lines.append("")
        lines.append("---")
        lines.append("")

    MEMORY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def append_note(content: str):
    """追加笔记"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(NOTES_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## [{timestamp}]\n{content}\n")


def write_task_progress(task_id: str, content: str):
    """写入任务进度文件"""
    task_file = TASKS_DIR / f"{task_id}.md"
    with open(task_file, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{content}\n")
