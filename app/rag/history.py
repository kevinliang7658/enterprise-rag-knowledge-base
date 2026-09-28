"""会话历史：用 sqlite 存多轮对话，让模型"记得"上文。

为什么自己写：
两张表 50 行代码。"记忆"的本质就是：
每次请求把历史消息一起塞进 prompt。没有魔法。

本质上就是一个用 sqlite 当存储的 MessageRepository，
sqlite3 是 Python 标准库，零依赖。
"""
import sqlite3
import time
import uuid
from pathlib import Path

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,        -- 会话ID，相当于一次对话的主键
    title       TEXT NOT NULL DEFAULT '',-- 会话标题（取首条问题前20字）
    created_at  REAL NOT NULL            -- 创建时间戳
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,           -- 外键：属于哪个会话
    role        TEXT NOT NULL,           -- user / assistant
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.SQLITE_PATH))
    conn.row_factory = sqlite3.Row  # 查询结果可以像字典一样按列名取值
    return conn


def init_db() -> None:
    """建表。IF NOT EXISTS 保证重复执行不报错（幂等，类似 Flyway 已执行过的脚本会跳过）。"""
    Path(config.SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def create_session(title: str = "") -> str:
    """新建会话，返回 session_id。"""
    init_db()
    session_id = uuid.uuid4().hex[:16]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id, title, created_at) VALUES (?, ?, ?)",
            (session_id, title, time.time()),
        )
    return session_id


def append_message(session_id: str, role: str, content: str) -> None:
    """往会话里追加一条消息。会话不存在则自动创建（宽容设计，简化前端逻辑）。"""
    init_db()
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO sessions(session_id, title, created_at) VALUES (?, ?, ?)",
                (session_id, content[:20] if role == "user" else "", time.time()),
            )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    """取最近 limit 轮历史，按时间正序返回，直接可拼进 OpenAI messages 参数。

    只取最近 N 轮的原因：token 是要花钱的，且模型上下文有限，
    全塞进去又贵又慢 —— 这叫"滑动窗口"记忆策略。
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    # 查出来是倒序的（最新的在前），翻回正序
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def list_sessions() -> list[dict]:
    """列出所有会话（前端侧边栏用）。"""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
