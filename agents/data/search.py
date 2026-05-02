#!/usr/bin/env python3
"""
SQLite FTS5 全文搜索引擎索引（用于会话条目）。

架构：
  - JSONL（session.py）→ 主存储（树形结构，仅追加）
  - SQLite FTS5（本模块）→ 搜索索引（内容索引，快速查询）

为什么需要两者？
  - JSONL 擅长：树形操作、分支、压缩、人类可读
  - SQLite FTS5 擅长：跨会话搜索、关键词匹配、
    按相关性排序、前缀/短语查询
  - jieba：中文分词（FTS5 内置分词器无法处理 CJK 字符）

数据流：
  1. SessionManager.append_message() → 写入 JSONL
  2. SearchIndex._tokenize_for_index() → jieba 对中文分词
  3. SearchIndex.index_entry()       → 将分词后的内容插入 FTS5
  4. 用户搜索                       → jieba 对查询分词 → FTS5 MATCH → 排序结果

FTS5 表结构：
  CREATE VIRTUAL TABLE session_fts USING fts5(
      entry_id, session_id, session_file, entry_type, role,
      content, cwd, timestamp, parent_id,
      tokenize='unicode61'   -- 基础 Unicode 分词；重活由 Python 中的 jieba 完成
  );

中文分词策略：
  - 索引时：jieba.lcut(text) → 用空格连接分词 → 存入 FTS5
  - 搜索时：jieba.lcut(query) → 用 AND 连接 → FTS5 MATCH 表达式
  - 英文原文不变（空格已分隔单词）

数据库位置：
  <SESSIONS_DIR>/.search.db   （与 .jsonl 文件同目录）
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union

from ..core.config import SESSIONS_DIR

# 延迟加载 jieba（仅在真正需要中文分词时才加载）
_jieba = None

def _get_jieba():
    """延迟加载 jieba，避免模块级导入时的性能开销。"""
    global _jieba
    if _jieba is None:
        import logging
        import sys
        import io
        # 关闭 jieba 的 logging 输出
        jieba_logger = logging.getLogger('jieba')
        jieba_logger.setLevel(logging.WARNING)
        jieba_logger.handlers = []
        # 临时抑制 stdout/stderr 来屏蔽 jieba 的 print 输出
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
        try:
            import jieba as _jb
            # 触发初始化（让 "Building prefix dict" 等输出被吞掉）
            _jb.lcut("init")
        finally:
            sys.stdout, sys.stderr = _old_stdout, _old_stderr
        _jieba = _jb
    return _jieba

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_SEARCH_DB_NAME = ".search.db"
_FTS_TABLE = "session_fts"

# 线程安全的单例连接
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_db_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# 数据库连接管理
# ---------------------------------------------------------------------------


def _get_db_path() -> Path:
    """获取搜索数据库的路径。"""
    global _db_path
    if _db_path is None:
        _db_path = SESSIONS_DIR / _SEARCH_DB_NAME
    return _db_path


def _get_connection() -> sqlite3.Connection:
    """
    获取或创建 SQLite 连接（线程安全单例）。
    首次使用时创建 FTS5 表。
    """
    global _conn, _db_path
    with _lock:
        if _conn is None:
            db_path = _get_db_path()
            # 确保父目录存在
            db_path.parent.mkdir(parents=True, exist_ok=True)

            _conn = sqlite3.connect(str(db_path))
            _conn.row_factory = sqlite3.Row

            # 启用 WAL 模式以提升并发读取性能
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")

            # 如果不存在则创建 FTS5 虚拟表
            _conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
                    entry_id,
                    session_id,
                    session_file,
                    entry_type,
                    role,
                    content,
                    cwd,
                    timestamp,
                    parent_id,
                    tokenize='unicode61'
                )
            """)
            _conn.commit()

        return _conn


def close_search_db():
    """关闭搜索数据库连接。"""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


# ---------------------------------------------------------------------------
# 中文分词（jieba）
# ---------------------------------------------------------------------------


# 检测 CJK 字符（中文/日文/韩文）的正则表达式
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')


def _contains_cjk(text: str) -> bool:
    """检查文本是否包含 CJK（中文）字符。"""
    return bool(_CJK_RE.search(text))


def _tokenize_for_index(text: str) -> str:
    """
    对文本进行 FTS5 索引分词。

    策略：
      - 如果文本包含 CJK：使用 jieba 分词，用空格连接。
        这样 FTS5 可以将每个中文词作为独立索引词。
      - 如果是纯英文：原样返回（FTS5 可处理空格分隔的单词）。
      - 中英混合内容：整体使用 jieba 分词。

    示例：
      'SQLite FTS5 全文搜索' → 'SQLite FTS5 全文 搜索'
      '树形JSONL会话管理'   → '树形 JSONL 会话 管理'
      'hello world'         → 'hello world'（不变）
    """
    if not text:
        return text
    if not _contains_cjk(text):
        return text  # 纯英文，无需分词

    jb = _get_jieba()
    # 使用 cut_for_search 模式以获得更好的搜索召回率
    tokens = jb.cut_for_search(text)
    # 过滤：保留有意义的词元（跳过单个标点、空格）
    filtered = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
    return " ".join(filtered)


def _tokenize_for_search(query: str) -> str:
    """
    对用户搜索查询进行 FTS5 MATCH 表达式分词。

    策略：
      - 如果查询包含 CJK：使用 jieba 分词，用 AND 连接。
        确保所有中文词都必须匹配（高精度）。
      - 如果查询看起来像高级 FTS5 表达式（含 AND/OR/NOT），
        原样返回（用户知道自己在做什么）。
      - 简单英文查询：原样返回或加引号进行短语匹配。

    示例：
      '全文搜索'       → '全文 AND 搜索'
      '会话管理'       → '会话 AND 管理'
      'session AND branch' → 'session AND branch'（不变）
      'SQLite FTS5'     → '"SQLite FTS5"'（短语匹配）
    """
    if not query:
        return query

    # 如果查询已含 FTS5 操作符，尊重用户意图
    fts5_ops = re.compile(r'\b(AND|OR|NOT|NEAR)\b|[()*"]')
    if fts5_ops.search(query):
        # 仍尝试对表达式中的 CJK 部分进行分词
        if _contains_cjk(query):
            jb = _get_jieba()
            tokens = jb.lcut(query)
            and_joined = " AND ".join(f'"{t}"' if _contains_cjk(t) else t
                                   for t in tokens if t.strip())
            return and_joined
        return query

    # 简单查询：分词并用 AND 连接以确保精度
    if _contains_cjk(query):
        jb = _get_jieba()
        tokens = jb.lcut(query)
        # 中文词元用 AND 连接以实现精确匹配
        and_joined = " AND ".join(f'"{t}"' if _contains_cjk(t) else t
                               for t in tokens if t.strip())
        return and_joined

    # 纯英文简单查询：短语匹配
    return f'"{query}"'


# ---------------------------------------------------------------------------
# 内容提取辅助函数
# ---------------------------------------------------------------------------


def _extract_text_from_message(message: dict) -> tuple[str, str]:
    """
    从消息字典中提取可搜索的文本和角色。
    返回 (text_content, role)。
    """
    role = message.get("role", "")
    content = message.get("content", "")

    if isinstance(content, str):
        return content, role
    elif isinstance(content, list):
        # 内容块（如 Anthropic 格式）
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    inp = block.get("input", {})
                    parts.append(f"[tool:{block.get('name','')}] {json.dumps(inp, default=str)[:500]}")
                elif block.get("type") == "tool_result":
                    parts.append(f"[result] {str(block.get('content', ''))[:500]}")
        return "\n".join(parts), role
    return str(content), role


def _extract_searchable_text(entry: dict) -> tuple[str, str]:
    """
    从任意类型的条目中提取可搜索文本和角色。
    返回 (text_content, role)。
    """
    etype = entry.get("type", "")

    if etype == "message":
        msg = entry.get("message", {})
        return _extract_text_from_message(msg)
    elif etype == "compaction":
        return entry.get("summary", ""), "compaction"
    elif etype == "branch_summary":
        return entry.get("summary", ""), "branch_summary"
    elif etype == "label":
        label = entry.get("label", "")
        target = entry.get("targetId", "")
        return f"[label] {label} (on {target})", "label"
    elif etype == "model_change":
        provider = entry.get("provider", "")
        model = entry.get("modelId", "")
        return f"[model_change] provider={provider} model={model}", "system"
    elif etype == "custom_message":
        return entry.get("content", ""), "user"
    elif etype == "custom":
        data = entry.get("data", "")
        ctype = entry.get("customType", "")
        return f"[{ctype}] {json.dumps(data, default=str)[:300] if data else ''}", "custom"
    else:
        # 兜底：将整个条目转为文本（截断）
        return json.dumps(entry, default=str)[:500], etype or "unknown"


# ---------------------------------------------------------------------------
# 索引操作
# ---------------------------------------------------------------------------


class SearchIndex:
    """
    管理 SQLite FTS5 会话条目搜索索引。

    用法：
        idx = SearchIndex()
        idx.index_entry(entry, session_id, session_file, cwd)
        results = idx.search("keyword")
        idx.remove_session(session_id)   # 删除时清理
    """

    def __init__(self, db_path: Union[str, Path] = None):
        self._local_db_path = db_path  # 用于测试；通常使用单例

    def _conn(self) -> sqlite3.Connection:
        if self._local_db_path:
            from pathlib import Path as _P
            _p = _P(self._local_db_path)
            _p.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(_p))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            # 确保本地连接也有 FTS5 表
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
                    entry_id, session_id, session_file, entry_type, role,
                    content, cwd, timestamp, parent_id,
                    tokenize='unicode61'
                )
            """)
            return conn
        return _get_connection()

    def index_entry(self, entry: dict, session_id: str,
                    session_file: Union[str, Path], cwd: str):
        """
        在 FTS5 索引中添加或更新一条记录。
        由 SessionManager 在每次追加后自动调用。
        """
        eid = entry.get("id", "")
        if not eid:
            return

        etype = entry.get("type", "")
        content_text, role = _extract_searchable_text(entry)
        # 存入 FTS5 前先用 jieba 对中文分词
        content_text = _tokenize_for_index(content_text)
        timestamp = entry.get("timestamp", "")
        parent_id = entry.get("parentId") or ""

        conn = self._conn()
        try:
            # FTS5 使用 INSERT OR REPLACE 实现更新或插入语义
            conn.execute(
                f"INSERT OR REPLACE INTO {_FTS_TABLE} "
                "(entry_id, session_id, session_file, entry_type, role, "
                "content, cwd, timestamp, parent_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    eid,
                    session_id,
                    str(session_file),
                    etype,
                    role,
                    content_text,
                    cwd,
                    timestamp,
                    parent_id,
                ),
            )
            conn.commit()
        except Exception:
            pass  # 非关键错误：搜索索引失败不应中断会话
        finally:
            if self._local_db_path:
                conn.close()

    def index_entries(self, entries: list[dict], session_id: str,
                      session_file: Union[str, Path], cwd: str):
        """批量索引多条记录（用于初始建索引）。"""
        conn = self._conn()
        try:
            for entry in entries:
                eid = entry.get("id", "")
                if not eid or entry.get("type") == "session":
                    continue
                etype = entry.get("type", "")
                content_text, role = _extract_searchable_text(entry)
                # 存入 FTS5 前先用 jieba 对中文分词
                content_text = _tokenize_for_index(content_text)
                timestamp = entry.get("timestamp", "")
                parent_id = entry.get("parentId") or ""

                conn.execute(
                    f"INSERT OR REPLACE INTO {_FTS_TABLE} "
                    "(entry_id, session_id, session_file, entry_type, role, "
                    "content, cwd, timestamp, parent_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (eid, session_id, str(session_file), etype, role,
                     content_text, cwd, timestamp, parent_id),
                )
            conn.commit()
        except Exception:
            pass
        finally:
            if self._local_db_path:
                conn.close()

    def remove_entry(self, entry_id: str):
        """从索引中移除单条记录。"""
        conn = self._conn()
        try:
            conn.execute(
                f"DELETE FROM {_FTS_TABLE} WHERE entry_id = ?",
                (entry_id,),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            if self._local_db_path:
                conn.close()

    def remove_session(self, session_id: str):
        """从索引中移除某个会话的所有记录。"""
        conn = self._conn()
        try:
            conn.execute(
                f"DELETE FROM {_FTS_TABLE} WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            if self._local_db_path:
                conn.close()

    # ------------------------------------------------------------------
    # 搜索查询
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20,
               session_id: str = None, cwd: str = None,
               entry_type: str = None, role: str = None) -> list[dict]:
        """
        对已索引的会话条目进行全文搜索。

        参数：
            query: FTS5 搜索表达式（支持 AND、OR、NOT、短语 "*"、
                  前缀 "prefix*"、NEAR 操作符）
            limit: 最大返回结果数
            session_id: 过滤指定会话
            cwd: 过滤工作目录
            entry_type: 按条目类型过滤（message、compaction 等）
            role: 按消息角色过滤（user、assistant）

        返回：
            结果字典列表，包含以下键：
                entry_id, session_id, session_file, entry_type, role,
                content（摘要/高亮）、cwd, timestamp, parent_id,
                rank（相关性得分，越低越相关）

        示例查询：
            "SQLite FTS5"           — 短语匹配
            "session AND branch"    — AND
            "session OR branch"     — OR
            "session NOT branch"    — NOT
            "compac*"               — 前缀匹配
            'NEAR("session" "search")'  — 近邻查询
        """
        conn = self._conn()
        try:
            # 动态构建 WHERE 子句
            conditions = []
            params: list = []

            conditions.append(f"{_FTS_TABLE} MATCH ?")
            # 使用 jieba 对中文查询分词以改善 CJK 搜索效果
            fts_query = _tokenize_for_search(query)
            params.append(fts_query)

            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            if cwd:
                conditions.append("cwd = ?")
                params.append(cwd)
            if entry_type:
                conditions.append("entry_type = ?")
                params.append(entry_type)
            if role:
                conditions.append("role = ?")
                params.append(role)

            where_clause = " AND ".join(conditions)

            # 使用 bm25() 排序（FTS5 内置函数）
            sql = f"""
                SELECT
                    entry_id,
                    session_id,
                    session_file,
                    entry_type,
                    role,
                    snippet({_FTS_TABLE}, 5, '[[', ']]', '...', 120) AS snippet,
                    cwd,
                    timestamp,
                    parent_id,
                    bm25({_FTS_TABLE}) AS rank
                FROM {_FTS_TABLE}
                WHERE {where_clause}
                ORDER BY rank
                LIMIT ?
            """
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            results = []
            seen_ids: set = set()
            for row in rows:
                eid = row["entry_id"]
                if eid in seen_ids:
                    continue  # 去重：同一 entry_id 只保留第一条
                seen_ids.add(eid)
                results.append({
                    "entry_id": eid,
                    "session_id": row["session_id"],
                    "session_file": row["session_file"],
                    "entry_type": row["entry_type"],
                    "role": row["role"],
                    "snippet": row["snippet"] or "",
                    "cwd": row["cwd"],
                    "timestamp": row["timestamp"],
                    "parent_id": row["parent_id"],
                    "rank": row["rank"],
                })
            return results
        except sqlite3.OperationalError as e:
            # 常见问题：无效的 FTS5 查询语法
            if "fts5:" in str(e).lower():
                return [{"error": f"Invalid search syntax: {e}"}]
            raise
        finally:
            if self._local_db_path:
                conn.close()

    def search_sessions(self, query: str, limit: int = 10) -> list[dict]:
        """
        跨会话搜索，每个会话返回最佳匹配结果。
        适用于「我在哪个会话聊过 X？」这类查询。

        注意：FTS5 的 bm25() 不能在子查询或 GROUP BY 中使用，
        所以采用两步法：先获取原始匹配，再在 Python 中聚合。
        """
        conn = self._conn()
        try:
            # 步骤1：获取所有匹配记录及其 bm25 排名
            entry_sql = f"""
                SELECT
                    session_id, session_file, cwd, entry_type,
                    bm25({_FTS_TABLE}) AS entry_rank
                FROM {_FTS_TABLE}
                WHERE {_FTS_TABLE} MATCH ?
                ORDER BY entry_rank
                LIMIT 200
            """
            rows = conn.execute(entry_sql, [_tokenize_for_search(query)]).fetchall()

            if not rows:
                return []

            # 步骤2：在 Python 中按 session_id 聚合
            from collections import defaultdict
            session_map: dict[str, dict] = {}
            for row in rows:
                sid = row["session_id"]
                if sid not in session_map:
                    session_map[sid] = {
                        "session_id": sid,
                        "session_file": row["session_file"],
                        "cwd": row["cwd"],
                        "match_count": 0,
                        "best_rank": row["entry_rank"],
                        "types": set(),
                    }
                s = session_map[sid]
                s["match_count"] += 1
                if row["entry_rank"] < s["best_rank"]:
                    s["best_rank"] = row["entry_rank"]
                if row["entry_type"]:
                    s["types"].add(row["entry_type"])

            # 按 best_rank 和 match_count 排序并截取
            results = sorted(
                session_map.values(),
                key=lambda x: (x["best_rank"], -x["match_count"]),
            )[:limit]

            # 将集合转为字符串以便输出
            for r in results:
                r["types"] = ",".join(sorted(r["types"]))

            return results
        except sqlite3.OperationalError as e:
            if "fts5:" in str(e).lower():
                return [{"error": f"Invalid search syntax: {e}"}]
            raise
        finally:
            if self._local_db_path:
                conn.close()

    # ------------------------------------------------------------------
    # 维护操作
    # ------------------------------------------------------------------

    def rebuild_index(self, session_manager=None):
        """
        从 JSONL 文件重建整个搜索索引。
        如果指定了 session_manager，只重建该会话的索引。
        否则重建所有会话的索引。
        """
        conn = self._conn()
        try:
            if session_manager:
                # 重建单个会话的索引
                self.remove_session(session_manager.session_id)
                entries = session_manager.get_entries()
                self.index_entries(
                    entries,
                    session_manager.session_id,
                    session_manager.session_file or "",
                    session_manager.cwd,
                )
            else:
                # 全量重建：清空 + 扫描所有 JSONL 文件
                conn.execute(f"DELETE FROM {_FTS_TABLE}")
                conn.commit()

                from .session import _load_entries, _is_valid_session_file, SESSIONS_DIR

                if SESSIONS_DIR.exists():
                    for subdir in SESSIONS_DIR.iterdir():
                        if not subdir.is_dir():
                            continue
                        for f in subdir.glob("*.jsonl"):
                            if not _is_valid_session_file(f):
                                continue
                            entries = _load_entries(f)
                            if not entries:
                                continue
                            header = entries[0]
                            sid = header.get("id", "")
                            scwd = header.get("cwd", "")
                            msg_entries = [
                                e for e in entries if e.get("type") != "session"
                            ]
                            self.index_entries(msg_entries, sid, f, scwd)
        except Exception as e:
            return f"Rebuild error: {e}"
        finally:
            if self._local_db_path:
                conn.close()
        return "Search index rebuilt."

    def get_stats(self) -> dict:
        """返回搜索索引的统计信息。"""
        conn = self._conn()
        try:
            total = conn.execute(
                f"SELECT COUNT(*) FROM {_FTS_TABLE}"
            ).fetchone()[0]

            sessions = conn.execute(
                f"SELECT COUNT(DISTINCT session_id) FROM {_FTS_TABLE}"
            ).fetchone()[0]

            cwds = conn.execute(
                f"SELECT COUNT(DISTINCT cwd) FROM {_FTS_TABLE}"
            ).fetchone()[0]

            types_row = conn.execute(f"""
                SELECT entry_type, COUNT(*) AS cnt
                FROM {_FTS_TABLE}
                GROUP BY entry_type
                ORDER BY cnt DESC
            """).fetchall()

            type_counts = {r["entry_type"]: r["cnt"] for r in types_row}

            db_size = _get_db_path().stat().st_size if _get_db_path().exists() else 0

            return {
                "total_entries": total,
                "sessions_indexed": sessions,
                "working_directories": cwds,
                "by_type": type_counts,
                "db_size_bytes": db_size,
                "db_size_human": _format_size(db_size),
                "db_path": str(_get_db_path()),
            }
        finally:
            if self._local_db_path:
                conn.close()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读字符串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# 全局便捷实例
# ---------------------------------------------------------------------------

SEARCH_INDEX = SearchIndex()

