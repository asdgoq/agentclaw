#!/usr/bin/env python3
"""
SQLite FTS5 Full-Text Search Index for Session Entries.

Architecture:
  - JSONL (session.py)  → primary storage (tree structure, append-only)
  - SQLite FTS5 (this)   → search index (content index, fast queries)

Why both?
  - JSONL is great for: tree ops, branching, compaction, human-readable
  - SQLite FTS5 is great for: cross-session search, keyword matching,
    ranking by relevance, prefix/phrase queries
  - jieba: Chinese text segmentation (FTS5's built-in tokenizer can't handle CJK)

Data flow:
  1. SessionManager.append_message() → writes to JSONL
  2. SearchIndex._tokenize_for_index() → jieba segments Chinese text
  3. SearchIndex.index_entry()       → inserts tokenized content into FTS5
  4. User searches                   → jieba segments query → FTS5 MATCH → ranked results

FTS5 table schema:
  CREATE VIRTUAL TABLE session_fts USING fts5(
      entry_id, session_id, session_file, entry_type, role,
      content, cwd, timestamp, parent_id,
      tokenize='unicode61'   -- basic Unicode; heavy lifting done by jieba in Python
  );

Chinese tokenization strategy:
  - On INDEX:  jieba.lcut(text) → join tokens with spaces → store in FTS5
  - On SEARCH: jieba.lcut(query) → join with AND → FTS5 MATCH expression
  - English text passes through unchanged (spaces already separate words)

DB location:
  <SESSIONS_DIR>/.search.db   (alongside the .jsonl files)
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union

from .config import SESSIONS_DIR

# Lazy-load jieba (only when actually used for Chinese tokenization)
_jieba = None

def _get_jieba():
    """Lazy-load jieba to avoid slow import at module level."""
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
# Constants
# ---------------------------------------------------------------------------

_SEARCH_DB_NAME = ".search.db"
_FTS_TABLE = "session_fts"

# Singleton connection with thread safety
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_db_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Database Connection Management
# ---------------------------------------------------------------------------


def _get_db_path() -> Path:
    """Get the path to the search database."""
    global _db_path
    if _db_path is None:
        _db_path = SESSIONS_DIR / _SEARCH_DB_NAME
    return _db_path


def _get_connection() -> sqlite3.Connection:
    """
    Get or create the SQLite connection (thread-safe singleton).
    Creates the FTS5 table on first use.
    """
    global _conn, _db_path
    with _lock:
        if _conn is None:
            db_path = _get_db_path()
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            _conn = sqlite3.connect(str(db_path))
            _conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrent read performance
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")

            # Create FTS5 virtual table if not exists
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
    """Close the search database connection."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


# ---------------------------------------------------------------------------
# Chinese Tokenization (jieba)
# ---------------------------------------------------------------------------


# Regex to detect CJK characters (Chinese/Japanese/Korean)
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')


def _contains_cjk(text: str) -> bool:
    """Check if text contains any CJK (Chinese) characters."""
    return bool(_CJK_RE.search(text))


def _tokenize_for_index(text: str) -> str:
    """
    Tokenize text for FTS5 indexing.

    Strategy:
      - If text contains CJK: use jieba to segment, join tokens with spaces.
        This lets FTS5 index each Chinese word as a separate token.
      - If text is English-only: return as-is (FTS5 handles space-separated words).
      - For mixed content: segment the whole thing with jieba.

    Examples:
      'SQLite FTS5 全文搜索' → 'SQLite FTS5 全文 搜索'
      '树形JSONL会话管理'   → '树形 JSONL 会话 管理'
      'hello world'         → 'hello world' (unchanged)
    """
    if not text:
        return text
    if not _contains_cjk(text):
        return text  # English only, no need to tokenize

    jb = _get_jieba()
    # Use cut_for_search mode for better recall in search scenarios
    tokens = jb.cut_for_search(text)
    # Filter: keep meaningful tokens (skip single punctuation, spaces)
    filtered = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
    return " ".join(filtered)


def _tokenize_for_search(query: str) -> str:
    """
    Tokenize a user's search query for FTS5 MATCH expression.

    Strategy:
      - If query contains CJK: use jieba to segment, join with AND.
        This ensures all Chinese terms must match (high precision).
      - If query looks like an advanced FTS5 expression (has AND/OR/NOT/"),
        pass it through unchanged (user knows what they're doing).
      - Simple English: pass through or wrap in quotes for phrase matching.

    Examples:
      '全文搜索'       → '全文 AND 搜索'
      '会话管理'       → '会话 AND 管理'
      'session AND branch' → 'session AND branch' (unchanged)
      'SQLite FTS5'     → '"SQLite FTS5"' (phrase)
    """
    if not query:
        return query

    # If query already contains FTS5 operators, respect user intent
    fts5_ops = re.compile(r'\b(AND|OR|NOT|NEAR)\b|[()*"]')
    if fts5_ops.search(query):
        # Still try to tokenize CJK parts within the expression
        if _contains_cjk(query):
            jb = _get_jieba()
            tokens = jb.lcut(query)
            and_joined = " AND ".join(f'"{t}"' if _contains_cjk(t) else t
                                   for t in tokens if t.strip())
            return and_joined
        return query

    # Simple query: tokenize and join with AND for precision
    if _contains_cjk(query):
        jb = _get_jieba()
        tokens = jb.lcut(query)
        # Join Chinese tokens with AND for precise matching
        and_joined = " AND ".join(f'"{t}"' if _contains_cjk(t) else t
                               for t in tokens if t.strip())
        return and_joined

    # Pure English simple query: phrase match
    return f'"{query}"'


# ---------------------------------------------------------------------------
# Content Extraction Helpers
# ---------------------------------------------------------------------------


def _extract_text_from_message(message: dict) -> tuple[str, str]:
    """
    Extract searchable text and role from a message dict.
    Returns (text_content, role).
    """
    role = message.get("role", "")
    content = message.get("content", "")

    if isinstance(content, str):
        return content, role
    elif isinstance(content, list):
        # Content blocks (e.g., Anthropic format)
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
    Extract searchable text and role from any entry type.
    Returns (text_content, role).
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
        # Fallback: dump entire entry as text (truncated)
        return json.dumps(entry, default=str)[:500], etype or "unknown"


# ---------------------------------------------------------------------------
# Index Operations
# ---------------------------------------------------------------------------


class SearchIndex:
    """
    Manages the SQLite FTS5 search index for session entries.

    Usage:
        idx = SearchIndex()
        idx.index_entry(entry, session_id, session_file, cwd)
        results = idx.search("keyword")
        idx.remove_session(session_id)   # cleanup on delete
    """

    def __init__(self, db_path: Union[str, Path] = None):
        self._local_db_path = db_path  # for testing; normally use singleton

    def _conn(self) -> sqlite3.Connection:
        if self._local_db_path:
            from pathlib import Path as _P
            _p = _P(self._local_db_path)
            _p.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(_p))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            # Ensure FTS5 table exists for local connections too
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
        Add or update an entry in the FTS5 index.
        Called automatically by SessionManager after each append.
        """
        eid = entry.get("id", "")
        if not eid:
            return

        etype = entry.get("type", "")
        content_text, role = _extract_searchable_text(entry)
        # Tokenize Chinese text with jieba before storing in FTS5
        content_text = _tokenize_for_index(content_text)
        timestamp = entry.get("timestamp", "")
        parent_id = entry.get("parentId") or ""

        conn = self._conn()
        try:
            # FTS5 uses INSERT OR REPLACE for upsert semantics
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
            pass  # non-critical: search index failure shouldn't break session
        finally:
            if self._local_db_path:
                conn.close()

    def index_entries(self, entries: list[dict], session_id: str,
                      session_file: Union[str, Path], cwd: str):
        """Bulk-index multiple entries (used for initial indexing)."""
        conn = self._conn()
        try:
            for entry in entries:
                eid = entry.get("id", "")
                if not eid or entry.get("type") == "session":
                    continue
                etype = entry.get("type", "")
                content_text, role = _extract_searchable_text(entry)
                # Tokenize Chinese text with jieba before storing in FTS5
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
        """Remove a single entry from the index."""
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
        """Remove all entries for a session from the index."""
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
    # Search Queries
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20,
               session_id: str = None, cwd: str = None,
               entry_type: str = None, role: str = None) -> list[dict]:
        """
        Full-text search across indexed session entries.

        Args:
            query: FTS5 search expression (supports AND, OR, NOT, phrase "*",
                   prefix "prefix*", NEAR operator)
            limit: max results to return
            session_id: filter to specific session
            cwd: filter to working directory
            entry_type: filter by entry type (message, compaction, etc.)
            role: filter by message role (user, assistant)

        Returns:
            List of result dicts with keys:
                entry_id, session_id, session_file, entry_type, role,
                content (snippet/highlighted), cwd, timestamp, parent_id,
                rank (relevance score, lower = more relevant)

        Example queries:
            "SQLite FTS5"           — phrase match
            "session AND branch"    — AND
            "session OR branch"     — OR
            "session NOT branch"    — NOT
            "compac*"               — prefix match
            'NEAR("session" "search")'  — proximity
        """
        conn = self._conn()
        try:
            # Build WHERE clause dynamically
            conditions = []
            params: list = []

            conditions.append(f"{_FTS_TABLE} MATCH ?")
            # Tokenize Chinese query with jieba for better CJK search
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

            # Use bm25() for ranking (built-in FTS5 function)
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
            # Common issue: invalid FTS5 query syntax
            if "fts5:" in str(e).lower():
                return [{"error": f"Invalid search syntax: {e}"}]
            raise
        finally:
            if self._local_db_path:
                conn.close()

    def search_sessions(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search across sessions, returning one best match per session.
        Useful for "which session was I talking about X?" queries.

        Note: FTS5's bm25() cannot be used in subqueries or GROUP BY context,
        so we use a two-step approach: get raw matches first, then aggregate in Python.
        """
        conn = self._conn()
        try:
            # Step 1: Get all matching entries with bm25 rank
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

            # Step 2: Aggregate by session_id in Python
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

            # Sort by best_rank, then match_count, and limit
            results = sorted(
                session_map.values(),
                key=lambda x: (x["best_rank"], -x["match_count"]),
            )[:limit]

            # Convert sets to strings for output
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
    # Maintenance
    # ------------------------------------------------------------------

    def rebuild_index(self, session_manager=None):
        """
        Rebuild the entire search index from JSONL files.
        If session_manager is given, only re-index that session.
        Otherwise, re-index ALL sessions.
        """
        conn = self._conn()
        try:
            if session_manager:
                # Re-index single session
                self.remove_session(session_manager.session_id)
                entries = session_manager.get_entries()
                self.index_entries(
                    entries,
                    session_manager.session_id,
                    session_manager.session_file or "",
                    session_manager.cwd,
                )
            else:
                # Re-index everything: clear + scan all JSONL files
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
        """Return statistics about the search index."""
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
# Utilities
# ---------------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Global convenience instance
# ---------------------------------------------------------------------------

SEARCH_INDEX = SearchIndex()

