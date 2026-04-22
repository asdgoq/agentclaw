#!/usr/bin/env python3
"""诊断搜索问题：为什么搜 '工具' 没结果？"""
import os
import sqlite3
import sys

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.search import _get_db_path, _tokenize_for_search, _tokenize_for_index, _contains_cjk

db = _get_db_path()
conn = sqlite3.connect(str(db))

# 1. 总条目数
total = conn.execute("SELECT count(*) FROM session_fts").fetchone()[0]
print(f"FTS5 总条目数: {total}")

# 2. 看几条样本内容
print("\n=== 样本数据（前10条）===")
rows = conn.execute("SELECT entry_id, entry_type, role, substr(content,1,100) FROM session_fts LIMIT 10").fetchall()
for r in rows:
    print(f"  [{r[0]}] type={r[1]} role={r[2]} content={r[3]}")

# 3. 测试中文分词
print("\n=== 分词测试 ===")
print(f"  _contains_cjk('工具'): {_contains_cjk('工具')}")
print(f"  _tokenize_for_index('工具'): '{_tokenize_for_index('工具')}'")
print(f"  _tokenize_for_search('工具'): '{_tokenize_for_search('工具')}'")
print(f"  _tokenize_for_search('执行 Shell 命令'): '{_tokenize_for_search('执行 Shell 命令')}'")

# 4. 直接用 FTS5 MATCH 搜
print("\n=== 直接 SQL 测试 ===")
# 试几种查询方式
queries = [
    "工具",
    "工具*",
    '"工具"',
    _tokenize_for_search("工具"),
]
for q in queries:
    try:
        cnt = conn.execute(f"SELECT count(*) FROM session_fts WHERE session_fts MATCH ?", (q,)).fetchone()[0]
        print(f"  MATCH '{q}' -> {cnt} 条")
    except Exception as e:
        print(f"  MATCH '{q}' -> 错误: {e}")

# 5. 用 LIKE 看看有没有包含"工具"的内容
print("\n=== LIKE 搜索 ===")
like_rows = conn.execute("SELECT entry_id, substr(content,1,120) FROM session_fts WHERE content LIKE '%工具%' LIMIT 5").fetchall()
print(f"  LIKE '%工具%' -> {len(like_rows)} 条")
for r in like_rows:
    print(f"    [{r[0]}] {r[1]}")

# 6. 也看看英文 tool
print("\n=== 英文 LIKE 搜索 ===")
en_rows = conn.execute("SELECT entry_id, substr(content,1,120) FROM session_fts WHERE content LIKE '%tool%' OR content LIKE '%Tool%' LIMIT 5").fetchall()
print(f"  LIKE %tool% -> {len(en_rows)} 条")
for r in en_rows:
    print(f"    [{r[0]}] {r[1]}")

conn.close()

