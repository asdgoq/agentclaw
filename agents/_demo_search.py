#!/usr/bin/env python3
"""
模拟 s_full.py REPL 中的 /session_search 命令实际输入输出
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from agents.data.search import close_search_db, _get_db_path
from agents.core.session import SessionManager

# 清理
close_search_db()
dp = _get_db_path()
if dp.exists():
    os.remove(dp)

# 模拟用户之前已经聊过的 session（有真实对话内容）
sm = SessionManager.in_memory(cwd="/Users/guochengao/project/learn-claude-code")
sm.append_message({"role": "user", "content": "hello"})
sm.append_message({"role": "assistant", "content": "你好！我是你的 coding agent。有什么可以帮你的？"})
sm.append_message({"role": "user", "content": "你的工具有哪些"})
sm.append_message({"role": "assistant", "content": (
    "我有这些工具：\n"
    "- bash: 执行 shell 命令\n"
    "- read_file / write_file / edit_file: 文件操作\n"
    "- TodoWrite: 任务追踪\n"
    "- task: 启动子 agent\n"
    "- compress: 压缩上下文\n"
    "- session_* 系列: 会话管理（list/switch/history/branch/search）\n"
    "- load_skill: 加载技能知识"
)})
sm.append_message({"role": "user", "content": "session branch咋用啊"})
sm.append_message({"role": "assistant", "content": (
    "session_branch 用于时间旅行！用法：\n"
    "1. /session_history 查看对话树\n"
    "2. /session_branch <entry_id> 跳到那个节点\n"
    "3. 之后发的消息就是新分支了（像 git checkout）\n"
    "零拷贝操作，原来的分支不受影响。"
)})
sm.append_message({"role": "user", "content": "给我看下 history 的树形结构"})
sm.append_message({"role": "assistant", "content": (
    "当前 session 历史：\n"
    "  e001 [user] hello\n"
    "    └─ e002 [assistant] 你好！...\n"
    "       └─ e003 [user] 你的工具有哪些\n"
    "          └─ e004 [assistant] 我有这些工具：...\n"
    "             └─ e005 [user] session branch咋用啊\n"
    "                └─ e006 [assistant] session_branch 用于时间旅行！...\n"
    "                   └─ ◆ e007 [user] 给我看下 history..."
)})
sm.append_message({"role": "user", "content": "jsonl"})
sm.append_message({"role": "assistant", "content": "JSONL (JSON Lines) 是我们的主存储格式，每行一个 JSON 对象。配合 FTS5 索引可以实现高效搜索。"})
sm.append_message({"role": "user", "content": "加上 SQLite FTS5 搜索"})
sm.append_message({"role": "assistant", "content": "好的，已经集成了 SQLite FTS5 全文搜索索引。现在可以用 /session_search 命令搜索所有历史会话了。"})

print("=" * 65)
print("  模拟 s_full >> REPL — 实际输入输出演示")
print("=" * 65)

# ================================================================
print("\n\033[36ms_full >> \033[0m\033[100m/session_search\033[0m")
print()
# 模拟 _do_session_search() 无参数 → 显示帮助
help_text = """Session Search (FTS5)
Usage:
  /session_search <query>           — search current session
  /session_search global <query>     — search ALL sessions
  /session_search stats              — show index stats
  /session_search rebuild            — rebuild index from JSONL files

Query syntax examples:
  'SQLite AND FTS5'       AND query
  'session OR branch'     OR query
  'compac*'             prefix wildcard
  '"full text"'        phrase match
  'session NOT branch'    NOT query"""
print(help_text)

# ================================================================
print("\033[36ms_full >> \033[0m\033[100m/session_search stats\033[0m")
print()
stats = SessionManager.search_stats()
lines = [
    "┌─────────────────────────────────────┐",
    "│     Search Index Statistics (FTS5)  │",
    "└─────────────────────────────────────┘",
    "",
    f"  Total entries:    {stats.get('total_entries', 0)}",
    f"  Sessions indexed: {stats.get('sessions_indexed', 0)}",
    f"  Work dirs:        {stats.get('working_directories', 0)}",
    f"  Database size:    {stats.get('db_size_human', '?')}",
    f"  DB path:          {stats.get('db_path', '?')}",
    "",
    "  By type:",
]
for t, c in stats.get("by_type", {}).items():
    lines.append(f"    {t}: {c}")
print("\n".join(lines))

# ================================================================
print(f"\n\033[36ms_full >> \033[0m\033[100m/session_search 工具\033[0m")
print()
results = sm.search("工具")
lines2 = [
    f"Search results (current session: {sm.session_id[:12]}):",
    f"Query: 工具",
    f"Matches: {len(results)}",
    "",
]
for r in results:
    eid = r.get("entry_id", "?")
    etype = r.get("entry_type", "?")
    role = r.get("role", "")
    snippet = r.get("snippet", "") or r.get("content", "(no content)")[:120]
    ts = r.get("timestamp", "?")[11:19]
    rank = r.get("rank", "?")
    marker = "◆" if eid == sm.leaf_id else " "
    lines2.append(f"  {marker} \033[33m{eid}\033[0m [{etype}/{role}] rank={rank} {ts}")
    for snip_line in snippet.split("\n")[:3]:
        lines2.append(f"      {snip_line.strip()[:100]}")
    lines2.append("")
lines2.append("Tip: use session_switch with entry context, or session_branch <entry_id> to go back.")
print("\n".join(lines2))

# ================================================================
print(f"\n\033[36ms_full >> \033[0m\033[100m/session_search session AND branch\033[0m")
print()
results2 = sm.search("session AND branch")
lines3 = [
    f"Search results (current session: {sm.session_id[:12]}):",
    f"Query: session AND branch",
    f"Matches: {len(results2)}",
    "",
]
for r in results2:
    eid = r.get("entry_id", "?")
    etype = r.get("entry_type", "?")
    role = r.get("role", "")
    snippet = r.get("snippet", "") or "(no content)"[:120]
    ts = r.get("timestamp", "?")[11:19]
    rank = r.get("rank", "?")
    marker = "◆" if eid == sm.leaf_id else " "
    lines3.append(f"  {marker} \033[33m{eid}\033[0m [{etype}/{role}] rank={rank} {ts}")
    for snip_line in snippet.split("\n")[:2]:
        lines3.append(f"      {snip_line.strip()[:100]}")
    lines3.append("")
print("\n".join(lines3))

# ================================================================
print(f"\n\033[36ms_full >> \033[0m\033[100m/session_search global FTS5\033[0m")
print()
gr = SessionManager.global_search("FTS5")
entries = gr.get("entries", [])
sessions = gr.get("sessions", [])
lines4 = [
    "╔══════════════════════════════════════════════╗",
    "║     Global Search Results (All Sessions)       ║",
    "╚══════════════════════════════════════════════╝",
    "",
]
if sessions:
    lines4.append(f"--- Matching Sessions ({len(sessions)}) ---")
    for s in sessions:
        sid = s.get("session_id", "?")[:12]
        count = s.get("match_count", 0)
        best_rank = s.get("best_rank", "?")
        cwd = s.get("cwd", "")[-40:]
        types = s.get("types", "")
        lines4.append(f"  \033[33m{sid}\033[0m  matches={count}  best_rank={best_rank}  cwd=...{cwd}  types={types}")
    lines4.append("")
if entries:
    show_entries = entries[:10]
    lines4.append(f"--- Top Entries ({len(show_entries)}) ---")
    for r in show_entries:
        eid = r.get("entry_id", "?")
        sid = r.get("session_id", "?")[:12]
        etype = r.get("entry_type", "?")
        role = r.get("role", "")
        snippet = r.get("snippet", "") or "(no content)"
        ts = r.get("timestamp", "?")[11:19]
        rank = r.get("rank", "?")
        lines4.append(f"  \033[33m{eid}\033[0m (sess:{sid}) [{etype}/{role}] rank={rank} {ts}")
        for snip_line in snippet.split("\n")[:2]:
            lines4.append(f"      {snip_line.strip()[:100]}")
        lines4.append("")
    lines4.append("Tip: use session_switch <session_id_prefix> to switch to a matching session.")
print("\n".join(lines4))

# ================================================================
print(f"\n\033[36ms_full >> \033[0m\033[100m/session_search \"JSONL\"\033[0m")
print()
results3 = sm.search('"JSONL"')
lines5 = [
    f"Search results (current session: {sm.session_id[:12]}):",
    f'Query: "JSONL"',
    f"Matches: {len(results3)}",
    "",
]
for r in results3:
    eid = r.get("entry_id", "?")
    etype = r.get("entry_type", "?")
    role = r.get("role", "")
    snippet = r.get("snippet", "") or "(no content)"[:120]
    ts = r.get("timestamp", "?")[11:19]
    rank = r.get("rank", "?")
    marker = "◆" if eid == sm.leaf_id else " "
    lines5.append(f"  {marker} \033[33m{eid}\033[0m [{etype}/{role}] rank={rank} {ts}")
    for snip_line in snippet.split("\n")[:2]:
        lines5.append(f"      {snip_line.strip()[:100]}")
    lines5.append("")
print("\n".join(lines5))

close_search_db()
if dp.exists():
    os.remove(dp)

print("\n\033[36m=== REPL 演示结束 ===\033[0m\n")

