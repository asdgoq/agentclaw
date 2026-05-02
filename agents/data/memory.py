#!/usr/bin/env python3
"""Long-term memory bank (JSONL-backed, user-driven).

用户通过工具显式控制记忆的增删查，不是 LLM 自动决定的。

存储：WORKDIR/.memory.jsonl（追加 only）
格式：每行一条 {"id", "ts", "category", "content"}

用法：
    mem = MemoryBank()
    mem.remember("preference", "我喜欢 Rust 不喜欢 Go")   # 存
    mem.list_all()                                         # 查全部
    mem.delete("a1b2c3d4")                                 # 删
    mem.recall_for_prompt()                                # 给 system prompt 用
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from ..core.config import WORKDIR

MEMORY_FILE = WORKDIR / ".memory.jsonl"

# 用户可选的分类（提示用，不强制）
CATEGORIES = [
    "preference",   # 偏好（语言、风格、习惯）
    "fact",         # 事实（环境信息、账号名等）
    "decision",     # 决策（技术选型、架构选择）
    "habit",        # 习惯（工作时间、review 风格）
    "context",      # 上下文（项目背景、业务知识）
    "people",       # 人物（同事分工、联系人）
]


class MemoryBank:
    """长期记忆库 — 追加-only 的 JSONL 存储。"""

    def __init__(self, filepath: Path = None):
        self._filepath = filepath or MEMORY_FILE
        self._entries: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load(self):
        """从 JSONL 加载所有记忆到内存。"""
        if not self._filepath.exists():
            return
        try:
            for line in self._filepath.read_text().strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                eid = entry.get("id")
                if eid:
                    self._entries.append(entry)
                    self._by_id[eid] = entry
        except (json.JSONDecodeError, OSError):
            pass  # 文件损坏时静默忽略

    def _append(self, entry: dict) -> str:
        """写入文件 + 更新索引。"""
        eid = entry["id"]
        self._entries.append(entry)
        self._by_id[eid] = entry
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return eid

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def remember(self, category: str, content: str) -> str:
        """记录一条记忆。返回新记忆的 ID。"""
        entry = {
            "id": uuid.uuid4().hex[:8],
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "category": category.strip(),
            "content": content.strip(),
        }
        self._append(entry)
        return f"已记住 [{entry['category']}] {entry['content'][:60]} (ID: {entry['id']})"

    def delete(self, memory_id: str) -> str:
        """按 ID 删除一条记忆（重写文件排除该条）。"""
        memory_id = memory_id.strip()
        if memory_id not in self._by_id:
            return f"错误：记忆 ID '{memory_id}' 不存在。先用 memory_list 查看。"

        # 从内存移除
        self._entries = [e for e in self._entries if e["id"] != memory_id]
        del self._by_id[memory_id]

        # 重写整个文件（JSONL 追加-only 架构下删除只能重写）
        with open(self._filepath, "w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        return f"已删除记忆 {memory_id}。"

    def list_all(self, category: str = None, limit: int = 50) -> str:
        """列出记忆，支持按分类过滤。"""
        entries = self._entries
        if category:
            cat = category.strip().lower()
            entries = [e for e in entries if e.get("category", "").lower() == cat]

        if not entries:
            hint = f"（分类 '{category}' 下）" if category else ""
            return f"没有记忆条目{hint}。使用 memory_record 来记录。"

        # 显示最近的 limit 条
        show = entries[-limit:] if len(entries) > limit else entries

        lines = [f"共有 {len(self._entries)} 条记忆{f'（显示最近 {len(show)} 条）' if len(self._entries) > limit else ''}：", ""]
        lines.append(f"  {'ID':<10} {'时间':<18} {'分类':<14} 内容")
        lines.append(f"  {'─'*10} {'─'*18} {'─'*14} {'─'*40}")

        for e in show:
            eid = e["id"]
            ts = e["ts"][:16].replace("T", " ")
            cat = e.get("category", "?")
            content = e["content"].replace("\n", " ")[:45]
            marker = " ◆" if e == show[-1] and len(show) == len(entries) else ""
            lines.append(f"  {eid:<10} {ts:<18} {cat:<14} {content}{marker}")

        # 可用分类提示
        cats = sorted(set(e.get("category", "?") for e in self._entries))
        lines.append("")
        lines.append(f"可用分类：{', '.join(cats)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM 集成
    # ------------------------------------------------------------------

    def recall_for_prompt(self, max_entries: int = 15) -> str:
        """生成注入到 system prompt 的记忆文本。返回空字符串表示无记忆。"""
        if not self._entries:
            return ""

        recent = self._entries[-max_entries:]
        lines = ["<long_term_memory>"]
        for e in recent:
            lines.append(f"[{e['ts'][:10]}][{e['category']}] {e['content']}")
        lines.append("</long_term_memory>")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def categories(self) -> list[str]:
        return sorted(set(e.get("category", "?") for e in self._entries))

