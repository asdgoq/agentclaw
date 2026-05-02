#!/usr/bin/env python3
"""自学习引擎 (s13) — Agent 自主经验积累。

架构:
  - 两层检测：基于规则（零 token 开销）+ LLM 驱动（通过 prompt 规则）
  - 存储：.learnings/ 目录下的 3 个 Markdown 文件（人类可读）
  - 注入：recall_for_prompt() 将高优先级条目注入系统 prompt
  - 升级：反复出现的条目（次数>=3）可升级到 MemoryBank 或 Skill

存储布局:
    .learnings/
    ├── LEARNINGS.md       # 修正、知识缺口、最佳实践
    ├── ERRORS.md          # 工具错误、命令失败
    └── FEATURE_REQUESTS.md # 能力缺口、用户需求

用法:
    lrn = LearningEngine()
    lrn.auto_capture(tool_name, tool_input, output)   # 基于规则的自动捕获
    lrn.record(category, summary, details, ...)        # 显式记录（LLM 调用此方法）
    text = lrn.recall_for_prompt()                      # 注入到系统 prompt

REPL 命令: /learnings [/list|/stats|/search|/promote]
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.config import WORKDIR

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

LEARNINGS_DIR = WORKDIR / ".learnings"

LEARNINGS_FILE = LEARNINGS_DIR / "LEARNINGS.md"
ERRORS_FILE = LEARNINGS_DIR / "ERRORS.md"
FEATURES_FILE = LEARNINGS_DIR / "FEATURE_REQUESTS.md"

# 条目类型 → 文件映射
TYPE_FILES = {
    "learning": LEARNINGS_FILE,
    "error": ERRORS_FILE,
    "feature": FEATURES_FILE,
}

# 各类型的有效分类
LEARNING_CATEGORIES = ("correction", "knowledge_gap", "best_practice")
ERROR_CATEGORIES = ("tool_error", "command_error", "lint_error", "runtime_error")
FEATURE_CATEGORIES = ("capability_gap", "workflow_request", "integration_request")

ALL_CATEGORIES = LEARNING_CATEGORIES + ERROR_CATEGORIES + FEATURE_CATEGORIES

PRIORITIES = ("low", "medium", "high", "critical")
STATUSES = ("pending", "in_progress", "resolved", "wont_fix",
            "promoted", "promoted_to_memory", "promoted_to_skill")
AREAS = ("frontend", "backend", "infra", "tests", "docs", "config", "tooling", "general")

# ---------------------------------------------------------------------------
# 基于规则的错误检测模式（零 token 开销，纯字符串匹配）
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"Error:", re.IGNORECASE),
    re.compile(r"ERROR\b"),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\bFAILED\b"),
    re.compile(r"Permission denied"),
    re.compile(r"No such file or directory"),
    re.compile(r"command not found"),
    re.compile(r"SyntaxError"),
    re.compile(r"TypeError"),
    re.compile(r"KeyError"),
    re.compile(r"ValueError"),
    re.compile(r"ImportError"),
    re.compile(r"ModuleNotFoundError"),
    re.compile(r"FileNotFoundError"),
    re.compile(r"PermissionError"),
    re.compile(r"ConnectionRefusedError|ConnectionError|TimeoutError"),
    re.compile(r"exit code [1-9]"),
    re.compile(r"non-zero exit code"),
    re.compile(r"错误："),           # Chinese error prefix used in existing code
    re.compile(r"不存在"),
    re.compile(r"失败"),
]

# 常见产生错误、需要自动捕获的工具
_ERROR_PRONE_TOOLS = {
    "bash", "read_file", "write_file", "edit_file",
    "git_commit", "git_push", "git_branch_create",
    "background_run",
}


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class LearningEntry:
    """单条学习经验条目。"""
    entry_id: str = ""
    entry_type: str = "learning"          # learning | error | feature
    category: str = "correction"
    priority: str = "medium"
    status: str = "pending"
    area: str = "general"
    summary: str = ""
    details: str = ""
    suggested_action: str = ""
    source: str = "auto"                  # auto | user_feedback | tool_error | self_reflection | llm_judgment
    related_files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    recurrence_count: int = 1
    created_at: str = ""
    resolved_at: str = ""

    def __post_init__(self):
        if not self.entry_id:
            prefix = "LRN" if self.entry_type == "learning" else \
                     "ERR" if self.entry_type == "error" else "FEAT"
            ts = datetime.now().strftime("%Y%m%d")
            self.entry_id = f"{prefix}-{ts}-{uuid.uuid4().hex[:6].upper()}"
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Markdown 序列化 / 反序列化
# ---------------------------------------------------------------------------

def _entry_to_markdown(entry: LearningEntry) -> str:
    """将 LearningEntry 转换为 Markdown 表示。"""
    lines = [
        f"## [{entry.entry_id}] {entry.category}",
        "",
        f"**Logged**: {entry.created_at}",
        f"**Priority**: {entry.priority}",
        f"**Status**: {entry.status}",
        f"**Area**: {entry.area}",
        "",
        "### Summary",
        entry.summary,
        "",
        "### Details",
        entry.details,
        "",
        "### Suggested Action",
        entry.suggested_action,
        "",
        "### Metadata",
        f"- Source: {entry.source}",
        f"- Related Files: {', '.join(entry.related_files) if entry.related_files else 'none'}",
        f"- Tags: {', '.join(entry.tags) if entry.tags else 'none'}",
        f"- Recurrence-Count: {entry.recurrence_count}",
    ]
    if entry.resolved_at:
        lines.extend([
            "",
            "### Resolution",
            f"- **Resolved**: {entry.resolved_at}",
        ])
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _parse_entries(markdown_text: str) -> list[LearningEntry]:
    """将 Markdown 文件解析回 LearningEntry 对象。

    健壮地处理 ## [ID] category 格式。
    """

    entries: list[LearningEntry] = []
    if not markdown_text.strip():
        return entries

    # 按条目分隔符拆分
    blocks = re.split(r"^---$", markdown_text, flags=re.MULTILINE)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        entry = LearningEntry()

        # 解析头部: ## [ID] category
        header_match = re.match(r"^##\s+\[(\w+(?:-\d{8}-\w+)?)\]\s*(.+)", block)
        if header_match:
            entry.entry_id = header_match.group(1)
            raw_category = header_match.group(2).strip()

            # 从 ID 前缀确定 entry_type
            if entry.entry_id.startswith("ERR"):
                entry.entry_type = "error"
            elif entry.entry_id.startswith("FEAT"):
                entry.entry_type = "feature"
            else:
                entry.entry_type = "learning"

            # 将原始分类映射到有效分类
            if raw_category in ALL_CATEGORIES:
                entry.category = raw_category
            elif raw_category in ("operation_name",):
                entry.category = "tool_error"

        # 使用正则模式解析字段
        logged_match = re.search(r'\*\*Logged\*\*:\s*(.+)', block)
        if logged_match:
            entry.created_at = logged_match.group(1).strip()

        priority_match = re.search(r'\*\*Priority\*\*:\s*(.+)', block)
        if priority_match:
            entry.priority = priority_match.group(1).strip()

        status_match = re.search(r'\*\*Status\*\*:\s*(.+)', block)
        if status_match:
            entry.status = status_match.group(1).strip()

        area_match = re.search(r'\*\*Area\*\*:\s*(.+)', block)
        if area_match:
            entry.area = area_match.group(1).strip()

        # 摘要: ### Summary 之后的第一行
        summary_match = re.search(r'### Summary\s*\n(.+?)(?=\n###|\Z)', block, re.DOTALL)
        if summary_match:
            entry.summary = summary_match.group(1).strip()

        # 详情
        details_match = re.search(r'### Details\s*\n(.+?)(?=\n### Suggested|\Z)', block, re.DOTALL)
        if details_match:
            entry.details = details_match.group(1).strip()

        # 建议操作
        action_match = re.search(r'### Suggested Action\s*\n(.+?)(?=\n### Metadata|\Z)', block, re.DOTALL)
        if action_match:
            entry.suggested_action = action_match.group(1).strip()

        # 来源
        source_match = re.search(r'- Source:\s*(.+)', block)
        if source_match:
            entry.source = source_match.group(1).strip()

        # 相关文件
        rf_match = re.search(r'- Related Files:\s*(.+)', block)
        if rf_match:
            raw_rf = rf_match.group(1).strip()
            entry.related_files = [f.strip() for f in raw_rf.split(",") if f.strip() != "none"]

        # 标签
        tags_match = re.search(r'- Tags:\s*(.+)', block)
        if tags_match:
            raw_tags = tags_match.group(1).strip()
            entry.tags = [t.strip() for t in raw_tags.split(",") if t.strip() != "none"]

        # 重复次数
        rc_match = re.search(r'- Recurrence-Count:\s*(\d+)', block)
        if rc_match:
            entry.recurrence_count = int(rc_match.group(1))

        # 解决时间戳
        resolved_match = re.search(r'- \*\*Resolved\*\*:\s*(.+)', block)
        if resolved_match:
            entry.resolved_at = resolved_match.group(1).strip()

        # 至少获取到 ID 才添加
        if entry.entry_id:
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# 核心引擎
# ---------------------------------------------------------------------------

class LearningEngine:
    """自学习引擎——从 Agent 交互中积累经验。

    两种输入路径:
      1. auto_capture()   — 基于规则，在 agent_loop 中工具执行后调用
      2. record()         — 显式调用，由 LLM 通过 record_learning 工具调用
    """

    def __init__(self, learnings_dir: Path = None):
        self._dir = learnings_dir or LEARNINGS_DIR
        self._entries: list[LearningEntry] = []
        self._by_id: dict[str, LearningEntry] = {}
        self._init_storage()
        self._load_all()

    # ------------------------------------------------------------------
    # 存储初始化与 I/O
    # ------------------------------------------------------------------

    def _init_storage(self):
        """确保 .learnings/ 目录和模板文件存在。"""
        self._dir.mkdir(exist_ok=True)
        for filepath in TYPE_FILES.values():
            if not filepath.exists():
                filepath.write_text("", encoding="utf-8")

    def _load_all(self):
        """从三个 Markdown 文件中加载所有条目。"""
        for etype, filepath in TYPE_FILES.items():
            if filepath.exists():
                text = filepath.read_text(encoding="utf-8").strip()
                entries = _parse_entries(text)
                for e in entries:
                    e.entry_type = etype
                    self._entries.append(e)
                    self._by_id[e.entry_id] = e

    def _append_to_file(self, entry: LearningEntry):
        """将单条条目的 Markdown 追加到对应文件。"""
        filepath = TYPE_FILES.get(entry.entry_type, LEARNINGS_FILE)
        md = _entry_to_markdown(entry)
        with open(filepath, "a", encoding="utf-") as f:
            f.write(md)

    def _rewrite_file(self, entry_type: str):
        """用当前内存中的条目重写指定文件（用于删除/更新）。"""
        filepath = TYPE_FILES.get(entry_type, LEARNINGS_FILE)
        matching = [e for e in self._entries if e.entry_type == entry_type]
        if not matching:
            filepath.write_text("", encoding="utf-8")
            return
        parts = [_entry_to_markdown(e) for e in matching]
        filepath.write_text("\n".join(parts), encoding="utf-8")

    # ------------------------------------------------------------------
    # 第一层：基于规则的自动检测（零 token 开销）
    # ------------------------------------------------------------------

    def auto_capture(self, tool_name: str, tool_input: dict,
                     output: str, area: str = "general") -> Optional[LearningEntry]:
        """从工具执行结果中自动检测错误信号。

        使用纯字符串/模式匹配——无需 LLM 调用。
        如果检测到信号则返回捕获的条目，否则返回 None。
        在 agent_loop 中每次工具执行后同步调用。
        """
        output_str = str(output)

        # 快速跳过：不是易错工具或输出看起来正常
        if tool_name not in _ERROR_PRONE_TOOLS:
            # 仍然检查任何工具输出中的通用错误模式
            pass

        # 与错误模式进行匹配
        matched_patterns = []
        for pat in _ERROR_PATTERNS:
            if pat.search(output_str):
                matched_patterns.append(pat.pattern[:60])

        # 同时检查现有代码库中使用的中文错误前缀
        is_error_like = (
            len(matched_patterns) > 0
            or output_str.startswith("错误：")
            or output_str.startswith("错误:")
            or (tool_name == "bash" and _has_nonzero_exit(output_str))
        )

        if not is_error_like:
            return None

        # 根据工具确定分类
        if tool_name == "bash":
            category = "command_error"
        elif "lint" in tool_name or "Lint" in output_str:
            category = "lint_error"
        elif "Traceback" in output_str or re.search(r"(TypeError|KeyError|ValueError|SyntaxError)\b", output_str):
            category = "runtime_error"
        else:
            category = "tool_error"

        # 从输出构建摘要（截断到可读长度）
        summary = self._extract_error_summary(output_str, tool_name)

        # 创建条目
        entry = LearningEntry(
            entry_type="error",
            category=category,
            priority=self._infer_priority(output_str),
            status="pending",
            area=area,
            summary=summary,
            details=output_str[:2000],  # cap detail size
            suggested_action=self._suggest_fix(tool_name, output_str),
            source="auto",
            tags=[tool_name, category],
            created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # 去重检查：最近是否记录过类似错误？
        existing = self._find_similar(entry)
        if existing is not None:
            # 增加重复次数而不是创建新条目
            existing.recurrence_count += 1
            existing.created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # refresh timestamp
            self._rewrite_file(existing.entry_type)
            return existing

        # 持久化
        self._append_to_file(entry)
        self._entries.append(entry)
        self._by_id[entry.entry_id] = entry
        return entry

    # ------------------------------------------------------------------
    # 第二层：显式记录（由 LLM 通过工具调用）
    # ------------------------------------------------------------------

    def record(self, *,
               entry_type: str = "learning",
               category: str = "correction",
               priority: str = "medium",
               area: str = "general",
               summary: str = "",
               details: str = "",
               suggested_action: str = "",
               source: str = "llm_judgment",
               related_files: Optional[list[str]] = None,
               tags: Optional[list[str]] = None,
               ) -> str:
        """显式记录一条学习经验。由 LLM 通过 record_learning 工具调用。

        返回包含新条目 ID 的结果消息。
        """
        # 验证
        if entry_type not in TYPE_FILES:
            return f"无效的 entry_type: '{entry_type}'，可选: {list(TYPE_FILES.keys())}"
        if entry_type == "learning" and category not in LEARNING_CATEGORIES:
            return f"无效的 learning category: '{category}'，可选: {LEARNING_CATEGORIES}"
        if entry_type == "error" and category not in ERROR_CATEGORIES:
            return f"无效的 error category: '{category}'，可选: {ERROR_CATEGORIES}"
        if entry_type == "feature" and category not in FEATURE_CATEGORIES:
            return f"无效的 feature category: '{category}'，可选: {FEATURE_CATEGORIES}"

        entry = LearningEntry(
            entry_type=entry_type,
            category=category,
            priority=priority,
            area=area,
            summary=summary[:200],
            details=details[:2000],
            suggested_action=suggested_action[:500],
            source=source,
            related_files=related_files or [],
            tags=tags or [],
        )

        # 去重
        existing = self._find_similar(entry)
        if existing is not None:
            existing.recurrence_count += 1
            existing.created_at = entry.created_at
            self._rewrite_file(existing.entry_type)
            return f"[去重合并] 相似经验已存在，recurrence_count++ → {existing.entry_id} (第{existing.recurrence_count}次)"

        self._append_to_file(entry)
        self._entries.append(entry)
        self._by_id[entry.entry_id] = entry
        return f"[已记录] {entry.entry_id} | [{category}] {summary[:60]}"

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def list_all(self, entry_type: str = None, category: str = None,
                 status: str = None, limit: int = 50) -> str:
        """列出学习条目，支持可选过滤。"""
        results = self._entries

        if entry_type:
            results = [e for e in results if e.entry_type == entry_type]
        if category:
            results = [e for e in results if e.category == category]
        if status:
            results = [e for e in results if e.status == status]

        if not results:
            hint = f" (type={entry_type}, category={category}, status={status})" if (entry_type or category or status) else ""
            return f"没有学习记录{hint}。"

        # 按最近优先显示
        results = sorted(results, key=lambda e: e.created_at, reverse=True)
        show = results[:limit]

        lines = [
            f"学习经验库（共 {len(self._entries)} 条，显示最近 {len(show)} 条）：",
            "",
            f"  {'ID':<24} {'类型':<8} {'分类':<16} {'优先级':<8} {'状态':<18} {'摘要'}",
            f"  {'─'*24} {'─'*8} {'─'*16} {'─'*8} {'─'*18} {'─'*40}",
        ]
        for e in show:
            eid_short = e.entry_id
            summary_display = e.summary.replace("\n", " ")[:42]
            rc_marker = f" ({e.recurrence_count}x)" if e.recurrence_count > 1 else ""
            lines.append(
                f"  {eid_short:<24} {e.entry_type:<8} {e.category:<16} "
                f"{e.priority:<8} {e.status:<18} {summary_display}{rc_marker}"
            )

        # 按类型统计
        type_counts: dict[str, int] = {}
        for e in self._entries:
            type_counts[e.entry_type] = type_counts.get(e.entry_type, 0) + 1
        lines.append("")
        lines.append(f"按类型: {dict(type_counts)}")

        # 待处理的高优/严重条目数
        pending_high = sum(1 for e in self._entries
                           if e.status in ("pending", "in_progress")
                           and e.priority in ("high", "critical"))
        if pending_high > 0:
            lines.append(f"⚠ 待处理高优条目: {pending_high}")

        return "\n".join(lines)

    def search(self, query: str, limit: int = 20) -> str:
        """按关键词搜索条目（不区分大小写的子串匹配）。"""
        q = query.lower().strip()
        if not q:
            return "用法：search_learnings <关键词>"

        matched = []
        for e in self._entries:
            searchable = f"{e.summary} {e.details} {e.suggested_action} {' '.join(e.tags)}".lower()
            if q in searchable:
                matched.append(e)

        if not matched:
            return f"未找到匹配 '{query}' 的学习记录。"

        matched = sorted(matched, key=lambda e: e.created_at, reverse=True)[:limit]
        lines = [
            f"搜索 '{query}'：找到 {len(matched)} 条结果",
            "",
        ]
        for e in matched:
            lines.append(f"  [{e.entry_id}] ({e.category}/{e.priority}) {e.summary[:80]}")
            if e.suggested_action:
                lines.append(f"    建议: {e.suggested_action[:70]}")
            lines.append("")

        return "\n".join(lines)

    def get_entry(self, entry_id: str) -> Optional[LearningEntry]:
        """按 ID 获取单条条目。"""
        return self._by_id.get(entry_id.strip())

    def update_status(self, entry_id: str, status: str) -> str:
        """更新条目的状态。"""
        entry = self._by_id.get(entry_id.strip())
        if not entry:
            return f"错误：'{entry_id}' 不存在。"
        if status not in STATUSES:
            return f"无效状态：'{status}'，可选: {STATUSES}"
        old_status = entry.status
        entry.status = status
        if status.startswith("resolved") or status.startswith("promoted") or status == "wont_fix":
            entry.resolved_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._rewrite_file(entry.entry_type)
        return f"已更新 {entry_id}: {old_status} → {status}"

    def delete(self, entry_id: str) -> str:
        """按 ID 删除条目。"""
        entry_id = entry_id.strip()
        entry = self._by_id.get(entry_id)
        if not entry:
            return f"错误：'{entry_id}' 不存在。"
        etype = entry.entry_type
        self._entries = [e for e in self._entries if e.entry_id != entry_id]
        del self._by_id[entry_id]
        self._rewrite_file(etype)
        return f"已删除 {entry_id}。"

    # ------------------------------------------------------------------
    # Prompt 注入——核心价值：将经验反馈到 LLM 上下文
    # ------------------------------------------------------------------

    def recall_for_prompt(self,
                          max_entries: int = 15,
                          max_chars: int = 4000,
                          include_resolved: bool = False) -> str:
        """生成注入到系统 prompt 的文本。

        策略:
          1. 过滤掉已解决/已升级的条目（已处理过）
          2. 按优先级排序: critical > high > medium，然后按时间排序
          3. 在字符预算内取前 N 条
          4. 格式化为紧凑的 <past_learnings> 块
        """
        if not self._entries:
            return ""

        # 过滤
        candidates = [e for e in self._entries if e.status in ("pending", "in_progress")]
        if include_resolved:
            candidates = [e for e in self._entries if e.status != "wont_fix"]

        if not candidates:
            return ""

        # 排序：先按优先级，再按重复次数（越高越重要），最后按时间
        _prio_rank = {p: i for i, p in enumerate(reversed(PRIORITIES))}
        candidates.sort(key=lambda e: (
            _prio_rank.get(e.priority, 99),
            -e.recurrence_count,
            e.created_at,
        ), reverse=False)

        # 在预算内构建紧凑输出
        lines = ["<past_learnings>"]
        chars = len(lines[0]) + len("</past_learnings>") + 1

        for e in candidates[:max_entries * 2]:  # take more candidates, trim by char budget
            # 每条条目的紧凑单行格式
            rc = f"({e.recurrence_count}x)" if e.recurrence_count > 1 else ""
            line = f"- [{e.entry_id}] [{e.category}/{e.priority}] {e.summary}"
            if e.suggested_action:
                line += f" → {e.suggested_action}"
            line += f" {rc}"

            if chars + len(line) > max_chars:
                break
            lines.append(line)
            chars += len(line) + 1

        lines.append("</past_learnings>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 升级：learning → memory / skill
    # ------------------------------------------------------------------

    def stats(self) -> str:
        """返回学习数据库的统计信息。"""
        total = len(self._entries)
        if total == 0:
            return "学习经验库为空。"

        # 按类型
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_area: dict[str, int] = {}
        total_recurrence = 0
        promotable = []

        for e in self._entries:
            by_type[e.entry_type] = by_type.get(e.entry_type, 0) + 1
            by_status[e.status] = by_status.get(e.status, 0) + 1
            by_category[f"{e.entry_type}/{e.category}"] = \
                by_category.get(f"{e.entry_type}/{e.category}", 0) + 1
            by_area[e.area] = by_area.get(e.area, 0) + 1
            total_recurrence += e.recurrence_count
            if e.recurrence_count >= 3 and e.status in ("pending", "in_progress"):
                promotable.append(e)

        lines = [
            "┌──────────────────────────────────┐",
            "│     学习经验库统计                       │",
            "└──────────────────────────────────┘",
            "",
            f"  总条目数:     {total}",
            f"  总重复次数:   {total_recurrence}",
            f"  存储目录:     {self._dir}",
            "",
            "  按类型:",
        ]
        for t, c in sorted(by_type.items()):
            lines.append(f"    {t}: {c}")

        lines.append("")
        lines.append("  按状态:")
        for s, c in sorted(by_status.items()):
            lines.append(f"    {s}: {c}")

        lines.append("")
        lines.append("  按分类:")
        for cat, c in sorted(by_category.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {cat}: {c}")

        if by_area:
            lines.append("")
            lines.append("  按领域:")
            for a, c in sorted(by_area.items(), key=lambda x: -x[1]):
                lines.append(f"    {a}: {c}")

        if promotable:
            lines.append("")
            lines.append(f"  ⚠ 可升级条目 (recurrence >= 3): {len(promotable)}")
            for e in promotable[:5]:
                lines.append(f"    - {e.entry_id} [{e.category}] "
                             f"{e.summary[:50]} ({e.recurrence_count}次)")

        return "\n".join(lines)

    def get_promotable(self, min_recurrence: int = 3) -> list[LearningEntry]:
        """获取可升级到 memory/skill 的候选条目。"""
        return [e for e in self._entries
                if e.recurrence_count >= min_recurrence
                and e.status in ("pending", "in_progress")]

    def promote_to_memory(self, entry_id: str,
                          memory_bank=None) -> str:
        """将学习条目升级为长期记忆（MemoryBank）。

        如果提供了 memory_bank，则调用 memory_bank.remember()。
        否则仅将状态更新为 promoted_to_memory。
        """
        entry = self._by_id.get(entry_id.strip())
        if not entry:
            return f"错误：'{entry_id}' 不存在。"

        if memory_bank:
            # 将学习分类映射到记忆分类
            cat_map = {
                "correction": "preference",
                "knowledge_gap": "fact",
                "best_practice": "decision",
                "tool_error": "fact",
                "command_error": "fact",
                "lint_error": "fact",
                "runtime_error": "fact",
            }
            mem_cat = cat_map.get(entry.category, "context")
            content = f"[从学习经验升级] {entry.summary}. {entry.suggested_action}"
            memory_bank.remember(mem_cat, content)

        self.update_status(entry_id, "promoted_to_memory")
        return f"已将 {entry_id} 升级到长期记忆。"

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _find_similar(self, entry: LearningEntry,
                      similarity_threshold: float = 0.7) -> Optional[LearningEntry]:
        """查找相似的已有条目用于去重。

        两级匹配策略:
          第一级（严格）：相同类型+分类 + 摘要中关键词高度重叠
          第二级（错误条目的宽松匹配）：相同类型+分类+工具标签 + 错误模式匹配
        """
        # 从新条目摘要中提取关键词
        new_keywords = set(re.findall(r'\w+', entry.summary.lower()))
        new_tags_lower = set(t.lower() for t in entry.tags)

        best_match = None
        best_score = 0.0

        for existing in self._entries:
            if existing.entry_type != entry.entry_type:
                continue
            if existing.category != entry.category:
                continue
            if existing.status in ("wont_fix", "promoted", "promoted_to_memory", "promoted_to_skill"):
                continue

            # --- 第一级：摘要关键词重叠 ---
            ex_keywords = set(re.findall(r'\w+', existing.summary.lower()))
            if new_keywords and ex_keywords:
                overlap = new_keywords & ex_keywords
                score = len(overlap) / max(len(new_keywords), len(ex_keywords))
                if score >= similarity_threshold and score > best_score:
                    best_score = score
                    best_match = existing
                    continue  # 找到了强匹配，但继续检查是否有更好的

            # --- 第二级：错误条目的宽松匹配 ---
            # 相同工具标签 + 详情中错误模式关键词重叠
            if entry.entry_type == "error" and new_tags_lower:
                ex_tags_lower = set(t.lower() for t in existing.tags)
                tag_overlap = new_tags_lower & ex_tags_lower
                if tag_overlap:
                    # 两者具有相同的工具名标签（例如都是 "bash" 错误）
                    # 检查详情中错误模式关键词是否重叠
                    new_detail_kw = set(re.findall(r'\w{4,}', (entry.details or "").lower()))
                    ex_detail_kw = set(re.findall(r'\w{4,}', (existing.details or "").lower()))
                    if new_detail_kw and ex_detail_kw:
                        detail_overlap = new_detail_kw & ex_detail_kw
                        # 常见错误模式如 "no such file"、"permission denied" 等
                        detail_score = len(detail_overlap) / min(len(new_detail_kw), len(ex_detail_kw)) if min(new_detail_kw, ex_detail_kw) else 0
                        if detail_score > 0.15:  # 低阈值——只需要一些共同的错误词汇
                            combined_score = 0.5 + detail_score * 0.3
                            if combined_score > best_score:
                                best_score = combined_score
                                best_match = existing

        return best_match

    @staticmethod
    def _extract_error_summary(output: str, tool_name: str) -> str:
        """从错误输出中提取简洁的一行摘要。"""
        # 尝试找到信息量最大的行
        lines = output.strip().splitlines()
        for line in lines:
            line = line.strip()
            # 跳过空行和堆栈跟踪头
            if not line:
                continue
            if line.startswith("  File ") or line.startswith("    "):
                continue
            if "Traceback" in line:
                continue
            # 返回第一条有意义的行，截断
            if len(line) > 10:
                return line[:150]

        # 兜底：第一个非空行
        for line in lines:
            if line.strip():
                return line.strip()[:150]

        return f"{tool_name} 执行失败"

    @staticmethod
    def _infer_priority(output: str) -> str:
        """从输出内容推断错误优先级。"""
        output_lower = output.lower()
        # 严重：权限、认证、数据丢失指标
        critical_kw = ("permission denied", "authentication failed",
                       "disk full", "out of memory", "segmentation fault")
        if any(kw in output_lower for kw in critical_kw):
            return "critical"
        # 高：常见硬性失败
        high_kw = ("traceback", "error:", "exception", "failed",
                   "非零退出码", "错误：")
        if any(kw in output_lower for kw in high_kw):
            return "high"
        return "medium"

    @staticmethod
    def _suggest_fix(tool_name: str, output: str) -> str:
        """根据工具和错误模式生成基本的修复建议。"""
        suggestions = {
            "bash": "检查命令语法、依赖是否安装、权限是否充足",
            "read_file": "确认文件路径正确，文件是否存在，检查权限",
            "write_file": "确认目录存在，检查磁盘空间和写入权限",
            "edit_file": "确认 old_text 精确匹配（注意空格和换行），文件是否存在",
            "git_commit": "检查是否有变更需要提交，检查 git 状态",
            "git_branch_create": "确认分支名合法，检查当前 git 状态",
        }
        base = suggestions.get(tool_name, "检查输入参数和环境状态")

        # 根据输出添加特定提示
        if "Permission denied" in output:
            base += "；可能需要 sudo 或修改文件权限"
        if "No such file" in output or "不存在" in output:
            base += "；确认文件/路径存在"
        if "command not found" in output:
            base += "；需要先安装该命令对应的工具"
        if "ModuleNotFoundError" in output or "ImportError" in output:
            base += "；需要安装缺失的 Python 依赖（pip install ...）"

        return base

    @property
    def count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# 模块级单例辅助函数
# ---------------------------------------------------------------------------

def _has_nonzero_exit(output_str: str) -> bool:
    """检查输出是否表示非零退出码。"""
    return bool(re.search(r'exit code [1-9]|non-zero|返回码.*[1-9]', output_str))

