#!/usr/bin/env python3
"""
树形 JSONL 会话管理器（灵感来自 pi-mono 的设计）。

核心概念：
  - 每个会话是一个独立的 .jsonl 文件
  - 每条记录都有 `id` + `parentId`，形成树形结构（类似 Git！）
  - `leaf`（叶节点）指针追踪当前位置
  - 追加操作在当前叶节点下创建子节点
  - 分支操作将叶节点移动到更早的记录（零拷贝，类似 git checkout）
  - 压缩在上下文过长时插入摘要节点

记录类型：
  - session_header: 文件元数据（id、时间戳、工作目录、版本）
  - message: 用户/助手/工具结果消息
  - compaction: LLM 生成的旧上下文摘要
  - branch_summary: 已放弃分支路径的摘要
  - model_change: 模型切换记录
  - label: 用户对某条记录的书签标记
  - custom: 扩展数据（不发送给 LLM）
  - custom_message: 扩展注入的消息（会发送给 LLM）
  - thought_chain: 思维链记录（CoT 推理过程，可选择性注入上下文）
  - thought_tree: 思维树记录（ToT 搜索过程，可选择性注入上下文）
  - thought_graph: 思维图记录（GoT 推理过程，可选择性注入上下文）
  - thinking_level_change: 思维模式切换记录

文件存储路径：
    ~/.sessions/<编码后的工作目录>/<时间戳>_<uuid>.jsonl
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..core.config import SESSIONS_DIR, WORKDIR

# ---------------------------------------------------------------------------
# 版本号与常量
# ---------------------------------------------------------------------------

CURRENT_SESSION_VERSION = 1


# ---------------------------------------------------------------------------
# 记录类型（使用普通字典模拟 dataclass，以简化 JSONL 序列化）
# ---------------------------------------------------------------------------


def make_session_header(session_id: str = None, cwd: str = None,
                        parent_session: str = None) -> dict:
    """创建会话头（每个 .jsonl 文件的第一行）。"""
    return {
        "type": "session",
        "version": CURRENT_SESSION_VERSION,
        "id": session_id or str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "cwd": cwd or str(WORKDIR),
        "parentSession": parent_session,
    }


def make_entry_base(entry_type: str, entry_id: str = None,
                    parent_id: str = None) -> dict:
    """创建所有记录共有的基础字段。"""
    return {
        "type": entry_type,
        "id": entry_id or _short_id(),
        "parentId": parent_id,  # 根记录为 null/None
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }


def make_message_entry(message: dict, parent_id: str = None) -> dict:
    """创建消息记录（用户 / 助手 / 工具结果）。"""
    entry = make_entry_base("message", parent_id=parent_id)
    entry["message"] = message
    return entry


def make_compaction_entry(summary: str, first_kept_entry_id: str,
                          tokens_before: int, parent_id: str = None,
                          details: dict = None) -> dict:
    """创建压缩（上下文压缩）记录。"""
    entry = make_entry_base("compaction", parent_id=parent_id)
    entry["summary"] = summary
    entry["firstKeptEntryId"] = first_kept_entry_id
    entry["tokensBefore"] = tokens_before
    if details:
        entry["details"] = details
    return entry


def make_branch_summary_entry(from_id: str, summary: str,
                               parent_id: str = None) -> dict:
    """创建分支摘要记录（记录被放弃的分支路径上下文）。"""
    entry = make_entry_base("branch_summary", parent_id=parent_id)
    entry["fromId"] = from_id
    entry["summary"] = summary
    return entry


def make_model_change_entry(provider: str, model_id: str,
                             parent_id: str = None) -> dict:
    """记录一次模型切换。"""
    entry = make_entry_base("model_change", parent_id=parent_id)
    entry["provider"] = provider
    entry["modelId"] = model_id
    return entry


def make_label_entry(target_id: str, label: str = None,
                      parent_id: str = None) -> dict:
    """用户对某条记录定义的书签/标记。"""
    entry = make_entry_base("label", parent_id=parent_id)
    entry["targetId"] = target_id
    entry["label"] = label
    return entry


def make_custom_entry(custom_type: str, data: Any = None,
                       parent_id: str = None) -> dict:
    """扩展专用数据（不发送给 LLM）。"""
    entry = make_entry_base("custom", parent_id=parent_id)
    entry["customType"] = custom_type
    if data is not None:
        entry["data"] = data
    return entry


def make_custom_message_entry(custom_type: str, content: str,
                               display: bool = True, details: Any = None,
                               parent_id: str = None) -> dict:
    """扩展注入的消息（会以用户角色发送给 LLM）。"""
    entry = make_entry_base("custom_message", parent_id=parent_id)
    entry["customType"] = custom_type
    entry["content"] = content
    entry["display"] = display
    if details is not None:
        entry["details"] = details
    return entry


def make_thought_chain_entry(chain_data: dict,
                              parent_id: str = None) -> dict:
    """创建思维链记录（CoT 推理过程）。"""
    entry = make_entry_base("thought_chain", parent_id=parent_id)
    entry.update(chain_data)
    return entry


def make_thinking_level_change_entry(level: str,
                                      reason: str = None,
                                      parent_id: str = None) -> dict:
    """记录思维模式切换。"""
    entry = make_entry_base("thinking_level_change", parent_id=parent_id)
    entry["thinkingLevel"] = level
    if reason:
        entry["reason"] = reason
    return entry


def make_thought_tree_entry(tree_data: dict,
                             parent_id: str = None) -> dict:
    """创建思维树记录（ToT 搜索过程）。"""
    entry = make_entry_base("thought_tree", parent_id=parent_id)
    entry.update(tree_data)
    return entry


def make_thought_graph_entry(graph_data: dict,
                              parent_id: str = None) -> dict:
    """创建思维图记录（GoT 推理过程）。"""
    entry = make_entry_base("thought_graph", parent_id=parent_id)
    entry.update(graph_data)
    return entry


# ---------------------------------------------------------------------------
# ID 生成
# ---------------------------------------------------------------------------

_existing_ids: set = set()


def _short_id(length: int = 8) -> str:
    """生成短唯一十六进制 ID（带冲突检查）。"""
    for _ in range(100):
        cid = uuid.uuid4().hex[:length]
        if cid not in _existing_ids:
            _existing_ids.add(cid)
            return cid
        return uuid.uuid4().hex  # 兜底方案


# ---------------------------------------------------------------------------
# 文件 I/O 辅助函数
# ---------------------------------------------------------------------------


def _encode_cwd(cwd: str) -> str:
    """将工作目录编码为安全的目录名。"""
    safe = cwd.replace("/", "-").replace("\\", "-").replace(":", "-")
    safe = safe.strip("-")
    return f"--{safe}--" if safe else "--root--"


def _get_session_dir(cwd: str = None) -> Path:
    """获取或创建指定工作目录的会话存储目录。"""
    encoded = _encode_cwd(cwd or str(WORKDIR))
    dir_path = SESSIONS_DIR / encoded
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _load_entries(filepath: Path) -> list:
    """解析 .jsonl 文件为记录字典列表。出错时返回空列表。"""
    if not filepath.exists():
        return []
    entries = []
    try:
        for line in filepath.read_text().strip().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        return []
    # 验证文件头
    if entries and entries[0].get("type") != "session":
        return []
    return entries


def _is_valid_session_file(filepath: Path) -> bool:
    """快速检查：这是否是一个有效的会话文件？"""
    try:
        first_line = filepath.open().readline().strip()
        if not first_line:
            return False
        header = json.loads(first_line)
        return header.get("type") == "session" and isinstance(header.get("id"), str)
    except (OSError, json.JSONDecodeError):
        return False


def _find_most_recent_session(session_dir: Path) -> Optional[Path]:
    """查找目录中最近修改的会话文件。"""
    if not session_dir.exists():
        return None
    candidates = []
    for f in session_dir.glob("*.jsonl"):
        if _is_valid_session_file(f):
            candidates.append((f.stat().st_mtime, f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# 会话管理器
# ---------------------------------------------------------------------------


class SessionManager:
    """
    树形 JSONL 会话管理器。

    将对话会话管理为存储在 JSONL 文件中的只追加树形结构。
    每条记录都有 id/parentId 形成树形结构。'leaf' 指针追踪当前
    位置。追加操作在叶节点下创建子节点。分支操作将叶节点回移。

    使用方式：
        sm = SessionManager.create()          # 新建会话
        sm.append_message(user_msg)           # 添加用户消息
        sm.append_message(assistant_msg)      # 添加助手消息
        sm.branch(entry_id)                   # 回到之前的节点
        sm.append_message(new_user_msg)       # 创建新分支！
        ctx = sm.build_context()              # → 生成给 LLM 用的扁平列表
    """

    def __init__(self, cwd: str = None, session_dir: Path = None,
                 session_file: Path = None, persist: bool = True):
        self.cwd = cwd or str(WORKDIR)
        self.session_dir = session_dir or _get_session_dir(self.cwd)
        self.persist = persist
        self.flushed = False

        if persist and not self.session_dir.exists():
            self.session_dir.mkdir(parents=True, exist_ok=True)

        # 内部状态
        self._session_id: str = ""
        self._session_file: Optional[Path] = None
        self._file_entries: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._labels: dict[str, str] = {}   # targetId → label text
        self._leaf_id: Optional[str] = None

        if session_file:
            self._open_session(session_file)
        else:
            self._new_session()

    # ------------------------------------------------------------------
    # 会话生命周期
    # ------------------------------------------------------------------

    def _new_session(self, session_id: str = None, parent_session: str = None):
        """初始化一个全新的会话。"""
        self._session_id = session_id or str(uuid.uuid4())
        header = make_session_header(
            session_id=self._session_id,
            cwd=self.cwd,
            parent_session=parent_session,
        )
        self._file_entries = [header]
        self._by_id = {}
        self._labels = {}
        self._leaf_id = None
        self.flushed = False

        if self.persist:
            ts = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
            self._session_file = self.session_dir / f"{ts}_{self._session_id}.jsonl"

    def _open_session(self, session_file: Path):
        """打开一个已有的会话文件。"""
        self._session_file = session_file.resolve()
        if session_file.exists():
            self._file_entries = _load_entries(session_file)
            if not self._file_entries:
                # 已损坏或空文件 —— 在同一路径重新开始
                self._new_session()
                self._session_file = session_file.resolve()
                self._rewrite_file()
                self.flushed = True
                return
            header = self._file_entries[0]
            self._session_id = header.get("id", str(uuid.uuid4()))
            self._build_index()
            self.flushed = True
            # 打开时将所有记录索引到 FTS5（尽力而为）
            try:
                from ..data.search import SEARCH_INDEX
                non_header = [e for e in self._file_entries if e.get("type") != "session"]
                SEARCH_INDEX.index_entries(
                    non_header,
                    session_id=self._session_id,
                    session_file=self._session_file or "",
                    cwd=self.cwd,
                )
            except Exception:
                pass
        else:
            # 在指定路径创建新文件
            actual_path = session_file.resolve()
            self._new_session()
            self._session_file = actual_path

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def _build_index(self):
        """从 file_entries 重建 by_id 索引和标签字典。"""
        self._by_id.clear()
        self._labels.clear()
        self._leaf_id = None
        for entry in self._file_entries:
            if entry.get("type") == "session":
                continue
            eid = entry.get("id")
            if eid:
                self._by_id[eid] = entry
                self._leaf_id = eid
            if entry.get("type") == "label":
                tid = entry.get("targetId")
                lbl = entry.get("label")
                if tid:
                    if lbl:
                        self._labels[tid] = lbl
                    else:
                        self._labels.pop(tid, None)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _rewrite_file(self):
        """将所有记录写入文件（全量重写）。"""
        if not self.persist or not self._session_file:
            return
        lines = "\n".join(json.dumps(e, default=str) for e in self._file_entries)
        self._session_file.write_text(lines + "\n")

    def _append_to_file(self, entry: dict):
        """向 JSONL 文件追加单条记录。"""
        if not self.persist or not self._session_file:
            return

        # 延迟写入，直到至少有一条助手消息
        has_assistant = any(
            e.get("type") == "message"
            and e.get("message", {}).get("role") == "assistant"
            for e in self._file_entries
        )
        if not has_assistant:
            self.flushed = False
            return

        if not self.flushed:
            # 首次刷新：写入所有内容
            with open(self._session_file, "a") as f:
                for e in self._file_entries:
                    f.write(json.dumps(e, default=str) + "\n")
            self.flushed = True
        else:
            # 后续：仅追加新记录
            with open(self._session_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_file(self) -> Optional[Path]:
        return self._session_file

    @property
    def leaf_id(self) -> Optional[str]:
        return self._leaf_id

    def get_cwd(self) -> str:
        return self.cwd

    def get_session_dir(self) -> Path:
        return self.session_dir

    def get_header(self) -> Optional[dict]:
        for e in self._file_entries:
            if e.get("type") == "session":
                return e
        return None

    # ------------------------------------------------------------------
    # 追加操作
    # ------------------------------------------------------------------

    def _append_entry(self, entry: dict) -> str:
        """内部方法：将记录添加到内存 + 持久化 + 建立搜索索引。"""
        self._file_entries.append(entry)
        eid = entry.get("id", "")
        if eid:
            self._by_id[eid] = entry
        self._leaf_id = eid
        self._append_to_file(entry)
        # 自动索引到 FTS5（非关键功能：索引失败不会影响会话）
        try:
            from ..data.search import SEARCH_INDEX
            SEARCH_INDEX.index_entry(
                entry,
                session_id=self._session_id,
                session_file=self._session_file or "",
                cwd=self.cwd,
            )
        except Exception:
            pass  # 搜索索引是尽力而为的，失败不影响主流程
        return eid

    def append_message(self, message: dict) -> str:
        """追加用户/助手/工具结果消息。返回记录 ID。"""
        entry = make_message_entry(message, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_compaction(self, summary: str, first_kept_entry_id: str,
                           tokens_before: int, details: dict = None) -> str:
        """追加压缩摘要记录。"""
        entry = make_compaction_entry(
            summary, first_kept_entry_id, tokens_before,
            parent_id=self._leaf_id, details=details,
        )
        return self._append_entry(entry)

    def append_branch_summary(self, from_id: str, summary: str) -> str:
        """追加分支摘要（来自被放弃路径的上下文）。"""
        entry = make_branch_summary_entry(from_id, summary, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_model_change(self, provider: str, model_id: str) -> str:
        """记录模型变更。"""
        entry = make_model_change_entry(provider, model_id, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_label(self, target_id: str, label: str = None) -> str:
        """在某条记录上设置/清除书签标签。"""
        if target_id not in self._by_id:
            raise ValueError(f"Entry {target_id} not found")
        entry = make_label_entry(target_id, label, parent_id=self._leaf_id)
        result_id = self._append_entry(entry)
        if label:
            self._labels[target_id] = label
        else:
            self._labels.pop(target_id, None)
        return result_id

    def append_custom(self, custom_type: str, data: Any = None) -> str:
        """追加扩展数据（不发送给 LLM）。"""
        entry = make_custom_entry(custom_type, data, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_custom_message(self, custom_type: str, content: str,
                               display: bool = True, details: Any = None) -> str:
        """追加扩展消息（注入到 LLM 上下文中）。"""
        entry = make_custom_message_entry(
            custom_type, content, display, details, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_thought_chain(self, chain_data: dict) -> str:
        """追加思维链记录。"""
        entry = make_thought_chain_entry(chain_data, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_thinking_level_change(self, level: str,
                                     reason: str = None) -> str:
        """记录思维模式切换。"""
        entry = make_thinking_level_change_entry(
            level, reason, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_thought_tree(self, tree_data: dict) -> str:
        """追加思维树记录。"""
        entry = make_thought_tree_entry(tree_data, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_thought_graph(self, graph_data: dict) -> str:
        """追加思维图记录。"""
        entry = make_thought_graph_entry(graph_data, parent_id=self._leaf_id)
        return self._append_entry(entry)

    # ------------------------------------------------------------------
    # 树遍历
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> Optional[dict]:
        """根据 ID 获取记录。"""
        return self._by_id.get(entry_id)

    def get_leaf_entry(self) -> Optional[dict]:
        """获取当前叶节点记录。"""
        if self._leaf_id:
            return self._by_id.get(self._leaf_id)
        return None

    def get_children(self, parent_id: str) -> list[dict]:
        """获取某条记录的直接子记录。"""
        return [e for e in self._by_id.values()
                if e.get("parentId") == parent_id]

    def get_label(self, entry_id: str) -> Optional[str]:
        """获取某条记录的标签。"""
        return self._labels.get(entry_id)

    def get_branch(self, from_id: str = None) -> list[dict]:
        """
        从指定记录走到根节点，返回有序路径（根→叶）。
        如果 from_id 为 None，则从当前叶节点开始走。
        """
        start_id = from_id or self._leaf_id
        path = []
        current = self._by_id.get(start_id) if start_id else None
        while current:
            path.insert(0, current)
            pid = current.get("parentId")
            current = self._by_id.get(pid) if pid else None
        return path

    def get_entries(self) -> list[dict]:
        """返回除文件头外的所有记录。"""
        return [e for e in self._file_entries if e.get("type") != "session"]

    def get_tree(self) -> list[dict]:
        """
        将树结构返回为嵌套字典，包含 'entry' 和 'children' 键。
        """
        entries = self.get_entries()
        node_map: dict[str, dict] = {}
        roots = []

        for entry in entries:
            eid = entry.get("id", "")
            label = self._labels.get(eid)
            node_map[eid] = {"entry": entry, "children": [], "_label": label}

        for entry in entries:
            eid = entry.get("id", "")
            pid = entry.get("parentId")
            node = node_map[eid]
            if pid is None or pid == eid or pid not in node_map:
                roots.append(node)
            else:
                node_map[pid]["children"].append(node)

        # 按时间戳排序子节点
        def sort_children(nodes):
            nodes.sort(key=lambda n: n["entry"].get("timestamp", ""))
            for n in nodes:
                sort_children(n["children"])

        sort_children(roots)
        return roots

    # ------------------------------------------------------------------
    # 分支操作
    # ------------------------------------------------------------------

    def branch(self, branch_from_id: str):
        """
        将叶节点指针移动到更早的记录（零拷贝！）。
        下次追加将在该记录下创建子节点 → 新分支。
        类似 `git checkout <commit>`。
        """
        if branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def branch_with_summary(self, branch_from_id: str, summary: str) -> str:
        """分支 + 记录被放弃路径上的内容。"""
        if branch_from_id and branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        return self.append_branch_summary(
            branch_from_id or "root", summary)

    def reset_leaf(self):
        """重置叶节点为 null（下次追加将成为根节点）。"""
        self._leaf_id = None

    # ------------------------------------------------------------------
    # 上下文构建（树 → 给 LLM 用的扁平列表）
    # ------------------------------------------------------------------

    def build_context(self) -> dict:
        """
        构建供 LLM 使用的扁平化上下文。

        从叶节点走到根节点，处理压缩和分支摘要。

        返回：
            {"messages": [...], "thinkingLevel": "...", "model": {...}|None}
        """
        entries = self.get_entries()
        if not entries:
            return {"messages": [], "thinkingLevel": "off", "model": None}

        # 构建索引
        by_id = self._by_id

        # 找到叶节点
        leaf = None
        if self._leaf_id is not None:
            leaf = by_id.get(self._leaf_id)
        if not leaf and entries:
            leaf = entries[-1]
        if not leaf:
            return {"messages": [], "thinkingLevel": "off", "model": None}

        # 从叶节点走到根节点，收集路径
        path = []
        current = leaf
        while current:
            path.insert(0, current)
            pid = current.get("parentId")
            current = by_id.get(pid) if pid else None

        # 沿路径提取设置信息
        thinking_level = "off"
        model_info = None
        compaction_entry = None

        for entry in path:
            etype = entry.get("type")
            if etype == "thinking_level_change":
                thinking_level = entry.get("thinkingLevel", "off")
            elif etype == "model_change":
                model_info = {
                    "provider": entry.get("provider"),
                    "modelId": entry.get("modelId"),
                }
            elif etype == "compaction":
                compaction_entry = entry

        # 构建消息列表
        messages = []

        def emit(entry):
            etype = entry.get("type")
            if etype == "message":
                messages.append(entry.get("message"))
            elif etype == "custom_message":
                messages.append({
                    "role": "user",
                    "content": entry.get("content", ""),
                    "_customType": entry.get("customType"),
                })
            elif etype == "branch_summary" and entry.get("summary"):
                messages.append({
                    "role": "user",
                    "content": f"[Branch Summary - from {entry.get('fromId')}]\n{entry['summary']}",
                    "_meta": "branch_summary",
                })
            elif etype == "thought_chain":
                # 思维链：作为压缩摘要注入上下文（可选）
                # 只注入结论和关键步骤，避免占用过多 token
                conclusion = entry.get("conclusion", "")
                steps = entry.get("steps", [])
                chain_goal = entry.get("goal", "")
                if conclusion or steps:
                    # 构建精简版思维链摘要
                    step_summary = "\n".join([
                        f"  - [{s.get('thought_type', '?')}] {s.get('content', '')[:100]}"
                        for s in steps[:4]  # 最多 4 个步骤
                    ])
                    thought_text = (
                        f"[Thought Chain] Goal: {chain_goal[:80]}\n"
                        f"{step_summary}\n"
                        f"Conclusion: {conclusion[:200]}"
                    )
                    messages.append({
                        "role": "user",
                        "content": thought_text,
                        "_meta": "thought_chain",
                    })
            elif etype == "thought_tree":
                # 思维树：注入最优路径摘要
                conclusion = entry.get("conclusion", "")
                best_path = entry.get("bestPath", [])
                tree_goal = entry.get("goal", "")
                strategy = entry.get("strategy", "?")
                best_score = entry.get("bestScore", 0)
                node_summary = entry.get("nodeSummary", [])
                if conclusion or best_path:
                    # 构建精简版思维树摘要（只显示最优路径）
                    path_nodes = []
                    for ns in node_summary:
                        if ns.get("id") in best_path:
                            path_nodes.append(
                                f"  [D{ns.get('depth', '?')}] {ns.get('content', '')[:80]} "
                                f"(score={ns.get('score', 0)})"
                            )
                    tot_text = (
                        f"[Thought Tree ToT] Goal: {tree_goal[:80]}\n"
                        f"Strategy: {strategy} | Best Score: {best_score:.2f}\n"
                        f"Best Path ({len(best_path)} nodes):\n"
                    )
                    if path_nodes:
                        tot_text += "\n".join(path_nodes[:6])
                    else:
                        tot_text += " → ".join([p[:30] for p in best_path[:5]])
                    tot_text += f"\nConclusion: {conclusion[:200]}"
                    messages.append({
                        "role": "user",
                        "content": tot_text,
                        "_meta": "thought_tree",
                    })
            elif etype == "thought_graph":
                # 思维图：注入关键发现和结论
                conclusion = entry.get("conclusion", "")
                graph_goal = entry.get("goal", "")
                gmode = entry.get("mode", "?")
                key_findings = entry.get("keyFindings", [])
                node_summary = entry.get("nodeSummary", [])
                if conclusion or key_findings:
                    # 按深度分组显示关键节点
                    by_depth = {}
                    for ns in node_summary:
                        d = ns.get("depth", 0)
                        if d not in by_depth:
                            by_depth[d] = []
                        by_depth[d].append(
                            f"  [{ns.get('type','?')}] {ns.get('content','')[:60]} "
                            f"(score={ns.get('score',0)})"
                        )
                    got_text = (
                        f"[Thought Graph GoT] Goal: {graph_goal[:80]}\n"
                        f"Mode: {gmode} | Nodes: {entry.get('totalNodes',0)} | "
                        f"Edges: {entry.get('totalEdges',0)}\n"
                    )
                    if key_findings:
                        got_text += "Key Findings:\n" + \
                                   "\n".join(f"  - {f}" for f in key_findings[:3]) + "\n"
                    for d in sorted(by_depth.keys()):
                        got_text += f"Depth {d}:\n" + "\n".join(by_depth[d][:3]) + "\n"
                    got_text += f"Conclusion: {conclusion[:200]}"
                    messages.append({
                        "role": "user",
                        "content": got_text,
                        "_meta": "thought_graph",
                    })

        if compaction_entry:
            messages.append({
                "role": "user",
                "content": (
                    f"[Context Compaction - was {compaction_entry['tokensBefore']} tokens]\n"
                    f"{compaction_entry['summary']}"
                ),
                "_meta": "compaction",
            })

            # 在路径中找到压缩位置
            comp_idx = None
            for i, e in enumerate(path):
                if e.get("id") == compaction_entry.get("id"):
                    comp_idx = i
                    break

            kept_id = compaction_entry.get("firstKeptEntryId")
            # 从 firstKeptEntryId 开始输出压缩前的消息
            if comp_idx is not None:
                keeping = False
                for i in range(comp_idx):
                    if path[i].get("id") == kept_id:
                        keeping = True
                    if keeping:
                        emit(path[i])
                # 输出压缩后的消息
                for i in range(comp_idx + 1, len(path)):
                    emit(path[i])
        else:
            # 无压缩 —— 输出所有消息类型记录
            for entry in path:
                emit(entry)

        return {
            "messages": messages,
            "thinkingLevel": thinking_level,
            "model": model_info,
        }

    # ------------------------------------------------------------------
    # 会话列表
    # ------------------------------------------------------------------

    @staticmethod
    def create(cwd: str = None, session_dir: Path = None) -> "SessionManager":
        """工厂方法：创建新会话。"""
        return SessionManager(cwd=cwd, session_dir=session_dir, persist=True)

    @staticmethod
    def open_session(filepath: str | Path) -> "SessionManager":
        """工厂方法：打开已有会话文件。"""
        p = Path(filepath)
        # Try to extract cwd from header
        entries = _load_entries(p)
        cwd = None
        if entries:
            cwd = entries[0].get("cwd")
        return SessionManager(cwd=cwd, session_file=p, persist=True)

    @staticmethod
    def continue_recent(cwd: str = None) -> "SessionManager":
        """工厂方法：继续最近的会话，或创建新会话。"""
        sdir = _get_session_dir(cwd)
        recent = _find_most_recent_session(sdir)
        if recent:
            return SessionManager.open_session(recent)
        return SessionManager.create(cwd=cwd)

    @staticmethod
    def in_memory(cwd: str = None) -> "SessionManager":
        """工厂方法：创建内存会话（不持久化）。"""
        return SessionManager(cwd=cwd, persist=False)

    @staticmethod
    def list_sessions(cwd: str = None) -> list[dict]:
        """列出指定工作目录下的所有会话。"""
        sdir = _get_session_dir(cwd)
        if not sdir.exists():
            return []
        sessions = []
        for f in sorted(sdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not _is_valid_session_file(f):
                continue
            entries = _load_entries(f)
            if not entries:
                continue
            header = entries[0]
            msg_count = sum(1 for e in entries if e.get("type") == "message")
            first_msg = ""
            for e in entries:
                if e.get("type") == "message":
                    msg = e.get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, str) and content:
                        first_msg = content[:80]
                        break
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                first_msg = part.get("text", "")[:80]
                                break
                        break  # 修正缩进：外层循环也需要 break
            sessions.append({
                "path": str(f),
                "id": header.get("id", "?"),
                "cwd": header.get("cwd", ""),
                "created": header.get("timestamp", ""),
                "modified": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.localtime(f.stat().st_mtime)),
                "messageCount": msg_count,
                "firstMessage": first_msg or "(no messages)",
            })
        return sessions

    @staticmethod
    def list_all_sessions() -> list[dict]:
        """列出所有工作目录下的会话。"""
        if not SESSIONS_DIR.exists():
            return []
        all_sessions = []
        for d in sorted(SESSIONS_DIR.iterdir()):
            if d.is_dir():
                all_sessions.extend(SessionManager.list_sessions(cwd=None))
                # 修正：list_sessions 使用 _get_session_dir 会编码 cwd，
                # 但这里我们需要每个子目录。直接扫描即可。
                all_sessions = []  # 重置，重新正确实现
                break

        # 正确的实现
        if SESSIONS_DIR.exists():
            for subdir in sorted(SESSIONS_DIR.iterdir()):
                if subdir.is_dir():
                    for f in sorted(subdir.glob("*.jsonl"),
                                    key=lambda p: p.stat().st_mtime, reverse=True):
                        if not _is_valid_session_file(f):
                            continue
                        entries = _load_entries(f)
                        if not entries:
                            continue
                        header = entries[0]
                        msg_count = sum(1 for e in entries if e.get("type") == "message")
                        all_sessions.append({
                            "path": str(f),
                            "id": header.get("id", "?"),
                            "cwd": header.get("cwd", ""),
                            "created": header.get("timestamp", ""),
                            "modified": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ",
                                time.localtime(f.stat().st_mtime)),
                            "messageCount": msg_count,
                        })
        all_sessions.sort(key=lambda s: s["modified"], reverse=True)
        return all_sessions

    # ------------------------------------------------------------------
    # FTS5 全文搜索集成
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20,
               entry_type: str = None, role: str = None) -> list[dict]:
        """
        使用 SQLite FTS5 在当前会话中搜索。

        参数：
            query: FTS5 搜索表达式（支持 AND/OR/NOT、短语、前缀通配符*）
            limit: 最大返回结果数
            entry_type: 按类型过滤（message、compaction 等）
            role: 按消息角色过滤（user、assistant）

        返回：
            结果字典列表，包含 entry_id、snippet（高亮）、
            rank、时间戳等字段。
        """
        try:
            from ..data.search import SEARCH_INDEX
            return SEARCH_INDEX.search(
                query=query,
                limit=limit,
                session_id=self._session_id,
                cwd=self.cwd,
                entry_type=entry_type,
                role=role,
            )
        except Exception as e:
            return [{"error": f"Search error: {e}"}]

    def rebuild_search_index(self) -> str:
        """重建当前会话的 FTS5 索引。"""
        try:
            from ..data.search import SEARCH_INDEX
            return SEARCH_INDEX.rebuild_index(session_manager=self)
        except Exception as e:
            return f"Rebuild failed: {e}"

    @staticmethod
    def global_search(query: str, limit: int = 20,
                      cwd: str = None) -> list[dict]:
        """
        使用 FTS5 搜索所有会话。

        参数：
            query: FTS5 搜索表达式
            limit: 每类最大返回结果数
            cwd: 可选的工作目录过滤

        返回：
            包含 'entries' 和 'sessions' 键的字典。
        """
        try:
            from ..data.search import SEARCH_INDEX
            entries = SEARCH_INDEX.search(
                query=query, limit=limit, cwd=cwd
            )
            sessions = SEARCH_INDEX.search_sessions(query, limit=limit)
            return {"entries": entries, "sessions": sessions}
        except Exception as e:
            return {"entries": [{"error": f"Search error: {e}"}],
                    "sessions": []}

    @staticmethod
    def search_stats() -> dict:
        """返回搜索索引的统计信息。"""
        try:
            from ..data.search import SEARCH_INDEX
            return SEARCH_INDEX.get_stats()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def rebuild_global_search_index() -> str:
        """从所有 JSONL 文件重建整个搜索索引。"""
        try:
            from ..data.search import SEARCH_INDEX
            return SEARCH_INDEX.rebuild_index()
        except Exception as e:
            return f"Global rebuild failed: {e}"

