#!/usr/bin/env python3
"""
Tree-structured JSONL Session Manager (inspired by pi-mono's design).

Core concepts:
  - Each session is a single .jsonl file
  - Every entry has `id` + `parentId`, forming a tree (like Git!)
  - A `leaf` pointer tracks the current position
  - Appending creates a child of the current leaf
  - Branching moves the leaf to an earlier entry (zero-copy, like git checkout)
  - Compaction inserts a summary node when context grows too long

Entry types:
  - session_header: file metadata (id, timestamp, cwd, version)
  - message: user/assistant/tool_result messages
  - compaction: LLM-generated summary of old context
  - branch_summary: summary of an abandoned branch path
  - model_change: model switch record
  - label: user bookmark on an entry
  - custom: extension data (not sent to LLM)
  - custom_message: extension-injected message (sent to LLM)

File layout:
    ~/.sessions/<encoded-cwd>/<timestamp>_<uuid>.jsonl
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .config import SESSIONS_DIR, WORKDIR

# ---------------------------------------------------------------------------
# Version & Constants
# ---------------------------------------------------------------------------

CURRENT_SESSION_VERSION = 1


# ---------------------------------------------------------------------------
# Entry Types (dataclass-like using plain dicts for JSONL simplicity)
# ---------------------------------------------------------------------------


def make_session_header(session_id: str = None, cwd: str = None,
                        parent_session: str = None) -> dict:
    """Create session header (first line of every .jsonl file)."""
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
    """Create base fields shared by all entries."""
    return {
        "type": entry_type,
        "id": entry_id or _short_id(),
        "parentId": parent_id,  # null/None for root entries
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }


def make_message_entry(message: dict, parent_id: str = None) -> dict:
    """Create a message entry (user / assistant / tool_result)."""
    entry = make_entry_base("message", parent_id=parent_id)
    entry["message"] = message
    return entry


def make_compaction_entry(summary: str, first_kept_entry_id: str,
                          tokens_before: int, parent_id: str = None,
                          details: dict = None) -> dict:
    """Create a compaction (context compression) entry."""
    entry = make_entry_base("compaction", parent_id=parent_id)
    entry["summary"] = summary
    entry["firstKeptEntryId"] = first_kept_entry_id
    entry["tokensBefore"] = tokens_before
    if details:
        entry["details"] = details
    return entry


def make_branch_summary_entry(from_id: str, summary: str,
                               parent_id: str = None) -> dict:
    """Create a branch summary entry (records abandoned path context)."""
    entry = make_entry_base("branch_summary", parent_id=parent_id)
    entry["fromId"] = from_id
    entry["summary"] = summary
    return entry


def make_model_change_entry(provider: str, model_id: str,
                             parent_id: str = None) -> dict:
    """Record a model switch."""
    entry = make_entry_base("model_change", parent_id=parent_id)
    entry["provider"] = provider
    entry["modelId"] = model_id
    return entry


def make_label_entry(target_id: str, label: str = None,
                      parent_id: str = None) -> dict:
    """User-defined bookmark/marker on an entry."""
    entry = make_entry_base("label", parent_id=parent_id)
    entry["targetId"] = target_id
    entry["label"] = label
    return entry


def make_custom_entry(custom_type: str, data: Any = None,
                       parent_id: str = None) -> dict:
    """Extension-specific data (NOT sent to LLM)."""
    entry = make_entry_base("custom", parent_id=parent_id)
    entry["customType"] = custom_type
    if data is not None:
        entry["data"] = data
    return entry


def make_custom_message_entry(custom_type: str, content: str,
                               display: bool = True, details: Any = None,
                               parent_id: str = None) -> dict:
    """Extension-injected message (IS sent to LLM as user role)."""
    entry = make_entry_base("custom_message", parent_id=parent_id)
    entry["customType"] = custom_type
    entry["content"] = content
    entry["display"] = display
    if details is not None:
        entry["details"] = details
    return entry


# ---------------------------------------------------------------------------
# ID Generation
# ---------------------------------------------------------------------------

_existing_ids: set = set()


def _short_id(length: int = 8) -> str:
    """Generate a short unique hex ID (collision-checked)."""
    for _ in range(100):
        cid = uuid.uuid4().hex[:length]
        if cid not in _existing_ids:
            _existing_ids.add(cid)
            return cid
    return uuid.uuid4().hex  # fallback


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------


def _encode_cwd(cwd: str) -> str:
    """Encode working directory into a safe directory name."""
    safe = cwd.replace("/", "-").replace("\\", "-").replace(":", "-")
    safe = safe.strip("-")
    return f"--{safe}--" if safe else "--root--"


def _get_session_dir(cwd: str = None) -> Path:
    """Get/create the session storage directory for a working directory."""
    encoded = _encode_cwd(cwd or str(WORKDIR))
    dir_path = SESSIONS_DIR / encoded
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _load_entries(filepath: Path) -> list:
    """Parse a .jsonl file into list of entry dicts. Returns [] on error."""
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
    # Validate header
    if entries and entries[0].get("type") != "session":
        return []
    return entries


def _is_valid_session_file(filepath: Path) -> bool:
    """Quick check: does this look like a valid session file?"""
    try:
        first_line = filepath.open().readline().strip()
        if not first_line:
            return False
        header = json.loads(first_line)
        return header.get("type") == "session" and isinstance(header.get("id"), str)
    except (OSError, json.JSONDecodeError):
        return False


def _find_most_recent_session(session_dir: Path) -> Optional[Path]:
    """Find the most recently modified session file in a directory."""
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
# Session Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """
    Tree-structured JSONL session manager.

    Manages conversation sessions as append-only trees stored in JSONL files.
    Each entry has id/parentId forming a tree. The 'leaf' pointer tracks current
    position. Appending creates a child of leaf. Branching moves leaf backward.

    Usage:
        sm = SessionManager.create()          # new session
        sm.append_message(user_msg)           # add user msg
        sm.append_message(assistant_msg)      # add assistant msg
        sm.branch(entry_id)                   # go back in time
        sm.append_message(new_user_msg)       # new branch!
        ctx = sm.build_context()              # → flat list for LLM
    """

    def __init__(self, cwd: str = None, session_dir: Path = None,
                 session_file: Path = None, persist: bool = True):
        self.cwd = cwd or str(WORKDIR)
        self.session_dir = session_dir or _get_session_dir(self.cwd)
        self.persist = persist
        self.flushed = False

        if persist and not self.session_dir.exists():
            self.session_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
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
    # Session lifecycle
    # ------------------------------------------------------------------

    def _new_session(self, session_id: str = None, parent_session: str = None):
        """Initialize a brand new session."""
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
        """Open an existing session file."""
        self._session_file = session_file.resolve()
        if session_file.exists():
            self._file_entries = _load_entries(session_file)
            if not self._file_entries:
                # Corrupted/empty — start fresh at same path
                self._new_session()
                self._session_file = session_file.resolve()
                self._rewrite_file()
                self.flushed = True
                return
            header = self._file_entries[0]
            self._session_id = header.get("id", str(uuid.uuid4()))
            self._build_index()
            self.flushed = True
            # Index all entries into FTS5 on open (best-effort)
            try:
                from .search import SEARCH_INDEX
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
            # New file at specified path
            actual_path = session_file.resolve()
            self._new_session()
            self._session_file = actual_path

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _build_index(self):
        """Rebuild by_id index and labels from file_entries."""
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
    # Persistence
    # ------------------------------------------------------------------

    def _rewrite_file(self):
        """Write all entries to file (full rewrite)."""
        if not self.persist or not self._session_file:
            return
        lines = "\n".join(json.dumps(e, default=str) for e in self._file_entries)
        self._session_file.write_text(lines + "\n")

    def _append_to_file(self, entry: dict):
        """Append a single entry to the JSONL file."""
        if not self.persist or not self._session_file:
            return

        # Delay write until we have at least one assistant message
        has_assistant = any(
            e.get("type") == "message"
            and e.get("message", {}).get("role") == "assistant"
            for e in self._file_entries
        )
        if not has_assistant:
            self.flushed = False
            return

        if not self.flushed:
            # First flush: write everything
            with open(self._session_file, "a") as f:
                for e in self._file_entries:
                    f.write(json.dumps(e, default=str) + "\n")
            self.flushed = True
        else:
            # Subsequent: append only new entry
            with open(self._session_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # ------------------------------------------------------------------
    # Properties
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
    # Append operations
    # ------------------------------------------------------------------

    def _append_entry(self, entry: dict) -> str:
        """Internal: add entry to memory + persist + index for search."""
        self._file_entries.append(entry)
        eid = entry.get("id", "")
        if eid:
            self._by_id[eid] = entry
        self._leaf_id = eid
        self._append_to_file(entry)
        # Auto-index to FTS5 (non-critical: failure won't break session)
        try:
            from .search import SEARCH_INDEX
            SEARCH_INDEX.index_entry(
                entry,
                session_id=self._session_id,
                session_file=self._session_file or "",
                cwd=self.cwd,
            )
        except Exception:
            pass  # search index is best-effort
        return eid

    def append_message(self, message: dict) -> str:
        """Append a user/assistant/tool_result message. Returns entry ID."""
        entry = make_message_entry(message, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_compaction(self, summary: str, first_kept_entry_id: str,
                           tokens_before: int, details: dict = None) -> str:
        """Append a compaction summary entry."""
        entry = make_compaction_entry(
            summary, first_kept_entry_id, tokens_before,
            parent_id=self._leaf_id, details=details,
        )
        return self._append_entry(entry)

    def append_branch_summary(self, from_id: str, summary: str) -> str:
        """Append a branch summary (context from abandoned path)."""
        entry = make_branch_summary_entry(from_id, summary, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_model_change(self, provider: str, model_id: str) -> str:
        """Record a model change."""
        entry = make_model_change_entry(provider, model_id, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_label(self, target_id: str, label: str = None) -> str:
        """Set/clear a bookmark label on an entry."""
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
        """Append extension data (not sent to LLM)."""
        entry = make_custom_entry(custom_type, data, parent_id=self._leaf_id)
        return self._append_entry(entry)

    def append_custom_message(self, custom_type: str, content: str,
                               display: bool = True, details: Any = None) -> str:
        """Append extension message (injected into LLM context)."""
        entry = make_custom_message_entry(
            custom_type, content, display, details, parent_id=self._leaf_id)
        return self._append_entry(entry)

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> Optional[dict]:
        """Get entry by ID."""
        return self._by_id.get(entry_id)

    def get_leaf_entry(self) -> Optional[dict]:
        """Get the current leaf entry."""
        if self._leaf_id:
            return self._by_id.get(self._leaf_id)
        return None

    def get_children(self, parent_id: str) -> list[dict]:
        """Get direct children of an entry."""
        return [e for e in self._by_id.values()
                if e.get("parentId") == parent_id]

    def get_label(self, entry_id: str) -> Optional[str]:
        """Get label for an entry."""
        return self._labels.get(entry_id)

    def get_branch(self, from_id: str = None) -> list[dict]:
        """
        Walk from entry to root, returning ordered path (root→leaf).
        If from_id is None, walks from current leaf.
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
        """Return all entries except header."""
        return [e for e in self._file_entries if e.get("type") != "session"]

    def get_tree(self) -> list[dict]:
        """
        Return tree structure as nested dicts with 'entry' and 'children' keys.
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

        # Sort children by timestamp
        def sort_children(nodes):
            nodes.sort(key=lambda n: n["entry"].get("timestamp", ""))
            for n in nodes:
                sort_children(n["children"])

        sort_children(roots)
        return roots

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch(self, branch_from_id: str):
        """
        Move leaf pointer to an earlier entry (zero-copy!).
        Next append will create a child of that entry → new branch.
        Like `git checkout <commit>`.
        """
        if branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def branch_with_summary(self, branch_from_id: str, summary: str) -> str:
        """Branch + record what was on the abandoned path."""
        if branch_from_id and branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        return self.append_branch_summary(
            branch_from_id or "root", summary)

    def reset_leaf(self):
        """Reset leaf to null (next append becomes root)."""
        self._leaf_id = None

    # ------------------------------------------------------------------
    # Context building (tree → flat list for LLM)
    # ------------------------------------------------------------------

    def build_context(self) -> dict:
        """
        Build the flattened context for LLM consumption.

        Walks from leaf to root, handling compaction and branch summaries.

        Returns:
            {"messages": [...], "thinkingLevel": "...", "model": {...}|None}
        """
        entries = self.get_entries()
        if not entries:
            return {"messages": [], "thinkingLevel": "off", "model": None}

        # Build index
        by_id = self._by_id

        # Find leaf
        leaf = None
        if self._leaf_id is not None:
            leaf = by_id.get(self._leaf_id)
        if not leaf and entries:
            leaf = entries[-1]
        if not leaf:
            return {"messages": [], "thinkingLevel": "off", "model": None}

        # Walk leaf → root, collect path
        path = []
        current = leaf
        while current:
            path.insert(0, current)
            pid = current.get("parentId")
            current = by_id.get(pid) if pid else None

        # Extract settings along path
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

        # Build messages
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

        if compaction_entry:
            # Emit compaction summary first
            messages.append({
                "role": "user",
                "content": (
                    f"[Context Compaction - was {compaction_entry['tokensBefore']} tokens]\n"
                    f"{compaction_entry['summary']}"
                ),
                "_meta": "compaction",
            })

            # Find compaction position in path
            comp_idx = None
            for i, e in enumerate(path):
                if e.get("id") == compaction_entry.get("id"):
                    comp_idx = i
                    break

            kept_id = compaction_entry.get("firstKeptEntryId")
            # Emit messages before compaction starting from firstKeptEntryId
            if comp_idx is not None:
                keeping = False
                for i in range(comp_idx):
                    if path[i].get("id") == kept_id:
                        keeping = True
                    if keeping:
                        emit(path[i])
                # Emit messages after compaction
                for i in range(comp_idx + 1, len(path)):
                    emit(path[i])
        else:
            # No compaction — emit all message-type entries
            for entry in path:
                emit(entry)

        return {
            "messages": messages,
            "thinkingLevel": thinking_level,
            "model": model_info,
        }

    # ------------------------------------------------------------------
    # Session listing
    # ------------------------------------------------------------------

    @staticmethod
    def create(cwd: str = None, session_dir: Path = None) -> "SessionManager":
        """Factory: create a new session."""
        return SessionManager(cwd=cwd, session_dir=session_dir, persist=True)

    @staticmethod
    def open_session(filepath: str | Path) -> "SessionManager":
        """Factory: open an existing session file."""
        p = Path(filepath)
        # Try to extract cwd from header
        entries = _load_entries(p)
        cwd = None
        if entries:
            cwd = entries[0].get("cwd")
        return SessionManager(cwd=cwd, session_file=p, persist=True)

    @staticmethod
    def continue_recent(cwd: str = None) -> "SessionManager":
        """Factory: continue most recent session, or create new."""
        sdir = _get_session_dir(cwd)
        recent = _find_most_recent_session(sdir)
        if recent:
            return SessionManager.open_session(recent)
        return SessionManager.create(cwd=cwd)

    @staticmethod
    def in_memory(cwd: str = None) -> "SessionManager":
        """Factory: create in-memory session (no persistence)."""
        return SessionManager(cwd=cwd, persist=False)

    @staticmethod
    def list_sessions(cwd: str = None) -> list[dict]:
        """List all sessions for a working directory."""
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
                                break
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
        """List sessions across ALL working directories."""
        if not SESSIONS_DIR.exists():
            return []
        all_sessions = []
        for d in sorted(SESSIONS_DIR.iterdir()):
            if d.is_dir():
                all_sessions.extend(SessionManager.list_sessions(cwd=None))
                # Fix: list_sessions uses _get_session_dir which encodes cwd,
                # but here we want each subdir. Let's just scan directly.
                all_sessions = []  # reset, redo properly
                break

        # Proper implementation
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
    # FTS5 Full-Text Search Integration
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20,
               entry_type: str = None, role: str = None) -> list[dict]:
        """
        Search within this session using SQLite FTS5.

        Args:
            query: FTS5 search expression (supports AND/OR/NOT, phrase, prefix*)
            limit: max results
            entry_type: filter by type (message, compaction, ...)
            role: filter by message role (user, assistant)

        Returns:
            List of result dicts with entry_id, snippet (highlighted),
            rank, timestamp, etc.
        """
        try:
            from .search import SEARCH_INDEX
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
        """Rebuild the FTS5 index for this session."""
        try:
            from .search import SEARCH_INDEX
            return SEARCH_INDEX.rebuild_index(session_manager=self)
        except Exception as e:
            return f"Rebuild failed: {e}"

    @staticmethod
    def global_search(query: str, limit: int = 20,
                      cwd: str = None) -> list[dict]:
        """
        Search across ALL sessions using FTS5.

        Args:
            query: FTS5 search expression
            limit: max results per category
            cwd: optional filter by working directory

        Returns:
            Dict with 'entries' and 'sessions' keys.
        """
        try:
            from .search import SEARCH_INDEX
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
        """Return statistics about the search index."""
        try:
            from .search import SEARCH_INDEX
            return SEARCH_INDEX.get_stats()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def rebuild_global_search_index() -> str:
        """Rebuild the entire search index from all JSONL files."""
        try:
            from .search import SEARCH_INDEX
            return SEARCH_INDEX.rebuild_index()
        except Exception as e:
            return f"Global rebuild failed: {e}"

