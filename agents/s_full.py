#!/usr/bin/env python3
# 总控：所有机制合一 —— 模型的完整驾驶舱
"""
s_full.py - 完整参考 Agent（模块化版本）

综合 s01-s11 所有机制的集大成实现。
s12（任务感知 worktree 隔离）单独讲授。
这不是教学会话 —— 这是「全部整合」的参考实现。

    +------------------------------------------------------------------+
    |                        完 整 Agent                              |
    |                                                                   |
    |  系统提示词（s05 技能、任务优先 + 可选 todo 提醒）         |
    |                                                                   |
    |  每次 LLM 调用前：                                              |
    |  +--------------------+  +------------------+  +--------------+  |
    |  | 微压缩 (s06)      |  | 排空后台(s08)   |  | 检查收件箱 |  |
    |  | 自动压缩 (s06)    |  | 通知             |  | (s09)       |  |
    |  +--------------------+  +------------------+  +--------------+  |
    |                                                                   |
    |  工具调度（s02 模式）：                                         |
    |  +--------+----------+----------+---------+-----------+          |
    |  | bash   | read     | write    | edit    | TodoWrite |          |
    |  | task   | load_sk  | compress | bg_run  | bg_check  |          |
    |  | t_crt  | t_get    | t_upd    | t_list  | spawn_tm  |          |
    |  | list_tm| send_msg | rd_inbox | bcast   | shutdown  |          |
    |  | plan   | idle     | claim    |         |           |          |
    |  +--------+----------+----------+---------+-----------+          |
    |                                                                   |
    |  子代理 (s04)：  生成 -> 执行 -> 返回摘要                        |
    |  队友 (s09)：    生成 -> 执行 -> 空闲 -> 自动认领(s11)        |
    |  关闭 (s10)：    request_id 握手协议                             |
    |  计划审批(s10)：提交 -> 审批/驳回                           |
    |                                                                   |
    |  会话（树形 JSONL）：分支 / 压缩 / 持久化                 |
    +------------------------------------------------------------------+

模块说明：
    config.py      - 全局配置、Provider 初始化、常量
    llm.py         - LLM 抽象层（call_llm, parse_llm_response）
    tools.py       - 基础工具（bash, read, write, edit）
    todos.py       - TodoManager（s03）
    subagent.py    - 子代理生成（s04）
    skills.py      - 技能加载器（s05）
    compression.py - 上下文压缩（s06）
    tasks.py       - 任务管理器（s07）
    background.py  - 后台任务管理器（s08）
    messaging.py   - 消息总线 + 关闭/计划协议（s09/s10）
    team.py        - 队友管理器（s09/s11）
    session.py     - 树形 JSONL 会话管理器
    search.py      - SQLite FTS5 全文搜索索引（含 jieba 中文分词）
    memory.py      - 长期记忆库（JSONL，用户驱动的增删查）
    worktree.py     - Git Worktree 管理器（子代理隔离环境）
    planner.py      - 任务规划器（分解、派发、监控代码库健康度）

存储架构：
    JSONL (.jsonl)  → 主存储：树形结构、分支、压缩
    SQLite (.db)    → 搜索索引：FTS5 全文检索所有会话

REPL 命令：/compact /tasks /team /inbox /session_list /session_switch
            /session_new /session_history /session_branch /session_search
            /memory [/category]

直接运行：
    python agents/s_full.py
或作为模块：
    python -m agents.s_full
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: ensure agents package is importable and .env is loaded.
# Works for all 3 modes:
#   1) python agents/s_full.py        (direct, __package__=None)
#   2) python -m agents.s_full         (module mode)
#   3) from agents.s_full import ...  (package import)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)       # project root (where .env lives)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Load .env from project root BEFORE any agent module imports
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)                       # try CWD first
    if not os.getenv("GLM_API_KEY") and not os.getenv("ANTHROPIC_BASE_URL"):
        _env_file = os.path.join(_PARENT_DIR, ".env")
        if os.path.exists(_env_file):
            load_dotenv(_env_file, override=True)     # fallback to project root
except ImportError:
    pass

# Now safe to import — use absolute imports that work everywhere
from agents.background import BackgroundManager
from agents.compression import estimate_tokens, microcompact, auto_compact
from agents.config import (WORKDIR, TOKEN_THRESHOLD, PROVIDER, MODEL,
                           SKILLS_DIR)
from agents.llm import stream_call_llm
from agents.messaging import (MessageBus, handle_shutdown_request,
                               handle_plan_review)
from agents.session import SessionManager
from agents.skills import SkillLoader
from agents.subagent import run_subagent, run_git_worker
from agents.tasks import TaskManager
from agents.team import TeammateManager
from agents.todos import TodoManager
from agents.tools import run_bash, run_read, run_write, run_edit
from agents.memory import MemoryBank, CATEGORIES
from agents.git import (
    git_status, git_branch, git_branch_create, git_branch_checkout,
    git_commit, git_diff, git_log, git_pr,
)
from agents.planner import PLANNER

# === Global Instances ===
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)
TASK_MGR = TaskManager()
BG = BackgroundManager()
BUS = MessageBus()
TEAM = TeammateManager(BUS, TASK_MGR)

# Session Manager (tree JSONL)
SESSION = SessionManager.continue_recent()

# Long-term memory bank
MEMORY = MemoryBank()

# Task planner (singleton)
PLANNER_INST = PLANNER


# === System Prompt ===
_memory_text = MEMORY.recall_for_prompt()
SYSTEM = f"""你是一个在 {WORKDIR} 工作的编程助手。使用工具来完成任务。
多步骤工作优先使用 task_create/task_update/task_list。简短清单用 TodoWrite。
用 task 调度子代理。用 load_skill 加载专业知识。
可用技能：{SKILLS.descriptions()}
{_memory_text}

## 自主决策规则（重要！）

你不需要等待用户命令，遇到以下情况应自主决策并执行：

1. **复杂任务 → 自动走 Planner 流程**
   当用户的目标涉及多个文件、多个模块、或明显可拆分为独立子任务时，
   立即调用 `planner_execute` 工具，让 Planner 自动分解任务、派发 Git Worker 并行执行、收集结果。
   不需要问用户"要不要用 planner"，直接用。

   判断标准：
   - 目标涉及 ≥2 个文件的修改
   - 目标包含多个独立功能点
   - 用户说"实现 XXX 功能"、"重构 XXX 模块"、"添加 XXX 系统"
   - 任务预估需要多次 commit

2. **单文件/简单任务 → 直接做或用 git_worker**
   如果是单一文件的修改、单个 bug fix、小功能，直接自己写代码 + git_commit 即可。
   如果需要隔离环境（避免影响当前工作区），用 `git_worker` 派发一个子代理。

3. **不需要用户确认的环节**
   - 不需要问"要不要开始"
   - 不需要问"要不要创建分支"
   - 不需要问"要不要提交"
   - 不需要问"要不要用 planner"
   - 这些你都自己决定

输出要求：
- 直接输出内容，不要使用任何装饰性格式（如横线、边框、分隔线、emoji 装饰等）
- 不要用 ┌─┐ │ └─┘ 等 Unicode 制表符画框
- 回答简洁明了，不要过度格式化"""


# === Tool Definitions ===
TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "TodoWrite":        lambda **kw: TODO.update(kw["items"]),
    "task":             lambda **kw: run_subagent(kw["prompt"], kw.get("agent_type", "Explore")),
    "load_skill":       lambda **kw: SKILLS.load(kw["name"]),
    "compress":         lambda **kw: _do_compact(),
    "background_run":   lambda **kw: BG.run(kw["command"], kw.get("timeout", 120)),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
    "task_create":      lambda **kw: TASK_MGR.create(kw["subject"], kw.get("description", "")),
    "task_get":         lambda **kw: TASK_MGR.get(kw["task_id"]),
    "task_update":      lambda **kw: TASK_MGR.update(kw["task_id"], kw.get("status"),
                                                     kw.get("add_blocked_by"), kw.get("remove_blocked_by")),
    "task_list":        lambda **kw: TASK_MGR.list_all(),
    "spawn_teammate":   lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":   lambda **kw: TEAM.list_all(),
    "send_message":     lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":       lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":        lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "plan_approval":    lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":             lambda **kw: "主代理不进入空闲状态。",
    "claim_task":       lambda **kw: TASK_MGR.claim(kw["task_id"], "lead"),
    # Session tools
    "session_branch":   lambda **kw: _do_session_branch(kw.get("entry_id")),
    "session_list":     lambda **kw: _do_session_list(),
    "session_history":  lambda **kw: _do_session_history(),
    "session_switch":   lambda **kw: _do_session_switch(kw.get("selection")),
    "session_search":    lambda **kw: _do_session_search(kw.get("query"), kw.get("global", False), kw.get("limit", 20)),
    "session_search_stats": lambda **kw: _format_search_stats(),
    # Git tools
    "git_status":        lambda **kw: git_status(),
    "git_branch":        lambda **kw: git_branch(kw.get("list_all", False)),
    "git_branch_create":  lambda **kw: git_branch_create(kw["name"], kw.get("from_ref", "HEAD")),
    "git_branch_checkout": lambda **kw: git_branch_checkout(kw["name"]),
    "git_commit":        lambda **kw: git_commit(kw["message"], kw.get("add", True), kw.get("files", ".")),
    "git_diff":          lambda **kw: git_diff(kw.get("staged", False), kw.get("file", "")),
    "git_log":           lambda **kw: git_log(kw.get("max_count", 10), kw.get("oneline", True)),
    "git_pr":            lambda **kw: git_pr(
                               kw.get("title", ""), kw.get("body", ""),
                               kw.get("base", "main"), kw.get("draft", False)),
    # Git Worker (full workflow subagent)
    "git_worker":        lambda **kw: run_git_worker(kw["task"], kw.get("base_branch", "main")),
    # Planner tools
    "planner_plan":       lambda **kw: PLANNER_INST.decompose(kw["goal"]),
    "planner_execute":    lambda **kw: PLANNER_INST.plan_and_execute(kw["goal"]),
    "planner_status":     lambda **kw: PLANNER_INST.get_status(),
}

TOOLS = [
    {"name": "bash", "description": "执行 Shell 命令。",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "读取文件内容。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "写入文件内容。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "替换文件中的精确文本。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "TodoWrite", "description": "更新任务跟踪清单。",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "activeForm": {"type": "string"}}, "required": ["content", "status", "activeForm"]}}}, "required": ["items"]}},
    {"name": "task", "description": "生成子代理，用于隔离式探索或执行任务。",
     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "agent_type": {"type": "string", "enum": ["Explore", "general-purpose"]}}, "required": ["prompt"]}},
    {"name": "load_skill", "description": "按名称加载专业技能知识。",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compress", "description": "手动压缩对话上下文。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "background_run", "description": "在后台线程中运行命令。",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}},
    {"name": "check_background", "description": "检查后台任务状态。",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
    {"name": "task_create", "description": "创建持久化文件任务。",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_get", "description": "根据 ID 获取任务详情。",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "task_update", "description": "更新任务状态或依赖关系。",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]}, "add_blocked_by": {"type": "array", "items": {"type": "integer"}}, "remove_blocked_by": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "列出所有任务。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "spawn_teammate", "description": "生成持久化自主队友。",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "列出所有队友。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "向队友发送消息。",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string"}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "读取并排空主代理收件箱。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "向所有队友广播消息。",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "请求队友关闭。",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "plan_approval", "description": "审批或驳回队友的计划。",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "进入空闲状态。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "从任务板上认领任务。",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    # Session tools
    {"name": "session_branch", "description": "将会话分支到之前的某个节点（类似 git checkout）。不传参数时显示可用条目列表。",
     "input_schema": {"type": "object", "properties": {"entry_id": {"type": "string", "description": "要分支到的条目 ID。省略时显示可用条目列表。"}}}},
    {"name": "session_list", "description": "列出所有已保存的会话，带序号方便选择。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "session_history", "description": "显示当前会话历史（树形结构）。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "session_switch", "description": "通过序号或 ID 前缀切换到其他会话。建议先用 session_list 查看序号。",
     "input_schema": {"type": "object", "properties": {"selection": {"type": "string", "description": "会话序号（如 '1'）或 ID 前缀（如 '41b74290'）。"}}}},
    {"name": "session_search", "description": "使用 SQLite FTS5 对会话条目进行全文搜索。支持 AND、OR、NOT、phrase queries, and 前缀通配符s.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "FTS5 search expression. Examples: 'SQLite AND FTS5', '\"compac*\"', 'NEAR(\"session\" \"search\")', 'session OR branch'"},
         "global": {"type": "boolean", "description": "为 True 时搜索所有会话，为 False（默认）仅搜索当前会话。"},
         "limit": {"type": "integer", "description": "返回的最大结果数量（默认 20）。"}
     }, "required": ["query"]}},
    {"name": "session_search_stats", "description": "显示搜索索引的统计信息（总条目数、已索引会话数、数据库大小等）。",
     "input_schema": {"type": "object", "properties": {}}},
    # Git tools
    {"name": "git_status", "description": "查看 Git 工作区状态（已修改/已暂存/未跟踪文件等）。自动检测是否在 Git 仓库内。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "git_branch", "description": "列出所有本地分支，当前分支标记为 *。显示每个分支的最新 commit 信息。",
     "input_schema": {"type": "object", "properties": {
         "list_all": {"type": "boolean", "description": "是否包含远程分支（暂未实现，固定为本地分支）"}
     }}},
    {"name": "git_branch_create", "description": "创建新分支并切换到它。相当于 git checkout -b <name> [<from_ref>]。from_ref 默认为 HEAD。",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "新分支名"},
         "from_ref": {"type": "string", "description": "基于哪个 ref 创建（默认 HEAD）"}
     }, "required": ["name"]}},
    {"name": "git_branch_checkout", "description": "切换到已有的分支。相当于 git checkout <name>。",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "要切换到的目标分支名"}
     }, "required": ["name"]}},
    {"name": "git_commit", "description": "提交更改到 Git。先自动 git add（可通过 add=False 关闭），然后 git commit。返回 commit hash 和变更统计。",
     "input_schema": {"type": "object", "properties": {
         "message": {"type": "string", "description": "commit message（必填）"},
         "add": {"type": "boolean", "description": "是否先执行 git add（默认 true）"},
         "files": {"type": "string", "description": "要暂存的文件或路径（默认 '.' 全部）"}
     }, "required": ["message"]}},
    {"name": "git_diff", "description": "查看代码差异。staged=True 查看已暂存的差异(--cached)，否则查看未暂存的差异。可指定具体文件路径。",
     "input_schema": {"type": "object", "properties": {
         "staged": {"type": "boolean", "description": "查看已暂存区的差异（--cached）"},
         "file": {"type": "string", "description": "只看某个文件的差异"}
     }}},
    {"name": "git_log", "description": "查看 Git 提交历史。默认显示最近 10 条，每条一行(--oneline)，含分支标签。",
     "input_schema": {"type": "object", "properties": {
         "max_count": {"type": "integer", "description": "显示条数（默认 10）"},
         "oneline": {"type": "boolean", "description": "每条显示为一行（默认 true）"}
     }}},
    {"name": "git_pr", "description": "使用 gh CLI 创建 GitHub Pull Request。自动 push 当前分支到远程后用 gh pr create 创建 PR。需要 gh CLI 已安装并登录。不在 main/master 上时可用。工作区有未提交更改时会拒绝操作。",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string", "description": "PR 标题（留空则用最后一条 commit message）"},
         "body": {"type": "string", "description": "PR 描述内容"},
         "base": {"type": "string", "description": "目标分支（默认 main）"},
         "draft": {"type": "boolean", "description": "是否创建为 draft PR（默认 false）"}
     }}},
    {"name": "git_worker", "description": "启动一个 Git 工作流子代理，自动完成「创建分支 -> 写代码 -> commit -> 合并到 master」的全流程。子代理会自动处理冲突、运行测试、生成有意义的 commit message。适用于需要完整实现功能的任务。",
     "input_schema": {"type": "object", "properties": {
         "task": {"type": "string", "description": "任务描述，说明要做什么功能或修复什么 bug（必填）"},
         "base_branch": {"type": "string", "description": "合并到的目标分支（默认 main）"}
     }, "required": ["task"]}},
    {"name": "planner_plan", "description": "将复杂目标分解为可并行的子任务。返回 JSON 格式的任务列表，包含每个任务的 ID、描述、依赖关系和优先级。适用于大功能开发前的规划。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要分解的目标/需求（必填）"}
     }, "required": ["goal"]}},
    {"name": "planner_execute", "description": "完整执行「目标分解 -> 派发 Git Worker 子代理 -> 收集结果」的全流程。Planner 自动处理依赖关系、并行派发、监控执行状态。适用于让 AI 自主完成大型功能的开发和集成。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要实现的目标/需求（必填）"}
     }, "required": ["goal"]}},
    {"name": "planner_status", "description": "查看 Planner 的历史执行记录和统计信息。",
     "input_schema": {"type": "object", "properties": {}}},
]


# === Output Formatting ===

def _format_tool_output(output, max_len: int = 300) -> str:
    """格式化工具输出：压缩 JSON、截断长文本、去掉多余空白。"""
    text = str(output)
    # 尝试美化 JSON（如果是 JSON 字符串则压缩显示）
    text_stripped = text.strip()
    if text_stripped.startswith('{') or text_stripped.startswith('['):
        try:
            obj = json.loads(text_stripped)
            # 压缩为一行，避免大 JSON 占满屏幕
            text = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        except (json.JSONDecodeError, ValueError):
            pass
    # 截断过长输出
    if len(text) > max_len:
        text = text[:max_len] + f"... (共 {len(text)} 字符)"
    return text


# === Session-aware helpers ===

def _do_compact() -> str:
    """Perform compaction using the session manager's context."""
    ctx = SESSION.build_context()
    messages = ctx["messages"]
    if len(messages) <= 2:
        return "上下文不足，无法压缩。"
    tokens = estimate_tokens(messages)
    # Generate summary via LLM and store as compaction entry
    conv_text = json.dumps(messages, default=str)[-80000:]
    from agents.llm import call_llm as _call, parse_llm_response as _parse
    resp = _call(
        messages=[{"role": "user", "content": f"请对以下对话进行摘要，保持上下文连贯性：\n{conv_text}"}],
        max_tokens=2000,
    )
    resp = _parse(resp)
    summary = resp.content[0].text if resp.content else "（摘要生成失败）"
    # Find first message entry ID as firstKeptEntryId
    entries = SESSION.get_entries()
    kept_id = None
    for e in entries:
        if e.get("type") == "message":
            kept_id = e.get("id")
            break
    if kept_id:
        SESSION.append_compaction(summary, kept_id, tokens)
    return f"已压缩（{tokens} tokens → 摘要）。会话已在压缩点分支。"


def _do_session_branch(entry_id: str = None) -> str:
    """Branch to an entry or show available entries."""
    if not entry_id:
        # Show available entries
        entries = SESSION.get_entries()
        if not entries:
            return "当前会话没有任何条目。"
        lines = [f"当前会话：{SESSION.session_id}", f"当前叶节点：{SESSION.leaf_id}",
                 f"文件： {SESSION.session_file}", "", "Entries:"]
        for e in entries:
            eid = e.get("id", "?")
            etype = e.get("type", "?")
            pid = e.get("parentId") or "root"
            label = SESSION.get_label(eid)
            marker = " <- * LEAF" if eid == SESSION.leaf_id else ""
            lbl_str = f" [label: {label}]" if label else ""
            # Show preview based on type
            if etype == "message":
                msg = e.get("message", {})
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, str):
                    preview = content[:50].replace("\n", " ")
                elif isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            parts.append(p.get("text", "")[:30])
                    preview = " ".join(parts)[:50]
                else:
                    preview = "(...)"
                preview_str = f"  [{role}] {preview}"
            elif etype == "compaction":
                summary = e.get("summary", "")[:50]
                preview_str = f"  [compaction] {summary}..."
            elif etype == "branch_summary":
                summary = e.get("summary", "")[:50]
                preview_str = f"  [branch_summary] {summary}..."
            else:
                preview_str = f"  [{etype}]"
            lines.append(f"  {eid} (parent={pid}){marker}{lbl_str}")
            lines.append(f"    {preview_str}")
        lines.append("")
        lines.append("提示：使用 session_branch <entry_id> 回到之前的节点。")
        return "\n".join(lines)
    else:
        try:
            SESSION.branch(entry_id)
            ctx = SESSION.build_context()
            return f"已分支到 {entry_id}。当前上下文有 {len(ctx['messages'])} 条消息。下一条消息将开始新分支。"
        except ValueError as ex:
            return f"错误：{ex}"


def _do_session_list() -> str:
    """List all sessions with numbers for easy selection."""
    sessions = SessionManager.list_sessions()
    if not sessions:
        return "没有保存的会话。使用 /session_new 创建一个。"
    current_id = SESSION.session_id
    lines = [
        f"工作目录 {WORKDIR} 下的所有会话：",
        f"（使用：/switch <序号> 或 session_switch(number=<n>)）",
        "",
    ]
    for i, s in enumerate(sessions, 1):
        is_current = s["id"] == current_id
        marker = " ◆ CURRENT" if is_current else ""
        # Relative time
        mod_time = s.get("modified", "")
        lines.append(
            f"  \033[33m[{i}]\033[0m  {s['id'][:12]}  "
            f"msgs={s['messageCount']:>3}  "
            f"{mod_time}{marker}"
        )
        preview = s.get("firstMessage", "(empty)")
        lines.append(f"       {preview[:65]}")
    lines.append("")
    lines.append(f"共 {len(sessions)} 个会话")
    return "\n".join(lines)


# Session switch state — stores the list so /switch <n> can look up by index
_cached_sessions = []


def _do_session_switch(selection: str) -> str:
    """
    Switch to a different session.
    selection can be:
      - A number (1-based index from /sessions list)
      - A session ID (full or prefix)
      - A file path
    """
    global SESSION, _cached_sessions

    selection = (selection or "").strip()
    if not selection:
        # No arg → show list and prompt
        _cached_sessions = SessionManager.list_sessions()
        if not _cached_sessions:
            return "没有可切换的会话。先用 /session_new 创建一个。"
        return _do_session_list() + '\n用法：/switch <序号> 或 /switch <会话ID前缀>'

    # Try numeric index first
    try:
        idx = int(selection)
        _cached_sessions = SessionManager.list_sessions()
        if 1 <= idx <= len(_cached_sessions):
            target = _cached_sessions[idx - 1]
            return _switch_to_session(target["path"], target["id"])
        return f"序号超出范围，有效范围：1-{len(_cached_sessions)}"
    except ValueError:
        pass

    # Try session ID prefix match
    _cached_sessions = SessionManager.list_sessions()
    matches = [s for s in _cached_sessions if s["id"].startswith(selection)]
    if len(matches) == 1:
        return _switch_to_session(matches[0]["path"], matches[0]["id"])
    elif len(matches) > 1:
        ids = [s["id"][:12] for s in matches]
        return f"前缀 '{selection}' 匹配到多个会话：\n  " + "\n  ".join(ids)

    # Try as file path
    from pathlib import Path as _P
    p = _P(selection)
    if p.exists():
        return _switch_to_session(str(p), "???")

    return f"未找到会话：'{selection}'。使用 /session_list 查看可用会话。"


def _switch_to_session(filepath: str, session_id: str) -> str:
    """Core switch logic: reload SESSION global, return status message."""
    global SESSION
    old_id = SESSION.session_id
    old_file = SESSION.session_file

    try:
        new_session = SessionManager.open_session(filepath)
        SESSION = new_session
        # Update the global that agent_loop / build_context will use
        entries = new_session.get_entries()
        msg_count = sum(1 for e in entries if e.get("type") == "message")
        return (
            f"\033[32m✓ Switched session\033[0m\n"
            f"  {old_id[:12]} → {session_id[:12]}\n"
            f"  文件： {filepath}\n"
            f"  Messages: {msg_count}\n"
            f"  Leaf: {SESSION.leaf_id}\n"
            f"  （后续消息将追加到此会话的当前分支）"
        )
    except Exception as e:
        return f"\033[31m切换会话失败：\033[0m {e}"


def _do_session_search(query: str = None, global_search: bool = False,
                        limit: int = 20) -> str:
    """Search session entries using SQLite FTS5 full-text search."""
    if not query:
        return (
            "会话全文搜索（FTS5）\n"
            "用法：\n"
            "  /session_search <query>           — 搜索当前会话\n"
            "  /session_search global <query>     — 搜索所有会话\n"
            "  /session_search stats              — 显示索引统计\n"
            "  /session_search rebuild            — 从 JSONL 文件重建索引\n"
            "\n"
            "查询语法示例：\n"
            "  'SQLite AND FTS5'       与查询（AND）\n"
            "  'session OR branch'     或查询（OR）\n"
            "  '\"compac*\"'             前缀通配符\n"
            '  "\'\"full text\"\'"        短语精确匹配\n'
            "  'session NOT branch'    排除查询（NOT）\n"
        )

    try:
        if global_search:
            result = SessionManager.global_search(query, limit=limit)
            return _format_global_search_result(result, limit)
        else:
            results = SESSION.search(query, limit=limit)
            return _format_search_results(results, query)
    except Exception as e:
        return f"搜索错误： {e}"


def _format_search_results(results: list[dict], query: str) -> str:
    """Format single-session search results for display."""
    if not results:
        return f"当前会话未找到匹配结果：{query}"

    if results and results[0].get("error"):
        return f"搜索错误： {results[0]['error']}"

    lines = [
        f"搜索结果（当前会话： {SESSION.session_id[:12]}):",
        f"查询词： {query}",
        f"匹配数： {len(results)}",
        "",
    ]
    for r in results:
        eid = r.get("entry_id", "?")
        etype = r.get("entry_type", "?")
        role = r.get("role", "")
        snippet = r.get("snippet", "") or r.get("content", "（无内容）")[:120]
        ts = r.get("timestamp", "?")[11:19]  # just time
        rank = r.get("rank", "?")
        marker = "◆" if eid == SESSION.leaf_id else " "
        lines.append(f"  {marker} \033[33m{eid}\033[0m [{etype}/{role}] rank={rank} {ts}")
        # Indented snippet
        for snip_line in snippet.split("\n")[:3]:
            lines.append(f"      {snip_line.strip()[:100]}")
        lines.append("")

    lines.append("提示：可使用 session_switch 切换到对应条目，或用 session_branch <entry_id> 回溯。")
    return "\n".join(lines)


def _format_global_search_result(result: dict, limit: int) -> str:
    """Format cross-session search results for display."""
    entries = result.get("entries", [])
    sessions = result.get("sessions", [])

    if (not entries or entries[0].get("error")) and not sessions:
        query_str = result.get("entries", [{}])[0].get("error", "unknown") if entries else "no results"
        return f"全局搜索： {query_str}"

    lines = [
        "╔══════════════════════════════════════════════╗",
        "║     全局搜索结果（所有会话）                     ║",
        "╚══════════════════════════════════════════════╝",
        "",
    ]

    # Show matching sessions first (grouped view)
    if sessions:
        lines.append(f"--- 匹配的会话 ({len(sessions)}) ---")
        for s in sessions:
            sid = s.get("session_id", "?")[:12]
            count = s.get("match_count", 0)
            best_rank = s.get("best_rank", "?")
            cwd = s.get("cwd", "")[-40:]
            types = s.get("types", "")
            lines.append(
                f"  \033[33m{sid}\033[0m  matches={count}  "
                f"best_rank={best_rank}  cwd=...{cwd}  types={types}"
            )
        lines.append("")

    # Show top individual entries
    if entries:
        show_entries = entries[:limit]
        lines.append(f"--- 条目列表 ({len(show_entries)} of {len(entries)}) ---")
        for r in show_entries:
            eid = r.get("entry_id", "?")
            sid = r.get("session_id", "?")[:12]
            etype = r.get("entry_type", "?")
            role = r.get("role", "")
            snippet = r.get("snippet", "") or "（无内容）"
            ts = r.get("timestamp", "?")[11:19]
            rank = r.get("rank", "?")
            lines.append(f"  \033[33m{eid}\033[0m (sess:{sid}) [{etype}/{role}] rank={rank} {ts}")
            for snip_line in snippet.split("\n")[:2]:
                lines.append(f"      {snip_line.strip()[:100]}")
            lines.append("")

    lines.append("提示：使用 session_switch <会话ID前缀> 切换到匹配的会话。")
    return "\n".join(lines)


def _format_search_stats() -> str:
    """Format search index statistics for display."""
    stats = SessionManager.search_stats()
    if "error" in stats:
        return f"搜索索引错误： {stats['error']}"

    lines = [
        "┌─────────────────────────────────────┐",
        "│     搜索索引统计（FTS5）  │",
        "└─────────────────────────────────────┘",
        "",
        f"  总条目数：    {stats.get('total_entries', 0)}",
        f"  已索引会话数： {stats.get('sessions_indexed', 0)}",
        f"  工作目录数：        {stats.get('working_directories', 0)}",
        f"  数据库大小：    {stats.get('db_size_human', '?')}",
        f"  数据库路径：          {stats.get('db_path', '?')}",
        "",
        "  按类型：",
    ]
    for t, c in stats.get("by_type", {}).items():
        lines.append(f"    {t}: {c}")
    return "\n".join(lines)


def _do_session_history() -> str:
    """Show session tree structure with pretty Unicode box-drawing."""
    tree = SESSION.get_tree()
    if not tree:
        return "当前会话没有任何条目。"

    def _preview(entry):
        etype = entry.get("type", "?")
        if etype == "message":
            msg = entry.get("message", {})
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content[:40].replace("\n", " ")
            else:
                text = "(...)"
            return f"[{role}] {text}"
        elif etype == "compaction":
            return f"[compaction] {entry.get('summary', '')[:40]}..."
        elif etype == "branch_summary":
            return f"[branch_summary] {entry.get('summary', '')[:40]}..."
        else:
            return f"[{etype}]"

    def _render(node, prefix: str = "", is_last: bool = True):
        """
        Unified tree renderer using prefix-accumulation.
        prefix: the indentation string for this node's level
        is_last: whether this node is the last child of its parent
        """
        entry = node["entry"]
        eid = entry.get("id", "?")
        children = node.get("children", [])
        label = node.get("_label")

        is_current_leaf = (eid == SESSION.leaf_id)
        marker = "◆ " if is_current_leaf else "  "
        detail = _preview(entry)
        lbl = f"  🏷{label}" if label else ""

        # Draw this node
        connector = "└─ " if (prefix == "" or is_last) else "├─ "
        lines = [f"{prefix}{connector}{marker}{eid}: {detail}{lbl}"]

        # Draw children
        n = len(children)
        for i, child in enumerate(children):
            child_is_last = (i == n - 1)
            # Extend prefix: add vertical continuation or space
            child_prefix = prefix + ("   " if is_last else "│  ")
            lines.extend(_render(child, child_prefix, child_is_last))

        return lines

    result = [
        f"会话ID： {SESSION.session_id}",
        f"文件： {SESSION.session_file}",
        "",
    ]
    for root in tree:
        result.extend(_render(root))
    return "\n".join(result)


# === Agent Loop (now session-aware) ===

def agent_loop(messages: list):
    """Main agent loop with full tool dispatch and session integration."""
    rounds_without_todo = 0
    while True:
        # s06: compression pipeline
        microcompact(messages)
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            print("[触发自动压缩]")
            messages[:] = auto_compact(messages)
        # s08: drain background notifications
        notifs = BG.drain()
        if notifs:
            txt = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
            messages.append({"role": "user", "content": f"<background-results>\n{txt}\n</background-results>"})
        # s10: check lead inbox
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"})
        # LLM streaming call — print text as it arrives
        response = None
        for chunk, evt, data in stream_call_llm(
            system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        ):
            if evt == "text_delta":
                # Stream text to terminal in real-time
                print(chunk, end="", flush=True)
            elif evt == "done":
                response = data

        if response is None:
            return  # should not happen, but safety guard

        messages.append({"role": "assistant", "content": response.content})

        # Persist to session (tree JSONL)
        if response.content:
            try:
                SESSION.append_message({"role": "assistant", "content": response.content})
            except Exception:
                pass  # non-critical

        if response.stop_reason != "tool_use":
            return
        # Tool execution
        results = []
        used_todo = False
        manual_compress = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compress":
                    manual_compress = True
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"未知工具： {block.name}"
                except Exception as e:
                    output = f"错误：{e}"
                print(f"> {block.name}: {_format_tool_output(output)}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                if block.name == "TodoWrite":
                    used_todo = True
        # s03: nag reminder
        rounds_without_todo = 0 if used_todo else rounds_without_todo + 1
        if TODO.has_open_items() and rounds_without_todo >= 3:
            results.append({"type": "text", "text": "<reminder>请更新你的待办事项。</reminder>"})
        messages.append({"role": "user", "content": results})
        # s06: manual compress
        if manual_compress:
            print("[手动压缩]")
            messages[:] = auto_compact(messages)
            return


# === REPL ===

if __name__ == "__main__":
    # Resume or create session
    print(f"会话ID： {SESSION.session_id}")
    print(f"文件： {SESSION.session_file}")
    print(f"工作目录：{WORKDIR}")
    print(f"提供商：{PROVIDER} / 模型：{MODEL}")
    print()

    while True:
        try:
            query = input("\033[36mAgentclaw >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # REPL commands
        cmd = query.strip()

        if cmd == "/compact":
            if any(True for m in SESSION.get_entries() if m.get("type") == "message"):
                print("[manual compact via /compact]")
                ctx = SESSION.build_context()
                messages = ctx["messages"]
                if len(messages) > 2:
                    messages[:] = auto_compact(messages)
                else:
                    print("内容不足，无法压缩。")
            continue

        if cmd == "/tasks":
            print(TASK_MGR.list_all())
            continue

        if cmd == "/team":
            print(TEAM.list_all())
            continue

        if cmd == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue

        # --- Session commands (all /session_*) ---
        if cmd == "/session_list":
            print(_do_session_list())
            continue

        if cmd.startswith("/session_switch"):
            arg = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else None
            print(_do_session_switch(arg))
            continue

        if cmd == "/session_history":
            print(_do_session_history())
            continue

        if cmd.startswith("/session_branch"):
            arg = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else None
            print(_do_session_branch(arg))
            continue

        if cmd == "/session_new":
            SESSION._new_session()
            print(f"新会话：{SESSION.session_id}")
            continue

        # --- Search commands ---
        if cmd.startswith("/session_search "):
            arg = cmd[len("/session_search "):].strip()
            if arg.lower() == "stats":
                print(_format_search_stats())
            elif arg.lower() == "rebuild":
                print(SessionManager.rebuild_global_search_index())
            elif arg.lower().startswith("global "):
                q = arg[7:].strip()
                print(_do_session_search(q, global_search=True))
            else:
                print(_do_session_search(arg))
            continue

        if cmd == "/session_search":
            print(_do_session_search())
            continue

        # --- Memory commands (user-only, NOT LLM tools) ---
        if cmd.startswith("/memory"):
            parts = cmd.strip().split(maxsplit=2)
            subcmd = parts[1] if len(parts) > 1 else None
            subarg = parts[2] if len(parts) > 2 else None

            if not subcmd or subcmd == "list":
                cat = subarg.strip() if subarg else None
                print(MEMORY.list_all(category=cat))

            elif subcmd == "push" or subcmd == "record" or subcmd == "add":
                if not subarg:
                    print("用法：/memory push <category> <内容>\n"
                          f"可用分类：{', '.join(CATEGORIES)}\n"
                          "示例：/memory push preference 我喜欢 Rust 不喜欢 Go")
                    continue
                mem_parts = subarg.split(maxsplit=1)
                if len(mem_parts) < 2:
                    print(f"错误：需要指定分类和内容。用法：/memory push <分类> <内容>")
                    continue
                cat, content = mem_parts[0], mem_parts[1]
                result = MEMORY.remember(cat, content)
                print(result)

            elif subcmd == "delete" or subcmd == "del" or subcmd == "rm":
                if not subarg:
                    print("用法：/memory delete <ID>\n先用 /memory list 查看记忆 ID")
                    continue
                print(MEMORY.delete(subarg.strip()))

            elif subcmd in ("stats", "stat", "info"):
                cats = MEMORY.categories
                print(f"┌────────────────────────────┐")
                print(f"│     长期记忆统计              │")
                print(f"└────────────────────────────┘")
                print(f"  总条目数：  {MEMORY.count}")
                print(f"  分类：      {', '.join(cats) if cats else '(无)'}")
                print(f"  文件：      {MEMORY._filepath}")

            elif subcmd == "help":
                print(
                    "长期记忆命令（用户直接操作，非 LLM）：\n"
                    "  /memory list [category]   列出记忆，可按分类过滤\n"
                    "  /memory push <cat> <内容> 存一条新记忆\n"
                    "  /memory del <ID>           按 ID 删除\n"
                    "  /memory stats              统计信息\n"
                    f"\n可用分类：{', '.join(CATEGORIES)}"
                )
            else:
                print(f"未知命令 '{subcmd}'。输入 /memory help 查看帮助。")
            continue

        # --- Planner commands ---
        if cmd.startswith("/plan"):
            parts = cmd.strip().split(maxsplit=1)
            goal = parts[1] if len(parts) > 1 else None
            if not goal:
                print(
                    "任务规划器命令：\n"
                    "  /plan <目标>              分解目标并执行完整流程\n"
                    "  /plan decompose <目标>    仅分解任务，不执行\n"
                    "  /plan status             查看历史记录\n"
                    "  /plan help               显示帮助\n"
                    "\n示例：\n"
                    "  /plan 给项目添加完整的错误处理和日志系统\n"
                    "  /plan decompose 实现用户认证模块"
                )
                continue
            if goal.lower() == "status":
                print(PLANNER_INST.get_status())
            elif goal.lower().startswith("decompose "):
                sub_goal = goal[len("decompose "):].strip()
                tasks = PLANNER_INST.decompose(sub_goal)
                import json as _json
                print(_json.dumps(tasks, ensure_ascii=False, indent=2))
            elif goal.lower() == "help":
                print(
                    "任务规划器 — 自主分解任务、派发 Worker、监控结果\n"
                    "\n命令：\n"
                    "  /plan <目标>              完整流程：分解→派发→监控→报告\n"
                    "  /plan decompose <目标>    仅查看分解结果（JSON）\n"
                    "  /plan status             历史执行记录\n"
                    "\n工作原理：\n"
                    "  1. LLM 将目标拆分为可并行的子任务\n"
                    "  2. 每个子任务派发给一个 Git Worker（独立 worktree）\n"
                    "  3. Worker 自行处理冲突、编译错误等\n"
                    "  4. Planner 汇总所有结果，发现遗留问题"
                )
            else:
                # Full plan and execute
                result = PLANNER_INST.plan_and_execute(goal)
                print(result)
            continue

        # Normal user input -> append to session + run agent loop (streaming)
        SESSION.append_message({"role": "user", "content": query})
        agent_loop([{"role": "user", "content": query}])
        # Note: text is already printed in real-time by stream_call_llm
        # No need to re-print here — just add a blank line for spacing
        print()
