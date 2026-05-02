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
# 引导流程：确保 agents 包可导入且 .env 已加载。
# 兼容 3 种运行模式：
#   1) python agents/core/s_full.py  （直接运行，__package__=None）
#   2) python -m agents.core.s_full    （模块模式）
#   3) from agents.core.s_full import ...  （包导入模式）
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.dirname(_THIS_DIR)       # agents/ 目录
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)     # 项目根目录（.env 所在位置）

# 将项目根目录和 agents 目录都加入 sys.path
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

# 在导入任何 agent 模块之前，先从项目根目录加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)                       # 先尝试当前工作目录
    if not os.getenv("GLM_API_KEY") and not os.getenv("ANTHROPIC_BASE_URL"):
        _env_file = os.path.join(_PROJECT_ROOT, ".env")
        if os.path.exists(_env_file):
            load_dotenv(_env_file, override=True)     # 回退到项目根目录
except ImportError:
    pass

# 现在可以安全导入 —— 使用绝对导入（因为 sys.path 已包含项目根目录）
from agents.agent.background import BackgroundManager
from agents.llm.compression import estimate_tokens, microcompact, auto_compact
from agents.core.config import (WORKDIR, TOKEN_THRESHOLD, PROVIDER, MODEL,
                               SKILLS_DIR)
from agents.llm.llm import stream_call_llm
from agents.agent.messaging import (MessageBus, handle_shutdown_request,
                                    handle_plan_review)
from agents.core.session import SessionManager
from agents.tools.skills import SkillLoader
from agents.agent.subagent import run_subagent, run_git_worker
from agents.data.tasks import TaskManager
from agents.agent.team import TeammateManager
from agents.data.todos import TodoManager
from agents.tools.tools import run_bash, run_read, run_write, run_edit
from agents.data.memory import MemoryBank, CATEGORIES
from agents.tools.git import (
    git_status, git_branch, git_branch_create, git_branch_checkout,
    git_commit, git_diff, git_log, git_pr,
)
from agents.agent.planner import PLANNER
from agents.data.learning import LearningEngine
from agents.llm.thought_chain import (
    ThoughtChainEngine, ThinkingMode,
)
from agents.llm.thought_tree import ThoughtTreeEngine
from agents.llm.thought_graph import ThoughtGraphEngine
from agents.llm.self_discover import SelfDiscoverEngine

# === 全局实例 ===
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)
TASK_MGR = TaskManager()
BG = BackgroundManager()
BUS = MessageBus()
TEAM = TeammateManager(BUS, TASK_MGR)

# 会话管理器（树形 JSONL）
SESSION = SessionManager.continue_recent()

# 长期记忆库
MEMORY = MemoryBank()

# 任务规划器（单例）
PLANNER_INST = PLANNER

# 自学习引擎（单例）
LEARNING = LearningEngine()

# 思维链引擎（单例）
THOUGHT = ThoughtChainEngine()

# 思维树引擎（单例）
TOT = ThoughtTreeEngine()

# 思维图引擎（单例）
GOT = ThoughtGraphEngine()

# Self-Discover 引擎（单例）
SELF_DISCOVER = SelfDiscoverEngine()


# === 系统提示词 ===
_memory_text = MEMORY.recall_for_prompt()
_learning_text = LEARNING.recall_for_prompt()
SYSTEM = f"""你是一个在 {WORKDIR} 工作的编程助手。使用工具来完成任务。
多步骤工作优先使用 task_create/task_update/task_list。简短清单用 TodoWrite。
用 task 调度子代理（可并行派发多个分析不同仓库）。用 load_skill 加载专业知识。
可用技能：{SKILLS.descriptions()}
{_memory_text}
{_learning_text}

## 自主推理（Self-Discovery）

你有 5 个思维推理工具，**根据问题复杂度自主选择，不需要用户指令**：

### 选择策略（按复杂度从低到高）

1. **`think` — 线性推理（CoT）**
   - **何时用**：需要系统性思考的单一问题（分析→假设→推导→验证→结论）
   - **场景**：代码逻辑分析、bug 根因推断、方案可行性判断
   - **成本**：1 次 LLM 调用，速度快

2. **`think_deep` — 深度推理（多轮 CoT）**
   - **何时用**：高复杂度决策，每步需要独立深入分析
   - **场景**：架构设计决策、性能瓶颈诊断、安全漏洞分析
   - **成本**：6~8 次 LLM 调用，较慢但更透彻

3. **`tot_explore` — 树状搜索（ToT）**
   - **何时用**：存在多个可行方案，需要探索对比找最优
   - **场景**：技术选型（数据库/框架/算法）、多种实现路径对比
   - **策略选择**：
     - `bfs`（默认）：适合方案数量明确时全面对比
     - `dfs`：适合需要深入追踪某条路径时
     - `best_first`：适合大规模搜索空间

4. **`got_explore` — 图状推理（GoT）**
   - **何时用**：问题有多维度、需要交叉验证或信息融合
   - **场景**：架构评审（多角度论证）、系统设计（多模块依赖）、风险评估
   - **模式选择**：
     - `diverge_converge`（默认）：发散探索 → 收敛结论
     - `parallel_validate`：多条路径独立验证后交叉比对
     - `iterative_refine`：逐步细化 + 反馈修正
     - `full_graph`：LLM 自主构建完整推理图

5. **`self_discover` — 自主发现推理结构（Self-Discover）⭐**
   - **何时用**：问题结构不明确、开放性强、难以预判需要哪些推理步骤
   - **场景**：开放式架构设计、跨领域问题分析、需要"先想清楚怎么想再想"的元推理场景
   - **核心能力**：
     - Stage 1: LLM 自主发现需要的推理模块（如"约束分析""边界检查""方案对比"...）
     - Stage 2: 将模块组织成 DAG，按依赖关系逐步执行，综合结论
   - **与 GoT 的区别**：GoT 的结构是人预设的模式（发散收敛/并行验证等），Self-Discover 的结构是 LLM 自己发现的
   - **成本**：较高（Stage1 + Stage2 多轮 LLM 调用），适合最复杂的开放性问题

### 关键规则
- **自动触发**：遇到上述场景直接调用对应工具，不要问用户
- **结论回注**：推理完成后结论会自动注入后续上下文，指导行动
- **不要过度推理**：简单任务（单文件修改、明显 bug fix）直接做，不用调用推理工具

## 自主决策规则（重要！）

你不需要等待用户命令，遇到以下情况应自主决策并执行：

1. **大目标/复杂项目 → 自动走 Planner 全自动流程**
   当用户的目标涉及多个文件、多个模块、或是一个完整项目时，
   立即调用 `planner_execute` 工具。Planner 会：
   - 自动将目标分解为子任务
   - 用 ThreadPoolExecutor 并行派发（动态 1~50 个 worker）
   - 每轮完成后自动扫描项目，发现新优化点
   - 循环迭代直到目标完全实现
   - 用户只需要说一句话，后续全自动运行

   判断标准：
   - 目标涉及 ≥2 个文件的修改
   - 目标包含多个独立功能点
   - 用户说"实现 XXX"、"写一个 XXX"、"构建 XXX 系统"
   - 本质上是一个"项目"而不是一个"改动"

2. **多仓库并行分析 → 同时派发多个 task**
   如果需要分析/对比多个代码仓库，同时调用多个 `task(Explore)` 工具，
   每个 prompt 指定不同的目录路径。它们会并行执行。

3. **单文件/简单任务 → 直接做**
   单一文件修改、单个 bug fix、小功能：直接写代码 + git_commit。

4. **不需要用户确认的环节**
   - 不需要问"要不要开始"、"要不要用 planner"、"要不要提交"
   - 这些全部自主决策

## 自我改进规则（重要！学习系统）

当你遇到以下情况时，必须调用 record_learning 工具记录经验：
1. **工具调用失败** — 命令报错、文件不存在、权限错误等 → entry_type=error
2. **用户明确纠正你的错误** → entry_type=learning, category=correction
3. **发现自己的知识缺口或认知错误** → entry_type=learning, category=knowledge_gap
4. **找到了比之前更好的实践方式** → entry_type=learning, category=best_practice
5. **用户提出了你当前无法满足的需求** → entry_type=feature

记录要简洁：summary 一句话概括，suggested_action 给出下次如何避免。
不需要每次都记录，只记录有价值的、可能再次遇到的经验。

输出要求：
- 直接输出内容，不要使用任何装饰性格式（如横线、边框、分隔线、emoji 装饰等）
- 不要用 ┌─┐ │ └─┘ 等 Unicode 制表符画框
- 回答简洁明了，不要过度格式化"""


# === 工具定义 ===
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
    # 会话工具
    "session_branch":   lambda **kw: _do_session_branch(kw.get("entry_id")),
    "session_list":     lambda **kw: _do_session_list(),
    "session_history":  lambda **kw: _do_session_history(),
    "session_switch":   lambda **kw: _do_session_switch(kw.get("selection")),
    "session_search":    lambda **kw: _do_session_search(kw.get("query"), kw.get("global", False), kw.get("limit", 20)),
    "session_search_stats": lambda **kw: _format_search_stats(),
    # Git 工具
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
    # Git Worker（完整工作流子代理）
    "git_worker":        lambda **kw: run_git_worker(kw["task"], kw.get("base_branch", "main")),
    # 规划器工具
    "planner_plan":       lambda **kw: PLANNER_INST.decompose(kw["goal"]),
    "planner_execute":    lambda **kw: PLANNER_INST.plan_and_execute(kw["goal"]),
    "planner_status":     lambda **kw: PLANNER_INST.get_status(),
    # 学习工具（s13：自我改进系统）
    "record_learning":   lambda **kw: LEARNING.record(
                               entry_type=kw.get("entry_type", "learning"),
                               category=kw.get("category", "correction"),
                               priority=kw.get("priority", "medium"),
                               area=kw.get("area", "general"),
                               summary=kw.get("summary", ""),
                               details=kw.get("details", ""),
                               suggested_action=kw.get("suggested_action", ""),
                               source=kw.get("source", "llm_judgment"),
                               related_files=kw.get("related_files"),
                               tags=kw.get("tags"),
                           ),
    "list_learnings":    lambda **kw: LEARNING.list_all(
                               entry_type=kw.get("entry_type"),
                               category=kw.get("category"),
                               status=kw.get("status"),
                               limit=kw.get("limit", 50),
                           ),
    "search_learnings":  lambda **kw: LEARNING.search(kw.get("query", ""), limit=kw.get("limit", 20)),
    "learning_stats":    lambda **kw: LEARNING.stats(),
    "update_learning":   lambda **kw: LEARNING.update_status(kw.get("entry_id"), kw.get("status")),
    "delete_learning":   lambda **kw: LEARNING.delete(kw.get("entry_id")),
    # --- 思维链工具 (CoT) ---
    "think":             lambda **kw: _do_think(kw.get("goal", ""), kw.get("context", ""), kw.get("mode", "explicit")),
    "think_deep":        lambda **kw: _do_think_deep(kw.get("goal", ""), kw.get("context", ""), kw.get("max_steps", 6)),
    # --- 思维树工具 (ToT) ---
    "tot_explore":        lambda **kw: _do_tot_explore(kw.get("goal", ""), kw.get("context", ""), kw.get("strategy", "bfs"), kw.get("max_depth", 4), kw.get("width", 3)),
    # --- 思维图工具 (GoT) ---
    "got_explore":        lambda **kw: _do_got_explore(kw.get("goal", ""), kw.get("context", ""), kw.get("mode", "diverge_converge"), kw.get("n_branches", 3)),
    # --- Self-Discover 工具 ---
    "self_discover":      lambda **kw: _do_self_discover(kw.get("goal", ""), kw.get("context", "")),
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
    # 会话工具
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
    # Git 工具
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
    {"name": "planner_execute", "description": "全自动执行流水线：目标分解 → 并行派发(1~50 worker) → 自动迭代优化 → 收敛停止。用户只需说一句话（如'写一个编译器'），Planner 自动拆分任务、并发执行、每轮扫描项目发现新优化点、循环直到目标完全实现。适用于任何大目标/完整项目的自主构建。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要实现的目标/需求（必填）。用户的一句话即可，如'写一个支持词法分析、语法分析和代码生成的编译器'"}
     }, "required": ["goal"]}},
    {"name": "planner_status", "description": "查看 Planner 的历史执行记录和统计信息。",
     "input_schema": {"type": "object", "properties": {}}},
    # --- 学习工具（s13）---
    {"name": "record_learning", "description": "记录一条学习经验到 .learnings/ 库。当工具调用失败、用户纠正你、发现知识缺口、找到更好实践、或用户提出能力诉求时调用。支持去重合并（相似条目自动 recurrence_count++）。",
     "input_schema": {"type": "object", "properties": {
         "entry_type": {"type": "string", "enum": ["learning", "error", "feature"], "description": "条目类型：learning(纠正/知识/实践)、error(错误)、feature(能力诉求)"},
         "category": {"type": "string", "description": "分类。learning: correction/knowledge_gap/best_practice；error: tool_error/command_error/lint_error/runtime_error；feature: capability_gap/workflow_request/integration_request"},
         "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "优先级（默认 medium）"},
         "area": {"type": "string", "enum": ["frontend", "backend", "infra", "tests", "docs", "config", "tooling", "general"], "description": "所属领域（默认 general）"},
         "summary": {"type": "string", "description": "一行摘要总结（必填，最重要字段）"},
         "details": {"type": "string", "description": "详细描述（发生了什么、原先哪里错了）"},
         "suggested_action": {"type": "string", "description": "下次如何避免或改进的建议动作"},
         "source": {"type": "string", "description": "来源：auto/user_feedback/tool_error/self_reflection/llm_judgment"},
         "related_files": {"type": "array", "items": {"type": "string"}, "description": "相关文件路径列表"},
         "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"}
     }, "required": ["summary"]}},
    {"name": "list_learnings", "description": "列出学习经验库中的条目，可按类型/分类/状态过滤。",
     "input_schema": {"type": "object", "properties": {
         "entry_type": {"type": "string", "enum": ["learning", "error", "feature"]},
         "category": {"type": "string"},
         "status": {"type": "string"},
         "limit": {"type": "integer"}
     }}},
    {"name": "search_learnings", "description": "按关键词搜索学习经验库。",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "搜索关键词"},
         "limit": {"type": "integer", "description": "返回结果上限（默认20）"}
     }, "required": ["query"]}},
    {"name": "learning_stats", "description": "查看学习经验库的统计信息（总数、类型分布、可升级条目等）。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "update_learning", "description": "更新学习条目的状态（如 resolved/promoted/wont_fix）。",
     "input_schema": {"type": "object", "properties": {
         "entry_id": {"type": "string", "description": "学习条目 ID（如 LRN-20260424-A1B2C3D4）"},
         "status": {"type": "string", "enum": ["pending", "in_progress", "resolved", "wont_fix", "promoted_to_memory", "promoted_to_skill"]}
     }, "required": ["entry_id", "status"]}},
    {"name": "delete_learning", "description": "删除一条学习记录。",
     "input_schema": {"type": "object", "properties": {
         "entry_id": {"type": "string", "description": "要删除的学习条目 ID"}
     }, "required": ["entry_id"]}},
    # --- 思维链工具 (CoT) ---
    {"name": "think", "description": "对问题进行显式推理（思维链）。生成结构化的思考步骤：分析→假设→推导→验证→结论。适用于需要系统性思考的复杂问题。单次 LLM 调用，速度快。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要推理的问题或目标（必填）"},
         "context": {"type": "string", "description": "额外的上下文信息（可选）"},
         "mode": {"type": "string", "enum": ["explicit", "implicit"], "description": "推理模式（默认 explicit）"}
     }, "required": ["goal"]}},
    {"name": "think_deep", "description": "深度推理模式（多轮思维链）。每步独立 LLM 调用：收集上下文→深度分析→提出假设→逻辑推导→验证检查→得出结论。适用于高复杂度决策场景。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要深入推理的问题或目标（必填）"},
         "context": {"type": "string", "description": "额外上下文（可选）"},
         "max_steps": {"type": "integer", "description": "最大推理步数（默认 6，最大 8）"}
     }, "required": ["goal"]}},
    # --- 思维树工具 (ToT) ---
    {"name": "tot_explore", "description": "思维树搜索（Tree of Thought）。对问题进行多候选探索：生成多个思路→LLM评估打分→选优→深入展开→回溯最优路径。支持3种策略：bfs(广度优先，适合方案对比)、dfs(深度优先，适合深入探索)、best_first(最佳优先，适合复杂规划)。适用于架构设计选型、算法选择、需要从多个方案中找最优解的场景。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要探索的问题或目标（必填）"},
         "context": {"type": "string", "description": "额外上下文信息（可选）"},
         "strategy": {"type": "string", "enum": ["bfs", "dfs", "best_first"], "description": "搜索策略（默认 bfs）"},
         "max_depth": {"type": "integer", "description": "最大搜索深度（默认 4）"},
         "width": {"type": "integer", "description": "每个节点展开的候选数/beam width（默认 3）"}
     }, "required": ["goal"]}},
    # --- 思维图工具 (GoT) ---
    {"name": "got_explore", "description": "思维图推理（Graph of Thought）。将问题建模为有向无环图（DAG）：分解为多个探索方向→并行推理→合并结论/交叉验证。支持4种模式：diverge_converge(发散-收敛，默认)、parallel_validate(并行验证，多路径交叉对比)、iterative_refine(迭代精炼，逐步细化+反馈修正)、full_graph(完全图)。适用于需要多角度论证、信息融合、交叉验证的复杂决策场景。GoT 与 ToT 的区别：ToT 是树状(单入度)，GoT 是图状(支持多入度合并)。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要分析的问题或目标（必填）"},
         "context": {"type": "string", "description": "额外上下文信息（可选）"},
         "mode": {"type": "string", "enum": ["diverge_converge", "parallel_validate", "iterative_refine", "full_graph"], "description": "推理模式（默认 diverge_converge）"},
         "n_branches": {"type": "integer", "description": "分解出的分支数（默认 3）"}
     }, "required": ["goal"]}},
    # --- Self-Discover 工具 ---
    {"name": "self_discover", "description": "Self-Discover 自主发现推理结构（Google DeepMind 2024）。不同于 CoT/ToT/GoT 的固定推理框架，Self-Discover 让 LLM 先自主「发现」解决该问题需要哪些推理模块（如：约束分析、边界检查、方案对比、复杂度评估），再将这些模块组织成 DAG 并按依赖关系逐步执行。适用于问题结构不明确、需要多维度推理的复杂场景。两阶段流程：Stage1 发现模块 → Stage2 组建DAG并执行 → 综合结论。",
     "input_schema": {"type": "object", "properties": {
         "goal": {"type": "string", "description": "要分析的问题或目标（必填）"},
         "context": {"type": "string", "description": "额外上下文信息（可选）"}
     }, "required": ["goal"]}},
]


# === 输出格式化 ===

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


def _serialize_content(content) -> list:
    """将 LLM 返回的 content 列表转为可 JSON 序列化的格式。

    ParsedText / ParsedToolUse 等对象有 .text / .name / .input 等属性，
    直接存会导致 __repr__ 被存入 JSONL（变成 '<... object at 0x...>'）。
    此函数提取对象的实际内容。
    """
    if not isinstance(content, list):
        return [str(content)]

    result = []
    for block in content:
        if hasattr(block, 'type'):
            # ParsedText 或 ParsedToolUse 对象
            item = {"type": block.type}
            if hasattr(block, 'text') and block.text:
                item["text"] = block.text
            if hasattr(block, 'name') and block.name:
                item["name"] = block.name
            if hasattr(block, 'input') and block.input:
                item["input"] = block.input
            if hasattr(block, 'id') and block.id:
                item["id"] = block.id
            result.append(item)
        elif isinstance(block, dict):
            result.append(block)
        elif isinstance(block, str):
            result.append({"type": "text", "text": block})
        else:
            result.append(str(block))
    return result


# === 思维链（CoT）工具处理函数 ===

def _do_think(goal: str, context: str = "", mode: str = "explicit") -> str:
    """执行显式思维链推理。"""
    if not goal:
        return "错误：需要提供推理目标（goal 参数）。用法：think(goal='你的问题')"
    try:
        chain = THOUGHT.think_explicit(goal, context=context, mode=mode)
        # 持久化到会话
        SESSION.append_thought_chain(chain.to_session_entry())
        # 记录模式切换（如果是第一次使用）
        if THOUGHT.get_mode() == ThinkingMode.OFF:
            THOUGHT.set_mode(ThinkingMode.EXPLICIT)
            SESSION.append_thinking_level_change("explicit", "首次使用 think 工具")
        # ---- 结论回注：将结论标记为待消费 ----
        if chain.conclusion:
            THOUGHT.set_pending_conclusion(chain.conclusion, goal=goal)
            print(f"  [CoT] 结论已准备回注到下一轮 Agent Loop ({len(chain.conclusion)} 字符)")
        return chain.display()
    except Exception as e:
        return f"思维链执行错误：{e}"


def _do_think_deep(goal: str, context: str = "", max_steps: int = 6) -> str:
    """执行深度思维链推理。"""
    if not goal:
        return "错误：需要提供推理目标（goal 参数）。用法：think_deep(goal='你的复杂问题')"
    try:
        print(f"  [CoT] 开始深度推理: {goal[:60]}... (最多 {max_steps} 步)")
        chain = THOUGHT.think_deeply(goal, context=context, max_steps=max_steps)
        # 持久化到会话
        SESSION.append_thought_chain(chain.to_session_entry())
        # 切换到 deep 模式
        if THOUGHT.get_mode() != ThinkingMode.DEEP:
            THOUGHT.set_mode(ThinkingMode.DEEP)
            SESSION.append_thinking_level_change("deep", f"切换到深度推理模式, goal={goal[:40]}")
        # ---- 结论回注：将结论标记为待消费 ----
        if chain.conclusion:
            THOUGHT.set_pending_conclusion(chain.conclusion, goal=goal)
            print(f"  [CoT] 深度推理结论已准备回注到下一轮 Agent Loop ({len(chain.conclusion)} 字符)")
        return chain.display()
    except Exception as e:
        return f"深度思维链执行错误：{e}"


def _do_set_think_mode(mode: str) -> str:
    """切换思维链模式。"""
    valid_modes = ["off", "implicit", "explicit", "deep"]
    if mode not in valid_modes:
        return f"无效模式: '{mode}'。可选值: {', '.join(valid_modes)}"
    try:
        new_mode = ThinkingMode(mode)
        old_mode = THOUGHT.get_mode()
        THOUGHT.set_mode(new_mode)
        # 记录到会话
        SESSION.append_thinking_level_change(
            mode, f"从 {old_mode.value} 切换到 {mode}"
        )
        # 如果开启隐式或显式模式，增强系统提示词
        global SYSTEM
        if mode in ("implicit", "explicit"):
            SYSTEM = THOUGHT.enhance_system_prompt(SYSTEM)
        elif mode == "off":
            # 重置系统提示词（移除思维链引导部分）
            _memory_text = MEMORY.recall_for_prompt()
            _learning_text = LEARNING.recall_for_prompt()
            SYSTEM = f"""你是一个在 {WORKDIR} 工作的编程助手。使用工具来完成任务。
多步骤工作优先使用 task_create/task_update/task_list。简短清单用 TodoWrite。
用 task 调度子代理（可并行派发多个分析不同仓库）。用 load_skill 加载专业知识。
可用技能：{SKILLS.descriptions()}
{_memory_text}
{_learning_text}

## 自主决策规则（重要！）

你不需要等待用户命令，遇到以下情况应自主决策并执行：

1. **大目标/复杂项目 → 自动走 Planner 全自动流程**
   当用户的目标涉及多个文件、多个模块、或是一个完整项目时，
   立即调用 `planner_execute` 工具。Planner 会：
   - 自动将目标分解为子任务
   - 用 ThreadPoolExecutor 并行派发（动态 1~50 个 worker）
   - 每轮完成后自动扫描项目，发现新优化点
   - 循环迭代直到目标完全实现
   - 用户只需要说一句话，后续全自动运行

   判断标准：
   - 目标涉及 ≥2 个文件的修改
   - 目标包含多个独立功能点
   - 用户说"实现 XXX"、"写一个 XXX"、"构建 XXX 系统"
   - 本质上是一个"项目"而不是一个"改动"

2. **多仓库并行分析 → 同时派发多个 task**
   如果需要分析/对比多个代码仓库，同时调用多个 `task(Explore)` 工具，
   每个 prompt 指定不同的目录路径。它们会并行执行。

3. **单文件/简单任务 → 直接做**
   单一文件修改、单个 bug fix、小功能：直接写代码 + git_commit。

4. **不需要用户确认的环节**
   - 不需要问"要不要开始"、"要不要用 planner"、"要不要提交"
   - 这些全部自主决策

## 自我改进规则（重要！学习系统）

当你遇到以下情况时，必须调用 record_learning 工具记录经验：
1. **工具调用失败** — 命令报错、文件不存在、权限错误等 → entry_type=error
2. **用户明确纠正你的错误** → entry_type=learning, category=correction
3. **发现自己的知识缺口或认知错误** → entry_type=learning, category=knowledge_gap
4. **找到了比之前更好的实践方式** → entry_type=learning, category=best_practice
5. **用户提出了你当前无法满足的需求** → entry_type=feature

记录要简洁：summary 一句话概括，suggested_action 给出下次如何避免。
不需要每次都记录，只记录有价值的、可能再次遇到的经验。

输出要求：
- 直接输出内容，不要使用任何装饰性格式（如横线、边框、分隔线、emoji 装饰等）
- 不要用 ┌─┐ │ └─┘ 等 Unicode 制表符画框
- 回答简洁明了，不要过度格式化"""

        return f"已切换思维模式: {old_mode.value} -> {mode}"
    except Exception as e:
        return f"切换模式失败: {e}"


# === 思维树（ToT）工具处理函数 ===

def _do_tot_explore(goal: str, context: str = "", strategy: str = "bfs",
                    max_depth: int = 4, width: int = 3) -> str:
    """执行思维树搜索。"""
    if not goal:
        return "错误：需要提供探索目标（goal 参数）。用法：tot_explore(goal='你的复杂问题')"
    valid_strategies = ["bfs", "dfs", "best_first"]
    if strategy not in valid_strategies:
        return f"无效策略: '{strategy}'。可选值: {', '.join(valid_strategies)}"
    try:
        print(f"  [ToT] 开始思维树搜索: {goal[:60]}... (策略={strategy}, "
              f"深度={max_depth}, 宽度={width})")
        tree = TOT.explore(
            goal=goal,
            context=context,
            strategy=strategy,
            max_depth=max_depth,
            width=width,
        )
        # 持久化到会话
        SESSION.append_thought_tree(tree.to_session_entry())
        # 结论回注（复用 CoT 的回注机制）
        if tree.conclusion:
            THOUGHT.set_pending_conclusion(tree.conclusion, goal=f"[ToT] {goal}")
            print(f"  [ToT] 搜索完成，结论已准备回注 ({len(tree.conclusion)} 字符)")
        print(f"  [ToT] 节点数: {tree.node_count} | 实际深度: {tree.depth} | "
              f"最优分数: {tree.best_score:.2f}")
        return tree.display()
    except Exception as e:
        return f"思维树搜索错误：{e}"


# === 思维图（GoT）工具处理函数 ===

def _do_got_explore(goal: str, context: str = "", mode: str = "diverge_converge",
                    n_branches: int = 3) -> str:
    """执行思维图推理。"""
    if not goal:
        return "错误：需要提供分析目标（goal 参数）。用法：got_explore(goal='你的复杂问题')"
    valid_modes = ["diverge_converge", "parallel_validate",
                   "iterative_refine", "full_graph"]
    if mode not in valid_modes:
        return f"无效模式: '{mode}'。可选值: {', '.join(valid_modes)}"
    try:
        print(f"  [GoT] 开始思维图推理: {goal[:60]}... (模式={mode}, "
              f"分支数={n_branches})")
        graph = GOT.explore(
            goal=goal,
            context=context,
            mode=mode,
            n_branches=n_branches,
        )
        # 持久化到会话
        SESSION.append_thought_graph(graph.to_session_entry())
        # 结论回注（复用 CoT 的回注机制）
        if graph.conclusion:
            THOUGHT.set_pending_conclusion(graph.conclusion, goal=f"[GoT] {goal}")
            print(f"  [GoT] 推理完成，结论已准备回注 ({len(graph.conclusion)} 字符)")
        print(f"  [GoT] 节点数: {graph.node_count} | 边数: {graph.edge_count} | "
              f"最大深度: {graph.max_depth} | 关键发现: {len(graph.key_findings)}")
        return graph.display()
    except Exception as e:
        return f"思维图推理错误：{e}"


def _do_self_discover(goal: str, context: str = "") -> str:
    """执行 Self-Discover 自主发现推理。"""
    if not goal:
        return "错误：需要提供分析目标（goal 参数）。用法：self_discover(goal='你的复杂问题')"
    try:
        print(f"  [Self-Discover] 开始自主推理: {goal[:60]}...")
        result = SELF_DISCOVER.discover(goal=goal, context=context)
        # 持久化到会话
        SESSION.append_message({
            "role": "assistant",
            "content": f"[Self-Discover] {result.summary}",
            "metadata": {"type": "self_discover", "session_id": result.session_id},
        })
        # 结论回注（复用 CoT 的回注机制）
        if result.conclusion and result.status == "completed":
            THOUGHT.set_pending_conclusion(result.conclusion, goal=f"[SD] {goal}")
            print(f"  [Self-Discover] 推理完成，结论已准备回注 "
                  f"({len(result.conclusion)} 字符)")
        print(f"  [Self-Discover] 模块: {result.completed_modules}/{result.module_count} | "
              f"耗时: {result.total_duration_ms}ms | 状态: {result.status}")
        return SELF_DISCOVER.display(result)
    except Exception as e:
        return f"Self-Discover 推理错误：{e}"


# === 会话感知辅助函数 ===

def _do_compact() -> str:
    """Perform compaction using the session manager's context."""
    ctx = SESSION.build_context()
    messages = ctx["messages"]
    if len(messages) <= 2:
        return "上下文不足，无法压缩。"
    tokens = estimate_tokens(messages)
    # 通过 LLM 生成摘要并存储为压缩记录
    conv_text = json.dumps(messages, default=str)[-80000:]
    from agents.llm.llm import call_llm as _call, parse_llm_response as _parse
    resp = _call(
        messages=[{"role": "user", "content": f"请对以下对话进行摘要，保持上下文连贯性：\n{conv_text}"}],
        max_tokens=2000,
    )
    resp = _parse(resp)
    summary = resp.content[0].text if resp.content else "（摘要生成失败）"
    # 找到第一条消息记录的 ID 作为 firstKeptEntryId
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
        # 显示可用记录列表
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
            # 根据类型显示预览
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
        # 相对时间
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


# 会话切换状态 —— 缓存列表以便 /switch <n> 可以按序号查找
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
        # 无参数 → 显示列表并提示用法
        _cached_sessions = SessionManager.list_sessions()
        if not _cached_sessions:
            return "没有可切换的会话。先用 /session_new 创建一个。"
        return _do_session_list() + '\n用法：/switch <序号> 或 /switch <会话ID前缀>'

    # 先尝试数字序号
    try:
        idx = int(selection)
        _cached_sessions = SessionManager.list_sessions()
        if 1 <= idx <= len(_cached_sessions):
            target = _cached_sessions[idx - 1]
            return _switch_to_session(target["path"], target["id"])
        return f"序号超出范围，有效范围：1-{len(_cached_sessions)}"
    except ValueError:
        pass

    # 尝试会话 ID 前缀匹配
    _cached_sessions = SessionManager.list_sessions()
    matches = [s for s in _cached_sessions if s["id"].startswith(selection)]
    if len(matches) == 1:
        return _switch_to_session(matches[0]["path"], matches[0]["id"])
    elif len(matches) > 1:
        ids = [s["id"][:12] for s in matches]
        return f"前缀 '{selection}' 匹配到多个会话：\n  " + "\n  ".join(ids)

    # 尝试作为文件路径
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
        # 更新全局变量，agent_loop / build_context 将使用新会话
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
        ts = r.get("timestamp", "?")[11:19]  # 仅取时间部分
        rank = r.get("rank", "?")
        marker = "◆" if eid == SESSION.leaf_id else " "
        lines.append(f"  {marker} \033[33m{eid}\033[0m [{etype}/{role}] rank={rank} {ts}")
        # 缩进显示的摘要片段
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

    # 先显示匹配的会话（分组视图）
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

    # 显示排名靠前的单条记录
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
                text = content[:50].replace("\n", " ")
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text" and block.get("text"):
                            parts.append(block["text"][:40])
                        elif btype == "tool_use":
                            name = block.get("name", "tool")
                            inp = block.get("input", {})
                            # 从 input 中提取关键信息
                            if isinstance(inp, dict):
                                keys = list(inp.keys())[:3]
                                detail = ", ".join(f"{k}={str(v)[:20]}" for k, v in zip(keys, [inp[k] for k in keys]))
                                parts.append(f"[{name}] {detail}")
                            else:
                                parts.append(f"[{name}]")
                        else:
                            parts.append(f"[{btype}]")
                    elif isinstance(block, str):
                        # 兼容旧数据："<... object at 0x...>"
                        if "<object at 0x" in block:
                            parts.append("(旧格式消息)")
                        else:
                            parts.append(block[:40])
                    else:
                        parts.append(str(block)[:40])
                text = " | ".join(parts) if parts else "(空)"
            else:
                text = str(content)[:50]
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

        # 绘制当前节点
        connector = "└─ " if (prefix == "" or is_last) else "├─ "
        lines = [f"{prefix}{connector}{marker}{eid}: {detail}{lbl}"]

        # 绘制子节点
        n = len(children)
        for i, child in enumerate(children):
            child_is_last = (i == n - 1)
            # 扩展前缀：添加竖线延续或空格
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
        # ---- CoT 结论回注：将思维链的结论注入到本轮上下文 ----
        # think/think_deep 执行完后结论会标记为 pending，
        # 这里消费它并作为 system 级上下文注入，让 LLM 在决策前"看到"推理结论。
        # 消费后立即清空，保证每条结论只注入一次。
        cot_conclusion = THOUGHT.consume_pending_conclusion()
        if cot_conclusion:
            print(f"  [CoT] 注入思维链结论到 Agent Loop ({len(cot_conclusion)} 字符)")
            messages.append({
                "role": "system",
                "content": cot_conclusion,
            })
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

        # Persist to session (tree JSONL) — 序列化前把 ParsedText 对象转为可存储格式
        if response.content:
            try:
                storable_content = _serialize_content(response.content)
                SESSION.append_message({"role": "assistant", "content": storable_content})
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
                # s13: auto-capture error signals (rule-based, zero token)
                if block.name not in ("record_learning", "list_learnings",
                                      "search_learnings", "learning_stats",
                                      "update_learning", "delete_learning"):
                    captured = LEARNING.auto_capture(
                        tool_name=block.name,
                        tool_input=block.input if hasattr(block, 'input') else {},
                        output=str(output),
                    )
                    if captured:
                        print(f"  [学习] 自动记录: {captured.entry_id} ({captured.summary[:60]})")
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

        # --- 上下文相关
        # Session commands (all /session_*) ---
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

        # --- 索引搜索相关
        #  Search commands ---
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

        # --- system 长期记忆相关
        # Memory commands (user-only, NOT LLM tools) ---
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

        # --- 计划相关
        # Planner commands ---
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

        # --- 自学习相关
        # Learning commands (s13: self-improvement) ---
        if cmd.startswith("/learnings") or cmd.startswith("/learning"):
            parts = cmd.strip().split(maxsplit=1)
            subcmd = parts[1] if len(parts) > 1 else None

            if not subcmd or subcmd == "list" or subcmd == "ls":
                print(LEARNING.list_all())

            elif subcmd.startswith("search "):
                q = subcmd[len("search "):].strip()
                limit_part = q.rsplit(maxsplit=1)
                if len(limit_part) == 2 and limit_part[-1].isdigit():
                    q, lim = limit_part[0], int(limit_part[1])
                    print(LEARNING.search(q, limit=lim))
                else:
                    print(LEARNING.search(q))

            elif subcmd == "stats":
                print(LEARNING.stats())

            elif subcmd in ("promote", "upgrade"):
                # Show promotable entries
                promotable = LEARNING.get_promotable()
                if not promotable:
                    print("当前没有可升级的条目（需要 recurrence_count >= 3）。")
                else:
                    lines = [f"可升级条目（共 {len(promotable)} 条）：", ""]
                    for e in promotable:
                        lines.append(f"  {e.entry_id}  [{e.category}/{e.priority}]  "
                                     f"{e.summary[:60]}  ({e.recurrence_count}次)")
                    lines.append("")
                    lines.append("用法：/learnings promote <entry_id>  升级到长期记忆")
                    print("\n".join(lines))

            elif subcmd.startswith("promote ") or subcmd.startswith("upgrade "):
                eid = subcmd.split(maxsplit=1)[1].strip()
                result = LEARNING.promote_to_memory(eid, memory_bank=MEMORY)
                print(result)

            elif (subcmd.startswith("update ") or subcmd.startswith("set ")) and len(subcmd.split()) >= 3:
                args = subcmd.split(maxsplit=2)
                eid, status = args[1], args[2]
                print(LEARNING.update_status(eid, status))

            elif (subcmd.startswith("delete ") or subcmd.startswith("del ") or subcmd.startswith("rm ")):
                eid = subcmd.split(maxsplit=1)[1].strip()
                print(LEARNING.delete(eid))

            elif subcmd == "help":
                print(
                    "学习经验库命令（s13 自我改进系统）：\n"
                    "  /learnings                  列出所有学习记录\n"
                    "  /learnings list             同上\n"
                    "  /learnings search <关键词>   按关键词搜索\n"
                    "  /learnings stats             统计信息（含可升级条目）\n"
                    "  /learnings promote           查看可升级到长期记忆的条目\n"
                    "  /learnings promote <ID>      将条目升级到 MemoryBank\n"
                    "  /learnings update <ID> <status>  更新状态(resolved/promoted/wont_fix)\n"
                    "  /learnings delete <ID>       删除一条记录\n"
                    "\n存储位置：.learnings/LEARNINGS.md, ERRORS.md, FEATURE_REQUESTS.md"
                )
            else:
                print(f"未知命令 '{subcmd}'。输入 /learnings help 查看帮助。")
            continue

        # Normal user input -> append to session + run agent loop (streaming)
        SESSION.append_message({"role": "user", "content": query})
        agent_loop([{"role": "user", "content": query}])
        # Note: text is already printed in real-time by stream_call_llm
        # No need to re-print here — just add a blank line for spacing
        print()
