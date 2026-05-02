# agents/ - Harness implementations (s01-s12) + full reference (s_full)
# Each file is self-contained and runnable: python agents/s_full.py
# The model is the agent. These files are the harness.
#
# 目录结构：
#   core/    - 核心模块（config, session, s_full 总控）
#   llm/     - LLM 抽象层（llm 调用, compression 压缩）
#   tools/   - 工具集（bash/read/write/edit, git, skills 加载）
#   agent/   - Agent 管理（subagent, planner, worktree, team, background, messaging）
#   data/    - 数据存储（search, memory, learning, tasks, todos）
#   before/  - 教学示例（s01-s12 分步讲解）

from agents.agent.background import BackgroundManager
from agents.agent.messaging import MessageBus
from agents.agent.planner import Planner, PLANNER
from agents.agent.subagent import run_subagent, run_git_worker
from agents.agent.team import TeammateManager
from agents.agent.worktree import WorktreeManager, WORKTREES_DIR, WORKTREES_INDEX
# 便捷重导出 —— 兼容旧的 `from agents.xxx import ...` 写法
from agents.core.config import (
    WORKDIR, PROVIDER, MODEL, client,
    TEAM_DIR, INBOX_DIR, TASKS_DIR, SKILLS_DIR,
    SESSIONS_DIR, TRANSCRIPT_DIR,
    TOKEN_THRESHOLD, POLL_INTERVAL, IDLE_TIMEOUT,
    VALID_MSG_TYPES, _init_provider,
)
from agents.core.session import SessionManager
from agents.data.learning import LearningEngine
from agents.data.memory import MemoryBank, CATEGORIES
from agents.data.search import SearchIndex, SEARCH_INDEX
from agents.data.tasks import TaskManager
from agents.data.todos import TodoManager
from agents.llm.compression import estimate_tokens, microcompact, auto_compact
from agents.llm.llm import call_llm, parse_llm_response, stream_call_llm, ParsedToolUse
from agents.llm.thought_chain import (
    ThoughtChainEngine, ThoughtChain, ThoughtStep,
    ThoughtType, ThinkingMode, THOUGHT_ENGINE,
)
from agents.llm.thought_graph import (
    ThoughtGraphEngine, ThoughtGraph, GraphNode, GraphEdge,
    GraphMode, NodeType, GOT_ENGINE,
)
from agents.llm.thought_tree import (
    ThoughtTreeEngine, ThoughtTree, ThoughtTreeNode,
    SearchStrategy, NodeStatus, TOT_ENGINE,
)
from agents.tools.git import (
    git_status, git_branch, git_branch_create, git_branch_checkout,
    git_commit, git_diff, git_log, git_pr,
)
from agents.tools.skills import SkillLoader
from agents.tools.tools import run_bash, run_read, run_write, run_edit, safe_path
