#!/usr/bin/env python3
"""Subagent spawning (s04) - isolated exploration or work tasks.

Supports agent types:
  - Explore:   read-only exploration (bash + read_file)
  - general-purpose: full edit capability (adds write_file + edit_file)
  - GitWorker: git workflow agent with self-healing (branch -> code -> commit ->
              resolve conflicts -> rebase -> merge -> cleanup)

Git Worker Self-Healing:
  - If merge/rebase has CONFLICTS: the worker reads conflicted files,
    fixes them, and continues rebase until success.
  - If build/test FAILS during development: the worker fixes errors
    and retries — build success is part of "task complete".
  - The worker NEVER gives up on its own bugs.
"""

from ..agent.worktree import WORKTREES
from ..llm.llm import call_llm, parse_llm_response
from ..tools.tools import run_bash, run_read, run_write, run_edit


def run_subagent(prompt: str, agent_type: str = "Explore") -> str:
    """Spawn a subagent with a limited tool set, return summary."""
    sub_tools = [
        {"name": "bash", "description": "Run command.",
         "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"name": "read_file", "description": "Read file.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    ]
    if agent_type != "Explore":
        sub_tools += [
            {"name": "write_file", "description": "Write file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Edit file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
        ]
    sub_handlers = {
        "bash": lambda **kw: run_bash(kw["command"]),
        "read_file": lambda **kw: run_read(kw["path"]),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
        "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    }
    sub_msgs = [{"role": "user", "content": prompt}]
    resp = None
    for _ in range(30):
        resp = call_llm(messages=sub_msgs, tools=sub_tools, max_tokens=8000)
        resp = parse_llm_response(resp)
        sub_msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                h = sub_handlers.get(b.name, lambda **kw: "Unknown tool")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": str(h(**b.input))[:50000]})
        sub_msgs.append({"role": "user", "content": results})
    if resp:
        return "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"
    return "(subagent failed)"


# ------------------------------------------------------------------
# Git Worker: worktree-isolated branch -> code -> commit -> merge
# With self-healing: conflict resolution + build fix loops
# ------------------------------------------------------------------

GIT_WORKER_SYSTEM_TEMPLATE = """你是一个自治的 Git 工作流代理。你在隔离的工作目录(worktree)中工作。

## 你的环境
- Worktree 名称：{wt_name}
- Worktree 路径：{wt_path}
- 当前分支：{wt_branch}（自动创建好的）
- 合并目标：{target_branch}

## 核心原则（必须遵守）
1. **自愈性**：你产生的任何问题（编译错误、测试失败、合并冲突）都必须由你自己修复。
   这不是可选的——修复这些问题是"完成任务"定义的一部分。
2. **不放弃**：如果编译失败，修代码重试。如果合并冲突，读冲突标记，手动合并，继续 rebase。
3. **质量优先**：不要为了"完成"而提交有 bug 的代码。确保构建通过、测试通过。

## 工作流程

### Phase 1: 了解现状
1. 先用 git_status 检查当前状态
2. 用 read_file 阅读需要修改的文件，理解现有代码

### Phase 2: 开发
3. 用 write_file 或 edit_file 修改/创建文件
4. 用 bash 运行测试或构建命令验证改动
5. **如果有错误 → 修复 → 重新验证（循环直到通过）**

### Phase 3: 提交
6. 用 git_diff 查看变更内容
7. 用 git_commit 提交（commit message 要简洁明了，中文）
8. 用 git_log 确认提交成功

### Phase 4: 合并（系统自动触发）
9. 系统会尝试将你的分支 rebase 到目标分支
10. **如果有冲突 → 你会进入冲突解决模式 → 手动解决每个冲突文件 → 继续 rebase**
11. 冲突全部解决后，系统自动清理 worktree

## 重要规则
- 你已经在独立的 worktree 分支上了，**不要**创建新分支或切换分支
- 每次 commit message 必须有意义
- 不要 push 到远程除非任务明确要求
- 所有文件操作都在当前 worktree 目录下进行
"""


# Conflict resolution tools (only available during conflict phase)
CONFLICT_TOOLS = [
    {"name": "read_conflict_file", "description": "读取仓库根目录下有冲突的文件内容（含冲突标记 <<<<<<< ======= >>>>>>>）。",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string", "description": "冲突文件的相对路径"}
     }, "required": ["file_path"]}},
    {"name": "resolve_conflict", "description": "写入解决后的文件内容（自动 git add）。用此工具替代 write_file 来解决冲突。",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string", "description": "冲突文件的相对路径"},
         "content": {"type": "string", "description": "解决冲突后的完整文件内容（不含冲突标记）"}
     }, "required": ["file_path", "content"]}},
    {"name": "accept_ours", "description": "接受我们的版本（保留 worktree 的更改），自动 git add。",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string", "description": "冲突文件的相对路径"}
     }, "required": ["file_path"]}},
    {"name": "accept_theirs", "description": "接受对方的版本（保留目标分支的更改），自动 git add。",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string", "description": "冲突文件的相对路径"}
     }, "required": ["file_path"]}},
    {"name": "continue_rebase", "description": "所有冲突文件都解决后，调用此命令继续 rebase。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_conflicts", "description": "列出当前所有还有冲突的文件。",
     "input_schema": {"type": "object", "properties": {}}},
]


def _make_git_worker_handlers(wt_name: str) -> dict:
    """Create tool handlers that execute inside a specific worktree.

    This is the key isolation mechanism — every handler uses the worktree's
    directory as its working root instead of the main WORKDIR.
    """
    wt_info = WORKTREES.get(wt_name)
    if not wt_info:
        return {}
    wt_path_str = wt_info["path"]

    def _run_bash_isolated(command: str) -> str:
        return WORKTREES.run_in(wt_name, command)

    def _run_read_isolated(path: str, limit=None) -> str:
        from pathlib import Path as _P
        full = (_P(wt_path_str) / path).resolve()
        if not full.is_relative_to(_P(wt_path_str)):
            return f"Error: Path escapes worktree: {path}"
        try:
            lines = full.read_text().splitlines()
            if limit and limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
            return "\n".join(lines)[:50000]
        except Exception as e:
            return f"Error: {e}"

    def _run_write_isolated(path: str, content: str) -> str:
        from pathlib import Path as _P
        full = (_P(wt_path_str) / path).resolve()
        if not full.is_relative_to(_P(wt_path_str)):
            return f"Error: Path escapes worktree: {path}"
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return f"Wrote {len(content)} bytes to {path} (in worktree)"
        except Exception as e:
            return f"Error: {e}"

    def _run_edit_isolated(path: str, old_text: str, new_text: str) -> str:
        from pathlib import Path as _P
        full = (_P(wt_path_str) / path).resolve()
        if not full.is_relative_to(_P(wt_path_str)):
            return f"Error: Path escapes worktree: {path}"
        try:
            c = full.read_text()
            if old_text not in c:
                return f"Error: Text not found in {path}"
            full.write_text(c.replace(old_text, new_text, 1))
            return f"Edited {path} (in worktree)"
        except Exception as e:
            return f"Error: {e}"

    def _git_status_wt() -> str:
        return WORKTREES.run_in(wt_name, "git status --short --branch")

    def _git_branch_wt() -> str:
        return WORKTREES.run_in(wt_name, "git branch -v --no-color")

    def _git_commit_wt(message: str) -> str:
        msg = message.strip()
        if not msg:
            return "Error: commit message is required."
        r1 = WORKTREES.run_in(wt_name, "git add -A")
        r2 = WORKTREES.run_in(wt_name, 'git diff --cached --stat')
        if not r2.strip():
            return "No changes to commit (working tree clean)."
        # Escape double quotes in message for shell safety
        safe_msg = msg.replace('"', '\\"').replace("'", "'\\''")
        r3 = WORKTREES.run_in(wt_name, f'git commit -m "{safe_msg}"')
        r4 = WORKTREES.run_in(wt_name, "git log --oneline -1")
        return f"Committed!\n{r4}\n{r2}"

    def _git_diff_wt(staged=False, file="") -> str:
        cmd = "git diff --cached" if staged else "git diff"
        if file:
            cmd += f" -- {file}"
        r = WORKTREES.run_in(wt_name, cmd)
        if not r.strip():
            return "No differences." + (" (staged empty)" if staged else " (clean)")
        return r[:20000] if len(r) > 20000 else r

    def _git_log_wt(max_count=10) -> str:
        return WORKTREES.run_in(wt_name, f"git log --oneline --no-color -{max_count}")

    return {
        "bash":             lambda **kw: _run_bash_isolated(kw["command"]),
        "read_file":        lambda **kw: _run_read_isolated(kw["path"], kw.get("limit")),
        "write_file":       lambda **kw: _run_write_isolated(kw["path"], kw["content"]),
        "edit_file":        lambda **kw: _run_edit_isolated(kw["path"], kw["old_text"], kw["new_text"]),
        "git_status":       lambda **kw: _git_status_wt(),
        "git_branch":       lambda **kw: _git_branch_wt(),
        "git_commit":       lambda **kw: _git_commit_wt(kw["message"]),
        "git_diff":         lambda **kw: _git_diff_wt(kw.get("staged", False), kw.get("file", "")),
        "git_log":          lambda **kw: _git_log_wt(kw.get("max_count", 10)),
    }


def _make_conflict_handlers():
    """Create handlers for the conflict resolution phase.

    These operate on repo_root (where the rebase is happening), not inside
    the worktree. They give the LLM ability to read/write conflicted files
    and control the rebase process.
    """
    return {
        "read_conflict_file": lambda **kw: WORKTREES.get_file_conflict_content(kw["file_path"]),
        "resolve_conflict":   lambda **kw: WORKTREES.write_resolved_file(kw["file_path"], kw["content"]),
        "accept_ours":        lambda **kw: WORKTREES._resolve_file_in_repo_root(kw["file_path"], strategy="ours"),
        "accept_theirs":      lambda **kw: WORKTREES._resolve_file_in_repo_root(kw["file_path"], strategy="theirs"),
        "continue_rebase":    lambda **kw: WORKTREES._continue_rebase(),
        "list_conflicts":     lambda **kw: (
            "Conflicted files:\n" +
            "\n".join(f"  - {f}" for f in WORKTREES._get_conflicted_files())
            or "  (no conflicts remaining)"
        ),
    }


# Tool definitions for Git Worker (development phase)
GIT_WORKER_TOOLS = [
    {"name": "bash", "description": "执行 Shell 命令（测试、构建等）。在 worktree 隔离环境中运行。如果构建/测试失败，必须修复后重试。",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "读取文件内容（相对于 worktree 根目录）。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "写入/创建文件（相对于 worktree 根目录）。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "替换文件中的精确文本（在 worktree 中）。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "git_status", "description": "查看 Git 工作区状态（worktree 内）。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "git_branch", "description": "列出本地分支。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "git_commit", "description": "暂存并提交更改（git add -A + commit）。",
     "input_schema": {"type": "object", "properties": {
         "message": {"type": "string", "description": "commit message（必填，简洁有意义的中文）"}
     }, "required": ["message"]}},
    {"name": "git_diff", "description": "查看代码差异。staged=True 看已暂存的差异。",
     "input_schema": {"type": "object", "properties": {
         "staged": {"type": "boolean", "description": "查看已暂存区的差异"},
         "file": {"type": "string", "description": "只看某个文件的差异"}
     }}},
    {"name": "git_log", "description": "查看提交历史。",
     "input_schema": {"type": "object", "properties": {
         "max_count": {"type": "integer", "description": "显示条数（默认 10）"}
     }}},
]


def _run_agent_loop(msgs, tools, handlers, max_rounds=50):
    """Run the standard LLM agent loop. Returns (response, messages, round_count)."""
    resp = None
    for round_idx in range(max_rounds):
        resp = call_llm(messages=msgs, tools=tools, max_tokens=8000)
        resp = parse_llm_response(resp)
        msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                h = handlers.get(b.name, lambda **kw: f"Unknown tool: {b.name}")
                try:
                    result = str(h(**b.input))[:50000]
                except Exception as e:
                    result = f"Tool error: {e}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": result})
        msgs.append({"role": "user", "content": results})
    return resp, msgs, round_idx


def _run_conflict_resolution_loop(wt_name, base_branch, initial_result):
    """Enter the conflict resolution self-healing loop.

    When rebase produces conflicts, the worker gets a new set of tools
    (read_conflict_file, resolve_conflict, accept_ours, accept_theirs,
    continue_rebase, list_conflicts) and must resolve all conflicts
    before the task is considered complete.

    The loop continues until:
      - All conflicts resolved and rebase succeeds → SUCCESS
      - Max rounds reached → FAILURE (report what's left)
    """
    conflict_handlers = _make_conflict_handlers()

    conflicted_files = initial_result.get("conflicted_files", [])
    details = initial_result.get("details", [])

    conflict_prompt = f"""## ⚠️ 进入冲突解决模式

你的代码在 rebase 到 `{base_branch}` 时产生了合并冲突。

### 冲突文件
{chr(10).join(f'- {f}' for f in conflicted_files)}

### 你的任务
1. 用 `read_conflict_file` 读取每个冲突文件，理解冲突内容
2. 用 `resolve_conflict` 写入正确的合并结果（去掉冲突标记）
   - 或者简单场景用 `accept_ours` / `accept_theirs` 快速选择
3. 所有文件都解决后，用 `continue_rebase` 继续 rebase
4. 如果 continue_rebase 报告新的冲突，重复以上步骤

### 冲突标记说明
- `<<<<<<< ours` (或 yours): 你的更改（worktree 分支的代码）
- `=======`: 分隔符
- `>>>>>>> theirs`: 对方的更改（目标分支已有的代码）

### 重要
- 这是你的责任。你不解决完冲突，任务就不算完成。
- 仔细阅读两边的代码，做出合理的合并决策。
- 如果不确定该保留哪边，通常优先保留功能更完整的版本。
"""

    msgs = [{"role": "user", "content": conflict_prompt}]
    resp = None
    max_conflict_rounds = 30  # Allow generous rounds for complex conflicts

    for cr_idx in range(max_conflict_rounds):
        resp = call_llm(messages=msgs, tools=CONFLICT_TOOLS, max_tokens=8000)
        resp = parse_llm_response(resp)
        msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            # Agent says it's done — check if actually resolved
            break

        results = []
        for b in resp.content:
            if b.type == "tool_use":
                h = conflict_handlers.get(b.name, lambda **kw: f"Unknown conflict tool: {b.name}")
                try:
                    result = h(**b.input)
                    # If result is a dict (from _continue_rebase), convert to string
                    if isinstance(result, dict):
                        result_str = (
                            f"success={result.get('success')}\n"
                            f"has_conflicts={result.get('has_conflicts')}\n"
                            f"conflicted_files={result.get('conflicted_files', [])}\n"
                            f"message={result.get('message', '')}"
                        )
                        # If rebase fully succeeded after this continue
                        if result.get("success"):
                            results.append({
                                "type": "tool_result",
                                "tool_use_id": b.id,
                                "content": result_str + "\n\n✅ REBASE 完成！所有冲突已解决，合并成功！"
                            })
                        else:
                            # Still have conflicts or other error
                            remaining = result.get("conflicted_files", [])
                            results.append({
                                "type": "tool_result",
                                "tool_use_id": b.id,
                                "content": result_str + "\n\n⚠️ 还有冲突需要解决:\n" +
                                           "\n".join(f"  - {f}" for f in remaining)
                            })
                    else:
                        result_str = str(result)[:50000]
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": result_str
                        })
                except Exception as e:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": f"Tool error: {e}"
                    })
        msgs.append({"role": "user", "content": results})

        # Check if rebase completed successfully (look at last continue_rebase result)
        for r in reversed(results):
            content = r.get("content", "")
            if isinstance(content, str) and ("REBASE 完成" in content or "rebase continued ok" in content):
                # Success!
                if resp:
                    summary = "".join(b.text for b in resp.content if hasattr(b, "text")) or "(resolved)"
                else:
                    summary = "(conflicts resolved)"
                return True, summary, cr_idx + 1, details

    # Exhausted conflict resolution rounds
    if resp:
        summary = "".join(b.text for b in resp.content if hasattr(b, "text")) or "(conflict resolution incomplete)"
    else:
        summary = "(conflict resolution failed — max rounds reached)"

    # Check final state
    remaining = WORKTREES._get_conflicted_files()
    if not remaining:
        # Maybe resolved on last round but we didn't catch it
        return True, summary, cr_idx + 1, details

    return False, summary + f"\n❌ 未解决的冲突文件: {remaining}", cr_idx + 1, details


def run_git_worker(task_description: str, base_branch: str = "main") -> str:
    """Spawn a Git Worker subagent in an isolated worktree.

    Full lifecycle with self-healing:
      1. Create a git worktree (isolated directory)
      2. Run the agent loop inside the worktree (code + commit)
         - If build/test fails → auto-retry loop (agent fixes its own bugs)
      3. Rebase onto target branch
         - If CONFLICTS → enter conflict resolution loop
           (agent reads conflicted files, merges manually, continues rebase)
      4. Clean up the worktree on success

    Multiple workers can run concurrently because each has its own directory.

    Args:
        task_description: What to implement (e.g., "给用户模块添加登录验证功能")
        base_branch: Branch to merge into. Default "main".

    Returns:
        Detailed summary including dev result, any conflicts, and final status.
    """
    # Step 1: Create isolated worktree
    try:
        wt = WORKTREES.create(task_description, base_ref=base_branch)
    except RuntimeError as e:
        return f"Failed to create worktree: {e}"

    wt_name = wt["name"]
    wt_path = wt["path"]
    wt_branch = wt["branch"]

    # Build isolated handlers that operate inside this worktree
    handlers = _make_git_worker_handlers(wt_name)
    if not handlers:
        return f"Failed to create handlers for worktree '{wt_name}'"

    # Build system prompt with worktree context
    system_prompt = GIT_WORKER_SYSTEM_TEMPLATE.format(
        wt_name=wt_name,
        wt_path=wt_path,
        wt_branch=wt_branch,
        target_branch=base_branch,
    )

    prompt = f"""{system_prompt}

## 你的任务
{task_description}

请按工作流程执行，完成后给出总结。记住：编译/测试失败必须自己修复，不能跳过。"""

    # Step 2: Run agent loop inside worktree (development phase)
    msgs = [{"role": "user", "content": prompt}]
    resp, msgs, dev_rounds = _run_agent_loop(
        msgs, GIT_WORKER_TOOLS, handlers, max_rounds=50
    )

    # Collect agent summary
    if resp:
        agent_summary = "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"
    else:
        agent_summary = "(agent failed or produced no output)"

    # Step 3: Attempt merge via rebase (auto-resolve mode)
    merge_result = WORKTREES.merge_to_main(
        wt_name, target=base_branch, cleanup=False, auto_resolve=True
    )

    # Build report
    report_lines = [
        f"=== Git Worker 报告 ===",
        f"Worktree: {wt_name} ({wt_branch})",
        f"开发轮次: {dev_rounds + 1}",
        f"",
        f"--- 开发阶段总结 ---",
        f"{agent_summary}",
    ]

    # Handle merge/rebase result
    if isinstance(merge_result, dict):
        if merge_result.get("success"):
            # Clean merge — clean up worktree
            cleanup = WORKTREES.remove(wt_name, force=True)
            report_lines.extend([
                f"",
                f"--- 合并结果 ---",
                f"✅ 成功 rebase 到 '{base_branch}'",
                f"{cleanup}",
            ])
        elif merge_result.get("has_conflicts"):
            # Enter conflict resolution self-healing loop
            report_lines.extend([
                f"",
                f"--- ⚠️ 检测到合并冲突 ---",
                f"冲突文件: {merge_result.get('conflicted_files', [])}",
                f"",
                f"--- 进入冲突自修复模式 ---",
            ])

            resolved, conflict_summary, conflict_rounds, details = \
                _run_conflict_resolution_loop(wt_name, base_branch, merge_result)

            report_lines.append(f"冲突解决轮次: {conflict_rounds}")
            report_lines.append(f"")
            report_lines.append(f"--- 冲突解决总结 ---")
            report_lines.append(conflict_summary)

            if resolved:
                # Cleanup successful resolution
                cleanup = WORKTREES.remove(wt_name, force=True)
                report_lines.extend([
                    f"",
                    f"--- 最终状态 ---",
                    f"✅ 任务完成（含冲突解决）",
                    f"{cleanup}",
                ])
            else:
                report_lines.extend([
                    f"",
                    f"--- ❌ 最终状态 ---",
                    f"⚠️ 冲突未完全解决，worktree 保留供人工处理",
                    f"可用命令手动处理: cd {WORKTREES.repo_root} && git status",
                ])
        else:
            # Other failure (not conflict)
            report_lines.extend([
                f"",
                f"--- 合并结果 ---",
                f"❌ 失败: {merge_result.get('message', 'unknown error')}",
            ])
            # Cleanup anyway on non-conflict failures
            WORKTREES.remove(wt_name, force=True)
    else:
        # String result (legacy mode fallback)
        report_lines.extend([
            f"",
            f"--- 合并结果 ---",
            f"{merge_result}",
        ])

    return "\n".join(report_lines)

