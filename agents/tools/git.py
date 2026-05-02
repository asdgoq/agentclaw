#!/usr/bin/env python3
"""Git tools for LLM agent — status, branch, commit, diff, log, PR.

All functions run git commands via subprocess and return formatted output.
Designed to be registered as LLM-callable tools in s_full.py.
"""

import subprocess

from ..core.config import WORKDIR


def _git(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command in WORKDIR. Returns CompletedProcess."""
    return subprocess.run(
        ["git"] + list(args),
        cwd=WORKDIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _is_git_repo() -> bool:
    """Check if WORKDIR is inside a git repository."""
    r = _git("rev-parse", "--is-inside-work-tree", timeout=10)
    return r.returncode == 0 and r.stdout.strip() == "true"


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def git_status() -> str:
    """Show working tree status — staged, unstaged, untracked files."""
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"

    r = _git("status", "--short", "--branch", timeout=15)
    if r.returncode != 0:
        return f"Error: {r.stderr.strip()}"

    lines = r.stdout.strip().splitlines()
    if not lines:
        return "工作区干净，没有改动。"

    # Count by status prefix
    staged = modified = deleted = untracked = renamed = 0
    for line in lines:
        if line.startswith("A ") or line.startswith("M ") or line.startswith("D "):
            staged += 1
        elif line.startswith(" M") or line.startswith(" D") or line.startswith("??"):
            if line.startswith("??"):
                untracked += 1
            elif line.startswith(" D") or line[2] == "D":
                deleted += 1
            else:
                modified += 1
        elif line.startswith("R "):
            renamed += 1

    parts = []
    if staged:
        parts.append(f"已暂存 {staged}")
    if modified:
        parts.append(f"已修改 {modified}")
    if deleted:
        parts.append(f"已删除 {deleted}")
    if untracked:
        parts.append(f"未跟踪 {untracked}")
    if renamed:
        parts.append(f"已重命名 {renamed}")

    summary = ", ".join(parts) if parts else "干净"
    result = f"Git 状态：{summary}\n\n"
    result += "\n".join(lines)
    return result


def git_branch(list_all: bool = False) -> str:
    """List, create, or switch branches.

    Args:
        list_all: If True, list all branches (local + remote). If False, show current branch only.
                  Call as tool with action="list"/"create"/"checkout".
    """
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"

    # This tool is called from LLM with different actions via parameters
    # For simplicity, we support listing; create/checkout use the action param
    r = _git("branch", "--verbose", "--no-color", timeout=15)
    if r.returncode != 0:
        return f"Error: {r.stderr.strip()}"

    lines = r.stdout.strip().splitlines()
    current = None
    branches = []
    for line in lines:
        line = line.strip()
        if line.startswith("* "):
            current = line[2:].split()[0]
            branches.append(f"* {line[2:]}")
        else:
            branches.append(f"  {line}")

    result = f"当前分支：{current or '(未知)'}\n\n"
    result += "\n".join(branches)
    return result


def git_branch_create(name: str, from_ref: str = "HEAD") -> str:
    """Create a new branch and switch to it."""
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"
    name = name.strip()
    if not name:
        return "Error: 分支名不能为空。"

    r = _git("checkout", "-b", name, from_ref, timeout=15)
    if r.returncode != 0:
        return f"Error: 创建分支失败 — {r.stderr.strip()}"
    return f"已创建并切换到分支：{name}"


def git_branch_checkout(name: str) -> str:
    """Switch to an existing branch."""
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"
    name = name.strip()
    if not name:
        return "Error: 分支名不能为空。"

    r = _git("checkout", name, timeout=15)
    if r.returncode != 0:
        return f"Error: 切换分支失败 — {r.stderr.strip()}"
    return f"已切换到分支：{name}"


def git_commit(message: str, add: bool = True, files: str = ".") -> str:
    """Stage changes and commit.

    Args:
        message: Commit message (required).
        add: Whether to run `git add` first. Default True.
        files: Files or path to stage. Default "." (all).
    """
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"
    message = message.strip()
    if not message:
        return "Error: commit message 不能为空。"

    # Stage
    if add:
        r = _git("add", files, timeout=30)
        if r.returncode != 0:
            return f"Error: git add 失败 — {r.stderr.strip()}"

    # Check if there's anything to commit
    r = _git("diff", "--cached", "--stat", timeout=10)
    if r.returncode == 0 and not r.stdout.strip():
        return "没有需要提交的更改（工作区干净或没有已暂存的更改）。"

    # Commit
    r = _git("commit", "-m", message, timeout=30)
    if r.returncode != 0:
        return f"Error: 提交失败 — {r.stderr.strip()}"

    # Get short status
    # Extract branch and commit hash
    r2 = _git("log", "--oneline", "-1", timeout=10)
    commit_line = r2.stdout.strip() if r2.returncode == 0 else "(unknown)"

    # Show what was committed
    r3 = _git("diff", "--cached", "--stat", "HEAD~1", timeout=10)
    stats = r3.stdout.strip() if r3.returncode == 0 else ""

    return f"提交成功！\n{commit_line}\n{stats}"


def git_diff(staged: bool = False, file: str = "") -> str:
    """Show diff of changes.

    Args:
        staged: If True, show staged changes (--cached). Otherwise unstaged.
        file: Optional specific file path.
    """
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"

    args = ["diff"]
    if staged:
        args.append("--cached")
    if file:
        args.extend(["--", file])

    r = _git(*args, timeout=20)
    if r.returncode != 0:
        return f"Error: {r.stderr.strip()}"

    output = r.stdout.strip()
    if not output:
        return "没有差异。" + (" （已暂存区为空）" if staged else " （工作区干净）")
    # Truncate very large diffs
    if len(output) > 20000:
        output = output[:20000] + f"\n... (截断，共 {len(output)} 字符)"
    return output


def git_log(max_count: int = 10, oneline: bool = True) -> str:
    """Show commit history.

    Args:
        max_count: Number of commits to show. Default 10.
        oneline: Show one line per commit. Default True.
    """
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"

    args = ["log", f"-{max_count}"]
    if oneline:
        args.append("--oneline")
    args.append("--no-color")
    args.append("--decorate")

    r = _git(*args, timeout=15)
    if r.returncode != 0:
        return f"Error: {r.stderr.strip()}"

    output = r.stdout.strip()
    if not output:
        return "没有提交历史。"
    return output


def git_pr(title: str = "", body: str = "", base: str = "main", draft: bool = False) -> str:
    """Create a GitHub Pull Request using `gh` CLI.

    Requires `gh` to be installed and authenticated.
    Pushes the current branch to remote first, then creates PR.

    Args:
        title: PR title. If empty, uses the last commit message.
        body: PR description/body.
        base: Target branch. Default "main".
        draft: Create as draft PR. Default False.
    """
    if not _is_git_repo():
        return "Error: 当前目录不是 Git 仓库。"

    # Check gh is available
    r = subprocess.run(["gh", "--version"], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return "Error: 'gh' CLI 未安装。请先安装 GitHub CLI: https://cli.github.com/"

    # Get current branch
    r = _git("branch", "--show-current", timeout=10)
    if r.returncode != 0:
        return f"Error: 无法获取当前分支 — {r.stderr.strip()}"
    branch = r.stdout.strip()

    # Check we're not on main/master
    if branch in ("main", "master"):
        return f"Error: 当前在 '{branch}' 分支上，请先切换到功能分支再创建 PR。"

    # Check for dirty state
    r = _git("diff", "--stat", timeout=10)
    if r.returncode == 0 and r.stdout.strip():
        return f"Warning: 工作区有未提交的更改。请先提交后再创建 PR。\n{r.stdout.strip()}"

    # Push to remote
    push_r = _git("push", "-u", "origin", branch, timeout=60)
    if push_r.returncode != 0:
        return f"Error: 推送失败 — {push_r.stderr.strip()}"

    # Build gh pr create command
    cmd = ["gh", "pr", "create", "--base", base, "--head", branch]
    if title:
        cmd += ["--title", title]
    if body:
        cmd += ["--body", body]
    if draft:
        cmd.append("--draft")

    r = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return f"Error: 创建 PR 失败 — {r.stderr.strip()}"
    return f"PR 创建成功！\n{r.stdout.strip()}"

