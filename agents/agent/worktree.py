#!/usr/bin/env python3
"""Worktree 管理器（来自 s12，适配 s_full 集成）。

为并行 Git Worker 子代理提供目录级隔离。
每个 Worker 获得独立的 worktree 和工作目录，
同时共享同一个 .git 对象存储。

    project/
      .worktrees/
        wt-add-logging/     ← Worker A 的隔离工作区
        wt-fix-auth/        ← Worker B 的隔离工作区
      .worktrees/index.json ← 生命周期追踪

使用方式：
    wm = WorktreeManager()
    wt = wm.create("feat-add-logging")   # 创建 worktree + 分支
    # ... 在 wt["path"] 中工作 ...
    wm.merge_and_cleanup("feat-add-logging", target="main")
"""

import json
import re
import subprocess
import time
from pathlib import Path

from ..core.config import WORKDIR

WORKTREES_DIR = WORKDIR / ".worktrees"
WORKTREES_INDEX = WORKTREES_DIR / "index.json"


def _detect_repo_root(cwd: Path):  # -> Path | None（兼容 py3.9）
    """如果 cwd 在 git 仓库内，返回仓库根目录。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


REPO_ROOT = _detect_repo_root(WORKDIR) or WORKDIR


class WorktreeManager:
    """管理 Git Worktree，用于隔离任务执行。"""

    def __init__(self):
        self.repo_root = REPO_ROOT
        self.dir = WORKTREES_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = WORKTREES_INDEX
        self._init_index()
        self.git_ok = self._is_git_repo()

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _init_index(self):
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))

    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        self.index_path.write_text(json.dumps(data, indent=2))

    def _find(self, name: str):  # -> dict | None (py3.9 compat)
        for wt in self._load_index().get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _is_git_repo(self) -> bool:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root, capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0 and r.stdout.strip() == "true"
        except Exception:
            return False

    def _git(self, *args: str, cwd=None, timeout: int = 60) -> subprocess.CompletedProcess:
        """运行 git 命令。默认使用 repo_root 作为工作目录。"""
        return subprocess.run(
            ["git"] + list(args),
            cwd=cwd or self.repo_root,
            capture_output=True, text=True, timeout=timeout,
        )

    @staticmethod
    def _sanitize_name(raw: str) -> str:
        """将任务描述转换为安全的 worktree/分支名。"""
        # 取前40个字符，将非字母数字替换为连字符，合并重复连字符
        s = raw.strip().lower()[:40]
        s = re.sub(r"[^a-z0-9]", "-", s)
        s = re.sub(r"-{2,}", "-", s)
        s = re.sub(r"^[-]|[-]$", "", s)
        return s or "unnamed"

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def create(self, task_description: str, base_ref: str = "HEAD") -> dict:
        """创建一个新的 worktree，并从 base_ref 派生分支。

        参数:
            task_description: 人类可读的任务描述（用于生成名称）。
            base_ref: 作为分支基础的 Git 引用。默认为 HEAD。

        返回:
            包含以下键的字典: name, path, branch, created_at。

        异常:
            RuntimeError: 如果 git 不可用或 worktree 创建失败。
        """
        if not self.git_ok:
            raise RuntimeError(f"Not a git repo: {self.repo_root}")

        safe_name = self._sanitize_name(task_description)
        name = f"wt-{safe_name}"
        branch = f"wt/{safe_name}"
        path = self.dir / name

        # 避免命名冲突 - 必要时追加数字
        idx = self._load_index()
        existing_names = {wt["name"] for wt in idx.get("worktrees", [])}
        if name in existing_names or path.exists():
            for i in range(2, 20):
                candidate = f"{name}-{i}"
                if candidate not in existing_names and not (self.dir / candidate).exists():
                    name = candidate
                    branch = f"wt/{safe_name}-{i}"
                    path = self.dir / name
                    break

        # 清理残留：分支存在但 worktree 不存在，先删除孤立分支
        r_check = self._git("branch", "--list", branch)
        if branch in r_check.stdout and not path.exists():
            self._git("branch", "-D", branch)

        # 创建 worktree 和分支
        r = self._git("worktree", "add", "-b", branch, str(path), base_ref)
        if r.returncode != 0:
            raise RuntimeError(
                f"Failed to create worktree '{name}': {r.stderr.strip()}")

        entry = {
            "name": name,
            "path": str(path),
            "branch": branch,
            "base_ref": base_ref,
            "task": task_description[:100],
            "status": "active",
            "created_at": time.time(),
        }

        idx["worktrees"].append(entry)
        self._save_index(idx)

        return entry

    def get(self, name: str):  # -> dict | None (py3.9 compat)
        """按名称获取 worktree 信息。"""
        return self._find(name)

    def list_all(self) -> str:
        """列出所有已追踪的 worktree。"""
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "没有活跃的 worktree。"

        lines = [f"{'Name':<28} {'Branch':<25} {'Status':<10} {'Task'}"]
        lines.append("-" * 90)
        for wt in wts:
            path_exists = "OK" if Path(wt["path"]).exists() else "MISSING"
            lines.append(
                f"{wt['name']:<28} {wt['branch']:<25} "
                f"{wt.get('status', '?'):<10} {wt.get('task', '-')[:40]}"
            )
            lines.append(f"  path: {wt['path']}  [{path_exists}]")
        return "\n".join(lines)

    def status(self, name: str) -> str:
        """显示 worktree 内的 git 状态。"""
        wt = self._find(name)
        if not wt:
            return f"Error: Worktree '{name}' not found."
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        r = self._git("status", "--short", "--branch", cwd=path, timeout=15)
        if r.returncode != 0:
            return f"Error: {r.stderr.strip()}"
        output = r.stdout.strip()
        return output or "(clean)"

    def run_in(self, name: str, command: str, timeout: int = 120) -> str:
        """在指定 worktree 内运行 shell 命令。

        这是关键的隔离原语——所有 Worker 操作都通过此方法执行。
        """
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
        if any(d in command for d in dangerous):
            return "错误：危险命令已被阻止"

        wt = self._find(name)
        if not wt:
            return f"Error: Worktree '{name}' not found."
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            r = subprocess.run(
                command, shell=True, cwd=path,
                capture_output=True, text=True, timeout=timeout,
            )
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Timeout ({timeout}s)"

    def _get_conflicted_files(self) -> list:
        """返回存在合并/rebase 冲突的文件列表。"""
        r = self._git("diff", "--name-only", "--diff-filter=U", timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return [f for f in r.stdout.strip().splitlines() if f]
        return []

    def _abort_merge_or_rebase(self):
        """中止 repo_root 上正在进行的合并或 rebase 操作。"""
        # 先尝试中止合并
        r = self._git("merge", "--abort", timeout=10)
        if r.returncode != 0:
            # 可能是 rebase
            self._git("rebase", "--abort", timeout=10)

    def _rebase_onto_target(self, name: str, target: str, results: list) -> dict:
        """将 worktree 分支 rebase 到目标分支上。返回状态字典。

        返回字典包含以下键:
            success: bool
            has_conflicts: bool
            conflicted_files: list[str]
            message: str
        """
        wt = self._find(name)
        if not wt:
            return {"success": False, "has_conflicts": False,
                    "message": f"Worktree '{name}' not found"}

        branch = wt["branch"]

        # 先在 repo_root 上检出目标分支
        r = self._git("checkout", target, timeout=15)
        if r.returncode != 0:
            r2 = self._git("checkout", "master", timeout=15)
            if r2.returncode != 0:
                return {"success": False, "has_conflicts": False,
                        "message": f"Cannot checkout '{target}' or 'master'"}
            target = "master"
            results.append(f"Switched to 'master' ('{target}' not found)")
        else:
            results.append(f"Switched to '{target}'")

        # 将 worktree 分支 rebase 到目标分支上
        r = self._git("rebase", "--onto", target, "HEAD", branch, timeout=60)
        if r.returncode == 0:
            # 成功！rebase 后的提交现在在目标分支上。
            # 但需要清理——原分支已成为孤立分支。
            results.append(f"Rebased '{branch}' onto '{target}' successfully.")
            return {"success": True, "has_conflicts": False,
                    "conflicted_files": [], "message": "rebase ok"}

        # rebase 过程中发生冲突
        conflicted = self._get_conflicted_files()
        results.append(
            f"REBASE CONFLICT in {len(conflicted)} file(s): {', '.join(conflicted)}"
        )
        return {
            "success": False, "has_conflicts": True,
            "conflicted_files": conflicted,
            "message": f"rebase conflict: {conflicted}"
        }

    def _continue_rebase(self) -> dict:
        """尝试执行 'git rebase --continue'。返回状态字典。"""
        r = self._git("rebase", "--continue", "--no-edit", timeout=30)
        if r.returncode == 0:
            return {"success": True, "has_conflicts": False,
                    "conflicted_files": [], "message": "rebase continued ok"}

        conflicted = self._get_conflicted_files()
        return {
            "success": False, "has_conflicts": len(conflicted) > 0,
            "conflicted_files": conflicted,
            "message": f"rebase continue failed: {conflicted}"
        }

    def _resolve_file_in_repo_root(self, file_path: str, strategy: str = "ours") -> str:
        """使用指定策略解决冲突文件。

        参数:
            file_path: 冲突文件的相对路径。
            strategy: 'ours'（保留传入/worktree 的修改）或 'theirs'（保留已有的内容）。
        """
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"

        # git checkout --ours/--theirs <文件>
        r = self._git("checkout", f"--{strategy}", "--", file_path, timeout=10)
        if r.returncode != 0:
            return f"Failed to resolve {file_path} with strategy={strategy}: {r.stderr.strip()}"

        # 暂存已解决的文件
        self._git("add", file_path, timeout=10)
        return f"Resolved {file_path} using --{strategy} strategy"

    def get_file_conflict_content(self, file_path: str) -> str:
        """读取冲突文件的内容（包含冲突标记）。
        供 LLM 理解冲突的具体内容。
        """
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"
        try:
            return full_path.read_text()[:30000]
        except Exception as e:
            return f"Error reading file: {e}"

    def write_resolved_file(self, file_path: str, content: str) -> str:
        """将解决后的内容写入冲突文件（在 repo_root 中）。

        写入后，暂存文件以便执行 rebase --continue。
        """
        full_path = self.repo_root / file_path
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self._git("add", file_path, timeout=10)
            return f"Wrote resolved content to {file_path} ({len(content)} bytes)"
        except Exception as e:
            return f"Error writing resolved file: {e}"

    def merge_to_main(self, name: str, target: str = "main",
                      cleanup: bool = True,
                      auto_resolve: bool = False) -> dict:
        """将 worktree 的分支合并/rebase 到目标分支。

        当 auto_resolve=False（手动使用的默认模式）时:
            使用 merge --no-ff，报告冲突，不自动解决。
            返回简单的字符串摘要。

        当 auto_resolve=True（用于 Git Worker 自修复）时:
            使用 rebase 以获得更干净的历史。
            返回包含详细状态的字典，供 Worker 决定后续操作。

        工作流（auto_resolve 模式）:
          1. 检查 worktree 是否干净
          2. 将分支 rebase 到目标分支上
          3. 如果有冲突 → 返回冲突详情供 Worker 解决
          4. Worker 解决冲突 → 调用 _continue_rebase()
          5. 成功后清理

        返回:
            包含以下键的字典: success, has_conflicts, conflicted_files,
                           message, details（日志行列表）
        """
        wt = self._find(name)
        if not wt:
            result = {"success": False, "has_conflicts": False,
                     "conflicted_files": [], "message": f"Worktree '{name}' not found",
                     "details": []}
            return result if auto_resolve else result["message"]

        path = Path(wt["path"])
        branch = wt["branch"]
        results = []

        # 步骤1：检查 worktree 是否干净
        r = self._git("status", "--porcelain", cwd=path, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            results.append(
                f"WARNING: Worktree '{name}' has uncommitted changes. "
                f"Commit them before merging.\n{r.stdout.strip()[:500]}"
            )

        if not auto_resolve:
            # 传统合并模式（返回简单字符串）
            return self._merge_simple(name, target, branch, results, cleanup)

        # 自动解决模式：使用 rebase
        rebase_result = self._rebase_onto_target(name, target, results)
        rebase_result["details"] = results

        if rebase_result["success"]:
            # 成功后清理
            if cleanup:
                rm_result = self.remove(name, force=True)
                results.append(rm_result)
            rebase_result["details"] = results
            return rebase_result

        # 存在冲突——标记状态，不清理
        self._set_status(name, "conflict")
        rebase_result["details"] = results
        return rebase_result

    def _merge_simple(self, name: str, target: str, branch: str,
                       results: list, cleanup: bool) -> str:
        """简单合并（非自动解决），返回字符串。"""
        # 将主工作区切换到目标分支
        r = self._git("checkout", target, timeout=15)
        if r.returncode != 0:
            r2 = self._git("checkout", "master", timeout=15)
            if r2.returncode != 0:
                return f"Error: Cannot checkout '{target}' or 'master'.\n{r.stderr.strip()}"
            target = "master"
            results.append(f"Switched to 'master' ('{target}' not found)")
        else:
            results.append(f"Switched to '{target}'")

        # 合并
        r = self._git("merge", "--no-ff", branch, "-m",
                       f"Merge {branch} into {target}", timeout=60)
        if r.returncode != 0:
            conflicted = self._get_conflicted_files()
            results.append(
                f"MERGE CONFLICT! Conflicted files: {conflicted}\n"
                f"{r.stderr.strip()}\n\n"
                f"To resolve:\n"
                f"  cd {self.repo_root}\n"
                f"  # fix conflicted files\n"
                f"  git add <files>\n"
                f"  git commit"
            )
            self._set_status(name, "conflict")
            return "\n".join(results)

        results.append(f"Merged '{branch}' into '{target}' successfully.")

        if cleanup:
            rm_result = self.remove(name, force=True)
            results.append(rm_result)

        return "\n".join(results)

    def remove(self, name: str, force: bool = False) -> str:
        """移除 worktree 及其分支。"""
        wt = self._find(name)
        if not wt:
            return f"Error: Worktree '{name}' not found."

        path = Path(wt["path"])
        branch = wt.get("branch", "")

        # 移除 worktree
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path))

        r = self._git(*args, timeout=15)
        if r.returncode != 0:
            # 尝试强制移除
            r2 = self._git("worktree", "remove", "--force", str(path), timeout=15)
            if r2.returncode != 0:
                return (
                    f"Warning: Could not remove worktree '{name}'. "
                    f"You may need to manually remove it.\n"
                    f"  git worktree remove --force {path}\n"
                    f"Error: {r.stderr.strip()}"
                )

        # 删除分支
        if branch:
            self._git("branch", "-D", branch, timeout=10)

        # 更新索引
        idx = self._load_index()
        idx["worktrees"] = [
            w for w in idx["worktrees"] if w["name"] != name
        ]
        self._save_index(idx)

        return f"Removed worktree '{name}' (branch: {branch})"

    def _set_status(self, name: str, status: str):
        """更新索引中的 worktree 状态。"""
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt["name"] == name:
                wt["status"] = status
                break
        self._save_index(idx)

    def get_worktree_path(self, name: str):  # -> Path | None (py3.9 compat)
        """获取 worktree 的文件系统路径。"""
        wt = self._find(name)
        if wt:
            return Path(wt["path"])
        return None


class WorktreePool:
    """预分配的 worktree 池，支持动态伸缩。

    与其为每个任务创建/销毁 worktree（开销大），
    不如由该池维护一组可复用的 worktree。Worker 获取
    worktree，完成工作，提交，然后归还给池。

    池动态伸缩:
      - 初始为空（或 min_size）
      - 按需增长，上限为 max_size
      - 空闲时收缩（通过 prune）

    用法:
        pool = WorktreePool(max_size=50)
        wt = pool.acquire("task-description")  # 获取或创建
        # ... 在 wt["path"] 中工作 ...
        pool.release(wt["name"])              # 重置并归还到池
        pool.prune()                          # 移除多余的空闲 worktree
    """

    def __init__(self, max_size: int = 50):
        """初始化池。

        参数:
            max_size: worktree 最大数量（并行度上限）。
        """
        self._wm = WorktreeManager()  # 复用已有的管理器进行 git 操作
        self.max_size = max_size
        self._available = []   # 空闲 worktree 的名称列表
        self._in_use = set()   # 当前已获取的 worktree 名称集合
        self._counter = 0      # 用于生成唯一名称
        # 不再预热，按需创建。每个 worktree 的分支名来自任务描述，有明确业务含义。

    def _warm_up(self):
        """已废弃：不再预热。worktree 按需创建，分支名由任务决定。"""

    def _create_one(self, prefix: str = "_pool"):
        """创建新的 worktree。返回名称或 None。"""
        self._counter += 1
        safe_name = f"{prefix}-{self._counter}"
        try:
            entry = self._wm.create(safe_name)
            return entry["name"]
        except Exception as e:
            print(f"  [WorktreePool] Failed to create worktree: {e}")
            return None

    def acquire(self, task_hint: str) -> dict:
        """为特定任务获取 worktree。

        始终创建一个新的 worktree，分支名来自任务描述。
        不使用池化/复用——每个任务获得独立的 worktree 和分支。

        参数:
            task_hint: 用于生成分支名的任务描述（必填）。

        返回:
            包含以下键的 worktree 字典: name, path, branch。
        """
        if not task_hint:
            raise ValueError("acquire() 必须提供 task_hint（任务描述），用于生成有意义的分支名")

        if len(self._in_use) >= self.max_size:
            raise RuntimeError(
                f"WorktreePool 已满 (max={self.max_size}, in_use={len(self._in_use)})"
            )

        name = self._create_one(task_hint)
        if not name:
            raise RuntimeError(f"Failed to create worktree for task: {task_hint}")

        wt = self._wm.get(name)
        if wt:
            self._in_use.add(name)
        return wt

    def release(self, name: str):
        """使用后释放并销毁 worktree。

        由于每个 worktree 都有任务专属分支，释放时直接销毁，
        而不是归还到池中。这样可以保持分支的语义明确性。
        """
        if name not in self._in_use:
            return

        try:
            self._wm.remove(name, force=True)
        except Exception as e:
            print(f"  [WorktreePool] Warning: failed to remove {name}: {e}")

        self._in_use.discard(name)

    def prune(self):
        """销毁所有空闲 worktree。由于不再使用池化，此方法为空操作。"""
        pass

    def stats(self) -> dict:
        """返回池的统计信息。"""
        return {
            "in_use": len(self._in_use),
            "max_size": self.max_size,
        }

    def cleanup_all(self):
        """移除所有池中的 worktree。关闭时调用。"""
        for name in list(self._available):
            try:
                self._wm.remove(name, force=True)
            except Exception:
                pass
        for name in list(self._in_use):
            try:
                self._wm.remove(name, force=True)
            except Exception:
                pass
        self._available.clear()
        self._in_use.clear()


# 模块级单例
WORKTREES = WorktreeManager()

# 模块级 worktree 池（用于高吞吐场景）
WTPOOL = WorktreePool(max_size=50)

