#!/usr/bin/env python3
"""Worktree Manager (from s12, adapted for s_full integration).

Provides directory-level isolation for parallel Git Worker subagents.
Each worker gets its own worktree with an independent working directory,
while sharing the same .git object store.

    project/
      .worktrees/
        wt-add-logging/     ← worker A's isolated workspace
        wt-fix-auth/        ← worker B's isolated workspace
      .worktrees/index.json ← lifecycle tracking

Usage:
    wm = WorktreeManager()
    wt = wm.create("feat-add-logging")   # creates worktree + branch
    # ... do work in wt["path"] ...
    wm.merge_and_cleanup("feat-add-logging", target="main")
"""

import json
import re
import subprocess
import time
from pathlib import Path

from .config import WORKDIR

WORKTREES_DIR = WORKDIR / ".worktrees"
WORKTREES_INDEX = WORKTREES_DIR / "index.json"


def _detect_repo_root(cwd: Path):  # -> Path | None (py3.9 compat)
    """Return git repo root if cwd is inside a repo."""
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
    """Manage git worktrees for isolated task execution."""

    def __init__(self):
        self.repo_root = REPO_ROOT
        self.dir = WORKTREES_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = WORKTREES_INDEX
        self._init_index()
        self.git_ok = self._is_git_repo()

    # ------------------------------------------------------------------
    # Internal helpers
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
        """Run git command. Defaults to repo_root as cwd."""
        return subprocess.run(
            ["git"] + list(args),
            cwd=cwd or self.repo_root,
            capture_output=True, text=True, timeout=timeout,
        )

    @staticmethod
    def _sanitize_name(raw: str) -> str:
        """Convert task description into a safe worktree/branch name."""
        # Take first 30 chars, replace non-alphanumeric with hyphens, collapse
        s = raw.strip().lower()[:40]
        s = re.sub(r"[^a-z0-9]", "-", s)
        s = re.sub(r"-{2,}", "-", s)
        s = re.sub(r"^[-]|[-]$", "", s)
        return s or "unnamed"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, task_description: str, base_ref: str = "HEAD") -> dict:
        """Create a new worktree with a branch derived from base_ref.

        Args:
            task_description: Human-readable task (used to generate name).
            base_ref: Git ref to base the branch on. Default HEAD.

        Returns:
            Dict with keys: name, path, branch, created_at.

        Raises:
            RuntimeError: If git not available or worktree creation fails.
        """
        if not self.git_ok:
            raise RuntimeError(f"Not a git repo: {self.repo_root}")

        safe_name = self._sanitize_name(task_description)
        name = f"wt-{safe_name}"
        branch = f"wt/{safe_name}"
        path = self.dir / name

        # Avoid collision - append number if needed
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

        # Create the worktree + branch
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
        """Get worktree info by name."""
        return self._find(name)

    def list_all(self) -> str:
        """List all tracked worktrees."""
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No active worktrees."

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
        """Show git status inside a worktree."""
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
        """Run a shell command inside a specific worktree.

        This is the key isolation primitive — all worker operations go through this.
        """
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"

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
        """Return list of files with merge/rebase conflicts."""
        r = self._git("diff", "--name-only", "--diff-filter=U", timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return [f for f in r.stdout.strip().splitlines() if f]
        return []

    def _abort_merge_or_rebase(self):
        """Abort any ongoing merge or rebase operation on repo_root."""
        # Try abort merge first
        r = self._git("merge", "--abort", timeout=10)
        if r.returncode != 0:
            # Maybe it's a rebase
            self._git("rebase", "--abort", timeout=10)

    def _rebase_onto_target(self, name: str, target: str, results: list) -> dict:
        """Rebase worktree branch onto target. Returns status dict.

        Returns dict with keys:
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

        # First checkout the target branch on repo_root
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

        # Rebase the worktree branch onto target
        r = self._git("rebase", branch, onto=target, timeout=60)
        if r.returncode == 0:
            # Success! The rebased commits are now on target.
            # But we need to clean up — the original branch is orphaned.
            results.append(f"Rebased '{branch}' onto '{target}' successfully.")
            return {"success": True, "has_conflicts": False,
                    "conflicted_files": [], "message": "rebase ok"}

        # Conflict during rebase
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
        """Attempt 'git rebase --continue'. Returns status dict."""
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
        """Resolve a conflicted file using the given strategy.

        Args:
            file_path: Relative path to the conflicted file.
            strategy: 'ours' (keep incoming/worktree changes) or 'theirs' (keep existing).
        """
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"

        # git checkout --ours/--theirs <file>
        r = self._git("checkout", f"--{strategy}", "--", file_path, timeout=10)
        if r.returncode != 0:
            return f"Failed to resolve {file_path} with strategy={strategy}: {r.stderr.strip()}"

        # Stage the resolved file
        self._git("add", file_path, timeout=10)
        return f"Resolved {file_path} using --{strategy} strategy"

    def get_file_conflict_content(self, file_path: str) -> str:
        """Read a conflicted file's content including conflict markers.
        Useful for the LLM to understand what conflicts look like.
        """
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"
        try:
            return full_path.read_text()[:30000]
        except Exception as e:
            return f"Error reading file: {e}"

    def write_resolved_file(self, file_path: str, content: str) -> str:
        """Write resolved content to a conflicted file (in repo_root).

        After writing, stages the file for rebase --continue.
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
        """Merge/rebase worktree's branch into target branch.

        When auto_resolve=False (default for manual use):
            Uses merge --no-ff, reports conflicts, does NOT auto-resolve.
            Returns a simple string summary.

        When auto_resolve=True (used by Git Worker self-healing):
            Uses rebase for cleaner history.
            Returns a dict with detailed status so the worker can decide what to do.

        Workflow (auto_resolve mode):
          1. Check worktree is clean
          2. Rebase branch onto target
          3. If conflict → return conflict details for worker to resolve
          4. Worker resolves → call _continue_rebase()
          5. Cleanup on success

        Returns:
            dict with keys: success, has_conflicts, conflicted_files,
                           message, details (list of log lines)
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

        # Step 1: Check worktree is clean
        r = self._git("status", "--porcelain", cwd=path, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            results.append(
                f"WARNING: Worktree '{name}' has uncommitted changes. "
                f"Commit them before merging.\n{r.stdout.strip()[:500]}"
            )

        if not auto_resolve:
            # Legacy merge mode (simple string result)
            return self._merge_simple(name, target, branch, results, cleanup)

        # Auto-resolve mode: use rebase
        rebase_result = self._rebase_onto_target(name, target, results)
        rebase_result["details"] = results

        if rebase_result["success"]:
            # Clean up on success
            if cleanup:
                rm_result = self.remove(name, force=True)
                results.append(rm_result)
            rebase_result["details"] = results
            return rebase_result

        # Has conflicts — mark status, don't clean up
        self._set_status(name, "conflict")
        rebase_result["details"] = results
        return rebase_result

    def _merge_simple(self, name: str, target: str, branch: str,
                       results: list, cleanup: bool) -> str:
        """Simple merge (non-auto-resolve) returning a string."""
        # Switch main to target branch
        r = self._git("checkout", target, timeout=15)
        if r.returncode != 0:
            r2 = self._git("checkout", "master", timeout=15)
            if r2.returncode != 0:
                return f"Error: Cannot checkout '{target}' or 'master'.\n{r.stderr.strip()}"
            target = "master"
            results.append(f"Switched to 'master' ('{target}' not found)")
        else:
            results.append(f"Switched to '{target}'")

        # Merge
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
        """Remove a worktree and its branch."""
        wt = self._find(name)
        if not wt:
            return f"Error: Worktree '{name}' not found."

        path = Path(wt["path"])
        branch = wt.get("branch", "")

        # Remove worktree
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path))

        r = self._git(*args, timeout=15)
        if r.returncode != 0:
            # Try harder
            r2 = self._git("worktree", "remove", "--force", str(path), timeout=15)
            if r2.returncode != 0:
                return (
                    f"Warning: Could not remove worktree '{name}'. "
                    f"You may need to manually remove it.\n"
                    f"  git worktree remove --force {path}\n"
                    f"Error: {r.stderr.strip()}"
                )

        # Delete the branch
        if branch:
            self._git("branch", "-D", branch, timeout=10)

        # Update index
        idx = self._load_index()
        idx["worktrees"] = [
            w for w in idx["worktrees"] if w["name"] != name
        ]
        self._save_index(idx)

        return f"Removed worktree '{name}' (branch: {branch})"

    def _set_status(self, name: str, status: str):
        """Update worktree status in index."""
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt["name"] == name:
                wt["status"] = status
                break
        self._save_index(idx)

    def get_worktree_path(self, name: str):  # -> Path | None (py3.9 compat)
        """Get the filesystem path of a worktree."""
        wt = self._find(name)
        if wt:
            return Path(wt["path"])
        return None


# Module-level singleton
WORKTREES = WorktreeManager()

