#!/usr/bin/env python3
"""Planner (任务规划器) - 任务分解、派发、监控代码库健康度。

职责：
1. 接收用户目标，分解为可并行执行的子任务
2. 将子任务派发给 Git Worker 子代理执行
3. 监控执行结果，发现遗留问题后重新派发
4. 对代码库健康度负总责

架构：

    用户目标
        ↓
    ┌──────────────┐
    │   Planner    │ ← 任务分解 + 调度 + 监控
    └──────┬───────┘
           │ 派发任务
     ┌─────┼─────┐
     ↓     ↓     ↓
  Worker-A Worker-B Worker-C  ← 并行执行（各自在独立 worktree）
     │     │     │
     ↓     ↓     ↓
  [合并到 main]  ← 可能有冲突，Worker 自行解决

使用方式：
    from .planner import Planner
    p = Planner()
    result = p.plan_and_execute("实现用户认证功能，包括登录、注册、密码重置")
"""

import time
import uuid

from .llm import call_llm, parse_llm_response
from .subagent import run_git_worker

# Planner's system prompt for task decomposition
PLANNER_SYSTEM = """你是一个任务规划器。你的职责是将复杂目标分解为可并行的子任务。

## 规则
1. 每个子任务应该是独立的、可以由一个 Git Worker 在隔离环境中完成的工作单元
2. 子任务之间尽量减少文件依赖——如果两个任务必须改同一个文件，它们应该串行执行
3. 每个任务描述要具体清晰，包含要改什么文件、实现什么功能
4. 标记任务间的依赖关系（如果有）

## 输出格式
你必须输出一个 JSON 数组，每个元素代表一个子任务：
```json
[
  {
    "id": "task-1",
    "description": "具体的任务描述",
    "depends_on": [],
    "priority": "high"
  }
]
```

- `depends_on`: 依赖的任务 ID 列表。空数组表示无依赖，可并行。
- `priority`: high / medium / low

## 分解原则
- 按模块/文件边界拆分（不同文件的任务可并行）
- 按功能边界拆分（独立功能可并行）
- 测试任务依赖被测代码任务
- 公共基础设施任务优先级最高（其他任务可能依赖它）"""


def _parse_task_plan(response_text: str) -> list:
    """Extract JSON task plan from LLM response.

    Handles cases where the LLM wraps JSON in markdown code blocks or
    includes explanatory text before/after.
    """
    import json
    import re

    text = response_text.strip()

    # Try direct JSON parse first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON array in the text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding ```json ... ``` block
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return []


class Planner:
    """Task planner that decomposes goals and dispatches workers.

    The planner is responsible for:
      - Understanding user intent
      - Breaking down complex tasks into parallelizable sub-tasks
      - Dispatching sub-tasks to Git Workers
      - Monitoring results and handling failures
      - Ensuring overall codebase health
    """

    def __init__(self, base_branch: str = "main", max_parallel: int = 3):
        """Initialize planner.

        Args:
            base_branch: Default branch to merge into.
            max_parallel: Maximum number of parallel workers.
        """
        self.base_branch = base_branch
        self.max_parallel = max_parallel
        self.task_history = []  # Record of all planned/executed tasks

    def decompose(self, goal: str) -> list:
        """Decompose a goal into sub-tasks using LLM.

        Args:
            goal: User's high-level goal description.

        Returns:
            List of task dicts with keys: id, description, depends_on, priority.
        """
        msgs = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"请将以下目标分解为可并行的子任务：\n\n{goal}"},
        ]

        resp = call_llm(messages=msgs, max_tokens=4000)
        resp = parse_llm_response(resp)

        response_text = ""
        if resp and resp.content:
            response_text = "".join(
                getattr(b, "text", "") or "" for b in resp.content
            )

        tasks = _parse_task_plan(response_text)

        if not tasks:
            # Fallback: create a single task from the whole goal
            tasks = [{
                "id": "task-1",
                "description": goal,
                "depends_on": [],
                "priority": "high",
            }]

        # Ensure all required fields exist
        for i, t in enumerate(tasks):
            t.setdefault("id", f"task-{i+1}")
            t.setdefault("depends_on", [])
            t.setdefault("priority", "medium")

        return tasks

    def execute_tasks(self, tasks: list) -> dict:
        """Execute tasks respecting dependencies and parallelism limits.

        Strategy:
          1. Topological sort by dependencies
          2. Execute independent tasks in parallel (up to max_parallel)
          3. Collect results
          4. Check for issues → create follow-up tasks if needed

        Args:
            tasks: List of task dicts from decompose().

        Returns:
            Execution report dict with results per task and summary.
        """
        execution_id = f"exec-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # Build dependency graph and determine execution order
        completed = set()
        failed = set()
        results = {}  # task_id -> result string
        task_map = {t["id"]: t for t in tasks}

        # Simple execution loop: iterate until all done or stuck
        rounds = 0
        max_rounds = len(tasks) * 2  # Safety limit

        while len(completed) + len(failed) < len(tasks) and rounds < max_rounds:
            rounds += 1

            # Find ready tasks (all dependencies satisfied, not yet run)
            ready = []
            for t in tasks:
                tid = t["id"]
                if tid in completed or tid in failed:
                    continue
                deps = t.get("depends_on", [])
                if all(d in completed for d in deps):
                    ready.append(t)

            if not ready:
                # Check if we have unresolvable dependencies
                remaining = [t["id"] for t in tasks
                            if t["id"] not in completed and t["id"] not in failed]
                if remaining:
                    # Mark remaining as failed due to dependency issues
                    for tid in remaining:
                        failed.add(tid)
                        results[tid] = f"SKIPPED: dependency failure for {tid}"
                break

            # Limit parallelism
            batch = ready[:self.max_parallel]

            # Execute batch (sequentially for now — true parallelism needs asyncio/threading)
            # But each worker works in its own isolated worktree
            batch_results = {}
            for t in batch:
                tid = t["id"]
                desc = t["description"]
                print(f"  [Planner] 开始执行任务 {tid}: {desc[:60]}...")

                try:
                    result = run_git_worker(desc, base_branch=self.base_branch)
                    batch_results[tid] = result
                    completed.add(tid)
                    print(f"  [Planner] ✅ 任务 {tid} 完成")
                except Exception as e:
                    batch_results[tid] = f"EXECUTION ERROR: {e}"
                    failed.add(tid)
                    print(f"  [Planner] ❌ 任务 {tid} 失败: {e}")

            results.update(batch_results)

        # Build execution report
        elapsed = time.time() - start_time
        report = {
            "execution_id": execution_id,
            "goal": "",  # Will be filled by caller
            "total_tasks": len(tasks),
            "completed": len(completed),
            "failed": len(failed),
            "elapsed_seconds": round(elapsed, 1),
            "tasks": [],
            "issues": [],  # Detected issues for follow-up
        }

        for t in tasks:
            tid = t["id"]
            report["tasks"].append({
                "id": tid,
                "description": t.get("description", "")[:100],
                "status": "completed" if tid in completed else "failed",
                "result_summary": results.get(tid, "")[:500],
            })

        # Health check — look for signs of problems in results
        for tid, result in results.items():
            if "❌" in result or "失败" in result or "error" in result.lower():
                report["issues"].append({
                    "source_task": tid,
                    "type": "execution_failure",
                    "detail": result[:200],
                })

        self.task_history.append(report)
        return report

    def plan_and_execute(self, goal: str) -> str:
        """Full pipeline: decompose goal → execute tasks → return report.

        This is the main entry point for the planner.

        Args:
            goal: User's goal (e.g., "给项目添加完整的错误处理和日志系统")

        Returns:
            Human-readable execution report.
        """
        print(f"[Planner] 📋 目标: {goal}")
        print(f"[Planner] 正在分解任务...")

        tasks = self.decompose(goal)

        print(f"[Planner] 分解为 {len(tasks)} 个子任务:")
        for t in tasks:
            deps = f" (依赖: {t['depends_on']})" if t.get("depends_on") else ""
            print(f"  - [{t['id']}] [{t.get('priority', '?')}] {t['description'][:70]}{deps}")

        print(f"\n[Planner] 🚀 开始执行...")
        print("=" * 60)

        report = self.execute_tasks(tasks)
        report["goal"] = goal

        print("=" * 60)

        # Format human-readable output
        lines = [
            f"=== Planner 执行报告 ===",
            f"",
            f"目标: {goal}",
            f"执行 ID: {report['execution_id']}",
            f"总耗时: {report['elapsed_seconds']}s",
            f"",
            f"任务总览: {report['completed']}/{report['total_tasks']} 成功",
            f"",
            f"--- 各任务详情 ---",
        ]

        for t in report["tasks"]:
            icon = "✅" if t["status"] == "completed" else "❌"
            lines.append(f"{icon} [{t['id']}] {t['description']}")
            # Show truncated result
            summary = t.get("result_summary", "")
            if summary:
                # Truncate long results
                if len(summary) > 300:
                    summary = summary[:300] + "... (truncated)"
                for line in summary.split("\n")[:10]:
                    lines.append(f"    {line}")

        if report["issues"]:
            lines.extend([
                f"",
                f"--- ⚠️ 发现的问题 ---",
            ])
            for issue in report["issues"]:
                lines.append(
                    f"- [{issue.get('type', '?')}] 来自任务 {issue.get('source_task')}: "
                    f"{issue.get('detail', '')[:150]}"
                )
            lines.extend([
                f"",
                f"💡 建议: 这些问题可以作为新目标重新提交给 Planner 处理。",
            ])
        else:
            lines.extend([
                f"",
                f"--- 🎉 全部完成 ---",
                f"所有任务均成功执行，未发现遗留问题。",
            ])

        return "\n".join(lines)

    def get_status(self) -> str:
        """Return status of recent executions."""
        if not self.task_history:
            return "Planner 尚未执行过任何任务。"

        lines = ["=== Planner 历史记录 ===", ""]
        for i, report in enumerate(reversed(self.task_history[-5:])):
            lines.append(
                f"[{i+1}] {report.get('goal', '?')[:60]} | "
                f"{report['completed']}/{report['total_tasks']} 成功 | "
                f"{report['elapsed_seconds']}s"
            )
        return "\n".join(lines)


# Module-level singleton
PLANNER = Planner()

