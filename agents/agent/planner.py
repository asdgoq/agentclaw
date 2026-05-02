#!/usr/bin/env python3
"""Planner (任务规划器) - 任务分解、并行派发、自动迭代。

职责：
1. 接收用户目标，分解为可并行的子任务（支持微任务粒度）
2. 将子任务并发派发给 Git Worker（ThreadPoolExecutor，动态扩缩 1~50）
3. 每轮结束后自动扫描项目状态，发现可优化点，生成新一轮微任务
4. 循环直到 LLM 判断目标完全实现（返回空任务列表）

架构：

    用户目标 ("写一个编译器")
        ↓
    ┌──────────────┐
    │   Planner    │ ← LLM 分解 + 依赖分析 + 动态调度 + 自动迭代
    └──────┬───────┘
           │ 并发派发 (ThreadPoolExecutor, 动态 1~50 worker)
     ┌─────┼─────┬─────┬─── ... ───┐
     ↓     ↓     ↓     ↓           ↓
    W-1   W-2   W-3   W-4       W-N   ← 真正并行！各自独立 worktree
     │     │     │     │           │
     ↓     ↓     ↓     ↓           ↓
   [merge to main] ← rebase 或 merge --ff

    一轮完成 → 自动扫描项目 → 发现新任务 → 下一轮
              → LLM 返回 [] → 停止 → 输出报告

使用方式：
    from .planner import Planner
    p = Planner(max_parallel=50)
    result = p.plan_and_execute("写一个 Rust 编译器")
    # 全自动运行，用户只说了一句话
"""

import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .subagent import run_git_worker
from ..llm.llm import call_llm, parse_llm_response

# =====================================================================
# 系统提示词
# =====================================================================

PLANNER_SYSTEM = """你是一个高级任务规划器。你的职责是将目标分解为可并行的子任务。

## 规则
1. **文件级隔离优先**：不同文件的任务可以完全并行；同一文件的任务必须串行
2. **粒度自适应**：
   - 新项目/大功能：拆为模块级任务（每个负责一个文件或完整子功能）
   - 已有项目的优化/修复：可以拆到很细（改一个函数、加一行逻辑、修一个 bug）
3. **依赖要准确**：如果任务 B 需要任务 A 先完成的代码，必须标记 depends_on
4. **描述要具体**：包含要改什么文件、实现什么功能、预期产出

## 输出格式
输出 JSON 数组：
```json
[
  {
    "id": "task-1",
    "description": "具体描述：做什么、改哪个文件、产出什么",
    "depends_on": [],
    "priority": "high",
    "task_type": "feature"
  }
]
```

- `depends_on`: 依赖的 ID 列表，空数组=无依赖可并行
- `priority`: high / medium / low
- `task_type`: feature / fix / refactor / optimize / test

## 分解策略
- 新项目：按模块拆（词法分析、语法分析、语义分析...）
- 迭代优化：按改动点拆（每个文件的具体改动是一个 task）
- 全局性改动（如加日志）：按文件拆"""


ITERATION_SYSTEM = """你是一个代码审查和自动优化规划器。

根据当前项目状态和上一轮执行结果，发现可以继续优化的点，生成新一轮微任务。

## 输入
- 原始目标
- 当前项目结构（目录列表、关键文件摘要）
- 最近 git 提交记录
- 上一轮执行结果

## 输出格式
JSON 数组（同上），或者空数组 [] 表示"目标已完全实现，不需要更多任务"。

## 发现方向
1. Bug: 上一轮引入的问题、边界 case、遗漏的错误处理
2. 优化: 性能瓶颈、冗余代码、可简化的逻辑
3. 完善: 错误处理缺失、日志不足、文档注释
4. 测试: 缺少单元测试的模块
5. 需求扩展: 目标中提到但还没实现的功能点

## 收敛原则
- 如果项目已经很好地实现了原始目标，返回 []
- 不要为了生成任务而生成——每个任务都要有实际价值
- 聚焦高价值任务，不要泛滥"""


def _parse_task_plan(response_text: str) -> list:
    """从 LLM 响应中提取 JSON 任务计划。"""
    text = response_text.strip()

    # 直接 JSON 解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 在文本中查找 JSON 数组
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # 查找 ```json ... ``` 代码块
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
    """支持真正并行执行和自动迭代的任务规划器。

    特性：
      - 通过 ThreadPoolExecutor 实现真正的并行（动态 1~50 个 worker）
      - 自动迭代优化，直到目标完成
      - 基于波次的执行，尊重依赖顺序
      - 智能并发：根据就绪任务数动态调整 worker 数量
    """

    def __init__(self, base_branch: str = "main", max_parallel: int = 50):
        """初始化规划器。

        Args:
            base_branch: 默认合并到的分支。
            max_parallel: 最大并行 worker 数（默认 50）。
        """
        self.base_branch = base_branch
        self.max_parallel = max_parallel
        self.task_history = []
        self._exec_counter = 0

    # ------------------------------------------------------------------
    # 任务分解
    # ------------------------------------------------------------------

    def decompose(self, goal: str) -> list:
        """使用 LLM 将目标分解为子任务。"""
        msgs = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"请将以下目标分解为可并行的子任务：\n\n{goal}"},
        ]

        resp = call_llm(messages=msgs, max_tokens=6000)
        resp = parse_llm_response(resp)

        response_text = ""
        if resp and resp.content:
            response_text = "".join(
                getattr(b, "text", "") or "" for b in resp.content
            )

        tasks = _parse_task_plan(response_text)

        if not tasks:
            tasks = [{
                "id": "task-1",
                "description": goal,
                "depends_on": [],
                "priority": "high",
                "task_type": "feature",
            }]

        for i, t in enumerate(tasks):
            t.setdefault("id", f"task-{i+1}")
            t.setdefault("depends_on", [])
            t.setdefault("priority", "medium")
            t.setdefault("task_type", "feature")

        return tasks

    # ------------------------------------------------------------------
    # 并行执行引擎
    # ------------------------------------------------------------------

    def _execute_one(self, task: dict) -> tuple:
        """通过 git_worker 执行单个任务。返回 (task_id, status, result)。"""
        tid = task.get("id", "unknown")
        desc = task.get("description") or task.get("desc") or str(task)
        try:
            # 限流：错开并行 LLM 调用，避免 429
            with _llm_semaphore:
                result = run_git_worker(desc, base_branch=self.base_branch)
            return (tid, "OK", result)
        except Exception as e:
            return (tid, "ERROR", str(e))

    def execute_tasks(self, tasks: list) -> dict:
        """以真正并行的方式执行任务。

        策略：
          1. 按依赖关系进行拓扑排序（基于波次）
          2. 每个波次使用动态 worker 数量：min(就绪任务数, max_parallel)
          3. Worker 是通过 ThreadPoolExecutor 创建的真实 OS 线程
          4. 下一波次仅在当前波次完成后才开始
        """
        self._exec_counter += 1
        exec_id = f"exec-{self._exec_counter}-{uuid.uuid4().hex[:6]}"
        start_time = time.time()

        completed = set()
        failed = set()
        results = {}
        total = len(tasks)

        wave_num = 0
        max_waves = total + 20  # Safety limit

        while len(completed) + len(failed) < total and wave_num < max_waves:
            wave_num += 1

            # 查找就绪任务（所有依赖已满足）
            ready = [t for t in tasks
                     if t["id"] not in completed and t["id"] not in failed
                     and all(d in completed for d in t.get("depends_on", []))]

            if not ready:
                # 无法解析的依赖 — 将剩余任务标记为失败
                for t in tasks:
                    tid = t["id"]
                    if tid not in completed and tid not in failed:
                        failed.add(tid)
                        results[tid] = "SKIPPED: dependency failure"
                break

            # 动态 worker 数量：根据任务量伸缩，上限为 max_parallel
            n_workers = min(len(ready), self.max_parallel)
            print(f"\n  [Planner] Wave {wave_num}: {len(ready)} tasks, "
                  f"{n_workers} workers")

            # === 并行执行 ===
            wave_results = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_id = {
                    pool.submit(self._execute_one, t): t["id"]
                    for t in ready
                }

                for future in as_completed(future_to_id):
                    tid = future_to_id[future]
                    try:
                        tid_result, status, detail = future.result()
                        wave_results[tid_result] = detail
                        if status == "OK":
                            completed.add(tid)
                            print(f"    [Planner] OK {tid}")
                        else:
                            failed.add(tid)
                            print(f"    [Planner] FAIL {tid}: {detail[:80]}")
                    except Exception as e:
                        failed.add(tid)
                        wave_results[tid] = f"FUTURE_ERROR: {e}"
                        print(f"    [Planner] ERROR {tid}: {e}")

            results.update(wave_results)

        # 构建报告
        elapsed = time.time() - start_time
        report = {
            "execution_id": exec_id,
            "wave_count": wave_num,
            "goal": "",
            "total_tasks": total,
            "completed": len(completed),
            "failed": len(failed),
            "elapsed_seconds": round(elapsed, 1),
            "tasks_per_second": round(total / elapsed, 2) if elapsed > 0 else 0,
            "tasks": [],
            "issues": [],
        }

        for t in tasks:
            tid = t["id"]
            report["tasks"].append({
                "id": tid,
                "description": t.get("description", "")[:120],
                "status": "completed" if tid in completed else "failed",
                "result_summary": results.get(tid, "")[:500],
            })

        # 健康检查
        for tid, res in results.items():
            markers = ["error", "失败", "failed", "exception"]
            if any(m in res.lower() for m in markers):
                report["issues"].append({
                    "source_task": tid,
                    "type": "execution_failure",
                    "detail": res[:200],
                })

        self.task_history.append(report)
        return report

    # ------------------------------------------------------------------
    # 自动迭代：每轮结束后发现后续任务
    # ------------------------------------------------------------------

    def discover_follow_ups(self, original_goal: str, last_report: dict) -> list:
        """让 LLM 扫描项目状态并建议下一轮微任务。

        当目标完全实现时返回空列表。
        """
        from .tools import run_bash

        # 收集项目上下文
        try:
            structure = run_bash("find . -type f | head -80")
            recent_commits = run_bash("git log --oneline -20")
            git_stats = run_bash("git diff --stat HEAD~10 HEAD 2>/dev/null || echo 'no stats'")
        except Exception:
            structure = "(unable to read)"
            recent_commits = "(unable to read)"
            git_stats = "(unable to read)"

        # 汇总上一轮结果
        summary_parts = [
            f"Round result: {last_report.get('completed', 0)}/{last_report.get('total_tasks', 0)} ok",
            f"Issues: {len(last_report.get('issues', []))}",
            f"Speed: {last_report.get('tasks_per_second', 0)} tasks/s",
            f"Time: {last_report.get('elapsed_seconds', 0)}s",
            "",
            "Tasks executed:",
        ]
        for t in last_report.get("tasks", [])[:15]:
            icon = "+" if t["status"] == "completed" else "x"
            summary_parts.append(f"  [{icon}] [{t['id']}] {t['description']}")

        msgs = [
            {"role": "system", "content": ITERATION_SYSTEM},
            {"role": "user", "content": (
                f"原始目标: {original_goal}\n\n"
                f"=== 项目状态 ===\n"
                f"文件结构:\n{structure}\n\n"
                f"最近提交:\n{recent_commits}\n\n"
                f"变更统计:\n{git_stats}\n\n"
                f"=== 上一轮报告 ===\n"
                + "\n".join(summary_parts) +
                "\n\n请分析：目标是否已经完全实现了？如果还有需要做的，列出具体任务。"
                "如果已经完整实现了，返回空数组 []。"
            )},
        ]

        try:
            resp = call_llm(messages=msgs, max_tokens=4000)
            resp = parse_llm_response(resp)

            response_text = ""
            if resp and resp.content:
                response_text = "".join(
                    getattr(b, "text", "") or "" for b in resp.content
                )

            tasks = _parse_task_plan(response_text)

            for i, t in enumerate(tasks):
                t.setdefault("id", f"r{len(self.task_history)+1}-t{i+1}")
                t.setdefault("depends_on", [])
                t.setdefault("priority", "medium")
                t.setdefault("task_type", "optimize")

            return tasks
        except Exception as e:
            print(f"  [Planner] Iteration scan error: {e}")
            return []

    # ------------------------------------------------------------------
    # 主入口：全自动流水线
    # ------------------------------------------------------------------

    def plan_and_execute(self, goal: str) -> str:
        """全自动流水线：分解 → 执行 → 迭代 → 收敛。

        自动运行，直到 LLM 判断目标已完成。
        用户只需说一句话。

        Args:
            goal: 用户的一句话目标。

        Returns:
            多轮执行报告。
        """
        overall_start = time.time()
        all_reports = []
        round_num = 0
        MAX_ROUNDS = 100  # 硬性安全上限

        print(f"[Planner] Goal: {goal}")
        print(f"[Planner] Max workers: {self.max_parallel}")
        print("=" * 60)

        while round_num < MAX_ROUNDS:
            round_num += 1
            print(f"\n{'='*50}")
            print(f"[Planner] Round {round_num}")
            print(f"{'='*50}")

            if round_num == 1:
                print("[Planner] Decomposing goal...")
                tasks = self.decompose(goal)
            else:
                print("[Planner] Scanning project for follow-ups...")
                tasks = self.discover_follow_ups(goal, all_reports[-1])

                if not tasks:
                    print("\n[Planner] GOAL ACHIEVED - no more tasks needed.")
                    break

                print(f"[Planner] Found {len(tasks)} follow-up tasks")

            # 显示计划
            print(f"[Planner] This round: {len(tasks)} tasks")
            for t in tasks[:20]:
                tid = t.get("id", "?")
                tdesc = t.get("description") or t.get("desc", "(no desc)")
                deps = f" (deps: {t['depends_on']})" if t.get("depends_on") else ""
                print(f"  [{tid}] {str(tdesc)[:70]}{deps}")
            if len(tasks) > 20:
                print(f"  ... and {len(tasks)-20} more tasks")

            # 执行
            print(f"\n[Planner] Executing...")
            report = self.execute_tasks(tasks)
            report["goal"] = goal
            report["round"] = round_num
            all_reports.append(report)

            print(f"\n[Planner] Round {round_num} done: "
                  f"{report['completed']}/{report['total_tasks']} ok, "
                  f"{report['elapsed_seconds']}s, "
                  f"{report.get('tasks_per_second', 0)} tasks/s")

        # 最终报告
        overall_elapsed = time.time() - overall_start
        total_ok = sum(r["completed"] for r in all_reports)
        total_tasks = sum(r["total_tasks"] for r in all_reports)

        lines = [
            f"=== Planner Final Report ({len(all_reports)} rounds) ===",
            f"",
            f"Goal: {goal}",
            f"Total time: {round(overall_elapsed, 1)}s",
            f"Total commits: {total_ok}/{total_tasks}",
            f"Avg throughput: {round(total_tasks/overall_elapsed, 2)} tasks/s"
                if overall_elapsed > 0 else "",
            f"",
            f"--- Per-round details ---",
        ]

        for idx, r in enumerate(all_reports, 1):
            lines.extend([
                f"",
                f"Round {idx} ({r.get('elapsed_seconds', 0)}s, "
                f"{r.get('tasks_per_second', 0)} tasks/s):",
                f"  {r['completed']}/{r['total_tasks']} succeeded",
            ])
            shown = r["tasks"][:12]
            for t in shown:
                icon = "+" if t["status"] == "completed" else "x"
                tdesc = t.get("description") or t.get("desc", "?")
                lines.append(f"  [{icon}] [{t['id']}] {tdesc}")
            if len(r["tasks"]) > 12:
                lines.append(f"  ... +{len(r['tasks'])-12} more")

        all_issues = []
        for r in all_reports:
            all_issues.extend(r.get("issues", []))
        if all_issues:
            lines.extend([f"", f"--- Issues ({len(all_issues)}) ---"])
            for issue in all_issues[:8]:
                lines.append(f"- [{issue.get('type','?')}] {issue.get('detail','')[:150]}")
        else:
            lines.extend([
                f"",
                f"--- All clean ---",
                f"Goal achieved. No issues.",
            ])

        return "\n".join(lines)

    def get_status(self) -> str:
        """返回最近执行的状态。"""
        if not self.task_history:
            return "规划器尚未执行任何任务。"

        lines = ["=== Planner History ===", ""]
        for i, r in enumerate(reversed(self.task_history[-10:])):
            lines.append(
                f"[{i+1}] {r.get('goal','?')[:50]} | "
                f"{r['completed']}/{r['total_tasks']} | "
                f"{r['elapsed_seconds']}s | "
                f"{r.get('tasks_per_second',0)} t/s"
            )
        return "\n".join(lines)


# ------------------------------------------------------------------
# LLM API 调用的全局限流器（防止 429 错误）
# ------------------------------------------------------------------
# 同时最大 LLM 调用数。根据你的 API 配额调整。
# 大多数免费套餐允许约 3-5 RPM；付费套餐可能允许更多。
_LLM_SEMAPHORE_MAX = 3
_llm_semaphore = threading.Semaphore(_LLM_SEMAPHORE_MAX)


# 模块级单例
PLANNER = Planner()

