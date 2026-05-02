#!/usr/bin/env python3
"""Self-Discover 引擎 — LLM 自主发现推理结构。

源自 Google DeepMind 2024 论文 "Self-Discover: Large Language Models
Self-Compose Reasoning Structures" (Zhou et al., 2024).

核心思想（与 CoT / ToT / GoT 的关键区别）：

  CoT / ToT / GoT：推理结构是**人预设的**（链/树/图）
  Self-Discover：推理结构是 **LLM 自己发现的**

两阶段流程：

  Stage 1 — Module Discovery（模块发现）
    给定问题，LLM 自主生成一组「推理模块」。
    每个模块是一个推理动作的描述，例如：
      - "列出所有约束条件"
      - "检查边界情况和异常输入"
      - "对比 2-3 种可行方案"
      - "评估时间/空间复杂度"
    这些模块不是固定的——不同问题会触发不同的模块集合。

  Stage 2 — Structure Adaptation + Reasoning Execution（结构适配 + 推理执行）
    LLM 将发现的模块组织成**有向无环图（DAG）**：
      - 确定模块间的依赖关系（哪些必须先做）
      - 确定并行性（哪些可以同时做）
      - 按拓扑序逐个/逐批执行每个模块
    最终汇总所有模块的输出，得出结论。

与 GoT 的关系：
  Self-Discover 的输出天然就是一个 DAG 推理图，
  可以直接复用 ThoughtGraph 的数据结构来存储和展示。

使用方式：
    from agents.llm.self_discover import SelfDiscoverEngine

    engine = SelfDiscoverEngine()
    result = engine.discover("如何设计一个高并发的消息队列？")
    print(result.display())
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class ReasoningModule:
    """一个推理模块——Self-Discover Stage 1 的产出。

    每个 Module 代表一个具体的推理动作/思考步骤。
    """
    module_id: str = ""
    name: str = ""              # 模块名称（简短标识）
    description: str = ""       # 模块描述（做什么、为什么重要）
    reasoning_type: str = ""    # 推理类型: analysis / generation / verification / refinement / etc.
    priority: int = 0           # 优先级（0=最高），用于排序
    estimated_complexity: str = "medium"  # low / medium / high
    depends_on: list = field(default_factory=list)  # 依赖的其他 module_id（Stage 2 填充）
    output: str = ""            # 该模块执行后的输出内容（Stage 2 填充）
    status: str = "pending"     # pending / executing / done / skipped
    tokens_used: int = 0
    duration_ms: int = 0

    def __post_init__(self):
        if not self.module_id:
            self.module_id = f"M-{uuid.uuid4().hex[:6]}"


@dataclass
class DiscoverResult:
    """一次 Self-Discover 的完整结果。"""
    session_id: str = ""
    goal: str = ""
    context: str = ""
    status: str = "pending"       # pending / discovering / adapting / reasoning / completed / error
    # Stage 1 产物
    modules: list = field(default_factory=list)   # list[ReasoningModule]
    # Stage 2 产物
    execution_order: list = field(default_factory=list)  # 按拓扑序排列的 module_id 列表
    parallel_groups: list = field(default_factory=list)  # 可并行的分组 [[id1,id2], [id3], ...]
    # 最终结论
    conclusion: str = ""
    summary: str = ""             # 结构化摘要
    # 统计
    total_tokens: int = 0
    total_duration_ms: int = 0
    stage1_duration_ms: int = 0
    stage2_duration_ms: int = 0
    # 日志
    log_entries: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"SD-{uuid.uuid4().hex[:10]}"

    def log(self, stage: str, message: str):
        ts = time.strftime("%H:%M:%S", time.localtime())
        self.log_entries.append(f"[{ts}] [{stage}] {message}")

    @property
    def module_count(self) -> int:
        return len(self.modules)

    @property
    def completed_modules(self) -> int:
        return sum(1 for m in self.modules if m.status == "done")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["module_count"] = self.module_count
        d["completed_modules"] = self.completed_modules
        return d


# =====================================================================
# Self-Discover 引擎
# =====================================================================

class SelfDiscoverEngine:
    """Self-Discover 推理引擎。

    流程：
      1. discover() → Stage 1: 发现推理模块
      2. _adapt_structure() → Stage 2a: 将模块组织成 DAG
      3. _execute_reasoning() → Stage 2b: 按 DAG 执行推理
      4. _synthesize() → 综合结论
    """

    def __init__(self,
                 max_modules: int = 8,
                 max_reasoning_rounds: int = 12,
                 min_confidence: float = 0.6):
        self.max_modules = max_modules          # 最大发现模块数
        self.max_reasoning_rounds = max_reasoning_rounds  # 最大推理轮次
        self.min_confidence = min_confidence     # 结论最低置信度
        self._history: list[DiscoverResult] = []

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def discover(self, goal: str, context: str = "") -> DiscoverResult:
        """执行完整的 Self-Discover 流程。

        Args:
            goal: 要分析的问题或目标
            context: 额外上下文信息

        Returns:
            DiscoverResult 包含完整的结果、日志和统计
        """
        result = DiscoverResult(goal=goal, context=context)
        start_time = time.time() * 1000

        try:
            # ===== Stage 1: 模块发现 =====
            result.status = "discovering"
            result.log("Stage 1", f"开始推理模块发现: {goal[:80]}")
            modules = self._discover_modules(goal, context)
            result.modules = modules
            result.stage1_duration_ms = int(time.time() * 1000) - int(start_time)
            result.log("Stage 1",
                       f"发现 {len(modules)} 个推理模块: "
                       + ", ".join(f"'{m.name}'" for m in modules))

            if not modules:
                result.status = "completed"
                result.conclusion = "未能发现有效的推理模块。建议改用 think 或 think_deep 工具。"
                result.log("Stage 1", "未发现模块，终止")
                return result

            # ===== Stage 2a: 结构适配 =====
            result.status = "adapting"
            result.log("Stage 2a", "开始结构适配——组织模块为 DAG")
            order, parallel_groups = self._adapt_structure(result, modules)
            result.execution_order = order
            result.parallel_groups = parallel_groups
            result.log("Stage 2a",
                       f"DAG 构建完成: 执行顺序={len(order)} 个模块, "
                       f"并行组={len(parallel_groups)} 组")

            # ===== Stage 2b: 推理执行 =====
            result.status = "reasoning"
            result.log("Stage 2b", "开始按 DAG 执行推理")
            stage2_start = time.time() * 1000
            self._execute_reasoning(result, modules, order, parallel_groups,
                                     goal, context)
            result.stage2_duration_ms = int(time.time() * 1000) - stage2_start

            # ===== 综合结论 =====
            result.log("Synthesis", "综合所有模块输出，生成最终结论")
            self._synthesize(result, goal, context)

            result.status = "completed"
            result.log("完成", f"总耗时: {result.total_duration_ms}ms, "
                       f"模块: {result.completed_modules}/{result.module_count}")

        except Exception as e:
            result.status = "error"
            result.metadata["error"] = str(e)
            result.log("ERROR", str(e))

        finally:
            result.total_duration_ms = int(time.time() * 1000) - int(start_time)
            self._history.append(result)

        return result

    # ------------------------------------------------------------------
    # Stage 1: 模块发现 (Module Discovery)
    # ------------------------------------------------------------------

    def _discover_modules(self, goal: str, context: str) -> list[ReasoningModule]:
        """让 LLM 根据问题自主发现需要的推理模块。

        这是 Self-Discover 的核心创新点——不使用预定义的推理模板，
        而是让 LLM 分析问题后自己决定需要哪些推理步骤。
        """
        from .llm import call_llm, parse_llm_response

        context_block = f"\n\n上下文信息:\n{context}" if context else ""

        prompt = f"""你是一个推理架构师。给定一个复杂问题，你的任务是**发现**解决该问题所需的一组推理模块（reasoning modules）。

## 问题
{goal}
{context_block}

## 任务
请分析这个问题，然后输出 3~8 个推理模块。每个模块是一个独立的推理动作。

## 输出格式（严格 JSON）
```json
{{
  "modules": [
    {{
      "name": "模块短名（如：约束分析）",
      "description": "详细描述这个模块要做什么、为什么对解决这个问题重要",
      "reasoning_type": "analysis|generation|comparison|verification|refinement|brainstorming|constraint_check|edge_case|complexity_analysis|other",
      "priority": 1,
      "estimated_complexity": "low|medium|high"
    }}
  ]
}}
```

## 推理类型说明
- analysis: 分析/拆解问题的某个方面
- generation: 生成候选方案或想法
- comparison: 对比多个选项
- verification: 验证某个假设或方案的正确性
- refinement: 在已有结果上细化改进
- brainstorming: 头脑风暴，发散思维
- constraint_check: 检查约束条件是否满足
- edge_case: 检查边界情况/异常场景
- complexity_analysis: 分析时间/空间复杂度
- other: 其他类型

## 要求
1. 模块之间应该**互补且不重复**
2. 覆盖问题的**多个维度**（不要只关注一个方面）
3. 按**逻辑依赖关系**排列优先级（先做的排前面）
4. 复杂问题多给几个模块（5~8），简单问题少给（3~5）
5. 只输出 JSON，不要其他文字"""

        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        parsed = parse_llm_response(response)

        # 提取文本内容
        text = ""
        for block in parsed.content:
            if hasattr(block, 'text'):
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        return self._parse_modules(text)

    def _parse_modules(self, text: str) -> list[ReasoningModule]:
        """从 LLM 输出中解析出推理模块列表。"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        modules = []
        for i, m in enumerate(data.get("modules", [])):
            if not m.get("name"):
                continue
            module = ReasoningModule(
                name=m["name"].strip(),
                description=m.get("description", "").strip(),
                reasoning_type=m.get("reasoning_type", "analysis"),
                priority=int(m.get("priority", i + 1)),
                estimated_complexity=m.get("estimated_complexity", "medium"),
            )
            modules.append(module)

        # 按 priority 排序
        modules.sort(key=lambda m: m.priority)
        return modules[:self.max_modules]

    # ------------------------------------------------------------------
    # Stage 2a: 结构适配 (Structure Adaptation)
    #
    # 让 LLM 决定模块间的依赖关系，构建 DAG 执行计划
    # ------------------------------------------------------------------

    def _adapt_structure(self, result: DiscoverResult,
                         modules: list[ReasoningModule]) -> tuple[list[str], list[list[str]]]:
        """将模块组织成有向无环图（DAG），确定执行顺序和并行分组。

        Returns:
            (execution_order, parallel_groups)
            - execution_order: 拓扑序的 module_id 列表
            - parallel_groups: 可并行执行的模块分组
        """
        from .llm import call_llm, parse_llm_response

        # 构建模块摘要供 LLM 参考
        module_list = "\n".join(
            f"  {i+1}. [{m.module_id}] {m.name} "
            f"(type={m.reasoning_type}, complexity={m.estimated_complexity})\n"
            f"     {m.description[:120]}"
            for i, m in enumerate(modules)
        )

        prompt = f"""你是一个推理结构规划器。给定一组推理模块，请将它们组织成一个**有向无环图（DAG）**执行计划。

## 推理模块
{module_list}

## 任务
1. 确定模块间的**依赖关系**（哪些模块的输出会被其他模块需要）
2. 将模块分组，**可并行执行的放在同一组**
3. 输出执行计划

## 输出格式（严格 JSON）
```json
{{
  "execution_plan": [
    {{
      "group": 1,
      "parallel": true,
      "modules": ["模块ID1", "模块ID2"],
      "description": "这组模块可以并行执行"
    }},
    {{
      "group": 2,
      "parallel": false,
      "modules": ["模块ID3"],
      "description": "依赖第1组的输出"
    }}
  ],
  "dependencies": {{
    "模块ID3": ["模块ID1", "模块ID2"],
    "模块ID4": ["模块ID3"]
  }}
}}
```

## 规则
- 如果模块 B 需要模块 A 的分析结果才能进行，则 B 依赖于 A
- 同组内的模块必须是无依赖关系的（真正可并行）
- 一般 2~4 个 group
- 只输出 JSON"""

        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        parsed = parse_llm_response(response)

        text = ""
        for block in parsed.content:
            if hasattr(block, 'text'):
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        return self._parse_execution_plan(text, modules)

    def _parse_execution_plan(self, text: str,
                               modules: list[ReasoningModule]) -> tuple[list[str], list[list[str]]]:
        """解析执行计划，返回拓扑序和并行分组。"""
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            # fallback: 所有模块按原顺序串行执行
            order = [m.module_id for m in modules]
            return order, [[mid] for mid in order]

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            order = [m.module_id for m in modules]
            return order, [[mid] for mid in order]

        # 解析执行计划
        plan = data.get("execution_plan", [])
        dependencies = data.get("dependencies", {})

        # 构建 execution_order 和 parallel_groups
        execution_order = []
        parallel_groups = []

        for group_info in plan:
            group_module_ids = group_info.get("modules", [])
            # 验证 module_id 存在
            valid_ids = [mid for mid in group_module_ids
                         if any(m.module_id == mid for m in modules)]
            if valid_ids:
                execution_order.extend(valid_ids)
                parallel_groups.append(valid_ids)

        # 补漏：如果有模块没被包含在计划中，追加到末尾
        all_planned = set(execution_order)
        for m in modules:
            if m.module_id not in all_planned:
                execution_order.append(m.module_id)
                parallel_groups.append([m.module_id])

        # 将依赖关系写入模块
        dep_map = {}
        for mid, deps in dependencies.items():
            dep_map[mid] = deps
        for m in modules:
            m.depends_on = dep_map.get(m.module_id, [])

        if not execution_order:
            execution_order = [m.module_id for m in modules]
            parallel_groups = [[mid] for mid in execution_order]

        return execution_order, parallel_groups

    # ------------------------------------------------------------------
    # Stage 2b: 推理执行 (Reasoning Execution)
    #
    # 按照适配好的 DAG 结构，逐步执行每个推理模块
    # ------------------------------------------------------------------

    def _execute_reasoning(self, result: DiscoverResult,
                           modules: list[ReasoningModule],
                           execution_order: list[str],
                           parallel_groups: list[list[str]],
                           goal: str, context: str):
        """按 DAG 结构执行推理，同组的模块并行处理（串行调 LLM）。"""

        module_map = {m.module_id: m for m in modules}

        # 收集已完成模块的输出，作为后续模块的上下文
        completed_outputs: dict[str, str] = {}

        for group_idx, group_ids in enumerate(parallel_groups):
            group_name = f"G{group_idx+1}"
            result.log("Stage 2b",
                       f"[{group_name}] 执行模块: "
                       + ", ".join(module_map.get(mid, mid).name for mid in group_ids))

            for mid in group_ids:
                module = module_map.get(mid)
                if not module:
                    continue

                module.status = "executing"
                mod_start = time.time() * 1000

                # 构建该模块的推理 prompt
                output = self._execute_single_module(
                    module, completed_outputs, goal, context, modules
                )

                module.output = output
                module.status = "done"
                module.duration_ms = int(time.time() * 1000) - mod_start
                completed_outputs[mid] = output
                result.total_tokens += module.tokens_used

                result.log("Stage 2b",
                           f"[{group_name}] ✓ {module.name} "
                           f"({module.duration_ms}ms, {module.tokens_used} tok)")

    def _execute_single_module(self, module: ReasoningModule,
                                completed_outputs: dict[str, str],
                                goal: str, context: str,
                                all_modules: list[ReasoningModule]) -> str:
        """执行单个推理模块，调用 LLM 完成该模块定义的推理动作。"""
        from .llm import call_llm, parse_llm_response

        # 构建依赖模块的输出摘要
        dependency_context = ""
        if module.depends_on:
            dep_summaries = []
            for dep_id in module.depends_on:
                dep_output = completed_outputs.get(dep_id, "")
                dep_module = next((m for m in all_modules if m.module_id == dep_id), None)
                if dep_module and dep_output:
                    dep_summaries.append(
                        f"### [{dep_module.name}] 的推理结果\n{dep_output[:800]}"
                    )
            if dep_summaries:
                dependency_context = (
                    "\n\n## 前置模块的推理结果（作为你本次推理的输入）\n"
                    + "\n\n".join(dep_summaries)
                )

        type_hints = {
            "analysis": "请进行深入分析，给出结构化的要点。",
            "generation": "请生成 2-4 个可行的候选方案或想法。",
            "comparison": "请从多个维度对比各选项的优劣。",
            "verification": "请验证给出的假设/方案是否正确，指出潜在问题。",
            "refinement": "请在已有基础上进一步细化和改进。",
            "brainstorming": "请发散思维，尽可能多地列出相关想法。",
            "constraint_check": "请检查所有约束条件是否满足。",
            "edge_case": "请考虑边界情况和可能的异常场景。",
            "complexity_analysis": "请分析时间和空间复杂度。",
            "other": "请完成该推理任务。",
        }
        hint = type_hints.get(module.reasoning_type, "请完成该推理任务。")

        ctx_block = ("\n" + context) if context else ""

        prompt = """你是推理执行引擎的一个模块。请完成指定的推理任务。

## 原始问题
""" + goal + """

""" + ctx_block + dependency_context + """

## 你的推理模块
- 名称: """ + module.name + """
- 描述: """ + module.description + """
- 类型: """ + module.reasoning_type + """

## 指令
""" + hint + """

请直接输出该模块的推理结果。要求：
1. 内容具体、有深度，不要泛泛而谈
2. 使用结构化的格式（列表/表格/分段）
3. 如果是分析类模块，给出明确的结论
4. 长度控制在 200~600 字"""

        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        parsed = parse_llm_response(response)

        text = ""
        for block in parsed.content:
            if hasattr(block, 'text'):
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        # 粗略估算 token 数
        module.tokens_used = len(text) // 2 + 200
        return text.strip()

    # ------------------------------------------------------------------
    # 综合 (Synthesis)
    #
    # 汇总所有模块输出，生成最终结论
    # ------------------------------------------------------------------

    def _synthesize(self, result: DiscoverResult, goal: str, context: str):
        """综合所有模块的推理输出，生成最终结论。"""
        from .llm import call_llm, parse_llm_response

        # 构建所有模块输出的摘要
        outputs_summary = ""
        for m in result.modules:
            if m.status == "done" and m.output:
                outputs_summary += (
                    f"\n### 模块: {m.name}\n"
                    f"- 类型: {m.reasoning_type}\n"
                    f"- 推理结果:\n{m.output[:500]}\n"
                )

        if not outputs_summary:
            result.conclusion = "推理执行未产生有效输出。"
            return

        ctx_block2 = ("\n" + context) if context else ""

        prompt = """你是综合推理引擎。下面是针对一个问题的多个推理模块的输出结果。

## 原始问题
""" + goal + """

""" + ctx_block2 + """

## 各推理模块的输出
""" + outputs_summary + """

## 任务
请基于以上所有模块的推理结果，综合出一个**清晰、有条理的最终结论**。

输出格式：
1. **关键发现**（3-5 个最重要的要点）
2. **最终结论**（直接回答原始问题）
3. **建议行动**（如果适用）

要求：
- 结论要**具体可操作**，不要泛泛而谈
- 如果各模块结论有冲突，明确指出并给出判断依据
- 控制在 300~800 字"""

        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        parsed = parse_llm_response(response)

        text = ""
        for block in parsed.content:
            if hasattr(block, 'text'):
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        result.conclusion = text.strip()

        # 生成摘要
        result.summary = self._build_summary(result)

    def _build_summary(self, result: DiscoverResult) -> str:
        """构建结构化的结果摘要。"""
        lines = [
            f"Self-Discover 完成 | 问题: {result.goal[:60]}",
            f"模块数: {result.completed_modules}/{result.module_count}",
            f"执行阶段: 发现→适配→推理→综合",
            f"总耗时: {result.total_duration_ms}ms",
            f"",
            f"发现的推理模块:",
        ]
        for m in result.modules:
            status_mark = "✓" if m.status == "done" else "✗"
            lines.append(
                f"  {status_mark} [{m.name}] ({m.reasoning_type}, "
                f"{m.estimated_complexity}) {m.duration_ms}ms"
            )

        if result.parallel_groups:
            lines.append("")
            lines.append("执行计划 (DAG):")
            for i, group in enumerate(result.parallel_groups):
                module_map = {m.module_id: m for m in result.modules}
                names = [module_map.get(gid, "?").name for gid in group]
                parallel_tag = " [可并行]" if len(group) > 1 else ""
                lines.append(f"  第{i+1}步:{parallel_tag} {' + '.join(names)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 展示 & 历史
    # ------------------------------------------------------------------

    def display(self, result: DiscoverResult) -> str:
        """生成人类可读的结果展示。"""
        sections = []

        # Header
        sections.append("=" * 60)
        sections.append(f"  Self-Discover 推理结果  [{result.session_id}]")
        sections.append(f"  状态: {result.status} | "
                       f"模块: {result.completed_modules}/{result.module_count} | "
                       f"耗时: {result.total_duration_ms}ms")
        sections.append("=" * 60)

        # Stage 1: 发现的模块
        sections.append("")
        sections.append("📦 Stage 1: 推理模块发现")
        sections.append("-" * 40)
        for i, m in enumerate(result.modules, 1):
            status_icon = "✅" if m.status == "done" else "⏭️" if m.status == "skipped" else "❌"
            sections.append(
                f"  {i}. {status_icon} {m.name}"
                f"\n     类型: {m.reasoning_type} | "
                f"复杂度: {m.estimated_complexity} | "
                f"耗时: {m.duration_ms}ms"
                f"\n     {m.description[:100]}"
            )
            if m.depends_on:
                dep_names = [next((nm.name for nm in result.modules
                                   if nm.module_id == did), did)
                             for did in m.depends_on]
                sections.append(f"     依赖: {', '.join(dep_names)}")

        # Stage 2: 执行计划 (DAG)
        if result.parallel_groups:
            sections.append("")
            sections.append("🔀 Stage 2: 推理结构 (DAG)")
            sections.append("-" * 40)
            module_map = {m.module_id: m for m in result.modules}
            for i, group in enumerate(result.parallel_groups, 1):
                names = [module_map.get(gid, "?").name for gid in group]
                parallel_tag = " 🔄 并行" if len(group) > 1 else ""
                sections.append(f"  Step {i}:{parallel_tag} {' + '.join(names)}")

        # 各模块输出
        sections.append("")
        sections.append("📝 各模块推理输出")
        sections.append("-" * 40)
        for m in result.modules:
            if m.status == "done" and m.output:
                # 截断过长的输出
                output_preview = m.output[:300]
                if len(m.output) > 300:
                    output_preview += f"... (共 {len(m.output)} 字)"
                sections.append(
                    f"  【{m.name}】\n"
                    f"  {output_preview}\n"
                )

        # 最终结论
        if result.conclusion:
            sections.append("")
            sections.append("🎯 最终结论")
            sections.append("-" * 40)
            sections.append(result.conclusion)

        # 结构化摘要
        if result.summary:
            sections.append("")
            sections.append("📊 摘要")
            sections.append("-" * 40)
            sections.append(result.summary)

        return "\n".join(sections)

    def get_history(self, limit: int = 10) -> list[DiscoverResult]:
        """获取历史记录。"""
        return self._history[-limit:]

    def clear_history(self):
        """清空历史记录。"""
        self._history.clear()

