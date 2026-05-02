#!/usr/bin/env python3
"""思维链（Chain of Thought）引擎。

提供结构化的推理能力，让 Agent 在执行动作前先进行显式的思考步骤。

核心概念：
  - ThoughtStep: 单个推理步骤，包含内容、类型、置信度
  - ThoughtChain: 一条完整的推理链，有序组织多个 ThoughtStep
  - ThoughtChainEngine: 引擎，管理思维链的生命周期（创建→推理→评估→持久化）

推理模式：
  1. off       — 关闭（默认，不使用思维链）
  2. implicit  — 隐式（在系统提示词中引导 LLM 自行思考，不单独记录）
  3. explicit  — 显式（每次调用 LLM 前先生成独立的思考步骤，可查看/回溯）
  4. deep      — 深度推理（多轮思考：分析→假设→验证→结论，每步独立 LLM 调用）

使用方式：
    from agents.llm.thought_chain import ThoughtChainEngine

    engine = ThoughtChainEngine()
    chain = engine.create_chain("用户的问题或目标")
    result = engine.think_deeply("具体要推理的问题", max_steps=5)
    # 或在 agent_loop 中自动触发
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Callable


# =====================================================================
# 枚举与数据结构
# =====================================================================

class ThoughtType(str, Enum):
    """推理步骤的类型。"""
    ANALYSIS = "analysis"         # 分析问题、拆解要素
    HYPOTHESIS = "hypothesis"     # 提出假设或可能方案
    DEDUCTION = "deduction"       # 逻辑推导
    VERIFICATION = "verification" # 验证某个想法
    CORRECTION = "correction"     # 纠正之前的错误思路
    CONCLUSION = "conclusion"     # 得出结论
    REFLECTION = "reflection"     # 反思/自我审视
    CONTEXT_GATHERING = "context_gathering"  # 收集上下文信息


class ThinkingMode(str, Enum):
    """思维链模式。"""
    OFF = "off"               # 关闭
    IMPLICIT = "implicit"     # 隐式（prompt 引导）
    EXPLICIT = "explicit"     # 显式（记录每步）
    DEEP = "deep"             # 深度推理（多轮独立调用）


@dataclass
class ThoughtStep:
    """单个推理步骤。"""
    step_id: str = ""
    step_number: int = 0
    thought_type: str = ThoughtType.ANALYSIS.value
    content: str = ""
    confidence: float = 0.8  # 0.0 ~ 1.0
    tokens_used: int = 0
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.step_id:
            self.step_id = _short_id()
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_display(self) -> str:
        """生成适合终端展示的单行摘要。"""
        type_icon = {
            "analysis": "🔍",
            "hypothesis": "💡",
            "deduction": "➡️",
            "verification": "✅",
            "correction": "🔧",
            "conclusion": "🎯",
            "reflection": "🪞",
            "context_gathering": "📚",
        }.get(self.thought_type, "💭")
        conf_icon = "◆" if self.confidence >= 0.8 else "◇" if self.confidence >= 0.5 else "○"
        return f"  {type_icon} [Step {self.step_number}] ({self.thought_type}) {conf_icon} {self.content[:100]}"


@dataclass
class ThoughtChain:
    """一条完整的思维链。"""
    chain_id: str = ""
    goal: str = ""
    mode: str = ThinkingMode.EXPLICIT.value
    steps: list = field(default_factory=list)
    status: str = "pending"  # pending / thinking / completed / error / aborted
    total_tokens: int = 0
    total_duration_ms: int = 0
    created_at: str = ""
    completed_at: str = ""
    conclusion: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.chain_id:
            self.chain_id = f"TC-{uuid.uuid4().hex[:10]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def last_step(self) -> Optional[ThoughtStep]:
        return self.steps[-1] if self.steps else None

    def add_step(self, step: ThoughtStep):
        step.step_number = len(self.steps) + 1
        self.steps.append(step)
        self.total_tokens += step.tokens_used
        self.total_duration_ms += step.duration_ms

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_count"] = self.step_count
        return d

    def to_session_entry(self) -> dict:
        """转换为会话记录格式（用于存入 JSONL）。"""
        return {
            "type": "thought_chain",
            "chainId": self.chain_id,
            "goal": self.goal,
            "mode": self.mode,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "totalTokens": self.total_tokens,
            "totalDurationMs": self.total_duration_ms,
            "conclusion": self.conclusion,
            "createdAt": self.created_at,
            "completedAt": self.completed_at,
        }

    def to_compact_summary(self) -> str:
        """生成压缩版摘要（用于上下文注入）。"""
        parts = [f"[思维链] 目标: {self.goal[:80]}"]
        for s in self.steps:
            parts.append(f"  Step {s.step_number} [{s.thought_type}] (置信度{s.confidence:.0%}): {s.content[:120]}")
        if self.conclusion:
            parts.append(f"  结论: {self.conclusion[:150]}")
        return "\n".join(parts)

    def display(self) -> str:
        """生成完整的终端展示。"""
        lines = [
            f"{'='*60}",
            f"思维链: {self.chain_id}",
            f"目标:   {self.goal}",
            f"模式:   {self.mode} | 状态: {self.status} | 步数: {self.step_count}",
            f"Token:  {self.total_tokens} | 耗时: {self.total_duration_ms}ms",
            f"{'='*60}",
        ]
        for s in self.steps:
            lines.append(s.to_display())
        if self.conclusion:
            lines.extend(["", f"  🎯 结论: {self.conclusion}"])
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# =====================================================================
# ID 生成
# =====================================================================

_existing_ids: set = set()


def _short_id(length: int = 8) -> str:
    for _ in range(100):
        cid = uuid.uuid4().hex[:length]
        if cid not in _existing_ids:
            _existing_ids.add(cid)
            return cid
    return uuid.uuid4().hex[:length]


# =====================================================================
# 系统提示词模板
# =====================================================================

IMPLICIT_SYSTEM_TEMPLATE = """\
{base_system}

## 思维链要求（隐式模式）
在回答问题和执行工具前，请在内心按以下步骤进行思考：
1. **理解**：用户真正想要什么？核心问题是什么？
2. **分析**：涉及哪些文件/模块？有哪些可能的方案？
3. **规划**：最优的执行顺序是什么？是否有依赖关系？
4. **预判**：可能出现什么问题？如何避免？

不需要输出思考过程，但要让你的回答体现这些思考的结果。
"""

EXPLICIT_THINK_PROMPT = """\
你是一个严谨的推理助手。请对以下问题进行逐步思考。

## 问题/任务
{goal}

## 上下文（可选）
{context}

## 要求
请按顺序输出 1~5 个推理步骤，每个步骤用 JSON 格式：
```json
{{
  "thought_type": "analysis|hypothesis|deduction|verification|conclusion",
  "content": "这一步的具体思考内容",
  "confidence": 0.0~1.0
}}
```

思考类型说明：
- analysis: 分析问题本质、拆解关键要素
- hypothesis: 提出可能的解决方案或假设
- deduction: 从已知信息进行逻辑推导
- verification: 验证某个想法是否合理
- conclusion: 总结得出最终结论

最后一步应该是 conclusion，给出明确的行动方向。
只输出 JSON 数组，不要有其他文字。
"""

DEEP_THINK_STEP_PROMPTS = {
    ThoughtType.CONTEXT_GATHERING: """\
## 第 1 步：收集上下文
问题：{goal}
请分析解决这个问题需要哪些信息？当前已知的约束条件是什么？
输出一个 context_gathering 类型的思考步骤。
""",

    ThoughtType.ANALYSIS: """\
## 第 N 步：深度分析
问题：{goal}
前序思考：{previous_steps}
请深入分析问题的核心矛盾和关键难点。
输出一个 analysis 类型的思考步骤。
""",

    ThoughtType.HYPOTHESIS: """\
## 第 N 步：提出假设
基于前面的分析，请提出 2-3 个可能的解决思路或假设方案。
对每个方案评估其可行性和风险。
输出一个 hypothesis 类型的思考步骤。
""",

    ThoughtType.DEDUCTION: """\
## 第 N 步：逻辑推导
针对选定的方案，请详细推导具体的执行步骤。
考虑每一步的前置条件和可能的副作用。
输出一个 deduction 类型的思考步骤。
""",

    ThoughtType.VERIFICATION: """\
## 第 N 步：验证检查
请审查前面的推理过程：
1. 是否有逻辑漏洞？
2. 是否有遗漏的边界情况？
3. 推理结论是否足够可靠？
输出一个 verification 类型的思考步骤。如果发现问题，同时输出 correction 步骤。
""",

    ThoughtType.CONCLUSION: """\
## 最后一步：得出结论
综合以上所有推理步骤，给出最终的行动建议。
应该做什么？按什么顺序做？预期结果是什么？
输出一个 conclusion 类型的思考步骤。
""",
}


# =====================================================================
# 核心引擎
# =====================================================================

class ThoughtChainEngine:
    """思维链引擎。

    管理思维链的创建、执行、存储和查询。

    使用方式：
        engine = ThoughtChainEngine()

        # 方式 1：显式思考（单次 LLM 调用）
        chain = engine.think_explicit("如何优化这个函数？")

        # 方式 2：深度推理（多轮 LLM 调用）
        chain = engine.think_deeply("设计一个编译器架构", max_steps=6)

        # 方式 3：获取系统提示词增强（隐式模式）
        enhanced_system = engine.enhance_system_prompt(base_system)
    """

    def __init__(self, mode: ThinkingMode = ThinkingMode.OFF,
                 max_steps: int = 8,
                 min_confidence: float = 0.5,
                 on_think_complete: Optional[Callable] = None):
        """
        Args:
            mode: 默认推理模式
            max_steps: 最大推理步数（防止无限循环）
            min_confidence: 最低置信度阈值
            on_think_complete: 思维链完成时的回调
        """
        self.mode = mode
        self.max_steps = max_steps
        self.min_confidence = min_confidence
        self.on_think_complete = on_think_complete
        self._history: list[ThoughtChain] = []
        self._lock = threading.Lock()
        # ---- 结论回注机制 ----
        # think/think_deep 执行完后，结论会存到这里。
        # agent_loop 每轮 LLM 调用前会消费一次，消费后自动清空，
        # 避免同一结论被重复注入多轮。
        self._pending_conclusion: str = ""
        self._pending_goal: str = ""

    # ------------------------------------------------------------------
    # 模式管理
    # ------------------------------------------------------------------

    def set_mode(self, mode: ThinkingMode | str):
        """切换推理模式。"""
        if isinstance(mode, str):
            mode = ThinkingMode(mode)
        self.mode = mode

    def get_mode(self) -> ThinkingMode:
        return self.mode

    # ------------------------------------------------------------------
    # 思维链创建
    # ------------------------------------------------------------------

    def create_chain(self, goal: str, mode: str = None,
                     metadata: dict = None) -> ThoughtChain:
        """创建一条空的思维链。"""
        return ThoughtChain(
            goal=goal,
            mode=mode or self.mode.value,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # 核心推理方法
    # ------------------------------------------------------------------

    def think_explicit(self, goal: str, context: str = "",
                       mode: str = None) -> ThoughtChain:
        """显式思维链（单次 LLM 调用）。

        让 LLM 一次性输出完整的推理步骤序列。
        适用于大多数场景，token 开销小。
        """
        from ..llm.llm import call_llm, parse_llm_response

        chain = self.create_chain(goal, mode=mode)
        chain.status = "thinking"
        start_time = time.time() * 1000

        try:
            prompt = EXPLICIT_THINK_PROMPT.format(
                goal=goal,
                context=context or "（无额外上下文）",
            )

            resp = call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            resp = parse_llm_response(resp)

            response_text = ""
            if resp and resp.content:
                response_text = "".join(
                    getattr(b, "text", "") or "" for b in resp.content
                )

            steps = self._parse_steps(response_text)

            for s in steps:
                chain.add_step(s)

            # 提取结论
            if chain.steps:
                last = chain.last_step
                if last.thought_type == ThoughtType.CONCLUSION.value:
                    chain.conclusion = last.content
                else:
                    # 如果没有 conclusion 步骤，从最后一步提取
                    chain.conclusion = last.content

            chain.status = "completed"
            chain.completed_at = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime())

        except Exception as e:
            chain.status = "error"
            chain.metadata["error"] = str(e)
        finally:
            chain.total_duration_ms = int(time.time() * 1000 - start_time)

        with self._lock:
            self._history.append(chain)

        if self.on_think_complete:
            try:
                self.on_think_complete(chain)
            except Exception:
                pass

        return chain

    def think_deeply(self, goal: str, context: str = "",
                     max_steps: int = 6) -> ThoughtChain:
        """深度思维链（多轮独立 LLM 调用）。

        每个推理步骤都是一次独立的 LLM 调用，
        后续步骤可以看到前面的推理过程。
        适用于复杂决策场景。
        """
        from ..llm.llm import call_llm, parse_llm_response

        chain = self.create_chain(goal, mode=ThinkingMode.DEEP.value)
        chain.status = "thinking"
        start_time = time.time() * 1000
        actual_max = min(max_steps, self.max_steps)

        try:
            # 定义深度推理的步骤流程
            step_flow = [
                ThoughtType.CONTEXT_GATHERING,
                ThoughtType.ANALYSIS,
                ThoughtType.HYPOTHESIS,
                ThoughtType.DEDUCTION,
                ThoughtType.VERIFICATION,
                ThoughtType.CONCLUSION,
            ]

            previous_summary = ""

            for i, step_type in enumerate(step_flow[:actual_max]):
                step_start = time.time() * 1000

                # 构建该步骤的 prompt
                if i == 0:
                    prompt_template = DEEP_THINK_STEP_PROMPTS[step_type]
                    prompt = prompt_template.format(
                        goal=goal,
                        previous_steps="（第一步）",
                    )
                else:
                    prompt_template = DEEP_THINK_STEP_PROMPTS.get(step_type)
                    if not prompt_template:
                        continue
                    prompt = prompt_template.format(
                        goal=goal,
                        previous_steps=previous_summary[-800:] if previous_summary else "（无）",
                    )
                    # 替换占位符 "第 N 步" 为实际编号
                    prompt = prompt.replace("第 N 步", f"第 {i+1} 步")

                # 调用 LLM
                resp = call_llm(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                )
                resp = parse_llm_response(resp)

                response_text = ""
                if resp and resp.content:
                    response_text = "".join(
                        getattr(b, "text", "") or "" for b in resp.content
                    )

                step_duration = int(time.time() * 1000 - step_start)

                # 解析步骤
                parsed = self._parse_single_step(response_text, step_type)
                parsed.step_number = i + 1
                parsed.tokens_used = len(response_text) // 4  # 粗略估算
                parsed.duration_ms = step_duration

                chain.add_step(parsed)
                previous_summary += f"\nStep {i+1} [{parsed.thought_type}]: {parsed.content}"

                # 如果是 verification 且发现需要纠正
                if (step_type == ThoughtType.VERIFICATION and
                    parsed.confidence < self.min_confidence):
                    correction_step = ThoughtStep(
                        thought_type=ThoughtType.CORRECTION.value,
                        content=f"置信度过低({parsed.confidence:.0%})，需要重新审视问题。",
                        confidence=parsed.confidence,
                    )
                    chain.add_step(correction_step)

            # 设置结论
            for s in reversed(chain.steps):
                if s.thought_type == ThoughtType.CONCLUSION.value:
                    chain.conclusion = s.content
                    break
            if not chain.conclusion and chain.steps:
                chain.conclusion = chain.last_step.content

            chain.status = "completed"
            chain.completed_at = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime())

        except Exception as e:
            chain.status = "error"
            chain.metadata["error"] = str(e)
        finally:
            chain.total_duration_ms = int(time.time() * 1000 - start_time)

        with self._lock:
            self._history.append(chain)

        if self.on_think_complete:
            try:
                self.on_think_complete(chain)
            except Exception:
                pass

        return chain

    # ------------------------------------------------------------------
    # Prompt 增强（用于隐式模式）
    # ------------------------------------------------------------------

    def enhance_system_prompt(self, base_system: str) -> str:
        """在系统提示词中注入思维链引导（隐式模式）。"""
        if self.mode == ThinkingMode.IMPLICIT:
            return IMPLICIT_SYSTEM_TEMPLATE.format(base_system=base_system)
        elif self.mode == ThinkingMode.EXPLICIT:
            # 显式模式也在 system 中加入轻量引导
            return base_system + "\n\n## 注意\n在执行复杂任务前，可以使用 think 工具进行推理。\n"
        return base_system

    # ------------------------------------------------------------------
    # 解析方法
    # ------------------------------------------------------------------

    def _parse_steps(self, text: str) -> list[ThoughtStep]:
        """从 LLM 响应中解析多个 ThoughtStep。"""
        steps = []

        # 尝试直接解析 JSON 数组
        text = text.strip()
        data = None

        # 方式 1：直接 JSON
        try:
            data = json.loads(text)
            if isinstance(data, list):
                pass
            elif isinstance(data, dict):
                data = [data]
            else:
                data = None
        except (json.JSONDecodeError, ValueError):
            pass

        # 方式 2：查找 JSON 数组
        if data is None:
            match = __import__("re").search(r'\[.*\]', text, __import__("re").DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    if not isinstance(data, list):
                        data = None
                except (json.JSONDecodeError, ValueError):
                    data = None

        # 方式 3：查找 ```json 代码块
        if data is None:
            match = __import__("re").search(
                r'```(?:json)?\s*\n(.*?)\n```', text,
                __import__("re").DOTALL
            )
            if match:
                try:
                    data = json.loads(match.group(1))
                    if not isinstance(data, list):
                        data = [data]
                except (json.JSONDecodeError, ValueError):
                    data = None

        # 方式 4：逐行解析（处理非 JSON 格式的输出）
        if data is None:
            return self._parse_freeform_text(text)

        # 构建 ThoughtStep 列表
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    steps.append(ThoughtStep(
                        thought_type=item.get("thought_type", "analysis"),
                        content=item.get("content", ""),
                        confidence=float(item.get("confidence", 0.8)),
                        metadata={k: v for k, v in item.items()
                                  if k not in ("thought_type", "content", "confidence")},
                    ))

        return steps

    def _parse_single_step(self, text: str,
                           default_type: ThoughtType) -> ThoughtStep:
        """解析单个推理步骤的响应。"""
        text = text.strip()

        # 尝试 JSON 格式
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return ThoughtStep(
                    thought_type=data.get("thought_type", default_type.value),
                    content=data.get("content", text),
                    confidence=float(data.get("confidence", 0.8)),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # 纯文本：直接作为内容
        return ThoughtStep(
            thought_type=default_type.value,
            content=text,
            confidence=0.8,
        )

    def _parse_freeform_text(self, text: str) -> list[ThoughtStep]:
        """解析自由格式的文本为推理步骤（兜底方案）。"""
        steps = []
        # 按段落或编号分割
        lines = text.split("\n")
        current_content = []
        type_order = ["analysis", "hypothesis", "deduction",
                      "verification", "conclusion"]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 检测是否是新步骤的开始
            is_new_step = False
            for prefix in ["Step", "步骤", "1.", "2.", "3.", "4.", "5.",
                          "- ", "* ", "•"]:
                if line.startswith(prefix):
                    is_new_step = True
                    break

            if is_new_step and current_content:
                idx = len(steps)
                ttype = type_order[idx] if idx < len(type_order) else "analysis"
                steps.append(ThoughtStep(
                    thought_type=ttype,
                    content="\n".join(current_content).strip(),
                ))
                current_content = [line]
            else:
                current_content.append(line)

        if current_content:
            idx = len(steps)
            ttype = type_order[idx] if idx < len(type_order) else "conclusion"
            steps.append(ThoughtStep(
                thought_type=ttype,
                content="\n".join(current_content).strip(),
            ))

        return steps

    # ------------------------------------------------------------------
    # 历史查询
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 20) -> list[ThoughtChain]:
        """获取最近的思维链历史。"""
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def get_last_chain(self) -> Optional[ThoughtChain]:
        """获取最近的一条思维链。"""
        with self._lock:
            return self._history[-1] if self._history else None

    def clear_history(self):
        """清空历史记录。"""
        with self._lock:
            self._history.clear()

    # ------------------------------------------------------------------
    # 结论回注机制（供 agent_loop 消费）
    # ------------------------------------------------------------------

    def set_pending_conclusion(self, conclusion: str, goal: str = ""):
        """设置待回注的结论（think/think_deep 完成后调用）。"""
        self._pending_conclusion = conclusion
        self._pending_goal = goal

    def consume_pending_conclusion(self) -> Optional[str]:
        """
        消费待回注的结论（agent_loop 每轮 LLM 调用前调用）。
        返回结论文本并立即清空，保证每条结论只被注入一次。
        """
        if not self._pending_conclusion:
            return None
        result = self._pending_conclusion
        goal = self._pending_goal
        self._pending_conclusion = ""
        self._pending_goal = ""
        if goal:
            return f"[深度推理结论 — 目标: {goal}]\n{result}"
        return f"[思维链推理结论]\n{result}"

    def has_pending_conclusion(self) -> bool:
        """是否有待消费的结论。"""
        return bool(self._pending_conclusion)


# =====================================================================
# 模块级单例（供 s_full.py 直接使用）
# =====================================================================

THOUGHT_ENGINE = ThoughtChainEngine()

