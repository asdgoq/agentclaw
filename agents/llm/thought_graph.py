#!/usr/bin/env python3
"""思维图（Graph of Thought, GoT）引擎。

将推理过程建模为**有向无环图（DAG）**，支持：
  - 多条推理路径的并行探索
  - 路径间的交叉验证与信息融合
  - 环形依赖检测与冲突解决
  - 关键节点识别与推理瓶颈分析

核心思想（源自 Besta et al. 2023 "Graph of Thoughts"）：
  CoT (链):   A → B → C → D          （线性，单一路径）
  ToT (树):   A → [B1, B2, B3]        （分支，但无合并）
              B1 → C1
              B2 → C2
  GoT (图):   A → [B1, B2]            （DAG，支持合并与交叉引用）
              B1 ─→ C ←── B2
              C ──┬── D1
                  └── D2
              D1 ╳ D2 (交叉验证)

与 CoT / ToT 的关键区别：
  - CoT: 线性序列，单路径
  - ToT: 树状结构，有分支无合并（父子关系）
  - GoT: 图状结构，支持多入度（一个节点可接收多个前驱的输入）、
         合并操作（merge）、交叉验证（cross-validate）、
         以及基于图拓扑的推理流控制

适用场景：
  - 需要多角度论证后综合结论的复杂问题
  - 需要交叉验证的关键决策（如安全审计、代码审查）
  - 需要信息融合的多源分析任务
  - 有循环依赖需要拆解的系统设计

使用方式：
    from agents.llm.thought_graph import ThoughtGraphEngine

    engine = ThoughtGraphEngine()
    graph = engine.explore("这个架构方案是否可行？", mode="diverge_converge")
    # 或
    graph = engine.explore("排查性能瓶颈", mode="parallel_validate")
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# =====================================================================
# 枚举与数据结构
# =====================================================================

class GraphMode(str, Enum):
    """思维图的推理模式。"""
    DIVERGE_CONVERGE = "diverge_converge"   # 发散-收敛：先多路探索，再合并结论
    PARALLEL_VALIDATE = "parallel_validate"  # 并行验证：多条路径独立推理后交叉验证
    ITERATIVE_REFINE = "iterative_refine"    # 迭代精炼：逐步细化+反馈修正
    FULL_GRAPH = "full_graph"               # 完全图：LLM 自主决定图结构


class NodeType(str, Enum):
    """思维图中节点的类型。"""
    ROOT = "root"                     # 根节点（问题本身）
    DECOMPOSE = "decompose"           # 问题分解
    EXPLORE = "explore"               # 单角度探索/推理
    ANALYZE = "analyze"               # 深度分析
    HYPOTHESIZE = "hypothesize"       # 提出假设
    VERIFY = "verify"                 # 验证某个观点
    MERGE = "merge"                   # 合并多个输入
    CROSS_VALIDATE = "cross_validate" # 交叉验证
    CONFLICT_RESOLVE = "conflict_resolve"  # 冲突解决
    CONCLUDE = "conclude"             # 得出最终结论
    FEEDBACK = "feedback"             # 反馈/修正信号


class NodeStatus(str, Enum):
    """节点的状态。"""
    PENDING = "pending"
    READY = "ready"                   # 前驱已就绪，可以执行
    RUNNING = "running"
    DONE = "done"
    MERGED = "merged"                 # 已被合并到下游节点
    CONFLICT = "conflict"             # 与其他节点存在冲突
    PRUNED = "pruned"


@dataclass
class GraphNode:
    """思维图中的节点。

    与 ToT 的 TreeNode 不同，GraphNode 支持多入度（multiple parents），
    这是 GoT 的核心特征——允许来自不同路径的信息汇聚。
    """
    node_id: str = ""
    content: str = ""
    node_type: str = NodeType.EXPLORE.value
    status: str = NodeStatus.PENDING.value
    depth: int = 0

    # ---- 多入度支持（GoT 核心）----
    parent_ids: list = field(default_factory=list)   # 前驱节点列表（可多个）
    children_ids: list = field(default_factory=list)  # 后继节点列表

    # ---- 评分与元数据 ----
    score: float = 0.0
    confidence: float = 0.8
    tokens_used: int = 0
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"G-{uuid.uuid4().hex[:6]}"
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    @property
    def in_degree(self) -> int:
        """入度（前驱数量）。"""
        return len(self.parent_ids)

    @property
    def out_degree(self) -> int:
        """出度（后继数量）。"""
        return len(self.children_ids)

    @property
    def is_leaf(self) -> bool:
        return self.out_degree == 0

    @property
    def is_root(self) -> bool:
        return self.in_degree == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["in_degree"] = self.in_degree
        d["out_degree"] = self.out_degree
        d["is_leaf"] = self.is_leaf
        return d


@dataclass
class GraphEdge:
    """思维图中的边（有向边 from → to）。"""
    edge_id: str = ""
    source_id: str = ""   # 起点 node_id
    target_id: str = ""   # 终点 node_id
    relation: str = "depends_on"  # 边的关系类型
    weight: float = 1.0  # 边权重（表示依赖强度）
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = f"E-{uuid.uuid4().hex[:4]}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThoughtGraph:
    """一张完整的思维图（DAG）。

    示例结构（diverge_converge 模式）：

                    ┌→ [B1: 技术可行性] ─┐
                    │                      ↓
      [A: 问题] ────┤→ [B2: 成本评估] ──→ [D: 合并结论]
                    │                      ↑
                    └→ [B3: 风险分析] ────┘

    示例结构（parallel_validate 模式）：

      [A: 问题] ──→ [B1: 方案A分析] ──→ [C1: A的结论]
                └─→ [B2: 方案B分析] ──→ [C2: B的结论]
                                              ↓
                                        [V: 交叉验证]
                                              ↓
                                        [F: 最终结论]
    """
    graph_id: str = ""
    goal: str = ""
    mode: str = GraphMode.DIVERGE_CONVERGE.value
    status: str = "pending"     # pending / building / running / completed / error
    total_nodes: int = 0
    total_edges: int = 0
    total_tokens: int = 0
    total_duration_ms: int = 0
    conclusion: str = ""
    key_findings: list = field(default_factory=list)
    conflicts_resolved: int = 0
    created_at: str = ""
    completed_at: str = ""
    execution_log: list = field(default_factory=list)  # 执行过程日志
    metadata: dict = field(default_factory=dict)

    # 内部存储
    _nodes: dict = field(default_factory=dict, repr=False)   # node_id → GraphNode
    _edges: list = field(default_factory=list, repr=False)   # GraphEdge 列表
    _adj: dict = field(default_factory=lambda: defaultdict(list), repr=False)  # 邻接表

    def __post_init__(self):
        if not self.graph_id:
            self.graph_id = f"GoT-{uuid.uuid4().hex[:10]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    @property
    def nodes(self) -> dict[str, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> list[GraphEdge]:
        return self._edges

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def max_depth(self) -> int:
        """图中节点的最大深度。"""
        return max((n.depth for n in self._nodes.values()), default=0)

    # ------------------------------------------------------------------
    # 图操作
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> str:
        """添加节点。"""
        self._nodes[node.node_id] = node
        self.total_nodes += 1
        self.total_tokens += node.tokens_used
        self.total_duration_ms += node.duration_ms
        return node.node_id

    def add_edge(self, source_id: str, target_id: str,
                 relation: str = "depends_on", weight: float = 1.0) -> Optional[GraphEdge]:
        """添加有向边 source → target。同时更新两端的 parent/children 列表。"""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
        )
        self._edges.append(edge)
        self.total_edges += 1

        # 更新邻接表和节点关系
        self._adj[source_id].append(target_id)
        source_node = self._nodes[source_id]
        target_node = self._nodes[target_id]

        if target_id not in source_node.children_ids:
            source_node.children_ids.append(target_id)
        if source_id not in target_node.parent_ids:
            target_node.parent_ids.append(source_id)

        return edge

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def get_roots(self) -> list[GraphNode]:
        """获取所有根节点（入度为 0 的节点）。"""
        return [n for n in self._nodes.values() if n.is_root]

    def get_leaves(self) -> list[GraphNode]:
        """获取所有叶子节点（出度为 0 的节点）。"""
        return [n for n in self._nodes.values() if n.is_leaf]

    def get_successors(self, node_id: str) -> list[GraphNode]:
        """获取直接后继节点。"""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids
                if cid in self._nodes]

    def get_predecessors(self, node_id: str) -> list[GraphNode]:
        """获取直接前驱节点（GoT 特性：可能有多个）。"""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[pid] for pid in node.parent_ids
                if pid in self._nodes]

    def get_ancestors(self, node_id: str) -> set[str]:
        """获取所有祖先节点 ID（传递闭包）。"""
        ancestors = set()
        stack = list(self._nodes.get(node_id, GraphNode()).parent_ids)
        while stack:
            nid = stack.pop()
            if nid not in ancestors:
                ancestors.add(nid)
                node = self._nodes.get(nid)
                if node:
                    stack.extend(node.parent_ids)
        return ancestors

    def topological_sort(self) -> list[str]:
        """拓扑排序（Kahn 算法），返回按依赖顺序排列的 node_id 列表。"""
        in_degree = {nid: n.in_degree for nid, n in self._nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # 按深度排序，同层按 score 排序
            queue.sort(key=lambda nid: (self._nodes[nid].depth, -self._nodes[nid].score))
            nid = queue.pop(0)
            result.append(nid)
            for child_id in self._adj.get(nid, []):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        # 检测环
        if len(result) != len(self._nodes):
            # 存在环，返回已排序的部分 + 剩余节点
            remaining = [nid for nid in self._nodes if nid not in set(result)]
            result.extend(remaining)

        return result

    def log(self, message: str, detail: str = ""):
        """记录执行日志。"""
        self.execution_log.append({
            "time": time.strftime("%H:%M:%S", time.localtime()),
            "message": message,
            "detail": detail,
        })

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "graphId": self.graph_id,
            "goal": self.goal,
            "mode": self.mode,
            "status": self.status,
            "totalNodes": self.total_nodes,
            "totalEdges": self.total_edges,
            "maxDepth": self.max_depth,
            "totalTokens": self.total_tokens,
            "totalDurationMs": self.total_duration_ms,
            "conclusion": self.conclusion,
            "keyFindings": self.key_findings,
            "conflictsResolved": self.conflicts_resolved,
            "createdAt": self.created_at,
            "completedAt": self.completed_at,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
        }
        return d

    def to_session_entry(self) -> dict:
        """转换为会话记录格式（用于存入 JSONL）。"""
        return {
            "type": "thought_graph",
            "graphId": self.graph_id,
            "goal": self.goal,
            "mode": self.mode,
            "status": self.status,
            "totalNodes": self.total_nodes,
            "totalEdges": self.total_edges,
            "maxDepth": self.max_depth,
            "totalTokens": self.total_tokens,
            "totalDurationMs": self.total_duration_ms,
            "conclusion": self.conclusion,
            "keyFindings": self.key_findings[:5],
            "conflictsResolved": self.conflicts_resolved,
            "createdAt": self.created_at,
            "completedAt": self.completed_at,
            # 精简版节点信息
            "nodeSummary": [
                {"id": n.node_id, "parents": n.parent_ids, "depth": n.depth,
                 "type": n.node_type, "content": n.content[:120],
                 "score": n.score, "status": n.status}
                for n in self._nodes.values()
            ],
            # 精简版边信息
            "edgeSummary": [
                {"from": e.source_id, "to": e.target_id, "rel": e.relation}
                for e in self._edges
            ],
        }

    def to_compact_summary(self) -> str:
        """生成压缩版摘要（用于上下文注入）。"""
        parts = [
            f"[思维图 GoT] 目标: {self.goal[:80]}",
            f"模式: {self.mode} | 节点: {self.total_nodes} | "
            f"边: {self.total_edges} | 最大深度: {self.max_depth}",
        ]
        if self.key_findings:
            parts.append(f"关键发现: {'; '.join(self.key_findings[:3])}")
        if self.conclusion:
            parts.append(f"结论: {self.conclusion[:200]}")
        return "\n".join(parts)

    def display(self) -> str:
        """生成终端展示（ASCII 图形 + 结构信息）。"""
        lines = [
            "",
            "=" * 66,
            f"  思维图 (Graph of Thought): {self.graph_id}",
            f"  目标:   {self.goal}",
            f"  模式:   {self.mode} | 状态: {self.status}",
            f"  节点:   {self.total_nodes} | 边: {self.total_edges} | "
            f"最大深度: {self.max_depth}",
            f"  Token:  {self.total_tokens} | 耗时: {self.total_duration_ms}ms",
            f"  冲突解决: {self.conflicts_resolved} 次",
            "=" * 66,
        ]

        if not self._nodes:
            lines.append("  (空图)")
            return "\n".join(lines)

        # 拓扑排序展示
        sorted_ids = self.topological_sort()

        # 按 depth 分组显示
        by_depth: dict[int, list] = defaultdict(list)
        for nid in sorted_ids:
            node = self._nodes.get(nid)
            if node:
                by_depth[node.depth].append(node)

        type_icons = {
            "root": "[ROOT]",
            "decompose": "[DEC]",
            "explore": "[EXP]",
            "analyze": "[ANA]",
            "hypothesize": "[HYP]",
            "verify": "[VER]",
            "merge": "[MRG]",
            "cross_validate": "[XVAL]",
            "conflict_resolve": "[FIX]",
            "conclude": "[CON]",
            "feedback": "[FBK]",
        }

        for depth in sorted(by_depth.keys()):
            nodes = by_depth[depth]
            lines.append(f"\n  --- Depth {depth} ({len(nodes)} nodes) ---")
            for node in nodes:
                icon = type_icons.get(node.node_type, "[???]")
                parent_info = ""
                if node.parent_ids:
                    parent_info = f" ← [{', '.join(p[:6] for p in node.parent_ids)}]"
                score_str = f"(score={node.score:.2f})" if node.score > 0 else ""
                conf_str = f"conf={node.confidence:.0%}" if node.confidence else ""
                line = (
                    f"    {icon} {node.content[:70]}"
                    f" {parent_info}"
                    f" {score_str}"
                )
                if conf_str:
                    line += f" [{conf_str}]"
                lines.append(line)

        # 关键发现
        if self.key_findings:
            lines.extend(["", "  关键发现:"])
            for i, finding in enumerate(self.key_findings[:5], 1):
                lines.append(f"    {i}. {finding}")

        # 结论
        if self.conclusion:
            lines.extend(["", f"  结论: {self.conclusion}"])

        lines.append("=" * 66)
        return "\n".join(lines)


# =====================================================================
# LLM Prompt 模板
# =====================================================================

GOT_DECOMPOSE_PROMPT = """\
你是一个结构化思维助手。请将以下问题分解为多个独立的探索方向。

## 目标问题
{goal}

## 上下文
{context}

## 要求
请将问题分解为 {n_branches} 个独立的分析维度或探索方向。
每个方向应该：
1. 关注问题的不同侧面（避免重复）
2. 可以独立进行深入分析
3. 最终能汇聚为对问题的全面理解

输出 JSON 数组：
```json
{{
  "branches": [
    {{
      "direction": "探索方向的名称",
      "description": "该方向的具体描述和分析重点",
      "priority": 0.9
    }},
    ...
  ]
}}
```

priority 表示该方向的重要性（0.0~1.0）。
只输出 JSON，不要有其他文字。
"""

GOT_EXPLORE_PROMPT = """\
你是{direction}领域的专家。请从「{direction}」的角度深入分析以下问题。

## 目标问题
{goal}

## 分析方向
{direction}: {description}

## 已有的其他视角分析（供参考，不要重复）
{peer_context}

## 要求
请给出在这个方向上的详细分析和判断，包括：
1. 该角度下的核心观察
2. 支持论据或证据
3. 该角度下的初步结论
4. 置信度评估

输出 JSON：
```json
{{
  "analysis": "详细分析内容...",
  "key_points": ["要点1", "要点2", ...],
  "confidence": 0.85,
  "verdict": "该角度下的判断/建议"
}}
```
只输出 JSON。
"""

GOT_MERGE_PROMPT = """\
你是一个综合决策者。请将以下多个分析视角的结果合并为一个综合结论。

## 原始目标
{goal}

## 各视角分析结果
{inputs_summary}

## 要求
综合所有输入，给出：
1. 各视角的一致点和矛盾点
2. 综合后的核心结论
3. 如果存在分歧，说明如何权衡

输出 JSON：
```json
{{
  "consensus_points": ["一致点1", ...],
  "conflicts": ["矛盾点描述", ...],
  "resolution": "如何解决矛盾",
  "conclusion": "综合结论",
  "confidence": 0.85,
  "key_findings": ["关键发现1", ...]
}}
```
只输出 JSON。
"""

GOT_CROSS_VALIDATE_PROMPT = """\
你是一个严格的审查者。请对以下两条独立推理路径进行交叉验证。

## 原始目标
{goal}

## 路径 A 的结论
{path_a}

## 路径 B 的结论
{path_b}

## 要求
对比两条路径：
1. 它们的结论是否一致？
2. 如果不一致，哪条更可信？为什么？
3. 是否有双方都遗漏的点？

输出 JSON：
```json
{{
  "is_consistent": true/false,
  "agreement_level": 0.0~1.0,
  "discrepancies": ["差异点..."],
  "more_reliable_path": "A 或 B 或 两者互补",
  "validation_conclusion": "验证后的结论",
  "suggested_action": "下一步建议"
}}
```
只输出 JSON。
"""


# =====================================================================
# 核心引擎
# =====================================================================

class ThoughtGraphEngine:
    """思维图引擎。

    管理 GoT 推理的完整生命周期：
      1. 创建图（根节点 = 问题）
      2. 分解（将问题拆分为多个探索方向）
      3. 并行探索（每个方向独立推理）
      4. 合并/验证（汇聚多路结果）
      5. 迭代精炼（可选：根据反馈修正）

    使用方式：
        engine = ThoughtGraphEngine()

        # 发散-收敛模式（最常用）
        graph = engine.explore("这个方案是否可行？", mode="diverge_converge")

        # 并行验证模式
        graph = engine.explore("选择哪个数据库？", mode="parallel_validate",
                                n_branches=3)
    """

    def __init__(self, default_mode: GraphMode = GraphMode.DIVERGE_CONVERGE,
                 max_depth: int = 5,
                 n_branches: int = 3,
                 min_confidence: float = 0.5):
        """
        Args:
            default_mode: 默认推理模式
            max_depth: 最大图深度
            n_branches: 默认分解出的分支数
            min_confidence: 最低置信度阈值
        """
        self.default_mode = default_mode
        self.max_depth = max_depth
        self.n_branches = n_branches
        self.min_confidence = min_confidence
        self._history: list[ThoughtGraph] = []

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def explore(self, goal: str, context: str = "",
                mode: str = None,
                n_branches: int = None,
                max_depth: int = None) -> ThoughtGraph:
        """执行一次完整的思维图推理。

        Args:
            goal: 要分析的问题或目标
            context: 额外上下文
            mode: 推理模式 (diverge_converge / parallel_validate /
                  iterative_refine / full_graph)
            n_branches: 分解出的分支数
            max_depth: 最大深度

        Returns:
            完整的 ThoughtGraph 对象
        """
        gmode = GraphMode(mode or self.default_mode.value)
        branches = n_branches or self.n_branches
        depth_limit = max_depth or self.max_depth

        # 创建图
        graph = ThoughtGraph(
            goal=goal,
            mode=gmode.value,
            status="building",
        )

        start_time = time.time() * 1000

        try:
            # 根据模式选择不同的构建策略
            if gmode == GraphMode.DIVERGE_CONVERGE:
                self._build_diverge_converge(graph, goal, context,
                                             branches, depth_limit)
            elif gmode == GraphMode.PARALLEL_VALIDATE:
                self._build_parallel_validate(graph, goal, context,
                                              branches, depth_limit)
            elif gmode == GraphMode.ITERATIVE_REFINE:
                self._build_iterative_refine(graph, goal, context,
                                              branches, depth_limit)
            elif gmode == GraphMode.FULL_GRAPH:
                self._build_full_graph(graph, goal, context,
                                       branches, depth_limit)

            graph.status = "completed"
            graph.completed_at = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime())

        except Exception as e:
            graph.status = "error"
            graph.metadata["error"] = str(e)
            graph.log("执行出错", str(e))
        finally:
            graph.total_duration_ms = int(time.time() * 1000 - start_time)

        self._history.append(graph)
        return graph

    # ------------------------------------------------------------------
    # 模式 1: 发散-收敛 (Diverge-Converge)
    #
    #   [Root] → [Decompose] → [B1] [B2] [B3]  (发散：多路并行探索)
    #                          \    |    /
    #                           → [Merge] → [Conclusion]  (收敛：合并结论)
    # ------------------------------------------------------------------

    def _build_diverge_converge(self, graph: ThoughtGraph, goal: str,
                                 context: str, n_branches: int,
                                 max_depth: int):
        """发散-收敛模式：先多路探索，再合并。"""
        graph.log("开始发散-收敛模式", f"分支数={n_branches}")

        # Step 1: 创建根节点
        root = GraphNode(content=goal, node_type=NodeType.ROOT.value,
                         depth=0, status=NodeStatus.DONE.value)
        root_id = graph.add_node(root)
        graph.log("创建根节点", goal[:60])

        # Step 2: 分解问题
        decompose_id = self._step_decompose(graph, root_id, goal, context,
                                            n_branches)
        if not decompose_id:
            return

        # Step 3: 并行探索各分支
        branch_ids = self._step_explore_branches(graph, decompose_id, goal,
                                                  context, n_branches)
        if not branch_ids:
            return

        # Step 4: 合并各分支结果
        merge_id = self._step_merge(graph, branch_ids, goal, context)
        if not merge_id:
            return

        # Step 5: 得出结论
        self._step_conclude(graph, merge_id, goal)

        graph.log("完成", f"总节点: {graph.node_count}, 总边数: {graph.edge_count}")

    # ------------------------------------------------------------------
    # 模式 2: 并行验证 (Parallel Validate)
    #
    #   [Root] → [B1] → [C1] ─┐
    #           → [B2] → [C2] ─→ [Validate] → [Final]
    #           → [B3] → [C3] ─┘
    # ------------------------------------------------------------------

    def _build_parallel_validate(self, graph: ThoughtGraph, goal: str,
                                  context: str, n_branches: int,
                                  max_depth: int):
        """并行验证模式：多条路径独立推理后交叉验证。"""
        graph.log("开始并行验证模式", f"路径数={n_branches}")

        # Step 1: 根节点
        root = GraphNode(content=goal, node_type=NodeType.ROOT.value,
                         depth=0, status=NodeStatus.DONE.value)
        root_id = graph.add_node(root)

        # Step 2: 分解
        decompose_id = self._step_decompose(graph, root_id, goal, context,
                                            n_branches)
        if not decompose_id:
            return

        # Step 3: 每个分支独立完成完整推理（探索→结论）
        branch_conclusions = []
        branch_ids = self._step_explore_branches(graph, decompose_id, goal,
                                                  context, n_branches)
        for bid in branch_ids:
            # 每个分支再做一个结论节点
            bnode = graph.get_node(bid)
            if not bnode:
                continue
            conclude_node = GraphNode(
                content=f"[{bnode.content[:40]}] 的初步结论",
                node_type=NodeType.CONCLUDE.value,
                depth=bnode.depth + 1,
                confidence=bnode.confidence,
                score=bnode.score,
                status=NodeStatus.DONE.value,
            )
            conclude_id = graph.add_node(conclude_node)
            graph.add_edge(bid, conclude_id, relation="leads_to")
            branch_conclusions.append(conclude_id)

        if not branch_conclusions:
            return

        # Step 4: 交叉验证
        validate_id = self._step_cross_validate(graph, branch_conclusions,
                                                goal, context)
        if not validate_id:
            return

        # Step 5: 最终结论
        self._step_conclude(graph, validate_id, goal)

        graph.log("完成", f"总节点: {graph.node_count}, 验证完成")

    # ------------------------------------------------------------------
    # 模式 3: 迭代精炼 (Iterative Refine)
    #
    #   [Root] → [Analyze] → [Hypothesis] → [Verify] ─┐
    #                                               │ (feedback)
    #                                               ↓
    #                              [Refined Analysis] → ...
    # ------------------------------------------------------------------

    def _build_iterative_refine(self, graph: ThoughtGraph, goal: str,
                                 context: str, n_branches: int,
                                 max_depth: int):
        """迭代精炼模式：逐步细化+反馈修正。"""
        graph.log("开始迭代精炼模式", f"最大深度={max_depth}")

        # Step 1: 根节点
        root = GraphNode(content=goal, node_type=NodeType.ROOT.value,
                         depth=0, status=NodeStatus.DONE.value)
        root_id = graph.add_node(root)

        current_id = root_id
        current_content = goal

        for iteration in range(min(max_depth, 4)):
            depth = iteration * 2 + 1

            # 分析阶段
            analyze_id = self._llm_analyze_node(graph, current_id, goal,
                                                 current_content, context,
                                                 depth, iteration)
            if not analyze_id:
                break

            # 假设/提议阶段
            hypothesize_id = self._llm_hypothesize_node(graph, analyze_id, goal,
                                                        context, depth + 1)
            if not hypothesize_id:
                break

            # 验证阶段
            verify_id = self._llm_verify_node(graph, hypothesize_id, goal,
                                              context, depth + 2)
            if not verify_id:
                break

            vnode = graph.get_node(verify_id)
            if vnode and vnode.confidence >= 0.85:
                # 置信度足够高，可以直接得出结论
                graph.log(f"第{iteration+1}轮置信度足够({vnode.confidence:.0%}), 收敛")
                self._step_conclude(graph, verify_id, goal)
                return

            # 反馈修正（如果还有迭代次数）
            if iteration < min(max_depth, 4) - 1:
                feedback_node = GraphNode(
                    content=f"第{iteration+1}轮反馈: 置信度{vnode.confidence:.0%}, 需要进一步细化",
                    node_type=NodeType.FEEDBACK.value,
                    depth=depth + 3,
                    status=NodeStatus.DONE.value,
                )
                feedback_id = graph.add_node(feedback_node)
                graph.add_edge(verify_id, feedback_id, relation="feedback")
                current_id = feedback_id
                current_content = feedback_node.content
                graph.log(f"第{iteration+1}轮完成", f"置信度={vnode.confidence:.0%}, 继续迭代")

        # 达到最大迭代次数，用最后一个节点做结论
        leaves = graph.get_leaves()
        if leaves:
            best = max(leaves, key=lambda n: n.confidence)
            self._step_conclude(graph, best.node_id, goal)

        graph.log("完成", f"迭代结束, 总节点: {graph.node_count}")

    # ------------------------------------------------------------------
    # 模式 4: 完全图 (Full Graph)
    # LLM 自主决定图的结构
    # ------------------------------------------------------------------

    def _build_full_graph(self, graph: ThoughtGraph, goal: str,
                           context: str, n_branches: int,
                           max_depth: int):
        """完全图模式：LLM 自主规划整个图结构。"""
        graph.log("开始完全图模式", "LLM 自主规划图结构")

        # 先用发散-收敛作为基础框架，但允许 LLM 在过程中添加额外的连接
        self._build_diverge_converge(graph, goal, context, n_branches, max_depth)

        # 额外：尝试在已有节点间添加交叉引用边
        self._add_cross_references(graph)

        graph.log("完成", f"完全图构建完毕, 节点: {graph.node_count}, 边: {graph.edge_count}")

    # ------------------------------------------------------------------
    # 公共步骤实现
    # ------------------------------------------------------------------

    def _step_decompose(self, graph: ThoughtGraph, parent_id: str,
                        goal: str, context: str,
                        n_branches: int) -> Optional[str]:
        """步骤：分解问题为多个探索方向。"""
        prompt = GOT_DECOMPOSE_PROMPT.format(
            goal=goal,
            context=context or "（无额外上下文）",
            n_branches=n_branches,
        )
        resp_text = self._call_llm(prompt)
        if not resp_text:
            return None

        data = self._extract_json(resp_text)
        branches = data.get("branches", []) if isinstance(data, dict) else []

        # 创建分解节点
        decompose_node = GraphNode(
            content=f"将问题分解为 {len(branches)} 个探索方向",
            node_type=NodeType.DECOMPOSE.value,
            depth=1,
            status=NodeStatus.DONE.value,
            metadata={"branches": [b.get("direction", "") for b in branches]},
        )
        decompose_id = graph.add_node(decompose_node)
        graph.add_edge(parent_id, decompose_id, relation="decomposes_to")
        graph.log("问题分解", f"得到 {len(branches)} 个方向: "
                  f"{', '.join(b.get('direction','?')[:15] for b in branches)}")

        # 将分支信息存储到分解节点的 metadata 中，供后续使用
        decompose_node.metadata["_branches_full"] = branches
        return decompose_id

    def _step_explore_branches(self, graph: ThoughtGraph, decompose_id: str,
                                goal: str, context: str,
                                n_branches: int) -> list[str]:
        """步骤：并行探索各个分支。"""
        decomp_node = graph.get_node(decompose_id)
        if not decomp_node:
            return []

        branches = decomp_node.metadata.get("_branches_full", [])
        branch_ids = []

        for i, branch in enumerate(branches[:n_branches]):
            direction = branch.get("direction", f"方向{i+1}")
            description = branch.get("description", "")
            priority = branch.get("priority", 0.8)

            # 收集其他分支的简要信息作为 peer_context
            peer_contexts = []
            for j, other in enumerate(branches[:n_branches]):
                if j != i:
                    peer_contexts.append(
                        f"- {other.get('direction', '?')}: {other.get('description', '')[:80]}"
                    )
            peer_text = "\n".join(peer_contexts) if peer_contexts else "（无其他视角）"

            # 调用 LLM 进行该方向的探索
            prompt = GOT_EXPLORE_PROMPT.format(
                goal=goal,
                direction=direction,
                description=description,
                peer_context=peer_text,
            )
            resp_text = self._call_llm(prompt)

            analysis = ""
            key_points = []
            confidence = 0.7
            verdict = ""

            if resp_text:
                data = self._extract_json(resp_text)
                if isinstance(data, dict):
                    analysis = data.get("analysis", resp_text[:300])
                    key_points = data.get("key_points", [])
                    confidence = data.get("confidence", 0.7)
                    verdict = data.get("verdict", "")
                else:
                    analysis = resp_text[:300]

            explore_node = GraphNode(
                content=f"[{direction}] {verdict or analysis[:80]}",
                node_type=NodeType.EXPLORE.value,
                depth=2,
                score=priority,
                confidence=confidence,
                status=NodeStatus.DONE.value,
                metadata={
                    "direction": direction,
                    "analysis": analysis,
                    "key_points": key_points,
                    "verdict": verdict,
                },
            )
            explore_id = graph.add_node(explore_node)
            graph.add_edge(decompose_id, explore_id, relation="explores")
            branch_ids.append(explore_id)

        graph.log("分支探索完成", f"共 {len(branch_ids)} 个分支")
        return branch_ids

    def _step_merge(self, graph: ThoughtGraph, input_ids: list[str],
                    goal: str, context: str) -> Optional[str]:
        """步骤：合并多个输入节点。"""
        # 构建输入摘要
        inputs_summary = []
        for iid in input_ids:
            node = graph.get_node(iid)
            if node:
                meta = node.metadata or {}
                direction = meta.get("direction", "未知方向")
                verdict = meta.get("verdict", node.content)
                points = meta.get("key_points", [])
                summary = (
                    f"### {direction}\n"
                    f"结论: {verdict}\n"
                    f"要点:\n" +
                    "\n".join(f"  - {p}" for p in points[:3])
                )
                inputs_summary.append(summary)

        prompt = GOT_MERGE_PROMPT.format(
            goal=goal,
            inputs_summary="\n\n".join(inputs_summary),
        )
        resp_text = self._call_llm(prompt)
        if not resp_text:
            return None

        data = self._extract_json(resp_text)
        conclusion = ""
        findings = []
        confidence = 0.7
        conflicts = []

        if isinstance(data, dict):
            conclusion = data.get("conclusion", "")
            findings = data.get("key_findings", [])
            confidence = data.get("confidence", 0.7)
            conflicts = data.get("conflicts", [])

        merge_node = GraphNode(
            content=f"合并 {len(input_ids)} 个视角的综合分析",
            node_type=NodeType.MERGE.value,
            depth=3,
            confidence=confidence,
            status=NodeStatus.DONE.value,
            metadata={
                "conclusion": conclusion,
                "key_findings": findings,
                "conflicts": conflicts,
                "resolution": data.get("resolution", "") if isinstance(data, dict) else "",
            },
        )
        merge_id = graph.add_node(merge_node)

        # 添加从各分支到合并节点的边（多入度！GoT 核心特性）
        for iid in input_ids:
            graph.add_edge(iid, merge_id, relation="merges_into")

        graph.log("合并完成", f"{len(input_ids)} 个输入 → 1 个合并节点, "
                  f"冲突数: {len(conflicts)}")
        return merge_id

    def _step_cross_validate(self, graph: ThoughtGraph, input_ids: list[str],
                              goal: str, context: str) -> Optional[str]:
        """步骤：交叉验证多条路径。"""
        paths = []
        for iid in input_ids:
            node = graph.get_node(iid)
            if node:
                meta = node.metadata or {}
                paths.append(meta.get("verdict", node.content))

        if len(paths) < 2:
            # 只有一条路径，不需要交叉验证
            return input_ids[0] if input_ids else None

        prompt = GOT_CROSS_VALIDATE_PROMPT.format(
            goal=goal,
            path_a=paths[0] if len(paths) > 0 else "",
            path_b="\n\n=== 另一条路径 ===\n".join(paths[1:]) if len(paths) > 1 else "",
        )
        resp_text = self._call_llm(prompt)
        if not resp_text:
            return input_ids[0]

        data = self._extract_json(resp_text)
        validation_conclusion = ""
        agreement = 0.5

        if isinstance(data, dict):
            validation_conclusion = data.get("validation_conclusion", "")
            agreement = data.get("agreement_level", 0.5)

        validate_node = GraphNode(
            content=f"交叉验证: 一致性={agreement:.0%}",
            node_type=NodeType.CROSS_VALIDATE.value,
            depth=4,
            confidence=agreement,
            score=agreement,
            status=NodeStatus.DONE.value,
            metadata={
                "is_consistent": data.get("is_consistent", True) if isinstance(data, dict) else True,
                "agreement_level": agreement,
                "discrepancies": data.get("discrepancies", []) if isinstance(data, dict) else [],
                "conclusion": validation_conclusion,
            },
        )
        validate_id = graph.add_node(validate_node)

        for iid in input_ids:
            graph.add_edge(iid, validate_id, relation="validated_by")

        graph.log("交叉验证", f"一致性={agreement:.0%}")
        return validate_id

    def _step_conclude(self, graph: ThoughtGraph, source_id: str, goal: str):
        """步骤：得出最终结论。"""
        source = graph.get_node(source_id)
        if not source:
            return

        # 从源节点提取结论
        meta = source.metadata or {}
        conclusion = (
            meta.get("conclusion") or
            meta.get("validation_conclusion") or
            meta.get("verdict") or
            source.content
        )
        findings = meta.get("key_findings", [])

        conclude_node = GraphNode(
            content=conclusion if conclusion else "综合分析完成",
            node_type=NodeType.CONCLUDE.value,
            depth=source.depth + 1,
            confidence=source.confidence,
            score=source.score,
            status=NodeStatus.DONE.value,
        )
        conclude_id = graph.add_node(conclude_node)
        graph.add_edge(source_id, conclude_id, relation="concludes")

        graph.conclusion = conclusion
        graph.key_findings = findings
        graph.log("得出结论", conclusion[:100] if conclusion else "(空)")

    # ------------------------------------------------------------------
    # 迭代模式的子步骤
    # ------------------------------------------------------------------

    def _llm_analyze_node(self, graph: ThoughtGraph, parent_id: str,
                           goal: str, current_content: str,
                           context: str, depth: int,
                           iteration: int) -> Optional[str]:
        """LLM 分析步骤（迭代模式用）。"""
        prompt = f"""请深入分析以下问题的当前状态。

## 目标
{goal}

## 当前状态
{current_content}

## 上下文
{context or '（无）'}

## 要求
这是第 {iteration+1} 轮分析。请给出：
1. 当前理解的核心要点
2. 还有哪些不确定的地方
3. 下一步应该重点关注什么

输出 JSON: {{"analysis": "...", "uncertainties": [...], "next_focus": "..."}}"""
        resp_text = self._call_llm(prompt)
        if not resp_text:
            return None

        data = self._extract_json(resp_text)
        analysis = data.get("analysis", resp_text[:200]) if isinstance(data, dict) else resp_text[:200]

        node = GraphNode(
            content=f"[分析] {analysis[:80]}",
            node_type=NodeType.ANALYZE.value,
            depth=depth,
            status=NodeStatus.DONE.value,
            metadata={"analysis": analysis},
        )
        node_id = graph.add_node(node)
        graph.add_edge(parent_id, node_id, relation="analyzes")
        return node_id

    def _llm_hypothesize_node(self, graph: ThoughtGraph, analyze_id: str,
                               goal: str, context: str,
                               depth: int) -> Optional[str]:
        """LLM 假设步骤（迭代模式用）。"""
        anode = graph.get_node(analyze_id)
        analysis = anode.metadata.get("analysis", "") if anode else ""

        prompt = f"""基于以下分析，提出假设或解决方案。

## 目标
{goal}

## 分析结果
{analysis}

## 要求
提出 1-2 个可能的解决方案或假设，并评估其可行性。

输出 JSON: {{"hypotheses": [{{"name": "...", "description": "...", "feasibility": 0.8}}], "recommended": "..."}}"""
        resp_text = self._call_llm(prompt)
        if not resp_text:
            return None

        data = self._extract_json(resp_text)
        hyps = data.get("hypotheses", []) if isinstance(data, dict) else []
        recommended = data.get("recommended", "") if isinstance(data, dict) else ""

        content = recommended or (hyps[0].get("name", "") if hyps else "假设已生成")
        node = GraphNode(
            content=f"[假设] {content[:80]}",
            node_type=NodeType.HYPOTHESIZE.value,
            depth=depth,
            status=NodeStatus.DONE.value,
            metadata={"hypotheses": hyps, "recommended": recommended},
        )
        node_id = graph.add_node(node)
        graph.add_edge(analyze_id, node_id, relation="hypothesizes")
        return node_id

    def _llm_verify_node(self, graph: ThoughtGraph, hypothesize_id: str,
                          goal: str, context: str,
                          depth: int) -> Optional[str]:
        """LLM 验证步骤（迭代模式用）。"""
        hnode = graph.get_node(hypothesize_id)
        hyps = hnode.metadata.get("hypotheses", []) if hnode else []
        rec = hnode.metadata.get("recommended", "") if hnode else ""

        prompt = f"""请严格验证以下假设/方案。

## 目标
{goal}

## 待验证的假设
推荐方案: {rec}
备选方案: {[h.get('name','') for h in hyps]}

## 要求
从以下角度验证：
1. 逻辑自洽性
2. 可行性（技术/资源/时间）
3. 潜在风险
4. 给出置信度评分

输出 JSON: {{"verification": "...", "confidence": 0.8, "risks": [...], "verdict": "通过/需修改/不推荐"}}"""
        resp_text = self._call_llm(prompt)
        if not resp_text:
            return None

        data = self._extract_json(resp_text)
        confidence = data.get("confidence", 0.7) if isinstance(data, dict) else 0.7
        verdict = data.get("verdict", "") if isinstance(data, dict) else ""

        node = GraphNode(
            content=f"[验证] {verdict or '验证完成'} (置信度={confidence:.0%})",
            node_type=NodeType.VERIFY.value,
            depth=depth,
            confidence=confidence,
            score=confidence,
            status=NodeStatus.DONE.value,
            metadata={"verdict": verdict, "risks": data.get("risks", []) if isinstance(data, dict) else []},
        )
        node_id = graph.add_node(node)
        graph.add_edge(hypothesize_id, node_id, relation="verifies")
        return node_id

    # ------------------------------------------------------------------
    # 交叉引用（Full Graph 模式的增强）
    # ------------------------------------------------------------------

    def _add_cross_references(self, graph: ThoughtGraph):
        """在已有的探索节点之间添加交叉引用边。"""
        explore_nodes = [
            n for n in graph._nodes.values()
            if n.node_type == NodeType.EXPLORE.value and n.depth == 2
        ]

        # 为每对探索节点检查是否应该添加交叉引用
        for i, n1 in enumerate(explore_nodes):
            for n2 in explore_nodes[i+1:]:
                # 如果两个节点的内容差异较大，添加 cross_reference 边
                content1 = n1.content.lower()
                content2 = n2.content.lower()
                similarity = len(set(content1.split()) & set(content2.split())) / \
                             max(len(set(content1.split())), len(set(content2.split())), 1)

                # 低相似度 = 不同角度，值得交叉参考
                if 0.1 < similarity < 0.6:
                    graph.add_edge(n1.node_id, n2.node_id,
                                   relation="cross_reference", weight=similarity)

    # ------------------------------------------------------------------
    # LLM 调用辅助
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 3000) -> str:
        """调用 LLM 并返回响应文本。"""
        try:
            from ..llm.llm import call_llm, parse_llm_response
            resp = call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            resp = parse_llm_response(resp)
            if resp and resp.content:
                return "".join(getattr(b, "text", "") or "" for b in resp.content)
        except Exception as e:
            pass
        return ""

    def _extract_json(self, text: str):
        """从文本中提取 JSON 数据。"""
        import re
        text = text.strip()
        parsers = [
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group())
                if re.search(r'\{.*\}', t, re.DOTALL) else None,
            lambda t: json.loads(re.search(r'\[.*\]', t, re.DOTALL).group())
                if re.search(r'\[.*\]', t, re.DOTALL) else None,
            lambda t: json.loads(re.search(r'```(?:json)?\s*\n(.*?)\n```',
                                           t, re.DOTALL).group(1))
                if re.search(r'```(?:json)?\s*\n(.*?)\n```', t, re.DOTALL) else None,
        ]
        for parser in parsers:
            try:
                result = parser(text)
                if result is not None:
                    return result
            except (json.JSONDecodeError, ValueError, AttributeError):
                continue
        return None

    # ------------------------------------------------------------------
    # 历史查询
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 20) -> list[ThoughtGraph]:
        return list(reversed(self._history[-limit:]))

    def get_last_graph(self) -> Optional[ThoughtGraph]:
        return self._history[-1] if self._history else None

    def clear_history(self):
        self._history.clear()


# =====================================================================
# 模块级单例
# =====================================================================

GOT_ENGINE = ThoughtGraphEngine()

