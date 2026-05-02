#!/usr/bin/env python3
"""思维树（Tree of Thought, ToT）引擎。

将思维链的线性推理扩展为树状搜索结构。

核心思想（源自 Yao et al. 2023 "Tree of Thoughts"）：
  1. 将问题分解为多个「思考步骤」（ThoughtStep），每个步骤可以有多个候选
  2. 形成「思维树」：根节点是问题，每个节点是一个思考状态
  3. 搜索策略：
     - BFS: 广度优先，每层展开所有候选，评估后选最优路径
     - DFS: 深度优先，沿一条路径深入探索，回溯尝试其他分支
     - BEST_FIRST: 最佳优先，用启发式评分优先探索最有希望的分支

与 CoT 的区别：
  - CoT 是线性的：Step1 → Step2 → Step3 → 结论
  - ToT 是树状的：问题 → [候选A, 候选B, 候选C]
                         候选A → [子候选A1, 子候选A2]
                         候选B → [子候选B1]
                       → 评估所有叶节点 → 选择最优路径 → 结论

适用场景：
  - 需要从多个方案中选择的复杂决策
  - 有多种可能解决路径的问题
  - 需要探索-验证-回溯的规划任务
  - 创意性/开放性问题（如架构设计、算法选择）

使用方式：
    from agents.llm.thought_tree import ThoughtTreeEngine

    engine = ThoughtTreeEngine()
    tree = engine.explore("如何设计一个高并发的消息队列？", strategy="bfs")
    # 或
    tree = engine.explore("实现一个 LRU 缓存", strategy="best_first", max_depth=4, width=3)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# =====================================================================
# 枚举与数据结构
# =====================================================================

class SearchStrategy(str, Enum):
    """搜索策略。"""
    BFS = "bfs"                     # 广度优先：逐层展开，评估后选最优
    DFS = "dfs"                     # 深度优先：深入一条路径，失败则回溯
    BEST_FIRST = "best_first"       # 最佳优先：启发式评分，优先探索高潜力分支


class NodeStatus(str, Enum):
    """思维树节点的状态。"""
    PENDING = "pending"             # 待展开
    EXPANDED = "expanded"           # 已展开（有子节点）
    EVALUATED = "evaluated"         # 已评估
    SELECTED = "selected"           # 被选中为最优路径
    PRUNED = "pruned"               # 被剪枝（低分淘汰）
    ROOT = "root"                   # 根节点


@dataclass
class ThoughtTreeNode:
    """思维树的节点。

    每个节点代表一个「思考状态」—— 即在某个决策点的一个具体思路。
    """
    node_id: str = ""
    content: str = ""               # 该节点的思考内容
    parent_id: Optional[str] = None  # 父节点 ID（根节点为 None)
    depth: int = 0                  # 树深度（根节点 depth=0）
    score: float = 0.0              # 评估分数 (0.0 ~ 1.0)
    status: str = NodeStatus.PENDING.value
    thought_type: str = "step"      # step / evaluation / conclusion
    tokens_used: int = 0
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""
    children_ids: list = field(default_factory=list)  # 子节点 ID 列表

    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"N-{uuid.uuid4().hex[:6]}"
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_leaf"] = self.is_leaf
        return d


@dataclass
class ThoughtTree:
    """一棵完整的思维树。

    结构：
              Root (问题)
              / | \\
            A   B   C          ← 第 1 层（候选方案）
           / \\     |
          A1  A2   B1          ← 第 2 层（子方案/细化）
          |
         A11                    ← 第 3 层（更深层）

    搜索完成后会标记出最优路径（SELECTED 状态）。
    """
    tree_id: str = ""
    goal: str = ""
    strategy: str = SearchStrategy.BFS.value
    max_depth: int = 4               # 最大搜索深度
    width: int = 3                   # 每个节点最多展开的子节点数（beam width）
    status: str = "pending"          # pending / searching / completed / error / aborted
    root_id: str = ""
    total_nodes: int = 0
    total_tokens: int = 0
    total_duration_ms: int = 0
    best_path: list = field(default_factory=list)  # 最优路径的 node_id 列表
    best_score: float = 0.0
    conclusion: str = ""
    created_at: str = ""
    completed_at: str = ""
    search_log: list = field(default_factory=list)  # 搜索过程日志
    metadata: dict = field(default_factory=dict)

    # 内部存储：node_id → ThoughtTreeNode
    _nodes: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.tree_id:
            self.tree_id = f"ToT-{uuid.uuid4().hex[:10]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    @property
    def nodes(self) -> dict[str, ThoughtTreeNode]:
        return self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def depth(self) -> int:
        """当前树的实际最大深度。"""
        if not self._nodes:
            return 0
        return max(n.depth for n in self._nodes.values()) if self._nodes else 0

    @property
    def root(self) -> Optional[ThoughtTreeNode]:
        return self._nodes.get(self.root_id) if self.root_id else None

    def add_node(self, node: ThoughtTreeNode) -> str:
        """添加节点到树中。"""
        self._nodes[node.node_id] = node
        self.total_nodes += 1
        self.total_tokens += node.tokens_used
        self.total_duration_ms += node.duration_ms
        # 如果父节点存在，更新父节点的 children_ids
        if node.parent_id and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node.node_id not in parent.children_ids:
                parent.children_ids.append(node.node_id)
        return node.node_id

    def get_node(self, node_id: str) -> Optional[ThoughtTreeNode]:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> list[ThoughtTreeNode]:
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids
                if cid in self._nodes]

    def get_leaves(self) -> list[ThoughtTreeNode]:
        """获取所有叶子节点。"""
        return [n for n in self._nodes.values() if n.is_leaf]

    def get_path_to_root(self, node_id: str) -> list[ThoughtTreeNode]:
        """从指定节点走到根节点的路径（含该节点和根节点）。"""
        path = []
        current_id = node_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = self._nodes.get(current_id)
            if not node:
                break
            path.append(node)
            current_id = node.parent_id
        return list(reversed(path))

    def mark_best_path(self, leaf_node_id: str):
        """标记从根到某叶节点的路径为最优路径。"""
        path = self.get_path_to_root(leaf_node_id)
        self.best_path = [n.node_id for n in path]
        for node in path:
            node.status = NodeStatus.SELECTED.value
        if path:
            self.best_score = path[-1].score
            # 取最后一个非 evaluation 节点的内容作为结论
            for n in reversed(path):
                if n.thought_type != "evaluation":
                    self.conclusion = n.content
                    break

    def log(self, message: str, detail: str = ""):
        """记录搜索日志。"""
        self.search_log.append({
            "time": time.strftime("%H:%M:%S", time.localtime()),
            "message": message,
            "detail": detail,
        })

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "treeId": self.tree_id,
            "goal": self.goal,
            "strategy": self.strategy,
            "status": self.status,
            "maxDepth": self.max_depth,
            "width": self.width,
            "totalNodes": self.total_nodes,
            "actualDepth": self.depth,
            "totalTokens": self.total_tokens,
            "totalDurationMs": self.total_duration_ms,
            "bestScore": self.best_score,
            "conclusion": self.conclusion,
            "bestPath": self.best_path,
            "rootId": self.root_id,
            "createdAt": self.created_at,
            "completedAt": self.completed_at,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
        }
        return d

    def to_session_entry(self) -> dict:
        """转换为会话记录格式（用于存入 JSONL）。"""
        return {
            "type": "thought_tree",
            "treeId": self.tree_id,
            "goal": self.goal,
            "strategy": self.strategy,
            "status": self.status,
            "totalNodes": self.total_nodes,
            "actualDepth": self.depth,
            "totalTokens": self.total_tokens,
            "totalDurationMs": self.total_duration_ms,
            "bestScore": self.best_score,
            "conclusion": self.conclusion,
            "bestPath": self.best_path,
            "rootId": self.root_id,
            "createdAt": self.created_at,
            "completedAt": self.completed_at,
            # 只保存精简版节点信息（避免 JSONL 过大）
            "nodeSummary": [
                {"id": n.node_id, "parent": n.parent_id, "depth": n.depth,
                 "content": n.content[:120], "score": n.score,
                 "status": n.status, "type": n.thought_type}
                for n in self._nodes.values()
            ],
        }

    def to_compact_summary(self) -> str:
        """生成压缩版摘要（用于上下文注入）。"""
        parts = [f"[思维树 ToT] 目标: {self.goal[:80]}",
                 f"策略: {self.strategy} | 节点数: {self.total_nodes} | "
                 f"深度: {self.depth} | 最优分数: {self.best_score:.2f}"]
        # 最优路径摘要
        if self.best_path:
            path_summary = " → ".join([
                self._nodes[nid].content[:40] if nid in self._nodes else nid
                for nid in self.best_path[:6]
            ])
            parts.append(f"最优路径: {path_summary}")
        if self.conclusion:
            parts.append(f"结论: {self.conclusion[:200]}")
        return "\n".join(parts)

    def display(self) -> str:
        """生成终端展示（ASCII 树形图）。"""
        lines = [
            "",
            "=" * 64,
            f"  思维树 (Tree of Thought): {self.tree_id}",
            f"  目标:   {self.goal}",
            f"  策略:   {self.strategy} | 宽度(beam): {self.width} | "
            f"最大深度: {self.max_depth}",
            f"  状态:   {self.status} | 节点数: {self.total_nodes} | "
            f"实际深度: {self.depth}",
            f"  Token:  {self.total_tokens} | 耗时: {self.total_duration_ms}ms",
            f"  最优分数: {self.best_score:.2f}",
            "=" * 64,
        ]

        if not self._nodes or not self.root_id:
            lines.append("  (空树)")
            return "\n".join(lines)

        # 递归渲染树
        def render_node(node_id: str, prefix: str = "", is_last: bool = True) -> list:
            result = []
            node = self._nodes.get(node_id)
            if not node:
                return result

            # 状态图标
            status_icon = {
                "selected": "*",
                "evaluated": "✓",
                "expanded": "+",
                "pruned": "✗",
                "pending": "○",
                "root": "◆",
            }.get(node.status, "?")

            score_str = f"({node.score:.2f})" if node.score > 0 else ""
            type_tag = f"[{node.thought_type}]" if node.thought_type != "step" else ""

            connector = "`- " if is_last else "|- "
            line = f"{prefix}{connector}{status_icon} {node.content[:80]} {type_tag} {score_str}".rstrip()
            result.append(line)

            children = self.get_children(node_id)
            for i, child in enumerate(children):
                child_is_last = (i == len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "|   ")
                result.extend(render_node(child.node_id, child_prefix, child_is_last))

            return result

        lines.extend(render_node(self.root_id))

        # 最优路径
        if self.best_path:
            lines.extend(["", "  最优路径:"])
            for i, nid in enumerate(self.best_path):
                node = self._nodes.get(nid)
                if node:
                    marker = "-->" if i < len(self.best_path) - 1 else "==="
                    lines.append(f"    {marker} [{node.thought_type}] {node.content[:60]} "
                                f"(score={node.score:.2f})")

        if self.conclusion:
            lines.extend(["", f"  结论: {self.conclusion}"])

        lines.append("=" * 64)
        return "\n".join(lines)


# =====================================================================
# ID 生成（复用 thought_chain 的或独立）
# =====================================================================

_existing_tot_ids: set = set()


def _tot_short_id(length: int = 6) -> str:
    for _ in range(100):
        cid = uuid.uuid4().hex[:length]
        if cid not in _existing_tot_ids:
            _existing_tot_ids.add(cid)
            return cid
    return uuid.uuid4().hex[:length]


# =====================================================================
# LLM Prompt 模板
# =====================================================================

TOT_GENERATE_PROMPT = """\
你是一个多角度思维助手。请为一个问题的某个思考阶段生成多个候选思路。

## 原始目标
{goal}

## 当前思考状态
{current_state}
{parent_context}

## 要求
请生成 {n_candidates} 个不同的候选思路/方案。每个候选应该：
1. 与其他候选有明显区别（不同方向、不同方法、不同侧重点）
2. 具体且可执行（不是空洞的描述）
3. 适合在这个阶段推进问题解决

输出 JSON 数组：
```json
{{
  "candidates": [
    {{
      "content": "候选思路 1 的具体描述",
      "reasoning": "为什么这个思路值得探索",
      "promise": 0.8
    }},
    ...
  ]
}}
```

promise 字段表示你对这个候选的初步期望值（0.0~1.0）。
只输出 JSON，不要有其他文字。
"""

TOT_EVALUATE_PROMPT = """\
你是一个严谨的思维评估者。请对以下候选思路进行评估。

## 原始目标
{goal}

## 当前待评估的候选思路
{candidates_text}

## 已有上下文
{context}

## 评估标准
请从以下维度给每个候选打分（0.0~1.0）：

1. **可行性** (feasibility): 技术上是否可行？是否有明显障碍？
2. **有效性** (effectiveness): 这个思路能在多大程度上推进目标？
3. **效率** (efficiency): 实现成本 vs 收益比如何？
4. **风险** (risk): 是否有潜在风险或副作用？（高分=低风险）

输出 JSON 数组：
```json
{{
  "evaluations": [
    {{
      "candidate_id": "候选编号或简述",
      "scores": {{
        "feasibility": 0.9,
        "effectiveness": 0.8,
        "efficiency": 0.7,
        "risk": 0.85
      }},
      "overall_score": 0.81,
      "reasoning": "评分理由"
    }},
    ...
  ],
  "best_candidate_id": "得分最高的候选ID",
  "should_continue": true
}}
```

should_continue 表示是否应该继续深入探索最佳候选（true）还是已经足够好可以停止（false）。
只输出 JSON，不要有其他文字。
"""


# =====================================================================
# 核心引擎
# =====================================================================

class ThoughtTreeEngine:
    """思维树引擎。

    管理 ToT 搜索的完整生命周期：
      1. 创建树（根节点 = 问题本身）
      2. 展开（调用 LLM 生成候选子节点）
      3. 评估（调用 LLM 对候选打分）
      4. 选择（根据策略选择要继续探索的节点）
      5. 重复 2-4 直到达到终止条件
      6. 回溯最优路径

    使用方式：
        engine = ThoughtTreeEngine()

        # BFS 搜索（适合方案对比）
        tree = engine.explore("如何设计缓存系统?", strategy="bfs")

        # 最佳优先搜索（适合复杂规划）
        tree = engine.explore("重构遗留代码", strategy="best_first",
                              max_depth=5, width=3)
    """

    def __init__(self, default_strategy: SearchStrategy = SearchStrategy.BFS,
                 max_depth: int = 4,
                 width: int = 3,
                 min_score_threshold: float = 0.3):
        """
        Args:
            default_strategy: 默认搜索策略
            max_depth: 默认最大搜索深度
            width: 每个节点最多展开的候选数（beam width）
            min_score_threshold: 最低分数阈值（低于此值的候选会被剪枝）
        """
        self.default_strategy = default_strategy
        self.max_depth = max_depth
        self.width = width
        self.min_score_threshold = min_score_threshold
        self._history: list[ThoughtTree] = []

    # ------------------------------------------------------------------
    # 核心搜索方法
    # ------------------------------------------------------------------

    def explore(self, goal: str, context: str = "",
                strategy: str = None,
                max_depth: int = None,
                width: int = None,
                n_candidates: int = None) -> ThoughtTree:
        """执行一次完整的思维树搜索。

        Args:
            goal: 要探索的问题或目标
            context: 额外上下文信息
            strategy: 搜索策略 (bfs/dfs/best_first)
            max_depth: 最大搜索深度
            width: 每个节点的 beam width
            n_candidates: 每次 LLM 生成的候选数量

        Returns:
            完整的 ThoughtTree 对象
        """

        # 参数规范化
        strat = SearchStrategy(strategy or self.default_strategy.value)
        depth_limit = max_depth or self.max_depth
        beam_width = width or self.width
        n_cand = n_candidates or max(beam_width + 1, 3)

        # 创建树
        tree = ThoughtTree(
            goal=goal,
            strategy=strat.value,
            max_depth=depth_limit,
            width=beam_width,
            status="searching",
        )

        start_time = time.time() * 1000

        try:
            # 创建根节点
            root = ThoughtTreeNode(
                content=goal,
                depth=0,
                status=NodeStatus.ROOT.value,
                thought_type="root",
            )
            tree.root_id = tree.add_node(root)
            tree.log("创建根节点", goal)

            # 根据策略执行搜索
            if strat == SearchStrategy.BFS:
                self._search_bfs(tree, context, depth_limit, beam_width, n_cand)
            elif strat == SearchStrategy.DFS:
                self._search_dfs(tree, context, depth_limit, beam_width, n_cand)
            elif strat == SearchStrategy.BEST_FIRST:
                self._search_best_first(tree, context, depth_limit, beam_width, n_cand)

            tree.status = "completed"
            tree.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        except Exception as e:
            tree.status = "error"
            tree.metadata["error"] = str(e)
            tree.log("搜索出错", str(e))
        finally:
            tree.total_duration_ms = int(time.time() * 1000 - start_time)

        self._history.append(tree)
        return tree

    # ------------------------------------------------------------------
    # BFS 搜索：广度优先
    # ------------------------------------------------------------------

    def _search_bfs(self, tree: ThoughtTree, context: str,
                    depth_limit: int, beam_width: int, n_cand: int):
        """BFS：逐层展开，每层评估后保留 top-k 分支。"""
        tree.log("开始 BFS 搜索", f"max_depth={depth_limit}, width={beam_width}")

        current_frontier = [tree.root_id]  # 当前沿（待展开的节点 ID）

        for depth in range(1, depth_limit + 1):
            if not current_frontier:
                break

            tree.log(f"BFS 第 {depth} 层", f"前沿节点数: {len(current_frontier)}")
            next_frontier = []

            for parent_id in current_frontier:
                parent = tree.get_node(parent_id)
                if not parent:
                    continue

                # 展开子节点
                candidates = self._generate_candidates(
                    tree, parent, context, n_cand)
                if not candidates:
                    continue

                # 评估候选
                evaluated = self._evaluate_candidates(
                    tree, candidates, context)

                # 按 overall_score 排序，取 top beam_width
                evaluated.sort(key=lambda c: c.get("overall_score", 0), reverse=True)
                selected = evaluated[:beam_width]

                # 添加到树中
                for ev in selected:
                    cand_content = ev.get("candidate_content", "")
                    score = ev.get("overall_score", 0.5)
                    child = ThoughtTreeNode(
                        content=cand_content,
                        parent_id=parent_id,
                        depth=depth,
                        score=score,
                        status=NodeStatus.EVALUATED.value,
                        thought_type="step",
                        metadata={"reasoning": ev.get("reasoning", "")},
                    )
                    child_id = tree.add_node(child)
                    next_frontier.append(child_id)

                # 未被选中的标记为 PRUNED
                for ev in evaluated[beam_width:]:
                    pruned = ThoughtTreeNode(
                        content=ev.get("candidate_content", ""),
                        parent_id=parent_id,
                        depth=depth,
                        score=ev.get("overall_score", 0),
                        status=NodeStatus.PRUNED.value,
                        thought_type="step",
                    )
                    tree.add_node(pruned)

            current_frontier = next_frontier

            # 如果所有前沿节点分数都很低，提前停止
            if current_frontier:
                scores = [tree.get_node(nid).score for nid in current_frontier
                          if tree.get_node(nid)]
                if scores and max(scores) < self.min_score_threshold:
                    tree.log("提前终止", "前沿节点分数过低")
                    break

        # 选择最优路径（分数最高的叶子节点）
        leaves = tree.get_leaves()
        if leaves:
            best_leaf = max(leaves, key=lambda n: n.score)
            tree.mark_best_path(best_leaf.node_id)
            tree.log("完成", f"最优路径分数: {tree.best_score:.2f}")

    # ------------------------------------------------------------------
    # DFS 搜索：深度优先
    # ------------------------------------------------------------------

    def _search_dfs(self, tree: ThoughtTree, context: str,
                    depth_limit: int, beam_width: int, n_cand: int):
        """DFS：沿一条路径深入，失败时回溯尝试其他分支。"""
        tree.log("开始 DFS 搜索", f"max_depth={depth_limit}, width={beam_width}")

        def dfs_expand(node_id: str, depth: int) -> bool:
            """递归展开节点。返回 True 表示找到了足够好的解。"""
            if depth >= depth_limit:
                return True

            node = tree.get_node(node_id)
            if not node:
                return False

            # 生成候选
            candidates = self._generate_candidates(tree, node, context, n_cand)
            if not candidates:
                node.status = NodeStatus.EXPANDED.value
                return False

            # 评估
            evaluated = self._evaluate_candidates(tree, candidates, context)
            evaluated.sort(key=lambda c: c.get("overall_score", 0), reverse=True)
            selected = evaluated[:beam_width]

            node.status = NodeStatus.EXPANDED.value
            added_children = []

            for ev in selected:
                child = ThoughtTreeNode(
                    content=ev.get("candidate_content", ""),
                    parent_id=node_id,
                    depth=depth + 1,
                    score=ev.get("overall_score", 0.5),
                    status=NodeStatus.EVALUATED.value,
                    thought_type="step",
                )
                child_id = tree.add_node(child)
                added_children.append(child_id)

            # 剪枝
            for ev in evaluated[beam_width:]:
                pruned = ThoughtTreeNode(
                    content=ev.get("candidate_content", ""),
                    parent_id=node_id,
                    depth=depth + 1,
                    score=ev.get("overall_score", 0),
                    status=NodeStatus.PRUNED.value,
                    thought_type="step",
                )
                tree.add_node(pruned)

            # 递归深入第一个（最好的）子节点
            if added_children:
                should_stop = dfs_expand(added_children[0], depth + 1)
                if should_stop and tree.get_node(added_children[0]).score >= 0.7:
                    return True

                # 如果第一条路不够好，尝试第二条
                if len(added_children) > 1:
                    dfs_expand(added_children[1], depth + 1)

            return False

        dfs_expand(tree.root_id, 0)

        # 选择最优叶子
        leaves = [l for l in tree.get_leaves() if l.status != NodeStatus.PRUNED.value]
        if leaves:
            best_leaf = max(leaves, key=lambda n: n.score)
            tree.mark_best_path(best_leaf.node_id)
            tree.log("完成", f"最优路径分数: {tree.best_score:.2f}")

    # ------------------------------------------------------------------
    # Best-First 搜索：最佳优先（类 A*）
    # ------------------------------------------------------------------

    def _search_best_first(self, tree: ThoughtTree, context: str,
                           depth_limit: int, beam_width: int, n_cand: int):
        """Best-First：始终优先展开最有希望的节点（贪心 + 回溯）。"""
        tree.log("开始 Best-First 搜索", f"max_depth={depth_limit}, width={beam_width}")

        # 优先队列：（负分数用于 heapq 的最小堆语义）
        import heapq
        frontier = []  # [(-score, node_id)]
        heapq.heappush(frontier, (0.0, tree.root_id))
        expanded_count = 0
        max_expansions = depth_limit * beam_width * 2  # 安全上限

        while frontier and expanded_count < max_expansions:
            neg_score, node_id = heapq.heappop(frontier)
            node = tree.get_node(node_id)
            if not node:
                continue

            # 深度检查
            if node.depth >= depth_limit:
                node.status = NodeStatus.EVALUATED.value
                continue

            # 如果已经找到高分解，可以停止
            if node.score >= 0.9 and node.depth > 0:
                tree.log("找到高分解", f"score={node.score:.2f}, depth={node.depth}")
                break

            expanded_count += 1

            # 展开
            candidates = self._generate_candidates(tree, node, context, n_cand)
            if not candidates:
                node.status = NodeStatus.EXPANDED.value
                continue

            evaluated = self._evaluate_candidates(tree, candidates, context)
            evaluated.sort(key=lambda c: c.get("overall_score", 0), reverse=True)
            selected = evaluated[:beam_width]

            node.status = NodeStatus.EXPANDED.value

            for ev in selected:
                child = ThoughtTreeNode(
                    content=ev.get("candidate_content", ""),
                    parent_id=node_id,
                    depth=node.depth + 1,
                    score=ev.get("overall_score", 0.5),
                    status=NodeStatus.EVALUATED.value,
                    thought_type="step",
                )
                child_id = tree.add_node(child)
                # 用负分数推入堆（最高分的先出）
                heapq.heappush(frontier, (-child.score, child_id))

            # 剪枝
            for ev in evaluated[beam_width:]:
                pruned = ThoughtTreeNode(
                    content=ev.get("candidate_content", ""),
                    parent_id=node_id,
                    depth=node.depth + 1,
                    score=ev.get("overall_score", 0),
                    status=NodeStatus.PRUNED.value,
                    thought_type="step",
                )
                tree.add_node(pruned)

        # 选择最优叶子
        leaves = [l for l in tree.get_leaves() if l.status != NodeStatus.PRUNED.value]
        if leaves:
            best_leaf = max(leaves, key=lambda n: n.score)
            tree.mark_best_path(best_leaf.node_id)
            tree.log("完成", f"最优路径分数: {tree.best_score:.2f}, "
                     f"展开节点数: {expanded_count}")

    # ------------------------------------------------------------------
    # LLM 辅助方法：生成候选
    # ------------------------------------------------------------------

    def _generate_candidates(self, tree: ThoughtTree, parent: ThoughtTreeNode,
                             context: str, n: int) -> list[dict]:
        """调用 LLM 为给定节点生成候选子节点。"""
        from ..llm.llm import call_llm, parse_llm_response

        # 构建父节点上下文
        parent_path = tree.get_path_to_root(parent.node_id)
        parent_context = "\n".join([
            f"  [Depth {p.depth}] {p.content}"
            for p in parent_path
        ])

        prompt = TOT_GENERATE_PROMPT.format(
            goal=tree.goal,
            current_state=f"当前节点(Depth {parent.depth}): {parent.content}",
            parent_context=f"\n历史路径:\n{parent_context}" if parent_path else "",
            n_candidates=n,
        )

        try:
            resp = call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
            resp = parse_llm_response(resp)

            response_text = ""
            if resp and resp.content:
                response_text = "".join(
                    getattr(b, "text", "") or "" for b in resp.content
                )

            parsed = self._parse_generate_response(response_text)
            tree.log(f"生成候选", f"parent={parent.node_id[:8]}, "
                     f"得到 {len(parsed)} 个候选")
            return parsed

        except Exception as e:
            tree.log("生成候选失败", str(e))
            return []

    # ------------------------------------------------------------------
    # LLM 辅助方法：评估候选
    # ------------------------------------------------------------------

    def _evaluate_candidates(self, tree: ThoughtTree,
                              candidates: list[dict],
                              context: str) -> list[dict]:
        """调用 LLM 对候选进行评估打分。"""
        from ..llm.llm import call_llm, parse_llm_response

        if not candidates:
            return []

        # 格式化候选列表
        cand_texts = []
        for i, c in enumerate(candidates):
            cand_texts.append(
                f"  候选 {i+1}: {c.get('content', '')}\n"
                f"    理由: {c.get('reasoning', '')}\n"
                f"    初步期望: {c.get('promise', 'N/A')}"
            )

        prompt = TOT_EVALUATE_PROMPT.format(
            goal=tree.goal,
            candidates_text="\n\n".join(cand_texts),
            context=context or "（无额外上下文）",
        )

        try:
            resp = call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
            resp = parse_llm_response(resp)

            response_text = ""
            if resp and resp.content:
                response_text = "".join(
                    getattr(b, "text", "") or "" for b in resp.content
            )

            parsed = self._parse_evaluate_response(response_text, candidates)
            tree.log("评估完成", f"评估了 {len(parsed)} 个候选")
            return parsed

        except Exception as e:
            tree.log("评估失败", str(e))
            # 降级：使用 promise 作为默认分数
            return [
                {**c, "overall_score": c.get("promise", 0.5),
                 "candidate_content": c.get("content", "")}
                for c in candidates
            ]

    # ------------------------------------------------------------------
    # 解析方法
    # ------------------------------------------------------------------

    def _parse_generate_response(self, text: str) -> list[dict]:
        """解析 LLM 的候选生成响应。"""
        data = self._extract_json(text)
        if isinstance(data, dict) and "candidates" in data:
            return data["candidates"]
        if isinstance(data, list):
            return data
        # 兜底：返回单个候选
        if isinstance(text, str) and text.strip():
            return [{"content": text.strip(), "reasoning": "", "promise": 0.7}]
        return []

    def _parse_evaluate_response(self, text: str,
                                  original_candidates: list[dict]) -> list[dict]:
        """解析 LLM 的评估响应，将结果映射回原始候选。"""
        data = self._extract_json(text)
        evaluations = []

        if isinstance(data, dict) and "evaluations" in data:
            evals = data["evaluations"]
            for i, ev in enumerate(evals):
                cand = original_candidates[i] if i < len(original_candidates) else {}
                evaluations.append({
                    **ev,
                    "candidate_content": cand.get("content", ""),
                    "overall_score": ev.get("overall_score",
                                            sum(ev.get("scores", {}).values()) / 4),
                })
            return evaluations

        if isinstance(data, list):
            return [
                {**d, "candidate_content": d.get("candidate_content", d.get("content", ""))}
                for d in data
            ]

        # 降级：使用原始 promise
        return [
            {**c, "overall_score": c.get("promise", 0.5),
             "candidate_content": c.get("content", "")}
            for c in original_candidates
        ]

    def _extract_json(self, text: str):
        """从文本中提取 JSON 数据（支持多种格式）。"""
        import re
        text = text.strip()

        # 直接 JSON
        for parser in [
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group())
                if re.search(r'\{.*\}', t, re.DOTALL) else None,
            lambda t: json.loads(re.search(r'\[.*\]', t, re.DOTALL).group())
                if re.search(r'\[.*\]', t, re.DOTALL) else None,
            lambda t: json.loads(re.search(r'```(?:json)?\s*\n(.*?)\n```',
                                           t, re.DOTALL).group(1))
                if re.search(r'```(?:json)?\s*\n(.*?)\n```', t, re.DOTALL) else None,
        ]:
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

    def get_history(self, limit: int = 20) -> list[ThoughtTree]:
        return list(reversed(self._history[-limit:]))

    def get_last_tree(self) -> Optional[ThoughtTree]:
        return self._history[-1] if self._history else None

    def clear_history(self):
        self._history.clear()


# =====================================================================
# 模块级单例
# =====================================================================

TOT_ENGINE = ThoughtTreeEngine()

