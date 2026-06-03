"""
重播式睡眠巩固引擎

模拟睡眠阶段的记忆巩固过程：
  1. 选择经验痕迹（重播采样策略）
  2. 强化成功记忆、削弱失败记忆
  3. 合并相似痕迹（相同符号键 → 保留最强）
  4. 时间衰减遗忘（巩固记忆受 0.6x 保护）
  5. 双通道检索（特征 + 语言符号加权）

所有数值计算使用 torch.Tensor，零 numpy 依赖。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.core.device import get_device


# ── 数据结构 ──────────────────────────────────────────────────

@dataclass
class MemoryTrace:
    """单条记忆痕迹

    Attributes:
        content: 场景特征字典，例如 {'color': [0.8, 0.2, 0.1], 'shape': 'round'}
        linguistic: 使用的语言符号列表，例如 ['red', 'round']
        timestamp: 产生的轮次编号
        strength: 记忆强度 [0, 1]，越高越不容易遗忘
        replay_count: 被重播巩固的次数
        success: 是否为成功经验
    """
    content: Dict
    linguistic: List[str]
    timestamp: int
    strength: float = 1.0
    replay_count: int = 0
    success: bool = False


# ── 巩固引擎 ──────────────────────────────────────────────────

class ConsolidationEngine:
    """重播式睡眠巩固引擎

    核心流程:
      add_trace → (多次积累) → consolidate → forget

    支持 4 种重播采样策略:
      random                — 均匀随机采样
      prioritize_success    — 优先重播成功经验
      prioritize_recent     — 优先重播近期经验
      prioritize_surprising — 优先重播出人意料的结果

    Args:
        replay_k: 每次巩固重播的痕迹数量
        strategy: 默认重播采样策略
        boost: 巩固时对强记忆的增强量
        decay_amount: 巩固时对弱记忆的削弱量
    """

    _STRATEGIES = ('random', 'prioritize_success',
                   'prioritize_recent', 'prioritize_surprising')

    def __init__(self, replay_k: int = 10, strategy: str = 'prioritize_success',
                 boost: float = 0.1, decay_amount: float = 0.05):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"未知策略 '{strategy}'，可选: {self._STRATEGIES}"
            )
        self.replay_k = replay_k
        self.strategy = strategy
        self.boost = boost
        self.decay_amount = decay_amount
        self.traces: List[MemoryTrace] = []
        self._device = get_device()

    # ── 添加痕迹 ──────────────────────────────────────────────

    def add_trace(self, content: Dict, linguistic: List[str],
                  timestamp: int, success: bool) -> MemoryTrace:
        """添加一条新的记忆痕迹

        Args:
            content: 场景特征字典
            linguistic: 语言符号列表
            timestamp: 轮次编号
            success: 是否成功

        Returns:
            创建的 MemoryTrace 实例
        """
        trace = MemoryTrace(
            content=content,
            linguistic=list(linguistic),
            timestamp=timestamp,
            strength=1.0,
            replay_count=0,
            success=success,
        )
        self.traces.append(trace)
        return trace

    # ── 痕迹选择 ──────────────────────────────────────────────

    def select_traces(self, strategy: Optional[str] = None,
                      k: Optional[int] = None) -> List[MemoryTrace]:
        """按策略选择 k 条痕迹用于重播

        Args:
            strategy: 采样策略，None 使用默认策略
            k: 选择数量，None 使用默认 replay_k

        Returns:
            选中的痕迹列表（可能不足 k 条）
        """
        strat = strategy or self.strategy
        count = k or self.replay_k
        count = min(count, len(self.traces))

        if count == 0:
            return []

        if strat == 'random':
            indices = torch.randperm(len(self.traces))[:count].tolist()
            return [self.traces[i] for i in indices]

        if strat == 'prioritize_success':
            sorted_traces = sorted(
                self.traces,
                key=lambda t: (t.success, t.strength),
                reverse=True,
            )
            return sorted_traces[:count]

        if strat == 'prioritize_recent':
            sorted_traces = sorted(
                self.traces,
                key=lambda t: t.timestamp,
                reverse=True,
            )
            return sorted_traces[:count]

        if strat == 'prioritize_surprising':
            # 出人意料 = 偏离 0.5 成功率程度最大的
            # success=True 且整体成功率 > 0.5 → 不意外
            # success=False 且整体成功率 < 0.5 → 不意外
            success_rate = (
                sum(1 for t in self.traces if t.success) / max(len(self.traces), 1)
            )
            sorted_traces = sorted(
                self.traces,
                key=lambda t: abs(float(t.success) - success_rate),
                reverse=True,
            )
            return sorted_traces[:count]

        # 回退到 random
        indices = torch.randperm(len(self.traces))[:count].tolist()
        return [self.traces[i] for i in indices]

    # ── 巩固 ──────────────────────────────────────────────────

    def consolidate(self) -> Dict:
        """执行一轮睡眠巩固

        步骤:
          1. 按当前策略选择痕迹
          2. 强化 strength > 0.5 的，削弱 strength <= 0.5 的
          3. 合并相同符号键的痕迹（保留最强者）

        Returns:
            {'strengthened': n, 'weakened': n, 'merged': n}
        """
        if len(self.traces) == 0:
            return {'strengthened': 0, 'weakened': 0, 'merged': 0}

        selected = self.select_traces()
        strengthened = 0
        weakened = 0

        for trace in selected:
            trace.replay_count += 1
            if trace.strength > 0.5:
                trace.strength = min(1.0, trace.strength + self.boost)
                strengthened += 1
            else:
                trace.strength = max(0.0, trace.strength - self.decay_amount)
                weakened += 1

        # 合并相似痕迹：相同排序后的符号键 → 保留最强者
        merged = self._merge_similar()

        return {
            'strengthened': strengthened,
            'weakened': weakened,
            'merged': merged,
        }

    def _merge_similar(self) -> int:
        """合并相同符号键的痕迹，保留最强者，被合并者的强度加到幸存者上

        Returns:
            被合并移除的痕迹数
        """
        groups: Dict[str, List[int]] = {}
        for i, trace in enumerate(self.traces):
            key = ','.join(sorted(trace.linguistic)) if trace.linguistic else ''
            groups.setdefault(key, []).append(i)

        to_remove: set = set()
        for key, indices in groups.items():
            if len(indices) <= 1:
                continue
            # 找到强度最高的作为幸存者
            best_idx = max(indices, key=lambda i: self.traces[i].strength)
            survivor = self.traces[best_idx]
            for i in indices:
                if i != best_idx:
                    # 将被合并者的部分强度转移
                    survivor.strength = min(
                        1.0,
                        survivor.strength + self.traces[i].strength * 0.3,
                    )
                    to_remove.add(i)

        # 反向删除，保持索引有效
        for i in sorted(to_remove, reverse=True):
            self.traces.pop(i)

        return len(to_remove)

    # ── 遗忘 ──────────────────────────────────────────────────

    def forget(self, decay_rate: float = 0.01) -> int:
        """时间衰减遗忘

        对所有痕迹应用衰减，巩固过的记忆享受 0.6x 保护率。
        移除 strength < 0.1 的痕迹。

        Args:
            decay_rate: 基础衰减率

        Returns:
            被清除的痕迹数量
        """
        initial_count = len(self.traces)

        for trace in self.traces:
            protection = 0.6 if trace.replay_count > 0 else 1.0
            trace.strength -= decay_rate * protection
            trace.strength = max(0.0, trace.strength)

        # 移除过弱的痕迹
        self.traces = [t for t in self.traces if t.strength >= 0.1]

        return initial_count - len(self.traces)

    # ── 检索 ──────────────────────────────────────────────────

    def retrieve(self, query_features: Dict, query_linguistic: List[str],
                 k: int = 5) -> List[MemoryTrace]:
        """双通道检索：特征重叠 + 语言符号 Jaccard 相似度

        评分 = 0.5 × 特征重叠度 + 0.5 × Jaccard(linguistic)
        再乘以痕迹的 strength 加权。

        Args:
            query_features: 查询特征字典
            query_linguistic: 查询符号列表
            k: 返回前 k 条

        Returns:
            按评分降序排列的痕迹列表
        """
        if len(self.traces) == 0:
            return []

        k = min(k, len(self.traces))
        query_keys = set(query_features.keys())
        query_ling_set = set(query_linguistic)

        scores = torch.zeros(len(self.traces), dtype=torch.float32)

        for i, trace in enumerate(self.traces):
            # 特征重叠度：交集键数 / 并集键数
            trace_keys = set(trace.content.keys())
            if len(query_keys) == 0 and len(trace_keys) == 0:
                feature_sim = 1.0
            else:
                union = query_keys | trace_keys
                if len(union) == 0:
                    feature_sim = 0.0
                else:
                    overlap = len(query_keys & trace_keys)
                    feature_sim = overlap / len(union)

            # Jaccard 相似度
            trace_ling_set = set(trace.linguistic)
            if len(query_ling_set) == 0 and len(trace_ling_set) == 0:
                ling_sim = 1.0
            else:
                ling_union = query_ling_set | trace_ling_set
                if len(ling_union) == 0:
                    ling_sim = 0.0
                else:
                    ling_inter = query_ling_set & trace_ling_set
                    ling_sim = len(ling_inter) / len(ling_union)

            # 加权评分 × 记忆强度
            scores[i] = (0.5 * feature_sim + 0.5 * ling_sim) * trace.strength

        topk_scores, topk_indices = torch.topk(scores, k)
        return [self.traces[idx] for idx in topk_indices.tolist()]

    # ── 统计 ──────────────────────────────────────────────────

    @property
    def trace_count(self) -> int:
        """当前痕迹总数"""
        return len(self.traces)

    @property
    def avg_strength(self) -> float:
        """平均痕迹强度"""
        if not self.traces:
            return 0.0
        return sum(t.strength for t in self.traces) / len(self.traces)

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        """序列化引擎状态"""
        traces_data = []
        for t in self.traces:
            traces_data.append({
                'content': t.content,
                'linguistic': t.linguistic,
                'timestamp': t.timestamp,
                'strength': t.strength,
                'replay_count': t.replay_count,
                'success': t.success,
            })
        return {
            'replay_k': self.replay_k,
            'strategy': self.strategy,
            'boost': self.boost,
            'decay_amount': self.decay_amount,
            'traces': traces_data,
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复引擎状态"""
        self.replay_k = state.get('replay_k', self.replay_k)
        self.strategy = state.get('strategy', self.strategy)
        self.boost = state.get('boost', self.boost)
        self.decay_amount = state.get('decay_amount', self.decay_amount)

        self.traces = []
        for td in state.get('traces', []):
            self.traces.append(MemoryTrace(
                content=td['content'],
                linguistic=td['linguistic'],
                timestamp=td['timestamp'],
                strength=td.get('strength', 1.0),
                replay_count=td.get('replay_count', 0),
                success=td.get('success', False),
            ))
