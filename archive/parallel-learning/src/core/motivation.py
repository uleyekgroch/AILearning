"""
内在动机系统 — 4 种驱动 + 知识评估 + 目标选择

驱动类型:
  curiosity    — 好奇心驱动: 预测误差 × 可学习性
  novelty      — 新奇性驱动: 1 / (1 + visit_count)
  empowerment  — 赋能感驱动: 动作影响的方差
  info_gain    — 信息增益驱动: 预测不确定性的减少量

辅助模块:
  KnowledgeAssessor — 评估知识图谱中各领域的掌握程度
  GoalSelector      — 基于知识评估选择学习目标

所有数值计算使用 torch.Tensor，零 numpy 依赖。
"""

import math
from typing import Dict, List, Optional

import torch

from src.core.device import get_device, to_device
from src.knowledge.graph import KnowledgeGraph


class IntrinsicMotivation:
    """4 种内在动机驱动的统一计算

    每种驱动返回 [0, 1] 区间的标量奖励，
    compute_drive_rewards 按权重合成综合内在奖励。

    Args:
        obs_dim: 观测向量维度
        action_dim: 动作空间维度
    """

    def __init__(self, obs_dim: int = 40, action_dim: int = 8):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._device = get_device()

        # 新奇性驱动: 动作访问计数
        self._visit_counts = torch.zeros(action_dim, dtype=torch.float32,
                                         device=self._device)

        # 好奇心驱动: 预测误差历史
        self._error_history: List[float] = []

        # 赋能感驱动: 缓存
        self._empowerment_map: Dict[int, float] = {}

    # ── 好奇心驱动 ────────────────────────────────────────────

    def curiosity_drive(self, prediction_error: float) -> float:
        """好奇心驱动: 预测误差 × 可学习性

        可学习性 = exp(-mean(error_history))：误差越高说明越不可学，
        但完全不熟悉的领域也需要探索。两者平衡。

        Args:
            prediction_error: 当前预测误差 (MSE)

        Returns:
            好奇心奖励值 [0, 1]
        """
        self._error_history.append(prediction_error)
        if len(self._error_history) > 200:
            self._error_history = self._error_history[-200:]

        # 可学习性: 误差历史越低 → 越可学（已掌握的不需要太多好奇心）
        mean_error = sum(self._error_history) / len(self._error_history)
        learnability = math.exp(-mean_error)

        # 预测误差归一化到 [0, 1]
        normalized_error = 1.0 - math.exp(-prediction_error)

        drive = normalized_error * learnability
        return min(max(drive, 0.0), 1.0)

    # ── 新奇性驱动 ────────────────────────────────────────────

    def novelty_drive(self, obs: torch.Tensor, memory=None) -> float:
        """新奇性驱动: 基于动作访问计数的逆频率

        新奇性 = 1 / (1 + visit_count)
        未访问过的动作返回 1.0，频繁访问的趋向 0.0。

        Args:
            obs: 当前观测张量 (obs_dim,)
            memory: 外部记忆模块（可选，用于观测级去重）

        Returns:
            新奇性奖励值 [0, 1]
        """
        obs = to_device(obs, self._device)

        # 从观测中推断动作偏好：取 obs 的前 action_dim 个分量做 softmax
        obs_slice = obs[:self.action_dim].float()
        probs = torch.softmax(obs_slice, dim=0)

        # 加权访问计数
        weighted_visits = (probs * self._visit_counts).sum()

        # 更新访问计数
        self._visit_counts += probs.detach()

        novelty = 1.0 / (1.0 + weighted_visits.item())
        return min(max(novelty, 0.0), 1.0)

    # ── 赋能感驱动 ────────────────────────────────────────────

    def empowerment_drive(self, obs: torch.Tensor, n_samples: int = 10) -> float:
        """赋能感驱动: 衡量动作对环境的影响力

        通过采样多个动作方向，计算观测变化量的方差。
        方差越大 → 动作越能产生差异化影响 → 赋能感越高。

        Args:
            obs: 当前观测张量 (obs_dim,)
            n_samples: 采样动作数

        Returns:
            赋能感奖励值 [0, 1]
        """
        obs = to_device(obs, self._device).float()
        obs_key = hash(obs.data_ptr())

        if obs_key in self._empowerment_map:
            return self._empowerment_map[obs_key]

        # 采样随机动作向量
        actions = torch.randn(n_samples, self.action_dim,
                              device=self._device)
        actions = torch.softmax(actions, dim=-1)

        # 模拟观测变化: 简化为 obs 与 action 的线性变换
        # 用一个固定投影矩阵模拟环境响应
        projection = torch.randn(self.action_dim, self.obs_dim,
                                 device=self._device) * 0.1
        effects = torch.matmul(actions, projection)  # (n_samples, obs_dim)

        # 每个动作效果的 L2 范数
        norms = effects.norm(dim=1)  # (n_samples,)

        # 方差归一化: variance / mean 作为影响力度量
        variance = norms.var().item()
        mean = norms.mean().item() + 1e-8
        empowerment = variance / (variance + mean)

        # 缓存结果
        self._empowerment_map[obs_key] = empowerment

        # 限制缓存大小
        if len(self._empowerment_map) > 500:
            oldest = list(self._empowerment_map.keys())[:250]
            for k in oldest:
                del self._empowerment_map[k]

        return min(max(empowerment, 0.0), 1.0)

    # ── 信息增益驱动 ──────────────────────────────────────────

    def info_gain_drive(self, obs: torch.Tensor, predictor=None) -> float:
        """信息增益驱动: 预测不确定性的减少量

        若有 predictor（预测编码模型），使用其前后不确定性差值。
        否则基于观测本身的熵估计。

        Args:
            obs: 当前观测张量 (obs_dim,)
            predictor: 预测模型（可选，需实现 get_uncertainty 方法）

        Returns:
            信息增益奖励值 [0, 1]
        """
        obs = to_device(obs, self._device).float()

        if predictor is not None and hasattr(predictor, 'get_uncertainty'):
            # 使用预测器的不确定性
            uncertainty_before = predictor.get_uncertainty(obs)
            # 假设学习后的不确定性降低
            uncertainty_after = uncertainty_before * 0.8
            info_gain = max(0.0, uncertainty_before - uncertainty_after)
            return min(info_gain, 1.0)

        # 无预测器时: 基于观测分布的熵估计
        # 归一化后计算各维度方差作为不确定性代理
        obs_normalized = obs / (obs.abs().max() + 1e-8)

        # 离散化到桶后计算经验熵
        n_bins = 10
        hist = torch.histc(obs_normalized, bins=n_bins, min=-1.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)

        # Shannon 熵
        entropy = -(hist * torch.log(hist + 1e-8)).sum().item()

        # 最大可能熵（均匀分布）
        max_entropy = math.log(n_bins)

        # 归一化到 [0, 1]
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        return min(max(normalized_entropy, 0.0), 1.0)

    # ── 综合奖励 ──────────────────────────────────────────────

    def compute_drive_rewards(self, obs: torch.Tensor, memory=None,
                               predictor=None, knowledge=None,
                               weights: Optional[Dict[str, float]] = None
                               ) -> torch.Tensor:
        """计算所有驱动的加权综合内在奖励

        Args:
            obs: 当前观测张量 (obs_dim,)
            memory: 外部记忆模块（用于 novelty_drive）
            predictor: 预测模型（用于 info_gain_drive）
            knowledge: 知识图谱（可选，用于动态调整权重）
            weights: 各驱动的权重字典，默认均分

        Returns:
            综合内在奖励标量张量
        """
        default_weights = {
            'curiosity': 0.4,
            'novelty': 0.2,
            'empowerment': 0.2,
            'info_gain': 0.2,
        }
        w = weights or default_weights

        curiosity_val = self.curiosity_drive(
            prediction_error=self._last_prediction_error()
        )
        novelty_val = self.novelty_drive(obs, memory)
        empowerment_val = self.empowerment_drive(obs)
        info_gain_val = self.info_gain_drive(obs, predictor)

        total = (
            w.get('curiosity', 0.0) * curiosity_val
            + w.get('novelty', 0.0) * novelty_val
            + w.get('empowerment', 0.0) * empowerment_val
            + w.get('info_gain', 0.0) * info_gain_val
        )

        return torch.tensor(total, dtype=torch.float32, device=self._device)

    def _last_prediction_error(self) -> float:
        """获取最近一次预测误差，无历史时返回 0.5"""
        if self._error_history:
            return self._error_history[-1]
        return 0.5


class KnowledgeAssessor:
    """评估知识图谱中每个领域的掌握程度

    对每种实体类型计算优先级:
      priority = (1 - avg_confidence) × log(1 + count)

    置信度低且数量多的领域优先级高（需要更多学习）。

    Args:
        knowledge: 知识图谱实例
    """

    def __init__(self, knowledge: KnowledgeGraph):
        self.knowledge = knowledge

    def assess(self) -> Dict[str, float]:
        """评估所有实体类型的掌握程度

        Returns:
            {entity_type: priority} 字典，
            priority 越高说明该领域越需要加强学习。
        """
        type_dist = self.knowledge.type_distribution()
        priorities: Dict[str, float] = {}

        for etype, count in type_dist.items():
            entities = self.knowledge.query(entity_type=etype)
            if not entities:
                priorities[etype] = 0.0
                continue

            avg_conf = sum(e.confidence for e in entities) / len(entities)
            priority = (1.0 - avg_conf) * math.log(1 + count)
            priorities[etype] = priority

        return priorities

    def get_weak_dimensions(self, threshold: float = 0.3) -> List[str]:
        """获取薄弱维度：平均置信度低于阈值的实体类型

        Args:
            threshold: 置信度阈值

        Returns:
            薄弱实体类型名称列表
        """
        weak = []
        for etype in self.knowledge.type_distribution():
            entities = self.knowledge.query(entity_type=etype)
            if not entities:
                weak.append(etype)
                continue
            avg_conf = sum(e.confidence for e in entities) / len(entities)
            if avg_conf < threshold:
                weak.append(etype)
        return weak

    def get_strong_dimensions(self, threshold: float = 0.7) -> List[str]:
        """获取优势维度：平均置信度高于阈值的实体类型

        Args:
            threshold: 置信度阈值

        Returns:
            优势实体类型名称列表
        """
        strong = []
        for etype in self.knowledge.type_distribution():
            entities = self.knowledge.query(entity_type=etype)
            if not entities:
                continue
            avg_conf = sum(e.confidence for e in entities) / len(entities)
            if avg_conf >= threshold:
                strong.append(etype)
        return strong


class GoalSelector:
    """基于知识评估选择学习目标

    使用 epsilon-greedy 策略:
      90% 概率选择优先级最高的目标
      10% 概率随机探索（避免过早收敛）

    Args:
        epsilon: 随机探索概率，默认 0.1
    """

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def select_goals(self, priorities: Dict[str, float],
                     num_goals: int = 3) -> List[str]:
        """从优先级字典中选择学习目标

        Args:
            priorities: {dimension: priority} 字典
            num_goals: 选择的目标数量

        Returns:
            目标维度名称列表
        """
        if not priorities:
            return []

        num_goals = min(num_goals, len(priorities))
        available = list(priorities.keys())

        # 按优先级降序排列
        sorted_dims = sorted(available, key=lambda d: priorities[d], reverse=True)

        goals: List[str] = []
        for _ in range(num_goals):
            rand_val = torch.rand(1).item()
            if rand_val < self.epsilon and len(available) > 0:
                # 随机探索
                idx = torch.randint(0, len(available), (1,)).item()
                chosen = available[idx]
            else:
                # 选择最高优先级中尚未被选中的
                chosen = sorted_dims[0]

            goals.append(chosen)

            # 从候选中移除已选，避免重复
            if chosen in available:
                available.remove(chosen)
            if chosen in sorted_dims:
                sorted_dims.remove(chosen)

            if not available:
                break

        return goals

    def generate_practice_focus(self, weak_dims: List[str]) -> Dict[str, float]:
        """生成针对薄弱维度的注意力权重

        薄弱维度分配更高的注意力权重。
        权重按 softmax 归一化，使总和为 1。

        Args:
            weak_dims: 薄弱维度名称列表

        Returns:
            {dimension: attention_weight} 字典
        """
        if not weak_dims:
            return {}

        # 每个薄弱维度分配一个基础分数（越多维度分数越均匀）
        n = len(weak_dims)
        raw_scores = torch.ones(n, dtype=torch.float32) * 2.0

        # 应用 softmax 得到归一化注意力权重
        attn_weights = torch.softmax(raw_scores, dim=0)

        return {dim: attn_weights[i].item() for i, dim in enumerate(weak_dims)}
