#!/usr/bin/env python3
"""主动推理 (Active Inference)

基于论文:
- Parr, Pezzulo & Friston 2022: "Active Inference: The Free Energy Principle
  in Mind, Brain, and Behavior"
- Friston et al. 2024: "Active inference in artificial agents and humans"
- Pezzulo et al. 2023: "Active inference, predictive coding, and slow-thinking"

核心思想:
  感知、学习和行动统一在自由能最小化框架下：

  F = D_KL[q(s)||p(s|o)] - E_q[ln p(o|s)]
    = 复杂度 - 准确度

  系统不被动接收数据，而是主动选择能最大化信息增益的行动：
  1. 认知主动推理 (Epistemic foraging): 选择最不确定的领域探索
  2. 实用主动推理 (Pragmatic foraging): 选择最有价值的目标
  3. 精度调制 (Precision modulation): 注意力分配基于预期信息量

  与好奇心驱动的关系:
  - 好奇心 = 内在奖励驱动探索
  - 主动推理 = 自由能最小化驱动的最优信息获取
  - 两者互补：主动推理提供"去哪里探索"的策略，
    好奇心提供"探索多深入"的动力

  对学习系统的意义:
  - 系统自主决定"下一步该学什么"
  - 不再是被动逐条读取语料，而是主动选择信息量最大的样本
  - 实现真正的自主学习闭环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class CandidateAction:
    """候选行动"""
    action_type: str          # 'explore', 'consolidate', 'ask', 'skip'
    target: str               # 目标内容/领域
    expected_info_gain: float # 预期信息增益
    expected_value: float     # 预期实用价值
    free_energy: float        # 预期自由能
    uncertainty: float        # 目标领域的不确定性


class GenerativeModel(nn.Module):
    """生成模型 p(o|s) — 从隐状态生成观测

    在主动推理中，智能体维护一个世界模型：
    给定内部状态s，预测会观测到什么o。

    自由能 = 实际观测与预测之间的差异
    最小化自由能 = 使预测尽可能准确
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # 状态到观测的映射（似然模型 p(o|s)）
        self.likelihood = nn.Sequential(
            nn.Linear(state_dim, obs_dim * 2),
            nn.ReLU(),
            nn.Linear(obs_dim * 2, obs_dim),
        )

        # 状态转移模型 p(s_t|s_{t-1})
        self.transition = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 先验偏好 p(s) — 哪些状态是"好的"
        self.prior_mean = nn.Parameter(torch.zeros(state_dim))
        self.prior_log_var = nn.Parameter(torch.zeros(state_dim))

    def predict_observation(self, state: torch.Tensor) -> torch.Tensor:
        """预测观测: p(o|s)"""
        return self.likelihood(state)

    def predict_next_state(self, state: torch.Tensor) -> torch.Tensor:
        """预测下一状态: p(s_t|s_{t-1})"""
        return self.transition(state)

    def compute_free_energy(self, state: torch.Tensor,
                            observation: torch.Tensor) -> torch.Tensor:
        """计算变分自由能

        F = -ln p(o|s) + D_KL[q(s)||p(s)]
          = 准确度项 + 复杂度项
        """
        # 准确度: 预测与观测的匹配度
        predicted = self.predict_observation(state)
        accuracy = -F.mse_loss(predicted, observation)

        # 复杂度: 后验与先验的KL散度
        prior_var = torch.exp(self.prior_log_var)
        kl_div = 0.5 * (
            self.prior_log_var.sum()
            + ((state - self.prior_mean) ** 2 / prior_var).sum()
        )

        return -accuracy + kl_div


class ActiveInferenceSystem:
    """主动推理系统

    实现Friston的主动推理框架：
    1. 维护内部生成模型（世界模型）
    2. 通过最小化自由能来更新信念
    3. 主动选择行动来最大化预期信息增益

    用于：
    - 决定下一步学什么（认知主动性）
    - 评估当前知识的不确定性
    - 在探索（信息获取）和利用（知识应用）间平衡
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 生成模型
        self.generative_model = GenerativeModel(d_model, d_model).to(self.device)
        self.model_optimizer = torch.optim.Adam(
            self.generative_model.parameters(), lr=1e-4
        )

        # 不确定性地图：记录每个领域/实体的不确定性
        self.uncertainty_map: Dict[str, float] = {}

        # 预测误差历史
        self.prediction_errors: List[float] = []

        # 已学习的领域
        self.learned_domains: Dict[str, int] = {}  # domain → count

        # 信息增益历史（用于元学习）
        self.info_gain_history: List[float] = []

    def update_beliefs(self, entity_id: str, observation: torch.Tensor) -> Dict:
        """更新对实体的信念

        核心：感知即推断。
        当观察到新信息时，更新内部状态使自由能最小化。

        Args:
            entity_id: 实体标识
            observation: 观测到的嵌入向量

        Returns:
            更新结果（自由能变化、不确定性变化等）
        """
        obs = observation.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 当前状态估计（用观测初始化）
        state = obs.detach().clone().requires_grad_(True)

        # 自由能最小化（梯度下降）
        prev_fe = float('inf')
        for step in range(5):  # 5步迭代推断
            fe = self.generative_model.compute_free_energy(state, obs)

            # 手动梯度下降更新状态
            if state.grad is not None:
                state.grad.zero_()
            fe.backward()

            with torch.no_grad():
                state -= 0.1 * state.grad

            current_fe = fe.item()
            if abs(prev_fe - current_fe) < 1e-5:
                break
            prev_fe = current_fe

        # 更新不确定性
        fe_value = prev_fe
        self.uncertainty_map[entity_id] = fe_value
        self.prediction_errors.append(fe_value)

        # 训练生成模型（更新参数使未来预测更好）
        self._train_generative_model(obs)

        return {
            'free_energy': fe_value,
            'uncertainty_change': fe_value - self.uncertainty_map.get(entity_id, fe_value),
        }

    def _train_generative_model(self, observation: torch.Tensor):
        """训练生成模型 — 让预测更准确"""
        self.generative_model.train()

        state = observation.detach().clone()
        predicted = self.generative_model.predict_observation(state)
        loss = F.mse_loss(predicted, observation)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def compute_expected_info_gain(self, candidate_embedding: torch.Tensor) -> float:
        """计算候选内容的预期信息增益

        信息增益 = 当前不确定性 × 预测准确度改善

        高信息增益 = 学这个会让我们的知识显著增加

        Args:
            candidate_embedding: 候选内容的嵌入

        Returns:
            预期信息增益（0-1）
        """
        emb = candidate_embedding.to(self.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        # 用生成模型预测
        with torch.no_grad():
            predicted = self.generative_model.predict_observation(emb)
            prediction_error = F.mse_loss(predicted, emb).item()

        # 预测误差越大 → 我们越不了解这个领域 → 信息增益越高
        info_gain = 1.0 - math.exp(-prediction_error)
        return min(1.0, info_gain)

    def select_next_learning_target(
        self,
        candidates: List[Tuple[str, torch.Tensor]],
        mode: str = 'balanced',
    ) -> Tuple[str, float, Dict]:
        """选择下一个学习目标 — 主动推理的核心决策

        Args:
            candidates: [(text, embedding), ...] 候选学习内容
            mode: 选择策略
                'balanced' — 平衡探索与利用
                'explore' — 纯探索（最大化信息增益）
                'exploit' — 纯利用（巩固已有知识）

        Returns:
            selected_text: 选中的文本
            expected_gain: 预期信息增益
            decision_info: 决策详情
        """
        if not candidates:
            return '', 0.0, {}

        scored = []
        for text, embedding in candidates:
            # 计算信息增益
            info_gain = self.compute_expected_info_gain(embedding)

            # 计算领域覆盖度（是否是新领域）
            domain_key = text[:5]  # 简单前缀作为领域
            domain_count = self.learned_domains.get(domain_key, 0)
            novelty = 1.0 / (1.0 + domain_count * 0.1)

            # 综合评分
            if mode == 'explore':
                score = info_gain * 0.8 + novelty * 0.2
            elif mode == 'exploit':
                score = novelty * 0.2 + (1.0 - info_gain) * 0.8
            else:  # balanced
                # 主动推理的最优策略：最大化预期自由能降低
                score = info_gain * 0.5 + novelty * 0.3 + (1.0 - domain_count / 100) * 0.2

            scored.append((score, text, embedding, info_gain))

        # 选择得分最高的
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text, best_emb, best_gain = scored[0]

        # 更新领域计数
        domain_key = best_text[:5]
        self.learned_domains[domain_key] = self.learned_domains.get(domain_key, 0) + 1
        self.info_gain_history.append(best_gain)

        return best_text, best_gain, {
            'score': best_score,
            'info_gain': best_gain,
            'mode': mode,
            'candidates_evaluated': len(candidates),
        }

    def select_curriculum(
        self,
        all_texts: List[str],
        encoder_fn,
        batch_size: int = 100,
    ) -> List[Tuple[str, float]]:
        """选择学习课程 — 从全部语料中选择信息量最大的子集

        这是主动推理的最重要应用：
        不是顺序读语料，而是选择最有价值的样本。

        策略：
        1. 随机采样候选
        2. 计算每个候选的信息增益
        3. 按信息增益排序
        4. 选择前batch_size个

        Args:
            all_texts: 全部可用文本
            encoder_fn: 编码函数 text → embedding
            batch_size: 选择多少条

        Returns:
            [(text, info_gain), ...] 排序后的选择
        """
        import random

        # 阶段1：随机采样候选（避免全量编码开销）
        sample_size = min(len(all_texts), batch_size * 5)
        sampled = random.sample(all_texts, sample_size)

        # 阶段2：计算信息增益
        scored = []
        for text in sampled:
            try:
                emb = encoder_fn(text)
                info_gain = self.compute_expected_info_gain(emb)
                scored.append((text, info_gain))
            except Exception:
                scored.append((text, 0.0))

        # 阶段3：排序选择
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:batch_size]

        return selected

    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_uncertainty = 0.0
        if self.uncertainty_map:
            avg_uncertainty = sum(self.uncertainty_map.values()) / len(self.uncertainty_map)

        avg_info_gain = 0.0
        if self.info_gain_history:
            avg_info_gain = sum(self.info_gain_history[-100:]) / len(self.info_gain_history[-100:])

        return {
            'tracked_entities': len(self.uncertainty_map),
            'avg_uncertainty': avg_uncertainty,
            'avg_info_gain': avg_info_gain,
            'domains_covered': len(self.learned_domains),
            'total_predictions': len(self.prediction_errors),
        }
