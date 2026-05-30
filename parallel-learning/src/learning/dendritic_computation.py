#!/usr/bin/env python3
"""树突计算 (Dendritic Computation)

基于论文:
- Chavlis & Poirazi, Nature Communications 2025:
  "Dendrites endow ANNs with accurate, robust and parameter-efficient learning"
- Larkum et al. 2009: "A new cellular mechanism for coupling inputs"
- Guerguiev et al. 2017: "Towards deep learning with segregated dendrites"

核心思想:
  生物神经元不是简单的点神经元，而是具有空间结构的计算单元。
  顶端树突(apical)接收上下文/反馈信号，基底树突(basal)接收前馈输入。
  两者在soma产生非线性交互（plateau potential），实现上下文关联学习。

  关键机制:
  1. 区室化(Compartmentalization): 不同树突区室有独立的可塑性规则
  2. 高原电位(Plateau Potential): 顶端+基底同时活跃时产生超线性响应
  3. 上下文调制(Context Modulation): 顶端输入不直接驱动，而是调制基底响应
  4. 树突非线性(Dendritic Nonlinearity): 二次积分规则（NeurIPS 2024）

  对学习系统的意义:
  - 一个实体可以根据上下文有不同的表征（消歧义）
  - 通过plateau potential实现单次关联学习（与BTSP互补）
  - 更少参数学到更多模式（parameter-efficient）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class DendriticNeuronState:
    """单个树突神经元的状态"""
    entity_id: str
    basal_input: torch.Tensor       # 基底树突输入（前馈）
    apical_input: torch.Tensor      # 顶端树突输入（上下文）
    somatic_output: torch.Tensor    # 胞体输出
    plateau_active: bool = False    # 是否产生高原电位
    context_id: str = ''            # 当前上下文标识


class DendriticCompartment(nn.Module):
    """树突区室 — 具有独立可塑性的计算单元

    每个区室有自己的权重和局部学习规则。
    顶端区室: 上下文关联（慢学习，稳定）
    基底区室: 前馈特征（快学习，灵活）
    """

    def __init__(self, input_dim: int, output_dim: int, compartment_type: str = 'basal'):
        super().__init__()
        self.compartment_type = compartment_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 权重矩阵
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)

        # 区室特异的学习率
        if compartment_type == 'apical':
            # 顶端: 慢学习，稳定记忆
            self.local_lr = 0.001
            self.decay_rate = 0.999  # 几乎不遗忘
        else:
            # 基底: 快学习，灵活适应
            self.local_lr = 0.01
            self.decay_rate = 0.99

        # 非线性激活阈值
        self.threshold = nn.Parameter(torch.tensor(0.5))

        # 局部活动历史（用于Hebbian学习）
        self.pre_trace = None
        self.post_trace = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 + 局部非线性"""
        # 线性变换
        linear_out = x @ self.weights

        # 树突非线性: 二次积分（NeurIPS 2024 findings）
        # 当输入模式匹配时产生超线性响应
        quadratic = (x @ self.weights) ** 2 * 0.1
        output = linear_out + quadratic

        # 阈值机制
        output = torch.where(output > self.threshold, output, output * 0.1)

        # 更新活动历史
        self.pre_trace = x.detach()
        self.post_trace = output.detach()

        return output

    def hebbian_update(self, reward_signal: float = 1.0):
        """局部Hebbian可塑性更新

        不依赖全局梯度，只使用局部pre/post活动。
        reward_signal作为第三因子调制（与GHL机制互补）。
        """
        if self.pre_trace is None or self.post_trace is None:
            return

        # Hebbian: ΔW = η * pre * post * reward
        pre = self.pre_trace
        post = self.post_trace

        # 外积计算
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)

        # 只更新协同活跃的连接
        delta_w = self.local_lr * reward_signal * (pre.T @ post)

        # 权重衰减（防止爆炸）
        self.weights.data *= self.decay_rate

        # 应用更新
        self.weights.data += delta_w

        # 权重归一化（保持稳定）
        norm = self.weights.data.norm(dim=0, keepdim=True).clamp(min=1.0)
        self.weights.data /= norm


class DendriticNeuron(nn.Module):
    """完整的树突神经元

    结构:
      apical dendrite (顶端) → context modulation
      basal dendrite (基底)  → feedforward input
      soma (胞体)            → 非线性整合 + 输出

    plateau potential:
      当apical和basal同时强烈活跃时，
      soma产生超线性burst（高原电位），
      触发强可塑性窗口。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 两个独立的树突区室
        self.basal = DendriticCompartment(d_model, d_model, 'basal')
        self.apical = DendriticCompartment(d_model, d_model, 'apical')

        # 胞体整合
        self.soma_gate = nn.Linear(d_model * 2, d_model)
        self.plateau_threshold = 0.7  # plateau触发阈值

        # 输出投射
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, feedforward: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Args:
            feedforward: 前馈输入（来自感知/下层）
            context: 上下文输入（来自高层反馈/记忆）

        Returns:
            output: 胞体输出
            plateau: 是否触发了plateau potential
        """
        # 基底处理前馈
        basal_out = self.basal(feedforward)

        # 顶端处理上下文
        apical_out = self.apical(context)

        # 检测plateau条件：两个区室都强烈活跃
        basal_activity = basal_out.norm().item()
        apical_activity = apical_out.norm().item()

        plateau = (basal_activity > self.plateau_threshold and
                   apical_activity > self.plateau_threshold)

        # 胞体非线性整合
        combined = torch.cat([basal_out.flatten(), apical_out.flatten()])
        if combined.shape[0] != self.d_model * 2:
            # 处理维度不匹配
            combined = F.adaptive_avg_pool1d(
                combined.unsqueeze(0).unsqueeze(0),
                self.d_model * 2
            ).squeeze()

        soma_out = self.soma_gate(combined)

        if plateau:
            # Plateau: 超线性增益 + 强可塑性
            soma_out = soma_out * 2.0  # burst模式
            # 触发强Hebbian学习
            self.basal.hebbian_update(reward_signal=2.0)
            self.apical.hebbian_update(reward_signal=2.0)

        output = self.output_proj(soma_out)
        return output, plateau


class DendriticComputationSystem:
    """树突计算系统

    管理多个树突神经元，实现:
    1. 上下文相关的实体表征（同一实体在不同上下文有不同激活）
    2. 单次关联学习（通过plateau potential）
    3. 灵活性-稳定性平衡（basal快适应, apical慢稳定）
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 每个实体/概念一个树突神经元
        self.neurons: Dict[str, DendriticNeuron] = {}

        # 上下文记忆：记录哪些上下文下看到过哪些实体
        self.context_associations: Dict[str, List[str]] = {}  # entity → [contexts]

        # plateau事件记录
        self.plateau_events: List[Dict] = []

        # 统计
        self.total_plateaus = 0
        self.total_activations = 0

    def get_or_create_neuron(self, entity_id: str) -> DendriticNeuron:
        """获取或创建实体对应的树突神经元"""
        if entity_id not in self.neurons:
            neuron = DendriticNeuron(self.d_model).to(self.device)
            self.neurons[entity_id] = neuron
        return self.neurons[entity_id]

    def compute_context_representation(
        self,
        entity_id: str,
        entity_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """计算实体在特定上下文下的表征

        这是树突计算的核心：同一个实体（如"苹果"）
        在"水果"上下文和"公司"上下文下会有不同的表征。

        Args:
            entity_id: 实体标识
            entity_embedding: 实体的基础嵌入（前馈）
            context_embedding: 当前上下文嵌入（来自句子/段落）

        Returns:
            contextualized_repr: 上下文化的表征
            plateau_fired: 是否触发了plateau（强关联学习信号）
        """
        neuron = self.get_or_create_neuron(entity_id)

        # 确保维度匹配
        if entity_embedding.shape[-1] != self.d_model:
            entity_embedding = F.adaptive_avg_pool1d(
                entity_embedding.unsqueeze(0).unsqueeze(0), self.d_model
            ).squeeze()
        if context_embedding.shape[-1] != self.d_model:
            context_embedding = F.adaptive_avg_pool1d(
                context_embedding.unsqueeze(0).unsqueeze(0), self.d_model
            ).squeeze()

        # 树突计算
        output, plateau = neuron(entity_embedding, context_embedding)

        self.total_activations += 1
        if plateau:
            self.total_plateaus += 1
            self.plateau_events.append({
                'entity': entity_id,
                'timestamp': self.total_activations,
            })
            # 限制事件记录大小
            if len(self.plateau_events) > 1000:
                self.plateau_events = self.plateau_events[-500:]

        # 记录上下文关联
        if entity_id not in self.context_associations:
            self.context_associations[entity_id] = []

        return output, plateau

    def associate_in_context(
        self,
        entity_a: str,
        entity_b: str,
        context_embedding: torch.Tensor,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> float:
        """在特定上下文中关联两个实体

        模拟plateau potential驱动的关联学习：
        当两个实体在同一上下文中共现时，
        它们的树突神经元同步产生plateau，
        强化彼此在该上下文下的联系。

        Returns:
            association_strength: 关联强度(0-1)
        """
        # 分别计算两个实体的上下文表征
        repr_a, plateau_a = self.compute_context_representation(
            entity_a, emb_a, context_embedding
        )
        repr_b, plateau_b = self.compute_context_representation(
            entity_b, emb_b, context_embedding
        )

        # 计算关联强度
        with torch.no_grad():
            similarity = F.cosine_similarity(
                repr_a.unsqueeze(0), repr_b.unsqueeze(0)
            ).item()

        # plateau同步加成
        if plateau_a and plateau_b:
            similarity = min(1.0, similarity + 0.3)

        return max(0.0, similarity)

    def disambiguate(
        self,
        entity_id: str,
        entity_embedding: torch.Tensor,
        candidate_contexts: List[Tuple[str, torch.Tensor]],
    ) -> Tuple[str, torch.Tensor]:
        """消歧义：确定实体在哪个上下文语义下被使用

        例如："苹果" 在 [水果, 科技公司] 两个上下文中，
        根据当前输入模式选择最匹配的上下文。

        Args:
            entity_id: 实体标识
            entity_embedding: 实体嵌入
            candidate_contexts: [(context_name, context_embedding), ...]

        Returns:
            best_context: 最佳上下文名
            best_repr: 该上下文下的表征
        """
        if not candidate_contexts:
            return '', entity_embedding

        neuron = self.get_or_create_neuron(entity_id)

        best_score = -1.0
        best_context = ''
        best_repr = entity_embedding

        for ctx_name, ctx_emb in candidate_contexts:
            repr_out, plateau = self.compute_context_representation(
                entity_id, entity_embedding, ctx_emb
            )
            # 活跃度作为匹配分数
            score = repr_out.norm().item()
            if plateau:
                score *= 1.5  # plateau是强匹配信号

            if score > best_score:
                best_score = score
                best_context = ctx_name
                best_repr = repr_out

        return best_context, best_repr

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_neurons': len(self.neurons),
            'total_activations': self.total_activations,
            'total_plateaus': self.total_plateaus,
            'plateau_rate': self.total_plateaus / max(1, self.total_activations),
            'context_associations': sum(len(v) for v in self.context_associations.values()),
        }
