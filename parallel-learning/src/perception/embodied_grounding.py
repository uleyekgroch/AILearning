"""感觉运动接地 — 从符号到体验

概念不仅通过语言学习，还通过物理交互接地。

核心思想（Xu et al. 2025 Nature HBM）：
- LLM在感觉运动域与人类表示显著偏离
- 视觉学习能改善视觉相关维度的对齐
- 需要通过物理交互"感受"概念

接地层次：
1. 具体名词：球、墙、桌子 → 视觉+触觉
2. 动作词：推、拉、抓 → 运动序列
3. 空间词：上、下、左、右 → 位置关系
4. 抽象概念：大、快、热 → 多模态融合

设计原则：
- 渐进接地：先具体后抽象
- 多模态融合：视觉+触觉+运动+本体感觉
- 交互学习：通过尝试和反馈学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SensorimotorExperience:
    """感觉运动经验"""
    visual: Optional[torch.Tensor] = None      # 视觉特征
    tactile: Optional[torch.Tensor] = None     # 触觉特征
    proprioceptive: Optional[torch.Tensor] = None  # 本体感觉
    motor: Optional[torch.Tensor] = None       # 运动序列
    reward: float = 0.0                        # 奖励
    timestamp: float = 0.0


class VisualEncoder(nn.Module):
    """视觉编码器

    将视觉输入编码为特征向量。
    简化实现：使用MLP处理向量化输入。
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, visual_input: torch.Tensor) -> torch.Tensor:
        """编码视觉输入"""
        return self.encoder(visual_input)


class TactileEncoder(nn.Module):
    """触觉编码器

    编码触觉信息（压力、温度、纹理）。
    """

    def __init__(self, input_dim: int = 32, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, tactile_input: torch.Tensor) -> torch.Tensor:
        """编码触觉输入"""
        return self.encoder(tactile_input)


class MotorEncoder(nn.Module):
    """运动编码器

    编码运动序列（关节角度、速度、加速度）。
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, motor_sequence: torch.Tensor) -> torch.Tensor:
        """编码运动序列

        Args:
            motor_sequence: (seq_len, input_dim) 运动序列

        Returns:
            (output_dim,) 运动特征
        """
        if motor_sequence.dim() == 1:
            motor_sequence = motor_sequence.unsqueeze(0).unsqueeze(0)
        elif motor_sequence.dim() == 2:
            motor_sequence = motor_sequence.unsqueeze(0)

        _, hidden = self.rnn(motor_sequence)
        return self.fc(hidden.squeeze(0)).squeeze(0)


class MultimodalFusion(nn.Module):
    """多模态融合层

    将视觉、触觉、运动特征融合为统一表示。
    """

    def __init__(self, visual_dim: int = 64, tactile_dim: int = 32,
                 motor_dim: int = 32, output_dim: int = 128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + tactile_dim + motor_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(visual_dim + tactile_dim + motor_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, visual: torch.Tensor, tactile: torch.Tensor,
                motor: torch.Tensor) -> torch.Tensor:
        """融合多模态特征"""
        # 拼接
        combined = torch.cat([visual, tactile, motor])

        # 注意力加权
        attn_weights = self.attention(combined)
        weighted_visual = visual * attn_weights[0]
        weighted_tactile = tactile * attn_weights[1]
        weighted_motor = motor * attn_weights[2]

        # 融合
        weighted_combined = torch.cat([weighted_visual, weighted_tactile, weighted_motor])
        return self.fusion(weighted_combined)


class ConceptGrounding(nn.Module):
    """概念接地层

    将语言概念与感觉运动经验关联。
    """

    def __init__(self, concept_dim: int = 128, experience_dim: int = 128):
        super().__init__()
        # 概念到经验的映射
        self.concept_to_experience = nn.Sequential(
            nn.Linear(concept_dim, experience_dim),
            nn.GELU(),
            nn.Linear(experience_dim, experience_dim),
        )

        # 经验到概念的映射
        self.experience_to_concept = nn.Sequential(
            nn.Linear(experience_dim, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, concept_dim),
        )

    def ground_concept(self, concept_embedding: torch.Tensor,
                      experience: torch.Tensor) -> torch.Tensor:
        """将概念接地到经验"""
        # 映射到经验空间
        concept_in_experience = self.concept_to_experience(concept_embedding)

        # 计算与经验的相似度
        similarity = F.cosine_similarity(
            concept_in_experience.unsqueeze(0),
            experience.unsqueeze(0)
        )

        return similarity

    def lift_experience(self, experience: torch.Tensor) -> torch.Tensor:
        """将经验提升为概念"""
        return self.experience_to_concept(experience)


class EmbodiedGroundingSystem:
    """感觉运动接地系统

    整合视觉、触觉、运动编码和多模态融合。
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

        # 编码器
        self.visual_encoder = VisualEncoder().to(self.device)
        self.tactile_encoder = TactileEncoder().to(self.device)
        self.motor_encoder = MotorEncoder().to(self.device)
        self.fusion = MultimodalFusion().to(self.device)
        self.grounding = ConceptGrounding().to(self.device)

        # 经验库
        self.experiences: Dict[str, List[SensorimotorExperience]] = {}

        # 概念接地映射
        self.grounded_concepts: Dict[str, torch.Tensor] = {}

    def perceive(self, visual: Optional[torch.Tensor] = None,
                tactile: Optional[torch.Tensor] = None,
                motor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """感知并融合多模态输入"""
        # 默认值
        if visual is None:
            visual = torch.zeros(256, device=self.device)
        else:
            visual = visual.to(self.device)
        if tactile is None:
            tactile = torch.zeros(32, device=self.device)
        else:
            tactile = tactile.to(self.device)
        if motor is None:
            motor = torch.zeros(64, device=self.device)
        else:
            motor = motor.to(self.device)

        # 编码
        visual_feat = self.visual_encoder(visual)
        tactile_feat = self.tactile_encoder(tactile)
        motor_feat = self.motor_encoder(motor)

        # 融合
        return self.fusion(visual_feat, tactile_feat, motor_feat)

    def ground_concept(self, concept: str, concept_embedding: torch.Tensor,
                      experience: Optional[SensorimotorExperience] = None) -> float:
        """将概念接地到感觉运动经验"""
        if experience is None:
            # 使用存储的经验
            if concept in self.experiences and self.experiences[concept]:
                experience = self.experiences[concept][-1]
            else:
                return 0.0

        # 感知经验
        exp_vector = self.perceive(
            experience.visual,
            experience.tactile,
            experience.motor,
        )

        # 接地
        similarity = self.grounding.ground_concept(concept_embedding, exp_vector)

        # 存储接地结果
        self.grounded_concepts[concept] = exp_vector

        return similarity.item()

    def store_experience(self, concept: str, experience: SensorimotorExperience):
        """存储感觉运动经验"""
        if concept not in self.experiences:
            self.experiences[concept] = []
        self.experiences[concept].append(experience)

    def get_grounded_embedding(self, concept: str) -> Optional[torch.Tensor]:
        """获取接地后的概念嵌入"""
        return self.grounded_concepts.get(concept)

    def progressive_grounding(self, concept: str, concept_type: str,
                            concept_embedding: torch.Tensor) -> torch.Tensor:
        """渐进接地

        根据概念类型选择接地策略：
        - 具体名词：视觉+触觉
        - 动作词：运动序列
        - 空间词：位置关系
        - 抽象概念：多模态融合
        """
        if concept_type == 'concrete':
            # 具体名词：视觉+触觉为主
            return self._ground_concrete(concept, concept_embedding)
        elif concept_type == 'action':
            # 动作词：运动为主
            return self._ground_action(concept, concept_embedding)
        elif concept_type == 'spatial':
            # 空间词：位置关系
            return self._ground_spatial(concept, concept_embedding)
        else:
            # 抽象概念：多模态融合
            return self._ground_abstract(concept, concept_embedding)

    def _ground_concrete(self, concept: str, embedding: torch.Tensor) -> torch.Tensor:
        """接地具体名词"""
        # 使用视觉+触觉经验
        if concept in self.experiences:
            experiences = self.experiences[concept]
            if experiences:
                latest = experiences[-1]
                return self.perceive(latest.visual, latest.tactile, None)
        return embedding

    def _ground_action(self, concept: str, embedding: torch.Tensor) -> torch.Tensor:
        """接地动作词"""
        # 使用运动经验
        if concept in self.experiences:
            experiences = self.experiences[concept]
            if experiences:
                latest = experiences[-1]
                return self.perceive(None, None, latest.motor)
        return embedding

    def _ground_spatial(self, concept: str, embedding: torch.Tensor) -> torch.Tensor:
        """接地空间词"""
        # 使用位置关系
        return embedding

    def _ground_abstract(self, concept: str, embedding: torch.Tensor) -> torch.Tensor:
        """接地抽象概念"""
        # 使用所有模态
        if concept in self.experiences:
            experiences = self.experiences[concept]
            if experiences:
                latest = experiences[-1]
                return self.perceive(latest.visual, latest.tactile, latest.motor)
        return embedding
