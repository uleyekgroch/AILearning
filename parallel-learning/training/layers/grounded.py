"""接地表示层

将概念与感知经验关联。

核心能力：
1. 感知编码 — 编码视觉、听觉、触觉信息
2. 概念接地 — 将抽象概念与具体经验关联
3. 多模态融合 — 融合不同感知通道

运行方式：
    python training/layers/grounded.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PerceptualExperience:
    """感知经验"""
    visual: Optional[torch.Tensor] = None  # 视觉特征
    auditory: Optional[torch.Tensor] = None  # 听觉特征
    tactile: Optional[torch.Tensor] = None  # 触觉特征
    proprioceptive: Optional[torch.Tensor] = None  # 本体感觉
    context: Dict = field(default_factory=dict)  # 上下文


class VisualEncoder(nn.Module):
    """视觉编码器"""

    def __init__(self, input_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码视觉输入"""
        # x: [batch, channels, height, width]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


class AuditoryEncoder(nn.Module):
    """听觉编码器"""

    def __init__(self, input_dim: int = 128, feature_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, feature_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码听觉输入"""
        # x: [batch, time, features]
        lstm_out, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.fc(hidden)


class TactileEncoder(nn.Module):
    """触觉编码器"""

    def __init__(self, input_dim: int = 32, feature_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码触觉输入"""
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MultimodalFusion(nn.Module):
    """多模态融合"""

    def __init__(self, visual_dim: int = 128, auditory_dim: int = 128,
                 tactile_dim: int = 128, output_dim: int = 256):
        super().__init__()

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + auditory_dim + tactile_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, visual: torch.Tensor, auditory: torch.Tensor,
                tactile: torch.Tensor) -> torch.Tensor:
        """融合多模态特征"""
        # 拼接
        combined = torch.cat([visual, auditory, tactile], dim=1)

        # 融合
        fused = self.fusion(combined)

        return fused


class ConceptGrounding(nn.Module):
    """概念接地

    将抽象概念与感知经验关联。
    """

    def __init__(self, concept_dim: int = 128, perceptual_dim: int = 256,
                 output_dim: int = 256):
        super().__init__()

        # 概念编码
        self.concept_encoder = nn.Linear(concept_dim, perceptual_dim)

        # 接地层
        self.grounding = nn.Sequential(
            nn.Linear(perceptual_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )

        # 相似度计算
        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, concept: torch.Tensor, perceptual: torch.Tensor) -> torch.Tensor:
        """接地概念"""
        # 编码概念
        concept_encoded = self.concept_encoder(concept)

        # 拼接
        combined = torch.cat([concept_encoded, perceptual], dim=1)

        # 接地
        grounded = self.grounding(combined)

        return grounded

    def compute_similarity(self, concept: torch.Tensor, perceptual: torch.Tensor) -> torch.Tensor:
        """计算概念与感知的相似度"""
        concept_encoded = self.concept_encoder(concept)
        return self.similarity(concept_encoded, perceptual)


class GroundedRepresentationSystem:
    """接地表示系统"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 编码器
        self.visual_encoder = VisualEncoder().to(self.device)
        self.auditory_encoder = AuditoryEncoder().to(self.device)
        self.tactile_encoder = TactileEncoder().to(self.device)
        self.fusion = MultimodalFusion().to(self.device)
        self.grounding = ConceptGrounding().to(self.device)

        # 概念库
        self.concepts: Dict[str, torch.Tensor] = {}
        self.perceptual_memories: Dict[str, torch.Tensor] = {}

        # 统计
        self.stats = {
            'concepts_grounded': 0,
            'perceptual_memories': 0,
        }

    def encode_perception(self, visual: torch.Tensor = None,
                         auditory: torch.Tensor = None,
                         tactile: torch.Tensor = None) -> torch.Tensor:
        """编码感知"""
        # 默认值
        if visual is None:
            visual = torch.zeros(1, 3, 32, 32).to(self.device)
        if auditory is None:
            auditory = torch.zeros(1, 10, 128).to(self.device)
        if tactile is None:
            tactile = torch.zeros(1, 32).to(self.device)

        # 编码
        visual_features = self.visual_encoder(visual)
        auditory_features = self.auditory_encoder(auditory)
        tactile_features = self.tactile_encoder(tactile)

        # 融合
        fused = self.fusion(visual_features, auditory_features, tactile_features)

        return fused

    def ground_concept(self, concept_name: str, perceptual: torch.Tensor):
        """接地概念"""
        # 创建概念向量
        concept_dim = 128
        if concept_name not in self.concepts:
            self.concepts[concept_name] = torch.randn(1, concept_dim).to(self.device)

        # 接地
        grounded = self.grounding(self.concepts[concept_name], perceptual)

        # 存储
        self.perceptual_memories[concept_name] = grounded
        self.stats['concepts_grounded'] += 1
        self.stats['perceptual_memories'] += 1

        return grounded

    def compute_grounded_similarity(self, concept1: str, concept2: str) -> float:
        """计算两个接地概念的相似度"""
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return 0.0

        c1 = self.concepts[concept1]
        c2 = self.concepts[concept2]

        similarity = F.cosine_similarity(c1, c2)
        return similarity.item()

    def associate_perception(self, concept_name: str, visual: torch.Tensor = None,
                           auditory: torch.Tensor = None, tactile: torch.Tensor = None):
        """关联感知"""
        perceptual = self.encode_perception(visual, auditory, tactile)
        self.ground_concept(concept_name, perceptual)

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'device': str(self.device),
            'concepts_count': len(self.concepts),
        }


def test_grounded_representation():
    """测试接地表示"""
    print("=" * 70)
    print("接地表示测试")
    print("=" * 70)

    system = GroundedRepresentationSystem()

    # 测试感知编码
    print("\n1. 感知编码测试:")
    visual = torch.randn(1, 3, 32, 32).to(system.device)
    auditory = torch.randn(1, 10, 128).to(system.device)
    tactile = torch.randn(1, 32).to(system.device)

    perceptual = system.encode_perception(visual, auditory, tactile)
    print(f"  感知编码维度: {perceptual.shape}")

    # 测试概念接地
    print("\n2. 概念接地测试:")
    concepts = ['猫', '狗', '汽车', '飞机']

    for concept in concepts:
        grounded = system.ground_concept(concept, perceptual)
        print(f"  '{concept}' → 接地维度: {grounded.shape}")

    # 测试相似度
    print("\n3. 概念相似度测试:")
    pairs = [('猫', '狗'), ('汽车', '飞机'), ('猫', '汽车')]

    for c1, c2 in pairs:
        similarity = system.compute_grounded_similarity(c1, c2)
        print(f"  '{c1}' vs '{c2}': {similarity:.3f}")

    # 统计
    print("\n统计:")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_grounded_representation()
