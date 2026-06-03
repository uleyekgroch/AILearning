#!/usr/bin/env python3
"""具身符号接地 (Embodied Symbol Grounding)

基于论文:
- INTSPEECH 2019: "A Computational Model of Early Language Acquisition
  from Multimodal Input" — 多模态(音频-视觉)引导词义学习。
- Harnad 1990: "The Symbol Grounding Problem" — 符号必须接地到感知。
- Barsalou 2008: "Grounded Cognition" — 认知根植于感知-行动系统。
- Cornell 2025: "Power of Babble" — 婴儿主动引导学习环境。

核心思想:
  符号（词语）本身没有意义，必须通过感知体验来"接地"：
  - "红色" = 看到过红色物体
  - "热" = 触摸过热的东西
  - "大" = 对比过大小不同的物体

  没有接地，系统只在做符号操作（就像中文房间论证）。
  有了接地，符号才有真正的语义。

  具身接地的三通道:
  1. 视觉通道: 形状、颜色、大小、位置
  2. 感觉通道: 温度、硬度、重量
  3. 行动通道: 推、拉、抓、放

  当前系统没有真实传感器，但我们用文本属性模拟:
  - "红色的苹果" → 视觉特征(红色) + 实体(苹果) 绑定
  - "热水" → 感觉特征(热) + 实体(水) 绑定
  - "用力推门" → 行动特征(推) + 实体(门) 绑定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class GroundedSymbol:
    """接地的符号

    一个符号（词/概念）与其感知特征的绑定。
    """
    symbol: str                      # 符号（如"红色"、"热"）
    sensory_features: Dict[str, float]  # 感觉特征（视觉/触觉/行动）
    bound_entities: List[str]        # 绑定的实体列表
    grounding_strength: float        # 接地强度(0-1)
    experience_count: int            # 体验次数


class EmbodiedGroundingSystem:
    """具身符号接地系统

    核心功能:
    1. 从文本中提取感知特征（颜色、大小、温度、行动等）
    2. 将符号与感知特征绑定（接地）
    3. 通过体验增强接地强度
    4. 支持基于接地的语义相似度计算
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 接地符号库
        self.grounded_symbols: Dict[str, GroundedSymbol] = {}

        # 感知特征模式（从文本提取）
        self.sensory_patterns = {
            # 视觉特征
            'color': ['红', '蓝', '绿', '黄', '白', '黑', '紫', '橙', '灰'],
            'size': ['大', '小', '巨大', '微小', '高', '矮', '长', '短', '宽', '窄'],
            'shape': ['圆', '方', '三角', '平', '尖', '弯曲', '直'],
            'brightness': ['亮', '暗', '发光', '闪'],

            # 感觉特征
            'temperature': ['热', '冷', '温', '冰', '烫', '凉'],
            'texture': ['软', '硬', '光滑', '粗糙', '粗糙', '柔软'],
            'weight': ['重', '轻', '沉'],
            'taste': ['甜', '苦', '酸', '辣', '咸', '淡'],

            # 行动特征
            'action': ['推', '拉', '抓', '放', '举', '扔', '打', '抱',
                       '走', '跑', '跳', '飞', '游', '爬'],
            'force': ['用力', '轻轻', '快速', '缓慢', '突然'],
        }

        # 特征到嵌入向量的映射（可学习）
        self.feature_embeddings: Dict[str, torch.Tensor] = {}

        # 统计
        self.stats = {
            'symbols_grounded': 0,
            'entities_bound': 0,
            'experiences_recorded': 0,
            'features_extracted': 0,
        }

    def extract_features(self, text: str) -> Dict[str, List[str]]:
        """从文本中提取感知特征

        Returns:
            {feature_type: [matched_features]}
        """
        extracted = {}

        for feature_type, patterns in self.sensory_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern in text:
                    matches.append(pattern)
                    self.stats['features_extracted'] += 1
            if matches:
                extracted[feature_type] = matches

        return extracted

    def ground_symbol(self, symbol: str, features: Dict[str, float],
                      entity: str = '') -> GroundedSymbol:
        """将符号与感知特征绑定（接地）

        Args:
            symbol: 要接地的符号
            features: 感知特征字典 {feature_name: intensity(0-1)}
            entity: 关联的实体名
        """
        if symbol in self.grounded_symbols:
            # 增强已有接地
            gs = self.grounded_symbols[symbol]

            # 更新特征（EMA）
            for feat, value in features.items():
                old = gs.sensory_features.get(feat, 0.0)
                gs.sensory_features[feat] = old * 0.7 + value * 0.3

            # 增强接地强度
            gs.grounding_strength = min(1.0, gs.grounding_strength + 0.1)
            gs.experience_count += 1

            # 添加新实体
            if entity and entity not in gs.bound_entities:
                gs.bound_entities.append(entity)
                self.stats['entities_bound'] += 1

            self.stats['experiences_recorded'] += 1
        else:
            # 创建新接地
            bound = [entity] if entity else []
            gs = GroundedSymbol(
                symbol=symbol,
                sensory_features=features,
                bound_entities=bound,
                grounding_strength=0.3,  # 初始弱接地
                experience_count=1,
            )
            self.grounded_symbols[symbol] = gs
            self.stats['symbols_grounded'] += 1
            if entity:
                self.stats['entities_bound'] += 1

        return gs

    def ground_from_text(self, text: str, entities: List[str]) -> List[str]:
        """从文本中自动提取并绑定感知特征

        扫描文本中的感知特征词，将其与相邻的实体绑定。
        """
        features = self.extract_features(text)
        grounded = []

        for feature_type, feature_words in features.items():
            for feat_word in feature_words:
                # 找最近的实体
                for entity in entities:
                    # 特征在实体附近 → 绑定
                    feat_pos = text.find(feat_word)
                    entity_pos = text.find(entity)
                    if feat_pos >= 0 and entity_pos >= 0:
                        distance = abs(feat_pos - entity_pos)
                        if distance < 20:  # 距离阈值
                            # 特征强度与距离反相关
                            intensity = max(0.3, 1.0 - distance / 20.0)
                            self.ground_symbol(
                                symbol=feat_word,
                                features={feature_type: intensity},
                                entity=entity,
                            )
                            grounded.append(f"{feat_word}→{entity}")

        return grounded

    def compute_grounded_similarity(self, symbol_a: str, symbol_b: str) -> float:
        """计算两个接地符号的相似度

        基于共享的感知特征，而非文本相似度。
        "红"和"热"可能有高相似度（都是"火"的属性）。
        """
        gs_a = self.grounded_symbols.get(symbol_a)
        gs_b = self.grounded_symbols.get(symbol_b)

        if not gs_a or not gs_b:
            return 0.0

        # 共享特征维度的相似度
        all_features = set(gs_a.sensory_features.keys()) | set(gs_b.sensory_features.keys())
        if not all_features:
            return 0.0

        similarity = 0.0
        for feat in all_features:
            val_a = gs_a.sensory_features.get(feat, 0.0)
            val_b = gs_b.sensory_features.get(feat, 0.0)
            # 特征值相似度
            similarity += 1.0 - abs(val_a - val_b)

        similarity /= len(all_features)

        # 共享实体的加成
        shared_entities = set(gs_a.bound_entities) & set(gs_b.bound_entities)
        if shared_entities:
            similarity = min(1.0, similarity + 0.2)

        return similarity

    def get_grounding_strength(self, symbol: str) -> float:
        """获取符号的接地强度"""
        gs = self.grounded_symbols.get(symbol)
        return gs.grounding_strength if gs else 0.0

    def get_stats(self) -> Dict:
        total_experiences = sum(
            gs.experience_count for gs in self.grounded_symbols.values()
        )
        return {
            **self.stats,
            'total_symbols': len(self.grounded_symbols),
            'avg_grounding_strength': (
                sum(gs.grounding_strength for gs in self.grounded_symbols.values()) /
                max(1, len(self.grounded_symbols))
            ),
            'total_experiences': total_experiences,
        }
