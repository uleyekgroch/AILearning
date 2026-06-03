"""通用学习循环 — 将 KnowledgeUnit 学习与现有 Learner 集成

核心流程：
1. 编码：将 KnowledgeUnit 转换为 Learner 可感知的输入
2. 感知：Learner.perceive() 编码为内部表示
3. 预测：Learner.predict_next() 产生预测
4. 学习：Learner.learn_from_experience() 通过预测误差更新
5. 掌握：更新 KnowledgeUnit 的掌握度

这不是一个新的 Learner，而是对现有 Learner 的适配层，
让 Learner 能够学习任何领域的 KnowledgeUnit。
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch

from src.core.learner import Learner
from src.knowledge.unit import KnowledgeUnit, MasteryLevel
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation


class UniversalLearner:
    """通用学习器 — 适配层，让 Learner 学习任何领域的知识单元

    职责：
    1. 将 KnowledgeUnit 编码为 Learner 可感知的输入
    2. 调用 Learner 的感知-预测-学习闭环
    3. 更新 KnowledgeUnit 的掌握度
    4. 将知识单元同步到 KnowledgeGraph
    """

    def __init__(self, learner: Learner, knowledge_graph: KnowledgeGraph):
        """
        Args:
            learner: 现有的 Learner 实例
            knowledge_graph: 知识图谱
        """
        self.learner = learner
        self.kg = knowledge_graph
        self._unit_cache: Dict[str, KnowledgeUnit] = {}

    def learn_unit(self, unit: KnowledgeUnit, n_practice: int = 3) -> Dict:
        """学习一个知识单元

        完整流程：
        1. 感知阶段：编码知识单元为多模态输入
        2. 预测阶段：产生预测
        3. 学习阶段：通过预测误差更新权重
        4. 练习阶段：多次练习巩固
        5. 掌握更新：更新掌握度

        Args:
            unit: 要学习的知识单元
            n_practice: 练习次数

        Returns:
            学习结果字典
        """
        results = {
            'unit_id': unit.id,
            'unit_name': unit.name,
            'domain': unit.domain,
            'phases': [],
            'final_mastery': unit.mastery,
            'mastery_change': 0.0,
        }

        old_mastery = unit.mastery

        # ── 阶段 1：感知 ──────────────────────────────────────
        obs = self._encode_unit(unit)
        unit.exposure_count += 1

        if unit.mastery_level == MasteryLevel.UNKNOWN:
            unit.mastery_level = MasteryLevel.EXPOSED

        results['phases'].append({
            'phase': 'perception',
            'obs_shape': list(obs.shape),
        })

        # ── 阶段 2：预测 + 学习 ──────────────────────────────
        action = self.learner.choose_action(obs)
        predicted = self.learner.predict_next(obs, action)

        # 生成目标（基于知识单元的定义和示例）
        target = self._generate_target(unit, obs)

        # 学习
        error = self.learner.learn_from_experience(obs, action, target)

        results['phases'].append({
            'phase': 'prediction_learning',
            'action': action,
            'error': round(error, 6),
        })

        # ── 阶段 3：练习 ──────────────────────────────────────
        practice_results = []
        for i in range(n_practice):
            # 生成练习变体
            practice_obs = self._generate_practice_variant(unit, obs, i)
            practice_action = self.learner.choose_action(practice_obs)
            practice_target = self._generate_target(unit, practice_obs)
            practice_error = self.learner.learn_from_experience(
                practice_obs, practice_action, practice_target
            )

            # 评估练习结果
            success = practice_error < 0.5
            quality = max(0.0, 1.0 - practice_error)

            # 更新掌握度
            unit.update_mastery(success, quality)
            unit.practice_count += 1

            practice_results.append({
                'practice': i + 1,
                'error': round(practice_error, 6),
                'success': success,
                'quality': round(quality, 4),
                'mastery': round(unit.mastery, 4),
            })

        results['phases'].append({
            'phase': 'practice',
            'practices': practice_results,
        })

        # ── 阶段 4：记忆存储 ──────────────────────────────────
        self.learner.remember(obs, action, target, 1.0 - error, error)

        # ── 阶段 5：知识图谱同步 ──────────────────────────────
        self._sync_to_knowledge_graph(unit)

        # ── 更新缓存 ──────────────────────────────────────────
        self._unit_cache[unit.id] = unit
        unit.last_practice = time.time()

        results['final_mastery'] = round(unit.mastery, 4)
        results['mastery_change'] = round(unit.mastery - old_mastery, 4)
        results['mastery_level'] = unit.mastery_level.name

        return results

    def review_unit(self, unit: KnowledgeUnit) -> Dict:
        """复习一个知识单元

        比学习更轻量：只做预测和少量练习。
        用于间隔重复。
        """
        results = {
            'unit_id': unit.id,
            'type': 'review',
            'old_mastery': round(unit.mastery, 4),
        }

        # 感知
        obs = self._encode_unit(unit)
        unit.exposure_count += 1

        # 预测
        action = self.learner.choose_action(obs)
        target = self._generate_target(unit, obs)
        error = self.learner.learn_from_experience(obs, action, target)

        # 单次练习
        success = error < 0.5
        quality = max(0.0, 1.0 - error)
        unit.update_mastery(success, quality)
        unit.practice_count += 1
        unit.last_practice = time.time()

        # 记忆
        self.learner.remember(obs, action, target, 1.0 - error, error)

        # 同步
        self._sync_to_knowledge_graph(unit)

        results['error'] = round(error, 6)
        results['success'] = success
        results['new_mastery'] = round(unit.mastery, 4)
        results['mastery_level'] = unit.mastery_level.name

        return results

    def assess_unit(self, unit: KnowledgeUnit) -> Dict:
        """评估一个知识单元的掌握程度

        不更新掌握度，只返回评估结果。
        """
        obs = self._encode_unit(unit)
        action = self.learner.choose_action(obs)
        target = self._generate_target(unit, obs)

        # 预测
        predicted = self.learner.predict_next(obs, action)
        error = torch.nn.functional.mse_loss(predicted, target).item()

        success = error < 0.5
        quality = max(0.0, 1.0 - error)

        return {
            'unit_id': unit.id,
            'unit_name': unit.name,
            'error': round(error, 6),
            'success': success,
            'quality': round(quality, 4),
            'current_mastery': round(unit.mastery, 4),
            'mastery_level': unit.mastery_level.name,
        }

    # ── 内部方法 ──────────────────────────────────────────────

    def _encode_unit(self, unit: KnowledgeUnit) -> torch.Tensor:
        """将知识单元编码为 Learner 可感知的观测向量

        使用哈希编码将文本信息转换为固定维度的向量。
        """
        # 构建文本描述
        text = f"{unit.name}: {unit.definition}"
        if unit.examples:
            text += f" Example: {unit.examples[0]}"

        # 哈希编码为多模态输入
        visual = self._text_to_visual(text)
        auditory = self._text_to_auditory(text)
        position = torch.zeros(2)

        raw_input = {
            'visual': visual,
            'auditory': auditory,
            'position': position,
        }

        return self.learner.perceive(raw_input)

    def _text_to_visual(self, text: str) -> torch.Tensor:
        """将文本哈希编码为视觉特征图 (4, 8, 8)"""
        channels, h, w = 4, 8, 8
        size = channels * h * w
        vec = torch.zeros(size)

        for i, ch in enumerate(text):
            idx = (hash(ch) + i * 7) % size
            vec[idx] += (ord(ch) % 100) / 100.0

        # 归一化
        max_val = vec.abs().max()
        if max_val > 0:
            vec = vec / max_val

        return vec.reshape(channels, h, w)

    def _text_to_auditory(self, text: str) -> torch.Tensor:
        """将文本哈希编码为听觉特征 (13,)"""
        vec = torch.zeros(13)
        for i, ch in enumerate(text):
            idx = (hash(ch) + i * 3) % 13
            vec[idx] += (ord(ch) % 100) / 100.0

        max_val = vec.abs().max()
        if max_val > 0:
            vec = vec / max_val

        return vec

    def _generate_target(self, unit: KnowledgeUnit, obs: torch.Tensor) -> torch.Tensor:
        """生成学习目标

        基于知识单元的定义和示例，生成一个与观测略有不同的目标向量。
        差异模拟了"学习新知识"的过程。
        """
        # 用定义文本编码目标
        text = unit.definition
        if unit.examples:
            text += " " + unit.examples[0]

        visual = self._text_to_visual(text)
        auditory = self._text_to_auditory(text)
        position = torch.zeros(2)

        raw_input = {
            'visual': visual,
            'auditory': auditory,
            'position': position,
        }

        target = self.learner.perceive(raw_input)

        # 添加小量噪声，模拟学习的不确定性
        noise = torch.randn_like(target) * 0.1
        target = target + noise

        return target

    def _generate_practice_variant(self, unit: KnowledgeUnit,
                                    original_obs: torch.Tensor,
                                    variant_idx: int) -> torch.Tensor:
        """生成练习变体

        使用不同的示例或关键词生成略有不同的输入，
        模拟从不同角度学习同一概念。
        """
        # 使用不同示例
        if variant_idx < len(unit.examples):
            text = f"{unit.name}: {unit.examples[variant_idx]}"
        elif unit.explanations:
            text = f"{unit.name}: {unit.explanations[0]}"
        else:
            text = f"{unit.name}: {unit.definition}"

        visual = self._text_to_visual(text + str(variant_idx))
        auditory = self._text_to_auditory(text + str(variant_idx))
        position = torch.zeros(2)

        raw_input = {
            'visual': visual,
            'auditory': auditory,
            'position': position,
        }

        return self.learner.perceive(raw_input)

    def _sync_to_knowledge_graph(self, unit: KnowledgeUnit) -> None:
        """将知识单元同步到知识图谱

        创建或更新实体，并建立关系。
        """
        # 检查实体是否已存在
        entity = self.kg.get_entity(unit.id)

        if entity is None:
            # 创建新实体
            entity = Entity(
                id=unit.id,
                type='knowledge_unit',
                properties={
                    'name': unit.name,
                    'domain': unit.domain,
                    'definition': unit.definition,
                    'difficulty': unit.difficulty,
                    'mastery': unit.mastery,
                    'mastery_level': unit.mastery_level.name,
                },
                embedding=None,
                confidence=unit.mastery,
                source=unit.source,
                tags=unit.tags,
            )
            self.kg.add_entity(entity)
        else:
            # 更新现有实体
            entity.confidence = unit.mastery
            entity.properties['mastery'] = unit.mastery
            entity.properties['mastery_level'] = unit.mastery_level.name

        # 建立关系
        for related_id in unit.related:
            self._ensure_relation(unit.id, related_id, 'related')

        for prereq_id in unit.prerequisites:
            self._ensure_relation(prereq_id, unit.id, 'prerequisite')

        for is_a_id in unit.is_a:
            self._ensure_relation(unit.id, is_a_id, 'is_a')

    def _ensure_relation(self, source_id: str, target_id: str,
                         relation_type: str) -> None:
        """确保关系存在"""
        # 检查目标实体是否存在
        if self.kg.get_entity(target_id) is None:
            # 创建占位实体
            target_entity = Entity(
                id=target_id,
                type='knowledge_unit',
                properties={},
                confidence=0.0,
            )
            self.kg.add_entity(target_entity)

        # 检查关系是否已存在
        existing = self.kg.get_relations_of(source_id, direction='outgoing')
        for rel in existing:
            if rel.target_id == target_id and rel.type == relation_type:
                return  # 已存在

        # 创建新关系
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            type=relation_type,
            confidence=0.5,
        )
        self.kg.add_relation(relation)
