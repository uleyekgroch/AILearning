#!/usr/bin/env python3
"""语言发展阶段 (Language Development Stages)

基于论文:
- Meta AI 2024: "Emergence of Language in the Developing Brain"
  LLM表征与人类神经语言发展的平行关系。
- MIT News 2026: "Language Development in the Brain"
  来回对话改变儿童大脑发育。
- PMC 2024: "Neural Mechanisms of Language Development in Infancy"
  6个月大婴儿的alpha频段功率预测表达性语言能力。
- Cornell 2025: "Power of Babble"
  婴儿咿呀声引导照料者简化语言——反馈闭环。

核心思想:
  儿童语言发展遵循严格的阶段顺序：
  1. 感知阶段(0-6月): 咿呀学语，感知语音模式
  2. 单词阶段(10-18月): 第一个有意义的词
  3. 双词阶段(18-24月): "妈妈抱"、"还要奶"
  4. 复杂句阶段(2-5岁): 完整句子，语法规则
  5. 读写阶段(5+岁): 文字理解与创造

  每个阶段的能力建立在前一阶段之上（脚手架效应）。
  跳过阶段会导致语言能力缺陷。

  对学习系统的意义:
  - 系统不应直接从语料学习复杂关系，而应遵循发展阶段
  - 先建立基础概念，再学关系，再学推理
  - 当前"能力水平"决定可以学习什么内容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import math


class LanguageStage(Enum):
    """语言发展阶段"""
    PRELINGUISTIC = 0   # 前语言期：感知模式
    HOLOPHRASE = 1       # 单词期：单个概念
    TWO_WORD = 2          # 双词期：简单组合
    TELEGRAPHIC = 3       # 电报式：简单句
    COMPLEX = 4           # 复杂句：语法规则
    LITERACY = 5          # 读写期：文本理解


@dataclass
class StageProgress:
    """阶段进度"""
    stage: LanguageStage
    concepts_known: int        # 已知概念数
    relations_known: int       # 已知关系数
    combinations_formed: int   # 已形成的组合数
    sentences_generated: int   # 生成的句子数
    readiness_score: float     # 准备好进入下一阶段的分数(0-1)


# 各阶段的晋升阈值
STAGE_THRESHOLDS = {
    LanguageStage.PRELINGUISTIC: {'concepts': 0, 'relations': 0},
    LanguageStage.HOLOPHRASE: {'concepts': 10, 'relations': 5},
    LanguageStage.TWO_WORD: {'concepts': 30, 'relations': 20},
    LanguageStage.TELEGRAPHIC: {'concepts': 100, 'relations': 60},
    LanguageStage.COMPLEX: {'concepts': 300, 'relations': 200},
    LanguageStage.LITERACY: {'concepts': 1000, 'relations': 500},
}


class LanguageDevelopmentSystem:
    """语言发展阶段系统

    核心功能:
    1. 跟踪当前语言发展阶段
    2. 根据阶段过滤学习内容（不学超纲内容）
    3. 自动评估阶段晋升
    4. 模拟儿童的语言习得过程
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 当前阶段
        self.current_stage = LanguageStage.PRELINGUISTIC

        # 各阶段的知识积累
        self.knowledge_by_stage: Dict[LanguageStage, Dict] = {
            stage: {
                'concepts': set(),
                'relations': [],
                'combinations': [],
                'sentences': [],
            }
            for stage in LanguageStage
        }

        # 词汇表（按学习顺序）
        self.vocabulary: List[str] = []

        # 语法规则（在复杂阶段习得）
        self.grammar_rules: List[Dict] = []

        # 统计
        self.stats = {
            'stage_transitions': 0,
            'total_concepts': 0,
            'total_relations': 0,
            'total_combinations': 0,
            'total_sentences': 0,
            'filtered_by_stage': 0,
        }

    def assess_readiness(self) -> StageProgress:
        """评估当前阶段进度和晋升准备度"""
        total_concepts = sum(
            len(s['concepts']) for s in self.knowledge_by_stage.values()
        )
        total_relations = sum(
            len(s['relations']) for s in self.knowledge_by_stage.values()
        )
        total_combinations = sum(
            len(s['combinations']) for s in self.knowledge_by_stage.values()
        )
        total_sentences = sum(
            len(s['sentences']) for s in self.knowledge_by_stage.values()
        )

        # 计算晋升准备度
        next_stage_value = self.current_stage.value + 1
        if next_stage_value <= LanguageStage.LITERACY.value:
            next_stage = LanguageStage(next_stage_value)
            threshold = STAGE_THRESHOLDS[next_stage]
            concept_ratio = total_concepts / max(1, threshold['concepts'])
            relation_ratio = total_relations / max(1, threshold['relations'])
            readiness = min(1.0, (concept_ratio + relation_ratio) / 2)
        else:
            readiness = 1.0

        self.stats['total_concepts'] = total_concepts
        self.stats['total_relations'] = total_relations
        self.stats['total_combinations'] = total_combinations
        self.stats['total_sentences'] = total_sentences

        return StageProgress(
            stage=self.current_stage,
            concepts_known=total_concepts,
            relations_known=total_relations,
            combinations_formed=total_combinations,
            sentences_generated=total_sentences,
            readiness_score=readiness,
        )

    def check_stage_transition(self) -> bool:
        """检查是否应该晋升到下一阶段"""
        progress = self.assess_readiness()
        next_value = self.current_stage.value + 1

        if next_value > LanguageStage.LITERACY.value:
            return False

        next_stage = LanguageStage(next_value)
        threshold = STAGE_THRESHOLDS[next_stage]

        if (progress.concepts_known >= threshold['concepts'] and
            progress.relations_known >= threshold['relations']):
            self.current_stage = next_stage
            self.stats['stage_transitions'] += 1
            return True

        return False

    def can_learn(self, content_type: str, complexity: int = 1) -> bool:
        """检查当前阶段是否可以学习此内容

        防止系统跳过发展阶段直接学习高复杂度内容。

        Args:
            content_type: 'concept', 'relation', 'combination', 'sentence'
            complexity: 1-5 (1=简单, 5=复杂)
        """
        stage_level = self.current_stage.value

        if content_type == 'concept':
            return True  # 任何阶段都可以学概念
        elif content_type == 'relation':
            return stage_level >= LanguageStage.HOLOPHRASE.value
        elif content_type == 'combination':
            return stage_level >= LanguageStage.TWO_WORD.value
        elif content_type == 'sentence':
            if complexity <= 2:
                return stage_level >= LanguageStage.TELEGRAPHIC.value
            elif complexity <= 4:
                return stage_level >= LanguageStage.COMPLEX.value
            else:
                return stage_level >= LanguageStage.LITERACY.value
        else:
            return False

    def filter_content(self, entities: List[str], triples: List) -> Tuple[List[str], List]:
        """根据当前阶段过滤学习内容

        前语言期: 只学概念名
        单词期: 学概念+简单关系
        双词期: 学概念+关系+双词组合
        电报式: 学所有但限制复杂度
        复杂句+: 无限制
        """
        filtered_entities = entities
        filtered_triples = triples

        if self.current_stage == LanguageStage.PRELINGUISTIC:
            # 前语言期：只保留概念名，不学关系
            filtered_triples = []
            self.stats['filtered_by_stage'] += len(triples)

        elif self.current_stage == LanguageStage.HOLOPHRASE:
            # 单词期：学简单关系（短实体名）
            filtered_triples = [
                t for t in triples
                if len(t) >= 3 and len(t[0]) <= 6 and len(t[2]) <= 10
            ]
            self.stats['filtered_by_stage'] += len(triples) - len(filtered_triples)

        elif self.current_stage == LanguageStage.TWO_WORD:
            # 双词期：限制三元组数量
            filtered_triples = triples[:3]
            self.stats['filtered_by_stage'] += max(0, len(triples) - 3)

        return filtered_entities, filtered_triples

    def register_concept(self, concept: str):
        """注册一个概念到当前阶段"""
        stage_knowledge = self.knowledge_by_stage[self.current_stage]
        stage_knowledge['concepts'].add(concept)
        if concept not in self.vocabulary:
            self.vocabulary.append(concept)

    def register_relation(self, subject: str, relation: str, obj: str):
        """注册一个关系到当前阶段"""
        stage_knowledge = self.knowledge_by_stage[self.current_stage]
        stage_knowledge['relations'].append((subject, relation, obj))

    def register_combination(self, word_a: str, word_b: str, context: str):
        """注册一个词组合"""
        stage_knowledge = self.knowledge_by_stage[self.current_stage]
        stage_knowledge['combinations'].append((word_a, word_b, context))

    def register_sentence(self, sentence: str):
        """注册一个完整句子"""
        stage_knowledge = self.knowledge_by_stage[self.current_stage]
        stage_knowledge['sentences'].append(sentence)

    def get_learning_priorities(self) -> List[str]:
        """获取当前阶段应该优先学习的内容类型"""
        if self.current_stage == LanguageStage.PRELINGUISTIC:
            return ['high_frequency_words', 'basic_concepts']
        elif self.current_stage == LanguageStage.HOLOPHRASE:
            return ['concrete_objects', 'actions', 'simple_properties']
        elif self.current_stage == LanguageStage.TWO_WORD:
            return ['entity_relations', 'simple_combinations', 'possessives']
        elif self.current_stage == LanguageStage.TELEGRAPHIC:
            return ['causal_relations', 'temporal_sequences', 'simple_sentences']
        elif self.current_stage == LanguageStage.COMPLEX:
            return ['abstract_concepts', 'complex_reasoning', 'analogies']
        else:
            return ['domain_knowledge', 'creative_expression', 'meta_learning']

    def get_stats(self) -> Dict:
        progress = self.assess_readiness()
        return {
            **self.stats,
            'current_stage': self.current_stage.name,
            'stage_value': self.current_stage.value,
            'readiness': progress.readiness_score,
            'vocabulary_size': len(self.vocabulary),
        }
