"""语言习得系统 — 语言附着在感知概念上

认知科学基础：

    1. Piaget 语言发展理论：
       语言不是独立能力，而是认知发展的副产品。
       感知运动阶段(0-2岁)→前运算阶段(2-7岁)→...
       每个阶段的概念基础决定了能学什么语言。

    2. Vygotsky 社会文化理论：
       语言首先出现在社会互动中，然后内化为思维工具。
       ZPD（最近发展区）：在"有点难但能学会"的范围内学习最有效。

    3. Tomasello 使用基础理论（Usage-Based Theory, 2003）：
       语法不是预装规则，而是从具体使用中归纳出来的。
       儿童：听到1000次"X是Y" → 归纳出"...是..."模式 → 组合新句子。
       这与Chomsky的"先天语法"理论对立。

    4. 过度规则化（Overregularization）：
       儿童：goed, catched, mouses — 不是模仿，是规则过度应用。
       这证明儿童在提取抽象规则，不是简单记忆。

核心设计：
    语言学习不再通过正则表达式提取，而是三个阶段：
    Stage 1: 标签映射 — 文本标签附着到已有的感知概念
    Stage 2: 组合涌现 — 从已知概念组合新表达
    Stage 3: 语法提取 — 从使用中归纳语法规则（允许过度规则化）

    与正则提取的根本区别：
    - 正则：文本 → 模式匹配 → 存储（不理解）
    - 本系统：文本 → 概念激活 → 场景构建 → 理解 → 表达
"""

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter


@dataclass
class GrammarRule:
    """语法规则 — 从使用中归纳出的模式"""
    pattern: str          # 规则模式（如 "X是Y的Z"）
    slots: List[str]      # 槽位（如 ['X', 'Y', 'Z']）
    frequency: int = 1    # 出现频率
    confirmed: bool = False  # 是否已确认
    overregularized: bool = False  # 是否过度规则化


class LanguageAcquisitionSystem:
    """语言习得系统

    使用方式：
        las = LanguageAcquisitionSystem(concept_system, statistical_learner)

        # 从文本学习标签（附着到感知概念）
        labels = las.learn_label("数学是研究数量的学科")

        # 从语料中提取语法规则
        rules = las.extract_grammar_from_usage()

        # 组合性表达
        sentence = las.compose(["数学", "研究", "数量"])

        # 理解文本（通过概念激活而非正则）
        meaning = las.comprehend("数学研究什么？")
    """

    # 中文功能词（不构成独立概念）
    FUNCTION_CHARS = set('是的有在了和与被把让给从到以也而又或但其如果')

    def __init__(self, concept_system=None, statistical_learner=None,
                 concept_space=None):
        """
        Args:
            concept_system: FunctionalConceptSystem 实例
            statistical_learner: StatisticalLearner 实例
            concept_space: ConceptSpace 实例
        """
        self.concept_system = concept_system
        self.statistical_learner = statistical_learner
        self.concept_space = concept_space

        # 语法规则库
        self.grammar_rules: Dict[str, GrammarRule] = {}

        # 过度规则化日志
        self.overregularization_log: List[Dict] = []

        # 标签映射表：文本标签 → 概念ID
        self.label_to_concept: Dict[str, str] = {}

        # 已学习的句子模式（用于语法提取）
        self.learned_patterns: List[Dict] = []

        # 统计
        self._stats = {
            'labels_learned': 0,
            'rules_extracted': 0,
            'compositions': 0,
            'comprehensions': 0,
        }

    def learn_label(self, text: str, context: Dict = None,
                     perceptual_input: Dict = None) -> List[str]:
        """学习语言标签 — 将文本映射到感知概念

        流程：
        1. 从统计学习者获取文本中的候选概念
        2. 在概念空间中找到匹配的感知概念
        3. 建立双向映射：文本↔感知概念
        4. 如果没有匹配 → 用快速映射创建新概念

        Args:
            text: 输入文本
            context: 上下文信息
            perceptual_input: 可选的感知输入

        Returns:
            学到的标签列表
        """
        learned_labels = []

        # 1. 从统计学习者获取候选概念
        candidates = self._extract_candidates(text)

        for label in candidates:
            # 2. 在概念空间中查找
            concept_id = self._find_matching_concept(label)

            if concept_id:
                # 已有概念 → 建立标签映射
                self.label_to_concept[label] = concept_id
                learned_labels.append(label)
            elif self.concept_system:
                # 没有匹配 → 用快速映射创建
                node = self.concept_system.fast_map(
                    label=label,
                    context=context or {'known_labels': list(self.label_to_concept.keys())},
                    perceptual_input=perceptual_input,
                )
                if node:
                    self.label_to_concept[label] = node.id
                    learned_labels.append(label)

            self._stats['labels_learned'] += 1

        # 3. 记录句子模式（用于语法提取）
        if learned_labels:
            self.learned_patterns.append({
                'text': text,
                'labels': learned_labels,
                'pattern': self._extract_pattern(text, learned_labels),
            })

        return learned_labels

    def extract_grammar_from_usage(self) -> List[GrammarRule]:
        """从使用中提取语法规则

        类似儿童发现"-ed"规则的过程：
        1. 收集所有已知表达模式
        2. 发现共同的子结构
        3. 归纳为规则模板
        4. 允许过度规则化
        """
        new_rules = []

        if len(self.learned_patterns) < 3:
            return new_rules

        # 1. 统计模式频率
        pattern_freq = Counter()
        for lp in self.learned_patterns:
            pattern = lp.get('pattern', '')
            if pattern:
                pattern_freq[pattern] += 1

        # 2. 发现高频模式 → 归纳为规则
        for pattern, freq in pattern_freq.items():
            if freq >= 2 and pattern not in self.grammar_rules:
                # 检查是否有足够的变化实例
                variants = [lp for lp in self.learned_patterns
                           if lp.get('pattern') == pattern]

                if len(variants) >= 2:
                    # 归纳规则
                    slots = self._identify_slots(pattern, variants)
                    rule = GrammarRule(
                        pattern=pattern,
                        slots=slots,
                        frequency=freq,
                        confirmed=freq >= 3,
                    )
                    self.grammar_rules[pattern] = rule
                    new_rules.append(rule)
                    self._stats['rules_extracted'] += 1

                    # 检测过度规则化
                    if freq >= 3 and self._check_overregularization(pattern, variants):
                        rule.overregularized = True
                        self.overregularization_log.append({
                            'pattern': pattern,
                            'variants': [v['text'] for v in variants[:5]],
                        })

        return new_rules

    def compose(self, concepts: List[str], goal: str = 'describe') -> str:
        """组合性表达 — 从已知概念组合新句子

        不使用模板拼接，而是基于语法规则和概念关系自然组合。

        Args:
            concepts: 概念ID列表
            goal: 表达目标（describe/compare/explain）

        Returns:
            组合后的自然语言句子
        """
        self._stats['compositions'] += 1

        if not concepts:
            return ""

        # 1. 尝试用已有语法规则组合
        for pattern, rule in self.grammar_rules.items():
            if rule.confirmed and len(rule.slots) == len(concepts):
                sentence = self._fill_rule(pattern, rule.slots, concepts)
                if sentence:
                    return sentence

        # 2. 回退到概念关系组合
        if len(concepts) == 1:
            return f"{concepts[0]}"

        # 找到概念间的关系
        relations = self._find_relations(concepts)

        if relations:
            # 用关系词连接
            parts = [concepts[0]]
            for i in range(1, len(concepts)):
                rel = relations[i - 1] if i - 1 < len(relations) else '与'
                parts.append(rel)
                parts.append(concepts[i])
            return ''.join(parts)

        # 3. 最简回退：用"的"或"是"连接
        if len(concepts) == 2:
            return f"{concepts[0]}是{concepts[1]}"
        else:
            return '和'.join(concepts)

    def comprehend(self, text: str) -> Dict:
        """理解文本 — 通过概念激活而非正则匹配

        流程：
        1. 分词（基于已学标签 + 统计学习辅助）
        2. 激活相关概念（概念空间激活扩散）
        3. 理解关系（语法规则匹配 + 概念关系推理）
        4. 构建情境模型

        Args:
            text: 输入文本

        Returns:
            {'tokens': List, 'activated_concepts': List, 'relations': List,
             'meaning': str}
        """
        self._stats['comprehensions'] += 1

        # 1. 分词
        tokens = self._tokenize(text)

        # 2. 激活概念
        activated = []
        if self.concept_space:
            for token in tokens:
                if token in self.label_to_concept:
                    cid = self.label_to_concept[token]
                    activated.append(cid)
                elif token in self.concept_space.concepts:
                    activated.append(token)

            # 概念空间激活扩散
            if activated and self.concept_space:
                activation_results = self.concept_space.activate(text, top_k=10)
                for ac in activation_results[:5]:
                    if ac.concept_id not in activated:
                        activated.append(ac.concept_id)

        # 3. 理解关系
        relations = self._extract_relations_from_tokens(tokens, activated)

        # 4. 构建意义表示
        meaning = self._build_meaning(tokens, activated, relations)

        return {
            'tokens': tokens,
            'activated_concepts': activated,
            'relations': relations,
            'meaning': meaning,
        }

    def get_stats(self) -> Dict:
        """获取语言习得统计"""
        return {
            **self._stats,
            'label_mappings': len(self.label_to_concept),
            'grammar_rules': len(self.grammar_rules),
            'confirmed_rules': sum(1 for r in self.grammar_rules.values() if r.confirmed),
            'overregularizations': len(self.overregularization_log),
            'learned_patterns': len(self.learned_patterns),
        }

    # ===== 内部方法 =====

    def _extract_candidates(self, text: str) -> List[str]:
        """从文本中提取候选概念标签"""
        candidates = []

        # 方法1: 从统计学习者获取
        if self.statistical_learner:
            try:
                emergent = self.statistical_learner.get_emergent_concepts(min_freq=2)
                emergent_labels = {cid for cid, conf in emergent}
                # 检查文本中包含哪些已涌现概念
                for label in emergent_labels:
                    if label in text:
                        candidates.append(label)
            except Exception:
                pass

        # 方法2: 基于已学标签的分词
        for label in self.label_to_concept:
            if label in text and label not in candidates:
                candidates.append(label)

        # 方法3: 简单的2-4字窗口
        if not candidates:
            for length in range(4, 1, -1):  # 优先长词
                for i in range(len(text) - length + 1):
                    fragment = text[i:i + length]
                    # 过滤功能词开头的片段
                    if fragment[0] in self.FUNCTION_CHARS:
                        continue
                    if fragment[-1] in self.FUNCTION_CHARS:
                        continue
                    if fragment not in candidates:
                        candidates.append(fragment)

        return candidates[:20]  # 限制数量

    def _find_matching_concept(self, label: str) -> Optional[str]:
        """在概念空间中查找匹配的概念"""
        if not self.concept_space:
            return None

        if label in self.concept_space.concepts:
            return label

        if label in self.label_to_concept:
            return self.label_to_concept[label]

        return None

    def _extract_pattern(self, text: str, labels: List[str]) -> str:
        """从文本中提取模式（将概念替换为槽位）

        例如："数学是研究数量的学科" → "X是研究Y的Z"
        """
        pattern = text
        for i, label in enumerate(sorted(labels, key=len, reverse=True)):
            slot = f'SLOT{i}'
            pattern = pattern.replace(label, slot)
        return pattern

    def _identify_slots(self, pattern: str, variants: List[Dict]) -> List[str]:
        """识别规则中的槽位"""
        slots = []
        slot_match = re.findall(r'SLOT\d+', pattern)
        for slot in slot_match:
            # 从变体中收集槽位的填充词
            slot_idx = int(slot.replace('SLOT', ''))
            fillers = set()
            for v in variants:
                if slot_idx < len(v.get('labels', [])):
                    fillers.add(v['labels'][slot_idx])
            slots.append(f'SLOT{slot_idx}({"/".join(list(fillers)[:3])})')
        return slots

    def _check_overregularization(self, pattern: str, variants: List[Dict]) -> bool:
        """检查是否出现过度规则化

        过度规则化 = 规则被应用到不该应用的场景。
        在本系统中，如果同一个模式被用于完全不同类型的概念，
        说明规则可能被过度泛化了。
        """
        # 收集所有填充词
        all_fillers = set()
        for v in variants:
            all_fillers.update(v.get('labels', []))

        # 如果填充词的语义类型差异很大 → 可能是过度规则化
        if self.concept_space:
            types = set()
            for filler in all_fillers:
                if filler in self.concept_space.concepts:
                    node = self.concept_space.concepts[filler]
                    feat_type = node.perceptual_features.get('type', 'unknown')
                    types.add(feat_type)

            # 超过3种不同类型 → 可能过度规则化
            return len(types) > 3

        return False

    def _fill_rule(self, pattern: str, slots: List[str],
                    concepts: List[str]) -> Optional[str]:
        """用概念填充语法规则"""
        result = pattern
        for i, slot in enumerate(slots):
            if i < len(concepts):
                result = result.replace(slot, concepts[i])
        # 检查是否所有槽位都被填充
        if 'SLOT' not in result:
            return result
        return None

    def _find_relations(self, concepts: List[str]) -> List[str]:
        """找到概念间的关系词"""
        relations = []

        if self.concept_space:
            for i in range(len(concepts) - 1):
                c1 = concepts[i]
                c2 = concepts[i + 1]
                if c1 in self.concept_space.concepts and c2 in self.concept_space.concepts:
                    # 查找直接关系
                    rel_weight = self.concept_space.relations.get(c1, {}).get(c2, 0)
                    if rel_weight > 0.3:
                        relations.append('与')
                    else:
                        relations.append('的')

        if not relations:
            relations = ['和'] * (len(concepts) - 1)

        return relations

    def _tokenize(self, text: str) -> List[str]:
        """分词 — 基于已学标签和统计学习"""
        tokens = []

        # 优先匹配已学标签（从长到短）
        sorted_labels = sorted(self.label_to_concept.keys(), key=len, reverse=True)
        remaining = text

        while remaining:
            matched = False
            for label in sorted_labels:
                if remaining.startswith(label):
                    tokens.append(label)
                    remaining = remaining[len(label):]
                    matched = True
                    break

            if not matched:
                # 跳过功能词
                if remaining[0] in self.FUNCTION_CHARS:
                    remaining = remaining[1:]
                else:
                    # 取下一个字符作为临时token
                    tokens.append(remaining[0])
                    remaining = remaining[1:]

        return tokens

    def _extract_relations_from_tokens(self, tokens: List[str],
                                        concepts: List[str]) -> List[Dict]:
        """从token中提取关系"""
        relations = []

        for i in range(len(tokens) - 1):
            t1 = tokens[i]
            t2 = tokens[i + 1]

            # 功能词可能是关系标记
            if t1 in self.FUNCTION_CHARS or t2 in self.FUNCTION_CHARS:
                rel_type = self._classify_function_char(t1 if t1 in self.FUNCTION_CHARS else t2)
                if concepts and len(concepts) >= 2:
                    relations.append({
                        'type': rel_type,
                        'between': [concepts[0], concepts[-1]],
                    })

        return relations[:5]

    def _classify_function_char(self, char: str) -> str:
        """分类功能词"""
        if char in '是的':
            return 'is_a'
        elif char in '有包含':
            return 'has'
        elif char in '在':
            return 'location'
        elif char in '和与':
            return 'conjunction'
        elif char in '从到':
            return 'direction'
        elif char in '被把':
            return 'passive'
        else:
            return 'other'

    def _build_meaning(self, tokens: List[str], concepts: List[str],
                       relations: List[Dict]) -> str:
        """构建意义表示"""
        if not concepts:
            return f"未理解: {''.join(tokens)}"

        parts = [f"涉及概念: {', '.join(concepts[:5])}"]
        if relations:
            rel_strs = [f"{r['between'][0]}-{r['type']}->{r['between'][1]}"
                       for r in relations]
            parts.append(f"关系: {'; '.join(rel_strs)}")

        return '; '.join(parts)
