"""
Phase 21: 抽象推理模块 —— 类比、隐喻、概念迁移

核心思想：
抽象 = 跨领域的结构映射
- "A 像 B" 意味着 A 和 B 共享关系结构，尽管特征不同
- 类比是从具体到抽象的桥梁

涌现条件：
1. Listener 不理解目标领域的特征（特征对其不透明）
2. 直接描述无法帮助 Listener 识别目标
3. 类比通过关系结构（而非特征）建立理解桥梁
4. "like" 从需要跨域描述时涌现

关键设计：
- 不同领域的特征值完全不同（特征隔离）
- 不同领域共享关系类型（如 "eats"/"used_for" 都是 "acts_on"）
- 类比通过关系结构帮助 Listener 识别目标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    ABSTRACT_MARKERS,
)


# ============================================================
# 通用关系类型（跨领域共享）
# 这些关系类型是类比的基础——不同领域的概念共享关系结构
# ============================================================

UNIVERSAL_RELATIONS = {
    'acts_on',      # 作用于（eats→food, used_for→task, carries→passenger）
    'avoided_by',   # 被...回避（runs_from→predator, moves_on→obstacle）
    'located_in',   # 位于（lives_in→habitat, part_of→system）
}


class Concept:
    """
    概念：具有表面特征和关系结构的实体

    表面特征：颜色、形状、大小等（领域特有）
    关系结构：与其他概念的关系（跨领域共享类型）
    领域：概念所属的领域
    """

    def __init__(self, name: str, features: Dict[str, str],
                 relations: Set[Tuple[str, str]], domain: str):
        self.name = name
        self.features = features
        self.relations = relations  # (relation_type, target_concept)
        self.domain = domain

    def structural_signature(self) -> frozenset:
        """获取结构签名（用于比较结构相似性）"""
        return frozenset(self.relations)

    def relation_types(self) -> Set[str]:
        """获取关系类型集合"""
        return {r[0] for r in self.relations}

    def to_symbols(self) -> List[str]:
        """转换为符号列表"""
        return list(self.features.values())

    def __repr__(self):
        return f"Concept({self.name}, {self.domain})"


class AnalogyMapping:
    """
    类比映射：源概念到目标概念的结构对齐

    表达方式：[源特征] like [目标特征]
    Listener 理解源特征（已知领域），通过 "like" 建立关系映射
    """

    def __init__(self, source: Concept, target: Concept):
        self.source = source
        self.target = target
        self.structural_similarity: float = 0.0

    def compute_similarity(self):
        """计算结构相似度（基于通用关系类型）"""
        source_types = to_universal_relations(self.source.relations)
        target_types = to_universal_relations(self.target.relations)

        if not source_types or not target_types:
            self.structural_similarity = 0.0
            return

        intersection = len(source_types & target_types)
        union = len(source_types | target_types)
        self.structural_similarity = intersection / union if union > 0 else 0.0

    def express(self) -> List[str]:
        """生成类比描述：[源特征] like [目标特征]"""
        symbols = []
        symbols.extend(self.source.to_symbols())
        symbols.append('like')
        symbols.extend(self.target.to_symbols())
        return symbols

    def __repr__(self):
        return f"Analogy({self.source.name} -> {self.target.name}, sim={self.structural_similarity:.2f})"


class AbstractionModule:
    """抽象推理模块"""

    def __init__(self):
        self.known_concepts: List[Concept] = []
        self.analogy_history: List[AnalogyMapping] = []

    def add_concept(self, concept: Concept):
        self.known_concepts.append(concept)

    def find_analogy(self, source: Concept, target_domain: str) -> Optional[AnalogyMapping]:
        """在目标领域中寻找与源概念结构相似的概念"""
        candidates = [c for c in self.known_concepts if c.domain == target_domain]
        if not candidates:
            return None

        best_mapping = None
        best_similarity = 0.0

        for target in candidates:
            mapping = AnalogyMapping(source, target)
            mapping.compute_similarity()
            if mapping.structural_similarity > best_similarity:
                best_similarity = mapping.structural_similarity
                best_mapping = mapping

        if best_mapping and best_mapping.structural_similarity > 0.3:
            self.analogy_history.append(best_mapping)
            return best_mapping
        return None

    def detect_structural_similarity(self, concept_a: Concept, concept_b: Concept) -> float:
        mapping = AnalogyMapping(concept_a, concept_b)
        mapping.compute_similarity()
        return mapping.structural_similarity

    def transfer_knowledge(self, source: Concept, target: Concept) -> Dict[str, str]:
        """从源概念迁移知识到目标概念"""
        transferred = {}
        for (rel_type, source_target) in source.relations:
            for (target_rel_type, target_target) in target.relations:
                if rel_type == target_rel_type:
                    for feat_name, feat_val in source.features.items():
                        if feat_name not in target.features:
                            transferred[feat_name] = feat_val
        return transferred

    def get_stats(self) -> Dict:
        return {
            'known_concepts': len(self.known_concepts),
            'analogies_made': len(self.analogy_history),
            'avg_similarity': np.mean([m.structural_similarity for m in self.analogy_history]) if self.analogy_history else 0.0,
        }


# ============================================================
# 领域特征注册表
# 每个领域的特征值是该领域独有的符号
# ============================================================

DOMAIN_FEATURES = {
    'animals': {'color': ['brown', 'white', 'gray'],
                'shape': ['small', 'big', 'medium'],
                'size': ['fast', 'slow', 'quick']},
    'tools': {'color': ['metal', 'wood', 'plastic'],
              'shape': ['long', 'short', 'round'],
              'size': ['hard', 'soft', 'sharp']},
    'vehicles': {'color': ['red', 'blue', 'green'],
                 'shape': ['huge', 'tiny', 'sleek'],
                 'size': ['speedy', 'steady', 'swift']},
}

DOMAIN_SYMBOL_SETS = {}
for domain, features in DOMAIN_FEATURES.items():
    symbols = set()
    for vals in features.values():
        symbols.update(vals)
    DOMAIN_SYMBOL_SETS[domain] = symbols


# ============================================================
# 通用关系映射
# 将领域特有关系类型映射到通用关系类型
# 这是类比的基础——不同领域的概念共享通用关系
# ============================================================

RELATION_TO_UNIVERSAL = {
    'eats': 'acts_on',
    'used_for': 'acts_on',
    'carries': 'acts_on',
    'runs_from': 'avoided_by',
    'moves_on': 'avoided_by',
    'needs': 'avoided_by',
    'lives_in': 'located_in',
    'part_of': 'located_in',
    'made_of': 'located_in',
}


def to_universal_relations(relations: Set[Tuple[str, str]]) -> Set[str]:
    """将领域关系转换为通用关系类型"""
    universal = set()
    for (rel_type, target) in relations:
        if rel_type in RELATION_TO_UNIVERSAL:
            universal.add(RELATION_TO_UNIVERSAL[rel_type])
        else:
            universal.add(rel_type)
    return universal


def generate_abstraction_scenario(mode: str = 'concrete',
                                   num_concepts: int = 6) -> Tuple[List[Concept], int, Optional[str]]:
    """
    生成抽象推理场景

    参数：
        mode: 场景模式
            - 'concrete': 所有概念在同一领域（不需要类比）
            - 'cross_domain': 目标在不熟悉领域，源在熟悉领域
            - 'structural_match': 两个概念有相同关系但不同特征
            - 'metaphor': 共享抽象属性的概念
    """
    animal_features = [
        {'color': 'brown', 'shape': 'small', 'size': 'fast'},
        {'color': 'white', 'shape': 'big', 'size': 'slow'},
        {'color': 'gray', 'shape': 'medium', 'size': 'quick'},
    ]
    tool_features = [
        {'color': 'metal', 'shape': 'long', 'size': 'hard'},
        {'color': 'wood', 'shape': 'short', 'size': 'soft'},
        {'color': 'plastic', 'shape': 'round', 'size': 'sharp'},
    ]
    vehicle_features = [
        {'color': 'red', 'shape': 'huge', 'size': 'speedy'},
        {'color': 'blue', 'shape': 'tiny', 'size': 'steady'},
        {'color': 'green', 'shape': 'sleek', 'size': 'swift'},
    ]

    # 关系结构（使用通用关系类型名称）
    animal_relations = {
        ('eats', 'food'), ('runs_from', 'predator'), ('lives_in', 'habitat')
    }
    tool_relations = {
        ('used_for', 'task'), ('made_of', 'material'), ('part_of', 'system')
    }
    vehicle_relations = {
        ('carries', 'passenger'), ('moves_on', 'road'), ('needs', 'fuel')
    }

    concepts = []

    for i, features in enumerate(animal_features):
        name = f"animal_{i}"
        relations = {(r[0], f"{r[1]}_{i}") for r in animal_relations}
        concepts.append(Concept(name, features, relations, 'animals'))

    for i, features in enumerate(tool_features):
        name = f"tool_{i}"
        relations = {(r[0], f"{r[1]}_{i}") for r in tool_relations}
        concepts.append(Concept(name, features, relations, 'tools'))

    for i, features in enumerate(vehicle_features):
        name = f"vehicle_{i}"
        relations = {(r[0], f"{r[1]}_{i}") for r in vehicle_relations}
        concepts.append(Concept(name, features, relations, 'vehicles'))

    if mode == 'concrete':
        target_idx = np.random.randint(0, min(3, len(concepts)))
        unfamiliar_domain = None

    elif mode == 'cross_domain':
        unfamiliar_domain = 'tools'
        tool_indices = [i for i, c in enumerate(concepts) if c.domain == unfamiliar_domain]
        target_idx = np.random.choice(tool_indices)

    elif mode == 'structural_match':
        target_idx = np.random.randint(3, 6)
        unfamiliar_domain = None

    elif mode == 'metaphor':
        fast_animals = [i for i, c in enumerate(concepts) if c.domain == 'animals' and c.features.get('size') == 'fast']
        fast_vehicles = [i for i, c in enumerate(concepts) if c.domain == 'vehicles' and c.features.get('size') == 'speedy']
        if fast_animals and fast_vehicles:
            target_idx = np.random.choice(fast_vehicles)
            unfamiliar_domain = 'vehicles'
        else:
            target_idx = 0
            unfamiliar_domain = None
    else:
        target_idx = 0
        unfamiliar_domain = None

    return concepts, target_idx, unfamiliar_domain


class AbstractCommunicationGame:
    """
    抽象交流游戏

    关键设计：
    1. 特征隔离：不同领域的特征值完全不同
    2. 关系共享：不同领域共享通用关系类型（acts_on, avoided_by, located_in）
    3. 匹配机制：
       - 直接描述：Listener 只理解已知领域的特征 → 无法匹配未知领域目标
       - 类比描述：Listener 通过关系结构匹配 → 可以识别未知领域目标

    类比如何工作：
    Speaker 说 "[animal_0] like [tool_0]"
    Listener 理解 animal_0（已知领域），发现它有 "acts_on" 关系
    Listener 寻找也有 "acts_on" 关系的概念 → 找到 tool_0
    """

    def __init__(self, known_domains: Set[str] = None):
        self.language = EmergingLanguage()
        self.abstraction = AbstractionModule()
        self.known_domains = known_domains or {'animals'}
        self.game_log = []
        self.analogy_used = 0
        self.analogy_success = 0
        self.direct_used = 0
        self.direct_success = 0

    def play_round(self, concepts: List[Concept],
                   target_idx: int,
                   unfamiliar_domain: str = None) -> bool:
        if target_idx >= len(concepts):
            return False

        target = concepts[target_idx]

        # 判断是否需要使用类比
        use_analogy = (unfamiliar_domain is not None
                       and target.domain == unfamiliar_domain
                       and target.domain not in self.known_domains)

        if use_analogy:
            # 寻找已知领域中的类比源
            source_concepts = [c for c in concepts if c.domain in self.known_domains]
            if source_concepts:
                best_source = None
                best_sim = -1
                for sc in source_concepts:
                    sim = self.abstraction.detect_structural_similarity(sc, target)
                    if sim > best_sim:
                        best_sim = sim
                        best_source = sc
                if best_source:
                    mapping = AnalogyMapping(best_source, target)
                    mapping.compute_similarity()
                    utterance = mapping.express()
                    self.analogy_used += 1
                else:
                    utterance = target.to_symbols()
                    self.direct_used += 1
            else:
                utterance = target.to_symbols()
                self.direct_used += 1
        else:
            utterance = target.to_symbols()
            self.direct_used += 1

        if not utterance:
            return False

        chosen_idx = self._listener_interpret(utterance, concepts)
        success = (chosen_idx == target_idx)

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1

        if use_analogy and success:
            self.analogy_success += 1
        elif not use_analogy and success:
            self.direct_success += 1

        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
            'used_analogy': use_analogy,
            'target_domain': target.domain,
        })
        return success

    def _listener_interpret(self, utterance: List[str],
                            concepts: List[Concept]) -> int:
        """
        Listener 解释描述

        匹配机制：
        - 无 "like"：直接特征匹配（只计算已知领域的特征）
        - 有 "like"：关系结构匹配（通过通用关系类型匹配目标）
        """
        # 收集 Listener 理解的符号
        understood_symbols = set()
        for domain in self.known_domains:
            if domain in DOMAIN_SYMBOL_SETS:
                understood_symbols.update(DOMAIN_SYMBOL_SETS[domain])

        if 'like' in utterance:
            return self._interpret_analogy(utterance, concepts, understood_symbols)
        else:
            return self._interpret_direct(utterance, concepts, understood_symbols)

    def _interpret_direct(self, utterance: List[str],
                          concepts: List[Concept],
                          understood_symbols: Set[str]) -> int:
        """直接描述匹配：只计算 Listener 理解的符号"""
        scores = []
        for i, concept in enumerate(concepts):
            concept_values = set(concept.to_symbols())
            matches = sum(1 for s in utterance
                          if s in concept_values and s in understood_symbols)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _interpret_analogy(self, utterance: List[str],
                           concepts: List[Concept],
                           understood_symbols: Set[str]) -> int:
        """
        类比描述匹配：通过关系结构匹配

        算法：
        1. 分离 "like" 前后的符号
        2. before_like 匹配已知领域的源概念
        3. 源概念的通用关系类型作为目标特征
        4. 寻找拥有相同通用关系类型的概念
        """
        like_idx = utterance.index('like')
        before_like = utterance[:like_idx]
        after_like = utterance[like_idx + 1:]

        # Step 1: 找到 before_like 匹配的源概念
        source_concept = None
        best_source_score = -1
        for concept in concepts:
            if concept.domain not in self.known_domains:
                continue
            concept_values = set(concept.to_symbols())
            matches = sum(1 for s in before_like if s in concept_values)
            if matches > best_source_score:
                best_source_score = matches
                source_concept = concept

        if source_concept is None:
            # 退化为直接匹配
            return self._interpret_direct(utterance, concepts, understood_symbols)

        # Step 2: 获取源概念的通用关系类型
        source_universal = to_universal_relations(source_concept.relations)

        # Step 3: 对每个概念评分
        scores = []
        for i, concept in enumerate(concepts):
            if concept.domain in self.known_domains:
                # 已知领域：直接特征匹配
                concept_values = set(concept.to_symbols())
                matches = sum(1 for s in before_like if s in concept_values)
            else:
                # 未知领域：关系结构匹配
                concept_universal = to_universal_relations(concept.relations)
                shared_relations = len(source_universal & concept_universal)
                # 关系匹配得分（通常 1-3 分）
                matches = shared_relations * 2
                # 如果 after_like 匹配，加分
                concept_values = set(concept.to_symbols())
                after_matches = sum(1 for s in after_like if s in concept_values)
                matches += after_matches

            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['analogy_used'] = self.analogy_used
        stats['analogy_success'] = self.analogy_success / max(1, self.analogy_used)
        stats['direct_used'] = self.direct_used
        stats['direct_success'] = self.direct_success / max(1, self.direct_used)
        stats['abstraction_stats'] = self.abstraction.get_stats()
        return stats


class BaselineAbstractionGame:
    """无抽象推理的基线游戏（Listener 理解所有领域，不需要类比）"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, concepts: List[Concept],
                   target_idx: int,
                   unfamiliar_domain: str = None) -> bool:
        if target_idx >= len(concepts):
            return False

        target = concepts[target_idx]
        utterance = target.to_symbols()

        scores = []
        for i, concept in enumerate(concepts):
            concept_values = set(concept.to_symbols())
            matches = sum(1 for s in utterance if s in concept_values)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        chosen_idx = scores[0][0]
        success = (chosen_idx == target_idx)

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success

    def get_stats(self) -> Dict:
        return self.language.get_stats()


def test_abstraction():
    """测试抽象推理机制"""
    print("=== 抽象推理机制测试 ===")

    animal = Concept("cat", {'color': 'brown', 'shape': 'small'},
                     {('eats', 'food'), ('runs_from', 'predator')}, 'animals')
    tool = Concept("hammer", {'color': 'metal', 'shape': 'long'},
                   {('used_for', 'nail'), ('made_of', 'steel')}, 'tools')

    print(f"\n动物概念: {animal}")
    print(f"  特征: {animal.features}")
    print(f"  关系: {animal.relations}")
    print(f"  通用关系: {to_universal_relations(animal.relations)}")

    print(f"\n工具概念: {tool}")
    print(f"  特征: {tool.features}")
    print(f"  关系: {tool.relations}")
    print(f"  通用关系: {to_universal_relations(tool.relations)}")

    mapping = AnalogyMapping(animal, tool)
    mapping.compute_similarity()
    print(f"\n类比映射: {mapping}")
    print(f"  结构相似度: {mapping.structural_similarity:.2f}")
    print(f"  类比描述: {mapping.express()}")

    # 测试 Listener 理解障碍
    print(f"\n--- Listener 理解障碍测试 ---")
    game = AbstractCommunicationGame(known_domains={'animals'})

    print(f"\n直接描述工具 'metal long hard':")
    print(f"  Listener 理解的符号: {DOMAIN_SYMBOL_SETS['animals']}")
    print(f"  工具符号: {tool.to_symbols()}")
    print(f"  交集: {set(tool.to_symbols()) & DOMAIN_SYMBOL_SETS['animals']}")
    print(f"  → Listener 无法匹配任何工具!")

    print(f"\n类比描述 'brown small fast like metal long hard':")
    print(f"  before_like: ['brown', 'small', 'fast'] → 匹配 animal_0")
    print(f"  animal_0 通用关系: {to_universal_relations(animal.relations)}")
    print(f"  tool_0 通用关系: {to_universal_relations(tool.relations)}")
    print(f"  共享通用关系: {to_universal_relations(animal.relations) & to_universal_relations(tool.relations)}")
    print(f"  → Listener 通过关系匹配找到 tool_0!")


if __name__ == '__main__':
    test_abstraction()
