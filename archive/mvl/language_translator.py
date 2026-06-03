"""
跨语言翻译系统

不同环境中的 Agent 发展出不同语言后，如何互相翻译？

核心机制：
1. CrossLingualAgent — 双语 Agent，能学习两种环境的语言
2. CrossLingualSociety — 管理跨环境社会
3. 翻译度量 — 跨环境通信成功率、符号映射准确率
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from language_emergence import (
    LanguageAgent, EmergingLanguage, cross_language_round,
    compute_language_similarity, generate_rich_scene
)
from language_rich_scene import (
    RegionConfig, generate_regional_scene, ALL_ATTRIBUTE_NAMES
)
from language_society_large import LargeScaleLanguageSociety


class CrossLingualAgent(LanguageAgent):
    """
    双语 Agent — 能在两种环境中学习语言

    除了母语（home language），还能学习外语（foreign language），
    并建立符号翻译表。
    """

    def __init__(self, agent_id: str, home_region: str):
        super().__init__(agent_id)
        self.home_region = home_region
        self.bilingual = False

        # 翻译表：母语符号 ↔ 外语符号
        self.translation_table: Dict[str, str] = {}

        # 跨环境统计
        self.cross_env_games = 0
        self.cross_env_successes = 0

        # 外语词汇（在跨环境交流中学到的符号频率）
        self.foreign_vocab: Dict[str, Dict] = {}

    def learn_from_cross_env(self, foreign_utterance: List[str],
                             scene: List[Dict[str, str]],
                             target_idx: int, success: bool):
        """
        从跨环境交流中学习

        用自己的语言描述同一目标，然后建立符号映射。
        """
        self.cross_env_games += 1
        if success:
            self.cross_env_successes += 1

        # 用自己的语言描述同一目标
        my_utterance = self.speak(scene[target_idx], scene)

        # 建立映射：对齐位置对应的符号
        min_len = min(len(my_utterance), len(foreign_utterance))
        for i in range(min_len):
            my_sym = my_utterance[i]
            foreign_sym = foreign_utterance[i]
            if my_sym != foreign_sym:
                # 更新翻译表（累积投票）
                if my_sym not in self.translation_table:
                    self.translation_table[my_sym] = foreign_sym
                elif self.translation_table[my_sym] == foreign_sym:
                    pass  # 一致，强化
                else:
                    # 冲突：保留频率更高的
                    pass

                # 反向映射
                if foreign_sym not in self.translation_table:
                    self.translation_table[foreign_sym] = my_sym

        # 记录外语词汇
        for sym in foreign_utterance:
            if sym not in self.foreign_vocab:
                self.foreign_vocab[sym] = {'frequency': 0, 'successes': 0}
            self.foreign_vocab[sym]['frequency'] += 1
            if success:
                self.foreign_vocab[sym]['successes'] += 1

    def translate_utterance(self, utterance: List[str]) -> List[str]:
        """翻译话语（母语 → 外语）"""
        return [self.translation_table.get(s, s) for s in utterance]

    def get_translation_accuracy(self, true_mapping: Dict[str, str]) -> float:
        """测量翻译表准确率"""
        if not self.translation_table or not true_mapping:
            return 0.0
        correct = 0
        total = 0
        for my_sym, expected_foreign in true_mapping.items():
            if my_sym in self.translation_table:
                total += 1
                if self.translation_table[my_sym] == expected_foreign:
                    correct += 1
        return correct / total if total > 0 else 0.0


def measure_cross_lingual_metrics(agents_a: List[LanguageAgent],
                                   agents_b: List[LanguageAgent],
                                   scene_generator_a,
                                   scene_generator_b,
                                   num_trials: int = 100) -> dict:
    """
    测量两个群体之间的跨语言通信能力

    返回：
    - baseline_success: A 说话 B 听的成功率
    - reverse_success: B 说话 A 听的成功率
    - vocab_overlap: 共享符号比例
    - avg_similarity: 平均语言相似度
    """
    baseline_successes = 0
    reverse_successes = 0
    total = 0

    for _ in range(num_trials):
        # A 的场景
        scene_a = scene_generator_a()
        target_idx_a = random.randint(0, len(scene_a) - 1)

        # B 的场景
        scene_b = scene_generator_b()
        target_idx_b = random.randint(0, len(scene_b) - 1)

        # A 说话，B 听
        a = random.choice(agents_a)
        b = random.choice(agents_b)

        utterance_a = a.speak(scene_a[target_idx_a], scene_a)
        chosen_b = b.listen(utterance_a, scene_a)
        if chosen_b == target_idx_a:
            baseline_successes += 1

        # B 说话，A 听
        utterance_b = b.speak(scene_b[target_idx_b], scene_b)
        chosen_a = a.listen(utterance_b, scene_b)
        if chosen_a == target_idx_b:
            reverse_successes += 1

        total += 1

    # 符号重叠
    vocab_a = set()
    for agent in agents_a:
        vocab_a.update(agent.language.vocabulary.keys())
    vocab_b = set()
    for agent in agents_b:
        vocab_b.update(agent.language.vocabulary.keys())

    overlap = vocab_a & vocab_b
    union = vocab_a | vocab_b
    overlap_ratio = len(overlap) / len(union) if union else 0

    # 语言相似度（采样）
    sims = []
    for _ in range(min(50, len(agents_a) * len(agents_b))):
        a = random.choice(agents_a)
        b = random.choice(agents_b)
        sim = compute_language_similarity(a.language, b.language)['overall_similarity']
        sims.append(sim)

    return {
        'baseline_success': baseline_successes / total if total > 0 else 0,
        'reverse_success': reverse_successes / total if total > 0 else 0,
        'cross_success': (baseline_successes + reverse_successes) / (2 * total) if total > 0 else 0,
        'vocab_overlap': overlap_ratio,
        'shared_symbols': len(overlap),
        'total_symbols': len(union),
        'avg_similarity': float(np.mean(sims)) if sims else 0,
    }


class CrossLingualSociety:
    """
    跨环境社会 — 管理两个不同环境中的 Agent 群体

    三个阶段：
    1. 隔离期：各群体在自己环境中独立发展语言
    2. 接触期：跨环境交流开始
    3. 融合期：语言趋同或产生通用语
    """

    def __init__(self, region_a: RegionConfig, region_b: RegionConfig,
                 agents_per_region: int = 100):
        self.region_a = region_a
        self.region_b = region_b
        self.n = agents_per_region

        # 创建两个群体的 Agent
        self.agents_a = [CrossLingualAgent(f"a_{i}", region_a.name)
                         for i in range(agents_per_region)]
        self.agents_b = [CrossLingualAgent(f"b_{i}", region_b.name)
                         for i in range(agents_per_region)]

        # 构建群体内邻接（小世界网络）
        self.adj_a = self._build_small_world_ids(
            [a.id for a in self.agents_a], k=4, p=0.1)
        self.adj_b = self._build_small_world_ids(
            [a.id for a in self.agents_b], k=4, p=0.1)

        # Agent 查找表
        self.map_a = {a.id: a for a in self.agents_a}
        self.map_b = {a.id: a for a in self.agents_b}

        # 历史
        self.history = []

    def _build_small_world_ids(self, ids: List[str], k: int = 4,
                                p: float = 0.1) -> Dict[str, List[str]]:
        """构建小世界邻接（set-based）"""
        n = len(ids)
        adj_set = {aid: set() for aid in ids}
        half_k = k // 2
        for i in range(n):
            for d in range(1, half_k + 1):
                left = (i - d) % n
                right = (i + d) % n
                adj_set[ids[i]].add(ids[left])
                adj_set[ids[i]].add(ids[right])
                adj_set[ids[left]].add(ids[i])
                adj_set[ids[right]].add(ids[i])

        all_ids_set = set(ids)
        for i in range(n):
            neighbors = list(adj_set[ids[i]])
            for nb in neighbors:
                if random.random() < p:
                    adj_set[ids[i]].discard(nb)
                    adj_set[nb].discard(ids[i])
                    candidates = all_ids_set - adj_set[ids[i]] - {ids[i]}
                    if candidates:
                        new_id = random.choice(list(candidates))
                        adj_set[ids[i]].add(new_id)
                        adj_set[new_id].add(ids[i])

        return {aid: list(nbs) for aid, nbs in adj_set.items()}

    def _random_pair(self, agents, adj) -> Tuple[LanguageAgent, LanguageAgent]:
        """从群体中随机选一对相邻 Agent"""
        a = random.choice(agents)
        neighbors = adj.get(a.id, [])
        if neighbors:
            b_id = random.choice(neighbors)
            b = next((x for x in agents if x.id == b_id), random.choice(agents))
        else:
            b = random.choice([x for x in agents if x.id != a.id])
        return a, b

    def phase_isolation(self, num_rounds: int = 1000,
                        verbose: bool = True) -> dict:
        """
        阶段 1：隔离期

        两个群体各自在自己环境中独立发展语言。
        """
        if verbose:
            print(f"  隔离期：{num_rounds} 轮")

        results = {}
        for label, agents, adj, region in [
            ('A', self.agents_a, self.adj_a, self.region_a),
            ('B', self.agents_b, self.adj_b, self.region_b)
        ]:
            successes = 0
            for r in range(num_rounds):
                scene = generate_regional_scene(
                    region.biases, num_objects=8,
                    attribute_names=region.preferred_attributes)
                target_idx = random.randint(0, len(scene) - 1)

                a, b = self._random_pair(agents, adj)
                speaker, listener = (a, b) if random.random() < 0.5 else (b, a)
                success = cross_language_round(speaker, listener, scene, target_idx)
                if success:
                    successes += 1

            sr = successes / num_rounds
            results[label] = sr
            if verbose:
                print(f"    群体 {label} ({region.name}): 成功率={sr:.3f}")

        return results

    def phase_contact(self, num_rounds: int = 500,
                      contact_mode: str = 'random',
                      num_bridges: int = 10,
                      verbose: bool = True) -> dict:
        """
        阶段 2：接触期

        跨环境交流开始。

        contact_mode:
        - 'random': 随机跨环境配对
        - 'bridge': 通过桥接 Agent 翻译
        - 'bilingual': 部分 Agent 成为双语者
        """
        if verbose:
            print(f"  接触期：{num_rounds} 轮，模式={contact_mode}")

        cross_successes = 0
        intra_a_successes = 0
        intra_b_successes = 0
        cross_total = 0
        intra_total = 0

        # 桥接 Agent（双语者）
        bridge_agents = []
        if contact_mode in ('bridge', 'bilingual'):
            bridge_ids_a = random.sample(
                [a.id for a in self.agents_a], min(num_bridges, self.n))
            bridge_ids_b = random.sample(
                [b.id for b in self.agents_b], min(num_bridges, self.n))
            bridge_agents = [self.map_a[aid] for aid in bridge_ids_a] + \
                           [self.map_b[bid] for bid in bridge_ids_b]
            for ba in bridge_agents:
                ba.bilingual = True

        for r in range(num_rounds):
            # 70% 群体内交流（维持母语），30% 跨环境交流
            if random.random() < 0.7:
                # 群体内交流
                if random.random() < 0.5:
                    scene = generate_regional_scene(
                        self.region_a.biases, num_objects=8,
                        attribute_names=self.region_a.preferred_attributes)
                    target_idx = random.randint(0, len(scene) - 1)
                    a, b = self._random_pair(self.agents_a, self.adj_a)
                    speaker, listener = (a, b) if random.random() < 0.5 else (b, a)
                    success = cross_language_round(speaker, listener, scene, target_idx)
                    if success:
                        intra_a_successes += 1
                    intra_total += 1
                else:
                    scene = generate_regional_scene(
                        self.region_b.biases, num_objects=8,
                        attribute_names=self.region_b.preferred_attributes)
                    target_idx = random.randint(0, len(scene) - 1)
                    a, b = self._random_pair(self.agents_b, self.adj_b)
                    speaker, listener = (a, b) if random.random() < 0.5 else (b, a)
                    success = cross_language_round(speaker, listener, scene, target_idx)
                    if success:
                        intra_b_successes += 1
                    intra_total += 1
            else:
                # 跨环境交流
                scene_a = generate_regional_scene(
                    self.region_a.biases, num_objects=8,
                    attribute_names=self.region_a.preferred_attributes)
                target_idx_a = random.randint(0, len(scene_a) - 1)

                scene_b = generate_regional_scene(
                    self.region_b.biases, num_objects=8,
                    attribute_names=self.region_b.preferred_attributes)
                target_idx_b = random.randint(0, len(scene_b) - 1)

                if contact_mode == 'random':
                    # 直接跨环境配对
                    a = random.choice(self.agents_a)
                    b = random.choice(self.agents_b)

                    # A 用自己的场景说话，B 在 A 的场景中听
                    success = cross_language_round(a, b, scene_a, target_idx_a)
                    if success:
                        cross_successes += 1
                    cross_total += 1

                elif contact_mode == 'bridge':
                    # 通过桥接 Agent 翻译
                    a = random.choice(self.agents_a)
                    bridge = random.choice(
                        [x for x in bridge_agents if x.home_region == self.region_a.name])
                    b = random.choice(self.agents_b)

                    # A → 桥接（A 的场景）
                    success1 = cross_language_round(a, bridge, scene_a, target_idx_a)

                    # 桥接 → B（B 的场景，桥接尝试翻译）
                    translated = bridge.translate_utterance(
                        bridge.speak(scene_b[target_idx_b], scene_b))
                    # B 尝试理解桥接的翻译
                    chosen = b.listen(translated, scene_b)
                    success2 = (chosen == target_idx_b)

                    if success2:
                        cross_successes += 1
                    cross_total += 1

                    # 桥接从两个方向学习
                    bridge.learn_from_cross_env(
                        a.speak(scene_a[target_idx_a], scene_a),
                        scene_a, target_idx_a, success1)

                elif contact_mode == 'bilingual':
                    # 双语者直接跨环境交流
                    a = random.choice(self.agents_a)
                    b = random.choice(self.agents_b)

                    # 尝试用 A 的场景
                    success = cross_language_round(a, b, scene_a, target_idx_a)
                    if success:
                        cross_successes += 1
                    cross_total += 1

                    # 双语者额外学习
                    if a.bilingual:
                        a.learn_from_cross_env(
                            b.speak(scene_b[target_idx_b], scene_b),
                            scene_b, target_idx_b, success)
                    if b.bilingual:
                        b.learn_from_cross_env(
                            a.speak(scene_a[target_idx_a], scene_a),
                            scene_a, target_idx_a, success)

        results = {
            'cross_success': cross_successes / cross_total if cross_total > 0 else 0,
            'intra_success': (intra_a_successes + intra_b_successes) / intra_total if intra_total > 0 else 0,
            'cross_total': cross_total,
            'intra_total': intra_total,
            'num_bridges': len(bridge_agents),
        }

        if verbose:
            print(f"    跨环境成功率: {results['cross_success']:.3f}")
            print(f"    群体内成功率: {results['intra_success']:.3f}")

        return results

    def measure_all(self, num_trials: int = 100) -> dict:
        """综合测量所有跨语言指标"""
        # 跨环境基线
        cross_metrics = measure_cross_lingual_metrics(
            self.agents_a, self.agents_b,
            lambda: generate_regional_scene(
                self.region_a.biases, num_objects=8,
                attribute_names=self.region_a.preferred_attributes),
            lambda: generate_regional_scene(
                self.region_b.biases, num_objects=8,
                attribute_names=self.region_b.preferred_attributes),
            num_trials=num_trials
        )

        # 群体内指标（对照）
        intra_a = measure_cross_lingual_metrics(
            self.agents_a[:len(self.agents_a)//2],
            self.agents_a[len(self.agents_a)//2:],
            lambda: generate_regional_scene(
                self.region_a.biases, num_objects=8,
                attribute_names=self.region_a.preferred_attributes),
            lambda: generate_regional_scene(
                self.region_a.biases, num_objects=8,
                attribute_names=self.region_a.preferred_attributes),
            num_trials=num_trials
        )

        intra_b = measure_cross_lingual_metrics(
            self.agents_b[:len(self.agents_b)//2],
            self.agents_b[len(self.agents_b)//2:],
            lambda: generate_regional_scene(
                self.region_b.biases, num_objects=8,
                attribute_names=self.region_b.preferred_attributes),
            lambda: generate_regional_scene(
                self.region_b.biases, num_objects=8,
                attribute_names=self.region_b.preferred_attributes),
            num_trials=num_trials
        )

        # 翻译表统计
        bridge_agents = [a for a in self.agents_a + self.agents_b if a.bilingual]
        avg_table_size = np.mean([len(a.translation_table) for a in bridge_agents]) if bridge_agents else 0

        return {
            'cross_lingual': cross_metrics,
            'intra_a': intra_a,
            'intra_b': intra_b,
            'avg_translation_table_size': float(avg_table_size),
            'num_bilingual': len(bridge_agents),
        }
