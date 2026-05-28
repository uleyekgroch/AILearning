"""
区域化语言社会：环境差异驱动方言分化

核心思想：
不同 agent 群体面对不同的环境分布（模拟地理隔离）。
- 区域 A 偏好 "red metal" 物体 → agent 倾向用 "red"/"metal" 描述
- 区域 B 偏好 "blue wood" 物体 → agent 倾向用 "blue"/"wood" 描述
- 同一物体，不同区域的 agent 可能发展出不同的描述策略

组合设计：
- RichScene（大概念空间） + NoisyListener（噪声反馈） + RegionalSociety（环境差异）
- 三者可独立或组合使用
"""

import random
import numpy as np
from typing import List, Dict, Optional, Set
from collections import defaultdict, deque

from language_emergence import (
    LanguageAgent, cross_language_round, compute_language_similarity,
    generate_rich_scene
)
from language_society_large import LargeScaleLanguageSociety
from language_rich_scene import (
    RegionConfig, generate_rich_scene_v2, generate_regional_scene,
    RICH_ATTRIBUTES, ALL_ATTRIBUTE_NAMES
)
from language_noisy import NoisyLanguageAgent, noisy_cross_language_round


class RegionalLanguageSociety(LargeScaleLanguageSociety):
    """
    区域化语言社会

    不同 agent 群体面对不同的环境分布。
    支持大概念空间 + 噪声反馈 + 环境差异。
    """

    def __init__(self, num_agents: int = 100,
                 num_regions: int = 5,
                 topology: str = 'small_world',
                 noise_rate: float = 0.15,
                 num_attributes: int = 4,
                 scene_size: int = 8,
                 history_maxlen: int = 10000):
        """
        Args:
            num_agents: agent 总数
            num_regions: 区域数
            topology: 网络拓扑
            noise_rate: Listener 噪声率
            num_attributes: 每个场景使用的属性维度数
            scene_size: 每个场景的物体数
        """
        self.num_regions = num_regions
        self.noise_rate = noise_rate
        self.num_attributes = num_attributes
        self.scene_size = scene_size

        # 创建区域配置
        self.regions = [RegionConfig(i) for i in range(num_regions)]

        # Agent 分配到区域
        self.agent_regions = {}
        for i in range(num_agents):
            self.agent_regions[i] = i % num_regions

        # 创建带噪声的 agents
        self.agents = [
            NoisyLanguageAgent(f"agent_{i}", noise_rate=noise_rate)
            for i in range(num_agents)
        ]
        self.agent_map = {a.id: a for a in self.agents}

        # 分组（按区域）
        groups = []
        for r in range(num_regions):
            group = [i for i in range(num_agents) if self.agent_regions[i] == r]
            groups.append(group)

        # 初始化父类的部分属性
        self.num_agents = num_agents
        self.topology = topology
        self.round_num = 0
        self.groups = groups
        self.adjacency = self._build_adjacency()
        self.history = deque(maxlen=history_maxlen)

        # 方言统计
        self._dialect_log = deque(maxlen=1000)

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """构建邻接关系（支持跨区域连接）"""
        n = self.num_agents
        ids = [a.id for a in self.agents]
        adj = {aid: [] for aid in ids}

        if self.topology == 'full':
            for i in range(n):
                for j in range(i + 1, n):
                    adj[ids[i]].append(ids[j])
                    adj[ids[j]].append(ids[i])

        elif self.topology == 'small_world':
            adj = self._build_small_world(ids, k=4, p=0.1)

        elif self.topology == 'scale_free':
            adj = self._build_scale_free(ids, m=3)

        elif self.topology == 'groups':
            for group in self.groups:
                for i in group:
                    for j in group:
                        if i != j:
                            adj[ids[i]].append(ids[j])

        elif self.topology == 'groups_with_bridges':
            # 组内全连接 + 少量跨组桥梁
            for group in self.groups:
                for i in group:
                    for j in group:
                        if i != j:
                            adj[ids[i]].append(ids[j])
            # 每组添加 2 个跨组桥梁
            for gi in range(len(self.groups)):
                for gj in range(gi + 1, len(self.groups)):
                    for _ in range(2):
                        i = random.choice(self.groups[gi])
                        j = random.choice(self.groups[gj])
                        if ids[j] not in adj[ids[i]]:
                            adj[ids[i]].append(ids[j])
                            adj[ids[j]].append(ids[i])

        return adj

    def _build_small_world(self, ids: List[str], k: int = 4,
                           p: float = 0.1) -> Dict[str, List[str]]:
        """小世界网络（复制自 LargeScaleLanguageSociety）"""
        n = len(ids)
        adj = {aid: [] for aid in ids}
        half_k = k // 2
        for i in range(n):
            for d in range(1, half_k + 1):
                left = (i - d) % n
                right = (i + d) % n
                if ids[left] not in adj[ids[i]]:
                    adj[ids[i]].append(ids[left])
                if ids[right] not in adj[ids[i]]:
                    adj[ids[i]].append(ids[right])
                if ids[i] not in adj[ids[left]]:
                    adj[ids[left]].append(ids[i])
                if ids[i] not in adj[ids[right]]:
                    adj[ids[right]].append(ids[i])
        for i in range(n):
            neighbors = list(adj[ids[i]])
            for nb in neighbors:
                if random.random() < p:
                    adj[ids[i]].remove(nb)
                    if ids[i] in adj[nb]:
                        adj[nb].remove(ids[i])
                    candidates = [j for j in range(n)
                                  if j != i and ids[j] not in adj[ids[i]]]
                    if candidates:
                        new_idx = random.choice(candidates)
                        adj[ids[i]].append(ids[new_idx])
                        adj[ids[new_idx]].append(ids[i])
        return adj

    def _build_scale_free(self, ids: List[str], m: int = 3) -> Dict[str, List[str]]:
        """无标度网络（复制自 LargeScaleLanguageSociety）"""
        n = len(ids)
        adj = {aid: [] for aid in ids}
        m0 = min(m + 1, n)
        for i in range(m0):
            for j in range(i + 1, m0):
                adj[ids[i]].append(ids[j])
                adj[ids[j]].append(ids[i])
        for i in range(m0, n):
            degrees = [len(adj[ids[j]]) for j in range(i)]
            total_degree = sum(degrees)
            if total_degree == 0:
                targets = random.sample(range(i), min(m, i))
            else:
                probs = [d / total_degree for d in degrees]
                targets = set()
                attempts = 0
                while len(targets) < min(m, i) and attempts < 100:
                    idx = np.random.choice(i, p=probs)
                    targets.add(idx)
                    attempts += 1
                targets = list(targets)
            for t in targets:
                adj[ids[i]].append(ids[t])
                adj[ids[t]].append(ids[i])
        return adj

    def step(self, scene: List[Dict[str, str]] = None,
             target_idx: int = None) -> bool:
        """
        一轮社会交流

        场景根据 speaker 所在区域生成。
        """
        a, b = self._random_adjacent_pair()

        if random.random() < 0.5:
            speaker, listener = a, b
        else:
            speaker, listener = b, a

        # 根据 speaker 所在区域生成场景
        speaker_idx = int(speaker.id.split('_')[1])
        region_id = self.agent_regions.get(speaker_idx, 0)
        region = self.regions[region_id]

        if scene is None:
            scene = region.generate_scene(
                self.scene_size, self.num_attributes
            )
        if target_idx is None:
            target_idx = random.randint(0, len(scene) - 1)

        # 跨语言交流（listener 使用 NoisyListener）
        success = noisy_cross_language_round(speaker, listener, scene, target_idx)

        self.history.append({
            'round': self.round_num,
            'speaker': speaker.id,
            'listener': listener.id,
            'speaker_region': region_id,
            'success': success,
        })
        self.round_num += 1

        return success

    def batch_step(self, scene=None, target_idx=None,
                   num_rounds: int = 10) -> float:
        """批量执行多轮通信"""
        successes = 0
        for _ in range(num_rounds):
            if self.step(scene, target_idx):
                successes += 1
        return successes / num_rounds

    def get_regional_metrics(self, sample_size: int = 100) -> Dict:
        """获取各区域的语言指标"""
        results = {}
        for r_id in range(self.num_regions):
            region_agents = [i for i in range(self.num_agents)
                             if self.agent_regions[i] == r_id]
            if len(region_agents) < 2:
                continue

            # 区域内相似度
            intra_sims = []
            pairs = set()
            while len(pairs) < min(sample_size, len(region_agents) * (len(region_agents) - 1) // 2):
                i, j = random.sample(region_agents, 2)
                pairs.add((min(i, j), max(i, j)))
            for i, j in pairs:
                sim = compute_language_similarity(
                    self.agents[i].language, self.agents[j].language
                )['overall_similarity']
                intra_sims.append(sim)

            results[f'region_{r_id}'] = {
                'num_agents': len(region_agents),
                'avg_intra_similarity': np.mean(intra_sims) if intra_sims else 0.0,
            }

        return results

    def get_cross_region_metrics(self, sample_size: int = 200) -> Dict:
        """获取跨区域的语言指标"""
        inter_sims = []
        pairs = set()
        while len(pairs) < sample_size:
            i = random.randint(0, self.num_agents - 1)
            j = random.randint(0, self.num_agents - 1)
            if i != j and self.agent_regions[i] != self.agent_regions[j]:
                pairs.add((min(i, j), max(i, j)))
        for i, j in pairs:
            sim = compute_language_similarity(
                self.agents[i].language, self.agents[j].language
            )['overall_similarity']
            inter_sims.append(sim)

        return {
            'avg_cross_region_similarity': np.mean(inter_sims) if inter_sims else 0.0,
            'num_pairs': len(inter_sims),
        }

    def get_dialect_divergence(self, sample_size: int = 200) -> float:
        """
        方言分化程度

        = 1 - (跨区域相似度 / 区域内相似度)
        值越大表示方言差异越大
        """
        regional = self.get_regional_metrics(sample_size // self.num_regions)
        cross = self.get_cross_region_metrics(sample_size)

        avg_intra = np.mean([r['avg_intra_similarity'] for r in regional.values()])
        avg_inter = cross['avg_cross_region_similarity']

        if avg_intra > 0:
            return 1.0 - (avg_inter / avg_intra)
        return 0.0

    def get_vocabulary_diversity(self) -> Dict:
        """获取各区域的词汇多样性"""
        results = {}
        for r_id in range(self.num_regions):
            region_agents = [i for i in range(self.num_agents)
                             if self.agent_regions[i] == r_id]
            # 收集区域内所有使用的符号
            all_symbols = set()
            for idx in region_agents:
                all_symbols.update(self.agents[idx].language.vocabulary.keys())
            results[f'region_{r_id}'] = {
                'vocabulary_size': len(all_symbols),
                'symbols': sorted(all_symbols),
            }
        return results

    def merge_regions(self):
        """打破区域隔离，所有 agent 都能互相交流"""
        self.groups = None
        self.topology = 'full'
        self.adjacency = self._build_adjacency()
