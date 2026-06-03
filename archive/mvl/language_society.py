"""
多 Agent 语言社会：方言分化与语言融合

核心思想：
每个 agent 拥有独立的语言。隔离的群体发展出不同的方言。
当不同方言的 agent 相遇时，观察语言的趋同或差异。

理论基础：
- 语言漂移（Language Drift）：随机因素导致语言演化路径分化
- 社会网络拓扑：谁和谁交流决定语言的传播路径
- 方言形成：地理隔离 → 独立演化 → 词汇/语法差异
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations

from language_emergence import (
    LanguageAgent, cross_language_round, compute_language_similarity,
    generate_rich_scene, EmergingLanguage
)


class LanguageSociety:
    """
    多 agent 语言社会

    管理多个 LanguageAgent，支持不同的交流拓扑：
    - full: 全连接，每对 agent 都能交流
    - star: 星形，中心 agent 与所有 agent 交流
    - line: 线形，只与邻居交流
    - groups: 分组，组内交流但组间隔离
    """

    def __init__(self, num_agents: int, topology: str = 'full',
                 groups: Optional[List[List[int]]] = None):
        self.agents = [LanguageAgent(f"agent_{i}") for i in range(num_agents)]
        self.topology = topology
        self.history = []
        self.round_num = 0

        # 构建邻接关系
        if topology == 'groups' and groups is not None:
            self.groups = groups
        else:
            self.groups = None
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """根据拓扑构建邻接关系"""
        n = len(self.agents)
        ids = [a.id for a in self.agents]
        adj = {aid: [] for aid in ids}

        if self.topology == 'full':
            for i, j in combinations(range(n), 2):
                adj[ids[i]].append(ids[j])
                adj[ids[j]].append(ids[i])

        elif self.topology == 'star':
            center = ids[0]
            for i in range(1, n):
                adj[center].append(ids[i])
                adj[ids[i]].append(center)

        elif self.topology == 'line':
            for i in range(n - 1):
                adj[ids[i]].append(ids[i + 1])
                adj[ids[i + 1]].append(ids[i])

        elif self.topology == 'groups':
            if self.groups is None:
                # 默认分成两组
                mid = n // 2
                self.groups = [list(range(mid)), list(range(mid, n))]
            for group in self.groups:
                for i, j in combinations(group, 2):
                    adj[ids[i]].append(ids[j])
                    adj[ids[j]].append(ids[i])

        return adj

    def _get_agent_by_id(self, agent_id: str) -> LanguageAgent:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise ValueError(f"Agent {agent_id} not found")

    def _random_adjacent_pair(self) -> Tuple[LanguageAgent, LanguageAgent]:
        """随机选择一对相邻 agent"""
        # 随机选一个 agent，再随机选它的一个邻居
        agent = np.random.choice(self.agents)
        neighbors = self.adjacency[agent.id]
        if not neighbors:
            # 如果没有邻居，随机选另一个
            others = [a for a in self.agents if a.id != agent.id]
            neighbor = np.random.choice(others)
        else:
            neighbor_id = np.random.choice(neighbors)
            neighbor = self._get_agent_by_id(neighbor_id)
        return agent, neighbor

    def step(self, scene: List[Dict[str, str]], target_idx: int) -> bool:
        """
        一轮社会交流

        随机选择一对相邻 agent，进行跨语言交流。
        speaker 和 listener 角色随机分配。
        """
        a, b = self._random_adjacent_pair()

        # 随机决定谁说谁听
        if np.random.random() < 0.5:
            speaker, listener = a, b
        else:
            speaker, listener = b, a

        success = cross_language_round(speaker, listener, scene, target_idx)

        self.history.append({
            'round': self.round_num,
            'speaker': speaker.id,
            'listener': listener.id,
            'success': success,
        })
        self.round_num += 1

        return success

    def get_group_stats(self) -> Dict:
        """获取各组的统计信息"""
        if self.groups is None:
            return {}

        group_stats = {}
        for gi, group in enumerate(self.groups):
            agents = [self.agents[i] for i in group]
            group_stats[f'group_{gi}'] = {
                'agent_ids': [a.id for a in agents],
                'avg_vocab_size': np.mean([a.language.get_vocabulary_size() for a in agents]),
                'avg_success_rate': np.mean([
                    a.language.total_successes / max(1, a.language.total_games)
                    for a in agents
                ]),
            }
        return group_stats

    def get_dialect_metrics(self) -> Dict:
        """
        计算方言分化指标

        返回：
            intra_group_similarity: 组内平均综合相似度
            inter_group_similarity: 组间平均综合相似度
            dialect_divergence: 方言分化程度（1 - 组间/组内）
        """
        if self.groups is None:
            return self._get_global_metrics()

        # 组内相似度
        intra_sims = []
        for group in self.groups:
            for i, j in combinations(group, 2):
                sim = compute_language_similarity(
                    self.agents[i].language, self.agents[j].language
                )
                intra_sims.append(sim['overall_similarity'])

        # 组间相似度
        inter_sims = []
        for gi, gj in combinations(range(len(self.groups)), 2):
            for i in self.groups[gi]:
                for j in self.groups[gj]:
                    sim = compute_language_similarity(
                        self.agents[i].language, self.agents[j].language
                    )
                    inter_sims.append(sim['overall_similarity'])

        avg_intra = np.mean(intra_sims) if intra_sims else 0.0
        avg_inter = np.mean(inter_sims) if inter_sims else 0.0

        return {
            'intra_group_similarity': avg_intra,
            'inter_group_similarity': avg_inter,
            'dialect_divergence': 1.0 - (avg_inter / max(0.001, avg_intra)),
            'num_intra_pairs': len(intra_sims),
            'num_inter_pairs': len(inter_sims),
        }

    def _get_global_metrics(self) -> Dict:
        """全局方言指标（无分组）"""
        sims = []
        for i, j in combinations(range(len(self.agents)), 2):
            sim = compute_language_similarity(
                self.agents[i].language, self.agents[j].language
            )
            sims.append(sim)

        if not sims:
            return {'avg_overall_similarity': 0.0}

        return {
            'avg_overall_similarity': np.mean([s['overall_similarity'] for s in sims]),
            'avg_vocab_freq_similarity': np.mean([s['vocab_freq_similarity'] for s in sims]),
            'avg_collocation_freq_similarity': np.mean([s['collocation_freq_similarity'] for s in sims]),
            'avg_order_match': np.mean([s['order_match'] for s in sims]),
        }

    def get_all_agent_stats(self) -> List[Dict]:
        """获取所有 agent 的统计"""
        return [a.get_stats() for a in self.agents]

    def merge_groups(self):
        """打破组间隔离，所有 agent 都能互相交流"""
        self.groups = None
        self.topology = 'full'
        self.adjacency = self._build_adjacency()

    def get_communication_success_rate(self, window: int = 100) -> float:
        """最近 window 轮的交流成功率"""
        if not self.history:
            return 0.0
        recent = self.history[-window:]
        return sum(1 for h in recent if h['success']) / len(recent)

    def get_cross_group_success_rate(self, window: int = 100) -> float:
        """跨组交流的成功率"""
        if not self.groups or not self.history:
            return 0.0

        # 建立 agent_id -> group_id 映射
        agent_to_group = {}
        for gi, group in enumerate(self.groups):
            for i in group:
                agent_to_group[self.agents[i].id] = gi

        recent = self.history[-window:]
        cross = [h for h in recent
                 if agent_to_group.get(h['speaker']) != agent_to_group.get(h['listener'])]
        if not cross:
            return 0.0
        return sum(1 for h in cross if h['success']) / len(cross)
