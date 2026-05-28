"""
大规模语言社会：100+ Agent 的语言演化

在 LanguageSociety 基础上解决 5 个性能瓶颈：
1. 采样指标计算：O(sample_size) 代替 O(n²)
2. Agent ID 字典查找：O(1) 代替 O(n)
3. 窗口化历史：deque(maxlen) 代替无限增长
4. 新拓扑：小世界网络 + 无标度网络
5. 批量通信：一轮多步

新增能力：
- 语言家族检测（层次聚类）
- 通用语检测（跨群体通信成功率）
- 语言灭绝/复兴追踪
"""

import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from itertools import combinations

from language_emergence import (
    LanguageAgent, cross_language_round, compute_language_similarity,
    generate_rich_scene
)


class LargeScaleLanguageSociety:
    """
    大规模语言社会

    支持 100+ Agent 的语言演化模拟。
    通过采样指标、字典查找、窗口化历史解决性能瓶颈。
    支持小世界网络和无标度网络拓扑。
    """

    def __init__(self, num_agents: int = 100, topology: str = 'full',
                 groups: Optional[List[List[int]]] = None,
                 history_maxlen: int = 10000):
        self.num_agents = num_agents
        self.topology = topology
        self.round_num = 0

        # 创建 agents
        self.agents = [LanguageAgent(f"agent_{i}") for i in range(num_agents)]

        # O(1) agent 查找
        self.agent_map = {a.id: a for a in self.agents}

        # 分组
        self.groups = groups

        # 邻接关系
        self.adjacency = self._build_adjacency()

        # 窗口化历史
        self.history = deque(maxlen=history_maxlen)

        # 语言家族缓存
        self._family_cache = None
        self._family_cache_round = -1

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """根据拓扑构建邻接关系"""
        n = self.num_agents
        ids = [a.id for a in self.agents]
        adj = {aid: [] for aid in ids}

        if self.topology == 'full':
            # 全连接：O(n²) 边，只适合小规模
            for i in range(n):
                for j in range(i + 1, n):
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

        elif self.topology == 'small_world':
            adj = self._build_small_world(ids, k=4, p=0.1)

        elif self.topology == 'scale_free':
            adj = self._build_scale_free(ids, m=3)

        elif self.topology == 'groups' and self.groups is not None:
            for group in self.groups:
                for i in group:
                    for j in group:
                        if i != j:
                            adj[ids[i]].append(ids[j])

        return adj

    def _build_small_world(self, ids: List[str], k: int = 4,
                           p: float = 0.1) -> Dict[str, List[str]]:
        """
        Watts-Strogatz 小世界网络

        每个节点连 k 个近邻，以概率 p 随机重连。
        特征：高聚类系数 + 短平均路径长度。
        """
        n = len(ids)
        adj = {aid: [] for aid in ids}

        # 第一步：环形网格，每个节点连 k/2 左右邻居
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

        # 第二步：以概率 p 随机重连
        for i in range(n):
            neighbors = list(adj[ids[i]])
            for nb in neighbors:
                if random.random() < p:
                    # 移除旧边
                    adj[ids[i]].remove(nb)
                    if ids[i] in adj[nb]:
                        adj[nb].remove(ids[i])
                    # 选新目标（排除自身和已有邻居）
                    candidates = [j for j in range(n)
                                  if j != i and ids[j] not in adj[ids[i]]]
                    if candidates:
                        new_idx = random.choice(candidates)
                        adj[ids[i]].append(ids[new_idx])
                        adj[ids[new_idx]].append(ids[i])

        return adj

    def _build_scale_free(self, ids: List[str],
                          m: int = 3) -> Dict[str, List[str]]:
        """
        Barabási-Albert 无标度网络

        优先连接模型：新节点倾向连接高度节点。
        特征：少数枢纽节点有大量连接，多数节点连接很少。
        """
        n = len(ids)
        adj = {aid: [] for aid in ids}

        # 初始完全图（m0 个节点）
        m0 = min(m + 1, n)
        for i in range(m0):
            for j in range(i + 1, m0):
                adj[ids[i]].append(ids[j])
                adj[ids[j]].append(ids[i])

        # 逐步添加新节点
        for i in range(m0, n):
            # 计算度数权重
            degrees = [len(adj[ids[j]]) for j in range(i)]
            total_degree = sum(degrees)
            if total_degree == 0:
                # 均匀选择
                targets = random.sample(range(i), min(m, i))
            else:
                probs = [d / total_degree for d in degrees]
                # 按概率选择 m 个不同目标
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

    def _random_adjacent_pair(self) -> Tuple[LanguageAgent, LanguageAgent]:
        """随机选择一对相邻 agent"""
        agent = random.choice(self.agents)
        neighbors = self.adjacency[agent.id]
        if not neighbors:
            others = [a for a in self.agents if a.id != agent.id]
            neighbor = random.choice(others)
        else:
            neighbor_id = random.choice(neighbors)
            neighbor = self.agent_map[neighbor_id]
        return agent, neighbor

    def step(self, scene: List[Dict[str, str]], target_idx: int) -> bool:
        """一轮社会交流"""
        a, b = self._random_adjacent_pair()

        if random.random() < 0.5:
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

    def batch_step(self, scene: List[Dict[str, str]], target_idx: int,
                   num_rounds: int = 10) -> float:
        """批量执行多轮通信，返回成功率"""
        successes = 0
        for _ in range(num_rounds):
            if self.step(scene, target_idx):
                successes += 1
        return successes / num_rounds

    # ============================================================
    # 采样指标计算：O(sample_size) 代替 O(n²)
    # ============================================================

    def get_global_metrics(self, sample_size: int = 200) -> dict:
        """
        采样计算全局语言指标

        随机采样 sample_size 对 agent 计算相似度，
        代替穷举所有 n*(n-1)/2 对。
        """
        n = len(self.agents)
        max_pairs = n * (n - 1) // 2
        actual_sample = min(sample_size, max_pairs)

        # 随机采样不重复的 (i, j) 对
        pairs = set()
        while len(pairs) < actual_sample:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                pair = (min(i, j), max(i, j))
                pairs.add(pair)
        pairs = list(pairs)

        sims = []
        for i, j in pairs:
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
            'num_sampled_pairs': len(sims),
            'total_possible_pairs': max_pairs,
        }

    def get_dialect_metrics(self, sample_size: int = 200) -> Dict:
        """
        采样计算方言分化指标
        """
        if self.groups is None:
            return self.get_global_metrics(sample_size)

        # 组内相似度（采样）
        intra_sims = []
        for group in self.groups:
            if len(group) < 2:
                continue
            group_pairs = min(sample_size // len(self.groups),
                              len(group) * (len(group) - 1) // 2)
            sampled = set()
            attempts = 0
            while len(sampled) < group_pairs and attempts < group_pairs * 3:
                i, j = random.sample(group, 2)
                pair = (min(i, j), max(i, j))
                sampled.add(pair)
                attempts += 1
            for i, j in sampled:
                sim = compute_language_similarity(
                    self.agents[i].language, self.agents[j].language
                )
                intra_sims.append(sim['overall_similarity'])

        # 组间相似度（采样）
        inter_sims = []
        all_groups = list(range(len(self.groups)))
        inter_pairs = sample_size
        sampled = set()
        attempts = 0
        while len(sampled) < inter_pairs and attempts < inter_pairs * 3:
            gi, gj = random.sample(all_groups, 2)
            i = random.choice(self.groups[gi])
            j = random.choice(self.groups[gj])
            pair = (min(i, j), max(i, j))
            sampled.add(pair)
            attempts += 1
        for i, j in sampled:
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

    # ============================================================
    # 语言家族检测
    # ============================================================

    def detect_language_families(self, threshold: float = 0.5,
                                 sample_size: int = 500) -> List[List[int]]:
        """
        基于相似度阈值的语言家族检测

        使用单链接层次聚类：如果两个 agent 的语言相似度 > threshold，
        它们属于同一个语言家族。
        """
        n = len(self.agents)

        # 采样计算相似度矩阵
        sim_matrix = np.zeros((n, n))
        np.fill_diagonal(sim_matrix, 1.0)

        pairs_needed = min(sample_size, n * (n - 1) // 2)
        sampled = set()
        while len(sampled) < pairs_needed:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                pair = (min(i, j), max(i, j))
                sampled.add(pair)

        for i, j in sampled:
            sim = compute_language_similarity(
                self.agents[i].language, self.agents[j].language
            )['overall_similarity']
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

        # 单链接层次聚类
        # 每个 agent 初始为一个家族
        families = [{i} for i in range(n)]

        # 合并相似度 > threshold 的家族
        merged = True
        while merged:
            merged = False
            for fi in range(len(families)):
                if not families[fi]:
                    continue
                for fj in range(fi + 1, len(families)):
                    if not families[fj]:
                        continue
                    # 检查两个家族之间是否有高相似度对
                    should_merge = False
                    for i in families[fi]:
                        for j in families[fj]:
                            if sim_matrix[i, j] > threshold:
                                should_merge = True
                                break
                        if should_merge:
                            break
                    if should_merge:
                        families[fi] = families[fi] | families[fj]
                        families[fj] = set()
                        merged = True

        # 过滤空家族，转为列表
        result = [sorted(list(f)) for f in families if f]
        self._family_cache = result
        self._family_cache_round = self.round_num
        return result

    # ============================================================
    # 通用语检测
    # ============================================================

    def detect_lingua_franca(self, cross_group_threshold: float = 0.6,
                             sample_size: int = 200) -> Optional[str]:
        """
        检测通用语（lingua franca）

        通用语 = 某个 agent 的语言在跨群体通信中成功率最高的语言。
        如果最高成功率 > threshold，则该语言为通用语。

        返回：通用语 agent 的 id，或 None
        """
        if self.groups is None or len(self.groups) < 2:
            return None

        # 对每个 agent，计算其语言与所有其他 agent 的平均相似度
        n = len(self.agents)
        avg_sims = {}

        for agent in self.agents:
            # 采样其他 agent
            others = [i for i in range(n) if self.agents[i].id != agent.id]
            sample_others = random.sample(others, min(sample_size, len(others)))

            sims = []
            for other_idx in sample_others:
                sim = compute_language_similarity(
                    agent.language, self.agents[other_idx].language
                )['overall_similarity']
                sims.append(sim)

            avg_sims[agent.id] = np.mean(sims) if sims else 0.0

        # 找到平均相似度最高的 agent
        best_id = max(avg_sims, key=avg_sims.get)
        best_score = avg_sims[best_id]

        if best_score > cross_group_threshold:
            return best_id
        return None

    # ============================================================
    # 语言灭绝/复兴追踪
    # ============================================================

    def get_language_diversity_index(self) -> float:
        """
        语言多样性指数（Shannon 熵）

        基于语言家族大小分布计算。
        值越高表示语言越多样。
        """
        families = self.detect_language_families()
        if not families:
            return 0.0

        total = sum(len(f) for f in families)
        if total == 0:
            return 0.0

        entropy = 0.0
        for family in families:
            p = len(family) / total
            if p > 0:
                entropy -= p * np.log(p)

        return entropy

    def find_extinct_candidates(self, threshold: float = 0.3) -> List[int]:
        """
        找到可能"灭绝"的语言（使用者少且与主流语言差异大）

        返回：可能灭绝的 agent 索引列表
        """
        # 计算每个 agent 与其他 agent 的平均相似度
        n = len(self.agents)
        avg_sims = []

        for i in range(n):
            sample_others = random.sample(
                [j for j in range(n) if j != i],
                min(50, n - 1)
            )
            sims = []
            for j in sample_others:
                sim = compute_language_similarity(
                    self.agents[i].language, self.agents[j].language
                )['overall_similarity']
                sims.append(sim)
            avg_sims.append(np.mean(sims) if sims else 0.0)

        # 找到相似度低于阈值的 agent
        extinct = [i for i, sim in enumerate(avg_sims) if sim < threshold]
        return extinct

    # ============================================================
    # 辅助方法
    # ============================================================

    def get_communication_success_rate(self, window: int = 100) -> float:
        """最近 window 轮的交流成功率"""
        if not self.history:
            return 0.0
        recent = list(self.history)[-window:]
        return sum(1 for h in recent if h['success']) / len(recent)

    def get_cross_group_success_rate(self, window: int = 100) -> float:
        """跨组交流的成功率"""
        if not self.groups or not self.history:
            return 0.0

        agent_to_group = {}
        for gi, group in enumerate(self.groups):
            for i in group:
                agent_to_group[self.agents[i].id] = gi

        recent = list(self.history)[-window:]
        cross = [h for h in recent
                 if agent_to_group.get(h['speaker']) != agent_to_group.get(h['listener'])]
        if not cross:
            return 0.0
        return sum(1 for h in cross if h['success']) / len(cross)

    def get_all_agent_stats(self) -> List[Dict]:
        """获取所有 agent 的统计"""
        return [a.get_stats() for a in self.agents]

    def get_topology_stats(self) -> Dict:
        """获取拓扑统计"""
        degrees = [len(self.adjacency[a.id]) for a in self.agents]
        return {
            'topology': self.topology,
            'num_agents': self.num_agents,
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'std_degree': np.std(degrees),
        }

    def merge_groups(self):
        """打破组间隔离"""
        self.groups = None
        self.topology = 'full'
        self.adjacency = self._build_adjacency()

    # ============================================================
    # 批量相似度计算（GPU 加速）
    # ============================================================

    def _build_vocab_matrix(self) -> Tuple[Dict[str, int], np.ndarray]:
        """
        构建全局词汇矩阵

        Returns:
            symbol_to_idx: 符号→索引映射
            freq_matrix: (n_agents, n_symbols) 频率矩阵
        """
        # 收集所有符号
        all_symbols = set()
        for agent in self.agents:
            all_symbols.update(agent.language.vocabulary.keys())
        symbol_list = sorted(all_symbols)
        symbol_to_idx = {s: i for i, s in enumerate(symbol_list)}

        # 构建频率矩阵
        n = len(self.agents)
        m = len(symbol_list)
        freq_matrix = np.zeros((n, m), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            for sym, data in agent.language.vocabulary.items():
                freq_matrix[i, symbol_to_idx[sym]] = data.get('frequency', 0)

        return symbol_to_idx, freq_matrix

    def batch_cosine_similarity(self, sample_size: int = 0) -> np.ndarray:
        """
        GPU 加速的批量余弦相似度计算

        Args:
            sample_size: 0 = 计算所有对（O(n²)），>0 = 随机采样

        Returns:
            sim_matrix: (n, n) 相似度矩阵（仅词汇频率维度）
        """
        _, freq_matrix = self._build_vocab_matrix()
        n = freq_matrix.shape[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        freq_t = torch.from_numpy(freq_matrix).to(device)

        # L2 归一化
        norms = torch.norm(freq_t, dim=1, keepdim=True).clamp(min=1e-8)
        freq_normed = freq_t / norms

        # 余弦相似度矩阵 = 归一化后矩阵乘法
        sim_matrix = (freq_normed @ freq_normed.t()).cpu().numpy()

        return sim_matrix

    def get_global_metrics_fast(self, sample_size: int = 200) -> dict:
        """
        GPU 加速的全局指标计算

        用批量余弦相似度替代逐对 compute_language_similarity 调用。
        """
        sim_matrix = self.batch_cosine_similarity()
        n = sim_matrix.shape[0]

        # 采样上三角元素
        max_pairs = n * (n - 1) // 2
        actual_sample = min(sample_size, max_pairs)

        # 提取上三角（排除对角线）
        triu_i, triu_j = np.triu_indices(n, k=1)
        all_sims = sim_matrix[triu_i, triu_j]

        if actual_sample < len(all_sims):
            indices = np.random.choice(len(all_sims), actual_sample, replace=False)
            sampled_sims = all_sims[indices]
        else:
            sampled_sims = all_sims

        return {
            'avg_vocab_similarity': float(np.mean(sampled_sims)),
            'num_sampled_pairs': len(sampled_sims),
            'total_possible_pairs': max_pairs,
        }

    def detect_language_families_fast(self, threshold: float = 0.5) -> List[List[int]]:
        """
        GPU 加速的语言家族检测

        用批量相似度矩阵 + 并查集替代逐对采样 + 层次聚类。
        """
        sim_matrix = self.batch_cosine_similarity()
        n = sim_matrix.shape[0]

        # 并查集
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # 批量合并：相似度 > threshold 的对
        triu_i, triu_j = np.triu_indices(n, k=1)
        sims = sim_matrix[triu_i, triu_j]
        mask = sims > threshold
        for i, j in zip(triu_i[mask], triu_j[mask]):
            union(i, j)

        # 收集家族
        families = defaultdict(list)
        for i in range(n):
            families[find(i)].append(i)

        result = [sorted(members) for members in families.values() if len(members) > 0]
        self._family_cache = result
        self._family_cache_round = self.round_num
        return result

    def replace_agents(self, agent_indices: List[int]):
        """替换指定 agent（模拟语言灭绝/复兴）"""
        for idx in agent_indices:
            old_id = self.agents[idx].id
            new_agent = LanguageAgent(old_id)
            self.agents[idx] = new_agent
            self.agent_map[old_id] = new_agent
        # 重建邻接关系
        self.adjacency = self._build_adjacency()
