"""区域语言社会 — 多区域、网络拓扑、方言分化

模拟语言在不同地理区域间的分化与演化：
- 多个区域各有不同的环境分布（导致不同语义需求）
- 区域间通过桥接代理有限通信
- 随时间推移产生方言分化
- 可触发区域合并以观察语言趋同
"""

import random
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch


class RegionConfig:
    """区域配置

    参数:
        region_id: 区域唯一标识
        name: 区域名称
        env_distribution: 属性分布权重（如 {'color': 0.3, 'shape': 0.7}）
        num_agents: 该区域代理数量
    """

    def __init__(self, region_id: int, name: str,
                 env_distribution: Dict[str, float],
                 num_agents: int):
        self.region_id = region_id
        self.name = name
        self.env_distribution = env_distribution
        self.num_agents = num_agents
        self.agent_ids: List[int] = []

    def to_dict(self) -> dict:
        return {
            'region_id': self.region_id,
            'name': self.name,
            'env_distribution': dict(self.env_distribution),
            'num_agents': self.num_agents,
            'agent_ids': list(self.agent_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RegionConfig':
        cfg = cls(
            region_id=d['region_id'],
            name=d['name'],
            env_distribution=d['env_distribution'],
            num_agents=d['num_agents'],
        )
        cfg.agent_ids = d.get('agent_ids', [])
        return cfg


class RegionalLanguageSociety:
    """区域语言社会 — 多区域、网络拓扑、方言分化

    支持的网络拓扑：
    - 'full': 全连接
    - 'small_world': Watts-Strogatz 小世界
    - 'scale_free': Barabasi-Albert 优先连接
    - 'groups': 分组隔离
    - 'groups_with_bridges': 分组 + 桥接代理

    参数:
        num_agents: 总代理数
        num_regions: 区域数
        topology: 网络拓扑类型
        noise_rate: 通信噪声率
        comm_range: 通信范围（空间距离阈值）
    """

    def __init__(self, num_agents: int = 20, num_regions: int = 3,
                 topology: str = 'groups_with_bridges',
                 noise_rate: float = 0.0,
                 comm_range: float = 3.0):
        self.num_agents = num_agents
        self.num_regions = num_regions
        self.topology = topology
        self.noise_rate = noise_rate
        self.comm_range = comm_range

        # 创建区域（每个区域有不同的环境偏好）
        self.regions: List[RegionConfig] = []
        self._create_regions()

        # 分配代理到区域
        self.agent_region: Dict[int, int] = {}
        self._assign_agents()

        # 构建邻接图
        self.adjacency: Dict[int, List[int]] = self._build_adjacency()

    # ── 初始化 ───────────────────────────────────────────────

    def _create_regions(self) -> None:
        """创建区域，每个区域有不同属性分布权重"""
        base_attrs = ['color', 'shape', 'size', 'material', 'texture']
        for i in range(self.num_regions):
            # 每个区域对属性的侧重不同：用 torch 生成随机权重
            weights = torch.softmax(torch.randn(len(base_attrs)), dim=0)
            dist = {
                attr: weights[j].item()
                for j, attr in enumerate(base_attrs)
            }
            agents_per_region = self.num_agents // self.num_regions
            name = f'region_{i}'
            self.regions.append(RegionConfig(i, name, dist, agents_per_region))

    def _assign_agents(self) -> None:
        """轮询分配代理到各区域"""
        agent_ids = list(range(self.num_agents))
        for idx, aid in enumerate(agent_ids):
            rid = idx % self.num_regions
            self.agent_region[aid] = rid
            self.regions[rid].agent_ids.append(aid)

    # ── 通信 ─────────────────────────────────────────────────

    def step(self,
             agent_vocabularies: Dict[int, Dict],
             agent_id: int,
             scene: Dict,
             target_idx: int) -> float:
        """执行一轮通信

        说话者（agent_id）描述场景中的目标物体，
        随机邻居作为听者尝试识别。

        参数:
            agent_vocabularies: 各代理的词汇表 {agent_id: {symbol: count}}
            agent_id: 说话者 ID
            scene: 场景描述（物体属性列表）
            target_idx: 目标物体在场景中的索引
        返回:
            通信成功率（1.0 成功 / 0.0 失败）
        """
        neighbors = self.adjacency.get(agent_id, [])
        if not neighbors:
            return 0.0

        listener_id = random.choice(neighbors)
        vocab_speaker = agent_vocabularies.get(agent_id, {})
        vocab_listener = agent_vocabularies.get(listener_id, {})

        if not scene or not vocab_speaker:
            return 0.0

        # 说话者选择与目标最相关的符号
        target = scene[target_idx] if target_idx < len(scene) else {}
        spoken_symbols = self._select_symbols(vocab_speaker, target)

        # 噪声：以 noise_rate 概率替换一个符号
        if self.noise_rate > 0 and random.random() < self.noise_rate:
            if spoken_symbols:
                idx_noise = random.randint(0, len(spoken_symbols) - 1)
                spoken_symbols[idx_noise] = f'noise_{random.randint(0, 99)}'

        # 听者匹配
        best_match = self._interpret(vocab_listener, scene, spoken_symbols)
        return 1.0 if best_match == target_idx else 0.0

    def _select_symbols(self, vocab: Dict, target: Dict) -> List[str]:
        """说话者从词汇表中选择与目标属性最匹配的符号"""
        if not vocab or not target:
            # 随机选符号
            symbols = list(vocab.keys())
            return random.sample(symbols, min(2, len(symbols))) if symbols else []

        scored = []
        for sym, count in vocab.items():
            # 符号与目标属性的重叠度
            overlap = 1.0 if sym in str(target.values()) else count / (sum(vocab.values()) + 1)
            scored.append((sym, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:2]]

    def _interpret(self, vocab: Dict, scene, symbols: List[str]) -> int:
        """听者根据符号匹配场景中最相似的物体"""
        if not vocab or not scene:
            return random.randint(0, max(len(scene) - 1, 0))

        best_idx = 0
        best_score = -1.0
        for idx, obj in enumerate(scene):
            score = 0.0
            obj_vals = set(str(v) for v in obj.values()) if isinstance(obj, dict) else set()
            for sym in symbols:
                if sym in vocab:
                    score += vocab[sym]
                if sym in obj_vals:
                    score += 2.0
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    # ── 相似度 ───────────────────────────────────────────────

    def compute_language_similarity(self,
                                   vocab_a: Dict, vocab_b: Dict) -> float:
        """计算两个词汇表之间的 Jaccard 相似度

        Jaccard = |A ∩ B| / |A ∪ B|
        """
        keys_a = set(vocab_a.keys())
        keys_b = set(vocab_b.keys())
        if not keys_a and not keys_b:
            return 1.0
        if not keys_a or not keys_b:
            return 0.0
        return len(keys_a & keys_b) / len(keys_a | keys_b)

    def get_regional_metrics(self,
                            agent_vocabularies: Dict[int, Dict]) -> Dict:
        """计算各区域的语言相似度指标

        返回:
            per_region_avg: 各区域内平均相似度
            cross_region_avg: 跨区域平均相似度
            region_sizes: 各区域代理数
        """
        per_region_avg: Dict[int, float] = {}
        cross_pairs: List[float] = []

        # 区域内相似度
        for region in self.regions:
            ids = region.agent_ids
            if len(ids) < 2:
                per_region_avg[region.region_id] = 1.0
                continue
            sims = []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    sim = self.compute_language_similarity(
                        agent_vocabularies.get(ids[i], {}),
                        agent_vocabularies.get(ids[j], {}),
                    )
                    sims.append(sim)
            per_region_avg[region.region_id] = (
                sum(sims) / len(sims) if sims else 0.0
            )

        # 跨区域相似度
        for ri in range(len(self.regions)):
            for rj in range(ri + 1, len(self.regions)):
                ids_i = self.regions[ri].agent_ids
                ids_j = self.regions[rj].agent_ids
                if ids_i and ids_j:
                    ai = random.choice(ids_i)
                    aj = random.choice(ids_j)
                    cross_pairs.append(self.compute_language_similarity(
                        agent_vocabularies.get(ai, {}),
                        agent_vocabularies.get(aj, {}),
                    ))

        return {
            'per_region_avg': per_region_avg,
            'cross_region_avg': sum(cross_pairs) / len(cross_pairs) if cross_pairs else 0.0,
            'region_sizes': {r.region_id: len(r.agent_ids) for r in self.regions},
        }

    def get_dialect_divergence(self,
                              agent_vocabularies: Dict[int, Dict]) -> float:
        """计算方言分化度 = 1 - (跨区域相似度 / 区域内相似度)

        值越高表示方言分化越严重：
        - 0.0 = 无分化
        - 1.0 = 完全分化
        """
        metrics = self.get_regional_metrics(agent_vocabularies)
        intra = sum(metrics['per_region_avg'].values())
        intra_count = len(metrics['per_region_avg'])
        if intra_count == 0:
            return 0.0
        intra_avg = intra / intra_count

        cross_avg = metrics['cross_region_avg']
        if intra_avg == 0:
            return 1.0
        return 1.0 - min(1.0, cross_avg / intra_avg)

    # ── 区域合并 ─────────────────────────────────────────────

    def merge_regions(self) -> None:
        """将所有区域合并为单一全连接种群

        用于模拟文化交流/全球化导致的语言趋同。
        """
        self.num_regions = 1
        all_ids = list(range(self.num_agents))
        self.regions = [RegionConfig(
            region_id=0,
            name='merged',
            env_distribution={'color': 0.2, 'shape': 0.2, 'size': 0.2,
                              'material': 0.2, 'texture': 0.2},
            num_agents=self.num_agents,
        )]
        self.regions[0].agent_ids = all_ids
        self.agent_region = {aid: 0 for aid in all_ids}
        # 全连接
        self.adjacency = {aid: [o for o in all_ids if o != aid] for aid in all_ids}

    # ── 网络拓扑 ─────────────────────────────────────────────

    def _build_adjacency(self) -> Dict[int, List[int]]:
        """根据拓扑类型构建邻接图"""
        all_ids = list(range(self.num_agents))

        if self.topology == 'full':
            return {aid: [o for o in all_ids if o != aid] for aid in all_ids}

        if self.topology == 'groups':
            return self._build_groups(all_ids)

        if self.topology == 'groups_with_bridges':
            return self._build_groups_with_bridges(all_ids)

        if self.topology == 'small_world':
            return self._build_small_world(all_ids)

        if self.topology == 'scale_free':
            return self._build_scale_free(all_ids)

        # 默认全连接
        return {aid: [o for o in all_ids if o != aid] for aid in all_ids}

    def _build_groups(self, ids: List[int]) -> Dict[int, List[int]]:
        """分组隔离：仅在区域内部连接"""
        adj: Dict[int, List[int]] = {aid: [] for aid in ids}
        for region in self.regions:
            for i, aid in enumerate(region.agent_ids):
                for j, bid in enumerate(region.agent_ids):
                    if i != j:
                        adj[aid].append(bid)
        return adj

    def _build_groups_with_bridges(self, ids: List[int]) -> Dict[int, List[int]]:
        """分组 + 桥接：区域内全连接，区域间随机桥接"""
        adj = self._build_groups(ids)

        # 每对区域之间添加 1-2 条桥接
        for i in range(len(self.regions)):
            for j in range(i + 1, len(self.regions)):
                ri_ids = self.regions[i].agent_ids
                rj_ids = self.regions[j].agent_ids
                if ri_ids and rj_ids:
                    n_bridges = min(2, len(ri_ids), len(rj_ids))
                    for _ in range(n_bridges):
                        a = random.choice(ri_ids)
                        b = random.choice(rj_ids)
                        if b not in adj[a]:
                            adj[a].append(b)
                        if a not in adj[b]:
                            adj[b].append(a)
        return adj

    def _build_small_world(self, ids: List[int],
                           k: int = 4, p: float = 0.1) -> Dict[int, List[int]]:
        """Watts-Strogatz 小世界网络

        参数:
            ids: 代理 ID 列表
            k: 每个节点的初始邻居数（环形格）
            p: 重连概率
        """
        n = len(ids)
        adj: Dict[int, List[int]] = {aid: [] for aid in ids}

        # 环形格：每个节点连接 k/2 个近邻
        for i in range(n):
            for offset in range(1, k // 2 + 1):
                j = (i + offset) % n
                if ids[j] not in adj[ids[i]]:
                    adj[ids[i]].append(ids[j])
                if ids[i] not in adj[ids[j]]:
                    adj[ids[j]].append(ids[i])

        # 随机重连
        for i in range(n):
            neighbors = list(adj[ids[i]])
            for nb in neighbors:
                if random.random() < p:
                    adj[ids[i]].remove(nb)
                    adj[nb].remove(ids[i])
                    new_nb = random.choice(ids)
                    while new_nb == ids[i] or new_nb in adj[ids[i]]:
                        new_nb = random.choice(ids)
                    adj[ids[i]].append(new_nb)
                    adj[new_nb].append(ids[i])

        return adj

    def _build_scale_free(self, ids: List[int],
                          m: int = 2) -> Dict[int, List[int]]:
        """Barabasi-Albert 优先连接网络

        参数:
            ids: 代理 ID 列表
            m: 每个新节点连接的已有节点数
        """
        n = len(ids)
        adj: Dict[int, List[int]] = {aid: [] for aid in ids}

        # 初始完全图（m+1 个节点）
        initial = ids[:m + 1]
        for i, a in enumerate(initial):
            for j, b in enumerate(initial):
                if i != j and b not in adj[a]:
                    adj[a].append(b)

        # 逐个加入，优先连接
        degrees = {aid: len(adj[aid]) for aid in ids}
        for new_id in ids[m + 1:]:
            # 计算已有节点的连接概率
            existing = [nid for nid in ids if nid != new_id and adj[nid]]
            if not existing:
                continue
            total_deg = sum(degrees[nid] for nid in existing)
            if total_deg == 0:
                targets = random.sample(existing, min(m, len(existing)))
            else:
                probs = torch.tensor([degrees[nid] / total_deg for nid in existing])
                probs = probs / probs.sum()
                n_targets = min(m, len(existing))
                targets_idx = torch.multinomial(probs, n_targets, replacement=False)
                targets = [existing[idx] for idx in targets_idx.tolist()]

            for t in targets:
                if new_id not in adj[t]:
                    adj[t].append(new_id)
                if t not in adj[new_id]:
                    adj[new_id].append(t)
                degrees[t] += 1
                degrees[new_id] += 1

        return adj

    # ── 持久化 ───────────────────────────────────────────────

    def save_state(self) -> dict:
        """保存社会状态为字典"""
        return {
            'num_agents': self.num_agents,
            'num_regions': self.num_regions,
            'topology': self.topology,
            'noise_rate': self.noise_rate,
            'comm_range': self.comm_range,
            'regions': [r.to_dict() for r in self.regions],
            'agent_region': dict(self.agent_region),
            'adjacency': {str(k): v for k, v in self.adjacency.items()},
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复社会状态"""
        self.num_agents = state['num_agents']
        self.num_regions = state['num_regions']
        self.topology = state['topology']
        self.noise_rate = state['noise_rate']
        self.comm_range = state['comm_range']
        self.regions = [RegionConfig.from_dict(r) for r in state.get('regions', [])]
        self.agent_region = {int(k): v for k, v in state.get('agent_region', {}).items()}
        self.adjacency = {
            int(k): v for k, v in state.get('adjacency', {}).items()
        }
