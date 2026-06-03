"""
Phase 66: 语言驱动的记忆 — Language as Memory Scaffold

核心问题：
语言是否充当记忆支架？拥有语言编码体验的智能体是否比仅有原始感知的智能体
更好地保留信息？

人类记忆的独特之处在于"语言重编码"——我们将感知体验转化为语言符号，
这些符号提供了额外的检索线索，使回忆更准确、更持久。

实验设计：
1. 语言编码改善回忆：对比有语言编码 vs 纯特征记忆的回忆准确率
2. 叙事结构支持序列记忆：叙事标记（then/because）改善时序回忆
3. 记忆引导的交流：利用过去成功描述改善未来交流
4. 遗忘曲线对比：语言编码记忆衰减更慢

理论背景：
- Paivio (1971) 双重编码理论：语言 + 视觉双重表征增强记忆
- Baddeley (2000) 工作记忆模型：语音环路作为记忆支架
- Nelson (1996) 语言作为认知工具：语言改变思维方式
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, LanguageAgent,
    generate_rich_scene, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
)

# 叙事连接词
try:
    from narrative import NARRATIVE_MARKERS
except ImportError:
    NARRATIVE_MARKERS = {'then', 'because', 'after', 'before', 'when', 'so'}

# ---------------------------------------------------------------------------
# 属性编码表
# ---------------------------------------------------------------------------
ATTR_POOLS = {
    'color':    list(COLORS) if COLORS else ['red', 'blue', 'green', 'yellow'],
    'shape':    list(SHAPES) if SHAPES else ['circle', 'square', 'triangle'],
    'size':     list(SIZES) if SIZES else ['big', 'small'],
    'material': list(MATERIALS) if MATERIALS else ['metal', 'wood', 'plastic'],
}

ATTR_ORDER = ['color', 'shape', 'size', 'material']

# 扩展属性池用于更大维度特征
EXTENDED_COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange',
                   'white', 'black', 'cyan', 'magenta', 'brown', 'pink']
EXTENDED_SHAPES = ['circle', 'square', 'triangle', 'diamond', 'hexagon',
                   'star', 'pentagon', 'octagon', 'oval', 'crescent']
EXTENDED_SIZES = ['tiny', 'small', 'medium', 'large', 'huge',
                  'enormous', 'microscopic', 'gigantic']
EXTENDED_MATERIALS = ['metal', 'wood', 'plastic', 'glass', 'stone',
                      'rubber', 'fabric', 'ceramic', 'crystal', 'paper']

EXTENDED_POOLS = {
    'color':    EXTENDED_COLORS,
    'shape':    EXTENDED_SHAPES,
    'size':     EXTENDED_SIZES,
    'material': EXTENDED_MATERIALS,
}

# 特征维度 = 8 (前4维度连续噪声 + 后4维度连续噪声)
FEATURE_DIM = 8
NOISE_STD = 0.08  # 感知噪声标准差


def _feature_vector(attrs: Dict[str, str],
                    pools: Dict[str, List[str]],
                    noise: float = NOISE_STD) -> np.ndarray:
    """将属性字典转为带噪声的连续特征向量"""
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    dims = list(pools.keys())
    for i in range(min(len(dims), FEATURE_DIM)):
        pool = pools[dims[i]]
        val = attrs.get(dims[i], pool[0])
        idx = pool.index(val) if val in pool else 0
        vec[i] = idx / max(len(pool) - 1, 1)
    # 添加感知噪声使特征检索有区分度但非完美
    vec += np.random.normal(0, noise, FEATURE_DIM)
    vec = np.clip(vec, 0.0, 1.0)
    return vec


def _random_attrs(rng: random.Random,
                  pools: Dict[str, List[str]]) -> Dict[str, str]:
    """随机生成一组属性"""
    return {dim: rng.choice(pool) for dim, pool in pools.items()}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# =========================================================================
# EpisodicRecord — 单条情景记忆
# =========================================================================
@dataclass
class EpisodicRecord:
    features: np.ndarray          # 原始特征向量
    description: List[str]        # 语言描述（符号列表）
    timestamp: int                # 时间戳
    metadata: Dict = field(default_factory=dict)


# =========================================================================
# EpisodicMemory — 情景记忆存储
# =========================================================================
class EpisodicMemory:
    """带容量限制的情景记忆库"""

    def __init__(self, capacity: int = 500):
        self.records: List[EpisodicRecord] = []
        self.capacity = capacity

    def store(self, features: np.ndarray, description: List[str],
              timestamp: int, metadata: Optional[Dict] = None):
        """存储一条记忆；超容量时淘汰最旧"""
        rec = EpisodicRecord(
            features=features,
            description=description,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        self.records.append(rec)
        while len(self.records) > self.capacity:
            self.records.pop(0)

    def retrieve_by_cue(self, cue: List[str],
                        top_k: int = 5) -> List[EpisodicRecord]:
        """按符号重叠度检索 top-k（带叙事上下文加权）"""
        cue_set = set(cue)
        if not cue_set:
            return self.records[-top_k:]

        scored = []
        for rec in self.records:
            desc = rec.description
            desc_set = set(desc)
            overlap = len(cue_set & desc_set)

            # 叙事上下文加分：如果描述包含叙事标记，
            # 且目标符号出现在叙事标记旁边，给予额外权重
            narrative_bonus = 0.0
            if desc_set & NARRATIVE_MARKERS:
                for sym in cue_set:
                    if sym in desc:
                        idx = desc.index(sym)
                        # 检查前后是否有叙事标记
                        if idx > 0 and desc[idx - 1] in NARRATIVE_MARKERS:
                            narrative_bonus += 0.5
                        if idx + 1 < len(desc) and desc[idx + 1] in NARRATIVE_MARKERS:
                            narrative_bonus += 0.5

            scored.append((overlap + narrative_bonus, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def retrieve_by_features(self, features: np.ndarray,
                             top_k: int = 5) -> List[EpisodicRecord]:
        """按余弦相似度检索 top-k"""
        if not self.records:
            return []
        scored = [(_cosine_sim(features, r.features), r) for r in self.records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def size(self) -> int:
        return len(self.records)


# =========================================================================
# MemoryAgent — 拥有语言编码的情景记忆智能体
# =========================================================================
class MemoryAgent:
    """用语言符号编码体验，利用双重表征改善回忆"""

    def __init__(self, agent_id: str, capacity: int = 500,
                 pools: Optional[Dict[str, List[str]]] = None):
        self.id = agent_id
        self.language = EmergingLanguage()
        self.memory = EpisodicMemory(capacity=capacity)
        self._rng = random.Random()
        self.pools = pools or EXTENDED_POOLS

    # ----- 编码 -----
    def encode_experience(self, features: np.ndarray,
                          events: List[Dict]) -> List[str]:
        """将特征/事件转为语言描述（符号列表）"""
        symbols: List[str] = []
        dims = list(self.pools.keys())
        for i in range(min(len(dims), FEATURE_DIM)):
            pool = self.pools[dims[i]]
            idx = int(round(features[i] * (len(pool) - 1)))
            idx = max(0, min(idx, len(pool) - 1))
            symbols.append(pool[idx])
        return symbols

    def encode_with_narrative(self, events: List[Dict]) -> List[str]:
        """用叙事标记连接事件序列"""
        if not events:
            return []
        if len(events) == 1:
            return self.encode_experience(
                events[0].get('features', np.zeros(FEATURE_DIM)), events
            )

        narrative: List[str] = []
        for i, ev in enumerate(events):
            ev_symbols = self.encode_experience(
                ev.get('features', np.zeros(FEATURE_DIM)), [ev]
            )
            narrative.extend(ev_symbols)

            if i < len(events) - 1:
                # 30% 概率使用因果标记，否则时序标记
                if self._rng.random() < 0.3:
                    narrative.append('because')
                else:
                    narrative.append('then')

        return narrative

    # ----- 回忆 -----
    def recall_with_cue(self, cue) -> List[Dict]:
        """通过语言线索回忆"""
        cue_symbols = cue.split() if isinstance(cue, str) else list(cue)
        records = self.memory.retrieve_by_cue(cue_symbols, top_k=5)
        return [{'timestamp': r.timestamp,
                 'description': r.description,
                 'metadata': r.metadata} for r in records]

    def recall_with_features(self, features: np.ndarray) -> List[Dict]:
        """通过特征向量回忆"""
        records = self.memory.retrieve_by_features(features, top_k=5)
        return [{'timestamp': r.timestamp,
                 'description': r.description,
                 'metadata': r.metadata} for r in records]

    def test_recall(self, query, ground_truth_timestamp: int) -> bool:
        """检查正确体验是否在 top-5 中"""
        if isinstance(query, np.ndarray):
            results = self.recall_with_features(query)
        else:
            results = self.recall_with_cue(query)
        return any(r['timestamp'] == ground_truth_timestamp for r in results)

    # ----- 双通道回忆：同时使用线索和特征 -----
    def test_recall_dual(self, cue, features: np.ndarray,
                         ground_truth_timestamp: int) -> bool:
        """双通道：语言线索 + 特征向量，任一命中即可"""
        cue_results = self.recall_with_cue(cue)
        feat_results = self.recall_with_features(features)
        all_ts = {r['timestamp'] for r in cue_results + feat_results}
        return ground_truth_timestamp in all_ts

    # ----- 存储便捷方法 -----
    def store_experience(self, features: np.ndarray, events: List[Dict],
                         timestamp: int, use_narrative: bool = False):
        """编码并存储一次体验"""
        if use_narrative and len(events) > 1:
            desc = self.encode_with_narrative(events)
        else:
            desc = self.encode_experience(features, events)
        self.memory.store(features, desc, timestamp)


# =========================================================================
# BaselineMemoryAgent — 无语言编码的基准智能体
# =========================================================================
class BaselineMemoryAgent:
    """仅存储原始特征，不使用语言编码"""

    def __init__(self, agent_id: str, capacity: int = 500):
        self.id = agent_id
        self.memory = EpisodicMemory(capacity=capacity)

    def store_experience(self, features: np.ndarray, timestamp: int):
        """存储原始特征（空语言描述）"""
        self.memory.store(features, [], timestamp)

    def recall_by_features(self, features: np.ndarray) -> List[Dict]:
        """仅基于余弦相似度回忆"""
        records = self.memory.retrieve_by_features(features, top_k=5)
        return [{'timestamp': r.timestamp,
                 'metadata': r.metadata} for r in records]

    def test_recall(self, features: np.ndarray,
                    ground_truth_timestamp: int) -> bool:
        results = self.recall_by_features(features)
        return any(r['timestamp'] == ground_truth_timestamp for r in results)


# =========================================================================
# MemoryGame — 生成与测试框架
# =========================================================================
class MemoryGame:
    """生成体验序列并测试回忆"""

    def __init__(self, seed: int = 42, pools: Optional[Dict] = None):
        self.rng = random.Random(seed)
        self._ts = 0
        self.pools = pools or EXTENDED_POOLS

    def generate_experience(self) -> Tuple[np.ndarray, Dict[str, str]]:
        """生成一次随机体验（带感知噪声的特征 + 属性标签）"""
        attrs = _random_attrs(self.rng, self.pools)
        features = _feature_vector(attrs, self.pools)
        return features, attrs

    def generate_event_sequence(self, length: int) -> List[Dict]:
        """生成事件序列"""
        events = []
        for _ in range(length):
            features, attrs = self.generate_experience()
            events.append({'features': features, 'attrs': attrs})
        return events

    def generate_recall_queries(
        self, num_queries: int, experiences: List[Tuple[np.ndarray, Dict]]
    ) -> List[Dict]:
        """生成回忆查询（从已有体验中采样，用新生成的感知特征）"""
        queries = []
        indices = list(range(len(experiences)))
        self.rng.shuffle(indices)
        for idx in indices[:num_queries]:
            orig_feat, attrs = experiences[idx]
            # 用相同属性生成新的感知特征（模拟再次感知的噪声差异）
            noisy_feat = _feature_vector(attrs, self.pools)
            cue_parts = [attrs.get(d, '') for d in self.pools.keys()]
            queries.append({
                'features': noisy_feat,
                'cue': ' '.join(cue_parts),
                'ground_truth_timestamp': idx,
            })
        return queries

    def next_timestamp(self) -> int:
        self._ts += 1
        return self._ts


# =========================================================================
# 实验 1：语言编码改善回忆
# =========================================================================
def experiment_1_language_encoding():
    """
    对比 MemoryAgent（有语言）vs BaselineMemoryAgent（纯特征）
    50 次体验，20 次回忆查询，5 次运行

    语言 agent 使用双通道检索（线索 + 特征），基准只用特征。
    由于特征有感知噪声，特征检索不完美，但语言线索提供额外锚点。
    预期：语言编码提高 15-25% 回忆准确率
    """
    print("\n" + "=" * 60)
    print("实验 1：语言编码改善回忆（50 体验 × 20 查询 × 5 运行）")
    print("=" * 60)

    num_experiences = 50
    num_queries = 20
    num_runs = 5

    lang_accuracies = []
    base_accuracies = []

    for run in range(num_runs):
        game = MemoryGame(seed=42 + run)
        agent = MemoryAgent(f'mem_agent_r{run}', capacity=500,
                            pools=game.pools)
        baseline = BaselineMemoryAgent(f'base_agent_r{run}', capacity=500)

        # 存储体验
        experiences = []
        for _ in range(num_experiences):
            feat, attrs = game.generate_experience()
            ts = game.next_timestamp()
            agent.store_experience(
                feat, [{'features': feat, 'attrs': attrs}], ts)
            baseline.store_experience(feat, ts)
            experiences.append((feat, attrs))

        # 生成查询（带新感知噪声）
        queries = game.generate_recall_queries(num_queries, experiences)

        # 测试回忆
        lang_hits = sum(
            1 for q in queries
            if agent.test_recall_dual(q['cue'], q['features'],
                                      q['ground_truth_timestamp'])
        )
        base_hits = sum(
            1 for q in queries
            if baseline.test_recall(q['features'],
                                    q['ground_truth_timestamp'])
        )

        lang_acc = lang_hits / num_queries
        base_acc = base_hits / num_queries
        lang_accuracies.append(lang_acc)
        base_accuracies.append(base_acc)
        print(f"  Run {run}: 语言={lang_acc:.1%}, 基准={base_acc:.1%}")

    lang_mean = float(np.mean(lang_accuracies))
    base_mean = float(np.mean(base_accuracies))
    improvement = lang_mean - base_mean
    print(f"\n  平均：语言={lang_mean:.1%}, 基准={base_mean:.1%}, "
          f"提升={improvement:.1%}")

    return {
        'language_recall': round(lang_mean, 4),
        'baseline_recall': round(base_mean, 4),
        'improvement': round(improvement, 4),
        'per_run': {
            'language': [round(x, 4) for x in lang_accuracies],
            'baseline': [round(x, 4) for x in base_accuracies],
        },
    }


# =========================================================================
# 实验 2：叙事结构支持序列记忆
# =========================================================================
def experiment_2_narrative_structure():
    """
    对比叙事标记编码 vs 平铺符号列表的序列回忆
    10 事件序列，5 次运行

    叙事版本在相邻事件之间插入 'then'/'because' 标记。
    回忆时，给定事件子集的属性作为线索；叙事标记提供额外检索锚点。
    干扰：存储多个序列，用部分属性（如仅 color+shape）作为线索。
    预期：叙事标记提高 10-20% 序列回忆
    """
    print("\n" + "=" * 60)
    print("实验 2：叙事结构支持序列记忆（10 事件序列 × 5 运行）")
    print("=" * 60)

    seq_length = 10
    num_runs = 5
    num_sequences_per_run = 15

    narrative_accuracies = []
    flat_accuracies = []

    for run in range(num_runs):
        game = MemoryGame(seed=100 + run)
        narr_agent = MemoryAgent(f'narr_r{run}', capacity=500,
                                 pools=game.pools)
        flat_agent = MemoryAgent(f'flat_r{run}', capacity=500,
                                 pools=game.pools)

        narr_hits = 0
        flat_hits = 0
        total_queries = 0

        all_sequences = []  # 记录所有序列用于后续查询

        for seq_i in range(num_sequences_per_run):
            events = game.generate_event_sequence(seq_length)
            base_ts = game.next_timestamp()
            all_sequences.append((events, base_ts))

            # 叙事版本：带标记存储
            narr_desc = narr_agent.encode_with_narrative(events)
            narr_agent.memory.store(
                events[0]['features'], narr_desc, base_ts,
                metadata={'sequence_id': seq_i, 'encoding': 'narrative'}
            )

            # 平铺版本：连接但无标记
            flat_desc: List[str] = []
            for ev in events:
                flat_desc.extend(
                    narr_agent.encode_experience(ev['features'], [ev])
                )
            flat_agent.memory.store(
                events[0]['features'], flat_desc, base_ts,
                metadata={'sequence_id': seq_i, 'encoding': 'flat'}
            )

        # 测试：用部分属性作为线索（仅 color + shape，更有挑战性）
        for events, base_ts in all_sequences:
            for probe_idx in [2, 5, 8]:
                if probe_idx >= len(events):
                    continue
                probe_attrs = events[probe_idx]['attrs']
                dims = list(game.pools.keys())
                # 只用部分属性作为线索（color + shape）
                partial_cue = ' '.join(probe_attrs.get(d, '') for d in dims[:2])

                narr_hit = narr_agent.test_recall(partial_cue, base_ts)
                flat_hit = flat_agent.test_recall(partial_cue, base_ts)

                narr_hits += int(narr_hit)
                flat_hits += int(flat_hit)
                total_queries += 1

        narr_acc = narr_hits / max(total_queries, 1)
        flat_acc = flat_hits / max(total_queries, 1)
        narrative_accuracies.append(narr_acc)
        flat_accuracies.append(flat_acc)
        print(f"  Run {run}: 叙事={narr_acc:.1%}, 平铺={flat_acc:.1%}")

    narr_mean = float(np.mean(narrative_accuracies))
    flat_mean = float(np.mean(flat_accuracies))
    improvement = narr_mean - flat_mean
    print(f"\n  平均：叙事={narr_mean:.1%}, 平铺={flat_mean:.1%}, "
          f"提升={improvement:.1%}")

    return {
        'narrative_recall': round(narr_mean, 4),
        'flat_recall': round(flat_mean, 4),
        'improvement': round(improvement, 4),
        'per_run': {
            'narrative': [round(x, 4) for x in narrative_accuracies],
            'flat': [round(x, 4) for x in flat_accuracies],
        },
    }


# =========================================================================
# 实验 3：记忆引导的交流
# =========================================================================
def _simple_describe(attrs: Dict[str, str],
                     known_symbols: set,
                     noise_prob: float = 0.3) -> List[str]:
    """简单描述函数：从属性中选取符号，有一定概率遗漏"""
    desc = []
    for dim in ATTR_ORDER:
        val = attrs.get(dim, '')
        if val and val in known_symbols and random.random() > noise_prob:
            desc.append(val)
    return desc


def _simple_interpret(utterance: List[str],
                      scene_features: List[Dict[str, str]]) -> Optional[int]:
    """简单解释函数：按符号匹配度选择物体"""
    if not utterance:
        return None
    utterance_set = set(utterance)
    best_idx = None
    best_score = -1

    for i, obj in enumerate(scene_features):
        obj_vals = set(obj.values())
        score = len(utterance_set & obj_vals)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def experiment_3_memory_guided_communication():
    """
    智能体利用过去成功描述改善未来交流
    200 轮，对比有记忆 vs 无记忆的交流成功率

    使用自定义描述/解释函数（而非 LanguageAgent.speak），
    以便控制噪声水平并让记忆发挥作用。
    有记忆的 speaker 从成功历史中学习更好的符号组合。
    预期：~10% 提升
    """
    print("\n" + "=" * 60)
    print("实验 3：记忆引导的交流（200 轮 × 5 运行）")
    print("=" * 60)

    num_rounds = 200
    num_runs = 5
    # 所有可能的符号集合
    all_symbols = set()
    for pool in ATTR_POOLS.values():
        all_symbols.update(pool)

    mem_success_rates = []
    nomem_success_rates = []

    for run in range(num_runs):
        rng = random.Random(200 + run)

        # 有记忆路径：积累成功符号偏好
        mem_success_symbols: Dict[str, int] = defaultdict(int)  # symbol → success count
        mem_memory = EpisodicMemory(capacity=300)

        # 无记忆路径：每轮独立
        nomem_successes = 0
        mem_successes = 0

        for r in range(num_rounds):
            scene = generate_rich_scene('medium')
            if not scene:
                continue
            target_idx = rng.randint(0, len(scene) - 1)
            target = scene[target_idx]

            # --- 有记忆路径 ---
            # 描述：偏好过去成功的符号
            mem_desc = []
            for dim in ATTR_ORDER:
                val = target.get(dim, '')
                if val:
                    # 记忆增强：过去成功使用的符号更可能被选中
                    success_count = mem_success_symbols.get(val, 0)
                    select_prob = min(0.4 + success_count * 0.06, 0.95)
                    if rng.random() < select_prob:
                        mem_desc.append(val)
            if not mem_desc:
                # 至少选一个
                for dim in ATTR_ORDER:
                    val = target.get(dim, '')
                    if val:
                        mem_desc.append(val)
                        break

            mem_chosen = _simple_interpret(mem_desc, scene)
            mem_hit = (mem_chosen == target_idx)

            # 更新记忆
            if mem_hit and mem_desc:
                for sym in mem_desc:
                    mem_success_symbols[sym] += 1
                target_feat = _feature_vector(target, ATTR_POOLS, noise=0.02)
                mem_memory.store(target_feat, mem_desc, r,
                                 metadata={'success': True})
            mem_successes += int(mem_hit)

            # --- 无记忆路径 ---
            nomem_desc = _simple_describe(target, all_symbols, noise_prob=0.3)
            nomem_chosen = _simple_interpret(nomem_desc, scene)
            nomem_hit = (nomem_chosen == target_idx)
            nomem_successes += int(nomem_hit)

        mem_sr = mem_successes / num_rounds
        nomem_sr = nomem_successes / num_rounds
        mem_success_rates.append(mem_sr)
        nomem_success_rates.append(nomem_sr)
        print(f"  Run {run}: 有记忆={mem_sr:.1%}, 无记忆={nomem_sr:.1%}")

    mem_mean = float(np.mean(mem_success_rates))
    nomem_mean = float(np.mean(nomem_success_rates))
    improvement = mem_mean - nomem_mean
    print(f"\n  平均：有记忆={mem_mean:.1%}, 无记忆={nomem_mean:.1%}, "
          f"提升={improvement:.1%}")

    return {
        'memory_guided_sr': round(mem_mean, 4),
        'no_memory_sr': round(nomem_mean, 4),
        'improvement': round(improvement, 4),
        'per_run': {
            'memory': [round(x, 4) for x in mem_success_rates],
            'no_memory': [round(x, 4) for x in nomem_success_rates],
        },
    }


# =========================================================================
# 实验 4：遗忘曲线对比
# =========================================================================
def experiment_4_forgetting_curve():
    """
    在不同间隔的干扰体验后测量回忆准确率
    延迟点: 1, 5, 10, 20, 50, 100, 150, 200, 300, 400

    特征有感知噪声，回忆时用新生成的噪声特征查询。
    语言 agent 还能用语言线索辅助检索。
    预期：语言编码记忆衰减更慢
    """
    print("\n" + "=" * 60)
    print("实验 4：遗忘曲线对比（10 延迟点 × 5 运行）")
    print("=" * 60)

    delay_points = [1, 5, 10, 20, 50, 100, 150, 200, 300, 400]
    num_runs = 5
    queries_per_delay = 10

    all_lang_curves = []
    all_base_curves = []

    for run in range(num_runs):
        game = MemoryGame(seed=300 + run)
        lang_curve = []
        base_curve = []

        for delay in delay_points:
            lang_hits = 0
            base_hits = 0

            for q in range(queries_per_delay):
                # 存储目标体验
                target_feat, target_attrs = game.generate_experience()
                target_ts = game.next_timestamp()

                lang_agent = MemoryAgent(
                    f'fc_l_{run}_{delay}_{q}', capacity=500,
                    pools=game.pools
                )
                base_agent = BaselineMemoryAgent(
                    f'fc_b_{run}_{delay}_{q}', capacity=500
                )

                lang_agent.store_experience(
                    target_feat,
                    [{'features': target_feat, 'attrs': target_attrs}],
                    target_ts,
                )
                base_agent.store_experience(target_feat, target_ts)

                # 插入干扰体验
                for _ in range(delay):
                    noise_feat, noise_attrs = game.generate_experience()
                    noise_ts = game.next_timestamp()
                    lang_agent.store_experience(
                        noise_feat,
                        [{'features': noise_feat, 'attrs': noise_attrs}],
                        noise_ts,
                    )
                    base_agent.store_experience(noise_feat, noise_ts)

                # 回忆：用新感知噪声的特征查询
                recall_feat = _feature_vector(target_attrs, game.pools)
                dims = list(game.pools.keys())
                cue_parts = [target_attrs.get(d, '') for d in dims]
                cue = ' '.join(cue_parts)

                # 语言 agent：双通道
                lang_hits += int(
                    lang_agent.test_recall_dual(cue, recall_feat, target_ts)
                )
                # 基准 agent：仅特征
                base_hits += int(
                    base_agent.test_recall(recall_feat, target_ts)
                )

            lang_curve.append(lang_hits / queries_per_delay)
            base_curve.append(base_hits / queries_per_delay)

        all_lang_curves.append(lang_curve)
        all_base_curves.append(base_curve)

        # 打印本 run 曲线
        lang_str = ', '.join(f'{v:.0%}' for v in lang_curve)
        base_str = ', '.join(f'{v:.0%}' for v in base_curve)
        print(f"  Run {run}:")
        print(f"    语言: [{lang_str}]")
        print(f"    基准: [{base_str}]")

    # 平均曲线
    avg_lang = [float(np.mean([c[i] for c in all_lang_curves]))
                for i in range(len(delay_points))]
    avg_base = [float(np.mean([c[i] for c in all_base_curves]))
                for i in range(len(delay_points))]

    print("\n  平均遗忘曲线：")
    for i, d in enumerate(delay_points):
        print(f"    delay={d:>3}: 语言={avg_lang[i]:.1%}, "
              f"基准={avg_base[i]:.1%}, "
              f"差={avg_lang[i] - avg_base[i]:+.1%}")

    # 计算衰减速率（从首点到末点的下降幅度）
    lang_decay = avg_lang[0] - avg_lang[-1]
    base_decay = avg_base[0] - avg_base[-1]

    return {
        'delay_points': delay_points,
        'language_curve': [round(x, 4) for x in avg_lang],
        'baseline_curve': [round(x, 4) for x in avg_base],
        'language_decay': round(float(lang_decay), 4),
        'baseline_decay': round(float(base_decay), 4),
        'slower_decay_with_language': bool(lang_decay < base_decay),
        'per_run': {
            'language': [[round(x, 4) for x in c] for c in all_lang_curves],
            'baseline': [[round(x, 4) for x in c] for c in all_base_curves],
        },
    }


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_language_encoding()
    results['experiment_2'] = experiment_2_narrative_structure()
    results['experiment_3'] = experiment_3_memory_guided_communication()
    results['experiment_4'] = experiment_4_forgetting_curve()

    with open('language_memory_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 language_memory_results.json")
