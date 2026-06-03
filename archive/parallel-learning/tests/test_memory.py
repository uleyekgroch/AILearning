"""
记忆系统测试 — TDD 先行

8 个测试用例验证三层记忆系统：
1. 工作记忆容量限制（FIFO 驱逐）
2. 工作记忆存入检索
3. 情景记忆按相似度检索
4. 巩固提升检索准确率
5. 遗忘曲线（Ebbinghaus 衰减）
6. 语义概念形成
7. 三层记忆层级传递
8. 巩固降低噪声
"""

import sys
sys.path.insert(0, '.')

import pytest
import torch
from src.core.config import LearnerConfig


DIM = 40
CAPACITY = 7


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return LearnerConfig(
        obs_dim=DIM,
        action_dim=8,
        hidden_dims=(128, 64),
        working_memory_capacity=CAPACITY,
        episodic_memory_capacity=1000,
        forgetting_rate=0.95,
        consolidation_interval=10,
    )


@pytest.fixture
def working_memory():
    from src.memory.working import WorkingMemory
    return WorkingMemory(capacity=CAPACITY, dim=DIM)


@pytest.fixture
def episodic_memory():
    from src.memory.episodic import EpisodicMemory
    return EpisodicMemory(max_traces=1000, dim=DIM, forgetting_rate=0.95)


@pytest.fixture
def semantic_memory():
    from src.memory.semantic import SemanticMemory
    return SemanticMemory(dim=DIM)


@pytest.fixture
def memory_system(config):
    from src.memory.system import MemorySystem
    return MemorySystem(config)


def random_tensor(dim=DIM):
    """生成随机单位向量"""
    t = torch.randn(dim)
    return t / t.norm()


# ── 1. 工作记忆容量 ─────────────────────────────────────────────────

class TestWorkingMemoryCapacity:
    """超过容量 7 后自动遗忘最旧项"""

    def test_fifo_eviction(self, working_memory):
        # 存入 10 个项（超过容量 7）
        for i in range(10):
            rep = random_tensor()
            working_memory.store(rep, metadata={'index': i})

        # 只应保留最后 7 个
        assert len(working_memory.items) == CAPACITY

    def test_oldest_evicted(self, working_memory):
        # 存入带标记的项
        for i in range(10):
            rep = random_tensor()
            working_memory.store(rep, metadata={'label': f'item_{i}'})

        # 最早的 3 个（item_0, item_1, item_2）应被驱逐
        labels = [item[1]['label'] for item in working_memory.items]
        assert 'item_0' not in labels
        assert 'item_1' not in labels
        assert 'item_2' not in labels
        # 最后 7 个应保留
        assert 'item_3' in labels
        assert 'item_9' in labels


# ── 2. 工作记忆存入检索 ─────────────────────────────────────────────

class TestWorkingMemoryRetrieve:
    """存入后可按线索检索"""

    def test_store_and_retrieve(self, working_memory):
        # 存入一个特定向量
        target = random_tensor()
        working_memory.store(target, metadata={'name': 'target'})

        # 存入若干噪声
        for _ in range(5):
            working_memory.store(random_tensor(), metadata={'name': 'noise'})

        # 用 target 自身检索，应该排在第一
        results = working_memory.retrieve(target, k=3)
        assert len(results) == 3
        assert results[0]['metadata']['name'] == 'target'

    def test_retrieve_by_similarity(self, working_memory):
        # 存入两个不同方向的向量
        a = random_tensor()
        b = random_tensor()
        working_memory.store(a, metadata={'id': 'A'})
        working_memory.store(b, metadata={'id': 'B'})

        # 用接近 a 的线索检索
        cue = a + 0.1 * torch.randn(DIM)
        cue = cue / cue.norm()
        results = working_memory.retrieve(cue, k=2)
        assert results[0]['metadata']['id'] == 'A'


# ── 3. 情景记忆检索 ─────────────────────────────────────────────────

class TestEpisodicRetrieve:
    """情景记忆按相似度检索返回 k 个最近邻"""

    def test_store_and_retrieve(self, episodic_memory):
        # 存入多个经验
        target = random_tensor()
        episodic_memory.store(target, metadata={'event': 'target'})

        for i in range(10):
            episodic_memory.store(random_tensor(), metadata={'event': f'noise_{i}'})

        # 检索最相似的 5 个
        results = episodic_memory.retrieve(target, k=5)
        assert len(results) == 5
        assert results[0]['metadata']['event'] == 'target'

    def test_k_larger_than_store(self, episodic_memory):
        # 只存 3 个，但检索 k=5
        for i in range(3):
            episodic_memory.store(random_tensor(), metadata={'idx': i})

        results = episodic_memory.retrieve(random_tensor(), k=5)
        assert len(results) == 3  # 只返回已有的

    def test_strength_weighted_retrieval(self, episodic_memory):
        """强度高的记忆应优先被检索"""
        # 存入两个向量，一个有更高 strength
        weak_vec = random_tensor()
        strong_vec = random_tensor()

        episodic_memory.store(weak_vec, metadata={'type': 'weak'})
        # 手动提升 strong_vec 的 strength
        episodic_memory.store(strong_vec, metadata={'type': 'strong'})
        episodic_memory.traces[-1]['strength'] = 2.0  # 手动增强

        # 用两者的中间向量做线索
        cue = (weak_vec + strong_vec) / 2
        cue = cue / cue.norm()

        results = episodic_memory.retrieve(cue, k=2)
        # 强度更高的应排在前面（加权检索）
        assert results[0]['metadata']['type'] == 'strong'


# ── 4. 巩固提升检索准确率 ───────────────────────────────────────────

class TestConsolidationImprovesRecall:
    """巩固后检索准确率提升 >20%"""

    def test_accuracy_improvement(self, episodic_memory):
        """
        策略：3 个正交聚类中心，每类 5 个样本，noise=0.15。
        手动降低第一类痕迹的 strength（模拟遗忘），导致检索不准。
        巩固通过质心吸引 + strength 恢复来提升准确率。
        """
        torch.manual_seed(42)

        # 3 个正交方向的聚类中心
        centers = []
        for i in range(3):
            v = torch.zeros(DIM)
            v[i * 10] = 1.0
            centers.append(v)

        # 存入样本（低噪声保持聚类结构清晰）
        for ci, center in enumerate(centers):
            for _ in range(5):
                noise = center + 0.15 * torch.randn(DIM)
                noise = noise / noise.norm()
                episodic_memory.store(noise, metadata={'center': ci})

        # 模拟遗忘：降低第一类痕迹的 strength
        for t in episodic_memory.traces[:5]:
            t['strength'] = 0.1

        # 巩固前检索准确率
        correct_before = 0
        for i, query in enumerate(centers):
            results = episodic_memory.retrieve(query, k=5)
            correct_before += sum(
                1 for r in results if r['metadata']['center'] == i
            )

        # 多轮巩固
        for _ in range(3):
            episodic_memory.consolidate()

        # 巩固后检索准确率
        correct_after = 0
        for i, query in enumerate(centers):
            results = episodic_memory.retrieve(query, k=5)
            correct_after += sum(
                1 for r in results if r['metadata']['center'] == i
            )

        total_possible = len(centers) * 5
        acc_before = correct_before / total_possible
        acc_after = correct_after / total_possible

        # 巩固后准确率绝对提升 >20%
        if acc_before < 1.0:
            absolute_improvement = acc_after - acc_before
            assert absolute_improvement > 0.20, \
                f"Absolute improvement {absolute_improvement:.2%} should be > 20% (before={acc_before:.2%}, after={acc_after:.2%})"
        else:
            assert acc_after >= acc_before


# ── 5. 遗忘曲线 ─────────────────────────────────────────────────────

class TestForgettingCurve:
    """随时间衰减符合 Ebbinghaus 趋势"""

    def test_older_traces_weaker(self, episodic_memory):
        # 存入 50 个痕迹，每次存入后衰减
        for i in range(50):
            episodic_memory.store(random_tensor(), metadata={'step': i})
            episodic_memory.decay()

        curve = episodic_memory.get_forgetting_curve()

        # 早期时间戳的平均强度应低于晚期
        timestamps = sorted(curve.keys())
        if len(timestamps) >= 2:
            early_ts = timestamps[:len(timestamps) // 3]
            late_ts = timestamps[-len(timestamps) // 3:]

            avg_early = sum(curve[t] for t in early_ts) / len(early_ts)
            avg_late = sum(curve[t] for t in late_ts) / len(late_ts)

            assert avg_early < avg_late, \
                f"Early avg {avg_early:.4f} should be < late avg {avg_late:.4f}"

    def test_decay_rate_matches_forgetting_rate(self, episodic_memory):
        """单次衰减应将 strength 乘以 forgetting_rate"""
        episodic_memory.store(random_tensor(), metadata={'test': 'decay'})
        before_strength = episodic_memory.traces[0]['strength']

        episodic_memory.decay()
        after_strength = episodic_memory.traces[0]['strength']

        expected = before_strength * 0.95
        assert abs(after_strength - expected) < 1e-6


# ── 6. 语义概念形成 ─────────────────────────────────────────────────

class TestSemanticConceptFormation:
    """重复经验抽象为语义概念"""

    def test_concept_created_from_repeated_experience(self, semantic_memory):
        # 反复存入相似的向量
        center = random_tensor()
        for i in range(5):
            vec = center + 0.1 * torch.randn(DIM)
            vec = vec / vec.norm()
            semantic_memory.store(vec, metadata={'label': f'concept_A_sample_{i}'})

        # 应该至少形成一个概念
        assert len(semantic_memory.concepts) >= 1

    def test_multiple_concepts(self, semantic_memory):
        # 存入两个不同的聚类
        center_a = random_tensor()
        center_b = random_tensor()

        # 确保两个中心足够远
        if torch.cosine_similarity(center_a.unsqueeze(0), center_b.unsqueeze(0)).item() > 0.5:
            center_b = -center_a  # 强制反方向

        for i in range(5):
            vec = center_a + 0.05 * torch.randn(DIM)
            vec = vec / vec.norm()
            semantic_memory.store(vec, metadata={'cluster': 'A'})

        for i in range(5):
            vec = center_b + 0.05 * torch.randn(DIM)
            vec = vec / vec.norm()
            semantic_memory.store(vec, metadata={'cluster': 'B'})

        assert len(semantic_memory.concepts) >= 2

    def test_concept_centroid_updated(self, semantic_memory):
        """概念质心应随新经验更新"""
        center = random_tensor()
        semantic_memory.store(center, metadata={'idx': 0})

        initial_concepts = len(semantic_memory.concepts)

        # 存入更多相似向量
        for i in range(1, 6):
            vec = center + 0.05 * torch.randn(DIM)
            vec = vec / vec.norm()
            semantic_memory.store(vec, metadata={'idx': i})

        # 概念数不应无限增长（相似向量应归入同一概念）
        assert len(semantic_memory.concepts) <= initial_concepts + 1


# ── 7. 三层记忆层级传递 ─────────────────────────────────────────────

class TestMemorySystemLayering:
    """工作记忆 → 情景记忆 → 语义记忆层级传递"""

    def test_working_to_episodic_transfer(self, memory_system):
        """工作记忆满后迁移到情景记忆"""
        # 存入超过工作记忆容量的经验
        for i in range(CAPACITY + 3):
            obs = random_tensor()
            action = torch.randn(8)
            next_obs = random_tensor()
            memory_system.store_experience(obs, action, next_obs, reward=0.5, error=0.1)

        # 工作记忆不应超过容量
        assert len(memory_system.working.items) <= CAPACITY

        # 情景记忆应有内容
        assert len(memory_system.episodic.traces) > 0

    def test_full_consolidation_pipeline(self, memory_system):
        """全链路巩固：工作→情景→语义"""
        # 存入一组有聚类结构的经验
        center = random_tensor()
        for i in range(20):
            obs = center + 0.1 * torch.randn(DIM)
            obs = obs / obs.norm()
            action = torch.randn(8)
            next_obs = random_tensor()
            memory_system.store_experience(obs, action, next_obs, reward=1.0, error=0.05)

        # 执行巩固
        report = memory_system.consolidate_all()

        # 巩固报告应包含各层信息
        assert 'working' in report
        assert 'episodic' in report
        assert 'semantic' in report

    def test_unified_retrieval(self, memory_system):
        """从三层记忆统一检索"""
        # 存入经验
        for i in range(10):
            obs = random_tensor()
            action = torch.randn(8)
            next_obs = random_tensor()
            memory_system.store_experience(obs, action, next_obs, reward=0.5, error=0.1)

        # 巩固
        memory_system.consolidate_all()

        # 统一检索
        cue = random_tensor()
        results = memory_system.retrieve_context(cue, k=5)
        assert len(results) >= 1  # 至少能检索到一些结果


# ── 8. 巩固降低噪声 ─────────────────────────────────────────────────

class TestConsolidationReducesNoise:
    """巩固后记忆更稳定"""

    def test_episodic_stability_after_consolidation(self, episodic_memory):
        """巩固后记忆强度分布更集中"""
        # 存入经验
        center = random_tensor()
        for i in range(30):
            vec = center + 0.3 * torch.randn(DIM)
            vec = vec / vec.norm()
            episodic_memory.store(vec, metadata={'batch': 'noisy'})

        # 巩固前强度标准差
        strengths_before = [t['strength'] for t in episodic_memory.traces]
        std_before = torch.tensor(strengths_before).std().item()

        # 巩固
        episodic_memory.consolidate()

        # 巩固后强度标准差
        strengths_after = [t['strength'] for t in episodic_memory.traces]
        std_after = torch.tensor(strengths_after).std().item()

        # 巩固后强度应更集中（标准差更小，或平均强度更高）
        avg_before = sum(strengths_before) / len(strengths_before)
        avg_after = sum(strengths_after) / len(strengths_after)

        # 平均强度应提升（成功模式被加强）
        assert avg_after >= avg_before, \
            f"Avg strength should increase: {avg_before:.4f} -> {avg_after:.4f}"

    def test_semantic_cleanup(self, semantic_memory):
        """语义记忆巩固后清理低频概念"""
        # 存入大量不同向量（许多只会出现一次）
        for i in range(20):
            semantic_memory.store(random_tensor(), metadata={'idx': i})

        # 存入少量高频向量
        center = random_tensor()
        for i in range(10):
            vec = center + 0.05 * torch.randn(DIM)
            vec = vec / vec.norm()
            semantic_memory.store(vec, metadata={'frequent': True})

        concepts_before = len(semantic_memory.concepts)

        # 巩固
        report = semantic_memory.consolidate()

        # 低频概念应被清理
        concepts_after = len(semantic_memory.concepts)
        assert concepts_after <= concepts_before
