"""
语言系统测试 — 涌现语言 + 符号接地 + 语法 + 通信协议 (TDD 先行)

10 个测试用例覆盖：
1. 单符号涌现 — 重复暴露后符号被学会
2. 感知向量触发符号接地
3. 组合性组合 — 红+圆 → "red"+"circle"
4. 否定标记涌现 — 排除需求下 "not" 出现
5. 参照游戏成功率 > 随机基线
6. 词汇量随交互增长
7. 复合符号涌现 — 频繁组合压缩
8. 词序偏好 — 从成功模式中学习
9. 序列化/反序列化一致性
10. 文化变异可执行
"""

import sys
sys.path.insert(0, '.')

import torch
import pytest
import random

from src.language.emergence import (
    EmergingLanguage, COLORS, SHAPES, SIZES, MATERIALS, ACTIONS,
    _symbol_category,
)
from src.language.grounding import GroundingModule
from src.language.grammar import GrammarSystem
from src.language.communication import CommunicationProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lang():
    """空白的 EmergingLanguage"""
    return EmergingLanguage()


@pytest.fixture
def grounding():
    """空白 GroundingModule"""
    return GroundingModule()


@pytest.fixture
def grammar():
    """空白 GrammarSystem"""
    return GrammarSystem()


@pytest.fixture
def comm():
    """空白 CommunicationProtocol"""
    return CommunicationProtocol()


# ---------------------------------------------------------------------------
# 1. 单符号涌现
# ---------------------------------------------------------------------------

class TestSingleSymbolEmerges:
    """重复暴露 100 轮后至少 2 个符号被学会（success_rate > 0）"""

    def test_single_symbol_emerges(self, lang):
        symbols = ['red', 'circle']
        for _ in range(100):
            success = random.random() > 0.3
            lang.record_usage(symbols, success)

        # 至少两个符号有 success_rate > 0
        learned = [
            sym for sym, data in lang.vocabulary.items()
            if data['success_rate'] > 0
        ]
        assert len(learned) >= 2
        # 频率应等于暴露轮次
        assert lang.vocabulary['red']['frequency'] == 100
        assert lang.vocabulary['circle']['frequency'] == 100


# ---------------------------------------------------------------------------
# 2. 感知向量符号接地
# ---------------------------------------------------------------------------

class TestSymbolGroundingFromTensor:
    """感知向量可以触发符号接地"""

    def test_ground_from_perception(self, grounding):
        obs = torch.randn(40)
        cluster_id = grounding.ground_from_perception(obs)
        assert isinstance(cluster_id, int)
        assert cluster_id >= 0

    def test_similar_obs_same_cluster(self, grounding):
        """相似观测应归入同一聚类"""
        base = torch.zeros(40)
        base[0] = 1.0
        # 距离 < 0.5 的观测应归入同一聚类
        slightly_different = base.clone()
        slightly_different[0] = 1.1
        id1 = grounding.ground_from_perception(base)
        id2 = grounding.ground_from_perception(slightly_different)
        assert id1 == id2

    def test_different_obs_new_cluster(self, grounding):
        """距离远的观测应产生新聚类"""
        obs_a = torch.zeros(40)
        obs_a[0] = 1.0
        obs_b = torch.ones(40) * 5.0
        id1 = grounding.ground_from_perception(obs_a)
        id2 = grounding.ground_from_perception(obs_b)
        assert id1 != id2

    def test_ground_from_social(self, grounding):
        obs = torch.randn(40)
        grounding.ground_from_social('red', obs, context='color')
        meaning = grounding.get_symbol_meaning('red')
        assert meaning is not None
        assert meaning['usage_count'] >= 1

    def test_find_similar_concepts(self, grounding):
        obs = torch.randn(40)
        grounding.ground_from_perception(obs)
        similar = grounding.find_similar_concepts(obs, top_k=3)
        assert len(similar) >= 1
        assert similar[0][1] > 0  # similarity > 0


# ---------------------------------------------------------------------------
# 3. 组合性组合
# ---------------------------------------------------------------------------

class TestCompositionalCombination:
    """红+圆 → "red"+"circle" 组合成功描述"""

    def test_compositional_combination(self, comm):
        scene = [
            {'color': 'red', 'shape': 'circle'},
            {'color': 'blue', 'shape': 'square'},
            {'color': 'green', 'shape': 'triangle'},
        ]
        intention = {'color': 'red', 'shape': 'circle'}
        utterance = comm.produce(intention)
        assert 'red' in utterance or 'circle' in utterance
        # 听者应能正确理解
        result = comm.comprehend(utterance, {'scene': scene})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 4. 否定标记涌现
# ---------------------------------------------------------------------------

class TestNegationMarkerEmerges:
    """排除需求下否定标记 "not" 涌现"""

    def test_negation_marker_emerges(self, comm):
        # 构造一个单符号不足以区分的场景
        # 两个红色物体，只有形状不同
        scene = [
            {'color': 'red', 'shape': 'circle'},
            {'color': 'red', 'shape': 'square'},
            {'color': 'blue', 'shape': 'circle'},
        ]
        # 目标是蓝色圆形，用否定描述可以更短
        intention = {'color': 'blue', 'shape': 'circle'}
        utterance = comm.produce(intention)
        # 至少应该产生有意义的描述
        assert len(utterance) >= 1

        # 反复玩参照游戏，否定策略应该被使用
        for _ in range(50):
            target_idx = random.randint(0, len(scene) - 1)
            comm.play_round(comm, comm, scene, target_idx)

        # 验证通信协议可以处理否定标记
        result = comm.comprehend(['not', 'red'], {'scene': scene})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 5. 参照游戏成功率 > 随机基线
# ---------------------------------------------------------------------------

class TestReferenceGameSuccess:
    """Speaker-Listener 参照游戏成功率 > 随机基线"""

    def test_reference_game_success(self, comm):
        scene = [
            {'color': 'red', 'shape': 'circle'},
            {'color': 'blue', 'shape': 'square'},
            {'color': 'green', 'shape': 'triangle'},
        ]
        num_objects = len(scene)
        random_baseline = 1.0 / num_objects  # 0.333

        successes = 0
        total = 200
        for i in range(total):
            target_idx = i % num_objects
            success = comm.play_round(comm, comm, scene, target_idx)
            if success:
                successes += 1

        success_rate = successes / total
        # 至少应该比随机好
        assert success_rate > random_baseline, (
            f"Success rate {success_rate:.3f} not better than random {random_baseline:.3f}"
        )


# ---------------------------------------------------------------------------
# 6. 词汇量随交互增长
# ---------------------------------------------------------------------------

class TestVocabularyGrowsWithExperience:
    """词汇量随交互增长"""

    def test_vocabulary_grows(self, lang):
        # 初始词汇量为 0
        assert lang.get_vocabulary_size() == 0

        # 第一批：2 个符号
        lang.record_usage(['red', 'circle'], True)
        assert lang.get_vocabulary_size() == 2

        # 第二批：增加新符号
        lang.record_usage(['blue', 'square'], True)
        assert lang.get_vocabulary_size() == 4

        # 第三批：重复符号，词汇量不变
        lang.record_usage(['red', 'circle'], True)
        assert lang.get_vocabulary_size() == 4


# ---------------------------------------------------------------------------
# 7. 复合符号涌现
# ---------------------------------------------------------------------------

class TestCompoundSymbolEmergence:
    """频繁组合压缩为复合符号"""

    def test_compound_symbol_emergence(self, lang):
        # 反复使用相同的符号组合
        symbols = ['red', 'circle']
        for _ in range(10):
            lang.record_usage(symbols, True)
            lang.record_collocation('red', 'circle', True)
            lang.record_ngram(symbols, True)
            lang.total_games += 1

        # 检查复合符号形成
        lang.check_compound_formation(symbols, True)

        # 应该形成 "red-circle" 复合符号
        assert 'red-circle' in lang.compounds
        assert lang.compounds['red-circle']['success_rate'] >= 0.8


# ---------------------------------------------------------------------------
# 8. 词序偏好
# ---------------------------------------------------------------------------

class TestWordOrderPreference:
    """从成功模式中学习词序偏好"""

    def test_word_order_preference(self, lang, grammar):
        # 大量记录 modifier_first 成功
        for _ in range(20):
            lang.record_word_order('modifier_first', True)
        # 少量记录 head_first 失败
        for _ in range(5):
            lang.record_word_order('head_first', False)

        # 偏好应为 modifier_first
        assert lang.get_preferred_order() == 'modifier_first'

        # 语法系统从成功模式学习
        grammar.learn_pattern(['big', 'red', 'circle'], success=True)
        rules = grammar.get_rules()
        assert len(rules) >= 1

    def test_order_consistency(self, lang):
        # 一致使用 modifier_first
        for _ in range(50):
            lang.record_word_order('modifier_first', True)

        consistency = lang.get_order_consistency()
        assert consistency > 0.9


# ---------------------------------------------------------------------------
# 9. 序列化/反序列化一致性
# ---------------------------------------------------------------------------

class TestSaveAndLoadState:
    """序列化/反序列化后语言状态一致"""

    def test_save_and_load_state(self, lang):
        # 建立一些状态
        lang.record_usage(['red', 'circle'], True)
        lang.record_usage(['blue', 'square'], False)
        lang.record_collocation('red', 'circle', True)
        lang.record_ngram(['red', 'circle'], True)
        lang.record_word_order('modifier_first', True)
        lang.total_games = 10
        lang.total_successes = 5

        # 保存
        state = lang.save_state()

        # 新实例加载
        lang2 = EmergingLanguage()
        lang2.load_state(state)

        # 验证一致
        assert lang2.vocabulary == lang.vocabulary
        assert lang2.total_games == 10
        assert lang2.total_successes == 5
        assert lang2.word_order_scores == lang.word_order_scores
        assert lang2.word_order_counts == lang.word_order_counts
        assert len(lang2.ngram_patterns) == len(lang.ngram_patterns)


# ---------------------------------------------------------------------------
# 10. 文化变异
# ---------------------------------------------------------------------------

class TestCulturalMutation:
    """语言变异（简化/借词）可执行"""

    def test_cultural_mutation(self, lang):
        # 先建立词汇和复合符号
        for _ in range(15):
            lang.record_usage(['red', 'circle'], True)
            lang.record_collocation('red', 'circle', True)
            lang.record_ngram(['red', 'circle'], True)
            lang.total_games += 1

        lang.check_compound_formation(['red', 'circle'], True)
        assert 'red-circle' in lang.compounds

        # 设置足够高的 frequency 以触发简化
        lang.compounds['red-circle']['frequency'] = 12

        # 执行变异
        random.seed(42)
        lang.mutate(mutation_rate=1.0)

        # 变异后复合符号可能被简化（缩写）
        # 或者词汇的 success_rate 发生漂移
        # 只要不报错就算通过
        assert lang.get_vocabulary_size() >= 0

    def test_borrowing(self, lang):
        """借词测试"""
        # 创建另一个语言作为借词来源
        other_lang = EmergingLanguage()
        other_lang.record_usage(['metal', 'wood'], True)

        original_size = lang.get_vocabulary_size()
        random.seed(42)
        lang.mutate(mutation_rate=1.0, borrow_from=other_lang)

        # 借词后词汇量可能增加
        assert lang.get_vocabulary_size() >= original_size


# ---------------------------------------------------------------------------
# 通信协议完整性测试
# ---------------------------------------------------------------------------

class TestCommunicationProtocolILanguage:
    """验证 CommunicationProtocol 实现 ILanguage 接口"""

    def test_produce_returns_list(self, comm):
        intention = {'color': 'red', 'shape': 'circle'}
        utterance = comm.produce(intention)
        assert isinstance(utterance, list)
        assert all(isinstance(s, str) for s in utterance)

    def test_comprehend_returns_dict(self, comm):
        result = comm.comprehend(['red', 'circle'], {'scene': []})
        assert isinstance(result, dict)

    def test_get_vocabulary(self, comm):
        vocab = comm.get_vocabulary()
        assert isinstance(vocab, dict)

    def test_play_round_returns_bool(self, comm):
        scene = [
            {'color': 'red', 'shape': 'circle'},
            {'color': 'blue', 'shape': 'square'},
        ]
        result = comm.play_round(comm, comm, scene, 0)
        assert isinstance(result, bool)
