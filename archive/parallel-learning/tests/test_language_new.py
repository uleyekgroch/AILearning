"""
6 个新语言模块的单元测试

覆盖：
1. InnerSpeechModule  — 构造、内部规划、序列化
2. Event / Narrative / NarrativeModule — 构造、叙事构建、连接词选择、解析
3. CrossModalGrounding — 构造、物体管理与消除歧义
4. AttentionModulator  — 构造、注意力调制、符号学习
5. DualCodingMemory    — 编码、语言/感知回忆、遗忘曲线
6. ContinuousConceptSpace — 观察、分类、模糊分类、对齐
"""

import sys

sys.path.insert(0, ".")

import torch
import pytest

from src.language.inner_speech import InnerSpeechModule
from src.language.narrative import Event, Narrative, NarrativeModule, NARRATIVE_MARKERS
from src.language.crossmodal import CrossModalGrounding, CrossModalObject, MODALITY_TYPES
from src.language.attention import AttentionModulator
from src.language.memory_scaffold import DualCodingMemory
from src.language.continuous_concepts import ContinuousConceptSpace


# =====================================================================
# 1. InnerSpeechModule
# =====================================================================


class TestInnerSpeechModule:
    """内部言语模块测试"""

    def test_construction_and_defaults(self):
        mod = InnerSpeechModule(obs_dim=32)
        assert mod.obs_dim == 32
        assert mod._encoder.shape == (32, 32)
        assert mod._attention_weights.shape == (32,)
        assert mod.get_planning_benefit() == 0.0

    def test_plan_description_basic(self):
        mod = InnerSpeechModule(obs_dim=20)

        scene = [
            {"color": "red", "shape": "circle"},
            {"color": "blue", "shape": "square"},
        ]
        # target_idx=0 → 目标是 red circle；"red" 和 "circle" 都是唯一的
        symbols = mod.plan_description(scene, target_idx=0)
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert mod._planning_log  # 规划日志有记录

    def test_plan_description_edge_cases(self):
        mod = InnerSpeechModule()
        # 空场景
        assert mod.plan_description([], 0) == []
        # target_idx 越界
        assert mod.plan_description([{"a": "1"}], 5) == []

    def test_encode_scene(self):
        mod = InnerSpeechModule(obs_dim=16)
        scene = [{"x": "1"}, {"x": "2"}]
        encoded = mod.encode_scene(scene)
        assert encoded.shape == (2, 16)

    def test_record_outcome_and_benefit(self):
        mod = InnerSpeechModule()
        # 规划成功 3 次
        for _ in range(3):
            mod.record_outcome(success=True, used_planning=True)
        # 无规划失败 3 次
        for _ in range(3):
            mod.record_outcome(success=False, used_planning=False)
        # 规划成功率 = 3/3 = 1.0; 无规划成功率 = 0/3 = 0.0; 差值 = 1.0
        assert mod.get_planning_benefit() == pytest.approx(1.0)

    def test_save_load_state(self):
        mod = InnerSpeechModule(obs_dim=8)
        mod.learn_mapping("color:red", "red")
        state = mod.save_state()
        mod2 = InnerSpeechModule(obs_dim=8)
        mod2.load_state(state)
        assert mod2._feature_symbol_map == {"color:red": ["red"]}


# =====================================================================
# 2. NarrativeModule / Event / Narrative
# =====================================================================


class TestEvent:
    """Event 数据类测试"""

    def test_construction_and_to_symbols(self):
        ev = Event(subject={"entity": "agent"}, action="push",
                   object_features={"entity": "block"})
        syms = ev.to_symbols()
        assert "agent" in syms
        assert "push" in syms
        assert "block" in syms

    def test_empty_object_features(self):
        ev = Event(subject={"who": "bob"}, action="walk")
        assert ev.object_features == {}
        syms = ev.to_symbols()
        assert "bob" in syms
        assert "walk" in syms


class TestNarrative:
    """Narrative 数据类测试"""

    def test_length_and_to_symbols(self):
        e1 = Event({"e": "a"}, "go")
        e2 = Event({"e": "a"}, "stop")
        nar = Narrative([e1, e2], ["then"])
        assert nar.length == 2
        syms = nar.to_symbols()
        # 应包含两事件符号 + 连接词
        assert "then" in syms

    def test_empty_narrative(self):
        nar = Narrative([], [])
        assert nar.length == 0
        assert nar.to_symbols() == []


class TestNarrativeModule:
    """NarrativeModule 核心测试"""

    def test_build_narrative_temporal(self):
        mod = NarrativeModule()
        # 用不同主体避免触发对比规则，纯粹测试时序
        events = [
            {"subject": {"entity": "cat"}, "action": "run", "time": 1},
            {"subject": {"entity": "dog"}, "action": "bark", "time": 2},
        ]
        nar = mod.build_narrative(events)
        assert nar.length == 2
        assert len(nar.connections) == 1
        # time 1 < 2 → 时序关系 → "then"
        assert nar.connections[0] == "then"

    def test_build_narrative_causal(self):
        mod = NarrativeModule()
        events = [
            {"subject": {"entity": "bob"}, "action": "push"},
            {"subject": {"entity": "block"}, "action": "fall"},
        ]
        nar = mod.build_narrative(events)
        # push → fall 是预设因果对
        assert nar.connections[0] == "so"

    def test_parse_narrative(self):
        mod = NarrativeModule()
        symbols = ["cat", "run", "away", "then", "dog", "chase", "cat"]
        nar = mod.parse_narrative(symbols)
        assert nar.length == 2
        assert nar.connections == ["then"]

    def test_empty_build(self):
        mod = NarrativeModule()
        nar = mod.build_narrative([])
        assert nar.length == 0


# =====================================================================
# 3. CrossModalGrounding
# =====================================================================


class TestCrossModalGrounding:
    """跨模态接地模块测试"""

    def test_construction(self):
        cm = CrossModalGrounding()
        assert isinstance(cm._modality_weights, torch.Tensor)
        assert len(cm._modality_names) == 4  # auditory, olfactory, tactile, visual
        assert cm.get_modality_contributions() == {m: 0.0 for m in MODALITY_TYPES}

    def test_add_and_get_objects(self):
        cm = CrossModalGrounding()
        obj = CrossModalObject(
            features={
                "visual": {"color": "red"},
                "auditory": {"sound": "loud"},
            },
            obj_id="obj1",
        )
        cm.add_object(obj)
        stored = cm.get_objects()
        assert len(stored) == 1
        assert stored[0].obj_id == "obj1"
        assert cm.get_symbol_modality("red") == "visual"
        assert cm.get_symbol_modality("loud") == "auditory"

    def test_disambiguate_visual_clear(self):
        cm = CrossModalGrounding()
        obj_a = CrossModalObject({"visual": {"color": "red"}}, obj_id="a")
        obj_b = CrossModalObject({"visual": {"color": "blue"}}, obj_id="b")
        symbols = cm.disambiguate([obj_a, obj_b], visual_ambiguous=False)
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        contribs = cm.get_modality_contributions()
        assert contribs["visual"] > 0

    def test_disambiguate_visual_ambiguous(self):
        cm = CrossModalGrounding()
        # 两个视觉相同但听觉不同的物体
        obj_a = CrossModalObject(
            {"visual": {"color": "red"}, "auditory": {"sound": "loud"}},
            obj_id="a",
        )
        obj_b = CrossModalObject(
            {"visual": {"color": "red"}, "auditory": {"sound": "quiet"}},
            obj_id="b",
        )
        symbols = cm.disambiguate([obj_a, obj_b], visual_ambiguous=True)
        # 听觉符号应涌现
        assert len(symbols) > 0
        cross_syms = cm.get_crossmodal_symbols()
        assert len(cross_syms) > 0


# =====================================================================
# 4. AttentionModulator
# =====================================================================


class TestAttentionModulator:
    """语言引导注意力模块测试"""

    def test_construction_and_default_weights(self):
        att = AttentionModulator()
        weights = att.get_weights()
        assert set(weights.keys()) == {"visual", "audio", "position"}
        # 默认权重经 softmax 后总和 ≈ 1
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_modulate_with_known_symbols(self):
        att = AttentionModulator(visual_dim=10, audio_dim=5, position_dim=2)
        obs = {
            "visual": torch.randn(10),
            "audio": torch.randn(5),
            "position": torch.randn(2),
        }
        result = att.modulate(obs, known_symbols=["red"])
        # "red" → visual boost → visual 权重应增大
        assert "visual" in result
        weights = att.get_weights()
        assert weights["visual"] > weights["audio"]

    def test_learn_symbol_modality(self):
        att = AttentionModulator()
        att.learn_symbol_modality("beep", "audio")
        att.add_known_symbol("beep")
        sym_map = att.get_symbol_modality_map()
        assert sym_map["beep"] == "audio"
        assert "beep" in att.get_known_symbols()

    def test_save_load_state(self):
        att = AttentionModulator()
        att.learn_symbol_modality("siren", "audio")
        state = att.save_state()
        att2 = AttentionModulator()
        att2.load_state(state)
        assert att2.get_symbol_modality_map()["siren"] == "audio"


# =====================================================================
# 5. DualCodingMemory
# =====================================================================


class TestDualCodingMemory:
    """双编码记忆支架测试"""

    def test_encode_and_symbol_recall(self):
        mem = DualCodingMemory(feature_dim=8)
        vec = torch.randn(8)
        mem.encode({"vector": vec}, symbols=["apple", "fruit"])
        results = mem.recall_with_symbols(["apple"], k=5)
        assert len(results) == 1
        assert results[0]["score"] > 0
        assert "apple" in results[0]["entry"]["symbols"]

    def test_perception_recall(self):
        mem = DualCodingMemory(feature_dim=8)
        vec = torch.randn(8)
        mem.encode({"vector": vec}, symbols=["ball"])
        # 用相同向量检索
        results = mem.recall_with_perception({"vector": vec}, k=5)
        assert len(results) == 1
        assert results[0]["score"] > 0.9  # 完全匹配

    def test_capacity_eviction(self):
        mem = DualCodingMemory(capacity=3, feature_dim=4)
        for i in range(5):
            mem.encode(
                {"vector": torch.randn(4)},
                symbols=[f"s{i}"],
            )
        # 容量=3，存了5条，应该只剩3条
        assert len(mem._entries) == 3

    def test_forgetting_curve(self):
        mem = DualCodingMemory(feature_dim=4)
        mem.encode({"vector": torch.randn(4)}, symbols=["cat"])
        # 存入更多经验以产生时间差
        for _ in range(10):
            mem.encode({"vector": torch.randn(4)}, symbols=[])
        curve = mem.get_forgetting_curve("cat")
        assert len(curve) == 1
        assert 0.0 <= curve[0] <= 1.0

    def test_dual_coding_benefit(self):
        mem = DualCodingMemory(feature_dim=8)
        # 双编码（有符号）
        mem.encode({"vector": torch.randn(8)}, symbols=["tagged"])
        # 单编码（无符号）
        mem.encode({"vector": torch.randn(8)}, symbols=[])
        # 触发检索统计
        mem.recall_with_symbols(["tagged"])
        benefit = mem.get_dual_coding_benefit()
        assert 0.0 <= benefit <= 1.0


# =====================================================================
# 6. ContinuousConceptSpace
# =====================================================================


class TestContinuousConceptSpace:
    """连续概念空间测试"""

    def test_observe_creates_categories(self):
        space = ContinuousConceptSpace(feature_dim=4)
        # 第一次观察 → 自动创建类别
        label = space.observe(torch.randn(4))
        assert label.startswith("C")
        assert len(space.get_prototypes()) == 1

    def test_observe_with_label(self):
        space = ContinuousConceptSpace(feature_dim=3)
        lbl = space.observe(torch.randn(3), label="red")
        assert lbl == "red"
        assert "red" in space.get_prototypes()

    def test_categorize_groups_similar(self):
        space = ContinuousConceptSpace(feature_dim=3, device="cpu")
        # 两个相近向量应归为同一类
        base = torch.tensor([1.0, 0.0, 0.0])
        similar = base + torch.randn(3) * 0.1
        space.observe(base, label="A")
        label = space.observe(similar)
        assert label == "A"

    def test_fuzzy_classify(self):
        space = ContinuousConceptSpace(feature_dim=3)
        space.observe(torch.tensor([1.0, 0.0, 0.0]), label="x")
        space.observe(torch.tensor([0.0, 1.0, 0.0]), label="y")
        probs = space.fuzzy_classify(torch.tensor([0.9, 0.1, 0.0]))
        assert "x" in probs
        assert "y" in probs
        assert probs["x"] > probs["y"]  # 更接近 x 原型

    def test_save_load_state(self):
        space = ContinuousConceptSpace(feature_dim=3)
        space.observe(torch.randn(3), label="cat")
        state = space.save_state()
        space2 = ContinuousConceptSpace(feature_dim=3)
        space2.load_state(state)
        assert "cat" in space2.get_prototypes()
