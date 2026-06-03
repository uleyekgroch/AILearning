"""
新模块单元测试 — adversarial / planning / nonstationary / registry / serializable
"""

import sys
sys.path.insert(0, '.')

import pytest
import torch

from src.social.adversarial import TrustEvaluator, ReputationSystem
from src.social.planning import CooperativePlanner, CooperativeTask, TaskStep
from src.core.nonstationary import NonstationaryAdapter, EnvironmentRegime, REGIME_SEQUENCES
from src.core.registry import ModuleRegistry
from src.core.serializable import Serializable


# ══════════════════════════════════════════════════════════════════════
# TrustEvaluator
# ══════════════════════════════════════════════════════════════════════

class TestTrustEvaluator:

    def test_initial_trust_for_unknown_source(self):
        te = TrustEvaluator(initial_trust=0.5)
        assert te.get_trust("unknown") == pytest.approx(0.5)

    def test_update_increases_trust_for_honest(self):
        te = TrustEvaluator(initial_trust=0.5, lr=0.1)
        for _ in range(10):
            te.update("agent_a", was_honest=True)
        trust = te.get_trust("agent_a")
        assert trust > 0.8

    def test_update_decreases_trust_for_dishonest(self):
        te = TrustEvaluator(initial_trust=0.5)
        for _ in range(10):
            te.update("agent_b", was_honest=False)
        trust = te.get_trust("agent_b")
        assert trust < 0.3

    def test_get_reliable_sources_filters_by_threshold(self):
        te = TrustEvaluator()
        # agent_x: honest many times → high trust
        for _ in range(20):
            te.update("agent_x", was_honest=True)
        # agent_y: dishonest many times → low trust
        for _ in range(20):
            te.update("agent_y", was_honest=False)
        reliable = te.get_reliable_sources(threshold=0.7)
        assert "agent_x" in reliable
        assert "agent_y" not in reliable

    def test_save_load_roundtrip(self):
        te = TrustEvaluator(initial_trust=0.6, lr=0.2)
        te.update("a", True)
        te.update("a", False)
        state = te.save_state()

        te2 = TrustEvaluator()
        te2.load_state(state)
        assert te2.initial_trust == 0.6
        assert te2.lr == 0.2
        assert te2.get_trust("a") == pytest.approx(te.get_trust("a"))


# ══════════════════════════════════════════════════════════════════════
# ReputationSystem
# ══════════════════════════════════════════════════════════════════════

class TestReputationSystem:

    def test_initial_reputation_is_neutral(self):
        rs = ReputationSystem()
        # unknown agent defaults to 0.5
        assert rs.get_reputation("unknown") == pytest.approx(0.5)

    def test_record_interaction_updates_reputation(self):
        rs = ReputationSystem()
        # honest & successful interactions → reputation > 0.5
        for _ in range(10):
            rs.record_interaction("speaker_1", "listener_1",
                                  success=True, was_honest=True)
        rep = rs.get_reputation("speaker_1")
        assert rep > 0.5

    def test_choose_speaker_prefers_high_reputation(self):
        rs = ReputationSystem()
        # speaker_good: consistently honest
        for _ in range(20):
            rs.record_interaction("good", "l1", True, True)
        # speaker_bad: consistently dishonest
        for _ in range(20):
            rs.record_interaction("bad", "l1", False, False)

        # Sample many times; good should be chosen overwhelmingly often
        counts = {"good": 0, "bad": 0}
        for _ in range(200):
            chosen = rs.choose_speaker(["good", "bad"])
            counts[chosen] += 1
        assert counts["good"] > counts["bad"]

    def test_save_load_roundtrip(self):
        rs = ReputationSystem(decay=0.99)
        rs.record_interaction("s1", "l1", True, True)
        rs.step()
        state = rs.save_state()

        rs2 = ReputationSystem()
        rs2.load_state(state)
        assert rs2.decay == 0.99
        assert rs2.get_reputation("s1") == pytest.approx(rs.get_reputation("s1"))


# ══════════════════════════════════════════════════════════════════════
# CooperativePlanner
# ══════════════════════════════════════════════════════════════════════

class TestCooperativePlanner:

    @pytest.fixture
    def planner(self):
        return CooperativePlanner(max_agents=4)

    def test_construction_defaults(self, planner):
        assert planner.max_agents == 4

    def test_decompose_task_distributes_steps(self, planner):
        task = CooperativeTask(
            task_id="test_task",
            steps=[
                TaskStep("i_push", "push", "rock", 1),
                TaskStep("you_carry", "carry", "fruit", 1),
            ],
            required_agents=2,
        )
        groups = planner.decompose_task(task, num_agents=2)
        # All steps should appear across groups
        total_steps = sum(len(g) for g in groups)
        assert total_steps == 2

    def test_decompose_task_empty_with_zero_agents(self, planner):
        task = CooperativeTask(task_id="t", steps=[], required_agents=0)
        assert planner.decompose_task(task, num_agents=0) == []

    def test_assign_roles_balances_load(self, planner):
        steps = [
            TaskStep("i_push", "push", "rock", 1),
            TaskStep("you_carry", "carry", "fruit", 1),
            TaskStep("i_lift", "lift", "log", 1),
        ]
        agents = ["a1", "a2"]
        assignment = planner.assign_roles(steps, agents)
        # Both agents should receive at least one step
        assert all(len(v) > 0 for v in assignment.values())

    def test_create_task_library_returns_tasks(self, planner):
        library = planner.create_task_library()
        assert len(library) > 0
        assert all(isinstance(t, CooperativeTask) for t in library)


# ══════════════════════════════════════════════════════════════════════
# NonstationaryAdapter
# ══════════════════════════════════════════════════════════════════════

class TestNonstationaryAdapter:

    def test_detect_regime_change(self):
        adapter = NonstationaryAdapter(
            regime_sequence='seasons',
            decay_rate=0.1,
            reactivation_threshold=0.1,
        )
        # seasons: each regime lasts 200 steps
        # After 200 steps, should detect regime change
        new_regime = None
        for _ in range(200):
            new_regime = adapter.step()
        assert new_regime == "summer"

    def test_no_change_within_regime(self):
        adapter = NonstationaryAdapter(regime_sequence='seasons')
        result = adapter.step()
        assert result is None

    def test_regime_cycles_back(self):
        adapter = NonstationaryAdapter(regime_sequence='day_night')
        # day_night: 100 + 50 + 100 + 50 = 300 steps per cycle
        for _ in range(300):
            adapter.step()
        assert adapter.get_current_regime().name == "day"
        assert adapter.get_cycle_count() == 1

    def test_forget_obsolete_reduces_unrelated_vocab(self):
        adapter = NonstationaryAdapter(regime_sequence='seasons')
        vocab = {"green": 1.0, "hot_summer_word": 0.8, "cold_winter_word": 0.6}
        updated = adapter.forget_obsolete(vocab, rate=0.5)
        # "green" matches 'spring' features → should survive
        assert "green" in updated
        # Words not matching current regime features should decay
        assert updated.get("hot_summer_word", 1.0) < 0.8
        assert updated.get("cold_winter_word", 1.0) < 0.6

    def test_save_load_roundtrip(self):
        adapter = NonstationaryAdapter(regime_sequence='seasons')
        for _ in range(50):
            adapter.step()
        state = adapter.save_state()

        adapter2 = NonstationaryAdapter(regime_sequence='seasons')
        adapter2.load_state(state)
        assert adapter2.get_current_regime().name == adapter.get_current_regime().name


# ══════════════════════════════════════════════════════════════════════
# ModuleRegistry
# ══════════════════════════════════════════════════════════════════════

class TestModuleRegistry:

    def test_register_and_get(self):
        reg = ModuleRegistry()
        reg.register("counter", lambda: {"value": 0})
        obj = reg.get("counter")
        assert obj == {"value": 0}

    def test_has_reflects_initialization(self):
        reg = ModuleRegistry()
        reg.register("mod", list)
        assert not reg.has("mod")
        reg.get("mod")
        assert reg.has("mod")

    def test_get_raises_on_unknown(self):
        reg = ModuleRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_set_allows_direct_instance(self):
        reg = ModuleRegistry()
        instance = [1, 2, 3]
        reg.set("my_list", instance)
        assert reg.get("my_list") is instance
        assert reg.has("my_list")

    def test_initialized_property(self):
        reg = ModuleRegistry()
        reg.register("a", int)
        reg.register("b", str)
        reg.get("a")
        init = reg.initialized
        assert "a" in init
        assert "b" not in init


# ══════════════════════════════════════════════════════════════════════
# Serializable mixin
# ══════════════════════════════════════════════════════════════════════

class TestSerializable:

    def test_save_state_captures_public_attrs(self):
        class Simple(Serializable):
            def __init__(self):
                self.x = 10
                self.name = "test"
                self._private = "hidden"

        obj = Simple()
        state = obj.save_state()
        assert state == {"x": 10, "name": "test"}
        assert "_private" not in state

    def test_load_state_restores_attrs(self):
        class Simple(Serializable):
            def __init__(self):
                self.x = 0
                self.name = ""

        obj = Simple()
        obj.load_state({"x": 42, "name": "restored"})
        assert obj.x == 42
        assert obj.name == "restored"

    def test_tensor_roundtrip(self):
        class TensorHolder(Serializable):
            def __init__(self):
                self.data = torch.tensor([1.0, 2.0, 3.0])

        obj = TensorHolder()
        state = obj.save_state()
        assert state["data"]["__tensor__"] is True

        obj2 = TensorHolder()
        obj2.load_state(state)
        assert torch.allclose(obj2.data, torch.tensor([1.0, 2.0, 3.0]))
