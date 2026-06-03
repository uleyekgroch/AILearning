"""Tests for the 6 new reasoning modules.

Covers:
  - theory_of_mind  (Perspective, TheoryOfMindModule)
  - causal          (CausalRule, CausalReasoningModule)
  - counterfactual  (CounterfactualModule, CounterfactualWorld)
  - tool_use        (ToolUseModule, Tool, Goal)
  - questioning     (QuestioningModule)
  - metaphor        (MetaphorTracker)
"""

import pytest

from src.reasoning import (
    Perspective,
    TheoryOfMindModule,
    PERSPECTIVE_MARKERS,
    CausalRule,
    CausalReasoningModule,
    CounterfactualModule,
    CounterfactualWorld,
    COUNTERFACTUAL_MARKERS,
    ToolUseModule,
    Tool,
    Goal,
    TOOL_TEMPLATES,
    GOAL_TEMPLATES,
    QuestioningModule,
    QUESTION_MARKERS,
    MetaphorTracker,
    METAPHOR_MAPPINGS,
)


# ── theory_of_mind ──────────────────────────────────────────────────


class TestPerspective:
    def test_creation_defaults(self):
        p = Perspective("agent_a")
        assert p.agent_id == "agent_a"
        assert p.known_facts == set()
        assert p.beliefs == {}
        assert p.attention == set()

    def test_observe_and_knows(self):
        p = Perspective("x")
        assert not p.knows("sky is blue")
        p.observe("sky is blue", confidence=0.9)
        assert p.knows("sky is blue")
        assert p.get_confidence("sky is blue") == 0.9
        # Unknown proposition returns neutral 0.5
        assert p.get_confidence("unknown") == 0.5

    def test_forget_and_save_load(self):
        p = Perspective("a")
        p.observe("fact1")
        p.observe("fact2")
        p.attention.add("focus1")
        p.forget("fact1")
        assert not p.knows("fact1")
        assert p.knows("fact2")

        state = p.save_state()
        p2 = Perspective("b")
        p2.load_state(state)
        assert p2.agent_id == "a"
        assert p2.knows("fact2")
        assert not p2.knows("fact1")


class TestTheoryOfMindModule:
    def test_model_other_and_asymmetry(self):
        tom = TheoryOfMindModule()
        tom.own_perspective.observe("secret")
        tom.model_other("bob", {"public_fact"})

        assert tom.estimate_other_knowledge("bob", "public_fact") == 1.0
        assert tom.estimate_other_knowledge("bob", "secret") == 0.0
        # Self knows secret, bob doesn't -> asymmetry
        assert tom.detect_asymmetry("bob", "secret") is True
        # Both know public_fact -> no asymmetry
        assert tom.detect_asymmetry("bob", "public_fact") is False

    def test_adjust_description_fallback(self):
        tom = TheoryOfMindModule()
        # No model for "alice" -> falls back to raw feature values
        features = {"color": "red", "shape": "round"}
        result = tom.adjust_description(features, [], "alice")
        assert set(result) == {"red", "round"}

    def test_choose_perspective_marker(self):
        tom = TheoryOfMindModule()
        # Low experience -> no marker
        assert tom.choose_perspective_marker(0.9, language_experience=5) is None
        # High experience, high certainty -> "know"
        marker = tom.choose_perspective_marker(0.9, language_experience=100)
        assert marker == "know"
        # High experience, medium certainty -> "think"
        marker = tom.choose_perspective_marker(0.6, language_experience=100)
        assert marker == "think"
        # High experience, low certainty -> "believe"
        marker = tom.choose_perspective_marker(0.3, language_experience=100)
        assert marker == "believe"


# ── causal ──────────────────────────────────────────────────────────


class TestCausalRule:
    def test_expression_high_confidence(self):
        rule = CausalRule(cause="fire", effect="heat", confidence=0.9,
                          evidence_count=8, total_observations=9)
        assert "because" in rule.expression()

    def test_expression_medium_confidence(self):
        rule = CausalRule(cause="cloud", effect="rain", confidence=0.65,
                          evidence_count=5, total_observations=8)
        assert "then" in rule.expression()

    def test_expression_low_confidence(self):
        rule = CausalRule(cause="wind", effect="rain", confidence=0.3,
                          evidence_count=1, total_observations=5)
        assert "if" in rule.expression()


class TestCausalReasoningModule:
    def test_observe_builds_rules(self):
        mod = CausalReasoningModule()
        mod.observe("fire", "heat")
        mod.observe("fire", "heat")
        mod.observe("fire", "heat")

        conf = mod.get_confidence("fire", "heat")
        assert conf > 0.5

        rule = mod.rules[("fire", "heat")]
        assert rule.evidence_count == 3
        assert rule.total_observations == 3

    def test_non_occurrence_lowers_confidence(self):
        mod = CausalReasoningModule()
        for _ in range(3):
            mod.observe("fire", "heat")
        mod.observe_non_occurrence("fire", "heat")
        conf_after = mod.get_confidence("fire", "heat")
        # 3 success, 1 failure: (1+3)/(1+3 + 1+1) = 4/6 ~ 0.667
        assert abs(conf_after - 4.0 / 6.0) < 1e-6

    def test_express_causal_and_get_confident(self):
        mod = CausalReasoningModule()
        for _ in range(10):
            mod.observe("rain", "wet")
        expr = mod.express_causal("rain", "wet")
        assert expr is not None
        assert "because" in expr

        confident = mod.get_confident_rules(threshold=0.6)
        assert any(r.cause == "rain" and r.effect == "wet" for r in confident)

        # Unknown pair
        assert mod.express_causal("snow", "hot") is None


# ── counterfactual ──────────────────────────────────────────────────


class TestCounterfactualWorld:
    def test_execute_deterministic(self):
        world = CounterfactualWorld()
        world.add_action_effect("jump", "land", 1.0)
        effect, causal = world.execute("jump")
        assert effect == "land"
        assert causal is True

    def test_execute_unknown_action(self):
        world = CounterfactualWorld()
        effect, causal = world.execute("fly")
        assert effect == "nothing"
        assert causal is False


class TestCounterfactualModule:
    def test_reason_returns_alternative(self):
        world = CounterfactualWorld()
        world.add_action_effect("jump", "land", 1.0)
        world.add_action_effect("sit", "rest", 1.0)

        mod = CounterfactualModule()
        result = mod.reason("jump", "land", world)
        # "sit" -> "rest" != "land" and != "nothing"
        assert result == "rest"

    def test_choose_marker_same_effect(self):
        mod = CounterfactualModule()
        marker = mod.choose_marker("same", "same")
        assert marker == "then"

    def test_choose_marker_different_effect(self):
        mod = CounterfactualModule()
        marker = mod.choose_marker("land", "rest")
        assert marker in {"if", "would", "instead"}

    def test_update_marker_success(self):
        mod = CounterfactualModule()
        mod.update_marker_success("if", True)
        mod.update_marker_success("if", False)
        stats = mod.marker_stats["if"]
        assert stats["frequency"] == 2
        assert stats["successes"] == 1
        assert abs(stats["success_rate"] - 0.5) < 1e-6


# ── tool_use ────────────────────────────────────────────────────────


class TestToolUseModule:
    def test_select_tool_matches_affordance(self):
        mod = ToolUseModule()
        tools = [
            Tool("stick", {"color": "brown"}, {"reach"}, {"reach_object": 0.9}),
            Tool("rock", {"color": "gray"}, {"hit"}, {"break_wall": 0.8}),
        ]
        goal = Goal("reach_object", "reach")
        idx = mod.select_tool(tools, goal)
        assert idx == 0  # stick matches "reach"

    def test_select_tool_no_match(self):
        mod = ToolUseModule()
        tools = [
            Tool("rock", {"color": "gray"}, {"hit"}, {}),
        ]
        goal = Goal("reach_object", "reach")
        idx = mod.select_tool(tools, goal)
        assert idx is None  # no tool has "reach" affordance

    def test_describe_functional(self):
        mod = ToolUseModule()
        tool = Tool("stick", {"color": "brown", "shape": "long"},
                    {"reach", "push"}, {"reach_object": 0.9})
        goal = Goal("reach_object", "reach")
        symbols = mod.describe_functional(tool, goal)
        # "reach" should come first (matches goal affordance)
        assert symbols[0] == "reach"
        assert "push" in symbols
        assert "brown" in symbols


# ── questioning ─────────────────────────────────────────────────────


class TestQuestioningModule:
    def test_generate_question_novel_feature(self):
        mod = QuestioningModule(surprise_threshold=0.1)
        prediction = {"color": 1.0}
        actual = {"color": 1.0, "shape": 5.0, "size": 10.0}
        observation = actual.copy()

        result = mod.generate_question(observation, prediction, actual)
        # Should detect novel features -> "what" marker, topic is the
        # feature with the largest numerical difference from prediction
        assert result is not None
        marker, topic = result
        assert marker == "what"
        assert topic in {"shape", "size"}

    def test_generate_question_below_threshold(self):
        mod = QuestioningModule(surprise_threshold=0.9)
        prediction = {"x": 1.0}
        actual = {"x": 1.0}
        result = mod.generate_question({}, prediction, actual)
        assert result is None

    def test_record_answer_and_adapt(self):
        mod = QuestioningModule()
        # Fill QA history to trigger adaptation
        for i in range(25):
            mod.record_answer("why", "topic", f"answer_{i}", helpful=(i % 2 == 0))
        old_threshold = mod.surprise_threshold
        mod.adapt_threshold()
        # After recording answers, threshold should still be in valid range
        assert 0.1 <= mod.surprise_threshold <= 0.9


# ── metaphor ────────────────────────────────────────────────────────


class TestMetaphorTracker:
    def test_record_and_score_below_minimum(self):
        tracker = MetaphorTracker()
        scene = {"features": [{"temperature": "hot"}]}
        tracker.record(["angry"], scene)
        # Fewer than 10 records -> score is 0.0
        score = tracker.get_metaphor_score("temperature_emotion", "hot", "angry")
        assert score == 0.0

    def test_detect_metaphors_after_many_records(self):
        tracker = MetaphorTracker()
        scene = {"features": [{"temperature": "hot"}]}
        # Record 12 times to exceed the minimum threshold of 10
        for _ in range(12):
            tracker.record(["angry"], scene)

        detected = tracker.detect_metaphors()
        # "temperature_emotion" mapping should appear with hot->angry pair
        if "temperature_emotion" in detected:
            assert "hot_angry" in detected["temperature_emotion"]
            assert detected["temperature_emotion"]["hot_angry"] > 0.0

    def test_save_load_roundtrip(self):
        tracker = MetaphorTracker()
        scene = {"features": [{"temperature": "warm"}, {"weight": "heavy"}]}
        for _ in range(5):
            tracker.record(["friendly", "important"], scene)

        state = tracker.save_state()
        tracker2 = MetaphorTracker()
        tracker2.load_state(state)
        assert tracker2._record_count == 5
        assert tracker2._source_usage["temperature"]["warm"] == 5
        assert tracker2._target_usage["emotion"]["friendly"] == 5
