"""
真正的AI系统测试

测试所有核心模块
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPerception:
    """感知模块测试"""

    def test_perceive_text(self):
        """测试感知文本"""
        from src.ai.core.perception import PerceptionModule

        module = PerceptionModule()
        result = module.perceive("这是一个测试")

        assert result.raw_input == "这是一个测试"
        assert result.confidence > 0
        assert len(result.features) > 0

    def test_perceive_question(self):
        """测试感知问题"""
        from src.ai.core.perception import PerceptionModule

        module = PerceptionModule()
        result = module.perceive("这是什么？")

        assert 'question' in result.patterns


class TestUnderstanding:
    """理解模块测试"""

    def test_learn_concept(self):
        """测试学习概念"""
        from src.ai.core.understanding import UnderstandingModule

        module = UnderstandingModule()
        module.learn_concept("数学", "研究数量的学科")

        assert "数学" in module.concept_kb

    def test_understand_text(self):
        """测试理解文本"""
        from src.ai.core.understanding import UnderstandingModule

        module = UnderstandingModule()
        module.learn_concept("数学", "研究数量的学科")

        result = module.understand("数学很重要")

        assert result.text == "数学很重要"
        assert result.confidence > 0


class TestReasoning:
    """推理模块测试"""

    def test_deductive_reasoning(self):
        """测试演绎推理"""
        from src.ai.core.reasoning import ReasoningModule

        module = ReasoningModule()
        module.add_rule("数学是学科", "数学很重要")

        result = module.reason_deductive("数学是学科")

        assert result.conclusion == "数学很重要"
        assert result.confidence > 0

    def test_inductive_reasoning(self):
        """测试归纳推理"""
        from src.ai.core.reasoning import ReasoningModule

        module = ReasoningModule()

        result = module.reason_inductive(["苹果是水果", "香蕉是水果"])

        assert result.confidence > 0


class TestCreation:
    """创造模块测试"""

    def test_create_expository(self):
        """测试创作说明文"""
        from src.ai.core.creation import CreationModule, CreationRequest

        module = CreationModule()
        module.learn_knowledge("数学", ["数学研究数量", "数学是科学基础"])

        request = CreationRequest(topic="数学", style="expository")
        result = module.create(request)

        assert "数学" in result.content
        assert result.quality_score > 0


class TestImprovement:
    """改进模块测试"""

    def test_improve_system(self):
        """测试改进系统"""
        from src.ai.core.improvement import ImprovementModule

        module = ImprovementModule()

        problem = module.identify_problem("学习效率低", "medium", "系统")
        plan = module.generate_plan(problem)
        result = module.implement_plan(plan, "学习效率提高")

        assert result.success is True


class TestTrueAISystem:
    """真正的AI系统测试"""

    def test_learn_and_think(self):
        """测试学习和思考"""
        from src.ai.system import TrueAISystem

        ai = TrueAISystem()

        # 学习
        ai.learn("数学", ["数学研究数量", "数学是科学基础"])

        # 思考
        response = ai.think("数学是什么")

        assert response.input_text == "数学是什么"
        assert response.confidence > 0

    def test_improve(self):
        """测试改进"""
        from src.ai.system import TrueAISystem

        ai = TrueAISystem()

        result = ai.improve("学习效率低")

        assert result.success is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
