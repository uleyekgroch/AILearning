"""
真正的学习系统测试

测试真正的学习能力：
1. 理解能力
2. 推理能力
3. 创造能力
4. 改进能力
"""

import pytest


class TestUnderstanding:
    """理解能力测试"""

    def test_learn_concept(self):
        """测试学习概念"""
        # Arrange
        from src.production.domain.true_learning.understanding import UnderstandingEngine

        engine = UnderstandingEngine()

        # Act
        semantics = engine.learn_concept(
            concept="数学",
            definition="研究数量、结构、变化的学科",
            properties={"type": "学科"},
            relations={"包含": ["算术", "代数", "几何"]}
        )

        # Assert
        assert semantics.concept == "数学"
        assert len(semantics.vector) > 0
        assert "包含" in semantics.relations

    def test_understand_text(self):
        """测试理解文本"""
        # Arrange
        from src.production.domain.true_learning.understanding import UnderstandingEngine

        engine = UnderstandingEngine()
        engine.learn_concept("数学", "研究数量的学科")
        engine.learn_concept("物理", "研究物质的学科")

        # Act
        result = engine.understand("数学和物理都是科学")

        # Assert
        assert result.input_text == "数学和物理都是科学"
        assert result.confidence > 0
        assert len(result.reasoning_chain) > 0


class TestReasoning:
    """推理能力测试"""

    def test_deductive_reasoning(self):
        """测试演绎推理"""
        # Arrange
        from src.production.domain.true_learning.reasoning import ReasoningEngine

        engine = ReasoningEngine()
        engine.add_rule("数学是学科", "数学很重要")

        # Act
        result = engine.reason_deductive("数学是学科")

        # Assert
        assert result.conclusion == "数学很重要"
        assert result.confidence > 0

    def test_inductive_reasoning(self):
        """测试归纳推理"""
        # Arrange
        from src.production.domain.true_learning.reasoning import ReasoningEngine

        engine = ReasoningEngine()

        # Act
        result = engine.reason_inductive(["苹果是水果", "香蕉是水果", "橙子是水果"])

        # Assert
        assert "共同特征" in result.conclusion
        assert result.confidence > 0

    def test_causal_reasoning(self):
        """测试因果推理"""
        # Arrange
        from src.production.domain.true_learning.reasoning import ReasoningEngine

        engine = ReasoningEngine()
        engine.add_knowledge("下雨", {"facts": ["下雨会导致地面湿"]}, {"导致": ["地面湿"]})

        # Act
        result = engine.reason_causal("下雨")

        # Assert
        assert "地面湿" in result.conclusion
        assert result.confidence > 0


class TestCreation:
    """创造能力测试"""

    def test_create_expository(self):
        """测试创作说明文"""
        # Arrange
        from src.production.domain.true_learning.creation import CreationEngine, CreationRequest

        engine = CreationEngine()
        engine.learn_knowledge("数学", ["数学研究数量", "数学是科学基础"])

        request = CreationRequest(
            topic="数学",
            style="expository",
            length="short",
            requirements=[],
            context={}
        )

        # Act
        result = engine.create(request)

        # Assert
        assert "数学" in result.content
        assert result.quality_score > 0
        assert len(result.creation_process) > 0

    def test_create_narrative(self):
        """测试创作叙事文"""
        # Arrange
        from src.production.domain.true_learning.creation import CreationEngine, CreationRequest

        engine = CreationEngine()
        engine.learn_knowledge("物理", ["牛顿发现万有引力"])

        request = CreationRequest(
            topic="物理",
            style="narrative",
            length="short",
            requirements=[],
            context={}
        )

        # Act
        result = engine.create(request)

        # Assert
        assert "物理" in result.content
        assert result.style == "narrative"


class TestImprovement:
    """改进能力测试"""

    def test_identify_and_solve_problem(self):
        """测试识别和解决问题"""
        # Arrange
        from src.production.domain.true_learning.improvement import ImprovementEngine

        engine = ImprovementEngine()

        # 识别问题
        problem = engine.identify_problem(
            description="学习效率低",
            severity="medium",
            location="学习系统",
            impact="学习速度"
        )

        # 生成计划
        plan = engine.generate_plan(problem)

        # 实施计划
        result = engine.implement_plan(plan, "学习效率提高")

        # Assert
        assert problem.problem_id is not None
        assert plan.plan_id is not None
        assert result.success is True

    def test_learn_from_experience(self):
        """测试从经验中学习"""
        # Arrange
        from src.production.domain.true_learning.improvement import ImprovementEngine

        engine = ImprovementEngine()

        # 创建经验
        problem = engine.identify_problem("问题1", "low", "loc1", "imp1")
        plan = engine.generate_plan(problem)
        engine.implement_plan(plan, "成功")

        # Act
        lessons = engine.learn_from_experience()

        # Assert
        assert len(lessons) > 0


class TestTrueLearner:
    """真正的学习器测试"""

    def test_learn_topic(self):
        """测试学习主题"""
        # Arrange
        from src.production.domain.true_learning.learner import TrueLearner

        learner = TrueLearner()

        # Act
        result = learner.learn_topic(
            topic="数学",
            facts=["数学研究数量", "数学是科学基础"],
            relations={"包含": ["算术", "代数"]}
        )

        # Assert
        assert result.topic == "数学"
        assert result.overall_score > 0
        assert result.understanding is not None
        assert result.reasoning is not None
        assert result.creation is not None

    def test_create_content(self):
        """测试创造内容"""
        # Arrange
        from src.production.domain.true_learning.learner import TrueLearner

        learner = TrueLearner()

        # 先学习
        learner.learn_topic("物理", ["物理研究物质"])

        # Act
        result = learner.create_content("物理", "expository")

        # Assert
        assert "物理" in result.content
        assert result.quality_score > 0

    def test_improve_system(self):
        """测试改进系统"""
        # Arrange
        from src.production.domain.true_learning.learner import TrueLearner

        learner = TrueLearner()

        # Act
        result = learner.improve_system("学习效率低")

        # Assert
        assert result is not None
        assert result.plan is not None

    def test_reason(self):
        """测试推理"""
        # Arrange
        from src.production.domain.true_learning.learner import TrueLearner

        learner = TrueLearner()

        # Act
        result = learner.reason("数学是学科", "deductive")

        # Assert
        assert result is not None
        assert result.conclusion is not None

    def test_understand(self):
        """测试理解"""
        # Arrange
        from src.production.domain.true_learning.learner import TrueLearner

        learner = TrueLearner()

        # 先学习概念
        learner.understanding_engine.learn_concept("数学", "研究数量的学科")

        # Act
        result = learner.understand("数学很重要")

        # Assert
        assert result is not None
        assert result.input_text == "数学很重要"
