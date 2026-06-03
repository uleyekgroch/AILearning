"""
人类学习系统测试

测试从第一性原理出发的人类学习机制
"""

import pytest
import numpy as np


class TestPredictiveLearning:
    """预测学习测试"""

    def test_predict(self):
        """测试预测"""
        # Arrange
        from src.production.domain.human_learning.predictive_system import PredictiveLearningSystem

        system = PredictiveLearningSystem(input_dim=4)

        # Act
        input_data = np.array([1.0, 0.0, 0.0, 0.0])
        prediction = system.predict(input_data)

        # Assert
        assert prediction is not None
        assert len(prediction) == 4

    def test_learn_from_error(self):
        """测试从误差中学习"""
        # Arrange
        from src.production.domain.human_learning.predictive_system import PredictiveLearningSystem

        system = PredictiveLearningSystem(input_dim=4)

        # Act - 多次学习
        for i in range(10):
            input_data = np.array([1.0, 0.0, 0.0, 0.0])
            actual = np.array([0.5, 0.5, 0.0, 0.0])
            result = system.perceive_and_learn(input_data, actual)

        # Assert
        assert result['error'] >= 0
        assert result['confidence'] >= 0

    def test_learning_progress(self):
        """测试学习进度"""
        # Arrange
        from src.production.domain.human_learning.predictive_system import PredictiveLearningSystem

        system = PredictiveLearningSystem(input_dim=4)

        # Act - 多次学习
        for i in range(20):
            input_data = np.array([1.0, 0.0, 0.0, 0.0])
            actual = np.array([0.5, 0.5, 0.0, 0.0])
            system.perceive_and_learn(input_data, actual)

        # Assert
        stats = system.get_stats()
        assert stats['total_learning_cycles'] == 20


class TestMemoryConsolidation:
    """记忆巩固测试"""

    def test_hippocampal_encoding(self):
        """测试海马体编码"""
        # Arrange
        from src.production.domain.human_learning.memory_consolidation import MemoryConsolidation

        system = MemoryConsolidation(hippocampal_capacity=5)

        # Act
        memory = system.learn("水在100度沸腾", importance=0.9)

        # Assert
        assert memory is not None
        assert memory.content == "水在100度沸腾"
        assert memory.strength == 1.0

    def test_sleep_consolidation(self):
        """测试睡眠巩固"""
        # Arrange
        from src.production.domain.human_learning.memory_consolidation import MemoryConsolidation

        system = MemoryConsolidation(hippocampal_capacity=5)

        # 学习一些内容
        system.learn("水在100度沸腾", importance=0.9)
        system.learn("冰在0度融化", importance=0.8)
        system.learn("不重要的信息", importance=0.1)

        # Act
        result = system.sleep()

        # Assert
        assert result.memories_consolidated > 0

    def test_recall(self):
        """测试回忆"""
        # Arrange
        from src.production.domain.human_learning.memory_consolidation import MemoryConsolidation

        system = MemoryConsolidation(hippocampal_capacity=5)

        # 学习一些内容
        system.learn("水在100度沸腾", importance=0.9)
        system.learn("冰在0度融化", importance=0.8)

        # Act
        memories = system.recall()

        # Assert
        assert len(memories) > 0


class TestBiologicalLearning:
    """生物学习测试"""

    def test_stdp_learning(self):
        """测试STDP学习"""
        # Arrange
        from src.production.domain.human_learning.biological_learning import BiologicalLearning

        system = BiologicalLearning()
        system.stdp.add_synapse("neuron_1", "neuron_2", initial_weight=0.5)

        # Act - 突触前先激活（因果关系）
        dw = system.learn_stdp("neuron_1", "neuron_2", pre_spike_time=0.0, post_spike_time=5.0)

        # Assert
        assert dw > 0  # 应该增强

    def test_hebbian_learning(self):
        """测试Hebbian学习"""
        # Arrange
        from src.production.domain.human_learning.biological_learning import BiologicalLearning

        system = BiologicalLearning()

        # Act
        changes = system.learn_hebbian(["neuron_1", "neuron_2", "neuron_3"])

        # Assert
        assert len(changes) > 0

    def test_predictive_coding(self):
        """测试预测编码"""
        # Arrange
        from src.production.domain.human_learning.biological_learning import BiologicalLearning

        system = BiologicalLearning(input_dim=4)

        # Act
        input_data = np.array([1.0, 0.0, 0.0, 0.0])
        actual = np.array([0.5, 0.5, 0.0, 0.0])
        error = system.learn_predictive(input_data, actual)

        # Assert
        assert error >= 0


class TestConceptFormation:
    """概念形成测试"""

    def test_form_concept(self):
        """测试概念形成"""
        # Arrange
        from src.production.domain.human_learning.concept_formation import ConceptFormation

        system = ConceptFormation(feature_dim=8)

        # Act
        concept = system.form_concept(
            name="狗",
            examples=["小狗1", "小狗2", "小狗3"]
        )

        # Assert
        assert concept is not None
        assert concept.name == "狗"
        assert len(concept.features) > 0

    def test_concept_hierarchy(self):
        """测试概念层次"""
        # Arrange
        from src.production.domain.human_learning.concept_formation import ConceptFormation

        system = ConceptFormation(feature_dim=8)

        # 创建父概念
        animal = system.form_concept(name="动物", examples=["动物1", "动物2"])

        # 创建子概念
        dog = system.form_concept(
            name="狗",
            examples=["小狗1", "小狗2"],
            parent_concepts={animal.concept_id}
        )

        # Act
        children = system.get_children(animal.concept_id)
        parents = system.get_parents(dog.concept_id)

        # Assert
        assert len(children) == 1
        assert children[0].name == "狗"
        assert len(parents) == 1
        assert parents[0].name == "动物"

    def test_analogy(self):
        """测试类比"""
        # Arrange
        from src.production.domain.human_learning.concept_formation import ConceptFormation

        system = ConceptFormation(feature_dim=8)

        # 创建两个相似概念
        concept1 = system.form_concept(name="太阳", examples=["太阳1", "太阳2"])
        concept2 = system.form_concept(name="火", examples=["火1", "火2"])

        # Act
        analogy = system.find_analogy(concept1.concept_id, concept2.concept_id, threshold=0.0)

        # Assert
        assert analogy is not None
        assert analogy.similarity >= 0
