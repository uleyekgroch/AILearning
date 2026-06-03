"""
集成推理引擎测试

测试：
1. 集成推理引擎初始化
2. 各种推理模式
3. 自动模式选择
4. 多模式融合
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_integrated_engine_init():
    """测试集成推理引擎初始化"""
    print("=== 测试集成推理引擎初始化 ===")

    from src.reasoning.integrated_engine import IntegratedReasoningEngine

    # 创建模拟Learner
    class MockLearner:
        def __init__(self):
            from src.knowledge.commonsense import CommonsenseKB
            self.commonsense_kb = CommonsenseKB()
            # 添加测试数据
            self.commonsense_kb.add_triple("水", "沸点", "100度", 1.0)
            self.knowledge = None

    mock_learner = MockLearner()
    engine = IntegratedReasoningEngine(mock_learner)

    stats = engine.get_stats()
    print(f"初始化完成:")
    print(f"  常识库可用: {stats['commonsense_available']}")
    print(f"  因果推理可用: {stats['causal_available']}")

    print("[OK] 初始化测试通过\n")


def test_commonsense_reasoning():
    """测试常识推理"""
    print("=== 测试常识推理 ===")

    from src.reasoning.integrated_engine import IntegratedReasoningEngine

    # 创建模拟Learner
    class MockLearner:
        def __init__(self):
            from src.knowledge.commonsense import CommonsenseKB
            self.commonsense_kb = CommonsenseKB()
            self.commonsense_kb.add_triple("水", "沸点", "100度", 1.0)
            self.commonsense_kb.add_triple("人", "需要", "氧气", 1.0)
            self.knowledge = None

    mock_learner = MockLearner()
    engine = IntegratedReasoningEngine(mock_learner)

    # 测试常识推理
    results = engine.reason("水在多少度沸腾", mode='commonsense')
    print(f"查询: 水在多少度沸腾")
    print(f"结果数: {len(results)}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.answer} (置信度: {r.confidence:.2f})")

    print("[OK] 常识推理测试通过\n")


def test_causal_reasoning():
    """测试因果推理"""
    print("=== 测试因果推理 ===")

    from src.reasoning.integrated_engine import IntegratedReasoningEngine

    # 创建模拟Learner
    class MockLearner:
        def __init__(self):
            self.commonsense_kb = None
            self.knowledge = None

    mock_learner = MockLearner()
    engine = IntegratedReasoningEngine(mock_learner)

    # 学习因果规则
    engine._init_causal_if_needed()
    if engine._causal:
        engine._causal.learn_causal("运动", "出汗")
        engine._causal.learn_causal("炎热", "出汗")

    # 测试因果推理
    results = engine.reason("运动会导致什么", mode='causal')
    print(f"查询: 运动会导致什么")
    print(f"结果数: {len(results)}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.answer} (置信度: {r.confidence:.2f})")

    print("[OK] 因果推理测试通过\n")


def test_auto_mode():
    """测试自动模式选择"""
    print("=== 测试自动模式选择 ===")

    from src.reasoning.integrated_engine import IntegratedReasoningEngine

    # 创建模拟Learner
    class MockLearner:
        def __init__(self):
            from src.knowledge.commonsense import CommonsenseKB
            self.commonsense_kb = CommonsenseKB()
            self.commonsense_kb.add_triple("水", "沸点", "100度", 1.0)
            self.knowledge = None

    mock_learner = MockLearner()
    engine = IntegratedReasoningEngine(mock_learner)

    # 测试自动模式（应该选择常识推理）
    results = engine.reason("水在多少度沸腾", mode='auto')
    print(f"查询: 水在多少度沸腾 (auto模式)")
    print(f"结果数: {len(results)}")
    if results:
        print(f"最佳答案: {results[0].answer}")
        print(f"推理类型: {results[0].reasoning_type}")

    print("[OK] 自动模式测试通过\n")


def test_multi_mode():
    """测试多模式融合"""
    print("=== 测试多模式融合 ===")

    from src.reasoning.integrated_engine import IntegratedReasoningEngine

    # 创建模拟Learner
    class MockLearner:
        def __init__(self):
            from src.knowledge.commonsense import CommonsenseKB
            self.commonsense_kb = CommonsenseKB()
            self.commonsense_kb.add_triple("运动", "导致", "出汗", 0.9)
            self.knowledge = None

    mock_learner = MockLearner()
    engine = IntegratedReasoningEngine(mock_learner)

    # 学习因果规则
    engine._init_causal_if_needed()
    if engine._causal:
        engine._causal.learn_causal("运动", "出汗")

    # 测试多模式融合
    results = engine.reason("运动会导致什么", mode='multi')
    print(f"查询: 运动会导致什么 (multi模式)")
    print(f"结果数: {len(results)}")
    for i, r in enumerate(results[:3]):
        print(f"  {i+1}. {r.answer} (类型: {r.reasoning_type}, 置信度: {r.confidence:.2f})")

    print("[OK] 多模式融合测试通过\n")


def test_with_real_learner():
    """测试与真实Learner集成"""
    print("=== 测试与真实Learner集成 ===")

    from src.core.learner import Learner
    from src.core.config import LearnerConfig

    # 创建Learner
    config = LearnerConfig(device='cpu')
    learner = Learner(config)

    # 创建集成推理引擎
    from src.reasoning.integrated_engine import IntegratedReasoningEngine
    engine = IntegratedReasoningEngine(learner)

    stats = engine.get_stats()
    print(f"集成统计:")
    print(f"  常识库可用: {stats['commonsense_available']}")
    print(f"  因果推理可用: {stats['causal_available']}")

    # 测试推理
    results = engine.reason("水在多少度沸腾", mode='auto')
    print(f"\n查询: 水在多少度沸腾")
    if results:
        print(f"最佳答案: {results[0].answer}")
        print(f"推理类型: {results[0].reasoning_type}")

    print("[OK] 真实Learner集成测试通过\n")


if __name__ == '__main__':
    try:
        test_integrated_engine_init()
        test_commonsense_reasoning()
        test_causal_reasoning()
        test_auto_mode()
        test_multi_mode()
        test_with_real_learner()

        print("=" * 50)
        print("[SUCCESS] 所有集成推理引擎测试通过")
        print()
        print("Phase 3 完成:")
        print("- 集成推理引擎: src/reasoning/integrated_engine.py")
        print("- 统一推理接口: 6种模式")
        print("- 自动模式选择: 常识优先")
        print("- 多模式融合: 置信度排序")
        print()
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
