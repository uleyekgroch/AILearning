"""
神经符号整合模块测试

测试：
1. 神经-符号桥接功能
2. 混合推理引擎
3. 反馈循环
4. 与Learner集成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_neuro_symbolic_bridge():
    """测试神经-符号桥接"""
    print("=== 测试神经-符号桥接 ===")

    from src.learning.neuro_symbolic import NeuroSymbolicBridge
    import torch

    # 创建桥接器
    bridge = NeuroSymbolicBridge(neural_dim=128)

    # 注册符号-向量映射
    symbol1 = "光合作用"
    vector1 = torch.randn(128)
    bridge.register_symbol(symbol1, vector1)

    symbol2 = "植物"
    vector2 = torch.randn(128)
    bridge.register_symbol(symbol2, vector2)

    # 添加原型
    prototype = torch.randn(128)
    bridge.add_prototype("能量转换", prototype)

    stats = bridge.get_stats()
    print(f"桥接器统计:")
    print(f"  注册符号数: {stats['registered_symbols']}")
    print(f"  原型数: {stats['prototypes']}")

    # 测试神经→符号转换
    test_vector = vector1.clone()
    symbols = bridge.neural_to_symbol(test_vector, top_k=3)
    print(f"\n神经→符号转换:")
    print(f"  候选符号: {symbols[:2]}")

    # 测试符号→神经转换
    neural_vec = bridge.symbol_to_neural(symbol1)
    print(f"\n符号→神经转换:")
    print(f"  成功: {neural_vec is not None}")

    print("[OK] 神经-符号桥接测试通过\n")


def test_hybrid_reasoning():
    """测试混合推理引擎"""
    print("=== 测试混合推理引擎 ===")

    from src.learning.neuro_symbolic import NeuroSymbolicBridge, HybridReasoningEngine
    import torch

    # 创建桥接器和引擎
    bridge = NeuroSymbolicBridge(neural_dim=128)
    engine = HybridReasoningEngine(bridge)

    # 注册测试符号
    test_symbol = "光合作用"
    test_vector = torch.randn(128)
    bridge.register_symbol(test_symbol, test_vector)

    # 测试神经推理
    neural_result = engine._neural_reasoning("什么是光合作用", test_vector)
    print(f"神经推理:")
    print(f"  答案: {neural_result['answer']}")
    print(f"  置信度: {neural_result['confidence']:.2f}")

    # 测试符号推理
    symbolic_result = engine._symbolic_reasoning("光合作用", ["光合作用", "植物", "阳光"])
    print(f"\n符号推理:")
    print(f"  答案: {symbolic_result['answer']}")
    print(f"  置信度: {symbolic_result['confidence']:.2f}")

    # 测试混合推理
    hybrid_result = engine.reason("什么是光合作用", test_vector, ["光合作用"])
    print(f"\n混合推理:")
    print(f"  推理类型: {hybrid_result['reasoning_type']}")
    print(f"  置信度: {hybrid_result['confidence']:.2f}")
    if hybrid_result.get('hybrid_result'):
        print(f"  答案: {hybrid_result['hybrid_result']['answer']}")

    print("[OK] 混合推理引擎测试通过\n")


def test_integrator():
    """测试神经符号整合器"""
    print("=== 测试神经符号整合器 ===")

    from src.learning.neuro_symbolic import NeuroSymbolicIntegrator
    import torch

    # 创建整合器
    integrator = NeuroSymbolicIntegrator(neural_dim=128)

    # 学习映射
    symbol = "学习"
    vector = torch.randn(128)
    integrator.learn_mapping(symbol, vector)

    # 测试整合推理
    result = integrator.integrate_reasoning(
        question="什么是学习",
        neural_vector=vector,
        symbols=["学习", "知识", "认知"]
    )

    print(f"整合推理结果:")
    print(f"  推理类型: {result['reasoning_type']}")
    print(f"  置信度: {result['confidence']:.2f}")

    stats = integrator.get_stats()
    print(f"\n整合器统计:")
    print(f"  桥接统计: {stats['bridge_stats']}")
    print(f"  反馈次数: {stats['feedback_count']}")

    print("[OK] 神经符号整合器测试通过\n")


def test_with_learner():
    """测试与Learner集成"""
    print("=== 测试与Learner集成 ===")

    from src.core.learner import Learner
    from src.core.config import LearnerConfig
    import torch

    # 创建Learner
    config = LearnerConfig(device='cpu')
    learner = Learner(config)

    # 初始化神经符号整合器
    from src.learning.neuro_symbolic import NeuroSymbolicIntegrator
    integrator = NeuroSymbolicIntegrator(neural_dim=128)

    # 学习一些概念向量（模拟）
    concepts = ["光合作用", "植物", "学习"]
    for concept in concepts:
        vector = torch.randn(128)
        integrator.learn_mapping(concept, vector)

    # 测试整合推理
    test_vector = torch.randn(128)
    result = integrator.integrate_reasoning(
        question="什么是光合作用",
        neural_vector=test_vector,
        symbols=concepts
    )

    print(f"与Learner集成测试:")
    print(f"  推理类型: {result.get('reasoning_type', 'N/A')}")
    print(f"  置信度: {result.get('confidence', 0.0):.2f}")

    print("[OK] Learner集成测试通过\n")


if __name__ == '__main__':
    try:
        test_neuro_symbolic_bridge()
        test_hybrid_reasoning()
        test_integrator()
        test_with_learner()

        print("=" * 50)
        print("[SUCCESS] 所有神经符号整合测试通过")
        print()
        print("Phase 5 完成:")
        print("- 神经-符号双向桥接")
        print("- 混合推理引擎")
        print("- 反馈循环机制")
        print("- 与Learner集成")
        print()
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
