"""
统一因果推理模块测试脚本（完整版）

测试：
1. Beta因果学习
2. DAG因果推理
3. 干预推理（do-calculus）
4. 因果链推理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_beta_learning():
    """测试Beta因果学习"""
    print("=== 测试Beta因果学习 ===")

    from src.reasoning.unified_causal import (
        UnifiedCausalReasoner, get_unified_causal_reasoner
    )

    reasoner = get_unified_causal_reasoner()

    # 学习因果规则
    rules_to_learn = [
        ("加热", "沸腾", 0.9),
        ("运动", "出汗", 0.85),
        ("下雨", "地湿", 0.95),
        ("吃饭", "饱腹", 0.9),
    ]

    for cause, effect, conf in rules_to_learn:
        reasoner.learn_causal(cause, effect, conf)
        print(f"学习: {cause} -> {effect}")

    # 查询规则
    print("\n查询'加热'相关规则:")
    results = reasoner.query_causal("加热", top_k=3)

    for r in results:
        print(f"  {r.cause} -> {r.effect} (置信度: {r.confidence:.2f}, 来源: {r.source.value})")

    assert len(results) > 0, "应该有查询结果"
    print("[OK] Beta因果学习测试通过\n")
    return reasoner


def test_dag_reasoning():
    """测试DAG因果推理"""
    print("=== 测试DAG因果推理 ===")

    from src.reasoning.unified_causal import get_unified_causal_reasoner

    reasoner = get_unified_causal_reasoner()

    # 构建因果图
    causal_rules = [
        ("加热", "温度升高"),
        ("温度升高", "沸腾"),
        ("沸腾", "产生蒸汽"),
        ("下雨", "地面湿润"),
        ("地面湿润", "路滑"),
    ]

    for cause, effect in causal_rules:
        reasoner.learn_causal(cause, effect, 1.0)

    # 测试因果链推理
    print("从'加热'到'产生蒸汽'的因果链:")
    paths = reasoner.get_causal_chain("加热", "产生蒸汽", max_depth=3)

    for i, path in enumerate(paths[:3], 1):
        path_str = " -> ".join(path)
        print(f"  {i}. {path_str}")

    assert len(paths) > 0, "应该有因果链"
    print("[OK] DAG因果推理测试通过\n")
    return reasoner


def test_intervention():
    """测试干预推理（do-calculus）"""
    print("=== 测试干预推理（do-calculus） ===")

    from src.reasoning.unified_causal import get_unified_causal_reasoner

    reasoner = get_unified_causal_reasoner()

    # 构建简单因果图
    reasoner.learn_causal("服药", "康复", 0.8)
    reasoner.learn_causal("休息", "康复", 0.7)

    # 测试干预
    intervention = {
        'action': 'do(服药)=true',
        'values': {'服药': 'true'}
    }

    result = reasoner.reason_intervention(intervention)

    print(f"干预: {intervention['action']}")
    print(f"受影响的变量: {result['affected_variables']}")
    print(f"推理类型: {result['reasoning_type']}")
    print(f"置信度: {result['confidence']}")

    assert 'affected_variables' in result, "应该有影响变量信息"
    print("[OK] 干预推理测试通过\n")


def test_backward_reasoning():
    """测试反向推理"""
    print("=== 测试反向推理 ===")

    from src.reasoning.unified_causal import get_unified_causal_reasoner

    reasoner = get_unified_causal_reasoner()

    # 学习因果规则
    reasoner.learn_causal("病毒", "感冒", 0.9)
    reasoner.learn_causal("受凉", "感冒", 0.7)
    reasoner.learn_causal("免疫力低", "感冒", 0.8)

    # 反向推理：从结果推断原因
    print("从'感冒'推断可能原因:")
    causes = reasoner.infer_backward("感冒", top_k=3)

    for c in causes:
        print(f"  可能原因: {c.cause} (置信度: {c.confidence:.2f})")

    assert len(causes) > 0, "应该有推断原因"
    print("[OK] 反向推理测试通过\n")


def test_unified_integration():
    """测试统一推理集成"""
    print("=== 测试统一推理集成 ===")

    from src.reasoning.unified_causal import get_unified_causal_reasoner

    reasoner = get_unified_causal_reasoner()

    # 构建完整因果知识
    kb_data = [
        ("加热", "温度上升"),
        ("温度上升", "沸腾"),
        ("沸腾", "蒸汽产生"),
        ("运动", "出汗"),
        ("出汗", "体温调节"),
        ("下雨", "地湿"),
    ]

    for cause, effect in kb_data:
        reasoner.learn_causal(cause, effect, 0.9)

    # 综合测试
    print("1. 查询'运动'相关:")
    results = reasoner.query_causal("运动", top_k=2)
    for r in results:
        print(f"   {r.cause} -> {r.effect}")

    print("\n2. 查询'加热'到'蒸汽'的因果链:")
    paths = reasoner.get_causal_chain("加热", "蒸汽产生")
    print(f"   找到 {len(paths)} 条路径")

    print("\n3. 反向推理'出汗'的原因:")
    causes = reasoner.infer_backward("出汗", top_k=2)
    for c in causes:
        print(f"   {c.cause}")

    # 获取统计
    stats = reasoner.get_stats()
    print(f"\n系统统计:")
    print(f"  总观察次数: {stats['observations']}")
    print(f"  学习规则数: {stats['rules_learned']}")
    print(f"  DAG节点数: {stats['dag_nodes']}")

    print("[OK] 统一推理集成测试通过\n")


if __name__ == '__main__':
    try:
        test_beta_learning()
        test_dag_reasoning()
        test_intervention()
        test_backward_reasoning()
        test_unified_integration()

        print("=" * 50)
        print("[SUCCESS] 统一因果推理模块（完整版）所有测试通过")
        print()
        print("Phase 2完成总结:")
        print("- Beta因果学习: [OK]")
        print("- DAG因果推理: [OK]")
        print("- 干预推理: [OK]")
        print("- 因果链推理: [OK]")
        print("- 反向推理: [OK]")
        print("- 统一推理集成: [OK]")
        print()
        print("下一步：Phase 3 - 集成推理引擎（完整版）")
        print()
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
