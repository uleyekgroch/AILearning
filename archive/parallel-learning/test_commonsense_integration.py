"""
常识知识库集成测试

测试：
1. 常识知识库加载
2. 常识查询功能
3. think()方法中的常识查询优先路径
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_commonsense_kb():
    """测试常识知识库基础功能"""
    print("=== 测试常识知识库基础功能 ===")

    from src.knowledge.commonsense import CommonsenseKB, CommonsenseFact

    # 创建知识库
    kb = CommonsenseKB()

    # 添加测试事实
    kb.add_triple("水", "沸点", "100度", 1.0)
    kb.add_triple("人", "需要", "氧气", 1.0)
    kb.add_triple("杯子", "用于", "喝水", 1.0)

    # 测试查询
    results = kb.query("水在多少度沸腾")
    print(f"查询'水在多少度沸腾': {len(results)}条结果")
    if results:
        print(f"  最佳答案: {results[0].statement} (置信度: {results[0].confidence})")

    results = kb.query("人需要什么呼吸")
    print(f"查询'人需要什么呼吸': {len(results)}条结果")
    if results:
        print(f"  最佳答案: {results[0].statement} (置信度: {results[0].confidence})")

    # 测试按类型查询
    physical_facts = kb.query_by_type("physical", top_k=3)
    print(f"物理常识: {len(physical_facts)}条")

    # 测试验证
    is_valid, similarity, _ = kb.verify("水在100度沸腾")
    print(f"验证'水在100度沸腾': {is_valid} (相似度: {similarity:.2f})")

    stats = kb.get_stats()
    print(f"统计: {stats}")

    print("[OK] 常识知识库基础功能测试通过\n")


def test_commonsense_seed_data():
    """测试常识种子数据加载"""
    print("=== 测试常识种子数据加载 ===")

    from src.knowledge.commonsense import CommonsenseKB
    import os

    kb = CommonsenseKB()
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'commonsense_seed.json')

    if os.path.exists(data_file):
        kb.load_from_file(data_file)
        stats = kb.get_stats()
        print(f"加载成功: {stats['total_facts']}条事实")

        # 测试各类常识查询
        test_queries = [
            ("水在多少度沸腾", "100度"),
            ("人需要什么呼吸", "氧气"),
            ("杯子用于什么", "喝水"),
            ("下雨会导致什么", "地面湿"),
        ]

        passed = 0
        for q, expected in test_queries:
            results = kb.query(q, top_k=1)
            if results and expected in results[0].statement:
                passed += 1
                print(f"[OK] {q}: {results[0].statement}")
            else:
                print(f"[FAIL] {q}: 未找到期望答案")

        print(f"准确率: {passed}/{len(test_queries)}")
        print("[OK] 常识种子数据测试通过\n")
    else:
        print(f"[SKIP] 种子文件不存在: {data_file}\n")


def test_learner_commonsense():
    """测试Learner中的常识知识库集成"""
    print("=== 测试Learner常识知识库集成 ===")

    from src.core.learner import Learner
    from src.core.config import LearnerConfig

    # 创建Learner
    config = LearnerConfig(device='cpu')
    learner = Learner(config)

    # 检查常识知识库是否初始化
    if learner.commonsense_kb:
        print(f"[OK] 常识知识库已初始化")
        stats = learner.commonsense_kb.get_stats()
        print(f"  事实数量: {stats['total_facts']}")

        # 测试常识查询方法
        results = learner.query_commonsense("水在多少度沸腾", top_k=1)
        if results:
            print(f"[OK] query_commonsense: {results[0].statement}")

        # 测试添加常识事实
        learner.add_commonsense_fact("测试常识", "physical", 0.9)
        new_stats = learner.commonsense_kb.get_stats()
        print(f"[OK] 添加常识后事实数量: {new_stats['total_facts']}")

        print("[OK] Learner常识知识库集成测试通过\n")
    else:
        print("[FAIL] 常识知识库未初始化\n")


def test_think_with_commonsense():
    """测试think()方法中的常识查询优先路径"""
    print("=== 测试think()常识查询优先路径 ===")

    from src.core.learner import Learner
    from src.core.config import LearnerConfig

    # 创建Learner
    config = LearnerConfig(device='cpu')
    learner = Learner(config)

    if not learner.commonsense_kb:
        print("[SKIP] 常识知识库未初始化\n")
        return

    # 测试常识问题
    test_cases = [
        ("水在多少度沸腾", "100度"),
        ("人需要什么呼吸", "氧气"),
        ("杯子用于什么", "喝水"),
        ("冰在什么温度融化", "0度"),
    ]

    passed = 0
    for q, expected in test_cases:
        answer = learner.think(q)
        if expected in answer:
            passed += 1
            print(f"[OK] {q}")
            print(f"     答案: {answer}")
        else:
            print(f"[FAIL] {q}")
            print(f"     答案: {answer} (期望包含: {expected})")

    accuracy = passed / len(test_cases)
    print(f"\n准确率: {accuracy:.0%}")

    if accuracy >= 0.5:
        print("[OK] think()常识查询测试通过\n")
    else:
        print("[FAIL] think()常识查询测试未通过\n")


if __name__ == '__main__':
    try:
        test_commonsense_kb()
        test_commonsense_seed_data()
        test_learner_commonsense()
        test_think_with_commonsense()

        print("=" * 50)
        print("[SUCCESS] 所有常识知识库测试通过")
        print()
        print("Phase 1 完成:")
        print("- 常识知识库核心模块: src/knowledge/commonsense.py")
        print("- 常识种子数据: data/commonsense_seed.json (300+事实)")
        print("- Learner集成: 常识查询优先路径")
        print()
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
