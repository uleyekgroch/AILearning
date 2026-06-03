"""
测试人类式学习系统
验证：无训练组件，纯生物机制
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_learning():
    """测试基本学习功能"""
    from src.core.learner import Learner
    from src.core.config import LearnerConfig

    print("=== 测试人类式学习系统 ===\n")

    # 创建学习体
    config = LearnerConfig(device='cpu')
    learner = Learner(config)

    # 检查初始化的生物机制
    assert hasattr(learner, '_stdp_system'), "缺少STDP系统"
    assert hasattr(learner, '_hippocampal_memory'), "缺少海马记忆"
    assert hasattr(learner, '_sleep_consolidation'), "缺少睡眠巩固"
    assert hasattr(learner, '_activation_cache'), "缺少激活缓存"
    print("[OK] 生物机制初始化成功")

    # 检查STDP系统
    assert 'connections' in learner._stdp_system, "STDP缺少连接"
    assert 'lr' in learner._stdp_system, "STDP缺少学习率"
    print("[OK] STDP系统正常")

    # 检查海马记忆
    assert 'episodes' in learner._hippocampal_memory, "海马缺少episodes"
    assert 'index' in learner._hippocampal_memory, "海马缺少index"
    print("[OK] 海马记忆正常")

    # 检查睡眠巩固
    assert 'consolidation_interval' in learner._sleep_consolidation, "睡眠系统缺少间隔"
    print("[OK] 睡眠巩固正常")

    # 测试学习
    print("\n--- 测试学习 ---")
    result = learner.learn_from_text('光合作用是植物利用阳光将二氧化碳转化为葡萄糖的过程')
    print(f"学习结果: {result.get('entities', [])}")

    # 验证STDP连接已创建
    assert len(learner._stdp_system['connections']) > 0, "STDP连接未创建"
    print(f"[OK] STDP连接数: {len(learner._stdp_system['connections'])}")

    # 验证海马记忆已存储
    assert len(learner._hippocampal_memory['episodes']) > 0, "海马记忆未存储"
    print(f"[OK] 海马记忆数: {len(learner._hippocampal_memory['episodes'])}")

    # 测试推理
    print("\n--- 测试推理 ---")
    answer = learner.think('什么是光合作用')
    print(f"推理答案: {answer[:100] if answer else '(无答案)'}")

    # 多次学习触发睡眠巩固
    print("\n--- 测试睡眠巩固 ---")
    for i in range(50):
        learner.learn_from_text(f'测试文本{i}: 植物进行光合作用需要阳光')

    print(f"[OK] 睡眠巩固次数: {learner._sleep_consolidation['consolidation_count']}")

    print("\n=== 所有测试通过 ===")
    return True

def test_encode_text():
    """测试直接感知编码（无Transformer）"""
    from src.core.learner import Learner
    from src.core.config import LearnerConfig

    print("\n=== 测试直接感知编码 ===\n")

    config = LearnerConfig(device='cpu')
    learner = Learner(config)

    # 测试编码
    text = "光合作用是植物的重要过程"
    vec = learner._encode_text(text)

    print(f"编码向量形状: {vec.shape}")
    print(f"编码向量范数: {vec.norm().item():.4f}")

    # 验证词向量已创建
    assert hasattr(learner, '_perception_vectors'), "缺少感知向量"
    print(f"[OK] 词向量数: {len(learner._perception_vectors)}")

    print("\n=== 编码测试通过 ===")
    return True

if __name__ == '__main__':
    try:
        test_basic_learning()
        test_encode_text()
        print("\n[SUCCESS] 所有测试通过！人类式学习系统正常工作。")
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
