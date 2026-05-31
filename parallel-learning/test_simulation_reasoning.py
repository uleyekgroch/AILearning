"""Phase 4 验证：模拟推理系统

验证目标：
1. 场景构建（从概念到虚拟场景）
2. 因果链追踪
3. 反事实模拟
4. 类比发现
5. 结果表达
6. 与Learner完整集成
"""

import sys
import os
import io
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.learning.concept_space import ConceptSpace
from src.learning.core_knowledge import CoreKnowledgeSystem
from src.learning.functional_concept import FunctionalConceptSystem
from src.learning.simulation_reasoning import SimulationReasoning


def test_scene_building():
    """测试1: 场景构建"""
    print("\n━━━ 测试1: 场景构建 ━━━")

    cs = ConceptSpace(dim=128)
    ck = CoreKnowledgeSystem()
    fcs = FunctionalConceptSystem(cs, ck)
    sr = SimulationReasoning(concept_space=cs)

    # 注册几个概念并建立关系
    fcs.form_concept("水", perceptual_features={'type': 'substance', 'state': 'liquid'})
    fcs.form_concept("加热", perceptual_features={'type': 'action'})
    fcs.form_concept("温度", perceptual_features={'type': 'property'})
    cs.learn_relation("水", "加热", strength=0.5)
    cs.learn_relation("加热", "温度", strength=0.4)

    # 构建场景
    result = sr.reason("把水加热会怎样", ["水", "加热", "温度"])

    assert result.scene is not None
    assert len(result.scene.concepts) == 3
    print(f"  场景概念: {result.scene.concepts}")
    print(f"  场景关系: {result.scene.relations}")
    print(f"  场景置信度: {result.scene.confidence:.2f}")
    assert result.scene.confidence > 0, "应有非零置信度"

    print("  场景构建测试通过 ✓")


def test_causal_tracing():
    """测试2: 因果链追踪"""
    print("\n━━━ 测试2: 因果链追踪 ━━━")

    cs = ConceptSpace(dim=128)
    sr = SimulationReasoning(concept_space=cs)

    # 注册概念
    for label in ["冰块", "加热", "融化", "水"]:
        cs.register(label, vector=torch.randn(128))

    # 建立因果链
    cs.learn_relation("冰块", "加热", strength=0.6)
    cs.learn_relation("加热", "融化", strength=0.8)
    cs.learn_relation("融化", "水", strength=0.7)

    # 推理
    result = sr.reason("加热冰块会发生什么", ["冰块", "加热", "融化", "水"])

    print(f"  因果链数量: {len(result.causal_chains)}")
    for i, chain in enumerate(result.causal_chains):
        print(f"    链{i+1}: {' → '.join(chain.steps)} (置信度={chain.confidence:.2f})")
        print(f"    证据: {chain.evidence}")

    assert len(result.causal_chains) > 0, "应找到因果链"
    print("  因果链追踪测试通过 ✓")


def test_counterfactual():
    """测试3: 反事实模拟"""
    print("\n━━━ 测试3: 反事实模拟 ━━━")

    cs = ConceptSpace(dim=128)
    sr = SimulationReasoning(concept_space=cs)

    # 注册概念
    for label in ["雨", "地面", "湿"]:
        cs.register(label, vector=torch.randn(128))

    # 反事实问题
    result = sr.reason("如果不下雨地面会怎样", ["雨", "地面", "湿"])

    print(f"  问题类型: {result.reasoning_type}")
    assert result.reasoning_type == 'counterfactual', "应识别为反事实问题"

    print(f"  反事实模拟数量: {len(result.counterfactuals)}")
    for cf in result.counterfactuals:
        print(f"    移除 '{cf['removed']}': {cf['description']}")

    assert len(result.counterfactuals) > 0, "应有反事实模拟"
    print("  反事实模拟测试通过 ✓")


def test_analogy():
    """测试4: 类比发现"""
    print("\n━━━ 测试4: 类比发现 ━━━")

    cs = ConceptSpace(dim=128)
    sr = SimulationReasoning(concept_space=cs)

    # 注册两对有类似关系的概念
    cs.register("数学", vector=torch.randn(128))
    cs.register("学科", vector=torch.randn(128))
    cs.register("物理", vector=torch.randn(128))
    cs.register("科学", vector=torch.randn(128))

    cs.learn_relation("数学", "学科", strength=0.6)
    cs.learn_relation("物理", "科学", strength=0.5)

    # 推理（寻找类比）
    result = sr.reason("数学和物理有什么关系", ["数学", "学科", "物理", "科学"])

    print(f"  类比数量: {len(result.analogies)}")
    for analogy in result.analogies:
        print(f"    {analogy['source']} ~ {analogy['target']} "
              f"(相似度={analogy['similarity']:.2f})")

    # 即使没找到类比也不应报错
    print("  类比发现测试通过 ✓")


def test_expression():
    """测试5: 结果表达"""
    print("\n━━━ 测试5: 结果表达 ━━━")

    cs = ConceptSpace(dim=128)
    sr = SimulationReasoning(concept_space=cs)

    cs.register("水", vector=torch.randn(128))
    cs.register("沸腾", vector=torch.randn(128))
    cs.learn_relation("水", "沸腾", strength=0.7)

    result = sr.reason("加热水会怎样", ["水", "沸腾"])
    answer = sr.express(result, "加热水会怎样")

    print(f"  推理结果: {answer}")
    assert len(answer) > 0, "应生成非空回答"
    print("  结果表达测试通过 ✓")


def test_learner_integration():
    """测试6: 与Learner完整集成"""
    print("\n━━━ 测试6: 与Learner完整集成 ━━━")

    from src.core.config import LearnerConfig
    from src.core.learner import Learner

    config = LearnerConfig(
        encoder_n_layers=1,
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
    )
    learner = Learner(config)
    learner._skip_ttt = True

    # 学习语料
    texts = [
        "数学是研究数量的学科",
        "物理学是研究物质的学科",
        "化学是研究物质变化的自然科学",
    ]
    for text in texts:
        learner.learn_from_text(text)

    # 获取组件
    cs = learner._registry.get('concept_space') if learner._registry.has('concept_space') else None
    kg = learner._registry.get('knowledge') if learner._registry.has('knowledge') else None

    if cs:
        sr = SimulationReasoning(concept_space=cs, knowledge_graph=kg)

        # 推理
        activated = cs.activate("数学研究什么", top_k=5)
        concepts = [ac.concept_id for ac in activated[:5]]
        print(f"  激活概念: {concepts}")

        result = sr.reason("数学研究什么", concepts)
        print(f"  推理类型: {result.reasoning_type}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  因果链: {len(result.causal_chains)}")
        for chain in result.causal_chains[:3]:
            print(f"    {' → '.join(chain.steps)}")

        answer = sr.express(result, "数学研究什么")
        print(f"  模拟推理回答: {answer}")

        # 对比 think() 的回答
        think_answer = learner.think("数学研究什么")
        print(f"  原始think()回答: {think_answer[:100]}...")

        print("  Learner集成测试通过 ✓")
    else:
        print("  ⚠ 概念空间未初始化，跳过")

    stats = sr.get_stats() if cs else {}
    print(f"  SR统计: {stats}")


# ===== 主函数 =====
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 4 验证：模拟推理系统")
    print("=" * 70)

    test_scene_building()
    test_causal_tracing()
    test_counterfactual()
    test_analogy()
    test_expression()
    test_learner_integration()

    print("\n" + "=" * 70)
    print("Phase 4 全部测试通过 ✓")
    print("=" * 70)
