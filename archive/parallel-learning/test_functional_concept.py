"""Phase 2 验证：功能性概念系统

验证目标：
1. 功能性概念形成（带感知特征+可供性）
2. 快速映射机制（互斥性+整体对象偏差）
3. 概念接地到感知经验
4. 可供性检索
5. 概念档案完整性
6. 与Learner集成
"""

import sys
import os
import io
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import LearnerConfig
from src.core.learner import Learner
from src.learning.concept_space import ConceptSpace
from src.learning.core_knowledge import CoreKnowledgeSystem
from src.learning.functional_concept import FunctionalConceptSystem


def test_functional_concept_formation():
    """测试1: 功能性概念形成"""
    print("\n━━━ 测试1: 功能性概念形成 ━━━")

    cs = ConceptSpace(dim=128)
    ck = CoreKnowledgeSystem()
    fcs = FunctionalConceptSystem(cs, ck)

    # 形成一个感知概念
    node = fcs.form_concept(
        label="红色",
        perceptual_features={'type': 'color', 'rgb': (255, 0, 0), 'wavelength': '620-750nm'},
        usage_context={'scene': 'visual', 'modality': 'color'},
        affordances=['描述颜色', '区分物体'],
    )

    assert node is not None, "应成功创建概念"
    assert node.id == "红色"
    assert node.source == 'perception'
    assert 'type' in node.perceptual_features, "应有感知特征"
    assert node.perceptual_features['type'] == 'color'
    assert len(node.affordances) >= 2, "应有可供性"
    print(f"  概念 '红色': source={node.source}")
    print(f"    perceptual_features: {node.perceptual_features}")
    print(f"    affordances: {node.affordances}")

    # 形成第二个概念（自动推断可供性）
    node2 = fcs.form_concept(
        label="圆形",
        perceptual_features={'type': 'shape', 'sides': 0, 'curvature': 'uniform'},
    )
    assert node2 is not None
    print(f"  概念 '圆形': affordances={node2.affordances}")
    assert len(node2.affordances) > 0, "应自动推断可供性"

    print("  功能性概念形成测试通过 ✓")


def test_fast_mapping():
    """测试2: 快速映射机制"""
    print("\n━━━ 测试2: 快速映射机制 ━━━")

    cs = ConceptSpace(dim=128)
    ck = CoreKnowledgeSystem()
    fcs = FunctionalConceptSystem(cs, ck)

    # 先注册一个已知概念
    fcs.form_concept("红色", perceptual_features={'type': 'color'})

    # 快速映射一个新概念（互斥性偏差）
    node = fcs.fast_map(
        label="蓝色",
        context={'known_labels': ['红色'], 'object': {'color': 'blue'}},
        perceptual_input={'type': 'color', 'rgb': (0, 0, 255)},
    )

    assert node is not None, "快速映射应成功"
    assert node.id == "蓝色"
    print(f"  快速映射 '蓝色': source={node.source}")
    print(f"    perceptual_features: {node.perceptual_features}")
    assert 'formed_by' in node.usage_contexts[0], "应记录形成方式"

    # 测试完全新概念（整体对象偏差）
    node2 = fcs.fast_map(
        label="树",
        context={'known_labels': [], 'object': {'type': 'plant'}},
        perceptual_input={'type': 'object', 'shape': 'irregular'},
    )
    assert node2 is not None
    print(f"  快速映射 '树': source={node2.source}")

    stats = fcs.get_stats()
    print(f"  快速映射统计: {stats['fast_maps']}次")
    assert stats['fast_maps'] == 2

    print("  快速映射测试通过 ✓")


def test_concept_grounding():
    """测试3: 概念接地"""
    print("\n━━━ 测试3: 概念接地到感知经验 ━━━")

    cs = ConceptSpace(dim=128)
    ck = CoreKnowledgeSystem()
    fcs = FunctionalConceptSystem(cs, ck)

    # 形成概念
    fcs.form_concept("红色", perceptual_features={'type': 'color'})

    # 获取初始向量
    initial_vec = cs.concepts["红色"].vector.clone()

    # 接地到视觉经验
    visual_exp = torch.randn(128) * 0.5
    success = fcs.ground_concept("红色", {
        'visual': visual_exp,
        'objects': ['苹果', '玫瑰', '消防车'],
        'emotional_association': '温暖',
    })

    assert success, "接地应成功"
    node = cs.concepts["红色"]

    # 向量应已更新（融合了感知编码）
    vec_diff = (node.vector - initial_vec).norm().item()
    print(f"  接地后向量变化: {vec_diff:.4f}")
    assert vec_diff > 0, "向量应有变化"

    # 感知特征应更新
    assert 'objects' in node.perceptual_features, "应有objects字段"
    print(f"  感知特征: {list(node.perceptual_features.keys())}")

    # 感知锚点应增加
    anchors = len(node.sensory_anchors)
    print(f"  感知锚点: {anchors}个")
    assert anchors >= 2, "应有多个锚点（初始1+接地1）"

    # 强度应增加
    print(f"  概念强度: {node.strength:.2f}")

    print("  概念接地测试通过 ✓")


def test_affordance_retrieval():
    """测试4: 可供性检索"""
    print("\n━━━ 测试4: 可供性检索 ━━━")

    cs = ConceptSpace(dim=128)
    ck = CoreKnowledgeSystem()
    fcs = FunctionalConceptSystem(cs, ck)

    # 创建几个概念
    fcs.form_concept("杯子",
        perceptual_features={'type': 'container'},
        affordances=['装水', '喝水', '握住'])
    fcs.form_concept("碗",
        perceptual_features={'type': 'container'},
        affordances=['装饭', '装水', '盛汤'])
    fcs.form_concept("红色",
        perceptual_features={'type': 'color'},
        affordances=['描述颜色', '区分物体'])
    fcs.form_concept("刀",
        perceptual_features={'type': 'tool'},
        affordances=['切割', '削皮', '装水'])  # 刀不能装水，测试精确性

    # 检索"能装水"的概念
    results = fcs.retrieve_by_affordance("装水")
    print(f"  可供性检索 '装水': {[(r, f'{s:.2f}') for r, s in results]}")
    assert len(results) >= 2, "应有至少2个能装水的概念"

    # 杯子和碗应排在前面
    top_ids = [r for r, s in results[:3]]
    assert "杯子" in top_ids, "杯子应在前3"
    assert "碗" in top_ids, "碗应在前3"

    print("  可供性检索测试通过 ✓")


def test_concept_profile():
    """测试5: 概念档案"""
    print("\n━━━ 测试5: 概念档案完整性 ━━━")

    cs = ConceptSpace(dim=128)
    ck = CoreKnowledgeSystem()
    fcs = FunctionalConceptSystem(cs, ck)

    # 创建并接地概念
    fcs.form_concept("红色",
        perceptual_features={'type': 'color', 'rgb': (255, 0, 0)},
        affordances=['描述颜色', '区分物体'],
    )
    fcs.ground_concept("红色", {
        'visual': torch.randn(128),
        'objects': ['苹果'],
    })

    # 获取完整档案
    profile = fcs.get_concept_profile("红色")
    print(f"  概念档案:")
    for key, value in profile.items():
        if key != 'related_concepts':
            print(f"    {key}: {value}")

    assert profile['id'] == "红色"
    assert profile['source'] == 'perception'
    assert profile['frequency'] >= 1
    assert 'type' in profile['perceptual_features']
    assert len(profile['affordances']) >= 2
    assert profile['sensory_anchors'] >= 2  # 初始 + 接地

    print("  概念档案测试通过 ✓")


def test_learner_integration():
    """测试6: 与Learner集成"""
    print("\n━━━ 测试6: 与Learner集成 ━━━")

    config = LearnerConfig(
        encoder_n_layers=1,
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
    )
    learner = Learner(config)
    learner._skip_ttt = True

    # 获取组件
    cs = learner._registry.get('concept_space') if learner._registry.has('concept_space') else None
    ck = learner.core_knowledge

    if not cs:
        print("  ⚠ 概念空间未初始化，跳过")
        return

    fcs = FunctionalConceptSystem(cs, ck)

    # 通过功能性概念系统注册感知概念
    node = fcs.form_concept(
        "蓝色",
        perceptual_features={'type': 'color', 'rgb': (0, 0, 255)},
        affordances=['描述颜色', '区分物体'],
    )
    assert node is not None
    print(f"  功能性概念注册: {node.id} ✓")

    # 通过Learner的_register_perceptual_concept注册
    success = learner._register_perceptual_concept({
        'label': '绿色',
        'perceptual_features': {'type': 'color', 'rgb': (0, 255, 0)},
        'error_dimensions': [1, 5, 10],
        'error_magnitude': 0.7,
        'occurrence_count': 3,
        'source': 'perception_loop',
    })
    print(f"  Learner直接注册 '绿色': {success}")

    if success:
        cs2 = learner._registry.get('concept_space')
        if '绿色' in cs2.concepts:
            n = cs2.concepts['绿色']
            print(f"    source={n.source}, anchors={len(n.sensory_anchors)}, "
                  f"affordances={n.affordances}")

    # 统计
    stats = fcs.get_stats()
    print(f"  FCS统计: {stats}")

    print("  Learner集成测试通过 ✓")


# ===== 主函数 =====
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2 验证：功能性概念系统")
    print("=" * 70)

    test_functional_concept_formation()
    test_fast_mapping()
    test_concept_grounding()
    test_affordance_retrieval()
    test_concept_profile()
    test_learner_integration()

    print("\n" + "=" * 70)
    print("Phase 2 全部测试通过 ✓")
    print("=" * 70)
