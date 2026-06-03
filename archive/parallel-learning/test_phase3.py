"""Phase 3 验证：感知接地 + 发展阶段级联

验证目标：
1. 多模态编码器投影层工作正常（40d→128d）
2. 发展阶段过滤实际约束学习内容
3. 概念空间的完整统计
4. 全流程回归测试
"""

import sys
import os
import io
import time
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import LearnerConfig
from src.core.learner import Learner

WIKI_TEXTS = [
    "数学是研究数量、结构、变化、空间以及信息等概念的一门学科",
    "物理学是研究物质运动最一般规律和物质基本结构的学科",
    "化学是研究物质的组成、结构、性质、变化规律的自然科学",
    "生物学是研究生命现象和生命活动规律的自然科学",
    "计算机科学是研究信息与计算的理论基础以及它们在计算机系统中的实现与应用的学科",
    "人工智能是计算机科学的一个分支，它企图了解智能的实质",
    "地球科学是研究地球系统的科学，包括大气科学、地理学、地质学等",
    "天文学是研究宇宙中天体和天体系统的科学",
    "经济学是研究人类社会在各个发展阶段的各种经济活动和经济关系的学科",
    "心理学是研究人类心理现象及其影响下的精神功能和行为活动的科学",
    "数学分析是数学的一个分支，主要研究极限、微分、积分和无穷级数",
    "量子力学是描述微观粒子运动规律的物理学理论",
    "有机化学是研究有机化合物的结构、性质、制备的化学分支",
    "分子生物学是在分子水平上研究生命现象的科学",
    "机器学习是人工智能的一个子领域，通过数据训练模型来完成任务",
    "牛顿力学是经典力学的基础，描述宏观物体的运动规律",
    "热力学是研究热现象中物质系统在平衡状态下的性质的学科",
    "概率论是研究随机现象数量规律的数学分支",
    "统计学是应用数学的一个分支，通过收集、分析、解释数据来推断事物本质",
    "信息论是研究信息的计量、传输、变换和存储的学科",
]


def test_phase3():
    print("=" * 70)
    print("Phase 3 验证：感知接地 + 发展阶段级联")
    print("=" * 70)

    config = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
    )

    learner = Learner(config)

    # ===== 实验1: 多模态编码器投影层 =====
    print("\n--- 实验1: 多模态编码器投影层 (40d → 128d) ---")
    perception = learner.perception
    print(f"  编码器输出维度: {perception.output_dim}")

    # 模拟一个感知输入
    fake_obs = {
        'visual': torch.randn(4, 8, 8),
        'auditory': torch.randn(13),
        'position': torch.randn(2),
    }
    encoded = perception.encode(fake_obs)
    print(f"  编码后维度: {encoded.shape}")
    print(f"  维度匹配obs_dim: {encoded.shape[0] == config.obs_dim}")

    # ===== 实验2: 发展阶段过滤 =====
    print("\n--- 实验2: 发展阶段过滤 ---")
    from src.learning.language_development import LanguageStage

    ld = learner.language_development
    print(f"  初始阶段: {ld.current_stage.name}")

    # 验证当前阶段（holophrase）的过滤行为
    test_entities = ["数学", "物理", "化学"]
    test_triples = [("数学", "是", "研究数量的学科", 0.9), ("物理", "导致", "运动", 0.7)]

    filtered_ent, filtered_tri = ld.filter_content(test_entities, test_triples)
    print(f"  %s期过滤: %d triples → %d triples" % (ld.current_stage.name, len(test_triples), len(filtered_tri)))

    # 手动测试前语言期过滤
    from src.learning.language_development import LanguageDevelopmentSystem as LDS, LanguageStage
    ld_pre = LDS(d_model=128, initial_stage='prelinguistic')
    _, filtered_pre = ld_pre.filter_content(test_entities, test_triples)
    print(f"  前语言期过滤(独立): %d triples → %d triples" % (len(test_triples), len(filtered_pre)))
    assert len(filtered_pre) == 0, "前语言期应该过滤掉所有关系"

    # 注册足够概念以晋升到单词期
    for i in range(60):
        ld.register_concept(f"概念_{i}")
    for i in range(35):
        ld.register_relation(f"a_{i}", "是", f"b_{i}")
    ld.check_stage_transition()
    print(f"  注册60概念+35关系后阶段: {ld.current_stage.name}")

    # 单词期：应该允许简单关系
    filtered_ent2, filtered_tri2 = ld.filter_content(test_entities, test_triples)
    print(f"  单词期过滤: {len(test_triples)} triples → {len(filtered_tri2)} triples")

    # ===== 实验3: 全流程学习（含发展阶段约束）=====
    print("\n--- 实验3: 20条Wiki语料全流程学习 ---")

    # 重置learner（用新实例）
    learner2 = Learner(config)
    learner2._skip_ttt = True

    t0 = time.time()
    for i, text in enumerate(WIKI_TEXTS):
        learner2.learn_from_text(text)
    t_learn = time.time() - t0
    print(f"  耗时: {t_learn:.1f}s ({t_learn/20:.1f}s/text)")

    # 统计
    ld2 = learner2.language_development
    progress = ld2.assess_readiness()
    print(f"\n  语言发展阶段: {ld2.current_stage.name}")
    print(f"  概念数: {progress.concepts_known}")
    print(f"  关系数: {progress.relations_known}")
    print(f"  晋升准备度: {progress.readiness_score:.3f}")
    print(f"  被阶段过滤的内容数: {ld2.stats['filtered_by_stage']}")

    cs = learner2._registry.get('concept_space')
    cs_stats = cs.get_stats()
    print(f"\n  概念空间:")
    print(f"    概念数: {cs_stats['total_concepts']}")
    print(f"    关系数: {cs_stats['total_relations']}")

    kg = learner2.knowledge
    print(f"  知识图谱: {len(kg.entities)} 实体, {len(kg.relations)} 关系")

    # ===== 实验4: Think() 质量 =====
    print("\n--- 实验4: Think() 质量 ---")
    learner2._skip_ttt = False
    questions = [
        "什么是数学？",
        "物理学研究什么？",
        "化学和生物学有什么关系？",
    ]
    for q in questions:
        answer = learner2.think(q)
        print(f"  Q: {q}")
        print(f"  A: {answer[:150]}")

    # ===== 实验5: 编码区分力 =====
    print("\n--- 实验5: 128维编码区分力 ---")
    if hasattr(learner2, '_learnable_encoder'):
        enc = learner2._learnable_encoder
        test_pairs = [
            ("数学", "物理学"),
            ("数学", "苹果"),
            ("人工智能", "计算机科学"),
        ]
        for a, b in test_pairs:
            with torch.no_grad():
                va = enc(a)
                vb = enc(b)
                sim = torch.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            print(f"    cos('{a}', '{b}') = {sim:.4f}")

    print("\n" + "=" * 70)
    print("Phase 3 验证完成")
    print("=" * 70)


if __name__ == "__main__":
    test_phase3()
