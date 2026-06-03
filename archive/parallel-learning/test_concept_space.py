"""Phase 2 验证：概念空间 + 激活扩散推理

验证目标：
1. 概念空间是否正确注册和学习
2. 激活扩散推理 vs 原有推理的回答质量
3. 128维编码器的区分力提升
4. Hebbian关系学习的效果
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


def test_phase2():
    print("=" * 70)
    print("Phase 2 验证：概念空间 + 激活扩散推理 + 128维")
    print("=" * 70)

    config = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
    )

    print(f"\n配置: obs_dim={config.obs_dim}")

    learner = Learner(config)

    # ===== 实验1: 编码器区分力（128维 vs 40维）=====
    print("\n--- 实验1: 128维编码器区分力 ---")
    if hasattr(learner, '_learnable_encoder'):
        enc = learner._learnable_encoder
        print(f"  encoder d_model={enc.d_model}, params={sum(p.numel() for p in enc.parameters())}")

        test_pairs = [
            ("数学", "物理学"),
            ("数学", "化学"),
            ("数学", "苹果"),
            ("研究", "学科"),
            ("研究", "睡觉"),
        ]
        print("  编码区分力（cosine similarity）:")
        for a, b in test_pairs:
            with torch.no_grad():
                va = enc(a)
                vb = enc(b)
                sim = torch.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            print(f"    cos('{a}', '{b}') = {sim:.4f}")

    # ===== 实验2: 学习 + 概念空间构建 =====
    print("\n--- 实验2: 学习20条Wiki语料 ---")
    t0 = time.time()
    learner._skip_ttt = True  # 加速批量学习
    for i, text in enumerate(WIKI_TEXTS):
        learner.learn_from_text(text)
    learner._skip_ttt = False
    t_learn = time.time() - t0
    print(f"  耗时: {t_learn:.1f}s ({t_learn/20:.1f}s/text)")

    # 概念空间统计
    cs = learner._registry.get('concept_space')
    cs_stats = cs.get_stats()
    print(f"\n  概念空间统计:")
    print(f"    概念数: {cs_stats['total_concepts']}")
    print(f"    关系数: {cs_stats['total_relations']}")
    print(f"    Top概念(按强度): {cs_stats['top_by_strength'][:5]}")

    # 知识图谱统计
    kg = learner.knowledge
    print(f"\n  知识图谱统计:")
    print(f"    实体数: {len(kg.entities)}")
    print(f"    关系数: {len(kg.relations)}")

    # ===== 实验3: 激活扩散推理 =====
    print("\n--- 实验3: 激活扩散推理测试 ---")
    test_queries = [
        "什么是数学？",
        "物理学研究什么？",
        "什么是人工智能？",
        "化学和生物学有什么关系？",
    ]

    for q in test_queries:
        print(f"\n  Q: {q}")

        # 概念空间激活
        activated = cs.activate(q, top_k=5, spread_depth=2)
        if activated:
            print(f"    激活的概念 (top 5):")
            for ac in activated[:5]:
                path_str = " -> ".join(ac.path)
                print(f"      {ac.concept_id}: activation={ac.activation:.3f} depth={ac.depth} path=[{path_str}]")

        # 完整think()
        answer = learner.think(q)
        print(f"    A: {answer[:150]}")

    # ===== 实验4: 向量类比 =====
    print("\n--- 实验4: 向量类比推理 ---")
    analogies = [
        ("数学", "数学家", "物理学"),
        ("化学", "化学家", "生物学"),
    ]
    for a, b, c in analogies:
        results = cs.analogy(a, b, c, top_k=3)
        if results:
            top = results[0]
            print(f"    {a}:{b} :: {c}:? -> {top[0]} (sim={top[1]:.3f})")
        else:
            print(f"    {a}:{b} :: {c}:? -> (无结果)")

    # ===== 实验5: Hebbian关系学习验证 =====
    print("\n--- 实验5: Hebbian关系学习 ---")
    test_pairs = [
        ("数学", "物理学"),
        ("数学", "学科"),
        ("人工智能", "计算机科学"),
    ]
    for a, b in test_pairs:
        related = cs.get_related(a, top_k=5)
        b_score = next((s for cid, s in related if cid == b), 0.0)
        va = cs.concepts.get(a)
        vb = cs.concepts.get(b)
        if va and vb:
            sim = torch.cosine_similarity(
                va.vector.unsqueeze(0), vb.vector.unsqueeze(0)
            ).item()
            rel = cs.relations.get(a, {}).get(b, 0.0)
            print(f"    {a} <-> {b}: vec_sim={sim:.3f} relation={rel:.3f} related_score={b_score:.3f}")

    print("\n" + "=" * 70)
    print("Phase 2 验证完成")
    print("=" * 70)


if __name__ == "__main__":
    test_phase2()
