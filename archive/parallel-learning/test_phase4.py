"""Phase 4 验证：对比学习 — 解决编码器"万能相似"问题

验证目标：
1. 对比损失收敛（loss 随学习下降）
2. 编码区分力提升（cos_gap > 0.2）
3. 相关概念更近、不相关更远
4. Think() 质量改善
5. 概念空间激活扩散质量提升
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

# 测试对：(text_a, text_b, category)
# category: 'related' = 应该相似, 'unrelated' = 应该不相似
ENCODING_TEST_PAIRS = [
    # 相关概念（应该相似）
    ("数学", "物理学", "related"),
    ("数学", "数学分析", "related"),
    ("人工智能", "机器学习", "related"),
    ("化学", "有机化学", "related"),
    ("物理学", "量子力学", "related"),
    ("生物学", "分子生物学", "related"),
    ("化学", "生物学", "related"),
    ("计算机科学", "人工智能", "related"),
    # 不相关概念（应该不相似）
    ("数学", "苹果", "unrelated"),
    ("物理学", "香蕉", "unrelated"),
    ("化学", "篮球", "unrelated"),
    ("心理学", "汽车", "unrelated"),
    ("天文学", "做饭", "unrelated"),
    ("经济学", "游泳", "unrelated"),
    ("数学", "心理学", "unrelated"),
    ("热力学", "音乐", "unrelated"),
]


def test_phase4():
    print("=" * 70)
    print("Phase 4 验证：对比学习 — 解决编码器万能相似")
    print("=" * 70)

    # ===== 实验1: 编码区分力基线（学习前）=====
    print("\n--- 实验1: 编码区分力基线（对比学习前）---")

    config = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
        contrastive_enabled=False,  # 先关闭对比学习，测基线
    )
    learner_baseline = Learner(config)
    learner_baseline._skip_ttt = True

    # 快速学习5条文本建立基础编码
    for text in WIKI_TEXTS[:5]:
        learner_baseline.learn_from_text(text)

    # 诊断基线
    if hasattr(learner_baseline, '_learnable_encoder'):
        enc = learner_baseline._learnable_encoder
        related_sims = []
        unrelated_sims = []
        for a, b, cat in ENCODING_TEST_PAIRS:
            with torch.no_grad():
                va = enc(a)
                vb = enc(b)
                sim = torch.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            if cat == 'related':
                related_sims.append(sim)
            else:
                unrelated_sims.append(sim)
            print(f"  cos('{a}', '{b}') = {sim:.4f}  [{cat}]")

        baseline_gap = (sum(related_sims) / len(related_sims)) - (sum(unrelated_sims) / len(unrelated_sims))
        print(f"\n  基线: cos(related)={sum(related_sims)/len(related_sims):.4f}, "
              f"cos(unrelated)={sum(unrelated_sims)/len(unrelated_sims):.4f}")
        print(f"  基线 GAP = {baseline_gap:.4f}")

    # ===== 实验2: 对比学习训练 =====
    print("\n--- 实验2: 对比学习训练（20条语料）---")

    config2 = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
        contrastive_enabled=True,
        contrastive_temperature=0.2,
        contrastive_warmup_texts=3,
        contrastive_update_freq=1,
    )
    learner = Learner(config2)
    learner._skip_ttt = True

    t0 = time.time()
    for i, text in enumerate(WIKI_TEXTS):
        learner.learn_from_text(text)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/20] 已学习 {i+1} 条 ({elapsed:.1f}s)")
    t_learn = time.time() - t0
    print(f"  总耗时: {t_learn:.1f}s ({t_learn/20:.1f}s/text)")

    # ===== 实验2b: 回放训练（额外epochs）=====
    print("\n--- 实验2b: 回放训练（5 epochs）---")
    ct = learner._registry.get('contrastive_trainer') if learner._registry.has('contrastive_trainer') else None
    if ct:
        # 从概念空间获取高质量正样本对
        cs = learner._registry.get('concept_space')
        high_quality_pairs = []
        if cs:
            for cid in list(cs.concepts.keys())[:50]:
                related = cs.get_related(cid, top_k=3)
                for rel_id, score in related:
                    if score > 0.05:
                        high_quality_pairs.append((cid, rel_id))
        print(f"  高质量正样本对: {len(high_quality_pairs)}")

        t1 = time.time()
        replay_stats = ct.replay_epochs(
            positive_pairs=high_quality_pairs,
            optimizer=learner._text_optimizer,
            n_epochs=5,
        )
        t_replay = time.time() - t1
        print(f"  回放耗时: {t_replay:.1f}s")
        print(f"  回放统计: {replay_stats}")

        # 同步概念空间向量
        if cs:
            with torch.no_grad():
                for cid, node in cs.concepts.items():
                    new_vec = learner._learnable_encoder(cid).detach()
                    node.vector = torch.nn.functional.normalize(new_vec, p=2, dim=0)

    # ===== 实验3: 对比学习后编码区分力 =====
    print("\n--- 实验3: 编码区分力（对比学习后）---")

    if hasattr(learner, '_learnable_encoder'):
        enc = learner._learnable_encoder
        related_sims = []
        unrelated_sims = []
        for a, b, cat in ENCODING_TEST_PAIRS:
            with torch.no_grad():
                va = enc(a)
                vb = enc(b)
                sim = torch.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            if cat == 'related':
                related_sims.append(sim)
            else:
                unrelated_sims.append(sim)
            print(f"  cos('{a}', '{b}') = {sim:.4f}  [{cat}]")

        post_gap = (sum(related_sims) / len(related_sims)) - (sum(unrelated_sims) / len(unrelated_sims))
        print(f"\n  对比学习后: cos(related)={sum(related_sims)/len(related_sims):.4f}, "
              f"cos(unrelated)={sum(unrelated_sims)/len(unrelated_sims):.4f}")
        print(f"  对比学习后 GAP = {post_gap:.4f}")

    # ===== 实验4: 对比学习训练器统计 =====
    print("\n--- 实验4: 对比学习训练器统计 ---")

    ct = learner._registry.get('contrastive_trainer')
    if ct:
        stats = ct.get_stats()
        print(f"  总更新次数: {stats['total_updates']}")
        print(f"  平均损失: {stats['avg_loss']}")
        print(f"  内存银行大小: {stats['memory_bank_size']}")
        print(f"  最近cos_gap: {stats['last_cos_gap']}")
        if 'avg_positive_sim' in stats:
            print(f"  平均正样本相似度: {stats['avg_positive_sim']}")
        if 'avg_negative_sim' in stats:
            print(f"  平均负样本相似度: {stats['avg_negative_sim']}")

    # ===== 实验5: 编码质量诊断 =====
    print("\n--- 实验5: 编码质量诊断 ---")

    if hasattr(learner, '_learnable_encoder') and ct:
        diagnosis = ct.diagnose_encoding_quality(ENCODING_TEST_PAIRS)
        print(f"  整体质量: {diagnosis['quality']}")
        print(f"  GAP (related - unrelated): {diagnosis['gap']:.4f}")

    # ===== 实验6: Think() 质量 =====
    print("\n--- 实验6: Think() 质量 ---")
    learner._skip_ttt = False
    questions = [
        "什么是数学？",
        "物理学研究什么？",
        "化学和生物学有什么关系？",
        "人工智能和计算机科学有什么关系？",
    ]
    for q in questions:
        answer = learner.think(q)
        print(f"  Q: {q}")
        print(f"  A: {answer[:200]}")
        print()

    # ===== 实验7: 概念空间质量 =====
    print("\n--- 实验7: 概念空间质量 ---")

    cs = learner._registry.get('concept_space')
    if cs:
        cs_stats = cs.get_stats()
        print(f"  概念数: {cs_stats['total_concepts']}")
        print(f"  关系数: {cs_stats['total_relations']}")
        print(f"  Top-5 概念(按强度):")
        for cid, strength in cs_stats['top_by_strength'][:5]:
            print(f"    '{cid}': strength={strength}")

        # 测试激活扩散
        results = cs.activate("数学和自然科学", top_k=5, spread_depth=2)
        print(f"\n  激活扩散('数学和自然科学'):")
        for ac in results[:8]:
            print(f"    {ac.concept_id}: activation={ac.activation:.4f} (depth={ac.depth})")

    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("Phase 4 总结")
    print("=" * 70)
    if hasattr(learner_baseline, '_learnable_encoder') and hasattr(learner, '_learnable_encoder'):
        improvement = post_gap - baseline_gap
        print(f"  基线 GAP:     {baseline_gap:.4f}")
        print(f"  对比学习 GAP: {post_gap:.4f}")
        print(f"  提升:         {improvement:+.4f}")
        if improvement > 0.05:
            print(f"  结论: 对比学习有效提升了编码区分力!")
        elif improvement > 0:
            print(f"  结论: 有轻微提升，需要更多训练数据或调参")
        else:
            print(f"  结论: 无提升，需要检查对比学习实现")
    print("=" * 70)


if __name__ == "__main__":
    test_phase4()
