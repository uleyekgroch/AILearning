"""Phase 5 验证：Think() 质量飞跃 + 数据规模化

验证目标：
1. Think() 不再输出碎片概念（"穷级数"、"物理学理"等）
2. 回答更自然（利用KG关系构建句式）
3. 50条多样化语料的规模化效果
4. 编码区分力在更多数据上的表现
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

# ===== 50条多样化语料 =====
TEXTS = [
    # 数学 (5)
    "数学是研究数量、结构、变化、空间以及信息等概念的一门学科",
    "数学分析是数学的一个分支，主要研究极限、微分、积分和无穷级数",
    "概率论是研究随机现象数量规律的数学分支",
    "统计学是应用数学的一个分支，通过收集分析解释数据来推断事物本质",
    "信息论是研究信息的计量传输变换和存储的学科",
    # 物理 (5)
    "物理学是研究物质运动最一般规律和物质基本结构的学科",
    "量子力学是描述微观粒子运动规律的物理学理论",
    "牛顿力学是经典力学的基础，描述宏观物体的运动规律",
    "热力学是研究热现象中物质系统在平衡状态下的性质的学科",
    "光学是研究光的本质传播规律及其应用的物理学分支",
    # 化学 (5)
    "化学是研究物质的组成结构性质变化规律的自然科学",
    "有机化学是研究有机化合物的结构性质制备的化学分支",
    "无机化学是研究无机化合物的性质结构和反应的化学分支",
    "分析化学是鉴定物质成分和测量各成分含量的学科",
    "物理化学是用物理方法研究化学过程的学科",
    # 生物 (5)
    "生物学是研究生命现象和生命活动规律的自然科学",
    "分子生物学是在分子水平上研究生命现象的科学",
    "细胞生物学是研究细胞的结构功能和生命活动规律的科学",
    "遗传学是研究生物体遗传和变异规律的科学",
    "生态学是研究生物与环境相互关系的科学",
    # 计算机 (5)
    "计算机科学是研究信息与计算的理论基础以及它们在计算机系统中的实现与应用的学科",
    "人工智能是计算机科学的一个分支，它企图了解智能的实质",
    "机器学习是人工智能的一个子领域，通过数据训练模型来完成任务",
    "数据结构是计算机存储和组织数据的方式",
    "算法是解决特定问题的一系列明确指令",
    # 地球/天文 (4)
    "地球科学是研究地球系统的科学，包括大气科学、地理学、地质学等",
    "天文学是研究宇宙中天体和天体系统的科学",
    "气象学是研究大气中物理现象和天气变化规律的科学",
    "地质学是研究地球的物质组成、内部结构和演化历史的科学",
    # 社会科学 (6)
    "经济学是研究人类社会在各个发展阶段的各种经济活动和经济关系的学科",
    "心理学是研究人类心理现象及其影响下的精神功能和行为活动的科学",
    "社会学是研究社会行为和人类群体的学科",
    "政治学是研究政治权力政治制度和政治行为的学科",
    "历史学是记录和研究人类过去活动的学科",
    "哲学是研究世界本质存在价值等根本问题的学科",
    # 文学/语言 (4)
    "文学是以语言文字为工具形象化地反映客观现实的艺术",
    "语言学是研究人类语言的结构功能和历史的学科",
    "中国文学是中国各民族文学的统称",
    "修辞学是研究语言表达技巧的学科",
    # 医学 (3)
    "医学是研究人类生命过程及防治疾病的科学",
    "外科学是医学的一个重要分支，以手术为主要治疗手段",
    "药理学是研究药物与生物体之间相互作用的科学",
    # 工程 (3)
    "电子工程是研究电子器件和电子系统的工程学科",
    "机械工程是研究机械装置设计和制造的工程学科",
    "土木工程是研究基础设施建设和维护的工程学科",
    # 艺术 (5)
    "音乐是通过有组织的声音来表达人类情感的艺术形式",
    "绘画是用色彩和线条在平面上创造视觉形象的艺术",
    "雕塑是用各种材料塑造三维空间形象的艺术",
    "建筑学是研究建筑物设计和建造的学科",
    "摄影是用光线记录影像的技术和艺术",
]

# 测试问题（覆盖不同领域和难度）
QUESTIONS = [
    "什么是数学？",
    "物理学研究什么？",
    "化学和生物学有什么关系？",
    "人工智能和计算机科学有什么关系？",
    "量子力学研究什么？",
    "经济学是什么学科？",
    "音乐是什么？",
    "医学研究什么？",
    "概率论和统计学有什么关系？",
    "地球科学包括哪些领域？",
]

ENCODING_TEST_PAIRS = [
    ("数学", "物理学", "related"),
    ("数学", "数学分析", "related"),
    ("人工智能", "机器学习", "related"),
    ("化学", "有机化学", "related"),
    ("物理学", "量子力学", "related"),
    ("生物学", "分子生物学", "related"),
    ("化学", "生物学", "related"),
    ("音乐", "绘画", "related"),
    ("数学", "苹果", "unrelated"),
    ("物理学", "香蕉", "unrelated"),
    ("化学", "篮球", "unrelated"),
    ("心理学", "汽车", "unrelated"),
    ("天文学", "做饭", "unrelated"),
    ("经济学", "游泳", "unrelated"),
    ("数学", "心理学", "unrelated"),
    ("热力学", "音乐", "unrelated"),
    ("概率论", "雕塑", "unrelated"),
    ("外科学", "摄影", "unrelated"),
]


def count_fragments(text: str) -> int:
    """检查文本中是否包含碎片概念"""
    # 碎片的特征：包含不自然的词边界
    function_chars = set('是的有在了和与被把让给从到以也而')
    words = text.replace('。', ' ').split()
    fragments = 0
    for w in words:
        # 提取概念（排除"与"、"是"、"导致"等连接词）
        if any(fc in w for fc in function_chars) or len(w) < 2:
            continue
        # 太长的词可能是碎片拼接
        if len(w) > 6:
            fragments += 1
    return fragments


def test_phase5():
    print("=" * 70)
    print("Phase 5 验证：Think() 质量飞跃 + 数据规模化 (50条语料)")
    print("=" * 70)

    # ===== 实验1: 基线（无对比学习）=====
    print("\n--- 实验1: 基线编码区分力 ---")

    config_base = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
        contrastive_enabled=False,
    )
    learner_base = Learner(config_base)
    learner_base._skip_ttt = True

    for i, text in enumerate(TEXTS[:20]):
        learner_base.learn_from_text(text)

    if hasattr(learner_base, '_learnable_encoder'):
        enc = learner_base._learnable_encoder
        related_sims, unrelated_sims = [], []
        for a, b, cat in ENCODING_TEST_PAIRS:
            with torch.no_grad():
                sim = torch.cosine_similarity(enc(a).unsqueeze(0), enc(b).unsqueeze(0)).item()
            (related_sims if cat == 'related' else unrelated_sims).append(sim)
        baseline_gap = (sum(related_sims)/len(related_sims)) - (sum(unrelated_sims)/len(unrelated_sims))
        print(f"  cos(related)={sum(related_sims)/len(related_sims):.4f}, "
              f"cos(unrelated)={sum(unrelated_sims)/len(unrelated_sims):.4f}")
        print(f"  基线 GAP = {baseline_gap:.4f}")

    # ===== 实验2: 完整训练（50条 + 对比学习 + 回放）=====
    print("\n--- 实验2: 50条语料全流程训练 ---")

    config = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
        contrastive_enabled=True,
        contrastive_temperature=0.2,
        contrastive_warmup_texts=5,
        contrastive_update_freq=1,
    )
    learner = Learner(config)
    learner._skip_ttt = True

    t0 = time.time()
    for i, text in enumerate(TEXTS):
        learner.learn_from_text(text)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(TEXTS)}] ({elapsed:.1f}s)")
    t_learn = time.time() - t0
    print(f"  学习总耗时: {t_learn:.1f}s ({t_learn/len(TEXTS):.1f}s/text)")

    # 回放训练
    print("\n  回放训练...")
    ct = learner._registry.get('contrastive_trainer') if learner._registry.has('contrastive_trainer') else None
    if ct:
        cs = learner._registry.get('concept_space')
        high_quality_pairs = []
        if cs:
            for cid in list(cs.concepts.keys())[:80]:
                related = cs.get_related(cid, top_k=3)
                for rel_id, score in related:
                    if score > 0.05:
                        high_quality_pairs.append((cid, rel_id))
        t1 = time.time()
        replay_stats = ct.replay_epochs(
            positive_pairs=high_quality_pairs,
            optimizer=learner._text_optimizer,
            n_epochs=5,
        )
        t_replay = time.time() - t1
        print(f"    回放: {replay_stats['pairs_used']}对 × {replay_stats['epochs']}轮 ({t_replay:.1f}s)")
        print(f"    回放损失: {replay_stats['total_loss']:.4f}")

        # 同步概念空间向量
        if cs:
            with torch.no_grad():
                for cid, node in cs.concepts.items():
                    new_vec = learner._learnable_encoder(cid).detach()
                    node.vector = torch.nn.functional.normalize(new_vec, p=2, dim=0)

    # ===== 实验3: 编码区分力 =====
    print("\n--- 实验3: 编码区分力（训练后）---")

    if hasattr(learner, '_learnable_encoder'):
        enc = learner._learnable_encoder
        related_sims, unrelated_sims = [], []
        for a, b, cat in ENCODING_TEST_PAIRS:
            with torch.no_grad():
                sim = torch.cosine_similarity(enc(a).unsqueeze(0), enc(b).unsqueeze(0)).item()
            (related_sims if cat == 'related' else unrelated_sims).append(sim)

        post_gap = (sum(related_sims)/len(related_sims)) - (sum(unrelated_sims)/len(unrelated_sims))
        print(f"  cos(related)={sum(related_sims)/len(related_sims):.4f}, "
              f"cos(unrelated)={sum(unrelated_sims)/len(unrelated_sims):.4f}")
        print(f"  训练后 GAP = {post_gap:.4f}")
        improvement = post_gap - baseline_gap
        print(f"  提升: {improvement:+.4f}")

    # ===== 实验4: Think() 质量（核心！）=====
    print("\n--- 实验4: Think() 质量（50条语料）---")
    learner._skip_ttt = False

    total_fragments = 0
    total_questions = len(QUESTIONS)

    for q in QUESTIONS:
        answer = learner.think(q)
        # 统计碎片
        fragments = count_fragments(answer)
        total_fragments += fragments
        quality = "OK" if fragments == 0 else f"碎片×{fragments}"
        print(f"  Q: {q}")
        print(f"  A: {answer[:200]}")
        print(f"  [{quality}]")
        print()

    print(f"  碎片总数: {total_fragments} / {total_questions} 个问题")

    # ===== 实验5: 概念空间质量 =====
    print("\n--- 实验5: 概念空间质量 ---")
    cs = learner._registry.get('concept_space')
    if cs:
        cs_stats = cs.get_stats()
        print(f"  概念数: {cs_stats['total_concepts']}")
        print(f"  关系数: {cs_stats['total_relations']}")
        print(f"  Top-10 概念(按强度):")
        for cid, strength in cs_stats['top_by_strength'][:10]:
            print(f"    '{cid}': strength={strength}")

        # 激活扩散测试
        results = cs.activate("数学和自然科学", top_k=5, spread_depth=2)
        print(f"\n  激活扩散('数学和自然科学'):")
        for ac in results[:8]:
            print(f"    {ac.concept_id}: activation={ac.activation:.4f} (depth={ac.depth})")

    # ===== 统计学习质量 =====
    print("\n--- 实验6: 统计学习质量 ---")
    stat_learner = learner._registry.get('statistical_learner')
    if stat_learner:
        stat_stats = stat_learner.get_stats()
        print(f"  统计学习状态: {stat_stats}")
        print(f"  Top-5 涌现概念:")
        for cid, conf in stat_stats.get('top_concepts', stat_stats.get('top_emergent', []))[:5]:
            print(f"    '{cid}': confidence={conf:.3f}")

    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("Phase 5 总结")
    print("=" * 70)
    print(f"  语料规模: {len(TEXTS)} 条")
    print(f"  学习耗时: {t_learn:.1f}s")
    if ct:
        ct_stats = ct.get_stats()
        print(f"  对比学习更新: {ct_stats['total_updates']}次")
        print(f"  内存银行: {ct_stats['memory_bank_size']}概念")
        print(f"  训练cos_gap: {ct_stats['last_cos_gap']:.4f}")
    print(f"  编码GAP: {baseline_gap:.4f} → {post_gap:.4f} ({improvement:+.4f})")
    print(f"  Think()碎片: {total_fragments}")
    if ct and ct_stats.get('avg_positive_sim') and ct_stats.get('avg_negative_sim'):
        print(f"  正/负样本相似度: {ct_stats['avg_positive_sim']:.4f} / {ct_stats['avg_negative_sim']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    test_phase5()
