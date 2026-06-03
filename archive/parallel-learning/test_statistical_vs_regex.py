"""Phase 1 验证：统计学习 vs 正则提取 对比实验

验证目标：
1. 统计学习能否从Wiki语料中涌现出有意义的概念
2. 统计学习涌现的概念 vs 正则提取的实体，哪个质量更高
3. 统计学习对 Think() 回答质量的影响
4. 统计学习的"意外度"是否能驱动好奇心学习
"""

import sys
import os
import time
import io

# Windows 终端 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import LearnerConfig
from src.core.learner import Learner


# 20条真实Wiki中文语料
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


def test_statistical_vs_regex():
    """对比测试：统计学习 vs 正则提取"""

    print("=" * 70)
    print("Phase 1 验证：统计学习 vs 正则提取")
    print("=" * 70)

    # ===== 实验1：纯统计学习 =====
    print("\n--- 实验1：纯统计学习（不用Learner）---")

    from src.learning.statistical_learner import StatisticalLearner
    stat = StatisticalLearner(min_freq=2, min_pmi=0.5)

    print("\n逐条观察文本（模拟学习过程）：")
    for i, text in enumerate(WIKI_TEXTS):
        result = stat.observe(text)
        new_concepts = result.get('new_concepts', [])
        if new_concepts:
            print(f"  文本{i+1}: 新涌现概念 = {new_concepts[:5]}")

    stats = stat.get_stats()
    print(f"\n统计学习最终状态：")
    print(f"  观察文本数: {stats['texts_observed']}")
    print(f"  候选概念数: {stats['candidate_concepts']}")
    print(f"  涌现概念数: {stats['emergent_concepts']}")
    print(f"  涌现关系数: {stats['emergent_relations']}")
    print(f"\n  Top 10 涌现概念:")
    for concept, conf in stats['top_concepts']:
        info = stat.get_concept_info(concept)
        print(f"    '{concept}' conf={conf:.3f} freq={info.frequency} contexts={info.total_contexts} pmi={info.pmi:.2f}")

    print(f"\n  Top 10 涌现关系:")
    for a, b, strength in stats['top_relations']:
        print(f"    '{a}' <-> '{b}' strength={strength:.3f}")

    # ===== 实验2：预测能力 =====
    print("\n--- 实验2：转移概率预测 ---")
    test_contexts = ["数", "物", "学", "科"]
    for ctx in test_contexts:
        preds = stat.predict_next(ctx, top_k=3)
        pred_str = ", ".join(f"'{c}'(p={p:.3f})" for c, p in preds)
        print(f"  '{ctx}' → {pred_str}")

    # ===== 实验3：意外度 =====
    print("\n--- 实验3：意外度（Surprise）---")
    surprise_tests = [
        "数学是研究数量的学科",  # 已多次见过，低意外
        "数学是研究数量的学科",  # 完全重复
        "量子计算是一种全新的计算范式",  # 包含未见过的组合，高意外
        "外星生命可能在木星的卫星上存在",  # 完全新颖，高意外
    ]
    for text in surprise_tests:
        surprise = stat.get_surprise(text)
        print(f"  surprise={surprise:.3f} | {text[:40]}")

    # ===== 实验4：集成到Learner =====
    print("\n--- 实验4：集成到Learner，对比学习效果 ---")

    # 4a. 统计学习模式
    print("\n  [统计学习模式]")
    config_stat = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
        statistical_min_freq=2,
        statistical_min_pmi=0.5,
    )
    learner_stat = Learner(config_stat)
    t0 = time.time()
    for text in WIKI_TEXTS:
        learner_stat.learn_from_text(text)
    t_stat = time.time() - t0
    stat_info = learner_stat.learn_from_text(WIKI_TEXTS[0]).get('statistical_learning', {})
    print(f"    耗时: {t_stat:.1f}s")
    print(f"    涌现概念数: {stat_info.get('emergent_concepts', 0)}")

    # 4b. 正则提取模式
    print("\n  [正则提取模式]")
    config_regex = LearnerConfig(
        statistical_learning_enabled=False,
        statistical_use_as_primary=False,
    )
    learner_regex = Learner(config_regex)
    t0 = time.time()
    for text in WIKI_TEXTS:
        learner_regex.learn_from_text(text)
    t_regex = time.time() - t0
    print(f"    耗时: {t_regex:.1f}s")

    # ===== 实验5：Think() 质量对比 =====
    print("\n--- 实验5：Think() 质量对比 ---")
    questions = [
        "什么是数学？",
        "物理学研究什么？",
        "什么是人工智能？",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        a_stat = learner_stat.think(q)
        a_regex = learner_regex.think(q)
        print(f"  [统计] A: {a_stat[:120]}...")
        print(f"  [正则] A: {a_regex[:120]}...")

    print("\n" + "=" * 70)
    print("验证完成")
    print("=" * 70)


if __name__ == "__main__":
    test_statistical_vs_regex()
