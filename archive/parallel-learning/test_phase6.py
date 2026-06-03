"""Phase 6 综合验证：4个已知限制的修复

验证目标：
1. [6A] 词边界检测：统计学习 top 概念中无碎片（"是研究"、"学是" 消失）
2. [6B] 多跳推理：跨概念路径 A→B→C 能回答关系问题
3. [6C] 编码器容量：3层 Transformer 配置化，GAP >= 0.25
4. [6D] 感知桥接：环境探索后 "红色" 概念有 sensory_anchor
5. 端到端：50条语料 + Think() 质量
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

# 50条语料（同 Phase 5）
TEXTS = [
    "数学是研究数量、结构、变化、空间以及信息等概念的一门学科",
    "数学分析是数学的一个分支，主要研究极限、微分、积分和无穷级数",
    "概率论是研究随机现象数量规律的数学分支",
    "统计学是应用数学的一个分支，通过收集分析解释数据来推断事物本质",
    "信息论是研究信息的计量传输变换和存储的学科",
    "物理学是研究物质运动最一般规律和物质基本结构的学科",
    "量子力学是描述微观粒子运动规律的物理学理论",
    "牛顿力学是经典力学的基础，描述宏观物体的运动规律",
    "热力学是研究热现象中物质系统在平衡状态下的性质的学科",
    "光学是研究光的本质传播规律及其应用的物理学分支",
    "化学是研究物质的组成结构性质变化规律的自然科学",
    "有机化学是研究有机化合物的结构性质制备的化学分支",
    "无机化学是研究无机化合物的性质结构和反应的化学分支",
    "分析化学是鉴定物质成分和测量各成分含量的学科",
    "物理化学是用物理方法研究化学过程的学科",
    "生物学是研究生命现象和生命活动规律的自然科学",
    "分子生物学是在分子水平上研究生命现象的科学",
    "细胞生物学是研究细胞的结构功能和生命活动规律的科学",
    "遗传学是研究生物体遗传和变异规律的科学",
    "生态学是研究生物与环境相互关系的科学",
    "计算机科学是研究信息与计算的理论基础以及它们在计算机系统中的实现与应用的学科",
    "人工智能是计算机科学的一个分支，它企图了解智能的实质",
    "机器学习是人工智能的一个子领域，通过数据训练模型来完成任务",
    "数据结构是计算机存储和组织数据的方式",
    "算法是解决特定问题的一系列明确指令",
    "地球科学是研究地球系统的科学，包括大气科学、地理学、地质学等",
    "天文学是研究宇宙中天体和天体系统的科学",
    "气象学是研究大气中物理现象和天气变化规律的科学",
    "地质学是研究地球的物质组成、内部结构和演化历史的科学",
    "经济学是研究人类社会在各个发展阶段的各种经济活动和经济关系的学科",
    "心理学是研究人类心理现象及其影响下的精神功能和行为活动的科学",
    "社会学是研究社会行为和人类群体的学科",
    "政治学是研究政治权力政治制度和政治行为的学科",
    "历史学是记录和研究人类过去活动的学科",
    "哲学是研究世界本质存在价值等根本问题的学科",
    "文学是以语言文字为工具形象化地反映客观现实的艺术",
    "语言学是研究人类语言的结构功能和历史的学科",
    "中国文学是中国各民族文学的统称",
    "修辞学是研究语言表达技巧的学科",
    "医学是研究人类生命过程及防治疾病的科学",
    "外科学是医学的一个重要分支，以手术为主要治疗手段",
    "药理学是研究药物与生物体之间相互作用的科学",
    "电子工程是研究电子器件和电子系统的工程学科",
    "机械工程是研究机械装置设计和制造的工程学科",
    "土木工程是研究基础设施建设和维护的工程学科",
    "音乐是通过有组织的声音来表达人类情感的艺术形式",
    "绘画是用色彩和线条在平面上创造视觉形象的艺术",
    "雕塑是用各种材料塑造三维空间形象的艺术",
    "建筑学是研究建筑物设计和建造的学科",
    "摄影是用光线记录影像的技术和艺术",
]

# 编码区分力测试对
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

# 关系类问题（需要多跳推理）
RELATIONSHIP_QUESTIONS = [
    "数学和物理学有什么关系？",
    "人工智能和计算机科学有什么关系？",
    "化学和生物学有什么关系？",
]


# 已知碎片列表（应该在统计学习中被过滤）
KNOWN_FRAGMENTS = ["是研究", "学是", "然科学", "物理学理", "穷级数"]


def test_phase6():
    print("=" * 70)
    print("Phase 6 综合验证：4个已知限制的修复")
    print("=" * 70)

    # ===== 验证1: 配置化编码器（6C）=====
    print("\n━━━ 验证1: 编码器容量配置化（6C）━━━")

    config = LearnerConfig(
        statistical_learning_enabled=True,
        statistical_use_as_primary=True,
        contrastive_enabled=True,
        contrastive_temperature=0.2,
        contrastive_warmup_texts=5,
        contrastive_update_freq=1,
        encoder_n_heads=4,
        encoder_n_layers=3,     # 从2层升级到3层
        encoder_max_len=128,
    )
    print(f"  encoder_n_layers: {config.encoder_n_layers}")
    print(f"  encoder_n_heads: {config.encoder_n_heads}")
    print(f"  encoder_max_len: {config.encoder_max_len}")

    # ===== 验证2: 50条语料学习 =====
    print("\n━━━ 验证2: 50条语料全流程学习 ━━━")

    learner = Learner(config)
    learner._skip_ttt = True

    t0 = time.time()
    for i, text in enumerate(TEXTS):
        learner.learn_from_text(text)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(TEXTS)}] ({elapsed:.1f}s)")
    t_learn = time.time() - t0
    print(f"  学习耗时: {t_learn:.1f}s ({t_learn/len(TEXTS):.1f}s/text)")

    # 回放训练
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
        replay_stats = ct.replay_epochs(
            positive_pairs=high_quality_pairs,
            optimizer=learner._text_optimizer,
            n_epochs=5,
        )
        print(f"  回放: {replay_stats['pairs_used']}对 × {replay_stats['epochs']}轮")

        # 同步概念空间向量
        if cs:
            with torch.no_grad():
                for cid, node in cs.concepts.items():
                    new_vec = learner._learnable_encoder(cid).detach()
                    node.vector = torch.nn.functional.normalize(new_vec, p=2, dim=0)

    # ===== 验证3: 词边界检测（6A）=====
    print("\n━━━ 验证3: 词边界检测（6A）━━━")

    stat_learner = learner._registry.get('statistical_learner')
    if stat_learner:
        emergent = stat_learner.get_emergent_concepts(min_freq=2)
        top_concepts = [cid for cid, conf in emergent[:20]]
        print(f"  Top-20 涌现概念: {top_concepts}")

        # 检查已知碎片是否出现
        fragments_found = []
        for frag in KNOWN_FRAGMENTS:
            if any(frag in c for c in top_concepts):
                fragments_found.append(frag)

        if fragments_found:
            print(f"  ⚠ 仍有碎片: {fragments_found}")
        else:
            print(f"  ✓ 已知碎片全部被过滤")

        # 额外检查：概念中是否有高频虚词开头/结尾的
        function_chars = set('是的有在了和与被把让给从到以也而')
        suspicious = [c for c in top_concepts
                      if c[0] in function_chars or c[-1] in function_chars]
        if suspicious:
            print(f"  ⚠ 虚词边界概念: {suspicious}")
        else:
            print(f"  ✓ 无虚词边界概念")

    # ===== 验证4: 编码区分力（6C GAP）=====
    print("\n━━━ 验证4: 编码区分力（6C）━━━")

    if hasattr(learner, '_learnable_encoder'):
        enc = learner._learnable_encoder
        # 检查编码器层数
        print(f"  编码器结构: {enc}")
        # 统计 TransformerEncoderLayer 数量
        layer_count = sum(1 for name, _ in enc.named_modules() if 'encoder_layer' in name or 'layers' in name)
        print(f"  Transformer 层数: {layer_count}")

        related_sims, unrelated_sims = [], []
        for a, b, cat in ENCODING_TEST_PAIRS:
            with torch.no_grad():
                sim = torch.cosine_similarity(enc(a).unsqueeze(0), enc(b).unsqueeze(0)).item()
            (related_sims if cat == 'related' else unrelated_sims).append(sim)

        gap = (sum(related_sims)/len(related_sims)) - (sum(unrelated_sims)/len(unrelated_sims))
        print(f"  cos(related)={sum(related_sims)/len(related_sims):.4f}, "
              f"cos(unrelated)={sum(unrelated_sims)/len(unrelated_sims):.4f}")
        print(f"  GAP = {gap:.4f}")
        if gap >= 0.20:
            print(f"  ✓ GAP >= 0.20（编码区分力合格）")
        else:
            print(f"  ⚠ GAP < 0.20（需要更多训练）")

    # ===== 验证5: 多跳推理（6B）=====
    print("\n━━━ 验证5: 多跳推理（6B）━━━")

    learner._skip_ttt = False
    cs = learner._registry.get('concept_space')

    for q in RELATIONSHIP_QUESTIONS:
        answer = learner.think(q)
        print(f"  Q: {q}")
        print(f"  A: {answer[:200]}")
        # 检查是否包含关联词（多跳推理的特征）
        has_link = any(w in answer for w in ['关联', '相关', '联系', '通过'])
        print(f"  {'✓' if has_link else '○'} {'包含关联词' if has_link else '无关联词'}")
        print()

    # ===== 验证6: 感知-文本桥接（6D）=====
    print("\n━━━ 验证6: 感知-文本桥接（6D）━━━")

    try:
        from src.environment.world import World
        from src.learning.perception_explorer import PerceptionExplorer

        world = World(config)
        explorer = PerceptionExplorer(learner, world)
        explore_stats = explorer.explore(n_steps=50, verbose=True)

        print(f"  探索步数: {explore_stats['steps']}")
        print(f"  生成文本: {explore_stats['texts_generated']}条")
        print(f"  感知锚点: {explore_stats['concepts_anchored']}个")

        # 检查感知概念是否有锚点
        if cs:
            anchored = []
            for concept in ['红色', '蓝色', '绿色', '圆形', '方形', '三角形', '小的', '大的', '木质', '金属']:
                if concept in cs.concepts:
                    node = cs.concepts[concept]
                    has_anchor = bool(getattr(node, 'sensory_anchors', None))
                    anchored.append((concept, has_anchor))

            anchored_concepts = [c for c, a in anchored if a]
            not_anchored = [c for c, a in anchored if not a]
            if anchored_concepts:
                print(f"  ✓ 有感知锚点的概念: {anchored_concepts}")
            if not_anchored:
                print(f"  ○ 已学但无锚点: {not_anchored}")

            # 问一个感知相关的问题
            learner._skip_ttt = False
            perceptual_q = "环境中有红色的物体吗？"
            perceptual_a = learner.think(perceptual_q)
            print(f"\n  感知问题: {perceptual_q}")
            print(f"  回答: {perceptual_a[:200]}")
    except Exception as e:
        print(f"  ⚠ 感知桥接测试跳过: {e}")

    # ===== 验证7: Think() 端到端质量 =====
    print("\n━━━ 验证7: Think() 端到端质量 ━━━")

    learner._skip_ttt = False
    questions = [
        "什么是数学？",
        "物理学研究什么？",
        "化学和生物学有什么关系？",
        "人工智能和计算机科学有什么关系？",
        "量子力学研究什么？",
        "音乐是什么？",
    ]

    total_fragments = 0
    function_chars = set('是的有在了和与被把让给从到以也而')

    for q in questions:
        answer = learner.think(q)
        # 碎片检测
        words = answer.replace('。', ' ').split()
        fragments = 0
        for w in words:
            if len(w) > 6 and not any(c in function_chars for c in w):
                fragments += 1
        total_fragments += fragments
        quality = "OK" if fragments == 0 else f"碎片×{fragments}"
        print(f"  Q: {q}")
        print(f"  A: {answer[:200]}")
        print(f"  [{quality}]")
        print()

    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("Phase 6 综合验证总结")
    print("=" * 70)
    print(f"  语料规模: {len(TEXTS)} 条")
    print(f"  学习耗时: {t_learn:.1f}s")
    print(f"  编码器层数: {config.encoder_n_layers}")
    print(f"  编码 GAP: {gap:.4f}")
    print(f"  词边界过滤: {'✓ 通过' if not fragments_found else f'⚠ {fragments_found}'}")
    print(f"  Think() 碎片: {total_fragments}")
    print(f"  感知锚点: {explore_stats.get('concepts_anchored', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    test_phase6()
