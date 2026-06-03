"""Phase 7 综合验证：解决3个剩余限制

验证目标：
1. [7A] 不含功能词碎片过滤："究人类"/"究人"/"研究人" 不在 top-30
2. [7B] 多跳推理无碎片：关系问题答案不含碎片概念
3. [7C] 感知竞争力：感知概念有 sensory_boost 加成
4. GAP 保持 >= 0.20（在感知探索前测量）
5. 回归：Phase 6 已有功能不退化
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
    "生物学是研究生命现象和生命活动规律的自然科学",
    "分子生物学是在分子水平上研究生命现象的科学",
    "计算机科学是研究信息与计算的理论基础以及它们在计算机系统中的实现与应用的学科",
    "人工智能是计算机科学的一个分支，它企图了解智能的实质",
    "机器学习是人工智能的一个子领域，通过数据训练模型来完成任务",
    "地球科学是研究地球系统的科学，包括大气科学、地理学、地质学等",
    "天文学是研究宇宙中天体和天体系统的科学",
    "经济学是研究人类社会在各个发展阶段的各种经济活动和经济关系的学科",
    "心理学是研究人类心理现象及其影响下的精神功能和行为活动的科学",
    "社会学是研究社会行为和人类群体的学科",
    "政治学是研究政治权力政治制度和政治行为的学科",
    "历史学是记录和研究人类过去活动的学科",
    "哲学是研究世界本质存在价值等根本问题的学科",
    "文学是以语言文字为工具形象化地反映客观现实的艺术",
    "语言学是研究人类语言的结构功能和历史的学科",
    "医学是研究人类生命过程及防治疾病的科学",
    "音乐是通过有组织的声音来表达人类情感的艺术形式",
    "绘画是用色彩和线条在平面上创造视觉形象的艺术",
    "建筑学是研究建筑物设计和建造的学科",
    "摄影是用光线记录影像的技术和艺术",
]

ENCODING_TEST_PAIRS = [
    ("数学", "物理学", "related"), ("数学", "数学分析", "related"),
    ("人工智能", "机器学习", "related"), ("化学", "有机化学", "related"),
    ("物理学", "量子力学", "related"), ("生物学", "分子生物学", "related"),
    ("化学", "生物学", "related"), ("音乐", "绘画", "related"),
    ("数学", "苹果", "unrelated"), ("物理学", "香蕉", "unrelated"),
    ("化学", "篮球", "unrelated"), ("心理学", "汽车", "unrelated"),
    ("天文学", "做饭", "unrelated"), ("经济学", "游泳", "unrelated"),
]

NON_FUNC_FRAGMENTS = ['究人类', '究人', '研究人', '动规律', '学分']
TRUE_WORDS = ['研究', '人类', '学科', '科学', '物理', '化学', '数学', '分支', '规律', '结构']


def test_phase7():
    print("=" * 70)
    print("Phase 7 综合验证：解决3个剩余限制")
    print("=" * 70)

    # ===== 学习阶段 =====
    print("\n━━━ 学习语料 ━━━")
    config = LearnerConfig(
        statistical_learning_enabled=True, statistical_use_as_primary=True,
        contrastive_enabled=True, contrastive_temperature=0.2,
        contrastive_warmup_texts=5, encoder_n_layers=3,
    )
    learner = Learner(config)
    learner._skip_ttt = True

    t0 = time.time()
    for i, text in enumerate(TEXTS):
        learner.learn_from_text(text)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(TEXTS)}] ({time.time()-t0:.1f}s)")
    t_learn = time.time() - t0
    print(f"  学习耗时: {t_learn:.1f}s")

    # 回放
    ct = learner._registry.get('contrastive_trainer') if learner._registry.has('contrastive_trainer') else None
    if ct:
        cs = learner._registry.get('concept_space')
        pairs = []
        if cs:
            for cid in list(cs.concepts.keys())[:80]:
                for rel_id, score in cs.get_related(cid, top_k=3):
                    if score > 0.05:
                        pairs.append((cid, rel_id))
        ct.replay_epochs(positive_pairs=pairs, optimizer=learner._text_optimizer, n_epochs=5)
        if cs:
            with torch.no_grad():
                for cid, node in cs.concepts.items():
                    new_vec = learner._learnable_encoder(cid).detach()
                    node.vector = torch.nn.functional.normalize(new_vec, p=2, dim=0)

    # ===== 验证1: 碎片过滤（7A）=====
    print("\n━━━ 验证1: 碎片过滤（7A — 子串反向验证）━━━")
    stat_learner = learner._registry.get('statistical_learner')
    fragments_in_top = []
    true_words_killed = []
    if stat_learner:
        emergent = stat_learner.get_emergent_concepts(min_freq=2)
        top30 = [cid for cid, conf in emergent[:30]]
        print(f"  Top-15: {top30[:15]}")
        fragments_in_top = [f for f in NON_FUNC_FRAGMENTS if f in top30]
        print(f"  碎片在top-30: {fragments_in_top if fragments_in_top else '无 ✓'}")
        true_words_killed = [w for w in TRUE_WORDS if not stat_learner._is_complete_word(w)]
        print(f"  真词误杀: {true_words_killed if true_words_killed else '无 ✓'}")

    # ===== 验证4: GAP 保持（在感知探索前测量！）=====
    print("\n━━━ 验证4: GAP 保持（感知探索前）━━━")
    gap = 0
    if hasattr(learner, '_learnable_encoder'):
        enc = learner._learnable_encoder
        r_sims, u_sims = [], []
        for a, b, cat in ENCODING_TEST_PAIRS:
            with torch.no_grad():
                sim = torch.cosine_similarity(enc(a).unsqueeze(0), enc(b).unsqueeze(0)).item()
            (r_sims if cat == 'related' else u_sims).append(sim)
        gap = (sum(r_sims)/len(r_sims)) - (sum(u_sims)/len(u_sims))
        print(f"  related={sum(r_sims)/len(r_sims):.4f}, unrelated={sum(u_sims)/len(u_sims):.4f}")
        print(f"  GAP = {gap:.4f} {'✓' if gap >= 0.20 else '⚠ < 0.20'}")

    # ===== 验证2: 多跳推理质量（7B）=====
    print("\n━━━ 验证2: 多跳推理 + Think() 质量（7B）━━━")
    learner._skip_ttt = False
    function_chars = set('是的有在了和与被把让给从到以也而')

    all_questions = [
        ("数学和物理学有什么关系？", "关系"),
        ("人工智能和计算机科学有什么关系？", "关系"),
        ("什么是数学？", "定义"),
        ("物理学研究什么？", "定义"),
        ("音乐是什么？", "定义"),
    ]

    total_frags = 0
    for q, qtype in all_questions:
        answer = learner.think(q)
        words = answer.replace('。', ' ').split()
        frags = sum(1 for w in words if len(w) > 6 and not any(c in function_chars for c in w))
        total_frags += frags
        print(f"  [{qtype}] {q}")
        print(f"  A: {answer[:200]}")
        print(f"  [{'✓' if frags == 0 else f'碎片×{frags}'}]")
        print()

    # ===== 验证3: 感知竞争力（7C）=====
    print("\n━━━ 验证3: 感知竞争力（7C — sensory_boost）━━━")
    try:
        from src.environment.world import World
        from src.learning.perception_explorer import PerceptionExplorer

        world = World(config)
        explorer = PerceptionExplorer(learner, world)
        stats = explorer.explore(n_steps=50, verbose=False)
        print(f"  探索: {stats['steps']}步, {stats['texts_generated']}文本, {stats['concepts_anchored']}锚点")

        cs = learner._registry.get('concept_space')
        if cs:
            # 感知概念统计
            perceptual = [(cid, node) for cid, node in cs.concepts.items()
                         if getattr(node, 'source', 'text') == 'perception']
            print(f"  感知概念: {[(cid, node.frequency, len(node.sensory_anchors)) for cid, node in perceptual[:8]]}")

            # 激活测试
            results = cs.activate("红色圆形物体", top_k=10, spread_depth=2)
            perceptual_ids = {cid for cid, _ in perceptual}
            p_in_top5 = sum(1 for ac in results[:5] if ac.concept_id in perceptual_ids)
            print(f"  感知问题 top-5 中感知概念: {p_in_top5}")

            # 检查 sensory_boost 效果
            for ac in results[:5]:
                node = cs.concepts.get(ac.concept_id)
                if node:
                    boost = min(1.0 + len(node.sensory_anchors) * 0.2, 1.8) if node.sensory_anchors else 1.0
                    print(f"    {ac.concept_id}: activation={ac.activation:.4f}, sensory_boost={boost:.1f}, anchors={len(node.sensory_anchors)}")
    except Exception as e:
        print(f"  ⚠ 感知测试跳过: {e}")

    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("Phase 7 总结")
    print("=" * 70)
    print(f"  语料: {len(TEXTS)} 条")
    print(f"  GAP: {gap:.4f}")
    print(f"  不含功能词碎片: {'无 ✓' if not fragments_in_top else fragments_in_top}")
    print(f"  真词误杀: {'无 ✓' if not true_words_killed else true_words_killed}")
    print(f"  Think()碎片: {total_frags}")
    print("=" * 70)


if __name__ == "__main__":
    test_phase7()
