#!/usr/bin/env python3
"""真实语料诊断脚本 — 逐步暴露学习系统的结构性问题

使用Wikipedia中文语料逐环节测试：
1. 文本编码质量
2. 实体抽取准确性
3. 关系抽取准确性
4. 嵌入训练有效性
5. 知识图谱存储/检索
6. think() 回答质量
7. consolidate() 整合效果
8. 新机制(14-27)的实际贡献
"""

import sys
import os
import json
import time
import torch
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 加载真实Wiki语料
# ============================================================
def load_wiki_samples(n=20, min_len=100, max_len=500):
    """从Wikipedia中文语料加载样本"""
    wiki_dir = os.path.join(os.path.dirname(__file__), 'data', 'extracted', 'wiki', 'wiki_zh', 'AA')
    samples = []

    if not os.path.exists(wiki_dir):
        print(f"[WARN] Wiki目录不存在: {wiki_dir}")
        return _fallback_samples()

    files = sorted([f for f in os.listdir(wiki_dir) if f.startswith('wiki_')])

    for fname in files:
        fpath = os.path.join(wiki_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    text = doc.get('text', '').strip()
                    title = doc.get('title', '')
                    if min_len <= len(text) <= max_len:
                        samples.append({'title': title, 'text': text})
                        if len(samples) >= n:
                            return samples
                except:
                    continue

    return samples if samples else _fallback_samples()


def _fallback_samples():
    """备用的简单测试文本"""
    return [
        {'title': '数学', 'text': '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。数学透过抽象化和逻辑推理的使用，由计数、计算、量度和对物体形状及运动的观察而产生。'},
        {'title': '物理', 'text': '物理学是研究物质、能量以及它们之间相互作用的自然科学。物理学的范围从最小的亚原子粒子到整个宇宙。'},
        {'title': '化学', 'text': '化学是研究物质的组成、结构、性质以及变化规律的科学。化学是自然科学的一个分支，与物理、生物等学科密切相关。'},
    ]


# ============================================================
# 诊断函数
# ============================================================

def diagnose_encoding(learner, samples):
    """诊断1: 文本编码质量"""
    print("\n" + "="*60)
    print("诊断1: 文本编码质量 (_encode_text)")
    print("="*60)

    encodings = []
    for i, sample in enumerate(samples[:5]):
        text = sample['text']
        t0 = time.time()

        # 编码两次：train=True 和 train=False
        enc_train = learner._encode_text(text, train=True)
        enc_eval = learner._encode_text(text, train=False)

        elapsed = time.time() - t0

        # 检查编码属性
        info = {
            'index': i,
            'title': sample['title'][:20],
            'text_len': len(text),
            'shape': tuple(enc_train.shape),
            'has_grad': enc_train.requires_grad,
            'mean': enc_train.mean().item(),
            'std': enc_train.std().item(),
            'norm': enc_train.norm().item(),
            'time_ms': elapsed * 1000,
        }
        encodings.append(info)

        print(f"\n  [{i}] {sample['title'][:20]} (len={len(text)})")
        print(f"       shape={info['shape']}, grad={info['has_grad']}, time={info['time_ms']:.1f}ms")
        print(f"       mean={info['mean']:.4f}, std={info['std']:.4f}, norm={info['norm']:.4f}")

    # 检查不同文本的编码区分度
    print("\n  --- 编码区分度分析 ---")
    if len(encodings) >= 2:
        vecs = []
        for sample in samples[:5]:
            vecs.append(learner._encode_text(sample['text']).detach())

        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sim = torch.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
                print(f"  cos({samples[i]['title'][:10]}, {samples[j]['title'][:10]}) = {sim:.4f}")

        # 检查所有编码是否几乎相同（区分力不足）
        avg_sim = 0
        count = 0
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sim = torch.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
                avg_sim += sim
                count += 1
        avg_sim /= max(count, 1)

        if avg_sim > 0.95:
            print(f"\n  [!!!] 严重: 平均相似度={avg_sim:.4f} > 0.95 — 编码几乎无区分力!")
        elif avg_sim > 0.8:
            print(f"\n  [!!] 警告: 平均相似度={avg_sim:.4f} > 0.8 — 编码区分力不足")
        else:
            print(f"\n  [OK] 平均相似度={avg_sim:.4f} — 编码有合理区分力")

    return encodings


def diagnose_entity_extraction(learner, samples):
    """诊断2: 实体抽取准确性"""
    print("\n" + "="*60)
    print("诊断2: 实体抽取 (extract_entities_from_repr)")
    print("="*60)

    for i, sample in enumerate(samples[:5]):
        text = sample['text']
        repr_vec = learner._encode_text(text)

        # 抽取实体
        entities = learner._extract_entities_from_repr(text, repr_vec)

        print(f"\n  [{i}] {sample['title'][:20]}")
        print(f"       文本: {text[:80]}...")
        print(f"       实体({len(entities)}): {entities[:15]}")

        # 检查实体质量
        if not entities:
            print("       [!!!] 未抽取到任何实体!")
        else:
            # 检查实体长度分布
            lengths = [len(e) for e in entities]
            short = sum(1 for l in lengths if l <= 1)
            long = sum(1 for l in lengths if l > 10)
            if short > 0:
                print(f"       [!] 有{short}个单字符实体（可能是噪声）")
            if long > 0:
                print(f"       [!] 有{long}个超长实体(>10字符)")

    return True


def diagnose_relation_extraction(learner, samples):
    """诊断3: 关系抽取准确性"""
    print("\n" + "="*60)
    print("诊断3: 关系抽取 (_extract_relations_from_repr)")
    print("="*60)

    for i, sample in enumerate(samples[:5]):
        text = sample['text']
        repr_vec = learner._encode_text(text)
        entities = learner._extract_entities_from_repr(text, repr_vec)

        # 抽取关系
        triples = learner._extract_relations_from_repr(text, entities, repr_vec)

        print(f"\n  [{i}] {sample['title'][:20]}")
        print(f"       文本: {text[:80]}...")
        print(f"       三元组({len(triples)}):")
        for t in triples[:8]:
            if len(t) >= 4:
                print(f"         ({t[0][:15]}) -[{t[1][:10]}]-> ({t[2][:15]}) conf={t[3]:.2f}" if isinstance(t[3], (int, float)) else f"         ({t[0][:15]}) -[{t[1][:10]}]-> ({t[2][:15]})")
            elif len(t) >= 3:
                print(f"         ({t[0][:15]}) -[{t[1][:10]}]-> ({t[2][:15]})")

        if not triples:
            print("       [!!!] 未抽取到任何关系!")
        else:
            # 检查关系质量
            low_conf = sum(1 for t in triples if len(t) >= 4 and isinstance(t[3], float) and t[3] < 0.3)
            if low_conf > len(triples) * 0.5:
                print(f"       [!] {low_conf}/{len(triples)} 个三元组低置信度(<0.3)")

    return True


def diagnose_embedding_training(learner, samples):
    """诊断4: 嵌入训练有效性"""
    print("\n" + "="*60)
    print("诊断4: 嵌入训练 (_train_embedding)")
    print("="*60)

    if not hasattr(learner, '_learnable_encoder'):
        print("  [SKIP] 编码器未初始化")
        return False

    text = samples[0]['text']
    repr_vec = learner._encode_text(text)
    entities = learner._extract_entities_from_repr(text, repr_vec)

    print(f"  文本: {text[:60]}...")
    print(f"  实体({len(entities)}): {entities[:10]}")

    if len(entities) < 2:
        print("  [SKIP] 实体不足2个，无法训练")
        return False

    # 记录训练前后的嵌入
    print("\n  --- 训练前 ---")
    before_text_emb = learner._encode_text(text).detach().clone()
    before_ent_embs = {e: learner._encode_text(e).detach().clone() for e in entities[:5]}

    for e, emb in before_ent_embs.items():
        sim = torch.cosine_similarity(before_text_emb.unsqueeze(0), emb.unsqueeze(0)).item()
        print(f"  cos(text, '{e}') = {sim:.4f}")

    # 执行训练
    print("\n  --- 执行训练 ---")
    has_optimizer = hasattr(learner, '_text_optimizer')
    print(f"  优化器存在: {has_optimizer}")

    try:
        learner._train_embedding(text, entities)
        print("  训练完成(无报错)")
    except Exception as ex:
        print(f"  [!!!] 训练报错: {ex}")
        traceback.print_exc()
        return False

    # 检查训练后变化
    print("\n  --- 训练后(清除缓存) ---")
    # 清除缓存以确保获取新的编码
    if hasattr(learner, '_embedding_cache'):
        learner._embedding_cache.clear()

    after_text_emb = learner._encode_text(text).detach().clone()
    after_ent_embs = {e: learner._encode_text(e).detach().clone() for e in entities[:5]}

    text_diff = (before_text_emb - after_text_emb).norm().item()
    print(f"  文本嵌入变化量: {text_diff:.6f}")

    for e in entities[:5]:
        before_sim = torch.cosine_similarity(before_text_emb.unsqueeze(0), before_ent_embs[e].unsqueeze(0)).item()
        after_sim = torch.cosine_similarity(after_text_emb.unsqueeze(0), after_ent_embs[e].unsqueeze(0)).item()
        print(f"  cos(text, '{e}'): {before_sim:.4f} -> {after_sim:.4f} (delta={after_sim-before_sim:+.4f})")

    if text_diff < 1e-6:
        print("\n  [!!!] 严重: 训练后嵌入无变化 — 梯度更新可能被跳过!")
    else:
        print(f"\n  [OK] 嵌入有变化 (delta_norm={text_diff:.6f})")

    return True


def diagnose_knowledge_graph(learner, samples):
    """诊断5: 知识图谱存储与检索"""
    print("\n" + "="*60)
    print("诊断5: 知识图谱 (learn_from_text -> think)")
    print("="*60)

    # 用5条样本学习
    learned_titles = []
    for i, sample in enumerate(samples[:5]):
        text = sample['text']
        title = sample['title']

        print(f"\n  学习 [{i}]: {title}")
        t0 = time.time()

        try:
            result = learner.learn_from_text(text)
            elapsed = time.time() - t0

            entities = result.get('entities', [])
            triples = result.get('triples', [])
            concepts = result.get('concepts', [])
            verification = result.get('verification', {})

            print(f"    耗时={elapsed:.1f}s, 实体={len(entities)}, 三元组={len(triples)}, 概念={len(concepts)}")

            if verification:
                verified = verification.get('verified', 0)
                total = verification.get('total', 0)
                rate = verified / max(total, 1) * 100
                print(f"    验证: {verified}/{total} ({rate:.0f}%)")

            learned_titles.append(title)
        except Exception as ex:
            print(f"    [!!!] 学习报错: {ex}")
            traceback.print_exc()

    # 检查知识图谱状态
    kg = learner.knowledge
    print(f"\n  --- 知识图谱状态 ---")
    print(f"  实体数: {len(kg.entities)}")
    print(f"  三元组数: {len(kg.triples)}")
    print(f"  概念数: {len(kg.concepts) if hasattr(kg, 'concepts') else 'N/A'}")

    # 显示部分实体
    print(f"\n  实体样例(前10):")
    for eid, entity in list(kg.entities.items())[:10]:
        attrs = entity.attributes if hasattr(entity, 'attributes') else {}
        print(f"    {eid}: attrs={len(attrs)}")

    # 测试检索
    print(f"\n  --- 测试检索 (think) ---")
    for title in learned_titles[:3]:
        question = f"什么是{title}?"
        t0 = time.time()
        answer = learner.think(question)
        elapsed = time.time() - t0
        print(f"\n  Q: {question}")
        print(f"  A: {answer[:200]}...")
        print(f"  (耗时: {elapsed:.1f}s)")

    return True


def diagnose_consolidation(learner):
    """诊断6: 整合效果"""
    print("\n" + "="*60)
    print("诊断6: 整合 (consolidate)")
    print("="*60)

    try:
        t0 = time.time()
        result = learner.consolidate()
        elapsed = time.time() - t0

        print(f"  耗时: {elapsed:.1f}s")
        print(f"  结果keys: {list(result.keys())}")

        for k, v in result.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            elif isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")
            elif isinstance(v, list):
                print(f"  {k}: [{len(v)} items]")
            else:
                print(f"  {k}: {type(v).__name__}")

    except Exception as ex:
        print(f"  [!!!] 整合报错: {ex}")
        traceback.print_exc()

    return True


def diagnose_new_mechanisms(learner, samples):
    """诊断7: 新机制(14-27)实际贡献"""
    print("\n" + "="*60)
    print("诊断7: 新机制(14-27)实际贡献")
    print("="*60)

    mechanisms = {
        'dendritic_system': '树突计算',
        'sleep_replay': '睡眠回放',
        'active_inference_learning': '主动推理',
        'complementary_learning': '互补学习',
        'schema_learning': '图式学习',
        'metacognitive_regulator': '元认知调控',
        'cross_domain_transfer': '跨域迁移',
        'hierarchical_concepts': '层次概念',
        'temporal_sequence': '时序预测',
        'attention_gate': '注意力门控',
        'language_development': '语言发展',
        'embodied_grounding': '具身接地',
        'social_feedback': '社会反馈',
        'knowledge_distillation': '知识蒸馏',
    }

    for attr, name in mechanisms.items():
        try:
            mech = getattr(learner, attr, None)
            if mech is None:
                print(f"  [X] {name}({attr}): 未初始化")
                continue

            stats = mech.get_stats() if hasattr(mech, 'get_stats') else {}
            print(f"  [OK] {name}: {stats}")
        except Exception as ex:
            print(f"  [X] {name}({attr}): 报错 - {ex}")

    return True


def diagnose_verification_self_reference(learner, samples):
    """诊断8: 验证自指问题检测"""
    print("\n" + "="*60)
    print("诊断8: 验证自指问题检测")
    print("="*60)

    text = samples[0]['text']
    title = samples[0]['title']

    # 步骤A: 在空知识图谱上学习
    kg_before = len(learner.knowledge.entities)

    # 手动执行 learn_from_text 的关键步骤
    repr_vec = learner._encode_text(text, train=True)
    entities = learner._extract_entities_from_repr(text, repr_vec)
    triples = learner._extract_relations_from_repr(text, entities, repr_vec)

    print(f"  学习前 KG实体数: {kg_before}")
    print(f"  抽取实体: {entities[:10]}")
    print(f"  抽取三元组: {len(triples)}")

    # 注入知识图谱（步骤4）
    for triple in triples:
        if len(triple) >= 3:
            try:
                conf = triple[3] if len(triple) >= 4 and isinstance(triple[3], (int, float)) else 0.5
                learner.knowledge.add_triple(triple[0], triple[1], triple[2], confidence=conf)
            except:
                pass

    kg_after = len(learner.knowledge.entities)
    print(f"  学习后 KG实体数: {kg_after}")

    # 步骤B: 验证 — 检查think()是否只是查回刚注入的内容
    if entities:
        test_entity = entities[0]
        answer = learner.think(f"什么是{test_entity}?")
        print(f"\n  验证问题: 什么是{test_entity}?")
        print(f"  回答: {answer[:200]}")

        # 检查回答中是否包含刚学到的实体
        mentioned = [e for e in entities if e in answer]
        print(f"  回答中提到的已知实体: {mentioned}")

        if len(mentioned) > 0:
            print("  [!] 验证可能自指: 回答引用了刚存入的实体")

    return True


# ============================================================
# 主流程
# ============================================================
def main():
    print("="*60)
    print("真实语料诊断 — 学习系统结构性问题检测")
    print("="*60)

    # 加载语料
    print("\n[1] 加载Wikipedia中文语料...")
    samples = load_wiki_samples(n=20)
    print(f"    加载了 {len(samples)} 条样本")
    for i, s in enumerate(samples[:5]):
        print(f"    [{i}] {s['title']} (len={len(s['text'])})")

    # 初始化学习器
    print("\n[2] 初始化学习器...")
    from src.core.learner import Learner, LearnerConfig
    cfg = LearnerConfig()
    learner = Learner(cfg)
    print(f"    Device: {learner.device}")
    print(f"    obs_dim: {cfg.obs_dim}")

    # 运行诊断
    print("\n[3] 开始逐环节诊断...")

    try:
        diagnose_encoding(learner, samples)
    except Exception as ex:
        print(f"\n  [FATAL] 编码诊断崩溃: {ex}")
        traceback.print_exc()

    try:
        diagnose_entity_extraction(learner, samples)
    except Exception as ex:
        print(f"\n  [FATAL] 实体诊断崩溃: {ex}")
        traceback.print_exc()

    try:
        diagnose_relation_extraction(learner, samples)
    except Exception as ex:
        print(f"\n  [FATAL] 关系诊断崩溃: {ex}")
        traceback.print_exc()

    try:
        diagnose_embedding_training(learner, samples)
    except Exception as ex:
        print(f"\n  [FATAL] 嵌入训练诊断崩溃: {ex}")
        traceback.print_exc()

    try:
        diagnose_knowledge_graph(learner, samples)
    except Exception as ex:
        print(f"\n  [FATAL] 知识图谱诊断崩溃: {ex}")
        traceback.print_exc()

    try:
        diagnose_consolidation(learner)
    except Exception as ex:
        print(f"\n  [FATAL] 整合诊断崩溃: {ex}")
        traceback.print_exc()

    try:
        diagnose_verification_self_reference(learner, samples)
    except Exception as ex:
        print(f"\n  [FATAL] 验证诊断崩溃: {ex}")
        traceback.print_exc()

    try:
        diagnose_new_mechanisms(learner, samples)
    except Exception as ex:
        print(f"\n  [FATAL] 新机制诊断崩溃: {ex}")
        traceback.print_exc()

    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == '__main__':
    main()
