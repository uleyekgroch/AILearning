#!/usr/bin/env python3
"""自主学习系统 — 主动推理驱动的课程选择

与optimized_learn.py的关键区别:
1. 不是顺序读语料，而是由系统主动选择最有价值的样本
2. 使用主动推理计算信息增益，优先学最不确定的内容
3. 睡眠回放防止灾难性遗忘（学习新知识时回放旧知识）
4. 互补学习系统自动将具体实例抽象为语义知识

三个学习阶段:
  Phase 1 — 快速采样: 从语料池中随机采样，建立初始知识
  Phase 2 — 主动选择: 用信息增益驱动，选择信息量最大的样本
  Phase 3 — 深度巩固: 睡眠回放+互补学习巩固，发现隐含联系
"""

import sys
import os
import json
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner, LearnerConfig
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation


def text_to_statement(title: str, answer: str) -> str:
    """将问答对转换为陈述句"""
    if '是什么' in title or '什么是' in title:
        return f"{title.replace('什么是', '').replace('是什么', '')}是{answer}"
    elif '为什么' in title:
        return f"{title.replace('为什么', '')}，因为{answer}"
    elif '怎么' in title or '如何' in title:
        return f"{title}的方法是{answer}"
    else:
        return f"{title}。{answer}"


def load_corpus_pool(filepath: str, max_items: int = 50000) -> list:
    """加载语料池（供系统选择）"""
    pool = []
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= max_items:
                break
            count += 1
            try:
                item = json.loads(line.strip())
                title = item.get('title', '').strip()
                answer = item.get('answer', '').strip().replace('\r\n', ' ').replace('\n', ' ')[:200]
                if len(title) < 4 or len(answer) < 4:
                    continue
                text = text_to_statement(title, answer)
                pool.append(text)
            except json.JSONDecodeError:
                continue
    return pool


def core_learn_one(learner: Learner, text: str) -> bool:
    """核心学习一条文本（性能优化版）

    只做核心步骤：编码→提取→注入图谱。
    新模块（树突/睡眠/互补）由外部批量调用。
    """
    try:
        # 1. 编码
        text_repr = learner._encode_text(text, train=True)

        # 2. 提取实体
        entities = learner._extract_entities_from_repr(text, text_repr)

        # 3. 提取关系
        triples = learner._extract_relations_from_repr(text, entities, text_repr)

        # 4. 注入知识图谱（用text_repr作为实体嵌入，避免重复编码）
        for item in triples:
            if len(item) >= 3:
                subj, rel, obj = item[0], item[1], item[2]
                confidence = item[3] if len(item) > 3 else 0.8

                learner.knowledge.add_entity(Entity(
                    id=subj, type='concept',
                    embedding=text_repr.detach()
                ))
                learner.knowledge.add_entity(Entity(
                    id=obj, type='concept',
                    embedding=text_repr.detach()
                ))
                learner.knowledge.add_relation(Relation(
                    source_id=subj, target_id=obj,
                    type=rel, confidence=confidence
                ))

        return len(triples) > 0
    except Exception:
        return False


def batch_new_modules(learner: Learner, batch: list):
    """批量更新新模块（每10条调用一次，而非每条都调用）

    将积累的文本批量送入树突/睡眠/互补学习系统。
    """
    for text, text_repr, entities in batch:
        # 树突计算：上下文关联（只处理前2个实体）
        for entity in entities[:2]:
            learner.dendritic_system.compute_context_representation(
                entity, text_repr, text_repr
            )

        # 睡眠回放：记录情节
        learner.sleep_replay.record_episode(
            text=text,
            embedding=text_repr.detach(),
            importance=0.5 if entities else 0.2,
        )

        # 互补学习：快速存储到海马体
        learner.complementary_learning.store_episode(
            content=text,
            embedding=text_repr.detach(),
            entities=entities,
            context='autonomous',
        )


def phase1_rapid_sampling(learner: Learner, pool: list, count: int) -> dict:
    """Phase 1: 快速采样 — 建立初始知识库

    从语料池中随机采样，快速建立基础实体和关系。
    不使用主动推理（因为还没有足够的不确定性估计）。
    """
    print(f'\n{"="*60}')
    print(f'Phase 1: 快速采样 ({count}条)')
    print(f'{"="*60}')
    sys.stdout.flush()

    start_time = time.time()
    success = 0
    batch = []  # 批量积累

    # 随机采样
    samples = random.sample(pool, min(count, len(pool)))

    for i, text in enumerate(samples):
        if core_learn_one(learner, text):
            success += 1

        # 积累批量数据
        if hasattr(learner, '_learnable_encoder'):
            text_repr = learner._encode_text(text)
        else:
            text_repr = learner._encode_text(text, train=True)
        entities = []
        # 快速提取实体名（不重新编码）
        import re
        entities = re.findall(r'[一-鿿]{2,6}', text[:50])[:3]
        batch.append((text, text_repr, entities))

        # 每10条批量更新新模块
        if len(batch) >= 10:
            batch_new_modules(learner, batch)
            batch = []

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f'  [{i+1}/{count}] 成功:{success} 速度:{speed:.1f}条/秒 '
                  f'实体:{len(learner.knowledge.entities)}')
            sys.stdout.flush()

    # 处理剩余批量
    if batch:
        batch_new_modules(learner, batch)

    elapsed = time.time() - start_time
    print(f'  Phase 1 完成: {success}/{count} 成功, {elapsed:.1f}秒')
    sys.stdout.flush()

    return {'count': count, 'success': success, 'elapsed': elapsed}


def phase2_active_selection(learner: Learner, pool: list, count: int) -> dict:
    """Phase 2: 主动选择 — 信息增益驱动的课程选择

    核心创新：系统自主决定学什么。
    不是顺序读语料，而是：
    1. 从语料池中采样候选
    2. 计算每个候选的信息增益
    3. 选择信息增益最高的样本学习
    """
    print(f'\n{"="*60}')
    print(f'Phase 2: 主动推理驱动的课程选择 ({count}条)')
    print(f'{"="*60}')
    sys.stdout.flush()

    start_time = time.time()
    success = 0
    total_info_gain = 0.0
    batch = []  # 批量积累

    # 将pool分成批次
    batch_size = min(500, len(pool))
    remaining_pool = list(pool)  # 复制

    learned = 0
    while learned < count and remaining_pool:
        # 从池中采样候选
        sample_size = min(batch_size, len(remaining_pool))
        candidates = random.sample(remaining_pool, sample_size)

        # 用主动推理选择最有价值的样本
        selected = learner.active_inference_learning.select_curriculum(
            all_texts=candidates,
            encoder_fn=learner._encode_text,
            batch_size=min(50, count - learned),
        )

        # 学习选中的样本
        for text, info_gain in selected:
            if learned >= count:
                break

            if core_learn_one(learner, text):
                success += 1

            total_info_gain += info_gain
            learned += 1

            # 积累批量
            text_repr = learner._encode_text(text)
            import re
            entities = re.findall(r'[一-鿿]{2,6}', text[:50])[:3]
            batch.append((text, text_repr, entities))

            # 每10条批量更新新模块
            if len(batch) >= 10:
                batch_new_modules(learner, batch)
                batch = []

            # 从池中移除已学习的
            if text in remaining_pool:
                remaining_pool.remove(text)

        if learned % 50 == 0:
            elapsed = time.time() - start_time
            speed = learned / elapsed if elapsed > 0 else 0
            avg_gain = total_info_gain / max(1, learned)
            print(f'  [{learned}/{count}] 成功:{success} 速度:{speed:.1f}条/秒 '
                  f'平均信息增益:{avg_gain:.4f} '
                  f'实体:{len(learner.knowledge.entities)}')
            sys.stdout.flush()

    # 处理剩余批量
    if batch:
        batch_new_modules(learner, batch)

    elapsed = time.time() - start_time
    avg_gain = total_info_gain / max(1, learned)
    print(f'  Phase 2 完成: {success}/{learned} 成功, {elapsed:.1f}秒')
    print(f'  平均信息增益: {avg_gain:.4f}')
    sys.stdout.flush()

    return {
        'count': learned,
        'success': success,
        'elapsed': elapsed,
        'avg_info_gain': avg_gain,
    }


def phase3_deep_consolidation(learner: Learner) -> dict:
    """Phase 3: 深度巩固 — 睡眠回放+互补学习

    模拟一个完整的"睡眠"周期：
    1. SWS慢波回放: 按重要性转移记忆到皮层
    2. REM回放: 创造性发现隐含联系
    3. Spindles: 选择性巩固重要记忆
    4. 互补学习巩固: 海马体→皮层转移
    """
    print(f'\n{"="*60}')
    print(f'Phase 3: 深度巩固 (睡眠周期)')
    print(f'{"="*60}')

    start_time = time.time()

    # 1. 系统内部巩固
    print('  [1/4] 系统巩固...')
    consolidation = learner.consolidate()
    print(f'    巩固: {consolidation}')

    # 2. 睡眠回放（完整周期）
    print('  [2/4] 睡眠回放...')
    sleep_result = learner.sleep_replay.sleep_cycle(
        knowledge_graph=learner.knowledge,
        encoder_fn=learner._encode_text,
    )
    print(f'    SWS: {sleep_result.get("sws", {})}')
    print(f'    REM: {sleep_result.get("rem", {})}')
    print(f'    Spindles: {sleep_result.get("spindles", {})}')

    # 3. 互补学习大规模巩固
    print('  [3/4] 互补学习巩固...')
    cls_result = learner.complementary_learning.replay_consolidate(batch_size=200)
    print(f'    巩固: {cls_result}')

    # 4. 皮层泛化：发现跨概念联系
    print('  [4/4] 皮层泛化...')
    generalizations = 0
    neocortex = learner.complementary_learning.neocortex
    concepts = list(neocortex.concept_index.keys())[:50]
    for concept in concepts:
        related = neocortex.find_related(concept, top_k=3)
        for related_concept, similarity in related:
            if similarity > 0.5:
                # 在知识图谱中建立关联
                try:
                    learner.knowledge.add_relation(Relation(
                        source_id=concept,
                        target_id=related_concept,
                        type='语义关联',
                        confidence=similarity * 0.6,
                    ))
                    generalizations += 1
                except Exception:
                    pass

    elapsed = time.time() - start_time
    print(f'  Phase 3 完成: {elapsed:.1f}秒')
    print(f'    泛化发现: {generalizations}个新联系')

    # 统计新系统状态
    stats = {
        'elapsed': elapsed,
        'generalizations': generalizations,
        'sleep': learner.sleep_replay.get_stats(),
        'cls': learner.complementary_learning.get_stats(),
        'dendritic': learner.dendritic_system.get_stats(),
        'active_inference': learner.active_inference_learning.get_stats(),
    }

    print(f'\n  === 新系统统计 ===')
    for name, s in stats.items():
        if isinstance(s, dict):
            print(f'    {name}: {s}')

    return stats


def test_questions(learner: Learner):
    """测试问答能力"""
    print(f'\n{"="*60}')
    print('问答测试')
    print(f'{"="*60}')

    questions = [
        '什么是人工智能',
        '牛顿发现了什么',
        '水在多少度沸腾',
        '什么是机器学习',
        '地球有多大',
        '光合作用是什么',
        '什么是深度学习',
        '为什么天空是蓝色的',
        '什么是DNA',
        '勾股定理是什么',
    ]

    correct = 0
    for q in questions:
        answer = learner.think(q)
        has_answer = '没有' not in answer and '抱歉' not in answer
        if has_answer:
            correct += 1
        status = 'OK' if has_answer else 'X '
        print(f'  [{status}] Q: {q}')
        print(f'      A: {answer[:120]}')
        print()

    print(f'  回答率: {correct}/{len(questions)} ({correct/len(questions)*100:.0f}%)')


def main():
    parser = argparse.ArgumentParser(description='自主学习系统 — 主动推理驱动')
    parser.add_argument('--phase1', type=int, default=500,
                       help='Phase 1 快速采样条数 (默认500)')
    parser.add_argument('--phase2', type=int, default=500,
                       help='Phase 2 主动选择条数 (默认500)')
    parser.add_argument('--pool-size', type=int, default=20000,
                       help='语料池大小 (默认20000)')
    parser.add_argument('--skip-phase1', action='store_true',
                       help='跳过Phase 1 (已有初始知识)')
    parser.add_argument('--skip-phase3', action='store_true',
                       help='跳过Phase 3 (不巩固)')
    parser.add_argument('--test', action='store_true',
                       help='学习后测试问答')
    parser.add_argument('--http', action='store_true',
                       help='学习后启动HTTP服务器')
    parser.add_argument('--port', type=int, default=8080,
                       help='HTTP端口 (默认8080)')
    args = parser.parse_args()

    print('=' * 60)
    print('自主学习系统 — 主动推理驱动的课程选择')
    print('=' * 60)
    print(f'Phase 1: {args.phase1}条 (快速采样)')
    print(f'Phase 2: {args.phase2}条 (主动选择)')
    print(f'语料池: {args.pool_size}条')
    print()

    # 初始化
    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    print('[初始化] 预初始化编码器...')
    _ = learner._encode_text('初始化', train=True)
    print('[初始化] 完成')

    # 加载语料池
    print(f'[加载] 从语料池加载 {args.pool_size} 条...')
    filepath = 'data/extracted/baike/baike_qa_train.json'
    pool = load_corpus_pool(filepath, args.pool_size)
    print(f'[加载] 完成: {len(pool)}条可用')

    # Phase 1
    if not args.skip_phase1:
        p1_stats = phase1_rapid_sampling(learner, pool, args.phase1)

    # Phase 2
    p2_stats = phase2_active_selection(learner, pool, args.phase2)

    # Phase 3
    if not args.skip_phase3:
        p3_stats = phase3_deep_consolidation(learner)

    # 测试
    if args.test:
        test_questions(learner)

    # 总体统计
    print(f'\n{"="*60}')
    print('学习完成 — 总体统计')
    print(f'{"="*60}')
    print(f'  实体: {len(learner.knowledge.entities)}')
    print(f'  关系: {len(learner.knowledge.relations) if hasattr(learner.knowledge, "relations") else "?"}')
    print(f'  新系统:')
    print(f'    树突神经元: {learner.dendritic_system.get_stats()["total_neurons"]}')
    print(f'    睡眠回放: {learner.sleep_replay.get_stats()["episodic_memory"]}情节 + '
          f'{learner.sleep_replay.get_stats()["semantic_memory"]}语义')
    print(f'    互补学习: {learner.complementary_learning.get_stats()["hippocampus_active"]}海马体 + '
          f'{learner.complementary_learning.get_stats()["neocortex_patterns"]}皮层')

    # HTTP服务器
    if args.http:
        print(f'\n[服务器] 启动在 http://localhost:{args.port}')
        from start_server import main as start_server
        start_server()


if __name__ == '__main__':
    main()
