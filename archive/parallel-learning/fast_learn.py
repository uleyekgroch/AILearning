#!/usr/bin/env python3
"""快速学习脚本 — 优化版本

跳过不必要的模块，专注于知识提取和存储。
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner, LearnerConfig


def fast_learn(limit=200):
    """快速学习"""
    print('=' * 50)
    print('快速学习模式')
    print('=' * 50)

    # 初始化
    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    count = 0
    success = 0
    start_time = time.time()

    # 学习
    filepath = 'data/extracted/baike/baike_qa_train.json'
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            count += 1

            try:
                item = json.loads(line.strip())
                title = item.get('title', '').strip()
                answer = item.get('answer', '').strip().replace('\r\n', ' ').replace('\n', ' ')[:100]

                if len(title) < 4 or len(answer) < 4:
                    continue

                # 转换为陈述句
                if '是什么' in title or '什么是' in title:
                    text = f"{title.replace('什么是', '').replace('是什么', '')}是{answer}"
                elif '为什么' in title:
                    text = f"{title.replace('为什么', '')}，因为{answer}"
                else:
                    text = f"{title}。{answer}"

                # 简化学习：只提取三元组并存储
                text_repr = learner._encode_text(text, train=True)
                entities = learner._extract_entities_from_repr(text, text_repr)
                triples = learner._extract_relations_from_repr(text, entities, text_repr)

                # 注入知识图谱
                for item in triples:
                    if len(item) >= 3:
                        subj, rel, obj = item[0], item[1], item[2]
                        from src.knowledge.entity import Entity
                        from src.knowledge.relation import Relation
                        learner.knowledge.add_entity(Entity(id=subj, type='concept', embedding=learner._encode_text(subj)))
                        learner.knowledge.add_entity(Entity(id=obj, type='concept', embedding=learner._encode_text(obj)))
                        learner.knowledge.add_relation(Relation(source_id=subj, target_id=obj, type=rel, confidence=0.8))

                if triples:
                    success += 1

                if count % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed if elapsed > 0 else 0
                    print(f'进度: {count}/{limit}, 成功: {success}, 速度: {speed:.1f}条/秒, 实体: {len(learner.knowledge.entities)}')

            except Exception as e:
                continue

    elapsed = time.time() - start_time
    print(f'\n完成: 学习{count}条, 成功{success}条, 耗时{elapsed:.1f}秒')
    print(f'实体: {len(learner.knowledge.entities)}')

    # 测试
    print('\n=== 问答测试 ===')
    questions = ['什么是人工智能', '牛顿发现了什么', '为什么地面湿了', '水在多少度沸腾']
    for q in questions:
        answer = learner.think(q)
        print(f'Q: {q}')
        print(f'A: {answer}')
        print()

    return learner


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    fast_learn(limit)
