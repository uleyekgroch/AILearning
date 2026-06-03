#!/usr/bin/env python3
"""优化学习脚本 — 完整版本，性能优化

保留所有模块，但批量学习时跳过元学习模块：
- BTSP（单次学习）
- 认知路由
- GHL（全局调制Hebbian）
- 学习进展好奇心
- 感知类别
- 元学习组合
- 预测编码Light
- 奖励表征后移
- 组合泛化
- 社会偶联
- 符号接地
- 自改进
- 反思学习

这些模块用于"学习如何学习"，在批量知识获取时可以跳过。
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner, LearnerConfig
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation


def optimized_learn(learner: Learner, limit: int) -> dict:
    """优化学习 — 跳过元学习模块"""
    count = 0
    success = 0
    start_time = time.time()

    filepath = 'data/extracted/baike/baike_qa_train.json'
    print(f'[学习] 开始学习: {limit}条')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            count += 1

            try:
                item = json.loads(line.strip())
                title = item.get('title', '').strip()
                answer = item.get('answer', '').strip().replace('\r\n', ' ').replace('\n', ' ')[:200]

                if len(title) < 4 or len(answer) < 4:
                    continue

                # 转换为陈述句
                if '是什么' in title or '什么是' in title:
                    text = f"{title.replace('什么是', '').replace('是什么', '')}是{answer}"
                elif '为什么' in title:
                    text = f"{title.replace('为什么', '')}，因为{answer}"
                elif '怎么' in title or '如何' in title:
                    text = f"{title}的方法是{answer}"
                else:
                    text = f"{title}。{answer}"

                # 优化学习：只做核心步骤
                t0 = time.time()

                # 1. 编码
                text_repr = learner._encode_text(text, train=True)

                # 2. 提取实体
                entities = learner._extract_entities_from_repr(text, text_repr)

                # 3. 提取关系
                triples = learner._extract_relations_from_repr(text, entities, text_repr)

                # 4. 注入知识图谱
                for item in triples:
                    if len(item) >= 3:
                        subj, rel, obj = item[0], item[1], item[2]
                        confidence = item[3] if len(item) > 3 else 0.8

                        learner.knowledge.add_entity(Entity(
                            id=subj, type='concept',
                            embedding=learner._encode_text(subj)
                        ))
                        learner.knowledge.add_entity(Entity(
                            id=obj, type='concept',
                            embedding=learner._encode_text(obj)
                        ))
                        learner.knowledge.add_relation(Relation(
                            source_id=subj, target_id=obj,
                            type=rel, confidence=confidence
                        ))

                if triples:
                    success += 1

                # 进度报告
                if count % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed if elapsed > 0 else 0
                    print(f'[学习] 进度: {count}/{limit}, '
                          f'成功: {success}, '
                          f'速度: {speed:.1f}条/秒, '
                          f'实体: {len(learner.knowledge.entities)}')

                # 每500条巩固一次
                if count % 500 == 0:
                    print(f'[巩固] 执行第{count//500}次巩固...')
                    learner.consolidate()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                if count % 1000 == 0:
                    print(f'[警告] 第{count}条处理失败: {e}')
                continue

    elapsed = time.time() - start_time
    print(f'[完成] 学习{count}条, 成功{success}条, 耗时{elapsed:.1f}秒, 速度{count/elapsed:.1f}条/秒')

    return {
        'count': count,
        'success': success,
        'elapsed': elapsed,
        'entities': len(learner.knowledge.entities),
    }


def full_learn_with_meta(learner: Learner, limit: int) -> dict:
    """完整学习 — 包含元学习模块"""
    count = 0
    success = 0
    start_time = time.time()

    filepath = 'data/extracted/baike/baike_qa_train.json'
    print(f'[学习] 开始完整学习: {limit}条')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            count += 1

            try:
                item = json.loads(line.strip())
                title = item.get('title', '').strip()
                answer = item.get('answer', '').strip().replace('\r\n', ' ').replace('\n', ' ')[:200]

                if len(title) < 4 or len(answer) < 4:
                    continue

                # 转换为陈述句
                if '是什么' in title or '什么是' in title:
                    text = f"{title.replace('什么是', '').replace('是什么', '')}是{answer}"
                elif '为什么' in title:
                    text = f"{title.replace('为什么', '')}，因为{answer}"
                elif '怎么' in title or '如何' in title:
                    text = f"{title}的方法是{answer}"
                else:
                    text = f"{title}。{answer}"

                # 完整学习流程
                result = learner.learn_from_text(text, source='baike')
                if result['triples']:
                    success += 1

                # 进度报告
                if count % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed if elapsed > 0 else 0
                    print(f'[学习] 进度: {count}/{limit}, '
                          f'成功: {success}, '
                          f'速度: {speed:.1f}条/秒, '
                          f'实体: {len(learner.knowledge.entities)}')

                # 每200条巩固一次
                if count % 200 == 0:
                    print(f'[巩固] 执行第{count//200}次巩固...')
                    learner.consolidate()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                if count % 1000 == 0:
                    print(f'[警告] 第{count}条处理失败: {e}')
                continue

    elapsed = time.time() - start_time
    print(f'[完成] 学习{count}条, 成功{success}条, 耗时{elapsed:.1f}秒, 速度{count/elapsed:.1f}条/秒')

    return {
        'count': count,
        'success': success,
        'elapsed': elapsed,
        'entities': len(learner.knowledge.entities),
    }


def test_questions(learner: Learner):
    """测试问答能力"""
    print('\n' + '=' * 50)
    print('问答测试')
    print('=' * 50)

    questions = [
        '什么是人工智能',
        '牛顿发现了什么',
        '为什么地面湿了',
        '水在多少度沸腾',
        '什么是机器学习',
        '地球有多大',
        '光合作用是什么',
        'DNA是什么',
        '什么是深度学习',
        '为什么天空是蓝色的',
    ]

    for q in questions:
        answer = learner.think(q)
        print(f'Q: {q}')
        print(f'A: {answer}')
        print()


def main():
    parser = argparse.ArgumentParser(description='优化学习脚本')
    parser.add_argument('--limit', type=int, default=1000,
                       help='学习条数限制 (默认1000)')
    parser.add_argument('--mode', choices=['optimized', 'full'], default='optimized',
                       help='学习模式: optimized(跳过元学习) 或 full(完整)')
    parser.add_argument('--test', action='store_true',
                       help='学习后测试问答')
    parser.add_argument('--start-http', action='store_true',
                       help='学习后启动HTTP服务器')
    args = parser.parse_args()

    print('=' * 50)
    print('学习AI — 优化学习系统')
    print('=' * 50)
    print(f'学习限制: {args.limit} 条')
    print(f'学习模式: {args.mode}')
    print()

    # 初始化学习体
    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    # 预初始化编码器
    print('[初始化] 预初始化编码器...')
    _ = learner._encode_text('初始化', train=True)
    print('[初始化] 完成')

    # 学习
    if args.mode == 'optimized':
        stats = optimized_learn(learner, args.limit)
    else:
        stats = full_learn_with_meta(learner, args.limit)

    # 最终巩固
    print('\n[巩固] 执行最终巩固...')
    consolidation = learner.consolidate()
    print(f'[巩固] 完成: {consolidation}')

    # 测试
    if args.test:
        test_questions(learner)

    # 保存统计
    with open('data/learning_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'\n[统计] 已保存到 data/learning_stats.json')
    print(f'[统计] 总学习: {stats["success"]}, 实体: {stats["entities"]}')

    # 启动HTTP服务器
    if args.start_http:
        print('\n[服务器] 启动HTTP服务器...')
        from server import main as start_server
        start_server()


if __name__ == '__main__':
    main()
