#!/usr/bin/env python3
"""完整学习脚本 — 优化版本

保留所有模块，优化性能：
1. 预初始化所有模块
2. 批量学习时跳过TTT（测试时训练）
3. 批量编码器
4. 高效知识图谱操作
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner, LearnerConfig


def pre_init_modules(learner: Learner):
    """预初始化所有模块"""
    print('[初始化] 预初始化所有模块...')
    _ = learner._encode_text('初始化', train=True)
    _ = learner.cognitive_router
    _ = learner.ghl_learning
    _ = learner.learning_progress
    _ = learner.perceptual_categories
    _ = learner.meta_composition
    _ = learner.predictive_coding_light
    _ = learner.reward_shift
    _ = learner.compositional_generalization
    _ = learner.social_contingency
    _ = learner.symbol_grounding
    _ = learner.self_improvement
    _ = learner.reflective_learning
    _ = learner.btsp_learning
    _ = learner.empowerment_exploration
    print('[初始化] 完成')


def learn_from_baike(learner: Learner, limit: int, skip_ttt: bool = True) -> dict:
    """从百科数据学习

    Args:
        learner: 学习体
        limit: 学习条数
        skip_ttt: 批量学习时跳过TTT以提高性能
    """
    count = 0
    success = 0
    start_time = time.time()

    # 批量学习时临时禁用TTT
    if skip_ttt:
        original_ttt = getattr(learner, '_test_time_trainer', None)
        learner._skip_ttt = True

    filepath = 'data/extracted/baike/baike_qa_train.json'
    print(f'[学习] 开始学习百科数据: {limit}条')

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

    # 恢复TTT
    if skip_ttt:
        learner._skip_ttt = False

    elapsed = time.time() - start_time
    print(f'[完成] 学习{count}条, 成功{success}条, 耗时{elapsed:.1f}秒, 速度{count/elapsed:.1f}条/秒')

    return {
        'count': count,
        'success': success,
        'elapsed': elapsed,
        'entities': len(learner.knowledge.entities),
    }


def learn_from_wiki(learner: Learner, limit: int) -> dict:
    """从维基百科数据学习"""
    count = 0
    success = 0
    start_time = time.time()

    wiki_dir = 'data/extracted/wiki/wiki_zh/AA'
    print(f'[学习] 开始学习维基数据: {limit}条')

    for filename in sorted(os.listdir(wiki_dir)):
        if count >= limit:
            break

        filepath = os.path.join(wiki_dir, filename)
        if not os.path.isfile(filepath):
            continue

        print(f'[学习] 处理文件: {filename}')

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= limit:
                    break
                count += 1

                try:
                    item = json.loads(line.strip())
                    text = item.get('text', '').strip()

                    if text and len(text) > 10:
                        text = text[:300]
                        result = learner.learn_from_text(text, source='wiki')
                        if result['triples']:
                            success += 1

                        if count % 100 == 0:
                            elapsed = time.time() - start_time
                            speed = count / elapsed if elapsed > 0 else 0
                            print(f'[学习] 维基进度: {count}/{limit}, 成功: {success}, 速度: {speed:.1f}条/秒')

                except Exception:
                    continue

    elapsed = time.time() - start_time
    print(f'[完成] 维基学习{count}条, 成功{success}条, 耗时{elapsed:.1f}秒')

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
    parser = argparse.ArgumentParser(description='完整学习脚本')
    parser.add_argument('--limit', type=int, default=500,
                       help='学习条数限制 (默认500)')
    parser.add_argument('--wiki', action='store_true',
                       help='同时学习维基百科')
    parser.add_argument('--test', action='store_true',
                       help='学习后测试问答')
    parser.add_argument('--start-http', action='store_true',
                       help='学习后启动HTTP服务器')
    args = parser.parse_args()

    print('=' * 50)
    print('学习AI — 完整学习系统')
    print('=' * 50)
    print(f'学习限制: {args.limit} 条')
    print()

    # 初始化学习体
    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    # 预初始化模块
    pre_init_modules(learner)

    # 1. 学习百科数据
    baike_stats = learn_from_baike(learner, args.limit, skip_ttt=True)

    # 2. 学习维基数据
    if args.wiki:
        wiki_stats = learn_from_wiki(learner, args.limit // 5)
    else:
        wiki_stats = {'count': 0, 'success': 0, 'elapsed': 0}

    # 3. 最终巩固
    print('\n[巩固] 执行最终巩固...')
    consolidation = learner.consolidate()
    print(f'[巩固] 完成: {consolidation}')

    # 4. 测试
    if args.test:
        test_questions(learner)

    # 5. 保存统计
    stats = {
        'baike': baike_stats,
        'wiki': wiki_stats,
        'total_learned': learner._learning_stats['total_learned'],
        'entities': len(learner.knowledge.entities),
    }
    with open('data/learning_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'\n[统计] 已保存到 data/learning_stats.json')
    print(f'[统计] 总学习: {stats["total_learned"]}, 实体: {stats["entities"]}')

    # 6. 启动HTTP服务器
    if args.start_http:
        print('\n[服务器] 启动HTTP服务器...')
        from server import main as start_server
        start_server()


if __name__ == '__main__':
    main()
