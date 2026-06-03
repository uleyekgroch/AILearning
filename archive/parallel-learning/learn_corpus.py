#!/usr/bin/env python3
"""语料学习脚本 — 流式学习data/extracted下的所有语料

使用方式：
  python learn_corpus.py                    # 学习所有语料（默认100条）
  python learn_corpus.py --limit 1000       # 学习1000条
  python learn_corpus.py --all              # 学习全部（142万条，需要很长时间）
  python learn_corpus.py --start-http       # 学习后启动HTTP服务器

学习策略：
1. 从百科Q&A中提取知识三元组
2. 从新闻中提取实体关系
3. 从维基百科中提取概念层次
4. 逐步巩固，每100条执行一次
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner, LearnerConfig


def extract_knowledge_from_qa(qa_item: dict) -> str:
    """从Q&A中提取可学习的文本

    将问答对转换为陈述句，便于学习。
    """
    title = qa_item.get('title', '').strip()
    answer = qa_item.get('answer', '').strip()
    category = qa_item.get('category', '').strip()

    if not title or not answer:
        return None

    # 清理文本
    answer = answer.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    answer = answer[:200]  # 限制长度

    # 过滤太短或无意义的
    if len(title) < 4 or len(answer) < 4:
        return None

    # 转换为陈述句
    if '是什么' in title or '什么是' in title:
        # "什么是X" -> "X是..."
        text = f"{title.replace('什么是', '').replace('是什么', '')}是{answer}"
    elif '为什么' in title:
        # "为什么X" -> "X，因为..."
        text = f"{title.replace('为什么', '')}，因为{answer}"
    elif '怎么' in title or '如何' in title:
        # "怎么X" -> "X的方法是..."
        text = f"{title}的方法是{answer}"
    else:
        # 通用：标题+答案
        text = f"{title}。{answer}"

    return text


def learn_from_baike(learner: Learner, filepath: str, limit: int,
                     log_file=None) -> int:
    """从百科Q&A数据学习"""
    count = 0
    success = 0
    start_time = time.time()

    print(f'[学习] 开始学习百科数据: {filepath}')
    print(f'[学习] 限制: {limit} 条')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break

            count += 1

            try:
                item = json.loads(line.strip())
                text = extract_knowledge_from_qa(item)

                if text:
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

                        # 写入日志
                        if log_file:
                            log_file.write(f'{count},{success},{speed:.1f},'
                                          f'{len(learner.knowledge.entities)}\n')
                            log_file.flush()

                    # 每500条巩固一次
                    if count % 500 == 0:
                        print(f'[巩固] 执行第{count//500}次巩固...')
                        learner.consolidate()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                if count % 10000 == 0:
                    print(f'[警告] 第{count}条处理失败: {e}')
                continue

    elapsed = time.time() - start_time
    print(f'[完成] 学习{count}条, 成功{success}条, 耗时{elapsed:.1f}秒')

    return success


def learn_from_wiki(learner: Learner, wiki_dir: str, limit: int) -> int:
    """从维基百科数据学习"""
    count = 0
    success = 0

    print(f'[学习] 开始学习维基数据: {wiki_dir}')

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
                        # 取前200字作为学习内容
                        text = text[:200]
                        result = learner.learn_from_text(text, source='wiki')
                        if result['triples']:
                            success += 1

                        if count % 100 == 0:
                            print(f'[学习] 维基进度: {count}/{limit}, 成功: {success}')

                except Exception:
                    continue

    print(f'[完成] 维基学习{count}条, 成功{success}条')
    return success


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
    ]

    for q in questions:
        answer = learner.think(q)
        print(f'Q: {q}')
        print(f'A: {answer}')
        print()


def main():
    parser = argparse.ArgumentParser(description='语料学习脚本')
    parser.add_argument('--limit', type=int, default=100,
                       help='学习条数限制 (默认100)')
    parser.add_argument('--all', action='store_true',
                       help='学习全部语料')
    parser.add_argument('--wiki', action='store_true',
                       help='同时学习维基百科')
    parser.add_argument('--start-http', action='store_true',
                       help='学习后启动HTTP服务器')
    parser.add_argument('--test', action='store_true',
                       help='学习后测试问答')
    args = parser.parse_args()

    limit = 1425170 if args.all else args.limit

    print('=' * 50)
    print('学习AI — 语料学习系统')
    print('=' * 50)
    print(f'学习限制: {limit} 条')
    print()

    # 初始化学习体
    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    # 打开日志
    log_path = 'data/learning_log.csv'
    os.makedirs('data', exist_ok=True)
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write('count,success,speed,entities\n')

    try:
        # 1. 学习百科数据
        baike_path = 'data/extracted/baike/baike_qa_train.json'
        if os.path.exists(baike_path):
            learn_from_baike(learner, baike_path, limit, log_file)
        else:
            print(f'[跳过] 百科数据不存在: {baike_path}')

        # 2. 学习维基数据
        if args.wiki:
            wiki_dir = 'data/extracted/wiki/wiki_zh/AA'
            if os.path.exists(wiki_dir):
                learn_from_wiki(learner, wiki_dir, limit // 10)
            else:
                print(f'[跳过] 维基数据不存在: {wiki_dir}')

        # 3. 最终巩固
        print('\n[巩固] 执行最终巩固...')
        consolidation = learner.consolidate()
        print(f'[巩固] 完成: {consolidation}')

        # 4. 测试
        if args.test:
            test_questions(learner)

        # 5. 保存统计
        stats = {
            'total_learned': learner._learning_stats['total_learned'],
            'entities': len(learner.knowledge.entities),
            'limit': limit,
        }
        with open('data/learning_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f'\n[统计] 已保存到 data/learning_stats.json')
        print(f'[统计] 总学习: {stats["total_learned"]}, 实体: {stats["entities"]}')

    finally:
        log_file.close()

    # 6. 启动HTTP服务器
    if args.start_http:
        print('\n[服务器] 启动HTTP服务器...')
        from server import start_server
        start_server(learner)


if __name__ == '__main__':
    main()
