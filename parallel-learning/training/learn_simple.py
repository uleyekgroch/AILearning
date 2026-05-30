"""简化版学习脚本 — 逐步学习所有语料

运行方式：
    python training/learn_simple.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.integrated_ai import IntegratedAI
from training.knowledge_persistence import KnowledgePersistence


def stream_jsonl(filepath, max_lines=None):
    """流式读取JSONL文件"""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield data
                count += 1
                if max_lines and count >= max_lines:
                    return
            except (json.JSONDecodeError, ValueError):
                continue


def learn_from_file(ai, filepath, source_prefix, max_lines=None):
    """从单个文件学习"""
    count = 0
    for data in stream_jsonl(filepath, max_lines):
        # 根据数据格式提取文本
        if 'title' in data and 'text' in data:
            # wiki格式
            title = data.get('title', '')
            text = data.get('text', '')
            if title and text:
                ai.learn(text, source=f"{source_prefix}:{title}")
                count += 1
        elif 'title' in data and 'answer' in data:
            # baike格式
            title = data.get('title', '')
            desc = data.get('desc', '')
            answer = data.get('answer', '')
            text = f"{title}\n{desc}\n{answer}"
            if len(text) > 10:
                ai.learn(text, source=f"{source_prefix}:{title}")
                count += 1
        elif 'title' in data and 'content' in data:
            # news格式
            title = data.get('title', '')
            content = data.get('content', '')
            keywords = data.get('keywords', '')
            text = f"{title}\n{keywords}\n{content}"
            if len(text) > 10:
                ai.learn(text, source=f"{source_prefix}:{title[:20]}")
                count += 1
        elif 'chinese' in data and 'english' in data:
            # translation格式
            chinese = data.get('chinese', '')
            if chinese:
                ai.learn(chinese, source=f"{source_prefix}")
                count += 1
        elif 'title' in data and 'desc' in data:
            # webtext格式
            title = data.get('title', '')
            desc = data.get('desc', '')
            content = data.get('content', '')
            text = f"{title}\n{desc}\n{content}"
            if len(text) > 10:
                ai.learn(text, source=f"{source_prefix}")
                count += 1

        if count % 100 == 0:
            print(f"  已学习: {count} 条")

    return count


def main():
    print("=" * 70)
    print("简化版学习脚本")
    print("=" * 70)

    # 初始化系统
    ai = IntegratedAI()
    persistence = KnowledgePersistence()

    data_dir = 'data/extracted'

    # 检查是否有检查点
    checkpoint_path = Path('data/knowledge/simple_checkpoint.json')
    start_dataset = 0

    if checkpoint_path.exists():
        print("\n加载检查点...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        start_dataset = checkpoint.get('last_dataset', 0)
        print(f"  从数据集 {start_dataset + 1} 继续")

    # 每个数据集学习的最大条数
    max_per_dataset = 1000  # 先学1000条测试

    start_time = time.time()

    # 学习各数据集
    datasets = [
        ('wiki', 'wiki/wiki_zh', 'wiki'),
        ('baike', 'baike', 'baike'),
        ('news', 'news', 'news'),
        ('translation', 'translation', 'translation'),
        ('webtext', 'webtext', 'webtext'),
    ]

    total_count = 0

    for i, (name, subdir, prefix) in enumerate(datasets):
        if i < start_dataset:
            print(f"\n跳过 {name} (已学习)")
            continue

        print(f"\n[{i+1}/5] 学习 {name}...")
        dataset_dir = os.path.join(data_dir, subdir)

        count = 0
        if name == 'wiki':
            # wiki有子目录
            wiki_dir = os.path.join(dataset_dir, 'wiki_zh')
            if os.path.exists(wiki_dir):
                for root, dirs, files in os.walk(wiki_dir):
                    for fname in files:
                        if fname.startswith('.'):
                            continue
                        filepath = os.path.join(root, fname)
                        count += learn_from_file(ai, filepath, prefix, max_per_dataset - count)
                        if count >= max_per_dataset:
                            break
                    if count >= max_per_dataset:
                        break
        else:
            # 其他数据集直接读取文件
            for fname in os.listdir(dataset_dir):
                if fname.endswith('.json'):
                    filepath = os.path.join(dataset_dir, fname)
                    count += learn_from_file(ai, filepath, prefix, max_per_dataset - count)
                    if count >= max_per_dataset:
                        break

        total_count += count
        print(f"  {name} 完成: {count} 条")

        # 保存检查点
        checkpoint = {
            'last_dataset': i + 1,
            'total_count': total_count,
            'timestamp': time.time(),
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)

    elapsed = time.time() - start_time

    # 最终统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)

    stats = ai.get_stats()
    print(f"  总耗时: {elapsed:.1f} 秒")
    print(f"  总文章数: {total_count:,}")
    print(f"  语义框架: {stats['semantic']['total_frames']:,}")
    print(f"  因果链接: {stats['causal']['total_links']:,}")
    print(f"  概念数: {stats['abstraction']['total_concepts']:,}")

    # 保存最终知识
    persistence.save_knowledge(stats, 'final_simple')
    print("\n最终知识已保存")

    # 测试查询
    print("\n" + "=" * 70)
    print("测试查询")
    print("=" * 70)

    test_questions = [
        "什么是人工智能",
        "牛顿发现了什么",
        "为什么地面湿了",
        "水在多少度沸腾",
    ]

    for q in test_questions:
        print(f"\n问: {q}")
        answer = ai.think(q)
        print(f"答: {answer}")


if __name__ == '__main__':
    main()
