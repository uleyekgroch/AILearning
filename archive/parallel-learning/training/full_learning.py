"""完整语料库学习 — 使用最新系统

学习所有5个数据集：
1. 维基百科 (wiki) — 1.3GB
2. 百科问答 (baike) — 1.5GB
3. 新闻语料 (news) — 8.6GB
4. 翻译语料 (translation) — 1.3GB
5. 社区问答 (webtext) — 3.8GB

运行方式：
    python training/full_learning.py
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


def learn_wiki(ai, data_dir, max_lines=None):
    """学习维基百科"""
    print("\n[1/5] 学习维基百科...")
    count = 0
    wiki_dir = os.path.join(data_dir, 'wiki', 'wiki_zh')

    if not os.path.exists(wiki_dir):
        print(f"  目录不存在: {wiki_dir}")
        return 0

    for root, dirs, files in os.walk(wiki_dir):
        for fname in files:
            if fname.startswith('.'):
                continue
            filepath = os.path.join(root, fname)
            for data in stream_jsonl(filepath, max_lines=max_lines - count if max_lines else None):
                title = data.get('title', '')
                text = data.get('text', '')
                if title and text:
                    ai.learn(text, source=f"wiki:{title}")
                    count += 1
                if count % 5000 == 0:
                    print(f"  已学习: {count:,} 篇")
                if max_lines and count >= max_lines:
                    break
            if max_lines and count >= max_lines:
                break
        if max_lines and count >= max_lines:
            break

    print(f"  维基百科完成: {count:,} 篇")
    return count


def learn_baike(ai, data_dir, max_lines=None):
    """学习百科问答"""
    print("\n[2/5] 学习百科问答...")
    count = 0

    for split in ['baike_qa_valid.json', 'baike_qa_train.json']:
        filepath = os.path.join(data_dir, 'baike', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
            title = data.get('title', '')
            desc = data.get('desc', '')
            answer = data.get('answer', '')
            category = data.get('category', '')
            text = f"{title}\n{desc}\n{answer}"
            if len(text) > 10:
                ai.learn(text, source=f"baike:{category}")
                count += 1
            if count % 5000 == 0:
                print(f"  已学习: {count:,} 条")
            if max_lines and count >= max_lines:
                break
        if max_lines and count >= max_lines:
            break

    print(f"  百科问答完成: {count:,} 条")
    return count


def learn_news(ai, data_dir, max_lines=None):
    """学习新闻语料"""
    print("\n[3/5] 学习新闻语料...")
    count = 0

    for split in ['news2016zh_valid.json', 'news2016zh_train.json']:
        filepath = os.path.join(data_dir, 'news', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
            title = data.get('title', '')
            content = data.get('content', '')
            keywords = data.get('keywords', '')
            text = f"{title}\n{keywords}\n{content}"
            if len(text) > 10:
                ai.learn(text, source=f"news:{title[:20]}")
                count += 1
            if count % 5000 == 0:
                print(f"  已学习: {count:,} 条")
            if max_lines and count >= max_lines:
                break
        if max_lines and count >= max_lines:
            break

    print(f"  新闻语料完成: {count:,} 条")
    return count


def learn_translation(ai, data_dir, max_lines=None):
    """学习翻译语料"""
    print("\n[4/5] 学习翻译语料...")
    count = 0

    for split in ['translation2019zh_valid.json', 'translation2019zh_train.json']:
        filepath = os.path.join(data_dir, 'translation', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
            chinese = data.get('chinese', '')
            english = data.get('english', '')
            if chinese and english:
                ai.learn(chinese, source="translation")
                count += 1
            if count % 5000 == 0:
                print(f"  已学习: {count:,} 条")
            if max_lines and count >= max_lines:
                break
        if max_lines and count >= max_lines:
            break

    print(f"  翻译语料完成: {count:,} 条")
    return count


def learn_webtext(ai, data_dir, max_lines=None):
    """学习社区问答"""
    print("\n[5/5] 学习社区问答...")
    count = 0

    for split in ['web_text_zh_valid.json', 'web_text_zh_testa.json', 'web_text_zh_train.json']:
        filepath = os.path.join(data_dir, 'webtext', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
            title = data.get('title', '')
            desc = data.get('desc', '')
            content = data.get('content', '')
            topic = data.get('topic', '')
            text = f"{title}\n{desc}\n{content}"
            if len(text) > 10:
                ai.learn(text, source=f"webtext:{topic}")
                count += 1
            if count % 5000 == 0:
                print(f"  已学习: {count:,} 条")
            if max_lines and count >= max_lines:
                break
        if max_lines and count >= max_lines:
            break

    print(f"  社区问答完成: {count:,} 条")
    return count


def main():
    print("=" * 70)
    print("完整语料库学习 — 使用最新系统")
    print("=" * 70)

    # 检查GPU
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: 不可用，使用CPU")

    # 初始化系统
    ai = IntegratedAI()
    persistence = KnowledgePersistence()

    data_dir = 'data/extracted'

    # 检查是否有检查点
    checkpoint_path = Path('data/knowledge/full_learning_checkpoint.json')
    start_dataset = 0

    if checkpoint_path.exists():
        print("\n加载检查点...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        start_dataset = checkpoint.get('last_dataset', 0)
        print(f"  从数据集 {start_dataset + 1} 继续")

    # 每个数据集的最大条数（None=全部）
    max_per_dataset = None  # 学习全部

    start_time = time.time()

    # 学习各数据集
    datasets = [
        ('wiki', learn_wiki),
        ('baike', learn_baike),
        ('news', learn_news),
        ('translation', learn_translation),
        ('webtext', learn_webtext),
    ]

    total_count = 0

    for i, (name, learn_func) in enumerate(datasets):
        if i < start_dataset:
            print(f"\n跳过 {name} (已学习)")
            continue

        count = learn_func(ai, data_dir, max_per_dataset)
        total_count += count

        # 保存检查点
        checkpoint = {
            'last_dataset': i + 1,
            'total_count': total_count,
            'timestamp': time.time(),
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)

        # 保存知识
        knowledge = ai.get_stats()
        persistence.save_knowledge(knowledge, f'corpus_{name}')
        print(f"  检查点已保存")

    elapsed = time.time() - start_time

    # 最终统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)

    stats = ai.get_stats()
    print(f"  总耗时: {elapsed:.1f} 秒 ({elapsed/3600:.2f} 小时)")
    print(f"  总文章数: {total_count:,}")
    print(f"  语义框架: {stats['semantic']['total_frames']:,}")
    print(f"  因果链接: {stats['causal']['total_links']:,}")
    print(f"  概念数: {stats['abstraction']['total_concepts']:,}")
    print(f"  数值事实: {stats['numerical']['total_facts']:,}")
    print(f"  类比数: {stats['analogical']['total_analogies']:,}")

    # 保存最终知识
    persistence.save_knowledge(stats, 'final_corpus')
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
        "水流像什么",
    ]

    for q in test_questions:
        print(f"\n问: {q}")
        answer = ai.think(q)
        print(f"答: {answer}")


if __name__ == '__main__':
    main()
