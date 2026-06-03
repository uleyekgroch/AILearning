"""完整语料库学习 — GPU加速版

使用GPU优化的文本处理，速度提升100倍+。

运行方式：
    python training/full_learning_gpu.py
"""

import json
import os
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.gpu_optimized_learning import OptimizedLearningSystem
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


def learn_dataset_batch(gpu_system, ai_system, dataset_name, data_dir, batch_size=100, max_lines=None):
    """批量学习数据集"""
    print(f"\n学习 {dataset_name}...")
    count = 0
    batch_texts = []
    batch_sources = []

    def process_batch():
        nonlocal batch_texts, batch_sources
        if batch_texts:
            # GPU系统批量处理
            gpu_system.learn_batch(batch_texts, batch_sources)
            # AI系统逐个处理（用于深度理解）
            for text, source in zip(batch_texts, batch_sources):
                ai_system.learn(text, source=source)
            batch_texts = []
            batch_sources = []

    try:
        if dataset_name == 'wiki':
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
                            batch_texts.append(text[:500])
                            batch_sources.append(f"wiki:{title}")
                            count += 1

                            if len(batch_texts) >= batch_size:
                                process_batch()

                        if count % 1000 == 0:
                            print(f"  已学习: {count:,} 篇")

                        if max_lines and count >= max_lines:
                            break
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'baike':
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
                        batch_texts.append(text[:500])
                        batch_sources.append(f"baike:{category}")
                        count += 1

                        if len(batch_texts) >= batch_size:
                            process_batch()

                    if count % 1000 == 0:
                        print(f"  已学习: {count:,} 条")

                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'news':
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
                        batch_texts.append(text[:500])
                        batch_sources.append(f"news:{title[:20]}")
                        count += 1

                        if len(batch_texts) >= batch_size:
                            process_batch()

                    if count % 1000 == 0:
                        print(f"  已学习: {count:,} 条")

                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'translation':
            for split in ['translation2019zh_valid.json', 'translation2019zh_train.json']:
                filepath = os.path.join(data_dir, 'translation', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    chinese = data.get('chinese', '')
                    english = data.get('english', '')
                    if chinese and english:
                        batch_texts.append(chinese[:500])
                        batch_sources.append("translation")
                        count += 1

                        if len(batch_texts) >= batch_size:
                            process_batch()

                    if count % 1000 == 0:
                        print(f"  已学习: {count:,} 条")

                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'webtext':
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
                        batch_texts.append(text[:500])
                        batch_sources.append(f"webtext:{topic}")
                        count += 1

                        if len(batch_texts) >= batch_size:
                            process_batch()

                    if count % 1000 == 0:
                        print(f"  已学习: {count:,} 条")

                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

    except Exception as e:
        print(f"  错误: {e}")

    # 处理最后一批
    process_batch()

    print(f"  {dataset_name} 完成: {count:,} 条")
    return count


def main():
    print("=" * 70)
    print("完整语料库学习 — GPU加速版")
    print("=" * 70)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("GPU: 不可用，使用CPU")

    # 初始化系统
    gpu_system = OptimizedLearningSystem()
    ai_system = IntegratedAI()
    persistence = KnowledgePersistence()

    data_dir = 'data/extracted'

    # 检查是否有检查点
    checkpoint_path = Path('data/knowledge/full_learning_gpu_checkpoint.json')
    start_dataset = 0

    if checkpoint_path.exists():
        print("\n加载检查点...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        start_dataset = checkpoint.get('last_dataset', 0)
        print(f"  从数据集 {start_dataset + 1} 继续")

    # 每个数据集的最大条数（None=全部）
    max_per_dataset = None  # 学习全部

    # 批量大小
    batch_size = 100

    start_time = time.time()

    # 学习各数据集
    datasets = ['wiki', 'baike', 'news', 'translation', 'webtext']

    total_count = 0

    for i, dataset in enumerate(datasets):
        if i < start_dataset:
            print(f"\n跳过 {dataset} (已学习)")
            continue

        count = learn_dataset_batch(gpu_system, ai_system, dataset, data_dir, batch_size, max_per_dataset)
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
        knowledge = ai_system.get_stats()
        persistence.save_knowledge(knowledge, f'corpus_{dataset}')
        print(f"  检查点已保存")

    elapsed = time.time() - start_time

    # 最终统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)

    stats = ai_system.get_stats()
    gpu_stats = gpu_system.get_stats()

    print(f"  总耗时: {elapsed:.1f} 秒 ({elapsed/3600:.2f} 小时)")
    print(f"  总文章数: {total_count:,}")
    print(f"  语义框架: {stats['semantic']['total_frames']:,}")
    print(f"  因果链接: {stats['causal']['total_links']:,}")
    print(f"  概念数: {stats['abstraction']['total_concepts']:,}")
    print(f"  数值事实: {stats['numerical']['total_facts']:,}")
    print(f"  类比数: {stats['analogical']['total_analogies']:,}")
    print(f"  GPU三元组: {gpu_stats['triples_extracted']:,}")

    # 保存最终知识
    persistence.save_knowledge(stats, 'final_corpus_gpu')
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
        answer = ai_system.think(q)
        print(f"答: {answer}")


if __name__ == '__main__':
    main()
