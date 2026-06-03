"""学习全部数据集 — 流式处理 1448 万条数据

数据集：
1. 百科问答 (baike2018qa) — 147万条
2. 新闻语料 (news2016zh) — 250万条
3. 翻译语料 (translation2019zh) — 520万条
4. 社区问答 (webtext2019zh) — 425万条
5. 维基百科 (wiki_zh_2019) — 104万条

运行方式：
    python training/learn_all_datasets.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.text_understanding import TextUnderstandingSystem


def stream_jsonl(filepath, max_lines=None):
    """流式读取 JSONL 文件"""
    count = 0
    try:
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
                except:
                    continue
    except GeneratorExit:
        pass


def learn_baike(system, data_dir, max_lines=None):
    """学习百科问答"""
    print("\n[1] 学习百科问答...")
    count = 0
    for split in ['baike_qa_valid.json', 'baike_qa_train.json']:
        filepath = os.path.join(data_dir, 'baike', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines):
            title = data.get('title', '')
            desc = data.get('desc', '')
            answer = data.get('answer', '')
            category = data.get('category', '')

            # 组合成文章
            text = f"{title}\n{desc}\n{answer}"
            if len(text) > 10:
                system.learn_from_article(f"问答:{category}", text)
                count += 1

            if count % 10000 == 0:
                print(f"  已学习: {count} 条, 三元组: {len(system.knowledge.triples)}")

    print(f"  百科问答完成: {count} 条")
    return count


def learn_news(system, data_dir, max_lines=None):
    """学习新闻语料"""
    print("\n[2] 学习新闻语料...")
    count = 0
    for split in ['news2016zh_valid.json', 'news2016zh_train.json']:
        filepath = os.path.join(data_dir, 'news', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines):
            title = data.get('title', '')
            content = data.get('content', '')
            keywords = data.get('keywords', '')

            text = f"{title}\n{keywords}\n{content}"
            if len(text) > 10:
                system.learn_from_article(f"新闻:{title[:20]}", text)
                count += 1

            if count % 10000 == 0:
                print(f"  已学习: {count} 条, 三元组: {len(system.knowledge.triples)}")

    print(f"  新闻语料完成: {count} 条")
    return count


def learn_translation(system, data_dir, max_lines=None):
    """学习翻译语料"""
    print("\n[3] 学习翻译语料...")
    count = 0
    for split in ['translation2019zh_valid.json', 'translation2019zh_train.json']:
        filepath = os.path.join(data_dir, 'translation', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines):
            chinese = data.get('chinese', '')
            english = data.get('english', '')

            if chinese and english:
                system.learn_from_article(f"翻译:{chinese[:20]}", chinese)
                system.learn_from_article(f"翻译:{english[:20]}", english)
                count += 1

            if count % 10000 == 0:
                print(f"  已学习: {count} 条, 三元组: {len(system.knowledge.triples)}")

    print(f"  翻译语料完成: {count} 条")
    return count


def learn_webtext(system, data_dir, max_lines=None):
    """学习社区问答"""
    print("\n[4] 学习社区问答...")
    count = 0
    for split in ['web_text_zh_valid.json', 'web_text_zh_testa.json', 'web_text_zh_train.json']:
        filepath = os.path.join(data_dir, 'webtext', split)
        if not os.path.exists(filepath):
            continue
        for data in stream_jsonl(filepath, max_lines):
            title = data.get('title', '')
            desc = data.get('desc', '')
            content = data.get('content', '')
            topic = data.get('topic', '')

            text = f"{title}\n{desc}\n{content}"
            if len(text) > 10:
                system.learn_from_article(f"社区:{topic}", text)
                count += 1

            if count % 10000 == 0:
                print(f"  已学习: {count} 条, 三元组: {len(system.knowledge.triples)}")

    print(f"  社区问答完成: {count} 条")
    return count


def learn_wiki(system, data_dir, max_lines=None):
    """学习维基百科"""
    print("\n[5] 学习维基百科...")
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
                    system.learn_from_article(title, text)
                    count += 1

                if count % 10000 == 0:
                    print(f"  已学习: {count} 条, 三元组: {len(system.knowledge.triples)}")

                if max_lines and count >= max_lines:
                    break
            if max_lines and count >= max_lines:
                break
        if max_lines and count >= max_lines:
            break

    print(f"  维基百科完成: {count} 条")
    return count


def main():
    print("=" * 70)
    print("学习全部数据集 — 1448万条数据")
    print("=" * 70)

    data_dir = 'data/extracted'
    output_path = Path('data/knowledge/learned/all_datasets_knowledge.json')

    # 检查是否有已有进度
    checkpoint_path = Path('data/knowledge/learned/all_datasets_checkpoint.json')

    system = TextUnderstandingSystem()

    # 加载检查点
    start_dataset = 0
    if checkpoint_path.exists():
        print("\n加载检查点...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        system.knowledge.triples = checkpoint.get('triples', [])
        system.knowledge.entities = checkpoint.get('entities', {})
        system.knowledge.summaries = checkpoint.get('summaries', {})
        system.total_articles = checkpoint.get('total_articles', 0)
        start_dataset = checkpoint.get('last_dataset', 0)
        print(f"  已有: {system.total_articles} 文章, {len(system.knowledge.triples)} 三元组")

    # 每个数据集的最大条数（用于测试，None=全部）
    max_per_dataset = None  # 设为 None 学习全部

    start_time = time.time()

    # 按顺序学习每个数据集
    datasets = [
        (learn_wiki, 100000),      # 维基百科 10万条
        (learn_baike, None),       # 百科问答 全部
        (learn_news, None),        # 新闻语料 全部
        (learn_translation, None), # 翻译语料 全部
        (learn_webtext, None),     # 社区问答 全部
    ]

    for i, (learn_func, max_lines) in enumerate(datasets):
        if i < start_dataset:
            print(f"\n跳过数据集 {i+1} (已学习)")
            continue

        try:
            count = learn_func(system, data_dir, max_lines)
        except Exception as e:
            print(f"  错误: {e}")
            continue

        # 保存检查点
        checkpoint = {
            'triples': system.knowledge.triples,
            'entities': system.knowledge.entities,
            'summaries': system.knowledge.summaries,
            'total_articles': system.total_articles,
            'last_dataset': i + 1,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        print(f"  检查点已保存")

    elapsed = time.time() - start_time

    # 保存最终结果
    print("\n保存最终结果...")
    save_data = {
        'triples': system.knowledge.triples,
        'entities': system.knowledge.entities,
        'summaries': system.knowledge.summaries,
        'total_articles': system.total_articles,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False)

    # 统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)
    print(f"  耗时: {elapsed:.1f} 秒 ({elapsed/3600:.1f} 小时)")
    print(f"  文章数: {system.total_articles:,}")
    print(f"  三元组: {len(system.knowledge.triples):,}")
    print(f"  实体数: {len(system.knowledge.entities):,}")
    print(f"  摘要数: {len(system.knowledge.summaries):,}")

    # 测试查询
    print("\n测试查询:")
    test_questions = [
        "什么是人工智能",
        "中国在哪里",
        "牛顿发现了什么",
        "Python是什么语言",
    ]
    for q in test_questions:
        result = system.query(q)
        print(f"  问: {q}")
        print(f"    结果: {len(result['results'])} 个")
        for r in result['results'][:2]:
            if 'subject' in r:
                print(f"      → {r['subject']} {r['relation']} {r['object']}")


if __name__ == '__main__':
    main()
