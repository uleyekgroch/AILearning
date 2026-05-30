"""完整 Wikipedia 语料学习

运行方式：
    python training/learn_wiki_full.py

功能：
1. 学习 Wikipedia 中文语料
2. 提取知识三元组
3. 构建知识图谱
4. 保存学习结果
5. 启动问答服务
"""

import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.text_understanding import TextUnderstandingSystem


def main():
    print("=" * 70)
    print("完整 Wikipedia 语料学习")
    print("=" * 70)

    # 初始化
    system = TextUnderstandingSystem()

    # 检查是否已有学习结果
    output_path = Path('data/knowledge/learned/wiki_knowledge.json')
    if output_path.exists():
        print("\n[1] 加载已有学习结果...")
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        system.knowledge.triples = data.get('triples', [])
        system.knowledge.entities = data.get('entities', {})
        system.knowledge.summaries = data.get('summaries', {})
        system.total_articles = data.get('total_articles', 0)
        print(f"  已加载: {system.total_articles} 文章, {len(system.knowledge.triples)} 三元组")
    else:
        # 学习语料
        print("\n[1] 学习 Wikipedia 语料...")
        start_time = time.time()

        zip_path = 'data/wiki_zh_2019.zip'
        if not os.path.exists(zip_path):
            print(f"  错误: 找不到 {zip_path}")
            return

        with zipfile.ZipFile(zip_path, 'r') as z:
            files = [f for f in z.namelist() if not f.endswith('/')]
            print(f"  文件数: {len(files)}")

            article_count = 0
            for file_path in files:
                try:
                    with z.open(file_path) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        for line in content.strip().split('\n'):
                            if not line.strip():
                                continue
                            try:
                                article = json.loads(line)
                                title = article.get('title', '')
                                text = article.get('text', '')
                                if title and text:
                                    system.learn_from_article(title, text)
                                    article_count += 1

                                    if article_count % 1000 == 0:
                                        print(f"    已学习: {article_count} 文章, "
                                              f"{len(system.knowledge.triples)} 三元组, "
                                              f"{len(system.knowledge.entities)} 实体")
                            except:
                                continue
                except:
                    continue

        elapsed = time.time() - start_time
        print(f"\n  学习完成!")
        print(f"  耗时: {elapsed:.1f} 秒")
        print(f"  文章数: {system.total_articles}")
        print(f"  三元组: {len(system.knowledge.triples)}")
        print(f"  实体数: {len(system.knowledge.entities)}")

        # 保存
        print("\n[2] 保存学习结果...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'triples': system.knowledge.triples,
            'entities': system.knowledge.entities,
            'summaries': system.knowledge.summaries,
            'total_articles': system.total_articles,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"  保存到: {output_path}")

    # 测试查询
    print("\n[3] 测试查询...")
    test_questions = [
        '什么是重力',
        'Python是什么',
        '太阳是什么',
        '中国在哪里',
        '牛顿发现了什么',
    ]

    for q in test_questions:
        result = system.query(q)
        print(f"\n  问: {q}")
        print(f"    结果: {len(result['results'])} 个")
        for r in result['results'][:2]:
            if 'subject' in r:
                print(f"      → {r['subject']} {r['relation']} {r['object']}")

    # 统计
    stats = system.get_stats()
    print(f"\n[4] 最终统计:")
    print(f"  文章数: {stats['total_articles']}")
    print(f"  三元组: {stats['total_triples']}")
    print(f"  实体数: {stats['total_entities']}")
    print(f"  摘要数: {stats['total_summaries']}")

    # 显示高频实体
    print(f"\n  高频实体:")
    for name, info in stats.get('top_entities', [])[:10]:
        print(f"    {name}: 出现 {info.get('count', 0)} 次")

    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
