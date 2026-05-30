"""整合学习系统 — 正则提取三元组 + 向量化快速检索

核心思想：
- 用正则提取结构化知识（三元组）
- 用向量化建立快速检索索引
- 两者结合，既理解语义又支持快速查询

运行方式：
    python training/learn_with_vector.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.text_understanding import TextUnderstandingSystem
from training.vector_knowledge import VectorKnowledgeSystem


class IntegratedLearningSystem:
    """整合学习系统

    结合：
    1. TextUnderstandingSystem — 提取三元组、构建知识图谱
    2. VectorKnowledgeSystem — 向量化索引、快速检索
    """

    def __init__(self):
        self.text_system = TextUnderstandingSystem()
        self.vector_system = VectorKnowledgeSystem(
            max_features=50000,
            n_clusters=1000,
            n_components=100,
        )
        self.article_count = 0

    def learn_from_article(self, title: str, text: str):
        """从一篇文章中学习"""
        # 1. 用正则系统提取三元组
        self.text_system.learn_from_article(title, text)

        # 2. 用向量系统索引文档
        self.vector_system.add_document(title, text[:1000])

        self.article_count += 1

    def fit(self):
        """训练向量模型"""
        if self.article_count > 0:
            print(f"\n训练向量模型 ({self.article_count} 文档)...")
            self.vector_system.fit()

    def query(self, question: str, top_k: int = 5):
        """查询知识

        同时使用：
        1. 三元组精确匹配
        2. 向量相似度搜索
        """
        # 三元组查询
        triple_results = self.text_system.query(question)

        # 向量查询
        vector_results = []
        if self.vector_system.fitted:
            vector_results = self.vector_system.search(question, top_k=top_k)

        return {
            'triples': triple_results['results'][:top_k],
            'vector_matches': vector_results,
            'total_triples': len(self.text_system.knowledge.triples),
            'total_documents': self.article_count,
        }

    def get_stats(self):
        """获取统计信息"""
        stats = {
            'articles': self.article_count,
            'triples': len(self.text_system.knowledge.triples),
            'entities': len(self.text_system.knowledge.entities),
        }
        if self.vector_system.fitted:
            stats['vector_docs'] = len(self.vector_system.documents)
            stats['vector_clusters'] = self.vector_system.n_clusters
        return stats

    def save(self, path: str):
        """保存系统"""
        import pickle
        data = {
            'text_system': {
                'triples': self.text_system.knowledge.triples,
                'entities': self.text_system.knowledge.entities,
                'summaries': self.text_system.knowledge.summaries,
                'total_articles': self.text_system.total_articles,
            },
            'article_count': self.article_count,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

        # 保存向量模型
        vector_path = path.replace('.pkl', '_vector.pkl')
        if self.vector_system.fitted:
            self.vector_system.save(vector_path)

        print(f"  保存到: {path}")

    def load(self, path: str):
        """加载系统"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # 恢复文本系统
        text_data = data['text_system']
        self.text_system.knowledge.triples = text_data['triples']
        self.text_system.knowledge.entities = text_data['entities']
        self.text_system.knowledge.summaries = text_data['summaries']
        self.text_system.total_articles = text_data['total_articles']
        self.article_count = data['article_count']

        # 重建索引
        for entity in self.text_system.knowledge.entities:
            self.text_system.knowledge.subject_index[entity] = []
            self.text_system.knowledge.object_index[entity] = []

        for i, (s, r, o) in enumerate(self.text_system.knowledge.triples):
            self.text_system.knowledge.subject_index[s].append(i)
            self.text_system.knowledge.object_index[o].append(i)

        # 加载向量模型
        vector_path = path.replace('.pkl', '_vector.pkl')
        if os.path.exists(vector_path):
            self.vector_system.load(vector_path)

        print(f"  加载: {self.article_count} 文章, {len(self.text_system.knowledge.triples)} 三元组")


def stream_jsonl(filepath, max_lines=None):
    """流式读取 JSONL 文件"""
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
                    break
            except:
                continue


def learn_dataset(system, dataset_name, data_dir, max_lines=None):
    """学习一个数据集"""
    print(f"\n学习 {dataset_name}...")
    count = 0

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
                            system.learn_from_article(title, text)
                            count += 1
                        if count % 10000 == 0:
                            print(f"  已学习: {count} 条")
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
                        system.learn_from_article(f"问答:{category}", text)
                        count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条")
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
                        system.learn_from_article(f"新闻:{title[:20]}", text)
                        count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条")
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
                        system.learn_from_article(f"翻译:{chinese[:20]}", chinese)
                        count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条")
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
                        system.learn_from_article(f"社区:{topic}", text)
                        count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

    except Exception as e:
        print(f"  错误: {e}")

    print(f"  {dataset_name} 完成: {count} 条")
    return count


def main():
    print("=" * 70)
    print("整合学习系统 — 正则提取 + 向量化检索")
    print("=" * 70)

    data_dir = 'data/extracted'
    output_dir = Path('data/knowledge/integrated')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否有检查点
    checkpoint_path = output_dir / 'checkpoint.pkl'
    system = IntegratedLearningSystem()

    if checkpoint_path.exists():
        print("\n加载检查点...")
        system.load(str(checkpoint_path))
        print(f"  已有: {system.get_stats()}")

    # 每个数据集学习的数量（测试用，设为 None 学习全部）
    max_per_dataset = 10000  # 先学1万条测试

    start_time = time.time()

    # 学习数据集
    datasets = ['wiki', 'baike', 'news', 'translation', 'webtext']

    for dataset in datasets:
        count = learn_dataset(system, dataset, data_dir, max_per_dataset)

        # 每个数据集后保存检查点
        system.save(str(checkpoint_path))
        print(f"  检查点已保存")

    # 训练向量模型
    system.fit()

    elapsed = time.time() - start_time

    # 保存最终结果
    final_path = output_dir / 'final_system.pkl'
    system.save(str(final_path))

    # 统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)
    stats = system.get_stats()
    print(f"  耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  文章数: {stats['articles']:,}")
    print(f"  三元组: {stats['triples']:,}")
    print(f"  实体数: {stats['entities']:,}")

    # 测试查询
    print("\n" + "=" * 70)
    print("测试查询")
    print("=" * 70)

    test_questions = [
        "什么是人工智能",
        "中国在哪里",
        "牛顿发现了什么",
        "Python是什么语言",
        "太阳是什么",
    ]

    for q in test_questions:
        print(f"\n问: {q}")
        result = system.query(q, top_k=3)

        print(f"  三元组结果 ({len(result['triples'])} 个):")
        for r in result['triples'][:3]:
            if 'subject' in r:
                print(f"    → {r['subject']} {r['relation']} {r['object']}")

        print(f"  向量匹配 ({len(result['vector_matches'])} 个):")
        for r in result['vector_matches'][:3]:
            print(f"    → {r['title']} (相似度: {r['similarity']:.3f})")


if __name__ == '__main__':
    main()
