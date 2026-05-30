"""语料学习脚本 — 从 Wikipedia/百科语料中学习知识

运行方式：
    python training/learn_from_corpus.py

功能：
1. 读取 wiki_zh_2019.zip 中的中文维基百科数据
2. 提取知识实体和关系
3. 构建知识图谱
4. 为问答系统提供知识基础
"""

import json
import os
import sys
import zipfile
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CorpusLearner:
    """语料学习器 — 从文本中提取知识"""

    def __init__(self):
        # 知识存储
        self.entities: Dict[str, Dict] = {}  # 实体
        self.relations: List[Tuple[str, str, str]] = []  # (subject, relation, object)
        self.concepts: Dict[str, List[str]] = defaultdict(list)  # 概念 → 相关词

        # 统计
        self.total_articles = 0
        self.total_entities = 0
        self.total_relations = 0

        # 停用词
        self.stopwords = set('的了是在我你他她它们这那个有不人大中上下来')

    def learn_from_zip(self, zip_path: str, max_articles: int = 10000):
        """从 zip 文件中学习"""
        print(f"  读取: {zip_path}")

        with zipfile.ZipFile(zip_path, 'r') as z:
            files = [f for f in z.namelist() if not f.endswith('/')]
            print(f"  文件数: {len(files)}")

            for file_path in files:
                if self.total_articles >= max_articles:
                    break

                try:
                    with z.open(file_path) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        self._process_content(content)
                except Exception as e:
                    continue

        print(f"  学习完成: {self.total_articles} 文章, {self.total_entities} 实体, {self.total_relations} 关系")

    def _process_content(self, content: str):
        """处理文件内容"""
        # 每行一个 JSON 对象
        for line in content.strip().split('\n'):
            if not line.strip():
                continue

            try:
                article = json.loads(line)
                self._process_article(article)
                self.total_articles += 1

                if self.total_articles % 1000 == 0:
                    print(f"    已学习: {self.total_articles} 文章, {self.total_entities} 实体")
            except:
                continue

    def _process_article(self, article: Dict):
        """处理单篇文章"""
        title = article.get('title', '')
        text = article.get('text', '')

        if not title or not text:
            return

        # 提取实体（标题）
        self._add_entity(title, 'article', text[:200])

        # 提取文本中的实体和关系
        self._extract_from_text(text, title)

    def _extract_from_text(self, text: str, context: str):
        """从文本中提取实体和关系"""
        # 简单的实体提取：中文词（2-4字）
        words = re.findall(r'[一-鿿]{2,4}', text)

        # 统计词频
        word_freq = defaultdict(int)
        for word in words:
            if word not in self.stopwords and len(word) >= 2:
                word_freq[word] += 1

        # 添加高频词为实体
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
            if freq >= 3:
                self._add_entity(word, 'concept', context[:100])
                self.concepts[context[:20]].append(word)

    def _add_entity(self, name: str, entity_type: str, context: str = ''):
        """添加实体"""
        if name not in self.entities:
            self.entities[name] = {
                'name': name,
                'type': entity_type,
                'contexts': [],
                'count': 0,
            }
            self.total_entities += 1

        self.entities[name]['count'] += 1
        if context and len(self.entities[name]['contexts']) < 5:
            self.entities[name]['contexts'].append(context[:100])

    def _add_relation(self, subject: str, relation: str, obj: str):
        """添加关系"""
        self.relations.append((subject, relation, obj))
        self.total_relations += 1

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索知识"""
        results = []

        # 精确匹配
        if query in self.entities:
            results.append({
                'name': query,
                'type': self.entities[query]['type'],
                'count': self.entities[query]['count'],
                'contexts': self.entities[query]['contexts'],
                'score': 1.0,
            })

        # 模糊匹配
        for name, entity in self.entities.items():
            if query in name or name in query:
                if name != query:
                    results.append({
                        'name': name,
                        'type': entity['type'],
                        'count': entity['count'],
                        'contexts': entity['contexts'],
                        'score': 0.5,
                    })

        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_articles': self.total_articles,
            'total_entities': self.total_entities,
            'total_relations': self.total_relations,
            'top_entities': sorted(
                self.entities.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10],
        }

    def save(self, path: str):
        """保存知识"""
        data = {
            'entities': self.entities,
            'relations': self.relations,
            'concepts': dict(self.concepts),
            'stats': self.get_stats(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  保存到: {path}")

    def load(self, path: str):
        """加载知识"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.entities = data.get('entities', {})
            self.relations = data.get('relations', [])
            self.concepts = defaultdict(list, data.get('concepts', {}))
            print(f"  加载: {len(self.entities)} 实体, {len(self.relations)} 关系")


def main():
    print("=" * 70)
    print("语料学习 — 从 Wikipedia 中文语料中学习知识")
    print("=" * 70)

    learner = CorpusLearner()

    # 检查是否已有学习结果
    knowledge_path = 'data/knowledge/corpus_knowledge.json'
    if os.path.exists(knowledge_path):
        print("\n[1] 加载已有知识...")
        learner.load(knowledge_path)
    else:
        # 学习语料
        print("\n[1] 学习语料...")
        start_time = time.time()

        # 学习 Wikipedia
        wiki_path = 'data/wiki_zh_2019.zip'
        if os.path.exists(wiki_path):
            learner.learn_from_zip(wiki_path, max_articles=5000)

        elapsed = time.time() - start_time
        print(f"  耗时: {elapsed:.1f} 秒")

        # 保存
        print("\n[2] 保存知识...")
        Path('data/knowledge').mkdir(parents=True, exist_ok=True)
        learner.save(knowledge_path)

    # 测试搜索
    print("\n[3] 测试搜索...")
    test_queries = ['数学', '物理', '化学', '历史', '地理']
    for query in test_queries:
        results = learner.search(query)
        print(f"  '{query}': {len(results)} 结果")
        if results:
            print(f"    最佳: {results[0]['name']} (出现 {results[0]['count']} 次)")

    # 统计
    stats = learner.get_stats()
    print(f"\n[4] 统计:")
    print(f"  文章数: {stats['total_articles']}")
    print(f"  实体数: {stats['total_entities']}")
    print(f"  关系数: {stats['total_relations']}")

    print("\n" + "=" * 70)
    print("语料学习完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
