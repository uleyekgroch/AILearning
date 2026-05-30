"""外部知识源集成

从外部来源获取知识：
1. Wikipedia API — 实时获取维基百科文章
2. Web搜索 — 搜索网络信息
3. 文件导入 — 从本地文件导入

运行方式：
    python training/external_sources.py
"""

import json
import os
import sys
import time
import re
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.integrated_ai import IntegratedAI


class WikipediaSource:
    """维基百科知识源"""

    def __init__(self):
        self.base_url = "https://zh.wikipedia.org/api/rest_v1"

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """搜索维基百科"""
        try:
            import requests

            url = f"{self.base_url}/page/search/{query}"
            params = {
                'limit': limit,
                'format': 'json',
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pages = data.get('pages', [])
                return [
                    {
                        'title': page.get('title', ''),
                        'description': page.get('description', ''),
                        'excerpt': page.get('excerpt', '')[:200],
                    }
                    for page in pages
                ]
        except Exception as e:
            print(f"Wikipedia搜索失败: {e}")

        return []

    def get_article(self, title: str) -> Optional[Dict]:
        """获取维基百科文章"""
        try:
            import requests

            url = f"{self.base_url}/page/html/{title}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                # 简单提取文本
                html = response.text
                text = self._extract_text(html)
                return {
                    'title': title,
                    'text': text[:2000],  # 限制长度
                }
        except Exception as e:
            print(f"获取文章失败: {e}")

        return None

    def _extract_text(self, html: str) -> str:
        """从HTML提取文本"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', html)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class ExternalKnowledgeManager:
    """外部知识源管理器"""

    def __init__(self):
        self.ai = IntegratedAI()
        self.wikipedia = WikipediaSource()

        # 统计
        self.stats = {
            'wikipedia_queries': 0,
            'articles_ingested': 0,
            'total_learned': 0,
        }

    def learn_from_wikipedia(self, query: str, max_articles: int = 3) -> Dict:
        """从维基百科学习"""
        results = {
            'query': query,
            'articles': [],
            'learned': 0,
        }

        # 搜索文章
        search_results = self.wikipedia.search(query, limit=max_articles)
        self.stats['wikipedia_queries'] += 1

        for article_info in search_results:
            title = article_info.get('title', '')
            if not title:
                continue

            # 获取文章内容
            article = self.wikipedia.get_article(title)
            if article is None:
                continue

            # 学习
            text = article.get('text', '')
            if len(text) > 10:
                self.ai.learn(text, source=f"wikipedia:{title}")
                results['articles'].append({
                    'title': title,
                    'length': len(text),
                })
                results['learned'] += 1
                self.stats['articles_ingested'] += 1
                self.stats['total_learned'] += 1

        return results

    def learn_from_text(self, text: str, source: str = "external") -> Dict:
        """从文本学习"""
        self.ai.learn(text, source=source)
        self.stats['total_learned'] += 1

        return {
            'success': True,
            'source': source,
        }

    def learn_from_file(self, filepath: str) -> Dict:
        """从文件学习"""
        results = {
            'filepath': filepath,
            'lines': 0,
            'learned': 0,
        }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) > 10:
                        self.ai.learn(line, source=f"file:{filepath}")
                        results['learned'] += 1
                        self.stats['total_learned'] += 1
                    results['lines'] += 1
        except Exception as e:
            results['error'] = str(e)

        return results

    def query(self, question: str) -> Dict:
        """查询"""
        answer = self.ai.think(question)
        return {
            'answer': answer,
            'stats': self.ai.get_stats(),
        }

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'ai_stats': self.ai.get_stats(),
        }


def test_external_sources():
    """测试外部知识源"""
    print("=" * 70)
    print("外部知识源测试")
    print("=" * 70)

    manager = ExternalKnowledgeManager()

    with open('external_sources_test.txt', 'w', encoding='utf-8') as f:
        f.write('外部知识源测试\n')
        f.write('=' * 70 + '\n\n')

        # 测试维基百科搜索
        f.write('维基百科搜索测试:\n')
        results = manager.wikipedia.search('人工智能', limit=3)
        for r in results:
            f.write(f'  - {r["title"]}: {r["description"]}\n')

        # 测试从维基百科学习
        f.write('\n从维基百科学习:\n')
        learn_result = manager.learn_from_wikipedia('Python编程语言', max_articles=1)
        f.write(f'  查询: {learn_result["query"]}\n')
        f.write(f'  学习文章数: {learn_result["learned"]}\n')
        for article in learn_result['articles']:
            f.write(f'  - {article["title"]} ({article["length"]} 字)\n')

        # 测试查询
        f.write('\n查询测试:\n')
        test_questions = [
            '什么是Python',
            '人工智能是什么',
        ]

        for q in test_questions:
            result = manager.query(q)
            f.write(f'\n问: {q}\n')
            f.write(f'答: {result["answer"]}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = manager.get_stats()
        for k, v in stats.items():
            if isinstance(v, dict):
                f.write(f'  {k}: {v}\n')
            else:
                f.write(f'  {k}: {v}\n')

    print('Written to external_sources_test.txt')


if __name__ == '__main__':
    test_external_sources()
