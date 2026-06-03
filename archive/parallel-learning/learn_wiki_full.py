"""
Wiki全量学习脚本

学习所有wiki数据，保存学习结果
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np
import torch
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.production.domain.true_learning.learner import TrueLearner


class WikiLearner:
    """Wiki学习器"""

    def __init__(self, embedding_dim: int = 64):
        """初始化学习器"""
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 真正的学习器
        self.learner = TrueLearner(embedding_dim)

        # 知识索引
        self.knowledge_index: Dict[str, Dict[str, Any]] = {}

        # 统计
        self.stats = {
            'total_articles': 0,
            'total_sentences': 0,
            'total_concepts': 0,
            'files_processed': 0,
            'learning_time': 0.0,
        }

    def load_wiki_file(self, filepath: str, limit: int = None) -> List[Dict[str, Any]]:
        """加载wiki文件"""
        articles = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    try:
                        article = json.loads(line.strip())
                        articles.append(article)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  错误: {e}")
        return articles

    def extract_sentences(self, text: str, max_sentences: int = 10) -> List[str]:
        """提取句子"""
        sentences = text.split('。')
        cleaned = []
        for s in sentences:
            s = s.strip()
            if len(s) > 5 and len(s) < 200:
                cleaned.append(s)
        return cleaned[:max_sentences]

    def learn_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """学习一篇文章"""
        title = article.get('title', '')
        text = article.get('text', '')

        # 提取句子
        sentences = self.extract_sentences(text)

        if not sentences:
            return {'title': title, 'sentences_learned': 0}

        # 学习主题
        result = self.learner.learn_topic(
            topic=title,
            facts=sentences,
            relations={}
        )

        # 索引知识
        self.knowledge_index[title] = {
            'facts': sentences,
            'concept': title,
            'score': result.overall_score,
        }

        # 更新统计
        self.stats['total_articles'] += 1
        self.stats['total_sentences'] += len(sentences)
        self.stats['total_concepts'] += 1

        return {
            'title': title,
            'sentences_learned': len(sentences),
            'score': result.overall_score,
        }

    def learn_all_wiki(self, wiki_dir: str, limit_per_file: int = 50) -> Dict[str, Any]:
        """学习所有wiki文件"""
        start_time = time.time()

        # 获取所有wiki文件
        wiki_files = sorted(Path(wiki_dir).rglob('wiki_*'))
        total_files = len(wiki_files)

        print(f"找到 {total_files} 个wiki文件")

        all_results = []
        for i, wiki_file in enumerate(wiki_files):
            print(f"\n处理文件 {i+1}/{total_files}: {wiki_file.name}")

            # 加载文章
            articles = self.load_wiki_file(str(wiki_file), limit=limit_per_file)
            print(f"  加载了 {len(articles)} 篇文章")

            # 学习每篇文章
            for j, article in enumerate(articles):
                result = self.learn_article(article)
                all_results.append(result)

                if (j + 1) == limit_per_file:
                    break

            self.stats['files_processed'] += 1

            # 每10个文件打印进度
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{total_files} 文件, {self.stats['total_articles']} 文章")

        # 计算统计
        self.stats['learning_time'] = time.time() - start_time

        return {
            'files_processed': self.stats['files_processed'],
            'articles_learned': self.stats['total_articles'],
            'sentences_learned': self.stats['total_sentences'],
            'concepts_formed': self.stats['total_concepts'],
            'learning_time': self.stats['learning_time'],
        }

    def save_learning_result(self, filepath: str):
        """保存学习结果"""
        result = {
            'stats': self.stats,
            'knowledge_index': self.knowledge_index,
            'device': str(self.device),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"学习结果已保存到: {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self.stats,
            'knowledge_count': len(self.knowledge_index),
            'device': str(self.device),
        }


def main():
    """主函数"""
    print("=" * 60)
    print("Wiki全量学习系统")
    print("=" * 60)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 创建学习器
    learner = WikiLearner(embedding_dim=32)

    # Wiki目录
    wiki_dir = "data/wiki_zh"

    # 检查目录
    if not os.path.exists(wiki_dir):
        print(f"错误: 目录不存在 {wiki_dir}")
        return

    # 学习所有wiki
    print("\n开始学习所有wiki文章...")
    stats = learner.learn_all_wiki(wiki_dir, limit_per_file=30)

    # 打印统计
    print("\n" + "=" * 60)
    print("学习完成!")
    print("=" * 60)
    print(f"处理文件数: {stats['files_processed']}")
    print(f"学习文章数: {stats['articles_learned']}")
    print(f"学习句子数: {stats['sentences_learned']}")
    print(f"形成概念数: {stats['concepts_formed']}")
    print(f"学习时间: {stats['learning_time']:.2f}秒")

    # 保存学习结果
    learner.save_learning_result("data/wiki_learning_result.json")

    print("\n学习完成！现在可以启动HTTP服务器进行测试。")


if __name__ == '__main__':
    main()
