"""
Wiki语料库学习脚本

使用人类学习系统学习wiki语料库
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.production.domain.human_learning.predictive_system import PredictiveLearningSystem
from src.production.domain.human_learning.memory_consolidation import MemoryConsolidation
from src.production.domain.human_learning.biological_learning import BiologicalLearning
from src.production.domain.human_learning.concept_formation import ConceptFormation
from src.production.application.services.knowledge_service import KnowledgeApplicationService


class WikiLearner:
    """Wiki语料库学习器

    使用人类学习系统学习wiki语料库
    """

    def __init__(self, input_dim: int = 64):
        """
        初始化Wiki学习器

        Args:
            input_dim: 输入维度
        """
        self.input_dim = input_dim

        # 人类学习系统
        self.predictive_system = PredictiveLearningSystem(input_dim)
        self.memory_system = MemoryConsolidation(hippocampal_capacity=10)
        self.biological_system = BiologicalLearning(input_dim)
        self.concept_system = ConceptFormation(feature_dim=input_dim)

        # 知识库
        self.knowledge_service = KnowledgeApplicationService()

        # 学习统计
        self.stats = {
            'total_articles': 0,
            'total_sentences': 0,
            'total_concepts': 0,
            'learning_time': 0.0,
            'avg_prediction_error': 0.0,
        }

    def load_wiki_data(self, filepath: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        加载wiki数据

        Args:
            filepath: 文件路径
            limit: 加载数量限制

        Returns:
            文章列表
        """
        articles = []

        print(f"加载wiki数据: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break

                try:
                    article = json.loads(line.strip())
                    articles.append(article)
                except json.JSONDecodeError:
                    continue

        print(f"加载了 {len(articles)} 篇文章")
        return articles

    def extract_sentences(self, text: str, max_sentences: int = 10) -> List[str]:
        """
        提取句子

        Args:
            text: 文本
            max_sentences: 最大句子数

        Returns:
            句子列表
        """
        # 按句号分割
        sentences = text.split('。')

        # 清理和过滤
        cleaned = []
        for s in sentences:
            s = s.strip()
            if len(s) > 5 and len(s) < 200:  # 过滤太短或太长的句子
                cleaned.append(s)

        return cleaned[:max_sentences]

    def text_to_vector(self, text: str) -> np.ndarray:
        """
        文本转向量

        Args:
            text: 文本

        Returns:
            向量
        """
        # 简化实现：基于字符哈希生成向量
        vector = np.zeros(self.input_dim)

        for i, char in enumerate(text[:self.input_dim]):
            vector[i % self.input_dim] += ord(char) / 1000000.0  # 更小的缩放

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def learn_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        学习一篇文章

        Args:
            article: 文章数据

        Returns:
            学习结果
        """
        title = article.get('title', '')
        text = article.get('text', '')

        # 提取句子
        sentences = self.extract_sentences(text)

        if not sentences:
            return {'title': title, 'sentences_learned': 0}

        # 学习每个句子
        prediction_errors = []
        for sentence in sentences:
            # 转换为向量
            vector = self.text_to_vector(sentence)

            # 预测学习
            result = self.predictive_system.perceive_and_learn(vector, vector)
            prediction_errors.append(result['error'])

            # 记忆巩固
            self.memory_system.learn(sentence, importance=0.5)

            # 生物学习
            self.biological_system.learn_predictive(vector, vector)

        # 形成概念
        concept = self.concept_system.form_concept(
            name=title,
            examples=sentences[:3]
        )

        # 添加到知识库
        self.knowledge_service.create_knowledge_base(title)

        # 更新统计
        self.stats['total_articles'] += 1
        self.stats['total_sentences'] += len(sentences)
        self.stats['total_concepts'] += 1

        return {
            'title': title,
            'sentences_learned': len(sentences),
            'avg_prediction_error': np.mean(prediction_errors),
            'concept_id': concept.concept_id,
        }

    def learn_corpus(self, filepath: str, limit: int = 100,
                    sleep_interval: int = 10) -> Dict[str, Any]:
        """
        学习语料库

        Args:
            filepath: 文件路径
            limit: 学习数量限制
            sleep_interval: 睡眠间隔（每学习N篇文章后睡眠巩固）

        Returns:
            学习统计
        """
        start_time = time.time()

        # 加载数据
        articles = self.load_wiki_data(filepath, limit)

        # 学习每篇文章
        results = []
        for i, article in enumerate(articles):
            result = self.learn_article(article)
            results.append(result)

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"已学习 {i + 1}/{len(articles)} 篇文章")

            # 定期睡眠巩固
            if (i + 1) % sleep_interval == 0:
                print(f"睡眠巩固中...")
                consolidation = self.memory_system.sleep()
                print(f"  巩固: {consolidation.memories_consolidated}, 遗忘: {consolidation.memories_forgotten}")

        # 最终睡眠巩固
        print("最终睡眠巩固...")
        final_consolidation = self.memory_system.sleep()

        # 计算统计
        self.stats['learning_time'] = time.time() - start_time
        self.stats['avg_prediction_error'] = np.mean([r['avg_prediction_error'] for r in results])

        return {
            'articles_learned': len(results),
            'total_sentences': self.stats['total_sentences'],
            'total_concepts': self.stats['total_concepts'],
            'learning_time': self.stats['learning_time'],
            'avg_prediction_error': self.stats['avg_prediction_error'],
            'final_consolidation': {
                'consolidated': final_consolidation.memories_consolidated,
                'forgotten': final_consolidation.memories_forgotten,
            },
        }

    def query(self, question: str) -> Dict[str, Any]:
        """
        查询

        Args:
            question: 问题

        Returns:
            查询结果
        """
        # 转换为向量
        vector = self.text_to_vector(question)

        # 预测
        prediction = self.predictive_system.predict(vector)

        # 回忆相关记忆
        memories = self.memory_system.recall()

        # 查找相关概念
        related_concepts = []
        for concept_id, concept in self.concept_system.hierarchy.concepts.items():
            # 简单匹配
            if any(char in question for char in concept.name):
                related_concepts.append(concept.name)

        return {
            'question': question,
            'related_memories': len(memories),
            'related_concepts': related_concepts[:5],
            'prediction_norm': np.linalg.norm(prediction),
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'predictive_stats': self.predictive_system.get_stats(),
            'memory_stats': self.memory_system.get_stats(),
            'biological_stats': self.biological_system.get_stats(),
            'concept_stats': self.concept_system.get_stats(),
        }


def main():
    """主函数"""
    print("=" * 50)
    print("Wiki语料库学习系统")
    print("=" * 50)

    # 创建学习器
    learner = WikiLearner(input_dim=32)

    # Wiki文件路径
    wiki_file = "data/extracted/wiki/wiki_zh/AA/wiki_00"

    # 检查文件是否存在
    if not os.path.exists(wiki_file):
        print(f"错误: 文件不存在 {wiki_file}")
        return

    # 学习语料库
    print("\n开始学习wiki语料库...")
    stats = learner.learn_corpus(wiki_file, limit=50, sleep_interval=10)

    # 打印统计
    print("\n" + "=" * 50)
    print("学习完成!")
    print("=" * 50)
    print(f"学习文章数: {stats['articles_learned']}")
    print(f"学习句子数: {stats['total_sentences']}")
    print(f"形成概念数: {stats['total_concepts']}")
    print(f"学习时间: {stats['learning_time']:.2f}秒")
    print(f"平均预测误差: {stats['avg_prediction_error']:.4f}")
    print(f"睡眠巩固: 巩固{stats['final_consolidation']['consolidated']}, 遗忘{stats['final_consolidation']['forgotten']}")

    # 测试查询
    print("\n" + "=" * 50)
    print("测试查询")
    print("=" * 50)

    test_queries = [
        "数学是什么",
        "物理",
        "化学",
        "历史",
    ]

    for query in test_queries:
        result = learner.query(query)
        print(f"\n查询: {query}")
        print(f"  相关记忆: {result['related_memories']}")
        print(f"  相关概念: {result['related_concepts']}")

    # 保存统计
    stats_file = "data/wiki_learning_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计信息已保存到: {stats_file}")


if __name__ == '__main__':
    main()
