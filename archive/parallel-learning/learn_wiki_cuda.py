"""
Wiki语料库学习脚本 - CUDA加速版

使用GPU加速学习
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np
import torch

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.production.domain.human_learning.predictive_system import PredictiveLearningSystem
from src.production.domain.human_learning.memory_consolidation import MemoryConsolidation
from src.production.domain.human_learning.biological_learning import BiologicalLearning
from src.production.domain.human_learning.concept_formation import ConceptFormation
from src.production.application.services.knowledge_service import KnowledgeApplicationService


class CUDAPredictiveModel:
    """CUDA加速的预测模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        初始化CUDA预测模型

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 预测权重（CUDA加速）
        self.weights_input_hidden = torch.randn(input_dim, hidden_dim, device=self.device) * 0.001
        self.weights_hidden_output = torch.randn(hidden_dim, input_dim, device=self.device) * 0.001
        self.bias_hidden = torch.zeros(hidden_dim, device=self.device)
        self.bias_output = torch.zeros(input_dim, device=self.device)

        # 学习率
        self.learning_rate = 0.0001

        # 统计
        self.stats = {
            'total_predictions': 0,
            'total_errors': 0,
            'avg_error': 0.0,
        }

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        预测

        Args:
            input_data: 输入数据

        Returns:
            预测值
        """
        # 前向传播
        hidden = torch.mm(input_data, self.weights_input_hidden) + self.bias_hidden
        hidden = torch.relu(hidden)
        prediction = torch.mm(hidden, self.weights_hidden_output) + self.bias_output

        self.stats['total_predictions'] += 1
        return prediction

    def learn(self, input_data: torch.Tensor, actual: torch.Tensor) -> float:
        """
        学习

        Args:
            input_data: 输入数据
            actual: 实际值

        Returns:
            预测误差
        """
        # 前向传播
        hidden = torch.mm(input_data, self.weights_input_hidden) + self.bias_hidden
        hidden = torch.relu(hidden)
        prediction = torch.mm(hidden, self.weights_hidden_output) + self.bias_output

        # 计算误差
        error = actual - prediction
        error_magnitude = torch.norm(error).item()

        # 梯度裁剪
        max_grad_norm = 1.0
        error_direction = error / (error_magnitude + 1e-8)
        error_norm = torch.norm(error_direction)
        if error_norm > max_grad_norm:
            error_direction = error_direction * (max_grad_norm / error_norm)

        # 反向传播
        output_gradient = error_direction
        self.weights_hidden_output -= self.learning_rate * torch.mm(hidden.t(), output_gradient)
        self.bias_output -= self.learning_rate * output_gradient.squeeze()

        hidden_gradient = torch.mm(output_gradient, self.weights_hidden_output.t())
        hidden_gradient[hidden <= 0] = 0
        hidden_norm = torch.norm(hidden_gradient)
        if hidden_norm > max_grad_norm:
            hidden_gradient = hidden_gradient * (max_grad_norm / hidden_norm)

        self.weights_input_hidden -= self.learning_rate * torch.mm(input_data.t(), hidden_gradient)
        self.bias_hidden -= self.learning_rate * hidden_gradient.squeeze()

        # 更新统计
        self.stats['total_errors'] += 1
        self.stats['avg_error'] = (
            self.stats['avg_error'] * (self.stats['total_errors'] - 1) + error_magnitude
        ) / self.stats['total_errors']

        return error_magnitude


class WikiLearnerCUDA:
    """Wiki语料库学习器 - CUDA加速版"""

    def __init__(self, input_dim: int = 64):
        """
        初始化CUDA学习器

        Args:
            input_dim: 输入维度
        """
        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CUDA加速的预测模型
        self.predictive_model = CUDAPredictiveModel(input_dim)

        # 其他系统（CPU）
        self.memory_system = MemoryConsolidation(hippocampal_capacity=10)
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
            'device': str(self.device),
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
            if len(s) > 5 and len(s) < 200:
                cleaned.append(s)

        return cleaned[:max_sentences]

    def text_to_tensor(self, text: str) -> torch.Tensor:
        """
        文本转CUDA张量

        Args:
            text: 文本

        Returns:
            CUDA张量
        """
        # 简化实现：基于字符哈希生成向量
        vector = np.zeros(self.input_dim)

        for i, char in enumerate(text[:self.input_dim]):
            vector[i % self.input_dim] += ord(char) / 1000000.0

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # 转换为CUDA张量
        tensor = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
        return tensor

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
            return {'title': title, 'sentences_learned': 0, 'avg_prediction_error': 0.0, 'concept_id': ''}

        # 学习每个句子
        prediction_errors = []
        for sentence in sentences:
            # 转换为CUDA张量
            input_tensor = self.text_to_tensor(sentence)

            # CUDA加速学习
            error = self.predictive_model.learn(input_tensor, input_tensor)
            prediction_errors.append(error)

            # 记忆巩固
            self.memory_system.learn(sentence, importance=0.5)

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
            sleep_interval: 睡眠间隔

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
            'device': self.stats['device'],
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
        # 转换为CUDA张量
        input_tensor = self.text_to_tensor(question)

        # CUDA加速预测
        prediction = self.predictive_model.predict(input_tensor)

        # 回忆相关记忆
        memories = self.memory_system.recall()

        # 查找相关概念
        related_concepts = []
        for concept_id, concept in self.concept_system.hierarchy.concepts.items():
            if any(char in question for char in concept.name):
                related_concepts.append(concept.name)

        return {
            'question': question,
            'related_memories': len(memories),
            'related_concepts': related_concepts[:5],
            'prediction_norm': torch.norm(prediction).item(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'predictive_stats': self.predictive_model.stats,
            'memory_stats': self.memory_system.get_stats(),
            'concept_stats': self.concept_system.get_stats(),
        }


def main():
    """主函数"""
    print("=" * 50)
    print("Wiki语料库学习系统 - CUDA加速版")
    print("=" * 50)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: CUDA不可用，使用CPU")

    # 创建学习器
    learner = WikiLearnerCUDA(input_dim=32)

    # Wiki文件路径
    wiki_files = [
        "data/extracted/wiki/wiki_zh/AA/wiki_00",
        "data/extracted/wiki/wiki_zh/AA/wiki_01",
        "data/extracted/wiki/wiki_zh/AA/wiki_02",
    ]

    # 学习多个wiki文件
    all_stats = []
    for wiki_file in wiki_files:
        if not os.path.exists(wiki_file):
            print(f"跳过: {wiki_file}")
            continue

        print(f"\n学习: {wiki_file}")
        stats = learner.learn_corpus(wiki_file, limit=100, sleep_interval=20)
        all_stats.append(stats)

    # 打印总统计
    print("\n" + "=" * 50)
    print("学习完成!")
    print("=" * 50)

    total_articles = sum(s['articles_learned'] for s in all_stats)
    total_sentences = sum(s['total_sentences'] for s in all_stats)
    total_concepts = sum(s['total_concepts'] for s in all_stats)
    total_time = sum(s['learning_time'] for s in all_stats)

    print(f"总学习文章数: {total_articles}")
    print(f"总学习句子数: {total_sentences}")
    print(f"总形成概念数: {total_concepts}")
    print(f"总学习时间: {total_time:.2f}秒")
    print(f"平均预测误差: {learner.stats['avg_prediction_error']:.4f}")
    print(f"使用设备: {learner.stats['device']}")

    # 测试查询
    print("\n" + "=" * 50)
    print("测试查询")
    print("=" * 50)

    test_queries = [
        "数学是什么",
        "物理",
        "化学",
        "历史",
        "计算机",
    ]

    for query in test_queries:
        result = learner.query(query)
        print(f"\n查询: {query}")
        print(f"  相关记忆: {result['related_memories']}")
        print(f"  相关概念: {result['related_concepts']}")

    # 保存统计
    stats_file = "data/wiki_learning_cuda_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'all_stats': all_stats,
            'total': {
                'articles': total_articles,
                'sentences': total_sentences,
                'concepts': total_concepts,
                'time': total_time,
            },
            'device': learner.stats['device'],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n统计信息已保存到: {stats_file}")


if __name__ == '__main__':
    main()
