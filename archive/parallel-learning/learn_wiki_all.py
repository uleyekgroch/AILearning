"""
Wiki全量学习脚本 - CUDA加速版

学习所有wiki文章，并审查问题
"""

import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.production.domain.human_learning.predictive_system import PredictiveLearningSystem
from src.production.domain.human_learning.memory_consolidation import MemoryConsolidation
from src.production.domain.human_learning.biological_learning import BiologicalLearning
from src.production.domain.human_learning.concept_formation import ConceptFormation
from src.production.application.services.knowledge_service import KnowledgeApplicationService


class WikiFullLearner:
    """Wiki全量学习器"""

    def __init__(self, input_dim: int = 64):
        """
        初始化学习器

        Args:
            input_dim: 输入维度
        """
        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 预测模型（CUDA加速）
        self.weights_input_hidden = torch.randn(input_dim, input_dim * 2, device=self.device) * 0.001
        self.weights_hidden_output = torch.randn(input_dim * 2, input_dim, device=self.device) * 0.001
        self.bias_hidden = torch.zeros(input_dim * 2, device=self.device)
        self.bias_output = torch.zeros(input_dim, device=self.device)
        self.learning_rate = 0.0001

        # 记忆系统
        self.memory_system = MemoryConsolidation(hippocampal_capacity=20)

        # 概念系统
        self.concept_system = ConceptFormation(feature_dim=input_dim)

        # 知识库
        self.knowledge_service = KnowledgeApplicationService()

        # 学习统计
        self.stats = {
            'total_articles': 0,
            'total_sentences': 0,
            'total_concepts': 0,
            'total_errors': 0,
            'avg_prediction_error': 0.0,
            'learning_time': 0.0,
            'files_processed': 0,
            'errors': [],
        }

        # 审查问题
        self.issues = []

    def load_wiki_file(self, filepath: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        加载wiki文件

        Args:
            filepath: 文件路径
            limit: 加载数量限制

        Returns:
            文章列表
        """
        articles = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break

                    try:
                        article = json.loads(line.strip())
                        articles.append(article)
                    except json.JSONDecodeError as e:
                        self.issues.append({
                            'type': 'json_error',
                            'file': filepath,
                            'line': i,
                            'error': str(e)
                        })
        except Exception as e:
            self.issues.append({
                'type': 'file_error',
                'file': filepath,
                'error': str(e)
            })

        return articles

    def extract_sentences(self, text: str, max_sentences: int = 20) -> List[str]:
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
            if len(s) > 5 and len(s) < 500:
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
        vector = np.zeros(self.input_dim)

        for i, char in enumerate(text[:self.input_dim]):
            vector[i % self.input_dim] += ord(char) / 1000000.0

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        tensor = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
        return tensor

    def learn_batch(self, sentences: List[str]) -> List[float]:
        """
        批量学习句子

        Args:
            sentences: 句子列表

        Returns:
            预测误差列表
        """
        errors = []

        for sentence in sentences:
            try:
                input_tensor = self.text_to_tensor(sentence)

                # 前向传播
                hidden = torch.mm(input_tensor, self.weights_input_hidden) + self.bias_hidden
                hidden = torch.relu(hidden)
                prediction = torch.mm(hidden, self.weights_hidden_output) + self.bias_output

                # 计算误差
                error = input_tensor - prediction
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

                self.weights_input_hidden -= self.learning_rate * torch.mm(input_tensor.t(), hidden_gradient)
                self.bias_hidden -= self.learning_rate * hidden_gradient.squeeze()

                errors.append(error_magnitude)

                # 记忆巩固
                self.memory_system.learn(sentence, importance=0.5)

            except Exception as e:
                self.issues.append({
                    'type': 'learning_error',
                    'sentence': sentence[:50],
                    'error': str(e)
                })

        return errors

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
            return {'title': title, 'sentences_learned': 0, 'avg_error': 0.0}

        # 批量学习
        errors = self.learn_batch(sentences)

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
        self.stats['total_errors'] += len(errors)

        avg_error = np.mean(errors) if errors else 0.0

        return {
            'title': title,
            'sentences_learned': len(sentences),
            'avg_error': avg_error,
            'concept_id': concept.concept_id,
        }

    def learn_all_wiki(self, wiki_dir: str, limit_per_file: int = None) -> Dict[str, Any]:
        """
        学习所有wiki文件

        Args:
            wiki_dir: wiki目录
            limit_per_file: 每个文件的学习限制

        Returns:
            学习统计
        """
        start_time = time.time()

        # 获取所有wiki文件
        wiki_files = sorted(Path(wiki_dir).glob('wiki_*'))
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

                if (j + 1) % 10 == 0:
                    print(f"  已学习 {j+1}/{len(articles)} 篇文章")

            # 定期睡眠巩固
            if (i + 1) % 5 == 0:
                print(f"  睡眠巩固中...")
                self.memory_system.sleep()

            self.stats['files_processed'] += 1

        # 最终睡眠巩固
        print("\n最终睡眠巩固...")
        self.memory_system.sleep()

        # 计算统计
        self.stats['learning_time'] = time.time() - start_time
        if self.stats['total_errors'] > 0:
            self.stats['avg_prediction_error'] = self.stats['total_errors'] / self.stats['total_sentences']

        return {
            'files_processed': self.stats['files_processed'],
            'articles_learned': self.stats['total_articles'],
            'sentences_learned': self.stats['total_sentences'],
            'concepts_formed': self.stats['total_concepts'],
            'learning_time': self.stats['learning_time'],
            'avg_prediction_error': self.stats['avg_prediction_error'],
            'issues_count': len(self.issues),
        }

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        查询

        Args:
            question: 问题
            top_k: 返回结果数量

        Returns:
            查询结果
        """
        # 转换为CUDA张量
        input_tensor = self.text_to_tensor(question)

        # 预测
        hidden = torch.mm(input_tensor, self.weights_input_hidden) + self.bias_hidden
        hidden = torch.relu(hidden)
        prediction = torch.mm(hidden, self.weights_hidden_output) + self.bias_output

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
            'related_concepts': related_concepts[:top_k],
            'prediction_norm': torch.norm(prediction).item(),
        }

    def get_issues_report(self) -> Dict[str, Any]:
        """
        获取问题报告

        Returns:
            问题报告
        """
        # 按类型分组
        issues_by_type = {}
        for issue in self.issues:
            issue_type = issue['type']
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

        return {
            'total_issues': len(self.issues),
            'issues_by_type': {k: len(v) for k, v in issues_by_type.items()},
            'sample_issues': {k: v[:5] for k, v in issues_by_type.items()},
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'device': str(self.device),
            'memory_stats': self.memory_system.get_stats(),
            'concept_stats': self.concept_system.get_stats(),
        }


def main():
    """主函数"""
    print("=" * 60)
    print("Wiki全量学习系统 - CUDA加速版")
    print("=" * 60)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: CUDA不可用，使用CPU")

    # 创建学习器
    learner = WikiFullLearner(input_dim=64)

    # Wiki目录
    wiki_dir = "data/extracted/wiki/wiki_zh/AA"

    # 检查目录
    if not os.path.exists(wiki_dir):
        print(f"错误: 目录不存在 {wiki_dir}")
        return

    # 学习所有wiki
    print("\n开始学习所有wiki文章...")
    stats = learner.learn_all_wiki(wiki_dir, limit_per_file=100)

    # 打印统计
    print("\n" + "=" * 60)
    print("学习完成!")
    print("=" * 60)
    print(f"处理文件数: {stats['files_processed']}")
    print(f"学习文章数: {stats['articles_learned']}")
    print(f"学习句子数: {stats['sentences_learned']}")
    print(f"形成概念数: {stats['concepts_formed']}")
    print(f"学习时间: {stats['learning_time']:.2f}秒")
    print(f"平均预测误差: {stats['avg_prediction_error']:.4f}")
    print(f"问题数量: {stats['issues_count']}")

    # 打印问题报告
    issues_report = learner.get_issues_report()
    print("\n" + "=" * 60)
    print("问题报告")
    print("=" * 60)
    print(f"总问题数: {issues_report['total_issues']}")
    print(f"问题类型: {issues_report['issues_by_type']}")

    if issues_report['sample_issues']:
        print("\n问题示例:")
        for issue_type, issues in issues_report['sample_issues'].items():
            print(f"\n{issue_type}:")
            for issue in issues[:3]:
                print(f"  - {issue}")

    # 测试查询
    print("\n" + "=" * 60)
    print("测试查询")
    print("=" * 60)

    test_queries = [
        "数学是什么",
        "物理定律",
        "化学反应",
        "历史事件",
        "计算机科学",
        "人工智能",
        "生物学",
        "经济学",
    ]

    for query in test_queries:
        result = learner.query(query)
        print(f"\n查询: {query}")
        print(f"  相关记忆: {result['related_memories']}")
        print(f"  相关概念: {result['related_concepts']}")

    # 保存统计
    stats_file = "data/wiki_full_learning_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'issues_report': issues_report,
            'device': str(learner.device),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n统计信息已保存到: {stats_file}")


if __name__ == '__main__':
    main()
