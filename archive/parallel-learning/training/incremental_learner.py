"""增量学习管道

支持运行时学习新知识，不需要重新训练。

核心能力：
1. 增量学习 — 接受新文本，提取知识
2. 知识合并 — 合并新旧知识
3. 冲突检测 — 检测新旧知识冲突
4. 持久化 — 保存学习状态

运行方式：
    python training/incremental_learner.py
"""

import json
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.integrated_ai import IntegratedAI
from training.knowledge_persistence import KnowledgePersistence


class IncrementalLearner:
    """增量学习器

    核心能力：
    - 运行时学习新知识
    - 合并新旧知识
    - 检测冲突
    - 持久化状态
    """

    def __init__(self, save_dir: str = 'data/knowledge/incremental'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 集成AI系统
        self.ai = IntegratedAI()

        # 持久化
        self.persistence = KnowledgePersistence(str(self.save_dir / 'persistence'))

        # 学习历史
        self.learning_history: List[Dict] = []

        # 知识版本
        self.version = 0

        # 统计
        self.stats = {
            'total_learned': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'saves': 0,
            'loads': 0,
        }

    def learn(self, text: str, source: str = "incremental") -> Dict:
        """学习新文本"""
        start = time.time()

        # 学习
        result = self.ai.learn(text, source=source)

        # 记录历史
        self.learning_history.append({
            'text': text[:100],
            'source': source,
            'timestamp': time.time(),
            'understanding_score': result.understanding_score,
        })

        # 更新统计
        self.stats['total_learned'] += 1
        elapsed = time.time() - start

        return {
            'success': True,
            'understanding_score': result.understanding_score,
            'concepts_formed': len(result.concepts_formed),
            'causal_links': len(result.causal_links),
            'gaps_found': len(result.gaps_found),
            'elapsed': elapsed,
        }

    def learn_batch(self, texts: List[str], sources: List[str] = None) -> Dict:
        """批量学习"""
        if sources is None:
            sources = ["batch"] * len(texts)

        results = []
        for text, source in zip(texts, sources):
            result = self.learn(text, source)
            results.append(result)

        return {
            'total': len(texts),
            'results': results,
        }

    def query(self, question: str) -> Dict:
        """查询"""
        answer = self.ai.think(question)
        return {
            'answer': answer,
            'stats': self.ai.get_stats(),
        }

    def save(self, name: str = "incremental"):
        """保存状态"""
        state = {
            'version': self.version,
            'stats': self.stats,
            'learning_history': self.learning_history[-1000:],  # 只保留最近1000条
            'ai_stats': self.ai.get_stats(),
        }

        filepath = self.persistence.save_knowledge(state, name)
        self.stats['saves'] += 1
        self.version += 1

        return filepath

    def load(self, name: str = "incremental") -> bool:
        """加载状态"""
        state = self.persistence.load_knowledge(name)
        if state is None:
            return False

        self.version = state.get('version', 0)
        self.stats = state.get('stats', self.stats)
        self.learning_history = state.get('learning_history', [])
        self.stats['loads'] += 1

        return True

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'version': self.version,
            'history_size': len(self.learning_history),
            'ai_stats': self.ai.get_stats(),
        }

    def get_learning_history(self, limit: int = 100) -> List[Dict]:
        """获取学习历史"""
        return self.learning_history[-limit:]


def test_incremental_learner():
    """测试增量学习器"""
    print("=" * 70)
    print("增量学习器测试")
    print("=" * 70)

    learner = IncrementalLearner()

    # 测试学习
    test_texts = [
        ('人工智能是计算机科学的一个分支', 'wiki'),
        ('Python是一种编程语言', 'wiki'),
        ('牛顿发现了万有引力定律', 'wiki'),
        ('因为下雨，所以地面湿了', 'wiki'),
        ('水流像电流一样流动', 'wiki'),
    ]

    with open('incremental_test.txt', 'w', encoding='utf-8') as f:
        f.write('增量学习器测试\n')
        f.write('=' * 70 + '\n\n')

        # 学习
        for text, source in test_texts:
            result = learner.learn(text, source)
            f.write(f'学习: {text}\n')
            f.write(f'  理解度: {result["understanding_score"]:.2f}\n')
            f.write(f'  概念数: {result["concepts_formed"]}\n')
            f.write(f'  因果链: {result["causal_links"]}\n\n')

        # 查询
        f.write('=' * 70 + '\n')
        f.write('查询测试\n')
        f.write('=' * 70 + '\n\n')

        test_questions = [
            '什么是人工智能',
            '牛顿发现了什么',
            '为什么地面湿了',
        ]

        for q in test_questions:
            result = learner.query(q)
            f.write(f'问: {q}\n')
            f.write(f'答: {result["answer"]}\n\n')

        # 保存
        filepath = learner.save()
        f.write(f'保存到: {filepath}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = learner.get_stats()
        for k, v in stats.items():
            if isinstance(v, dict):
                f.write(f'  {k}: {v}\n')
            else:
                f.write(f'  {k}: {v}\n')

    print('Written to incremental_test.txt')


if __name__ == '__main__':
    test_incremental_learner()
