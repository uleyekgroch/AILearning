"""Layer 5: 元认知层

知道自己知道什么，能自我评估和改进。

核心能力：
1. 自我评估 — 知道自己知道什么、不知道什么
2. 好奇心 — 主动探索未知
3. 学习策略 — 选择最有效的学习方式
4. 错误修正 — 发现并纠正错误
5. 元推理 — 推理自己的推理过程

运行方式：
    python training/layers/metacognition.py
"""

import re
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class KnowledgeItem:
    """知识项"""
    content: str
    confidence: float
    source: str
    timestamp: float
    verification_count: int = 0
    contradiction_count: int = 0

    def verify(self):
        """验证知识"""
        self.verification_count += 1
        self.confidence = min(1.0, self.confidence + 0.1)

    def contradict(self):
        """矛盾知识"""
        self.contradiction_count += 1
        self.confidence = max(0.0, self.confidence - 0.2)


@dataclass
class KnowledgeGap:
    """知识空白"""
    topic: str
    context: str
    importance: float
    timestamp: float

    def to_question(self) -> str:
        """转换为问题"""
        return f"什么是{self.topic}？"


@dataclass
class LearningStrategy:
    """学习策略"""
    name: str
    description: str
    effectiveness: float = 0.5
    usage_count: int = 0
    success_count: int = 0

    def update_effectiveness(self, success: bool):
        """更新效果"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.effectiveness = self.success_count / self.usage_count


class Metacognition:
    """元认知层

    核心能力：
    - 知道自己知道什么
    - 主动探索未知
    - 自我改进
    """

    def __init__(self):
        # 知识库
        self.knowledge: Dict[str, KnowledgeItem] = {}

        # 知识空白
        self.knowledge_gaps: List[KnowledgeGap] = []

        # 学习策略
        self.strategies: Dict[str, LearningStrategy] = self._init_strategies()

        # 学习历史
        self.learning_history: List[Dict] = []

        # 好奇心队列
        self.curiosity_queue: List[str] = []

        # 自我评估
        self.self_assessment = {
            'knowledge_coverage': 0.0,
            'reasoning_ability': 0.0,
            'learning_efficiency': 0.0,
        }

        # 统计
        self.stats = {
            'items_learned': 0,
            'gaps_identified': 0,
            'strategies_used': 0,
            'self_corrections': 0,
        }

    def _init_strategies(self) -> Dict[str, LearningStrategy]:
        """初始化学习策略"""
        return {
            'repetition': LearningStrategy(
                name='repetition',
                description='重复学习加深记忆',
            ),
            'elaboration': LearningStrategy(
                name='elaboration',
                description='详细解释加深理解',
            ),
            'association': LearningStrategy(
                name='association',
                description='关联已有知识',
            ),
            'practice': LearningStrategy(
                name='practice',
                description='通过实践学习',
            ),
            'teaching': LearningStrategy(
                name='teaching',
                description='通过教别人学习',
            ),
        }

    def learn(self, content: str, source: str = "text") -> Dict:
        """学习新内容

        不只是存储，还要：
        1. 评估理解程度
        2. 检查是否与已有知识矛盾
        3. 识别知识空白
        4. 选择学习策略
        """
        result = {
            'content': content,
            'understood': False,
            'confidence': 0.0,
            'gaps_found': [],
            'contradictions': [],
            'strategy_used': None,
        }

        # 1. 评估理解程度
        understanding = self._assess_understanding(content)
        result['understood'] = understanding > 0.5
        result['confidence'] = understanding

        # 2. 检查矛盾
        contradictions = self._check_contradictions(content)
        result['contradictions'] = contradictions

        # 3. 识别知识空白
        gaps = self._identify_gaps(content)
        result['gaps_found'] = gaps

        # 4. 选择学习策略
        strategy = self._select_strategy(content, understanding)
        result['strategy_used'] = strategy.name if strategy else None

        # 5. 存储知识（用hash避免key冲突）
        item = KnowledgeItem(
            content=content,
            confidence=understanding,
            source=source,
            timestamp=time.time(),
        )
        key = str(hash(content))
        self.knowledge[key] = item

        # 6. 更新统计
        self.stats['items_learned'] += 1
        self.stats['gaps_identified'] += len(gaps)

        # 7. 记录学习历史
        self.learning_history.append({
            'timestamp': time.time(),
            'content': content[:100],
            'understanding': understanding,
            'strategy': strategy.name if strategy else None,
        })

        return result

    def _assess_understanding(self, content: str) -> float:
        """评估理解程度"""
        # 基于内容特征评估
        score = 0.5  # 基础分

        # 包含关键词的更容易理解
        if any(word in content for word in ['是', '属于', '包括', '位于']):
            score += 0.1

        # 包含具体细节的更容易理解
        if re.search(r'\d+', content):
            score += 0.1

        # 包含因果关系的更容易理解
        if any(word in content for word in ['因为', '所以', '导致', '引起']):
            score += 0.1

        # 太短或太长的都难理解
        if len(content) < 10:
            score -= 0.2
        elif len(content) > 200:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _check_contradictions(self, content: str) -> List[Dict]:
        """检查是否与已有知识矛盾"""
        contradictions = []

        # 提取关键信息
        key_info = self._extract_key_info(content)

        for key, value in key_info.items():
            # 检查是否已有相反的信息
            for existing_key, existing_item in self.knowledge.items():
                # 从已有知识中提取信息
                existing_info = self._extract_key_info(existing_item.content)
                # 检查同一个主语是否有不同的宾语
                if key in existing_info and existing_info[key] != value:
                    contradictions.append({
                        'key': key,
                        'new_value': value,
                        'existing_value': existing_info[key],
                        'existing_content': existing_item.content[:50],
                    })
                    # 标记矛盾
                    existing_item.contradict()

        return contradictions

    def _extract_key_info(self, content: str) -> Dict[str, str]:
        """提取关键信息"""
        info = {}

        # X是Y模式
        matches = re.findall(r'(.{2,10})是(.{2,20})', content)
        for match in matches:
            key = match[0].strip()
            value = match[1].strip()
            info[key] = value

        return info

    def _identify_gaps(self, content: str) -> List[KnowledgeGap]:
        """识别知识空白"""
        gaps = []

        # 提取实体
        entities = re.findall(r'[一-鿿]{2,6}', content)

        for entity in entities:
            # 检查是否已有这个实体的详细知识
            has_detailed_knowledge = False
            for key, item in self.knowledge.items():
                if entity in key and item.confidence > 0.7:
                    has_detailed_knowledge = True
                    break

            if not has_detailed_knowledge:
                gap = KnowledgeGap(
                    topic=entity,
                    context=content[:100],
                    importance=0.5,
                    timestamp=time.time(),
                )
                gaps.append(gap)
                self.knowledge_gaps.append(gap)

        return gaps

    def _select_strategy(self, content: str, understanding: float) -> Optional[LearningStrategy]:
        """选择学习策略"""
        # 根据理解程度选择策略
        if understanding < 0.3:
            # 理解度低，用详细解释
            return self.strategies.get('elaboration')
        elif understanding < 0.6:
            # 中等理解度，用关联
            return self.strategies.get('association')
        else:
            # 高理解度，用实践
            return self.strategies.get('practice')

    def ask_question(self) -> Optional[str]:
        """提出问题（好奇心驱动）"""
        # 优先从知识空白中提问
        if self.knowledge_gaps:
            # 按重要性排序
            self.knowledge_gaps.sort(key=lambda x: x.importance, reverse=True)
            gap = self.knowledge_gaps.pop(0)
            question = gap.to_question()
            self.curiosity_queue.append(question)
            return question

        # 从低置信度知识中提问
        low_confidence = [
            (key, item) for key, item in self.knowledge.items()
            if item.confidence < 0.5
        ]
        if low_confidence:
            key, item = low_confidence[0]
            question = f"能更详细地解释一下：{item.content[:30]}？"
            self.curiosity_queue.append(question)
            return question

        return None

    def self_evaluate(self) -> Dict:
        """自我评估"""
        # 计算知识覆盖率
        if self.knowledge:
            avg_confidence = sum(item.confidence for item in self.knowledge.values()) / len(self.knowledge)
            self.self_assessment['knowledge_coverage'] = avg_confidence

        # 计算学习效率
        if self.learning_history:
            recent = self.learning_history[-100:]
            avg_understanding = sum(h['understanding'] for h in recent) / len(recent)
            self.self_assessment['learning_efficiency'] = avg_understanding

        # 计算推理能力（基于矛盾检测）
        total_contradictions = sum(
            item.contradiction_count for item in self.knowledge.values()
        )
        if self.knowledge:
            contradiction_rate = total_contradictions / len(self.knowledge)
            self.self_assessment['reasoning_ability'] = 1.0 - min(1.0, contradiction_rate)

        return self.self_assessment

    def correct_error(self, content: str, correction: str):
        """纠正错误"""
        # 找到错误知识
        for key, item in self.knowledge.items():
            if content in item.content:
                # 更新知识
                item.content = correction
                item.confidence = 0.8  # 纠正后的置信度
                item.verification_count += 1

                self.stats['self_corrections'] += 1

                # 记录纠正历史
                self.learning_history.append({
                    'timestamp': time.time(),
                    'type': 'correction',
                    'original': content,
                    'correction': correction,
                })

                return True

        return False

    def get_learning_report(self) -> Dict:
        """生成学习报告"""
        self.self_evaluate()

        return {
            'total_knowledge': len(self.knowledge),
            'knowledge_gaps': len(self.knowledge_gaps),
            'self_assessment': self.self_assessment,
            'stats': self.stats,
            'top_strategies': sorted(
                self.strategies.values(),
                key=lambda x: x.effectiveness,
                reverse=True
            )[:3],
            'recent_learning': self.learning_history[-10:] if self.learning_history else [],
        }

    def learn_from_text(self, text: str):
        """从文本中学习"""
        # 分句学习
        sentences = re.split(r'[。！？；\n]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            self.learn(sentence)

    def query(self, question: str) -> Dict:
        """查询元认知知识"""
        results = {
            'knowledge': [],
            'gaps': [],
            'strategies': [],
            'self_assessment': self.self_assessment,
        }

        # 搜索相关知识
        for key, item in self.knowledge.items():
            if any(word in item.content for word in question.split()):
                results['knowledge'].append({
                    'content': item.content,
                    'confidence': item.confidence,
                    'source': item.source,
                })

        # 相关知识空白
        for gap in self.knowledge_gaps:
            if gap.topic in question:
                results['gaps'].append({
                    'topic': gap.topic,
                    'importance': gap.importance,
                })

        # 学习策略建议
        results['strategies'] = [
            {'name': s.name, 'description': s.description, 'effectiveness': s.effectiveness}
            for s in self.strategies.values()
        ]

        return results

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'total_knowledge': len(self.knowledge),
            'gaps_remaining': len(self.knowledge_gaps),
            'curiosity_queue_size': len(self.curiosity_queue),
        }


def test_metacognition():
    """测试元认知"""
    print("=" * 70)
    print("元认知层测试")
    print("=" * 70)

    meta = Metacognition()

    # 测试文本
    test_texts = [
        "人工智能是计算机科学的一个分支。",
        "Python是一种编程语言。",
        "太阳是太阳系的中心。",
        "水在100度沸腾。",
        "人工智能是一种编程语言。",  # 故意矛盾
    ]

    with open('metacognition_test.txt', 'w', encoding='utf-8') as f:
        f.write('元认知层测试\n')
        f.write('=' * 70 + '\n')

        for text in test_texts:
            f.write(f'\n输入: {text}\n')
            result = meta.learn(text)

            f.write(f'  理解度: {result["confidence"]:.2f}\n')
            f.write(f'  已理解: {result["understood"]}\n')

            if result['contradictions']:
                f.write(f'  矛盾: {result["contradictions"]}\n')

            if result['gaps_found']:
                f.write(f'  知识空白: {len(result["gaps_found"])}个\n')

            if result['strategy_used']:
                f.write(f'  使用策略: {result["strategy_used"]}\n')

        # 自我评估
        f.write('\n' + '=' * 70 + '\n')
        f.write('自我评估\n')
        f.write('=' * 70 + '\n')

        assessment = meta.self_evaluate()
        for k, v in assessment.items():
            f.write(f'  {k}: {v:.2f}\n')

        # 好奇心问题
        f.write('\n' + '=' * 70 + '\n')
        f.write('好奇心问题\n')
        f.write('=' * 70 + '\n')

        for _ in range(3):
            question = meta.ask_question()
            if question:
                f.write(f'  - {question}\n')

        # 学习报告
        f.write('\n' + '=' * 70 + '\n')
        f.write('学习报告\n')
        f.write('=' * 70 + '\n')

        report = meta.get_learning_report()
        f.write(f'  总知识: {report["total_knowledge"]}\n')
        f.write(f'  知识空白: {report["knowledge_gaps"]}\n')
        f.write(f'  自我评估: {report["self_assessment"]}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = meta.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to metacognition_test.txt')


if __name__ == '__main__':
    test_metacognition()
