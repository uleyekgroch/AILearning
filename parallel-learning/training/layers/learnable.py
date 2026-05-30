"""可学习的模式识别层

从示例中学习新的模式，而不是只依赖硬编码的正则表达式。

核心能力：
1. 模式学习 — 从示例中学习新的语义/因果模式
2. 模式匹配 — 使用学习到的模式识别新文本
3. 模式泛化 — 从具体示例泛化到一般模式

运行方式：
    python training/layers/learnable.py
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class LearnedPattern:
    """学习到的模式"""
    pattern: str           # 正则模式
    relation: str          # 关系类型
    examples: List[str]    # 学习示例
    confidence: float      # 置信度
    usage_count: int = 0   # 使用次数
    success_count: int = 0 # 成功次数

    def update_confidence(self, success: bool):
        """更新置信度"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.confidence = self.success_count / self.usage_count if self.usage_count > 0 else 0.5


class LearnablePatternSystem:
    """可学习的模式识别系统

    核心能力：
    - 从示例中学习新模式
    - 使用学习到的模式识别新文本
    - 模式泛化
    """

    def __init__(self):
        # 学习到的模式库
        self.learned_patterns: Dict[str, List[LearnedPattern]] = defaultdict(list)

        # 实体嵌入（简单的共现向量）
        self.entity_embeddings: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # 关系嵌入
        self.relation_embeddings: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # 统计
        self.stats = {
            'patterns_learned': 0,
            'patterns_applied': 0,
            'entities_tracked': 0,
        }

    def learn_from_example(self, text: str, subject: str, relation: str, obj: str):
        """从示例中学习模式"""
        # 生成模式
        pattern = self._generate_pattern(text, subject, obj)

        # 检查是否已有类似模式
        existing = self._find_similar_pattern(relation, pattern)
        if existing:
            # 更新现有模式
            existing.examples.append(text)
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            # 创建新模式
            new_pattern = LearnedPattern(
                pattern=pattern,
                relation=relation,
                examples=[text],
                confidence=0.5,
            )
            self.learned_patterns[relation].append(new_pattern)
            self.stats['patterns_learned'] += 1

        # 更新实体嵌入
        self._update_entity_embedding(subject, relation, obj)
        self._update_relation_embedding(relation, subject, obj)

    def _generate_pattern(self, text: str, subject: str, obj: str) -> str:
        """从示例生成正则模式"""
        # 转义特殊字符
        subject_escaped = re.escape(subject)
        obj_escaped = re.escape(obj)

        # 替换为通配符
        pattern = text.replace(subject, '(.{2,10}?)', 1)
        pattern = pattern.replace(obj, '(.{2,10}?)', 1)

        # 转义其他特殊字符
        pattern = re.escape(pattern)
        pattern = pattern.replace('\\(\\.\\{2,10\\}\\?\\)', '(.{2,10}?)')

        return pattern

    def _find_similar_pattern(self, relation: str, pattern: str) -> Optional[LearnedPattern]:
        """查找类似的模式"""
        for existing in self.learned_patterns.get(relation, []):
            # 简单相似度：检查模式是否相似
            if self._patterns_similar(existing.pattern, pattern):
                return existing
        return None

    def _patterns_similar(self, pattern1: str, pattern2: str) -> bool:
        """判断两个模式是否相似"""
        # 简单实现：检查模式长度和结构
        if len(pattern1) != len(pattern2):
            return False

        # 检查有多少字符相同
        same_count = sum(1 for a, b in zip(pattern1, pattern2) if a == b)
        similarity = same_count / len(pattern1) if len(pattern1) > 0 else 0

        return similarity > 0.8

    def _update_entity_embedding(self, subject: str, relation: str, obj: str):
        """更新实体嵌入"""
        # 实体与关系共现
        self.entity_embeddings[subject][relation] += 1
        self.entity_embeddings[obj][relation] += 1

        # 实体间共现
        self.entity_embeddings[subject][f"related_to:{obj}"] += 1
        self.entity_embeddings[obj][f"related_to:{subject}"] += 1

        self.stats['entities_tracked'] = len(self.entity_embeddings)

    def _update_relation_embedding(self, relation: str, subject: str, obj: str):
        """更新关系嵌入"""
        self.relation_embeddings[relation][subject] += 1
        self.relation_embeddings[relation][obj] += 1

    def extract_with_learned_patterns(self, text: str) -> List[Tuple[str, str, str]]:
        """使用学习到的模式提取三元组"""
        results = []

        for relation, patterns in self.learned_patterns.items():
            for pattern_obj in patterns:
                matches = re.finditer(pattern_obj.pattern, text)
                for match in matches:
                    if len(match.groups()) >= 2:
                        subject = match.group(1).strip()
                        obj = match.group(2).strip()

                        if len(subject) >= 2 and len(obj) >= 2:
                            results.append((subject, relation, obj))
                            pattern_obj.usage_count += 1
                            self.stats['patterns_applied'] += 1

        return results

    def find_similar_entities(self, entity: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """查找相似实体"""
        if entity not in self.entity_embeddings:
            return []

        # 计算与其他实体的相似度
        similarities = []
        entity_vec = self.entity_embeddings[entity]

        for other_entity, other_vec in self.entity_embeddings.items():
            if other_entity == entity:
                continue

            # 计算余弦相似度
            common_keys = set(entity_vec.keys()) & set(other_vec.keys())
            if not common_keys:
                continue

            dot_product = sum(entity_vec[k] * other_vec[k] for k in common_keys)
            norm1 = sum(v ** 2 for v in entity_vec.values()) ** 0.5
            norm2 = sum(v ** 2 for v in other_vec.values()) ** 0.5

            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                similarities.append((other_entity, similarity))

        # 排序返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def query(self, question: str) -> Dict:
        """查询学习到的模式"""
        # 提取关键词
        keywords = self._extract_keywords(question)

        results = {
            'patterns': [],
            'similar_entities': [],
        }

        # 搜索相关模式
        for keyword in keywords:
            for relation, patterns in self.learned_patterns.items():
                for pattern_obj in patterns:
                    if any(keyword in ex for ex in pattern_obj.examples):
                        results['patterns'].append({
                            'relation': relation,
                            'pattern': pattern_obj.pattern,
                            'confidence': pattern_obj.confidence,
                            'examples': pattern_obj.examples[:3],
                        })

        # 搜索相似实体
        for keyword in keywords:
            similar = self.find_similar_entities(keyword)
            for entity, similarity in similar:
                results['similar_entities'].append({
                    'entity': entity,
                    'similarity': similarity,
                })

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 中文关键词
        zh_keywords = re.findall(r'[一-鿿]{2,6}', text)

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        keywords = [k for k in zh_keywords if k not in stopwords and len(k) >= 2]

        return keywords

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'total_patterns': sum(len(ps) for ps in self.learned_patterns.values()),
            'relations_covered': list(self.learned_patterns.keys()),
        }


def test_learnable_patterns():
    """测试可学习的模式识别"""
    print("=" * 70)
    print("可学习的模式识别测试")
    print("=" * 70)

    system = LearnablePatternSystem()

    # 学习示例
    examples = [
        ('人工智能是计算机科学的一个分支', '人工智能', '是', '计算机科学的一个分支'),
        ('Python是一种编程语言', 'Python', '是', '编程语言'),
        ('牛顿发现了万有引力定律', '牛顿', '发现', '万有引力定律'),
        ('爱因斯坦提出了相对论', '爱因斯坦', '提出', '相对论'),
    ]

    with open('learnable_test.txt', 'w', encoding='utf-8') as f:
        f.write('可学习的模式识别测试\n')
        f.write('=' * 70 + '\n\n')

        # 学习
        for text, subject, relation, obj in examples:
            f.write(f'学习: {text}\n')
            f.write(f'  主语: {subject}, 关系: {relation}, 宾语: {obj}\n')
            system.learn_from_example(text, subject, relation, obj)

        # 显示学习到的模式
        f.write('\n' + '=' * 70 + '\n')
        f.write('学习到的模式\n')
        f.write('=' * 70 + '\n')

        for relation, patterns in system.learned_patterns.items():
            f.write(f'\n关系: {relation}\n')
            for pattern in patterns:
                f.write(f'  模式: {pattern.pattern}\n')
                f.write(f'  置信度: {pattern.confidence:.2f}\n')
                f.write(f'  示例数: {len(pattern.examples)}\n')

        # 测试模式应用
        f.write('\n' + '=' * 70 + '\n')
        f.write('模式应用测试\n')
        f.write('=' * 70 + '\n')

        test_texts = [
            '深度学习是机器学习的一个分支',
            '居里夫人发现了镭',
        ]

        for text in test_texts:
            f.write(f'\n输入: {text}\n')
            results = system.extract_with_learned_patterns(text)
            for s, r, o in results:
                f.write(f'  提取: ({s}, {r}, {o})\n')

        # 测试相似实体
        f.write('\n' + '=' * 70 + '\n')
        f.write('相似实体测试\n')
        f.write('=' * 70 + '\n')

        test_entities = ['人工智能', 'Python']
        for entity in test_entities:
            f.write(f'\n实体: {entity}\n')
            similar = system.find_similar_entities(entity)
            for e, s in similar:
                f.write(f'  {e}: {s:.3f}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = system.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to learnable_test.txt')


if __name__ == '__main__':
    test_learnable_patterns()
