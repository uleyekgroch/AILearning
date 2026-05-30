"""多文档推理层

跨文档整合知识，解决知识冲突。

核心能力：
1. 知识整合 — 从多个文档中整合知识
2. 冲突检测 — 检测不同文档中的矛盾
3. 冲突解决 — 解决知识冲突
4. 知识融合 — 融合互补的知识

运行方式：
    python training/layers/multi_document.py
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class KnowledgeSource:
    """知识来源"""
    content: str           # 内容
    source: str            # 来源（文档名）
    confidence: float      # 置信度
    timestamp: float       # 时间戳


@dataclass
class Conflict:
    """知识冲突"""
    topic: str             # 主题
    sources: List[KnowledgeSource]  # 冲突的来源
    resolution: Optional[str] = None  # 解决方案


class MultiDocumentReasoning:
    """多文档推理层

    核心能力：
    - 跨文档整合知识
    - 检测和解决冲突
    - 知识融合
    """

    def __init__(self):
        # 知识库：主题 -> 来源列表
        self.knowledge: Dict[str, List[KnowledgeSource]] = defaultdict(list)

        # 冲突库
        self.conflicts: List[Conflict] = []

        # 已解决的冲突
        self.resolved_conflicts: List[Conflict] = []

        # 统计
        self.stats = {
            'documents_processed': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
        }

    def add_knowledge(self, topic: str, content: str, source: str, confidence: float = 0.8):
        """添加知识"""
        import time

        knowledge_source = KnowledgeSource(
            content=content,
            source=source,
            confidence=confidence,
            timestamp=time.time(),
        )

        self.knowledge[topic].append(knowledge_source)

        # 检测冲突
        self._detect_conflicts(topic)

    def _detect_conflicts(self, topic: str):
        """检测冲突"""
        sources = self.knowledge[topic]
        if len(sources) < 2:
            return

        # 比较所有来源
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                if self._are_conflicting(source1.content, source2.content):
                    # 创建冲突记录
                    conflict = Conflict(
                        topic=topic,
                        sources=[source1, source2],
                    )
                    self.conflicts.append(conflict)
                    self.stats['conflicts_detected'] += 1

    def _are_conflicting(self, content1: str, content2: str) -> bool:
        """判断两个内容是否冲突"""
        # 1. 检查否定词冲突
        negations = ['不', '没', '无', '非', '否']

        for neg in negations:
            if neg in content1 and neg not in content2:
                # 检查是否是同一主题
                if self._extract_main_subject(content1) == self._extract_main_subject(content2):
                    return True
            if neg in content2 and neg not in content1:
                if self._extract_main_subject(content1) == self._extract_main_subject(content2):
                    return True

        # 2. 检查"是"关系冲突（如"X是Y" vs "X是Z"）
        pattern = r'(.{2,10})是(.{2,20})'
        match1 = re.search(pattern, content1)
        match2 = re.search(pattern, content2)

        if match1 and match2:
            # 如果主语相同，宾语不同，则冲突
            if match1.group(1).strip() == match2.group(1).strip():
                if match1.group(2).strip() != match2.group(2).strip():
                    return True

        return False

    def _extract_main_subject(self, content: str) -> str:
        """提取主要内容"""
        # 提取前10个字符作为主题
        return content[:10]

    def resolve_conflict(self, conflict: Conflict, resolution: str):
        """解决冲突"""
        conflict.resolution = resolution
        self.resolved_conflicts.append(conflict)
        self.stats['conflicts_resolved'] += 1

    def auto_resolve_conflicts(self):
        """自动解决冲突"""
        for conflict in self.conflicts:
            if conflict.resolution:
                continue

            # 策略1：选择置信度最高的
            best_source = max(conflict.sources, key=lambda x: x.confidence)
            conflict.resolution = f"采用最高置信度来源: {best_source.source}"

            # 策略2：如果置信度相同，选择最新的
            if len(set(s.confidence for s in conflict.sources)) == 1:
                latest_source = max(conflict.sources, key=lambda x: x.timestamp)
                conflict.resolution = f"采用最新来源: {latest_source.source}"

            self.resolved_conflicts.append(conflict)
            self.stats['conflicts_resolved'] += 1

    def get_integrated_knowledge(self, topic: str) -> Dict:
        """获取整合后的知识"""
        sources = self.knowledge.get(topic, [])

        # 检查是否有冲突
        conflicts = [c for c in self.conflicts if c.topic == topic]
        resolved = [c for c in self.resolved_conflicts if c.topic == topic]

        # 整合知识
        integrated = {
            'topic': topic,
            'sources': [{
                'content': s.content,
                'source': s.source,
                'confidence': s.confidence,
            } for s in sources],
            'conflicts': [{
                'sources': [s.source for s in c.sources],
                'resolution': c.resolution,
            } for c in conflicts],
            'resolved': len(resolved),
            'unresolved': len(conflicts) - len(resolved),
        }

        return integrated

    def query(self, question: str) -> Dict:
        """查询多文档知识"""
        # 提取主题
        topics = self._extract_topics(question)

        results = {
            'topics': [],
            'integrated_knowledge': [],
            'conflicts': [],
        }

        for topic in topics:
            # 获取整合知识
            knowledge = self.get_integrated_knowledge(topic)
            results['topics'].append(topic)
            results['integrated_knowledge'].append(knowledge)

            # 获取冲突
            conflicts = [c for c in self.conflicts if c.topic == topic]
            results['conflicts'].extend([{
                'topic': c.topic,
                'sources': [s.source for s in c.sources],
                'resolution': c.resolution,
            } for c in conflicts])

        return results

    def _extract_topics(self, text: str) -> List[str]:
        """提取主题"""
        # 中文主题
        zh_topics = re.findall(r'[一-鿿]{2,6}', text)

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        topics = [t for t in zh_topics if t not in stopwords and len(t) >= 2]

        return topics

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'total_topics': len(self.knowledge),
            'total_sources': sum(len(s) for s in self.knowledge.values()),
        }


def test_multi_document_reasoning():
    """测试多文档推理"""
    print("=" * 70)
    print("多文档推理层测试")
    print("=" * 70)

    reasoning = MultiDocumentReasoning()

    # 测试知识添加
    test_knowledge = [
        ('人工智能', '人工智能是计算机科学的一个分支', '文档1', 0.9),
        ('人工智能', '人工智能是一种编程语言', '文档2', 0.6),
        ('Python', 'Python是一种编程语言', '文档1', 0.9),
        ('Python', 'Python是蛇的意思', '文档2', 0.7),
    ]

    with open('multi_document_test.txt', 'w', encoding='utf-8') as f:
        f.write('多文档推理层测试\n')
        f.write('=' * 70 + '\n\n')

        for topic, content, source, confidence in test_knowledge:
            f.write(f'添加知识: {topic} - {content} (来源: {source})\n')
            reasoning.add_knowledge(topic, content, source, confidence)

        # 显示冲突
        f.write('\n' + '=' * 70 + '\n')
        f.write('检测到的冲突\n')
        f.write('=' * 70 + '\n')

        for conflict in reasoning.conflicts:
            f.write(f'\n主题: {conflict.topic}\n')
            for source in conflict.sources:
                f.write(f'  来源: {source.source} - {source.content}\n')

        # 自动解决冲突
        reasoning.auto_resolve_conflicts()

        f.write('\n' + '=' * 70 + '\n')
        f.write('冲突解决\n')
        f.write('=' * 70 + '\n')

        for conflict in reasoning.resolved_conflicts:
            f.write(f'\n主题: {conflict.topic}\n')
            f.write(f'  解决方案: {conflict.resolution}\n')

        # 测试查询
        f.write('\n' + '=' * 70 + '\n')
        f.write('查询测试\n')
        f.write('=' * 70 + '\n')

        test_questions = [
            '什么是人工智能',
            'Python是什么',
        ]

        for q in test_questions:
            f.write(f'\n问: {q}\n')
            result = reasoning.query(q)
            for knowledge in result['integrated_knowledge']:
                f.write(f'  主题: {knowledge["topic"]}\n')
                for source in knowledge['sources']:
                    f.write(f'    - {source["content"]} (置信度: {source["confidence"]})\n')
                if knowledge['conflicts']:
                    f.write(f'    冲突: {len(knowledge["conflicts"])}个\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = reasoning.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to multi_document_test.txt')


if __name__ == '__main__':
    test_multi_document_reasoning()
