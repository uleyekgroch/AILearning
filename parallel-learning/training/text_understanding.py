"""文本理解系统 — 从语料中真正学习知识

核心能力：
1. 解析句子，提取主语-关系-宾语三元组
2. 构建知识图谱
3. 回答问题

人类学习方式：
- 读一句话 → 理解意思 → 记住事实 → 能回答问题
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class KnowledgeGraph:
    """知识图谱 — 存储学到的知识"""

    def __init__(self):
        # 三元组：(主语, 关系, 宾语)
        self.triples: List[Tuple[str, str, str]] = []
        # 实体信息
        self.entities: Dict[str, Dict] = {}
        # 关系索引
        self.subject_index: Dict[str, List[int]] = defaultdict(list)
        self.object_index: Dict[str, List[int]] = defaultdict(list)
        # 文章摘要
        self.summaries: Dict[str, str] = {}

    def add_triple(self, subject: str, relation: str, obj: str):
        """添加三元组"""
        idx = len(self.triples)
        self.triples.append((subject, relation, obj))
        self.subject_index[subject].append(idx)
        self.object_index[obj].append(idx)

        # 更新实体
        for entity in [subject, obj]:
            if entity not in self.entities:
                self.entities[entity] = {
                    'name': entity,
                    'count': 0,
                    'relations': [],
                }
            self.entities[entity]['count'] += 1

    def add_entity_info(self, name: str, info: Dict):
        """添加实体详细信息"""
        if name not in self.entities:
            self.entities[name] = {'name': name, 'count': 0, 'relations': []}
        self.entities[name].update(info)

    def add_summary(self, title: str, summary: str):
        """添加文章摘要"""
        self.summaries[title] = summary

    def query(self, question: str) -> Dict:
        """查询知识图谱"""
        keywords = self._extract_keywords(question)
        results = []

        # 搜索相关三元组
        for keyword in keywords:
            # 作为主语
            for idx in self.subject_index.get(keyword, []):
                s, r, o = self.triples[idx]
                results.append({
                    'subject': s,
                    'relation': r,
                    'object': o,
                    'score': 1.0,
                })

            # 作为宾语
            for idx in self.object_index.get(keyword, []):
                s, r, o = self.triples[idx]
                results.append({
                    'subject': s,
                    'relation': r,
                    'object': o,
                    'score': 0.8,
                })

            # 模糊匹配实体
            for entity_name, entity_info in self.entities.items():
                if keyword in entity_name or entity_name in keyword:
                    results.append({
                        'type': 'entity',
                        'name': entity_name,
                        'info': entity_info,
                        'score': 0.5,
                    })

            # 搜索摘要
            for title, summary in self.summaries.items():
                if keyword in title or keyword in summary:
                    results.append({
                        'type': 'summary',
                        'title': title,
                        'summary': summary[:200],
                        'score': 0.6,
                    })

        # 如果没有结果，尝试搜索所有实体
        if not results:
            for entity_name, entity_info in self.entities.items():
                # 检查问题中是否包含实体名
                if entity_name in question:
                    results.append({
                        'type': 'entity',
                        'name': entity_name,
                        'info': entity_info,
                        'score': 0.7,
                    })
                    # 搜索该实体的三元组
                    for idx in self.subject_index.get(entity_name, []):
                        s, r, o = self.triples[idx]
                        results.append({
                            'subject': s,
                            'relation': r,
                            'object': o,
                            'score': 0.9,
                        })

        # 去重并排序
        seen = set()
        unique_results = []
        for r in results:
            key = str(r)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        return {
            'keywords': keywords,
            'results': unique_results[:10],
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 英文
        en_words = re.findall(r'[a-zA-Z]+', text.lower())

        # 中文：按标点分割后取词
        zh_words = []
        parts = re.split(r'[，。？！；：、\s]+', text)
        for part in parts:
            zh_part = re.findall(r'[一-鿿]+', part)
            for w in zh_part:
                if 2 <= len(w) <= 6:
                    zh_words.append(w)

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        all_words = en_words + zh_words
        return [w for w in all_words if w not in stopwords and len(w) >= 2]

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
            'total_summaries': len(self.summaries),
            'top_entities': sorted(
                self.entities.items(),
                key=lambda x: x[1].get('count', 0),
                reverse=True
            )[:10],
        }


class TextUnderstandingSystem:
    """文本理解系统 — 从语料中学习知识"""

    def __init__(self):
        self.knowledge = KnowledgeGraph()
        self.total_articles = 0
        self.total_triples_extracted = 0

    def learn_from_article(self, title: str, text: str):
        """从一篇文章中学习"""
        self.total_articles += 1

        # 1. 提取摘要（第一段）
        paragraphs = text.split('\n\n')
        summary = paragraphs[0] if paragraphs else text[:200]
        self.knowledge.add_summary(title, summary)

        # 2. 提取实体和关系
        self._extract_knowledge(title, text)

    def _extract_knowledge(self, title: str, text: str):
        """从文本中提取知识"""
        # 分句
        sentences = re.split(r'[。！？；\n]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # 提取三元组
            triples = self._extract_triples(sentence, title)
            for s, r, o in triples:
                self.knowledge.add_triple(s, r, o)
                self.total_triples_extracted += 1

    def _extract_triples(self, sentence: str, context: str) -> List[Tuple[str, str, str]]:
        """从句子中提取三元组"""
        triples = []

        # 模式1：X是Y
        patterns = [
            (r'(.{2,10})是(.{2,20})', '是'),
            (r'(.{2,10})属于(.{2,20})', '属于'),
            (r'(.{2,10})包括(.{2,20})', '包括'),
            (r'(.{2,10})位于(.{2,20})', '位于'),
            (r'(.{2,10})又称(.{2,20})', '又称'),
            (r'(.{2,10})又叫(.{2,20})', '又叫'),
            (r'(.{2,10})叫做(.{2,20})', '叫做'),
            (r'(.{2,10})称为(.{2,20})', '称为'),
            (r'(.{2,10})指的是(.{2,20})', '指的是'),
            (r'(.{2,10})是一种(.{2,20})', '是一种'),
            (r'(.{2,10})有(.{2,20})', '有'),
            (r'(.{2,10})使用(.{2,20})', '使用'),
            (r'(.{2,10})用于(.{2,20})', '用于'),
            (r'(.{2,10})产生(.{2,20})', '产生'),
            (r'(.{2,10})导致(.{2,20})', '导致'),
            (r'(.{2,10})引起(.{2,20})', '引起'),
            (r'(.{2,10})发现于(.{2,20})', '发现于'),
            (r'(.{2,10})发明于(.{2,20})', '发明于'),
            (r'(.{2,10})成立于(.{2,20})', '成立于'),
            (r'(.{2,10})创建于(.{2,20})', '创建于'),
        ]

        for pattern, relation in patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                subject = match[0].strip()
                obj = match[1].strip()

                # 过滤太短或太长的
                if 2 <= len(subject) <= 10 and 2 <= len(obj) <= 20:
                    triples.append((subject, relation, obj))

        # 模式2：X，Y的Z
        pattern2 = re.findall(r'(.{2,6})[，,]是(.{2,10})的(.{2,10})', sentence)
        for match in pattern2:
            triples.append((match[0], '是', f"{match[1]}的{match[2]}"))

        # 模式3：标题相关
        if context and len(triples) == 0:
            # 如果没有提取到关系，至少记录标题和句子的关联
            words = re.findall(r'[一-鿿]{2,4}', sentence)
            if words and context in sentence:
                triples.append((context, '相关内容', words[0]))

        return triples

    def query(self, question: str) -> Dict:
        """回答问题"""
        return self.knowledge.query(question)

    def get_stats(self) -> Dict:
        """获取统计"""
        stats = self.knowledge.get_stats()
        stats['total_articles'] = self.total_articles
        stats['total_triples_extracted'] = self.total_triples_extracted
        return stats


def test_system():
    """测试系统"""
    print("=" * 70)
    print("测试文本理解系统")
    print("=" * 70)

    system = TextUnderstandingSystem()

    # 测试文章
    test_articles = [
        ("重力", "重力是地球对物体的吸引力。重力使物体落向地面。重力的大小与物体的质量成正比。牛顿发现了万有引力定律。"),
        ("Python", "Python是一种编程语言。Python由吉多·范罗苏姆发明。Python用于Web开发、数据分析和人工智能。Python的特点是简洁易读。"),
        ("太阳", "太阳是太阳系的中心天体。太阳是一颗恒星。太阳的表面温度约为5500摄氏度。太阳为地球提供光和热。"),
    ]

    for title, text in test_articles:
        print(f"\n学习: {title}")
        system.learn_from_article(title, text)

    # 测试查询
    print("\n" + "=" * 70)
    print("测试查询")
    print("=" * 70)

    test_questions = [
        "什么是重力？",
        "Python是什么？",
        "太阳是什么？",
        "谁发现了万有引力？",
    ]

    for question in test_questions:
        print(f"\n问: {question}")
        result = system.query(question)
        print(f"  关键词: {result['keywords']}")
        print(f"  结果数: {len(result['results'])}")
        for r in result['results'][:3]:
            if 'subject' in r:
                print(f"    {r['subject']} {r['relation']} {r['object']}")
            elif 'title' in r:
                print(f"    摘要: {r['summary'][:50]}...")

    # 统计
    print("\n" + "=" * 70)
    print("统计")
    print("=" * 70)
    stats = system.get_stats()
    print(f"  文章数: {stats['total_articles']}")
    print(f"  三元组: {stats['total_triples']}")
    print(f"  实体数: {stats['total_entities']}")


if __name__ == '__main__':
    test_system()
