"""增强学习系统 — 提升知识提取质量和效率

改进点：
1. 更精确的三元组提取（细粒度关系）
2. 实体属性提取（日期、数字、位置）
3. 层次关系（is-a、part-of）
4. 批处理提升性能
5. 质量过滤去除低质量三元组
6. 实体链接（跨文档关联实体）

运行方式：
    python training/enhanced_learning.py
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Entity:
    """实体类"""

    def __init__(self, name: str):
        self.name = name
        self.attributes: Dict[str, List[str]] = defaultdict(list)  # 属性名 -> 值列表
        self.relations: List[Tuple[str, str]] = []  # (关系, 对象)
        self.mentions: List[str] = []  # 出现的文档
        self.aliases: Set[str] = set()  # 别名
        self.type: Optional[str] = None  # 实体类型

    def add_attribute(self, attr_name: str, attr_value: str):
        """添加属性"""
        if attr_value not in self.attributes[attr_name]:
            self.attributes[attr_name].append(attr_value)

    def add_relation(self, relation: str, obj: str):
        """添加关系"""
        self.relations.append((relation, obj))

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.type,
            'attributes': dict(self.attributes),
            'relations': self.relations[:100],  # 限制大小
            'aliases': list(self.aliases),
            'mention_count': len(self.mentions),
        }


class EnhancedKnowledgeGraph:
    """增强知识图谱"""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.triples: List[Tuple[str, str, str]] = []
        self.triple_index: Dict[str, List[int]] = defaultdict(list)  # 关系 -> 三元组索引

        # 统计
        self.relation_stats = Counter()
        self.entity_type_stats = Counter()

    def add_entity(self, name: str) -> Entity:
        """添加或获取实体"""
        if name not in self.entities:
            self.entities[name] = Entity(name)
        return self.entities[name]

    def add_triple(self, subject: str, relation: str, obj: str, confidence: float = 1.0):
        """添加三元组"""
        # 过滤低质量三元组
        if not self._validate_triple(subject, relation, obj):
            return

        idx = len(self.triples)
        self.triples.append((subject, relation, obj))
        self.triple_index[relation].append(idx)
        self.relation_stats[relation] += 1

        # 更新实体
        subj_entity = self.add_entity(subject)
        obj_entity = self.add_entity(obj)
        subj_entity.add_relation(relation, obj)

    def add_attribute(self, entity_name: str, attr_name: str, attr_value: str):
        """添加实体属性"""
        entity = self.add_entity(entity_name)
        entity.add_attribute(attr_name, attr_value)

    def _validate_triple(self, subject: str, relation: str, obj: str) -> bool:
        """验证三元组质量"""
        # 过滤太短的
        if len(subject) < 2 or len(obj) < 2:
            return False

        # 过滤太长的（可能是截断错误）
        if len(subject) > 20 or len(obj) > 50:
            return False

        # 过滤纯标点
        if re.match(r'^[\W\s]+$', subject) or re.match(r'^[\W\s]+$', obj):
            return False

        # 过滤"相关内容"关系（信息量太低）
        if relation == '相关内容':
            return False

        return True

    def query(self, question: str, top_k: int = 10) -> Dict:
        """查询知识"""
        keywords = self._extract_keywords(question)

        # 过滤通用词
        generic_words = set('什么 是 有 在 于 的 了 和 与 或')
        filtered_keywords = [kw for kw in keywords if kw not in generic_words and len(kw) >= 2]
        filtered_keywords.sort(key=len, reverse=True)

        # 搜索三元组
        results = []
        seen = set()

        for keyword in filtered_keywords[:5]:
            # 精确匹配主语
            for entity_name, entity in self.entities.items():
                if keyword == entity_name or keyword in entity_name:
                    for relation, obj in entity.relations:
                        key = f"{entity_name}|{relation}|{obj}"
                        if key not in seen:
                            seen.add(key)
                            results.append({
                                'subject': entity_name,
                                'relation': relation,
                                'object': obj,
                                'score': 1.0 if keyword == entity_name else 0.9,
                                'match_type': 'exact' if keyword == entity_name else 'partial',
                                'keyword': keyword,
                            })

                    # 添加属性信息
                    for attr_name, attr_values in entity.attributes.items():
                        for attr_value in attr_values[:3]:
                            results.append({
                                'subject': entity_name,
                                'relation': f'属性:{attr_name}',
                                'object': attr_value,
                                'score': 0.8,
                                'match_type': 'attribute',
                                'keyword': keyword,
                            })

        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)

        return {
            'keywords': filtered_keywords[:10],
            'results': results[:top_k],
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 英文单词
        en_words = re.findall(r'[a-zA-Z]+', text.lower())
        # 中文词（2-4字）- 滑动窗口
        zh_words = []
        for i in range(len(text)):
            for j in range(i+2, min(i+5, len(text)+1)):
                word = text[i:j]
                if re.match(r'^[一-鿿]+$', word):
                    zh_words.append(word)
        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        all_words = en_words + zh_words
        return [w for w in all_words if w not in stopwords and len(w) >= 2]

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
            'relation_distribution': dict(self.relation_stats.most_common(20)),
            'entity_type_distribution': dict(self.entity_type_stats.most_common(10)),
        }


class EnhancedTextExtractor:
    """增强文本提取器"""

    def __init__(self):
        # 细粒度关系模式
        self.relation_patterns = [
            # 基本关系
            (r'(.{2,10})是(.{2,30})', '是'),
            (r'(.{2,10})属于(.{2,20})', '属于'),
            (r'(.{2,10})包括(.{2,30})', '包括'),
            (r'(.{2,10})位于(.{2,20})', '位于'),

            # 身份关系
            (r'(.{2,10})又称(.{2,15})', '又称'),
            (r'(.{2,10})又叫(.{2,15})', '又叫'),
            (r'(.{2,10})叫做(.{2,15})', '叫做'),
            (r'(.{2,10})称为(.{2,15})', '称为'),
            (r'(.{2,10})指的是(.{2,30})', '指的是'),
            (r'(.{2,10})是一种(.{2,20})', '是一种'),
            (r'(.{2,10})是(.{2,10})的一种', '是...的一种'),

            # 功能关系
            (r'(.{2,10})使用(.{2,20})', '使用'),
            (r'(.{2,10})用于(.{2,20})', '用于'),
            (r'(.{2,10})利用(.{2,20})', '利用'),
            (r'(.{2,10})采用(.{2,20})', '采用'),

            # 因果关系
            (r'(.{2,10})产生(.{2,20})', '产生'),
            (r'(.{2,10})导致(.{2,20})', '导致'),
            (r'(.{2,10})引起(.{2,20})', '引起'),
            (r'(.{2,10})造成(.{2,20})', '造成'),
            (r'(.{2,10})引发(.{2,20})', '引发'),

            # 时间关系
            (r'(.{2,10})发现于(\d{4}年?)', '发现于'),
            (r'(.{2,10})发明于(\d{4}年?)', '发明于'),
            (r'(.{2,10})成立于(\d{4}年?)', '成立于'),
            (r'(.{2,10})创建于(\d{4}年?)', '创建于'),
            (r'(.{2,10})出生于(\d{4}年?)', '出生于'),
            (r'(.{2,10})诞生于(\d{4}年?)', '诞生于'),
            (r'(.{2,10})出版于(\d{4}年?)', '出版于'),
            (r'(.{2,10})发表于(\d{4}年?)', '发表于'),

            # 空间关系
            (r'(.{2,10})坐落在(.{2,20})', '坐落在'),
            (r'(.{2,10})坐落于(.{2,20})', '坐落于'),
            (r'(.{2,10})建立在(.{2,20})', '建立在'),

            # 创造关系
            (r'(.{2,10})发明了(.{2,20})', '发明了'),
            (r'(.{2,10})发现了(.{2,20})', '发现了'),
            (r'(.{2,10})创造了(.{2,20})', '创造了'),
            (r'(.{2,10})提出了(.{2,20})', '提出了'),
            (r'(.{2,10})开发了(.{2,20})', '开发了'),
            (r'(.{2,10})设计了(.{2,20})', '设计了'),
            (r'(.{2,10})编写了(.{2,20})', '编写了'),

            # 包含关系
            (r'(.{2,10})包含(.{2,20})', '包含'),
            (r'(.{2,10})含有(.{2,20})', '含有'),
            (r'(.{2,10})拥有(.{2,20})', '拥有'),
            (r'(.{2,10})具有(.{2,20})', '具有'),

            # 比较关系
            (r'(.{2,10})类似于(.{2,20})', '类似于'),
            (r'(.{2,10})区别于(.{2,20})', '区别于'),
            (r'(.{2,10})不同于(.{2,20})', '不同于'),
        ]

        # 属性模式
        self.attribute_patterns = [
            # 数值属性
            (r'(.{2,10})的(\w+)为(\d+(?:\.\d+)?)', '数值'),
            (r'(.{2,10})的(\w+)是(\d+(?:\.\d+)?)', '数值'),
            (r'(.{2,10})(\w+)为(\d+(?:\.\d+)?)', '数值'),
            (r'(.{2,10})(\w+)是(\d+(?:\.\d+)?)', '数值'),

            # 时间属性
            (r'(\d{4})年(\d{1,2})月(\d{1,2})日', '日期'),
            (r'(\d{4})年(\d{1,2})月', '年月'),
            (r'(\d{4})年', '年份'),

            # 地点属性
            (r'位于(.{2,15})', '地点'),
            (r'坐落在(.{2,15})', '地点'),
        ]

        # 实体类型模式
        self.entity_type_patterns = [
            (r'(.{2,6})(?:是|属于)(?:一种|一个|一位|一名)?(.{2,6})(?:学科|理论|技术|语言|工具|设备|人物|公司|组织|国家|城市|地点)', '类型'),
        ]

    def extract_triples(self, text: str, title: str = '') -> List[Tuple[str, str, str, float]]:
        """提取三元组，返回 (主语, 关系, 宾语, 置信度)"""
        triples = []

        # 分句
        sentences = re.split(r'[。！？；\n]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # 提取关系三元组
            for pattern, relation in self.relation_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    subject = match[0].strip()
                    obj = match[1].strip()

                    # 验证长度
                    if 2 <= len(subject) <= 15 and 2 <= len(obj) <= 30:
                        # 计算置信度
                        confidence = self._calculate_confidence(subject, relation, obj)
                        triples.append((subject, relation, obj, confidence))

        return triples

    def extract_attributes(self, text: str, entity_name: str) -> List[Tuple[str, str, str]]:
        """提取实体属性，返回 (实体名, 属性名, 属性值)"""
        attributes = []

        # 数值属性
        num_patterns = [
            (r'(\w+)为(\d+(?:\.\d+)?)\s*(?:米|千米|公里|公斤|千克|吨|个|只|条|块|元|美元|%)', '数值'),
            (r'(\w+)是(\d+(?:\.\d+)?)\s*(?:米|千米|公里|公斤|千克|吨|个|只|条|块|元|美元|%)', '数值'),
        ]

        for pattern, attr_type in num_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                attr_name = match[0]
                attr_value = match[1]
                if len(attr_name) >= 2 and len(attr_name) <= 10:
                    attributes.append((entity_name, attr_name, attr_value))

        # 时间属性
        year_matches = re.findall(r'(\d{4})年', text)
        for year in year_matches[:3]:  # 最多取3个年份
            attributes.append((entity_name, '年份', year))

        return attributes

    def _calculate_confidence(self, subject: str, relation: str, obj: str) -> float:
        """计算三元组置信度"""
        confidence = 0.5

        # 关系越具体，置信度越高
        specific_relations = ['位于', '属于', '发明于', '成立于', '出生于', '发现于']
        if relation in specific_relations:
            confidence += 0.3

        # 主语和宾语长度适中，置信度更高
        if 3 <= len(subject) <= 8 and 3 <= len(obj) <= 15:
            confidence += 0.1

        # 包含数字的宾语（可能是具体事实）
        if re.search(r'\d', obj):
            confidence += 0.1

        return min(1.0, confidence)


def stream_jsonl(filepath, max_lines=None):
    """流式读取 JSONL 文件"""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield data
                count += 1
                if max_lines and count >= max_lines:
                    break
            except:
                continue


def process_article(article_data: Dict, dataset_name: str, extractor: EnhancedTextExtractor) -> Dict:
    """处理单篇文章"""
    result = {
        'triples': [],
        'attributes': [],
        'entities': set(),
    }

    try:
        # 根据数据集类型提取文本
        if dataset_name == 'wiki':
            title = article_data.get('title', '')
            text = article_data.get('text', '')
        elif dataset_name == 'baike':
            title = article_data.get('title', '')
            desc = article_data.get('desc', '')
            answer = article_data.get('answer', '')
            text = f"{title}\n{desc}\n{answer}"
        elif dataset_name == 'news':
            title = article_data.get('title', '')
            content = article_data.get('content', '')
            keywords = article_data.get('keywords', '')
            text = f"{title}\n{keywords}\n{content}"
        elif dataset_name == 'translation':
            chinese = article_data.get('chinese', '')
            english = article_data.get('english', '')
            title = chinese[:20] if chinese else ''
            text = chinese
        elif dataset_name == 'webtext':
            title = article_data.get('title', '')
            desc = article_data.get('desc', '')
            content = article_data.get('content', '')
            text = f"{title}\n{desc}\n{content}"
        else:
            return result

        if len(text) < 10:
            return result

        # 提取三元组
        triples = extractor.extract_triples(text, title)
        for s, r, o, conf in triples:
            result['triples'].append((s, r, o))
            result['entities'].add(s)
            result['entities'].add(o)

        # 提取属性
        if title:
            attributes = extractor.extract_attributes(text, title)
            result['attributes'].extend(attributes)
            for e, a, v in attributes:
                result['entities'].add(e)

    except Exception as e:
        pass

    return result


def learn_dataset_batch(graph: EnhancedKnowledgeGraph, dataset_name: str, data_dir: str,
                        extractor: EnhancedTextExtractor, max_lines: int = None, batch_size: int = 1000):
    """批量学习数据集"""
    print(f"\n学习 {dataset_name}...")
    count = 0
    triple_count = 0
    attr_count = 0

    try:
        if dataset_name == 'wiki':
            wiki_dir = os.path.join(data_dir, 'wiki', 'wiki_zh')
            if not os.path.exists(wiki_dir):
                print(f"  目录不存在: {wiki_dir}")
                return 0
            for root, dirs, files in os.walk(wiki_dir):
                for fname in files:
                    if fname.startswith('.'):
                        continue
                    filepath = os.path.join(root, fname)
                    for data in stream_jsonl(filepath, max_lines=max_lines - count if max_lines else None):
                        result = process_article(data, dataset_name, extractor)
                        for s, r, o in result['triples']:
                            graph.add_triple(s, r, o)
                            triple_count += 1
                        for e, a, v in result['attributes']:
                            graph.add_attribute(e, a, v)
                            attr_count += 1
                        count += 1
                        if count % 10000 == 0:
                            print(f"  已学习: {count} 条, 三元组: {triple_count}, 属性: {attr_count}")
                        if max_lines and count >= max_lines:
                            break
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'baike':
            for split in ['baike_qa_valid.json', 'baike_qa_train.json']:
                filepath = os.path.join(data_dir, 'baike', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    result = process_article(data, dataset_name, extractor)
                    for s, r, o in result['triples']:
                        graph.add_triple(s, r, o)
                        triple_count += 1
                    for e, a, v in result['attributes']:
                        graph.add_attribute(e, a, v)
                        attr_count += 1
                    count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条, 三元组: {triple_count}, 属性: {attr_count}")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'news':
            for split in ['news2016zh_valid.json', 'news2016zh_train.json']:
                filepath = os.path.join(data_dir, 'news', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    result = process_article(data, dataset_name, extractor)
                    for s, r, o in result['triples']:
                        graph.add_triple(s, r, o)
                        triple_count += 1
                    for e, a, v in result['attributes']:
                        graph.add_attribute(e, a, v)
                        attr_count += 1
                    count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条, 三元组: {triple_count}, 属性: {attr_count}")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'translation':
            for split in ['translation2019zh_valid.json', 'translation2019zh_train.json']:
                filepath = os.path.join(data_dir, 'translation', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    result = process_article(data, dataset_name, extractor)
                    for s, r, o in result['triples']:
                        graph.add_triple(s, r, o)
                        triple_count += 1
                    for e, a, v in result['attributes']:
                        graph.add_attribute(e, a, v)
                        attr_count += 1
                    count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条, 三元组: {triple_count}, 属性: {attr_count}")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'webtext':
            for split in ['web_text_zh_valid.json', 'web_text_zh_testa.json', 'web_text_zh_train.json']:
                filepath = os.path.join(data_dir, 'webtext', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    result = process_article(data, dataset_name, extractor)
                    for s, r, o in result['triples']:
                        graph.add_triple(s, r, o)
                        triple_count += 1
                    for e, a, v in result['attributes']:
                        graph.add_attribute(e, a, v)
                        attr_count += 1
                    count += 1
                    if count % 10000 == 0:
                        print(f"  已学习: {count} 条, 三元组: {triple_count}, 属性: {attr_count}")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

    except Exception as e:
        print(f"  错误: {e}")

    print(f"  {dataset_name} 完成: {count} 条, 三元组: {triple_count}, 属性: {attr_count}")
    return count


def main():
    print("=" * 70)
    print("增强学习系统 — 提升知识提取质量和效率")
    print("=" * 70)

    data_dir = 'data/extracted'
    output_dir = Path('data/knowledge/enhanced')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化
    graph = EnhancedKnowledgeGraph()
    extractor = EnhancedTextExtractor()

    # 每个数据集学习的数量（测试用）
    max_per_dataset = 50000  # 先学5万条测试

    start_time = time.time()

    # 学习数据集
    datasets = ['wiki', 'baike', 'news', 'translation', 'webtext']

    for dataset in datasets:
        learn_dataset_batch(graph, dataset, data_dir, extractor, max_per_dataset)

    elapsed = time.time() - start_time

    # 保存结果
    print("\n保存结果...")
    save_data = {
        'entities': {name: entity.to_dict() for name, entity in graph.entities.items()},
        'triples': graph.triples,
        'stats': graph.get_stats(),
    }

    output_path = output_dir / 'enhanced_knowledge.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    # 统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)
    stats = graph.get_stats()
    print(f"  耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  三元组: {stats['total_triples']:,}")
    print(f"  实体数: {stats['total_entities']:,}")

    print("\n  关系类型分布:")
    for rel, count in list(stats['relation_distribution'].items())[:10]:
        print(f"    {rel}: {count:,}")

    # 测试查询
    print("\n" + "=" * 70)
    print("测试查询")
    print("=" * 70)

    test_questions = [
        "什么是人工智能",
        "中国在哪里",
        "牛顿发现了什么",
        "Python是什么语言",
    ]

    for q in test_questions:
        print(f"\n问: {q}")
        result = graph.query(q, top_k=5)
        print(f"  关键词: {result['keywords'][:5]}")
        print(f"  结果: {len(result['results'])} 个")
        for r in result['results'][:3]:
            print(f"    → {r['subject']} {r['relation']} {r['object']}")


if __name__ == '__main__':
    main()
