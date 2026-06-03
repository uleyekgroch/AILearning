"""数值理解层（增强版）

理解数值信息，回答"多少"、"多大"、"多重"等问题。

核心能力：
1. 数值提取 — 从文本中提取数值
2. 单位识别 — 识别度量单位
3. 数值关联 — 将数值与实体关联
4. 数值查询 — 回答数值问题
5. 范围查询 — 支持"100-200度"等范围
6. 复合单位 — 支持"千米/小时"等复合单位
7. 数值比较 — 支持"大于"、"小于"等比较
8. 数值推理 — 支持简单数值推理

运行方式：
    python training/layers/numerical.py
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class NumericalFact:
    """数值事实"""
    entity: str           # 实体
    attribute: str        # 属性（温度、重量、长度等）
    value: float          # 数值
    unit: str             # 单位
    context: str          # 上下文
    confidence: float     # 置信度


class NumericalUnderstanding:
    """数值理解层"""

    def __init__(self):
        # 数值事实库
        self.facts: List[NumericalFact] = []

        # 实体-属性索引
        self.entity_index: Dict[str, List[NumericalFact]] = defaultdict(list)
        self.attribute_index: Dict[str, List[NumericalFact]] = defaultdict(list)

        # 单位转换表
        self.unit_conversions = self._load_unit_conversions()

        # 数值模式
        self.numerical_patterns = self._load_numerical_patterns()

        # 统计
        self.stats = {
            'facts_stored': 0,
            'queries_answered': 0,
        }

    def _load_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """加载单位转换表"""
        return {
            '温度': {
                '摄氏度': 1.0,
                '℃': 1.0,
                '华氏度': 0.5556,  # (F-32)*5/9
                '开尔文': 1.0,  # K-273.15
            },
            '长度': {
                '米': 1.0,
                '千米': 1000.0,
                '公里': 1000.0,
                '厘米': 0.01,
                '毫米': 0.001,
                '英里': 1609.34,
                '英尺': 0.3048,
                '英寸': 0.0254,
            },
            '重量': {
                '千克': 1.0,
                '公斤': 1.0,
                '克': 0.001,
                '吨': 1000.0,
                '磅': 0.4536,
                '盎司': 0.02835,
            },
            '时间': {
                '秒': 1.0,
                '分钟': 60.0,
                '小时': 3600.0,
                '天': 86400.0,
                '周': 604800.0,
                '月': 2592000.0,  # 30天
                '年': 31536000.0,  # 365天
            },
            '速度': {
                '米/秒': 1.0,
                '千米/小时': 0.2778,
                '公里/小时': 0.2778,
                '英里/小时': 0.4470,
            },
            '面积': {
                '平方米': 1.0,
                '平方公里': 1000000.0,
                '公顷': 10000.0,
                '亩': 666.67,
            },
            '体积': {
                '立方米': 1.0,
                '升': 0.001,
                '毫升': 0.000001,
                '加仑': 0.003785,
            },
        }

    def _load_numerical_patterns(self) -> List[Tuple[str, str, str]]:
        """加载数值模式

        格式: (模式, 属性类型, 单位)
        """
        return [
            # 温度
            (r'(\d+(?:\.\d+)?)\s*(?:摄氏度|℃|度)', '温度', '摄氏度'),
            (r'(\d+(?:\.\d+)?)\s*(?:华氏度|℉)', '温度', '华氏度'),
            (r'(\d+(?:\.\d+)?)\s*(?:开尔文|K)', '温度', '开尔文'),
            (r'温度(?:为|是|达到|有)\s*(\d+(?:\.\d+)?)', '温度', '摄氏度'),
            (r'(\d+(?:\.\d+)?)\s*度', '温度', '摄氏度'),

            # 长度
            (r'(\d+(?:\.\d+)?)\s*(?:米|m)', '长度', '米'),
            (r'(\d+(?:\.\d+)?)\s*(?:千米|公里|km)', '长度', '千米'),
            (r'(\d+(?:\.\d+)?)\s*(?:厘米|cm)', '长度', '厘米'),
            (r'(\d+(?:\.\d+)?)\s*(?:毫米|mm)', '长度', '毫米'),
            (r'(?:高|长|宽|深)(?:为|是|达)\s*(\d+(?:\.\d+)?)', '长度', '米'),

            # 重量
            (r'(\d+(?:\.\d+)?)\s*(?:千克|公斤|kg)', '重量', '千克'),
            (r'(\d+(?:\.\d+)?)\s*(?:克|g)', '重量', '克'),
            (r'(\d+(?:\.\d+)?)\s*(?:吨|t)', '重量', '吨'),
            (r'(?:重|重量)(?:为|是|达)\s*(\d+(?:\.\d+)?)', '重量', '千克'),

            # 时间
            (r'(\d+(?:\.\d+)?)\s*(?:年)', '时间', '年'),
            (r'(\d+(?:\.\d+)?)\s*(?:月)', '时间', '月'),
            (r'(\d+(?:\.\d+)?)\s*(?:天|日)', '时间', '天'),
            (r'(\d+(?:\.\d+)?)\s*(?:小时|时)', '时间', '小时'),
            (r'(\d+(?:\.\d+)?)\s*(?:分钟|分)', '时间', '分钟'),
            (r'(\d+(?:\.\d+)?)\s*(?:秒)', '时间', '秒'),

            # 数量
            (r'(\d+(?:\.\d+)?)\s*(?:个|只|条|块|颗|张|本|台|套)', '数量', '个'),
            (r'(?:有|共有|总计|达到)\s*(\d+(?:\.\d+)?)', '数量', '个'),

            # 百分比
            (r'(\d+(?:\.\d+)?)\s*(?:%|百分之)', '百分比', '%'),

            # 年份
            (r'(\d{4})\s*年', '年份', '年'),
        ]

    def extract_numerical_facts(self, text: str, entity: str = "") -> List[NumericalFact]:
        """从文本中提取数值事实"""
        facts = []

        # 提取范围数值（如"100-200度"）
        range_pattern = r'(\d+(?:\.\d+)?)\s*[-~到至]\s*(\d+(?:\.\d+)?)\s*(度|米|千克|公里|%)'
        range_matches = re.finditer(range_pattern, text)
        for match in range_matches:
            value1 = float(match.group(1))
            value2 = float(match.group(2))
            unit = match.group(3)

            # 确定属性类型
            attr_type = self._get_attribute_type(unit)

            if not entity:
                entity = self._extract_entity_from_context(text, match.start())

            # 存储范围的平均值
            fact = NumericalFact(
                entity=entity,
                attribute=attr_type,
                value=(value1 + value2) / 2,
                unit=unit,
                context=text[:100],
                confidence=0.7,
            )
            facts.append(fact)

        # 提取复合单位（如"千米/小时"）
        compound_pattern = r'(\d+(?:\.\d+)?)\s*(千米|公里|米)/(小时|分钟|秒)'
        compound_matches = re.finditer(compound_pattern, text)
        for match in compound_matches:
            value = float(match.group(1))
            unit1 = match.group(2)
            unit2 = match.group(3)

            if not entity:
                entity = self._extract_entity_from_context(text, match.start())

            fact = NumericalFact(
                entity=entity,
                attribute='速度',
                value=value,
                unit=f'{unit1}/{unit2}',
                context=text[:100],
                confidence=0.8,
            )
            facts.append(fact)

        # 提取普通数值
        for pattern, attr_type, unit in self.numerical_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                value = float(match.group(1))

                # 尝试从上下文提取实体
                if not entity:
                    entity = self._extract_entity_from_context(text, match.start())

                fact = NumericalFact(
                    entity=entity,
                    attribute=attr_type,
                    value=value,
                    unit=unit,
                    context=text[:100],
                    confidence=0.8,
                )
                facts.append(fact)

        return facts

    def _get_attribute_type(self, unit: str) -> str:
        """根据单位确定属性类型"""
        unit_map = {
            '度': '温度',
            '米': '长度',
            '千米': '长度',
            '公里': '长度',
            '厘米': '长度',
            '毫米': '长度',
            '千克': '重量',
            '公斤': '重量',
            '克': '重量',
            '吨': '重量',
            '%': '百分比',
        }
        return unit_map.get(unit, '数值')

    def _extract_entity_from_context(self, text: str, position: int) -> str:
        """从上下文中提取实体"""
        # 向前搜索实体
        before = text[:position]
        # 提取最后的中文实体
        entities = re.findall(r'[一-鿿]{2,6}', before)
        if entities:
            return entities[-1]

        # 向后搜索实体
        after = text[position:]
        entities = re.findall(r'[一-鿿]{2,6}', after)
        if entities:
            return entities[0]

        return "未知"

    def learn_from_text(self, text: str, entity: str = ""):
        """从文本中学习数值事实"""
        facts = self.extract_numerical_facts(text, entity)

        for fact in facts:
            self.add_fact(fact)

    def add_fact(self, fact: NumericalFact):
        """添加数值事实"""
        self.facts.append(fact)
        self.entity_index[fact.entity].append(fact)
        self.attribute_index[fact.attribute].append(fact)
        self.stats['facts_stored'] += 1

    def query(self, question: str) -> Dict:
        """查询数值信息"""
        # 提取问题中的实体和属性
        entities = self._extract_entities(question)
        attributes = self._extract_attributes(question)

        results = {
            'facts': [],
            'answers': [],
            'comparisons': [],
        }

        # 检查是否是比较问题
        comparison = self._detect_comparison(question)
        if comparison:
            results['comparisons'] = self._compare_values(comparison)

        # 搜索相关事实
        for entity in entities:
            for fact in self.entity_index.get(entity, []):
                results['facts'].append({
                    'entity': fact.entity,
                    'attribute': fact.attribute,
                    'value': fact.value,
                    'unit': fact.unit,
                    'confidence': fact.confidence,
                })

        # 按属性搜索
        for attr in attributes:
            for fact in self.attribute_index.get(attr, []):
                if fact not in [f for f in results['facts']]:
                    results['facts'].append({
                        'entity': fact.entity,
                        'attribute': fact.attribute,
                        'value': fact.value,
                        'unit': fact.unit,
                        'confidence': fact.confidence,
                    })

        # 生成答案
        for fact in results['facts'][:5]:
            answer = f"{fact['entity']}的{fact['attribute']}为{fact['value']}{fact['unit']}"
            results['answers'].append(answer)

        self.stats['queries_answered'] += 1
        return results

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        # 中文实体
        zh_entities = re.findall(r'[一-鿿]{2,6}', text)
        # 英文实体
        en_entities = re.findall(r'[A-Za-z]+', text)

        # 过滤
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样多少几')
        entities = []
        for e in zh_entities + en_entities:
            if e not in stopwords and len(e) >= 2:
                entities.append(e)

        return entities

    def _extract_attributes(self, text: str) -> List[str]:
        """提取属性"""
        # 属性关键词
        attr_keywords = {
            '温度': ['温度', '度', '热', '冷'],
            '长度': ['长度', '高', '长', '宽', '深', '远'],
            '重量': ['重量', '重', '质量'],
            '时间': ['时间', '年', '月', '天', '小时', '分钟', '秒'],
            '数量': ['数量', '多少', '几个', '几个'],
            '速度': ['速度', '快', '慢'],
            '面积': ['面积', '大'],
            '体积': ['体积', '容积'],
        }

        attributes = []
        for attr, keywords in attr_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    attributes.append(attr)
                    break

        return attributes

    def _detect_comparison(self, text: str) -> Optional[Dict]:
        """检测比较问题"""
        # 比较模式
        comparison_patterns = [
            (r'(.+)和(.+)哪个(大|小|高|低|重|轻|快|慢)', 'which_is_more'),
            (r'(.+)比(.+)(大|小|高|低|重|轻|快|慢)', 'compare'),
            (r'(.+)是否(大于|小于|等于|高于|低于)(.+)', 'check'),
        ]

        for pattern, comp_type in comparison_patterns:
            match = re.search(pattern, text)
            if match:
                if comp_type == 'which_is_more':
                    return {
                        'type': comp_type,
                        'entity1': match.group(1).strip(),
                        'entity2': match.group(2).strip(),
                        'attribute': match.group(3).strip(),
                    }
                elif comp_type == 'compare':
                    return {
                        'type': comp_type,
                        'entity1': match.group(1).strip(),
                        'entity2': match.group(2).strip(),
                        'attribute': match.group(3).strip(),
                    }
                elif comp_type == 'check':
                    return {
                        'type': comp_type,
                        'entity1': match.group(1).strip(),
                        'entity2': match.group(3).strip(),
                        'attribute': match.group(2).strip(),
                    }

        return None

    def _compare_values(self, comparison: Dict) -> List[Dict]:
        """比较数值"""
        results = []

        entity1 = comparison['entity1']
        entity2 = comparison['entity2']
        attribute = comparison['attribute']

        # 获取两个实体的数值
        facts1 = self.entity_index.get(entity1, [])
        facts2 = self.entity_index.get(entity2, [])

        # 找到相同属性的数值
        for f1 in facts1:
            for f2 in facts2:
                if f1.attribute == f2.attribute:
                    # 比较数值
                    if f1.value > f2.value:
                        results.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'attribute': f1.attribute,
                            'value1': f1.value,
                            'value2': f2.value,
                            'result': f'{entity1}更大',
                        })
                    elif f1.value < f2.value:
                        results.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'attribute': f1.attribute,
                            'value1': f1.value,
                            'value2': f2.value,
                            'result': f'{entity2}更大',
                        })
                    else:
                        results.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'attribute': f1.attribute,
                            'value1': f1.value,
                            'value2': f2.value,
                            'result': '相等',
                        })

        return results

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'total_facts': len(self.facts),
            'entities_tracked': len(self.entity_index),
            'attributes_tracked': len(self.attribute_index),
        }


def test_numerical_understanding():
    """测试数值理解"""
    print("=" * 70)
    print("数值理解层测试")
    print("=" * 70)

    understanding = NumericalUnderstanding()

    # 测试文本
    test_texts = [
        '水在100度沸腾。',
        '光速是300000千米/秒。',
        '地球的年龄约为46亿年。',
        '珠穆朗玛峰高8848米。',
        '中国有14亿人口。',
    ]

    with open('numerical_test.txt', 'w', encoding='utf-8') as f:
        f.write('数值理解层测试\n')
        f.write('=' * 70 + '\n\n')

        for text in test_texts:
            f.write(f'输入: {text}\n')
            understanding.learn_from_text(text)

        # 显示学到的事实
        f.write('\n' + '=' * 70 + '\n')
        f.write('学到的数值事实\n')
        f.write('=' * 70 + '\n')

        for fact in understanding.facts:
            f.write(f'  {fact.entity}: {fact.attribute} = {fact.value} {fact.unit}\n')

        # 测试查询
        f.write('\n' + '=' * 70 + '\n')
        f.write('查询测试\n')
        f.write('=' * 70 + '\n')

        test_questions = [
            '水在多少度沸腾',
            '光速是多少',
            '珠穆朗玛峰有多高',
            '中国有多少人口',
        ]

        for q in test_questions:
            f.write(f'\n问: {q}\n')
            result = understanding.query(q)
            for answer in result['answers']:
                f.write(f'  答: {answer}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = understanding.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to numerical_test.txt')


if __name__ == '__main__':
    test_numerical_understanding()
