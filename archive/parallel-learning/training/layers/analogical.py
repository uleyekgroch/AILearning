"""类比推理层

用已知理解未知，通过类比学习新概念。

核心能力：
1. 结构映射 — 映射源域到目标域
2. 类比发现 — 发现相似结构
3. 类比生成 — 生成新的类比
4. 类比验证 — 验证类比的有效性

运行方式：
    python training/layers/analogical.py
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Concept:
    """概念"""
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class Analogy:
    """类比"""
    source: str            # 源域
    target: str            # 目标域
    mappings: Dict[str, str]  # 映射关系
    confidence: float      # 置信度
    explanation: str       # 解释


class AnalogicalReasoning:
    """类比推理层

    核心能力：
    - 发现相似结构
    - 用已知理解未知
    - 生成新的类比
    """

    def __init__(self):
        # 概念库
        self.concepts: Dict[str, Concept] = {}

        # 类比库
        self.analogies: List[Analogy] = []

        # 已知的类比模式
        self.analogy_patterns = self._load_analogy_patterns()

        # 统计
        self.stats = {
            'concepts_stored': 0,
            'analogies_found': 0,
            'analogies_generated': 0,
        }

    def _load_analogy_patterns(self) -> List[Tuple[str, str, str, float]]:
        """加载类比模式

        格式: (源域, 目标域, 映射描述, 基础置信度)
        """
        return [
            # 物理类比
            ('水流', '电流', '水流像电流', 0.8),
            ('水管', '导线', '水管像导线', 0.8),
            ('水泵', '电池', '水泵像电池', 0.7),
            ('水压', '电压', '水压像电压', 0.8),

            # 生物类比
            ('心脏', '水泵', '心脏像水泵', 0.7),
            ('大脑', '计算机', '大脑像计算机', 0.6),
            ('眼睛', '相机', '眼睛像相机', 0.7),
            ('耳朵', '麦克风', '耳朵像麦克风', 0.7),

            # 社会类比
            ('公司', '机器', '公司像机器', 0.6),
            ('团队', '球队', '团队像球队', 0.7),
            ('领导', '教练', '领导像教练', 0.6),

            # 抽象类比
            ('时间', '河流', '时间像河流', 0.5),
            ('知识', '大厦', '知识像大厦', 0.5),
            ('学习', '旅行', '学习像旅行', 0.5),
        ]

    def add_concept(self, name: str, properties: Dict[str, Any] = None,
                   relations: Dict[str, List[str]] = None):
        """添加概念"""
        concept = Concept(
            name=name,
            properties=properties or {},
            relations=relations or defaultdict(list),
        )
        self.concepts[name] = concept
        self.stats['concepts_stored'] += 1

    def find_analogy(self, source: str, target: str) -> Optional[Analogy]:
        """查找类比"""
        # 检查已知模式
        for src, tgt, desc, conf in self.analogy_patterns:
            if (source in src and target in tgt) or (source in tgt and target in src):
                analogy = Analogy(
                    source=source,
                    target=target,
                    mappings={source: target},
                    confidence=conf,
                    explanation=desc,
                )
                self.analogies.append(analogy)
                self.stats['analogies_found'] += 1
                return analogy

        # 尝试基于概念相似性生成类比
        if source in self.concepts and target in self.concepts:
            analogy = self._generate_analogy(source, target)
            if analogy:
                self.analogies.append(analogy)
                self.stats['analogies_generated'] += 1
                return analogy

        return None

    def _generate_analogy(self, source: str, target: str) -> Optional[Analogy]:
        """基于概念相似性生成类比"""
        source_concept = self.concepts[source]
        target_concept = self.concepts[target]

        # 计算属性相似度
        common_props = set(source_concept.properties.keys()) & set(target_concept.properties.keys())
        if not common_props:
            return None

        # 计算相似度
        similarity = len(common_props) / max(
            len(source_concept.properties),
            len(target_concept.properties)
        )

        if similarity < 0.3:
            return None

        # 生成映射
        mappings = {}
        for prop in common_props:
            mappings[f"{source}.{prop}"] = f"{target}.{prop}"

        # 生成解释
        explanation = f"{source}和{target}在{len(common_props)}个属性上相似"

        return Analogy(
            source=source,
            target=target,
            mappings=mappings,
            confidence=similarity,
            explanation=explanation,
        )

    def explain_analogy(self, analogy: Analogy) -> str:
        """解释类比"""
        parts = []
        parts.append(f"类比: {analogy.source} 像 {analogy.target}")
        parts.append(f"解释: {analogy.explanation}")
        parts.append(f"置信度: {analogy.confidence:.2f}")

        if analogy.mappings:
            parts.append("映射关系:")
            for src, tgt in list(analogy.mappings.items())[:5]:
                parts.append(f"  {src} → {tgt}")

        return '\n'.join(parts)

    def learn_from_text(self, text: str, source: str = ""):
        """从文本中学习类比"""
        # 提取类比模式（非贪婪，去除标点）
        analogy_patterns = [
            (r'(.{2,10}?)像(.{2,10}?)[一样]', '像'),
            (r'(.{2,10}?)像(.{2,10}?)[。，]', '像'),
            (r'(.{2,10}?)如同(.{2,10}?)[一样]', '如同'),
            (r'(.{2,10}?)好比(.{2,10}?)[。，]', '好比'),
            (r'(.{2,10}?)相当于(.{2,10}?)[。，]', '相当于'),
        ]

        for pattern, relation in analogy_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                src = match.group(1).strip()
                tgt = match.group(2).strip()

                if len(src) >= 2 and len(tgt) >= 2:
                    # 添加类比
                    analogy = Analogy(
                        source=src,
                        target=tgt,
                        mappings={src: tgt},
                        confidence=0.7,
                        explanation=f"{src}像{tgt}",
                    )
                    self.analogies.append(analogy)
                    self.stats['analogies_generated'] += 1

    def query(self, question: str) -> Dict:
        """查询类比知识"""
        # 提取关键词
        keywords = self._extract_keywords(question)

        results = {
            'analogies': [],
            'explanations': [],
        }

        # 搜索相关类比
        for analogy in self.analogies:
            for keyword in keywords:
                if keyword in analogy.source or keyword in analogy.target:
                    results['analogies'].append({
                        'source': analogy.source,
                        'target': analogy.target,
                        'confidence': analogy.confidence,
                        'explanation': analogy.explanation,
                    })
                    results['explanations'].append(self.explain_analogy(analogy))
                    break

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 先用常见词分割
        separators = r'[像什么如何怎样的了是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'
        parts = re.split(separators, text)

        zh_keywords = []
        for part in parts:
            part = part.strip()
            if part and len(part) >= 2:
                zh_keywords.extend(re.findall(r'[一-鿿]{2,6}', part))

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        keywords = [k for k in zh_keywords if k not in stopwords and len(k) >= 2]

        return keywords

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'total_analogies': len(self.analogies),
        }


def test_analogical_reasoning():
    """测试类比推理"""
    print("=" * 70)
    print("类比推理层测试")
    print("=" * 70)

    reasoning = AnalogicalReasoning()

    # 测试概念添加
    concepts = [
        ('水流', {'方向': '向下', '速度': '可变'}, {'流向': ['河流', '管道']}),
        ('电流', {'方向': '可变', '速度': '光速'}, {'流向': ['导线', '电路']}),
        ('心脏', {'功能': '泵血', '位置': '胸腔'}, {'连接': ['血管']}),
        ('水泵', {'功能': '泵水', '位置': '管道'}, {'连接': ['水管']}),
    ]

    with open('analogical_test.txt', 'w', encoding='utf-8') as f:
        f.write('类比推理层测试\n')
        f.write('=' * 70 + '\n\n')

        for name, props, rels in concepts:
            reasoning.add_concept(name, props, rels)
            f.write(f'添加概念: {name}\n')

        # 测试类比查找
        f.write('\n' + '=' * 70 + '\n')
        f.write('类比查找\n')
        f.write('=' * 70 + '\n')

        test_pairs = [
            ('水流', '电流'),
            ('心脏', '水泵'),
            ('大脑', '计算机'),
        ]

        for source, target in test_pairs:
            f.write(f'\n查找类比: {source} → {target}\n')
            analogy = reasoning.find_analogy(source, target)
            if analogy:
                f.write(f'  {reasoning.explain_analogy(analogy)}\n')
            else:
                f.write(f'  未找到类比\n')

        # 测试从文本学习
        f.write('\n' + '=' * 70 + '\n')
        f.write('从文本学习类比\n')
        f.write('=' * 70 + '\n')

        test_texts = [
            '时间像河流一样流逝。',
            '知识像大厦一样需要基础。',
            '学习像旅行一样充满发现。',
        ]

        for text in test_texts:
            f.write(f'\n输入: {text}\n')
            reasoning.learn_from_text(text)

        # 测试查询
        f.write('\n' + '=' * 70 + '\n')
        f.write('查询测试\n')
        f.write('=' * 70 + '\n')

        test_questions = [
            '水流像什么',
            '心脏像什么',
            '时间像什么',
        ]

        for q in test_questions:
            f.write(f'\n问: {q}\n')
            result = reasoning.query(q)
            for explanation in result['explanations'][:2]:
                f.write(f'  {explanation}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = reasoning.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to analogical_test.txt')


if __name__ == '__main__':
    test_analogical_reasoning()
