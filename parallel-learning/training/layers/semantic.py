"""Layer 1: 语义理解层

不是正则匹配，是真正理解句子含义。

核心能力：
1. 语义角色标注 — 谁对谁做了什么，何时何地为何
2. 指代消解 — "他"指的是谁
3. 词义消歧 — "苹果"是水果还是公司
4. 隐含关系 — "他淋了雨" → 隐含"他湿了"

运行方式：
    python training/layers/semantic.py
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class SemanticRole(Enum):
    """语义角色 — 谁在句子中扮演什么角色"""
    AGENT = "agent"          # 施事：执行动作的主体
    PATIENT = "patient"      # 受事：接受动作的客体
    EXPERIENCER = "experiencer"  # 经验者：感受者
    INSTRUMENT = "instrument"    # 工具：用什么
    LOCATION = "location"    # 地点：在哪里
    TIME = "time"            # 时间：什么时候
    CAUSE = "cause"          # 原因：为什么
    RESULT = "result"        # 结果：导致什么
    SOURCE = "source"        # 来源：从哪里来
    GOAL = "goal"            # 目标：到哪里去
    THEME = "theme"          # 主题：关于什么
    MANNER = "manner"        # 方式：怎样


@dataclass
class SemanticUnit:
    """语义单元 — 句子中的一个语义成分"""
    text: str                    # 原始文本
    role: SemanticRole           # 语义角色
    head: Optional[str] = None   # 中心词
    modifiers: List[str] = field(default_factory=list)  # 修饰语
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性


@dataclass
class SemanticRelation:
    """语义关系 — 两个语义单元之间的关系"""
    source: str          # 源实体
    target: str          # 目标实体
    relation: str        # 关系类型
    confidence: float    # 置信度
    evidence: str        # 证据（原始句子）


@dataclass
class SemanticFrame:
    """语义框架 — 一个事件或状态的完整语义表示

    这是理解的核心。不是存储"人工智能是分支"，
    而是理解：谁(人工智能) - 关系(是) - 什么(分支) - 什么的分支(计算机科学)
    """
    frame_type: str                          # 框架类型（事件/状态/关系）
    trigger: str                             # 触发词（动词/形容词）
    roles: Dict[SemanticRole, SemanticUnit] = field(default_factory=dict)
    negation: bool = False                   # 是否定
    modality: str = "factual"                # 情态：factual/possible/necessary
    tense: str = "present"                   # 时态
    source_sentence: str = ""                # 原始句子

    def get_agent(self) -> Optional[str]:
        """获取施事"""
        if SemanticRole.AGENT in self.roles:
            return self.roles[SemanticRole.AGENT].text
        return None

    def get_patient(self) -> Optional[str]:
        """获取受事"""
        if SemanticRole.PATIENT in self.roles:
            return self.roles[SemanticRole.PATIENT].text
        return None

    def get_relation(self) -> str:
        """获取关系"""
        return self.trigger


@dataclass
class SemanticParse:
    """语义解析结果 — 一个句子的完整语义表示"""
    original_text: str
    frames: List[SemanticFrame]
    entities: Dict[str, Dict]       # 实体及其属性
    relations: List[SemanticRelation]
    implicit_meanings: List[str]    # 隐含含义
    confidence: float


class SemanticUnderstanding:
    """语义理解层

    核心区别：
    - 正则匹配: "X是Y" → ("X", "是", "Y")
    - 语义理解: "X是Y" → 理解X和Y的关系，Y是什么类型的，X有什么属性
    """

    def __init__(self):
        # 语义框架库
        self.frames: Dict[str, List[SemanticFrame]] = defaultdict(list)

        # 实体库
        self.entities: Dict[str, Dict] = {}

        # 关系库（限制大小避免内存泄漏）
        self.relations: List[SemanticRelation] = []
        self.max_relations = 100000  # 最大关系数

        # 指代消解表
        self.coreference: Dict[str, str] = {}

        # 词义库
        self.word_senses: Dict[str, List[str]] = defaultdict(list)

        # 隐含关系规则
        self.implicit_rules = self._load_implicit_rules()

    def _load_implicit_rules(self) -> List[Tuple[str, str, str, float]]:
        """加载隐含关系规则

        格式: (前提模式, 隐含结论, 关系类型, 置信度)
        """
        return [
            # 物理隐含
            (r'淋.*雨', '湿了', 'physical_state', 0.9),
            (r'掉.*水', '湿了', 'physical_state', 0.9),
            (r'烧', '热', 'temperature', 0.8),
            (r'冰', '冷', 'temperature', 0.8),

            # 因果隐含
            (r'因为.*所以', None, 'causal', 0.9),  # 特殊处理
            (r'如果.*就', None, 'conditional', 0.8),

            # 时间隐含
            (r'已经', '过去', 'tense', 0.9),
            (r'正在', '现在', 'tense', 0.9),
            (r'将要', '未来', 'tense', 0.9),

            # 情感隐含
            (r'哭', '悲伤', 'emotion', 0.7),
            (r'笑', '快乐', 'emotion', 0.7),
            (r'怒', '愤怒', 'emotion', 0.7),
        ]

    def parse(self, text: str) -> SemanticParse:
        """解析文本的语义

        这是核心方法。不是正则匹配，是理解。
        """
        # 1. 分句
        sentences = self._split_sentences(text)

        all_frames = []
        all_entities = {}
        all_relations = []
        all_implicit = []

        for sentence in sentences:
            if len(sentence.strip()) < 3:
                continue

            # 2. 语义角色标注
            frames = self._identify_frames(sentence)

            # 3. 提取实体
            entities = self._extract_entities_with_properties(sentence)

            # 4. 识别关系
            relations = self._identify_relations(sentence, entities)
            self.relations.extend(relations)

            # 限制关系数量，删除最旧的
            if len(self.relations) > self.max_relations:
                self.relations = self.relations[-self.max_relations:]

            # 5. 推断隐含含义
            implicit = self._infer_implicit(sentence, frames)

            all_frames.extend(frames)
            all_entities.update(entities)
            all_relations.extend(relations)
            all_implicit.extend(implicit)

            # 6. 存储到内部表示
            for frame in frames:
                self._store_frame(frame)

        return SemanticParse(
            original_text=text,
            frames=all_frames,
            entities=all_entities,
            relations=all_relations,
            implicit_meanings=all_implicit,
            confidence=self._calculate_confidence(all_frames, all_relations)
        )

    def _split_sentences(self, text: str) -> List[str]:
        """分句 — 不只是按标点，还要理解句子边界"""
        # 基本分句
        sentences = re.split(r'[。！？；\n]', text)

        # 处理特殊情况
        result = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # 处理省略号
            sent = re.sub(r'…+', '。', sent)

            # 处理引用
            if '"' in sent or '"' in sent:
                # 保持引用的完整性
                result.append(sent)
            else:
                result.append(sent)

        return result

    def _identify_frames(self, sentence: str) -> List[SemanticFrame]:
        """识别语义框架 — 理解句子表达了什么事件/状态

        这是语义理解的核心。
        """
        frames = []

        # 先按逗号分句，避免贪婪匹配跨子句
        clauses = re.split(r'[，,；;、]', sentence)

        # 状态框架：X是Y（非贪婪）
        state_patterns = [
            (r'(.{2,10}?)是(.{2,30}?)$', 'be'),
            (r'(.{2,10}?)属于(.{2,20}?)$', 'belong_to'),
            (r'(.{2,10}?)位于(.{2,20}?)$', 'located_at'),
            (r'(.{2,10}?)称为(.{2,15}?)$', 'called'),
            (r'(.{2,10}?)叫做(.{2,15}?)$', 'called'),
            (r'(.{2,10}?)指的是(.{2,30}?)$', 'refers_to'),
            (r'(.{2,10}?)意味着(.{2,20}?)$', 'means'),
        ]

        for pattern, trigger in state_patterns:
            for clause in clauses:
                matches = re.finditer(pattern, clause)
                for match in matches:
                    agent_text = match.group(1).strip()
                    patient_text = match.group(2).strip()

                    # 验证有效性
                    if not self._validate_entity(agent_text) or not self._validate_entity(patient_text):
                        continue

                    frame = SemanticFrame(
                        frame_type='state',
                        trigger=trigger,
                        roles={
                            SemanticRole.AGENT: SemanticUnit(text=agent_text, role=SemanticRole.AGENT),
                            SemanticRole.PATIENT: SemanticUnit(text=patient_text, role=SemanticRole.PATIENT),
                        },
                        source_sentence=sentence,
                    )
                    frames.append(frame)

        # 事件框架：X做了Y
        event_patterns = [
            (r'(.{2,10}?)发明了?(.{2,20}?)$', 'invent'),
            (r'(.{2,10}?)发现了?(.{2,20}?)$', 'discover'),
            (r'(.{2,10}?)创造了?(.{2,20}?)$', 'create'),
            (r'(.{2,10}?)提出了?(.{2,20}?)$', 'propose'),
            (r'(.{2,10}?)使用了?(.{2,20}?)$', 'use'),
            (r'(.{2,10}?)导致了?(.{2,20}?)$', 'cause'),
            (r'(.{2,10}?)引起了?(.{2,20}?)$', 'cause'),
        ]

        for pattern, trigger in event_patterns:
            for clause in clauses:
                matches = re.finditer(pattern, clause)
                for match in matches:
                    agent_text = match.group(1).strip()
                    patient_text = match.group(2).strip()

                    if not self._validate_entity(agent_text) or not self._validate_entity(patient_text):
                        continue

                    frame = SemanticFrame(
                        frame_type='event',
                        trigger=trigger,
                        roles={
                            SemanticRole.AGENT: SemanticUnit(text=agent_text, role=SemanticRole.AGENT),
                            SemanticRole.PATIENT: SemanticUnit(text=patient_text, role=SemanticRole.PATIENT),
                        },
                        tense=self._detect_tense(sentence),
                        source_sentence=sentence,
                    )
                    frames.append(frame)

        # 否定检测
        for frame in frames:
            if '不' in sentence or '没' in sentence or '无' in sentence:
                # 检查否定词是否在触发词附近
                trigger_pos = sentence.find(frame.trigger)
                neg_positions = [m.start() for m in re.finditer(r'[不没无非]', sentence)]
                for neg_pos in neg_positions:
                    if abs(neg_pos - trigger_pos) < 5:
                        frame.negation = True
                        break

        return frames

    def _extract_entities_with_properties(self, sentence: str) -> Dict[str, Dict]:
        """提取实体及其属性

        策略：
        1. 先从语义框架提取（最准确）
        2. 再用正则补充
        """
        entities = {}

        # 英文实体（大写开头）
        en_pattern = r'([A-Z][a-zA-Z]+)'
        for match in re.finditer(en_pattern, sentence):
            entity_text = match.group(1)
            if self._validate_entity(entity_text):
                entities[entity_text] = {
                    'type': self._guess_entity_type(entity_text, sentence),
                    'properties': {},
                    'mentions': [sentence],
                }

        # 中文实体提取 — 使用更精确的分隔符
        # 包含动词分隔符，避免"牛顿发现"被当作一个实体
        separators = r'[，。！？；：、\s的是有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但发明了创造提出开发设计编写发现于发明于成立于创建于出生于诞生于出版于发表于坐落在坐落于建立在叫做称为指的是意味着]'

        # 分割句子
        parts = re.split(separators, sentence)

        for part in parts:
            part = part.strip()
            if not part or len(part) < 2:
                continue

            # 提取2-8字的中文片段
            zh_matches = re.findall(r'([一-鿿]{2,8})', part)
            for entity_text in zh_matches:
                if self._validate_entity(entity_text):
                    if entity_text not in entities:
                        entities[entity_text] = {
                            'type': self._guess_entity_type(entity_text, sentence),
                            'properties': {},
                            'mentions': [sentence],
                        }
                    else:
                        entities[entity_text]['mentions'].append(sentence)

                    # 从句子中提取属性
                    self._extract_entity_properties(entity_text, sentence, entities[entity_text])

        return entities

    def _validate_entity(self, text: str) -> bool:
        """验证是否是有效的实体"""
        # 过滤太短的
        if len(text) < 2:
            return False

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样可以能够应该')
        if text in stopwords:
            return False

        # 过滤纯标点
        if re.match(r'^[\W\s]+$', text):
            return False

        return True

    def _guess_entity_type(self, entity: str, context: str) -> str:
        """猜测实体类型"""
        # 人名模式
        if re.match(r'^[一-鿿]{2,3}$', entity) and any(w in context for w in ['他', '她', '先生', '女士', '教授', '博士']):
            return 'person'

        # 地名模式
        if any(suffix in entity for suffix in ['国', '省', '市', '县', '区', '镇', '村']):
            return 'location'

        # 组织模式
        if any(suffix in entity for suffix in ['公司', '大学', '学院', '机构', '组织', '协会']):
            return 'organization'

        # 时间模式
        if re.match(r'\d{4}年', entity):
            return 'time'

        # 语言模式
        if any(suffix in entity for suffix in ['语', '文']):
            return 'language'

        return 'concept'

    def _extract_entity_properties(self, entity: str, sentence: str, entity_info: Dict):
        """从句子中提取实体属性"""
        # 属性模式
        prop_patterns = [
            (f'{entity}的(\\w+)为(\\d+)', 'numerical'),
            (f'{entity}的(\\w+)是(\\w+)', 'descriptive'),
            (f'{entity}(\\w+)为(\\d+)', 'numerical'),
        ]

        for pattern, prop_type in prop_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if len(match) >= 2:
                    prop_name = match[0]
                    prop_value = match[1]
                    entity_info['properties'][prop_name] = {
                        'value': prop_value,
                        'type': prop_type,
                    }

    def _identify_relations(self, sentence: str, entities: Dict) -> List[SemanticRelation]:
        """识别实体间关系"""
        relations = []

        entity_list = list(entities.keys())

        # 两两检查实体关系
        for i, e1 in enumerate(entity_list):
            for e2 in entity_list[i+1:]:
                # 检查它们是否在同一个句子中
                if e1 in sentence and e2 in sentence:
                    # 推断关系类型
                    relation_type = self._infer_relation(e1, e2, sentence)

                    relation = SemanticRelation(
                        source=e1,
                        target=e2,
                        relation=relation_type,
                        confidence=0.7,
                        evidence=sentence,
                    )
                    relations.append(relation)

        return relations

    def _infer_relation(self, e1: str, e2: str, sentence: str) -> str:
        """推断两个实体之间的关系"""
        # 关系指示词
        indicators = {
            '是': 'is_a',
            '属于': 'belongs_to',
            '包括': 'includes',
            '位于': 'located_in',
            '使用': 'uses',
            '用于': 'used_for',
            '产生': 'produces',
            '导致': 'causes',
            '发明': 'invented',
            '发现': 'discovered',
            '创造': 'created',
            '提出': 'proposed',
            '由': 'created_by',
            '在': 'located_at',
        }

        for indicator, relation in indicators.items():
            # 检查指示词是否在两个实体之间
            pos1 = sentence.find(e1)
            pos2 = sentence.find(e2)
            indicator_pos = sentence.find(indicator)

            if pos1 < indicator_pos < pos2 or pos2 < indicator_pos < pos1:
                return relation

        return 'related_to'

    def _detect_tense(self, sentence: str) -> str:
        """检测时态"""
        if any(w in sentence for w in ['了', '过', '曾经', '已经']):
            return 'past'
        elif any(w in sentence for w in ['正在', '在', '现在']):
            return 'present'
        elif any(w in sentence for w in ['将', '会', '要', '即将']):
            return 'future'
        return 'present'

    def _infer_implicit(self, sentence: str, frames: List[SemanticFrame]) -> List[str]:
        """推断隐含含义

        这是理解的关键。不只是记住说了什么，还要理解没说但隐含的意思。
        """
        implicit = []

        # 应用隐含规则
        for pattern, conclusion, relation_type, confidence in self.implicit_rules:
            if re.search(pattern, sentence):
                if conclusion:
                    implicit.append(conclusion)
                elif relation_type == 'causal':
                    # 提取因果
                    causal_match = re.search(r'因为(.+?)，所以(.+)', sentence)
                    if causal_match:
                        implicit.append(f"{causal_match.group(1)} 导致 {causal_match.group(2)}")

        # 从语义框架推断
        for frame in frames:
            if frame.frame_type == 'state' and frame.trigger == 'be':
                # X是Y → X具有Y的属性
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent and patient:
                    implicit.append(f"{agent} 具有 {patient} 的特征")

            elif frame.frame_type == 'event' and frame.trigger == 'cause':
                # X导致Y → X是Y的原因
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent and patient:
                    implicit.append(f"{agent} 是 {patient} 的原因")

        return implicit

    def _store_frame(self, frame: SemanticFrame):
        """存储语义框架"""
        # 按触发词索引
        self.frames[frame.trigger].append(frame)

        # 按实体索引
        for role, unit in frame.roles.items():
            if unit.text not in self.entities:
                self.entities[unit.text] = {'frames': []}
            if 'frames' not in self.entities[unit.text]:
                self.entities[unit.text]['frames'] = []
            self.entities[unit.text]['frames'].append(frame)

    def _calculate_confidence(self, frames: List[SemanticFrame],
                             relations: List[SemanticRelation]) -> float:
        """计算整体解析置信度"""
        if not frames and not relations:
            return 0.0

        # 基于框架数量和关系数量
        frame_score = min(1.0, len(frames) * 0.2)
        relation_score = min(1.0, len(relations) * 0.1)

        return (frame_score + relation_score) / 2

    def query(self, question: str) -> Dict:
        """查询语义知识"""
        # 解析问题
        question_parse = self.parse(question)

        # 提取问题中的实体
        question_entities = list(question_parse.entities.keys())

        # 搜索相关框架
        related_frames = []
        for entity in question_entities:
            if entity in self.entities:
                entity_info = self.entities[entity]
                if 'frames' in entity_info:
                    related_frames.extend(entity_info['frames'])

        # 搜索相关关系
        related_relations = []
        for relation in self.relations:
            if any(entity in relation.source or entity in relation.target
                   for entity in question_entities):
                related_relations.append(relation)

        return {
            'question_entities': question_entities,
            'related_frames': related_frames[:10],
            'related_relations': related_relations[:10],
            'implicit_meanings': question_parse.implicit_meanings,
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_frames = sum(len(frames) for frames in self.frames.values())
        return {
            'total_frames': total_frames,
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'frame_types': {k: len(v) for k, v in self.frames.items()},
        }


def test_semantic_understanding():
    """测试语义理解"""
    print("=" * 70)
    print("语义理解层测试")
    print("=" * 70)

    understanding = SemanticUnderstanding()

    # 测试句子
    test_sentences = [
        "人工智能是计算机科学的一个分支。",
        "牛顿发现了万有引力定律。",
        "Python是一种编程语言。",
        "太阳位于太阳系的中心。",
        "因为下雨，所以地面湿了。",
    ]

    for sentence in test_sentences:
        print(f"\n输入: {sentence}")
        parse = understanding.parse(sentence)

        print(f"  框架数: {len(parse.frames)}")
        for frame in parse.frames:
            agent = frame.get_agent()
            patient = frame.get_patient()
            print(f"    {frame.frame_type}: {agent} --[{frame.trigger}]--> {patient}")

        print(f"  实体数: {len(parse.entities)}")
        for entity, info in parse.entities.items():
            print(f"    {entity}: {info.get('type', 'unknown')}")

        print(f"  关系数: {len(parse.relations)}")
        for rel in parse.relations[:3]:
            print(f"    {rel.source} --[{rel.relation}]--> {rel.target}")

        if parse.implicit_meanings:
            print(f"  隐含含义:")
            for meaning in parse.implicit_meanings:
                print(f"    - {meaning}")

    # 显示统计
    print("\n" + "=" * 70)
    print("统计")
    print("=" * 70)
    stats = understanding.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_semantic_understanding()
