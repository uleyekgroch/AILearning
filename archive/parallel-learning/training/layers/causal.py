"""Layer 2: 因果推理层

理解原因和结果，能推理因果链。

核心能力：
1. 因果识别 — 从文本中识别因果关系
2. 因果推理 — 从已知原因推导结果
3. 反事实推理 — "如果X没发生，会怎样"
4. 因果链 — A导致B，B导致C → A导致C

运行方式：
    python training/layers/causal.py
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CausalEvent:
    """因果事件"""
    event: str              # 事件描述
    timestamp: Optional[float] = None  # 时间戳
    location: Optional[str] = None     # 地点
    agents: List[str] = field(default_factory=list)  # 参与者
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性


@dataclass
class CausalLink:
    """因果链接 — 表示两个事件之间的因果关系"""
    cause: CausalEvent
    effect: CausalEvent
    mechanism: Optional[str] = None  # 因果机制
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)  # 证据
    counter_evidence: List[str] = field(default_factory=list)  # 反证

    def update_confidence(self, confirmed: bool):
        """更新置信度"""
        if confirmed:
            self.evidence.append("confirmed")
        else:
            self.counter_evidence.append("denied")
        total = len(self.evidence) + len(self.counter_evidence)
        positive = len(self.evidence)  # 所有证据都是正面的
        self.confidence = positive / total if total > 0 else 0.5


@dataclass
class CausalChain:
    """因果链 — A → B → C"""
    links: List[CausalLink]

    def get_root_cause(self) -> str:
        """获取根本原因"""
        if self.links:
            return self.links[0].cause.event
        return ""

    def get_final_effect(self) -> str:
        """获取最终结果"""
        if self.links:
            return self.links[-1].effect.event
        return ""

    def is_circular(self) -> bool:
        """检查是否有循环"""
        events = set()
        for link in self.links:
            if link.cause.event in events:
                return True
            events.add(link.cause.event)
            events.add(link.effect.event)
        return False


class CausalReasoning:
    """因果推理层

    核心能力：
    - 识别因果关系
    - 推理因果链
    - 反事实推理
    """

    def __init__(self):
        # 因果知识库
        self.causal_links: List[CausalLink] = []
        self.causal_graph: Dict[str, List[str]] = defaultdict(list)  # cause -> [effects]
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)  # effect -> [causes]
        self.link_index: Dict[Tuple[str, str], CausalLink] = {}  # (cause, effect) -> link

        # 因果模式库
        self.causal_patterns = self._load_causal_patterns()

        # 统计
        self.stats = {
            'total_links': 0,
            'total_chains': 0,
            'inferences_made': 0,
        }

    def _load_causal_patterns(self) -> List[Tuple[str, str, str, float]]:
        """加载因果模式

        格式: (模式, 因果类型, 机制描述, 基础置信度)
        """
        return [
            # 直接因果
            (r'因为(.+?)，所以(.+)', 'direct', '直接因果', 0.9),
            (r'由于(.+?)，(.+)', 'direct', '直接因果', 0.9),
            (r'(.+)导致(.+)', 'direct', '导致关系', 0.8),
            (r'(.+)引起(.+)', 'direct', '引起关系', 0.8),
            (r'(.+)造成(.+)', 'direct', '造成关系', 0.8),
            (r'(.+)引发(.+)', 'direct', '引发关系', 0.8),
            (r'(.+)促使(.+)', 'direct', '促使关系', 0.8),
            (r'(.+)致使(.+)', 'direct', '致使关系', 0.8),
            (r'(.+)产生(.+)', 'direct', '产生关系', 0.7),

            # 条件因果
            (r'如果(.+)，那么(.+)', 'conditional', '条件关系', 0.7),
            (r'只要(.+)，就(.+)', 'conditional', '充分条件', 0.8),
            (r'只有(.+)，才(.+)', 'conditional', '必要条件', 0.8),
            (r'一旦(.+)，(.+)', 'conditional', '条件关系', 0.7),
            (r'除非(.+)，否则(.+)', 'conditional', '必要条件', 0.7),

            # 时间因果
            (r'(.+)之后，(.+)', 'temporal', '时间先后', 0.6),
            (r'(.+)随后(.+)', 'temporal', '随后发生', 0.6),
            (r'(.+)进而(.+)', 'temporal', '进而导致', 0.7),
            (r'(.+)从而(.+)', 'temporal', '从而导致', 0.7),

            # 目的因果
            (r'为了(.+)，(.+)', 'purpose', '目的关系', 0.7),
            (r'(.+)是为了(.+)', 'purpose', '目的关系', 0.7),
            (r'(.+)以便(.+)', 'purpose', '目的关系', 0.7),
            (r'(.+)来(.+)', 'purpose', '目的关系', 0.6),

            # 使能因果
            (r'(.+)使得(.+)', 'enablement', '使能关系', 0.7),
            (r'(.+)让(.+)', 'enablement', '使能关系', 0.7),
            (r'(.+)帮助(.+)', 'enablement', '帮助关系', 0.6),
            (r'(.+)促进(.+)', 'enablement', '促进关系', 0.7),
            (r'(.+)推动(.+)', 'enablement', '推动关系', 0.7),

            # 阻止因果
            (r'(.+)阻止(.+)', 'prevention', '阻止关系', 0.7),
            (r'(.+)防止(.+)', 'prevention', '防止关系', 0.7),
            (r'(.+)避免(.+)', 'prevention', '避免关系', 0.7),
            (r'(.+)抑制(.+)', 'prevention', '抑制关系', 0.7),

            # 转折因果
            (r'虽然(.+)，但是(.+)', 'contrast', '转折关系', 0.6),
            (r'尽管(.+)，(.+)', 'contrast', '转折关系', 0.6),
            (r'(.+)然而(.+)', 'contrast', '转折关系', 0.6),
        ]

    def extract_causal_relations(self, text: str) -> List[CausalLink]:
        """从文本中提取因果关系"""
        links = []

        # 分句
        sentences = re.split(r'[。！？；\n]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # 应用因果模式
            for pattern, causal_type, mechanism, base_confidence in self.causal_patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    cause_text = match.group(1).strip()
                    effect_text = match.group(2).strip()

                    # 验证有效性
                    if not self._validate_causal(cause_text, effect_text):
                        continue

                    # 创建因果链接
                    cause_event = CausalEvent(event=cause_text)
                    effect_event = CausalEvent(event=effect_text)

                    link = CausalLink(
                        cause=cause_event,
                        effect=effect_event,
                        mechanism=mechanism,
                        confidence=base_confidence,
                        evidence=[sentence],
                    )

                    links.append(link)

        return links

    def _validate_causal(self, cause: str, effect: str) -> bool:
        """验证因果关系的有效性"""
        # 过滤太短的
        if len(cause) < 2 or len(effect) < 2:
            return False

        # 过滤太长的（可能是解析错误）
        if len(cause) > 30 or len(effect) > 30:
            return False

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        if cause in stopwords or effect in stopwords:
            return False

        # 检查因果不能相同
        if cause == effect:
            return False

        return True

    def learn_from_text(self, text: str):
        """从文本中学习因果关系"""
        links = self.extract_causal_relations(text)

        for link in links:
            self.add_causal_link(link)

    def add_causal_link(self, link: CausalLink):
        """添加因果链接"""
        # 检查是否已存在
        key = (link.cause.event, link.effect.event)
        existing = self.link_index.get(key)
        if existing:
            # 合并证据
            existing.evidence.extend(link.evidence)
            existing.update_confidence(True)
        else:
            self.causal_links.append(link)
            self.causal_graph[link.cause.event].append(link.effect.event)
            self.reverse_graph[link.effect.event].append(link.cause.event)
            self.link_index[key] = link
            self.stats['total_links'] += 1

    def _find_existing_link(self, cause: str, effect: str) -> Optional[CausalLink]:
        """查找已存在的因果链接（O(1)查找）"""
        return self.link_index.get((cause, effect))

    def infer_effects(self, cause: str, max_depth: int = 3) -> List[CausalChain]:
        """从原因推导结果

        给定一个原因，推理出所有可能的结果。
        """
        chains = []
        self._dfs_effects(cause, [], chains, set(), max_depth)
        self.stats['inferences_made'] += 1
        return chains

    def _dfs_effects(self, current: str, path: List[CausalLink],
                     chains: List[CausalChain], visited: Set[str], depth: int):
        """深度优先搜索结果"""
        if depth <= 0:
            return

        if current in visited:
            return

        visited.add(current)

        # 查找直接结果
        for effect in self.causal_graph.get(current, []):
            link = self._find_existing_link(current, effect)
            if link:
                new_path = path + [link]
                chains.append(CausalChain(links=new_path.copy()))
                self._dfs_effects(effect, new_path, chains, visited, depth - 1)

        visited.remove(current)

    def infer_causes(self, effect: str, max_depth: int = 3) -> List[CausalChain]:
        """从结果推导原因

        给定一个结果，推理出所有可能的原因。
        """
        chains = []
        self._dfs_causes(effect, [], chains, set(), max_depth)
        self.stats['inferences_made'] += 1
        return chains

    def _dfs_causes(self, current: str, path: List[CausalLink],
                    chains: List[CausalChain], visited: Set[str], depth: int):
        """深度优先搜索原因"""
        if depth <= 0:
            return

        if current in visited:
            return

        visited.add(current)

        # 查找直接原因
        for cause in self.reverse_graph.get(current, []):
            link = self._find_existing_link(cause, current)
            if link:
                new_path = [link] + path  # 注意：原因链是反向的
                chains.append(CausalChain(links=new_path.copy()))
                self._dfs_causes(cause, new_path, chains, visited, depth - 1)

        visited.remove(current)

    def counterfactual_reasoning(self, event: str, happened: bool) -> Dict:
        """反事实推理

        如果某个事件没有发生，会怎样？
        """
        results = {
            'original_event': event,
            'happened': happened,
            'implications': [],
        }

        if happened:
            # 事件发生了，推理它的结果
            chains = self.infer_effects(event, max_depth=2)
            for chain in chains:
                results['implications'].append({
                    'type': 'effect',
                    'chain': f"{chain.get_root_cause()} → {chain.get_final_effect()}",
                })
        else:
            # 事件没有发生，推理什么不会发生
            chains = self.infer_effects(event, max_depth=2)
            for chain in chains:
                results['implications'].append({
                    'type': 'prevented',
                    'chain': f"如果没有{event}，则不会{chain.get_final_effect()}",
                })

            # 推理替代原因
            # 如果A不会导致B，还有什么会导致B？
            for effect in self.causal_graph.get(event, []):
                alternative_causes = self.reverse_graph.get(effect, [])
                for alt_cause in alternative_causes:
                    if alt_cause != event:
                        results['implications'].append({
                            'type': 'alternative',
                            'chain': f"如果没有{event}，{effect}仍可能由{alt_cause}导致",
                        })

        self.stats['inferences_made'] += 1
        return results

    def find_causal_path(self, start: str, end: str, max_depth: int = 5) -> List[CausalChain]:
        """找到从start到end的所有因果路径"""
        chains = []
        self._find_path(start, end, [], chains, set(), max_depth)
        return chains

    def _find_path(self, current: str, target: str, path: List[CausalLink],
                   chains: List[CausalChain], visited: Set[str], depth: int):
        """查找因果路径"""
        if depth <= 0:
            return

        if current == target and path:
            chains.append(CausalChain(links=path.copy()))
            return

        if current in visited:
            return

        visited.add(current)

        for effect in self.causal_graph.get(current, []):
            link = self._find_existing_link(current, effect)
            if link:
                self._find_path(effect, target, path + [link], chains, visited, depth - 1)

        visited.remove(current)

    def query(self, question: str) -> Dict:
        """查询因果知识"""
        # 提取关键实体
        entities = self._extract_entities(question)

        results = {
            'causal_links': [],
            'chains': [],
            'counterfactual': None,
        }

        # 搜索相关因果链接（精确匹配 + 模糊匹配）
        for entity in entities:
            # 精确匹配：作为原因
            for effect in self.causal_graph.get(entity, []):
                link = self._find_existing_link(entity, effect)
                if link:
                    results['causal_links'].append({
                        'cause': entity,
                        'effect': effect,
                        'confidence': link.confidence,
                        'mechanism': link.mechanism,
                    })

            # 精确匹配：作为结果
            for cause in self.reverse_graph.get(entity, []):
                link = self._find_existing_link(cause, entity)
                if link:
                    results['causal_links'].append({
                        'cause': cause,
                        'effect': entity,
                        'confidence': link.confidence,
                        'mechanism': link.mechanism,
                    })

            # 模糊匹配：检查因果图中的键是否与实体相似
            for graph_entity in self.causal_graph.keys():
                if self._entities_similar(entity, graph_entity):
                    for effect in self.causal_graph[graph_entity]:
                        link = self._find_existing_link(graph_entity, effect)
                        if link:
                            results['causal_links'].append({
                                'cause': graph_entity,
                                'effect': effect,
                                'confidence': link.confidence,
                                'mechanism': link.mechanism,
                            })

            # 模糊匹配：检查反向图中的键是否与实体相似
            for graph_entity in self.reverse_graph.keys():
                if self._entities_similar(entity, graph_entity):
                    for cause in self.reverse_graph[graph_entity]:
                        link = self._find_existing_link(cause, graph_entity)
                        if link:
                            results['causal_links'].append({
                                'cause': cause,
                                'effect': graph_entity,
                                'confidence': link.confidence,
                                'mechanism': link.mechanism,
                            })

            # 推理因果链
            chains = self.infer_effects(entity, max_depth=2)
            for chain in chains:
                results['chains'].append({
                    'root_cause': chain.get_root_cause(),
                    'final_effect': chain.get_final_effect(),
                    'length': len(chain.links),
                })

        return results

    def _entities_similar(self, entity1: str, entity2: str) -> bool:
        """判断两个实体是否相似"""
        # 完全匹配
        if entity1 == entity2:
            return True

        # 包含匹配
        if entity1 in entity2 or entity2 in entity1:
            return True

        # 前缀匹配（至少2个字符相同）
        min_len = min(len(entity1), len(entity2))
        if min_len >= 2:
            # 检查前缀
            for i in range(min_len, 1, -1):
                if entity1[:i] == entity2[:i]:
                    return True
            # 检查后缀
            for i in range(min_len, 1, -1):
                if entity1[-i:] == entity2[-i:]:
                    return True

        return False

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        # 先用常见疑问词和虚词分割（不包括"了"，因为"地面湿了"是一个完整实体）
        # 注意：不要用"为了"，因为"了"会被当作分隔符
        separators = r'[为什么怎么如何的是有在位于属于包括使用产生导致引起因所如果那但而且或者而]'
        parts = re.split(separators, text)

        entities = []
        for part in parts:
            part = part.strip()
            if not part or len(part) < 2:
                continue

            # 中文实体（允许"了"结尾）
            zh_entities = re.findall(r'[一-鿿]{2,6}了?', part)
            for e in zh_entities:
                if len(e) >= 2:
                    entities.append(e)

        # 英文实体
        en_entities = re.findall(r'[A-Za-z]+', text)
        entities.extend(en_entities)

        # 去重
        return list(set(entities))

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'total_chains': len(self.causal_links),
            'graph_size': len(self.causal_graph),
        }


def test_causal_reasoning():
    """测试因果推理"""
    print("=" * 70)
    print("因果推理层测试")
    print("=" * 70)

    reasoning = CausalReasoning()

    # 测试文本
    test_texts = [
        "因为下雨，所以地面湿了。",
        "由于全球变暖，冰川开始融化。",
        "如果温度达到100度，水就会沸腾。",
        "他淋了雨，导致感冒了。",
        "为了考试及格，他每天学习。",
    ]

    with open('causal_test.txt', 'w', encoding='utf-8') as f:
        f.write('因果推理层测试\n')
        f.write('=' * 70 + '\n')

        for text in test_texts:
            f.write(f'\n输入: {text}\n')
            reasoning.learn_from_text(text)

        # 显示学到的因果关系
        f.write('\n' + '=' * 70 + '\n')
        f.write('学到的因果关系\n')
        f.write('=' * 70 + '\n')

        for link in reasoning.causal_links:
            f.write(f'  {link.cause.event} → {link.effect.event}\n')
            f.write(f'    机制: {link.mechanism}\n')
            f.write(f'    置信度: {link.confidence:.2f}\n')

        # 测试推理
        f.write('\n' + '=' * 70 + '\n')
        f.write('因果推理测试\n')
        f.write('=' * 70 + '\n')

        # 正向推理
        f.write('\n问: 如果下雨，会怎样？\n')
        chains = reasoning.infer_effects('下雨')
        for chain in chains:
            f.write(f'  → {chain.get_root_cause()} 导致 {chain.get_final_effect()}\n')

        # 反向推理
        f.write('\n问: 地面湿了，可能是什么原因？\n')
        chains = reasoning.infer_causes('地面湿')
        for chain in chains:
            f.write(f'  ← {chain.get_final_effect()} 由 {chain.get_root_cause()} 导致\n')

        # 反事实推理
        f.write('\n问: 如果没有下雨，会怎样？\n')
        result = reasoning.counterfactual_reasoning('下雨', happened=False)
        for impl in result['implications']:
            f.write(f'  - {impl["chain"]}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = reasoning.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to causal_test.txt')


if __name__ == '__main__':
    test_causal_reasoning()
