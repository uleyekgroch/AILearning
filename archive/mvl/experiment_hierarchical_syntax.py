"""
Phase 69: 层级语法 —— 词序承载语义角色

核心思想：
Phase 32 发现词序冗余（词袋匹配就够了），Phase 16a 的从句从未涌现。
这是因为当前系统中符号是无序的集合，位置不承载信息。

本阶段测试：当位置确定语义角色时，层级语法是否涌现？
- Slot 0 = 施事者（谁做的）
- Slot 1 = 动作（做了什么）
- Slot 2 = 受事者（对谁做的）

关键压力：场景中多个同类别对象时，只有位置信息能区分角色。

涌现条件：
1. 场景包含 agent-action-patient 三元组关系
2. 多个同类别对象使词袋匹配失效
3. 位置敏感的解释能消除歧义
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, ACTIONS,
)
from language_rich_scene import generate_rich_scene_v2


# ============================================================
# 关系场景
# ============================================================

def generate_relational_scene(num_events: int = 4) -> List[Dict]:
    """
    生成包含 agent-action-patient 关系的场景

    每个事件：
    {
        'agent': {'color': 'red', 'shape': 'circle'},
        'action': 'push',
        'patient': {'color': 'blue', 'shape': 'square'},
    }
    """
    colors = list(COLORS)
    shapes = list(SHAPES)
    actions = list(ACTIONS)

    events = []
    for _ in range(num_events):
        event = {
            'agent': {'color': random.choice(colors), 'shape': random.choice(shapes)},
            'action': random.choice(actions),
            'patient': {'color': random.choice(colors), 'shape': random.choice(shapes)},
        }
        events.append(event)

    return events


def events_to_features(events: List[Dict]) -> List[Dict]:
    """将事件转换为特征列表（用于语言系统）"""
    features = []
    for event in events:
        features.append({
            'agent_color': event['agent']['color'],
            'agent_shape': event['agent']['shape'],
            'action': event['action'],
            'patient_color': event['patient']['color'],
            'patient_shape': event['patient']['shape'],
        })
    return features


# ============================================================
# 槽位语法
# ============================================================

class SlotGrammar:
    """
    槽位语法：位置决定语义角色

    Slot 0 = agent（施事者）
    Slot 1 = action（动作）
    Slot 2 = patient（受事者）

    示例：["red", "push", "blue"] = 红色(施事) 推了 蓝色(受事)
    """

    SLOTS = ['agent', 'action', 'patient']

    def __init__(self):
        self.slot_order = ['agent', 'action', 'patient']
        self.slot_consistency = defaultdict(lambda: defaultdict(int))
        self.total_plays = 0

    def parse(self, utterance: List[str]) -> Dict[str, str]:
        """按位置分配语义角色"""
        result = {}
        for i, sym in enumerate(utterance):
            if i < len(self.slot_order):
                result[self.slot_order[i]] = sym
        return result

    def generate(self, slots: Dict[str, str]) -> List[str]:
        """从角色字典生成有序话语"""
        result = []
        for slot in self.slot_order:
            if slot in slots:
                result.append(slots[slot])
        return result

    def record_order(self, utterance: List[str], success: bool):
        """记录位置使用"""
        self.total_plays += 1
        for i, sym in enumerate(utterance):
            cat = _symbol_category(sym) or sym
            self.slot_consistency[i][cat] += 1

    def get_consistency(self) -> float:
        """测量位置一致性（1.0 = 完全一致，0.0 = 完全随机）"""
        if self.total_plays == 0:
            return 0.0
        total_consistency = 0.0
        for slot_idx in self.slot_consistency:
            counts = self.slot_consistency[slot_idx]
            total = sum(counts.values())
            if total > 0:
                max_count = max(counts.values())
                total_consistency += max_count / total
        n_slots = len(self.slot_consistency)
        return total_consistency / max(1, n_slots)


class RecursiveGrammar(SlotGrammar):
    """
    递归语法：嵌入子句

    "red push blue that green hit"
    = 红色推了被绿色撞的蓝色

    嵌入标记 "that" 分隔主句和子句。
    max_depth=2 允许一层嵌入。
    """

    def __init__(self, embedding_marker: str = 'that', max_depth: int = 2):
        super().__init__()
        self.embedding_marker = embedding_marker
        self.max_depth = max_depth
        self.embedding_usage = 0

    def parse_recursive(self, utterance: List[str]) -> List[Dict]:
        """
        解析带嵌入的话语

        Returns: 子句列表
        """
        clauses = []
        current = []

        for sym in utterance:
            if sym == self.embedding_marker and len(current) >= 3:
                # 找到嵌入标记，结束当前子句
                clauses.append(self.parse(current))
                current = []
            else:
                current.append(sym)

        if current:
            clauses.append(self.parse(current))

        return clauses

    def generate_recursive(self, main_event: Dict,
                           relative_event: Optional[Dict] = None) -> List[str]:
        """生成带嵌入子句的话语"""
        main_slots = {
            'agent': main_event['agent']['color'],
            'action': main_event['action'],
            'patient': main_event['patient']['color'],
        }
        result = self.generate(main_slots)

        if relative_event and self.max_depth > 1:
            rel_slots = {
                'agent': relative_event['agent']['color'],
                'action': relative_event['action'],
                'patient': relative_event['patient']['color'],
            }
            result.append(self.embedding_marker)
            result.extend(self.generate(rel_slots))
            self.embedding_usage += 1

        return result


# ============================================================
# 层级说话者/听者
# ============================================================

class HierarchicalSpeaker:
    """
    层级说话者

    使用槽位语法生成位置敏感的话语。
    当场景需要区分多个同颜色对象时，使用递归嵌入。
    """

    def __init__(self, language: EmergingLanguage, grammar: SlotGrammar):
        self.language = language
        self.grammar = grammar
        self.total_descriptions = 0

    def describe(self, target_event: Dict, all_events: List[Dict],
                 target_idx: int) -> List[str]:
        """
        描述目标事件

        Args:
            target_event: 目标事件
            all_events: 所有事件
            target_idx: 目标事件索引
        """
        self.total_descriptions += 1

        # 基本描述：agent_color + action + patient_color
        utterance = self.grammar.generate({
            'agent': target_event['agent']['color'],
            'action': target_event['action'],
            'patient': target_event['patient']['color'],
        })

        # 检查是否有歧义（另一个事件有相同的颜色）
        ambiguous = False
        for i, event in enumerate(all_events):
            if i == target_idx:
                continue
            if (event['agent']['color'] == target_event['agent']['color'] and
                event['action'] == target_event['action']):
                ambiguous = True
                break

        # 如果有歧义且有递归语法，添加嵌入子句
        if ambiguous and isinstance(self.grammar, RecursiveGrammar):
            # 找一个能区分的事件
            for event in all_events:
                if event['patient']['color'] == target_event['patient']['color']:
                    utterance = self.grammar.generate_recursive(
                        target_event, event
                    )
                    break

        return utterance


class HierarchicalListener:
    """
    层级听者

    位置敏感的解释：
    - Slot 0 的颜色用于过滤 agent
    - Slot 1 的动作用于匹配
    - Slot 2 的颜色用于过滤 patient
    """

    def __init__(self, language: EmergingLanguage, grammar: SlotGrammar):
        self.language = language
        self.grammar = grammar

    def interpret(self, utterance: List[str],
                  events: List[Dict]) -> Optional[int]:
        """
        解释话语，返回匹配的事件索引

        使用位置敏感匹配：不同位置的颜色匹配不同角色
        """
        if isinstance(self.grammar, RecursiveGrammar):
            # 递归解析
            clauses = self.grammar.parse_recursive(utterance)
            if not clauses:
                return None
            # 用主句匹配
            return self._match_event(clauses[0], events)
        else:
            parsed = self.grammar.parse(utterance)
            return self._match_event(parsed, events)

    def _match_event(self, parsed: Dict[str, str],
                     events: List[Dict]) -> Optional[int]:
        """用解析后的角色字典匹配事件"""
        best_idx = None
        best_score = -1

        for i, event in enumerate(events):
            score = 0

            # Agent 匹配（slot 0）
            if 'agent' in parsed:
                if event['agent']['color'] == parsed['agent']:
                    score += 2
                elif event['agent']['shape'] == parsed['agent']:
                    score += 1

            # Action 匹配（slot 1）
            if 'action' in parsed:
                if event['action'] == parsed['action']:
                    score += 2

            # Patient 匹配（slot 2）
            if 'patient' in parsed:
                if event['patient']['color'] == parsed['patient']:
                    score += 2
                elif event['patient']['shape'] == parsed['patient']:
                    score += 1

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx


class BagOfSymbolsListener:
    """词袋听者：忽略位置，只看符号集合"""

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def interpret(self, utterance: List[str],
                  events: List[Dict]) -> Optional[int]:
        """词袋匹配：忽略位置"""
        utterance_set = set(utterance)
        best_idx = None
        best_score = -1

        for i, event in enumerate(events):
            score = 0
            # 把事件所有属性值放进集合
            event_values = set()
            event_values.add(event['agent']['color'])
            event_values.add(event['agent']['shape'])
            event_values.add(event['action'])
            event_values.add(event['patient']['color'])
            event_values.add(event['patient']['shape'])

            score = len(utterance_set & event_values)

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx


# ============================================================
# 游戏
# ============================================================

class HierarchicalGame:
    """层级语法游戏"""

    def __init__(self, speaker: HierarchicalSpeaker,
                 listener: HierarchicalListener,
                 language: EmergingLanguage):
        self.speaker = speaker
        self.listener = listener
        self.language = language
        self.games_played = 0
        self.successes = 0

    def play_round(self) -> Dict:
        events = generate_relational_scene(4)
        target_idx = random.randint(0, len(events) - 1)

        utterance = self.speaker.describe(events[target_idx], events, target_idx)
        chosen_idx = self.listener.interpret(utterance, events)

        success = (chosen_idx == target_idx)

        self.games_played += 1
        if success:
            self.successes += 1

        self.language.record_usage(utterance, success)
        self.speaker.grammar.record_order(utterance, success)

        return {'success': success, 'utterance': utterance}


class BaselineBagGame:
    """基线：词袋匹配，位置无关"""

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.games_played = 0
        self.successes = 0

    def play_round(self) -> Dict:
        events = generate_relational_scene(4)
        target_idx = random.randint(0, len(events) - 1)

        # 词袋描述（无序）
        target = events[target_idx]
        utterance = [target['agent']['color'], target['action'],
                     target['patient']['color']]
        random.shuffle(utterance)

        listener = BagOfSymbolsListener(self.language)
        chosen_idx = listener.interpret(utterance, events)
        success = (chosen_idx == target_idx)

        self.games_played += 1
        if success:
            self.successes += 1
        self.language.record_usage(utterance, success)

        return {'success': success}


# ============================================================
# 实验
# ============================================================

def experiment_1_positional_emergence(num_rounds: int = 300) -> Dict:
    """
    实验 1：位置语法涌现

    追踪 slot 一致性是否随时间增长。
    """
    print("=" * 60)
    print("实验 1：位置语法涌现")
    print("=" * 60)

    grammar = SlotGrammar()
    lang = EmergingLanguage()
    speaker = HierarchicalSpeaker(lang, grammar)
    listener = HierarchicalListener(lang, grammar)
    game = HierarchicalGame(speaker, listener, lang)

    snapshots = []
    for r in range(num_rounds):
        game.play_round()
        if (r + 1) % 50 == 0:
            consistency = grammar.get_consistency()
            sr = game.successes / max(1, game.games_played)
            snapshots.append({
                'round': r + 1,
                'consistency': round(consistency, 4),
                'success_rate': round(sr, 4),
            })
            print(f"  Round {r+1}: 一致性={consistency:.3f}, SR={sr:.3f}")

    print(f"\n  最终一致性: {grammar.get_consistency():.3f}")

    return {
        'final_consistency': round(grammar.get_consistency(), 4),
        'final_success_rate': round(game.successes / max(1, game.games_played), 4),
        'snapshots': snapshots,
    }


def experiment_2_recursive_embedding(num_rounds: int = 200) -> Dict:
    """
    实验 2：递归嵌入

    使用 RecursiveGrammar，追踪 "that" 嵌入标记使用。
    """
    print("=" * 60)
    print("实验 2：递归嵌入")
    print("=" * 60)

    grammar = RecursiveGrammar()
    lang = EmergingLanguage()
    speaker = HierarchicalSpeaker(lang, grammar)
    listener = HierarchicalListener(lang, grammar)
    game = HierarchicalGame(speaker, listener, lang)

    embedding_counts = []
    for r in range(num_rounds):
        game.play_round()
        if (r + 1) % 50 == 0:
            sr = game.successes / max(1, game.games_played)
            emb = grammar.embedding_usage
            has_that = 'that' in lang.vocabulary
            embedding_counts.append({
                'round': r + 1,
                'embedding_count': emb,
                'has_that_marker': has_that,
                'success_rate': round(sr, 4),
            })
            print(f"  Round {r+1}: SR={sr:.3f}, 嵌入次数={emb}, "
                  f"'that' 在词汇={'是' if has_that else '否'}")

    return {
        'final_sr': round(game.successes / max(1, game.games_played), 4),
        'embedding_count': grammar.embedding_usage,
        'has_that': 'that' in lang.vocabulary,
        'snapshots': embedding_counts,
    }


def experiment_3_word_order_vs_bag(num_rounds: int = 200,
                                    num_runs: int = 5) -> Dict:
    """
    实验 3：词序 vs 词袋

    对比位置敏感匹配 vs 词袋匹配。
    """
    print("=" * 60)
    print("实验 3：词序 vs 词袋匹配")
    print("=" * 60)

    hierarchical_srs = []
    bag_srs = []

    for run in range(num_runs):
        # 层级语法
        grammar = SlotGrammar()
        lang = EmergingLanguage()
        sp = HierarchicalSpeaker(lang, grammar)
        li = HierarchicalListener(lang, grammar)
        game = HierarchicalGame(sp, li, lang)
        for _ in range(num_rounds):
            game.play_round()
        h_sr = game.successes / max(1, game.games_played)
        hierarchical_srs.append(h_sr)

        # 词袋
        lang2 = EmergingLanguage()
        bag_game = BaselineBagGame(lang2)
        for _ in range(num_rounds):
            bag_game.play_round()
        b_sr = bag_game.successes / max(1, bag_game.games_played)
        bag_srs.append(b_sr)

    avg_h = float(np.mean(hierarchical_srs))
    avg_b = float(np.mean(bag_srs))
    improvement = (avg_h - avg_b) / max(0.01, avg_b) * 100

    print(f"  层级语法: {avg_h:.3f}")
    print(f"  词袋匹配: {avg_b:.3f}")
    print(f"  层级优势: {improvement:.1f}%")

    return {
        'hierarchical_sr': round(avg_h, 4),
        'bag_sr': round(avg_b, 4),
        'improvement_pct': round(improvement, 2),
    }


def experiment_4_cross_agent_convergence(num_pairs: int = 5,
                                          num_rounds: int = 300) -> Dict:
    """
    实验 4：跨 agent 语法收敛

    独立配对是否收敛到相同的语序？
    """
    print("=" * 60)
    print("实验 4：跨 Agent 语法收敛")
    print("=" * 60)

    orders = []
    for pair in range(num_pairs):
        grammar = SlotGrammar()
        lang = EmergingLanguage()
        sp = HierarchicalSpeaker(lang, grammar)
        li = HierarchicalListener(lang, grammar)
        game = HierarchicalGame(sp, li, lang)

        for _ in range(num_rounds):
            game.play_round()

        # 提取每个位置最常见的类别
        order = []
        for slot_idx in sorted(grammar.slot_consistency.keys()):
            counts = grammar.slot_consistency[slot_idx]
            if counts:
                dominant = max(counts, key=counts.get)
                order.append(dominant)
            else:
                order.append('?')

        orders.append(tuple(order))
        sr = game.successes / max(1, game.games_played)
        consistency = grammar.get_consistency()
        print(f"  Pair {pair}: 语序={order}, SR={sr:.3f}, 一致性={consistency:.3f}")

    # 计算收敛率
    from collections import Counter
    order_counts = Counter(orders)
    most_common = order_counts.most_common(1)[0]
    convergence_rate = most_common[1] / num_pairs

    print(f"\n  最常见语序: {most_common[0]} ({most_common[1]}/{num_pairs})")
    print(f"  收敛率: {convergence_rate:.3f}")

    return {
        'convergence_rate': round(convergence_rate, 4),
        'most_common_order': list(most_common[0]),
        'all_orders': [list(o) for o in orders],
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_positional_emergence()
    results['experiment_2'] = experiment_2_recursive_embedding()
    results['experiment_3'] = experiment_3_word_order_vs_bag()
    results['experiment_4'] = experiment_4_cross_agent_convergence()

    output_file = 'hierarchical_syntax_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
