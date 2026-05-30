"""
Phase 83: 谈判与讨价还价语言 —— 谈判标记涌现

核心思想：
谈判标记（"more"/"less"/"fair"/"deal"/"compromise"）从讨价还价压力中涌现。
当 Agent 有不同偏好时，需要通过语言协商达成互利交易。

本阶段测试：
- 谈判标记是否从讨价还价交互中自发涌现
- 有语言的谈判是否比随机交易产生更多双赢结果
- 语言是否帮助 Agent 在更少轮次内达成协议
- "fair" 标记是否与双赢交易相关

理论依据：
- 博弈论（Game Theory）：纳什均衡与互利分配
- 语用学（Pragmatics）：谈判中的言语行为
- 公平感知（Fairness Perception）：合作进化中的利他惩罚
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import EmergingLanguage, COLORS, SHAPES, SIZES, MATERIALS


# ============================================================
# 谈判标记
# ============================================================

NEGOTIATION_MARKERS = {
    'more', 'less', 'fair', 'deal', 'no_deal',
    'compromise', 'better', 'worse', 'exchange', 'split',
}


# ============================================================
# 交易物品
# ============================================================

class TradeItem:
    """交易物品，带名称、价值和属性"""

    _id_counter = 0

    def __init__(self, name: str, value: float, properties: Optional[Dict] = None):
        self.name = name
        self.value = value
        self.properties = properties or {}
        TradeItem._id_counter += 1
        self.id = TradeItem._id_counter

    @classmethod
    def generate_pool(cls, size: int = 8) -> List['TradeItem']:
        """生成随机物品池"""
        items = []
        colors = list(COLORS)
        shapes = list(SHAPES)
        sizes = list(SIZES)
        materials = list(MATERIALS)

        for i in range(size):
            name = f'{random.choice(sizes)}_{random.choice(colors)}_{random.choice(shapes)}'
            value = round(random.uniform(0.1, 1.0), 2)
            props = {
                'color': random.choice(colors),
                'shape': random.choice(shapes),
                'size': random.choice(sizes),
                'material': random.choice(materials),
            }
            items.append(cls(name, value, props))

        return items


# ============================================================
# 谈判 Agent
# ============================================================

class NegotiationAgent:
    """
    带语言和偏好的谈判 Agent

    使用谈判标记构造提议，根据偏好评估交易，
    并从结果中学习标记的有效性。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage,
                 preferences: Optional[Dict[str, float]] = None):
        self.agent_id = agent_id
        self.language = language
        self.preferences: Dict[str, float] = preferences or {}
        self.negotiation_history: List[Dict] = []
        # 标记使用与效果追踪
        self.marker_stats: Dict[str, Dict] = {
            m: {'used': 0, 'success': 0} for m in NEGOTIATION_MARKERS
        }
        # 经验驱动的标记采纳概率
        self.marker_adoption: Dict[str, float] = {m: 0.2 for m in NEGOTIATION_MARKERS}
        # 初始化部分标记有更高的基础概率
        self.marker_adoption['more'] = 0.4
        self.marker_adoption['deal'] = 0.35
        self.marker_adoption['fair'] = 0.3

    def item_value(self, item: TradeItem) -> float:
        """根据自身偏好计算物品价值"""
        base = item.value
        bonus = 0.0
        for prop_key, prop_val in item.properties.items():
            pref_key = f'{prop_key}:{prop_val}'
            if pref_key in self.preferences:
                bonus += self.preferences[pref_key] * 0.3
        return base + bonus

    def propose_offer(self, self_items: List[TradeItem],
                      partner_items: List[TradeItem]) -> Dict:
        """
        生成提议：选择自己给出的物品 + 想要的对方物品

        根据标记策略区分：
        - 自利型（'more', 'better'）
        - 公平型（'fair', 'split'）
        - 让步型（'less', 'compromise'）
        """
        # 计算各物品对自己的价值
        self_values = [(item, self.item_value(item)) for item in self_items]
        partner_values = [(item, self.item_value(item)) for item in partner_items]

        # 策略：基于标记采纳概率选择倾向
        markers_used = []

        # 选择想要对方的物品（按价值排序，取最有价值的）
        partner_values.sort(key=lambda x: x[1], reverse=True)
        want_count = min(random.randint(1, 2), len(partner_values))
        wanted = [item for item, _ in partner_values[:want_count]]

        # 选择自己给出的物品（按价值排序，给价值最低的）
        self_values.sort(key=lambda x: x[1])
        give_count = min(random.randint(1, 2), len(self_values))
        offered = [item for item, _ in self_values[:give_count]]

        # 计算提议价值差异
        offered_value = sum(self.item_value(it) for it in offered)
        wanted_value = sum(self.item_value(it) for it in wanted)
        diff = wanted_value - offered_value

        # 基于价值差异选择标记
        if diff > 0.3:
            # 明显有利 → 自利标记
            if random.random() < self.marker_adoption['more']:
                markers_used.append('more')
            if random.random() < self.marker_adoption['better']:
                markers_used.append('better')
        elif abs(diff) <= 0.3:
            # 接近公平 → 公平标记
            if random.random() < self.marker_adoption['fair']:
                markers_used.append('fair')
            if random.random() < self.marker_adoption['split']:
                markers_used.append('split')
        else:
            # 不利 → 让步标记
            if random.random() < self.marker_adoption['less']:
                markers_used.append('less')
            if random.random() < self.marker_adoption['compromise']:
                markers_used.append('compromise')

        # exchange 标记：提议时总有可能出现
        if random.random() < self.marker_adoption['exchange']:
            markers_used.append('exchange')

        # 记录标记使用
        for m in markers_used:
            self.marker_stats[m]['used'] += 1

        offer = {
            'offered': offered,
            'wanted': wanted,
            'markers': markers_used,
            'proposer': self.agent_id,
            'value_diff': round(diff, 3),
        }

        # 记录到语言系统
        utterance = self._offer_to_utterance(offer)
        self.language.record_usage(utterance, True)

        return offer

    def evaluate_offer(self, offer: Dict) -> float:
        """从自身视角评估提议的价值分数"""
        offered_to_me = offer['wanted']  # 对方想要的 = 我要给出的
        wanted_from_me = offer['offered']  # 对方提供的 = 我要得到的

        # 反转：offer 的 wanted 是对方要的（我要给出的）
        # offer 的 offered 是对方给的（我要得到的）
        gain = sum(self.item_value(it) for it in offer['offered'])
        loss = sum(self.item_value(it) for it in offer['wanted'])

        net = gain - loss
        # 归一化到 [0, 1]
        score = 0.5 + net * 0.5
        return max(0.0, min(1.0, score))

    def respond_to_offer(self, offer: Dict, round_num: int) -> Dict:
        """
        回应提议：接受 / 拒绝 / 反提议

        返回 {'action': 'accept'|'reject'|'counter', 'markers': [...], 'counter_offer': ...}
        """
        score = self.evaluate_offer(offer)

        # 阈值：轮次越多越倾向于接受
        accept_threshold = 0.35 + round_num * 0.05

        if score >= accept_threshold:
            markers = ['deal']
            if score >= 0.6 and random.random() < self.marker_adoption['fair']:
                markers.append('fair')
            self._record_markers(markers, True)
            utterance = markers + [offer.get('proposer', 'unknown')]
            self.language.record_usage(utterance, True)
            return {'action': 'accept', 'markers': markers, 'counter_offer': None}

        elif score < 0.2 or round_num >= 4:
            markers = ['no_deal']
            if score < 0.1 and random.random() < self.marker_adoption['worse']:
                markers.append('worse')
            self._record_markers(markers, False)
            utterance = markers
            self.language.record_usage(utterance, False)
            return {'action': 'reject', 'markers': markers, 'counter_offer': None}

        else:
            markers = ['compromise']
            if random.random() < self.marker_adoption['more']:
                markers.append('more')
            if random.random() < self.marker_adoption['less']:
                markers.append('less')
            self._record_markers(markers, False)
            return {'action': 'counter', 'markers': markers, 'counter_offer': None}

    def make_concession(self, previous_offer: Dict,
                        partner_preferences: Optional[Dict] = None,
                        concession_rate: float = 0.2) -> Dict:
        """
        让步：以 concession_rate 比例向对方立场移动

        增加 20% 的让步量，修改提议中的物品交换。
        """
        offered = list(previous_offer['offered'])
        wanted = list(previous_offer['wanted'])

        # 让步：多给一个或要少一个
        if random.random() < concession_rate and len(offered) < 3:
            # 多给一个物品
            extra = random.choice([it for it in wanted if it not in offered] or wanted)
            if extra not in offered:
                offered.append(extra)

        if random.random() < concession_rate and len(wanted) > 1:
            # 少要一个物品
            wanted.pop(random.randrange(len(wanted)))

        markers = ['compromise', 'less']
        if random.random() < self.marker_adoption['fair']:
            markers.append('fair')

        self._record_markers(markers, True)

        concession_offer = {
            'offered': offered,
            'wanted': wanted,
            'markers': markers,
            'proposer': self.agent_id,
            'value_diff': round(
                sum(self.item_value(it) for it in wanted) -
                sum(self.item_value(it) for it in offered), 3
            ),
        }

        utterance = self._offer_to_utterance(concession_offer)
        self.language.record_usage(utterance, True)

        return concession_offer

    def update_from_outcome(self, utterance: List[str], deal_reached: bool,
                            self_gain: float):
        """根据谈判结果更新标记效果追踪"""
        for sym in utterance:
            if sym in NEGOTIATION_MARKERS:
                if deal_reached:
                    self.marker_stats[sym]['success'] += 1
                    # 成功增加采纳概率
                    self.marker_adoption[sym] = min(
                        self.marker_adoption[sym] + 0.01, 0.9
                    )
                else:
                    # 失败微降
                    self.marker_adoption[sym] = max(
                        self.marker_adoption[sym] - 0.003, 0.05
                    )

        self.negotiation_history.append({
            'utterance': utterance,
            'deal_reached': deal_reached,
            'self_gain': round(self_gain, 3),
        })

    def get_marker_summary(self) -> Dict[str, float]:
        """获取标记成功率"""
        result = {}
        for m, stats in self.marker_stats.items():
            if stats['used'] > 0:
                result[m] = round(stats['success'] / stats['used'], 3)
            else:
                result[m] = 0.0
        return result

    def _record_markers(self, markers: List[str], success: bool):
        """记录标记使用"""
        for m in markers:
            if m in self.marker_stats:
                self.marker_stats[m]['used'] += 1
                if success:
                    self.marker_stats[m]['success'] += 1

    def _offer_to_utterance(self, offer: Dict) -> List[str]:
        """将提议转换为符号序列"""
        tokens = list(offer.get('markers', []))
        for item in offer.get('offered', []):
            tokens.append(item.name)
        for item in offer.get('wanted', []):
            tokens.append(item.name)
        return tokens


# ============================================================
# 谈判游戏
# ============================================================

class NegotiationGame:
    """
    双 Agent 谈判游戏

    两个 Agent 有不同偏好，通过多轮讨价还价达成交易。
    追踪：协议达成率、标记使用、双方收益、双赢比例。
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.total_negotiations = 0
        self.deals_reached = 0
        self.win_win_deals = 0
        self.marker_usage: Dict[str, int] = defaultdict(int)
        self.round_distribution: List[int] = []  # 每次谈判用了几轮
        self.agent_gains: List[Tuple[float, float]] = []

    def play_negotiation(self, agent_a: NegotiationAgent,
                         agent_b: NegotiationAgent,
                         items_a: List[TradeItem],
                         items_b: List[TradeItem]) -> Dict:
        """
        进行一次完整谈判（最多 5 轮）

        Returns:
            谈判结果字典
        """
        self.total_negotiations += 1
        max_rounds = 5

        current_offer = None
        rounds_taken = 0
        deal_reached = False
        final_offer = None

        # 计算 Agent 的最大可能收益（用于判断双赢）
        max_gain_a = sum(agent_a.item_value(it) for it in items_b)
        max_gain_b = sum(agent_b.item_value(it) for it in items_a)

        for rnd in range(max_rounds):
            rounds_taken = rnd + 1

            if rnd == 0:
                # Agent A 提议
                current_offer = agent_a.propose_offer(items_a, items_b)
            else:
                # 交替提议
                if rnd % 2 == 1:
                    response = agent_b.respond_to_offer(current_offer, rnd)
                    self._track_markers(response.get('markers', []))

                    if response['action'] == 'accept':
                        deal_reached = True
                        final_offer = current_offer
                        break
                    elif response['action'] == 'reject':
                        break
                    else:
                        # 反提议
                        current_offer = agent_b.make_concession(
                            current_offer, concession_rate=0.2 + rnd * 0.05
                        )
                else:
                    response = agent_a.respond_to_offer(current_offer, rnd)
                    self._track_markers(response.get('markers', []))

                    if response['action'] == 'accept':
                        deal_reached = True
                        final_offer = current_offer
                        break
                    elif response['action'] == 'reject':
                        break
                    else:
                        current_offer = agent_a.make_concession(
                            current_offer, concession_rate=0.2 + rnd * 0.05
                        )

            self._track_markers(current_offer.get('markers', []))

        # 计算收益
        gain_a = 0.0
        gain_b = 0.0
        if deal_reached and final_offer:
            # final_offer 的 offered 是提议者给出的，wanted 是提议者想要的
            # 如果 A 提议：offered = A 给的，wanted = A 要的（B 给的）
            gain_a = sum(agent_a.item_value(it) for it in final_offer['wanted']) - \
                     sum(agent_a.item_value(it) for it in final_offer['offered'])
            gain_b = sum(agent_b.item_value(it) for it in final_offer['offered']) - \
                     sum(agent_b.item_value(it) for it in final_offer['wanted'])
            self.deals_reached += 1

        # 归一化收益
        norm_gain_a = gain_a / max(max_gain_a, 0.01) if deal_reached else 0.0
        norm_gain_b = gain_b / max(max_gain_b, 0.01) if deal_reached else 0.0

        # 判断双赢
        win_win = deal_reached and norm_gain_a > 0.5 and norm_gain_b > 0.5
        if win_win:
            self.win_win_deals += 1

        self.agent_gains.append((norm_gain_a, norm_gain_b))
        self.round_distribution.append(rounds_taken)

        # 更新 Agent 的结果追踪
        utterance_a = agent_a._offer_to_utterance(current_offer) if current_offer else []
        utterance_b = []
        agent_a.update_from_outcome(utterance_a, deal_reached, norm_gain_a)
        agent_b.update_from_outcome(utterance_b, deal_reached, norm_gain_b)

        return {
            'deal_reached': deal_reached,
            'win_win': win_win,
            'rounds': rounds_taken,
            'gain_a': round(norm_gain_a, 3),
            'gain_b': round(norm_gain_b, 3),
        }

    def _track_markers(self, markers: List[str]):
        """追踪标记使用"""
        for m in markers:
            if m in NEGOTIATION_MARKERS:
                self.marker_usage[m] += 1

    def get_stats(self) -> Dict:
        """获取谈判统计"""
        deal_rate = self.deals_reached / max(self.total_negotiations, 1)
        winwin_rate = self.win_win_deals / max(self.total_negotiations, 1)
        avg_gain_a = np.mean([g[0] for g in self.agent_gains]) if self.agent_gains else 0
        avg_gain_b = np.mean([g[1] for g in self.agent_gains]) if self.agent_gains else 0
        avg_rounds = np.mean(self.round_distribution) if self.round_distribution else 0

        return {
            'deal_rate': round(deal_rate, 3),
            'win_win_rate': round(winwin_rate, 3),
            'avg_gain_a': round(avg_gain_a, 3),
            'avg_gain_b': round(avg_gain_b, 3),
            'avg_rounds': round(avg_rounds, 2),
            'total_negotiations': self.total_negotiations,
            'marker_usage': dict(self.marker_usage),
        }


# ============================================================
# 基线游戏（无语言，随机接受）
# ============================================================

class BaselineNoNegotiationGame:
    """
    对照组：无谈判语言，随机接受提议

    测试假设：有语言的谈判 > 随机交易
    """

    def __init__(self):
        self.total_negotiations = 0
        self.deals_reached = 0
        self.win_win_deals = 0
        self.agent_gains: List[Tuple[float, float]] = []
        self.round_distribution: List[int] = []

    def play_negotiation(self, pref_a: Dict[str, float],
                         pref_b: Dict[str, float],
                         items_a: List[TradeItem],
                         items_b: List[TradeItem]) -> Dict:
        """随机交易：提议后以 50% 概率接受"""
        self.total_negotiations += 1

        # 随机提议
        offered = random.sample(items_a, min(random.randint(1, 2), len(items_a)))
        wanted = random.sample(items_b, min(random.randint(1, 2), len(items_b)))

        # 随机接受
        deal_reached = random.random() < 0.5
        rounds_taken = random.randint(1, 5)

        gain_a = 0.0
        gain_b = 0.0

        # 简单偏好函数
        def _item_val(item, prefs):
            base = item.value
            for pk, pv in item.properties.items():
                key = f'{pk}:{pv}'
                if key in prefs:
                    base += prefs[key] * 0.3
            return base

        max_gain_a = sum(_item_val(it, pref_a) for it in items_b)
        max_gain_b = sum(_item_val(it, pref_b) for it in items_a)

        if deal_reached:
            gain_a = sum(_item_val(it, pref_a) for it in wanted) - \
                     sum(_item_val(it, pref_a) for it in offered)
            gain_b = sum(_item_val(it, pref_b) for it in offered) - \
                     sum(_item_val(it, pref_b) for it in wanted)
            self.deals_reached += 1

        norm_gain_a = gain_a / max(max_gain_a, 0.01) if deal_reached else 0.0
        norm_gain_b = gain_b / max(max_gain_b, 0.01) if deal_reached else 0.0

        win_win = deal_reached and norm_gain_a > 0.5 and norm_gain_b > 0.5
        if win_win:
            self.win_win_deals += 1

        self.agent_gains.append((norm_gain_a, norm_gain_b))
        self.round_distribution.append(rounds_taken)

        return {
            'deal_reached': deal_reached,
            'win_win': win_win,
            'rounds': rounds_taken,
            'gain_a': round(norm_gain_a, 3),
            'gain_b': round(norm_gain_b, 3),
        }

    def get_stats(self) -> Dict:
        deal_rate = self.deals_reached / max(self.total_negotiations, 1)
        winwin_rate = self.win_win_deals / max(self.total_negotiations, 1)
        avg_gain_a = np.mean([g[0] for g in self.agent_gains]) if self.agent_gains else 0
        avg_gain_b = np.mean([g[1] for g in self.agent_gains]) if self.agent_gains else 0
        avg_rounds = np.mean(self.round_distribution) if self.round_distribution else 0

        return {
            'deal_rate': round(deal_rate, 3),
            'win_win_rate': round(winwin_rate, 3),
            'avg_gain_a': round(avg_gain_a, 3),
            'avg_gain_b': round(avg_gain_b, 3),
            'avg_rounds': round(avg_rounds, 2),
        }


# ============================================================
# 辅助函数
# ============================================================

def _generate_preferences() -> Dict[str, float]:
    """生成随机偏好"""
    prefs = {}
    for color in random.sample(list(COLORS), 3):
        prefs[f'color:{color}'] = random.uniform(0.2, 0.8)
    for shape in random.sample(list(SHAPES), 3):
        prefs[f'shape:{shape}'] = random.uniform(0.2, 0.8)
    for mat in random.sample(list(MATERIALS), 2):
        prefs[f'material:{mat}'] = random.uniform(0.1, 0.6)
    return prefs


# ============================================================
# 实验 1：谈判标记涌现
# ============================================================

def experiment_1_negotiation_markers(num_rounds: int = 300):
    """
    实验 1：300 次谈判，追踪谈判标记的涌现

    预期：7-10 个标记从交互中涌现。
    """
    print("=" * 60)
    print("实验 1: 谈判标记涌现")
    print("=" * 60)

    language = EmergingLanguage()
    game = NegotiationGame(language)

    agent_a = NegotiationAgent(0, EmergingLanguage(), _generate_preferences())
    agent_b = NegotiationAgent(1, EmergingLanguage(), _generate_preferences())

    marker_emergence = {m: -1 for m in NEGOTIATION_MARKERS}
    snapshots = []

    for r in range(num_rounds):
        # 每次谈判生成新物品池
        pool = TradeItem.generate_pool(8)
        items_a = pool[:4]
        items_b = pool[4:]

        game.play_negotiation(agent_a, agent_b, items_a, items_b)

        # 检查标记涌现
        for m in NEGOTIATION_MARKERS:
            if marker_emergence[m] == -1 and game.marker_usage.get(m, 0) > 0:
                marker_emergence[m] = r

        # 每 50 轮打印快照
        if (r + 1) % 50 == 0:
            stats = game.get_stats()
            snapshot = {
                'round': r + 1,
                'deal_rate': stats['deal_rate'],
                'win_win_rate': stats['win_win_rate'],
                'marker_usage': {m: game.marker_usage.get(m, 0)
                                 for m in NEGOTIATION_MARKERS},
            }
            snapshots.append(snapshot)
            print(f"  Round {r+1}: "
                  f"成交率={stats['deal_rate']:.3f}, "
                  f"双赢率={stats['win_win_rate']:.3f}, "
                  f"已涌现标记={sum(1 for v in marker_emergence.values() if v >= 0)}")

    # 最终统计
    final_stats = game.get_stats()
    emerged = sum(1 for v in marker_emergence.values() if v >= 0)

    print(f"\n最终结果:")
    print(f"  成交率: {final_stats['deal_rate']:.3f}")
    print(f"  双赢率: {final_stats['win_win_rate']:.3f}")
    print(f"  涌现标记数: {emerged}")
    print(f"  标记涌现轮次: {marker_emergence}")

    return {
        'final_deal_rate': final_stats['deal_rate'],
        'final_win_win_rate': final_stats['win_win_rate'],
        'emerged_markers': emerged,
        'marker_emergence_rounds': marker_emergence,
        'final_marker_usage': dict(game.marker_usage),
        'snapshots': snapshots,
    }


# ============================================================
# 实验 2：有语言谈判 vs 随机交易
# ============================================================

def experiment_2_negotiation_vs_random(num_rounds: int = 200, num_runs: int = 5):
    """
    实验 2：对比 NegotiationGame vs BaselineNoNegotiationGame

    追踪：成交率、双赢率、平均收益。
    预期：谈判双赢率 ~60-70%，随机 ~35-45%。
    """
    print(f"\n{'=' * 60}")
    print("实验 2: 有语言谈判 vs 随机交易")
    print(f"{'=' * 60}")

    negotiation_results = []
    baseline_results = []

    for run in range(num_runs):
        # --- 谈判组 ---
        lang_neg = EmergingLanguage()
        game_neg = NegotiationGame(lang_neg)
        pref_a = _generate_preferences()
        pref_b = _generate_preferences()
        agent_a = NegotiationAgent(0, EmergingLanguage(), pref_a)
        agent_b = NegotiationAgent(1, EmergingLanguage(), pref_b)

        for r in range(num_rounds):
            pool = TradeItem.generate_pool(8)
            game_neg.play_negotiation(agent_a, agent_b, pool[:4], pool[4:])

        neg_stats = game_neg.get_stats()
        negotiation_results.append(neg_stats)

        # --- 基线组 ---
        game_base = BaselineNoNegotiationGame()

        for r in range(num_rounds):
            pool = TradeItem.generate_pool(8)
            pref_a_run = _generate_preferences()
            pref_b_run = _generate_preferences()
            game_base.play_negotiation(pref_a_run, pref_b_run, pool[:4], pool[4:])

        base_stats = game_base.get_stats()
        baseline_results.append(base_stats)

        print(f"  Run {run+1}/{num_runs}: "
              f"谈判 成交={neg_stats['deal_rate']:.3f} 双赢={neg_stats['win_win_rate']:.3f}, "
              f"随机 成交={base_stats['deal_rate']:.3f} 双赢={base_stats['win_win_rate']:.3f}")

    # 汇总
    avg_neg_deal = np.mean([r['deal_rate'] for r in negotiation_results])
    avg_neg_winwin = np.mean([r['win_win_rate'] for r in negotiation_results])
    avg_neg_gain = np.mean([
        (r['avg_gain_a'] + r['avg_gain_b']) / 2 for r in negotiation_results
    ])
    avg_base_deal = np.mean([r['deal_rate'] for r in baseline_results])
    avg_base_winwin = np.mean([r['win_win_rate'] for r in baseline_results])
    avg_base_gain = np.mean([
        (r['avg_gain_a'] + r['avg_gain_b']) / 2 for r in baseline_results
    ])

    print(f"\n最终对比:")
    print(f"  谈判组: 成交={avg_neg_deal:.3f}, 双赢={avg_neg_winwin:.3f}, 平均收益={avg_neg_gain:.3f}")
    print(f"  随机组: 成交={avg_base_deal:.3f}, 双赢={avg_base_winwin:.3f}, 平均收益={avg_base_gain:.3f}")
    print(f"  双赢优势: +{avg_neg_winwin - avg_base_winwin:.3f}")

    return {
        'negotiation_avg_deal_rate': round(avg_neg_deal, 3),
        'negotiation_avg_winwin_rate': round(avg_neg_winwin, 3),
        'negotiation_avg_gain': round(avg_neg_gain, 3),
        'baseline_avg_deal_rate': round(avg_base_deal, 3),
        'baseline_avg_winwin_rate': round(avg_base_winwin, 3),
        'baseline_avg_gain': round(avg_base_gain, 3),
        'winwin_advantage': round(avg_neg_winwin - avg_base_winwin, 3),
        'per_run': {
            'negotiation': negotiation_results,
            'baseline': baseline_results,
        },
    }


# ============================================================
# 实验 3：让步模式
# ============================================================

def experiment_3_concession_patterns(num_rounds: int = 200):
    """
    实验 3：追踪让步行为随谈判轮次的变化

    语言是否帮助 Agent 更快达成协议？
    预期：有语言时平均 2-3 轮达成协议，无语言 4-5 轮。
    """
    print(f"\n{'=' * 60}")
    print("实验 3: 让步模式与协议速度")
    print(f"{'=' * 60}")

    language = EmergingLanguage()
    game = NegotiationGame(language)

    agent_a = NegotiationAgent(0, EmergingLanguage(), _generate_preferences())
    agent_b = NegotiationAgent(1, EmergingLanguage(), _generate_preferences())

    rounds_with_language = []
    concession_marker_usage = defaultdict(int)
    snapshots = []

    for r in range(num_rounds):
        pool = TradeItem.generate_pool(8)
        result = game.play_negotiation(agent_a, agent_b, pool[:4], pool[4:])

        if result['deal_reached']:
            rounds_with_language.append(result['rounds'])

        # 追踪让步标记
        for m in ['compromise', 'less', 'fair']:
            concession_marker_usage[m] += game.marker_usage.get(m, 0)

        # 每 50 轮打印
        if (r + 1) % 50 == 0:
            stats = game.get_stats()
            avg_r = np.mean(rounds_with_language) if rounds_with_language else 0
            snapshot = {
                'round': r + 1,
                'avg_rounds_to_deal': round(avg_r, 2),
                'deal_rate': stats['deal_rate'],
                'concession_markers': dict(concession_marker_usage),
            }
            snapshots.append(snapshot)
            print(f"  Round {r+1}: "
                  f"平均轮次={avg_r:.2f}, "
                  f"成交率={stats['deal_rate']:.3f}, "
                  f"让步标记={dict(concession_marker_usage)}")

    # 基线对照：无语言随机交易的轮次
    baseline_rounds = []
    for _ in range(num_rounds):
        pool = TradeItem.generate_pool(8)
        game_base = BaselineNoNegotiationGame()
        pref_a = _generate_preferences()
        pref_b = _generate_preferences()
        result = game_base.play_negotiation(pref_a, pref_b, pool[:4], pool[4:])
        if result['deal_reached']:
            baseline_rounds.append(result['rounds'])

    avg_lang_rounds = np.mean(rounds_with_language) if rounds_with_language else 0
    avg_base_rounds = np.mean(baseline_rounds) if baseline_rounds else 0

    print(f"\n最终结果:")
    print(f"  有语言 平均达成轮次: {avg_lang_rounds:.2f}")
    print(f"  无语言 平均达成轮次: {avg_base_rounds:.2f}")
    print(f"  轮次减少: {avg_base_rounds - avg_lang_rounds:.2f}")

    return {
        'with_language_avg_rounds': round(avg_lang_rounds, 2),
        'without_language_avg_rounds': round(avg_base_rounds, 2),
        'round_reduction': round(avg_base_rounds - avg_lang_rounds, 2),
        'concession_marker_usage': dict(concession_marker_usage),
        'snapshots': snapshots,
    }


# ============================================================
# 实验 4：公平感知
# ============================================================

def experiment_4_fairness_perception(num_rounds: int = 200, num_runs: int = 3):
    """
    实验 4：Agent 对交易公平性的感知

    追踪 'fair' 标记的使用是否与双赢交易相关。
    预期：'fair' 标记在双赢交易中出现频率更高。
    """
    print(f"\n{'=' * 60}")
    print("实验 4: 公平感知与双赢交易")
    print(f"{'=' * 60}")

    run_results = []

    for run in range(num_runs):
        language = EmergingLanguage()
        game = NegotiationGame(language)

        pref_a = _generate_preferences()
        pref_b = _generate_preferences()
        agent_a = NegotiationAgent(0, EmergingLanguage(), pref_a)
        agent_b = NegotiationAgent(1, EmergingLanguage(), pref_b)

        fair_in_winwin = 0
        fair_in_non_winwin = 0
        winwin_count = 0
        non_winwin_deal_count = 0

        snapshots = []

        for r in range(num_rounds):
            pool = TradeItem.generate_pool(8)
            result = game.play_negotiation(agent_a, agent_b, pool[:4], pool[4:])

            # 检查本轮是否使用了 'fair' 标记
            used_fair = game.marker_usage.get('fair', 0) > 0

            if result['win_win']:
                winwin_count += 1
                if used_fair:
                    fair_in_winwin += 1
            elif result['deal_reached']:
                non_winwin_deal_count += 1
                if used_fair:
                    fair_in_non_winwin += 1

            # 每 50 轮打印
            if (r + 1) % 50 == 0:
                stats = game.get_stats()
                fair_rate_winwin = fair_in_winwin / max(winwin_count, 1)
                fair_rate_other = fair_in_non_winwin / max(non_winwin_deal_count, 1)
                snapshot = {
                    'round': r + 1,
                    'winwin_rate': stats['win_win_rate'],
                    'fair_in_winwin_rate': round(fair_rate_winwin, 3),
                    'fair_in_other_rate': round(fair_rate_other, 3),
                    'fair_marker_count': game.marker_usage.get('fair', 0),
                }
                snapshots.append(snapshot)
                print(f"  Run {run+1} Round {r+1}: "
                      f"双赢率={stats['win_win_rate']:.3f}, "
                      f"fair在双赢中={fair_rate_winwin:.3f}, "
                      f"fair在非双赢中={fair_rate_other:.3f}")

        fair_rate_winwin = fair_in_winwin / max(winwin_count, 1)
        fair_rate_other = fair_in_non_winwin / max(non_winwin_deal_count, 1)

        marker_summary_a = agent_a.get_marker_summary()
        marker_summary_b = agent_b.get_marker_summary()

        run_results.append({
            'fair_rate_in_winwin': round(fair_rate_winwin, 3),
            'fair_rate_in_other': round(fair_rate_other, 3),
            'winwin_count': winwin_count,
            'non_winwin_deal_count': non_winwin_deal_count,
            'fair_correlation': round(fair_rate_winwin - fair_rate_other, 3),
            'marker_summary_a': marker_summary_a,
            'marker_summary_b': marker_summary_b,
            'snapshots': snapshots,
        })

    # 汇总
    avg_fair_winwin = np.mean([r['fair_rate_in_winwin'] for r in run_results])
    avg_fair_other = np.mean([r['fair_rate_in_other'] for r in run_results])
    avg_correlation = np.mean([r['fair_correlation'] for r in run_results])

    print(f"\n最终结果:")
    print(f"  fair 在双赢交易中出现率: {avg_fair_winwin:.3f}")
    print(f"  fair 在非双赢交易中出现率: {avg_fair_other:.3f}")
    print(f"  fair-双赢相关性: {avg_correlation:.3f}")

    return {
        'avg_fair_rate_in_winwin': round(avg_fair_winwin, 3),
        'avg_fair_rate_in_other': round(avg_fair_other, 3),
        'fair_winwin_correlation': round(avg_correlation, 3),
        'per_run': run_results,
    }


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_negotiation_markers()
    results['experiment_2'] = experiment_2_negotiation_vs_random()
    results['experiment_3'] = experiment_3_concession_patterns()
    results['experiment_4'] = experiment_4_fairness_perception()

    with open('negotiation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 negotiation_results.json")
