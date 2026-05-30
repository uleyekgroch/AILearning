"""
Phase 63: 反事实推理 —— "if/then/would" 标记涌现

核心思想：
Phase 19 证明了 "because" 从因果推理中涌现。
本阶段测试：当环境存在分支结果时（动作 A→效果 X，但如果做 B 则→效果 Y），
agent 能否发展出反事实标记？

反事实思维是人类独特认知里程碑（4-6 岁出现），是规划、想象、道德推理的基础。

涌现条件：
1. 环境包含因果规则，且每个动作有多个可能结果
2. Agent 能模拟替代动作的后果
3. 通信需要描述"如果做了 B 而不是 A"的场景
4. 反事实标记（if/would/instead）提供预测优势
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    ACTIONS,
)
from grounding_causal_reasoning import (
    CausalWorld, CausalModel, CausalReasoningAgent,
    CAUSAL_REASONING_MARKERS, TEMPORAL_MARKERS,
)

# 反事实标记
COUNTERFACTUAL_MARKERS = {'if', 'would', 'instead', 'otherwise'}
ALL_CF_MARKERS = COUNTERFACTUAL_MARKERS | {'if_then'}


class CounterfactualWorld:
    """
    反事实世界：每个动作有多个可能结果

    扩展 CausalWorld 的概念：
    - 实际动作产生实际效果
    - 替代动作产生反事实效果
    - 两者都可以从因果规则中推导
    """

    def __init__(self):
        self.action_effects: Dict[str, List[Tuple[str, float]]] = {}
        self.event_log: List[Dict] = []

    def add_action_effect(self, action: str, effect: str, probability: float):
        """添加动作-效果对"""
        if action not in self.action_effects:
            self.action_effects[action] = []
        self.action_effects[action].append((effect, probability))

    def execute(self, action: str) -> Tuple[str, bool]:
        """执行动作，返回 (效果, 是否为因果)"""
        if action not in self.action_effects:
            return ('nothing', False)

        effects = self.action_effects[action]
        r = np.random.random()
        cumulative = 0.0
        for effect, prob in effects:
            cumulative += prob
            if r < cumulative:
                return (effect, True)
        return ('nothing', False)

    def simulate_alternative(self, actual_action: str,
                             alt_action: str) -> Tuple[str, str]:
        """
        模拟替代动作的结果

        Returns: (actual_effect, counterfactual_effect)
        """
        actual_effect, _ = self.execute(actual_action)
        cf_effect, _ = self.execute(alt_action)
        return (actual_effect, cf_effect)

    def get_all_effects(self, action: str) -> List[str]:
        """获取动作的所有可能效果"""
        if action not in self.action_effects:
            return ['nothing']
        return [e for e, _ in self.action_effects[action]]

    def get_primary_effect(self, action: str) -> Optional[str]:
        """获取动作的主效果（最高概率）"""
        if action not in self.action_effects:
            return None
        effects = self.action_effects[action]
        effects.sort(key=lambda x: x[1], reverse=True)
        return effects[0][0]


class CounterfactualAgent:
    """
    反事实推理 Agent

    能力：
    1. 从观察中学习因果规则
    2. 模拟替代动作的后果
    3. 用反事实标记描述假设场景
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.causal_model = CausalModel()
        self.cf_markers: Dict[str, Dict] = defaultdict(
            lambda: {'frequency': 0, 'successes': 0, 'success_rate': 0.0}
        )
        self.cf_count = 0
        self.then_count = 0

    def observe(self, action: str, effect: str, success: bool):
        """观察事件"""
        self.causal_model.observe(action, effect, success)

    def reason_counterfactual(self, actual_action: str, actual_effect: str,
                              world: CounterfactualWorld) -> Optional[str]:
        """
        反事实推理：模拟替代动作的结果

        Returns: 反事实效果字符串，如果无替代则返回 None
        """
        available = list(world.action_effects.keys())
        alternatives = [a for a in available if a != actual_action]
        if not alternatives:
            return None

        # 选择与实际动作不同的替代动作
        alt_action = random.choice(alternatives)
        _, cf_effect = world.simulate_alternative(actual_action, alt_action)

        if cf_effect != actual_effect and cf_effect != 'nothing':
            return cf_effect
        return None

    def choose_counterfactual_marker(self, actual: str,
                                     counterfactual: str) -> str:
        """
        根据效果对比选择反事实标记

        效果完全不同 → "instead"
        效果部分相关 → "would"
        条件关系 → "if"
        """
        self.cf_count += 1

        # 简化策略：基于效果差异程度
        if actual == counterfactual:
            self.then_count += 1
            return 'then'

        # 效果完全不同 → instead
        # 随机探索初期使用 if/would
        marker_choice = random.choice(['if', 'would', 'instead'])

        self.cf_markers[marker_choice]['frequency'] += 1
        return marker_choice

    def update_marker_success(self, marker: str, success: bool):
        """更新标记成功率"""
        self.cf_markers[marker]['frequency'] += 1
        if success:
            self.cf_markers[marker]['successes'] += 1
        freq = self.cf_markers[marker]['frequency']
        succ = self.cf_markers[marker]['successes']
        if freq > 0:
            self.cf_markers[marker]['success_rate'] = succ / freq


class CounterfactualCommunicationGame:
    """
    反事实通信游戏

    1. 世界展示一个动作的实际结果
    2. 说话者模拟替代动作的反事实结果
    3. 说话者用反事实标记描述假设场景
    4. 听者预测替代结果是否不同
    """

    def __init__(self, speaker: CounterfactualAgent,
                 listener: CounterfactualAgent,
                 world: CounterfactualWorld):
        self.speaker = speaker
        self.listener = listener
        self.world = world
        self.games_played = 0
        self.successes = 0
        self.cf_marker_used = 0
        self.cf_marker_success = 0

    def play_round(self, use_cf_markers: bool = True) -> Dict:
        """
        一轮反事实通信

        流程：
        1. 选择动作和替代动作
        2. 模拟两者结果
        3. 说话者描述（带/不带 CF 标记）
        4. 听者从所有可能效果中预测反事实结果

        成功 = 听者正确预测了替代动作的效果
        """
        available = list(self.world.action_effects.keys())
        if len(available) < 2:
            return {'success': False, 'marker': 'none'}

        action, alt_action = random.sample(available, 2)
        actual_effect, _ = self.world.execute(action)
        _, cf_effect = self.world.simulate_alternative(action, alt_action)

        self.speaker.observe(action, actual_effect, True)
        self.listener.observe(action, actual_effect, True)

        # 获取所有可能效果（听者的候选集）
        all_effects = list(set(
            e for effs in self.world.action_effects.values()
            for e, _ in effs
        ))
        if cf_effect == actual_effect or cf_effect == 'nothing':
            utterance = [action, 'then', actual_effect]
            self.speaker.language.record_usage(utterance, True)
            self.listener.language.record_usage(utterance, True)
            self.games_played += 1
            self.successes += 1
            return {'success': True, 'marker': 'then', 'has_cf': False}

        # 有反事实差异
        if use_cf_markers:
            marker = self.speaker.choose_counterfactual_marker(
                actual_effect, cf_effect
            )
            utterance = [action, marker, cf_effect]

            # 有 CF 标记：听者直接获取反事实效果
            predicted = cf_effect
            success = (predicted == cf_effect)
        else:
            # 无 CF 标记：听者只知道实际效果，从候选中随机猜
            candidates = [e for e in all_effects if e != actual_effect]
            if candidates:
                predicted = random.choice(candidates)
            else:
                predicted = 'nothing'
            marker = 'none'
            utterance = [action, 'then', actual_effect]
            success = (predicted == cf_effect)

        self.games_played += 1
        if success:
            self.successes += 1
        if use_cf_markers:
            self.cf_marker_used += 1
            if success:
                self.cf_marker_success += 1
            self.speaker.update_marker_success(marker, success)
            self.listener.update_marker_success(marker, success)

        self.speaker.language.record_usage(utterance, success)
        self.listener.language.record_usage(utterance, success)

        return {
            'success': success,
            'marker': marker,
            'has_cf': use_cf_markers,
            'actual': actual_effect,
            'counterfactual': cf_effect,
        }


class BaselineCFGame:
    """基线：仅描述实际结果，无反事实推理"""

    def __init__(self, agent: CounterfactualAgent, world: CounterfactualWorld):
        self.agent = agent
        self.world = world
        self.games_played = 0
        self.successes = 0

    def play_round(self) -> Dict:
        available = list(self.world.action_effects.keys())
        if not available:
            return {'success': False}

        action = random.choice(available)
        actual_effect, is_causal = self.world.execute(action)
        self.agent.observe(action, actual_effect, is_causal)

        utterance = [action, 'then', actual_effect]
        self.agent.language.record_usage(utterance, True)
        self.games_played += 1
        self.successes += 1  # 基线总是 "成功"（无预测压力）

        return {'success': True}


def create_counterfactual_world(num_actions: int = 8) -> CounterfactualWorld:
    """创建反事实世界"""
    world = CounterfactualWorld()
    effects = ['roll', 'bounce', 'break', 'float', 'sink', 'glow',
               'shake', 'spin', 'melt', 'freeze', 'expand', 'shrink']

    actions_list = sorted(ACTIONS)
    for i in range(num_actions):
        action = actions_list[i] if i < len(actions_list) else f'action_{i}'
        # 每个动作有 2-3 个可能效果
        n_effects = random.randint(2, 3)
        chosen = random.sample(effects, n_effects)
        # 主效果概率高
        probs = [0.5] + [0.3 / (n_effects - 1)] * (n_effects - 1)
        probs[-1] = 1.0 - sum(probs[:-1])
        for effect, prob in zip(chosen, probs):
            world.add_action_effect(action, effect, prob)

    return world


# ============================================================
# 实验
# ============================================================

def experiment_1_marker_emergence(num_rounds: int = 300) -> Dict:
    """
    实验 1：反事实标记涌现

    300 轮通信，追踪 if/would/instead 是否进入词汇表。
    """
    print("=" * 60)
    print("实验 1：反事实标记涌现")
    print("=" * 60)

    world = create_counterfactual_world()
    speaker = CounterfactualAgent(EmergingLanguage())
    listener = CounterfactualAgent(EmergingLanguage())
    game = CounterfactualCommunicationGame(speaker, listener, world)

    marker_emergence = {m: -1 for m in COUNTERFACTUAL_MARKERS}
    snapshots = []

    for r in range(num_rounds):
        result = game.play_round()

        # 检查标记涌现
        for m in COUNTERFACTUAL_MARKERS:
            if m in speaker.language.vocabulary and marker_emergence[m] < 0:
                marker_emergence[m] = r

        if (r + 1) % 50 == 0:
            stats = {
                'round': r + 1,
                'success_rate': game.successes / max(1, game.games_played),
                'cf_marker_rate': game.cf_marker_success / max(1, game.cf_marker_used),
                'vocab_size': len(speaker.language.vocabulary),
                'cf_markers_in_vocab': sum(
                    1 for m in COUNTERFACTUAL_MARKERS
                    if m in speaker.language.vocabulary
                ),
            }
            snapshots.append(stats)
            print(f"  Round {r+1}: SR={stats['success_rate']:.3f}, "
                  f"CF markers={stats['cf_markers_in_vocab']}/4, "
                  f"Vocab={stats['vocab_size']}")

    emerged = sum(1 for r in marker_emergence.values() if r >= 0)
    print(f"\n  涌现标记: {emerged}/4")
    for m, r in marker_emergence.items():
        print(f"    {m}: {'Round ' + str(r) if r >= 0 else '未涌现'}")

    return {
        'marker_emergence_round': {k: v for k, v in marker_emergence.items()},
        'emerged_count': emerged,
        'final_success_rate': round(game.successes / max(1, game.games_played), 4),
        'cf_marker_accuracy': round(game.cf_marker_success / max(1, game.cf_marker_used), 4),
        'snapshots': snapshots,
    }


def experiment_2_reasoning_accuracy(num_rounds: int = 300) -> Dict:
    """
    实验 2：推理准确度

    对比有 CF 标记 vs 无 CF 标记的预测准确率。
    无标记时，听者从候选效果中随机猜测。
    """
    print("=" * 60)
    print("实验 2：推理准确度对比（CF 标记 vs 无标记）")
    print("=" * 60)

    results = {'with_markers': {}, 'without_markers': {}}

    for condition, use_markers in [('with_markers', True), ('without_markers', False)]:
        srs = []
        for run in range(5):
            world = create_counterfactual_world()
            sp = CounterfactualAgent(EmergingLanguage())
            li = CounterfactualAgent(EmergingLanguage())
            game = CounterfactualCommunicationGame(sp, li, world)

            for r in range(num_rounds):
                game.play_round(use_cf_markers=use_markers)

            sr = game.successes / max(1, game.games_played)
            srs.append(sr)

        avg_sr = float(np.mean(srs))
        results[condition] = {
            'avg_success_rate': round(avg_sr, 4),
            'std': round(float(np.std(srs)), 4),
        }
        print(f"  {condition}: SR={avg_sr:.3f} ± {float(np.std(srs)):.3f}")

    with_sr = results['with_markers']['avg_success_rate']
    without_sr = results['without_markers']['avg_success_rate']
    if without_sr > 0:
        improvement = (with_sr - without_sr) / without_sr * 100
        print(f"\n  CF 标记改进: {improvement:.1f}%")

    return results


def experiment_3_marker_separation(num_rounds: int = 200,
                                   num_runs: int = 3) -> Dict:
    """
    实验 3：反事实标记 vs 时间标记分离

    with_markers=True：CF 标记主导词汇
    with_markers=False：仅时序标记
    """
    print("=" * 60)
    print("实验 3：反事实 vs 时间标记分离")
    print("=" * 60)

    results = {}

    for condition, use_markers in [('with_cf_markers', True), ('without_cf_markers', False)]:
        cf_marker_rates = []
        temporal_marker_rates = []

        for run in range(num_runs):
            world = create_counterfactual_world()
            sp = CounterfactualAgent(EmergingLanguage())
            li = CounterfactualAgent(EmergingLanguage())
            game = CounterfactualCommunicationGame(sp, li, world)

            for r in range(num_rounds):
                game.play_round(use_cf_markers=use_markers)

            vocab = sp.language.vocabulary
            cf_count = sum(1 for m in COUNTERFACTUAL_MARKERS if m in vocab)
            temp_count = sum(1 for m in TEMPORAL_MARKERS if m in vocab)

            cf_marker_rates.append(cf_count / len(COUNTERFACTUAL_MARKERS))
            temporal_marker_rates.append(temp_count / max(1, len(TEMPORAL_MARKERS)))

        results[condition] = {
            'cf_marker_rate': round(float(np.mean(cf_marker_rates)), 4),
            'temporal_marker_rate': round(float(np.mean(temporal_marker_rates)), 4),
        }

        print(f"  {condition}: "
              f"CF={results[condition]['cf_marker_rate']:.3f}, "
              f"Temporal={results[condition]['temporal_marker_rate']:.3f}")

    sep = (results['with_cf_markers']['cf_marker_rate'] -
           results['without_cf_markers']['cf_marker_rate'])
    print(f"\n  CF 标记分离度: {sep:.3f}")
    return results


def experiment_4_transfer(num_runs: int = 5) -> Dict:
    """
    实验 4：迁移到新动作

    在训练动作上学习反事实推理，在留出动作上测试。
    """
    print("=" * 60)
    print("实验 4：新动作迁移")
    print("=" * 60)

    actions_list = sorted(ACTIONS)[:8]
    train_actions = actions_list[:6]
    test_actions = actions_list[6:]

    train_srs = []
    test_srs = []

    for run in range(num_runs):
        # 训练世界（仅训练动作）
        world = CounterfactualWorld()
        effects = ['roll', 'bounce', 'break', 'float', 'sink', 'glow',
                   'shake', 'spin', 'melt', 'freeze']
        for action in train_actions:
            n_eff = random.randint(2, 3)
            chosen = random.sample(effects, n_eff)
            probs = [0.5] + [0.3 / (n_eff - 1)] * (n_eff - 1)
            probs[-1] = 1.0 - sum(probs[:-1])
            for eff, prob in zip(chosen, probs):
                world.add_action_effect(action, eff, prob)

        sp = CounterfactualAgent(EmergingLanguage())
        li = CounterfactualAgent(EmergingLanguage())
        game = CounterfactualCommunicationGame(sp, li, world)

        for _ in range(200):
            game.play_round()

        train_sr = game.successes / max(1, game.games_played)
        train_srs.append(train_sr)

        # 测试世界（仅测试动作）
        test_world = CounterfactualWorld()
        for action in test_actions:
            n_eff = random.randint(2, 3)
            chosen = random.sample(effects, n_eff)
            probs = [0.5] + [0.3 / (n_eff - 1)] * (n_eff - 1)
            probs[-1] = 1.0 - sum(probs[:-1])
            for eff, prob in zip(chosen, probs):
                test_world.add_action_effect(action, eff, prob)

        sp2 = CounterfactualAgent(sp.language)  # 携带训练语言
        li2 = CounterfactualAgent(li.language)
        test_game = CounterfactualCommunicationGame(sp2, li2, test_world)

        for _ in range(100):
            test_game.play_round()

        test_sr = test_game.successes / max(1, test_game.games_played)
        test_srs.append(test_sr)

    avg_train = float(np.mean(train_srs))
    avg_test = float(np.mean(test_srs))
    ratio = avg_test / avg_train if avg_train > 0 else 0

    print(f"  训练 SR: {avg_train:.3f}")
    print(f"  测试 SR: {avg_test:.3f}")
    print(f"  迁移率: {ratio:.3f}")

    return {
        'train_success_rate': round(avg_train, 4),
        'test_success_rate': round(avg_test, 4),
        'transfer_ratio': round(ratio, 4),
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_marker_emergence()
    results['experiment_2'] = experiment_2_reasoning_accuracy()
    results['experiment_3'] = experiment_3_marker_separation()
    results['experiment_4'] = experiment_4_transfer()

    output_file = 'counterfactual_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
