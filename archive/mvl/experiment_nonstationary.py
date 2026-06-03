"""
Phase 74: 非平稳环境语言适应 — 词汇淘汰与新词涌现

核心思想：
所有前 73 个 Phase 假设环境是静态的——物体分布、类别、
关联关系不变。但真实环境不断变化（季节、灾难、社会变迁），
语言必须持续适应。

本阶段测试：
- 环境规则变化时，旧词汇是否被淘汰
- 新词是否快速涌现
- 适应型 Agent 是否优于静态 Agent
- 循环变化时旧词汇是否被重新激活（记忆效应）

涌现条件：
1. 环境分布在不同 regime 下差异显著
2. 旧词汇在新 regime 下通信成功率骤降
3. 适应机制（遗忘旧词、探索新词）提供选择优势
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
)

# ============================================================
# 动态环境
# ============================================================

REGIME_COLORS = {
    'spring': {'green', 'yellow', 'white'},
    'summer': {'red', 'orange', 'bright'},
    'autumn': {'brown', 'orange', 'dark'},
    'winter': {'white', 'gray', 'blue'},
}

REGIME_SHAPES = {
    'spring': {'circle', 'oval'},
    'summer': {'triangle', 'star'},
    'autumn': {'square', 'rectangle'},
    'winter': {'cube', 'pyramid'},
}

REGIME_SIZES = {
    'spring': {'small', 'tiny'},
    'summer': {'medium', 'large'},
    'autumn': {'large', 'huge'},
    'winter': {'medium', 'small'},
}


class DynamicEnvironment:
    """
    动态环境：regime 周期性变化

    每个 regime 有不同的颜色、形状、大小分布，
    导致旧词汇失效，新词汇需要涌现。
    """

    def __init__(self, regime_schedule: Optional[List[Tuple[int, str]]] = None):
        self.all_regimes = ['spring', 'summer', 'autumn', 'winter']
        self.current_regime = 'spring'
        self.regime_schedule = regime_schedule or [
            (0, 'spring'), (150, 'summer'), (300, 'autumn'), (450, 'winter'),
        ]
        self.regime_index = 0
        self.change_count = 0

    def check_regime_change(self, round_num: int) -> bool:
        """检查是否需要切换 regime"""
        if self.regime_index < len(self.regime_schedule) - 1:
            next_round, next_regime = self.regime_schedule[self.regime_index + 1]
            if round_num >= next_round:
                self.current_regime = next_regime
                self.regime_index += 1
                self.change_count += 1
                return True
        return False

    def generate_scene(self, num_objects: int = 4) -> List[Dict]:
        """根据当前 regime 生成场景"""
        regime = self.current_regime
        colors = list(REGIME_COLORS.get(regime, COLORS))
        shapes = list(REGIME_SHAPES.get(regime, SHAPES))
        sizes = list(REGIME_SIZES.get(regime, SIZES))

        scene = []
        for _ in range(num_objects):
            obj = {
                'color': random.choice(colors),
                'shape': random.choice(shapes),
                'size': random.choice(sizes),
            }
            scene.append(obj)
        return scene

    def get_regime_symbols(self) -> set:
        """获取当前 regime 的特征符号"""
        regime = self.current_regime
        symbols = set()
        symbols.update(REGIME_COLORS.get(regime, set()))
        symbols.update(REGIME_SHAPES.get(regime, set()))
        symbols.update(REGIME_SIZES.get(regime, set()))
        return symbols


# ============================================================
# 适应型 Agent
# ============================================================

class AdaptiveAgent:
    """
    适应型 Agent

    检测通信失败，淘汰旧词汇，探索新词。
    """

    def __init__(self, adaptation_rate: float = 0.3,
                 forgetting_threshold: int = 30):
        self.language = EmergingLanguage()
        self.adaptation_rate = adaptation_rate
        self.forgetting_threshold = forgetting_threshold
        self.consecutive_failures = 0
        self.symbol_last_used: Dict[str, int] = {}
        self.symbol_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        self.total_descriptions = 0
        self.vocabulary_history: List[Dict] = []

    def describe(self, target: Dict, round_num: int) -> List[str]:
        """生成描述，跳过低分符号（模拟适应）"""
        self.total_descriptions += 1
        utterance = []
        for attr in ['color', 'shape', 'size']:
            if attr in target:
                sym = target[attr]
                score = self.symbol_scores.get(sym, 1.0)
                if score < 0.3:
                    # 低分符号被淘汰，尝试替换
                    continue
                utterance.append(sym)
                self.symbol_last_used[sym] = round_num
        # 如果所有符号都被淘汰，退回使用原始属性
        if not utterance:
            for attr in ['color', 'shape', 'size']:
                if attr in target:
                    utterance.append(target[attr])
        return utterance

    def interpret(self, utterance: List[str], scene: List[Dict]) -> Optional[int]:
        """解释话语，低分符号降权"""
        best_idx = None
        best_score = -1

        for i, obj in enumerate(scene):
            score = 0
            for sym in utterance:
                weight = self.symbol_scores.get(sym, 1.0)
                for attr, value in obj.items():
                    if value == sym:
                        score += weight
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def update_from_result(self, utterance: List[str], success: bool,
                           round_num: int):
        """根据通信结果更新符号分数"""
        self.language.record_usage(utterance, success)

        if success:
            self.consecutive_failures = 0
            for sym in utterance:
                self.symbol_scores[sym] = min(2.0,
                    self.symbol_scores[sym] + self.adaptation_rate * 0.1)
        else:
            self.consecutive_failures += 1
            for sym in utterance:
                self.symbol_scores[sym] = max(0.1,
                    self.symbol_scores[sym] - self.adaptation_rate * 0.2)

        # 遗忘长期未使用的符号
        stale = [s for s, last in self.symbol_last_used.items()
                 if round_num - last > self.forgetting_threshold]
        for s in stale:
            self.symbol_scores[s] *= 0.5

    def get_active_vocabulary_size(self) -> int:
        """活跃词汇量（分数 > 0.3 的符号数）"""
        return sum(1 for s, sc in self.symbol_scores.items() if sc > 0.3)

    def get_vocabulary_snapshot(self) -> Dict:
        """当前词汇快照"""
        active = {s: round(sc, 3) for s, sc in self.symbol_scores.items()
                  if sc > 0.3}
        return {
            'active_count': len(active),
            'total_known': len(self.symbol_scores),
            'top_symbols': dict(sorted(active.items(),
                                       key=lambda x: -x[1])[:10]),
        }


class StaticAgent(AdaptiveAgent):
    """静态 Agent：不适应环境变化"""

    def __init__(self):
        super().__init__(adaptation_rate=0.0, forgetting_threshold=999999)


# ============================================================
# 通信游戏
# ============================================================

class NonstationaryGame:
    """非平稳环境通信游戏"""

    def __init__(self, speaker: AdaptiveAgent, listener: AdaptiveAgent,
                 env: DynamicEnvironment):
        self.speaker = speaker
        self.listener = listener
        self.env = env
        self.games_played = 0
        self.successes = 0
        self.regime_snapshots: List[Dict] = []

    def play_round(self, round_num: int) -> Dict:
        """进行一轮游戏"""
        regime_changed = self.env.check_regime_change(round_num)
        scene = self.env.generate_scene(4)
        target_idx = random.randint(0, len(scene) - 1)

        utterance = self.speaker.describe(scene[target_idx], round_num)
        chosen = self.listener.interpret(utterance, scene)

        success = (chosen == target_idx)
        self.games_played += 1
        if success:
            self.successes += 1

        self.speaker.update_from_result(utterance, success, round_num)
        self.listener.update_from_result(utterance, success, round_num)

        return {
            'success': success,
            'regime': self.env.current_regime,
            'regime_changed': regime_changed,
        }


# ============================================================
# 实验
# ============================================================

def experiment_1_vocabulary_turnover(num_rounds: int = 500) -> Dict:
    """
    实验 1：词汇淘汰与新词涌现

    追踪 3 次 regime 变化中的词汇周转。
    """
    print("=" * 60)
    print("实验 1：词汇淘汰与新词涌现")
    print("=" * 60)

    schedule = [(0, 'spring'), (150, 'summer'), (300, 'autumn')]
    env = DynamicEnvironment(schedule)
    speaker = AdaptiveAgent()
    listener = AdaptiveAgent()
    game = NonstationaryGame(speaker, listener, env)

    snapshots = []
    prev_regime = 'spring'
    for r in range(num_rounds):
        result = game.play_round(r)
        if (r + 1) % 50 == 0 or result['regime_changed']:
            sp_snap = speaker.get_vocabulary_snapshot()
            li_snap = listener.get_vocabulary_snapshot()
            sr = game.successes / max(1, game.games_played)
            snapshots.append({
                'round': r + 1,
                'regime': env.current_regime,
                'regime_changed': result['regime_changed'],
                'success_rate': round(sr, 4),
                'speaker_vocab': sp_snap,
                'listener_vocab': li_snap,
            })
            if result['regime_changed']:
                print(f"  *** REGIME CHANGE: {prev_regime} → {env.current_regime} ***")
                prev_regime = env.current_regime
            print(f"  Round {r+1}: regime={env.current_regime}, "
                  f"SR={sr:.3f}, 活跃词={sp_snap['active_count']}")

    return {
        'final_sr': round(game.successes / max(1, game.games_played), 4),
        'regime_changes': env.change_count,
        'snapshots': snapshots,
    }


def experiment_2_adaptive_vs_static(num_rounds: int = 300,
                                     num_runs: int = 5) -> Dict:
    """
    实验 2：适应型 vs 静态 Agent

    对比有/无适应机制的通信成功率。
    """
    print("=" * 60)
    print("实验 2：适应型 vs 静态 Agent")
    print("=" * 60)

    adaptive_srs = []
    static_srs = []

    for run in range(num_runs):
        schedule = [(0, 'spring'), (100, 'summer'), (200, 'autumn')]
        env1 = DynamicEnvironment(schedule)
        sp1 = AdaptiveAgent()
        li1 = AdaptiveAgent()
        game1 = NonstationaryGame(sp1, li1, env1)

        for r in range(num_rounds):
            game1.play_round(r)

        sr1 = game1.successes / max(1, game1.games_played)
        adaptive_srs.append(sr1)

        env2 = DynamicEnvironment(schedule)
        sp2 = StaticAgent()
        li2 = StaticAgent()
        game2 = NonstationaryGame(sp2, li2, env2)

        for r in range(num_rounds):
            game2.play_round(r)

        sr2 = game2.successes / max(1, game2.games_played)
        static_srs.append(sr2)

    avg_adaptive = float(np.mean(adaptive_srs))
    avg_static = float(np.mean(static_srs))
    advantage = (avg_adaptive - avg_static) / max(0.01, avg_static) * 100

    print(f"  适应型: {avg_adaptive:.3f}")
    print(f"  静态: {avg_static:.3f}")
    print(f"  适应优势: {advantage:.1f}%")

    return {
        'adaptive_sr': round(avg_adaptive, 4),
        'static_sr': round(avg_static, 4),
        'adaptive_advantage_pct': round(advantage, 2),
    }


def experiment_3_change_speed(speeds: List[str] = None,
                               num_rounds: int = 400) -> Dict:
    """
    实验 3：变化速度影响

    测试不同 regime 变化频率下的适应成功率。
    """
    print("=" * 60)
    print("实验 3：变化速度影响")
    print("=" * 60)

    if speeds is None:
        speeds = ['slow', 'medium', 'fast']

    speed_schedules = {
        'slow': [(0, 'spring'), (200, 'summer'), (350, 'autumn')],
        'medium': [(0, 'spring'), (100, 'summer'), (200, 'autumn'),
                   (300, 'winter')],
        'fast': [(0, 'spring'), (50, 'summer'), (100, 'autumn'),
                 (150, 'winter'), (200, 'spring'), (250, 'summer'),
                 (300, 'autumn'), (350, 'winter')],
    }

    results = {}
    for speed in speeds:
        schedule = speed_schedules[speed]
        env = DynamicEnvironment(schedule)
        sp = AdaptiveAgent()
        li = AdaptiveAgent()
        game = NonstationaryGame(sp, li, env)

        for r in range(num_rounds):
            game.play_round(r)

        sr = game.successes / max(1, game.games_played)
        vocab = sp.get_vocabulary_snapshot()
        results[speed] = {
            'success_rate': round(sr, 4),
            'active_vocab': vocab['active_count'],
            'regime_changes': env.change_count,
        }
        print(f"  {speed}: SR={sr:.3f}, 活跃词={vocab['active_count']}, "
              f"变化次数={env.change_count}")

    return results


def experiment_4_memory_retention(num_cycles: int = 3) -> Dict:
    """
    实验 4：记忆保留

    环境循环回到旧 regime 时，旧词汇是否恢复更快。
    """
    print("=" * 60)
    print("实验 4：记忆保留")
    print("=" * 60)

    # 循环 schedule：spring → summer → spring → summer → ...
    schedule = []
    round_offset = 0
    for cycle in range(num_cycles):
        schedule.append((round_offset, 'spring'))
        schedule.append((round_offset + 80, 'summer'))
        round_offset += 160

    total_rounds = round_offset
    env = DynamicEnvironment(schedule)
    sp = AdaptiveAgent()
    li = AdaptiveAgent()
    game = NonstationaryGame(sp, li, env)

    cycle_recovery = []
    actual_r = 0
    while actual_r < total_rounds:
        result = game.play_round(actual_r)

        # 每次回到 spring 时记录恢复速度
        if result['regime_changed'] and env.current_regime == 'spring':
            recovery_successes = 0
            recovery_total = 20
            for rr in range(recovery_total):
                actual_r += 1
                res = game.play_round(actual_r)
                if res['success']:
                    recovery_successes += 1
            recovery_sr = recovery_successes / max(1, recovery_total)
            cycle_recovery.append({
                'cycle': len(cycle_recovery) + 1,
                'recovery_sr': round(recovery_sr, 4),
            })
            print(f"  循环 {len(cycle_recovery)}: 恢复 SR={recovery_sr:.3f}")
        actual_r += 1

    # 检查恢复是否随循环加速
    if len(cycle_recovery) >= 2:
        first_sr = cycle_recovery[0]['recovery_sr']
        last_sr = cycle_recovery[-1]['recovery_sr']
        improvement = (last_sr - first_sr) / max(0.01, first_sr) * 100
    else:
        improvement = 0.0

    print(f"  首次恢复: {cycle_recovery[0]['recovery_sr'] if cycle_recovery else 0:.3f}")
    print(f"  最终恢复: {cycle_recovery[-1]['recovery_sr'] if cycle_recovery else 0:.3f}")
    print(f"  恢复提升: {improvement:.1f}%")

    return {
        'cycle_recovery': cycle_recovery,
        'recovery_improvement_pct': round(improvement, 2),
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_vocabulary_turnover()
    results['experiment_2'] = experiment_2_adaptive_vs_static()
    results['experiment_3'] = experiment_3_change_speed()
    results['experiment_4'] = experiment_4_memory_retention()

    output_file = 'nonstationary_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
