"""
语言涌现实验

对比三种语言系统的交流效果：
A. 单符号系统（baseline，只能用一个符号）
B. 组合系统（可以组合多个符号）
C. 语法系统（有词序偏好）

核心问题：语法是怎么从交流压力中涌现的？
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import (
    CommunicationGame, EmergingLanguage, Speaker, Listener,
    generate_scene, generate_ambiguous_scene
)


class SingleSymbolGame:
    """
    单符号系统（baseline）

    Speaker 只能用一个符号描述目标。
    这是最原始的语言——没有组合，没有语法。
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, scene: List[Dict[str, str]], target_idx: int) -> bool:
        if target_idx >= len(scene):
            return False

        target = scene[target_idx]

        # 只选一个符号（优先选形状，因为更有区分力）
        shape = target.get('shape', '')
        color = target.get('color', '')
        utterance = [shape] if shape else [color]

        # Listener 只看第一个符号
        chosen = None
        for i, obj in enumerate(scene):
            if utterance[0] in obj.values():
                chosen = i
                break

        success = (chosen == target_idx)

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target': target, 'utterance': utterance,
            'chosen': chosen, 'success': success,
        })
        return success


def generate_highly_ambiguous_scene() -> List[Dict[str, str]]:
    """
    生成高度歧义的场景

    关键：每个属性值在场景中出现多次，
    迫使 Speaker 必须组合符号才能唯一标识目标。
    """
    # 4种颜色 × 3种形状 = 12个物体
    # 每种颜色出现3次，每种形状出现4次
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle']

    scene = []
    for c in colors:
        for s in shapes:
            scene.append({'color': c, 'shape': s})

    np.random.shuffle(scene)
    return scene


def run_experiment(num_rounds: int = 500, scene_size: int = 6,
                   verbose: bool = False) -> Dict:
    """
    运行语言涌现实验

    对比三个系统在相同场景序列下的表现。
    """
    np.random.seed(42)

    # 三个系统
    single_game = SingleSymbolGame()
    comp_game = CommunicationGame()  # 组合系统

    results = {
        'single': {'successes': [], 'vocab': [], 'combo_rate': []},
        'compositional': {'successes': [], 'vocab': [], 'combo_rate': [], 'order_consistency': []},
    }

    for round_idx in range(num_rounds):
        # 生成高度歧义的场景（这是语言组合涌现的关键条件）
        scene = generate_highly_ambiguous_scene()

        # 随机选择目标
        target_idx = np.random.randint(0, len(scene))

        # 单符号系统
        single_game.play_round(scene, target_idx)

        # 组合系统
        comp_game.play_round(scene, target_idx)

        # 记录每50轮的统计
        if (round_idx + 1) % 50 == 0:
            window = 50

            single_stats = single_game.language.get_stats()
            comp_stats = comp_game.language.get_stats()

            # 计算滑动窗口成功率
            single_recent = single_game.game_log[-window:]
            comp_recent = comp_game.game_log[-window:]

            results['single']['successes'].append(
                sum(1 for g in single_recent if g['success']) / window
            )
            results['compositional']['successes'].append(
                sum(1 for g in comp_recent if g['success']) / window
            )

            results['single']['vocab'].append(single_stats['vocabulary_size'])
            results['compositional']['vocab'].append(comp_stats['vocabulary_size'])

            results['single']['combo_rate'].append(0.0)  # 单符号系统无组合
            results['compositional']['combo_rate'].append(comp_stats['combination_rate'])

            results['compositional']['order_consistency'].append(
                comp_stats['order_consistency']
            )

            if verbose:
                print(f"轮次 {round_idx+1}:")
                print(f"  单符号: 成功率={results['single']['successes'][-1]:.1%} "
                      f"词汇={single_stats['vocabulary_size']}")
                print(f"  组合:   成功率={results['compositional']['successes'][-1]:.1%} "
                      f"词汇={comp_stats['vocabulary_size']} "
                      f"组合率={comp_stats['combination_rate']:.1%} "
                      f"词序一致={comp_stats['order_consistency']:.1%}")

    # 最终统计
    final = {}
    for name, game in [('single', single_game), ('compositional', comp_game)]:
        stats = game.language.get_stats()
        final[name] = stats

    return results, final


def main():
    print("语言涌现实验：从符号到语法的自发涌现")
    print("=" * 60)
    print()

    results, final = run_experiment(num_rounds=500, verbose=True)

    print()
    print("=" * 60)
    print("最终结果")
    print("=" * 60)

    print(f"\n{'系统':>12} | {'成功率':>8} | {'词汇':>6} | {'组合率':>8} | {'词序一致':>8}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    for name, label in [('single', '单符号'), ('compositional', '组合')]:
        stats = final[name]
        order = f"{stats.get('order_consistency', 0):.1%}" if name != 'single' else "N/A"
        print(f"{label:>12} | {stats['success_rate']:>8.1%} | "
              f"{stats['vocabulary_size']:>6} | "
              f"{stats['combination_rate']:>8.1%} | {order:>8}")

    # 检查关键假设
    print()
    print("关键验证:")

    comp_rate = final['compositional']['success_rate']
    single_rate = final['single']['success_rate']
    if comp_rate > single_rate:
        print(f"  ✓ 组合系统优于单符号系统 ({comp_rate:.1%} > {single_rate:.1%})")
    else:
        print(f"  ✗ 组合系统未优于单符号系统 ({comp_rate:.1%} <= {single_rate:.1%})")

    order_consist = final['compositional'].get('order_consistency', 0)
    if order_consist > 0.7:
        print(f"  ✓ 词序一致性收敛 ({order_consist:.1%} > 70%)")
    else:
        print(f"  △ 词序一致性未充分收敛 ({order_consist:.1%})")

    # 提取语法规则
    comp_game = CommunicationGame()
    rules = comp_game.language.extract_grammar_rules()
    if rules:
        print(f"  ✓ 语法规则涌现 ({len(rules)} 条)")
    else:
        print(f"  △ 语法规则未涌现")

    # 保存结果
    import json
    output = {
        'final_stats': {k: v for k, v in final.items()},
        'convergence': {
            'compositional_successes': results['compositional']['successes'],
            'order_consistency': results['compositional']['order_consistency'],
        }
    }
    with open('D:/mayAi/AILearning_v0527/mvl/language_emergence_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: language_emergence_results.json")


if __name__ == '__main__':
    main()
