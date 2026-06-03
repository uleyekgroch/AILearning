"""
大规模语言涌现实验：更多属性、更复杂语法

对比 4 种场景复杂度下的语言涌现：
- simple  (12 物体): 4色×3形，2符号足够
- medium  (24 物体): 4色×3形×2大小，可能需要3符号
- complex (36 物体): 采样，经常需要3符号
- extreme (72 物体): 4色×3形×2大小×3材质，必须3符号

核心问题：
1. 3符号组合是否在需要时涌现？
2. 形容词层级排序是否收敛？
3. 语法模式（n-gram）是否从成功经验中固化？
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import (
    CommunicationGame, EmergingLanguage, generate_rich_scene
)


def run_single_game(complexity: str, num_rounds: int,
                    seed: int = 42) -> Dict:
    """运行单个复杂度级别的语言涌现实验"""
    np.random.seed(seed)

    game = CommunicationGame()
    log = {
        'successes': [], 'vocab': [], 'combo_rate': [],
        'tri_rate': [], 'order_consistency': [],
    }

    for round_idx in range(num_rounds):
        scene = generate_rich_scene(complexity)
        target_idx = np.random.randint(0, len(scene))
        game.play_round(scene, target_idx)

        if (round_idx + 1) % 50 == 0:
            window = 50
            recent = game.game_log[-window:]
            stats = game.get_stats()

            log['successes'].append(
                sum(1 for g in recent if g['success']) / window
            )
            log['vocab'].append(stats['vocabulary_size'])
            log['combo_rate'].append(stats['combination_rate'])
            log['tri_rate'].append(stats['tri_symbol_rate'])
            log['order_consistency'].append(stats['order_consistency'])

    return {
        'log': log,
        'final': game.get_stats(),
        'game': game,
    }


def run_progressive_experiment(num_rounds: int = 1000,
                                verbose: bool = True) -> Dict:
    """
    渐进复杂度实验

    从简单到极端逐步增加复杂度，
    观察语言如何适应更复杂的交流需求。
    """
    np.random.seed(42)

    game = CommunicationGame()
    log = {
        'successes': [], 'vocab': [], 'combo_rate': [],
        'tri_rate': [], 'order_consistency': [],
        'complexity': [],
    }

    # 阶段划分
    phases = [
        (0, 200, 'simple', '阶段1: 简单(12物体)'),
        (200, 400, 'medium', '阶段2: 中等(24物体)'),
        (400, 700, 'complex', '阶段3: 复杂(36物体)'),
        (700, 1000, 'extreme', '阶段4: 极端(72物体)'),
    ]

    for round_idx in range(num_rounds):
        # 确定当前复杂度
        current_complexity = 'simple'
        for start, end, comp, _ in phases:
            if start <= round_idx < end:
                current_complexity = comp
                break

        scene = generate_rich_scene(current_complexity)
        target_idx = np.random.randint(0, len(scene))
        game.play_round(scene, target_idx)

        if (round_idx + 1) % 50 == 0:
            window = 50
            recent = game.game_log[-window:]
            stats = game.get_stats()

            log['successes'].append(
                sum(1 for g in recent if g['success']) / window
            )
            log['vocab'].append(stats['vocabulary_size'])
            log['combo_rate'].append(stats['combination_rate'])
            log['tri_rate'].append(stats['tri_symbol_rate'])
            log['order_consistency'].append(stats['order_consistency'])
            log['complexity'].append(current_complexity)

            if verbose:
                print(f"轮次 {round_idx+1:4d} [{current_complexity:>7s}]: "
                      f"成功率={log['successes'][-1]:.1%} "
                      f"词汇={stats['vocabulary_size']:2d} "
                      f"组合率={stats['combination_rate']:.1%} "
                      f"3符号率={stats['tri_symbol_rate']:.1%} "
                      f"词序一致={stats['order_consistency']:.1%}")

    return {
        'log': log,
        'final': game.get_stats(),
        'game': game,
    }


def main():
    print("大规模语言涌现实验：更多属性、更复杂语法")
    print("=" * 70)

    # === 实验 1: 各复杂度独立对比 ===
    print("\n实验 1: 各复杂度独立对比 (500轮/复杂度)")
    print("-" * 70)

    complexities = ['simple', 'medium', 'complex', 'extreme']
    results = {}

    for comp in complexities:
        result = run_single_game(comp, num_rounds=500)
        results[comp] = result
        final = result['final']

        print(f"\n  [{comp:>7s}] 最终统计:")
        print(f"    成功率:     {final['success_rate']:.1%}")
        print(f"    词汇量:     {final['vocabulary_size']}")
        print(f"    组合率:     {final['combination_rate']:.1%}")
        print(f"    3符号率:    {final['tri_symbol_rate']:.1%}")
        print(f"    词序一致性: {final['order_consistency']:.1%}")
        print(f"    n-gram模式: {final['ngram_patterns']}")

        # 显示形容词层级偏好
        mod_order = final.get('preferred_modifier_order', [])
        if mod_order:
            print(f"    形容词层级: {' > '.join(mod_order)}")

    # === 实验 2: 渐进复杂度 ===
    print("\n" + "=" * 70)
    print("实验 2: 渐进复杂度 (1000轮，simple→medium→complex→extreme)")
    print("-" * 70)

    prog_result = run_progressive_experiment(num_rounds=1000, verbose=True)
    final = prog_result['final']

    print(f"\n渐进实验最终统计:")
    print(f"  成功率:     {final['success_rate']:.1%}")
    print(f"  词汇量:     {final['vocabulary_size']}")
    print(f"  组合率:     {final['combination_rate']:.1%}")
    print(f"  3符号率:    {final['tri_symbol_rate']:.1%}")
    print(f"  词序一致性: {final['order_consistency']:.1%}")
    print(f"  n-gram模式: {final['ngram_patterns']}")

    mod_order = final.get('preferred_modifier_order', [])
    if mod_order:
        print(f"  形容词层级: {' > '.join(mod_order)}")

    # === 关键验证 ===
    print("\n" + "=" * 70)
    print("关键验证")
    print("=" * 70)

    # 1. simple 场景：2符号组合成功
    simple_rate = results['simple']['final']['success_rate']
    if simple_rate > 0.8:
        print(f"  ✓ simple 场景：2符号组合成功 ({simple_rate:.1%})")
    else:
        print(f"  ✗ simple 场景成功率偏低 ({simple_rate:.1%})")

    # 2. extreme 场景：3符号组合涌现
    extreme_tri = results['extreme']['final']['tri_symbol_rate']
    if extreme_tri > 0.5:
        print(f"  ✓ extreme 场景：3符号组合涌现 ({extreme_tri:.1%})")
    else:
        print(f"  △ extreme 场景3符号率偏低 ({extreme_tri:.1%})")

    # 3. 形容词层级收敛
    mod_order = results['extreme']['final'].get('preferred_modifier_order', [])
    if mod_order and len(mod_order) >= 2:
        print(f"  ✓ 形容词层级收敛: {' > '.join(mod_order)}")
    else:
        print(f"  △ 形容词层级未充分收敛")

    # 4. n-gram 模式涌现
    ngram = results['extreme']['final']['ngram_patterns']
    if ngram > 10:
        print(f"  ✓ n-gram 模式涌现 ({ngram} 种)")
    else:
        print(f"  △ n-gram 模式较少 ({ngram} 种)")

    # === 保存结果 ===
    import json
    output = {
        'independent': {
            comp: {
                'success_rate': r['final']['success_rate'],
                'vocabulary_size': r['final']['vocabulary_size'],
                'combination_rate': r['final']['combination_rate'],
                'tri_symbol_rate': r['final']['tri_symbol_rate'],
                'order_consistency': r['final']['order_consistency'],
                'ngram_patterns': r['final']['ngram_patterns'],
                'preferred_modifier_order': r['final'].get('preferred_modifier_order', []),
            }
            for comp, r in results.items()
        },
        'progressive': {
            'success_rate': final['success_rate'],
            'vocabulary_size': final['vocabulary_size'],
            'tri_symbol_rate': final['tri_symbol_rate'],
            'preferred_modifier_order': final.get('preferred_modifier_order', []),
        }
    }

    with open('D:/mayAi/AILearning_v0527/mvl/language_rich_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: language_rich_results.json")


if __name__ == '__main__':
    main()
