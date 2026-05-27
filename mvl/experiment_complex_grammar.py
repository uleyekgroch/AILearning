"""
复杂语法涌现实验（Phase 14）

4 个实验：
1. 否定实验：验证 "not" 符号涌现和否定匹配
2. 时态实验：验证时态符号（past/present/future）涌现
3. 递归从句实验：验证 "that" 相对从句标记涌现
4. 统一实验：三种语法结构共存
"""

import json
import numpy as np
from typing import Dict

from language_emergence import (
    _symbol_category, NEGATION_MARKERS, TENSE_MARKERS, RELATIVE_MARKERS,
)
from grounding_negation import (
    NegationCommunicationGame, NegationLanguageAgent,
    cross_negation_round, generate_negation_scene,
)
from grounding_temporal import (
    TemporalCommunicationGame, TemporalLanguageAgent,
    cross_temporal_round, generate_temporal_scene,
)
from grounding_recursive import (
    RecursiveCommunicationGame, RecursiveLanguageAgent,
    cross_recursive_round, generate_recursive_scene,
)


def experiment_negation(num_rounds: int = 500) -> Dict:
    """
    实验 1：否定接地

    验证：
    - "not" 符号在词汇表中涌现
    - Listener 正确解释否定
    - 否定在需要时被使用
    """
    print("=" * 60)
    print("实验 1：否定接地")
    print("=" * 60)

    game = NegationCommunicationGame()
    results = {'rounds': [], 'success_rate': [], 'negation_rate': []}

    window = 50
    successes_window = []
    negation_window = []

    for round_idx in range(num_rounds):
        scene, target_idx = generate_negation_scene(num_objects=8, complexity='medium')
        success = game.play_round(scene, target_idx)
        successes_window.append(success)

        # 检测是否使用了否定
        last_log = game.game_log[-1] if game.game_log else {}
        negation_window.append(1 if last_log.get('used_negation', False) else 0)

        if (round_idx + 1) % window == 0:
            rate = sum(successes_window) / len(successes_window)
            neg_rate = sum(negation_window) / len(negation_window)
            results['rounds'].append(round_idx + 1)
            results['success_rate'].append(rate)
            results['negation_rate'].append(neg_rate)
            print(f"  轮次 {round_idx+1}: 成功率 {rate:.1%}, 否定使用率 {neg_rate:.1%}, "
                  f"词汇量 {game.language.get_vocabulary_size()}")
            successes_window = []
            negation_window = []

    # 分析否定符号
    vocab = game.language.vocabulary
    negation_symbols = [s for s in vocab if _symbol_category(s) == 'negation']
    other_symbols = [s for s in vocab if _symbol_category(s) not in ('negation', None)]

    stats = game.get_stats()
    print(f"\n  否定符号: {negation_symbols}")
    print(f"  否定使用次数: {stats['negation_used']}")
    print(f"  否定成功次数: {stats['negation_success']}")

    return {
        'success_rates': results['success_rate'],
        'negation_rates': results['negation_rate'],
        'negation_symbols': negation_symbols,
        'other_symbols': other_symbols,
        'final_vocab_size': game.language.get_vocabulary_size(),
        'negation_used': stats['negation_used'],
        'negation_success': stats['negation_success'],
        'combination_rate': game.language.get_combination_rate(),
    }


def experiment_temporal(num_rounds: int = 500) -> Dict:
    """
    实验 2：时态接地

    验证：
    - 时态符号（past/present/future）在词汇表中涌现
    - 时态与动作符号形成组合
    - 交流成功率随训练提升
    """
    print("\n" + "=" * 60)
    print("实验 2：时态接地")
    print("=" * 60)

    game = TemporalCommunicationGame()
    results = {'rounds': [], 'success_rate': [], 'vocab_size': []}

    window = 50
    successes_window = []

    for round_idx in range(num_rounds):
        scene, temporal_events = generate_temporal_scene(num_objects=6, complexity='medium')
        if not temporal_events:
            continue

        target_idx = np.random.randint(len(temporal_events))
        success = game.play_round(scene, temporal_events, target_idx)
        successes_window.append(success)

        if (round_idx + 1) % window == 0:
            rate = sum(successes_window) / len(successes_window)
            results['rounds'].append(round_idx + 1)
            results['success_rate'].append(rate)
            results['vocab_size'].append(game.language.get_vocabulary_size())
            print(f"  轮次 {round_idx+1}: 成功率 {rate:.1%}, 词汇量 {game.language.get_vocabulary_size()}")
            successes_window = []

    # 分析时态符号
    vocab = game.language.vocabulary
    tense_symbols = [s for s in vocab if _symbol_category(s) == 'tense']
    action_symbols = [s for s in vocab if _symbol_category(s) == 'action']
    other_symbols = [s for s in vocab if _symbol_category(s) not in ('tense', 'action', None)]

    print(f"\n  时态符号: {tense_symbols}")
    print(f"  动作符号: {action_symbols}")
    print(f"  组合率: {game.language.get_combination_rate():.1%}")

    return {
        'success_rates': results['success_rate'],
        'vocab_sizes': results['vocab_size'],
        'tense_symbols': tense_symbols,
        'action_symbols': action_symbols,
        'other_symbols': other_symbols,
        'final_vocab_size': game.language.get_vocabulary_size(),
        'combination_rate': game.language.get_combination_rate(),
    }


def experiment_recursive(num_rounds: int = 500) -> Dict:
    """
    实验 3：递归从句接地

    验证：
    - "that" 相对从句标记涌现
    - 两阶段匹配正确工作
    - 从句在需要时被使用
    """
    print("\n" + "=" * 60)
    print("实验 3：递归从句接地")
    print("=" * 60)

    game = RecursiveCommunicationGame()
    results = {'rounds': [], 'success_rate': [], 'relative_rate': []}

    window = 50
    successes_window = []
    relative_window = []

    for round_idx in range(num_rounds):
        scene, target_idx = generate_recursive_scene(num_objects=8, complexity='medium')
        success = game.play_round(scene, target_idx)
        successes_window.append(success)

        # 检测是否使用了从句
        last_log = game.game_log[-1] if game.game_log else {}
        relative_window.append(1 if last_log.get('used_relative', False) else 0)

        if (round_idx + 1) % window == 0:
            rate = sum(successes_window) / len(successes_window)
            rel_rate = sum(relative_window) / len(relative_window)
            results['rounds'].append(round_idx + 1)
            results['success_rate'].append(rate)
            results['relative_rate'].append(rel_rate)
            print(f"  轮次 {round_idx+1}: 成功率 {rate:.1%}, 从句使用率 {rel_rate:.1%}, "
                  f"词汇量 {game.language.get_vocabulary_size()}")
            successes_window = []
            relative_window = []

    # 分析从句符号
    vocab = game.language.vocabulary
    relative_symbols = [s for s in vocab if _symbol_category(s) == 'relative']
    action_symbols = [s for s in vocab if _symbol_category(s) == 'action']
    static_symbols = [s for s in vocab if _symbol_category(s) in ('color', 'shape', 'size', 'material')]

    stats = game.get_stats()
    print(f"\n  从句符号: {relative_symbols}")
    print(f"  动作符号: {action_symbols}")
    print(f"  静态符号: {static_symbols}")
    print(f"  从句使用次数: {stats['relative_used']}")
    print(f"  从句成功次数: {stats['relative_success']}")

    return {
        'success_rates': results['success_rate'],
        'relative_rates': results['relative_rate'],
        'relative_symbols': relative_symbols,
        'action_symbols': action_symbols,
        'static_symbols': static_symbols,
        'final_vocab_size': game.language.get_vocabulary_size(),
        'relative_used': stats['relative_used'],
        'relative_success': stats['relative_success'],
        'combination_rate': game.language.get_combination_rate(),
    }


def experiment_unified_complex_grammar(num_rounds: int = 500) -> Dict:
    """
    实验 4：统一复杂语法

    验证三种语法结构可以共存于同一语言系统。
    使用统一场景，同时包含需要否定、时态、从句的情况。
    """
    print("\n" + "=" * 60)
    print("实验 4：统一复杂语法（否定+时态+从句）")
    print("=" * 60)

    from grounding_unified import generate_grounded_scene, UnifiedCommunicationGame

    game = UnifiedCommunicationGame()
    results = {'rounds': [], 'success_rate': [], 'vocab_size': []}

    window = 50
    successes_window = []

    for round_idx in range(num_rounds):
        # 混合使用不同类型的场景
        scene_type = np.random.choice(['negation', 'temporal', 'recursive', 'unified'])

        if scene_type == 'negation':
            scene, target_idx = generate_negation_scene(num_objects=6, complexity='medium')
        elif scene_type == 'temporal':
            scene, events = generate_temporal_scene(num_objects=6, complexity='medium')
            if events:
                target_idx = events[0].target_obj_idx
            else:
                target_idx = 0
        elif scene_type == 'recursive':
            scene, target_idx = generate_recursive_scene(num_objects=6, complexity='medium')
        else:
            scene, metadata = generate_grounded_scene(
                num_objects=6, complexity='medium',
                include_static=True, include_actions=True,
                include_emotions=True, include_causal=True,
            )
            target_idx = np.random.randint(len(scene))

        success = game.play_round(scene, target_idx)
        successes_window.append(success)

        if (round_idx + 1) % window == 0:
            rate = sum(successes_window) / len(successes_window)
            results['rounds'].append(round_idx + 1)
            results['success_rate'].append(rate)
            results['vocab_size'].append(game.language.get_vocabulary_size())
            print(f"  轮次 {round_idx+1}: 成功率 {rate:.1%}, 词汇量 {game.language.get_vocabulary_size()}")
            successes_window = []

    # 分析所有符号类别
    vocab = game.language.vocabulary
    category_counts = {}
    for sym in vocab:
        cat = _symbol_category(sym)
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + vocab[sym]['frequency']

    # 检查特殊符号
    negation_symbols = [s for s in vocab if _symbol_category(s) == 'negation']
    tense_symbols = [s for s in vocab if _symbol_category(s) == 'tense']
    relative_symbols = [s for s in vocab if _symbol_category(s) == 'relative']

    print(f"\n  否定符号: {negation_symbols}")
    print(f"  时态符号: {tense_symbols}")
    print(f"  从句符号: {relative_symbols}")
    print(f"  类别使用分布:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    return {
        'success_rates': results['success_rate'],
        'vocab_sizes': results['vocab_size'],
        'negation_symbols': negation_symbols,
        'tense_symbols': tense_symbols,
        'relative_symbols': relative_symbols,
        'category_usage': category_counts,
        'final_vocab_size': game.language.get_vocabulary_size(),
        'combination_rate': game.language.get_combination_rate(),
    }


def run_all_experiments():
    """运行全部 4 个实验"""
    print("复杂语法涌现实验 (Phase 14)")
    print("=" * 60)

    np.random.seed(42)

    results = {}

    results['negation'] = experiment_negation(500)
    results['temporal'] = experiment_temporal(500)
    results['recursive'] = experiment_recursive(500)
    results['unified'] = experiment_unified_complex_grammar(500)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'实验':<15} {'成功率':<10} {'词汇量':<8} {'特殊符号'}")
    print("-" * 60)

    neg_count = len(results['negation']['negation_symbols'])
    tense_count = len(results['temporal']['tense_symbols'])
    rel_count = len(results['recursive']['relative_symbols'])

    print(f"{'否定接地':<15} {results['negation']['success_rates'][-1] if results['negation']['success_rates'] else 0:<10.1%} "
          f"{results['negation']['final_vocab_size']:<8} 否定{neg_count}个")
    print(f"{'时态接地':<15} {results['temporal']['success_rates'][-1] if results['temporal']['success_rates'] else 0:<10.1%} "
          f"{results['temporal']['final_vocab_size']:<8} 时态{tense_count}个")
    print(f"{'从句接地':<15} {results['recursive']['success_rates'][-1] if results['recursive']['success_rates'] else 0:<10.1%} "
          f"{results['recursive']['final_vocab_size']:<8} 从句{rel_count}个")
    unified_rate = results['unified']['success_rates'][-1] if results['unified']['success_rates'] else 0
    print(f"{'统一语法':<15} {unified_rate:<10.1%} "
          f"{results['unified']['final_vocab_size']:<8} 全部")

    # 保存结果
    save_results = {}
    for key, val in results.items():
        save_results[key] = {k: v for k, v in val.items()
                             if not isinstance(v, np.ndarray)}
    with open('complex_grammar_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存至 complex_grammar_results.json")

    return results


if __name__ == '__main__':
    run_all_experiments()
