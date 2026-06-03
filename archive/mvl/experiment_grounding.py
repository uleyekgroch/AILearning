"""
符号接地深化实验（Phase 13）

4 个实验：
1. 动作接地：验证动作符号涌现、动作-物体组合形成
2. 情感接地：验证情感符号与内部状态的相关性
3. 因果接地：验证因果标记涌现、规则置信度
4. 统一接地：同时包含三种符号，验证共存和组合
"""

import json
import numpy as np
from typing import Dict, List

from language_emergence import _symbol_category
from grounding_actions import (
    ActionCommunicationGame, ActionLanguageAgent,
    cross_action_round, generate_action_scene,
)
from grounding_emotions import (
    EmotionCommunicationGame, EmotionLanguageAgent,
    EmotionMapper, cross_emotion_round, generate_emotion_scene,
)
from grounding_causal import (
    CausalCommunicationGame, CausalLanguageAgent,
    cross_causal_round, generate_causal_scene,
)
from grounding_unified import run_unified_experiment


def experiment_action_grounding(num_rounds: int = 500) -> Dict:
    """
    实验 1：动作接地

    验证：
    - 动作符号在词汇表中涌现
    - 动作-物体符号形成组合
    - 交流成功率随训练提升
    """
    print("=" * 60)
    print("实验 1：动作接地")
    print("=" * 60)

    game = ActionCommunicationGame()
    results = {'rounds': [], 'success_rate': [], 'vocab_size': []}

    window = 50
    successes_window = []

    for round_idx in range(num_rounds):
        scene, action_events = generate_action_scene(num_objects=6, complexity='medium')
        if not action_events:
            continue

        target_idx = np.random.randint(len(action_events))
        success = game.play_round(scene, action_events, target_idx)
        successes_window.append(success)

        if (round_idx + 1) % window == 0:
            rate = sum(successes_window) / len(successes_window)
            results['rounds'].append(round_idx + 1)
            results['success_rate'].append(rate)
            results['vocab_size'].append(game.language.get_vocabulary_size())
            print(f"  轮次 {round_idx+1}: 成功率 {rate:.1%}, 词汇量 {game.language.get_vocabulary_size()}")
            successes_window = []

    # 分析符号类别
    vocab = game.language.vocabulary
    action_symbols = [s for s in vocab if _symbol_category(s) == 'action']
    effect_symbols = [s for s in vocab if _symbol_category(s) == 'action_effect']
    static_symbols = [s for s in vocab if _symbol_category(s) in ('color', 'shape', 'size', 'material')]

    print(f"\n  动作符号: {action_symbols}")
    print(f"  效果符号: {effect_symbols}")
    print(f"  静态符号: {static_symbols}")
    print(f"  组合率: {game.language.get_combination_rate():.1%}")

    return {
        'success_rates': results['success_rate'],
        'vocab_sizes': results['vocab_size'],
        'action_symbols': action_symbols,
        'effect_symbols': effect_symbols,
        'static_symbols': static_symbols,
        'final_vocab_size': game.language.get_vocabulary_size(),
        'combination_rate': game.language.get_combination_rate(),
        'ngram_patterns': len(game.language.ngram_patterns),
    }


def experiment_emotion_grounding(num_rounds: int = 500) -> Dict:
    """
    实验 2：情感接地

    验证：
    - 情感符号在词汇表中涌现
    - 不同情感状态产生不同的交流模式
    - 情感-物体组合形成
    """
    print("\n" + "=" * 60)
    print("实验 2：情感接地")
    print("=" * 60)

    game = EmotionCommunicationGame()
    results = {'rounds': [], 'success_rate': [], 'vocab_size': []}
    emotion_usage = {}

    window = 50
    successes_window = []

    for round_idx in range(num_rounds):
        scene, assigned_emotions = generate_emotion_scene(num_objects=6, complexity='medium')
        target_idx = np.random.randint(min(6, len(scene)))
        # 场景中每个物体已有不同情感，直接使用
        success = game.play_round(scene, target_idx, assigned_emotions[target_idx])
        successes_window.append(success)

        for em in assigned_emotions:
            emotion_usage[em] = emotion_usage.get(em, 0) + 1

        if (round_idx + 1) % window == 0:
            rate = sum(successes_window) / len(successes_window)
            results['rounds'].append(round_idx + 1)
            results['success_rate'].append(rate)
            results['vocab_size'].append(game.language.get_vocabulary_size())
            print(f"  轮次 {round_idx+1}: 成功率 {rate:.1%}, 词汇量 {game.language.get_vocabulary_size()}")
            successes_window = []

    # 分析情感符号
    vocab = game.language.vocabulary
    emotion_symbols = [s for s in vocab if _symbol_category(s) == 'emotion']
    static_symbols = [s for s in vocab if _symbol_category(s) in ('color', 'shape', 'size', 'material')]

    print(f"\n  情感符号: {emotion_symbols}")
    print(f"  静态符号: {static_symbols}")
    print(f"  情感使用分布: {emotion_usage}")

    return {
        'success_rates': results['success_rate'],
        'vocab_sizes': results['vocab_size'],
        'emotion_symbols': emotion_symbols,
        'static_symbols': static_symbols,
        'emotion_usage': emotion_usage,
        'final_vocab_size': game.language.get_vocabulary_size(),
        'combination_rate': game.language.get_combination_rate(),
    }


def experiment_causal_grounding(num_rounds: int = 500) -> Dict:
    """
    实验 3：因果接地

    验证：
    - 因果标记（if/then/because/so）涌现
    - 因果规则置信度随观察增加
    - 因果-物体组合形成
    """
    print("\n" + "=" * 60)
    print("实验 3：因果接地")
    print("=" * 60)

    game = CausalCommunicationGame()
    results = {'rounds': [], 'success_rate': [], 'vocab_size': []}

    window = 50
    successes_window = []

    for round_idx in range(num_rounds):
        scene, causal_rules = generate_causal_scene(num_objects=6, complexity='medium')
        if not causal_rules:
            continue

        target_idx = np.random.randint(len(causal_rules))
        success = game.play_round(scene, causal_rules, target_idx)
        successes_window.append(success)

        if (round_idx + 1) % window == 0:
            rate = sum(successes_window) / len(successes_window)
            results['rounds'].append(round_idx + 1)
            results['success_rate'].append(rate)
            results['vocab_size'].append(game.language.get_vocabulary_size())
            print(f"  轮次 {round_idx+1}: 成功率 {rate:.1%}, 词汇量 {game.language.get_vocabulary_size()}")
            successes_window = []

    # 分析因果符号
    vocab = game.language.vocabulary
    causal_symbols = [s for s in vocab if _symbol_category(s) == 'causal']
    static_symbols = [s for s in vocab if _symbol_category(s) in ('color', 'shape', 'size', 'material')]

    # 分析因果规则
    causal_stats = game.causal_module.get_stats()

    print(f"\n  因果标记: {causal_symbols}")
    print(f"  静态符号: {static_symbols}")
    print(f"  因果规则: {causal_stats['total_rules']} 条, 置信度>0.6: {causal_stats['confident_rules']} 条")
    if causal_stats['strongest']:
        print(f"  最强规则:")
        for cause, effect, conf in causal_stats['strongest']:
            print(f"    {cause} → {effect}: {conf:.2f}")

    return {
        'success_rates': results['success_rate'],
        'vocab_sizes': results['vocab_size'],
        'causal_symbols': causal_symbols,
        'static_symbols': static_symbols,
        'causal_stats': causal_stats,
        'final_vocab_size': game.language.get_vocabulary_size(),
        'combination_rate': game.language.get_combination_rate(),
    }


def experiment_unified_grounding(num_rounds: int = 500) -> Dict:
    """
    实验 4：统一接地

    验证：
    - 三种符号类型共存
    - 跨类别组合涌现
    - 词汇量随复杂度增长
    """
    print("\n" + "=" * 60)
    print("实验 4：统一接地（动作+情感+因果）")
    print("=" * 60)

    result = run_unified_experiment(
        num_rounds=num_rounds,
        num_objects=6,
        complexity='medium',
    )

    category_usage = result['category_usage']
    print(f"\n  成功率: {result['success_rate']:.1%}")
    print(f"  词汇量: {result['vocab_size']}")
    print(f"  组合率: {result['combination_rate']:.1%}")
    print(f"  3符号率: {result['tri_symbol_rate']:.1%}")
    print(f"  n-gram 模式: {result['ngram_patterns']}")
    print(f"  类别使用分布:")
    for cat, count in sorted(category_usage.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    return result


def run_all_experiments():
    """运行全部 4 个实验"""
    print("符号接地深化实验 (Phase 13)")
    print("=" * 60)

    np.random.seed(42)

    results = {}

    results['action'] = experiment_action_grounding(500)
    results['emotion'] = experiment_emotion_grounding(500)
    results['causal'] = experiment_causal_grounding(500)
    results['unified'] = experiment_unified_grounding(500)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'实验':<15} {'成功率':<10} {'词汇量':<8} {'组合率':<8} {'特殊符号'}")
    print("-" * 60)

    action_syms = len(results['action']['action_symbols']) + len(results['action']['effect_symbols'])
    emotion_syms = len(results['emotion']['emotion_symbols'])
    causal_syms = len(results['causal']['causal_symbols'])

    print(f"{'动作接地':<15} {results['action']['success_rates'][-1] if results['action']['success_rates'] else 0:<10.1%} "
          f"{results['action']['final_vocab_size']:<8} {results['action']['combination_rate']:<8.1%} "
          f"动作{action_syms}个")
    print(f"{'情感接地':<15} {results['emotion']['success_rates'][-1] if results['emotion']['success_rates'] else 0:<10.1%} "
          f"{results['emotion']['final_vocab_size']:<8} {results['emotion']['combination_rate']:<8.1%} "
          f"情感{emotion_syms}个")
    print(f"{'因果接地':<15} {results['causal']['success_rates'][-1] if results['causal']['success_rates'] else 0:<10.1%} "
          f"{results['causal']['final_vocab_size']:<8} {results['causal']['combination_rate']:<8.1%} "
          f"因果{causal_syms}个")
    print(f"{'统一接地':<15} {results['unified']['success_rate']:<10.1%} "
          f"{results['unified']['vocab_size']:<8} {results['unified']['combination_rate']:<8.1%} "
          f"全部")

    # 保存结果
    save_results = {}
    for key, val in results.items():
        save_results[key] = {k: v for k, v in val.items()
                             if not isinstance(v, np.ndarray)}
    with open('grounding_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存至 grounding_results.json")

    return results


if __name__ == '__main__':
    run_all_experiments()
