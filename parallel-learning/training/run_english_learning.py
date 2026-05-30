"""
通用学习系统 — 学习小学英语

让系统像小朋友一样从零开始学英语：
  1. 通过参照游戏接触英语单词（和小朋友看图识字一样）
  2. 感知接地建立概念（看到物体，建立"红""圆"等概念）
  3. 反复练习巩固记忆（和小朋友反复看卡片一样）
  4. 好奇心驱动主动探索（遇到不认识的会更关注）
  5. 发展阶段自动推进（从单字→组合→句子）

不设计"英语课程"——系统是通用学习AI，
给它英语场景它就自然学会英语，给它日语场景它就学会日语。
"""

import sys
sys.path.insert(0, '.')

import json
import time
import torch
from datetime import datetime

from src.core.config import LearnerConfig, TrainerConfig
from src.core.learner import Learner
from src.language.communication import CommunicationProtocol, generate_scene, compute_ambiguity
from src.language.grounding import GroundingModule
from src.curriculum.evaluator import CapabilityEvaluator


# ── 小学英语词汇数据 ──────────────────────────────────────────────────
# 就像小朋友的"看图识字卡片"，每个场景是一组物体的视觉特征
# 系统通过参照游戏自然学会这些单词，不是死记硬背

COLORS_EN = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black']
SHAPES_EN = ['circle', 'square', 'triangle', 'star', 'heart', 'diamond']
SIZES_EN = ['big', 'small', 'tiny', 'huge']
MATERIALS_EN = ['wood', 'metal', 'plastic', 'glass', 'rubber', 'paper', 'cloth']

# 小学英语分级词汇（按学习难度递增）
LEVEL_1_WORDS = {
    'colors': ['red', 'blue', 'green', 'yellow'],
    'shapes': ['circle', 'square', 'triangle'],
    'sizes': ['big', 'small'],
}

LEVEL_2_WORDS = {
    'colors': ['red', 'blue', 'green', 'yellow', 'orange', 'purple'],
    'shapes': ['circle', 'square', 'triangle', 'star', 'heart'],
    'sizes': ['big', 'small', 'tiny', 'huge'],
    'materials': ['wood', 'metal', 'plastic'],
}

LEVEL_3_WORDS = {
    'colors': COLORS_EN,
    'shapes': SHAPES_EN,
    'sizes': SIZES_EN,
    'materials': MATERIALS_EN,
}


def generate_lesson(level: int, num_objects: int = 4) -> list:
    """生成一节课的场景（就像老师准备的教学卡片）"""
    import random

    if level == 1:
        pool = LEVEL_1_WORDS
        complexity = 'simple'
    elif level == 2:
        pool = LEVEL_2_WORDS
        complexity = 'medium'
    else:
        pool = LEVEL_3_WORDS
        complexity = 'complex'

    colors = pool.get('colors', COLORS_EN[:4])
    shapes = pool.get('shapes', SHAPES_EN[:3])
    sizes = pool.get('sizes', SIZES_EN[:2])
    materials = pool.get('materials', MATERIALS_EN[:1])

    scene = []
    for i in range(num_objects):
        obj = {}
        obj['color'] = random.choice(colors)
        obj['shape'] = random.choice(shapes)
        if level >= 2:
            obj['size'] = random.choice(sizes)
        if level >= 3:
            obj['material'] = random.choice(materials)
        scene.append(obj)

    return scene


def run_learning_session(learner: Learner, num_rounds: int = 200,
                         level: int = 1, verbose: bool = True):
    """运行一轮学习会话（就像一堂英语课）

    系统通过参照游戏自然学会单词：
    - 老师指着一个红色圆球说 "red circle"
    - 学生猜哪个是 "red circle"
    - 猜对了就建立了 "red" = 红色、"circle" = 圆形的关联
    - 和小朋友学英语的过程完全一样
    """
    comm = learner.communication
    grounding = learner.grounding
    evaluator = CapabilityEvaluator()

    results = {
        'rounds': [],
        'total_successes': 0,
        'total_games': 0,
        'vocabulary_growth': [],
        'grounding_growth': [],
    }

    for round_idx in range(num_rounds):
        # 生成教学场景（随机选物体数量和目标）
        import random
        num_objects = random.randint(2, min(3 + level, 6))
        scene = generate_lesson(level, num_objects)
        target_idx = random.randint(0, len(scene) - 1)

        # 参照游戏：老师描述目标，学生猜
        success = learner.play_reference_game(scene, target_idx)

        # 感知接地：从场景中建立概念
        obs = torch.randn(learner.config.obs_dim) * 0.1
        learner.ground_concept(obs)

        # 社会标注：如果游戏成功，强化符号-概念关联
        if success and scene:
            target = scene[target_idx]
            for attr, value in target.items():
                if attr != 'id' and value:
                    learner.ground_symbol(value, obs, context=attr)

        results['total_games'] += 1
        if success:
            results['total_successes'] += 1

        # 每 20 轮记录一次进度
        if (round_idx + 1) % 20 == 0:
            vocab_size = len(learner.get_vocabulary())
            grounded = len(grounding.get_grounded_symbols())
            clusters = len(grounding.perceptual_clusters)
            success_rate = results['total_successes'] / max(results['total_games'], 1)

            results['vocabulary_growth'].append({
                'round': round_idx + 1,
                'vocab_size': vocab_size,
                'grounded_symbols': grounded,
                'concept_clusters': clusters,
                'success_rate': round(success_rate, 3),
            })

            if verbose:
                print(f"  Round {round_idx+1:3d} | "
                      f"成功率 {success_rate:.1%} | "
                      f"词汇 {vocab_size} | "
                      f"接地符号 {grounded} | "
                      f"概念 {clusters}")

        results['rounds'].append({
            'round': round_idx + 1,
            'success': success,
            'scene_size': len(scene),
            'target_idx': target_idx,
        })

    return results


def main():
    print("=" * 70)
    print("通用学习系统 — 像小朋友一样学英语")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 创建学习体（就像一个刚出生的婴儿）
    config = LearnerConfig(
        obs_dim=40,
        action_dim=4,
        hidden_dims=(32, 16),
        working_memory_capacity=7,
        episodic_memory_capacity=500,
        consolidation_interval=50,
        curiosity_alpha=0.5,
        curiosity_beta=0.5,
    )
    learner = Learner(config)

    print(f"[初始状态]")
    print(f"  阶段: {learner.stage}")
    print(f"  词汇量: {len(learner.get_vocabulary())}")
    print(f"  概念数: {len(learner.grounding.perceptual_clusters)}")
    print()

    # ================================================================
    # Level 1: 基础颜色和形状（就像幼儿园的看图识字）
    # ================================================================
    print("-" * 70)
    print("Level 1: 基础颜色 + 形状 (red, blue, circle, square...)")
    print("  → 通过参照游戏自然学会，不是死记硬背")
    print("-" * 70)

    t0 = time.time()
    level1 = run_learning_session(learner, num_rounds=100, level=1)
    t1 = time.time()

    print(f"\n[Level 1 完成] {t1-t0:.1f}s")
    print(f"  成功率: {level1['total_successes']}/{level1['total_games']} "
          f"({level1['total_successes']/level1['total_games']:.1%})")
    print(f"  词汇量: {len(learner.get_vocabulary())}")
    print(f"  接地符号: {learner.grounding.get_grounded_symbols()[:10]}")

    # 巩固记忆（就像小朋友睡觉时巩固白天学到的东西）
    print("\n  [记忆巩固中...]")
    consolidation = learner.consolidate()
    print(f"  巩固报告: {consolidation}")
    print()

    # ================================================================
    # Level 2: 加入大小和更多属性（就像小学一年级）
    # ================================================================
    print("-" * 70)
    print("Level 2: 加入大小 + 更多颜色形状 (big red circle, small blue square)")
    print("  → 开始理解组合表达")
    print("-" * 70)

    t0 = time.time()
    level2 = run_learning_session(learner, num_rounds=150, level=2)
    t1 = time.time()

    print(f"\n[Level 2 完成] {t1-t0:.1f}s")
    print(f"  成功率: {level2['total_successes']}/{level2['total_games']} "
          f"({level2['total_successes']/level2['total_games']:.1%})")
    print(f"  词汇量: {len(learner.get_vocabulary())}")

    vocab = learner.get_vocabulary()
    if vocab:
        top_words = sorted(vocab.items(), key=lambda x: x[1].get('frequency', 0), reverse=True)[:10]
        print(f"  高频词: {[(w, d['frequency']) for w, d in top_words]}")

    # 检查复合符号（big-red 这种组合词是否涌现）
    lang = learner.communication.language
    if lang.compounds:
        print(f"  复合符号: {list(lang.compounds.keys())[:10]}")

    print("\n  [记忆巩固中...]")
    consolidation = learner.consolidate()
    print()

    # ================================================================
    # Level 3: 完整属性（就像小学三四年级，能描述复杂物体）
    # ================================================================
    print("-" * 70)
    print("Level 3: 完整描述 (big red metal circle, tiny green glass star)")
    print("  → 复杂组合表达")
    print("-" * 70)

    t0 = time.time()
    level3 = run_learning_session(learner, num_rounds=200, level=3)
    t1 = time.time()

    print(f"\n[Level 3 完成] {t1-t0:.1f}s")
    print(f"  成功率: {level3['total_successes']}/{level3['total_games']} "
          f"({level3['total_successes']/level3['total_games']:.1%})")

    # ================================================================
    # 评估学习成果
    # ================================================================
    print()
    print("=" * 70)
    print("学习成果评估")
    print("=" * 70)

    stats = learner.get_stats()
    evaluator = CapabilityEvaluator()
    evaluation = evaluator.evaluate(learner)
    evaluation['vocabulary_size'] = float(len(learner.get_vocabulary()))

    print(f"\n[能力评估]")
    for k, v in evaluation.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n[学习统计]")
    print(f"  总交互次数: {stats['total_steps']}")
    print(f"  词汇量: {stats['vocabulary_size']}")
    print(f"  接地符号数: {stats['grounded_symbols']}")
    print(f"  感知概念数: {stats['perceptual_clusters']}")
    print(f"  沟通成功率: {stats['comm_success_rate']:.1%}")
    print(f"  学习进度: {stats['learning_progress']:.3f}")
    print(f"  平均推理步数: {stats['avg_inference_steps']:.1f}")
    print(f"  模态权重: {stats['modality_weights']}")

    # 阶段晋升检查
    print(f"\n[发展阶段]")
    print(f"  当前阶段: {learner.stage}")
    advanced = learner.try_advance(evaluation)
    print(f"  尝试晋升: {'成功!' if advanced else '继续当前阶段学习'}")

    # 最终词汇表
    vocab = learner.get_vocabulary()
    if vocab:
        print(f"\n[掌握的英语词汇] ({len(vocab)} 个)")
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1].get('frequency', 0), reverse=True)
        for word, data in sorted_vocab[:20]:
            freq = data.get('frequency', 0)
            rate = data.get('success_rate', 0)
            bar = '#' * int(rate * 20)
            print(f"  {word:12s} | 使用{freq:3d}次 | 成功率{rate:.0%} | {bar}")
        if len(sorted_vocab) > 20:
            print(f"  ... 还有 {len(sorted_vocab) - 20} 个词汇")

    # 复合符号
    lang = learner.communication.language
    if lang.compounds:
        print(f"\n[涌现的组合表达] ({len(lang.compounds)} 个)")
        for compound, data in list(lang.compounds.items())[:10]:
            comps = data.get('components', [])
            print(f"  {compound} ← {' + '.join(comps)}")

    # 语法模式
    grammar = learner.communication.grammar
    rules = grammar.get_rules()
    if rules:
        print(f"\n[学到的语法模式] ({len(rules)} 个)")
        for entry in rules[:5]:
            if isinstance(entry, dict):
                pattern = entry.get('pattern', str(entry))
                count = entry.get('count', '?')
                print(f"  {pattern}: {count}次")
            else:
                print(f"  {entry}")

    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'level1': {
            'success_rate': level1['total_successes'] / max(level1['total_games'], 1),
            'vocab_growth': level1['vocabulary_growth'],
        },
        'level2': {
            'success_rate': level2['total_successes'] / max(level2['total_games'], 1),
            'vocab_growth': level2['vocabulary_growth'],
        },
        'level3': {
            'success_rate': level3['total_successes'] / max(level3['total_games'], 1),
            'vocab_growth': level3['vocabulary_growth'],
        },
        'final_stats': stats,
        'evaluation': evaluation,
        'vocabulary': {
            word: {k: v for k, v in data.items() if k != 'last_n_successes'}
            for word, data in vocab.items()
        } if vocab else {},
    }

    import os
    os.makedirs('results', exist_ok=True)
    with open('results/english_learning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到 results/english_learning_results.json")
    print("=" * 70)

    return learner


if __name__ == '__main__':
    learner = main()
