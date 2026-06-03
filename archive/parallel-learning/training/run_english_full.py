"""
通用学习系统 — 学完小学全部英语

6 个年级，循序渐进，像小朋友一样从 Grade 1 学到 Grade 6。
每学完一个年级自动保存，下次可以直接加载继续。

保存/恢复验证：每学完一个年级，保存 → 重新加载 → 验证一致 → 继续。
"""

import sys
sys.path.insert(0, '.')

import json
import time
import random
import os
import torch
from datetime import datetime

from src.core.config import LearnerConfig
from src.core.learner import Learner
from src.language.communication import CommunicationProtocol, generate_scene
from src.curriculum.evaluator import CapabilityEvaluator

CHECKPOINT_DIR = 'checkpoints/english'
RESULTS_DIR = 'results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 小学英语词汇库 — 人教版(PEP) 1-6 年级
# ══════════════════════════════════════════════════════════════════

GRADES = {
    1: {
        'name': 'Grade 1 — 启蒙期',
        'rounds': 200,
        'num_objects': 2,
        'attributes': {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
            'shape': ['circle', 'square', 'triangle', 'star'],
            'animal': ['cat', 'dog', 'bird', 'fish', 'pig', 'duck'],
        },
    },
    2: {
        'name': 'Grade 2 — 基础词汇扩展',
        'rounds': 250,
        'num_objects': 3,
        'attributes': {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'brown', 'purple'],
            'shape': ['circle', 'square', 'triangle', 'star', 'heart', 'diamond'],
            'animal': ['cat', 'dog', 'bird', 'fish', 'pig', 'duck', 'rabbit', 'monkey', 'bear', 'tiger', 'lion', 'elephant'],
            'size': ['big', 'small', 'long', 'short'],
            'number': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
        },
    },
    3: {
        'name': 'Grade 3 — 学校与家庭',
        'rounds': 300,
        'num_objects': 3,
        'attributes': {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'brown', 'purple'],
            'shape': ['circle', 'square', 'triangle', 'star', 'heart', 'diamond'],
            'animal': ['cat', 'dog', 'bird', 'fish', 'pig', 'duck', 'rabbit', 'monkey', 'bear', 'tiger', 'lion', 'elephant', 'snake', 'frog', 'horse', 'sheep'],
            'size': ['big', 'small', 'long', 'short', 'tall'],
            'school': ['book', 'pen', 'bag', 'desk', 'chair', 'ruler', 'pencil', 'eraser'],
            'body': ['head', 'eye', 'ear', 'nose', 'mouth', 'hand', 'foot', 'arm', 'leg'],
            'family': ['mom', 'dad', 'brother', 'sister', 'grandma', 'grandpa'],
        },
    },
    4: {
        'name': 'Grade 4 — 食物与日常',
        'rounds': 300,
        'num_objects': 4,
        'attributes': {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'brown', 'purple'],
            'animal': ['cat', 'dog', 'bird', 'fish', 'pig', 'duck', 'rabbit', 'monkey', 'bear', 'tiger', 'lion', 'elephant', 'snake', 'frog', 'horse', 'sheep'],
            'food': ['apple', 'banana', 'cake', 'egg', 'milk', 'rice', 'bread', 'juice', 'water', 'chicken', 'fish', 'noodle'],
            'clothes': ['shirt', 'dress', 'hat', 'shoe', 'coat', 'pants', 'sock', 'jacket'],
            'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'hot', 'cold', 'warm'],
            'place': ['school', 'home', 'park', 'shop', 'farm', 'zoo', 'library'],
            'action': ['run', 'jump', 'swim', 'fly', 'eat', 'drink', 'read', 'write', 'sing', 'dance', 'play', 'sleep'],
        },
    },
    5: {
        'name': 'Grade 5 — 时间与方位',
        'rounds': 350,
        'num_objects': 4,
        'attributes': {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'brown', 'purple', 'grey'],
            'animal': ['cat', 'dog', 'bird', 'fish', 'pig', 'duck', 'rabbit', 'monkey', 'bear', 'tiger', 'lion', 'elephant', 'snake', 'frog', 'horse', 'sheep', 'whale', 'shark', 'eagle', 'panda'],
            'food': ['apple', 'banana', 'cake', 'egg', 'milk', 'rice', 'bread', 'juice', 'water', 'chicken', 'fish', 'noodle', 'beef', 'soup', 'salad', 'pizza', 'sandwich', 'grape', 'orange', 'mango'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow', 'yesterday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
            'direction': ['left', 'right', 'up', 'down', 'near', 'far', 'here', 'there', 'front', 'back', 'inside', 'outside'],
            'subject': ['math', 'english', 'music', 'art', 'science', 'sport', 'computer'],
            'emotion': ['happy', 'sad', 'angry', 'tired', 'sick', 'scared', 'excited'],
            'transport': ['car', 'bus', 'bike', 'train', 'plane', 'boat', 'taxi', 'subway'],
        },
    },
    6: {
        'name': 'Grade 6 — 综合+复习',
        'rounds': 400,
        'num_objects': 5,
        'attributes': {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'brown', 'purple', 'grey', 'gold', 'silver'],
            'animal': ['cat', 'dog', 'bird', 'fish', 'pig', 'duck', 'rabbit', 'monkey', 'bear', 'tiger', 'lion', 'elephant', 'snake', 'frog', 'horse', 'sheep', 'whale', 'shark', 'eagle', 'panda', 'giraffe', 'penguin', 'dolphin', 'koala'],
            'food': ['apple', 'banana', 'cake', 'egg', 'milk', 'rice', 'bread', 'juice', 'water', 'chicken', 'fish', 'noodle', 'beef', 'soup', 'salad', 'pizza', 'sandwich', 'grape', 'orange', 'mango', 'tomato', 'potato', 'carrot', 'onion', 'strawberry', 'watermelon'],
            'job': ['teacher', 'doctor', 'farmer', 'driver', 'nurse', 'cook', 'pilot', 'police', 'scientist', 'artist', 'writer', 'singer'],
            'hobby': ['reading', 'painting', 'cooking', 'fishing', 'hiking', 'camping', 'swimming', 'dancing', 'climbing', 'running'],
            'nature': ['river', 'mountain', 'forest', 'ocean', 'island', 'desert', 'lake', 'rainbow', 'flower', 'tree', 'grass', 'rock'],
            'adj': ['beautiful', 'clever', 'brave', 'kind', 'funny', 'strong', 'fast', 'slow', 'old', 'young', 'rich', 'poor', 'clean', 'dirty'],
            'place': ['school', 'home', 'park', 'shop', 'farm', 'zoo', 'library', 'hospital', 'cinema', 'museum', 'airport', 'restaurant', 'hotel', 'stadium'],
        },
    },
}


def make_scene(grade_config: dict) -> list:
    """按年级配置生成场景"""
    attrs = grade_config['attributes']
    attr_keys = list(attrs.keys())
    n = grade_config['num_objects']

    scene = []
    for _ in range(n):
        obj = {}
        n_attrs = random.randint(2, min(3, len(attr_keys)))
        chosen = random.sample(attr_keys, n_attrs)
        for key in chosen:
            obj[key] = random.choice(attrs[key])
        scene.append(obj)
    return scene


def scene_to_raw_input(scene: list) -> dict:
    """把场景字典转为感知编码器的输入格式

    视觉：场景物体的属性 hash → (4, 8, 8) feature map
    听觉：属性类别 one-hot → (7,)
    位置：场景物体数量归一化 → (2,)
    """
    visual = torch.zeros(4, 8, 8)
    for i, obj in enumerate(scene[:4]):
        for j, (key, val) in enumerate(obj.items()):
            if j >= 8:
                break
            seed = hash(f"{key}:{val}") % 10000
            g = torch.Generator().manual_seed(seed)
            visual[i, j, :] = torch.randn(8, generator=g) * 0.5 + 0.5

    categories = ['color', 'shape', 'size', 'animal', 'food', 'verb', 'action',
                  'agent', 'patient', 'adj', 'adverb', 'place', 'modal']
    auditory = torch.zeros(13)
    for obj in scene:
        for key in obj:
            if key in categories:
                auditory[categories.index(key)] += 1.0
    if auditory.sum() > 0:
        auditory = auditory / auditory.sum()

    position = torch.tensor([len(scene) / 10.0, random.random()])

    return {'visual': visual, 'auditory': auditory, 'position': position}


def teach_grade(learner: Learner, grade: int, verbose: bool = True) -> dict:
    """教一个年级的英语

    双通道学习：
    1. 感知-预测闭环（GPU）→ 像儿童不停感知和预测环境
    2. 参照游戏（符号层）→ 像儿童玩指物认字游戏
    """
    gc = GRADES[grade]
    total_successes = 0
    total_games = 0

    for r in range(gc['rounds']):
        scene = make_scene(gc)
        target = random.randint(0, len(scene) - 1)

        # ── 通道 1：感知-预测闭环（GPU 工作）──
        raw_input = scene_to_raw_input(scene)
        obs = learner.perceive(raw_input)
        action = learner.choose_action(obs)

        next_scene = make_scene(gc)
        next_raw = scene_to_raw_input(next_scene)
        next_obs = learner.perceive(next_raw)

        error = learner.learn_from_experience(obs, action, next_obs)
        learner.remember(obs, action, next_obs, reward=1.0, error=error)

        # ── 通道 2：参照游戏（符号层）──
        success = learner.play_reference_game(scene, target)

        total_games += 1
        if success:
            total_successes += 1

        # 无论成功失败都接地 — 像儿童即使猜错也在学习
        target_obj = scene[target]
        symbols_contexts = [(val, attr) for attr, val in target_obj.items() if val]
        if symbols_contexts:
            learner.grounding.ground_symbols_batch(symbols_contexts, obs)

        if (r + 1) % 50 == 0 and verbose:
            rate = total_successes / total_games
            vocab = len(learner.get_vocabulary())
            grounded = len(learner.grounding.get_grounded_symbols())
            errors = list(learner._error_history)[-20:] if learner._error_history else [0]
            avg_err = sum(errors) / len(errors)
            print(f"    Round {r+1:4d}/{gc['rounds']} | "
                  f"成功率 {rate:.1%} | 词汇 {vocab} | 接地 {grounded} | "
                  f"预测误差 {avg_err:.3f}")

    # 学完巩固
    learner.consolidate()

    rate = total_successes / max(total_games, 1)
    return {'success_rate': rate, 'total_games': total_games, 'total_successes': total_successes}


def verify_save_load(learner: Learner, grade: int) -> bool:
    """保存 → 加载 → 验证一致性"""
    path = os.path.join(CHECKPOINT_DIR, f'grade_{grade}.pt')
    learner.save(path)

    # 记录保存前的状态
    vocab_before = len(learner.get_vocabulary())
    grounded_before = len(learner.grounding.get_grounded_symbols())
    stage_before = learner.stage
    steps_before = learner._total_steps

    # 重新加载
    config = learner.config
    loaded = Learner(config)
    loaded.load(path)

    # 验证
    vocab_after = len(loaded.get_vocabulary())
    grounded_after = len(loaded.grounding.get_grounded_symbols())
    stage_after = loaded.stage
    steps_after = loaded._total_steps

    ok = (vocab_before == vocab_after and
          grounded_before == grounded_after and
          stage_before == stage_after and
          steps_before == steps_after and
          torch.allclose(learner.engine.W1, loaded.engine.W1))

    return ok


def main():
    print("=" * 70)
    print("通用学习系统 — 学完小学全部英语 (Grade 1 ~ Grade 6)")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    config = LearnerConfig(
        obs_dim=40, action_dim=4, hidden_dims=(32, 16),
        working_memory_capacity=7, episodic_memory_capacity=2000,
        consolidation_interval=50,
    )

    # 检查是否有之前的检查点可以恢复
    latest_grade = 0
    learner = Learner(config)
    for g in range(1, 7):
        path = os.path.join(CHECKPOINT_DIR, f'grade_{g}.pt')
        if os.path.exists(path):
            latest_grade = g

    if latest_grade > 0:
        path = os.path.join(CHECKPOINT_DIR, f'grade_{latest_grade}.pt')
        learner.load(path)
        print(f"[恢复] 找到 Grade {latest_grade} 检查点，继续学习")
        print(f"  已有词汇: {len(learner.get_vocabulary())}")
        print(f"  已有接地: {len(learner.grounding.get_grounded_symbols())}")
        print(f"  已有概念: {len(learner.grounding.perceptual_clusters)}")
        print()
    else:
        print("[全新] 从零开始学习")
        print()

    all_results = {}
    start_time = time.time()

    for grade in range(latest_grade + 1, 7):
        gc = GRADES[grade]
        print(f"{'='*70}")
        print(f"Grade {grade}: {gc['name']}")
        print(f"  词汇范围: {', '.join(list(gc['attributes'].keys()))}")
        print(f"  练习轮数: {gc['rounds']}")
        print(f"{'='*70}")

        t0 = time.time()
        result = teach_grade(learner, grade)
        t1 = time.time()

        # 评估 + 阶段晋升
        evaluator = CapabilityEvaluator()
        evaluation = evaluator.evaluate(learner)
        stage_before = learner.stage
        advanced = learner.try_advance(evaluation)
        stage_after = learner.stage

        # 保存 + 验证
        save_ok = verify_save_load(learner, grade)

        vocab = len(learner.get_vocabulary())
        compounds = list(learner.communication.language.compounds.keys())[:5]

        print(f"\n  [Grade {grade} 完成] {t1-t0:.1f}s")
        print(f"  成功率: {result['success_rate']:.1%} ({result['total_successes']}/{result['total_games']})")
        print(f"  累计词汇: {vocab}")
        print(f"  评估: 词汇={evaluation['vocabulary_size']:.0f} 组合率={evaluation['composition_rate']:.2f} "
              f"语法={evaluation['grammar_complexity']:.2f}")
        if advanced:
            print(f"  阶段晋升: {stage_before} → {stage_after}")
        else:
            print(f"  当前阶段: {stage_after}")
        print(f"  保存验证: {'通过' if save_ok else '失败!'}")
        if compounds:
            print(f"  组合表达: {compounds}")
        print()

        all_results[f'grade_{grade}'] = {
            **result,
            'vocab_after': vocab,
            'time_seconds': round(t1 - t0, 1),
            'save_verified': save_ok,
            'stage': stage_after,
            'advanced': advanced,
            'evaluation': {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in evaluation.items()},
        }

    # ═══════════════════════════════════════════════════════════════
    # 最终评估
    # ═══════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    stats = learner.get_stats()

    print("=" * 70)
    print("小学英语学习完成 — 最终评估")
    print("=" * 70)

    print(f"\n[总耗时] {total_time:.1f}s")
    print(f"[总交互] {stats['comm_games']} 次参照游戏")
    print(f"[最终词汇] {stats['vocabulary_size']} 个")
    print(f"[接地符号] {stats['grounded_symbols']} 个")
    print(f"[感知概念] {stats['perceptual_clusters']} 个")
    print(f"[沟通成功率] {stats['comm_success_rate']:.1%}")
    print(f"[发展阶段] {learner.stage}")

    # 各年级成绩
    print(f"\n[各年级成绩]")
    for g in range(1, 7):
        key = f'grade_{g}'
        if key in all_results:
            r = all_results[key]
            bar = '#' * int(r['success_rate'] * 20)
            stage = r.get('stage', '?')
            adv = '↑' if r.get('advanced') else ' '
            print(f"  Grade {g}: {r['success_rate']:.1%} {bar} | "
                  f"词汇 {r['vocab_after']} | {r['time_seconds']}s | "
                  f"阶段 {stage}{adv} | "
                  f"保存 {'OK' if r['save_verified'] else 'FAIL'}")

    # 词汇表
    vocab = learner.get_vocabulary()
    if vocab:
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1].get('frequency', 0), reverse=True)
        print(f"\n[掌握的全部英语词汇] ({len(sorted_vocab)} 个)")
        for word, data in sorted_vocab:
            freq = data.get('frequency', 0)
            rate = data.get('success_rate', 0)
            bar = '#' * max(1, int(rate * 15))
            print(f"  {word:15s} | {freq:3d}次 | {rate:.0%} | {bar}")

    # 复合符号
    compounds = learner.communication.language.compounds
    if compounds:
        print(f"\n[涌现的组合表达] ({len(compounds)} 个)")
        for compound, data in list(compounds.items())[:20]:
            comps = data.get('components', [])
            print(f"  {compound:25s} <- {' + '.join(comps)}")

    # 最终保存
    learner.save(os.path.join(CHECKPOINT_DIR, 'grade_6_final.pt'))

    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': round(total_time, 1),
        'grades': all_results,
        'final_stats': {k: v for k, v in stats.items() if not isinstance(v, dict)},
        'vocabulary': {
            word: {k: v for k, v in data.items() if k != 'last_n_successes'}
            for word, data in vocab.items()
        },
    }
    with open(os.path.join(RESULTS_DIR, 'english_full_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n检查点保存在: {CHECKPOINT_DIR}/")
    print(f"结果保存在: {RESULTS_DIR}/english_full_results.json")
    print("\n下次运行会自动从最新检查点恢复，继续学习。")
    print("=" * 70)

    return learner


if __name__ == '__main__':
    learner = main()
