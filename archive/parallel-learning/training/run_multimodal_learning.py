"""
多模态内容驱动学习 — 主训练脚本

像真实学生一样从文章中学习听说读写。
加载 content/ 目录的真实文本，执行 5 通道并行学习。

用法：python training/run_multimodal_learning.py
"""

import os
import sys
import json
import time
import torch
import random

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.config import LearnerConfig
from src.core.learner import Learner
from src.core.device import get_device
from src.curriculum.evaluator import CapabilityEvaluator
from src.content.loader import ContentLoader
from src.content.analyzer import DifficultyAnalyzer
from src.content.text_unit import TextUnit
from src.perception.text_encoder import TextEncoder
from src.skills.reading import generate_cloze
from src.skills.writing import generate_scramble, evaluate_sentence
from src.skills.listening import generate_listen_exercise, text_to_audio_features
from src.skills.grammar_ex import extract_patterns, generate_pattern_completion, generate_error_correction


def load_or_create_learner(checkpoint_path: str) -> Learner:
    """加载已有学习体或创建新的"""
    config = LearnerConfig()

    if os.path.exists(checkpoint_path):
        print(f"[加载] 从 {checkpoint_path} 恢复学习体...")
        learner = Learner(config)
        learner.load(checkpoint_path)
        vocab_size = len(learner.get_vocabulary())
        print(f"[加载] 阶段={learner.stage}, 词汇量={vocab_size}")
    else:
        print("[新建] 创建全新学习体")
        learner = Learner(config)

    return learner


def get_known_vocabulary(learner: Learner) -> set:
    """从学习体中提取已知词汇"""
    vocab = learner.get_vocabulary()
    known = set()
    for word, data in vocab.items():
        # 至少有 3 次暴露且成功率 > 20% 的词算"已知"
        freq = data.get('frequency', 0)
        exposures = data.get('exposures', freq)
        rate = data.get('success_rate', 0)
        if exposures >= 3 or rate > 0.2:
            known.add(word)
    return known


def run_text_learning(learner, encoder, text_unit, known_vocab):
    """通道 1: 文本编码 → 感知预测闭环

    模拟真实阅读：当前句子的表示 → 预测下一句 → 比较实际下一句
    """
    sentences = text_unit.sentences or [text_unit.text]
    total_error = 0.0
    count = 0

    for i in range(len(sentences)):
        raw_input = encoder.encode_sentence(sentences[i])
        obs = learner.perceive(raw_input)

        if i + 1 < len(sentences):
            # 有下一句：用当前句预测下一句（真正的预测误差）
            next_raw = encoder.encode_sentence(sentences[i + 1])
            next_obs = learner.perceive(next_raw)
            error = learner.learn_from_experience(obs, None, next_obs)
        else:
            # 最后一句：用不同编码产生对比
            # 用段落首句编码作为"期望" — 模拟对段落主题的预期
            first_raw = encoder.encode_sentence(sentences[0])
            expected_obs = learner.perceive(first_raw)
            error = learner.learn_from_experience(obs, None, expected_obs)

        total_error += error
        count += 1

        # 接地：把文本中的词与感知体验关联
        for word in text_unit.unique_words:
            if word not in known_vocab:
                learner.ground_symbol(word, obs, context=text_unit.text[:50])

    return total_error / max(count, 1)


def run_reading_skill(learner, comm, text_unit, known_vocab):
    """通道 2: 完形填空"""
    items = generate_cloze(text_unit, blank_ratio=0.2, known_vocabulary=known_vocab)
    if not items:
        return []

    results = []
    for item in items:
        success = comm.play_cloze_game(item.original, item.target_word, item.options)
        learner.reading_history.append(success)
        results.append(success)

    return results


def run_writing_skill(learner, comm, text_unit):
    """通道 3: 句子重组"""
    items = generate_scramble(text_unit)
    if not items:
        return []

    results = []
    for item in items:
        success = comm.play_sentence_game(item.target_order, item.scrambled)
        learner.writing_history.append(success)
        results.append(success)

    return results


def run_listening_skill(learner, comm, text_unit):
    """通道 4: 听力匹配"""
    sentences = text_unit.sentences or [text_unit.text]
    items = generate_listen_exercise(text_unit, num_options=3,
                                     distractor_pool=[s for s in sentences if s != text_unit.text])
    if not items:
        return []

    results = []
    for item in items:
        success = comm.play_listening_game(item.target, item.text_options)
        learner.listening_history.append(success)
        results.append(success)

    return results


def run_grammar_skill(learner, comm, text_unit, known_vocab):
    """通道 5: 语法模式 — 真正尝试填空"""
    patterns = extract_patterns(text_unit)
    if not patterns:
        return []

    results = []
    for pattern in patterns[:3]:
        exercise = generate_pattern_completion(pattern, known_words=known_vocab)
        if exercise.target and exercise.options:
            # 暴露所有相关词
            for w in pattern.words:
                comm.language.expose_symbol(w)

            # 真正尝试：从选项中选（用 cloze 游戏机制）
            success = comm.play_cloze_game(
                exercise.prompt, exercise.target, exercise.options
            )
            learner.grammar_history.append(success)
            results.append(success)

    return results


def print_progress(round_num, learner, skill_results):
    """打印进度"""
    stats = learner.get_stats()
    vocab_size = stats['vocabulary_size']
    stage = stats['stage']
    comm_rate = stats['comm_success_rate']

    reading_acc = stats.get('reading_accuracy', 0)
    writing_acc = stats.get('writing_accuracy', 0)
    listening_acc = stats.get('listening_accuracy', 0)
    grammar_acc = stats.get('grammar_accuracy', 0)

    print(f"\n{'='*60}")
    print(f"  轮次 {round_num} | 阶段: {stage} | 词汇量: {vocab_size}")
    print(f"  沟通成功率: {comm_rate:.1%}")
    print(f"  阅读: {reading_acc:.1%} | 写作: {writing_acc:.1%} "
          f"| 听力: {listening_acc:.1%} | 语法: {grammar_acc:.1%}")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("  多模态内容驱动学习系统")
    print("  像真实学生一样从文章中学习听说读写")
    print("=" * 60)

    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'multimodal.pt')

    # 初始化
    learner = load_or_create_learner(checkpoint_path)
    encoder = TextEncoder(device=learner.config.device)
    evaluator = CapabilityEvaluator()
    loader = ContentLoader(os.path.join(PROJECT_ROOT, 'content'))
    comm = learner.communication

    # 加载内容
    print("\n[内容] 加载学习材料...")
    all_texts = []
    for level in ['elementary', 'intermediate', 'advanced']:
        texts = loader.load_by_difficulty(level)
        for text in texts:
            all_texts.append((level, text))
        print(f"  {level}: {len(texts)} 篇")

    if not all_texts:
        print("[错误] content/ 目录下没有学习材料！")
        return

    print(f"\n[内容] 共 {len(all_texts)} 篇文本")

    # 训练配置
    num_rounds = 5
    results_log = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- 第 {round_num}/{num_rounds} 轮 ---")
        known_vocab = get_known_vocabulary(learner)
        round_results = {'round': round_num}
        skill_totals = {'reading': 0, 'writing': 0, 'listening': 0, 'grammar': 0}
        skill_success = {'reading': 0, 'writing': 0, 'listening': 0, 'grammar': 0}

        # 按难度排序（先易后难）
        level_order = {'elementary': 0, 'intermediate': 1, 'advanced': 2}
        sorted_texts = sorted(all_texts, key=lambda x: level_order.get(x[0], 0))

        analyzer = DifficultyAnalyzer(known_vocabulary=known_vocab)

        for level, text in sorted_texts:
            units = analyzer.extract_learnable_units(text, source=level)
            if not units:
                continue

            for unit in units[:3]:  # 每篇文本最多 3 个学习单元
                # 通道 1: 感知预测
                run_text_learning(learner, encoder, unit, known_vocab)

                # 通道 2: 阅读（完形填空）
                r_results = run_reading_skill(learner, comm, unit, known_vocab)
                skill_totals['reading'] += len(r_results)
                skill_success['reading'] += sum(r_results)

                # 通道 3: 写作（句子重组）
                w_results = run_writing_skill(learner, comm, unit)
                skill_totals['writing'] += len(w_results)
                skill_success['writing'] += sum(w_results)

                # 通道 4: 听力
                l_results = run_listening_skill(learner, comm, unit)
                skill_totals['listening'] += len(l_results)
                skill_success['listening'] += sum(l_results)

                # 通道 5: 语法
                g_results = run_grammar_skill(learner, comm, unit, known_vocab)
                skill_totals['grammar'] += len(g_results)
                skill_success['grammar'] += sum(g_results)

        # 巩固记忆
        learner.consolidate()

        # 评估
        evaluation = evaluator.evaluate(learner)
        advanced = learner.try_advance(evaluation)

        # 打印进度
        print_progress(round_num, learner, skill_totals)

        for skill in ['reading', 'writing', 'listening', 'grammar']:
            total = skill_totals[skill]
            succ = skill_success[skill]
            if total > 0:
                print(f"  {skill}: {succ}/{total} ({succ/total:.1%})")

        if advanced:
            print(f"  >>> 晋升到新阶段: {learner.stage}！")

        round_results.update({
            'stage': learner.stage,
            'vocabulary_size': evaluation['vocabulary_size'],
            'comm_success_rate': evaluation['communication_success'],
            'skills': {
                k: f"{skill_success[k]}/{skill_totals[k]}"
                for k in skill_totals
            },
        })
        results_log.append(round_results)

        # 更新已知词汇
        known_vocab = get_known_vocabulary(learner)

    # 保存
    learner.save(checkpoint_path)
    print(f"\n[保存] 检查点 → {checkpoint_path}")

    # 最终统计
    stats = learner.get_stats()
    print("\n" + "=" * 60)
    print("  最终学习结果")
    print("=" * 60)
    print(f"  阶段: {stats['stage']}")
    print(f"  词汇量: {stats['vocabulary_size']}")
    print(f"  沟通成功率: {stats['comm_success_rate']:.1%}")
    print(f"  阅读准确率: {stats.get('reading_accuracy', 0):.1%}")
    print(f"  写作准确率: {stats.get('writing_accuracy', 0):.1%}")
    print(f"  听力准确率: {stats.get('listening_accuracy', 0):.1%}")
    print(f"  语法准确率: {stats.get('grammar_accuracy', 0):.1%}")
    print(f"  感知聚类: {stats['perceptual_clusters']}")
    print(f"  接地符号: {stats['grounded_symbols']}")

    # 保存结果日志
    results_path = os.path.join(PROJECT_ROOT, 'mvl', 'multimodal_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'final_stats': {k: v for k, v in stats.items()
                           if isinstance(v, (int, float, str))},
            'rounds': results_log,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[结果] → {results_path}")


if __name__ == '__main__':
    main()
