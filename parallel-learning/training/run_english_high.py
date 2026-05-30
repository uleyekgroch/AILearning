"""
通用学习系统 — 高中英语（10-12 年级）

从小学检查点加载继续学习。系统是通用学习 AI，
给它高中场景它自然学会高中英语。

场景类型扩展：物体参照 + 事件参照
  物体场景: {'color': 'red', 'shape': 'circle'} → 哪个物体？
  事件场景: {'agent': 'cat', 'action': 'eat', 'patient': 'fish'} → 哪个事件？

事件场景仍是 Dict[str, str]，参照游戏协议不需要改动。
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
from src.language.communication import CommunicationProtocol
from src.curriculum.evaluator import CapabilityEvaluator

CHECKPOINT_DIR = 'checkpoints/english'
RESULTS_DIR = 'results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 高中英语词汇库 — 人教版(PEP) 10-12 年级
# ══════════════════════════════════════════════════════════════════

GRADES = {
    10: {
        'name': 'Grade 10 (高一) — 学术英语入门',
        'rounds': 400,
        'num_objects': 5,
        'event_ratio': 0.4,
        'attributes': {
            'verb': ['walk', 'talk', 'think', 'know', 'understand', 'believe', 'remember',
                     'learn', 'study', 'teach', 'help', 'want', 'need', 'like', 'love',
                     'try', 'start', 'stop', 'begin', 'finish', 'open', 'close',
                     'give', 'take', 'buy', 'sell', 'make', 'bring', 'send',
                     'show', 'tell', 'ask', 'answer', 'call', 'meet', 'find', 'keep',
                     'discover', 'invent', 'protect', 'destroy', 'create', 'imagine',
                     'decide', 'choose', 'compare', 'explain', 'describe', 'suggest',
                     'develop', 'improve', 'achieve', 'consider', 'include', 'provide',
                     'analyze', 'evaluate', 'organize', 'establish', 'recognize', 'indicate'],
            'pronoun': ['i', 'you', 'he', 'she', 'it', 'we', 'they',
                        'me', 'him', 'her', 'us', 'them',
                        'my', 'your', 'his', 'its', 'our', 'their',
                        'this', 'that', 'these', 'those'],
            'modal': ['will', 'would', 'can', 'could', 'should', 'may', 'might', 'must'],
            'adverb': ['carefully', 'quickly', 'slowly', 'happily', 'sadly', 'quietly',
                       'loudly', 'suddenly', 'finally', 'usually', 'often', 'never',
                       'always', 'sometimes', 'already', 'still', 'just', 'almost'],
            'abstract_adj': ['important', 'difficult', 'easy', 'possible', 'impossible',
                             'necessary', 'true', 'false', 'real', 'main',
                             'different', 'similar', 'special', 'common', 'certain',
                             'obvious', 'essential', 'significant', 'remarkable', 'permanent'],
            'relationship': ['friend', 'classmate', 'neighbor', 'stranger', 'leader',
                             'member', 'partner', 'team', 'group', 'family', 'society',
                             'teacher', 'student', 'doctor', 'engineer', 'scientist'],
            'place': ['school', 'home', 'park', 'shop', 'library', 'hospital',
                      'classroom', 'playground', 'office', 'factory',
                      'airport', 'station', 'bank', 'supermarket', 'theater',
                      'university', 'laboratory', 'community', 'countryside'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow', 'yesterday',
                     'now', 'then', 'soon', 'later', 'before', 'after'],
            'discourse': ['topic', 'opinion', 'reason', 'example', 'fact', 'idea',
                          'problem', 'solution', 'advantage', 'disadvantage', 'conclusion',
                          'introduction', 'argument', 'evidence', 'summary', 'detail'],
            'science': ['experiment', 'theory', 'result', 'method', 'energy', 'force',
                        'speed', 'distance', 'temperature', 'pressure', 'electricity',
                        'atom', 'molecule', 'cell', 'organism', 'evolution'],
        },
        'event_roles': ['agent', 'action', 'patient', 'location', 'time', 'manner'],
    },
    11: {
        'name': 'Grade 11 (高二) — 批判性思维',
        'rounds': 450,
        'num_objects': 5,
        'event_ratio': 0.5,
        'attributes': {
            'verb': ['walk', 'talk', 'think', 'know', 'understand', 'believe', 'remember',
                     'learn', 'study', 'teach', 'help', 'want', 'need', 'like', 'love', 'hate',
                     'try', 'start', 'stop', 'begin', 'finish', 'open', 'close',
                     'give', 'take', 'buy', 'sell', 'make', 'bring', 'send',
                     'discover', 'invent', 'protect', 'destroy', 'create', 'imagine',
                     'decide', 'choose', 'compare', 'explain', 'describe', 'suggest',
                     'develop', 'improve', 'achieve', 'consider', 'include', 'provide',
                     'analyze', 'evaluate', 'organize', 'establish', 'recognize', 'indicate',
                     'assume', 'demonstrate', 'emphasize', 'investigate', 'participate',
                     'contribute', 'accomplish', 'represent', 'transform', 'negotiate'],
            'pronoun': ['i', 'you', 'he', 'she', 'it', 'we', 'they',
                        'me', 'him', 'her', 'us', 'them',
                        'my', 'your', 'his', 'its', 'our', 'their',
                        'someone', 'everyone', 'anything', 'nothing', 'whatever', 'whoever'],
            'modal': ['will', 'would', 'can', 'could', 'should', 'may', 'might', 'must'],
            'adverb': ['carefully', 'quickly', 'slowly', 'happily', 'sadly', 'quietly',
                       'loudly', 'suddenly', 'finally', 'usually', 'often', 'never',
                       'always', 'sometimes', 'already', 'still', 'just', 'almost',
                       'actually', 'especially', 'probably', 'certainly', 'simply',
                       'furthermore', 'nevertheless', 'meanwhile', 'therefore', 'otherwise'],
            'abstract_adj': ['important', 'difficult', 'easy', 'possible', 'impossible',
                             'necessary', 'true', 'false', 'real', 'main',
                             'different', 'similar', 'special', 'common', 'certain',
                             'obvious', 'essential', 'significant', 'remarkable', 'permanent',
                             'controversial', 'independent', 'responsible', 'sustainable', 'flexible'],
            'world': ['country', 'city', 'capital', 'population', 'language', 'culture',
                      'history', 'geography', 'economy', 'technology', 'environment',
                      'pollution', 'protection', 'development', 'education', 'tradition',
                      'government', 'law', 'freedom', 'peace', 'war'],
            'complex_adj': ['ancient', 'modern', 'traditional', 'international', 'national',
                            'local', 'personal', 'social', 'physical', 'mental',
                            'natural', 'artificial', 'political', 'economic', 'scientific',
                            'creative', 'practical', 'theoretical', 'visible', 'invisible'],
            'relationship': ['friend', 'classmate', 'neighbor', 'leader',
                             'member', 'partner', 'team', 'group', 'society',
                             'teacher', 'student', 'doctor', 'engineer', 'scientist',
                             'parent', 'child', 'adult', 'teenager', 'citizen',
                             'president', 'minister', 'professor', 'journalist', 'volunteer'],
            'academic': ['chapter', 'paragraph', 'sentence', 'word', 'article',
                         'essay', 'report', 'speech', 'conversation', 'discussion', 'debate',
                         'dictionary', 'encyclopedia', 'magazine', 'newspaper',
                         'thesis', 'hypothesis', 'research', 'survey', 'statistic'],
            'place': ['school', 'home', 'park', 'library', 'hospital',
                      'classroom', 'office', 'factory', 'university', 'laboratory',
                      'museum', 'cinema', 'restaurant', 'hotel', 'stadium',
                      'community', 'countryside', 'downtown', 'suburb', 'coast'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow', 'yesterday',
                     'now', 'then', 'soon', 'later', 'before', 'after',
                     'century', 'decade', 'year', 'month', 'week', 'period', 'era'],
        },
        'event_roles': ['agent', 'action', 'patient', 'location', 'time', 'manner', 'reason'],
    },
    12: {
        'name': 'Grade 12 (高三) — 综合应用',
        'rounds': 500,
        'num_objects': 6,
        'event_ratio': 0.55,
        'attributes': {
            'verb': ['walk', 'talk', 'think', 'know', 'understand', 'believe', 'remember',
                     'learn', 'study', 'teach', 'help', 'want', 'need', 'like', 'love', 'hate', 'hope', 'wish',
                     'try', 'start', 'stop', 'begin', 'finish', 'open', 'close',
                     'give', 'take', 'buy', 'sell', 'make', 'bring', 'send',
                     'discover', 'invent', 'protect', 'destroy', 'create', 'imagine',
                     'decide', 'choose', 'compare', 'explain', 'describe', 'suggest',
                     'develop', 'improve', 'achieve', 'consider', 'include', 'provide',
                     'analyze', 'evaluate', 'organize', 'establish', 'recognize', 'indicate',
                     'assume', 'demonstrate', 'emphasize', 'investigate', 'participate',
                     'contribute', 'accomplish', 'represent', 'transform', 'negotiate',
                     'advocate', 'implement', 'anticipate', 'collaborate', 'compromise',
                     'distinguish', 'implement', 'incorporate', 'prioritize', 'validate'],
            'pronoun': ['i', 'you', 'he', 'she', 'it', 'we', 'they',
                        'me', 'him', 'her', 'us', 'them',
                        'my', 'your', 'his', 'its', 'our', 'their',
                        'someone', 'everyone', 'anything', 'nothing', 'whatever', 'whoever',
                        'myself', 'yourself', 'himself', 'herself', 'ourselves', 'themselves'],
            'modal': ['will', 'would', 'can', 'could', 'should', 'may', 'might', 'must'],
            'adverb': ['carefully', 'quickly', 'slowly', 'happily', 'sadly', 'quietly',
                       'loudly', 'suddenly', 'finally', 'usually', 'often', 'never',
                       'always', 'sometimes', 'already', 'still', 'just', 'almost',
                       'actually', 'especially', 'probably', 'certainly', 'simply',
                       'furthermore', 'nevertheless', 'meanwhile', 'therefore', 'otherwise',
                       'consequently', 'moreover', 'specifically', 'approximately', 'subsequently'],
            'discourse': ['topic', 'opinion', 'reason', 'example', 'fact', 'idea',
                          'problem', 'solution', 'advantage', 'disadvantage', 'conclusion',
                          'introduction', 'argument', 'evidence', 'summary', 'detail',
                          'perspective', 'criterion', 'phenomenon', 'controversy', 'implication',
                          'hypothesis', 'framework', 'principle', 'strategy', 'perspective'],
            'academic': ['chapter', 'paragraph', 'sentence', 'word', 'article',
                         'essay', 'report', 'speech', 'conversation', 'discussion', 'debate',
                         'thesis', 'hypothesis', 'research', 'survey', 'statistic',
                         'methodology', 'literature', 'citation', 'reference', 'bibliography'],
            'world': ['country', 'city', 'capital', 'population', 'language', 'culture',
                      'history', 'geography', 'economy', 'technology', 'environment',
                      'pollution', 'protection', 'development', 'education', 'tradition',
                      'government', 'law', 'freedom', 'peace', 'war',
                      'globalization', 'sustainability', 'biodiversity', 'infrastructure', 'innovation'],
            'complex_adj': ['ancient', 'modern', 'traditional', 'international', 'national',
                            'local', 'personal', 'social', 'physical', 'mental',
                            'natural', 'artificial', 'political', 'economic', 'scientific',
                            'creative', 'practical', 'theoretical', 'visible', 'invisible',
                            'comprehensive', 'controversial', 'fundamental', 'predominant', 'unprecedented'],
            'relationship': ['friend', 'classmate', 'neighbor', 'leader', 'member',
                             'partner', 'team', 'group', 'society',
                             'teacher', 'student', 'doctor', 'engineer', 'scientist',
                             'parent', 'child', 'adult', 'citizen',
                             'president', 'minister', 'professor', 'journalist', 'volunteer',
                             'colleague', 'supervisor', 'entrepreneur', 'philosopher', 'humanity'],
            'place': ['school', 'home', 'park', 'library', 'hospital',
                      'classroom', 'office', 'university', 'laboratory',
                      'museum', 'cinema', 'restaurant', 'hotel', 'stadium',
                      'community', 'countryside', 'downtown', 'suburb', 'coast',
                      'continent', 'territory', 'province', 'district', 'headquarters'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow', 'yesterday',
                     'now', 'then', 'soon', 'later', 'before', 'after',
                     'century', 'decade', 'year', 'month', 'week', 'period', 'era',
                     'deadline', 'schedule', 'timetable', 'transition', 'milestone'],
        },
        'event_roles': ['agent', 'action', 'patient', 'location', 'time', 'manner', 'reason'],
    },
}


def make_object_scene(grade_config: dict) -> list:
    """物体场景（和小学一样）"""
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


def make_event_scene(grade_config: dict) -> list:
    """事件场景：谁对谁做了什么，在哪里，什么时候

    事件仍是 Dict[str, str]，参照游戏协议不需要改动。
    """
    attrs = grade_config['attributes']
    roles = grade_config['event_roles']
    n = grade_config['num_objects']
    scene = []

    agent_pool = attrs.get('pronoun', []) + attrs.get('animal', []) + attrs.get('relationship', [])
    action_pool = attrs.get('verb', []) + attrs.get('action', [])
    patient_pool = attrs.get('pronoun', []) + attrs.get('animal', []) + attrs.get('food', []) + attrs.get('relationship', [])
    location_pool = attrs.get('place', [])
    time_pool = attrs.get('time', [])
    manner_pool = attrs.get('adverb', [])

    for _ in range(n):
        event = {}
        for role in roles:
            if role == 'agent':
                event['agent'] = random.choice(agent_pool) if agent_pool else random.choice(attrs.get('pronoun', ['i']))
            elif role == 'action':
                event['action'] = random.choice(action_pool) if action_pool else 'do'
            elif role == 'patient':
                event['patient'] = random.choice(patient_pool) if patient_pool else 'it'
            elif role == 'location':
                if random.random() < 0.7 and location_pool:
                    event['location'] = random.choice(location_pool)
            elif role == 'time':
                if random.random() < 0.5 and time_pool:
                    event['time'] = random.choice(time_pool)
            elif role == 'manner':
                if random.random() < 0.4 and manner_pool:
                    event['manner'] = random.choice(manner_pool)
            elif role == 'reason':
                if random.random() < 0.3:
                    reason_pool = attrs.get('discourse', [])
                    if reason_pool:
                        event['reason'] = random.choice(reason_pool)
        scene.append(event)
    return scene


def make_high_scene(grade_config: dict) -> list:
    """混合场景：物体 + 事件"""
    if random.random() < grade_config.get('event_ratio', 0.3):
        return make_event_scene(grade_config)
    return make_object_scene(grade_config)


def scene_to_raw_input(scene: list) -> dict:
    """把场景字典转为感知编码器的输入格式

    视觉：场景物体的属性 hash → (4, 8, 8) feature map
    听觉：属性类别 one-hot → (7,)
    位置：场景物体数量归一化 → (2,)
    """
    # 视觉 (4, 8, 8) = 256 维
    visual = torch.zeros(4, 8, 8)
    for i, obj in enumerate(scene[:4]):  # 最多 4 个物体
        for j, (key, val) in enumerate(obj.items()):
            if j >= 8:
                break
            seed = hash(f"{key}:{val}") % 10000
            g = torch.Generator().manual_seed(seed)
            visual[i, j, :] = torch.randn(8, generator=g) * 0.5 + 0.5

    # 听觉 (7,) = 7 个属性类别
    categories = ['color', 'shape', 'size', 'animal', 'food', 'verb', 'action',
                  'agent', 'patient', 'adj', 'adverb', 'place', 'modal']
    auditory = torch.zeros(13)
    for obj in scene:
        for key in obj:
            if key in categories:
                auditory[categories.index(key)] += 1.0
    if auditory.sum() > 0:
        auditory = auditory / auditory.sum()

    # 位置 (2,)
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
        scene = make_high_scene(gc)
        target = random.randint(0, len(scene) - 1)

        # ── 通道 1：感知-预测闭环（GPU 工作）──
        raw_input = scene_to_raw_input(scene)
        obs = learner.perceive(raw_input)
        action = learner.choose_action(obs)

        # 生成下一个场景作为"预测目标"
        next_scene = make_high_scene(gc)
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

    learner.consolidate()
    rate = total_successes / max(total_games, 1)
    return {'success_rate': rate, 'total_games': total_games, 'total_successes': total_successes}


def verify_save_load(learner: Learner, grade: int) -> bool:
    """保存 → 加载 → 验证一致性"""
    path = os.path.join(CHECKPOINT_DIR, f'high_{grade}.pt')
    learner.save(path)

    vocab_before = len(learner.get_vocabulary())
    grounded_before = len(learner.grounding.get_grounded_symbols())
    stage_before = learner.stage
    steps_before = learner._total_steps

    config = learner.config
    loaded = Learner(config)
    loaded.load(path)

    ok = (vocab_before == len(loaded.get_vocabulary()) and
          grounded_before == len(loaded.grounding.get_grounded_symbols()) and
          stage_before == loaded.stage and
          steps_before == loaded._total_steps and
          torch.allclose(learner.engine.W1, loaded.engine.W1))
    return ok


def main():
    print("=" * 70)
    print("通用学习系统 — 高中英语 (Grade 10 ~ Grade 12)")
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

    # 加载小学检查点
    learner = Learner(config)
    elementary_path = os.path.join(CHECKPOINT_DIR, 'middle_9_final.pt')
    high_latest = 0

    if os.path.exists(elementary_path):
        learner.load(elementary_path)
        print(f"[中学基础] 已加载中学 Grade 9 检查点")
        print(f"  已有词汇: {len(learner.get_vocabulary())}")
        print(f"  已有接地: {len(learner.grounding.get_grounded_symbols())}")
        print(f"  发展阶段: {learner.stage}")
        print()

        # 检查是否有高中检查点可恢复
        for g in [10, 11, 12]:
            path = os.path.join(CHECKPOINT_DIR, f'high_{g}.pt')
            if os.path.exists(path):
                high_latest = g

        if high_latest > 0:
            path = os.path.join(CHECKPOINT_DIR, f'high_{high_latest}.pt')
            learner.load(path)
            print(f"[恢复] 找到高中 Grade {high_latest} 检查点，继续学习")
            print(f"  已有词汇: {len(learner.get_vocabulary())}")
            print(f"  已有接地: {len(learner.grounding.get_grounded_symbols())}")
            print(f"  发展阶段: {learner.stage}")
            print()
    else:
        print("[警告] 未找到中学检查点，从零开始（建议先运行 run_english_middle.py）")
        print()

    all_results = {}
    start_time = time.time()

    for grade in range(high_latest + 1 if high_latest > 0 else 10, 13):
        gc = GRADES[grade]
        print(f"{'='*70}")
        print(f"Grade {grade}: {gc['name']}")
        print(f"  词汇范围: {', '.join(list(gc['attributes'].keys()))}")
        print(f"  事件场景比例: {gc['event_ratio']:.0%}")
        print(f"  练习轮数: {gc['rounds']}")
        print(f"{'='*70}")

        t0 = time.time()
        result = teach_grade(learner, grade)
        t1 = time.time()

        # 评估 + 阶段检查
        evaluator = CapabilityEvaluator()
        evaluation = evaluator.evaluate(learner)
        stage_before = learner.stage
        advanced = learner.try_advance(evaluation)
        stage_after = learner.stage

        # 保存 + 验证
        save_ok = verify_save_load(learner, grade)

        vocab = len(learner.get_vocabulary())

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
    print("高中英语学习完成 — 最终评估")
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
    for g in range(10, 13):
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

    # 词汇表（只显示高中新增的高频词）
    vocab = learner.get_vocabulary()
    if vocab:
        # 找出小学没学过的词（频率<=小学阶段可能的最大值）
        new_words = {w: d for w, d in vocab.items() if d.get('frequency', 0) > 0}
        sorted_new = sorted(new_words.items(), key=lambda x: x[1].get('frequency', 0), reverse=True)
        print(f"\n[全部词汇] ({len(sorted_new)} 个)")
        for word, data in sorted_new[:30]:
            freq = data.get('frequency', 0)
            rate = data.get('success_rate', 0)
            bar = '#' * max(1, int(rate * 15))
            print(f"  {word:15s} | {freq:3d}次 | {rate:.0%} | {bar}")
        if len(sorted_new) > 30:
            print(f"  ... 还有 {len(sorted_new) - 30} 个词汇")

    # 复合符号
    compounds = learner.communication.language.compounds
    if compounds:
        print(f"\n[涌现的组合表达] ({len(compounds)} 个)")
        for compound, data in list(compounds.items())[:20]:
            comps = data.get('components', [])
            print(f"  {compound:25s} <- {' + '.join(comps)}")

    # 语法模式
    rules = learner.communication.grammar.get_rules()
    if rules:
        print(f"\n[学到的语法模式] ({len(rules)} 个)")
        for entry in rules[:10]:
            if isinstance(entry, dict):
                pattern = entry.get('category_pattern', entry.get('pattern', str(entry)))
                count = entry.get('count', '?')
                conf = entry.get('confidence', 0)
                print(f"  {pattern}: {count}次 (置信度 {conf:.1%})")

    # 最终保存
    learner.save(os.path.join(CHECKPOINT_DIR, 'high_12_final.pt'))

    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': round(total_time, 1),
        'elementary_vocab': 324,  # 中学毕业基础
        'grades': all_results,
        'final_stats': {k: v for k, v in stats.items() if not isinstance(v, dict)},
        'vocabulary': {
            word: {k: v for k, v in data.items() if k != 'last_n_successes'}
            for word, data in vocab.items()
        },
    }
    with open(os.path.join(RESULTS_DIR, 'english_high_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n检查点保存在: {CHECKPOINT_DIR}/")
    print(f"结果保存在: {RESULTS_DIR}/english_high_results.json")
    print("\n下次运行会自动从最新检查点恢复，继续学习。")
    print("=" * 70)

    return learner


if __name__ == '__main__':
    learner = main()
