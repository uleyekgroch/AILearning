"""
10 阶段发展式学习管线 — 从感知预测到博士研究

模拟人类语言与认知发展的完整路径：
  阶段 1: 感知预测（0-2 岁）→ 物理世界因果直觉
  阶段 2: 符号接地（2-3 岁）→ 参照游戏 + 命名
  阶段 3: 组合表达（3-4 岁）→ 填空 + 造句（难度脚手架）
  阶段 4: 因果推理（4-5 岁）→ 激活 CausalReasoning + Counterfactual
  阶段 5: 社会理解（5-6 岁）→ 激活 TheoryOfMind + 增强对话
  阶段 6: 初中（12-15 岁）→ 语法深化 + 叙事连接词 + 隐喻萌芽
  阶段 7: 高中（15-18 岁）→ 批判性思维 + 论证 + 类比推理
  阶段 8: 大学（18-22 岁）→ 学术英语 + 语用精通 + 跨模态
  阶段 9: 硕士（22-24 岁）→ 研究写作 + 元认知 + 假设验证
  阶段 10: 博士（24-28 岁）→ 原创贡献 + 同行评审 + 教学

不修改任何模块源码。所有集成在此脚本内完成。
"""

import sys
sys.path.insert(0, '.')

import json
import time
import random
import os
import re
import torch
from datetime import datetime

from src.core.config import LearnerConfig
from src.core.learner import Learner
from src.environment.world import World
from src.language.pragmatics import PragmaticsModule, SocialContext
from src.language.narrative import NarrativeModule
from src.reasoning.causal import CausalReasoningModule
from src.reasoning.counterfactual import CounterfactualModule, CounterfactualWorld
from src.reasoning.theory_of_mind import TheoryOfMindModule
from src.reasoning.metaphor import MetaphorTracker
from src.curriculum.evaluator import CapabilityEvaluator

CHECKPOINT_DIR = 'checkpoints/developmental'
RESULTS_DIR = 'results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 词汇库 — 按难度分层
# ══════════════════════════════════════════════════════════════════

SIMPLE_VOCAB = {
    'agent': ['baby', 'child', 'cat', 'dog', 'bird', 'fish'],
    'action': ['see', 'hear', 'touch', 'eat', 'drink', 'sleep', 'walk', 'run'],
    'patient': ['ball', 'cup', 'toy', 'food', 'milk', 'book'],
    'adj': ['big', 'small', 'red', 'blue', 'happy', 'sad'],
}

MEDIUM_VOCAB = {
    'agent': ['student', 'teacher', 'friend', 'doctor', 'child', 'mother'],
    'action': ['study', 'work', 'read', 'write', 'speak', 'listen', 'think',
               'learn', 'teach', 'help', 'want', 'need', 'like', 'try'],
    'patient': ['book', 'problem', 'idea', 'plan', 'question', 'answer',
                'story', 'letter', 'email', 'project', 'exam'],
    'adj': ['important', 'difficult', 'easy', 'interesting', 'useful',
            'possible', 'different', 'similar', 'special', 'common'],
    'adverb': ['quickly', 'slowly', 'carefully', 'easily', 'often', 'usually'],
}

COMPLEX_VOCAB = {
    'agent': ['student', 'teacher', 'friend', 'doctor', 'child', 'mother',
              'father', 'scientist', 'artist', 'worker', 'manager',
              'he', 'she', 'they', 'we'],
    'action': ['study', 'work', 'read', 'write', 'speak', 'listen', 'think',
               'know', 'learn', 'teach', 'help', 'want', 'need', 'like',
               'love', 'try', 'start', 'stop', 'begin', 'finish', 'open',
               'close', 'give', 'take', 'buy', 'sell', 'make', 'bring'],
    'patient': ['book', 'problem', 'idea', 'plan', 'question', 'answer',
                'story', 'letter', 'email', 'project', 'report', 'exam',
                'class', 'lesson', 'test', 'homework', 'library', 'computer',
                'phone', 'music', 'movie', 'game', 'sport', 'food', 'money'],
    'adj': ['important', 'difficult', 'easy', 'interesting', 'useful',
            'necessary', 'possible', 'different', 'similar', 'special',
            'common', 'simple', 'complex', 'modern', 'traditional'],
    'adverb': ['quickly', 'slowly', 'carefully', 'easily', 'often', 'usually',
               'always', 'never', 'sometimes', 'already', 'still', 'also'],
    'place': ['university', 'classroom', 'office', 'hospital', 'museum',
              'theater', 'restaurant', 'airport', 'station', 'market'],
    'modal': ['can', 'should', 'must', 'will', 'would', 'could', 'may'],
}

SIMPLE_TEMPLATES = [
    "the {agent} {action} the {patient}",
    "the {adj} {patient}",
    "{agent} {action}",
]

MEDIUM_TEMPLATES = [
    "the {agent} {action} the {patient}",
    "we {action} {adverb} at the {place}",
    "the {adj} {patient} is very {adj}",
    "{agent} {modal} {action} the {patient}",
    "they {action} the {adj} {patient}",
]

COMPLEX_TEMPLATES = [
    "the {agent} {action} the {patient}",
    "we {action} {adverb} at the {place}",
    "the {adj} {patient} is very {adj}",
    "{agent} {modal} {action} the {patient}",
    "they {action} the {adj} {patient}",
    "the {agent} {modal} {action} {adverb}",
    "because the {patient} is {adj} we {action} it",
]


# ── 初中词汇 (Grade 7-9) ──────────────────────────────────────────

MIDDLE_SCHOOL_VOCAB = {
    'agent': ['student', 'teacher', 'parent', 'friend', 'scientist', 'artist',
              'athlete', 'musician', 'writer', 'engineer', 'pilot', 'chef'],
    'action': ['discover', 'create', 'improve', 'compare', 'describe', 'explain',
               'discuss', 'argue', 'suggest', 'imagine', 'predict', 'observe',
               'measure', 'calculate', 'design', 'develop', 'explore', 'analyze',
               'communicate', 'cooperate', 'compete', 'celebrate', 'complain'],
    'patient': ['experiment', 'project', 'presentation', 'competition', 'festival',
                'adventure', 'mystery', 'solution', 'discovery', 'invention',
                'technology', 'environment', 'culture', 'tradition', 'history',
                'geography', 'biology', 'chemistry', 'physics', 'mathematics'],
    'adj': ['creative', 'curious', 'confident', 'independent', 'responsible',
            'organized', 'patient', 'flexible', 'reliable', 'ambitious',
            'scientific', 'historical', 'geographical', 'biological', 'chemical',
            'physical', 'mathematical', 'technological', 'environmental', 'cultural'],
    'adverb': ['carefully', 'quickly', 'slowly', 'quietly', 'loudly',
               'easily', 'hardly', 'nearly', 'exactly', 'probably',
               'certainly', 'suddenly', 'gradually', 'frequently', 'occasionally'],
    'place': ['laboratory', 'library', 'guseum', 'stadium', 'theater',
              'studio', 'workshop', 'factory', 'farm', 'harbor'],
    'modal': ['can', 'could', 'should', 'would', 'must', 'might', 'may'],
    'connector': ['because', 'although', 'however', 'therefore', 'moreover',
                  'furthermore', 'meanwhile', 'otherwise', 'instead', 'besides'],
}

MIDDLE_SCHOOL_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "because the {agent} {action} the {patient} it was {adj}",
    "the {agent} {modal} {action} {adverb} at the {place}",
    "although the {patient} is {adj} the {agent} {action} it",
    "the {agent} {action} the {patient} and {action} the {patient}",
    "however the {adj} {patient} {modal} be {action}ed by the {agent}",
    "the {agent} {modal} {action} the {patient} because it is {adj}",
]


# ── 高中词汇 (Grade 10-12) ──────────────────────────────────────────

HIGH_SCHOOL_VOCAB = {
    'agent': ['researcher', 'philosopher', 'economist', 'politician',
              'journalist', 'psychologist', 'sociologist', 'historian',
              'critic', 'advocate', 'theorist', 'practitioner'],
    'action': ['investigate', 'evaluate', 'interpret', 'synthesize',
               'critique', 'formulate', 'hypothesize', 'validate',
               'demonstrate', 'illustrate', 'emphasize', 'distinguish',
               'contrast', 'correlate', 'generalize', 'particularize',
               'abstract', 'concretize', 'contextualize', 'deconstruct'],
    'patient': ['hypothesis', 'theory', 'paradigm', 'framework', 'methodology',
                'phenomenon', 'evidence', 'argument', 'perspective', 'assumption',
                'implication', 'consequence', 'contradiction', 'paradox',
                'complexity', 'nuance', 'context', 'discourse', 'narrative', 'rhetoric'],
    'adj': ['empirical', 'theoretical', 'philosophical', 'ideological',
            'systematic', 'analytical', 'critical', 'dialectical',
            'comprehensive', 'fundamental', 'controversial', 'ambiguous',
            'paradoxical', 'ironic', 'subtle', 'explicit', 'implicit'],
    'adverb': ['empirically', 'theoretically', 'philosophically', 'systematically',
               'critically', 'analytically', 'fundamentally', 'significantly',
               'consequently', 'nevertheless', 'furthermore', 'meanwhile'],
    'place': ['academy', 'institute', 'seminar', 'symposium', 'forum',
              'journal', 'archive', 'laboratory', 'observatory', 'seminar_room'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must', 'shall'],
    'connector': ['although', 'however', 'nevertheless', 'conversely',
                  'consequently', 'furthermore', 'moreover', 'whereas',
                  'while', 'despite', 'notwithstanding', 'accordingly'],
}

HIGH_SCHOOL_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} argues that the {patient} is {adj}",
    "although the {agent} {action} the {patient} the {patient} remains {adj}",
    "the {agent} {modal} {action} the {patient} because it is {adj}",
    "however the {adj} {patient} {modal} be {action}ed differently",
    "the {agent} {action} the {patient} to {action} the {adj} {patient}",
    "while the {patient} is {adj} the {agent} nevertheless {action} it",
    "the {agent} {action} that the {adj} {patient} {modal} be {action}ed",
]


# ── 大学词汇 (CET-4/6 level) ──────────────────────────────────────

UNIVERSITY_VOCAB = {
    'agent': ['student', 'professor', 'researcher', 'lecturer', 'dean',
              'undergraduate', 'graduate', 'scholar', 'academic', 'intellectual'],
    'action': ['lecture', 'seminar', 'tutorial', 'thesis', 'dissertation',
               'publish', 'cite', 'reference', 'plagiarize', 'paraphrase',
               'summarize', 'synthesize', 'evaluate', 'assess', 'examine',
               'investigate', 'demonstrate', 'establish', 'challenge', 'refute'],
    'patient': ['thesis', 'dissertation', 'journal', 'conference', 'peer_review',
                'abstract', 'introduction', 'methodology', 'results', 'discussion',
                'bibliography', 'appendix', 'footnote', 'citation', 'reference',
                'curriculum', 'syllabus', 'semester', 'credit', 'grade'],
    'adj': ['academic', 'scholarly', 'peer_reviewed', 'published', 'cited',
            'original', 'novel', 'innovative', 'rigorous', 'systematic',
            'comprehensive', 'preliminary', 'definitive', 'controversial',
            'significant', 'substantial', 'marginal', 'negligible'],
    'adverb': ['significantly', 'substantially', 'considerably', 'marginally',
               'preliminarily', 'definitively', 'controversially', 'originally',
               'methodologically', 'theoretically', 'empirically', 'statistically'],
    'place': ['university', 'campus', 'department', 'faculty', 'institute',
              'research_center', 'lecture_hall', 'library', 'laboratory', 'office'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must', 'shall'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'whereas', 'whilst', 'albeit', 'notwithstanding'],
}

UNIVERSITY_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} argues that the {patient} is {adj}",
    "the {adj} {patient} was {action}ed by the {agent}",
    "the {agent} {modal} {action} the {patient} because it is {adj}",
    "however the {agent} {action} the {adj} {patient} {adverb}",
    "the {agent} {action} that the {patient} {modal} be {adj}",
    "furthermore the {agent} {action} the {patient} to {action} the {adj} {patient}",
    "the {adj} {patient} {modal} be {action}ed {adverb} by the {agent}",
]


# ── 硕士词汇 (Academic research) ──────────────────────────────────

MASTER_VOCAB = {
    'agent': ['researcher', 'principal_investigator', 'supervisor', 'examiner',
              'reviewer', 'panelist', 'candidate', 'fellow', 'postdoc', 'assistant'],
    'action': ['propose', 'formulate', 'operationalize', 'calibrate', 'validate',
               'replicate', 'synthesize', 'meta_analyze', 'triangulate', 'corroborate',
               'falsify', 'refine', 'elaborate', 'qualify', 'quantify',
               'contextualize', 'problematize', 'reconceptualize', 'operationalize'],
    'patient': ['methodology', 'framework', 'paradigm', 'epistemology', 'ontology',
                'axiology', 'heuristic', 'algorithm', 'protocol', 'instrument',
                'variable', 'construct', 'indicator', 'dimension', 'factor',
                'correlation', 'causation', 'mediation', 'moderation', 'interaction'],
    'adj': ['methodological', 'epistemological', 'ontological', 'axiological',
            'heuristic', 'empirical', 'quasi_experimental', 'longitudinal',
            'cross_sectional', 'mixed_methods', 'qualitative', 'quantitative',
            'grounded', 'phenomenological', 'ethnographic', 'case_study'],
    'adverb': ['methodologically', 'epistemologically', 'ontologically',
               'heuristically', 'empirically', 'statistically', 'significantly',
               'robustly', 'consistently', 'reproducibly', 'replicably'],
    'place': ['research_group', 'lab', 'field_site', 'conference', 'workshop',
              'seminar', 'colloquium', 'defense', 'viva', 'examination'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'whereas', 'whilst', 'albeit', 'notwithstanding',
                  'more_specifically', 'in_particular', 'for_instance'],
}

MASTER_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} proposes that the {patient} is {adj}",
    "the {adj} {patient} was {action}ed using {adj} methods",
    "the {agent} {modal} {action} the {patient} to {action} the {adj} {patient}",
    "however the {agent} {action} that the {patient} {modal} be {adj}",
    "the {agent} {action} the {patient} by {action}ing the {adj} {patient}",
    "furthermore the {adj} {patient} {modal} be {action}ed {adverb}",
    "the {agent} argues that the {adj} {patient} {modal} be {action}ed differently",
]


# ── 博士词汇 (Original contribution) ──────────────────────────────

PHD_VOCAB = {
    'agent': ['candidate', 'supervisor', 'examiner', 'reviewer', 'panel',
              'committee', 'scholar', 'authority', 'pioneer', 'founder'],
    'action': ['contribute', 'advance', 'pioneer', 'establish', 'challenge',
               'overturn', 'revolutionize', 'transform', 'redefine', 'reconceptualize',
               'synthesize', 'integrate', 'bridge', 'unify', 'reconcile',
               'critique', 'deconstruct', 'reconstruct', 'reimagine', 'envision'],
    'patient': ['contribution', 'advancement', 'breakthrough', 'paradigm_shift',
                'theoretical_framework', 'conceptual_model', 'analytical_tool',
                'research_agenda', 'scholarly_discourse', 'intellectual_tradition',
                'epistemological_assumption', 'ontological_commitment',
                'methodological_innovation', 'empirical_finding', 'theoretical_insight'],
    'adj': ['groundbreaking', 'seminal', 'influential', 'controversial',
            'paradigmatic', 'foundational', 'cutting_edge', 'state_of_the_art',
            'interdisciplinary', 'transdisciplinary', 'multidisciplinary',
            'original', 'novel', 'unprecedented', 'revolutionary', 'transformative'],
    'adverb': ['fundamentally', 'paradigmatically', 'revolutionarily',
               'transformatively', 'originally', 'novelly', 'unprecedentedly',
               'groundbreakingly', 'seminal', 'influentially'],
    'place': ['defense_chamber', 'review_panel', 'editorial_board', 'peer_review',
              'conference_podium', 'keynote_stage', 'plenary_session', 'roundtable',
              'symposium', 'colloquium'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'whereas', 'whilst', 'albeit', 'notwithstanding',
                  'more_specifically', 'in_particular', 'for_instance',
                  'to_put_it_differently', 'in_other_words', 'that_is'],
}

PHD_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} argues that the {adj} {patient} {modal} be {action}ed",
    "the {adj} {patient} represents a {adj} contribution to {patient}",
    "the {agent} {action} the {patient} by {action}ing the {adj} {patient}",
    "however the {agent} contends that the {patient} is fundamentally {adj}",
    "the {agent} {action} that the {adj} {patient} {modal} {action} the field",
    "furthermore the {agent} {action} the {adj} {patient} as a {adj} {patient}",
    "the {agent} {action} the {patient} to {action} the {adj} {patient} {adverb}",
]


# ══════════════════════════════════════════════════════════════════
# 辅助函数 — 练习生成
# ══════════════════════════════════════════════════════════════════

def generate_sentence_from_vocab(vocab: dict, templates: list) -> tuple:
    """从词汇库和模板生成句子"""
    template = random.choice(templates)
    placeholders = re.findall(r'\{(\w+)\}', template)
    fills = {}
    for ph in placeholders:
        ph_lower = ph.lower()
        if ph_lower in vocab:
            fills[ph] = random.choice(vocab[ph_lower])
        else:
            fills[ph] = random.choice(vocab.get('action', ['do']))

    sentence = template
    for ph, val in fills.items():
        sentence = sentence.replace('{' + ph + '}', val)

    content_words = [v for v in fills.values() if len(v) > 2]
    if not content_words:
        content_words = list(fills.values())
    target_word = random.choice(content_words)
    target_cat = None
    for ph, val in fills.items():
        if val == target_word:
            target_cat = ph.lower()
            break

    return sentence, target_word, target_cat, list(fills.values())


def generate_cloze_from_vocab(vocab: dict, templates: list) -> tuple:
    """生成填空题"""
    sentence, target_word, target_cat, _ = generate_sentence_from_vocab(vocab, templates)

    distractors = []
    if target_cat and target_cat in vocab:
        pool = [w for w in vocab[target_cat] if w != target_word]
        distractors = random.sample(pool, min(3, len(pool)))
    if len(distractors) < 3:
        all_words = [w for ws in vocab.values() for w in ws if w != target_word]
        while len(distractors) < 3:
            d = random.choice(all_words)
            if d not in distractors:
                distractors.append(d)

    options = [target_word] + distractors[:3]
    random.shuffle(options)
    blanked = sentence.replace(target_word, '___', 1)
    return blanked, target_word, options, sentence


def generate_sentence_exercise_from_vocab(vocab: dict, templates: list) -> tuple:
    """生成造句题"""
    _, _, _, words = generate_sentence_from_vocab(vocab, templates)
    if not words:
        words = ['the', 'student', 'study', 'hard']

    target_words = [w for w in words if w]
    all_words = [w for ws in vocab.values() for w in ws]
    distractor_count = random.randint(1, 2)
    distractors = random.sample(all_words, min(distractor_count, len(all_words)))

    available_words = target_words + distractors
    random.shuffle(available_words)
    return target_words, available_words


# ══════════════════════════════════════════════════════════════════
# 辅助函数 — 事件检测（从 run_english_advanced.py 复用）
# ══════════════════════════════════════════════════════════════════

def detect_events(world, scene_before: list, scene_after: list) -> list:
    """从物理状态变化检测事件"""
    events = []

    collisions = world.physics.detect_collisions()
    for ia, ib in collisions:
        subj = scene_before[ia] if ia < len(scene_before) else {'color': 'unknown'}
        obj = scene_before[ib] if ib < len(scene_before) else {'color': 'unknown'}
        events.append({
            'subject': dict(subj),
            'action': 'collide',
            'object': dict(obj),
            'caused_by': len(events) - 1 if events else None,
        })

    for i, (b, a) in enumerate(zip(scene_before, scene_after)):
        if b != a and {'subject': dict(b), 'action': 'move', 'object': dict(a)} not in events:
            events.append({
                'subject': dict(b),
                'action': 'move',
                'object': dict(a),
            })

    if not events and scene_before:
        events.append({
            'subject': dict(scene_before[0]),
            'action': 'observe',
            'object': dict(scene_before[min(1, len(scene_before) - 1)]),
        })

    return events


def derive_social_context(world, target_idx: int, context: SocialContext) -> dict:
    """从物理世界状态推导社会情境参数"""
    if not world.physics.objects or target_idx >= len(world.physics.objects):
        return {'has_conflict': False}

    target_obj = world.physics.objects[target_idx]
    agent_pos = world.agent_pos
    ctx = dict(target_obj.get_properties())
    n_objs = len(world.physics.objects)

    if context == SocialContext.DEBATE:
        same_color = [o for o in world.physics.objects
                      if o.color == target_obj.color and o.obj_id != target_obj.obj_id]
        ctx['has_conflict'] = len(same_color) > 0
    elif context == SocialContext.POLITENESS:
        dist = (target_obj.position[:2] - agent_pos).norm().item()
        ctx['social_distance'] = min(dist / 5.0, 1.0)
        ctx['is_stranger'] = dist > 3.0
    elif context == SocialContext.NEGOTIATION:
        ctx['goal_conflict'] = n_objs > 5
        ctx['resource_scarcity'] = len([o for o in world.physics.objects
                                         if o.color == target_obj.color]) <= 1
    elif context == SocialContext.OWNERSHIP:
        dist = (target_obj.position[:2] - agent_pos).norm().item()
        ctx['contested_resource'] = dist < 2.0
    elif context == SocialContext.SPATIAL:
        ctx['spatial_ambiguous'] = n_objs > 6
        ctx['position'] = target_obj.position[:2].tolist()
    elif context == SocialContext.QUANTITATIVE:
        ctx['needs_counting'] = n_objs > 4
        ctx['count_total'] = n_objs
    elif context == SocialContext.EMPATHY:
        ctx['different_perspective'] = n_objs > 5
        ctx['understanding_score'] = 0.5
    elif context == SocialContext.REPAIR:
        ctx['understanding_score'] = 0.3

    return ctx


def describe_event_subject(ev: dict) -> str:
    """从事件中提取主语描述"""
    subj = ev.get('subject', {})
    if isinstance(subj, dict):
        parts = [str(v) for v in subj.values() if v]
        return '_'.join(parts[:2]) if parts else 'thing'
    return str(subj)


def describe_event_action(ev: dict) -> str:
    """从事件中提取动作描述"""
    action = ev.get('action', 'unknown')
    obj = ev.get('object', {})
    if isinstance(obj, dict):
        parts = [str(v) for v in obj.values() if v]
        target = '_'.join(parts[:2]) if parts else 'thing'
    else:
        target = str(obj)
    return f"{action}_{target}"


# ══════════════════════════════════════════════════════════════════
# 阶段 1: 感知预测（0-2 岁）
# ══════════════════════════════════════════════════════════════════

def stage_perception_prediction(learner, world, config, rounds=300,
                                 verbose=True):
    """0-2 岁：纯物理感知预测

    在 World 环境中观察物体运动，预测下一帧状态。
    不涉及任何语言，只训练 PredictiveCodingEngine。
    """
    world.configure_for_stage('sensorimotor')
    errors = []

    for step in range(rounds):
        # 观察当前状态
        raw_input = world.observe()
        obs = learner.perceive(raw_input)

        # 选择动作
        action = learner.choose_action(obs)

        # 物理步进
        world.physics.step(dt=0.1)

        # 观察下一状态
        raw_next = world.observe()
        next_obs = learner.perceive(raw_next)

        # 学习
        error = learner.learn_from_experience(obs, action, next_obs)
        errors.append(error)

        if verbose and (step + 1) % 50 == 0:
            recent = errors[-50:]
            avg = sum(recent) / len(recent)
            print(f"  感知预测 step {step+1}/{rounds} | "
                  f"avg_error={avg:.4f} | "
                  f"prediction_accuracy={max(0, 1-avg):.3f}")

    # 最终评估
    recent = errors[-50:] if len(errors) >= 50 else errors
    avg_err = sum(recent) / len(recent)
    accuracy = max(0, 1 - avg_err)
    if verbose:
        print(f"  → 感知预测完成 | prediction_accuracy={accuracy:.3f}")
    return accuracy


# ══════════════════════════════════════════════════════════════════
# 阶段 2: 符号接地（2-3 岁）
# ══════════════════════════════════════════════════════════════════

def stage_symbol_grounding(learner, world, config, rounds=200,
                            verbose=True):
    """2-3 岁：参照游戏 + 符号接地

    从 World 物理场景中学习命名。参照游戏只在此阶段使用。
    """
    world.configure_for_stage('early_preoperational')
    successes = 0

    for step in range(rounds):
        # 物理步进产生变化
        world.physics.step(dt=0.1)

        # 感知
        raw_input = world.observe()
        obs = learner.perceive(raw_input)

        # 从物理场景生成参照场景
        scene = world.generate_scene_features()
        if not scene:
            continue

        target = random.randint(0, len(scene) - 1)

        # 符号接地
        target_obj = scene[target]
        symbols_contexts = [(val, attr) for attr, val in target_obj.items() if val]
        if symbols_contexts:
            try:
                learner.grounding.ground_symbols_batch(symbols_contexts, obs)
            except Exception:
                pass

        # 参照游戏
        success = learner.play_reference_game(scene, target)
        if success:
            successes += 1

        if verbose and (step + 1) % 50 == 0:
            rate = successes / (step + 1)
            vocab = len(learner.get_vocabulary())
            print(f"  符号接地 step {step+1}/{rounds} | "
                  f"success_rate={rate:.3f} | vocab={vocab}")

    # 最终评估
    rate = successes / max(rounds, 1)
    vocab = len(learner.get_vocabulary())
    if verbose:
        print(f"  → 符号接地完成 | success_rate={rate:.3f} | vocab={vocab}")
    return rate, vocab


# ══════════════════════════════════════════════════════════════════
# 阶段 3: 组合表达（3-4 岁）
# ══════════════════════════════════════════════════════════════════

def stage_compositional_expression(learner, world, config, rounds=300,
                                    verbose=True):
    """3-4 岁：填空 + 造句，难度脚手架

    难度梯度：简单(2 词) → 中等(4 词) → 复杂(6 词+)
    """
    # 难度级别
    difficulty_levels = [
        {'name': '简单', 'vocab': SIMPLE_VOCAB, 'templates': SIMPLE_TEMPLATES,
         'rounds': rounds // 3},
        {'name': '中等', 'vocab': MEDIUM_VOCAB, 'templates': MEDIUM_TEMPLATES,
         'rounds': rounds // 3},
        {'name': '复杂', 'vocab': COMPLEX_VOCAB, 'templates': COMPLEX_TEMPLATES,
         'rounds': rounds - 2 * (rounds // 3)},
    ]

    cloze_successes = 0
    sentence_successes = 0
    total_cloze = 0
    total_sentence = 0

    for level in difficulty_levels:
        if verbose:
            print(f"  --- 难度: {level['name']} ({level['rounds']} rounds) ---")

        for step in range(level['rounds']):
            if random.random() < 0.5:
                # 填空题
                blanked, target, options, _ = generate_cloze_from_vocab(
                    level['vocab'], level['templates'])
                success = learner.communication.play_cloze_game(
                    blanked, target, options)
                learner.reading_history.append(1.0 if success else 0.0)
                total_cloze += 1
                if success:
                    cloze_successes += 1
            else:
                # 造句题
                target_words, available = generate_sentence_exercise_from_vocab(
                    level['vocab'], level['templates'])
                success = learner.communication.play_sentence_game(
                    target_words, available)
                learner.writing_history.append(1.0 if success else 0.0)
                total_sentence += 1
                if success:
                    sentence_successes += 1

            if verbose and (step + 1) % 50 == 0:
                c_rate = cloze_successes / max(total_cloze, 1)
                s_rate = sentence_successes / max(total_sentence, 1)
                print(f"    step {step+1}/{level['rounds']} | "
                      f"cloze={c_rate:.3f} | sentence={s_rate:.3f}")

    # 最终评估
    c_rate = cloze_successes / max(total_cloze, 1)
    s_rate = sentence_successes / max(total_sentence, 1)
    if verbose:
        print(f"  → 组合表达完成 | cloze={c_rate:.3f} | sentence={s_rate:.3f}")
    return c_rate, s_rate


# ══════════════════════════════════════════════════════════════════
# 阶段 4: 因果推理（4-5 岁）
# ══════════════════════════════════════════════════════════════════

def stage_causal_reasoning(learner, world, config, rounds=200,
                            verbose=True):
    """4-5 岁：激活因果推理 + 反事实

    从物理碰撞事件中学习因果关系，表达 "because"。
    """
    # 激活沉睡模块
    causal = learner.causal
    cf = learner.counterfactual

    if causal is None:
        print("  [警告] CausalReasoningModule 未注册")
        return 0
    if cf is None:
        print("  [警告] CounterfactualModule 未注册")
        return 0

    world.configure_for_stage('late_concrete')
    causal_expressions = 0

    for step in range(rounds):
        # 物理状态变化
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        scene_after = world.generate_scene_features()

        # 检测事件
        events = detect_events(world, scene_before, scene_after)

        # 因果观察
        for ev in events:
            cause = describe_event_subject(ev)
            effect = describe_event_action(ev)
            causal.observe(cause, effect)

            # 检查是否产生因果表达
            expr = causal.express_causal(cause, effect)
            if expr:
                causal_expressions += 1

        # 反事实推理
        if events:
            actual_ev = events[0]
            actual_action = describe_event_action(actual_ev)
            actual_effect = describe_event_subject(actual_ev)

            # 构造反事实世界
            cf_world = CounterfactualWorld()
            for alt in ['push', 'pull', 'drop', 'throw', 'roll']:
                cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
            cf_world.add_action_effect(actual_action, actual_effect, 0.8)

            cf_effect = cf.reason(actual_action, actual_effect, cf_world)
            if cf_effect:
                cf._cf_count += 1

        if verbose and (step + 1) % 50 == 0:
            n_rules = len(causal.get_confident_rules(0.6))
            print(f"  因果推理 step {step+1}/{rounds} | "
                  f"causal_rules={n_rules} | "
                  f"cf_count={cf._cf_count}")

    # 最终评估
    n_rules = len(causal.get_confident_rules(0.6))
    strongest = causal.get_strongest_rules(3)
    if verbose:
        print(f"  → 因果推理完成 | rules={n_rules} | "
              f"cf_count={cf._cf_count}")
        for r in strongest:
            print(f"    {r.expression()} (conf={r.confidence:.3f})")
    return n_rules


# ══════════════════════════════════════════════════════════════════
# 阶段 5: 社会理解（5-6 岁）
# ══════════════════════════════════════════════════════════════════

def stage_social_understanding(learner, world, config, pragmatics,
                                rounds=200, verbose=True):
    """5-6 岁：激活心智理论 + 增强对话

    信息不对称对话 → 视角建模 → 语用修饰。
    """
    # 激活心智理论模块
    tom = learner.theory_of_mind
    if tom is None:
        print("  [警告] TheoryOfMindModule 未注册")
        return 0

    world.configure_for_stage('late_concrete')
    perspective_markers = 0
    dialogue_successes = 0

    for step in range(rounds):
        world.physics.step(dt=0.1)
        scene = world.generate_scene_features()
        if not scene:
            continue

        target = random.randint(0, len(scene) - 1)
        target_features = scene[target]

        # 信息不对称：listener 只看到部分物体
        visible_count = max(1, len(scene) // 2)
        visible_to_listener = random.sample(scene, visible_count)

        # 为 listener 建立认知模型
        listener_facts = set()
        for obj in visible_to_listener:
            for val in obj.values():
                if val:
                    listener_facts.add(str(val))
        tom.model_other('listener', listener_facts)

        # 自己知道的事实
        for val in target_features.values():
            if val:
                tom.own_perspective.observe(str(val))

        # 检测知识不对称
        for val in target_features.values():
            if val and tom.detect_asymmetry('listener', str(val)):
                # 不对称 → 需要调整描述
                pass

        # 根据 listener 视角调整描述
        adjusted = tom.adjust_description(target_features, scene, 'listener')

        # 选择视角标记
        certainty = 0.7  # 默认确定性
        marker = tom.choose_perspective_marker(
            certainty, language_experience=step)
        if marker:
            perspective_markers += 1

        # 对话语用
        context = random.choice(list(SocialContext))
        utterance = adjusted if adjusted else list(target_features.values())[:3]
        try:
            ctx = derive_social_context(world, target, context)
            result = pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        # 参照游戏测试
        raw_input = world.observe()
        obs = learner.perceive(raw_input)
        success = learner.play_reference_game(scene, target)
        if success:
            dialogue_successes += 1

        if verbose and (step + 1) % 50 == 0:
            rate = dialogue_successes / (step + 1)
            print(f"  社会理解 step {step+1}/{rounds} | "
                  f"dialogue={rate:.3f} | "
                  f"perspective_markers={perspective_markers}")

    # 最终评估
    rate = dialogue_successes / max(rounds, 1)
    if verbose:
        print(f"  → 社会理解完成 | dialogue={rate:.3f} | "
              f"perspective_markers={perspective_markers}")
    return perspective_markers


# ══════════════════════════════════════════════════════════════════
# 阶段 6: 初中（12-15 岁）— 语法深化 + 叙事连接词 + 隐喻萌芽
# ══════════════════════════════════════════════════════════════════

def stage_middle_school(learner, world, config, pragmatics,
                        rounds=400, verbose=True):
    """初中阶段：语法复杂化 + 叙事连接词 + 隐喻初现

    核心转变：从简单句到复合句，从单事件到多事件叙事，
    开始理解隐喻性语言。
    """
    narrative = learner.narrative
    metaphor = learner.metaphor_tracker
    world.configure_for_stage('late_concrete')

    cloze_successes = 0
    sentence_successes = 0
    narrative_successes = 0
    total_games = 0
    emergent_connectors = []

    for step in range(rounds):
        game_type = random.random()

        if game_type < 0.35:
            # 填空题 — 初中词汇
            blanked, target, options, _ = generate_cloze_from_vocab(
                MIDDLE_SCHOOL_VOCAB, MIDDLE_SCHOOL_TEMPLATES)
            success = learner.communication.play_cloze_game(
                blanked, target, options)
            learner.reading_history.append(1.0 if success else 0.0)
            if success:
                cloze_successes += 1
            total_games += 1

        elif game_type < 0.65:
            # 造句题 — 复合句
            target_words, available = generate_sentence_exercise_from_vocab(
                MIDDLE_SCHOOL_VOCAB, MIDDLE_SCHOOL_TEMPLATES)
            success = learner.communication.play_sentence_game(
                target_words, available)
            learner.writing_history.append(1.0 if success else 0.0)
            if success:
                sentence_successes += 1
            total_games += 1

        else:
            # 叙事理解 — 多事件 + 连接词
            world.physics.step(dt=0.15)
            scene_before = world.generate_scene_features()
            obs = learner.perceive(world.observe())
            action = learner.choose_action(obs)
            world.physics.step(dt=0.2)
            scene_after = world.generate_scene_features()

            events = detect_events(world, scene_before, scene_after)
            narr = narrative.build_narrative(events)
            symbols = narr.to_symbols()

            if symbols:
                # 参照游戏测试叙事理解
                scene = scene_before if scene_before else [{'color': 'red'}]
                target = random.randint(0, len(scene) - 1)
                success = learner.play_reference_game(scene, target)
                if success:
                    narrative_successes += 1

                # 记录隐喻
                if metaphor:
                    metaphor.record(symbols, {'features': scene})

            total_games += 1

        if verbose and (step + 1) % 100 == 0:
            c_rate = cloze_successes / max(total_games * 0.35, 1)
            s_rate = sentence_successes / max(total_games * 0.3, 1)
            n_rate = narrative_successes / max(total_games * 0.35, 1)
            connectors = narrative.get_emergent_connectors()
            print(f"  初中 step {step+1}/{rounds} | "
                  f"cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
                  f"narrative={n_rate:.3f} | connectors={len(connectors)}")

    # 最终评估
    c_rate = cloze_successes / max(rounds * 0.35, 1)
    s_rate = sentence_successes / max(rounds * 0.3, 1)
    n_rate = narrative_successes / max(rounds * 0.35, 1)
    connectors = narrative.get_emergent_connectors()

    # 检测隐喻涌现
    metaphors_found = {}
    if metaphor:
        metaphors_found = metaphor.detect_metaphors()

    if verbose:
        print(f"  → 初中完成 | cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
              f"narrative={n_rate:.3f}")
        print(f"    连接词涌现: {connectors}")
        print(f"    隐喻涌现: {list(metaphors_found.keys())}")
    return c_rate, s_rate, len(connectors), len(metaphors_found)


# ══════════════════════════════════════════════════════════════════
# 阶段 7: 高中（15-18 岁）— 批判性思维 + 论证 + 类比推理
# ══════════════════════════════════════════════════════════════════

def stage_high_school(learner, world, config, pragmatics,
                      rounds=400, verbose=True):
    """高中阶段：批判性阅读 + 论证写作 + 类比推理

    核心转变：从理解到批判，从描述到论证，
    开始使用类比和隐喻进行抽象思维。
    """
    metaphor = learner.metaphor_tracker
    world.configure_for_stage('early_formal')

    cloze_successes = 0
    sentence_successes = 0
    debate_successes = 0
    total_games = 0

    for step in range(rounds):
        game_type = random.random()

        if game_type < 0.30:
            # 填空题 — 高中词汇
            blanked, target, options, _ = generate_cloze_from_vocab(
                HIGH_SCHOOL_VOCAB, HIGH_SCHOOL_TEMPLATES)
            success = learner.communication.play_cloze_game(
                blanked, target, options)
            learner.reading_history.append(1.0 if success else 0.0)
            if success:
                cloze_successes += 1
            total_games += 1

        elif game_type < 0.55:
            # 造句题 — 论证性句子
            target_words, available = generate_sentence_exercise_from_vocab(
                HIGH_SCHOOL_VOCAB, HIGH_SCHOOL_TEMPLATES)
            success = learner.communication.play_sentence_game(
                target_words, available)
            learner.writing_history.append(1.0 if success else 0.0)
            if success:
                sentence_successes += 1
            total_games += 1

        else:
            # 辩论/论证对话 — DEBATE + NEGOTIATION
            world.physics.step(dt=0.1)
            scene = world.generate_scene_features()
            if not scene:
                total_games += 1
                continue

            target = random.randint(0, len(scene) - 1)
            context = random.choice([SocialContext.DEBATE, SocialContext.NEGOTIATION,
                                      SocialContext.MORAL, SocialContext.EMPATHY])

            # 感知
            raw_input = world.observe()
            obs = learner.perceive(raw_input)

            # 内在言语规划
            try:
                planned = learner.inner_speech.plan_description(scene, target)
                used_planning = len(planned) > 0
            except Exception:
                planned = []
                used_planning = False

            # 语用修饰
            target_features = scene[target] if isinstance(scene[target], dict) else {}
            utterance = planned if planned else list(target_features.values())[:3]
            try:
                ctx = derive_social_context(world, target, context)
                result = pragmatics.apply_context(context, utterance, ctx)
            except Exception:
                pass

            # 参照游戏
            success = learner.play_reference_game(scene, target)
            if success:
                debate_successes += 1

            # 记录隐喻
            if metaphor:
                symbols = list(target_features.values())
                metaphor.record(symbols, {'features': scene})

            # 反事实推理练习
            cf = learner.counterfactual
            if cf:
                cf_world = CounterfactualWorld()
                for alt in ['argue', 'agree', 'compromise', 'ignore']:
                    cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
                actual_action = random.choice(['argue', 'agree'])
                actual_effect = f"result_{actual_action}"
                cf.reason(actual_action, actual_effect, cf_world)

            total_games += 1

        if verbose and (step + 1) % 100 == 0:
            c_rate = cloze_successes / max(total_games * 0.3, 1)
            s_rate = sentence_successes / max(total_games * 0.25, 1)
            d_rate = debate_successes / max(total_games * 0.45, 1)
            print(f"  高中 step {step+1}/{rounds} | "
                  f"cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
                  f"debate={d_rate:.3f}")

    # 最终评估
    c_rate = cloze_successes / max(rounds * 0.3, 1)
    s_rate = sentence_successes / max(rounds * 0.25, 1)
    d_rate = debate_successes / max(rounds * 0.45, 1)

    metaphors_found = {}
    if metaphor:
        metaphors_found = metaphor.detect_metaphors()

    if verbose:
        print(f"  → 高中完成 | cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
              f"debate={d_rate:.3f}")
        print(f"    隐喻涌现: {list(metaphors_found.keys())}")
    return c_rate, s_rate, d_rate, len(metaphors_found)


# ══════════════════════════════════════════════════════════════════
# 阶段 8: 大学（18-22 岁）— 学术英语 + 语用精通 + 跨模态
# ══════════════════════════════════════════════════════════════════

def stage_university(learner, world, config, pragmatics,
                     rounds=500, verbose=True):
    """大学阶段：学术写作 + 专业沟通 + 跨模态理解

    核心转变：从日常语言到学术语言，掌握 13 种语用情境，
    内在言语规划成为习惯。
    """
    world.configure_for_stage('late_formal')

    cloze_successes = 0
    sentence_successes = 0
    dialogue_successes = 0
    listening_successes = 0
    total_games = 0

    # 大学级别词汇配置
    uni_config = {
        'attributes': UNIVERSITY_VOCAB,
        'templates': UNIVERSITY_TEMPLATES,
        'num_objects': 6,
    }

    for step in range(rounds):
        game_type = random.random()

        if game_type < 0.25:
            # 填空题 — 学术词汇
            blanked, target, options, _ = generate_cloze_from_vocab(
                UNIVERSITY_VOCAB, UNIVERSITY_TEMPLATES)
            success = learner.communication.play_cloze_game(
                blanked, target, options)
            learner.reading_history.append(1.0 if success else 0.0)
            if success:
                cloze_successes += 1
            total_games += 1

        elif game_type < 0.45:
            # 造句题 — 学术写作
            target_words, available = generate_sentence_exercise_from_vocab(
                UNIVERSITY_VOCAB, UNIVERSITY_TEMPLATES)
            success = learner.communication.play_sentence_game(
                target_words, available)
            learner.writing_history.append(1.0 if success else 0.0)
            if success:
                sentence_successes += 1
            total_games += 1

        elif game_type < 0.75:
            # 对话语用 — 全部 13 种情境
            world.physics.step(dt=0.1)
            scene = world.generate_scene_features()
            if not scene:
                total_games += 1
                continue

            target = random.randint(0, len(scene) - 1)
            context = random.choice(list(SocialContext))

            raw_input = world.observe()
            obs = learner.perceive(raw_input)

            # 内在言语规划
            try:
                planned = learner.inner_speech.plan_description(scene, target)
            except Exception:
                planned = []

            target_features = scene[target] if isinstance(scene[target], dict) else {}
            utterance = planned if planned else list(target_features.values())[:3]
            try:
                ctx = derive_social_context(world, target, context)
                result = pragmatics.apply_context(context, utterance, ctx)
            except Exception:
                pass

            success = learner.play_reference_game(scene, target)
            if success:
                dialogue_successes += 1
            total_games += 1

        else:
            # 叙事 + 听力
            world.physics.step(dt=0.1)
            scene_before = world.generate_scene_features()
            obs = learner.perceive(world.observe())
            action = learner.choose_action(obs)
            world.physics.step(dt=0.2)
            scene_after = world.generate_scene_features()

            events = detect_events(world, scene_before, scene_after)
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()

            if symbols:
                narrative_text = ' '.join(symbols)
                options = [narrative_text]
                for _ in range(2):
                    # 生成不同场景的干扰叙事
                    world.physics.step(dt=0.05)
                    d_before = world.generate_scene_features()
                    world.physics.step(dt=0.1)
                    d_after = world.generate_scene_features()
                    d_events = detect_events(world, d_before, d_after)
                    d_narr = learner.narrative.build_narrative(d_events)
                    d_text = ' '.join(d_narr.to_symbols())
                    if d_text and d_text != narrative_text:
                        options.append(d_text)

                if len(options) >= 2:
                    random.shuffle(options)
                    success = learner.communication.play_listening_game(
                        narrative_text, options)
                    learner.listening_history.append(1.0 if success else 0.0)
                    if success:
                        listening_successes += 1
            total_games += 1

        if verbose and (step + 1) % 100 == 0:
            c_rate = cloze_successes / max(total_games * 0.25, 1)
            s_rate = sentence_successes / max(total_games * 0.2, 1)
            d_rate = dialogue_successes / max(total_games * 0.3, 1)
            l_rate = listening_successes / max(total_games * 0.25, 1)
            print(f"  大学 step {step+1}/{rounds} | "
                  f"cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
                  f"dialogue={d_rate:.3f} | listening={l_rate:.3f}")

    # 最终评估
    c_rate = cloze_successes / max(rounds * 0.25, 1)
    s_rate = sentence_successes / max(rounds * 0.2, 1)
    d_rate = dialogue_successes / max(rounds * 0.3, 1)
    l_rate = listening_successes / max(rounds * 0.25, 1)

    # 语用标记涌现
    emergent_markers = pragmatics.get_emergent_markers()

    if verbose:
        print(f"  → 大学完成 | cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
              f"dialogue={d_rate:.3f} | listening={l_rate:.3f}")
        print(f"    语用标记涌现: {list(emergent_markers.keys())[:5]}...")
    return c_rate, s_rate, d_rate, l_rate, len(emergent_markers)


# ══════════════════════════════════════════════════════════════════
# 阶段 9: 硕士（22-24 岁）— 研究写作 + 元认知 + 假设验证
# ══════════════════════════════════════════════════════════════════

def stage_masters(learner, world, config, pragmatics,
                  rounds=400, verbose=True):
    """硕士阶段：研究方法论 + 元认知自评估 + 假设-验证循环

    核心转变：从学习知识到生产知识，从被动接受到主动质疑，
    元认知能力使学习者能自我监控和调整学习策略。
    """
    metaphor = learner.metaphor_tracker
    world.configure_for_stage('late_formal')

    cloze_successes = 0
    sentence_successes = 0
    research_successes = 0
    total_games = 0
    strategies_used = {}

    for step in range(rounds):
        game_type = random.random()

        if game_type < 0.25:
            # 填空题 — 硕士级学术词汇
            blanked, target, options, _ = generate_cloze_from_vocab(
                MASTER_VOCAB, MASTER_TEMPLATES)
            success = learner.communication.play_cloze_game(
                blanked, target, options)
            learner.reading_history.append(1.0 if success else 0.0)
            if success:
                cloze_successes += 1
            total_games += 1

        elif game_type < 0.45:
            # 造句题 — 研究方法论表达
            target_words, available = generate_sentence_exercise_from_vocab(
                MASTER_VOCAB, MASTER_TEMPLATES)
            success = learner.communication.play_sentence_game(
                target_words, available)
            learner.writing_history.append(1.0 if success else 0.0)
            if success:
                sentence_successes += 1
            total_games += 1

        else:
            # 研究叙事 — 假设 → 实验 → 结论
            world.physics.step(dt=0.15)
            scene_before = world.generate_scene_features()
            obs = learner.perceive(world.observe())
            action = learner.choose_action(obs)
            world.physics.step(dt=0.2)
            scene_after = world.generate_scene_features()

            # 构建研究叙事
            events = detect_events(world, scene_before, scene_after)
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()

            # 元认知自评估
            meta = learner.metacognition
            if meta:
                try:
                    eval_result = meta.self_evaluate()
                    plan = meta.plan_next_learning()
                    strategy_name = plan.get('strategy', {}).get('name', 'explore')
                    strategies_used[strategy_name] = strategies_used.get(strategy_name, 0) + 1
                except Exception:
                    pass

            # 因果推理
            causal = learner.causal
            if causal and events:
                for ev in events:
                    cause = describe_event_subject(ev)
                    effect = describe_event_action(ev)
                    causal.observe(cause, effect)

            # 参照游戏测试
            if scene_before:
                target = random.randint(0, len(scene_before) - 1)
                success = learner.play_reference_game(scene_before, target)
                if success:
                    research_successes += 1

            # 隐喻记录
            if metaphor and symbols:
                metaphor.record(symbols, {'features': scene_before})

            total_games += 1

        if verbose and (step + 1) % 100 == 0:
            c_rate = cloze_successes / max(total_games * 0.25, 1)
            s_rate = sentence_successes / max(total_games * 0.2, 1)
            r_rate = research_successes / max(total_games * 0.55, 1)
            print(f"  硕士 step {step+1}/{rounds} | "
                  f"cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
                  f"research={r_rate:.3f} | strategies={strategies_used}")

    # 最终评估
    c_rate = cloze_successes / max(rounds * 0.25, 1)
    s_rate = sentence_successes / max(rounds * 0.2, 1)
    r_rate = research_successes / max(rounds * 0.55, 1)

    # 隐喻涌现
    metaphors_found = {}
    if metaphor:
        metaphors_found = metaphor.detect_metaphors()

    if verbose:
        print(f"  → 硕士完成 | cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
              f"research={r_rate:.3f}")
        print(f"    隐喻: {list(metaphors_found.keys())}")
        print(f"    策略分布: {strategies_used}")
    return c_rate, s_rate, r_rate, len(metaphors_found), strategies_used


# ══════════════════════════════════════════════════════════════════
# 阶段 10: 博士（24-28 岁）— 原创贡献 + 同行评审 + 教学
# ══════════════════════════════════════════════════════════════════

def stage_phd(learner, world, config, pragmatics,
              rounds=400, verbose=True):
    """博士阶段：原创研究 + 同行评审 + 教学相长

    核心转变：从学习者到创造者，从消费者到生产者，
    能够独立提出新理论、评审他人工作、教学相长。
    """
    metaphor = learner.metaphor_tracker
    tom = learner.theory_of_mind
    world.configure_for_stage('late_formal')

    cloze_successes = 0
    sentence_successes = 0
    teaching_successes = 0
    review_successes = 0
    total_games = 0
    strategies_used = {}

    for step in range(rounds):
        game_type = random.random()

        if game_type < 0.20:
            # 填空题 — 博士级专业词汇
            blanked, target, options, _ = generate_cloze_from_vocab(
                PHD_VOCAB, PHD_TEMPLATES)
            success = learner.communication.play_cloze_game(
                blanked, target, options)
            learner.reading_history.append(1.0 if success else 0.0)
            if success:
                cloze_successes += 1
            total_games += 1

        elif game_type < 0.35:
            # 造句题 — 原创性表达
            target_words, available = generate_sentence_exercise_from_vocab(
                PHD_VOCAB, PHD_TEMPLATES)
            success = learner.communication.play_sentence_game(
                target_words, available)
            learner.writing_history.append(1.0 if success else 0.0)
            if success:
                sentence_successes += 1
            total_games += 1

        elif game_type < 0.60:
            # 教学场景 — 向他人解释概念
            world.physics.step(dt=0.1)
            scene = world.generate_scene_features()
            if not scene:
                total_games += 1
                continue

            target = random.randint(0, len(scene) - 1)
            target_features = scene[target] if isinstance(scene[target], dict) else {}

            # 教学者视角：假设听众只知道部分信息
            if tom:
                visible_count = max(1, len(scene) // 3)
                visible_to_student = random.sample(scene, visible_count)
                student_facts = set()
                for obj in visible_to_student:
                    for val in obj.values():
                        if val:
                            student_facts.add(str(val))
                tom.model_other('student', student_facts)

                # 调整描述以适应学生水平
                adjusted = tom.adjust_description(target_features, scene, 'student')
                utterance = adjusted if adjusted else list(target_features.values())[:3]
            else:
                utterance = list(target_features.values())[:3]

            # 教学对话
            context = SocialContext.EMPATHY
            try:
                ctx = derive_social_context(world, target, context)
                pragmatics.apply_context(context, utterance, ctx)
            except Exception:
                pass

            raw_input = world.observe()
            obs = learner.perceive(raw_input)
            success = learner.play_reference_game(scene, target)
            if success:
                teaching_successes += 1
            total_games += 1

        elif game_type < 0.80:
            # 同行评审 — 批判性评估
            world.physics.step(dt=0.1)
            scene = world.generate_scene_features()
            if not scene:
                total_games += 1
                continue

            # 用 DEBATE/MORAL 模拟评审中的批判
            target = random.randint(0, len(scene) - 1)
            context = random.choice([SocialContext.DEBATE, SocialContext.MORAL,
                                      SocialContext.REPAIR])

            raw_input = world.observe()
            obs = learner.perceive(raw_input)

            # 内在言语规划评审意见
            try:
                planned = learner.inner_speech.plan_description(scene, target)
            except Exception:
                planned = []

            target_features = scene[target] if isinstance(scene[target], dict) else {}
            utterance = planned if planned else list(target_features.values())[:3]
            try:
                ctx = derive_social_context(world, target, context)
                pragmatics.apply_context(context, utterance, ctx)
            except Exception:
                pass

            success = learner.play_reference_game(scene, target)
            if success:
                review_successes += 1
            total_games += 1

        else:
            # 元认知 + 策略优化
            meta = learner.metacognition
            if meta:
                try:
                    eval_result = meta.self_evaluate()
                    plan = meta.plan_next_learning()
                    strategy_name = plan.get('strategy', {}).get('name', 'explore')
                    strategies_used[strategy_name] = strategies_used.get(strategy_name, 0) + 1
                except Exception:
                    pass

            # 原创研究叙事
            world.physics.step(dt=0.15)
            scene_before = world.generate_scene_features()
            obs = learner.perceive(world.observe())
            action = learner.choose_action(obs)
            world.physics.step(dt=0.2)
            scene_after = world.generate_scene_features()

            events = detect_events(world, scene_before, scene_after)
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()

            # 因果 + 反事实
            causal = learner.causal
            cf = learner.counterfactual
            if causal and events:
                for ev in events:
                    cause = describe_event_subject(ev)
                    effect = describe_event_action(ev)
                    causal.observe(cause, effect)

            if cf:
                cf_world = CounterfactualWorld()
                for alt in ['propose', 'challenge', 'synthesize', 'reject']:
                    cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
                actual_action = random.choice(['propose', 'challenge'])
                cf.reason(actual_action, f"result_{actual_action}", cf_world)

            # 隐喻
            if metaphor and symbols:
                metaphor.record(symbols, {'features': scene_before})

            total_games += 1

        if verbose and (step + 1) % 100 == 0:
            c_rate = cloze_successes / max(total_games * 0.20, 1)
            s_rate = sentence_successes / max(total_games * 0.15, 1)
            t_rate = teaching_successes / max(total_games * 0.25, 1)
            r_rate = review_successes / max(total_games * 0.20, 1)
            print(f"  博士 step {step+1}/{rounds} | "
                  f"cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
                  f"teaching={t_rate:.3f} | review={r_rate:.3f}")

    # 最终评估
    c_rate = cloze_successes / max(rounds * 0.20, 1)
    s_rate = sentence_successes / max(rounds * 0.15, 1)
    t_rate = teaching_successes / max(rounds * 0.25, 1)
    r_rate = review_successes / max(rounds * 0.20, 1)

    metaphors_found = {}
    if metaphor:
        metaphors_found = metaphor.detect_metaphors()

    if verbose:
        print(f"  → 博士完成 | cloze={c_rate:.3f} | sentence={s_rate:.3f} | "
              f"teaching={t_rate:.3f} | review={r_rate:.3f}")
        print(f"    隐喻: {list(metaphors_found.keys())}")
        print(f"    策略分布: {strategies_used}")
    return c_rate, s_rate, t_rate, r_rate, len(metaphors_found), strategies_used


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("10 阶段发展式学习管线 — 从感知到博士")
    print("=" * 60)

    config = LearnerConfig()
    learner = Learner(config)
    world = World(config)
    pragmatics = PragmaticsModule()
    evaluator = CapabilityEvaluator()

    # 尝试从检查点恢复
    latest_ckpt = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    if os.path.exists(latest_ckpt):
        try:
            learner.load(latest_ckpt)
            print(f"从检查点恢复: {latest_ckpt}")
        except Exception as e:
            print(f"检查点加载失败: {e}，从头开始")

    start_time = time.time()
    results = {}

    # ════════════════════════════════════════════════════════════════
    # 童年阶段（0-6 岁）
    # ════════════════════════════════════════════════════════════════

    # ── 阶段 1: 感知预测 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 1: 感知预测（0-2 岁）")
    print("目标: 从物理世界学习因果直觉")
    print(f"{'='*60}")
    pred_acc = stage_perception_prediction(learner, world, config, rounds=300)
    results['stage1_perception'] = {'prediction_accuracy': pred_acc}

    # ── 阶段 2: 符号接地 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 2: 符号接地（2-3 岁）")
    print("目标: 从感知中发现概念 → 命名")
    print(f"{'='*60}")
    comm_rate, vocab_size = stage_symbol_grounding(learner, world, config,
                                                    rounds=200)
    results['stage2_grounding'] = {
        'communication_success': comm_rate,
        'vocabulary_size': vocab_size,
    }

    # ── 阶段 3: 组合表达 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 3: 组合表达（3-4 岁）")
    print("目标: 从单词到句子")
    print(f"{'='*60}")
    cloze_rate, sent_rate = stage_compositional_expression(
        learner, world, config, rounds=300)
    results['stage3_composition'] = {
        'cloze_rate': cloze_rate,
        'sentence_rate': sent_rate,
    }

    # ── 阶段 4: 因果推理 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 4: 因果推理（4-5 岁）")
    print("目标: 理解因果关系 → 表达 because")
    print(f"{'='*60}")
    causal_rules = stage_causal_reasoning(learner, world, config, rounds=200)
    results['stage4_causal'] = {'causal_rules': causal_rules}

    # ── 阶段 5: 社会理解 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 5: 社会理解（5-6 岁）")
    print("目标: 理解他人视角 → 调整表达")
    print(f"{'='*60}")
    persp_markers = stage_social_understanding(
        learner, world, config, pragmatics, rounds=200)
    results['stage5_social'] = {'perspective_markers': persp_markers}

    # ════════════════════════════════════════════════════════════════
    # 中学阶段（12-18 岁）
    # ════════════════════════════════════════════════════════════════

    # ── 阶段 6: 初中 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 6: 初中（12-15 岁）")
    print("目标: 语法深化 + 叙事连接词 + 隐喻萌芽")
    print(f"{'='*60}")
    ms_cloze, ms_sent, ms_conn, ms_meta = stage_middle_school(
        learner, world, config, pragmatics, rounds=400)
    results['stage6_middle_school'] = {
        'cloze_rate': ms_cloze, 'sentence_rate': ms_sent,
        'connectors': ms_conn, 'metaphors': ms_meta,
    }

    # ── 阶段 7: 高中 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 7: 高中（15-18 岁）")
    print("目标: 批判性思维 + 论证 + 类比推理")
    print(f"{'='*60}")
    hs_cloze, hs_sent, hs_debate, hs_meta = stage_high_school(
        learner, world, config, pragmatics, rounds=400)
    results['stage7_high_school'] = {
        'cloze_rate': hs_cloze, 'sentence_rate': hs_sent,
        'debate_rate': hs_debate, 'metaphors': hs_meta,
    }

    # ════════════════════════════════════════════════════════════════
    # 高等教育阶段（18-28 岁）
    # ════════════════════════════════════════════════════════════════

    # ── 阶段 8: 大学 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 8: 大学（18-22 岁）")
    print("目标: 学术英语 + 语用精通 + 跨模态")
    print(f"{'='*60}")
    uni_cloze, uni_sent, uni_dial, uni_listen, uni_markers = stage_university(
        learner, world, config, pragmatics, rounds=500)
    results['stage8_university'] = {
        'cloze_rate': uni_cloze, 'sentence_rate': uni_sent,
        'dialogue_rate': uni_dial, 'listening_rate': uni_listen,
        'pragmatic_markers': uni_markers,
    }

    # ── 阶段 9: 硕士 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 9: 硕士（22-24 岁）")
    print("目标: 研究写作 + 元认知 + 假设验证")
    print(f"{'='*60}")
    m_cloze, m_sent, m_res, m_meta, m_strat = stage_masters(
        learner, world, config, pragmatics, rounds=400)
    results['stage9_masters'] = {
        'cloze_rate': m_cloze, 'sentence_rate': m_sent,
        'research_rate': m_res, 'metaphors': m_meta,
        'strategies': m_strat,
    }

    # ── 阶段 10: 博士 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("阶段 10: 博士（24-28 岁）")
    print("目标: 原创贡献 + 同行评审 + 教学")
    print(f"{'='*60}")
    p_cloze, p_sent, p_teach, p_review, p_meta, p_strat = stage_phd(
        learner, world, config, pragmatics, rounds=400)
    results['stage10_phd'] = {
        'cloze_rate': p_cloze, 'sentence_rate': p_sent,
        'teaching_rate': p_teach, 'review_rate': p_review,
        'metaphors': p_meta, 'strategies': p_strat,
    }

    # ── 最终评估 ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    metrics = evaluator.evaluate(learner)

    print(f"\n{'='*60}")
    print("最终评估")
    print(f"{'='*60}")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # 保存检查点
    try:
        learner.save(latest_ckpt)
        print(f"\n检查点已保存: {latest_ckpt}")
    except Exception as e:
        print(f"\n检查点保存失败: {e}")

    # 保存结果
    results['final_metrics'] = metrics
    results['elapsed'] = elapsed
    results_file = os.path.join(
        RESULTS_DIR,
        f"developmental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"结果已保存: {results_file}")

    # 总结
    print(f"\n{'='*60}")
    print("10 阶段发展总结")
    print(f"{'='*60}")
    print(f"  阶段 1 感知预测 (0-2岁):  prediction_accuracy={pred_acc:.3f}")
    print(f"  阶段 2 符号接地 (2-3岁):  vocab={vocab_size}, comm_rate={comm_rate:.3f}")
    print(f"  阶段 3 组合表达 (3-4岁):  cloze={cloze_rate:.3f}, sentence={sent_rate:.3f}")
    print(f"  阶段 4 因果推理 (4-5岁):  causal_rules={causal_rules}")
    print(f"  阶段 5 社会理解 (5-6岁):  perspective_markers={persp_markers}")
    print(f"  阶段 6 初中 (12-15岁):    cloze={ms_cloze:.3f}, connectors={ms_conn}")
    print(f"  阶段 7 高中 (15-18岁):    cloze={hs_cloze:.3f}, debate={hs_debate:.3f}")
    print(f"  阶段 8 大学 (18-22岁):    cloze={uni_cloze:.3f}, dialogue={uni_dial:.3f}")
    print(f"  阶段 9 硕士 (22-24岁):    research={m_res:.3f}, metaphors={m_meta}")
    print(f"  阶段 10 博士 (24-28岁):   teaching={p_teach:.3f}, review={p_review:.3f}")


if __name__ == '__main__':
    main()
