"""
通用学习系统 — 大学英语进阶 (CET-4 → CET-6 → Academic → Professional)

从高中检查点加载继续学习。激活沉睡的游戏类型和模块：
  - play_cloze_game()     填空题 → 阅读理解
  - play_sentence_game()  造句题 → 语法涌现（解决 compounds=0 问题）
  - play_listening_game() 听力题 → 听觉理解
  - NarrativeModule       叙事理解 → 事件序列 + 连接词
  - PragmaticsModule      对话语用 → 社会情境标记
  - InnerSpeechModule     内在言语 → 说话前规划
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
from src.environment.world import World
from src.language.pragmatics import PragmaticsModule, SocialContext
from src.curriculum.evaluator import CapabilityEvaluator

CHECKPOINT_DIR = 'checkpoints/english'
RESULTS_DIR = 'results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 大学英语词汇库 + 场景配置
# ══════════════════════════════════════════════════════════════════

LEVELS = {
    1: {
        'name': 'CET-4 — 大学英语四级',
        'rounds': 500,
        'num_objects': 4,
        'scene_mix': {'reference': 0.40, 'cloze': 0.25, 'sentence': 0.15, 'narrative': 0.10, 'dialogue': 0.10},
        'attributes': {
            'agent': ['student', 'teacher', 'friend', 'doctor', 'child', 'mother', 'father',
                      'scientist', 'artist', 'worker', 'manager', 'he', 'she', 'they', 'we'],
            'action': ['study', 'work', 'read', 'write', 'speak', 'listen', 'think', 'know',
                       'learn', 'teach', 'help', 'want', 'need', 'like', 'love', 'try',
                       'start', 'stop', 'begin', 'finish', 'open', 'close', 'give', 'take',
                       'buy', 'sell', 'make', 'bring', 'send', 'show', 'tell', 'ask',
                       'answer', 'call', 'meet', 'find', 'keep', 'change', 'follow', 'lead',
                       'build', 'grow', 'move', 'travel', 'visit', 'enjoy', 'share'],
            'patient': ['book', 'problem', 'idea', 'plan', 'question', 'answer', 'story',
                        'letter', 'email', 'project', 'report', 'exam', 'class', 'lesson',
                        'test', 'homework', 'library', 'computer', 'phone', 'music',
                        'movie', 'game', 'sport', 'food', 'money', 'time', 'job'],
            'adj': ['important', 'difficult', 'easy', 'interesting', 'useful', 'necessary',
                    'possible', 'different', 'similar', 'special', 'common', 'simple',
                    'complex', 'modern', 'traditional', 'natural', 'social', 'cultural',
                    'academic', 'professional', 'personal', 'public', 'private'],
            'adverb': ['quickly', 'slowly', 'carefully', 'easily', 'often', 'usually',
                       'always', 'never', 'sometimes', 'already', 'still', 'also', 'only'],
            'place': ['university', 'classroom', 'office', 'hospital', 'museum', 'theater',
                      'restaurant', 'airport', 'station', 'market', 'internet'],
            'modal': ['can', 'should', 'must', 'will', 'would', 'could', 'may', 'might'],
        },
        'templates': [
            "the {agent} {action} the {patient}",
            "we {action} {adverb} at the {place}",
            "the {adj} {patient} is very {adj}",
            "{agent} {modal} {action} the {patient}",
            "they {action} the {adj} {patient}",
        ],
    },
    2: {
        'name': 'CET-6 — 大学英语六级',
        'rounds': 550,
        'num_objects': 5,
        'scene_mix': {'reference': 0.30, 'cloze': 0.20, 'sentence': 0.20, 'narrative': 0.15, 'dialogue': 0.15},
        'attributes': {
            'agent': ['researcher', 'professor', 'engineer', 'economist', 'journalist',
                      'politician', 'philosopher', 'psychologist', 'sociologist', 'analyst',
                      'critic', 'advocate', 'entrepreneur', 'volunteer', 'citizen'],
            'action': ['analyze', 'evaluate', 'investigate', 'demonstrate', 'establish',
                       'indicate', 'suggest', 'reveal', 'confirm', 'challenge', 'transform',
                       'contribute', 'influence', 'determine', 'identify', 'explore',
                       'promote', 'regulate', 'implement', 'coordinate', 'negotiate'],
            'patient': ['theory', 'hypothesis', 'phenomenon', 'evidence', 'conclusion',
                        'argument', 'perspective', 'approach', 'method', 'framework',
                        'system', 'policy', 'strategy', 'principle', 'concept',
                        'institution', 'community', 'environment', 'economy', 'technology'],
            'adj': ['significant', 'comprehensive', 'fundamental', 'controversial', 'innovative',
                    'sustainable', 'efficient', 'effective', 'relevant', 'adequate',
                    'crucial', 'substantial', 'remarkable', 'inevitable', 'prevalent'],
            'adverb': ['significantly', 'considerably', 'substantially', 'ultimately',
                       'consequently', 'furthermore', 'nevertheless', 'meanwhile',
                       'specifically', 'relatively', 'approximately', 'predominantly'],
            'place': ['society', 'industry', 'sector', 'region', 'continent', 'market',
                      'network', 'platform', 'institution', 'organization'],
            'modal': ['can', 'should', 'must', 'will', 'would', 'could', 'may', 'might',
                      'shall', 'ought'],
        },
        'templates': [
            "the {agent} {action} the {adj} {patient}",
            "because the {patient} is {adj} we {modal} {action} it",
            "the {agent} {modal} {action} the {patient} {adverb}",
            "research {modal} that the {patient} is {adj}",
            "{adverb} the {agent} {action} the {patient}",
        ],
    },
    3: {
        'name': 'Academic — 学术英语',
        'rounds': 600,
        'num_objects': 5,
        'scene_mix': {'reference': 0.20, 'cloze': 0.20, 'sentence': 0.20, 'narrative': 0.20, 'dialogue': 0.20},
        'attributes': {
            'agent': ['study', 'research', 'analysis', 'survey', 'experiment',
                      'review', 'meta-analysis', 'observation', 'simulation', 'model'],
            'action': ['examine', 'assess', 'compare', 'correlate', 'validate',
                       'replicate', 'synthesize', 'contextualize', 'theorize', 'quantify',
                       'qualify', 'differentiate', 'integrate', 'optimize', 'formalize'],
            'patient': ['variable', 'correlation', 'causation', 'distribution', 'probability',
                        'significance', 'sample', 'population', 'parameter', 'criterion',
                        'paradigm', 'methodology', 'epistemology', 'ontology', 'taxonomy'],
            'adj': ['empirical', 'theoretical', 'qualitative', 'quantitative', 'longitudinal',
                    'cross-sectional', 'peer-reviewed', 'multidisciplinary', 'robust', 'rigorous',
                    'systematic', 'comparative', 'exploratory', 'confirmatory', 'normative'],
            'adverb': ['empirically', 'theoretically', 'methodologically', 'statistically',
                       'respectively', 'independently', 'collectively', 'reciprocally'],
            'place': ['literature', 'dataset', 'repository', 'archive', 'registry'],
            'modal': ['can', 'may', 'might', 'should', 'would', 'could'],
        },
        'templates': [
            "the {agent} {modal} {action} the {adj} {patient}",
            "the {patient} {modal} be {action}ed by the {agent}",
            "results {modal} that the {patient} is {adj}",
            "this {adj} {patient} {modal} be {action}ed {adverb}",
            "the {agent} {action} {adverb} to {action} the {patient}",
        ],
    },
    4: {
        'name': 'Professional — 专业英语',
        'rounds': 650,
        'num_objects': 6,
        'scene_mix': {'reference': 0.15, 'cloze': 0.15, 'sentence': 0.20, 'narrative': 0.25, 'dialogue': 0.25},
        'attributes': {
            'agent': ['committee', 'board', 'council', 'panel', 'team',
                      'stakeholder', 'executive', 'consultant', 'director', 'coordinator'],
            'action': ['authorize', 'allocate', 'prioritize', 'facilitate', 'streamline',
                       'benchmark', 'outsourcing', 'consolidate', 'diversify', 'leverage',
                       'mitigate', 'escalate', 'delegate', 'compliance', 'implement'],
            'patient': ['initiative', 'framework', 'infrastructure', 'roadmap', 'milestone',
                        'deliverable', 'budget', 'timeline', 'workforce', 'portfolio',
                        'stakeholder', 'partnership', 'governance', 'compliance', 'benchmark'],
            'adj': ['strategic', 'operational', 'tactical', 'scalable', 'actionable',
                    'measurable', 'sustainable', 'innovative', 'competitive', 'compliant',
                    'transparent', 'accountable', 'collaborative', 'cross-functional', 'mission-critical'],
            'adverb': ['proactively', 'strategically', 'collaboratively', 'transparently',
                       'systematically', 'iteratively', 'effectively', 'efficiently'],
            'place': ['headquarters', 'subsidiary', 'division', 'department', 'branch'],
            'modal': ['shall', 'will', 'must', 'should', 'would', 'could', 'may'],
        },
        'templates': [
            "the {agent} {modal} {action} the {adj} {patient}",
            "the {adj} {patient} {modal} be {action}ed before the {patient}",
            "we {modal} {action} the {patient} {adverb}",
            "{adverb} the {agent} {action} the {patient} to {action} the {patient}",
            "the {patient} {modal} be {action}ed by the {agent} {adverb}",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════
# 场景生成
# ══════════════════════════════════════════════════════════════════

def make_object_scene(lc: dict) -> list:
    """生成物体参照场景"""
    attrs = lc['attributes']
    attr_keys = list(attrs.keys())
    n = lc['num_objects']
    scene = []
    for _ in range(n):
        obj = {}
        n_attrs = random.randint(2, min(3, len(attr_keys)))
        for key in random.sample(attr_keys, n_attrs):
            obj[key] = random.choice(attrs[key])
        scene.append(obj)
    return scene


def generate_sentence(lc: dict) -> tuple:
    """从模板生成句子，返回 (sentence, target_word, category_of_target)"""
    template = random.choice(lc['templates'])
    attrs = lc['attributes']

    # 收集模板中的占位符
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    fills = {}
    for ph in placeholders:
        ph_lower = ph.lower()
        if ph_lower in attrs:
            fills[ph] = random.choice(attrs[ph_lower])
        elif ph_lower == 'verb_ed':
            verbs = attrs.get('action', ['use'])
            fills[ph] = random.choice(verbs) + 'ed'
        else:
            fills[ph] = random.choice(attrs.get('action', ['do']))

    # 填充模板
    sentence = template
    for ph, val in fills.items():
        sentence = sentence.replace('{' + ph + '}', val)

    # 选择目标词（从内容词中选）
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


def generate_cloze_exercise(lc: dict) -> tuple:
    """生成填空题：sentence, target_word, options"""
    sentence, target_word, target_cat, _ = generate_sentence(lc)
    attrs = lc['attributes']

    # 从同一类别生成干扰项
    distractors = []
    if target_cat and target_cat in attrs:
        pool = [w for w in attrs[target_cat] if w != target_word]
        distractors = random.sample(pool, min(3, len(pool)))
    if len(distractors) < 3:
        all_words = [w for ws in attrs.values() for w in ws if w != target_word]
        while len(distractors) < 3:
            d = random.choice(all_words)
            if d not in distractors:
                distractors.append(d)

    options = [target_word] + distractors[:3]
    random.shuffle(options)
    # 替换句子中的目标词为 ___
    blanked = sentence.replace(target_word, '___', 1)
    return blanked, target_word, options, sentence


def generate_sentence_exercise(lc: dict) -> tuple:
    """生成造句题：target_words, available_words"""
    _, _, _, words = generate_sentence(lc)
    if not words:
        words = ['the', 'student', 'study', 'hard']

    target_words = [w for w in words if w]

    # 加入干扰词
    attrs = lc['attributes']
    all_words = [w for ws in attrs.values() for w in ws]
    distractor_count = random.randint(1, 2)
    distractors = random.sample(all_words, min(distractor_count, len(all_words)))

    available_words = target_words + distractors
    random.shuffle(available_words)
    return target_words, available_words


def generate_narrative_events(lc: dict) -> list:
    """生成叙事事件序列"""
    attrs = lc['attributes']
    agents = attrs.get('agent', ['he', 'she'])
    actions = attrs.get('action', ['do', 'go'])
    patients = attrs.get('patient', ['thing', 'work'])
    adjs = attrs.get('adj', ['good', 'big'])

    n_events = random.randint(2, 4)
    events = []
    for _ in range(n_events):
        agent = random.choice(agents)
        action = random.choice(actions)
        patient = random.choice(patients)
        adj = random.choice(adjs)

        events.append({
            'subject': {'entity': agent, 'property': adj},
            'action': action,
            'object': {'entity': patient},
        })
    return events


def scene_to_raw_input(scene: list) -> dict:
    """场景 → 感知输入"""
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


# ══════════════════════════════════════════════════════════════════
# 感知适配层 — 从物理世界推导社会情境
# ══════════════════════════════════════════════════════════════════

# SocialContext → 压力类型映射（与 PragmaticsModule 内部一致）
_CONTEXT_PRESSURE = {
    SocialContext.DEBATE: 'conflict',
    SocialContext.POLITENESS: 'social_distance',
    SocialContext.HUMOR: 'playfulness',
    SocialContext.MORAL: 'resource_conflict',
    SocialContext.EMPATHY: 'perspective_shift',
    SocialContext.NEGOTIATION: 'goal_conflict',
    SocialContext.DIALECT: 'identity',
    SocialContext.CRYPTOLECT: 'secrecy',
    SocialContext.REPAIR: 'misunderstanding',
    SocialContext.OWNERSHIP: 'possession',
    SocialContext.SOCIAL_NORMS: 'coordination',
    SocialContext.QUANTITATIVE: 'counting',
    SocialContext.SPATIAL: 'spatial',
}


def derive_social_context(world, target_idx: int, context: SocialContext) -> dict:
    """从物理世界状态推导 PragmaticsModule 需要的社会情境参数。

    核心思路：不直接生成符号字典，而是从真实物理状态中推导社会压力。
    """
    if not world.physics.objects or target_idx >= len(world.physics.objects):
        return {'has_conflict': False}

    target_obj = world.physics.objects[target_idx]
    agent_pos = world.agent_pos

    # 基础：物体的感知属性（纯字符串）
    ctx = dict(target_obj.get_properties())

    n_objs = len(world.physics.objects)

    # 根据情境类型推导社会压力
    if context == SocialContext.DEBATE:
        same_color = [o for o in world.physics.objects
                      if o.color == target_obj.color and o.obj_id != target_obj.obj_id]
        ctx['has_conflict'] = len(same_color) > 0

    elif context == SocialContext.POLITENESS:
        dist = (target_obj.position[:2] - agent_pos).norm().item()
        ctx['social_distance'] = min(dist / 5.0, 1.0)
        ctx['is_stranger'] = dist > 3.0

    elif context == SocialContext.HUMOR:
        ctx['is_play'] = True
        ctx['unexpected'] = target_obj.size in ('tiny', 'huge')

    elif context == SocialContext.MORAL:
        same_mat = [o for o in world.physics.objects
                    if o.material == target_obj.material and o.obj_id != target_obj.obj_id]
        ctx['contested_resource'] = len(same_mat) > 0

    elif context == SocialContext.EMPATHY:
        ctx['different_perspective'] = n_objs > 5
        ctx['understanding_score'] = 0.5

    elif context == SocialContext.NEGOTIATION:
        ctx['goal_conflict'] = n_objs > 5
        ctx['resource_scarcity'] = len([o for o in world.physics.objects
                                         if o.color == target_obj.color]) <= 1

    elif context == SocialContext.DIALECT:
        ctx['region'] = target_obj.material  # 用材质代表方言区域

    elif context == SocialContext.CRYPTOLECT:
        ctx['is_group_member'] = target_obj.shape == 'circle'  # 圆形 = 内群体

    elif context == SocialContext.REPAIR:
        ctx['understanding_score'] = 0.3  # 低理解分触发修复

    elif context == SocialContext.OWNERSHIP:
        dist = (target_obj.position[:2] - agent_pos).norm().item()
        ctx['contested_resource'] = dist < 2.0

    elif context == SocialContext.SOCIAL_NORMS:
        ctx['coordination_needed'] = n_objs > 4

    elif context == SocialContext.QUANTITATIVE:
        ctx['needs_counting'] = n_objs > 4
        ctx['count_total'] = n_objs

    elif context == SocialContext.SPATIAL:
        ctx['spatial_ambiguous'] = n_objs > 6
        ctx['position'] = target_obj.position[:2].tolist()

    return ctx


def detect_events(world, scene_before: list, scene_after: list) -> list:
    """从物理状态变化检测事件，构造叙事事件序列。

    从真实的物理碰撞和状态变化中产生事件，而非随机生成。
    """
    events = []

    # 检测碰撞事件（detect_collisions 返回索引元组）
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

    # 检测位置变化（移动事件）
    for i, (b, a) in enumerate(zip(scene_before, scene_after)):
        if b != a and {'subject': dict(b), 'action': 'move', 'object': dict(a)} not in events:
            events.append({
                'subject': dict(b),
                'action': 'move',
                'object': dict(a),
            })

    # 保底：至少一个观察事件
    if not events and scene_before:
        events.append({
            'subject': dict(scene_before[0]),
            'action': 'observe',
            'object': dict(scene_before[min(1, len(scene_before) - 1)]),
        })

    return events


# ══════════════════════════════════════════════════════════════════
# 场景执行
# ══════════════════════════════════════════════════════════════════

def run_reference_scene(learner, lc: dict, world=None) -> bool:
    """参照游戏 — 优先使用 World 物理场景"""
    if world is not None:
        # 从真实物理世界生成场景
        world.configure_for_stage('late_concrete')
        world.physics.step(dt=0.1)
        scene = world.generate_scene_features()
        raw_input = world.observe()
    else:
        # 回退：从词汇池生成
        scene = make_object_scene(lc)
        raw_input = scene_to_raw_input(scene)

    target = random.randint(0, len(scene) - 1)
    obs = learner.perceive(raw_input)
    action = learner.choose_action(obs)

    # 下一步感知（用于学习）
    if world is not None:
        world.physics.step(dt=0.1)
        next_raw = world.observe()
    else:
        next_scene = make_object_scene(lc)
        next_raw = scene_to_raw_input(next_scene)
    next_obs = learner.perceive(next_raw)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward=1.0, error=error)

    success = learner.play_reference_game(scene, target)

    # 符号接地
    target_obj = scene[target]
    symbols_contexts = [(val, attr) for attr, val in target_obj.items() if val]
    if symbols_contexts:
        learner.grounding.ground_symbols_batch(symbols_contexts, obs)

    return success


def run_cloze_scene(learner, lc: dict) -> bool:
    """填空题"""
    blanked, target_word, options, full_sentence = generate_cloze_exercise(lc)
    success = learner.communication.play_cloze_game(blanked, target_word, options)
    learner.reading_history.append(1.0 if success else 0.0)
    return success


def run_sentence_scene(learner, lc: dict) -> bool:
    """造句题 — 触发 n-gram/collocation/compound/grammar 涌现"""
    target_words, available_words = generate_sentence_exercise(lc)
    success = learner.communication.play_sentence_game(target_words, available_words)
    learner.writing_history.append(1.0 if success else 0.0)
    return success


def run_narrative_scene(learner, lc: dict, world=None) -> bool:
    """叙事理解 — 从物理状态变化构造事件"""
    if world is not None:
        # 从真实物理世界生成事件
        world.configure_for_stage('early_concrete')
        world.physics.step(dt=0.05)
        scene_before = world.generate_scene_features()

        # agent 执行动作，产生碰撞/移动
        obs = learner.perceive(world.observe())
        action = learner.choose_action(obs)
        world.physics.step(dt=0.2)

        scene_after = world.generate_scene_features()
        events = detect_events(world, scene_before, scene_after)
    else:
        # 回退：从词汇池生成
        events = generate_narrative_events(lc)

    narrative = learner.narrative.build_narrative(events)
    symbols = narrative.to_symbols()

    if not symbols:
        return False

    # 将叙事事件转为参照场景
    scene = []
    for ev in events:
        obj = dict(ev.get('subject', {}))
        obj['action'] = ev.get('action', '')
        if ev.get('object'):
            obj.update(ev['object'])
        scene.append(obj)

    if not scene:
        return False

    target = random.randint(0, len(scene) - 1)

    # 感知编码
    if world is not None:
        raw_input = world.observe()
    else:
        raw_input = scene_to_raw_input(scene)
    obs = learner.perceive(raw_input)

    success = learner.play_reference_game(scene, target)

    # 听力测试
    narrative_text = ' '.join(symbols)
    options = [narrative_text]
    for _ in range(2):
        if world is not None:
            d_events = detect_events(world, scene_before, scene_after)
        else:
            d_events = generate_narrative_events(lc)
        d_narr = learner.narrative.build_narrative(d_events)
        d_text = ' '.join(d_narr.to_symbols())
        if d_text:
            options.append(d_text)

    if len(options) >= 2:
        listening_ok = learner.communication.play_listening_game(narrative_text, options)
        learner.listening_history.append(1.0 if listening_ok else 0.0)

    return success


def run_dialogue_scene(learner, lc: dict, pragmatics: PragmaticsModule,
                       world=None) -> bool:
    """对话语用 — 基于 World 物理感知"""
    contexts = list(SocialContext)
    context = random.choice(contexts)

    # 1. 从 World 物理场景生成（核心改变）
    if world is not None:
        world.configure_for_stage('late_concrete')
        world.physics.step(dt=0.1)
        scene = world.generate_scene_features()  # 纯字符串字典
    else:
        scene = make_object_scene(lc)

    if not scene:
        return False

    target = random.randint(0, len(scene) - 1)

    # 2. 感知编码
    if world is not None:
        raw_input = world.observe()
    else:
        raw_input = scene_to_raw_input(scene)
    obs = learner.perceive(raw_input)

    # 3. 内在言语规划（兼容 World 特征）
    try:
        planned = learner.inner_speech.plan_description(scene, target)
        used_planning = len(planned) > 0
    except Exception:
        planned = []
        used_planning = False

    # 4. 语用修饰（从物理状态推导社会压力）
    target_features = scene[target] if isinstance(scene[target], dict) else {}
    utterance = planned if planned else list(target_features.values())[:3]
    try:
        ctx = derive_social_context(world, target, context) if world else {'has_conflict': False}
        result = pragmatics.apply_context(context, utterance, ctx)
        if isinstance(result, dict):
            utterance = result.get('utterance', utterance)
    except Exception:
        pass

    # 5. 符号接地
    symbols_contexts = [(val, attr) for attr, val in target_features.items() if val]
    if symbols_contexts:
        try:
            learner.grounding.ground_symbols_batch(symbols_contexts, obs)
        except Exception:
            pass

    # 6. 参照游戏测试理解
    success = learner.play_reference_game(scene, target)

    # 7. 记录内在言语效果
    try:
        learner.inner_speech.record_outcome(success, used_planning)
    except Exception:
        pass

    return success


# ══════════════════════════════════════════════════════════════════
# 训练循环
# ══════════════════════════════════════════════════════════════════

SCENE_DISPATCH = {
    'reference': run_reference_scene,
    'cloze': run_cloze_scene,
    'sentence': run_sentence_scene,
    'narrative': run_narrative_scene,
    'dialogue': run_dialogue_scene,
}


def teach_level(learner, level_num, pragmatics, world=None, verbose=True):
    """教一个级别"""
    lc = LEVELS[level_num]
    scene_mix = lc['scene_mix']
    scene_types = list(scene_mix.keys())
    scene_weights = [scene_mix[st] for st in scene_types]

    stats = {st: {'success': 0, 'total': 0} for st in scene_types}

    for r in range(lc['rounds']):
        # 按权重随机选择场景类型
        scene_type = random.choices(scene_types, weights=scene_weights, k=1)[0]

        if scene_type == 'dialogue':
            success = run_dialogue_scene(learner, lc, pragmatics, world)
        elif scene_type in ('reference', 'narrative'):
            success = SCENE_DISPATCH[scene_type](learner, lc, world)
        else:
            success = SCENE_DISPATCH[scene_type](learner, lc)

        stats[scene_type]['total'] += 1
        if success:
            stats[scene_type]['success'] += 1

        # 进度日志
        if (r + 1) % 50 == 0 and verbose:
            total_s = sum(s['success'] for s in stats.values())
            total_g = sum(s['total'] for s in stats.values())
            rate = total_s / max(total_g, 1)
            vocab = len(learner.get_vocabulary())
            compounds = len(learner.communication.language.compounds)
            grammar = len(learner.communication.grammar.get_rules())
            print(f"    Round {r+1:4d}/{lc['rounds']} | "
                  f"总成功率 {rate:.1%} | 词汇 {vocab} | "
                  f"组合 {compounds} | 语法 {grammar}")

    # 巩固
    learner.consolidate()

    total_s = sum(s['success'] for s in stats.values())
    total_g = sum(s['total'] for s in stats.values())
    return {
        'success_rate': total_s / max(total_g, 1),
        'total_games': total_g,
        'total_successes': total_s,
        'scene_stats': {k: {**v, 'rate': v['success'] / max(v['total'], 1)}
                        for k, v in stats.items()},
    }


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("通用学习系统 — 大学英语进阶 (CET-4 → CET-6 → Academic → Professional)")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 从高中检查点加载
    config = LearnerConfig(obs_dim=40, action_dim=4, hidden_dims=(32, 16))
    learner = Learner(config)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'high_12_final.pt')
    if not os.path.exists(checkpoint_path):
        # 尝试寻找最新的检查点
        for name in ['high_12_final.pt', 'middle_9_final.pt', 'grade_6_final.pt']:
            p = os.path.join(CHECKPOINT_DIR, name)
            if os.path.exists(p):
                checkpoint_path = p
                break

    learner.load(checkpoint_path)
    print(f"[加载] {checkpoint_path}")
    print(f"  已有词汇: {len(learner.get_vocabulary())}")
    print(f"  已有接地: {len(learner.grounding.get_grounded_symbols())}")
    print(f"  沟通成功率: {learner.comm_success_rate:.1%}")
    print(f"  复合符号: {len(learner.communication.language.compounds)}")
    print(f"  语法规则: {len(learner.communication.grammar.get_rules())}")
    print()

    # PragmaticsModule（不在 Learner registry 中，单独管理）
    pragmatics = PragmaticsModule(vocabulary=learner.get_vocabulary())

    # World 物理环境（用于感知接地的对话/叙事场景）
    world = World(config)

    all_results = {}
    start_time = time.time()

    for level_num in range(1, 5):
        lc = LEVELS[level_num]
        print(f"{'='*70}")
        print(f"Level {level_num}: {lc['name']}")
        print(f"  场景配比: {lc['scene_mix']}")
        print(f"  练习轮数: {lc['rounds']}")
        print(f"{'='*70}")

        t0 = time.time()
        result = teach_level(learner, level_num, pragmatics, world)
        t1 = time.time()

        # 评估
        evaluator = CapabilityEvaluator()
        evaluation = evaluator.evaluate(learner)
        advanced = learner.try_advance(evaluation)

        # 保存检查点
        save_name = {1: 'advanced_cet4', 2: 'advanced_cet6',
                     3: 'academic', 4: 'professional_final'}[level_num]
        save_path = os.path.join(CHECKPOINT_DIR, f'{save_name}.pt')
        learner.save(save_path)

        vocab = len(learner.get_vocabulary())
        compounds = list(learner.communication.language.compounds.keys())[:5]
        grammar_count = len(learner.communication.grammar.get_rules())

        print(f"\n  [Level {level_num} 完成] {t1-t0:.1f}s")
        print(f"  总成功率: {result['success_rate']:.1%}")
        print(f"  各场景:")
        for st, ss in result['scene_stats'].items():
            print(f"    {st:12s}: {ss['rate']:.1%} ({ss['success']}/{ss['total']})")
        print(f"  累计词汇: {vocab}")
        print(f"  复合符号: {len(learner.communication.language.compounds)}")
        print(f"  语法规则: {grammar_count}")
        if compounds:
            print(f"  组合表达: {compounds}")
        print(f"  阶段: {learner.stage}")
        print(f"  保存: {save_path}")
        print()

        # 更新 pragmatics 词汇
        pragmatics = PragmaticsModule(vocabulary=learner.get_vocabulary())

        all_results[f'level_{level_num}'] = {
            **result,
            'vocab_after': vocab,
            'compounds': len(learner.communication.language.compounds),
            'grammar_rules': grammar_count,
            'time_seconds': round(t1 - t0, 1),
            'stage': learner.stage,
        }

    # ═══════════════════════════════════════════════════════════════
    # 最终评估
    # ═══════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    stats = learner.get_stats()

    print("=" * 70)
    print("大学英语进阶完成 — 最终评估")
    print("=" * 70)

    print(f"\n[总耗时] {total_time:.1f}s")
    print(f"[最终词汇] {stats['vocabulary_size']} 个")
    print(f"[沟通成功率] {stats['comm_success_rate']:.1%}")
    print(f"[复合符号] {len(learner.communication.language.compounds)} 个")
    print(f"[语法规则] {len(learner.communication.grammar.get_rules())} 个")

    # 各级别成绩
    print(f"\n[各级别成绩]")
    for lv in range(1, 5):
        key = f'level_{lv}'
        if key in all_results:
            r = all_results[key]
            bar = '#' * int(r['success_rate'] * 20)
            print(f"  Level {lv}: {r['success_rate']:.1%} {bar} | "
                  f"词汇 {r['vocab_after']} | 组合 {r['compounds']} | "
                  f"语法 {r['grammar_rules']} | {r['time_seconds']}s")

    # 词汇表（前 20）
    vocab = learner.get_vocabulary()
    if vocab:
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1].get('frequency', 0), reverse=True)
        print(f"\n[高频英语词汇] ({len(sorted_vocab)} 个)")
        for word, data in sorted_vocab[:20]:
            freq = data.get('frequency', 0)
            rate = data.get('success_rate', 0)
            bar = '#' * max(1, int(rate * 15))
            print(f"  {word:15s} | {freq:3d}次 | {rate:.0%} | {bar}")

    # 复合符号
    lang = learner.communication.language
    if lang.compounds:
        print(f"\n[涌现的组合表达] ({len(lang.compounds)} 个)")
        for compound, data in list(lang.compounds.items())[:15]:
            comps = data.get('components', [])
            print(f"  {compound:25s} <- {' + '.join(comps)}")

    # 语法规则
    grammar = learner.communication.grammar
    rules = grammar.get_rules()
    if rules:
        print(f"\n[涌现的语法规则] ({len(rules)} 个)")
        for r in rules[:10]:
            if hasattr(r, 'pattern'):
                print(f"  {r.pattern} ({r.category_pattern}) conf={r.confidence:.2f}")
            elif isinstance(r, dict):
                print(f"  {r.get('pattern', '?')} conf={r.get('confidence', 0):.2f}")

    # 搭配
    if lang.collocations:
        top = sorted(lang.collocations.items(), key=lambda x: x[1].get('count', 0), reverse=True)[:10]
        print(f"\n[高频搭配] ({len(lang.collocations)} 个)")
        for pair, data in top:
            print(f"  {pair}: {data.get('count', 0)}次, 成功率{data.get('success_rate', 0):.0%}")

    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': round(total_time, 1),
        'levels': all_results,
        'final_stats': {k: v for k, v in stats.items() if not isinstance(v, dict)},
    }
    with open(os.path.join(RESULTS_DIR, 'english_advanced_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n检查点: {CHECKPOINT_DIR}/")
    print(f"结果: {RESULTS_DIR}/english_advanced_results.json")
    print("=" * 70)

    return learner


if __name__ == '__main__':
    learner = main()
