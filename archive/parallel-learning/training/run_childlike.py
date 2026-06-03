"""
闭环学习系统 — 像小孩一样学英语

解决 6 个根本缺陷：
  1. 自言自语 → NaiveAgent（多 Agent 参照游戏）
  2. 无遗忘曲线 → Ebbinghaus 遗忘 + 间隔重复
  3. 符号硬编码 → 从感知聚类中涌现新符号
  4. 认知模块沉睡 → Causal/ToM/Metaphor/Meta 闭环
  5. 世界无动机 → MotivatedWorld（需求驱动）
  6. 语言与感知脱节 → GroundingModule 连接感知与符号

不修改任何模块源码。所有闭环逻辑在此脚本内实现。
"""

import sys
sys.path.insert(0, '.')

import json
import time
import random
import os
import re
import math
import torch
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from src.core.config import LearnerConfig
from src.core.learner import Learner
from src.environment.world import World
from src.language.communication import CommunicationProtocol
from src.language.emergence import EmergingLanguage
from src.language.pragmatics import PragmaticsModule, SocialContext
from src.language.narrative import NarrativeModule
from src.reasoning.causal import CausalReasoningModule
from src.reasoning.counterfactual import CounterfactualModule, CounterfactualWorld
from src.reasoning.theory_of_mind import TheoryOfMindModule
from src.reasoning.metaphor import MetaphorTracker
from src.curriculum.evaluator import CapabilityEvaluator

CHECKPOINT_DIR = 'checkpoints/childlike'
RESULTS_DIR = 'results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 修复 1: NaiveAgent — 多 Agent 参照游戏
# ══════════════════════════════════════════════════════════════════

class NaiveAgent:
    """
    第二 Agent（模拟父母/老师）

    关键特性：
    - 独立的 CommunicationProtocol（不同的 mastery）
    - 初始 mastery 更高（模拟已会说话的大人）
    - 成功时双方都学习，但 learner 学得更快
    """

    def __init__(self):
        self.communication = CommunicationProtocol()
        # 预设一些基础词汇的 mastery（模拟大人已有的知识）
        self._preset_vocabulary()

    def _preset_vocabulary(self):
        """预设基础词汇 — 模拟大人已会的词"""
        basic_words = [
            'red', 'blue', 'green', 'yellow',
            'circle', 'square', 'triangle',
            'small', 'medium', 'big',
            'wood', 'plastic', 'metal',
        ]
        for word in basic_words:
            self.communication.language.expose_symbol(word)
            # 多次暴露模拟已掌握
            for _ in range(5):
                self.communication.language.expose_symbol(word)


class DualAgentProtocol:
    """
    双 Agent 通信协议 — 解决自言自语问题

    learner 和 teacher 各有独立的 CommunicationProtocol，
    play_round 时 speaker 和 listener 是不同的对象。
    """

    def __init__(self, learner_comm: CommunicationProtocol,
                 teacher_comm: CommunicationProtocol):
        self.learner_comm = learner_comm
        self.teacher_comm = teacher_comm

    def play_round(self, scene: List[Dict], target_idx: int,
                   speaker_is_learner: bool = True) -> bool:
        """
        一轮参照游戏 — speaker 和 listener 是不同 Agent

        Args:
            scene: 场景特征列表
            target_idx: 目标物体索引
            speaker_is_learner: True=learner 说 teacher 听, False=反过来

        Returns:
            是否成功
        """
        if speaker_is_learner:
            speaker = self.learner_comm
            listener = self.teacher_comm
        else:
            speaker = self.teacher_comm
            listener = self.learner_comm

        return speaker.play_round(speaker, listener, scene, target_idx)


# ══════════════════════════════════════════════════════════════════
# 修复 2: 遗忘曲线 + 间隔重复
# ══════════════════════════════════════════════════════════════════

class ForgettingCurve:
    """
    Ebbinghaus 遗忘曲线 + 间隔重复调度

    每个符号独立跟踪 last_review 和 mastery 衰减。
    低 mastery 的符号排在 review_queue 前面。
    """

    def __init__(self, decay_rate: float = 0.05, review_interval: int = 100):
        self.decay_rate = decay_rate       # 每 review_interval 步衰减比例
        self.review_interval = review_interval
        self.symbol_data: Dict[str, Dict] = {}  # symbol -> {mastery, last_review, times_wrong}
        self.review_queue: List[str] = []

    def update(self, symbol: str, success: bool, current_step: int):
        """更新符号的掌握度"""
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = {
                'mastery': 0.1,
                'last_review': current_step,
                'times_wrong': 0,
            }

        data = self.symbol_data[symbol]
        if success:
            # 成功：增加掌握度（间隔重复奖励）
            data['mastery'] = min(1.0, data['mastery'] + 0.15 * (1.0 - data['mastery']))
        else:
            # 失败：小幅降低 + 记录错误次数
            data['mastery'] = max(0.01, data['mastery'] * 0.85)
            data['times_wrong'] += 1

        data['last_review'] = current_step

    def apply_decay(self, current_step: int):
        """对所有符号应用遗忘衰减"""
        for symbol, data in self.symbol_data.items():
            elapsed = current_step - data['last_review']
            if elapsed > 0:
                decay = (1.0 - self.decay_rate) ** (elapsed / self.review_interval)
                data['mastery'] *= decay
                data['mastery'] = max(0.01, data['mastery'])

    def get_review_symbols(self, n: int = 5) -> List[str]:
        """获取需要复习的符号（按 mastery 升序排列）"""
        sorted_symbols = sorted(
            self.symbol_data.items(),
            key=lambda x: x[1]['mastery']
        )
        return [s for s, _ in sorted_symbols[:n]]

    def get_mastery(self, symbol: str) -> float:
        """获取符号当前掌握度"""
        if symbol in self.symbol_data:
            return self.symbol_data[symbol]['mastery']
        return 0.0

    def get_low_mastery_count(self, threshold: float = 0.3) -> int:
        """统计低掌握度符号数量"""
        return sum(1 for d in self.symbol_data.values() if d['mastery'] < threshold)


# ══════════════════════════════════════════════════════════════════
# 修复 3: 涌现词汇 — 从感知聚类中产生新符号
# ══════════════════════════════════════════════════════════════════

class EmergentVocabulary:
    """
    涌现词汇管理器

    符号从 GroundingModule 的感知聚类中涌现，
    而非从预定义的 COLORS/SHAPES 列表中选择。
    """

    def __init__(self):
        self.cluster_to_symbol: Dict[int, str] = {}
        self.symbol_to_cluster: Dict[str, int] = {}
        self.next_id = 0

    def discover_symbol(self, learner, obs: torch.Tensor,
                        scene_features: Optional[Dict] = None) -> str:
        """
        从感知中发现或创建符号

        1. 用 GroundingModule 聚类
        2. 如果聚类已有名称，返回它
        3. 否则创造新符号

        Args:
            learner: Learner 实例
            obs: 感知向量
            scene_features: 可选的场景特征字典（用于语义命名）

        Returns:
            符号字符串
        """
        cluster_id = learner.ground_concept(obs)

        if cluster_id in self.cluster_to_symbol:
            return self.cluster_to_symbol[cluster_id]

        # 新聚类 → 创造新符号
        if scene_features:
            # 尝试用场景特征命名（如 "red-circle"）
            name_parts = []
            for key in ['color', 'shape', 'size', 'material']:
                if key in scene_features:
                    name_parts.append(str(scene_features[key]))
            if name_parts:
                new_symbol = '-'.join(name_parts)
            else:
                new_symbol = f"thing-{self.next_id}"
        else:
            new_symbol = f"thing-{self.next_id}"

        self.next_id += 1
        self.cluster_to_symbol[cluster_id] = new_symbol
        self.symbol_to_cluster[new_symbol] = cluster_id
        return new_symbol

    def get_symbol_count(self) -> int:
        return len(self.cluster_to_symbol)


# ══════════════════════════════════════════════════════════════════
# 修复 5: MotivatedWorld — 需求驱动的世界
# ══════════════════════════════════════════════════════════════════

class MotivatedWorld(World):
    """
    有需求的世界 — Agent 有内在动机学习语言

    需求动态变化，驱动不同行为：
    - 好奇心：看到新事物时升高 → 驱动探索
    - 社交需求：孤独时升高 → 驱动参照游戏
    - 沟通需求：有想法但说不出来时升高 → 驱动语言学习
    """

    def __init__(self, config: LearnerConfig):
        super().__init__(config)
        self.needs = {
            'curiosity': 1.0,
            'social': 0.3,
            'communication': 0.0,
        }
        self._steps_since_social = 0
        self._new_symbols_count = 0

    def step(self, action):
        """执行动作，同时更新需求"""
        obs, reward, done = super().step(action)

        # 需求动态变化
        self._steps_since_social += 1
        self.needs['curiosity'] = min(1.0, self.needs['curiosity'] + 0.005)
        self.needs['social'] = min(1.0, 0.3 + self._steps_since_social * 0.002)
        # 沟通需求：有新感知但无法表达时升高
        self.needs['communication'] = min(1.0, self._new_symbols_count * 0.05)

        return obs, reward, done

    def on_social_interaction(self):
        """社交互动后重置社交需求"""
        self._steps_since_social = 0
        self.needs['social'] = max(0.0, self.needs['social'] - 0.3)

    def on_symbol_discovered(self):
        """发现新符号时增加沟通需求"""
        self._new_symbols_count += 1
        self.needs['communication'] = min(1.0, self.needs['communication'] + 0.1)

    def on_communication_success(self):
        """沟通成功时降低沟通需求"""
        self._new_symbols_count = max(0, self._new_symbols_count - 1)
        self.needs['communication'] = max(0.0, self.needs['communication'] - 0.15)

    def get_dominant_need(self) -> str:
        """获取当前最强烈的需求"""
        return max(self.needs, key=self.needs.get)


# ══════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════

def detect_events(world, scene_before: list, scene_after: list) -> list:
    """检测场景变化事件（碰撞、移动等）

    返回的每个事件 dict 包含 'subject'（Dict[str,str]）和 'action'（str），
    符合 NarrativeModule.build_narrative() 的输入格式。
    """
    events = []
    if not scene_before or not scene_after:
        return events

    # 检测属性变化（color/shape/size/material）
    for i, (before, after) in enumerate(zip(scene_before, scene_after)):
        if before != after:
            changed_keys = [k for k in before if k in after and before[k] != after[k]]
            if changed_keys:
                events.append({
                    'subject': dict(before),
                    'action': 'change',
                    'object': dict(after),
                })

    # 检测物理碰撞（通过 PhysicsEngine 的碰撞记录）
    collisions = world.physics.detect_collisions()
    for ia, ib in collisions:
        subj = scene_before[ia] if ia < len(scene_before) else {'color': 'unknown'}
        obj = scene_before[ib] if ib < len(scene_before) else {'color': 'unknown'}
        events.append({
            'subject': dict(subj),
            'action': 'collide',
            'object': dict(obj),
        })

    # 如果没有事件，生成一个观察事件
    if not events and scene_before:
        events.append({
            'subject': dict(scene_before[0]),
            'action': 'observe',
            'object': dict(scene_before[min(1, len(scene_before) - 1)]),
        })

    return events


def describe_event_subject(ev: dict) -> str:
    """事件主体描述 — 从 dict 特征中提取可读字符串"""
    subj = ev.get('subject', {})
    if isinstance(subj, dict):
        parts = [str(v) for v in subj.values() if v]
        return '_'.join(parts[:2]) if parts else 'thing'
    return str(subj)


def describe_event_action(ev: dict) -> str:
    """事件动作描述"""
    action = ev.get('action', 'unknown')
    obj = ev.get('object', {})
    if isinstance(obj, dict):
        parts = [str(v) for v in obj.values() if v]
        target = '_'.join(parts[:2]) if parts else 'thing'
    else:
        target = str(obj)
    return f"{action}_{target}"


def action_to_tensor(action: int, action_dim: int) -> torch.Tensor:
    """将 action 整数转换为 one-hot tensor（world.step 需要）"""
    t = torch.zeros(action_dim)
    t[action] = 1.0
    return t


def choose_action_with_causal_bias(learner, obs: torch.Tensor,
                                    causal_rules: list) -> int:
    """
    带因果偏置的动作选择

    如果有高置信度的因果规则，偏向选择能触发已知因果的动作。
    否则回退到默认的好奇心驱动选择。
    """
    n_actions = learner.config.action_dim

    if causal_rules and random.random() < 0.3:
        # 30% 概率使用因果偏置
        # 选择高置信度因果规则对应的动作
        best_rule = max(causal_rules, key=lambda r: r.confidence)
        # 用 cause 的哈希值映射到动作空间
        action = hash(best_rule.cause) % n_actions
    else:
        # 70% 概率使用默认的好奇心驱动
        action = learner.choose_action(obs)

    return action


def derive_social_context(world, target_idx: int) -> dict:
    """从世界状态推导社会上下文"""
    scene = world.generate_scene_features()
    if target_idx < len(scene):
        target = scene[target_idx]
        return {
            'target_color': target.get('color', 'unknown'),
            'target_shape': target.get('shape', 'unknown'),
            'num_objects': len(scene),
        }
    return {'num_objects': len(scene)}


# ══════════════════════════════════════════════════════════════════
# 词汇库 — 按难度分层
# ══════════════════════════════════════════════════════════════════

SIMPLE_VOCAB = {
    'agent': ['baby', 'child', 'cat', 'dog', 'bird'],
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
}

SENTENCE_TEMPLATES = {
    'simple': [
        "the {agent} {action}",
        "{agent} {action} the {patient}",
    ],
    'medium': [
        "the {agent} {action} the {adj} {patient}",
        "{agent} {action} {adverb}",
        "the {adj} {patient} is very {adj}",
    ],
    'complex': [
        "the {agent} {action} the {adj} {patient} {adverb}",
        "{agent} {action} the {patient} because it is {adj}",
        "the {agent} {action} and then {action} the {patient}",
    ],
}


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
    'place': ['laboratory', 'library', 'museum', 'stadium', 'theater',
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


# ── 博后词汇 (Cross-domain research) ──────────────────────────────

POSTDOC_VOCAB = {
    'agent': ['postdoc', 'principal_investigator', 'co_investigator', 'collaborator',
              'partner', 'stakeholder', 'consultant', 'advisor', 'mentor', 'mentee'],
    'action': ['transfer', 'adapt', 'integrate', 'synthesize', 'bridge',
               'translate', 'scale', 'deploy', 'iterate', 'pivot',
               'commercialize', 'disseminate', 'popularize', 'democratize',
               'institutionalize', 'mainstream', 'operationalize', 'catalyze'],
    'patient': ['cross_domain_insight', 'interdisciplinary_method', 'transferable_skill',
                'collaborative_framework', 'shared_vocabulary', 'common_ground',
                'translation_gap', 'implementation_barrier', 'adoption_curve',
                'impact_metric', 'stakeholder_map', 'value_chain', 'ecosystem'],
    'adj': ['interdisciplinary', 'translational', 'applied', 'practical',
            'impactful', 'scalable', 'sustainable', 'replicable', 'adaptable',
            'modular', 'flexible', 'robust', 'resilient', 'antifragile'],
    'adverb': ['collaboratively', 'iteratively', 'incrementally', 'systematically',
               'strategically', 'tactically', 'operationally', 'commercially',
               'institutionally', 'globally', 'locally', 'cross_functionally'],
    'place': ['incubator', 'accelerator', 'innovation_hub', 'tech_transfer_office',
              'industry_partner', 'government_agency', 'ngo', 'think_tank',
              'startup', 'venture_studio'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'in_light_of', 'given_that', 'to_that_end', 'with_respect_to'],
}

POSTDOC_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} proposes to {action} the {patient} using {adj} methods",
    "the {adj} {patient} {modal} be {action}ed across {place}",
    "the {agent} {action} the {patient} by {action}ing the {adj} {patient}",
    "however the {agent} argues that the {patient} requires {adj} {action}",
    "furthermore the {adj} {patient} {modal} {action} the {patient} {adverb}",
    "the {agent} {action} that the {adj} {patient} is {adverb} {adj}",
    "the {agent} {action} the {patient} to {action} {adj} {patient}",
]


# ── 行业领袖词汇 (Strategic leadership) ──────────────────────────

INDUSTRY_VOCAB = {
    'agent': ['executive', 'director', 'vice_president', 'chief_officer',
              'board_member', 'chairperson', 'founder', 'co_founder',
              'general_manager', 'division_head', 'team_lead', 'stakeholder'],
    'action': ['strategize', 'orchestrate', 'align', 'empower', 'delegate',
               'negotiate', 'influence', 'persuade', 'inspire', 'transform',
               'restructure', 'optimize', 'innovate', 'disrupt', 'scale',
               'pivot', 'consolidate', 'diversify', 'globalize', 'localize'],
    'patient': ['strategy', 'vision', 'mission', 'value_proposition',
                'competitive_advantage', 'market_position', 'business_model',
                'revenue_stream', 'cost_structure', 'growth_engine',
                'organizational_culture', 'talent_pipeline', 'succession_plan',
                'governance_framework', 'risk_matrix', 'compliance_regime'],
    'adj': ['strategic', 'tactical', 'operational', 'visionary', 'transformative',
            'disruptive', 'innovative', 'sustainable', 'scalable', 'agile',
            'resilient', 'customer_centric', 'data_driven', 'evidence_based',
            'cross_functional', 'enterprise_wide', 'global', 'local'],
    'adverb': ['strategically', 'tactically', 'operationally', 'visionarily',
               'transformatically', 'disruptively', 'innovatively', 'sustainably',
               'scalably', 'agilely', 'resiliently', 'globally', 'locally'],
    'place': ['boardroom', 'executive_suite', 'war_room', 'strategy_offsite',
              'investor_meeting', 'analyst_day', 'earnings_call', 'town_hall',
              'innovation_lab', 'skunkworks'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must', 'shall'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'in_light_of', 'given_that', 'to_that_end', 'with_respect_to',
                  'bottom_line', 'net_net', 'at_the_end_of_the_day'],
}

INDUSTRY_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} argues that the {patient} {modal} be {action}ed {adverb}",
    "the {adj} {patient} represents a {adj} {patient} for the {agent}",
    "the {agent} {action} the {patient} to {action} {adj} {patient}",
    "however the {agent} contends that the {patient} requires {adj} {action}",
    "the {agent} {action} the {adj} {patient} by {action}ing the {patient}",
    "furthermore the {adj} {patient} {modal} be {action}ed across {place}",
    "the {agent} argues that {adj} {patient} is the key to {action}ing the {patient}",
]


# ── 思想领袖词汇 (Public intellectual discourse) ──────────────────

THOUGHT_LEADER_VOCAB = {
    'agent': ['thinker', 'intellectual', 'public_intellectual', 'commentator',
              'columnist', 'author', 'speaker', 'keynote_speaker', 'panelist',
              'moderator', 'influencer', 'opinion_leader', 'tastemaker', 'curator'],
    'action': ['articulate', 'advocate', 'champion', 'provoke', 'challenge',
               'reframe', 'recontextualize', 'humanize', 'demystify', 'popularize',
               'synthesize', 'distill', 'curate', 'amplify', 'catalyze',
               'galvanize', 'mobilize', 'legitimize', 'normalize', 'destigmatize'],
    'patient': ['narrative', 'discourse', 'conversation', 'dialogue', 'debate',
                'movement', 'cause', 'agenda', 'manifesto', 'vision',
                'worldview', 'lens', 'framework', 'paradigm', 'zeitgeist',
                'collective_consciousness', 'public_imagination', 'cultural_moment'],
    'adj': ['compelling', 'provocative', 'nuanced', 'accessible', 'resonant',
            'timely', 'timeless', 'urgent', 'important', 'necessary',
            'counterintuitive', 'paradigm_shifting', 'thought_provoking',
            'conversation_changing', 'culture_defining', 'movement_building'],
    'adverb': ['compellingly', 'provocatively', 'nuanced', 'accessibly',
               'resonantly', 'urgently', 'importantly', 'counterintuitively',
               'paradigmatically', 'categorically', 'unequivocally', 'fundamentally'],
    'place': ['podium', 'stage', 'platform', 'forum', 'salon',
              'op_ed_page', 'podcast_studio', 'broadcast_center',
              'public_square', 'digital_town_square'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'to_put_it_simply', 'in_plain_english', 'the_point_is',
                  'what_this_means', 'the_implication', 'the_takeaway'],
}

THOUGHT_LEADER_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} argues that the {patient} is {adverb} {adj}",
    "the {adj} {patient} {modal} be {action}ed by the {agent}",
    "the {agent} {action} the {patient} to {action} the {adj} {patient}",
    "however the {agent} contends that the {patient} {modal} be {adj}",
    "the {agent} {action} that the {adj} {patient} {modal} {action} the {patient}",
    "furthermore the {agent} {action} the {adj} {patient} as a {adj} {patient}",
    "the {agent} {action} the {patient} by {action}ing {adj} {patient}",
]


# ── 跨学科大师词汇 (Polymath synthesis) ──────────────────────────

POLYMATH_VOCAB = {
    'agent': ['polymath', 'renaissance_person', 'synthesizer', 'integrator',
              'bridge_builder', 'systems_thinker', 'complexity_scholar',
              'meta_theorist', 'grand_synthesizer', 'paradigm_architect',
              'intellectual_architect', 'knowledge_weaver', 'pattern_recognizer'],
    'action': ['synthesize', 'integrate', 'unify', 'reconcile', 'transcend',
               'reimagine', 'reconceive', 'reconstruct', 'reconfigure', 'recombine',
               'cross_pollinate', 'hybridize', 'emerge', 'crystallize',
               'distill', 'abstract', 'generalize', 'universalize', 'axiomatize'],
    'patient': ['grand_unified_theory', 'meta_framework', 'universal_principle',
                'first_principle', 'axiom', 'postulate', 'theorem',
                'intellectual_tradition', 'knowledge_system', 'worldview',
                'cosmovision', 'episteme', 'paradigm', 'metaparadigm',
                'complexity_theory', 'systems_theory', 'general_theory'],
    'adj': ['unified', 'integrated', 'holistic', 'systems_level', 'meta_level',
            'transcendent', 'universal', 'fundamental', 'axiomatic', 'first_principle',
            'cross_disciplinary', 'multi_paradigmatic', 'complexity_aware',
            'emergent', 'self_organizing', 'adaptive', 'evolutionary'],
    'adverb': ['fundamentally', 'universally', 'axiomatically', 'holistically',
               'systemically', 'emergently', 'adaptively', 'evolutionarily',
               'transcendently', 'integratively', 'synthetically', 'meta_theoretically'],
    'place': ['crossroads', 'nexus', 'convergence_point', 'synthesis_space',
              'intellectual_marketplace', 'knowledge_commons', 'paradigm_workshop',
              'grand_synthesis_lab', 'meta_theory_studio', 'complexity_institute'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'at_a_higher_level', 'from_first_principles', 'in_the_limit',
                  'taken_together', 'synthesized', 'unified_under'],
}

POLYMATH_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} argues that the {patient} {modal} be {action}ed from first principles",
    "the {adj} {patient} represents a {adj} {patient} across {patient}",
    "the {agent} {action} the {patient} by {action}ing the {adj} {patient}",
    "however the {agent} contends that the {patient} {modal} be {adverb} {adj}",
    "the {agent} {action} that the {adj} {patient} {modal} {action} the {patient}",
    "furthermore the {adj} {patient} {modal} be {action}ed {adverb} across {patient}",
    "the {agent} {action} the {patient} to {action} the {adj} {patient}",
]


# ── 宗师词汇 (Grandmaster — legacy & wisdom) ──────────────────────

GRANDMASTER_VOCAB = {
    'agent': ['grandmaster', 'sage', 'luminary', 'elder', 'patriarch',
              'matriarch', 'founding_figure', 'paradigm_creator', 'legacy_builder',
              'wisdom_keeper', 'knowledge_steward', 'intellectual_heritage'],
    'action': ['envision', 'consecrate', 'institutionalize', 'perpetuate',
               'transmit', 'bequeath', 'enshrine', 'codify', 'canonize',
               'immortalize', 'crystallize', 'distill', 'preserve', 'guard',
               'steward', 'nurture', 'cultivate', 'incubate', 'seed', 'germinate'],
    'patient': ['legacy', 'heritage', 'tradition', 'canon', 'oeuvre',
                'intellectual_estate', 'knowledge_inheritance', 'wisdom_tradition',
                'living_tradition', 'school_of_thought', 'lineage', 'genealogy',
                'intellectual_dna', 'cultural_gene', 'meme', 'paradigm_legacy'],
    'adj': ['enduring', 'timeless', 'eternal', 'immortal', 'canonical',
            'foundational', 'seminal', 'definitive', 'authoritative', 'magisterial',
            'visionary', 'prophetic', 'oracular', 'sibylline', 'enigmatic',
            'profound', 'deep', 'abiding', 'resonant', 'reverberating'],
    'adverb': ['enduringly', 'timelessly', 'eternally', 'immortally',
               'canonically', 'fundamentally', 'seminal', 'definitively',
               'authoritatively', 'magisterially', 'visionarily', 'prophetically',
               'profoundly', 'deeply', 'abidingly', 'resonantly'],
    'place': ['pantheon', 'hall_of_fame', 'memorial', 'monument', 'archive',
              'special_collection', 'rare_books_room', 'heritage_site',
              'intellectual_shrine', 'wisdom_temple', 'knowledge_cathedral'],
    'modal': ['could', 'would', 'should', 'might', 'may', 'must', 'shall'],
    'connector': ['however', 'furthermore', 'moreover', 'nevertheless',
                  'consequently', 'accordingly', 'hence', 'thus',
                  'in_the_final_analysis', 'when_all_is_said_and_done',
                  'from_where_i_stand', 'as_i_see_it', 'in_my_estimation'],
}

GRANDMASTER_TEMPLATES = [
    "the {agent} {action} the {adj} {patient}",
    "the {agent} argues that the {patient} {modal} be {action}ed {adverb}",
    "the {adj} {patient} represents the {adj} {patient} of {patient}",
    "the {agent} {action} the {patient} by {action}ing the {adj} {patient}",
    "however the {agent} contends that the {patient} is {adverb} {adj}",
    "the {agent} {action} that the {adj} {patient} {modal} {action} the {patient}",
    "furthermore the {adj} {patient} {modal} be {action}ed as a {adj} {patient}",
    "the {agent} {action} the {patient} to {action} the {adj} {patient} {adverb}",
]


def generate_sentence_from_vocab(vocab: dict, template: str) -> Tuple[str, str]:
    """从词汇库和模板生成句子，返回 (句子, 缺词位置)"""
    sentence = template
    missing_word = None
    missing_pos = None

    for key, words in vocab.items():
        if f'{{{key}}}' in sentence:
            word = random.choice(words)
            if missing_word is None and random.random() < 0.3:
                # 30% 概率挖空
                missing_word = word
                missing_pos = key
                sentence = sentence.replace(f'{{{key}}}', '___', 1)
            else:
                sentence = sentence.replace(f'{{{key}}}', word, 1)

    return sentence, missing_word


def generate_cloze_from_vocab(vocab: dict, template: str) -> Tuple[str, str, List[str]]:
    """生成填空题：句子、正确答案、选项"""
    sentence, missing_word = generate_sentence_from_vocab(vocab, template)
    if missing_word is None:
        # 强制挖一个词
        for key, words in vocab.items():
            if f'{{{key}}}' in sentence:
                missing_word = random.choice(words)
                sentence = sentence.replace(f'{{{key}}}', '___', 1)
                break

    # 生成干扰项
    all_words = [w for words in vocab.values() for w in words]
    distractors = random.sample([w for w in all_words if w != missing_word],
                                min(3, len(all_words) - 1))
    options = [missing_word] + distractors
    random.shuffle(options)

    return sentence, missing_word, options


def generate_sentence_from_vocab_advanced(vocab: dict, templates: list) -> tuple:
    """从词汇库和模板列表生成句子（高级版）

    Returns: (sentence, target_word, target_category, all_fills)
    """
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


def generate_cloze_from_vocab_advanced(vocab: dict, templates: list) -> tuple:
    """生成填空题（高级版）— 从模板列表中选择

    Returns: (blanked_sentence, target_word, options, full_sentence)
    """
    sentence, target_word, target_cat, _ = generate_sentence_from_vocab_advanced(
        vocab, templates)

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
    """生成造句题（高级版）

    Returns: (target_words, available_words)
    """
    _, _, _, words = generate_sentence_from_vocab_advanced(vocab, templates)
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
# 阶段 1: 感知探索（0-2 岁）
# ══════════════════════════════════════════════════════════════════

def stage_perception_exploration(learner: Learner, world: MotivatedWorld,
                                  teacher: NaiveAgent, step: int,
                                  forgetting: ForgettingCurve,
                                  emergent_vocab: EmergentVocabulary):
    """
    阶段 1: 纯物理感知探索

    循环：感知 → 预测 → 学习 → 好奇心驱动探索
    无语言，纯物理直觉建立。
    """
    raw = world.observe()
    obs = learner.perceive(raw)

    # 好奇心驱动的动作选择
    action = learner.choose_action(obs)

    # 执行动作
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)

    # 从经验中学习（预测编码闭环）
    error = learner.learn_from_experience(obs, action, next_obs)

    # 记忆
    learner.remember(obs, action, next_obs, reward, error)

    # 每 50 步巩固记忆
    if step % 50 == 0 and step > 0:
        learner.consolidate()

    # 尝试发现新符号（从感知聚类）
    scene = world.generate_scene_features()
    if scene:
        obj = random.choice(scene)
        # 用一个随机物体的特征来发现符号
        raw_obj = world.observe()
        obs_obj = learner.perceive(raw_obj)
        symbol = emergent_vocab.discover_symbol(learner, obs_obj, obj)
        world.on_symbol_discovered()


# ══════════════════════════════════════════════════════════════════
# 阶段 2: 符号涌现（2-3 岁）
# ══════════════════════════════════════════════════════════════════

def stage_symbol_emergence(learner: Learner, world: MotivatedWorld,
                            teacher: NaiveAgent, step: int,
                            forgetting: ForgettingCurve,
                            emergent_vocab: EmergentVocabulary,
                            dual_protocol: DualAgentProtocol):
    """
    阶段 2: 符号涌现 + 多 Agent 参照游戏

    循环：感知 → 聚类 → 命名 → 与 NaiveAgent 玩参照游戏
    """
    # 感知
    raw = world.observe()
    obs = learner.perceive(raw)

    # 动作选择
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)

    # 学习
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 参照游戏（根据需求决定频率）
    social_need = world.needs['social']
    if random.random() < max(0.3, social_need):
        scene = world.generate_scene_features()
        if scene:
            target_idx = random.randint(0, len(scene) - 1)

            # 发现目标物体的符号
            target = scene[target_idx]
            raw_target = world.observe()
            obs_target = learner.perceive(raw_target)
            symbol = emergent_vocab.discover_symbol(learner, obs_target, target)

            # 暴露符号给双方
            learner.communication.language.expose_symbol(symbol)
            teacher.communication.language.expose_symbol(symbol)

            # 双 Agent 参照游戏（交替 speaker）
            speaker_is_learner = random.random() < 0.5
            success = dual_protocol.play_round(scene, target_idx, speaker_is_learner)

            # 记录结果
            learner.communication_history.append(success)
            learner._update_comm_rate()

            # 遗忘曲线更新
            for val in target.values():
                forgetting.update(str(val), success, step)

            # 需求更新
            world.on_social_interaction()
            if success:
                world.on_communication_success()

    # 应用遗忘衰减
    if step % 10 == 0:
        forgetting.apply_decay(step)

    # 间隔重复：每 20 步复习低 mastery 符号
    if step % 20 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            # 暴露给 learner
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 3: 组合表达（3-4 岁）
# ══════════════════════════════════════════════════════════════════

def stage_compositional(learner: Learner, world: MotivatedWorld,
                         teacher: NaiveAgent, step: int,
                         forgetting: ForgettingCurve,
                         emergent_vocab: EmergentVocabulary,
                         dual_protocol: DualAgentProtocol):
    """
    阶段 3: 组合表达 + 难度脚手架 + 遗忘曲线

    循环：参照游戏 → 填空 → 造句 → 遗忘衰减 → 间隔重复
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 难度脚手架：根据步骤递增
    if step < 100:
        vocab = SIMPLE_VOCAB
        templates = SENTENCE_TEMPLATES['simple']
    elif step < 250:
        vocab = MEDIUM_VOCAB
        templates = SENTENCE_TEMPLATES['medium']
    else:
        vocab = COMPLEX_VOCAB
        templates = SENTENCE_TEMPLATES['complex']

    # 语言活动（交替进行）
    activity = random.choice(['reference', 'cloze', 'sentence'])

    if activity == 'reference':
        # 参照游戏
        scene = world.generate_scene_features()
        if scene:
            target_idx = random.randint(0, len(scene) - 1)
            speaker_is_learner = random.random() < 0.5
            success = dual_protocol.play_round(scene, target_idx, speaker_is_learner)
            learner.communication_history.append(success)
            learner._update_comm_rate()
            world.on_social_interaction()

    elif activity == 'cloze':
        # 填空题
        template = random.choice(templates)
        sentence, answer, options = generate_cloze_from_vocab(vocab, template)
        success = learner.communication.play_cloze_game(sentence, answer, options)
        forgetting.update(answer, success, step)

    elif activity == 'sentence':
        # 造句题
        template = random.choice(templates)
        sentence, _ = generate_sentence_from_vocab(vocab, template)
        # 将句子拆分为词语列表
        target_words = sentence.split()
        # 可用词 = 目标词 + 从词汇库中随机抽取的干扰词
        all_words = [w for words in vocab.values() for w in words]
        distractors = random.sample(all_words, min(5, len(all_words)))
        available_words = list(set(target_words + distractors))
        success = learner.communication.play_sentence_game(target_words, available_words)
        for w in target_words:
            forgetting.update(w, success, step)

    # 遗忘衰减
    if step % 5 == 0:
        forgetting.apply_decay(step)

    # 间隔重复
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 4: 因果推理（4-5 岁）— 认知闭环
# ══════════════════════════════════════════════════════════════════

def stage_causal_reasoning(learner: Learner, world: MotivatedWorld,
                            teacher: NaiveAgent, step: int,
                            forgetting: ForgettingCurve,
                            emergent_vocab: EmergentVocabulary,
                            dual_protocol: DualAgentProtocol):
    """
    阶段 4: 因果推理闭环

    因果观察 → 因果表达 → 因果影响动作选择（闭环！）
    """
    # 记录碰撞前场景
    scene_before = world.generate_scene_features()

    # 感知 + 带因果偏置的动作选择
    raw = world.observe()
    obs = learner.perceive(raw)

    # 修复 4: 因果推理影响动作选择
    causal_rules = learner.causal.get_confident_rules() if hasattr(learner.causal, 'get_confident_rules') else []
    action = choose_action_with_causal_bias(learner, obs, causal_rules)

    # 执行动作
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)

    # 学习
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 碰撞后场景 → 因果观察
    scene_after = world.generate_scene_features()
    events = detect_events(world, scene_before, scene_after)

    for ev in events:
        cause = describe_event_subject(ev)
        effect = describe_event_action(ev)
        learner.causal.observe(cause, effect)

    # 反事实推理（偶尔）
    if events and step % 20 == 0:
        ev = events[0]
        actual_cause = describe_event_subject(ev)
        actual_effect = describe_event_action(ev)
        # 构造反事实
        alt_causes = ['push', 'pull', 'drop', 'throw']
        alt_cause = random.choice(alt_causes)
        cf_world = CounterfactualWorld()
        cf_world.add_action_effect(actual_cause, actual_effect, probability=0.9)
        cf_world.add_action_effect(alt_cause, f"different_{actual_effect}", probability=0.5)
        learner.counterfactual.reason(actual_cause, actual_effect, cf_world)

    # 参照游戏（保持语言学习）
    if random.random() < 0.3:
        scene = world.generate_scene_features()
        if scene:
            target_idx = random.randint(0, len(scene) - 1)
            speaker_is_learner = random.random() < 0.5
            success = dual_protocol.play_round(scene, target_idx, speaker_is_learner)
            learner.communication_history.append(success)
            learner._update_comm_rate()
            world.on_social_interaction()

    # 因果表达：用语言描述因果关系
    if events and step % 10 == 0:
        ev = events[0]
        cause = describe_event_subject(ev)
        effect = describe_event_action(ev)
        # 让 learner 尝试用 "because" 表达因果
        causal_sentence = f"{cause} because {effect}"
        learner.communication.language.expose_symbol('because')

    # 遗忘衰减
    if step % 5 == 0:
        forgetting.apply_decay(step)

    # 间隔重复
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 5: 社会理解（5-6 岁）— ToM 闭环
# ══════════════════════════════════════════════════════════════════

def stage_social_understanding(learner: Learner, world: MotivatedWorld,
                                teacher: NaiveAgent, step: int,
                                forgetting: ForgettingCurve,
                                emergent_vocab: EmergentVocabulary,
                                dual_protocol: DualAgentProtocol):
    """
    阶段 5: 社会理解闭环

    信息不对称 → 视角建模 → 语用修饰 → ToM 影响语言产出（闭环！）
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 社会互动
    scene = world.generate_scene_features()
    if scene and random.random() < 0.5:
        target_idx = random.randint(0, len(scene) - 1)

        # 修复 4: TheoryOfMind 闭环
        # 信息不对称：teacher 只看到部分物体
        visible_to_teacher = random.sample(scene, max(1, len(scene) // 2))
        teacher_facts = [f"{o.get('color', '')}_{o.get('shape', '')}" for o in visible_to_teacher]

        # learner 建模 teacher 的视角
        learner.theory_of_mind.model_other('teacher', teacher_facts)

        # 检测信息不对称（检查 learner 知道但 teacher 不知道的事实）
        hidden_facts = [f"{o.get('color', '')}_{o.get('shape', '')}"
                        for o in scene if o not in visible_to_teacher]
        asymmetry = False
        for fact in hidden_facts[:1]:  # 检查一个隐藏事实
            asymmetry = learner.theory_of_mind.detect_asymmetry('teacher', fact)
            if asymmetry:
                break

        # 根据 teacher 视角调整描述
        if asymmetry:
            # learner 知道 teacher 不知道某些东西 → 调整描述
            target_features = scene[target_idx]
            adjusted = learner.theory_of_mind.adjust_description(
                target_features, scene, 'teacher'
            )

        # 选择视角标记（基于确信度）
        certainty = 0.7 if asymmetry else 0.9
        perspective_marker = learner.theory_of_mind.choose_perspective_marker(
            certainty, language_experience=step
        )
        if perspective_marker:
            learner.communication.language.expose_symbol(perspective_marker)

        # 参照游戏
        speaker_is_learner = random.random() < 0.5
        success = dual_protocol.play_round(scene, target_idx, speaker_is_learner)
        learner.communication_history.append(success)
        learner._update_comm_rate()
        world.on_social_interaction()
        if success:
            world.on_communication_success()

        # 遗忘曲线更新
        for val in scene[target_idx].values():
            forgetting.update(str(val), success, step)

    # 隐喻追踪
    if step % 20 == 0:
        symbols = learner.get_vocabulary()
        symbol_list = list(symbols.keys()) if isinstance(symbols, dict) else []
        if scene and symbol_list:
            learner.metaphor_tracker.record(symbol_list, scene[0] if scene else {})

    # 因果推理（延续阶段 4）
    scene_before = world.generate_scene_features()
    if step % 5 == 0:
        world.physics.step(dt=0.1)
    scene_after = world.generate_scene_features()
    events = detect_events(world, scene_before, scene_after)
    for ev in events:
        learner.causal.observe(describe_event_subject(ev), describe_event_action(ev))

    # 遗忘衰减
    if step % 5 == 0:
        forgetting.apply_decay(step)

    # 间隔重复
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 6: 初中（12-15 岁）— 语法深化 + 叙事 + 隐喻萌芽
# ══════════════════════════════════════════════════════════════════

def stage_middle_school(learner: Learner, world: MotivatedWorld,
                        teacher: NaiveAgent, step: int,
                        forgetting: ForgettingCurve,
                        emergent_vocab: EmergentVocabulary,
                        dual_protocol: DualAgentProtocol):
    """
    阶段 6: 初中 — 语法深化 + 叙事连接词 + 隐喻萌芽

    核心转变：从简单句到复合句，从单事件到多事件叙事，
    开始理解隐喻性语言。认知模块全面参与闭环。
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 35% + 造句 30% + 叙事 35%
    activity = random.random()

    if activity < 0.35:
        # 填空题 — 初中词汇
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            MIDDLE_SCHOOL_VOCAB, MIDDLE_SCHOOL_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.65:
        # 造句题 — 复合句
        target_words, available = generate_sentence_exercise_from_vocab(
            MIDDLE_SCHOOL_VOCAB, MIDDLE_SCHOOL_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    else:
        # 叙事理解 — 多事件 + 连接词
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        if symbols:
            # 参照游戏测试叙事理解
            scene = scene_before if scene_before else [{'color': 'red'}]
            target = random.randint(0, len(scene) - 1)
            success = dual_protocol.play_round(scene, target,
                                                random.random() < 0.5)
            learner.communication_history.append(success)
            learner._update_comm_rate()
            world.on_social_interaction()

            # 记录隐喻
            learner.metaphor_tracker.record(symbols, {'features': scene})

        # 记录叙事连接词
        for sym in symbols:
            forgetting.update(sym, True, step)

    # 因果推理延续
    if step % 10 == 0:
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.1)
        scene_after = world.generate_scene_features()
        events = detect_events(world, scene_before, scene_after)
        for ev in events:
            learner.causal.observe(describe_event_subject(ev),
                                   describe_event_action(ev))

    # 涌现符号发现
    if step % 20 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            sym = emergent_vocab.discover_symbol(learner, obs_obj, obj)

    # 遗忘衰减 + 间隔重复
    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 7: 高中（15-18 岁）— 批判性思维 + 论证 + 类比推理
# ══════════════════════════════════════════════════════════════════

def stage_high_school(learner: Learner, world: MotivatedWorld,
                      teacher: NaiveAgent, step: int,
                      forgetting: ForgettingCurve,
                      emergent_vocab: EmergentVocabulary,
                      dual_protocol: DualAgentProtocol):
    """
    阶段 7: 高中 — 批判性阅读 + 论证写作 + 类比推理

    核心转变：从理解到批判，从描述到论证，
    开始使用类比和隐喻进行抽象思维。
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 30% + 造句 25% + 辩论 45%
    activity = random.random()

    if activity < 0.30:
        # 填空题 — 高中词汇
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            HIGH_SCHOOL_VOCAB, HIGH_SCHOOL_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.55:
        # 造句题 — 论证性句子
        target_words, available = generate_sentence_exercise_from_vocab(
            HIGH_SCHOOL_VOCAB, HIGH_SCHOOL_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    else:
        # 辩论/论证 — DEBATE + NEGOTIATION + MORAL
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        context = random.choice([SocialContext.DEBATE, SocialContext.NEGOTIATION,
                                  SocialContext.MORAL, SocialContext.EMPATHY])

        # 内在言语规划
        try:
            planned = learner.inner_speech.plan_description(scene, target)
        except Exception:
            planned = []

        # 语用修饰
        target_features = scene[target] if isinstance(scene[target], dict) else {}
        utterance = planned if planned else list(target_features.values())[:3]
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        # 参照游戏
        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()
        world.on_social_interaction()

        # 记录隐喻
        symbols = list(target_features.values())
        learner.metaphor_tracker.record(symbols, {'features': scene})

        # 反事实推理
        cf_world = CounterfactualWorld()
        for alt in ['argue', 'agree', 'compromise', 'ignore']:
            cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
        actual_action = random.choice(['argue', 'agree'])
        learner.counterfactual.reason(actual_action, f"result_{actual_action}", cf_world)

        # 遗忘曲线更新
        for val in target_features.values():
            forgetting.update(str(val), success, step)

    # 因果推理延续
    if step % 10 == 0:
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.1)
        scene_after = world.generate_scene_features()
        events = detect_events(world, scene_before, scene_after)
        for ev in events:
            learner.causal.observe(describe_event_subject(ev),
                                   describe_event_action(ev))

    # ToM 延续
    if step % 15 == 0:
        scene = world.generate_scene_features()
        if scene:
            visible = random.sample(scene, max(1, len(scene) // 2))
            teacher_facts = [f"{o.get('color', '')}_{o.get('shape', '')}" for o in visible]
            learner.theory_of_mind.model_other('teacher', teacher_facts)

    # 涌现符号
    if step % 20 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    # 遗忘衰减 + 间隔重复
    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 8: 大学（18-22 岁）— 学术英语 + 语用精通 + 跨模态
# ══════════════════════════════════════════════════════════════════

def stage_university(learner: Learner, world: MotivatedWorld,
                     teacher: NaiveAgent, step: int,
                     forgetting: ForgettingCurve,
                     emergent_vocab: EmergentVocabulary,
                     dual_protocol: DualAgentProtocol):
    """
    阶段 8: 大学 — 学术写作 + 专业沟通 + 跨模态理解

    核心转变：从日常语言到学术语言，掌握 13 种语用情境，
    内在言语规划成为习惯。
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 25% + 造句 20% + 对话 30% + 听力 25%
    activity = random.random()

    if activity < 0.25:
        # 填空题 — 学术词汇
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            UNIVERSITY_VOCAB, UNIVERSITY_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.45:
        # 造句题 — 学术写作
        target_words, available = generate_sentence_exercise_from_vocab(
            UNIVERSITY_VOCAB, UNIVERSITY_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    elif activity < 0.75:
        # 对话语用 — 全部 13 种情境
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        context = random.choice(list(SocialContext))

        # 内在言语规划
        try:
            planned = learner.inner_speech.plan_description(scene, target)
        except Exception:
            planned = []

        target_features = scene[target] if isinstance(scene[target], dict) else {}
        utterance = planned if planned else list(target_features.values())[:3]
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()
        world.on_social_interaction()

    else:
        # 叙事 + 听力理解
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        if symbols:
            narrative_text = ' '.join(symbols)
            options = [narrative_text]
            for _ in range(2):
                world.physics.step(dt=0.05)
                d_before = world.generate_scene_features()
                world.physics.step(dt=0.1)
                d_after = world.generate_scene_features()
                d_events = detect_events(world, d_before, d_after)
                try:
                    d_narr = learner.narrative.build_narrative(d_events)
                    d_text = ' '.join(d_narr.to_symbols())
                except (AttributeError, TypeError):
                    d_text = ''
                if d_text and d_text != narrative_text:
                    options.append(d_text)

            if len(options) >= 2:
                random.shuffle(options)
                success = learner.communication.play_listening_game(
                    narrative_text, options)
                learner.listening_history.append(1.0 if success else 0.0)

    # 因果推理 + 反事实（延续）
    if step % 10 == 0:
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.1)
        scene_after = world.generate_scene_features()
        events = detect_events(world, scene_before, scene_after)
        for ev in events:
            learner.causal.observe(describe_event_subject(ev),
                                   describe_event_action(ev))

    # ToM 延续
    if step % 15 == 0:
        scene = world.generate_scene_features()
        if scene:
            visible = random.sample(scene, max(1, len(scene) // 2))
            teacher_facts = [f"{o.get('color', '')}_{o.get('shape', '')}" for o in visible]
            learner.theory_of_mind.model_other('teacher', teacher_facts)

    # 隐喻追踪
    if step % 20 == 0:
        symbols_list = list(learner.get_vocabulary().keys()) if isinstance(
            learner.get_vocabulary(), dict) else []
        scene = world.generate_scene_features()
        if symbols_list and scene:
            learner.metaphor_tracker.record(symbols_list, scene[0])

    # 涌现符号
    if step % 25 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    # 遗忘衰减 + 间隔重复
    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 9: 硕士（22-24 岁）— 研究写作 + 元认知 + 假设验证
# ══════════════════════════════════════════════════════════════════

def stage_masters(learner: Learner, world: MotivatedWorld,
                  teacher: NaiveAgent, step: int,
                  forgetting: ForgettingCurve,
                  emergent_vocab: EmergentVocabulary,
                  dual_protocol: DualAgentProtocol):
    """
    阶段 9: 硕士 — 研究方法论 + 元认知自评估 + 假设-验证循环

    核心转变：从学习知识到生产知识，从被动接受到主动质疑，
    元认知能力使学习者能自我监控和调整学习策略。
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 25% + 造句 20% + 研究叙事 55%
    activity = random.random()

    if activity < 0.25:
        # 填空题 — 硕士级学术词汇
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            MASTER_VOCAB, MASTER_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.45:
        # 造句题 — 研究方法论表达
        target_words, available = generate_sentence_exercise_from_vocab(
            MASTER_VOCAB, MASTER_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    else:
        # 研究叙事 — 假设 → 实验 → 结论
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        # 元认知自评估 — 闭环！
        try:
            eval_result = learner.metacognition.self_evaluate()
            plan = learner.metacognition.plan_next_learning()
        except Exception:
            pass

        # 因果推理
        if events:
            for ev in events:
                learner.causal.observe(describe_event_subject(ev),
                                       describe_event_action(ev))

        # 参照游戏测试
        if scene_before:
            target = random.randint(0, len(scene_before) - 1)
            success = dual_protocol.play_round(scene_before, target,
                                                random.random() < 0.5)
            learner.communication_history.append(success)
            learner._update_comm_rate()

        # 隐喻记录
        if symbols:
            learner.metaphor_tracker.record(symbols, {'features': scene_before})

    # ToM 延续
    if step % 15 == 0:
        scene = world.generate_scene_features()
        if scene:
            visible = random.sample(scene, max(1, len(scene) // 2))
            teacher_facts = [f"{o.get('color', '')}_{o.get('shape', '')}" for o in visible]
            learner.theory_of_mind.model_other('teacher', teacher_facts)

    # 涌现符号
    if step % 20 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    # 遗忘衰减 + 间隔重复
    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 10: 博士（24-28 岁）— 原创贡献 + 同行评审 + 教学
# ══════════════════════════════════════════════════════════════════

def stage_phd(learner: Learner, world: MotivatedWorld,
              teacher: NaiveAgent, step: int,
              forgetting: ForgettingCurve,
              emergent_vocab: EmergentVocabulary,
              dual_protocol: DualAgentProtocol):
    """
    阶段 10: 博士 — 原创研究 + 同行评审 + 教学相长

    核心转变：从学习者到创造者，从消费者到生产者，
    能够独立提出新理论、评审他人工作、教学相长。
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 20% + 造句 15% + 教学 25% + 评审 20% + 元认知 20%
    activity = random.random()

    if activity < 0.20:
        # 填空题 — 博士级专业词汇
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            PHD_VOCAB, PHD_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.35:
        # 造句题 — 原创性表达
        target_words, available = generate_sentence_exercise_from_vocab(
            PHD_VOCAB, PHD_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    elif activity < 0.60:
        # 教学场景 — 向他人解释概念（ToM 闭环）
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        target_features = scene[target] if isinstance(scene[target], dict) else {}

        # 教学者视角：假设听众只知道部分信息
        visible_count = max(1, len(scene) // 3)
        visible_to_student = random.sample(scene, visible_count)
        student_facts = set()
        for obj in visible_to_student:
            for val in obj.values():
                if val:
                    student_facts.add(str(val))
        learner.theory_of_mind.model_other('student', student_facts)

        # 调整描述以适应学生水平
        adjusted = learner.theory_of_mind.adjust_description(
            target_features, scene, 'student')
        utterance = adjusted if adjusted else list(target_features.values())[:3]

        # 教学对话
        try:
            ctx = derive_social_context(world, target, SocialContext.EMPATHY)
            learner.pragmatics.apply_context(SocialContext.EMPATHY, utterance, ctx)
        except Exception:
            pass

        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()
        world.on_social_interaction()

    elif activity < 0.80:
        # 同行评审 — 批判性评估
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        context = random.choice([SocialContext.DEBATE, SocialContext.MORAL,
                                  SocialContext.REPAIR])

        # 内在言语规划评审意见
        try:
            planned = learner.inner_speech.plan_description(scene, target)
        except Exception:
            planned = []

        target_features = scene[target] if isinstance(scene[target], dict) else {}
        utterance = planned if planned else list(target_features.values())[:3]
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()

    else:
        # 元认知 + 策略优化 + 原创研究叙事
        try:
            eval_result = learner.metacognition.self_evaluate()
            plan = learner.metacognition.plan_next_learning()
        except Exception:
            pass

        # 原创研究叙事
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        # 因果 + 反事实
        if events:
            for ev in events:
                learner.causal.observe(describe_event_subject(ev),
                                       describe_event_action(ev))

        cf_world = CounterfactualWorld()
        for alt in ['propose', 'challenge', 'synthesize', 'reject']:
            cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
        actual_action = random.choice(['propose', 'challenge'])
        learner.counterfactual.reason(actual_action, f"result_{actual_action}", cf_world)

        # 隐喻
        if symbols:
            learner.metaphor_tracker.record(symbols, {'features': scene_before})

    # 涌现符号
    if step % 20 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    # 遗忘衰减 + 间隔重复
    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    # 巩固
    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 11: 博后（28-30 岁）— 跨域研究 + 转化 + 协作
# ══════════════════════════════════════════════════════════════════

def stage_postdoc(learner: Learner, world: MotivatedWorld,
                  teacher: NaiveAgent, step: int,
                  forgetting: ForgettingCurve,
                  emergent_vocab: EmergentVocabulary,
                  dual_protocol: DualAgentProtocol):
    """
    阶段 11: 博后 — 跨域研究 + 转化应用 + 协作网络

    核心转变：从单一领域深耕到跨域迁移，从个人研究到协作网络，
    学会将基础研究转化为实际应用。
    """
    # 感知 + 学习
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 20% + 造句 15% + 跨域叙事 35% + 协作对话 30%
    activity = random.random()

    if activity < 0.20:
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            POSTDOC_VOCAB, POSTDOC_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.35:
        target_words, available = generate_sentence_exercise_from_vocab(
            POSTDOC_VOCAB, POSTDOC_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    elif activity < 0.70:
        # 跨域叙事 — 从不同物理配置中发现共通模式
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        # 元认知 — 跨域反思
        try:
            eval_result = learner.metacognition.self_evaluate()
            plan = learner.metacognition.plan_next_learning()
        except Exception:
            pass

        # 因果 + 反事实 — 假设迁移
        if events:
            for ev in events:
                learner.causal.observe(describe_event_subject(ev),
                                       describe_event_action(ev))

        cf_world = CounterfactualWorld()
        for alt in ['transfer', 'adapt', 'replicate', 'reject']:
            cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
        actual = random.choice(['transfer', 'adapt'])
        learner.counterfactual.reason(actual, f"result_{actual}", cf_world)

        if symbols:
            learner.metaphor_tracker.record(symbols, {'features': scene_before})

    else:
        # 协作对话 — 多视角协商
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        context = random.choice([SocialContext.NEGOTIATION, SocialContext.DEBATE,
                                  SocialContext.EMPATHY, SocialContext.REPAIR])

        try:
            planned = learner.inner_speech.plan_description(scene, target)
        except Exception:
            planned = []

        target_features = scene[target] if isinstance(scene[target], dict) else {}
        utterance = planned if planned else list(target_features.values())[:3]
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        # ToM — 理解协作者视角
        visible = random.sample(scene, max(1, len(scene) // 2))
        collaborator_facts = [f"{o.get('color', '')}_{o.get('shape', '')}" for o in visible]
        learner.theory_of_mind.model_other('collaborator', collaborator_facts)

        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()

    # 涌现符号
    if step % 20 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    # 遗忘衰减 + 间隔重复
    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 12: 行业领袖（30-35 岁）— 战略沟通 + 领导力
# ══════════════════════════════════════════════════════════════════

def stage_industry_leader(learner: Learner, world: MotivatedWorld,
                          teacher: NaiveAgent, step: int,
                          forgetting: ForgettingCurve,
                          emergent_vocab: EmergentVocabulary,
                          dual_protocol: DualAgentProtocol):
    """
    阶段 12: 行业领袖 — 战略沟通 + 组织思维 + 影响力

    核心转变：从研究者到领导者，从学术语言到商业语言，
    学会在复杂利益相关者网络中沟通和影响。
    """
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 20% + 造句 15% + 战略叙事 30% + 领导力对话 35%
    activity = random.random()

    if activity < 0.20:
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            INDUSTRY_VOCAB, INDUSTRY_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.35:
        target_words, available = generate_sentence_exercise_from_vocab(
            INDUSTRY_VOCAB, INDUSTRY_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    elif activity < 0.65:
        # 战略叙事 — 从物理场景中提取战略洞察
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        # 元认知 — 战略反思
        try:
            learner.metacognition.self_evaluate()
            learner.metacognition.plan_next_learning()
        except Exception:
            pass

        if symbols:
            learner.metaphor_tracker.record(symbols, {'features': scene_before})

    else:
        # 领导力对话 — 多利益相关者沟通
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)

        # 不同利益相关者视角
        stakeholder = random.choice(['investor', 'customer', 'employee', 'board_member'])
        visible = random.sample(scene, max(1, len(scene) // 2))
        stakeholder_facts = [f"{o.get('color', '')}_{o.get('shape', '')}" for o in visible]
        learner.theory_of_mind.model_other(stakeholder, stakeholder_facts)

        context = random.choice([SocialContext.NEGOTIATION, SocialContext.DEBATE,
                                  SocialContext.POLITENESS, SocialContext.EMPATHY])
        try:
            planned = learner.inner_speech.plan_description(scene, target)
        except Exception:
            planned = []

        target_features = scene[target] if isinstance(scene[target], dict) else {}
        utterance = planned if planned else list(target_features.values())[:3]
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()

    # 因果推理 — 商业因果
    if step % 10 == 0:
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.1)
        scene_after = world.generate_scene_features()
        events = detect_events(world, scene_before, scene_after)
        for ev in events:
            learner.causal.observe(describe_event_subject(ev),
                                   describe_event_action(ev))

    # 涌现符号
    if step % 20 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 13: 思想领袖（35-40 岁）— 公共话语 + 影响力
# ══════════════════════════════════════════════════════════════════

def stage_thought_leader(learner: Learner, world: MotivatedWorld,
                         teacher: NaiveAgent, step: int,
                         forgetting: ForgettingCurve,
                         emergent_vocab: EmergentVocabulary,
                         dual_protocol: DualAgentProtocol):
    """
    阶段 13: 思想领袖 — 公共话语 + 叙事构建 + 社会影响

    核心转变：从专业沟通到公共表达，从技术语言到大众语言，
    学会构建引人入胜的叙事来影响公共话语。
    """
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 15% + 造句 10% + 公共叙事 40% + 影响力对话 35%
    activity = random.random()

    if activity < 0.15:
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            THOUGHT_LEADER_VOCAB, THOUGHT_LEADER_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.25:
        target_words, available = generate_sentence_exercise_from_vocab(
            THOUGHT_LEADER_VOCAB, THOUGHT_LEADER_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    elif activity < 0.65:
        # 公共叙事 — 构建引人入胜的多事件叙事
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        # 叙事理解测试 — 听力
        if symbols:
            narrative_text = ' '.join(symbols)
            options = [narrative_text]
            for _ in range(2):
                world.physics.step(dt=0.05)
                d_before = world.generate_scene_features()
                world.physics.step(dt=0.1)
                d_after = world.generate_scene_features()
                d_events = detect_events(world, d_before, d_after)
                try:
                    d_narr = learner.narrative.build_narrative(d_events)
                    d_text = ' '.join(d_narr.to_symbols())
                except (AttributeError, TypeError):
                    d_text = ''
                if d_text and d_text != narrative_text:
                    options.append(d_text)

            if len(options) >= 2:
                random.shuffle(options)
                success = learner.communication.play_listening_game(
                    narrative_text, options)
                learner.listening_history.append(1.0 if success else 0.0)

        # 隐喻 — 公共话语中大量使用隐喻
        if symbols:
            learner.metaphor_tracker.record(symbols, {'features': scene_before})

    else:
        # 影响力对话 — 公共演讲 + 辩论
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        context = random.choice([SocialContext.DEBATE, SocialContext.MORAL,
                                  SocialContext.EMPATHY, SocialContext.POLITENESS])

        try:
            planned = learner.inner_speech.plan_description(scene, target)
        except Exception:
            planned = []

        target_features = scene[target] if isinstance(scene[target], dict) else {}
        utterance = planned if planned else list(target_features.values())[:3]
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        # ToM — 理解听众视角
        visible = random.sample(scene, max(1, len(scene) // 2))
        audience_facts = [f"{o.get('color', '')}_{o.get('shape', '')}" for o in visible]
        learner.theory_of_mind.model_other('audience', audience_facts)

        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()

    # 元认知 — 反思影响力
    if step % 20 == 0:
        try:
            learner.metacognition.self_evaluate()
            learner.metacognition.plan_next_learning()
        except Exception:
            pass

    # 涌现符号
    if step % 25 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 14: 跨学科大师（40-50 岁）— 综合 + 系统思维
# ══════════════════════════════════════════════════════════════════

def stage_polymath(learner: Learner, world: MotivatedWorld,
                   teacher: NaiveAgent, step: int,
                   forgetting: ForgettingCurve,
                   emergent_vocab: EmergentVocabulary,
                   dual_protocol: DualAgentProtocol):
    """
    阶段 14: 跨学科大师 — 跨域综合 + 系统思维 + 范式创造

    核心转变：从领域专家到跨学科综合者，从应用知识到创造范式，
    能够在不同知识体系之间发现深层联系。
    """
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 15% + 造句 10% + 综合叙事 40% + 范式对话 35%
    activity = random.random()

    if activity < 0.15:
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            POLYMATH_VOCAB, POLYMATH_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.25:
        target_words, available = generate_sentence_exercise_from_vocab(
            POLYMATH_VOCAB, POLYMATH_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    elif activity < 0.65:
        # 综合叙事 — 从多个物理场景中发现统一模式
        # 场景 A
        scene_a_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs_a = learner.perceive(world.observe())
        action_a = learner.choose_action(obs_a)
        world.physics.step(dt=0.2)
        scene_a_after = world.generate_scene_features()

        # 场景 B（不同配置）
        world.physics.step(dt=0.3)
        scene_b_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs_b = learner.perceive(world.observe())
        action_b = learner.choose_action(obs_b)
        world.physics.step(dt=0.2)
        scene_b_after = world.generate_scene_features()

        # 从两个场景中发现共通因果模式
        events_a = detect_events(world, scene_a_before, scene_a_after)
        events_b = detect_events(world, scene_b_before, scene_b_after)

        for ev in events_a + events_b:
            learner.causal.observe(describe_event_subject(ev),
                                   describe_event_action(ev))

        # 叙事综合
        all_events = events_a + events_b
        try:
            narr = learner.narrative.build_narrative(all_events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        # 反事实 — 如果换一种范式会怎样
        cf_world = CounterfactualWorld()
        for alt in ['synthesize', 'unify', 'transcend', 'reconcile']:
            cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
        actual = random.choice(['synthesize', 'unify'])
        learner.counterfactual.reason(actual, f"result_{actual}", cf_world)

        if symbols:
            learner.metaphor_tracker.record(symbols, {'features': scene_a_before})

        # 元认知 — 范式级反思
        try:
            learner.metacognition.self_evaluate()
            learner.metacognition.plan_next_learning()
        except Exception:
            pass

    else:
        # 范式对话 — 教学 + 跨域讨论
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        target_features = scene[target] if isinstance(scene[target], dict) else {}

        # 教学视角 — 向不同领域的人解释
        listener = random.choice(['artist', 'engineer', 'philosopher', 'student'])
        visible = random.sample(scene, max(1, len(scene) // 3))
        listener_facts = set()
        for obj in visible:
            for val in obj.values():
                if val:
                    listener_facts.add(str(val))
        learner.theory_of_mind.model_other(listener, listener_facts)

        adjusted = learner.theory_of_mind.adjust_description(
            target_features, scene, listener)
        utterance = adjusted if adjusted else list(target_features.values())[:3]

        context = random.choice([SocialContext.EMPATHY, SocialContext.REPAIR,
                                  SocialContext.MORAL])
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        success = dual_protocol.play_round(scene, target, random.random() < 0.5)
        learner.communication_history.append(success)
        learner._update_comm_rate()

    # 涌现符号 — 大量涌现
    if step % 15 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 阶段 15: 宗师（50+ 岁）— 传承 + 智慧 + 遗产
# ══════════════════════════════════════════════════════════════════

def stage_grandmaster(learner: Learner, world: MotivatedWorld,
                      teacher: NaiveAgent, step: int,
                      forgetting: ForgettingCurve,
                      emergent_vocab: EmergentVocabulary,
                      dual_protocol: DualAgentProtocol):
    """
    阶段 15: 宗师 — 知识传承 + 智慧凝练 + 永恒遗产

    核心转变：从创造者到传承者，从个人成就到集体遗产，
    将毕生所学凝练为可传承的智慧体系。
    """
    raw = world.observe()
    obs = learner.perceive(raw)
    action = learner.choose_action(obs)
    obs_next, reward, done = world.step(action_to_tensor(action, learner.config.action_dim))
    next_obs = learner.perceive(obs_next)
    error = learner.learn_from_experience(obs, action, next_obs)
    learner.remember(obs, action, next_obs, reward, error)

    # 活动分布：填空 15% + 造句 10% + 智慧叙事 35% + 传承对话 25% + 元认知 15%
    activity = random.random()

    if activity < 0.15:
        blanked, target, options, _ = generate_cloze_from_vocab_advanced(
            GRANDMASTER_VOCAB, GRANDMASTER_TEMPLATES)
        success = learner.communication.play_cloze_game(blanked, target, options)
        learner.reading_history.append(1.0 if success else 0.0)
        forgetting.update(target, success, step)

    elif activity < 0.25:
        target_words, available = generate_sentence_exercise_from_vocab(
            GRANDMASTER_VOCAB, GRANDMASTER_TEMPLATES)
        success = learner.communication.play_sentence_game(target_words, available)
        learner.writing_history.append(1.0 if success else 0.0)
        for w in target_words:
            forgetting.update(w, success, step)

    elif activity < 0.60:
        # 智慧叙事 — 从毕生经验中凝练洞察
        scene_before = world.generate_scene_features()
        world.physics.step(dt=0.15)
        obs2 = learner.perceive(world.observe())
        action2 = learner.choose_action(obs2)
        world.physics.step(dt=0.2)
        scene_after = world.generate_scene_features()

        events = detect_events(world, scene_before, scene_after)
        try:
            narr = learner.narrative.build_narrative(events)
            symbols = narr.to_symbols()
        except (AttributeError, TypeError):
            symbols = []

        # 因果 + 反事实 + 隐喻 — 全部认知模块参与
        if events:
            for ev in events:
                learner.causal.observe(describe_event_subject(ev),
                                       describe_event_action(ev))

        cf_world = CounterfactualWorld()
        for alt in ['envision', 'bequeath', 'codify', 'transcend']:
            cf_world.add_action_effect(alt, f"result_{alt}", 0.5)
        actual = random.choice(['envision', 'bequeath'])
        learner.counterfactual.reason(actual, f"result_{actual}", cf_world)

        if symbols:
            learner.metaphor_tracker.record(symbols, {'features': scene_before})

    elif activity < 0.85:
        # 传承对话 — 向后辈传授智慧
        scene = world.generate_scene_features()
        if not scene:
            return
        target = random.randint(0, len(scene) - 1)
        target_features = scene[target] if isinstance(scene[target], dict) else {}

        # 后辈视角 — 知道得更少
        visible_count = max(1, len(scene) // 4)
        visible_to_student = random.sample(scene, visible_count)
        student_facts = set()
        for obj in visible_to_student:
            for val in obj.values():
                if val:
                    student_facts.add(str(val))
        learner.theory_of_mind.model_other('mentee', student_facts)

        adjusted = learner.theory_of_mind.adjust_description(
            target_features, scene, 'mentee')
        utterance = adjusted if adjusted else list(target_features.values())[:3]

        context = SocialContext.EMPATHY
        try:
            ctx = derive_social_context(world, target, context)
            learner.pragmatics.apply_context(context, utterance, ctx)
        except Exception:
            pass

        success = dual_protocol.play_round(scene, target, True)  # 宗师总是 speaker
        learner.communication_history.append(success)
        learner._update_comm_rate()

        # 听力测试 — 理解后辈的反馈
        if random.random() < 0.5:
            feedback_text = ' '.join(list(target_features.values())[:3])
            distractors = []
            for other_scene in [world.generate_scene_features()]:
                if other_scene:
                    other = random.choice(other_scene)
                    distractors.append(' '.join(list(other.values())[:3]))
            if distractors:
                options = [feedback_text] + distractors
                random.shuffle(options)
                success = learner.communication.play_listening_game(
                    feedback_text, options)
                learner.listening_history.append(1.0 if success else 0.0)

    else:
        # 元认知 — 终身学习反思
        try:
            eval_result = learner.metacognition.self_evaluate()
            plan = learner.metacognition.plan_next_learning()
        except Exception:
            pass

        # 内在言语 — 默默反思
        scene = world.generate_scene_features()
        if scene:
            target = random.randint(0, len(scene) - 1)
            try:
                learner.inner_speech.plan_description(scene, target)
            except Exception:
                pass

    # 涌现符号 — 持续发现
    if step % 15 == 0:
        scene = world.generate_scene_features()
        if scene:
            obj = random.choice(scene)
            raw_obj = world.observe()
            obs_obj = learner.perceive(raw_obj)
            emergent_vocab.discover_symbol(learner, obs_obj, obj)

    if step % 5 == 0:
        forgetting.apply_decay(step)
    if step % 15 == 0:
        review_symbols = forgetting.get_review_symbols(3)
        for sym in review_symbols:
            learner.communication.language.expose_symbol(sym)

    if step % 100 == 0 and step > 0:
        learner.consolidate()


# ══════════════════════════════════════════════════════════════════
# 评估辅助
# ══════════════════════════════════════════════════════════════════

def meets_criteria(metrics: dict, criteria: dict) -> bool:
    """检查是否满足所有退出条件"""
    for key, threshold in criteria.items():
        if metrics.get(key, 0.0) < threshold:
            return False
    return True


def print_metrics(metrics: dict, prefix: str = "  "):
    """打印关键指标"""
    keys = ['prediction_accuracy', 'vocabulary_size', 'communication_success',
            'composition_rate', 'grammar_complexity',
            'reading_accuracy', 'writing_accuracy']
    parts = []
    for k in keys:
        v = metrics.get(k, 0.0)
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    print(prefix + " | ".join(parts))


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("闭环学习系统 — 像小孩一样学英语")
    print("=" * 60)

    # 初始化
    config = LearnerConfig()
    learner = Learner(config)
    world = MotivatedWorld(config)
    teacher = NaiveAgent()
    evaluator = CapabilityEvaluator()

    # 闭环组件
    forgetting = ForgettingCurve(decay_rate=0.05, review_interval=100)
    emergent_vocab = EmergentVocabulary()
    dual_protocol = DualAgentProtocol(learner.communication, teacher.communication)

    # 阶段配置（10 阶段，自适应退出）
    stages = [
        ("感知探索 (0-2岁)", stage_perception_exploration,
         {'prediction_accuracy': 0.7}, 'sensorimotor'),
        ("符号涌现 (2-3岁)", stage_symbol_emergence,
         {'vocabulary_size': 15.0, 'communication_success': 0.5}, 'early_preoperational'),
        ("组合表达 (3-4岁)", stage_compositional,
         {'composition_rate': 0.3}, 'late_preoperational'),
        ("因果推理 (4-5岁)", stage_causal_reasoning,
         {}, 'early_concrete'),
        ("社会理解 (5-6岁)", stage_social_understanding,
         {}, 'late_concrete'),
        ("初中 (12-15岁)", stage_middle_school,
         {'vocabulary_size': 80.0}, 'early_formal'),
        ("高中 (15-18岁)", stage_high_school,
         {'vocabulary_size': 120.0}, 'early_formal'),
        ("大学 (18-22岁)", stage_university,
         {'reading_accuracy': 0.6, 'writing_accuracy': 0.6}, 'late_formal'),
        ("硕士 (22-24岁)", stage_masters,
         {'grammar_complexity': 0.5}, 'late_formal'),
        ("博士 (24-28岁)", stage_phd,
         {'grammar_complexity': 0.7}, 'late_formal'),
        ("博后 (28-30岁)", stage_postdoc,
         {'grammar_complexity': 0.8, 'writing_accuracy': 0.7}, 'late_formal'),
        ("行业领袖 (30-35岁)", stage_industry_leader,
         {'writing_accuracy': 0.8, 'listening_accuracy': 0.7}, 'late_formal'),
        ("思想领袖 (35-40岁)", stage_thought_leader,
         {'writing_accuracy': 0.85}, 'late_formal'),
        ("跨学科大师 (40-50岁)", stage_polymath,
         {'reading_accuracy': 0.85, 'writing_accuracy': 0.85}, 'late_formal'),
        ("宗师 (50+岁)", stage_grandmaster,
         {'grammar_complexity': 0.9, 'writing_accuracy': 0.9}, 'late_formal'),
    ]

    results = {}

    for stage_idx, (name, stage_fn, criteria, world_stage) in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"阶段 {stage_idx+1}: {name}")
        print(f"{'='*60}")

        # 配置世界
        world.configure_for_stage(world_stage)

        max_rounds = 500
        stage_start = time.time()
        achieved = False

        for round_num in range(max_rounds):
            # 调用阶段函数
            if stage_idx == 0:
                stage_fn(learner, world, teacher, round_num,
                         forgetting, emergent_vocab)
            else:
                stage_fn(learner, world, teacher, round_num,
                         forgetting, emergent_vocab, dual_protocol)

            # 每 50 步评估
            if round_num % 50 == 0 and round_num > 0:
                metrics = evaluator.evaluate(learner)
                print(f"  [{round_num:4d}] ", end="")
                print_metrics(metrics)

                # 检查退出条件
                if criteria and meets_criteria(metrics, criteria):
                    elapsed = time.time() - stage_start
                    print(f"  >>> 达标于第 {round_num} 轮 ({elapsed:.1f}s)")
                    achieved = True
                    break

        if not achieved:
            elapsed = time.time() - stage_start
            print(f"  >>> 完成 {max_rounds} 轮 ({elapsed:.1f}s)")

        # 最终评估
        metrics = evaluator.evaluate(learner)
        print(f"  最终指标: ", end="")
        print_metrics(metrics)

        # 额外指标
        causal_rules = len(learner.causal.rules) if hasattr(learner.causal, 'rules') else 0
        if hasattr(learner.causal, 'get_confident_rules'):
            causal_rules = len(learner.causal.get_confident_rules())
        print(f"  因果规则: {causal_rules}")
        print(f"  涌现符号: {emergent_vocab.get_symbol_count()}")
        print(f"  遗忘队列: {forgetting.get_low_mastery_count()} 个低掌握度符号")

        # 隐喻检测
        metaphors = learner.metaphor_tracker.detect_metaphors()
        print(f"  涌现隐喻: {len(metaphors)} 种映射")

        results[name] = metrics

        # 保存检查点
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'stage_{stage_idx+1}.pt')
        learner.save(ckpt_path)
        print(f"  检查点: {ckpt_path}")

    # 保存结果
    results_path = os.path.join(RESULTS_DIR, 'childlike_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {results_path}")

    # 最终总结
    print(f"\n{'='*60}")
    print("最终总结")
    print(f"{'='*60}")
    final_metrics = evaluator.evaluate(learner)
    print_metrics(final_metrics)
    print(f"涌现符号: {emergent_vocab.get_symbol_count()}")
    print(f"遗忘队列: {forgetting.get_low_mastery_count()} 个低掌握度符号")


if __name__ == '__main__':
    main()
