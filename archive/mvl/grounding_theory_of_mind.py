"""
Phase 20: 心智理论模块 —— 理解他人的信念和意图

核心思想：
心智理论 = 建模他人的心理状态
- Agent 能估计他人知道什么（知识状态）
- Agent 能建模他人相信什么（信念状态）
- Agent 能根据他人的知识调整描述（视角切换）

涌现条件：
1. Speaker 和 Listener 有不同的知识（信息不对称）
2. Speaker 必须根据 Listener 的知识调整描述
3. 视角标记词（"know", "think", "believe"）从需要传达认知状态时涌现
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
    PERSPECTIVE_MARKERS,
)


class Perspective:
    """
    视角：Agent 对世界的认知状态

    包含：
    - 已知事实（观察到的）
    - 信念置信度（对命题的信心）
    - 注意焦点（当前关注的内容）
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.known_facts: Set[str] = set()  # 已知事实
        self.beliefs: Dict[str, float] = {}  # 命题 -> 置信度
        self.attention: Set[str] = set()  # 当前注意的内容
        self.observation_history: List[Dict] = []  # 观察历史

    def observe(self, fact: str, confidence: float = 1.0):
        """观察一个事实"""
        self.known_facts.add(fact)
        self.beliefs[fact] = confidence
        self.observation_history.append({
            'fact': fact,
            'confidence': confidence,
            'timestamp': len(self.observation_history),
        })

    def update_belief(self, proposition: str, evidence_strength: float):
        """更新信念置信度"""
        if proposition in self.beliefs:
            # 贝叶斯更新
            prior = self.beliefs[proposition]
            likelihood = evidence_strength
            posterior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood))
            self.beliefs[proposition] = posterior
        else:
            self.beliefs[proposition] = evidence_strength

    def knows(self, fact: str) -> bool:
        """是否知道某个事实"""
        return fact in self.known_facts

    def get_confidence(self, proposition: str) -> float:
        """获取对命题的置信度"""
        return self.beliefs.get(proposition, 0.5)

    def forget(self, fact: str):
        """遗忘一个事实（用于模拟过时信息）"""
        self.known_facts.discard(fact)
        if fact in self.beliefs:
            del self.beliefs[fact]

    def get_stats(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'known_facts': len(self.known_facts),
            'beliefs': len(self.beliefs),
            'attention': len(self.attention),
        }


class TheoryOfMindAgent:
    """
    具有心智理论能力的 Agent

    能力：
    1. 建模他人的知识状态（他人知道什么）
    2. 建模他人的信念状态（他人相信什么）
    3. 根据他人的知识调整描述（视角切换）
    4. 检测分歧（自己和他人的信念不一致）
    """

    def __init__(self, language: EmergingLanguage, agent_id: str):
        self.language = language
        self.agent_id = agent_id
        self.own_perspective = Perspective(agent_id)
        self.model_of_other: Dict[str, Perspective] = {}  # 对他人视角的建模
        self.perspective_adjustments = 0  # 视角切换次数
        self.knowledge_asymmetry_detected = 0  # 检测到知识不对称的次数

    def model_other_agent(self, other_id: str, observed_facts: Set[str]):
        """建模另一个 agent 的知识状态"""
        if other_id not in self.model_of_other:
            self.model_of_other[other_id] = Perspective(other_id)
        other_perspective = self.model_of_other[other_id]
        for fact in observed_facts:
            other_perspective.observe(fact)

    def estimate_other_knowledge(self, other_id: str, fact: str) -> float:
        """估计另一个 agent 是否知道某个事实"""
        if other_id not in self.model_of_other:
            return 0.5  # 未知
        other_perspective = self.model_of_other[other_id]
        if other_perspective.knows(fact):
            return 1.0
        else:
            return 0.0

    def detect_knowledge_asymmetry(self, other_id: str, fact: str) -> bool:
        """检测知识不对称（自己知道但他人不知道）"""
        self_knows = self.own_perspective.knows(fact)
        other_knows = self.estimate_other_knowledge(other_id, fact) > 0.5
        if self_knows and not other_knows:
            self.knowledge_asymmetry_detected += 1
            return True
        return False

    def adjust_description_for_other(self, target_features: Dict[str, str],
                                      scene_features: List[Dict[str, str]],
                                      other_id: str) -> List[str]:
        """
        根据他人的知识调整描述

        如果他人不知道某个特征，就不使用该特征描述
        如果他人有过时信息，使用他人的（过时）信念
        """
        # 获取目标的关键特征
        target_values = set(target_features.values())

        # 过滤掉他人不知道的特征
        known_to_other = []
        for val in target_values:
            if self.estimate_other_knowledge(other_id, val) > 0.5:
                known_to_other.append(val)

        # 如果有他人知道的特征，使用这些特征
        if known_to_other:
            self.perspective_adjustments += 1
            return known_to_other[:2]  # 最多 2 个特征

        # 否则使用默认描述
        return list(target_values)[:2]

    def choose_perspective_marker(self, certainty: float) -> Optional[str]:
        """
        根据确定性选择视角标记

        高确定性 → "know"
        中确定性 → "think"
        低确定性 → "believe"

        只有当语言经验足够时才使用标记（需要先学会标记的通信价值）
        """
        # 需要一定的语言经验才会使用标记
        if self.language.total_games < 10:
            return None

        # 探索-利用：早期随机尝试，后期基于经验选择
        if self.language.total_games < 50:
            # 早期：随机选择标记（探索）
            if np.random.random() < 0.3:
                if certainty > 0.8:
                    return 'know'
                elif certainty > 0.5:
                    return 'think'
                else:
                    return 'believe'
            return None

        # 后期：基于确定性选择标记
        if certainty > 0.8:
            return 'know'
        elif certainty > 0.5:
            return 'think'
        else:
            return 'believe'

    def get_stats(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'perspective_adjustments': self.perspective_adjustments,
            'knowledge_asymmetry_detected': self.knowledge_asymmetry_detected,
            'model_of_other_count': len(self.model_of_other),
        }


class AsymmetricCommunicationGame:
    """
    不对称信息通信游戏

    游戏流程：
    1. 生成场景，部分信息对 Listener 隐藏
    2. Speaker 知道完整信息，Listener 只知道部分
    3. Speaker 必须根据 Listener 的知识调整描述
    4. Listener 根据描述选择目标

    涌现压力：
    - 如果 Speaker 使用 Listener 不知道的特征，Listener 无法理解
    - Speaker 必须建模 Listener 的知识状态
    - 视角标记词（"know", "think", "believe"）从需要传达认知状态时涌现
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = TheoryOfMindAgent(self.language, 'speaker')
        self.listener = TheoryOfMindAgent(self.language, 'listener')
        self.game_log = []
        self.perspective_marker_used = 0
        self.perspective_marker_success = 0
        self.adjustment_used = 0
        self.adjustment_success = 0

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int,
                   hidden_from_listener: Set[str] = None,
                   speaker_confidence: float = None) -> bool:
        """
        进行一轮不对称信息通信游戏

        参数：
            scene_features: 场景特征列表
            target_idx: 目标物体索引
            hidden_from_listener: 对 Listener 隐藏的特征值集合
            speaker_confidence: Speaker 对目标的置信度 (0-1)，用于视角标记选择
        """
        if target_idx >= len(scene_features):
            return False

        if hidden_from_listener is None:
            hidden_from_listener = set()

        target = scene_features[target_idx]

        # 建模 Listener 的知识状态
        for i, obj in enumerate(scene_features):
            for val in obj.values():
                if val not in hidden_from_listener:
                    self.speaker.model_other_agent('listener', {val})

        # Speaker 选择描述策略
        use_adjustment = len(hidden_from_listener) > 0 and np.random.random() < 0.7

        if use_adjustment:
            # 使用视角调整描述
            utterance = self.speaker.adjust_description_for_other(
                target, scene_features, 'listener'
            )
            self.adjustment_used += 1
        else:
            # 使用标准描述
            utterance = list(target.values())[:2]

        if not utterance:
            return False

        # 视角标记选择：根据置信度添加标记
        if speaker_confidence is not None:
            marker = self.speaker.choose_perspective_marker(speaker_confidence)
            if marker:
                utterance = [marker] + utterance
                self.perspective_marker_used += 1

        # Listener 解释
        chosen_idx = self._listener_interpret(utterance, scene_features, hidden_from_listener)
        success = (chosen_idx == target_idx)

        # 更新统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1

        if use_adjustment and success:
            self.adjustment_success += 1

        # 检查是否使用了视角标记且成功
        has_marker = any(s in PERSPECTIVE_MARKERS for s in utterance)
        if has_marker and success:
            self.perspective_marker_success += 1

        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
            'hidden_features': hidden_from_listener,
            'used_adjustment': use_adjustment,
            'speaker_confidence': speaker_confidence,
        })
        return success

    def _listener_interpret(self, utterance: List[str],
                            scene_features: List[Dict[str, str]],
                            hidden_features: Set[str]) -> int:
        """Listener 解释描述（支持视角标记）"""
        # 检测视角标记
        CONFIDENCE_BOOST = 1.5  # "know" 的加分
        CONFIDENCE_MILD = 0.5   # "think" 的加分

        marker = None
        for s in utterance:
            if s in PERSPECTIVE_MARKERS:
                marker = s
                break

        # 过滤掉标记词，只保留描述性符号
        desc_symbols = [s for s in utterance if s not in PERSPECTIVE_MARKERS]

        scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set(obj.values())
            # 过滤掉隐藏的特征
            visible_values = obj_values - hidden_features
            matches = sum(1 for s in desc_symbols if s in visible_values)

            # 视角标记加分：如果描述符号匹配，且有标记，额外加分
            if marker and matches > 0:
                if marker == 'know':
                    matches += CONFIDENCE_BOOST
                elif marker == 'think':
                    matches += CONFIDENCE_MILD

            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['adjustment_used'] = self.adjustment_used
        stats['adjustment_success'] = self.adjustment_success / max(1, self.adjustment_used)
        stats['perspective_marker_used'] = self.perspective_marker_used
        stats['perspective_marker_success'] = self.perspective_marker_success / max(1, self.perspective_marker_used)
        stats['speaker_stats'] = self.speaker.get_stats()
        stats['listener_stats'] = self.listener.get_stats()
        return stats


def generate_tom_scenario(mode: str = 'shared',
                           num_objects: int = 4) -> Tuple[List[Dict[str, str]], int, Set[str]]:
    """
    生成心智理论场景

    参数：
        mode: 场景模式
            - 'shared': 双方看到相同场景（不需要心智理论）
            - 'hidden_object': Speaker 知道 Listener 看不到的物体
            - 'false_belief': Listener 对物体有过时信息
            - 'joint_attention': 双方看到场景但注意不同部分
        num_objects: 物体数量

    返回：
        (scene_features, target_idx, hidden_from_listener)
    """
    colors = list(COLORS)
    shapes = list(SHAPES)
    sizes = list(SIZES)

    np.random.shuffle(colors)
    np.random.shuffle(shapes)
    np.random.shuffle(sizes)

    scene = []
    for i in range(num_objects):
        obj = {
            'color': colors[i % len(colors)],
            'shape': shapes[i % len(shapes)],
            'size': sizes[i % len(sizes)],
        }
        scene.append(obj)

    target_idx = np.random.randint(0, len(scene))
    hidden_from_listener = set()

    if mode == 'shared':
        # 双方看到相同场景
        pass

    elif mode == 'hidden_object':
        # Speaker 知道 Listener 看不到的物体
        # 隐藏目标物体的某个特征
        target = scene[target_idx]
        hidden_feature = np.random.choice(list(target.values()))
        hidden_from_listener.add(hidden_feature)

    elif mode == 'false_belief':
        # Listener 对物体有过时信息
        # 模拟：目标物体改变了颜色，但 Listener 不知道
        old_color = scene[target_idx]['color']
        new_color = np.random.choice([c for c in colors if c != old_color])
        scene[target_idx]['color'] = new_color
        hidden_from_listener.add(new_color)  # Listener 不知道新颜色

    elif mode == 'joint_attention':
        # 双方看到场景但注意不同部分
        # 隐藏部分物体的特征
        for i in range(len(scene)):
            if i != target_idx and np.random.random() < 0.3:
                for val in scene[i].values():
                    hidden_from_listener.add(val)

    return scene, target_idx, hidden_from_listener


def generate_confidence_scenario(mode: str = 'identical_visual',
                                  num_objects: int = 4) -> Tuple[List[Dict[str, str]], int, Set[str], float]:
    """
    生成置信度场景：驱动视角标记涌现

    核心设计：两个物体共享完全相同的视觉特征，
    但 Speaker 对目标的置信度不同。
    "know" = 高置信度，"think" = 中/低置信度。
    Listener 根据标记调整选择策略。

    参数：
        mode: 场景模式
            - 'identical_visual': 两个物体外观相同，Speaker 置信度不同
            - 'shared_knowledge': 双方都知道所有信息（基线）
            - 'asymmetric': Speaker 有额外信息
            - 'mixed': 混合场景
        num_objects: 物体数量

    返回：
        (scene_features, target_idx, hidden_from_listener, speaker_confidence)
    """
    colors = list(COLORS)
    shapes = list(SHAPES)
    sizes = list(SIZES)

    np.random.shuffle(colors)
    np.random.shuffle(shapes)
    np.random.shuffle(sizes)

    if mode == 'identical_visual':
        # 关键场景：Speaker 有时高置信度、有时低置信度
        # 高置信度时描述准确，低置信度时描述可能有噪声
        # Listener 根据标记判断描述的可靠性

        scene = []
        for i in range(num_objects):
            scene.append({
                'color': colors[i % len(colors)],
                'shape': shapes[i % len(shapes)],
                'size': sizes[i % len(sizes)],
            })

        target_idx = np.random.randint(0, len(scene))

        # 随机决定 Speaker 的置信度
        # 高置信度（0.7-1.0）：描述准确
        # 低置信度（0.3-0.6）：描述可能有噪声
        if np.random.random() < 0.5:
            speaker_confidence = np.random.uniform(0.7, 1.0)
        else:
            speaker_confidence = np.random.uniform(0.3, 0.6)

        return scene, target_idx, set(), speaker_confidence

    elif mode == 'shared_knowledge':
        # 基线：双方都知道所有信息，不需要置信度标记
        scene = []
        for i in range(num_objects):
            scene.append({
                'color': colors[i % len(colors)],
                'shape': shapes[i % len(shapes)],
                'size': sizes[i % len(sizes)],
            })

        target_idx = np.random.randint(0, len(scene))
        return scene, target_idx, set(), 1.0  # 高置信度，但不需要标记

    elif mode == 'asymmetric':
        # Speaker 有额外信息
        scene = []
        for i in range(num_objects):
            scene.append({
                'color': colors[i % len(colors)],
                'shape': shapes[i % len(shapes)],
                'size': sizes[i % len(sizes)],
            })

        target_idx = np.random.randint(0, len(scene))
        target = scene[target_idx]
        hidden_feature = np.random.choice(list(target.values()))
        hidden_from_listener = {hidden_feature}

        # Speaker 置信度中等（有额外信息但不完全确定）
        speaker_confidence = np.random.uniform(0.5, 0.8)

        return scene, target_idx, hidden_from_listener, speaker_confidence

    elif mode == 'mixed':
        # 混合场景：有时需要置信度标记，有时不需要
        if np.random.random() < 0.5:
            return generate_confidence_scenario('identical_visual', num_objects)
        else:
            return generate_confidence_scenario('asymmetric', num_objects)

    # 默认
    scene = []
    for i in range(num_objects):
        scene.append({
            'color': colors[i % len(colors)],
            'shape': shapes[i % len(shapes)],
            'size': sizes[i % len(sizes)],
        })
    target_idx = np.random.randint(0, len(scene))
    return scene, target_idx, set(), 1.0


class ConfidenceCommunicationGame:
    """
    置信度通信游戏

    扩展 AsymmetricCommunicationGame，专门用于测试视角标记涌现。
    Speaker 的置信度决定使用哪个标记（know/think/believe）。
    Listener 根据标记调整选择策略。
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = TheoryOfMindAgent(self.language, 'speaker')
        self.listener = TheoryOfMindAgent(self.language, 'listener')
        self.game_log = []
        self.perspective_marker_used = 0
        self.perspective_marker_success = 0
        self.marker_counts = defaultdict(int)  # marker -> 使用次数
        self.marker_success_counts = defaultdict(int)  # marker -> 成功次数

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int,
                   hidden_from_listener: Set[str] = None,
                   speaker_confidence: float = 0.5) -> bool:
        if target_idx >= len(scene_features):
            return False

        if hidden_from_listener is None:
            hidden_from_listener = set()

        target = scene_features[target_idx]

        # 建模 Listener 的知识状态
        for i, obj in enumerate(scene_features):
            for val in obj.values():
                if val not in hidden_from_listener:
                    self.speaker.model_other_agent('listener', {val})

        # 低置信度时，描述可能指向错误的物体（噪声）
        actual_target_idx = target_idx
        if speaker_confidence < 0.6 and np.random.random() > speaker_confidence:
            # 低置信度 + 随机 chance → 描述错误的物体
            wrong_indices = [i for i in range(len(scene_features)) if i != target_idx]
            if wrong_indices:
                actual_target_idx = np.random.choice(wrong_indices)

        actual_target = scene_features[actual_target_idx]

        # 生成描述
        utterance = list(actual_target.values())[:2]

        # 视角标记选择
        marker = self.speaker.choose_perspective_marker(speaker_confidence)
        if marker:
            utterance = [marker] + utterance
            self.perspective_marker_used += 1
            self.marker_counts[marker] += 1

        if not utterance:
            return False

        # Listener 解释
        chosen_idx = self._listener_interpret(utterance, scene_features, hidden_from_listener)
        success = (chosen_idx == target_idx)

        # 更新统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1

        has_marker = any(s in PERSPECTIVE_MARKERS for s in utterance)
        if has_marker and success:
            self.perspective_marker_success += 1
            if marker:
                self.marker_success_counts[marker] += 1

        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
            'speaker_confidence': speaker_confidence,
            'marker': marker,
            'actual_target': actual_target_idx,
        })
        return success

    def _listener_interpret(self, utterance: List[str],
                            scene_features: List[Dict[str, str]],
                            hidden_features: Set[str]) -> int:
        CONFIDENCE_BOOST = 1.5
        CONFIDENCE_MILD = 0.5

        marker = None
        for s in utterance:
            if s in PERSPECTIVE_MARKERS:
                marker = s
                break

        desc_symbols = [s for s in utterance if s not in PERSPECTIVE_MARKERS]

        scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set(obj.values())
            visible_values = obj_values - hidden_features
            matches = sum(1 for s in desc_symbols if s in visible_values)

            # 标记影响匹配策略
            if marker == 'know':
                # 高置信度：严格匹配，加分给最佳匹配
                if matches > 0:
                    matches += CONFIDENCE_BOOST
            elif marker == 'think':
                # 中置信度：宽松匹配，轻微加分
                if matches > 0:
                    matches += CONFIDENCE_MILD
            # 无标记或 "believe"：不加分

            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['perspective_marker_used'] = self.perspective_marker_used
        stats['perspective_marker_success'] = (
            self.perspective_marker_success / max(1, self.perspective_marker_used)
        )
        stats['marker_counts'] = dict(self.marker_counts)
        stats['marker_success_rates'] = {
            m: self.marker_success_counts[m] / max(1, c)
            for m, c in self.marker_counts.items()
        }
        stats['speaker_stats'] = self.speaker.get_stats()
        return stats


class BaselineConfidenceGame:
    """无视角标记的基线游戏（不使用 know/think/believe，但有同样的噪声）"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int,
                   hidden_from_listener: Set[str] = None,
                   speaker_confidence: float = 0.5) -> bool:
        if target_idx >= len(scene_features):
            return False

        if hidden_from_listener is None:
            hidden_from_listener = set()

        # 低置信度时同样有噪声（与 ConfidentCommunicationGame 一致）
        actual_target_idx = target_idx
        if speaker_confidence < 0.6 and np.random.random() > speaker_confidence:
            wrong_indices = [i for i in range(len(scene_features)) if i != target_idx]
            if wrong_indices:
                actual_target_idx = np.random.choice(wrong_indices)

        actual_target = scene_features[actual_target_idx]
        utterance = list(actual_target.values())[:2]

        # Listener 解释（无标记）
        scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set(obj.values())
            visible_values = obj_values - hidden_from_listener
            matches = sum(1 for s in utterance if s in visible_values)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        chosen_idx = scores[0][0]
        success = (chosen_idx == target_idx)

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success

    def get_stats(self) -> Dict:
        return self.language.get_stats()


class BaselineTomGame:
    """无心智理论的基线游戏（不调整描述）"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int,
                   hidden_from_listener: Set[str] = None) -> bool:
        if target_idx >= len(scene_features):
            return False

        if hidden_from_listener is None:
            hidden_from_listener = set()

        target = scene_features[target_idx]

        # 基线：不调整描述，使用目标的所有特征
        utterance = list(target.values())[:2]

        # Listener 解释
        scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set(obj.values())
            visible_values = obj_values - hidden_from_listener
            matches = sum(1 for s in utterance if s in visible_values)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        chosen_idx = scores[0][0]
        success = (chosen_idx == target_idx)

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success

    def get_stats(self) -> Dict:
        return self.language.get_stats()


def test_theory_of_mind():
    """测试心智理论机制"""
    print("=== 心智理论机制测试 ===")

    # 创建 Agent
    language = EmergingLanguage()
    agent = TheoryOfMindAgent(language, 'test_agent')

    # 测试视角建模
    agent.model_other_agent('other', {'red', 'circle'})
    print(f"\n视角建模:")
    print(f"  其他知道 'red': {agent.estimate_other_knowledge('other', 'red')}")
    print(f"  其他知道 'blue': {agent.estimate_other_knowledge('other', 'blue')}")

    # 测试知识不对称检测
    agent.own_perspective.observe('red')
    agent.own_perspective.observe('blue')
    asymmetry = agent.detect_knowledge_asymmetry('other', 'blue')
    print(f"\n知识不对称检测:")
    print(f"  自己知道 'blue', 其他不知道: {asymmetry}")

    # 测试视角调整描述
    scene = [{'color': 'red', 'shape': 'circle'}, {'color': 'blue', 'shape': 'square'}]
    target = scene[0]
    adjusted = agent.adjust_description_for_other(target, scene, 'other')
    print(f"\n视角调整描述:")
    print(f"  目标特征: {target}")
    print(f"  调整后描述: {adjusted}")

    print(f"\nAgent 统计: {agent.get_stats()}")


if __name__ == '__main__':
    test_theory_of_mind()
