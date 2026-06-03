"""
语用学模块 — 13 种高级社会语言现象

整合辩论、礼貌、幽默、道德、共情、谈判、方言、暗语、
修复、所有权、社会规范、量化、空间等语用情境的涌现追踪。
纯 PyTorch 实现，零 numpy 依赖。
"""

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

from src.core.device import get_device


# =====================================================================
# 社会语用情境枚举
# =====================================================================

class SocialContext(Enum):
    """13 种社会语用情境"""
    DEBATE = 'debate'                # 辩论
    POLITENESS = 'politeness'        # 礼貌
    HUMOR = 'humor'                  # 幽默
    MORAL = 'moral'                  # 道德
    EMPATHY = 'empathy'              # 共情
    NEGOTIATION = 'negotiation'      # 谈判
    DIALECT = 'dialect'              # 方言
    CRYPTOLECT = 'cryptolect'        # 暗语
    REPAIR = 'repair'                # 修复
    OWNERSHIP = 'ownership'          # 所有权
    SOCIAL_NORMS = 'norms'           # 社会规范
    QUANTITATIVE = 'quantitative'    # 量化
    SPATIAL = 'spatial'              # 空间


# =====================================================================
# 每种情境对应的标记定义
# =====================================================================

CONTEXT_MARKERS: Dict[SocialContext, Dict] = {
    SocialContext.DEBATE: {
        'markers': ['because', 'but', 'wrong', 'agree', 'disagree', 'reason'],
        'pressure': 'conflict',
        'description': '辩论中涌现因果和反驳标记',
    },
    SocialContext.POLITENESS: {
        'markers': ['please', 'sorry', 'excuse', 'thank', 'welcome'],
        'pressure': 'social_distance',
        'description': '礼貌标记随社会距离涌现',
    },
    SocialContext.HUMOR: {
        'markers': ['play', 'funny', 'joke', 'surprise', 'repeat'],
        'pressure': 'non_instrumental',
        'description': '幽默标记在非工具性交流中涌现',
    },
    SocialContext.MORAL: {
        'markers': ['fair', 'unfair', 'share', 'greedy', 'good', 'bad'],
        'pressure': 'resource_conflict',
        'description': '道德语言在资源冲突中涌现',
    },
    SocialContext.EMPATHY: {
        'markers': ['feel', 'sad', 'happy', 'understand', 'same'],
        'pressure': 'perspective_taking',
        'description': '共情标记在视角采择中涌现',
    },
    SocialContext.NEGOTIATION: {
        'markers': ['offer', 'accept', 'reject', 'compromise', 'deal'],
        'pressure': 'conflicting_goals',
        'description': '谈判标记在目标冲突中涌现',
    },
    SocialContext.DIALECT: {
        'markers': [],
        'pressure': 'geographic_isolation',
        'description': '方言标记在地理隔离中涌现',
    },
    SocialContext.CRYPTOLECT: {
        'markers': [],
        'pressure': 'in_group_secrecy',
        'description': '暗语在群体内保密需求中涌现',
    },
    SocialContext.REPAIR: {
        'markers': ['what', 'again', 'clarify', 'meaning', 'not_understand'],
        'pressure': 'communication_failure',
        'description': '修复标记在通信失败中涌现',
    },
    SocialContext.OWNERSHIP: {
        'markers': ['mine', 'yours', 'ours', 'give', 'take'],
        'pressure': 'resource_competition',
        'description': '所有权概念在资源竞争中涌现',
    },
    SocialContext.SOCIAL_NORMS: {
        'markers': ['should', 'must', 'not_allowed', 'rule', 'always'],
        'pressure': 'group_coordination',
        'description': '社会规范语言在群体协调中涌现',
    },
    SocialContext.QUANTITATIVE: {
        'markers': ['one', 'two', 'three', 'more', 'less', 'same_count'],
        'pressure': 'counting_pressure',
        'description': '量化语言在计数压力中涌现',
    },
    SocialContext.SPATIAL: {
        'markers': ['left', 'right', 'above', 'below', 'near', 'far'],
        'pressure': 'spatial_ambiguity',
        'description': '空间语言在空间歧义中涌现',
    },
}


# =====================================================================
# 辅助：基于 torch 的简单嵌入查找
# =====================================================================

def _token_to_index(token: str, vocab: Dict[str, int]) -> int:
    """将词元映射到词表索引，未知返回 0"""
    return vocab.get(token, 0)


def _bag_of_markers(utterance: List[str], markers: List[str],
                    vocab: Dict[str, int], device: torch.device) -> torch.Tensor:
    """
    构建标记的词袋向量。

    返回 shape (len(markers),) 的浮点张量，
    第 i 位为 1.0 当且仅当 utterance 中出现了 markers[i]。
    """
    if not markers:
        return torch.zeros(0, device=device)
    vec = torch.zeros(len(markers), device=device)
    marker_set = set(markers)
    for i, m in enumerate(markers):
        if m in utterance:
            vec[i] = 1.0
    return vec


# =====================================================================
# PragmaticsModule 主类
# =====================================================================

class PragmaticsModule:
    """
    语用学模块 — 13 种高级社会语言现象

    职责：
    1. 在特定社会情境中施加语用压力，驱动标记涌现
    2. 追踪每个情境下标记的使用频率
    3. 检测标记是否已在共享词表中涌现
    4. 为每种情境生成对应的场景
    """

    def __init__(self, vocabulary: Optional[Dict] = None,
                 knowledge_graph=None, device: str = 'auto'):
        """
        初始化语用学模块。

        Args:
            vocabulary: 共享词表 dict，key 为词元字符串
            knowledge_graph: 可选的知识图谱引用
            device: 'auto' | 'cpu' | 'cuda'
        """
        self.vocabulary: Dict = vocabulary or {}
        self.knowledge = knowledge_graph
        self.device = get_device(device)

        # 每个情境下每个标记的使用次数
        self._marker_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # 情境交互历史（最近 200 条）
        self._context_history: List[Dict] = []

        # 每个情境的涌现计数器（用于归一化）
        self._context_total: Dict[str, int] = defaultdict(int)

        # 方言 / 暗语的动态标记池
        self._dynamic_markers: Dict[str, List[str]] = {
            SocialContext.DIALECT.value: [],
            SocialContext.CRYPTOLECT.value: [],
        }

    # ------------------------------------------------------------------
    # 核心方法：施加语用压力
    # ------------------------------------------------------------------

    def apply_context(self, context: SocialContext,
                      utterance: List[str], scene: Dict) -> Dict:
        """
        将社会情境约束施加到话语上。

        检查话语是否包含情境相关标记，计算匹配度，
        并根据语用压力类型决定是否需要对话语进行修改。

        Args:
            context: 社会情境枚举
            utterance: 当前话语（词元列表）
            scene: 场景描述字典

        Returns:
            {
                'utterance': List[str],        # 可能被修改的话语
                'matched': List[str],          # 命中的标记
                'match_ratio': float,          # 命中比例
                'pressure_applied': str,       # 施加的压力类型
                'modifications': List[str],    # 修改记录
            }
        """
        meta = CONTEXT_MARKERS[context]
        markers = meta['markers']
        pressure = meta['pressure']

        # 动态标记（方言/暗语）从自身池中取
        if context in (SocialContext.DIALECT, SocialContext.CRYPTOLECT):
            markers = self._dynamic_markers.get(context.value, [])

        # 计算匹配
        marker_set = set(markers)
        matched = [w for w in utterance if w in marker_set]
        match_ratio = len(matched) / max(len(markers), 1)

        modifications: List[str] = []
        modified = list(utterance)

        # 根据压力类型施加修改
        if pressure == 'conflict':
            # 辩论：若缺少因果标记且场景有冲突，追加 'because'
            if scene.get('has_conflict') and 'because' not in modified:
                modified.append('because')
                modifications.append('added_because_for_conflict')

        elif pressure == 'social_distance':
            # 礼貌：若场景社会距离高且无礼貌标记，追加 'please'
            if scene.get('social_distance', 0) > 0.7 and 'please' not in modified:
                modified.append('please')
                modifications.append('added_please_for_distance')

        elif pressure == 'non_instrumental':
            # 幽默：若场景标记为游戏且无幽默标记，追加 'play'
            if scene.get('is_play') and 'play' not in modified:
                modified.append('play')
                modifications.append('added_play_marker')

        elif pressure == 'resource_conflict':
            # 道德：若资源分布不均且无道德标记，追加 'fair' 或 'unfair'
            dist = scene.get('resource_distribution')
            if dist is not None:
                diff = abs(dist.get('agent_a', 0) - dist.get('agent_b', 0))
                if diff > 2 and 'fair' not in modified and 'unfair' not in modified:
                    marker = 'unfair' if diff > 4 else 'fair'
                    modified.append(marker)
                    modifications.append(f'added_{marker}_for_resource_conflict')

        elif pressure == 'perspective_taking':
            # 共情：若场景有视角冲突，追加 'understand'
            if scene.get('different_perspective') and 'understand' not in modified:
                modified.append('understand')
                modifications.append('added_understand_for_empathy')

        elif pressure == 'conflicting_goals':
            # 谈判：若双方目标冲突，追加 'offer'
            if scene.get('goal_conflict') and 'offer' not in modified:
                modified.append('offer')
                modifications.append('added_offer_for_negotiation')

        elif pressure == 'communication_failure':
            # 修复：若理解分数低，追加 'clarify'
            if scene.get('understanding_score', 1.0) < 0.3 and 'clarify' not in modified:
                modified.append('clarify')
                modifications.append('added_clarify_for_repair')

        elif pressure == 'resource_competition':
            # 所有权：若场景有竞争资源，追加 'mine'
            if scene.get('contested_resource') and 'mine' not in modified:
                modified.append('mine')
                modifications.append('added_mine_for_ownership')

        elif pressure == 'group_coordination':
            # 社会规范：若场景有协调需求，追加 'should'
            if scene.get('coordination_needed') and 'should' not in modified:
                modified.append('should')
                modifications.append('added_should_for_norms')

        elif pressure == 'counting_pressure':
            # 量化：若场景需计数，追加数字标记
            if scene.get('needs_counting') and 'more' not in modified:
                modified.append('more')
                modifications.append('added_more_for_quantitative')

        elif pressure == 'spatial_ambiguity':
            # 空间：若场景有空间歧义，追加方位标记
            if scene.get('spatial_ambiguous') and 'left' not in modified:
                modified.append('left')
                modifications.append('added_spatial_marker')

        elif pressure == 'geographic_isolation':
            # 方言：随机从动态池中选一个标记
            if self._dynamic_markers[SocialContext.DIALECT.value]:
                import random
                dialect_m = random.choice(self._dynamic_markers[SocialContext.DIALECT.value])
                if dialect_m not in modified:
                    modified.append(dialect_m)
                    modifications.append(f'added_dialect_{dialect_m}')

        elif pressure == 'in_group_secrecy':
            # 暗语：从动态池中选一个标记
            if self._dynamic_markers[SocialContext.CRYPTOLECT.value]:
                import random
                crypto_m = random.choice(self._dynamic_markers[SocialContext.CRYPTOLECT.value])
                if crypto_m not in modified:
                    modified.append(crypto_m)
                    modifications.append(f'added_cryptolect_{crypto_m}')

        # 记录使用
        for m in matched:
            self._marker_usage[context.value][m] += 1
        self._context_total[context.value] += 1

        # 写入历史
        record = {
            'context': context.value,
            'pressure': pressure,
            'matched': matched,
            'match_ratio': match_ratio,
            'modifications': modifications,
        }
        self._context_history.append(record)
        if len(self._context_history) > 200:
            self._context_history = self._context_history[-200:]

        return {
            'utterance': modified,
            'matched': matched,
            'match_ratio': match_ratio,
            'pressure_applied': pressure,
            'modifications': modifications,
        }

    # ------------------------------------------------------------------
    # 标记涌现检测
    # ------------------------------------------------------------------

    def check_marker_emergence(self, context: SocialContext,
                                vocabulary: Optional[Dict] = None) -> Dict[str, float]:
        """
        检测给定情境中哪些标记已经在词表中涌现。

        Args:
            context: 社会情境
            vocabulary: 可选覆盖词表，默认用 self.vocabulary

        Returns:
            marker -> emergence_score (0.0 ~ 1.0)。
            涌现分数 = 使用频率归一化值，若标记也在词表中则再加 0.3。
        """
        vocab = vocabulary or self.vocabulary
        markers = CONTEXT_MARKERS[context]['markers']

        # 动态标记补充
        if context in (SocialContext.DIALECT, SocialContext.CRYPTOLECT):
            markers = self._dynamic_markers.get(context.value, [])

        total = max(self._context_total.get(context.value, 1), 1)
        result: Dict[str, float] = {}

        for m in markers:
            usage = self._marker_usage[context.value].get(m, 0)
            freq = usage / total
            # 基础分数来自使用频率
            score = min(freq, 1.0)
            # 若标记也在词表中，额外加分
            if m in vocab:
                score = min(score + 0.3, 1.0)
            result[m] = round(score, 4)

        return result

    def get_emergent_markers(self) -> Dict[str, List[str]]:
        """
        返回所有情境中已涌现的标记。

        涌现阈值: emergence_score >= 0.15。

        Returns:
            context_name -> [已涌现标记列表]
        """
        threshold = 0.15
        emergent: Dict[str, List[str]] = {}
        for ctx in SocialContext:
            scores = self.check_marker_emergence(ctx)
            emerged = [m for m, s in scores.items() if s >= threshold]
            if emerged:
                emergent[ctx.value] = emerged
        return emergent

    # ------------------------------------------------------------------
    # 使用记录
    # ------------------------------------------------------------------

    def record_usage(self, context: SocialContext,
                     markers_used: List[str]) -> None:
        """
        记录某情境下使用了哪些标记。

        Args:
            context: 社会情境
            markers_used: 本次使用的标记列表
        """
        for m in markers_used:
            self._marker_usage[context.value][m] += 1
        self._context_total[context.value] += 1

    # ------------------------------------------------------------------
    # 场景生成 — 每种情境生成不同的场景结构
    # ------------------------------------------------------------------

    def generate_social_scene(self, context: SocialContext,
                               num_objects: int = 4) -> Tuple[List[Dict], int]:
        """
        根据社会情境生成对应的场景。

        每种情境产生不同的场景结构：
        - DEBATE: 冲突属性的对象
        - POLITENESS: 不同社会距离等级
        - HUMOR: 游戏性/非工具性特征
        - MORAL: 可共享 vs 竞争性资源
        - EMPATHY: 视角差异场景
        - NEGOTIATION: 目标冲突配置
        - DIALECT: 地理区域标记
        - CRYPTOLECT: 群体内秘密信号
        - REPAIR: 通信障碍配置
        - OWNERSHIP: 可声明所有权的资源
        - SOCIAL_NORMS: 群体协调规则场景
        - QUANTITATIVE: 需要计数的集合
        - SPATIAL: 空间排列场景

        Args:
            context: 社会情境
            num_objects: 场景中对象数量

        Returns:
            (scene_features, target_idx) — 场景特征列表与目标对象索引
        """
        n = max(num_objects, 2)
        target = n // 2  # 默认目标索引

        if context == SocialContext.DEBATE:
            scene = self._scene_debate(n)
        elif context == SocialContext.POLITENESS:
            scene = self._scene_politeness(n)
        elif context == SocialContext.HUMOR:
            scene = self._scene_humor(n)
        elif context == SocialContext.MORAL:
            scene = self._scene_moral(n)
        elif context == SocialContext.EMPATHY:
            scene = self._scene_empathy(n)
        elif context == SocialContext.NEGOTIATION:
            scene = self._scene_negotiation(n)
        elif context == SocialContext.DIALECT:
            scene = self._scene_dialect(n)
        elif context == SocialContext.CRYPTOLECT:
            scene = self._scene_cryptolect(n)
        elif context == SocialContext.REPAIR:
            scene = self._scene_repair(n)
        elif context == SocialContext.OWNERSHIP:
            scene = self._scene_ownership(n)
        elif context == SocialContext.SOCIAL_NORMS:
            scene = self._scene_social_norms(n)
        elif context == SocialContext.QUANTITATIVE:
            scene = self._scene_quantitative(n)
        elif context == SocialContext.SPATIAL:
            scene = self._scene_spatial(n)
        else:
            # 通用回退场景
            scene = self._scene_generic(n)

        return scene, target

    # ----- 各情境场景生成器 -----

    def _scene_debate(self, n: int) -> List[Dict]:
        """
        辩论场景：对象具有冲突属性（颜色/形状/用途不一致），
        需要用 because/but 等标记进行推理和反驳。
        """
        props = ['red', 'blue', 'round', 'square', 'useful', 'useless']
        scene = []
        for i in range(n):
            # 前半部分对象有一组属性，后半有冲突属性
            if i < n // 2:
                obj = {
                    'id': i,
                    'color': 'red',
                    'shape': 'round',
                    'useful': True,
                    'has_conflict': False,
                }
            else:
                obj = {
                    'id': i,
                    'color': 'blue',
                    'shape': 'square',
                    'useful': False,
                    'has_conflict': True,
                }
            scene.append(obj)
        return scene

    def _scene_politeness(self, n: int) -> List[Dict]:
        """
        礼貌场景：对象附带不同社会距离等级，
        距离越高越需要礼貌标记。
        """
        distances = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        scene = []
        for i in range(n):
            scene.append({
                'id': i,
                'social_distance': distances[i % len(distances)],
                'is_stranger': distances[i % len(distances)] > 0.5,
                'request_type': 'resource',
            })
        return scene

    def _scene_humor(self, n: int) -> List[Dict]:
        """
        幽默场景：对象带有游戏性和非工具性标记，
        如意外属性、重复模式、反常组合。
        """
        scene = []
        for i in range(n):
            scene.append({
                'id': i,
                'is_play': True,
                'has_surprise': (i % 3 == 0),
                'is_repeated': (i % 2 == 0),
                'instrumental_value': 0.0,
                'funny_factor': torch.rand(1, device=self.device).item(),
            })
        return scene

    def _scene_moral(self, n: int) -> List[Dict]:
        """
        道德场景：资源分布不均，涉及公平/不公平判断，
        以及共享与贪婪的选择。
        """
        scene = []
        for i in range(n):
            a_share = torch.randint(0, 10, (1,), device=self.device).item()
            b_share = torch.randint(0, 10, (1,), device=self.device).item()
            scene.append({
                'id': i,
                'resource_distribution': {'agent_a': a_share, 'agent_b': b_share},
                'can_share': True,
                'is_greedy': a_share > 2 * max(b_share, 1),
                'fairness': round(1.0 - abs(a_share - b_share) / 10.0, 2),
            })
        return scene

    def _scene_empathy(self, n: int) -> List[Dict]:
        """
        共情场景：对象呈现不同情绪状态和视角，
        要求 Agent 采择他人视角。
        """
        emotions = ['sad', 'happy', 'angry', 'fearful', 'neutral']
        scene = []
        for i in range(n):
            scene.append({
                'id': i,
                'different_perspective': True,
                'observed_emotion': emotions[i % len(emotions)],
                'needs_comfort': emotions[i % len(emotions)] in ('sad', 'fearful'),
                'self_state': 'neutral',
            })
        return scene

    def _scene_negotiation(self, n: int) -> List[Dict]:
        """
        谈判场景：对象具有双方不同估值和目标冲突，
        需要用 offer/accept/reject/compromise 标记协商。
        """
        scene = []
        for i in range(n):
            val_a = torch.randint(1, 10, (1,), device=self.device).item()
            val_b = torch.randint(1, 10, (1,), device=self.device).item()
            scene.append({
                'id': i,
                'goal_conflict': val_a != val_b,
                'value_agent_a': val_a,
                'value_agent_b': val_b,
                'is_compromisable': abs(val_a - val_b) <= 3,
            })
        return scene

    def _scene_dialect(self, n: int) -> List[Dict]:
        """
        方言场景：对象来自不同地理区域，
        每个区域有独特的变体标记。
        """
        regions = ['north', 'south', 'east', 'west']
        region_markers = {
            'north': 'northern_variant',
            'south': 'southern_variant',
            'east': 'eastern_variant',
            'west': 'western_variant',
        }
        scene = []
        for i in range(n):
            region = regions[i % len(regions)]
            # 动态注册方言标记
            dialect_m = region_markers[region]
            if dialect_m not in self._dynamic_markers[SocialContext.DIALECT.value]:
                self._dynamic_markers[SocialContext.DIALECT.value].append(dialect_m)
            scene.append({
                'id': i,
                'region': region,
                'dialect_marker': dialect_m,
                'isolation_level': torch.rand(1, device=self.device).item(),
            })
        return scene

    def _scene_cryptolect(self, n: int) -> List[Dict]:
        """
        暗语场景：对象分为群体内/外，需要秘密标记通信。
        """
        in_group_markers = ['secret_a', 'secret_b', 'secret_c']
        # 动态注册暗语标记
        for m in in_group_markers:
            if m not in self._dynamic_markers[SocialContext.CRYPTOLECT.value]:
                self._dynamic_markers[SocialContext.CRYPTOLECT.value].append(m)

        scene = []
        for i in range(n):
            is_in = i % 2 == 0
            scene.append({
                'id': i,
                'is_in_group': is_in,
                'crypto_marker': in_group_markers[i % len(in_group_markers)] if is_in else None,
                'needs_secrecy': is_in,
                'out_group_present': not is_in,
            })
        return scene

    def _scene_repair(self, n: int) -> List[Dict]:
        """
        修复场景：通信质量递降，需要修复标记来纠正理解。
        """
        scene = []
        for i in range(n):
            noise_level = torch.rand(1, device=self.device).item()
            scene.append({
                'id': i,
                'understanding_score': round(1.0 - noise_level, 2),
                'communication_noise': round(noise_level, 2),
                'needs_repair': noise_level > 0.5,
                'repair_type': 'clarification',
            })
        return scene

    def _scene_ownership(self, n: int) -> List[Dict]:
        """
        所有权场景：资源可被声明所有权，存在竞争关系。
        """
        scene = []
        for i in range(n):
            scene.append({
                'id': i,
                'contested_resource': torch.rand(1, device=self.device).item() > 0.5,
                'current_owner': 'agent_a' if i % 2 == 0 else 'agent_b',
                'transferable': True,
                'rival_claims': torch.randint(0, 3, (1,), device=self.device).item(),
            })
        return scene

    def _scene_social_norms(self, n: int) -> List[Dict]:
        """
        社会规范场景：群体需要协调一致行为，
        违反规范的场景触发 should/must 标记。
        """
        rules = ['wait_turn', 'share_equally', 'no_push', 'help_other']
        scene = []
        for i in range(n):
            scene.append({
                'id': i,
                'coordination_needed': True,
                'active_rule': rules[i % len(rules)],
                'violation': torch.rand(1, device=self.device).item() > 0.6,
                'group_size': torch.randint(3, 8, (1,), device=self.device).item(),
            })
        return scene

    def _scene_quantitative(self, n: int) -> List[Dict]:
        """
        量化场景：对象数量不等，需要计数和比较标记。
        """
        scene = []
        for i in range(n):
            count_a = torch.randint(1, 8, (1,), device=self.device).item()
            count_b = torch.randint(1, 8, (1,), device=self.device).item()
            scene.append({
                'id': i,
                'needs_counting': count_a != count_b,
                'count_group_a': count_a,
                'count_group_b': count_b,
                'comparison': 'more' if count_a > count_b else (
                    'less' if count_a < count_b else 'same_count'
                ),
            })
        return scene

    def _scene_spatial(self, n: int) -> List[Dict]:
        """
        空间场景：对象在二维空间中的排列，
        需要用方位标记消除歧义。
        """
        positions = [
            (-1.0, 0.0),   # left
            (1.0, 0.0),    # right
            (0.0, 1.0),    # above
            (0.0, -1.0),   # below
        ]
        scene = []
        for i in range(n):
            px, py = positions[i % len(positions)]
            # 加一点噪声
            px += torch.rand(1, device=self.device).item() * 0.1 - 0.05
            py += torch.rand(1, device=self.device).item() * 0.1 - 0.05
            scene.append({
                'id': i,
                'x': round(px, 3),
                'y': round(py, 3),
                'spatial_ambiguous': abs(px) < 0.3 or abs(py) < 0.3,
                'relation_to_center': (
                    'left' if px < -0.2 else 'right' if px > 0.2 else
                    'above' if py > 0.2 else 'below' if py < -0.2 else 'near'
                ),
            })
        return scene

    def _scene_generic(self, n: int) -> List[Dict]:
        """通用回退场景：简单对象列表"""
        scene = []
        for i in range(n):
            scene.append({
                'id': i,
                'features': torch.rand(10, device=self.device),
            })
        return scene

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_context_stats(self) -> Dict:
        """
        返回所有情境的统计数据。

        Returns:
            {
                context_name: {
                    'total_interactions': int,
                    'marker_usage': {marker: count},
                    'top_markers': [str],
                    'emergence_scores': {marker: float},
                }
            }
        """
        stats: Dict = {}
        for ctx in SocialContext:
            name = ctx.value
            total = self._context_total.get(name, 0)
            usage = dict(self._marker_usage.get(name, {}))
            scores = self.check_marker_emergence(ctx)

            # 取使用次数前 3 的标记
            sorted_markers = sorted(
                usage.items(), key=lambda x: x[1], reverse=True
            )[:3]
            top = [m for m, _ in sorted_markers]

            stats[name] = {
                'total_interactions': total,
                'marker_usage': usage,
                'top_markers': top,
                'emergence_scores': scores,
            }
        return stats

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """
        将模块状态序列化为字典。

        Returns:
            包含所有内部状态的字典，可被 load_state 恢复。
        """
        marker_usage_plain: Dict[str, Dict[str, int]] = {}
        for ctx_name, inner in self._marker_usage.items():
            marker_usage_plain[ctx_name] = dict(inner)

        context_total_plain = dict(self._context_total)

        return {
            'marker_usage': marker_usage_plain,
            'context_total': context_total_plain,
            'context_history': list(self._context_history),
            'dynamic_markers': dict(self._dynamic_markers),
            'vocabulary': dict(self.vocabulary),
        }

    def load_state(self, state: dict) -> None:
        """
        从字典恢复模块状态。

        Args:
            state: save_state 返回的字典
        """
        self._marker_usage = defaultdict(lambda: defaultdict(int))
        for ctx_name, inner in state.get('marker_usage', {}).items():
            for marker, count in inner.items():
                self._marker_usage[ctx_name][marker] = count

        self._context_total = defaultdict(int)
        for ctx_name, total in state.get('context_total', {}).items():
            self._context_total[ctx_name] = total

        self._context_history = state.get('context_history', [])
        self._dynamic_markers = state.get('dynamic_markers', {
            SocialContext.DIALECT.value: [],
            SocialContext.CRYPTOLECT.value: [],
        })
        self.vocabulary = state.get('vocabulary', {})
