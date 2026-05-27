"""
Phase 26: 环境-语言桥梁

将环境探索经验转化为语言交流场景，
让语言从真实的环境交互中涌现。
"""

import random
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounding_unified_language import (
    UnifiedObject, UnifiedScene, VISUAL_FEATURES,
    AUDITORY_FEATURES, TACTILE_FEATURES, AFFORDANCES,
    generate_unified_scenario,
)


# ============================================================
# 环境→语言桥梁
# ============================================================

class EnvironmentLanguageBridge:
    """将环境探索经验转化为语言交流场景"""

    # 推导映射表
    MATERIAL_TEXTURE = {
        'metal': 'hard', 'wood': 'rough', 'plastic': 'smooth',
        'glass': 'smooth', 'fabric': 'soft_tactile', 'stone': 'hard',
    }

    WEIGHT_SOUND = [
        (1.3, 'loud'), (0.7, 'quiet'), (0.0, 'soft'),
    ]

    SHAPE_AFFORDANCE = {
        'square': ['support'], 'circle': ['hit'],
        'triangle': ['cut'],
    }

    MATERIAL_AFFORDANCE = {
        'glass': ['contain'], 'metal': ['contain'],
    }

    def observation_to_scene(self, observation: dict,
                              target_idx: int = None) -> UnifiedScene:
        """将环境观测转换为统一通信场景"""
        visible = observation.get('visible_objects', [])
        if len(visible) < 2:
            return None

        objects = [self._object_to_unified(v['object']) for v in visible]

        if target_idx is None:
            target_idx = random.randint(0, len(objects) - 1)

        ambiguity_types = self._detect_ambiguity(objects, target_idx)

        return UnifiedScene(
            objects=objects,
            target_idx=target_idx,
            ambiguity_types=ambiguity_types,
            speaker_confidence=random.uniform(0.3, 0.95),
        )

    def _object_to_unified(self, obj) -> UnifiedObject:
        """环境物体 → 统一物体"""
        # visual
        visual = {'color': obj.color, 'shape': obj.shape}
        if hasattr(obj, 'material') and obj.material:
            visual['material'] = obj.material
        if hasattr(obj, 'weight'):
            if obj.weight < 1.0:
                visual['size'] = 'small'
            elif obj.weight > 1.2:
                visual['size'] = 'big'
            else:
                visual['size'] = 'medium'

        # auditory
        sound = getattr(obj, 'sound', None) or self._infer_sound(obj)
        auditory = {'sound': sound}

        # tactile
        texture = getattr(obj, 'texture', None) or self._infer_texture(obj)
        tactile = {'texture': texture}

        # affordances
        affordances = getattr(obj, 'affordances', None) or self._infer_affordances(obj)

        return UnifiedObject(
            visual=visual, auditory=auditory,
            tactile=tactile, affordances=affordances,
        )

    def _infer_sound(self, obj) -> str:
        """从 weight 推导声音"""
        w = getattr(obj, 'weight', 1.0)
        for threshold, sound in self.WEIGHT_SOUND:
            if w >= threshold:
                return sound
        return 'soft'

    def _infer_texture(self, obj) -> str:
        """从 material 推导触感"""
        mat = getattr(obj, 'material', None)
        if mat and mat in self.MATERIAL_TEXTURE:
            return self.MATERIAL_TEXTURE[mat]
        w = getattr(obj, 'weight', 1.0)
        return 'hard' if w > 1.0 else 'soft_tactile'

    def _infer_affordances(self, obj) -> List[str]:
        """从 shape + material 推导功能"""
        affs = ['reach']
        shape = getattr(obj, 'shape', None)
        mat = getattr(obj, 'material', None)

        if shape in self.SHAPE_AFFORDANCE:
            affs.extend(self.SHAPE_AFFORDANCE[shape])
        if mat in self.MATERIAL_AFFORDANCE:
            affs.extend(self.MATERIAL_AFFORDANCE[mat])

        return affs

    def _detect_ambiguity(self, objects: List[UnifiedObject],
                           target_idx: int) -> Set[str]:
        """自动检测场景中的歧义类型"""
        types = set()
        target = objects[target_idx]
        others = [o for i, o in enumerate(objects) if i != target_idx]

        # 视觉歧义：有物体视觉签名相同
        target_vis = target.visual_signature()
        for other in others:
            if other.visual_signature() == target_vis:
                types.add('visual_ambiguous')
                types.add('crossmodal')
                break

        # 子集歧义：目标特征是其他物体的子集
        target_keys = set(target.visual.keys())
        target_vals = set(target.visual.values())
        for other in others:
            other_vals = set(other.visual.values())
            if target_vals.issubset(other_vals) and target_vals != other_vals:
                types.add('subset')
                break

        # 跨模态歧义：视觉相同但听觉/触觉不同
        for other in others:
            if (other.visual_signature() == target_vis and
                (other.auditory != target.auditory or
                 other.tactile != target.tactile)):
                types.add('crossmodal')
                break

        # 工具歧义：视觉相同但功能不同
        for other in others:
            if (other.visual_signature() == target_vis and
                set(other.affordances) != set(target.affordances)):
                types.add('tool')
                break

        # 如果没有检测到歧义，默认 visual_ambiguous
        if not types:
            types.add('visual_ambiguous')

        return types


# ============================================================
# 经验驱动的场景生成器
# ============================================================

class ExperienceDrivenScenarioGenerator:
    """从探索经验中生成通信场景"""

    def __init__(self, bridge: EnvironmentLanguageBridge):
        self.bridge = bridge
        self.experience_buffer: List[dict] = []

    def record_observation(self, observation: dict):
        """记录探索中遇到的物体组合"""
        visible = observation.get('visible_objects', [])
        if len(visible) >= 2:
            self.experience_buffer.append(observation)

    def generate_scenario(self, fallback_ambiguity: Set[str] = None) -> UnifiedScene:
        """从经验缓冲中采样场景，经验不足时回退到随机生成"""
        if self.experience_buffer:
            obs = random.choice(self.experience_buffer)
            scene = self.bridge.observation_to_scene(obs)
            if scene is not None:
                return scene

        # 回退到随机生成
        return generate_unified_scenario(
            ambiguity_types=fallback_ambiguity or {'visual_ambiguous', 'crossmodal'},
            num_objects=3,
        )

    @property
    def experience_count(self) -> int:
        return len(self.experience_buffer)
