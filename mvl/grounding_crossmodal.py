"""
Phase 23: 跨模态语言模块 —— 视觉、听觉、触觉整合

核心思想：
当视觉特征无法区分物体时，听觉和触觉特征成为必要描述。
这驱动跨模态符号涌现。

涌现条件：
1. 两个物体外观完全相同
2. 但听觉或触觉特征不同
3. Speaker 必须用非视觉特征描述
4. "loud", "rough" 等跨模态符号从需要区分时涌现
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
    AUDITORY_SYMBOLS, TACTILE_SYMBOLS,
)


class CrossModalObject:
    """
    跨模态物体：具有视觉、听觉、触觉三种模态特征

    视觉特征：颜色、形状、大小
    听觉特征：声音属性（loud/quiet, sharp/soft 等）
    触觉特征：触感属性（rough/smooth, hot/cold 等）
    """

    def __init__(self, visual: Dict[str, str],
                 auditory: Dict[str, str] = None,
                 tactile: Dict[str, str] = None):
        self.visual = visual or {}
        self.auditory = auditory or {}
        self.tactile = tactile or {}

    def to_visual_symbols(self) -> List[str]:
        return list(self.visual.values())

    def to_auditory_symbols(self) -> List[str]:
        return list(self.auditory.values())

    def to_tactile_symbols(self) -> List[str]:
        return list(self.tactile.values())

    def to_all_symbols(self) -> List[str]:
        symbols = []
        symbols.extend(self.visual.values())
        symbols.extend(self.auditory.values())
        symbols.extend(self.tactile.values())
        return symbols

    def to_dict(self) -> Dict[str, str]:
        """合并所有特征为一个字典"""
        d = {}
        d.update(self.visual)
        d.update(self.auditory)
        d.update(self.tactile)
        return d

    def __repr__(self):
        return f"CrossModalObject(visual={self.visual}, auditory={self.auditory}, tactile={self.tactile})"


# 听觉特征模板
AUDITORY_FEATURES = {
    'sound': ['loud', 'quiet', 'sharp', 'soft', 'buzz', 'click'],
    'pitch': ['high', 'low'],
    'rhythm': ['steady', 'irregular'],
}

# 触觉特征模板
TACTILE_FEATURES = {
    'texture': ['rough', 'smooth', 'fuzzy'],
    'hardness': ['hard', 'soft_tactile'],
    'temperature': ['hot', 'cold'],
    'moisture': ['wet', 'dry'],
}


def generate_crossmodal_scenario(mode: str = 'visual_unique',
                                  num_objects: int = 4) -> Tuple[List[CrossModalObject], int]:
    """
    生成跨模态场景

    参数：
        mode: 场景模式
            - 'visual_unique': 外观唯一（基线，不需要跨模态）
            - 'visual_ambiguous': 外观相同，需要非视觉特征
            - 'crossmodal_only': 只有非视觉特征能区分
            - 'mixed': 混合场景
        num_objects: 物体数量

    返回：
        (objects, target_idx)
    """
    colors = list(COLORS)
    shapes = list(SHAPES)
    sizes = list(SIZES)

    np.random.shuffle(colors)
    np.random.shuffle(shapes)
    np.random.shuffle(sizes)

    # 听觉和触觉符号池
    auditory_pool = list(AUDITORY_SYMBOLS)
    tactile_pool = list(TACTILE_SYMBOLS)
    np.random.shuffle(auditory_pool)
    np.random.shuffle(tactile_pool)

    if mode == 'visual_unique':
        # 基线：每个物体外观不同，不需要跨模态
        objects = []
        for i in range(num_objects):
            visual = {
                'color': colors[i % len(colors)],
                'shape': shapes[i % len(shapes)],
                'size': sizes[i % len(sizes)],
            }
            # 也有听觉和触觉特征，但不需要用它们来区分
            auditory = {'sound': auditory_pool[i % len(auditory_pool)]}
            tactile = {'texture': tactile_pool[i % len(tactile_pool)]}
            objects.append(CrossModalObject(visual, auditory, tactile))

        target_idx = np.random.randint(0, len(objects))
        return objects, target_idx

    elif mode == 'visual_ambiguous':
        # 关键场景：两个物体外观相同，但听觉/触觉不同
        shared_visual = {
            'color': colors[0],
            'shape': shapes[0],
            'size': sizes[0],
        }

        # 两个外观相同的物体
        obj_a = CrossModalObject(
            visual=dict(shared_visual),
            auditory={'sound': auditory_pool[0]},
            tactile={'texture': tactile_pool[0]},
        )
        obj_b = CrossModalObject(
            visual=dict(shared_visual),
            auditory={'sound': auditory_pool[1]},
            tactile={'texture': tactile_pool[1]},
        )

        # 其他物体（外观不同）
        other_objects = []
        for i in range(num_objects - 2):
            visual = {
                'color': colors[(i + 1) % len(colors)],
                'shape': shapes[(i + 1) % len(shapes)],
                'size': sizes[(i + 1) % len(sizes)],
            }
            auditory = {'sound': auditory_pool[(i + 2) % len(auditory_pool)]}
            tactile = {'texture': tactile_pool[(i + 2) % len(tactile_pool)]}
            other_objects.append(CrossModalObject(visual, auditory, tactile))

        all_objects = [obj_a, obj_b] + other_objects
        np.random.shuffle(all_objects)

        # 目标是其中一个相同外观的物体
        target_obj = obj_a if np.random.random() < 0.5 else obj_b
        target_idx = all_objects.index(target_obj)

        return all_objects, target_idx

    elif mode == 'crossmodal_only':
        # 只有非视觉特征能区分（所有物体外观相同）
        shared_visual = {
            'color': colors[0],
            'shape': shapes[0],
            'size': sizes[0],
        }

        objects = []
        for i in range(num_objects):
            auditory = {'sound': auditory_pool[i % len(auditory_pool)]}
            tactile = {'texture': tactile_pool[i % len(tactile_pool)]}
            objects.append(CrossModalObject(
                visual=dict(shared_visual),
                auditory=auditory,
                tactile=tactile,
            ))

        target_idx = np.random.randint(0, len(objects))
        return objects, target_idx

    elif mode == 'mixed':
        # 混合场景：有时需要跨模态，有时不需要
        if np.random.random() < 0.5:
            return generate_crossmodal_scenario('visual_ambiguous', num_objects)
        else:
            return generate_crossmodal_scenario('visual_unique', num_objects)

    # 默认
    return generate_crossmodal_scenario('visual_unique', num_objects)


class CrossModalGame:
    """
    跨模态通信游戏

    游戏流程：
    1. 生成场景：多个物体（可能有相同外观）
    2. Speaker 描述目标物体
    3. 如果外观唯一，用视觉描述
    4. 如果外观不唯一，用听觉/触觉描述
    5. Listener 根据描述选择物体

    涌现压力：
    - 外观相同时，必须用非视觉特征区分
    - "loud", "rough" 等跨模态符号从需要区分时涌现
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []
        self.visual_only_used = 0
        self.crossmodal_used = 0
        self.visual_success = 0
        self.crossmodal_success = 0

    def play_round(self, objects: List[CrossModalObject],
                   target_idx: int) -> bool:
        if target_idx >= len(objects):
            return False

        target = objects[target_idx]

        # 检查视觉是否唯一
        visual_unique = self._is_visual_unique(target, objects)

        if visual_unique:
            # 外观唯一，用视觉描述
            utterance = target.to_visual_symbols()
            self.visual_only_used += 1
        else:
            # 外观不唯一，用跨模态描述
            utterance = self._describe_crossmodal(target, objects)
            self.crossmodal_used += 1

        if not utterance:
            return False

        # Listener 解释
        chosen_idx = self._listener_interpret(utterance, objects)
        success = (chosen_idx == target_idx)

        # 更新统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1

        if visual_unique and success:
            self.visual_success += 1
        elif not visual_unique and success:
            self.crossmodal_success += 1

        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
            'visual_unique': visual_unique,
        })
        return success

    def _is_visual_unique(self, target: CrossModalObject,
                          objects: List[CrossModalObject]) -> bool:
        """检查目标的视觉特征是否唯一"""
        target_visual = set(target.to_visual_symbols())
        for i, obj in enumerate(objects):
            if obj is target:
                continue
            obj_visual = set(obj.to_visual_symbols())
            # 如果有重叠视觉特征，外观不唯一
            if len(target_visual & obj_visual) >= 2:
                return False
        return True

    def _describe_crossmodal(self, target: CrossModalObject,
                              objects: List[CrossModalObject]) -> List[str]:
        """生成跨模态描述"""
        symbols = []

        # 先用视觉特征（缩小范围）
        visual = target.to_visual_symbols()
        symbols.extend(visual[:1])  # 最多 1 个视觉特征

        # 添加听觉特征
        auditory = target.to_auditory_symbols()
        if auditory:
            symbols.extend(auditory[:1])

        # 添加触觉特征
        tactile = target.to_tactile_symbols()
        if tactile:
            symbols.extend(tactile[:1])

        return symbols

    def _listener_interpret(self, utterance: List[str],
                            objects: List[CrossModalObject]) -> int:
        """Listener 解释描述"""
        scores = []
        for i, obj in enumerate(objects):
            obj_symbols = set(obj.to_all_symbols())
            matches = sum(1 for s in utterance if s in obj_symbols)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['visual_only_used'] = self.visual_only_used
        stats['crossmodal_used'] = self.crossmodal_used
        stats['visual_success'] = self.visual_success / max(1, self.visual_only_used)
        stats['crossmodal_success'] = self.crossmodal_success / max(1, self.crossmodal_used)
        return stats


class BaselineCrossModalGame:
    """只使用视觉描述的基线游戏"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, objects: List[CrossModalObject],
                   target_idx: int) -> bool:
        if target_idx >= len(objects):
            return False

        target = objects[target_idx]

        # 基线：只用视觉描述
        utterance = target.to_visual_symbols()

        # Listener 只用视觉匹配
        scores = []
        for i, obj in enumerate(objects):
            obj_visual = set(obj.to_visual_symbols())
            matches = sum(1 for s in utterance if s in obj_visual)
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


def test_crossmodal():
    """测试跨模态机制"""
    print("=== 跨模态机制测试 ===")

    # 创建跨模态物体
    obj = CrossModalObject(
        visual={'color': 'red', 'shape': 'circle'},
        auditory={'sound': 'loud'},
        tactile={'texture': 'rough'},
    )
    print(f"\n物体: {obj}")
    print(f"  视觉: {obj.to_visual_symbols()}")
    print(f"  听觉: {obj.to_auditory_symbols()}")
    print(f"  触觉: {obj.to_tactile_symbols()}")
    print(f"  全部: {obj.to_all_symbols()}")

    # 测试场景生成
    objects, target_idx = generate_crossmodal_scenario(mode='visual_ambiguous', num_objects=4)
    print(f"\n场景 (visual_ambiguous):")
    print(f"  物体数: {len(objects)}")
    print(f"  目标: {target_idx}")
    for i, obj in enumerate(objects):
        print(f"  [{i}] visual={obj.to_visual_symbols()}, auditory={obj.to_auditory_symbols()}")

    # 测试视觉唯一性检测
    game = CrossModalGame()
    target = objects[target_idx]
    is_unique = game._is_visual_unique(target, objects)
    print(f"\n  视觉唯一: {is_unique}")


if __name__ == '__main__':
    test_crossmodal()
