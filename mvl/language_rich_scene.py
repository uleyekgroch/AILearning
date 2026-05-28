"""
大概念空间场景生成：从 72 到 10,368 唯一对象

核心思想：
当前语言系统只有 4 属性（color/shape/size/material）= 72 唯一对象。
在如此小的空间中，最优编码唯一，所有 agent 必然快速趋同。

扩展到 10 属性 = 10,368 唯一对象后：
- 每个场景只展示 8 个对象，从 10,368 中随机采样
- 两个 agent 很少看到完全相同的对象组合
- 对同一对象，不同 agent 可能发展出不同的最优描述策略
- 最优编码不再唯一 → 方言分化成为可能

区域化设计：
- 不同 agent 群体面对不同的属性分布偏好
- 模拟地理隔离导致的环境差异
- 区域 A 偏好 "red metal" → agent 倾向用 "red"/"metal" 描述
- 区域 B 偏好 "blue wood" → agent 倾向用 "blue"/"wood" 描述
"""

import random
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field


# ============================================================
# 10 属性维度，总空间 = 10,368
# ============================================================

RICH_ATTRIBUTES = {
    'color':      ['red', 'blue', 'green', 'yellow', 'white', 'black'],       # 6
    'shape':      ['circle', 'square', 'triangle', 'star', 'diamond'],        # 5
    'size':       ['tiny', 'small', 'medium', 'big', 'huge'],                 # 5
    'material':   ['metal', 'wood', 'plastic', 'stone', 'fabric', 'glass'],   # 6
    'texture':    ['smooth', 'rough', 'bumpy', 'fuzzy'],                      # 4
    'weight':     ['light', 'medium', 'heavy'],                                # 3
    'temperature': ['hot', 'warm', 'cold'],                                    # 3
    'brightness': ['bright', 'dim', 'dark'],                                   # 3
    'pattern':    ['solid', 'striped', 'spotted'],                             # 3
    'origin':     ['natural', 'artificial', 'magical'],                        # 3
}

# 总空间大小
TOTAL_SPACE = 1
for values in RICH_ATTRIBUTES.values():
    TOTAL_SPACE *= len(values)
# 6×5×5×6×4×3×3×3×3×3 = 10,368

# 所有属性名
ALL_ATTRIBUTE_NAMES = list(RICH_ATTRIBUTES.keys())

# 所有符号值（用于 _symbol_category 扩展）
ALL_RICH_SYMBOLS = set()
for values in RICH_ATTRIBUTES.values():
    ALL_RICH_SYMBOLS.update(values)


def generate_rich_scene_v2(num_objects: int = 8,
                           num_attributes: int = 4,
                           attribute_names: List[str] = None) -> List[Dict[str, str]]:
    """
    生成大概念空间场景

    Args:
        num_objects: 场景中的物体数量
        num_attributes: 每个物体使用的属性维度数
        attribute_names: 指定使用的属性名列表（None 则随机选择）

    Returns:
        List[Dict[str, str]] — 物体列表，每个物体是属性→值的字典
    """
    if attribute_names is None:
        attribute_names = random.sample(ALL_ATTRIBUTE_NAMES,
                                        min(num_attributes, len(ALL_ATTRIBUTE_NAMES)))

    scene = []
    for _ in range(num_objects):
        obj = {}
        for attr in attribute_names:
            values = RICH_ATTRIBUTES[attr]
            obj[attr] = random.choice(values)
        scene.append(obj)

    random.shuffle(scene)
    return scene


def generate_regional_scene(region_biases: Dict[str, Dict[str, float]],
                            num_objects: int = 8,
                            num_attributes: int = 4,
                            attribute_names: List[str] = None) -> List[Dict[str, str]]:
    """
    根据区域偏好生成场景

    Args:
        region_biases: {attribute_name: {value: probability}} 的偏好分布
            例如 {'color': {'red': 0.5, 'blue': 0.2, ...}} 表示该区域偏好红色物体
        num_objects: 场景中的物体数量
        num_attributes: 每个物体使用的属性维度数
        attribute_names: 指定使用的属性名列表

    Returns:
        List[Dict[str, str]] — 带区域偏好的物体列表
    """
    if attribute_names is None:
        attribute_names = random.sample(ALL_ATTRIBUTE_NAMES,
                                        min(num_attributes, len(ALL_ATTRIBUTE_NAMES)))

    scene = []
    for _ in range(num_objects):
        obj = {}
        for attr in attribute_names:
            values = RICH_ATTRIBUTES[attr]
            if attr in region_biases:
                # 按区域偏好分布采样
                bias = region_biases[attr]
                probs = [bias.get(v, 1.0 / len(values)) for v in values]
                total = sum(probs)
                probs = [p / total for p in probs]
                obj[attr] = np.random.choice(values, p=probs)
            else:
                obj[attr] = random.choice(values)
        scene.append(obj)

    random.shuffle(scene)
    return scene


@dataclass
class RegionConfig:
    """
    区域配置：定义一个区域的属性分布偏好

    每个区域偏好特定的属性值组合，模拟地理隔离导致的环境差异。
    例如：
    - 热带区域：偏好 red/green 色、big/huge 尺寸、hot/warm 温度
    - 极地区域：偏好 white/blue 色、small/tiny 尺寸、cold 温度
    - 工业区域：偏好 metal/plastic 材质、artificial 起源
    """
    region_id: int
    name: str = ""
    biases: Dict[str, Dict[str, float]] = field(default_factory=dict)
    preferred_attributes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = f"region_{self.region_id}"
        if not self.biases:
            self.biases = self._default_biases()
        if not self.preferred_attributes:
            self.preferred_attributes = random.sample(ALL_ATTRIBUTE_NAMES, 4)

    def _default_biases(self) -> Dict[str, Dict[str, float]]:
        """根据 region_id 生成默认偏好"""
        # 5 种预定义区域类型
        templates = [
            # 热带：偏好暖色、大尺寸、热温度
            {
                'color': {'red': 0.4, 'green': 0.3, 'yellow': 0.2, 'blue': 0.05, 'white': 0.03, 'black': 0.02},
                'size': {'tiny': 0.05, 'small': 0.1, 'medium': 0.2, 'big': 0.35, 'huge': 0.3},
                'temperature': {'hot': 0.5, 'warm': 0.35, 'cold': 0.15},
                'origin': {'natural': 0.6, 'artificial': 0.2, 'magical': 0.2},
            },
            # 极地：偏好冷色、小尺寸、冷温度
            {
                'color': {'white': 0.4, 'blue': 0.3, 'black': 0.2, 'green': 0.05, 'red': 0.03, 'yellow': 0.02},
                'size': {'tiny': 0.3, 'small': 0.35, 'medium': 0.2, 'big': 0.1, 'huge': 0.05},
                'temperature': {'cold': 0.6, 'warm': 0.3, 'hot': 0.1},
                'brightness': {'bright': 0.5, 'dim': 0.3, 'dark': 0.2},
            },
            # 工业：偏好金属、人工制品
            {
                'material': {'metal': 0.4, 'plastic': 0.3, 'stone': 0.15, 'wood': 0.1, 'fabric': 0.03, 'glass': 0.02},
                'origin': {'artificial': 0.6, 'natural': 0.2, 'magical': 0.2},
                'texture': {'smooth': 0.4, 'rough': 0.3, 'bumpy': 0.2, 'fuzzy': 0.1},
                'weight': {'heavy': 0.4, 'medium': 0.35, 'light': 0.25},
            },
            # 森林：偏好自然物、木质
            {
                'material': {'wood': 0.4, 'fabric': 0.25, 'stone': 0.2, 'natural': 0.1, 'glass': 0.03, 'metal': 0.02},
                'origin': {'natural': 0.7, 'magical': 0.2, 'artificial': 0.1},
                'pattern': {'solid': 0.3, 'striped': 0.35, 'spotted': 0.35},
                'texture': {'rough': 0.3, 'bumpy': 0.3, 'fuzzy': 0.25, 'smooth': 0.15},
            },
            # 魔法：偏好魔法起源、特殊图案
            {
                'origin': {'magical': 0.6, 'natural': 0.2, 'artificial': 0.2},
                'brightness': {'bright': 0.5, 'dim': 0.2, 'dark': 0.3},
                'pattern': {'spotted': 0.4, 'striped': 0.35, 'solid': 0.25},
                'temperature': {'hot': 0.3, 'warm': 0.4, 'cold': 0.3},
            },
        ]
        template = templates[self.region_id % len(templates)]
        return template

    def generate_scene(self, num_objects: int = 8,
                       num_attributes: int = 4) -> List[Dict[str, str]]:
        """为该区域生成一个场景"""
        return generate_regional_scene(
            self.biases, num_objects, num_attributes, self.preferred_attributes
        )


def get_all_rich_symbols() -> Set[str]:
    """获取所有大概念空间的符号值"""
    return ALL_RICH_SYMBOLS.copy()


def get_symbol_category_rich(sym: str) -> Optional[str]:
    """
    为大概念空间的符号值返回类别名

    用于扩展 language_emergence.py 的 _symbol_category 函数。
    """
    for attr_name, values in RICH_ATTRIBUTES.items():
        if sym in values:
            return attr_name
    return None
