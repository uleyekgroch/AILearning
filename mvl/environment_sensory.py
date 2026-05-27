"""
感官网格世界：从抽象特征到原始像素/声音

在 SimpleGridWorld 基础上添加：
- 2D 俯视渲染器：生成 8x8x4 的视觉数组（depth, R, G, B）
- 音频事件系统：生成 7 维的音频事件向量

这是从"上帝视角"到"婴儿视角"的关键转变。
"""

import numpy as np
from typing import List, Tuple, Optional
from environment import SimpleGridWorld, Object


# 颜色映射：颜色名 → RGB 值
COLOR_RGB = {
    'red':    np.array([1.0, 0.0, 0.0]),
    'blue':   np.array([0.0, 0.0, 1.0]),
    'green':  np.array([0.0, 1.0, 0.0]),
    'yellow': np.array([1.0, 1.0, 0.0]),
}

# 音频事件类型
AUDIO_TYPES = {'movement': 0, 'proximity': 1, 'collision': 2}


class SensoryGridWorld(SimpleGridWorld):
    """
    感官网格世界

    继承 SimpleGridWorld，添加原始感官输出：
    - visual: (8, 8, 4) 2D 俯视图 [depth, R, G, B]
    - audio: (7,) 音频事件向量 [type_onehot(3), amplitude, direction(3)]
    """

    def __init__(self, width: int = 10, height: int = 10,
                 visual_size: int = 8, audio_dim: int = 7):
        super().__init__(width, height)
        self.visual_size = visual_size
        self.audio_dim = audio_dim

        # 上一步动作（用于判断是否有移动事件）
        self._last_action = None
        self._last_agent_x = None
        self._last_agent_y = None

    def _render_visual(self) -> np.ndarray:
        """
        2D 俯视渲染

        生成 (visual_size, visual_size, 4) 数组：
        - 以 agent 为中心的局部视图
        - 通道：[depth, R, G, B]
        - agent 位置：亮度最高（R=1.0）
        - 物体：按颜色填入 RGB
        - 距离越远 depth 越大

        Returns:
            (8, 8, 4) 视觉数组
        """
        size = self.visual_size
        visual = np.zeros((size, size, 4))  # [depth, R, G, B]

        # 以 agent 为中心，视野范围
        half = size // 2

        for i in range(size):
            for j in range(size):
                # 网格坐标（以 agent 为中心）
                gx = self.agent_x + (j - half)
                gy = self.agent_y + (i - half)

                # 距离（归一化到 0-1）
                dist = abs(j - half) + abs(i - half)
                visual[i, j, 0] = dist / (size)  # depth channel

                # 检查是否有物体在此位置
                for obj in self.objects:
                    if obj.x == gx and obj.y == gy:
                        rgb = COLOR_RGB.get(obj.color, np.array([0.5, 0.5, 0.5]))
                        visual[i, j, 1] = rgb[0]  # R
                        visual[i, j, 2] = rgb[1]  # G
                        visual[i, j, 3] = rgb[2]  # B
                        break

        # Agent 位置标记（最亮）
        visual[half, half, 1] = 1.0  # R
        visual[half, half, 2] = 1.0  # G
        visual[half, half, 3] = 1.0  # B
        visual[half, half, 0] = 0.0  # depth = 0

        return visual

    def _get_audio_events(self, action: int) -> np.ndarray:
        """
        音频事件系统

        根据动作和环境状态生成音频事件：
        - 移动事件：agent 移动时产生
        - 接近事件：靠近物体时产生
        - 碰撞事件：撞到物体时产生

        Returns:
            (7,) 音频向量 [type_onehot(3), amplitude(1), direction(3)]
        """
        audio = np.zeros(self.audio_dim)
        event_type = None
        amplitude = 0.0
        direction = np.array([0.0, 0.0, 0.0])

        # 检查是否有移动
        moved = (self.agent_x != self._last_agent_x or
                 self.agent_y != self._last_agent_y)

        if moved and action in (0, 1, 2, 3):
            event_type = 'movement'
            amplitude = 0.3
            # 方向：移动方向
            if action == 0: direction = np.array([0, -1, 0])  # 上
            elif action == 1: direction = np.array([0, 1, 0])  # 下
            elif action == 2: direction = np.array([-1, 0, 0])  # 左
            elif action == 3: direction = np.array([1, 0, 0])  # 右

        # 检查是否靠近物体（距离 <= 2）
        min_dist = float('inf')
        closest_obj = None
        for obj in self.objects:
            dist = abs(obj.x - self.agent_x) + abs(obj.y - self.agent_y)
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj

        if closest_obj is not None and min_dist <= 2:
            if event_type is None:
                event_type = 'proximity'
            amplitude = max(amplitude, 1.0 / (1.0 + min_dist))
            # 方向：指向物体
            dx = closest_obj.x - self.agent_x
            dy = closest_obj.y - self.agent_y
            norm = max(1, abs(dx) + abs(dy))
            direction = np.array([dx / norm, dy / norm, 0])

        # 检查碰撞（action=4 且位置有物体）
        if action == 4:
            for obj in self.objects:
                if obj.x == self.agent_x and obj.y == self.agent_y:
                    event_type = 'collision'
                    amplitude = 1.0
                    break

        # 编码
        if event_type is not None:
            type_idx = AUDIO_TYPES[event_type]
            audio[type_idx] = 1.0  # one-hot
        audio[3] = amplitude
        audio[4:7] = direction

        return audio

    def get_observation(self) -> dict:
        """
        重写观测：添加原始感官数据

        Returns:
            dict with:
            - agent_position: (2,) 归一化位置
            - visible_objects: list of object dicts
            - visual: (8, 8, 4) 2D 俯视图
            - audio: (7,) 音频事件向量
            - step: int
        """
        obs = super().get_observation()
        obs['visual'] = self._render_visual()
        obs['audio'] = self._get_audio_events(self._last_action or 0)
        return obs

    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        重写 step：记录动作状态用于音频事件
        """
        self._last_action = action
        self._last_agent_x = self.agent_x
        self._last_agent_y = self.agent_y
        return super().step(action)

    def reset(self) -> dict:
        """
        重写 reset：初始化状态
        """
        obs = super().reset()  # 先重置位置
        self._last_action = None
        self._last_agent_x = self.agent_x  # 再记录重置后的位置
        self._last_agent_y = self.agent_y
        return obs


def create_sensory_world() -> SensoryGridWorld:
    """创建感性测试世界"""
    env = SensoryGridWorld(10, 10)
    env.add_object(Object(0, 2, 2, 'red', 'circle', 1.0, material='metal'))
    env.add_object(Object(1, 5, 5, 'blue', 'square', 1.5, material='stone'))
    env.add_object(Object(2, 7, 3, 'green', 'triangle', 0.8, material='fabric'))
    env.add_object(Object(3, 3, 7, 'yellow', 'circle', 1.2, material='wood'))
    for obj in env.objects:
        obj.enrich_features()
    return env
