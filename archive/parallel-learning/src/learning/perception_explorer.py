"""感知-文本桥接器（Phase 6D）

认知科学背景：
    人类婴儿通过感知运动阶段（Piaget, 0-2岁）学习基础概念：
    - 看到"红色"圆形物体 → 建立"红色"、"圆形"概念
    - 触摸"木质"方块 → 建立"木质"、"方块"概念
    - 这些感知锚点（sensory anchors）让抽象概念接地

    双重编码理论（Paivio, 1971）：
    概念同时有言语编码（文本描述）和感知编码（感官特征），
    两者互为锚点，使概念理解更深刻。

核心设计：
    1. 探索模拟环境 → 获取感知观测
    2. 从场景特征生成中文文本描述
    3. 同时调用 learn_from_experience() 和 learn_from_text()
    4. 为概念注册 sensory_anchors（感知锚点）
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from src.core.config import LearnerConfig


# ===== 英文 → 中文属性映射 =====
COLOR_MAP = {
    'red': '红色', 'blue': '蓝色', 'green': '绿色', 'yellow': '黄色',
    'orange': '橙色', 'purple': '紫色', 'pink': '粉色', 'brown': '棕色',
    'black': '黑色', 'white': '白色', 'gray': '灰色', 'cyan': '青色',
}

SHAPE_MAP = {
    'circle': '圆形', 'square': '方形', 'triangle': '三角形',
    'rectangle': '长方形', 'pentagon': '五边形', 'hexagon': '六边形',
    'star': '星形', 'diamond': '菱形',
}

SIZE_MAP = {
    'tiny': '极小的', 'small': '小的', 'medium': '中等的',
    'big': '大的', 'large': '大的', 'huge': '巨大的',
}

MATERIAL_MAP = {
    'wood': '木质', 'plastic': '塑料', 'metal': '金属',
    'glass': '玻璃', 'rubber': '橡胶', 'fabric': '布料',
    'stone': '石头', 'paper': '纸质',
}


class PerceptionExplorer:
    """感知-文本桥接器

    让系统通过探索模拟环境来建立概念的感知基础：
    1. 探索环境 → 获取感知观测（视觉、位置等）
    2. 从场景特征生成中文文本描述
    3. 同时调用 learn_from_experience() 和 learn_from_text()
    4. 注册概念时添加 sensory_anchors

    使用方法：
        world = World(config)
        world.configure_for_stage('sensorimotor')

        explorer = PerceptionExplorer(learner, world)
        explorer.explore(n_steps=100)

        # 之后 learner.think("红色的东西是什么") 能回答
    """

    def __init__(self, learner, world):
        """
        Args:
            learner: Learner 实例（需支持 learn_from_experience, learn_from_text）
            world: World 实例（需支持 reset, step, generate_scene_features）
        """
        self.learner = learner
        self.world = world

        # 统计
        self._steps_done = 0
        self._texts_generated = 0
        self._concepts_anchored = 0

    def explore(self, n_steps: int = 100, verbose: bool = False) -> Dict:
        """探索环境 n 步，同时学习感知和语言

        流程：
        1. 配置环境（根据 learner 的当前阶段）
        2. 循环 n_steps：
           a. 随机动作 → step()
           b. 场景特征 → 中文文本
           c. learn_from_experience() — 感知学习
           d. learn_from_text() — 语言学习
           e. 注册感知锚点 — 双重编码

        Args:
            n_steps: 探索步数
            verbose: 是否打印进度

        Returns:
            统计信息字典
        """
        # 根据学习体的发展阶段配置环境
        # 注意：configure_for_stage() 内部会调 reset() 并创建物体
        # 之后再调 reset() 会清空所有物体！所以不要重复 reset
        stage = getattr(self.learner, '_stage', 'sensorimotor')
        try:
            self.world.configure_for_stage(stage)
        except Exception:
            # 如果阶段名不匹配，用最简单的配置
            self.world.configure_for_stage('sensorimotor')

        obs = self.world.observe()
        action_dim = self.learner.config.action_dim

        for step in range(n_steps):
            # 随机动作探索（好奇心驱动）
            action = torch.randn(action_dim) * 0.5

            # 执行动作
            next_obs, reward, done = self.world.step(action)

            # === 感知学习 ===
            try:
                obs_tensor = self._to_tensor(obs)
                next_obs_tensor = self._to_tensor(next_obs)
                self.learner.learn_from_experience(obs_tensor, action, next_obs_tensor)
            except Exception:
                pass

            # === 场景 → 文本学习 ===
            scene_text = self._scene_to_text()
            if scene_text:
                self._texts_generated += 1

                try:
                    self.learner.learn_from_text(scene_text)
                except Exception:
                    pass

                # === 注册感知锚点并标记来源 ===
                self._register_sensory_anchors()

            self._steps_done += 1

            if verbose and (step + 1) % 20 == 0:
                print(f"  [探索 {step+1}/{n_steps}] 生成文本: {self._texts_generated}条, "
                      f"感知锚点: {self._concepts_anchored}个")

            obs = next_obs
            if done:
                obs = self.world.reset()

        stats = {
            'steps': self._steps_done,
            'texts_generated': self._texts_generated,
            'concepts_anchored': self._concepts_anchored,
        }
        if verbose:
            print(f"  探索完成: {stats}")
        return stats

    def _scene_to_text(self) -> str:
        """将场景特征转为中文文本描述

        关键设计：每个属性独立成句，用逗号/顿号分隔。
        这样统计学习者能正确切出 "红色"、"圆形" 等独立概念，
        而不是碎片 "色圆形小"。

        例如：
            [{'color':'red','shape':'circle','size':'small','material':'wood'}]
            → "这个物体是红色的。它是圆形的。它是小的。它是木质材质。"
        """
        try:
            features = self.world.generate_scene_features()
        except Exception:
            return ""

        if not features:
            return ""

        sentences = []
        for f in features[:3]:  # 最多描述3个物体
            if 'color' in f:
                cn = COLOR_MAP.get(f['color'])
                if cn:
                    sentences.append(f"这个物体是{cn}的")
            if 'shape' in f:
                cn = SHAPE_MAP.get(f['shape'])
                if cn:
                    sentences.append(f"它是{cn}的")
            if 'size' in f:
                cn = SIZE_MAP.get(f['size'])
                if cn:
                    sentences.append(f"它是{cn}")
            if 'material' in f:
                cn = MATERIAL_MAP.get(f['material'])
                if cn:
                    sentences.append(f"它是{cn}材质")

        if not sentences:
            return ""

        return "。".join(sentences) + "。"

    def _register_sensory_anchors(self):
        """为概念空间中的感知概念注册感知锚点

        Phase 7 修复：learn_from_text() 的层5过滤会拒绝所有未统计验证的2字概念，
        导致"红色"/"圆形"等感知概念永远进不了概念空间。
        修复策略：如果感知概念不在概念空间中，直接注册（绕过文本学习的严格过滤）。
        """
        try:
            cs = self.learner._registry.get('concept_space')
        except Exception:
            return

        if not cs:
            return

        try:
            features = self.world.generate_scene_features()
        except Exception:
            return

        # 从场景提取所有中文属性值
        sensory_concepts = set()
        for f in features:
            for attr, value in f.items():
                cn_map = {
                    'color': COLOR_MAP,
                    'shape': SHAPE_MAP,
                    'size': SIZE_MAP,
                    'material': MATERIAL_MAP,
                }
                if attr in cn_map:
                    cn_value = cn_map[attr].get(value)
                    if cn_value:
                        sensory_concepts.add(cn_value)

        # 在概念空间中找到这些概念并标记
        for concept in sensory_concepts:
            if concept in cs.concepts:
                # 概念已存在 → 标记感知来源和锚点
                node = cs.concepts[concept]

                # 标记来源为感知
                node.source = 'perception'

                # 添加感知锚点（避免重复）
                anchor = f'sensory:{concept}'
                if node.sensory_anchors is None:
                    node.sensory_anchors = []
                if anchor not in node.sensory_anchors:
                    node.sensory_anchors.append(anchor)
                    self._concepts_anchored += 1
            else:
                # Phase 7 新增：概念不在空间中 → 直接注册
                # 感知概念通过感知通道进入，不需要通过文本学习的严格过滤
                try:
                    entity_repr = self.learner._encode_text(concept)
                    anchor = f'sensory:{concept}'
                    cs.register(
                        concept,
                        vector=entity_repr,
                        source='perception',
                        sensory_anchors=[anchor],
                    )
                    self._concepts_anchored += 1
                except Exception:
                    pass

    def _to_tensor(self, obs: Dict) -> torch.Tensor:
        """将观测字典转为张量

        支持多种观测格式：
        - Dict[str, Tensor] → concat 所有张量
        - Tensor → 直接返回
        """
        if isinstance(obs, torch.Tensor):
            return obs

        if isinstance(obs, dict):
            tensors = []
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, torch.Tensor):
                    tensors.append(val.flatten())
            if tensors:
                return torch.cat(tensors)
            return torch.zeros(self.learner.config.obs_dim)

        return torch.zeros(self.learner.config.obs_dim)

    def get_stats(self) -> Dict:
        """获取探索统计"""
        return {
            'steps': self._steps_done,
            'texts_generated': self._texts_generated,
            'concepts_anchored': self._concepts_anchored,
        }
