"""
统一学习系统 — 交互协议

参考游戏、模仿学习、联合注意。
"""

import torch
import random
from typing import Dict, List

from .agent import SocialAgent


class InteractionProtocol:
    """多 Agent 交互协议"""

    def reference_game(
        self,
        speaker: SocialAgent,
        listener: SocialAgent,
        scene: List[Dict],
        target_idx: int,
    ) -> bool:
        """
        指称游戏：说话者描述目标物体，听话者猜测。
        返回是否成功。
        """
        if not scene or target_idx >= len(scene):
            return False

        target = scene[target_idx]
        # 简化：说话者生成描述属性，听话者匹配
        description = {
            'color': target.get('color', ''),
            'shape': target.get('shape', ''),
        }

        # 听话者选择最匹配的物体
        best_idx = 0
        best_score = 0
        for i, obj in enumerate(scene):
            score = 0
            for key in description:
                if obj.get(key) == description[key]:
                    score += 1
            if score > best_score:
                best_score = score
                best_idx = i

        success = best_idx == target_idx
        return success

    def imitation(
        self,
        observer: SocialAgent,
        demonstrator: SocialAgent,
        action: torch.Tensor,
    ) -> bool:
        """
        模仿学习：观察者尝试复制示范者的动作。
        返回是否足够接近。
        """
        # 观察者复制动作（加噪声）
        noise = torch.randn_like(action) * 0.1
        copied_action = action + noise

        # 判断是否足够接近
        distance = (copied_action - action).norm().item()
        success = distance < 1.0

        if success:
            observer.observation_history.append({
                'action': copied_action,
                'outcome': {'type': 'imitation', 'success': True},
            })

        return success

    def joint_attention(
        self,
        agent_a: SocialAgent,
        agent_b: SocialAgent,
        target: Dict,
    ) -> bool:
        """
        联合注意：两个 Agent 同时关注同一目标。
        返回是否成功建立联合注意。
        """
        # 简化：以 80% 概率成功建立联合注意
        success = random.random() < 0.8

        if success:
            for agent in [agent_a, agent_b]:
                agent.observation_history.append({
                    'action': torch.zeros(2),
                    'outcome': {'type': 'joint_attention', 'target': target},
                })

        return success
