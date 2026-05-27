"""
Phase 27: 探索 Agent + 共同探索游戏

每个 Agent 有独立的语言系统，在共享环境中探索并用语言交流发现。
"""

import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_env import MultiAgentGridWorld, create_multi_agent_world
from language_emergence import (
    LanguageAgent, cross_language_round, compute_language_similarity,
    EmergingLanguage,
)


class ExploringAgent:
    """具有独立语言系统的探索 Agent"""

    def __init__(self, agent_id: int, env: MultiAgentGridWorld):
        self.agent_id = agent_id
        self.env = env
        self.lang_agent = LanguageAgent(f"agent_{agent_id}")
        self.discovered_objects: Set[int] = set()
        self.communication_log: List[dict] = []

    def explore_step(self) -> dict:
        """执行一步探索（随机移动）"""
        action = random.randint(0, 3)
        obs = self.env.step_agent(self.agent_id, action)
        for vis in obs['visible_objects']:
            self.discovered_objects.add(vis['object'].id)
        return obs

    def communicate_with(self, partner: 'ExploringAgent') -> bool:
        """与伙伴通信：描述自己看到的物体"""
        my_obs = self.env.get_agent_observation(self.agent_id)
        visible = my_obs['visible_objects']

        if len(visible) < 2:
            return False

        # 选择一个目标物体
        target_vis = random.choice(visible)
        target_obj = target_vis['object']

        # 构造 scene_features（所有可见物体的特征）
        scene_features = []
        target_idx = 0
        for i, vis in enumerate(visible):
            obj = vis['object']
            features = self._object_to_features(obj)
            scene_features.append(features)
            if obj.id == target_obj.id:
                target_idx = i

        # 跨语言通信
        success = cross_language_round(
            speaker_agent=self.lang_agent,
            listener_agent=partner.lang_agent,
            scene_features=scene_features,
            target_idx=target_idx,
        )

        self.communication_log.append({
            'speaker': self.agent_id,
            'listener': partner.agent_id,
            'success': success,
            'target': target_obj.id,
        })

        return success

    def _object_to_features(self, obj) -> Dict[str, str]:
        """环境物体 → 特征字典（LanguageAgent 格式）"""
        features = {
            'color': obj.color,
            'shape': obj.shape,
        }
        if hasattr(obj, 'weight'):
            if obj.weight < 1.0:
                features['size'] = 'small'
            elif obj.weight > 1.2:
                features['size'] = 'big'
            else:
                features['size'] = 'medium'
        if hasattr(obj, 'material') and obj.material:
            features['material'] = obj.material
        if hasattr(obj, 'sound') and obj.sound:
            features['sound'] = obj.sound
        if hasattr(obj, 'texture') and obj.texture:
            features['texture'] = obj.texture
        return features


class CoExplorationGame:
    """多 Agent 共同探索游戏"""

    def __init__(self, num_agents: int = 2, world_size: int = 12):
        self.env = create_multi_agent_world(num_agents)
        self.agents = [
            ExploringAgent(i, self.env) for i in range(num_agents)
        ]
        self.stats = {
            'total_comm': 0,
            'success_comm': 0,
            'comm_history': [],
        }

    def run(self, total_steps: int = 500, comm_prob: float = 0.3):
        """运行共同探索"""
        for step in range(total_steps):
            # 每个 Agent 探索一步
            for agent in self.agents:
                agent.explore_step()

            # 随机触发通信
            if random.random() < comm_prob and len(self.agents) >= 2:
                a, b = random.sample(self.agents, 2)
                # 只有双方视野中都有物体时才通信
                obs_a = self.env.get_agent_observation(a.agent_id)
                obs_b = self.env.get_agent_observation(b.agent_id)
                if len(obs_a['visible_objects']) >= 2 and len(obs_b['visible_objects']) >= 2:
                    success = a.communicate_with(b)
                    self.stats['total_comm'] += 1
                    if success:
                        self.stats['success_comm'] += 1
                    self.stats['comm_history'].append({
                        'step': step,
                        'speaker': a.agent_id,
                        'listener': b.agent_id,
                        'success': success,
                    })

    def get_stats(self) -> dict:
        """获取统计"""
        total = self.stats['total_comm']
        success = self.stats['success_comm']

        # 每个 Agent 的发现数
        discoveries = {}
        for agent in self.agents:
            discoveries[agent.agent_id] = len(agent.discovered_objects)

        # 语言相似度
        lang_sim = {}
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                sim = compute_language_similarity(
                    self.agents[i].lang_agent.language,
                    self.agents[j].lang_agent.language,
                )
                lang_sim[f'{i}-{j}'] = sim

        # 总发现数
        all_discovered = self.env.get_total_discoveries()

        return {
            'total_comm': total,
            'success_comm': success,
            'comm_success_rate': success / max(1, total),
            'discoveries': discoveries,
            'total_discovered': len(all_discovered),
            'total_objects': len(self.env.objects),
            'language_similarity': lang_sim,
        }


class SoloExplorationGame:
    """单 Agent 独自探索（对比基线）"""

    def __init__(self, world_size: int = 12):
        self.env = create_multi_agent_world(num_agents=1)
        self.agent = ExploringAgent(0, self.env)

    def run(self, total_steps: int = 500):
        """运行独自探索"""
        for _ in range(total_steps):
            self.agent.explore_step()

    def get_stats(self) -> dict:
        return {
            'discoveries': {0: len(self.agent.discovered_objects)},
            'total_discovered': len(self.agent.discovered_objects),
            'total_objects': len(self.env.objects),
        }
