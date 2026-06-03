"""
统一学习系统 — 社会 Agent

实现 ISocialAgent 接口，支持交互、观察、教学。
"""

import torch
from typing import Dict

from src.core.interfaces import ISocialAgent, IEnvironment


class SocialAgent(ISocialAgent):
    """社会 Agent"""

    def __init__(self, agent_id: str, learner=None):
        self.agent_id = agent_id
        self.learner = learner
        self.observation_history: list = []

    def interact(self, partner: 'SocialAgent', env: IEnvironment) -> Dict:
        """与另一个 Agent 在环境中交互"""
        obs = env.observe()

        # 简单交互：双方交换观测结果
        partner_obs = env.observe()

        result = {
            'self_id': self.agent_id,
            'partner_id': partner.agent_id,
            'shared_observation': obs,
            'success': True,
        }
        return result

    def observe_partner(self, partner_action: torch.Tensor, partner_outcome: Dict) -> None:
        """观察伙伴的行为和结果（社会学习）"""
        self.observation_history.append({
            'action': partner_action.clone(),
            'outcome': partner_outcome,
        })

    def teach(self, learner: 'SocialAgent', topic: str) -> Dict:
        """向另一个 Agent 教授特定主题"""
        lesson = {
            'teacher': self.agent_id,
            'student': learner.agent_id,
            'topic': topic,
            'transmitted': True,
        }
        # 学生记录教学事件
        learner.observation_history.append({
            'action': torch.zeros(2),
            'outcome': {'type': 'lesson', 'topic': topic},
        })
        return lesson
