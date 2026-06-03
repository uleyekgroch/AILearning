"""
高级睡眠巩固系统 - 基于2024年最新研究

Sleep micro-structure organizes memory replay
Context-driven memory reactivation
Awake replay: off the clock but on the job
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional


class AdvancedSleepConsolidation:
    """
    高级睡眠巩固系统

    基于2024年最新研究：
    1. 睡眠微观结构组织记忆回放
    2. 上下文驱动的记忆重激活
    3. 清醒时回放支持决策
    4. 海马电路平衡记忆重激活
    """

    def __init__(self, hippocampal_capacity=5000, neocortical_capacity=50000):
        # 海马系统（快速学习）
        self.hippocampus = {
            'episodes': [],
            'index': defaultdict(list),
            'capacity': hippocampal_capacity,
            'importance': [],  # 每个情节的重要性
        }

        # 皮层系统（慢速整合）
        self.neocortex = {
            'schemas': defaultdict(list),  # 图式
            'patterns': defaultdict(int),  # 模式频率
            'capacity': neocortical_capacity,
        }

        # 上下文模型（用于选择性回放）
        self.context_model = {}

        # 睡眠阶段
        self.sleep_stages = ['NREM', 'REM', 'Awake']
        self.current_stage = 'Awake'

        # 回放历史
        self.replay_history = deque(maxlen=1000)

        # 统计
        self.consolidation_count = 0
        self.replay_count = 0

    def store_episode(self, episode: dict, importance: float = 0.5):
        """
        存储情节到海马

        Args:
            episode: 情节数据 {entities, relations, context, timestamp}
            importance: 重要性评分 [0, 1]
        """
        # 添加时间戳
        episode['timestamp'] = len(self.hippocampus['episodes'])
        episode['importance'] = importance

        # 索引实体
        for entity in episode.get('entities', []):
            self.hippocampus['index'][entity].append(episode['timestamp'])

        # 存储情节
        self.hippocampus['episodes'].append(episode)
        self.hippocampus['importance'].append(importance)

        # 容量管理：选择性遗忘
        if len(self.hippocampus['episodes']) > self.hippocampus['capacity']:
            self._selective_forgetting()

    def _selective_forgetting(self):
        """选择性遗忘：优先遗忘不重要的情节"""
        # 找到最不重要的N个情节
        num_to_forget = 10
        indices = np.argsort(self.hippocampus['importance'])[:num_to_forget]

        # 删除这些情节
        for idx in sorted(indices, reverse=True):
            # 从索引中删除
            for entity_list in self.hippocampus['index'].values():
                entity_list = [i for i in entity_list if i != idx]

            # 删除情节
            if idx < len(self.hippocampus['episodes']):
                del self.hippocampus['episodes'][idx]
                del self.hippocampus['importance'][idx]

    def sleep_cycle(self, num_cycles: int = 5):
        """
        执行睡眠周期

        一个睡眠周期包括多个阶段的回放
        """
        for cycle in range(num_cycles):
            # NREM阶段：系统巩固
            self.current_stage = 'NREM'
            self._replay_stage('NREM', duration=10)

            # REM阶段：快速整合
            self.current_stage = 'REM'
            self._replay_stage('REM', duration=5)

            # 微觉醒：清醒回放
            self.current_stage = 'Awake'
            self._replay_stage('Awake', duration=2)

        self.consolidation_count += num_cycles

    def _replay_stage(self, stage: str, duration: int):
        """特定回放阶段"""
        # 选择性回放：基于重要性和上下文
        episodes_to_replay = self._select_episodes_for_replay(
            stage, duration
        )

        for episode in episodes_to_replay:
            self._replay_episode(episode, stage)

    def _select_episodes_for_replay(self, stage: str,
                                   duration: int) -> list:
        """
        选择回放情节

        基于上下文和重要性的选择机制
        """
        if not self.hippocampus['episodes']:
            return []

        # 计算选择概率
        num_episodes = len(self.hippocampus['episodes'])
        probabilities = np.zeros(num_episodes)

        for i, episode in enumerate(self.hippocampus['episodes']):
            # 基础概率：重要性
            prob = episode.get('importance', 0.5)

            # 上下文增强：相似情节更容易一起回放
            context = episode.get('context', '')
            if context in self.context_model:
                # 这个上下文最近被访问过，增加回放概率
                prob *= (1.0 + self.context_model[context])

            # 阶段调节
            if stage == 'NREM':
                # NREM倾向于回放近期、重要的记忆
                recency = 1.0 - (i / num_episodes)
                prob *= (1.0 + recency)
            elif stage == 'REM':
                # REM倾向于回放远期、抽象的记忆
                recency = i / num_episodes
                prob *= (1.0 + recency * 0.5)

            probabilities[i] = prob

        # 归一化
        probabilities = probabilities / (probabilities.sum() + 1e-10)

        # 采样
        num_to_replay = min(duration * 2, num_episodes)
        selected_indices = np.random.choice(
            num_episodes, size=num_to_replay, replace=False,
            p=probabilities
        )

        return [self.hippocampus['episodes'][i] for i in selected_indices]

    def _replay_episode(self, episode: dict, stage: str):
        """回放单个情节"""
        # 记录回放
        self.replay_count += 1
        self.replay_history.append({
            'episode_timestamp': episode.get('timestamp'),
            'stage': stage,
            'importance': episode.get('importance', 0.5),
        })

        # 提取模式
        entities = episode.get('entities', [])
        relations = episode.get('relations', [])

        if len(entities) >= 2:
            # 创建模式（简化）
            pattern = tuple(entities[:3])
            self.neocortex['patterns'][pattern] += 1

        # 更新图式
        if entities and relations:
            schema_id = entities[0]  # 简化：以第一个实体为图式ID
            self.neocortex['schemas'][schema_id].extend(relations)

        # 更新上下文模型
        context = episode.get('context', '')
        if context:
            if context not in self.context_model:
                self.context_model[context] = 0.0
            self.context_model[context] *= 0.9  # 衰减

    def awake_replay(self, context_query: str = None):
        """
        清醒回放：支持决策

        当需要回忆或决策时，进行快速回放
        """
        self.current_stage = 'Awake'

        if context_query:
            # 上下文驱动的回放
            relevant_episodes = []
            for episode in self.hippocampus['episodes']:
                context = episode.get('context', '')
                if context_query in context or context in context_query:
                    relevant_episodes.append(episode)

            # 回放相关情节
            for episode in relevant_episodes[:5]:
                self._replay_episode(episode, 'Awake')
        else:
            # 随机回放近期情节
            recent = self.hippocampus['episodes'][-10:]
            for episode in recent:
                self._replay_episode(episode, 'Awake')

    def recall(self, query_entity: str, top_k: int = 5) -> list:
        """回忆相关情节"""
        if query_entity not in self.hippocampus['index']:
            return []

        indices = self.hippocampus['index'][query_entity]
        results = []
        for idx in indices[-top_k:]:
            if idx < len(self.hippocampus['episodes']):
                results.append(self.hippocampus['episodes'][idx])

        return results

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'hippocampal_episodes': len(self.hippocampus['episodes']),
            'neocortical_patterns': len(self.neocortex['patterns']),
            'neocortical_schemas': len(self.neocortex['schemas']),
            'consolidation_cycles': self.consolidation_count,
            'replay_count': self.replay_count,
            'current_stage': self.current_stage,
        }


class SleepWakeScheduler:
    """
    睡眠-觉醒调度器

    调节睡眠和清醒周期，优化学习与巩固
    """

    def __init__(self, sleep_interval=50):
        self.sleep_interval = sleep_interval
        self.learn_count = 0
        self.is_sleeping = False

    def tick(self):
        """每次学习后调用"""
        self.learn_count += 1

        # 检查是否需要睡眠
        if self.learn_count >= self.sleep_interval and not self.is_sleeping:
            self.is_sleeping = True
            self.learn_count = 0
            return True  # 应该睡眠

        return False

    def wake(self):
        """醒来"""
        self.is_sleeping = False


if __name__ == '__main__':
    print("=== 高级睡眠巩固系统 ===")
    print()
    print("基于2024年最新研究：")
    print("1. 睡眠微观结构组织记忆回放")
    print("2. 上下文驱动的记忆重激活")
    print("3. 清醒时回放支持决策")
    print("4. 海马电路平衡记忆重激活")
    print()
    print("核心特性：")
    print("- 海马-皮层互补学习")
    print("- 选择性回放（重要性+上下文）")
    print("- 多阶段睡眠周期（NREM/REM/Awake）")
    print("- 容量管理与选择性遗忘")
