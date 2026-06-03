"""
三层记忆系统编排器

工作记忆 → 情景记忆 → 语义记忆 层级传递。
定期将工作记忆迁移到情景记忆，情景记忆巩固后提炼到语义记忆。
"""

from typing import Dict, List
import torch

from src.core.config import LearnerConfig
from src.core.device import get_device
from src.memory.working import WorkingMemory
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory


class MemorySystem:
    """三层记忆系统编排器"""

    def __init__(self, config: LearnerConfig):
        self.config = config
        self.device = get_device(config.device)
        dim = config.obs_dim

        self.working = WorkingMemory(
            capacity=config.working_memory_capacity,
            dim=dim,
            device=self.device,
        )
        self.episodic = EpisodicMemory(
            max_traces=config.episodic_memory_capacity,
            dim=dim,
            forgetting_rate=config.forgetting_rate,
            device=self.device,
        )
        self.semantic = SemanticMemory(
            dim=dim,
            device=self.device,
        )

        self._step_count: int = 0

    def store_experience(self, obs: torch.Tensor, action: torch.Tensor,
                         next_obs: torch.Tensor, reward: float,
                         error: float) -> None:
        """存入工作记忆，定期迁移到情景记忆"""
        metadata = {
            'action': action,
            'next_obs': next_obs,
            'reward': reward,
            'error': error,
            'step': self._step_count,
        }

        # 工作记忆满时，先迁移最旧项到情景记忆
        if len(self.working.items) >= self.config.working_memory_capacity:
            report = self.working.consolidate()
            for rep, meta in report['items']:
                self.episodic.store(rep, meta)

        self.working.store(obs, metadata)
        self._step_count += 1

        # 定期触发衰减
        if self._step_count % self.config.consolidation_interval == 0:
            self.episodic.decay()

    def retrieve_context(self, cue: torch.Tensor, k: int = 5) -> List[Dict]:
        """从三层记忆统一检索"""
        all_results = []

        # 工作记忆检索
        wm_results = self.working.retrieve(cue, k=k)
        for r in wm_results:
            r['source'] = 'working'
            all_results.append(r)

        # 情景记忆检索
        em_results = self.episodic.retrieve(cue, k=k)
        for r in em_results:
            r['source'] = 'episodic'
            all_results.append(r)

        # 语义记忆检索
        sm_results = self.semantic.retrieve(cue, k=k)
        for r in sm_results:
            r['source'] = 'semantic'
            all_results.append(r)

        # 按相似度统一排序，返回 top-k
        all_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        return all_results[:k]

    def consolidate_all(self) -> Dict:
        """工作记忆→情景记忆→语义记忆 全链路巩固"""
        report = {}

        # 1. 工作记忆 → 情景记忆
        wm_report = self.working.consolidate()
        for rep, meta in wm_report['items']:
            self.episodic.store(rep, meta)
        report['working'] = {
            'transferred': wm_report['transferred'],
        }

        # 2. 情景记忆巩固
        em_report = self.episodic.consolidate()
        report['episodic'] = em_report

        # 3. 情景记忆强痕迹 → 语义记忆概念提取
        for trace in self.episodic.traces:
            if trace['strength'] > 1.0:
                self.semantic.store(trace['repr'], trace['meta'])

        # 4. 语义记忆巩固
        sm_report = self.semantic.consolidate()
        report['semantic'] = sm_report

        return report
