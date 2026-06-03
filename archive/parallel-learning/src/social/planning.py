"""合作规划 — 角色分配与顺序任务分解

多步骤联合任务需要角色标记 (role markers) 和顺序标记 (sequence markers)。
角色标记如 'you_push'/'i_carry' 自然涌现于合作规划过程。
顺序标记如 'first'/'then'/'together' 用于协调任务时序。

核心组件:
  CooperativePlanner — 多 Agent 合作规划器

所有数值计算使用 torch.Tensor，零 numpy 依赖。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.core.device import get_device

# 角色标记词汇表
ROLE_MARKERS = {
    'you_push', 'i_push', 'you_carry', 'i_carry',
    'you_guide', 'i_guide', 'you_lift', 'i_lift',
}

# 顺序标记词汇表
SEQUENCE_MARKERS = {'first', 'then', 'together', 'after', 'before'}


@dataclass
class TaskStep:
    """单个任务步骤"""
    role: str              # 角色标记
    action: str            # 动作描述
    target: str            # 目标对象
    required_agents: int = 1  # 所需 Agent 数量


@dataclass
class CooperativeTask:
    """合作任务定义"""
    task_id: str
    steps: List[TaskStep]
    required_agents: int = 2
    difficulty: int = 1   # 难度等级 1-5


class CooperativePlanner:
    """多 Agent 合作规划器

    核心能力:
    1. decompose_task — 将复杂任务分解为可并行的子步骤组
    2. assign_roles   — 将子步骤分配给各 Agent
    3. generate_plan_utterance — 生成带角色/顺序标记的规划话语
    4. evaluate_plan  — 评估规划质量

    Args:
        max_agents: 最大参与 Agent 数，默认 4
    """

    def __init__(self, max_agents: int = 4):
        self._device = get_device()
        self.max_agents = max_agents

        # 历史规划记录
        self._plan_history: List[Dict] = []
        self._plan_count = 0

        # 角色能力评估: {agent_id: {role: success_rate}}
        self._agent_capabilities: Dict[str, Dict[str, float]] = {}

    def decompose_task(self, task: CooperativeTask,
                       num_agents: int) -> List[List[TaskStep]]:
        """将任务分解为可分配给各 Agent 的子步骤组

        分解策略:
        - 如果步骤的 required_agents == num_agents → 所有 Agent 一起执行
        - 如果步骤可以独立完成 → 分配给单个 Agent
        - 尽量均衡各 Agent 的负载

        Args:
            task: 合作任务
            num_agents: 参与 Agent 数量

        Returns:
            子步骤组列表，长度 <= num_agents
        """
        if num_agents <= 0:
            return []

        # 按依赖关系分组: 独立步骤尽量分散，联合步骤放在一起
        groups: List[List[TaskStep]] = [[] for _ in range(min(num_agents, self.max_agents))]

        for i, step in enumerate(task.steps):
            if step.required_agents >= num_agents:
                # 所有 Agent 都需要参与的步骤: 分配到所有组
                for g in groups:
                    g.append(step)
            else:
                # 独立步骤: 轮询分配
                target_group = i % len(groups)
                groups[target_group].append(step)

        # 过滤空组
        return [g for g in groups if g]

    def assign_roles(self, steps: List[TaskStep],
                     agents: List[str]) -> Dict[str, List[TaskStep]]:
        """将步骤分配给各 Agent，考虑角色匹配

        策略:
        - 优先匹配 Agent 历史成功率最高的角色
        - 如果无历史数据，轮询分配

        Args:
            steps: 待分配的步骤列表
            agents: 可用 Agent ID 列表

        Returns:
            {agent_id: [assigned_steps]} 字典
        """
        if not agents or not steps:
            return {}

        assignment: Dict[str, List[TaskStep]] = {aid: [] for aid in agents}

        # 为每个 Agent 找到最佳角色匹配
        agent_scores = []
        for aid in agents:
            caps = self._agent_capabilities.get(aid, {})
            scores = torch.zeros(len(ROLE_MARKERS), dtype=torch.float32,
                                device=self._device)
            role_list = sorted(ROLE_MARKERS)
            for j, role in enumerate(role_list):
                scores[j] = caps.get(role, 0.3)  # 默认能力 0.3
            agent_scores.append(scores)

        for i, step in enumerate(steps):
            # 找到负载最轻且角色匹配的 Agent
            best_agent = None
            best_score = -1.0

            for j, aid in enumerate(agents):
                # 负载惩罚: 已分配步骤越多，惩罚越大
                load = len(assignment[aid])
                load_penalty = 1.0 / (1.0 + load)

                # 角色匹配分数
                role_match = self._agent_capabilities.get(aid, {}).get(
                    step.role, 0.3
                )

                score = role_match * load_penalty
                if score > best_score:
                    best_score = score
                    best_agent = aid

            if best_agent is not None:
                assignment[best_agent].append(step)

        return assignment

    def generate_plan_utterance(self,
                                assignment: Dict[str, List[TaskStep]]) -> List[str]:
        """从角色分配生成带标记的规划话语

        输出格式: [seq_marker, role_marker, action, target, ...]
        例: ['first', 'i_push', 'push', 'rock', 'then', 'you_carry', 'carry', 'fruit']

        Args:
            assignment: {agent_id: [steps]} 角色分配

        Returns:
            规划话语符号列表
        """
        utterance: List[str] = []
        all_steps: List[Tuple[int, str, TaskStep]] = []

        # 展平并保留 Agent 信息和原始顺序
        for idx, (aid, steps) in enumerate(assignment.items()):
            for step_idx, step in enumerate(steps):
                all_steps.append((step_idx, aid, step))

        # 按步骤顺序排序
        all_steps.sort(key=lambda x: x[0])

        for seq_pos, (step_idx, aid, step) in enumerate(all_steps):
            # 顺序标记
            if seq_pos == 0:
                utterance.append('first')
            elif seq_pos == len(all_steps) - 1 and len(all_steps) > 1:
                utterance.append('together')
            else:
                utterance.append('then')

            # 角色标记 (根据 Agent 索引决定 you/i)
            # 简化: 第一个 Agent 用 'i_' 前缀，其余用 'you_' 前缀
            agent_list = list(assignment.keys())
            if aid == agent_list[0]:
                role_prefix = 'i_'
            else:
                role_prefix = 'you_'

            role_marker = role_prefix + step.action
            if role_marker not in ROLE_MARKERS:
                role_marker = step.role  # 回退到原始角色
            utterance.append(role_marker)

            # 动作与目标
            utterance.append(step.action)
            utterance.append(step.target)

        return utterance

    def evaluate_plan(self, plan: List[str],
                      task: CooperativeTask) -> float:
        """评估规划质量

        评分维度:
        1. 完整性: 所有步骤是否被覆盖 (0-0.4)
        2. 标记使用: 角色/顺序标记是否恰当 (0-0.3)
        3. 角色均衡: 任务分配是否均衡 (0-0.3)

        Args:
            plan: 规划话语
            task: 原始任务

        Returns:
            规划质量 [0, 1]
        """
        plan_set = set(plan)

        # 1. 完整性: 任务步骤中的 target 和 action 是否出现在 plan 中
        task_targets = {s.target for s in task.steps}
        task_actions = {s.action for s in task.steps}
        task_symbols = task_targets | task_actions

        if task_symbols:
            coverage = len(plan_set & task_symbols) / len(task_symbols)
        else:
            coverage = 1.0
        completeness_score = coverage * 0.4

        # 2. 标记使用
        role_count = len(plan_set & ROLE_MARKERS)
        seq_count = len(plan_set & SEQUENCE_MARKERS)

        expected_roles = min(len(task.steps), len(ROLE_MARKERS))
        role_score = min(role_count / max(expected_roles, 1), 1.0)
        seq_score = min(seq_count / max(len(task.steps), 1), 1.0)
        marker_score = (role_score * 0.5 + seq_score * 0.5) * 0.3

        # 3. 角色均衡: 'i_' 和 'you_' 标记数量是否接近
        i_roles = sum(1 for s in plan if s.startswith('i_'))
        you_roles = sum(1 for s in plan if s.startswith('you_'))
        total_roles = i_roles + you_roles

        if total_roles > 0:
            balance = 1.0 - abs(i_roles - you_roles) / total_roles
        else:
            balance = 1.0
        balance_score = balance * 0.3

        total = completeness_score + marker_score + balance_score
        return min(total, 1.0)

    def create_task_library(self) -> List[CooperativeTask]:
        """创建预设的合作任务库

        Returns:
            预定义的合作任务列表
        """
        library = [
            CooperativeTask(
                task_id='move_boulder',
                steps=[
                    TaskStep('i_push', 'push', 'boulder', 1),
                    TaskStep('you_guide', 'guide', 'path', 1),
                    TaskStep('together', 'lift', 'boulder', 2),
                ],
                required_agents=2,
                difficulty=2,
            ),
            CooperativeTask(
                task_id='gather_fruit',
                steps=[
                    TaskStep('i_lift', 'lift', 'branch', 1),
                    TaskStep('you_carry', 'carry', 'fruit', 1),
                    TaskStep('together', 'sort', 'fruit', 2),
                ],
                required_agents=2,
                difficulty=1,
            ),
            CooperativeTask(
                task_id='build_shelter',
                steps=[
                    TaskStep('i_carry', 'carry', 'wood', 1),
                    TaskStep('you_carry', 'carry', 'leaves', 1),
                    TaskStep('i_push', 'push', 'logs', 1),
                    TaskStep('you_guide', 'guide', 'placement', 1),
                    TaskStep('together', 'lift', 'roof', 2),
                ],
                required_agents=3,
                difficulty=4,
            ),
            CooperativeTask(
                task_id='cross_river',
                steps=[
                    TaskStep('i_push', 'push', 'log', 1),
                    TaskStep('you_guide', 'guide', 'crossing', 1),
                    TaskStep('together', 'lift', 'log', 2),
                ],
                required_agents=2,
                difficulty=3,
            ),
            CooperativeTask(
                task_id='hunt_game',
                steps=[
                    TaskStep('i_guide', 'guide', 'prey', 1),
                    TaskStep('you_push', 'push', 'trap', 1),
                    TaskStep('i_carry', 'carry', 'bait', 1),
                    TaskStep('together', 'lift', 'trap', 2),
                ],
                required_agents=3,
                difficulty=5,
            ),
        ]
        return library

    def record_plan_result(self, task_id: str, plan: List[str],
                           success: bool) -> None:
        """记录规划执行结果，用于更新能力评估

        Args:
            task_id: 任务 ID
            plan: 规划话语
            success: 执行是否成功
        """
        self._plan_count += 1
        self._plan_history.append({
            'task_id': task_id,
            'plan': plan,
            'success': success,
        })

        # 限制历史记录长度
        if len(self._plan_history) > 200:
            self._plan_history = self._plan_history[-200:]

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'max_agents': self.max_agents,
            'plan_history': self._plan_history,
            'plan_count': self._plan_count,
            'agent_capabilities': self._agent_capabilities,
        }

    def load_state(self, state: dict) -> None:
        self.max_agents = state.get('max_agents', 4)
        self._plan_history = state.get('plan_history', [])
        self._plan_count = state.get('plan_count', 0)
        self._agent_capabilities = state.get('agent_capabilities', {})
