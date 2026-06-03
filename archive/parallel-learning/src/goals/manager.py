"""目标管理器 — 目标的创建、分解、进度追踪和状态管理"""

from typing import Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.goals.goal import Goal, GoalStatus
from src.goals.decomposer import GoalDecomposer
from src.goals.planner import LearningPlan, LearningPlanner


class GoalManager:
    """目标管理器

    统一管理学习目标的完整生命周期：
    创建 → 分解 → 规划 → 执行 → 追踪 → 完成。
    """

    def __init__(self, knowledge_graph: KnowledgeGraph, metacognition=None):
        self.goals: Dict[str, Goal] = {}
        self.decomposer = GoalDecomposer(knowledge_graph, metacognition)
        self.planner = LearningPlanner(knowledge_graph, metacognition)
        self._plans: Dict[str, LearningPlan] = {}
        self._counter = 0

    # ── 目标创建 ──────────────────────────────────────────────────

    def create_goal(self, description: str, priority: float = 0.5) -> Goal:
        """创建新目标"""
        self._counter += 1
        goal_id = f"goal_{self._counter}"
        goal = Goal(
            id=goal_id,
            description=description,
            status=GoalStatus.PENDING,
            priority=priority,
            created_step=self._counter,
        )

        # 自动识别所需知识
        goal.required_knowledge = self.decomposer.identify_required_knowledge(goal)

        self.goals[goal_id] = goal
        return goal

    # ── 分解与规划 ────────────────────────────────────────────────

    def decompose_and_plan(self, goal_id: str) -> LearningPlan:
        """分解目标并生成学习计划"""
        goal = self.goals.get(goal_id)
        if goal is None:
            raise ValueError(f"目标不存在: {goal_id}")

        # 分解为子目标
        sub_goals = self.decomposer.decompose_goal(goal)
        for sub in sub_goals:
            self.goals[sub.id] = sub
            goal.sub_goals.append(sub.id)

        # 更新父目标状态
        goal.status = GoalStatus.IN_PROGRESS

        # 生成学习计划
        plan = self.planner.plan_learning(goal)
        self._plans[goal_id] = plan

        return plan

    # ── 进度管理 ──────────────────────────────────────────────────

    def update_progress(self, goal_id: str, progress: float) -> None:
        """更新目标进度"""
        goal = self.goals.get(goal_id)
        if goal is None:
            return

        goal.progress = max(0.0, min(1.0, progress))

        if goal.progress >= 1.0:
            goal.status = GoalStatus.COMPLETED
        elif goal.progress > 0.0:
            goal.status = GoalStatus.IN_PROGRESS

        # 向上传播进度到父目标
        if goal.parent_goal:
            self._propagate_progress(goal.parent_goal)

    def _propagate_progress(self, parent_id: str) -> None:
        """将子目标进度传播到父目标"""
        parent = self.goals.get(parent_id)
        if parent is None or not parent.sub_goals:
            return

        sub_progresses = []
        for sub_id in parent.sub_goals:
            sub = self.goals.get(sub_id)
            if sub:
                sub_progresses.append(sub.progress)

        if sub_progresses:
            parent.progress = round(sum(sub_progresses) / len(sub_progresses), 4)
            if parent.progress >= 1.0:
                parent.status = GoalStatus.COMPLETED
            elif parent.progress > 0.0:
                parent.status = GoalStatus.IN_PROGRESS

    # ── 行动选择 ──────────────────────────────────────────────────

    def get_next_action(self) -> Optional[Dict]:
        """获取下一步应该执行的动作

        优先级：
        1. 有正在进行的学习计划 → 取计划的下一个未完成步骤
        2. 有 PENDING 目标 → 分解并规划
        3. 无目标 → None
        """
        # 1. 从现有计划中找下一步
        for goal_id, plan in self._plans.items():
            goal = self.goals.get(goal_id)
            if goal and goal.status == GoalStatus.IN_PROGRESS:
                for step in plan.steps:
                    if step.get('action') == 'execute':
                        continue
                    if not step.get('done', False):
                        return {
                            'goal_id': goal_id,
                            **step,
                        }

        # 2. 有 PENDING 目标
        pending = [g for g in self.goals.values()
                   if g.status == GoalStatus.PENDING]
        if pending:
            # 按优先级排序
            pending.sort(key=lambda g: g.priority, reverse=True)
            target = pending[0]
            plan = self.decompose_and_plan(target.id)
            if plan.steps:
                first = plan.steps[0]
                if not first.get('done', False):
                    return {
                        'goal_id': target.id,
                        **first,
                    }

        # 3. 没有可执行的动作
        return None

    # ── 查询 ──────────────────────────────────────────────────────

    def get_active_goals(self) -> List[Goal]:
        """获取所有活跃目标（PENDING 或 IN_PROGRESS）"""
        return [
            g for g in self.goals.values()
            if g.status in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS)
        ]

    def check_completion(self, goal_id: str) -> bool:
        """检查目标是否完成"""
        goal = self.goals.get(goal_id)
        if goal is None:
            return False

        # 检查子目标是否全部完成
        if goal.sub_goals:
            all_subs_done = all(
                self.goals.get(sub_id, Goal(id=sub_id, description='')).status == GoalStatus.COMPLETED
                for sub_id in goal.sub_goals
            )
            if all_subs_done:
                goal.status = GoalStatus.COMPLETED
                goal.progress = 1.0
                return True

        return goal.status == GoalStatus.COMPLETED

    # ── 持久化 ────────────────────────────────────────────────────

    def save_state(self) -> dict:
        """保存状态"""
        return {
            'goals': {gid: g.to_dict() for gid, g in self.goals.items()},
            'plans': {
                gid: {
                    'goal_id': p.goal_id,
                    'steps': p.steps,
                    'estimated_effort': p.estimated_effort,
                }
                for gid, p in self._plans.items()
            },
            'counter': self._counter,
        }

    def load_state(self, state: dict) -> None:
        """加载状态"""
        self.goals.clear()
        self._plans.clear()

        for gid, gdata in state.get('goals', {}).items():
            self.goals[gid] = Goal.from_dict(gdata)

        for gid, pdata in state.get('plans', {}).items():
            self._plans[gid] = LearningPlan(
                goal_id=pdata['goal_id'],
                steps=pdata.get('steps', []),
                estimated_effort=pdata.get('estimated_effort', 0.0),
            )

        self._counter = state.get('counter', 0)
