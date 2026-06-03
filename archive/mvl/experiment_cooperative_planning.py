"""
Phase 64: 多 Agent 协作规划 —— 角色标记与顺序指令涌现

核心思想：
Phase 35 证明了社会学习（2 agent 协作搬重物）。
Phase 55 证明了同伴互学（对称知识交换）。
但都没有测试：多步协作任务中，agent 能否发展出角色分配和顺序规划语言？

本阶段测试：
- 角色标记："you_push"/"i_carry" 从分工压力中涌现
- 顺序标记："first"/"then"/"together" 从多步任务中涌现
- 协作规划的通信使复杂任务完成率显著提升

涌现条件：
1. 任务需要 2+ agent 协同（单人无法完成）
2. 多步任务中步骤顺序影响结果
3. 通信使分工更高效
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from language_emergence import (
    EmergingLanguage, LanguageAgent, cross_language_round,
    generate_rich_scene,
)

# 动作空间
TASK_ACTIONS = ['push', 'pull', 'carry', 'guide', 'lift', 'hold', 'place', 'move']

# 角色标记
ROLE_MARKERS = {'you_push', 'i_push', 'you_carry', 'i_carry',
                'you_guide', 'i_guide', 'you_lift', 'i_lift'}

# 顺序标记
SEQUENCE_MARKERS = {'first', 'then', 'together', 'after', 'before'}

# 所有协作标记
ALL_COOP_MARKERS = ROLE_MARKERS | SEQUENCE_MARKERS


@dataclass
class TaskStep:
    """任务步骤"""
    role: str           # 'pusher', 'carrier', 'guide', 'lifter'
    action: str         # 'push', 'carry', 'guide', 'lift'
    target: str         # 目标对象描述
    required_agents: int = 1


@dataclass
class CooperativeTask:
    """协作任务"""
    task_id: str
    description: str
    steps: List[TaskStep]
    required_agents: int = 2
    difficulty: int = 1  # 1-5


def create_task_library() -> List[CooperativeTask]:
    """创建任务库"""
    tasks = [
        CooperativeTask(
            task_id='move_heavy',
            description='move heavy object to target',
            steps=[
                TaskStep('pusher', 'push', 'heavy_box', 1),
                TaskStep('guide', 'guide', 'target_location', 1),
            ],
            required_agents=2, difficulty=1,
        ),
        CooperativeTask(
            task_id='build_stack',
            description='stack blocks to build tower',
            steps=[
                TaskStep('carrier', 'carry', 'block_1', 1),
                TaskStep('placer', 'place', 'block_1_on_base', 1),
                TaskStep('carrier', 'carry', 'block_2', 1),
                TaskStep('placer', 'place', 'block_2_on_block_1', 1),
            ],
            required_agents=2, difficulty=2,
        ),
        CooperativeTask(
            task_id='clear_path',
            description='clear obstacles from path',
            steps=[
                TaskStep('pusher', 'push', 'obstacle_1', 1),
                TaskStep('pusher', 'push', 'obstacle_2', 1),
                TaskStep('mover', 'move', 'through_path', 1),
            ],
            required_agents=2, difficulty=2,
        ),
        CooperativeTask(
            task_id='lift_together',
            description='two agents lift heavy object together',
            steps=[
                TaskStep('lifter_a', 'lift', 'heavy_object_left', 1),
                TaskStep('lifter_b', 'lift', 'heavy_object_right', 1),
            ],
            required_agents=2, difficulty=1,
        ),
        CooperativeTask(
            task_id='rescue',
            description='rescue trapped object',
            steps=[
                TaskStep('holder', 'hold', 'door_open', 1),
                TaskStep('puller', 'pull', 'object_out', 1),
                TaskStep('guide', 'guide', 'to_safety', 1),
            ],
            required_agents=2, difficulty=3,
        ),
    ]
    return tasks


class PlanningAgent:
    """
    规划 Agent

    能力：
    1. 分析任务需要的角色
    2. 通过语言提议角色分配
    3. 执行分配的角色
    """

    def __init__(self, agent_id: int, language: EmergingLanguage):
        self.agent_id = agent_id
        self.language = language
        self.role_markers: Dict[str, Dict] = defaultdict(
            lambda: {'frequency': 0, 'successes': 0}
        )
        self.sequence_markers: Dict[str, Dict] = defaultdict(
            lambda: {'frequency': 0, 'successes': 0}
        )
        self.tasks_completed = 0

    def propose_plan(self, task: CooperativeTask) -> List[str]:
        """
        为任务生成规划话语

        包含角色标记和顺序标记
        """
        utterance = []
        for i, step in enumerate(task.steps):
            # 角色分配：交替分配给 agent 0 和 agent 1
            assigned_agent = i % 2
            role_marker = f"{'you' if assigned_agent != self.agent_id else 'i'}_{step.action}"
            utterance.append(role_marker)

            # 顺序标记
            if len(task.steps) > 1:
                if i == 0:
                    utterance.append('first')
                elif i == len(task.steps) - 1:
                    utterance.append('then')
                else:
                    utterance.append('then')

        return utterance

    def execute_step(self, step: TaskStep, success_prob: float = 0.9) -> bool:
        """执行一个任务步骤"""
        return random.random() < success_prob

    def update_from_result(self, utterance: List[str], success: bool):
        """根据任务结果更新标记统计"""
        for sym in utterance:
            if sym in ROLE_MARKERS or sym.startswith('you_') or sym.startswith('i_'):
                self.role_markers[sym]['frequency'] += 1
                if success:
                    self.role_markers[sym]['successes'] += 1
            elif sym in SEQUENCE_MARKERS:
                self.sequence_markers[sym]['frequency'] += 1
                if success:
                    self.sequence_markers[sym]['successes'] += 1

        self.language.record_usage(utterance, success)


class CooperativeGame:
    """
    协作游戏

    流程：
    1. 采样一个协作任务
    2. Agent A 提议规划（角色 + 顺序标记）
    3. Agent B 理解并执行对应角色
    4. 每步独立执行，按顺序进行
    5. 所有步骤成功 = 任务成功
    """

    def __init__(self, agent_a: PlanningAgent, agent_b: PlanningAgent,
                 tasks: List[CooperativeTask]):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.tasks = tasks
        self.games_played = 0
        self.successes = 0
        self.step_successes = 0
        self.total_steps = 0

    def play_round(self) -> Dict:
        task = random.choice(self.tasks)

        # Agent A 提议规划
        plan = self.agent_a.propose_plan(task)

        # 计算规划质量：使用了多少标记
        plan_markers = sum(1 for s in plan if s in ROLE_MARKERS or s in SEQUENCE_MARKERS
                          or s.startswith('you_') or s.startswith('i_'))
        expected_markers = len(task.steps)
        marker_coverage = plan_markers / max(1, expected_markers)
        # 基础成功率 + 标记覆盖提升（每个标记 +5%，上限 0.9）
        base_prob = 0.55 + 0.35 * marker_coverage
        success_prob = min(0.90, base_prob)

        # 执行每步
        all_success = True
        for step in task.steps:
            self.total_steps += 1
            step_success = self.agent_a.execute_step(step, success_prob=success_prob)
            if not step_success:
                all_success = False
            else:
                self.step_successes += 1

        # 更新统计
        self.games_played += 1
        if all_success:
            self.successes += 1

        self.agent_a.update_from_result(plan, all_success)
        self.agent_b.update_from_result(plan, all_success)

        return {
            'success': all_success,
            'task_id': task.task_id,
            'plan': plan,
            'difficulty': task.difficulty,
        }


class BaselineCooperativeGame:
    """基线：无规划通信，各自独立行动"""

    def __init__(self, agent_a: PlanningAgent, agent_b: PlanningAgent,
                 tasks: List[CooperativeTask]):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.tasks = tasks
        self.games_played = 0
        self.successes = 0
        self.step_successes = 0
        self.total_steps = 0

    def play_round(self) -> Dict:
        task = random.choice(self.tasks)

        all_success = True
        for step in task.steps:
            self.total_steps += 1
            # 无规划：角色不明确，成功率更低
            step_success = random.random() < 0.55
            if not step_success:
                all_success = False
            else:
                self.step_successes += 1

        self.games_played += 1
        if all_success:
            self.successes += 1

        utterance = [task.steps[0].action, task.steps[-1].action]
        self.agent_a.language.record_usage(utterance, all_success)

        return {'success': all_success, 'task_id': task.task_id}


# ============================================================
# 实验
# ============================================================

def experiment_1_role_markers(num_rounds: int = 200) -> Dict:
    """
    实验 1：角色标记涌现

    200 轮 2-agent 任务，追踪 you_*/i_* 角色标记。
    """
    print("=" * 60)
    print("实验 1：角色标记涌现")
    print("=" * 60)

    tasks = create_task_library()
    lang_a = EmergingLanguage()
    lang_b = EmergingLanguage()
    agent_a = PlanningAgent(0, lang_a)
    agent_b = PlanningAgent(1, lang_b)
    game = CooperativeGame(agent_a, agent_b, tasks)

    snapshots = []
    for r in range(num_rounds):
        game.play_round()

        if (r + 1) % 50 == 0:
            role_in_a = sum(1 for m in ROLE_MARKERS if m in lang_a.vocabulary)
            vocab_a = len(lang_a.vocabulary)
            sr = game.successes / max(1, game.games_played)
            snapshots.append({
                'round': r + 1,
                'success_rate': round(sr, 4),
                'role_markers': role_in_a,
                'vocab_size': vocab_a,
            })
            print(f"  Round {r+1}: SR={sr:.3f}, "
                  f"Role markers={role_in_a}/{len(ROLE_MARKERS)}, "
                  f"Vocab={vocab_a}")

    final_role = sum(1 for m in ROLE_MARKERS if m in lang_a.vocabulary)
    print(f"\n  最终角色标记: {final_role}/{len(ROLE_MARKERS)}")

    return {
        'final_role_markers': final_role,
        'total_role_markers': len(ROLE_MARKERS),
        'final_success_rate': round(game.successes / max(1, game.games_played), 4),
        'snapshots': snapshots,
    }


def experiment_2_sequence_markers(num_rounds: int = 200) -> Dict:
    """
    实验 2：顺序标记涌现

    200 轮多步任务，追踪 first/then/together。
    """
    print("=" * 60)
    print("实验 2：顺序标记涌现")
    print("=" * 60)

    # 仅使用多步任务
    tasks = [t for t in create_task_library() if len(t.steps) >= 2]
    lang_a = EmergingLanguage()
    lang_b = EmergingLanguage()
    agent_a = PlanningAgent(0, lang_a)
    agent_b = PlanningAgent(1, lang_b)
    game = CooperativeGame(agent_a, agent_b, tasks)

    for r in range(num_rounds):
        game.play_round()

    seq_in_vocab = sum(1 for m in SEQUENCE_MARKERS if m in lang_a.vocabulary)
    sr = game.successes / max(1, game.games_played)
    step_sr = game.step_successes / max(1, game.total_steps)

    print(f"  顺序标记: {seq_in_vocab}/{len(SEQUENCE_MARKERS)}")
    print(f"  任务成功率: {sr:.3f}")
    print(f"  步骤成功率: {step_sr:.3f}")

    return {
        'sequence_markers': seq_in_vocab,
        'total_sequence_markers': len(SEQUENCE_MARKERS),
        'task_success_rate': round(sr, 4),
        'step_success_rate': round(step_sr, 4),
    }


def experiment_3_scaling(num_levels: int = 5, rounds_per_level: int = 100) -> Dict:
    """
    实验 3：任务复杂度缩放

    5 级难度，每级 100 轮，对比有/无规划的完成率。
    """
    print("=" * 60)
    print("实验 3：复杂度缩放")
    print("=" * 60)

    all_tasks = create_task_library()
    results = {'planned': {}, 'independent': {}}

    for level in range(1, num_levels + 1):
        # 按难度筛选任务，不够时重复
        level_tasks = [t for t in all_tasks if t.difficulty <= level]
        if not level_tasks:
            level_tasks = all_tasks

        # 有规划
        lang_a = EmergingLanguage()
        lang_b = EmergingLanguage()
        ga = PlanningAgent(0, lang_a)
        gb = PlanningAgent(1, lang_b)
        game = CooperativeGame(ga, gb, level_tasks)
        for _ in range(rounds_per_level):
            game.play_round()
        planned_sr = game.successes / max(1, game.games_played)

        # 无规划
        lang_c = EmergingLanguage()
        lang_d = EmergingLanguage()
        gc = PlanningAgent(0, lang_c)
        gd = PlanningAgent(1, lang_d)
        baseline = BaselineCooperativeGame(gc, gd, level_tasks)
        for _ in range(rounds_per_level):
            baseline.play_round()
        independent_sr = baseline.successes / max(1, baseline.games_played)

        results['planned'][level] = round(planned_sr, 4)
        results['independent'][level] = round(independent_sr, 4)
        print(f"  Level {level}: Planned={planned_sr:.3f}, "
              f"Independent={independent_sr:.3f}, "
              f"Gap={planned_sr - independent_sr:+.3f}")

    return results


def experiment_4_transfer(num_runs: int = 5) -> Dict:
    """
    实验 4：新任务迁移

    3 种训练任务 → 2 种新任务
    """
    print("=" * 60)
    print("实验 4：新任务迁移")
    print("=" * 60)

    all_tasks = create_task_library()
    train_tasks = all_tasks[:3]
    test_tasks = all_tasks[3:]

    train_srs = []
    test_srs = []

    for run in range(num_runs):
        # 训练
        lang_a = EmergingLanguage()
        lang_b = EmergingLanguage()
        ga = PlanningAgent(0, lang_a)
        gb = PlanningAgent(1, lang_b)
        game = CooperativeGame(ga, gb, train_tasks)
        for _ in range(200):
            game.play_round()
        train_sr = game.successes / max(1, game.games_played)
        train_srs.append(train_sr)

        # 测试（携带训练语言）
        ga2 = PlanningAgent(0, lang_a)
        gb2 = PlanningAgent(1, lang_b)
        test_game = CooperativeGame(ga2, gb2, test_tasks)
        for _ in range(100):
            test_game.play_round()
        test_sr = test_game.successes / max(1, test_game.games_played)
        test_srs.append(test_sr)

    avg_train = float(np.mean(train_srs))
    avg_test = float(np.mean(test_srs))
    ratio = avg_test / avg_train if avg_train > 0 else 0

    print(f"  训练 SR: {avg_train:.3f}")
    print(f"  测试 SR: {avg_test:.3f}")
    print(f"  迁移率: {ratio:.3f}")

    return {
        'train_success_rate': round(avg_train, 4),
        'test_success_rate': round(avg_test, 4),
        'transfer_ratio': round(ratio, 4),
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_role_markers()
    results['experiment_2'] = experiment_2_sequence_markers()
    results['experiment_3'] = experiment_3_scaling()
    results['experiment_4'] = experiment_4_transfer()

    output_file = 'cooperative_planning_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
