"""
Phase 28 实验：从婴儿到青少年的完整认知发展路径

3 个实验验证 8 阶段认知发展：
1. 完整发展轨迹（4000 步）
2. 阶段必要性对比
3. 认知里程碑涌现顺序
"""

import random
import sys
import json
import numpy as np
sys.path.insert(0, sys.path[0] if sys.path[0] else '.')

from environment import SimpleGridWorld, Object, create_simple_world
from agent import LearningAgent
from teacher import SimpleTeacher
from cognitive_tasks import run_stage_tasks, run_all_tasks, STAGE_TASKS


def create_rich_world() -> SimpleGridWorld:
    """创建丰富的测试世界（更多物体，更多交互机会）"""
    env = SimpleGridWorld(12, 12)

    # 多种材质的物体，提供丰富的感知经验
    objects = [
        Object(0, 2, 2, 'red', 'circle', 1.0, material='metal'),
        Object(1, 5, 5, 'blue', 'square', 1.5, material='stone'),
        Object(2, 7, 3, 'green', 'triangle', 0.8, material='fabric'),
        Object(3, 3, 7, 'yellow', 'circle', 1.2, material='wood'),
        Object(4, 9, 9, 'red', 'square', 0.5, material='glass'),
        Object(5, 1, 8, 'blue', 'triangle', 1.8, material='metal'),
        Object(6, 8, 1, 'green', 'circle', 0.6, material='fabric'),
        Object(7, 4, 10, 'yellow', 'square', 1.3, material='stone'),
        Object(8, 10, 4, 'red', 'triangle', 0.9, material='wood'),
        Object(9, 6, 6, 'blue', 'circle', 1.1, material='plastic'),
    ]

    for obj in objects:
        env.add_object(obj)
        obj.enrich_features()

    return env


def experiment_1_full_trajectory():
    """实验 1：完整发展轨迹（4000 步）"""
    print("=" * 60)
    print("实验 1: 完整发展轨迹（4000 步）")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    env = create_rich_world()
    agent = LearningAgent(obs_dim=12, action_dim=5)
    teacher = SimpleTeacher()

    # 记录
    stage_transitions = []
    milestone_records = []
    task_scores_history = []
    prev_stage = agent.development.current_stage

    for step in range(4000):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        # 教师教学
        target = teacher.observe(obs)
        teach_action = teacher.decide_teaching_action(agent.get_stats(), target)
        if teach_action and target:
            teacher.execute_teaching(teach_action, target, agent)

        # 检查阶段转换
        current_stage = agent.development.current_stage
        if current_stage != prev_stage:
            stage_transitions.append((step, current_stage))

            # 运行该阶段的认知任务
            tasks = run_stage_tasks(current_stage, agent, env)
            task_scores_history.append({
                'step': step,
                'stage': current_stage,
                'tasks': tasks,
            })

            print(f"\n  [步骤 {step}] 阶段转换: {prev_stage} -> {current_stage}")
            if tasks:
                for task_name, score in tasks.items():
                    print(f"    {task_name}: {score:.3f}")

            prev_stage = current_stage

        if done:
            env.reset()

        # 定期报告
        if step % 500 == 0:
            stage_info = agent.development.get_stage_info()
            print(f"  步骤 {step}: 阶段={stage_info['name']}, "
                  f"误差={error:.4f}, "
                  f"符号={len(agent.grounding.get_grounded_symbols())}, "
                  f"经验={len(agent.experiences)}")

    # 最终认知任务测试
    print(f"\n--- 最终认知任务测试 ---")
    final_tasks = run_all_tasks(agent, env)
    for stage, tasks in final_tasks.items():
        stage_name = DevelopmentEngine.STAGES.get(stage, {}).get('name', stage)
        print(f"  {stage_name} ({stage}):")
        for task_name, score in tasks.items():
            print(f"    {task_name}: {score:.3f}")

    # 最终报告
    print(f"\n--- 发展轨迹总结 ---")
    print(f"  最终阶段: {agent.development.get_stage_info()['name']}")
    print(f"  阶段转换次数: {len(stage_transitions)}")
    print(f"  总经验: {len(agent.experiences)}")
    print(f"  总符号: {len(agent.grounding.get_grounded_symbols())}")
    print(f"  阶段转换时间线:")
    for step, stage in stage_transitions:
        stage_name = DevelopmentEngine.STAGES.get(stage, {}).get('name', stage)
        print(f"    步骤 {step:>4d}: -> {stage_name}")

    return stage_transitions, task_scores_history, agent


def experiment_2_stage_necessity():
    """实验 2：阶段必要性对比"""
    print("\n" + "=" * 60)
    print("实验 2: 阶段必要性对比")
    print("=" * 60)

    results = {}

    # 条件 A：8 阶段渐进解锁（默认）
    print("\n--- 条件 A：8 阶段渐进解锁 ---")
    random.seed(42)
    np.random.seed(42)
    env_a = create_rich_world()
    agent_a = LearningAgent(obs_dim=12, action_dim=5)
    teacher_a = SimpleTeacher()

    for step in range(3000):
        obs = env_a.get_observation()
        action = agent_a.act(obs)
        next_obs, _, done = env_a.step(action)
        agent_a.learn_from_experience(obs, action, next_obs)
        target = teacher_a.observe(obs)
        teach_action = teacher_a.decide_teaching_action(agent_a.get_stats(), target)
        if teach_action and target:
            teacher_a.execute_teaching(teach_action, target, agent_a)
        if done:
            env_a.reset()

    results['A'] = {
        'final_stage': agent_a.development.current_stage,
        'final_error': float(np.mean(list(agent_a.predictive_model.error_history)[-50:])),
        'symbols': len(agent_a.grounding.get_grounded_symbols()),
        'stages_visited': len(set(s for s, _ in agent_a.development.stage_history)),
    }
    print(f"  最终阶段: {agent_a.development.get_stage_info()['name']}")
    print(f"  最终误差: {results['A']['final_error']:.4f}")
    print(f"  符号数: {results['A']['symbols']}")

    # 条件 B：跳过中间阶段（sensorimotor -> early_formal）
    print("\n--- 条件 B：跳过中间阶段 ---")
    random.seed(42)
    np.random.seed(42)
    env_b = create_rich_world()
    agent_b = LearningAgent(obs_dim=12, action_dim=5)
    # 跳到 early_formal
    agent_b.development.current_stage = 'early_formal'
    agent_b.development.stage_history = [('sensorimotor', 0), ('early_formal', 1)]
    teacher_b = SimpleTeacher()

    for step in range(3000):
        obs = env_b.get_observation()
        action = agent_b.act(obs)
        next_obs, _, done = env_b.step(action)
        agent_b.learn_from_experience(obs, action, next_obs)
        target = teacher_b.observe(obs)
        teach_action = teacher_b.decide_teaching_action(agent_b.get_stats(), target)
        if teach_action and target:
            teacher_b.execute_teaching(teach_action, target, agent_b)
        if done:
            env_b.reset()

    results['B'] = {
        'final_stage': agent_b.development.current_stage,
        'final_error': float(np.mean(list(agent_b.predictive_model.error_history)[-50:])),
        'symbols': len(agent_b.grounding.get_grounded_symbols()),
    }
    print(f"  最终阶段: {agent_b.development.get_stage_info()['name']}")
    print(f"  最终误差: {results['B']['final_error']:.4f}")
    print(f"  符号数: {results['B']['symbols']}")

    # 条件 C：无阶段限制（全部能力从开始解锁）
    print("\n--- 条件 C：无阶段限制 ---")
    random.seed(42)
    np.random.seed(42)
    env_c = create_rich_world()
    agent_c = LearningAgent(obs_dim=12, action_dim=5)
    agent_c.development.current_stage = 'adolescent'
    agent_c._get_available_actions = lambda: list(range(5))
    teacher_c = SimpleTeacher()

    for step in range(3000):
        obs = env_c.get_observation()
        action = agent_c.act(obs)
        next_obs, _, done = env_c.step(action)
        agent_c.learn_from_experience(obs, action, next_obs)
        target = teacher_c.observe(obs)
        teach_action = teacher_c.decide_teaching_action(agent_c.get_stats(), target)
        if teach_action and target:
            teacher_c.execute_teaching(teach_action, target, agent_c)
        if done:
            env_c.reset()

    results['C'] = {
        'final_stage': agent_c.development.current_stage,
        'final_error': float(np.mean(list(agent_c.predictive_model.error_history)[-50:])),
        'symbols': len(agent_c.grounding.get_grounded_symbols()),
    }
    print(f"  最终阶段: {agent_c.development.get_stage_info()['name']}")
    print(f"  最终误差: {results['C']['final_error']:.4f}")
    print(f"  符号数: {results['C']['symbols']}")

    # 泛化测试
    print(f"\n--- 泛化测试 ---")
    test_env = create_simple_world()
    for name, agent in [('A', agent_a), ('B', agent_b), ('C', agent_c)]:
        errors = []
        for _ in range(30):
            obs = test_env.get_observation()
            action = agent.act(obs)
            next_obs, _, done = test_env.step(action)
            obs_vec = agent.perceive(obs)
            next_obs_vec = agent.perceive(next_obs)
            predicted = agent.predictive_model.predict(obs_vec, action)
            err = np.mean((predicted - next_obs_vec) ** 2)
            errors.append(err)
            if done:
                test_env.reset()
        results[name]['generalization_error'] = float(np.mean(errors))
        print(f"  条件 {name}: 泛化误差 = {results[name]['generalization_error']:.4f}")

    return results


def experiment_3_milestone_order():
    """实验 3：认知里程碑涌现顺序"""
    print("\n" + "=" * 60)
    print("实验 3: 认知里程碑涌现顺序（5 次运行）")
    print("=" * 60)

    milestone_order = []

    for run in range(5):
        random.seed(42 + run)
        np.random.seed(42 + run)

        env = create_rich_world()
        agent = LearningAgent(obs_dim=12, action_dim=5)
        teacher = SimpleTeacher()

        run_milestones = {}
        prev_stage = 'sensorimotor'

        for step in range(4000):
            obs = env.get_observation()
            action = agent.act(obs)
            next_obs, _, done = env.step(action)
            agent.learn_from_experience(obs, action, next_obs)

            target = teacher.observe(obs)
            teach_action = teacher.decide_teaching_action(agent.get_stats(), target)
            if teach_action and target:
                teacher.execute_teaching(teach_action, target, agent)

            current_stage = agent.development.current_stage
            if current_stage != prev_stage:
                if current_stage not in run_milestones:
                    run_milestones[current_stage] = step
                prev_stage = current_stage

            if done:
                env.reset()

        milestone_order.append(run_milestones)
        print(f"  运行 {run+1}: {len(run_milestones)} 个阶段转换")
        for stage, step in sorted(run_milestones.items(), key=lambda x: x[1]):
            stage_name = DevelopmentEngine.STAGES.get(stage, {}).get('name', stage)
            print(f"    {stage_name}: 步骤 {step}")

    # 分析一致性
    print(f"\n--- 涌现顺序一致性 ---")
    all_stages = set()
    for m in milestone_order:
        all_stages.update(m.keys())

    for stage in sorted(all_stages):
        steps = [m.get(stage, -1) for m in milestone_order if stage in m]
        if steps:
            avg_step = np.mean(steps)
            std_step = np.std(steps)
            stage_name = DevelopmentEngine.STAGES.get(stage, {}).get('name', stage)
            print(f"  {stage_name}: 平均步骤 {avg_step:.0f} +/- {std_step:.0f}")

    return milestone_order


if __name__ == '__main__':
    # 导入 DevelopmentEngine 用于阶段名查找
    from agent import DevelopmentEngine

    print("Phase 28: 从婴儿到青少年的完整认知发展路径实验")
    print("=" * 60)

    # 显示 8 阶段设计
    print("\n--- 8 阶段设计 ---")
    for stage_key, stage_info in DevelopmentEngine.STAGES.items():
        age = stage_info.get('age', '?')
        print(f"  {stage_info['name']} ({age}岁): {stage_info['description']}")
        print(f"    能力: {', '.join(stage_info['abilities'])}")

    # 运行实验
    transitions, task_history, final_agent = experiment_1_full_trajectory()
    necessity_results = experiment_2_stage_necessity()
    milestone_order = experiment_3_milestone_order()

    # 保存结果
    output = {
        'experiment_1': {
            'stage_transitions': [(int(s), st) for s, st in transitions],
            'task_scores': task_history,
        },
        'experiment_2': necessity_results,
        'experiment_3': [
            {k: int(v) for k, v in m.items()}
            for m in milestone_order
        ],
    }

    with open('D:/mayAi/AILearning_v0527/mvl/developmental_path_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("所有实验完成")
    print("结果已保存到: developmental_path_results.json")
    print("=" * 60)
