"""
自适应模型选择 v2 实验

对比：
A. 固定线性模型
B. 固定神经网络
C. 自适应选择（基于预测误差 + 迟滞 + 冷却期）

v2 改进：
- 切换信号：预测误差（而非环境复杂度）
- 迟滞机制：上阈值 0.15 / 下阈值 0.08
- 冷却期：100 步
- 知识迁移：只迁移符号知识，不迁移跨架构权重
"""

import sys
import numpy as np
from typing import Dict

sys.stdout.reconfigure(encoding='utf-8')

from environment_rich import create_rich_world
from agent import LearningAgent
from teacher import SimpleTeacher


def create_complex_world():
    """创建更复杂的环境（比 rich_world 更多物体和交互）"""
    from environment import SimpleGridWorld, Object
    env = SimpleGridWorld(15, 15)

    # 12 个物体，4 种颜色，3 种形状，不同重量
    objects = [
        Object(0, 2, 2, 'red', 'circle', 1.0),
        Object(1, 4, 8, 'red', 'square', 1.5),
        Object(2, 6, 3, 'red', 'triangle', 0.8),
        Object(3, 8, 10, 'blue', 'square', 1.5),
        Object(4, 10, 5, 'blue', 'triangle', 0.8),
        Object(5, 12, 2, 'blue', 'circle', 1.2),
        Object(6, 3, 12, 'green', 'triangle', 0.8),
        Object(7, 7, 7, 'green', 'circle', 1.2),
        Object(8, 11, 11, 'green', 'square', 1.0),
        Object(9, 1, 6, 'yellow', 'circle', 1.2),
        Object(10, 5, 14, 'yellow', 'square', 1.0),
        Object(11, 9, 1, 'yellow', 'triangle', 1.5),
    ]
    for obj in objects:
        env.add_object(obj)
    return env


class CurriculumEnvironment:
    """
    渐进复杂度环境

    模拟真实学习场景：从简单到复杂。
    前 300 步只有 3 个物体，后 500 步增加到 12 个。
    这是自适应模型应该闪耀的场景。
    """

    def __init__(self):
        from environment import SimpleGridWorld, Object
        self.env = SimpleGridWorld(15, 15)
        self.step_count = 0
        self.phase = 'simple'

        # 简单阶段的物体（3 个）
        self.simple_objects = [
            Object(0, 5, 5, 'red', 'circle', 1.0),
            Object(1, 10, 10, 'blue', 'square', 1.5),
            Object(2, 3, 12, 'green', 'triangle', 0.8),
        ]

        # 复杂阶段的物体（12 个）
        self.complex_objects = [
            Object(0, 2, 2, 'red', 'circle', 1.0),
            Object(1, 4, 8, 'red', 'square', 1.5),
            Object(2, 6, 3, 'red', 'triangle', 0.8),
            Object(3, 8, 10, 'blue', 'square', 1.5),
            Object(4, 10, 5, 'blue', 'triangle', 0.8),
            Object(5, 12, 2, 'blue', 'circle', 1.2),
            Object(6, 3, 12, 'green', 'triangle', 0.8),
            Object(7, 7, 7, 'green', 'circle', 1.2),
            Object(8, 11, 11, 'green', 'square', 1.0),
            Object(9, 1, 6, 'yellow', 'circle', 1.2),
            Object(10, 5, 14, 'yellow', 'square', 1.0),
            Object(11, 9, 1, 'yellow', 'triangle', 1.5),
        ]

        # 初始使用简单物体
        for obj in self.simple_objects:
            self.env.add_object(obj)

    def get_observation(self):
        return self.env.get_observation()

    def step(self, action):
        self.step_count += 1

        # 在步骤 300 时切换到复杂环境
        if self.step_count == 300 and self.phase == 'simple':
            self.phase = 'complex'
            # 清空并重新添加物体
            self.env.objects.clear()
            for obj in self.complex_objects:
                self.env.add_object(obj)
            self.env.reset()

        return self.env.step(action)

    def reset(self):
        self.env.reset()

    @property
    def objects(self):
        return self.env.objects

    @property
    def width(self):
        return self.env.width

    @property
    def height(self):
        return self.env.height


def run_experiment(model_type: str, num_steps: int = 800,
                   use_curriculum: bool = True) -> Dict:
    """运行单个配置的实验"""
    if use_curriculum:
        env = CurriculumEnvironment()
    else:
        env = create_complex_world()
    agent = LearningAgent(obs_dim=12, action_dim=5, model_type=model_type)
    teacher = SimpleTeacher()

    trajectory = {
        'prediction_errors': [],
        'symbols': [],
        'model_types': [],
        'switch_events': [],  # [(step, from, to, error), ...]
    }

    for step in range(num_steps):
        obs = env.get_observation()

        # 教师教学
        target = teacher.observe(obs)
        if target:
            teaching_action = teacher.decide_teaching_action(
                agent.get_stats(), target
            )
            if teaching_action:
                teacher.execute_teaching(teaching_action, target, agent)

        # 学习体行动
        action = agent.act(obs)
        next_obs, _, done = env.step(action)
        error = agent.learn_from_experience(obs, action, next_obs)

        # 记录
        trajectory['prediction_errors'].append(float(error))
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['model_types'].append(type(agent.predictive_model).__name__)

        if done:
            env.reset()

    # 收集切换事件
    if model_type == 'adaptive':
        switch_log = agent.error_selector.switch_log
        trajectory['switch_events'] = switch_log

    return trajectory


def main():
    """主函数"""
    print("自适应模型选择 v2 实验")
    print("对比：固定线性 / 固定神经网络 / 自适应（误差驱动）\n")

    configs = [
        ('固定线性', 'linear'),
        ('固定神经网络', 'neural_network'),
        ('自适应（误差驱动）', 'adaptive'),
    ]

    results = {}
    num_steps = 800
    num_runs = 3  # 多次运行取平均

    for name, model_type in configs:
        print(f"运行: {name} ({num_runs} 次)...")

        all_errors = []
        all_symbols = []
        all_switches = []

        for run in range(num_runs):
            trajectory = run_experiment(model_type, num_steps)
            final_error = np.mean(trajectory['prediction_errors'][-50:])
            all_errors.append(final_error)
            all_symbols.append(trajectory['symbols'][-1])
            all_switches.append(len(trajectory['switch_events']))

            if trajectory['switch_events'] and run == 0:
                print(f"  [run 0] 切换事件:")
                for evt in trajectory['switch_events']:
                    print(f"    步骤 {evt['step']}: {evt['from']} → {evt['to']} "
                          f"(误差={evt['avg_error']:.4f})")

        results[name] = {
            'final_error': float(np.mean(all_errors)),
            'final_error_std': float(np.std(all_errors)),
            'final_symbols': int(np.mean(all_symbols)),
            'switch_count': int(np.mean(all_switches)),
        }

        print(f"  最终误差: {np.mean(all_errors):.4f} ± {np.std(all_errors):.4f}")
        print(f"  符号数: {np.mean(all_symbols):.1f}")
        print(f"  切换次数: {np.mean(all_switches):.1f}")

    # 对比分析
    print(f"\n{'='*60}")
    print("对比分析")
    print(f"{'='*60}")

    print(f"\n{'模型':>16} | {'最终误差':>10} | {'符号数':>6} | {'切换次数':>8}")
    print(f"{'-'*16}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}")

    for name in results:
        r = results[name]
        print(f"{name:>16} | {r['final_error']:>10.4f} | {r['final_symbols']:>6} | "
              f"{r['switch_count']:>8}")

    # 判断自适应是否最优
    adaptive_error = results['自适应（误差驱动）']['final_error']
    best_fixed = min(results['固定线性']['final_error'],
                     results['固定神经网络']['final_error'])

    if adaptive_error < best_fixed:
        improvement = (best_fixed - adaptive_error) / best_fixed * 100
        print(f"\n✓ 自适应模型优于所有固定模型（改善 {improvement:.1f}%）")
    else:
        degradation = (adaptive_error - best_fixed) / best_fixed * 100
        print(f"\n✗ 自适应模型劣于固定模型（退化 {degradation:.1f}%）")

    # 保存结果
    import json
    output = {}
    for name, data in results.items():
        output[name] = {
            'final_error': data['final_error'],
            'final_symbols': data['final_symbols'],
            'switch_count': data['switch_count'],
        }

    with open('D:/mayAi/AILearning_v0527/mvl/adaptive_v2_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: adaptive_v2_comparison.json")


if __name__ == '__main__':
    main()
