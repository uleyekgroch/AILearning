"""
Phase 37 实验：指令执行闭环——从计划到物理执行 + 反馈

3 个实验：
1. 物理指令执行：Agent 在 3D 世界中实际执行 grab/push 指令
2. 指令协作：Speaker 指挥 Listener 抓取远处物体
3. 反馈学习：Speaker 根据执行反馈调整指令策略
"""

import sys
import random
import numpy as np
import json
from typing import Dict

sys.stdout.reconfigure(encoding='utf-8')

from multi_agent_3d_env import MultiAgent3DEnv
from agent_social import SocialAgent3D
from environment_3d import MATERIAL_NAMES
from instruction_grounding import (
    Instruction, InstructionGenerator, InstructionParser,
    InstructionFollower, InstructionGame, InstructionExecutor,
)
from language_emergence import EmergingLanguage


# ============================================================
# 实验 1：物理指令执行
# ============================================================

def experiment_1_physical_execution(num_trials: int = 50,
                                     verbose: bool = True) -> Dict:
    """
    物理指令执行实验

    Agent 在 3D 物理世界中实际执行 grab/push 指令。
    与 Phase 36 实验 1 的区别：这里测试的是物理执行，不是解析/规划。

    测量：物理执行成功率（grab 是否真的抓到了物体）
    预期：成功率 > 60%
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：物理指令执行")
        print("=" * 60)

    game = InstructionGame()

    grab_successes = 0
    push_successes = 0
    total_grab = 0
    total_push = 0
    navigation_failures = 0

    for trial in range(num_trials):
        # 创建环境
        env = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        env.physics.add_random_objects(4,
                                        shapes=['sphere', 'cube', 'cylinder'],
                                        materials=MATERIAL_NAMES[:4])

        # 添加 listener agent
        listener_id = env.add_agent(pos=np.array([5.0, 5.0, 1.5]))

        # 获取物体特征
        scene_objects = []
        for obj in env.physics.objects:
            features = env.physics.get_object_features(obj.id)
            features['id'] = obj.id
            features['position'] = obj.position.tolist()
            scene_objects.append(features)

        if not scene_objects:
            continue

        # 选择目标物体（偏好近的）
        target = min(scene_objects,
                     key=lambda o: np.linalg.norm(
                         np.array(o['position'])[:2] - np.array([5.0, 5.0])[:2]))

        for verb in ['grab', 'push']:
            # 重新创建环境（每次试验独立）
            env = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
            env.physics.add_random_objects(4,
                                            shapes=['sphere', 'cube', 'cylinder'],
                                            materials=MATERIAL_NAMES[:4])

            # 放置 listener 在固定位置
            listener_id = env.add_agent(pos=np.array([5.0, 5.0, 1.5]))

            # 重新获取物体
            scene_objects = []
            for obj in env.physics.objects:
                features = env.physics.get_object_features(obj.id)
                features['id'] = obj.id
                features['position'] = obj.position.tolist()
                scene_objects.append(features)

            if not scene_objects:
                continue

            target = min(scene_objects,
                         key=lambda o: np.linalg.norm(
                             np.array(o['position'])[:2] - np.array([5.0, 5.0])[:2]))

            # 生成指令
            gen = InstructionGenerator()
            parser = InstructionParser()
            follower = InstructionFollower()

            symbols = gen.generate_for_target(target, scene_objects, verb)
            instruction = parser.parse(symbols)

            if instruction is None:
                continue

            # 物理执行（闭环导航 + 交互）
            executor = InstructionExecutor()
            result = executor.execute_instruction(
                env, listener_id, instruction,
                scene_objects=scene_objects, other_agents=[]
            )

            if verb == 'grab':
                total_grab += 1
                if result['success']:
                    grab_successes += 1
            else:
                total_push += 1
                if result['success']:
                    push_successes += 1

        if verbose and (trial + 1) % 10 == 0:
            grab_sr = grab_successes / total_grab if total_grab > 0 else 0
            push_sr = push_successes / total_push if total_push > 0 else 0
            print(f"  Trial {trial+1:3d}: grab={grab_sr:.3f}, push={push_sr:.3f}, "
                  f"nav_fail={navigation_failures}")

    grab_sr = grab_successes / total_grab if total_grab > 0 else 0
    push_sr = push_successes / total_push if total_push > 0 else 0
    total_sr = (grab_successes + push_successes) / (total_grab + total_push) if (total_grab + total_push) > 0 else 0

    if verbose:
        print(f"\n物理指令执行结果:")
        print(f"  grab 成功率: {grab_sr:.3f} ({grab_successes}/{total_grab})")
        print(f"  push 成功率: {push_sr:.3f} ({push_successes}/{total_push})")
        print(f"  总成功率: {total_sr:.3f}")
        print(f"  导航失败: {navigation_failures}")

    return {
        'grab_success_rate': grab_sr,
        'push_success_rate': push_sr,
        'total_success_rate': total_sr,
        'navigation_failures': navigation_failures,
    }


# ============================================================
# 实验 2：指令协作
# ============================================================

def experiment_2_cooperative_instruction(num_trials: int = 40,
                                          verbose: bool = True) -> Dict:
    """
    指令协作实验

    Speaker 在远处，Listener 在物体附近。
    Speaker 生成指令，Listener 执行。
    测试协作是否比各自行动更有效。

    测量：协作成功率
    预期：成功率 > 50%
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：指令协作")
        print("=" * 60)

    game = InstructionGame()

    cooperative_successes = 0
    total = 0
    solo_successes = 0

    for trial in range(num_trials):
        # 创建环境
        env = MultiAgent3DEnv(bounds=(12.0, 12.0, 5.0))
        env.physics.add_random_objects(4,
                                        shapes=['sphere', 'cube', 'cylinder'],
                                        materials=MATERIAL_NAMES[:4])

        # Speaker 在左上角
        speaker_id = env.add_agent(pos=np.array([2.0, 2.0, 0.5]))
        # Listener 在右下角
        listener_id = env.add_agent(pos=np.array([10.0, 10.0, 0.5]))

        # 获取物体
        scene_objects = []
        for obj in env.physics.objects:
            features = env.physics.get_object_features(obj.id)
            features['id'] = obj.id
            features['position'] = obj.position.tolist()
            scene_objects.append(features)

        if not scene_objects:
            continue

        # 选择目标物体（在 listener 附近）
        listener_pos = env.agents[listener_id].pos
        target = min(scene_objects,
                     key=lambda o: np.linalg.norm(
                         np.array(o['position'])[:2] - listener_pos[:2]))

        # 协作：Speaker 指令 Listener
        gen = InstructionGenerator()
        parser = InstructionParser()
        follower = InstructionFollower()

        symbols = gen.generate_for_target(target, scene_objects, 'grab')
        instruction = parser.parse(symbols)

        if instruction:
            executor = InstructionExecutor()
            result = executor.execute_instruction(
                env, listener_id, instruction,
                scene_objects=scene_objects,
                other_agents=[speaker_id]
            )
            if result['success']:
                cooperative_successes += 1

        total += 1

        # 对比：Speaker 独自行动（从远处尝试抓取）
        env2 = MultiAgent3DEnv(bounds=(12.0, 12.0, 5.0))
        env2.physics.add_random_objects(4,
                                         shapes=['sphere', 'cube', 'cylinder'],
                                         materials=MATERIAL_NAMES[:4])
        solo_id = env2.add_agent(pos=np.array([2.0, 2.0, 0.5]))

        solo_objects = []
        for obj in env2.physics.objects:
            features = env2.physics.get_object_features(obj.id)
            features['id'] = obj.id
            features['position'] = obj.position.tolist()
            solo_objects.append(features)

        if solo_objects:
            solo_target = min(solo_objects,
                              key=lambda o: np.linalg.norm(
                                  np.array(o['position'])[:2] - np.array([2.0, 2.0])[:2]))
            solo_symbols = gen.generate_for_target(solo_target, solo_objects, 'grab')
            solo_instr = parser.parse(solo_symbols)
            if solo_instr:
                solo_exec = InstructionExecutor()
                solo_result = solo_exec.execute_instruction(
                    env2, solo_id, solo_instr,
                    scene_objects=solo_objects, other_agents=[]
                )
                if solo_result['success']:
                    solo_successes += 1

        if verbose and (trial + 1) % 10 == 0:
            coop_sr = cooperative_successes / total if total > 0 else 0
            solo_sr = solo_successes / total if total > 0 else 0
            print(f"  Trial {trial+1:3d}: coop={coop_sr:.3f}, solo={solo_sr:.3f}")

    coop_sr = cooperative_successes / total if total > 0 else 0
    solo_sr = solo_successes / total if total > 0 else 0

    if verbose:
        print(f"\n指令协作结果:")
        print(f"  协作成功率: {coop_sr:.3f} ({cooperative_successes}/{total})")
        print(f"  独自成功率: {solo_sr:.3f} ({solo_successes}/{total})")
        print(f"  协作优势: +{(coop_sr - solo_sr):.3f}")

    return {
        'cooperative_success_rate': coop_sr,
        'solo_success_rate': solo_sr,
        'advantage': coop_sr - solo_sr,
    }


# ============================================================
# 实验 3：反馈学习
# ============================================================

def experiment_3_feedback_learning(num_rounds: int = 60,
                                    verbose: bool = True) -> Dict:
    """
    反馈学习实验

    Speaker 根据执行反馈调整指令策略：
    - 如果 Listener 抓取失败，Speaker 尝试更详细的描述
    - 如果成功，Speaker 尝试更简洁的描述

    测量：指令质量（描述长度）和成功率的变化
    预期：成功率随轮次提升
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：反馈学习")
        print("=" * 60)

    gen = InstructionGenerator()
    parser = InstructionParser()
    follower = InstructionFollower()
    executor = InstructionExecutor()

    # 记录每轮结果
    log = {'rounds': [], 'success_rate': [], 'avg_symbols': []}
    recent_results = []  # 最近 10 轮的结果
    total_success = 0
    total_symbols = 0

    for round_num in range(num_rounds):
        # 创建环境
        env = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        env.physics.add_random_objects(4,
                                        shapes=['sphere', 'cube', 'cylinder'],
                                        materials=MATERIAL_NAMES[:4])

        listener_id = env.add_agent(pos=np.array([5.0, 5.0, 1.5]))

        scene_objects = []
        for obj in env.physics.objects:
            features = env.physics.get_object_features(obj.id)
            features['id'] = obj.id
            features['position'] = obj.position.tolist()
            scene_objects.append(features)

        if not scene_objects:
            continue

        target = random.choice(scene_objects)

        # 根据反馈调整描述策略
        if len(recent_results) >= 5:
            recent_sr = sum(recent_results[-5:]) / 5
            if recent_sr < 0.5:
                # 成功率低，使用更详细的描述（回退到所有特征）
                symbols = gen.generate_for_target(target, scene_objects, 'grab')
                # 强制添加更多描述
                for key, val in target.items():
                    if isinstance(val, str) and key not in ('id', 'distance'):
                        if val not in symbols:
                            symbols.append(val)
            else:
                # 成功率高，尝试更简洁的描述
                symbols = gen.generate_for_target(target, scene_objects, 'grab')
        else:
            symbols = gen.generate_for_target(target, scene_objects, 'grab')

        instruction = parser.parse(symbols)
        if instruction is None:
            continue

        result = executor.execute_instruction(
            env, listener_id, instruction,
            scene_objects=scene_objects, other_agents=[]
        )

        success = result['success']
        recent_results.append(1 if success else 0)
        if success:
            total_success += 1
        total_symbols += len(symbols)

        if (round_num + 1) % 10 == 0:
            sr = total_success / (round_num + 1)
            avg_sym = total_symbols / (round_num + 1)
            log['rounds'].append(round_num + 1)
            log['success_rate'].append(sr)
            log['avg_symbols'].append(avg_sym)

            if verbose:
                recent_sr = sum(recent_results[-10:]) / min(10, len(recent_results))
                print(f"  Round {round_num+1:3d}: overall={sr:.3f}, "
                      f"recent={recent_sr:.3f}, avg_symbols={avg_sym:.1f}")

    final_sr = total_success / num_rounds if num_rounds > 0 else 0
    final_avg_sym = total_symbols / num_rounds if num_rounds > 0 else 0

    # 计算前后对比
    early_sr = sum(recent_results[:10]) / 10 if len(recent_results) >= 10 else 0
    late_sr = sum(recent_results[-10:]) / 10 if len(recent_results) >= 10 else 0

    if verbose:
        print(f"\n反馈学习结果:")
        print(f"  总成功率: {final_sr:.3f}")
        print(f"  平均符号数: {final_avg_sym:.1f}")
        print(f"  前 10 轮成功率: {early_sr:.3f}")
        print(f"  后 10 轮成功率: {late_sr:.3f}")
        print(f"  提升: {(late_sr - early_sr):+.3f}")

    return {
        'total_success_rate': final_sr,
        'avg_symbols': final_avg_sym,
        'early_success_rate': early_sr,
        'late_success_rate': late_sr,
        'improvement': late_sr - early_sr,
        'log': log,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 37: 指令执行闭环")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：物理指令执行
    results['exp1'] = experiment_1_physical_execution(
        num_trials=50, verbose=True
    )

    # 实验 2：指令协作
    results['exp2'] = experiment_2_cooperative_instruction(
        num_trials=40, verbose=True
    )

    # 实验 3：反馈学习
    results['exp3'] = experiment_3_feedback_learning(
        num_rounds=60, verbose=True
    )

    # 保存结果
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))
        return obj

    with open('instruction_execution_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 instruction_execution_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 37 总结")
    print("=" * 60)
    print(f"  物理执行成功率: {results['exp1']['total_success_rate']:.3f}")
    print(f"  协作成功率: {results['exp2']['cooperative_success_rate']:.3f} vs "
          f"{results['exp2']['solo_success_rate']:.3f}")
    print(f"  反馈学习提升: {results['exp3']['improvement']:+.3f}")

    return results


if __name__ == '__main__':
    main()
