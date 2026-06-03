"""
Phase 36 实验：语言 Grounding 深化

4 个实验：
1. 基本指令执行：固定指令，测量执行成功率
2. 指令 vs 无指令：有指令的协作 vs 无指令的随机探索
3. 指令涌现：自由通信，观察动词符号是否涌现
4. 复合指令：多步指令执行
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from multi_agent_3d_env import MultiAgent3DEnv
from agent_social import SocialAgent3D
from environment_3d import MATERIAL_NAMES
from instruction_grounding import (
    Instruction, InstructionGenerator, InstructionParser,
    InstructionFollower, InstructionGame,
)
from language_emergence import EmergingLanguage, ACTIONS


# ============================================================
# 实验 1：基本指令执行
# ============================================================

def experiment_1_basic_instruction(num_trials: int = 50,
                                    verbose: bool = True) -> Dict:
    """
    基本指令执行实验

    固定指令 "grab/push + 物体特征"，测量 Listener 的执行成功率。

    测量：指令解析成功率、目标识别成功率
    预期：成功率 > 70%
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：基本指令执行")
        print("=" * 60)

    game = InstructionGame()
    parser = InstructionParser()
    follower = InstructionFollower()
    gen = InstructionGenerator()

    grab_successes = 0
    push_successes = 0
    total_grab = 0
    total_push = 0
    parse_failures = 0

    for trial in range(num_trials):
        # 创建环境和物体
        env = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        env.physics.add_random_objects(5,
                                        shapes=['sphere', 'cube', 'cylinder'],
                                        materials=MATERIAL_NAMES[:4])

        # 获取物体特征
        scene_objects = []
        for obj in env.physics.objects:
            features = env.physics.get_object_features(obj.id)
            features['id'] = obj.id
            features['position'] = obj.position.tolist()
            features['distance'] = 0.0
            scene_objects.append(features)

        if not scene_objects:
            continue

        # 选择目标物体
        target = random.choice(scene_objects)
        target_id = target['id']

        # 生成指令
        for verb in ['grab', 'push']:
            symbols = gen.generate_for_target(target, scene_objects, verb)

            # 解析指令
            instruction = parser.parse(symbols)
            if instruction is None:
                parse_failures += 1
                continue

            # 执行指令（用 (0,0) 位置和 0 朝向，因为只测目标识别）
            actions = follower.plan_actions(
                instruction, np.array([0.0, 0.0, 0.5]), 0.0,
                scene_objects, held_object=None
            )

            # 评估：找到的目标是否正确
            found = follower._find_target(
                instruction.target_features, scene_objects
            )
            found_id = found.get('id') if found else None
            success = (found_id == target_id)

            if verb == 'grab':
                total_grab += 1
                if success:
                    grab_successes += 1
            else:
                total_push += 1
                if success:
                    push_successes += 1

        if verbose and (trial + 1) % 10 == 0:
            grab_sr = grab_successes / total_grab if total_grab > 0 else 0
            push_sr = push_successes / total_push if total_push > 0 else 0
            print(f"  Trial {trial+1:3d}: grab={grab_sr:.3f}, push={push_sr:.3f}")

    grab_sr = grab_successes / total_grab if total_grab > 0 else 0
    push_sr = push_successes / total_push if total_push > 0 else 0
    total_sr = (grab_successes + push_successes) / (total_grab + total_push) if (total_grab + total_push) > 0 else 0

    if verbose:
        print(f"\n基本指令执行结果:")
        print(f"  grab 成功率: {grab_sr:.3f} ({grab_successes}/{total_grab})")
        print(f"  push 成功率: {push_sr:.3f} ({push_successes}/{total_push})")
        print(f"  总成功率: {total_sr:.3f}")
        print(f"  解析失败: {parse_failures}")

    return {
        'grab_success_rate': grab_sr,
        'push_success_rate': push_sr,
        'total_success_rate': total_sr,
        'parse_failures': parse_failures,
    }


# ============================================================
# 实验 2：指令 vs 无指令
# ============================================================

def experiment_2_instruction_vs_random(num_trials: int = 100,
                                        verbose: bool = True) -> Dict:
    """
    指令 vs 无指令对比（目标识别准确率）

    场景：给定一组物体，选择一个目标。
    - 有指令：用指令描述定位目标（语义匹配）
    - 无指令：随机猜一个物体

    测量：目标识别准确率
    预期：有指令 >> 无指令
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：指令 vs 无指令（目标识别）")
        print("=" * 60)

    gen = InstructionGenerator()
    parser = InstructionParser()
    follower = InstructionFollower()

    instructed_hits = 0
    random_hits = 0
    total = 0

    for trial in range(num_trials):
        # 创建场景
        env = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        env.physics.add_random_objects(6,
                                        shapes=['sphere', 'cube', 'cylinder'],
                                        materials=MATERIAL_NAMES[:4])

        scene_objects = []
        for obj in env.physics.objects:
            features = env.physics.get_object_features(obj.id)
            features['id'] = obj.id
            scene_objects.append(features)

        if len(scene_objects) < 2:
            continue

        # 选择目标
        target = random.choice(scene_objects)
        target_id = target['id']

        # 有指令：生成指令，然后用指令匹配
        symbols = gen.generate_for_target(target, scene_objects, 'grab')
        instruction = parser.parse(symbols)

        if instruction:
            found = follower._find_target(
                instruction.target_features, scene_objects
            )
            found_id = found.get('id') if found else None
            if found_id == target_id:
                instructed_hits += 1

        # 无指令：随机猜
        guess = random.choice(scene_objects)
        if guess.get('id') == target_id:
            random_hits += 1

        total += 1

        if verbose and (trial + 1) % 25 == 0:
            inst_sr = instructed_hits / total if total > 0 else 0
            rand_sr = random_hits / total if total > 0 else 0
            print(f"  Trial {trial+1:3d}: instructed={inst_sr:.3f}, "
                  f"random={rand_sr:.3f}")

    inst_sr = instructed_hits / total if total > 0 else 0
    rand_sr = random_hits / total if total > 0 else 0

    if verbose:
        print(f"\n指令 vs 无指令结果:")
        print(f"  有指令准确率: {inst_sr:.3f} ({instructed_hits}/{total})")
        print(f"  无指令准确率: {rand_sr:.3f} ({random_hits}/{total})")
        print(f"  指令优势: +{(inst_sr - rand_sr):.3f}")

    return {
        'instructed_accuracy': inst_sr,
        'random_accuracy': rand_sr,
        'advantage': inst_sr - rand_sr,
    }


# ============================================================
# 实验 3：指令涌现
# ============================================================

def experiment_3_instruction_emergence(num_agents: int = 4,
                                        num_steps: int = 800,
                                        verbose: bool = True) -> Dict:
    """
    指令涌现实验

    多 Agent 在共享世界中自由通信。
    其中一些通信可以是指令性的（描述需要对方做的事）。

    测量：动词符号使用频率、指令成功率
    预期：动词符号从指令需求中涌现
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：指令涌现")
        print("=" * 60)

    env = MultiAgent3DEnv(bounds=(12.0, 12.0, 5.0), comm_range=3.0)
    env.physics.add_random_objects(6,
                                    shapes=['sphere', 'cube', 'cylinder'],
                                    materials=MATERIAL_NAMES[:4])

    agents = {}
    for i in range(num_agents):
        agent_id = env.add_agent()
        agents[agent_id] = SocialAgent3D(agent_id, use_language=True)

    gen = InstructionGenerator()
    parser = InstructionParser()

    log = {'steps': [], 'verb_count': [], 'instruction_success': []}
    verb_usage = {}  # verb -> count
    total_instructions = 0
    successful_instructions = 0
    all_symbols = set()

    for step in range(num_steps):
        # 每个 agent 选择动作
        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id] = agent.choose_action_social(epsilon=0.15)

        # 执行
        observations = env.step(actions)

        # 更新学习
        for agent_id, obs in observations.items():
            agents[agent_id].step_with_observation(obs, actions[agent_id])

        # 近距离 agent 之间尝试指令通信
        pairs = env.get_nearby_pairs()
        for id_a, id_b in pairs:
            agent_a = agents[id_a]
            agent_b = agents[id_b]

            visible = env.physics.get_nearby_objects(
                env.agents[id_a].pos, radius=3.0
            )
            if not visible:
                continue

            # Agent A 选择：描述性通信 or 指令性通信
            if np.random.random() < 0.5:
                # 描述性通信（原有方式）
                target = random.choice(visible)
                agent_a.try_communicate(agent_b, target)
            else:
                # 指令性通信
                target = random.choice(visible)
                verb = random.choice(['grab', 'push'])

                symbols = agent_a.generate_instruction(
                    target, visible, verb
                )

                # 记录符号
                for s in symbols:
                    all_symbols.add(s)

                # 解析指令
                instruction = parser.parse(symbols)
                if instruction:
                    # 评估：能否找到正确的物体
                    follower = InstructionFollower()
                    found = follower._find_target(
                        instruction.target_features, visible
                    )
                    target_id = target.get('id')
                    found_id = found.get('id') if found else None
                    success = (found_id == target_id)

                    total_instructions += 1
                    if success:
                        successful_instructions += 1

                    # 更新语言
                    if agent_a.language:
                        agent_a.language.record_usage(symbols, success)
                    if agent_b.language:
                        agent_b.language.record_usage(symbols, success)

                    # 记录动词使用
                    if verb not in verb_usage:
                        verb_usage[verb] = 0
                    verb_usage[verb] += 1

        if (step + 1) % 200 == 0:
            sr = successful_instructions / total_instructions if total_instructions > 0 else 0
            log['steps'].append(step + 1)
            log['verb_count'].append(len(verb_usage))
            log['instruction_success'].append(sr)

            if verbose:
                print(f"  Step {step+1:5d}: verbs={len(verb_usage)}, "
                      f"instructions={total_instructions}, success={sr:.3f}")

    sr = successful_instructions / total_instructions if total_instructions > 0 else 0

    if verbose:
        print(f"\n指令涌现结果:")
        print(f"  总符号数: {len(all_symbols)}")
        print(f"  动词使用: {verb_usage}")
        print(f"  总指令数: {total_instructions}")
        print(f"  指令成功率: {sr:.3f}")
        # 检查哪些动词符号进入了词汇表
        verb_in_vocab = []
        for agent in agents.values():
            if agent.language:
                for v in ACTIONS:
                    if v in agent.language.vocabulary:
                        verb_in_vocab.append(v)
        print(f"  词汇表中的动词: {sorted(set(verb_in_vocab))}")

    return {
        'log': log,
        'all_symbols': sorted(list(all_symbols)),
        'verb_usage': verb_usage,
        'total_instructions': total_instructions,
        'success_rate': sr,
        'verbs_in_vocab': sorted(set(verb_in_vocab)) if 'verb_in_vocab' in dir() else [],
    }


# ============================================================
# 实验 4：复合指令
# ============================================================

def experiment_4_compound_instruction(num_trials: int = 50,
                                       verbose: bool = True) -> Dict:
    """
    复合指令实验

    测试两步指令的解析和规划能力：
    Step 1: "grab X" — 识别目标并规划抓取
    Step 2: "throw" — 规划投掷

    测量：指令解析和动作规划的正确性
    预期：成功率 > 50%
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：复合指令")
        print("=" * 60)

    gen = InstructionGenerator()
    parser = InstructionParser()
    follower = InstructionFollower()

    step1_ok = 0  # grab 解析+规划成功
    step2_ok = 0  # throw 解析+规划成功
    both_ok = 0   # 两步都成功
    total = 0

    for trial in range(num_trials):
        # 创建场景
        env = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        env.physics.add_random_objects(5,
                                        shapes=['sphere', 'cube', 'cylinder'],
                                        materials=MATERIAL_NAMES[:4])

        scene_objects = []
        for obj in env.physics.objects:
            features = env.physics.get_object_features(obj.id)
            features['id'] = obj.id
            scene_objects.append(features)

        if not scene_objects:
            continue

        target = random.choice(scene_objects)
        target_id = target['id']

        # Step 1: 生成 grab 指令，检查解析和目标识别
        grab_symbols = gen.generate_for_target(target, scene_objects, 'grab')
        grab_instruction = parser.parse(grab_symbols)

        s1 = False
        if grab_instruction and grab_instruction.verb == 'grab':
            # 检查能否找到正确目标
            found = follower._find_target(
                grab_instruction.target_features, scene_objects
            )
            if found and found.get('id') == target_id:
                # 规划 grab 动作
                actions = follower.plan_actions(
                    grab_instruction, np.array([5.0, 5.0, 0.5]), 0.0,
                    scene_objects, held_object=None
                )
                if actions and follower.GRAB in actions:
                    s1 = True

        if s1:
            step1_ok += 1

        # Step 2: 生成 throw 指令
        throw_symbols = ['throw']
        throw_instruction = parser.parse(throw_symbols)

        s2 = False
        if throw_instruction and throw_instruction.verb == 'throw':
            actions = follower.plan_actions(
                throw_instruction, np.array([5.0, 5.0, 0.5]), 0.0,
                scene_objects, held_object=target_id
            )
            if actions and follower.THROW in actions:
                s2 = True

        if s2:
            step2_ok += 1

        if s1 and s2:
            both_ok += 1

        total += 1

        if verbose and (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1:3d}: grab={step1_ok/total:.3f}, "
                  f"throw={step2_ok/total:.3f}, both={both_ok/total:.3f}")

    s1_rate = step1_ok / total if total > 0 else 0
    s2_rate = step2_ok / total if total > 0 else 0
    both_rate = both_ok / total if total > 0 else 0

    if verbose:
        print(f"\n复合指令结果:")
        print(f"  grab 规划成功率: {s1_rate:.3f} ({step1_ok}/{total})")
        print(f"  throw 规划成功率: {s2_rate:.3f} ({step2_ok}/{total})")
        print(f"  两步都成功: {both_rate:.3f}")

    return {
        'grab_success_rate': s1_rate,
        'throw_success_rate': s2_rate,
        'compound_success_rate': both_rate,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 36: 语言 Grounding 深化")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：基本指令执行
    results['exp1'] = experiment_1_basic_instruction(
        num_trials=50, verbose=True
    )

    # 实验 2：指令 vs 无指令
    results['exp2'] = experiment_2_instruction_vs_random(
        num_trials=100, verbose=True
    )

    # 实验 3：指令涌现
    results['exp3'] = experiment_3_instruction_emergence(
        num_agents=4, num_steps=800, verbose=True
    )

    # 实验 4：复合指令
    results['exp4'] = experiment_4_compound_instruction(
        num_trials=40, verbose=True
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

    with open('instruction_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 instruction_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 36 总结")
    print("=" * 60)
    print(f"  基本指令成功率: {results['exp1']['total_success_rate']:.3f}")
    print(f"  指令准确率: {results['exp2']['instructed_accuracy']:.3f} vs {results['exp2']['random_accuracy']:.3f}")
    print(f"  指令成功率: {results['exp3']['success_rate']:.3f}")
    print(f"  复合指令成功率: {results['exp4']['compound_success_rate']:.3f}")

    return results


if __name__ == '__main__':
    main()
