"""
Phase 29 实验：真实感官输入

实验 1：感官 vs 手工特征对比
实验 2：编码器消融（视觉/音频/位置分支）
实验 3：Detach vs 不 Detach
实验 4：编码器表示分析
"""

import numpy as np
import json
from typing import Dict, List
from collections import deque

from environment import SimpleGridWorld, Object
from environment_sensory import SensoryGridWorld, create_sensory_world
from agent import LearningAgent
from agent_sensory import SensoryAgent


def run_sensory_agent(env: SensoryGridWorld, agent: SensoryAgent,
                      n_steps: int = 1000, verbose: bool = False) -> Dict:
    """运行 SensoryAgent 实验"""
    obs = env.reset()
    errors = []
    diversities = []

    for step in range(n_steps):
        # 选择动作
        action = agent.act(obs)

        # 执行动作
        next_obs, reward, done = env.step(action)

        # 学习
        error = agent.learn_from_experience(obs, action, next_obs)
        errors.append(error)

        # 定期记录探索多样性
        if step % 50 == 0:
            div = agent.compute_exploration_diversity()
            diversities.append(div)

        obs = next_obs
        if done:
            obs = env.reset()

    return {
        'errors': errors,
        'diversities': diversities,
        'avg_error': np.mean(errors[-100:]) if errors else 0.0,
        'final_error': np.mean(errors[-50:]) if len(errors) >= 50 else np.mean(errors),
        'exploration_diversity': np.mean(diversities[-5:]) if diversities else 0.0,
        'param_count': agent.encoder.get_param_count(),
    }


def run_baseline_agent(env: SimpleGridWorld, agent: LearningAgent,
                       n_steps: int = 1000) -> Dict:
    """运行基线 LearningAgent 实验"""
    obs = env.reset()
    errors = []

    for step in range(n_steps):
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        error = agent.learn_from_experience(obs, action, next_obs)
        errors.append(error)
        obs = next_obs
        if done:
            obs = env.reset()

    return {
        'errors': errors,
        'avg_error': np.mean(errors[-100:]) if errors else 0.0,
        'final_error': np.mean(errors[-50:]) if len(errors) >= 50 else np.mean(errors),
    }


# ============================================================
# 实验 1：感官 vs 手工特征对比
# ============================================================

def experiment_1_comparison(n_steps: int = 1000, verbose: bool = True):
    """感官 Agent vs 手工特征 Agent 对比"""
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：感官 vs 手工特征对比")
        print("=" * 60)

    # A: SensoryAgent + SensoryGridWorld
    env_a = create_sensory_world()
    agent_a = SensoryAgent(
        visual_shape=(8, 8, 4), audio_dim=7, pos_dim=2,
        action_dim=5, learning_rate=0.001
    )
    result_a = run_sensory_agent(env_a, agent_a, n_steps, verbose=False)

    # B: LearningAgent + SimpleGridWorld
    env_b = SimpleGridWorld(10, 10)
    env_b.add_object(Object(0, 2, 2, 'red', 'circle', 1.0))
    env_b.add_object(Object(1, 5, 5, 'blue', 'square', 1.5))
    env_b.add_object(Object(2, 7, 3, 'green', 'triangle', 0.8))
    env_b.add_object(Object(3, 3, 7, 'yellow', 'circle', 1.2))
    agent_b = LearningAgent(obs_dim=12, action_dim=5)
    result_b = run_baseline_agent(env_b, agent_b, n_steps)

    if verbose:
        print(f"\nA: SensoryAgent（编码器+预测模型）")
        print(f"   编码器参数: {result_a['param_count']}")
        print(f"   最终误差: {result_a['final_error']:.4f}")
        print(f"   探索多样性: {result_a['exploration_diversity']:.3f}")

        print(f"\nB: LearningAgent（手工12维特征）")
        print(f"   最终误差: {result_b['final_error']:.4f}")

        # 收敛速度对比
        errors_a = result_a['errors']
        errors_b = result_b['errors']
        window = 50
        converg_a = [np.mean(errors_a[max(0,i-window):i+1]) for i in range(0, len(errors_a), 100)]
        converg_b = [np.mean(errors_b[max(0,i-window):i+1]) for i in range(0, len(errors_b), 100)]

        print(f"\n收敛曲线（每100步平均误差）:")
        print(f"  SensoryAgent: {[f'{e:.4f}' for e in converg_a[:5]]}")
        print(f"  LearningAgent: {[f'{e:.4f}' for e in converg_b[:5]]}")

    return {'A': result_a, 'B': result_b}


# ============================================================
# 实验 2：编码器消融
# ============================================================

def experiment_2_ablation(n_steps: int = 1000, verbose: bool = True):
    """编码器分支消融实验"""
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：编码器消融")
        print("=" * 60)

    results = {}

    # A: 完整编码器（视觉+音频+位置）
    env_a = create_sensory_world()
    agent_a = SensoryAgent(learning_rate=0.001)
    results['full'] = run_sensory_agent(env_a, agent_a, n_steps)

    # B: 仅视觉分支（音频和位置置零）
    env_b = create_sensory_world()
    agent_b = SensoryAgent(learning_rate=0.001)

    # 修改 agent 的 perceive 方法只用视觉
    original_perceive = agent_b.perceive
    def perceive_visual_only(obs):
        visual = obs.get('visual', np.zeros((8, 8, 4)))
        audio = np.zeros(7)
        position = np.zeros(2)
        return agent_b.encoder.forward(visual, audio, position)
    agent_b.perceive = perceive_visual_only

    original_detached = agent_b.perceive_detached
    def perceive_detached_visual_only(obs):
        visual = obs.get('visual', np.zeros((8, 8, 4)))
        audio = np.zeros(7)
        position = np.zeros(2)
        # 独立前向传播
        conv_out = agent_b.encoder._conv2d_forward(visual)
        conv_relu = np.maximum(0, conv_out)
        conv_flat = conv_relu.flatten()
        vis_feat = np.maximum(0, conv_flat @ agent_b.encoder.vis_W + agent_b.encoder.vis_b)
        aud_feat = np.zeros(agent_b.encoder.audio_out)
        pos_feat = np.zeros(agent_b.encoder.pos_out)
        return np.concatenate([vis_feat, aud_feat, pos_feat])
    agent_b.perceive_detached = perceive_detached_visual_only

    results['visual_only'] = run_sensory_agent(env_b, agent_b, n_steps)

    # C: 仅音频分支
    env_c = create_sensory_world()
    agent_c = SensoryAgent(learning_rate=0.001)

    def perceive_audio_only(obs):
        visual = np.zeros((8, 8, 4))
        audio = obs.get('audio', np.zeros(7))
        position = np.zeros(2)
        return agent_c.encoder.forward(visual, audio, position)
    agent_c.perceive = perceive_audio_only

    def perceive_detached_audio_only(obs):
        visual = np.zeros((8, 8, 4))
        audio = obs.get('audio', np.zeros(7))
        position = np.zeros(2)
        conv_out = agent_c.encoder._conv2d_forward(visual)
        conv_relu = np.maximum(0, conv_out)
        conv_flat = conv_relu.flatten()
        vis_feat = np.zeros(agent_c.encoder.visual_out)
        aud_feat = np.maximum(0, audio @ agent_c.encoder.aud_W + agent_c.encoder.aud_b)
        pos_feat = np.zeros(agent_c.encoder.pos_out)
        return np.concatenate([vis_feat, aud_feat, pos_feat])
    agent_c.perceive_detached = perceive_detached_audio_only

    results['audio_only'] = run_sensory_agent(env_c, agent_c, n_steps)

    # D: 仅位置分支
    env_d = create_sensory_world()
    agent_d = SensoryAgent(learning_rate=0.001)

    def perceive_position_only(obs):
        visual = np.zeros((8, 8, 4))
        audio = np.zeros(7)
        position = obs.get('agent_position', np.zeros(2))
        return agent_d.encoder.forward(visual, audio, position)
    agent_d.perceive = perceive_position_only

    def perceive_detached_position_only(obs):
        visual = np.zeros((8, 8, 4))
        audio = np.zeros(7)
        position = obs.get('agent_position', np.zeros(2))
        conv_out = agent_d.encoder._conv2d_forward(visual)
        conv_relu = np.maximum(0, conv_out)
        conv_flat = conv_relu.flatten()
        vis_feat = np.zeros(agent_d.encoder.visual_out)
        aud_feat = np.zeros(agent_d.encoder.audio_out)
        pos_feat = position @ agent_d.encoder.pos_W + agent_d.encoder.pos_b
        return np.concatenate([vis_feat, aud_feat, pos_feat])
    agent_d.perceive_detached = perceive_detached_position_only

    results['position_only'] = run_sensory_agent(env_d, agent_d, n_steps)

    if verbose:
        print(f"\n消融结果（最终误差，越低越好）:")
        for name, result in results.items():
            print(f"  {name:15s}: {result['final_error']:.4f}")

        # 计算各分支贡献
        full_error = results['full']['final_error']
        print(f"\n各分支贡献（误差减少百分比）:")
        for name in ['visual_only', 'audio_only', 'position_only']:
            branch_error = results[name]['final_error']
            if branch_error > 0:
                contribution = (branch_error - full_error) / branch_error * 100
            else:
                contribution = 0
            print(f"  {name:15s}: {contribution:+.1f}%")

    return results


# ============================================================
# 实验 3：Detach vs 不 Detach
# ============================================================

def experiment_3_detach(n_steps: int = 1000, verbose: bool = True):
    """Detach vs 不 Detach 对比"""
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：Detach vs 不 Detach")
        print("=" * 60)

    # A: Detach（正确方式）
    env_a = create_sensory_world()
    agent_a = SensoryAgent(learning_rate=0.001)
    result_a = run_sensory_agent(env_a, agent_a, n_steps)

    # B: 不 Detach（梯度流入编码器的目标编码）
    # 注意：此简化实现中，learn_and_get_input_gradient 只返回对输入的梯度，
    # 不返回对目标的梯度。因此"不 detach"的效果是缓存损坏（用 next_obs 的缓存
    # 更新 current_obs 的权重），而非真正的表示坍缩。完整的 no-detach 需要
    # 额外计算 d_loss/d_target 并反向传播。
    env_b = create_sensory_world()
    agent_b = SensoryAgent(learning_rate=0.001)

    # 重写 learn_from_experience：不 detach（修复缓存损坏）
    def learn_no_detach(observation, action, next_observation):
        # 编码当前观测（保留缓存用于 backward）
        encoded_obs = agent_b.perceive(observation)
        # 保存当前观测的缓存
        cache_snapshot = {k: v.copy() if isinstance(v, np.ndarray) else v
                         for k, v in agent_b.encoder._cache.items()}

        # 编码下一时刻观测（不 detach！使用 perceive 而非 perceive_detached）
        encoded_next_target = agent_b.perceive(next_observation)

        # 恢复当前观测的缓存（避免缓存损坏）
        agent_b.encoder._cache = cache_snapshot

        # 预测模型学习
        prediction_error, d_encoded = agent_b.predictor.learn_and_get_input_gradient(
            encoded_obs, action, encoded_next_target
        )

        # 编码器学习
        agent_b.encoder.backward(d_encoded, agent_b.encoder.lr)

        # 更新统计
        agent_b.step_count += 1
        agent_b.stats['total_steps'] += 1
        agent_b.stats['total_prediction_error'] += prediction_error

        if agent_b.step_count % 50 == 0:
            agent_b.stats['encoder_weight_norm'] = np.sqrt(
                np.sum(agent_b.encoder.conv_W ** 2) +
                np.sum(agent_b.encoder.vis_W ** 2) +
                np.sum(agent_b.encoder.aud_W ** 2) +
                np.sum(agent_b.encoder.pos_W ** 2)
            )

        return prediction_error

    agent_b.learn_from_experience = learn_no_detach
    result_b = run_sensory_agent(env_b, agent_b, n_steps)

    if verbose:
        print(f"\nA: Detach（正确方式）")
        print(f"   最终误差: {result_a['final_error']:.4f}")
        print(f"   编码器权重范数: {agent_a.stats['encoder_weight_norm']:.4f}")

        print(f"\nB: 不 Detach（梯度流入目标编码）")
        print(f"   最终误差: {result_b['final_error']:.4f}")
        print(f"   编码器权重范数: {agent_b.stats['encoder_weight_norm']:.4f}")

        # 表示坍缩度：编码器输出的方差
        print(f"\n表示坍缩度（编码方差，越高越好）:")
        # 用不同物体测试编码方差
        env_test = create_sensory_world()
        obs = env_test.reset()

        codes_a = []
        codes_b = []
        for _ in range(20):
            obs, _, done = env_test.step(np.random.randint(5))
            if done:
                obs = env_test.reset()
            codes_a.append(agent_a.perceive(obs))
            codes_b.append(agent_b.perceive(obs))

        var_a = np.mean(np.var(codes_a, axis=0))
        var_b = np.mean(np.var(codes_b, axis=0))
        print(f"  Detach:     {var_a:.6f}")
        print(f"  不 Detach:  {var_b:.6f}")

        if var_b < var_a * 0.5:
            print(f"  ⚠ 不 Detach 导致表示坍缩（方差降低 {100*(1-var_b/var_a):.1f}%）")

    return {'detach': result_a, 'no_detach': result_b}


# ============================================================
# 实验 4：编码器表示分析
# ============================================================

def experiment_4_representation(n_steps: int = 1000, verbose: bool = True):
    """编码器表示分析"""
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：编码器表示分析")
        print("=" * 60)

    # 训练 agent
    env = create_sensory_world()
    agent = SensoryAgent(learning_rate=0.001)
    result = run_sensory_agent(env, agent, n_steps, verbose=False)

    # 收集不同物体的编码
    env_test = create_sensory_world()
    obs = env_test.reset()

    # 为每个物体收集编码
    object_codes = {obj.id: [] for obj in env_test.objects}

    for step in range(200):
        action = np.random.randint(5)
        obs, _, done = env_test.step(action)
        if done:
            obs = env_test.reset()

        # 编码当前观测
        code = agent.perceive(obs)

        # 找到最近的物体
        for obj in env_test.objects:
            dist = abs(obj.x - env_test.agent_x) + abs(obj.y - env_test.agent_y)
            if dist <= 2:
                object_codes[obj.id].append(code.copy())

    # 分析编码
    if verbose:
        print(f"\n编码器参数: {agent.encoder.get_param_count()}")
        print(f"编码维度: {agent.encoder.output_dim}")

        # 类内/类间距离
        centroids = {}
        for obj_id, codes in object_codes.items():
            if codes:
                centroids[obj_id] = np.mean(codes, axis=0)

        # 类内距离
        intra_distances = []
        for obj_id, codes in object_codes.items():
            if len(codes) > 1:
                centroid = centroids[obj_id]
                dists = [np.sqrt(np.sum((c - centroid) ** 2)) for c in codes]
                intra_distances.append(np.mean(dists))

        # 类间距离
        inter_distances = []
        obj_ids = list(centroids.keys())
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                dist = np.sqrt(np.sum(
                    (centroids[obj_ids[i]] - centroids[obj_ids[j]]) ** 2
                ))
                inter_distances.append(dist)

        avg_intra = np.mean(intra_distances) if intra_distances else 0
        avg_inter = np.mean(inter_distances) if inter_distances else 0

        print(f"\n类内距离（同物体编码分散度）: {avg_intra:.4f}")
        print(f"类间距离（不同物体编码差异）: {avg_inter:.4f}")
        if avg_intra > 0:
            ratio = avg_inter / avg_intra
            print(f"类间/类内比: {ratio:.2f}（>2 表示良好聚类）")

        # 各物体编码中心
        print(f"\n各物体编码中心（前8维）:")
        for obj in env_test.objects:
            if obj.id in centroids:
                center = centroids[obj.id][:8]
                print(f"  {obj.color:6s} {obj.shape:8s}: [{', '.join(f'{v:.3f}' for v in center)}]")

    return {
        'final_error': result['final_error'],
        'object_codes': {str(k): [c.tolist() for c in v[:3]] for k, v in object_codes.items()},
        'centroids': {str(k): v.tolist() for k, v in centroids.items()},
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 29: 真实感官输入实验")
    print("=" * 60)

    results = {}

    # 实验 1
    results['exp1'] = experiment_1_comparison(n_steps=1000, verbose=True)

    # 实验 2
    results['exp2'] = experiment_2_ablation(n_steps=1000, verbose=True)

    # 实验 3
    results['exp3'] = experiment_3_detach(n_steps=1000, verbose=True)

    # 实验 4
    results['exp4'] = experiment_4_representation(n_steps=1000, verbose=True)

    # 保存结果
    # Convert numpy arrays to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open('sensory_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 sensory_results.json")

    return results


if __name__ == '__main__':
    main()
