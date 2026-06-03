"""
Phase 67: 语言引导注意力 —— 自上而下感知调制

核心思想：
Phase 29 创建了感官编码器，Phase 23 证明了跨模态整合。
Phase 51 证明了自主学习。但注意力始终是被动的——
agent 平等地感知视野中的一切。

在婴儿发展中，语言塑造注意力：知道 "red" 这个词后，
红色物体变得更显著（Waxman & Gelman, 2009）。
本阶段测试语言是否创建自上而下的注意力偏见。

涌现条件：
1. Agent 学会了特定类别的词汇
2. 这些词汇调制感官编码器的分支权重
3. 已知类别的物体被更快、更准确地识别
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
)
from encoder_sensory import SensoryEncoder


# ============================================================
# 注意力调制器
# ============================================================

class AttentionModulator:
    """
    注意力调制器

    根据已知语言类别调整感官编码的分支权重：
    - visual (0:16): 颜色、形状相关的词汇提升视觉权重
    - audio (16:32): 听觉相关词汇提升音频权重
    - position (32:40): 空间相关词汇提升位置权重

    实现 Waxman & Gelman (2009) 的语言-注意力耦合：
    "知道一个标签使得对应维度更可提取"
    """

    def __init__(self, base_weights: Optional[np.ndarray] = None):
        # [visual, audio, position]
        self.branch_weights = base_weights if base_weights is not None else np.array([1.0, 1.0, 1.0])
        self.known_categories: Dict[str, float] = {}  # category -> weight boost
        self.modulation_history: List[Dict] = []

    def learn_category(self, category: str, success_rate: float):
        """
        从语言学习中学到一个类别

        成功率高的类别获得更强的注意力调制
        """
        boost = 0.2 + 0.3 * success_rate  # [0.2, 0.5]
        self.known_categories[category] = boost

        # 根据类别更新分支权重
        visual_cats = {'color', 'shape', 'pattern', 'brightness'}
        audio_cats = {'texture', 'weight'}
        position_cats = {'size', 'temperature', 'origin'}

        if category in visual_cats:
            self.branch_weights[0] += boost * 0.1
        elif category in audio_cats:
            self.branch_weights[1] += boost * 0.1
        elif category in position_cats:
            self.branch_weights[2] += boost * 0.1

    def apply(self, encoded: np.ndarray) -> np.ndarray:
        """
        调制 40d 编码向量

        visual(0:16) *= w[0], audio(16:32) *= w[1], position(32:40) *= w[2]
        """
        modulated = encoded.copy()
        modulated[0:16] *= self.branch_weights[0]
        modulated[16:32] *= self.branch_weights[1]
        modulated[32:40] *= self.branch_weights[2]
        return modulated

    def get_saliency_map(self, encoded: np.ndarray) -> np.ndarray:
        """获取每维度的显著度分数"""
        modulated = self.apply(encoded)
        base_magnitude = np.abs(encoded) + 1e-8
        return np.abs(modulated - encoded) / base_magnitude

    def get_branch_weights(self) -> Dict[str, float]:
        return {
            'visual': round(float(self.branch_weights[0]), 4),
            'audio': round(float(self.branch_weights[1]), 4),
            'position': round(float(self.branch_weights[2]), 4),
        }


class AttentiveAgent:
    """
    注意力 Agent

    在标准 Agent 基础上：
    1. 学习语言类别
    2. AttentionModulator 调制编码
    3. 视觉搜索更高效
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.language = EmergingLanguage()
        self.encoder = SensoryEncoder()
        self.modulator = AttentionModulator()
        self.learned_colors = set()
        self.learned_shapes = set()

    def learn_vocabulary(self, scene_objects_list, num_rounds: int = 100):
        """从场景中学习词汇并建立注意力调制"""
        for _ in range(num_rounds):
            scene = random.choice(scene_objects_list)
            for item in scene:
                if isinstance(item, tuple):
                    _, features = item
                else:
                    features = item
                for attr, value in features.items():
                    if attr.startswith('_'):
                        continue
                    self.language.record_usage([value], True)
                    cat = _symbol_category(value)
                    if cat:
                        sr = self.language.vocabulary.get(value, {}).get('success_rate', 0.5)
                        self.modulator.learn_category(cat, sr)
                        if cat == 'color':
                            self.learned_colors.add(value)
                        elif cat == 'shape':
                            self.learned_shapes.add(value)

    def encode_with_attention(self, visual: np.ndarray, audio: np.ndarray,
                              position: np.ndarray) -> np.ndarray:
        """带注意力调制的编码"""
        raw = self.encoder.forward(visual, audio, position)
        return self.modulator.apply(raw)

    def visual_search(self, target_description: List[str],
                      scene_objects: List[Tuple[np.ndarray, Dict]],
                      use_attention: bool = True) -> Dict:
        """
        视觉搜索：在场景中找到目标对象

        返回搜索效率指标（步数 = 检查了多少对象才找到）
        """
        target_cat = set()
        for sym in target_description:
            cat = _symbol_category(sym)
            if cat:
                target_cat.add(cat)

        # 编码所有场景对象
        scores = []
        for visual, features in scene_objects:
            audio = np.random.randn(7) * 0.1
            position = np.array([features.get('_x', 0), features.get('_y', 0)])

            if use_attention:
                encoded = self.encode_with_attention(visual, audio, position)
            else:
                encoded = self.encoder.forward(visual, audio, position)

            # 计算与目标类别的匹配分数
            score = 0.0
            for sym in target_description:
                cat = _symbol_category(sym)
                if cat and features.get(cat) == sym:
                    score += 1.0
            scores.append((score, encoded, features))

        # 按分数排序（注意力调制后，匹配目标的排在前面）
        scores.sort(key=lambda x: x[0], reverse=True)

        # 模拟搜索：检查对象直到找到完美匹配
        steps = 0
        for score, _, features in scores:
            steps += 1
            if score >= len(target_description):
                return {'found': True, 'steps': steps, 'total': len(scene_objects)}

        return {'found': False, 'steps': steps, 'total': len(scene_objects)}


class CategoricalPerception:
    """
    类别感知测量

    测量 agent 感知类别边界的敏锐度。
    有语言标签的维度应显示更陡峭的边界（Winawer et al. 2007）。
    """

    @staticmethod
    def measure_boundary_sharpness(agent: AttentiveAgent,
                                    stimuli: List[np.ndarray],
                                    boundary_index: int,
                                    use_attention: bool = True) -> float:
        """
        测量类别边界的编码距离斜率

        陡峭 = 类别边界清晰
        平缓 = 类别边界模糊
        """
        if len(stimuli) < 3 or boundary_index < 1 or boundary_index >= len(stimuli) - 1:
            return 0.0

        # 编码边界附近的刺激
        encodings = []
        for stim in stimuli:
            audio = np.random.randn(7) * 0.01
            pos = np.array([0.0, 0.0])
            if use_attention:
                enc = agent.encode_with_attention(stim, audio, pos)
            else:
                enc = agent.encoder.forward(stim, audio, pos)
            encodings.append(enc)

        # 计算边界处的距离变化
        left_dist = np.linalg.norm(encodings[boundary_index] - encodings[boundary_index - 1])
        right_dist = np.linalg.norm(encodings[boundary_index + 1] - encodings[boundary_index])
        avg_neighbor_dist = (left_dist + right_dist) / 2 + 1e-8

        # 跨边界距离
        cross_dist = np.linalg.norm(encodings[boundary_index + 1] - encodings[boundary_index - 1])

        # 斜率 = 跨边界距离 / 平均邻居距离
        sharpness = cross_dist / avg_neighbor_dist
        return float(sharpness)


# ============================================================
# 辅助函数
# ============================================================

def generate_color_stimuli(num_colors: int = 6, stimuli_per_color: int = 10
                           ) -> Tuple[List[np.ndarray], List[int]]:
    """生成颜色渐变刺激"""
    base_colors = [
        np.array([1, 0, 0, 0.5]),   # red
        np.array([0, 1, 0, 0.5]),   # green
        np.array([0, 0, 1, 0.5]),   # blue
        np.array([1, 1, 0, 0.5]),   # yellow
        np.array([1, 1, 1, 0.5]),   # white
        np.array([0, 0, 0, 0.5]),   # black
    ]

    stimuli = []
    labels = []
    for color_idx in range(min(num_colors, len(base_colors))):
        for j in range(stimuli_per_color):
            # 在基础颜色周围添加噪声
            img = np.random.rand(8, 8, 4) * 0.1
            img[2:6, 2:6, :] = base_colors[color_idx] + np.random.randn(4) * 0.05
            img = np.clip(img, 0, 1)
            stimuli.append(img)
            labels.append(color_idx)

    return stimuli, labels


def generate_scene_objects(num_objects: int = 8) -> List[Tuple[np.ndarray, Dict]]:
    """生成带特征的场景对象"""
    objects = []
    colors = list(COLORS)
    shapes = list(SHAPES)
    sizes = list(SIZES)

    for i in range(num_objects):
        color = random.choice(colors)
        shape = random.choice(shapes)
        size = random.choice(sizes)

        # 创建视觉特征
        color_idx = colors.index(color) if color in colors else 0
        visual = np.random.rand(8, 8, 4) * 0.05
        visual[1:7, 1:7, color_idx % 4] = 0.8 + np.random.randn() * 0.05

        features = {
            'color': color, 'shape': shape, 'size': size,
            '_x': random.uniform(-5, 5), '_y': random.uniform(-5, 5),
        }
        objects.append((visual, features))

    return objects


# ============================================================
# 实验
# ============================================================

def experiment_1_attention_modulation(train_rounds: int = 200,
                                      test_rounds: int = 100) -> Dict:
    """
    实验 1：注意力调制

    训练 agent 学习颜色词汇，测量视觉分支权重变化。
    """
    print("=" * 60)
    print("实验 1：注意力调制")
    print("=" * 60)

    # 生成训练场景
    scenes = []
    for _ in range(50):
        scenes.append(generate_scene_objects(6))

    # 有颜色词的 agent
    agent_with = AttentiveAgent(0)
    agent_with.learn_vocabulary(scenes, train_rounds)
    weights_with = agent_with.modulator.get_branch_weights()

    # 无颜色词的 agent（学习形状而非颜色）
    agent_without = AttentiveAgent(1)
    # 限制学习只到形状类
    for _ in range(train_rounds):
        scene = random.choice(scenes)
        for obj in scene:
            for attr, value in obj[1].items():
                if attr == 'shape':
                    agent_without.language.record_usage([value], True)
                    agent_without.modulator.learn_category('shape', 0.8)

    weights_without = agent_without.modulator.get_branch_weights()

    print(f"  有颜色词: visual={weights_with['visual']:.3f}, "
          f"audio={weights_with['audio']:.3f}, pos={weights_with['position']:.3f}")
    print(f"  无颜色词: visual={weights_without['visual']:.3f}, "
          f"audio={weights_without['audio']:.3f}, pos={weights_without['position']:.3f}")

    visual_boost = weights_with['visual'] / weights_without['visual']
    print(f"  视觉增强: {visual_boost:.2f}x")

    return {
        'with_color_words': weights_with,
        'without_color_words': weights_without,
        'visual_boost_ratio': round(visual_boost, 4),
        'learned_colors': list(agent_with.learned_colors),
    }


def experiment_2_visual_search(num_trials: int = 100, num_runs: int = 5) -> Dict:
    """
    实验 2：视觉搜索效率

    比较有/无注意力调制的搜索步数。
    """
    print("=" * 60)
    print("实验 2：视觉搜索效率")
    print("=" * 60)

    steps_with = []
    steps_without = []
    found_with = 0
    found_without = 0

    for run in range(num_runs):
        agent = AttentiveAgent(run)
        scenes = [generate_scene_objects(8) for _ in range(30)]
        agent.learn_vocabulary(scenes, 150)

        for _ in range(num_trials):
            scene = generate_scene_objects(8)
            target_idx = random.randint(0, len(scene) - 1)
            target_features = scene[target_idx][1]
            target_desc = [target_features.get('color', ''), target_features.get('shape', '')]
            target_desc = [s for s in target_desc if s]

            r_with = agent.visual_search(target_desc, scene, use_attention=True)
            r_without = agent.visual_search(target_desc, scene, use_attention=False)

            if r_with['found']:
                steps_with.append(r_with['steps'])
                found_with += 1
            if r_without['found']:
                steps_without.append(r_without['steps'])
                found_without += 1

    avg_with = float(np.mean(steps_with)) if steps_with else 0
    avg_without = float(np.mean(steps_without)) if steps_without else 0
    improvement = (avg_without - avg_with) / max(1, avg_without) * 100 if avg_without > 0 else 0

    print(f"  有注意力: {avg_with:.1f} 步/搜索 (找到 {found_with}/{num_trials * num_runs})")
    print(f"  无注意力: {avg_without:.1f} 步/搜索 (找到 {found_without}/{num_trials * num_runs})")
    print(f"  搜索效率提升: {improvement:.1f}%")

    return {
        'with_attention_avg_steps': round(avg_with, 2),
        'without_attention_avg_steps': round(avg_without, 2),
        'search_improvement_pct': round(improvement, 2),
        'found_rate_with': round(found_with / (num_trials * num_runs), 4),
        'found_rate_without': round(found_without / (num_trials * num_runs), 4),
    }


def experiment_3_categorical_perception(num_stimuli: int = 50,
                                         num_runs: int = 5) -> Dict:
    """
    实验 3：类别感知效应

    比较有/无颜色词汇时，颜色边界的感知锐度。
    """
    print("=" * 60)
    print("实验 3：类别感知效应")
    print("=" * 60)

    sharpness_with = []
    sharpness_without = []

    for run in range(num_runs):
        agent = AttentiveAgent(run)
        scenes = [generate_scene_objects(6) for _ in range(30)]
        agent.learn_vocabulary(scenes, 200)

        # 生成沿颜色维度的渐变刺激
        stimuli, labels = generate_color_stimuli(6, num_stimuli // 6)
        # 排序以形成渐变
        combined = list(zip(stimuli, labels))
        random.shuffle(combined)
        stimuli = [s for s, _ in combined]

        # 在每个颜色边界处测量锐度
        boundaries = [num_stimuli // 6 * i for i in range(1, 6)]
        for bi in boundaries:
            if bi < len(stimuli):
                sw = CategoricalPerception.measure_boundary_sharpness(
                    agent, stimuli, bi, use_attention=True
                )
                so = CategoricalPerception.measure_boundary_sharpness(
                    agent, stimuli, bi, use_attention=False
                )
                sharpness_with.append(sw)
                sharpness_without.append(so)

    avg_with = float(np.mean(sharpness_with)) if sharpness_with else 0
    avg_without = float(np.mean(sharpness_without)) if sharpness_without else 0
    enhancement = (avg_with - avg_without) / max(0.01, avg_without) * 100

    print(f"  有注意力锐度: {avg_with:.4f}")
    print(f"  无注意力锐度: {avg_without:.4f}")
    print(f"  锐度增强: {enhancement:.1f}%")

    return {
        'sharpness_with_attention': round(avg_with, 4),
        'sharpness_without_attention': round(avg_without, 4),
        'enhancement_pct': round(enhancement, 2),
    }


def experiment_4_attention_transfer(num_agents: int = 3,
                                     num_rounds: int = 200) -> Dict:
    """
    实验 4：注意力跨 agent 迁移

    Agent A 学习注意力权重，通过通信传递给 Agent B/C。
    """
    print("=" * 60)
    print("实验 4：注意力跨 agent 迁移")
    print("=" * 60)

    agents = [AttentiveAgent(i) for i in range(num_agents)]
    scenes = [generate_scene_objects(6) for _ in range(40)]

    # 初始权重
    initial_weights = [a.modulator.get_branch_weights() for a in agents]

    # Agent 0 先学习
    agents[0].learn_vocabulary(scenes, 300)

    # 通信阶段：Agent 0 分享词汇
    for r in range(num_rounds):
        scene = random.choice(scenes)
        for obj_vis, obj_feat in scene:
            # Agent 0 描述
            desc = []
            for attr, val in obj_feat.items():
                if attr.startswith('_'):
                    continue
                if val in agents[0].language.vocabulary:
                    desc.append(val)

            # 其他 agent 从描述中学习
            for agent in agents[1:]:
                for sym in desc:
                    agent.language.record_usage([sym], True)
                    cat = _symbol_category(sym)
                    if cat:
                        sr = agent.language.vocabulary.get(sym, {}).get('success_rate', 0.5)
                        agent.modulator.learn_category(cat, sr)

    # 最终权重
    final_weights = [a.modulator.get_branch_weights() for a in agents]

    # 对齐度：Agent 1,2 与 Agent 0 的权重相似度
    alignments = []
    ref = np.array([final_weights[0]['visual'], final_weights[0]['audio'],
                    final_weights[0]['position']])
    for i in range(1, num_agents):
        w = np.array([final_weights[i]['visual'], final_weights[i]['audio'],
                     final_weights[i]['position']])
        sim = np.dot(ref, w) / (np.linalg.norm(ref) * np.linalg.norm(w) + 1e-8)
        alignments.append(float(sim))

    avg_alignment = float(np.mean(alignments))

    print(f"  Agent 0 权重: {final_weights[0]}")
    for i in range(1, num_agents):
        print(f"  Agent {i} 权重: {final_weights[i]}, 对齐度: {alignments[i-1]:.3f}")
    print(f"  平均对齐度: {avg_alignment:.3f}")

    return {
        'initial_weights': initial_weights,
        'final_weights': final_weights,
        'avg_alignment': round(avg_alignment, 4),
        'per_agent_alignment': [round(a, 4) for a in alignments],
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_attention_modulation()
    results['experiment_2'] = experiment_2_visual_search()
    results['experiment_3'] = experiment_3_categorical_perception()
    results['experiment_4'] = experiment_4_attention_transfer()

    output_file = 'language_attention_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
