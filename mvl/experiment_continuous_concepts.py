"""
Phase 65: 连续概念空间 — 从连续感知中涌现模糊类别

核心问题：
语言中的离散类别（"红"、"大"）能否从连续感知中涌现？
当感知输入是连续的 40 维编码时，agent 能否自发发现有意义的聚类？

人类婴儿面对的是连续的光谱、连续的声学输入，
但最终学会了离散的颜色词、物体类别。
这个 phase 测试连续→离散的涌现过程。

实验设计：
1. 类别发现：500 物体, k=6 聚类，测量与 ground-truth 属性类别的纯度
2. 边界漂移：不同粒度的 agent 交流后，类别边界是否移动
3. 模糊 vs 二值：近边界物体上 FuzzyListener 对比 BaselineBinaryGame
4. 跨 agent 对齐：3 个独立 agent 交流后类别对齐度
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional

from encoder_sensory import SensoryEncoder


# ============================================================
# 辅助函数
# ============================================================

def generate_visual_input(category: int, noise: float = 0.1) -> np.ndarray:
    """
    生成 8x8x4 视觉输入，按主导通道分组

    channel 0=红色系, 1=绿色系, 2=蓝色系, 3=白色系
    category 0-3 对应各主导通道
    """
    base = np.random.rand(8, 8, 4) * noise
    dominant_channel = category % 4
    base[:, :, dominant_channel] += 0.5 + np.random.rand(8, 8) * 0.3
    return np.clip(base, 0.0, 1.0)


def generate_audio_input(category: int, dim: int = 7) -> np.ndarray:
    """生成与类别相关的音频向量"""
    audio = np.random.randn(dim) * 0.1
    # 每个类别有不同的频率偏移
    freq_bin = category % dim
    audio[freq_bin] += 1.0
    return audio


def generate_position_input(category: int, dim: int = 2) -> np.ndarray:
    """生成与类别相关的位置"""
    angle = category * np.pi / 3 + np.random.randn() * 0.3
    radius = 2.0 + np.random.rand() * 1.0
    pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    return pos


def generate_object_embedding(encoder: SensoryEncoder, category: int) -> np.ndarray:
    """生成一个物体的 40 维编码"""
    visual = generate_visual_input(category)
    audio = generate_audio_input(category)
    position = generate_position_input(category)
    return encoder.forward(visual, audio, position)


def generate_dataset(encoder: SensoryEncoder, n_objects: int, n_categories: int = 4
                     ) -> Tuple[np.ndarray, List[int]]:
    """
    生成 n_objects 个物体的编码数据集

    Returns:
        embeddings: (n_objects, 40) 编码矩阵
        labels: 每个物体的 ground-truth 类别
    """
    embeddings = []
    labels = []
    for i in range(n_objects):
        cat = i % n_categories
        emb = generate_object_embedding(encoder, cat)
        embeddings.append(emb)
        labels.append(cat)
    return np.array(embeddings), labels


def compute_cluster_purity(predicted_labels: List[int], true_labels: List[int]) -> float:
    """计算聚类纯度（不用 sklearn）"""
    n = len(true_labels)
    if n == 0:
        return 0.0

    # 统计每个预测簇中真实类别的分布
    cluster_classes: Dict[int, Dict[int, int]] = {}
    for pl, tl in zip(predicted_labels, true_labels):
        if pl not in cluster_classes:
            cluster_classes[pl] = {}
        cluster_classes[pl][tl] = cluster_classes[pl].get(tl, 0) + 1

    # 纯度 = 每个簇中最多的真实类别数之和 / 总数
    correct = sum(max(counts.values()) for counts in cluster_classes.values())
    return correct / n


# ============================================================
# ContinuousConceptLearner — 在线 k-means 聚类
# ============================================================

class ContinuousConceptLearner:
    """
    从连续嵌入中发现模糊类别

    使用在线 k-means 维护 k 个质心，
    并提供高斯隶属度评分和边界锐度分析。
    """

    def __init__(self, embedding_dim: int = 40):
        self.embedding_dim = embedding_dim
        self.centroids: Optional[np.ndarray] = None  # (k, 40)
        self.labels: List[str] = []
        self.k: int = 0
        self._assignment_counts: Optional[np.ndarray] = None
        self._sigma: float = 1.0  # 高斯隶属度带宽

    def discover_categories(self, embeddings: np.ndarray, k: int,
                            lr: float = 0.05, epochs: int = 5) -> None:
        """
        在线 k-means 聚类

        初始化：取前 k 个样本作为初始质心
        更新：centroid += lr * (sample - centroid)（最近质心）
        多轮遍历以收敛
        """
        n = embeddings.shape[0]
        self.k = k

        # 初始化质心
        indices = np.random.choice(n, k, replace=False)
        self.centroids = embeddings[indices].copy()
        self.labels = [f"cat_{i}" for i in range(k)]
        self._assignment_counts = np.ones(k)  # 每个簇的样本计数

        # 在线 k-means 迭代
        for epoch in range(epochs):
            order = np.random.permutation(n)
            for idx in order:
                sample = embeddings[idx]
                # 找最近质心
                distances = np.linalg.norm(self.centroids - sample, axis=1)
                nearest = int(np.argmin(distances))
                # 在线更新
                self._assignment_counts[nearest] += 1
                effective_lr = lr / (1.0 + 0.01 * self._assignment_counts[nearest])
                self.centroids[nearest] += effective_lr * (sample - self.centroids[nearest])

        # 用全部数据重新计算隶属度带宽 sigma
        all_dists = []
        for i in range(n):
            dists = np.linalg.norm(self.centroids - embeddings[i], axis=1)
            all_dists.append(dists.min())
        if len(all_dists) > 0:
            self._sigma = max(np.std(all_dists), 0.1)

    def classify(self, embedding: np.ndarray) -> str:
        """最近质心分类"""
        if self.centroids is None:
            return "cat_0"
        distances = np.linalg.norm(self.centroids - embedding, axis=1)
        nearest = int(np.argmin(distances))
        return self.labels[nearest]

    def classify_index(self, embedding: np.ndarray) -> int:
        """返回最近质心的索引"""
        if self.centroids is None:
            return 0
        distances = np.linalg.norm(self.centroids - embedding, axis=1)
        return int(np.argmin(distances))

    def get_membership(self, embedding: np.ndarray) -> np.ndarray:
        """
        高斯隶属度评分

        membership_i = exp(-||emb - centroid_i||^2 / (2 * sigma^2))
        """
        if self.centroids is None:
            return np.array([1.0])
        dists_sq = np.sum((self.centroids - embedding) ** 2, axis=1)
        membership = np.exp(-dists_sq / (2.0 * self._sigma ** 2))
        return membership

    def boundary_sharpness(self, dim: int) -> float:
        """
        沿某个维度的决策边界锐度

        在该维度上，找相邻质心间距最小的边界，
        用质心在该维度的标准差归一化。
        锐度越高 = 边界越清晰。
        """
        if self.centroids is None or self.k < 2:
            return 0.0

        dim_values = self.centroids[:, dim]
        sorted_vals = np.sort(dim_values)

        # 相邻质心在该维度上的间距
        gaps = np.diff(sorted_vals)
        min_gap = gaps.min() if len(gaps) > 0 else 1.0

        # 用该维度值的范围归一化
        val_range = sorted_vals[-1] - sorted_vals[0]
        if val_range < 1e-8:
            return 1.0

        sharpness = min_gap / val_range
        return float(sharpness)

    def apply_language_pressure(self, category: str, success_rate: float,
                                sample: np.ndarray, strength: float = 0.01) -> None:
        """
        语言压力：成功率高的类别质心向样本靠拢

        success_rate 高 → 质心微调（强化）
        success_rate 低 → 不调整（让其他类别占据）
        """
        if self.centroids is None:
            return
        if category not in self.labels:
            return
        idx = self.labels.index(category)
        # 成功率越高，移动越大
        delta = strength * success_rate * (sample - self.centroids[idx])
        self.centroids[idx] += delta


# ============================================================
# FuzzyListener — 模糊匹配的倾听者
# ============================================================

class FuzzyListener:
    """
    用高斯隶属度代替二值匹配的倾听者

    当场景中有近边界物体时，模糊匹配能更好地区分。
    """

    def __init__(self, learner: ContinuousConceptLearner):
        self.learner = learner

    def fuzzy_match(self, utterance: List[str], scene_embeddings: np.ndarray,
                    category_map: Dict[str, int]) -> np.ndarray:
        """
        根据话语中的类别名和场景物体的高斯隶属度打分

        utterance: 类别名列表 (如 ["cat_0", "cat_2"])
        scene_embeddings: (n_objects, 40) 场景物体编码
        category_map: 类别名 → 簇索引映射

        Returns:
            scores: (n_objects,) 每个物体的匹配分数
        """
        n_objects = scene_embeddings.shape[0]
        scores = np.zeros(n_objects)

        for sym in utterance:
            if sym not in category_map:
                continue
            cluster_idx = category_map[sym]
            for i in range(n_objects):
                membership = self.learner.get_membership(scene_embeddings[i])
                scores[i] += membership[cluster_idx]

        return scores

    def choose(self, utterance: List[str], scene_embeddings: np.ndarray,
               category_map: Dict[str, int]) -> int:
        """选择得分最高的物体"""
        scores = self.fuzzy_match(utterance, scene_embeddings, category_map)
        if scores.max() == 0:
            return 0
        return int(np.argmax(scores))


# ============================================================
# BaselineBinaryGame — 二值匹配基线
# ============================================================

class BaselineBinaryGame:
    """
    标准二值匹配基线：最近质心硬分配

    不使用模糊隶属度，直接用最近质心做二值匹配。
    """

    def __init__(self, learner: ContinuousConceptLearner):
        self.learner = learner

    def binary_match(self, utterance: List[str], scene_embeddings: np.ndarray,
                     category_map: Dict[str, int]) -> np.ndarray:
        """二值匹配：只对精确匹配的物体给 1 分"""
        n_objects = scene_embeddings.shape[0]
        scores = np.zeros(n_objects)

        for i in range(n_objects):
            cat = self.learner.classify(scene_embeddings[i])
            if cat in utterance:
                scores[i] = 1.0

        return scores

    def choose(self, utterance: List[str], scene_embeddings: np.ndarray,
               category_map: Dict[str, int]) -> int:
        scores = self.binary_match(utterance, scene_embeddings, category_map)
        if scores.max() == 0:
            return 0
        # 多个匹配时随机选
        candidates = np.where(scores == scores.max())[0]
        return int(random.choice(candidates))


# ============================================================
# ContinuousConceptGame — 连续概念交流游戏
# ============================================================

class ContinuousConceptGame:
    """
    连续概念空间上的交流游戏

    流程：
    1. 编码器将物体编码为 40 维向量
    2. 学习者发现类别并记录质心
    3. 发言者描述目标物体的类别
    4. 倾听者用模糊匹配找到目标
    5. 成功/失败反馈驱动边界调整
    """

    def __init__(self, encoder: SensoryEncoder,
                 speaker_learner: ContinuousConceptLearner,
                 listener_learner: ContinuousConceptLearner,
                 fuzzy_listener: FuzzyListener):
        self.encoder = encoder
        self.speaker_learner = speaker_learner
        self.listener_learner = listener_learner
        self.fuzzy_listener = fuzzy_listener

    def generate_scene(self, n_objects: int = 4, n_categories: int = 4
                       ) -> Tuple[np.ndarray, List[int]]:
        """生成场景：n_objects 个物体的编码和类别"""
        embeddings = []
        categories = []
        for i in range(n_objects):
            cat = random.randint(0, n_categories - 1)
            emb = generate_object_embedding(self.encoder, cat)
            embeddings.append(emb)
            categories.append(cat)
        return np.array(embeddings), categories

    def play_round(self, n_objects: int = 4, n_categories: int = 4) -> Dict:
        """
        玩一轮交流游戏

        Returns:
            {'success': bool, 'target_cat': int, 'boundary_dist': float}
        """
        embeddings, categories = self.generate_scene(n_objects, n_categories)
        target_idx = random.randint(0, n_objects - 1)
        target_cat = categories[target_idx]

        # 发言者：描述目标类别
        target_label = self.speaker_learner.classify(embeddings[target_idx])
        utterance = [target_label]

        # 倾听者类别映射
        category_map = {lbl: i for i, lbl in enumerate(self.listener_learner.labels)}

        # 倾听者：模糊匹配
        chosen_idx = self.fuzzy_listener.choose(utterance, embeddings, category_map)

        success = (chosen_idx == target_idx)

        # 语言压力调整
        if success:
            sr = 1.0
        else:
            sr = 0.0
        self.speaker_learner.apply_language_pressure(
            target_label, sr, embeddings[target_idx]
        )
        self.listener_learner.apply_language_pressure(
            target_label, sr, embeddings[target_idx]
        )

        # 计算目标到最近边界的距离
        target_emb = embeddings[target_idx]
        dists = np.linalg.norm(self.speaker_learner.centroids - target_emb, axis=1)
        sorted_dists = np.sort(dists)
        boundary_dist = sorted_dists[1] - sorted_dists[0] if len(sorted_dists) > 1 else 0.0

        return {
            'success': success,
            'target_cat': target_cat,
            'boundary_dist': float(boundary_dist),
        }

    def run(self, n_rounds: int = 200, n_objects: int = 4, n_categories: int = 4) -> Dict:
        """运行多轮游戏"""
        successes = 0
        boundary_dists = []
        cat_successes: Dict[int, int] = {}
        cat_totals: Dict[int, int] = {}

        for r in range(n_rounds):
            result = self.play_round(n_objects, n_categories)
            if result['success']:
                successes += 1
            boundary_dists.append(result['boundary_dist'])

            cat = result['target_cat']
            cat_totals[cat] = cat_totals.get(cat, 0) + 1
            if result['success']:
                cat_successes[cat] = cat_successes.get(cat, 0) + 1

        # 计算各维度边界锐度
        sharpnesses = []
        for dim in range(self.speaker_learner.embedding_dim):
            s = self.speaker_learner.boundary_sharpness(dim)
            sharpnesses.append(s)

        return {
            'success_rate': round(successes / max(n_rounds, 1), 4),
            'avg_boundary_dist': round(float(np.mean(boundary_dists)), 4),
            'boundary_sharpness_avg': round(float(np.mean(sharpnesses)), 4),
            'category_success_rates': {
                str(cat): round(cat_successes.get(cat, 0) / max(cat_totals.get(cat, 1), 1), 4)
                for cat in sorted(cat_totals.keys())
            },
        }


# ============================================================
# 实验
# ============================================================

def experiment_1_category_discovery():
    """
    实验 1：类别发现（500 物体, k=6）

    用 encoder 编码 500 个物体（4 个 ground-truth 类别），
    运行在线 k-means (k=6)，测量聚类纯度。
    """
    print("=" * 60)
    print("实验 1：类别发现（500 物体, k=6 聚类）")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)
    encoder = SensoryEncoder()

    # 生成 500 个物体，4 个 ground-truth 类别
    n_objects = 500
    n_gt_categories = 4
    embeddings, true_labels = generate_dataset(encoder, n_objects, n_gt_categories)

    # 用 k=6 聚类（多于 ground-truth 类别数，测试能否发现子结构）
    k = 6
    learner = ContinuousConceptLearner(embedding_dim=40)
    learner.discover_categories(embeddings, k, lr=0.05, epochs=10)

    # 计算聚类纯度
    predicted = [learner.classify_index(emb) for emb in embeddings]
    purity = compute_cluster_purity(predicted, true_labels)

    # 各维度边界锐度
    sharpnesses = {}
    for dim in range(40):
        sharpnesses[f"dim_{dim}"] = round(learner.boundary_sharpness(dim), 4)

    # 前 10 维的平均锐度
    top10_sharp = np.mean([learner.boundary_sharpness(d) for d in range(10)])
    all_sharp = np.mean([learner.boundary_sharpness(d) for d in range(40)])

    # 质心间距离矩阵
    centroid_dists = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            centroid_dists[i][j] = np.linalg.norm(learner.centroids[i] - learner.centroids[j])

    print(f"\n  聚类纯度: {purity:.4f}")
    print(f"  前10维平均边界锐度: {top10_sharp:.4f}")
    print(f"  全40维平均边界锐度: {all_sharp:.4f}")
    print(f"  质心间平均距离: {centroid_dists[centroid_dists > 0].mean():.4f}")

    # 每个 ground-truth 类别被分配到哪个簇
    for gt_cat in range(n_gt_categories):
        gt_indices = [i for i, l in enumerate(true_labels) if l == gt_cat]
        gt_predicted = [predicted[i] for i in gt_indices]
        dominant = max(set(gt_predicted), key=gt_predicted.count)
        print(f"  GT 类别 {gt_cat} → 主要分配到簇 {dominant} ({gt_predicted.count(dominant)}/{len(gt_predicted)})")

    return {
        'purity': round(purity, 4),
        'top10_sharpness': round(float(top10_sharp), 4),
        'all_sharpness': round(float(all_sharp), 4),
        'avg_centroid_distance': round(float(centroid_dists[centroid_dists > 0].mean()), 4),
        'k': k,
        'n_objects': n_objects,
    }


def experiment_2_boundary_shift():
    """
    实验 2：边界漂移（300 轮, 粗粒度 vs 细粒度 agent）

    Agent A 有 6 个类别（细粒度），Agent B 有 3 个类别（粗粒度）。
    交流 300 轮后，测量 A 的边界是否向 B 的粒度漂移。
    """
    print("\n" + "=" * 60)
    print("实验 2：边界漂移（300 轮, 6-簇 vs 3-簇）")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)
    encoder = SensoryEncoder()

    # 生成共享数据集
    n_objects = 200
    n_gt_categories = 4
    embeddings, true_labels = generate_dataset(encoder, n_objects, n_gt_categories)

    # Agent A: 细粒度 (6 簇)
    learner_a = ContinuousConceptLearner(embedding_dim=40)
    learner_a.discover_categories(embeddings, k=6, lr=0.05, epochs=10)
    boundaries_a_before = [learner_a.boundary_sharpness(d) for d in range(40)]

    # Agent B: 粗粒度 (3 簇)
    learner_b = ContinuousConceptLearner(embedding_dim=40)
    learner_b.discover_categories(embeddings, k=3, lr=0.05, epochs=10)

    # 交流 300 轮
    n_rounds = 300
    fuzzy_b = FuzzyListener(learner_b)
    category_map_b = {lbl: i for i, lbl in enumerate(learner_b.labels)}

    shifts = []
    for r in range(n_rounds):
        # 随机选一个物体
        idx = random.randint(0, n_objects - 1)
        emb = embeddings[idx]

        # Agent A 描述
        label_a = learner_a.classify(emb)
        # Agent B 理解
        membership_b = learner_b.get_membership(emb)
        chosen_b = int(np.argmax(membership_b))

        # 如果 B 的理解与 A 的标签一致（用质心距离判断）
        dist_a = np.linalg.norm(learner_a.centroids[learner_a.labels.index(label_a)] - emb)
        dist_b = np.linalg.norm(learner_b.centroids[chosen_b] - emb)

        # 成功 = A 和 B 选的质心到物体的距离都较小
        threshold = np.mean([dist_a, dist_b]) + np.std([dist_a, dist_b])
        success = (dist_a < threshold and dist_b < threshold)

        sr = 1.0 if success else 0.0
        learner_a.apply_language_pressure(label_a, sr, emb, strength=0.02)

        if r % 50 == 0:
            shift_r = np.mean([
                abs(learner_a.boundary_sharpness(d) - boundaries_a_before[d])
                for d in range(40)
            ])
            shifts.append(round(float(shift_r), 6))

    boundaries_a_after = [learner_a.boundary_sharpness(d) for d in range(40)]

    # 平均边界变化量
    avg_shift = np.mean([
        abs(boundaries_a_after[d] - boundaries_a_before[d])
        for d in range(40)
    ])

    # 纯度变化
    predicted_before = [learner_a.classify_index(emb) for emb in embeddings]
    purity_before = compute_cluster_purity(predicted_before, true_labels)

    # 重新计算 classify（质心已经漂移了）
    predicted_after = [
        int(np.argmin(np.linalg.norm(learner_a.centroids - emb, axis=1)))
        for emb in embeddings
    ]
    purity_after = compute_cluster_purity(predicted_after, true_labels)

    print(f"\n  交流前平均边界锐度: {np.mean(boundaries_a_before):.4f}")
    print(f"  交流后平均边界锐度: {np.mean(boundaries_a_after):.4f}")
    print(f"  平均边界漂移量: {avg_shift:.6f}")
    print(f"  聚类纯度: {purity_before:.4f} → {purity_after:.4f}")
    print(f"  50轮间隔的漂移: {shifts}")

    return {
        'sharpness_before': round(float(np.mean(boundaries_a_before)), 4),
        'sharpness_after': round(float(np.mean(boundaries_a_after)), 4),
        'avg_boundary_shift': round(float(avg_shift), 6),
        'purity_before': round(purity_before, 4),
        'purity_after': round(purity_after, 4),
        'shift_trajectory': shifts,
        'n_rounds': n_rounds,
    }


def experiment_3_fuzzy_vs_binary():
    """
    实验 3：模糊 vs 二值匹配（200 轮 x 5 次）

    在近边界物体上比较 FuzzyListener 和 BaselineBinaryGame。
    近边界物体 = 到最近两个质心距离差 < 阈值的物体。
    """
    print("\n" + "=" * 60)
    print("实验 3：模糊匹配 vs 二值匹配（200 轮 x 5 次）")
    print("=" * 60)

    n_runs = 5
    n_rounds = 200
    fuzzy_results = []
    binary_results = []

    for run in range(n_runs):
        np.random.seed(42 + run)
        random.seed(42 + run)
        encoder = SensoryEncoder()

        # 生成数据并聚类
        n_objects = 150
        n_categories = 4
        embeddings, true_labels = generate_dataset(encoder, n_objects, n_categories)

        learner = ContinuousConceptLearner(embedding_dim=40)
        learner.discover_categories(embeddings, k=6, lr=0.05, epochs=10)

        fuzzy_listener = FuzzyListener(learner)
        binary_game = BaselineBinaryGame(learner)
        category_map = {lbl: i for i, lbl in enumerate(learner.labels)}

        # 找近边界物体
        boundary_threshold = 0.5
        boundary_indices = []
        for i in range(n_objects):
            dists = np.linalg.norm(learner.centroids - embeddings[i], axis=1)
            sorted_dists = np.sort(dists)
            if len(sorted_dists) > 1 and (sorted_dists[1] - sorted_dists[0]) < boundary_threshold:
                boundary_indices.append(i)

        n_boundary = len(boundary_indices)
        if n_boundary == 0:
            boundary_indices = list(range(min(20, n_objects)))
            n_boundary = len(boundary_indices)

        # 模糊匹配测试
        fuzzy_correct = 0
        binary_correct = 0
        total_near = 0

        for r in range(n_rounds):
            # 随机选近边界物体作为目标
            target_idx = random.choice(boundary_indices)
            target_emb = embeddings[target_idx]
            target_cat = learner.classify(target_emb)
            utterance = [target_cat]

            # 生成干扰物场景（4 个物体含目标）
            scene_size = 4
            scene_indices = [target_idx]
            while len(scene_indices) < scene_size:
                cand = random.randint(0, n_objects - 1)
                if cand not in scene_indices:
                    scene_indices.append(cand)
            random.shuffle(scene_indices)
            new_target_pos = scene_indices.index(target_idx)

            scene_embs = embeddings[scene_indices]

            # 模糊匹配
            scores_fuzzy = fuzzy_listener.fuzzy_match(utterance, scene_embs, category_map)
            chosen_fuzzy = int(np.argmax(scores_fuzzy))
            if chosen_fuzzy == new_target_pos:
                fuzzy_correct += 1

            # 二值匹配
            scores_binary = binary_game.binary_match(utterance, scene_embs, category_map)
            if scores_binary.max() > 0:
                candidates = np.where(scores_binary == scores_binary.max())[0]
                chosen_binary = int(random.choice(candidates))
            else:
                chosen_binary = 0
            if chosen_binary == new_target_pos:
                binary_correct += 1

            total_near += 1

        fuzzy_sr = fuzzy_correct / max(total_near, 1)
        binary_sr = binary_correct / max(total_near, 1)
        fuzzy_results.append(fuzzy_sr)
        binary_results.append(binary_sr)

        print(f"  Run {run}: 模糊={fuzzy_sr:.4f}, 二值={binary_sr:.4f}, "
              f"近边界物体={n_boundary}")

    avg_fuzzy = float(np.mean(fuzzy_results))
    avg_binary = float(np.mean(binary_results))
    improvement = avg_fuzzy - avg_binary

    print(f"\n  平均模糊匹配: {avg_fuzzy:.4f}")
    print(f"  平均二值匹配: {avg_binary:.4f}")
    print(f"  模糊提升: {improvement:+.4f}")

    return {
        'fuzzy_avg': round(avg_fuzzy, 4),
        'binary_avg': round(avg_binary, 4),
        'improvement': round(improvement, 4),
        'fuzzy_per_run': [round(r, 4) for r in fuzzy_results],
        'binary_per_run': [round(r, 4) for r in binary_results],
    }


def experiment_4_cross_agent_alignment():
    """
    实验 4：跨 agent 对齐（3 个 agent, 200 轮）

    3 个 agent 独立发现类别后互相交流，
    测量交流后类别标签的对齐程度。
    """
    print("\n" + "=" * 60)
    print("实验 4：跨 agent 对齐（3 个 agent, 200 轮）")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)
    encoder = SensoryEncoder()

    # 共享数据集
    n_objects = 200
    n_categories = 4
    embeddings, true_labels = generate_dataset(encoder, n_objects, n_categories)

    # 3 个独立 agent
    n_agents = 3
    learners = []
    for a in range(n_agents):
        learner = ContinuousConceptLearner(embedding_dim=40)
        learner.discover_categories(embeddings, k=6, lr=0.05, epochs=10)
        learners.append(learner)

    # 对齐度：计算 agent 对之间的簇标签一致率
    def compute_pairwise_alignment(l1: ContinuousConceptLearner,
                                   l2: ContinuousConceptLearner) -> float:
        """计算两个 agent 之间类别标签对齐度（基于样本分配的一致性）"""
        assignments_1 = [l1.classify_index(emb) for emb in embeddings]
        assignments_2 = [l2.classify_index(emb) for emb in embeddings]

        # 对每一对样本，检查两个 agent 是否将它们分到同一个/不同簇
        n_pairs = 0
        agree = 0
        sample_size = min(500, n_objects * (n_objects - 1) // 2)
        for _ in range(sample_size):
            i = random.randint(0, n_objects - 1)
            j = random.randint(0, n_objects - 1)
            if i == j:
                continue
            same_1 = (assignments_1[i] == assignments_1[j])
            same_2 = (assignments_2[i] == assignments_2[j])
            if same_1 == same_2:
                agree += 1
            n_pairs += 1

        return agree / max(n_pairs, 1)

    # 交流前对齐度
    pre_alignments = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            align = compute_pairwise_alignment(learners[i], learners[j])
            pre_alignments.append(align)
            print(f"  交流前 Agent {i}-{j} 对齐度: {align:.4f}")

    # 交流阶段：200 轮 pairwise 交流
    n_rounds = 200
    for r in range(n_rounds):
        # 随机选 speaker 和 listener
        sp_idx = random.randint(0, n_agents - 1)
        li_idx = random.randint(0, n_agents - 1)
        if sp_idx == li_idx:
            li_idx = (sp_idx + 1) % n_agents

        speaker = learners[sp_idx]
        listener = learners[li_idx]

        # 随机选一个物体
        obj_idx = random.randint(0, n_objects - 1)
        emb = embeddings[obj_idx]

        # Speaker 描述
        label_sp = speaker.classify(emb)
        sp_cluster_idx = speaker.labels.index(label_sp)

        # Listener 用模糊匹配理解
        membership_li = listener.get_membership(emb)
        li_cluster_idx = int(np.argmax(membership_li))

        # 成功 = speaker 的质心和 listener 选的质心到物体距离都小
        dist_sp = np.linalg.norm(speaker.centroids[sp_cluster_idx] - emb)
        dist_li = np.linalg.norm(listener.centroids[li_cluster_idx] - emb)

        avg_dist = (dist_sp + dist_li) / 2.0
        success = avg_dist < np.mean([np.linalg.norm(c) for c in listener.centroids])

        sr = 1.0 if success else 0.0

        # 双向调整
        speaker.apply_language_pressure(label_sp, sr, emb, strength=0.01)
        listener.apply_language_pressure(
            listener.labels[li_cluster_idx], sr, emb, strength=0.01
        )

    # 交流后对齐度
    post_alignments = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            align = compute_pairwise_alignment(learners[i], learners[j])
            post_alignments.append(align)
            print(f"  交流后 Agent {i}-{j} 对齐度: {align:.4f}")

    avg_pre = float(np.mean(pre_alignments))
    avg_post = float(np.mean(post_alignments))
    improvement = avg_post - avg_pre

    # 各 agent 交流后的聚类纯度
    purity_per_agent = []
    for i, learner in enumerate(learners):
        predicted = [learner.classify_index(emb) for emb in embeddings]
        purity = compute_cluster_purity(predicted, true_labels)
        purity_per_agent.append(purity)

    print(f"\n  平均对齐度: {avg_pre:.4f} → {avg_post:.4f} (提升 {improvement:+.4f})")
    print(f"  各 agent 纯度: {[round(p, 4) for p in purity_per_agent]}")

    return {
        'pre_alignment': round(avg_pre, 4),
        'post_alignment': round(avg_post, 4),
        'alignment_improvement': round(improvement, 4),
        'pre_per_pair': [round(a, 4) for a in pre_alignments],
        'post_per_pair': [round(a, 4) for a in post_alignments],
        'purity_per_agent': [round(p, 4) for p in purity_per_agent],
        'n_agents': n_agents,
        'n_rounds': n_rounds,
    }


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_category_discovery()
    results['experiment_2'] = experiment_2_boundary_shift()
    results['experiment_3'] = experiment_3_fuzzy_vs_binary()
    results['experiment_4'] = experiment_4_cross_agent_alignment()

    with open('continuous_concepts_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 continuous_concepts_results.json")
