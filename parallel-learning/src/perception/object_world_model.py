"""对象中心世界模型 — 从向量到对象的认知跃迁

替代整体向量预测，将世界分解为对象及其关系。

核心组件：
1. 对象检测器 — 从观测中分解对象
2. 对象属性编码器 — 颜色、形状、大小、位置
3. 对象关系编码器 — 接触、支撑、包含
4. 对象级预测器 — 预测每个对象的下一状态
5. 想象式规划器 — 在世界模型中rollout

设计原则：
- 对象是世界的基本单元，不是向量
- 关系是对象间的连接，不是整体的状态
- 规划是在想象中测试方案，不是直接行动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Object:
    """世界中的对象"""
    id: str
    attributes: Dict[str, float] = field(default_factory=dict)  # 属性
    position: Tuple[float, ...] = (0.0,)  # 位置
    embedding: Optional[torch.Tensor] = None  # 语义嵌入


@dataclass
class Relation:
    """对象间的关系"""
    source: str
    target: str
    type: str  # 'contact', 'support', 'contain', 'near', 'far'
    strength: float = 1.0


class ObjectDetector(nn.Module):
    """对象检测器

    从观测向量中分解出对象及其属性。
    使用注意力机制选择性地关注不同部分。
    """

    def __init__(self, obs_dim: int = 128, max_objects: int = 10, attr_dim: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.max_objects = max_objects
        self.attr_dim = attr_dim

        # 对象查询（可学习的查询向量）
        self.object_queries = nn.Parameter(torch.randn(max_objects, obs_dim) * 0.01)

        # 属性预测头
        self.attr_predictor = nn.Sequential(
            nn.Linear(obs_dim, obs_dim // 2),
            nn.GELU(),
            nn.Linear(obs_dim // 2, attr_dim),
        )

        # 对象存在性预测
        self.existence_predictor = nn.Sequential(
            nn.Linear(obs_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> List[Object]:
        """从观测中检测对象

        Args:
            obs: (obs_dim,) 观测向量

        Returns:
            检测到的对象列表
        """
        objects = []

        for i in range(self.max_objects):
            # 计算查询与观测的相似度
            query = self.object_queries[i]
            attention = torch.cosine_similarity(obs.unsqueeze(0), query.unsqueeze(0))

            # 预测对象存在性
            existence = self.existence_predictor(obs)

            if existence.item() > 0.5:
                # 预测属性
                attributes = self.attr_predictor(obs)

                # 创建对象
                obj = Object(
                    id=f'obj_{i}',
                    attributes={
                        'attr_0': attributes[0].item(),
                        'attr_1': attributes[1].item(),
                        'attr_2': attributes[2].item(),
                    },
                    embedding=obs.clone(),
                )
                objects.append(obj)

        return objects


class ObjectRelationEncoder(nn.Module):
    """对象关系编码器

    编码对象间的关系（接触、支撑、包含等）。
    """

    def __init__(self, obs_dim: int = 128, n_relation_types: int = 5):
        super().__init__()
        self.relation_classifier = nn.Sequential(
            nn.Linear(obs_dim * 2, obs_dim),
            nn.GELU(),
            nn.Linear(obs_dim, n_relation_types),
        )
        self.relation_types = ['contact', 'support', 'contain', 'near', 'far']

    def forward(self, obj1: Object, obj2: Object) -> List[Relation]:
        """编码两个对象间的关系"""
        if obj1.embedding is None or obj2.embedding is None:
            return []

        combined = torch.cat([obj1.embedding, obj2.embedding])
        logits = self.relation_classifier(combined)
        probs = F.softmax(logits, dim=-1)

        relations = []
        for i, rel_type in enumerate(self.relation_types):
            if probs[i].item() > 0.3:
                relations.append(Relation(
                    source=obj1.id,
                    target=obj2.id,
                    type=rel_type,
                    strength=probs[i].item(),
                ))

        return relations


class ObjectStatePredictor(nn.Module):
    """对象状态预测器

    预测对象在下一时间步的状态。
    """

    def __init__(self, obs_dim: int = 128, attr_dim: int = 16):
        super().__init__()
        self.state_predictor = nn.Sequential(
            nn.Linear(obs_dim + attr_dim, obs_dim),
            nn.GELU(),
            nn.Linear(obs_dim, obs_dim),
        )

    def forward(self, obj: Object, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测对象的下一状态"""
        if obj.embedding is None:
            return torch.zeros(128)

        # 编码属性
        attr_vec = torch.tensor(list(obj.attributes.values()), dtype=torch.float32)
        if len(attr_vec) < 16:
            attr_vec = F.pad(attr_vec, (0, 16 - len(attr_vec)))
        attr_vec = attr_vec.to(obj.embedding.device)

        combined = torch.cat([obj.embedding, attr_vec])
        return self.state_predictor(combined)


class ImaginationPlanner:
    """想象式规划器

    在世界模型中rollout多条轨迹，选择最优方案。
    """

    def __init__(self, world_model: 'ObjectCentricWorldModel', n_trajectories: int = 5,
                 horizon: int = 3):
        self.world_model = world_model
        self.n_trajectories = n_trajectories
        self.horizon = horizon

    def plan(self, current_state: List[Object], goal: str) -> List[Dict]:
        """规划从当前状态到目标的动作序列

        Returns:
            多条候选轨迹，每条包含动作序列和预期状态
        """
        trajectories = []

        for _ in range(self.n_trajectories):
            trajectory = self._rollout(current_state, goal)
            trajectories.append(trajectory)

        # 按目标接近度排序
        trajectories.sort(key=lambda t: t['score'], reverse=True)

        return trajectories

    def _rollout(self, initial_state: List[Object], goal: str) -> Dict:
        """单次rollout"""
        state = initial_state
        actions = []
        total_reward = 0.0

        for step in range(self.horizon):
            # 选择随机动作（简化）
            action = torch.randn(128)

            # 预测下一状态
            next_state = self.world_model.predict_next_state(state, action)

            # 计算奖励（与目标的接近度）
            goal_repr = self.world_model.encode_text(goal)
            state_repr = self._state_to_repr(next_state)
            reward = torch.cosine_similarity(
                goal_repr.unsqueeze(0), state_repr.unsqueeze(0)
            ).item()

            total_reward += reward
            actions.append({'action': action, 'reward': reward})
            state = next_state

        return {
            'actions': actions,
            'final_state': state,
            'score': total_reward / self.horizon,
        }

    def _state_to_repr(self, state: List[Object]) -> torch.Tensor:
        """将对象状态转换为向量表示"""
        if not state:
            return torch.zeros(128)

        # 平均所有对象的嵌入
        embeddings = [obj.embedding for obj in state if obj.embedding is not None]
        if not embeddings:
            return torch.zeros(128)

        return torch.stack(embeddings).mean(dim=0)


class ObjectCentricWorldModel:
    """对象中心世界模型

    整合对象检测、关系编码、状态预测和想象规划。
    """

    def __init__(self, obs_dim: int = 128, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.device = torch.device(device)

        # 组件
        self.detector = ObjectDetector(obs_dim).to(self.device)
        self.relation_encoder = ObjectRelationEncoder(obs_dim).to(self.device)
        self.state_predictor = ObjectStatePredictor(obs_dim).to(self.device)
        self.planner = ImaginationPlanner(self)

        # 对象注册表
        self.objects: Dict[str, Object] = {}

    def perceive(self, obs: torch.Tensor) -> Dict:
        """感知观测，分解为对象和关系

        Returns:
            {
                'objects': [Object, ...],
                'relations': [Relation, ...],
                'embedding': torch.Tensor,
            }
        """
        # 检测对象
        objects = self.detector(obs)

        # 更新对象注册表
        for obj in objects:
            self.objects[obj.id] = obj

        # 编码关系
        relations = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    rels = self.relation_encoder(obj1, obj2)
                    relations.extend(rels)

        return {
            'objects': objects,
            'relations': relations,
            'embedding': obs,
        }

    def predict_next_state(self, current_state: List[Object],
                          action: torch.Tensor) -> List[Object]:
        """预测下一状态"""
        next_state = []

        for obj in current_state:
            # 预测新嵌入
            new_embedding = self.state_predictor(obj, action)

            # 创建新对象
            new_obj = Object(
                id=obj.id,
                attributes=obj.attributes.copy(),
                position=obj.position,
                embedding=new_embedding,
            )
            next_state.append(new_obj)

        return next_state

    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本（简化版本）"""
        vec = torch.zeros(self.obs_dim)
        for i, c in enumerate(text[:self.obs_dim]):
            vec[i % self.obs_dim] += ord(c) / 10000.0
        return vec

    def imagine(self, goal: str) -> List[Dict]:
        """想象达成目标的方案"""
        current_state = list(self.objects.values())
        return self.planner.plan(current_state, goal)
