"""核心知识先验系统 — Spelke 核心知识系统的计算实现

认知科学背景：
    Elizabeth Spelke (MIT/Harvard) 通过数十年婴儿实验发现，
    人类婴儿天生具有6个领域特定的核心知识系统（Spelke & Kinzler, 2007）：

    1. ObjectSystem — 物体持久性(object permanence)、连续性、内聚性
       婴儿预期：物体不会凭空消失，被遮挡的物体仍然存在
       实验证据：Baillargeon (1987) 违反预期范式

    2. NumberSystem — 近似数系统(Approximate Number System, ANS)
       婴儿能区分数量比例（Weber分数 λ≈0.15）
       实验证据：Xu & Spelke (2000) 6月大婴儿区分8 vs 16

    3. AgentSystem — 自主行为识别、目标归因
       婴儿预期：自主运动的实体有目标和意图
       实验证据：Gergely et al. (1995) 理性模仿实验

    4. GeometrySystem — 距离、方向、形状感知
       婴儿能用几何信息重新定向
       实验证据：Hermer & Spelke (1996)

    5. SocialSystem — 社会伙伴识别、共同注意
       婴儿预期：他人是和自己一样有心理状态的存在
       实验证据：Onishi & Baillargeon (2005) 错误信念理解

    6. CausalitySystem — 因果检测
       婴儿预期：因果需要时间和空间上的接触
       实验证据：Leslie & Keeble (1987) 米奇老鼠因果感知

核心设计：
    这些先验不是"知识"，而是"预期"——它们为学习提供约束，
    缩小假设空间，使从少量数据学习成为可能。

    例如：
    - 没有物体持久性先验：看到物体消失 → 需要学习"物体可能还在"
    - 有物体持久性先验：看到物体消失 → 自动预测"物体还在后面"
      → 只有当预测被违反时才产生"意外"信号 → 驱动学习

参考文献：
    Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge.
    Developmental Science, 10(1), 89-96.
"""

import torch
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ===========================================================================
# 1. 物体系统 — Object Permanence
# ===========================================================================

class ObjectSystem:
    """物体核心知识系统

    进化先验：
    - 物体是内聚的（不会突然分裂成两半）
    - 物体是连续的（不会瞬间移动）
    - 物体是持久的（被遮挡不会消失）
    - 物体有边界（两个物体不能同时占据同一位置）

    计算实现：
    维护一个"物体记忆"——即使物体不可见，也保留其预期存在。
    当预期被违反时，生成意外信号（surprise）。
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        # 物体记忆：{object_id: {last_pos, velocity, confidence, time_since_seen}}
        self.object_memory: Dict[str, Dict] = {}
        # 物体持久性衰减率：每步confidence *= persistence_decay
        self.persistence_decay = 0.98
        # 意外阈值：confidence > surprise_threshold时消失才算"意外"
        self.surprise_threshold = 0.3

    def process(self, visible_objects: Dict[str, Dict]) -> Dict:
        """处理一帧感知输入，维护物体记忆

        Args:
            visible_objects: {obj_id: {'position': [x,y], 'velocity': [vx,vy], ...}}

        Returns:
            {
                'expected_but_missing': [...],  # 预期存在但不可见的物体
                'surprise': float,              # 违反预期的意外程度
                'all_objects': Dict,             # 可见+预期物体
                'predictions': Dict,            # 对每个物体的位置预测
            }
        """
        surprise = 0.0
        expected_but_missing = []
        predictions = {}

        # 1. 更新可见物体的记忆
        for obj_id, props in visible_objects.items():
            pos = props.get('position', [0.0, 0.0])
            vel = props.get('velocity', [0.0, 0.0])

            self.object_memory[obj_id] = {
                'last_pos': pos,
                'velocity': vel,
                'confidence': 1.0,  # 可见=100%置信
                'time_since_seen': 0,
            }
            # 预测下一帧位置（匀速运动假设）
            predictions[obj_id] = {
                'predicted_pos': [pos[0] + vel[0], pos[1] + vel[1]],
            }

        # 2. 检查记忆中但不可见的物体
        visible_ids = set(visible_objects.keys())
        for obj_id, memory in list(self.object_memory.items()):
            if obj_id not in visible_ids:
                # 物体不可见 → 增加时间计数，衰减置信度
                memory['time_since_seen'] += 1
                memory['confidence'] *= self.persistence_decay

                # 如果置信度仍然较高 → 预期物体存在
                if memory['confidence'] > self.surprise_threshold:
                    expected_but_missing.append({
                        'id': obj_id,
                        'confidence': memory['confidence'],
                        'last_pos': memory['last_pos'],
                        # 基于速度预测当前位置
                        'predicted_pos': [
                            memory['last_pos'][0] + memory['velocity'][0] * memory['time_since_seen'],
                            memory['last_pos'][1] + memory['velocity'][1] * memory['time_since_seen'],
                        ]
                    })

                # 如果置信度太低 → 清除记忆（物体确实消失了）
                if memory['confidence'] < 0.05:
                    del self.object_memory[obj_id]

        # 3. 计算意外信号
        # 物体突然消失（高置信度但不可见）= 违反持久性
        for missing in expected_but_missing:
            surprise += missing['confidence']  # 越确信存在却消失了 → 越意外

        return {
            'expected_but_missing': expected_but_missing,
            'surprise': min(surprise, 2.0),
            'all_objects': {**visible_objects, **{
                m['id']: {'position': m['predicted_pos'], 'inferred': True}
                for m in expected_but_missing
            }},
            'predictions': predictions,
        }

    def check_violation(self, event: Dict) -> float:
        """检查特定事件是否违反物体先验

        Args:
            event: {'type': 'disappear'|'appear'|'teleport', 'object_id': str, ...}

        Returns:
            违反程度 [0, 1]，0=不违反，1=严重违反
        """
        event_type = event.get('type', '')

        if event_type == 'disappear':
            # 物体凭空消失 → 违反持久性
            obj_id = event.get('object_id', '')
            memory = self.object_memory.get(obj_id)
            if memory and memory['confidence'] > 0.5:
                return memory['confidence']  # 越确信 → 违反越严重
            return 0.1  # 低置信度消失不太意外

        elif event_type == 'appear':
            # 物体凭空出现 → 违反连续性（轻微）
            return 0.2

        elif event_type == 'teleport':
            # 物体瞬间移动 → 违反连续性
            distance = event.get('distance', 0.0)
            return min(1.0, distance / 5.0)  # 距离越远越意外

        return 0.0


# ===========================================================================
# 2. 近似数系统 — Approximate Number System (ANS)
# ===========================================================================

class NumberSystem:
    """近似数系统 — Weber分数感知

    进化先验：
    人类（和很多动物）天生具有近似数感知能力。
    精度受 Weber 分数约束：能区分 n1 和 n2 当且仅当
    |n1 - n2| / min(n1, n2) > λ（Weber分数，成人≈0.15，婴儿≈0.5）。

    计算实现：
    给定两个数量，预测"它们是否不同"以及"差异有多大"。
    遵循 Weber-Fechner 定律：感知差异 ∝ log(n1/n2)。
    """

    def __init__(self, weber_fraction: float = 0.25):
        """
        Args:
            weber_fraction: Weber分数λ，越小越精确
                           成人≈0.15，6月婴儿≈0.5，本研究取中间值
        """
        self.weber_fraction = weber_fraction

    def compare(self, n1: float, n2: float) -> Dict:
        """比较两个数量

        Returns:
            {
                'discriminable': bool,      # 能否区分
                'ratio': float,             # n1/n2
                'weber_distance': float,    # Weber距离 |n1-n2|/min(n1,n2)
                'larger': int,              # 哪个更大
                'confidence': float,        # 区分的置信度 [0,1]
            }
        """
        if n1 <= 0 or n2 <= 0:
            return {
                'discriminable': False, 'ratio': 1.0,
                'weber_distance': 0.0, 'larger': 0, 'confidence': 0.0,
            }

        ratio = max(n1, n2) / min(n1, n2)
        weber_distance = abs(n1 - n2) / min(n1, n2)

        # Weber定律：区分概率 = sigmoid((ratio - 1) / weber_fraction)
        discriminability = 1.0 / (1.0 + math.exp(-(ratio - 1.0) / self.weber_fraction))
        discriminable = weber_distance > self.weber_fraction

        return {
            'discriminable': discriminable,
            'ratio': ratio,
            'weber_distance': weber_distance,
            'larger': 1 if n1 > n2 else (2 if n2 > n1 else 0),
            'confidence': discriminability,
        }

    def estimate_count(self, items: list) -> Dict:
        """估计一组物品的数量（ANS近似）

        人类ANS的精度随数量增加而下降：
        小数量（1-4）→ 精确（subitizing）
        大数量（5+）→ 近似（ANS）
        """
        true_count = len(items)

        if true_count <= 4:
            # Subitizing范围：精确计数
            return {
                'estimate': true_count,
                'confidence': 0.95,
                'range': (true_count, true_count),
            }
        else:
            # ANS范围：近似估计，标准差 = weber * count
            std = self.weber_fraction * true_count
            estimate = true_count  # 无偏估计
            return {
                'estimate': estimate,
                'confidence': max(0.3, 1.0 - std / true_count),
                'range': (max(1, int(estimate - 1.5 * std)), int(estimate + 1.5 * std)),
            }

    def check_conservation(self, before_count: int, after_count: int,
                           transformation: str = 'unknown') -> float:
        """检查数量守恒是否被违反

        幼儿（4-5岁前）在Piaget守恒任务中失败，
        但更基本的数量守恒预期在婴儿期就已存在。
        """
        if transformation in ('add', 'remove', 'create', 'destroy'):
            # 这些变换预期会改变数量，不违反
            return 0.0

        # 其他变换（移动、排列变化）不应改变数量
        if before_count != after_count:
            return min(1.0, abs(before_count - after_count) / max(before_count, 1))

        return 0.0


# ===========================================================================
# 3. 代理系统 — Agency Detection
# ===========================================================================

class AgentSystem:
    """代理检测系统

    进化先验：
    人类天生能区分"自主运动"（有目标的代理）和"物理运动"（被动的物体）。
    判断标准：
    - 自主运动：非匀速、非直线、似乎有目标
    - 物理运动：符合惯性定律（匀速直线，或受外力改变方向）

    计算实现：
    分析运动轨迹，检测是否偏离物理预测（惯性路径）。
    偏离越大 → 越可能是自主代理。
    """

    def __init__(self):
        # 运动历史：{entity_id: [(pos_x, pos_y, vel_x, vel_y), ...]}
        self.trajectory_history: Dict[str, List[Tuple]] = defaultdict(list)
        self.max_history = 20

    def analyze_motion(self, entity_id: str,
                       position: Tuple[float, float],
                       velocity: Tuple[float, float]) -> Dict:
        """分析一个实体的运动，判断是代理还是物体

        Args:
            entity_id: 实体标识
            position: 当前位置 (x, y)
            velocity: 当前速度 (vx, vy)

        Returns:
            {
                'is_agent': bool,              # 是否是代理
                'agency_score': float,         # 代理程度 [0,1]
                'has_goal': bool,              # 是否检测到目标导向
                'deviation_from_inertia': float, # 偏离惯性路径的程度
            }
        """
        # 记录轨迹
        self.trajectory_history[entity_id].append(
            (position[0], position[1], velocity[0], velocity[1])
        )
        if len(self.trajectory_history[entity_id]) > self.max_history:
            self.trajectory_history[entity_id] = self.trajectory_history[entity_id][-self.max_history:]

        trajectory = self.trajectory_history[entity_id]

        if len(trajectory) < 3:
            # 数据不足，无法判断
            return {
                'is_agent': False,
                'agency_score': 0.0,
                'has_goal': False,
                'deviation_from_inertia': 0.0,
            }

        # 1. 计算偏离惯性路径的程度
        deviation = self._compute_inertia_deviation(trajectory)

        # 2. 检测目标导向性（运动是否指向某个目标）
        goal_directedness = self._compute_goal_directedness(trajectory)

        # 3. 检测自主性（速度变化是否由外力引起 vs 自发）
        spontaneity = self._compute_spontaneity(trajectory)

        # 4. 综合评分
        agency_score = min(1.0, deviation * 0.4 + goal_directedness * 0.3 + spontaneity * 0.3)

        return {
            'is_agent': agency_score > 0.5,
            'agency_score': agency_score,
            'has_goal': goal_directedness > 0.5,
            'deviation_from_inertia': deviation,
        }

    def _compute_inertia_deviation(self, trajectory: List[Tuple]) -> float:
        """计算运动轨迹偏离惯性路径的程度

        惯性预测：物体应该按当前速度匀速直线运动。
        偏离越大 → 越可能是自主运动。
        """
        if len(trajectory) < 3:
            return 0.0

        total_deviation = 0.0
        count = 0

        for i in range(2, len(trajectory)):
            # 两步前的位置和速度
            prev_pos = trajectory[i - 1][:2]
            prev_vel = trajectory[i - 1][2:4]

            # 惯性预测
            predicted_x = prev_pos[0] + prev_vel[0]
            predicted_y = prev_pos[1] + prev_vel[1]

            # 实际位置
            actual_x, actual_y = trajectory[i][0], trajectory[i][1]

            # 偏差
            dx = actual_x - predicted_x
            dy = actual_y - predicted_y
            deviation = math.sqrt(dx * dx + dy * dy)

            total_deviation += deviation
            count += 1

        if count == 0:
            return 0.0

        avg_deviation = total_deviation / count
        # 归一化到 [0, 1]，假设最大合理偏差约2.0
        return min(1.0, avg_deviation / 2.0)

    def _compute_goal_directedness(self, trajectory: List[Tuple]) -> float:
        """检测运动是否指向某个目标（运动方向的一致性）"""
        if len(trajectory) < 4:
            return 0.0

        # 计算速度方向的变化频率
        direction_changes = 0
        for i in range(2, len(trajectory)):
            vx1, vy1 = trajectory[i - 1][2], trajectory[i - 1][3]
            vx2, vy2 = trajectory[i][2], trajectory[i][3]

            # 方向角变化
            angle1 = math.atan2(vy1, vx1)
            angle2 = math.atan2(vy2, vx2)
            angle_diff = abs(angle2 - angle1)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff

            if angle_diff > math.pi / 4:  # 超过45度算一次方向变化
                direction_changes += 1

        # 适度变化表示目标导向（不是太随机也不是太直线）
        n = len(trajectory) - 2
        if n == 0:
            return 0.0
        change_rate = direction_changes / n

        # 倒U型：完全不变(0)或频繁变化(1)都是低目标导向
        goal_score = 1.0 - abs(change_rate - 0.3) / 0.7
        return max(0.0, min(1.0, goal_score))

    def _compute_spontaneity(self, trajectory: List[Tuple]) -> float:
        """检测运动的自发性（速度变化是否与碰撞等外力相关）"""
        if len(trajectory) < 3:
            return 0.0

        # 计算加速度变化（非外力引起的加速度=自发性）
        accelerations = []
        for i in range(1, len(trajectory)):
            dvx = trajectory[i][2] - trajectory[i - 1][2]
            dvy = trajectory[i][3] - trajectory[i - 1][3]
            accel = math.sqrt(dvx * dvx + dvy * dvy)
            accelerations.append(accel)

        if not accelerations:
            return 0.0

        # 高加速度变化 → 可能是自发运动
        avg_accel = sum(accelerations) / len(accelerations)
        return min(1.0, avg_accel / 1.0)  # 归一化


# ===========================================================================
# 4. 几何系统 — Geometry
# ===========================================================================

class GeometrySystem:
    """几何核心知识系统

    进化先验：
    婴儿对距离、方向、角度有基本感知。
    能用几何信息重新定向（Hermer & Spelke, 1996）。
    """

    def __init__(self):
        pass

    def compute_distance(self, pos1: Tuple[float, float],
                         pos2: Tuple[float, float]) -> float:
        """计算两点间的欧氏距离"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def compute_direction(self, from_pos: Tuple[float, float],
                          to_pos: Tuple[float, float]) -> float:
        """计算从from_pos指向to_pos的角度（弧度）"""
        return math.atan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])

    def classify_shape(self, vertices: List[Tuple[float, float]]) -> Dict:
        """分类几何形状

        Args:
            vertices: 多边形顶点列表

        Returns:
            {'type': 'triangle'|'rectangle'|'circle'|'irregular'|...,
             'regularity': float,  # 正则程度 [0,1]
             'symmetry': float}    # 对称程度 [0,1]
        """
        n = len(vertices)
        if n < 3:
            return {'type': 'point', 'regularity': 1.0, 'symmetry': 1.0}

        # 计算所有边长
        edges = []
        for i in range(n):
            j = (i + 1) % n
            edges.append(self.compute_distance(vertices[i], vertices[j]))

        if not edges or max(edges) == 0:
            return {'type': 'point', 'regularity': 0.0, 'symmetry': 0.0}

        # 边长的一致性（正则性指标）
        avg_edge = sum(edges) / len(edges)
        edge_variance = sum((e - avg_edge) ** 2 for e in edges) / len(edges)
        regularity = 1.0 / (1.0 + edge_variance / (avg_edge ** 2 + 1e-8))

        # 角度分析
        angles = []
        for i in range(n):
            prev_v = vertices[(i - 1) % n]
            curr_v = vertices[i]
            next_v = vertices[(i + 1) % n]
            angle = self._compute_angle(prev_v, curr_v, next_v)
            angles.append(angle)

        # 类型判断
        if n == 3:
            shape_type = 'triangle'
        elif n == 4:
            # 检查是否接近矩形（4个角接近90度）
            avg_angle = sum(angles) / len(angles)
            if abs(avg_angle - math.pi / 2) < 0.3:
                if regularity > 0.8:
                    shape_type = 'square'
                else:
                    shape_type = 'rectangle'
            else:
                shape_type = 'quadrilateral'
        elif n >= 8 and regularity > 0.9:
            shape_type = 'circle'
        else:
            shape_type = 'polygon'

        return {
            'type': shape_type,
            'regularity': regularity,
            'symmetry': regularity,  # 简化：正则性≈对称性
            'n_vertices': n,
            'edges': edges,
        }

    def _compute_angle(self, p1, p2, p3):
        """计算 p1-p2-p3 形成的角度"""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 * mag2 == 0:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.acos(cos_angle)


# ===========================================================================
# 5. 社会系统 — Social Partner Detection
# ===========================================================================

class SocialSystem:
    """社会核心知识系统

    进化先验：
    婴儿天生偏好人脸、人声，预期他人是有心理状态的社会伙伴。
    共同注意（joint attention）是社会学习的基础。
    """

    def __init__(self):
        # 社会伙伴记录
        self.known_agents: Dict[str, Dict] = {}
        # 共同注意状态
        self.joint_attention_target: Optional[str] = None

    def detect_social_partner(self, entity_features: Dict) -> Dict:
        """检测一个实体是否是社会伙伴

        社会伙伴特征（婴儿偏好）：
        - 面状轮廓（椭圆+内部特征）
        - 双眼偏好
        - 生物运动
        - 声音产出
        """
        features = entity_features

        # 面部特征检测
        face_score = 0.0
        if features.get('has_face_like_shape'):
            face_score += 0.4
        if features.get('has_eyes'):
            face_score += 0.3
        if features.get('produces_sound'):
            face_score += 0.2
        if features.get('is_agent'):  # 来自AgentSystem
            face_score += 0.1

        is_social = face_score > 0.5

        if is_social:
            entity_id = features.get('id', 'unknown')
            if entity_id not in self.known_agents:
                self.known_agents[entity_id] = {
                    'encounters': 0,
                    'gaze_direction': None,
                    'attention_target': None,
                }
            self.known_agents[entity_id]['encounters'] += 1

        return {
            'is_social_partner': is_social,
            'social_score': face_score,
        }

    def update_joint_attention(self, agent_id: str,
                               attention_target: Optional[str]) -> Dict:
        """更新共同注意状态

        共同注意 = 两个人关注同一个对象。
        这是词汇学习的关键机制（父母指向杯子说"杯子"）。
        """
        if agent_id in self.known_agents:
            self.known_agents[agent_id]['attention_target'] = attention_target

        # 检查是否形成共同注意
        shared = False
        if attention_target and self.joint_attention_target == attention_target:
            shared = True
        self.joint_attention_target = attention_target

        return {
            'joint_attention': shared,
            'target': attention_target,
        }


# ===========================================================================
# 6. 因果系统 — Causality Detection
# ===========================================================================

class CausalitySystem:
    """因果检测核心知识系统

    进化先验：
    婴儿预期因果需要时间和空间上的接触（Michotte, 1963）。
    - 因果传递：A碰撞B → B移动（接触因果）
    - 时序约束：原因必须先于结果
    - 空间约束：因果传递需要空间邻近

    计算实现：
    检测事件序列是否符合因果先验，不符合则产生意外信号。
    """

    def __init__(self):
        # 近期事件缓冲
        self.recent_events: List[Dict] = []
        self.max_events = 30
        # 因果链记录
        self.causal_pairs: Dict[str, float] = defaultdict(float)

    def observe_event(self, event: Dict) -> Dict:
        """观察一个事件，检测是否符合因果先验

        Args:
            event: {'type': str, 'source': str, 'target': str,
                    'position': [x,y], 'time': float, ...}

        Returns:
            {'causal_score': float, 'causal_chain': List, 'surprise': float}
        """
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]

        # 1. 检测接触因果（碰撞引发运动）
        causal_score = 0.0
        causal_chain = []
        surprise = 0.0

        source = event.get('source', '')
        target = event.get('target', '')
        event_type = event.get('type', '')
        position = event.get('position', [0.0, 0.0])

        if event_type == 'motion' and source and target:
            # 检查是否有接触事件先于运动
            for prev_event in reversed(self.recent_events[:-1]):
                if prev_event.get('type') == 'contact':
                    prev_source = prev_event.get('source', '')
                    prev_target = prev_event.get('target', '')
                    prev_pos = prev_event.get('position', [0.0, 0.0])

                    # 时序检查：接触在前
                    time_gap = event.get('time', 0) - prev_event.get('time', 0)
                    if 0 < time_gap < 2.0:  # 2秒内的因果窗口
                        # 空间邻近检查
                        dist = math.sqrt(
                            (position[0] - prev_pos[0]) ** 2 +
                            (position[1] - prev_pos[1]) ** 2
                        )
                        if dist < 3.0:  # 空间邻近阈值
                            causal_score = max(0.0, 1.0 - dist / 3.0)
                            causal_chain = [
                                {'event': 'contact', 'source': prev_source, 'target': prev_target},
                                {'event': 'motion', 'source': source, 'target': target},
                            ]
                            # 记录因果对
                            key = f"{prev_source}->{target}"
                            self.causal_pairs[key] += causal_score
                            break

        elif event_type == 'motion' and not source:
            # 运动但没有先前的接触 → 违反因果先验（可能是自发运动）
            # 检查是否有接触事件
            has_contact = any(
                e.get('type') == 'contact' for e in self.recent_events[-5:]
            )
            if not has_contact:
                # 无因运动 → 轻微意外
                surprise = 0.3

        return {
            'causal_score': causal_score,
            'causal_chain': causal_chain,
            'surprise': surprise,
        }

    def get_causal_expectation(self, cause_type: str) -> Dict:
        """根据先验预测因果效应

        例如：碰撞 → 预期被碰物体运动
        """
        expectations = {
            'contact': {'expected_effect': 'motion', 'confidence': 0.8},
            'push': {'expected_effect': 'motion', 'confidence': 0.9},
            'release': {'expected_effect': 'fall', 'confidence': 0.7},
            'heat': {'expected_effect': 'temperature_change', 'confidence': 0.6},
        }
        return expectations.get(cause_type, {'expected_effect': 'unknown', 'confidence': 0.1})


# ===========================================================================
# 核心 — 统一系统
# ===========================================================================

class CoreKnowledgeSystem:
    """核心知识系统 — 统一入口

    整合6个Spelke核心系统，提供：
    1. 感知预处理（对输入施加先验约束）
    2. 学习偏差（为后续学习提供先验）
    3. 意外信号（违反先验时的surprise）
    4. 概念评估（评估候选概念是否符合先验）

    使用方式：
        core_knowledge = CoreKnowledgeSystem()
        # 处理感知输入
        result = core_knowledge.process_observation(features)
        # 获取学习偏差
        biases = core_knowledge.provide_learning_biases()
        # 评估意外
        surprise = core_knowledge.get_surprise(observed, predicted)
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

        # 6个子系统
        self.objects = ObjectSystem(device=device)
        self.numbers = NumberSystem(weber_fraction=0.25)
        self.agents = AgentSystem()
        self.geometry = GeometrySystem()
        self.social = SocialSystem()
        self.causality = CausalitySystem()

        # 意外历史
        self._surprise_history: List[float] = []

    def process_observation(self, obs_features: Dict) -> Dict:
        """对感知输入施加核心知识先验

        Args:
            obs_features: {
                'visible_objects': Dict,   # 可见物体 {id: {position, velocity, ...}}
                'events': List[Dict],      # 发生的事件
                'entities': List[Dict],    # 实体列表
            }

        Returns:
            {
                'object_priors': Dict,     # 物体系统结果
                'number_estimate': Dict,   # 数量估计
                'agency_detection': Dict,  # 代理检测结果
                'social_detection': Dict,  # 社会伙伴检测结果
                'causal_detection': Dict,  # 因果检测结果
                'total_surprise': float,   # 总意外信号
            }
        """
        total_surprise = 0.0

        # 1. 物体系统
        visible_objects = obs_features.get('visible_objects', {})
        object_result = self.objects.process(visible_objects)
        total_surprise += object_result.get('surprise', 0.0)

        # 2. 数量系统
        entities = obs_features.get('entities', [])
        number_result = self.numbers.estimate_count(entities)

        # 3. 代理检测
        agency_results = {}
        for entity in entities:
            entity_id = entity.get('id', '')
            if entity_id:
                pos = entity.get('position', (0.0, 0.0))
                vel = entity.get('velocity', (0.0, 0.0))
                agency_results[entity_id] = self.agents.analyze_motion(entity_id, pos, vel)

        # 4. 社会检测
        social_results = {}
        for entity in entities:
            entity_id = entity.get('id', '')
            if entity_id:
                # 注入代理检测结果
                entity_with_agency = {**entity, 'is_agent': agency_results.get(entity_id, {}).get('is_agent', False)}
                social_results[entity_id] = self.social.detect_social_partner(entity_with_agency)

        # 5. 因果检测
        causal_results = {}
        events = obs_features.get('events', [])
        for event in events:
            event_result = self.causality.observe_event(event)
            if event_result['causal_score'] > 0 or event_result['surprise'] > 0:
                causal_results[event.get('type', 'unknown')] = event_result
                total_surprise += event_result.get('surprise', 0.0)

        # 记录意外
        self._surprise_history.append(total_surprise)

        return {
            'object_priors': object_result,
            'number_estimate': number_result,
            'agency_detection': agency_results,
            'social_detection': social_results,
            'causal_detection': causal_results,
            'total_surprise': total_surprise,
        }

    def provide_learning_biases(self) -> Dict:
        """为后续学习提供先验偏差

        这些偏差模拟了人类婴儿天生的学习偏好：
        1. whole_object_bias — 新标签指向整个物体
        2. mutual_exclusivity — 每个对象只有一个标签
        3. shape_bias — 按形状分类（非颜色/材质）
        4. causality_prior — 接触→因果的时间优先偏好
        5. agency_prior — 自主运动→有意图的偏好
        """
        return {
            # 标签学习偏差（Carey & Bartlett, 1978; Markman, 1989）
            'whole_object_bias': 0.8,       # 新标签→整个物体的概率
            'mutual_exclusivity': 0.7,      # 已有标签→推断新类别的概率
            'shape_bias': 0.6,              # 按形状（vs 颜色/材质）分类的偏好

            # 因果学习偏差（Gopnik et al., 2004）
            'causality_prior': 0.7,         # 接触后运动→因果的先验概率
            'temporal_priority': 0.8,       # 原因先于结果的先验

            # 代理感知偏差
            'agency_prior': 0.5,            # 非惯性运动→有意图的先验
            'gaze_following': 0.6,          # 跟随视线的偏好

            # 数量感知偏差
            'weber_fraction': self.numbers.weber_fraction,
            'subitizing_range': 4,          # 精确计数范围

            # 物体感知偏差
            'object_persistence': 0.98,     # 物体持久性置信衰减率
            'cohesion': 0.9,                # 物体内聚性先验
        }

    def get_surprise(self, observed: Dict, predicted: Dict) -> float:
        """基于核心知识先验计算意外信号

        意外 = 违反先验预测的程度。
        高意外 → 驱动学习（"这和我预期的不一样！"）
        低意外 → 确认已有知识（"果然如此"）

        Args:
            observed: 实际观察到的特征
            predicted: 根据先验预测的特征

        Returns:
            意外程度 [0, ∞)
        """
        surprise = 0.0

        # 1. 物体意外
        if 'events' in observed:
            for event in observed['events']:
                event_type = event.get('type', '')
                if event_type in ('disappear', 'teleport'):
                    surprise += self.objects.check_violation(event)

        # 2. 数量意外
        if 'count_before' in observed and 'count_after' in observed:
            transform = observed.get('transformation', 'unknown')
            conservation_violation = self.numbers.check_conservation(
                observed['count_before'], observed['count_after'], transform
            )
            surprise += conservation_violation

        # 3. 因果意外
        if 'events' in observed:
            for event in observed['events']:
                causal_result = self.causality.observe_event(event)
                surprise += causal_result.get('surprise', 0.0)

        return surprise

    def score_concept_candidate(self, concept_data: Dict) -> float:
        """用先验偏差评估一个候选概念的质量

        高分 → 符合先验 → 概念可能是有意义的
        低分 → 违反先验 → 概念可能是噪声/碎片
        """
        biases = self.provide_learning_biases()
        score = 0.0

        label = concept_data.get('label', '')
        features = concept_data.get('features', {})
        context = concept_data.get('context', {})

        # 1. 长度偏好：2-4个字符的概念更可能是有意义的（中文）
        if 2 <= len(label) <= 4:
            score += 0.3
        elif len(label) == 1:
            score -= 0.2
        elif len(label) > 6:
            score -= 0.1

        # 2. 感知绑定加分：有感知特征的概念更接地
        if features.get('has_perceptual_binding'):
            score += 0.3 * biases['whole_object_bias']

        # 3. 形状偏好：按形状分类的概念
        if features.get('shape_category'):
            score += 0.2 * biases['shape_bias']

        # 4. 频率合理性：频率3-50的概念更可能是真词
        freq = concept_data.get('frequency', 0)
        if 3 <= freq <= 50:
            score += 0.2
        elif freq > 100:
            score -= 0.1  # 过于频繁可能是功能词

        return max(0.0, min(1.0, score))

    def get_stats(self) -> Dict:
        """获取核心知识系统统计"""
        return {
            'objects_tracked': len(self.objects.object_memory),
            'agents_detected': len(self.agents.trajectory_history),
            'social_partners': len(self.social.known_agents),
            'causal_pairs': len(self.causality.causal_pairs),
            'avg_surprise': (sum(self._surprise_history[-100:]) / max(1, len(self._surprise_history[-100:]))
                            if self._surprise_history else 0.0),
        }
