"""
指令 Grounding：从通信游戏到真实指令执行

核心思想：语言不是用来描述的，而是用来做事的。
指令 = 动作动词 + 物体描述

组件：
- Instruction: 指令数据结构
- InstructionGenerator: 生成指令符号
- InstructionParser: 解析指令符号
- InstructionFollower: 将指令翻译为 3D 动作序列
- InstructionGame: 指令通信游戏
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from language_emergence import (
    ACTIONS, ACTION_EFFECTS, COLORS, SHAPES, SIZES, MATERIALS,
    EmergingLanguage, _symbol_category,
)
from environment_3d import PhysicsWorld3D, MATERIALS as PHYS_MATERIALS


# ============================================================
# 指令数据结构
# ============================================================

@dataclass
class Instruction:
    """指令 = 动作动词 + 物体描述"""
    verb: str                               # 'grab', 'push', 'throw', 'drop', 'move', 'turn'
    target_features: Dict[str, str] = field(default_factory=dict)  # {'shape': 'sphere', ...}
    target_values: List[str] = field(default_factory=list)  # 所有描述值的列表
    direction: Optional[str] = None         # 'forward', 'backward', 'left', 'right' (move/turn 用)

    def to_symbols(self) -> List[str]:
        """转为符号列表"""
        symbols = [self.verb]
        if self.target_values:
            symbols.extend(self.target_values)
        else:
            for v in self.target_features.values():
                symbols.append(v)
        if self.direction:
            symbols.append(self.direction)
        return symbols

    def __repr__(self):
        return f"Instruction({self.to_symbols()})"


# ============================================================
# 指令生成器
# ============================================================

class InstructionGenerator:
    """
    生成指令符号

    根据场景和目标物体，生成最优的指令符号组合。
    类似 Speaker，但输出是指令而非描述。
    """

    def __init__(self, language: Optional[EmergingLanguage] = None):
        self.language = language

    def generate_for_target(self, target_features: Dict[str, str],
                            scene_objects: List[Dict],
                            verb: str = 'grab') -> List[str]:
        """
        为指定目标生成指令

        策略：选择最少的描述符号，使目标在场景中唯一。
        """
        # 找到能唯一区分目标的最小描述
        distinguishing = self._find_minimal_description(
            target_features, scene_objects
        )

        symbols = [verb] + distinguishing
        return symbols

    def generate_for_action(self, verb: str, direction: str = None) -> List[str]:
        """生成动作指令（不需要物体描述）"""
        symbols = [verb]
        if direction:
            symbols.append(direction)
        return symbols

    def _find_minimal_description(self, target: Dict[str, str],
                                   scene: List[Dict]) -> List[str]:
        """找到能唯一区分目标的最小符号组合"""
        candidate_keys = [k for k in target if k not in ('id', 'distance')]

        # 收集目标的所有字符串值
        target_vals = set()
        for key in candidate_keys:
            val = target[key]
            if isinstance(val, str) and val not in ('id', 'distance'):
                target_vals.add(val)

        val_list = sorted(target_vals)

        # 逐个尝试不同长度的组合，找最短的唯一描述
        for length in range(1, len(val_list) + 1):
            from itertools import combinations
            for combo in combinations(val_list, length):
                if self._description_matches_only(combo, target, scene):
                    return list(combo)

        # 回退
        return val_list

    def _description_matches_only(self, symbols, target: Dict,
                                   scene: List[Dict]) -> bool:
        """检查符号组合是否只匹配目标物体"""
        match_count = 0
        for obj in scene:
            obj_vals = set()
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        obj_vals.add(v)
            if all(s in obj_vals for s in symbols):
                match_count += 1
                if match_count > 1:
                    return False
        return match_count == 1


# ============================================================
# 指令解析器
# ============================================================

class InstructionParser:
    """
    解析指令符号序列

    识别动词和物体描述，返回 Instruction 对象。
    """

    def parse(self, symbols: List[str]) -> Optional[Instruction]:
        """解析符号列表为指令"""
        if not symbols:
            return None

        verb = None
        descriptions = []
        direction = None

        for sym in symbols:
            cat = _symbol_category(sym)
            if cat == 'action' and sym in ACTIONS:
                verb = sym
            elif sym in ('forward', 'backward', 'left', 'right',
                         'up', 'down'):
                direction = sym
            elif cat in ('shape', 'size', 'material', 'weight',
                         'bouncy', 'hardness', 'motion', 'height',
                         'color'):
                descriptions.append((cat, sym))
            else:
                # 尝试从已知集合推断类别
                from language_emergence import (
                    COLORS, SHAPES, SIZES, MATERIALS, ACTIONS as ACTS
                )
                if sym in COLORS:
                    descriptions.append(('color', sym))
                elif sym in SHAPES:
                    descriptions.append(('shape', sym))
                elif sym in SIZES:
                    descriptions.append(('size', sym))
                elif sym in MATERIALS:
                    descriptions.append(('material', sym))

        if verb is None:
            return None

        # 保留所有描述值
        target_values = [val for _, val in descriptions]
        target_features = {}
        for cat, val in descriptions:
            if cat not in target_features:
                target_features[cat] = val

        return Instruction(
            verb=verb,
            target_features=target_features,
            target_values=target_values,
            direction=direction,
        )


# ============================================================
# 指令执行器
# ============================================================

class InstructionFollower:
    """
    将指令翻译为 3D 世界的离散动作序列

    执行流程：
    1. 解析指令
    2. 识别目标物体
    3. 转向目标
    4. 前进到交互范围
    5. 执行交互动作
    """

    # 离散动作常量
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    GRAB = 8
    DROP = 9
    THROW = 10
    PUSH = 11

    GRAB_RANGE = 1.5
    INTERACTION_RANGE = 2.0

    def __init__(self):
        self.parser = InstructionParser()

    def plan_actions(self, instruction: Instruction,
                     agent_pos: np.ndarray,
                     agent_facing: float,
                     scene_objects: List[Dict],
                     held_object: Optional[int] = None) -> List[int]:
        """
        根据指令规划动作序列

        Returns:
            离散动作列表
        """
        verb = instruction.verb

        # 不需要物体的动作
        if verb == 'drop':
            if held_object is not None:
                return [self.DROP]
            return []

        if verb == 'throw':
            if held_object is not None:
                return [self.THROW]
            return []

        if verb == 'move':
            return [self._direction_to_action(instruction.direction)]

        if verb == 'turn':
            return [self._turn_to_action(instruction.direction)]

        # 需要物体的动作 (grab, push)
        target = self._find_target(instruction.target_features, scene_objects,
                                     instruction.target_values)
        if target is None:
            return []  # 目标未找到

        target_pos = np.array(target.get('position', [0, 0, 0]))
        target_id = target.get('id')

        # 检查是否已在交互范围内
        dist = np.linalg.norm(target_pos[:2] - agent_pos[:2])

        actions = []

        if verb == 'grab':
            if dist > self.GRAB_RANGE:
                actions.extend(self._navigate_to(agent_pos, agent_facing, target_pos))
            actions.append(self.GRAB)

        elif verb == 'push':
            if dist > self.INTERACTION_RANGE:
                actions.extend(self._navigate_to(agent_pos, agent_facing, target_pos))
            actions.append(self.PUSH)

        elif verb == 'throw':
            actions.append(self.THROW)

        return actions

    def _find_target(self, features: Dict[str, str],
                      scene_objects: List[Dict],
                      target_values: List[str] = None) -> Optional[Dict]:
        """在场景中找到匹配特征的目标物体"""
        # 优先使用 target_values（完整的描述值列表）
        match_values = target_values if target_values else list(features.values())

        if not match_values:
            # 没有描述，返回最近的物体
            if scene_objects:
                return min(scene_objects, key=lambda o: o.get('distance', float('inf')))
            return None

        best_obj = None
        best_score = -1

        for obj in scene_objects:
            obj_vals = set()
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        obj_vals.add(v)

            # 计算匹配的描述值数量
            score = sum(1 for val in match_values if val in obj_vals)

            if score > best_score:
                best_score = score
                best_obj = obj

        if best_score > 0:
            return best_obj
        return None

    def _navigate_to(self, agent_pos: np.ndarray, agent_facing: float,
                      target_pos: np.ndarray) -> List[int]:
        """规划从当前位置到目标位置的动作序列"""
        actions = []
        to_target = target_pos[:2] - agent_pos[:2]
        dist = np.linalg.norm(to_target)

        if dist < 0.01:
            return []

        # 计算目标方向角
        target_angle = np.arctan2(to_target[1], to_target[0])
        angle_diff = target_angle - agent_facing

        # 归一化到 [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # 转向
        if abs(angle_diff) > 0.3:  # ~17 度
            if angle_diff > 0:
                actions.append(self.TURN_RIGHT)
            else:
                actions.append(self.TURN_LEFT)

        # 前进（根据距离决定步数）
        steps = max(1, int(dist / 1.5))  # 每步约 1.5 单位
        for _ in range(min(steps, 5)):
            actions.append(self.FORWARD)

        return actions

    def _direction_to_action(self, direction: Optional[str]) -> int:
        """方向文字 → 离散动作"""
        mapping = {
            'forward': self.FORWARD,
            'backward': self.BACKWARD,
            'left': self.LEFT,
            'right': self.RIGHT,
            'up': self.UP,
            'down': self.DOWN,
        }
        return mapping.get(direction, self.FORWARD)

    def _turn_to_action(self, direction: Optional[str]) -> int:
        """转向方向 → 离散动作"""
        if direction == 'left':
            return self.TURN_LEFT
        return self.TURN_RIGHT


# ============================================================
# 指令物理执行器
# ============================================================

class InstructionExecutor:
    """
    在 3D 物理世界中执行指令

    使用闭环导航：每步重新计算方向，直到到达目标后执行交互动作。
    """

    GRAB_RANGE = 1.5
    PUSH_RANGE = 1.5
    MAX_NAV_STEPS = 100

    def execute_instruction(self, env, agent_id: int,
                            instruction: 'Instruction',
                            scene_objects: List[Dict],
                            other_agents: List[int] = None) -> Dict:
        """
        闭环执行指令：导航到目标 → 执行交互动作

        Args:
            env: MultiAgent3DEnv 实例
            agent_id: 执行指令的 Agent ID
            instruction: 解析后的指令
            scene_objects: 场景中所有物体
            other_agents: 其他 Agent ID 列表（保持静止）

        Returns:
            执行结果 dict
        """
        if other_agents is None:
            other_agents = []

        follower = InstructionFollower()
        results = {
            'steps_executed': 0,
            'held_object': None,
            'target_reached': False,
            'success': False,
        }

        verb = instruction.verb

        # 不需要物体的动作
        if verb == 'drop':
            self._step(env, agent_id, InstructionFollower.DROP, other_agents)
            results['steps_executed'] = 1
            agent = env.agents.get(agent_id)
            results['success'] = agent and agent.held_object is None
            return results

        if verb == 'throw':
            self._step(env, agent_id, InstructionFollower.THROW, other_agents)
            results['steps_executed'] = 1
            agent = env.agents.get(agent_id)
            results['success'] = agent and agent.held_object is None
            return results

        if verb == 'move':
            action = follower._direction_to_action(instruction.direction)
            for _ in range(5):
                self._step(env, agent_id, action, other_agents)
            results['steps_executed'] = 5
            results['success'] = True
            return results

        if verb == 'turn':
            action = follower._turn_to_action(instruction.direction)
            self._step(env, agent_id, action, other_agents)
            results['steps_executed'] = 1
            results['success'] = True
            return results

        # 需要物体的动作 (grab, push)：闭环导航
        target = follower._find_target(
            instruction.target_features, scene_objects,
            instruction.target_values
        )
        if target is None:
            return results

        target_pos = np.array(target.get('position', [0, 0, 0]))
        target_id = target.get('id')
        interaction_range = self.GRAB_RANGE if verb == 'grab' else self.PUSH_RANGE

        # 闭环导航：每步调整方向并前进
        # 使用微调角度（10 度/步）避免 90 度转向的振荡问题
        TURN_PER_STEP = np.radians(10)  # 每步最多转 10 度

        for step in range(self.MAX_NAV_STEPS):
            agent = env.agents.get(agent_id)
            if agent is None:
                break

            # 更新目标位置（物体可能因重力/碰撞移动）
            if target_id is not None:
                obj = env.physics._get_object(target_id)
                if obj is not None:
                    target_pos = obj.position.copy()

            dist = np.linalg.norm(agent.pos - target_pos)

            # 到达交互范围，先对准目标再执行交互
            if dist <= interaction_range:
                # 对准目标（直接设置 facing）
                to_obj = target_pos[:2] - agent.pos[:2]
                if np.linalg.norm(to_obj) > 0.01:
                    agent.facing = np.arctan2(to_obj[1], to_obj[0])

                results['target_reached'] = True
                if verb == 'grab':
                    self._step(env, agent_id, InstructionFollower.GRAB, other_agents)
                    results['steps_executed'] += 1
                    agent = env.agents.get(agent_id)
                    results['held_object'] = agent.held_object if agent else None
                    results['success'] = results['held_object'] is not None
                elif verb == 'push':
                    self._step(env, agent_id, InstructionFollower.PUSH, other_agents)
                    results['steps_executed'] += 1
                    results['success'] = True
                return results

            # 计算目标方向
            to_target = target_pos[:2] - agent.pos[:2]
            target_angle = np.arctan2(to_target[1], to_target[0])
            angle_diff = target_angle - agent.facing
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # 微调方向（直接修改 facing，避免 90 度粗转向）
            if abs(angle_diff) > 0.1:  # 偏差 > ~6 度才调整
                turn_amount = np.clip(angle_diff, -TURN_PER_STEP, TURN_PER_STEP)
                agent.facing += turn_amount

            # 前进
            self._step(env, agent_id, InstructionFollower.FORWARD, other_agents)
            results['steps_executed'] += 1

        return results

    def _step(self, env, agent_id: int, action: int,
              other_agents: List[int]):
        """执行一步（listener 执行动作，其他静止）"""
        all_actions = {agent_id: action}
        for other_id in other_agents:
            all_actions[other_id] = InstructionFollower.FORWARD  # 静止
        env.step(all_actions)


# ============================================================
# 指令通信游戏
# ============================================================

class InstructionGame:
    """
    指令通信游戏

    Speaker 观察场景，决定需要 Listener 做什么，
    生成指令。Listener 执行指令。
    成功 = 执行了正确的动作。
    """

    def __init__(self):
        self.parser = InstructionParser()
        self.follower = InstructionFollower()
        self.generator = InstructionGenerator()

    def play_grab_round(self, speaker_vocab: EmergingLanguage,
                         listener_vocab: EmergingLanguage,
                         scene_objects: List[Dict],
                         speaker_pos: np.ndarray,
                         listener_pos: np.ndarray,
                         listener_facing: float,
                         target_object: Dict) -> Dict:
        """
        执行一轮抓取指令游戏

        Speaker 告诉 Listener 去抓某个物体。
        """
        # Speaker 生成指令
        verb = 'grab'
        symbols = self.generator.generate_for_target(
            target_object, scene_objects, verb
        )

        # Listener 解析指令
        instruction = self.parser.parse(symbols)
        if instruction is None:
            return {'success': False, 'symbols': symbols, 'reason': 'parse_failed'}

        # Listener 规划动作
        actions = self.follower.plan_actions(
            instruction, listener_pos, listener_facing,
            scene_objects, held_object=None
        )

        # 评估：Listener 是否找到了正确的物体
        target = self.follower._find_target(
            instruction.target_features, scene_objects
        )
        target_id = target_object.get('id')
        found_id = target.get('id') if target else None
        success = (found_id == target_id)

        # 更新语言统计
        if speaker_vocab:
            speaker_vocab.record_usage(symbols, success)
        if listener_vocab:
            listener_vocab.record_usage(symbols, success)

        return {
            'success': success,
            'symbols': symbols,
            'instruction': instruction,
            'actions': actions,
            'target_id': target_id,
            'found_id': found_id,
        }

    def play_push_round(self, speaker_vocab: EmergingLanguage,
                         listener_vocab: EmergingLanguage,
                         scene_objects: List[Dict],
                         listener_pos: np.ndarray,
                         listener_facing: float,
                         target_object: Dict) -> Dict:
        """执行一轮推指令游戏"""
        verb = 'push'
        symbols = self.generator.generate_for_target(
            target_object, scene_objects, verb
        )

        instruction = self.parser.parse(symbols)
        if instruction is None:
            return {'success': False, 'symbols': symbols, 'reason': 'parse_failed'}

        actions = self.follower.plan_actions(
            instruction, listener_pos, listener_facing,
            scene_objects, held_object=None
        )

        target = self.follower._find_target(
            instruction.target_features, scene_objects
        )
        target_id = target_object.get('id')
        found_id = target.get('id') if target else None
        success = (found_id == target_id)

        if speaker_vocab:
            speaker_vocab.record_usage(symbols, success)
        if listener_vocab:
            listener_vocab.record_usage(symbols, success)

        return {
            'success': success,
            'symbols': symbols,
            'instruction': instruction,
            'actions': actions,
        }

    def play_free_round(self, speaker_vocab: EmergingLanguage,
                         listener_vocab: EmergingLanguage,
                         scene_objects: List[Dict],
                         listener_pos: np.ndarray,
                         listener_facing: float,
                         speaker_held: Optional[int] = None) -> Dict:
        """
        自由指令游戏

        Speaker 选择最合适的指令类型（grab/push/drop/throw/move）
        和目标，生成指令。
        """
        if not scene_objects:
            return {'success': False, 'symbols': [], 'reason': 'no_objects'}

        # 选择指令类型
        verb = self._choose_verb(scene_objects, listener_pos, speaker_held)

        if verb in ('drop', 'throw'):
            symbols = [verb]
            instruction = Instruction(verb=verb)
        elif verb in ('move', 'turn'):
            direction = np.random.choice(['forward', 'backward', 'left', 'right'])
            symbols = [verb, direction]
            instruction = Instruction(verb=verb, direction=direction)
        else:
            # 选择目标物体
            target = self._choose_target(scene_objects, listener_pos)
            symbols = self.generator.generate_for_target(
                target, scene_objects, verb
            )
            instruction = self.parser.parse(symbols)

        if instruction is None:
            return {'success': False, 'symbols': symbols, 'reason': 'parse_failed'}

        actions = self.follower.plan_actions(
            instruction, listener_pos, listener_facing,
            scene_objects, held_object=speaker_held
        )

        # 评估：动作是否合理
        success = len(actions) > 0

        if speaker_vocab:
            speaker_vocab.record_usage(symbols, success)
        if listener_vocab:
            listener_vocab.record_usage(symbols, success)

        return {
            'success': success,
            'symbols': symbols,
            'instruction': instruction,
            'actions': actions,
            'verb': verb,
        }

    def play_execution_round(self, speaker_vocab: EmergingLanguage,
                              listener_vocab: EmergingLanguage,
                              env, listener_id: int,
                              scene_objects: List[Dict],
                              target_object: Dict,
                              other_agents: List[int] = None) -> Dict:
        """
        带物理执行的指令游戏

        Speaker 生成指令 → Listener 在 3D 世界中实际执行 → 返回结果

        Args:
            env: MultiAgent3DEnv 实例
            listener_id: 执行指令的 Agent ID
            scene_objects: 场景中所有物体
            target_object: 目标物体
            other_agents: 其他 Agent ID 列表
        """
        # 获取 listener 状态
        listener = env.agents.get(listener_id)
        if listener is None:
            return {'success': False, 'reason': 'no_listener'}

        listener_pos = listener.pos
        listener_facing = listener.facing

        # Speaker 生成指令
        verb = 'grab'
        symbols = self.generator.generate_for_target(
            target_object, scene_objects, verb
        )

        # Listener 解析指令
        instruction = self.parser.parse(symbols)
        if instruction is None:
            return {'success': False, 'symbols': symbols, 'reason': 'parse_failed'}

        # 物理执行（闭环导航 + 交互）
        executor = InstructionExecutor()
        exec_result = executor.execute_instruction(
            env, listener_id, instruction,
            scene_objects=scene_objects,
            other_agents=other_agents
        )

        # 更新语言统计
        success = exec_result['success']
        if speaker_vocab:
            speaker_vocab.record_usage(symbols, success)
        if listener_vocab:
            listener_vocab.record_usage(symbols, success)

        return {
            'success': success,
            'symbols': symbols,
            'instruction': instruction,
            'execution': exec_result,
            'target_id': target_object.get('id'),
            'held_object': exec_result.get('held_object'),
        }

    def _choose_verb(self, scene_objects: List[Dict],
                      listener_pos: np.ndarray,
                      speaker_held: Optional[int]) -> str:
        """选择最合适的动词"""
        # 如果有物体在附近，grab 或 push
        nearby = [o for o in scene_objects if o.get('distance', 999) < 3.0]
        if nearby:
            return np.random.choice(['grab', 'push'])

        # 如果有远物体，move
        far = [o for o in scene_objects if o.get('distance', 0) > 3.0]
        if far:
            return 'move'

        return 'grab'

    def _choose_target(self, scene_objects: List[Dict],
                        listener_pos: np.ndarray) -> Dict:
        """选择目标物体（偏好近的）"""
        if not scene_objects:
            return {}

        # 按距离排序，偏好近的（带随机性）
        sorted_objs = sorted(scene_objects, key=lambda o: o.get('distance', 999))
        # 以 70% 概率选最近的，30% 随机
        if np.random.random() < 0.7 and sorted_objs:
            return sorted_objs[0]
        return np.random.choice(scene_objects) if scene_objects else {}
