"""Layer 4: 世界模型层

建立对世界的内部模型，能预测和模拟。

核心能力：
1. 物理直觉 — 物体如何运动、重力、碰撞
2. 社会直觉 — 人的行为和动机
3. 时间模型 — 事件的先后顺序
4. 空间模型 — 位置和方向
5. 预测能力 — 基于模型预测未来

运行方式：
    python training/layers/world_model.py
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PhysicalObject:
    """物理对象"""
    name: str
    position: Tuple[float, float, float] = (0, 0, 0)  # x, y, z
    velocity: Tuple[float, float, float] = (0, 0, 0)
    mass: float = 1.0
    size: Tuple[float, float, float] = (1, 1, 1)
    properties: Dict[str, Any] = field(default_factory=dict)

    def move(self, dt: float):
        """移动物体"""
        x = self.position[0] + self.velocity[0] * dt
        y = self.position[1] + self.velocity[1] * dt
        z = self.position[2] + self.velocity[2] * dt
        self.position = (x, y, z)

    def apply_force(self, force: Tuple[float, float, float], dt: float):
        """施加力"""
        # F = ma, a = F/m
        ax = force[0] / self.mass
        ay = force[1] / self.mass
        az = force[2] / self.mass

        # 更新速度
        vx = self.velocity[0] + ax * dt
        vy = self.velocity[1] + ay * dt
        vz = self.velocity[2] + az * dt
        self.velocity = (vx, vy, vz)


@dataclass
class Event:
    """事件"""
    name: str
    timestamp: float
    duration: float = 0.0
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    cause: Optional[str] = None
    effects: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """智能体（人或动物）"""
    name: str
    beliefs: Dict[str, Any] = field(default_factory=dict)
    desires: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)
    knowledge: Set[str] = field(default_factory=set)


class WorldModel:
    """世界模型

    核心能力：
    - 模拟物理世界
    - 理解社会行为
    - 预测未来状态
    """

    def __init__(self):
        # 物理世界
        self.objects: Dict[str, PhysicalObject] = {}
        self.gravity = (0, -9.8, 0)  # 重力加速度

        # 社会世界
        self.agents: Dict[str, Agent] = {}

        # 时间线
        self.events: List[Event] = []
        self.current_time = 0.0

        # 空间模型
        self.locations: Dict[str, Dict] = {}

        # 因果规则
        self.causal_rules: Dict[str, List[str]] = defaultdict(list)

        # 统计
        self.stats = {
            'objects_tracked': 0,
            'events_recorded': 0,
            'predictions_made': 0,
        }

    def add_object(self, name: str, **kwargs) -> PhysicalObject:
        """添加物理对象"""
        obj = PhysicalObject(name=name, **kwargs)
        self.objects[name] = obj
        self.stats['objects_tracked'] += 1
        return obj

    def add_agent(self, name: str, **kwargs) -> Agent:
        """添加智能体"""
        agent = Agent(name=name, **kwargs)
        self.agents[name] = agent
        return agent

    def add_event(self, name: str, **kwargs) -> Event:
        """添加事件"""
        event = Event(name=name, timestamp=self.current_time, **kwargs)
        self.events.append(event)
        self.stats['events_recorded'] += 1

        # 更新因果链
        if event.cause:
            self.causal_rules[event.cause].append(event.name)

        return event

    def simulate_physics(self, dt: float, steps: int = 1):
        """模拟物理过程"""
        for _ in range(steps):
            for obj in self.objects.values():
                # 施加重力
                obj.apply_force(self.gravity, dt)

                # 移动物体
                obj.move(dt)

                # 地面碰撞检测
                if obj.position[1] < 0:
                    obj.position = (obj.position[0], 0, obj.position[2])
                    obj.velocity = (obj.velocity[0], 0, obj.velocity[2])

            self.current_time += dt

    def predict_next_state(self, object_name: str, dt: float) -> Dict:
        """预测物体的下一个状态"""
        if object_name not in self.objects:
            return {}

        obj = self.objects[object_name]

        # 简单物理预测
        new_x = obj.position[0] + obj.velocity[0] * dt
        new_y = obj.position[1] + obj.velocity[1] * dt - 0.5 * 9.8 * dt * dt
        new_z = obj.position[2] + obj.velocity[2] * dt

        self.stats['predictions_made'] += 1

        return {
            'position': (new_x, new_y, new_z),
            'velocity': (obj.velocity[0], obj.velocity[1] - 9.8 * dt, obj.velocity[2]),
        }

    def predict_event_sequence(self, start_event: str) -> List[str]:
        """预测事件序列"""
        sequence = [start_event]
        current = start_event

        while current in self.causal_rules:
            next_events = self.causal_rules[current]
            if next_events:
                current = next_events[0]  # 取第一个可能的结果
                sequence.append(current)
            else:
                break

        self.stats['predictions_made'] += 1
        return sequence

    def infer_location(self, entity: str, context: str) -> Optional[str]:
        """推断实体的位置"""
        # 位置模式
        location_patterns = [
            (r'在(.+?)里', 'in'),
            (r'在(.+?)上', 'on'),
            (r'在(.+?)下', 'under'),
            (r'位于(.+)', 'at'),
            (r'坐落在(.+)', 'at'),
        ]

        for pattern, relation in location_patterns:
            match = re.search(pattern, context)
            if match:
                location = match.group(1).strip()
                if entity not in self.locations:
                    self.locations[entity] = {}
                self.locations[entity]['current'] = location
                self.locations[entity]['relation'] = relation
                return location

        return None

    def infer_agent_intention(self, agent_name: str, action: str) -> Optional[str]:
        """推断智能体的意图"""
        if agent_name not in self.agents:
            return None

        agent = self.agents[agent_name]

        # 意图模式
        intention_patterns = [
            (r'为了(.+)', 'goal'),
            (r'想要(.+)', 'desire'),
            (r'打算(.+)', 'plan'),
            (r'希望(.+)', 'hope'),
        ]

        for pattern, intention_type in intention_patterns:
            match = re.search(pattern, action)
            if match:
                intention = match.group(1).strip()
                agent.intentions.append(intention)
                return intention

        return None

    def learn_from_text(self, text: str):
        """从文本中学习世界知识"""
        # 分句
        sentences = re.split(r'[。！？；\n]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # 提取物理知识
            self._extract_physical_knowledge(sentence)

            # 提取社会知识
            self._extract_social_knowledge(sentence)

            # 提取时间知识
            self._extract_temporal_knowledge(sentence)

    def _extract_physical_knowledge(self, sentence: str):
        """提取物理知识"""
        # 物体运动模式
        motion_patterns = [
            (r'(.+)向(.+)移动', 'move_towards'),
            (r'(.+)从(.+)掉落', 'fall_from'),
            (r'(.+)飞向(.+)', 'fly_to'),
            (r'(.+)滚向(.+)', 'roll_to'),
            (r'(.+)上升', 'rise'),
            (r'(.+)下降', 'fall'),
            (r'(.+)旋转', 'rotate'),
            (r'(.+)振动', 'vibrate'),
        ]

        for pattern, motion_type in motion_patterns:
            match = re.search(pattern, sentence)
            if match:
                obj_name = match.group(1).strip()
                if len(obj_name) > 10:
                    continue

                if obj_name not in self.objects:
                    self.add_object(obj_name)

                # 记录运动
                self.add_event(f"{obj_name}_{motion_type}", participants=[obj_name])

        # 物体属性模式
        property_patterns = [
            (r'(.+)的(\w+)为(\d+(?:\.\d+)?)', 'numerical'),
            (r'(.+)的(\w+)是(\w+)', 'descriptive'),
            (r'(.+)重(\d+(?:\.\d+)?)', 'weight'),
            (r'(.+)高(\d+(?:\.\d+)?)', 'height'),
            (r'(.+)长(\d+(?:\.\d+)?)', 'length'),
        ]

        for pattern, prop_type in property_patterns:
            match = re.search(pattern, sentence)
            if match:
                obj_name = match.group(1).strip()
                if len(obj_name) > 10:
                    continue

                if obj_name not in self.objects:
                    self.add_object(obj_name)

                obj = self.objects[obj_name]
                if prop_type == 'numerical':
                    prop_name = match.group(2)
                    prop_value = match.group(3)
                    obj.properties[prop_name] = float(prop_value)
                elif prop_type in ('weight', 'height', 'length'):
                    prop_value = match.group(2)
                    obj.properties[prop_type] = float(prop_value)

    def _extract_social_knowledge(self, sentence: str):
        """提取社会知识"""
        # 人物行为模式
        action_patterns = [
            (r'(.+)告诉(.+)', 'tell'),
            (r'(.+)帮助(.+)', 'help'),
            (r'(.+)欺骗(.+)', 'deceive'),
            (r'(.+)合作(.+)', 'cooperate'),
            (r'(.+)竞争(.+)', 'compete'),
            (r'(.+)教导(.+)', 'teach'),
            (r'(.+)指导(.+)', 'guide'),
            (r'(.+)领导(.+)', 'lead'),
            (r'(.+)跟随(.+)', 'follow'),
            (r'(.+)支持(.+)', 'support'),
            (r'(.+)反对(.+)', 'oppose'),
            (r'(.+)批评(.+)', 'criticize'),
            (r'(.+)赞扬(.+)', 'praise'),
            (r'(.+)发明了?(.+)', 'invent'),
            (r'(.+)发现了?(.+)', 'discover'),
            (r'(.+)创造了?(.+)', 'create'),
            (r'(.+)提出了?(.+)', 'propose'),
            (r'(.+)开发了?(.+)', 'develop'),
            (r'(.+)设计了?(.+)', 'design'),
        ]

        for pattern, action_type in action_patterns:
            match = re.search(pattern, sentence)
            if match:
                agent1 = match.group(1).strip()
                agent2 = match.group(2).strip()

                # 过滤太长的
                if len(agent1) > 15 or len(agent2) > 30:
                    continue

                # 确保智能体存在
                if agent1 not in self.agents:
                    self.add_agent(agent1)
                if agent2 not in self.agents:
                    self.add_agent(agent2)

                # 记录社会关系
                self.add_event(f"{agent1}_{action_type}_{agent2}",
                             participants=[agent1, agent2])

                # 更新智能体知识
                self.agents[agent1].knowledge.add(f"{action_type}:{agent2}")

    def _extract_temporal_knowledge(self, sentence: str):
        """提取时间知识"""
        # 时间模式
        time_patterns = [
            (r'(\d{4})年(\d{1,2})月(\d{1,2})日', 'full_date'),
            (r'(\d{4})年(\d{1,2})月', 'year_month'),
            (r'(\d{4})年', 'year'),
            (r'(\d{1,2})月(\d{1,2})日', 'month_day'),
            (r'之前', 'before'),
            (r'之后', 'after'),
            (r'同时', 'simultaneous'),
            (r'昨天', 'yesterday'),
            (r'今天', 'today'),
            (r'明天', 'tomorrow'),
            (r'去年', 'last_year'),
            (r'今年', 'this_year'),
            (r'明年', 'next_year'),
        ]

        for pattern, time_type in time_patterns:
            match = re.search(pattern, sentence)
            if match:
                # 记录时间信息到相关事件
                if self.events:
                    last_event = self.events[-1]
                    if time_type == 'full_date':
                        last_event.properties['date'] = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                    elif time_type == 'year':
                        last_event.properties['year'] = match.group(1)
                    elif time_type in ('before', 'after', 'simultaneous'):
                        last_event.properties['temporal_relation'] = time_type

    def query(self, question: str) -> Dict:
        """查询世界模型"""
        results = {
            'objects': [],
            'agents': [],
            'events': [],
            'predictions': [],
        }

        # 提取实体
        entities = self._extract_entities(question)

        for entity in entities:
            # 物理对象
            if entity in self.objects:
                obj = self.objects[entity]
                results['objects'].append({
                    'name': entity,
                    'position': obj.position,
                    'velocity': obj.velocity,
                    'properties': obj.properties,
                })

            # 智能体
            if entity in self.agents:
                agent = self.agents[entity]
                results['agents'].append({
                    'name': entity,
                    'beliefs': agent.beliefs,
                    'desires': agent.desires,
                    'intentions': agent.intentions,
                    'emotions': agent.emotions,
                })

            # 事件
            for event in self.events:
                if entity in event.name or entity in event.participants:
                    results['events'].append({
                        'name': event.name,
                        'timestamp': event.timestamp,
                        'participants': event.participants,
                    })

        return results

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        # 中文实体
        zh_entities = re.findall(r'[一-鿿]{2,6}', text)
        # 英文实体
        en_entities = re.findall(r'[A-Za-z]+', text)

        # 过滤
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样可以能够应该')
        entities = []
        for e in zh_entities + en_entities:
            if e not in stopwords and len(e) >= 2:
                entities.append(e)

        return entities

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'agents_tracked': len(self.agents),
            'locations_known': len(self.locations),
        }


def test_world_model():
    """测试世界模型"""
    print("=" * 70)
    print("世界模型层测试")
    print("=" * 70)

    model = WorldModel()

    # 测试文本
    test_texts = [
        "苹果从树上掉落。",
        "张三告诉李四一个秘密。",
        "太阳从东方升起。",
        "水在100度沸腾。",
    ]

    with open('world_model_test.txt', 'w', encoding='utf-8') as f:
        f.write('世界模型层测试\n')
        f.write('=' * 70 + '\n')

        for text in test_texts:
            f.write(f'\n输入: {text}\n')
            model.learn_from_text(text)

        # 显示物理对象
        f.write('\n' + '=' * 70 + '\n')
        f.write('物理对象\n')
        f.write('=' * 70 + '\n')

        for name, obj in model.objects.items():
            f.write(f'  {name}: position={obj.position}, velocity={obj.velocity}\n')

        # 显示智能体
        f.write('\n' + '=' * 70 + '\n')
        f.write('智能体\n')
        f.write('=' * 70 + '\n')

        for name, agent in model.agents.items():
            f.write(f'  {name}: intentions={agent.intentions}\n')

        # 显示事件
        f.write('\n' + '=' * 70 + '\n')
        f.write('事件\n')
        f.write('=' * 70 + '\n')

        for event in model.events[:10]:
            f.write(f'  {event.name}: participants={event.participants}\n')

        # 测试预测
        f.write('\n' + '=' * 70 + '\n')
        f.write('预测测试\n')
        f.write('=' * 70 + '\n')

        # 物理预测
        if '苹果' in model.objects:
            prediction = model.predict_next_state('苹果', 1.0)
            position = prediction.get('position', 'unknown')
            f.write(f'\n苹果在1秒后的位置: {position}\n')

        # 事件序列预测
        if model.causal_rules:
            for cause, effects in list(model.causal_rules.items())[:3]:
                sequence = model.predict_event_sequence(cause)
                f.write(f'事件序列: {" → ".join(sequence)}\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = model.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to world_model_test.txt')


if __name__ == '__main__':
    test_world_model()
