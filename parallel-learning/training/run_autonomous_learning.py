"""自主学习系统训练脚本

运行方式：
    python training/run_autonomous_learning.py

功能：
1. 创建一个简单的模拟环境
2. 系统自主探索环境
3. 发现因果规律
4. 形成抽象概念
5. 将语言接地到世界模型
6. 自我迭代改进

演示：
- 系统如何自主发现"火是热的"、"球会弹"等规律
- 系统如何从具体经验中归纳出抽象概念
- 系统如何将词汇与世界模型关联
"""

import sys
import os
import time
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.autonomous_system import AutonomousLearningSystem


class SimpleEnvironment:
    """简单模拟环境 — 用于演示自主学习

    模拟一个物理世界，有以下规律：
    1. 火是热的（fire → hot）
    2. 冰是冷的（ice → cold）
    3. 球会弹（ball → bounce）
    4. 重物会下落（heavy → fall）
    5. 水往低处流（water → flow_down）

    系统需要自主发现这些规律。
    """

    def __init__(self):
        # 物理规律（系统不知道，需要自己发现）
        self.laws = {
            'fire': {'temperature': 0.9, 'state': 'hot'},
            'ice': {'temperature': 0.1, 'state': 'cold'},
            'ball': {'bounce': 0.8, 'state': 'elastic'},
            'rock': {'weight': 0.9, 'state': 'heavy'},
            'water': {'flow': 0.7, 'state': 'liquid'},
        }

        # 当前状态
        self.current_object = None
        self.current_state = {}
        self.step_count = 0

        # 词汇表（系统会学习这些词的含义）
        self.vocabulary = ['fire', 'ice', 'ball', 'rock', 'water',
                          'hot', 'cold', 'bounce', 'fall', 'flow']

    def get_state(self) -> dict:
        """获取当前状态"""
        if self.current_object:
            return dict(self.current_state)
        return {}

    def get_available_actions(self) -> list:
        """获取可用动作"""
        return [
            {'type': 'touch', 'object': obj}
            for obj in self.laws.keys()
        ]

    def step(self, action: dict) -> dict:
        """执行动作，返回结果"""
        self.step_count += 1
        obj = action.get('object', random.choice(list(self.laws.keys())))
        self.current_object = obj

        # 基于物理规律计算结果
        law = self.laws.get(obj, {})
        state = {}
        next_state = {}

        # 生成当前状态
        for key, value in law.items():
            if isinstance(value, (int, float)):
                state[key] = value + random.gauss(0, 0.05)
            else:
                state[key] = value

        # 生成下一状态（加入因果关系）
        next_state = dict(state)
        if 'temperature' in state:
            # 触摸热的东西 → 感觉烫
            if state['temperature'] > 0.7:
                next_state['pain'] = 0.8
                next_state['reaction'] = 'pull_away'
            elif state['temperature'] < 0.3:
                next_state['pain'] = 0.0
                next_state['reaction'] = 'keep_touching'

        if 'bounce' in state:
            # 弹性物体 → 弹跳
            next_state['height'] = state['bounce'] * 0.6

        if 'weight' in state:
            # 重物 → 下落
            next_state['fall_speed'] = state['weight'] * 0.8

        if 'flow' in state:
            # 液体 → 流动
            next_state['direction'] = 'down'

        self.current_state = next_state

        return {
            'state': state,
            'next_state': next_state,
            'object': obj,
        }

    def get_context(self) -> dict:
        """获取上下文（包含当前物体名称）"""
        return {'word': self.current_object}


class PhysicsEnvironment:
    """物理环境 — 更复杂的因果关系"""

    def __init__(self):
        self.objects = {
            'ball': {'mass': 0.5, 'elasticity': 0.8, 'color': 'red'},
            'rock': {'mass': 2.0, 'elasticity': 0.1, 'color': 'gray'},
            'feather': {'mass': 0.01, 'elasticity': 0.3, 'color': 'white'},
            'spring': {'mass': 0.3, 'elasticity': 0.95, 'color': 'silver'},
        }

        self.forces = ['push', 'pull', 'drop', 'throw']
        self.current_state = {}
        self.step_count = 0

    def get_state(self) -> dict:
        return dict(self.current_state)

    def get_available_actions(self) -> list:
        actions = []
        for obj in self.objects:
            for force in self.forces:
                actions.append({'object': obj, 'force': force})
        return actions

    def step(self, action: dict) -> dict:
        self.step_count += 1
        obj = action.get('object', 'ball')
        force = action.get('force', 'push')
        obj_props = self.objects.get(obj, {})

        state = {'object': obj, **obj_props}
        next_state = dict(state)

        # 物理规律
        mass = obj_props.get('mass', 1.0)
        elasticity = obj_props.get('elasticity', 0.5)

        if force == 'drop':
            next_state['velocity'] = mass * 9.8
            next_state['fall_time'] = math.sqrt(2 * mass)
            if elasticity > 0.5:
                next_state['bounce_height'] = elasticity * 0.6

        elif force == 'throw':
            next_state['velocity'] = 5.0 / mass
            next_state['distance'] = 5.0 / mass * 0.8
            if elasticity > 0.5:
                next_state['bounce_count'] = int(elasticity * 5)

        elif force == 'push':
            next_state['velocity'] = 2.0 / mass
            next_state['acceleration'] = 2.0 / mass

        elif force == 'pull':
            next_state['velocity'] = 1.0 / mass
            next_state['tension'] = mass * 1.0

        self.current_state = next_state
        return {'state': state, 'next_state': next_state}

    def get_context(self) -> dict:
        obj = self.current_state.get('object', '')
        force = self.current_state.get('force', '')
        return {'word': obj, 'action': force}


def main():
    print("=" * 70)
    print("自主学习系统 — 像人类一样学习")
    print("=" * 70)

    # ── 初始化系统 ──────────────────────────────────────────────
    system = AutonomousLearningSystem()

    # ── 阶段1：简单环境探索 ──────────────────────────────────────
    print("\n[1] 阶段1：简单环境探索")
    print("  系统将自主探索环境，发现因果规律...")
    env = SimpleEnvironment()

    report1 = system.run_learning_cycles(env, n_cycles=200, log_interval=50)

    print(f"\n  发现的因果规则: {report1['world_model_summary']['total_rules']}")
    print(f"  可靠规则: {report1['world_model_summary']['reliable_rules']}")
    print(f"  形成的概念: {report1['concept_stats']['total_concepts']}")
    print(f"  接地词汇: {report1['language_stats']['grounded_words']}")

    # 显示发现的规律
    top_rules = report1['world_model_summary'].get('top_rules', [])
    if top_rules:
        print("\n  发现的规律:")
        for i, rule in enumerate(top_rules[:5], 1):
            print(f"    {i}. {rule['trigger']} → {rule['effect']} "
                  f"(置信度: {rule['confidence']:.3f})")

    # ── 阶段2：复杂物理环境 ──────────────────────────────────────
    print("\n[2] 阶段2：复杂物理环境")
    print("  系统将学习更复杂的物理规律...")
    env2 = PhysicsEnvironment()

    report2 = system.run_learning_cycles(env2, n_cycles=300, log_interval=100)

    print(f"\n  累计因果规则: {report2['world_model_summary']['total_rules']}")
    print(f"  累计概念: {report2['concept_stats']['total_concepts']}")
    print(f"  累计接地词汇: {report2['language_stats']['grounded_words']}")

    # ── 阶段3：语言理解测试 ──────────────────────────────────────
    print("\n[3] 阶段3：语言理解测试")
    print("  测试系统是否能通过世界模型理解语言...")

    test_sentences = [
        "fire hot",
        "ball bounce",
        "rock fall",
        "water flow",
    ]

    for sentence in test_sentences:
        result = system.understand_language(sentence)
        print(f"  '{sentence}': "
              f"understood={result['understood']} "
              f"confidence={result['confidence']:.3f} "
              f"words_grounded={result['words_grounded']}/{result['total_words']}")

    # ── 最终报告 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("最终报告")
    print("=" * 70)

    final = system.get_system_state()

    print(f"\n  总步数: {final['total_steps']}")
    print(f"  总交互: {final['total_interactions']}")

    print(f"\n  世界模型:")
    wm = final['world_model']
    print(f"    因果规则: {wm['total_rules']}")
    print(f"    可靠规则: {wm['reliable_rules']}")
    print(f"    平均惊讶度: {wm['avg_surprise']}")
    print(f"    总预测: {wm['total_predictions']}")

    print(f"\n  探索统计:")
    es = final['explorer']
    print(f"    总探索: {es.get('total_explorations', 0)}")
    print(f"    平均惊讶度: {es.get('avg_surprise', 0):.3f}")
    print(f"    唯一区域: {es.get('unique_areas', 0)}")
    print(f"    发现知识: {es.get('knowledge_discovered', 0)}")

    print(f"\n  概念形成:")
    cs = final['concepts']
    print(f"    总概念: {cs['total_concepts']}")
    print(f"    概念形成次数: {cs['total_formations']}")
    print(f"    概念合并次数: {cs['total_merges']}")
    print(f"    根概念: {cs['root_concepts']}")

    print(f"\n  语言接地:")
    ls = final['language']
    print(f"    接地词汇: {ls['grounded_words']}")
    print(f"    总接地次数: {ls['total_groundings']}")
    print(f"    平均置信度: {ls['avg_confidence']:.3f}")
    print(f"    平均规则/词: {ls['avg_rules_per_word']:.1f}")

    print(f"\n  自我迭代:")
    si = final['self_iteration']
    print(f"    评估次数: {si.get('evaluations', 0)}")
    print(f"    改进次数: {si.get('improvements', 0)}")
    print(f"    准确率趋势: {si.get('accuracy_trend', 0):.3f}")

    # 显示发现的规律
    top_rules = wm.get('top_rules', [])
    if top_rules:
        print(f"\n  发现的因果规律:")
        for i, rule in enumerate(top_rules[:10], 1):
            print(f"    {i}. {rule['trigger']} → {rule['effect']} "
                  f"(置信度: {rule['confidence']:.3f}, 证据: {rule['evidence']})")

    print("\n" + "=" * 70)
    print("自主学习完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
