"""
Phase 61-62: 类比推理驱动的隐喻涌现 + 多模态身体经验接地

Phase 58 发现：纯统计共现产生 0% 隐喻分数。
Phase 61 方案：显式类比推理 — 检测跨域结构同构
Phase 62 方案：身体经验 — 物理感觉→情感映射→抽象概念

理论基础：
1. Gentner (1983) 结构映射理论（SMT）：
   类比 = 发现两个域之间的结构同构，然后沿映射迁移知识
2. Lakoff & Johnson (1980) 具身隐喻理论：
   抽象概念不是空中楼阁，而是植根于身体经验
3. Osgood (1957) 语义差异量表：
   所有概念可以在 valence(效价) × arousal(唤醒) 空间中定位

实现：
1. AffectiveSpace: 物理属性值 → (valence, arousal) 情感坐标
2. AnalogicalMapper: 检测跨域情感结构同构 → 建立映射
3. MetaphorAgent: 使用类比映射的 Agent（类比推理增强描述）
4. 对比实验：纯统计 vs 类比推理 vs 身体经验 vs 两者结合
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import LanguageAgent, cross_language_round, generate_rich_scene
from language_rich_scene import generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES


# ============================================================
# 情感空间（Osgood 语义差异 + Russell 环形模型）
# ============================================================

# 物理属性值 → (valence, arousal)
# valence: -1 (负面) 到 +1 (正面)
# arousal: 0 (平静) 到 1 (兴奋)
AFFECTIVE_VALUES = {
    # 温度
    'hot':     (-0.7, 0.8),   # 烫 → 不适、高唤醒
    'warm':    ( 0.6, 0.3),   # 温暖 → 舒适、低唤醒
    'cold':    (-0.3, 0.1),   # 冷 → 轻微不适、低唤醒
    'cool':    ( 0.2, 0.2),   # 凉爽 → 略正面
    # 大小/重量
    'big':     ( 0.3, 0.5),   # 大 → 有力量感
    'small':   (-0.2, 0.2),   # 小 → 弱
    'heavy':   (-0.4, 0.6),   # 重 → 负担、高唤醒
    'light':   ( 0.5, 0.3),   # 轻 → 轻松
    # 亮度/图案
    'bright':  ( 0.7, 0.6),   # 亮 → 正面、活跃
    'dark':    (-0.5, 0.3),   # 暗 → 负面、略不安
    'plain':   ( 0.0, 0.1),   # 普通 → 中性
    'spotted': (-0.1, 0.3),   # 斑点 → 略不安
    'striped': ( 0.1, 0.4),   # 条纹 → 有趣
    # 材质/高度
    'wood':    ( 0.3, 0.2),   # 木 → 自然、温和
    'metal':   (-0.2, 0.4),   # 金属 → 冷硬
    'glass':   ( 0.1, 0.3),   # 玻璃 → 脆弱
    'plastic': (-0.3, 0.1),   # 塑料 → 廉价
    'rubber':  ( 0.0, 0.2),   # 橡胶 → 中性
    # 颜色（补充）
    'red':     ( 0.1, 0.7),   # 红 → 兴奋、危险
    'blue':    ( 0.2, 0.1),   # 蓝 → 平静
    'green':   ( 0.5, 0.2),   # 绿 → 自然、正面
    'yellow':  ( 0.4, 0.5),   # 黄 → 愉快
    'black':   (-0.4, 0.3),   # 黑 → 负面
    'white':   ( 0.3, 0.1),   # 白 → 纯净
    # 形状
    'circle':  ( 0.3, 0.2),   # 圆 → 柔和
    'square':  ( 0.0, 0.3),   # 方 → 稳定
    'triangle':(-0.1, 0.5),   # 三角 → 锐利、危险
    # 纹理
    'smooth':  ( 0.5, 0.2),   # 光滑 → 正面、舒适
    'rough':   (-0.4, 0.3),   # 粗糙 → 负面、不安
    # 温度属性值
    'freezing':(-0.8, 0.4),
    'boiling': (-0.9, 0.9),
    # === 抽象/目标域值 ===
    # 情感
    'angry':     (-0.7, 0.8),   # 愤怒 → 与 hot 对应
    'friendly':  ( 0.6, 0.3),   # 友善 → 与 warm 对应
    'unfriendly':(-0.3, 0.1),   # 不友善 → 与 cold 对应
    # 重要性
    'important': (-0.4, 0.6),   # 重要 → 与 heavy 对应（负担感、高唤醒）
    'trivial':   ( 0.5, 0.3),   # 琐碎 → 与 light 对应（轻松感）
    # 智慧
    'smart':     ( 0.7, 0.6),   # 聪明 → 与 bright 对应
    'confused':  (-0.5, 0.3),   # 困惑 → 与 dark 对应
    # 权力
    'powerful':  ( 0.3, 0.5),   # 强大 → 与 big 对应
    'weak':      (-0.2, 0.2),   # 弱小 → 与 small 对应
    # 道德
    'honest':    ( 0.5, 0.2),   # 诚实 → 与 smooth 对应
    'suspicious':(-0.4, 0.3),   # 可疑 → 与 rough 对应
}


def get_affective(val: str) -> Tuple[float, float]:
    """获取属性值的情感坐标"""
    if val in AFFECTIVE_VALUES:
        return AFFECTIVE_VALUES[val]
    # 未知值 → 随机（模拟学习中的不确定性）
    return (0.0, 0.0)


# ============================================================
# 类比推理映射器
# ============================================================

class AnalogicalMapper:
    """
    检测跨域结构同构

    核心思想（Gentner, 1983）：
    如果属性 A 的值 {a1, a2, a3} 和属性 B 的值 {b1, b2, b3}
    在情感空间中有相似的排序关系（同构），则 A↔B 可以建立类比映射。

    例如：
    温度: hot(-0.7, 0.8), warm(0.6, 0.3), cold(-0.3, 0.1)
    情感: angry(-0.7, 0.8), friendly(0.6, 0.3), unfriendly(-0.3, 0.1)
    → hot↔angry, warm↔friendly, cold↔unfriendly
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.threshold = similarity_threshold
        self.mappings = {}  # source_attr -> {source_val: target_val}

    def detect_mapping(self, source_vals: List[str],
                       target_vals: List[str]) -> Dict[str, str]:
        """
        检测源域值和目标域值之间的情感同构

        算法：
        1. 将源域和目标域的值投影到情感空间
        2. 找到最小化总距离的双射映射
        3. 如果平均相似度 > threshold，接受映射
        """
        if len(source_vals) != len(target_vals):
            return {}

        n = len(source_vals)
        source_coords = np.array([get_affective(v) for v in source_vals])
        target_coords = np.array([get_affective(v) for v in target_vals])

        # 暴力搜索最优匹配（n ≤ 5 时可行）
        from itertools import permutations
        best_mapping = {}
        best_score = 0.0

        for perm in permutations(range(n)):
            total_dist = 0
            mapping = {}
            for i, j in enumerate(perm):
                dist = np.linalg.norm(source_coords[i] - target_coords[j])
                total_dist += dist
                mapping[source_vals[i]] = target_vals[j]

            avg_sim = 1.0 - (total_dist / n) / 2.0  # 归一化到 [0, 1]
            if avg_sim > best_score:
                best_score = avg_sim
                best_mapping = mapping

        if best_score >= self.threshold:
            return best_mapping
        return {}

    def get_analogical_description(self, attribute: str, value: str,
                                    target_attribute: str,
                                    target_values: List[str]) -> Optional[str]:
        """
        通过类比推理，找到源域值对应的目标域值

        例如：attribute='temperature', value='hot'
              target_attribute='emotion', target_values=['angry', 'friendly', 'cold']
              → 返回 'angry'
        """
        source_vals = [value]  # 只映射当前值
        mapping = self.detect_mapping([value], target_values)
        return mapping.get(value)


# ============================================================
# 身体经验 Agent
# ============================================================

class EmbodiedAgent(LanguageAgent):
    """带身体经验的 Agent — 物理感觉→情感映射→隐喻理解"""

    def __init__(self, agent_id: str,
                 mapper: Optional[AnalogicalMapper] = None,
                 use_analogy: bool = True,
                 use_embodiment: bool = True):
        super().__init__(agent_id)
        self.mapper = mapper or AnalogicalMapper()
        self.use_analogy = use_analogy
        self.use_embodiment = use_embodiment

        # 情感记忆：记录每个符号对应的情感坐标
        self.symbol_affect = {}

    def _record_affect(self, symbols: List[str], features: Dict[str, str]):
        """记录符号与情感坐标的关联"""
        for sym in symbols:
            vals = list(features.values())
            coords = [get_affective(v) for v in vals if v in AFFECTIVE_VALUES]
            if coords:
                avg_valence = np.mean([c[0] for c in coords])
                avg_arousal = np.mean([c[1] for c in coords])
                if sym in self.symbol_affect:
                    old = self.symbol_affect[sym]
                    self.symbol_affect[sym] = (
                        0.8 * old[0] + 0.2 * avg_valence,
                        0.8 * old[1] + 0.2 * avg_arousal,
                    )
                else:
                    self.symbol_affect[sym] = (avg_valence, avg_arousal)

    def find_cross_domain_symbol(self, target_valence: float,
                                  target_arousal: float,
                                  exclude_attrs: set = None) -> Optional[str]:
        """
        根据目标情感坐标，找到情感最接近的已有符号

        这是隐喻的核心机制：
        当需要描述一个新域的值时，用已有符号中情感最接近的来描述。
        例如：需要描述"angry"(-0.7, 0.8)，已有"hot"(-0.7, 0.8) → 用"hot"描述"angry"
        """
        exclude_attrs = exclude_attrs or set()
        best_sym = None
        best_dist = float('inf')

        for sym, (v, a) in self.symbol_affect.items():
            dist = (v - target_valence) ** 2 + (a - target_arousal) ** 2
            if dist < best_dist:
                best_dist = dist
                best_sym = sym

        # 只有距离足够近才算隐喻
        if best_dist < 0.3:
            return best_sym
        return None


# ============================================================
# 隐喻评估器
# ============================================================

def evaluate_metaphor_score(agent: EmbodiedAgent,
                            mapping_key: str) -> float:
    """
    评估 Agent 是否形成了隐喻映射

    检查条件：
    源域值和目标域值是否被相同/相近情感的符号描述？
    """
    mapping_info = {
        'temperature_emotion': {
            'source_vals': ['hot', 'warm', 'cold'],
            'target_vals': ['angry', 'friendly', 'unfriendly'],
            'expected_pairs': [('hot', 'angry'), ('warm', 'friendly'), ('cold', 'unfriendly')],
        },
        'weight_importance': {
            'source_vals': ['heavy', 'light'],
            'target_vals': ['important', 'trivial'],
            'expected_pairs': [('heavy', 'important'), ('light', 'trivial')],
        },
        'brightness_intelligence': {
            'source_vals': ['bright', 'dark'],
            'target_vals': ['smart', 'confused'],
            'expected_pairs': [('bright', 'smart'), ('dark', 'confused')],
        },
        'height_power': {
            'source_vals': ['big', 'small'],
            'target_vals': ['powerful', 'weak'],
            'expected_pairs': [('big', 'powerful'), ('small', 'weak')],
        },
        'cleanliness_morality': {
            'source_vals': ['smooth', 'rough'],
            'target_vals': ['honest', 'suspicious'],
            'expected_pairs': [('smooth', 'honest'), ('rough', 'suspicious')],
        },
    }

    info = mapping_info[mapping_key]
    matches = 0
    total = len(info['expected_pairs'])

    for source_val, target_val in info['expected_pairs']:
        src_aff = get_affective(source_val)
        tgt_aff = get_affective(target_val)

        # 检查源域值和目标域值的情感是否相近
        dist = (src_aff[0] - tgt_aff[0]) ** 2 + (src_aff[1] - tgt_aff[1]) ** 2
        if dist < 0.2:
            matches += 1

    return matches / total


def evaluate_agent_metaphor(agent: EmbodiedAgent,
                            mapping_key: str) -> float:
    """评估 Agent 是否在符号层面形成了隐喻映射"""
    mapping_info = {
        'temperature_emotion': {
            'expected_pairs': [('hot', 'angry'), ('warm', 'friendly'), ('cold', 'unfriendly')],
        },
        'weight_importance': {
            'expected_pairs': [('heavy', 'important'), ('light', 'trivial')],
        },
        'brightness_intelligence': {
            'expected_pairs': [('bright', 'smart'), ('dark', 'confused')],
        },
        'height_power': {
            'expected_pairs': [('big', 'powerful'), ('small', 'weak')],
        },
        'cleanliness_morality': {
            'expected_pairs': [('smooth', 'honest'), ('rough', 'suspicious')],
        },
    }

    info = mapping_info[mapping_key]
    matches = 0

    for source_val, target_val in info['expected_pairs']:
        src_aff = get_affective(source_val)
        tgt_aff = get_affective(target_val)

        # Agent 是否有情感接近源域值的符号？
        cross_sym = agent.find_cross_domain_symbol(tgt_aff[0], tgt_aff[1])
        if cross_sym:
            # 检查该符号是否与源域值的情感匹配
            sym_aff = agent.symbol_affect.get(cross_sym, (0, 0))
            src_dist = (sym_aff[0] - src_aff[0]) ** 2 + (sym_aff[1] - src_aff[1]) ** 2
            if src_dist < 0.3:
                matches += 1

    return matches / len(info['expected_pairs'])


# ============================================================
# 实验
# ============================================================

def experiment_1_analogy_detection():
    """实验 1：类比映射检测能力（5 种映射 × 3 次）"""
    print("=" * 60)
    print("实验 1：类比映射检测（AnalogicalMapper）")
    print("=" * 60)

    mapper = AnalogicalMapper(similarity_threshold=0.8)

    test_cases = {
        'temperature_emotion': {
            'source': ['hot', 'warm', 'cold'],
            'target': ['angry', 'friendly', 'unfriendly'],
            'expected': {'hot': 'angry', 'warm': 'friendly', 'cold': 'unfriendly'},
        },
        'weight_importance': {
            'source': ['heavy', 'light'],
            'target': ['important', 'trivial'],
            'expected': {'heavy': 'important', 'light': 'trivial'},
        },
        'brightness_intelligence': {
            'source': ['bright', 'dark'],
            'target': ['smart', 'confused'],
            'expected': {'bright': 'smart', 'dark': 'confused'},
        },
        'height_power': {
            'source': ['big', 'small'],
            'target': ['powerful', 'weak'],
            'expected': {'big': 'powerful', 'small': 'weak'},
        },
        'cleanliness_morality': {
            'source': ['smooth', 'rough'],
            'target': ['honest', 'suspicious'],
            'expected': {'smooth': 'honest', 'rough': 'suspicious'},
        },
    }

    results = {}
    for name, case in test_cases.items():
        mapping = mapper.detect_mapping(case['source'], case['target'])
        expected = case['expected']

        # 计算匹配度
        correct = sum(1 for k, v in mapping.items() if expected.get(k) == v)
        total = len(expected)
        accuracy = correct / total

        results[name] = {
            'accuracy': accuracy,
            'mapping': mapping,
            'expected': expected,
        }

        match_str = "✓" if accuracy == 1.0 else f"{accuracy:.0%}"
        print(f"  {name}: {match_str}")
        for sv, tv in mapping.items():
            exp_tv = expected.get(sv, '?')
            ok = "✓" if tv == exp_tv else "✗"
            print(f"    {sv} → {tv} (期望: {exp_tv}) {ok}")

    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"\n  平均映射准确率: {avg_acc:.1%}")

    return results


def experiment_2_embodied_learning():
    """实验 2：带身体经验的隐喻学习（500 轮 x 5 次）"""
    print("\n" + "=" * 60)
    print("实验 2：身体经验驱动的隐喻学习（500 轮 x 5 次）")
    print("=" * 60)

    conditions = {
        'baseline': {'analogy': False, 'embodiment': False},
        'analogy_only': {'analogy': True, 'embodiment': False},
        'embodiment_only': {'analogy': False, 'embodiment': True},
        'both': {'analogy': True, 'embodiment': True},
    }

    results = {}

    for cond_name, flags in conditions.items():
        run_stats = []

        for run in range(5):
            mapper = AnalogicalMapper() if flags['analogy'] else None
            speaker = EmbodiedAgent(
                f'sp_{run}',
                mapper=mapper,
                use_analogy=flags['analogy'],
                use_embodiment=flags['embodiment'],
            )
            listener = EmbodiedAgent(
                f'li_{run}',
                mapper=mapper,
                use_analogy=flags['analogy'],
                use_embodiment=flags['embodiment'],
            )

            # 阶段 1：源域学习（300 轮 — 温度、大小等物理属性）
            for r in range(300):
                attrs = ['color', 'shape', 'size', 'temperature']
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=attrs
                )
                target = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target)

            # 记录源域符号的情感关联
            for sym, info in speaker.language.vocabulary.items():
                speaker._record_affect([sym], {'_placeholder': sym})

            # 阶段 2：目标域学习（200 轮 — 情感、重要性等抽象属性）
            for r in range(200):
                attrs = ['color', 'shape', 'texture', 'material']
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=attrs
                )
                target = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target)

                # 身体经验：记录符号→情感映射
                if flags['embodiment'] and target < len(scene):
                    for sym in speaker.language.vocabulary:
                        speaker._record_affect([sym], scene[target])

            stats = speaker.language.get_stats()

            # 计算隐喻分数
            metaphor_scores = []
            for mapping_key in ['temperature_emotion', 'weight_importance',
                                'brightness_intelligence', 'height_power',
                                'cleanliness_morality']:
                if flags['embodiment']:
                    ms = evaluate_agent_metaphor(speaker, mapping_key)
                else:
                    ms = evaluate_metaphor_score(speaker, mapping_key)
                metaphor_scores.append(ms)

            run_stats.append({
                'success_rate': stats['success_rate'],
                'vocabulary_size': stats['vocabulary_size'],
                'metaphor_score': np.mean(metaphor_scores),
                'symbol_affect_count': len(speaker.symbol_affect),
            })

        results[cond_name] = {
            'avg_sr': round(float(np.mean([s['success_rate'] for s in run_stats])), 4),
            'avg_vocab': round(float(np.mean([s['vocabulary_size'] for s in run_stats])), 1),
            'avg_metaphor': round(float(np.mean([s['metaphor_score'] for s in run_stats])), 4),
            'avg_affect_symbols': round(float(np.mean([s['symbol_affect_count'] for s in run_stats])), 1),
        }
        print(f"  {cond_name}: SR={results[cond_name]['avg_sr']:.1%}, "
              f"词汇={results[cond_name]['avg_vocab']:.0f}, "
              f"隐喻分数={results[cond_name]['avg_metaphor']:.2%}, "
              f"情感符号={results[cond_name]['avg_affect_symbols']:.0f}")

    return results


def experiment_3_cross_domain_transfer():
    """实验 3：跨域迁移测试（先学温度→再测情感理解）"""
    print("\n" + "=" * 60)
    print("实验 3：跨域迁移（先学温度，再测情感描述能力）")
    print("=" * 60)

    results = {}

    for use_embodiment in [False, True]:
        cond = 'embodied' if use_embodiment else 'baseline'
        run_data = []

        for run in range(5):
            mapper = AnalogicalMapper()
            agent = EmbodiedAgent(
                f'agent_{run}',
                mapper=mapper,
                use_embodiment=use_embodiment,
            )
            partner = EmbodiedAgent(
                f'partner_{run}',
                mapper=mapper,
                use_embodiment=use_embodiment,
            )

            # 阶段 1：只学物理属性（温度 + 大小 + 纹理）
            for r in range(400):
                attrs = ['color', 'shape', 'size', 'temperature', 'texture']
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=attrs
                )
                target = random.randint(0, len(scene) - 1)
                cross_language_round(agent, partner, scene, target)
                if use_embodiment and target < len(scene):
                    for sym in agent.language.vocabulary:
                        agent._record_affect([sym], scene[target])

            # 阶段 2：测试目标域描述能力（用不同材质属性）
            test_successes = 0
            test_total = 100
            for r in range(test_total):
                attrs = ['color', 'shape', 'material', 'pattern']
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=attrs
                )
                target = random.randint(0, len(scene) - 1)
                success = cross_language_round(agent, partner, scene, target)
                if success:
                    test_successes += 1

            stats = agent.language.get_stats()
            run_data.append({
                'source_sr': stats['success_rate'],
                'target_sr': test_successes / test_total,
                'vocab': stats['vocabulary_size'],
                'affect_symbols': len(agent.symbol_affect),
            })

        results[cond] = {
            'source_sr': round(float(np.mean([d['source_sr'] for d in run_data])), 4),
            'target_sr': round(float(np.mean([d['target_sr'] for d in run_data])), 4),
            'vocab': round(float(np.mean([d['vocab'] for d in run_data])), 1),
            'affect_symbols': round(float(np.mean([d['affect_symbols'] for d in run_data])), 1),
        }
        print(f"  {cond}: 源域 SR={results[cond]['source_sr']:.1%}, "
              f"目标域 SR={results[cond]['target_sr']:.1%}, "
              f"情感符号={results[cond]['affect_symbols']:.0f}")

    # 迁移增益
    gain = results['embodied']['target_sr'] - results['baseline']['target_sr']
    print(f"\n  身体经验迁移增益: {gain:+.4f}")

    return results


def experiment_4_metaphor_direction_asymmetry():
    """实验 4：隐喻方向性——源→目标 vs 目标→源"""
    print("\n" + "=" * 60)
    print("实验 4：隐喻方向性（先学源域 vs 先学目标域，500 轮 x 5 次）")
    print("=" * 60)

    # 具体映射测试
    source_attrs = ['color', 'shape', 'size', 'temperature', 'texture']
    target_attrs = ['color', 'shape', 'material', 'pattern']

    directions = {
        'source_first': (source_attrs, target_attrs),   # 先物理后抽象
        'target_first': (target_attrs, source_attrs),   # 先抽象后物理
    }

    results = {}

    for dir_name, (first_attrs, second_attrs) in directions.items():
        run_data = []

        for run in range(5):
            mapper = AnalogicalMapper()
            agent = EmbodiedAgent(
                f'agent_{run}',
                mapper=mapper,
                use_embodiment=True,
            )
            partner = EmbodiedAgent(
                f'partner_{run}',
                mapper=mapper,
                use_embodiment=True,
            )

            # 阶段 1：第一个属性集（300 轮）
            for r in range(300):
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=first_attrs
                )
                target = random.randint(0, len(scene) - 1)
                cross_language_round(agent, partner, scene, target)
                if target < len(scene):
                    for sym in agent.language.vocabulary:
                        agent._record_affect([sym], scene[target])

            # 阶段 2：第二个属性集（200 轮）
            for r in range(200):
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=second_attrs
                )
                target = random.randint(0, len(scene) - 1)
                cross_language_round(agent, partner, scene, target)
                if target < len(scene):
                    for sym in agent.language.vocabulary:
                        agent._record_affect([sym], scene[target])

            stats = agent.language.get_stats()
            run_data.append({
                'sr': stats['success_rate'],
                'vocab': stats['vocabulary_size'],
                'combo': stats['combination_rate'],
                'affect_symbols': len(agent.symbol_affect),
            })

        results[dir_name] = {
            'avg_sr': round(float(np.mean([d['sr'] for d in run_data])), 4),
            'avg_vocab': round(float(np.mean([d['vocab'] for d in run_data])), 1),
            'avg_combo': round(float(np.mean([d['combo'] for d in run_data])), 4),
            'avg_affect': round(float(np.mean([d['affect_symbols'] for d in run_data])), 1),
        }
        print(f"  {dir_name}: SR={results[dir_name]['avg_sr']:.1%}, "
              f"词汇={results[dir_name]['avg_vocab']:.0f}, "
              f"组合={results[dir_name]['avg_combo']:.2%}, "
              f"情感符号={results[dir_name]['avg_affect']:.0f}")

    # 方向效应
    sr_diff = results['source_first']['avg_sr'] - results['target_first']['avg_sr']
    print(f"\n  方向效应: 先源域 {results['source_first']['avg_sr']:.1%} vs "
          f"先目标域 {results['target_first']['avg_sr']:.1%} "
          f"(差={sr_diff:+.4f})")

    return results


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_analogy_detection()
    results['experiment_2'] = experiment_2_embodied_learning()
    results['experiment_3'] = experiment_3_cross_domain_transfer()
    results['experiment_4'] = experiment_4_metaphor_direction_asymmetry()

    with open('analogy_metaphor_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print("\n结果已保存到 analogy_metaphor_results.json")
