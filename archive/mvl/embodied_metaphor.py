"""
Phase 58: 具身隐喻接地 — 从感官体验涌现抽象概念

Lakoff & Johnson (1980) 的概念隐喻理论：
抽象概念不是空中楼阁，而是植根于身体经验。

经典隐喻映射：
- 温度 → 情感：warm person（温暖的人）= 友善的
- 重量 → 重要性：heavy responsibility（沉重的责任）= 重要的
- 亮度 → 智慧/道德：bright idea（光明的想法）= 聪明的
- 空间 → 权力：high status（高的地位）= 有权力的
- 清洁 → 道德：clean conscience（干净的良心）= 无辜的

实验设计：
1. 5 种隐喻映射涌现测试
2. 跨域映射强度分析
3. 隐喻 vs 字面义的上下文依赖
4. 文化差异对隐喻方向的影响
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from language_emergence import LanguageAgent, cross_language_round
from language_rich_scene import generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES


# ============================================================
# 隐喻域定义
# ============================================================

# 源域 → 目标域映射
METAPHOR_MAPPINGS = {
    'temperature_emotion': {
        'source': 'temperature',
        'target': 'emotion',
        'mapping': {
            'hot': 'angry',
            'warm': 'friendly',
            'cold': 'unfriendly',
        },
        'description': '温度 → 情感（warm person = 友善的）',
    },
    'weight_importance': {
        'source': 'weight',
        'target': 'importance',
        'mapping': {
            'heavy': 'important',
            'light': 'trivial',
        },
        'description': '重量 → 重要性（heavy responsibility = 重要的）',
    },
    'brightness_intelligence': {
        'source': 'brightness',
        'target': 'intelligence',
        'mapping': {
            'bright': 'smart',
            'dark': 'confused',
        },
        'description': '亮度 → 智慧（bright idea = 聪明的）',
    },
    'height_power': {
        'source': 'height',
        'target': 'power',
        'mapping': {
            'high': 'powerful',
            'low': 'weak',
        },
        'description': '高度 → 权力（high status = 有权力的）',
    },
    'cleanliness_morality': {
        'source': 'texture',  # 纹理作为清洁度的代理
        'target': 'morality',
        'mapping': {
            'smooth': 'honest',
            'rough': 'suspicious',
        },
        'description': '质地 → 道德（clean conscience = 无辜的）',
    },
}


def generate_metaphor_scene(mapping_key: str,
                            ambiguous: bool = False) -> Tuple[List[Dict], Dict]:
    """
    生成隐喻场景

    场景中既有源域物体（如 hot/warm/cold 物体），
    也有目标域物体（如 angry/friendly/unfriendly Agent），
    需要通过源域词汇描述目标域特征。
    """
    mapping = METAPHOR_MAPPINGS[mapping_key]
    source = mapping['source']

    # 属性名映射到 ALL_ATTRIBUTE_NAMES
    attr_map = {
        'temperature': 'temperature',
        'weight': 'size',       # 用 size 作 weight 代理
        'brightness': 'pattern',  # 用 pattern 作 brightness 代理
        'height': 'material',   # 用 material 作 height 代理
        'texture': 'texture',
    }
    source_attr = attr_map.get(source, 'color')

    scene = []

    # 源域物体（物理特征）
    for val in mapping['mapping']:
        obj = {
            'color': random.choice(['red', 'blue', 'green']),
            'shape': random.choice(['circle', 'square', 'triangle']),
            'size': random.choice(['big', 'small']),
            source_attr: val,
        }
        if ambiguous:
            # 添加共享属性增加歧义
            obj['shape'] = 'circle'
        scene.append(obj)

    # 目标域物体（抽象特征）
    for source_val, target_val in mapping['mapping'].items():
        obj = {
            'color': random.choice(['red', 'blue', 'green']),
            'shape': random.choice(['circle', 'square', 'triangle']),
            'size': random.choice(['big', 'small']),
            source_attr: target_val,  # 用目标域值
        }
        scene.append(obj)

    # 随机选一个目标
    target_idx = random.randint(0, len(scene) - 1)
    return scene, scene[target_idx]


# ============================================================
# 隐喻涌现追踪器
# ============================================================

class MetaphorTracker:
    """追踪隐喻映射是否涌现"""

    def __init__(self):
        self.source_usage = defaultdict(lambda: defaultdict(int))
        self.target_usage = defaultdict(lambda: defaultdict(int))
        self.cross_domain_cooccurrence = defaultdict(lambda: defaultdict(int))

    def record(self, symbols: List[str], scene: List[Dict]):
        """记录符号使用与场景属性的共现"""
        for sym in symbols:
            for obj in scene:
                for attr, val in obj.items():
                    if attr in ('color', 'shape', 'size', 'material',
                                'temperature', 'texture', 'pattern',
                                'weight', 'brightness', 'origin'):
                        self.source_usage[sym][f"{attr}_{val}"] += 1

    def get_metaphor_score(self, source_val: str, target_val: str) -> float:
        """
        计算隐喻分数：
        源域符号和目标域符号共现频率 / 各自总频率
        """
        source_count = sum(self.source_usage.get(source_val, {}).values())
        target_count = sum(self.source_usage.get(target_val, {}).values())

        if source_count == 0 or target_count == 0:
            return 0.0

        cooc = self.cross_domain_cooccurrence.get(source_val, {}).get(target_val, 0)
        return cooc / max(source_count, target_count)

    def detect_metaphors(self) -> Dict[str, float]:
        """检测涌现的隐喻映射"""
        detected = {}

        for mapping_key, mapping in METAPHOR_MAPPINGS.items():
            scores = []
            for source_val, target_val in mapping['mapping'].items():
                # 检查是否有符号同时关联源域值和目标域值
                source_syms = {s for s, attrs in self.source_usage.items()
                               if any(source_val in str(v) for v in attrs)}
                target_syms = {s for s, attrs in self.source_usage.items()
                               if any(target_val in str(v) for v in attrs)}
                overlap = len(source_syms & target_syms)
                total = max(len(source_syms | target_syms), 1)
                scores.append(overlap / total)

            detected[mapping_key] = round(float(np.mean(scores)), 4) if scores else 0.0

        return detected


# ============================================================
# 实验
# ============================================================

def experiment_1_metaphor_emergence():
    """实验 1：5 种隐喻映射涌现测试（500 轮 x 5 次）"""
    print("=" * 60)
    print("实验 1：隐喻映射涌现（500 轮 x 5 次 x 5 种映射）")
    print("=" * 60)

    results = {}

    for mapping_key in METAPHOR_MAPPINGS:
        mapping = METAPHOR_MAPPINGS[mapping_key]
        run_data = []

        for run in range(5):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')
            tracker = MetaphorTracker()

            successes = 0
            for r in range(500):
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=random.sample(ALL_ATTRIBUTE_NAMES, 4)
                )
                target = random.randint(0, len(scene) - 1)
                success = cross_language_round(speaker, listener, scene, target)
                if success:
                    successes += 1

            stats = speaker.language.get_stats()
            run_data.append({
                'success_rate': stats['success_rate'],
                'vocabulary_size': stats['vocabulary_size'],
                'combination_rate': stats['combination_rate'],
            })

        results[mapping_key] = {
            'avg_success_rate': round(float(np.mean([d['success_rate'] for d in run_data])), 4),
            'avg_vocab': round(float(np.mean([d['vocabulary_size'] for d in run_data])), 1),
            'avg_combo': round(float(np.mean([d['combination_rate'] for d in run_data])), 4),
            'description': mapping['description'],
        }
        print(f"\n  {mapping_key}:")
        print(f"    {mapping['description']}")
        print(f"    成功率={results[mapping_key]['avg_success_rate']:.1%}, "
              f"词汇={results[mapping_key]['avg_vocab']:.0f}, "
              f"组合率={results[mapping_key]['avg_combo']:.2%}")

    return results


def experiment_2_cross_domain_strength():
    """实验 2：源域-目标域映射强度（混合场景 500 轮）"""
    print("\n" + "=" * 60)
    print("实验 2：跨域映射强度分析（500 轮 x 3 次）")
    print("=" * 60)

    results = {}

    for mapping_key in METAPHOR_MAPPINGS:
        mapping = METAPHOR_MAPPINGS[mapping_key]
        run_scores = []

        for run in range(3):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')

            # 统计源域符号是否被用于目标域描述
            source_symbols_used = set()
            target_contexts = defaultdict(list)

            for r in range(500):
                # 生成包含源域和目标域特征的混合场景
                source_attrs = list(mapping['mapping'].keys())
                target_attrs = list(mapping['mapping'].values())

                # 构建场景：部分物体有源域值，部分有目标域值
                scene = []
                for sv in source_attrs:
                    obj = {'color': random.choice(['red', 'blue']),
                           'shape': 'circle',
                           'size': random.choice(['big', 'small'])}
                    # 找到哪个属性对应源域
                    for attr in ALL_ATTRIBUTE_NAMES:
                        obj[attr] = sv
                        break
                    scene.append(obj)

                for tv in target_attrs:
                    obj = {'color': random.choice(['red', 'blue']),
                           'shape': 'circle',
                           'size': random.choice(['big', 'small'])}
                    scene.append(obj)

                target = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target)

            stats = speaker.language.get_stats()

            # 计算隐喻分数：源域值和目标域值是否被同一符号描述
            vocab = speaker.language.vocabulary
            metaphor_matches = 0
            for sv, tv in mapping['mapping'].items():
                sv_syms = {s for s in vocab if sv in s}
                tv_syms = {s for s in vocab if tv in s}
                overlap = sv_syms & tv_syms
                if overlap:
                    metaphor_matches += 1

            score = metaphor_matches / len(mapping['mapping'])
            run_scores.append(score)

        results[mapping_key] = {
            'avg_score': round(float(np.mean(run_scores)), 4),
            'description': mapping['description'],
        }
        print(f"  {mapping_key}: 隐喻分数={np.mean(run_scores):.2%}")
        print(f"    {mapping['description']}")

    return results


def experiment_3_literal_vs_metaphorical():
    """实验 3：字面义 vs 隐喻义的上下文依赖"""
    print("\n" + "=" * 60)
    print("实验 3：字面义 vs 隐喻义上下文依赖（300 轮 x 5 次）")
    print("=" * 60)

    conditions = {
        'literal_only': ['color', 'shape', 'size', 'material'],
        'metaphor_source': ['color', 'shape', 'size', 'temperature'],
        'metaphor_target': ['color', 'shape', 'size', 'texture'],
        'mixed': ['color', 'shape', 'temperature', 'texture'],
    }

    results = {}

    for cond, attrs in conditions.items():
        run_stats = []

        for run in range(5):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')

            for r in range(300):
                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=attrs
                )
                target = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target)

            stats = speaker.language.get_stats()
            run_stats.append(stats)

        results[cond] = {
            'avg_sr': round(float(np.mean([s['success_rate'] for s in run_stats])), 4),
            'avg_vocab': round(float(np.mean([s['vocabulary_size'] for s in run_stats])), 1),
            'avg_combo': round(float(np.mean([s['combination_rate'] for s in run_stats])), 4),
            'avg_tri': round(float(np.mean([s['tri_symbol_rate'] for s in run_stats])), 4),
        }
        print(f"  {cond}: SR={results[cond]['avg_sr']:.1%}, "
              f"词汇={results[cond]['avg_vocab']:.0f}, "
              f"组合={results[cond]['avg_combo']:.2%}, "
              f"三符号={results[cond]['avg_tri']:.2%}")

    return results


def experiment_4_bidirectional_mapping():
    """实验 4：双向隐喻测试（warm→friendly 是否也 friendly→warm？）"""
    print("\n" + "=" * 60)
    print("实验 4：双向隐喻映射测试（500 轮 x 5 次）")
    print("=" * 60)

    # 正向：源域 → 目标域（temperature → emotion）
    # 反向：目标域 → 源域（emotion → temperature）
    directions = {
        'forward_source_to_target': True,
        'backward_target_to_source': False,
    }

    results = {}

    for direction, is_forward in directions.items():
        run_stats = []

        for run in range(5):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')

            for r in range(500):
                if is_forward:
                    # 正向：先学温度（源域），再看情感（目标域）
                    if r < 250:
                        attrs = ['color', 'shape', 'size', 'temperature']
                    else:
                        attrs = ['color', 'shape', 'size', 'texture']
                else:
                    # 反向：先学情感（目标域），再看温度（源域）
                    if r < 250:
                        attrs = ['color', 'shape', 'size', 'texture']
                    else:
                        attrs = ['color', 'shape', 'size', 'temperature']

                scene = generate_rich_scene_v2(
                    num_objects=random.randint(4, 6),
                    attribute_names=attrs
                )
                target = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target)

            stats = speaker.language.get_stats()
            run_stats.append(stats)

        results[direction] = {
            'avg_sr': round(float(np.mean([s['success_rate'] for s in run_stats])), 4),
            'avg_vocab': round(float(np.mean([s['vocabulary_size'] for s in run_stats])), 1),
            'avg_combo': round(float(np.mean([s['combination_rate'] for s in run_stats])), 4),
        }
        print(f"  {direction}: SR={results[direction]['avg_sr']:.1%}, "
              f"词汇={results[direction]['avg_vocab']:.0f}, "
              f"组合={results[direction]['avg_combo']:.2%}")

    # 比较方向效应
    fwd_sr = results['forward_source_to_target']['avg_sr']
    bwd_sr = results['backward_target_to_source']['avg_sr']
    print(f"\n  方向效应: 正向={fwd_sr:.1%} vs 反向={bwd_sr:.1%} "
          f"(差={abs(fwd_sr - bwd_sr):.4f})")

    return results


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_metaphor_emergence()
    results['experiment_2'] = experiment_2_cross_domain_strength()
    results['experiment_3'] = experiment_3_literal_vs_metaphorical()
    results['experiment_4'] = experiment_4_bidirectional_mapping()

    with open('embodied_metaphor_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 embodied_metaphor_results.json")
