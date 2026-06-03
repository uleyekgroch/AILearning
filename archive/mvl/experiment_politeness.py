"""
Phase 80: 礼貌与面子语言涌现 — 社会语境下的语用标记生成

核心问题：Agent 在不同社会距离/权力关系中是否自发涌现礼貌标记？
礼貌标记（please/sorry/maybe/kinds 等）能否提高交际成功率？

4 个实验：
1. 礼貌标记涌现（300 轮随机社会语境）
2. 礼貌 vs 直接策略对比（300 轮 x 5 次）
3. 社会距离效应（5 个距离梯度）
4. 礼貌迁移（混合训练 → 陌生等级语境）
"""

import json
import random
import numpy as np
from language_emergence import EmergingLanguage, generate_rich_scene

# ============================================================
# 常量
# ============================================================

POLITENESS_MARKERS = {'please', 'sorry', 'maybe', 'thanks', 'excuse', 'kinda', 'perhaps', 'pardon'}

# 修正/感谢场景可用的标记
CORRECTION_MARKERS = ['sorry', 'excuse', 'pardon']
REQUEST_MARKERS = ['please', 'maybe', 'kinda', 'perhaps']
THANKS_MARKERS = ['thanks']


# ============================================================
# 社会语境
# ============================================================

class SocialContext:
    """描述一次交互的社会语境"""

    def __init__(self, relationship: str = 'stranger',
                 social_distance: float = 0.5,
                 power_asymmetry: float = 0.0,
                 face_threat_level: float = 0.0):
        self.relationship = relationship  # stranger/acquaintance/friend/hierarchical
        self.social_distance = social_distance  # 0.0 (close) ~ 1.0 (distant)
        self.power_asymmetry = power_asymmetry  # 0.0 (equal) ~ 1.0 (subordinate)
        self.face_threat_level = face_threat_level  # 0.0 (none) ~ 1.0 (high)

    def to_dict(self):
        return {
            'relationship': self.relationship,
            'social_distance': self.social_distance,
            'power_asymmetry': self.power_asymmetry,
            'face_threat_level': self.face_threat_level,
        }


def random_social_context(rng: random.Random = random) -> SocialContext:
    """随机生成一个社会语境"""
    rel = rng.choice(['stranger', 'acquaintance', 'friend', 'hierarchical'])
    dist_map = {
        'stranger': (0.7, 1.0),
        'acquaintance': (0.4, 0.7),
        'friend': (0.0, 0.3),
        'hierarchical': (0.5, 0.9),
    }
    d_lo, d_hi = dist_map[rel]
    sd = rng.uniform(d_lo, d_hi)
    pa = rng.uniform(0.3, 1.0) if rel == 'hierarchical' else rng.uniform(0.0, 0.3)
    ft = rng.uniform(0.0, 1.0)
    return SocialContext(rel, round(sd, 2), round(pa, 2), round(ft, 2))


# ============================================================
# 礼貌 Agent
# ============================================================

class PoliteAgent:
    """带有礼貌标记能力的 Agent"""

    def __init__(self, name: str):
        self.name = name
        self.lang = EmergingLanguage()
        self.politeness_history = []  # [{utterance, markers, context, success}]

    def describe_with_politeness(self, target: dict, context: SocialContext) -> list:
        """根据语境在描述中加入礼貌标记"""
        # 基本属性符号
        symbols = []
        for attr in ['color', 'shape', 'size', 'material']:
            if attr in target:
                symbols.append(target[attr])

        # 从已有词汇中选成功率最高的符号组合
        known = [s for s in symbols if s in self.lang.vocabulary]
        if not known:
            known = symbols  # 全部尝试

        # 判断是否需要礼貌标记
        markers = []
        sd = context.social_distance
        ft = context.face_threat_level

        if sd > 0.5 or ft > 0.5:
            # 请求/建议场景
            marker = random.choice(REQUEST_MARKERS)
            markers.append(marker)

        if ft > 0.7 and context.power_asymmetry > 0.3:
            # 高面子威胁 + 下级 → 双重礼貌
            marker = random.choice(REQUEST_MARKERS)
            if marker not in markers:
                markers.append(marker)

        if context.relationship == 'friend' and ft < 0.3:
            # 朋友之间低威胁场景不加标记
            markers = []

        # 感谢场景
        if random.random() < 0.15:
            markers.append('thanks')

        # 修正场景
        if ft > 0.6 and random.random() < 0.3:
            marker = random.choice(CORRECTION_MARKERS)
            if marker not in markers:
                markers.append(marker)

        # 组装：礼貌前缀 + 描述 + 感谢后缀
        prefix = [m for m in markers if m != 'thanks']
        suffix = [m for m in markers if m == 'thanks']
        utterance = prefix + known + suffix

        return utterance

    def interpret_polite(self, utterance: list, scene: list) -> dict:
        """剥离礼貌标记后进行标准解读"""
        # 去除礼貌标记
        content = [s for s in utterance if s not in POLITENESS_MARKERS]

        # 在场景中寻找最佳匹配
        best_match = None
        best_score = 0
        for obj in scene:
            score = 0
            for sym in content:
                for attr in ['color', 'shape', 'size', 'material']:
                    if obj.get(attr) == sym:
                        score += 1
            if score > best_score:
                best_score = score
                best_match = obj

        return {'match': best_match, 'score': best_score, 'content': content}

    def update_politeness(self, utterance: list, success: bool, context: SocialContext):
        """记录使用情况，追踪哪些标记在什么语境下有效"""
        markers_used = [s for s in utterance if s in POLITENESS_MARKERS]
        self.politeness_history.append({
            'markers': markers_used,
            'context': context.to_dict(),
            'success': success,
        })

        # 同时记录到语言系统（含标记）
        self.lang.record_usage(utterance, success)


# ============================================================
# 直接 Agent（基线）
# ============================================================

class DirectAgent:
    """无礼貌标记的基线 Agent"""

    def __init__(self, name: str):
        self.name = name
        self.lang = EmergingLanguage()

    def describe_direct(self, target: dict) -> list:
        """直接描述，不加任何礼貌标记"""
        symbols = []
        for attr in ['color', 'shape', 'size', 'material']:
            if attr in target:
                symbols.append(target[attr])
        known = [s for s in symbols if s in self.lang.vocabulary]
        return known if known else symbols

    def interpret_direct(self, utterance: list, scene: list) -> dict:
        """直接解读"""
        best_match = None
        best_score = 0
        for obj in scene:
            score = 0
            for sym in utterance:
                for attr in ['color', 'shape', 'size', 'material']:
                    if obj.get(attr) == sym:
                        score += 1
            if score > best_score:
                best_score = score
                best_match = obj
        return {'match': best_match, 'score': best_score}


# ============================================================
# 游戏引擎
# ============================================================

def _compute_success(base_rate: float, context: SocialContext,
                     markers_used: list) -> float:
    """根据语境和礼貌标记计算成功率"""
    bonus = 0.0
    sd = context.social_distance
    ft = context.face_threat_level

    # 陌生人 + please/maybe → +10%
    if context.relationship == 'stranger' and any(m in REQUEST_MARKERS for m in markers_used):
        bonus += 0.10

    # 高面子威胁 + 修正标记 → +8%
    if ft > 0.5 and any(m in CORRECTION_MARKERS for m in markers_used):
        bonus += 0.08

    # 下级对上级 + 礼貌 → +7%
    if context.power_asymmetry > 0.5 and any(m in REQUEST_MARKERS for m in markers_used):
        bonus += 0.07

    # 陌生人 + 无礼貌 → -5%
    if context.relationship == 'stranger' and not markers_used:
        bonus -= 0.05

    # 高面子威胁 + 无礼貌 → -8%
    if ft > 0.5 and not markers_used:
        bonus -= 0.08

    # 朋友 + please → 无加成无惩罚
    # 熟人 + please → +2%
    if context.relationship == 'acquaintance' and any(m in REQUEST_MARKERS for m in markers_used):
        bonus += 0.02

    return min(1.0, max(0.0, base_rate + bonus))


class PolitenessGame:
    """礼貌博弈：随机社会语境下的交互"""

    def __init__(self, speaker: PoliteAgent, listener: PoliteAgent):
        self.speaker = speaker
        self.listener = listener

    def play_round(self, context: SocialContext) -> dict:
        """执行一轮交互"""
        scene = generate_rich_scene('medium')
        target = random.choice(scene)

        # 说话者生成带礼貌的描述
        utterance = self.speaker.describe_with_politeness(target, context)
        markers_used = [s for s in utterance if s in POLITENESS_MARKERS]
        content = [s for s in utterance if s not in POLITENESS_MARKERS]

        # 基础匹配成功率
        interpretation = self.listener.interpret_polite(utterance, scene)
        base_score = interpretation['score']
        max_score = len(content)
        base_rate = base_score / max(max_score, 1)

        # 应用礼貌修饰
        effective_rate = _compute_success(base_rate, context, markers_used)
        success = random.random() < effective_rate

        # 更新双方
        self.speaker.update_politeness(utterance, success, context)
        self.listener.update_politeness(utterance, success, context)

        return {
            'success': success,
            'effective_rate': effective_rate,
            'base_rate': base_rate,
            'markers_used': markers_used,
            'content': content,
            'context': context.to_dict(),
        }


class BaselineDirectGame:
    """基线博弈：直接描述，无礼貌标记"""

    def __init__(self, speaker: DirectAgent, listener: DirectAgent):
        self.speaker = speaker
        self.listener = listener

    def play_round(self, context: SocialContext) -> dict:
        """执行一轮交互（无礼貌标记）"""
        scene = generate_rich_scene('medium')
        target = random.choice(scene)

        utterance = self.speaker.describe_direct(target)
        interpretation = self.listener.interpret_direct(utterance, scene)
        base_score = interpretation['score']
        max_score = len(utterance)
        base_rate = base_score / max(max_score, 1)

        # 基线也要承受无礼貌惩罚
        effective_rate = _compute_success(base_rate, context, [])
        success = random.random() < effective_rate

        self.speaker.lang.record_usage(utterance, success)
        self.listener.lang.record_usage(utterance, success)

        return {
            'success': success,
            'effective_rate': effective_rate,
            'base_rate': base_rate,
            'markers_used': [],
            'context': context.to_dict(),
        }


# ============================================================
# 实验 1：礼貌标记涌现
# ============================================================

def experiment_1_politeness_emergence(num_rounds=300):
    """300 轮随机社会语境，追踪礼貌标记的涌现"""
    print("=" * 60)
    print(f"实验 1：礼貌标记涌现（{num_rounds} 轮）")
    print("=" * 60)

    speaker = PoliteAgent('sp_emergence')
    listener = PoliteAgent('li_emergence')
    game = PolitenessGame(speaker, listener)

    marker_counts = {m: 0 for m in POLITENESS_MARKERS}
    context_marker_map = {rel: [] for rel in ['stranger', 'acquaintance', 'friend', 'hierarchical']}
    round_results = []

    for rd in range(num_rounds):
        ctx = random_social_context()
        result = game.play_round(ctx)

        for m in result['markers_used']:
            marker_counts[m] = marker_counts.get(m, 0) + 1
        context_marker_map[ctx.relationship].extend(result['markers_used'])

        round_results.append(result)

        if (rd + 1) % 100 == 0:
            sr = np.mean([r['success'] for r in round_results[-100:]])
            print(f"  轮次 {rd + 1}: 近 100 轮成功率={sr:.1%}")

    # 汇总
    total_successes = sum(1 for r in round_results if r['success'])
    overall_sr = total_successes / num_rounds

    emerged = {m: c for m, c in marker_counts.items() if c > 0}
    num_emerged = len(emerged)

    # 语境适配率：高距离/高威胁场景中正确使用标记的比例
    appropriate = 0
    appropriate_total = 0
    for r in round_results:
        ctx = r['context']
        sd = ctx['social_distance']
        ft = ctx['face_threat_level']
        if sd > 0.5 or ft > 0.5:
            appropriate_total += 1
            if r['markers_used']:
                appropriate += 1
    context_appropriate = appropriate / max(appropriate_total, 1)

    print(f"\n  涌现的礼貌标记: {emerged}")
    print(f"  标记数量: {num_emerged}")
    print(f"  整体成功率: {overall_sr:.1%}")
    print(f"  语境适配率: {context_appropriate:.1%}")

    # 各关系类型下使用的标记
    print(f"\n  各关系类型的标记使用:")
    for rel, markers in context_marker_map.items():
        if markers:
            from collections import Counter
            cnt = Counter(markers)
            top3 = cnt.most_common(3)
            print(f"    {rel}: {top3}")

    return {
        'overall_success_rate': round(overall_sr, 4),
        'emerged_markers': emerged,
        'num_emerged': num_emerged,
        'context_appropriate_rate': round(context_appropriate, 4),
        'marker_counts': marker_counts,
        'context_distribution': {rel: len(ms) for rel, ms in context_marker_map.items()},
    }


# ============================================================
# 实验 2：礼貌 vs 直接策略对比
# ============================================================

def experiment_2_polite_vs_direct(num_rounds=300, num_runs=5):
    """比较礼貌策略 vs 直接策略在不同社会语境下的表现"""
    print("\n" + "=" * 60)
    print(f"实验 2：礼貌 vs 直接对比（{num_rounds} 轮 x {num_runs} 次）")
    print("=" * 60)

    polite_srs = []
    direct_srs = []

    for run in range(num_runs):
        # 礼貌组
        sp_p = PoliteAgent(f'sp_p_{run}')
        li_p = PoliteAgent(f'li_p_{run}')
        pg = PolitenessGame(sp_p, li_p)

        polite_results = []
        for _ in range(num_rounds):
            ctx = random_social_context()
            polite_results.append(pg.play_round(ctx))

        p_sr = np.mean([r['success'] for r in polite_results])
        polite_srs.append(p_sr)

        # 直接组（使用相同语境序列）
        sp_d = DirectAgent(f'sp_d_{run}')
        li_d = DirectAgent(f'li_d_{run}')
        dg = BaselineDirectGame(sp_d, li_d)

        direct_results = []
        random.seed(42 + run)  # 重置以获得相同语境分布
        for _ in range(num_rounds):
            ctx = random_social_context()
            direct_results.append(dg.play_round(ctx))
        random.seed(42)  # 重置

        d_sr = np.mean([r['success'] for r in direct_results])
        direct_srs.append(d_sr)

        print(f"  运行 {run + 1}: 礼貌={p_sr:.1%}, 直接={d_sr:.1%}")

    avg_polite = np.mean(polite_srs)
    std_polite = np.std(polite_srs)
    avg_direct = np.mean(direct_srs)
    std_direct = np.std(direct_srs)

    print(f"\n  礼貌策略: {avg_polite:.1%} +/- {std_polite:.1%}")
    print(f"  直接策略: {avg_direct:.1%} +/- {std_direct:.1%}")
    print(f"  差异: {(avg_polite - avg_direct):.1%}")

    verdict = "PASS" if avg_polite > avg_direct + 0.05 else "REVIEW"
    print(f"  判定: {verdict} (礼貌 > 直接 +5% = PASS)")

    return {
        'polite_sr': round(float(avg_polite), 4),
        'polite_std': round(float(std_polite), 4),
        'direct_sr': round(float(avg_direct), 4),
        'direct_std': round(float(std_direct), 4),
        'difference': round(float(avg_polite - avg_direct), 4),
        'verdict': verdict,
    }


# ============================================================
# 实验 3：社会距离效应
# ============================================================

def experiment_3_social_distance_effect(distances=None):
    """固定社会距离，观察礼貌标记密度的变化"""
    if distances is None:
        distances = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n" + "=" * 60)
    print(f"实验 3：社会距离效应（{len(distances)} 个距离梯度 x 200 轮）")
    print("=" * 60)

    all_results = {}

    for dist in distances:
        marker_density_list = []
        sr_list = []

        for run in range(3):
            sp = PoliteAgent(f'sp_d{dist}_r{run}')
            li = PoliteAgent(f'li_d{dist}_r{run}')
            game = PolitenessGame(sp, li)

            markers_per_round = []
            successes = 0
            total = 200

            for _ in range(total):
                ctx = SocialContext(
                    relationship='stranger' if dist > 0.6 else
                                ('friend' if dist < 0.2 else 'acquaintance'),
                    social_distance=dist,
                    power_asymmetry=0.0,
                    face_threat_level=0.3,
                )
                result = game.play_round(ctx)
                markers_per_round.append(len(result['markers_used']))
                if result['success']:
                    successes += 1

            density = np.mean(markers_per_round)
            sr = successes / total
            marker_density_list.append(density)
            sr_list.append(sr)

        avg_density = np.mean(marker_density_list)
        avg_sr = np.mean(sr_list)
        all_results[str(dist)] = {
            'marker_density': round(float(avg_density), 4),
            'success_rate': round(float(avg_sr), 4),
        }

        print(f"  距离={dist:.2f}: 标记密度={avg_density:.2f}, 成功率={avg_sr:.1%}")

    # 检查趋势
    densities = [all_results[str(d)]['marker_density'] for d in distances]
    increasing = all(densities[i] <= densities[i + 1] for i in range(len(densities) - 1))
    trend = "单调递增" if increasing else "非单调"
    print(f"\n  标记密度趋势: {trend}")
    print(f"  密度范围: {min(densities):.2f} ~ {max(densities):.2f}")

    verdict = "PASS" if densities[-1] > densities[0] else "REVIEW"
    print(f"  判定: {verdict} (最远 > 最近 = PASS)")

    return {
        'distances': all_results,
        'trend': trend,
        'density_range': [round(float(min(densities)), 4),
                          round(float(max(densities)), 4)],
        'verdict': verdict,
    }


# ============================================================
# 实验 4：礼貌迁移
# ============================================================

def experiment_4_politeness_transfer(num_rounds=200, num_runs=3):
    """在混合语境训练后，测试在全新语境（hierarchical）中的迁移能力"""
    print("\n" + "=" * 60)
    print(f"实验 4：礼貌迁移（训练 {num_rounds} 轮混合 → 测试 100 轮等级语境 x {num_runs} 次）")
    print("=" * 60)

    transfer_srs = []
    train_srs = []

    for run in range(num_runs):
        # 训练阶段：混合语境
        sp = PoliteAgent(f'sp_tr_{run}')
        li = PoliteAgent(f'li_tr_{run}')
        game = PolitenessGame(sp, li)

        train_successes = 0
        for _ in range(num_rounds):
            ctx = random_social_context()
            result = game.play_round(ctx)
            if result['success']:
                train_successes += 1

        train_sr = train_successes / num_rounds
        train_srs.append(train_sr)

        # 测试阶段：全新等级语境（高权力不对称）
        test_successes = 0
        test_total = 100
        for _ in range(test_total):
            ctx = SocialContext(
                relationship='hierarchical',
                social_distance=random.uniform(0.7, 0.95),
                power_asymmetry=random.uniform(0.7, 1.0),
                face_threat_level=random.uniform(0.5, 1.0),
            )
            result = game.play_round(ctx)
            if result['success']:
                test_successes += 1

        test_sr = test_successes / test_total
        transfer_srs.append(test_sr)

        # 未经训练的基线
        sp_base = PoliteAgent(f'sp_base_{run}')
        li_base = PoliteAgent(f'li_base_{run}')
        base_game = PolitenessGame(sp_base, li_base)
        base_successes = 0
        for _ in range(test_total):
            ctx = SocialContext(
                relationship='hierarchical',
                social_distance=random.uniform(0.7, 0.95),
                power_asymmetry=random.uniform(0.7, 1.0),
                face_threat_level=random.uniform(0.5, 1.0),
            )
            result = base_game.play_round(ctx)
            if result['success']:
                base_successes += 1
        base_sr = base_successes / test_total

        print(f"  运行 {run + 1}: 训练SR={train_sr:.1%}, "
              f"迁移SR={test_sr:.1%}, 基线SR={base_sr:.1%}")

    avg_transfer = np.mean(transfer_srs)
    avg_train = np.mean(train_srs)
    transfer_rate = avg_transfer / max(avg_train, 0.01)

    print(f"\n  训练成功率: {avg_train:.1%}")
    print(f"  迁移成功率: {avg_transfer:.1%}")
    print(f"  迁移率（迁移/训练）: {transfer_rate:.1%}")

    verdict = "PASS" if transfer_rate > 0.6 else "REVIEW"
    print(f"  判定: {verdict} (迁移率 > 60% = PASS)")

    return {
        'train_sr': round(float(avg_train), 4),
        'transfer_sr': round(float(avg_transfer), 4),
        'transfer_rate': round(float(transfer_rate), 4),
        'verdict': verdict,
    }


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1_politeness_emergence'] = experiment_1_politeness_emergence()
    results['experiment_2_polite_vs_direct'] = experiment_2_polite_vs_direct()
    results['experiment_3_social_distance_effect'] = experiment_3_social_distance_effect()
    results['experiment_4_politeness_transfer'] = experiment_4_politeness_transfer()

    with open('politeness_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 politeness_results.json")
