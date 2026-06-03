"""
Phase 85: 隐语（Cryptolect）—— 群内秘密语言的涌现

核心思想：
隐语（cryptolect / cant / argot）是特定群体为了对外保密而发展的
内部语言变体。在人类社会中，工匠行会、走私者、青少年群体都发展
了外人听不懂的词汇替换系统。

本阶段测试：
- 群内 agent 是否自发发展出秘密词汇映射
- 秘密语言能否在群内保持高通信成功率
- 外人（out-group）对隐语的理解率是否显著降低
- 更大的群体是否发展更复杂的隐语但泄露率更高

涌现条件：
1. 群内成员有保密压力（被窃听威胁）
2. 词汇替换需保持群内可理解性
3. 外人可通过暴露学习逐步破解隐语

关键度量：
- 群内通信成功率（in-group SR）
- 群外理解率（out-group comprehension）
- 秘密词汇量（secret vocabulary size）
- 泄露率（leakage rate）
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, generate_rich_scene, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
)

# ============================================================
# 隐语转换映射：标准符号 → 秘密替代符号
# ============================================================
CRYPTO_TRANSFORMATION = {
    'red': 'scarlet', 'blue': 'azure', 'green': 'emerald',
    'big': 'massive', 'small': 'tiny_wee', 'circle': 'round_one',
    'square': 'boxy', 'triangle': 'pointy',
}


class CryptolectAgent:
    """
    隐语 Agent

    拥有标准语言和秘密词汇映射。
    群内通信时使用秘密符号替换，群外通信使用标准符号。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage,
                 group_id: int):
        self.agent_id = agent_id
        self.language = language
        self.group_id = group_id
        # secret_map: 标准符号 → 秘密替代
        self.secret_map: Dict[str, str] = {}
        # reverse_map: 秘密符号 → 标准符号（快速查找）
        self.reverse_map: Dict[str, str] = {}

    def learn_secret_mapping(self, standard: str, secret: str):
        """学习一个秘密词汇映射"""
        self.secret_map[standard] = secret
        self.reverse_map[secret] = standard

    def describe_secret(self, target: Dict) -> List[str]:
        """
        用秘密语言描述目标对象

        将标准符号替换为秘密替代，基于已知映射
        """
        utterance = []
        for attr, value in target.items():
            if value in self.secret_map:
                utterance.append(self.secret_map[value])
            else:
                utterance.append(value)
        # 去重
        seen = set()
        unique = []
        for s in utterance:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:3]

    def describe_standard(self, target: Dict) -> List[str]:
        """用标准语言描述目标对象（无秘密映射）"""
        utterance = []
        for attr, value in target.items():
            utterance.append(value)
        seen = set()
        unique = []
        for s in utterance:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:3]

    def interpret_secret(self, utterance: List[str],
                         scene: List[Dict]) -> int:
        """
        群内解读：将秘密符号反映射为标准符号后匹配
        """
        # 反映射
        mapped = []
        for sym in utterance:
            if sym in self.reverse_map:
                mapped.append(self.reverse_map[sym])
            else:
                mapped.append(sym)

        scores = []
        for obj in scene:
            score = sum(1 for s in mapped for v in obj.values() if v == s)
            scores.append(score)
        return int(np.argmax(scores)) if scores else 0

    def interpret_standard(self, utterance: List[str],
                           scene: List[Dict]) -> int:
        """标准解读（无秘密映射）"""
        scores = []
        for obj in scene:
            score = sum(1 for s in utterance for v in obj.values() if v == s)
            scores.append(score)
        return int(np.argmax(scores)) if scores else 0

    def detect_foreign(self, utterance: List[str]) -> bool:
        """
        检测话语是否来自外人

        如果话语使用了非秘密的标准符号（有映射的符号用了标准形式），
        说明说话者不在群内。
        """
        for sym in utterance:
            # 如果这个标准符号已被映射为秘密符号，
            # 但说话者用了标准形式 → 外人
            if sym in self.secret_map:
                return True
        return False


class InGroup:
    """
    群内组织

    管理一群共享秘密语言的 agent。
    """

    def __init__(self, group_id: int, members: List[CryptolectAgent],
                 secrecy_level: float = 0.5):
        self.group_id = group_id
        self.members = members
        self.language = EmergingLanguage()
        self.secrecy_level = secrecy_level  # 0.0=开放, 1.0=完全秘密
        self.secret_vocabulary: Dict[str, str] = {}

    def develop_cryptolect(self, rounds: int = 100):
        """
        发展隐语：在群内通信中逐步建立秘密词汇映射

        每轮：
        1. 随机选两个成员通信
        2. 根据保密压力决定是否引入新秘密映射
        3. 群内通信测试成功率
        """
        available_mappings = dict(CRYPTO_TRANSFORMATION)
        # 已使用的映射从可用池中移除
        for std in self.secret_vocabulary:
            available_mappings.pop(std, None)

        for r in range(rounds):
            # 根据保密压力决定是否引入新映射
            if (available_mappings
                    and random.random() < self.secrecy_level * 0.15):
                # 选择一个新的映射
                std = random.choice(list(available_mappings.keys()))
                secret = available_mappings.pop(std)
                self.secret_vocabulary[std] = secret
                # 教给所有成员
                for member in self.members:
                    member.learn_secret_mapping(std, secret)

            # 群内通信测试
            if len(self.members) < 2:
                continue
            speaker, listener = random.sample(self.members, 2)
            scene = generate_rich_scene('medium')
            target_idx = random.randint(0, len(scene) - 1)

            utterance = speaker.describe_secret(scene[target_idx])
            chosen = listener.interpret_secret(utterance, scene)
            success = (chosen == target_idx)

            self.language.record_usage(utterance, success)
            speaker.language.record_usage(utterance, success)
            listener.language.record_usage(utterance, success)

    def leakage_rate(self) -> float:
        """
        泄露率：秘密词汇被群外知晓的比例

        模拟：每个成员有一定概率在群外暴露秘密词汇
        """
        if not self.secret_vocabulary:
            return 0.0
        # 泄露与群体大小和保密压力成反比
        base_leak = 0.1 * (1.0 - self.secrecy_level)
        size_factor = len(self.members) / 10.0
        return min(1.0, base_leak * (1 + size_factor))


class CryptolectGame:
    """
    隐语通信游戏

    2 个群内 agent + 1 个群外窃听者。
    群内用秘密语言通信，窃听者尝试解读。
    """

    def __init__(self, in_group: InGroup,
                 eavesdropper: CryptolectAgent):
        self.in_group = in_group
        self.eavesdropper = eavesdropper
        self.games_played = 0
        self.in_group_successes = 0
        self.out_group_correct = 0
        self.secret_vocab_snapshots = []

    def play_round(self) -> Dict:
        """执行一轮游戏"""
        members = self.in_group.members
        if len(members) < 2:
            return {'in_success': False, 'out_correct': False}

        speaker, listener = random.sample(members, 2)
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)

        # 群内通信（用秘密语言）
        utterance = speaker.describe_secret(scene[target_idx])
        in_chosen = listener.interpret_secret(utterance, scene)
        in_success = (in_chosen == target_idx)

        # 窃听者尝试解读（用标准语言知识）
        out_chosen = self.eavesdropper.interpret_standard(utterance, scene)
        out_correct = (out_chosen == target_idx)

        # 记录
        self.games_played += 1
        if in_success:
            self.in_group_successes += 1
        if out_correct:
            self.out_group_correct += 1

        self.in_group.language.record_usage(utterance, in_success)
        speaker.language.record_usage(utterance, in_success)

        return {
            'in_success': in_success,
            'out_correct': out_correct,
            'utterance': utterance,
            'target_idx': target_idx,
        }


class BaselineOpenGame:
    """
    基线游戏：无秘密映射，开放通信

    用于对比：群外理解率应与群内接近
    """

    def __init__(self, agents: List[CryptolectAgent],
                 outsider: CryptolectAgent):
        self.agents = agents
        self.outsider = outsider
        self.games_played = 0
        self.in_group_successes = 0
        self.out_group_correct = 0

    def play_round(self) -> Dict:
        if len(self.agents) < 2:
            return {'in_success': False, 'out_correct': False}

        speaker, listener = random.sample(self.agents, 2)
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)

        # 标准通信（无秘密映射）
        utterance = speaker.describe_standard(scene[target_idx])
        in_chosen = listener.interpret_standard(utterance, scene)
        in_success = (in_chosen == target_idx)

        # 外人解读
        out_chosen = self.outsider.interpret_standard(utterance, scene)
        out_correct = (out_chosen == target_idx)

        self.games_played += 1
        if in_success:
            self.in_group_successes += 1
        if out_correct:
            self.out_group_correct += 1

        return {
            'in_success': in_success,
            'out_correct': out_correct,
        }


# ============================================================
# 实验
# ============================================================

def experiment_1_cryptolect_development(num_rounds: int = 200) -> Dict:
    """
    实验 1：隐语发展

    3 个群内 agent 发展隐语，追踪：
    - 秘密词汇量增长
    - 群内通信成功率
    - 群外理解率

    预期：秘密词汇增长到 8-12 个，群内 SR ~70%，群外 ~25%
    """
    print("=" * 60)
    print("实验 1：隐语发展")
    print("=" * 60)

    # 创建群内成员
    in_lang = EmergingLanguage()
    members = [
        CryptolectAgent(i, EmergingLanguage(), group_id=0)
        for i in range(3)
    ]
    group = InGroup(group_id=0, members=members, secrecy_level=0.7)

    # 窃听者
    eavesdropper = CryptolectAgent(
        agent_id=99, language=EmergingLanguage(), group_id=-1
    )
    game = CryptolectGame(group, eavesdropper)

    snapshots = []
    for r in range(num_rounds):
        # 每轮同时发展隐语和测试通信
        if len(group.members) >= 2:
            speaker, listener = random.sample(group.members, 2)

            # 根据保密压力引入新映射
            available = {
                k: v for k, v in CRYPTO_TRANSFORMATION.items()
                if k not in group.secret_vocabulary
            }
            if available and random.random() < group.secrecy_level * 0.12:
                std = random.choice(list(available.keys()))
                secret = available[std]
                group.secret_vocabulary[std] = secret
                for m in group.members:
                    m.learn_secret_mapping(std, secret)

            # 通信
            result = game.play_round()

        if (r + 1) % 40 == 0:
            in_sr = game.in_group_successes / max(1, game.games_played)
            out_sr = game.out_group_correct / max(1, game.games_played)
            snapshots.append({
                'round': r + 1,
                'secret_vocab_size': len(group.secret_vocabulary),
                'in_group_sr': round(in_sr, 4),
                'out_group_sr': round(out_sr, 4),
            })

    final_in_sr = game.in_group_successes / max(1, game.games_played)
    final_out_sr = game.out_group_correct / max(1, game.games_played)

    print(f"  秘密词汇量: {len(group.secret_vocabulary)}")
    print(f"  群内 SR: {final_in_sr:.3f}")
    print(f"  群外理解率: {final_out_sr:.3f}")
    print(f"  秘密映射: {group.secret_vocabulary}")

    return {
        'secret_vocab_size': len(group.secret_vocabulary),
        'secret_vocabulary': group.secret_vocabulary,
        'in_group_sr': round(final_in_sr, 4),
        'out_group_sr': round(final_out_sr, 4),
        'secrecy_advantage': round(final_in_sr - final_out_sr, 4),
        'snapshots': snapshots,
    }


def experiment_2_secret_vs_open(num_rounds: int = 200,
                                num_runs: int = 5) -> Dict:
    """
    实验 2：隐语 vs 开放通信

    对比 CryptolectGame 和 BaselineOpenGame。
    预期：群内 SR 相近，群外理解率：隐语 ~25% vs 开放 ~65%

    Returns: 对比结果
    """
    print("=" * 60)
    print("实验 2：隐语 vs 开放通信对比")
    print("=" * 60)

    crypto_in_srs = []
    crypto_out_srs = []
    open_in_srs = []
    open_out_srs = []

    for run in range(num_runs):
        # --- 隐语游戏 ---
        members_c = [
            CryptolectAgent(i, EmergingLanguage(), group_id=0)
            for i in range(3)
        ]
        group_c = InGroup(group_id=0, members=members_c, secrecy_level=0.7)
        eaves_c = CryptolectAgent(99, EmergingLanguage(), group_id=-1)
        game_c = CryptolectGame(group_c, eaves_c)

        for r in range(num_rounds):
            # 逐步引入映射
            available = {
                k: v for k, v in CRYPTO_TRANSFORMATION.items()
                if k not in group_c.secret_vocabulary
            }
            if available and random.random() < group_c.secrecy_level * 0.12:
                std = random.choice(list(available.keys()))
                secret = available[std]
                group_c.secret_vocabulary[std] = secret
                for m in group_c.members:
                    m.learn_secret_mapping(std, secret)
            game_c.play_round()

        crypto_in_srs.append(
            game_c.in_group_successes / max(1, game_c.games_played)
        )
        crypto_out_srs.append(
            game_c.out_group_correct / max(1, game_c.games_played)
        )

        # --- 开放游戏 ---
        members_o = [
            CryptolectAgent(i, EmergingLanguage(), group_id=0)
            for i in range(3)
        ]
        out_o = CryptolectAgent(99, EmergingLanguage(), group_id=-1)
        game_o = BaselineOpenGame(members_o, out_o)

        for _ in range(num_rounds):
            game_o.play_round()

        open_in_srs.append(
            game_o.in_group_successes / max(1, game_o.games_played)
        )
        open_out_srs.append(
            game_o.out_group_correct / max(1, game_o.games_played)
        )

    avg_crypto_in = float(np.mean(crypto_in_srs))
    avg_crypto_out = float(np.mean(crypto_out_srs))
    avg_open_in = float(np.mean(open_in_srs))
    avg_open_out = float(np.mean(open_out_srs))

    print(f"  隐语 - 群内 SR: {avg_crypto_in:.3f}, "
          f"群外理解: {avg_crypto_out:.3f}")
    print(f"  开放 - 群内 SR: {avg_open_in:.3f}, "
          f"群外理解: {avg_open_out:.3f}")
    print(f"  保密优势: {avg_crypto_in - avg_crypto_out:.3f} vs "
          f"{avg_open_in - avg_open_out:.3f}")

    return {
        'cryptolect': {
            'in_group_sr': round(avg_crypto_in, 4),
            'out_group_sr': round(avg_crypto_out, 4),
            'std_in': round(float(np.std(crypto_in_srs)), 4),
            'std_out': round(float(np.std(crypto_out_srs)), 4),
        },
        'open': {
            'in_group_sr': round(avg_open_in, 4),
            'out_group_sr': round(avg_open_out, 4),
            'std_in': round(float(np.std(open_in_srs)), 4),
            'std_out': round(float(np.std(open_out_srs)), 4),
        },
        'secrecy_advantage_cryptolect': round(
            avg_crypto_in - avg_crypto_out, 4
        ),
        'secrecy_advantage_open': round(
            avg_open_in - avg_open_out, 4
        ),
        'num_runs': num_runs,
    }


def experiment_3_cryptolect_complexity(
        group_sizes: List[int] = None) -> Dict:
    """
    实验 3：群体大小与隐语复杂度

    更大群体 → 更复杂的隐语？但泄露率更高？
    预期：更大群体发展更多秘密词汇但泄露更严重

    Args:
        group_sizes: 不同群体大小列表
    """
    if group_sizes is None:
        group_sizes = [3, 5, 8, 12]

    print("=" * 60)
    print("实验 3：群体大小与隐语复杂度")
    print("=" * 60)

    results = {}
    for size in group_sizes:
        members = [
            CryptolectAgent(i, EmergingLanguage(), group_id=0)
            for i in range(size)
        ]
        group = InGroup(group_id=0, members=members, secrecy_level=0.7)
        eavesdropper = CryptolectAgent(
            99, EmergingLanguage(), group_id=-1
        )
        game = CryptolectGame(group, eavesdropper)

        # 发展阶段
        group.develop_cryptolect(rounds=100)

        # 测试阶段
        for _ in range(200):
            available = {
                k: v for k, v in CRYPTO_TRANSFORMATION.items()
                if k not in group.secret_vocabulary
            }
            if available and random.random() < group.secrecy_level * 0.1:
                std = random.choice(list(available.keys()))
                secret = available[std]
                group.secret_vocabulary[std] = secret
                for m in group.members:
                    m.learn_secret_mapping(std, secret)
            game.play_round()

        in_sr = game.in_group_successes / max(1, game.games_played)
        out_sr = game.out_group_correct / max(1, game.games_played)
        leakage = group.leakage_rate()

        results[str(size)] = {
            'secret_vocab_size': len(group.secret_vocabulary),
            'in_group_sr': round(in_sr, 4),
            'out_group_sr': round(out_sr, 4),
            'leakage_rate': round(leakage, 4),
        }

        print(f"  群体 {size}人: 秘密词={len(group.secret_vocabulary)}, "
              f"群内SR={in_sr:.3f}, 群外={out_sr:.3f}, "
              f"泄露={leakage:.3f}")

    # 趋势分析
    sizes_list = sorted(group_sizes)
    vocab_sizes = [results[str(s)]['secret_vocab_size'] for s in sizes_list]
    leakages = [results[str(s)]['leakage_rate'] for s in sizes_list]

    if len(sizes_list) > 1:
        vocab_trend = float(np.polyfit(sizes_list, vocab_sizes, 1)[0])
        leak_trend = float(np.polyfit(sizes_list, leakages, 1)[0])
    else:
        vocab_trend = 0.0
        leak_trend = 0.0

    return {
        'by_group_size': results,
        'vocab_trend_per_member': round(vocab_trend, 4),
        'leakage_trend_per_member': round(leak_trend, 4),
    }


def experiment_4_cryptolect_robustness(
        exposure_levels: List[int] = None) -> Dict:
    """
    实验 4：隐语鲁棒性

    外人需要多少次暴露才能破解隐语？
    预期：50+ 次暴露后群外理解率上升到 ~50%

    Args:
        exposure_levels: 不同暴露次数列表
    """
    if exposure_levels is None:
        exposure_levels = [0, 10, 50, 100]

    print("=" * 60)
    print("实验 4：隐语鲁棒性（暴露破解）")
    print("=" * 60)

    results = {}

    for exposure in exposure_levels:
        # 每个暴露级别独立运行
        members = [
            CryptolectAgent(i, EmergingLanguage(), group_id=0)
            for i in range(3)
        ]
        group = InGroup(group_id=0, members=members, secrecy_level=0.7)

        # 先发展隐语
        group.develop_cryptolect(rounds=150)
        secret_vocab = dict(group.secret_vocabulary)

        if not secret_vocab:
            # 如果没有发展出隐语，强制注入几个映射
            for std, sec in list(CRYPTO_TRANSFORMATION.items())[:6]:
                group.secret_vocabulary[std] = sec
                for m in group.members:
                    m.learn_secret_mapping(std, sec)
            secret_vocab = dict(group.secret_vocabulary)

        # 窃听者暴露学习
        eavesdropper = CryptolectAgent(
            99, EmergingLanguage(), group_id=-1
        )

        # 暴露阶段：窃听者观察到群内通信
        known_by_eavesdropper = set()
        for e in range(exposure):
            scene = generate_rich_scene('medium')
            target_idx = random.randint(0, len(scene) - 1)
            speaker = random.choice(group.members)
            utterance = speaker.describe_secret(scene[target_idx])

            # 窃听者尝试从上下文推断映射
            for sym in utterance:
                if sym in secret_vocab.values():
                    # 反向查找
                    for std, sec in secret_vocab.items():
                        if sec == sym:
                            # 有一定概率学到这个映射
                            if random.random() < 0.15:
                                known_by_eavesdropper.add(sym)
                                eavesdropper.learn_secret_mapping(std, sec)

        # 测试阶段
        game = CryptolectGame(group, eavesdropper)
        for _ in range(200):
            game.play_round()

        in_sr = game.in_group_successes / max(1, game.games_played)
        out_sr = game.out_group_correct / max(1, game.games_played)
        cracked = len(known_by_eavesdropper)
        total_secret = len(secret_vocab)

        results[str(exposure)] = {
            'in_group_sr': round(in_sr, 4),
            'out_group_sr': round(out_sr, 4),
            'secret_symbols_cracked': cracked,
            'total_secret_symbols': total_secret,
            'crack_rate': round(
                cracked / max(1, total_secret), 4
            ),
        }

        print(f"  暴露 {exposure}次: 群内SR={in_sr:.3f}, "
              f"群外={out_sr:.3f}, "
              f"破解={cracked}/{total_secret}")

    # 趋势：暴露次数与群外理解率的关系
    exposures = sorted(exposure_levels)
    out_srs = [results[str(e)]['out_group_sr'] for e in exposures]

    if len(exposures) > 1:
        improvement_rate = float(np.polyfit(exposures, out_srs, 1)[0])
    else:
        improvement_rate = 0.0

    return {
        'by_exposure': results,
        'improvement_rate_per_exposure': round(improvement_rate, 5),
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
    results['experiment_1'] = experiment_1_cryptolect_development()
    results['experiment_2'] = experiment_2_secret_vs_open()
    results['experiment_3'] = experiment_3_cryptolect_complexity()
    results['experiment_4'] = experiment_4_cryptolect_robustness()

    output_file = 'cryptolect_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
