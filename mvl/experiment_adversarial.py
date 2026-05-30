"""
Phase 68: 对抗性通信与欺骗 —— 信任、怀疑与声誉

核心思想：
所有前 62 个 Phase 假设合作通信——说话者总是真实描述，
听者总是试图正确理解。但欺骗是复杂通信系统的自然副产品
（在儿童 4 岁左右随心智理论出现）。

本阶段测试：
- 当说话者激励与听者不一致时，欺骗是否出现
- 听者是否发展出信任评估机制
- 信任标记（"sure"/"doubt"）是否涌现
- 声誉系统是否在多 agent 社会中形成

涌现条件：
1. 部分说话者有误导听者的激励
2. 听者可以通过经验学习区分可靠/不可靠信息源
3. 信任机制提供预测优势
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, Speaker, Listener, CommunicationGame,
    generate_rich_scene, _symbol_category,
)

# 信任相关标记
TRUST_MARKERS = {'sure', 'doubt', 'honest', 'trust', 'verify', 'warn'}


class DeceptiveSpeaker:
    """
    欺骗性说话者

    激励错位时可能描述非目标物体（策略性误导，非随机）。
    选择最迷惑的替代对象（与目标最相似的）。
    """

    def __init__(self, language: EmergingLanguage,
                 incentive_alignment: float = 0.0):
        """
        Args:
            incentive_alignment: 0.0 = 完全合作, 1.0 = 完全对抗
        """
        self.language = language
        self.incentive = incentive_alignment
        self.deception_count = 0
        self.honest_count = 0

    def describe(self, target_idx: int, scene_features: List[Dict],
                 listener_trust: float = 1.0) -> Tuple[List[str], bool]:
        """
        描述场景中的目标对象

        Returns: (utterance, was_honest)
        """
        target = scene_features[target_idx]

        # 决定是否欺骗
        will_deceive = (random.random() < self.incentive)

        if not will_deceive:
            # 诚实描述
            utterance = self._honest_describe(target, scene_features)
            self.honest_count += 1
            return (utterance, True)

        # 策略性欺骗：选择最迷惑的替代对象
        best_decoy_idx = self._find_best_decoy(target_idx, scene_features)
        if best_decoy_idx is None or best_decoy_idx == target_idx:
            utterance = self._honest_describe(target, scene_features)
            self.honest_count += 1
            return (utterance, True)

        decoy = scene_features[best_decoy_idx]
        utterance = self._honest_describe(decoy, scene_features)
        self.deception_count += 1

        # 低信任时可能添加信任标记
        if listener_trust < 0.5 and random.random() < 0.3:
            utterance = ['sure'] + utterance

        return (utterance, False)

    def _honest_describe(self, target: Dict,
                         scene_features: List[Dict]) -> List[str]:
        """诚实描述一个对象"""
        utterance = []
        for attr, value in target.items():
            utterance.append(value)

        # 去重
        seen = set()
        unique = []
        for s in utterance:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:3]  # 最多 3 个符号

    def _find_best_decoy(self, target_idx: int,
                         scene_features: List[Dict]) -> Optional[int]:
        """找到与目标最相似的替代对象（最迷惑）"""
        target = scene_features[target_idx]
        best_idx = None
        best_sim = -1

        for i, obj in enumerate(scene_features):
            if i == target_idx:
                continue
            # 计算特征重叠
            shared = sum(1 for k in target if k in obj and obj[k] == target[k])
            total = max(len(target), 1)
            sim = shared / total
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx


class SkepticalListener:
    """
    怀疑性听者

    维护每个说话者的信任分数。
    低信任时惩罚最佳匹配，考虑替代选项。
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.trust_scores: Dict[int, float] = defaultdict(lambda: 0.7)
        self.verification_count = 0
        self.correct_detections = 0

    def interpret(self, utterance: List[str], scene_features: List[Dict],
                  speaker_id: int) -> int:
        """
        解释话语，选择目标对象

        低信任时对最佳匹配施加惩罚
        """
        trust = self.trust_scores[speaker_id]

        # 标准匹配分数
        scores = []
        for obj in scene_features:
            score = 0.0
            for sym in utterance:
                for attr, value in obj.items():
                    if value == sym:
                        score += 1.0
            scores.append(score)

        if trust < 0.5:
            # 低信任：惩罚最佳匹配，考虑第二选择
            best_idx = np.argmax(scores)
            scores[best_idx] *= trust  # 按信任比例降低

            # 添加怀疑标记
            if random.random() < 0.2:
                self.language.record_usage(['doubt'], False)

        return int(np.argmax(scores))

    def update_trust(self, speaker_id: int, was_honest: bool):
        """Bayesian 更新信任分数"""
        prior = self.trust_scores[speaker_id]

        if was_honest:
            # 诚实：信任上升
            likelihood = 0.9
        else:
            # 欺骗：信任下降
            likelihood = 0.1
            self.correct_detections += 1

        posterior = (prior * likelihood) / (
            prior * likelihood + (1 - prior) * (1 - likelihood)
        )
        self.trust_scores[speaker_id] = np.clip(posterior, 0.05, 0.95)

    def get_trust(self, speaker_id: int) -> float:
        return self.trust_scores[speaker_id]


class AdversarialGame:
    """
    对抗性通信游戏

    1. 随机分配说话者激励（对齐/错位）
    2. 欺骗性说话者可能误导
    3. 怀疑性听者用信任判断
    4. 追踪欺骗成功率、信任校准、信任标记涌现
    """

    def __init__(self, speakers: Dict[int, DeceptiveSpeaker],
                 listener: SkepticalListener,
                 adversarial_ratio: float = 0.3):
        self.speakers = speakers
        self.listener = listener
        self.adversarial_ratio = adversarial_ratio
        self.games_played = 0
        self.successes = 0
        self.deceptions_attempted = 0
        self.deceptions_succeeded = 0
        self.honest_games = 0
        self.honest_successes = 0

    def play_round(self) -> Dict:
        # 选择说话者
        speaker_id = random.choice(list(self.speakers.keys()))
        speaker = self.speakers[speaker_id]

        # 生成场景
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)

        # 获取听者信任
        trust = self.listener.get_trust(speaker_id)

        # 说话者描述
        utterance, was_honest = speaker.describe(
            target_idx, scene, trust
        )

        # 听者选择
        chosen_idx = self.listener.interpret(utterance, scene, speaker_id)

        # 验证
        success = (chosen_idx == target_idx)

        # 更新信任
        self.listener.update_trust(speaker_id, was_honest)

        # 统计
        self.games_played += 1
        if success:
            self.successes += 1

        if not was_honest:
            self.deceptions_attempted += 1
            if not success:
                # 欺骗失败 = 听者正确抵抗
                pass
            else:
                self.deceptions_succeeded += 1
        else:
            self.honest_games += 1
            if success:
                self.honest_successes += 1

        # 记录语言使用
        self.listener.language.record_usage(utterance, success)

        return {
            'success': success,
            'was_honest': was_honest,
            'speaker_id': speaker_id,
        }


class BaselineTrustGame:
    """基线：有激励错位但无信任机制"""

    def __init__(self, speakers: Dict[int, DeceptiveSpeaker]):
        self.speakers = speakers
        self.language = EmergingLanguage()
        self.games_played = 0
        self.successes = 0

    def play_round(self) -> Dict:
        speaker_id = random.choice(list(self.speakers.keys()))
        speaker = self.speakers[speaker_id]
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)

        utterance, was_honest = speaker.describe(target_idx, scene, 1.0)

        # 标准匹配（无信任机制）
        scores = []
        for obj in scene:
            score = sum(1 for s in utterance for a, v in obj.items() if v == s)
            scores.append(score)
        chosen = int(np.argmax(scores))
        success = (chosen == target_idx)

        self.games_played += 1
        if success:
            self.successes += 1

        return {'success': success, 'was_honest': was_honest}


def create_speakers(num_speakers: int = 5,
                    adversarial_ids: List[int] = None) -> Dict[int, DeceptiveSpeaker]:
    """创建说话者集合"""
    if adversarial_ids is None:
        adversarial_ids = []
    speakers = {}
    for i in range(num_speakers):
        incentive = 0.8 if i in adversarial_ids else 0.0
        speakers[i] = DeceptiveSpeaker(EmergingLanguage(), incentive)
    return speakers


# ============================================================
# 实验
# ============================================================

def experiment_1_deception_emergence(num_rounds: int = 300) -> Dict:
    """
    实验 1：欺骗涌现

    5 个说话者（2 个对抗性），1 个听者。300 轮。
    """
    print("=" * 60)
    print("实验 1：欺骗涌现与信任标记")
    print("=" * 60)

    speakers = create_speakers(5, adversarial_ids=[2, 4])
    listener = SkepticalListener(EmergingLanguage())
    game = AdversarialGame(speakers, listener, adversarial_ratio=0.4)

    snapshots = []
    for r in range(num_rounds):
        game.play_round()
        if (r + 1) % 50 == 0:
            trust_vals = {k: round(v, 3) for k, v in listener.trust_scores.items()}
            deception_sr = (game.deceptions_succeeded / max(1, game.deceptions_attempted))
            honest_sr = game.honest_successes / max(1, game.honest_games)
            trust_markers = sum(1 for m in TRUST_MARKERS if m in listener.language.vocabulary)
            snapshots.append({
                'round': r + 1,
                'overall_sr': round(game.successes / max(1, game.games_played), 4),
                'deception_sr': round(deception_sr, 4),
                'honest_sr': round(honest_sr, 4),
                'trust_scores': trust_vals,
                'trust_markers': trust_markers,
            })

    print(f"\n  整体成功率: {game.successes / max(1, game.games_played):.3f}")
    print(f"  欺骗成功率: {game.deceptions_succeeded / max(1, game.deceptions_attempted):.3f}")
    print(f"  诚实成功率: {game.honest_successes / max(1, game.honest_games):.3f}")
    print(f"  信任分数: {dict(listener.trust_scores)}")
    print(f"  信任标记: {sum(1 for m in TRUST_MARKERS if m in listener.language.vocabulary)}/{len(TRUST_MARKERS)}")

    return {
        'final_overall_sr': round(game.successes / max(1, game.games_played), 4),
        'deception_success_rate': round(game.deceptions_succeeded / max(1, game.deceptions_attempted), 4),
        'honest_success_rate': round(game.honest_successes / max(1, game.honest_games), 4),
        'trust_scores': {k: round(v, 3) for k, v in listener.trust_scores.items()},
        'trust_markers': sum(1 for m in TRUST_MARKERS if m in listener.language.vocabulary),
        'snapshots': snapshots,
    }


def experiment_2_trust_calibration(num_rounds: int = 300) -> Dict:
    """
    实验 2：信任校准

    不同对抗比例下的信任校准准确率。
    """
    print("=" * 60)
    print("实验 2：信任校准")
    print("=" * 60)

    ratios = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = {}

    for ratio in ratios:
        num_adv = max(1, int(5 * ratio))
        adv_ids = list(range(num_adv))
        speakers = create_speakers(5, adversarial_ids=adv_ids)
        listener = SkepticalListener(EmergingLanguage())
        game = AdversarialGame(speakers, listener, adversarial_ratio=ratio)

        for _ in range(num_rounds):
            game.play_round()

        # 校准：信任分数与实际诚实度的相关性
        trust_vals = []
        honest_vals = []
        for sid in speakers:
            trust_vals.append(listener.get_trust(sid))
            honest_vals.append(1.0 if sid not in adv_ids else 0.0)

        if len(set(trust_vals)) > 1 and len(set(honest_vals)) > 1:
            correlation = np.corrcoef(trust_vals, honest_vals)[0, 1]
        else:
            correlation = 0.0

        results[str(ratio)] = {
            'trust_correlation': round(float(correlation), 4),
            'trust_scores': {k: round(v, 3) for k, v in listener.trust_scores.items()},
            'success_rate': round(game.successes / max(1, game.games_played), 4),
        }
        print(f"  对抗比 {ratio}: 校准={correlation:.3f}, "
              f"信任={results[str(ratio)]['trust_scores']}, "
              f"SR={results[str(ratio)]['success_rate']:.3f}")

    return results


def experiment_3_reputation(num_agents: int = 5, num_rounds: int = 500) -> Dict:
    """
    实验 3：多 agent 声誉

    每个 agent 同时是说话者和听者，维护对所有人的信任分数。
    2 个欺骗者应被隔离。
    """
    print("=" * 60)
    print("实验 3：多 Agent 声誉系统")
    print("=" * 60)

    deceptive_ids = [1, 3]
    agents = {}
    for i in range(num_agents):
        incentive = 0.8 if i in deceptive_ids else 0.0
        agents[i] = {
            'speaker': DeceptiveSpeaker(EmergingLanguage(), incentive),
            'listener': SkepticalListener(EmergingLanguage()),
            'is_deceptive': i in deceptive_ids,
        }

    # 每轮随机选一对通信
    for _ in range(num_rounds):
        sp_id, li_id = random.sample(range(num_agents), 2)
        speaker = agents[sp_id]['speaker']
        listener = agents[li_id]['listener']

        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)

        trust = listener.get_trust(sp_id)
        utterance, was_honest = speaker.describe(target_idx, scene, trust)
        chosen = listener.interpret(utterance, scene, sp_id)
        success = (chosen == target_idx)
        listener.update_trust(sp_id, was_honest)
        listener.language.record_usage(utterance, success)

    # 分析信任网络
    trust_matrix = {}
    for li_id in range(num_agents):
        trust_matrix[li_id] = {
            str(sp_id): round(agents[li_id]['listener'].get_trust(sp_id), 3)
            for sp_id in range(num_agents) if sp_id != li_id
        }

    # 诚实 agent 对欺骗者的平均信任
    honest_to_deceptive = []
    honest_to_honest = []
    for li_id in range(num_agents):
        if agents[li_id]['is_deceptive']:
            continue
        for sp_id in range(num_agents):
            if sp_id == li_id:
                continue
            t = agents[li_id]['listener'].get_trust(sp_id)
            if agents[sp_id]['is_deceptive']:
                honest_to_deceptive.append(t)
            else:
                honest_to_honest.append(t)

    avg_h2d = float(np.mean(honest_to_deceptive)) if honest_to_deceptive else 0
    avg_h2h = float(np.mean(honest_to_honest)) if honest_to_honest else 0
    isolation = avg_h2h - avg_h2d

    print(f"  诚实→诚实: {avg_h2h:.3f}")
    print(f"  诚实→欺骗: {avg_h2d:.3f}")
    print(f"  隔离度: {isolation:.3f}")

    return {
        'honest_to_honest_trust': round(avg_h2h, 4),
        'honest_to_deceptive_trust': round(avg_h2d, 4),
        'isolation_gap': round(isolation, 4),
        'trust_matrix': trust_matrix,
    }


def experiment_4_trust_recovery(pre_rounds: int = 200,
                                 post_rounds: int = 100) -> Dict:
    """
    实验 4：信任恢复

    Agent 欺骗 200 轮，然后转为诚实 100 轮。
    测量信任恢复速度。
    """
    print("=" * 60)
    print("实验 4：信任恢复")
    print("=" * 60)

    snapshots = []
    listener = SkepticalListener(EmergingLanguage())

    # Phase 1: 欺骗阶段
    speaker = DeceptiveSpeaker(EmergingLanguage(), incentive_alignment=0.8)
    for r in range(pre_rounds):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        trust = listener.get_trust(0)
        utterance, was_honest = speaker.describe(target_idx, scene, trust)
        chosen = listener.interpret(utterance, scene, 0)
        listener.update_trust(0, was_honest)
        if (r + 1) % 50 == 0:
            snapshots.append({
                'phase': 'deceptive',
                'round': r + 1,
                'trust': round(listener.get_trust(0), 4),
            })

    trust_after_deception = listener.get_trust(0)
    print(f"  欺骗后信任: {trust_after_deception:.3f}")

    # Phase 2: 恢复阶段
    speaker.incentive_alignment = 0.0  # 转为诚实
    recovery_snapshots = []
    for r in range(post_rounds):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        trust = listener.get_trust(0)
        utterance, was_honest = speaker.describe(target_idx, scene, trust)
        chosen = listener.interpret(utterance, scene, 0)
        listener.update_trust(0, was_honest)
        if (r + 1) % 25 == 0:
            snapshots.append({
                'phase': 'recovery',
                'round': r + 1,
                'trust': round(listener.get_trust(0), 4),
            })
            recovery_snapshots.append({
                'round': r + 1,
                'trust': round(listener.get_trust(0), 4),
            })

    trust_after_recovery = listener.get_trust(0)
    recovery_ratio = trust_after_recovery / 0.7  # 相对于基线 0.7

    print(f"  恢复后信任: {trust_after_recovery:.3f}")
    print(f"  恢复率: {recovery_ratio:.3f}")

    return {
        'trust_after_deception': round(float(trust_after_deception), 4),
        'trust_after_recovery': round(float(trust_after_recovery), 4),
        'recovery_ratio': round(float(recovery_ratio), 4),
        'snapshots': snapshots,
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_deception_emergence()
    results['experiment_2'] = experiment_2_trust_calibration()
    results['experiment_3'] = experiment_3_reputation()
    results['experiment_4'] = experiment_4_trust_recovery()

    output_file = 'adversarial_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
