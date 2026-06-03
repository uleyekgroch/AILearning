"""
跨代知识传递：文化的积累与传承

核心机制：
1. 直接交流 — 老 agent 用自己的语言描述，新 agent 用自己的语言理解
2. 知识播种 — 将老 agent 的语言统计数据复制给新 agent

代际更替流程：
第1代: 从零学习 → 教学期 → 第2代诞生
第2代: 从老 agent 学习 → 教学期 → 第3代诞生
...
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from language_emergence import (
    LanguageAgent, cross_language_round, generate_rich_scene, EmergingLanguage
)


class GenerationalChain:
    """代际传递链"""

    def __init__(self, population_size: int = 3,
                 rounds_per_generation: int = 500,
                 teaching_rounds: int = 500,
                 transfer_mode: str = 'direct',
                 complexity: str = 'extreme'):
        self.population_size = population_size
        self.rounds_per_gen = rounds_per_generation
        self.teaching_rounds = teaching_rounds
        self.transfer_mode = transfer_mode  # 'direct', 'seed', 'none'
        self.complexity = complexity

        self.current_generation = 0
        self.current_agents: List[LanguageAgent] = []
        self.generations: List[Dict] = []  # 每代的记录

    def _create_generation(self, gen_num: int) -> List[LanguageAgent]:
        """创建新一代 agent"""
        return [LanguageAgent(f"G{gen_num}_A{i}") for i in range(self.population_size)]

    def _learning_phase(self, agents: List[LanguageAgent],
                        rounds: int, seed: int = 42) -> Dict:
        """学习期：agent 之间互相交流学习"""
        np.random.seed(seed)

        successes = 0
        total = 0

        for round_idx in range(rounds):
            scene = generate_rich_scene(self.complexity)
            target_idx = np.random.randint(0, len(scene))

            # 随机配对
            a, b = agents[np.random.randint(len(agents))], agents[np.random.randint(len(agents))]
            while a.id == b.id and len(agents) > 1:
                b = agents[np.random.randint(len(agents))]

            if np.random.random() < 0.5:
                speaker, listener = a, b
            else:
                speaker, listener = b, a

            success = cross_language_round(speaker, listener, scene, target_idx)
            successes += int(success)
            total += 1

        return {
            'success_rate': successes / total if total > 0 else 0,
            'total_rounds': rounds,
        }

    def _teaching_phase_direct(self, old_agents: List[LanguageAgent],
                               new_agents: List[LanguageAgent],
                               rounds: int, seed: int = 42) -> Dict:
        """教学期：直接交流 — 老 agent 教新 agent"""
        np.random.seed(seed)

        successes = 0
        total = 0

        for round_idx in range(rounds):
            scene = generate_rich_scene(self.complexity)
            target_idx = np.random.randint(0, len(scene))

            # 老 agent 做 speaker，新 agent 做 listener
            old = old_agents[np.random.randint(len(old_agents))]
            new = new_agents[np.random.randint(len(new_agents))]

            # 老 agent 用自己的语言描述
            utterance = old.speak(scene[target_idx], scene)

            if utterance:
                # 新 agent 用自己的语言理解
                guess = new.listen(utterance, scene)
                success = guess == target_idx

                # 双方都从交流中学习
                old.update_from_communication(utterance, success)
                new.update_from_communication(utterance, success)

                successes += int(success)
                total += 1

        return {
            'success_rate': successes / total if total > 0 else 0,
            'total_rounds': rounds,
        }

    def _teaching_phase_seed(self, old_agents: List[LanguageAgent],
                             new_agents: List[LanguageAgent]) -> Dict:
        """教学期：知识播种 — 复制语言统计数据"""
        # 收集老 agent 的语言统计
        avg_freq = {}  # {symbol: avg_frequency}
        avg_collocations = {}  # {pair: avg_count}

        for agent in old_agents:
            for symbol, stats in agent.language.vocabulary.items():
                freq = stats.get('frequency', 0)
                avg_freq[symbol] = avg_freq.get(symbol, 0) + freq
            for pair, stats in agent.language.collocations.items():
                count = stats.get('count', 0)
                avg_collocations[pair] = avg_collocations.get(pair, 0) + count

        # 取平均
        n = len(old_agents)
        for symbol in avg_freq:
            avg_freq[symbol] /= n
        for pair in avg_collocations:
            avg_collocations[pair] /= n

        # 播种给新 agent
        for agent in new_agents:
            # 设置词汇频率（带随机扰动）
            for symbol, base_freq in avg_freq.items():
                noise = np.random.normal(0, max(base_freq * 0.2, 1))
                freq = max(1, int(base_freq + noise))
                successes = max(0, freq // 2)
                agent.language.vocabulary[symbol] = {
                    'frequency': freq,
                    'successes': successes,
                    'success_rate': successes / freq,
                }

            # 设置组合模式（带随机扰动）
            for pair, base_count in avg_collocations.items():
                noise = np.random.normal(0, max(base_count * 0.2, 1))
                count = max(1, int(base_count + noise))
                agent.language.collocations[pair] = {
                    'count': count,
                    'successes': count // 2,
                }

        return {
            'symbols_transferred': len(avg_freq),
            'collocations_transferred': len(avg_collocations),
        }

    def _evaluate_generation(self, agents: List[LanguageAgent],
                             rounds: int = 200, seed: int = 99) -> Dict:
        """评估一代的学习成果（冻结模式：不更新语言）"""
        np.random.seed(seed)

        successes = 0
        total = 0
        all_vocab = set()
        total_combos = 0

        for round_idx in range(rounds):
            scene = generate_rich_scene(self.complexity)
            target_idx = np.random.randint(0, len(scene))

            a, b = agents[np.random.randint(len(agents))], agents[np.random.randint(len(agents))]
            while a.id == b.id and len(agents) > 1:
                b = agents[np.random.randint(len(agents))]

            if np.random.random() < 0.5:
                speaker, listener = a, b
            else:
                speaker, listener = b, a

            # 冻结模式：只测试，不更新语言
            utterance = speaker.speak(scene[target_idx], scene)
            if utterance:
                guess = listener.listen(utterance, scene)
                success = (guess == target_idx)
            else:
                success = False
            successes += int(success)
            total += 1

        # 收集语言统计
        for agent in agents:
            all_vocab.update(agent.language.vocabulary.keys())
            total_combos += len(agent.language.collocations)

        return {
            'success_rate': successes / total if total > 0 else 0,
            'vocabulary_size': len(all_vocab),
            'avg_collocations': total_combos / len(agents),
        }

    def run_single_generation(self, gen_num: int,
                              parent_agents: Optional[List[LanguageAgent]] = None,
                              seed: int = 42) -> Dict:
        """运行一代：创建 → 学习/教学 → 评估"""
        self.current_generation = gen_num

        # 创建新一代
        new_agents = self._create_generation(gen_num)

        if parent_agents is not None and self.transfer_mode != 'none':
            # 有父代：教学期
            if self.transfer_mode == 'direct':
                teach_result = self._teaching_phase_direct(
                    parent_agents, new_agents, self.teaching_rounds, seed
                )
            elif self.transfer_mode == 'seed':
                teach_result = self._teaching_phase_seed(parent_agents, new_agents)
                # 播种后还需要学习期
                learn_result = self._learning_phase(
                    new_agents, self.rounds_per_gen, seed + 1000
                )
                teach_result['learning'] = learn_result
            else:
                teach_result = {}
        else:
            # 无父代：纯学习期
            learn_result = self._learning_phase(new_agents, self.rounds_per_gen, seed)
            teach_result = {'learning': learn_result}

        # 评估（冻结模式：不更新语言，30 轮快速评估）
        eval_result = self._evaluate_generation(new_agents, rounds=30, seed=seed + 5000)

        record = {
            'generation': gen_num,
            'transfer_mode': self.transfer_mode,
            'teaching': teach_result,
            'evaluation': eval_result,
            'agents': new_agents,
        }

        self.generations.append(record)
        self.current_agents = new_agents

        return record

    def run_chain(self, num_generations: int, base_seed: int = 42) -> List[Dict]:
        """运行完整的代际链"""
        results = []

        for gen in range(num_generations):
            parent = self.current_agents if gen > 0 else None
            result = self.run_single_generation(gen, parent, seed=base_seed + gen * 100)
            results.append(result)

        return results


def compare_transfer_modes(num_generations: int = 4,
                           population_size: int = 3,
                           rounds_per_generation: int = 500,
                           teaching_rounds: int = 500,
                           complexity: str = 'extreme',
                           seed: int = 42) -> Dict[str, List[Dict]]:
    """对比不同传递模式"""
    modes = ['direct', 'seed', 'none']
    results = {}

    for mode in modes:
        chain = GenerationalChain(
            population_size=population_size,
            rounds_per_generation=rounds_per_generation,
            teaching_rounds=teaching_rounds,
            transfer_mode=mode,
            complexity=complexity,
        )
        chain_results = chain.run_chain(num_generations, seed)
        results[mode] = chain_results

    return results


def vary_teaching_duration(teaching_rounds_list: List[int] = None,
                           num_generations: int = 3,
                           population_size: int = 3,
                           rounds_per_generation: int = 500,
                           complexity: str = 'extreme',
                           seed: int = 42) -> Dict[int, List[Dict]]:
    """对比不同教学时长"""
    if teaching_rounds_list is None:
        teaching_rounds_list = [100, 300, 500, 1000]

    results = {}

    for teach_rounds in teaching_rounds_list:
        chain = GenerationalChain(
            population_size=population_size,
            rounds_per_generation=rounds_per_generation,
            teaching_rounds=teach_rounds,
            transfer_mode='direct',
            complexity=complexity,
        )
        chain_results = chain.run_chain(num_generations, seed)
        results[teach_rounds] = chain_results

    return results
