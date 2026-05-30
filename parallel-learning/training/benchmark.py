"""评估基准套件

系统化评估学习质量：
1. 知识保留测试 — 学习后能否回忆
2. 推理准确率 — 因果、类比推理
3. 概念覆盖度 — 知识图谱完整性
4. 增量学习效率 — 新知识学习速度

运行方式：
    python training/benchmark.py
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.integrated_ai import IntegratedAI


class BenchmarkSuite:
    """评估基准套件"""

    def __init__(self):
        self.ai = IntegratedAI()
        self.results = {}

    def run_all(self) -> Dict:
        """运行所有测试"""
        print("=" * 70)
        print("AI学习系统基准测试")
        print("=" * 70)

        # 1. 知识保留测试
        print("\n[1/4] 知识保留测试...")
        self.results['retention'] = self.test_retention()

        # 2. 推理准确率测试
        print("\n[2/4] 推理准确率测试...")
        self.results['reasoning'] = self.test_reasoning()

        # 3. 概念覆盖度测试
        print("\n[3/4] 概念覆盖度测试...")
        self.results['coverage'] = self.test_coverage()

        # 4. 增量学习效率测试
        print("\n[4/4] 增量学习效率测试...")
        self.results['efficiency'] = self.test_efficiency()

        # 计算总分
        self.results['overall'] = self._calculate_overall_score()

        return self.results

    def test_retention(self) -> Dict:
        """知识保留测试"""
        test_cases = [
            # (学习文本, 问题, 预期关键词)
            ('人工智能是计算机科学的一个分支', '什么是人工智能', ['计算机科学', '分支']),
            ('Python是一种编程语言', 'Python是什么', ['编程语言']),
            ('牛顿发现了万有引力定律', '牛顿发现了什么', ['万有引力']),
            ('水在100度沸腾', '水在多少度沸腾', ['100']),
            ('因为下雨，所以地面湿了', '为什么地面湿了', ['下雨']),
        ]

        correct = 0
        total = len(test_cases)
        details = []

        for text, question, expected_keywords in test_cases:
            # 学习
            self.ai.learn(text)

            # 查询
            answer = self.ai.think(question)

            # 检查是否包含预期关键词
            found = any(kw in answer for kw in expected_keywords)
            if found:
                correct += 1

            details.append({
                'text': text,
                'question': question,
                'answer': answer,
                'expected': expected_keywords,
                'correct': found,
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'details': details,
        }

    def test_reasoning(self) -> Dict:
        """推理准确率测试"""
        test_cases = [
            # 因果推理
            ('因为全球变暖，冰川开始融化。', '为什么冰川融化', '全球变暖'),
            ('由于他努力学习，所以考试及格了。', '为什么考试及格', '努力学习'),

            # 类比推理
            ('水流像电流一样流动。', '水流像什么', '电流'),
            ('心脏像水泵一样跳动。', '心脏像什么', '水泵'),
        ]

        correct = 0
        total = len(test_cases)
        details = []

        for text, question, expected in test_cases:
            # 学习
            self.ai.learn(text)

            # 查询
            answer = self.ai.think(question)

            # 检查是否包含预期答案
            found = expected in answer
            if found:
                correct += 1

            details.append({
                'text': text,
                'question': question,
                'answer': answer,
                'expected': expected,
                'correct': found,
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'details': details,
        }

    def test_coverage(self) -> Dict:
        """概念覆盖度测试"""
        # 学习多个领域的知识
        domains = {
            '科学': ['牛顿发现了万有引力定律', '爱因斯坦提出了相对论', '达尔文提出了进化论'],
            '技术': ['人工智能是计算机科学的一个分支', 'Python是一种编程语言', '互联网连接了全世界'],
            '自然': ['太阳是太阳系的中心', '水在100度沸腾', '地球围绕太阳公转'],
            '社会': ['张三告诉李四一个秘密', '公司开发了新产品', '政府制定了新政策'],
        }

        coverage = {}
        for domain, texts in domains.items():
            for text in texts:
                self.ai.learn(text)

            # 统计该领域的概念
            stats = self.ai.get_stats()
            coverage[domain] = {
                'texts': len(texts),
                'concepts': stats.get('abstraction', {}).get('total_concepts', 0),
            }

        # 计算总体覆盖度
        total_concepts = sum(c['concepts'] for c in coverage.values())
        avg_concepts = total_concepts / len(coverage) if coverage else 0

        return {
            'domains': coverage,
            'total_concepts': total_concepts,
            'avg_concepts_per_domain': avg_concepts,
        }

    def test_efficiency(self) -> Dict:
        """增量学习效率测试"""
        # 测试学习速度
        test_texts = [
            '人工智能是计算机科学的一个分支',
            'Python是一种编程语言',
            '牛顿发现了万有引力定律',
            '太阳是太阳系的中心',
            '水在100度沸腾',
        ]

        start = time.time()
        for text in test_texts:
            self.ai.learn(text)
        elapsed = time.time() - start

        speed = len(test_texts) / elapsed if elapsed > 0 else 0

        return {
            'texts': len(test_texts),
            'elapsed': elapsed,
            'speed': speed,
        }

    def _calculate_overall_score(self) -> Dict:
        """计算总分"""
        scores = []

        # 知识保留分数
        if 'retention' in self.results:
            scores.append(self.results['retention']['accuracy'])

        # 推理分数
        if 'reasoning' in self.results:
            scores.append(self.results['reasoning']['accuracy'])

        # 覆盖度分数 (归一化)
        if 'coverage' in self.results:
            coverage_score = min(1.0, self.results['coverage']['total_concepts'] / 100)
            scores.append(coverage_score)

        # 效率分数 (归一化)
        if 'efficiency' in self.results:
            efficiency_score = min(1.0, self.results['efficiency']['speed'] / 100)
            scores.append(efficiency_score)

        overall = sum(scores) / len(scores) if scores else 0

        return {
            'score': overall,
            'grade': self._score_to_grade(overall),
            'components': {
                'retention': self.results.get('retention', {}).get('accuracy', 0),
                'reasoning': self.results.get('reasoning', {}).get('accuracy', 0),
                'coverage': self.results.get('coverage', {}).get('total_concepts', 0),
                'efficiency': self.results.get('efficiency', {}).get('speed', 0),
            },
        }

    def _score_to_grade(self, score: float) -> str:
        """分数转等级"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def generate_report(self, output_file: str = 'benchmark_report.txt'):
        """生成报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('AI学习系统基准测试报告\n')
            f.write('=' * 70 + '\n\n')

            # 总体分数
            overall = self.results.get('overall', {})
            f.write(f'总体分数: {overall.get("score", 0):.2f}\n')
            f.write(f'等级: {overall.get("grade", "N/A")}\n\n')

            # 各项分数
            f.write('各项分数:\n')
            components = overall.get('components', {})
            for k, v in components.items():
                f.write(f'  {k}: {v:.2f}\n')

            # 详细结果
            for test_name, test_result in self.results.items():
                if test_name == 'overall':
                    continue

                f.write(f'\n{"=" * 70}\n')
                f.write(f'{test_name}\n')
                f.write('=' * 70 + '\n')

                if isinstance(test_result, dict):
                    for k, v in test_result.items():
                        if k == 'details':
                            f.write(f'\n详细结果:\n')
                            for detail in v:
                                f.write(f'  问题: {detail.get("question", "")}\n')
                                f.write(f'  答案: {detail.get("answer", "")}\n')
                                f.write(f'  正确: {detail.get("correct", False)}\n\n')
                        else:
                            f.write(f'  {k}: {v}\n')

        print(f'报告已生成: {output_file}')


def main():
    benchmark = BenchmarkSuite()
    results = benchmark.run_all()
    benchmark.generate_report()

    # 打印摘要
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    overall = results.get('overall', {})
    print(f"总分: {overall.get('score', 0):.2f}")
    print(f"等级: {overall.get('grade', 'N/A')}")


if __name__ == '__main__':
    main()
