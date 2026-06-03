"""完整知识学习脚本 — 使用获取的真实数据

运行方式：
    python training/run_full_knowledge_learning.py

数据来源：
- WordNet 语义关系（2390 词）
- Free Dictionary API 定义（1224 词）
- 自动生成的搭配和例句

学习目标：
1. 学习所有 2461 个英语词汇
2. 掌握语义关系网络
3. 理解词义和用法
4. 达到 CEFR C2 水平
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import LearnerConfig
from src.data.source import (
    EnglishDataSource,
    MathDataSource,
    PhysicsDataSource,
    CSDataSource,
    ChemistryDataSource,
)
from src.learning.universal_system import UniversalLearningSystem


def load_acquired_data():
    """加载获取的知识数据"""
    data_dir = Path('data/knowledge')

    data = {}
    for name in ['semantic_relations', 'collocations', 'examples']:
        path = data_dir / f'{name}.json'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data[name] = json.load(f)
            print(f"  加载 {name}: {len(data[name])} 词")

    return data


def main():
    print("=" * 70)
    print("完整知识学习 — 使用 WordNet + Dictionary API 真实数据")
    print("=" * 70)

    # ── 加载获取的数据 ──────────────────────────────────────────────
    print("\n[1] 加载获取的知识数据...")
    acquired = load_acquired_data()

    # ── 配置 ──────────────────────────────────────────────
    config = LearnerConfig(
        obs_dim=40,
        action_dim=8,
        hidden_dims=(128, 64),
        learning_rate=0.001,
        initial_stage='sensorimotor',
    )

    # ── 初始化系统 ──────────────────────────────────────────────
    system = UniversalLearningSystem(config=config)

    # ── 添加数据源 ──────────────────────────────────────────────
    print("\n[2] 添加数据源...")
    system.add_data_source(EnglishDataSource(max_words=3000, use_api=False))
    system.add_data_source(MathDataSource())
    system.add_data_source(PhysicsDataSource())
    system.add_data_source(CSDataSource())
    system.add_data_source(ChemistryDataSource())

    # ── 加载数据 ──────────────────────────────────────────────
    print("\n[3] 加载知识单元...")
    counts = system.load_all_sources()
    for domain, count in counts.items():
        print(f"  {domain}: {count} 个单元")
    total = sum(counts.values())
    print(f"  总计: {total} 个单元")

    # ── 初始评估 ──────────────────────────────────────────────
    print("\n[4] 初始评估...")
    initial = system.assess_proficiency('english')
    print(f"  初始 CEFR: {initial['level']}")
    print(f"  初始分数: {initial['score']}")

    # ── 学习阶段 ──────────────────────────────────────────────
    print("\n[5] 开始学习...")
    n_epochs = 500
    learn_per_epoch = 20
    review_per_epoch = 10

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        result = system.run_learning_cycle(
            n_learn=learn_per_epoch,
            n_review=review_per_epoch,
        )

        if epoch % 100 == 0 or epoch == 1:
            proficiency = system.assess_proficiency('english')
            schedule = system.get_schedule_stats()

            print(f"\n  [Epoch {epoch}/{n_epochs}]")
            print(f"    CEFR: {proficiency['level']}")
            print(f"    分数: {proficiency['score']:.4f}")
            print(f"    词汇深度: {proficiency['semantic_depth']:.4f}")
            print(f"    搭配知识: {proficiency['collocation_knowledge']:.4f}")
            print(f"    词族覆盖: {proficiency['word_family_coverage']:.4f}")
            print(f"    记忆保持: {schedule['retention_rate']:.2%}")

    elapsed = time.time() - start_time

    # ── 最终评估 ──────────────────────────────────────────────
    print("\n[6] 最终评估...")

    # CEFR 评估
    proficiency = system.assess_proficiency('english')
    print(f"\n  CEFR 专业水平:")
    print(f"    等级: {proficiency['level']}")
    print(f"    总分: {proficiency['score']:.4f}")
    print(f"    接收词汇: {proficiency['receptive_vocabulary']}")
    print(f"    产出词汇: {proficiency['productive_vocabulary']}")
    print(f"    语义深度: {proficiency['semantic_depth']:.4f}")
    print(f"    搭配知识: {proficiency['collocation_knowledge']:.4f}")
    print(f"    词族覆盖: {proficiency['word_family_coverage']:.4f}")

    # 维度详情
    print(f"    维度:")
    for dim, score in proficiency.get('dimensions', {}).items():
        print(f"      {dim}: {score:.4f}")

    # 掌握度评估
    final = system.assess_all()
    print(f"\n  掌握度评估:")
    print(f"    总单元数: {final['total_units']}")
    print(f"    已精通: {final['total_mastered']}")
    print(f"    精通率: {final['mastery_rate']:.2%}")

    for domain, stats in final.get('domains', {}).items():
        print(f"\n    {domain}:")
        print(f"      总单元: {stats['total_units']}")
        print(f"      精通: {stats['level_distribution'].get('MASTERED', 0)}")
        print(f"      平均掌握度: {stats['avg_mastery']:.3f}")
        print(f"      覆盖率: {stats['coverage']:.2%}")

    # 间隔重复统计
    schedule = system.get_schedule_stats()
    print(f"\n  间隔重复统计:")
    print(f"    总调度: {schedule['total']}")
    print(f"    平均间隔: {schedule['avg_interval']:.1f} 天")
    print(f"    记忆保持率: {schedule['retention_rate']:.2%}")

    # ── 保存检查点 ──────────────────────────────────────────────
    print("\n[7] 保存检查点...")
    checkpoint_path = system.save_checkpoint('full_knowledge')
    print(f"  保存到: {checkpoint_path}")

    # ── 统计 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("完整知识学习完成!")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  训练轮数: {n_epochs}")
    print(f"  总学习单元: {n_epochs * learn_per_epoch}")
    print(f"  总复习单元: {n_epochs * review_per_epoch}")
    print(f"  最终 CEFR: {proficiency['level']}")
    print(f"  最终分数: {proficiency['score']:.4f}")
    print(f"  接收词汇: {proficiency['receptive_vocabulary']}")
    print(f"  产出词汇: {proficiency['productive_vocabulary']}")
    print(f"  记忆保持率: {schedule['retention_rate']:.2%}")
    print("=" * 70)


if __name__ == '__main__':
    main()
