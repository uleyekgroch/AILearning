"""通用学习系统训练脚本 — 完整牛津辞典学习

运行方式：
    python training/run_universal_learning.py

功能：
1. 初始化通用学习系统
2. 加载牛津 3000 核心词汇 + 其他领域知识
3. 运行主动学习循环（间隔重复调度）
4. 定期评估 CEFR 水平和掌握度
5. 输出最终架构和学习报告
"""

import sys
import os
import time

# 添加项目根目录到路径
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


def main():
    print("=" * 70)
    print("通用学习系统 — 完整牛津辞典学习")
    print("=" * 70)

    # ── 配置 ──────────────────────────────────────────────────
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
    print("\n[1] 添加数据源...")
    # 英语：加载完整牛津词汇（不使用 API，使用内置数据）
    system.add_data_source(EnglishDataSource(max_words=3000, use_api=False))
    # 其他领域
    system.add_data_source(MathDataSource())
    system.add_data_source(PhysicsDataSource())
    system.add_data_source(CSDataSource())
    system.add_data_source(ChemistryDataSource())

    # ── 加载数据 ──────────────────────────────────────────────
    print("\n[2] 加载知识单元...")
    counts = system.load_all_sources()
    for domain, count in counts.items():
        print(f"  {domain}: {count} 个单元")
    total = sum(counts.values())
    print(f"  总计: {total} 个单元")

    # ── 初始评估 ──────────────────────────────────────────────
    print("\n[3] 初始评估...")
    initial_assessment = system.assess()
    print(f"  总单元数: {initial_assessment['total_units']}")
    print(f"  已精通: {initial_assessment['total_mastered']}")
    print(f"  精通率: {initial_assessment['mastery_rate']:.2%}")

    # 初始 CEFR 评估
    proficiency = system.assess_proficiency('english')
    print(f"  初始 CEFR: {proficiency['level']}")
    print(f"  接收词汇: {proficiency['receptive_vocabulary']}")
    print(f"  产出词汇: {proficiency['productive_vocabulary']}")

    # ── 学习阶段 ──────────────────────────────────────────────
    print("\n[4] 开始学习...")
    n_epochs = 500
    learn_per_epoch = 20
    review_per_epoch = 10

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        # 运行学习周期
        result = system.run_learning_cycle(
            n_learn=learn_per_epoch,
            n_review=review_per_epoch,
        )

        # 打印进度
        if epoch % 20 == 0 or epoch == 1:
            assessment = result['assessment']
            schedule_stats = system.get_schedule_stats()

            print(f"\n  [Epoch {epoch}/{n_epochs}]")
            print(f"    学习: {result['learn_count']} 个单元")
            print(f"    复习: {result['review_count']} 个单元")
            print(f"    学习者水平: {system.current_level:.3f}")

            # 间隔重复统计
            print(f"    调度统计:")
            print(f"      总调度: {schedule_stats['total']}")
            print(f"      到期: {schedule_stats['due']}")
            print(f"      过期: {schedule_stats['overdue']}")
            print(f"      未来 7 天: {schedule_stats['upcoming_7d']}")
            print(f"      平均间隔: {schedule_stats['avg_interval']:.1f} 天")
            print(f"      记忆保持率: {schedule_stats['retention_rate']:.2%}")

            # 领域详情
            for domain, stats in assessment.get('domains', {}).items():
                print(f"    {domain}: "
                      f"精通 {stats['level_distribution'].get('MASTERED', 0)}/"
                      f"{stats['total_units']}, "
                      f"平均掌握度 {stats['avg_mastery']:.3f}")

    elapsed = time.time() - start_time

    # ── 最终评估 ──────────────────────────────────────────────
    print("\n[5] 最终评估...")

    # 掌握度评估
    final_assessment = system.assess_all()
    print(f"\n  掌握度评估:")
    print(f"    总单元数: {final_assessment['total_units']}")
    print(f"    已精通: {final_assessment['total_mastered']}")
    print(f"    精通率: {final_assessment['mastery_rate']:.2%}")

    for domain, stats in final_assessment.get('domains', {}).items():
        print(f"\n    {domain}:")
        print(f"      总单元: {stats['total_units']}")
        print(f"      精通: {stats['level_distribution'].get('MASTERED', 0)}")
        print(f"      平均掌握度: {stats['avg_mastery']:.3f}")
        print(f"      覆盖率: {stats['coverage']:.2%}")

        # 最强单元
        if stats.get('strongest_units'):
            print(f"      最强单元:")
            for u in stats['strongest_units'][:3]:
                print(f"        {u['name']}: {u['mastery']:.3f}")

    # CEFR 专业水平评估
    print("\n  CEFR 专业水平评估:")
    proficiency = system.assess_proficiency('english')
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

    # 等级分布
    print(f"    等级分布:")
    for level, count in proficiency.get('level_distribution', {}).items():
        print(f"      {level}: {count} 词")

    # 建议
    if proficiency.get('recommendations'):
        print(f"    建议:")
        for rec in proficiency['recommendations']:
            print(f"      - {rec}")

    # 间隔重复统计
    print("\n  间隔重复统计:")
    schedule_stats = system.get_schedule_stats()
    print(f"    总调度: {schedule_stats['total']}")
    print(f"    到期: {schedule_stats['due']}")
    print(f"    过期: {schedule_stats['overdue']}")
    print(f"    未来 7 天: {schedule_stats['upcoming_7d']}")
    print(f"    平均间隔: {schedule_stats['avg_interval']:.1f} 天")
    print(f"    平均难度因子: {schedule_stats['avg_ease']:.4f}")
    print(f"    记忆保持率: {schedule_stats['retention_rate']:.2%}")

    # ── 优先级报告 ──────────────────────────────────────────────
    print("\n[6] 下一步学习优先级 (Top 10)...")
    priority = system.get_priority_report(top_n=10)
    for i, item in enumerate(priority, 1):
        print(f"  {i}. {item['name']} ({item['domain']}) "
              f"score={item['score']:.3f} mastery={item['mastery']:.3f}")

    # ── 保存检查点 ──────────────────────────────────────────────
    print("\n[7] 保存检查点...")
    checkpoint_path = system.save_checkpoint('final')
    print(f"  保存到: {checkpoint_path}")

    # ── 最终架构输出 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("最终架构")
    print("=" * 70)
    print("""
+-------------------------------------------------------------------+
|                 UniversalLearningSystem (编排器)                    |
|                                                                   |
|  +-----------+  +-----------+  +-----------+  +-----------+      |
|  | 数据层     |  | 选择层     |  | 学习层     |  | 评估层     |      |
|  |           |  |           |  |           |  |           |      |
|  | DataSource|  | Active    |  | Universal |  | Mastery   |      |
|  | (5 领域)  |->| Selector  |->| Learner   |->| Assessor  |      |
|  |           |  |           |  |           |  |           |      |
|  +-----------+  +-----------+  +-----------+  +-----------+      |
|        |              |              |              |              |
|        +--------------+--------------+--------------+              |
|                                   |                                |
|  +------------------------------------------------------------+  |
|  |              现有系统（复用）                                |  |
|  |  Learner | PredictiveCodingEngine | MemorySystem           |  |
|  |  KnowledgeGraph | IntrinsicMotivation | MetaAssessor       |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |              新增组件                                      |  |
|  |  SpacedRepetitionScheduler (SM-2 间隔重复)                  |  |
|  |  ProficiencyTester (CEFR A1-C2 专业评估)                   |  |
|  |  OxfordWords (牛津 3000 词汇 + 词族 + 词根词缀)             |  |
|  +------------------------------------------------------------+  |
+-------------------------------------------------------------------+
""")

    # ── 统计 ──────────────────────────────────────────────────
    print("=" * 70)
    print("训练完成!")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  训练轮数: {n_epochs}")
    print(f"  总学习单元: {n_epochs * learn_per_epoch}")
    print(f"  总复习单元: {n_epochs * review_per_epoch}")
    print(f"  最终 CEFR: {proficiency['level']}")
    print(f"  接收词汇: {proficiency['receptive_vocabulary']}")
    print(f"  产出词汇: {proficiency['productive_vocabulary']}")
    print(f"  记忆保持率: {schedule_stats['retention_rate']:.2%}")
    print("=" * 70)


if __name__ == '__main__':
    main()
