"""通用学习系统编排器 — 协调数据源、选择器、学习器、评估器

这是通用学习架构的顶层组件。
它协调所有子系统，实现完整的主动学习循环：

1. 加载：从数据源加载知识单元
2. 选择：主动选择最有学习价值的单元
3. 学习：通过感知-预测-学习闭环学习
4. 评估：评估掌握程度
5. 调度：基于遗忘曲线安排复习
6. 报告：生成学习进度报告

支持多领域并行学习（英语、数学、物理、CS、化学等）。
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import LearnerConfig
from src.core.learner import Learner
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.unit import KnowledgeUnit, MasteryLevel
from src.data.source import DataSource
from src.learning.selector import ActiveSelector
from src.learning.universal_learner import UniversalLearner
from src.learning.spaced_repetition import SpacedRepetitionScheduler
from src.assessment.mastery import MasteryAssessor
from src.assessment.proficiency import ProficiencyTester


class LearningSession:
    """单次学习会话的结果"""

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.units_learned: List[Dict] = []
        self.units_reviewed: List[Dict] = []
        self.total_error: float = 0.0
        self.total_practices: int = 0

    def finish(self):
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'duration': round(self.duration, 2),
            'units_learned': len(self.units_learned),
            'units_reviewed': len(self.units_reviewed),
            'total_error': round(self.total_error, 6),
            'total_practices': self.total_practices,
            'learned_details': self.units_learned,
            'reviewed_details': self.units_reviewed,
        }


class UniversalLearningSystem:
    """通用学习系统 — 编排器

    协调数据源、选择器、学习器、评估器，
    实现完整的主动学习循环。
    """

    def __init__(self,
                 config: Optional[LearnerConfig] = None,
                 save_dir: str = 'checkpoints/universal'):
        """
        Args:
            config: Learner 配置
            save_dir: 检查点保存目录
        """
        self.config = config or LearnerConfig()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 核心组件
        self.knowledge_graph = KnowledgeGraph()
        self.learner = Learner(self.config)
        self.universal_learner = UniversalLearner(self.learner, self.knowledge_graph)
        self.selector = ActiveSelector(epsilon=0.1)
        self.assessor = MasteryAssessor()
        self.scheduler = SpacedRepetitionScheduler()
        self.proficiency_tester = ProficiencyTester()

        # 数据源
        self.data_sources: Dict[str, DataSource] = {}

        # 知识单元库
        self.units: Dict[str, KnowledgeUnit] = {}  # id -> unit
        self.units_by_domain: Dict[str, List[KnowledgeUnit]] = {}  # domain -> [unit]

        # 学习历史
        self.sessions: List[LearningSession] = []
        self.current_session: Optional[LearningSession] = None
        self._session_counter = 0

        # 当前学习者水平
        self.current_level = 0.3  # 从较低水平开始

    # ── 数据源管理 ──────────────────────────────────────────────

    def add_data_source(self, source: DataSource) -> None:
        """添加数据源

        Args:
            source: 数据源实例
        """
        domain = source.get_domain()
        self.data_sources[domain] = source

    def load_all_sources(self) -> Dict[str, int]:
        """从所有数据源加载知识单元

        Returns:
            每个领域加载的单元数量
        """
        counts = {}
        for domain, source in self.data_sources.items():
            units = source.load_units()
            self.units_by_domain[domain] = units
            for unit in units:
                self.units[unit.id] = unit
            counts[domain] = len(units)
        return counts

    def load_source(self, domain: str) -> int:
        """从指定数据源加载知识单元

        Args:
            domain: 领域名称

        Returns:
            加载的单元数量
        """
        if domain not in self.data_sources:
            return 0

        source = self.data_sources[domain]
        units = source.load_units()
        self.units_by_domain[domain] = units
        for unit in units:
            self.units[unit.id] = unit
        return len(units)

    # ── 学习循环 ──────────────────────────────────────────────

    def start_session(self) -> LearningSession:
        """开始新的学习会话"""
        self._session_counter += 1
        self.current_session = LearningSession(self._session_counter)
        return self.current_session

    def learn_step(self, domain: Optional[str] = None) -> Optional[Dict]:
        """执行一步学习

        1. 选择最有学习价值的知识单元
        2. 学习该单元
        3. 更新掌握度

        Args:
            domain: 指定领域，None 表示跨领域选择

        Returns:
            学习结果，或 None（无可用单元）
        """
        # 获取候选单元
        candidates = self._get_candidates(domain)
        if not candidates:
            return None

        # 主动选择
        selected = self.selector.select_next(
            candidates,
            all_units=self.units,
            current_level=self.current_level,
        )

        if selected is None:
            return None

        # 学习
        result = self.universal_learner.learn_unit(selected, n_practice=3)

        # 更新调度器
        self.scheduler.update_from_unit(selected)

        # 记录
        if self.current_session:
            self.current_session.units_learned.append(result)
            self.current_session.total_error += result.get('phases', [{}])[-1].get('practices', [{}])[-1].get('error', 0) if result.get('phases') else 0
            self.current_session.total_practices += 3

        # 更新学习者水平
        self._update_level()

        return result

    def learn_batch(self, batch_size: int = 10,
                    domain: Optional[str] = None) -> List[Dict]:
        """执行一批学习

        Args:
            batch_size: 批次大小
            domain: 指定领域

        Returns:
            学习结果列表
        """
        results = []
        for _ in range(batch_size):
            result = self.learn_step(domain)
            if result:
                results.append(result)
            else:
                break
        return results

    def review_step(self, domain: Optional[str] = None) -> Optional[Dict]:
        """执行一步复习

        选择最需要复习的知识单元（基于间隔重复调度器）。

        Args:
            domain: 指定领域

        Returns:
            复习结果
        """
        # 优先使用调度器选择到期复习
        due_reviews = self.scheduler.get_due_reviews(limit=5)
        if due_reviews:
            # 过滤领域
            if domain:
                due_reviews = [e for e in due_reviews
                               if self.units.get(e.unit_id, None)
                               and self.units[e.unit_id].domain == domain]

            if due_reviews:
                selected_id = due_reviews[0].unit_id
                selected = self.units.get(selected_id)
                if selected:
                    result = self.universal_learner.review_unit(selected)
                    # 更新调度器
                    self.scheduler.update_from_unit(selected)
                    if self.current_session:
                        self.current_session.units_reviewed.append(result)
                    return result

        # 回退：使用传统方法选择复习单元
        review_candidates = self._get_review_candidates(domain)
        if not review_candidates:
            return None

        selected = min(review_candidates, key=lambda u: u.mastery)
        result = self.universal_learner.review_unit(selected)

        # 更新调度器
        self.scheduler.update_from_unit(selected)

        if self.current_session:
            self.current_session.units_reviewed.append(result)

        return result

    def run_learning_cycle(self, n_learn: int = 10, n_review: int = 5,
                           domain: Optional[str] = None) -> Dict:
        """运行完整学习周期

        1. 学习 n_learn 个新单元
        2. 复习 n_review 个旧单元
        3. 评估进度

        Args:
            n_learn: 学习数量
            n_review: 复习数量
            domain: 指定领域

        Returns:
            周期报告
        """
        session = self.start_session()

        # 学习
        learn_results = self.learn_batch(n_learn, domain)

        # 复习
        review_results = []
        for _ in range(n_review):
            result = self.review_step(domain)
            if result:
                review_results.append(result)

        session.finish()

        # 评估
        assessment = self.assess(domain)

        return {
            'session': session.to_dict(),
            'learn_count': len(learn_results),
            'review_count': len(review_results),
            'assessment': assessment,
        }

    def run_full_training(self, epochs: int = 100,
                          learn_per_epoch: int = 10,
                          review_per_epoch: int = 5,
                          domain: Optional[str] = None,
                          log_interval: int = 10) -> Dict:
        """运行完整训练

        Args:
            epochs: 训练轮数
            learn_per_epoch: 每轮学习数量
            review_per_epoch: 每轮复习数量
            domain: 指定领域
            log_interval: 日志间隔

        Returns:
            训练报告
        """
        history = []

        for epoch in range(1, epochs + 1):
            cycle_result = self.run_learning_cycle(
                n_learn=learn_per_epoch,
                n_review=review_per_epoch,
                domain=domain,
            )
            cycle_result['epoch'] = epoch
            history.append(cycle_result)

            if epoch % log_interval == 0:
                self._log_progress(epoch, epochs, cycle_result)

        # 最终评估
        final_assessment = self.assess_all()

        return {
            'epochs': epochs,
            'history': history,
            'final_assessment': final_assessment,
        }

    # ── 评估 ──────────────────────────────────────────────────

    def assess(self, domain: Optional[str] = None) -> Dict:
        """评估掌握程度

        Args:
            domain: 指定领域，None 表示评估所有领域

        Returns:
            评估报告
        """
        if domain:
            units = self.units_by_domain.get(domain, [])
            return self.assessor.assess_domain(units)
        else:
            return self.assessor.assess_all_domains(self.units_by_domain)

    def assess_unit(self, unit_id: str) -> Optional[Dict]:
        """评估单个知识单元"""
        unit = self.units.get(unit_id)
        if not unit:
            return None
        result = self.assessor.assess(unit)
        return {
            'unit_id': result.unit_id,
            'unit_name': result.unit_name,
            'domain': result.domain,
            'overall_mastery': result.overall_mastery,
            'mastery_level': result.mastery_level.name,
            'dimensions': result.dimensions,
            'strengths': result.strengths,
            'weaknesses': result.weaknesses,
            'recommendations': result.recommendations,
        }

    def assess_all(self) -> Dict:
        """评估所有领域"""
        return self.assessor.assess_all_domains(self.units_by_domain)

    def assess_proficiency(self, domain: str = 'english') -> Dict:
        """评估专业水平（CEFR 等级）

        Args:
            domain: 领域名称（默认英语）

        Returns:
            专业水平评估报告
        """
        units = self.units_by_domain.get(domain, [])
        if not units:
            return {'level': 'A1', 'score': 0.0}

        report = self.proficiency_tester.assess(units)

        return {
            'level': report.level,
            'score': report.score,
            'receptive_vocabulary': report.receptive_vocabulary,
            'productive_vocabulary': report.productive_vocabulary,
            'semantic_depth': report.semantic_depth,
            'collocation_knowledge': report.collocation_knowledge,
            'word_family_coverage': report.word_family_coverage,
            'dimensions': report.dimensions,
            'recommendations': report.recommendations,
            'level_distribution': report.level_distribution,
        }

    def get_schedule_stats(self) -> Dict:
        """获取间隔重复调度统计"""
        return self.scheduler.get_schedule_stats()

    def get_due_reviews(self, limit: int = 10) -> List[Dict]:
        """获取到期的复习"""
        due = self.scheduler.get_due_reviews(limit)
        results = []
        for entry in due:
            unit = self.units.get(entry.unit_id)
            if unit:
                results.append({
                    'unit_id': entry.unit_id,
                    'unit_name': unit.name,
                    'domain': unit.domain,
                    'interval': entry.interval,
                    'ease_factor': entry.ease_factor,
                    'next_review': entry.next_review,
                })
        return results

    # ── 查询 ──────────────────────────────────────────────────

    def get_priority_report(self, domain: Optional[str] = None,
                            top_n: int = 10) -> List[Dict]:
        """获取优先级报告

        Args:
            domain: 指定领域
            top_n: 返回前 N 个

        Returns:
            按优先级排序的知识单元列表
        """
        candidates = self._get_candidates(domain)
        return self.selector.get_priority_report(
            candidates,
            all_units=self.units,
            current_level=self.current_level,
            top_n=top_n,
        )

    def get_domain_summary(self) -> Dict:
        """获取领域摘要"""
        summary = {}
        for domain, units in self.units_by_domain.items():
            total = len(units)
            mastered = sum(1 for u in units if u.mastery_level == MasteryLevel.MASTERED)
            avg_mastery = sum(u.mastery for u in units) / total if total > 0 else 0

            summary[domain] = {
                'total_units': total,
                'mastered': mastered,
                'mastery_rate': round(mastered / total, 4) if total > 0 else 0,
                'avg_mastery': round(avg_mastery, 4),
            }

        return summary

    def get_unit(self, unit_id: str) -> Optional[KnowledgeUnit]:
        """获取知识单元"""
        return self.units.get(unit_id)

    def get_units_by_status(self, status: MasteryLevel,
                            domain: Optional[str] = None) -> List[KnowledgeUnit]:
        """按状态获取知识单元"""
        if domain:
            units = self.units_by_domain.get(domain, [])
        else:
            units = list(self.units.values())

        return [u for u in units if u.mastery_level == status]

    # ── 持久化 ──────────────────────────────────────────────

    def save_checkpoint(self, name: str = 'latest') -> str:
        """保存检查点

        Args:
            name: 检查点名称

        Returns:
            保存路径
        """
        checkpoint = {
            'units': {uid: u.to_dict() for uid, u in self.units.items()},
            'current_level': self.current_level,
            'session_counter': self._session_counter,
            'timestamp': time.time(),
        }

        path = self.save_dir / f'{name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        return str(path)

    def load_checkpoint(self, name: str = 'latest') -> bool:
        """加载检查点

        Args:
            name: 检查点名称

        Returns:
            是否成功
        """
        path = self.save_dir / f'{name}.json'
        if not path.exists():
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            # 恢复知识单元
            self.units = {}
            self.units_by_domain = {}
            for uid, unit_dict in checkpoint.get('units', {}).items():
                unit = KnowledgeUnit.from_dict(unit_dict)
                self.units[uid] = unit
                domain = unit.domain
                if domain not in self.units_by_domain:
                    self.units_by_domain[domain] = []
                self.units_by_domain[domain].append(unit)

            self.current_level = checkpoint.get('current_level', 0.3)
            self._session_counter = checkpoint.get('session_counter', 0)

            return True
        except Exception:
            return False

    # ── 内部方法 ──────────────────────────────────────────────

    def _get_candidates(self, domain: Optional[str] = None) -> List[KnowledgeUnit]:
        """获取候选知识单元"""
        if domain:
            return self.units_by_domain.get(domain, [])
        else:
            return list(self.units.values())

    def _get_review_candidates(self, domain: Optional[str] = None) -> List[KnowledgeUnit]:
        """获取需要复习的候选

        选择已学习但未精通的单元，按掌握度排序。
        """
        if domain:
            units = self.units_by_domain.get(domain, [])
        else:
            units = list(self.units.values())

        # 已学习但未精通
        reviewable = [
            u for u in units
            if u.mastery_level.value >= MasteryLevel.EXPOSED.value
            and u.mastery_level.value < MasteryLevel.MASTERED.value
        ]

        # 按掌握度排序（低的优先复习）
        reviewable.sort(key=lambda u: u.mastery)
        return reviewable

    def _update_level(self) -> None:
        """更新学习者水平

        基于所有已学习单元的平均掌握度。
        """
        learned = [u for u in self.units.values() if u.mastery > 0]
        if not learned:
            return

        avg_mastery = sum(u.mastery for u in learned) / len(learned)
        # 平滑更新
        self.current_level = 0.9 * self.current_level + 0.1 * avg_mastery

    def _log_progress(self, epoch: int, total: int, result: Dict) -> None:
        """打印进度"""
        assessment = result.get('assessment', {})
        learned = result.get('learn_count', 0)
        reviewed = result.get('review_count', 0)

        print(f"[Epoch {epoch}/{total}] "
              f"learned={learned}, reviewed={reviewed}, "
              f"level={self.current_level:.3f}")
