"""
集成推理引擎 - 统一推理入口

整合：
- UnifiedReasoningEngine（多模式推理）
- ReasoningEngine（任务分发）
- 常识推理
- 统一因果推理

提供单一推理接口，支持：
- 自动模式选择
- 多模式融合
- 常识优先
- 结果置信度排序
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field


@dataclass
class ReasoningResult:
    """推理结果"""
    answer: str
    confidence: float
    reasoning_type: str  # 'commonsense', 'causal', 'deductive', 'inductive', 'analogical', 'multi'
    source: str = ""  # 推理来源
    evidence: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class IntegratedReasoningEngine:
    """集成推理引擎 - 单一入口

    整合功能：
    - 任务分发（原ReasoningEngine）
    - 多模式推理（原UnifiedReasoningEngine）
    - 知识图谱查询
    - 常识推理
    - 统一因果推理
    """

    def __init__(self, learner):
        self.learner = learner

        # 推理子模块（延迟初始化）
        self._deductive = None
        self._inductive = None
        self._analogical = None
        self._causal = None
        self._unified_engine = None

        # 常识知识库
        self.commonsense = getattr(learner, 'commonsense_kb', None)

        # 统计信息
        self._query_count = 0
        self._mode_stats = {}

    def reason(self, question: str, mode: str = 'auto',
               context: dict = None) -> List[ReasoningResult]:
        """统一推理接口

        Args:
            question: 问题文本
            mode: 推理模式
                   - auto: 自动选择最佳推理模式
                   - commonsense: 常识推理
                   - causal: 因果推理
                   - deductive: 演绎推理
                   - inductive: 归纳推理
                   - analogical: 类比推理
                   - multi: 多模式融合
            context: 额外上下文

        Returns:
            推理结果列表（按置信度排序）
        """
        self._query_count += 1

        if mode == 'commonsense':
            return self._commonsense_reasoning(question)
        elif mode == 'causal':
            return self._causal_reasoning(question)
        elif mode == 'deductive':
            return self._deductive_reasoning(question)
        elif mode == 'inductive':
            return self._inductive_reasoning(question)
        elif mode == 'analogical':
            return self._analogical_reasoning(question)
        elif mode == 'multi':
            return self._multi_mode_reasoning(question, context)
        elif mode == 'auto':
            return self._auto_reason(question, context)
        else:
            return self._auto_reason(question, context)

    # ===================================================================
    # 推理模式
    # ===================================================================

    def _commonsense_reasoning(self, question: str) -> List[ReasoningResult]:
        """常识推理"""
        if not self.commonsense:
            return []

        facts = self.commonsense.query(question, top_k=5)

        results = []
        for fact in facts:
            results.append(ReasoningResult(
                answer=fact.statement,
                confidence=fact.confidence,
                reasoning_type='commonsense',
                source=fact.fact_id,
                metadata={'fact_type': fact.fact_type}
            ))

        self._update_mode_stats('commonsense', len(results))
        return results

    def _causal_reasoning(self, question: str) -> List[ReasoningResult]:
        """因果推理"""
        self._init_causal_if_needed()

        if not self._causal:
            return []

        causal_results = self._causal.query_causal(question, top_k=5)

        results = []
        for cr in causal_results:
            # 生成自然语言答案
            answer = self._format_causal_answer(cr)
            results.append(ReasoningResult(
                answer=answer,
                confidence=cr.confidence,
                reasoning_type='causal',
                source=cr.source,
                evidence=[cr.reasoning] if cr.reasoning else []
            ))

        self._update_mode_stats('causal', len(results))
        return results

    def _deductive_reasoning(self, question: str) -> List[ReasoningResult]:
        """演绎推理"""
        self._init_deductive_if_needed()

        if not self._deductive:
            return []

        try:
            # 使用演绎推理器
            results = self._deductive.deduce(question)
            return [ReasoningResult(
                answer=str(results),
                confidence=0.7,
                reasoning_type='deductive',
                source='deductive_engine'
            )]
        except Exception:
            return []

    def _inductive_reasoning(self, question: str) -> List[ReasoningResult]:
        """归纳推理"""
        self._init_inductive_if_needed()

        if not self._inductive:
            return []

        try:
            # 使用归纳推理器
            patterns = self._inductive.induce(question)
            return [ReasoningResult(
                answer=str(patterns),
                confidence=0.6,
                reasoning_type='inductive',
                source='inductive_engine'
            )]
        except Exception:
            return []

    def _analogical_reasoning(self, question: str) -> List[ReasoningResult]:
        """类比推理"""
        self._init_analogical_if_needed()

        if not self._analogical:
            return []

        try:
            # 使用类比推理器
            analogies = self._analogical.solve_analogy(question)
            return [ReasoningResult(
                answer=str(analogies),
                confidence=0.5,
                reasoning_type='analogical',
                source='analogical_engine'
            )]
        except Exception:
            return []

    def _multi_mode_reasoning(self, question: str, context: dict = None) -> List[ReasoningResult]:
        """多模式融合推理"""
        all_results = []

        # 尝试所有推理模式
        all_results.extend(self._commonsense_reasoning(question))
        all_results.extend(self._causal_reasoning(question))
        all_results.extend(self._deductive_reasoning(question))
        all_results.extend(self._inductive_reasoning(question))
        all_results.extend(self._analogical_reasoning(question))

        # 去重和融合
        unique_results = self._deduplicate_results(all_results)

        # 按置信度排序
        unique_results.sort(key=lambda r: r.confidence, reverse=True)

        return unique_results[:5]

    def _auto_reason(self, question: str, context: dict = None) -> List[ReasoningResult]:
        """自动选择最佳推理模式

        策略：
        1. 先尝试常识推理（最快）
        2. 再尝试因果推理
        3. 如果置信度不够，尝试多模式融合
        """
        # 快速路径：常识推理
        if self.commonsense:
            results = self._commonsense_reasoning(question)
            if results and results[0].confidence > 0.7:
                return results

        # 中等路径：因果推理
        results = self._causal_reasoning(question)
        if results and results[0].confidence > 0.5:
            return results

        # 完整路径：多模式融合
        return self._multi_mode_reasoning(question, context)

    # ===================================================================
    # 辅助方法
    # ===================================================================

    def _init_causal_if_needed(self):
        """延迟初始化因果推理器"""
        if self._causal is None:
            try:
                from src.reasoning.causal_unified import UnifiedCausalReasoner
                self._causal = UnifiedCausalReasoner()
            except Exception as e:
                print(f"[WARN] 因果推理器初始化失败: {e}")

    def _init_deductive_if_needed(self):
        """延迟初始化演绎推理器"""
        if self._deductive is None:
            try:
                from src.reasoning.deductive import DeductiveReasoner
                kg = getattr(self.learner, 'knowledge', None)
                self._deductive = DeductiveReasoner(kg)
            except Exception:
                pass

    def _init_inductive_if_needed(self):
        """延迟初始化归纳推理器"""
        if self._inductive is None:
            try:
                from src.reasoning.inductive import InductiveReasoner
                kg = getattr(self.learner, 'knowledge', None)
                self._inductive = InductiveReasoner(kg)
            except Exception:
                pass

    def _init_analogical_if_needed(self):
        """延迟初始化类比推理器"""
        if self._analogical is None:
            try:
                from src.reasoning.analogical import AnalogicalReasoner
                kg = getattr(self.learner, 'knowledge', None)
                self._analogical = AnalogicalReasoner(kg)
            except Exception:
                pass

    def _format_causal_answer(self, causal_result) -> str:
        """格式化因果推理结果为自然语言"""
        return f"{causal_result.cause}导致{causal_result.effect}"

    def _deduplicate_results(self, results: List[ReasoningResult]) -> List[ReasoningResult]:
        """去重推理结果"""
        seen = set()
        unique = []

        for result in results:
            # 使用答案作为去重键
            key = (result.answer, result.reasoning_type)
            if key not in seen:
                seen.add(key)
                unique.append(result)

        return unique

    def _update_mode_stats(self, mode: str, count: int):
        """更新模式统计"""
        if mode not in self._mode_stats:
            self._mode_stats[mode] = 0
        self._mode_stats[mode] += count

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'query_count': self._query_count,
            'mode_stats': self._mode_stats.copy(),
            'commonsense_available': self.commonsense is not None,
            'causal_available': self._causal is not None,
        }


if __name__ == '__main__':
    print("=== 集成推理引擎 ===")
    print()
    print("统一推理接口:")
    print("- commonsense: 常识推理")
    print("- causal: 因果推理")
    print("- deductive: 演绎推理")
    print("- inductive: 归纳推理")
    print("- analogical: 类比推理")
    print("- multi: 多模式融合")
    print("- auto: 自动选择")
    print()
    print("核心特性:")
    print("- 单一入口")
    print("- 自动模式选择")
    print("- 结果置信度排序")
    print("- 延迟初始化")
