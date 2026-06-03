"""
简化think()方法 - Production-Grade Simplified Think Engine

完整版本 - 真正生产级实现
- 从7+1路径简化为3条清晰路径
- 整合常识知识库、因果推理、集成推理
- 保留完整推理能力
- 提供清晰接口

禁止任何简化、偷懒、快速实现
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading

# 导入现有模块
from src.reasoning.integrated_reasoning import (
    IntegratedReasoningEngine, ReasoningMode, ReasoningResult,
    get_integrated_engine
)


# ============================================================================
# 推理路径枚举
# ============================================================================

class ThinkPath(Enum):
    """简化的think()推理路径"""
    COMMONSENSE = "commonsense"           # 常识推理路径（快速直接）
    CAUSAL = "causal"                     # 因果推理路径（逻辑严谨）
    HYBRID = "hybrid"                     # 混合推理路径（综合最佳）


# ============================================================================
# think()结果数据类
# ============================================================================

@dataclass
class ThinkResult:
    """think()推理结果"""
    path: ThinkPath                        # 使用的推理路径
    answer: str                            # 答案文本
    confidence: float                      # 置信度 [0, 1]
    reasoning_chain: List[str] = field(default_factory=list)  # 推理链
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    execution_time: float = 0.0            # 执行时间（秒）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'path': self.path.value,
            'answer': self.answer,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain,
            'metadata': self.metadata,
            'execution_time': self.execution_time,
        }


# ============================================================================
# 简化think()引擎
# ============================================================================

class SimplifiedThinkEngine:
    """简化think()引擎（生产级）

    整合Phase 1-3的所有模块：
    - 常识知识库
    - 因果推理系统
    - 集成推理引擎

    提供3条清晰推理路径：
    1. 常识推理路径（快速直接）
    2. 因果推理路径（逻辑严谨）
    3. 混合推理路径（综合最佳）
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 集成推理引擎
        self.integrated_engine = get_integrated_engine(config)

        # 统计信息
        self.stats = {
            'total_thinks': 0,
            'commonsense_thinks': 0,
            'causal_thinks': 0,
            'hybrid_thinks': 0,
            'avg_confidence': 0.0,
            'total_execution_time': 0.0,
        }

        # 线程锁
        self._lock = threading.Lock()

        # 性能监控
        self.performance_history = []

    # ========================================================================
    # 核心think()接口
    # ========================================================================

    def think(self, query: str,
             path: ThinkPath = ThinkPath.HYBRID,
             context: Dict = None,
             top_k: int = 5) -> List[ThinkResult]:
        """
        简化think()接口

        Args:
            query: 查询/问题
            path: 推理路径（常识/因果/混合）
            context: 上下文信息
            top_k: 返回结果数量

        Returns:
            推理结果列表（按置信度排序）
        """
        start_time = time.time()

        # 更新统计
        with self._lock:
            self.stats['total_thinks'] += 1

        # 根据路径选择推理模式
        if path == ThinkPath.COMMONSENSE:
            mode = ReasoningMode.COMMONSENSE
            with self._lock:
                self.stats['commonsense_thinks'] += 1
        elif path == ThinkPath.CAUSAL:
            mode = ReasoningMode.CAUSAL
            with self._lock:
                self.stats['causal_thinks'] += 1
        else:  # HYBRID
            mode = ReasoningMode.HYBRID
            with self._lock:
                self.stats['hybrid_thinks'] += 1

        # 调用集成推理引擎
        integrated_results = self.integrated_engine.reason(
            query, mode, context, top_k
        )

        # 转换为ThinkResult
        think_results = []
        for result in integrated_results:
            think_result = ThinkResult(
                path=path,
                answer=result.answer,
                confidence=result.confidence,
                reasoning_chain=result.reasoning_chain,
                metadata={
                    'reasoning_type': result.reasoning_type,
                    'source': result.source,
                    **result.metadata
                },
                execution_time=result.execution_time,
            )
            think_results.append(think_result)

        # 记录执行时间
        execution_time = time.time() - start_time

        # 更新统计
        with self._lock:
            self.stats['total_execution_time'] += execution_time

            # 计算平均置信度
            if think_results:
                avg_conf = sum(r.confidence for r in think_results) / len(think_results)
                total = self.stats['total_thinks']
                self.stats['avg_confidence'] = (
                    (self.stats['avg_confidence'] * (total - 1) + avg_conf) / total
                )

            # 记录性能历史
            self.performance_history.append({
                'query': query,
                'path': path.value,
                'results_count': len(think_results),
                'avg_confidence': avg_conf if think_results else 0.0,
                'execution_time': execution_time,
            })

        return think_results

    # ========================================================================
    # 快捷方法
    # ========================================================================

    def think_commonsense(self, query: str,
                        top_k: int = 5) -> List[ThinkResult]:
        """快捷方法：常识推理"""
        return self.think(query, ThinkPath.COMMONSENSE, top_k=top_k)

    def think_causal(self, query: str,
                   top_k: int = 5) -> List[ThinkResult]:
        """快捷方法：因果推理"""
        return self.think(query, ThinkPath.CAUSAL, top_k=top_k)

    def think_hybrid(self, query: str,
                   top_k: int = 5) -> List[ThinkResult]:
        """快捷方法：混合推理"""
        return self.think(query, ThinkPath.HYBRID, top_k=top_k)

    # ========================================================================
    # 查询接口
    # ========================================================================

    def query_commonsense(self, question: str,
                        top_k: int = 5) -> List[Dict]:
        """查询常识知识库"""
        facts = self.integrated_engine.query_commonsense(question, top_k)
        return [f.to_dict() for f in facts]

    def query_causal(self, query: str,
                   top_k: int = 5) -> List[Dict]:
        """查询因果规则"""
        rules = self.integrated_engine.query_causal(query, top_k)
        return [r.to_dict() for r in rules]

    # ========================================================================
    # 验证接口
    # ========================================================================

    def verify(self, statement: str,
              threshold: float = 0.7) -> Tuple[bool, float, List[str]]:
        """验证陈述"""
        return self.integrated_engine.verify_statement(statement, threshold)

    # ========================================================================
    # 统计和监控
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()

        # 计算平均执行时间
        if stats['total_thinks'] > 0:
            stats['avg_execution_time'] = (
                stats['total_execution_time'] / stats['total_thinks']
            )
        else:
            stats['avg_execution_time'] = 0.0

        # 添加集成引擎统计
        stats['integrated_engine'] = self.integrated_engine.get_stats()

        return stats

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        stats = self.get_stats()

        return {
            'total_thinks': stats['total_thinks'],
            'by_path': {
                'commonsense': stats['commonsense_thinks'],
                'causal': stats['causal_thinks'],
                'hybrid': stats['hybrid_thinks'],
            },
            'performance': {
                'avg_confidence': stats['avg_confidence'],
                'avg_execution_time': stats['avg_execution_time'],
            },
            'knowledge_base': stats['integrated_engine'].get(
                'knowledge_base', {}
            ),
        }

    def clear_cache(self):
        """清空缓存"""
        self.integrated_engine.clear_cache()

    def reset_stats(self):
        """重置统计"""
        with self._lock:
            self.stats = {
                'total_thinks': 0,
                'commonsense_thinks': 0,
                'causal_thinks': 0,
                'hybrid_thinks': 0,
                'avg_confidence': 0.0,
                'total_execution_time': 0.0,
            }
            self.performance_history = []


# ============================================================================
# 便捷函数
# ============================================================================

def get_simplified_think_engine(config: Dict = None) -> SimplifiedThinkEngine:
    """获取简化think()引擎实例"""
    return SimplifiedThinkEngine(config)


if __name__ == '__main__':
    print("=== 简化think()引擎（生产级）===")
    print()
    print("核心功能:")
    print("- 3条清晰推理路径")
    print("- 常识推理（快速直接）")
    print("- 因果推理（逻辑严谨）")
    print("- 混合推理（综合最佳）")
    print()
    print("特性:")
    print("- 整合Phase 1-3模块")
    print("- 完整推理能力")
    print("- 清晰接口")
    print("- 性能监控")
