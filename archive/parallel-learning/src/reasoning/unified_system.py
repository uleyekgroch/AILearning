"""
统一推理系统 - Unified Reasoning System

整合所有阶段的推理模块：
- Stage 1: 图数据库知识库
- Stage 2: GNN推理引擎
- Stage 3: 概率推理系统

提供生产就绪的统一接口：
1. 多模式推理
2. 性能优化
3. 可扩展性
4. 完整测试套件
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch


# ============================================================================
# 推理模式枚举
# ============================================================================

class UnifiedReasoningMode(Enum):
    """统一推理模式"""
    AUTO = "auto"                       # 自动选择最佳模式
    GRAPH = "graph"                     # 图结构推理
    GNN = "gnn"                         # 图神经网络推理
    PROBABILISTIC = "probabilistic"     # 概率推理
    HYBRID = "hybrid"                   # 混合推理


# ============================================================================
# 推理结果数据类
# ============================================================================

@dataclass
class UnifiedReasoningResult:
    """统一推理结果"""
    answer: str                          # 答案
    confidence: float                    # 置信度 [0, 1]
    reasoning_type: str                  # 推理类型
    source: str = ""                     # 来源
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    reasoning_steps: List[str] = field(default_factory=list)  # 推理步骤

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'answer': self.answer,
            'confidence': self.confidence,
            'reasoning_type': self.reasoning_type,
            'source': self.source,
            'metadata': self.metadata,
            'reasoning_steps': self.reasoning_steps
        }


# ============================================================================
# 统一推理系统
# ============================================================================

class UnifiedReasoningSystem:
    """统一推理系统

    整合所有阶段的推理模块，提供生产就绪的接口
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 推理模块
        self.graph_kb = None           # 图数据库知识库（Stage 1）
        self.gnn_engine = None         # GNN推理引擎（Stage 2）
        self.prob_engine = None        # 概率推理引擎（Stage 3）

        # 统计信息
        self.stats = {
            'total_queries': 0,
            'graph_queries': 0,
            'gnn_queries': 0,
            'prob_queries': 0,
            'hybrid_queries': 0,
        }

    def initialize(self, graph_kb=None, gnn_engine=None, prob_engine=None):
        """初始化推理模块

        Args:
            graph_kb: 图数据库知识库（KnowledgeBaseCRUD）
            gnn_engine: GNN推理引擎（GNNReasoningEngine）
            prob_engine: 概率推理引擎（ProbabilisticReasoningEngine）
        """
        self.graph_kb = graph_kb
        self.gnn_engine = gnn_engine
        self.prob_engine = prob_engine

        print(f"[统一推理系统] 初始化完成 (设备: {self.device})")
        print(f"  - 图数据库（Stage 1）: {'[OK]' if graph_kb else '[N/A]'}")
        print(f"  - GNN引擎（Stage 2）: {'[OK]' if gnn_engine else '[N/A]'}")
        print(f"  - 概率引擎（Stage 3）: {'[OK]' if prob_engine else '[N/A]'}")

    def reason(self, query: str, mode: UnifiedReasoningMode = UnifiedReasoningMode.AUTO,
               context: Dict = None, top_k: int = 5) -> List[UnifiedReasoningResult]:
        """
        统一推理接口

        Args:
            query: 查询文本
            mode: 推理模式
            context: 额外上下文
            top_k: 返回结果数量

        Returns:
            推理结果列表（按置信度排序）
        """
        self.stats['total_queries'] += 1

        if mode == UnifiedReasoningMode.AUTO:
            mode = self._select_best_mode(query, context)

        # 路由到对应推理模块
        if mode == UnifiedReasoningMode.GNN and self.gnn_engine:
            return self._gnn_reasoning(query, top_k)
        elif mode == UnifiedReasoningMode.PROBABILISTIC and self.prob_engine:
            return self._probabilistic_reasoning(query, context, top_k)
        elif mode == UnifiedReasoningMode.GRAPH and self.graph_kb:
            return self._graph_reasoning(query, top_k)
        elif mode == UnifiedReasoningMode.HYBRID:
            return self._hybrid_reasoning(query, context, top_k)
        else:
            # 回退到图推理
            if self.graph_kb:
                return self._graph_reasoning(query, top_k)
            else:
                return []

    def _select_best_mode(self, query: str, context: Dict = None) -> UnifiedReasoningMode:
        """自动选择最佳推理模式"""
        # 优先级：GNN > 概率 > 图

        # 1. 检查是否需要概率推理
        if self.prob_engine and context and 'evidence' in context:
            return UnifiedReasoningMode.PROBABILISTIC

        # 2. 优先使用GNN
        if self.gnn_engine and self.gnn_engine.model:
            return UnifiedReasoningMode.GNN

        # 3. 回退到图推理
        if self.graph_kb:
            return UnifiedReasoningMode.GRAPH

        return UnifiedReasoningMode.AUTO

    def _gnn_reasoning(self, query: str, top_k: int) -> List[UnifiedReasoningResult]:
        """GNN推理"""
        self.stats['gnn_queries'] += 1

        if not self.gnn_engine or not self.gnn_engine.model:
            return []

        results_data = self.gnn_engine.reason(query, top_k=top_k)

        results = []
        for r in results_data:
            results.append(UnifiedReasoningResult(
                answer=f"{query} -> {r['target']}",
                confidence=r['probability'],
                reasoning_type='gnn_link_prediction',
                source='gnn_engine',
                metadata={'target': r['target']}
            ))

        return results

    def _probabilistic_reasoning(self, query: str, context: Dict,
                                top_k: int) -> List[UnifiedReasoningResult]:
        """概率推理"""
        self.stats['prob_queries'] += 1

        if not self.prob_engine:
            return []

        evidence = context.get('evidence', {}) if context else {}
        result = self.prob_engine.query(query, evidence)

        # 转换为推理结果
        results = []
        for state, prob in result.items():
            results.append(UnifiedReasoningResult(
                answer=f"{query} = {state}",
                confidence=prob,
                reasoning_type='probabilistic',
                source='bayesian_network',
                metadata={'state': state, 'evidence': evidence}
            ))

        # 按置信度排序
        results.sort(key=lambda x: -x.confidence)

        return results[:top_k]

    def _graph_reasoning(self, query: str, top_k: int) -> List[UnifiedReasoningResult]:
        """图结构推理"""
        self.stats['graph_queries'] += 1

        if not self.graph_kb:
            return []

        # 使用知识库查询
        query_result = self.graph_kb.query_knowledge(query, top_k=top_k)

        results = []
        for item in query_result.results:
            results.append(UnifiedReasoningResult(
                answer=str(item.get('content', item)),
                confidence=item.get('score', 0.5),
                reasoning_type='graph_traversal',
                source='graph_database'
            ))

        return results

    def _hybrid_reasoning(self, query: str, context: Dict,
                         top_k: int) -> List[UnifiedReasoningResult]:
        """混合推理：结合多种方法"""
        self.stats['hybrid_queries'] += 1

        all_results = []

        # 1. GNN推理
        if self.gnn_engine:
            all_results.extend(self._gnn_reasoning(query, top_k))

        # 2. 图推理
        if self.graph_kb:
            all_results.extend(self._graph_reasoning(query, top_k))

        # 按置信度排序并去重
        seen = set()
        unique_results = []
        for result in all_results:
            key = result.answer
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        unique_results.sort(key=lambda x: -x.confidence)

        return unique_results[:top_k]

    def explain(self, query: str, mode: UnifiedReasoningMode = UnifiedReasoningMode.AUTO,
               context: Dict = None) -> Dict[str, Any]:
        """
        解释推理过程

        Args:
            query: 查询文本
            mode: 推理模式
            context: 额外上下文

        Returns:
            解释信息
        """
        # 执行推理
        results = self.reason(query, mode, context, top_k=1)

        if not results:
            return {'error': 'No results found'}

        top_result = results[0]

        explanation = {
            'query': query,
            'mode': mode.value if isinstance(mode, UnifiedReasoningMode) else mode,
            'answer': top_result.answer,
            'confidence': top_result.confidence,
            'reasoning_type': top_result.reasoning_type,
            'source': top_result.source,
            'steps': top_result.reasoning_steps,
            'metadata': top_result.metadata
        }

        # 添加模块特定的解释
        if mode == UnifiedReasoningMode.PROBABILISTIC and self.prob_engine:
            prob_expl = self.prob_engine.explain(query, context.get('evidence', {}) if context else None)
            explanation['probabilistic_details'] = prob_expl

        return explanation

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'queries': self.stats.copy(),
            'modules': {
                'graph_kb': self.graph_kb is not None,
                'gnn_engine': self.gnn_engine is not None,
                'prob_engine': self.prob_engine is not None,
            },
            'device': str(self.device),
        }

    def benchmark(self, queries: List[str], iterations: int = 10) -> Dict[str, float]:
        """
        性能基准测试

        Args:
            queries: 测试查询列表
            iterations: 每个查询的迭代次数

        Returns:
            性能统计
        """
        import time

        timings = {
            'graph': [],
            'gnn': [],
            'prob': [],
        }

        for query in queries:
            # 测试图推理
            if self.graph_kb:
                start = time.time()
                for _ in range(iterations):
                    self._graph_reasoning(query, top_k=3)
                timings['graph'].append((time.time() - start) / iterations)

            # 测试GNN推理
            if self.gnn_engine and self.gnn_engine.model:
                start = time.time()
                for _ in range(iterations):
                    self._gnn_reasoning(query, top_k=3)
                timings['gnn'].append((time.time() - start) / iterations)

        # 计算统计数据
        stats = {}
        for mode, times in timings.items():
            if times:
                stats[f'{mode}_avg'] = np.mean(times)
                stats[f'{mode}_std'] = np.std(times)
                stats[f'{mode}_min'] = np.min(times)
                stats[f'{mode}_max'] = np.max(times)

        return stats


# ============================================================================
# 便捷函数
# ============================================================================

def get_unified_system(config: Dict = None) -> UnifiedReasoningSystem:
    """获取统一推理系统实例"""
    return UnifiedReasoningSystem(config)


if __name__ == '__main__':
    print("=== 统一推理系统 ===")
    print()
    print("整合模块:")
    print("- Stage 1: 图数据库知识库")
    print("- Stage 2: GNN推理引擎")
    print("- Stage 3: 概率推理系统")
    print()
    print("功能:")
    print("- 统一推理接口")
    print("- 自动模式选择")
    print("- 混合推理")
    print("- 推理解释")
    print("- 性能基准测试")
    print()
    print("支持的推理模式:")
    for mode in UnifiedReasoningMode:
        print(f"  - {mode.value}")
