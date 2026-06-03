"""
统一因果推理模块 - 整合三重实现

整合：
1. CausalReasoningModule (causal.py) - Beta后验规则学习
2. CausalEngine (causal_engine.py) - DAG遍历和do-calculus
3. CausalitySystem (core_knowledge.py) - 先验约束

提供统一的因果推理接口，支持：
- 经验学习（Beta后验）
- 结构化推理（DAG遍历）
- 干预推理（do-calculus）
- 先验验证
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class CausalResult:
    """因果推理结果"""
    cause: str
    effect: str
    confidence: float
    source: str  # 'beta_learning', 'dag_inference', 'prior_constraint'
    reasoning: str = ""  # 推理过程说明


class UnifiedCausalReasoner:
    """统一因果推理模块

    整合三个因果推理系统：
    1. Beta学习器：从经验中学习因果规则
    2. DAG引擎：结构化因果推理
    3. 先验系统：Spelke核心知识约束
    """

    def __init__(self):
        # Beta后验学习器
        self.beta_learner = None
        self._init_beta_learner()

        # DAG因果引擎
        self.dag_engine = None
        self._init_dag_engine()

        # 先验约束系统
        self.prior_system = None
        self._init_prior_system()

        # 统计信息
        self._query_count = 0

    def _init_beta_learner(self):
        """初始化Beta后验学习器"""
        try:
            from .causal import CausalReasoningModule
            self.beta_learner = CausalReasoningModule()
        except Exception as e:
            print(f"[WARN] Beta学习器初始化失败: {e}")

    def _init_dag_engine(self):
        """初始化DAG因果引擎"""
        try:
            from .causal_engine import CausalEngine
            self.dag_engine = CausalEngine()
        except Exception as e:
            print(f"[WARN] DAG引擎初始化失败: {e}")

    def _init_prior_system(self):
        """初始化先验约束系统"""
        try:
            from ..learning.core_knowledge import CausalitySystem
            self.prior_system = CausalitySystem()
        except Exception as e:
            print(f"[WARN] 先验系统初始化失败: {e}")

    # ===================================================================
    # 学习接口
    # ===================================================================

    def learn_causal(self, cause: str, effect: str, confidence: float = 1.0):
        """学习新因果规则（来自经验）

        Args:
            cause: 原因
            effect: 结果
            confidence: 置信度
        """
        if self.beta_learner:
            self.beta_learner.observe(cause, effect)

        if self.dag_engine:
            self.dag_engine.add_edge(cause, effect, strength=confidence)

    def observe_causal(self, cause: str, effect: str):
        """观察到因果共现

        Args:
            cause: 原因
            effect: 结果
        """
        if self.beta_learner:
            self.beta_learner.observe(cause, effect)

        if self.dag_engine:
            self.dag_engine.observe({cause: 1.0, effect: 1.0})

    # ===================================================================
    # 推理接口
    # ===================================================================

    def query_causal(self, query: str, top_k: int = 5) -> List[CausalResult]:
        """查询因果关系

        Args:
            query: 查询文本
            top_k: 返回top-k个结果

        Returns:
            因果关系结果列表
        """
        self._query_count += 1
        results = []

        # 从Beta学习器查询
        if self.beta_learner:
            beta_rules = self._get_beta_rules(query)
            for rule in beta_rules:
                results.append(CausalResult(
                    cause=rule.cause,
                    effect=rule.effect,
                    confidence=rule.confidence,
                    source='beta_learning',
                    reasoning=f"Beta后验: {rule.evidence_count}/{rule.total_observations}次观察"
                ))

        # 从DAG引擎查询
        if self.dag_engine:
            dag_results = self._query_dag(query)
            for dr in dag_results:
                results.append(CausalResult(
                    cause=dr['cause'],
                    effect=dr['effect'],
                    confidence=dr.get('confidence', 0.5),
                    source='dag_inference',
                    reasoning="DAG因果图推理"
                ))

        # 按置信度排序
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:top_k]

    def _get_beta_rules(self, query: str) -> list:
        """从Beta学习器获取相关规则"""
        if not self.beta_learner:
            return []

        # 提取查询中的关键词
        keywords = self._extract_keywords(query)

        # 查找包含关键词的规则
        matching_rules = []
        for (cause, effect), rule in self.beta_learner.rules.items():
            # 检查原因或结果是否包含关键词
            for keyword in keywords:
                if keyword in cause or keyword in effect:
                    matching_rules.append(rule)
                    break

        # 按置信度排序
        matching_rules.sort(key=lambda r: -r.confidence)
        return matching_rules[:5]

    def _query_dag(self, query: str) -> list:
        """从DAG引擎查询"""
        if not self.dag_engine:
            return []

        keywords = self._extract_keywords(query)
        results = []

        # 查找以关键词为起点的因果边
        for keyword in keywords:
            if keyword in self.dag_engine.nodes:
                node = self.dag_engine.nodes[keyword]
                # 返回该节点的所有直接子节点
                for child in node.children:
                    # 找到对应的边
                    for edge in self.dag_engine.edges:
                        if edge.source == keyword and edge.target == child:
                            results.append({
                                'cause': keyword,
                                'effect': child,
                                'confidence': edge.strength,
                                'source': 'dag_inference'
                            })

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re
        keywords = re.findall(r'[一-鿿]{2,4}', text)
        stopwords = {'的', '了', '在', '是', '和', '有', '与', '被', '将', '把'}
        return [k for k in keywords if k not in stopwords]

    # ===================================================================
    # 高级推理接口
    # ===================================================================

    def get_causal_chain(self, source: str, target: str,
                         max_depth: int = 5) -> List[List[str]]:
        """获取因果链

        Args:
            source: 起始节点
            target: 目标节点
            max_depth: 最大深度

        Returns:
            因果路径列表 [[A, B, C], [A, D, C], ...]
        """
        if not self.dag_engine:
            return []

        paths = []
        self._find_paths(source, target, [], paths, max_depth)
        return paths

    def _find_paths(self, current: str, target: str, path: List[str],
                   paths: List[List[str]], max_depth: int):
        """DFS查找因果路径"""
        if current not in self.dag_engine.nodes:
            return

        path.append(current)

        if len(path) > max_depth:
            path.pop()
            return

        if current == target and len(path) > 1:
            paths.append(path[:])
            path.pop()
            return

        # 遍历子节点
        node = self.dag_engine.nodes[current]
        for child in node.children:
            self._find_paths(child, target, path, paths, max_depth)

        path.pop()

    def reason_intervention(self, intervention: Dict) -> Dict:
        """干预推理：do-calculus

        Args:
            intervention: {'node': 节点名, 'value': 干预值}

        Returns:
            干预结果 {'effect': ..., 'confidence': ...}
        """
        if not self.dag_engine:
            return {'error': 'DAG引擎未初始化'}

        node = intervention.get('node')
        value = intervention.get('value', 1.0)

        if not node or node not in self.dag_engine.nodes:
            return {'error': '无效的干预节点'}

        try:
            result = self.dag_engine.do_intervention(node, value)
            return {
                'effect': result,
                'confidence': 0.7,
                'reasoning': f"干预 {node}={value} 的效果"
            }
        except Exception as e:
            return {'error': f'干预推理失败: {e}'}

    def check_violation(self, event: Dict) -> float:
        """检查事件是否违反因果先验

        Args:
            event: 事件数据

        Returns:
            违反程度 [0, 1]
        """
        if not self.prior_system:
            return 0.0

        try:
            # 调用先验系统检查
            result = self.prior_system.process_observation(event)
            surprise = result.get('total_surprise', 0.0)
            return min(1.0, surprise / 10.0)  # 归一化到[0,1]
        except Exception:
            return 0.0

    # ===================================================================
    # 统计接口
    # ===================================================================

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = {
            'query_count': self._query_count,
            'beta_rules': 0,
            'dag_nodes': 0,
            'dag_edges': 0,
            'prior_active': False
        }

        if self.beta_learner:
            stats['beta_rules'] = len(self.beta_learner.rules)

        if self.dag_engine:
            stats['dag_nodes'] = len(self.dag_engine.nodes)
            stats['dag_edges'] = len(self.dag_engine.edges)

        if self.prior_system:
            stats['prior_active'] = True

        return stats


if __name__ == '__main__':
    print("=== 统一因果推理模块 ===")
    print()
    print("整合三个因果推理系统:")
    print("1. Beta后验学习器 - 从经验中学习因果规则")
    print("2. DAG因果引擎 - 结构化因果推理")
    print("3. 先验约束系统 - Spelke核心知识")
    print()
    print("核心功能:")
    print("- 因果规则学习")
    print("- 因果关系查询")
    print("- 因果链追踪")
    print("- 干预推理")
    print("- 先验验证")
