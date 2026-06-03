"""
神经符号整合模块 - Neural-Symbolic Integration

核心思想：
- 将神经网络的模式识别能力与符号推理的逻辑推理能力结合
- 双向映射：神经表征 ↔ 符号表征
- 混合推理：结合神经和符号的优势

架构层次：
1. 神经层：感知、模式识别（向量空间）
2. 符号层：推理、逻辑操作（离散符号）
3. 接口层：双向映射与转换

基于2024-2025年研究：
- Neuro-Symbolic VQA
- Differentiable Neural Computer
- Neural Theorem Provers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class NeuralActivation:
    """神经激活模式"""
    vector: torch.Tensor  # 向量表征
    confidence: float     # 置信度
    source: str          # 来源模块


@dataclass
class SymbolicRepresentation:
    """符号表征"""
    symbols: List[str]    # 符号列表
    structure: Dict       # 结构化表示
    confidence: float     # 置信度


class NeuroSymbolicBridge:
    """
    神经-符号双向桥接器

    负责：
    1. 神经→符号：从向量中提取符号
    2. 符号→神经：从符号生成向量
    3. 混合推理：神经直觉 + 符号验证
    """

    def __init__(self, neural_dim: int = 128, symbol_space_size: int = 10000):
        self.neural_dim = neural_dim
        self.symbol_space_size = symbol_space_size

        # 符号到向量的映射
        self.symbol_to_vector: Dict[str, torch.Tensor] = {}

        # 向量到符号的索引
        self.vector_to_symbol: Dict[int, str] = {}

        # 神经原型
        self.prototypes: Dict[str, torch.Tensor] = {}

        # 可学习的映射网络
        self.neural_to_symbol_net = None
        self.symbol_to_neural_net = None

    def register_symbol(self, symbol: str, vector: torch.Tensor):
        """注册符号-向量映射

        Args:
            symbol: 符号（如概念名称）
            vector: 神经向量表征
        """
        self.symbol_to_vector[symbol] = vector.detach().clone()

        # 创建向量到符号的索引（简化：使用向量哈希）
        vector_hash = hash(tuple(vector.tolist())) % self.symbol_space_size
        self.vector_to_symbol[vector_hash] = symbol

    def neural_to_symbol(self, neural_vector: torch.Tensor,
                         top_k: int = 3) -> List[Tuple[str, float]]:
        """神经→符号转换

        Args:
            neural_vector: 神经向量
            top_k: 返回top-k个候选符号

        Returns:
            [(symbol, similarity), ...]
        """
        candidates = []

        # 方法1：精确匹配（如果已注册）
        vector_hash = hash(tuple(neural_vector.tolist())) % self.symbol_space_size
        if vector_hash in self.vector_to_symbol:
            symbol = self.vector_to_symbol[vector_hash]
            candidates.append((symbol, 1.0))

        # 方法2：原型匹配
        for symbol, prototype in self.prototypes.items():
            similarity = self._cosine_similarity(neural_vector, prototype)
            if similarity > 0.5:
                candidates.append((symbol, similarity))

        # 方法3：符号空间搜索
        if len(candidates) < top_k:
            for symbol, symbol_vector in self.symbol_to_vector.items():
                similarity = self._cosine_similarity(neural_vector, symbol_vector)
                if similarity > 0.3:
                    candidates.append((symbol, similarity))

        # 排序并返回top-k
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def symbol_to_neural(self, symbol: str) -> Optional[torch.Tensor]:
        """符号→神经转换

        Args:
            symbol: 符号

        Returns:
            神经向量（如果存在）
        """
        if symbol in self.symbol_to_vector:
            return self.symbol_to_vector[symbol].clone()
        return None

    def add_prototype(self, symbol: str, prototype: torch.Tensor):
        """添加符号原型

        Args:
            symbol: 符号名称
            prototype: 原型向量
        """
        self.prototypes[symbol] = prototype.detach().clone()

    def _cosine_similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """计算余弦相似度"""
        return torch.dot(v1, v2).item() / (torch.norm(v1) * torch.norm(v2) + 1e-8)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'registered_symbols': len(self.symbol_to_vector),
            'prototypes': len(self.prototypes),
            'neural_dim': self.neural_dim,
        }


class HybridReasoningEngine:
    """
    混合推理引擎

    结合：
    1. 神经推理：快速、直觉、基于模式
    2. 符号推理：精确、逻辑、可解释
    """

    def __init__(self, bridge: NeuroSymbolicBridge):
        self.bridge = bridge

        # 推理历史
        self.neural_reasoning_history = []
        self.symbolic_reasoning_history = []
        self.hybrid_reasoning_history = []

    def reason(self, question: str,
               neural_context: Optional[torch.Tensor] = None,
               symbolic_context: Optional[List[str]] = None) -> Dict:
        """混合推理

        Args:
            question: 问题
            neural_context: 神经上下文（向量）
            symbolic_context: 符号上下文（概念列表）

        Returns:
            推理结果
        """
        results = {
            'neural_result': None,
            'symbolic_result': None,
            'hybrid_result': None,
            'confidence': 0.0,
            'reasoning_type': 'unknown'
        }

        # 1. 神经推理（直觉、快速）
        if neural_context is not None:
            neural_result = self._neural_reasoning(question, neural_context)
            results['neural_result'] = neural_result
            results['neural_confidence'] = neural_result.get('confidence', 0.0)

        # 2. 符号推理（逻辑、精确）
        if symbolic_context is not None:
            symbolic_result = self._symbolic_reasoning(question, symbolic_context)
            results['symbolic_result'] = symbolic_result
            results['symbolic_confidence'] = symbolic_result.get('confidence', 0.0)

        # 3. 混合推理（融合）
        results.update(self._hybrid_reasoning(question, results))

        return results

    def _neural_reasoning(self, question: str,
                         context: torch.Tensor) -> Dict:
        """神经推理

        基于神经激活模式的推理（快速、直觉）
        """
        # 从神经上下文中提取符号
        symbols = self.bridge.neural_to_symbol(context, top_k=5)

        if symbols:
            best_symbol, confidence = symbols[0]
            return {
                'answer': f"基于神经模式，最相关的是{best_symbol}",
                'confidence': confidence,
                'reasoning_type': 'neural',
                'symbols': symbols
            }

        return {
            'answer': "神经推理未找到匹配",
            'confidence': 0.0,
            'reasoning_type': 'neural'
        }

    def _symbolic_reasoning(self, question: str,
                           context: List[str]) -> Dict:
        """符号推理

        基于符号逻辑的推理（精确、可解释）
        """
        # 简化：基于符号集合的逻辑推理
        if question in context:
            return {
                'answer': f"是的，{question}在已知知识中",
                'confidence': 1.0,
                'reasoning_type': 'symbolic'
            }

        # 关系推理
        related = [s for s in context if s != question and len(set(s) & set(question)) > 0]
        if related:
            return {
                'answer': f"相关的概念包括：{'、'.join(related[:3])}",
                'confidence': 0.7,
                'reasoning_type': 'symbolic'
            }

        return {
            'answer': "符号推理未找到答案",
            'confidence': 0.0,
            'reasoning_type': 'symbolic'
        }

    def _hybrid_reasoning(self, question: str, partial_results: Dict) -> Dict:
        """混合推理

        融合神经和符号推理结果
        """
        neural_conf = partial_results.get('neural_confidence', 0.0)
        symbolic_conf = partial_results.get('symbolic_confidence', 0.0)

        # 决策策略
        if neural_conf > 0.7:
            # 神经推理高置信度，直接采用
            return {
                'hybrid_result': partial_results['neural_result'],
                'confidence': neural_conf,
                'reasoning_type': 'neural_dominant'
            }
        elif symbolic_conf > 0.7:
            # 符号推理高置信度，直接采用
            return {
                'hybrid_result': partial_results['symbolic_result'],
                'confidence': symbolic_conf,
                'reasoning_type': 'symbolic_dominant'
            }
        elif neural_conf > 0.3 and symbolic_conf > 0.3:
            # 两者都有一定置信度，融合
            return {
                'hybrid_result': {
                    'answer': f"神经推理: {partial_results['neural_result']['answer']}; "
                              f"符号推理: {partial_results['symbolic_result']['answer']}"
                },
                'confidence': (neural_conf + symbolic_conf) / 2,
                'reasoning_type': 'hybrid'
            }
        else:
            # 都不置信，返回不确定
            return {
                'hybrid_result': {
                    'answer': "需要更多信息来回答"
                },
                'confidence': 0.0,
                'reasoning_type': 'uncertain'
            }

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'neural_reasoning_count': len(self.neural_reasoning_history),
            'symbolic_reasoning_count': len(self.symbolic_reasoning_history),
            'hybrid_reasoning_count': len(self.hybrid_reasoning_history),
        }


class NeuroSymbolicIntegrator:
    """
    神经符号整合器 - 主入口

    负责协调：
    1. 神经-符号桥接
    2. 混合推理
    3. 反馈循环（推理结果→更新系统）
    """

    def __init__(self, neural_dim: int = 128):
        self.bridge = NeuroSymbolicBridge(neural_dim)
        self.hybrid_engine = HybridReasoningEngine(self.bridge)

        # 反馈循环配置
        self.feedback_enabled = True
        self.feedback_history = []

    def integrate_reasoning(self, question: str,
                            neural_vector: Optional[torch.Tensor] = None,
                            symbols: Optional[List[str]] = None) -> Dict:
        """整合推理

        Args:
            question: 问题
            neural_vector: 神经向量
            symbols: 符号列表

        Returns:
            推理结果
        """
        # 执行混合推理
        result = self.hybrid_engine.reason(question, neural_vector, symbols)

        # 反馈：如果推理成功，更新桥接
        if self.feedback_enabled and result.get('confidence', 0.0) > 0.7:
            self._feedback(result, neural_vector, symbols)

        return result

    def _feedback(self, result: Dict, neural_vector: torch.Tensor, symbols: List[str]):
        """反馈循环：更新系统

        Args:
            result: 推理结果
            neural_vector: 神经向量
            symbols: 符号列表
        """
        feedback_record = {
            'result': result,
            'timestamp': len(self.feedback_history)
        }

        # 如果符号推理成功，注册符号-向量映射
        if result.get('reasoning_type') in ['symbolic_dominant', 'hybrid']:
            if neural_vector is not None and symbols:
                for symbol in symbols:
                    if symbol not in self.bridge.symbol_to_vector:
                        self.bridge.register_symbol(symbol, neural_vector)
                        feedback_record['new_mappings'] = len(symbols)

        self.feedback_history.append(feedback_record)

    def learn_mapping(self, symbol: str, vector: torch.Tensor):
        """学习符号-向量映射

        Args:
            symbol: 符号
            vector: 向量
        """
        self.bridge.register_symbol(symbol, vector)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'bridge_stats': self.bridge.get_stats(),
            'engine_stats': self.hybrid_engine.get_stats(),
            'feedback_count': len(self.feedback_history),
        }


if __name__ == '__main__':
    print("=== 神经符号整合模块 ===")
    print()
    print("核心功能:")
    print("- 神经-符号双向桥接")
    print("- 混合推理（神经+符号）")
    print("- 反馈循环（推理结果→更新）")
    print()
    print("应用场景:")
    print("- 感知推理（神经感知→符号理解）")
    print("- 可解释推理（符号逻辑）")
    print("- 学习加速（符号指导神经）")
