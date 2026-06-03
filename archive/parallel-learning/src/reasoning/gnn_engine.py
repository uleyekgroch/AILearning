"""
图神经网络推理引擎 - GNN-based Reasoning Engine

基于2024-2025年最新研究：
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Message Passing Neural Networks (MPNN)
- Knowledge Graph Embedding

功能：
1. 图卷积层（GCN）
2. 图注意力层（GAT）
3. 消息传递机制
4. 节点嵌入学习
5. 链接预测推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import numpy as np


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class GraphStructure:
    """图结构数据"""
    # 节点映射：节点ID -> 索引
    node_to_idx: Dict[str, int]
    # 索引映射：索引 -> 节点ID
    idx_to_node: Dict[int, str]
    # 边列表：(源索引, 目标索引, 关系类型)
    edges: List[Tuple[int, int, str]]
    # 节点特征
    node_features: torch.Tensor
    # 边特征（可选）
    edge_features: Optional[torch.Tensor] = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_to_idx)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


@dataclass
class GNNConfig:
    """GNN配置"""
    # 模型参数
    input_dim: int = 128              # 输入维度
    hidden_dim: int = 256             # 隐藏层维度
    output_dim: int = 128             # 输出维度
    num_layers: int = 3                # GNN层数
    dropout: float = 0.1               # Dropout率

    # 注意力参数（GAT）
    num_heads: int = 8                # 注意力头数
    attention_dim: int = 64            # 注意力维度

    # 训练参数
    learning_rate: float = 0.001       # 学习率
    weight_decay: float = 1e-5         # 权重衰减
    epochs: int = 100                  # 训练轮数

    # 推理参数
    num_hops: int = 3                  # 推理时的跳数
    top_k: int = 5                     # 返回top-k结果


# ============================================================================
# 基础GNN层
# ============================================================================

class GraphConvLayer(nn.Module):
    """图卷积层 (GCN)

    公式：H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 线性变换
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, in_features]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            更新后的节点特征 [num_nodes, out_features]
        """
        # 归一化邻接矩阵：D^(-1/2) A D^(-1/2)
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        # 归一化
        norm_adj = adj * deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(1, -1)

        # 图卷积：A_norm * X * W
        support = torch.mm(x, self.weight)
        output = torch.mm(norm_adj, support)

        # 加偏置
        output = output + self.bias

        # Dropout
        output = self.dropout(output)

        return output


class GraphAttentionLayer(nn.Module):
    """图注意力层 (GAT)

    使用注意力机制聚合邻居信息：
    α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    """

    def __init__(self, in_features: int, out_features: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        # 每个头的线性变换
        self.weight = nn.Parameter(torch.FloatTensor(num_heads, in_features, self.head_dim))

        # 注意力参数
        self.attention = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.head_dim, 1))

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, in_features]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            更新后的节点特征 [num_nodes, out_features]
        """
        num_nodes = x.size(0)

        # 多头注意力
        outputs = []
        for head in range(self.num_heads):
            # 线性变换：[num_nodes, head_dim]
            h = torch.mm(x, self.weight[head])

            # 计算注意力分数
            # 拼接所有节点的特征对
            h_left = h.unsqueeze(1).expand(num_nodes, num_nodes, self.head_dim)  # [N, N, D]
            h_right = h.unsqueeze(0).expand(num_nodes, num_nodes, self.head_dim)  # [N, N, D]
            h_concat = torch.cat([h_left, h_right], dim=2)  # [N, N, 2D]

            # 注意力分数
            e = self.leaky_relu(torch.matmul(h_concat, self.attention[head]))  # [N, N, 1]
            e = e.squeeze(2)  # [N, N]

            # Mask（只对存在的边计算注意力）
            mask = -1e9 * torch.ones_like(e)
            e = torch.where(adj > 0, e, mask)

            # Softmax
            attention = F.softmax(e, dim=1)  # [N, N]
            attention = self.dropout(attention)

            # 聚合邻居信息
            h_prime = torch.mm(attention, h)  # [N, head_dim]
            outputs.append(h_prime)

        # 拼接多头输出
        output = torch.cat(outputs, dim=1)  # [N, out_features]

        return output


class MessagePassingLayer(nn.Module):
    """消息传递层 (MPNN)

    通用消息传递框架：
    m_ij = f_message(h_i, h_j, e_ij)
    h_i' = f_update(h_i, Σ_{j∈N(i)} m_ij)
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 消息函数
        self.message_fn = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 更新函数
        self.update_fn = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 聚合函数（平均）
        self.aggregation = 'mean'

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, in_features]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            更新后的节点特征 [num_nodes, out_features]
        """
        num_nodes = x.size(0)

        # 计算消息
        messages = torch.zeros(num_nodes, num_nodes, self.out_features, device=x.device)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] > 0:  # 有边
                    # 消息：f(h_i, h_j)
                    h_i = x[i]
                    h_j = x[j]
                    h_concat = torch.cat([h_i, h_j], dim=0).unsqueeze(0)
                    messages[i, j] = self.message_fn(h_concat).squeeze(0)

        # 聚合消息
        if self.aggregation == 'mean':
            # 计算每个节点的邻居数量
            neighbor_counts = torch.sum(adj, dim=1, keepdim=True)
            aggregated = torch.sum(messages, dim=1) / (neighbor_counts + 1e-6)
        else:
            aggregated = torch.sum(messages, dim=1)

        # 更新节点特征
        h_concat = torch.cat([x, aggregated], dim=1)
        output = self.update_fn(h_concat)

        return output


# ============================================================================
# GNN模型
# ============================================================================

class GNNReasoningModel(nn.Module):
    """GNN推理模型

    整合多层GNN进行图级别的推理任务
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # 输入嵌入层
        self.embedding = nn.Linear(config.input_dim, config.hidden_dim)

        # GNN层
        self.gnn_layers = nn.ModuleList()

        for i in range(config.num_layers):
            if i == 0:
                in_dim = config.hidden_dim
            else:
                in_dim = config.hidden_dim

            # 使用GCN层
            self.gnn_layers.append(
                GraphConvLayer(in_dim, config.hidden_dim, config.dropout)
            )

        # 输出层
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # 链接预测头
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, graph: GraphStructure) -> torch.Tensor:
        """
        前向传播

        Args:
            graph: 图结构

        Returns:
            节点嵌入 [num_nodes, output_dim]
        """
        # 构建邻接矩阵
        adj = self._build_adjacency_matrix(graph)

        # 初始化节点特征
        x = graph.node_features
        if x.dim() == 1:
            x = x.unsqueeze(1)

        # 投影到隐藏维度
        if x.size(1) != self.config.hidden_dim:
            x = self.embedding(x)

        # 通过GNN层
        for layer in self.gnn_layers:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, p=self.config.dropout, training=self.training)

        # 输出投影
        embeddings = self.output_proj(x)

        return embeddings

    def predict_link(self, embeddings: torch.Tensor,
                     source_idx: int, target_idx: int) -> float:
        """
        预测链接概率

        Args:
            embeddings: 节点嵌入 [num_nodes, output_dim]
            source_idx: 源节点索引
            target_idx: 目标节点索引

        Returns:
            链接概率 [0, 1]
        """
        # 使用内部方法获取Tensor，然后转换为float
        prob_tensor = self._predict_link_tensor(embeddings, source_idx, target_idx)
        return prob_tensor.item()

    def _predict_link_tensor(self, embeddings: torch.Tensor,
                            source_idx: int, target_idx: int) -> torch.Tensor:
        """预测链接概率（返回Tensor，用于训练）"""
        # 拼接源和目标节点的嵌入
        h_source = embeddings[source_idx]
        h_target = embeddings[target_idx]
        h_concat = torch.cat([h_source, h_target], dim=0).unsqueeze(0)

        # 预测
        prob = self.link_predictor(h_concat).squeeze(0)
        return prob

    def _build_adjacency_matrix(self, graph: GraphStructure) -> torch.Tensor:
        """构建邻接矩阵"""
        num_nodes = graph.num_nodes
        # 使用与节点特征相同的设备
        device = graph.node_features.device
        adj = torch.zeros(num_nodes, num_nodes, device=device)

        for src, tgt, _ in graph.edges:
            adj[src, tgt] = 1.0
            # 无向图
            adj[tgt, src] = 1.0

        return adj


# ============================================================================
# GNN推理引擎
# ============================================================================

class GNNReasoningEngine:
    """GNN推理引擎

    整合GNN模型进行知识推理：
    1. 训练阶段：学习节点嵌入
    2. 推理阶段：链接预测、路径查找
    """

    def __init__(self, config: GNNConfig = None):
        self.config = config or GNNConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_cache = None

    def build_graph(self, triples: List[Tuple[str, str, str]],
                    node_features: Dict[str, np.ndarray] = None) -> GraphStructure:
        """
        从三元组构建图

        Args:
            triples: 三元组列表 [(subject, relation, object), ...]
            node_features: 节点特征 {node_id: feature_vector}

        Returns:
            图结构
        """
        # 收集所有节点
        nodes = set()
        for subj, rel, obj in triples:
            nodes.add(subj)
            nodes.add(obj)

        # 创建节点映射
        node_to_idx = {node: i for i, node in enumerate(sorted(nodes))}
        idx_to_node = {i: node for node, i in node_to_idx.items()}

        # 创建边列表
        edges = []
        for subj, rel, obj in triples:
            src_idx = node_to_idx[subj]
            tgt_idx = node_to_idx[obj]
            edges.append((src_idx, tgt_idx, rel))

        # 创建节点特征矩阵
        if node_features:
            feature_vectors = []
            for node in sorted(nodes):
                if node in node_features:
                    feature_vectors.append(node_features[node])
                else:
                    # 随机初始化
                    feature_vectors.append(np.random.randn(self.config.input_dim))

            node_features_tensor = torch.tensor(
                np.array(feature_vectors),
                dtype=torch.float32,
                device=self.device
            )
        else:
            # 随机初始化
            node_features_tensor = torch.randn(
                len(nodes), self.config.input_dim,
                device=self.device
            )

        return GraphStructure(
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            edges=edges,
            node_features=node_features_tensor
        )

    def train(self, graph: GraphStructure,
              validation_triples: List[Tuple[int, int]] = None):
        """
        训练GNN模型

        Args:
            graph: 训练图
            validation_triples: 验证三元组 [(src_idx, tgt_idx), ...]
        """
        # 创建模型
        self.model = GNNReasoningModel(self.config).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 缓存图结构
        self.graph_cache = graph

        # 训练循环
        print(f"开始训练GNN模型 ({self.config.epochs} epochs)")
        for epoch in range(self.config.epochs):
            self.model.train()

            # 前向传播
            embeddings = self.model(graph)

            # 计算损失（链接预测任务）
            loss = self._compute_link_prediction_loss(
                embeddings, graph.edges
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {loss.item():.4f}")

        print("训练完成")

    def reason(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        推理：预测与查询相关的链接

        Args:
            query: 查询节点ID
            top_k: 返回top-k结果

        Returns:
            推理结果列表
        """
        if self.model is None or self.graph_cache is None:
            raise ValueError("模型尚未训练")

        self.model.eval()

        with torch.no_grad():
            # 获取节点嵌入
            embeddings = self.model(self.graph_cache)

            # 查找查询节点
            if query not in self.graph_cache.node_to_idx:
                return []

            query_idx = self.graph_cache.node_to_idx[query]

            # 预测所有可能的链接
            scores = []
            for node_id, node_idx in self.graph_cache.node_to_idx.items():
                if node_idx == query_idx:
                    continue

                # 预测链接概率
                prob = self.model.predict_link(embeddings, query_idx, node_idx)

                scores.append({
                    'target': node_id,
                    'probability': prob,
                    'reasoning_type': 'gnn_link_prediction'
                })

            # 按概率排序
            scores.sort(key=lambda x: -x['probability'])

            return scores[:top_k]

    def find_reasoning_path(self, source: str, target: str,
                           max_hops: int = 3) -> List[List[str]]:
        """
        查找推理路径（基于图结构）

        Args:
            source: 源节点
            target: 目标节点
            max_hops: 最大跳数

        Returns:
            路径列表 [[node1, node2, ...], ...]
        """
        if self.graph_cache is None:
            return []

        # BFS路径查找
        from collections import deque

        if source not in self.graph_cache.node_to_idx or \
           target not in self.graph_cache.node_to_idx:
            return []

        source_idx = self.graph_cache.node_to_idx[source]
        target_idx = self.graph_cache.node_to_idx[target]

        # 构建邻接表
        adj_list = {i: [] for i in range(self.graph_cache.num_nodes)}
        for src, tgt, _ in self.graph_cache.edges:
            adj_list[src].append(tgt)
            adj_list[tgt].append(src)  # 无向图

        # BFS搜索
        queue = deque([(source_idx, [source])])
        visited = {source_idx}
        paths = []

        while queue:
            curr_idx, path = queue.popleft()

            if curr_idx == target_idx:
                paths.append(path)
                if len(paths) >= 5:  # 最多返回5条路径
                    break
                continue

            if len(path) > max_hops + 1:
                continue

            # 探索邻居
            for neighbor_idx in adj_list[curr_idx]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    neighbor_id = self.graph_cache.idx_to_node[neighbor_idx]
                    queue.append((neighbor_idx, path + [neighbor_id]))

        return paths

    def _compute_link_prediction_loss(self, embeddings: torch.Tensor,
                                      edges: List[Tuple[int, int, str]]) -> torch.Tensor:
        """计算链接预测损失"""
        # 正样本：存在的边
        pos_samples = [(src, tgt) for src, tgt, _ in edges]

        # 负样本：随机采样不存在的边
        num_nodes = embeddings.size(0)
        neg_samples = []
        for _ in range(len(pos_samples)):
            src = np.random.randint(0, num_nodes)
            tgt = np.random.randint(0, num_nodes)
            neg_samples.append((src, tgt))

        # 计算损失
        loss = 0.0

        # 正样本损失
        for src, tgt in pos_samples:
            prob = self.model._predict_link_tensor(embeddings, src, tgt)
            loss += -torch.log(prob + 1e-6)

        # 负样本损失
        for src, tgt in neg_samples:
            prob = self.model._predict_link_tensor(embeddings, src, tgt)
            loss += -torch.log(1.0 - prob + 1e-6)

        return loss / (len(pos_samples) + len(neg_samples))


# ============================================================================
# 便捷函数
# ============================================================================

def get_gnn_engine(config: GNNConfig = None) -> GNNReasoningEngine:
    """获取GNN推理引擎实例"""
    return GNNReasoningEngine(config)


if __name__ == '__main__':
    print("=== GNN推理引擎 ===")
    print()
    print("核心组件:")
    print("- 图卷积层 (GCN)")
    print("- 图注意力层 (GAT)")
    print("- 消息传递层 (MPNN)")
    print("- GNN推理模型")
    print("- GNN推理引擎")
    print()
    print("功能:")
    print("- 图构建")
    print("- 模型训练")
    print("- 链接预测")
    print("- 路径推理")
