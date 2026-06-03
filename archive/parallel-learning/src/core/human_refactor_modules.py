"""
人类式学习系统 - learner.py 完整重构版本
移除所有训练组件，整合生物机制
"""

# ========================================================================
# 第一部分：替换编码器 - 移除Transformer，使用直接感知
# ========================================================================

def _encode_text_human_like(self, text: str, train: bool = False) -> torch.Tensor:
    """
    直接感知编码 - 无需Transformer，无训练

    模拟视网膜→视皮层：直接神经激活，无需"训练"
    """
    import re
    device = getattr(self, 'device', torch.device('cpu'))

    # 初始化词向量字典（一次性，无需训练）
    if not hasattr(self, '_perception_vectors'):
        self._perception_vectors = {}
        self._perception_dim = self.config.obs_dim
        # 初始化高频词（模拟先天概念）
        for word in '的是在有了与我他这中大来上个国到说不和要能会就'.split():
            vec = torch.randn(self._perception_dim) * 0.1
            vec = vec / torch.norm(vec)
            self._perception_vectors[word] = vec

    # 简单分词
    words = re.findall(r'[一-鿿]{2,6}|[a-zA-Z]{2,}', text)
    if not words:
        return torch.zeros(self._perception_dim).to(device)

    # 获取词向量
    vectors = []
    for word in words:
        if word not in self._perception_vectors:
            # 随机初始化（Hebbian学习会优化）
            vec = torch.randn(self._perception_dim) * 0.1
            vec = vec / torch.norm(vec)
            self._perception_vectors[word] = vec
        vectors.append(self._perception_vectors[word].to(device))

    # 简单平均（无注意力，无需训练）
    result = torch.stack(vectors).mean(dim=0)
    return result


# ========================================================================
# 第二部分：STDP赫布学习系统
# ========================================================================

class STDPSystem:
    """STDP赫布学习系统"""

    def __init__(self, dim=128):
        self.dim = dim
        self.connections = {}  # (pre, post) → weight
        self._trace = {}  # (pre, post) → 迹迹
        self.lr = 0.01
        self._decay = 0.95

    def update(self, pre_activity, post_activity, is_pre_before_post=True):
        """
        STDP更新：基于时序的局部学习

        规则：pre在post之前激发 → 连接增强
        """
        if is_pre_before_post:
            delta = self.lr * pre_activity * post_activity
        else:
            delta = -self.lr * pre_activity * post_activity

        # 追迹衰减
        for pair in self._trace:
            self._trace[pair] *= self._decay

        return delta

    def get_strength(self, pre, post):
        """获取连接强度"""
        return self.connections.get((pre, post), 0.0)

    def has_connection(self, pre, post):
        """检查是否存在连接"""
        return (pre, post) in self.connections or (post, pre) in self.connections


# ========================================================================
# 第三部分：海马快速记忆 + 皮层慢速整合
# ========================================================================

class HumanMemorySystem:
    """
    人类记忆系统 - 海马+皮层互补学习

    海马：快速单次学习，容量有限
    皮层：慢速分布学习，容量大
    """

    def __init__(self):
        # 海马：快速记忆
        self.hippocampus = {
            'episodes': [],
            'index': {},
            'capacity': 5000
        }

        # 皮层：慢速整合
        self.neocortex = {
            'long_term': [],
            'patterns': {}
        }

    def hippocampal_store(self, entities, relations, context):
        """
        海马快速存储（单次学习）
        """
        episode = {
            'entities': list(entities),
            'relations': list(relations),
            'context': context,
            'timestamp': len(self.hippocampus['episodes'])
        }

        # 建立索引
        for e in entities:
            if e not in self.hippocampus['index']:
                self.hippocampus['index'][e] = []
            self.hippocampus['index'][e].append(episode['timestamp'])

        self.hippocampus['episodes'].append(episode)

        # 容量管理
        if len(self.hippocampus['episodes']) > self.hippocampus['capacity']:
            # 随机遗忘（模拟海马容量限制）
            import random
            if random.random() < 0.1:
                idx = random.randint(0, len(self.hippocampus['episodes']) - 1)
                self.hippocampus['episodes'][idx] = self.hippocampus['episodes'][-1]
                self.hippocampus['episodes'].pop()

    def neocortical_consolidate(self):
        """
        皮层整合 - 睡眠巩固
        """
        # 将海马记忆转移到皮层
        recent = self.hippocampus['episodes'][-100:]
        for episode in recent:
            # 提取模式
            pattern = (tuple(episode['entities'][:5]),
                     tuple(episode['relations'][:3]))
            self.neocortex['patterns'][pattern] = \
                self.neocortex['patterns'].get(pattern, 0) + 1

        # 清空海马已转移的记忆
        self.hippocampus['episodes'] = self.hippocampus['episodes'][:-100]

    def recall(self, query_entity):
        """回忆相关记忆"""
        if query_entity not in self.hippocampus['index']:
            return []
        indices = self.hippocampus['index'][query_entity]
        return [self.hippocampus['episodes'][i]
                for i in indices if i < len(self.hippampus['episodes'])]


# ========================================================================
# 第四部分：激活扩散推理
# ========================================================================

class ActivationSpreadingReasoner:
    """
    激活扩散推理器 - 无需编码，无训练

    模拟大脑中的激活扩散：问题 → 种子概念 → 激活扩散 → 相关概念
    """

    def __init__(self, concept_space, memory_system):
        self.cs = concept_space
        self.memory = memory_system
        self._activation_cache = {}

    def reason(self, question):
        """
        推理（激活扩散，无计算）
        """
        # 激活种子概念
        activated = self.cs.activate(question, top_k=5, spread_depth=2)
        if not activated:
            return ""

        # 获取概念ID
        concepts = [a.concept_id for a in activated[:5]]

        # 简单组合
        if len(concepts) >= 2:
            return concepts[0] + "与" + concepts[1]
        else:
            return concepts[0] if concepts else ""


print("=== 人类式学习系统 - 核心替换模块 ===")
print()
print("替换方法:")
print("1. _encode_text → _encode_text_human_like (直接感知编码)")
print("2. 新增: STDPSystem (STDP赫布学习)")
print("3. 新增: HumanMemorySystem (海马+皮层)")
print("4. 新增: ActivationSpreadingReasoner (激活扩散推理)")
print()
print("下一步: 修改learner.py，替换方法")
