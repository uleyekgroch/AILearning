"""
语言驱动的记忆支架（双编码理论）—— PyTorch 版

核心思想：语言编码通过双重表征（感知 + 语言）提升回忆准确率。
叙事标记支持序列记忆与时间线索检索。

参考：
- Paivio (1971) 双编码理论
- Ebbinghaus 遗忘曲线
"""

from typing import Dict, List, Optional

import torch

from src.core.device import get_device


class DualCodingMemory:
    """双编码记忆：感知痕迹 + 语言痕迹"""

    def __init__(self, capacity: int = 1000, feature_dim: int = 40,
                 device: str = 'auto'):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.device = get_device(device)

        # 经验存储：每个经验包含感知向量和语言标注
        # 按插入顺序管理，超出容量时淘汰最旧
        self._entries: List[Dict] = []
        # 符号 -> 经验索引列表（语言索引）
        self._symbol_index: Dict[str, List[int]] = {}

        # 统计：单编码 / 双编码命中率
        self._single_hits: int = 0
        self._dual_hits: int = 0
        self._total_queries: int = 0

        # 感知矩阵缓存（用于批量距离计算）
        self._percept_matrix: Optional[torch.Tensor] = None
        self._dirty: bool = True

    # -------------------------------------------------------------------
    # 编码
    # -------------------------------------------------------------------

    def encode(self, perceptual_features: Dict[str, torch.Tensor],
               symbols: List[str], context: str = "") -> None:
        """
        用感知 + 语言双重痕迹编码一次经验

        Args:
            perceptual_features: 感知特征字典，必须包含 'vector' 键 (Tensor)
            symbols: 与此次经验关联的语言符号列表
            context: 可选的叙事上下文描述
        """
        vec = perceptual_features.get('vector', torch.zeros(self.feature_dim))
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu()
        else:
            vec = torch.tensor(vec, dtype=torch.float32)

        if vec.dim() == 0:
            vec = vec.unsqueeze(0)
        if vec.shape[0] == 0:
            vec = torch.zeros(self.feature_dim)

        entry = {
            'perceptual': vec,
            'symbols': list(symbols),
            'context': context,
            'has_linguistic': len(symbols) > 0 or bool(context),
            'timestamp': len(self._entries),  # 单调递增时间戳
        }

        # 容量管理：淘汰最旧条目
        if len(self._entries) >= self.capacity:
            self._evict_oldest()

        idx = len(self._entries)
        self._entries.append(entry)

        # 更新语言索引
        for sym in symbols:
            if sym not in self._symbol_index:
                self._symbol_index[sym] = []
            self._symbol_index[sym].append(idx)

        self._dirty = True

    # -------------------------------------------------------------------
    # 语言线索回忆
    # -------------------------------------------------------------------

    def recall_with_symbols(self, cue_symbols: List[str],
                            k: int = 5) -> List[Dict]:
        """
        通过语言线索回忆经验

        检索所有包含至少一个 cue_symbols 的经验，按匹配度排序。

        Args:
            cue_symbols: 检索线索符号
            k: 返回前 k 个结果

        Returns:
            匹配的经验列表，每项包含 'entry' 和 'score'
        """
        self._total_queries += 1

        # 收集候选索引及其匹配分数
        candidate_scores: Dict[int, float] = {}
        for sym in cue_symbols:
            indices = self._symbol_index.get(sym, [])
            for idx in indices:
                candidate_scores[idx] = candidate_scores.get(idx, 0) + 1.0

        # 按分数排序
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1],
                        reverse=True)
        results = []
        for idx, score in ranked[:k]:
            entry = self._entries[idx]
            results.append({
                'entry': {
                    'symbols': entry['symbols'],
                    'context': entry['context'],
                    'timestamp': entry['timestamp'],
                },
                'score': score,
            })

        # 统计双编码优势
        if results:
            has_dual = any(
                self._entries[idx]['has_linguistic']
                for idx, _ in ranked[:k]
                if idx < len(self._entries)
            )
            if has_dual:
                self._dual_hits += 1
            else:
                self._single_hits += 1

        return results

    # -------------------------------------------------------------------
    # 感知线索回忆
    # -------------------------------------------------------------------

    def recall_with_perception(self, cue_features: Dict[str, torch.Tensor],
                               k: int = 5) -> List[Dict]:
        """
        通过感知线索回忆经验

        用余弦相似度检索与 cue 特征最接近的经验。

        Args:
            cue_features: 必须包含 'vector' 键的感知特征
            k: 返回前 k 个结果

        Returns:
            匹配的经验列表，每项包含 'entry' 和 'score'
        """
        self._total_queries += 1

        if not self._entries:
            return []

        cue_vec = cue_features.get('vector', torch.zeros(self.feature_dim))
        if not isinstance(cue_vec, torch.Tensor):
            cue_vec = torch.tensor(cue_vec, dtype=torch.float32)
        cue_vec = cue_vec.to(self.device).float()
        if cue_vec.dim() == 1:
            cue_vec = cue_vec.unsqueeze(0)

        # 重建感知矩阵
        self._rebuild_percept_matrix()
        if self._percept_matrix is None:
            return []

        # 余弦相似度
        cue_norm = torch.nn.functional.normalize(cue_vec, dim=1)
        mat_norm = torch.nn.functional.normalize(self._percept_matrix, dim=1)
        similarities = (cue_norm @ mat_norm.T).squeeze(0)  # (num_entries,)

        topk = min(k, similarities.shape[0])
        top_scores, top_indices = torch.topk(similarities, topk)

        results = []
        for i in range(topk):
            idx = top_indices[i].item()
            entry = self._entries[idx]
            results.append({
                'entry': {
                    'symbols': entry['symbols'],
                    'context': entry['context'],
                    'timestamp': entry['timestamp'],
                },
                'score': top_scores[i].item(),
            })

        if results:
            if any(self._entries[idx]['has_linguistic']
                   for idx in top_indices.tolist()
                   if idx < len(self._entries)):
                self._dual_hits += 1
            else:
                self._single_hits += 1

        return results

    # -------------------------------------------------------------------
    # 分析指标
    # -------------------------------------------------------------------

    def get_dual_coding_benefit(self) -> float:
        """
        双编码相对于单编码的回忆提升率

        Returns:
            0.0 ~ 1.0，值越大说明双编码优势越明显
        """
        total = self._dual_hits + self._single_hits
        if total == 0:
            return 0.0
        return self._dual_hits / total

    def get_forgetting_curve(self, symbol: str) -> List[float]:
        """
        Ebbinghaus 遗忘曲线：某符号关联记忆的遗忘率随时间衰减

        模拟指数遗忘 R = e^(-t/S)，S 为记忆稳定性。
        双编码记忆的稳定性 S 更高。

        Args:
            symbol: 目标符号

        Returns:
            每个关联经验的当前保留率列表
        """
        indices = self._symbol_index.get(symbol, [])
        if not indices:
            return []

        current_time = len(self._entries)
        retention_rates = []

        for idx in indices:
            if idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            elapsed = current_time - entry['timestamp']
            # 双编码稳定性更高
            stability = 50.0 if entry['has_linguistic'] else 20.0
            retention = float(torch.exp(
                torch.tensor(-elapsed / stability)
            ))
            retention_rates.append(retention)

        return retention_rates

    # -------------------------------------------------------------------
    # 序列化
    # -------------------------------------------------------------------

    def save_state(self) -> dict:
        """将模块状态序列化为字典"""
        entries_serialized = []
        for e in self._entries:
            entries_serialized.append({
                'perceptual': e['perceptual'].tolist(),
                'symbols': e['symbols'],
                'context': e['context'],
                'has_linguistic': e['has_linguistic'],
                'timestamp': e['timestamp'],
            })

        symbol_index_serialized = {
            sym: list(idxs) for sym, idxs in self._symbol_index.items()
        }

        return {
            'entries': entries_serialized,
            'symbol_index': symbol_index_serialized,
            'single_hits': self._single_hits,
            'dual_hits': self._dual_hits,
            'total_queries': self._total_queries,
            'capacity': self.capacity,
            'feature_dim': self.feature_dim,
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复模块状态"""
        self.capacity = state.get('capacity', self.capacity)
        self.feature_dim = state.get('feature_dim', self.feature_dim)
        self._single_hits = state.get('single_hits', 0)
        self._dual_hits = state.get('dual_hits', 0)
        self._total_queries = state.get('total_queries', 0)

        self._entries = []
        for e in state.get('entries', []):
            self._entries.append({
                'perceptual': torch.tensor(e['perceptual'], dtype=torch.float32),
                'symbols': e['symbols'],
                'context': e['context'],
                'has_linguistic': e['has_linguistic'],
                'timestamp': e['timestamp'],
            })

        self._symbol_index = {}
        for sym, idxs in state.get('symbol_index', {}).items():
            self._symbol_index[sym] = list(idxs)

        self._dirty = True

    # -------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """淘汰最旧条目并清理索引"""
        if not self._entries:
            return
        evicted = self._entries.pop(0)
        # 清理符号索引中的旧索引，全部减 1
        new_index: Dict[str, List[int]] = {}
        for sym in evicted.get('symbols', []):
            old_list = self._symbol_index.get(sym, [])
            # 移除第一个匹配（即被淘汰的条目），其余索引减 1
            updated = [i - 1 for i in old_list if i != 0]
            if updated:
                new_index[sym] = updated
            elif sym in self._symbol_index:
                new_index[sym] = []

        # 对未被清理的符号也要调整索引
        for sym, idxs in self._symbol_index.items():
            if sym not in new_index:
                new_index[sym] = [i - 1 for i in idxs if i > 0]

        self._symbol_index = new_index

    def _rebuild_percept_matrix(self) -> None:
        """重建感知向量矩阵缓存"""
        if not self._dirty or not self._entries:
            return
        vecs = [e['perceptual'].to(self.device).float()
                for e in self._entries]
        self._percept_matrix = torch.stack(vecs)
        self._dirty = False
