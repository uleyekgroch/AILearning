"""
跨模态接地模块 — 从非视觉模态涌现的符号

核心理念：当视觉特征模糊（物体外观相似）时，听觉/触觉符号
（"loud"/"rough"/"soft"）涌现来消除歧义。

支持四种感觉模态：visual、auditory、tactile、olfactory。

纯 PyTorch 实现，零 numpy 依赖。
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from src.core.device import get_device


# =====================================================================
# 模态类型
# =====================================================================

MODALITY_TYPES = {'visual', 'auditory', 'tactile', 'olfactory'}

# 默认模态权重 — 视觉通常占主导
DEFAULT_MODALITY_WEIGHTS: Dict[str, float] = {
    'visual': 0.5,
    'auditory': 0.2,
    'tactile': 0.2,
    'olfactory': 0.1,
}


# =====================================================================
# CrossModalObject — 多感官物体
# =====================================================================

class CrossModalObject:
    """
    具有多感官特征的物体。

    features 结构示例：
        {
            'visual':   {'color': 'red', 'shape': 'round'},
            'auditory': {'sound': 'loud'},
            'tactile':  {'texture': 'rough'},
        }
    """

    def __init__(self, features: Dict[str, Dict[str, str]],
                 obj_id: str = ''):
        # 过滤只保留合法模态
        self.features: Dict[str, Dict[str, str]] = {
            mod: feats for mod, feats in features.items()
            if mod in MODALITY_TYPES
        }
        self.obj_id = obj_id

    def get_modality_features(self, modality: str) -> Dict[str, str]:
        """获取指定模态的特征"""
        return self.features.get(modality, {})

    def all_symbols(self) -> List[str]:
        """获取所有特征值作为符号"""
        symbols: List[str] = []
        for mod_feats in self.features.values():
            for val in mod_feats.values():
                if val:
                    symbols.append(val)
        return symbols

    def __repr__(self) -> str:
        parts = [f'{mod}:{feats}' for mod, feats in self.features.items()]
        return f'CrossModalObject({self.obj_id}, [{", ".join(parts)}])'


# =====================================================================
# CrossModalGrounding — 跨模态接地
# =====================================================================

class CrossModalGrounding:
    """
    跨模态接地模块。

    职责：
    1. 管理多模态物体库
    2. 当视觉特征不足以区分时，利用其他模态消除歧义
    3. 追踪各模态对消除歧义的贡献
    4. 记录涌现的跨模态符号
    """

    def __init__(self, device: str = 'auto'):
        self.device = get_device(device)

        # 物体库
        self._objects: List[CrossModalObject] = []

        # 模态权重张量 — 可学习
        weight_list = [DEFAULT_MODALITY_WEIGHTS.get(m, 0.1) for m in sorted(MODALITY_TYPES)]
        self._modality_names: List[str] = sorted(MODALITY_TYPES)
        self._modality_weights = torch.tensor(
            weight_list, dtype=torch.float32, device=self.device,
        )

        # 跨模态符号记录：symbol -> {modality, usage_count}
        self._crossmodal_symbols: Dict[str, Dict] = {}

        # 各模态贡献统计
        self._modality_contributions: Dict[str, int] = {
            m: 0 for m in MODALITY_TYPES
        }
        self._total_disambiguations: int = 0

        # 符号-模态映射（已学到的）
        self._symbol_modality_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 物体管理
    # ------------------------------------------------------------------

    def add_object(self, obj: CrossModalObject) -> None:
        """添加物体到库中"""
        # 注册该物体的所有符号到对应模态
        for modality, feats in obj.features.items():
            for val in feats.values():
                if val:
                    self._symbol_modality_map[val] = modality
        self._objects.append(obj)

    def remove_object(self, obj_id: str) -> None:
        """按 ID 移除物体"""
        self._objects = [o for o in self._objects if o.obj_id != obj_id]

    def get_objects(self) -> List[CrossModalObject]:
        """返回所有物体"""
        return list(self._objects)

    # ------------------------------------------------------------------
    # 消除歧义
    # ------------------------------------------------------------------

    def disambiguate(self, objects: List[CrossModalObject],
                     visual_ambiguous: bool) -> List[str]:
        """
        利用跨模态信息消除物体间的歧义。

        当视觉特征无法区分物体时，依次尝试听觉、触觉、嗅觉特征。

        Args:
            objects: 需要区分的物体列表
            visual_ambiguous: 视觉特征是否已不足以区分

        Returns:
            用于区分的符号列表
        """
        if len(objects) < 2:
            return objects[0].all_symbols() if objects else []

        # 如果视觉不模糊，直接返回视觉特征
        if not visual_ambiguous:
            self._modality_contributions['visual'] += 1
            self._total_disambiguations += 1
            return self._disambiguate_by_modality(objects, 'visual')

        # 视觉模糊：依次尝试其他模态
        for modality in ['auditory', 'tactile', 'olfactory']:
            disambig_symbols = self._disambiguate_by_modality(objects, modality)
            if disambig_symbols:
                self._modality_contributions[modality] += 1
                self._total_disambiguations += 1
                # 注册为跨模态符号
                for sym in disambig_symbols:
                    if sym not in self._crossmodal_symbols:
                        self._crossmodal_symbols[sym] = {
                            'modality': modality,
                            'usage_count': 1,
                        }
                    else:
                        self._crossmodal_symbols[sym]['usage_count'] += 1
                return disambig_symbols

        # 所有模态都无法区分，返回所有特征的并集
        all_syms: List[str] = []
        for obj in objects:
            all_syms.extend(obj.all_symbols())
        self._total_disambiguations += 1
        return list(set(all_syms))

    def _disambiguate_by_modality(self, objects: List[CrossModalObject],
                                   modality: str) -> List[str]:
        """
        尝试用单一模态的特征区分物体。

        如果该模态特征能区分所有物体，返回区分性符号；
        否则返回空列表。
        """
        # 收集每个物体在该模态下的特征值集合
        per_obj: List[Set[str]] = []
        for obj in objects:
            feats = obj.get_modality_features(modality)
            per_obj.append(set(feats.values()))

        # 检查是否有能区分的符号
        unique_symbols: List[str] = []
        all_values: Set[str] = set()
        for vals in per_obj:
            all_values.update(vals)

        for val in all_values:
            count = sum(1 for vals in per_obj if val in vals)
            if count == 1:
                unique_symbols.append(val)

        # 如果有至少一个区分性符号，返回
        if unique_symbols:
            return unique_symbols

        # 检查组合区分性：不同物体的特征组合是否不同
        if len(set(tuple(sorted(vals)) for vals in per_obj)) == len(objects):
            # 组合可区分 — 返回所有该模态特征
            result: List[str] = []
            for vals in per_obj:
                result.extend(vals)
            return list(set(result))

        return []

    # ------------------------------------------------------------------
    # 查询 API
    # ------------------------------------------------------------------

    def get_crossmodal_symbols(self) -> List[str]:
        """
        返回所有跨模态符号（在非视觉模态中涌现的）。

        Returns:
            符号列表，按使用频率降序排列
        """
        sorted_syms = sorted(
            self._crossmodal_symbols.items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True,
        )
        return [sym for sym, _ in sorted_syms]

    def get_modality_contributions(self) -> Dict[str, float]:
        """
        返回各模态对消除歧义的贡献比例。

        Returns:
            modality -> 贡献比例 (0.0 ~ 1.0)
        """
        total = max(self._total_disambiguations, 1)
        return {
            mod: count / total
            for mod, count in self._modality_contributions.items()
        }

    def get_symbol_modality(self, symbol: str) -> Optional[str]:
        """获取符号所属的模态"""
        return self._symbol_modality_map.get(symbol)

    def get_modality_weights(self) -> Dict[str, float]:
        """获取当前可学习的模态权重"""
        weights = torch.softmax(self._modality_weights, dim=0)
        return {
            name: weights[i].item()
            for i, name in enumerate(self._modality_names)
        }

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """序列化模块状态"""
        # 将物体序列化为特征字典
        objects_data = [
            {'features': obj.features, 'id': obj.obj_id}
            for obj in self._objects
        ]
        return {
            'objects': objects_data,
            'modality_weights': self._modality_weights.detach().cpu(),
            'crossmodal_symbols': dict(self._crossmodal_symbols),
            'modality_contributions': dict(self._modality_contributions),
            'total_disambiguations': self._total_disambiguations,
            'symbol_modality_map': dict(self._symbol_modality_map),
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复模块状态"""
        # 恢复物体库
        self._objects = []
        for obj_data in state.get('objects', []):
            self._objects.append(CrossModalObject(
                features=obj_data['features'],
                obj_id=obj_data.get('id', ''),
            ))

        if 'modality_weights' in state:
            self._modality_weights = state['modality_weights'].to(self.device)

        self._crossmodal_symbols = state.get('crossmodal_symbols', {})
        self._modality_contributions = state.get('modality_contributions', {
            m: 0 for m in MODALITY_TYPES
        })
        self._total_disambiguations = state.get('total_disambiguations', 0)
        self._symbol_modality_map = state.get('symbol_modality_map', {})
