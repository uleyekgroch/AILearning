"""可序列化 mixin — 减少重复的 save_state/load_state 样板代码"""

import torch
from typing import Any


class Serializable:
    """默认 save_state/load_state 实现

    自动序列化所有非私有、非方法属性。
    子类可覆盖 save_state/load_state 进行自定义序列化。
    """

    def save_state(self) -> dict:
        state = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            state[key] = self._serialize_value(value)
        return state

    def load_state(self, state: dict) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, self._deserialize_value(value))

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {'__tensor__': True, 'data': value.detach().cpu().tolist()}
        if hasattr(value, 'save_state'):
            return {'__serializable__': True, 'data': value.save_state()}
        return value

    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get('__tensor__'):
                return torch.tensor(value['data'])
            if value.get('__serializable__'):
                return value['data']  # caller handles reconstruction
        return value
