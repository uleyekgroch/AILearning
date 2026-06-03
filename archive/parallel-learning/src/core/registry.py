"""能力注册表 — 替代 31 个懒初始化属性"""

from typing import Any, Callable, Dict, Optional


class ModuleRegistry:
    """懒初始化模块注册表

    替代 Learner 中的 31 个 _module = None + @property 模式。
    新增模块只需 register() 一行。
    """

    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}

    def register(self, name: str, factory: Callable) -> None:
        """注册模块工厂"""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """获取模块（懒初始化）"""
        if name not in self._instances:
            if name not in self._factories:
                raise KeyError(f"Module '{name}' not registered")
            self._instances[name] = self._factories[name]()
        return self._instances[name]

    def has(self, name: str) -> bool:
        """模块是否已初始化"""
        return name in self._instances

    def set(self, name: str, instance: Any) -> None:
        """直接设置模块实例（用于 load 恢复）"""
        self._instances[name] = instance

    @property
    def initialized(self) -> Dict[str, Any]:
        """所有已初始化的模块"""
        return dict(self._instances)
