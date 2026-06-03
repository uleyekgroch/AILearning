"""
ConfigCenter配置中心

负责配置管理
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
import threading


@dataclass
class ConfigWatcher:
    """配置监听器"""
    key: str
    callback: Callable[[str, Any, Any], None]


class ConfigCenter:
    """配置中心

    职责：
    - 配置设置
    - 配置获取
    - 配置监听
    - 配置变更通知

    Attributes:
        configs: 配置存储
        watchers: 监听器列表
        lock: 线程锁
    """

    def __init__(self):
        """初始化配置中心"""
        self.configs: Dict[str, Any] = {}
        self.watchers: List[ConfigWatcher] = []
        self.lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        """设置配置

        Args:
            key: 配置键
            value: 配置值
        """
        with self.lock:
            old_value = self.configs.get(key)
            self.configs[key] = value

            # 通知监听器
            for watcher in self.watchers:
                if watcher.key == key:
                    try:
                        watcher.callback(key, old_value, value)
                    except Exception:
                        pass  # 忽略监听器异常

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值，如果不存在返回默认值
        """
        with self.lock:
            return self.configs.get(key, default)

    def delete(self, key: str) -> bool:
        """删除配置

        Args:
            key: 配置键

        Returns:
            是否删除成功
        """
        with self.lock:
            if key in self.configs:
                old_value = self.configs[key]
                del self.configs[key]

                # 通知监听器
                for watcher in self.watchers:
                    if watcher.key == key:
                        try:
                            watcher.callback(key, old_value, None)
                        except Exception:
                            pass

                return True
            return False

    def exists(self, key: str) -> bool:
        """检查配置是否存在

        Args:
            key: 配置键

        Returns:
            是否存在
        """
        with self.lock:
            return key in self.configs

    def watch(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """监听配置变化

        Args:
            key: 配置键
            callback: 回调函数
        """
        with self.lock:
            watcher = ConfigWatcher(key=key, callback=callback)
            self.watchers.append(watcher)

    def unwatch(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """取消监听配置变化

        Args:
            key: 配置键
            callback: 回调函数
        """
        with self.lock:
            self.watchers = [
                w for w in self.watchers
                if not (w.key == key and w.callback == callback)
            ]

    def keys(self) -> List[str]:
        """获取所有配置键

        Returns:
            配置键列表
        """
        with self.lock:
            return list(self.configs.keys())

    def clear(self) -> None:
        """清空配置"""
        with self.lock:
            self.configs.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        with self.lock:
            return {
                "total_configs": len(self.configs),
                "total_watchers": len(self.watchers),
            }
