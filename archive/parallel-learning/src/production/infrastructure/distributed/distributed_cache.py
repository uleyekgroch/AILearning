"""
DistributedCache分布式缓存

负责分布式缓存管理
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import time


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class DistributedCache:
    """分布式缓存

    职责：
    - 缓存设置
    - 缓存获取
    - 缓存删除
    - 过期管理

    Attributes:
        cache: 缓存存储
        lock: 线程锁
    """

    def __init__(self):
        """初始化分布式缓存"""
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.Lock()

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        with self.lock:
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at
            )

            self.cache[key] = entry

    def get(self, key: str) -> Any:
        """获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或已过期返回None
        """
        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                return None

            if entry.is_expired():
                del self.cache[key]
                return None

            return entry.value

    def delete(self, key: str) -> bool:
        """删除缓存

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                return False

            if entry.is_expired():
                del self.cache[key]
                return False

            return True

    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """获取缓存大小

        Returns:
            缓存条目数量
        """
        with self.lock:
            # 清理过期条目
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            return len(self.cache)

    def keys(self) -> List[str]:
        """获取所有缓存键

        Returns:
            缓存键列表
        """
        with self.lock:
            # 清理过期条目
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            return list(self.cache.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        with self.lock:
            total = len(self.cache)
            expired = sum(1 for entry in self.cache.values() if entry.is_expired())

            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
            }
