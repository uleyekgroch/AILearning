"""
LoadBalancer负载均衡器

负责服务实例的选择和负载均衡
"""

from typing import Dict, List, Any, Optional
import random


class LoadBalancer:
    """负载均衡器

    职责：
    - 服务选择
    - 负载均衡策略
    - 健康检查

    Attributes:
        round_robin_index: 轮询索引
    """

    def __init__(self):
        """初始化负载均衡器"""
        self.round_robin_index = 0

    def select_service(
        self,
        services: List[Dict[str, Any]],
        strategy: str = "round_robin"
    ) -> Optional[Dict[str, Any]]:
        """选择服务实例

        Args:
            services: 服务实例列表
            strategy: 负载均衡策略（round_robin/random/least_connections）

        Returns:
            选中的服务实例，如果列表为空返回None
        """
        if not services:
            return None

        if strategy == "round_robin":
            return self._round_robin(services)
        elif strategy == "random":
            return self._random(services)
        elif strategy == "least_connections":
            return self._least_connections(services)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _round_robin(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """轮询策略

        Args:
            services: 服务实例列表

        Returns:
            选中的服务实例
        """
        service = services[self.round_robin_index % len(services)]
        self.round_robin_index += 1
        return service

    def _random(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """随机策略

        Args:
            services: 服务实例列表

        Returns:
            选中的服务实例
        """
        return random.choice(services)

    def _least_connections(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """最少连接策略

        Args:
            services: 服务实例列表

        Returns:
            选中的服务实例
        """
        # 简化实现：返回第一个服务
        # 实际应该根据连接数选择
        return services[0]

    def reset(self) -> None:
        """重置负载均衡器"""
        self.round_robin_index = 0
