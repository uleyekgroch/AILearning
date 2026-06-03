"""
ServiceRegistry服务注册中心

负责服务的注册、注销和发现
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ServiceInfo:
    """服务信息"""
    service_id: str
    service_name: str
    host: str
    port: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)


class ServiceRegistry:
    """服务注册中心

    职责：
    - 服务注册
    - 服务注销
    - 服务发现
    - 健康检查

    Attributes:
        services: 服务映射
    """

    def __init__(self):
        """初始化服务注册中心"""
        self.services: Dict[str, ServiceInfo] = {}

    def register_service(
        self,
        service_id: str,
        service_name: str,
        host: str,
        port: int,
        metadata: Dict[str, Any] = None
    ) -> None:
        """注册服务

        Args:
            service_id: 服务ID
            service_name: 服务名称
            host: 主机地址
            port: 端口号
            metadata: 额外元数据
        """
        service_info = ServiceInfo(
            service_id=service_id,
            service_name=service_name,
            host=host,
            port=port,
            metadata=metadata or {}
        )

        self.services[service_id] = service_info

    def deregister_service(self, service_id: str) -> bool:
        """注销服务

        Args:
            service_id: 服务ID

        Returns:
            是否注销成功
        """
        if service_id in self.services:
            del self.services[service_id]
            return True
        return False

    def discover_service(self, service_name: str) -> List[Dict[str, Any]]:
        """发现服务

        Args:
            service_name: 服务名称

        Returns:
            服务信息列表
        """
        result = []

        for service_id, service_info in self.services.items():
            if service_info.service_name == service_name:
                result.append({
                    "service_id": service_info.service_id,
                    "service_name": service_info.service_name,
                    "host": service_info.host,
                    "port": service_info.port,
                    "metadata": service_info.metadata,
                    "registered_at": service_info.registered_at.isoformat(),
                    "last_heartbeat": service_info.last_heartbeat.isoformat(),
                })

        return result

    def get_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        """获取服务信息

        Args:
            service_id: 服务ID

        Returns:
            服务信息，如果不存在返回None
        """
        service_info = self.services.get(service_id)

        if service_info:
            return {
                "service_id": service_info.service_id,
                "service_name": service_info.service_name,
                "host": service_info.host,
                "port": service_info.port,
                "metadata": service_info.metadata,
                "registered_at": service_info.registered_at.isoformat(),
                "last_heartbeat": service_info.last_heartbeat.isoformat(),
            }

        return None

    def heartbeat(self, service_id: str) -> bool:
        """更新心跳

        Args:
            service_id: 服务ID

        Returns:
            是否更新成功
        """
        if service_id in self.services:
            self.services[service_id].last_heartbeat = datetime.now()
            return True
        return False

    def list_services(self) -> List[str]:
        """列出所有服务ID

        Returns:
            服务ID列表
        """
        return list(self.services.keys())

    def list_service_names(self) -> List[str]:
        """列出所有服务名称

        Returns:
            服务名称列表（去重）
        """
        names = set()
        for service_info in self.services.values():
            names.add(service_info.service_name)
        return list(names)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        service_names = {}
        for service_info in self.services.values():
            name = service_info.service_name
            service_names[name] = service_names.get(name, 0) + 1

        return {
            "total_services": len(self.services),
            "service_names": service_names,
        }
