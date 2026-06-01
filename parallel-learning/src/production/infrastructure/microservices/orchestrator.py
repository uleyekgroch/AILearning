"""
MicroserviceOrchestrator微服务编排器

负责微服务编排和协调
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ServiceRegistration:
    """服务注册信息"""
    service_name: str
    service_type: str
    endpoints: List[str]
    dependencies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)


class MicroserviceOrchestrator:
    """微服务编排器

    职责：
    - 服务注册
    - 服务编排
    - 依赖管理
    - 请求协调

    Attributes:
        services: 服务注册信息
    """

    def __init__(self):
        """初始化微服务编排器"""
        self.services: Dict[str, ServiceRegistration] = {}

    def register_service(
        self,
        service_name: str,
        service_type: str,
        endpoints: List[str],
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """注册服务

        Args:
            service_name: 服务名称
            service_type: 服务类型
            endpoints: 端点列表
            dependencies: 依赖服务列表
            metadata: 额外元数据
        """
        registration = ServiceRegistration(
            service_name=service_name,
            service_type=service_type,
            endpoints=endpoints,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )

        self.services[service_name] = registration

    def unregister_service(self, service_name: str) -> bool:
        """注销服务

        Args:
            service_name: 服务名称

        Returns:
            是否注销成功
        """
        if service_name in self.services:
            del self.services[service_name]
            return True
        return False

    def orchestrate(
        self,
        request_id: str,
        service_name: str,
        method: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """编排请求

        Args:
            request_id: 请求ID
            service_name: 服务名称
            method: 方法名
            params: 参数

        Returns:
            编排结果
        """
        # 检查服务是否存在
        if service_name not in self.services:
            return {
                "status": "error",
                "error_code": "SERVICE_NOT_FOUND",
                "message": f"Service {service_name} not found"
            }

        service = self.services[service_name]

        # 简化实现：返回模拟结果
        # 实际应该：
        # 1. 检查依赖服务
        # 2. 协调多个服务调用
        # 3. 聚合结果
        return {
            "status": "success",
            "request_id": request_id,
            "service_name": service_name,
            "method": method,
            "params": params or {},
            "service_type": service.service_type,
            "endpoints": service.endpoints,
            "dependencies": service.dependencies,
            "timestamp": datetime.now().isoformat(),
        }

    def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取服务信息

        Args:
            service_name: 服务名称

        Returns:
            服务信息，如果不存在返回None
        """
        service = self.services.get(service_name)

        if service:
            return {
                "service_name": service.service_name,
                "service_type": service.service_type,
                "endpoints": service.endpoints,
                "dependencies": service.dependencies,
                "metadata": service.metadata,
                "registered_at": service.registered_at.isoformat(),
            }

        return None

    def list_services(self) -> List[str]:
        """列出所有服务名称

        Returns:
            服务名称列表
        """
        return list(self.services.keys())

    def get_dependencies(self, service_name: str) -> List[str]:
        """获取服务依赖

        Args:
            service_name: 服务名称

        Returns:
            依赖服务列表
        """
        service = self.services.get(service_name)

        if service:
            return service.dependencies

        return []

    def get_dependents(self, service_name: str) -> List[str]:
        """获取依赖此服务的服务

        Args:
            service_name: 服务名称

        Returns:
            依赖此服务的服务列表
        """
        dependents = []

        for name, service in self.services.items():
            if service_name in service.dependencies:
                dependents.append(name)

        return dependents

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        total_services = len(self.services)

        # 按类型统计
        by_type = {}
        for service in self.services.values():
            service_type = service.service_type
            by_type[service_type] = by_type.get(service_type, 0) + 1

        return {
            "total_services": total_services,
            "by_type": by_type,
        }
