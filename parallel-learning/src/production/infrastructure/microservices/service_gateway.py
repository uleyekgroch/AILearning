"""
ServiceGateway服务网关

负责请求路由和转发
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Route:
    """路由信息"""
    path: str
    service_name: str
    methods: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceGateway:
    """服务网关

    职责：
    - 请求路由
    - 请求转发
    - 负载均衡
    - 认证授权

    Attributes:
        routes: 路由映射
    """

    def __init__(self):
        """初始化服务网关"""
        self.routes: Dict[str, Route] = {}

    def register_route(
        self,
        path: str,
        service_name: str,
        methods: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """注册路由

        Args:
            path: 路由路径
            service_name: 服务名称
            methods: HTTP方法列表
            metadata: 额外元数据
        """
        route = Route(
            path=path,
            service_name=service_name,
            methods=methods or ["GET"],
            metadata=metadata or {}
        )

        self.routes[path] = route

    def unregister_route(self, path: str) -> bool:
        """注销路由

        Args:
            path: 路由路径

        Returns:
            是否注销成功
        """
        if path in self.routes:
            del self.routes[path]
            return True
        return False

    def route_request(
        self,
        path: str,
        method: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """路由请求

        Args:
            path: 路由路径
            method: HTTP方法
            params: 请求参数

        Returns:
            路由结果
        """
        # 查找路由
        route = self.routes.get(path)

        if route is None:
            return {
                "status": "error",
                "error_code": "NOT_FOUND",
                "message": f"Route {path} not found"
            }

        # 检查方法
        if method not in route.methods:
            return {
                "status": "error",
                "error_code": "METHOD_NOT_ALLOWED",
                "message": f"Method {method} not allowed for {path}"
            }

        # 简化实现：返回模拟结果
        # 实际应该转发到对应服务
        return {
            "status": "success",
            "service_name": route.service_name,
            "path": path,
            "method": method,
            "params": params or {},
            "metadata": route.metadata,
        }

    def get_route(self, path: str) -> Optional[Dict[str, Any]]:
        """获取路由信息

        Args:
            path: 路由路径

        Returns:
            路由信息，如果不存在返回None
        """
        route = self.routes.get(path)

        if route:
            return {
                "path": route.path,
                "service_name": route.service_name,
                "methods": route.methods,
                "metadata": route.metadata,
            }

        return None

    def list_routes(self) -> List[str]:
        """列出所有路由

        Returns:
            路由路径列表
        """
        return list(self.routes.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        service_names = {}
        for route in self.routes.values():
            name = route.service_name
            service_names[name] = service_names.get(name, 0) + 1

        return {
            "total_routes": len(self.routes),
            "service_names": service_names,
        }
