"""
RequestTracer请求追踪器

负责请求链路追踪
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid


@dataclass
class Span:
    """跨度"""
    span_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """追踪"""
    trace_id: str
    request_id: str
    service_name: str
    method: str
    params: Dict[str, Any]
    start_time: datetime
    end_time: datetime = None
    status: str = "pending"
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestTracer:
    """请求追踪器

    职责：
    - 请求追踪
    - 跨度记录
    - 链路分析
    - 性能统计

    Attributes:
        traces: 追踪映射
    """

    def __init__(self):
        """初始化请求追踪器"""
        self.traces: Dict[str, Trace] = {}

    def start_trace(
        self,
        request_id: str,
        service_name: str,
        method: str,
        params: Dict[str, Any] = None
    ) -> str:
        """开始追踪

        Args:
            request_id: 请求ID
            service_name: 服务名称
            method: 方法名
            params: 参数

        Returns:
            追踪ID
        """
        trace_id = str(uuid.uuid4())

        trace = Trace(
            trace_id=trace_id,
            request_id=request_id,
            service_name=service_name,
            method=method,
            params=params or {},
            start_time=datetime.now()
        )

        self.traces[trace_id] = trace

        return trace_id

    def add_span(
        self,
        trace_id: str,
        span_name: str,
        duration: float,
        status: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """添加跨度

        Args:
            trace_id: 追踪ID
            span_name: 跨度名称
            duration: 持续时间（秒）
            status: 状态
            metadata: 额外元数据
        """
        trace = self.traces.get(trace_id)

        if trace:
            span = Span(
                span_name=span_name,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=duration,
                status=status,
                metadata=metadata or {}
            )

            trace.spans.append(span)

    def end_trace(self, trace_id: str, status: str) -> None:
        """结束追踪

        Args:
            trace_id: 追踪ID
            status: 状态
        """
        trace = self.traces.get(trace_id)

        if trace:
            trace.end_time = datetime.now()
            trace.status = status

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """获取追踪信息

        Args:
            trace_id: 追踪ID

        Returns:
            追踪信息，如果不存在返回None
        """
        trace = self.traces.get(trace_id)

        if trace:
            return {
                "trace_id": trace.trace_id,
                "request_id": trace.request_id,
                "service_name": trace.service_name,
                "method": trace.method,
                "params": trace.params,
                "start_time": trace.start_time.isoformat(),
                "end_time": trace.end_time.isoformat() if trace.end_time else None,
                "status": trace.status,
                "spans": [
                    {
                        "span_name": span.span_name,
                        "duration": span.duration,
                        "status": span.status,
                        "metadata": span.metadata,
                    }
                    for span in trace.spans
                ],
                "metadata": trace.metadata,
            }

        return None

    def list_traces(self) -> List[str]:
        """列出所有追踪ID

        Returns:
            追踪ID列表
        """
        return list(self.traces.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        total_traces = len(self.traces)
        success_traces = sum(1 for t in self.traces.values() if t.status == "success")
        failed_traces = sum(1 for t in self.traces.values() if t.status == "failed")

        return {
            "total_traces": total_traces,
            "success_traces": success_traces,
            "failed_traces": failed_traces,
            "pending_traces": total_traces - success_traces - failed_traces,
        }
