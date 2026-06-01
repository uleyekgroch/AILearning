"""
线程池 - 生产级实现

提供真正的并发处理能力
"""

import threading
import queue
from typing import Callable, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future
import logging

logger = logging.getLogger(__name__)


class ThreadPool:
    """线程池

    职责：
    - 管理线程池
    - 提交任务
    - 获取结果

    Attributes:
        max_workers: 最大线程数
        executor: 线程池执行器
    """

    def __init__(self, max_workers: int = 10):
        """
        初始化线程池

        Args:
            max_workers: 最大线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: List[Future] = []
        logger.info(f"Thread pool initialized with {max_workers} workers")

    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """
        提交任务

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Future对象
        """
        future = self.executor.submit(func, *args, **kwargs)
        self._futures.append(future)
        return future

    def map(self, func: Callable, iterable) -> List[Any]:
        """
        映射函数到可迭代对象

        Args:
            func: 要执行的函数
            iterable: 可迭代对象

        Returns:
            结果列表
        """
        return list(self.executor.map(func, iterable))

    def shutdown(self, wait: bool = True) -> None:
        """
        关闭线程池

        Args:
            wait: 是否等待所有任务完成
        """
        self.executor.shutdown(wait=wait)
        logger.info("Thread pool shut down")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class AsyncExecutor:
    """异步执行器

    职责：
    - 异步执行任务
    - 任务队列管理
    - 结果收集

    Attributes:
        max_workers: 最大线程数
        task_queue: 任务队列
        result_queue: 结果队列
        workers: 工作线程列表
    """

    def __init__(self, max_workers: int = 5):
        """
        初始化异步执行器

        Args:
            max_workers: 最大线程数
        """
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """启动执行器"""
        with self._lock:
            if self._running:
                return

            self._running = True

            # 创建工作线程
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"AsyncWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)

            logger.info(f"Async executor started with {self.max_workers} workers")

    def stop(self) -> None:
        """停止执行器"""
        with self._lock:
            if not self._running:
                return

            self._running = False

            # 发送停止信号
            for _ in range(self.max_workers):
                self.task_queue.put(None)

            # 等待工作线程结束
            for worker in self.workers:
                worker.join(timeout=5)

            self.workers.clear()
            logger.info("Async executor stopped")

    def submit(self, func: Callable, *args, **kwargs) -> None:
        """
        提交任务

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
        """
        task = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
        }
        self.task_queue.put(task)

    def get_result(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        获取结果

        Args:
            timeout: 超时时间（秒）

        Returns:
            任务结果，如果超时返回None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self) -> None:
        """工作线程循环"""
        while self._running:
            try:
                # 获取任务
                task = self.task_queue.get(timeout=1)

                # 检查停止信号
                if task is None:
                    break

                # 执行任务
                try:
                    result = task['func'](*task['args'], **task['kwargs'])
                    self.result_queue.put({
                        'success': True,
                        'result': result,
                    })
                except Exception as e:
                    self.result_queue.put({
                        'success': False,
                        'error': str(e),
                    })

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class RateLimiter:
    """速率限制器

    职责：
    - 限制请求速率
    - 防止过载

    Attributes:
        max_requests: 最大请求数
        time_window: 时间窗口（秒）
        requests: 请求记录
    """

    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        初始化速率限制器

        Args:
            max_requests: 最大请求数
            time_window: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """
        检查是否允许请求

        Returns:
            是否允许
        """
        import time

        with self._lock:
            now = time.time()

            # 清理过期请求
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.time_window
            ]

            # 检查是否超过限制
            if len(self.requests) >= self.max_requests:
                return False

            # 记录请求
            self.requests.append(now)
            return True

    def get_remaining(self) -> int:
        """
        获取剩余请求数

        Returns:
            剩余请求数
        """
        import time

        with self._lock:
            now = time.time()

            # 清理过期请求
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.time_window
            ]

            return max(0, self.max_requests - len(self.requests))
