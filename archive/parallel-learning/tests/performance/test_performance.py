"""
性能测试

测试系统性能指标
"""

import pytest
import time
from datetime import datetime
from typing import Dict, List, Any


# ============================================================================
# 性能测试用例
# ============================================================================

class TestKnowledgePerformance:
    """知识库性能测试"""

    def test_knowledge_creation_performance(self):
        """测试知识库创建性能"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        service = KnowledgeApplicationService()

        # Act
        start_time = time.time()

        for i in range(100):
            service.create_knowledge_base(f"kb_{i}")

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 1.0  # 应该在1秒内完成
        print(f"创建100个知识库耗时: {duration:.3f}秒")

    def test_fact_addition_performance(self):
        """测试事实添加性能"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        service = KnowledgeApplicationService()
        service.create_knowledge_base("test_kb")

        # Act
        start_time = time.time()

        for i in range(1000):
            service.add_fact("test_kb", {
                'fact_id': f'fact_{i}',
                'statement': f'测试事实 {i}',
                'subject': f'主题_{i}',
                'predicate': '是',
                'object': f'对象_{i}',
                'confidence': 0.9
            })

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 5.0  # 应该在5秒内完成
        print(f"添加1000个事实耗时: {duration:.3f}秒")
        print(f"平均每秒添加: {1000/duration:.0f}个事实")

    def test_fact_query_performance(self):
        """测试事实查询性能"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        service = KnowledgeApplicationService()
        service.create_knowledge_base("test_kb")

        # 预先添加数据
        for i in range(1000):
            service.add_fact("test_kb", {
                'fact_id': f'fact_{i}',
                'statement': f'测试事实 {i}',
                'subject': f'主题_{i % 100}',  # 100个不同主题
                'predicate': '是',
                'object': f'对象_{i}',
                'confidence': 0.9
            })

        # Act
        start_time = time.time()

        for _ in range(100):
            service.query_by_subject("test_kb", "主题_50")

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 1.0  # 应该在1秒内完成
        print(f"100次查询耗时: {duration:.3f}秒")
        print(f"平均每秒查询: {100/duration:.0f}次")


class TestReasoningPerformance:
    """推理性能测试"""

    def test_session_creation_performance(self):
        """测试会话创建性能"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        service = ReasoningApplicationService()

        # Act
        start_time = time.time()

        for i in range(100):
            service.create_session(f"session_{i}")

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 1.0  # 应该在1秒内完成
        print(f"创建100个会话耗时: {duration:.3f}秒")

    def test_task_execution_performance(self):
        """测试任务执行性能"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        service = ReasoningApplicationService()
        service.create_session("test_session")

        # 添加任务
        for i in range(100):
            service.add_task("test_session", {
                'task_id': f'task_{i}',
                'query': f'测试查询 {i}',
                'reasoning_type': 'commonsense'
            })

        # Act
        start_time = time.time()

        for i in range(100):
            service.execute_task("test_session", f"task_{i}")

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 5.0  # 应该在5秒内完成
        print(f"执行100个任务耗时: {duration:.3f}秒")
        print(f"平均每秒执行: {100/duration:.0f}个任务")


class TestQueryPerformance:
    """查询性能测试"""

    def test_query_service_performance(self):
        """测试查询服务性能"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.application.services.query_service import QueryApplicationService

        kb_service = KnowledgeApplicationService()
        query_service = QueryApplicationService(kb_service)

        # 预先添加数据
        kb_service.create_knowledge_base("test_kb")
        for i in range(1000):
            kb_service.add_fact("test_kb", {
                'fact_id': f'fact_{i}',
                'statement': f'测试事实 {i}',
                'subject': f'主题_{i % 100}',
                'predicate': '是',
                'object': f'对象_{i}',
                'confidence': 0.9
            })

        # Act
        start_time = time.time()

        for _ in range(100):
            query_service.query_commonsense("主题_50", "test_kb")

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 2.0  # 应该在2秒内完成
        print(f"100次查询耗时: {duration:.3f}秒")
        print(f"平均每秒查询: {100/duration:.0f}次")


class TestAPIPerformance:
    """API性能测试"""

    def test_api_response_time(self):
        """测试API响应时间"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI

        service = KnowledgeApplicationService()
        api = KnowledgeAPI(service)

        # 创建知识库
        api.create_knowledge_base({'name': 'test_kb'})

        # 添加数据
        for i in range(100):
            api.add_fact('test_kb', {
                'fact_id': f'fact_{i}',
                'statement': f'测试事实 {i}',
                'subject': f'主题_{i}',
                'predicate': '是',
                'object': f'对象_{i}',
                'confidence': 0.9
            })

        # Act
        start_time = time.time()

        for _ in range(100):
            api.query_facts('test_kb', {'subject': '主题_50'})

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 2.0  # 应该在2秒内完成
        print(f"100次API调用耗时: {duration:.3f}秒")
        print(f"平均响应时间: {duration/100*1000:.1f}毫秒")


class TestConcurrencyPerformance:
    """并发性能测试"""

    def test_concurrent_knowledge_operations(self):
        """测试并发知识操作"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        import threading

        service = KnowledgeApplicationService()
        service.create_knowledge_base("test_kb")

        results = []
        errors = []

        def add_facts(thread_id):
            try:
                for i in range(100):
                    service.add_fact("test_kb", {
                        'fact_id': f'fact_{thread_id}_{i}',
                        'statement': f'测试事实 {thread_id}_{i}',
                        'subject': f'主题_{thread_id}',
                        'predicate': '是',
                        'object': f'对象_{i}',
                        'confidence': 0.9
                    })
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Act
        start_time = time.time()

        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_facts, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert len(errors) == 0  # 不应该有错误
        assert len(results) == 5  # 所有线程都应该成功
        print(f"5个并发线程各添加100个事实耗时: {duration:.3f}秒")
        print(f"总添加事实数: {5*100}")


class TestMemoryPerformance:
    """内存性能测试"""

    def test_memory_usage(self):
        """测试内存使用"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        import sys

        service = KnowledgeApplicationService()
        service.create_knowledge_base("test_kb")

        # 记录初始内存
        initial_size = sys.getsizeof(service)

        # 添加数据
        for i in range(1000):
            service.add_fact("test_kb", {
                'fact_id': f'fact_{i}',
                'statement': f'测试事实 {i}',
                'subject': f'主题_{i}',
                'predicate': '是',
                'object': f'对象_{i}',
                'confidence': 0.9
            })

        # 记录最终内存
        final_size = sys.getsizeof(service)

        # 计算内存增长
        memory_growth = final_size - initial_size

        # Assert
        assert memory_growth < 10 * 1024 * 1024  # 不应该超过10MB
        print(f"添加1000个事实后内存增长: {memory_growth / 1024:.1f} KB")


class TestPerformanceBenchmark:
    """性能基准测试"""

    def test_system_benchmark(self):
        """系统性能基准测试"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.application.services.reasoning_service import ReasoningApplicationService
        from src.production.application.services.query_service import QueryApplicationService

        # 创建服务
        kb_service = KnowledgeApplicationService()
        reasoning_service = ReasoningApplicationService()
        query_service = QueryApplicationService(kb_service)

        # 准备数据
        kb_service.create_knowledge_base("benchmark_kb")
        for i in range(500):
            kb_service.add_fact("benchmark_kb", {
                'fact_id': f'fact_{i}',
                'statement': f'测试事实 {i}',
                'subject': f'主题_{i % 50}',
                'predicate': '是',
                'object': f'对象_{i}',
                'confidence': 0.9
            })

        # Act
        start_time = time.time()

        # 1. 知识查询
        for _ in range(50):
            query_service.query_commonsense("主题_25", "benchmark_kb")

        # 2. 推理任务
        reasoning_service.create_session("benchmark_session")
        for i in range(50):
            reasoning_service.add_task("benchmark_session", {
                'task_id': f'task_{i}',
                'query': f'测试查询 {i}',
                'reasoning_type': 'commonsense'
            })
            reasoning_service.execute_task("benchmark_session", f"task_{i}")

        end_time = time.time()
        duration = end_time - start_time

        # Assert
        assert duration < 10.0  # 应该在10秒内完成
        print(f"系统基准测试耗时: {duration:.3f}秒")
        print(f"平均每秒操作: {100/duration:.0f}次")
