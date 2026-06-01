"""
生产级系统测试

测试新的生产级组件
"""

import pytest
import tempfile
import os


class TestDatabase:
    """数据库测试"""

    def test_sqlite_connection(self):
        """测试SQLite连接"""
        # Arrange
        from src.production.infrastructure.persistence.database import SQLiteConnection

        # 使用临时数据库
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = SQLiteConnection(db_path)

            # Act
            db.connect()

            # Assert
            assert db.is_connected()

            # 创建表
            db.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
            """)

            # 插入数据
            db.execute("INSERT INTO test_table (id, name) VALUES (?, ?)", {'id': 1, 'name': 'test'})

            # 查询数据
            result = db.fetch_one("SELECT * FROM test_table WHERE id = ?", {'id': 1})
            assert result is not None
            assert result['name'] == 'test'

            # 清理
            db.disconnect()
        finally:
            os.unlink(db_path)

    def test_sqlite_transaction(self):
        """测试SQLite事务"""
        # Arrange
        from src.production.infrastructure.persistence.database import SQLiteConnection

        db = SQLiteConnection(":memory:")
        db.connect()

        # 创建表
        db.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)

        # Act - 事务成功
        with db.transaction():
            db.execute("INSERT INTO test_table (id, name) VALUES (?, ?)", {'id': 1, 'name': 'test1'})
            db.execute("INSERT INTO test_table (id, name) VALUES (?, ?)", {'id': 2, 'name': 'test2'})

        # Assert
        result = db.fetch_all("SELECT * FROM test_table")
        assert len(result) == 2

        # 清理
        db.disconnect()


class TestSQLKnowledgeRepository:
    """SQL知识库仓储测试"""

    def test_save_and_find(self):
        """测试保存和查找"""
        # Arrange
        from src.production.infrastructure.persistence.database import SQLiteConnection, DatabaseManager
        from src.production.infrastructure.persistence.sql_knowledge_repository import SQLKnowledgeRepository
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        db = SQLiteConnection(":memory:")
        db_manager = DatabaseManager(db)
        db_manager.initialize()

        repo = SQLKnowledgeRepository(db)
        kb = KnowledgeBase(name="test_kb")

        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )
        kb.add_fact(fact)

        # Act
        repo.save(kb)
        found = repo.find_by_name("test_kb")

        # Assert
        assert found is not None
        assert found.name == "test_kb"
        assert found.fact_count == 1

        # 清理
        db_manager.close()

    def test_query_facts(self):
        """测试查询事实"""
        # Arrange
        from src.production.infrastructure.persistence.database import SQLiteConnection, DatabaseManager
        from src.production.infrastructure.persistence.sql_knowledge_repository import SQLKnowledgeRepository
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        db = SQLiteConnection(":memory:")
        db_manager = DatabaseManager(db)
        db_manager.initialize()

        repo = SQLKnowledgeRepository(db)
        kb = KnowledgeBase(name="test_kb")

        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )
        kb.add_fact(fact)
        repo.save(kb)

        # Act
        results = repo.query_facts("test_kb", subject="水")

        # Assert
        assert len(results) == 1
        assert results[0].statement == "水在100度沸腾"

        # 清理
        db_manager.close()


class TestLogger:
    """日志系统测试"""

    def test_logger_factory(self):
        """测试日志工厂"""
        # Arrange
        from src.production.infrastructure.logging.logger import LoggerFactory

        # Act
        logger = LoggerFactory.get_logger("test")

        # Assert
        assert logger is not None
        assert logger.name == "test"


class TestConfigManager:
    """配置管理器测试"""

    def test_config_from_env(self):
        """测试从环境变量加载配置"""
        # Arrange
        from src.production.infrastructure.config.config_manager import ConfigManager

        # 设置环境变量
        os.environ["DATABASE_TYPE"] = "sqlite"
        os.environ["API_PORT"] = "9000"

        try:
            manager = ConfigManager()

            # Act
            manager.load()

            # Assert
            assert manager.config.database.type == "sqlite"
            assert manager.config.api.port == 9000
        finally:
            # 清理环境变量
            del os.environ["DATABASE_TYPE"]
            del os.environ["API_PORT"]


class TestExceptions:
    """异常测试"""

    def test_not_found_exception(self):
        """测试NotFoundException"""
        # Arrange & Act
        from src.production.infrastructure.exceptions.exceptions import NotFoundException

        exc = NotFoundException("KnowledgeBase", "test_kb")

        # Assert
        assert exc.error_code == "NOT_FOUND"
        assert exc.status_code == 404
        assert "test_kb" in exc.message

    def test_validation_exception(self):
        """测试ValidationException"""
        # Arrange & Act
        from src.production.infrastructure.exceptions.exceptions import ValidationException

        exc = ValidationException("Invalid value", field="confidence", value=1.5)

        # Assert
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.status_code == 400
        assert exc.field == "confidence"


class TestThreadPool:
    """线程池测试"""

    def test_thread_pool_submit(self):
        """测试线程池提交任务"""
        # Arrange
        from src.production.infrastructure.concurrency.thread_pool import ThreadPool

        pool = ThreadPool(max_workers=2)

        def task(x):
            return x * 2

        # Act
        future = pool.submit(task, 5)
        result = future.result()

        # Assert
        assert result == 10

        # 清理
        pool.shutdown()


class TestRateLimiter:
    """速率限制器测试"""

    def test_rate_limiter_allow(self):
        """测试速率限制器允许请求"""
        # Arrange
        from src.production.infrastructure.concurrency.thread_pool import RateLimiter

        limiter = RateLimiter(max_requests=5, time_window=60)

        # Act & Assert
        for _ in range(5):
            assert limiter.allow_request() == True

        assert limiter.allow_request() == False
        assert limiter.get_remaining() == 0


class TestApplication:
    """应用测试"""

    def test_application_initialize(self):
        """测试应用初始化"""
        # Arrange
        from src.production.application.app import Application

        # 使用临时数据库
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # 设置环境变量
            os.environ["DATABASE_TYPE"] = "sqlite"
            os.environ["DATABASE_NAME"] = db_path

            app = Application()

            # Act
            app.initialize()

            # Assert
            assert app._initialized == True
            assert app.health_check()['status'] == 'healthy'

            # 清理
            app.shutdown()
        finally:
            os.unlink(db_path)
            del os.environ["DATABASE_TYPE"]
            del os.environ["DATABASE_NAME"]

    def test_application_services(self):
        """测试应用服务"""
        # Arrange
        from src.production.application.app import Application

        # 使用临时数据库
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # 设置环境变量
            os.environ["DATABASE_TYPE"] = "sqlite"
            os.environ["DATABASE_NAME"] = db_path

            app = Application()
            app.initialize()

            # Act
            kb_service = app.get_knowledge_service()
            reasoning_service = app.get_reasoning_service()
            query_service = app.get_query_service()

            # Assert
            assert kb_service is not None
            assert reasoning_service is not None
            assert query_service is not None

            # 清理
            app.shutdown()
        finally:
            os.unlink(db_path)
            del os.environ["DATABASE_TYPE"]
            del os.environ["DATABASE_NAME"]
