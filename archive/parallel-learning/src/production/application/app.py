"""
应用启动器 - 生产级实现

整合所有组件，提供统一的应用入口
"""

import logging
from typing import Optional
from contextlib import contextmanager

from src.production.infrastructure.config.config_manager import ConfigManager, AppConfig
from src.production.infrastructure.logging.logger import LoggerFactory
from src.production.infrastructure.persistence.database import DatabaseFactory, DatabaseManager
from src.production.infrastructure.persistence.sql_knowledge_repository import SQLKnowledgeRepository
from src.production.infrastructure.persistence.sql_reasoning_repository import SQLReasoningRepository
from src.production.infrastructure.concurrency.thread_pool import ThreadPool, RateLimiter
from src.production.infrastructure.exceptions.exceptions import BaseException

from src.production.application.services.knowledge_service import KnowledgeApplicationService
from src.production.application.services.reasoning_service import ReasoningApplicationService
from src.production.application.services.query_service import QueryApplicationService

logger = logging.getLogger(__name__)


class Application:
    """应用类

    职责：
    - 初始化所有组件
    - 管理应用生命周期
    - 提供统一的服务访问

    Attributes:
        config: 应用配置
        db_manager: 数据库管理器
        knowledge_repo: 知识库仓储
        reasoning_repo: 推理仓储
        knowledge_service: 知识应用服务
        reasoning_service: 推理应用服务
        query_service: 查询应用服务
        thread_pool: 线程池
        rate_limiter: 速率限制器
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化应用

        Args:
            config_file: 配置文件路径（可选）
        """
        # 加载配置
        self.config_manager = ConfigManager()
        self.config_manager.load(config_file)
        self.config = self.config_manager.config

        # 初始化日志
        LoggerFactory.initialize(
            log_level=self.config.log.level,
            log_dir=self.config.log.dir,
            console_output=self.config.log.console_output,
            file_output=self.config.log.file_output
        )

        # 初始化数据库
        self.db_connection = DatabaseFactory.create_connection(
            self.config.database.type,
            host=self.config.database.host,
            port=self.config.database.port,
            database=self.config.database.database,
            user=self.config.database.user,
            password=self.config.database.password,
        )
        self.db_manager = DatabaseManager(self.db_connection)

        # 初始化仓储
        self.knowledge_repo = SQLKnowledgeRepository(self.db_connection)
        self.reasoning_repo = SQLReasoningRepository(self.db_connection)

        # 初始化应用服务
        self.knowledge_service = KnowledgeApplicationService()
        self.reasoning_service = ReasoningApplicationService()
        self.query_service = QueryApplicationService(self.knowledge_service)

        # 初始化并发组件
        self.thread_pool = ThreadPool(max_workers=10)
        self.rate_limiter = RateLimiter(max_requests=100, time_window=60)

        self._initialized = False

        logger.info(f"Application initialized: {self.config.name} v{self.config.version}")

    def initialize(self) -> None:
        """初始化应用"""
        if self._initialized:
            return

        # 初始化数据库
        self.db_manager.initialize()

        # 验证配置
        if not self.config_manager.validate():
            raise BaseException(
                message="Invalid configuration",
                error_code="CONFIG_ERROR"
            )

        self._initialized = True
        logger.info("Application initialized successfully")

    def shutdown(self) -> None:
        """关闭应用"""
        if not self._initialized:
            return

        # 关闭线程池
        self.thread_pool.shutdown()

        # 关闭数据库
        self.db_manager.close()

        self._initialized = False
        logger.info("Application shut down")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def get_knowledge_service(self) -> KnowledgeApplicationService:
        """获取知识应用服务"""
        return self.knowledge_service

    def get_reasoning_service(self) -> ReasoningApplicationService:
        """获取推理应用服务"""
        return self.reasoning_service

    def get_query_service(self) -> QueryApplicationService:
        """获取查询应用服务"""
        return self.query_service

    def get_knowledge_repo(self) -> SQLKnowledgeRepository:
        """获取知识库仓储"""
        return self.knowledge_repo

    def get_reasoning_repo(self) -> SQLReasoningRepository:
        """获取推理仓储"""
        return self.reasoning_repo

    def get_thread_pool(self) -> ThreadPool:
        """获取线程池"""
        return self.thread_pool

    def get_rate_limiter(self) -> RateLimiter:
        """获取速率限制器"""
        return self.rate_limiter

    def get_config(self) -> AppConfig:
        """获取配置"""
        return self.config

    def health_check(self) -> dict:
        """
        健康检查

        Returns:
            健康状态
        """
        return {
            'status': 'healthy',
            'version': self.config.version,
            'environment': self.config.environment,
            'database': {
                'type': self.config.database.type,
                'connected': self.db_connection.is_connected(),
            },
            'rate_limiter': {
                'remaining': self.rate_limiter.get_remaining(),
            },
        }


# 全局应用实例
_app: Optional[Application] = None


def get_app() -> Application:
    """
    获取应用实例

    Returns:
        应用实例
    """
    global _app
    if _app is None:
        _app = Application()
    return _app


def initialize_app(config_file: Optional[str] = None) -> Application:
    """
    初始化应用

    Args:
        config_file: 配置文件路径（可选）

    Returns:
        应用实例
    """
    global _app
    _app = Application(config_file)
    _app.initialize()
    return _app


def shutdown_app() -> None:
    """关闭应用"""
    global _app
    if _app:
        _app.shutdown()
        _app = None
