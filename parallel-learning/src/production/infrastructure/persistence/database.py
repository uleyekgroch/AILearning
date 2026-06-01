"""
数据库抽象层 - 生产级实现

提供真正的数据库支持，而不是内存实现
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic
from datetime import datetime
from contextlib import contextmanager
import logging

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class DatabaseConnection(ABC):
    """数据库连接抽象基类"""

    @abstractmethod
    def connect(self) -> None:
        """建立连接"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查是否连接"""
        pass

    @abstractmethod
    def execute(self, query: str, params: Dict[str, Any] = None) -> Any:
        """执行查询"""
        pass

    @abstractmethod
    def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> int:
        """批量执行查询"""
        pass

    @abstractmethod
    def fetch_one(self, query: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """获取单条记录"""
        pass

    @abstractmethod
    def fetch_many(self, query: str, params: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取多条记录"""
        pass

    @abstractmethod
    def fetch_all(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """获取所有记录"""
        pass

    @abstractmethod
    def begin_transaction(self) -> None:
        """开始事务"""
        pass

    @abstractmethod
    def commit(self) -> None:
        """提交事务"""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """回滚事务"""
        pass

    @contextmanager
    def transaction(self):
        """事务上下文管理器"""
        self.begin_transaction()
        try:
            yield self
            self.commit()
        except Exception as e:
            self.rollback()
            logger.error(f"Transaction failed: {e}")
            raise


class SQLiteConnection(DatabaseConnection):
    """SQLite数据库连接"""

    def __init__(self, db_path: str = ":memory:"):
        """
        初始化SQLite连接

        Args:
            db_path: 数据库文件路径，":memory:"表示内存数据库
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def connect(self) -> None:
        """建立连接"""
        import sqlite3
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            logger.info(f"Connected to SQLite database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    def disconnect(self) -> None:
        """断开连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Disconnected from SQLite database")

    def is_connected(self) -> bool:
        """检查是否连接"""
        return self.connection is not None

    def execute(self, query: str, params: Dict[str, Any] = None) -> Any:
        """执行查询"""
        try:
            if params:
                # SQLite使用?作为占位符，需要将字典转换为元组
                if '?' in query:
                    # 将字典值转换为元组
                    values = tuple(params.values())
                    result = self.cursor.execute(query, values)
                else:
                    # 使用命名占位符
                    result = self.cursor.execute(query, params)
            else:
                result = self.cursor.execute(query)
            self.connection.commit()
            return result
        except Exception as e:
            logger.error(f"Execute failed: {query}, error: {e}")
            raise

    def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> int:
        """批量执行查询"""
        try:
            count = 0
            for params in params_list:
                self.cursor.execute(query, params)
                count += 1
            self.connection.commit()
            return count
        except Exception as e:
            logger.error(f"Execute many failed: {query}, error: {e}")
            raise

    def fetch_one(self, query: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """获取单条记录"""
        try:
            if params:
                # SQLite使用?作为占位符，需要将字典转换为元组
                if '?' in query:
                    values = tuple(params.values())
                    self.cursor.execute(query, values)
                else:
                    self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Fetch one failed: {query}, error: {e}")
            raise

    def fetch_many(self, query: str, params: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取多条记录"""
        try:
            if params:
                # SQLite使用?作为占位符，需要将字典转换为元组
                if '?' in query:
                    values = tuple(params.values())
                    self.cursor.execute(query, values)
                else:
                    self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            rows = self.cursor.fetchmany(limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Fetch many failed: {query}, error: {e}")
            raise

    def fetch_all(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """获取所有记录"""
        try:
            if params:
                # SQLite使用?作为占位符，需要将字典转换为元组
                if '?' in query:
                    values = tuple(params.values())
                    self.cursor.execute(query, values)
                else:
                    self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Fetch all failed: {query}, error: {e}")
            raise

    def begin_transaction(self) -> None:
        """开始事务"""
        self.execute("BEGIN TRANSACTION")

    def commit(self) -> None:
        """提交事务"""
        self.connection.commit()

    def rollback(self) -> None:
        """回滚事务"""
        self.connection.rollback()


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL数据库连接"""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """
        初始化PostgreSQL连接

        Args:
            host: 主机地址
            port: 端口号
            database: 数据库名
            user: 用户名
            password: 密码
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.cursor = None

    def connect(self) -> None:
        """建立连接"""
        import psycopg2
        import psycopg2.extras
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            logger.info(f"Connected to PostgreSQL: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def disconnect(self) -> None:
        """断开连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Disconnected from PostgreSQL")

    def is_connected(self) -> bool:
        """检查是否连接"""
        return self.connection is not None and not self.connection.closed

    def execute(self, query: str, params: Dict[str, Any] = None) -> Any:
        """执行查询"""
        try:
            if params:
                result = self.cursor.execute(query, params)
            else:
                result = self.cursor.execute(query)
            self.connection.commit()
            return result
        except Exception as e:
            logger.error(f"Execute failed: {query}, error: {e}")
            raise

    def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> int:
        """批量执行查询"""
        try:
            count = 0
            for params in params_list:
                self.cursor.execute(query, params)
                count += 1
            self.connection.commit()
            return count
        except Exception as e:
            logger.error(f"Execute many failed: {query}, error: {e}")
            raise

    def fetch_one(self, query: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """获取单条记录"""
        try:
            if params:
                # SQLite使用?作为占位符，需要将字典转换为元组
                if '?' in query:
                    values = tuple(params.values())
                    self.cursor.execute(query, values)
                else:
                    self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Fetch one failed: {query}, error: {e}")
            raise

    def fetch_many(self, query: str, params: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取多条记录"""
        try:
            if params:
                # SQLite使用?作为占位符，需要将字典转换为元组
                if '?' in query:
                    values = tuple(params.values())
                    self.cursor.execute(query, values)
                else:
                    self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            rows = self.cursor.fetchmany(limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Fetch many failed: {query}, error: {e}")
            raise

    def fetch_all(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """获取所有记录"""
        try:
            if params:
                # SQLite使用?作为占位符，需要将字典转换为元组
                if '?' in query:
                    values = tuple(params.values())
                    self.cursor.execute(query, values)
                else:
                    self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Fetch all failed: {query}, error: {e}")
            raise

    def begin_transaction(self) -> None:
        """开始事务"""
        self.execute("BEGIN TRANSACTION")

    def commit(self) -> None:
        """提交事务"""
        self.connection.commit()

    def rollback(self) -> None:
        """回滚事务"""
        self.connection.rollback()


class DatabaseFactory:
    """数据库工厂"""

    @staticmethod
    def create_connection(db_type: str, **kwargs) -> DatabaseConnection:
        """
        创建数据库连接

        Args:
            db_type: 数据库类型（sqlite/postgresql）
            **kwargs: 连接参数

        Returns:
            数据库连接实例
        """
        if db_type == "sqlite":
            return SQLiteConnection(kwargs.get("db_path", ":memory:"))
        elif db_type == "postgresql":
            return PostgreSQLConnection(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 5432),
                database=kwargs.get("database", "commonsense"),
                user=kwargs.get("user", "postgres"),
                password=kwargs.get("password", "")
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, connection: DatabaseConnection):
        """
        初始化数据库管理器

        Args:
            connection: 数据库连接
        """
        self.connection = connection
        self._initialized = False

    def initialize(self) -> None:
        """初始化数据库"""
        if not self._initialized:
            self.connection.connect()
            self._create_tables()
            self._initialized = True
            logger.info("Database initialized")

    def _create_tables(self) -> None:
        """创建表结构"""
        # 知识库表
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 事实表
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL,
                statement TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                fact_type TEXT DEFAULT 'physical',
                source TEXT DEFAULT 'manual',
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id)
            )
        """)

        # 推理会话表
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_sessions (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'created',
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 推理任务表
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                reasoning_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES reasoning_sessions(id)
            )
        """)

        # 推理结果表
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_results (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                status TEXT DEFAULT 'completed',
                reasoning_chain TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES reasoning_tasks(id)
            )
        """)

        # 创建索引
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_facts_kb ON facts(knowledge_base_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON reasoning_tasks(session_id)")

        logger.info("Database tables created")

    def close(self) -> None:
        """关闭数据库连接"""
        if self._initialized:
            self.connection.disconnect()
            self._initialized = False
            logger.info("Database connection closed")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
