"""
配置管理系统 - 生产级实现

支持环境变量、配置文件、配置验证等功能
"""

import os
import json
from typing import Dict, Any, Optional, Type, TypeVar
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class DatabaseConfig:
    """数据库配置"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "commonsense"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    max_connections: int = 10


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    dir: str = "logs"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True


@dataclass
class AppConfig:
    """应用配置"""
    name: str = "commonsense-reasoning"
    version: str = "1.0.0"
    environment: str = "development"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log: LogConfig = field(default_factory=LogConfig)


class ConfigManager:
    """配置管理器

    职责：
    - 加载配置（环境变量、配置文件）
    - 配置验证
    - 配置访问

    Attributes:
        config: 应用配置
    """

    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None

    def __new__(cls) -> 'ConfigManager':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def config(self) -> AppConfig:
        """获取配置"""
        if self._config is None:
            self.load()
        return self._config

    def load(self, config_file: Optional[str] = None) -> None:
        """
        加载配置

        Args:
            config_file: 配置文件路径（可选）
        """
        # 从环境变量加载
        env_config = self._load_from_env()

        # 从配置文件加载
        file_config = {}
        if config_file:
            file_config = self._load_from_file(config_file)

        # 合并配置（环境变量优先）
        merged_config = {**file_config, **env_config}

        # 创建配置对象
        self._config = self._create_config(merged_config)

        logger.info(f"Configuration loaded: environment={self._config.environment}")

    def _load_from_env(self) -> Dict[str, Any]:
        """
        从环境变量加载配置

        Returns:
            配置字典
        """
        config = {}

        # 数据库配置
        if os.getenv("DATABASE_TYPE"):
            config["database_type"] = os.getenv("DATABASE_TYPE")
        if os.getenv("DATABASE_HOST"):
            config["database_host"] = os.getenv("DATABASE_HOST")
        if os.getenv("DATABASE_PORT"):
            config["database_port"] = int(os.getenv("DATABASE_PORT"))
        if os.getenv("DATABASE_NAME"):
            config["database_name"] = os.getenv("DATABASE_NAME")
        if os.getenv("DATABASE_USER"):
            config["database_user"] = os.getenv("DATABASE_USER")
        if os.getenv("DATABASE_PASSWORD"):
            config["database_password"] = os.getenv("DATABASE_PASSWORD")

        # Redis配置
        if os.getenv("REDIS_HOST"):
            config["redis_host"] = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            config["redis_port"] = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            config["redis_password"] = os.getenv("REDIS_PASSWORD")

        # API配置
        if os.getenv("API_HOST"):
            config["api_host"] = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config["api_port"] = int(os.getenv("API_PORT"))
        if os.getenv("API_DEBUG"):
            config["api_debug"] = os.getenv("API_DEBUG").lower() == "true"

        # 日志配置
        if os.getenv("LOG_LEVEL"):
            config["log_level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_DIR"):
            config["log_dir"] = os.getenv("LOG_DIR")

        # 环境
        if os.getenv("ENVIRONMENT"):
            config["environment"] = os.getenv("ENVIRONMENT")

        return config

    def _load_from_file(self, config_file: str) -> Dict[str, Any]:
        """
        从配置文件加载配置

        Args:
            config_file: 配置文件路径

        Returns:
            配置字典
        """
        try:
            path = Path(config_file)
            if not path.exists():
                logger.warning(f"Config file not found: {config_file}")
                return {}

            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            logger.info(f"Loaded config from file: {config_file}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {}

    def _create_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """
        创建配置对象

        Args:
            config_dict: 配置字典

        Returns:
            配置对象
        """
        # 数据库配置
        database = DatabaseConfig(
            type=config_dict.get("database_type", "sqlite"),
            host=config_dict.get("database_host", "localhost"),
            port=config_dict.get("database_port", 5432),
            database=config_dict.get("database_name", "commonsense"),
            user=config_dict.get("database_user", "postgres"),
            password=config_dict.get("database_password", ""),
        )

        # Redis配置
        redis = RedisConfig(
            host=config_dict.get("redis_host", "localhost"),
            port=config_dict.get("redis_port", 6379),
            password=config_dict.get("redis_password", ""),
        )

        # API配置
        api = APIConfig(
            host=config_dict.get("api_host", "0.0.0.0"),
            port=config_dict.get("api_port", 8000),
            debug=config_dict.get("api_debug", False),
        )

        # 日志配置
        log = LogConfig(
            level=config_dict.get("log_level", "INFO"),
            dir=config_dict.get("log_dir", "logs"),
        )

        return AppConfig(
            name=config_dict.get("name", "commonsense-reasoning"),
            version=config_dict.get("version", "1.0.0"),
            environment=config_dict.get("environment", "development"),
            database=database,
            redis=redis,
            api=api,
            log=log,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键（支持点号分隔，如"database.host"）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default

        return value

    def validate(self) -> bool:
        """
        验证配置

        Returns:
            是否有效
        """
        config = self.config

        # 验证数据库配置
        if config.database.type not in ["sqlite", "postgresql"]:
            logger.error(f"Invalid database type: {config.database.type}")
            return False

        # 验证端口范围
        if not (1 <= config.api.port <= 65535):
            logger.error(f"Invalid API port: {config.api.port}")
            return False

        # 验证日志级别
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log.level.upper() not in valid_levels:
            logger.error(f"Invalid log level: {config.log.level}")
            return False

        return True


# 全局配置管理器实例
config_manager = ConfigManager()
