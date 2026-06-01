"""
日志系统 - 生产级实现

提供结构化日志、日志级别、日志轮转等功能
"""

import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录

        Args:
            record: 日志记录

        Returns:
            格式化后的日志字符串
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # 添加异常信息
        if record.exc_info and record.exc_info[0]:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data

        return json.dumps(log_data, ensure_ascii=False)


class LoggerFactory:
    """日志工厂"""

    _loggers: Dict[str, logging.Logger] = {}
    _initialized: bool = False

    @classmethod
    def initialize(
        cls,
        log_level: str = "INFO",
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True,
        file_output: bool = True
    ) -> None:
        """
        初始化日志系统

        Args:
            log_level: 日志级别
            log_dir: 日志目录
            max_bytes: 单个日志文件最大字节数
            backup_count: 备份文件数量
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        if cls._initialized:
            return

        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 设置根日志级别
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        # 清除现有处理器
        root_logger.handlers.clear()

        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # 文件处理器（结构化JSON格式）
        if file_output:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path / "app.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(file_handler)

            # 错误日志单独文件
            error_handler = logging.handlers.RotatingFileHandler(
                filename=log_path / "error.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(error_handler)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取日志器

        Args:
            name: 日志器名称

        Returns:
            日志器实例
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]


class ApplicationLogger:
    """应用日志器"""

    def __init__(self, name: str):
        """
        初始化应用日志器

        Args:
            name: 日志器名称
        """
        self.logger = LoggerFactory.get_logger(name)

    def info(self, message: str, **kwargs) -> None:
        """记录信息日志"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """记录错误日志"""
        self._log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""
        self._log(logging.DEBUG, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误日志"""
        self._log(logging.CRITICAL, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs) -> None:
        """
        记录日志

        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外数据
        """
        extra = {}
        if kwargs:
            extra['extra_data'] = kwargs

        self.logger.log(level, message, extra=extra)


# 全局日志器实例
app_logger = ApplicationLogger("app")
api_logger = ApplicationLogger("api")
db_logger = ApplicationLogger("db")
service_logger = ApplicationLogger("service")
