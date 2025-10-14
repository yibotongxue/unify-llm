from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from logging import Handler
from typing import Any


class Logger:
    """一个封装了 Python logging 模块的日志记录器"""

    def __init__(
        self,
        name: str = "app",
        level: str = "INFO",
        console: bool = True,
        log_dir: str = "logs",
        log_file: str | None = None,
        file_level: str = "DEBUG",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        json_format: bool = False,
    ):
        """
        初始化日志记录器

        :param name: 日志记录器名称
        :param level: 控制台日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        :param console: 是否启用控制台输出
        :param log_file: 日志文件路径，如为None则不写入文件
        :param file_level: 文件日志级别
        :param max_bytes: 日志文件最大字节数 (用于日志轮转)
        :param backup_count: 保留的备份日志文件数量
        :param json_format: 是否使用JSON格式输出日志
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 设置最低级别，由handler过滤

        # 清除现有处理器，避免重复添加
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 创建处理器
        handlers: list[Handler] = []

        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level))
            handlers.append(console_handler)

        if not log_file:
            log_file = f"{log_dir}/{name}.log"

        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 使用 RotatingFileHandler 实现日志轮转
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, file_level))
        handlers.append(file_handler)

        # 设置日志格式
        if json_format:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"module": "%(module)s", "function": "%(funcName)s", '
                '"line": %(lineno)d, "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
            )

        # 应用格式到所有处理器
        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 添加异常捕获钩子
        sys.excepthook = self.handle_exception

    def handle_exception(self, exc_type, exc_value, exc_traceback) -> None:  # type: ignore [no-untyped-def]
        """捕获未处理的异常并记录到日志"""
        self.logger.error("未处理的异常", exc_info=(exc_type, exc_value, exc_traceback))

    def debug(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """记录调试信息"""
        self.logger.debug(msg, extra=extra)

    def info(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """记录一般信息"""
        self.logger.info(msg, extra=extra)

    def warning(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """记录警告信息"""
        self.logger.warning(msg, extra=extra)

    def error(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """记录错误信息"""
        self.logger.error(msg, extra=extra)

    def critical(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """记录严重错误信息"""
        self.logger.critical(msg, extra=extra)

    def exception(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(msg, extra=extra)

    def log(self, level: int, msg: str, extra: dict[str, Any] | None = None) -> None:
        """通用日志记录方法"""
        self.logger.log(level, msg, extra=extra)


app_logger = Logger(
    name="my_app",
    level="DEBUG",
    log_file="logs/app.log",
    max_bytes=5 * 1024 * 1024,  # 5MB
    backup_count=7,
)
