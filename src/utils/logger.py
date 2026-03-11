"""
Logging utilities for the AI Current Affairs Video Generator
"""

import sys
from pathlib import Path
from loguru import logger

# Global logger instance
_logger_configured = False


def setup_logger(
    log_level: str = "INFO",
    log_file: str = None,
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    Configure the global logger with console and optional file output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotation: When to rotate log file
        retention: How long to keep old log files
    """
    global _logger_configured

    if _logger_configured:
        return

    # Remove default handler
    logger.remove()

    # Console handler with colorful output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

    _logger_configured = True
    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: str = None):
    """
    Get a logger instance.

    Args:
        name: Logger name (module name)

    Returns:
        Logger instance
    """
    if not _logger_configured:
        setup_logger()

    if name:
        return logger.bind(name=name)
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to any class"""

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
