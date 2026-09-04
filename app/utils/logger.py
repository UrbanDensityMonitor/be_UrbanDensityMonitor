"""
Logging configuration utilities
Centralized logging setup dengan consistent format
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup logger dengan consistent format
    
    Args:
        name: Logger name (biasanya __name__)
        log_level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_to_file: Whether to log to file
        log_dir: Directory untuk log files
    
    Returns:
        Configured logger
    
    Usage:
        from app.utils.logger import setup_logger
        
        logger = setup_logger(__name__)
        logger.info("Starting service...")
        logger.error("Error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # ═══════════════════════════════════════════════════════════
    # Console Handler (colored output)
    # ═══════════════════════════════════════════════════════════
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # ═══════════════════════════════════════════════════════════
    # File Handler (optional)
    # ═══════════════════════════════════════════════════════════
    if log_to_file:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create file handler
        log_file = log_path / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LoggerContext:
    """
    Context manager untuk temporary log level change
    
    Usage:
        with LoggerContext("app.services", "DEBUG"):
            # This code will have DEBUG logging
            service.do_something()
        # Back to original log level
    """
    
    def __init__(self, logger_name: str, temp_level: str):
        self.logger = logging.getLogger(logger_name)
        self.temp_level = getattr(logging, temp_level.upper())
        self.original_level = None
    
    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.temp_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
