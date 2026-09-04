"""
Logging configuration for Traffic State Discovery.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog


class ProjectLogger:
    """
    Custom logger with colored output and file logging.
    """
    
    _instance: Optional['ProjectLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls) -> 'ProjectLogger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with handlers."""
        self._logger = logging.getLogger("TrafficStateDiscovery")
        self._logger.setLevel(logging.DEBUG)
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Color format
        console_format = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_format)
        self._logger.addHandler(console_handler)
        
        # File handler for all logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"traffic_state_{timestamp}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        self._logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get configured logger instance."""
        return self._logger


# Global logger instance
logger = ProjectLogger().get_logger()