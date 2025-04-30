import logging
import sys
from pathlib import Path
from typing import Optional
from .config import Config

class Logger:
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file = log_config.get("file", "investment_system.log")
        
        logger = logging.getLogger("investment_system")
        logger.setLevel(log_level)
        
        formatter = logging.Formatter(log_format)
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    def debug(self, message: str) -> None:
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        self.logger.critical(message) 