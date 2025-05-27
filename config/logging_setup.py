import logging
import logging.handlers
import json
import os
from typing import Dict, Optional
from config.utils import ConfigUtils, ConfigError

class LoggingSetup:
    """Configures rotating debug logging for the Datascriber project.

    Sets up a rotating file handler (5 MB, 5 backups) and console output. Provides
    loggers for all components.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger_name (str): Root logger name (default: 'datascriber').
        log_file (str): Log file name (default: 'datascriber.log').
        config_file (str): Logging config file (default: 'logging_config.json').
        logger (logging.Logger): Root logger instance.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logger_name: str = "datascriber",
        log_file: str = "datascriber.log",
        config_file: str = "logging_config.json"
    ):
        """Initialize LoggingSetup.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger_name (str): Root logger name.
            log_file (str): Log file name.
            config_file (str): Logging config file name.
        """
        self.config_utils = config_utils
        self.logger_name = logger_name
        self.log_file = log_file
        self.config_file = config_file
        self.logger = None
        self._setup_logging()

    def _load_logging_config(self) -> Dict:
        """Load logging configuration.

        Returns:
            Dict: Logging configuration.

        Falls back to defaults if file is missing/invalid.
        """
        default_config = {
            "log_level": "DEBUG",
            "max_bytes": 5 * 1024 * 1024,  # 5 MB
            "backup_count": 5,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
        file_path = os.path.join(self.config_utils.config_dir, self.config_file)
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if config.get("log_level", "").upper() not in valid_levels:
                config["log_level"] = default_config["log_level"]
            config["max_bytes"] = int(config.get("max_bytes", default_config["max_bytes"]))
            config["backup_count"] = int(config.get("backup_count", default_config["backup_count"]))
            return config
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Failed to load logging config {file_path}: {str(e)}. Using defaults.")
            return default_config

    def _setup_logging(self) -> None:
        """Configure logging with rotating file handler."""
        log_config = self._load_logging_config()
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(getattr(logging, log_config["log_level"].upper()))
        self.logger.handlers.clear()

        log_path = os.path.join(self.config_utils.logs_dir, self.log_file)
        handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=log_config["max_bytes"],
            backupCount=log_config["backup_count"]
        )
        formatter = logging.Formatter(log_config["format"])
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.debug(f"Logging configured: level={log_config['log_level']}, file={log_path}")

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get logger for a module.

        Args:
            name (Optional[str]): Module name (e.g., 'tia'). If None, returns root logger.

        Returns:
            logging.Logger: Configured logger.
        """
        return logging.getLogger(f"{self.logger_name}.{name}" if name else self.logger_name)