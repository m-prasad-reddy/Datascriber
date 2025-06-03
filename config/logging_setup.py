import logging
import logging.config
from pathlib import Path
import configparser
from typing import Optional
from threading import Lock
from config.utils import ConfigUtils, ConfigError

class LoggingSetup:
    """Manages logging configuration for the Datascriber system.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        loggers (dict): Cache of logger instances.
        _instance (LoggingSetup): Singleton instance.
        _configured (bool): Flag to prevent redundant configuration.
        _lock (Lock): Thread lock for singleton initialization.
    """

    _instance = None
    _configured = False
    _lock = Lock()

    @classmethod
    def get_instance(cls, config_utils: ConfigUtils = None) -> 'LoggingSetup':
        """Get the singleton instance of LoggingSetup.

        Args:
            config_utils (ConfigUtils, optional): Configuration utility instance.

        Returns:
            LoggingSetup: Singleton instance.
        """
        with cls._lock:
            logging.debug(f"get_instance called with config_utils: {id(config_utils) if config_utils else None}")
            if cls._instance is None:
                if config_utils is None:
                    raise ConfigError("ConfigUtils required for first initialization")
                cls._instance = cls(config_utils)
                logging.debug(f"Created new LoggingSetup instance: {id(cls._instance)}")
            else:
                logging.debug(f"Returning existing LoggingSetup instance: {id(cls._instance)}")
            return cls._instance

    def __init__(self, config_utils: ConfigUtils):
        """Initialize LoggingSetup.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            ConfigError: If logging configuration fails.
        """
        if self._instance is not None:
            raise ConfigError("LoggingSetup is a singleton. Use get_instance().")
        self.config_utils = config_utils
        self.loggers = {}
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configure logging based on logging_config.ini.

        Raises:
            ConfigError: If configuration file is invalid or missing.
        """
        if self._configured:
            logging.getLogger().debug("Logging already configured, skipping")
            return
        try:
            root_logger = logging.getLogger()
            existing_handlers = len(root_logger.handlers)
            logging.debug(f"Root logger has {existing_handlers} handlers before configuration")
            root_logger.handlers.clear()

            config_path = self.config_utils.config_dir / "logging_config.ini"
            if not config_path.exists():
                self._create_default_logging_config(config_path)

            config = configparser.ConfigParser()
            config.read(config_path)
            logging.config.fileConfig(config, disable_existing_loggers=True)
            logging.getLogger().debug("Logging configuration applied from logging_config.ini")
            self._configured = True
        except Exception as e:
            raise ConfigError(f"Failed to configure logging: {str(e)}")

    def _create_default_logging_config(self, config_path: Path) -> None:
        """Create a default logging configuration file.

        Args:
            config_path (Path): Path to the logging configuration file.
        """
        config = configparser.ConfigParser()
        config['loggers'] = {
            'keys': 'root,datascriber'
        }
        config['handlers'] = {
            'keys': 'console,file'
        }
        config['formatters'] = {
            'keys': 'standard'
        }
        config['logger_root'] = {
            'level': 'DEBUG',
            'handlers': 'console,file'
        }
        config['logger_datascriber'] = {
            'level': 'DEBUG',
            'handlers': 'console,file',
            'qualname': 'datascriber',
            'propagate': '0'
        }
        config['handler_console'] = {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'args': '(sys.stdout,)'
        }
        config['handler_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'args': f"('{self.config_utils.logs_dir / 'system.log'}', 'a', 10485760, 2)"
        }
        config['formatter_standard'] = {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            config.write(f)

    def get_logger(self, name: str, component: str = "system") -> logging.Logger:
        """Get a logger instance for a specific component.

        Args:
            name (str): Logger name (e.g., 'cli', 'orchestrator').
            component (str): Component identifier (e.g., 'system', datasource name).

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger_name = f"datascriber.{name}.{component}"
        if logger_name not in self.loggers:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.propagate = False
            logger.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.handlers.RotatingFileHandler(
                self.config_utils.logs_dir / 'system.log',
                maxBytes=10485760,
                backupCount=2
            )
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            logger.debug(f"Created logger {logger_name} with level {logging.getLevelName(logger.level)}; "
                        f"file_handler level: {logging.getLevelName(file_handler.level)}")
            self.loggers[logger_name] = logger
        return self.loggers[logger_name]