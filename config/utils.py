import json
import os
import configparser
import logging
from pathlib import Path
from typing import Dict, Optional, List

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigUtils:
    """Utility class for managing configurations in the Datascriber project.

    Handles loading of JSON and INI configuration files and provides access to
    project directories. Supports datasource-specific data directories and metadata.

    Attributes:
        base_dir (Path): Project base directory.
        config_dir (Path): Configuration directory (app-config/).
        data_dir (Path): Base data directory (data/).
        models_dir (Path): Models directory (models/).
        logs_dir (Path): Logs directory (logs/).
        temp_dir (Path): Temporary files directory (temp/).
    """

    def __init__(self):
        """Initialize ConfigUtils.

        Sets up project directories and ensures they exist.

        Raises:
            ConfigError: If directory creation fails.
        """
        try:
            self.base_dir = Path(__file__).resolve().parent.parent
            self.config_dir = self.base_dir / "app-config"
            self.data_dir = self.base_dir / "data"
            self.models_dir = self.base_dir / "models"
            self.logs_dir = self.base_dir / "logs"
            self.temp_dir = self.base_dir / "temp"
            self._ensure_directories()
            logging.debug(f"Initialized ConfigUtils with base directory: {self.base_dir}")
        except Exception as e:
            logging.error(f"Failed to initialize ConfigUtils: {str(e)}")
            raise ConfigError(f"Failed to initialize ConfigUtils: {str(e)}")

    def _ensure_directories(self) -> None:
        """Ensure required directories exist.

        Raises:
            ConfigError: If directory creation fails.
        """
        try:
            for directory in [self.config_dir, self.data_dir, self.models_dir, self.logs_dir, self.temp_dir]:
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories: {str(e)}")
            raise ConfigError(f"Failed to create directories: {str(e)}")

    def load_db_configurations(self) -> Dict:
        """Load database configurations from db_configurations.json.

        Supports SQL Server and S3 datasources with schema- and table-based configurations.
        Validates 'schemas' and 'tables' keys, parsing table formats (<schema>.<table_name> or <table_name>).
        If 'tables' is non-empty, it overrides 'schemas'. Sets default schema to 'dbo' for SQL Server if empty.

        Returns:
            Dict: Database configurations with parsed tables.

        Raises:
            ConfigError: If loading or validation fails.
        """
        config_path = self.config_dir / "db_configurations.json"
        try:
            if not config_path.exists():
                logging.error(f"Database configuration file not found: {config_path}")
                raise ConfigError(f"Database configuration file not found: {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            if not isinstance(config.get("datasources", []), list):
                logging.error("Invalid db_configurations.json: 'datasources' must be a list")
                raise ConfigError("Invalid db_configurations.json: 'datasources' must be a list")

            for ds in config["datasources"]:
                if not isinstance(ds, dict) or not all(key in ds for key in ["name", "type", "connection"]):
                    logging.error("Invalid datasource format: missing 'name', 'type', or 'connection'")
                    raise ConfigError("Invalid datasource format")

                # Validate schemas and tables
                if "schemas" not in ds["connection"]:
                    ds["connection"]["schemas"] = ["dbo"] if ds["type"].lower() == "sqlserver" else ["default"]
                elif not isinstance(ds["connection"]["schemas"], list):
                    logging.error(f"Invalid 'schemas' format for datasource: {ds['name']}")
                    raise ConfigError(f"Invalid 'schemas' format")
                if "tables" not in ds["connection"]:
                    ds["connection"]["tables"] = []
                elif not isinstance(ds["connection"]["tables"], list):
                    logging.error(f"Invalid 'tables' format for datasource: {ds['name']}")
                    raise ConfigError(f"Invalid 'tables' format")

                # Parse tables into schema and table name
                parsed_tables = []
                default_schema = ds["connection"]["schemas"][0] if ds["connection"]["schemas"] else "dbo" if ds["type"].lower() == "sqlserver" else "default"
                for table in ds["connection"]["tables"]:
                    if "." in table:
                        schema, table_name = table.split(".", 1)
                        if not schema or not table_name:
                            logging.error(f"Invalid table format '{table}' for datasource: {ds['name']}")
                            raise ConfigError(f"Invalid table format: {table}")
                        parsed_tables.append({"schema": schema, "table": table_name})
                    else:
                        parsed_tables.append({"schema": default_schema, "table": table})
                ds["connection"]["parsed_tables"] = parsed_tables

                # Validate connection details
                if ds["type"].lower() == "sqlserver":
                    required = ["host", "port", "database", "username", "password"]
                    for key in required:
                        if key not in ds["connection"]:
                            logging.error(f"Missing {key} in sqlserver connection for datasource: {ds['name']}")
                            raise ConfigError(f"Missing {key} in sqlserver connection")
                elif ds["type"].lower() == "s3":
                    if "bucket_name" in ds["connection"] and "database" in ds["connection"]:
                        required = ["bucket_name", "database", "region"]
                        for key in required:
                            if key not in ds["connection"]:
                                logging.error(f"Missing {key} in s3 connection for datasource: {ds['name']}")
                                raise ConfigError(f"Missing {key} in s3 connection")
                    elif "bucket" in ds["connection"] and "prefix" in ds["connection"]:
                        required = ["bucket", "prefix", "region"]
                        for key in required:
                            if key not in ds["connection"]:
                                logging.error(f"Missing {key} in s3 connection for datasource: {ds['name']}")
                                raise ConfigError(f"Missing {key} in s3 connection")
                    else:
                        logging.error(f"Invalid s3 connection for datasource: {ds['name']}")
                        raise ConfigError("Invalid s3 connection: must provide 'bucket_name' and 'database' or 'bucket' and 'prefix'")
                else:
                    logging.error(f"Unsupported datasource type: {ds['type']}")
                    raise ConfigError(f"Unsupported datasource type: {ds['type']}")

            logging.debug(f"Loaded database configurations from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse db_configurations.json: {str(e)}")
            raise ConfigError(f"Failed to parse db_configurations.json: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load db_configurations.json: {str(e)}")
            raise ConfigError(f"Failed to load db_configurations.json: {str(e)}")

    def load_aws_config(self) -> Dict:
        """Load AWS configurations from aws_config.json.

        Returns:
            Dict: AWS configuration.

        Raises:
            ConfigError: If loading fails.
        """
        config_path = self.config_dir / "aws_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"No AWS config found at {config_path}, using boto3 defaults")
                return {}
            with open(config_path, "r") as f:
                config = json.load(f)
            required = ["aws_access_key_id", "aws_secret_access_key", "region", "s3_bucket"]
            for key in required:
                if key not in config:
                    logging.error(f"Missing {key} in aws_config.json")
                    raise ConfigError(f"Missing {key} in aws_config.json")
            logging.debug(f"Loaded AWS configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse aws_config.json: {str(e)}")
            raise ConfigError(f"Failed to parse aws_config.json: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load aws_config.json: {str(e)}")
            raise ConfigError(f"Failed to load aws_config.json: {str(e)}")

    def load_synonym_config(self) -> Dict:
        """Load synonym configuration from synonym_config.json.

        Returns:
            Dict: Synonym configuration.

        Raises:
            ConfigError: If loading fails.
        """
        config_path = self.config_dir / "synonym_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"No synonym config found at {config_path}, using default")
                return {"synonym_mode": "static"}
            with open(config_path, "r") as f:
                config = json.load(f)
            logging.debug(f"Loaded synonym configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse synonym_config.json: {str(e)}")
            raise ConfigError(f"Failed to parse synonym_config.json: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load synonym_config.json: {str(e)}")
            raise ConfigError(f"Failed to load synonym_config.json: {str(e)}")

    def load_model_config(self) -> Dict:
        """Load model configuration from model_config.json.

        Returns:
            Dict: Model configuration.

        Raises:
            ConfigError: If loading fails.
        """
        config_path = self.config_dir / "model_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"No model config found at {config_path}, using default")
                return {
                    "model_type": "sentence-transformers",
                    "model_name": "all-MiniLM-L6-v2",
                    "confidence_threshold": 0.8
                }
            with open(config_path, "r") as f:
                config = json.load(f)
            logging.debug(f"Loaded model configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse model_config.json: {str(e)}")
            raise ConfigError(f"Failed to parse model_config.json: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load model_config.json: {str(e)}")
            raise ConfigError(f"Failed to load model_config.json: {str(e)}")

    def load_llm_config(self) -> Dict:
        """Load LLM configuration from llm_config.json.

        Returns:
            Dict: LLM configuration.

        Raises:
            ConfigError: If loading fails.
        """
        config_path = self.config_dir / "llm_config.json"
        try:
            if not config_path.exists():
                logging.debug(f"No LLM config found at {config_path}, using default")
                return {"api_key": "", "endpoint": "", "mock_enabled": False, "mock_endpoint": ""}
            with open(config_path, "r") as f:
                config = json.load(f)
            required = ["api_key", "endpoint", "mock_enabled", "mock_endpoint"]
            for key in required:
                if key not in config:
                    logging.error(f"Missing {key} in llm_config.json")
                    raise ConfigError(f"Missing {key} in llm_config.json")
            logging.debug(f"Loaded LLM configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse llm_config.json: {str(e)}")
            raise ConfigError(f"Failed to parse llm_config.json: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load llm_config.json: {str(e)}")
            raise ConfigError(f"Failed to load llm_config.json: {str(e)}")

    def load_logging_config(self) -> configparser.ConfigParser:
        """Load logging configuration from logging_config.ini.

        Returns:
            configparser.ConfigParser: Logging configuration.

        Raises:
            ConfigError: If loading fails.
        """
        config_path = self.config_dir / "logging_config.ini"
        try:
            if not config_path.exists():
                logging.error(f"Logging configuration file not found: {config_path}")
                raise ConfigError(f"Logging configuration file not found: {config_path}")
            config = configparser.ConfigParser()
            config.read(config_path)
            logging.debug(f"Loaded logging configuration from {config_path}")
            return config
        except configparser.Error as e:
            logging.error(f"Failed to parse logging_config.ini: {str(e)}")
            raise ConfigError(f"Failed to parse logging_config.ini: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load logging_config.ini: {str(e)}")
            raise ConfigError(f"Failed to load logging_config.ini: {str(e)}")

    def load_metadata(self, datasource: Dict, schema: str = "default") -> Dict:
        """Load metadata for a datasource and schema.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name, defaults to 'default'.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            ConfigError: If loading fails.
        """
        metadata_path = self.get_datasource_data_dir(datasource["name"]) / f"metadata_data_{schema}.json"
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logging.debug(f"Loaded metadata from {metadata_path}")
            return metadata
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load {metadata_path}: {str(e)}")
            raise ConfigError(f"Failed to load metadata: {str(e)}")

    def get_datasource_data_dir(self, datasource_name: str) -> Path:
        """Get data directory for a specific datasource.

        Args:
            datasource_name (str): Name of the datasource (e.g., 'bikestores').

        Returns:
            Path: Path to datasource-specific data directory.

        Raises:
            ConfigError: If directory creation fails.
        """
        datasource_dir = self.data_dir / datasource_name
        try:
            datasource_dir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured datasource data directory exists: {datasource_dir}")
            return datasource_dir
        except OSError as e:
            logging.error(f"Failed to create datasource directory {datasource_dir}: {str(e)}")
            raise ConfigError(f"Failed to create datasource directory: {str(e)}")