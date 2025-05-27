import json
import os
from typing import Dict
import logging

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigUtils:
    """Utility class for managing configuration files and directories.

    Handles loading of database, AWS, model, LLM, and metadata configurations,
    and ensures required directories exist.

    Attributes:
        base_dir (str): Base directory for the project.
        config_dir (str): Directory for configuration files.
        logs_dir (str): Directory for log files.
        data_dir (str): Directory for data files.
        models_dir (str): Directory for model files.
        logger (logging.Logger): Logger for configuration operations.
    """

    def __init__(self):
        """Initialize ConfigUtils and create necessary directories.

        Raises:
            ConfigError: If directory creation fails.
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.config_dir = os.path.join(self.base_dir, "config")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.data_dir = os.path.join(self.base_dir, "data")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.logger = logging.getLogger(__name__)

        try:
            for directory in [self.config_dir, self.logs_dir, self.data_dir, self.models_dir]:
                os.makedirs(directory, exist_ok=True)
            self.logger.debug("Initialized ConfigUtils with directories")
        except OSError as e:
            self.logger.error(f"Failed to create directories: {str(e)}")
            raise ConfigError(f"Failed to create directories: {str(e)}")

    def load_db_config(self) -> Dict:
        """Load database configurations from db_configurations.json.

        Returns:
            Dict: Database configuration dictionary.

        Raises:
            ConfigError: If the configuration file is missing or invalid.
        """
        config_path = os.path.join(self.config_dir, "db_configurations.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            if not config.get("databases"):
                raise ConfigError("No databases found in db_configurations.json")
            self.logger.debug("Loaded database configurations")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load db_configurations.json: {str(e)}")
            raise ConfigError(f"Failed to load db_configurations.json: {str(e)}")

    def load_aws_config(self) -> Dict:
        """Load AWS configurations from aws_config.json.

        Returns:
            Dict: AWS configuration dictionary.

        Raises:
            ConfigError: If the configuration file is missing or invalid.
        """
        config_path = os.path.join(self.config_dir, "aws_config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            required = ["aws_access_key_id", "aws_secret_access_key", "region", "s3_bucket"]
            for key in required:
                if key not in config:
                    raise ConfigError(f"Missing {key} in aws_config.json")
            self.logger.debug("Loaded AWS configurations")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load aws_config.json: {str(e)}")
            raise ConfigError(f"Failed to load aws_config.json: {str(e)}")

    def load_model_config(self) -> Dict:
        """Load model configurations from model_config.json.

        Returns:
            Dict: Model configuration dictionary.

        Raises:
            ConfigError: If the configuration file is missing or invalid.
        """
        config_path = os.path.join(self.config_dir, "model_config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            required = ["model_type", "confidence_threshold"]
            for key in required:
                if key not in config:
                    raise ConfigError(f"Missing {key} in model_config.json")
            self.logger.debug("Loaded model configurations")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load model_config.json: {str(e)}")
            raise ConfigError(f"Failed to load model_config.json: {str(e)}")

    def load_llm_config(self) -> Dict:
        """Load LLM configurations from llm_config.json.

        Returns:
            Dict: LLM configuration dictionary.

        Raises:
            ConfigError: If the configuration file is missing or invalid.
        """
        config_path = os.path.join(self.config_dir, "llms_config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            required = ["api_key", "endpoint", "mock_enabled", "mock_endpoint"]
            for key in required:
                if key not in config:
                    raise ConfigError(f"Missing {key} in llm_config.json")
            self.logger.debug("Loaded LLM configurations")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}")
            raise ConfigError(f"Failed to load llm_config.json: {str(e)}")

    def load_metadata(self, datasource: Dict, schema: str) -> Dict:
        """Load metadata for a datasource from metadatafile.json.

        Validates the metadata structure, including table/column descriptions and references.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name (e.g., 'default').

        Returns:
            Dict: Metadata dictionary.

        Raises:
            ConfigError: If the metadata file is missing, invalid, or has incorrect structure.
        """
        metadata_path = os.path.join(self.config_dir, f"metadatafile.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if metadata.get("schema") != schema:
                raise ConfigError(f"Schema mismatch: expected {schema}, got {metadata.get('schema')}")
            if not metadata.get("tables"):
                raise ConfigError("No tables found in metadatafile.json")
            self._validate_metadata(metadata)
            self.logger.debug(f"Loaded metadata for datasource: {datasource['name']}, schema: {schema}")
            return metadata
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load metadatafile.json: {str(e)}")
            raise ConfigError(f"Failed to load metadatafile.json: {str(e)}")

    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate the structure of metadatafile.json.

        Ensures tables, columns, descriptions, and references are correctly formatted.

        Args:
            metadata (Dict): Metadata dictionary to validate.

        Raises:
            ConfigError: If metadata is invalid or missing required fields.
        """
        try:
            if not isinstance(metadata.get("tables"), list):
                raise ConfigError("Metadata 'tables' must be a list")
            table_names = set()
            for table in metadata["tables"]:
                if not isinstance(table, dict) or "name" not in table or "columns" not in table:
                    raise ConfigError("Invalid table format: missing 'name' or 'columns'")
                if table["name"] in table_names:
                    raise ConfigError(f"Duplicate table name: {table['name']}")
                table_names.add(table["name"])
                if not isinstance(table.get("description"), str):
                    raise ConfigError(f"Missing or invalid 'description' for table: {table['name']}")
                if not isinstance(table["columns"], list):
                    raise ConfigError(f"Invalid columns format for table: {table['name']}")
                for column in table["columns"]:
                    if not isinstance(column, dict) or "name" not in column or "type" not in column:
                        raise ConfigError(f"Invalid column format in table: {table['name']}")
                    if not isinstance(column.get("description"), str):
                        raise ConfigError(f"Missing or invalid 'description' for column: {column['name']} in table: {table['name']}")
                    ref = column.get("references")
                    if ref is not None:
                        if not isinstance(ref, dict) or "table" not in ref or "column" not in ref:
                            raise ConfigError(f"Invalid 'references' format for column: {column['name']} in table: {table['name']}")
                        if ref["table"] not in table_names:
                            raise ConfigError(f"Invalid reference table: {ref['table']} in column: {column['name']} of table: {table['name']}")
            self.logger.debug("Metadata validation successful")
        except Exception as e:
            self.logger.error(f"Metadata validation failed: {str(e)}")
            raise ConfigError(f"Metadata validation failed: {str(e)}")