import logging
import json
import pandas as pd
from typing import Dict, List, Optional
import traceback
from config.utils import ConfigUtils
from config.logging_setup import LoggingSetup
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError
from tia.table_identifier import TableIdentifier, TIAError
from nlp.nlp_processor import NLPProcessor, NLPError
from proga.prompt_generator import PromptGenerator, PromptError
from opden.data_executor import DataExecutor, ExecutionError

class OrchestrationError(Exception):
    """Custom exception for orchestration errors."""
    pass

class Orchestrator:
    """Orchestrator for the Datascriber system.

    Coordinates user login, datasource selection, metadata validation, NLQ processing,
    and admin tasks through TIA, NLP, prompt generation, and data execution components.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        logging_setup (LoggingSetup): Logging setup instance.
        user (Optional[str]): Current user (admin/datauser).
        datasource (Optional[Dict]): Selected datasource configuration.
        db_manager (Optional[DBManager]): SQL Server manager.
        storage_manager (Optional[StorageManager]): S3 manager.
        nlp_processor (Optional[NLPProcessor]): NLP processor.
        table_identifier (Optional[TableIdentifier]): TIA instance.
        prompt_generator (Optional[PromptGenerator]): Prompt generator instance.
        data_executor (Optional[DataExecutor]): Data executor instance.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize Orchestrator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            OrchestrationError: If initialization fails.
        """
        try:
            self.config_utils = config_utils
            # Initialize a temporary logger in case LoggingSetup fails
            self.logger = logging.getLogger("orchestrator.temp")
            try:
                self.logging_setup = LoggingSetup.get_instance(self.config_utils)
                self.logger = self.logging_setup.get_logger("orchestrator", "system")
            except Exception as e:
                self.logger.error(f"Failed to initialize LoggingSetup: {str(e)}")
                raise OrchestrationError(f"Failed to initialize LoggingSetup: {str(e)}")
            self.user = None
            self.datasource = None
            self.db_manager = None
            self.storage_manager = None
            self.nlp_processor = None
            self.table_identifier = None
            self.prompt_generator = None
            self.data_executor = None
            self.logger.debug("Initialized Orchestrator")
        except Exception as e:
            self.logger.error(f"Failed to initialize Orchestrator: {str(e)}\n{traceback.format_exc()}")
            raise OrchestrationError(f"Failed to initialize Orchestrator: {str(e)}")

    def login(self, username: str) -> bool:
        """Authenticate user.

        Args:
            username (str): Username (admin or datauser).

        Returns:
            bool: True if login successful, False otherwise.
        """
        try:
            if username in ["admin", "datauser"]:
                self.user = username
                self.logger.info(f"User {username} logged in")
                return True
            self.logger.error(f"Invalid username: {username}")
            return False
        except Exception as e:
            self.logger.error(f"Login failed for {username}: {str(e)}")
            return False

    def select_datasource(self, datasource_name: str) -> bool:
        """Select and initialize datasource.

        Args:
            datasource_name (str): Datasource name (e.g., 'bikestores').

        Returns:
            bool: True if selection successful, False otherwise.

        Raises:
            OrchestrationError: If datasource initialization fails.
        """
        try:
            datasources = self.config_utils.load_db_configurations().get("datasources", [])
            self.datasource = next((ds for ds in datasources if ds["name"] == datasource_name), None)
            if not self.datasource:
                self.logger.error(f"Datasource {datasource_name} not found")
                return False

            if self.datasource["type"] == "sqlserver":
                self.db_manager = DBManager(self.config_utils, self.logging_setup, self.datasource, user_type=self.user)
                self.storage_manager = None
            elif self.datasource["type"] == "s3":
                self.storage_manager = StorageManager(self.config_utils, self.logging_setup, self.datasource)
                self.db_manager = None
            else:
                self.logger.error(f"Unsupported datasource type: {self.datasource['type']}")
                return False

            self.nlp_processor = NLPProcessor(
                self.config_utils, self.logging_setup, self.datasource, self.db_manager, self.storage_manager
            )
            self.table_identifier = TableIdentifier(
                self.config_utils, self.logging_setup, self.datasource, self.nlp_processor,
                self.db_manager, self.storage_manager
            )
            self.prompt_generator = PromptGenerator(
                self.config_utils, self.logging_setup, self.datasource, self.nlp_processor,
                self.db_manager, self.storage_manager
            )
            self.data_executor = DataExecutor(
                self.config_utils, self.logging_setup, self.datasource, self.db_manager, self.storage_manager
            )
            self.logger.info(f"Selected datasource: {datasource_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to select datasource {datasource_name}: {str(e)}")
            raise OrchestrationError(f"Failed to select datasource: {str(e)}")

    def validate_metadata(self, schema: str = None) -> bool:
        """Validate metadata for the selected datasource and schema(s).

        Args:
            schema (str, optional): Schema name. If None, validates all configured schemas.

        Returns:
            bool: True if metadata valid, False otherwise.

        Raises:
            OrchestrationError: If validation fails critically.
        """
        try:
            if not self.datasource:
                self.logger.error("No datasource selected")
                raise OrchestrationError("No datasource selected")

            schemas = self.datasource["connection"].get("schemas", []) if schema is None else [schema]
            if not schemas:
                self.logger.error("No schemas configured in db_configurations.json")
                return False
            valid = True
            for s in schemas:
                if self.datasource["type"] == "sqlserver" and self.db_manager:
                    if not self.db_manager.validate_metadata(s, self.user):
                        if self.user == "admin":
                            self.db_manager.fetch_metadata(generate_rich_template=True)
                            valid &= self.db_manager.validate_metadata(s, self.user)
                        else:
                            valid = False
                elif self.datasource["type"] == "s3" and self.storage_manager:
                    valid &= self.storage_manager.validate_metadata(s, self.user)
                else:
                    valid = False
                if not valid:
                    self.logger.warning(f"Metadata validation failed for schema {s}")
                    if self.user == "datauser":
                        self.logger.info("Logging out datauser due to invalid metadata")
                        self.user = None
                        self.datasource = None
                        return False
            return valid
        except Exception as e:
            self.logger.error(f"Failed to validate metadata: {str(e)}")
            raise OrchestrationError(f"Failed to validate metadata: {str(e)}")

    def process_nlq(self, nlq: str, schema: str = "default") -> Optional[Dict]:
        """Process a natural language query.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name, defaults to 'default'.

        Returns:
            Optional[Dict]: Query result or None if processing fails.

        Raises:
            OrchestrationError: If processing fails.
        """
        try:
            if not self.validate_metadata(schema):
                self.notify_admin(nlq, schema, "Invalid metadata")
                return None

            tia_result = self.table_identifier.predict_tables(nlq, self.user, schema)
            if not tia_result or not tia_result.get("tables"):
                self.notify_admin(nlq, schema, "No tables predicted by TIA")
                self.logger.error(f"TIA failed to predict tables for NLQ: {nlq}")
                return None

            prompt = self.prompt_generator.generate_prompt(nlq, tia_result, schema)
            self.logger.info(f"Generated prompt for NLQ: {nlq}")

            sample_data, csv_path = self.data_executor.execute_query(prompt, schema, self.user, nlq)
            if sample_data is None:
                self.notify_admin(nlq, schema, "Query execution returned no data")
                return None
            self.logger.info(f"Executed query for NLQ: {nlq}")

            result = {
                "tables": tia_result["tables"],
                "columns": tia_result["columns"],
                "extracted_values": tia_result["extracted_values"],
                "placeholders": tia_result["placeholders"],
                "prompt": prompt,
                "sample_data": sample_data.to_dict(orient="records") if sample_data is not None else [],
                "csv_path": csv_path
            }

            training_data = {
                "db_source_type": self.datasource["type"],
                "db_name": self.datasource["name"],
                "user_query": nlq,
                "related_tables": ",".join(tia_result["tables"]),
                "specific_columns": ",".join(tia_result["columns"]),
                "extracted_values": json.dumps(tia_result["extracted_values"]),
                "placeholders": json.dumps(tia_result["placeholders"]),
                "relevant_sql": prompt.split("SQL Query:\n")[-1].strip()
            }
            if self.db_manager:
                self.db_manager.store_training_data(training_data)
            elif self.storage_manager:
                self.logger.warning("S3 training data storage not implemented")
            else:
                self.logger.warning("No storage manager available, skipping training data storage")

            return result
        except (TIAError, NLPError, DBError, StorageError, PromptError, ExecutionError) as e:
            self.notify_admin(nlq, schema, str(e))
            self.logger.error(f"Failed to process NLQ '{nlq}': {str(e)}")
            return None

    def notify_admin(self, nlq: str, schema: str, reason: str) -> None:
        """Notify admin of a failed query by logging to rejected_queries table.

        Args:
            nlq (str): Failed natural language query.
            schema (str): Schema name.
            reason (str): Reason for failure.
        """
        try:
            if self.db_manager:
                self.db_manager.store_rejected_query(
                    query=nlq,
                    reason=reason,
                    user=self.user or "unknown",
                    error_type="NLQProcessingFailure"
                )
                self.logger.info(f"Notified admin: Logged rejected query '{nlq}' for {self.datasource['name']}")
            elif self.storage_manager:
                self.logger.warning("S3 rejected query storage not implemented")
            else:
                self.logger.error("No storage manager available to log rejected query")
        except Exception as e:
            self.logger.error(f"Failed to notify admin for query '{nlq}': {str(e)}")

    def map_failed_query(self, nlq: str, tables: List[str], columns: List[str], sql: str) -> bool:
        """Map a failed query to tables, columns, and SQL for training.

        Args:
            nlq (str): Failed query.
            tables (List[str]): Associated tables.
            columns (List[str]): Associated columns.
            sql (str): SQL query.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self.db_manager:
                self.logger.error("No db_manager available for mapping failed query")
                return False
            training_data = {
                "db_source_type": self.datasource["type"],
                "db_name": self.datasource["name"],
                "user_query": nlq,
                "related_tables": ",".join(tables),
                "specific_columns": ",".join(columns),
                "extracted_values": json.dumps({}),
                "placeholders": json.dumps([]),
                "relevant_sql": sql
            }
            self.db_manager.store_training_data(training_data)
            self.logger.info(f"Mapped failed query '{nlq}' for {self.datasource['name']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to map query '{nlq}': {str(e)}")
            return False

    def refresh_metadata(self, schema: str) -> bool:
        """Refresh metadata for the selected datasource and schema.

        Args:
            schema (str): Schema name.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            OrchestrationError: If refresh fails.
        """
        try:
            if not self.datasource:
                self.logger.error("No datasource selected")
                raise OrchestrationError("No datasource selected")
            if self.datasource["type"] == "sqlserver" and self.db_manager:
                self.db_manager.fetch_metadata(generate_rich_template=True)
                self.db_manager.update_rich_metadata(schema)
                self.logger.info(f"Refreshed metadata for schema {schema} (SQL Server)")
                return True
            elif self.datasource["type"] == "s3" and self.storage_manager:
                self.storage_manager.fetch_metadata()
                self.logger.info(f"Refreshed metadata for schema {schema} (S3)")
                return True
            else:
                self.logger.error(f"Unsupported datasource type: {self.datasource['type']}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to refresh metadata for schema {schema}: {str(e)}")
            return False

    def train_model(self) -> bool:
        """Train the prediction model for the selected datasource.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            OrchestrationError: If training fails.
        """
        try:
            if not self.datasource:
                self.logger.error("No datasource selected")
                raise OrchestrationError("No datasource selected")
            if not self.db_manager:
                self.logger.error("No database manager initialized for training")
                raise OrchestrationError("No database manager initialized")
            if not self.table_identifier:
                self.logger.error("No table identifier initialized")
                raise OrchestrationError("No table identifier initialized")
            
            self.logger.debug(f"Training model for datasource: {self.datasource['name']}")
            training_data = self.db_manager.get_training_data(self.datasource["name"])
            self.logger.debug(f"Retrieved {len(training_data)} training data entries for training")
            
            if training_data:
                self.logger.debug(f"Using {len(training_data)} training data entries for model training")
                self.table_identifier.train(training_data)
                self.logger.info("Trained prediction model")
            else:
                self.logger.warning("No training data available for model training")
                self.table_identifier.generate_model()
                self.logger.info("Generated default prediction model")
            return True
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}\n{traceback.format_exc()}")
            return False