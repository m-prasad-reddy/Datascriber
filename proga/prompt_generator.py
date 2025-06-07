import json
import os
from typing import Dict, List, Optional
import logging
import requests
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from nlp.nlp_processor import NLPProcessor
from storage.db_manager import DBManager
from storage.storage_manager import StorageManager

class PromptError(Exception):
    """Custom exception for prompt generation errors."""
    pass

class PromptGenerator:
    """Prompt generator for creating SQL query prompts in the Datascriber project.

    Generates prompts based on NLQ processing results, table/column mappings, and rich metadata.
    Supports mock LLM mode for testing and prepares for real LLM integration.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logging_setup (LoggingSetup): Logging setup instance.
        logger (logging.Logger): Prompt generator logger.
        datasource (Dict): Datasource configuration.
        nlp_processor (NLPProcessor): NLP processor instance.
        db_manager (Optional[DBManager]): SQL Server manager.
        storage_manager (Optional[StorageManager]): S3 manager.
        llm_config (Dict): LLM configuration from llm_config.json.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logging_setup: LoggingSetup,
        datasource: Dict,
        nlp_processor: NLPProcessor,
        db_manager: Optional[DBManager] = None,
        storage_manager: Optional[StorageManager] = None
    ):
        """Initialize PromptGenerator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logging_setup (LoggingSetup): Logging setup instance.
            datasource (Dict): Datasource configuration.
            nlp_processor (NLPProcessor): NLP processor instance.
            db_manager (Optional[DBManager]): SQL Server manager.
            storage_manager (Optional[StorageManager]): S3 manager.

        Raises:
            PromptError: If initialization fails.
        """
        try:
            self.config_utils = config_utils
            self.logging_setup = logging_setup
            self.logger = logging_setup.get_logger("prompt_generator", datasource.get("name"))
            self.datasource = datasource
            self.nlp_processor = nlp_processor
            self.db_manager = db_manager
            self.storage_manager = storage_manager
            self.llm_config = self._load_llm_config()
            self.logger.debug(f"Initialized PromptGenerator for datasource: {datasource['name']}")
        except Exception as e:
            self.logger.error(f"Failed to initialize PromptGenerator: {str(e)}")
            raise PromptError(f"Failed to initialize PromptGenerator: {str(e)}")

    def _load_llm_config(self) -> Dict:
        """Load LLM configuration from llm_config.json.

        Returns:
            Dict: LLM configuration.

        Raises:
            PromptError: If configuration loading fails.
        """
        try:
            config = self.config_utils.load_llm_config()
            self.logger.debug("Loaded LLM configuration")
            return config
        except ConfigError as e:
            self.logger.error(f"Failed to load LLM configuration: {str(e)}")
            raise PromptError(f"Failed to load LLM configuration: {str(e)}")

    def _get_metadata(self, schema: str) -> Dict:
        """Fetch rich metadata for a schema.

        Args:
            schema (str): Schema name (e.g., 'default').

        Returns:
            Dict: Rich metadata dictionary.

        Raises:
            PromptError: If metadata fetching fails.
        """
        try:
            datasource_dir = self.config_utils.get_datasource_data_dir(self.datasource["name"])
            metadata_path = os.path.join(datasource_dir, f"metadata_data_{schema}_rich.json")
            os.makedirs(datasource_dir, exist_ok=True)
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.debug(f"Loaded rich metadata from {metadata_path}")
                return metadata
            elif self.datasource["type"] == "sqlserver" and self.db_manager:
                metadata = self.db_manager.get_metadata("dbo" if schema == "default" else schema)
            elif self.datasource["type"] == "s3" and self.storage_manager:
                metadata = self.storage_manager.get_metadata(schema)
            else:
                self.logger.error(f"No metadata source for datasource type {self.datasource['type']}")
                raise PromptError(f"No metadata source for datasource type {self.datasource['type']}")
            self.logger.debug(f"Fetched metadata for schema {schema}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to fetch metadata for schema {schema}: {str(e)}")
            raise PromptError(f"Failed to fetch metadata: {str(e)}")

    def _load_synonyms(self, schema: str) -> Dict[str, List[str]]:
        """Load synonyms for a schema.

        Args:
            schema (str): Schema name (e.g., 'default').

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            PromptError: If synonym loading fails.
        """
        try:
            datasource_dir = self.config_utils.get_datasource_data_dir(self.datasource["name"])
            synonym_path = os.path.join(datasource_dir, f"synonyms_{schema}.json")
            synonyms = {
                "order_date": ["order date", "date", "orderdate"]
            }  # Default synonyms
            os.makedirs(datasource_dir, exist_ok=True)
            if os.path.exists(synonym_path):
                with open(synonym_path, "r") as f:
                    file_synonyms = json.load(f)
                    synonyms.update(file_synonyms)
                self.logger.debug(f"Loaded synonyms from {synonym_path}")
            return synonyms
        except Exception as e:
            self.logger.error(f"Failed to load synonyms for schema {schema}: {str(e)}")
            raise PromptError(f"Failed to load synonyms: {str(e)}")

    def generate_prompt(self, nlq: str, tia_result: Dict, schema: str = "default") -> Optional[str]:
        """Generate an SQL query prompt based on NLQ and TIA results.

        Args:
            nlq (str): Natural language query.
            tia_result (Dict): TIA prediction result with tables, columns, extracted_values, placeholders, entities.
            schema (str): Schema name, defaults to 'default'.

        Returns:
            Optional[str]: SQL query prompt or None if generation fails.

        Raises:
            PromptError: If prompt generation fails critically.
        """
        try:
            self.logger.debug(f"Generating prompt for NLQ: {nlq}, tia_result: {tia_result}")
            metadata = self._get_metadata(schema)
            synonyms = self._load_synonyms(schema)
            tables = tia_result.get("tables", [])
            columns = tia_result.get("columns", [])
            extracted_values = tia_result.get("extracted_values", {})
            placeholders = tia_result.get("placeholders", [])
            entities = tia_result.get("entities", {})
            is_s3 = self.datasource["type"] == "s3"

            if not tables:
                self.logger.warning(f"No tables predicted for NLQ: {nlq}")
                return None

            # Map DATE entities
            if "DATE" in entities:
                date_value = entities["DATE"]
                mapped = False
                if date_value in synonyms:
                    for table in metadata.get("tables", []):
                        if table["name"] in [t.split(".")[-1] for t in tables]:
                            for col in table.get("columns", []):
                                if col["name"] in synonyms[date_value]:
                                    if col["name"] not in columns:
                                        columns.append(col["name"])
                                    extracted_values[col["name"]] = date_value
                                    self.logger.debug(f"Mapped DATE entity {date_value} to column {col['name']}")
                                    mapped = True
                                    break
                            if mapped:
                                break
                if not mapped:
                    # Fallback to date columns
                    for table in metadata.get("tables", []):
                        if table["name"] in [t.split(".")[-1] for t in tables]:
                            for col in table.get("columns", []):
                                col_type = col.get("type", "").lower()
                                is_date_column = any(t in col_type for t in ["date", "datetime", "string"])
                                if is_date_column and col["name"] not in columns:
                                    columns.append(col["name"])
                                    extracted_values[col["name"]] = date_value
                                    self.logger.debug(f"Fallback: Mapped DATE entity {date_value} to column {col['name']}")
                                    mapped = True
                                    break
                            if mapped:
                                break

            # Use default columns if none provided
            if not columns:
                for table in metadata.get("tables", []):
                    if table["name"] in [t.split(".")[-1] for t in tables]:
                        columns = [col["name"] for col in table.get("columns", [])[:2]]
                        self.logger.debug(f"Using default columns {columns} for table {table['name']}")
                        break
                if not columns:
                    columns = ["order_id", "order_date"]
                    self.logger.warning(f"No columns available for tables {tables}, using fallback: {columns}")

            # Build context
            context = []
            for table in metadata.get("tables", []):
                if table["name"] in [t.split(".")[-1] for t in tables]:
                    context.append(f"Table: {table['name']}")
                    context.append(f"Description: {table.get('description', 'No description')}")
                    cols = []
                    for col in table.get("columns", []):
                        if col["name"] in columns or col["name"] in extracted_values:
                            col_info = f"{col['name']} ({col.get('type', 'unknown')})"
                            if col.get("description"):
                                col_info += f": {col['description']}"
                            if col.get("references"):
                                col_info += f", References: {col['references']['table']}.{col['references']['column']}"
                            cols.append(f"  - {col_info}")
                    if cols:
                        context.append("Columns:")
                        context.extend(cols)

            # Build mock query
            mock_query = f"SELECT {', '.join(columns)} FROM {tables[0].split('.')[-1]}"
            if extracted_values:
                conditions = []
                for col, val in extracted_values.items():
                    if "date" in col.lower() or col in columns:
                        if is_s3:
                            conditions.append(f"strftime('%Y', {col}) = '{val}'")
                        else:
                            conditions.append(f"YEAR({col}) = '{val}'")
                    else:
                        conditions.append(f"{col} = '{val}'")
                if conditions:
                    mock_query += f" WHERE {' AND '.join(conditions)}"

            # Build prompt
            prompt = [
                "Generate an SQL query for the following request.",
                f"Based on Natural Language Query: {nlq}",
                "Database Schema:"
            ]
            prompt.extend(context)
            if extracted_values:
                prompt.append("Extracted Values:")
                for col, val in extracted_values.items():
                    prompt.append(f"  - {col}: {val}")
            if placeholders:
                prompt.append("Placeholders:")
                for val in placeholders:
                    prompt.append(f"  - {val}")

            prompt.append("\nSQL Query:")
            if self.llm_config.get("mock_enabled", False):
                prompt.append(mock_query)
                self.logger.debug(f"Generated mock SQL prompt: {mock_query}")
            else:
                prompt.append("# TODO: LLM-generated SQL query")
                self.logger.debug(f"Prepared SQL prompt for LLM processing: {nlq}")

            final_prompt = "\n".join(prompt)
            self.logger.info(f"Generated prompt for NLQ: {nlq}")
            return final_prompt
        except Exception as e:
            self.logger.error(f"Failed to generate prompt for '{nlq}': {str(e)}")
            raise PromptError(f"Failed to generate prompt: {str(e)}")

    def mock_llm_call(self, prompt: str) -> str:
        """Simulate an LLM call for testing purposes.

        Args:
            prompt (str): Generated prompt.

        Returns:
            str: Mock SQL query response.

        Raises:
            PromptError: If mock call fails.
        """
        try:
            if self.llm_config.get("mock_enabled", False):
                is_s3 = self.datasource["type"] == "s3"
                lines = prompt.split("\n")
                tables = []
                columns = []
                conditions = []
                for line in lines:
                    if line.startswith("Table: "):
                        tables.append(line.replace("Table: ", ""))
                    elif line.startswith("  - ") and "(" in line:
                        col_name = line.split("(")[0].split("-")[-1].strip()
                        columns.append(col_name)
                    elif line.startswith("  - ") and ":" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            col, val = parts[0].replace("-", "").strip(), parts[1].strip()
                            if "date" in col.lower():
                                if is_s3:
                                    conditions.append(f"strftime('%Y', {col}) = '{val}'")
                                else:
                                    conditions.append(f"YEAR({col}) = '{val}'")
                            else:
                                conditions.append(f"{col} = '{val}'")
                if tables and columns:
                    query = f"SELECT {', '.join(columns)} FROM {tables[0]}"
                    if conditions:
                        query += f" WHERE {' AND '.join(conditions)}"
                    self.logger.debug(f"Mock LLM response: {query}")
                    return query
                self.logger.warning("Insufficient data for mock LLM response")
                return "# Mock SQL query: insufficient data"
            self.logger.error("Mock LLM call attempted but mock_enabled is False")
            raise PromptError("Mock LLM call not enabled")
        except Exception as e:
            self.logger.error(f"Failed to simulate LLM call: {str(e)}")
            raise PromptError(f"Failed to simulate LLM call: {str(e)}")