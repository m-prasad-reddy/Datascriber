import json
import os
import sqlite3
import logging
import requests
import pandas as pd
import pyodbc
import pyarrow.dataset as ds
import pyarrow.compute as pc
import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError

class ExecutionError(Exception):
    """Custom exception for data execution errors."""
    pass

class DataExecutor:
    """Data executor for running SQL queries in the Datascriber project.

    Executes SQL queries derived from prompts or direct SQL, handling SQL Server and S3 datasources.
    Supports Parquet, CSV, ORC, and TXT files for S3 with automatic file type detection.
    Provides sample data (5 rows) for validation and saves results as CSV/JSON.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logging_setup (LoggingSetup): Logging setup instance.
        logger (logging.Logger): Data executor logger.
        datasource (Dict): Datasource configuration.
        db_manager (Optional[DBManager]): SQL Server manager.
        storage_manager (Optional[StorageManager]): S3 manager.
        llm_config (Dict): LLM configuration from llm_config.json.
        temp_dir (str): Temporary directory for query results.
        connection (Optional[pyodbc.Connection]): SQL Server connection.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logging_setup: LoggingSetup,
        datasource: Dict,
        db_manager: Optional[DBManager] = None,
        storage_manager: Optional[StorageManager] = None
    ):
        """Initialize DataExecutor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logging_setup (LoggingSetup): Logging setup instance.
            datasource (Dict): Datasource configuration.
            db_manager (Optional[DBManager]): SQL Server manager.
            storage_manager (Optional[StorageManager]): S3 manager.

        Raises:
            ExecutionError: If initialization fails.
        """
        try:
            self.config_utils = config_utils
            self.logging_setup = logging_setup
            self.logger = logging_setup.get_logger("data_executor", datasource.get("name"))
            self.datasource = datasource
            self.db_manager = db_manager
            self.storage_manager = storage_manager
            self.llm_config = self._load_llm_config()
            self.temp_dir = os.path.join(self.config_utils.temp_dir, "query_results")
            self.connection = None
            os.makedirs(self.temp_dir, exist_ok=True)
            if self.datasource["type"] == "sqlserver":
                self._init_sql_server_connection()
            self.logger.debug(f"Initialized DataExecutor for datasource: {datasource['name']}")
        except Exception as e:
            self.logger.error(f"Failed to initialize DataExecutor: {str(e)}")
            raise ExecutionError(f"Failed to initialize DataExecutor: {str(e)}")

    def _load_llm_config(self) -> Dict:
        """Load LLM configuration from llm_config.json.

        Returns:
            Dict: LLM configuration.

        Raises:
            ExecutionError: If configuration loading fails.
        """
        try:
            config = self.config_utils.load_llm_config()
            self.logger.debug("Loaded LLM configuration")
            return config
        except ConfigError as e:
            self.logger.error(f"Failed to load LLM configuration: {str(e)}")
            raise ExecutionError(f"Failed to load LLM configuration: {str(e)}")

    def _init_sql_server_connection(self) -> None:
        """Initialize SQL Server connection with pyodbc.

        Raises:
            ExecutionError: If connection fails.
        """
        try:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.datasource['connection']['host']};"
                f"DATABASE={self.datasource['connection']['database']};"
                f"UID={self.datasource['connection']['username']};"
                f"PWD={self.datasource['connection']['password']}"
            )
            self.connection = pyodbc.connect(conn_str)
            self.connection.autocommit = False
            self.logger.debug(f"Connected to SQL Server: {self.datasource['name']}")
        except pyodbc.Error as e:
            self.logger.error(f"Failed to connect to SQL Server {self.datasource['name']}: {str(e)}")
            raise ExecutionError(f"Failed to connect to SQL Server: {str(e)}")

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM to convert prompt to SQL query.

        Args:
            prompt (str): SQL prompt from PromptGenerator.

        Returns:
            Optional[str]: Generated SQL query, or None if call fails.

        Raises:
            ExecutionError: If LLM call fails critically.
        """
        try:
            if self.llm_config.get("mock_enabled", False):
                endpoint = self.llm_config.get("mock_endpoint", "http://localhost:9000/api")
                self.logger.debug(f"Attempting to use mock LLM at {endpoint}")

                # Send prompt to mock endpoint
                headers = {"Content-Type": "application/json"}
                payload = {"prompt": prompt}
                response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
                response.raise_for_status()
                # Handle both response formats
                response_json = response.json()
                sql_query = response_json.get("sql_query", "")
                if not sql_query and "choices" in response_json and response_json["choices"]:
                    sql_query = response_json["choices"][0].get("text", "")
                if not sql_query:
                    self.logger.warning("Empty SQL query from mock LLM")
                    return None
                self.logger.debug(f"Extracted mock SQL: {sql_query}")
                return sql_query.strip()
            else:
                # OpenAI endpoint
                if not self.llm_config.get("api_key") or self.llm_config["api_key"] == "your_openai_api_key_here":
                    self.logger.error("Invalid OpenAI API key in LLM configuration")
                    raise ExecutionError("Invalid OpenAI API key")
                headers = {"Authorization": f"Bearer {self.llm_config['api_key']}", "Content-Type": "application/json"}
                payload = {"prompt": prompt, "max_tokens": 500}
                response = requests.post(self.llm_config["endpoint"], json=payload, headers=headers)
                response.raise_for_status()
                sql_query = response.json().get("sql_query", "")
                if not sql_query:
                    self.logger.error("Empty SQL query from LLM")
                    raise ExecutionError("Empty SQL query from LLM")
                self.logger.debug(f"Generated SQL from LLM: {sql_query}")
                return sql_query.strip()
        except requests.RequestException as e:
            self.logger.error(f"LLM API call failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to call LLM: {str(e)}")
            raise ExecutionError(f"Failed to call LLM: {str(e)}")

    def _detect_s3_file_type(self, bucket: str, prefix: str) -> str:
        """Detect the file type in the S3 prefix.

        Args:
            bucket (str): S3 bucket name.
            prefix (str): S3 prefix for the datasource.

        Returns:
            str: File format ('parquet', 'csv', 'orc', 'txt').

        Raises:
            ExecutionError: If no files found or mixed file types detected.
        """
        s3_client = boto3.client("s3")
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            files = [obj["Key"] for obj in response.get("Contents", []) if not obj["Key"].endswith("/")]
            if not files:
                raise ExecutionError(f"No files found in s3://{bucket}/{prefix}")

            extensions = {os.path.splitext(f)[1].lower() for f in files}
            valid_extensions = {".parquet", ".csv", ".orc", ".txt"}
            detected_extensions = extensions & valid_extensions
            if not detected_extensions:
                raise ExecutionError(f"No supported file types found: {extensions}")
            if len(detected_extensions) > 1:
                raise ExecutionError(f"Mixed file types detected: {detected_extensions}")

            ext = detected_extensions.pop()
            return {"parquet": "parquet", "csv": "csv", "orc": "orc", "txt": "csv"}[ext[1:]]
        except ClientError as e:
            self.logger.error(f"Failed to list S3 files: {str(e)}")
            raise ExecutionError(f"Failed to list S3 files: {str(e)}")

    def _execute_sql_server_query(self, sql_query: str, user: str, nlq: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute SQL query on SQL Server and return sample data or CSV.

        Args:
            sql_query (str): SQL query to execute.
            user (str): User submitting the query.
            nlq (str): Original natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data (5 rows) and CSV path, or (None, None).

        Raises:
            ExecutionError: If query execution fails.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(5)
            sample_data = pd.DataFrame([tuple(row) for row in rows], columns=columns)
            self.logger.debug(f"Retrieved sample data (5 rows) for NLQ: {nlq}")

            if not sample_data.empty:
                cursor.execute(sql_query)
                all_rows = cursor.fetchall()
                df = pd.DataFrame([tuple(row) for row in all_rows], columns=columns)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = os.path.join(self.temp_dir, f"output_{timestamp}.csv")
                json_path = os.path.join(self.temp_dir, f"output_{timestamp}.json")
                df.to_csv(csv_path, index=False)
                df.to_json(json_path, orient="records", indent=2)
                self.logger.info(f"Generated outputs: {csv_path}, {json_path}")
                cursor.close()
                return sample_data, csv_path
            else:
                self.logger.warning("No data returned for query")
                cursor.close()
                return None, None
        except pyodbc.Error as e:
            self.logger.error(f"SQL Server query execution failed: {str(e)}")
            try:
                self.storage_manager.store_rejected_query(
                    query=nlq,
                    reason=f"Query execution failed: {str(e)}",
                    user=user,
                    error_type="EXECUTION_ERROR"
                )
            except StorageError as se:
                self.logger.error(f"Failed to store rejected query: {str(se)}")
            raise ExecutionError(f"SQL Server query execution failed: {str(e)}")

    def _execute_s3_query(self, sql_query: str, user: str, nlq: str, schema: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute query on S3 data using PyArrow Dataset API.

        Args:
            sql_query (str): SQL query to execute (parsed for S3 processing).
            user (str): User submitting the query.
            nlq (str): Original natural language query.
            schema (str): Schema name.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data and CSV path, or (None, None).

        Raises:
            ExecutionError: If query execution fails.
        """
        try:
            metadata = self.storage_manager.get_metadata(schema)
            tables = metadata.get("tables", [])
            table_name = tables[0]["name"] if tables else None
            if not table_name:
                raise ExecutionError("No tables found in metadata")

            bucket = self.datasource["connection"]["bucket_name"]
            prefix = os.path.join(self.datasource["connection"]["database"], schema, table_name)
            file_format = self._detect_s3_file_type(bucket, prefix)

            s3_filesystem = ds.S3FileSystem(region="us-east-1")
            format_options = {"format": file_format}
            if file_format == "csv":
                format_options["csv_options"] = {"delimiter": "\t" if file_format == "txt" else ","}

            dataset = ds.dataset(
                f"s3://{bucket}/{prefix}",
                **format_options,
                filesystem=s3_filesystem
            )

            table = dataset.to_table()
            sample_data = table.slice(0, 5).to_pandas()
            self.logger.debug(f"Retrieved sample data (5 rows) from S3 ({file_format}) for NLQ: {nlq}")

            if not sample_data.empty:
                df = table.to_pandas()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = os.path.join(self.temp_dir, f"output_{timestamp}.csv")
                json_path = os.path.join(self.temp_dir, f"output_{timestamp}.json")
                df.to_csv(csv_path, index=False)
                df.to_json(json_path, orient="records", indent=2)
                self.logger.info(f"Generated outputs: {csv_path}, {json_path}")
                return sample_data, csv_path
            else:
                self.logger.warning("No data returned from S3")
                return None, None
        except (ClientError, ValueError, StorageError) as e:
            self.logger.error(f"S3 query execution failed: {str(e)}")
            try:
                self.storage_manager.store_rejected_query(
                    query=nlq,
                    reason=f"S3 query execution failed: {str(e)}",
                    user=user,
                    error_type="EXECUTION_ERROR"
                )
            except StorageError as se:
                self.logger.error(f"Failed to store rejected query: {str(se)}")
            raise ExecutionError(f"S3 query execution failed: {str(e)}")

    def execute_query(self, prompt: str, schema: str = "default", user: str = "datauser", nlq: str = "") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute a query based on the provided prompt.

        Args:
            prompt (str): SQL prompt from PromptGenerator.
            schema (str): Schema name, defaults to 'default'.
            user (str): User submitting the query, defaults to 'datauser'.
            nlq (str): Original natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data (5 rows) and CSV path, or (None, None).

        Raises:
            ExecutionError: If execution fails.
        """
        try:
            sql_query = self._call_llm(prompt)
            if not sql_query or sql_query.startswith("#"):
                self.logger.error("Invalid or empty SQL query generated")
                # Skip rejected query storage for S3 datasources
                if self.datasource["type"] != "sqlserver":
                    self.logger.warning("Rejected query storage not implemented for S3")
                else:
                    try:
                        self.storage_manager.store_rejected_query(
                            query=nlq,
                            reason="Invalid or empty SQL query generated",
                            user=user,
                            error_type="INVALID_SQL"
                        )
                    except StorageError as se:
                        self.logger.error(f"Failed to store rejected query: {str(se)}")
                raise ExecutionError("Invalid or empty SQL query generated")

            if self.datasource["type"] == "sqlserver":
                results = self._execute_sql_server_query(sql_query, user, nlq)
            else:
                results = self._execute_s3_query(sql_query, user, nlq, schema)

            if results[0] is not None:
                self.logger.info(f"Executed query for NLQ: {nlq}, saved results to {results[1]}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            raise ExecutionError(f"Failed to execute query: {str(e)}")

    def close_connection(self) -> None:
        """Close SQL Server connection if open."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.debug(f"Closed SQL Server connection for {self.datasource['name']}")
            except pyodbc.Error as e:
                self.logger.error(f"Failed to close connection: {str(e)}")
            finally:
                self.connection = None

if __name__ == "__main__":
    try:
        config_utils = ConfigUtils()
        logging_setup = LoggingSetup(config_utils)
        db_config = config_utils.load_db_configurations()
        datasource = next(ds for ds in db_config["datasources"] if ds["type"] == "sqlserver")
        storage_manager = StorageManager(config_utils, logging_setup, datasource)
        db_manager = DBManager(config_utils, logging_setup, datasource)

        executor = DataExecutor(config_utils, logging_setup, datasource, db_manager, storage_manager)
        prompt = """
        Natural Language Query: Show top 5 products
        Database Schema:
        Table: products
        Columns:
          - product_name (varchar): Name of the product
          - category_id (int): Category identifier
        SQL Query:
        SELECT TOP 5 product_name, category_id FROM products
        """
        sample_data, csv_path = executor.execute_query(prompt, schema="default", user="datauser", nlq="Show top 5 products")
        print("Sample Data:\n", sample_data)
        print("CSV Path:", csv_path)
    except (ConfigError, StorageError, ExecutionError) as e:
        print(f"Error: {str(e)}")
    finally:
        executor.close_connection()