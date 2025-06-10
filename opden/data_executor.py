import json
import os
import sqlite3
import logging
import requests
import pandas as pd
import pyodbc
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.csv as csv
import boto3
import s3fs
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError
from pandasql import sqldf

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
        s3_filesystem (s3fs.S3FileSystem): S3 filesystem for data access.
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

            # Initialize S3 filesystem
            if self.datasource["type"] == "s3":
                self.s3_filesystem = self._init_s3_filesystem()
            else:
                self.s3_filesystem = None

            if self.datasource["type"] == "sqlserver":
                self._init_sql_server_connection()
            self.logger.debug(f"Initialized DataExecutor for datasource: {datasource['name']}")
        except Exception as e:
            self.logger.error(f"Failed to initialize DataExecutor: {str(e)}")
            raise ExecutionError(f"Failed to initialize DataExecutor: {str(e)}")

    def _init_s3_filesystem(self) -> s3fs.S3FileSystem:
        """Initialize S3 filesystem using s3fs.

        Returns:
            s3fs.S3FileSystem: Configured S3 filesystem.

        Raises:
            ExecutionError: If S3 initialization fails.
        """
        try:
            aws_config = self.config_utils.load_aws_config()
            access_key = aws_config.get("aws_access_key_id")
            secret_key = aws_config.get("aws_secret_access_key")

            if access_key and secret_key:
                fs = s3fs.S3FileSystem(key=access_key, secret=secret_key)
            else:
                fs = s3fs.S3FileSystem()
            self.logger.debug("Initialized S3 filesystem")
            return fs
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 filesystem: {str(e)}")
            raise ExecutionError(f"Failed to initialize S3 filesystem: {str(e)}")

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
                # Extract mock SQL directly from prompt if available
                prompt_lines = prompt.split("\n")
                for line in prompt_lines:
                    if line.strip().startswith("SELECT") and "strftime('%Y'" in line:
                        sql_query = line.strip()
                        self.logger.debug(f"Using mock SQL from prompt: {sql_query}")
                        return sql_query

                # Fallback to mock LLM endpoint
                endpoint = self.llm_config.get("mock_endpoint", "http://localhost:9000/api")
                self.logger.debug(f"Attempting to use mock LLM at {endpoint}")

                headers = {"Content-Type": "application/json"}
                payload = {"prompt": prompt}
                response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
                response.raise_for_status()
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

    def _detect_s3_file_type(self, bucket: str, prefix: str, table_name: str) -> Tuple[str, str]:
        """Detect the file type and path in the S3 prefix.

        Args:
            bucket (str): S3 bucket name.
            prefix (str): S3 prefix (e.g., 'data-files/orders' or 'data-files/default/orders').
            table_name (str): Table name.

        Returns:
            Tuple[str, str]: File format ('parquet', 'csv', 'orc', 'txt') and S3 path.

        Raises:
            ExecutionError: If no files found or mixed file types detected.
        """
        s3_client = boto3.client("s3")
        try:
            # Validate table_name
            if not table_name or table_name.isspace():
                raise ExecutionError(f"Invalid table name: {table_name}")
            if "/" in table_name or "\\" in table_name or "ordersorders" in table_name.lower():
                raise ExecutionError(f"Corrupted table name: {table_name}")

            valid_extensions = {".parquet", ".csv", ".orc", ".txt"}
            format_mapping = {"parquet": "parquet", "csv": "csv", "orc": "orc", "txt": "csv"}

            # Check single file (e.g., data-files/orders.csv or data-files/default/orders.csv)
            for ext in valid_extensions:
                file_prefix = f"{prefix}{ext}".replace("\\", "/")
                self.logger.debug(f"Checking file prefix: s3://{bucket}/{file_prefix}")
                response = s3_client.list_objects_v2(Bucket=bucket, Prefix=file_prefix)
                files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"] == file_prefix]
                if files:
                    self.logger.debug(f"Detected single file: s3://{bucket}/{files[0]}")
                    return format_mapping[ext[1:]], file_prefix

            # Check folder (e.g., data-files/orders/ or data-files/default/orders/)
            folder_prefix = f"{prefix}/".replace("\\", "/")
            self.logger.debug(f"Checking folder prefix: s3://{bucket}/{folder_prefix}")
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=folder_prefix)
            files = [obj["Key"] for obj in response.get("Contents", []) if not obj["Key"].endswith("/")]
            if files:
                extensions = {os.path.splitext(f)[1].lower() for f in files}
                detected_extensions = extensions & valid_extensions
                if not detected_extensions:
                    raise ExecutionError(f"No supported file types found in s3://{bucket}/{folder_prefix}: {extensions}")
                if len(detected_extensions) > 1:
                    raise ExecutionError(f"Mixed file types detected in s3://{bucket}/{folder_prefix}: {detected_extensions}")
                ext = detected_extensions.pop()
                self.logger.debug(f"Detected folder file type {ext[1:]} for {table_name}")
                return format_mapping[ext[1:]], folder_prefix

            raise ExecutionError(f"No files found in s3://{bucket}/{prefix} or {folder_prefix}")
        except ClientError as e:
            self.logger.error(f"Failed to list S3 files: {str(e)}")
            raise ExecutionError(f"Failed to list S3 files: {str(e)}")

    def _execute_s3_query(self, sql_query: str, user: str, nlq: str, schema: str, tia_result: Optional[Dict] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute query on S3 data using PyArrow Dataset API and pandasql.

        Args:
            sql_query (str): SQL query to execute.
            user (str): User submitting the query.
            nlq (str): Original natural language query.
            schema (str): Schema name.
            tia_result (Optional[Dict]): TIA prediction result with tables, columns, extracted_values.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data and CSV path, or (None, None).

        Raises:
            ExecutionError: If query execution fails.
        """
        try:
            self.logger.debug(f"TIA result: {tia_result}")
            metadata = self.storage_manager.get_metadata(schema)
            # Prefer TIA-predicted table
            table_name = None
            if tia_result and tia_result.get("tables"):
                table_name = tia_result.get("tables")[0].split(".")[-1]
                self.logger.debug(f"Using TIA-predicted table: {table_name}")
            else:
                self.logger.warning("No TIA result provided, falling back to metadata")
                tables = metadata.get("tables", [])
                if not tables:
                    raise ExecutionError("No tables found in metadata")
                table_name = tables[0]["name"]
                self.logger.debug(f"Using metadata table: {table_name}")

            bucket = self.datasource["connection"]["bucket_name"]
            database = self.datasource["connection"]["database"]
            # Construct prefix for all possible structures
            base_prefix = os.path.join(database, table_name).replace("\\", "/")
            schema_prefix = os.path.join(database, schema, table_name).replace("\\", "/") if schema != "default" else base_prefix
            prefixes = [base_prefix, schema_prefix]  # Check both with and without schema

            file_format = None
            file_path = None
            for prefix in prefixes:
                try:
                    file_format, file_path = self._detect_s3_file_type(bucket, prefix, table_name)
                    if file_format and file_path:
                        break
                except ExecutionError:
                    continue

            if not file_format or not file_path:
                raise ExecutionError(f"No valid files found for table {table_name} in prefixes: {prefixes}")

            s3_path = f"s3://{bucket}/{file_path}"
            self.logger.debug(f"Loading data from: {s3_path} with format: {file_format}")

            # Load data using PyArrow Dataset
            if file_format == "csv":
                csv_format = ds.CsvFileFormat(parse_options=csv.ParseOptions(delimiter=","))
                dataset = ds.dataset(
                    s3_path,
                    format=csv_format,
                    filesystem=self.s3_filesystem
                )
            elif file_format == "parquet":
                dataset = ds.dataset(s3_path, format="parquet", filesystem=self.s3_filesystem)
            elif file_format == "orc":
                dataset = ds.dataset(s3_path, format="orc", filesystem=self.s3_filesystem)
            else:
                raise ExecutionError(f"Unsupported file format: {file_format}")

            # Convert to pandas DataFrame
            table = dataset.to_table()
            df = table.to_pandas()
            self.logger.debug(f"Loaded {len(df)} rows from {s3_path}")

            # Execute SQL query using pandasql
            locals_dict = {table_name: df}
            result_df = sqldf(sql_query, locals_dict)
            sample_data = result_df.head(5)
            self.logger.debug(f"Retrieved sample data (5 rows) from S3 ({file_format}) for NLQ: {nlq}")

            if not sample_data.empty:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = os.path.join(self.temp_dir, f"output_{timestamp}.csv")
                json_path = os.path.join(self.temp_dir, f"output_{timestamp}.json")
                result_df.to_csv(csv_path, index=False)
                result_df.to_json(json_path, orient="records", indent=2)
                self.logger.info(f"Generated outputs: {csv_path}, {json_path}")
                return sample_data, csv_path
            else:
                self.logger.warning("No data returned from S3 query")
                return None, None
        except Exception as e:
            self.logger.error(f"S3 query execution failed: {str(e)}")
            if self.datasource["type"] == "sqlserver":
                try:
                    self.storage_manager.store_rejected_query(
                        query=nlq,
                        reason=f"S3 query execution failed: {str(e)}",
                        user=user,
                        error_type="EXECUTION_ERROR"
                    )
                except StorageError as se:
                    self.logger.error(f"Failed to store rejected query: {str(se)}")
            else:
                self.logger.warning("S3 rejected query storage not implemented")
            raise ExecutionError(f"S3 query execution failed: {str(e)}")

    def execute_query(
        self,
        prompt: str,
        schema: str = "default",
        user: str = "datauser",
        nlq: str = "",
        max_results: int = 10,
        tia_result: Optional[Dict] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute a query based on the provided prompt.

        Args:
            prompt (str): SQL prompt from PromptGenerator.
            schema (str): Schema name, defaults to 'default'.
            user (str): User submitting the query, defaults to 'datauser'.
            nlq (str): Original natural language query.
            max_results (int): Maximum number of rows to return (default is 10).
            tia_result (Optional[Dict]): TIA prediction result.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data (max_results rows) and CSV path, or (None, None).

        Raises:
            ExecutionError: If execution fails.
        """
        try:
            sql_query = self._call_llm(prompt)
            if not sql_query or sql_query.startswith("#"):
                self.logger.error("Data execution error or empty SQL query")
                if self.datasource["type"] != "sqlserver":
                    self.logger.warning("Failed to store rejected data for S3")
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
                if schema == "default":
                    schema = "schema"
                results = self._execute_sql_server_query(sql_query, user, nlq)
            else:
                results = self._execute_s3_query(sql_query, user, nlq, schema, tia_result)

            if results[0] is not None:
                self.logger.info(f"Query executed for NLQ: {nlq}, saved results to {results[1]}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            raise ExecutionError(f"Failed to execute: {str(e)}")

    def close_connection(self):
        """Close SQL Server connection if open."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.debug(f"Closed SQL Server: {self.datasource['name']}")
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
          - product_name (varchar): Product name
          - category_id (int): Category ID
        SQL Query:
        SELECT TOP 5 product_name, category_id FROM products
        """
        sample_data, csv_path = executor.execute_query(prompt, schema="default", user="sales", nlq="Show top 5 products")
        print("Sample Data:\n", sample_data)
        print("CSV Path:", csv_path)
    except (ConfigError, StorageError, ExecutionError) as e:
        print(f"Error: {str(e)}")
    finally:
        executor.close_connection()
