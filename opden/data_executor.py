import pyodbc
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import boto3
from botocore.exceptions import ClientError
import os
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from storage.storage_manager import StorageManager, StorageError

class OPDENError(Exception):
    """Custom exception for On-Premises Data Execution Engine errors."""
    pass

class DataExecutor:
    """On-Premises Data Execution Engine for executing queries and processing data.

    This class executes SQL queries on SQL Server and processes data from S3 using PyArrow.
    It supports Parquet, CSV, ORC, and TXT files with automatic file type detection.
    Generates sample data (5 rows) for user validation and produces CSV outputs.

    Attributes:
        config_utils (ConfigUtils): Instance for configuration and metadata access.
        logger (logging.Logger): Logger instance for OPDEN operations.
        storage_manager (StorageManager): Instance for database operations.
        datasource (Dict): Current datasource configuration.
        connection (pyodbc.Connection): SQL Server connection (None for S3).
    """

    def __init__(self, config_utils: ConfigUtils, logging_setup: LoggingSetup, storage_manager: StorageManager, datasource: Dict):
        """Initialize DataExecutor with configuration, logging, and storage.

        Args:
            config_utils (ConfigUtils): Instance for configuration access.
            logging_setup (LoggingSetup): Instance for logging setup.
            storage_manager (StorageManager): Instance for storage operations.
            datasource (Dict): Datasource configuration from db_configurations.json.

        Raises:
            OPDENError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logging_setup.get_logger("opden")
        self.storage_manager = storage_manager
        self.datasource = datasource
        self.connection = None
        if self.datasource["type"] == "SQL Server":
            self._init_sql_server_connection()
        self.logger.debug(f"Initialized OPDEN for datasource: {datasource['name']}")

    def _init_sql_server_connection(self) -> None:
        """Initialize SQL Server connection with pyodbc.

        Uses connection pooling to prevent locks.

        Raises:
            OPDENError: If connection fails.
        """
        try:
            conn_str = (
                f"DRIVER={self.datasource['driver']};"
                f"SERVER={self.datasource['server']};"
                f"DATABASE={self.datasource['database']};"
                f"UID={self.datasource['username']};"
                f"PWD={self.datasource['password']}"
            )
            self.connection = pyodbc.connect(conn_str)
            self.connection.autocommit = False
            self.logger.debug(f"Connected to SQL Server: {self.datasource['name']}")
        except pyodbc.Error as e:
            self.logger.error(f"Failed to connect to SQL Server {self.datasource['name']}: {str(e)}")
            raise OPDENError(f"Failed to connect to SQL Server: {str(e)}")

    def execute_query(self, sql_query: str, user: str, nlq: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute a query and return sample data (5 rows) or CSV path.

        Args:
            sql_query (str): SQL query to execute.
            user (str): User submitting the query (datauser or admin).
            nlq (str): Original natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data (5 rows) and CSV path if acknowledged, else (None, None).

        Raises:
            OPDENError: If query execution fails critically.
        """
        if self.datasource["type"] == "SQL Server":
            return self._execute_sql_server_query(sql_query, user, nlq)
        else:
            return self._execute_s3_query(sql_query, user, nlq)

    def _execute_sql_server_query(self, sql_query: str, user: str, nlq: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute SQL query on SQL Server and return sample data or CSV.

        Args:
            sql_query (str): SQL query to execute.
            user (str): User submitting the query.
            nlq (str): Original natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data and CSV path, or (None, None).
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(5)  # Get 5 rows for sample
            sample_data = pd.DataFrame([tuple(row) for row in rows], columns=columns)
            self.logger.debug(f"Retrieved sample data (5 rows) for NLQ: {nlq}")

            # Assume user acknowledges correctness (placeholder for CLI interaction)
            if not sample_data.empty:
                # Fetch all rows for CSV
                cursor.execute(sql_query)
                all_rows = cursor.fetchall()
                df = pd.DataFrame([tuple(row) for row in all_rows], columns=columns)
                csv_path = os.path.join(self.config_utils.logs_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Generated CSV output: {csv_path}")
                cursor.close()
                return sample_data, csv_path

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
            return None, None

    def _detect_s3_file_type(self, bucket: str, prefix: str) -> str:
        """Detect the file type in the S3 prefix.

        Args:
            bucket (str): S3 bucket name.
            prefix (str): S3 prefix for the datasource.

        Returns:
            str: File format ('parquet', 'csv', 'orc', 'txt').

        Raises:
            OPDENError: If no files found or mixed file types detected.
        """
        s3_client = boto3.client(
            "s3",
            endpoint_url=self.datasource["s3_endpoint"],
            aws_access_key_id=self.datasource["access_key"],
            aws_secret_access_key=self.datasource["secret_key"]
        )
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            files = [obj["Key"] for obj in response.get("Contents", []) if not obj["Key"].endswith("/")]
            if not files:
                raise OPDENError(f"No files found in s3://{bucket}/{prefix}")

            extensions = {os.path.splitext(f)[1].lower() for f in files}
            valid_extensions = {".parquet", ".csv", ".orc", ".txt"}
            detected_extensions = extensions & valid_extensions
            if not detected_extensions:
                raise OPDENError(f"No supported file types found: {extensions}")
            if len(detected_extensions) > 1:
                raise OPDENError(f"Mixed file types detected: {detected_extensions}")

            ext = detected_extensions.pop()
            return {"parquet": "parquet", "csv": "csv", "orc": "orc", "txt": "csv"}[ext[1:]]  # Map txt to csv
        except ClientError as e:
            self.logger.error(f"Failed to list S3 files: {str(e)}")
            raise OPDENError(f"Failed to list S3 files: {str(e)}")

    def _execute_s3_query(self, sql_query: str, user: str, nlq: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute query on S3 data using PyArrow Dataset API.

        Supports Parquet, CSV, ORC, and TXT files with automatic file type detection.

        Args:
            sql_query (str): SQL query to execute (parsed for S3 processing).
            user (str): User submitting the query.
            nlq (str): Original natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data and CSV path, or (None, None).

        Note:
            Simplified implementation; assumes single-table query. Extend with sqlparse for joins/filters.
        """
        try:
            # Initialize S3 client
            s3_client = boto3.client(
                "s3",
                endpoint_url=self.datasource["s3_endpoint"],
                aws_access_key_id=self.datasource["access_key"],
                aws_secret_access_key=self.datasource["secret_key"]
            )

            # Load metadata to identify tables
            metadata = self.config_utils.load_metadata(self.datasource, schema="default")
            tables = metadata.get("tables", [])
            table_name = tables[0]["name"] if tables else None
            if not table_name:
                raise OPDENError("No tables found in metadata")

            # Detect file type
            bucket = self.datasource["bucket_name"]
            prefix = os.path.join(self.datasource["database"], "default", table_name)
            file_format = self._detect_s3_file_type(bucket, prefix)

            # Configure dataset
            s3_filesystem = ds.S3FileSystem(
                endpoint_override=self.datasource["s3_endpoint"],
                access_key=self.datasource["access_key"],
                secret_key=self.datasource["secret_key"]
            )
            format_options = {"format": file_format}
            if file_format == "csv":
                format_options["csv_options"] = {"delimiter": "\t" if file_format == "txt" else ","}
            
            dataset = ds.dataset(
                f"s3://{bucket}/{prefix}",
                **format_options,
                filesystem=s3_filesystem
            )

            # Load sample data (5 rows)
            table = dataset.to_table()
            sample_data = table.slice(0, 5).to_pandas()
            self.logger.debug(f"Retrieved sample data (5 rows) from S3 ({file_format}) for NLQ: {nlq}")

            # Assume user acknowledges correctness
            if not sample_data.empty:
                # Convert entire table to CSV
                df = table.to_pandas()
                csv_path = os.path.join(self.config_utils.logs_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Generated CSV output: {csv_path}")
                return sample_data, csv_path

            self.logger.warning("No data returned from S3")
            return None, None
        except (ClientError, ValueError, OPDENError) as e:
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
            return None, None

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
    # Example usage for testing
    from datetime import datetime
    config_utils = ConfigUtils()
    logging_setup = LoggingSetup(config_utils)
    db_config = config_utils.load_db_config()
    datasource = next(db for db in db_config["databases"] if db["type"] == "SQL Server")
    storage = StorageManager(config_utils, logging_setup, datasource)

    try:
        opden = DataExecutor(config_utils, logging_setup, storage, datasource)
        
        # Test SQL Server query
        sql_query = "SELECT TOP 5 product_name, category_id FROM production.products"
        sample_data, csv_path = opden.execute_query(sql_query, "datauser", "list top 5 products")
        print("Sample Data:\n", sample_data)
        print("CSV Path:", csv_path)
    except (ConfigError, StorageError, OPDENError) as e:
        print(f"Error: {str(e)}")
    finally:
        opden.close_connection()
        storage.close_connection()