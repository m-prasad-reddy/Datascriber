import boto3
import logging
from typing import Dict, List, Optional
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.orc as orc
from io import BytesIO, StringIO
from config.utils import ConfigUtils, ConfigError
import os

class StorageError(Exception):
    """Custom exception for storage-related errors."""
    pass

class StorageManager:
    """Manages S3 storage operations for the Datascriber project.

    Handles reading, writing, and validating S3 files (CSV, ORC, Parquet, TXT).
    Validates training data against metadata, including reference columns.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Logger for storage operations.
        datasource (Dict): Datasource configuration.
        s3_client (boto3.client): S3 client for AWS operations.
        bucket_name (str): S3 bucket name from configuration.
    """

    def __init__(self, config_utils: ConfigUtils, logger: logging.Logger, datasource: Dict):
        """Initialize StorageManager with configuration and S3 client.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): Logger instance.
            datasource (Dict): Datasource configuration.

        Raises:
            StorageError: If S3 client initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        self.datasource = datasource
        try:
            aws_config = self.config_utils.load_aws_config()
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"],
                region_name=aws_config["region"]
            )
            self.bucket_name = aws_config["s3_bucket"]
            self.logger.debug(f"Initialized StorageManager for bucket: {self.bucket_name}")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize StorageManager: {str(e)}")
            raise StorageError(f"Failed to initialize StorageManager: {str(e)}")

    def read_s3_file(self, file_path: str) -> pd.DataFrame:
        """Read a file from S3 and return as a DataFrame.

        Supports CSV, ORC, Parquet, and TXT files, using metadata delimiter for TXT.

        Args:
            file_path (str): S3 file path (e.g., 'data/file.csv').

        Returns:
            pd.DataFrame: DataFrame containing the file data.

        Raises:
            StorageError: If file reading fails or format is unsupported.
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_path)
            file_content = obj["Body"].read()

            if file_extension == ".csv":
                df = pd.read_csv(BytesIO(file_content))
            elif file_extension == ".parquet":
                df = pq.read_table(BytesIO(file_content)).to_pandas()
            elif file_extension == ".orc":
                df = orc.read_table(BytesIO(file_content)).to_pandas()
            elif file_extension == ".txt":
                metadata = self.config_utils.load_metadata(self.datasource, schema="default")
                delimiter = metadata.get("delimiter", "\t")
                df = pd.read_csv(BytesIO(file_content), sep=delimiter)
            else:
                self.logger.error(f"Unsupported file format: {file_extension}")
                raise StorageError(f"Unsupported file format: {file_extension}")

            self.logger.debug(f"Read file from S3: {file_path}, rows: {len(df)}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read S3 file {file_path}: {str(e)}")
            raise StorageError(f"Failed to read S3 file {file_path}: {str(e)}")

    def write_s3_file(self, df: pd.DataFrame, file_path: str) -> None:
        """Write a DataFrame to an S3 file.

        Supports CSV, ORC, Parquet, and TXT files, using metadata delimiter for TXT.

        Args:
            df (pd.DataFrame): DataFrame to write.
            file_path (str): S3 file path (e.g., 'data/output.csv').

        Raises:
            StorageError: If file writing fails or format is unsupported.
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            buffer = BytesIO()

            if file_extension == ".csv":
                df.to_csv(buffer, index=False)
            elif file_extension == ".parquet":
                df.to_parquet(buffer, index=False)
            elif file_extension == ".orc":
                table = orc.Table.from_pandas(df)
                table.write_to(buffer)
            elif file_extension == ".txt":
                metadata = self.config_utils.load_metadata(self.datasource, schema="default")
                delimiter = metadata.get("delimiter", "\t")
                df.to_csv(buffer, sep=delimiter, index=False)
            else:
                self.logger.error(f"Unsupported file format: {file_extension}")
                raise StorageError(f"Unsupported file format: {file_extension}")

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=buffer.getvalue()
            )
            self.logger.debug(f"Wrote file to S3: {file_path}, rows: {len(df)}")
        except Exception as e:
            self.logger.error(f"Failed to write S3 file {file_path}: {str(e)}")
            raise StorageError(f"Failed to write S3 file {file_path}: {str(e)}")

    def validate_training_data(self, training_data: Dict) -> None:
        """Validate training data against metadata.

        Checks if related tables, specific columns, and reference columns exist in metadata.

        Args:
            training_data (Dict): Training data with db_config, db_name, user_query,
                                 related_tables, specific_columns, relevant_sql.

        Raises:
            StorageError: If validation fails due to missing or invalid tables/columns/references.
        """
        try:
            metadata = self.config_utils.load_metadata(self.datasource, schema="default")
            tables = metadata.get("tables", [])
            table_names = {table["name"] for table in tables}

            # Validate related tables
            related_tables = training_data.get("related_tables", "").split(",")
            for table in related_tables:
                if table.strip() and table.strip() not in table_names:
                    self.logger.error(f"Invalid table in training data: {table}")
                    raise StorageError(f"Invalid table in training data: {table}")

            # Validate specific columns and their references
            specific_columns = training_data.get("specific_columns", "").split(",")
            for col in specific_columns:
                col = col.strip()
                if not col:
                    continue
                found = False
                for table in tables:
                    for column in table.get("columns", []):
                        if column["name"] == col and table["name"] in related_tables:
                            found = True
                            # Validate reference if present
                            ref = column.get("references")
                            if ref and isinstance(ref, dict):
                                self._validate_references(ref, tables)
                            break
                    if found:
                        break
                if not found:
                    self.logger.error(f"Invalid column in training data: {col}")
                    raise StorageError(f"Invalid column in training data: {col}")

            self.logger.debug("Training data validation successful")
        except ConfigError as e:
            self.logger.error(f"Failed to load metadata: {str(e)}")
            raise StorageError(f"Failed to load metadata: {str(e)}")
        except Exception as e:
            self.logger.error(f"Training data validation failed: {str(e)}")
            raise StorageError(f"Training data validation failed: {str(e)}")

    def _validate_references(self, ref: Dict, tables: List[Dict]) -> None:
        """Validate a reference column against metadata.

        Args:
            ref (Dict): Reference dictionary with 'table' and 'column' keys.
            tables (List[Dict]): List of table metadata dictionaries.

        Raises:
            StorageError: If the referenced table or column is invalid.
        """
        ref_table = ref.get("table")
        ref_column = ref.get("column")
        if not ref_table or not ref_column:
            self.logger.error("Invalid reference: missing table or column")
            raise StorageError("Invalid reference: missing table or column")

        for table in tables:
            if table["name"] == ref_table:
                for column in table.get("columns", []):
                    if column["name"] == ref_column:
                        return
                self.logger.error(f"Invalid reference column: {ref_table}.{ref_column}")
                raise StorageError(f"Invalid reference column: {ref_table}.{ref_column}")
        self.logger.error(f"Invalid reference table: {ref_table}")
        raise StorageError(f"Invalid reference table: {ref_table}")