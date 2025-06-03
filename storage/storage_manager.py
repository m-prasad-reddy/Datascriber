import boto3
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.orc as orc
import logging
import json
from typing import Dict, Optional, List
from pathlib import Path
from config.utils import ConfigUtils
from config.logging_setup import LoggingSetup

class StorageError(Exception):
    """Custom exception for storage-related errors."""
    pass

class StorageManager:
    """Manages S3 storage operations for the Datascriber project.

    Handles metadata fetching, validation, and data reading for S3 buckets. Supports
    multiple file types (csv, parquet, orc, txt) and schema/table-based configurations.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Datasource-specific logger.
        datasource (Dict): S3 datasource configuration.
        s3_client (boto3.client): S3 client instance.
        bucket_name (str): S3 bucket name.
        database (str): S3 database folder.
    """

    def __init__(self, config_utils: ConfigUtils, logging_setup: LoggingSetup, datasource: Dict):
        """Initialize StorageManager.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logging_setup (LoggingSetup): Logging setup instance.
            datasource (Dict): S3 datasource configuration.

        Raises:
            StorageError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logging_setup.get_logger("storage", datasource.get("name"))
        self.datasource = datasource
        self.bucket_name = datasource["connection"].get("bucket_name")
        self.database = datasource["connection"].get("database", "data")
        self.region = datasource["connection"].get("region", "us-east-1")
        self.s3_client = None
        self._validate_datasource()
        self._init_s3_client()
        self.logger.debug(f"Initialized StorageManager for S3 datasource: {self.datasource['name']}")

    def _validate_datasource(self) -> None:
        """Validate S3 datasource configuration.

        Raises:
            StorageError: If required keys are missing.
        """
        required_keys = ["name", "type", "connection"]
        if not all(key in self.datasource for key in required_keys):
            self.logger.error("Missing required keys in S3 datasource configuration")
            raise StorageError("Missing required keys in S3 datasource configuration")
        if self.datasource["type"] != "s3":
            self.logger.error(f"Invalid datasource type for S3: {self.datasource['type']}")
            raise StorageError(f"Invalid datasource type: {self.datasource['type']}")
        conn_keys = ["bucket_name", "database", "region"]
        if not all(key in self.datasource["connection"] for key in conn_keys):
            self.logger.error("Missing required connection keys for S3 datasource")
            raise StorageError("Missing required connection keys for S3 datasource")
        if not self.datasource["connection"].get("schemas") and not self.datasource["connection"].get("parsed_tables"):
            self.logger.warning("No schemas or tables configured for S3 datasource")

    def _init_s3_client(self) -> None:
        """Initialize S3 client with boto3.

        Uses provided credentials or boto3 defaults.

        Raises:
            StorageError: If client initialization fails.
        """
        try:
            aws_config = self.config_utils.load_aws_config()
            session_params = {"region_name": self.region}
            if aws_config.get("aws_access_key_id") and aws_config.get("aws_secret_access_key"):
                session_params["aws_access_key_id"] = aws_config["aws_access_key_id"]
                session_params["aws_secret_access_key"] = aws_config["aws_secret_access_key"]
            session = boto3.Session(**session_params)
            self.s3_client = session.client("s3")
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.debug(f"Connected to S3 bucket: {self.bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise StorageError(f"Failed to initialize S3 client: {str(e)}")

    def fetch_metadata(self, generate_rich_template: bool = False) -> Dict:
        """Fetch metadata from S3 bucket.

        Scans bucket for schemas (subfolders) and tables (files/folders) based on db_configurations.json.
        Saves metadata to data/<datasourcename>/metadata_data_<schema>.json.

        Args:
            generate_rich_template (bool): If True, generates rich metadata template.

        Returns:
            Dict: Dictionary of schema metadata.

        Raises:
            StorageError: If metadata fetching or saving fails.
        """
        metadata_by_schema = {}
        prefix = f"{self.database}/"
        schemas = self.datasource["connection"].get("schemas", ["default"])
        parsed_tables = self.datasource["connection"].get("parsed_tables", [])

        try:
            for schema in schemas:
                metadata = {"schema": schema, "delimiter": "\t", "tables": []}
                rich_metadata = {"schema": schema, "delimiter": "\t", "tables": []}
                schema_prefix = f"{prefix}{schema}/"

                if parsed_tables:
                    tables = [t["table"] for t in parsed_tables if t["schema"] == schema]
                else:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name, Prefix=schema_prefix, Delimiter="/"
                    )
                    tables = []
                    for obj in response.get("CommonPrefixes", []):
                        table_name = obj["Prefix"].rstrip("/").split("/")[-1]
                        tables.append(table_name)
                    for obj in response.get("Contents", []):
                        if any(obj["Key"].endswith(ext) for ext in [".parquet", ".csv", ".orc", ".txt"]):
                            table_name = obj["Key"].split("/")[-1].rsplit(".", 1)[0]
                            if table_name not in tables:
                                tables.append(table_name)

                for table in tables:
                    table_metadata = {"name": table, "description": "", "columns": []}
                    rich_table_metadata = {"name": table, "description": "", "columns": []}
                    table_path = f"{schema_prefix}{table}/"
                    file_type = self._detect_file_type(table_path)

                    if file_type:
                        try:
                            obj_key = f"{table_path}{table}.{file_type}"
                            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=obj_key)
                            if file_type == "parquet":
                                parquet_file = pq.read_table(obj["Body"])
                                columns = parquet_file.schema.names
                                column_types = [str(field.type) for field in parquet_file.schema]
                            elif file_type == "orc":
                                orc_file = orc.read_table(obj["Body"])
                                columns = orc_file.schema.names
                                column_types = [str(field.type) for field in orc_file.schema]
                            elif file_type in ["csv", "txt"]:
                                df = pd.read_csv(obj["Body"], sep="\t", nrows=0)
                                columns = df.columns.tolist()
                                column_types = ["string"] * len(columns)

                            for col_name, col_type in zip(columns, column_types):
                                col_metadata = {
                                    "name": col_name,
                                    "type": col_type,
                                    "description": "",
                                    "references": None
                                }
                                rich_col_metadata = {
                                    "name": col_name,
                                    "type": col_type,
                                    "description": "",
                                    "references": None,
                                    "unique_values": [],
                                    "synonyms": [],
                                    "range": None,
                                    "date_format": "YYYY-MM-DD" if "timestamp" in col_type.lower() or "date" in col_type.lower() else None
                                }
                                table_metadata["columns"].append(col_metadata)
                                rich_table_metadata["columns"].append(rich_col_metadata)
                        except self.s3_client.exceptions.NoSuchKey:
                            self.logger.warning(f"No {file_type} file found for table {table} in schema {schema}")
                            continue

                    metadata["tables"].append(table_metadata)
                    rich_metadata["tables"].append(rich_table_metadata)

                # Save metadata
                datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource["name"])
                metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(f"Saved base metadata for schema {schema} to {metadata_path}")

                if generate_rich_template:
                    rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
                    with open(rich_metadata_path, "w") as f:
                        json.dump(rich_metadata, f, indent=2)
                    self.logger.info(f"Generated rich metadata template for schema {schema} to {rich_metadata_path}")

                metadata_by_schema[schema] = metadata

            return metadata_by_schema
        except Exception as e:
            self.logger.error(f"Failed to fetch S3 metadata: {str(e)}")
            raise StorageError(f"Failed to fetch S3 metadata: {str(e)}")

    def _detect_file_type(self, table_path: str) -> Optional[str]:
        """Detect file type in an S3 table folder.

        Args:
            table_path (str): S3 path to table folder.

        Returns:
            Optional[str]: File extension ('csv', 'parquet', 'orc', 'txt') or None.

        Raises:
            StorageError: If detection fails.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=table_path
            )
            if "Contents" not in response:
                self.logger.error(f"No files found in S3 path: {table_path}")
                return None
            for obj in response["Contents"]:
                file_key = obj["Key"]
                extension = file_key.rsplit(".", 1)[-1].lower() if "." in file_key else ""
                if extension in ["csv", "parquet", "orc", "txt"]:
                    self.logger.debug(f"Detected file type {extension} at {file_key}")
                    return extension
            self.logger.warning(f"No supported file types found in {table_path}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to detect file type at {table_path}: {str(e)}")
            raise StorageError(f"Failed to detect file type: {str(e)}")

    def read_table_data(self, schema: str, table: str) -> pd.DataFrame:
        """Read table data from S3 into a pandas DataFrame.

        Args:
            schema (str): Schema name.
            table (str): Table name.

        Returns:
            pd.DataFrame: Table data.

        Raises:
            StorageError: If data reading fails.
        """
        table_path = f"{self.database}/{schema}/{table}/"
        file_type = self._detect_file_type(table_path)
        if not file_type:
            self.logger.error(f"No supported file type found for table {table} in schema {schema}")
            raise StorageError(f"No supported file type for table {table}")

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=table_path
            )
            file_key = None
            for obj in response.get("Contents", []):
                if obj["Key"].endswith(f".{file_type}"):
                    file_key = obj["Key"]
                    break
            if not file_key:
                self.logger.error(f"No {file_type} file found in {table_path}")
                raise StorageError(f"No {file_type} file found")

            temp_file = self.config_utils.temp_dir / f"temp_{table}.{file_type}"
            self.config_utils.temp_dir.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, file_key, temp_file)
            self.logger.debug(f"Downloaded S3 file {file_key} to {temp_file}")

            if file_type == "csv":
                df = pd.read_csv(temp_file, sep="\t")
            elif file_type == "parquet":
                df = pq.read_table(temp_file).to_pandas()
            elif file_type == "orc":
                df = orc.read_table(temp_file).to_pandas()
            elif file_type == "txt":
                df = pd.read_csv(temp_file, sep="\t")
            else:
                self.logger.error(f"Unsupported file type: {file_type}")
                raise StorageError(f"Unsupported file type: {file_type}")

            temp_file.unlink(missing_ok=True)
            self.logger.debug(f"Read table {table} from schema {schema} into DataFrame")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read table {table} from schema {schema}: {str(e)}")
            raise StorageError(f"Failed to read table data: {str(e)}")
        finally:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    def execute_query(self, query: str, schema: str, table: str, extracted_values: Dict) -> pd.DataFrame:
        """Execute a SQL-like query on S3 table data.

        Args:
            query (str): SQL query with placeholders.
            schema (str): Schema name.
            table (str): Table name.
            extracted_values (Dict): Values for placeholders.

        Returns:
            pd.DataFrame: Query result.

        Raises:
            StorageError: If query execution fails.
        """
        try:
            df = self.read_table_data(schema, table)
            conditions = []
            for key, value in extracted_values.items():
                if isinstance(value, list):
                    conditions.append(f"{key} in @value")
                else:
                    conditions.append(f"{key} == @value")
            if conditions:
                query_str = " and ".join(conditions)
                result_df = df.query(query_str, local_dict={"value": value for key, value in extracted_values.items()})
            else:
                result_df = df
            self.logger.info(f"Executed query on S3 table {schema}.{table}: {query}")
            return result_df
        except Exception as e:
            self.logger.error(f"Failed to execute query on S3 table {schema}.{table}: {str(e)}")
            raise StorageError(f"Failed to execute query: {str(e)}")

    def validate_metadata(self, schema: str, user_type: str) -> bool:
        """Validate metadata existence for a schema.

        Args:
            schema (str): Schema name.
            user_type (str): 'admin' or 'datauser'.

        Returns:
            bool: True if valid, False if invalid.

        Raises:
            StorageError: If validation fails critically.
        """
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource["name"])
        base_metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
        rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"

        if not base_metadata_path.exists():
            self.logger.error(f"Base metadata missing for schema {schema}: {base_metadata_path}")
            if user_type == "datauser":
                self.logger.info("Logging out datauser due to missing metadata")
                return False
            self.logger.warning("Admin user: Base metadata missing, generating")
            self.fetch_metadata(generate_rich_template=True)
            return True
        if not rich_metadata_path.exists():
            self.logger.warning(f"Rich metadata missing for schema {schema}: {rich_metadata_path}")
            if user_type == "datauser":
                self.logger.info("Logging out datauser due to missing rich metadata")
                return False
            self.logger.warning("Admin user: Rich metadata missing, generating")
            self.fetch_metadata(generate_rich_template=True)
            return True
        return True

    def get_metadata(self, schema: str) -> Dict:
        """Load metadata for a schema, preferring rich metadata.

        Args:
            schema (str): Schema name.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            StorageError: If loading fails.
        """
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource["name"])
        rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
        if rich_metadata_path.exists():
            try:
                with open(rich_metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.debug(f"Loaded rich metadata for schema {schema} from {rich_metadata_path}")
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to load rich metadata: {str(e)}")
                raise StorageError(f"Failed to load rich metadata: {str(e)}")
        base_metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
        if base_metadata_path.exists():
            try:
                with open(base_metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.debug(f"Loaded base metadata for schema {schema} from {base_metadata_path}")
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to load base metadata: {str(e)}")
                raise StorageError(f"Failed to load base metadata: {str(e)}")
        self.logger.info(f"No metadata found for schema {schema}, fetching from S3")
        metadata_by_schema = self.fetch_metadata()
        return metadata_by_schema.get(schema, {"schema": schema, "delimiter": "\t", "tables": []})