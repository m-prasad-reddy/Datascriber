import json
import sqlite3
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional
import pyodbc
import traceback
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup

class DBError(Exception):
    """Custom exception for database-related errors."""
    pass

class DBManager:
    """Manages database operations for the Datascriber project.

    Handles SQL Server metadata fetching (read-only) and SQLite storage for training data,
    rejected queries, model metrics, and rich metadata. Supports hybrid metadata approach
    with semi-automated base metadata and manual rich metadata.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Datasource-specific logger.
        datasource (Dict): SQL Server datasource configuration.
        sql_server_conn (pyodbc.Connection): SQL Server connection.
        sqlite_conn (sqlite3.Connection): SQLite connection.
        sqlite_db_path (Path): Path to SQLite database file.
    """

    def __init__(self, config_utils: ConfigUtils, logging_setup: LoggingSetup, database: Dict, user_type: str = "datauser"):
        """Initialize DBManager.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logging_setup (LoggingSetup): Logging setup instance.
            database (Dict): SQL Server datasource configuration.
            user_type (str): 'admin' or 'datauser', defaults to 'datauser'.

        Raises:
            DBError: If connection or initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logging_setup.get_logger("db", database.get("name"))
        self.datasource = database
        self.sql_server_conn = None
        self.sqlite_conn = None
        self.sqlite_db_path = self.config_utils.get_datasource_data_dir(self.datasource['name']) / f"{self.datasource['name']}.db"
        self._validate_datasource()
        self._init_sqlite_connection()
        if self.datasource["type"] == "sqlserver":
            self._init_sql_server_connection()
            self._update_existing_rich_metadata()
            if user_type == "admin":
                self._ensure_all_data()

    def _validate_datasource(self) -> None:
        """Validate datasource configuration.

        Raises:
            DBError: If required keys are missing.
        """
        required_keys = ["name", "type", "connection"]
        if not all(key in self.datasource for key in required_keys):
            self.logger.error("Missing required keys in datasource configuration")
            raise DBError("Missing required keys in datasource configuration")
        if self.datasource["type"] == "sqlserver":
            conn_keys = ["host", "port", "database", "username", "password"]
            if not all(key in self.datasource["connection"] for key in conn_keys):
                self.logger.error("Missing required connection keys for SQL Server")
                raise DBError("Missing required connection keys for SQL Server")
        self.logger.debug(f"Validated datasource: {self.datasource['name']}")

    def _init_sql_server_connection(self) -> None:
        """Initialize read-only SQL Server connection.

        Raises:
            DBError: If connection fails.
        """
        try:
            available_drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
            driver = next((d for d in available_drivers if "ODBC Driver" in d), "SQL Server")
            if not driver:
                self.logger.error("No suitable SQL Server ODBC driver found")
                raise DBError("No suitable SQL Server ODBC driver found")

            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.datasource['connection']['host']},{self.datasource['connection']['port']};"
                f"DATABASE={self.datasource['connection']['database']};"
                f"UID={self.datasource['connection']['username']};"
                f"PWD={self.datasource['connection']['password']};"
                f"ReadOnly=True"
            )
            self.sql_server_conn = pyodbc.connect(conn_str)
            self.logger.debug(f"Connected to SQL Server: {self.datasource['name']}")
        except pyodbc.Error as e:
            self.logger.error(f"Failed to connect to SQL Server {self.datasource['name']}: {str(e)}")
            raise DBError(f"Failed to connect to SQL Server: {str(e)}")

    def _init_sqlite_connection(self) -> None:
        """Initialize SQLite connection and create tables.

        Raises:
            DBError: If SQLite connection or table creation fails.
        """
        try:
            self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.sqlite_conn = sqlite3.connect(self.sqlite_db_path, check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            self.logger.debug(f"Initialized SQLite connection to: {self.sqlite_db_path}")
            self._create_sqlite_tables()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize SQLite database {self.sqlite_db_path}: {str(e)}")
            raise DBError(f"Failed to initialize SQLite database: {str(e)}")

    def _create_sqlite_tables(self) -> None:
        """Create SQLite tables for training data, rejected queries, metrics, and rich metadata.

        Raises:
            DBError: If table creation fails.
        """
        tables = [
            (
                f"training_data_{self.datasource['name']}",
                """
                CREATE TABLE IF NOT EXISTS training_data_{name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    db_source_type TEXT,
                    db_name TEXT,
                    user_query TEXT,
                    related_tables TEXT,
                    specific_columns TEXT,
                    extracted_values TEXT,
                    placeholders TEXT,
                    relevant_sql TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            ),
            (
                f"rejected_queries_{self.datasource['name']}",
                """
                CREATE TABLE IF NOT EXISTS rejected_queries_{name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp TEXT,
                    reason TEXT,
                    user TEXT,
                    datasource TEXT,
                    error_type TEXT
                )
                """
            ),
            (
                f"model_metrics_{self.datasource['name']}",
                """
                CREATE TABLE IF NOT EXISTS model_metrics_{name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    timestamp TEXT,
                    precision REAL,
                    recall REAL,
                    nlq_breakdown TEXT
                )
                """
            ),
            (
                "rich_metadata",
                """
                CREATE TABLE IF NOT EXISTS rich_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datasource_name TEXT,
                    schema_name TEXT,
                    table_name TEXT,
                    column_name TEXT,
                    unique_values TEXT,
                    synonyms TEXT,
                    range TEXT,
                    date_format TEXT,
                    UNIQUE(datasource_name, schema_name, table_name, column_name)
                )
                """
            )
        ]

        try:
            cursor = self.sqlite_conn.cursor()
            for table_name, create_query in tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if cursor.fetchone():
                    cursor.execute(f"SELECT count(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    self.logger.debug(f"Table {table_name} exists with {row_count} rows, skipping creation")
                    continue
                cursor.execute(create_query.format(name=self.datasource['name']))
                self.logger.info(f"Created SQLite table {table_name} in {self.sqlite_db_path}")
            self.sqlite_conn.commit()
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to create SQLite tables: {str(e)}")
            raise DBError(f"Failed to create SQLite tables: {str(e)}")
        finally:
            cursor.close()

    def _update_existing_rich_metadata(self) -> None:
        """Update SQLite rich_metadata table with existing rich metadata JSON files."""
        schemas = self.datasource["connection"].get("schemas", [])
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
        for schema in schemas:
            rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
            if rich_metadata_path.exists():
                try:
                    self.update_rich_metadata(schema)
                    self.logger.debug(f"Populated rich metadata for schema {schema} from {rich_metadata_path}")
                except DBError as e:
                    self.logger.warning(f"Failed to populate rich metadata for schema {schema}: {str(e)}")

    def _ensure_all_data(self) -> None:
        """Ensure metadata files exist for configured schemas, generating them if missing.

        Raises:
            DBError: If metadata generation fails.
        """
        schemas = self.datasource["connection"].get("schemas", [])
        parsed_tables = self.datasource["connection"].get("parsed_tables", [])
        self.logger.debug(f"Checking metadata for schemas: {schemas}")
        if not schemas and not parsed_tables:
            self.logger.warning("No schemas or tables configured for metadata generation")
            return
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
        for schema in schemas:
            base_metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
            if not base_metadata_path.exists():
                self.logger.info(f"Metadata missing for schema {schema}, generating")
                self.fetch_metadata(generate_rich_template=True)

    def fetch_metadata(self, generate_rich_template: bool = False) -> Dict:
        """Fetch metadata from SQL Server and save to schema-specific JSON files.

        Uses schemas and tables from db_configurations.json, excluding system schemas.
        Saves base metadata to data/<datasourcename>/metadata_data_<schema>.json.

        Args:
            generate_rich_template (bool): If True, generates rich metadata template.

        Returns:
            Dict: Dictionary of schema metadata.

        Raises:
            DBError: If metadata fetching or saving fails.
        """
        if self.datasource["type"] != "sqlserver":
            self.logger.error("Metadata fetching only supported for SQL Server")
            raise DBError("Metadata fetching only supported for SQL Server")

        metadata_by_schema = {}
        system_schemas = ["sys", "information_schema", "guest", "db_owner", "db_accessadmin"]
        schemas = [s for s in self.datasource["connection"].get("schemas", []) if s.lower() not in system_schemas]
        parsed_tables = self.datasource["connection"].get("parsed_tables", [])

        if not schemas and not parsed_tables:
            self.logger.warning("No schemas or parsed tables configured")
            return metadata_by_schema

        try:
            cursor = self.sql_server_conn.cursor()

            if parsed_tables:
                schema_tables = {}
                for table in parsed_tables:
                    schema = table["schema"]
                    if schema not in schema_tables:
                        schema_tables[schema] = []
                    schema_tables[schema].append(table["table"])
            else:
                schema_tables = {schema: None for schema in schemas}

            for schema in schema_tables:
                metadata = {"schema": schema, "delimiter": "\t", "tables": []}
                rich_metadata = {"schema": schema, "delimiter": "\t", "tables": []}

                if schema_tables[schema]:
                    tables = schema_tables[schema]
                else:
                    cursor.execute("""
                        SELECT TABLE_NAME
                        FROM INFORMATION_SCHEMA.TABLES
                        WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'
                    """, schema)
                    tables = [row[0] for row in cursor.fetchall()]

                if not tables:
                    self.logger.warning(f"No tables found for schema {schema}")

                for table in tables:
                    table_metadata = {"name": table, "description": "", "columns": []}
                    rich_table_metadata = {"name": table, "description": "", "columns": []}

                    cursor.execute("""
                        SELECT COLUMN_NAME, DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    """, schema, table)
                    columns = cursor.fetchall()

                    for column in columns:
                        col_metadata = {
                            "name": column[0],
                            "type": column[1],
                            "description": "",
                            "references": None
                        }
                        rich_col_metadata = {
                            "name": column[0],
                            "type": column[1],
                            "description": "",
                            "references": None,
                            "unique_values": [],
                            "synonyms": [],
                            "range": None,
                            "date_format": "YYYY-MM-DD" if column[1] in ["DATE", "DATETIME"] else None
                        }

                        cursor.execute("""
                            SELECT 
                                kcu2.TABLE_SCHEMA AS ref_schema,
                                kcu2.TABLE_NAME AS ref_table,
                                kcu2.COLUMN_NAME AS ref_column
                            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu1
                                ON rc.CONSTRAINT_NAME = kcu1.CONSTRAINT_NAME
                                AND kcu1.TABLE_SCHEMA = ?
                                AND kcu1.TABLE_NAME = ?
                                AND kcu1.COLUMN_NAME = ?
                            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu2
                                ON rc.UNIQUE_CONSTRAINT_NAME = kcu2.CONSTRAINT_NAME
                        """, schema, table, column[0])
                        ref = cursor.fetchone()
                        if ref:
                            ref_table = f"{ref[0]}.{ref[1]}" if ref[0] != schema else ref[1]
                            col_metadata["references"] = {"table": ref_table, "column": ref[2]}
                            rich_col_metadata["references"] = {"table": ref_table, "column": ref[2]}

                        table_metadata["columns"].append(col_metadata)
                        rich_table_metadata["columns"].append(rich_col_metadata)

                    metadata["tables"].append(table_metadata)
                    rich_metadata["tables"].append(rich_table_metadata)

                datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
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
        except pyodbc.Error as e:
            self.logger.error(f"Failed to fetch SQL Server metadata: {str(e)}")
            raise DBError(f"Failed to fetch SQL Server metadata: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise DBError(f"Failed to save metadata: {str(e)}")
        finally:
            cursor.close()

    def get_metadata(self, schema: str) -> Dict:
        """Load metadata for a schema, preferring rich metadata if available.

        Args:
            schema (str): Schema name.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            DBError: If metadata loading fails.
        """
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
        rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
        if rich_metadata_path.exists():
            try:
                with open(rich_metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.debug(f"Loaded rich metadata for schema {schema} from {rich_metadata_path}")
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to load rich metadata from {rich_metadata_path}: {str(e)}")
                raise DBError(f"Failed to load rich metadata: {str(e)}")

        base_metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
        if base_metadata_path.exists():
            try:
                with open(base_metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.debug(f"Loaded base metadata for schema {schema} from {base_metadata_path}")
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to load base metadata from {base_metadata_path}: {str(e)}")
                raise DBError(f"Failed to load base metadata: {str(e)}")

        if self.datasource["type"] == "sqlserver":
            self.logger.info(f"No metadata found for schema {schema}, fetching from SQL Server")
            metadata_by_schema = self.fetch_metadata(generate_rich_template=True)
            return metadata_by_schema.get(schema, {"schema": schema, "delimiter": "\t", "tables": []})
        else:
            self.logger.error(f"No metadata available for schema {schema} and datasource type {self.datasource['type']}")
            raise DBError(f"No metadata available for schema {schema}")

    def update_rich_metadata(self, schema: str) -> None:
        """Load rich metadata from JSON file and update SQLite rich_metadata table.

        Args:
            schema (str): Schema name.

        Raises:
            DBError: If update fails.
        """
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
        rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"
        if not rich_metadata_path.exists():
            self.logger.error(f"Rich metadata file not found: {rich_metadata_path}")
            raise DBError(f"Rich metadata file not found: {rich_metadata_path}")

        try:
            self.logger.debug(f"Loading rich metadata from {rich_metadata_path}")
            with open(rich_metadata_path, "r") as f:
                rich_metadata = json.load(f)
            self.logger.debug(f"Loaded rich metadata for schema {schema} with {len(rich_metadata.get('tables', []))} tables")

            cursor = self.sqlite_conn.cursor()
            rows_inserted = 0
            for table in rich_metadata.get("tables", []):
                for column in table.get("columns", []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO rich_metadata (
                            datasource_name, schema_name, table_name, column_name,
                            unique_values, synonyms, range, date_format
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        self.datasource["name"],
                        schema,
                        table["name"],
                        column["name"],
                        json.dumps(column.get("unique_values", [])),
                        json.dumps(column.get("synonyms", [])),
                        json.dumps(column.get("range", None)),
                        column.get("date_format", None)
                    ))
                    rows_inserted += 1
            self.sqlite_conn.commit()
            self.logger.info(f"Updated rich metadata for schema {schema} in SQLite with {rows_inserted} rows")
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to update rich metadata for schema {schema}: {str(e)}")
            raise DBError(f"Failed to update rich metadata: {str(e)}")
        finally:
            cursor.close()

    def validate_metadata(self, schema: str, user_type: str) -> bool:
        """Validate metadata existence for a schema.

        Args:
            schema (str): Schema name.
            user_type (str): 'admin' or 'datauser'.

        Returns:
            bool: True if valid, False if invalid (with appropriate logging/action).

        Raises:
            DBError: If validation fails critically.
        """
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
        base_metadata_path = datasource_data_dir / f"metadata_data_{schema}.json"
        rich_metadata_path = datasource_data_dir / f"metadata_data_{schema}_rich.json"

        if not base_metadata_path.exists():
            self.logger.error(f"Base metadata missing for schema {schema}: {base_metadata_path}")
            if user_type == "datauser":
                self.logger.info("Logging out datauser due to missing base metadata")
                return False
            self.logger.warning("Admin user: Base metadata missing, generating")
            if self.datasource["type"] == "sqlserver":
                self.fetch_metadata(generate_rich_template=True)
            else:
                return False

        if user_type == "datauser":
            cursor = self.sqlite_conn.cursor()
            try:
                cursor.execute("""
                    SELECT COUNT(*) FROM rich_metadata
                    WHERE datasource_name = ? AND schema_name = ?
                """, (self.datasource["name"], schema))
                rich_count = cursor.fetchone()[0]
                if rich_count == 0:
                    self.logger.warning(f"No rich metadata in SQLite for schema {schema}")
                    if rich_metadata_path.exists():
                        self.update_rich_metadata(schema)
                    self.logger.info(f"Data user proceeding with base metadata for schema {schema}")
                return True
            except sqlite3.Error as e:
                self.logger.error(f"Failed to validate rich metadata: {str(e)}")
                raise DBError(f"Failed to validate rich metadata: {str(e)}")
            finally:
                cursor.close()
        else:  # Admin user
            cursor = self.sqlite_conn.cursor()
            try:
                cursor.execute("""
                    SELECT COUNT(*) FROM rich_metadata
                    WHERE datasource_name = ? AND schema_name = ?
                """, (self.datasource["name"], schema))
                rich_count = cursor.fetchone()[0]
                if rich_count == 0:
                    self.logger.warning(f"No rich metadata in SQLite for schema {schema}")
                    if rich_metadata_path.exists():
                        self.update_rich_metadata(schema)
                        return True
                    self.logger.warning("Admin user: Rich metadata missing, please provide")
                    return False
                return True
            except sqlite3.Error as e:
                self.logger.error(f"Failed to validate rich metadata: {str(e)}")
                raise DBError(f"Failed to validate rich metadata: {str(e)}")
            finally:
                cursor.close()

    def store_training_data(self, data: Dict) -> None:
        """Store training data in SQLite.

        Args:
            data (Dict): Training data with db_source_type, db_name, user_query, etc.

        Raises:
            DBError: If storage fails.
        """
        table_name = f"training_data_{self.datasource['name']}"
        required_keys = ["db_source_type", "db_name", "user_query", "related_tables", "specific_columns", "relevant_sql"]
        optional_keys = ["extracted_values", "placeholders"]
        for key in required_keys:
            if key not in data:
                self.logger.error(f"Missing key '{key}' in training data")
                raise DBError(f"Missing key '{key}'")
        extracted_values = json.dumps(data.get("extracted_values", {}))
        placeholders = json.dumps(data.get("placeholders", []))

        try:
            cursor = self.sqlite_conn.cursor()
            check_query = f"SELECT id FROM {table_name} WHERE user_query = ?"
            cursor.execute(check_query, (data["user_query"],))
            existing = cursor.fetchone()
            self.logger.debug(f"Checked for existing query '{data['user_query']}': {'found' if existing else 'not found'}")

            if existing:
                update_query = f"""
                UPDATE {table_name}
                SET db_source_type = ?, db_name = ?, related_tables = ?, specific_columns = ?,
                    extracted_values = ?, placeholders = ?, relevant_sql = ?, timestamp = ?
                WHERE user_query = ?
                """
                cursor.execute(update_query, (
                    data["db_source_type"], data["db_name"], data["related_tables"],
                    data["specific_columns"], extracted_values, placeholders,
                    data["relevant_sql"], datetime.now().isoformat(), data["user_query"]
                ))
                self.logger.debug(f"Updated training data for NLQ: {data['user_query']}")
            else:
                insert_query = f"""
                INSERT INTO {table_name} (
                    db_source_type, db_name, user_query, related_tables, specific_columns,
                    extracted_values, placeholders, relevant_sql, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (
                    data["db_source_type"], data["db_name"], data["user_query"],
                    data["related_tables"], data["specific_columns"], extracted_values,
                    placeholders, data["relevant_sql"], datetime.now().isoformat()
                ))
                self.logger.debug(f"Inserted training data for NLQ: {data['user_query']}")

            self.sqlite_conn.commit()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE user_query = ?", (data["user_query"],))
            row_count = cursor.fetchone()[0]
            self.logger.info(f"Stored training data for NLQ: {data['user_query']}, table now has {row_count} matching rows")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to store training data: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to store training data: {str(e)}")
        finally:
            cursor.close()

    def store_rejected_query(self, query: str, reason: str, user: str, error_type: str) -> None:
        """Store rejected query in SQLite.

        Args:
            query (str): Rejected NLQ.
            reason (str): Rejection reason.
            user (str): User who submitted the query.
            error_type (str): Error type.

        Raises:
            DBError: If storage fails.
        """
        table_name = f"rejected_queries_{self.datasource['name']}"
        insert_query = f"""
            INSERT INTO {table_name} (query, timestamp, reason, user, datasource, error_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(insert_query, (
                query, datetime.now().isoformat(), reason, user,
                self.datasource["name"], error_type
            ))
            self.sqlite_conn.commit()
            self.logger.info(f"Stored rejected query: {query}")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to store rejected query: {str(e)}")
            raise DBError(f"Failed to store rejected query: {str(e)}")
        finally:
            cursor.close()

    def log_rejected_query(self, nlq: str, datasource: str) -> None:
        """Log a rejected query to SQLite (compatibility method).

        Args:
            nlq (str): Natural language query.
            datasource (str): Datasource name.

        Raises:
            DBError: If storage fails.
        """
        try:
            self.store_rejected_query(
                query=nlq,
                reason="Failed to process NLQ",
                user="unknown",
                error_type="NLQProcessingFailure"
            )
        except Exception as e:
            self.logger.error(f"Failed to log rejected query '{nlq}': {str(e)}")
            raise DBError(f"Failed to log rejected query: {str(e)}")

    def get_training_data(self, datasource: str) -> List[Dict]:
        """Retrieve training data from SQLite.

        Args:
            datasource (str): Datasource name.

        Returns:
            List[Dict]: List of training data entries.

        Raises:
            DBError: If retrieval fails.
        """
        self.logger.debug(f"Entering get_training_data for datasource: {datasource}")
        table_name = f"training_data_{datasource}"
        query = f"""
            SELECT db_source_type, db_name, user_query, related_tables,
                   specific_columns, extracted_values, placeholders, relevant_sql
            FROM {table_name}
        """
        self.logger.debug(f"Executing query: {query} on table: {table_name}")
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            self.logger.debug(f"Fetched {len(rows)} rows from {table_name}")
            data = [
                {
                    "db_source_type": row["db_source_type"],
                    "db_name": row["db_name"],
                    "user_query": row["user_query"],
                    "related_tables": row["related_tables"],
                    "specific_columns": row["specific_columns"],
                    "extracted_values": json.loads(row["extracted_values"]),
                    "placeholders": json.loads(row["placeholders"]),
                    "relevant_sql": row["relevant_sql"]
                }
                for row in rows
            ]
            if not data:
                self.logger.warning(f"No training data found for {datasource}")
            else:
                self.logger.info(f"Retrieved {len(data)} training data entries for {datasource}")
            return data
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve training data for {datasource}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to retrieve training data: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON in training data for {datasource}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Failed to parse JSON in training data: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving training data for {datasource}: {str(e)}\n{traceback.format_exc()}")
            raise DBError(f"Unexpected error retrieving training data: {str(e)}")
        finally:
            cursor.close()

    def store_model_metrics(self, metrics: Dict) -> None:
        """Store model metrics in SQLite.

        Args:
            metrics (Dict): Metrics with model_version, precision, recall, nlq_breakdown.

        Raises:
            DBError: If storage fails.
        """
        table_name = f"model_metrics_{self.datasource['name']}"
        insert_query = f"""
            INSERT INTO {table_name} (model_version, timestamp, precision, recall, nlq_breakdown)
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(insert_query, (
                metrics["model_version"], datetime.now().isoformat(),
                metrics["precision"], metrics["recall"],
                json.dumps(metrics["nlq_breakdown"])
            ))
            self.sqlite_conn.commit()
            self.logger.info(f"Stored model metrics: version={metrics['model_version']}")
        except sqlite3.Error as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"Failed to store model metrics: {str(e)}")
            raise DBError(f"Failed to store model metrics: {str(e)}")
        finally:
            cursor.close()

    def close_connections(self) -> None:
        """Close SQL Server and SQLite connections."""
        if self.sql_server_conn:
            try:
                self.sql_server_conn.close()
                self.logger.debug(f"Closed SQL Server connection: {self.datasource['name']}")
            except pyodbc.Error as e:
                self.logger.error(f"Failed to close SQL Server connection: {str(e)}")
            finally:
                self.sql_server_conn = None
        if self.sqlite_conn:
            try:
                self.sqlite_conn.commit()
                self.sqlite_conn.close()
                self.logger.debug(f"Closed SQLite connection: {self.sqlite_db_path}")
            except sqlite3.Error as e:
                self.logger.error(f"Failed to close SQLite connection: {str(e)}")
            finally:
                self.sqlite_conn = None