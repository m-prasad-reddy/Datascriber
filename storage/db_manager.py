import pyodbc
from typing import Dict, List, Optional
import logging
from datetime import datetime
from config.utils import ConfigUtils, ConfigError

class DBError(Exception):
    """Custom exception for database-related errors."""
    pass

class DBManager:
    """Manages SQL Server database operations for the Datascriber project.

    Handles connections and operations for training_data_<name>, rejected_queries_<name>,
    and model_metrics_<name> tables.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Logger for database operations.
        datasource (Dict): SQL Server datasource configuration.
        connection (pyodbc.Connection): SQL Server connection.
    """

    def __init__(self, config_utils: ConfigUtils, logger: logging.Logger, datasource: Dict):
        """Initialize DBManager.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): Logger instance.
            datasource (Dict): SQL Server datasource configuration.

        Raises:
            DBError: If connection fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        self.datasource = datasource
        self.connection = None
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize SQL Server connection.

        Raises:
            DBError: If connection fails.
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
            raise DBError(f"Failed to connect to SQL Server: {str(e)}")

    def create_training_table(self) -> None:
        """Create training_data_<name> table.

        Raises:
            DBError: If table creation fails.
        """
        table_name = f"training_data_{self.datasource['name']}"
        create_query = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}')
        CREATE TABLE {table_name} (
            id INT IDENTITY(1,1) PRIMARY KEY,
            db_config NVARCHAR(255),
            db_name NVARCHAR(255),
            user_query NVARCHAR(MAX),
            related_tables NVARCHAR(MAX),
            specific_columns NVARCHAR(MAX),
            relevant_sql NVARCHAR(MAX),
            timestamp DATETIME DEFAULT GETDATE()
        )
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_query)
            self.connection.commit()
            self.logger.info(f"Created table {table_name}")
        except pyodbc.Error as e:
            self.connection.rollback()
            self.logger.error(f"Failed to create table {table_name}: {str(e)}")
            raise DBError(f"Failed to create table {table_name}: {str(e)}")
        finally:
            cursor.close()

    def create_rejected_queries_table(self) -> None:
        """Create rejected_queries_<name> table.

        Raises:
            DBError: If table creation fails.
        """
        table_name = f"rejected_queries_{self.datasource['name']}"
        create_query = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}')
        CREATE TABLE {table_name} (
            id INT IDENTITY(1,1) PRIMARY KEY,
            query NVARCHAR(MAX),
            timestamp DATETIME DEFAULT GETDATE(),
            reason NVARCHAR(255),
            user NVARCHAR(50),
            datasource NVARCHAR(255),
            error_type NVARCHAR(50)
        )
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_query)
            self.connection.commit()
            self.logger.info(f"Created table {table_name}")
        except pyodbc.Error as e:
            self.connection.rollback()
            self.logger.error(f"Failed to create table {table_name}: {str(e)}")
            raise DBError(f"Failed to create table {table_name}: {str(e)}")
        finally:
            cursor.close()

    def create_model_metrics_table(self) -> None:
        """Create model_metrics_<name> table.

        Raises:
            DBError: If table creation fails.
        """
        table_name = f"model_metrics_{self.datasource['name']}"
        create_query = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}')
        CREATE TABLE {table_name} (
            id INT IDENTITY(1,1) PRIMARY KEY,
            model_version NVARCHAR(50),
            timestamp DATETIME DEFAULT GETDATE(),
            precision FLOAT,
            recall FLOAT,
            nlq_breakdown NVARCHAR(MAX)
        )
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_query)
            self.connection.commit()
            self.logger.info(f"Created table {table_name}")
        except pyodbc.Error as e:
            self.connection.rollback()
            self.logger.error(f"Failed to create table {table_name}: {str(e)}")
            raise DBError(f"Failed to create table {table_name}: {str(e)}")
        finally:
            cursor.close()

    def store_training_data(self, data: Dict) -> None:
        """Store training data.

        Args:
            data (Dict): Training data with db_config, db_name, user_query, related_tables,
                         specific_columns, relevant_sql.

        Raises:
            DBError: If storage fails.
        """
        table_name = f"training_data_{self.datasource['name']}"
        required_keys = ["db_config", "db_name", "user_query", "related_tables", "specific_columns", "relevant_sql"]
        for key in required_keys:
            if key not in data:
                self.logger.error(f"Missing key '{key}' in training data")
                raise DBError(f"Missing key '{key}'")

        try:
            cursor = self.connection.cursor()
            check_query = f"SELECT id FROM {table_name} WHERE user_query = ?"
            cursor.execute(check_query, data["user_query"])
            existing = cursor.fetchone()

            if existing:
                update_query = f"""
                UPDATE {table_name}
                SET db_config = ?, db_name = ?, related_tables = ?, specific_columns = ?, relevant_sql = ?, timestamp = ?
                WHERE user_query = ?
                """
                cursor.execute(update_query, (
                    data["db_config"], data["db_name"], data["related_tables"],
                    data["specific_columns"], data["relevant_sql"], datetime.now(), data["user_query"]
                ))
                self.logger.info(f"Updated training data for NLQ: {data['user_query']}")
            else:
                insert_query = f"""
                INSERT INTO {table_name} (db_config, db_name, user_query, related_tables, specific_columns, relevant_sql, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (
                    data["db_config"], data["db_name"], data["user_query"],
                    data["related_tables"], data["specific_columns"], data["relevant_sql"], datetime.now()
                ))
                self.logger.info(f"Stored training data for NLQ: {data['user_query']}")

            self.connection.commit()
        except pyodbc.Error as e:
            self.connection.rollback()
            self.logger.error(f"Failed to store training data: {str(e)}")
            raise DBError(f"Failed to store training data: {str(e)}")
        finally:
            cursor.close()

    def store_rejected_query(self, query: str, reason: str, user: str, error_type: str) -> None:
        """Store rejected query.

        Args:
            query (str): Rejected NLQ.
            reason (str): Rejection reason.
            user (str): User who submitted the query.
            error_type (str): Error type (e.g., TIA_FAILURE).

        Raises:
            DBError: If storage fails.
        """
        table_name = f"rejected_queries_{self.datasource['name']}"
        insert_query = f"""
        INSERT INTO {table_name} (query, timestamp, reason, user, datasource, error_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(insert_query, (
                query, datetime.now(), reason, user,
                self.datasource["name"], error_type
            ))
            self.connection.commit()
            self.logger.info(f"Stored rejected query: {query}")
        except pyodbc.Error as e:
            self.connection.rollback()
            self.logger.error(f"Failed to store rejected query: {str(e)}")
            raise DBError(f"Failed to store rejected query: {str(e)}")
        finally:
            cursor.close()

    def store_model_metrics(self, metrics: Dict) -> None:
        """Store model metrics.

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
            cursor = self.connection.cursor()
            cursor.execute(insert_query, (
                metrics["model_version"], datetime.now(),
                metrics["precision"], metrics["recall"],
                json.dumps(metrics["nlq_breakdown"])
            ))
            self.connection.commit()
            self.logger.info(f"Stored model metrics: version={metrics['model_version']}")
        except pyodbc.Error as e:
            self.connection.rollback()
            self.logger.error(f"Failed to store model metrics: {str(e)}")
            raise DBError(f"Failed to store model metrics: {str(e)}")
        finally:
            cursor.close()

    def close_connection(self) -> None:
        """Close SQL Server connection."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.debug(f"Closed SQL Server connection: {self.datasource['name']}")
            except pyodbc.Error as e:
                self.logger.error(f"Failed to close connection: {str(e)}")
            finally:
                self.connection = None