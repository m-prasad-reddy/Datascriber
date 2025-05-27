import getpass
import logging
import pandas as pd
from typing import Dict, List, Optional
import csv
from io import StringIO
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError
from tia.table_identifier import TableIdentifier, TIAError
from proga.prompt_generator import PromptGenerator, PROGAError
from opden.data_executor import DataExecutor, OPDENError

class CLIError(Exception):
    """Custom exception for CLI interface errors."""
    pass

class CLIInterface:
    """Command-line interface for the Datascriber project.

    Provides role-based access for End Users (datauser: query execution) and Admin Users (admin: training, model management).
    Handles authentication, datasource selection, and interaction with TIA, PROGA, and OPDEN components.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance for loading configs.
        logger (logging.Logger): Logger for CLI operations.
        db_config (Dict): Database configuration from db_configurations.json.
        user (Optional[str]): Authenticated user (datauser or admin).
        datasource (Optional[Dict]): Selected datasource configuration.
        db_manager (Optional[DBManager]): SQL Server database manager.
        storage_manager (Optional[StorageManager]): S3 storage manager.
        tia (Optional[TableIdentifier]): Table Identifier Agent instance.
        proga (Optional[PromptGenerator]): Prompt Generator Agent instance.
        opden (Optional[DataExecutor]): Data Executor instance.
    """

    def __init__(self):
        """Initialize CLIInterface with configuration and logging.

        Raises:
            CLIError: If initialization fails due to configuration issues.
        """
        try:
            self.config_utils = ConfigUtils()
            self.logger = LoggingSetup(self.config_utils).get_logger("cli")
            self.db_config = self.config_utils.load_db_config()
            self.user = None
            self.datasource = None
            self.db_manager = None
            self.storage_manager = None
            self.tia = None
            self.proga = None
            self.opden = None
            self.logger.debug("Initialized CLIInterface")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize CLI: {str(e)}")
            raise CLIError(f"Failed to initialize CLI: {str(e)}")

    def authenticate(self) -> bool:
        """Authenticate user with hardcoded credentials.

        Prompts for username and password, validating against datauser/admin with password 'pass123'.

        Returns:
            bool: True if authentication succeeds, False otherwise.
        """
        print("Enter username (datauser or admin):")
        username = input().strip()
        password = getpass.getpass("Enter password: ").strip()

        valid_users = {"datauser": "pass123", "admin": "pass123"}
        if username in valid_users and password == valid_users[username]:
            self.user = username
            self.logger.info(f"Authenticated user: {username}")
            return True
        self.logger.error(f"Authentication failed for user: {username}")
        print("Invalid username or password")
        return False

    def select_datasource(self) -> bool:
        """Prompt user to select a datasource from available options.

        Initializes component instances (db_manager, storage_manager, tia, proga, opden) upon successful selection.

        Returns:
            bool: True if selection succeeds, False otherwise.
        """
        print("\nAvailable Datasources:")
        for idx, db in enumerate(self.db_config["databases"], 1):
            print(f"{idx}. {db['display_name']} ({db['name']})")

        try:
            choice = int(input("\nSelect datasource (number): ")) - 1
            if 0 <= choice < len(self.db_config["databases"]):
                self.datasource = self.db_config["databases"][choice]
                self.logger.info(f"Selected datasource: {self.datasource['name']}")
                self.db_manager = DBManager(self.config_utils, self.logger, self.datasource)
                self.storage_manager = StorageManager(self.config_utils, self.logger, self.datasource)
                self.tia = TableIdentifier(self.config_utils, self.logger, self.db_manager, self.storage_manager, self.datasource)
                self.proga = PromptGenerator(self.config_utils, self.logger, self.db_manager, self.datasource)
                self.opden = DataExecutor(self.config_utils, self.logger, self.db_manager, self.storage_manager, self.datasource)
                self.logger.debug("Initialized components for datasource")
                return True
            print("Invalid selection")
            self.logger.error("Invalid datasource selection")
            return False
        except ValueError:
            print("Please enter a valid number")
            self.logger.error("Invalid input for datasource selection")
            return False

    def run(self) -> None:
        """Run the CLI interface, handling authentication and role-based menus.

        Ensures proper cleanup of database connections on exit.
        """
        try:
            if not self.authenticate():
                return
            if not self.select_datasource():
                return
            if self.user == "datauser":
                self._run_end_user()
            else:
                self._run_admin()
        except Exception as e:
            self.logger.error(f"CLI error: {str(e)}")
            print(f"An error occurred: {str(e)}")
        finally:
            if self.db_manager:
                self.db_manager.close_connection()
                self.logger.debug("Closed database connection")

    def _run_end_user(self) -> None:
        """Run the End User interface for querying data.

        Provides options to enter NLQs or exit.
        """
        while True:
            print("\nEnd User Menu:")
            print("1. Enter Natural Language Query")
            print("2. Exit")
            choice = input("Select option: ").strip()
            if choice == "1":
                nlq = input("Enter your query: ").strip()
                self._process_end_user_query(nlq)
            elif choice == "2":
                self.logger.info("End User session ended")
                break
            else:
                print("Invalid option")
                self.logger.warning(f"Invalid menu option: {choice}")

    def _process_end_user_query(self, nlq: str) -> None:
        """Process an End User natural language query.

        Retrieves suggestions, predicts tables, generates SQL, executes query, and handles user confirmation.

        Args:
            nlq (str): Natural language query.
        """
        try:
            tia_output = self.tia.predict_tables(nlq, self.user)
            if not tia_output:
                print("Unable to process your request, our Engineering team will contact you soon")
                self.logger.warning(f"TIA failed for NLQ: {nlq}")
                return

            suggestions = self._get_suggestions(nlq, tia_output)
            if suggestions:
                print("\nSuggested Queries:")
                for idx, sug in enumerate(suggestions[:3], 1):
                    print(f"{idx}. {sug['user_query']}")
                choice = input("\nSelect suggestion (number) or press Enter to continue: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                    nlq = suggestions[int(choice) - 1]["user_query"]
                    tia_output = self.tia.predict_tables(nlq, self.user)
                    if not tia_output:
                        print("Unable to process selected suggestion")
                        return

            sql_query = self.proga.generate_sql(tia_output, self.user)
            if not sql_query:
                print("Cannot resolve query")
                self.logger.warning(f"SQL generation failed for NLQ: {nlq}")
                return

            sample_data, csv_path = self.opden.execute_query(sql_query, self.user, nlq)
            if sample_data is not None and not sample_data.empty:
                print("\nSample Data (5 rows):")
                print(sample_data.to_string(index=False))
                confirm = input("\nIs this data correct? (y/n): ").strip().lower()
                if confirm == "y":
                    print(f"Data saved to: {csv_path}")
                    self.logger.info(f"User confirmed data for NLQ: {nlq}, saved to {csv_path}")
                else:
                    self.db_manager.store_rejected_query(
                        query=nlq,
                        reason="User rejected data",
                        user=self.user,
                        error_type="USER_REJECTED"
                    )
                    print("Query rejected")
                    self.logger.info(f"User rejected data for NLQ: {nlq}")
            else:
                print("No data returned")
                self.logger.warning(f"No data returned for NLQ: {nlq}")
        except (TIAError, PROGAError, OPDENError, DBError) as e:
            self.logger.error(f"Query processing error: {str(e)}")
            print(f"Error: {str(e)}")

    def _get_suggestions(self, nlq: str, tia_output: Dict) -> List[Dict]:
        """Retrieve query suggestions based on training data.

        Args:
            nlq (str): Current natural language query.
            tia_output (Dict): TIA output containing predicted tables.

        Returns:
            List[Dict]: List of suggested training data entries.
        """
        table_name = f"training_data_{self.datasource['name']}"
        query = f"SELECT user_query, related_tables FROM {table_name}"
        try:
            cursor = self.db_manager.connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            suggestions = []
            predicted_tables = set(tia_output["tables"])
            for row in rows:
                user_query, related_tables = row
                tables = set(related_tables.split(","))
                if tables & predicted_tables:
                    suggestions.append({"user_query": user_query, "related_tables": related_tables})
            self.logger.debug(f"Retrieved {len(suggestions)} suggestions for NLQ: {nlq}")
            return suggestions
        except pyodbc.Error as e:
            self.logger.error(f"Failed to fetch suggestions: {str(e)}")
            return []

    def _run_admin(self) -> None:
        """Run the Admin interface for training and model management.

        Provides options to initialize tables, train manually, train in bulk, generate models, view rejected queries, or exit.
        """
        while True:
            print("\nAdmin Menu:")
            print("1. Initialize Training Table")
            print("2. Manual Training")
            print("3. Bulk Training")
            print("4. Generate Model")
            print("5. View Rejected Queries")
            print("6. Exit")
            choice = input("Select option: ").strip()
            if choice == "1":
                self._initialize_training_table()
            elif choice == "2":
                self._manual_train()
            elif choice == "3":
                self._bulk_train()
            elif choice == "4":
                self._generate_model()
            elif choice == "5":
                self._view_rejected_queries()
            elif choice == "6":
                self.logger.info("Admin session ended")
                break
            else:
                print("Invalid option")
                self.logger.warning(f"Invalid menu option: {choice}")

    def _initialize_training_table(self) -> None:
        """Initialize tables for training data, rejected queries, and model metrics."""
        try:
            self.db_manager.create_training_table()
            self.db_manager.create_rejected_queries_table()
            self.db_manager.create_model_metrics_table()
            print("Tables initialized successfully")
            self.logger.info("Initialized database tables")
        except DBError as e:
            self.logger.error(f"Failed to initialize tables: {str(e)}")
            print(f"Error: {e}")

    def _manual_train(self) -> None:
        """Perform manual training by inputting NLQ, tables, columns, and SQL."""
        try:
            nlq = input("Enter natural language query: ").strip()
            tables = input("Enter related tables (comma-separated): ").strip().split(",")
            columns = input("Enter specific columns (comma-separated): ").strip().split(",")
            sql = input("Enter equivalent SQL query: ").strip()
            self.tia.train_manual(nlq, [t.strip() for t in tables], [c.strip() for c in columns], sql)
            print("Training data saved successfully")
            self.logger.info(f"Manual training completed for NLQ: {nlq}")
        except TIAError as e:
            self.logger.error(f"Manual training failed: {str(e)}")
            print(f"Error: {e}")

    def _bulk_train(self) -> None:
        """Perform bulk training from a CSV input.

        Expects CSV format: db_config,db_name,user_query,related_tables,specific_columns,relevant_sql
        """
        try:
            print("Enter CSV data (one row per line, press Enter twice to finish):")
            csv_lines = []
            while True:
                line = input().strip()
                if not line:
                    break
                csv_lines.append(line)
            if not csv_lines:
                print("No data provided")
                return

            training_data = []
            csv_reader = csv.DictReader(StringIO("\n".join(csv_lines)))
            for row in csv_reader:
                required = ["db_config", "db_name", "user_query", "related_tables", "specific_columns", "relevant_sql"]
                if not all(key in row for key in required):
                    self.logger.error("Invalid CSV format")
                    print("Invalid CSV format")
                    return
                training_data.append({
                    "db_config": row["db_config"],
                    "db_name": row["db_name"],
                    "user_query": row["user_query"],
                    "related_tables": row["related_tables"],
                    "specific_columns": row["specific_columns"],
                    "relevant_sql": row["relevant_sql"]
                })
            self.tia.train_bulk(training_data)
            print("Bulk training completed")
            self.logger.info(f"Completed bulk training with {len(training_data)} records")
        except TIAError as e:
            self.logger.error(f"Bulk training failed: {str(e)}")
            print(f"Error: {e}")

    def _generate_model(self) -> None:
        """Generate or update the TIA prediction model."""
        try:
            self.tia.generate_model()
            print("Model generated successfully")
            self.logger.info("Generated TIA model")
        except TIAError as e:
            self.logger.error(f"Model generation failed: {str(e)}")
            print(f"Error: {e}")

    def _view_rejected_queries(self) -> None:
        """Display rejected queries from the database."""
        table_name = f"rejected_queries_{self.datasource['name']}"
        query = f"SELECT query, reason, user, error_type, timestamp FROM {table_name}"
        try:
            cursor = self.db_manager.connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            if rows:
                print("\nRejected Queries:")
                for row in rows:
                    print(f"Query: {row[0]}\nReason: {row[1]}\nUser: {row[2]}\nError Type: {row[3]}\nTimestamp: {row[4]}\n")
                self.logger.info(f"Displayed {len(rows)} rejected queries")
            else:
                print("No rejected queries found")
                self.logger.info("No rejected queries found")
        except pyodbc.Error as e:
            self.logger.error(f"Failed to view rejected queries: {str(e)}")
            print(f"Error: {e}")