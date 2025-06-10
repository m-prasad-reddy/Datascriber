import argparse
import json
from pathlib import Path
from typing import Optional, List
import logging
import shlex
from tabulate import tabulate
from datetime import datetime
import re
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from core.orchestrator import Orchestrator, OrchestrationError
import traceback

class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass

class Interface:
    """Command-line interface for the Datascriber system.

    Provides a menu-driven CLI with query mode for data users and admins.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): CLI logger.
        orchestrator (Orchestrator): Orchestrator instance.
        username (Optional[str]): Current user.
        datasource (Optional[str]): Selected datasource.
        parser (argparse.ArgumentParser): Argument parser.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize Interface.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            CLIError: If initialization fails.
        """
        try:
            self.config_utils = config_utils
            self.logging_setup = LoggingSetup.get_instance(self.config_utils)
            self.logger = self.logging_setup.get_logger("cli", "system")
            self.logger.debug("Initializing Orchestrator")
            try:
                self.orchestrator = Orchestrator(self.config_utils)
            except Exception as e:
                self.logger.error(f"Orchestrator initialization failed: {str(e)}\n{traceback.format_exc()}")
                raise CLIError(f"Failed to initialize Orchestrator: {str(e)}")
            self.username = None
            self.datasource = None
            self.parser = self._create_parser()
            self.logger.debug("Initialized CLI Interface")
            self._load_llm_config()
        except (ConfigError, CLIError) as e:
            self.logger.error(f"Failed to initialize CLI: {str(e)}")
            raise CLIError(f"Failed to initialize CLI: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during CLI initialization: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Unexpected error during CLI initialization: {str(e)}")

    def _load_llm_config(self) -> None:
        """Load LLM configuration for validation rules."""
        try:
            with open('app-config/llm_config.json', 'r') as f:
                self.llm_config = json.load(f)
            self.logger.debug("Loaded LLM config for validation")
        except Exception as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}")
            raise CLIError(f"Failed to load LLM config: {str(e)}")

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser.

        Returns:
            argparse.ArgumentParser: Configured parser.
        """
        parser = argparse.ArgumentParser(description="Datascriber CLI", add_help=False)
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        login_parser = subparsers.add_parser("login", help="Log in as a user")
        login_parser.add_argument("username", help="Username (admin/datauser)")
        ds_parser = subparsers.add_parser("select-datasource", help="Select a datasource")
        ds_parser.add_argument("name", nargs="?", help="Datasource name")
        query_parser = subparsers.add_parser("query", help="Submit an NLQ")
        query_parser.add_argument("nlq", nargs="+", type=str, help="Natural language query")
        subparsers.add_parser("refresh-metadata", help="Refresh metadata (admin only)")
        subparsers.add_parser("train-model", help="Train prediction model (admin only)")
        mode_parser = subparsers.add_parser("set-synonym-mode", help="Set synonym mode (admin only)")
        mode_parser.add_argument("mode", choices=["static", "dynamic"], help="Synonym mode")
        subparsers.add_parser("map-failed-query", help="Map failed queries (admin only)")
        subparsers.add_parser("list-datasources", help="List available datasources")
        subparsers.add_parser("list-schemas", help="List available schemas")
        subparsers.add_parser("query-mode", help="Enter query mode (admin/datauser)")
        list_cols_parser = subparsers.add_parser("list-columns", help="List columns for a table")
        list_cols_parser.add_argument("table", help="Table name")
        notif_parser = subparsers.add_parser("manage-notifications", help="Manage notifications (view/resolve/delete/retry)")
        notif_parser.add_argument("action", choices=["view", "resolve", "delete", "retry"], help="Notification action")
        notif_parser.add_argument("id", nargs="?", type=int, help="Notification ID (required for resolve/delete/retry)")
        subparsers.add_parser("exit", help="Exit the CLI")
        return parser

    def run(self) -> None:
        """Run the CLI loop.

        Raises:
            CLIError: If execution fails.
        """
        try:
            print("Welcome to Datascriber CLI. Type 'exit' to quit.")
            while True:
                try:
                    command = input("> ").strip()
                    if not command:
                        continue
                    args = self.parse_args_from_input(command)
                    if args.command == "exit":
                        print("Exiting...")
                        self.logger.info("User exited CLI")
                        return
                    self.execute_command(args)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    self.logger.info("User interrupted CLI")
                    return
                except (argparse.ArgumentError, ValueError) as e:
                    print(f"Invalid command: {str(e)}. Use 'login', 'select-datasource', 'query', 'query-mode', 'list-columns', 'manage-notifications', etc.")
                    self.logger.warning(f"Invalid command: {command}")
                except (CLIError, OrchestrationError) as e:
                    print(f"Error: {str(e)}")
                    self.logger.error(f"Command error: {str(e)}")
        except Exception as e:
            self.logger.error(f"CLI execution failed: {str(e)}")
            raise CLIError(f"CLI execution failed: {str(e)}")

    def parse_args_from_input(self, command: str) -> argparse.Namespace:
        """Parse command input.

        Args:
            command (str): Raw command string.

        Returns:
            argparse.Namespace: Parsed arguments.

        Raises:
            argparse.ArgumentError: If parsing fails.
        """
        self.logger.debug(f"Parsing command: {command}")
        try:
            args = shlex.split(command)
            if not args:
                raise argparse.ArgumentError(None, "Empty command")
            if args[0].lower() == "exit":
                args[0] = "exit"
            if args[0] == "select-datasource" and len(args) == 1:
                from argparse import Namespace
                return argparse.Namespace(command="select-datasource", name=None)
            return self.parser.parse_args(args)
        except ValueError:
            raise argparse.ArgumentError(None, "Invalid command syntax")

    def _validate_date_format(self, nlq: str) -> bool:
        """Validate date formats in the NLQ.

        Args:
            nlq (str): Natural language query.

        Returns:
            bool: True if valid, False if invalid.

        Raises:
            CLIError: If date format is invalid.
        """
        date_formats = self.llm_config['prompt_settings']['validation']['rules']['date_format']
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
        ]
        for pattern in date_patterns:
            if re.search(pattern, nlq):
                try:
                    matched_date = re.search(pattern, nlq).group(0)
                    if pattern == r'\d{4}-\d{2}-\d{2}':
                        datetime.strptime(matched_date, '%Y-%m-%d')
                    elif pattern == r'\d{2}/\d{2}/\d{4}':
                        datetime.strptime(matched_date, '%m/%d/%Y')
                    elif pattern == r'\d{2}-\d{2}-\d{4}':
                        datetime.strptime(matched_date, '%d-%m-%Y')
                    self.logger.debug(f"Valid date format found in NLQ: {matched_date}")
                    return True
                except ValueError:
                    self.logger.error(f"Invalid date format in NLQ: {matched_date}")
                    raise CLIError(self.llm_config['prompt_settings']['validation']['rules']['error_message'])
        return True  # No date found, assume valid

    def _extract_entities(self, nlq: str) -> dict:
        """Extract entities (dates, names, objects, places) from NLQ.

        Args:
            nlq (str): Natural language query.

        Returns:
            dict: Extracted entities.
        """
        entities = {"dates": [], "names": [], "objects": [], "places": []}
        # Date extraction
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, nlq)
            entities["dates"].extend(matches)
        # Placeholder for names, objects, places (to be enhanced with LLM if needed)
        # Example: Simple keyword-based extraction for places
        place_keywords = ["New York", "California", "London"]
        for place in place_keywords:
            if place.lower() in nlq.lower():
                entities["places"].append(place)
        self.logger.debug(f"Extracted entities: {entities}")
        return entities

    def execute_command(self, args: argparse.Namespace) -> None:
        """Execute a CLI command.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Raises:
            CLIError: If execution fails.
        """
        self.logger.debug(f"Executing command: {args.command}")
        try:
            if args.command == "login":
                self.login(args.username)
            elif args.command == "select-datasource":
                if args.name is None:
                    self.list_datasource()
                    print("Usage: select-datasource <name>")
                else:
                    self.select_datasource(args.name)
            elif args.command == "query":
                nlq = " ".join(args.nlq)
                if self._validate_date_format(nlq):
                    entities = self._extract_entities(nlq)
                    self.submit_query(nlq, entities)
            elif args.command == "refresh-metadata":
                self.refresh_metadata()
            elif args.command == "train-model":
                self.train_model()
            elif args.command == "set-synonym-mode":
                self.set_synonym_mode(args.mode)
            elif args.command == "map-failed-query":
                self.map_failed_query()
            elif args.command == "list-datasources":
                self.list_datasource()
            elif args.command == "list-schemas":
                self.list_schemas()
            elif args.command == "query-mode":
                self.enter_query_mode()
            elif args.command == "list-columns":
                self.list_columns(args.table)
            elif args.command == "manage-notifications":
                self.manage_notifications(args.action, args.id)
            else:
                print("Unknown command")
                self.logger.warning(f"Unknown command: {args.command}")
        except Exception as e:
            self.logger.error(f"Failed to execute command {args.command}: {str(e)}")
            raise CLIError(f"Failed to execute command: {str(e)}")

    def login(self, username: str) -> None:
        """Log in a user.

        Args:
            username (str): Username.

        Raises:
            CLIError: If login fails.
        """
        if self.orchestrator.login(username):
            self.username = username
            print(f"Logged in as {username}")
            self.logger.info(f"User {username} logged in")
        else:
            self.logger.error(f"Login failed for username: {username}")
            raise CLIError("Invalid username")

    def select_datasource(self, name: str) -> None:
        """Select a datasource.

        Args:
            name (str): Datasource name.

        Raises:
            CLIError: If selection fails.
        """
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("Please log in first")
        if self.orchestrator.select_datasource(name):
            self.datasource = name
            if self.orchestrator.validate_metadata():
                print(f"Selected datasource: {name}")
                self.logger.info(f"Selected datasource: {name}")
                if self.username == "datauser":
                    self.enter_query_mode()
                else:
                    print("Type 'query-mode' to enter query mode or 'query <nlq>' to submit a query.")
            else:
                self.logger.error("Metadata validation failed")
                self.datasource = None
                if self.username == "datauser":
                    self.username = None
                    raise CLIError("Logged out due to invalid metadata")
                raise CLIError("Invalid or unavailable metadata")
        else:
            self.logger.error(f"Datasource selection failed: {name}")
            raise CLIError("Invalid datasource")

    def enter_query_mode(self) -> None:
        """Enter query mode for data users or admins.

        Allows continuous NLQ input until 'exit' is entered.
        """
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("Please log in first")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("Please select a datasource first")
        print("Entered query mode. Type your query or 'exit' to return to main menu.")
        while True:
            try:
                query = input("Query> ").strip()
                if query.lower() == "exit":
                    print("Exiting query mode.")
                    self.logger.info("Exited query mode")
                    return
                if not query:
                    continue
                if self._validate_date_format(query):
                    entities = self._extract_entities(query)
                    self.submit_query(query, entities)
            except KeyboardInterrupt:
                print("\nExiting query mode.")
                self.logger.info("User interrupted query mode")
                return
            except (CLIError, OrchestrationError) as e:
                print(f"Error: {str(e)}")
                self.logger.error(f"Query error: {str(e)}")

    def submit_query(self, nlq: str, entities: dict = None) -> None:
        """Submit an NLQ across all schemas.

        Args:
            nlq (str): Natural language query.
            entities (dict, optional): Extracted entities (dates, names, objects, places).

        Raises:
            CLIError: If query fails.
        """
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("Please log in first")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("Please select a datasource first")
        try:
            config = self.config_utils.load_db_configurations()
            ds_config = next(ds for ds in config["datasources"] if ds["name"] == self.datasource)
            schemas = ds_config["connection"].get("schemas", [])
            if not schemas:
                self.logger.error("No schemas configured for datasource")
                raise CLIError("No schemas configured in db_configurations.json")
            results = []
            for schema in schemas:
                result = self.orchestrator.process_nlq(nlq, schema, entities=entities)
                if result:
                    results.append(result)
            if results:
                for result in results:
                    schema_name = result["tables"][0].split(".")[0] if result["tables"] else schema
                    print(f"Results for schema {schema_name}:")
                    print(f"Tables: {', '.join(result['tables']) if result['tables'] else 'None'}")
                    print(f"Columns: {', '.join(result['columns']) if result['columns'] else 'None'}")
                    print("Sample Data:")
                    if result["sample_data"]:
                        self.logger.debug(f"Formatting {len(result['sample_data'])} rows as table")
                        table = tabulate(
                            result["sample_data"],
                            headers="keys",
                            tablefmt="grid",
                            stralign="left",
                            numalign="right"
                        )
                        print(table)
                    else:
                        print("No sample data available")
                    print(f"Full results saved to: {result['csv_path']}")
                    self.logger.info(f"Query results saved to: {result['csv_path']}")
            else:
                message = "Query cannot be processed. Notified admin."
                self.logger.warning(f"No results for query: {nlq}")
                self.orchestrator.notify_admin(nlq, schemas[0], "No results returned", entities=entities)
                print(message)
        except Exception as e:
            self.logger.error(f"Failed to process NLQ: {str(e)}")
            self.orchestrator.notify_admin(nlq, schemas[0] if schemas else "unknown", str(e), entities=entities)
            raise CLIError(f"Failed to process query: {str(e)}")

    def refresh_metadata(self) -> None:
        """Refresh metadata (admin only).

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("Please select a datasource first")
        try:
            config = self.config_utils.load_db_configurations()
            ds_config = next(ds for ds in config["datasources"] if ds["name"] == self.datasource)
            schemas = ds_config["connection"].get("schemas", [])
            if not schemas:
                raise CLIError("No schemas configured in db_configurations.json")
            for schema in schemas:
                if self.orchestrator.refresh_metadata(schema):
                    print(f"Metadata refreshed for schema: {schema}")
                    self.logger.info(f"Refreshed metadata for schema: {schema}")
                else:
                    self.logger.error(f"Metadata refresh failed for schema: {schema}")
                    raise CLIError(f"Metadata refresh failed for schema: {schema}")
        except Exception as e:
            self.logger.error(f"Failed to refresh metadata: {str(e)}")
            raise CLIError(f"Failed to refresh metadata: {str(e)}")

    def train_model(self) -> None:
        """Train prediction model (admin only).

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("Please select a datasource first")
        try:
            if self.orchestrator.train_model():
                print("Model training completed")
                self.logger.info("Trained prediction model")
            else:
                raise CLIError("Model training failed")
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            raise CLIError(f"Failed to train model: {str(e)}")

    def set_synonym_mode(self, mode: str) -> None:
        """Set synonym mode (admin only).

        Args:
            mode (str): 'static' or 'dynamic'.

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        try:
            config_path = self.config_utils.config_dir / "synonym_config.json"
            config = {"synonym_mode": mode}
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Set synonym mode to: {mode}")
            self.logger.info(f"Set synonym mode to: {mode}")
        except Exception as e:
            self.logger.error(f"Failed to set synonym mode: {str(e)}")
            raise CLIError(f"Failed to set synonym mode: {str(e)}")

    def map_failed_query(self) -> None:
        """Map a failed query to tables, columns, and SQL (admin only).

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("Please select a datasource first")
        try:
            print("Enter failed NLQ:")
            nlq = input("> ").strip()
            print("Enter tables (comma-separated):")
            tables = input("> ").strip().split(",")
            print("Enter columns (comma-separated):")
            columns = input("> ").strip().split(",")
            print("Enter SQL query:")
            sql = input("> ").strip()
            if self.orchestrator.map_failed_query(nlq, tables, columns, sql):
                print("Failed query mapped successfully")
                self.logger.info(f"Mapped failed query: {nlq}")
            else:
                raise CLIError("Failed to map query")
        except Exception as e:
            self.logger.error(f"Failed to map failed query: {str(e)}")
            raise CLIError(f"Failed to map failed query: {str(e)}")

    def list_datasource(self) -> None:
        """List available datasources.

        Raises:
            CLIError: If listing fails.
        """
        try:
            config = self.config_utils.load_db_configurations()
            datasources = [ds.get("name") for ds in config.get("datasources", [])]
            if not datasources:
                print("No datasources available")
            else:
                print("Available datasources:")
                for ds in datasources:
                    print(f" - {ds}")
            self.logger.info("Listed available datasources")
        except ConfigError as e:
            self.logger.error(f"Failed to list datasources: {str(e)}")
            raise CLIError(f"Failed to list datasources: {str(e)}")

    def list_schemas(self) -> None:
        """List available schemas for the selected datasource.

        Raises:
            CLIError: If listing fails.
        """
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        try:
            config = self.config_utils.load_db_configurations()
            ds_config = next((ds for ds in config["datasources"] if ds["name"] == self.datasource), None)
            if not ds_config:
                raise CLIError("Datasource configuration not found")
            schemas = ds_config["connection"].get("schemas", [])
            if not schemas:
                print("No schemas configured")
            else:
                print("Available schemas:")
                for schema in schemas:
                    print(f" - {schema}")
            self.logger.info(f"Listed schemas for datasource: {self.datasource}")
        except ConfigError as e:
            self.logger.error(f"Failed to list schemas: {str(e)}")
            raise CLIError(f"Failed to list schemas: {str(e)}")

    def list_columns(self, table: str) -> None:
        """List columns for a specified table.

        Args:
            table (str): Table name.

        Raises:
            CLIError: If listing fails.
        """
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("Please log in first")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("Please select a datasource first")
        try:
            config = self.config_utils.load_db_configurations()
            ds_config = next((ds for ds in config["datasources"] if ds["name"] == self.datasource), None)
            if not ds_config:
                raise CLIError("Datasource configuration not found")
            schemas = ds_config["connection"].get("schemas", [])
            if not schemas:
                raise CLIError("No schemas configured")
            columns = []
            for schema in schemas:
                cols = self.orchestrator.get_table_columns(schema, table)
                if cols:
                    columns.extend(cols)
            if columns:
                print(f"Columns for table {table}:")
                for col in columns:
                    print(f" - {col}")
                self.logger.info(f"Listed columns for table: {table}")
            else:
                print(f"No columns found for table {table}")
                self.logger.warning(f"No columns found for table: {table}")
        except Exception as e:
            self.logger.error(f"Failed to list columns for table {table}: {str(e)}")
            raise CLIError(f"Failed to list columns: {str(e)}")

    def manage_notifications(self, action: str, notification_id: Optional[int] = None) -> None:
        """Manage notifications (view/resolve/delete/retry).

        Args:
            action (str): Action to perform (view/resolve/delete/retry).
            notification_id (Optional[int]): Notification ID for resolve/delete/retry.

        Raises:
            CLIError: If action fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        try:
            log_file = Path("logs/notifications.log")
            notifications = []
            if log_file.exists():
                with open(log_file, "r") as f:
                    notifications = [json.loads(line.strip()) for line in f if line.strip()]
            notifications = notifications[-100:]  # Keep last 100

            if action == "view":
                if not notifications:
                    print("No notifications available")
                else:
                    print("Notifications (last 100):")
                    table = [[n["id"], n["timestamp"], n["query"], n["error"]] for n in notifications]
                    print(tabulate(table, headers=["ID", "Timestamp", "Query", "Error"], tablefmt="grid"))
                self.logger.info("Viewed notifications")
            elif action in ["resolve", "delete", "retry"]:
                if notification_id is None:
                    raise CLIError("Notification ID required for resolve/delete/retry")
                target = next((n for n in notifications if n["id"] == notification_id), None)
                if not target:
                    raise CLIError(f"Notification ID {notification_id} not found")
                if action == "resolve":
                    print(f"Notification {notification_id} marked as resolved")
                    self.logger.info(f"Resolved notification ID {notification_id}")
                elif action == "delete":
                    notifications = [n for n in notifications if n["id"] != notification_id]
                    with open(log_file, "w") as f:
                        for n in notifications:
                            f.write(json.dumps(n) + "\n")
                    print(f"Notification {notification_id} deleted")
                    self.logger.info(f"Deleted notification ID {notification_id}")
                elif action == "retry":
                    if self._validate_date_format(target["query"]):
                        entities = self._extract_entities(target["query"])
                        self.submit_query(target["query"], entities)
                        print(f"Retried notification {notification_id}")
                        self.logger.info(f"Retried notification ID {notification_id}")
        except Exception as e:
            self.logger.error(f"Failed to manage notifications: {str(e)}")
            raise CLIError(f"Failed to manage notifications: {str(e)}")

def main():
    """Main entry point for the CLI."""
    try:
        config_utils = ConfigUtils()
        cli = Interface(config_utils)
        cli.run()
    except CLIError as e:
        print(f"Error: {str(e)}")
        logging.error(f"CLI terminated: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
        logging.info("CLI terminated by user")

if __name__ == "__main__":
    main()
