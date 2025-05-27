import requests
from typing import Dict, Optional, List
import logging
import json
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError

class PROGAError(Exception):
    """Custom exception for Prompt Generation Agent errors."""
    pass

class PromptGenerator:
    """Prompt Generation Agent for generating SQL queries using LLMs.

    Generates LLM prompts from TIA outputs, incorporating table/column descriptions and references
    from metadata. Interacts with LLMs via API or mock endpoints to produce SQL queries. Failed
    queries are stored in the rejected_queries table.

    Attributes:
        config_utils (ConfigUtils): Instance for configuration access.
        logger (logging.Logger): Logger instance for PROGA operations.
        db_manager (DBManager): Instance for database operations.
        storage_manager (StorageManager): Instance for S3 storage operations.
        datasource (Dict): Current datasource configuration.
        llm_config (Dict): LLM configuration (api_key, endpoint, mock_enabled).
        prompt_template (str): Template for LLM prompts.
    """

    def __init__(self, config_utils: ConfigUtils, logging_setup: LoggingSetup, db_manager: DBManager, storage_manager: StorageManager, datasource: Dict):
        """Initialize PromptGenerator with configuration, logging, and storage.

        Args:
            config_utils (ConfigUtils): Instance for configuration access.
            logging_setup (LoggingSetup): Instance for logging setup.
            db_manager (DBManager): Instance for database operations.
            storage_manager (StorageManager): Instance for S3 operations.
            datasource (Dict): Datasource configuration from db_configurations.json.

        Raises:
            PROGAError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logging_setup.get_logger("proga")
        self.db_manager = db_manager
        self.storage_manager = storage_manager
        self.datasource = datasource
        try:
            self.llm_config = self.config_utils.load_llm_config()
            self.mock_enabled = self.llm_config.get("mock_enabled", False)
            self.mock_endpoint = self.llm_config.get("mock_endpoint")
        except ConfigError as e:
            self.logger.error(f"Failed to load LLM config: {str(e)}")
            raise PROGAError(f"Failed to load LLM config: {str(e)}")
        self.prompt_template = """
        Given the following natural language query and database context, generate a valid SQL query for {db_type}.
        
        **Database Context:**
        {schema_info}
        
        **Query:**
        {nlq}
        
        **Conditions:**
        {conditions}
        
        **Aggregations (if applicable):**
        {aggregations}
        
        **Example Query:**
        - Example NLQ: {example_nlq}
        - Example Schema: {example_schema}
        - Example SQL: {example_sql}
        
        **Instructions:**
        - Generate a SELECT query only; do not include INSERT, UPDATE, or DELETE.
        - Use table aliases for clarity if multiple tables are involved.
        - Include appropriate JOINs for referenced tables.
        - Ensure the query is compatible with {db_type} syntax.
        - Return only the SQL query, without explanations.
        """
        self.logger.debug(f"Initialized PROGA for datasource: {datasource['name']}")

    def generate_sql(self, tia_output: Dict, user: str) -> Optional[str]:
        """Generate SQL query from TIA output using the LLM.

        Args:
            tia_output (Dict): TIA output with tables, columns, DDL, NLQ, conditions, and optional SQL.
            user (str): User submitting the query (datauser or admin).

        Returns:
            Optional[str]: Generated SQL query, or None if LLM fails.

        Raises:
            PROGAError: If prompt generation or API call fails critically.
        """
        if not tia_output:
            self.logger.error("Empty TIA output provided")
            raise PROGAError("Empty TIA output provided")

        required_keys = ["tables", "columns", "ddl", "nlq", "conditions"]
        for key in required_keys:
            if key not in tia_output:
                self.logger.error(f"Missing required key '{key}' in TIA output")
                raise PROGAError(f"Missing required key '{key}' in TIA output")

        # Check if stored SQL is available
        if tia_output.get("sql"):
            self.logger.debug(f"Using stored SQL for NLQ: {tia_output['nlq']}")
            return tia_output["sql"]

        # Generate prompt
        try:
            prompt = self._generate_prompt(tia_output)
            self.logger.debug(f"Generated prompt for NLQ: {tia_output['nlq']}")
        except Exception as e:
            self.logger.error(f"Failed to generate prompt: {str(e)}")
            raise PROGAError(f"Failed to generate prompt: {str(e)}")

        # Call LLM
        sql_query = self._call_llm(prompt)
        if sql_query:
            self.logger.info(f"Generated SQL for NLQ: {tia_output['nlq']}")
            return sql_query

        # Handle failure
        error_message = "At this moment I cannot resolve your query"
        self.logger.error(f"LLM failed for NLQ: {tia_output['nlq']}")
        try:
            self.db_manager.store_rejected_query(
                query=tia_output["nlq"],
                reason=error_message,
                user=user,
                error_type="NO_LLM_RESPONSE"
            )
        except DBError as e:
            self.logger.error(f"Failed to store rejected query: {str(e)}")
        return None

    def _generate_prompt(self, tia_output: Dict) -> str:
        """Generate LLM prompt from TIA output.

        Includes table/column descriptions and referenced tables from metadata.

        Args:
            tia_output (Dict): TIA output with tables, columns, DDL, NLQ, conditions.

        Returns:
            str: Formatted prompt string.
        """
        try:
            metadata = self.config_utils.load_metadata(self.datasource, schema="default")
            tables = metadata.get("tables", [])
            nlq = tia_output["nlq"]
            predicted_tables = tia_output["tables"]
            predicted_columns = tia_output["columns"]

            # Gather table and column info with descriptions
            table_info = []
            for table in tables:
                if table["name"] in predicted_tables:
                    table_desc = table.get("description", "No description available")
                    columns_info = []
                    for col in table.get("columns", []):
                        if col["name"] in predicted_columns:
                            col_desc = col.get("description", "No description available")
                            ref_info = ""
                            if col.get("references"):
                                ref = col["references"]
                                ref_info = f" (references {ref['table']}.{ref['column']})"
                            columns_info.append(f"{col['name']}: {col_desc}{ref_info}")
                    table_info.append(f"Table: {table['name']} ({table_desc})\nColumns: {', '.join(columns_info)}")

            # Include referenced tables
            referenced_tables = set()
            for table in tables:
                if table["name"] in predicted_tables:
                    for col in table.get("columns", []):
                        if col.get("references"):
                            referenced_tables.add(col["references"]["table"])
            for ref_table in referenced_tables:
                if ref_table not in predicted_tables:
                    for table in tables:
                        if table["name"] == ref_table:
                            table_desc = table.get("description", "No description available")
                            columns_info = [f"{col['name']}: {col.get('description', 'No description available')}"
                                           for col in table.get("columns", [])]
                            table_info.append(f"Table: {table['name']} ({table_desc})\nColumns: {', '.join(columns_info)}")

            schema_info = "\n\n".join(table_info) if table_info else "No schema information available."
            db_type = "SQL Server" if self.datasource["type"] == "SQL Server" else "S3 (CSV/Parquet/ORC/TXT)"
            conditions_str = json.dumps(tia_output["conditions"]) if tia_output["conditions"] else "None"
            aggregations = self._detect_aggregations(nlq)
            example_nlq = "List all products by brand"
            example_schema = "Table: products (product_name VARCHAR(255), brand_id INT)\nTable: brands (brand_id INT, brand_name VARCHAR(255))"
            example_sql = "SELECT p.product_name, b.brand_name FROM products p JOIN brands b ON p.brand_id = b.brand_id"
            return self.prompt_template.format(
                db_type=db_type,
                schema_info=schema_info,
                nlq=nlq,
                conditions=conditions_str,
                aggregations=", ".join(aggregations) if aggregations else "None",
                example_nlq=example_nlq,
                example_schema=example_schema,
                example_sql=example_sql
            )
        except ConfigError as e:
            self.logger.error(f"Failed to generate prompt: {str(e)}")
            raise PROGAError(f"Failed to generate prompt: {str(e)}")

    def _detect_aggregations(self, nlq: str) -> List[str]:
        """Detect aggregation types in NLQ (e.g., COUNT, SUM).

        Args:
            nlq (str): Natural language query.

        Returns:
            List[str]: List of detected aggregations.
        """
        aggregations = []
        nlq_lower = nlq.lower()
        if "count" in nlq_lower:
            aggregations.append("COUNT")
        if "sum" in nlq_lower:
            aggregations.append("SUM")
        if "average" in nlq_lower or "avg" in nlq_lower:
            aggregations.append("AVG")
        return aggregations

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM API to generate SQL query.

        Args:
            prompt (str): Formatted prompt for the LLM.

        Returns:
            Optional[str]: Generated SQL query, or None if API fails.
        """
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7
            }
            endpoint = self.mock_endpoint if self.mock_enabled else self.llm_config["endpoint"]
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            sql_query = response.json().get("choices", [{}])[0].get("text", "").strip()
            if sql_query:
                self.logger.debug(f"LLM response: {sql_query}")
                return sql_query
            self.logger.error("Empty LLM response")
            return None
        except requests.HTTPError as e:
            self.logger.error(f"LLM API HTTP error: {str(e)}")
            return None
        except (requests.RequestException, ValueError) as e:
            self.logger.error(f"LLM API call failed: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage for testing
    config_utils = ConfigUtils()
    logging_setup = LoggingSetup(config_utils)
    db_config = config_utils.load_db_config()
    datasource = next(db for db in db_config["databases"] if db["type"] == "SQL Server")
    db_manager = DBManager(config_utils, logging_setup.get_logger("db"), datasource)
    storage_manager = StorageManager(config_utils, logging_setup.get_logger("storage"), datasource)

    try:
        proga = PromptGenerator(config_utils, logging_setup, db_manager, storage_manager, datasource)
        
        # Test SQL generation
        tia_output = {
            "tables": ["products", "brands"],
            "columns": ["product_name", "brand_name", "brand_id"],
            "ddl": "CREATE TABLE products (product_name VARCHAR(255), brand_id INT); CREATE TABLE brands (brand_id INT, brand_name VARCHAR(255));",
            "nlq": "List all products by brand",
            "conditions": {"brand_id": "> 0"},
            "sql": None
        }
        sql = proga.generate_sql(tia_output, "datauser")
        print("Generated SQL:", sql)
    except (ConfigError, DBError, StorageError, PROGAError) as e:
        print(f"Error: {str(e)}")
    finally:
        db_manager.close_connection()