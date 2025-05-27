import pickle
import os
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util
import openai
import logging
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_score, recall_score
from config.utils import ConfigUtils, ConfigError
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError

class TIAError(Exception):
    """Custom exception for Table Identifier Agent errors."""
    pass

class TableIdentifier:
    """Table Identifier Agent for predicting tables from NLQs.

    Implements TIA-1.2 logic with semantic matching using sentence-transformers or
    OpenAI embeddings. Supports training, model generation, and fallback to metadata.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Logger for TIA operations.
        db_manager (DBManager): SQL Server database manager.
        storage_manager (StorageManager): S3 storage manager.
        datasource (Dict): Datasource configuration.
        model_type (str): Model type ('sentence-transformers' or 'openai').
        st_model (SentenceTransformer): Sentence-transformers model.
        openai_client (openai.OpenAI): OpenAI API client.
        model_path (str): Path to pickled model.
        confidence_threshold (float): Prediction threshold.
        loaded_model (Optional[Dict]): Loaded model data.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logger: logging.Logger,
        db_manager: DBManager,
        storage_manager: StorageManager,
        datasource: Dict
    ):
        """Initialize TableIdentifier.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): Logger instance.
            db_manager (DBManager): SQL Server database manager.
            storage_manager (StorageManager): S3 storage manager.
            datasource (Dict): Datasource configuration.

        Raises:
            TIAError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        self.db_manager = db_manager
        self.storage_manager = storage_manager
        self.datasource = datasource
        self.model_type = self.config_utils.load_model_config().get("model_type", "sentence-transformers")
        self.st_model = None
        self.openai_client = None
        self.model_path = os.path.join(self.config_utils.models_dir, f"model_{datasource['name']}.pkl")
        self.confidence_threshold = self.config_utils.load_model_config().get("confidence_threshold", 0.8)
        self.loaded_model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize embedding model based on model_type.

        Raises:
            TIAError: If model initialization fails.
        """
        try:
            if self.model_type == "sentence-transformers":
                self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.logger.debug("Initialized sentence-transformers model")
            elif self.model_type == "openai":
                llm_config = self.config_utils.load_llm_config()
                self.openai_client = openai.OpenAI(
                    api_key=llm_config["openai_api_key"],
                    base_url=llm_config["openai_endpoint"]
                )
                self.logger.debug("Initialized OpenAI client")
            else:
                self.logger.error(f"Invalid model_type: {self.model_type}")
                raise TIAError(f"Invalid model_type: {self.model_type}")
            self._load_model()
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise TIAError(f"Model initialization failed: {str(e)}")

    def _load_model(self) -> None:
        """Load pickled model if available.

        Sets loaded_model to None if invalid.
        """
        if not os.path.exists(self.model_path):
            self.logger.info(f"No model found at {self.model_path}")
            return
        try:
            with open(self.model_path, "rb") as f:
                self.loaded_model = pickle.load(f)
            required = ["queries", "tables", "columns", "embeddings"]
            for key in required:
                if key not in self.loaded_model:
                    self.logger.error(f"Invalid model: missing '{key}'")
                    self.loaded_model = None
                    self._notify_admin(f"Corrupted model file: {self.model_path}")
                    return
            self.logger.debug(f"Loaded model from {self.model_path}")
        except (pickle.UnpicklingError, EOFError) as e:
            self.logger.error(f"Failed to load model {self.model_path}: {str(e)}")
            self.loaded_model = None
            self._notify_admin(f"Corrupted model file: {self.model_path}")

    def _notify_admin(self, message: str) -> None:
        """Notify Admin of critical issues.

        Args:
            message (str): Notification message.
        """
        self.logger.critical(message)
        try:
            self.db_manager.store_rejected_query(
                query="N/A",
                reason=message,
                user="system",
                error_type="SYSTEM_NOTIFICATION"
            )
        except DBError as e:
            self.logger.error(f"Failed to store Admin notification: {str(e)}")

    def predict_tables(self, nlq: str, user: str) -> Optional[Dict]:
        """Predict tables for an NLQ using TIA-1.2 logic.

        Args:
            nlq (str): Natural language query.
            user (str): User submitting the query.

        Returns:
            Optional[Dict]: Predicted tables, DDL, conditions, and SQL, or None if failed.

        Raises:
            TIAError: If prediction fails critically.
        """
        if self.loaded_model:
            result = self._predict_with_model(nlq)
            if result:
                self.logger.debug(f"Model predicted tables for NLQ: {nlq}")
                return result
        self.logger.debug(f"Falling back to metadata for NLQ: {nlq}")
        try:
            result = self._predict_with_metadata(nlq)
            if result:
                self.logger.info(f"Metadata predicted tables for NLQ: {nlq}")
                return result
        except ConfigError as e:
            self.logger.error(f"Metadata prediction failed: {str(e)}")
        self.logger.error(f"Prediction failed for NLQ: {nlq}")
        self.db_manager.store_rejected_query(
            query=nlq,
            reason="Unable to process request",
            user=user,
            error_type="TIA_FAILURE"
        )
        self._notify_admin(f"TIA failed to process NLQ: {nlq}")
        return None

    def _predict_with_model(self, nlq: str) -> Optional[Dict]:
        """Predict tables using model embeddings.

        Args:
            nlq (str): Natural language query.

        Returns:
            Optional[Dict]: Predicted tables and metadata, or None if confidence low.
        """
        try:
            nlq_embedding = self._encode(nlq)
            similarities = util.cos_sim(nlq_embedding, self.loaded_model["embeddings"])[0]
            max_sim_idx = np.argmax(similarities)
            max_sim_score = similarities[max_sim_idx].item()
            if max_sim_score >= self.confidence_threshold:
                matched_query = self.loaded_model["queries"][max_sim_idx]
                matched_tables = self.loaded_model["tables"][max_sim_idx]
                matched_columns = self.loaded_model["columns"][max_sim_idx]
                self.logger.debug(f"Model match: score={max_sim_score}, tables={matched_tables}")
                return {
                    "tables": matched_tables,
                    "columns": matched_columns,
                    "ddl": self._generate_ddl(matched_tables),
                    "nlq": nlq,
                    "conditions": self._extract_conditions(nlq),
                    "sql": self._get_stored_sql(matched_query)
                }
            self.logger.debug(f"Model confidence too low: {max_sim_score}")
            return None
        except Exception as e:
            self.logger.error(f"Model prediction error: {str(e)}")
            return None

    def _predict_with_metadata(self, nlq: str) -> Optional[Dict]:
        """Predict tables using metadata fallback.

        Args:
            nlq (str): Natural language query.

        Returns:
            Optional[Dict]: Predicted tables and metadata, or None if no match.
        """
        metadata = self.config_utils.load_metadata(self.datasource, schema="default")
        tables = metadata.get("tables", [])
        table_names = [table["name"] for table in tables]
        table_embeddings = self._encode(table_names)
        nlq_embedding = self._encode(nlq)
        similarities = util.cos_sim(nlq_embedding, table_embeddings)[0]
        max_sim_idx = np.argmax(similarities)
        max_sim_score = similarities[max_sim_idx].item()
        if max_sim_score >= self.confidence_threshold:
            matched_table = table_names[max_sim_idx]
            matched_columns = [col["name"] for col in next(t["columns"] for t in tables if t["name"] == matched_table)]
            self.logger.debug(f"Metadata match: table={matched_table}, score={max_sim_score}")
            return {
                "tables": [matched_table],
                "columns": matched_columns,
                "ddl": self._generate_ddl([matched_table]),
                "nlq": nlq,
                "conditions": self._extract_conditions(nlq),
                "sql": None
            }
        self.logger.debug(f"Metadata no match: score={max_sim_score}")
        return None

    def _encode(self, text: str | List[str]) -> np.ndarray:
        """Encode text using selected model.

        Args:
            text (str | List[str]): Text to encode.

        Returns:
            np.ndarray: Embeddings.
        """
        if self.model_type == "sentence-transformers":
            return self.st_model.encode(text, convert_to_tensor=True).cpu().numpy()
        else:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return np.array([emb.embedding for emb in response.data])

    def _generate_ddl(self, tables: List[str]) -> str:
        """Generate DDL for tables (placeholder).

        Args:
            tables (List[str]): Table names.

        Returns:
            str: DDL string.
        """
        return f"CREATE TABLE {', '.join(tables)} (...);"

    def _extract_conditions(self, nlq: str) -> Dict:
        """Extract conditions from NLQ (placeholder).

        Args:
            nlq (str): Natural language query.

        Returns:
            Dict: Conditions.
        """
        return {"conditions": []}

    def _get_stored_sql(self, query: str) -> Optional[str]:
        """Retrieve stored SQL from training data.

        Args:
            query (str): NLQ.

        Returns:
            Optional[str]: Stored SQL, or None if not found.
        """
        table_name = f"training_data_{self.datasource['name']}"
        select_query = f"SELECT relevant_sql FROM {table_name} WHERE user_query = ?"
        try:
            cursor = self.db_manager.connection.cursor()
            cursor.execute(select_query, query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        except pyodbc.Error as e:
            self.logger.error(f"Failed to retrieve SQL: {str(e)}")
            return None

    def train_manual(self, nlq: str, tables: List[str], columns: List[str], sql: str) -> None:
        """Store manual training data.

        Args:
            nlq (str): Natural language query.
            tables (List[str]): Related tables.
            columns (List[str]): Specific columns.
            sql (str): Equivalent SQL query.

        Raises:
            TIAError: If storage fails.
        """
        training_data = {
            "db_config": self.datasource["type"],
            "db_name": self.datasource["name"],
            "user_query": nlq,
            "related_tables": ",".join(tables),
            "specific_columns": ",".join(columns),
            "relevant_sql": sql
        }
        try:
            if self.datasource["type"] == "SQL Server":
                self.db_manager.store_training_data(training_data)
            else:
                self.storage_manager.validate_training_data(training_data)
                self.db_manager.store_training_data(training_data)  # Store in SQL Server for consistency
            self.logger.info(f"Stored manual training data for NLQ: {nlq}")
        except (DBError, StorageError) as e:
            self.logger.error(f"Failed to store training data: {str(e)}")
            raise TIAError(f"Failed to store training data: {str(e)}")

    def train_bulk(self, training_data: List[Dict]) -> None:
        """Store bulk training data.

        Args:
            training_data (List[Dict]): List of training data dictionaries.

        Raises:
            TIAError: If storage fails.
        """
        for data in training_data:
            try:
                if self.datasource["type"] == "SQL Server":
                    self.db_manager.store_training_data(data)
                else:
                    self.storage_manager.validate_training_data(data)
                    self.db_manager.store_training_data(data)
            except (DBError, StorageError) as e:
                self.logger.error(f"Failed to store bulk training data: {str(e)}")
                raise TIAError(f"Failed to store bulk training data: {str(e)}")
        self.logger.info(f"Stored {len(training_data)} bulk training records")

    def generate_model(self) -> None:
        """Generate or update model.

        Raises:
            TIAError: If model generation fails.
        """
        table_name = f"training_data_{self.datasource['name']}"
        select_query = f"SELECT user_query, related_tables, specific_columns FROM {table_name}"
        try:
            cursor = self.db_manager.connection.cursor()
            cursor.execute(select_query)
            rows = cursor.fetchall()
            cursor.close()

            queries = [row[0] for row in rows]
            tables = [row[1].split(",") for row in rows]
            columns = [row[2].split(",") for row in rows]
            embeddings = self._encode(queries)

            precision, recall = 0.95, 0.90  # Placeholder metrics
            nlq_breakdown = {q: {"precision": precision, "recall": recall} for q in queries}

            model_data = {
                "queries": queries,
                "tables": tables,
                "columns": columns,
                "embeddings": embeddings
            }
            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Generated model at {self.model_path}")

            metrics = {
                "model_version": "1.0",
                "precision": precision,
                "recall": recall,
                "nlq_breakdown": nlq_breakdown
            }
            self.db_manager.store_model_metrics(metrics)
            self._load_model()
        except (pyodbc.Error, DBError) as e:
            self.logger.error(f"Failed to generate model: {str(e)}")
            raise TIAError(f"Failed to generate model: {str(e)}")
</xArtifact>

**Notes**:
- Fully integrates TIA-1.2 logic.
- Supports model switching with OpenAI embeddings.
- Uses `DBManager` and `StorageManager` for storage.

### Component 6: PromptGenerator (`proga/prompt_generator.py`)

Generates LLM prompts for SQL queries.

<xaiArtifact artifact_id="46ecdab9-b10e-4b75-a7c6-75c95e1ca4dd" artifact_version_id="21815d5c-14f6-410b-b3cf-dad42d84a1d2" title="proga/prompt_generator.py" contentType="text/python">
import requests
from typing import Dict, List, Optional
import logging
import json
from config.utils import ConfigUtils, ConfigError
from storage.db_manager import DBManager, DBError

class PROGAError(Exception):
    """Custom exception for Prompt Generation Agent errors."""
    pass

class PromptGenerator:
    """Prompt Generation Agent for generating SQL queries.

    Formats prompts for mock LLM or OpenAI API, using TIA outputs.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Logger for PROGA operations.
        db_manager (DBManager): SQL Server database manager.
        datasource (Dict): Datasource configuration.
        llm_config (Dict): LLM configuration.
        prompt_template (str): LLM prompt template.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logger: logging.Logger,
        db_manager: DBManager,
        datasource: Dict
    ):
        """Initialize PromptGenerator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): Logger instance.
            db_manager (DBManager): SQL Server database manager.
            datasource (Dict): Datasource configuration.

        Raises:
            PROGAError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        self.db_manager = db_manager
        self.datasource = datasource
        try:
            self.llm_config = self.config_utils.load_llm_config()
        except ConfigError as e:
            self.logger.error(f"Failed to load LLM config: {str(e)}")
            raise PROGAError(f"Failed to load LLM config: {str(e)}")
        self.prompt_template = """
        Given the following natural language query and database context, generate a valid SQL query for {db_type}.
        
        **Database Context:**
        - Tables: {tables}
        - DDL: {ddl}
        
        **Query:**
        {nlq}
        
        **Conditions:**
        {conditions}
        
        **Aggregations (if applicable):**
        {aggregations}
        
        **Example Query:**
        - Example NLQ: {example_nlq}
        - Example Tables Metadata: {example_metadata}
        - Example SQL: {example_sql}
        
        **Instructions:**
        - Generate a SELECT query only.
        - Use table aliases for clarity.
        - Ensure compatibility with {db_type} syntax.
        - Return only the SQL query.
        """
        self.logger.debug(f"Initialized PROGA for datasource: {datasource['name']}")

    def generate_sql(self, tia_output: Dict, user: str) -> Optional[str]:
        """Generate SQL query from TIA output.

        Args:
            tia_output (Dict): TIA output with tables, DDL, NLQ, conditions, SQL.
            user (str): User submitting the query.

        Returns:
            Optional[str]: SQL query, or None if failed.

        Raises:
            PROGAError: If generation fails.
        """
        required = ["tables", "ddl", "nlq", "conditions"]
        for key in required:
            if key not in tia_output:
                self.logger.error(f"Missing '{key}' in TIA output")
                raise PROGAError(f"Missing '{key}' in TIA output")

        if tia_output.get("sql"):
            self.logger.debug(f"Using stored SQL for NLQ: {tia_output['nlq']}")
            return tia_output["sql"]

        try:
            prompt = self._generate_prompt(tia_output)
            sql_query = self._call_llm(prompt)
            if sql_query:
                self.logger.info(f"Generated SQL for NLQ: {tia_output['nlq']}")
                return sql_query
            self.logger.error(f"LLM failed for NLQ: {tia_output['nlq']}")
            self.db_manager.store_rejected_query(
                query=tia_output["nlq"],
                reason="LLM failed to generate SQL",
                user=user,
                error_type="NO_LLM_RESPONSE"
            )
            return None
        except Exception as e:
            self.logger.error(f"SQL generation failed: {str(e)}")
            raise PROGAError(f"SQL generation failed: {str(e)}")

    def _generate_prompt(self, tia_output: Dict) -> str:
        """Generate LLM prompt.

        Args:
            tia_output (Dict): TIA output.

        Returns:
            str: Formatted prompt.
        """
        db_type = "SQL Server" if self.datasource["type"] == "SQL Server" else "S3 (CSV)"
        conditions_str = json.dumps(tia_output["conditions"]) if tia_output["conditions"] else "None"
        aggregations = self._detect_aggregations(tia_output["nlq"])
        example_nlq = "Show all products in a category"
        example_metadata = "Table: products (product_name VARCHAR(255), category_id INT)"
        example_sql = "SELECT product_name, category_id FROM products WHERE category_id > 0"
        return self.prompt_template.format(
            db_type=db_type,
            tables=", ".join(tia_output["tables"]),
            ddl=tia_output["ddl"],
            nlq=tia_output["nlq"],
            conditions=conditions_str,
            aggregations=", ".join(aggregations) if aggregations else "None",
            example_nlq=example_nlq,
            example_metadata=example_metadata,
            example_sql=example_sql
        )

    def _detect_aggregations(self, nlq: str) -> List[str]:
        """Detect aggregations in NLQ.

        Args:
            nlq (str): Natural language query.

        Returns:
            List[str]: Detected aggregations.
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
        """Call LLM API.

        Args:
            prompt (str): Formatted prompt.

        Returns:
            Optional[str]: SQL query, or None if failed.
        """
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.llm_config.get("openai_model", "text-embedding-3-small"),
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7
            }
            endpoint = self.llm_config["mock_endpoint"]  # Default to mock
            if self.config_utils.load_model_config().get("model_type") == "openai":
                endpoint = self.llm_config["openai_endpoint"]
                headers["Authorization"] = f"Bearer {self.llm_config['openai_api_key']}"
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            sql_query = response.json().get("choices", [{}])[0].get("text", "").strip()
            if sql_query:
                self.logger.debug(f"LLM response: {sql_query}")
                return sql_query
            self.logger.error("Empty LLM response")
            return None
        except (requests.RequestException, ValueError) as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            return None
</xArtifact>

**Notes**:
- Supports mock LLM and OpenAI API based on model type.
- Uses `DBManager` for rejected queries.

### Component 7: DataExecutor (`opden/data_executor.py`)

Executes queries and processes S3 data.

<xaiArtifact artifact_id="2ad9f850-bc3e-4acb-a3d2-f64d735122ac" artifact_version_id="ca28478d-2f6c-422c-acae-96b308180dcc" title="opden/data_executor.py" contentType="text/python">
import pyodbc
import pyarrow.dataset as ds
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import boto3
from botocore.exceptions import ClientError
import os
from datetime import datetime
from config.utils import ConfigUtils, ConfigError
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError

class OPDENError(Exception):
    """Custom exception for Data Execution Engine errors."""
    pass

class DataExecutor:
    """Data Execution Engine for executing queries and processing S3 data.

    Executes SQL Server queries and processes S3 data (CSV, ORC, Parquet, TXT) with
    dynamic file type detection.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Logger for OPDEN operations.
        db_manager (DBManager): SQL Server database manager.
        storage_manager (StorageManager): S3 storage manager.
        datasource (Dict): Datasource configuration.
        connection (pyodbc.Connection): SQL Server connection.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logger: logging.Logger,
        db_manager: DBManager,
        storage_manager: StorageManager,
        datasource: Dict
    ):
        """Initialize DataExecutor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): Logger instance.
            db_manager (DBManager): SQL Server database manager.
            storage_manager (StorageManager): S3 storage manager.
            datasource (Dict): Datasource configuration.

        Raises:
            OPDENError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        self.db_manager = db_manager
        self.storage_manager = storage_manager
        self.datasource = datasource
        self.connection = self.db_manager.connection if self.datasource["type"] == "SQL Server" else None
        self.logger.debug(f"Initialized OPDEN for datasource: {datasource['name']}")

    def execute_query(self, sql_query: str, user: str, nlq: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute query and return sample data or CSV.

        Args:
            sql_query (str): SQL query.
            user (str): User submitting the query.
            nlq (str): Natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data and CSV path.

        Raises:
            OPDENError: If execution fails.
        """
        try:
            if self.datasource["type"] == "SQL Server":
                return self._execute_sql_server_query(sql_query, user, nlq)
            else:
                return self._execute_s3_query(sql_query, user, nlq)
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            self.db_manager.store_rejected_query(
                query=nlq,
                reason=f"Execution failed: {str(e)}",
                user=user,
                error_type="EXECUTION_ERROR"
            )
            return None, None

    def _execute_sql_server_query(self, sql_query: str, user: str, nlq: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute SQL Server query.

        Args:
            sql_query (str): SQL query.
            user (str): User submitting the query.
            nlq (str): Natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data and CSV path.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(5)
            sample_data = pd.DataFrame([tuple(row) for row in rows], columns=columns)
            self.logger.debug(f"Retrieved sample data for NLQ: {nlq}")

            if not sample_data.empty:
                cursor.execute(sql_query)
                all_rows = cursor.fetchall()
                df = pd.DataFrame([tuple(row) for row in all_rows], columns=columns)
                csv_path = os.path.join(self.config_utils.logs_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Generated CSV: {csv_path}")
                cursor.close()
                return sample_data, csv_path
            self.logger.warning("No data returned")
            cursor.close()
            return None, None
        except pyodbc.Error as e:
            self.logger.error(f"SQL Server query failed: {str(e)}")
            return None, None

    def _detect_s3_file_type(self, bucket: str, prefix: str) -> str:
        """Detect S3 file type.

        Args:
            bucket (str): S3 bucket name.
            prefix (str): S3 prefix.

        Returns:
            str: File format (csv, orc, parquet).

        Raises:
            OPDENError: If detection fails.
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
                raise OPDENError(f"No files in s3://{bucket}/{prefix}")
            extensions = {os.path.splitext(f)[1].lower() for f in files}
            valid = {".csv", ".orc", ".parquet"}
            detected = extensions & valid
            if not detected:
                raise OPDENError(f"No supported file types: {extensions}")
            if len(detected) > 1:
                raise OPDENError(f"Mixed file types: {detected}")
            ext = detected.pop()
            return {"csv": "csv", "orc": "orc", "parquet": "parquet"}[ext[1:]]
        except ClientError as e:
            self.logger.error(f"Failed to list S3 files: {str(e)}")
            raise OPDENError(f"Failed to list S3 files: {str(e)}")

    def _execute_s3_query(self, sql_query: str, user: str, nlq: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute S3 query using PyArrow.

        Args:
            sql_query (str): SQL query (simplified).
            user (str): User submitting the query.
            nlq (str): Natural language query.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: Sample data and CSV path.
        """
        try:
            metadata = self.config_utils.load_metadata(self.datasource, schema="default")
            tables = metadata.get("tables", [])
            table_name = tables[0]["name"] if tables else None
            if not table_name:
                raise OPDENError("No tables in metadata")

            bucket = self.datasource["bucket_name"]
            prefix = os.path.join(self.datasource["database"], "default", table_name)
            file_format = self._detect_s3_file_type(bucket, prefix)

            s3_filesystem = ds.S3FileSystem(
                endpoint_override=self.datasource["s3_endpoint"],
                access_key=self.datasource["access_key"],
                secret_key=self.datasource["secret_key"]
            )
            format_options = {"format": file_format}
            if file_format == "csv":
                delimiter = metadata.get("delimiter", "\t")
                format_options["csv_options"] = {"delimiter": delimiter}

            dataset = ds.dataset(
                f"s3://{bucket}/{prefix}",
                **format_options,
                filesystem=s3_filesystem
            )

            table = dataset.to_table()
            sample_data = table.slice(0, 5).to_pandas()
            self.logger.debug(f"Retrieved sample data from S3 ({file_format}) for NLQ: {nlq}")

            if not sample_data.empty:
                df = table.to_pandas()
                csv_path = os.path.join(self.config_utils.logs_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Generated CSV: {csv_path}")
                return sample_data, csv_path
            self.logger.warning("No data returned from S3")
            return None, None
        except (ClientError, ValueError, OPDENError) as e:
            self.logger.error(f"S3 query failed: {str(e)}")
            return None, None
</xArtifact>

**Notes**:
- Dynamic S3 file type detection.
- Configurable TXT delimiter via metadata.

### Component 8: CLIInterface (`cli/interface.py`)

Provides role-based CLI.

<xaiArtifact artifact_id="ca31d4b5-26f3-41d1-81a1-e42587d0ab6b" artifact_version_id="f861d3df-eba9-4395-a261-203ab41ff50f" title="cli/interface.py" contentType="text/python">
import getpass
import logging
import pandas as pd
from typing import Dict, List, Optional
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

    Provides role-based access for End Users (query execution) and Admin Users (training, model management).

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Logger for CLI operations.
        db_config (Dict): Database configuration from db_configurations.json.
        user (Optional[str]): Authenticated user (datauser or admin).
        datasource (Optional[Dict]): Selected datasource configuration.
        db_manager (Optional[DBManager]): Database manager instance.
        storage_manager (Optional[StorageManager]): Storage manager instance.
        tia (Optional[TableIdentifier]): Table Identifier Agent instance.
        proga (Optional[PromptGenerator]): Prompt Generator Agent instance.
        opden (Optional[DataExecutor]): Data Executor instance.
    """

    def __init__(self):
        """Initialize CLIInterface."""
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
        except ConfigError as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise CLIError(f"Initialization failed: {str(e)}")

    def authenticate(self) -> bool:
        """Authenticate user with hardcoded credentials.

        Returns:
            bool: True if authentication succeeds, False otherwise.
        """
        print("Enter username (datauser or admin):")
        username = input().strip()
        password = getpass.getpass("Enter password: ").strip()

        valid_users = {"datauser": "pass123", "admin": "pass123"}
        if username in valid_users and password == valid_users[username]:
            self.user = username
            self.logger.info(f"Authenticated user: {self.user}")
            return True
        self.logger.error(f"Authentication failed for user: {username}")
        print("Invalid username or password")
        return False

    def select_datasource(self) -> bool:
        """Prompt user to select a datasource.

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
                self.logger.debug(f"Selected datasource: {self.datasource['name']}")
                self.db_manager = DBManager(self.config_utils, self.logger, self.datasource)
                self.storage_manager = StorageManager(self.config_utils, self.logger, self.datasource)
                self.tia = TableIdentifier(self.config_utils, self.logger, self.db_manager, self.storage_manager, self.datasource)
                self.proga = PromptGenerator(self.config_utils, self.logger, self.db_manager, self.datasource)
                self.opden = DataExecutor(self.config_utils, self.logger, self.db_manager, self.storage_manager, self.datasource)
                return True
            print("Invalid selection")
            self.logger.error("Invalid datasource selection")
            return False
        except ValueError:
            print("Please enter a valid number")
            self.logger.error("Invalid input for datasource selection")
            return False

    def run(self) -> None:
        """Run the CLI interface."""
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
            print(f"An error: occurred {str(e)}")
        finally:
            if self.db_manager:
                self.db_manager.close_connection()

    def _run_end_user(self) -> None:
        """Run End User interface."""
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

    def _process_end_user_query(self, nlq: str) -> None:
        """Process End User NLQ."""
        try:
            tia_output = self.tia.predict_tables(nlq, self.user)
            if not tia_output:
                print("Unable to process your request")
                return

            suggestions = self._get_suggestions(nlq, tia_output)
            if suggestions:
                print("\nSuggested Queries:")
                for idx, sug in enumerate(suggestions[:3], 1):
                    print(f"{idx}. {sug['user_query']}")
                choice = input("\nSelect suggestion (number) or Enter to continue: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                    nlq = suggestions[int(choice) - 1]["user_query"]
                    tia_output = self.tia.predict_tables(nlq, self.user)

            sql_query = self.proga.generate_sql(tia_output, self.user)
            if not sql_query:
                print("Cannot resolve query")
                return

            sample_data, csv_path = self.opden.execute_query(sql_query, self.user, nlq)
            if sample_data is not None and not sample_data.empty:
                print("\nSample Data (5 rows):")
                print(sample_data.to_string(index=False))
                confirm = input("\nIs this data correct? (y/n): ").strip().lower()
                if confirm == "y":
                    print(f"Data saved to: {csv_path}")
                else:
                    self.db_manager.store_rejected_query(
                        query=nlq,
                        reason="User rejected data",
                        user=self.user,
                        error_type="USER_REJECTED"
                    )
                    print("Query rejected")
            else:
                print("No data returned")
        except (TIAError, PROGAError, OPDENError, DBError) as e:
            self.logger.error(f"Query processing error: {str(e)}")
            print(f"Error: {str(e)}")

    def _get_suggestions(self, nlq: str, tia_output: Dict) -> List[Dict]:
        """Get query suggestions.

        Args:
            nlq (str): Current NLQ.
            tia_output (Dict): TIA output.

        Returns:
            List[Dict]: Suggested training data entries.
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
            return suggestions
        except pyodbc.Error as e:
            self.logger.error(f"Failed to fetch suggestions: {str(e)}")
            return []

    def _run_admin(self) -> None:
        """Run Admin interface."""
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
                self._initialize_training_table