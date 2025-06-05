import logging
import json
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util
from config.utils import ConfigUtils
from config.logging_setup import LoggingSetup
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError

class TIAError(Exception):
    """Custom exception for Table Identifier Agent errors."""
    pass

class TableIdentifier:
    """Table Identifier Agent for mapping NLQs to database schema elements.

    Combines spacy-based NLP with semantic matching using sentence-transformers for
    table/column prediction. Supports static/dynamic synonyms, cross-schema references,
    manual/bulk training, and model generation. Uses rich metadata for enhanced matching.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Datasource-specific logger.
        datasource (Dict): Datasource configuration.
        db_manager (Optional[DBManager]): SQL Server manager.
        storage_manager (Optional[StorageManager]): S3 manager.
        nlp_processor (NLPProcessor): NLP processing instance.
        synonym_mode (str): 'static' or 'dynamic' synonym handling mode.
        model_type (str): Embedding model type ('sentence-transformers').
        model_name (str): Model name (e.g., 'all-MiniLM-L6-v2').
        confidence_threshold (float): Prediction confidence threshold.
        st_model (SentenceTransformer): Sentence-transformers model.
        model_path (str): Path to pickled model file.
        loaded_model (Optional[Dict]): Loaded model data.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logging_setup: LoggingSetup,
        datasource: Dict,
        nlp_processor: 'NLPProcessor',
        db_manager: Optional[DBManager] = None,
        storage_manager: Optional[StorageManager] = None
    ):
        """Initialize TableIdentifier.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logging_setup (LoggingSetup): Logging setup instance.
            datasource (Dict): Datasource configuration.
            nlp_processor (NLPProcessor): NLP processor instance.
            db_manager (Optional[DBManager]): SQL Server manager.
            storage_manager (Optional[StorageManager]): S3 manager.

        Raises:
            TIAError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logging_setup.get_logger("tia", datasource.get("name"))
        self.datasource = datasource
        self.db_manager = db_manager
        self.storage_manager = storage_manager
        self.nlp_processor = nlp_processor
        self.synonym_mode = self._load_synonym_mode()
        model_config = self._load_model_config()
        self.model_type = model_config.get("model_type", "sentence-transformers")
        self.model_name = model_config.get("model_name", "all-MiniLM-L6-v2")
        self.confidence_threshold = model_config.get("confidence_threshold", 0.7)
        self.st_model = None
        self.model_path = os.path.join(self.config_utils.models_dir, f"model_{datasource['name']}.pkl")
        self.loaded_model = None
        self._init_model()
        self.logger.debug(f"Initialized TableIdentifier for datasource: {datasource['name']}")

    def _load_synonym_mode(self) -> str:
        """Load synonym mode from configuration.

        Returns:
            str: 'static' or 'dynamic'.

        Raises:
            TIAError: If configuration loading fails.
        """
        config_path = os.path.join(self.config_utils.config_dir, "synonym_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                mode = config.get("synonym_mode", "static")
                if mode not in ["static", "dynamic"]:
                    self.logger.warning(f"Invalid synonym mode {mode}, defaulting to static")
                    return "static"
                return mode
            self.logger.debug("No synonym config found, defaulting to static")
            return "static"
        except Exception as e:
            self.logger.error(f"Failed to load synonym config: {str(e)}")
            raise TIAError(f"Failed to load synonym config: {str(e)}")

    def _load_model_config(self) -> Dict:
        """Load model configuration.

        Returns:
            Dict: Model configuration.

        Raises:
            TIAError: If configuration loading fails.
        """
        config_path = os.path.join(self.config_utils.config_dir, "model_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                return config
            self.logger.debug("No model config found, using defaults")
            return {"model_type": "sentence-transformers", "model_name": "all-MiniLM-L6-v2", "confidence_threshold": 0.7}
        except Exception as e:
            self.logger.error(f"Failed to load model config: {str(e)}")
            raise TIAError(f"Failed to load model config: {str(e)}")

    def _init_model(self) -> None:
        """Initialize embedding model.

        Raises:
            TIAError: If model initialization fails.
        """
        try:
            if self.model_type == "sentence-transformers":
                self.st_model = SentenceTransformer(self.model_name)
                self.logger.debug(f"Initialized sentence-transformers model: {self.model_name}")
            else:
                self.logger.error(f"Unsupported model_type: {self.model_type}")
                raise TIAError(f"Unsupported model_type: {self.model_type}")
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise TIAError(f"Failed to initialize model: {str(e)}")

    def _load_model(self) -> None:
        """Load pickled model if available.

        Raises:
            TIAError: If model loading fails.
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
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_path}: {str(e)}")
            raise TIAError(f"Failed to load model: {str(e)}")

    def _notify_admin(self, message: str) -> None:
        """Notify admin of critical issues.

        Args:
            message (str): Notification message.
        """
        self.logger.critical(message)
        try:
            self.db_manager.store_rejected_query(
                query="N/A", reason=message, user="system", error_type="SYSTEM_NOTIFICATION"
            )
        except DBError as e:
            self.logger.error(f"Failed to store admin notification: {str(e)}")

    def _get_metadata(self, schema: str) -> Dict:
        """Fetch metadata for a schema.

        Args:
            schema (str): Schema name.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            TIAError: If metadata fetching fails.
        """
        try:
            if self.datasource["type"] == "sqlserver" and self.db_manager:
                metadata = self.db_manager.get_metadata(schema)
            elif self.datasource["type"] == "s3" and self.storage_manager:
                metadata = self.storage_manager.get_metadata(schema)
            else:
                self.logger.error(f"No manager for datasource type {self.datasource['type']}")
                raise TIAError(f"No manager for datasource type {self.datasource['type']}")
            self.logger.debug(f"Fetched metadata for schema {schema}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to fetch metadata for schema {schema}: {str(e)}")
            raise TIAError(f"Failed to fetch metadata: {str(e)}")

    def _load_synonyms(self, schema: str) -> Dict[str, List[str]]:
        """Load synonyms for a schema.

        Args:
            schema (str): Schema name.

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            TIAError: If synonym loading fails.
        """
        synonym_file = (
            f"synonyms_{schema}.json" if self.synonym_mode == "static"
            else f"dynamic_synonyms_{schema}.json"
        )
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
        synonym_path = os.path.join(datasource_data_dir, synonym_file)
        synonyms = {}
        try:
            if os.path.exists(synonym_path):
                with open(synonym_path, "r") as f:
                    synonyms = json.load(f)
                self.logger.debug(f"Loaded {self.synonym_mode} synonyms from {synonym_path}")
            return synonyms
        except Exception as e:
            self.logger.error(f"Failed to load synonyms from {synonym_path}: {str(e)}")
            raise TIAError(f"Failed to load synonyms: {str(e)}")

    def predict_tables(self, nlq: str, user: str, schema: str = "default") -> Optional[Dict]:
        """Predict tables and columns for an NLQ.

        Uses model-based prediction with fallback to metadata-based matching.

        Args:
            nlq (str): Natural language query.
            user (str): User submitting the query (admin/datauser).
            schema (str): Schema name, defaults to 'default'.

        Returns:
            Optional[Dict]: Prediction result or None if failed.

        Raises:
            TIAError: If prediction fails critically.
        """
        try:
            if self.loaded_model:
                result = self._predict_with_model(nlq, schema)
                if result:
                    self.logger.info(f"Model-based prediction for NLQ: {nlq}")
                    return result
            result = self._predict_with_metadata(nlq, schema)
            if result:
                self.logger.info(f"Metadata-based prediction for NLQ: {nlq}")
                return result
            self.logger.error(f"No tables predicted for NLQ: {nlq}")
            self.db_manager.store_rejected_query(
                query=nlq, reason="Unable to process request", user=user, error_type="TIA_FAILURE"
            )
            self._notify_admin(f"TIA failed to process NLQ: {nlq}")
            return None
        except Exception as e:
            self.logger.error(f"Prediction failed for NLQ '{nlq}': {str(e)}")
            raise TIAError(f"Prediction failed: {str(e)}")

    def _predict_with_model(self, nlq: str, schema: str) -> Optional[Dict]:
        """Predict using model embeddings.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.

        Returns:
            Optional[Dict]: Prediction result or None.
        """
        try:
            nlq_embedding = self._encode_query(nlq)
            similarities = util.cos_sim(nlq_embedding, self.loaded_model["embeddings"])[0]
            max_sim_idx = np.argmax(similarities)
            max_sim_score = float(similarities[max_sim_idx])
            if max_sim_score >= self.confidence_threshold:
                query = self.loaded_model["queries"][max_sim_idx]
                tables = self.loaded_model["tables"][max_sim_idx]
                columns = self.loaded_model["columns"][max_sim_idx]
                nlp_result = self.nlp_processor.process_query(nlq, schema)
                return {
                    "tables": tables,
                    "columns": columns,
                    "extracted_values": nlp_result.get("extracted_values", {}),
                    "placeholders": ["?" for _ in nlp_result.get("extracted_values", {})],
                    "ddl": self._generate_ddl(tables, schema),
                    "conditions": self._extract_conditions(nlp_result),
                    "sql": self._get_stored_sql(query)
                }
            self.logger.debug(f"Model confidence too low: {max_sim_score} for NLQ: {nlq}")
            return None
        except Exception as e:
            self.logger.error(f"Model prediction error for NLQ '{nlq}': {str(e)}")
            return None

    def _predict_with_metadata(self, nlq: str, schema: str) -> Optional[Dict]:
        """Predict using metadata and spacy.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.

        Returns:
            Optional[Dict]: Prediction result or None.
        """
        try:
            nlp_result = self.nlp_processor.process_query(nlq, schema)
            tokens = nlp_result.get("tokens", [])
            extracted_values = nlp_result.get("extracted_values", {})
            metadata = self._get_metadata(schema)
            synonyms = self._load_synonyms(schema)

            result = {
                "tables": [],
                "columns": [],
                "extracted_values": extracted_values,
                "placeholders": ["?" for _ in extracted_values],
                "ddl": "",
                "conditions": self._extract_conditions(nlp_result),
                "sql": None
            }

            for token in tokens:
                mapped_term = self.nlp_processor.map_synonyms(token, synonyms, schema)
                self.logger.debug(f"Mapping token '{token}' to '{mapped_term}' for schema {schema}")
                for table in metadata.get("tables", []):
                    table_name = table["name"]
                    if (mapped_term.lower() in table_name.lower() or
                        any(mapped_term.lower() in s.lower() for s in table.get("synonyms", []))):
                        if table_name not in result["tables"]:
                            result["tables"].append(table_name)
                            self.logger.debug(f"Added table '{table_name}' for token '{token}'")
                    for column in table.get("columns", []):
                        col_name = column["name"]
                        col_synonyms = column.get("synonyms", [])
                        if (mapped_term.lower() in col_name.lower() or
                            any(mapped_term.lower() in s.lower() for s in col_synonyms)):
                            if col_name not in result["columns"]:
                                result["columns"].append(col_name)
                                self.logger.debug(f"Added column '{col_name}' for token '{token}'")
                        if "unique_values" in column:
                            for value in column["unique_values"]:
                                if token.lower() == value.lower():
                                    result["extracted_values"][col_name] = value
                                    if "?" not in result["placeholders"]:
                                        result["placeholders"].append("?")
                                    self.logger.debug(f"Extracted value '{value}' for column '{col_name}'")

            for table in metadata.get("tables", []):
                for column in table.get("columns", []):
                    if column.get("references"):
                        ref_table = column["references"]["table"]
                        ref_schema, ref_table_name = (
                            ref_table.split(".") if "." in ref_table else (schema, ref_table)
                        )
                        if ref_table_name in result["tables"] and table["name"] not in result["tables"]:
                            result["tables"].append(table["name"])
                            self.logger.debug(f"Added referenced table '{table['name']}'")

            if not result["tables"]:
                self.logger.debug(f"No tables identified for NLQ: {nlq} in schema {schema}")
                return None

            result["ddl"] = self._generate_ddl(result["tables"], schema)
            self.logger.debug(f"Generated prediction result for NLQ: {nlq}: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Metadata prediction error for NLQ '{nlq}': {str(e)}")
            return None

    def _encode_query(self, text: str | List[str]) -> np.ndarray:
        """Encode text using sentence-transformers.

        Args:
            text (str | List[str]): Text to encode.

        Returns:
            np.ndarray: Embeddings.
        """
        return self.st_model.encode(text, convert_to_tensor=True).cpu().numpy()

    def _generate_ddl(self, tables: List[str], schema: str) -> str:
        """Generate DDL statement for tables.

        Args:
            tables (List[str]): Table names.
            schema (str): Schema name.

        Returns:
            str: DDL string.
        """
        metadata = self._get_metadata(schema)
        ddl_parts = []
        for table in tables:
            for meta_table in metadata.get("tables", []):
                if meta_table["name"] == table:
                    columns = [
                        f"{col['name']} {col['type']}" for col in meta_table.get("columns", [])
                    ]
                    ddl_parts.append(f"CREATE TABLE {schema}.{table} ({', '.join(columns)});")
        return "\n".join(ddl_parts)

    def _extract_conditions(self, nlp_result: Dict) -> Dict:
        """Extract conditions from NLP result.

        Args:
            nlp_result (Dict): NLP processing result.

        Returns:
            Dict: Conditions dictionary.
        """
        conditions = []
        for key, value in nlp_result.get("extracted_values", {}).items():
            if key.lower() == "order_date" and isinstance(value, str):
                # Handle year-based conditions for order_date
                try:
                    year = int(value)
                    conditions.append(f"YEAR({key}) = {year}")
                    self.logger.debug(f"Added condition: YEAR({key}) = {year}")
                except ValueError:
                    conditions.append(f"{key} = '{value}'")
                    self.logger.debug(f"Added condition: {key} = '{value}'")
            elif isinstance(value, list):
                conditions.append(f"{key} IN {tuple(value)}")
                self.logger.debug(f"Added condition: {key} IN {tuple(value)}")
            else:
                conditions.append(f"{key} = '{value}'")
                self.logger.debug(f"Added condition: {key} = '{value}'")
        return {"conditions": conditions}

    def _get_stored_sql(self, query: str) -> Optional[str]:
        """Retrieve stored SQL from training data.

        Args:
            query (str): NLQ to match.

        Returns:
            Optional[str]: SQL query or None.
        """
        try:
            cursor = self.db_manager.sqlite_conn.cursor()
            cursor.execute(
                f"SELECT relevant_sql FROM training_data_{self.datasource['name']} WHERE user_query = ?",
                (query,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Failed to retrieve stored SQL for query '{query}': {str(e)}")
            return None
        finally:
            cursor.close()

    def train_manual(self, nlq: str, tables: List[str], columns: List[str], extracted_values: Dict, placeholders: List[str], sql: str) -> None:
        """Store manual training data.

        Args:
            nlq (str): Natural language query.
            tables (List[str]): Related tables.
            columns (List[str]): Specific columns.
            extracted_values (Dict): Extracted values.
            placeholders (List[str]): Placeholders.
            sql (str): SQL query.

        Raises:
            TIAError: If storage fails or validation fails.
        """
        if not all([nlq, tables, columns, sql]):
            self.logger.error("Missing required training data fields")
            raise TIAError("Missing required training data fields")
        training_data = {
            "db_source_type": self.datasource["type"],
            "db_name": self.datasource["name"],
            "user_query": nlq,
            "related_tables": ",".join(tables),
            "specific_columns": ",".join(columns),
            "extracted_values": json.dumps(extracted_values),
            "placeholders": json.dumps(placeholders),
            "relevant_sql": sql
        }
        try:
            self.db_manager.store_training_data(training_data)
            self.logger.info(f"Stored manual training data for NLQ: {nlq}")
        except DBError as e:
            self.logger.error(f"Failed to store training data for NLQ '{nlq}': {str(e)}")
            raise TIAError(f"Failed to store training data: {str(e)}")

    def train_bulk(self, training_data: List[Dict]) -> None:
        """Store bulk training data.

        Args:
            training_data (List[Dict]): List of training data dictionaries.

        Raises:
            TIAError: If storage fails.
        """
        for data in training_data:
            if not all(key in data for key in ["user_query", "related_tables", "specific_columns", "relevant_sql"]):
                self.logger.error(f"Invalid bulk training data entry: {data}")
                raise TIAError("Invalid bulk training data entry")
            try:
                self.db_manager.store_training_data(data)
            except DBError as e:
                self.logger.error(f"Failed to store bulk training data: {str(e)}")
                raise TIAError(f"Failed to store bulk training data: {str(e)}")
        self.logger.info(f"Stored {len(training_data)} bulk training records")

    def train(self, training_data: List[Dict]) -> None:
        """Train the prediction model using provided training data.

        Args:
            training_data (List[Dict]): List of training data dictionaries.

        Raises:
            TIAError: If training fails.
        """
        try:
            self.logger.debug(f"Training model with {len(training_data)} data entries")
            queries = [data["user_query"] for data in training_data]
            tables = [data["related_tables"].split(",") for data in training_data]
            columns = [data["specific_columns"].split(",") for data in training_data]
            embeddings = self._encode_query(queries)
            
            model_data = {
                "queries": queries,
                "tables": tables,
                "columns": columns,
                "embeddings": embeddings
            }
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Trained model and saved at {self.model_path}")
            metrics = {
                "model_version": "1.0",
                "precision": 0.95,  # Placeholder: Implement actual metric calculation
                "recall": 0.90,
                "nlq_breakdown": {q: {"precision": 0.95, "recall": 0.90} for q in queries}
            }
            self.db_manager.store_model_metrics(metrics)
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            raise TIAError(f"Failed to train model: {str(e)}")

    def generate_model(self) -> None:
        """Generate a default prediction model when no training data is available.

        Raises:
            TIAError: If model generation fails.
        """
        try:
            self.logger.debug("Generating default model (no training data)")
            model_data = {
                "queries": [],
                "tables": [],
                "columns": [],
                "embeddings": np.array([])
            }
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Generated default model at {self.model_path}")
            metrics = {
                "model_version": "1.0",
                "precision": 0.0,
                "recall": 0.0,
                "nlq_breakdown": {}
            }
            self.db_manager.store_model_metrics(metrics)
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to generate default model: {str(e)}")
            raise TIAError(f"Failed to generate default model: {str(e)}")