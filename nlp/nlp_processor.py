import spacy
import json
import os
from typing import Dict, List, Optional
import logging
from config.utils import ConfigUtils
from config.logging_setup import LoggingSetup
from storage.db_manager import DBManager
from storage.storage_manager import StorageManager

class NLPError(Exception):
    """Custom exception for NLP processing errors."""
    pass

class NLPProcessor:
    """NLP Processor for handling natural language queries in the Datascriber project.

    Uses spacy for tokenization, named entity recognition (NER), and value extraction.
    Supports static and dynamic synonym handling, with dynamic synonyms generated via
    spacy word embeddings. Integrates with rich metadata for enhanced matching.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): Datasource-specific logger.
        datasource (Dict): Datasource configuration.
        db_manager (Optional[DBManager]): SQL Server manager.
        storage_manager (Optional[StorageManager]): S3 manager.
        nlp (spacy.language.Language): Spacy NLP model.
        synonym_mode (str): 'static' or 'dynamic' synonym handling mode.
    """

    def __init__(
        self,
        config_utils: ConfigUtils,
        logging_setup: LoggingSetup,
        datasource: Dict,
        db_manager: Optional[DBManager] = None,
        storage_manager: Optional[StorageManager] = None
    ):
        """Initialize NLPProcessor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logging_setup (LoggingSetup): Logging setup instance.
            datasource (Dict): Datasource configuration.
            db_manager (Optional[DBManager]): SQL Server manager.
            storage_manager (Optional[StorageManager]): S3 manager.

        Raises:
            NLPError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logging_setup.get_logger("nlp", datasource.get("name"))
        self.datasource = datasource
        self.db_manager = db_manager
        self.storage_manager = storage_manager
        self.nlp = None
        self.synonym_mode = self._load_synonym_mode()
        self._init_nlp()
        self.logger.debug(f"Initialized NLPProcessor for datasource: {datasource['name']}")

    def _init_nlp(self) -> None:
        """Initialize spacy NLP model.

        Raises:
            NLPError: If spacy model loading fails.
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.debug("Loaded spacy model: en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {str(e)}")
            raise NLPError(f"Failed to load spacy model: {str(e)}")

    def _load_synonym_mode(self) -> str:
        """Load synonym mode from configuration.

        Returns:
            str: 'static' or 'dynamic'.

        Raises:
            NLPError: If configuration loading fails.
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
            raise NLPError(f"Failed to load synonym config: {str(e)}")

    def _load_synonyms(self, schema: str) -> Dict[str, List[str]]:
        """Load synonyms for a schema.

        Args:
            schema (str): Schema name (e.g., 'default').

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            NLPError: If synonym loading fails.
        """
        synonym_file = (
            f"synonyms_{schema}.json" if self.synonym_mode == "static"
            else f"dynamic_synonyms_{schema}.json"
        )
        datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
        synonym_path = os.path.join(datasource_data_dir, synonym_file)
        synonyms = {}
        try:
            os.makedirs(datasource_data_dir, exist_ok=True)
            if os.path.exists(synonym_path):
                with open(synonym_path, "r") as f:
                    synonyms = json.load(f)
                self.logger.debug(f"Loaded {self.synonym_mode} synonyms from {synonym_path}")
            else:
                self.logger.debug(f"No synonym file found at {synonym_path}, returning empty synonyms")
            return synonyms
        except Exception as e:
            self.logger.error(f"Failed to load synonyms from {synonym_path}: {str(e)}")
            raise NLPError(f"Failed to load synonyms: {str(e)}")

    def _generate_dynamic_synonyms(self, term: str, schema: str) -> List[str]:
        """Generate dynamic synonyms using spacy word embeddings.

        Args:
            term (str): Term to find synonyms for.
            schema (str): Schema name.

        Returns:
            List[str]: List of synonyms.

        Raises:
            NLPError: If synonym generation fails.
        """
        try:
            metadata = self._get_metadata(schema)
            candidate_terms = set()
            for table in metadata.get("tables", []):
                candidate_terms.add(table["name"])
                candidate_terms.update(col["name"] for col in table.get("columns", []))
                candidate_terms.update(sum((col.get("synonyms", []) for col in table.get("columns", [])), []))
                candidate_terms.update(sum((col.get("unique_values", []) for col in table.get("columns", [])), []))

            term_doc = self.nlp(term)
            if not term_doc or not term_doc[0].has_vector:
                self.logger.debug(f"No vector for term: {term}")
                return []

            synonyms = []
            for candidate in candidate_terms:
                candidate_doc = self.nlp(candidate)
                if candidate_doc and candidate_doc[0].has_vector:
                    similarity = term_doc[0].similarity(candidate_doc[0])
                    if similarity > 0.7:  # Similarity threshold
                        synonyms.append(candidate)
                        self.logger.debug(f"Generated synonym: {candidate} for {term} (similarity: {similarity})")

            # Save dynamic synonyms
            datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
            synonym_file = os.path.join(datasource_data_dir, f"dynamic_synonyms_{schema}.json")
            os.makedirs(datasource_data_dir, exist_ok=True)
            existing_synonyms = self._load_synonyms(schema)
            existing_synonyms[term] = synonyms
            with open(synonym_file, "w") as f:
                json.dump(existing_synonyms, f, indent=2)
            self.logger.info(f"Updated dynamic synonyms for schema {schema} at {synonym_file}")
            return synonyms
        except Exception as e:
            self.logger.error(f"Failed to generate dynamic synonyms for {term}: {str(e)}")
            raise NLPError(f"Failed to generate dynamic synonyms: {str(e)}")

    def _get_metadata(self, schema: str) -> Dict:
        """Fetch metadata for a schema.

        Args:
            schema (str): Schema name.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            NLPError: If metadata fetching fails.
        """
        try:
            if self.datasource["type"] == "sqlserver" and self.db_manager:
                metadata = self.db_manager.get_metadata(schema)
            elif self.datasource["type"] == "s3" and self.storage_manager:
                metadata = self.storage_manager.get_metadata(schema)
            else:
                self.logger.error(f"No manager for datasource type {self.datasource['type']}")
                raise NLPError(f"No manager for datasource type {self.datasource['type']}")
            self.logger.debug(f"Fetched metadata for schema {schema}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to fetch metadata for schema {schema}: {str(e)}")
            raise NLPError(f"Failed to fetch metadata: {str(e)}")

    def process_query(self, nlq: str, schema: str = "default") -> Dict:
        """Process an NLQ to extract tokens, entities, and values.

        Args:
            nlq (str): Natural language query (e.g., "Show products in category Bikes").
            schema (str): Schema name, defaults to 'default'.

        Returns:
            Dict: Dictionary with tokens, entities, and extracted values.

        Raises:
            NLPError: If processing fails.
        """
        try:
            doc = self.nlp(nlq)
            tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            entities = {ent.label_: ent.text for ent in doc.ents}
            extracted_values = {}

            metadata = self._get_metadata(schema)
            synonyms = self._load_synonyms(schema)

            for token in tokens:
                # Map token to synonyms
                mapped_term = self.map_synonyms(token, synonyms, schema)
                for table in metadata.get("tables", []):
                    for column in table.get("columns", []):
                        col_name = column["name"]
                        col_synonyms = column.get("synonyms", [])
                        if (mapped_term.lower() == col_name.lower() or
                            any(mapped_term.lower() == s.lower() for s in col_synonyms)):
                            # Check for values in subsequent tokens
                            for t in tokens[tokens.index(token)+1:]:
                                if "unique_values" in column and t in [v.lower() for v in column["unique_values"]]:
                                    extracted_values[col_name] = t
                                    self.logger.debug(f"Extracted value: {t} for column {col_name}")
                                elif entities.get("DATE") and column.get("date_format"):
                                    extracted_values[col_name] = entities["DATE"]
                                    self.logger.debug(f"Extracted date: {entities['DATE']} for column {col_name}")

            result = {
                "tokens": tokens,
                "entities": entities,
                "extracted_values": extracted_values
            }
            self.logger.info(f"Processed NLQ: {nlq}, result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to process NLQ '{nlq}': {str(e)}")
            raise NLPError(f"Failed to process NLQ: {str(e)}")

    def map_synonyms(self, term: str, synonyms: Dict[str, List[str]], schema: str) -> str:
        """Map a term to its canonical form using synonyms.

        Args:
            term (str): Term to map.
            synonyms (Dict[str, List[str]]): Synonym mappings.
            schema (str): Schema name for dynamic synonym generation.

        Returns:
            str: Canonical term or original term if no mapping found.

        Raises:
            NLPError: If mapping fails.
        """
        try:
            term_lower = term.lower()
            for canonical, synonym_list in synonyms.items():
                if term_lower == canonical.lower() or term_lower in [s.lower() for s in synonym_list]:
                    self.logger.debug(f"Mapped term '{term}' to canonical '{canonical}'")
                    return canonical

            if self.synonym_mode == "dynamic":
                dynamic_synonyms = self._generate_dynamic_synonyms(term, schema)
                for synonym in dynamic_synonyms:
                    for canonical, synonym_list in synonyms.items():
                        if synonym.lower() == canonical.lower() or synonym.lower() in [s.lower() for s in synonym_list]:
                            synonyms[canonical].append(term)
                            datasource_data_dir = self.config_utils.get_datasource_data_dir(self.datasource['name'])
                            synonym_file = os.path.join(datasource_data_dir, f"dynamic_synonyms_{schema}.json")
                            os.makedirs(datasource_data_dir, exist_ok=True)
                            with open(synonym_file, "w") as f:
                                json.dump(synonyms, f, indent=2)
                            self.logger.debug(f"Added dynamic synonym '{term}' to '{canonical}'")
                            return canonical

            self.logger.debug(f"No synonym mapping for term: {term}")
            return term
        except Exception as e:
            self.logger.error(f"Failed to map synonyms for term '{term}': {str(e)}")
            raise NLPError(f"Failed to map synonyms: {str(e)}")