# Datascriber Requirements Report

## Overview
Datascriber is a Text-to-SQL system that enables users to query SQL Server and S3 datasources using natural language queries (NLQs). It leverages NLP (`spacy`), sentence embeddings (`all-MiniLM-L6-v2` or Open AI’s `text-embedding-3-small`), and a prediction model (Table Identifier, TIA) to generate SQL queries. The system supports metadata-driven query generation, manual and bulk training workflows, and user-friendly CLI interactions for **Data Users** (end users) and **Admin Users**. It aligns with TIA-1.2 (https://github.com/m-prasad-reddy/TIA-1.2) and ensures scalability, security, and maintainability.

## Functional Requirements

### 1. User Roles

#### Data User (End User)
- **Description**: Non-technical business users querying data with layman NLQs.
- **Actions**:
  - Log in: `login datauser`.
  - Select a datasource: `select-datasource <name>` (e.g., `bikestores`).
  - Enter query mode to submit NLQs (e.g., "Get me sales orders of 2016").
  - View results: Tables, columns, sample data (5 rows), and CSV path.
  - Acknowledge sample data accuracy:
    - **Correct**: Full data query executed, CSV download link provided.
    - **Incorrect**: Notification ("Our backend team is notified. They will assist you soon."), proceed to next NLQ.
  - Exit query mode: `exit`.
- **Constraints**:
  - No schema/table knowledge required.
  - Datasource must have metadata (`metadata_data_<schema>_rich.json`) and a trained model (`model_<datasource>.pkl`).
- **Data Flow**:
  - NLQ → NLP processing → TIA prediction → SQL generation → sample data → user acknowledgment → full data (if approved) or admin notification (if rejected).

#### Admin User
- **Description**: Technical users configuring, training, and maintaining the system.
- **Actions**:
  - Log in: `login admin`.
  - Configure datasources: Edit `app-config/db_configurations.json`.
  - Manage metadata: `refresh-metadata` to generate/update `metadata_data_<schema>.json`.
  - Set synonym mode: `set-synonym-mode static|dynamic`.
  - Train TIA model:
    - **Manual**: `map-failed-query` to map NLQs to tables, columns, SQL.
    - **Bulk**: Upload `training_data.csv` for batch training.
  - Handle failed queries:
    - Receive notifications via `rejected_queries_<datasource>` table.
    - Edit tables, columns, SQL for failed NLQs.
  - Retrain model: `train-model`.
  - Test NLQs: `query <nlq>`.
- **Constraints**:
  - Requires admin privileges.
  - Must ensure metadata and training data exist before data user access.
- **Data Flow**:
  - Training: NLQ + mappings → `training_data_<datasource>` → model training → `model_<datasource>.pkl`.
  - Failed Query: Notification → manual mapping → update `training_data_<datasource>` → retrain.

### 2. Components and Workflows

#### CLI (`cli/interface.py`)
- **Purpose**: Command-line interface for user interactions.
- **Inputs**:
  - Commands: `login <username>`, `select-datasource <name>`, `query <nlq>`, `query-mode`, `refresh-metadata`, `train-model`, `set-synonym-mode <mode>`, `map-failed-query`, `list-datasources`, `list-schemas`, `exit`.
  - NLQs in query mode.
- **Outputs**:
  - Query results: Tables, columns, sample data (5 rows), CSV path.
  - Notifications: "Query cannot be processed. Notified admin.", "Our backend team is notified."
  - Confirmation/error messages.
- **Validations**:
  - Username: `admin` or `datauser`.
  - Datasource: Exists in `db_configurations.json`.
  - NLQ: Non-empty, ≤500 characters.
- **Navigation**:
  - **Data User**: `login datauser` → `select-datasource` → query mode → NLQ or `exit`.
  - **Admin**: `login admin` → `select-datasource` → commands → `exit`.

#### Orchestrator (`core/orchestrator.py`)
- **Purpose**: Coordinates NLQ processing, training, and metadata management.
- **Inputs**:
  - NLQ, datasource, schema (optional).
  - Training data (NLQ, tables, columns, SQL).
- **Outputs**:
  - Query results: `{tables, columns, sample_data, csv_path}`.
  - Failure: Log to `rejected_queries_<datasource>`, notify admin.
- **Validations**:
  - Metadata files exist.
  - TIA model exists or is trainable.

#### NLP Processor (`nlp/nlp_processor.py`)
- **Purpose**: Extracts tokens, entities, and values from NLQs using `spacy` (`en_core_web_sm`).
- **Inputs**:
  - NLQ (e.g., "Get me sales orders of 2017").
  - Metadata (`metadata_data_<schema>_rich.json`).
- **Outputs**:
  - `{tokens, entities, extracted_values}` (e.g., `extracted_values: {"order_date": "2017"}`).
- **Validations**:
  - `spacy` model loaded.
  - NLQ non-empty.

#### Table Identifier (`tia/table_identifier.py`)
- **Purpose**: Predicts tables/columns using sentence embeddings.
- **Inputs**:
  - NLQ, `extracted_values`, metadata.
- **Outputs**:
  - `{tables, columns, extracted_values, placeholders, sql}`.
- **Validations**:
  - Model exists or trainable.
  - Confidence >0.7.
- **New Feature**:
  - Switch between `all-MiniLM-L6-v2` and `text-embedding-3-small` via `model_config.json`.

#### Prompt Generator (`proga/prompt_generator.py`)
- **Purpose**: Generates SQL queries (mock mode or future LLM integration).
- **Inputs**:
  - NLQ, tables, columns, `extracted_values`, `entities`.
- **Outputs**:
  - SQL query (e.g., `SELECT order_id, order_date FROM sales.orders WHERE YEAR(order_date) = '2017'`).
- **Validations**:
  - `llm_config.json` specifies `mock_enabled`.

#### Data Executor (`data_executor.py`)
- **Purpose**: Executes SQL queries on SQL Server or S3.
- **Inputs**:
  - SQL query, `extracted_values`.
- **Outputs**:
  - Sample data (5 rows), full data CSV.
- **Validations**:
  - Valid database connection.

#### DB Manager (`storage/db_manager.py`)
- **Purpose**: Manages SQL Server metadata and data access.
- **Inputs**:
  - Datasource configuration, schema.
- **Outputs**:
  - Metadata, table data.

#### Storage Manager (`storage/storage_manager.py`)
- **Purpose**: Manages S3 metadata and data access.
- **Inputs**:
  - Datasource configuration, schema.
- **Outputs**:
  - Metadata, table data.

### 3. Training Workflows

#### Manual Training (Admin)
- **Steps**:
  1. `login admin`, `select-datasource <name>`.
  2. `map-failed-query` or manual NLQ input.
  3. Enter NLQ (e.g., "Get me sales orders of 2017").
  4. TIA suggests tables (e.g., `sales.orders`).
  5. Admin confirms or edits tables.
  6. Enter reference columns (e.g., `order_id, order_date`).
     - **Validation**: Columns must belong to suggested tables.
  7. Enter SQL query (e.g., `SELECT order_id, order_date FROM sales.orders WHERE YEAR(order_date) = '2017'`).
  8. Review NLQ, tables, columns, SQL; confirm.
  9. `data_executor` runs SQL for sample data (5 rows).
     - **Success**: Display sample data, save to `training_data_<datasource>`.
     - **Failure**: Re-prompt for SQL.
  10. Transaction logged to `training_data_<datasource>`.
  11. `train-model` to update `model_<datasource>.pkl`.
- **Data Flow**:
  - NLQ + mappings → `training_data_<datasource>` → model training.
- **Validation**:
  - Columns from suggested tables.
  - SQL executes successfully.

#### Bulk Training (Admin)
- **Steps**:
  1. `login admin`, `select-datasource <name>`.
  2. Provide `training_data.csv` path.
  3. System validates CSV fields:
     - Required: `SCENARIO_ID`, `DB_CONFIG_TYPE`, `DB_NAME`, `USER_QUERY`, `RELATED_TABLES`, `SPECIFIC_COLUMNS`, `RELEVANT_SQL`.
     - Optional: `EXTRACTED_VALUES`, `PLACE_HOLDERS`, `LLM_SQL`, `IS_LSQL_VALID`, `CONTEXT_TEXT1`, `CONTEXT_TEXT2`.
  4. For existing NLQs:
     - Prompt: Ignore or update `training_data_<datasource>`.
  5. Insert validated scenarios into `training_data_<datasource>`.
  6. `train-model` to update `model_<datasource>.pkl`.
  7. Back-test with NLQs.
- **Data Flow**:
  - CSV → validation → `training_data_<datasource>` → model training.

#### Training Data Structure (`training_data_<datasource>`)
- **Fields**:
  - `SCENARIO_ID`: Unique identifier.
  - `DB_CONFIG_TYPE`: `sqlserver` or `s3`.
  - `DB_NAME`: Datasource name (e.g., `bikestores`).
  - `USER_QUERY`: NLQ (e.g., "Get me sales orders of 2017").
  - `RELATED_TABLES`: Comma-separated (e.g., `sales.orders`).
  - `SPECIFIC_COLUMNS`: Comma-separated (e.g., `order_id,order_date`).
  - `EXTRACTED_VALUES`: JSON (e.g., `{"order_date": "2017"}`).
  - `PLACE_HOLDERS`: JSON (e.g., `["?"]`).
  - `RELEVANT_SQL`: Admin-provided SQL.
  - `LLM_SQL`: LLM-generated SQL (post-user acknowledgment).
  - `IS_LSQL_VALID`: `Y` or `N` (data user acknowledgment).
  - `CONTEXT_TEXT1`, `CONTEXT_TEXT2`: Reserved for future enhancements.
- **Validation**:
  - Required fields non-null.
  - `EXTRACTED_VALUES`, `PLACE_HOLDERS` valid JSON.

#### Data User Query Scenarios
- **Scenario 1: Existing NLQ Match**:
  - NLQ matches `training_data_<datasource>` (e.g., "Get me sales orders of 2016").
  - TIA suggests matching NLQs.
  - User selects NLQ → use associated `RELEVANT_SQL` → sample data → acknowledgment.
- **Scenario 2: New NLQ**:
  - No match in `training_data_<datasource>`.
  - TIA predicts tables/columns (confidence >0.7).
  - Select closest NLQ for context.
  - `prompt_generator` builds prompt with metadata, example SQL.
  - Mock SQL generated → sample data → acknowledgment:
    - **Correct**: Set `IS_LSQL_VALID='Y'`, store `LLM_SQL`, execute full query.
    - **Incorrect**: Set `IS_LSQL_VALID='N'`, store `LLM_SQL`, notify admin, show "Backend team notified."

### 4. New Feature: Sentence Transformer Switching
- **Purpose**: Allow seamless switching between `all-MiniLM-L6-v2` and Open AI’s `text-embedding-3-small` in `table_identifier`.
- **Implementation**:
  - **Configuration**: Add to `app-config/model_config.json`:
    ```json
    {
      "embedding_model": "all-MiniLM-L6-v2",
      "openai_api_key": "<key>",
      "openai_model": "text-embedding-3-small"
    }
    ```
  - **Table Identifier**:
    - Load model based on `embedding_model`.
    - For `text-embedding-3-small`, use Open AI API with `openai_api_key`.
  - **CLI Command**: `set-embedding-model <model>` (admin only).
  - **Training**: Retrain model when switching to ensure compatibility.
- **Validations**:
  - `openai_api_key` valid if `text-embedding-3-small` selected.
  - Model switch triggers retraining prompt.

### 5. Non-Functional Requirements
- **Performance**:
  - NLQ processing: <5s (SQL Server), <10s (S3).
  - Model training: <5min for 1000 NLQs.
- **Scalability**:
  - 100+ tables/schema, 10+ schemas/datasource.
  - 100 concurrent data user queries.
- **Reliability**:
  - 99.9% CLI uptime.
  - S3 retry: 3 attempts, 5s delay.
- **Security**:
  - Encrypt credentials in `db_configurations.json`.
  - IAM roles for S3.
  - Log sensitive operations to `logs/system.log`.
- **Usability**:
  - Intuitive CLI with help documentation.
- **Maintainability**:
  - Modular code, centralized logging.
- **Compatibility**:
  - Python 3.8+.
  - Dependencies: `spacy`, `sentence-transformers`, `openai`, `pandas`, `pyodbc`, `boto3`, `pyarrow`.

### 6. Directory Structure
```
core/
├── orchestrator.py
config/
├── utils.py
├── logging_setup.py
cli/
├── interface.py
storage/
├── db_manager.py
├── storage_manager.py
nlp/
├── nlp_processor.py
tia/
├── table_identifier.py
proga/
├── prompt_generator.py
data_executor.py
app-config/
├── db_configurations.json
├── synonym_config.json
├── model_config.json
├── llm_config.json
├── aws_config.json
├── logging_config.ini
data/
├── <datasource>/
│   ├── metadata_data_<schema>.json
│   ├── metadata_data_<schema>_rich.json
│   ├── synonyms_<schema>.json
│   ├── <datasource>.db
models/
├── model_<datasource>.pkl
temp/
├── query_results/
logs/
├── system.log
```

### 7. Configuration Files
- **db_configurations.json**: Datasource details (SQL Server, S3).
- **synonym_config.json**: `{"synonym_mode": "static|dynamic"}`.
- **model_config.json**: Embedding model selection.
- **llm_config.json**: `{"mock_enabled": true}`.
- **logging_config.ini**: Configures `RotatingFileHandler`.