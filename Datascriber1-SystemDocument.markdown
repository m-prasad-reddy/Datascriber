# Datascriber 1.1 System Documentation

## 1. Overview
**Datascriber** is a Text-to-SQL system that converts natural language queries (NLQs) into SQL queries, enabling users to interact with SQL Server and S3 datasources without technical schema knowledge. It leverages NLP (`spacy`), sentence embeddings (`all-MiniLM-L6-v2`), and a Table Identifier (TIA) model to predict tables and columns, generate SQL queries via a Language Model (LLM), and execute queries. The system supports two user roles: **Data Users** (business users querying data) and **Admin Users** (technical users configuring and training the system). Datascriber aligns with TIA 1.2 (https://github.com/m-prasad-reddy/TIA-1.2) and is hosted at https://github.com/m-prasad-reddy/Datascriber.

- **Version**: 1.1 (main branch, latest as of June 8, 2025)
- **Checkpoint**: Established May 24, 2025, 04:36 PM IST
- **Current Date/Time**: June 8, 2025, 12:12 AM IST

## 2. User Roles and Responsibilities

### 2.1 Data User (End User)
- **Description**: Non-technical business users querying data using layman NLQs.
- **Actions**:
  - Log in with `login datauser`.
  - Select a datasource (e.g., `select-datasource bikestores_s3`) from `app-config/db_configurations.json`.
  - Enter query mode (`Query>`) to submit NLQs (e.g., "Get me sales orders of 2016").
  - Review sample data (5 rows) and acknowledge accuracy:
    - If accurate, full data is executed and saved as CSV with a downloadable link.
    - If inaccurate, receive "Our backend team is notified. Admin will assist soon." and proceed to next query.
  - Admin is notified of inaccurate queries with details: NLQ, TIA inputs to PROGA, LLM SQL, and error codes (if any).
- **Constraints**:
  - No schema or table knowledge required.
  - Queries proceed only if metadata and rich metadata exist for the datasource, created by Admin.
- **Workflow**:
  1. Login → Select datasource → Enter query mode.
  2. Submit NLQ → TIA validates → LLM generates SQL → OPDEN executes for 5 rows.
  3. Acknowledge sample data:
     - Accurate: Execute full query → CSV output.
     - Inaccurate: Notify admin → Continue querying.
  4. Exit query mode with `exit`.

### 2.2 Admin User
- **Description**: Technical users responsible for system configuration and training.
- **Actions**:
  - Log in with `login admin`.
  - Configure datasources in `app-config/db_configurations.json`.
  - Train TIA model via manual or bulk input:
    - **Manual Training**:
      - Enter NLQ → TIA suggests tables → Confirm or edit.
      - Specify reference columns (1+ from suggested tables, strictly validated).
      - Enter equivalent SQL query → Validate → OPDEN executes for 5 rows.
      - Confirm NLQ, tables, columns, SQL → Save as transaction in `training_data` table.
      - Failed transactions are flushed.
    - **Bulk Training**:
      - Upload `training_data.csv` with NLQs, tables, columns, SQL.
      - System validates for duplicates, prompting to ignore or update.
      - Update TIA model post-upload.
  - Handle failed Data User queries via notifications:
    - Edit tables, columns, SQL → Validate → Save to `training_data` → Retrain.
  - Update TIA model (`models/model_<datasource>.pkl`) incrementally.
  - Back-test with NLQs.
  - Manage synonym modes (`static` or `dynamic`).
- **Constraints**:
  - Requires admin privileges.
  - Responsible for metadata and model creation.
  - No full data query execution required.
- **Workflow**:
  1. Login → Select datasource → Execute commands (`train-model`, `map-failed-query`, etc.).
  2. Manual Training:
     - NLQ → Table suggestion → Column input → SQL input → Validation → Save.
  3. Bulk Training:
     - Upload `training_data.csv` → Validate → Update model.
  4. Failed Query Handling:
     - Notification → Edit mappings → Validate → Save → Retrain.
  5. Exit with `exit`.

## 3. Component-Wise Functionalities

### 3.1 CLI (`cli/interface.py`)
- **Functionality**:
  - Provides a command-line interface for user interactions.
  - Parses commands using `argparse`.
  - Supports query mode for Data Users and admin commands.
- **Commands**:
  - `login <username>`: Authenticate as `datauser` or `admin`.
  - `select-datasource <name>`: Select datasource from `db_configurations.json`.
  - `query <nlq>`: Submit NLQ (Data Users in query mode, Admins for testing).
  - `train-model`: Train TIA model (Admin).
  - `refresh-metadata`: Update metadata files (Admin).
  - `set-synonym-mode <mode>`: Set synonym mode (Admin, `static` or `dynamic`).
  - `list-datasources`, `list-schemas`: List available datasources/schemas.
  - `map-failed-query`: Handle failed queries (Admin).
  - `exit`: Exit query mode or CLI.
- **Inputs**:
  - User commands, NLQs (max 500 characters).
- **Outputs**:
  - Query results: Tables, columns, sample data (5 rows), CSV path.
  - Notifications (e.g., "Query cannot be processed. Notified admin.").
  - Error or confirmation messages.
- **Validations**:
  - Username: `admin` or `datauser`.
  - Datasource: Exists in `db_configurations.json`.
  - NLQ: Non-empty, max 500 characters.
- **Dependencies**:
  - **Upstream**: User input, `config/utils.py` (configuration loading).
  - **Downstream**: `core/orchestrator.py` (query processing, training).

### 3.2 Orchestrator (`core/orchestrator.py`)
- **Functionality**:
  - Coordinates NLQ processing, training, and metadata management.
  - Integrates components: NLP, TIA, Prompt Generator, Data Executor.
- **Inputs**:
  - NLQ, datasource name, schema (optional for Admin).
  - Training data (NLQ, tables, columns, SQL).
- **Outputs**:
  - Query results: `{tables, columns, sample_data, csv_path}`.
  - Failure: Log to `rejected_queries_<datasource>`, notify Admin.
- **Validations**:
  - Metadata files exist (`data/<datasource>/metadata_data_<schema>.json`, `metadata_data_<schema>_rich.json`).
  - TIA model exists (`models/model_<datasource>.pkl`) or trainable.
- **Dependencies**:
  - **Upstream**: `cli/interface.py`, `config/utils.py`, `storage/db_manager.py`, `storage/storage_manager.py`.
  - **Downstream**: `nlp/nlp_processor.py`, `tia/tia.py`, `prompt_generator.py`, `data_executor.py`.

### 3.3 NLP Processor (`nlp/nlp_processor.py`)
- **Functionality**:
  - Extracts entities and values from NLQs using `spacy` (`en_core_web_sm`).
  - Maps terms to synonyms using `data/<datasource>/synonyms_<schema>.json`.
- **Inputs**:
  - NLQ (e.g., "Get me sales orders of 2016").
  - Metadata (`metadata_data_<schema>_rich.json`).
- **Outputs**:
  - `extracted_values`: JSON (e.g., `{"order_date": "2016"}`).
  - `placeholders`: JSON (e.g., `{"order_date": "?"}`).
- **Validations**:
  - `spacy` model loaded.
  - NLQ non-empty.
- **Dependencies**:
  - **Upstream**: `core/orchestrator.py`.
  - **Downstream**: `tia/tia.py`, `prompt_generator.py`.

### 3.4 TIA (`tia/tia.py`)
- **Functionality**:
  - Predicts tables and columns using `all-MiniLM-L6-v2` embeddings.
  - Supports training via manual or bulk input.
- **Inputs**:
  - NLQ, `extracted_values`, metadata.
  - Training data (NLQ, tables, columns, SQL).
- **Outputs**:
  - Predicted tables/columns (confidence >0.7).
  - Failure status if confidence <0.7.
- **Validations**:
  - Model exists or trainable (>10 samples).
- **Dependencies**:
  - **Upstream**: `core/orchestrator.py`, `nlp/nlp_processor.py`.
  - **Downstream**: `prompt_generator.py`.

### 3.5 Prompt Generator (`prompt_generator.py`)
- **Functionality**:
  - Generates SQL query templates using an LLM (mock or real).
  - Uses SQLite-compatible SQL (`strftime('%Y', order_date)`) for S3.
- **Inputs**:
  - NLQ, predicted tables/columns, `extracted_values`, `placeholders`.
- **Outputs**:
  - SQL query (e.g., `SELECT order_date FROM orders WHERE strftime('%Y', order_date) = '2016'`).
- **Validations**:
  - Valid LLM configuration (`app-config/llm_config.json`).
- **Dependencies**:
  - **Upstream**: `core/orchestrator.py`, `tia/tia.py`.
  - **Downstream**: `data_executor.py`.

### 3.6 Data Executor (`data_executor.py`)
- **Functionality**:
  - Executes SQL queries on SQL Server (`pyodbc`) or S3 (`pandasql`, `pyarrow`).
  - Saves results as CSV/JSON in `temp/query_results/`.
  - Bypasses `mock_llm.py` for S3 queries with `strftime` in mock mode.
- **Inputs**:
  - SQL query, `extracted_values`.
- **Outputs**:
  - Pandas DataFrame (sample: 5 rows, full data if acknowledged).
  - CSV/JSON paths (e.g., `output_<timestamp>.csv`).
- **Validations**:
  - Valid database connection (SQL Server or S3).
- **Dependencies**:
  - **Upstream**: `core/orchestrator.py`, `prompt_generator.py`.
  - **Downstream**: `temp/query_results/`.

### 3.7 Storage Manager (`storage/storage_manager.py`)
- **Functionality**:
  - Manages S3 operations: metadata generation, data reading.
  - Supports `csv`, `parquet`, `orc`, `txt` files.
- **Inputs**:
  - Datasource configuration (`bucket_name`, `database`, `region`, `schemas`).
  - Schema name.
- **Outputs**:
  - Metadata (`data/<datasource>/metadata_data_<schema>.json`, `metadata_data_<schema>_rich.json`).
  - Table data (pandas DataFrame).
- **Validations**:
  - S3 bucket accessible via `boto3`.
- **Dependencies**:
  - **Upstream**: `core/orchestrator.py`, `config/utils.py`.
  - **Downstream**: `core/orchestrator.py`, `data_executor.py`.

### 3.8 DB Manager (`storage/db_manager.py`)
- **Functionality**:
  - Manages SQL Server operations: metadata generation, data reading.
- **Inputs**:
  - Datasource configuration (`host`, `port`, `database`, `username`, `password`, `schemas`).
  - Schema name.
- **Outputs**:
  - Metadata (`data/<datasource>/metadata_data_<schema>.json`, `metadata_data_<schema>_rich.json`).
  - Table data (pandas DataFrame).
- **Validations**:
  - SQL Server connection via `pyodbc`.
- **Dependencies**:
  - **Upstream**: `core/orchestrator.py`.
  - **Downstream**: `core/orchestrator.py`, `data_executor.py`.

## 4. Interactions and Data Flow

### 4.1 Data User Query Flow
1. **CLI**: User submits `login datauser`, `select-datasource bikestores_s3`, enters query mode, submits NLQ ("Get me sales orders of 2016").
2. **Orchestrator**: Validates metadata, passes NLQ to NLP Processor.
3. **NLP Processor**: Extracts entities (`DATE: 2016`), maps to `order_date`, produces `extracted_values` and `placeholders`.
4. **TIA**:
   - Scenario 1: Matches NLQ to existing training data → Uses stored SQL.
   - Scenario 2: Predicts tables (`orders`) and columns, uses nearest NLQ’s SQL as example.
5. **Prompt Generator**: Generates SQL (`SELECT order_date FROM orders WHERE strftime('%Y', order_date) = '2016'`).
6. **Data Executor**:
   - Executes SQL for 5 rows using `pandasql` on S3 data (`s3://bike-stores-bucket/data-files/orders.csv`).
   - Displays sample data, awaits user acknowledgment.
   - If accurate: Executes full query, saves to CSV/JSON.
   - If inaccurate: Logs to `rejected_queries_bikestores_s3`, notifies Admin, stores LLM SQL in `training_data` with `IS_LSQL_VALID='N'`.
7. **CLI**: Displays results or notification.

### 4.2 Admin Manual Training Flow
1. **CLI**: Admin submits `login admin`, `select-datasource bikestores_s3`, `train-model`.
2. **Orchestrator**: Initiates training transaction.
3. **CLI**: Prompts for NLQ (e.g., "Get me sales orders of 2016").
4. **TIA**: Suggests tables (`orders`) → Admin confirms or edits.
5. **CLI**: Prompts for reference columns (e.g., `order_id`, `order_date`) → Validates (from suggested tables).
6. **CLI**: Prompts for SQL → Admin enters `SELECT order_id, order_date FROM orders WHERE strftime('%Y', order_date) = '2016'`.
7. **Data Executor**: Executes SQL for 5 rows → Displays results.
8. **CLI**: Admin confirms NLQ, tables, columns, SQL → Saves to `training_data` table.
9. **TIA**: Updates model (`models/model_bikestores_s3.pkl`).
10. **Orchestrator**: Commits transaction or flushes if failed.

### 4.3 Admin Bulk Training Flow
1. **CLI**: Admin submits `train-model`, provides `training_data.csv` path.
2. **Orchestrator**: Loads CSV, validates entries (`SCENARIO_ID`, `DB_CONFIG_TYPE`, `DB_NAME`, `USER_QUERY`, `RELATED_TABLES`, `SPECIFIC_COLUMNS`, `RELEVANT_SQL` non-null).
3. **TIA**: Checks for duplicate NLQs, prompts Admin to ignore or update.
4. **Orchestrator**: Saves valid entries to `training_data` table.
5. **TIA**: Updates model (`models/model_<datasource>.pkl`).
6. **CLI**: Confirms completion, allows back-testing with NLQs.

### 4.4 Admin Failed Query Handling
1. **Orchestrator**: Notifies Admin via `rejected_queries_<datasource>`.
2. **CLI**: Admin runs `map-failed-query`, views NLQ, TIA inputs, LLM SQL, error (if any).
3. **CLI**: Admin edits tables, columns, SQL → Validates → Executes for 5 rows.
4. **Orchestrator**: Saves to `training_data` table.
5. **TIA**: Updates model.
6. **CLI**: Confirms resolution, reprocesses query.

## 5. Workflows

### 5.1 Data User Workflow
```plaintext
Start
  ↓
Login (datauser)
  ↓
Select Datasource (bikestores_s3)
  ↓
Enter Query Mode
  ↓
Submit NLQ ("Get me sales orders of 2016")
  ↓
TIA Validates
  ↓
[Scenario 1: Existing NLQ] → Use Stored SQL
  ↓
[Scenario 2: New NLQ] → TIA Predicts Tables → Prompt Generator → LLM SQL
  ↓
Data Executor: Execute SQL (5 rows)
  ↓
Display Sample Data
  ↓
User Acknowledges
  ↓
[Accurate] → Execute Full Query → Save CSV → Display Link
  ↓
[Inaccurate] → Notify Admin → Log to rejected_queries → Continue Querying
  ↓
Exit Query Mode
  ↓
End
```

### 5.2 Admin Manual Training Workflow
```plaintext
Start
  ↓
Login (admin)
  ↓
Select Datasource
  ↓
Run train-model
  ↓
Enter NLQ
  ↓
TIA Suggests Tables
  ↓
Admin Confirms/Edits Tables
  ↓
Enter Reference Columns (Validated)
  ↓
Enter SQL
  ↓
Data Executor: Execute SQL (5 rows)
  ↓
Admin Confirms NLQ, Tables, Columns, SQL
  ↓
Save to training_data Table
  ↓
Update TIA Model
  ↓
Commit Transaction
  ↓
End
```

### 5.3 Admin Bulk Training Workflow
```plaintext
Start
  ↓
Login (admin)
  ↓
Select Datasource
  ↓
Run train-model
  ↓
Provide training_data.csv Path
  ↓
Validate Entries (Non-null Fields)
  ↓
Check Duplicates → Prompt Ignore/Update
  ↓
Save to training_data Table
  ↓
Update TIA Model
  ↓
Back-Test with NLQs
  ↓
End
```

## 6. Architecture

### 6.1 High-Level Architecture
```plaintext
+-------------------+
|       CLI         |
| (interface.py)    |
+-------------------+
          ↓
+-------------------+
|   Orchestrator    |
| (orchestrator.py) |
+-------------------+
  ↓      ↓      ↓      ↓
+-------+-------+-------+-------+
|  NLP  |  TIA  | Prompt|  Data |
|Processor(tia.py)Generator|Executor|
|(nlp_processor.py) |(prompt_generator.py)|(data_executor.py)|
+-------+-------+-------+-------+
  ↓      ↓      ↓      ↓
+-------+-------+-------+-------+
| Storage Manager|  DB Manager   |
| (storage_manager.py) | (db_manager.py) |
+---------------+---------------+
  ↓              ↓
+---------------+---------------+
|    S3         |  SQL Server   |
| (boto3, pyarrow) | (pyodbc)      |
+---------------+---------------+
```

### 6.2 Component Interactions
- **CLI ↔ Orchestrator**: CLI parses user commands/NLQs, passes to Orchestrator for processing.
- **Orchestrator ↔ NLP/TIA/Prompt/Data Executor**: Orchestrator coordinates pipeline, passing NLQ and metadata through components.
- **NLP ↔ TIA**: NLP extracts entities, TIA predicts tables/columns.
- **TIA ↔ Prompt Generator**: TIA provides predictions, Prompt Generator builds SQL.
- **Prompt Generator ↔ Data Executor**: Prompt Generator supplies SQL, Data Executor executes and saves results.
- **Storage/DB Manager ↔ Orchestrator/Data Executor**: Provide metadata and data access.

## 7. Sequence Diagrams

### 7.1 Data User Query (New NLQ)
```plaintext
actor DataUser
participant CLI
participant Orchestrator
participant NLPProcessor
participant TIA
participant PromptGenerator
participant DataExecutor
participant StorageManager
participant S3

DataUser -> CLI: login datauser
CLI -> Orchestrator: authenticate
Orchestrator -> CLI: success
DataUser -> CLI: select-datasource bikestores_s3
CLI -> Orchestrator: select datasource
Orchestrator -> StorageManager: validate metadata
StorageManager -> S3: check bucket
StorageManager -> Orchestrator: metadata
Orchestrator -> CLI: enter query mode
DataUser -> CLI: Query> Get me sales orders of 2016
CLI -> Orchestrator: process NLQ
Orchestrator -> NLPProcessor: extract entities
NLPProcessor -> Orchestrator: {"DATE": "2016"}
Orchestrator -> TIA: predict tables/columns
TIA -> Orchestrator: {tables: ["orders"], columns: ["order_date"]}
Orchestrator -> PromptGenerator: generate SQL
PromptGenerator -> Orchestrator: SELECT order_date FROM orders WHERE strftime('%Y', order_date) = '2016'
Orchestrator -> DataExecutor: execute query (5 rows)
DataExecutor -> StorageManager: load data
StorageManager -> S3: read orders.csv
StorageManager -> DataExecutor: pandas DataFrame
DataExecutor -> Orchestrator: sample data, CSV path
Orchestrator -> CLI: display sample data
DataUser -> CLI: acknowledge accurate
CLI -> Orchestrator: execute full query
Orchestrator -> DataExecutor: execute full query
DataExecutor -> Orchestrator: CSV path
Orchestrator -> CLI: display CSV link
```

### 7.2 Admin Manual Training
```plaintext
actor Admin
participant CLI
participant Orchestrator
participant TIA
participant DataExecutor
participant StorageManager
participant training_data

Admin -> CLI: login admin
CLI -> Orchestrator: authenticate
Orchestrator -> CLI: success
Admin -> CLI: select-datasource bikestores_s3
CLI -> Orchestrator: select datasource
Orchestrator -> CLI: success
Admin -> CLI: train-model
CLI -> Orchestrator: start training
Orchestrator -> CLI: prompt for NLQ
Admin -> CLI: Get me sales orders of 2016
CLI -> Orchestrator: NLQ
Orchestrator -> TIA: suggest tables
TIA -> Orchestrator: ["orders"]
Orchestrator -> CLI: display tables
Admin -> CLI: confirm tables
CLI -> Orchestrator: tables confirmed
Orchestrator -> CLI: prompt for columns
Admin -> CLI: order_id, order_date
CLI -> Orchestrator: validate columns
Orchestrator -> CLI: prompt for SQL
Admin -> CLI: SELECT order_id, order_date FROM orders WHERE strftime('%Y', order_date) = '2016'
CLI -> Orchestrator: SQL
Orchestrator -> DataExecutor: execute SQL (5 rows)
DataExecutor -> StorageManager: load data
StorageManager -> DataExecutor: data
DataExecutor -> Orchestrator: sample results
Orchestrator -> CLI: display results
Admin -> CLI: confirm NLQ, tables, columns, SQL
CLI -> Orchestrator: save
Orchestrator -> training_data: store transaction
Orchestrator -> TIA: update model
TIA -> Orchestrator: model updated
Orchestrator -> CLI: training complete
```

## 8. Git Repository
- **URL**: https://github.com/m-prasad-reddy/Datascriber
- **Branch**: main
- **Structure**:
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
  ├── tia.py
  app-config/
  ├── db_configurations.json
  ├── synonym_config.json
  ├── model_config.json
  ├── llm_config.json
  ├── aws_config.json
  ├── logging_config.json
  data/
  ├── bikestores_s3/
  │   ├── metadata_data_default.json
  │   ├── metadata_data_default_rich.json
  │   ├── synonyms_default.json
  models/
  ├── model_bikestores_s3.pkl
  temp/
  ├── query_results/
  logs/
  ├── system.log
  ├── mock_llm.log
  main.py
  mock_llm.py
  ```
- **Key Files**:
  - `main.py`: CLI entry point.
  - `opdata/data_executor.py`: Query execution, updated to bypass `mock_llm.py` for S3.
  - `proga/prompt_generator.py`: SQL generation, SQLite-compatible for S3.
  - `core/orchestrator.py`: Pipeline orchestration.
  - `mock_llm.py`: Mock LLM server, may need update.
  - `app-config/db_configurations.json`: Datasource settings.
  - `app-config/llm_config.json`: LLM settings (`mock_enabled=True`).
- **Verification**:
  ```bash
  git clone https://github.com/m-prasad-reddy/Datasetorcher.git
  cd Datascriber
  grep -A 5 "Using mock SQL from prompt" opdata/data_executor.py
  python -m py_compile main.py opdata/data_executor.py proga/prompt_generator.py core/orchestrator.py mock_llm.py
  ```

## 9. Planned Enhancements
Based on `datascriber_full_requirements.markdown` and prior discussions, the following enhancements are planned:

### 9.1 Enhancement 1: Support Complex NLQs (Joins, Aggregations)
- **Description**: Enable NLQs like "Get sales orders with customer details for 2016" (joins) or "Count orders by month for 2017" (aggregations).
- **Requirements**:
  - TIA to predict multiple tables for joins.
  - Prompt Generator to create complex SQL (e.g., `SELECT o.order_id, c.customer_name FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE strftime('%Y', o.order_date) = '2016'`).
  - Data Executor to validate and execute complex queries.
- **Implementation Plan**:
  - Update `tia/tia.py` to predict multiple tables with join conditions.
  - Enhance `prompt_generator.py` to include join and aggregation templates.
  - Test with `training_data.csv` including join/aggregation scenarios.
- **Challenges**:
  - Ensuring TIA accuracy for multi-table predictions.
  - Handling performance with `pandasql` for joins.
- **Dependencies**:
  - `pandasql` or switch to DuckDB for better performance.

### 9.2 Enhancement 2: Optimize S3 Query Performance
- **Description**: Replace `pandasql` with DuckDB for faster S3 query execution.
- **Requirements**:
  - Execute queries <5 seconds for 10,000 rows.
  - Maintain compatibility with existing schemas.
- **Implementation Plan**:
  - Add `duckdb` dependency (`pip install duckdb`).
  - Update `data_executor.py` to use DuckDB engine:
    ```python
    import duckdb
    result_df = duckdb.query(sql_query).to_df()
    ```
  - Test with large datasets (e.g., 10,000+ rows).
- **Challenges**:
  - Ensuring DuckDB compatibility with `pyarrow` datasets.
  - Migrating existing queries to DuckDB syntax.
- **Dependencies**:
  - `duckdb`, `pyarrow`.

### 9.3 Enhancement 3: Enhance Mock LLM
- **Description**: Update `mock_llm.py` to support complex queries and S3-specific SQL.
- **Requirements**:
  - Handle joins, aggregations, and `strftime` for S3.
  - Remove bypass logic in `data_executor.py` if robust.
- **Implementation Plan**:
  - Apply updated `mock_llm.py` (artifact ID `a2e7fef3-8553-4b11-b4c8-09018b79a161`):
    ```python
    if is_s3:
        sql_query = f"SELECT order_id, order_date FROM orders WHERE strftime('%Y', order_date) = '{year}'"
    ```
  - Add logic for joins/aggregations.
  - Test with varied NLQs.
- **Challenges**:
  - Ensuring robust year extraction and query parsing.
- **Dependencies**:
  - `flask`, `requests`.

### 9.4 Enhancement 4: Add Notification System
- **Description**: Implement a robust notification system for Admin (e.g., email, in-CLI alerts).
- **Requirements**:
  - Notify Admin of failed queries with NLQ, TIA inputs, LLM SQL, errors.
  - Support email or file-based notifications.
- **Implementation Plan**:
  - Add `smtplib` for email notifications in `orchestrator.py`:
    ```python
    import smtplib
    def notify_admin(nlq, tia_inputs, llm_sql, error):
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('admin@example.com', 'password')
            server.sendmail('admin@example.com', 'admin@example.com', f"Failed NLQ: {nlq}, Error: {error}")
    ```
  - Log notifications to `logs/notifications.log`.
- **Challenges**:
  - Securing email credentials.
  - Handling notification failures.
- **Dependencies**:
  - `smtplib`.

### 9.5 Enhancement 5: Support Additional Datasources
- **Description**: Add support for other datasources (e.g., PostgreSQL, Snowflake).
- **Requirements**:
  - Extend `db_configurations.json` for new datasource types.
  - Update `db_manager.py` and `data_executor.py` for connectivity.
- **Implementation Plan**:
  - Add PostgreSQL support using `psycopg2`:
    ```python
    import psycopg2
    conn = psycopg2.connect(host=..., dbname=..., user=..., password=...)
    ```
  - Update `db_manager.py` to handle new connection types.
  - Test with PostgreSQL instance.
- **Challenges**:
  - Ensuring compatibility with existing pipeline.
  - Managing multiple connection types.
- **Dependencies**:
  - `psycopg2` or other drivers.

## 10. Clarification Questions
To ensure accurate implementation of enhancements, please clarify:

1. **Complex NLQs**:
   - What specific NLQ types are prioritized (e.g., joins, aggregations, subqueries)?
   - Are there sample NLQs for testing?
   - Should `pandasql` be retained, or is DuckDB preferred?

2. **Performance Optimization**:
   - What is the target query execution time for S3?
   - What is the expected dataset size (e.g., 10,000 rows, 1M rows)?
   - Any constraints on adding new dependencies (e.g., `duckdb`)?

3. **Mock LLM**:
   - Should `mock_llm.py` handle all query types, removing the `data_executor.py` bypass?
   - Are there specific query patterns to prioritize?
   - Is a real LLM integration planned?

4. **Notification System**:
   - Preferred notification method (email, CLI, file-based)?
   - Email provider and security requirements?
   - Frequency and volume of notifications expected?

5. **Additional Datasources**:
   - Which datasources are targeted (e.g., PostgreSQL, Snowflake)?
   - Connection details available for testing?
   - Schema complexity (e.g., number of tables, schemas)?

6. **Training Data**:
   - Sample `training_data.csv` available for reference?
   - Expected volume of training scenarios (e.g., 1000 NLQs)?
   - Frequency of bulk training updates?

7. **General**:
   - Priority order of enhancements?
   - Any performance or compatibility constraints?
   - Testing environment details (e.g., AWS credentials, SQL Server access)?

## 11. Getting Started for Features and Troubleshooting

### 11.1 Adding New Features
1. **Clone Repository**:
   ```bash
   git clone https://github.com/m-prasad-reddy/Datascriber.git
   cd Datascriber
   ```
2. **Install Dependencies**:
   ```bash
   pip install spacy sentence-transformers pandas numpy pyodbc boto3 pyarrow requests s3fs aiobotocore pandasql
   python -m spacy download en_core_web_sm
   ```
3. **Configure**:
   - Update `app-config/db_configurations.json` with datasource details.
   - Set AWS credentials in `app-config/aws_config.json`.
   - Verify `app-config/llm_config.json` (`mock_enabled=True`).
4. **Test Existing Functionality**:
   ```bash
   python main.py
   login datauser
   select-datasource bikestores_s3
   Query> Get me sales orders of 2016
   ```
5. **Implement Features**:
   - Modify relevant components (e.g., `tia/tia.py` for joins).
   - Add tests in `tests/` directory.
   - Commit to feature branch:
     ```bash
     git checkout -b feature/<name>
     git commit -m "Add <feature>"
     git push origin feature/<name>
     ```
6. **Validate**:
   - Check logs: `tail -f logs/system.log`.
   - Verify outputs: `head temp/query_results/output_*.csv`.

### 11.2 Troubleshooting
1. **S3 Query Errors**:
   - Check `logs/system.log` for `sqlite3.OperationalError`.
   - Verify `data_executor.py` bypass logic:
     ```bash
     grep -A 5 "Using mock SQL from prompt" opdata/data_executor.py
     ```
   - Update `mock_llm.py` if needed.
2. **TIA Prediction Failures**:
   - Ensure model exists: `ls models/model_bikestores_s3.pkl`.
   - Retrain with `train-model` command.
   - Check `training_data` table for sufficient scenarios.
3. **Connection Issues**:
   - SQL Server: Verify `pyodbc` connection in `db_manager.py`.
   - S3: Check AWS credentials and bucket access:
     ```bash
     aws s3 ls s3://bike-stores-bucket/data-files/
     ```
4. **Log Analysis**:
   ```bash
   tail -f logs/system.log logs/mock_llm.log
   ```
5. **Debug**:
   - Add debug logs in components (e.g., `logger.debug`).
   - Share logs, `orders.csv` sample, and configuration files.

## 12. Non-Functional Details
- **Performance**:
  - NLQ processing: <5s (SQL Server), <10s (S3).
  - Metadata generation: <1min/schema.
  - Model training: <5min/1000 NLQs.
- **Scalability**:
  - 100+ tables/schema, 10+ schemas/datasource.
  - 100 concurrent queries.
- **Reliability**:
  - 99.9% CLI uptime.
  - S3 retries: 3 attempts, 5s delay.
- **Security**:
  - Encrypt SQL Server credentials (Fernet planned).
  - Use AWS IAM roles for S3.
  - Log sensitive operations to `logs/system.log`.
- **Usability**:
  - Intuitive CLI with clear errors.
  - Minimal commands for Data Users.
- **Maintainability**:
  - Modular code with logging (`logging_setup.py`).
  - Configuration-driven (`app-config/`).
- **Compatibility**:
  - Python 3.8+.
  - Dependencies: See `requirements.txt`.

## 13. Conclusion
This document provides a comprehensive overview of Datascriber 1.1, detailing user roles, component functionalities, workflows, architecture, and enhancement plans. It serves as a guide for adding features, troubleshooting, and continuing development. Clarifications on enhancements will drive the next steps. Reference the repository (https://github.com/m-prasad-reddy/Datascriber) for the latest code and test with provided instructions to ensure stability.