# Datascriber Documentation

This document provides a comprehensive overview of the Datascriber project, a natural language query (NLQ) to SQL generation tool designed for retail databases, such as BikeStores. Hosted at https://github.com/m-prasad-reddy/Datascriber.git, Datascriber enables users to query data using plain English, leveraging semantic matching and LLMs to produce accurate SQL queries. This documentation covers the architecture, data flow, workflow, and component-level details, with diagrammatic representations to support collaboration, updates, and bug fixes.

## Overview
Datascriber is a modular Python application that processes NLQs to generate SQL queries for SQL Server or S3-based data sources. It uses the Table Identifier Agent (TIA) for table prediction, the Prompt Generator Agent (PROGA) for SQL generation, and the Data Executor (OPDEN) for query execution. The system supports user roles (datauser, admin), S3 file operations, and model switching between `sentence-transformers` and OpenAI embeddings.

## Architecture
Datascriber’s architecture is modular, with components handling specific tasks:
- **CLI Interface**: User interaction via `cli/interface.py`.
- **Table Identifier (TIA)**: Predicts tables/columns (`tia/table_identifier.py`).
- **Prompt Generator (PROGA)**: Generates SQL queries (`proga/prompt_generator.py`).
- **Data Executor (OPDEN)**: Executes queries (`opden/data_executor.py`).
- **Storage**: Manages SQL Server (`storage/db_manager.py`) and S3 (`storage/storage_manager.py`).
- **Configuration**: Handles configs (`config/utils.py`, `config/logging_setup.py`).
- **Mock LLM**: Simulates LLM for testing (`mock_llm.py`).
- **Metadata/Configs**: Defines schema and settings (`metadatafile.json`, `model_config.json`, `llm_config.json`, `db_configurations.json`, `aws_config.json`).

The system uses a layered approach: CLI collects user input, TIA processes NLQs, PROGA generates SQL, OPDEN executes queries, and Storage manages data access. Configuration and logging ensure flexibility and traceability.

### Architecture Diagram
The following diagram illustrates the component interactions:

```mermaid
graph TD
    A[User] -->|NLQ| B[CLI Interface]
    B -->|Authenticate| C[ConfigUtils]
    B -->|Predict Tables| D[TableIdentifier]
    D -->|Load Metadata| C
    D -->|Use Embeddings| E[Model Config]
    D -->|Store Training| F[DBManager]
    B -->|Generate SQL| G[PromptGenerator]
    G -->|Load LLM Config| C
    G -->|Call LLM| H[Mock LLM/OpenAI]
    G -->|Store Rejections| F
    B -->|Execute Query| I[DataExecutor]
    I -->|SQL Server| F
    I -->|S3| J[StorageManager]
    J -->|AWS Config| C
    J -->|Read/Write Files| K[S3 Bucket]
```

## Data Flow
The data flow for an NLQ is:
1. **CLI Input**: User enters an NLQ (e.g., “List products by brand”) via `cli/interface.py`.
2. **TIA Processing**: `TableIdentifier` predicts tables/columns using embeddings and metadata (`metadatafile.json`), producing `tia_output` (tables, columns, DDL, NLQ, conditions, SQL if trained).
3. **PROGA SQL Generation**: `PromptGenerator` constructs a prompt with table/column descriptions and references, calling an LLM to generate SQL.
4. **OPDEN Execution**: `DataExecutor` runs the SQL on SQL Server or S3, returning results to the CLI.
5. **Storage Operations**: `DBManager` stores rejected queries/training data; `StorageManager` reads/writes S3 files.
6. **Output**: Results or errors are displayed in the CLI, logged in `logs/`.

### Data Flow Diagram
The following sequence diagram shows the NLQ processing:

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLIInterface
    participant TIA as TableIdentifier
    participant PROGA as PromptGenerator
    participant OPDEN as DataExecutor
    participant DB as DBManager
    participant S3 as StorageManager
    participant LLM as Mock LLM/OpenAI
    U->>CLI: Enter NLQ
    CLI->>TIA: Predict tables/columns
    TIA->>DB: Load training data
    TIA->>CLI: Return tia_output
    CLI->>PROGA: Generate SQL
    PROGA->>LLM: Call LLM with prompt
    LLM->>PROGA: Return SQL
    PROGA->>DB: Store rejected query (if failed)
    PROGA->>CLI: Return SQL
    CLI->>OPDEN: Execute query
    OPDEN->>DB: Run SQL (SQL Server)
    OPDEN->>S3: Read data (S3)
    OPDEN->>CLI: Return results
    CLI->>U: Display results
```

## Workflow
- **Datauser Role**:
  - Authenticate via CLI (`datauser`, password: `user123`).
  - Select a datasource (e.g., `bikestores`).
  - Submit NLQs, receive SQL results or error messages.
- **Admin Role**:
  - Authenticate (`admin`, password: `admin123`).
  - Train NLQs with tables, columns, and SQL for `TableIdentifier`.
  - View rejected queries or system metrics.
- **System Process**:
  - Initialize configs (`ConfigUtils`).
  - Process NLQs through TIA → PROGA → OPDEN.
  - Log operations (`LoggingSetup`) and store data (`DBManager`, `StorageManager`).

### Workflow Diagram
The following flowchart illustrates user interaction and system steps:

```mermaid
graph TD
    A[Start] --> B[User Login]
    B -->|datauser| C[Select Datasource]
    B -->|admin| D[Admin Menu]
    C --> E[Enter NLQ]
    E --> F[Predict Tables]
    F --> G[Generate SQL]
    G --> H[Execute Query]
    H --> I[Display Results]
    I --> J[Log Output]
    D --> K[Train NLQ]
    D --> L[View Metrics]
    K --> M[Store Training Data]
    L --> N[Display Metrics]
    M --> J
    N --> J
    J --> O[End]
```

## Component-Level Details

### 1. `main.py`
- **Purpose**: Entry point for the CLI interface.
- **Functionality**:
  - Initializes `ConfigUtils`, `LoggingSetup`, `DBManager`, `StorageManager`.
  - Creates `CLIInterface` to handle user interaction.
- **Inputs**: None (command-line execution).
- **Outputs**: Runs CLI loop, processes NLQs.
- **Dependencies**: `cli.interface`, `config.utils`, `config.logging_setup`, `storage.db_manager`, `storage.storage_manager`.
- **Interactions**: Starts the application, delegates to `CLIInterface`.

### 2. `cli/interface.py`
- **Purpose**: Manages user interaction via command-line.
- **Functionality**:
  - Authenticates users (`datauser`, `admin`).
  - Lists datasources from `db_configurations.json`.
  - Collects NLQs, calls `TableIdentifier`, `PromptGenerator`, `DataExecutor`.
  - Supports admin training and metrics viewing.
- **Inputs**: User credentials, NLQs, datasource selection.
- **Outputs**: SQL results, errors, or training confirmation.
- **Dependencies**: `config.utils`, `tia.table_identifier`, `proga.prompt_generator`, `opden.data_executor`, `storage.db_manager`, `storage.storage_manager`.
- **Interactions**: Coordinates TIA, PROGA, OPDEN; logs via `LoggingSetup`.

### 3. `tia/table_identifier.py`
- **Purpose**: Predicts tables/columns for NLQs (TIA-1.2).
- **Functionality**:
  - Uses `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) or `openai` (e.g., `text-embedding-3-small`) embeddings.
  - Loads model type/name from `model_config.json`.
  - Predicts via model-based (`_predict_with_model`) or metadata-based (`_predict_with_metadata`) matching.
  - Uses `description` and `references` from `metadatafile.json` for semantic matching.
  - Supports training (`train_manual`, `train_bulk`) and model generation (`generate_model`).
- **Inputs**: NLQ, user role, datasource.
- **Outputs**: Dictionary with tables, columns, DDL, NLQ, conditions, SQL (if trained).
- **Dependencies**: `sentence_transformers`, `openai`, `numpy`, `config.utils`, `storage.db_manager`, `storage.storage_manager`.
- **Interactions**: Called by `CLIInterface`, uses `ConfigUtils` for metadata, stores data via `DBManager`.

### 4. `proga/prompt_generator.py`
- **Purpose**: Generates SQL queries from TIA output.
- **Functionality**:
  - Constructs prompts with table/column descriptions and references (`metadatafile.json`).
  - Calls LLM (OpenAI or mock) via `requests`, configured by `llm_config.json`.
  - Detects aggregations (`_detect_aggregations`).
  - Stores rejected queries in `rejected_queries` table.
- **Inputs**: TIA output (tables, columns, DDL, NLQ, conditions), user role.
- **Outputs**: SQL query or None if failed.
- **Dependencies**: `requests`, `json`, `config.utils`, `config.logging_setup`, `storage.db_manager`, `storage.storage_manager`.
- **Interactions**: Called by `CLIInterface`, uses `ConfigUtils` for metadata, stores errors via `DBManager`.

### 5. `opden/data_executor.py`
- **Purpose**: Executes SQL queries on SQL Server or S3.
- **Functionality**:
  - Runs SQL via `pyodbc` for SQL Server or processes S3 files (`StorageManager`).
  - Returns query results as DataFrames.
  - Logs execution details.
- **Inputs**: SQL query, datasource.
- **Outputs**: Query results (DataFrame) or error.
- **Dependencies**: `pandas`, `pyodbc`, `config.utils`, `storage.db_manager`, `storage.storage_manager`.
- **Interactions**: Called by `CLIInterface`, uses `StorageManager` for S3, `DBManager` for SQL Server.

### 6. `storage/db_manager.py`
- **Purpose**: Manages SQL Server operations.
- **Functionality**:
  - Connects to SQL Server using `db_configurations.json`.
  - Stores training data (`training_data_<datasource>`), rejected queries (`rejected_queries`), and metrics.
  - Executes SQL queries for `DataExecutor`.
- **Inputs**: Datasource, queries, training data.
- **Outputs**: Query results, stored data.
- **Dependencies**: `pyodbc`, `config.utils`.
- **Interactions**: Used by `TableIdentifier`, `PromptGenerator`, `DataExecutor`.

### 7. `storage/storage_manager.py`
- **Purpose**: Manages S3 file operations.
- **Functionality**:
  - Reads/writes CSV, ORC, Parquet, TXT files using `boto3`.
  - Validates training data against `metadatafile.json` (`description`, `references`).
  - Supports dynamic delimiter from metadata.
- **Inputs**: File paths, training data, datasource.
- **Outputs**: DataFrames, validation results.
- **Dependencies**: `boto3`, `pandas`, `pyarrow`, `config.utils`.
- **Interactions**: Used by `TableIdentifier`, `DataExecutor`, `PromptGenerator`.

### 8. `config/utils.py`
- **Purpose**: Manages configuration loading and directory setup.
- **Functionality**:
  - Loads `db_configurations.json`, `aws_config.json`, `model_config.json`, `llm_config.json`, `metadatafile.json`.
  - Validates metadata structure (`description`, `references`).
  - Creates directories (`config`, `logs`, `data`, `models`).
- **Inputs**: Configuration file paths, datasource, schema.
- **Outputs**: Configuration dictionaries.
- **Dependencies**: `json`, `os`.
- **Interactions**: Used by all components for configuration.

### 9. `config/logging_setup.py`
- **Purpose**: Configures logging for the application.
- **Functionality**:
  - Sets up loggers for `cli`, `tia`, `proga`, `db`, `storage`.
  - Writes logs to `logs/<component>.log`.
- **Inputs**: Logger name.
- **Outputs**: Configured logger.
- **Dependencies**: `logging`, `config.utils`.
- **Interactions**: Used by all components for logging.

### 10. `mock_llm.py`
- **Purpose**: Simulates LLM for testing.
- **Functionality**:
  - Runs a Flask server at `http://localhost:9000/api`.
  - Returns mock SQL responses for prompts.
- **Inputs**: Prompt via HTTP POST.
- **Outputs**: JSON with mock SQL query.
- **Dependencies**: `flask`, `json`.
- **Interactions**: Used by `PromptGenerator` when `mock_enabled: true`.

### 11. `metadatafile.json`
- **Purpose**: Defines BikeStores schema.
- **Functionality**:
  - Lists tables (`brands`, `products`, etc.) with columns, types, descriptions, and references.
  - Specifies delimiter (`\t`) for S3 TXT files.
- **Structure**:
  ```json
  {
      "schema": "default",
      "delimiter": "\t",
      "tables": [
          {
              "name": "products",
              "description": "Stores product information",
              "columns": [
                  {"name": "brand_id", "type": "INT", "description": "Identifier of the brand", "references": {"table": "brands", "column": "brand_id"}}
              ]
          }
      ]
  }
  ```
- **Interactions**: Used by `TableIdentifier`, `StorageManager`, `PromptGenerator`, `ConfigUtils`.

### 12. `model_config.json`
- **Purpose**: Configures TIA embedding model.
- **Functionality**: Specifies `model_type` (`sentence-transformers`, `openai`), `model_name` (e.g., `all-MiniLM-L6-v2`), and `confidence_threshold`.
- **Interactions**: Used by `TableIdentifier`.

### 13. `llm_config.json`
- **Purpose**: Configures PROGA LLM.
- **Functionality**: Specifies `api_key`, `endpoint`, `mock_enabled`, `mock_endpoint`.
- **Interactions**: Used by `PromptGenerator`.

### 14. `db_configurations.json`
- **Purpose**: Defines datasources.
- **Functionality**: Lists SQL Server and S3 datasources with credentials/paths.
- **Interactions**: Used by `CLIInterface`, `DBManager`, `StorageManager`.

### 15. `aws_config.json`
- **Purpose**: Configures AWS S3 access.
- **Functionality**: Specifies `aws_access_key_id`, `aws_secret_access_key`, `region`, `s3_bucket`.
- **Interactions**: Used by `StorageManager`.

## Collaboration Guidelines
- **Repository**: Contribute via https://github.com/m-prasad-reddy/Datascriber.git.
- **Issues**: Report bugs or suggest features using GitHub Issues.
- **Pull Requests**: Submit PRs with clear descriptions and tests.
- **Coding Standards**: Follow PEP 8, include docstrings, and log errors.
- **Testing**: Add unit tests for new features, verify with `SETUP_AND_TESTING.md`.
- **Documentation**: Update this file for architectural changes.

## Future Enhancements
- Add support for more databases (e.g., PostgreSQL).
- Improve aggregation detection in `PromptGenerator`.
- Optimize S3 operations for large datasets.
- Enhance error messages for better user feedback.