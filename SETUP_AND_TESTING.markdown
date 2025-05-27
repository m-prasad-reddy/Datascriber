# Setup and Testing Instructions for Datascriber

This document provides step-by-step instructions to set up and test the Datascriber project, a natural language query (NLQ) to SQL generation tool for retail databases. The project is hosted at https://github.com/m-prasad-reddy/Datascriber.git.

## Prerequisites
- **Operating System**: Linux, macOS, or Windows.
- **Python**: Version 3.8 or higher.
- **AWS Account**: Access to S3 with valid credentials.
- **SQL Server**: Optional, for database operations (e.g., Microsoft SQL Server).
- **OpenAI API Key**: Optional, for `text-embedding-3-small` model (https://platform.openai.com/api-keys).
- **Git**: Installed to clone the repository.
- **Internet Access**: For dependency installation and API calls.

## Setup Instructions

### 1. Clone the Repository
Clone the Datascriber repository to your local machine:
```bash
git clone https://github.com/m-prasad-reddy/Datascriber.git
cd Datascriber
```

### 2. Set Up a Virtual Environment
Create and activate a Python virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install required Python packages listed in `requirements.txt`:
```bash
pip install --upgrade pip
pip install sentence-transformers openai numpy pandas pyodbc boto3 flask requests
```
Alternatively, create a `requirements.txt` with:
```
sentence-transformers
openai
numpy
pandas
pyodbc
boto3
flask
requests
```
Then run:
```bash
pip install -r requirements.txt
```

### 4. Configure the Project
Create a `config/` directory in the project root and add the following configuration files:

#### a. `db_configurations.json`
Defines SQL Server and S3 datasources:
```json
{
    "databases": [
        {
            "name": "bikestores",
            "type": "SQL Server",
            "server": "your_server_name",
            "database": "bikestores_db",
            "username": "your_username",
            "password": "your_password"
        },
        {
            "name": "bikestores_s3",
            "type": "s3",
            "path": "s3://your-bucket/bikestores/"
        }
    ]
}
```
- Replace `your_server_name`, `your_username`, `your_password` with SQL Server credentials.
- Update `s3://your-bucket/bikestores/` with your S3 bucket path.

#### b. `aws_config.json`
Configures AWS S3 access:
```json
{
    "aws_access_key_id": "your_access_key",
    "aws_secret_access_key": "your_secret_key",
    "region": "us-east-1",
    "s3_bucket": "your-bucket"
}
```
- Obtain keys from AWS IAM (https://aws.amazon.com/iam/).
- Set `region` and `s3_bucket` to match your AWS setup.

#### c. `model_config.json`
Configures the embedding model for `TableIdentifier`:
```json
{
    "model_type": "sentence-transformers",
    "model_name": "all-MiniLM-L6-v2",
    "confidence_threshold": 0.8
}
```
- For OpenAI, use:
```json
{
    "model_type": "openai",
    "model_name": "text-embedding-3-small",
    "confidence_threshold": 0.8
}
```

#### d. `llm_config.json`
Configures the LLM for `PromptGenerator`:
```json
{
    "api_key": "your_openai_api_key",
    "endpoint": "https://api.openai.com/v1",
    "mock_enabled": true,
    "mock_endpoint": "http://localhost:9000/api"
}
```
- Replace `your_openai_api_key` with your OpenAI key or set `mock_enabled` to `true` for mock LLM.
- Ensure `mock_endpoint` matches `mock_llm.py` if used.

#### e. `metadatafile.json`
Defines BikeStores schema with table/column descriptions and references (use the updated version from artifact ID `e376ee4d-bc0a-4770-a24f-6a632b6386e0`).

### 5. Set Up Directory Structure
Ensure the following directories exist (created automatically by `ConfigUtils`):
- `config/`: Stores configuration files.
- `logs/`: Stores log files (e.g., `tia.log`, `proga.log`).
- `data/`: Stores S3 data files or temporary outputs.
- `models/`: Stores trained models (e.g., `model_bikestores.pkl`).

Run the following to verify:
```bash
mkdir -p config logs data models
```

### 6. Start the Mock LLM (Optional)
If using `mock_enabled: true`, start the mock LLM server:
```bash
python mock_llm.py
```
- Ensure it runs on `http://localhost:9000/api` (or update `llm_config.json`).

### 7. Run the Application
Execute the main CLI interface:
```bash
python main.py
```
- Follow prompts to authenticate (`datauser` or `admin`), select a datasource, and enter NLQs (e.g., “List all products by brand”).

## Testing Instructions

### 1. Test CLI Interface
- **Objective**: Verify user authentication and NLQ submission.
- **Steps**:
  1. Run `python main.py`.
  2. Log in as `datauser` (password: `user123`) or `admin` (password: `admin123`).
  3. Select a datasource (e.g., `bikestores`).
  4. Enter an NLQ: “Show all products with price above 500”.
  5. Verify the generated SQL query appears (e.g., `SELECT * FROM products WHERE list_price > 500`).
- **Expected Output**: SQL query or error message for invalid NLQs.
- **Check Logs**: Inspect `logs/cli.log` for authentication and query logs.

### 2. Test Table Identification (TIA)
- **Objective**: Verify `TableIdentifier` predicts correct tables/columns.
- **Steps**:
  1. Use the CLI to enter NLQs:
     - “List products by brand” (expect `products`, `brands` tables).
     - “Show customer orders” (expect `customers`, `orders`).
  2. Verify logs in `logs/tia.log` for predictions (e.g., “Metadata match: table=products”).
  3. Switch `model_config.json` to `openai` and repeat, ensuring API calls succeed.
- **Expected Output**: Correct tables/columns predicted, with referenced tables included (e.g., `brands` via `products.brand_id`).
- **Check**: Ensure `confidence_threshold` (0.8) filters low-confidence predictions.

### 3. Test SQL Generation (PROGA)
- **Objective**: Verify `PromptGenerator` produces accurate SQL.
- **Steps**:
  1. With `mock_enabled: true`, enter an NLQ: “List products by brand”.
  2. Verify the mock LLM returns SQL (e.g., `SELECT p.product_name, b.brand_name FROM products p JOIN brands b ON p.brand_id = b.brand_id`).
  3. Set `mock_enabled: false`, ensure OpenAI generates similar SQL.
  4. Check `logs/proga.log` for prompt and response details.
- **Expected Output**: Valid SQL with JOINs for referenced tables.
- **Check**: Ensure table/column descriptions and references enhance prompts.

### 4. Test S3 Operations
- **Objective**: Verify `StorageManager` reads/writes S3 files.
- **Steps**:
  1. Upload a test CSV to your S3 bucket (e.g., `s3://your-bucket/bikestores/products.csv`).
  2. Run a CLI query accessing S3 data: “List products from S3”.
  3. Verify data is read (`logs/storage.log` shows “Read file from S3”).
  4. Test writing: Modify `opden/data_executor.py` to write results to S3 and verify.
- **Expected Output**: Data read/written successfully, delimiter (`\t`) respected.
- **Check**: Ensure `aws_config.json` credentials are valid.

### 5. Test Error Handling
- **Objective**: Verify rejected queries are stored.
- **Steps**:
  1. Enter an invalid NLQ: “Show invalid table data”.
  2. Check `rejected_queries` table in SQL Server for entry (reason: “Unable to process request”).
  3. Verify `logs/tia.log` or `logs/proga.log` for error details.
- **Expected Output**: Rejected query logged with `TIA_FAILURE` or `NO_LLM_RESPONSE`.

### 6. Test Admin Features
- **Objective**: Verify admin training functionality.
- **Steps**:
  1. Log in as `admin` in the CLI.
  2. Train a new NLQ: “Show products by brand” with tables `products`, `brands` and SQL.
  3. Verify `training_data_bikestores` table in SQL Server is updated.
  4. Re-run the NLQ as `datauser` to use stored SQL.
- **Expected Output**: Training data stored, stored SQL used in predictions.

## Troubleshooting
- **Dependency Issues**: Ensure all packages are installed (`pip list`).
- **AWS Errors**: Verify `aws_config.json` credentials and bucket access.
- **OpenAI Errors**: Check `llm_config.json` API key and network connectivity.
- **SQL Server Errors**: Confirm `db_configurations.json` credentials and server availability.
- **Logs**: Check `logs/` for detailed error messages.

## Notes
- **Security**: Store API keys and credentials securely (e.g., environment variables).
- **Mock LLM**: Use `mock_llm.py` for testing without OpenAI costs.
- **Scalability**: For large datasets, optimize S3 file formats (e.g., Parquet).
- **Updates**: Monitor https://github.com/m-prasad-reddy/Datascriber.git for bug fixes.