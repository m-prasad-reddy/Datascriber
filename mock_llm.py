from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mock_llm.log")
    ]
)
logger = logging.getLogger(__name__)

@app.route("/api", methods=["GET"])
def health_check():
    """Health check endpoint for the mock LLM server.

    Returns:
        JSON response indicating the server is running.
    """
    logger.info("Health check requested")
    return jsonify({"status": "Mock LLM server is running"}), 200

@app.route("/api", methods=["POST"])
def mock_llm():
    """Mock LLM API endpoint that returns SQL from the prompt.

    Expects a JSON payload with a 'prompt' field containing the LLM prompt.
    Extracts the SQL from 'Example SQL:' or returns a default SQL query if not found.

    Returns:
        JSON response with the extracted or default SQL query.
    """
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            logger.error("Invalid request: missing 'prompt' field")
            return jsonify({"error": "Missing 'prompt' field"}), 400

        prompt = data["prompt"]
        logger.debug(f"Received prompt: {prompt[:100]}...")

        # Extract SQL from the prompt
        example_sql_start = prompt.find("Example SQL:")
        if example_sql_start == -1:
            logger.warning("Example SQL not found in prompt, returning default SQL")
            default_sql = "SELECT order_id, order_date FROM orders WHERE YEAR(order_date) = 2017"
            return jsonify({"sql_query": default_sql})

        # Find the SQL query after "Example SQL:"
        sql_start = example_sql_start + len("Example SQL:")
        sql_query = prompt[sql_start:].strip().split("\n")[0].strip()

        if not sql_query:
            logger.warning("Empty SQL query extracted, returning default SQL")
            default_sql = "SELECT order_id, order_date FROM orders WHERE YEAR(order_date) = 2017"
            return jsonify({"sql_query": default_sql})

        logger.info(f"Returning mock SQL: {sql_query}")
        return jsonify({"sql_query": sql_query})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting mock LLM server on http://localhost:9000/api")
    app.run(host="0.0.0.0", port=9000, debug=True)