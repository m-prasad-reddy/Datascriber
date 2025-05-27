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

@app.route("/api", methods=["POST"])
def mock_llm():
    """Mock LLM API endpoint that returns example SQL from the prompt.

    Expects a JSON payload with a 'prompt' field containing the LLM prompt.
    Extracts the example SQL from the prompt and returns it as the response.

    Returns:
        JSON response with the extracted SQL query or an error message.
    """
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            logger.error("Invalid request: missing 'prompt' field")
            return jsonify({"error": "Missing 'prompt' field"}), 400

        prompt = data["prompt"]
        logger.debug(f"Received prompt: {prompt[:100]}...")

        # Extract example SQL from the prompt
        example_sql_start = prompt.find("Example SQL:")
        if example_sql_start == -1:
            logger.error("Example SQL not found in prompt")
            return jsonify({"error": "Example SQL not found in prompt"}), 400

        # Find the SQL query after "Example SQL:"
        sql_start = example_sql_start + len("Example SQL:")
        sql_query = prompt[sql_start:].strip().split("\n")[0].strip()

        logger.info(f"Returning mock SQL: {sql_query}")
        return jsonify({
            "choices": [{"text": sql_query}]
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting mock LLM server on http://localhost:5000/api")
    app.run(host="0.0.0.0", port=5000, debug=True)