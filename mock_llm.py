import re
import json
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_mock_llm():
    """Set up the mock LLM server."""
    @app.route('/api', methods=['POST'])
    def mock_llm():
        try:
            data = request.get_json()
            if not data or 'prompt' not in data:
                logging.error("Invalid request: No prompt provided")
                return jsonify({"error": "No prompt provided"}), 400

            prompt = data['prompt']
            logging.debug(f"Received prompt: {prompt}")

            # Check if prompt contains S3 datasource indicator
            is_s3 = "s3" in prompt.lower() or "bikestores_s3" in prompt.lower()

            # Extract year from prompt
            year_match = re.search(r"['\"](20\d{2})['\"']", prompt)
            if year_match:
                year = year_match.group(1)
                # Generate SQLite-compatible SQL for S3, standard SQL for others
                if is_s3:
                    sql_query = f"SELECT order_id, order_date FROM orders WHERE strftime('%Y', order_date) = '{year}'"
                else:
                    sql_query = f"SELECT order_id, order_date FROM orders WHERE YEAR(order_date) = {year}"
                logging.info(f"Returning mock SQL: {sql_query}")
                return jsonify({"sql_query": sql_query})
            
            # Fallback for specific date cases
            if "DATE" in prompt.upper() and "'2019'" in prompt:
                if is_s3:
                    sql_query = "SELECT order_id, order_date FROM orders WHERE strftime('%Y', order_date) = '2019'"
                else:
                    sql_query = "SELECT order_id, order_date FROM orders WHERE YEAR(order_date) = 2019"
                logging.info(f"Returning mock SQL: {sql_query}")
                return jsonify({"sql_query": sql_query})

            # Default response
            logging.warning("No specific year found in prompt, returning default response")
            return jsonify({"sql_query": "SELECT order_id, order_date FROM orders LIMIT 10"})

        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    setup_mock_llm()
    app.run(host='0.0.0.0', port=9000, debug=False)