import sqlite3
import json

def verify_training_data():
    db_path = "data/bikestores/bikestores.db"
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
#         cursor.execute("""
#     INSERT INTO training_data_bikestores (
#         db_source_type, db_name, user_query, related_tables, specific_columns,
#         extracted_values, placeholders, relevant_sql
#     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
# """, (
#     "sqlserver", "bikestores", "Get me sales orders of 2016", "sales.orders",
#     "order_id,order_date", json.dumps({"year": "2016"}), json.dumps(["?"]),
#     "SELECT order_id, order_date FROM sales.orders WHERE YEAR(order_date) = ?"
# ))
#         conn.commit()
        cursor.execute("""
            SELECT db_source_type, db_name, user_query, related_tables,
                   specific_columns, extracted_values, placeholders, relevant_sql
            FROM training_data_bikestores
        """)
        rows = cursor.fetchall()
        print(f"Found {len(rows)} training data entries:")
        for row in rows:
            print("\nTraining Data Entry:")
            print(f"  db_source_type: {row['db_source_type']}")
            print(f"  db_name: {row['db_name']}")
            print(f"  user_query: {row['user_query']}")
            print(f"  related_tables: {row['related_tables']}")
            print(f"  specific_columns: {row['specific_columns']}")
            print(f"  extracted_values: {json.loads(row['extracted_values'])}")
            print(f"  placeholders: {json.loads(row['placeholders'])}")
            print(f"  relevant_sql: {row['relevant_sql']}")
        conn.close()
    except sqlite3.Error as e:
        print(f"Error accessing SQLite database: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

if __name__ == "__main__":
    verify_training_data()