import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from gemini_agent import agent


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "API is running"}, 200

# API endpoint
@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    answer = agent(user_query)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port)


