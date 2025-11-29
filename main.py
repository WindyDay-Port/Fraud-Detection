import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from utils.inference_pipeline import pipeline
from utils.param_loader import load_all_parameters

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

# Load .env
load_dotenv()

# Load all preprocessors + models
components = load_all_parameters()   # <-- FIXED

@app.route("/", methods=["GET"])
def home():
    return {"status": "API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Allow single object or list of objects
        if isinstance(data, dict):
            data = [data]

        results = []
        for record in data:
            processed = pipeline(record)   # gọi inference
            results.append(processed)

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
