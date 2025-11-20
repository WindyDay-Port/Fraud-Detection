import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from utils.inference_pipeline import pipeline
from utils.param_loader import load_all_parameters

app = Flask(__name__)

load_dotenv()
components = load_all_parameters

@app.route("/", methods=["GET"])
def home():
  return {"status": "API is running"}

@app.route("/predict", methods=["POST"])
def predict():
  try:
    data = request.json
    result = pipeline(data)
    return jsonify(result)
  
  except Exception as e:
    return jsonify({
      "error": str(e),
      "trace": traceback.format_exc()
    }), 500
    
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)
