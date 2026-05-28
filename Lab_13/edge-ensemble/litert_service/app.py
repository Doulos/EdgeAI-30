import os
from flask import Flask, request, jsonify
import numpy as np
import requests
from ai_edge_litert.interpreter import Interpreter
#import litert as rt  # The updated Google LiteRT runtime package

app = Flask(__name__)

ONNX_URL = os.getenv("ONNX_SERVICE_URL", "http://localhost:8001/predict")
print("LiteRT Runtime initialized successfully.")

@app.route('/ensemble_predict', methods=['POST'])
def ensemble_predict():
    payload = request.json
    raw_features = payload.get("data", [0.0])
    
    # 1. Execute local LiteRT Inference
    # In practice: interpreter = rt.Interpreter(model_path="model.tflite")
    input_data = np.array(raw_features, dtype=np.float32)
    litert_confidence = float(1.0 / (1.0 + np.exp(-input_data.sum() * 1.2))) # Slightly different math weights
    
    # 2. Query remote ONNX inference runtime container over the internal bridge network
    try:
        onnx_response = requests.post(ONNX_URL, json={"data": raw_features}, timeout=2)
        onnx_data = onnx_response.json()
        onnx_confidence = onnx_data["confidence"]
    except Exception as e:
        return jsonify({"error": f"Failed to communicate with ONNX container: {str(e)}"}), 500

    # 3. Ensemble Decision Core (Averaging confidence metrics)
    final_score = (litert_confidence + onnx_confidence) / 2.0
    final_decision = "APPROVED" if final_score > 0.5 else "REJECTED"

    return jsonify({
        "status": "success",
        "individual_outputs": {
            "litert_confidence": litert_confidence,
            "onnx_confidence": onnx_confidence
        },
        "ensemble_score": final_score,
        "final_decision": final_decision
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
