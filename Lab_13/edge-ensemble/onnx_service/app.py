from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Mock initializing an ONNX session. In real use: ort.InferenceSession("model.onnx")
print("ONNX Runtime initialized successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json
    input_data = np.array(payload.get("data", [0.0]), dtype=np.float32)
    
    # Mock inference calculation: simple scalar scaling
    # Replace this block with actual: session.run(None, {input_name: input_data})
    onnx_confidence = float(1.0 / (1.0 + np.exp(-input_data.sum()))) 
    
    return jsonify({
        "runtime": "onnx",
        "confidence": onnx_confidence,
        "prediction": "Class_A" if onnx_confidence > 0.5 else "Class_B"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)