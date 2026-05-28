import alsaaudio
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import time

# 1. Configuration
MODEL_PATH = "classifier.tflite"
SAMPLING_RATE = 16000
WINDOW_SECONDS = 0.975 # TFLite YAMNet expects exactly 15600 samples
SAMPLES_PER_WINDOW = int(SAMPLING_RATE * WINDOW_SECONDS)

# Safety mapping
CRITICAL_INDICES = {494:"SILENCE", 434: "CRASH", 505: "TIRE SQUEAL", 508: "SKIDDING"}
THRESHOLD = 0.25

# 2. Initialize TFLite Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# YAMNet TFLite usually has index 0 as input, and index 0 of outputs as scores
input_index = input_details[0]['index']
scores_index = output_details[0]['index']

# 3. Initialize ALSA

mic = alsaaudio.PCM(
    alsaaudio.PCM_CAPTURE,
    cardindex=3, 
    channels=1, 
    rate=SAMPLING_RATE, 
    format=alsaaudio.PCM_FORMAT_S16_LE, 
    periodsize=1024
)


print("TFLite Safety Monitor Active...")

audio_buffer = np.array([], dtype=np.float32)

try:
    while True:
        l, data = mic.read()
        if l > 0:
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.append(audio_buffer, chunk)

            if len(audio_buffer) >= SAMPLES_PER_WINDOW:
                # Prepare input tensor [1, 15600]
                input_data = audio_buffer[:SAMPLES_PER_WINDOW].astype(np.float32)
                
                # Run Inference
                interpreter.set_tensor(input_index, input_data)
                interpreter.invoke()
                
                # Get Scores
                scores = interpreter.get_tensor(scores_index)
                mean_scores = np.mean(scores, axis=0) # Average over frames
                
                # Check for events
                for idx, label in CRITICAL_INDICES.items():
                    if mean_scores[idx] > THRESHOLD:
                        print(f"[!!!] {label} DETECTED (Score: {mean_scores[idx]:.2f})")
                
                # Shift buffer (50% overlap)
                audio_buffer = audio_buffer[int(SAMPLES_PER_WINDOW / 2):]

except KeyboardInterrupt:
    print("Shutting down.")
