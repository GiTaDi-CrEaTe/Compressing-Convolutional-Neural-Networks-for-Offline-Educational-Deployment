"""
error_analysis.py
Per-Class Degradation Engine

Isolates and visualizes the specific structural vulnerabilities 
introduced by INT8 quantization by comparing its failure modes 
against the FP32 baseline.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_predictions(model_path, x_data):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        x_data = np.round((x_data / scale) + zero_point).astype(np.int8)

    preds = []
    for i in range(len(x_data)):
        interpreter.set_tensor(input_details[0]['index'], x_data[i:i+1])
        interpreter.invoke()
        preds.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    return np.array(preds)

def generate_degradation_matrix():
    print("Loading MNIST test set...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, -1)
    
    print("Running FP32 Inference...")
    preds_fp32 = get_predictions("models/model_fp32.tflite", x_test)
    
    print("Running INT8 Inference...")
    preds_int8 = get_predictions("models/model_int8.tflite", x_test)
    
    # Isolate instances where FP32 was correct, but INT8 failed
    degraded_indices = np.where((preds_fp32 == y_test) & (preds_int8 != y_test))[0]
    print(f"Total isolated quantization degradations: {len(degraded_indices)} / {len(y_test)}")
    
    if len(degraded_indices) == 0:
        print("No degradation detected. FP32 and INT8 are perfectly aligned.")
        return

    y_true_degraded = y_test[degraded_indices]
    y_pred_degraded = preds_int8[degraded_indices]
    
    cm = confusion_matrix(y_true_degraded, y_pred_degraded, labels=np.arange(10))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title('Quantization Degradation: INT8 Failure Modes vs FP32 Baseline', pad=20)
    plt.ylabel('True Digit (FP32 Correct)')
    plt.xlabel('Quantized Prediction (INT8 Incorrect)')
    
    plt.tight_layout()
    plt.savefig('quantization_degradation_matrix.png', dpi=300)
    print("Saved degradation matrix to 'quantization_degradation_matrix.png'")

if __name__ == "__main__":
    generate_degradation_matrix()