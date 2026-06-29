import os
import numpy as np
import tensorflow as tf

# Load the 28x28 dataset for evaluation
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test_scaled = X_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

def evaluate_tflite_model(model_path, is_int8=False):
    if not os.path.exists(model_path):
        return None, None
        
    size_bytes = os.path.getsize(model_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct_predictions = 0
    
    # We evaluate on a subset (10000 mages)
    test_subset = X_test_scaled[:10000]
    y_test_subset = y_test[:10000]
    
    for i in range(len(test_subset)):
        input_data = np.expand_dims(test_subset[i], axis=0)
        
        if is_int8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.int8)
            
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        
        if prediction == y_test_subset[i]:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(test_subset)
    return accuracy, size_bytes

print("--- FINAL 28x28 EDGE AI COMPRESSION MATRIX ---")
models = [
    ("Baseline (FP32)", "model_fp32.tflite", False),
    ("Quantized (INT8)", "model_int8.tflite", True),
    ("Pruned 70% (FP32)", "model_pruned_70_fp32.tflite", False),
    ("Pruned 70% + INT8", "model_pruned_70_int8.tflite", True)
]

baseline_size = None

for name, path, is_int8 in models:
    acc, size = evaluate_tflite_model(path, is_int8)
    if acc is not None:
        if baseline_size is None:
            baseline_size = size
        comp_ratio = baseline_size / size
        print(f"{name:<20} | Acc: {acc*100:.2f}% | Size: {size:<6} bytes | Ratio: {comp_ratio:.2f}x")
    else:
        print(f"{name:<20} | FILE NOT FOUND")