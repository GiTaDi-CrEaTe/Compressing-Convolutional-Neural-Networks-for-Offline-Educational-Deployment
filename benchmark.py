"""
benchmark.py
Edge Inference Benchmarking Harness

This script evaluates the physical footprint and inference latency of the 
compiled TensorFlow Lite artifacts. It simulates deployment constraints on 
low-resource edge hardware by strictly tracking RAM allocations and 
per-inference execution time.
"""

import time
import tracemalloc
import numpy as np
import tensorflow as tf

def load_mnist_test_data(num_samples=1000):
    """Loads a subset of MNIST test data for benchmarking."""
    (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test[:num_samples]

def benchmark_model(model_path, x_data):
    """Measures latency and memory consumption for a given TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if model expects INT8 inputs (fully quantized) or FP32
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        x_data = np.round((x_data / scale) + zero_point).astype(np.int8)

    # Warm-up run
    interpreter.set_tensor(input_details[0]['index'], x_data[0:1])
    interpreter.invoke()

    # Memory Tracking
    tracemalloc.start()
    
    # Latency Tracking
    start_time = time.perf_counter()
    for i in range(len(x_data)):
        interpreter.set_tensor(input_details[0]['index'], x_data[i:i+1])
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.perf_counter()
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_latency_ms = ((end_time - start_time) / len(x_data)) * 1000
    peak_ram_kb = peak_memory / 1024

    return avg_latency_ms, peak_ram_kb

if __name__ == "__main__":
    print("Initializing Edge Inference Benchmark...")
    x_test = load_mnist_test_data(num_samples=1000)
    
    models = {
        "Baseline (FP32)": "models/model_fp32.tflite",
        "Quantized (INT8)": "models/model_int8.tflite"
    }
    
    print("\n| Model Configuration | Avg Latency (ms/img) | Peak RAM (KB) |")
    print("|---------------------|----------------------|---------------|")
    
    for name, path in models.items():
        try:
            latency, ram = benchmark_model(path, x_test)
            print(f"| {name:<19} | {latency:>20.2f} | {ram:>13.2f} |")
        except ValueError as e:
            print(f"| {name:<19} | ERROR: {str(e)[:15]}... |           --- |")
            
    print("\nBenchmark complete.")