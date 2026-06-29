"""
test_pipeline.py
Automated Guardrails for Quantization Tolerance

Ensures that the INT8 quantization process does not catastrophically 
degrade model performance compared to the FP32 baseline.
"""

import numpy as np
import tensorflow as tf
import pytest

def predict_batch(model_path, x_data):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        x_data = np.round((x_data / scale) + zero_point).astype(np.int8)

    predictions = []
    for i in range(len(x_data)):
        interpreter.set_tensor(input_details[0]['index'], x_data[i:i+1])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output))
    return np.array(predictions)

@pytest.fixture(scope="module")
def mnist_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, -1)
    # Use a small determinisic batch for fast CI testing
    return x_test[:500], y_test[:500]

def test_quantization_tolerance(mnist_data):
    """
    Validates that INT8 quantization accuracy is within a 0.5% 
    tolerance threshold of the FP32 baseline.
    """
    x_test, y_test = mnist_data
    
    preds_fp32 = predict_batch("models/model_fp32.tflite", x_test)
    preds_int8 = predict_batch("models/model_int8.tflite", x_test)
    
    acc_fp32 = np.mean(preds_fp32 == y_test)
    acc_int8 = np.mean(preds_int8 == y_test)
    
    # The absolute critical assertion: INT8 must not degrade > 0.5%
    assert acc_int8 >= (acc_fp32 - 0.005), \
        f"Quantization degradation exceeded threshold! FP32: {acc_fp32:.4f}, INT8: {acc_int8:.4f}"

def test_prediction_invariance(mnist_data):
    """
    Validates that the exact class predictions remain identical 
    across precision formats for at least 98% of the samples.
    """
    x_test, _ = mnist_data
    
    preds_fp32 = predict_batch("models/model_fp32.tflite", x_test)
    preds_int8 = predict_batch("models/model_int8.tflite", x_test)
    
    match_rate = np.mean(preds_fp32 == preds_int8)
    assert match_rate >= 0.98, f"High prediction variance detected. Match rate: {match_rate:.4f}"