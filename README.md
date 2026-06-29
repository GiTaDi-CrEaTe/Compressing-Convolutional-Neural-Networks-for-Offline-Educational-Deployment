# Compressing-Convolutional-Neural-Networks-for-Offline-Educational-Deploymen

# Edge AI Compression: Pruning and INT8 Quantization on MNIST

Author: Adityajyoti Kar

# ⚡ Edge AI Compression: Pruning & INT8 Quantization

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> **Mission:** Access to artificial intelligence tools in rural and low-resource educational settings is fundamentally constrained by hardware. This project provides a transparent, low-cost, and entirely reproducible pipeline for deploying handwriting-recognition tools on inexpensive edge hardware without requiring GPUs, cloud services, or internet connectivity.

## 📊 Core Benchmarks

By combining iterative magnitude-based weight pruning (TF-MOT) with post-training full-integer quantization (TFLite), this architecture achieves a **3.64x memory reduction** with zero degradation to the baseline prediction accuracy.

| Deployment Configuration | Test Accuracy | File Size (Bytes) | Compression Ratio |
|--------------------------|---------------|-------------------|-------------------|
| Baseline (FP32)          | 98.78%        | 224,892           | 1.00x             |
| Pruned (70% Sparsity)    | 98.77%        | 224,892* | 1.00x             |
| Quantized (INT8)         | 98.69%        | 61,800            | 3.64x             |
| **Pruned + INT8** | **98.78%** | **61,800** | **3.64x** |

*\*Note: FP32 pruned size remains unchanged due to the dense serialization layout of standard TensorFlow Lite flatbuffers, demonstrating that structural sparsity requires quantization or specialized sparsity-aware inference engines to realize physical footprint reductions.*

## 🏗️ Repository Architecture

The codebase is structured for production-grade CI/CD and automated tolerance testing.

```text
.
|-- LICENSE
|-- README.md
|-- benchmark.py
|-- error_analysis.py
|-- evaluate_models.py
|-- figure1.png
|-- figure2.png
|-- figure3.png
|-- figure4.png
|-- make_figures.py
|-- models
|   |-- baseline_cnn_28.keras
|   |-- model_fp32.tflite
|   |-- model_fp32_28.tflite
|   |-- model_int8.tflite
|   |-- model_int8_28.tflite
|   |-- model_pruned_70_fp32.tflite
|   |-- model_pruned_70_int8.tflite
|   `-- true_baseline.weights.h5
|-- src
|   |-- cnn_baseline.py
|   `-- cnn_pruning.py
`-- tests
    `-- test_pipeline.py

🚀 Quick Start

Reproduce the exact compression ratios and accuracy benchmarks in under 5 minutes using a standard CPU.

1. Install Dependencies

pip install tensorflow tensorflow-model-optimization scikit-learn matplotlib numpy pytest seaborn

2. Run the Benchmark Harness
Instantly verify the 98.78% accuracy and 61.8 KB file size footprint of the compiled artifacts against the full 10,000-image test set.

python benchmark.py

3. Execute Automated Guardrails
Verify that INT8 precision loss remains within the strict <0.5% tolerance threshold.

pytest tests/test_pipeline.py -v
