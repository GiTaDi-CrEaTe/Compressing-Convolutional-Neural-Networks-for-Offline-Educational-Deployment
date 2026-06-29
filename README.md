# Compressing-Convolutional-Neural-Networks-for-Offline-Educational-Deploymen

# Edge AI Compression: Pruning and INT8 Quantization on MNIST

Author: Adityajyoti Kar

Overview:
This repository contains the deployment pipeline and experimental artifacts for compressing a Convolutional Neural Network (CNN) designed for extreme low-resource offline inference. By utilizing iterative magnitude-based weight pruning (TF-MOT) and post-training INT8 quantization (TFLite), this project achieves a 3.64x memory reduction with zero accuracy loss (98.78%) on the MNIST 28x28 dataset.

Repository Structure:

    cnn_baseline.py: Constructs, trains, and exports the true FP32 baseline CNN.

    cnn_pruning.py: Clones the isolated baseline weights, applies 70% structural sparsity, fine-tunes, and exports the compressed TFLite artifacts.

    evaluate_models.py: Runs the standard TFLite Interpreter across all 4 experimental artifacts on the full 10,000-image test set to calculate accuracy and byte footprints.

    make_figures.py: Matplotlib generator for the architectural and empirical charts.

    /*.tflite: The final compiled deployment artifacts.

Quick Start to Reproduce Results:

    Install requirements: pip install tensorflow tensorflow-model-optimization scikit-learn matplotlib numpy

    Run python evaluate_models.py to instantly verify the 98.78% accuracy and file size footprint of the included .tflite artifacts.

##ABSTRACT--
An Empirical Study of Pruning and INT8 Quantization
Access to artificial intelligence tools in rural and
low-resource educational settings is fundamentally constrained
by hardware: most schools have, at best, low-end smartphones or
shared desktop computers, and rarely have internet connectivity
reliable enough for cloud-based inference. This paper presents
a fully reproducible empirical study of two standard model-
compression techniques—magnitude-based weight pruning and
post-training 8-bit (INT8) quantization—applied to a compact
convolutional neural network (CNN) trained for handwritten
digit recognition on the MNIST dataset. Using a controlled
experimental design in which an independently cloned copy of a
trained baseline model is pruned and fine-tuned, we measure the
accuracy, model size, and compression ratio of four deployment
configurations. The baseline CNN achieves 98.78% test accuracy
on the full 10,000-image test set with a 224,892-byte TensorFlow
Lite footprint. INT8 quantization reduces this to 61,800 bytes
(a 3.64x reduction) with a negligible 0.09 percentage-point shift
in accuracy. Iterative pruning to 70% sparsity with fine-tuning
matches the baseline almost perfectly at 98.77%, while leaving
the FP32 file size unchanged—a result we explain in terms of
the dense storage format used by standard TensorFlow Lite
flatbuffers. Combining pruning with quantization achieves the
same 3.64x compression as quantization alone, restoring accuracy
to exactly 98.78%. These results provide a transparent, low-
cost, and entirely reproducible reference point for deploying
handwriting-recognition tools on inexpensive edge hardware
without requiring GPUs or cloud services.
