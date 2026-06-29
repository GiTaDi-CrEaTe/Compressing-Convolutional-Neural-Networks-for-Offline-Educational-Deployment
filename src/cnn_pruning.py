import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------
# 1. Load and prep 28x28 MNIST data
# ---------------------------------------------------------------------
print("Loading 28x28 MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape to 3D tensors and normalize to 0-1
X_train_scaled = X_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
X_test_scaled = X_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

# ---------------------------------------------------------------------
# 2. Build and Pre-train the TRUE Baseline CNN
# ---------------------------------------------------------------------
print("Building and Pre-training TRUE Baseline CNN...")
base_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the true baseline
print("Training Baseline...")
base_model.fit(X_train_scaled, y_train, epochs=3, batch_size=64, verbose=1)

# --- THE CRITICAL FIX: ISOLATION ---
print("\nSaving true baseline weights to disk...")
base_model.save_weights("true_baseline.weights.h5")

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train_scaled).batch(1).take(100):
        yield [input_value]

print("Exporting Independent Baseline Artifacts...")
converter_base_fp32 = tf.lite.TFLiteConverter.from_keras_model(base_model)
with open("model_fp32.tflite", "wb") as f:
    f.write(converter_base_fp32.convert())

converter_base_int8 = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter_base_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_base_int8.representative_dataset = representative_data_gen
converter_base_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_base_int8.inference_input_type = tf.int8
converter_base_int8.inference_output_type = tf.int8
with open("model_int8.tflite", "wb") as f:
    f.write(converter_base_int8.convert())

# ---------------------------------------------------------------------
# 3. Clone for Independent Pruning Branch
# ---------------------------------------------------------------------
print("\nCloning network for structurally independent Prune-and-Retrain path...")
pruned_base = tf.keras.models.clone_model(base_model)
pruned_base.load_weights("true_baseline.weights.h5")

epochs_prune = 2
batch_size = 64
num_images = X_train_scaled.shape[0]
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs_prune

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.0,
          final_sparsity=0.70, # Target 70% sparsity
          begin_step=0,
          end_step=end_step)
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(pruned_base, **pruning_params)
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ---------------------------------------------------------------------
# 4. Fine-Tune the Pruned Model
# ---------------------------------------------------------------------
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

print("Initiating Prune-and-Retrain sequence (70% Target Sparsity)...")
pruned_model.fit(X_train_scaled, y_train, epochs=epochs_prune, batch_size=batch_size, validation_data=(X_test_scaled, y_test), callbacks=callbacks, verbose=1)

# ---------------------------------------------------------------------
# 5. Export Independent Pruned Artifacts
# ---------------------------------------------------------------------
model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

print("\nExporting Pruned Deployment Artifacts...")
converter_pruned_fp32 = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
with open("model_pruned_70_fp32.tflite", "wb") as f:
    f.write(converter_pruned_fp32.convert())

converter_pruned_int8 = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter_pruned_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_pruned_int8.representative_dataset = representative_data_gen
converter_pruned_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_pruned_int8.inference_input_type = tf.int8
converter_pruned_int8.inference_output_type = tf.int8
with open("model_pruned_70_int8.tflite", "wb") as f:
    f.write(converter_pruned_int8.convert())

print("\nAll 4 fully independent experimental artifacts generated successfully.")