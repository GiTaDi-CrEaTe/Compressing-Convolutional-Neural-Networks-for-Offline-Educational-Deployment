import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------
# 1. Load and prep 28x28 MNIST data (The Academic Standard)
# ---------------------------------------------------------------------
print("Loading 28x28 MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape to (28, 28, 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# ---------------------------------------------------------------------
# 2. Building the Edge CNN (Optimized for 28x28)
# ---------------------------------------------------------------------
print("Constructing the Convolutional Neural Network...")
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------------------------------------------------
# 3. Training the Baseline
# ---------------------------------------------------------------------
# 5 epochs is enough to reach >98% on MNIST
print("Initiating training sequence...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

model.save("baseline_cnn_28.keras")

# ---------------------------------------------------------------------
# 4. TFLite Conversion
# ---------------------------------------------------------------------
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_fp32 = converter.convert()
with open("model_fp32_28.tflite", "wb") as f:
    f.write(tflite_model_fp32)

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_data_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8
tflite_model_int8 = converter_int8.convert()
with open("model_int8_28.tflite", "wb") as f:
    f.write(tflite_model_int8)

print("\nArtifacts saved as: model_fp32_28.tflite and model_int8_28.tflite")