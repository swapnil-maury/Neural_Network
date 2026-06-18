import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------
# RMSNorm Layer
# --------------------------------------------------
# tf.config.threading.set_intra_op_parallelism_threads(10)
# tf.config.threading.set_inter_op_parallelism_threads(10)

# print("CPU Text this time")
class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True
        )

    def call(self, x):
        rms = tf.sqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
            + self.eps
        )
        return (x / rms) * self.scale

# --------------------------------------------------
# Start Program Timer
# --------------------------------------------------

program_start = time.perf_counter()

# --------------------------------------------------
# Load MNIST
# --------------------------------------------------

print("Loading data...")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

print("Data loaded completely!")
print("Building model...")

# --------------------------------------------------
# Build Model
# --------------------------------------------------

model = keras.Sequential([
    layers.Input(shape=(784,)),

    layers.Dense(2048, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(2048, activation="relu"),

    RMSNorm(),

    layers.Dense(10)  # matches your C++ exactly
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# --------------------------------------------------
# Train
# --------------------------------------------------

epochs = 10
batch_size = 256

train_start = time.perf_counter()

history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

train_end = time.perf_counter()

print(
    f"\nTraining Time: "
    f"{train_end - train_start:.2f} seconds"
)

# --------------------------------------------------
# Predict
# --------------------------------------------------

print("\nSwitching to evaluation mode...")
print("Making predictions on test data...")

pred_start = time.perf_counter()

predictions = model.predict(
    x_test,
    batch_size=batch_size,
    verbose=1
)

pred_end = time.perf_counter()

print(
    f"\nPrediction Time: "
    f"{pred_end - pred_start:.2f} seconds"
)

# --------------------------------------------------
# Total Timer
# --------------------------------------------------

program_end = time.perf_counter()

print(
    f"\nTotal Execution Time: "
    f"{program_end - program_start:.2f} seconds"
)

# --------------------------------------------------
# Accuracy
# --------------------------------------------------

pred_labels = np.argmax(predictions, axis=1)

accuracy = np.mean(pred_labels == y_test)

print("\n========================================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("========================================")

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nGenerating Model Summary...\n")
model.summary()

