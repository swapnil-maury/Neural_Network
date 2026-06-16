import numpy as np
from tensorflow.keras.datasets import mnist
import os

print("Downloading MNIST data...")
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("Processing images...")
# Flatten 28x28 images into 784-element arrays
# Cast to float64 to perfectly match C++ 'double' (8 bytes)
X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0

print("One-hot encoding labels...")
# Convert labels (0-9) to 10-element arrays (e.g., 3 -> [0,0,0,1,0,0,0,0,0,0])
Y_train_oh = np.zeros((Y_train.size, 10), dtype=np.float64)
Y_train_oh[np.arange(Y_train.size), Y_train] = 1.0

Y_test_oh = np.zeros((Y_test.size, 10), dtype=np.float64)
Y_test_oh[np.arange(Y_test.size), Y_test] = 1.0

print("Saving directly to raw binary files (.bin)...")
# .tofile() dumps pure C-contiguous memory bytes. C++ will read this instantly!
X_train.tofile("X_train.bin")
Y_train_oh.tofile("Y_train.bin")
X_test.tofile("X_test.bin")
Y_test_oh.tofile("Y_test.bin")

print("Done! You now have 4 binary files ready for C++.")

