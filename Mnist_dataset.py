import numpy as np
from tensorflow.keras.datasets import mnist

def export_mnist_to_bin():
    print("Downloading MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0

    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train].astype(np.float64)
    y_test_onehot = np.eye(num_classes)[y_test].astype(np.float64)

    # 3. Save to raw binary files
    print("Saving binary files...")
    x_train_flat.tofile("train_X.bin")
    y_train_onehot.tofile("train_Y.bin")
    x_test_flat.tofile("test_X.bin")
    y_test_onehot.tofile("test_Y.bin")

    print("Done! Files are ready for your C++ framework.")
    print(f"Train X shape: {x_train_flat.shape}, Train Y shape: {y_train_onehot.shape}")

if __name__ == "__main__":
    export_mnist_to_bin()