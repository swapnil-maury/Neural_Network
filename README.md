# Custom C++ Neural Network Framework

A high-performance Deep Learning framework built entirely from scratch in modern C++17.

Unlike educational implementations that model individual neurons and connections as separate objects, this framework is built around matrix algebra using Eigen, enabling efficient training, inference, and future GPU portability.

The framework supports Dense Neural Networks, Convolutional Neural Networks (CNNs), normalization layers, transfer learning, model serialization, custom losses, custom optimizers, OpenMP acceleration, and binary dataset pipelines.

---

# 🚀 Key Features

* Matrix-first architecture
* Dense Neural Networks
* Convolutional Neural Networks (CNNs)
* Batch Normalization
* Layer Normalization
* RMS Normalization (RMSNorm)
* Dropout Regularization
* Transfer Learning
* Model Serialization
* Binary Dataset Loading
* OpenMP Multithreading
* Eigen Matrix Backend
* Custom Activations
* Custom Loss Functions
* Custom Optimizers
* Multi-output Regression
* Binary Classification
* Multi-class Classification
* Terminal Visualization Tools
* Interactive Prediction Inspection

---

# 🏗️ Supported Layers

| Layer               | Supported |
| ------------------- | --------- |
| Dense Layer         | ✅         |
| Conv2D Layer        | ✅         |
| Flatten Layer       | ✅         |
| Batch Normalization | ✅         |
| Layer Normalization | ✅         |
| RMS Normalization   | ✅         |
| Dropout Layer       | ✅         |
| Activation Layer    | ✅         |

---

## Dense Layers

Features:

* Fully connected feed-forward computation
* Matrix-based forward propagation
* Matrix-based backpropagation
* Trainable parameters
* Layer freezing support
* Transfer learning support

---

## Convolutional Layers

Features:

* Multi-channel convolutions
* Configurable kernel size
* Configurable stride
* Configurable padding
* im2col acceleration
* Spatial backpropagation

Example:

```cpp
model.add_layer(
    nn::Conv2DLayer(
        1,      // input channels
        16,     // output channels
        3,      // kernel size
        1,      // stride
        1,      // padding
        28,     // input height
        28      // input width
    )
);
```

---

## Normalization Layers

Supported:

* Batch Normalization
* Layer Normalization
* RMS Normalization (RMSNorm)

Benefits:

* Faster convergence
* Improved gradient flow
* More stable training
* Better deep-network optimization

---

## Regularization

### Dropout

Features:

* Random neuron masking
* Reduced overfitting
* Automatic train/evaluation behavior

Example:

```cpp
model.add_layer(nn::DropoutLayer(0.2));
```

---

# ⚡ Activation Functions

| Activation   | Supported |
| ------------ | --------- |
| Identity     | ✅         |
| Binary Step  | ✅         |
| Sigmoid      | ✅         |
| Tanh         | ✅         |
| ReLU         | ✅         |
| ReLU6        | ✅         |
| Leaky ReLU   | ✅         |
| PReLU        | ✅         |
| ELU          | ✅         |
| SELU         | ✅         |
| Softplus     | ✅         |
| Softsign     | ✅         |
| Swish        | ✅         |
| SiLU         | ✅         |
| GELU         | ✅         |
| Mish         | ✅         |
| Hard Sigmoid | ✅         |
| Hard Swish   | ✅         |
| Softmax      | ✅         |

Parameterized activations:

* LeakyReLU
* PReLU
* ELU
* SELU
* Swish

Example:

```cpp
activations::LeakyReLU(0.01);
activations::PReLU(0.25);
activations::Swish(1.0);
```

---

# 📉 Loss Functions

The framework uses a modular loss abstraction:

```cpp
LossFunction(
    loss_function,
    gradient_function,
    "name"
);
```

Loss functions are responsible for:

* Computing scalar loss values
* Computing output gradients

Losses never update model parameters directly.

---

## Supported Loss Functions

### Regression

| Loss                                  | Supported |
| ------------------------------------- | --------- |
| Mean Squared Error (MSE)              | ✅         |
| Mean Absolute Error (MAE)             | ✅         |
| Huber Loss                            | ✅         |
| Log-Cosh Loss                         | ✅         |
| Mean Squared Logarithmic Error (MSLE) | ✅         |
| Poisson Loss                          | ✅         |

### Classification

| Loss                             | Supported |
| -------------------------------- | --------- |
| Binary Cross Entropy             | ✅         |
| Categorical Cross Entropy        | ✅         |
| Softmax Cross Entropy            | ✅         |
| Sparse Categorical Cross Entropy | ✅         |

### Margin-Based

| Loss               | Supported |
| ------------------ | --------- |
| Hinge Loss         | ✅         |
| Squared Hinge Loss | ✅         |

### Distribution-Based

| Loss          | Supported |
| ------------- | --------- |
| KL Divergence | ✅         |

---

## Custom Loss Functions

Custom objectives can be implemented without modifying framework internals:

```cpp
LossFunction custom_loss(
    loss_fn,
    gradient_fn,
    "CustomLoss"
);
```

---

# ⚡ Optimizers

The optimizer system is fully decoupled from layers and loss functions.

Features:

* Learning rate management
* Internal optimizer states
* Momentum buffers
* Adaptive learning rates
* Automatic state initialization

---

## Supported Optimizers

| Optimizer    | Supported |
| ------------ | --------- |
| SGD          | ✅         |
| Momentum SGD | ✅         |
| RMSProp      | ✅         |
| Adam         | ✅         |

---

## Optimizer State Management

Optimizer states are automatically allocated according to layer dimensions.

### Momentum

Stores:

* Velocity Weights
* Velocity Biases

### RMSProp

Stores:

* Running Gradient Squares

### Adam

Stores:

* First Moment Estimates
* Second Moment Estimates
* Timestep Tracking

---

## Custom Optimizers

Custom optimizers can be implemented through:

```cpp
Optimizer(
    update_rule,
    learning_rate,
    state_count,
    optimizer_name
);
```

Potential extensions:

* AdamW
* AdaGrad
* NAdam
* Lion
* LAMB

---

# 🎯 Supported Tasks

## Regression

Supports:

* Single-output regression
* Multi-output regression

Metrics:

* R² Score

---

## Binary Classification

Typical setup:

```cpp
Sigmoid + BinaryCrossEntropy
```

---

## Multi-Class Classification

Typical setup:

```cpp
Softmax + SoftmaxCrossEntropy
```

Metrics:

* Accuracy
* Confusion Matrix

---

# 🔄 Transfer Learning

Freeze previously trained layers while fine-tuning new layers.

Example:

```cpp
for(auto& layer : model.get_layers())
{
    layer->set_trainable(false);
}
```

Applications:

* Domain adaptation
* Feature reuse
* Fine-tuning
* Incremental learning

---

# 💾 Model Serialization

Models can be exported directly into C++ source code.

Save:

```cpp
model.save_model();
```

Load:

```cpp
model.load_model(model_param);
```

Benefits:

* No external model files
* Fast deployment
* Easy distribution
* Header-only model storage

Generated file:

```text
output.h
```

---

# 📂 Dataset Utilities

Supported features:

* Binary dataset loading
* High-speed I/O
* One-hot encoded labels
* Large dataset support

Example:

```cpp
auto X_train = load_X_bin("train_X.bin", 60000, 784);
auto Y_train = load_Y_bin("train_Y.bin", 60000, 10);
```

---

# 📊 Evaluation Metrics

## Classification

* Accuracy Score
* Multiclass Accuracy
* Confusion Matrix

Example:

```cpp
accuracy_score_multiclass(...)
confusion_matrix(...)
```

---

## Regression

* R² Score

Example:

```cpp
r2_score(...)
```

---

# 🖥️ Visualization & Debugging

The framework includes command-line debugging and visualization tools.

---

## Terminal Image Visualization

Render grayscale images directly inside the terminal using RGB ANSI colors.

Example:

```cpp
printImage(image, 28, 28);
```

Features:

* Grayscale rendering
* MNIST visualization
* Dataset inspection
* Prediction debugging
* No GUI dependencies

---

## Prediction Analysis

Inspect:

* Raw logits
* Softmax probabilities
* Predicted labels
* Ground truth labels
* Classification confidence

---

## Model Summary

Generate architecture reports:

```cpp
model.summary();
```

Displays:

* Layer types
* Layer dimensions
* Parameter dimensions
* Trainable status
* Network structure

---

# ⚙️ Performance

The framework is optimized around matrix operations rather than neuron-level object graphs.

Technologies:

* Eigen 3.4
* OpenMP
* SIMD Vectorization
* Batch Processing

Benefits:

* Faster training
* Faster inference
* Better cache locality
* Reduced memory overhead
* Future GPU portability

Recommended build:

```bash
g++ -std=c++17 -O3 -march=native -fopenmp main.cpp
```

For large models:

```bash
ulimit -s unlimited && ./a.out
```

---

# 🏛️ Design Philosophy

This framework intentionally avoids neuron-level architectures.

Avoided design:

```text
Neuron -> Object
Connection -> Object
Perceptron -> Object
```

Framework design:

```text
Input
  ↓
Layer
  ↓
Activation
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Optimizer
```

Everything is computed using matrix algebra through Eigen.

Advantages:

* Cleaner implementation
* Better scalability
* Improved cache utilization
* Easier maintenance
* GPU-ready architecture

---

# 🔧 Extensibility

The framework was designed to support custom components without modifying the training loop.

Users can implement:

* Custom Layers
* Custom Activations
* Custom Loss Functions
* Custom Optimizers

This enables experimentation while preserving the core infrastructure.

---

# 🚧 Roadmap

### Completed

* Dense Networks
* CNN Foundations
* Backpropagation
* Batch Normalization
* Layer Normalization
* RMSNorm
* Dropout
* Transfer Learning
* Model Serialization
* OpenMP Parallelization
* MNIST Support

### In Progress

* CNN Optimization
* Additional Layer Types
* Memory Optimizations
* Serialization Improvements

### Planned

* CUDA Backend
* GPU Acceleration
* Learning Rate Scheduling
* Checkpointing
* Additional Optimizers
* Residual Connections
* Transformer Components
* Mixed Precision Training

---

# 📜 License

This project is intended for educational, research, and experimental deep-learning development in modern C++.
