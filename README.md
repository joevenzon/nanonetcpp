# nanonetcpp

`nanonetcpp` is a tiny, high-performance, minimal C++ neural networking library designed for educational purposes and quickly training tiny networks. It implements a custom automatic differentiation (AutoGrad) engine and provides the building blocks necessary to construct modern transformer-based or convolution architectures. Even though it runs single-threaded on the CPU, due to its minimal overhead, it's often faster for training very small (less than 100k node) networks than pytorch on a GPU.

## Key Features

- **Custom AutoGrad Engine**: A lightweight automatic differentiation system that uses a pre-allocated node pool to minimize heap allocations and maximize cache locality during the forward and backward passes.
- **Optimization**: Includes a full implementation of the **Adam Optimizer** with learning rate decay.
- **Persistence**: `ParameterCheckpoint` system for saving and restoring model weights to binary files.
- **Modern C++**: Written in **C++20**, leveraging `std::span` and other modern features for safety and performance.
- **Curiously fast**: Very low overhead allocation schemes and AVX2 matmul implementation lead to blazing fast small model training.

## Getting Started

### Prerequisites

- A C++20 compatible compiler (e.g., MSVC 2019+, GCC 10+, Clang 10+).
- [CMake](https://cmake.org/) (version 3.16 or higher).

### Building the Project

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Running the Examples

The project includes several examples, including one that trains a small GPT-style transformer to generate human-like names based on a dataset.

The GPT example will:
- Load the names.txt dataset.
- Initialize a transformer model.
- Train the model using the Adam optimizer.
- Periodically report training and validation loss.
- Generate 20 "hallucinated" names after training.

Running 10,000 training steps for this 4,256 parameter neural network should take about 0.5 seconds.

## Architecture Overview

### AutoGrad Node Pool
Unlike many AutoGrad implementations that create a graph of objects on the heap, `nanonetcpp` uses a linearly allocated `Node` pool. This approach:
- Reduces allocation overhead.
- Ensures that the computation graph is stored contiguously in memory.
- Simplifies the backward pass to a simple reverse iteration over the pool.

### Model Design
The library follows a modular layer-based design. Each layer (e.g., `LinearLayer`, `AttentionLayer`) handles its own forward pass by interacting with the `AutoGrad` engine to register operations in the computation graph.

## How to Use the Library

### 1. Setting Up the AutoGrad Engine

Every program starts by creating an `AutoGrad<float>` engine. This engine manages the computation graph, tensor memory, and gradient storage. You can supply a different numeric type if you want, although the float version contains specialized optimizations.

```cpp
#include "autograd.h"
#include "linearlayer.h"
#include "adamoptimizer.h"
#include "parametercheckpoint.h"

AutoGrad<float> grad;
```

The engine uses pre-allocated memory arenas for nodes and values, so it is fast and avoids per-step heap allocations. You can customize the arena sizes and random seed at initialization:

```cpp
grad.init(/* seed */ 42, /* node_capacity */ 8192, /* value_capacity */ 131072);
```

### 2. Defining a Model

Models are typically expressed as a struct that holds layer objects. Each layer is initialized by calling its `init()` method with a reference to the `AutoGrad` engine, which registers the layer's trainable parameters. This isn't required but is a nice pattern that mirrors typical usage of pytorch.

```cpp
struct Model
{
    static const int hidden_dim = 64;

    LinearLayer<float> first_layer;
    LinearLayer<float> second_layer;
    LinearLayer<float> output_layer;

    void init(AutoGrad<float> & grad)
    {
        first_layer.init(grad, 1, hidden_dim);
        second_layer.init(grad, hidden_dim, hidden_dim);
        output_layer.init(grad, hidden_dim, 1);
    }
};
```

Available layers include:

| Layer | Header | Purpose |
|---|---|---|
| `LinearLayer` | `linearlayer.h` | Fully-connected (matrix multiply + optional bias) |
| `Conv2DLayer` | `conv2dlayer.h` | 2D convolution |
| `EmbeddingLayer` | `embeddinglayer.h` | Token-to-vector lookup |
| `SoftmaxLayer` | `softmaxlayer.h` | Softmax across a dimension |
| `RMSNormLayer` | `rmsnormlayer.h` | RMS normalization |
| `AttentionLayer` | `attentionlayer.h` | Multi-head self-attention |
| `MLPLayer` | `mlplayer.h` | Multi-layer perceptron sub-block |
| `TransformerBlock` | `transformerblock.h` | Full transformer block (attention + MLP + norms) |

### 3. Implementing the Forward Pass

The forward pass constructs the computation graph by chaining operations through the `AutoGrad` engine. Input data is wrapped in a leaf tensor, and each layer or activation returns a `TensorHandle` that references a node in the graph.

```cpp
TensorHandle forward(AutoGrad<float> & grad, float inputvalue)
{
    // Create a leaf tensor from raw input data
    TensorHandle input = grad.tensor_leaf(TensorShape{ 1, 1 }, inputvalue);

    // Pass through layers, interleaving with activations
    TensorHandle h = first_layer.forward(grad, input);
    h = grad.value_tanh(h);
    h = second_layer.forward(grad, h);
    h = grad.value_tanh(h);
    TensorHandle output = output_layer.forward(grad, h);

    return output;
}
```

Several operations are available on the AutoGrad class, including element-wise arithmetic, activation functions, matrix multiplication, reduction, and reshaping. Here's a list of a few examples:

| Operation | Method |
|---|---|
| Addition | `grad.value_add(a, b)` |
| Subtraction | `grad.value_sub(a, b)` or `grad.value_sub_const(a, c)` |
| Multiplication | `grad.value_mul(a, b)` or `grad.value_mul_const(a, c)` |
| Power | `grad.value_pow(a, exponent)` |
| Tanh | `grad.value_tanh(a)` |
| ReLU | `grad.value_relu(a)` |
| Matrix multiply | `grad.value_matmul(a, b)` |
| Sum reduction | `grad.value_sum(a)` |

### 4. Setting Up the Optimizer and Checkpoint

After initializing the model, snapshot the parameters and create an optimizer and checkpoint. Snapshotting the parameters after initializing the model (but before any forward passes are run) is critical to ensuring the memory allocation scheme works.

```cpp
Model model;
model.init(grad);
grad.snapshot_parameters();

AdamOptimizer<float> optimizer;
optimizer.lr = 0.005f;

ParameterCheckpoint<float> checkpoint;
checkpoint.init(grad);
```

The `ParameterCheckpoint` records the current parameter values and can save/restore them to binary files. Its use is optional. The `AdamOptimizer` maintains per-parameter first and second moment estimates.

### 5. Training Loop

A standard training step follows this pattern:

```cpp
for (int step = 0; step < num_training_steps; step++)
{
    // 1. Zero out gradients from the previous step
    grad.zero_grad();

    // 2. Forward pass: generate prediction and compute loss
    TensorHandle prediction = model.forward(grad, train_x);
    TensorHandle loss = grad.value_pow(
        grad.value_sub_const(prediction, train_y), 2);

    // 3. Backward pass: compute gradients
    grad.backward(loss, /* zero_gradients */ false);

    // 4. Update parameters with the optimizer
    optimizer.step(grad);
}
```

#### Gradient Accumulation

For larger effective batch sizes, accumulate gradients across multiple micro-steps before calling `optimizer.step()`:

```cpp
grad.zero_grad();
for (int acc = 0; acc < accumulation_steps; acc++)
{
    TensorHandle prediction = model.forward(grad, train_x);
    TensorHandle loss = grad.value_pow(
        grad.value_sub_const(prediction, train_y), 2);

    // Scale loss so the sum equals the mean
    float scale = 1.0f / accumulation_steps;
    TensorHandle scaled_loss = grad.value_mul_const(loss, scale);

    grad.backward(scaled_loss, false);
}
optimizer.step(grad);
```

#### Memory Management During Inference

Call `grad.restore_allocators()` at the start of each inference pass to reset the node pool and value arena without deallocating memory. Note that calling backward() will also reset the arena allocators, so you don't need to call this during training.

```cpp
grad.restore_allocators(); 
```

### 6. Inference

After training, run inference by calling the forward pass and extracting values from the resulting tensor:

```cpp
grad.restore_allocators();
TensorHandle prediction = model.forward(grad, input_value);
float result = grad.get(prediction).tensor.values()[0];
```

### 7. Saving and Loading Checkpoints

Save the trained weights to a binary file:

```cpp
checkpoint.update(grad, optimizer.step_count);
checkpoint.saveToFile("checkpoint.chk");
```

Load them back later:

```cpp
checkpoint.loadFromFile("checkpoint.chk");
checkpoint.apply(grad);  // restore weights into the AutoGrad engine
```
