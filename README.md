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

The project includes several examples, including an example that trains a small GPT-style transformer to generate human-like names based on a dataset.

That example will:
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
