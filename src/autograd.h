#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <span>

#include "handle.h"

template <typename DataType>
class AutoGrad
{
public:
    // =============================================================================
    // Node type
    // =============================================================================
    struct Node
    {
        static const int k_max_children = 2;

        DataType data; // The scalar value computed during the forward pass
        DataType grad; // Accumulated gradient (d_loss/d_this_node) from the backward pass
        std::array<NodeHandle, k_max_children> children; // Pool indices of up to k_max_children child nodes
        std::array<DataType, k_max_children> local_grads; // Local derivatives (d_this_node/d_child[i])
        int n_children; // size of above arrays
    };

    // =============================================================================
    // RANDOM GENERATORS
    // =============================================================================
    DataType rand_uniform()
    { 
        return uniform(rng); 
    }

    DataType rand_gaussian(DataType mean, DataType std_dev)
    {
        return std::normal_distribution<DataType>(mean, std_dev)(rng);
    }

    // =============================================================================
    // NODE POOL MANAGEMENT
    // =============================================================================
    NodeHandle allocate()
    {
        NodeHandle index = pool.size();
        pool.resize(index+1); // allocate the node
        return index; // return the index of the allocation
    }

    // Allocate a matrix of leaf nodes in the pool and return the starting offset.
    // @param num_rows    Number of rows in the matrix
    // @param num_cols    Number of columns in the matrix
    // @param std_dev     Standard deviation for Gaussian initialization
    // @return            Pool index of the first element (top-left corner)
    NodeHandle allocate_matrix(int num_rows, int num_cols, DataType std_dev)
    {
        NodeHandle start_offset = pool.size();
        int total_elements = num_rows * num_cols;
        for (int k = 0; k < total_elements; k++)
        {
            value_leaf(rand_gaussian(0, std_dev));
        }
        return start_offset;
    }

    void reserve(size_t capacity)
    {
        pool.reserve(capacity);
    }

    void reset()
    {
        pool.clear();
    }

    int size()
    {
        return pool.size();
    }

    // careful holding on to the returned value because any value_* functions may cause the pool to reallocate and move
    Node & get(NodeHandle index)
    {
        return pool[index];
    }

    void restore_parameter_values(const std::span <DataType> values)
    {
        pool.resize(values.size());
        for (int i = 0; i < values.size(); i++)
        {
            get(NodeHandle(i)).data = values[i];
        }
    }

    // =============================================================================
    // NODE CONSTRUCTORS
    // =============================================================================
    // Each constructor allocates a new node in the pool and initializes its fields.

    // Create a leaf node (no children). Used for constants and parameters.
    NodeHandle value_leaf(DataType data)
    {
        NodeHandle handle = allocate();
        Node & node = get(handle);
        node.data = data;
        node.grad = 0.0f;
        node.n_children = 0;
        return handle;  // Return the pool index of the new node
    }

    // Create a constant (leaf) node. Alias for clarity.
    NodeHandle value_const(DataType data) 
    {
        return value_leaf(data);
    }

    // Addition: out = a + b
    // Local gradients: d_out/d_a = 1, d_out/d_b = 1
    NodeHandle value_add(NodeHandle a, NodeHandle b)
    {
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = get(a).data + get(b).data;
        node.grad = 0.0f;
        node.n_children = 2;
        node.children[0] = a;
        node.children[1] = b;
        node.local_grads[0] = 1.0f;  // d(a+b)/d_a = 1
        node.local_grads[1] = 1.0f;  // d(a+b)/d_b = 1
        return handle;
    }

    NodeHandle value_add_const(NodeHandle a, DataType c)
    {
        NodeHandle handle = allocate();
        Node & node = get(handle);
        node.data = get(a).data + c;
        node.grad = 0.0f;
        node.n_children = 1;
        node.children[0] = a;
        node.local_grads[0] = 1;  // d(a+c)/d_a = 1, c vanishes
        return handle;
    }

    // Subtraction: out = a - b
    // Local gradients: d_out/d_a = 1, d_out/d_b = -1
    NodeHandle value_sub(NodeHandle a, NodeHandle b) 
    {
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = get(a).data - get(b).data;
        node.grad = 0.0f;
        node.n_children = 2;
        node.children[0] = a;
        node.children[1] = b;
        node.local_grads[0] = 1.0f;   // d(a-b)/d_a = 1
        node.local_grads[1] = -1.0f;  // d(a-b)/d_b = -1
        return handle;
    }

    // Multiplication: out = a * b
    // Local gradients: d_out/d_a = b, d_out/d_b = a
    NodeHandle value_mul(NodeHandle a, NodeHandle b) 
    {
        DataType data_a = get(a).data;
        DataType data_b = get(b).data;
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = data_a * data_b;
        node.grad = 0.0f;
        node.n_children = 2;
        node.children[0] = a;
        node.children[1] = b;
        node.local_grads[0] = data_b;  // d(a*b)/d_a = b
        node.local_grads[1] = data_a;  // d(a*b)/d_b = a
        return handle;
    }

    NodeHandle value_mul_const(NodeHandle a, DataType c)
    {
        DataType data_a = get(a).data;
        NodeHandle handle = allocate();
        Node & node = get(handle);
        node.data = data_a * c;
        node.grad = 0.0f;
        node.n_children = 1;
        node.children[0] = a;
        node.local_grads[0] = c;
        return handle;
    }

    // Division: out = a / b
    // Local gradients: d_out/d_a = 1/b, d_out/d_b = -a/b^2
    NodeHandle value_div(NodeHandle a, NodeHandle b) 
    {
        DataType data_a = get(a).data;
        DataType data_b = get(b).data;
        DataType inv_b = 1.0f / data_b;
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = data_a * inv_b;
        node.grad = 0.0f;
        node.n_children = 2;
        node.children[0] = a;
        node.children[1] = b;
        node.local_grads[0] = inv_b;              // d(a/b)/d_a = 1/b
        node.local_grads[1] = -data_a * inv_b * inv_b;  // d(a/b)/d_b = -a/b^2
        return handle;
    }

    // Power: out = a^p (where p is a constant DataType)
    // Local gradient: d_out/d_a = p * a^(p-1)
    NodeHandle value_pow(NodeHandle a, DataType power) 
    {
        DataType data_a = get(a).data;
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = std::pow(data_a, power);
        node.grad = 0.0f;
        node.n_children = 1;
        node.children[0] = a;
        node.local_grads[0] = power * std::pow(data_a, power - 1.0f);
        return handle;
    }

    // Natural logarithm: out = log(a)
    // Local gradient: d_out/d_a = 1/a
    NodeHandle value_log(NodeHandle a) 
    {
        DataType data_a = get(a).data;
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = std::log(data_a);
        node.grad = 0.0f;
        node.n_children = 1;
        node.children[0] = a;
        node.local_grads[0] = 1.0f / data_a;
        return handle;
    }

    // Exponential: out = exp(a)
    // Local gradient: d_out/d_a = exp(a)
    NodeHandle value_exp(NodeHandle a) 
    {
        DataType exp_val = std::exp(get(a).data);
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = exp_val;
        node.grad = 0.0f;
        node.n_children = 1;
        node.children[0] = a;
        node.local_grads[0] = exp_val;  // d(exp(a))/d_a = exp(a)
        return handle;
    }

    // ReLU activation: out = max(0, a)
    // Local gradient: d_out/d_a = 1 if a > 0, else 0
    NodeHandle value_relu(NodeHandle a) 
    {
        DataType data_a = get(a).data;
        NodeHandle handle = allocate();
        Node &node = get(handle);
        node.data = (data_a > 0.0f) ? data_a : 0.0f;
        node.grad = 0.0f;
        node.n_children = 1;
        node.children[0] = a;
        node.local_grads[0] = (data_a > 0.0f) ? 1.0f : 0.0f;
        return handle;
    }

    // Error function: out = erf(a)
    // Local gradient: d/da erf(a) = (2/sqrt(pi)) * exp(-a^2)
    NodeHandle value_erf(NodeHandle a)
    {
        DataType data_a = get(a).data;
        NodeHandle handle = allocate();
        Node & node = get(handle);
        node.data = std::erf(data_a);
        node.grad = 0.0f;
        node.n_children = 1;
        node.children[0] = a;

        // (2/sqrt(pi)) * exp(-a^2)
        static const DataType TWO_OVER_SQRT_PI = 1.1283791670955126;
        node.local_grads[0] = TWO_OVER_SQRT_PI * std::exp(-data_a * data_a);
        return handle;
    }

    NodeHandle value_gelu(NodeHandle a)
    {
        static const DataType INV_SQRT2 = 1.0 / std::sqrt(2.0);

        // x / sqrt(2)
        NodeHandle x_scaled = value_mul_const(a, INV_SQRT2);
        
        // erf(x / sqrt(2))
        NodeHandle erf_val = value_erf(x_scaled);

        // 1 + erf(x / sqrt(2))
        NodeHandle one_plus_erf = value_add_const(erf_val, 1);

        // 0.5 * x * (1 + erf(x / sqrt(2)))
        return value_mul_const(
            value_mul(a, one_plus_erf),
            0.5f
        );
    }

    // =============================================================================
    // BACKWARD PASS
    // =============================================================================
    //
    // Compute gradients for all nodes in the computation graph by traversing the
    // pool in reverse order (which is a valid reverse topological order since
    // children are always created before parents).
    //
    // For each node, propagate its gradient to its children using the chain rule:
    //   child.grad += node.local_grads[c] * node.grad
    //
    // root_index: The pool index of the loss node (starting point for backward).

    void backward(NodeHandle root_index)
    {
        // Step 1: Reset all gradients to zero.
        for (Node & node : pool)
        {
            node.grad = 0.0f;
        }

        // Step 2: Seed the backward pass � the gradient of the loss with respect to itself is 1.
        Node & root_node = get(root_index);
        root_node.grad = 1.0f;

        // Step 3: Traverse the pool in reverse (reverse topological order).
        for (int i = pool.size() - 1; i >= 0; i--)
        {
            Node & node = get(i);
            const DataType incoming_grad = node.grad;
            const int num_children = node.n_children;

            // Propagate the gradient to each child.
            for (int c = 0; c < num_children; c++)
            {
                const float local_grad = node.local_grads[c];

                NodeHandle child_idx = node.children[c];
                Node & child = pool[child_idx];

                child.grad += local_grad * incoming_grad;
            }
        }
    }

private:
    std::vector<Node> pool;
    std::mt19937_64 rng{ 42 };
    std::uniform_real_distribution<float> uniform{ 0, 1 };
    std::normal_distribution<float> gaussian{ 0, 1 };
};

// activation function wrappers
template <typename DataType>
struct ReLU
{
    static float apply(AutoGrad<DataType> & ag, NodeHandle x)
    {
        return ag.value_relu(x);
    }
};
template <typename DataType>
struct GeLU
{
    static float apply(AutoGrad<DataType> & ag, NodeHandle x)
    {
        return ag.value_gelu(x);
    }
};
