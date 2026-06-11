#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <span>
#include <cassert>
#include <string>
#include <algorithm>

#include "handle.h"
#include "arena.h"
#include "plainarray.h"
#include "tensor.h"

template <typename DataType>
class AutoGrad
{
public:
    AutoGrad()
    {
        nodes.resize(8192);
        values.resize(131072);
        grads.resize(131072);
    }

    // =============================================================================
    // Node type
    // =============================================================================
    struct Node
    {
        static const int k_max_children = 3;
        static const int k_max_backward_scratch_values = 4;

        Tensor <DataType> tensor;

        PlainArray <NodeHandle, k_max_children> children; // Pool indices of up to k_max_children child nodes

        // Backward op: given access to the engine, propagate this node's grad
        // into the grad buffers of its children. Empty for leaves/constants.
        using BackwardFn = void(*)(AutoGrad &, const Node &);
        BackwardFn backward_fn;
        std::array <DataType, k_max_backward_scratch_values> backward_scratch;
        std::array <int, k_max_backward_scratch_values> backward_scratch_int;
    };

    struct LeafMatrixRecord
    {
        NodeMatrixHandle matrix;
        std::string name;
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
    // ARENA MANAGEMENT
    // =============================================================================
    void init(size_t node_capacity, size_t value_capacity)
    {
        nodes.resize(node_capacity);
        values.resize(value_capacity);
        grads.resize(value_capacity);
        leaf_matrices.clear();
    }

    Tensor <DataType> allocate_tensor(const TensorShape & shape)
    {
        size_t num = shape.numel();
        return Tensor<DataType>(values.allocate(num), grads.allocate(num), shape);
    }

    // this also allocates the node's corresponding tensor
    NodeHandle allocate_node(const TensorShape & shape)
    {
        NodeHandle index = nodes.allocate();
        Node & node = nodes[index];

        node.tensor = allocate_tensor(shape);
        node.children.clear();
        node.backward_fn = nullptr;

        return index; // return the index of the allocation
    }

    // =============================================================================
    // NODE CONSTRUCTORS
    // =============================================================================
    // Each constructor allocates a new node in the pool and initializes its fields.

    // A leaf tensor of the given shape, filled by `fill`.
    NodeHandle tensor_leaf(const TensorShape & shape, DataType fill = DataType(0))
    {
        NodeHandle h = allocate_node(shape);
        Node & node = nodes[h];

        const int n = shape.numel();
        for (int i = 0; i < n; i++) node.tensor.values()[i] = fill;

        return h;
    }

    // Scalar convenience (rank-1, single element)
    NodeHandle value_leaf(DataType data)
    {
        NodeHandle h = tensor_leaf({ 1 }, data);
        return h;
    }
    NodeHandle value_const(DataType data) { return value_leaf(data); }

    // Allocate a matrix of leaf nodes in the pool and return the starting offset & length.
    // @param num_rows    Number of rows in the matrix
    // @param num_cols    Number of columns in the matrix
    // @param mean        Initial value mean
    // @param std_dev     Standard deviation for Gaussian initialization, or zero to simply set the values to mean
    // @return            Pool index of the first element (top-left corner)
    NodeMatrixHandle allocate_parameter_matrix(int num_rows, int num_cols, DataType mean, DataType std_dev, const char * optional_name_hint = NULL)
    {
        NodeHandle handle = allocate_node(TensorShape({ num_rows, num_cols }));
        Node & node = get(handle);
        int total_elements = num_rows * num_cols;
        for (int k = 0; k < total_elements; k++)
        {
            if (std_dev == 0)
                node.tensor.values()[k] = mean;
            else
                node.tensor.values()[k] = rand_gaussian(mean, std_dev);
        }

        NodeMatrixHandle result(handle, num_rows, num_cols);
        LeafMatrixRecord record;
        record.matrix = result;
        if (optional_name_hint) record.name = optional_name_hint;
        leaf_matrices.push_back(record);
        return result;
    }

    // Allocate a matrix of leaf nodes in the pool and return the starting offset & length.
    // @param num_elems   Number of elements in the vector
    // @param mean        Initial value mean
    // @param std_dev     Standard deviation for Gaussian initialization, or zero to simply set the values to mean
    // @return            Pool index of the first element (top-left corner)
    NodeMatrixHandle allocate_parameter_vector(int num_elems, DataType mean, DataType std_dev, const char * optional_name_hint = NULL)
    {
        NodeHandle handle = allocate_node(TensorShape({ num_elems }));
        Node & node = get(handle);
        for (int k = 0; k < num_elems; k++)
        {
            if (std_dev == 0)
                node.tensor.values()[k] = mean;
            else
                node.tensor.values()[k] = rand_gaussian(mean, std_dev);
        }

        NodeMatrixHandle result(handle, num_elems, 1);
        LeafMatrixRecord record;
        record.matrix = result;
        if (optional_name_hint) record.name = optional_name_hint;
        leaf_matrices.push_back(record);
        return result;
    }

    void reset()
    {
        nodes.reset();
        values.reset();
        grads.reset();
    }

    size_t nodes_used() const
    {
        return nodes.size();
    }

    size_t values_used() const
    {
        return values.size();
    }

    std::span <const DataType> get_values() const { return values.span(); }
    std::span <const DataType> get_gradients() const { return grads.span(); }

    size_t node_high_water_mark() const
    {
        return nodes.high_water_mark();
    }

    size_t value_high_water_mark() const
    {
        return values.high_water_mark();
    }

    std::span<const LeafMatrixRecord> get_leaf_matrices() const { return std::span<const LeafMatrixRecord>(leaf_matrices); }

    Node & get(NodeHandle index)
    {
        return nodes[index];
    }

    void snapshot_parameters()
    {
        persistent_node_count = nodes.size();
    }

    void restore_parameter_values(const std::span <DataType> newvalues, const std::span <DataType> newgradients)
    {
        assert(newvalues.size() <= values.size());
        assert(newgradients.size() <= grads.size());

        nodes.reset_to(persistent_node_count);

        for (int i = 0; i < newvalues.size(); i++)
        {
            values[i] = newvalues[i];
        }
        values.reset_to(newvalues.size());

        for (int i = 0; i < newgradients.size(); i++)
        {
            grads[i] = newgradients[i];
        }
        grads.reset_to(newgradients.size());
    }

    // =============================================================================
    // ELEMENTWISE OPS (require identical shapes; no broadcasting for brevity)
    // =============================================================================

    static void bwd_add(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        DataType * gb = g.get(out.children[1]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) { ga[i] += go[i]; gb[i] += go[i]; }
    }

    NodeHandle value_add(NodeHandle a, NodeHandle b)
    {
        const TensorShape& shape = get(a).tensor.get_shape();
        assert(shape == get(b).tensor.get_shape());

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * pa = get(a).tensor.values().data();
        const DataType * pb = get(b).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) { po[i] = pa[i] + pb[i]; }

        node.backward_fn = &AutoGrad<DataType>::bwd_add;

        return h;
    }

    static void bwd_sub(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        DataType * gb = g.get(out.children[1]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) { ga[i] += go[i]; gb[i] -= go[i]; }
    }

    NodeHandle value_sub(NodeHandle a, NodeHandle b)
    {
        const TensorShape& shape = get(a).tensor.get_shape();
        assert(shape == get(b).tensor.get_shape());

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * pa = get(a).tensor.values().data();
        const DataType * pb = get(b).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) { po[i] = pa[i] - pb[i]; }

        node.backward_fn = &AutoGrad<DataType>::bwd_sub;

        return h;
    }

    // Elementwise (Hadamard) multiply.
    static void bwd_mul(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        const DataType * pb = g.get(out.children[1]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        DataType * gb = g.get(out.children[1]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) { ga[i] += go[i] * pb[i]; gb[i] += go[i] * pa[i]; }
    }

    NodeHandle value_mul(NodeHandle a, NodeHandle b)
    {
        const TensorShape& shape = get(a).tensor.get_shape();
        assert(shape == get(b).tensor.get_shape());

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * pa = get(a).tensor.values().data();
        const DataType * pb = get(b).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) { po[i] = pa[i] * pb[i]; }

        node.backward_fn = &AutoGrad<DataType>::bwd_mul;

        return h;
    }

    static void bwd_div(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        const DataType * pb = g.get(out.children[1]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        DataType * gb = g.get(out.children[1]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) {
            const DataType inv = DataType(1) / pb[i];
            ga[i] += go[i] * inv;
            gb[i] += go[i] * (-pa[i] * inv * inv);
        }
    }

    NodeHandle value_div(NodeHandle a, NodeHandle b)
    {
        const TensorShape& shape = get(a).tensor.get_shape();
        assert(shape == get(b).tensor.get_shape());

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * pa = get(a).tensor.values().data();
        const DataType * pb = get(b).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) { po[i] = pa[i] / pb[i]; }

        node.backward_fn = &AutoGrad<DataType>::bwd_div;

        return h;
    }

    // =============================================================================
    // SCALAR-CONSTANT OPS
    // =============================================================================

    static void bwd_add_const(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go[i];
    }

    NodeHandle value_add_const(NodeHandle a, DataType c)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = pa[i] + c;

        node.backward_fn = &AutoGrad<DataType>::bwd_add_const;

        return h;
    }

    static void bwd_mul_const(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType c = out.backward_scratch[0];
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go[i] * c;
    }

    NodeHandle value_mul_const(NodeHandle a, DataType c)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        // Store constant in scratch for the backward pass
        node.backward_scratch[0] = c;

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = pa[i] * c;

        node.backward_fn = &AutoGrad<DataType>::bwd_mul_const;

        return h;
    }

    NodeHandle value_sub_const(NodeHandle a, DataType c)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = pa[i] - c;

        // backward is same as add_const: d(a-c)/da = 1
        node.backward_fn = &AutoGrad<DataType>::bwd_add_const;

        return h;
    }

    static void bwd_div_const(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType c = out.backward_scratch[0];
        const DataType inv_c = DataType(1) / c;
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go[i] * inv_c;
    }

    NodeHandle value_div_const(NodeHandle a, DataType c)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        // Store constant in scratch for the backward pass
        node.backward_scratch[0] = c;

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = pa[i] / c;

        node.backward_fn = &AutoGrad<DataType>::bwd_div_const;

        return h;
    }

    // Tile a scalar (rank-1, single element) into a tensor of shape {N}.
    // Useful for broadcasting a scalar into shape-matched ops.
    static void bwd_tile_scalar(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[0] += go[i];
    }

    NodeHandle value_tile_scalar(NodeHandle scalar, int n)
    {
        NodeHandle h = allocate_node(TensorShape{ n });
        Node & node = get(h);

        node.children.push_back(scalar);

        const DataType val = get(scalar).tensor.values().data()[0];
        DataType * po = node.tensor.values().data();
        for (int i = 0; i < n; i++) po[i] = val;

        node.backward_fn = &AutoGrad<DataType>::bwd_tile_scalar;

        return h;
    }

    // =============================================================================
    // SCALAR BROADCAST OPS (scalar is a NodeHandle of shape {1}, not a constant)
    // These avoid materializing a tiled tensor — the scalar value is read once
    // and applied across all elements. Gradients flow back to the scalar node.
    // =============================================================================

    static void bwd_mul_scalar(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType c = out.backward_scratch[0];
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        DataType * gc = g.get(out.children[1]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) {
            ga[i] += go[i] * c;       // d/d_input = grad_out * scalar
            gc[0] += go[i] * pa[i];   // d/d_scalar = sum(grad_out * input)
        }
    }

    NodeHandle value_mul_scalar(NodeHandle input, NodeHandle scalar)
    {
        const TensorShape& shape = get(input).tensor.get_shape();
        assert(get(scalar).tensor.numel() == 1);

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(input);
        node.children.push_back(scalar);
        node.backward_scratch[0] = get(scalar).tensor.values().data()[0];

        const DataType val = node.backward_scratch[0];
        const DataType * pa = get(input).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = pa[i] * val;

        node.backward_fn = &AutoGrad<DataType>::bwd_mul_scalar;

        return h;
    }

    static void bwd_div_scalar(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType c = out.backward_scratch[0];
        const DataType inv_c = DataType(1) / c;
        const DataType inv_c2 = inv_c * inv_c;
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        DataType * gc = g.get(out.children[1]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) {
            ga[i] += go[i] * inv_c;           // d/d_input = grad_out / scalar
            gc[0] += go[i] * (-pa[i] * inv_c2); // d/d_scalar = sum(grad_out * -input / scalar^2)
        }
    }

    NodeHandle value_div_scalar(NodeHandle input, NodeHandle scalar)
    {
        const TensorShape& shape = get(input).tensor.get_shape();
        assert(get(scalar).tensor.numel() == 1);

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(input);
        node.children.push_back(scalar);
        node.backward_scratch[0] = get(scalar).tensor.values().data()[0];

        const DataType val = node.backward_scratch[0];
        const DataType * pa = get(input).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = pa[i] / val;

        node.backward_fn = &AutoGrad<DataType>::bwd_div_scalar;

        return h;
    }

    // =============================================================================
    // UNARY ELEMENTWISE OPS
    // =============================================================================

    static void bwd_relu(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) {
            if (pa[i] > DataType(0)) ga[i] += go[i];
        }
    }

    NodeHandle value_relu(NodeHandle a)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) {
            po[i] = pa[i] > DataType(0) ? pa[i] : DataType(0);
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_relu;

        return h;
    }

    static void bwd_log(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go[i] * (DataType(1) / pa[i]);
    }

    NodeHandle value_log(NodeHandle a)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = std::log(pa[i]);

        node.backward_fn = &AutoGrad<DataType>::bwd_log;

        return h;
    }

    static void bwd_exp(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        const DataType * po = out.tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go[i] * po[i];
    }

    NodeHandle value_exp(NodeHandle a)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = std::exp(pa[i]);

        node.backward_fn = &AutoGrad<DataType>::bwd_exp;

        return h;
    }

    static void bwd_pow(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType p = out.backward_scratch[0];
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go[i] * p * std::pow(pa[i], p - DataType(1));
    }

    NodeHandle value_pow(NodeHandle a, DataType p)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        // Store exponent in scratch for the backward pass
        node.backward_scratch[0] = p;

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = std::pow(pa[i], p);

        node.backward_fn = &AutoGrad<DataType>::bwd_pow;

        return h;
    }

    static void bwd_erf(AutoGrad<DataType> & g, const Node & out)
    {
        static const DataType TWO_OVER_SQRT_PI = DataType(1.1283791670955126);
        const DataType * go = out.tensor.gradients().data();
        const DataType * pa = g.get(out.children[0]).tensor.values().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go[i] * TWO_OVER_SQRT_PI * std::exp(-pa[i] * pa[i]);
    }

    NodeHandle value_erf(NodeHandle a)
    {
        const TensorShape& shape = get(a).tensor.get_shape();

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = get(a).tensor.values().data();
        DataType * po = node.tensor.values().data();
        const int n = shape.numel();
        for (int i = 0; i < n; i++) po[i] = std::erf(pa[i]);

        node.backward_fn = &AutoGrad<DataType>::bwd_erf;

        return h;
    }

    // GELU via the exact erf formulation, composed from existing ops (works tensor-wide).
    NodeHandle value_gelu(NodeHandle a)
    {
        static const DataType INV_SQRT2 = DataType(1) / std::sqrt(DataType(2));
        NodeHandle x_scaled = value_mul_const(a, INV_SQRT2);
        NodeHandle erf_val = value_erf(x_scaled);
        NodeHandle one_plus_erf = value_add_const(erf_val, DataType(1));
        return value_mul_const(value_mul(a, one_plus_erf), DataType(0.5));
    }

    // =============================================================================
    // SELECT ROW: extract row `row_index` from an (M x N) matrix -> (N,)
    // =============================================================================
    static void bwd_select_row(AutoGrad<DataType> & g, const Node & out)
    {
        const int row_index = out.backward_scratch_int[0];
        Node & parent = g.get(out.children[0]);
        const int num_cols = parent.tensor.get_shape().dims[1];

        const DataType * go = out.tensor.gradients().data();
        DataType * ga = parent.tensor.gradients().data();
        for (int c = 0; c < num_cols; c++) ga[row_index * num_cols + c] += go[c];
    }

    NodeHandle value_select_row(NodeHandle matrix, int row_index)
    {
        const Node & nm = get(matrix);
        assert(nm.tensor.get_shape().rank() == 2);
        const int num_cols = nm.tensor.get_shape().dims[1];
        assert(row_index >= 0 && row_index < nm.tensor.get_shape().dims[0]);

        NodeHandle h = allocate_node(TensorShape{ num_cols });
        Node & node = get(h);

        node.children.push_back(matrix);
        node.backward_scratch_int[0] = row_index;

        const DataType * pm = nm.tensor.values().data();
        DataType * po = node.tensor.values().data();
        for (int c = 0; c < num_cols; c++) po[c] = pm[row_index * num_cols + c];

        node.backward_fn = &AutoGrad<DataType>::bwd_select_row;

        return h;
    }

    // =============================================================================
    // SLICE COLS: extract columns [col_start, col_start+count) from {M, N} -> {M, count}
    // =============================================================================
    static void bwd_slice_cols(AutoGrad<DataType> & g, const Node & out)
    {
        const int col_start = out.backward_scratch_int[0];
        const int count = out.backward_scratch_int[1];
        Node & parent = g.get(out.children[0]);
        const int num_cols = parent.tensor.get_shape().dims[1];

        const DataType * go = out.tensor.gradients().data();
        DataType * ga = parent.tensor.gradients().data();
        for (int m = 0; m < out.tensor.get_shape().dims[0]; m++)
            for (int c = 0; c < count; c++) ga[m * num_cols + col_start + c] += go[m * count + c];
    }

    NodeHandle value_slice_cols(NodeHandle matrix, int col_start, int count)
    {
        const Node & nm = get(matrix);
        assert(nm.tensor.get_shape().rank() == 2);
        const int M = nm.tensor.get_shape().dims[0];
        const int num_cols = nm.tensor.get_shape().dims[1];
        assert(col_start >= 0 && col_start + count <= num_cols);

        NodeHandle h = allocate_node(TensorShape{ M, count });
        Node & node = get(h);

        node.children.push_back(matrix);
        node.backward_scratch_int[0] = col_start;
        node.backward_scratch_int[1] = count;

        const DataType * pm = nm.tensor.values().data();
        DataType * po = node.tensor.values().data();
        for (int m = 0; m < M; m++)
            for (int c = 0; c < count; c++) po[m * count + c] = pm[m * num_cols + col_start + c];

        node.backward_fn = &AutoGrad<DataType>::bwd_slice_cols;

        return h;
    }

    // =============================================================================
    // SCATTER ROW: copy row `row_index` of `src` ({N}) into row `row_index` of `dst` ({M, N})
    //   Forward:  dst[row_index, :] = src
    //   Backward: d_src += d_dst[row_index, :]
    // =============================================================================
    static void bwd_scatter_row(AutoGrad<DataType> & g, const Node & out)
    {
        const int row_index = out.backward_scratch_int[0];
        const int num_cols = out.tensor.get_shape().dims[1];
        const int M = out.tensor.get_shape().dims[0];

        const DataType * go = out.tensor.gradients().data();
        DataType * gd = g.get(out.children[0]).tensor.gradients().data(); // dst
        DataType * gs = g.get(out.children[1]).tensor.gradients().data(); // src

        // dst grad = out grad everywhere EXCEPT the overwritten row
        for (int i = 0; i < M * num_cols; i++) gd[i] += go[i];
        for (int c = 0; c < num_cols; c++) {
            gd[row_index * num_cols + c] -= go[row_index * num_cols + c]; // remove overwritten row
            gs[c] += go[row_index * num_cols + c];
        }
    }

    NodeHandle value_scatter_row(NodeHandle dst, NodeHandle src, int row_index)
    {
        const Node & nd = get(dst);
        const Node & ns = get(src);
        assert(nd.tensor.get_shape().rank() == 2);
        assert(ns.tensor.get_shape().rank() == 1);
        const int M = nd.tensor.get_shape().dims[0];
        const int num_cols = nd.tensor.get_shape().dims[1];
        assert(ns.tensor.numel() == num_cols);
        assert(row_index >= 0 && row_index < M);

        NodeHandle h = allocate_node(TensorShape{ M, num_cols });
        Node & node = get(h);

        node.children.push_back(dst);
        node.children.push_back(src);
        node.backward_scratch_int[0] = row_index;

        const DataType * pd = nd.tensor.values().data();
        const DataType * ps = ns.tensor.values().data();
        DataType * po = node.tensor.values().data();

        // Copy dst
        for (int i = 0; i < M * num_cols; i++) po[i] = pd[i];
        // Overwrite row_index with src
        for (int c = 0; c < num_cols; c++) po[row_index * num_cols + c] = ps[c];

        node.backward_fn = &AutoGrad<DataType>::bwd_scatter_row;

        return h;
    }

    // =============================================================================
    // SCATTER COLS: copy `src` ({M, K}) into `dst` ({M, N}) starting at column `col_start`
    //   Forward:  dst[:, col_start : col_start+K] = src
    //   Backward: d_src += d_dst[:, col_start : col_start+K]
    // =============================================================================
    static void bwd_scatter_cols(AutoGrad<DataType> & g, const Node & out)
    {
        const int col_start = out.backward_scratch_int[0];
        const int K = out.backward_scratch_int[1];
        const int M = out.tensor.get_shape().dims[0];
        const int N = out.tensor.get_shape().dims[1];

        const DataType * go = out.tensor.gradients().data();
        DataType * gd = g.get(out.children[0]).tensor.gradients().data(); // dst
        DataType * gs = g.get(out.children[1]).tensor.gradients().data(); // src

        for (int m = 0; m < M; m++)
        {
            for (int c = 0; c < N; c++)
            {
                const bool overwritten = (c >= col_start && c < col_start + K);
                const DataType g_out = go[m * N + c];
                if (overwritten)
                    gs[m * K + (c - col_start)] += g_out; // goes to src
                else
                    gd[m * N + c] += g_out;                // goes to dst
            }
        }
    }

    NodeHandle value_scatter_cols(NodeHandle dst, NodeHandle src, int col_start)
    {
        const Node & nd = get(dst);
        const Node & ns = get(src);
        assert(nd.tensor.get_shape().rank() == 2);
        assert(ns.tensor.get_shape().rank() == 2);
        const int M = nd.tensor.get_shape().dims[0];
        const int N = nd.tensor.get_shape().dims[1];
        const int K = ns.tensor.get_shape().dims[1];
        assert(ns.tensor.get_shape().dims[0] == M);
        assert(col_start >= 0 && col_start + K <= N);

        NodeHandle h = allocate_node(TensorShape{ M, N });
        Node & node = get(h);

        node.children.push_back(dst);
        node.children.push_back(src);
        node.backward_scratch_int[0] = col_start;
        node.backward_scratch_int[1] = K;

        const DataType * pd = nd.tensor.values().data();
        const DataType * ps = ns.tensor.values().data();
        DataType * po = node.tensor.values().data();

        // Copy dst
        for (int i = 0; i < M * N; i++)
        {
            po[i] = pd[i];
        }
        // Overwrite columns [col_start, col_start+K) with src
        for (int m = 0; m < M; m++)
        {
            for (int c = 0; c < K; c++) po[m * N + col_start + c] = ps[m * K + c];
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_scatter_cols;

        return h;
    }

    // =============================================================================
    // SELECT ELEMENT: extract element at flat index `idx` -> scalar {1}
    // =============================================================================
    static void bwd_select_element(AutoGrad<DataType> & g, const Node & out)
    {
        const int idx = out.backward_scratch_int[0];
        const DataType go = out.tensor.gradients().data()[0];
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        ga[idx] += go;
    }

    NodeHandle value_select_element(NodeHandle a, int idx)
    {
        const Node & na = get(a);
        const int n = na.tensor.numel();
        assert(idx >= 0 && idx < n);

        NodeHandle h = allocate_node(TensorShape{ 1 });
        Node & node = get(h);

        node.children.push_back(a);
        node.backward_scratch_int[0] = idx;
        node.tensor.values().data()[0] = na.tensor.values().data()[idx];

        node.backward_fn = &AutoGrad<DataType>::bwd_select_element;

        return h;
    }

    // =============================================================================
    // MATMUL: (M x K) @ (K x N) -> (M x N)
    // =============================================================================
    static void bwd_matmul(AutoGrad<DataType> & g, const Node & out)
    {
        const Node & na = g.get(out.children[0]);
        const Node & nb = g.get(out.children[1]);
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        const int N = nb.tensor.get_shape().dims[1];

        const DataType * A = na.tensor.values().data();
        const DataType * B = nb.tensor.values().data();
        const DataType * GC = out.tensor.gradients().data();
        DataType * GA = g.get(out.children[0]).tensor.gradients().data(); // dL/dA = GC @ B^T  (M x K)
        DataType * GB = g.get(out.children[1]).tensor.gradients().data(); // dL/dB = A^T @ GC  (K x N)

        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                DataType acc = DataType(0);
                for (int n = 0; n < N; n++) acc += GC[m * N + n] * B[k * N + n];
                GA[m * K + k] += acc;
            }
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++) {
                DataType acc = DataType(0);
                for (int m = 0; m < M; m++) acc += A[m * K + k] * GC[m * N + n];
                GB[k * N + n] += acc;
            }
    }

    NodeHandle value_matmul(NodeHandle a, NodeHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2 && nb.tensor.get_shape().rank() == 2);
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[0] == K);
        const int N = nb.tensor.get_shape().dims[1];

        NodeHandle h = allocate_node(TensorShape{ M, N });
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * A = na.tensor.values().data();
        const DataType * B = nb.tensor.values().data();
        DataType * C = node.tensor.values().data();
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++) {
                DataType acc = DataType(0);
                for (int k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
                C[m * N + n] = acc;
            }
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_matmul;

        return h;
    }

    // =============================================================================
    // MATMUL BT: (M x K) @ (N x K)^T -> (M x N)  —  "matmul, B transposed"
    // Forward:  C[m,n] = Σ_k A[m,k] * B[n,k]
    // Backward: dA = dC @ B   (M×N) @ (N×K) -> (M×K)
    //           dB = dC^T @ A (N×M) @ (M×K) -> (N×K)
    // =============================================================================
    static void bwd_matmul_bt(AutoGrad<DataType> & g, const Node & out)
    {
        const Node & na = g.get(out.children[0]);  // A: {M, K}
        const Node & nb = g.get(out.children[1]);  // B: {N, K}
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        const int N = nb.tensor.get_shape().dims[0];

        const DataType * A  = na.tensor.values().data();
        const DataType * B  = nb.tensor.values().data();
        const DataType * GC = out.tensor.gradients().data();   // {M, N}
        DataType *       GA = g.get(out.children[0]).tensor.gradients().data();
        DataType *       GB = g.get(out.children[1]).tensor.gradients().data();

        // dA = dC @ B  (M×N) @ (N×K)
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                DataType acc = DataType(0);
                for (int n = 0; n < N; n++) acc += GC[m * N + n] * B[n * K + k];
                GA[m * K + k] += acc;
            }
        // dB = dC^T @ A  (N×M) @ (M×K)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) {
                DataType acc = DataType(0);
                for (int m = 0; m < M; m++) acc += GC[m * N + n] * A[m * K + k];
                GB[n * K + k] += acc;
            }
    }

    NodeHandle value_matmul_bt(NodeHandle a, NodeHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2 && nb.tensor.get_shape().rank() == 2);
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[1] == K);
        const int N = nb.tensor.get_shape().dims[0];

        NodeHandle h = allocate_node(TensorShape{ M, N });
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * A = na.tensor.values().data();
        const DataType * B = nb.tensor.values().data();
        DataType * C = node.tensor.values().data();
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                DataType acc = DataType(0);
                for (int k = 0; k < K; k++) acc += A[m * K + k] * B[n * K + k];
                C[m * N + n] = acc;
            }

        node.backward_fn = &AutoGrad<DataType>::bwd_matmul_bt;

        return h;
    }

    // =============================================================================
    // SOFTMAX ROWS: applies numerically stable softmax across the column dimension of each row independently
    // =============================================================================
    static void bwd_softmax_rows(AutoGrad<DataType> & g, const Node & out)
    {
        const int M = out.tensor.get_shape().dims[0];
        const int N = out.tensor.get_shape().dims[1];
        const DataType * s = out.tensor.values().data();
        const DataType * go = out.tensor.gradients().data();
        DataType * gx = g.get(out.children[0]).tensor.gradients().data();

        for (int i = 0; i < M; i++) {
            const DataType * si = s + i * N;
            const DataType * goi = go + i * N;
            DataType * gxi = gx + i * N;
            DataType dot = 0;
            for (int k = 0; k < N; k++) dot += goi[k] * si[k];
            for (int j = 0; j < N; j++) gxi[j] += si[j] * (goi[j] - dot);
        }
    }

    NodeHandle value_softmax_rows(NodeHandle a)
    {
        const TensorShape & shape = get(a).tensor.get_shape();
        assert(shape.rank() == 2);
        const int M = shape.dims[0], N = shape.dims[1];

        NodeHandle h = allocate_node(shape);
        Node & node = get(h);
        node.children.push_back(a);

        const DataType * px = get(a).tensor.values().data();
        DataType * py = node.tensor.values().data();

        for (int i = 0; i < M; i++) {
            const DataType * row = px + i * N;
            DataType * out = py + i * N;
            DataType mx = *std::max_element(row, row + N);
            DataType sum = 0;
            for (int j = 0; j < N; j++) { out[j] = std::exp(row[j] - mx); sum += out[j]; }
            for (int j = 0; j < N; j++) out[j] /= sum;
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_softmax_rows;
        return h;
    }

    // =============================================================================
    // REDUCTION: max element -> scalar (rank-1, single element)
    // =============================================================================
    static void bwd_max(AutoGrad<DataType> & g, const Node & out)
    {
        const int idx = out.backward_scratch_int[0];
        const DataType go = out.tensor.gradients().data()[0];
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        ga[idx] += go; // gradient only flows to the argmax element
    }

    NodeHandle value_max(NodeHandle a)
    {
        const Node & na = get(a);
        const DataType * pa = na.tensor.values().data();
        const int n = na.tensor.numel();

        // Find the index of the max element
        int max_idx = 0;
        DataType max_val = pa[0];
        for (int i = 1; i < n; i++) {
            if (pa[i] > max_val) { max_val = pa[i]; max_idx = i; }
        }

        NodeHandle h = allocate_node(TensorShape{ 1 });
        Node & node = get(h);

        node.children.push_back(a);
        node.backward_scratch_int[0] = max_idx;
        node.tensor.values().data()[0] = max_val;

        node.backward_fn = &AutoGrad<DataType>::bwd_max;

        return h;
    }

    // =============================================================================
    // REDUCTION: sum all elements -> scalar (rank-1, single element)
    // Useful for building a scalar loss to call backward() on.
    // =============================================================================
    static void bwd_sum(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType go = out.tensor.gradients().data()[0];
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = g.get(out.children[0]).tensor.numel();
        for (int i = 0; i < n; i++) ga[i] += go; // broadcast scalar grad
    }

    NodeHandle value_sum(NodeHandle a)
    {
        NodeHandle h = allocate_node(TensorShape{ 1 });
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = get(a).tensor.values().data();
        const int n = get(a).tensor.numel();
        DataType acc = DataType(0);
        for (int i = 0; i < n; i++) acc += pa[i];
        node.tensor.values().data()[0] = acc;

        node.backward_fn = &AutoGrad<DataType>::bwd_sum;

        return h;
    }

    // =============================================================================
    // ROW-WISE REDUCTION: sum each row -> return rank-1 {M}
    // Forward:  out[i] = sum(a[i*N .. (i+1)*N - 1])
    // Backward: ga[i*N + j] += go[i]   for all j  (broadcast row grad across columns)
    // =============================================================================
    static void bwd_sum_rows(AutoGrad<DataType> & g, const Node & out)
    {
        Node & na = g.get(out.children[0]);
        const int M = out.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];

        const DataType * go = out.tensor.gradients().data();
        DataType * ga = na.tensor.gradients().data();

        for (int i = 0; i < M; i++) {
            const DataType row_grad = go[i];
            for (int j = 0; j < N; j++) ga[i * N + j] += row_grad;
        }
    }

    NodeHandle value_sum_rows(NodeHandle a)
    {
        const Node & na = get(a);
        assert(na.tensor.get_shape().rank() == 2);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];

        NodeHandle h = allocate_node(TensorShape{ M });
        Node & node = get(h);

        node.children.push_back(a);

        const DataType * pa = na.tensor.values().data();
        DataType * po = node.tensor.values().data();
        for (int i = 0; i < M; i++) {
            DataType acc = DataType(0);
            for (int j = 0; j < N; j++) acc += pa[i * N + j];
            po[i] = acc;
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_sum_rows;

        return h;
    }

    // =============================================================================
    // ROW-WISE SCALE: A {M, N} * v {M} -> {M, N}
    // Each row m of A is multiplied by scalar v[m].
    // Forward:  out[m, n] = A[m, n] * v[m]
    // Backward:
    //   dA[m, n] += dOut[m, n] * v[m]
    //   dv[m]    += sum_n(dOut[m, n] * A[m, n])
    // =============================================================================
    static void bwd_scale_rows(AutoGrad<DataType> & g, const Node & out)
    {
        Node & na = g.get(out.children[0]);  // A: {M, N}
        Node & nv = g.get(out.children[1]);  // v: {M}
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];

        const DataType * A  = na.tensor.values().data();
        const DataType * V  = nv.tensor.values().data();
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = na.tensor.gradients().data();
        DataType * gv = nv.tensor.gradients().data();

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                ga[i * N + j] += go[i * N + j] * V[i];
                gv[i]         += go[i * N + j] * A[i * N + j];
            }
    }

    NodeHandle value_scale_rows(NodeHandle a, NodeHandle v)
    {
        const Node & na = get(a);
        const Node & nv = get(v);
        assert(na.tensor.get_shape().rank() == 2);
        assert(nv.tensor.get_shape().rank() == 1);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];
        assert(nv.tensor.get_shape().dims[0] == M);

        NodeHandle h = allocate_node(TensorShape{ M, N });
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(v);

        const DataType * pa = na.tensor.values().data();
        const DataType * pv = nv.tensor.values().data();
        DataType * po = node.tensor.values().data();

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) po[i * N + j] = pa[i * N + j] * pv[i];

        node.backward_fn = &AutoGrad<DataType>::bwd_scale_rows;

        return h;
    }

    // =============================================================================
    // ROW-WISE BROADCAST MUL: A {M, N} * b {N} -> {M, N}
    // Each row of A is elementwise-multiplied by b.
    // Forward:  out[i*N + j] = a[i*N + j] * b[j]
    // Backward: ga[i*N + j] += go[i*N + j] * b[j]
    //           gb[j]       += sum_i(go[i*N + j] * a[i*N + j])
    // =============================================================================
    static void bwd_mul_rows(AutoGrad<DataType> & g, const Node & out)
    {
        Node & na = g.get(out.children[0]);  // A: {M, N}
        Node & nb = g.get(out.children[1]);  // b: {N}
        const int M = na.tensor.get_shape().dims[0];
        const int N = nb.tensor.get_shape().dims[0];

        const DataType * A  = na.tensor.values().data();
        const DataType * B  = nb.tensor.values().data();
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = na.tensor.gradients().data();
        DataType * gb = nb.tensor.gradients().data();

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                ga[i * N + j] += go[i * N + j] * B[j];
                gb[j]         += go[i * N + j] * A[i * N + j];
            }
    }

    NodeHandle value_mul_rows(NodeHandle a, NodeHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2);
        assert(nb.tensor.get_shape().rank() == 1);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[0] == N);

        NodeHandle h = allocate_node(TensorShape{ M, N });
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * pa = na.tensor.values().data();
        const DataType * pb = nb.tensor.values().data();
        DataType * po = node.tensor.values().data();

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) po[i * N + j] = pa[i * N + j] * pb[j];

        node.backward_fn = &AutoGrad<DataType>::bwd_mul_rows;

        return h;
    }

    // =============================================================================
    // ROW-WISE BROADCAST ADD: A {M, N} + b {N} -> {M, N}
    // Each row of A is elementwise-added to b.
    // Forward:  out[i*N + j] = a[i*N + j] + b[j]
    // Backward: ga[i*N + j] += go[i*N + j]
    //           gb[j]       += sum_i(go[i*N + j])
    // =============================================================================
    static void bwd_add_rows(AutoGrad<DataType> & g, const Node & out)
    {
        Node & na = g.get(out.children[0]);  // A: {M, N}
        Node & nb = g.get(out.children[1]);  // b: {N}
        const int M = na.tensor.get_shape().dims[0];
        const int N = nb.tensor.get_shape().dims[0];

        const DataType * go = out.tensor.gradients().data();
        DataType * ga = na.tensor.gradients().data();
        DataType * gb = nb.tensor.gradients().data();

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                ga[i * N + j] += go[i * N + j];
                gb[j]         += go[i * N + j];
            }
    }

    NodeHandle value_add_rows(NodeHandle a, NodeHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2);
        assert(nb.tensor.get_shape().rank() == 1);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[0] == N);

        NodeHandle h = allocate_node(TensorShape{ M, N });
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const DataType * pa = na.tensor.values().data();
        const DataType * pb = nb.tensor.values().data();
        DataType * po = node.tensor.values().data();

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) po[i * N + j] = pa[i * N + j] + pb[j];

        node.backward_fn = &AutoGrad<DataType>::bwd_add_rows;

        return h;
    }

    // =============================================================================
    // BACKWARD PASS
    // =============================================================================
    //
    // The pool is in reverse-topological order (children created before parents),
    // so iterating the node pool backward and invoking each node's backward_fn
    // yields correct accumulation. We zero the grad arena, seed the root with 1,
    // then scatter.
    //
    void backward(NodeHandle root_index)
    {
        // Zero all grads currently in use.
        std::fill(grads.span().begin(), grads.span().end(), DataType(0));

        // Seed: dLoss/dRoot = 1 over the root tensor (typically a scalar).
        {
            Node & root = get(root_index);
            DataType * gr = root.tensor.gradients().data();
            const int n = root.tensor.numel();
            for (int i = 0; i < n; i++) gr[i] = DataType(1);
        }

        // Reverse-topological scatter.
        for (int i = (int)nodes.size() - 1; i >= 0; i--)
        {
            const Node & node = nodes[i];
            if (node.backward_fn) node.backward_fn(*this, node);
        }
    }

private:
    Arena<Node> nodes;
    Arena<DataType> values;
    Arena<DataType> grads;

    int persistent_node_count{ 0 };
    std::mt19937_64 rng{ 42 };
    std::uniform_real_distribution<DataType> uniform{ 0, 1 };
    std::vector <LeafMatrixRecord> leaf_matrices;
};

// activation function wrappers
template <typename DataType>
struct ReLU
{
    static NodeHandle apply(AutoGrad<DataType> & ag, NodeHandle x)
    {
        return ag.value_relu(x);
    }
};
template <typename DataType>
struct GeLU
{
    static NodeHandle apply(AutoGrad<DataType> & ag, NodeHandle x)
    {
        return ag.value_gelu(x);
    }
};
