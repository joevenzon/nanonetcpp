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
        init();
    }

    // =============================================================================
    // Node type
    // =============================================================================
    struct Node
    {
        static const int k_max_children = 3;
        static const int k_max_backward_scratch_values = 4;

        Tensor <DataType> tensor;

        PlainArray <TensorHandle, k_max_children> children; // Pool indices of up to k_max_children child nodes

        // Backward op: given access to the engine, propagate this node's grad
        // into the grad buffers of its children. Empty for leaves/constants.
        using BackwardFn = void(*)(AutoGrad &, const Node &);
        BackwardFn backward_fn;
        std::array <DataType, k_max_backward_scratch_values> backward_scratch;
        std::array <int, k_max_backward_scratch_values> backward_scratch_int;
    };

    struct LeafParameterRecord
    {
        int param_index;
        TensorShape shape;
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
    void init(unsigned long long seed = 42, size_t node_capacity = 8192, size_t value_capacity = 131072)
    {
        rng.seed(seed);
        nodes.resize(node_capacity);
        values.resize(value_capacity);
        grads.resize(value_capacity);
        leaf_parameters.clear();
    }

    Tensor <DataType> allocate_tensor(const TensorShape & shape)
    {
        size_t num = shape.numel();
        Tensor<DataType> tensor(values.allocate(num), grads.allocate(num), shape);
        std::fill(tensor.gradients().begin(), tensor.gradients().end(), DataType(0));
        return tensor;
    }

    // this also allocates the node's corresponding tensor
    TensorHandle allocate_node(const TensorShape & shape)
    {
        TensorHandle h(static_cast<int>(nodes.allocate()));
        Node & node = get(h);

        node.tensor = allocate_tensor(shape);
        node.children.clear();
        node.backward_fn = nullptr;

        return h; // return the index of the allocation
    }

    // =============================================================================
    // NODE CONSTRUCTORS
    // =============================================================================
    // Each constructor allocates a new node in the pool and initializes its fields.

    // A leaf tensor of the given shape, filled by `fill`.
    TensorHandle tensor_leaf(const TensorShape & shape, DataType fill = DataType(0))
    {
        TensorHandle h = allocate_node(shape);
        Node & node = get(h);

        const int n = shape.numel();
        for (int i = 0; i < n; i++) node.tensor.values()[i] = fill;

        return h;
    }

    // Scalar convenience (rank-1, single element)
    TensorHandle value_leaf(DataType data)
    {
        TensorHandle h = tensor_leaf({ 1 }, data);
        return h;
    }
    TensorHandle value_const(DataType data) { return value_leaf(data); }

    // Allocate a matrix of leaf nodes in the pool and return the starting offset & length.
    // @param num_rows    Number of rows in the matrix
    // @param num_cols    Number of columns in the matrix
    // @param mean        Initial value mean
    // @param std_dev     Standard deviation for Gaussian initialization, or zero to simply set the values to mean
    // @return            Pool index of the first element (top-left corner)
    TensorHandle allocate_parameter_matrix(int num_rows, int num_cols, DataType mean, DataType std_dev, const char * optional_name_hint = NULL)
    {
        TensorShape shape({ num_rows, num_cols });
        TensorHandle handle = allocate_node(shape);
        Node & node = get(handle);
        int total_elements = num_rows * num_cols;
        for (int k = 0; k < total_elements; k++)
        {
            if (std_dev == 0)
                node.tensor.values()[k] = mean;
            else
                node.tensor.values()[k] = rand_gaussian(mean, std_dev);
        }

        LeafParameterRecord record;
        record.param_index = &node.tensor.values()[0] - &values[0];
        record.shape = shape;
        if (optional_name_hint) record.name = optional_name_hint;
        leaf_parameters.push_back(record);
        return handle;
    }

    // Allocate a matrix of leaf nodes in the pool and return the starting offset & length.
    // @param num_elems   Number of elements in the vector
    // @param mean        Initial value mean
    // @param std_dev     Standard deviation for Gaussian initialization, or zero to simply set the values to mean
    // @return            Pool index of the first element (top-left corner)
    TensorHandle allocate_parameter_vector(int num_elems, DataType mean, DataType std_dev, const char * optional_name_hint = NULL)
    {
        TensorShape shape({ num_elems });
        TensorHandle handle = allocate_node(shape);
        Node & node = get(handle);
        for (int k = 0; k < num_elems; k++)
        {
            if (std_dev == 0)
                node.tensor.values()[k] = mean;
            else
                node.tensor.values()[k] = rand_gaussian(mean, std_dev);
        }

        LeafParameterRecord record;
        record.param_index = &node.tensor.values()[0] - &values[0];
        record.shape = shape;
        if (optional_name_hint) record.name = optional_name_hint;
        leaf_parameters.push_back(record);
        return handle;
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
    std::span <DataType> get_values() { return values.span(); }
    std::span <DataType> get_gradients() { return grads.span(); }

    size_t node_high_water_mark() const
    {
        return nodes.high_water_mark();
    }

    size_t value_high_water_mark() const
    {
        return values.high_water_mark();
    }

    std::span<const LeafParameterRecord> get_leaf_parameters() const { return std::span<const LeafParameterRecord>(leaf_parameters); }

    Node & get(TensorHandle index)
    {
        return nodes[index.get_node_index()];
    }

    void snapshot_parameters()
    {
        persistent_node_count = nodes.size();
        persistent_param_count = values.size();
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

    void restore_allocators()
    {
        nodes.reset_to(persistent_node_count);
        values.reset_to(persistent_param_count);
        grads.reset_to(persistent_param_count);
    }

    // =============================================================================
    // ELEMENTWISE OPS (require identical shapes; no broadcasting for brevity)
    // =============================================================================

    private:
        // Elementwise Node Helpers: these factor out common element-wise code for binary,
        // unary, and unary-plus-constant operations

        template<auto GradA, auto GradB>
        static void bwd_binary_ew(AutoGrad & g, const Node & out)
        {
            const DataType * go = out.tensor.gradients().data();
            const DataType * pa = g.get(out.children[0]).tensor.values().data();
            const DataType * pb = g.get(out.children[1]).tensor.values().data();
            DataType * ga = g.get(out.children[0]).tensor.gradients().data();
            DataType * gb = g.get(out.children[1]).tensor.gradients().data();
            const int n = out.tensor.numel();
            for (int i = 0; i < n; i++) {
                ga[i] += GradA(go[i], pa[i], pb[i]);
                gb[i] += GradB(go[i], pa[i], pb[i]);
            }
        }

        // Passes (go, in, out) so the lambda can use whichever it needs.
        // bwd_exp needs the output value; bwd_relu/log/erf need the input value.
        template<auto GradExpr>
        static void bwd_unary_ew(AutoGrad & g, const Node & out)
        {
            const DataType * go = out.tensor.gradients().data();
            const DataType * pa = g.get(out.children[0]).tensor.values().data();
            const DataType * po = out.tensor.values().data();
            DataType * ga = g.get(out.children[0]).tensor.gradients().data();
            const int n = out.tensor.numel();
            for (int i = 0; i < n; i++) ga[i] += GradExpr(go[i], pa[i], po[i]);
        }

        template<auto GradExpr>
        static void bwd_unary_const_ew(AutoGrad & g, const Node & out)
        {
            const DataType c = out.backward_scratch[0];
            const DataType * go = out.tensor.gradients().data();
            const DataType * pa = g.get(out.children[0]).tensor.values().data();
            DataType * ga = g.get(out.children[0]).tensor.gradients().data();
            const int n = out.tensor.numel();
            for (int i = 0; i < n; i++) ga[i] += GradExpr(go[i], pa[i], c);
        }

        template<auto FwdFn, auto GradA, auto GradB>
        TensorHandle make_binary_ew(TensorHandle a, TensorHandle b)
        {
            const TensorShape & shape = get(a).tensor.get_shape();
            assert(shape == get(b).tensor.get_shape());
            TensorHandle h = allocate_node(shape);
            Node & node = get(h);
            node.children.push_back(a);
            node.children.push_back(b);
            const DataType * pa = get(a).tensor.values().data();
            const DataType * pb = get(b).tensor.values().data();
            DataType * po = node.tensor.values().data();
            const int n = shape.numel();
            for (int i = 0; i < n; i++) po[i] = FwdFn(pa[i], pb[i]);
            node.backward_fn = &AutoGrad::bwd_binary_ew<GradA, GradB>;
            return h;
        }

        template<auto FwdFn, auto GradExpr>
        TensorHandle make_unary_ew(TensorHandle a)
        {
            const TensorShape & shape = get(a).tensor.get_shape();
            TensorHandle h = allocate_node(shape);
            Node & node = get(h);
            node.children.push_back(a);
            const DataType * pa = get(a).tensor.values().data();
            DataType * po = node.tensor.values().data();
            const int n = shape.numel();
            for (int i = 0; i < n; i++) po[i] = FwdFn(pa[i]);
            node.backward_fn = &AutoGrad::bwd_unary_ew<GradExpr>;
            return h;
        }

        template<auto FwdFn, auto GradExpr>
        TensorHandle make_unary_const_ew(TensorHandle a, DataType c)
        {
            const TensorShape & shape = get(a).tensor.get_shape();
            TensorHandle h = allocate_node(shape);
            Node & node = get(h);
            node.children.push_back(a);
            node.backward_scratch[0] = c;
            const DataType * pa = get(a).tensor.values().data();
            DataType * po = node.tensor.values().data();
            const int n = shape.numel();
            for (int i = 0; i < n; i++) po[i] = FwdFn(pa[i], c);
            node.backward_fn = &AutoGrad::bwd_unary_const_ew<GradExpr>;
            return h;
        }

public:

    TensorHandle value_add(TensorHandle a, TensorHandle b) {
        constexpr auto fwd = [](auto x, auto y) { return x + y; };
        constexpr auto gA = [](auto go, auto, auto) { return go; };
        constexpr auto gB = [](auto go, auto, auto) { return go; };
        return make_binary_ew<fwd, gA, gB>(a, b);
    }

    TensorHandle value_sub(TensorHandle a, TensorHandle b) {
        constexpr auto fwd = [](auto x, auto y) { return x - y; };
        constexpr auto gA = [](auto go, auto, auto) { return go; };
        constexpr auto gB = [](auto go, auto, auto) { return -go; };
        return make_binary_ew<fwd, gA, gB>(a, b);
    }

    // Elementwise (Hadamard) multiply.
    TensorHandle value_mul(TensorHandle a, TensorHandle b) {
        constexpr auto fwd = [](auto x, auto y) { return x * y; };
        constexpr auto gA = [](auto go, auto, auto b) { return go * b; };
        constexpr auto gB = [](auto go, auto a, auto) { return go * a; };
        return make_binary_ew<fwd, gA, gB>(a, b);
    }

    TensorHandle value_div(TensorHandle a, TensorHandle b) {
        constexpr auto fwd = [](auto x, auto y) { return x / y; };
        constexpr auto gA = [](auto go, auto, auto b) { return go / b; };
        constexpr auto gB = [](auto go, auto a, auto b) { return go * (-a / (b * b)); };
        return make_binary_ew<fwd, gA, gB>(a, b);
    }

    // =============================================================================
    // SCALAR-CONSTANT OPS
    // =============================================================================

    TensorHandle value_add_const(TensorHandle a, DataType c) {
        constexpr auto fwd = [](auto x, auto k) { return x + k; };
        constexpr auto g = [](auto go, auto, auto) { return go; };
        return make_unary_const_ew<fwd, g>(a, c);
    }

    TensorHandle value_sub_const(TensorHandle a, DataType c) {
        constexpr auto fwd = [](auto x, auto k) { return x - k; };
        constexpr auto g = [](auto go, auto, auto) { return go; };
        return make_unary_const_ew<fwd, g>(a, c);
    }

    TensorHandle value_mul_const(TensorHandle a, DataType c) {
        constexpr auto fwd = [](auto x, auto k) { return x * k; };
        constexpr auto g = [](auto go, auto, auto k) { return go * k; };
        return make_unary_const_ew<fwd, g>(a, c);
    }

    TensorHandle value_div_const(TensorHandle a, DataType c) {
        constexpr auto fwd = [](auto x, auto k) { return x / k; };
        constexpr auto g = [](auto go, auto, auto k) { return go / k; };
        return make_unary_const_ew<fwd, g>(a, c);
    }

    // =============================================================================
    // TILE OPS
    // =============================================================================

    // Tile a scalar (rank-1, single element) into a tensor of shape {N}.
    // Useful for broadcasting a scalar into shape-matched ops.
    static void bwd_tile_scalar(AutoGrad<DataType> & g, const Node & out)
    {
        const DataType * go = out.tensor.gradients().data();
        DataType * ga = g.get(out.children[0]).tensor.gradients().data();
        const int n = out.tensor.numel();
        for (int i = 0; i < n; i++) ga[0] += go[i];
    }

    TensorHandle value_tile_scalar(TensorHandle scalar, int n)
    {
        TensorHandle h = allocate_node(TensorShape{ n });
        Node & node = get(h);

        node.children.push_back(scalar);

        const DataType val = get(scalar).tensor.values().data()[0];
        DataType * po = node.tensor.values().data();
        for (int i = 0; i < n; i++) po[i] = val;

        node.backward_fn = &AutoGrad<DataType>::bwd_tile_scalar;

        return h;
    }

    // =============================================================================
    // SCALAR BROADCAST OPS (scalar is a TensorHandle of shape {1}, not a constant)
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

    TensorHandle value_mul_scalar(TensorHandle input, TensorHandle scalar)
    {
        const TensorShape& shape = get(input).tensor.get_shape();
        assert(get(scalar).tensor.numel() == 1);

        TensorHandle h = allocate_node(shape);
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

    TensorHandle value_div_scalar(TensorHandle input, TensorHandle scalar)
    {
        const TensorShape& shape = get(input).tensor.get_shape();
        assert(get(scalar).tensor.numel() == 1);

        TensorHandle h = allocate_node(shape);
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

    TensorHandle value_relu(TensorHandle a) {
        constexpr auto fwd = [](auto x) { return x > 0 ? x : decltype(x)(0); };
        constexpr auto g = [](auto go, auto x, auto) { return x > 0 ? go : decltype(go)(0); };
        return make_unary_ew<fwd, g>(a);
    }

    TensorHandle value_log(TensorHandle a) {
        constexpr auto fwd = [](auto x) { return std::log(x); };
        constexpr auto g = [](auto go, auto x, auto) { return go / x; };
        return make_unary_ew<fwd, g>(a);
    }

    TensorHandle value_exp(TensorHandle a) {
        constexpr auto fwd = [](auto x) { return std::exp(x); };
        constexpr auto g = [](auto go, auto, auto out) { return go * out; };
        return make_unary_ew<fwd, g>(a);
    }

    TensorHandle value_pow(TensorHandle a, DataType p) {
        constexpr auto fwd = [](auto x, auto k) { return std::pow(x, k); };
        constexpr auto g = [](auto go, auto x, auto k) { return go * k * std::pow(x, k - decltype(x)(1)); };
        return make_unary_const_ew<fwd, g>(a, p);
    }

    TensorHandle value_erf(TensorHandle a) {
        constexpr auto fwd = [](auto x) { return std::erf(x); };
        constexpr auto g = [](auto go, auto x, auto) {
            constexpr auto k = decltype(x)(1.1283791670955126); // 2/sqrt(pi)
            return go * k * std::exp(-x * x);
            };
        return make_unary_ew<fwd, g>(a);
    }

    // GELU via the exact erf formulation, composed from existing ops
    TensorHandle value_gelu(TensorHandle a)
    {
        static const DataType INV_SQRT2 = DataType(1) / std::sqrt(DataType(2));
        TensorHandle x_scaled = value_mul_const(a, INV_SQRT2);
        TensorHandle erf_val = value_erf(x_scaled);
        TensorHandle one_plus_erf = value_add_const(erf_val, DataType(1));
        return value_mul_const(value_mul(a, one_plus_erf), DataType(0.5));
    }

    // Tanh: tanh(x), gradient = 1 - tanh(x)^2
    TensorHandle value_tanh(TensorHandle a) {
        constexpr auto fwd = [](auto x) { return std::tanh(x); };
        constexpr auto g = [](auto go, auto x, auto tanh_x) {
            return go * (DataType(1) - tanh_x * tanh_x);
        };
        return make_unary_ew<fwd, g>(a);
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

    TensorHandle value_select_row(TensorHandle matrix, int row_index)
    {
        const Node & nm = get(matrix);
        assert(nm.tensor.get_shape().rank() == 2);
        const int num_cols = nm.tensor.get_shape().dims[1];
        assert(row_index >= 0 && row_index < nm.tensor.get_shape().dims[0]);

        TensorHandle h = allocate_node(TensorShape{ num_cols });
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

    TensorHandle value_slice_cols(TensorHandle matrix, int col_start, int count)
    {
        const Node & nm = get(matrix);
        assert(nm.tensor.get_shape().rank() == 2);
        const int M = nm.tensor.get_shape().dims[0];
        const int num_cols = nm.tensor.get_shape().dims[1];
        assert(col_start >= 0 && col_start + count <= num_cols);

        TensorHandle h = allocate_node(TensorShape{ M, count });
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

    TensorHandle value_scatter_row(TensorHandle dst, TensorHandle src, int row_index)
    {
        const Node & nd = get(dst);
        const Node & ns = get(src);
        assert(nd.tensor.get_shape().rank() == 2);
        assert(ns.tensor.get_shape().rank() == 1);
        const int M = nd.tensor.get_shape().dims[0];
        const int num_cols = nd.tensor.get_shape().dims[1];
        assert(ns.tensor.numel() == num_cols);
        assert(row_index >= 0 && row_index < M);

        TensorHandle h = allocate_node(TensorShape{ M, num_cols });
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

    TensorHandle value_scatter_cols(TensorHandle dst, TensorHandle src, int col_start)
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

        TensorHandle h = allocate_node(TensorShape{ M, N });
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

    TensorHandle value_select_element(TensorHandle a, int idx)
    {
        const Node & na = get(a);
        const int n = na.tensor.numel();
        assert(idx >= 0 && idx < n);

        TensorHandle h = allocate_node(TensorShape{ 1 });
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
    
    // generic version for numeric DataTypes
    static void bwd_matmul_general(AutoGrad<DataType> & g, const Node & out)
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
        {
            for (int k = 0; k < K; k++)
            {
                DataType acc = DataType(0);
                for (int n = 0; n < N; n++) acc += GC[m * N + n] * B[k * N + n];
                GA[m * K + k] += acc;
            }
        }
        for (int k = 0; k < K; k++)
        {
            for (int n = 0; n < N; n++)
            {
                DataType acc = DataType(0);
                for (int m = 0; m < M; m++) acc += A[m * K + k] * GC[m * N + n];
                GB[k * N + n] += acc;
            }
        }
    }

    TensorHandle value_matmul_general(TensorHandle a, TensorHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2 && nb.tensor.get_shape().rank() == 2);
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[0] == K);
        const int N = nb.tensor.get_shape().dims[1];

        TensorHandle h = allocate_node(TensorShape{ M, N });
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

        node.backward_fn = &AutoGrad<DataType>::bwd_matmul_general;

        return h;
    }

    // AVX2 version for floats
    
    // Helper: horizontal sum of 8 floats
    static inline float hsum256_ps(__m256 v)
    {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);                              // 4 partial sums
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));           // 2 partial sums
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));    // final sum in lane 0
        return _mm_cvtss_f32(lo);
    }

    static void bwd_matmul_float(AutoGrad<DataType> & g, const Node & out)
    {
        static_assert(std::is_same<DataType, float>::value, "AVX2 path assumes float");

        const Node & na = g.get(out.children[0]);
        const Node & nb = g.get(out.children[1]);
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        const int N = nb.tensor.get_shape().dims[1];

        const float * A = na.tensor.values().data();
        const float * B = nb.tensor.values().data();
        const float * GC = out.tensor.gradients().data();
        float * GA = g.get(out.children[0]).tensor.gradients().data(); // dL/dA = GC @ B^T  (M x K)
        float * GB = g.get(out.children[1]).tensor.gradients().data(); // dL/dB = A^T @ GC  (K x N)

        // ---- dL/dA: row-row dot products over n ----
        for (int m = 0; m < M; m++)
        {
            const float * GCrow = GC + m * N;
            for (int k = 0; k < K; k++)
            {
                const float * Brow = B + k * N;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                int n = 0;
                for (; n + 16 <= N; n += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(GCrow + n), _mm256_loadu_ps(Brow + n), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(GCrow + n + 8), _mm256_loadu_ps(Brow + n + 8), acc1);
                }
                for (; n + 8 <= N; n += 8)
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(GCrow + n), _mm256_loadu_ps(Brow + n), acc0);

                float acc = hsum256_ps(_mm256_add_ps(acc0, acc1));
                for (; n < N; n++) acc += GCrow[n] * Brow[n];

                GA[m * K + k] += acc;
            }
        }

        // ---- dL/dB: loops reordered (m outer) so n is the vectorized, contiguous axis ----
        for (int m = 0; m < M; m++)
        {
            const float * GCrow = GC + m * N;
            for (int k = 0; k < K; k++)
            {
                const float a_scalar = A[m * K + k];
                const __m256 a = _mm256_set1_ps(a_scalar);
                float * GBrow = GB + k * N;
                int n = 0;
                for (; n + 8 <= N; n += 8)
                    _mm256_storeu_ps(GBrow + n,
                        _mm256_fmadd_ps(a, _mm256_loadu_ps(GCrow + n), _mm256_loadu_ps(GBrow + n)));
                for (; n < N; n++) GBrow[n] += a_scalar * GCrow[n];
            }
        }
    }

    TensorHandle value_matmul_float(TensorHandle a, TensorHandle b)
    {
        static_assert(std::is_same<DataType, float>::value, "AVX2 path assumes float");

        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2 && nb.tensor.get_shape().rank() == 2);
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[0] == K);
        const int N = nb.tensor.get_shape().dims[1];

        TensorHandle h = allocate_node(TensorShape{ M, N });
        Node & node = get(h);

        node.children.push_back(a);
        node.children.push_back(b);

        const float * A = na.tensor.values().data();
        const float * B = nb.tensor.values().data();
        float * C = node.tensor.values().data();

        // C = A @ B, broadcast-FMA form: all loads/stores are row-contiguous
        for (int m = 0; m < M; m++)
        {
            float * Crow = C + m * N;

            // Zero the output row
            int n = 0;
            const __m256 zero = _mm256_setzero_ps();
            for (; n + 8 <= N; n += 8) _mm256_storeu_ps(Crow + n, zero);
            for (; n < N; n++) Crow[n] = 0.0f;

            for (int k = 0; k < K; k++)
            {
                const float a_scalar = A[m * K + k];
                const __m256 av = _mm256_set1_ps(a_scalar);
                const float * Brow = B + k * N;
                n = 0;
                for (; n + 8 <= N; n += 8)
                    _mm256_storeu_ps(Crow + n,
                        _mm256_fmadd_ps(av, _mm256_loadu_ps(Brow + n), _mm256_loadu_ps(Crow + n)));
                for (; n < N; n++) Crow[n] += a_scalar * Brow[n];
            }
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_matmul_float;

        return h;
    }

    TensorHandle value_matmul(TensorHandle a, TensorHandle b)
    {
        if constexpr (std::is_same_v<DataType, float>)
        {
            return value_matmul_float(a, b);
        }
        else
        {
            // Call the generic implementation for non-float types
            return value_matmul_general(a, b);
        }
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

    TensorHandle value_matmul_bt(TensorHandle a, TensorHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2 && nb.tensor.get_shape().rank() == 2);
        const int M = na.tensor.get_shape().dims[0];
        const int K = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[1] == K);
        const int N = nb.tensor.get_shape().dims[0];

        TensorHandle h = allocate_node(TensorShape{ M, N });
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
    // IM2COL
    //
    // Unfolds every k×k spatial patch of the input into a single row so that
    // the entire convolution reduces to one call to value_matmul.
    //
    // -- Shape contract
    //
    //   Input  : {batch * H_in * W_in, C_in}   (NHWC-fused rows)
    //   Output : {batch * H_out * W_out, C_in * k * k}
    //
    //   H_out = (H_in - k) / stride + 1   (valid convolution, no padding)
    //   W_out = (W_in - k) / stride + 1
    //
    // -- Column layout inside each output row
    //
    //   col = c * k*k + ph * k + pw
    //   where  c  -> [0, C_in)
    //          ph -> [0, k)   (kernel row)
    //          pw -> [0, k)   (kernel col)
    //
    //   This matches the weight matrix layout {C_in*k*k, C_out} used by Conv2dLayer.
    //
    // -- Backward (col2im)
    //
    //   Each upstream gradient element is scattered back to the input pixel that
    //   contributed to that (batch, out-row, out-col, channel, ph, pw) slot.
    //   Multiple patches may overlap the same input pixel (when stride < k), so
    //   contributions are accumulated with +=
    //
    // -- backward_scratch_int layout
    //
    //   [0] H_in   [1] W_in   [2] k (kernel_size)   [3] stride
    //   C_in is recovered from children[0].shape.dims[1] at no extra cost.
    // =============================================================================

    static void bwd_im2col(AutoGrad<DataType> & g, const Node & out)
    {
        Node & nx = g.get(out.children[0]);
        const int H_in = out.backward_scratch_int[0];
        const int W_in = out.backward_scratch_int[1];
        const int k = out.backward_scratch_int[2];
        const int stride = out.backward_scratch_int[3];
        const int C_in = nx.tensor.get_shape().dims[1];
        const int batch = nx.tensor.get_shape().dims[0] / (H_in * W_in);
        const int H_out = (H_in - k) / stride + 1;
        const int W_out = (W_in - k) / stride + 1;
        const int kk = k * k;

        const DataType * go = out.tensor.gradients().data(); // {batch*H_out*W_out, C_in*kk}
        DataType * gx = nx.tensor.gradients().data();  // {batch*H_in*W_in,   C_in}

        // Scatter each grad element back to the input pixel that produced it.
        // Overlapping patches (stride < k) accumulate via +=.
        for (int b = 0; b < batch; b++)
        {
            for (int oh = 0; oh < H_out; oh++)
            {
                for (int ow = 0; ow < W_out; ow++)
                {
                    const DataType * go_row = go + (b * H_out * W_out + oh * W_out + ow) * (C_in * kk);
                    for (int c = 0; c < C_in; c++)
                    {
                        for (int ph = 0; ph < k; ph++)
                        {
                            for (int pw = 0; pw < k; pw++)
                            {
                                const int ih = oh * stride + ph;
                                const int iw = ow * stride + pw;
                                const int in_px = (b * H_in * W_in + ih * W_in + iw) * C_in + c;
                                gx[in_px] += go_row[c * kk + ph * k + pw];
                            }
                        }
                    }
                }
            }
        }
    }

    // value_im2col — public forward function
    //
    //   x      : shape {batch * H_in * W_in, C_in}
    //   H_in   : height of each feature map
    //   W_in   : width  of each feature map
    //   k      : square kernel size
    //   stride : convolution stride (default 1)
    TensorHandle value_im2col(TensorHandle x, int H_in, int W_in, int k, int stride = 1)
    {
        const Node & nx = get(x);
        assert(nx.tensor.get_shape().rank() == 2);
        const int C_in = nx.tensor.get_shape().dims[1];
        const int total_in = nx.tensor.get_shape().dims[0];
        const int batch = total_in / (H_in * W_in);
        assert(batch * H_in * W_in == total_in && "Input rows must equal batch * H_in * W_in");
        assert(k > 0 && stride > 0 && H_in >= k && W_in >= k);

        const int H_out = (H_in - k) / stride + 1;
        const int W_out = (W_in - k) / stride + 1;
        const int kk = k * k;

        TensorHandle h = allocate_node(TensorShape{ batch * H_out * W_out, C_in * kk });
        Node & node = get(h);
        node.children.push_back(x);
        node.backward_scratch_int[0] = H_in;
        node.backward_scratch_int[1] = W_in;
        node.backward_scratch_int[2] = k;
        node.backward_scratch_int[3] = stride;

        const DataType * px = nx.tensor.values().data();
        DataType * po = node.tensor.values().data();

        // Copy each patch element into its row.
        for (int b = 0; b < batch; b++)
        {
            for (int oh = 0; oh < H_out; oh++)
            {
                for (int ow = 0; ow < W_out; ow++)
                {
                    DataType * po_row = po + (b * H_out * W_out + oh * W_out + ow) * (C_in * kk);
                    for (int c = 0; c < C_in; c++)
                    {
                        for (int ph = 0; ph < k; ph++)
                        {
                            for (int pw = 0; pw < k; pw++)
                            {
                                const int ih = oh * stride + ph;
                                const int iw = ow * stride + pw;
                                const int in_px = (b * H_in * W_in + ih * W_in + iw) * C_in + c;
                                po_row[c * kk + ph * k + pw] = px[in_px];
                            }
                        }
                    }
                }
            }
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_im2col;
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

    TensorHandle value_softmax_rows(TensorHandle a)
    {
        const TensorShape & shape = get(a).tensor.get_shape();
        assert(shape.rank() == 2);
        const int M = shape.dims[0], N = shape.dims[1];

        TensorHandle h = allocate_node(shape);
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
    // LOG-SOFTMAX ROWS: numerically stable log-softmax across columns, per row
    // Input:  X  {M, N}
    // Output: Y  {M, N}   Y[i,j] = X[i,j] - max_row_i - log( Σ_k exp(X[i,k] - max_row_i) )
    //
    // Equivalently: Y = log(softmax(X, dim=1)), but computed without materialising softmax
    // to avoid catastrophic cancellation.
    //
    // Backward:
    //   Let s[i,j] = softmax(X)[i,j] = exp(Y[i,j])  (recoverable from the output values)
    //   ∂L/∂X[i,k] = go[i,k]  −  s[i,k] · Σ_j go[i,j]
    // =============================================================================
    static void bwd_log_softmax_rows(AutoGrad<DataType> & g, const Node & out)
    {
        const int M = out.tensor.get_shape().dims[0];
        const int N = out.tensor.get_shape().dims[1];

        // out.tensor.values() holds log-probabilities Y[i,j]; exp(Y[i,j]) = softmax[i,j].
        const DataType * y = out.tensor.values().data();
        const DataType * go = out.tensor.gradients().data();
        DataType * gx = g.get(out.children[0]).tensor.gradients().data();

        for (int i = 0; i < M; i++)
        {
            const DataType * yi = y + i * N;
            const DataType * goi = go + i * N;
            DataType * gxi = gx + i * N;

            // Σ_j go[i,j]
            DataType sum_grad = DataType(0);
            for (int j = 0; j < N; j++) sum_grad += goi[j];

            // gx[i,k] += go[i,k] - exp(y[i,k]) * sum_grad
            for (int k = 0; k < N; k++)
                gxi[k] += goi[k] - std::exp(yi[k]) * sum_grad;
        }
    }

    TensorHandle value_log_softmax_rows(TensorHandle a)
    {
        const TensorShape & shape = get(a).tensor.get_shape();
        assert(shape.rank() == 2);
        const int M = shape.dims[0];
        const int N = shape.dims[1];

        TensorHandle h = allocate_node(shape);
        Node & node = get(h);
        node.children.push_back(a);

        const DataType * px = get(a).tensor.values().data();
        DataType * py = node.tensor.values().data();

        for (int i = 0; i < M; i++)
        {
            const DataType * row = px + i * N;
            DataType * out = py + i * N;

            // Numerically stable: subtract per-row max before exp.
            DataType mx = *std::max_element(row, row + N);

            DataType sum_exp = DataType(0);
            for (int j = 0; j < N; j++) sum_exp += std::exp(row[j] - mx);

            const DataType log_sum_exp = mx + std::log(sum_exp);   // log Σ exp(x_j)

            for (int j = 0; j < N; j++) out[j] = row[j] - log_sum_exp;
        }

        node.backward_fn = &AutoGrad<DataType>::bwd_log_softmax_rows;
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

    TensorHandle value_max(TensorHandle a)
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

        TensorHandle h = allocate_node(TensorShape{ 1 });
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

    TensorHandle value_sum(TensorHandle a)
    {
        TensorHandle h = allocate_node(TensorShape{ 1 });
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

    TensorHandle value_sum_rows(TensorHandle a)
    {
        const Node & na = get(a);
        assert(na.tensor.get_shape().rank() == 2);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];

        TensorHandle h = allocate_node(TensorShape{ M });
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

    TensorHandle value_scale_rows(TensorHandle a, TensorHandle v)
    {
        const Node & na = get(a);
        const Node & nv = get(v);
        assert(na.tensor.get_shape().rank() == 2);
        assert(nv.tensor.get_shape().rank() == 1);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];
        assert(nv.tensor.get_shape().dims[0] == M);

        TensorHandle h = allocate_node(TensorShape{ M, N });
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

    TensorHandle value_mul_rows(TensorHandle a, TensorHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2);
        assert(nb.tensor.get_shape().rank() == 1);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[0] == N);

        TensorHandle h = allocate_node(TensorShape{ M, N });
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

    TensorHandle value_add_rows(TensorHandle a, TensorHandle b)
    {
        const Node & na = get(a);
        const Node & nb = get(b);
        assert(na.tensor.get_shape().rank() == 2);
        assert(nb.tensor.get_shape().rank() == 1);
        const int M = na.tensor.get_shape().dims[0];
        const int N = na.tensor.get_shape().dims[1];
        assert(nb.tensor.get_shape().dims[0] == N);

        TensorHandle h = allocate_node(TensorShape{ M, N });
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

    void zero_grad()
    {
        // Zero all grads currently in use.
        std::fill(grads.span().begin(), grads.span().end(), DataType(0));
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
    void backward(TensorHandle root_index, bool zero_gradients = true)
    {
        if (zero_gradients)
            zero_grad();

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

        restore_allocators();
    }

private:
    Arena<Node> nodes;
    Arena<DataType> values;
    Arena<DataType> grads;

    int persistent_node_count{ 0 };
    int persistent_param_count{ 0 };
    std::mt19937_64 rng{ 42 };
    std::uniform_real_distribution<DataType> uniform{ 0, 1 };
    std::vector <LeafParameterRecord> leaf_parameters;
};

// activation function wrappers
template <typename DataType>
struct ReLU
{
    static TensorHandle apply(AutoGrad<DataType> & ag, TensorHandle x)
    {
        return ag.value_relu(x);
    }
};
template <typename DataType>
struct GeLU
{
    static TensorHandle apply(AutoGrad<DataType> & ag, TensorHandle x)
    {
        return ag.value_gelu(x);
    }
};
template <typename DataType>
struct Tanh
{
    static TensorHandle apply(AutoGrad<DataType> & ag, TensorHandle x)
    {
        return ag.value_tanh(x);
    }
};
