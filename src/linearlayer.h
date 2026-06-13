#pragma once

#include "autograd.h"

#include <cassert>

// ---------------------------------------------------------------------------
// LINEAR LAYER (matrix multiplication)
//   out = x @ W
// ---------------------------------------------------------------------------
template <typename DataType>
struct LinearLayer
{
    TensorHandle weights; // weight matrix W {in_dim, out_dim}
    TensorHandle bias;  // optional bias, shape {1, out_dim}
    bool use_bias = false;

    // pass zero into std_dev to use He/Kaiming gaussian initialization
    void init(AutoGrad<DataType> & grad, int num_rows, int num_cols,
        bool should_use_bias = true, DataType std_dev = 0, const char * optional_name_hint = nullptr)
    {
        DataType std_dev1 = std_dev > 0 ? std_dev : DataType(2) / std::sqrt(static_cast<DataType>(num_rows));
        weights = grad.allocate_parameter_matrix(num_rows, num_cols, DataType(0), std_dev1,
            optional_name_hint ? optional_name_hint : "linear");
        use_bias = should_use_bias;
        if (use_bias)
        {
            bias = grad.allocate_parameter_vector(num_cols, 0, 0, "linear_bias");
        }
    }

    // `input` has shape {seq_len, in_dim}.  weights is {in_dim, out_dim}.
    // Returns the output tensor node of shape {seq_len, out_dim}.
    TensorHandle forward(AutoGrad<DataType> & grad, TensorHandle input)
    {
        TensorHandle out = grad.value_matmul(input, weights);
        if (use_bias)
            return grad.value_add_rows(out, bias);
        else
            return out;
    }
};
