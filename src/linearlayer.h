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
    ParameterHandle parameters; // weight matrix W {in_dim, out_dim}

    void init(AutoGrad<DataType> & grad, int num_rows, int num_cols,
        DataType std_dev, const char * optional_name_hint = nullptr)
    {
        parameters = grad.allocate_parameter_matrix(num_rows, num_cols, DataType(0), std_dev,
            optional_name_hint ? optional_name_hint : "linear");
    }

    // `input` has shape {seq_len, in_dim}.  parameters is {in_dim, out_dim}.
    // Returns the output tensor node of shape {seq_len, out_dim}.
    NodeHandle forward(AutoGrad<DataType> & grad, NodeHandle input)
    {
        return grad.value_matmul(input, parameters.start);
    }
};
