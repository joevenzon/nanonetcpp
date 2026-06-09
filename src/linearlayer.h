#pragma once

#include "autograd.h"

#include <cassert>

// ---------------------------------------------------------------------------
// LINEAR LAYER (matrix multiplication)
//   out = x @ W , where x is (seq_len x cols), W is (cols x rows), out is (seq_len x rows)
// ---------------------------------------------------------------------------
template <typename DataType>
struct LinearLayer
{
    NodeMatrixHandle parameters; // weight matrix W (rows x cols)

    void init(AutoGrad<DataType> & grad, int num_rows, int num_cols,
        DataType std_dev, const char * optional_name_hint = nullptr)
    {
        parameters = grad.allocate_parameter_matrix(num_rows, num_cols, DataType(0), std_dev,
            optional_name_hint ? optional_name_hint : "linear");
    }

    // `input` has shape {seq_len, cols}.  parameters is {cols, rows}.
    // Returns the output tensor node of shape {seq_len, rows}.
    NodeHandle forward(AutoGrad<DataType> & grad, NodeHandle input)
    {
        return grad.value_matmul(input, parameters.start);
    }
};
