#pragma once

#include "autograd.h"

#include <cassert>

// ---------------------------------------------------------------------------
// LINEAR LAYER (MATRIX MULTIPLICATION)
// ---------------------------------------------------------------------------
//
// Compute: out[o] = sum_i (w[o][i] * x[i])
//
// The weight matrix is stored in row-major order starting at w_offset.
// Each output element is computed as a dot product of the weight row and input.

template <typename DataType>
struct LinearLayer
{
    NodeMatrixHandle parameters;

	void init(AutoGrad<DataType> & grad, int num_rows, int num_cols, DataType std_dev, const char * optional_name_hint = NULL)
	{
		parameters = grad.allocate_matrix(num_rows, num_cols, std_dev, optional_name_hint ? optional_name_hint : "linear");
	}

    // @param input_indices  Array of pool indices for the input vector
    // @param output_indices Array to receive pool indices for the output vector
    void forward(
        AutoGrad<DataType> & grad,
        const std::span <NodeHandle> input_indices,
        std::span <NodeHandle> output_indices)
    {
        if (input_indices.empty()) return;

        // Assert dimension compatibility: input size must match columns, output size must match rows
        assert((int)input_indices.size() == parameters.cols);
        assert((int)output_indices.size() == parameters.rows);

        for (int o = 0; o < output_indices.size(); o++)
        {
            // Compute the dot product: w[o] * x
            // Start with the first term.
            NodeHandle accumulator = grad.value_mul(parameters.get(o, 0), input_indices[0]);

            // Add remaining terms.
            for (int i = 1; i < input_indices.size(); i++)
            {
                NodeHandle product_term = grad.value_mul(parameters.get(o, i), input_indices[i]);
                accumulator = grad.value_add(accumulator, product_term);
            }

            output_indices[o] = accumulator;
        }
    }
};