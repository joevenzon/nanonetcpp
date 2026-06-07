#pragma once

#include "autograd.h"

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
    NodeHandle parameters;

	void init(AutoGrad<DataType> & grad, int num_rows, int num_cols, DataType std_dev)
	{
		parameters = grad.allocate_matrix(num_rows, num_cols, std_dev);
	}

    // @param input_indices  Array of pool indices for the input vector
    // @param output_indices Array to receive pool indices for the output vector
	void forward(
        AutoGrad<DataType> & grad,
		const std::span <NodeHandle> input_indices,
		std::span <NodeHandle> output_indices)
	{
        if (input_indices.empty()) return;

        for (int o = 0; o < output_indices.size(); o++)
        {
            // Pointer to the start of row o in the weight matrix.
            NodeHandle weight_row_start = parameters + o * input_indices.size();

            // Compute the dot product: w[o] * x
            // Start with the first term.
            NodeHandle accumulator = grad.value_mul(weight_row_start + 0, input_indices[0]);

            // Add remaining terms.
            for (int i = 1; i < input_indices.size(); i++)
            {
                NodeHandle product_term = grad.value_mul(weight_row_start + i, input_indices[i]);
                accumulator = grad.value_add(accumulator, product_term);
            }

            output_indices[o] = accumulator;
        }
	}
};