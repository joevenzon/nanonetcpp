#pragma once

#include "autograd.h"

// ---------------------------------------------------------------------------
// SOFTMAX
// ---------------------------------------------------------------------------
//
// Compute the softmax of a vector of logits:
//   softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
//
// The subtraction of max(x) is a numerical stability trick to prevent
// overflow in the exponential function.

template <typename DataType>
struct SoftmaxLayer
{
    // @param input_indices  Array of pool indices for the inputs
    // @param output_indices Array to receive pool indices for the softmax output
    void forward(
        AutoGrad<DataType> & grad,
        const std::span <NodeHandle> input_indices,
        std::span <NodeHandle> output_indices)
    {
        if (input_indices.empty()) return;

        // Step 1: Find the maximum logit value for numerical stability.
        DataType max_value = grad.get(input_indices[0]).data;
        for (int i = 1; i < input_indices.size(); i++)
        {
            DataType current = grad.get(input_indices[i]).data;
            if (current > max_value)
            {
                max_value = current;
            }
        }

        // Step 2: Create a constant node for the maximum value.
        NodeHandle max_const = grad.value_const(max_value);

        // Step 3: Accumulate the sum of exp(x[i] - max) directly into a graph node.
        NodeHandle sum_exps = grad.value_exp(grad.value_sub(input_indices[0], max_const));
        for (int i = 1; i < input_indices.size(); i++)
        {
            sum_exps = grad.value_add(sum_exps, grad.value_exp(grad.value_sub(input_indices[i], max_const)));
        }

        // Step 4: Recompute each exp(x[i] - max) and divide by the sum.
        // value_sub reuses max_const (already in the pool), so no extra node for it.
        for (int i = 0; i < input_indices.size(); i++)
        {
            output_indices[i] = grad.value_div(
                grad.value_exp(grad.value_sub(input_indices[i], max_const)),
                sum_exps
            );
        }
    }
};