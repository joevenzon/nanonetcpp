#pragma once

#include "autograd.h"

#include <cassert>

// ---------------------------------------------------------------------------
// SOFTMAX
// ---------------------------------------------------------------------------
//
// Compute the softmax of a tensor of logits:
//   softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
//
// The subtraction of max(x) is a numerical stability trick to prevent
// overflow in the exponential function.

template <typename DataType>
struct SoftmaxLayer
{
    // `input` is a tensor node of shape {N} (or {N, 1}).
    // Returns the output tensor node of the same shape.
    NodeHandle forward(AutoGrad<DataType> & grad, NodeHandle input)
    {
        // Step 1: Find the maximum logit value for numerical stability.
        // Read the raw value directly — don't create a graph node, because max
        // is only used as a numerical-stability constant here, not as a differentiable op.
        const AutoGrad<DataType>::Node & in_node = grad.get(input);
        const DataType * pvalues = in_node.tensor.values().data();
        const int n = in_node.tensor.numel();
        DataType max_value = pvalues[0];
        for (int i = 1; i < n; i++)
        {
            if (pvalues[i] > max_value) max_value = pvalues[i];
        }

        // Step 2: Subtract max from all logits (elementwise, same shape).
        NodeHandle shifted = grad.value_sub_const(input, max_value);

        // Step 3: Exponentiate each shifted logit.
        NodeHandle exp_shifted = grad.value_exp(shifted);

        // Step 4: Sum the exponentials -> scalar {1}.
        NodeHandle sum_exps = grad.value_sum(exp_shifted);

        // Step 5: Divide by the scalar sum to normalize.
        // Uses value_div_scalar so the scalar isn't materialized as a tiled tensor.
        return grad.value_div_scalar(exp_shifted, sum_exps);
    }
};
