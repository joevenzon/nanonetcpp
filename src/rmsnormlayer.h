#pragma once

#include "autograd.h"

#include <cassert>

// ---------------------------------------------------------------------------
// RMSNORM (ROOT MEAN SQUARE NORMALIZATION)
// ---------------------------------------------------------------------------
//
// Compute RMSNorm: scale each element by 1/sqrt(mean(x^2) + epsilon)
//
// This is a simplified version of LayerNorm without bias/weight parameters
// and without subtracting the mean. It normalizes by the root mean square.
//
// Formula: out[i] = x[i] / sqrt(mean(x^2) + epsilon)
//         = x[i] * (mean(x^2) + epsilon)^(-0.5)
//
// @param input_indices  Array of pool indices for the input vector
// @param output_indices Array to receive pool indices for the normalized output

template <typename DataType>
struct SimpleRMSNormLayer
{
    DataType epsilon = 1e-5f;

    // @param input_indices  Array of pool indices for the inputs
    // @param output_indices Array to receive pool indices for the softmax output
    void forward(
        AutoGrad<DataType> & grad,
        const std::span <NodeHandle> input_indices,
        std::span <NodeHandle> output_indices)
    {
        if (input_indices.empty()) return;

        // Assert input and output sizes match
        assert(input_indices.size() == output_indices.size());

        // Step 1: Compute sum of squares: sum(x[i]^2)
        NodeHandle sum_of_squares = grad.value_mul(input_indices[0], input_indices[0]);
        for (int i = 1; i < input_indices.size(); i++)
        {
            NodeHandle square = grad.value_mul(input_indices[i], input_indices[i]);
            sum_of_squares = grad.value_add(sum_of_squares, square);
        }

        // Step 2: Compute mean of squares: mean(x^2) = sum(x^2) / dim
        NodeHandle mean_of_squares = grad.value_div(sum_of_squares, grad.value_const(input_indices.size()));

        // Step 3: Add epsilon for numerical stability: mean(x^2) + epsilon
        NodeHandle mean_with_eps = grad.value_add_const(mean_of_squares, epsilon);

        // Step 4: Compute scale factor: (mean(x^2) + epsilon)^(-0.5)
        NodeHandle scale_factor = grad.value_pow(mean_with_eps, -0.5f);

        // Step 5: Scale each input element by the scale factor.
        for (int i = 0; i < input_indices.size(); i++)
        {
            output_indices[i] = grad.value_mul(input_indices[i], scale_factor);
        }
    }
};

// more complex version of rmsnorm that supports learnable scale & bias
template <typename DataType>
struct RMSNormLayer
{
    NodeMatrixHandle gamma;
    NodeMatrixHandle beta;
    DataType epsilon = 1e-5f;

    void init(AutoGrad<DataType> & grad, int dim, DataType std_dev)
    {
        gamma = grad.allocate_matrix(dim, 1, std_dev);
        beta = grad.allocate_matrix(dim, 1, std_dev);
    }

    // @param input_indices  Array of pool indices for the inputs
    // @param output_indices Array to receive pool indices for the output
    void forward(
        AutoGrad<DataType> & grad,
        const std::span <NodeHandle> input_indices,
        std::span <NodeHandle> output_indices)
    {
        if (input_indices.empty())
            return;

        // Assert dimension compatibility: input/output must match gamma and beta dimensions
        assert((int)input_indices.size() == gamma.rows);
        assert((int)output_indices.size() == gamma.rows);
        assert(input_indices.size() == output_indices.size());

        // Step 1: Compute sum of squares: sum(x[i]^2)
        NodeHandle sum_of_squares = grad.value_mul(input_indices[0], input_indices[0]);
        for (int i = 1; i < (int)input_indices.size(); i++)
        {
            NodeHandle square = grad.value_mul(input_indices[i], input_indices[i]);
            sum_of_squares = grad.value_add(sum_of_squares, square);
        }

        // Step 2: Compute mean of squares: mean(x^2) = sum(x^2) / dim
        NodeHandle mean_of_squares = grad.value_div(sum_of_squares, grad.value_const((float)input_indices.size()));

        // Step 3: Add epsilon for numerical stability: mean(x^2) + epsilon
        NodeHandle mean_with_eps = grad.value_add_const(mean_of_squares, epsilon);

        // Step 4: Compute scale factor: (mean(x^2) + epsilon)^(-0.5)
        NodeHandle rms_inv = grad.value_pow(mean_with_eps, -0.5f);

        // Step 5: Normalize, then apply per-element affine transform: gamma * x_norm + beta
        for (int i = 0; i < (int)input_indices.size(); i++)
        {
            NodeHandle normalized = grad.value_mul(input_indices[i], rms_inv);
            NodeHandle scaled = grad.value_mul(normalized, gamma.get(i));
            output_indices[i] = grad.value_add(scaled, beta.get(i));
        }
    }
};
