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

    // `input` is a tensor node of shape {dim} (rank-1) or {M, dim} (rank-2).
    // Returns the normalized output tensor node of the same shape.
    //
    // Each row is normalized independently: out[i] = x[i] * (mean(x[i]^2) + epsilon)^(-0.5)
    TensorHandle forward(AutoGrad<DataType> & grad, TensorHandle input)
    {
        const auto & nin = grad.get(input);
        const int rank = nin.tensor.get_shape().rank();

        if (rank == 1)
        {
            // Single vector: use scalar ops
            const int dim = nin.tensor.numel();

            TensorHandle squares = grad.value_mul(input, input);
            TensorHandle sum_sq = grad.value_sum(squares);
            TensorHandle mean_sq = grad.value_div_const(sum_sq, DataType(dim));
            TensorHandle mean_eps = grad.value_add_const(mean_sq, epsilon);
            TensorHandle scale_factor = grad.value_pow(mean_eps, DataType(-0.5));

            return grad.value_mul_scalar(input, scale_factor);
        }
        else
        {
            // Batch of rows: use row-wise ops
            const int dim = nin.tensor.get_shape().dims[1];

            TensorHandle squares = grad.value_mul(input, input);
            TensorHandle sum_rows = grad.value_sum_rows(squares);
            TensorHandle mean_rows = grad.value_div_const(sum_rows, DataType(dim));
            TensorHandle mean_eps = grad.value_add_const(mean_rows, epsilon);
            TensorHandle scale_factor = grad.value_pow(mean_eps, DataType(-0.5));

            return grad.value_scale_rows(input, scale_factor);
        }
    }
};

// more complex version of rmsnorm that supports learnable scale & bias
template <typename DataType>
struct RMSNormLayer
{
    TensorHandle gamma;
    TensorHandle beta;
    DataType epsilon = 1e-5f;

    void init(AutoGrad<DataType> & grad, int dim, const char * optional_name_hint = NULL)
    {
        gamma = grad.allocate_parameter_vector(dim, 1, 0, optional_name_hint ? optional_name_hint : "rmsnorm_gamma");
        beta = grad.allocate_parameter_vector(dim, 0, 0, optional_name_hint ? optional_name_hint : "rmsnorm_beta");
    }

    // `input` is a tensor node of shape {dim} (rank-1) or {M, dim} (rank-2).
    // Returns the normalized + affine-transformed output tensor node of the same shape.
    //
    // Each row is normalized independently, then gamma/beta are applied per-element:
    //   out[i] = gamma * (x[i] / rms(x[i])) + beta
    TensorHandle forward(AutoGrad<DataType> & grad, TensorHandle input)
    {
        const auto & nin = grad.get(input);
        const int rank = nin.tensor.get_shape().rank();

        if (rank == 1)
        {
            // Single vector: use scalar ops
            const int dim = nin.tensor.numel();

            TensorHandle squares = grad.value_mul(input, input);
            TensorHandle sum_sq = grad.value_sum(squares);
            TensorHandle mean_sq = grad.value_div_const(sum_sq, DataType(dim));
            TensorHandle mean_eps = grad.value_add_const(mean_sq, epsilon);
            TensorHandle scale_factor = grad.value_pow(mean_eps, DataType(-0.5));
            TensorHandle normalized = grad.value_mul_scalar(input, scale_factor);

            // Affine transform: gamma * normalized + beta
            TensorHandle scaled = grad.value_mul(normalized, gamma);
            return grad.value_add(scaled, beta);
        }
        else
        {
            // Batch of rows: use row-wise ops
            const int dim = nin.tensor.get_shape().dims[1];

            TensorHandle squares = grad.value_mul(input, input);
            TensorHandle sum_rows = grad.value_sum_rows(squares);
            TensorHandle mean_rows = grad.value_div_const(sum_rows, DataType(dim));
            TensorHandle mean_eps = grad.value_add_const(mean_rows, epsilon);
            TensorHandle scale_factor = grad.value_pow(mean_eps, DataType(-0.5));
            TensorHandle normalized = grad.value_scale_rows(input, scale_factor);

            // Affine transform: gamma * normalized + beta
            // gamma.start and beta.start are nodes of shape {dim} (from allocate_parameter_vector).
            // Broadcast them across each row.
            TensorHandle scaled = grad.value_mul_rows(normalized, gamma);
            return grad.value_add_rows(scaled, beta);
        }
    }
};
