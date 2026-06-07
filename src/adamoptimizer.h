#pragma once
#include "autograd.h"

#include "parametercheckpoint.h"

// ---------------------------------------------------------------------------
// ADAM OPTIMIZER
// ---------------------------------------------------------------------------
//
// Implements the Adam optimization algorithm (Kingma & Ba, 2014).
// Maintains first and second moment estimates of the gradients and uses
// them to adaptively scale the learning rate per parameter.
//
// Update rule:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   theta = theta - lr * m_hat / (sqrt(v_hat) + eps)

template <typename DataType>
struct AdamOptimizer
{
    DataType lr = 1e-3f;
    DataType beta1 = 0.9f;
    DataType beta2 = 0.999f;
    DataType eps = 1e-8f;
    DataType grad_clip = 0; // 0 to disable

    // Apply gradient clipping in-place (global norm clipping).
    // Scales all gradients down if the global L2 norm exceeds grad_clip.
    static void clip_gradients(std::span <DataType> grads, DataType clip)
    {
        DataType norm_sq = 0;
        for (DataType g : grads)
            norm_sq += g * g;

        DataType norm = std::sqrt(norm_sq);
        if (norm > clip)
        {
            DataType scale = clip / norm;
            for (DataType & g : grads)
                g *= scale;
        }
    }

    void step(ParameterCheckpoint<DataType> & params)
    {
        params.step_count++;

        if (grad_clip > 0.0f)
            clip_gradients(params.grads, grad_clip);

        // Bias correction terms
        DataType bc1 = 1.0f - std::pow(beta1, (DataType)params.step_count);
        DataType bc2 = 1.0f - std::pow(beta2, (DataType)params.step_count);

        for (int i = 0; i < params.size(); i++)
        {
            DataType g = params.grads[i];

            // Update moment estimates
            params.moment1[i] = beta1 * params.moment1[i] + (1.0f - beta1) * g;
            params.moment2[i] = beta2 * params.moment2[i] + (1.0f - beta2) * g * g;

            // bias-corrected estimates
            DataType m_hat = params.moment1[i] / bc1;
            DataType v_hat = params.moment2[i] / bc2;

            // update parameter value directly
            params.values[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
};
