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

        DataType clip_sq = clip * clip;

        if (norm_sq > clip_sq)
        {
            DataType norm = std::sqrt(norm_sq);

            DataType scale = clip / norm;
            for (DataType & g : grads)
                g *= scale;
        }
    }

    void step(ParameterCheckpoint<DataType> & params)
    {
        params.step_count++;

        if (grad_clip > 0)
            clip_gradients(params.grads, grad_clip);

        // Bias correction terms
        DataType bc1 = 1 - std::pow(beta1, (DataType)params.step_count);
        DataType bc2 = 1 - std::pow(beta2, (DataType)params.step_count);

        DataType step_size = lr / bc1;          // replaces lr * m_hat/bc1
        DataType inv_sqrt_bc2 = DataType{ 1 } / std::sqrt(bc2);  // pulled out of loop

        for (int i = 0; i < params.size(); i++)
        {
            DataType g = params.grads[i];
            params.moment1[i] = beta1 * params.moment1[i] + (DataType{ 1 } - beta1) * g;
            params.moment2[i] = beta2 * params.moment2[i] + (DataType{ 1 } - beta2) * g * g;

            // v_hat = moment2[i] / bc2, sqrt(v_hat) = sqrt(moment2[i]) * inv_sqrt_bc2
            DataType denom = std::sqrt(params.moment2[i]) * inv_sqrt_bc2 + eps;
            params.values[i] -= step_size * params.moment1[i] / denom;
        }
    }
};
