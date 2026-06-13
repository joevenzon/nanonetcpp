#pragma once

#include "autograd.h"

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

    size_t step_count = 0;
    std::vector<DataType> moment1;		// Adam first moment
    std::vector<DataType> moment2;		// Adam second moment

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

    void step(AutoGrad<DataType> & ag)
    {
        step_count++;

        std::span <DataType> values = ag.get_values();
        std::span <DataType> grads = ag.get_gradients();

        moment1.resize(values.size());
        moment2.resize(values.size());

        if (grad_clip > 0)
            clip_gradients(grads, grad_clip);

        // Bias correction terms
        DataType bc1 = 1 - std::pow(beta1, (DataType)step_count);
        DataType bc2 = 1 - std::pow(beta2, (DataType)step_count);

        DataType step_size = lr / bc1;          // replaces lr * m_hat/bc1
        DataType inv_sqrt_bc2 = DataType{ 1 } / std::sqrt(bc2);  // pulled out of loop

        for (int i = 0; i < values.size(); i++)
        {
            DataType g = grads[i];
            moment1[i] = beta1 * moment1[i] + (DataType{ 1 } - beta1) * g;
            moment2[i] = beta2 * moment2[i] + (DataType{ 1 } - beta2) * g * g;

            // v_hat = moment2[i] / bc2, sqrt(v_hat) = sqrt(moment2[i]) * inv_sqrt_bc2
            DataType denom = std::sqrt(moment2[i]) * inv_sqrt_bc2 + eps;
            values[i] -= step_size * moment1[i] / denom;
        }
    }
};
