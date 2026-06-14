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

// ---------------------------------------------------------------------------
// AVX2 specializations for float
// ---------------------------------------------------------------------------

// Horizontal sum of an __m256 (8 floats)
static inline float avx2_hsum(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);                       // 4 partial sums
    __m128 shuf = _mm_movehdup_ps(lo);             // [1,1,3,3]
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

template <>
inline void AdamOptimizer<float>::clip_gradients(std::span<float> grads, float clip)
{
    const size_t n = grads.size();
    float * g = grads.data();

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    size_t i = 0;
    // Two accumulators to hide FMA latency, 16 floats per iteration
    for (; i + 16 <= n; i += 16)
    {
        __m256 a = _mm256_loadu_ps(g + i);
        __m256 b = _mm256_loadu_ps(g + i + 8);
        acc0 = _mm256_fmadd_ps(a, a, acc0);
        acc1 = _mm256_fmadd_ps(b, b, acc1);
    }
    for (; i + 8 <= n; i += 8)
    {
        __m256 a = _mm256_loadu_ps(g + i);
        acc0 = _mm256_fmadd_ps(a, a, acc0);
    }

    float norm_sq = avx2_hsum(_mm256_add_ps(acc0, acc1));

    // Scalar tail
    for (; i < n; i++)
        norm_sq += g[i] * g[i];

    const float clip_sq = clip * clip;
    if (norm_sq > clip_sq)
    {
        const float scale = clip / std::sqrt(norm_sq);
        const __m256 vscale = _mm256_set1_ps(scale);

        i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m256 v = _mm256_loadu_ps(g + i);
            v = _mm256_mul_ps(v, vscale);
            _mm256_storeu_ps(g + i, v);
        }
        for (; i < n; i++)
            g[i] *= scale;
    }
}

template <>
inline void AdamOptimizer<float>::step(AutoGrad<float> & ag)
{
    step_count++;

    std::span<float> values = ag.get_values();
    std::span<float> grads = ag.get_gradients();

    const size_t n = values.size();
    moment1.resize(n);
    moment2.resize(n);

    if (grad_clip > 0)
        clip_gradients(grads, grad_clip);

    const float bc1 = 1.0f - std::pow(beta1, (float)step_count);
    const float bc2 = 1.0f - std::pow(beta2, (float)step_count);

    const float step_size = lr / bc1;
    const float inv_sqrt_bc2 = 1.0f / std::sqrt(bc2);

    // Broadcast constants
    const __m256 vbeta1 = _mm256_set1_ps(beta1);
    const __m256 vbeta2 = _mm256_set1_ps(beta2);
    const __m256 vom_beta1 = _mm256_set1_ps(1.0f - beta1);
    const __m256 vom_beta2 = _mm256_set1_ps(1.0f - beta2);
    const __m256 veps = _mm256_set1_ps(eps);
    const __m256 vstep = _mm256_set1_ps(step_size);
    const __m256 vinv_bc2 = _mm256_set1_ps(inv_sqrt_bc2);

    float * vptr = values.data();
    float * gptr = grads.data();
    float * m1 = moment1.data();
    float * m2 = moment2.data();

    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 g = _mm256_loadu_ps(gptr + i);
        __m256 e1 = _mm256_loadu_ps(m1 + i);
        __m256 e2 = _mm256_loadu_ps(m2 + i);
        __m256 v = _mm256_loadu_ps(vptr + i);

        // moment1 = beta1*m1 + (1-beta1)*g
        e1 = _mm256_fmadd_ps(vbeta1, e1, _mm256_mul_ps(vom_beta1, g));

        // moment2 = beta2*m2 + (1-beta2)*g*g
        __m256 gg = _mm256_mul_ps(g, g);
        e2 = _mm256_fmadd_ps(vbeta2, e2, _mm256_mul_ps(vom_beta2, gg));

        // denom = sqrt(m2)*inv_sqrt_bc2 + eps
        __m256 denom = _mm256_fmadd_ps(_mm256_sqrt_ps(e2), vinv_bc2, veps);

        // values -= step_size * m1 / denom
        __m256 upd = _mm256_div_ps(_mm256_mul_ps(vstep, e1), denom);
        v = _mm256_sub_ps(v, upd);

        _mm256_storeu_ps(m1 + i, e1);
        _mm256_storeu_ps(m2 + i, e2);
        _mm256_storeu_ps(vptr + i, v);
    }

    // Scalar tail
    for (; i < n; i++)
    {
        float g = gptr[i];
        m1[i] = beta1 * m1[i] + (1.0f - beta1) * g;
        m2[i] = beta2 * m2[i] + (1.0f - beta2) * g * g;

        float denom = std::sqrt(m2[i]) * inv_sqrt_bc2 + eps;
        vptr[i] -= step_size * m1[i] / denom;
    }
}
