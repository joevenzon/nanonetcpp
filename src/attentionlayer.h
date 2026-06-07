#pragma once

#include "autograd.h"
#include "softmaxlayer.h"

#include <cassert>
#include <vector>

template <typename DataType>
struct AttentionLayer
{
    LinearLayer<DataType> wq, wk, wv, wo;
    SoftmaxLayer<DataType> softmax;
    int emb_dim, num_heads, head_dim;

    void init(AutoGrad<DataType> & ag, int emb_dim, int num_heads)
    {
        this->emb_dim = emb_dim;
        this->num_heads = num_heads;
        this->head_dim = emb_dim / num_heads;

        DataType std = 1.0 / std::sqrt(emb_dim);
        wq.init(ag, emb_dim, emb_dim, std);
        wk.init(ag, emb_dim, emb_dim, std);
        wv.init(ag, emb_dim, emb_dim, std);
        wo.init(ag, emb_dim, emb_dim, std);
    }

    // Whole-sequence forward with causal masking.
    // input[t]  : sequence of emb_dim node handles for token t
    // output[t] : sequence of emb_dim node handles to write for token t
    void forward(AutoGrad<DataType> & ag,
        std::span<const std::span<NodeHandle> > input,
        std::span<std::span<NodeHandle> > output)
    {
        int seq_len = (int)input.size();
        assert((int)output.size() == seq_len);

        // -------------------------------------------------------------------
        // STEP 1: PROJECT EVERY POSITION TO Q, K, V (all live graph nodes)
        // -------------------------------------------------------------------
        // q[t], k[t], v[t] each hold emb_dim node handles.
        std::vector<std::vector<NodeHandle> > q(seq_len), k(seq_len), v(seq_len);
        for (int t = 0; t < seq_len; t++)
        {
            assert((int)input[t].size() == emb_dim);
            q[t].resize(emb_dim);
            k[t].resize(emb_dim);
            v[t].resize(emb_dim);
            wq.forward(ag, input[t], q[t]);
            wk.forward(ag, input[t], k[t]);
            wv.forward(ag, input[t], v[t]);
        }

        NodeHandle scale = ag.value_const(1.0 / std::sqrt(head_dim));

        // concat_out[t][emb_dim] : per-token concatenated head outputs
        std::vector<std::vector<NodeHandle>> concat_out(seq_len);
        for (int t = 0; t < seq_len; t++)
            concat_out[t].resize(emb_dim);

        // -------------------------------------------------------------------
        // STEP 2: PER-HEAD, PER-QUERY CAUSAL ATTENTION
        // -------------------------------------------------------------------
        for (int h = 0; h < num_heads; h++)
        {
            int head_start = h * head_dim;

            for (int qi = 0; qi < seq_len; qi++)
            {
                // Causal mask: query qi attends only to keys 0..qi.
                int num_keys = qi + 1;

                // --- scores[ki] = dot(q[qi], k[ki]) * scale  for ki <= qi ---
                std::vector<NodeHandle> scores(num_keys);
                for (int ki = 0; ki < num_keys; ki++)
                {
                    NodeHandle dot = ag.value_mul(
                        q[qi][head_start + 0],
                        k[ki][head_start + 0]);

                    for (int j = 1; j < head_dim; j++)
                    {
                        NodeHandle prod = ag.value_mul(
                            q[qi][head_start + j],
                            k[ki][head_start + j]);
                        dot = ag.value_add(dot, prod);
                    }
                    scores[ki] = ag.value_mul(dot, scale);
                }

                // --- softmax over the (causally masked) scores ---
                std::vector<NodeHandle> attn_weights(num_keys);
                softmax.forward(ag, scores, attn_weights);

                // --- weighted sum of values: out[qi][j] = sum_ki w[ki]*v[ki][j] ---
                for (int j = 0; j < head_dim; j++)
                {
                    NodeHandle weighted_sum = ag.value_mul(
                        attn_weights[0],
                        v[0][head_start + j]);

                    for (int ki = 1; ki < num_keys; ki++)
                    {
                        NodeHandle weighted = ag.value_mul(
                            attn_weights[ki],
                            v[ki][head_start + j]);
                        weighted_sum = ag.value_add(weighted_sum, weighted);
                    }
                    concat_out[qi][head_start + j] = weighted_sum;
                }
            }
        }

        // -------------------------------------------------------------------
        // STEP 3: OUTPUT PROJECTION (per position)
        // -------------------------------------------------------------------
        for (int t = 0; t < seq_len; t++)
        {
            std::span<NodeHandle> co(concat_out[t].data(), emb_dim);
            wo.forward(ag, co, output[t]);
        }
    }
};
