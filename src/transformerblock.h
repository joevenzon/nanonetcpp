#pragma once

#include "attentionlayer.h"
#include "mlplayer.h"
#include "rmsnormlayer.h"

#include <cassert>
#include <vector>

template <typename DataType>
struct TransformerBlock
{
    AttentionLayer<DataType> attention;
    MLPLayer<DataType, GeLU<DataType> > mlp;
    RMSNormLayer<DataType> norm1;
    RMSNormLayer<DataType> norm2;

    void init(AutoGrad<DataType> & ag, int emb_dim, int num_heads, int ffn_dim, float std_dev)
    {
        attention.init(ag, emb_dim, num_heads);
        mlp.init(ag, emb_dim, ffn_dim);
        norm1.init(ag, emb_dim, std_dev);
        norm2.init(ag, emb_dim, std_dev);
    }

    // input[t], output[t] : span of emb_dim node handles per token
    void forward(AutoGrad<DataType> & ag,
        std::span<const std::span<NodeHandle>> input,
        std::span<std::span<NodeHandle>> output)
    {
        int seq_len = (int)input.size();
        assert((int)output.size() == seq_len);
        int emb_dim = (int)input[0].size();

        // --- attention sub-block with residual ---
        // norm1 applied per token
        std::vector<std::vector<NodeHandle> > normed1(seq_len);
        std::vector<std::span<NodeHandle> >   normed1_spans(seq_len);
        for (int t = 0; t < seq_len; t++)
        {
            normed1[t].resize(emb_dim);
            norm1.forward(ag, input[t], normed1[t]);
            normed1_spans[t] = std::span<NodeHandle>(normed1[t].data(), emb_dim);
        }

        std::vector<std::vector<NodeHandle> > attn_out(seq_len);
        std::vector<std::span<NodeHandle> >   attn_out_spans(seq_len);
        for (int t = 0; t < seq_len; t++)
        {
            attn_out[t].resize(emb_dim);
            attn_out_spans[t] = std::span<NodeHandle>(attn_out[t].data(), emb_dim);
        }

        attention.forward(
            ag,
            std::span<const std::span<NodeHandle>>(normed1_spans.data(), seq_len),
            std::span<std::span<NodeHandle>>(attn_out_spans.data(), seq_len));

        // residual1[t] = input[t] + attn_out[t]
        std::vector<std::vector<NodeHandle> > residual1(seq_len);
        std::vector<std::span<NodeHandle> >   residual1_spans(seq_len);
        for (int t = 0; t < seq_len; t++)
        {
            residual1[t].resize(emb_dim);
            for (int j = 0; j < emb_dim; j++)
                residual1[t][j] = ag.value_add(input[t][j], attn_out[t][j]);
            residual1_spans[t] = std::span<NodeHandle>(residual1[t].data(), emb_dim);
        }

        // --- mlp sub-block with residual (per token) ---
        for (int t = 0; t < seq_len; t++)
        {
            std::vector<NodeHandle> normed2(emb_dim);
            norm2.forward(ag, residual1_spans[t], normed2);

            std::vector<NodeHandle> mlp_out(emb_dim);
            mlp.forward(ag, normed2, mlp_out);

            for (int j = 0; j < emb_dim; j++)
                output[t][j] = ag.value_add(residual1[t][j], mlp_out[j]);
        }
    }
};