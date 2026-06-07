#pragma once

#include "attentionlayer.h"
#include "mlplayer.h"
#include "rmsnormlayer.h"

#include <cassert>

template <typename DataType>
struct TransformerBlock
{
    AttentionLayer<DataType> attention;
    MLPLayer<DataType, GeLU<DataType> > mlp;
    RMSNormLayer<DataType> norm1;
    RMSNormLayer<DataType> norm2;

    void init(AutoGrad<DataType> & ag, int emb_dim, int num_heads, int max_seq_len, int ffn_dim, float std_dev)
    {
        attention.init(ag, emb_dim, num_heads, max_seq_len);
        mlp.init(ag, emb_dim, ffn_dim);
        norm1.init(ag, emb_dim, std_dev);
        norm2.init(ag, emb_dim, std_dev);
    }

    void forward(AutoGrad<DataType> & ag,
        const std::span<NodeHandle> input,
        std::span<NodeHandle> output)
    {
        // Assert input and output sizes match
        assert(input.size() == output.size());

        // attention sub-block with residual
        std::vector<NodeHandle> normed1(input.size());
        norm1.forward(ag, input, normed1);

        std::vector<NodeHandle> attn_out(input.size());
        attention.forward(ag, normed1, attn_out);

        std::vector<NodeHandle> residual1(input.size());
        for (int j = 0; j < input.size(); j++)
            residual1[j] = ag.value_add(input[j], attn_out[j]);

        // mlp sub-block with residual
        std::vector<NodeHandle> normed2(input.size());
        norm2.forward(ag, residual1, normed2);

        std::vector<NodeHandle> mlp_out(input.size());
        mlp.forward(ag, normed2, mlp_out);

        for (int j = 0; j < input.size(); j++)
            output[j] = ag.value_add(residual1[j], mlp_out[j]);
    }

    void reset_kv_cache()
    {
        attention.reset_cache();
    }
};