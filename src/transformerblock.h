#pragma once

#include "attentionlayer.h"
#include "mlplayer.h"
#include "rmsnormlayer.h"

#include <cassert>

template <typename DataType>
struct TransformerBlock
{
    AttentionLayer<DataType> attention;
    MLPLayer<DataType, GeLU<DataType>> mlp;
    RMSNormLayer<DataType> norm1;
    RMSNormLayer<DataType> norm2;

    void init(AutoGrad<DataType> & ag, int emb_dim, int num_heads, int ffn_dim, float std_dev)
    {
        attention.init(ag, emb_dim, num_heads);
        mlp.init(ag, emb_dim, ffn_dim, "transformer_mlp");
        norm1.init(ag, emb_dim, "transformer_norm1");
        norm2.init(ag, emb_dim, "transformer_norm2");
    }

    // input  : tensor node of shape {seq_len, emb_dim}
    // returns: tensor node of shape {seq_len, emb_dim}
    NodeHandle forward(AutoGrad<DataType> & ag, NodeHandle input)
    {
        // --- attention sub-block with residual ---
        NodeHandle normed1 = norm1.forward(ag, input);
        NodeHandle attn_out = attention.forward(ag, normed1);
        NodeHandle residual1 = ag.value_add(input, attn_out);

        // --- mlp sub-block with residual ---
        NodeHandle normed2 = norm2.forward(ag, residual1);
        NodeHandle mlp_out = mlp.forward(ag, normed2);
        NodeHandle output = ag.value_add(residual1, mlp_out);

        return output;
    }
};
