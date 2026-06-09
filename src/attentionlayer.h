#pragma once

#include "autograd.h"

#include <cassert>
#include <vector>

template <typename DataType>
struct AttentionLayer
{
    LinearLayer<DataType> wq, wk, wv;
    std::vector<NodeMatrixHandle> wo_heads; // one per head: {head_dim, emb_dim}
    int emb_dim, num_heads, head_dim;

    void init(AutoGrad<DataType> & ag, int emb_dim, int num_heads)
    {
        this->emb_dim = emb_dim;
        this->num_heads = num_heads;
        this->head_dim = emb_dim / num_heads;

        DataType std = 1.0 / std::sqrt(emb_dim);
        wq.init(ag, emb_dim, emb_dim, std, "wq");
        wk.init(ag, emb_dim, emb_dim, std, "wk");
        wv.init(ag, emb_dim, emb_dim, std, "wv");

        wo_heads.resize(num_heads);
        for (int h = 0; h < num_heads; h++)
        {
            char name[64];
            snprintf(name, sizeof(name), "wo_%d", h);
            wo_heads[h] = ag.allocate_parameter_matrix(head_dim, emb_dim, DataType(0), std, name);
        }
    }

    // Whole-sequence forward with causal masking.
    // input : tensor node of shape {seq_len, emb_dim}
    // returns: tensor node of shape {seq_len, emb_dim}
    NodeHandle forward(AutoGrad<DataType> & ag, NodeHandle input)
    {
        const AutoGrad<DataType>::Node & n_input = ag.get(input);
        assert(n_input.tensor.get_shape().rank() == 2);
        int seq_len = n_input.tensor.get_shape().dims[0];
        assert(n_input.tensor.get_shape().dims[1] == emb_dim);

        // -------------------------------------------------------------------
        // STEP 1: PROJECT TO Q, K, V  —  single matmul each
        // -------------------------------------------------------------------
        //   W: {emb_dim, emb_dim}, input: {seq_len, emb_dim}
        //   value_matmul expects (M×K) @ (K×N), so swap order: input @ W
        NodeHandle Q = ag.value_matmul(input, wq.parameters.start);
        NodeHandle K = ag.value_matmul(input, wk.parameters.start);
        NodeHandle V = ag.value_matmul(input, wv.parameters.start);

        // -------------------------------------------------------------------
        // STEP 2: BUILD CAUSAL MASK (constant leaf, no backward_fn)
        // -------------------------------------------------------------------
        // TODO: cache this somewhere (though note size is variable based on seq_len)
        NodeHandle mask = ag.tensor_leaf({ seq_len, seq_len }, DataType(0));
        {
            const std::span <DataType> & mv = ag.get(mask).tensor.values();
            for (int qi = 0; qi < seq_len; qi++)
                for (int ki = qi + 1; ki < seq_len; ki++)
                    mv[qi * seq_len + ki] = DataType(-1e9);
        }

        // -------------------------------------------------------------------
        // STEP 3: PER-HEAD ATTENTION + OUTPUT PROJECTION
        // -------------------------------------------------------------------
        DataType scale = DataType(1.0) / std::sqrt(head_dim);

        NodeHandle acc = {}; // set on first head, value_add'd for subsequent
        for (int h = 0; h < num_heads; h++)
        {
            NodeHandle Q_h = ag.value_slice_cols(Q, h * head_dim, head_dim);
            NodeHandle K_h = ag.value_slice_cols(K, h * head_dim, head_dim);
            NodeHandle V_h = ag.value_slice_cols(V, h * head_dim, head_dim);

            // scores = (Q_h @ K_h^T) * scale  —  {seq_len, seq_len}
            NodeHandle scores = ag.value_mul_const(
                ag.value_matmul_bt(Q_h, K_h), scale);

            // Causal mask + row-wise softmax
            NodeHandle weights = ag.value_softmax_rows(ag.value_add(scores, mask));

            // weighted values: {seq_len, seq_len} @ {seq_len, head_dim} -> {seq_len, head_dim}
            NodeHandle head_out = ag.value_matmul(weights, V_h);

            // project to emb_dim: {seq_len, head_dim} @ {head_dim, emb_dim} -> {seq_len, emb_dim}
            NodeHandle contrib = ag.value_matmul(head_out, wo_heads[h].start);

            acc = (h == 0) ? contrib : ag.value_add(acc, contrib);
        }

        return acc;
    }
};
