#pragma once

#include "autograd.h"
#include "linearlayer.h"

#include <cassert>
#include <vector>

template <typename DataType>
struct AttentionLayer
{
    LinearLayer<DataType> wq, wk, wv, wo;
    int emb_dim, num_heads, head_dim;

    void init(AutoGrad<DataType> & ag, int emb_dim, int num_heads)
    {
        this->emb_dim = emb_dim;
        this->num_heads = num_heads;
        this->head_dim = emb_dim / num_heads;

        DataType std_dev = 1.0 / std::sqrt(emb_dim);
        wq.init(ag, emb_dim, emb_dim, false, std_dev, "wq");
        wk.init(ag, emb_dim, emb_dim, false, std_dev, "wk");
        wv.init(ag, emb_dim, emb_dim, false, std_dev, "wv");
        wo.init(ag, emb_dim, emb_dim, false, std_dev, "wo");
    }

    // Whole-sequence forward with causal masking.
    // input : tensor node of shape {seq_len, emb_dim}
    // returns: tensor node of shape {seq_len, emb_dim}
    TensorHandle forward(AutoGrad<DataType> & ag, TensorHandle input)
    {
        const typename AutoGrad<DataType>::Node & n_input = ag.get(input);
        assert(n_input.tensor.get_shape().rank() == 2);
        int seq_len = n_input.tensor.get_shape().dims[0];
        assert(n_input.tensor.get_shape().dims[1] == emb_dim);

        // -------------------------------------------------------------------
        // STEP 1: PROJECT TO Q, K, V (Standard Linear Projections)
        // -------------------------------------------------------------------
        // input: {seq_len, emb_dim}, weights: {emb_dim, emb_dim} -> {seq_len, emb_dim}
        TensorHandle Q = ag.value_matmul(input, wq.weights);
        TensorHandle K = ag.value_matmul(input, wk.weights);
        TensorHandle V = ag.value_matmul(input, wv.weights);

        // -------------------------------------------------------------------
        // STEP 2: BUILD CAUSAL MASK
        // -------------------------------------------------------------------
        TensorHandle mask = ag.tensor_leaf({ seq_len, seq_len }, DataType(0));
        {
            const std::span <DataType> & mv = ag.get(mask).tensor.values();
            for (int qi = 0; qi < seq_len; qi++)
                for (int ki = qi + 1; ki < seq_len; ki++)
                    mv[qi * seq_len + ki] = DataType(-1e9);
        }

        // -------------------------------------------------------------------
        // STEP 3: MULTI-HEAD ATTENTION (Concatenation Approach)
        // -------------------------------------------------------------------
        DataType scale = DataType(1.0) / std::sqrt(head_dim);

        // We create a "concatenated" buffer to hold the output of all heads side-by-side
        // Shape: {seq_len, emb_dim} (since num_heads * head_dim == emb_dim)
        TensorHandle concatenated = ag.tensor_leaf({ seq_len, emb_dim }, DataType(0));

        for (int h = 0; h < num_heads; h++)
        {
            // Slice the large Q, K, V into head-specific chunks: {seq_len, head_dim}
            TensorHandle Q_h = ag.value_slice_cols(Q, h * head_dim, head_dim);
            TensorHandle K_h = ag.value_slice_cols(K, h * head_dim, head_dim);
            TensorHandle V_h = ag.value_slice_cols(V, h * head_dim, head_dim);

            // Scaled Dot-Product: (Q_h @ K_h^T) * scale -> {seq_len, seq_len}
            TensorHandle scores = ag.value_mul_const(
                ag.value_matmul_bt(Q_h, K_h), scale);

            // Apply causal mask and row-wise softmax -> {seq_len, seq_len}
            TensorHandle weights = ag.value_softmax_rows(ag.value_add(scores, mask));

            // Weighted values: {seq_len, seq_len} @ {seq_len, head_dim} -> {seq_len, head_dim}
            TensorHandle head_out = ag.value_matmul(weights, V_h);

            // Place this head's output into its dedicated slot in the concatenated matrix.
            // concatenated: {seq_len, emb_dim}, head_out: {seq_len, head_dim}, start_col: h * head_dim
            concatenated = ag.value_scatter_cols(concatenated, head_out, h * head_dim);
        }

        // -------------------------------------------------------------------
        // STEP 4: FINAL OUTPUT PROJECTION
        // -------------------------------------------------------------------
        // In a standard Transformer, the concatenated heads are projected 
        // through one final weight matrix W_O.
        // {seq_len, emb_dim} @ {emb_dim, emb_dim} -> {seq_len, emb_dim}
        return wo.forward(ag, concatenated);
    }
};
