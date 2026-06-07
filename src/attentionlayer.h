#pragma once

#include "autograd.h"
#include "softmaxlayer.h"

template <typename DataType>
struct KVCache
{
    // These are values from previous forward passes, injected as constants when needed
    std::vector<DataType> keys;    // [max_seq_len x emb_dim]
    std::vector<DataType> values;  // [max_seq_len x emb_dim]
    int current_length = 0;
    int max_seq_len = 0;
    int emb_dim = 0;

    void init(int max_seq_len, int emb_dim)
    {
        this->max_seq_len = max_seq_len;
        this->emb_dim = emb_dim;
        keys.resize(max_seq_len * emb_dim, 0);
        values.resize(max_seq_len * emb_dim, 0);
    }

    void reset() { current_length = 0; }

    // Store the current key/value (extract raw floats from graph nodes)
    void store(AutoGrad<DataType> & ag,
        std::span<NodeHandle> key_nodes,
        std::span<NodeHandle> value_nodes)
    {
        int offset = current_length * emb_dim;
        for (int j = 0; j < emb_dim; j++)
        {
            keys[offset + j] = ag.get(key_nodes[j]).data;
            values[offset + j] = ag.get(value_nodes[j]).data;
        }
        current_length++;
    }

    // Inject a past key or value as a constant node into the graph
    NodeHandle get_key(AutoGrad<DataType> & ag, int pos, int j) const
    {
        return ag.value_const(keys[pos * emb_dim + j]);
    }

    NodeHandle get_value(AutoGrad<DataType> & ag, int pos, int j) const
    {
        return ag.value_const(values[pos * emb_dim + j]);
    }
};

template <typename DataType>
struct AttentionLayer
{
    LinearLayer<DataType> wq, wk, wv, wo;
    KVCache<DataType> cache;
    SoftmaxLayer<DataType> softmax;
    int emb_dim, num_heads, head_dim;

    void init(AutoGrad<DataType> & ag, int emb_dim, int num_heads, int max_seq_len)
    {
        this->emb_dim = emb_dim;
        this->num_heads = num_heads;
        this->head_dim = emb_dim / num_heads;

        DataType std = 1.0 / std::sqrt(emb_dim);
        wq.init(ag, emb_dim, emb_dim, std);
        wk.init(ag, emb_dim, emb_dim, std);
        wv.init(ag, emb_dim, emb_dim, std);
        wo.init(ag, emb_dim, emb_dim, std);

        cache.init(max_seq_len, emb_dim);
    }

    void reset_cache() { cache.reset(); }

    void forward(AutoGrad<DataType> & ag,
        std::span<NodeHandle> input,
        std::span<NodeHandle> output)
    {
        // -----------------------------------------------------------------------
        // STEP 1: PROJECT INPUT TO Q, K, V
        // -----------------------------------------------------------------------
        std::vector<NodeHandle> q(emb_dim), k(emb_dim), v(emb_dim);
        wq.forward(ag, input, q);
        wk.forward(ag, input, k);
        wv.forward(ag, input, v);

        // -----------------------------------------------------------------------
        // STEP 2: STORE CURRENT K, V INTO CACHE
        // -----------------------------------------------------------------------
        // Extract raw floats from the graph and store them outside it.
        // Past positions are injected back as constants during attention,
        // so they don't participate in backprop — only the current position does.
        cache.store(ag, k, v);
        int seq_len = cache.current_length;  // includes current position

        // -----------------------------------------------------------------------
        // STEP 3: PER-HEAD SCALED DOT-PRODUCT ATTENTION
        // -----------------------------------------------------------------------
        // scale factor: 1 / sqrt(head_dim), shared across all heads and positions
        NodeHandle scale = ag.value_const(1.0 / std::sqrt(head_dim));

        // output accumulator — heads write into their own slice, then we project
        std::vector<NodeHandle> concat_out(emb_dim);

        for (int h = 0; h < num_heads; h++)
        {
            int head_start = h * head_dim;

            // Slice of q belonging to this head
            std::span<NodeHandle> q_h(q.data() + head_start, head_dim);

            // -------------------------------------------------------------------
            // STEP 3a: COMPUTE ATTENTION SCORES
            // score[pos] = dot(q_h, k_pos_h) * scale
            // -------------------------------------------------------------------
            std::vector<NodeHandle> scores(seq_len);

            for (int pos = 0; pos < seq_len; pos++)
            {
                // Dot product of query with cached key at this position and head
                NodeHandle dot = ag.value_mul(
                    q_h[0],
                    cache.get_key(ag, pos, head_start + 0));

                for (int j = 1; j < head_dim; j++)
                {
                    NodeHandle prod = ag.value_mul(
                        q_h[j],
                        cache.get_key(ag, pos, head_start + j));
                    dot = ag.value_add(dot, prod);
                }

                scores[pos] = ag.value_mul(dot, scale);
            }

            // -------------------------------------------------------------------
            // STEP 3b: SOFTMAX OVER SCORES
            // -------------------------------------------------------------------
            std::vector<NodeHandle> attn_weights(seq_len);
            softmax.forward(ag, scores, attn_weights);

            // -------------------------------------------------------------------
            // STEP 3c: WEIGHTED SUM OF VALUES
            // out_h[j] = sum_pos( attn_weights[pos] * v_cache[pos][head_start + j] )
            // -------------------------------------------------------------------
            for (int j = 0; j < head_dim; j++)
            {
                NodeHandle weighted_sum = ag.value_mul(
                    attn_weights[0],
                    cache.get_value(ag, 0, head_start + j));

                for (int pos = 1; pos < seq_len; pos++)
                {
                    NodeHandle weighted = ag.value_mul(
                        attn_weights[pos],
                        cache.get_value(ag, pos, head_start + j));
                    weighted_sum = ag.value_add(weighted_sum, weighted);
                }

                // Write directly into the correct slice of the concatenated output
                concat_out[head_start + j] = weighted_sum;
            }
        }

        // -----------------------------------------------------------------------
        // STEP 4: OUTPUT PROJECTION
        // -----------------------------------------------------------------------
        // Project the concatenated head outputs back to emb_dim.
        wo.forward(ag, concat_out, output);
    }
};
