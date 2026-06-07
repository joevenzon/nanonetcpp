#pragma once

#include "autograd.h"

#include <cassert>

// ---------------------------------------------------------------------------
// EMBEDDING LAYER
// ---------------------------------------------------------------------------
//
// A lookup table mapping integer token IDs to dense vectors.
// The weight matrix is stored in row-major order: [num_embeddings x emb_dim]
//
// Forward pass is just selecting row token_id from the weight matrix;
// no multiply needed, since the "input" is an index, not a vector of values.
//
// out[j] = weights[token_id * emb_dim + j]

template <typename DataType>
struct EmbeddingLayer
{
    NodeHandle parameters;
    int emb_dim = 0;

    void init(AutoGrad<DataType> & grad, int num_embeddings, int emb_dim, DataType std_dev)
    {
        this->emb_dim = emb_dim;
        parameters = grad.allocate_matrix(num_embeddings, emb_dim, std_dev);
    }

    // @param token_id       Integer index into the embedding table
    // @param output_indices Array to receive pool indices for the output vector (size: emb_dim)
    void forward(
        AutoGrad<DataType> & grad,
        int token_id,
        std::span<NodeHandle> output_indices)
    {
        assert(emb_dim > 0); // ensure we got initialized

        NodeHandle row_start = parameters + token_id * emb_dim;
        for (int j = 0; j < emb_dim; j++)
        {
            output_indices[j] = row_start + j;
        }
    }
};
