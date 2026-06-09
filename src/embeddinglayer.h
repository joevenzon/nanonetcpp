#pragma once

#include "autograd.h"

#include <span>
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
    NodeMatrixHandle parameters; // weight matrix [num_embeddings x emb_dim]

    void init(AutoGrad<DataType> & grad, int num_embeddings, int emb_dim, DataType std_dev, const char * optional_name_hint = NULL)
    {
        parameters = grad.allocate_parameter_matrix(num_embeddings, emb_dim, DataType(0), std_dev,
            optional_name_hint ? optional_name_hint : "embedding");
    }

    // @param token_id  Integer index into the embedding table
    // Returns a tensor node of shape {emb_dim} — the embedding vector for that token.
    NodeHandle forward(AutoGrad<DataType> & grad, int token_id)
    {
        return grad.value_select_row(parameters.start, token_id);
    }
};
