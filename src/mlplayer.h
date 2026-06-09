#pragma once

#include "linearlayer.h"

#include <cassert>

// ---------------------------------------------------------------------------
// MLP LAYER (two fully-connected layers with activation)
//   hidden = Activation(W1 @ x)
//   out    = W2 @ hidden
// ---------------------------------------------------------------------------
template <typename DataType, typename Activation>
struct MLPLayer
{
    LinearLayer<DataType> fc1;
    LinearLayer<DataType> fc2;
    int hidden_dim = 0;

    void init(AutoGrad<DataType> & ag, int in_dim, int hidden_dim, const char * optional_name_hint = nullptr)
    {
        this->hidden_dim = hidden_dim;
        fc1.init(ag, in_dim, hidden_dim, 1.0 / std::sqrt(static_cast<DataType>(in_dim)),
            optional_name_hint ? optional_name_hint : "mlp_fc1");
        fc2.init(ag, hidden_dim, in_dim, 1.0 / std::sqrt(static_cast<DataType>(hidden_dim)),
            optional_name_hint ? optional_name_hint : "mlp_fc2");
    }

    // {seq_len, in_dim} -> {seq_len, in_dim}
    NodeHandle forward(AutoGrad<DataType> & ag, NodeHandle input)
    {
        assert(hidden_dim > 0);

        NodeHandle hidden = fc1.forward(ag, input);
        hidden = Activation::apply(ag, hidden);
        return fc2.forward(ag, hidden);
    }
};
