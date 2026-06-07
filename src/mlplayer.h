#pragma once

#include "linearlayer.h"

#include <cassert>

template <typename DataType, typename Activation>
struct MLPLayer
{
    LinearLayer<DataType> fc1;
    LinearLayer<DataType> fc2;
    int hidden_dim = 0;

    void init(AutoGrad<DataType> & ag, int in_dim, int hidden_dim)
    {
        this->hidden_dim = hidden_dim;
        fc1.init(ag, hidden_dim, in_dim, 1.0 / std::sqrt(in_dim));
        fc2.init(ag, in_dim, hidden_dim, 1.0 / std::sqrt(hidden_dim));
    }

    void forward(AutoGrad<DataType> & ag,
        std::span<NodeHandle> input,
        std::span<NodeHandle> output)
    {
        assert(hidden_dim > 0);

        std::vector<NodeHandle> hidden(hidden_dim);
        fc1.forward(ag, input, hidden);

        for (NodeHandle & n : hidden)
        {
            n = Activation::apply(ag, n);
        }

        fc2.forward(ag, hidden, output);
    }
};
