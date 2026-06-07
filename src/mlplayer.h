#pragma once

#include "linearlayer.h"

#include <cassert>

template <typename DataType, typename Activation>
struct MLPLayer
{
    LinearLayer<DataType> fc1;
    LinearLayer<DataType> fc2;
    int hidden_dim = 0;

    void init(AutoGrad<DataType> & ag, int in_dim, int hidden_dim, const char * optional_name_hint = NULL)
    {
        this->hidden_dim = hidden_dim;
        fc1.init(ag, hidden_dim, in_dim, 1.0 / std::sqrt(in_dim), optional_name_hint);
        fc2.init(ag, in_dim, hidden_dim, 1.0 / std::sqrt(hidden_dim), optional_name_hint);
    }

    void forward(AutoGrad<DataType> & ag,
        std::span<NodeHandle> input,
        std::span<NodeHandle> output)
    {
        assert(hidden_dim > 0);
        
        // Assert dimension compatibility: input must match fc1 columns, output must match fc2 rows
        assert((int)input.size() == fc1.parameters.cols);
        assert((int)output.size() == fc2.parameters.rows);

        std::vector<NodeHandle> hidden(hidden_dim);
        fc1.forward(ag, input, hidden);

        for (NodeHandle & n : hidden)
        {
            n = Activation::apply(ag, n);
        }

        fc2.forward(ag, hidden, output);
    }
};
