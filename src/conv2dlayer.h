#pragma once

#include "autograd.h"

#include <cassert>
#include <cmath>

// ---------------------------------------------------------------------------
// CONV2D LAYER  (im2col + matmul)
//
// Implements 2D convolution by unfolding input patches into a matrix and
// delegating the arithmetic to the optimized value_matmul path.
//
// -- Tensor layout convention (NHWC-fused)
//
//   input  : {batch * H_in  * W_in,  C_in}
//   output : {batch * H_out * W_out, C_out}
//
//   where  H_out = (H_in - kernel_size) / stride + 1   (valid / no padding)
//          W_out = (W_in - kernel_size) / stride + 1
//
// -- Weight layout
//
//   weights : {C_in * k * k, C_out}
//
//   Column index inside the im2col patch:  c * k*k + ph * k + pw
//   (channel-major, then spatial row, then spatial col)
// ---------------------------------------------------------------------------

template <typename DataType>
struct Conv2dLayer
{
    TensorHandle weights;   // {C_in * k * k, C_out}
    TensorHandle bias;      // {C_out}  (optional)
    bool use_bias = false;

    int H_in = 0;
    int W_in = 0;
    int kernel_size = 0;
    int stride = 1;

    // -------------------------------------------------------------------------
    // init
    //
    //   in_H / in_W     : spatial size of the input feature map
    //   C_in            : input channels
    //   C_out           : output channels (number of filters)
    //   k               : square kernel edge length (e.g. 3 -> 3×3)
    //   conv_stride     : stride (default 1)
    //   should_use_bias : add a per-output-channel learnable bias
    //   std_dev         : weight std dev; pass 0 for He/Kaiming init (sqrt(2/fan_in))
    // -------------------------------------------------------------------------
    void init(AutoGrad<DataType> & grad,
        int in_H, int in_W, int C_in, int C_out, int k,
        int  conv_stride = 1,
        bool should_use_bias = true,
        DataType std_dev = 0,
        const char * name = nullptr)
    {
        H_in = in_H;
        W_in = in_W;
        kernel_size = k;
        stride = conv_stride;

        assert(k > 0 && conv_stride > 0);
        assert(in_H >= k && in_W >= k && "Kernel larger than input spatial dimension");

        // He/Kaiming init: std = sqrt(2 / fan_in), fan_in = C_in * k * k
        const int    fan_in = C_in * k * k;
        const DataType sigma = std_dev > 0
            ? std_dev
            : DataType(2) / std::sqrt(static_cast<DataType>(fan_in));

        weights = grad.allocate_parameter_matrix(
            fan_in, C_out, DataType(0), sigma,
            name ? name : "conv2d");

        use_bias = should_use_bias;
        if (use_bias)
            bias = grad.allocate_parameter_vector(C_out, 0, 0, "conv2d_bias");
    }

    // -------------------------------------------------------------------------
    // forward
    //
    //   input : TensorHandle of shape {batch * H_in * W_in, C_in}
    //   returns              shape {batch * H_out * W_out, C_out}
    //
    // Computation graph:
    //
    //   patches = im2col(input)          {batch*H_out*W_out, C_in*k*k}
    //   out     = patches @ weights      {batch*H_out*W_out, C_out}
    //   return  out + bias (broadcast)   (if use_bias)
    // -------------------------------------------------------------------------
    TensorHandle forward(AutoGrad<DataType> & grad, TensorHandle input)
    {
        // Step 1 — im2col: rearrange every k×k patch into a row.
        //   {batch * H_in * W_in, C_in}  ?  {batch * H_out * W_out, C_in * k * k}
        //
        // This is the only non-trivial bookkeeping; the actual multiply is
        // handled entirely by the optimised matmul below.
        TensorHandle patches = grad.value_im2col(input, H_in, W_in, kernel_size, stride);

        // Step 2 — matmul: applies all C_out filters simultaneously.
        //   {batch * H_out * W_out, C_in*k*k} @ {C_in*k*k, C_out}
        //   ?  {batch * H_out * W_out, C_out}
        //
        // For float DataType this dispatches to the AVX2 path automatically.
        TensorHandle out = grad.value_matmul(patches, weights);

        // Step 3 — bias: same broadcast-add used by LinearLayer.
        if (use_bias)
            return grad.value_add_rows(out, bias);
        else
            return out;
    }
};
