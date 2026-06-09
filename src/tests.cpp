#include "autograd.h"
#include "linearlayer.h"
#include "rmsnormlayer.h"
#include "softmaxlayer.h"
#include "embeddinglayer.h"
#include "attentionlayer.h"
#include "mlplayer.h"
#include "transformerblock.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>

typedef float DataType;

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------
static int g_passed = 0;
static int g_failed  = 0;

#define ASSERT_FLOAT_EQ(expected, actual, msg)                                 \
    do {                                                                       \
        float _e = (expected), _a = (actual);                                  \
        if (std::abs(_e - _a) > 1e-4f) {                                      \
            ++g_failed;                                                        \
            printf("  FAIL: %s  (expected %f, got %f)\n", msg, _e, _a);       \
        } else {                                                               \
            ++g_passed;                                                        \
        }                                                                      \
    } while (0)

#define ASSERT_INT_EQ(expected, actual, msg)                                   \
    do {                                                                       \
        int _e = (expected), _a = (actual);                                    \
        if (_e != _a) {                                                        \
            ++g_failed;                                                        \
            printf("  FAIL: %s  (expected %d, got %d)\n", msg, _e, _a);       \
        } else {                                                               \
            ++g_passed;                                                        \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// value_leaf / value_const --------------------------------------------------
static void test_leaf_node(AutoGrad<DataType> &ag)
{
    printf("test_leaf_node ... ");
    NodeHandle h = ag.value_leaf(42.0f);
    ASSERT_FLOAT_EQ(42.0f, ag.get(h).tensor.values()[0], "leaf data");
    ASSERT_FLOAT_EQ(0.0f,  ag.get(h).tensor.gradients()[0], "leaf grad initialized to 0");
    ASSERT_INT_EQ(0, ag.get(h).children.size(), "leaf has no children");
    printf("\n");
}

// value_add -----------------------------------------------------------------
static void test_add(AutoGrad<DataType> &ag)
{
    printf("test_add ... ");
    NodeHandle a = ag.value_leaf(3.0f);
    NodeHandle b = ag.value_leaf(5.0f);
    NodeHandle s = ag.value_add(a, b);

    ASSERT_FLOAT_EQ(8.0f, ag.get(s).tensor.values()[0], "3 + 5 == 8");
    ASSERT_INT_EQ(2, ag.get(s).children.size(), "add has 2 children");

    ag.backward(s);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[0], "d(a+b)/da == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(b).tensor.gradients()[0], "d(a+b)/db == 1");
    printf("\n");
}

// value_sub -----------------------------------------------------------------
static void test_sub(AutoGrad<DataType> &ag)
{
    printf("test_sub ... ");
    NodeHandle a = ag.value_leaf(10.0f);
    NodeHandle b = ag.value_leaf(4.0f);
    NodeHandle d = ag.value_sub(a, b);

    ASSERT_FLOAT_EQ(6.0f, ag.get(d).tensor.values()[0], "10 - 4 == 6");

    ag.backward(d);
    ASSERT_FLOAT_EQ( 1.0f, ag.get(a).tensor.gradients()[0], "d(a-b)/da ==  1");
    ASSERT_FLOAT_EQ(-1.0f, ag.get(b).tensor.gradients()[0], "d(a-b)/db == -1");
    printf("\n");
}

// value_mul -----------------------------------------------------------------
static void test_mul(AutoGrad<DataType> &ag)
{
    printf("test_mul ... ");
    NodeHandle a = ag.value_leaf(3.0f);
    NodeHandle b = ag.value_leaf(7.0f);
    NodeHandle p = ag.value_mul(a, b);

    ASSERT_FLOAT_EQ(21.0f, ag.get(p).tensor.values()[0], "3 * 7 == 21");

    ag.backward(p);
    ASSERT_FLOAT_EQ(7.0f, ag.get(a).tensor.gradients()[0], "d(a*b)/da == b == 7");
    ASSERT_FLOAT_EQ(3.0f, ag.get(b).tensor.gradients()[0], "d(a*b)/db == a == 3");
    printf("\n");
}

// value_div -----------------------------------------------------------------
static void test_div(AutoGrad<DataType> &ag)
{
    printf("test_div ... ");
    NodeHandle a = ag.value_leaf(10.0f);
    NodeHandle b = ag.value_leaf(4.0f);
    NodeHandle q = ag.value_div(a, b);

    ASSERT_FLOAT_EQ(2.5f, ag.get(q).tensor.values()[0], "10 / 4 == 2.5");

    ag.backward(q);
    ASSERT_FLOAT_EQ(0.25f,    ag.get(a).tensor.gradients()[0], "d(a/b)/da == 1/b == 0.25");
    ASSERT_FLOAT_EQ(-0.625f, ag.get(b).tensor.gradients()[0], "d(a/b)/db == -a/b^2 == -0.625");
    printf("\n");
}

// value_pow -----------------------------------------------------------------
static void test_pow(AutoGrad<DataType> &ag)
{
    printf("test_pow ... ");
    NodeHandle a = ag.value_leaf(2.0f);
    NodeHandle r = ag.value_pow(a, 3.0f);

    ASSERT_FLOAT_EQ(8.0f, ag.get(r).tensor.values()[0], "2^3 == 8");

    ag.backward(r);
    ASSERT_FLOAT_EQ(12.0f, ag.get(a).tensor.gradients()[0], "d(a^3)/da == 3*a^2 == 12");
    printf("\n");
}

// value_log -----------------------------------------------------------------
static void test_log(AutoGrad<DataType> &ag)
{
    printf("test_log ... ");
    NodeHandle a = ag.value_leaf(std::exp(1.0f));  // e
    NodeHandle r = ag.value_log(a);

    ASSERT_FLOAT_EQ(1.0f, ag.get(r).tensor.values()[0], "log(e) == 1");

    ag.backward(r);
    ASSERT_FLOAT_EQ(1.0f / std::exp(1.0f), ag.get(a).tensor.gradients()[0], "d(log(a))/da == 1/e");
    printf("\n");
}

// value_exp -----------------------------------------------------------------
static void test_exp(AutoGrad<DataType> &ag)
{
    printf("test_exp ... ");
    NodeHandle a = ag.value_leaf(2.0f);
    NodeHandle r = ag.value_exp(a);

    float exp2 = std::exp(2.0f);
    ASSERT_FLOAT_EQ(exp2, ag.get(r).tensor.values()[0], "exp(2)");

    ag.backward(r);
    ASSERT_FLOAT_EQ(exp2, ag.get(a).tensor.gradients()[0], "d(exp(a))/da == exp(2)");
    printf("\n");
}

// value_add_const -----------------------------------------------------------
static void test_add_const(AutoGrad<DataType> &ag)
{
    printf("test_add_const ... ");
    NodeHandle a = ag.value_leaf(3.0f);
    NodeHandle r = ag.value_add_const(a, 5.0f);

    ASSERT_FLOAT_EQ(8.0f, ag.get(r).tensor.values()[0], "3 + 5 == 8");

    ag.backward(r);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[0], "d(a+c)/da == 1");
    printf("\n");
}

// value_mul_const -----------------------------------------------------------
static void test_mul_const(AutoGrad<DataType> &ag)
{
    printf("test_mul_const ... ");
    NodeHandle a = ag.value_leaf(4.0f);
    NodeHandle r = ag.value_mul_const(a, 3.0f);

    ASSERT_FLOAT_EQ(12.0f, ag.get(r).tensor.values()[0], "4 * 3 == 12");

    ag.backward(r);
    ASSERT_FLOAT_EQ(3.0f, ag.get(a).tensor.gradients()[0], "d(a*c)/da == c == 3");
    printf("\n");
}

// value_sub_const -----------------------------------------------------------
static void test_sub_const(AutoGrad<DataType> &ag)
{
    printf("test_sub_const ... ");
    NodeHandle a = ag.value_leaf(10.0f);
    NodeHandle r = ag.value_sub_const(a, 4.0f);

    ASSERT_FLOAT_EQ(6.0f, ag.get(r).tensor.values()[0], "10 - 4 == 6");

    ag.backward(r);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[0], "d(a-c)/da == 1");
    printf("\n");
}

// value_div_const -----------------------------------------------------------
static void test_div_const(AutoGrad<DataType> &ag)
{
    printf("test_div_const ... ");
    NodeHandle a = ag.value_leaf(10.0f);
    NodeHandle r = ag.value_div_const(a, 4.0f);

    ASSERT_FLOAT_EQ(2.5f, ag.get(r).tensor.values()[0], "10 / 4 == 2.5");

    ag.backward(r);
    ASSERT_FLOAT_EQ(0.25f, ag.get(a).tensor.gradients()[0], "d(a/c)/da == 1/c == 0.25");
    printf("\n");
}

// value_tile_scalar ---------------------------------------------------------
static void test_tile_scalar(AutoGrad<DataType> &ag)
{
    printf("test_tile_scalar ... ");
    NodeHandle s = ag.value_leaf(7.0f);
    NodeHandle t = ag.value_tile_scalar(s, 4);

    ASSERT_INT_EQ(4, ag.get(t).tensor.numel(), "tiled to 4 elements");
    for (int i = 0; i < 4; i++)
        ASSERT_FLOAT_EQ(7.0f, ag.get(t).tensor.values()[i], "all elements == 7");

    ag.backward(t);
    ASSERT_FLOAT_EQ(4.0f, ag.get(s).tensor.gradients()[0], "grad sums to 4 (one per element)");
    printf("\n");
}

// value_mul_scalar ----------------------------------------------------------
static void test_mul_scalar(AutoGrad<DataType> &ag)
{
    printf("test_mul_scalar ... ");
    NodeHandle input = ag.value_leaf(6.0f);
    NodeHandle scalar = ag.value_leaf(3.0f);
    NodeHandle r = ag.value_mul_scalar(input, scalar);

    ASSERT_FLOAT_EQ(18.0f, ag.get(r).tensor.values()[0], "6 * 3 == 18");

    ag.backward(r);
    ASSERT_FLOAT_EQ(3.0f, ag.get(input).tensor.gradients()[0], "d(input*scalar)/d_input == scalar");
    ASSERT_FLOAT_EQ(6.0f, ag.get(scalar).tensor.gradients()[0], "d(input*scalar)/d_scalar == input");
    printf("\n");
}

// value_div_scalar ----------------------------------------------------------
static void test_div_scalar(AutoGrad<DataType> &ag)
{
    printf("test_div_scalar ... ");
    NodeHandle input = ag.value_leaf(12.0f);
    NodeHandle scalar = ag.value_leaf(4.0f);
    NodeHandle r = ag.value_div_scalar(input, scalar);

    ASSERT_FLOAT_EQ(3.0f, ag.get(r).tensor.values()[0], "12 / 4 == 3");

    ag.backward(r);
    ASSERT_FLOAT_EQ(0.25f, ag.get(input).tensor.gradients()[0], "d(input/scalar)/d_input == 1/scalar");
    ASSERT_FLOAT_EQ(-0.75f, ag.get(scalar).tensor.gradients()[0], "d(input/scalar)/d_scalar == -input/scalar^2");
    printf("\n");
}

// value_relu ----------------------------------------------------------------
static void test_relu_positive(AutoGrad<DataType> &ag)
{
    printf("test_relu_positive ... ");
    NodeHandle a = ag.value_leaf(3.0f);
    NodeHandle r = ag.value_relu(a);

    ASSERT_FLOAT_EQ(3.0f, ag.get(r).tensor.values()[0], "relu(3) == 3");

    ag.backward(r);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[0], "d(relu)/da == 1 when a > 0");
    printf("\n");
}

// value_erf -----------------------------------------------------------------
static void test_erf(AutoGrad<DataType> &ag)
{
    printf("test_erf ... ");
    NodeHandle a = ag.value_leaf(0.0f);
    NodeHandle r = ag.value_erf(a);

    ASSERT_FLOAT_EQ(0.0f, ag.get(r).tensor.values()[0], "erf(0) == 0");

    ag.backward(r);
    // d(erf(x))/dx = 2/sqrt(pi) * exp(-x^2), at x=0 that's 2/sqrt(pi) ~= 1.128379
    ASSERT_FLOAT_EQ(1.128379f, ag.get(a).tensor.gradients()[0], "d(erf)/dx at 0 == 2/sqrt(pi)");
    printf("\n");
}

// value_gelu ----------------------------------------------------------------
static void test_gelu(AutoGrad<DataType> &ag)
{
    printf("test_gelu ... ");
    NodeHandle a = ag.value_leaf(0.0f);
    NodeHandle r = ag.value_gelu(a);

    ASSERT_FLOAT_EQ(0.0f, ag.get(r).tensor.values()[0], "gelu(0) == 0");

    ag.backward(r);
    // d(gelu(x))/dx at x=0 is 0.5
    ASSERT_FLOAT_EQ(0.5f, ag.get(a).tensor.gradients()[0], "d(gelu)/dx at 0 == 0.5");
    printf("\n");
}

static void test_relu_negative(AutoGrad<DataType> &ag)
{
    printf("test_relu_negative ... ");
    NodeHandle a = ag.value_leaf(-2.0f);
    NodeHandle r = ag.value_relu(a);

    ASSERT_FLOAT_EQ(0.0f, ag.get(r).tensor.values()[0], "relu(-2) == 0");

    ag.backward(r);
    ASSERT_FLOAT_EQ(0.0f, ag.get(a).tensor.gradients()[0], "d(relu)/da == 0 when a <= 0");
    printf("\n");
}

// value_select_row ----------------------------------------------------------
static void test_select_row(AutoGrad<DataType> &ag)
{
    printf("test_select_row ... ");
    // Create a 2x3 matrix: {{1,2,3},{4,5,6}}
    NodeHandle m = ag.tensor_leaf({2, 3});
    DataType *mv = ag.get(m).tensor.values().data();
    mv[0]=1; mv[1]=2; mv[2]=3; mv[3]=4; mv[4]=5; mv[5]=6;

    NodeHandle r = ag.value_select_row(m, 1);  // select row 1 -> {4,5,6}
    ASSERT_FLOAT_EQ(4.0f, ag.get(r).tensor.values()[0], "row1[0] == 4");
    ASSERT_FLOAT_EQ(5.0f, ag.get(r).tensor.values()[1], "row1[1] == 5");
    ASSERT_FLOAT_EQ(6.0f, ag.get(r).tensor.values()[2], "row1[2] == 6");

    ag.backward(r);
    ASSERT_FLOAT_EQ(0.0f, ag.get(m).tensor.gradients()[0], "row 0 grad == 0");
    ASSERT_FLOAT_EQ(1.0f, ag.get(m).tensor.gradients()[3], "row 1 grad == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(m).tensor.gradients()[4], "row 1 grad == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(m).tensor.gradients()[5], "row 1 grad == 1");
    printf("\n");
}

// value_slice_cols ----------------------------------------------------------
static void test_slice_cols(AutoGrad<DataType> &ag)
{
    printf("test_slice_cols ... ");
    // Create a 2x4 matrix: {{1,2,3,4},{5,6,7,8}}
    NodeHandle m = ag.tensor_leaf({2, 4});
    DataType *mv = ag.get(m).tensor.values().data();
    mv[0]=1; mv[1]=2; mv[2]=3; mv[3]=4;
    mv[4]=5; mv[5]=6; mv[6]=7; mv[7]=8;

    NodeHandle s = ag.value_slice_cols(m, 1, 2);  // cols [1..2] -> {{2,3},{6,7}}
    ASSERT_FLOAT_EQ(2.0f, ag.get(s).tensor.values()[0], "s[0,0] == 2");
    ASSERT_FLOAT_EQ(3.0f, ag.get(s).tensor.values()[1], "s[0,1] == 3");
    ASSERT_FLOAT_EQ(6.0f, ag.get(s).tensor.values()[2], "s[1,0] == 6");
    ASSERT_FLOAT_EQ(7.0f, ag.get(s).tensor.values()[3], "s[1,1] == 7");

    ag.backward(s);
    ASSERT_FLOAT_EQ(0.0f, ag.get(m).tensor.gradients()[0], "col 0 not sliced");
    ASSERT_FLOAT_EQ(1.0f, ag.get(m).tensor.gradients()[1], "col 1 sliced");
    ASSERT_FLOAT_EQ(1.0f, ag.get(m).tensor.gradients()[2], "col 2 sliced");
    ASSERT_FLOAT_EQ(0.0f, ag.get(m).tensor.gradients()[3], "col 3 not sliced");
    printf("\n");
}

// value_select_element ------------------------------------------------------
static void test_select_element(AutoGrad<DataType> &ag)
{
    printf("test_select_element ... ");
    NodeHandle a = ag.tensor_leaf({3});
    DataType *av = ag.get(a).tensor.values().data();
    av[0] = 10; av[1] = 20; av[2] = 30;

    NodeHandle e = ag.value_select_element(a, 2);  // select element at index 2
    ASSERT_FLOAT_EQ(30.0f, ag.get(e).tensor.values()[0], "element[2] == 30");

    ag.backward(e);
    ASSERT_FLOAT_EQ(0.0f, ag.get(a).tensor.gradients()[0], "grad at 0 == 0");
    ASSERT_FLOAT_EQ(0.0f, ag.get(a).tensor.gradients()[1], "grad at 1 == 0");
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[2], "grad at 2 == 1");
    printf("\n");
}

// value_scatter_row ---------------------------------------------------------
static void test_scatter_row(AutoGrad<DataType> &ag)
{
    printf("test_scatter_row ... ");
    // dst: 3x2 matrix {{1,1},{1,1},{1,1}}
    NodeHandle dst = ag.tensor_leaf({3, 2}, 1.0f);
    // src: {10,20}
    NodeHandle src = ag.tensor_leaf({2});
    DataType *sv = ag.get(src).tensor.values().data();
    sv[0] = 10; sv[1] = 20;

    NodeHandle out = ag.value_scatter_row(dst, src, 1);  // scatter src into row 1
    ASSERT_FLOAT_EQ(1.0f, ag.get(out).tensor.values()[0], "row 0 unchanged");
    ASSERT_FLOAT_EQ(1.0f, ag.get(out).tensor.values()[1], "row 0 unchanged");
    ASSERT_FLOAT_EQ(10.0f, ag.get(out).tensor.values()[2], "row 1 overwritten");
    ASSERT_FLOAT_EQ(20.0f, ag.get(out).tensor.values()[3], "row 1 overwritten");
    ASSERT_FLOAT_EQ(1.0f, ag.get(out).tensor.values()[4], "row 2 unchanged");

    ag.backward(out);
    ASSERT_FLOAT_EQ(1.0f, ag.get(src).tensor.gradients()[0], "src grad[0] == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(src).tensor.gradients()[1], "src grad[1] == 1");
    printf("\n");
}

// value_scatter_cols --------------------------------------------------------
static void test_scatter_cols(AutoGrad<DataType> &ag)
{
    printf("test_scatter_cols ... ");
    // dst: 2x4 matrix {{1,1,1,1},{1,1,1,1}}
    NodeHandle dst = ag.tensor_leaf({2, 4}, 1.0f);
    // src: 2x2 matrix {{10,20},{30,40}}
    NodeHandle src = ag.tensor_leaf({2, 2});
    DataType *sv = ag.get(src).tensor.values().data();
    sv[0] = 10; sv[1] = 20; sv[2] = 30; sv[3] = 40;

    NodeHandle out = ag.value_scatter_cols(dst, src, 1);  // scatter src into cols 1-2
    // Row 0: {1, 10, 20, 1}
    ASSERT_FLOAT_EQ(1.0f, ag.get(out).tensor.values()[0], "row 0 col 0 unchanged");
    ASSERT_FLOAT_EQ(10.0f, ag.get(out).tensor.values()[1], "row 0 col 1 overwritten");
    ASSERT_FLOAT_EQ(20.0f, ag.get(out).tensor.values()[2], "row 0 col 2 overwritten");
    ASSERT_FLOAT_EQ(1.0f, ag.get(out).tensor.values()[3], "row 0 col 3 unchanged");
    // Row 1: {1, 30, 40, 1}
    ASSERT_FLOAT_EQ(1.0f, ag.get(out).tensor.values()[4], "row 1 col 0 unchanged");
    ASSERT_FLOAT_EQ(30.0f, ag.get(out).tensor.values()[5], "row 1 col 1 overwritten");
    ASSERT_FLOAT_EQ(40.0f, ag.get(out).tensor.values()[6], "row 1 col 2 overwritten");
    ASSERT_FLOAT_EQ(1.0f, ag.get(out).tensor.values()[7], "row 1 col 3 unchanged");

    ag.backward(out);
    ASSERT_FLOAT_EQ(1.0f, ag.get(src).tensor.gradients()[0], "src grad[0] == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(src).tensor.gradients()[1], "src grad[1] == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(src).tensor.gradients()[2], "src grad[2] == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(src).tensor.gradients()[3], "src grad[3] == 1");
    printf("\n");
}

// value_matmul --------------------------------------------------------------
static void test_matmul(AutoGrad<DataType> &ag)
{
    printf("test_matmul ... ");
    // A: 2x2 {{1,2},{3,4}}, B: 2x2 {{5,6},{7,8}}
    NodeHandle a = ag.tensor_leaf({2, 2});
    DataType *av = ag.get(a).tensor.values().data();
    av[0]=1; av[1]=2; av[2]=3; av[3]=4;

    NodeHandle b = ag.tensor_leaf({2, 2});
    DataType *bv = ag.get(b).tensor.values().data();
    bv[0]=5; bv[1]=6; bv[2]=7; bv[3]=8;

    NodeHandle c = ag.value_matmul(a, b);
    ASSERT_FLOAT_EQ(19.0f, ag.get(c).tensor.values()[0], "C[0,0] == 1*5+2*7");
    ASSERT_FLOAT_EQ(22.0f, ag.get(c).tensor.values()[1], "C[0,1] == 1*6+2*8");
    ASSERT_FLOAT_EQ(43.0f, ag.get(c).tensor.values()[2], "C[1,0] == 3*5+4*7");
    ASSERT_FLOAT_EQ(50.0f, ag.get(c).tensor.values()[3], "C[1,1] == 3*6+4*8");

    ag.backward(c);
    ASSERT_FLOAT_EQ(11.0f, ag.get(a).tensor.gradients()[0], "dL/dA[0,0]");
    ASSERT_FLOAT_EQ(15.0f, ag.get(a).tensor.gradients()[1], "dL/dA[0,1]");
    ASSERT_FLOAT_EQ(4.0f, ag.get(b).tensor.gradients()[0], "dL/dB[0,0]");
    ASSERT_FLOAT_EQ(6.0f, ag.get(b).tensor.gradients()[2], "dL/dB[1,0]");
    printf("\n");
}

// value_matmul_bt -----------------------------------------------------------
static void test_matmul_bt(AutoGrad<DataType> &ag)
{
    printf("test_matmul_bt ... ");
    // A: 2x2 {{1,2},{3,4}}, B: 2x2 {{5,6},{7,8}}  (N=2, K=2)
    // C[m,n] = sum_k A[m,k]*B[n,k]
    NodeHandle a = ag.tensor_leaf({2, 2});
    DataType *av = ag.get(a).tensor.values().data();
    av[0]=1; av[1]=2; av[2]=3; av[3]=4;

    NodeHandle b = ag.tensor_leaf({2, 2});
    DataType *bv = ag.get(b).tensor.values().data();
    bv[0]=5; bv[1]=6; bv[2]=7; bv[3]=8;

    NodeHandle c = ag.value_matmul_bt(a, b);
    ASSERT_FLOAT_EQ(17.0f, ag.get(c).tensor.values()[0], "C[0,0] == 1*5+2*7");
    ASSERT_FLOAT_EQ(23.0f, ag.get(c).tensor.values()[1], "C[0,1] == 1*6+2*8");
    ASSERT_FLOAT_EQ(39.0f, ag.get(c).tensor.values()[2], "C[1,0] == 3*5+4*7");
    ASSERT_FLOAT_EQ(53.0f, ag.get(c).tensor.values()[3], "C[1,1] == 3*6+4*8");

    ag.backward(c);
    ASSERT_FLOAT_EQ(12, ag.get(a).tensor.gradients()[0], "dL/dA[0,0]");
    ASSERT_FLOAT_EQ(14.0f, ag.get(a).tensor.gradients()[1], "dL/dA[0,1]");
    ASSERT_FLOAT_EQ(4.0f, ag.get(b).tensor.gradients()[0], "dL/dB[0,0]");
    ASSERT_FLOAT_EQ(6.0f, ag.get(b).tensor.gradients()[1], "dL/dB[0,1]");
    printf("\n");
}

// value_softmax_rows --------------------------------------------------------
static void test_softmax_rows(AutoGrad<DataType> &ag)
{
    printf("test_softmax_rows ... ");
    // 2x3 matrix: {{1,2,3},{0,0,0}}
    NodeHandle m = ag.tensor_leaf({2, 3});
    DataType *mv = ag.get(m).tensor.values().data();
    mv[0]=1; mv[1]=2; mv[2]=3;
    mv[3]=0; mv[4]=0; mv[5]=0;

    NodeHandle s = ag.value_softmax_rows(m);

    // Row 1: softmax(0,0,0) = {1/3, 1/3, 1/3}
    ASSERT_FLOAT_EQ(0.333333f, ag.get(s).tensor.values()[3], "softmax row 1, col 0");
    ASSERT_FLOAT_EQ(0.333333f, ag.get(s).tensor.values()[4], "softmax row 1, col 1");

    // Row 0 should sum to ~1
    float row0sum = ag.get(s).tensor.values()[0] + ag.get(s).tensor.values()[1] + ag.get(s).tensor.values()[2];
    ASSERT_FLOAT_EQ(1.0f, row0sum, "softmax row 0 sums to 1");

    ag.backward(s);
    printf("\n");
}

// value_max -----------------------------------------------------------------
static void test_max(AutoGrad<DataType> &ag)
{
    printf("test_max ... ");
    NodeHandle a = ag.tensor_leaf({4});
    DataType *av = ag.get(a).tensor.values().data();
    av[0] = 3; av[1] = 7; av[2] = 1; av[3] = 5;

    NodeHandle mx = ag.value_max(a);
    ASSERT_FLOAT_EQ(7.0f, ag.get(mx).tensor.values()[0], "max == 7");

    ag.backward(mx);
    ASSERT_FLOAT_EQ(0.0f, ag.get(a).tensor.gradients()[0], "grad at non-max == 0");
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[1], "grad at max == 1");
    ASSERT_FLOAT_EQ(0.0f, ag.get(a).tensor.gradients()[2], "grad at non-max == 0");
    ASSERT_FLOAT_EQ(0.0f, ag.get(a).tensor.gradients()[3], "grad at non-max == 0");
    printf("\n");
}

// value_sum -----------------------------------------------------------------
static void test_sum(AutoGrad<DataType> &ag)
{
    printf("test_sum ... ");
    NodeHandle a = ag.tensor_leaf({3});
    DataType *av = ag.get(a).tensor.values().data();
    av[0] = 1; av[1] = 2; av[2] = 3;

    NodeHandle s = ag.value_sum(a);
    ASSERT_FLOAT_EQ(6.0f, ag.get(s).tensor.values()[0], "sum == 6");

    ag.backward(s);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[0], "grad[0] == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[1], "grad[1] == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[2], "grad[2] == 1");
    printf("\n");
}

// value_sum_rows ------------------------------------------------------------
static void test_sum_rows(AutoGrad<DataType> &ag)
{
    printf("test_sum_rows ... ");
    // 2x3 matrix: {{1,2,3},{4,5,6}}
    NodeHandle m = ag.tensor_leaf({2, 3});
    DataType *mv = ag.get(m).tensor.values().data();
    mv[0]=1; mv[1]=2; mv[2]=3;
    mv[3]=4; mv[4]=5; mv[5]=6;

    NodeHandle s = ag.value_sum_rows(m);
    ASSERT_FLOAT_EQ(6.0f, ag.get(s).tensor.values()[0], "row 0 sum == 6");
    ASSERT_FLOAT_EQ(15.0f, ag.get(s).tensor.values()[1], "row 1 sum == 15");

    ag.backward(s);
    ASSERT_FLOAT_EQ(1.0f, ag.get(m).tensor.gradients()[0], "grad[0,0] == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(m).tensor.gradients()[5], "grad[1,2] == 1");
    printf("\n");
}

// value_scale_rows ----------------------------------------------------------
static void test_scale_rows(AutoGrad<DataType> &ag)
{
    printf("test_scale_rows ... ");
    // A: 2x2 {{1,2},{3,4}}, v: {2, 3}
    NodeHandle a = ag.tensor_leaf({2, 2});
    DataType *av = ag.get(a).tensor.values().data();
    av[0]=1; av[1]=2; av[2]=3; av[3]=4;

    NodeHandle v = ag.tensor_leaf({2});
    DataType *vv = ag.get(v).tensor.values().data();
    vv[0]=2; vv[1]=3;

    NodeHandle out = ag.value_scale_rows(a, v);
    ASSERT_FLOAT_EQ(2.0f, ag.get(out).tensor.values()[0], "out[0,0] == 1*2");
    ASSERT_FLOAT_EQ(4.0f, ag.get(out).tensor.values()[1], "out[0,1] == 2*2");
    ASSERT_FLOAT_EQ(9.0f, ag.get(out).tensor.values()[2], "out[1,0] == 3*3");
    ASSERT_FLOAT_EQ(12.0f, ag.get(out).tensor.values()[3], "out[1,1] == 4*3");

    ag.backward(out);
    ASSERT_FLOAT_EQ(2.0f, ag.get(a).tensor.gradients()[0], "dA[0,0] == v[0]");
    ASSERT_FLOAT_EQ(3.0f, ag.get(a).tensor.gradients()[2], "dA[1,0] == v[1]");
    ASSERT_FLOAT_EQ(3.0f, ag.get(v).tensor.gradients()[0], "dv[0] == 1*1+1*2");
    ASSERT_FLOAT_EQ(7.0f, ag.get(v).tensor.gradients()[1], "dv[1] == 1*3+1*4");
    printf("\n");
}

// value_mul_rows ------------------------------------------------------------
static void test_mul_rows(AutoGrad<DataType> &ag)
{
    printf("test_mul_rows ... ");
    // A: 2x2 {{1,2},{3,4}}, b: {5,6}
    NodeHandle a = ag.tensor_leaf({2, 2});
    DataType *av = ag.get(a).tensor.values().data();
    av[0]=1; av[1]=2; av[2]=3; av[3]=4;

    NodeHandle b = ag.tensor_leaf({2});
    DataType *bv = ag.get(b).tensor.values().data();
    bv[0]=5; bv[1]=6;

    NodeHandle out = ag.value_mul_rows(a, b);
    ASSERT_FLOAT_EQ(5.0f, ag.get(out).tensor.values()[0], "out[0,0] == 1*5");
    ASSERT_FLOAT_EQ(12.0f, ag.get(out).tensor.values()[1], "out[0,1] == 2*6");
    ASSERT_FLOAT_EQ(15.0f, ag.get(out).tensor.values()[2], "out[1,0] == 3*5");
    ASSERT_FLOAT_EQ(24.0f, ag.get(out).tensor.values()[3], "out[1,1] == 4*6");

    ag.backward(out);
    ASSERT_FLOAT_EQ(5.0f, ag.get(a).tensor.gradients()[0], "dA[0,0] == b[0]");
    ASSERT_FLOAT_EQ(6.0f, ag.get(a).tensor.gradients()[1], "dA[0,1] == b[1]");
    ASSERT_FLOAT_EQ(4.0f, ag.get(b).tensor.gradients()[0], "db[0] == 1+3");
    ASSERT_FLOAT_EQ(6.0f, ag.get(b).tensor.gradients()[1], "db[1] == 2+4");
    printf("\n");
}

// value_add_rows ------------------------------------------------------------
static void test_add_rows(AutoGrad<DataType> &ag)
{
    printf("test_add_rows ... ");
    // A: 2x2 {{1,2},{3,4}}, b: {10,20}
    NodeHandle a = ag.tensor_leaf({2, 2});
    DataType *av = ag.get(a).tensor.values().data();
    av[0]=1; av[1]=2; av[2]=3; av[3]=4;

    NodeHandle b = ag.tensor_leaf({2});
    DataType *bv = ag.get(b).tensor.values().data();
    bv[0]=10; bv[1]=20;

    NodeHandle out = ag.value_add_rows(a, b);
    ASSERT_FLOAT_EQ(11.0f, ag.get(out).tensor.values()[0], "out[0,0] == 1+10");
    ASSERT_FLOAT_EQ(22.0f, ag.get(out).tensor.values()[1], "out[0,1] == 2+20");
    ASSERT_FLOAT_EQ(13.0f, ag.get(out).tensor.values()[2], "out[1,0] == 3+10");
    ASSERT_FLOAT_EQ(24.0f, ag.get(out).tensor.values()[3], "out[1,1] == 4+20");

    ag.backward(out);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).tensor.gradients()[0], "dA[0,0] == 1");
    ASSERT_FLOAT_EQ(2.0f, ag.get(b).tensor.gradients()[0], "db[0] == 1+1");
    ASSERT_FLOAT_EQ(2.0f, ag.get(b).tensor.gradients()[1], "db[1] == 1+1");
    printf("\n");
}

// tensor_leaf ---------------------------------------------------------------
static void test_tensor_leaf(AutoGrad<DataType> &ag)
{
    printf("test_tensor_leaf ... ");
    NodeHandle t = ag.tensor_leaf({2, 3}, 5.0f);

    ASSERT_INT_EQ(6, ag.get(t).tensor.numel(), "tensor has 6 elements");
    for (int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(5.0f, ag.get(t).tensor.values()[i], "all elements == 5");
    ASSERT_INT_EQ(0, ag.get(t).children.size(), "leaf has no children");
    printf("\n");
}

// Chained computation:  f(x) = (x * 2 + 3) ^ 2   ,  x = 4
//   f(4) = (8 + 3)^2 = 121
//   df/dx = 2 * (x*2+3) * 2 = 44
static void test_chained_computation(AutoGrad<DataType> &ag)
{
    printf("test_chained_computation ... ");
    NodeHandle x = ag.value_leaf(4.0f);
    NodeHandle t1 = ag.value_mul(x, ag.value_leaf(2.0f));   // x * 2
    NodeHandle t2 = ag.value_add(t1, ag.value_leaf(3.0f));  // x * 2 + 3
    NodeHandle t3 = ag.value_pow(t2, 2.0f);                 // (x*2+3)^2

    ASSERT_FLOAT_EQ(121.0f, ag.get(t3).tensor.values()[0], "f(4) == 121");

    ag.backward(t3);
    ASSERT_FLOAT_EQ(44.0f, ag.get(x).tensor.gradients()[0], "df/dx == 44");
    printf("\n");
}

// Shared node:  loss = a + a  (a is used twice)
//   d(loss)/da should be 2
static void test_shared_node(AutoGrad<DataType> &ag)
{
    printf("test_shared_node ... ");
    NodeHandle a = ag.value_leaf(5.0f);
    NodeHandle s = ag.value_add(a, a);

    ASSERT_FLOAT_EQ(10.0f, ag.get(s).tensor.values()[0], "5 + 5 == 10");

    ag.backward(s);
    ASSERT_FLOAT_EQ(2.0f, ag.get(a).tensor.gradients()[0], "d(a+a)/da == 2 (gradient accumulated)");
    printf("\n");
}

// reset() -------------------------------------------------------------------
static void test_reset(AutoGrad<DataType> &ag)
{
    printf("test_reset ... ");
    ag.value_leaf(1.0f);
    ag.value_leaf(2.0f);
    ag.value_add(ag.value_leaf(3.0f), ag.value_leaf(4.0f));

    ag.reset();
    ASSERT_INT_EQ(0, (int)ag.allocate_node(TensorShape({1})), "first node after reset is index 0");
    printf("\n");
}

// allocate_matrix -----------------------------------------------------------
static void test_allocate_matrix(AutoGrad<DataType> &ag)
{
    printf("test_allocate_matrix ... ");
    NodeMatrixHandle matrix = ag.allocate_parameter_matrix(2, 3, 0, 0.1f);

    // Quick sanity: nodes exist and have data in a reasonable range
    bool ok = true;
    for (int row = 0; row < 2; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            float val = ag.get(matrix.start).tensor(0,0);
            if (std::abs(val) > 3.0f) ok = false;  // 3-sigma bound, very unlikely to fail
        }
    }
    if (!ok) {
        ++g_failed;
        printf("  FAIL: allocate_matrix values out of expected range\n");
    } else {
        ++g_passed;
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Layer Tests
// ---------------------------------------------------------------------------

// TransformerBlock ----------------------------------------------------------
static void test_transformer_block(AutoGrad<DataType> &ag)
{
    printf("test_transformer_block ... ");

    const int emb_dim = 8;
    const int num_heads = 2;
    const int ffn_dim = 16;
    const int seq_len = 3;

    TransformerBlock<DataType> block;
    block.init(ag, emb_dim, num_heads, ffn_dim, 0.1f);

    // Create input {seq_len, emb_dim}
    NodeHandle input = ag.tensor_leaf({seq_len, emb_dim});
    {
        DataType *iv = ag.get(input).tensor.values().data();
        for (int i = 0; i < seq_len * emb_dim; i++)
            iv[i] = DataType(i % 5 + 1);
    }

    NodeHandle output = block.forward(ag, input);
    {
        const auto &shape = ag.get(output).tensor.get_shape();
        ASSERT_INT_EQ(2, shape.rank(), "output rank == 2");
        ASSERT_INT_EQ(seq_len, shape.dims[0], "output seq_len == 3");
        ASSERT_INT_EQ(emb_dim, shape.dims[1], "output emb_dim == 8");
    }

    // Verify backward passes without error
    ag.backward(output);

    // Verify gradients flow back to input
    {
        const auto &gr = ag.get(input).tensor.gradients();
        bool any_grad = false;
        for (int i = 0; i < seq_len * emb_dim; i++)
            if (std::abs(gr[i]) > 1e-8f)
                any_grad = true;
        if (!any_grad)
        {
            ++g_failed;
            printf("  FAIL: no gradients flowed back to input\n");
        }
        else
        {
            ++g_passed;
        }
    }

    printf("\n");
}

// AttentionLayer ------------------------------------------------------------
static void test_attention_layer(AutoGrad<DataType> &ag)
{
    printf("test_attention_layer ... ");

    const int emb_dim = 8;
    const int num_heads = 2;
    const int seq_len = 3;

    AttentionLayer<DataType> layer;
    layer.init(ag, emb_dim, num_heads);

    // Create input {seq_len, emb_dim}
    NodeHandle input = ag.tensor_leaf({seq_len, emb_dim});
    {
        DataType *iv = ag.get(input).tensor.values().data();
        for (int i = 0; i < seq_len * emb_dim; i++)
            iv[i] = DataType(i % 7 + 1);  // fill with small values
    }

    NodeHandle output = layer.forward(ag, input);
    {
        const auto &shape = ag.get(output).tensor.get_shape();
        ASSERT_INT_EQ(2, shape.rank(), "output rank == 2");
        ASSERT_INT_EQ(seq_len, shape.dims[0], "output seq_len == 3");
        ASSERT_INT_EQ(emb_dim, shape.dims[1], "output emb_dim == 8");
    }

    // Verify backward passes without error
    ag.backward(output);

    printf("\n");
}

// EmbeddingLayer ------------------------------------------------------------
static void test_embedding_layer(AutoGrad<DataType> &ag)
{
    printf("test_embedding_layer ... ");

    EmbeddingLayer<DataType> layer;
    layer.init(ag, 4, 3, 0.1f, "test_emb");

    // Set embedding weights to known values:
    // row 0: {10, 20, 30}
    // row 1: {40, 50, 60}
    // row 2: {70, 80, 90}
    // row 3: {100, 110, 120}
    {
        DataType *wv = ag.get(layer.parameters.start).tensor.values().data();
        for (int i = 0; i < 12; i++)
            wv[i] = DataType(10 * (i + 1));
    }

    // Lookup token 1 -> {40, 50, 60}
    NodeHandle output = layer.forward(ag, 1);
    ASSERT_INT_EQ(3, ag.get(output).tensor.numel(), "emb_dim == 3");
    ASSERT_FLOAT_EQ(40.0f, ag.get(output).tensor.values()[0], "emb[1][0] == 40");
    ASSERT_FLOAT_EQ(50.0f, ag.get(output).tensor.values()[1], "emb[1][1] == 50");
    ASSERT_FLOAT_EQ(60.0f, ag.get(output).tensor.values()[2], "emb[1][2] == 60");

    // Backward: grad should only flow to row 1 of the embedding matrix
    ag.backward(output);
    {
        const auto &gr = ag.get(layer.parameters.start).tensor.gradients();
        // row 0 should be all zeros
        ASSERT_FLOAT_EQ(0.0f, gr[0], "emb grad row 0 == 0");
        ASSERT_FLOAT_EQ(0.0f, gr[1], "emb grad row 0 == 0");
        ASSERT_FLOAT_EQ(0.0f, gr[2], "emb grad row 0 == 0");
        // row 1 should be all ones (gradient of output w.r.t. selected row)
        ASSERT_FLOAT_EQ(1.0f, gr[3], "emb grad row 1 == 1");
        ASSERT_FLOAT_EQ(1.0f, gr[4], "emb grad row 1 == 1");
        ASSERT_FLOAT_EQ(1.0f, gr[5], "emb grad row 1 == 1");
        // row 2 should be all zeros
        ASSERT_FLOAT_EQ(0.0f, gr[6], "emb grad row 2 == 0");
    }

    printf("\n");
}

// SoftmaxLayer --------------------------------------------------------------
static void test_softmax_layer(AutoGrad<DataType> &ag)
{
    printf("test_softmax_layer ... ");

    SoftmaxLayer<DataType> layer;

    // Input: {1, 2, 3}
    // exp(1-max) = exp(-2) = 0.135335
    // exp(2-max) = exp(-1) = 0.367879
    // exp(3-max) = exp(0)  = 1.000000
    // sum = 1.503214
    // softmax = {0.090031, 0.244728, 0.665242}
    NodeHandle input = ag.tensor_leaf({3});
    {
        DataType *iv = ag.get(input).tensor.values().data();
        iv[0] = 1; iv[1] = 2; iv[2] = 3;
    }

    NodeHandle output = layer.forward(ag, input);
    ASSERT_FLOAT_EQ(0.090031f, ag.get(output).tensor.values()[0], "softmax[0]");
    ASSERT_FLOAT_EQ(0.244728f, ag.get(output).tensor.values()[1], "softmax[1]");
    ASSERT_FLOAT_EQ(0.665242f, ag.get(output).tensor.values()[2], "softmax[2]");

    // Verify softmax sums to 1
    float sum = 0;
    for (int i = 0; i < 3; i++)
        sum += ag.get(output).tensor.values()[i];
    ASSERT_FLOAT_EQ(1.0f, sum, "softmax sums to 1");

    // Verify backward passes without error
    ag.backward(output);

    printf("\n");
}

// SimpleRMSNormLayer --------------------------------------------------------
static void test_simple_rmsnorm_layer(AutoGrad<DataType> &ag)
{
    printf("test_simple_rmsnorm_layer ... ");

    SimpleRMSNormLayer<DataType> layer;

    // Single vector: {3, 4}
    // mean(x^2) = (9 + 16) / 2 = 12.5
    // scale = 1 / sqrt(12.5 + eps) ≈ 1 / 3.53553 ≈ 0.282843
    // output ≈ {0.848527, 1.131371}
    NodeHandle input = ag.tensor_leaf({2});
    {
        DataType *iv = ag.get(input).tensor.values().data();
        iv[0] = 3; iv[1] = 4;
    }

    NodeHandle output = layer.forward(ag, input);
    ASSERT_FLOAT_EQ(0.848527f, ag.get(output).tensor.values()[0], "norm[0]");
    ASSERT_FLOAT_EQ(1.131371f, ag.get(output).tensor.values()[1], "norm[1]");

    // Verify backward passes without error
    ag.backward(output);
    ASSERT_INT_EQ(2, ag.get(input).tensor.numel(), "input grad shape preserved");

    printf("\n");
}

// RMSNormLayer --------------------------------------------------------------
static void test_rmsnorm_layer(AutoGrad<DataType> &ag)
{
    printf("test_rmsnorm_layer ... ");

    RMSNormLayer<DataType> layer;
    layer.init(ag, 2, 0.1f, "test_rmsnorm");

    // Set gamma = {1, 1}, beta = {0, 0} so it behaves like SimpleRMSNorm
    {
        DataType *gv = ag.get(layer.gamma.start).tensor.values().data();
        gv[0] = 1; gv[1] = 1;
        DataType *bv = ag.get(layer.beta.start).tensor.values().data();
        bv[0] = 0; bv[1] = 0;
    }

    // Input: {3, 4} -> same result as SimpleRMSNorm
    NodeHandle input = ag.tensor_leaf({2});
    {
        DataType *iv = ag.get(input).tensor.values().data();
        iv[0] = 3; iv[1] = 4;
    }

    NodeHandle output = layer.forward(ag, input);
    ASSERT_FLOAT_EQ(0.848527f, ag.get(output).tensor.values()[0], "norm[0] with gamma=1,beta=0");
    ASSERT_FLOAT_EQ(1.131371f, ag.get(output).tensor.values()[1], "norm[1] with gamma=1,beta=0");

    // Now test with gamma = {2, 0.5}, beta = {10, -5}
    // scaled = {0.848527 * 2, 1.131371 * 0.5} = {1.697054, 0.565686}
    // output = {1.697054 + 10, 0.565686 - 5} = {11.697054, -4.434314}
    ag.reset();
    layer.init(ag, 2, 0.1f, "test_rmsnorm2");

    {
        DataType *gv = ag.get(layer.gamma.start).tensor.values().data();
        gv[0] = 2; gv[1] = 0.5f;
        DataType *bv = ag.get(layer.beta.start).tensor.values().data();
        bv[0] = 10; bv[1] = -5;
    }

    input = ag.tensor_leaf({2});
    {
        DataType *iv = ag.get(input).tensor.values().data();
        iv[0] = 3; iv[1] = 4;
    }

    output = layer.forward(ag, input);
    ASSERT_FLOAT_EQ(11.697054f, ag.get(output).tensor.values()[0], "affine norm[0]");
    ASSERT_FLOAT_EQ(-4.434314f, ag.get(output).tensor.values()[1], "affine norm[1]");

    // Verify backward passes without error
    ag.backward(output);

    printf("\n");
}

// LinearLayer ---------------------------------------------------------------
static void test_linear_layer(AutoGrad<DataType> &ag)
{
    printf("test_linear_layer ... ");

    // LinearLayer: out = input @ W
    // input: {seq_len=2, cols=3}, W: {cols=3, rows=2} -> output: {2, 2}
    LinearLayer<DataType> layer;
    layer.init(ag, 3, 2, 0.1f, "test_linear");

    // Set weights to known values: W = {{1, 2}, {3, 4}, {5, 6}}
    // allocate_parameter_matrix stores in row-major: {3 rows, 2 cols}
    {
        auto &w = ag.get(layer.parameters.start).tensor;
        DataType *pv = w.values().data();
        pv[0] = 1; pv[1] = 2;  // row 0
        pv[2] = 3; pv[3] = 4;  // row 1
        pv[4] = 5; pv[5] = 6;  // row 2
    }

    // Set input: {2, 3} -> {{1, 0, 0}, {0, 1, 0}}
    NodeHandle input = ag.tensor_leaf({2, 3});
    {
        DataType *iv = ag.get(input).tensor.values().data();
        iv[0] = 1; iv[1] = 0; iv[2] = 0;  // row 0
        iv[3] = 0; iv[4] = 1; iv[5] = 0;  // row 1
    }

    NodeHandle output = layer.forward(ag, input);
    {
        const auto &shape = ag.get(output).tensor.get_shape();
        ASSERT_INT_EQ(2, shape.rank(), "output rank == 2");
        ASSERT_INT_EQ(2, shape.dims[0], "output rows == 2");
        ASSERT_INT_EQ(2, shape.dims[1], "output cols == 2");

        // row 0 = input_row0 @ W = {1,0,0} @ W_col = {1, 2}
        ASSERT_FLOAT_EQ(1.0f, ag.get(output).tensor.values()[0], "out[0,0] == 1");
        ASSERT_FLOAT_EQ(2.0f, ag.get(output).tensor.values()[1], "out[0,1] == 2");
        // row 1 = input_row1 @ W = {0,1,0} @ W_col = {3, 4}
        ASSERT_FLOAT_EQ(3.0f, ag.get(output).tensor.values()[2], "out[1,0] == 3");
        ASSERT_FLOAT_EQ(4.0f, ag.get(output).tensor.values()[3], "out[1,1] == 4");
    }

    // Backward through the layer to verify gradient flows
    ag.backward(output);
    {
        // Gradient w.r.t. input should be d(output)/d(input) = W^T
        // grad_input[0,0] = W[0,0] = 1, grad_input[0,1] = W[0,1] = 2
        ASSERT_FLOAT_EQ(3.0f, ag.get(input).tensor.gradients()[0], "grad_input[0,0] == W[0,0]");
        ASSERT_FLOAT_EQ(7.0f, ag.get(input).tensor.gradients()[1], "grad_input[0,1] == W[0,1]");
        ASSERT_FLOAT_EQ(11.0f, ag.get(input).tensor.gradients()[2], "grad_input[0,2] == W[0,2]");
        ASSERT_FLOAT_EQ(3.0f, ag.get(input).tensor.gradients()[3], "grad_input[1,0] == W[1,0]");
        ASSERT_FLOAT_EQ(7.0f, ag.get(input).tensor.gradients()[4], "grad_input[1,1] == W[1,1]");
        ASSERT_FLOAT_EQ(11.0f, ag.get(input).tensor.gradients()[5], "grad_input[1,2] == W[1,2]");
    }

    printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    printf("=== AutoGrad Unit Tests ===\n\n");

    AutoGrad<DataType> ag;

    test_leaf_node(ag);
    ag.reset();

    test_tensor_leaf(ag);
    ag.reset();

    test_add(ag);
    ag.reset();

    test_sub(ag);
    ag.reset();

    test_mul(ag);
    ag.reset();

    test_div(ag);
    ag.reset();

    test_pow(ag);
    ag.reset();

    test_log(ag);
    ag.reset();

    test_exp(ag);
    ag.reset();

    test_add_const(ag);
    ag.reset();

    test_mul_const(ag);
    ag.reset();

    test_sub_const(ag);
    ag.reset();

    test_div_const(ag);
    ag.reset();

    test_tile_scalar(ag);
    ag.reset();

    test_mul_scalar(ag);
    ag.reset();

    test_div_scalar(ag);
    ag.reset();

    test_relu_positive(ag);
    ag.reset();

    test_relu_negative(ag);
    ag.reset();

    test_erf(ag);
    ag.reset();

    test_gelu(ag);
    ag.reset();

    test_select_row(ag);
    ag.reset();

    test_slice_cols(ag);
    ag.reset();

    test_select_element(ag);
    ag.reset();

    test_scatter_row(ag);
    ag.reset();

    test_scatter_cols(ag);
    ag.reset();

    test_matmul(ag);
    ag.reset();

    test_matmul_bt(ag);
    ag.reset();

    test_softmax_rows(ag);
    ag.reset();

    test_max(ag);
    ag.reset();

    test_sum(ag);
    ag.reset();

    test_sum_rows(ag);
    ag.reset();

    test_scale_rows(ag);
    ag.reset();

    test_mul_rows(ag);
    ag.reset();

    test_add_rows(ag);
    ag.reset();

    test_chained_computation(ag);
    ag.reset();

    test_shared_node(ag);
    ag.reset();

    test_reset(ag);

    test_allocate_matrix(ag);

    // --- Layer Tests ---
    ag.reset();
    test_softmax_layer(ag);

    ag.reset();
    test_simple_rmsnorm_layer(ag);

    ag.reset();
    test_rmsnorm_layer(ag);

    ag.reset();
    test_embedding_layer(ag);

    ag.reset();
    test_linear_layer(ag);

    ag.reset();
    test_attention_layer(ag);

    ag.reset();
    test_transformer_block(ag);

    printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
