#include "autograd.h"
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
    ASSERT_FLOAT_EQ(42.0f, ag.get(h).data, "leaf data");
    ASSERT_FLOAT_EQ(0.0f,  ag.get(h).grad, "leaf grad initialized to 0");
    ASSERT_INT_EQ(0, ag.get(h).n_children, "leaf has no children");
    printf("\n");
}

// value_add -----------------------------------------------------------------
static void test_add(AutoGrad<DataType> &ag)
{
    printf("test_add ... ");
    NodeHandle a = ag.value_leaf(3.0f);
    NodeHandle b = ag.value_leaf(5.0f);
    NodeHandle s = ag.value_add(a, b);

    ASSERT_FLOAT_EQ(8.0f, ag.get(s).data, "3 + 5 == 8");
    ASSERT_INT_EQ(2, ag.get(s).n_children, "add has 2 children");

    ag.backward(s);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).grad, "d(a+b)/da == 1");
    ASSERT_FLOAT_EQ(1.0f, ag.get(b).grad, "d(a+b)/db == 1");
    printf("\n");
}

// value_sub -----------------------------------------------------------------
static void test_sub(AutoGrad<DataType> &ag)
{
    printf("test_sub ... ");
    NodeHandle a = ag.value_leaf(10.0f);
    NodeHandle b = ag.value_leaf(4.0f);
    NodeHandle d = ag.value_sub(a, b);

    ASSERT_FLOAT_EQ(6.0f, ag.get(d).data, "10 - 4 == 6");

    ag.backward(d);
    ASSERT_FLOAT_EQ( 1.0f, ag.get(a).grad, "d(a-b)/da ==  1");
    ASSERT_FLOAT_EQ(-1.0f, ag.get(b).grad, "d(a-b)/db == -1");
    printf("\n");
}

// value_mul -----------------------------------------------------------------
static void test_mul(AutoGrad<DataType> &ag)
{
    printf("test_mul ... ");
    NodeHandle a = ag.value_leaf(3.0f);
    NodeHandle b = ag.value_leaf(7.0f);
    NodeHandle p = ag.value_mul(a, b);

    ASSERT_FLOAT_EQ(21.0f, ag.get(p).data, "3 * 7 == 21");

    ag.backward(p);
    ASSERT_FLOAT_EQ(7.0f, ag.get(a).grad, "d(a*b)/da == b == 7");
    ASSERT_FLOAT_EQ(3.0f, ag.get(b).grad, "d(a*b)/db == a == 3");
    printf("\n");
}

// value_div -----------------------------------------------------------------
static void test_div(AutoGrad<DataType> &ag)
{
    printf("test_div ... ");
    NodeHandle a = ag.value_leaf(10.0f);
    NodeHandle b = ag.value_leaf(4.0f);
    NodeHandle q = ag.value_div(a, b);

    ASSERT_FLOAT_EQ(2.5f, ag.get(q).data, "10 / 4 == 2.5");

    ag.backward(q);
    ASSERT_FLOAT_EQ(0.25f,    ag.get(a).grad, "d(a/b)/da == 1/b == 0.25");
    ASSERT_FLOAT_EQ(-0.625f, ag.get(b).grad, "d(a/b)/db == -a/b^2 == -0.625");
    printf("\n");
}

// value_pow -----------------------------------------------------------------
static void test_pow(AutoGrad<DataType> &ag)
{
    printf("test_pow ... ");
    NodeHandle a = ag.value_leaf(2.0f);
    NodeHandle r = ag.value_pow(a, 3.0f);

    ASSERT_FLOAT_EQ(8.0f, ag.get(r).data, "2^3 == 8");

    ag.backward(r);
    ASSERT_FLOAT_EQ(12.0f, ag.get(a).grad, "d(a^3)/da == 3*a^2 == 12");
    printf("\n");
}

// value_log -----------------------------------------------------------------
static void test_log(AutoGrad<DataType> &ag)
{
    printf("test_log ... ");
    NodeHandle a = ag.value_leaf(std::exp(1.0f));  // e
    NodeHandle r = ag.value_log(a);

    ASSERT_FLOAT_EQ(1.0f, ag.get(r).data, "log(e) == 1");

    ag.backward(r);
    ASSERT_FLOAT_EQ(1.0f / std::exp(1.0f), ag.get(a).grad, "d(log(a))/da == 1/e");
    printf("\n");
}

// value_exp -----------------------------------------------------------------
static void test_exp(AutoGrad<DataType> &ag)
{
    printf("test_exp ... ");
    NodeHandle a = ag.value_leaf(2.0f);
    NodeHandle r = ag.value_exp(a);

    float exp2 = std::exp(2.0f);
    ASSERT_FLOAT_EQ(exp2, ag.get(r).data, "exp(2)");

    ag.backward(r);
    ASSERT_FLOAT_EQ(exp2, ag.get(a).grad, "d(exp(a))/da == exp(2)");
    printf("\n");
}

// value_relu ----------------------------------------------------------------
static void test_relu_positive(AutoGrad<DataType> &ag)
{
    printf("test_relu_positive ... ");
    NodeHandle a = ag.value_leaf(3.0f);
    NodeHandle r = ag.value_relu(a);

    ASSERT_FLOAT_EQ(3.0f, ag.get(r).data, "relu(3) == 3");

    ag.backward(r);
    ASSERT_FLOAT_EQ(1.0f, ag.get(a).grad, "d(relu)/da == 1 when a > 0");
    printf("\n");
}

static void test_relu_negative(AutoGrad<DataType> &ag)
{
    printf("test_relu_negative ... ");
    NodeHandle a = ag.value_leaf(-2.0f);
    NodeHandle r = ag.value_relu(a);

    ASSERT_FLOAT_EQ(0.0f, ag.get(r).data, "relu(-2) == 0");

    ag.backward(r);
    ASSERT_FLOAT_EQ(0.0f, ag.get(a).grad, "d(relu)/da == 0 when a <= 0");
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

    ASSERT_FLOAT_EQ(121.0f, ag.get(t3).data, "f(4) == 121");

    ag.backward(t3);
    ASSERT_FLOAT_EQ(44.0f, ag.get(x).grad, "df/dx == 44");
    printf("\n");
}

// Shared node:  loss = a + a  (a is used twice)
//   d(loss)/da should be 2
static void test_shared_node(AutoGrad<DataType> &ag)
{
    printf("test_shared_node ... ");
    NodeHandle a = ag.value_leaf(5.0f);
    NodeHandle s = ag.value_add(a, a);

    ASSERT_FLOAT_EQ(10.0f, ag.get(s).data, "5 + 5 == 10");

    ag.backward(s);
    ASSERT_FLOAT_EQ(2.0f, ag.get(a).grad, "d(a+a)/da == 2 (gradient accumulated)");
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
    ASSERT_INT_EQ(0, (int)ag.allocate(), "first node after reset is index 0");
    printf("\n");
}

// allocate_matrix -----------------------------------------------------------
static void test_allocate_matrix(AutoGrad<DataType> &ag)
{
    printf("test_allocate_matrix ... ");
    NodeHandle offset = ag.allocate_matrix(2, 3, 0.1f);

    // Quick sanity: nodes exist and have data in a reasonable range
    bool ok = true;
    for (int i = 0; i < 6; i++) {
        float val = ag.get(offset + i).data;
        if (std::abs(val) > 3.0f) ok = false;  // 3-sigma bound, very unlikely to fail
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
// Main
// ---------------------------------------------------------------------------
int main()
{
    printf("=== AutoGrad<DataType> Unit Tests ===\n\n");

    AutoGrad<DataType> ag;

    test_leaf_node(ag);
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

    test_relu_positive(ag);
    ag.reset();

    test_relu_negative(ag);
    ag.reset();

    test_chained_computation(ag);
    ag.reset();

    test_shared_node(ag);
    ag.reset();

    test_reset(ag);

    test_allocate_matrix(ag);

    printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
