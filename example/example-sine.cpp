#include "../src/autograd.h"
#include "../src/embeddinglayer.h"
#include "../src/linearlayer.h"
#include "../src/rmsnormlayer.h"
#include "../src/softmaxlayer.h"
#include "../src/parametercheckpoint.h"
#include "../src/transformerblock.h"
#include "../src/adamoptimizer.h"

#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>

constexpr double PI = 3.14159265358979323846;

// =============================================================================
// DATA LOADING
// =============================================================================

// =============================================================================
// MODEL
// =============================================================================

struct Model
{
    // model hyperparameters
    static const int hidden_dim = 64;

    LinearLayer<float> first_layer;
    LinearLayer<float> second_layer;
    LinearLayer<float> output_layer;

    void init(AutoGrad<float> & grad)
    {
        first_layer.init(grad, 1, hidden_dim);
        second_layer.init(grad, hidden_dim, hidden_dim);
        output_layer.init(grad, hidden_dim, 1);
    }

    TensorHandle forward(AutoGrad<float> & grad, float inputvalue)
    {
        TensorHandle input = grad.tensor_leaf(TensorShape{ 1,1 }, inputvalue);
        TensorHandle intermediate = first_layer.forward(grad, input);
        intermediate = grad.value_tanh(intermediate);
        intermediate = second_layer.forward(grad, intermediate);
        intermediate = grad.value_tanh(intermediate);
        TensorHandle output = output_layer.forward(grad, intermediate);

        return output;
    }
};

// =============================================================================
// VALIDATION
// =============================================================================

// Compute validation loss by running forward pass over the validation set.
// Returns the average negative log likelihood over all validation positions.
float compute_validation_loss(
    ParameterCheckpoint<float> & checkpoint,
    AutoGrad<float> & grad, Model & model)
{
    float total_loss = 0.0f;
    const int num_val = 1000;

    static std::random_device rd;
    static std::mt19937 gen;
    gen.seed(123);
    std::uniform_real_distribution<float> uniform(-PI, PI);

    for (int set_idx = 0; set_idx < num_val; set_idx++)
    {
        // this resets memory use, sort of like "with torch.no_grad():"
        grad.restore_allocators();

        // select a random input in the range from above
        float input = uniform(gen);

        // make a prediction
        TensorHandle prediction = model.forward(grad, input);

        // extract the float value from the tensor handle
        float prediction_value = grad.get(prediction).tensor.values()[0];
        float error = prediction_value - sin(input);
        
        // sum of squared errors
        total_loss += error * error;
    }

    // divide by number to get the mean of the squared errors
    return total_loss / (float)num_val;
}

// =============================================================================
// MAIN
// =============================================================================

int main(void)
{
    // -----------------------------------------------------------------------
    // PHASE 0: CONSTANTS
    // -----------------------------------------------------------------------

    const int    num_training_steps = 2000;
    const float base_learning_rate = 0.005f;
    const int    accumulation_steps = 128;
    constexpr int log_interval = 10;     // Print training status every N steps
    constexpr int val_interval = 100;     // Evaluate validation loss every N steps

    // -----------------------------------------------------------------------
    // PHASE 1: DATA LOADING AND PREPROCESSING
    // -----------------------------------------------------------------------
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform(-PI, PI);
    gen.seed(321);

    // -----------------------------------------------------------------------
    // PHASE 2: INITIALIZE MODEL PARAMETERS
    // -----------------------------------------------------------------------
    
    AutoGrad<float> grad;

    Model model;
    model.init(grad);
    grad.snapshot_parameters();

    AdamOptimizer<float> optimizer;

    // Record the total number of parameters and copy initial values (which are the model weights) to
    // persistent storage in the checkpoint
    ParameterCheckpoint<float> checkpoint;
    checkpoint.init(grad);

    std::printf("num params: %d\n", (int)checkpoint.size());

    float initial_val_loss = compute_validation_loss(checkpoint, grad, model);
    std::printf("init validation loss: %f\n", initial_val_loss);

    // -----------------------------------------------------------------------
    // PHASE 3: TRAINING LOOP
    // -----------------------------------------------------------------------

    // Start the training timer.
    std::chrono::steady_clock::time_point train_start = std::chrono::steady_clock::now();

    for (int step = 0; step < num_training_steps; step++)
    {
        grad.zero_grad();

        float last_train_loss = 0.0f;

        // --- GRADIENT ACCUMULATION LOOP ---
        for (int acc = 0; acc < accumulation_steps; acc++)
        {
            int micro_step = step * accumulation_steps + acc;

            // Generate a training input/output pair
            float train_x = uniform(gen);
            float train_y = sin(train_x); // expected value
            
            // Make a prediction
            TensorHandle prediction = model.forward(grad, train_x);

            // Compute loss by comparing to expected value and squaring the error
            TensorHandle loss = grad.value_pow(grad.value_sub_const(prediction, train_y), 2);

            // Scale loss for accumulation: divide by accumulation_steps so that
            // the sum of gradients across micro-steps equals the mean.
            float scale = 1.0f / ((float)accumulation_steps);
            TensorHandle loss_node = grad.value_mul_const(loss, scale);

            // Backward pass: compute gradients for all parameters.
            bool zero_gradients = false;
            grad.backward(loss_node, zero_gradients);

            last_train_loss = grad.get(loss_node).tensor.values().data()[0];
        }

        // capture current parameter values in the checkpoint
        checkpoint.update(grad, optimizer.step_count);

        // -------------------------------------------------------------------
        // PARAMETER UPDATE (ADAM OPTIMIZER)
        // -------------------------------------------------------------------
        // Apply linear learning rate decay: lr decreases from base_lr to 0.
        float current_lr = base_learning_rate * (1.0f - (float)step / (float)num_training_steps);
        optimizer.lr = current_lr;
        optimizer.step(grad);

        // Log training progress and periodically evaluate validation loss.
        float current_val_loss = 0.0f;
        float train_loss = last_train_loss;
        bool do_print = ((step + 1) % log_interval == 0 || step == num_training_steps - 1);
        bool do_val = ((step + 1) % val_interval == 0 || step == num_training_steps - 1);

        if (do_val)
        {
            current_val_loss = compute_validation_loss(checkpoint, grad, model);
        }

        if (do_print) {
            if (do_val) {
                std::printf("step %4d / %4d | train_loss %.4f | val_loss %.4f\n",
                    step + 1, num_training_steps, train_loss, current_val_loss);
            }
            else {
                std::printf("step %4d / %4d | train_loss %.4f\r",
                    step + 1, num_training_steps, train_loss);
            }
            std::fflush(stdout);
        }
    }

    std::printf("\n");

    // Print the total training time.
    std::chrono::steady_clock::time_point train_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> train_duration = train_end - train_start;
    double total_seconds = train_duration.count();
    int minutes = (int)(total_seconds / 60.0);
    double seconds = total_seconds - (double)minutes * 60.0;
    std::printf("training time: %02d:%05.2f\n", minutes, seconds);

    // -----------------------------------------------------------------------
    // PHASE 4: INFERENCE
    // -----------------------------------------------------------------------
    // Show off the result by predicting tokens.

    std::printf("--- inference ---\n");
    gen.seed(999);

    for (int sample_idx = 0; sample_idx < 20; sample_idx++)
    {
        // this resets memory use, sort of like "with torch.no_grad():"
        grad.restore_allocators();

        // select a random input
        float input = uniform(gen);

        // make a prediction
        TensorHandle prediction = model.forward(grad, input);

        // extract the float value from the tensor handle
        float prediction_value = grad.get(prediction).tensor.values()[0];
        float expected = sin(input);

        std::printf("sample %2d %+7.2f: %f (expected %f)\n", sample_idx + 1, input, prediction_value, expected);
    }

    std::printf("node pool high water mark: %d\n", (int)grad.node_high_water_mark());
    std::printf("value arena high water mark: %d\n", (int)grad.value_high_water_mark());

    // -----------------------------------------------------------------------
    // PHASE 5: SAVE FINAL WEIGHTS
    // -----------------------------------------------------------------------
    // Write the final parameter values to a binary file.

    if (checkpoint.saveToFile("checkpoint.chk"))
    {
        std::printf("wrote checkpoint to disk (%d weights)", (int)checkpoint.values.size());
    }

    return 0;
}
