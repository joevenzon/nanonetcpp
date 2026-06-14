#include "../src/autograd.h"
#include "../src/embeddinglayer.h"
#include "../src/linearlayer.h"
#include "../src/rmsnormlayer.h"
#include "../src/softmaxlayer.h"
#include "../src/conv2dlayer.h"
#include "../src/parametercheckpoint.h"
#include "../src/adamoptimizer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <span>
#include <vector>
#include <cmath>

constexpr double PI = 3.14159265358979323846;

// =============================================================================
// DATA LOADING
// =============================================================================

static const int k_rows = 28;
static const int k_cols = 28;
static const int DOC_LENGTH = k_rows * k_cols;

// images
typedef std::array <unsigned char, DOC_LENGTH> t_document;
static std::array <t_document, 60000> g_train_documents;
static std::array <t_document, 10000> g_val_documents;

// answers
typedef int t_label;
static std::array <t_label, 60000> g_train_labels;
static std::array <t_label, 10000> g_val_labels;

static int g_train_doc_count = 0;
static int g_val_doc_count = 0;

// Load labels from a binary file in MNIST idx1-ubyte format.
// Returns the number of labels loaded, or -1 on error.
static int load_labels(const char * file_path, std::span <t_label> output)
{
    FILE * file = std::fopen(file_path, "rb");
    if (!file){
        std::fprintf(stderr,
            "Error: '%s' not found. Please download the MNIST dataset in ubyte format.\n",
            file_path);
        return -1;
    }

    // Read header: magic number, label count (big-endian uint32)
    uint32_t header[2];
    if (std::fread(reinterpret_cast<char *>(header), sizeof(uint32_t), 2, file) != 2) {
        std::fprintf(stderr, "Error: failed to read header from '%s'\n", file_path);
        std::fclose(file);
        return -1;
    }

    // helper to byteswap big to little endian
    auto read_be32 = [](uint32_t raw) -> uint32_t {
        return (static_cast<uint32_t>(raw >> 24)) |
               ((raw >> 8) & 0xFF00) |
               ((raw << 8) & 0xFF0000) |
               (static_cast<uint32_t>(raw << 24));
    };

    uint32_t magic      = read_be32(header[0]);
    uint32_t label_count = read_be32(header[1]);

    // MNIST label magic number is 2049 (0x00000801)
    if (magic != 2049) {
        std::fprintf(stderr, "Error: '%s' has invalid magic number %u (expected 2049)\n", file_path, magic);
        std::fclose(file);
        return -1;
    }

    // Make sure the output span can hold all labels
    if (label_count > static_cast<uint32_t>(output.size())) {
        std::fprintf(stderr, "Error: '%s' contains %u labels, but output can only hold %zu\n",
            file_path, label_count, output.size());
        std::fclose(file);
        return -1;
    }

    // Read label data: each label is a single unsigned byte (values 0-9)
    std::vector<unsigned char> label_data(label_count);
    size_t read_count = std::fread(label_data.data(), 1, label_count, file);
    std::fclose(file);

    if (read_count != label_count) {
        std::fprintf(stderr, "Error: failed to read all labels from '%s' (got %zu, expected %u)\n",
            file_path, read_count, label_count);
        return -1;
    }

    // Convert unsigned bytes to t_label integers
    for (uint32_t i = 0; i < label_count; ++i) {
        output[i] = static_cast<t_label>(label_data[i]);
    }

    return static_cast<int>(label_count);
}

// Load documents (28x28 images) from a binary file in MNIST idx3-ubyte format.
// Returns the number of documents loaded, or -1 on error.
static int load_documents(const char * file_path, std::span <t_document> output)
{
    FILE * file = std::fopen(file_path, "rb");
    if (!file) {
        std::fprintf(stderr,
            "Error: '%s' not found. Please download the MNIST dataset in ubyte format.\n",
            file_path);
        return -1;
    }

    // Read header: magic number, image count, row count, column count (big-endian uint32)
    uint32_t header[4];
    if (std::fread(reinterpret_cast<char *>(header), sizeof(uint32_t), 4, file) != 4) {
        std::fprintf(stderr, "Error: failed to read header from '%s'\n", file_path);
        std::fclose(file);
        return -1;
    }

    // helper to byteswap big to little endian
    auto read_be32 = [](uint32_t raw) -> uint32_t {
        return (static_cast<uint32_t>(raw >> 24)) |
               ((raw >> 8) & 0xFF00) |
               ((raw << 8) & 0xFF0000) |
               (static_cast<uint32_t>(raw << 24));
    };

    uint32_t magic     = read_be32(header[0]);
    uint32_t img_count = read_be32(header[1]);
    uint32_t rows      = read_be32(header[2]);
    uint32_t cols      = read_be32(header[3]);

    // MNIST image magic number is 2051 (0x00000803)
    if (magic != 2051) {
        std::fprintf(stderr, "Error: '%s' has invalid magic number %u (expected 2051)\n", file_path, magic);
        std::fclose(file);
        return -1;
    }

    // Verify dimensions match our expectations
    if (rows != static_cast<uint32_t>(k_rows) || cols != static_cast<uint32_t>(k_cols)) {
        std::fprintf(stderr, "Error: '%s' has dimensions %ux%u, expected %dx%d\n",
            file_path, rows, cols, k_rows, k_cols);
        std::fclose(file);
        return -1;
    }

    // Make sure the output span can hold all images
    if (img_count > static_cast<uint32_t>(output.size())) {
        std::fprintf(stderr, "Error: '%s' contains %u images, but output can only hold %zu\n",
            file_path, img_count, output.size());
        std::fclose(file);
        return -1;
    }

    // Read pixel data directly into the output span (no temp buffer needed)
    size_t total_bytes = img_count * rows * cols;
    size_t read_count = std::fread(reinterpret_cast<unsigned char *>(output.data()), 1, total_bytes, file);
    std::fclose(file);

    if (read_count != total_bytes) {
        std::fprintf(stderr, "Error: '%s' truncated — expected %zu bytes, got %zu\n",
            file_path, total_bytes, read_count);
        return -1;
    }

    return static_cast<int>(img_count);
}

static bool load_all_documents()
{
    g_val_doc_count = load_documents("t10k-images-idx3-ubyte", std::span(g_val_documents.begin(), g_val_documents.end()));
    g_train_doc_count = load_documents("train-images-idx3-ubyte", std::span(g_train_documents.begin(), g_train_documents.end()));
    
    int val_label_count = load_labels("t10k-labels-idx1-ubyte", std::span(g_val_labels.begin(), g_val_labels.end()));
    int train_label_count = load_labels("train-labels-idx1-ubyte", std::span(g_train_labels.begin(), g_train_labels.end()));

    if (val_label_count != g_val_doc_count)
    {
        std::fprintf(stderr, "Error: have %d images but loaded %d labels\n", g_val_doc_count, val_label_count);
        return false;
    }

    if (train_label_count != g_train_doc_count)
    {
        std::fprintf(stderr, "Error: have %d images but loaded %d labels\n", g_train_doc_count, train_label_count);
        return false;
    }

    return g_val_doc_count > 0 && g_train_doc_count > 0;
}

// =============================================================================
// MODEL
// =============================================================================

struct Model
{
    // model hyperparameters
    static const int feature_map_1_size = 32;
    static const int feature_map_2_size = 64;
    static const int kernel_size = 3;
    static const int pool_amount = 2;
    static const int hidden_layer_size = 128;

    Conv2dLayer<float> conv1;
    Conv2dLayer<float> conv2;
    LinearLayer<float> fc1;
    LinearLayer<float> fc2;
    
    void init(AutoGrad<float> & grad)
    {
        conv1.init(grad, k_rows, k_cols, 1, feature_map_1_size, kernel_size, 1, 1, true, 0, "conv1"); // 28x28 -> 32x28x28
        conv2.init(grad, k_rows/2, k_cols/2, feature_map_1_size, feature_map_2_size, kernel_size, 1, 1, true, 0, "conv1"); // 14x14 -> 64x14x14
        fc1.init(grad, feature_map_2_size * (k_rows / 4) * (k_cols / 4), hidden_layer_size, true, 0, "fc1");
        fc2.init(grad, hidden_layer_size, 10, true, 0, "fc1"); // 10 number outputs
    }

    TensorHandle forward(AutoGrad<float> & grad, TensorHandle flattened_image)
    {
        TensorHandle imagefilters1 = grad.value_relu(conv1.forward(grad, flattened_image)); // creates 32 28x28 filters
        TensorHandle downsample_imagefilters1 = grad.value_max_pool2d(imagefilters1, k_rows, k_cols, pool_amount); // downsample to 32 14x14 filters
        TensorHandle imagefilters2 = grad.value_relu(conv2.forward(grad, downsample_imagefilters1)); // double image filters to 64 14x14 filters
        TensorHandle downsample_imagefilters2 = grad.value_max_pool2d(imagefilters2, k_rows/2, k_cols/2, pool_amount); // downsample to 64 7x7 filters
        TensorHandle flattened = grad.value_reshape(downsample_imagefilters2, TensorShape({ 1, feature_map_2_size * (k_rows / 4) * (k_cols / 4) })); // flatten to 64*7*7 input size for the linear layer
        TensorHandle intermediate = grad.value_relu(fc1.forward(grad, flattened));
        TensorHandle output = fc2.forward(grad, intermediate);

        return output;
    }
};

// =============================================================================
// VALIDATION
// =============================================================================

TensorHandle document_to_tensor(AutoGrad<float> & grad, const t_document & doc)
{
    // package up the document as a tensor
    TensorHandle input = grad.tensor_leaf({ k_rows * k_cols, 1 }); // { pixel data, color channels }
    for (int j = 0; j < k_rows * k_cols; j++)
    {
        grad.get(input).tensor(j, 0) = float(doc[j])/255.0f;
    }

    return input;
}

// Compute validation loss by running forward pass over the validation set.
// Returns the average negative log likelihood over all validation positions.
float compute_validation_loss(
    ParameterCheckpoint<float> & checkpoint,
    AutoGrad<float> & grad, 
    Model & model,
    int num_to_check,
    float * out_percent_correct = NULL)
{
    float total_loss = 0.0f;
    int correct = 0;
    
    for (int i = 0; i < num_to_check; i++)
    {
        // reset memory use
        grad.restore_allocators();

        // get an input
        TensorHandle input = document_to_tensor(grad, g_val_documents[i]);

        // get prediction likelihoods, one output per number, shape {1, 10}
        TensorHandle logits = model.forward(grad, input);

        // compare against the expected label
        TensorHandle loss = grad.value_cross_entropy_loss(
            logits, std::span<const int>(&g_val_labels[i], 1));

        // add up loss
        total_loss += grad.get(loss).tensor.values().data()[0];

        // see which result was the actual guess, and if it was correct
        int guess_index = 0;
        const int correct_answer = g_val_labels[i];
        for (int j = 0; j < 10; j++)
        {
            if (grad.get(logits).tensor(0, j) > grad.get(logits).tensor(0, guess_index))
                guess_index = j;
        }

        if (guess_index == correct_answer)
            correct++;
    }

    if (out_percent_correct)
        *out_percent_correct = correct / (float)num_to_check;

    // divide by number to get the mean of the squared errors
    return total_loss / (float)num_to_check;
}

// =============================================================================
// MAIN
// =============================================================================

int main(void)
{
    // -----------------------------------------------------------------------
    // PHASE 0: CONSTANTS
    // -----------------------------------------------------------------------

    const int    num_training_steps = 6000;
    const float base_learning_rate = 0.005f;
    const int    accumulation_steps = 10;
    constexpr int log_interval = 10;     // Print training status every N steps
    constexpr int val_interval = 100;     // Evaluate validation loss every N steps
    constexpr int validation_number = 100;

    // -----------------------------------------------------------------------
    // PHASE 1: DATA LOADING AND PREPROCESSING
    // -----------------------------------------------------------------------
    
    bool loaded = load_all_documents();
    if (!loaded)
    {
        std::fprintf(stderr, "Load error, exiting\n");
        return -1;
    }

    // -----------------------------------------------------------------------
    // PHASE 2: INITIALIZE MODEL PARAMETERS
    // -----------------------------------------------------------------------
    
    AutoGrad<float> grad;
    grad.init(42, 8192, 1024*1024);

    Model model;
    model.init(grad);
    grad.snapshot_parameters();

    AdamOptimizer<float> optimizer;

    // Record the total number of parameters and copy initial values (which are the model weights) to
    // persistent storage in the checkpoint
    ParameterCheckpoint<float> checkpoint;
    checkpoint.init(grad);

    std::printf("num params: %d\n", (int)checkpoint.size());

    float percent_correct = 0;
    float initial_val_loss = compute_validation_loss(checkpoint, grad, model, validation_number, &percent_correct);
    std::printf("init validation loss: %f (%.2f%% correct)\n", initial_val_loss, percent_correct*100.f);

    // -----------------------------------------------------------------------
    // PHASE 3: TRAINING LOOP
    // -----------------------------------------------------------------------

    // Start the training timer.
    std::chrono::steady_clock::time_point train_start = std::chrono::steady_clock::now();

    int document_index = 0;

    for (int step = 0; step < num_training_steps; step++)
    {
        grad.zero_grad();

        float last_train_loss = 0.0f;

        // --- GRADIENT ACCUMULATION LOOP ---
        for (int acc = 0; acc < accumulation_steps; acc++)
        {
            int micro_step = step * accumulation_steps + acc;

            // get a training input/output pair
            TensorHandle input = document_to_tensor(grad, g_train_documents[document_index % g_train_doc_count]);
            std::span<const int> expected_output(&g_train_labels[document_index % g_train_doc_count], 1);

            // Run the model forward
            // gets prediction likelihoods, one output per number, shape {1, 10}
            TensorHandle logits = model.forward(grad, input);

            // compute loss: compare against the expected label
            TensorHandle loss = grad.value_cross_entropy_loss(logits, expected_output);

            // Scale loss for accumulation: divide by accumulation_steps so that
            // the sum of gradients across micro-steps equals the mean.
            float scale = 1.0f / ((float)accumulation_steps);
            TensorHandle loss_node = grad.value_mul_const(loss, scale);

            // Backward pass: compute gradients for all parameters.
            bool zero_gradients = false;
            grad.backward(loss_node, zero_gradients);

            last_train_loss += grad.get(loss_node).tensor.values().data()[0];

            document_index++;
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
            current_val_loss = compute_validation_loss(checkpoint, grad, model, validation_number, &percent_correct);
        }

        if (do_print) {
            if (do_val) {
                std::printf("step %4d / %4d | train_loss %.4f | val_loss %.4f (%.2f%% correct)\n",
                    step + 1, num_training_steps, train_loss, current_val_loss, percent_correct * 100.0f);
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

    /*// -----------------------------------------------------------------------
    // PHASE 4: INFERENCE
    // -----------------------------------------------------------------------
    // Show off the result by predicting tokens.

    std::printf("--- inference ---\n");

    for (int sample_idx = 0; sample_idx < 20; sample_idx++)
    {
        // this resets memory use, sort of like "with torch.no_grad():"
        grad.restore_allocators();

        
    }

    std::printf("node pool high water mark: %d\n", (int)grad.node_high_water_mark());
    std::printf("value arena high water mark: %d\n", (int)grad.value_high_water_mark());*/

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
