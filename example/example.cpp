#include "../src/autograd.h"
#include "../src/embeddinglayer.h"
#include "../src/linearlayer.h"
#include "../src/rmsnormlayer.h"
#include "../src/softmaxlayer.h"
#include "../src/parametercheckpoint.h"
#include "../src/transformerblock.h"
#include "../src/adamoptimizer.h"

#include <algorithm>
#include <vector>

// =============================================================================
// DATA LOADING AND VOCABULARY CONSTRUCTION
// =============================================================================

// --- Dataset and Resource Limits ---
constexpr int vocab_size = 27; // (unique characters) + BOS token
constexpr int MAX_NUM_DOCUMENTS = 40000;   // Maximum number of documents (names) to load
constexpr int MAX_DOC_LENGTH = 32;      // Maximum length of a single document (in characters)
constexpr int MAX_NUM_TOKENS = MAX_DOC_LENGTH + 4;  // Maximum tokens per document (doc + BOS tokens)
constexpr float val_split_ratio = 0.05f;    // Fraction of data for validation

// The dataset consists of a list of documents (names). The vocabulary is built
// from the unique characters in the dataset, plus a special BOS (Beginning of
// Sequence) token appended at the end.
static char g_documents[MAX_NUM_DOCUMENTS][MAX_DOC_LENGTH];
static int  g_num_documents = 0;

// Train/validation split indices.
static int  g_train_indices[MAX_NUM_DOCUMENTS];
static int  g_num_train = 0;
static int  g_val_indices[MAX_NUM_DOCUMENTS];
static int  g_num_val = 0;

static int  g_token_bos = 26;                 // Token ID for the BOS (Beginning of Sequence) token

// Map a character to its token ID. Returns -1 if the character is not in the vocabulary.
static int char_to_token_id(char c)
{
    int result = c - 'a';
    if (result < 0 || result > 25)
        result = 0;
    return result;
}

// Load documents (names) from a text file.
// Each line in the file is treated as a separate document.
// @param file_path  Path to the input text file
static void load_documents(const char * file_path)
{
    FILE * file = std::fopen(file_path, "r");
    if (!file) {
        std::fprintf(stderr,
            "Error: '%s' not found. Please download names.txt:\n"
            "  curl -o input.txt https://raw.githubusercontent.com/"
            "karpathy/makemore/988aa59/names.txt\n", file_path);
        std::exit(1);
    }

    char line_buffer[MAX_DOC_LENGTH];
    while (std::fgets(line_buffer, sizeof(line_buffer), file)) {
        // Remove trailing whitespace (newlines, carriage returns, spaces, tabs).
        int line_length = (int)std::strlen(line_buffer);
        while (line_length > 0 &&
            (line_buffer[line_length - 1] == '\n' ||
                line_buffer[line_length - 1] == '\r' ||
                line_buffer[line_length - 1] == ' ' ||
                line_buffer[line_length - 1] == '\t')) {
            line_buffer[--line_length] = '\0';
        }

        // Skip empty lines.
        if (line_length == 0) continue;

        // Stop if we've reached the maximum number of documents.
        if (g_num_documents >= MAX_NUM_DOCUMENTS) break;

        // Copy the cleaned line into the documents array.
        std::strncpy(g_documents[g_num_documents], line_buffer, MAX_DOC_LENGTH - 1);
        g_documents[g_num_documents][MAX_DOC_LENGTH - 1] = '\0';
        g_num_documents++;
    }

    std::fclose(file);
}

// Shuffle the documents array in place
static void shuffle_documents()
{
    std::mt19937_64 rng{ 42 };
    std::shuffle(g_documents, g_documents + g_num_documents, rng);
}

// Split documents into train and validation sets.
// After shuffling, the first (1 - val_split_ratio) documents go to train,
// and the rest go to validation.
static void split_train_val()
{
    int num_val = (int)((float)g_num_documents * val_split_ratio);
    if (num_val < 1) num_val = 1;  // At least one validation document.
    int num_train = g_num_documents - num_val;

    g_num_train = num_train;
    g_num_val = num_val;

    for (int i = 0; i < num_train; i++) {
        g_train_indices[i] = i;
    }
    for (int i = 0; i < num_val; i++) {
        g_val_indices[i] = num_train + i;
    }
}

struct Model
{
    // model hyperparameters
    static const int num_layers = 1; // Depth of the transformer (number of stacked layers)
    static const int emb_dim = 16; // Width of the network (embedding dimension)
    static const int block_size = 16; // Maximum context length for the attention window
    static const int num_heads = 4;   // Number of parallel attention heads
    static const int head_dim = emb_dim / num_heads;  // Dimension per attention head (16/4=4)
    static const int FOUR_X_EMB_DIM = 4 * emb_dim;          // MLP hidden dimension (4x embedding size)
    static constexpr float std_dev = 0.08f;

    EmbeddingLayer<float> wte;
    EmbeddingLayer<float> wpe;
    SimpleRMSNormLayer<float> embed_norm;
    LinearLayer<float> lm_head;
    std::vector <TransformerBlock<float> > transformer;

    void init(AutoGrad<float> & grad, int vocab_size)
    {
        // Token embedding table: maps each token ID to an emb_dim-dimensional vector.
        wte.init(grad, vocab_size, emb_dim, std_dev);

        // Position embedding table: maps each position to an emb_dim-dimensional vector.
        wpe.init(grad, block_size, emb_dim, std_dev);

        // Language model head (unembedding): maps hidden state to vocabulary logits.
        lm_head.init(grad, vocab_size, emb_dim, std_dev);

        // Per-layer weights.
        transformer.resize(num_layers);
        for (int li = 0; li < num_layers; li++)
        {
            transformer[li].init(grad, emb_dim, num_heads, block_size, FOUR_X_EMB_DIM, std_dev);
        }
    }

    void reset_kv_cache()
    {
        for (int li = 0; li < num_layers; li++)
        {
            transformer[li].reset_kv_cache();
        }
    }

    void forward(AutoGrad<float> & grad,
        int input_token_id,
        int input_position,
        std::span<NodeHandle> output)
    {
        // -----------------------------------------------------------------------
        // STEP 1: TOKEN AND POSITION EMBEDDING
        // -----------------------------------------------------------------------
        // Look up the token embedding from wte[token_id] and position embedding
        // from wpe[position], then add them together and apply RMSNorm.

        std::array <NodeHandle, emb_dim> token_embedded, pos_embedded, embedded;
        wte.forward(grad, input_token_id, token_embedded);
        wpe.forward(grad, input_position, pos_embedded);

        for (int j = 0; j < emb_dim; j++)
        {
            // x[j] = wte[token_id][j] + wpe[position][j]
            embedded[j] = grad.value_add(token_embedded[j], pos_embedded[j]);
        }

        // Apply RMSNorm to the combined embedding.
        std::array <NodeHandle, emb_dim> embedded_normalized;
        embed_norm.forward(grad, embedded, embedded_normalized);
        
        // -----------------------------------------------------------------------
        // STEP 2: TRANSFORMER LAYERS
        // -----------------------------------------------------------------------
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
        {
            // output back to the (now unused) embedded vector
            transformer[layer_idx].forward(grad, embedded_normalized, embedded);
        }

        // -----------------------------------------------------------------------
        // STEP 3: LANGUAGE MODEL HEAD (UNEMBEDDING)
        // -----------------------------------------------------------------------
        // Project the final hidden state to vocabulary-size logits.
        // These logits represent the unnormalized log probabilities for the next token.
        lm_head.forward(grad, embedded, output);
    }
};

// Compute validation loss by running forward pass on all validation documents.
// Returns the average negative log likelihood over all validation positions.
float compute_validation_loss(ParameterCheckpoint<float> & checkpoint, AutoGrad<float> & grad, Model & model)
{
    std::array <int, vocab_size> logit_nodes;
    std::array <int, vocab_size> prob_nodes;
    std::array <int, MAX_NUM_TOKENS> token_sequence;

    float total_loss = 0.0f;
    int total_positions = 0;

    for (int doc_idx = 0; doc_idx < g_num_val; doc_idx++)
    {
        int doc_file_idx = g_val_indices[doc_idx];
        const char * current_doc = g_documents[doc_file_idx];
        int doc_length = (int)std::strlen(current_doc);

        // Reset KV cache & pool for each document.
        grad.restore_parameter_values(checkpoint.values);
        model.reset_kv_cache();

        // Tokenize the document.
        int num_tokens = 0;
        token_sequence[num_tokens++] = g_token_bos;
        for (int i = 0; i < doc_length; i++) {
            token_sequence[num_tokens++] = char_to_token_id(current_doc[i]);
        }
        token_sequence[num_tokens++] = g_token_bos;

        int num_positions = num_tokens - 1;
        if (num_positions > model.block_size)
        {
            num_positions = model.block_size;
        }

        // Forward pass without accumulating gradients (no backward pass).
        for (int pos = 0; pos < num_positions; pos++)
        {
            int current_token = token_sequence[pos];
            int target_token = token_sequence[pos + 1];

            model.forward(grad, current_token, pos, logit_nodes);
            
            SoftmaxLayer<float> softmax_layer;
            softmax_layer.forward(grad, logit_nodes, prob_nodes);

            // Compute negative log likelihood for this position.
            float prob = grad.get(prob_nodes[target_token]).data;
            if (prob < 1e-7f) prob = 1e-7f;  // Clamp for numerical stability.
            total_loss -= std::log(prob);
            total_positions++;
        }
    }

    if (total_positions == 0) return 0.0f;
    return total_loss / (float)total_positions;
}

int main(void)
{
    // -----------------------------------------------------------------------
    // PHASE 0: CONSTANTS
    // -----------------------------------------------------------------------

    // --- Training Configuration ---
    constexpr int log_interval = 100;     // Print training status every N steps
    constexpr int val_interval = 1000;     // Evaluate validation loss every N steps

    // -----------------------------------------------------------------------
    // PHASE 1: DATA LOADING AND PREPROCESSING
    // -----------------------------------------------------------------------
    load_documents("names.txt");
    std::printf("num docs: %d\n", g_num_documents);

    std::printf("vocab size: %d\n", vocab_size);

    shuffle_documents();

    split_train_val();
    std::printf("train docs: %d, val docs: %d\n", g_num_train, g_num_val);

    // -----------------------------------------------------------------------
    // PHASE 2: INITIALIZE MODEL PARAMETERS
    // -----------------------------------------------------------------------
    // Each parameter is a leaf Value node in the pool.
    // Matrices are initialized with small Gaussian random values.

    const size_t k_MB = 1024 * 1024;
    AutoGrad<float> grad;
    // 20MB allocation for parameters
    // run at least one training iteration in debug mode to see if you're going to run out of memory
    grad.init(20 * k_MB);

    Model model;
    model.init(grad, vocab_size);

    AdamOptimizer<float> optimizer;

    // Record the total number of parameters and copy initial values to persistent storage.
    ParameterCheckpoint<float> checkpoint;
    checkpoint.init(grad);
    
    std::printf("num params: %d\n", (int)checkpoint.size());

    // -----------------------------------------------------------------------
    // PHASE 3: TRAINING LOOP
    // -----------------------------------------------------------------------
    const int    num_training_steps = 10000;
    const float base_learning_rate = 0.01f;
    const float adam_beta1 = 0.9f;   // Exponential decay rate for the first moment
    const float adam_beta2 = 0.999f; // Exponential decay rate for the second moment
    const float adam_epsilon = 1e-8f; // Small constant for numerical stability

    // Buffers for tokens, logits, and probabilities.
    std::array <NodeHandle, MAX_NUM_TOKENS> token_sequence;
    std::array <NodeHandle, vocab_size> logit_nodes;
    std::array <NodeHandle, vocab_size> prob_nodes;

    for (int step = 0; step < num_training_steps; step++)
    {
        // Restore parameter values into the pool (pool is rebuilt each step).
        grad.restore_parameter_values(checkpoint.values);

        // Reset the KV cache for a fresh forward pass.
        model.reset_kv_cache();

        // Select a document from the training set and tokenize it.
        // Tokens are surrounded by BOS tokens on both sides.
        const char * current_doc = g_documents[g_train_indices[step % g_num_train]];
        int doc_length = (int)std::strlen(current_doc);
        int num_tokens = 0;

        token_sequence[num_tokens++] = g_token_bos;  // Leading BOS token
        for (int i = 0; i < doc_length; i++) 
        {
            token_sequence[num_tokens++] = char_to_token_id(current_doc[i]);
        }
        token_sequence[num_tokens++] = g_token_bos;  // Trailing BOS token (end of sequence)

        // Number of training positions (we predict the next token, so n = num_tokens - 1).
        int num_positions = num_tokens - 1;
        if (num_positions > model.block_size) 
        {
            num_positions = model.block_size;  // Clamp to the maximum context length.
        }

        // Forward pass: compute the loss as the average negative log likelihood.
        int loss_accumulator = -1;  // Sentinel value indicating "not yet initialized"

        for (int pos = 0; pos < num_positions; pos++)
        {
            int current_token = token_sequence[pos];
            int target_token = token_sequence[pos + 1];  // The token we want to predict

            // Run the GPT forward pass to get logits for the next token.
            model.forward(grad, current_token, pos, logit_nodes);

            // Convert logits to probabilities via softmax.
            SoftmaxLayer<float> softmax_layer;
            softmax_layer.forward(grad, logit_nodes, prob_nodes);

            // Compute negative log likelihood for the target token:
            // loss = -log(prob[target_token])
            int neg_log_prob = grad.value_mul_const(grad.value_log(prob_nodes[target_token]), -1);

            // Accumulate the loss (initialize on first iteration, then add).
            if (loss_accumulator < 0) {
                loss_accumulator = neg_log_prob;
            }
            else {
                loss_accumulator = grad.value_add(loss_accumulator, neg_log_prob);
            }
        }

        // Average the loss over all positions in the sequence.
        int loss_node = grad.value_mul_const(loss_accumulator, 1.0 / num_positions);

        // Backward pass: compute gradients for all parameters.
        grad.backward(loss_node);

        // capture current parameter values in the checkpoint
        checkpoint.update(grad);

        // -------------------------------------------------------------------
        // PARAMETER UPDATE (ADAM OPTIMIZER)
        // -------------------------------------------------------------------
        // Apply linear learning rate decay: lr decreases from base_lr to 0.
        float current_lr = base_learning_rate * (1.0f - (float)step / (float)num_training_steps);

        optimizer.step(checkpoint);

        // Log training progress and periodically evaluate validation loss.
        float current_val_loss = 0.0f;
        float train_loss = grad.get(loss_node).data;
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
                std::printf("step %4d / %4d | train_loss %.4f\n",
                    step + 1, num_training_steps, train_loss);
            }
            std::fflush(stdout);
        }
    }

    std::printf("\n");

    // -----------------------------------------------------------------------
    // PHASE 4: INFERENCE (TEXT GENERATION)
    // -----------------------------------------------------------------------
    // Generate new samples by autoregressively predicting tokens.
    // Temperature scaling controls the "creativity" of the output.

    const float temperature = 0.5f;  // Lower = more deterministic, higher = more creative

    std::printf("--- inference (new, hallucinated names) ---\n");

    std::mt19937_64 rng{ 42 };
    std::uniform_real_distribution<float> unif(0, 1);

    for (int sample_idx = 0; sample_idx < 20; sample_idx++)
    {
        // Restore parameters and reset the KV cache for each sample.
        grad.restore_parameter_values(checkpoint.values);
        model.reset_kv_cache();

        // Start with the BOS token.
        int current_token = g_token_bos;

        // Buffer to store the generated characters.
        char generated_text[model.block_size + 1];
        int text_length = 0;

        // Generate tokens one at a time, up to block_size positions.
        for (int pos = 0; pos < model.block_size; pos++)
        {
            // Forward pass to get logits for the next token.
            model.forward(grad, current_token, pos, logit_nodes);

            // Apply temperature scaling: logits / temperature
            // Lower temperature sharpens the distribution (more confident).
            std::array <int,vocab_size> scaled_logits;
            if (temperature > 0)
            {
                for (int i = 0; i < vocab_size; i++)
                {
                    scaled_logits[i] = grad.value_mul_const(logit_nodes[i], 1.0f / temperature);
                }
            }

            // Convert scaled logits to probabilities.
            SoftmaxLayer<float> softmax;
            softmax.forward(grad, scaled_logits, prob_nodes);

            // Sample the next token from the probability distribution.
            // Use cumulative probability sampling.
            float random_threshold = unif(rng);
            float cumulative_prob = 0.0f;
            int next_token = vocab_size - 1;  // Default to last token

            for (int i = 0; i < vocab_size; i++)
            {
                cumulative_prob += grad.get(prob_nodes[i]).data;
                if (random_threshold < cumulative_prob)
                {
                    next_token = i;
                    break;
                }
            }

            // If we sample the BOS token, treat it as the end of the sequence.
            if (next_token == g_token_bos) break;

            // Convert the token ID back to a character and append to the output.
            generated_text[text_length++] = ((char)next_token) + 'a';

            // The sampled token becomes the input for the next position.
            current_token = next_token;
        }

        // Null-terminate and print the generated text.
        generated_text[text_length] = '\0';
        std::printf("sample %2d: %s\n", sample_idx + 1, generated_text);
    }

    // -----------------------------------------------------------------------
    // PHASE 5: SAVE FINAL WEIGHTS
    // -----------------------------------------------------------------------
    // Write the final parameter values to a binary file.

    checkpoint.saveToFile("checkpoint.chk");

    return 0;
}