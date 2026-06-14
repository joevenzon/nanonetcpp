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

// =============================================================================
// DATA LOADING AND VOCABULARY CONSTRUCTION
// =============================================================================

// --- Dataset and Resource Limits ---
constexpr int vocab_size = 27; // (unique characters) + BOS token
constexpr int MAX_NUM_DOCUMENTS = 40000;   // Maximum number of documents (names) to load
constexpr int MAX_DOC_LENGTH = 32;      // Maximum length of a single document (in characters)
constexpr int MAX_NUM_TOKENS = MAX_DOC_LENGTH + 4;  // Maximum tokens per document (doc + BOS tokens)
constexpr float val_split_ratio = 0.01f;    // Fraction of data for validation

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

// =============================================================================
// MODEL
// =============================================================================

struct Model
{
    // model hyperparameters
    static const int num_layers = 1;
    static const int emb_dim = 16;
    static const int block_size = 16;
    static const int num_heads = 4;
    static const int head_dim = emb_dim / num_heads;
    static const int FOUR_X_EMB_DIM = 4 * emb_dim;
    static constexpr float std_dev = 0.03f;

    EmbeddingLayer<float> wte;
    EmbeddingLayer<float> wpe;
    RMSNormLayer<float> embed_norm;
    RMSNormLayer<float> final_norm;
    LinearLayer<float> lm_head;
    std::vector<TransformerBlock<float> > transformer;

    void init(AutoGrad<float> & grad, int vocab_size)
    {
        wte.init(grad, vocab_size, emb_dim, std_dev, "wte");
        wpe.init(grad, block_size, emb_dim, std_dev, "wpe");
        lm_head.init(grad, emb_dim, vocab_size, false, std_dev, "lm_head");
        embed_norm.init(grad, emb_dim, "embed_norm");
        final_norm.init(grad, emb_dim, "final_norm");

        transformer.resize(num_layers);
        for (int li = 0; li < num_layers; li++)
            transformer[li].init(grad, emb_dim, num_heads, FOUR_X_EMB_DIM, std_dev);
    }

    // Whole-sequence forward.
    // tokens     : seq_len token ids (positions 0..seq_len-1)
    // returns    : logits tensor of shape {seq_len, vocab_size}
    TensorHandle forward(AutoGrad<float> & grad, std::span<const int> tokens)
    {
        int seq_len = (int)tokens.size();

        // --- STEP 1: token + position embedding, scatter into {seq_len, emb_dim} ---
        TensorHandle embedded = grad.tensor_leaf({ seq_len, emb_dim }, float(0));

        for (int t = 0; t < seq_len; t++)
        {
            TensorHandle tok_emb = wte.forward(grad, tokens[t]);   // {emb_dim}
            TensorHandle pos_emb = wpe.forward(grad, t);           // {emb_dim}
            TensorHandle summed  = grad.value_add(tok_emb, pos_emb); // {emb_dim}
            TensorHandle normed  = embed_norm.forward(grad, summed); // {emb_dim}
            embedded = grad.value_scatter_row(embedded, normed, t); // {seq_len, emb_dim}
        }

        // --- STEP 2: transformer layers ---
        TensorHandle hidden = embedded;
        for (int li = 0; li < num_layers; li++)
            hidden = transformer[li].forward(grad, hidden);

        // --- STEP 3: lm head hidden @ W -> {seq_len, vocab_size} ---
        // lm_head.parameters is {emb_dim, vocab_size}, hidden is {seq_len, emb_dim}
        TensorHandle final_hidden = final_norm.forward(grad, hidden);
        TensorHandle logits = lm_head.forward(grad, final_hidden);

        return logits;
    }
};

// =============================================================================
// VALIDATION
// =============================================================================

// Compute validation loss by running forward pass on all validation documents.
// Returns the average negative log likelihood over all validation positions.
float compute_validation_loss(
    ParameterCheckpoint<float> & checkpoint,
    AutoGrad<float> & grad, Model & model)
{
    float total_loss = 0.0f;
    int total_positions = 0;

    std::array<int, MAX_NUM_TOKENS> token_sequence;

    for (int doc_idx = 0; doc_idx < g_num_val; doc_idx++)
    {
        const char * current_doc = g_documents[g_val_indices[doc_idx]];
        int doc_length = (int)std::strlen(current_doc);

        grad.restore_parameter_values(checkpoint.values, checkpoint.grads);

        int num_tokens = 0;
        token_sequence[num_tokens++] = g_token_bos;
        for (int i = 0; i < doc_length; i++)
        {
            token_sequence[num_tokens++] = char_to_token_id(current_doc[i]);
        }
        token_sequence[num_tokens++] = g_token_bos;

        int seq_len = num_tokens - 1;          // we predict tokens 1..num_tokens-1
        if (seq_len > model.block_size) seq_len = model.block_size;

        // Forward: logits shape {seq_len, vocab_size}
        TensorHandle logits = model.forward(grad, std::span<const int>(token_sequence.data(), seq_len));

        // Cross-entropy loss (no backward needed for validation — just read the scalar)
        TensorHandle loss = grad.value_cross_entropy_loss(
            logits, std::span<const int>(token_sequence.data() + 1, seq_len));

        total_loss += grad.get(loss).tensor.values().data()[0] * seq_len;
        total_positions += seq_len;
    }

    return total_positions == 0 ? 0.0f : total_loss / (float)total_positions;
}

// =============================================================================
// MAIN
// =============================================================================

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

    AutoGrad<float> grad;
    // run at least one training iteration in debug mode to see if you're going to run out of memory
    grad.init(8192, 1e6);

    Model model;
    model.init(grad, vocab_size);
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
    const int    num_training_steps = 10000;
    const float base_learning_rate = 0.005f;
    const int    accumulation_steps = 1;

    // Buffers for tokens.
    std::array <TensorHandle, vocab_size> logit_nodes;

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

            // Select a document from the training set and tokenize it.
            // Tokens are surrounded by BOS tokens on both sides.
            const char * current_doc = g_documents[g_train_indices[micro_step % g_num_train]];
            int doc_length = (int)std::strlen(current_doc);

            std::array<int, MAX_NUM_TOKENS> token_sequence;
            int num_tokens = 0;
            for (int & t : token_sequence)
            {
                t = g_token_bos;
            }
            token_sequence[num_tokens++] = g_token_bos;  // Leading BOS token
            for (int i = 0; i < doc_length; i++)
            {
                token_sequence[num_tokens++] = char_to_token_id(current_doc[i]);
            }
            token_sequence[num_tokens++] = g_token_bos;  // Trailing BOS token (end of sequence)

            // Number of training positions (we predict the next token, so n = num_tokens - 1).
            int seq_len = num_tokens - 1;
            if (seq_len > model.block_size) seq_len = model.block_size;

            // SINGLE whole-sequence forward pass -> logits {seq_len, vocab_size}
            TensorHandle logits = model.forward(grad, std::span<const int>(token_sequence.data(), seq_len));

            // determine targets by fast forwarding the token sequence by 1
            std::span<const int> targets(token_sequence.data() + 1, seq_len);

            // Cross-entropy loss (mean over positions, with gradient support)
            TensorHandle loss_node = grad.value_cross_entropy_loss(logits, targets);

            // Scale loss for gradient accumulation: divide by accumulation_steps so that
            // the sum of gradients across micro-steps equals the mean.
            loss_node = grad.value_mul_const(loss_node, 1.0f / (float)accumulation_steps);

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
        // no KV cache, so we must re-run the whole prefix each step
        std::array<int, MAX_NUM_TOKENS> seq;
        int seq_len = 0;
        seq[seq_len++] = g_token_bos;

        char generated_text[model.block_size + 1];
        int text_length = 0;

        for (int pos = 0; pos < model.block_size; pos++)
        {
            // Restore parameters for each sample.
            grad.restore_parameter_values(checkpoint.values, checkpoint.grads);

            // forward over current prefix -> logits {seq_len, vocab_size}
            TensorHandle logits = model.forward(grad,
                std::span<const int>(seq.data(), seq_len));

            // Extract the LAST position's logits: select row (seq_len-1) -> {vocab_size}
            TensorHandle last_logits = grad.value_select_row(logits, seq_len - 1);

            // Scale by temperature
            TensorHandle scaled = grad.value_mul_const(last_logits, 1.0f / temperature);

            // Softmax -> probabilities {vocab_size}
            SoftmaxLayer<float> softmax;
            TensorHandle probs = softmax.forward(grad, scaled);

            float r = unif(rng), cum = 0.0f;
            int next_token = vocab_size - 1;
            const float * pvals = grad.get(probs).tensor.values().data();
            for (int i = 0; i < vocab_size; i++)
            {
                cum += pvals[i];
                if (r < cum) { next_token = i; break; }
            }

            if (next_token == g_token_bos) break;
            generated_text[text_length++] = (char)next_token + 'a';

            if (seq_len < MAX_NUM_TOKENS) seq[seq_len++] = next_token;
        }

        // Null-terminate and print the generated text.
        generated_text[text_length] = '\0';
        std::printf("sample %2d: %s\n", sample_idx + 1, generated_text);
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
