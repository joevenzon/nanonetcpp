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
    SimpleRMSNormLayer<float> embed_norm;
    LinearLayer<float> lm_head;
    std::vector<TransformerBlock<float> > transformer;

    void init(AutoGrad<float> & grad, int vocab_size)
    {
        wte.init(grad, vocab_size, emb_dim, std_dev, "wte");
        wpe.init(grad, block_size, emb_dim, std_dev, "wtp");
        lm_head.init(grad, vocab_size, emb_dim, std_dev, "lm_head");

        transformer.resize(num_layers);
        for (int li = 0; li < num_layers; li++)
            transformer[li].init(grad, emb_dim, num_heads, FOUR_X_EMB_DIM, std_dev);
    }

    // Whole-sequence forward.
    // tokens     : seq_len token ids (positions 0..seq_len-1)
    // out_logits : seq_len spans, each of size vocab_size, written here
    void forward(
        AutoGrad<float> & grad,
        std::span<const int> tokens,
        std::span<std::span<NodeHandle> > out_logits)
    {
        int seq_len = (int)tokens.size();
        assert((int)out_logits.size() == seq_len);

        // --- STEP 1: token + position embedding, then RMSNorm (per position) ---
        std::vector<std::array<NodeHandle, emb_dim> > normed(seq_len);
        std::vector<std::span<NodeHandle> >           normed_spans(seq_len);

        for (int t = 0; t < seq_len; t++)
        {
            std::array<NodeHandle, emb_dim> token_embedded, pos_embedded, embedded;
            wte.forward(grad, tokens[t], token_embedded);
            wpe.forward(grad, t, pos_embedded);

            for (int j = 0; j < emb_dim; j++)
                embedded[j] = grad.value_add(token_embedded[j], pos_embedded[j]);

            embed_norm.forward(grad, embedded, normed[t]);
            normed_spans[t] = std::span<NodeHandle>(normed[t].data(), emb_dim);
        }

        // --- STEP 2: transformer layers (whole sequence) ---
        std::vector<std::array<NodeHandle, emb_dim>> hidden(seq_len);
        std::vector<std::span<NodeHandle>>           hidden_spans(seq_len);
        for (int t = 0; t < seq_len; t++)
            hidden_spans[t] = std::span<NodeHandle>(hidden[t].data(), emb_dim);

        // ping-pong buffers between layers
        std::span<const std::span<NodeHandle>> layer_in(normed_spans.data(), seq_len);
        std::span<std::span<NodeHandle>>       layer_out(hidden_spans.data(), seq_len);

        for (int li = 0; li < num_layers; li++)
        {
            transformer[li].forward(grad, layer_in, layer_out);
            // after first layer, feed hidden back in as input
            layer_in = std::span<const std::span<NodeHandle>>(hidden_spans.data(), seq_len);
        }

        // --- STEP 3: lm head per position ---
        for (int t = 0; t < seq_len; t++)
            lm_head.forward(grad, hidden_spans[t], out_logits[t]);
    }
};

// Compute validation loss by running forward pass on all validation documents.
// Returns the average negative log likelihood over all validation positions.
float compute_validation_loss(
    ParameterCheckpoint<float> & checkpoint,
    AutoGrad<float> & grad, Model & model)
{
    SoftmaxLayer<float> softmax_layer;
    float total_loss = 0.0f;
    int total_positions = 0;

    std::array<int, MAX_NUM_TOKENS> token_sequence;

    for (int doc_idx = 0; doc_idx < g_num_val; doc_idx++)
    {
        const char * current_doc = g_documents[g_val_indices[doc_idx]];
        int doc_length = (int)std::strlen(current_doc);

        grad.restore_parameter_values(checkpoint.values);

        int num_tokens = 0;
        token_sequence[num_tokens++] = g_token_bos;
        for (int i = 0; i < doc_length; i++)
            token_sequence[num_tokens++] = char_to_token_id(current_doc[i]);
        token_sequence[num_tokens++] = g_token_bos;

        int seq_len = num_tokens - 1;          // we predict tokens 1..num_tokens-1
        if (seq_len > model.block_size) seq_len = model.block_size;

        // logits buffer: seq_len x vocab_size
        std::vector<std::array<NodeHandle, vocab_size> > logits(seq_len);
        std::vector<std::span<NodeHandle> >              logit_spans(seq_len);
        for (int t = 0; t < seq_len; t++)
            logit_spans[t] = std::span<NodeHandle>(logits[t].data(), vocab_size);

        model.forward(grad,
            std::span<const int>(token_sequence.data(), seq_len),
            std::span<std::span<NodeHandle>>(logit_spans.data(), seq_len));

        std::array<NodeHandle, vocab_size> prob_nodes;
        for (int pos = 0; pos < seq_len; pos++)
        {
            int target_token = token_sequence[pos + 1];
            softmax_layer.forward(grad, logit_spans[pos], prob_nodes);

            float prob = grad.get(prob_nodes[target_token]).data;
            if (prob < 1e-7f) prob = 1e-7f;
            total_loss -= std::log(prob);
            total_positions++;
        }
    }

    return total_positions == 0 ? 0.0f : total_loss / (float)total_positions;
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

    AutoGrad<float> grad;
    // run at least one training iteration in debug mode to see if you're going to run out of memory
    grad.init(1e6);

    Model model;
    model.init(grad, vocab_size);

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

    // Buffers for tokens, logits, and probabilities.
    std::array <NodeHandle, MAX_NUM_TOKENS> token_sequence;
    std::array <NodeHandle, vocab_size> logit_nodes;
    std::array <NodeHandle, vocab_size> prob_nodes;

    for (int step = 0; step < num_training_steps; step++)
    {
        // Restore parameter values into the pool (pool is rebuilt each step).
        grad.restore_parameter_values(checkpoint.values);

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
        int seq_len = num_tokens - 1;
        if (seq_len > model.block_size) seq_len = model.block_size;

        // logits: seq_len x vocab_size
        std::vector<std::array<NodeHandle, vocab_size>> logits(seq_len);
        std::vector<std::span<NodeHandle>>              logit_spans(seq_len);
        for (int t = 0; t < seq_len; t++)
            logit_spans[t] = std::span<NodeHandle>(logits[t].data(), vocab_size);

        // SINGLE whole-sequence forward pass
        model.forward(grad,
            std::span<const int>(token_sequence.data(), seq_len),
            std::span<std::span<NodeHandle>>(logit_spans.data(), seq_len));

        // Accumulate loss over all positions
        SoftmaxLayer<float> softmax_layer;
        std::array<NodeHandle, vocab_size> prob_nodes;
        int loss_accumulator = -1;

        for (int pos = 0; pos < seq_len; pos++)
        {
            int target_token = token_sequence[pos + 1];
            softmax_layer.forward(grad, logit_spans[pos], prob_nodes);

            int neg_log_prob = grad.value_mul_const(
                grad.value_log(prob_nodes[target_token]), -1);

            loss_accumulator = (loss_accumulator < 0)
                ? neg_log_prob
                : grad.value_add(loss_accumulator, neg_log_prob);
        }

        // Average the loss over all positions in the sequence.
        int loss_node = grad.value_mul_const(loss_accumulator, 1.0 / seq_len);

        // Backward pass: compute gradients for all parameters.
        grad.backward(loss_node);

        // capture current parameter values in the checkpoint
        checkpoint.update(grad);

        // -------------------------------------------------------------------
        // PARAMETER UPDATE (ADAM OPTIMIZER)
        // -------------------------------------------------------------------
        // Apply linear learning rate decay: lr decreases from base_lr to 0.
        float current_lr = base_learning_rate * (1.0f - (float)step / (float)num_training_steps);
        optimizer.lr = current_lr;
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
                std::printf("step %4d / %4d | train_loss %.4f\r",
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
        // no KV cache, so we must re-run the whole prefix each step
        std::array<int, MAX_NUM_TOKENS> seq;
        int seq_len = 0;
        seq[seq_len++] = g_token_bos;

        char generated_text[model.block_size + 1];
        int text_length = 0;

        for (int pos = 0; pos < model.block_size; pos++)
        {
            // Restore parameters for each sample.
            grad.restore_parameter_values(checkpoint.values);

            // forward over current prefix
            std::vector<std::array<NodeHandle, vocab_size> > logits(seq_len);
            std::vector<std::span<NodeHandle> >              logit_spans(seq_len);
            for (int t = 0; t < seq_len; t++)
                logit_spans[t] = std::span<NodeHandle>(logits[t].data(), vocab_size);

            model.forward(grad,
                std::span<const int>(seq.data(), seq_len),
                std::span<std::span<NodeHandle>>(logit_spans.data(), seq_len));

            // use the LAST position's logits to predict the next token
            std::span<NodeHandle> last = logit_spans[seq_len - 1];

            std::array<NodeHandle, vocab_size> scaled_logits;
            for (int i = 0; i < vocab_size; i++)
                scaled_logits[i] = grad.value_mul_const(last[i], 1.0f / temperature);

            SoftmaxLayer<float> softmax;
            std::array<NodeHandle, vocab_size> prob_nodes;
            softmax.forward(grad, scaled_logits, prob_nodes);

            float r = unif(rng), cum = 0.0f;
            int next_token = vocab_size - 1;
            for (int i = 0; i < vocab_size; i++)
            {
                cum += grad.get(prob_nodes[i]).data;
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

    std::printf("pool high water mark: %d\n", (int)grad.high_water_mark());

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