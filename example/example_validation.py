"""
PyTorch equivalent of example.cpp — a small character-level Transformer
trained to generate names (Karpathy's makemore dataset).

Dataset:
  curl -o names.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt

Usage:
  python train.py
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# HYPERPARAMETERS  (mirrors example.cpp)
# =============================================================================

VOCAB_SIZE        = 27        # 'a'-'z' (indices 0-25) + BOS token (index 26)
TOKEN_BOS         = 26

MAX_NUM_DOCUMENTS = 40_000
MAX_DOC_LENGTH    = 32
MAX_NUM_TOKENS    = MAX_DOC_LENGTH + 4
VAL_SPLIT_RATIO   = 0.01

NUM_LAYERS  = 1
EMB_DIM     = 16
BLOCK_SIZE  = 16
NUM_HEADS   = 4
HEAD_DIM    = EMB_DIM // NUM_HEADS
FFN_DIM     = 4 * EMB_DIM    # 64
STD_DEV     = 0.03            # init std for embeddings / lm_head / MLP weights

NUM_TRAINING_STEPS = 10_000
BASE_LR            = 0.005
LOG_INTERVAL       = 100
VAL_INTERVAL       = 1_000
TEMPERATURE        = 0.5


# =============================================================================
# DATA LOADING
# =============================================================================

def char_to_token_id(c: str) -> int:
    """Map a single character to its token id (0-25). Non-alpha maps to 0."""
    result = ord(c) - ord('a')
    if result < 0 or result > 25:
        result = 0
    return result


def load_documents(file_path: str) -> list[str]:
    try:
        docs = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.rstrip('\n\r \t')
                if not line:
                    continue
                if len(docs) >= MAX_NUM_DOCUMENTS:
                    break
                docs.append(line[:MAX_DOC_LENGTH - 1])
        return docs
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please download names.txt:")
        print("  curl -o names.txt https://raw.githubusercontent.com/"
              "karpathy/makemore/988aa59/names.txt")
        raise SystemExit(1)


def tokenize(doc: str) -> list[int]:
    """[BOS, char0, char1, ..., charN, BOS]"""
    tokens = [TOKEN_BOS]
    for c in doc:
        tokens.append(char_to_token_id(c))
    tokens.append(TOKEN_BOS)
    return tokens


# =============================================================================
# MODEL
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root-Mean-Square layer normalisation with learnable scale and bias.
    Matches RMSNormLayer in rmsnormlayer.h:  out = gamma * x / rms(x) + beta
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.gamma * x / (rms + 1e-8) + self.beta


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention matching attentionlayer.h:
      - Q/K/V projected with weight matrices (no bias)
      - Causal mask built per forward call
      - Heads concatenated manually, then projected through W_O
    Init std = 1 / sqrt(emb_dim)  (matches wq/wk/wv/wo in AttentionLayer::init)
    """

    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim   = emb_dim
        self.num_heads = num_heads
        self.head_dim  = emb_dim // num_heads

        std = 1.0 / math.sqrt(emb_dim)
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)
        for layer in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(layer.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, emb_dim)
        seq_len = x.shape[0]

        Q = self.wq(x)   # (seq_len, emb_dim)
        K = self.wk(x)
        V = self.wv(x)

        # Causal mask: upper-triangle filled with -1e9
        mask = torch.zeros(seq_len, seq_len, device=x.device, dtype=x.dtype)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                       diagonal=1),
            -1e9
        )

        scale = 1.0 / math.sqrt(self.head_dim)

        head_outs = []
        for h in range(self.num_heads):
            s = h * self.head_dim
            e = s + self.head_dim
            Q_h = Q[:, s:e]                          # (seq_len, head_dim)
            K_h = K[:, s:e]
            V_h = V[:, s:e]

            scores  = (Q_h @ K_h.T) * scale          # (seq_len, seq_len)
            weights = F.softmax(scores + mask, dim=-1)
            head_outs.append(weights @ V_h)           # (seq_len, head_dim)

        concatenated = torch.cat(head_outs, dim=-1)   # (seq_len, emb_dim)
        return self.wo(concatenated)


class MLP(nn.Module):
    """
    Two-layer feed-forward network with GeLU activation.
    Matches MLPLayer<DataType, GeLU<DataType>> in transformerblock.h.
    Init std = STD_DEV = 0.03  (matches mlp.init(..., std_dev) in TransformerBlock::init)
    """

    def __init__(self, emb_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(emb_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, emb_dim, bias=False)
        nn.init.normal_(self.w1.weight, std=STD_DEV)
        nn.init.normal_(self.w2.weight, std=STD_DEV)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (mirrors transformerblock.h)."""

    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(emb_dim)
        self.attn  = CausalSelfAttention(emb_dim, num_heads)
        self.norm2 = RMSNorm(emb_dim)
        self.mlp   = MLP(emb_dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NameTransformer(nn.Module):
    """
    Full model matching struct Model in example.cpp:
      wte + wpe -> embed_norm -> transformer blocks -> final_norm -> lm_head
    """

    def __init__(self):
        super().__init__()
        self.wte        = nn.Embedding(VOCAB_SIZE, EMB_DIM)
        self.wpe        = nn.Embedding(BLOCK_SIZE, EMB_DIM)
        self.embed_norm = RMSNorm(EMB_DIM)
        self.transformer = nn.ModuleList([
            TransformerBlock(EMB_DIM, NUM_HEADS, FFN_DIM)
            for _ in range(NUM_LAYERS)
        ])
        self.final_norm = RMSNorm(EMB_DIM)
        self.lm_head    = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)

        nn.init.normal_(self.wte.weight,     std=STD_DEV)
        nn.init.normal_(self.wpe.weight,     std=STD_DEV)
        nn.init.normal_(self.lm_head.weight, std=STD_DEV)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : LongTensor of shape (seq_len,)
        returns: logits of shape (seq_len, vocab_size)
        """
        seq_len   = tokens.shape[0]
        positions = torch.arange(seq_len, device=tokens.device)

        # Token + position embeddings, normalised per-token (matches the per-token
        # loop in Model::forward that calls embed_norm on each summed embedding).
        x = self.embed_norm(self.wte(tokens) + self.wpe(positions))

        for block in self.transformer:
            x = block(x)

        return self.lm_head(self.final_norm(x))


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def compute_val_loss(model: NameTransformer, val_docs: list[str]) -> float:
    """Average NLL over all positions in all validation documents."""
    model.eval()
    total_loss      = 0.0
    total_positions = 0

    with torch.no_grad():
        for doc in val_docs:
            tokens  = tokenize(doc)
            seq_len = min(len(tokens) - 1, BLOCK_SIZE)

            inputs  = torch.tensor(tokens[:seq_len],        dtype=torch.long)
            targets = torch.tensor(tokens[1:seq_len + 1],   dtype=torch.long)

            logits = model(inputs)           # (seq_len, vocab_size)
            probs  = F.softmax(logits, dim=-1)

            for pos in range(seq_len):
                p = probs[pos, targets[pos]].item()
                p = max(p, 1e-7)             # numerical safety (matches C++)
                total_loss -= math.log(p)
                total_positions += 1

    return total_loss / total_positions if total_positions > 0 else 0.0


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # Phase 1: data
    # -------------------------------------------------------------------------
    documents = load_documents("names.txt")
    print(f"num docs: {len(documents)}")
    print(f"vocab size: {VOCAB_SIZE}")

    random.seed(42)
    random.shuffle(documents)

    num_val   = max(1, int(len(documents) * VAL_SPLIT_RATIO))
    num_train = len(documents) - num_val
    train_docs = documents[:num_train]
    val_docs   = documents[num_train:]
    print(f"train docs: {num_train}, val docs: {num_val}")

    # -------------------------------------------------------------------------
    # Phase 2: model
    # -------------------------------------------------------------------------
    model     = NameTransformer()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num params: {num_params}")

    # -------------------------------------------------------------------------
    # Phase 3: initial validation loss
    # -------------------------------------------------------------------------
    init_val_loss = compute_val_loss(model, val_docs)
    print(f"init validation loss: {init_val_loss:.6f}")

    # -------------------------------------------------------------------------
    # Phase 4: training loop
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    val_loss  = 0.0

    train_start = time.time()

    for step in range(NUM_TRAINING_STEPS):
        model.train()

        doc     = train_docs[step % num_train]
        tokens  = tokenize(doc)
        seq_len = min(len(tokens) - 1, BLOCK_SIZE)

        inputs  = torch.tensor(tokens[:seq_len],      dtype=torch.long)
        targets = torch.tensor(tokens[1:seq_len + 1], dtype=torch.long)

        logits = model(inputs)    # (seq_len, vocab_size)

        # Matches C++ loss: sum of -log(p_target) / seq_len
        # F.cross_entropy is numerically equivalent (log-softmax internally)
        loss = F.cross_entropy(logits, targets)

        # Linear LR decay: lr goes from base_lr down to 0
        current_lr = BASE_LR * (1.0 - step / NUM_TRAINING_STEPS)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        do_print = ((step + 1) % LOG_INTERVAL == 0 or step == NUM_TRAINING_STEPS - 1)
        do_val   = ((step + 1) % VAL_INTERVAL == 0 or step == NUM_TRAINING_STEPS - 1)

        if do_val:
            val_loss = compute_val_loss(model, val_docs)

        if do_print:
            train_loss_val = loss.item()
            if do_val:
                print(f"step {step + 1:4d} / {NUM_TRAINING_STEPS} | "
                      f"train_loss {train_loss_val:.4f} | val_loss {val_loss:.4f}")
            else:
                print(f"step {step + 1:4d} / {NUM_TRAINING_STEPS} | "
                      f"train_loss {train_loss_val:.4f}", end='\r')

    print()
    elapsed = time.time() - train_start
    minutes = int(elapsed / 60)
    seconds = elapsed - minutes * 60
    print(f"training time: {minutes:02d}:{seconds:05.2f}")

    # -------------------------------------------------------------------------
    # Phase 5: inference / text generation
    # -------------------------------------------------------------------------
    print("--- inference (new, hallucinated names) ---")
    model.eval()
    rng = random.Random(42)

    for sample_idx in range(20):
        seq       = [TOKEN_BOS]
        generated = []

        for _ in range(BLOCK_SIZE):
            inputs = torch.tensor(seq, dtype=torch.long)
            with torch.no_grad():
                logits = model(inputs)

            last_logits = logits[-1] / TEMPERATURE
            probs = F.softmax(last_logits, dim=-1).tolist()

            # Ancestral sampling (matches C++ loop)
            r          = rng.random()
            cum        = 0.0
            next_token = VOCAB_SIZE - 1
            for i, p in enumerate(probs):
                cum += p
                if r < cum:
                    next_token = i
                    break

            if next_token == TOKEN_BOS:
                break

            generated.append(chr(next_token + ord('a')))
            if len(seq) < MAX_NUM_TOKENS:
                seq.append(next_token)

        print(f"sample {sample_idx + 1:2d}: {''.join(generated)}")

    # -------------------------------------------------------------------------
    # Phase 6: save weights
    # -------------------------------------------------------------------------
    torch.save(model.state_dict(), "checkpoint.pt")
    print(f"wrote checkpoint to disk ({num_params} weights)")


if __name__ == "__main__":
    main()
