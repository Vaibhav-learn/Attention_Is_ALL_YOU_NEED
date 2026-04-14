# Attention Is All You Need: Transformer Implementation

A principled implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), focusing on clean, modular code with proper masking, normalization, and attention mechanisms for sequence-to-sequence neural machine translation (English ↔ Hindi).

## Architecture Overview

### Core Innovations

The Transformer replaces recurrent connections with scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q, K, V$ are Query, Key, Value projections (dimension $d_k = d_{\text{model}}/h$)
- Scaling by $\sqrt{d_k}$ prevents attention weights from growing too large
- Multi-head attention runs $h$ attention operations in parallel: $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

### Computational Complexity

- **Self-Attention**: $O(n^2 \cdot d)$ where $n$ is sequence length, $d$ is model dimension
- **Feed-Forward**: $O(n \cdot d^2)$ (applied position-wise)
- **Total per layer**: Dominated by attention for $n < d$ (typical in NMT with $d=512$)
- **Parallelization**: Enable across all positions simultaneously (unlike RNNs)

## Project Structure

```
transformer/
├── config.py                    # Hyperparameter configuration
├── train.py                     # Training loop, data pipeline, vocab building
├── requirements.txt             # Dependencies
├── models/
│   ├── encoder_decoder.py       # EncoderLayer, DecoderLayer with residual connections
│   ├── attention.py             # MultiHeadAttention (scaled dot-product)
│   └── transformer.py           # Full Transformer (Encoder-Decoder stack)
└── utils/
    ├── mask.py                  # create_padding_mask, create_look_ahead_mask
    └── positional_encoding.py   # Sine/cosine positional encodings
```

## Installation & Setup

```bash
cd transformer
pip install -r requirements.txt
```

**Dependencies**:
- `torch`: Autograd, nn.Module, distributed training support
- `datasets`: HuggingFace for IITB corpus (automatic download/caching)
- `numpy`: Numerical operations
- `matplotlib`: Loss visualization
- `sentencepiece`: Optional for subword tokenization (not currently used)

## Hyperparameter Configuration

[config.py](transformer/config.py) defines:

```python
class Config:
    # Embedding & Model Dim
    d_model = 512           # Model dimension (query/key/value projection size)
    
    # Attention Parameters
    num_heads = 8           # Number of parallel attention heads
    d_k = d_model // num_heads = 64  # Per-head key/query dimension
    d_v = d_model // num_heads = 64  # Per-head value dimension
    
    # Architecture Depth
    num_layers = 6          # Number of identical encoder/decoder layers (N=6 from paper)
    
    # Feed-Forward Network
    d_ff = 2048             # Hidden dimension in FFN (4x expansion ratio)
    
    # Regularization
    dropout = 0.1           # Applied to: attention weights, FFN hidden, embeddings
    
    # Vocabulary
    src_vocab_size = 5000   # Source language vocabulary
    tgt_vocab_size = 5000   # Target language vocabulary
    max_len = 512           # Maximum sequence length (positional encoding limit)
    
    # Training
    batch_size = 32
    lr = 1e-4               # Recommend: NoamOpt with lr=1.0 instead
    epochs = 20
    
    # Special Tokens
    pad_idx = 0             # Masked in attention computations
```

**Note**: The paper uses learning rate schedule: $\text{lr}(step) = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$

## Implementation Details

### 1. Multi-Head Attention ([models/attention.py](transformer/models/attention.py))

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Single linear projection for all heads (more efficient than h separate projections)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Project and reshape: (batch_size, seq_len, d_model) 
        # → (batch_size, seq_len, num_heads, d_k)
        # → (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Output: (batch_size, num_heads, seq_len, d_k)
        output = attn_weights @ V
        
        # Concatenate heads: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        
        return self.W_o(output)
```

**Key design choices**:
- Single projection matrix (d_model → d_model) instead of h separate ones: improved efficiency
- Masked fill with `-inf` before softmax (not after): numerical stability
- Dropout applied to attention weights (not stored)
- Contiguous view for efficient memory access

### 2. Encoder/Decoder Layers ([models/encoder_decoder.py](transformer/models/encoder_decoder.py))

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention block with Post-LN residual connection
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.norm1(x)
        
        # Feed-forward block
        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Masked self-attention (prevents attending to future tokens)
        x = x + self.dropout(self.self_attn(x, x, x, tgt_mask))
        x = self.norm1(x)
        
        # Cross-attention to encoder (Q from decoder, K,V from encoder)
        x = x + self.dropout(self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.norm2(x)
        
        # Feed-forward
        x = x + self.dropout(self.ff(x))
        x = self.norm3(x)
        return x
```

**Normalization strategy**: Post-LN residual (standard in paper):
- $\text{LayerNorm}(\text{input} + \text{sublayer}(\text{input}))$
- Pre-LN alternative: $\text{LayerNorm}(\text{input}) \rightarrow$ sublayer $\rightarrow +$ input (sometimes more stable)

### 3. Masking ([utils/mask.py](transformer/utils/mask.py))

**Padding Mask** (applied to encoder & cross-attention):
```python
def create_padding_mask(seq, pad_idx=0):
    # Shape: (batch_size, 1, 1, seq_len)
    # mask[b, 1, 1, i] = 0 if seq[b, i] == pad_idx else 1
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```
- Prevents attention to padding tokens
- Applied: `scores = scores.masked_fill(mask == 0, float('-inf'))`

**Look-Ahead Mask** (applied to decoder self-attention):
```python
def create_look_ahead_mask(seq_len):
    # Shape: (1, 1, seq_len, seq_len)
    # Lower triangular matrix: allows attention only to current & past positions
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)
```
- Ensures causal dependency (position $i$ can only attend to $j \leq i$)
- Combined with padding mask via element-wise AND

### 4. Positional Encoding ([utils/positional_encoding.py](transformer/utils/positional_encoding.py))

```python
def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    
    # Sinusoidal encoding with varying frequencies
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(log(10000) / d_model))
    
    pe[:, 0::2] = sin(position * div_term)      # Even dimensions
    pe[:, 1::2] = cos(position * div_term)      # Odd dimensions
    
    return pe
```

**Properties**:
- Unique for each position (encodes absolute position)
- Learnable wavelengths via exponential spacing
- Continuous function (interpolates for longer sequences)
- $\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d_{\text{model}}})$
- $\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d_{\text{model}}})$

### 5. Training Pipeline ([train.py](transformer/train.py))

```python
# Dataset loading
dataset = load_dataset("cfilt/iitb-english-hindi")
train_data = dataset["train"].select(range(2000))

# Dynamic vocabulary building
src_vocab = build_vocab(src_texts, max_size=5000)
tgt_vocab = build_vocab(tgt_texts, max_size=5000)

# Encoding: word → token IDs with BOS/EOS
def encode(sentence, vocab):
    tokens = [vocab.get(w, 0) for w in sentence.split()]
    tokens = tokens[:config.max_len - 2]
    tokens = [1] + tokens + [2]  # [BOS, ..., EOS]
    return torch.tensor(tokens, dtype=torch.long)

# Dataset & DataLoader for batching
class TranslationDataset(Dataset):
    def __getitem__(self, idx):
        src = encode(src_texts[idx], src_vocab)
        tgt = encode(tgt_texts[idx], tgt_vocab)
        return src, tgt

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(config.epochs):
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src_mask = create_padding_mask(src, pad_idx)
        tgt_mask = create_padding_mask(tgt, pad_idx) & create_look_ahead_mask(tgt.size(1))
        
        # Teacher forcing: decoder input is target shifted by 1
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]
        
        output = model(src, decoder_input, src_mask, tgt_mask)
        
        loss = criterion(output.view(-1, tgt_vocab_size), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Training specifics**:
- Teacher forcing: decoder uses ground truth at training time
- Shift targets by 1 position (predict next token)
- Ignore padding tokens in loss (`ignore_index`)
- Loss computed flattened: $(batch \times seq\_len, vocab\_size)$

## Running the Model

```bash
cd transformer
python train.py
```

Outputs:
- Loss per epoch logged to console
- Final model saved (optional)
- Matplotlib loss curves (optional)

**Dataset**: [IITB English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
- Automatically downloaded via HuggingFace `datasets` library
- Default: 2000 samples (see [train.py](transformer/train.py) line 15)

## Results & Performance Metrics

### Original Paper vs. Current Implementation

#### Vaswani et al. (2017) Paper Results
**Benchmark**: WMT 2014 English-German & English-French translation

| Metric | EN-DE | EN-FR | Notes |
|--------|-------|-------|-------|
| BLEU Score | 28.4 | 41.0 | State-of-the-art at time of publication |
| Training Time | 3.5 days | 4.0 days | 8× P100 GPUs |
| Model Size | 65M parameters | 65M parameters | Base model |
| Precision (Token) | ~97% | ~97% | On test sets |
| Inference Speed | 4.9 sec/sentence | 5.1 sec/sentence | Beam size=4 |

---

### Current Implementation Results

#### Training on IITB English-Hindi Corpus (2000 samples)

| Aspect | Original Paper | Current Run | Difference | Status |
|--------|---|---|---|---|
| **Task** | EN→DE, EN→FR (Large corpora: ~4.5M pairs) | EN→HI (Small corpus: 2000 samples) | Smaller dataset, different language pair | 🔄 In Progress |
| **Model Precision** | 97% (token-level) | [TBD - Add after training] | Expected: Lower (smaller data) | ⏳ Pending |
| **BLEU Score** | 28.4 (WMT EN-DE) | [TBD - Add after training] | Expected: Lower (smaller data) | ⏳ Pending |
| **Training Loss** | Final: ~4.8 | [TBD - Add after training] | TBD | ⏳ Pending |
| **Dataset Size** | 4.5M sentence pairs | 2,000 sentence pairs | 2,250× smaller | ✓ Baseline |
| **Training Time** | 3.5 days (8× P100) | [TBD] | TBD | ⏳ Monitoring |
| **Vocab Size** | ~32K tokens | 5,000 tokens | Limited vocab | ✓ Baseline |

**Note**: Direct comparison not applicable due to different datasets, languages, and model sizes. Current results serve as baseline for optimization.

---

### Precision Improvements: Paper vs. Current

```
Token-Level Accuracy Comparison
═════════════════════════════════════════════════════════════

Original Vaswani et al. (2017) - WMT EN-DE
├─ Model Precision:     ████████████████████░ 97.2%
├─ BLEU Score:          ██████████░░░░░░░░░░ 28.4
└─ Notes: Large corpus, multiple languages, 8× P100 GPUs

Current Implementation - IITB EN-HI (2000 samples)
├─ Model Precision:     [████░░░░░░░░░░░░░░░ TBD%] ◄─ Update after training
├─ BLEU Score:          [██░░░░░░░░░░░░░░░░░ TBD] ◄─ Update after training
└─ Notes: Small corpus, single language pair, local GPU

Improvement Opportunities:
• Increase dataset size to 50K+ samples
• Implement learning rate scheduling (warmup + decay)
• Add label smoothing (ε = 0.1)
• Use shared embeddings (tie input/output weights)
• Implement gradient clipping
• Extend to 10K+ vocabulary
```

---

### Training Progress: Current Run

**Model Configuration:**
- d_model: 512
- Heads: 8
- Layers: 6
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 20

**Current Metrics:**

| Epoch | Training Loss | Validation Loss | Precision | BLEU | Status |
|-------|---|---|---|---|---|
| 0 (Init) | - | - | - | - | Starting |
| 1 | [TBD] | [TBD] | [TBD]% | [TBD] | ◄─ Update here |
| 5 | [TBD] | [TBD] | [TBD]% | [TBD] | ◄─ Update here |
| 10 | [TBD] | [TBD] | [TBD]% | [TBD] | ◄─ Update here |
| 15 | [TBD] | [TBD] | [TBD]% | [TBD] | ◄─ Update here |
| 20 | [TBD] | [TBD] | [TBD]% | [TBD] | ◄─ **FINAL RESULT** |

---

### Experimental Runs Comparison

#### Baseline (Paper Reproduction)
```
Configuration: d_model=512, heads=8, layers=6
Dataset: IITB EN-HI (2000 samples)
Result:
  ✓ Precision: [ADD YOUR RESULT]%
  ✓ Final Loss: [ADD YOUR RESULT]
  ✓ Best BLEU: [ADD YOUR RESULT]
  ✓ Training Time: [ADD YOUR RESULT]
  ✓ Date Completed: [ADD DATE]
```

#### Run 2: [Name Your Improvement]
```
Configuration: [TBD - e.g., Learning rate scheduling]
Dataset: [TBD]
Result:
  ✓ Precision: [ADD YOUR RESULT]%
  ✓ Final Loss: [ADD YOUR RESULT]
  ✓ Best BLEU: [ADD YOUR RESULT]
  ✓ Improvement: [+TBD%] over baseline
```

#### Run 3: [Name Your Next Improvement]
```
Configuration: [TBD - e.g., Larger vocab, label smoothing]
Dataset: [TBD]
Result:
  ✓ Precision: [ADD YOUR RESULT]%
  ✓ Final Loss: [ADD YOUR RESULT]
  ✓ Best BLEU: [ADD YOUR RESULT]
  ✓ Improvement: [+TBD%] over baseline
```

---

### Metric Definitions & Calculations

**Precision (Token-Level)**
$$\text{Precision} = \frac{\text{Correct Tokens}}{\text{Total Non-Padding Tokens}} \times 100\%$$
- Measures exact token prediction accuracy
- Higher is better (expected: 40-70% on small datasets)

**BLEU Score (Bilingual Evaluation Understudy)**
$$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$
Where:
- $p_n$ = n-gram precision
- $BP$ = brevity penalty (for short translations)
- Weights: $w_1=0.25, w_2=0.25, w_3=0.25, w_4=0.25$
- Range: 0-100 (higher is better)
- Typical: 28.4 (WMT), 5-15 (small datasets)

**Training Loss**
$$L = -\sum_{j=1}^{n} \log P(y_j | y_{<j}, x)$$
- Cross-entropy on vocabulary distribution
- Lower is better (typical: 5-8 initial, <2 final)

**Validation Loss**
$$L_{val} = \frac{1}{|V|} \sum_{x,y \in V} -\log P(y|x)$$
- Generalization indicator
- Overfitting if $L_{val} \gg L_{train}$

---

### Performance Analysis

**Expected vs. Actual Results**

| Factor | Expected | Actual | Analysis |
|--------|----------|--------|----------|
| Dataset Size | 2K samples | [Update] | Small; suitable for baseline testing |
| Precision | 30-60% | [Update] | Language pair difficulty varies |
| BLEU | 5-15 | [Update] | Compare against paper's 28.4 |
| Training Convergence | 5-10 epochs | [Update] | Watch for early stopping |
| Overfitting Risk | High (small data) | [Update] | Monitor val loss divergence |

**Key Observations to Track:**
- [ ] Does precision plateau or keep improving?
- [ ] Is validation loss diverging from training loss?
- [ ] How many epochs until convergence?
- [ ] Any anomalies in loss curves?

## Performance Considerations

| Aspect | Value | Impact |
|--------|-------|--------|
| **Attention complexity** | $O(n^2 \cdot d)$ | Quadratic in sequence length; limits max_len |
| **FFN complexity** | $O(n \cdot d^2)$ | Typically dominated by attention for NMT |
| **Memory usage** | ~$O(batch \times seq\_len \times d)$ per layer | 6 layers × 8 heads → careful with large batches |
| **Parallelization** | Full across positions | All 512 positions computed simultaneously (unlike RNNs) |

**Optimization variants**:
- Local attention: $O(n \cdot w \cdot d)$ (window size $w$)
- Sparse patterns: Strided attention, BigBird
- Linear attention approximations: Performer, kernel attention

## Known Limitations & Extensions

1. **No learned embeddings**: Currently uses one-hot encoding via vocab indices
2. **Basic decoding**: Greedy selection (not beam search)
3. **No label smoothing**: Can hurt generalization
4. **Fixed length sequences**: Padding to max_len
5. **No layer pre-training**: Cold start training

**Recommended additions**:
- Shared input/output embeddings
- Learned vs. fixed positional encoding comparison
- Label smoothing with $\epsilon = 0.1$
- Warmup + decay learning rate schedule
- Gradient clipping
- Model ensembling

## References

- Vaswani, A., et al. (2017). ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). 
  *Advances in Neural Information Processing Systems (NeurIPS)*.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Alammar, J.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Weng, L.

## Citation

If you use this implementation, please cite:
```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Nik and Uszkoreit, Jakob and 
          Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```