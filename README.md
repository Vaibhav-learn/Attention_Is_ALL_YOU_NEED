# Attention Is All You Need: Transformer Implementation

A clean, educational implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). This project implements a sequence-to-sequence model for neural machine translation (English ↔ Hindi).

## Overview

The Transformer architecture revolutionized deep learning by introducing the self-attention mechanism, eliminating the need for recurrent layers and enabling better parallelization. This implementation demonstrates the core components of the architecture including:

- **Multi-Head Self-Attention**: Allows the model to attend to different representation subspaces
- **Positional Encoding**: Provides position information to the model
- **Feed-Forward Networks**: Dense layers applied to each position separately and identically
- **Layer Normalization & Residual Connections**: Stabilizes training and enables deep networks
- **Encoder-Decoder Stack**: Complete sequence-to-sequence architecture

## Project Structure

```
transformer/
├── config.py                 # Configuration settings for the model
├── train.py                  # Training script and dataset handling
├── requirements.txt          # Python dependencies
├── models/
│   ├── encoder_decoder.py    # Encoder and decoder layer implementations
│   └── attention.py          # Multi-head attention mechanism
└── utils/
    ├── mask.py              # Masking utilities (padding, look-ahead masks)
    └── positional_encoding.py  # Positional encoding implementation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Attention_Is_ALL_YOU_NEED
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
cd transformer
pip install -r requirements.txt
```

## Dependencies

- **torch**: Deep learning framework
- **datasets**: Hugging Face datasets library (includes IITB English-Hindi corpus)
- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **sentencepiece**: Subword tokenization

## Configuration

Model hyperparameters can be adjusted in [transformer/config.py](transformer/config.py):

```python
class Config:
    # Model Architecture
    d_model = 512              # Embedding dimension
    num_heads = 8              # Number of attention heads
    num_layers = 6             # Number of encoder/decoder layers
    d_ff = 2048                # Feed-forward hidden dimension
    dropout = 0.1              # Dropout rate
    
    # Vocabulary
    src_vocab_size = 5000      # Source language vocabulary size
    tgt_vocab_size = 5000      # Target language vocabulary size
    max_len = 512              # Maximum sequence length
    
    # Training
    batch_size = 32            # Batch size for training
    lr = 1e-4                  # Learning rate
    epochs = 20                # Number of training epochs
    
    # Special Tokens
    pad_idx = 0                # Padding token index
```

## Usage

### Training

To train the model on the English-Hindi translation task:

```bash
cd transformer
python train.py
```

The training script will:
1. Load the IITB English-Hindi dataset (2000 samples by default)
2. Build vocabulary from source and target languages
3. Create the Transformer model
4. Train for the specified number of epochs
5. Save training metrics and visualizations

### Dataset

By default, the project uses the [IITB English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi). You can modify the dataset size in [train.py](transformer/train.py):

```python
train_data = dataset["train"].select(range(2000))  # Adjust range as needed
```

## Architecture Details

### Encoder
- Stack of 6 identical layers
- Each layer contains:
  - Multi-head self-attention (8 heads)
  - Feed-forward network (2048 hidden units)
  - Residual connections and layer normalization

### Decoder
- Stack of 6 identical layers
- Each layer contains:
  - Masked multi-head self-attention (prevents attending to future tokens)
  - Cross-attention to encoder outputs
  - Feed-forward network
  - Residual connections and layer normalization

### Attention Mechanism
- Multi-head attention with 8 parallel representation subspaces
- Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / √d_k)V
- Enables the model to jointly attend to information from different representation subspaces

## Key Files

- **config.py**: Centralized configuration for reproducibility
- **train.py**: Complete training pipeline including data loading, preprocessing, and model training
- **models/encoder_decoder.py**: Encoder and decoder layer implementations with residual connections
- **models/attention.py**: Multi-head attention implementation
- **utils/mask.py**: Padding masks for variable-length sequences and look-ahead masks for autoregressive decoding
- **utils/positional_encoding.py**: Sine/cosine positional encodings

## Features Implemented

- ✅ Multi-head self-attention mechanism
- ✅ Positional encoding (sine/cosine)
- ✅ Feed-forward networks with residual connections
- ✅ Layer normalization
- ✅ Encoder-decoder architecture
- ✅ Padding and look-ahead masking
- ✅ Support for variable-length sequences
- ✅ Vocabulary building from raw text
- ✅ Training loop with loss tracking

## Advanced Features

- **Masking**: Properly handles padding tokens and prevents attention to future positions during decoding
- **Batch Processing**: Efficient batch training with DataLoader
- **Vocabulary Building**: Dynamic vocabulary construction from training data
- **Tokenization**: Special tokens for padding, start-of-sequence, and end-of-sequence

## Training Tips

1. **Larger Dataset**: Start with a larger portion of the IITB corpus for better results
2. **Learning Rate Scheduling**: Consider implementing a learning rate schedule (e.g., warmup + decay)
3. **Evaluation**: Add validation set evaluation to monitor overfitting
4. **Checkpointing**: Save model weights at regular intervals
5. **Beam Search**: Implement beam search for decoding instead of greedy decoding

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
  ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). *NeurIPS 2017*.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [IITB English-Hindi Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi)

## Future Enhancements

- [ ] Beam search decoding
- [ ] BLEU score evaluation
- [ ] Learning rate scheduling (warmup)
- [ ] Model checkpointing and resuming
- [ ] Pre/post normalization variants
- [ ] Label smoothing
- [ ] Mixed precision training

## License

This project is provided for educational purposes. Please cite the original paper if you use this implementation in your research.

## Author Notes

This is an implementation of the seminal Transformer architecture, designed to help understand the core concepts behind modern deep learning models like BERT, GPT, and others. The code prioritizes clarity and understanding over production optimization.