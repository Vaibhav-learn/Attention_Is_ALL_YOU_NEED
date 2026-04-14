class Config:
    # Model params
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1

    # Vocab
    src_vocab_size = 1000
    tgt_vocab_size = 1000

    # Training
    batch_size = 32
    lr = 1e-4
    epochs = 20

    # Special tokens
    max_len = 0
    pad_idx = 0

config = Config()