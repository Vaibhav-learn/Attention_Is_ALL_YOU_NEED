class Config:
    d_model = 256
    num_heads = 4
    num_layers = 3
    d_ff = 1024
    dropout = 0.1

    src_vocab_size = 15000
    tgt_vocab_size = 15000

    batch_size = 8  
    lr = 1e-4
    epochs = 50

    max_len = 512
    pad_idx = 0

config = Config()