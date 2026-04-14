import torch.nn as nn
from models.embeddings import TokenEmbedding, PositionalEncoding
from models.encoder_decoder import EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    def __init__(self, vocab, d_model, layers, heads, d_ff, dropout):
        super().__init__()
        self.embed = TokenEmbedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(layers)
        ])

    def forward(self, x, mask=None):
        x = self.pos(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, layers, heads, d_ff, dropout):
        super().__init__()
        self.embed = TokenEmbedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(layers)
        ])

        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.pos(self.embed(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, layers, heads, d_ff, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, layers, heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, layers, heads, d_ff, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.encoder(src, src_mask)
        return self.decoder(tgt, enc, src_mask, tgt_mask)