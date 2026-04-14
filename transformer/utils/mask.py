import torch

def create_padding_mask(seq, pad_idx):
    """Creates padding mask for attention.
    Shape: (batch, 1, 1, seq_len) -> broadcasts to (batch, heads, seq_len, seq_len)
    Values: True = attend, False = mask out (padding)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    """Creates causal (look-ahead) mask to prevent attending to future tokens.
    Shape: (1, 1, size, size)
    Values: True = attend, False = mask out (future position)
    Lower triangular matrix: allows attending to current & past positions only
    """
    # Lower triangular: 1s on and below diagonal, 0s above
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)

def combine_masks(padding_mask, look_ahead_mask):
    """Combines padding mask and look-ahead mask.
    Both masks must be broadcastable to (batch, heads, seq_len, seq_len)
    Result: True = attend, False = mask out
    """
    return padding_mask & look_ahead_mask