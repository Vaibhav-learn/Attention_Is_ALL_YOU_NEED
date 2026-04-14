import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset
from typing import Any
import pickle

from config import config
from models.transformer import Transformer
from utils.mask import create_padding_mask, create_look_ahead_mask


dataset = load_dataset("cfilt/iitb-english-hindi")
train_data = dataset["train"].select(range(2000))  # you can increase later


def build_vocab(sentences, max_size=5000):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3

    for sent in sentences:
        for word in sent.split():
            if word not in vocab and len(vocab) < max_size:
                vocab[word] = idx
                idx += 1

    return vocab


src_texts = [x["translation"]["en"] for x in train_data] # type: ignore
tgt_texts = [x["translation"]["hi"] for x in train_data] # type: ignore

src_vocab = build_vocab(src_texts)
tgt_vocab = build_vocab(tgt_texts)


def encode(sentence, vocab):
    tokens = [vocab.get(w, 0) for w in sentence.split()]
    tokens = tokens[:config.max_len - 2]
    tokens = [1] + tokens + [2]  # <sos>, <eos>
    return torch.tensor(tokens, dtype=torch.long)


class TranslationDataset(Dataset):
    def __len__(self):
        return len(train_data)

    def __getitem__(self, idx):
        item: Any = train_data[idx]

        src = encode(item["translation"]["en"], src_vocab)
        tgt = encode(item["translation"]["hi"], tgt_vocab)

        return src, tgt


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=config.pad_idx # type: ignore
    )

    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=config.pad_idx # type: ignore
    )

    return src_batch, tgt_batch


loader = DataLoader(
    TranslationDataset(),
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Transformer(
    len(src_vocab),
    len(tgt_vocab),
    config.d_model,
    config.num_layers,
    config.num_heads,
    config.d_ff,
    config.dropout
).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


loss_history = []

for epoch in range(config.epochs):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Masks
        src_mask = create_padding_mask(src, config.pad_idx).to(device)

        tgt_len = tgt_input.size(1)
        tgt_mask = create_look_ahead_mask(tgt_len).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        # Forward
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Reshape
        output = output.reshape(-1, len(tgt_vocab))
        tgt_output = tgt_output.reshape(-1)

        # Loss
        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "transformer_model.pth")
print("Model saved!")


with open("src_vocab.pkl", "wb") as f:
    pickle.dump(src_vocab, f)

with open("tgt_vocab.pkl", "wb") as f:
    pickle.dump(tgt_vocab, f)

print("Vocabulary saved!")


plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()