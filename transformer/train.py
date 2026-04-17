import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import pickle
import math

from config import config
from models.transformer import Transformer
from utils.mask import create_padding_mask, create_look_ahead_mask


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("results", exist_ok=True)


dataset = load_dataset("cfilt/iitb-english-hindi")
data = dataset["train"]

src_texts = [x["translation"]["en"] for x in data]
tgt_texts = [x["translation"]["hi"] for x in data]

train_src, val_src, train_tgt, val_tgt = train_test_split(
    src_texts, tgt_texts, test_size=0.2, random_state=42
)


def build_vocab(sentences, max_size=15000):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3

    for sent in sentences:
        for word in sent.split():
            if word not in vocab and len(vocab) < max_size:
                vocab[word] = idx
                idx += 1
    return vocab

src_vocab = build_vocab(train_src)
tgt_vocab = build_vocab(train_tgt)

def encode(sentence, vocab):
    tokens = [vocab.get(w, 0) for w in sentence.split()]
    tokens = tokens[:config.max_len - 2]
    tokens = [1] + tokens + [2]
    return torch.tensor(tokens)


class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return encode(self.src[idx], src_vocab), encode(self.tgt[idx], tgt_vocab)

def collate_fn(batch):
    src, tgt = zip(*batch)

    src = torch.nn.utils.rnn.pad_sequence(
        src, batch_first=True, padding_value=config.pad_idx
    )
    tgt = torch.nn.utils.rnn.pad_sequence(
        tgt, batch_first=True, padding_value=config.pad_idx
    )

    return src, tgt

train_loader = DataLoader(
    TranslationDataset(train_src, train_tgt),
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    TranslationDataset(val_src, val_tgt),
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)


model = Transformer(
    len(src_vocab),
    len(tgt_vocab),
    config.d_model,
    config.num_layers,
    config.num_heads,
    config.d_ff,
    config.dropout
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9
)

criterion = nn.CrossEntropyLoss(
    ignore_index=config.pad_idx, label_smoothing=0.1
)


def create_tgt_mask(tgt, pad_idx):
    seq_len = tgt.size(1)

    look_ahead = create_look_ahead_mask(seq_len).to(tgt.device)
    padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    return look_ahead & padding_mask


loss_history = []
val_loss_history = []

for epoch in range(30):
    model.train()
    train_loss = 0

    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src, config.pad_idx).to(device)
        tgt_mask = create_tgt_mask(tgt_input, config.pad_idx)

        output = model(src, tgt_input, src_mask, tgt_mask)

        output = output.contiguous().view(-1, len(tgt_vocab))
        tgt_output = tgt_output.contiguous().view(-1)

        loss = criterion(output, tgt_output)


        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠ Skipping batch (NaN loss)")
            continue

        optimizer.zero_grad()
        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()


    model.eval()
    val_loss = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_padding_mask(src, config.pad_idx).to(device)
            tgt_mask = create_tgt_mask(tgt_input, config.pad_idx)

            output = model(src, tgt_input, src_mask, tgt_mask)

            output = output.contiguous().view(-1, len(tgt_vocab))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")


def generate(model, src, max_len=20):
    model.eval()

    src_mask = create_padding_mask(src, config.pad_idx).to(device)
    enc_out = model.encoder(src, src_mask)

    tgt = torch.tensor([[1]]).to(device)

    for _ in range(max_len):
        tgt_mask = create_tgt_mask(tgt, config.pad_idx)

        out = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        next_token = out[:, -1, :].argmax(-1).unsqueeze(1)

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == 2:
            break

    return tgt[0].cpu().tolist()


def evaluate():
    smooth = SmoothingFunction().method1

    preds, targets, bleu_scores = [], [], []

    dataset = TranslationDataset(val_src, val_tgt)

    for i in range(min(200, len(dataset))):
        src, tgt = dataset[i]
        src = src.unsqueeze(0).to(device)

        pred = generate(model, src)
        tgt_tokens = tgt.tolist()

        bleu_scores.append(
            sentence_bleu([tgt_tokens], pred, smoothing_function=smooth)
        )

        min_len = min(len(pred), len(tgt_tokens))
        preds.extend(pred[:min_len])
        targets.extend(tgt_tokens[:min_len])

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    bleu = sum(bleu_scores) / len(bleu_scores)

    print("\n===== FINAL METRICS =====")
    print(f"BLEU: {bleu:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")

evaluate()


torch.save(model.state_dict(), "transformer_final.pth")

with open("src_vocab.pkl", "wb") as f:
    pickle.dump(src_vocab, f)

with open("tgt_vocab.pkl", "wb") as f:
    pickle.dump(tgt_vocab, f)

print("\nTraining complete. Model saved.")