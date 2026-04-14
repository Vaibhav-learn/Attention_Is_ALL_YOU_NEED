import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import sentencepiece as spm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config import config
from models.transformer import Transformer
from utils.mask import create_padding_mask, create_look_ahead_mask


# ==========================
# Setup
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("results", exist_ok=True)


# ==========================
# Load MULTI datasets (FIXED)
# ==========================
en_hi = load_dataset("cfilt/iitb-english-hindi")["train"].select(range(40000))
en_fr = load_dataset("opus_books", "en-fr")["train"].select(range(30000))
de_en = load_dataset("opus_books", "de-en")["train"].select(range(30000))  # ✅ FIXED

data = concatenate_datasets([en_hi, en_fr, de_en])
print("Total samples:", len(data))


# ==========================
# Prepare tokenizer data
# ==========================
with open("multi_data.txt", "w", encoding="utf-8") as f:
    for x in data:
        for lang in x["translation"]:
            f.write(x["translation"][lang] + "\n")


# ==========================
# Train SentencePiece
# ==========================
spm.SentencePieceTrainer.train(
    input="multi_data.txt",
    model_prefix="spm",
    vocab_size=8000
)

sp = spm.SentencePieceProcessor()
sp.load("spm.model")


# ==========================
# Encode
# ==========================
def encode(sentence):
    tokens = sp.encode(sentence, out_type=int)
    tokens = tokens[:config.max_len - 2]
    tokens = [1] + tokens + [2]
    return torch.tensor(tokens)


# ==========================
# Fix Language Direction
# ==========================
def get_pair(item):
    t = item["translation"]

    if "en" in t:
        src = t["en"]
        tgt = [v for k, v in t.items() if k != "en"][0]
    else:
        tgt = t["en"]
        src = [v for k, v in t.items() if k != "en"][0]

    return src, tgt


# ==========================
# Dataset
# ==========================
class MultiDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = get_pair(self.data[idx])

        src = encode(src_text)
        tgt = encode(tgt_text)

        return src, tgt


# ==========================
# Collate
# ==========================
def collate_fn(batch):
    src, tgt = zip(*batch)

    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)

    return src, tgt


loader = DataLoader(
    MultiDataset(data),
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)


# ==========================
# Model
# ==========================
model = Transformer(
    8000, 8000,
    config.d_model,
    config.num_layers,
    config.num_heads,
    config.d_ff,
    config.dropout
).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ==========================
# Training
# ==========================
loss_history = []

for epoch in range(15):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src, 0).to(device)

        tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        output = model(src, tgt_input, src_mask, tgt_mask)

        output = output.reshape(-1, 8000)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")


# ==========================
# Generate
# ==========================
def generate(model, src, max_len=20):
    model.eval()

    src_mask = create_padding_mask(src, 0).to(device)
    enc_out = model.encoder(src, src_mask)

    tgt = torch.tensor([[1]]).to(device)

    for _ in range(max_len):
        tgt_mask = create_look_ahead_mask(tgt.size(1)).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        out = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        next_token = out[:, -1, :].argmax(-1).unsqueeze(1)

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == 2:
            break

    return tgt[0].cpu().tolist()


# ==========================
# Evaluation + Save
# ==========================
def evaluate():
    smooth = SmoothingFunction().method1

    preds, targets, bleu_scores = [], [], []

    for i in range(200):
        src, tgt = loader.dataset[i]
        src = src.unsqueeze(0).to(device)

        pred = generate(model, src)
        tgt_tokens = tgt.tolist()

        bleu_scores.append(sentence_bleu([tgt_tokens], pred, smoothing_function=smooth))

        min_len = min(len(pred), len(tgt_tokens))
        preds.extend(pred[:min_len])
        targets.extend(tgt_tokens[:min_len])

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    bleu = sum(bleu_scores) / len(bleu_scores)

    print("\n===== Evaluation =====")
    print(f"BLEU: {bleu:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")

    with open("results/metrics.txt", "w") as f:
        f.write(f"BLEU: {bleu}\nAccuracy: {acc}\nF1: {f1}")

    cm = confusion_matrix(targets[:500], preds[:500])

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, cmap="Blues")
    plt.savefig("results/confusion_matrix.png")
    plt.close()


evaluate()


# ==========================
# Save Loss Graph
# ==========================
plt.plot(loss_history)
plt.title("Loss Curve")
plt.savefig("results/loss.png")
plt.close()


# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "multi_transformer.pth")
print("All results saved in /results")