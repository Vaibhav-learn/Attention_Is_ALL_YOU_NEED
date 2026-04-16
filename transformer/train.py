import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import pickle
import math

from config import config
from models.transformer import Transformer
from utils.mask import create_padding_mask, create_look_ahead_mask



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("results", exist_ok=True)



dataset = load_dataset("cfilt/iitb-english-hindi")
data = dataset["train"]
print(f"Total dataset size: {len(data)}")



src_texts = [x["translation"]["en"] for x in data] # type: ignore
tgt_texts = [x["translation"]["hi"] for x in data] # type: ignore

train_src, val_src, train_tgt, val_tgt = train_test_split(
    src_texts, tgt_texts, test_size=0.2, random_state=42
)

print(f"Train: {len(train_src)} | Val: {len(val_src)}")


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

    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=config.pad_idx) # type: ignore
    tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=config.pad_idx) # type: ignore

    return src, tgt


train_loader = DataLoader(
    TranslationDataset(train_src, train_tgt),
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    TranslationDataset(val_src, val_tgt),
    batch_size=8,
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


criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1)  # 🔥 Added label smoothing
optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

class WarmupScheduler:
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

scheduler = WarmupScheduler(optimizer, config.d_model, warmup_steps=2000)



class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("⚠ Early stopping triggered!")
                return True
            return False


class ModelCheckpoint:
    def __init__(self, path="best_model.pth"):
        self.best_loss = float("inf")
        self.path = path

    def __call__(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            print("Best model saved!")


early_stopping = EarlyStopping(patience=3)
checkpoint = ModelCheckpoint()



loss_history = []
val_loss_history = []
bleu_history = []

print("\n===== Starting Training with Optimizations =====")
print(f"Batch size: 8 | Vocab: 15K | Epochs: 50 | LR: Warmup Scheduler")
print(f"Dataset: {len(train_src)} train, {len(val_src)} val")
print("="*60)

for epoch in range(50):
    model.train()
    train_loss = 0

    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src, config.pad_idx).to(device)

        tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        output = model(src, tgt_input, src_mask, tgt_mask)

        output = output.reshape(-1, len(tgt_vocab))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔥 Gradient clipping
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_padding_mask(src, config.pad_idx).to(device)

            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

            output = model(src, tgt_input, src_mask, tgt_mask)

            output = output.reshape(-1, len(tgt_vocab))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)
    
    print(f"Epoch {epoch+1:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    loss_history.append(train_loss)

    checkpoint(model, val_loss)

    if early_stopping(val_loss):
        break



def generate(model, src, max_len=20):
    model.eval()

    src_mask = create_padding_mask(src, config.pad_idx).to(device)
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


def evaluate():
    smooth = SmoothingFunction().method1

    preds, targets, bleu_scores = [], [], []

    dataset = TranslationDataset(val_src, val_tgt)

    for i in range(min(500, len(dataset))):  # 🔥 Evaluate on 500 samples
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

    print("\n===== Evaluation =====")
    print(f"BLEU: {bleu:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")

    with open("results/metrics.txt", "w") as f:
        f.write(f"BLEU: {bleu}\nAccuracy: {acc}\nF1: {f1}")
    
    print("\n" + "="*60)
    print("\ud83d� FINAL EVALUATION METRICS:")
    print(f"BLEU Score:    {bleu:.4f} (↑ Target: >20)")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("="*60)


evaluate()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(loss_history, label='Train Loss', marker='o', markersize=3)
axes[0].plot(val_loss_history, label='Val Loss', marker='s', markersize=3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss Comparison')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_metrics.png", dpi=150)
plt.close()
print("\u2705 Training metrics saved to results/training_metrics.png")


torch.save(model.state_dict(), "transformer_final.pth")

with open("src_vocab.pkl", "wb") as f:
    pickle.dump(src_vocab, f)

with open("tgt_vocab.pkl", "wb") as f:
    pickle.dump(tgt_vocab, f)

print("\nAll results saved in /results")
print(f"\nFiles saved:")
print(f"  • best_model.pth - Best checkpoint")
print(f"  • transformer_final.pth - Final model")
print(f"  • src_vocab.pkl, tgt_vocab.pkl - Vocabularies")
print(f"  • results/training_metrics.png - Training curves")
print(f"  • results/metrics.txt - Final metrics")