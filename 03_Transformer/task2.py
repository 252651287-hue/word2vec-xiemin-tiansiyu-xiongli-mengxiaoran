import argparse
import copy
import math
import os
import random
import time
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

try:
    from nltk.translate.bleu_score import corpus_bleu
except Exception as e:
    raise ImportError("请先安装 nltk：pip install nltk") from e


# =========================
# 1. 工具与数据
# =========================
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.strip().lower())


def read_parallel_file(path: str):
    src, tgt = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            s, t = line.split("\t", 1)
            src.append(s)
            tgt.append(t)
    return src, tgt


class Vocab:
    def __init__(self, token_lists, min_freq=1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        self.specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.stoi = {tok: idx for idx, tok in enumerate(self.specials)}
        idx = len(self.specials)
        for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq and tok not in self.stoi:
                self.stoi[tok] = idx
                idx += 1
        self.itos = {idx: tok for tok, idx in self.stoi.items()}
        self.pad_idx = self.stoi["<pad>"]
        self.unk_idx = self.stoi["<unk>"]
        self.sos_idx = self.stoi["<sos>"]
        self.eos_idx = self.stoi["<eos>"]

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens, add_sos=False, add_eos=False):
        ids = []
        if add_sos:
            ids.append(self.sos_idx)
        ids.extend([self.stoi.get(tok, self.unk_idx) for tok in tokens])
        if add_eos:
            ids.append(self.eos_idx)
        return ids


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=30):
        self.examples = []
        for s, t in zip(src_sentences, tgt_sentences):
            s_tok = tokenize(s)[:max_len]
            t_tok = tokenize(t)[: max_len - 1]
            src_ids = src_vocab.encode(s_tok, add_eos=True)
            tgt_ids = tgt_vocab.encode(t_tok, add_sos=True, add_eos=True)
            self.examples.append((src_ids, tgt_ids, s, t))
        self.src_pad_idx = src_vocab.pad_idx
        self.tgt_pad_idx = tgt_vocab.pad_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Collator:
    def __init__(self, src_pad_idx, tgt_pad_idx):
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def __call__(self, batch):
        src_ids, tgt_ids, raw_src, raw_tgt = zip(*batch)
        src_max = max(len(x) for x in src_ids)
        tgt_max = max(len(x) for x in tgt_ids)
        src_batch = torch.full((len(batch), src_max), self.src_pad_idx, dtype=torch.long)
        tgt_batch = torch.full((len(batch), tgt_max), self.tgt_pad_idx, dtype=torch.long)
        for i, ids in enumerate(src_ids):
            src_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        for i, ids in enumerate(tgt_ids):
            tgt_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return src_batch, tgt_batch, raw_src, raw_tgt


def split_train_val(src, tgt, val_ratio=0.1, seed=42):
    idxs = list(range(len(src)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    val_size = int(len(idxs) * val_ratio)
    val_idx = set(idxs[:val_size])
    train_src, train_tgt, val_src, val_tgt = [], [], [], []
    for i, (s, t) in enumerate(zip(src, tgt)):
        if i in val_idx:
            val_src.append(s)
            val_tgt.append(t)
        else:
            train_src.append(s)
            train_tgt.append(t)
    return train_src, train_tgt, val_src, val_tgt


def build_loaders(train_path, test_path, max_len, batch_size, val_ratio, seed, num_workers=0):
    train_src_raw, train_tgt_raw = read_parallel_file(train_path)
    test_src_raw, test_tgt_raw = read_parallel_file(test_path)
    train_src_raw, train_tgt_raw, val_src_raw, val_tgt_raw = split_train_val(train_src_raw, train_tgt_raw, val_ratio=val_ratio, seed=seed)

    src_vocab = Vocab([tokenize(x)[:max_len] for x in train_src_raw], min_freq=1)
    tgt_vocab = Vocab([tokenize(x)[: max_len - 1] for x in train_tgt_raw], min_freq=1)

    train_ds = TranslationDataset(train_src_raw, train_tgt_raw, src_vocab, tgt_vocab, max_len=max_len)
    val_ds = TranslationDataset(val_src_raw, val_tgt_raw, src_vocab, tgt_vocab, max_len=max_len)
    test_ds = TranslationDataset(test_src_raw, test_tgt_raw, src_vocab, tgt_vocab, max_len=max_len)

    collate = Collator(src_vocab.pad_idx, tgt_vocab.pad_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# =========================
# 2. 模型定义
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class AdditiveAttention(nn.Module):
    def __init__(self, d_head, d_attn=64, dropout=0.1):
        super().__init__()
        self.w_q = nn.Linear(d_head, d_attn, bias=False)
        self.w_k = nn.Linear(d_head, d_attn, bias=False)
        self.v = nn.Linear(d_attn, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q,k,v: [B,H,L,D]
        q_proj = self.w_q(q).unsqueeze(3)   # [B,H,Lq,1,A]
        k_proj = self.w_k(k).unsqueeze(2)   # [B,H,1,Lk,A]
        scores = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)  # [B,H,Lq,Lk]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_type="dot", d_attn=64, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if attention_type == "dot":
            self.attn = ScaledDotProductAttention(dropout)
        elif attention_type == "additive":
            self.attn = AdditiveAttention(self.d_head, d_attn=d_attn, dropout=dropout)
        else:
            raise ValueError("attention_type 只能是 'dot' 或 'additive'")

    def forward(self, q, k, v, mask=None):
        bsz = q.size(0)
        q = self.w_q(q).view(bsz, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(bsz, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(bsz, -1, self.n_heads, self.d_head).transpose(1, 2)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        out, _ = self.attn(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        return self.dropout(self.w_o(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attention_type="dot", d_attn=64, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, attention_type, d_attn, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        x = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(x))
        x = self.ff(src)
        src = self.norm2(src + self.dropout(x))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attention_type="dot", d_attn=64, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, attention_type, d_attn, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, attention_type, d_attn, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        x = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(x))
        x = self.cross_attn(tgt, memory, memory, src_mask)
        tgt = self.norm2(tgt + self.dropout(x))
        x = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(x))
        return tgt


class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_heads=4, n_layers=3, d_ff=1024,
                 attention_type="dot", d_attn=64, dropout=0.15, max_len=256):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, attention_type, d_attn, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, attention_type, d_attn, dropout)
            for _ in range(n_layers)
        ])
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.generator.weight = self.tgt_embed.weight
        self.d_model = d_model

    def make_src_mask(self, src, pad_idx=0):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        L = tgt.size(1)
        causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=tgt.device)).unsqueeze(0).unsqueeze(1)
        return pad_mask & causal

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.dropout(self.pos(x))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.dropout(self.pos(x))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.generator(x)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)


# =========================
# 3. 训练、评估、展示
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for src, tgt, _, _ in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src, tgt, _, _ in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def greedy_decode(model, src_batch, tgt_vocab, device, max_len=40):
    model.eval()
    src_batch = src_batch.to(device)
    memory, src_mask = model.encode(src_batch)
    ys = torch.full((src_batch.size(0), 1), tgt_vocab.sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(src_batch.size(0), dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = model.decode(ys, memory, src_mask)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        finished |= next_tok.squeeze(1).eq(tgt_vocab.eos_idx)
        if finished.all():
            break
    return ys


@torch.no_grad()
def evaluate_bleu(model, loader, tgt_vocab, device, decode_max_len=40):
    model.eval()
    refs, hyps = [], []
    for src, tgt, _, _ in loader:
        pred = greedy_decode(model, src, tgt_vocab, device, max_len=decode_max_len).cpu().tolist()
        tgt = tgt.tolist()
        for p_ids, t_ids in zip(pred, tgt):
            ref = []
            for idx in t_ids:
                if idx == tgt_vocab.eos_idx:
                    break
                tok = tgt_vocab.itos.get(idx, "<unk>")
                if tok not in ["<pad>", "<sos>"]:
                    ref.append(tok)
            hyp = []
            for idx in p_ids:
                if idx == tgt_vocab.eos_idx:
                    break
                tok = tgt_vocab.itos.get(idx, "<unk>")
                if tok not in ["<pad>", "<sos>"]:
                    hyp.append(tok)
            refs.append([ref])
            hyps.append(hyp)
    bleu1 = corpus_bleu(refs, hyps, weights=(1.0, 0, 0, 0)) * 100
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    return bleu1, bleu4


@torch.no_grad()
def show_examples(model, test_loader, tgt_vocab, device, num_examples=6):
    outputs = []
    for src, tgt, raw_src, raw_tgt in test_loader:
        pred = greedy_decode(model, src, tgt_vocab, device, max_len=40).cpu().tolist()
        for s, t, p_ids in zip(raw_src, raw_tgt, pred):
            hyp = []
            for idx in p_ids:
                if idx == tgt_vocab.eos_idx:
                    break
                tok = tgt_vocab.itos.get(idx, "<unk>")
                if tok not in ["<pad>", "<sos>"]:
                    hyp.append(tok)
            outputs.append((s, t, " ".join(hyp)))
            if len(outputs) >= num_examples:
                return outputs
    return outputs


def train_single_model(name, model, train_loader, val_loader, test_loader, tgt_vocab, args, device, save_path):
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_losses, val_losses, epoch_times = [], [], []
    best_val = float("inf")
    bad_epochs = 0

    print("=" * 60)
    print(f"开始训练：{name}")
    print("=" * 60)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(elapsed)

        print(f"Epoch {epoch:02d}/{args.epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 耗时: {elapsed:.2f}s")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"{name} 触发早停：连续 {args.patience} 轮验证集未提升。")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    bleu1, bleu4 = evaluate_bleu(model, test_loader, tgt_vocab, device, decode_max_len=args.max_len + 10)
    examples = show_examples(model, test_loader, tgt_vocab, device, num_examples=6)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "best_val": best_val,
        "bleu1": bleu1,
        "bleu4": bleu4,
        "examples": examples,
    }


def plot_losses(dot_res, add_res, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(dot_res["train_losses"], label="Dot Train Loss")
    plt.plot(dot_res["val_losses"], label="Dot Val Loss")
    plt.plot(add_res["train_losses"], label="Additive Train Loss")
    plt.plot(add_res["val_losses"], label="Additive Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task2: Dot vs Additive Attention Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task2: 点积注意力 vs 加性注意力 对比实验")
    parser.add_argument("--train_path", type=str, default="eng-fra_train_data.txt")
    parser.add_argument("--test_path", type=str, default="eng-fra_test_data.txt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--d_attn", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dot_save", type=str, default="task2_best_dot_model.pt")
    parser.add_argument("--add_save", type=str, default="task2_best_additive_model.pt")
    parser.add_argument("--figure_save", type=str, default="task2_loss_compare.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_loaders(
        args.train_path, args.test_path, args.max_len, args.batch_size, args.val_ratio, args.seed
    )

    print("=" * 60)
    print("Task 2: 点积注意力改为加性注意力，并比较效果")
    print(f"设备: {device}")
    print("=" * 60)
    print(f"训练集 batch 数: {len(train_loader)}")
    print(f"验证集 batch 数: {len(val_loader)}")
    print(f"测试集 batch 数: {len(test_loader)}")
    print(f"英文词表大小: {len(src_vocab)}")
    print(f"法文词表大小: {len(tgt_vocab)}")

    common_kwargs = dict(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=max(args.max_len + 5, 128),
    )

    dot_model = TransformerTranslator(attention_type="dot", d_attn=args.d_attn, **common_kwargs).to(device)
    add_model = TransformerTranslator(attention_type="additive", d_attn=args.d_attn, **common_kwargs).to(device)

    dot_res = train_single_model("点积注意力模型", dot_model, train_loader, val_loader, test_loader, tgt_vocab, args, device, args.dot_save)
    add_res = train_single_model("加性注意力模型", add_model, train_loader, val_loader, test_loader, tgt_vocab, args, device, args.add_save)

    plot_losses(dot_res, add_res, args.figure_save)

    print("\n===== Task 2 最终对比结果 =====")
    print("| 注意力类型 | BLEU-1(%) | BLEU-4(%) | 平均epoch耗时(s) | 最优验证损失 |")
    print("|-----------|-----------|-----------|------------------|-------------|")
    print(f"| 点积注意力 | {dot_res['bleu1']:.2f} | {dot_res['bleu4']:.2f} | {np.mean(dot_res['epoch_times']):.2f} | {dot_res['best_val']:.4f} |")
    print(f"| 加性注意力 | {add_res['bleu1']:.2f} | {add_res['bleu4']:.2f} | {np.mean(add_res['epoch_times']):.2f} | {add_res['best_val']:.4f} |")

    print("\n===== 点积注意力翻译示例 =====")
    for src_text, tgt_text, pred_text in dot_res["examples"]:
        print(f"英文: {src_text}")
        print(f"真实法文: {tgt_text}")
        print(f"模型法文: {pred_text}")
        print("-" * 50)

    print("\n===== 加性注意力翻译示例 =====")
    for src_text, tgt_text, pred_text in add_res["examples"]:
        print(f"英文: {src_text}")
        print(f"真实法文: {tgt_text}")
        print(f"模型法文: {pred_text}")
        print("-" * 50)

    print(f"点积注意力最优模型: {args.dot_save}")
    print(f"加性注意力最优模型: {args.add_save}")
    print(f"损失曲线图: {args.figure_save}")


if __name__ == "__main__":
    main()
