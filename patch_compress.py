"""
patch_compress.py

Stage 2 of the preprocessing pipeline.

Reads instructions.bin / responses.bin produced by tokenize.py,
compresses sequences that exceed max_seq_len using MegaByte-style
patch encoding, and writes new .bin files ready for the transformer.

Output .bin format (compressed-aware):
    [4 bytes]  num_sequences
    per sequence:
        [1 byte]   type flag  (0 = normal tokens, 1 = compressed patches)
        [2 bytes]  num_steps  (number of token IDs  OR  number of patch vectors)
        if type == 0 (normal):
            [num_steps * 4 bytes]  token IDs  (uint32)
        if type == 1 (compressed):
            [4 bytes]   patch_size used
            [4 bytes]   embed_dim
            [num_steps * embed_dim * 4 bytes]  patch vectors  (float32)

Usage:
    # First time: train the PatchEncoder on your dataset
    python patch_compress.py \
        --instructions instructions.bin \
        --responses    responses.bin \
        --max_seq_len  512 \
        --embed_dim    256 \
        --codebook_size 512 \
        --train_encoder \
        --epochs       3 \
        --instructions_output instructions_compressed.bin \
        --responses_output    responses_compressed.bin

    # After encoder is trained: just compress
    python patch_compress.py \
        --instructions instructions.bin \
        --responses    responses.bin \
        --max_seq_len  512 \
        --embed_dim    256 \
        --encoder_path patch_encoder.pt \
        --instructions_output instructions_compressed.bin \
        --responses_output    responses_compressed.bin
"""

from __future__ import annotations

import argparse
import math
import os
import struct
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Binary I/O  (matches tokenize.py format exactly)
# ─────────────────────────────────────────────────────────────────────────────

def read_bin(path: str) -> List[List[int]]:
    """Read a .bin file written by tokenize.py → list of token-ID lists."""
    sequences = []
    with open(path, "rb") as f:
        (num_seqs,) = struct.unpack("<I", f.read(4))
        for _ in range(num_seqs):
            (length,) = struct.unpack("<H", f.read(2))
            ids = list(struct.unpack(f"<{length}I", f.read(length * 4)))
            sequences.append(ids)
    print(f"[read_bin] {path}: {len(sequences):,} sequences")
    return sequences


def write_compressed_bin(path: str, records: list) -> None:
    """
    Write compressed-aware .bin file.

    records is a list of dicts:
        {"type": 0, "ids": [...]}                            # normal
        {"type": 1, "patch_size": int, "vectors": tensor}   # compressed
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(records)))
        for rec in records:
            if rec["type"] == 0:
                ids = rec["ids"]
                f.write(struct.pack("<B", 0))               # type flag
                f.write(struct.pack("<H", len(ids)))        # num steps
                f.write(struct.pack(f"<{len(ids)}I", *ids)) # token IDs
            else:
                vecs = rec["vectors"]                       # (num_patches, embed_dim)
                num_patches, embed_dim = vecs.shape
                f.write(struct.pack("<B", 1))               # type flag
                f.write(struct.pack("<H", num_patches))     # num steps
                f.write(struct.pack("<I", rec["patch_size"]))
                f.write(struct.pack("<I", embed_dim))
                # write float32 vectors row by row
                flat = vecs.cpu().float().numpy().flatten()
                f.write(struct.pack(f"<{len(flat)}f", *flat))
    print(f"[write] {path}: {len(records):,} records")


# ─────────────────────────────────────────────────────────────────────────────
#  Patch utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_patch_size(seq_len: int, max_len: int) -> int:
    """Minimum patch size so that num_patches <= max_len."""
    return math.ceil(seq_len / max_len)


def pad_and_split(ids: List[int], patch_size: int) -> List[List[int]]:
    """
    Pad sequence to a multiple of patch_size, then split into patches.
    Padding token = 0 (safe: transformer will learn to ignore it).
    """
    remainder = len(ids) % patch_size
    if remainder != 0:
        ids = ids + [0] * (patch_size - remainder)
    patches = [ids[i: i + patch_size] for i in range(0, len(ids), patch_size)]
    return patches


# ─────────────────────────────────────────────────────────────────────────────
#  PatchEncoder  — local transformer with CLS token
# ─────────────────────────────────────────────────────────────────────────────

class PatchEncoder(nn.Module):
    """
    Encodes a variable patch_size group of token IDs into a single vector.

    Architecture:
        token IDs  →  embedding  →  prepend CLS  →  small transformer  →  CLS output
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int = 4,
                 num_layers: int = 2, max_patch_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # token embedding shared with the main model vocabulary
        self.token_emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # positional encoding for patch positions (max_patch_size + 1 for CLS)
        self.pos_emb = nn.Embedding(max_patch_size + 1, embed_dim)

        # small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,          # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, num_patches, patch_size)  — token IDs

        Returns:
            (batch, num_patches, embed_dim)  — one vector per patch
        """
        B, N, P = patches.shape

        # embed tokens: (B, N, P, embed_dim)
        x = self.token_emb(patches)

        # flatten batch and patch dims for transformer: (B*N, P, embed_dim)
        x = x.view(B * N, P, self.embed_dim)

        # prepend CLS token: (B*N, P+1, embed_dim)
        cls = self.cls_token.expand(B * N, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # add positional embeddings
        positions = torch.arange(P + 1, device=patches.device)
        x = x + self.pos_emb(positions)

        # run transformer
        x = self.transformer(x)                   # (B*N, P+1, embed_dim)

        # take only CLS output (position 0)
        cls_out = x[:, 0, :]                       # (B*N, embed_dim)

        # project and normalize
        cls_out = self.norm(self.proj(cls_out))    # (B*N, embed_dim)

        # reshape back: (B, N, embed_dim)
        cls_out = cls_out.view(B, N, self.embed_dim)

        return cls_out


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder  — reconstructs token logits from patch vector (for training only)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDecoder(nn.Module):
    """
    Decodes a patch vector back into token ID logits.
    Used ONLY during encoder training for reconstruction loss.
    Not needed at inference time.
    """

    def __init__(self, vocab_size: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.decode = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * vocab_size),
        )
        self.vocab_size = vocab_size

    def forward(self, patch_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_vec: (batch, num_patches, embed_dim)

        Returns:
            logits: (batch, num_patches, patch_size, vocab_size)
        """
        B, N, D = patch_vec.shape
        out = self.decode(patch_vec)                               # (B, N, patch_size * vocab_size)
        out = out.view(B, N, self.patch_size, self.vocab_size)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Training dataset  — feeds patches from long sequences only
# ─────────────────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Builds fixed-size patches from sequences that exceed max_len.
    All patches in the dataset have the same patch_size for batching.
    """

    def __init__(self, sequences: List[List[int]], max_len: int, patch_size: int):
        self.patches = []

        for ids in sequences:
            if len(ids) <= max_len:
                continue  # only train on long sequences

            padded_patches = pad_and_split(ids, patch_size)
            for p in padded_patches:
                self.patches.append(p)

        print(f"[PatchDataset] {len(self.patches):,} patches from long sequences")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder(
    sequences: List[List[int]],
    vocab_size: int,
    embed_dim: int,
    patch_size: int,
    max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    save_path: str,
) -> PatchEncoder:

    print(f"\n[train] Training PatchEncoder")
    print(f"  patch_size  = {patch_size}")
    print(f"  embed_dim   = {embed_dim}")
    print(f"  epochs      = {epochs}")
    print(f"  device      = {device}")

    encoder = PatchEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_patch_size=patch_size,
    ).to(device)

    decoder = PatchDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(device)

    dataset = PatchDataset(sequences, max_len, patch_size)

    if len(dataset) == 0:
        print("[train] No long sequences found — skipping encoder training.")
        print("[train] All sequences fit within max_len already.")
        torch.save(encoder.state_dict(), save_path)
        return encoder

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        total_loss = 0.0
        total_steps = 0

        for step, batch in enumerate(loader):
            # batch: (B, patch_size)  — token IDs
            batch = batch.to(device)

            # add a fake num_patches dim: (B, 1, patch_size)
            batch_in = batch.unsqueeze(1)

            # encode → (B, 1, embed_dim)
            patch_vec = encoder(batch_in)

            # decode → (B, 1, patch_size, vocab_size)
            logits = decoder(patch_vec)

            # reconstruction loss: predict original token IDs
            # logits: (B, patch_size, vocab_size) after squeeze
            logits = logits.squeeze(1)                            # (B, patch_size, vocab_size)
            targets = batch                                        # (B, patch_size)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # ignore padding
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_steps += 1

            if (step + 1) % 100 == 0:
                avg = total_loss / total_steps
                print(f"  epoch {epoch+1}/{epochs}  step {step+1}/{len(loader)}  loss={avg:.4f}",
                      end="\r", flush=True)

        avg_loss = total_loss / max(total_steps, 1)
        print(f"\n  epoch {epoch+1}/{epochs} complete  avg_loss={avg_loss:.4f}")

    torch.save(encoder.state_dict(), save_path)
    print(f"[train] Encoder saved → {save_path}")

    return encoder


# ─────────────────────────────────────────────────────────────────────────────
#  Compression pass  — compress long sequences using trained encoder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compress_sequences(
    sequences: List[List[int]],
    encoder: PatchEncoder,
    max_len: int,
    device: torch.device,
) -> list:

    encoder.eval()
    records = []

    normal_count     = 0
    compressed_count = 0

    for idx, ids in enumerate(sequences):

        if len(ids) <= max_len:
            # ── short sequence: keep as normal tokens ─────────────────────
            records.append({"type": 0, "ids": ids})
            normal_count += 1

        else:
            # ── long sequence: compress via patching ──────────────────────
            patch_size = compute_patch_size(len(ids), max_len)
            patches    = pad_and_split(ids, patch_size)

            # tensor: (1, num_patches, patch_size)
            patches_tensor = torch.tensor(patches, dtype=torch.long) \
                                   .unsqueeze(0).to(device)

            # encode: (1, num_patches, embed_dim)
            patch_vecs = encoder(patches_tensor)

            # store: (num_patches, embed_dim)
            records.append({
                "type":       1,
                "patch_size": patch_size,
                "vectors":    patch_vecs.squeeze(0).cpu(),
            })
            compressed_count += 1

        if (idx + 1) % 1000 == 0:
            print(f"  compressed {idx+1:,}/{len(sequences):,}", end="\r", flush=True)

    print()
    print(f"[compress] Normal     : {normal_count:,}")
    print(f"[compress] Compressed : {compressed_count:,}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: compress long token sequences using MegaByte-style patch encoding"
    )

    # input/output
    parser.add_argument("--instructions",         required=True,  help="instructions.bin from tokenize.py")
    parser.add_argument("--responses",            required=True,  help="responses.bin from tokenize.py")
    parser.add_argument("--instructions_output",  default="instructions_compressed.bin")
    parser.add_argument("--responses_output",     default="responses_compressed.bin")

    # compression settings
    parser.add_argument("--max_seq_len",   type=int, default=512)
    parser.add_argument("--embed_dim",     type=int, default=256,   help="patch vector dimension")
    parser.add_argument("--vocab_size",    type=int, default=50257, help="must match tokenizer vocab size")

    # encoder training
    parser.add_argument("--train_encoder", action="store_true",  help="train the PatchEncoder before compressing")
    parser.add_argument("--encoder_path",  default="patch_encoder.pt", help="path to save/load encoder weights")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)

    # hardware
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"[patch_compress] device      = {device}")
    print(f"[patch_compress] max_seq_len = {args.max_seq_len}")
    print(f"[patch_compress] embed_dim   = {args.embed_dim}")

    # ── load sequences ────────────────────────────────────────────────────────
    inst_seqs = read_bin(args.instructions)
    resp_seqs = read_bin(args.responses)

    # ── determine global patch size from longest sequence ────────────────────
    all_seqs    = inst_seqs + resp_seqs
    max_found   = max(len(s) for s in all_seqs)
    patch_size  = compute_patch_size(max_found, args.max_seq_len)

    print(f"[patch_compress] longest sequence = {max_found:,} tokens")
    print(f"[patch_compress] patch_size       = {patch_size}")

    # ── build or load encoder ─────────────────────────────────────────────────
    encoder = PatchEncoder(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        max_patch_size=patch_size,
    ).to(device)

    if args.train_encoder:
        encoder = train_encoder(
            sequences   = all_seqs,
            vocab_size  = args.vocab_size,
            embed_dim   = args.embed_dim,
            patch_size  = patch_size,
            max_len     = args.max_seq_len,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            device      = device,
            save_path   = args.encoder_path,
        )
    elif os.path.exists(args.encoder_path):
        print(f"[patch_compress] Loading encoder from {args.encoder_path}")
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    else:
        print(f"[patch_compress] WARNING: no encoder found at {args.encoder_path}")
        print(f"[patch_compress] Using randomly initialized encoder (add --train_encoder to fix this)")

    # ── compress instructions ─────────────────────────────────────────────────
    print(f"\n[compress] Processing instructions...")
    inst_records = compress_sequences(inst_seqs, encoder, args.max_seq_len, device)
    write_compressed_bin(args.instructions_output, inst_records)

    # ── compress responses ────────────────────────────────────────────────────
    print(f"\n[compress] Processing responses...")
    resp_records = compress_sequences(resp_seqs, encoder, args.max_seq_len, device)
    write_compressed_bin(args.responses_output, resp_records)

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  DONE")
    print("=" * 50)
    print(f"  Instructions : {args.instructions_output}")
    print(f"  Responses    : {args.responses_output}")
    print(f"  Encoder      : {args.encoder_path}")
    print(f"  patch_size   : {patch_size}")
    print(f"  embed_dim    : {args.embed_dim}")
    print()
    print("  Next step: update your C++ dataloader to read")
    print("  the type flag (byte 0/1) before each sequence.")
    print("=" * 50)


if __name__ == "__main__":
    main()