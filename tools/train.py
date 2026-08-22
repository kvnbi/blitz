"""Train the Blitz NNUE network and export it in the engine's binary format.

Architecture (must stay in step with src/nnue/arch.h):

    (KING_BUCKETS x 768) --EmbeddingBag--> HL   per perspective
    concat(us, them) -> SCReLU -> per-output-bucket linear -> 1

The network predicts a raw value `y`; the engine reads `y * NET_SCALE` as
centipawns, and `sigmoid(y)` is trained against a blend of the search score and
the game result.
"""
import argparse
import os
import pathlib
import re
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ---- must match src/nnue/arch.h -------------------------------------------
def _arch(name, default):
    """Read a constant straight out of src/nnue/arch.h.

    These values must match the engine exactly. Parsing the header rather than
    duplicating the numbers here means they cannot silently drift apart, which
    has happened before and is invisible until the parity test runs.
    """
    header = pathlib.Path(__file__).resolve().parent.parent / "src" / "nnue" / "arch.h"
    try:
        text = header.read_text()
    except OSError:
        return default
    m = re.search(r"constexpr int " + name + r"\s*=\s*([0-9*+\- ]+);", text)
    if not m:
        return default
    return int(eval(m.group(1), {"__builtins__": {}}, {}))


FT_IN = _arch("FT_IN", 768)
KING_BUCKETS = _arch("KING_BUCKETS", 8)
HL = int(os.environ.get("BLITZ_HL", 1024))
OUTPUT_BUCKETS = _arch("OUTPUT_BUCKETS", 8)
QA = _arch("QA", 255)
QB = _arch("QB", 64)
NET_SCALE = _arch("NET_SCALE", 400)
NET_MAGIC = 0x564E4554
NET_VERSION = 1

KING_BUCKET_MAP = np.array([
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 3, 3, 3, 3, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
], dtype=np.int64)

NUM_FEATURES = KING_BUCKETS * FT_IN
PAD_IDX = NUM_FEATURES          # never contributes; EmbeddingBag padding_idx
MAX_PIECES = 32


# ---------------------------------------------------------------------------
def decode_batch(raw):
    """raw: uint8 array [B, 32] -> (idx_white, idx_black, stm, score, result, bucket).

    Fully vectorised; mirrors nnue::feature_index() exactly.
    """
    b = raw.shape[0]
    occ = raw[:, 0:8].copy().view(np.uint64).reshape(b)
    nibbles = raw[:, 8:24]
    score = raw[:, 24:26].copy().view(np.int16).reshape(b).astype(np.float32)
    result = raw[:, 26].astype(np.float32)
    stm = raw[:, 27].astype(np.int64)

    # [b, 64] occupancy mask, LSB = A1
    bits = ((occ[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1))
    mask = bits.astype(bool)

    # Piece code per occupied square, in ascending square order.
    codes32 = np.empty((b, MAX_PIECES), dtype=np.int64)
    codes32[:, 0::2] = nibbles & 0xF
    codes32[:, 1::2] = nibbles >> 4

    rows, sqs = np.nonzero(mask)
    order = np.cumsum(mask, axis=1) - 1            # rank of each square within its row
    piece = codes32[rows, order[rows, sqs]]        # 0..11 = colour*6 + type

    colour = piece // 6
    ptype = piece % 6                              # 0 pawn .. 5 king

    # King squares per colour.
    wk = np.zeros(b, dtype=np.int64)
    bk = np.zeros(b, dtype=np.int64)
    is_king = ptype == 5
    wk[rows[is_king & (colour == 0)]] = sqs[is_king & (colour == 0)]
    bk[rows[is_king & (colour == 1)]] = sqs[is_king & (colour == 1)]

    piece_count = mask.sum(axis=1)
    bucket = np.minimum((piece_count - 2) // ((32 - 2 + OUTPUT_BUCKETS - 1) // OUTPUT_BUCKETS),
                        OUTPUT_BUCKETS - 1)

    def features(persp):
        ksq = wk if persp == 0 else bk
        ksq_r = ksq[rows]
        mirror = ((ksq_r & 7) >= 4).astype(np.int64) * 7
        oriented = (sqs ^ (persp * 56)) ^ mirror
        okings = (ksq ^ (persp * 56)) ^ (((ksq & 7) >= 4).astype(np.int64) * 7)
        bkt = KING_BUCKET_MAP[okings][rows]
        rel = (colour != persp).astype(np.int64)
        idx = bkt * FT_IN + (rel * 6 + ptype) * 64 + oriented

        out = np.full((b, MAX_PIECES), PAD_IDX, dtype=np.int64)
        out[rows, order[rows, sqs]] = idx
        return out

    return features(0), features(1), stm, score, result, bucket.astype(np.int64)


# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.EmbeddingBag(NUM_FEATURES + 1, HL, mode="sum", padding_idx=PAD_IDX)
        self.ft_bias = nn.Parameter(torch.zeros(HL))
        self.out_w = nn.Parameter(torch.zeros(OUTPUT_BUCKETS, 2 * HL))
        self.out_b = nn.Parameter(torch.zeros(OUTPUT_BUCKETS))

        # Small init keeps the quantised weights inside int16 from the start.
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)
        with torch.no_grad():
            self.ft.weight[PAD_IDX].zero_()
        nn.init.uniform_(self.out_w, -0.05, 0.05)

    def forward(self, idx_w, idx_b, stm, bucket):
        acc_w = self.ft(idx_w) + self.ft_bias
        acc_b = self.ft(idx_b) + self.ft_bias

        # "us" first, "them" second, exactly as the engine concatenates them.
        stm_f = stm.view(-1, 1).float()
        us = acc_w * (1 - stm_f) + acc_b * stm_f
        them = acc_b * (1 - stm_f) + acc_w * stm_f

        x = torch.cat([us, them], dim=1).clamp(0.0, 1.0) ** 2
        w = self.out_w[bucket]
        return (x * w).sum(dim=1) + self.out_b[bucket]


# ---------------------------------------------------------------------------
def export(net, path):
    with torch.no_grad():
        ft_w = net.ft.weight[:NUM_FEATURES].cpu().numpy()
        ft_b = net.ft_bias.cpu().numpy()
        o_w = net.out_w.cpu().numpy()
        o_b = net.out_b.cpu().numpy()

    def q(a, scale, name):
        v = np.round(a * scale)
        if np.abs(v).max() > 32767:
            raise SystemExit(f"{name} overflows int16 after quantisation "
                             f"(max {np.abs(v).max():.0f}); lower the learning rate "
                             f"or add weight clipping")
        return v.astype(np.int16)

    with open(path, "wb") as f:
        f.write(struct.pack("<IIIII", NET_MAGIC, NET_VERSION, HL, KING_BUCKETS, OUTPUT_BUCKETS))
        f.write(q(ft_w, QA, "feature weights").tobytes())
        f.write(q(ft_b, QA, "feature bias").tobytes())
        f.write(q(o_w, QB, "output weights").tobytes())
        f.write(q(o_b, QA * QB, "output bias").tobytes())
    print(f"exported {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="datagen .bin file")
    ap.add_argument("-o", "--out", default="blitz.nnue")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wdl", type=float, default=0.4,
                    help="weight of the game result vs the search score")
    ap.add_argument("--device", default=None)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    raw = np.fromfile(args.data, dtype=np.uint8)
    n = raw.size // 32
    raw = raw[:n * 32].reshape(n, 32)
    print(f"{n} positions", flush=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    raw = raw[perm]
    n_val = max(1, int(n * args.val_frac))
    val_raw, train_raw = raw[:n_val], raw[n_val:]

    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val = float("inf")

    def batches(data, batch, shuffle=True):
        order = np.random.permutation(len(data)) if shuffle else np.arange(len(data))
        for i in range(0, len(order) - batch + 1, batch):
            yield data[order[i:i + batch]]

    def to_device(chunk):
        iw, ib, stm, score, result, bucket = decode_batch(chunk)
        t = lambda a, d=torch.long: torch.as_tensor(a, dtype=d, device=device)
        return (t(iw), t(ib), t(stm), t(bucket),
                t(score, torch.float32), t(result, torch.float32))

    for epoch in range(args.epochs):
        net.train()
        t0, tot, seen = time.time(), 0.0, 0
        nbatches = len(train_raw) // args.batch
        for bi, chunk in enumerate(batches(train_raw, args.batch)):
            iw, ib, stm, bucket, score, result = to_device(chunk)
            target = (args.wdl * (result / 2.0)
                      + (1 - args.wdl) * torch.sigmoid(score / NET_SCALE))
            pred = torch.sigmoid(net(iw, ib, stm, bucket))
            loss = ((pred - target) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():           # keep quantisation headroom
                net.ft.weight.clamp_(-1.98, 1.98)      # * QA  must fit int16
                net.out_w.clamp_(-127.0, 127.0)        # * QB  must fit int16
                net.out_b.clamp_(-1.98, 1.98)          # * QA*QB must fit int16
            tot += loss.item() * len(chunk)
            seen += len(chunk)
            if bi % 25 == 0:
                rate = seen / max(time.time() - t0, 1e-6)
                print(f"  epoch {epoch + 1} batch {bi}/{nbatches} "
                      f"loss {tot / max(seen, 1):.5f} {rate / 1000:.0f}k pos/s",
                      flush=True)

        net.eval()
        with torch.no_grad():
            vtot = vseen = 0
            for chunk in batches(val_raw, min(args.batch, len(val_raw)), shuffle=False):
                iw, ib, stm, bucket, score, result = to_device(chunk)
                target = (args.wdl * (result / 2.0)
                          + (1 - args.wdl) * torch.sigmoid(score / NET_SCALE))
                pred = torch.sigmoid(net(iw, ib, stm, bucket))
                vtot += ((pred - target) ** 2).mean().item() * len(chunk)
                vseen += len(chunk)
        sched.step()
        val = vtot / max(vseen, 1)
        print(f"epoch {epoch + 1}/{args.epochs}  train {tot / max(seen,1):.5f}  "
              f"val {val:.5f}  {time.time() - t0:.1f}s", flush=True)

        # Export only when validation actually improved.  With a small dataset
        # the last epoch is often not the best one, and silently shipping it
        # would hand the engine a worse network than it already had.
        if val < best_val:
            best_val = val
            export(net, args.out)
        else:
            print(f"  val did not improve on {best_val:.5f}; keeping the earlier net",
                  flush=True)

    print(f"best validation loss: {best_val:.5f} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
