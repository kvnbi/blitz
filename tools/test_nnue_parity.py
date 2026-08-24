"""Check the C++ NNUE inference against the PyTorch model that produced the net.

A mismatch in feature indexing, perspective ordering, or output bucketing is
invisible in training but destroys playing strength, so the two implementations
are compared directly on real positions.
"""
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "tools")
from train import (Net, decode_batch, NET_SCALE, export, QA, QB, HL,
                   NUM_FEATURES, OUTPUT_BUCKETS)
from verify_data import decode as decode_sample
from uci_driver import Engine

def trunc_div(a, b):
    """C++ integer division truncates toward zero; numpy // floors."""
    return np.trunc(a / b).astype(np.int64)

def quantised_reference(net, iw, ib, stm, bucket):
    """Replay the engine's exact int16/int64 arithmetic in numpy.

    This separates "the two implementations disagree" (a bug) from "int16
    quantisation loses a few centipawns" (expected).
    """
    with torch.no_grad():
        ftw = np.round(net.ft.weight[:NUM_FEATURES].numpy() * QA).astype(np.int64)
        ftb = np.round(net.ft_bias.numpy() * QA).astype(np.int64)
        ow = np.round(net.out_w.numpy() * QB).astype(np.int64)
        ob = np.round(net.out_b.numpy() * QA * QB).astype(np.int64)

    padded = np.concatenate([ftw, np.zeros((1, HL), dtype=np.int64)], axis=0)
    acc_w = padded[iw].sum(axis=1) + ftb
    acc_b = padded[ib].sum(axis=1) + ftb

    sel = stm.reshape(-1, 1)
    us = np.where(sel == 0, acc_w, acc_b)
    them = np.where(sel == 0, acc_b, acc_w)

    cu = np.clip(us, 0, QA)
    ct = np.clip(them, 0, QA)
    total = ((cu * cu) * ow[bucket][:, :HL]).sum(axis=1) \
          + ((ct * ct) * ow[bucket][:, HL:]).sum(axis=1)

    v = trunc_div(total, QA) + ob[bucket]
    return trunc_div(v * NET_SCALE, QA * QB)

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dg.bin"
    net_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/parity.nnue"
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    engine_path = sys.argv[4] if len(sys.argv) > 4 else "./blitz"

    raw = np.fromfile(data_path, dtype=np.uint8)
    n = raw.size // 32
    raw = raw[:n * 32].reshape(n, 32)
    rng = np.random.default_rng(1234)
    sel = raw[rng.choice(n, size=min(n_test, n), replace=False)]

    torch.manual_seed(7)
    net = Net()
    with torch.no_grad():
        net.ft.weight.uniform_(-0.4, 0.4)
        net.ft.weight[net.ft.padding_idx].zero_()
        net.ft_bias.uniform_(-0.3, 0.3)
        net.out_w.uniform_(-0.8, 0.8)
        net.out_b.uniform_(-0.2, 0.2)
    net.eval()
    export(net, net_path)

    iw, ib, stm, score, result, bucket = decode_batch(sel)
    with torch.no_grad():
        y = net(torch.as_tensor(iw), torch.as_tensor(ib),
                torch.as_tensor(stm), torch.as_tensor(bucket))
    ref_float = (y.numpy() * NET_SCALE)
    ref_quant = quantised_reference(net, iw, ib, stm, bucket)

    eng = Engine(engine_path, options={"EvalFile": net_path},
                 stderr_path="/tmp/parity.err")
    diffs = []
    got_all = []
    worst = None
    for k in range(len(sel)):
        board, _, _, _ = decode_sample(sel[k].tobytes())
        fen = board.fen()
        eng.send("position fen " + fen)
        eng.send("eval")
        line = eng.wait_for("NNUE:")
        got = int(line.split(":")[1].strip())
        got_all.append(got)
        d = abs(got - ref_quant[k])
        diffs.append(d)
        if worst is None or d > worst[0]:
            worst = (d, fen, got, int(ref_quant[k]))
    eng.quit()

    diffs = np.array(diffs)
    got_all = np.array(got_all, dtype=np.float64)
    qloss = np.abs(got_all - ref_float)

    print(f"\ncompared {len(diffs)} positions   engine {engine_path}   HL={HL}")
    print("  engine vs exact integer reference (implementation parity):")
    print(f"    mean |diff| : {diffs.mean():.3f} cp")
    print(f"    max  |diff| : {diffs.max():.3f} cp")
    print(f"    worst       : engine={worst[2]} int-ref={worst[3]}  {worst[1]}")
    print("  engine vs float PyTorch model (pure quantisation loss):")
    print(f"    mean |diff| : {qloss.mean():.2f} cp")
    print(f"    max  |diff| : {qloss.max():.2f} cp")

    ok = diffs.max() == 0
    print("\nPARITY OK (integer math matches exactly)" if ok
          else "\nPARITY FAILED -- the C++ integer path does not match the reference")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
