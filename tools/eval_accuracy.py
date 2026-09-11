import sys

import numpy as np

sys.path.insert(0, "tools")
from uci_driver import Engine
from verify_data import decode as decode_sample

NET_SCALE = 400.0

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/all.bin"
    net_path = sys.argv[2] if len(sys.argv) > 2 else "blitz.nnue"
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    wdl = 0.4

    raw = np.fromfile(data_path, dtype=np.uint8)
    n = raw.size // 32
    raw = raw[:n * 32].reshape(n, 32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    holdout = raw[perm][: max(1, int(n * 0.02))]
    sel = holdout[np.random.default_rng(9).choice(len(holdout), n_test, replace=False)]

    eng = Engine(options={"EvalFile": net_path}, stderr_path="/tmp/acc.err")
    nnue_v, hce_v, targets, labels = [], [], [], []

    for rec in sel:
        board, score, result, _ = decode_sample(rec.tobytes())
        eng.send("position fen " + board.fen())
        eng.send("eval")
        line_n = eng.wait_for("NNUE:")
        line_h = eng.wait_for("HCE:")
        nnue_v.append(float(line_n.split(":")[1]))
        hce_v.append(float(line_h.split(":")[1]))
        labels.append(score)
        targets.append(wdl * (result / 2.0) + (1 - wdl) / (1 + np.exp(-score / NET_SCALE)))
    eng.quit()

    nnue_v = np.array(nnue_v)
    hce_v = np.array(hce_v)
    targets = np.array(targets)
    labels = np.array(labels)

    def report(name, v):
        pred = 1 / (1 + np.exp(-v / NET_SCALE))
        mse = float(((pred - targets) ** 2).mean())
        corr = float(np.corrcoef(v, labels)[0, 1])
        print(f"  {name:<6} mse {mse:.5f}   corr-with-search-score {corr:+.3f}   "
              f"mean {v.mean():+7.1f}  sd {v.std():6.1f}  "
              f"p1 {np.percentile(v,1):+.0f}  p99 {np.percentile(v,99):+.0f}")
        return mse

    print(f"\nheld-out positions: {len(sel)}   network: {net_path}")
    print(f"  {'label':<6} "
          f"mean {labels.mean():+7.1f}  sd {labels.std():6.1f}  "
          f"p1 {np.percentile(labels,1):+.0f}  p99 {np.percentile(labels,99):+.0f}")
    m_n = report("nnue", nnue_v)
    m_h = report("hce", hce_v)
    print(f"\n  -> {'NNUE' if m_n < m_h else 'HCE'} is the better predictor "
          f"({min(m_n, m_h):.5f} vs {max(m_n, m_h):.5f})")

if __name__ == "__main__":
    sys.exit(main())
