import argparse
import sys

from match import elo, openings, play
from uci_driver import Engine

def run(engine, opts, movetime, sf_nodes, pairs, seed):
    a = Engine(engine, opts, stderr_path="/tmp/to_a.err")
    b = Engine("stockfish", {"Threads": 1, "Hash": 64}, stderr_path="/tmp/to_b.err")
    limits = {id(a): f"movetime {movetime}", id(b): f"nodes {sf_nodes}"}
    score = 0.0
    w = d = l = 0
    for i, fen in enumerate(openings(pairs, seed=seed)):
        for swap in (False, True):
            res = play(b if swap else a, a if swap else b, fen, movetime, limits=limits)
            sa = (1.0 - res) if swap else res
            score += sa
            w += sa == 1.0
            d += sa == 0.5
            l += sa == 0.0
        print(f"\r  sf nodes {sf_nodes:>8}: {(i+1)*2:3d} games  +{w} ={d} -{l}  "
              f"({score/((i+1)*2):.3f})   ", end="", flush=True)
    print()
    a.quit()
    b.quit()
    n = pairs * 2
    diff, width = elo(score, n)
    return score / n, diff, width / 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./blitz")
    ap.add_argument("--opts", default="EvalFile=none,Hash=64,Threads=1")
    ap.add_argument("--movetime", type=int, default=200)
    ap.add_argument("--nodes", default="2000,8000,32000")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    opts = {}
    for part in args.opts.split(","):
        k, _, v = part.partition("=")
        opts[k.strip()] = v.strip()

    print(f"{args.engine} at {args.movetime}ms vs Stockfish at fixed nodes, both 1 thread\n")
    rows = []
    for n in [int(x) for x in args.nodes.split(",")]:
        rate, diff, err = run(args.engine, opts, args.movetime, n, args.pairs, args.seed + n)
        rows.append((n, rate, diff, err))
        print(f"    score {rate:.3f}  Elo diff {diff:+.0f} +/- {err:.0f}\n")

    print("summary")
    print(f"  {'sf nodes':>9} {'score':>7} {'diff':>8}")
    for n, rate, diff, err in rows:
        print(f"  {n:>9} {rate:>7.3f} {diff:>+8.0f}")

if __name__ == "__main__":
    sys.exit(main())
