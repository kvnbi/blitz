"""Estimate Blitz's absolute rating by playing Stockfish's limited-strength ladder.

Stockfish's UCI_LimitStrength/UCI_Elo gives opponents of approximately known
rating, so a match against a few rungs pins Blitz down on the same scale.  The
scale is Stockfish's own calibration and is approximate -- treat the result as a
band, not a precise number.
"""
import argparse
import math
import sys

from match import elo, openings, play
from uci_driver import Engine


def run_level(engine_a, opts_a, sf_elo, pairs, movetime, seed):
    a = Engine(engine_a, opts_a, stderr_path="/tmp/cal_a.err")
    b = Engine("stockfish",
               {"Threads": 1, "Hash": 32,
                "UCI_LimitStrength": "true", "UCI_Elo": sf_elo},
               stderr_path="/tmp/cal_b.err")
    score = 0.0
    w = d = l = 0
    for i, fen in enumerate(openings(pairs, seed=seed)):
        for swap in (False, True):
            res = play(b if swap else a, a if swap else b, fen, movetime)
            sa = (1.0 - res) if swap else res
            score += sa
            w += sa == 1.0
            d += sa == 0.5
            l += sa == 0.0
        print(f"\r  vs SF {sf_elo}: {(i+1)*2:3d} games  +{w} ={d} -{l}  "
              f"({score/((i+1)*2):.3f})   ", end="", flush=True)
    print()
    a.quit()
    b.quit()
    n = pairs * 2
    diff, err = elo(score, n)
    return score / n, diff, err / 2, (w, d, l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./blitz")
    ap.add_argument("--opts", default="EvalFile=none,Hash=32,Threads=1")
    ap.add_argument("--levels", default="2000,2400,2800")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--movetime", type=int, default=100)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    opts = {}
    for part in args.opts.split(","):
        k, _, v = part.partition("=")
        opts[k.strip()] = v.strip()

    print(f"calibrating {args.engine} [{args.opts}] at {args.movetime} ms/move\n")
    estimates = []
    for lvl in [int(x) for x in args.levels.split(",")]:
        rate, diff, err, wdl = run_level(args.engine, opts, lvl, args.pairs,
                                         args.movetime, args.seed + lvl)
        est = lvl + diff
        estimates.append((lvl, rate, diff, err, est))
        print(f"  -> score {rate:.3f}, Elo diff {diff:+.0f} +/- {err:.0f}, "
              f"implies Blitz ~{est:.0f}\n")

    print("summary")
    print(f"  {'SF level':>9} {'score':>7} {'diff':>8} {'implied':>9}")
    for lvl, rate, diff, err, est in estimates:
        print(f"  {lvl:>9} {rate:>7.3f} {diff:>+8.0f} {est:>9.0f}")

    # Weight each rung by how informative it is: scores near 0.5 pin the rating
    # down best, lopsided ones barely constrain it at all.
    usable = [(e, 1.0 / (1.0 + abs(r - 0.5) * 10)) for _, r, _, _, e in estimates]
    total = sum(w for _, w in usable)
    if total:
        print(f"\n  weighted estimate: ~{sum(e*w for e, w in usable)/total:.0f} Elo "
              f"(Stockfish's UCI_Elo scale, approximate)")


if __name__ == "__main__":
    sys.exit(main())
