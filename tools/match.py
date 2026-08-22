"""Play a match between two engine configurations and report the Elo difference.

Used to decide whether a change is actually an improvement.  Games are played in
pairs from the same opening with colours swapped, which removes most of the
variance caused by the opening itself.
"""
import argparse
import math
import random
import sys

import chess
from uci_driver import Engine


def openings(count, plies=8, seed=0):
    """Random but legal and roughly balanced opening positions."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        board = chess.Board()
        ok = True
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
        if ok and not board.is_game_over():
            out.append(board.fen())
    return out


def play(white, black, fen, movetime, max_plies=300):
    board = chess.Board(fen)
    moves = []
    for eng in (white, black):
        eng.newgame()
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        eng = white if board.turn == chess.WHITE else black
        eng.send("position fen " + fen + (" moves " + " ".join(moves) if moves else ""))
        eng.send(f"go movetime {movetime}")
        best = eng.wait_for("bestmove", collect=True)[-1].split()[1]
        mv = chess.Move.from_uci(best)
        if mv not in board.legal_moves:
            raise RuntimeError(f"illegal move {best} in {board.fen()}")
        board.push(mv)
        moves.append(best)
    if board.is_game_over(claim_draw=True):
        r = board.result(claim_draw=True)
        return {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5)
    return 0.5


def elo(score, n):
    """Elo difference and a 95% confidence interval, from the score rate."""
    if n == 0:
        return 0.0, 0.0
    p = min(max(score / n, 1e-6), 1 - 1e-6)
    e = -400 * math.log10(1 / p - 1)
    # Standard error of the mean score, propagated through the logistic.
    var = max(score / n - (score / n) ** 2, 1e-9)
    se = math.sqrt(var / n)
    margin = 1.96 * se
    lo = min(max(p - margin, 1e-6), 1 - 1e-6)
    hi = min(max(p + margin, 1e-6), 1 - 1e-6)
    return e, (-400 * math.log10(1 / hi - 1)) - (-400 * math.log10(1 / lo - 1))


def parse_opts(text):
    out = {}
    if text:
        for part in text.split(","):
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", default="./blitz")
    ap.add_argument("--engine-b", default="./blitz")
    ap.add_argument("--opts-a", default="")
    ap.add_argument("--opts-b", default="")
    ap.add_argument("--pairs", type=int, default=50, help="game pairs (2 games each)")
    ap.add_argument("--movetime", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = Engine(args.engine_a, parse_opts(args.opts_a), stderr_path="/tmp/match_a.err")
    b = Engine(args.engine_b, parse_opts(args.opts_b), stderr_path="/tmp/match_b.err")

    fens = openings(args.pairs, seed=args.seed)
    score_a = 0.0
    w = d = l = 0
    for i, fen in enumerate(fens):
        for swap in (False, True):
            res = play(b if swap else a, a if swap else b, fen, args.movetime)
            sa = (1.0 - res) if swap else res
            score_a += sa
            if sa == 1.0:
                w += 1
            elif sa == 0.5:
                d += 1
            else:
                l += 1
        n = (i + 1) * 2
        e, err = elo(score_a, n)
        print(f"\r{n:4d} games  A: +{w} ={d} -{l}  ({score_a / n:.3f})  "
              f"Elo {e:+.0f} +/- {err / 2:.0f}   ", end="", flush=True)

    print()
    a.quit()
    b.quit()
    n = len(fens) * 2
    e, err = elo(score_a, n)
    print(f"\nA = {args.engine_a} [{args.opts_a or 'defaults'}]")
    print(f"B = {args.engine_b} [{args.opts_b or 'defaults'}]")
    print(f"result: A scored {score_a}/{n}  (+{w} ={d} -{l})")
    print(f"Elo(A - B) = {e:+.0f} +/- {err / 2:.0f}  (95% CI)")


if __name__ == "__main__":
    sys.exit(main())
