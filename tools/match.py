"""Play a match between two engine configurations and report the Elo difference.

Used to decide whether a change is actually an improvement. Games are played in
pairs from the same opening with colours swapped, which removes most of the
variance caused by the opening itself. Pairs run several at a time, and with
SPRT the match stops as soon as the result is decisive.
"""
import argparse
import atexit
import math
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def play(white, black, fen, movetime, max_plies=300, limits=None):
    board = chess.Board(fen)
    moves = []
    for eng in (white, black):
        eng.newgame()
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        eng = white if board.turn == chess.WHITE else black
        eng.send("position fen " + fen + (" moves " + " ".join(moves) if moves else ""))
        limit = (limits or {}).get(id(eng)) or f"movetime {movetime}"
        eng.send("go " + limit)
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
    var = max(score / n - (score / n) ** 2, 1e-9)
    se = math.sqrt(var / n)
    margin = 1.96 * se
    lo = min(max(p - margin, 1e-6), 1 - 1e-6)
    hi = min(max(p + margin, 1e-6), 1 - 1e-6)
    return e, (-400 * math.log10(1 / hi - 1)) - (-400 * math.log10(1 / lo - 1))

def elo_to_score(e):
    return 1.0 / (1.0 + 10 ** (-e / 400.0))

def sprt_llr(pair_scores, elo0, elo1):
    """Log likelihood ratio over paired game results.

    Each pair contributes a score from 0 to 2. Modelling pairs rather than
    single games is what keeps the test honest, because the two games in a
    pair share an opening and are not independent.

    The variance is floored so that a run of identical results cannot report
    infinite certainty from a handful of pairs.
    """
    n = len(pair_scores)
    if n < 2:
        return 0.0
    mu = sum(pair_scores) / n
    var = sum((p - mu) ** 2 for p in pair_scores) / n
    var = max(var, 0.01)
    mu0 = 2.0 * elo_to_score(elo0)
    mu1 = 2.0 * elo_to_score(elo1)
    return n * (mu1 - mu0) * (mu - (mu0 + mu1) / 2.0) / var

def sprt_bounds(alpha, beta):
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)

def parse_opts(text):
    out = {}
    if text:
        for part in text.split(","):
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out

_A = None
_B = None

def _worker_init(engine_a, opts_a, engine_b, opts_b, tmpdir):
    global _A, _B
    pid = os.getpid()
    _A = Engine(engine_a, opts_a, stderr_path=os.path.join(tmpdir, f"a{pid}.err"))
    _B = Engine(engine_b, opts_b, stderr_path=os.path.join(tmpdir, f"b{pid}.err"))
    atexit.register(_worker_quit)

def _worker_quit():
    for eng in (_A, _B):
        try:
            if eng is not None:
                eng.quit()
        except Exception:
            pass

def _play_pair(task):
    fen, limit = task
    out = []
    for swap in (False, True):
        white = _B if swap else _A
        black = _A if swap else _B
        limits = {id(white): limit, id(black): limit}
        r = play(white, black, fen, 0, limits=limits)
        out.append((1.0 - r) if swap else r)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", default="./blitz")
    ap.add_argument("--engine-b", default="./blitz")
    ap.add_argument("--opts-a", default="")
    ap.add_argument("--opts-b", default="")
    ap.add_argument("--pairs", type=int, default=50, help="game pairs (2 games each)")
    ap.add_argument("--movetime", type=int, default=100)
    ap.add_argument("--nodes", type=int, default=0,
                    help="fixed nodes per move instead of movetime, makes a match reproducible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="pairs run at once, 0 picks a safe default from core count")
    ap.add_argument("--sprt", action="store_true", help="stop as soon as the result is decisive")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--min-pairs", type=int, default=10,
                    help="pairs to play before SPRT may stop the match")
    ap.add_argument("--tmpdir", default="/tmp")
    args = ap.parse_args()

    conc = args.concurrency or max(1, (os.cpu_count() or 4) // 2 - 1)
    fens = openings(args.pairs, seed=args.seed)
    limit = f"nodes {args.nodes}" if args.nodes else f"movetime {args.movetime}"
    tasks = [(fen, limit) for fen in fens]
    lower, upper = sprt_bounds(args.alpha, args.beta)

    pool = mp.Pool(conc, initializer=_worker_init,
                   initargs=(args.engine_a, parse_opts(args.opts_a),
                             args.engine_b, parse_opts(args.opts_b), args.tmpdir))

    pair_scores = []
    score_a = 0.0
    w = d = l = 0
    verdict = None
    llr = 0.0

    try:
        for start in range(0, len(tasks), conc):
            batch = tasks[start:start + conc]
            for res in pool.map(_play_pair, batch):
                for sa in res:
                    score_a += sa
                    if sa == 1.0:
                        w += 1
                    elif sa == 0.5:
                        d += 1
                    else:
                        l += 1
                pair_scores.append(sum(res))

            n = len(pair_scores) * 2
            e, err = elo(score_a, n)
            if args.sprt:
                llr = sprt_llr(pair_scores, args.elo0, args.elo1)
            tail = f"  LLR {llr:+.2f}" if args.sprt else ""
            print(f"\r{n:5d} games  A: +{w} ={d} -{l}  ({score_a / n:.3f})  "
                  f"Elo {e:+.0f} +/- {err / 2:.0f}{tail}   ", end="", flush=True)

            if args.sprt and len(pair_scores) >= args.min_pairs:
                if llr >= upper:
                    verdict = f"H1 accepted, gain is at least {args.elo1:.0f} Elo"
                    break
                if llr <= lower:
                    verdict = f"H0 accepted, gain is below {args.elo1:.0f} Elo"
                    break
    finally:
        pool.close()
        pool.join()

    print()
    n = len(pair_scores) * 2
    e, err = elo(score_a, n)
    print(f"\nA = {args.engine_a} [{args.opts_a or 'defaults'}]")
    print(f"B = {args.engine_b} [{args.opts_b or 'defaults'}]")
    print(f"limit: {limit}, concurrency: {conc} pairs at once")
    print(f"result: A scored {score_a}/{n}  (+{w} ={d} -{l})")
    print(f"Elo(A - B) = {e:+.0f} +/- {err / 2:.0f}  (95% CI)")
    if args.sprt:
        print(f"LLR {llr:+.2f}  bounds [{lower:+.2f}, {upper:+.2f}]")
        print(f"verdict: {verdict or 'inconclusive, ran out of pairs'}")

if __name__ == "__main__":
    sys.exit(main())
