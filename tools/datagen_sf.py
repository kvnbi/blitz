import argparse
import os
import random
import struct
import sys
import time

import chess
import chess.engine

PIECE_CODE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

EVAL_SCALE = 2
SCORE_CAP = 4000 * EVAL_SCALE
ADJUDICATE = 2500 * EVAL_SCALE
ADJUDICATE_PLIES = 12

def encode(board, score_stm):
    occ = 0
    nibbles = bytearray(b"\xff" * 16)
    i = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        occ |= 1 << sq
        code = PIECE_CODE[(p.piece_type, p.color)]
        if i % 2 == 0:
            nibbles[i // 2] = (nibbles[i // 2] & 0xF0) | code
        else:
            nibbles[i // 2] = (nibbles[i // 2] & 0x0F) | (code << 4)
        i += 1
    s = max(-SCORE_CAP, min(SCORE_CAP, int(score_stm)))
    return struct.pack("<Q16shBB4x", occ, bytes(nibbles), s, 1,
                       0 if board.turn == chess.WHITE else 1)

def play_game(eng, rng, nodes, opening_plies):
    board = chess.Board()
    for _ in range(opening_plies):
        moves = list(board.legal_moves)
        if not moves:
            return []
        board.push(rng.choice(moves))
    if board.is_game_over():
        return []

    pending = []
    result = 1
    decided = 0
    limit = chess.engine.Limit(nodes=nodes)

    for _ in range(400):
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            result = {"1-0": 2, "0-1": 0}.get(r, 1)
            break

        info = eng.analyse(board, limit)
        pov = info["score"].pov(board.turn)
        if pov.is_mate():
            cp = SCORE_CAP if pov.mate() > 0 else -SCORE_CAP
        else:
            cp = pov.score() * EVAL_SCALE

        best = info.get("pv", [None])[0]
        if best is None:
            break

        stm_white = board.turn == chess.WHITE
        decided = decided + 1 if abs(cp) >= ADJUDICATE else 0

        quiet = not board.is_check() and not board.is_capture(best) and best.promotion is None
        if quiet:
            pending.append(encode(board, cp))

        if decided >= ADJUDICATE_PLIES:
            winning_white = (cp > 0) == stm_white
            result = 2 if winning_white else 0
            break

        board.push(best)

    out = []
    for rec in pending:
        stm = rec[27]
        r = result if stm == 0 else 2 - result
        out.append(rec[:26] + bytes([r]) + rec[27:])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("positions", type=int)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else os.getpid())
    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    eng.configure({"Threads": args.threads, "Hash": args.hash})

    written = 0
    games = 0
    t0 = time.time()
    with open(args.out, "ab") as f:
        while written < args.positions:
            recs = play_game(eng, rng, args.nodes, args.opening_plies)
            games += 1
            if not recs:
                continue
            f.write(b"".join(recs))
            f.flush()
            written += len(recs)
            if games % 10 == 0:
                dt = max(time.time() - t0, 1e-9)
                print(f"\r{written} positions, {games} games, {written/dt:.0f} pos/s   ",
                      end="", flush=True)
    eng.quit()
    print(f"\ndone: {written} positions to {args.out}")

if __name__ == "__main__":
    sys.exit(main())
