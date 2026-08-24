"""Decode a datagen file and check every sample is a legal, quiet position.

Encoding bugs here are silent poison: the trainer would happily learn from
garbage, so this is checked independently with python-chess.
"""
import struct
import sys

import chess
import numpy as np

PIECES = "PNBRQKpnbrqk"

def decode(rec):
    occ, = struct.unpack_from("<Q", rec, 0)
    nibbles = rec[8:24]
    score, result, stm = struct.unpack_from("<hBB", rec, 24)

    board = chess.Board.empty()
    i = 0
    b = occ
    while b:
        sq = (b & -b).bit_length() - 1
        b &= b - 1
        code = (nibbles[i // 2] & 0xF) if i % 2 == 0 else (nibbles[i // 2] >> 4)
        if code > 11:
            raise ValueError(f"bad piece nibble {code}")
        board.set_piece_at(sq, chess.Piece.from_symbol(PIECES[code]))
        i += 1
    board.turn = chess.WHITE if stm == 0 else chess.BLACK
    return board, score, result, i

def main(path, limit=200000):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n = len(data) // 32
    checked = bad = 0
    results = {0: 0, 1: 0, 2: 0}
    scores = []
    pieces = []
    rng = np.random.default_rng(0)
    idx = (rng.choice(n, size=min(n, limit), replace=False) if n > limit
           else np.arange(n))
    for k in idx:
        rec = bytes(data[k * 32:(k + 1) * 32])
        try:
            board, score, result, pcount = decode(rec)
        except Exception as e:
            print(f"record {k}: decode failed: {e}")
            bad += 1
            continue
        checked += 1
        results[result] = results.get(result, 0) + 1
        scores.append(score)
        pieces.append(pcount)

        problems = []
        if not board.is_valid():
            problems.append(f"invalid board ({board.status()!r})")
        if board.is_check():
            problems.append("position is in check (should have been filtered)")
        if pcount != bin(struct.unpack_from('<Q', rec, 0)[0]).count("1"):
            problems.append("piece count does not match occupancy")
        if result not in (0, 1, 2):
            problems.append(f"bad result {result}")
        if problems:
            bad += 1
            if bad <= 5:
                print(f"record {k}: {'; '.join(problems)}\n   {board.fen()}")

    print(f"\nchecked {checked} random samples from {path} ({n} total)")
    print(f"  invalid: {bad}")
    print(f"  results: loss={results[0]} draw={results[1]} win={results[2]}")
    if scores:
        sc = np.abs(np.array(scores))
        pc = np.array(pieces)
        srt = sorted(scores)
        print(f"  score  : min={srt[0]} p50={srt[len(srt)//2]} max={srt[-1]}")
        print(f"  |score|>2500: {100*(sc>2500).mean():.2f}%   "
              f"won endgames (<=8 pieces & |score|>=2000): {100*((pc<=8)&(sc>=2000)).mean():.3f}%")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dg.bin"))
