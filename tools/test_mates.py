"""Verify the engine finds forced mates.

Ground truth is computed independently with python-chess by exhaustive search,
so this test does not depend on any remembered "expected" values.
"""
import sys
import chess
from uci_driver import Engine


def mate_in(board, plies):
    """Return the shortest forced mate for the side to move, in plies, or None."""
    for d in range(1, plies + 1):
        if forced_mate(board, d):
            return d
    return None


def forced_mate(board, plies):
    if plies <= 0:
        return False
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            board.pop()
            return True
        ok = plies >= 3 and all_replies_lose(board, plies - 1)
        board.pop()
        if ok:
            return True
    return False


def all_replies_lose(board, plies):
    if not any(board.legal_moves):
        return False
    for m in board.legal_moves:
        board.push(m)
        ok = forced_mate(board, plies - 1)
        board.pop()
        if not ok:
            return False
    return True


POSITIONS = [
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1",
    "3r1r1k/1p3pbp/p3n1p1/8/1q6/1P1Q1N1P/P4PP1/2R1R1K1 w - - 0 1",
    "6k1/p4ppp/1p6/8/8/1P3q2/P4PPP/3R2K1 b - - 0 1",
    "2bqkbn1/2pppp2/np2N3/r3P1p1/p2N2B1/5Q2/PPPPKPP1/RNB2r2 w - - 0 1",
    "8/8/8/8/8/6K1/6P1/6k1 w - - 0 1",
    "1r3rk1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1",
    "r1b2k1r/ppp1bppp/8/1B1Q4/5q2/2P5/PPP2PPP/R3R1K1 w - - 1 0",
    "5rk1/1p1q2bp/p2p2p1/2pP4/2P1n3/1P2NN1P/P4PP1/3QR1K1 b - - 0 1",
    "r1bqk2r/pppp1Npp/2n2n2/2b5/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 1",
]

eng = Engine(options={"Hash": 64, "Threads": 1})
tested = passed = 0
for fen in POSITIONS:
    board = chess.Board(fen)
    truth = mate_in(board, 5)          # forced mate within 5 plies (mate in 1, 2 or 3)
    if truth is None:
        continue
    tested += 1
    best, infos = eng.go(fen, depth=14)
    # Last reported score for the main pv
    score = None
    for line in infos:
        if " multipv 1 " in line and " score mate " in line:
            score = int(line.split(" score mate ")[1].split()[0])
    expected_moves = (truth + 1) // 2
    ok = score is not None and score == expected_moves
    passed += ok
    print(f"{'ok  ' if ok else 'FAIL'}  mate in {expected_moves}  engine={score}  best={best}  {fen}")

eng.quit()
print(f"\n{passed}/{tested} forced mates found correctly")
sys.exit(0 if passed == tested else 1)
