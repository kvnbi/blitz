"""Draw detection: perpetual check, the fifty-move rule, and dead-drawn endings.

The perpetual position is verified here to be genuinely forced (Black has exactly
one legal reply to each check and the four-ply cycle returns to the start), so an
engine that scores it as anything other than a draw is wrong, not unlucky.
"""
import sys
import chess
from uci_driver import Engine

PERPETUAL = "5r1k/8/6Q1/8/8/7K/q7/8 w - - 0 1"
CYCLE = ["g6h6", "h8g8", "h6g6", "g8h8"]

def verify_forced(fen, cycle):
    board = chess.Board(fen)
    assert board.is_valid() and not board.is_check(), "bad test position"
    for i, uci in enumerate(cycle):
        legal = [m.uci() for m in board.legal_moves]
        if i % 2 == 1:
            assert legal == [uci], f"reply {i} is not forced: {legal}"
        assert uci in legal
        board.push_uci(uci)
    assert board.board_fen() == chess.Board(fen).board_fen() and board.turn == chess.WHITE
    return True

def score_of(infos):
    lines = [l for l in infos if " multipv 1 " in l and " score " in l]
    if not lines:
        return None, None
    parts = lines[-1].split(" score ")[1].split()
    return parts[0], int(parts[1])

def main():
    verify_forced(PERPETUAL, CYCLE)
    eng = Engine(options={"Hash": 32, "Threads": 1}, stderr_path="/tmp/rep.err")
    failures = 0

    eng.newgame()
    best, infos = eng.go(PERPETUAL, depth=18)
    kind, val = score_of(infos)
    print(f"forced perpetual   : best={best} score {kind} {val}")
    if not (kind == "cp" and abs(val) < 150):
        print("  FAIL: the forced perpetual should be scored as a draw")
        failures += 1
    if best not in ("g6h6", "g6g7"):
        print(f"  note: expected the checking move g6h6, got {best}")

    eng.newgame()
    fifty = "7k/8/8/8/8/8/6QK/8 w - - 98 120"
    _, infos = eng.go(fifty, depth=14)
    kind, val = score_of(infos)
    print(f"fifty-move rule    : score {kind} {val}")
    if not (kind == "cp" and abs(val) < 150):
        print("  FAIL: the fifty-move rule should force a draw score here")
        failures += 1

    eng.newgame()
    _, infos = eng.go("7k/8/8/8/8/8/6QK/8 w - - 0 1", depth=14)
    kind, val = score_of(infos)
    print(f"same, clock reset  : score {kind} {val}")
    if not ((kind == "cp" and val > 400) or kind == "mate"):
        print("  FAIL: queen up with a fresh clock should be clearly winning")
        failures += 1

    eng.newgame()
    _, infos = eng.go("8/8/8/4k3/8/4K3/8/8 w - - 0 1", depth=14)
    kind, val = score_of(infos)
    print(f"bare kings         : score {kind} {val}")
    if not (kind == "cp" and abs(val) < 50):
        print("  FAIL: K vs K should evaluate as a draw")
        failures += 1

    eng.quit()
    print(f"\n{'OK' if failures == 0 else str(failures) + ' problem(s)'}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
