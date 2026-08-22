"""Play the engine against itself and validate every move with python-chess.

Catches illegal-move bugs, crashes, hangs, and time-management failures that a
fixed-depth test would never reach.
"""
import sys
import time
import chess
from uci_driver import Engine


def play_game(white, black, movetime=60, max_plies=300, start_fen=None, chess960=False):
    board = chess.Board(start_fen or chess.STARTING_FEN, chess960=chess960)
    moves = []
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            return board, moves, board.result(claim_draw=True)

        eng = white if board.turn == chess.WHITE else black
        eng.send("position fen " + (start_fen or chess.STARTING_FEN)
                 + (" moves " + " ".join(moves) if moves else ""))
        eng.send(f"go movetime {movetime}")
        lines = eng.wait_for("bestmove", collect=True)
        best = lines[-1].split()[1]

        try:
            mv = chess.Move.from_uci(best)
        except ValueError:
            return board, moves, f"ILLEGAL-UCI:{best}"
        if mv not in board.legal_moves:
            # Chess960 castling may come back as king-takes-rook.
            if not (chess960 and board.is_castling(mv)):
                return board, moves, f"ILLEGAL:{best}"
        board.push(mv)
        moves.append(best)
    return board, moves, "max-plies"


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    threads = sys.argv[2] if len(sys.argv) > 2 else "1"
    opts = {"Hash": 32, "Threads": threads}
    white = Engine(options=opts, stderr_path="/tmp/blitz_white.err")
    black = Engine(options=opts, stderr_path="/tmp/blitz_black.err")

    failures = 0
    for g in range(games):
        white.newgame()
        black.newgame()
        t0 = time.time()
        board, moves, result = play_game(white, black)
        bad = result.startswith("ILLEGAL")
        failures += bad
        print(f"game {g + 1}: {len(moves)} plies, result={result}, "
              f"{time.time() - t0:.1f}s{'  <-- FAILURE' if bad else ''}")
        if bad:
            print("   final fen:", board.fen())
            print("   moves:", " ".join(moves))

    white.quit()
    black.quit()
    print(f"\n{games - failures}/{games} games completed with only legal moves")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
