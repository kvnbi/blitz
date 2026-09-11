import sys
import time

import chess
from uci_driver import Engine

def play_clocked(white, black, base_ms, inc_ms, max_plies=300, overhead_ms=10):
    board = chess.Board()
    clock = {chess.WHITE: base_ms, chess.BLACK: base_ms}
    moves = []
    worst_margin = base_ms
    overuse = []

    for eng in (white, black):
        eng.newgame()

    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        side = board.turn
        eng = white if side == chess.WHITE else black
        eng.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        eng.send(f"go wtime {clock[chess.WHITE]} btime {clock[chess.BLACK]} "
                 f"winc {inc_ms} binc {inc_ms}")
        t0 = time.time()
        best = eng.wait_for("bestmove", collect=True)[-1].split()[1]
        used = int((time.time() - t0) * 1000)

        clock[side] -= used
        if clock[side] < 0:
            return board, moves, f"FLAG:{'white' if side == chess.WHITE else 'black'}", \
                   clock[side], overuse
        worst_margin = min(worst_margin, clock[side])
        clock[side] += inc_ms

        if used > 0.2 * (clock[side] + used) and used > 50:
            overuse.append((len(moves), used, clock[side] + used))

        mv = chess.Move.from_uci(best)
        if mv not in board.legal_moves:
            return board, moves, f"ILLEGAL:{best}", worst_margin, overuse
        board.push(mv)
        moves.append(best)

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "unfinished"
    return board, moves, result, worst_margin, overuse

def main():
    controls = [(10000, 100), (5000, 0), (3000, 30)]
    opts = {"Hash": 32, "Threads": 1, "EvalFile": "none"}
    white = Engine(options=opts, stderr_path="/tmp/tm_w.err")
    black = Engine(options=opts, stderr_path="/tmp/tm_b.err")

    failures = 0
    for base, inc in controls:
        board, moves, result, margin, overuse = play_clocked(white, black, base, inc)
        flagged = result.startswith("FLAG") or result.startswith("ILLEGAL")
        failures += flagged
        print(f"  {base/1000:.0f}s+{inc/1000:.2f}s : {len(moves):3d} plies, result={result}, "
              f"tightest margin {margin} ms, {len(overuse)} reckless moves"
              f"{'   <-- FAILURE' if flagged else ''}")
        for ply, used, had in overuse[:3]:
            print(f"      ply {ply}: used {used} ms of {had} ms remaining")

    white.quit()
    black.quit()
    print(f"\n{'OK - no flags' if failures == 0 else str(failures) + ' game(s) lost on time'}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
