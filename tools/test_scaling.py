"""Measure how the search scales with threads.

Reports time-to-depth, which is the standard proxy for SMP gain: if doubling the
threads does not meaningfully shorten the time to reach a given depth, the extra
cores are being wasted no matter how impressive the raw node rate looks.
"""
import statistics
import sys
import time

from uci_driver import Engine

POSITIONS = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
]


def time_to_depth(engine_path, threads, depth, hash_mb):
    eng = Engine(engine_path, {"Threads": threads, "Hash": hash_mb, "EvalFile": "none"},
                 stderr_path="/tmp/scale.err")
    times, nodes = [], []
    for fen in POSITIONS:
        eng.newgame()
        eng.send("position fen " + fen)
        t0 = time.time()
        eng.send(f"go depth {depth}")
        lines = eng.wait_for("bestmove", collect=True)
        times.append(time.time() - t0)
        last = [l for l in lines if l.startswith("info depth") and " nodes " in l]
        if last:
            nodes.append(int(last[-1].split(" nodes ")[1].split()[0]))
    eng.quit()
    return sum(times), sum(nodes)


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "./blitz"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    thread_counts = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 4, 8]
    hash_mb = 256

    print(f"engine {engine}   depth {depth}   {len(POSITIONS)} positions   hash {hash_mb} MB\n")
    print(f"  {'threads':>7} {'time(s)':>9} {'speedup':>8} {'efficiency':>11} "
          f"{'nodes':>12} {'nps':>10}")
    base = None
    for t in thread_counts:
        secs, nodes = time_to_depth(engine, t, depth, hash_mb)
        if base is None:
            base = secs
        speedup = base / secs
        print(f"  {t:>7} {secs:>9.2f} {speedup:>8.2f}x {100*speedup/t:>10.0f}% "
              f"{nodes:>12} {int(nodes/secs):>10}")

    print("\n  speedup is time-to-depth relative to 1 thread; efficiency is speedup/threads.")
    print("  Lazy SMP typically reaches 1.6-1.8x at 2 threads and 3-4x at 8.")


if __name__ == "__main__":
    sys.exit(main())
