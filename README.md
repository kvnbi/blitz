# blitz

A UCI chess engine written from scratch in C++.

## Strength

```
one core      ~2400   CCRL scale, estimated
one core      ~2450   Stockfish UCI_Elo scale
eight cores   ~2915   Stockfish UCI_Elo scale
```

All measured at 200ms a move. The two scales differ by roughly 500 points. The CCRL number is the one to use when comparing against other engines.

The CCRL figure is estimated. The engine plays level with Stockfish capped at about 600 nodes per move, and that gap works out to roughly 2400 once Stockfish's strength per node is accounted for.

## What each change was worth

```
+407   going from one core to eight
 +93   killer moves
 +63   fixing the evaluation scale
 -89   second network, rejected
-221   first network, rejected
```

## Hardware and conditions

```
4.0M      nodes per second, one core, hand written evaluation
215 KB    binary, no dependencies
1.5 MB    network file, when one is loaded
64 MB     default hash
```

## Build and run

```bash
xcode-select --install     # macOS only, gets you a compiler
make                       # builds ./blitz
```

Start it with `./blitz`.

```
position startpos                        engine plays white
position startpos moves e2e4             engine plays black
position startpos moves e2e4 e7e5 g1f3   later in the game
go movetime 3000
```

It thinks for three seconds and prints a line starting with `bestmove`. Resend the whole game every turn. Moves are from square then to square like `e2e4`, castling is the king's own move, so `e1g1` kingside and `e1c1` queenside, promotion adds the piece like `e7e8q`. Type `quit` to leave.
