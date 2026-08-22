# blitz

A UCI chess engine written from scratch in C++.

## Elo over time

```
2450   one thread
2890   eight threads
2915   after fixing the evaluation scale
```

These came from games against Stockfish 18 with its strength limiter turned on. Each move got 200ms. The scale is Stockfish's own and does not line up with FIDE.

## What each change was worth

```
+407   going from 1 thread to 8
 +93   killer moves
 +63   fixing the evaluation scale
 -89   second network, rejected
-221   first network, rejected
```

## Build and run

```bash
xcode-select --install     # macOS only, gets you a compiler
make                       # builds ./blitz
```

Start it with `./blitz`. It prints no prompt and just waits for you to type.

To make it play white from the opening position:

```
position startpos
go movetime 3000
```

To make it play black, hand it your first move instead:

```
position startpos moves e2e4
go movetime 3000
```

Either way it thinks for three seconds and prints a line starting with `bestmove`. That is its move.

After that you resend the whole game every turn. Add its move and then yours:

```
position startpos moves e2e4 e7e5 g1f3
go movetime 3000
```

The engine keeps nothing between commands so the move list is the entire game. Moves are written as the square it came from then the square it goes to like `e2e4`. Castling is the king move `e1g1`. Promotion adds the new piece on the end like `e7e8q`.

Type `quit` to leave. For an actual game point a chess GUI at the binary and play on a board instead.

## Tests

```bash
python3 -m pip install python-chess numpy torch
make test
```

Counts 8.26 billion positions and checks the totals against python-chess. Also checks mates, draws, self play, the clock, and the network code.

There is no neural network in it. Two were trained but both played worse than the hand written evaluation.
