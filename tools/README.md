# Tools

| Script | Purpose |
| --- | --- |
| `run_tests.sh` | Runs the whole correctness suite (`make test`) |
| `uci_driver.py` | Minimal UCI client the other scripts share |
| `test_mates.py` | Forced mates, with ground truth computed by python-chess |
| `test_repetition.py` | Perpetual check, fifty-move rule, dead-drawn endings |
| `test_selfplay.py` | Full self-play games with every move validated |
| `test_timeman.py` | Games on a real clock; checks the engine never flags |
| `test_scaling.py` | Time-to-depth vs thread count |
| `eval_accuracy.py` | NNUE vs hand-crafted eval as predictors on held-out data |
| `calibrate.py` | Estimates absolute rating against Stockfish's limited ladder |
| `verify_data.py` | Decodes a datagen file and checks every sample |
| `test_nnue_parity.py` | C++ inference vs the PyTorch model, must match exactly |
| `train.py` | Trains and exports an NNUE network |
| `match.py` | Plays two configurations against each other and reports Elo |

All of them need `python-chess`; `train.py` also needs `torch` and `numpy`.

```
python3 -m pip install python-chess numpy torch
```
