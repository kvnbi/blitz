#!/usr/bin/env bash
# Full correctness suite.  Everything here is checked against an independent
# reference (python-chess, or a numpy replay of the engine's own integer maths)
# rather than against values the engine could influence.
set -u
cd "$(dirname "$0")/.."

TMP=${TMPDIR:-/tmp}/blitz-tests
mkdir -p "$TMP"

make -s blitz perft || { echo "build failed"; exit 1; }

fail=0
run() {
    echo
    echo "=============================================================="
    echo "== $1"
    echo "=============================================================="
    shift
    if "$@"; then echo "-> PASS"; else echo "-> FAIL"; fail=1; fi
}

run "perft (movegen + do_move/undo_move)"  ./perft
run "forced mates"                         python3 tools/test_mates.py
run "draw detection"                       python3 tools/test_repetition.py
run "self-play, 1 thread"                  python3 tools/test_selfplay.py 3 1
run "self-play, 8 threads"                 python3 tools/test_selfplay.py 3 8
run "time management (real clock)"         python3 tools/test_timeman.py

# NNUE tests need a data file and a network; both are cheap to make from scratch.
echo
echo "== generating a small sample set for the NNUE tests"
rm -f "$TMP/sample.bin"
./blitz datagen "$TMP/sample.bin" 40000 2000 >/dev/null 2>&1

run "training data encoding"               python3 tools/verify_data.py "$TMP/sample.bin"
run "NNUE C++/PyTorch parity"              python3 tools/test_nnue_parity.py \
                                               "$TMP/sample.bin" "$TMP/parity.nnue" 200
run "NNUE incremental accumulator" \
    bash -c "printf 'setoption name EvalFile value $TMP/parity.nnue\nnnuecheck 2000\nquit\n' \
             | ./blitz | tee /dev/stderr | grep -q '0 mismatches'"

echo
if [ $fail -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
