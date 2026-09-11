import os
import subprocess
import sys
import tempfile

class Engine:
    def __init__(self, path="./blitz", options=None, stderr_path=None):
        if sys.platform == "win32" and not path.endswith(".exe"):
            path += ".exe"
        if stderr_path:
            stderr_path = os.path.join(tempfile.gettempdir(), os.path.basename(stderr_path))
        self.err = open(stderr_path, "w") if stderr_path else None
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err, text=True, bufsize=1)
        self.send("uci")
        self.wait_for("uciok")
        for k, v in (options or {}).items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait_for(self, tok, collect=False):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"engine died (rc={self.p.poll()})")
            line = line.strip()
            if collect:
                lines.append(line)
            if line.startswith(tok):
                return lines if collect else line

    def go(self, fen, **limits):
        self.send("position fen " + fen)
        args = " ".join(f"{k} {v}" for k, v in limits.items())
        self.send("go " + args)
        lines = self.wait_for("bestmove", collect=True)
        best = [l for l in lines if l.startswith("bestmove")][-1].split()[1]
        infos = [l for l in lines if l.startswith("info depth")]
        return best, infos

    def newgame(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok")

    def quit(self):
        self.send("quit")
        self.p.wait(timeout=10)
