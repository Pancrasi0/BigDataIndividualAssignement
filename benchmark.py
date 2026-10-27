#!/usr/bin/env python3
"""Benchmark driver: basic triple-loop matrix multiplication in C, Java and Python.

Protocol (fixed for all implementations)
  * identical deterministic inputs, same i-j-k loop order, 32-bit-range integers
  * every measurement is a FRESH OS process (no state shared between repetitions)
  * the time reported by the program covers ONLY the multiplication
    (monotonic wall clock: clock_gettime / System.nanoTime / time.perf_counter)
  * peak memory = peak RSS of the whole process, measured with ./memrun (wait4 ru_maxrss)
  * repetitions are interleaved (size -> repetition -> implementation) to spread drift
  * a run exceeding --timeout seconds is killed and recorded as "timeout"; larger sizes
    of that implementation are then skipped
  * every result checksum is compared with a NumPy int64 reference (verification only,
    NumPy is never used inside the timed code)

Usage:  python3 benchmark.py [--reps 5] [--timeout 60] [--sizes 50 100 ...] [--java-warmup 3]
Output: results.csv, env.txt
"""
import argparse
import csv
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIZES = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def reference_checksum(n):
    i = np.arange(n, dtype=np.int64)[:, None]
    j = np.arange(n, dtype=np.int64)[None, :]
    a = (7 * i + 3 * j + 1) % 10 + 1
    b = (5 * i + 11 * j + 2) % 10 + 1
    c = a @ b
    w = (i + 1) * (j + 1)
    return int((w * c).sum())


def compile_all():
    steps = [
        ["gcc", "-O3", "matrix.c", "-o", "matrix_c"],
        ["gcc", "-O2", "memrun.c", "-o", "memrun"],
        ["javac", "main.java"],
    ]
    for cmd in steps:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"Compilation failed: {' '.join(cmd)}\n{r.stderr}")


def run_once(cmd, timeout):
    """Run cmd through memrun. Returns (status, seconds, checksum, maxrss_kb, wall_total)."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        t0 = time.perf_counter()
        p = subprocess.Popen([os.path.join(HERE, "memrun")] + cmd, cwd=HERE,
                             stdout=out, stderr=err, start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return "timeout", None, None, None, time.perf_counter() - t0
        wall = time.perf_counter() - t0
        out.seek(0); err.seek(0)
        stdout, stderr = out.read(), err.read()
    m = re.search(r"MAXRSS_KB=(\d+)", stderr)
    parts = stdout.split()
    if p.returncode != 0 or len(parts) != 2 or not m:
        return "error", None, None, None, wall
    return "ok", float(parts[0]), int(parts[1]), int(m.group(1)), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--java-warmup", type=int, default=3)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    print("Compiling...", flush=True)
    compile_all()

    impls = {
        "C":           ["./matrix_c", "{n}"],
        "Java-cold":   ["java", "main", "{n}", "0"],
        "Java-warm":   ["java", "main", "{n}", str(args.java_warmup)],
        "Python":      [sys.executable, "matrix.py", "{n}"],
    }

    with open(os.path.join(HERE, "env.txt"), "w") as f:
        f.write("date: " + sh("date -u +%FT%TZ") + "\n")
        f.write("platform: " + platform.platform() + "\n")
        f.write("cpu: " + sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|MHz|L1d|L2|L3'") + "\n")
        f.write("mem: " + sh("grep MemTotal /proc/meminfo") + "\n")
        f.write("gcc: " + sh("gcc --version | head -1") + "\n")
        f.write("javac: " + sh("javac -version 2>&1 | grep -v 'Picked up'") + "\n")
        f.write("java: " + sh("java -version 2>&1 | grep -v 'Picked up'") + "\n")
        f.write("python: " + sys.version.replace("\n", " ") + " / " + platform.python_implementation() + "\n")
        f.write("numpy (verification only): " + np.__version__ + "\n")
        f.write(f"reps={args.reps} timeout={args.timeout}s sizes={args.sizes} java_warmup={args.java_warmup}\n")
        f.write("C flags: gcc -O3 | JVM flags: default (no -Xmx/-Xint/-XX) | Python: CPython, no extensions\n")

    dead = set()
    rows = []
    with open(os.path.join(HERE, args.out), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["impl", "n", "rep", "status", "seconds", "maxrss_kb", "wall_total_s", "checksum_ok"])
        for n in args.sizes:
            ref = reference_checksum(n)
            for rep in range(1, args.reps + 1):
                for name, tmpl in impls.items():
                    if name in dead:
                        continue
                    cmd = [x.replace("{n}", str(n)) for x in tmpl]
                    status, secs, chk, rss, wall = run_once(cmd, args.timeout)
                    ok = (chk == ref) if status == "ok" else ""
                    w.writerow([name, n, rep, status, secs if secs is not None else "",
                                rss if rss is not None else "", f"{wall:.3f}", ok])
                    fh.flush()
                    print(f"n={n:5d} rep={rep} {name:10s} {status:7s} "
                          f"t={secs if secs is None else format(secs, '.4f')} "
                          f"rss={rss} ok={ok}", flush=True)
                    if status != "ok":
                        dead.add(name)
                        print(f"   -> {name} marked as failed at n={n}; larger sizes skipped", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
