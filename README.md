# Basic matrix multiplication: C vs Java vs Python

Assignment 1 (Big Data) - Pablo Martín Perdomo, group 17, 42.

The naive triple-loop (i-j-k) `n x n` integer matrix multiplication written three times
(C, Java, pure Python), plus everything needed to measure it properly: repeated runs,
peak memory, correctness checks, plots, and a LaTeX report that fills itself from the data.

No BLAS / NumPy in the timed code. NumPy is only used to check results.

## What's in here

| File | What it does |
|---|---|
| `matrix.c` | C version (`int**`, compiled with `gcc -O3`) |
| `main.java` | Java version (`int[][]`, optional untimed warm-up runs) |
| `matrix.py` | Pure Python version (list of lists) |
| `memrun.c` | Tiny wrapper that reports peak RSS of a process (stand-in for `/usr/bin/time -f %M`) |
| `benchmark.py` | Runs everything: 4 configs (C, Java cold, Java warm-up, Python), sizes 50..1000, 5 reps, 60 s timeout |
| `test_correctness.py` | 33 exact checks against a NumPy reference (tolerance is zero, all ints) |
| `analyze.py` | Builds `summary.csv`, figures, LaTeX tables and the exponent fit |
| `report.tex` / `report.pdf` | The report. Pulls its numbers from the generated files |
| `results.csv`, `summary.csv`, `correctness.csv`, `env.txt`, `benchmark.log` | Raw data and environment of the run included in the report |

All three programs use the same deterministic inputs, so they multiply the exact same matrices:

```
A[i][j] = (7i + 3j + 1) % 10 + 1
B[i][j] = (5i + 11j + 2) % 10 + 1
```

Each program prints `<seconds> <checksum>` where checksum = sum of (i+1)(j+1)C[i][j].

## Running it

You need `gcc`, a JDK, Python 3 with `numpy` and `matplotlib`, and `pdflatex`.

```bash
pip install numpy matplotlib
python3 test_correctness.py                  # should say 33/33 checks passed
python3 benchmark.py --reps 5 --timeout 60   # ~8 min, writes results.csv + env.txt
python3 analyze.py                           # figures, tables, summary.csv
pdflatex report.tex && pdflatex report.tex
```

Useful flags: `--sizes 100 200 400`, `--reps 3`, `--java-warmup 5`, `--timeout 120`.

Quick manual run of a single program:

```bash
gcc -O3 matrix.c -o matrix_c && ./matrix_c 500
javac main.java && java main 500 3     # 3 untimed warm-up runs first
python3 matrix.py 200
```

## Measurement protocol (short version)

- Every measurement is a fresh process; repetitions are interleaved across implementations.
- Only the multiplication is timed (monotonic wall clock). Start-up and input generation are excluded.
- Memory metric = peak resident set size of the whole process (includes JVM / interpreter).
- A run over the timeout gets killed and that config skips bigger sizes.
- Every run's checksum is compared against a NumPy int64 reference.

## Headline results (from the run in the report)

- n=750: C 0.41 s, Java 0.68-0.72 s, Python 36.9 s (about 90x slower than C)
- n=1000: C 1.12 s, Java 1.73 s, Python timed out (>60 s)
- Fitted growth exponents (n >= 200): 2.9 to 3.5, so roughly cubic
- Peak memory is quadratic in n for all three, but the baseline is very different (C ~2 MiB, JVM ~42 MiB, Python ~9 MiB)
