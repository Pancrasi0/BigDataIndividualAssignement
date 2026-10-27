"""Basic triple-loop (i-j-k) matrix multiplication in pure Python (no NumPy).

Usage: python3 matrix.py <n> [dump]
Output (stdout): "<seconds> <checksum>"   (or the full result matrix with "dump")
Same deterministic inputs and checksum as matrix.c / main.java.
"""
import sys
import time


def multiply_matrices(a, b, n):
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def main():
    n = int(sys.argv[1])
    dump = len(sys.argv) > 2 and sys.argv[2] == "dump"
    a = [[(7 * i + 3 * j + 1) % 10 + 1 for j in range(n)] for i in range(n)]
    b = [[(5 * i + 11 * j + 2) % 10 + 1 for j in range(n)] for i in range(n)]

    t0 = time.perf_counter()
    c = multiply_matrices(a, b, n)
    t1 = time.perf_counter()

    if dump:
        for row in c:
            print(" ".join(map(str, row)))
    else:
        checksum = sum((i + 1) * (j + 1) * c[i][j] for i in range(n) for j in range(n))
        print(f"{t1 - t0:.6f} {checksum}")


if __name__ == "__main__":
    main()
