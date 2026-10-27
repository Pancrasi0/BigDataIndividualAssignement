/* Basic triple-loop (i-j-k) matrix multiplication in C.
 * Usage: ./matrix_c <n> [dump]
 * Output (stdout): "<seconds> <checksum>"   (or the full result matrix with "dump")
 * Inputs are deterministic so all languages multiply the SAME matrices:
 *   A[i][j] = (7*i + 3*j + 1) % 10 + 1
 *   B[i][j] = (5*i + 11*j + 2) % 10 + 1
 * checksum = sum_{i,j} (i+1)*(j+1)*C[i][j]   (64-bit integer)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int** allocate_matrix(int n) {
    int** m = (int**)malloc((size_t)n * sizeof(int*));
    for (int i = 0; i < n; i++) m[i] = (int*)malloc((size_t)n * sizeof(int));
    return m;
}

static void free_matrix(int** m, int n) {
    for (int i = 0; i < n; i++) free(m[i]);
    free(m);
}

static void fill_matrices(int** a, int** b, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i][j] = (7 * i + 3 * j + 1) % 10 + 1;
            b[i][j] = (5 * i + 11 * j + 2) % 10 + 1;
        }
}

static void multiply_matrices(int** a, int** b, int** c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
            c[i][j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <n> [dump]\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    int dump = (argc > 2 && strcmp(argv[2], "dump") == 0);

    int** a = allocate_matrix(n);
    int** b = allocate_matrix(n);
    int** c = allocate_matrix(n);
    fill_matrices(a, b, n);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    multiply_matrices(a, b, c, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    if (dump) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) printf(j ? " %d" : "%d", c[i][j]);
            printf("\n");
        }
    } else {
        long long checksum = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) checksum += (long long)(i + 1) * (j + 1) * c[i][j];
        printf("%.6f %lld\n", secs, checksum);
    }
    free_matrix(a, n); free_matrix(b, n); free_matrix(c, n);
    return 0;
}
