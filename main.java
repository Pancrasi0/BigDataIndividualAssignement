/* Basic triple-loop (i-j-k) matrix multiplication in Java.
 * Usage: java main <n> [warmup] [dump]
 *   warmup = number of UNTIMED multiplications of the same size run before the timed one
 *            (0 = cold JVM: JIT compilation is included in the measurement).
 * Output (stdout): "<seconds> <checksum>"   (or the full result matrix with "dump")
 * Same deterministic inputs and checksum as matrix.c / matrix.py.
 */
public class main {
    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        int warmup = 0;
        boolean dump = false;
        for (int i = 1; i < args.length; i++) {
            if (args[i].equals("dump")) dump = true;
            else warmup = Integer.parseInt(args[i]);
        }

        int[][] a = new int[n][n];
        int[][] b = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                a[i][j] = (7 * i + 3 * j + 1) % 10 + 1;
                b[i][j] = (5 * i + 11 * j + 2) % 10 + 1;
            }

        int[][] c = null;
        for (int w = 0; w < warmup; w++) c = multiplyMatrices(a, b, n);   // untimed warm-up

        long t0 = System.nanoTime();
        c = multiplyMatrices(a, b, n);
        long t1 = System.nanoTime();
        double secs = (t1 - t0) / 1_000_000_000.0;

        StringBuilder sb = new StringBuilder();
        if (dump) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) { if (j > 0) sb.append(' '); sb.append(c[i][j]); }
                sb.append('\n');
            }
            System.out.print(sb);
        } else {
            long checksum = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) checksum += (long) (i + 1) * (j + 1) * c[i][j];
            System.out.println(String.format(java.util.Locale.US, "%.6f %d", secs, checksum));
        }
    }

    public static int[][] multiplyMatrices(int[][] a, int[][] b, int n) {
        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                result[i][j] = sum;
            }
        }
        return result;
    }
}
