/* memrun: run a command and report its peak resident set size (ru_maxrss).
 * Replacement for `/usr/bin/time -f %M` (not installed everywhere).
 * Usage: ./memrun <cmd> [args...]
 * The child's stdout/stderr pass through; on exit memrun prints "MAXRSS_KB=<n>" to stderr.
 * Metric: peak RSS in KiB as reported by wait4()/getrusage (Linux), for the whole process
 * (JVM and interpreter included). memrun itself is tiny, so it does not inflate the value.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s cmd [args...]\n", argv[0]); return 2; }
    pid_t pid = fork();
    if (pid == 0) { execvp(argv[1], argv + 1); perror("execvp"); _exit(127); }
    int status; struct rusage ru;
    if (wait4(pid, &status, 0, &ru) < 0) { perror("wait4"); return 2; }
    fprintf(stderr, "MAXRSS_KB=%ld\n", ru.ru_maxrss);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
