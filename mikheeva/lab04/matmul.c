/* Build: gcc -O3 -fopenmp -Wall -Wextra -o matmul matmul.c -lm
 * Run:   ./matmul N block threads version
 * version: ijk_seq, ijk_omp, ikj_seq, ikj_omp, blocked_omp.
 * Five timed runs; print the median. Sequential versions
 * use one thread. Allocation, initialization, reset and checks are not timed.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ijk_seq(const double *a, const double *b, double *c, long long n)
{
    for (long long i = 0; i < n; ++i)
        for (long long j = 0; j < n; ++j) {
            double sum = 0.0;
            for (long long k = 0; k < n; ++k)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

static void ijk_omp(const double *a, const double *b, double *c, long long n)
{
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        for (long long j = 0; j < n; ++j) {
            double sum = 0.0;
            for (long long k = 0; k < n; ++k)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
    }
}

static void ikj_seq(const double *a, const double *b, double *c, long long n)
{
    for (long long i = 0; i < n; ++i)
        for (long long k = 0; k < n; ++k) {
            double aik = a[i * n + k];
            for (long long j = 0; j < n; ++j)
                c[i * n + j] += aik * b[k * n + j];
        }
}

static void ikj_omp(const double *a, const double *b, double *c, long long n)
{
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        for (long long k = 0; k < n; ++k) {
            double aik = a[i * n + k];
            for (long long j = 0; j < n; ++j)
                c[i * n + j] += aik * b[k * n + j];
        }
    }
}

static void blocked_omp(const double *a, const double *b, double *c,
                       long long n, long long block)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long ii = 0; ii < n; ii += block)
        for (long long jj = 0; jj < n; jj += block) {
            long long imax = ii + block < n ? ii + block : n;
            long long jmax = jj + block < n ? jj + block : n;
            for (long long kk = 0; kk < n; kk += block) {
                long long kmax = kk + block < n ? kk + block : n;
                for (long long i = ii; i < imax; ++i)
                    for (long long k = kk; k < kmax; ++k) {
                        double aik = a[i * n + k];
                        for (long long j = jj; j < jmax; ++j)
                            c[i * n + j] += aik * b[k * n + j];
                    }
            }
        }
}

static int run(const double *a, const double *b, double *c,
               const double *reference, long long n, long long block,
               int threads, const char *version, size_t bytes)
{
    double times[5];
    for (int repeat = 0; repeat < 5; ++repeat) {
        memset(c, 0, bytes);
        double start = omp_get_wtime();
        if (strcmp(version, "ijk_seq") == 0)
            ijk_seq(a, b, c, n);
        else if (strcmp(version, "ijk_omp") == 0)
            ijk_omp(a, b, c, n);
        else if (strcmp(version, "ikj_seq") == 0)
            ikj_seq(a, b, c, n);
        else if (strcmp(version, "ikj_omp") == 0)
            ikj_omp(a, b, c, n);
        else
            blocked_omp(a, b, c, n, block);
        double elapsed = omp_get_wtime() - start;

        long long mismatches = 0;
        for (long long k = 0; k < n * n; ++k) {
            double error = fabs(c[k] - reference[k]);
            if (!isfinite(c[k]) || error > 1e-9 * (1.0 + fabs(reference[k])))
                ++mismatches;
        }
        if (mismatches != 0) {
            fprintf(stderr, "check=FAIL mismatches=%lld\n", mismatches);
            return 0;
        }
        times[repeat] = elapsed;
    }
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (times[j] < times[i]) {
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }
    printf("N=%lld b=%lld threads=%d version=%s time=%.9f s "
           "check=OK\n", n, block, threads, version, times[2]);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s N block threads "
                "ijk_seq|ijk_omp|ikj_seq|ikj_omp|blocked_omp\n", argv[0]);
        return EXIT_FAILURE;
    }
    long long n = atoll(argv[1]);
    long long block = atoll(argv[2]);
    int threads = atoi(argv[3]);
    if (n <= 0 || block <= 0 || threads <= 0)
        return EXIT_FAILURE;
    const char *version = argv[4];
    if (strcmp(version, "ijk_seq") != 0 && strcmp(version, "ijk_omp") != 0 &&
        strcmp(version, "ikj_seq") != 0 && strcmp(version, "ikj_omp") != 0 &&
        strcmp(version, "blocked_omp") != 0) {
        fprintf(stderr, "Unknown version.\n");
        return EXIT_FAILURE;
    }
    if (strcmp(version, "ijk_seq") == 0 || strcmp(version, "ikj_seq") == 0)
        threads = 1;
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double *a = malloc(bytes), *b = malloc(bytes);
    double *c = malloc(bytes), *reference = malloc(bytes);
    for (long long k = 0; k < n * n; ++k) {
        a[k] = ((double)(k % 97) - 48.0) / 97.0;
        b[k] = ((double)(k % 89) - 44.0) / 89.0;
    }
    ijk_seq(a, b, reference, n);
    omp_set_dynamic(0);
    omp_set_num_threads((int)threads);
    int correct = run(a, b, c, reference, n, block, (int)threads, version, bytes);
    free(a); free(b); free(c); free(reference);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
