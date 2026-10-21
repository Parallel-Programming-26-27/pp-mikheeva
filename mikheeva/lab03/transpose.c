#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void transpose_rows(const double *a, double *b, long long n)
{
    for (long long i = 0; i < n; ++i)
        for (long long j = 0; j < n; ++j)
            b[j * n + i] = a[i * n + j];
}

static void transpose_cols(const double *a, double *b, long long n)
{
    for (long long j = 0; j < n; ++j)
        for (long long i = 0; i < n; ++i)
            b[j * n + i] = a[i * n + j];
}

static void transpose_blocked(const double *a, double *b, long long n,
                              long long block)
{
    for (long long ii = 0; ii < n; ii += block)
        for (long long jj = 0; jj < n; jj += block) {
            long long imax = ii + block < n ? ii + block : n;
            long long jmax = jj + block < n ? jj + block : n;
            for (long long i = ii; i < imax; ++i)
                for (long long j = jj; j < jmax; ++j)
                    b[j * n + i] = a[i * n + j];
        }
}

static int transpose_rows_omp(const double *a, double *b, long long n)
{
    int actual_threads = 1;
    #pragma omp parallel for schedule(runtime)
    for (long long i = 0; i < n; ++i) {
        if (i == 0)
            actual_threads = omp_get_num_threads();
        for (long long j = 0; j < n; ++j)
            b[j * n + i] = a[i * n + j];
    }
    return actual_threads;
}

static int transpose_blocked_omp(const double *a, double *b, long long n,
                                long long block)
{
    int actual_threads = 1;
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < n; ii += block) {
        if (ii == 0)
            actual_threads = omp_get_num_threads();
        for (long long jj = 0; jj < n; jj += block) {
            long long imax = ii + block < n ? ii + block : n;
            long long jmax = jj + block < n ? jj + block : n;
            for (long long i = ii; i < imax; ++i)
                for (long long j = jj; j < jmax; ++j)
                    b[j * n + i] = a[i * n + j];
        }
    }
    return actual_threads;
}

static int parse_positive_integer(const char *text, long long *value)
{
    char *end;
    errno = 0;
    *value = strtoll(text, &end, 10);
    return errno == 0 && end != text && *end == '\0' &&
           *value > 0 && *value <= INT_MAX;
}

static int run(const double *a, double *b, const double *reference,
               long long n, long long block, int threads,
               const char *version, const char *schedule)
{
    double times[5];
    int actual_threads = 1;

    /* Run 0 is a warm-up; runs 1-5 are measured. */
    for (int run = 0; run <= 5; ++run) {
        for (long long k = 0; k < n * n; ++k)
            b[k] = NAN;
        double start = omp_get_wtime();
        if (strcmp(version, "rows") == 0)
            transpose_rows(a, b, n);
        else if (strcmp(version, "cols") == 0)
            transpose_cols(a, b, n);
        else if (strcmp(version, "blocked") == 0)
            transpose_blocked(a, b, n, block);
        else if (strcmp(version, "rows_omp") == 0)
            actual_threads = transpose_rows_omp(a, b, n);
        else
            actual_threads = transpose_blocked_omp(a, b, n, block);
        double elapsed = omp_get_wtime() - start;

        long long mismatches = 0;
        for (long long k = 0; k < n * n; ++k)
            if (b[k] != reference[k])
                ++mismatches;
        if (mismatches != 0 || actual_threads != threads) {
            fprintf(stderr, "check=FAIL mismatches=%lld threads=%d (requested %d)\n",
                    mismatches, actual_threads, threads);
            return 0;
        }
        if (run > 0)
            times[run - 1] = elapsed;
    }

    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (times[j] < times[i]) {
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }
    double bandwidth = times[2] > 0.0 ?
        2.0 * n * n * sizeof(double) / times[2] / 1e9 : NAN;
    printf("N=%lld b=%lld threads=%d version=%s schedule=%s "
           "time=%.9f s BW=%.3f GB/s check=OK mismatches=0\n",
           n, block, actual_threads, version, schedule, times[2], bandwidth);
    if (times[2] <= 0.0)
        printf("Elapsed time is too small. Increase N.\n");
    return 1;
}

int main(int argc, char **argv)
{
    long long n, block, threads;
    if ((argc != 5 && argc != 6) || !parse_positive_integer(argv[1], &n) ||
        !parse_positive_integer(argv[2], &block) ||
        !parse_positive_integer(argv[3], &threads)) {
        fprintf(stderr, "Usage: %s N block threads version [static|static,1|static,8]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    const char *version = argv[4];
    if (strcmp(version, "rows") != 0 && strcmp(version, "cols") != 0 &&
        strcmp(version, "blocked") != 0 && strcmp(version, "rows_omp") != 0 &&
        strcmp(version, "blocked_omp") != 0) {
        fprintf(stderr, "Unknown version.\n");
        return EXIT_FAILURE;
    }
    if (strcmp(version, "rows_omp") != 0 && strcmp(version, "blocked_omp") != 0)
        threads = 1;

    const char *schedule = argc == 6 ? argv[5] : "static";
    int chunk = 0;
    if (strcmp(schedule, "static,1") == 0)
        chunk = 1;
    else if (strcmp(schedule, "static,8") == 0)
        chunk = 8;
    else if (strcmp(schedule, "static") != 0) {
        fprintf(stderr, "Unknown schedule.\n");
        return EXIT_FAILURE;
    }
    if (argc == 6 && strcmp(version, "rows_omp") != 0) {
        fprintf(stderr, "Schedule is only used by rows_omp.\n");
        return EXIT_FAILURE;
    }
    if ((size_t)n > SIZE_MAX / sizeof(double) / (size_t)n) {
        fprintf(stderr, "Matrix size exceeds available address space.\n");
        return EXIT_FAILURE;
    }

    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double *a = malloc(bytes), *b = malloc(bytes), *reference = malloc(bytes);
    if (a == NULL || b == NULL || reference == NULL) {
        fprintf(stderr, "Cannot allocate matrices.\n");
        free(a);
        free(b);
        free(reference);
        return EXIT_FAILURE;
    }
    for (long long k = 0; k < n * n; ++k)
        a[k] = (double)k;
    transpose_rows(a, reference, n);
    omp_set_dynamic(0);
    omp_set_num_threads((int)threads);
    omp_set_schedule(omp_sched_static, chunk);

    int correct = run(a, b, reference, n, block, (int)threads, version, schedule);
    free(a);
    free(b);
    free(reference);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
