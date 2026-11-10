/* Build: mpicc -O2 -Wall -Wextra mpi_array_sum.c -o mpi_array_sum
 * Run: mpiexec -n 4 ./mpi_array_sum 1000000
 * Aggregation timing: mpiexec -n 4 ./mpi_array_sum 1000000 10000
 */
#include <mpi.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int positive_int(const char *text, int *value)
{
    char *end;
    errno = 0;
    long number = strtol(text, &end, 10);
    if (errno || end == text || *end || number <= 0 || number > INT_MAX)
        return 0;
    *value = (int)number;
    return 1;
}

static void *allocate(size_t count, size_t width)
{
    void *memory = calloc(count, width);
    if (!memory) {
        fprintf(stderr, "Cannot allocate memory.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
    return memory;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Synchronize before timing; exclude MPI startup and command-line parsing. */
    int parameters[3] = {0, 0, 0}; /* N, aggregation repeats, valid */
    if (rank == 0) {
        parameters[2] = (argc == 2 || argc == 3) &&
            positive_int(argv[1], &parameters[0]);
        if (parameters[2] && argc == 3)
            parameters[2] = positive_int(argv[2], &parameters[1]);
        if (parameters[0] % size != 0) parameters[2] = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    MPI_Bcast(parameters, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parameters[2]) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpiexec -n P mpi_array_sum N [repeats]\n"
                    "N must be positive, <= INT_MAX and divisible by P; "
                    "repeats must be positive.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n = parameters[0], count = n / size;
    int *array = NULL;
    long long *partials = NULL;
    if (rank == 0) {
        array = allocate((size_t)n, sizeof(*array));
        partials = allocate((size_t)size, sizeof(*partials));
        for (int i = 0; i < n; ++i) array[i] = i + 1;
    }
    int *local = allocate((size_t)count, sizeof(*local));
    MPI_Scatter(array, count, MPI_INT, local, count, MPI_INT,
                0, MPI_COMM_WORLD);
    long long local_sum = 0, gathered = 0, reduced = 0, total = 0;
    for (int i = 0; i < count; ++i) local_sum += local[i];

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, partials, 1,
               MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        for (int r = 0; r < size; ++r) gathered += partials[r];
    MPI_Reduce(&local_sum, &reduced, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum, &total, 1, MPI_LONG_LONG_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    long long expected = (long long)n * ((long long)n + 1) / 2;
    int correct = total == expected;
    if (rank == 0)
        correct = correct && gathered == expected && reduced == expected;
    int all_correct;
    MPI_Allreduce(&correct, &all_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("P=%d N=%d expected=%lld gather=%lld reduce=%lld "
               "allreduce=%lld seconds=%.9f check=%s\n",
               size, n, expected, gathered, reduced, total, max_elapsed,
               all_correct ? "OK" : "FAIL");
        printf("Local sums:");
        for (int r = 0; r < size; ++r) printf(" %lld", partials[r]);
        printf("\nAllreduce verified on all %d processes.\n", size);
    }

    /* Optional comparison: exclude setup and validate every repetition.
     * Measure complete batches, then take the maximum across processes. */
    int repeats = parameters[1];
    if (repeats) {
        const int warmup = 20;
        for (int method = 0; method < 2; ++method) {
            double batch_start = 0;
            int batch_ok = 1;
            for (long long i = -warmup; i < repeats; ++i) {
                if (i == 0) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    batch_start = MPI_Wtime();
                }
                if (method == 0) {
                    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, partials, 1,
                               MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
                    if (rank == 0) {
                        gathered = 0;
                        for (int r = 0; r < size; ++r) gathered += partials[r];
                        if (gathered != expected) batch_ok = 0;
                    }
                } else {
                    MPI_Reduce(&local_sum, &reduced, 1, MPI_LONG_LONG_INT,
                               MPI_SUM, 0, MPI_COMM_WORLD);
                    if (rank == 0 && reduced != expected) batch_ok = 0;
                }
            }
            elapsed = (MPI_Wtime() - batch_start) / repeats;
            MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                       0, MPI_COMM_WORLD);
            MPI_Bcast(&batch_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            all_correct = all_correct && batch_ok;
            if (rank == 0)
                printf("aggregation=%s repeats=%d seconds=%.9f check=%s\n",
                       method == 0 ? "gather" : "reduce", repeats,
                       max_elapsed, batch_ok ? "OK" : "FAIL");
        }
    }

    free(array);
    free(local);
    free(partials);
    MPI_Finalize();
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
