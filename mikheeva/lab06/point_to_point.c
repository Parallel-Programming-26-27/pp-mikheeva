/* Build: mpicc -O2 -Wall -Wextra point_to_point.c -o point_to_point
 * Run: mpiexec -n 4 ./point_to_point
 * Timing: mpiexec -n 2 ./point_to_point timing
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TAG_MESSAGE = 1, TAG_REPLY = 2 };

static void exchange(int rank, int size)
{
    char message[128];
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            snprintf(message, sizeof(message), "Hello, process %d!", r);
            printf("Process 0 -> %d: %s\n", r, message);
            MPI_Send(message, (int)strlen(message) + 1, MPI_CHAR,
                     r, TAG_MESSAGE, MPI_COMM_WORLD);
        }
        for (int r = 1; r < size; ++r) {
            MPI_Recv(message, sizeof(message), MPI_CHAR,
                     r, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process 0 received from %d: %s\n", r, message);
        }
    } else {
        MPI_Recv(message, sizeof(message), MPI_CHAR,
                 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received from 0: %s\n", rank, message);
        snprintf(message, sizeof(message), "Message received by process %d", rank);
        printf("Process %d -> 0: %s\n", rank, message);
        MPI_Send(message, (int)strlen(message) + 1, MPI_CHAR,
                 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

static void timing(int rank)
{
    const int sizes[] = {1, 1024, 65536, 1048576};
    const int warmup = 10, repeats = 1000;
    char *buffer = malloc(sizes[3]);
    if (!buffer) {
        fprintf(stderr, "Process %d: cannot allocate buffer.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return;
    }
    for (int s = 0; s < 4; ++s) {
        int count = sizes[s];
        memset(buffer, 'A', count);
        double start = 0.0;
        for (int i = -warmup; i < repeats; ++i) {
            if (rank == 0) {
                if (i == 0) start = MPI_Wtime();
                MPI_Send(buffer, count, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
                MPI_Recv(buffer, count, MPI_CHAR, 1, TAG_REPLY,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buffer, count, MPI_CHAR, 0, TAG_MESSAGE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, count, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
            }
        }
        if (rank == 0) {
            double elapsed = MPI_Wtime() - start;
            for (int j = 0; j < count; ++j)
                if (buffer[j] != 'A') {
                    fprintf(stderr, "Incorrect reply.\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            printf("bytes=%d repeats=%d round_trip=%.9f check=OK\n",
                   count, repeats, elapsed / repeats);
        }
    }
    free(buffer);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    setvbuf(stdout, NULL, _IONBF, 0);

    int measure = argc == 2 && strcmp(argv[1], "timing") == 0;
    if (size < 2 || (measure && size != 2) ||
        (argc != 1 && !measure)) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpiexec -n P point_to_point (P >= 2)\n"
                    "Timing: mpiexec -n 2 point_to_point timing\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (measure) timing(rank);
    else exchange(rank, size);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
