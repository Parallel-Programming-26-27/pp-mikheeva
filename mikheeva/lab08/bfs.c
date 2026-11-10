/* Build: mpicc -O2 -Wall -Wextra -fopenmp bfs.c -o bfs
 * Run: OMP_NUM_THREADS=2 mpiexec -n 4 ./bfs graph.txt 0 directed dynamic 5
 * Input: N M, followed by M pairs u v (zero-based). */
#include <mpi.h>
#include <omp.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int n, m; int *offsets, *edges; } Graph;
typedef struct { int *data; size_t size, capacity; int failed; } Candidates;

static void fail(const char *message)
{
    fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t width)
{
    if (count > SIZE_MAX / width) fail("Allocation size overflow.");
    void *p = calloc(count ? count : 1, width);
    if (!p) fail("Cannot allocate memory.");
    return p;
}

static int integer(const char *text, int minimum, int *value)
{
    char *end;
    errno = 0;
    long x = strtol(text, &end, 10);
    if (errno || end == text || *end || x < minimum || x >= INT_MAX) return 0;
    *value = (int)x;
    return 1;
}

/* Only rank 0 reads the file; broadcasting CSR guarantees identical graphs. */
static Graph read_graph(const char *path, int undirected)
{
    FILE *file = fopen(path, "r");
    if (!file) fail("Cannot open graph file.");
    long long n, m;
    if (fscanf(file, "%lld %lld", &n, &m) != 2 || n <= 0 || n >= INT_MAX ||
        m < 0 || m > (INT_MAX - 1) / (undirected ? 2 : 1))
        fail("Invalid graph header.");
    Graph g = {(int)n, (int)m * (undirected ? 2 : 1), NULL, NULL};
    g.offsets = allocate((size_t)g.n + 1, sizeof(int));
    g.edges = allocate((size_t)g.m, sizeof(int));
    int *from = allocate((size_t)m, sizeof(int));
    int *to = allocate((size_t)m, sizeof(int));
    for (int i = 0; i < (int)m; ++i) {
        long long u, v;
        if (fscanf(file, "%lld %lld", &u, &v) != 2 ||
            u < 0 || u >= n || v < 0 || v >= n) fail("Invalid graph edge.");
        from[i] = (int)u;
        to[i] = (int)v;
        ++g.offsets[u + 1];
        if (undirected) ++g.offsets[v + 1];
    }
    char extra;
    if (fscanf(file, " %c", &extra) == 1 || ferror(file))
        fail("Unexpected data after graph edges.");
    fclose(file);
    for (int v = 0; v < g.n; ++v) g.offsets[v + 1] += g.offsets[v];
    int *cursor = allocate((size_t)g.n, sizeof(int));
    memcpy(cursor, g.offsets, (size_t)g.n * sizeof(int));
    for (int i = 0; i < (int)m; ++i) {
        g.edges[cursor[from[i]]++] = to[i];
        if (undirected) g.edges[cursor[to[i]]++] = from[i];
    }
    free(cursor);
    free(from);
    free(to);
    return g;
}

static void sequential(const Graph *g, int source, int *dist, int *queue)
{
    for (int v = 0; v < g->n; ++v) dist[v] = -1;
    int head = 0, tail = 0;
    dist[source] = 0;
    queue[tail++] = source;
    while (head < tail) {
        int v = queue[head++];
        for (int j = g->offsets[v]; j < g->offsets[v + 1]; ++j) {
            int u = g->edges[j];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
}

/* Called by the owning thread only; MPI calls remain outside OpenMP regions. */
static void append(Candidates *list, int vertex)
{
    if (list->failed) return;
    if (list->size == list->capacity) {
        if (list->capacity > SIZE_MAX / (2 * sizeof(int))) {
            list->failed = 1;
            return;
        }
        size_t capacity = list->capacity * 2;
        int *data = realloc(list->data, capacity * sizeof(int));
        if (!data) { list->failed = 1; return; }
        list->data = data;
        list->capacity = capacity;
    }
    list->data[list->size++] = vertex;
}

static int hybrid(const Graph *g, int source, int rank, int processes,
                  int threads, Candidates *lists, int *dist, int *frontier,
                  int *next, int *local, int *global, int *widths,
                  long long *vertices, long long *edges, double *communication)
{
    for (int v = 0; v < g->n; ++v) dist[v] = -1;
    dist[source] = 0;
    frontier[0] = source;
    int size = 1, level = 0;
    *vertices = *edges = 0;
    *communication = 0;
    while (size) {
        widths[level] = size;
        memset(local, 0, (size_t)g->n * sizeof(int));
        for (int t = 0; t < threads; ++t) lists[t].size = 0;
        int count = size / processes + (rank < size % processes);
        long long visited_edges = 0;
        #pragma omp parallel reduction(+:visited_edges)
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(runtime)
            for (int k = 0; k < count; ++k) {
                int v = frontier[rank + k * processes];
                visited_edges += g->offsets[v + 1] - g->offsets[v];
                for (int j = g->offsets[v]; j < g->offsets[v + 1]; ++j) {
                    int u = g->edges[j];
                    if (dist[u] == -1) append(&lists[tid], u);
                }
            }
        }
        *vertices += count;
        *edges += visited_edges;
        for (int t = 0; t < threads; ++t) {
            if (lists[t].failed) fail("Cannot grow thread candidate list.");
            for (size_t j = 0; j < lists[t].size; ++j)
                local[lists[t].data[j]] = 1;
        }
        double start = MPI_Wtime();
        MPI_Allreduce(local, global, g->n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        *communication += MPI_Wtime() - start;
        int next_size = 0;
        /* All ranks scan in the same order; dist is written only here. */
        for (int u = 0; u < g->n; ++u)
            if (global[u] && dist[u] == -1) {
                dist[u] = level + 1;
                next[next_size++] = u;
            }
        int *swap = frontier; frontier = next; next = swap;
        size = next_size;
        ++level;
    }
    return level;
}

int main(int argc, char **argv)
{
    int provided, rank, processes;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required.");
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int source = 0, repeats = 1;
    if (argc < 3 || argc > 6 || !integer(argv[2], 0, &source) ||
        (argc >= 4 && strcmp(argv[3], "directed") && strcmp(argv[3], "undirected")) ||
        (argc >= 5 && strcmp(argv[4], "static") && strcmp(argv[4], "dynamic")) ||
        (argc == 6 && !integer(argv[5], 1, &repeats))) {
        if (!rank) fprintf(stderr, "Usage: bfs graph.txt source "
                           "[directed|undirected [static|dynamic [repeats]]]\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int undirected = argc >= 4 && !strcmp(argv[3], "undirected");
    const char *schedule = argc >= 5 ? argv[4] : "dynamic";
    omp_set_dynamic(0);
    omp_set_schedule(!strcmp(schedule, "static") ? omp_sched_static : omp_sched_dynamic, 1);
    int threads = omp_get_max_threads();
    Graph g = {0, 0, NULL, NULL};
    if (!rank) g = read_graph(argv[1], undirected);
    int header[2] = {g.n, g.m};
    MPI_Bcast(header, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (source >= header[0]) fail("Source vertex is outside graph.");
    if (rank) {
        g.n = header[0]; g.m = header[1];
        g.offsets = allocate((size_t)g.n + 1, sizeof(int));
        g.edges = allocate((size_t)g.m, sizeof(int));
    }
    MPI_Bcast(g.offsets, g.n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.edges, g.m, MPI_INT, 0, MPI_COMM_WORLD);
    int *reference = allocate((size_t)g.n, sizeof(int));
    int *dist = allocate((size_t)g.n, sizeof(int));
    int *frontier = allocate((size_t)g.n, sizeof(int));
    int *next = allocate((size_t)g.n, sizeof(int));
    int *local = allocate((size_t)g.n, sizeof(int));
    int *global = allocate((size_t)g.n, sizeof(int));
    int *widths = allocate((size_t)g.n, sizeof(int));
    long long *work = allocate((size_t)processes * 2, sizeof(long long));
    Candidates *lists = allocate((size_t)threads, sizeof(*lists));
    for (int t = 0; t < threads; ++t) {
        lists[t].capacity = 256;
        lists[t].data = allocate(lists[t].capacity, sizeof(int));
    }
    if (!rank) printf("N=%d M=%d source=%d mode=%s P=%d T=%d schedule=%s repeats=%d\n",
                      g.n, g.m, source, undirected ? "undirected" : "directed",
                      processes, threads, schedule, repeats);
    int correct = 1, levels = 0;
    for (int run = 0; run < repeats; ++run) {
        double seq = 0;
        if (!rank) {
            double start = MPI_Wtime();
            sequential(&g, source, reference, frontier);
            seq = MPI_Wtime() - start;
        }
        MPI_Bcast(reference, g.n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        long long counts[2];
        double communication;
        double start = MPI_Wtime();
        levels = hybrid(&g, source, rank, processes, threads, lists, dist,
                        frontier, next, local, global, widths,
                        &counts[0], &counts[1], &communication);
        double elapsed = MPI_Wtime() - start, maximum = 0, max_comm = 0;
        MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&communication, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        int mismatches = 0, max_mismatches = 0;
        for (int v = 0; v < g.n; ++v) mismatches += dist[v] != reference[v];
        MPI_Allreduce(&mismatches, &max_mismatches, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (max_mismatches) correct = 0;
        MPI_Gather(counts, 2, MPI_LONG_LONG_INT, work, 2, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (!rank) {
            int unreachable = 0, max_frontier = 0;
            for (int v = 0; v < g.n; ++v) unreachable += dist[v] == -1;
            for (int l = 0; l < levels; ++l)
                if (widths[l] > max_frontier) max_frontier = widths[l];
            double speedup = maximum > 0 ? seq / maximum : 0;
            printf("run=%d seq=%.9f par=%.9f comm=%.9f S=%.6f E=%.6f "
                   "levels=%d unreachable=%d mismatches=%d max_frontier=%d check=%s\n",
                   run + 1, seq, maximum, max_comm, speedup,
                   speedup / ((double)processes * threads), levels, unreachable,
                   max_mismatches, max_frontier, max_mismatches ? "FAIL" : "OK");
        }
    }
    if (!rank) {
        printf("Frontiers:");
        for (int l = 0; l < levels; ++l) printf(" %d", widths[l]);
        printf("\nVertices per rank:");
        for (int r = 0; r < processes; ++r) printf(" %lld", work[2 * r]);
        printf("\nEdges per rank:");
        for (int r = 0; r < processes; ++r) printf(" %lld", work[2 * r + 1]);
        if (g.n <= 32) {
            printf("\nDistances:");
            for (int v = 0; v < g.n; ++v) printf(" %d", dist[v]);
        }
        printf("\n");
    }
    for (int t = 0; t < threads; ++t) free(lists[t].data);
    free(lists); free(work); free(widths); free(global); free(local);
    free(next); free(frontier); free(dist); free(reference);
    free(g.offsets); free(g.edges);
    MPI_Finalize();
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
