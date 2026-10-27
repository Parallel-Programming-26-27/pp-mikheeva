/* Build: gcc -O3 -march=native -fopenmp linreg_simd.c -o linreg_simd -lm
 * Run: ./linreg_simd N d K lambda eta threads [all|serial|omp|simd|omp_simd]
 * Each version starts from w = 0, b = 0. Print the median of five runs.
 * Allocation, data generation, reset and correctness checks are not timed.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double soft_threshold(double z, double alpha)
{
    if (z > alpha) return z - alpha;
    if (z < -alpha) return z + alpha;
    return 0.0;
}

static void train_serial(const double *x, const double *y, double *w,
                         double *b, double *g, int n, int d, int k,
                         double lambda, double eta)
{
    for (int iteration = 0; iteration < k; ++iteration) {
        memset(g, 0, (size_t)d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < n; ++i) {
            const double *row = x + (size_t)i * d;
            double prediction = 0.0;
            for (int j = 0; j < d; ++j)
                prediction += row[j] * w[j];
            double error = prediction + *b - y[i];
            gb += error;
            for (int j = 0; j < d; ++j)
                g[j] += error * row[j];
        }
        for (int j = 0; j < d; ++j)
            w[j] = soft_threshold(w[j] - eta * g[j] / n, eta * lambda);
        *b -= eta * gb / n; 
    }
}

static void train_simd(const double *x, const double *y, double *w,
                       double *b, double *g, int n, int d, int k,
                       double lambda, double eta)
{
    for (int iteration = 0; iteration < k; ++iteration) {
        memset(g, 0, (size_t)d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < n; ++i) {
            const double *row = x + (size_t)i * d;
            double prediction = 0.0;
            #pragma omp simd reduction(+:prediction)
            for (int j = 0; j < d; ++j)
                prediction += row[j] * w[j];
            double error = prediction + *b - y[i];
            gb += error;
            #pragma omp simd
            for (int j = 0; j < d; ++j)
                g[j] += error * row[j];
        }
        for (int j = 0; j < d; ++j)
            w[j] = soft_threshold(w[j] - eta * g[j] / n, eta * lambda);
        *b -= eta * gb / n;
    }
}

static void train_omp(const double *x, const double *y, double *w,
                      double *b, double *partial, int n, int d, int k,
                      double lambda, double eta, int threads, int use_simd)
{
    for (int iteration = 0; iteration < k; ++iteration) {
        memset(partial, 0, (size_t)threads * d * sizeof(double));
        double gb = 0.0;
        #pragma omp parallel reduction(+:gb)
        {
            double *g = partial + (size_t)omp_get_thread_num() * d;
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const double *row = x + (size_t)i * d;
                double prediction = 0.0;
                if (use_simd) {
                    #pragma omp simd reduction(+:prediction)
                    for (int j = 0; j < d; ++j)
                        prediction += row[j] * w[j];
                } else {
                    for (int j = 0; j < d; ++j)
                        prediction += row[j] * w[j];
                }
                double error = prediction + *b - y[i];
                gb += error;
                if (use_simd) {
                    #pragma omp simd
                    for (int j = 0; j < d; ++j)
                        g[j] += error * row[j];
                } else {
                    for (int j = 0; j < d; ++j)
                        g[j] += error * row[j];
                }
            }
        }
        for (int j = 0; j < d; ++j) {
            double sum = 0.0;
            for (int t = 0; t < threads; ++t)
                sum += partial[(size_t)t * d + j];
            w[j] = soft_threshold(w[j] - eta * sum / n, eta * lambda);
        }
        *b -= eta * gb / n;
    }
}

static double loss(const double *x, const double *y, const double *w,
                   double b, int n, int d, double lambda)
{
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        double prediction = b;
        for (int j = 0; j < d; ++j)
            prediction += x[(size_t)i * d + j] * w[j];
        double error = prediction - y[i];
        result += error * error;
    }
    result /= 2.0 * n;
    for (int j = 0; j < d; ++j)
        result += lambda * fabs(w[j]);
    return result;
}

static int run(const double *x, const double *y, double *w, double *g,
               double *partial, const double *reference, double reference_b,
               double reference_loss, int n, int d, int k, double lambda,
               double eta, int threads, const char *version)
{
    double times[5], f = 0.0, dw = 0.0, db = 0.0;
    int zeros = 0;
    int parallel = strcmp(version, "omp") == 0 || strcmp(version, "omp_simd") == 0;
    for (int repeat = 0; repeat < 5; ++repeat) {
        memset(w, 0, (size_t)d * sizeof(double));
        double b = 0.0;
        double start = omp_get_wtime();
        if (strcmp(version, "serial") == 0)
            train_serial(x, y, w, &b, g, n, d, k, lambda, eta);
        else if (strcmp(version, "simd") == 0)
            train_simd(x, y, w, &b, g, n, d, k, lambda, eta);
        else
            train_omp(x, y, w, &b, partial, n, d, k, lambda, eta,
                      threads, strcmp(version, "omp_simd") == 0);
        times[repeat] = omp_get_wtime() - start;

        f = loss(x, y, w, b, n, d, lambda);
        dw = 0.0;
        zeros = 0;
        for (int j = 0; j < d; ++j) {
            if (!isfinite(w[j])) return 0;
            if (fabs(w[j] - reference[j]) > dw)
                dw = fabs(w[j] - reference[j]);
            if (fabs(w[j]) < 1e-6) ++zeros;
        }
        db = fabs(b - reference_b);
        if (!isfinite(f) || !isfinite(b) || dw > 1e-8 || db > 1e-8 ||
            fabs(f - reference_loss) > 1e-8) {
            fprintf(stderr, "%s check=FAIL\n", version);
            return 0;
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (times[j] < times[i]) {
                double temp = times[i]; times[i] = times[j]; times[j] = temp;
            }
    printf("version=%s N=%d d=%d K=%d lambda=%.6g eta=%.6g threads=%d "
           "time=%.9f F=%.12g dw=%.3e db=%.3e dF=%.3e zeros=%d check=OK\n",
           version, n, d, k, lambda, eta, parallel ? threads : 1,
           times[2], f, dw, db, fabs(f - reference_loss), zeros);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 7 && argc != 8) {
        fprintf(stderr, "Usage: %s N d K lambda eta threads "
                "[all|serial|omp|simd|omp_simd]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]), d = atoi(argv[2]), k = atoi(argv[3]);
    double lambda = atof(argv[4]), eta = atof(argv[5]);
    int threads = atoi(argv[6]);
    const char *version = argc == 8 ? argv[7] : "all";
    const char *versions[] = {"serial", "omp", "simd", "omp_simd"};
    int valid = strcmp(version, "all") == 0;
    for (int v = 0; v < 4; ++v)
        if (strcmp(version, versions[v]) == 0) valid = 1;
    if (!valid || n <= 0 || d <= 0 || k <= 0 || threads <= 0 ||
        !isfinite(lambda) || !isfinite(eta) || lambda < 0.0 || eta <= 0.0)
        return EXIT_FAILURE;

    double *x = malloc((size_t)n * d * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    double *w = calloc((size_t)d, sizeof(double));
    double *reference = calloc((size_t)d, sizeof(double));
    double *g = malloc((size_t)d * sizeof(double));
    double *partial = malloc((size_t)threads * d * sizeof(double));
    if (!x || !y || !w || !reference || !g || !partial) {
        free(x); free(y); free(w); free(reference); free(g); free(partial);
        return EXIT_FAILURE;
    }
    srand(42);
    for (int i = 0; i < n; ++i) {
        y[i] = 0.5;
        for (int j = 0; j < d; ++j) {
            double value = 2.0 * rand() / RAND_MAX - 1.0;
            x[(size_t)i * d + j] = value;
            if (j < (d + 3) / 4)
                y[i] += value * (j % 2 ? -1.0 : 1.0) / (j + 1);
        }
        y[i] += 0.05 * (2.0 * rand() / RAND_MAX - 1.0);
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        #pragma omp single
        threads = omp_get_num_threads();
    }
    double reference_b = 0.0;
    train_serial(x, y, reference, &reference_b, g, n, d, k, lambda, eta);
    double reference_loss = loss(x, y, reference, reference_b, n, d, lambda);
    int correct = 1;
    for (int v = 0; v < 4 && correct; ++v)
        if (strcmp(version, "all") == 0 || strcmp(version, versions[v]) == 0)
            correct = run(x, y, w, g, partial, reference, reference_b,
                          reference_loss, n, d, k, lambda, eta, threads, versions[v]);
    free(x); free(y); free(w); free(reference); free(g); free(partial);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
