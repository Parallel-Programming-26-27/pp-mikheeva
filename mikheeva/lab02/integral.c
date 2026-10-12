
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static double function(double x)
{
    return x * sin(x * x);
}

static double integrate(double a, double b, long long n, int parallel,
                        int threads, int *actual_threads)
{
    const double h = (b - a) / (double)n;
    double sum = 0.0;
    *actual_threads = 1;

    if (parallel) {
        #pragma omp parallel num_threads(threads) reduction(+:sum)
        {
            #pragma omp single
            *actual_threads = omp_get_num_threads();

            #pragma omp for schedule(static)
            for (long long i = 0; i < n; ++i) {
                double x = a + ((double)i + 0.5) * h;
                sum += function(x);
            }
        }
    } else {
        for (long long i = 0; i < n; ++i) {
            double x = a + ((double)i + 0.5) * h;
            sum += function(x);
        }
    }
    return h * sum;
}

static int parse_double(const char *text, double *value)
{
    char *end;
    errno = 0;
    *value = strtod(text, &end);
    return errno == 0 && end != text && *end == '\0' && isfinite(*value);
}

static int parse_positive_integer(const char *text, long long *value)
{
    char *end;
    errno = 0;
    *value = strtoll(text, &end, 10);
    return errno == 0 && end != text && *end == '\0' && *value > 0;
}

static int run(double a, double b, long long initial_n, int threads,
               int parallel, double epsilon, double exact, double *elapsed)
{
    long long n = initial_n;
    int actual_threads = 1;
    double value, error;
    const double start = omp_get_wtime();

    for (;;) {
        value = integrate(a, b, n, parallel, threads, &actual_threads);
        error = fabs(value - exact);
        if (!isfinite(value)) {
            fprintf(stderr, "Non-finite integral; check the input interval.\n");
            return 0;
        }
        if (epsilon == 0.0 || error <= epsilon)
            break;
        if (n > 4503599627370496LL / 2) {
            fprintf(stderr, "Accuracy was not reached before the n limit.\n");
            return 0;
        }
        n *= 2;
    }

    *elapsed = omp_get_wtime() - start;
    printf("%s: n=%lld, threads=%d, integral=%.17g, error=%.6e, time=%.6f s\n",
           parallel ? "Parallel" : "Sequential", n, actual_threads,
           value, error, *elapsed);
    return 1;
}

int main(int argc, char **argv)
{
    double a = 0.0, b = 1.0, epsilon = 0.0;
    long long n = 100000000, threads = 8;
    double sequential_time, parallel_time;

    if (argc != 1 && argc != 5 && argc != 6) {
        fprintf(stderr, "Usage: %s [a b n threads [epsilon]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 5 &&
        (!parse_double(argv[1], &a) || !parse_double(argv[2], &b) ||
         !parse_positive_integer(argv[3], &n) ||
         !parse_positive_integer(argv[4], &threads))) {
        fprintf(stderr, "Invalid interval, n or thread count.\n");
        return EXIT_FAILURE;
    }
    if (argc == 6 && (!parse_double(argv[5], &epsilon) || epsilon <= 0.0)) {
        fprintf(stderr, "Epsilon must be a finite positive number.\n");
        return EXIT_FAILURE;
    }
    if (threads > INT_MAX || n > 4503599627370496LL ||
        !isfinite(b - a) || !isfinite(a * a) || !isfinite(b * b)) {
        fprintf(stderr, "Input exceeds supported numeric limits.\n");
        return EXIT_FAILURE;
    }

    const double exact = (cos(a * a) - cos(b * b)) / 2.0;
    omp_set_dynamic(0);
    printf("Variant K: x*sin(x*x), interval=[%.17g, %.17g]\n", a, b);
    printf("Exact integral: %.17g\n", exact);
    if (epsilon > 0.0)
        printf("Accuracy mode: absolute error <= %.6e\n", epsilon);

    if (!run(a, b, n, (int)threads, 0, epsilon, exact, &sequential_time) ||
        !run(a, b, n, (int)threads, 1, epsilon, exact, &parallel_time))
        return EXIT_FAILURE;

    if (epsilon == 0.0 && parallel_time > 0.0)
        printf("Speedup: %.3f\n", sequential_time / parallel_time);
    return EXIT_SUCCESS;
}
