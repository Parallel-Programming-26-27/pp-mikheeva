#include <stdio.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void)
{
    long long n;
    int threads;

    printf("Number of terms n: ");
    fflush(stdout);
    if (scanf("%lld", &n) != 1 || n <= 0) {
        printf("Error: n must be a positive integer.\n");
        return 1;
    }

    printf("Number of threads: ");
    fflush(stdout);
    if (scanf("%d", &threads) != 1 || threads <= 0) {
        printf("Error: the number of threads must be a positive integer.\n");
        return 1;
    }

    double sum = 0.0;
    double start = omp_get_wtime();
    for (long long i = 0; i < n; i++) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }
    double pi_seq = 4.0 * sum;
    double time_seq = omp_get_wtime() - start;

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    int actual_threads = 0;
    sum = 0.0;
    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; i++) {
    
        if (i == 0) {
            actual_threads = omp_get_num_threads();
        }
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }
    double pi_par = 4.0 * sum;
    double time_par = omp_get_wtime() - start;

    printf("\nn = %lld\n\n", n);
    printf("Method           Pi                   Abs. error   Time (s)     Threads\n");
    printf("-----------------------------------------------------------------------\n");
    printf("Sequential       %.15f    %.3e    %.6f     1\n",
           pi_seq, fabs(pi_seq - M_PI), time_seq);
    printf("Parallel         %.15f    %.3e    %.6f     %d\n",
           pi_par, fabs(pi_par - M_PI), time_par, actual_threads);

    if (time_par > 0.0) {
        printf("\nSpeedup: %.2f\n", time_seq / time_par);
    } else {
        printf("\nElapsed time is too small. Increase n.\n");
    }

    return 0;
}
