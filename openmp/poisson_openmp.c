#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>


void saveMatrixToCSV(const char *filename, int rows, int cols, double *array) {
    FILE *file = fopen(filename, "w");

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            fprintf(file, "%f", array[idx]);
            if (j < cols - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}


bool isPointInArea(double x, double y) {
    return (y*y < x) && (x < 1);
}


double makeGrid(int M, int N, double A1, double B1, double A2, double B2, double *x, double *y) {
    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;

    #pragma omp parallel for
    for (int i = 0; i <= M; ++i) {
        x[i] = A1 + i * h1;
    }

    #pragma omp parallel for
    for (int i = 0; i <= N; ++i) {
        y[i] = A2 + i * h2;
    }

    double h = fmax(h1, h2);

    return h*h;
}


void calculateCoefficients_simple(int M, int N, const double *x, const double *y, double *a, double *b, double epsilon) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            int idx = i * N + j;
            if (isPointInArea(x[i], y[j])) {
                a[idx] = 1.0;
                b[idx] = 1.0;
            } else {
                a[idx] = 1.0 / epsilon;
                b[idx] = 1.0 / epsilon;
            }
        }
    }
}


void calculateFValues(int M, int N, double *F, const double *x, const double *y, double epsilon, double h1, double h2) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            int idx = i * N + j;

            double x_min, x_max, y_min, y_max;

            x_min = x[i] - 0.5 * h1;
            x_max = x[i] + 0.5 * h1;

            y_min = y[j] - 0.5 * h2;
            y_max = y[j] + 0.5 * h2;

            bool need_double = false;

            if (y_min < 0 && y_max > 0) {
                y_min = 0;
                need_double = true;
            }

            if (y_min <= 0 && y_max <= 0) {
                y_min = -y_min;
                y_max = -y_max;
            }


            if (isPointInArea(x_min, y_min) && isPointInArea(x_min, y_max) && isPointInArea(x_max, y_min) && isPointInArea(x_max, y_max)) {
                F[idx] = 1.0;
            } else if (!isPointInArea(x_min, y_min) && !isPointInArea(x_min, y_max) && !isPointInArea(x_max, y_min) && !isPointInArea(x_max, y_max)) {
                F[idx] = 0.0;
            } else if (isPointInArea(x_min, y_min) && !isPointInArea(x_min, y_max) && isPointInArea(x_max, y_min) && isPointInArea(x_max, y_max)) {
                double s = h1 * h2 - (y_max * y_max - x_min) * (y_max - sqrt(x_min)) / 2;
                F[idx] = s / (h1 * h2);
            } else if (!isPointInArea(x_min, y_min) && !isPointInArea(x_min, y_max) && isPointInArea(x_max, y_min) && !isPointInArea(x_max, y_max)) {
                double s = (sqrt(x_max) - y_min) * (x_max - y_min * y_min) / 2;
                F[idx] = s / (h1 * h2);
            } else if (isPointInArea(x_min, y_min) && !isPointInArea(x_min, y_max) && isPointInArea(x_max, y_min) && !isPointInArea(x_max, y_max)) {
                double s = ((sqrt(x_min) - y_min) + (sqrt(x_max) - y_min)) / 2 * h1;
                F[idx] = s / (h1 * h2);
            } else {
                double s = ((x_max - y_max * y_max) + (x_max - y_min * y_min)) / 2 * h2;
                F[idx] = s / (h1 * h2);
            }

            if (need_double) {
                F[idx] *= 2;
            }
        }
    }
}


void calculateCoefficients(int M, int N, const double *x, const double *y, double *a, double *b, double epsilon, double h1, double h2) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            int idx = i * N + j;

            double x_min, x_max, y_min, y_max;

            x_min = x[i] - 0.5 * h1;
            x_max = x[i] + 0.5 * h1;

            y_min = y[j] - 0.5 * h2;
            y_max = y[j] + 0.5 * h2;


            if (y[j] >= 0) {
                if (isPointInArea(x_min, y_min) && isPointInArea(x_min, y_max)) {
                    a[idx] = 1.0;
                } else if (!isPointInArea(x_min, y_min) && !isPointInArea(x_min, y_max)) {
                    a[idx] = 1.0 / epsilon;
                } else {
                    double l = sqrt(x_min) - y_min;
                    a[idx] = l / h2 + (1 - l / h2) / epsilon;
                }


                if (isPointInArea(x_min, y_min) && isPointInArea(x_max, y_min)) {
                    b[idx] = 1.0;
                } else if (!isPointInArea(x_min, y_min) && !isPointInArea(x_max, y_min)) {
                    b[idx] = 1.0 / epsilon;
                } else {
                    double l = x_max - y_min * y_min;
                    b[idx] = l / h1 + (1 - l / h1) / epsilon;
                }

            } else {
                if (isPointInArea(x_min, y_min) && isPointInArea(x_min, y_max)) {
                    a[idx] = 1.0;
                } else if (!isPointInArea(x_min, y_min) && !isPointInArea(x_min, y_max)) {
                    a[idx] = 1.0 / epsilon;
                } else {
                    double l = sqrt(x_min) + y_max;
                    a[idx] = l / h2 + (1 - l / h2) / epsilon;
                }


                if (isPointInArea(x_min, y_min) && isPointInArea(x_max, y_min)) {
                    b[idx] = 1.0;
                } else if (!isPointInArea(x_min, y_min) && !isPointInArea(x_max, y_min)) {
                    b[idx] = 1.0 / epsilon;
                } else {
                    double l = x_max - y_min * y_min;
                    b[idx] = l / h1 + (1 - l / h1) / epsilon;
                }
            }
        }
    }
}



void steepestDescent(int M, int N, const double *F, double *w, const double *a, const double *b, double tol, int maxIter, double h1, double h2) {
    int totalPoints = (M + 1) * (N + 1);
    double *r = (double*) malloc(totalPoints * sizeof(double));
    double *Ar = (double*) malloc(totalPoints * sizeof(double));

    for (int iter = 0; iter < maxIter; ++iter) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= M - 1; ++i) {
            for (int j = 1; j <= N - 1; ++j) {
                int idx = i * N + j;
                r[idx] = ( -1.0 / h1 * (a[idx+N]*(w[idx+N] - w[idx]) / h1 - a[idx]*(w[idx] - w[idx-N]) / h1) -
                            1.0 / h2 * (b[idx+1]*(w[idx+1] - w[idx]) / h2 - b[idx]*(w[idx] - w[idx-1]) / h2)
                           ) - F[idx];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= M - 1; ++i) {
            for (int j = 1; j <= N - 1; ++j) {
                int idx = i * N + j;
                Ar[idx] = ( -1.0 / h1 * (a[idx+N]*(r[idx+N] - r[idx]) / h1 - a[idx]*(r[idx] - r[idx-N]) / h1) -
                             1.0 / h2 * (b[idx+1]*(r[idx+1] - r[idx]) / h2 - b[idx]*(r[idx] - r[idx-1]) / h2)
                           );
            }
        }


        double rr = 0, Ar_r = 0;

        #pragma omp parallel for reduction(+:rr,Ar_r)
        for (int k = 0; k < totalPoints; ++k) {
            rr += r[k] * r[k];
            Ar_r += Ar[k] * r[k];
        }

        double tau = rr / Ar_r;

        #pragma omp parallel for
        for (int k = 0; k < totalPoints; ++k) {
            w[k] = w[k] - tau * r[k];
        }


        if (sqrt(rr) < tol) {
            printf("Converged at iteration: %d\n", iter);
            free(r);
            free(Ar);
            return;
        }
    }
    printf("Reached max iteration: %d\n", maxIter);
    free(r);
    free(Ar);
}




int main(int argc, char **argv) {

    double start_time = omp_get_wtime();

    double A1 = 0.0, B1 = 1.0;
    double A2 = -1.0, B2 = 1.0;
    int M = 10, N = 10;
    double tol = 1e-6;
    int maxIter = 400000;

    char *outfile_name;

    if (argc > 1) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }

    if (argc > 3) {
        outfile_name = argv[3];
    }

    double *x = (double*) calloc(M + 1, sizeof(double));
    double *y = (double*) calloc(N + 1, sizeof(double));

    double epsilon = makeGrid(M, N, A1, B1, A2, B2, x, y);


    double *w = (double*) calloc((M + 1) * (N + 1), sizeof(double));
    double *F = (double*) calloc(M * N, sizeof(double));
    double *a = (double*) calloc((M + 1) * (N + 1), sizeof(double));
    double *b = (double*) calloc((M + 1) * (N + 1), sizeof(double));

    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;

    calculateCoefficients(M, N, x, y, a, b, epsilon, h1, h2);

    calculateFValues(M, N, F, x, y, epsilon, h1, h2);

    steepestDescent(M, N, F, w, a, b, tol, maxIter, h1, h2);

    printf("Time taken is %f\n\n", omp_get_wtime() - start_time);

    saveMatrixToCSV(outfile_name, M, N, w);



    free(x);
    free(y);
    free(w);
    free(F);
    free(a);
    free(b);

    return 0;
}