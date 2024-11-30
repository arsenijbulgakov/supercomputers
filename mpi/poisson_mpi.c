#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "mpi.h"


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


double makeGrid(double h1, double h2, int M, int N, double A1, double A2, double *x, double *y, int grid_offset_x, int grid_offset_y) {

    for (int i = 0; i <= M; ++i) {
        x[i] = A1 + i * h1;
    }


    for (int i = 0; i <= N; ++i) {
        y[i] = A2 + i * h2;
    }

    double h = fmax(h1, h2);

    return h*h;
}


void calculateFValues(int local_M, int local_N, int grid_offset_x, int grid_offset_y, int M, int N, double *F, const double *x, const double *y, double epsilon, double h1, double h2) {

    int start_i = grid_offset_x;
    int start_j = grid_offset_y;

    if (start_i == 0) {
        start_i = 1;
    }

    if (start_j == 0) {
        start_j = 1;
    }

    for (int i = start_i; i < local_M + grid_offset_x && i < M; ++i) {
        for (int j = start_j; j < local_N + grid_offset_y && j < N; ++j) {
            int idx = i * (N) + j;

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


void calculateCoefficients(int local_M, int local_N, int grid_offset_x, int grid_offset_y, int M, int N, const double *x, const double *y, double *a, double *b, double epsilon, double h1, double h2) {

    int start_i = grid_offset_x;
    int start_j = grid_offset_y;

    if (start_i == 0) {
        start_i = 1;
    }

    if (start_j == 0) {
        start_j = 1;
    }


    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            int idx = i * (N) + j;

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


void print_matrix(double *matrix, int M, int N) {
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            int idx = i * (N) + j;
            printf("%.2f", matrix[idx]);
            if (j < M) {
                printf(" ");
            }
        }
        printf("\n");
    }
}



void steepestDescent(int rank, int local_M, int local_N, int grid_offset_x, int grid_offset_y, int M, int N, const double *F, double *w, const double *a, const double *b, double tol, int maxIter, double h1, double h2) {
    int totalPoints = (M + 1) * (N + 1);
    double *r = (double*) calloc((M + 1) * (N + 1), sizeof(double));
    double *Ar = (double*) calloc((M + 1) * (N + 1), sizeof(double));

    double *r_local = (double*) calloc((M + 1) * (N + 1), sizeof(double));
    double *Ar_local = (double*) calloc((M + 1) * (N + 1), sizeof(double));
    double *w_local = (double*) calloc((M + 1) * (N + 1), sizeof(double));


    int start_i = grid_offset_x;
    int start_j = grid_offset_y;

    if (start_i == 0) {
        start_i = 1;
    }

    if (start_j == 0) {
        start_j = 1;
    }

    for (int iter = 0; iter < maxIter; ++iter) {

        for (int i = start_i; i < local_M + grid_offset_x && i <= M - 1; ++i) {
            for (int j = start_j; j < local_N + grid_offset_y && j <= N - 1; ++j) {
                int idx = i * (N) + j;
                r_local[idx] = ( -1.0 / h1 * (a[idx+N]*(w[idx+N] - w[idx]) / h1 - a[idx]*(w[idx] - w[idx-N]) / h1) -
                            1.0 / h2 * (b[idx+1]*(w[idx+1] - w[idx]) / h2 - b[idx]*(w[idx] - w[idx-1]) / h2)
                           ) - F[idx];
            }
        }

        MPI_Allreduce(r_local, r, (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int i = start_i; i < local_M + grid_offset_x && i <= M - 1; ++i) {
            for (int j = start_j; j < local_N + grid_offset_y && j <= N - 1; ++j) {
                int idx = i * (N) + j;
                Ar_local[idx] = ( -1.0 / h1 * (a[idx+N]*(r[idx+N] - r[idx]) / h1 - a[idx]*(r[idx] - r[idx-N]) / h1) -
                             1.0 / h2 * (b[idx+1]*(r[idx+1] - r[idx]) / h2 - b[idx]*(r[idx] - r[idx-1]) / h2)
                           );
            }
        }

        MPI_Allreduce(Ar_local, Ar, (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        double rr_local = 0, Ar_r_local = 0;
        double rr = 0, Ar_r = 0;

        for (int i = start_i; i < local_M + grid_offset_x && i <= M - 1; ++i) {
            for (int j = start_j; j < local_N + grid_offset_y && j <= N - 1; ++j) {
                int idx = i * (N) + j;
                rr_local += r[idx] * r[idx];
                Ar_r_local += Ar[idx] * r[idx];
            }
        }

        MPI_Allreduce(&Ar_r_local, &Ar_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        double tau = rr / Ar_r;

        for (int i = start_i; i < local_M + grid_offset_x && i <= M - 1; ++i) {
            for (int j = start_j; j < local_N + grid_offset_y && j <= N - 1; ++j) {
                int idx = i * (N) + j;
                w_local[idx] = w[idx] - tau * r[idx];
            }
        }

        MPI_Allreduce(w_local, w, (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


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

    int p_x;
    int p_y;

    int rank, size;

    double A1 = 0.0, B1 = 1.0;
    double A2 = -1.0, B2 = 1.0;
    int M = 10, N = 10;
    double tol = 1e-6;
    int maxIter = 400000;

    if (argc > 1) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }


    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;

    char *outfile_name;

    if (argc > 3) {
        outfile_name = argv[3];
    }

    double start_time;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) start_time = MPI_Wtime();

    int min_diff = M + N;
    for (int div = 1; div <= sqrt(size); div++) {
        if (size % div == 0) {
            int cur_p_x = div;
            int cur_p_y = size / div;

            int diff = abs((M / cur_p_x) - (N / cur_p_y));
            if (diff < min_diff) {
                p_x = cur_p_x;
                p_y = cur_p_y;
                min_diff = diff;
            }
        }
    }
    printf("%d  %d\n\n", p_x, p_y);


    int x_index = rank % p_x;
    int y_index = rank / p_x;

    int *domain_sizes_x = calloc(p_x, sizeof(int));
    int *domain_sizes_y = calloc(p_y, sizeof(int));



    int base_x = (M+1) / p_x;
    int remainder_x = (M+1) % p_x;
    for (int i = 0; i < p_x; i++) {
        domain_sizes_x[i] = base_x;
        if (i < remainder_x) {
            domain_sizes_x[i]++;
        }
    }

    int base_y = (N+1) / p_y;
    int remainder_y = (N+1) % p_y;
    for (int i = 0; i < p_y; i++) {
        domain_sizes_y[i] = base_y;
        if (i < remainder_y) {
            domain_sizes_y[i]++;
        }
    }

    int local_M = domain_sizes_x[x_index];
    int local_N = domain_sizes_y[y_index];


    int grid_offset_x = 0;
    int grid_offset_y = 0;
    for (int i = 0; i < x_index; i++) grid_offset_x += domain_sizes_x[i];
    for (int j = 0; j < y_index; j++) grid_offset_y += domain_sizes_y[j];



    double *x = (double*) calloc(M + 1, sizeof(double));
    double *y = (double*) calloc(N + 1, sizeof(double));

    double epsilon = makeGrid(h1, h2, M, N, A1, A2, x, y, grid_offset_x, grid_offset_y);


    double *w = (double*) calloc((M + 1) * (N + 1), sizeof(double));

    double *F = (double*) calloc(M * N, sizeof(double));
    double *a = (double*) calloc((M + 1) * (N + 1), sizeof(double));
    double *b = (double*) calloc((M + 1) * (N + 1), sizeof(double));


    calculateCoefficients(local_M, local_N, grid_offset_x, grid_offset_y, M, N, x, y, a, b, epsilon, h1, h2);
    calculateFValues(local_M, local_N, grid_offset_x, grid_offset_y, M, N, F, x, y, epsilon, h1, h2);
    steepestDescent(rank, local_M, local_N, grid_offset_x, grid_offset_y, M, N, F, w, a, b, tol, maxIter, h1, h2);


    if (rank == 0) {
        printf("Time taken is %f\n\n", MPI_Wtime() - start_time);
        saveMatrixToCSV(outfile_name, M, N, w);
    }

    MPI_Finalize();

    return 0;
}
