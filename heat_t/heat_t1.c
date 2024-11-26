#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WALL_TEMP 20.0
#define FIREPLACE_TEMP 100.0

#define FIREPLACE_START 3
#define FIREPLACE_END 7
#define ROOM_SIZE 10

void initialize(double **h, int n) {
    int fireplace_start = (FIREPLACE_START * n) / ROOM_SIZE;
    int fireplace_end = (FIREPLACE_END * n) / ROOM_SIZE;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                if (i == n - 1 && j >= fireplace_start && j <= fireplace_end) {
                    h[i][j] = FIREPLACE_TEMP;
                } else {
                    h[i][j] = WALL_TEMP;
                }
            } else {
                h[i][j] = 0.0;
            }
        }
    }
}

void jacobi_iteration(double **h, double **g, int n, int max_iterations, double tolerance) {
    for (int iter = 0; iter < max_iterations; iter++) {
        double max_diff = 0.0;

        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                g[i][j] = 0.25 * (h[i - 1][j] + h[i + 1][j] + h[i][j - 1] + h[i][j + 1]);
                double diff = fabs(g[i][j] - h[i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        if (max_diff < tolerance) {
            break;
        }

        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                h[i][j] = g[i][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <tamanho_da_matriz> <max_iteracoes> <tolerancia>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int max_iterations = atoi(argv[2]);
    double tolerance = atof(argv[3]);

    double **h = (double **)malloc(n * sizeof(double *));
    double **g = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        h[i] = (double *)malloc(n * sizeof(double));
        g[i] = (double *)malloc(n * sizeof(double));
    }

    initialize(h, n);

    jacobi_iteration(h, g, n, max_iterations, tolerance);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < n; i++) {
        free(h[i]);
        free(g[i]);
    }
    free(h);
    free(g);

    return 0;
}
