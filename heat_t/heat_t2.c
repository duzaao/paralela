#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define WALL_TEMP 20.0
#define FIREPLACE_TEMP 100.0

#define FIREPLACE_START 3
#define FIREPLACE_END 7
#define ROOM_SIZE 10

__global__ void initialize_kernel(double *h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int fireplace_start = (FIREPLACE_START * n) / ROOM_SIZE;
    int fireplace_end = (FIREPLACE_END * n) / ROOM_SIZE;

    if (i < n && j < n) {
        if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
            if (i == n - 1 && j >= fireplace_start && j <= fireplace_end) {
                h[i * n + j] = FIREPLACE_TEMP;
            } else {
                h[i * n + j] = WALL_TEMP;
            }
        } else {
            h[i * n + j] = 0.0;
        }
    }
}

__global__ void jacobi_kernel(double *h, double *g, int n, double *max_diff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        g[i * n + j] = 0.25 * (h[(i - 1) * n + j] + h[(i + 1) * n + j] +
                               h[i * n + (j - 1)] + h[i * n + (j + 1)]);
        double diff = fabs(g[i * n + j] - h[i * n + j]);
        atomicMax((int *)max_diff, __double_as_int(diff));
    }
}

void initialize(double *h, int n) {
    double *d_h;
    size_t size = n * n * sizeof(double);
    cudaMalloc(&d_h, size);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initialize_kernel<<<numBlocks, threadsPerBlock>>>(d_h, n);
    cudaMemcpy(h, d_h, size, cudaMemcpyDeviceToHost);

    cudaFree(d_h);
}

void jacobi_iteration(double *h, int n, int max_iterations, double tolerance) {
    double *d_h, *d_g, *d_max_diff;
    size_t size = n * n * sizeof(double);
    cudaMalloc(&d_h, size);
    cudaMalloc(&d_g, size);
    cudaMemcpy(d_h, h, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_max_diff, sizeof(double));
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int iter = 0; iter < max_iterations; iter++) {
        double max_diff = 0.0;
        cudaMemcpy(d_max_diff, &max_diff, sizeof(double), cudaMemcpyHostToDevice);

        jacobi_kernel<<<numBlocks, threadsPerBlock>>>(d_h, d_g, n, d_max_diff);
        cudaMemcpy(&max_diff, d_max_diff, sizeof(double), cudaMemcpyDeviceToHost);

        if (max_diff < tolerance) break;

        double *temp = d_h;
        d_h = d_g;
        d_g = temp;
    }

    cudaMemcpy(h, d_h, size, cudaMemcpyDeviceToHost);

    cudaFree(d_h);
    cudaFree(d_g);
    cudaFree(d_max_diff);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <tamanho_da_matriz> <max_iteracoes> <tolerancia>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int max_iterations = atoi(argv[2]);
    double tolerance = atof(argv[3]);

    double *h = (double *)malloc(n * n * sizeof(double));

    initialize(h, n);

    jacobi_iteration(h, n, max_iterations, tolerance);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h[i * n + j]);
        }
        printf("\n");
    }

    free(h);

    return 0;
}
