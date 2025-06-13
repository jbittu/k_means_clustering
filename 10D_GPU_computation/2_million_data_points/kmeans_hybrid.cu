#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_LINE_LEN 4096
#define MAX_DIM 10

// Global settings
int DIM = 10;
int num_points = 0, max_iter = 100, K = 4, NUM_THREADS = 4;
float *points = NULL;
char input_file[256] = "10D_data_1mil.csv";

// ---------------- GPU Distance ----------------

__device__ float distance(const float *a, const float *b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

__global__ void assign_labels(float *points, float *centroids, int *labels, int N, int K, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float min_dist = FLT_MAX;
    int label = 0;

    for (int c = 0; c < K; c++) {
        float d = distance(&points[i * dim], &centroids[c * dim], dim);
        if (d < min_dist) {
            min_dist = d;
            label = c;
        }
    }

    labels[i] = label;
}

// ---------------- CSV Reading ----------------

int count_columns(const char *line) {
    int count = 1;
    for (const char *p = line; *p; p++) {
        if (*p == ',') count++;
    }
    return count;
}

void read_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { perror("fopen"); exit(1); }

    char line[MAX_LINE_LEN];
    fgets(line, MAX_LINE_LEN, file);
    DIM = count_columns(line);

    while (fgets(line, MAX_LINE_LEN, file)) num_points++;
    rewind(file);
    fgets(line, MAX_LINE_LEN, file); // skip header

    points = (float *)malloc(num_points * DIM * sizeof(float));
    int i = 0;
    while (fgets(line, MAX_LINE_LEN, file)) {
        char *token = strtok(line, ",");
        for (int d = 0; d < DIM && token; d++) {
            points[i * DIM + d] = atof(token);
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(file);
}

// ---------------- CPU Helper ----------------

void compute_centroids(float *points, int *labels, float *centroids, int N, int K, int dim) {
    int *counts = (int *)calloc(K, sizeof(int));
    float *sums = (float *)calloc(K * dim, sizeof(float));

    for (int i = 0; i < N; i++) {
        int c = labels[i];
        counts[c]++;
        for (int d = 0; d < dim; d++) {
            sums[c * dim + d] += points[i * dim + d];
        }
    }

    for (int c = 0; c < K; c++) {
        if (counts[c] > 0) {
            for (int d = 0; d < dim; d++) {
                centroids[c * dim + d] = sums[c * dim + d] / counts[c];
            }
        }
    }

    free(counts);
    free(sums);
}

// ---------------- K-Means GPU ----------------

void kmeans_gpu(float *points_host, int N, int K, int max_iter, float tol, double *exec_time, int *conv_iter, int dim) {
    float *centroids = (float *)malloc(K * dim * sizeof(float));
    for (int i = 0; i < K; i++)
        memcpy(&centroids[i * dim], &points_host[i * dim], dim * sizeof(float));

    float *d_points, *d_centroids;
    int *d_labels, *labels = (int *)malloc(N * sizeof(int));

    cudaMalloc(&d_points, N * dim * sizeof(float));
    cudaMalloc(&d_centroids, K * dim * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMemcpy(d_points, points_host, N * dim * sizeof(float), cudaMemcpyHostToDevice);

    double start_time = omp_get_wtime();
    for (int it = 0; it < max_iter; it++) {
        cudaMemcpy(d_centroids, centroids, K * dim * sizeof(float), cudaMemcpyHostToDevice);

        assign_labels<<<(N + 255) / 256, 256>>>(d_points, d_centroids, d_labels, N, K, dim);
        cudaDeviceSynchronize();
        cudaMemcpy(labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        float *prev_centroids = (float *)malloc(K * dim * sizeof(float));
        memcpy(prev_centroids, centroids, K * dim * sizeof(float));

        compute_centroids(points_host, labels, centroids, N, K, dim);

        float max_shift = 0.0f;
        for (int i = 0; i < K; i++) {
            float shift = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = centroids[i * dim + d] - prev_centroids[i * dim + d];
                shift += diff * diff;
            }
            if (sqrtf(shift) > max_shift)
                max_shift = sqrtf(shift);
        }
        free(prev_centroids);

        if (max_shift < tol) {
            *conv_iter = it + 1;
            break;
        }
    }

    *exec_time = omp_get_wtime() - start_time;

    free(centroids);
    free(labels);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
}

// ---------------- Main ----------------

int main(int argc, char **argv) {
    if (argc >= 2) strcpy(input_file, argv[1]);
    if (argc >= 3) K = atoi(argv[2]);
    if (argc >= 4) NUM_THREADS = atoi(argv[3]);
    if (argc >= 5) max_iter = atoi(argv[4]);

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024);
    omp_set_num_threads(NUM_THREADS);
    read_csv(input_file);

    FILE *log = fopen("10D_1mil_sm_gpu_result.csv", "w");
    fprintf(log, "K,ExecTime,ConvergedIn,Dim\n");

    for (int current_K = 4; current_K <= K && current_K <= num_points; current_K *= 2) {
        double exec_time = 0.0;
        int conv_iter = 0;
        if (current_K > num_points) continue;

        kmeans_gpu(points, num_points, current_K, max_iter, 1e-4, &exec_time, &conv_iter, DIM);
        fprintf(log, "%d,%.6f,%d,%d\n", current_K, exec_time, conv_iter, DIM);
        printf("K=%d, Time=%.3f s, Iter=%d, Dim=%d\n", current_K, exec_time, conv_iter, DIM);
    }

    fclose(log);
    free(points);
    return 0;
}
