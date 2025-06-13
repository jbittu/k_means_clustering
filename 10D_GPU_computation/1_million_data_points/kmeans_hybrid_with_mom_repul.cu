#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_LINE_LEN 1024

int num_points = 0, max_iter = 100, K = 4, NUM_THREADS = 4, D = 0;
float *points = NULL;
char input_file[256] = "10D_data_1mil.csv";

void read_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { perror("fopen"); exit(1); }

    char line[MAX_LINE_LEN];
    fgets(line, MAX_LINE_LEN, file);

    char *token = strtok(line, ",");
    while (token) {
        D++;
        token = strtok(NULL, ",");
    }

    rewind(file); fgets(line, MAX_LINE_LEN, file);
    while (fgets(line, MAX_LINE_LEN, file)) num_points++;

    rewind(file); fgets(line, MAX_LINE_LEN, file);
    points = (float *)malloc(num_points * D * sizeof(float));

    int i = 0;
    while (fgets(line, MAX_LINE_LEN, file)) {
        token = strtok(line, ",");
        for (int d = 0; d < D && token; d++) {
            points[i * D + d] = atof(token);
            token = strtok(NULL, ",");
        }
        i++;
    }

    fclose(file);
}

__global__ void assign_labels(const float *points, const float *centroids, int *labels, int N, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float min_dist = FLT_MAX;
    int label = 0;

    for (int c = 0; c < K; c++) {
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = points[i * D + d] - centroids[c * D + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            label = c;
        }
    }

    labels[i] = label;
}

__global__ void compute_centroids_kernel(const float *points, const int *labels, float *sums, int *counts, int N, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int label = labels[i];
    for (int d = 0; d < D; d++) {
        atomicAdd(&sums[label * D + d], points[i * D + d]);
    }
    atomicAdd(&counts[label], 1);
}

__global__ void update_centroids_kernel(float *centroids, const float *prev_centroids, const float *sums, const int *counts, float alpha, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;

    for (int d = 0; d < D; d++) {
        float mean = (counts[i] > 0) ? (sums[i * D + d] / counts[i]) : centroids[i * D + d];
        centroids[i * D + d] = alpha * prev_centroids[i * D + d] + (1.0f - alpha) * mean;
    }
}

__global__ void apply_repulsion_kernel(float *centroids, int K, int D, float repulsion_strength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;

    for (int j = 0; j < K; j++) {
        if (i == j) continue;

        float dist_sq = 1e-6f;
        for (int d = 0; d < D; d++) {
            float diff = centroids[i * D + d] - centroids[j * D + d];
            dist_sq += diff * diff;
        }

        float factor = repulsion_strength / dist_sq;
        for (int d = 0; d < D; d++) {
            float diff = centroids[i * D + d] - centroids[j * D + d];
            centroids[i * D + d] += factor * diff;
        }
    }
}

void kmeans_gpu(float *points_host, int N, int K, int max_iter, float tol, double *exec_time, int *conv_iter, float alpha, float repulsion_strength) {
    float *centroids = (float *)malloc(K * D * sizeof(float));
    float *prev_centroids = (float *)malloc(K * D * sizeof(float));
    int *labels = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < K * D; i++) centroids[i] = points_host[i];

    float *d_points, *d_centroids, *d_prev_centroids, *d_sums;
    int *d_labels, *d_counts;

    cudaMalloc(&d_points, N * D * sizeof(float));
    cudaMalloc(&d_centroids, K * D * sizeof(float));
    cudaMalloc(&d_prev_centroids, K * D * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_sums, K * D * sizeof(float));
    cudaMalloc(&d_counts, K * sizeof(int));

    cudaMemcpy(d_points, points_host, N * D * sizeof(float), cudaMemcpyHostToDevice);

    double start_time = omp_get_wtime();

    for (int it = 0; it < max_iter; it++) {
        cudaMemcpy(d_prev_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

        assign_labels<<<(N + 255) / 256, 256>>>(d_points, d_centroids, d_labels, N, K, D);
        cudaDeviceSynchronize();

        cudaMemset(d_sums, 0, K * D * sizeof(float));
        cudaMemset(d_counts, 0, K * sizeof(int));

        compute_centroids_kernel<<<(N + 255) / 256, 256>>>(d_points, d_labels, d_sums, d_counts, N, K, D);
        cudaDeviceSynchronize();

        update_centroids_kernel<<<(K + 31) / 32, 32>>>(d_centroids, d_prev_centroids, d_sums, d_counts, alpha, K, D);
        cudaDeviceSynchronize();

        apply_repulsion_kernel<<<(K + 31) / 32, 32>>>(d_centroids, K, D, repulsion_strength);
        cudaDeviceSynchronize();

        cudaMemcpy(centroids, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(prev_centroids, d_prev_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);

        float max_shift = 0.0f;
        for (int i = 0; i < K; i++) {
            float shift = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = centroids[i * D + d] - prev_centroids[i * D + d];
                shift += diff * diff;
            }
            max_shift = fmaxf(max_shift, sqrtf(shift));
        }

        if (max_shift < tol) {
            *conv_iter = it + 1;
            break;
        }
    }

    *exec_time = omp_get_wtime() - start_time;

    free(centroids);
    free(prev_centroids);
    free(labels);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_prev_centroids);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);
}

int main(int argc, char **argv) {
    if (argc >= 2) strcpy(input_file, argv[1]);
    if (argc >= 3) K = atoi(argv[2]);
    if (argc >= 4) NUM_THREADS = atoi(argv[3]);
    if (argc >= 5) max_iter = atoi(argv[4]);

    float alpha = 0.5f;
    float repulsion_strength = 0.01f;
    if (argc >= 6) alpha = atof(argv[5]);
    if (argc >= 7) repulsion_strength = atof(argv[6]);

    omp_set_num_threads(NUM_THREADS);
    read_csv(input_file);

    FILE *log = fopen("10d_with_mum_rupel_gpu_results.csv", "w");
    fprintf(log, "K,ExecTime,ConvergedIn,Alpha,RepulsionStrength\n");

    for (int current_K = 2; current_K <= K && current_K <= num_points; current_K *= 2) {
        double exec_time = 0.0;
        int conv_iter = 0;

        kmeans_gpu(points, num_points, current_K, max_iter, 1e-4, &exec_time, &conv_iter, alpha, repulsion_strength);
        fprintf(log, "%d,%.6f,%d,%.2f,%.3f\n", current_K, exec_time, conv_iter, alpha, repulsion_strength);
        printf("K=%d, Time=%.3f s, Iter=%d, a=%.2f, r=%.3f\n", current_K, exec_time, conv_iter, alpha, repulsion_strength);
    }

    fclose(log);
    free(points);
    return 0;
}
