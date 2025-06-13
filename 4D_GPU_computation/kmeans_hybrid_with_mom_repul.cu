#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_LINE_LEN 1024

typedef struct {
    float x, y, z, t;
} Point;

__device__ __forceinline__ float distance(Point p1, Point p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    float dt = p1.t - p2.t;
    return sqrtf(dx * dx + dy * dy + dz * dz + dt * dt);
}

__global__ void assign_labels(Point *points, Point *centroids, int *labels, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float min_dist = FLT_MAX;
    int label = 0;

    for (int c = 0; c < K; c++) {
        float d = distance(points[i], centroids[c]);
        if (d < min_dist) {
            min_dist = d;
            label = c;
        }
    }

    labels[i] = label;
}

__global__ void compute_centroids_kernel(Point *points, int *labels, Point *centroid_sums, int *counts, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int label = labels[i];
    atomicAdd(&centroid_sums[label].x, points[i].x);
    atomicAdd(&centroid_sums[label].y, points[i].y);
    atomicAdd(&centroid_sums[label].z, points[i].z);
    atomicAdd(&centroid_sums[label].t, points[i].t);
    atomicAdd(&counts[label], 1);
}

__global__ void update_centroids_kernel(Point *centroids, Point *prev_centroids, Point *sums, int *counts, float alpha, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;

    Point mean = centroids[i];
    if (counts[i] > 0) {
        mean.x = sums[i].x / counts[i];
        mean.y = sums[i].y / counts[i];
        mean.z = sums[i].z / counts[i];
        mean.t = sums[i].t / counts[i];
    }

    centroids[i].x = alpha * prev_centroids[i].x + (1.0f - alpha) * mean.x;
    centroids[i].y = alpha * prev_centroids[i].y + (1.0f - alpha) * mean.y;
    centroids[i].z = alpha * prev_centroids[i].z + (1.0f - alpha) * mean.z;
    centroids[i].t = alpha * prev_centroids[i].t + (1.0f - alpha) * mean.t;
}

__global__ void apply_repulsion_kernel(Point *centroids, int K, float repulsion_strength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;

    for (int j = 0; j < K; j++) {
        if (i == j) continue;
        float dx = centroids[i].x - centroids[j].x;
        float dy = centroids[i].y - centroids[j].y;
        float dz = centroids[i].z - centroids[j].z;
        float dt = centroids[i].t - centroids[j].t;
        float dist_sq = dx * dx + dy * dy + dz * dz + dt * dt + 1e-6f;
        float factor = repulsion_strength / dist_sq;

        centroids[i].x += factor * dx;
        centroids[i].y += factor * dy;
        centroids[i].z += factor * dz;
        centroids[i].t += factor * dt;
    }
}

int num_points = 0, max_iter = 100, K = 4, NUM_THREADS = 4;
Point *points = NULL;
char input_file[256] = "4d_data_1mil.csv";

void read_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { perror("fopen"); exit(1); }

    char line[MAX_LINE_LEN];
    fgets(line, MAX_LINE_LEN, file);
    while (fgets(line, MAX_LINE_LEN, file)) num_points++;

    rewind(file); fgets(line, MAX_LINE_LEN, file);

    points = (Point *)malloc(num_points * sizeof(Point));
    int i = 0;
    while (fgets(line, MAX_LINE_LEN, file)) {
        sscanf(line, "%f,%f,%f,%f", &points[i].x, &points[i].y, &points[i].z, &points[i].t);
        i++;
    }
    fclose(file);
}

void kmeans_gpu(Point *points_host, int N, int K, int max_iter, float tol, double *exec_time, int *conv_iter, float alpha, float repulsion_strength) {
    if (K > num_points) {
        fprintf(stderr, "ERROR: K=%d > N=%d\n", K, num_points);
        exit(EXIT_FAILURE);
    }

    Point *centroids = (Point *)malloc(K * sizeof(Point));
    Point *prev_centroids = (Point *)malloc(K * sizeof(Point));
    for (int i = 0; i < K; i++) centroids[i] = points_host[i];

    Point *d_points, *d_centroids, *d_prev_centroids, *d_sums;
    int *d_labels, *labels = (int *)malloc(N * sizeof(int));
    int *d_counts;

    cudaMalloc(&d_points, N * sizeof(Point));
    cudaMalloc(&d_centroids, K * sizeof(Point));
    cudaMalloc(&d_prev_centroids, K * sizeof(Point));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_sums, K * sizeof(Point));
    cudaMalloc(&d_counts, K * sizeof(int));

    cudaMemcpy(d_points, points_host, N * sizeof(Point), cudaMemcpyHostToDevice);

    double start_time = omp_get_wtime();

    for (int it = 0; it < max_iter; it++) {
        cudaMemcpy(d_prev_centroids, centroids, K * sizeof(Point), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroids, centroids, K * sizeof(Point), cudaMemcpyHostToDevice);

        assign_labels<<<(N + 255) / 256, 256>>>(d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();

        cudaMemset(d_sums, 0, K * sizeof(Point));
        cudaMemset(d_counts, 0, K * sizeof(int));

        compute_centroids_kernel<<<(N + 255) / 256, 256>>>(d_points, d_labels, d_sums, d_counts, N);
        cudaDeviceSynchronize();

        update_centroids_kernel<<<(K + 31) / 32, 32>>>(d_centroids, d_prev_centroids, d_sums, d_counts, alpha, K);
        cudaDeviceSynchronize();

        apply_repulsion_kernel<<<(K + 31) / 32, 32>>>(d_centroids, K, repulsion_strength);
        cudaDeviceSynchronize();

        cudaMemcpy(centroids, d_centroids, K * sizeof(Point), cudaMemcpyDeviceToHost);
        cudaMemcpy(prev_centroids, d_prev_centroids, K * sizeof(Point), cudaMemcpyDeviceToHost);

        float max_shift = 0.0f;
        for (int i = 0; i < K; i++) {
            float dx = centroids[i].x - prev_centroids[i].x;
            float dy = centroids[i].y - prev_centroids[i].y;
            float dz = centroids[i].z - prev_centroids[i].z;
            float dt = centroids[i].t - prev_centroids[i].t;
            float shift = sqrtf(dx * dx + dy * dy + dz * dz + dt * dt);
            if (shift > max_shift) max_shift = shift;
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

    FILE *log = fopen("4d_1mil_gpu_moum_repul_result.csv", "w");
    fprintf(log, "K,ExecTime,ConvergedIn,Alpha,RepulsionStrength\n");

    for (int current_K = 2; current_K <= K && current_K <= num_points; current_K *= 2) {
        double exec_time = 0.0;
        int conv_iter = 0;

        kmeans_gpu(points, num_points, current_K, max_iter, 1e-4, &exec_time, &conv_iter, alpha, repulsion_strength);
        fprintf(log, "%d,%.6f,%d,%.2f,%.3f\n", current_K, exec_time, conv_iter, alpha, repulsion_strength);
        printf("K=%d, Time=%.3f s, Iter=%d, a=%.2f, ?=%.3f\n", current_K, exec_time, conv_iter, alpha, repulsion_strength);
    }

    fclose(log);
    free(points);
    return 0;
}
