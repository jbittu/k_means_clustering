#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_LINE_LEN 1024
#define MAX_DIM 4

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

int num_points = 0, max_iter = 100, K = 2, NUM_THREADS = 5;
Point *points = NULL;
char input_file[256] = "4d_data_1mil.csv";

void read_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { perror("fopen"); exit(1); }

    char line[MAX_LINE_LEN];
    fgets(line, MAX_LINE_LEN, file); // skip header
    while (fgets(line, MAX_LINE_LEN, file)) num_points++;

    rewind(file); fgets(line, MAX_LINE_LEN, file); // skip header again

    points = (Point *)malloc(num_points * sizeof(Point));
    int i = 0;
    while (fgets(line, MAX_LINE_LEN, file)) {
        sscanf(line, "%f,%f,%f,%f", &points[i].x, &points[i].y, &points[i].z, &points[i].t);
        i++;
    }
    fclose(file);
}

void compute_centroids(Point *points, int *labels, Point *centroids, int N, int K) {
    int *counts = (int *)calloc(K, sizeof(int));
    Point *sums = (Point *)calloc(K, sizeof(Point));

    for (int i = 0; i < N; i++) {
        int c = labels[i];
        if (c < 0 || c >= K) {
            fprintf(stderr, "ERROR: Invalid cluster label %d at index %d\n", c, i);
            exit(EXIT_FAILURE);
        }
        sums[c].x += points[i].x;
        sums[c].y += points[i].y;
        sums[c].z += points[i].z;
        sums[c].t += points[i].t;
        counts[c]++;
    }

    for (int i = 0; i < K; i++) {
        if (counts[i] > 0) {
            centroids[i].x = sums[i].x / counts[i];
            centroids[i].y = sums[i].y / counts[i];
            centroids[i].z = sums[i].z / counts[i];
            centroids[i].t = sums[i].t / counts[i];
        }
    }

    free(sums); free(counts);
}

void kmeans_gpu(Point *points_host, int N, int K, int max_iter, float tol, double *exec_time, int *conv_iter) {
    if (K > num_points) {
        fprintf(stderr, "ERROR: Number of clusters (K=%d) cannot exceed number of data points (%d).\n", K, num_points);
        exit(EXIT_FAILURE);
    }

    Point *centroids = (Point *)malloc(K * sizeof(Point));
    for (int i = 0; i < K; i++) centroids[i] = points_host[i];

    Point *d_points, *d_centroids;
    int *d_labels, *labels = (int *)malloc(N * sizeof(int));

    cudaMalloc(&d_points, N * sizeof(Point));
    cudaMalloc(&d_centroids, K * sizeof(Point));
    cudaMalloc(&d_labels, N * sizeof(int));

    cudaMemcpy(d_points, points_host, N * sizeof(Point), cudaMemcpyHostToDevice);

    double start_time = omp_get_wtime();

    for (int it = 0; it < max_iter; it++) {
        cudaMemcpy(d_centroids, centroids, K * sizeof(Point), cudaMemcpyHostToDevice);

        assign_labels<<<(N + 255) / 256, 256>>>(d_points, d_centroids, d_labels, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
        cudaMemcpy(labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        // ? Only validate labels after copying from GPU
        for (int i = 0; i < N; i++) {
            if (labels[i] < 0 || labels[i] >= K) {
                fprintf(stderr, "ERROR: Invalid cluster label %d at index %d (K=%d)\n", labels[i], i, K);
                exit(EXIT_FAILURE);
            }
        }

        Point *prev_centroids = (Point *)malloc(K * sizeof(Point));
        memcpy(prev_centroids, centroids, K * sizeof(Point));

        compute_centroids(points_host, labels, centroids, N, K);

        float max_shift = 0.0f;
        for (int i = 0; i < K; i++) {
            float dx = centroids[i].x - prev_centroids[i].x;
            float dy = centroids[i].y - prev_centroids[i].y;
            float dz = centroids[i].z - prev_centroids[i].z;
            float dt = centroids[i].t - prev_centroids[i].t;
            float shift = sqrtf(dx*dx + dy*dy + dz*dz + dt*dt);
            if (shift > max_shift) max_shift = shift;
        }

        free(prev_centroids);
        if (max_shift < tol) {
            *conv_iter = it + 1;
            break;
        }
    }

    *exec_time = omp_get_wtime() - start_time;

    free(centroids); free(labels);
    cudaFree(d_points); cudaFree(d_centroids); cudaFree(d_labels);
}

int main(int argc, char **argv) {
    if (argc >= 2) strcpy(input_file, argv[1]);
    if (argc >= 3) K = atoi(argv[2]);
    if (argc >= 4) NUM_THREADS = atoi(argv[3]);
    if (argc >= 5) max_iter = atoi(argv[4]);

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024); // optional: increase printf buffer
    omp_set_num_threads(NUM_THREADS);

    read_csv(input_file);

    FILE *log = fopen("4d_1mil_sm_gpu_result.csv", "w");
    fprintf(log, "K,ExecTime,ConvergedIn\n");

    for (int current_K = 2; current_K <= K && current_K <= num_points; current_K *= 2) {
        if (current_K >= num_points) {
            fprintf(stderr, "K=%d is greater than number of data points (%d). Skipping.\n", current_K, num_points);
            continue;
        }

        double exec_time = 0.0;
        int conv_iter = 0;

        kmeans_gpu(points, num_points, current_K, max_iter, 1e-4, &exec_time, &conv_iter);
        fprintf(log, "%d,%.6f,%d\n", current_K, exec_time, conv_iter);
        printf("K=%d, Time=%.3f s, Iter=%d\n", current_K, exec_time, conv_iter);
    }

    fclose(log);
    free(points);
    return 0;
}
