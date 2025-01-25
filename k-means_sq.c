#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

typedef struct {
    double x, y;
} Point;

int K;
int max_iter;
Point* points;
Point* cntroid;
int n_ponts = 1000000;
double tolrnc = 1e-6;

void gen_rand_data() {
    points = (Point*)malloc(n_ponts * sizeof(Point));
    srand(time(NULL));
    for (int i = 0; i < n_ponts; i++) {
        points[i].x = (double)(rand() % 10001);
        points[i].y = (double)(rand() % 10001);
    }
}

void init_Cntroid() {
    cntroid = (Point*)malloc(K * sizeof(Point));
    int* chosen = (int*)calloc(n_ponts, sizeof(int));
    for (int i = 0; i < K; i++) {
        int index;
        do {
            index = rand() % n_ponts;
        } while (chosen[index]);
        cntroid[i] = points[index];
        chosen[index] = 1;
    }
    free(chosen);
}

double distance(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

int* ass_clustr() {
    int* clstr = (int*)malloc(n_ponts * sizeof(int));

    for (int i = 0; i < n_ponts; i++) {
        double minDist = DBL_MAX;
        int nrst_cntroid = -1;
        for (int j = 0; j < K; j++) {
            double dist = distance(points[i], cntroid[j]);
            if (dist < minDist) {
                minDist = dist;
                nrst_cntroid = j;
            }
        }
        clstr[i] = nrst_cntroid;
    }
    return clstr;
}

void updat_cntroid(int* clstr) {
    Point* new_cntroid = (Point*)calloc(K, sizeof(Point));
    int* counts = (int*)calloc(K, sizeof(int));

    for (int i = 0; i < n_ponts; i++) {
        int cluster = clstr[i];
        new_cntroid[cluster].x += points[i].x;
        new_cntroid[cluster].y += points[i].y;
        counts[cluster]++;
    }

    for (int i = 0; i < K; i++) {
        if (counts[i] > 0) {
            new_cntroid[i].x /= counts[i];
            new_cntroid[i].y /= counts[i];
        }
        cntroid[i] = new_cntroid[i];
    }

    free(new_cntroid);
    free(counts);
}

int clst_ing() {
    init_Cntroid();
    int* clstr = NULL;
    Point* old_cntroid = (Point*)malloc(K * sizeof(Point));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < K; i++) {
            old_cntroid[i] = cntroid[i];
        }

        clstr = ass_clustr();
        updat_cntroid(clstr);

        double mxcntrshft = 0.0;
        for (int i = 0; i < K; i++) {
            double dx = cntroid[i].x - old_cntroid[i].x;
            double dy = cntroid[i].y - old_cntroid[i].y;
            double shift = sqrt(dx * dx + dy * dy);
            mxcntrshft = fmax(mxcntrshft, shift);
        }

        if (mxcntrshft < tolrnc) {
            free(clstr);
            break;
        }

        free(clstr);
    }

    free(old_cntroid);
    return iter + 1;
}

int main() {
    gen_rand_data();
    max_iter = 1000;

    FILE* outputFile = fopen("clst_rslt_2D_Sq.csv", "w");

    fprintf(outputFile, "K,Exe_time,conve_itr\n");

    for (K = 2; K <= 1024; K *= 2) {
        for (int i = 0; i < 3; i++) {
            clock_t strt_time = clock();

            int conv_itratn = clst_ing();

            clock_t end_time = clock();
            double ext_time = (double)(end_time - strt_time) / CLOCKS_PER_SEC;

            fprintf(outputFile, "%d,%.6f,%d\n", K, ext_time, conv_itratn);
            fflush(outputFile);
            printf("K=%d completed in %.6f seconds, Converged in %d iterations\n", K, ext_time, conv_itratn);

            free(cntroid);
        }
    }

    fclose(outputFile);
    free(points);
    return 0;
}
