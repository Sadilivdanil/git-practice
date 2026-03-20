#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

void my_gemmtr_lower(int n, int k, float a, float* A, float* B, float b, float* C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int kk = 0; kk < k; kk++)
                s += A[i * k + kk] * B[kk * n + j];
            C[i * n + j] = a * s + b * C[i * n + j];
        }
}

int test(int n, int k, float a, float* A, float* B, float b, float* C) {
    my_gemmtr_lower(n, k, a, A, B, b, C);
    return 1;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void run() {
    int n = 2048, k = 2048;
    float *A, *B, *C1, *C2;
    
    A = (float*)malloc(n * k * sizeof(float));
    B = (float*)malloc(k * n * sizeof(float));
    C1 = (float*)malloc(n * n * sizeof(float));
    C2 = (float*)malloc(n * n * sizeof(float));
    
    for (int i = 0; i < n * k; i++) A[i] = 1;
    for (int i = 0; i < k * n; i++) B[i] = 1;
    for (int i = 0; i < n * n; i++) C1[i] = 0;
    for (int i = 0; i < n * n; i++) C2[i] = 0;
    
    float a = 1, b = 0;
    int thr[] = {1, 2, 4, 8, 16};
    
    printf("Тест производительности\n");
    
    for (int ti = 0; ti < 5; ti++) {
        int t = thr[ti];
        openblas_set_num_threads(t);
        double geom = 1;
        
        printf("\nПотоков: %d\n", t);
        
        for (int r = 0; r < 10; r++) {
            double t1, t2, tob, tmy, perf;
            
            t1 = get_time();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        n, n, k, a, A, k, B, n, b, C2, n);
            t2 = get_time();
            tob = t2 - t1;
            
            t1 = get_time();
            test(n, k, a, A, B, b, C1);
            t2 = get_time();
            tmy = t2 - t1;
            
            perf = (tob / tmy) * 100;
            geom *= perf;
            
            printf("  %d: My=%.3fs OB=%.3fs %.2f%%\n", r+1, tmy, tob, perf);
        }
        
        printf("Ср.геом: %.2f%%\n", pow(geom, 0.1));
    }
    
    free(A); free(B); free(C1); free(C2);
}

int main() { run(); return 0; }