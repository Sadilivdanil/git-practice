#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <setjmp.h>
#include <signal.h>

int passed = 0;
int failed = 0;
jmp_buf env;

void segfault_handler(int sig) {
    printf("[SEGFAULT] Тест упал с segmentation fault\n");
    failed++;
    longjmp(env, 1);
}

void check_f(const char *name, float expected, float got) {
    if (fabsf(expected - got) < 1e-4f) {
        printf("[ПРОЙДЕН] %s\n", name);
        passed++;
    } else {
        printf("[НЕ ПРОЙДЕН] %s: %.4f != %.4f\n", name, expected, got);
        failed++;
    }
}

void check_d(const char *name, double expected, double got) {
    if (fabs(expected - got) < 1e-9) {
        printf("[ПРОЙДЕН] %s\n", name);
        passed++;
    } else {
        printf("[НЕ ПРОЙДЕН] %s: %.6f != %.6f\n", name, expected, got);
        failed++;
    }
}

#define RUN_TEST(test) do { \
    printf("\n--- Запуск %s ---\n", #test); \
    if (setjmp(env) == 0) { \
        test(); \
    } else { \
        printf("[ПРОПУЩЕН] %s пропущен из-за ошибки\n", #test); \
    } \
} while(0)

// ТЕСТ 1: СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ (float)
void test_sdot() {
    float x[] = {1,2,3};
    float y[] = {4,5,6};
    check_f("sdot", 32, cblas_sdot(3, x, 1, y, 1));
    
    float x2[] = {2,3};
    float y2[] = {5,7};
    check_f("sdot2", 2*5 + 3*7, cblas_sdot(2, x2, 1, y2, 1));
}

// ТЕСТ 2: СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ (double)
void test_ddot() {
    double x[] = {1,2,3};
    double y[] = {4,5,6};
    check_d("ddot", 32, cblas_ddot(3, x, 1, y, 1));
    
    double x2[] = {2,3};
    double y2[] = {5,7};
    check_d("ddot2", 2*5 + 3*7, cblas_ddot(2, x2, 1, y2, 1));
}

// ТЕСТ 3: НОРМА (float)
void test_snrm2() {
    float x[] = {3,4};
    check_f("snrm2", 5, cblas_snrm2(2, x, 1));
    
    float x2[] = {5,12};
    check_f("snrm2_2", 13, cblas_snrm2(2, x2, 1));
}

// ТЕСТ 4: НОРМА (double)
void test_dnrm2() {
    double x[] = {3,4};
    check_d("dnrm2", 5, cblas_dnrm2(2, x, 1));
    
    double x2[] = {5,12};
    check_d("dnrm2_2", 13, cblas_dnrm2(2, x2, 1));
}

// ТЕСТ 5: СУММА МОДУЛЕЙ (float)
void test_sasum() {
    float x[] = {1,-2,3};
    check_f("sasum", 6, cblas_sasum(3, x, 1));
    
    float x2[] = {-4,5,-6};
    check_f("sasum2", 15, cblas_sasum(3, x2, 1));
}

// ТЕСТ 6: СУММА МОДУЛЕЙ (double)
void test_dasum() {
    double x[] = {1,-2,3};
    check_d("dasum", 6, cblas_dasum(3, x, 1));
    
    double x2[] = {-4,5,-6};
    check_d("dasum2", 15, cblas_dasum(3, x2, 1));
}

// ТЕСТ 7: ИНДЕКС МАКСИМУМА (float)
void test_isamax() {
    float x[] = {1,-5,3};
    check_f("isamax_val", 5, fabs(x[cblas_isamax(3, x, 1)]));
    
    float x2[] = {2,8,-3,4};
    check_f("isamax_val2", 8, fabs(x2[cblas_isamax(4, x2, 1)]));
}

// ТЕСТ 8: ИНДЕКС МАКСИМУМА (double)
void test_idamax() {
    double x[] = {1,2,-7,3};
    check_d("idamax_val", 7, fabs(x[cblas_idamax(4, x, 1)]));
    
    double x2[] = {5,-9,2,6};
    check_d("idamax_val2", 9, fabs(x2[cblas_idamax(4, x2, 1)]));
}

// ТЕСТ 9: КОПИРОВАНИЕ (float)
void test_scopy() {
    float x[] = {1,2,3};
    float y[3] = {0};
    cblas_scopy(3, x, 1, y, 1);
    check_f("scopy[0]", 1, y[0]);
    check_f("scopy[2]", 3, y[2]);
}

// ТЕСТ 10: КОПИРОВАНИЕ (double)
void test_dcopy() {
    double x[] = {4,5,6};
    double y[3] = {0};
    cblas_dcopy(3, x, 1, y, 1);
    check_d("dcopy[0]", 4, y[0]);
    check_d("dcopy[2]", 6, y[2]);
}

// ТЕСТ 11: ОБМЕН (float)
void test_sswap() {
    float x[] = {1,2};
    float y[] = {9,8};
    cblas_sswap(2, x, 1, y, 1);
    check_f("sswap x[0]", 9, x[0]);
    check_f("sswap y[0]", 1, y[0]);
}

// ТЕСТ 12: AXPY (float)
void test_saxpy() {
    float x[] = {1,2,3};
    float y[] = {4,5,6};
    cblas_saxpy(3, 2, x, 1, y, 1);
    check_f("saxpy[0]", 6, y[0]);
    check_f("saxpy[2]", 12, y[2]);
}

// ТЕСТ 13: МАСШТАБИРОВАНИЕ (float)
void test_sscal() {
    float x[] = {1,2,3};
    cblas_sscal(3, 3, x, 1);
    check_f("sscal[0]", 3, x[0]);
    check_f("sscal[2]", 9, x[2]);
}

// ТЕСТ 14: ВРАЩЕНИЕ (float)
void test_srotg() {
    float a=3, b=4, c, s;
    cblas_srotg(&a, &b, &c, &s);
    check_f("srotg r", 5, a);
    check_f("srotg c", 0.6, c);
}

int main() {
    signal(SIGSEGV, segfault_handler);
    
    printf("\n=== OpenBLAS Level 1 Тесты (14 тестов, 28 проверок) ===\n\n");
    
    RUN_TEST(test_sdot);
    RUN_TEST(test_ddot);
    RUN_TEST(test_snrm2);
    RUN_TEST(test_dnrm2);
    RUN_TEST(test_sasum);
    RUN_TEST(test_dasum);
    RUN_TEST(test_isamax);
    RUN_TEST(test_idamax);
    RUN_TEST(test_scopy);
    RUN_TEST(test_dcopy);
    RUN_TEST(test_sswap);
    RUN_TEST(test_saxpy);
    RUN_TEST(test_sscal);
    RUN_TEST(test_srotg);
    
    printf("\n=== РЕЗУЛЬТАТЫ ===\n");
    printf(" ПРОЙДЕНО: %d\n", passed);
    printf(" НЕ ПРОЙДЕНО: %d\n", failed);
    printf(" ВСЕГО: %d\n", passed + failed);
    
    return 0;
}
