#include "rmsnorm.h"

static void compute_vars(float *data, float *vars, size_t n, int d_model);
static void compute_rsqrt(float *data, float *vars, int n_vars, size_t n, int d_model);
static void compute_norm_out(float *data, float *weights, size_t n, int d_model);

void compute_rms_norm(float *data, float *weights, size_t n, int d_model) {
    int n_vars = (int)(n / d_model);
    float vars[n_vars];
    compute_vars(data, vars, n, d_model);
    compute_rsqrt(data, vars, n_vars, n, d_model);
    compute_norm_out(data, weights, n, d_model);
}

static void compute_vars(float *data, float *vars, size_t n, int d_model) {
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < n; i += d_model) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int j = i; j < i + d_model; j++) {
            sum += data[j] * data[j];
        }
        vars[(int)(i / d_model)] = sum / d_model;
    }
}

static void compute_rsqrt(float *data, float *vars, int n_vars, size_t n, int d_model) {
    float var_eps = 1e-5;
    for (int i = 0; i < n_vars; i++) {
        vars[i] = 1. / sqrt(vars[i] + var_eps);
    }
    #pragma omp parallel for
    for (int i = 0; i < n; i += d_model) {
        for (int j = i; j < i + d_model; j++) {
            data[j] = data[j] * vars[(int)(i / d_model)];
        }
    }
}

static void compute_norm_out(float *data, float *weights, size_t n, int d_model) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n / d_model; i++) {
        int base = i * d_model;
        for (int j = 0; j < d_model; j++) {
            data[base + j] *= weights[j];
        }
    }
}