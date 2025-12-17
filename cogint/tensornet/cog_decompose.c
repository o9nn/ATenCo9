/**
 * CogDecompose - Tensor Decomposition Algorithms
 * 
 * Implements various tensor decomposition methods:
 * - SVD (Singular Value Decomposition)
 * - QR Decomposition
 * - Tucker Decomposition (HOSVD)
 * - Tensor Train (TT) Decomposition
 * - CP (CANDECOMP/PARAFAC) Decomposition
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../include/cogint.h"
#include "../include/cog_tensornet.h"

/*============================================================================
 * Utility Functions
 *============================================================================*/

/* Compute Frobenius norm */
static double frobenius_norm(float *data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i] * data[i];
    }
    return sqrt(sum);
}

/* Matrix transpose */
static void transpose(float *src, float *dst, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/* Matrix-matrix multiplication: C = A * B */
static void matmul(float *A, float *B, float *C,
                   int64_t m, int64_t k, int64_t n) {
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* Compute A^T * A */
static void ata(float *A, float *ATA, int64_t m, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            ATA[i * n + j] = sum;
        }
    }
}

/* Power iteration for dominant eigenvector */
static void power_iteration(float *A, float *v, int64_t n, int max_iter) {
    float *Av = malloc(n * sizeof(float));
    
    /* Initialize v randomly */
    for (int64_t i = 0; i < n; i++) {
        v[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* Av = A * v */
        for (int64_t i = 0; i < n; i++) {
            Av[i] = 0.0f;
            for (int64_t j = 0; j < n; j++) {
                Av[i] += A[i * n + j] * v[j];
            }
        }
        
        /* Normalize */
        float norm = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            norm += Av[i] * Av[i];
        }
        norm = sqrtf(norm);
        
        if (norm < 1e-10f) break;
        
        for (int64_t i = 0; i < n; i++) {
            v[i] = Av[i] / norm;
        }
    }
    
    free(Av);
}

/*============================================================================
 * SVD Implementation
 *============================================================================*/

/* One-sided Jacobi SVD for small matrices */
static void jacobi_svd(float *A, float *U, float *S, float *V,
                       int64_t m, int64_t n) {
    int64_t min_mn = m < n ? m : n;
    
    /* Copy A to work matrix */
    float *work = malloc(m * n * sizeof(float));
    memcpy(work, A, m * n * sizeof(float));
    
    /* Initialize U and V as identity */
    for (int64_t i = 0; i < m * m; i++) U[i] = 0.0f;
    for (int64_t i = 0; i < m; i++) U[i * m + i] = 1.0f;
    
    for (int64_t i = 0; i < n * n; i++) V[i] = 0.0f;
    for (int64_t i = 0; i < n; i++) V[i * n + i] = 1.0f;
    
    /* Jacobi rotations */
    int max_sweeps = 30;
    double tol = 1e-10;
    
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off_diag = 0.0;
        
        for (int64_t i = 0; i < n - 1; i++) {
            for (int64_t j = i + 1; j < n; j++) {
                /* Compute 2x2 submatrix of A^T * A */
                float aii = 0.0f, ajj = 0.0f, aij = 0.0f;
                for (int64_t k = 0; k < m; k++) {
                    aii += work[k * n + i] * work[k * n + i];
                    ajj += work[k * n + j] * work[k * n + j];
                    aij += work[k * n + i] * work[k * n + j];
                }
                
                off_diag += aij * aij;
                
                if (fabsf(aij) < tol * sqrtf(aii * ajj)) continue;
                
                /* Compute rotation angle */
                float tau = (ajj - aii) / (2.0f * aij);
                float t = (tau >= 0 ? 1.0f : -1.0f) / 
                          (fabsf(tau) + sqrtf(1.0f + tau * tau));
                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = t * c;
                
                /* Apply rotation to work matrix (columns i, j) */
                for (int64_t k = 0; k < m; k++) {
                    float wi = work[k * n + i];
                    float wj = work[k * n + j];
                    work[k * n + i] = c * wi - s * wj;
                    work[k * n + j] = s * wi + c * wj;
                }
                
                /* Apply rotation to V */
                for (int64_t k = 0; k < n; k++) {
                    float vi = V[k * n + i];
                    float vj = V[k * n + j];
                    V[k * n + i] = c * vi - s * vj;
                    V[k * n + j] = s * vi + c * vj;
                }
            }
        }
        
        if (off_diag < tol * tol) break;
    }
    
    /* Extract singular values and U */
    for (int64_t i = 0; i < min_mn; i++) {
        float sigma = 0.0f;
        for (int64_t k = 0; k < m; k++) {
            sigma += work[k * n + i] * work[k * n + i];
        }
        sigma = sqrtf(sigma);
        S[i] = sigma;
        
        if (sigma > 1e-10f) {
            for (int64_t k = 0; k < m; k++) {
                U[k * min_mn + i] = work[k * n + i] / sigma;
            }
        }
    }
    
    free(work);
}

/* Randomized SVD for large matrices */
static void randomized_svd(float *A, float *U, float *S, float *V,
                           int64_t m, int64_t n, int64_t rank,
                           int oversampling, int power_iter) {
    int64_t k = rank + oversampling;
    if (k > n) k = n;
    
    /* Generate random matrix Omega (n x k) */
    float *Omega = malloc(n * k * sizeof(float));
    for (int64_t i = 0; i < n * k; i++) {
        Omega[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    /* Y = A * Omega (m x k) */
    float *Y = malloc(m * k * sizeof(float));
    matmul(A, Omega, Y, m, n, k);
    
    /* Power iterations for better accuracy */
    float *temp = malloc(n * k * sizeof(float));
    for (int p = 0; p < power_iter; p++) {
        /* temp = A^T * Y */
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int64_t l = 0; l < m; l++) {
                    sum += A[l * n + i] * Y[l * k + j];
                }
                temp[i * k + j] = sum;
            }
        }
        /* Y = A * temp */
        matmul(A, temp, Y, m, n, k);
    }
    free(temp);
    
    /* QR decomposition of Y to get Q (m x k) */
    float *Q = malloc(m * k * sizeof(float));
    float *R = malloc(k * k * sizeof(float));
    
    /* Simple Gram-Schmidt QR */
    memcpy(Q, Y, m * k * sizeof(float));
    for (int64_t j = 0; j < k; j++) {
        /* Orthogonalize against previous columns */
        for (int64_t i = 0; i < j; i++) {
            float dot = 0.0f;
            for (int64_t l = 0; l < m; l++) {
                dot += Q[l * k + i] * Q[l * k + j];
            }
            for (int64_t l = 0; l < m; l++) {
                Q[l * k + j] -= dot * Q[l * k + i];
            }
        }
        /* Normalize */
        float norm = 0.0f;
        for (int64_t l = 0; l < m; l++) {
            norm += Q[l * k + j] * Q[l * k + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int64_t l = 0; l < m; l++) {
                Q[l * k + j] /= norm;
            }
        }
    }
    
    /* B = Q^T * A (k x n) */
    float *B = malloc(k * n * sizeof(float));
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < m; l++) {
                sum += Q[l * k + i] * A[l * n + j];
            }
            B[i * n + j] = sum;
        }
    }
    
    /* SVD of small matrix B */
    float *Ub = malloc(k * k * sizeof(float));
    float *Sb = malloc(k * sizeof(float));
    float *Vb = malloc(n * n * sizeof(float));
    
    jacobi_svd(B, Ub, Sb, Vb, k, n);
    
    /* U = Q * Ub */
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < rank; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; l++) {
                sum += Q[i * k + l] * Ub[l * k + j];
            }
            U[i * rank + j] = sum;
        }
    }
    
    /* Copy singular values and V */
    for (int64_t i = 0; i < rank; i++) {
        S[i] = Sb[i];
    }
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < rank; j++) {
            V[i * rank + j] = Vb[i * n + j];
        }
    }
    
    free(Omega);
    free(Y);
    free(Q);
    free(R);
    free(B);
    free(Ub);
    free(Sb);
    free(Vb);
}

COGINT_API CogDecompResult* cog_tensor_svd(CogTensor *tensor, int64_t max_rank,
                                            double truncation) {
    if (!tensor || tensor->ndim != 2) return NULL;
    
    CogDecompResult *result = calloc(1, sizeof(CogDecompResult));
    result->method = COG_DECOMP_SVD;
    
    int64_t m = tensor->shape[0];
    int64_t n = tensor->shape[1];
    int64_t min_mn = m < n ? m : n;
    int64_t rank = max_rank > 0 ? (max_rank < min_mn ? max_rank : min_mn) : min_mn;
    
    /* Allocate result tensors */
    int64_t u_shape[] = {m, rank};
    int64_t s_shape[] = {rank};
    int64_t v_shape[] = {n, rank};
    
    result->U = cog_tensor_create(NULL, u_shape, 2, tensor->dtype);
    result->S = cog_tensor_create(NULL, s_shape, 1, tensor->dtype);
    result->V = cog_tensor_create(NULL, v_shape, 2, tensor->dtype);
    
    float *A = (float*)tensor->data;
    float *U = (float*)result->U->data;
    float *S = (float*)result->S->data;
    float *V = (float*)result->V->data;
    
    /* Choose algorithm based on size */
    if (m * n < 10000 || rank >= min_mn / 2) {
        /* Full SVD for small matrices or high rank */
        float *U_full = malloc(m * min_mn * sizeof(float));
        float *S_full = malloc(min_mn * sizeof(float));
        float *V_full = malloc(n * n * sizeof(float));
        
        jacobi_svd(A, U_full, S_full, V_full, m, n);
        
        /* Copy truncated result */
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < rank; j++) {
                U[i * rank + j] = U_full[i * min_mn + j];
            }
        }
        for (int64_t i = 0; i < rank; i++) {
            S[i] = S_full[i];
        }
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < rank; j++) {
                V[i * rank + j] = V_full[i * n + j];
            }
        }
        
        free(U_full);
        free(S_full);
        free(V_full);
    } else {
        /* Randomized SVD for large matrices */
        randomized_svd(A, U, S, V, m, n, rank, 10, 2);
    }
    
    /* Apply truncation threshold */
    if (truncation > 0) {
        double total_energy = 0.0;
        for (int64_t i = 0; i < rank; i++) {
            total_energy += S[i] * S[i];
        }
        
        double cumulative = 0.0;
        int64_t effective_rank = rank;
        for (int64_t i = 0; i < rank; i++) {
            cumulative += S[i] * S[i];
            if (cumulative / total_energy >= 1.0 - truncation * truncation) {
                effective_rank = i + 1;
                break;
            }
        }
        
        /* Update shapes if truncated */
        if (effective_rank < rank) {
            result->U->shape[1] = effective_rank;
            result->S->shape[0] = effective_rank;
            result->V->shape[1] = effective_rank;
        }
    }
    
    /* Compute statistics */
    result->original_size = m * n * sizeof(float);
    result->compressed_size = (m * rank + rank + n * rank) * sizeof(float);
    result->compression_ratio = (double)result->original_size / result->compressed_size;
    
    return result;
}

/*============================================================================
 * QR Decomposition
 *============================================================================*/

COGINT_API CogDecompResult* cog_tensor_qr(CogTensor *tensor) {
    if (!tensor || tensor->ndim != 2) return NULL;
    
    CogDecompResult *result = calloc(1, sizeof(CogDecompResult));
    result->method = COG_DECOMP_QR;
    
    int64_t m = tensor->shape[0];
    int64_t n = tensor->shape[1];
    int64_t k = m < n ? m : n;
    
    /* Allocate result tensors */
    int64_t q_shape[] = {m, k};
    int64_t r_shape[] = {k, n};
    
    result->Q = cog_tensor_create(NULL, q_shape, 2, tensor->dtype);
    result->R = cog_tensor_create(NULL, r_shape, 2, tensor->dtype);
    
    float *A = (float*)tensor->data;
    float *Q = (float*)result->Q->data;
    float *R = (float*)result->R->data;
    
    /* Copy A to Q */
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j++) {
            Q[i * k + j] = (j < n) ? A[i * n + j] : 0.0f;
        }
    }
    
    /* Modified Gram-Schmidt */
    for (int64_t j = 0; j < k; j++) {
        /* Compute R[j,j] = ||Q[:,j]|| */
        float norm = 0.0f;
        for (int64_t i = 0; i < m; i++) {
            norm += Q[i * k + j] * Q[i * k + j];
        }
        norm = sqrtf(norm);
        R[j * n + j] = norm;
        
        /* Normalize Q[:,j] */
        if (norm > 1e-10f) {
            for (int64_t i = 0; i < m; i++) {
                Q[i * k + j] /= norm;
            }
        }
        
        /* Orthogonalize remaining columns */
        for (int64_t l = j + 1; l < k; l++) {
            /* R[j,l] = Q[:,j]^T * Q[:,l] */
            float dot = 0.0f;
            for (int64_t i = 0; i < m; i++) {
                dot += Q[i * k + j] * Q[i * k + l];
            }
            R[j * n + l] = dot;
            
            /* Q[:,l] -= R[j,l] * Q[:,j] */
            for (int64_t i = 0; i < m; i++) {
                Q[i * k + l] -= dot * Q[i * k + j];
            }
        }
    }
    
    return result;
}

/*============================================================================
 * Tucker Decomposition (HOSVD)
 *============================================================================*/

/* Unfold tensor along mode k */
static CogTensor* tensor_unfold(CogTensor *tensor, int mode) {
    int64_t rows = tensor->shape[mode];
    int64_t cols = tensor->numel / rows;
    
    int64_t shape[] = {rows, cols};
    CogTensor *unfolded = cog_tensor_create(NULL, shape, 2, tensor->dtype);
    
    float *src = (float*)tensor->data;
    float *dst = (float*)unfolded->data;
    
    /* Compute strides for unfolding */
    int64_t *strides = malloc(tensor->ndim * sizeof(int64_t));
    strides[tensor->ndim - 1] = 1;
    for (int i = tensor->ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * tensor->shape[i + 1];
    }
    
    /* Unfold */
    int64_t *indices = calloc(tensor->ndim, sizeof(int64_t));
    for (int64_t i = 0; i < tensor->numel; i++) {
        /* Compute multi-index */
        int64_t idx = i;
        for (int d = 0; d < tensor->ndim; d++) {
            indices[d] = idx / strides[d];
            idx %= strides[d];
        }
        
        /* Compute row (mode index) and column (other indices) */
        int64_t row = indices[mode];
        int64_t col = 0;
        int64_t col_stride = 1;
        for (int d = tensor->ndim - 1; d >= 0; d--) {
            if (d != mode) {
                col += indices[d] * col_stride;
                col_stride *= tensor->shape[d];
            }
        }
        
        dst[row * cols + col] = src[i];
    }
    
    free(strides);
    free(indices);
    
    return unfolded;
}

COGINT_API CogDecompResult* cog_tensor_tucker(CogTensor *tensor,
                                               int64_t *ranks, int n_modes) {
    if (!tensor || !ranks || n_modes != tensor->ndim) return NULL;
    
    CogDecompResult *result = calloc(1, sizeof(CogDecompResult));
    result->method = COG_DECOMP_TUCKER;
    result->n_factors = n_modes;
    result->factors = malloc(n_modes * sizeof(CogTensor*));
    
    /* Compute factor matrices via HOSVD */
    for (int mode = 0; mode < n_modes; mode++) {
        /* Unfold tensor along mode */
        CogTensor *unfolded = tensor_unfold(tensor, mode);
        
        /* SVD of unfolded matrix */
        CogDecompResult *svd = cog_tensor_svd(unfolded, ranks[mode], 0.0);
        
        /* Factor matrix is U from SVD */
        result->factors[mode] = svd->U;
        svd->U = NULL;  /* Transfer ownership */
        
        cog_decomp_free(svd);
        cog_tensor_free(unfolded);
    }
    
    /* Compute core tensor: G = tensor x_1 U_1^T x_2 U_2^T ... */
    CogTensor *core = cog_tensor_clone(tensor);
    
    for (int mode = 0; mode < n_modes; mode++) {
        CogTensor *U = result->factors[mode];
        
        /* Contract core with U^T along mode */
        /* Simplified - full implementation would use n-mode product */
        int64_t new_shape[8];
        for (int i = 0; i < core->ndim; i++) {
            new_shape[i] = (i == mode) ? ranks[mode] : core->shape[i];
        }
        
        CogTensor *new_core = cog_tensor_create(NULL, new_shape, core->ndim, core->dtype);
        
        /* Perform n-mode product (simplified) */
        float *core_data = (float*)core->data;
        float *U_data = (float*)U->data;
        float *new_data = (float*)new_core->data;
        
        /* This is a simplified implementation */
        memset(new_data, 0, new_core->numel * sizeof(float));
        
        /* For mode-0 product: new_core[i,...] = sum_j U[i,j] * core[j,...] */
        if (mode == 0) {
            int64_t rest = core->numel / core->shape[0];
            for (int64_t i = 0; i < ranks[0]; i++) {
                for (int64_t j = 0; j < core->shape[0]; j++) {
                    for (int64_t k = 0; k < rest; k++) {
                        new_data[i * rest + k] += U_data[j * ranks[0] + i] * 
                                                   core_data[j * rest + k];
                    }
                }
            }
        }
        
        cog_tensor_free(core);
        core = new_core;
    }
    
    result->core = core;
    
    /* Compute statistics */
    result->original_size = tensor->numel * sizeof(float);
    result->compressed_size = core->numel * sizeof(float);
    for (int i = 0; i < n_modes; i++) {
        result->compressed_size += result->factors[i]->numel * sizeof(float);
    }
    result->compression_ratio = (double)result->original_size / result->compressed_size;
    
    return result;
}

/*============================================================================
 * Tensor Train (TT) Decomposition
 *============================================================================*/

COGINT_API CogDecompResult* cog_tensor_tt(CogTensor *tensor,
                                           int64_t *ranks, int n_cores) {
    if (!tensor || !ranks) return NULL;
    if (n_cores != tensor->ndim) return NULL;
    
    CogDecompResult *result = calloc(1, sizeof(CogDecompResult));
    result->method = COG_DECOMP_TT;
    result->n_cores = n_cores;
    result->tt_cores = malloc(n_cores * sizeof(CogTensor*));
    
    /* TT-SVD algorithm */
    CogTensor *remainder = cog_tensor_clone(tensor);
    
    int64_t r_prev = 1;
    
    for (int k = 0; k < n_cores - 1; k++) {
        int64_t n_k = remainder->shape[0];
        int64_t rest = remainder->numel / n_k;
        
        /* Reshape to matrix: (r_{k-1} * n_k) x (rest) */
        int64_t mat_shape[] = {r_prev * n_k, rest};
        CogTensor *mat = cog_tensor_create(NULL, mat_shape, 2, tensor->dtype);
        memcpy(mat->data, remainder->data, remainder->numel * sizeof(float));
        mat->shape[0] = r_prev * n_k;
        mat->shape[1] = rest;
        
        /* SVD with truncation */
        int64_t r_k = ranks[k];
        if (r_k > r_prev * n_k) r_k = r_prev * n_k;
        if (r_k > rest) r_k = rest;
        
        CogDecompResult *svd = cog_tensor_svd(mat, r_k, 0.0);
        
        /* Core k: reshape U to (r_{k-1}, n_k, r_k) */
        int64_t core_shape[] = {r_prev, n_k, r_k};
        result->tt_cores[k] = cog_tensor_create(NULL, core_shape, 3, tensor->dtype);
        memcpy(result->tt_cores[k]->data, svd->U->data, 
               r_prev * n_k * r_k * sizeof(float));
        
        /* Remainder = S * V^T */
        cog_tensor_free(remainder);
        
        int64_t rem_shape[] = {r_k, rest};
        remainder = cog_tensor_create(NULL, rem_shape, 2, tensor->dtype);
        
        float *S = (float*)svd->S->data;
        float *V = (float*)svd->V->data;
        float *rem = (float*)remainder->data;
        
        for (int64_t i = 0; i < r_k; i++) {
            for (int64_t j = 0; j < rest; j++) {
                rem[i * rest + j] = S[i] * V[j * r_k + i];
            }
        }
        
        r_prev = r_k;
        
        cog_decomp_free(svd);
        cog_tensor_free(mat);
    }
    
    /* Last core */
    int64_t last_shape[] = {r_prev, remainder->shape[1], 1};
    result->tt_cores[n_cores - 1] = cog_tensor_create(NULL, last_shape, 3, tensor->dtype);
    memcpy(result->tt_cores[n_cores - 1]->data, remainder->data,
           remainder->numel * sizeof(float));
    
    cog_tensor_free(remainder);
    
    /* Compute statistics */
    result->original_size = tensor->numel * sizeof(float);
    result->compressed_size = 0;
    for (int i = 0; i < n_cores; i++) {
        result->compressed_size += result->tt_cores[i]->numel * sizeof(float);
    }
    result->compression_ratio = (double)result->original_size / result->compressed_size;
    
    return result;
}

/*============================================================================
 * General Decomposition Interface
 *============================================================================*/

COGINT_API CogDecompResult* cog_tensor_decompose(CogTensor *tensor,
                                                  CogDecompConfig *config) {
    if (!tensor || !config) return NULL;
    
    switch (config->method) {
        case COG_DECOMP_SVD:
            return cog_tensor_svd(tensor, config->max_rank, config->truncation_error);
            
        case COG_DECOMP_QR:
            return cog_tensor_qr(tensor);
            
        case COG_DECOMP_TUCKER:
        case COG_DECOMP_HOSVD:
            return cog_tensor_tucker(tensor, config->tucker_ranks, config->tucker_n_modes);
            
        case COG_DECOMP_TT:
            return cog_tensor_tt(tensor, config->tt_ranks, config->tt_n_cores);
            
        default:
            return NULL;
    }
}

/*============================================================================
 * Reconstruction
 *============================================================================*/

COGINT_API CogTensor* cog_decomp_reconstruct(CogDecompResult *decomp) {
    if (!decomp) return NULL;
    
    switch (decomp->method) {
        case COG_DECOMP_SVD: {
            /* A = U * diag(S) * V^T */
            if (!decomp->U || !decomp->S || !decomp->V) return NULL;
            
            int64_t m = decomp->U->shape[0];
            int64_t k = decomp->S->shape[0];
            int64_t n = decomp->V->shape[0];
            
            int64_t shape[] = {m, n};
            CogTensor *result = cog_tensor_create(NULL, shape, 2, decomp->U->dtype);
            
            float *U = (float*)decomp->U->data;
            float *S = (float*)decomp->S->data;
            float *V = (float*)decomp->V->data;
            float *R = (float*)result->data;
            
            for (int64_t i = 0; i < m; i++) {
                for (int64_t j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (int64_t l = 0; l < k; l++) {
                        sum += U[i * k + l] * S[l] * V[j * k + l];
                    }
                    R[i * n + j] = sum;
                }
            }
            
            return result;
        }
        
        case COG_DECOMP_QR: {
            /* A = Q * R */
            if (!decomp->Q || !decomp->R) return NULL;
            
            int64_t m = decomp->Q->shape[0];
            int64_t k = decomp->Q->shape[1];
            int64_t n = decomp->R->shape[1];
            
            int64_t shape[] = {m, n};
            CogTensor *result = cog_tensor_create(NULL, shape, 2, decomp->Q->dtype);
            
            matmul((float*)decomp->Q->data, (float*)decomp->R->data,
                   (float*)result->data, m, k, n);
            
            return result;
        }
        
        case COG_DECOMP_TT: {
            /* Contract TT cores */
            if (!decomp->tt_cores || decomp->n_cores == 0) return NULL;
            
            CogTensor *result = cog_tensor_clone(decomp->tt_cores[0]);
            
            for (int i = 1; i < decomp->n_cores; i++) {
                CogTensor *core = decomp->tt_cores[i];
                
                /* Contract result with core along last/first axes */
                int axes_a[] = {result->ndim - 1};
                int axes_b[] = {0};
                
                CogTensor *new_result = cog_tensor_contract(result, core, 
                                                             axes_a, axes_b, 1);
                cog_tensor_free(result);
                result = new_result;
            }
            
            return result;
        }
        
        default:
            return NULL;
    }
}

/*============================================================================
 * Cleanup
 *============================================================================*/

COGINT_API void cog_decomp_free(CogDecompResult *decomp) {
    if (!decomp) return;
    
    if (decomp->U) cog_tensor_free(decomp->U);
    if (decomp->S) cog_tensor_free(decomp->S);
    if (decomp->V) cog_tensor_free(decomp->V);
    if (decomp->Q) cog_tensor_free(decomp->Q);
    if (decomp->R) cog_tensor_free(decomp->R);
    if (decomp->core) cog_tensor_free(decomp->core);
    
    if (decomp->factors) {
        for (int i = 0; i < decomp->n_factors; i++) {
            if (decomp->factors[i]) cog_tensor_free(decomp->factors[i]);
        }
        free(decomp->factors);
    }
    
    if (decomp->tt_cores) {
        for (int i = 0; i < decomp->n_cores; i++) {
            if (decomp->tt_cores[i]) cog_tensor_free(decomp->tt_cores[i]);
        }
        free(decomp->tt_cores);
    }
    
    free(decomp);
}
