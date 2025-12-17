/**
 * CogTensorNet - Tensor Network Core Implementation
 * 
 * Implements tensor network structures, contraction algorithms, and
 * optimization strategies for efficient tensor network computation.
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "../include/cogint.h"
#include "../include/cog_tensornet.h"

/*============================================================================
 * Utility Functions
 *============================================================================*/

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int64_t product(int64_t *arr, int n) {
    int64_t p = 1;
    for (int i = 0; i < n; i++) p *= arr[i];
    return p;
}

/*============================================================================
 * Tensor Network Creation
 *============================================================================*/

COGINT_API CogTensorNetwork* cog_tn_create(CogContext *ctx, const char *name,
                                            CogTNType type) {
    CogTensorNetwork *tn = calloc(1, sizeof(CogTensorNetwork));
    if (!tn) return NULL;
    
    tn->name = name ? strdup(name) : strdup("unnamed");
    tn->type = type;
    tn->ctx = ctx;
    
    /* Initial capacity */
    tn->node_capacity = 64;
    tn->edge_capacity = 128;
    
    tn->nodes = calloc(tn->node_capacity, sizeof(CogTNNode*));
    tn->edges = calloc(tn->edge_capacity, sizeof(CogTNEdge*));
    
    return tn;
}

COGINT_API void cog_tn_free(CogTensorNetwork *tn) {
    if (!tn) return;
    
    /* Free nodes */
    for (size_t i = 0; i < tn->n_nodes; i++) {
        CogTNNode *node = tn->nodes[i];
        if (node) {
            free(node->name);
            free(node->shape);
            free(node->edges);
            free(node->edge_axes);
            if (node->tensor) cog_tensor_free(node->tensor);
            if (node->grad) cog_tensor_free(node->grad);
            free(node);
        }
    }
    free(tn->nodes);
    
    /* Free edges */
    for (size_t i = 0; i < tn->n_edges; i++) {
        CogTNEdge *edge = tn->edges[i];
        if (edge) {
            free(edge->name);
            free(edge);
        }
    }
    free(tn->edges);
    
    free(tn->open_edges);
    free(tn->contraction_order);
    free(tn->node_partitions);
    free(tn->name);
    free(tn);
}

COGINT_API uint32_t cog_tn_add_node(CogTensorNetwork *tn, CogTensor *tensor,
                                     const char *name) {
    if (!tn || !tensor) return (uint32_t)-1;
    
    /* Expand if needed */
    if (tn->n_nodes >= tn->node_capacity) {
        tn->node_capacity *= 2;
        tn->nodes = realloc(tn->nodes, tn->node_capacity * sizeof(CogTNNode*));
    }
    
    CogTNNode *node = calloc(1, sizeof(CogTNNode));
    node->id = tn->n_nodes;
    node->name = name ? strdup(name) : NULL;
    node->type = COG_TN_NODE_TENSOR;
    node->tensor = tensor;
    
    /* Copy shape */
    node->ndim = tensor->ndim;
    node->shape = malloc(tensor->ndim * sizeof(int64_t));
    memcpy(node->shape, tensor->shape, tensor->ndim * sizeof(int64_t));
    
    /* Estimate computational cost */
    node->cost = (double)tensor->numel;
    
    tn->nodes[tn->n_nodes++] = node;
    
    return node->id;
}

COGINT_API uint32_t cog_tn_add_edge(CogTensorNetwork *tn,
                                     uint32_t node_a, int axis_a,
                                     uint32_t node_b, int axis_b,
                                     const char *name) {
    if (!tn) return (uint32_t)-1;
    if (node_a >= tn->n_nodes || node_b >= tn->n_nodes) return (uint32_t)-1;
    
    CogTNNode *na = tn->nodes[node_a];
    CogTNNode *nb = tn->nodes[node_b];
    
    if (axis_a >= na->ndim || axis_b >= nb->ndim) return (uint32_t)-1;
    if (na->shape[axis_a] != nb->shape[axis_b]) return (uint32_t)-1;
    
    /* Expand if needed */
    if (tn->n_edges >= tn->edge_capacity) {
        tn->edge_capacity *= 2;
        tn->edges = realloc(tn->edges, tn->edge_capacity * sizeof(CogTNEdge*));
    }
    
    CogTNEdge *edge = calloc(1, sizeof(CogTNEdge));
    edge->id = tn->n_edges;
    edge->name = name ? strdup(name) : NULL;
    edge->dim = na->shape[axis_a];
    edge->node_a = node_a;
    edge->node_b = node_b;
    edge->axis_a = axis_a;
    edge->axis_b = axis_b;
    edge->weight = 1.0;
    
    tn->edges[tn->n_edges++] = edge;
    
    /* Update node connectivity */
    na->edges = realloc(na->edges, (na->n_edges + 1) * sizeof(uint32_t));
    na->edge_axes = realloc(na->edge_axes, (na->n_edges + 1) * sizeof(int));
    na->edges[na->n_edges] = edge->id;
    na->edge_axes[na->n_edges] = axis_a;
    na->n_edges++;
    
    nb->edges = realloc(nb->edges, (nb->n_edges + 1) * sizeof(uint32_t));
    nb->edge_axes = realloc(nb->edge_axes, (nb->n_edges + 1) * sizeof(int));
    nb->edges[nb->n_edges] = edge->id;
    nb->edge_axes[nb->n_edges] = axis_b;
    nb->n_edges++;
    
    return edge->id;
}

COGINT_API int cog_tn_mark_open(CogTensorNetwork *tn, uint32_t edge_id) {
    if (!tn || edge_id >= tn->n_edges) return COG_ERR_INVALID;
    
    tn->edges[edge_id]->is_open = 1;
    
    tn->open_edges = realloc(tn->open_edges, (tn->n_open + 1) * sizeof(uint32_t));
    tn->open_edges[tn->n_open++] = edge_id;
    
    return COG_OK;
}

COGINT_API CogTNNode* cog_tn_get_node(CogTensorNetwork *tn, uint32_t id) {
    if (!tn || id >= tn->n_nodes) return NULL;
    return tn->nodes[id];
}

COGINT_API CogTNEdge* cog_tn_get_edge(CogTensorNetwork *tn, uint32_t id) {
    if (!tn || id >= tn->n_edges) return NULL;
    return tn->edges[id];
}

/*============================================================================
 * Tensor Contraction
 *============================================================================*/

COGINT_API CogTensor* cog_tensor_contract(CogTensor *a, CogTensor *b,
                                           int *axes_a, int *axes_b, int n_axes) {
    if (!a || !b || !axes_a || !axes_b || n_axes <= 0) return NULL;
    
    /* Validate axes and dimensions */
    for (int i = 0; i < n_axes; i++) {
        if (axes_a[i] >= a->ndim || axes_b[i] >= b->ndim) return NULL;
        if (a->shape[axes_a[i]] != b->shape[axes_b[i]]) return NULL;
    }
    
    /* Compute result shape */
    int result_ndim = a->ndim + b->ndim - 2 * n_axes;
    int64_t *result_shape = malloc(result_ndim * sizeof(int64_t));
    
    /* Mark contracted axes */
    int *a_contracted = calloc(a->ndim, sizeof(int));
    int *b_contracted = calloc(b->ndim, sizeof(int));
    for (int i = 0; i < n_axes; i++) {
        a_contracted[axes_a[i]] = 1;
        b_contracted[axes_b[i]] = 1;
    }
    
    /* Build result shape from non-contracted axes */
    int idx = 0;
    int *a_free = malloc(a->ndim * sizeof(int));
    int *b_free = malloc(b->ndim * sizeof(int));
    int n_a_free = 0, n_b_free = 0;
    
    for (int i = 0; i < a->ndim; i++) {
        if (!a_contracted[i]) {
            result_shape[idx++] = a->shape[i];
            a_free[n_a_free++] = i;
        }
    }
    for (int i = 0; i < b->ndim; i++) {
        if (!b_contracted[i]) {
            result_shape[idx++] = b->shape[i];
            b_free[n_b_free++] = i;
        }
    }
    
    /* Create result tensor */
    CogTensor *result = cog_tensor_create(NULL, result_shape, result_ndim, a->dtype);
    if (!result) {
        free(result_shape);
        free(a_contracted);
        free(b_contracted);
        free(a_free);
        free(b_free);
        return NULL;
    }
    
    /* Compute contraction dimension */
    int64_t contract_dim = 1;
    for (int i = 0; i < n_axes; i++) {
        contract_dim *= a->shape[axes_a[i]];
    }
    
    /* Compute free dimensions */
    int64_t a_free_dim = 1, b_free_dim = 1;
    for (int i = 0; i < n_a_free; i++) a_free_dim *= a->shape[a_free[i]];
    for (int i = 0; i < n_b_free; i++) b_free_dim *= b->shape[b_free[i]];
    
    /* Perform contraction (simplified - treats as matrix multiplication) */
    float *ra = (float*)a->data;
    float *rb = (float*)b->data;
    float *rc = (float*)result->data;
    
    /* This is a simplified implementation - real implementation would
       handle arbitrary axis ordering */
    for (int64_t i = 0; i < a_free_dim; i++) {
        for (int64_t j = 0; j < b_free_dim; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < contract_dim; k++) {
                sum += ra[i * contract_dim + k] * rb[k * b_free_dim + j];
            }
            rc[i * b_free_dim + j] = sum;
        }
    }
    
    free(result_shape);
    free(a_contracted);
    free(b_contracted);
    free(a_free);
    free(b_free);
    
    return result;
}

/*============================================================================
 * Einsum Implementation
 *============================================================================*/

/* Parse einsum subscript string */
typedef struct {
    char *indices[32];           /* Index labels for each tensor */
    int n_indices[32];           /* Number of indices per tensor */
    int n_tensors;
    char *output_indices;
    int n_output;
    int implicit_output;         /* No explicit output specified */
} EinsumParsed;

static EinsumParsed* parse_einsum(const char *subscripts) {
    EinsumParsed *parsed = calloc(1, sizeof(EinsumParsed));
    
    char *sub = strdup(subscripts);
    char *arrow = strstr(sub, "->");
    
    if (arrow) {
        *arrow = '\0';
        parsed->output_indices = strdup(arrow + 2);
        parsed->n_output = strlen(parsed->output_indices);
    } else {
        parsed->implicit_output = 1;
    }
    
    /* Parse input tensors */
    char *token = strtok(sub, ",");
    while (token && parsed->n_tensors < 32) {
        /* Skip whitespace */
        while (*token == ' ') token++;
        
        parsed->indices[parsed->n_tensors] = strdup(token);
        parsed->n_indices[parsed->n_tensors] = strlen(token);
        parsed->n_tensors++;
        
        token = strtok(NULL, ",");
    }
    
    free(sub);
    return parsed;
}

static void free_einsum_parsed(EinsumParsed *parsed) {
    for (int i = 0; i < parsed->n_tensors; i++) {
        free(parsed->indices[i]);
    }
    free(parsed->output_indices);
    free(parsed);
}

COGINT_API CogTensor* cog_tensor_einsum(const char *subscripts,
                                         CogTensor **tensors, int n_tensors) {
    if (!subscripts || !tensors || n_tensors <= 0) return NULL;
    
    EinsumParsed *parsed = parse_einsum(subscripts);
    if (parsed->n_tensors != n_tensors) {
        free_einsum_parsed(parsed);
        return NULL;
    }
    
    /* Build index dimension map */
    int64_t index_dims[256] = {0};  /* Map char -> dimension */
    
    for (int t = 0; t < n_tensors; t++) {
        for (int i = 0; i < parsed->n_indices[t]; i++) {
            char idx = parsed->indices[t][i];
            if (index_dims[(int)idx] == 0) {
                index_dims[(int)idx] = tensors[t]->shape[i];
            }
        }
    }
    
    /* For simple cases, use direct contraction */
    if (n_tensors == 2) {
        /* Find common indices */
        int axes_a[8], axes_b[8];
        int n_common = 0;
        
        for (int i = 0; i < parsed->n_indices[0]; i++) {
            char idx = parsed->indices[0][i];
            for (int j = 0; j < parsed->n_indices[1]; j++) {
                if (parsed->indices[1][j] == idx) {
                    /* Check if this index is in output */
                    int in_output = 0;
                    if (parsed->output_indices) {
                        for (int k = 0; k < parsed->n_output; k++) {
                            if (parsed->output_indices[k] == idx) {
                                in_output = 1;
                                break;
                            }
                        }
                    }
                    
                    if (!in_output) {
                        axes_a[n_common] = i;
                        axes_b[n_common] = j;
                        n_common++;
                    }
                }
            }
        }
        
        CogTensor *result = cog_tensor_contract(tensors[0], tensors[1],
                                                 axes_a, axes_b, n_common);
        free_einsum_parsed(parsed);
        return result;
    }
    
    /* For more tensors, contract pairwise */
    CogTensor *result = cog_tensor_clone(tensors[0]);
    
    for (int t = 1; t < n_tensors; t++) {
        /* Find common indices between result and next tensor */
        /* Simplified - assumes sequential contraction */
        int axes_a[8] = {0}, axes_b[8] = {0};
        int n_common = 1;  /* Simplified */
        
        CogTensor *new_result = cog_tensor_contract(result, tensors[t],
                                                     axes_a, axes_b, n_common);
        cog_tensor_free(result);
        result = new_result;
        
        if (!result) break;
    }
    
    free_einsum_parsed(parsed);
    return result;
}

/*============================================================================
 * Contraction Path Optimization
 *============================================================================*/

/* Cost function for contracting two nodes */
static double contraction_cost(CogTensorNetwork *tn, uint32_t node_a, uint32_t node_b,
                               int64_t *result_size) {
    CogTNNode *na = tn->nodes[node_a];
    CogTNNode *nb = tn->nodes[node_b];
    
    if (!na || !nb) return DBL_MAX;
    
    /* Find shared edges */
    int64_t contract_dim = 1;
    int n_shared = 0;
    
    for (size_t i = 0; i < na->n_edges; i++) {
        CogTNEdge *edge = tn->edges[na->edges[i]];
        if ((edge->node_a == node_a && edge->node_b == node_b) ||
            (edge->node_a == node_b && edge->node_b == node_a)) {
            contract_dim *= edge->dim;
            n_shared++;
        }
    }
    
    if (n_shared == 0) {
        /* No shared edges - outer product */
        *result_size = product(na->shape, na->ndim) * product(nb->shape, nb->ndim);
        return (double)(*result_size);
    }
    
    /* Compute result size */
    int64_t a_size = product(na->shape, na->ndim);
    int64_t b_size = product(nb->shape, nb->ndim);
    *result_size = (a_size / contract_dim) * (b_size / contract_dim);
    
    /* FLOPs = 2 * result_size * contract_dim (multiply-add) */
    return 2.0 * (*result_size) * contract_dim;
}

/* Greedy contraction ordering */
static CogContractionPath* greedy_optimize(CogTensorNetwork *tn, CogContractConfig *config) {
    CogContractionPath *path = calloc(1, sizeof(CogContractionPath));
    
    /* Track which nodes are still available */
    int *available = malloc(tn->n_nodes * sizeof(int));
    for (size_t i = 0; i < tn->n_nodes; i++) available[i] = 1;
    
    /* Node sizes (updated as we contract) */
    int64_t *sizes = malloc(tn->n_nodes * sizeof(int64_t));
    for (size_t i = 0; i < tn->n_nodes; i++) {
        sizes[i] = product(tn->nodes[i]->shape, tn->nodes[i]->ndim);
    }
    
    path->steps = malloc((tn->n_nodes - 1) * sizeof(CogContractionStep));
    path->peak_memory = 0;
    
    int64_t current_memory = 0;
    for (size_t i = 0; i < tn->n_nodes; i++) {
        current_memory += sizes[i] * sizeof(float);
    }
    
    uint32_t next_node_id = tn->n_nodes;
    
    /* Greedily select best contraction */
    while (path->n_steps < tn->n_nodes - 1) {
        double best_cost = DBL_MAX;
        uint32_t best_a = 0, best_b = 0;
        int64_t best_result_size = 0;
        
        /* Find pair with lowest cost */
        for (size_t i = 0; i < tn->n_nodes; i++) {
            if (!available[i]) continue;
            
            for (size_t j = i + 1; j < tn->n_nodes; j++) {
                if (!available[j]) continue;
                
                /* Check if nodes share an edge */
                int connected = 0;
                CogTNNode *na = tn->nodes[i];
                for (size_t k = 0; k < na->n_edges; k++) {
                    CogTNEdge *edge = tn->edges[na->edges[k]];
                    if ((edge->node_a == i && edge->node_b == j) ||
                        (edge->node_a == j && edge->node_b == i)) {
                        connected = 1;
                        break;
                    }
                }
                
                if (!connected) continue;
                
                int64_t result_size;
                double cost = contraction_cost(tn, i, j, &result_size);
                
                /* Optionally weight by memory */
                if (config && config->minimize_memory) {
                    cost = (double)result_size;
                }
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_a = i;
                    best_b = j;
                    best_result_size = result_size;
                }
            }
        }
        
        if (best_cost == DBL_MAX) {
            /* No more contractions possible */
            break;
        }
        
        /* Record contraction step */
        CogContractionStep *step = &path->steps[path->n_steps++];
        step->node_a = best_a;
        step->node_b = best_b;
        step->result_node = next_node_id++;
        step->flops = best_cost;
        step->memory = best_result_size * sizeof(float);
        
        path->total_flops += best_cost;
        
        /* Update memory tracking */
        current_memory -= sizes[best_a] * sizeof(float);
        current_memory -= sizes[best_b] * sizeof(float);
        current_memory += step->memory;
        
        if (current_memory > path->peak_memory) {
            path->peak_memory = current_memory;
        }
        
        /* Mark nodes as contracted */
        available[best_a] = 0;
        available[best_b] = 0;
        
        /* Expand arrays for new node */
        available = realloc(available, next_node_id * sizeof(int));
        sizes = realloc(sizes, next_node_id * sizeof(int64_t));
        available[step->result_node] = 1;
        sizes[step->result_node] = best_result_size;
    }
    
    free(available);
    free(sizes);
    
    return path;
}

/* Random-greedy optimization with multiple trials */
static CogContractionPath* random_greedy_optimize(CogTensorNetwork *tn, 
                                                   CogContractConfig *config) {
    CogContractionPath *best_path = NULL;
    double best_cost = DBL_MAX;
    
    int trials = config ? config->greedy_trials : 128;
    double temperature = config ? config->greedy_temperature : 1.0;
    
    for (int trial = 0; trial < trials; trial++) {
        /* Create a modified config with randomness */
        CogContractConfig trial_config = *config;
        trial_config.greedy_temperature = temperature * (1.0 + 0.1 * trial);
        
        CogContractionPath *path = greedy_optimize(tn, &trial_config);
        
        if (path->total_flops < best_cost) {
            if (best_path) cog_path_free(best_path);
            best_path = path;
            best_cost = path->total_flops;
        } else {
            cog_path_free(path);
        }
    }
    
    return best_path;
}

/* Dynamic programming optimization (exact for small networks) */
static CogContractionPath* dp_optimize(CogTensorNetwork *tn, CogContractConfig *config) {
    size_t n = tn->n_nodes;
    
    /* For large networks, fall back to greedy */
    if (n > 20) {
        return greedy_optimize(tn, config);
    }
    
    /* DP table: cost[subset] = minimum cost to contract subset */
    size_t n_subsets = 1UL << n;
    double *cost = malloc(n_subsets * sizeof(double));
    uint32_t *split = malloc(n_subsets * sizeof(uint32_t));  /* Best split point */
    
    for (size_t i = 0; i < n_subsets; i++) {
        cost[i] = DBL_MAX;
        split[i] = 0;
    }
    
    /* Base case: single nodes have zero cost */
    for (size_t i = 0; i < n; i++) {
        cost[1UL << i] = 0;
    }
    
    /* Fill DP table */
    for (size_t size = 2; size <= n; size++) {
        for (size_t subset = 0; subset < n_subsets; subset++) {
            if (__builtin_popcountl(subset) != (int)size) continue;
            
            /* Try all ways to split subset into two parts */
            for (size_t sub1 = (subset - 1) & subset; sub1 > 0; sub1 = (sub1 - 1) & subset) {
                size_t sub2 = subset ^ sub1;
                if (sub2 == 0 || sub1 > sub2) continue;  /* Avoid duplicates */
                
                double c = cost[sub1] + cost[sub2];
                
                /* Add cost of contracting the two subsets */
                /* Simplified - would need to track intermediate tensor sizes */
                c += 1e6;  /* Placeholder */
                
                if (c < cost[subset]) {
                    cost[subset] = c;
                    split[subset] = sub1;
                }
            }
        }
    }
    
    /* Reconstruct path from DP table */
    CogContractionPath *path = calloc(1, sizeof(CogContractionPath));
    path->steps = malloc((n - 1) * sizeof(CogContractionStep));
    path->total_flops = cost[n_subsets - 1];
    
    /* TODO: Reconstruct actual steps from split array */
    
    free(cost);
    free(split);
    
    /* For now, fall back to greedy for path reconstruction */
    cog_path_free(path);
    return greedy_optimize(tn, config);
}

COGINT_API CogContractionPath* cog_tn_optimize_path(CogTensorNetwork *tn,
                                                     CogContractConfig *config) {
    if (!tn || tn->n_nodes == 0) return NULL;
    
    CogContractMethod method = config ? config->method : COG_CONTRACT_GREEDY;
    
    switch (method) {
        case COG_CONTRACT_GREEDY:
            return greedy_optimize(tn, config);
            
        case COG_CONTRACT_RANDOM_GREEDY:
            return random_greedy_optimize(tn, config);
            
        case COG_CONTRACT_OPTIMAL:
        case COG_CONTRACT_DP:
            return dp_optimize(tn, config);
            
        case COG_CONTRACT_SIMULATED_ANNEALING:
            /* TODO: Implement SA */
            return greedy_optimize(tn, config);
            
        case COG_CONTRACT_GENETIC:
            /* TODO: Implement GA */
            return greedy_optimize(tn, config);
            
        default:
            return greedy_optimize(tn, config);
    }
}

/*============================================================================
 * Network Contraction Execution
 *============================================================================*/

COGINT_API CogTensor* cog_tn_contract(CogTensorNetwork *tn,
                                       CogContractionPath *path) {
    if (!tn || !path || path->n_steps == 0) return NULL;
    
    /* Store intermediate results */
    CogTensor **intermediates = calloc(tn->n_nodes + path->n_steps, sizeof(CogTensor*));
    
    /* Copy initial tensors */
    for (size_t i = 0; i < tn->n_nodes; i++) {
        intermediates[i] = tn->nodes[i]->tensor;
    }
    
    /* Execute contraction steps */
    for (size_t i = 0; i < path->n_steps; i++) {
        CogContractionStep *step = &path->steps[i];
        
        CogTensor *a = intermediates[step->node_a];
        CogTensor *b = intermediates[step->node_b];
        
        if (!a || !b) {
            /* Clean up and return error */
            for (size_t j = tn->n_nodes; j < tn->n_nodes + i; j++) {
                if (intermediates[j]) cog_tensor_free(intermediates[j]);
            }
            free(intermediates);
            return NULL;
        }
        
        /* Find contraction axes from edges */
        int axes_a[8], axes_b[8];
        int n_axes = 0;
        
        /* For simplicity, contract along last axis of a and first axis of b */
        if (a->ndim > 0 && b->ndim > 0 && a->shape[a->ndim - 1] == b->shape[0]) {
            axes_a[0] = a->ndim - 1;
            axes_b[0] = 0;
            n_axes = 1;
        }
        
        CogTensor *result = cog_tensor_contract(a, b, axes_a, axes_b, n_axes);
        intermediates[step->result_node] = result;
        
        /* Free intermediate tensors that are no longer needed */
        if (step->node_a >= tn->n_nodes) {
            cog_tensor_free(intermediates[step->node_a]);
            intermediates[step->node_a] = NULL;
        }
        if (step->node_b >= tn->n_nodes) {
            cog_tensor_free(intermediates[step->node_b]);
            intermediates[step->node_b] = NULL;
        }
    }
    
    /* Get final result */
    CogTensor *result = intermediates[path->steps[path->n_steps - 1].result_node];
    
    free(intermediates);
    
    return result;
}

COGINT_API void cog_path_free(CogContractionPath *path) {
    if (!path) return;
    free(path->steps);
    free(path);
}

/*============================================================================
 * Utility Functions
 *============================================================================*/

COGINT_API double cog_tn_estimate_flops(CogTensorNetwork *tn,
                                         CogContractionPath *path) {
    if (!path) return 0.0;
    return path->total_flops;
}

COGINT_API int64_t cog_tn_estimate_memory(CogTensorNetwork *tn,
                                           CogContractionPath *path) {
    if (!path) return 0;
    return path->peak_memory;
}

COGINT_API char* cog_tn_to_dot(CogTensorNetwork *tn) {
    if (!tn) return NULL;
    
    size_t buf_size = 4096 + tn->n_nodes * 256 + tn->n_edges * 128;
    char *buf = malloc(buf_size);
    char *p = buf;
    
    p += sprintf(p, "graph TensorNetwork {\n");
    p += sprintf(p, "  rankdir=LR;\n");
    p += sprintf(p, "  node [shape=box];\n\n");
    
    /* Nodes */
    for (size_t i = 0; i < tn->n_nodes; i++) {
        CogTNNode *node = tn->nodes[i];
        p += sprintf(p, "  n%u [label=\"%s\\n", node->id,
                     node->name ? node->name : "tensor");
        
        /* Shape info */
        p += sprintf(p, "(");
        for (int j = 0; j < node->ndim; j++) {
            if (j > 0) p += sprintf(p, ",");
            p += sprintf(p, "%ld", node->shape[j]);
        }
        p += sprintf(p, ")\"];\n");
    }
    
    p += sprintf(p, "\n");
    
    /* Edges */
    for (size_t i = 0; i < tn->n_edges; i++) {
        CogTNEdge *edge = tn->edges[i];
        if (edge->is_open) continue;
        
        p += sprintf(p, "  n%u -- n%u [label=\"%s\\ndim=%ld\"];\n",
                     edge->node_a, edge->node_b,
                     edge->name ? edge->name : "",
                     edge->dim);
    }
    
    p += sprintf(p, "}\n");
    
    return buf;
}

COGINT_API void cog_path_print(CogContractionPath *path) {
    if (!path) return;
    
    printf("Contraction Path:\n");
    printf("  Total FLOPs: %.2e\n", path->total_flops);
    printf("  Peak Memory: %ld bytes\n", path->peak_memory);
    printf("  Steps:\n");
    
    for (size_t i = 0; i < path->n_steps; i++) {
        CogContractionStep *step = &path->steps[i];
        printf("    %zu: contract(%u, %u) -> %u  [%.2e FLOPs, %ld bytes]\n",
               i, step->node_a, step->node_b, step->result_node,
               step->flops, step->memory);
    }
}
