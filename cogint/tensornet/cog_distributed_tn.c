/**
 * CogDistributedTN - Distributed Tensor Network Computation
 * 
 * Implements distributed execution of tensor network contractions:
 * - Graph partitioning for load balancing
 * - Sliced contraction for memory-limited execution
 * - Communication-aware scheduling
 * - 9P-based tensor distribution
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include "../include/cogint.h"
#include "../include/cog_tensornet.h"
#include "../include/cog_inferno.h"

/*============================================================================
 * Partition Structures
 *============================================================================*/

typedef struct {
    uint32_t *nodes;
    size_t n_nodes;
    double total_cost;
    int64_t total_memory;
    
    /* Communication with other partitions */
    uint32_t *boundary_edges;
    size_t n_boundary;
} TNPartition;

typedef struct {
    TNPartition *partitions;
    int n_partitions;
    
    /* Edge cut information */
    uint32_t *cut_edges;
    size_t n_cut_edges;
    double communication_cost;
} TNPartitioning;

/*============================================================================
 * Partitioning Algorithms
 *============================================================================*/

/* Simple greedy partitioning */
static TNPartitioning* greedy_partition(CogTensorNetwork *tn, int n_parts) {
    TNPartitioning *result = calloc(1, sizeof(TNPartitioning));
    result->n_partitions = n_parts;
    result->partitions = calloc(n_parts, sizeof(TNPartition));
    
    /* Initialize partitions */
    for (int i = 0; i < n_parts; i++) {
        result->partitions[i].nodes = malloc(tn->n_nodes * sizeof(uint32_t));
        result->partitions[i].n_nodes = 0;
    }
    
    /* Assign nodes to partitions in round-robin fashion */
    /* (A real implementation would use graph partitioning algorithms) */
    for (size_t i = 0; i < tn->n_nodes; i++) {
        int part = i % n_parts;
        TNPartition *p = &result->partitions[part];
        p->nodes[p->n_nodes++] = i;
        
        CogTNNode *node = tn->nodes[i];
        p->total_cost += node->cost;
        p->total_memory += product(node->shape, node->ndim) * sizeof(float);
    }
    
    /* Find boundary edges (edges between partitions) */
    tn->node_partitions = malloc(tn->n_nodes * sizeof(int));
    for (int p = 0; p < n_parts; p++) {
        for (size_t i = 0; i < result->partitions[p].n_nodes; i++) {
            tn->node_partitions[result->partitions[p].nodes[i]] = p;
        }
    }
    
    result->cut_edges = malloc(tn->n_edges * sizeof(uint32_t));
    result->n_cut_edges = 0;
    
    for (size_t i = 0; i < tn->n_edges; i++) {
        CogTNEdge *edge = tn->edges[i];
        if (tn->node_partitions[edge->node_a] != tn->node_partitions[edge->node_b]) {
            result->cut_edges[result->n_cut_edges++] = i;
            
            /* Add to boundary of both partitions */
            int pa = tn->node_partitions[edge->node_a];
            int pb = tn->node_partitions[edge->node_b];
            
            TNPartition *part_a = &result->partitions[pa];
            TNPartition *part_b = &result->partitions[pb];
            
            part_a->boundary_edges = realloc(part_a->boundary_edges,
                                              (part_a->n_boundary + 1) * sizeof(uint32_t));
            part_a->boundary_edges[part_a->n_boundary++] = i;
            
            part_b->boundary_edges = realloc(part_b->boundary_edges,
                                              (part_b->n_boundary + 1) * sizeof(uint32_t));
            part_b->boundary_edges[part_b->n_boundary++] = i;
            
            result->communication_cost += edge->dim * sizeof(float);
        }
    }
    
    return result;
}

/* Metis-style multilevel partitioning (simplified) */
static TNPartitioning* multilevel_partition(CogTensorNetwork *tn, int n_parts) {
    /* For now, fall back to greedy */
    return greedy_partition(tn, n_parts);
}

COGINT_API int cog_tn_partition(CogTensorNetwork *tn, int n_partitions) {
    if (!tn || n_partitions <= 0) return COG_ERR_INVALID;
    
    TNPartitioning *partitioning = multilevel_partition(tn, n_partitions);
    if (!partitioning) return COG_ERR_NOMEM;
    
    tn->n_partitions = n_partitions;
    
    /* Store partition assignments in nodes */
    for (int p = 0; p < n_partitions; p++) {
        for (size_t i = 0; i < partitioning->partitions[p].n_nodes; i++) {
            uint32_t node_id = partitioning->partitions[p].nodes[i];
            tn->nodes[node_id]->priority = p;  /* Reuse priority field */
        }
    }
    
    /* Free partitioning structure (keep assignments in tn) */
    for (int i = 0; i < n_partitions; i++) {
        free(partitioning->partitions[i].nodes);
        free(partitioning->partitions[i].boundary_edges);
    }
    free(partitioning->partitions);
    free(partitioning->cut_edges);
    free(partitioning);
    
    return COG_OK;
}

/*============================================================================
 * Distributed Contraction
 *============================================================================*/

/* Worker task for distributed contraction */
typedef struct {
    CogTensorNetwork *tn;
    CogContractionPath *path;
    int partition_id;
    size_t start_step;
    size_t end_step;
    CogTensor **intermediates;
    CogChan *result_chan;
} DistContractTask;

static int dist_contract_execute(CogTask *task, CogContext *ctx) {
    DistContractTask *dct = (DistContractTask*)task->params;
    
    /* Execute assigned contraction steps */
    for (size_t i = dct->start_step; i < dct->end_step; i++) {
        CogContractionStep *step = &dct->path->steps[i];
        
        CogTensor *a = dct->intermediates[step->node_a];
        CogTensor *b = dct->intermediates[step->node_b];
        
        if (!a || !b) continue;
        
        /* Find contraction axes */
        int axes_a[8], axes_b[8];
        int n_axes = 0;
        
        /* Simple case: contract along matching dimensions */
        if (a->ndim > 0 && b->ndim > 0 && a->shape[a->ndim - 1] == b->shape[0]) {
            axes_a[0] = a->ndim - 1;
            axes_b[0] = 0;
            n_axes = 1;
        }
        
        CogTensor *result = cog_tensor_contract(a, b, axes_a, axes_b, n_axes);
        dct->intermediates[step->result_node] = result;
    }
    
    return COG_OK;
}

COGINT_API CogTensor* cog_tn_contract_distributed(CogTensorNetwork *tn,
                                                   CogContractionPath *path,
                                                   CogRuntime *rt) {
    if (!tn || !path || !rt) return NULL;
    
    /* Partition if not already done */
    if (tn->n_partitions == 0) {
        int n_workers = rt->local_pool ? rt->local_pool->n_workers : 4;
        cog_tn_partition(tn, n_workers);
    }
    
    /* Allocate shared intermediate storage */
    size_t n_intermediates = tn->n_nodes + path->n_steps;
    CogTensor **intermediates = calloc(n_intermediates, sizeof(CogTensor*));
    
    /* Copy initial tensors */
    for (size_t i = 0; i < tn->n_nodes; i++) {
        intermediates[i] = tn->nodes[i]->tensor;
    }
    
    /* Create result channel */
    CogChan *result_chan = cog_chan_create("dist_results", COG_CHAN_TENSOR, 64);
    
    /* Assign steps to partitions based on node locations */
    int *step_partition = malloc(path->n_steps * sizeof(int));
    for (size_t i = 0; i < path->n_steps; i++) {
        CogContractionStep *step = &path->steps[i];
        
        /* Assign to partition of first operand */
        if (step->node_a < tn->n_nodes) {
            step_partition[i] = tn->node_partitions[step->node_a];
        } else {
            step_partition[i] = 0;
        }
    }
    
    /* Group consecutive steps for same partition */
    size_t current_start = 0;
    int current_part = step_partition[0];
    
    CogTaskGraph *graph = cog_graph_create();
    
    for (size_t i = 1; i <= path->n_steps; i++) {
        int part = (i < path->n_steps) ? step_partition[i] : -1;
        
        if (part != current_part || i == path->n_steps) {
            /* Create task for current range */
            DistContractTask *dct = malloc(sizeof(DistContractTask));
            dct->tn = tn;
            dct->path = path;
            dct->partition_id = current_part;
            dct->start_step = current_start;
            dct->end_step = i;
            dct->intermediates = intermediates;
            dct->result_chan = result_chan;
            
            CogTask *task = cog_task_custom(dist_contract_execute, dct, sizeof(DistContractTask));
            cog_graph_add(graph, task);
            
            current_start = i;
            current_part = part;
        }
    }
    
    /* Execute task graph */
    cog_graph_execute(graph, rt->local_pool);
    
    /* Get final result */
    CogTensor *result = NULL;
    if (path->n_steps > 0) {
        result = intermediates[path->steps[path->n_steps - 1].result_node];
        if (result) {
            result = cog_tensor_clone(result);  /* Make a copy */
        }
    }
    
    /* Cleanup */
    for (size_t i = tn->n_nodes; i < n_intermediates; i++) {
        if (intermediates[i]) cog_tensor_free(intermediates[i]);
    }
    free(intermediates);
    free(step_partition);
    cog_graph_free(graph);
    cog_chan_free(result_chan);
    
    return result;
}

/*============================================================================
 * Sliced Contraction
 *============================================================================*/

COGINT_API CogTensor* cog_tn_contract_sliced(CogTensorNetwork *tn,
                                              CogContractionPath *path,
                                              int slice_dim, int n_slices) {
    if (!tn || !path || n_slices <= 0) return NULL;
    
    /* Find the edge to slice */
    CogTNEdge *slice_edge = NULL;
    for (size_t i = 0; i < tn->n_edges; i++) {
        if (tn->edges[i]->dim == slice_dim) {
            slice_edge = tn->edges[i];
            break;
        }
    }
    
    if (!slice_edge) {
        /* No suitable edge found, use regular contraction */
        return cog_tn_contract(tn, path);
    }
    
    /* Determine slice size */
    int64_t slice_size = (slice_edge->dim + n_slices - 1) / n_slices;
    
    /* Accumulate results from each slice */
    CogTensor *result = NULL;
    
    for (int s = 0; s < n_slices; s++) {
        int64_t start = s * slice_size;
        int64_t end = (s + 1) * slice_size;
        if (end > slice_edge->dim) end = slice_edge->dim;
        
        /* Create sliced tensors */
        CogTNNode *node_a = tn->nodes[slice_edge->node_a];
        CogTNNode *node_b = tn->nodes[slice_edge->node_b];
        
        /* Slice tensor A along axis_a */
        int64_t new_shape_a[8];
        memcpy(new_shape_a, node_a->shape, node_a->ndim * sizeof(int64_t));
        new_shape_a[slice_edge->axis_a] = end - start;
        
        CogTensor *sliced_a = cog_tensor_create(NULL, new_shape_a, node_a->ndim,
                                                 node_a->tensor->dtype);
        
        /* Copy sliced data (simplified - assumes contiguous along slice axis) */
        float *src_a = (float*)node_a->tensor->data;
        float *dst_a = (float*)sliced_a->data;
        
        int64_t pre_size = 1, post_size = 1;
        for (int i = 0; i < slice_edge->axis_a; i++) {
            pre_size *= node_a->shape[i];
        }
        for (int i = slice_edge->axis_a + 1; i < node_a->ndim; i++) {
            post_size *= node_a->shape[i];
        }
        
        for (int64_t p = 0; p < pre_size; p++) {
            for (int64_t i = start; i < end; i++) {
                for (int64_t q = 0; q < post_size; q++) {
                    dst_a[p * (end - start) * post_size + (i - start) * post_size + q] =
                        src_a[p * slice_edge->dim * post_size + i * post_size + q];
                }
            }
        }
        
        /* Similarly slice tensor B */
        int64_t new_shape_b[8];
        memcpy(new_shape_b, node_b->shape, node_b->ndim * sizeof(int64_t));
        new_shape_b[slice_edge->axis_b] = end - start;
        
        CogTensor *sliced_b = cog_tensor_create(NULL, new_shape_b, node_b->ndim,
                                                 node_b->tensor->dtype);
        
        float *src_b = (float*)node_b->tensor->data;
        float *dst_b = (float*)sliced_b->data;
        
        pre_size = 1;
        post_size = 1;
        for (int i = 0; i < slice_edge->axis_b; i++) {
            pre_size *= node_b->shape[i];
        }
        for (int i = slice_edge->axis_b + 1; i < node_b->ndim; i++) {
            post_size *= node_b->shape[i];
        }
        
        for (int64_t p = 0; p < pre_size; p++) {
            for (int64_t i = start; i < end; i++) {
                for (int64_t q = 0; q < post_size; q++) {
                    dst_b[p * (end - start) * post_size + (i - start) * post_size + q] =
                        src_b[p * slice_edge->dim * post_size + i * post_size + q];
                }
            }
        }
        
        /* Create temporary network with sliced tensors */
        CogTensor *orig_a = node_a->tensor;
        CogTensor *orig_b = node_b->tensor;
        node_a->tensor = sliced_a;
        node_b->tensor = sliced_b;
        
        /* Contract this slice */
        CogTensor *slice_result = cog_tn_contract(tn, path);
        
        /* Restore original tensors */
        node_a->tensor = orig_a;
        node_b->tensor = orig_b;
        cog_tensor_free(sliced_a);
        cog_tensor_free(sliced_b);
        
        /* Accumulate result */
        if (!result) {
            result = slice_result;
        } else if (slice_result) {
            /* Add slice_result to result */
            float *r = (float*)result->data;
            float *sr = (float*)slice_result->data;
            for (size_t i = 0; i < result->numel; i++) {
                r[i] += sr[i];
            }
            cog_tensor_free(slice_result);
        }
    }
    
    return result;
}

/*============================================================================
 * MPS Operations
 *============================================================================*/

COGINT_API CogMPS* cog_mps_from_tensor(CogTensor *tensor, int64_t max_bond) {
    if (!tensor) return NULL;
    
    CogMPS *mps = calloc(1, sizeof(CogMPS));
    mps->n_sites = tensor->ndim;
    mps->tensors = malloc(mps->n_sites * sizeof(CogTensor*));
    mps->bond_dims = malloc((mps->n_sites + 1) * sizeof(int64_t));
    
    /* TT decomposition to get MPS */
    int64_t *ranks = malloc(mps->n_sites * sizeof(int64_t));
    for (int i = 0; i < mps->n_sites; i++) {
        ranks[i] = max_bond;
    }
    
    CogDecompResult *tt = cog_tensor_tt(tensor, ranks, mps->n_sites);
    
    if (tt && tt->tt_cores) {
        for (int i = 0; i < mps->n_sites; i++) {
            mps->tensors[i] = tt->tt_cores[i];
            tt->tt_cores[i] = NULL;  /* Transfer ownership */
            
            mps->bond_dims[i] = mps->tensors[i]->shape[0];
        }
        mps->bond_dims[mps->n_sites] = 1;
        mps->phys_dim = tensor->shape[0];
    }
    
    free(ranks);
    cog_decomp_free(tt);
    
    return mps;
}

COGINT_API CogMPS* cog_mps_random(int n_sites, int64_t phys_dim, int64_t bond_dim) {
    CogMPS *mps = calloc(1, sizeof(CogMPS));
    mps->n_sites = n_sites;
    mps->phys_dim = phys_dim;
    mps->tensors = malloc(n_sites * sizeof(CogTensor*));
    mps->bond_dims = malloc((n_sites + 1) * sizeof(int64_t));
    
    mps->bond_dims[0] = 1;
    for (int i = 1; i < n_sites; i++) {
        mps->bond_dims[i] = bond_dim;
    }
    mps->bond_dims[n_sites] = 1;
    
    /* Create random site tensors */
    for (int i = 0; i < n_sites; i++) {
        int64_t shape[] = {mps->bond_dims[i], phys_dim, mps->bond_dims[i + 1]};
        mps->tensors[i] = cog_tensor_create(NULL, shape, 3, COG_DTYPE_FLOAT32);
        
        /* Initialize with random values */
        float *data = (float*)mps->tensors[i]->data;
        for (size_t j = 0; j < mps->tensors[i]->numel; j++) {
            data[j] = (float)rand() / RAND_MAX - 0.5f;
        }
    }
    
    return mps;
}

COGINT_API CogTensor* cog_mps_to_tensor(CogMPS *mps) {
    if (!mps || mps->n_sites == 0) return NULL;
    
    /* Contract all MPS tensors */
    CogTensor *result = cog_tensor_clone(mps->tensors[0]);
    
    for (int i = 1; i < mps->n_sites; i++) {
        /* Contract along bond dimension */
        int axes_a[] = {result->ndim - 1};
        int axes_b[] = {0};
        
        CogTensor *new_result = cog_tensor_contract(result, mps->tensors[i],
                                                     axes_a, axes_b, 1);
        cog_tensor_free(result);
        result = new_result;
        
        if (!result) return NULL;
    }
    
    return result;
}

COGINT_API int cog_mps_canonicalize(CogMPS *mps, int center) {
    if (!mps || center < 0 || center >= mps->n_sites) return COG_ERR_INVALID;
    
    /* Left-canonicalize from 0 to center-1 */
    for (int i = 0; i < center; i++) {
        CogTensor *A = mps->tensors[i];
        
        /* Reshape to matrix: (r_i * d) x r_{i+1} */
        int64_t m = A->shape[0] * A->shape[1];
        int64_t n = A->shape[2];
        
        int64_t mat_shape[] = {m, n};
        CogTensor *mat = cog_tensor_create(NULL, mat_shape, 2, A->dtype);
        memcpy(mat->data, A->data, A->numel * sizeof(float));
        
        /* QR decomposition */
        CogDecompResult *qr = cog_tensor_qr(mat);
        
        /* Update A with Q */
        int64_t new_r = qr->Q->shape[1];
        int64_t new_shape[] = {A->shape[0], A->shape[1], new_r};
        cog_tensor_free(mps->tensors[i]);
        mps->tensors[i] = cog_tensor_create(NULL, new_shape, 3, A->dtype);
        memcpy(mps->tensors[i]->data, qr->Q->data, 
               A->shape[0] * A->shape[1] * new_r * sizeof(float));
        
        /* Absorb R into next tensor */
        if (i + 1 < mps->n_sites) {
            CogTensor *B = mps->tensors[i + 1];
            
            /* Contract R with B */
            int axes_a[] = {1};
            int axes_b[] = {0};
            CogTensor *new_B = cog_tensor_contract(qr->R, B, axes_a, axes_b, 1);
            
            cog_tensor_free(mps->tensors[i + 1]);
            mps->tensors[i + 1] = new_B;
        }
        
        mps->bond_dims[i + 1] = new_r;
        
        cog_decomp_free(qr);
        cog_tensor_free(mat);
    }
    
    /* Right-canonicalize from n_sites-1 to center+1 */
    for (int i = mps->n_sites - 1; i > center; i--) {
        CogTensor *A = mps->tensors[i];
        
        /* Reshape to matrix: r_i x (d * r_{i+1}) */
        int64_t m = A->shape[0];
        int64_t n = A->shape[1] * A->shape[2];
        
        int64_t mat_shape[] = {m, n};
        CogTensor *mat = cog_tensor_create(NULL, mat_shape, 2, A->dtype);
        
        /* Transpose for RQ decomposition */
        float *src = (float*)A->data;
        float *dst = (float*)mat->data;
        for (int64_t j = 0; j < m; j++) {
            for (int64_t k = 0; k < n; k++) {
                dst[j * n + k] = src[j * n + k];
            }
        }
        
        /* QR of transpose gives RQ */
        int64_t trans_shape[] = {n, m};
        CogTensor *trans = cog_tensor_create(NULL, trans_shape, 2, A->dtype);
        for (int64_t j = 0; j < m; j++) {
            for (int64_t k = 0; k < n; k++) {
                ((float*)trans->data)[k * m + j] = dst[j * n + k];
            }
        }
        
        CogDecompResult *qr = cog_tensor_qr(trans);
        
        /* Update A with Q^T */
        int64_t new_r = qr->Q->shape[1];
        int64_t new_shape[] = {new_r, A->shape[1], A->shape[2]};
        cog_tensor_free(mps->tensors[i]);
        mps->tensors[i] = cog_tensor_create(NULL, new_shape, 3, A->dtype);
        
        /* Transpose Q back */
        float *Q_data = (float*)qr->Q->data;
        float *new_data = (float*)mps->tensors[i]->data;
        for (int64_t j = 0; j < new_r; j++) {
            for (int64_t k = 0; k < A->shape[1] * A->shape[2]; k++) {
                new_data[j * A->shape[1] * A->shape[2] + k] = Q_data[k * new_r + j];
            }
        }
        
        /* Absorb R^T into previous tensor */
        if (i > 0) {
            CogTensor *B = mps->tensors[i - 1];
            
            /* Contract B with R^T */
            int axes_a[] = {2};
            int axes_b[] = {1};
            
            /* Transpose R */
            int64_t rt_shape[] = {qr->R->shape[1], qr->R->shape[0]};
            CogTensor *RT = cog_tensor_create(NULL, rt_shape, 2, A->dtype);
            float *R_data = (float*)qr->R->data;
            float *RT_data = (float*)RT->data;
            for (int64_t j = 0; j < qr->R->shape[0]; j++) {
                for (int64_t k = 0; k < qr->R->shape[1]; k++) {
                    RT_data[k * qr->R->shape[0] + j] = R_data[j * qr->R->shape[1] + k];
                }
            }
            
            CogTensor *new_B = cog_tensor_contract(B, RT, axes_a, axes_b, 1);
            
            cog_tensor_free(mps->tensors[i - 1]);
            mps->tensors[i - 1] = new_B;
            cog_tensor_free(RT);
        }
        
        mps->bond_dims[i] = new_r;
        
        cog_decomp_free(qr);
        cog_tensor_free(mat);
        cog_tensor_free(trans);
    }
    
    mps->center = center;
    mps->is_canonical = 1;
    
    return COG_OK;
}

COGINT_API double cog_mps_inner(CogMPS *mps1, CogMPS *mps2) {
    if (!mps1 || !mps2 || mps1->n_sites != mps2->n_sites) return 0.0;
    
    /* Contract from left to right */
    CogTensor *left = NULL;
    
    for (int i = 0; i < mps1->n_sites; i++) {
        CogTensor *A = mps1->tensors[i];
        CogTensor *B = mps2->tensors[i];
        
        if (!left) {
            /* First site: contract physical indices */
            int axes_a[] = {1};
            int axes_b[] = {1};
            left = cog_tensor_contract(A, B, axes_a, axes_b, 1);
        } else {
            /* Contract left environment with A */
            int axes_a[] = {0};
            int axes_b[] = {0};
            CogTensor *temp = cog_tensor_contract(left, A, axes_a, axes_b, 1);
            cog_tensor_free(left);
            
            /* Contract with B */
            int axes_c[] = {0, 1};
            int axes_d[] = {0, 1};
            left = cog_tensor_contract(temp, B, axes_c, axes_d, 2);
            cog_tensor_free(temp);
        }
    }
    
    /* Extract scalar result */
    double result = 0.0;
    if (left && left->numel == 1) {
        result = ((float*)left->data)[0];
    }
    
    if (left) cog_tensor_free(left);
    
    return result;
}

COGINT_API int cog_mps_compress(CogMPS *mps, int64_t max_bond, double truncation) {
    if (!mps) return COG_ERR_INVALID;
    
    /* Canonicalize to leftmost site */
    cog_mps_canonicalize(mps, 0);
    
    /* Sweep right, truncating bonds */
    for (int i = 0; i < mps->n_sites - 1; i++) {
        CogTensor *A = mps->tensors[i];
        
        /* Reshape to matrix */
        int64_t m = A->shape[0] * A->shape[1];
        int64_t n = A->shape[2];
        
        int64_t mat_shape[] = {m, n};
        CogTensor *mat = cog_tensor_create(NULL, mat_shape, 2, A->dtype);
        memcpy(mat->data, A->data, A->numel * sizeof(float));
        
        /* Truncated SVD */
        CogDecompResult *svd = cog_tensor_svd(mat, max_bond, truncation);
        
        /* Update A with U */
        int64_t new_r = svd->S->shape[0];
        int64_t new_shape[] = {A->shape[0], A->shape[1], new_r};
        cog_tensor_free(mps->tensors[i]);
        mps->tensors[i] = cog_tensor_create(NULL, new_shape, 3, A->dtype);
        memcpy(mps->tensors[i]->data, svd->U->data,
               A->shape[0] * A->shape[1] * new_r * sizeof(float));
        
        /* Absorb S*V^T into next tensor */
        CogTensor *B = mps->tensors[i + 1];
        
        /* Compute S*V^T */
        int64_t sv_shape[] = {new_r, svd->V->shape[0]};
        CogTensor *SV = cog_tensor_create(NULL, sv_shape, 2, A->dtype);
        float *S = (float*)svd->S->data;
        float *V = (float*)svd->V->data;
        float *sv = (float*)SV->data;
        
        for (int64_t j = 0; j < new_r; j++) {
            for (int64_t k = 0; k < svd->V->shape[0]; k++) {
                sv[j * svd->V->shape[0] + k] = S[j] * V[k * new_r + j];
            }
        }
        
        /* Contract SV with B */
        int axes_a[] = {1};
        int axes_b[] = {0};
        CogTensor *new_B = cog_tensor_contract(SV, B, axes_a, axes_b, 1);
        
        cog_tensor_free(mps->tensors[i + 1]);
        mps->tensors[i + 1] = new_B;
        
        mps->bond_dims[i + 1] = new_r;
        
        cog_decomp_free(svd);
        cog_tensor_free(mat);
        cog_tensor_free(SV);
    }
    
    return COG_OK;
}

COGINT_API void cog_mps_free(CogMPS *mps) {
    if (!mps) return;
    
    for (int i = 0; i < mps->n_sites; i++) {
        if (mps->tensors[i]) cog_tensor_free(mps->tensors[i]);
    }
    free(mps->tensors);
    free(mps->bond_dims);
    
    if (mps->environments_L) {
        for (int i = 0; i < mps->n_sites; i++) {
            if (mps->environments_L[i]) cog_tensor_free(mps->environments_L[i]);
        }
        free(mps->environments_L);
    }
    
    if (mps->environments_R) {
        for (int i = 0; i < mps->n_sites; i++) {
            if (mps->environments_R[i]) cog_tensor_free(mps->environments_R[i]);
        }
        free(mps->environments_R);
    }
    
    free(mps);
}

COGINT_API void cog_mpo_free(CogMPO *mpo) {
    if (!mpo) return;
    
    for (int i = 0; i < mpo->n_sites; i++) {
        if (mpo->tensors[i]) cog_tensor_free(mpo->tensors[i]);
    }
    free(mpo->tensors);
    free(mpo->bond_dims);
    free(mpo);
}
