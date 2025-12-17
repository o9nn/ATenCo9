/**
 * CogTensorNet - Tensor Network Optimization for ATenCo9
 * 
 * This header defines tensor network structures and optimization algorithms:
 * 
 * - Tensor Network Representations (MPS, MPO, PEPS, TTN, MERA)
 * - Contraction Ordering Optimization (greedy, dynamic programming, simulated annealing)
 * - Tensor Decomposition (SVD, QR, Tucker, Tensor Train)
 * - Distributed Tensor Network Computation
 * - Automatic Differentiation for Tensor Networks
 * 
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    Tensor Network Layer                          │
 * │  ┌─────────────────────────────────────────────────────────────┐│
 * │  │                 Network Structures                           ││
 * │  │  MPS  │  MPO  │  PEPS  │  TTN  │  MERA  │  Custom           ││
 * │  └─────────────────────────────────────────────────────────────┘│
 * │  ┌─────────────────────────────────────────────────────────────┐│
 * │  │              Contraction Optimizer                           ││
 * │  │  Greedy  │  DP  │  Simulated Annealing  │  Genetic          ││
 * │  └─────────────────────────────────────────────────────────────┘│
 * │  ┌─────────────────────────────────────────────────────────────┐│
 * │  │              Decomposition Engine                            ││
 * │  │  SVD  │  QR  │  Tucker  │  TT  │  Hierarchical Tucker       ││
 * │  └─────────────────────────────────────────────────────────────┘│
 * │  ┌─────────────────────────────────────────────────────────────┐│
 * │  │              Distributed Execution                           ││
 * │  │  Partitioning  │  Load Balancing  │  Communication          ││
 * │  └─────────────────────────────────────────────────────────────┘│
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#ifndef COG_TENSORNET_H
#define COG_TENSORNET_H

#include "cogint.h"

/* Forward declarations */
typedef struct CogRuntime CogRuntime;

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Tensor Network Node Types
 *============================================================================*/

typedef enum {
    COG_TN_NODE_TENSOR = 0,      /* Regular tensor node */
    COG_TN_NODE_IDENTITY = 1,    /* Identity tensor */
    COG_TN_NODE_DELTA = 2,       /* Kronecker delta */
    COG_TN_NODE_RANDOM = 3,      /* Random tensor */
    COG_TN_NODE_ZERO = 4,        /* Zero tensor */
    COG_TN_NODE_COPY = 5,        /* Copy tensor (for broadcasting) */
} CogTNNodeType;

/*============================================================================
 * Tensor Network Edge (Index/Bond)
 *============================================================================*/

typedef struct CogTNEdge {
    uint32_t id;
    char *name;
    int64_t dim;                 /* Bond dimension */
    
    /* Connected nodes */
    uint32_t node_a;
    uint32_t node_b;
    int axis_a;                  /* Axis on node_a */
    int axis_b;                  /* Axis on node_b */
    
    /* Edge properties */
    int is_open;                 /* Open (external) index */
    int is_traced;               /* Traced (summed) index */
    double weight;               /* Edge weight for optimization */
    
    /* For distributed computation */
    int partition_id;
} CogTNEdge;

/*============================================================================
 * Tensor Network Node
 *============================================================================*/

typedef struct CogTNNode {
    uint32_t id;
    char *name;
    CogTNNodeType type;
    
    /* Tensor data */
    CogTensor *tensor;
    int64_t *shape;
    int ndim;
    
    /* Connectivity */
    uint32_t *edges;             /* Edge IDs */
    int *edge_axes;              /* Which axis each edge connects to */
    size_t n_edges;
    
    /* Optimization metadata */
    double cost;                 /* Computational cost */
    int priority;                /* Contraction priority */
    int visited;                 /* For graph traversal */
    
    /* Gradient for autodiff */
    CogTensor *grad;
    int requires_grad;
} CogTNNode;

/*============================================================================
 * Tensor Network Structure
 *============================================================================*/

typedef enum {
    COG_TN_TYPE_GENERAL = 0,     /* General tensor network */
    COG_TN_TYPE_MPS = 1,         /* Matrix Product State */
    COG_TN_TYPE_MPO = 2,         /* Matrix Product Operator */
    COG_TN_TYPE_PEPS = 3,        /* Projected Entangled Pair State */
    COG_TN_TYPE_TTN = 4,         /* Tree Tensor Network */
    COG_TN_TYPE_MERA = 5,        /* Multi-scale Entanglement Renormalization */
    COG_TN_TYPE_TT = 6,          /* Tensor Train */
} CogTNType;

typedef struct CogTensorNetwork {
    char *name;
    CogTNType type;
    
    /* Nodes and edges */
    CogTNNode **nodes;
    size_t n_nodes;
    size_t node_capacity;
    
    CogTNEdge **edges;
    size_t n_edges;
    size_t edge_capacity;
    
    /* Open indices (external legs) */
    uint32_t *open_edges;
    size_t n_open;
    
    /* Contraction order */
    uint32_t *contraction_order;
    size_t order_len;
    double total_cost;
    
    /* For distributed computation */
    int n_partitions;
    int *node_partitions;
    
    /* Context */
    CogContext *ctx;
} CogTensorNetwork;

/*============================================================================
 * Contraction Path
 *============================================================================*/

typedef struct {
    uint32_t node_a;
    uint32_t node_b;
    uint32_t result_node;
    double flops;
    int64_t memory;
} CogContractionStep;

typedef struct {
    CogContractionStep *steps;
    size_t n_steps;
    double total_flops;
    int64_t peak_memory;
    double estimated_time;
} CogContractionPath;

/*============================================================================
 * Contraction Optimizer Configuration
 *============================================================================*/

typedef enum {
    COG_CONTRACT_GREEDY = 0,
    COG_CONTRACT_OPTIMAL = 1,    /* Dynamic programming (exact) */
    COG_CONTRACT_DP = 2,         /* Dynamic programming with pruning */
    COG_CONTRACT_RANDOM_GREEDY = 3,
    COG_CONTRACT_BRANCH_BOUND = 4,
    COG_CONTRACT_SIMULATED_ANNEALING = 5,
    COG_CONTRACT_GENETIC = 6,
} CogContractMethod;

typedef struct {
    CogContractMethod method;
    
    /* Greedy options */
    int greedy_trials;           /* Number of random restarts */
    double greedy_temperature;   /* For random-greedy */
    
    /* DP options */
    int64_t dp_memory_limit;     /* Max memory for DP table */
    int dp_max_width;            /* Max intermediate tensor size */
    
    /* Simulated annealing options */
    double sa_initial_temp;
    double sa_cooling_rate;
    int sa_iterations;
    
    /* Genetic algorithm options */
    int ga_population;
    int ga_generations;
    double ga_mutation_rate;
    double ga_crossover_rate;
    
    /* General options */
    int minimize_memory;         /* Prioritize memory over flops */
    int allow_slicing;           /* Allow index slicing */
    int max_slice_count;
    double timeout_seconds;
} CogContractConfig;

/*============================================================================
 * Tensor Decomposition Configuration
 *============================================================================*/

typedef enum {
    COG_DECOMP_SVD = 0,          /* Singular Value Decomposition */
    COG_DECOMP_QR = 1,           /* QR Decomposition */
    COG_DECOMP_LU = 2,           /* LU Decomposition */
    COG_DECOMP_CHOLESKY = 3,     /* Cholesky Decomposition */
    COG_DECOMP_TUCKER = 4,       /* Tucker Decomposition */
    COG_DECOMP_TT = 5,           /* Tensor Train Decomposition */
    COG_DECOMP_HOSVD = 6,        /* Higher-Order SVD */
    COG_DECOMP_CP = 7,           /* CP (CANDECOMP/PARAFAC) */
} CogDecompMethod;

typedef struct {
    CogDecompMethod method;
    
    /* SVD options */
    int64_t max_rank;            /* Maximum rank (0 = full) */
    double truncation_error;     /* Truncation threshold */
    int compute_uv;              /* Compute U and V matrices */
    
    /* Tucker options */
    int64_t *tucker_ranks;       /* Ranks for each mode */
    int tucker_n_modes;
    
    /* TT options */
    int64_t *tt_ranks;           /* TT ranks */
    int tt_n_cores;
    
    /* CP options */
    int cp_rank;
    int cp_max_iter;
    double cp_tolerance;
    
    /* General options */
    int use_randomized;          /* Use randomized algorithms */
    int oversampling;            /* Oversampling for randomized SVD */
    int power_iterations;        /* Power iterations for accuracy */
} CogDecompConfig;

/*============================================================================
 * Decomposition Result
 *============================================================================*/

typedef struct {
    CogDecompMethod method;
    
    /* SVD result: A = U * S * V^T */
    CogTensor *U;
    CogTensor *S;                /* Singular values */
    CogTensor *V;
    
    /* QR result: A = Q * R */
    CogTensor *Q;
    CogTensor *R;
    
    /* Tucker result: G, U1, U2, ... */
    CogTensor *core;             /* Core tensor */
    CogTensor **factors;         /* Factor matrices */
    int n_factors;
    
    /* TT result: cores */
    CogTensor **tt_cores;
    int n_cores;
    
    /* Statistics */
    double relative_error;
    double compression_ratio;
    int64_t original_size;
    int64_t compressed_size;
} CogDecompResult;

/*============================================================================
 * MPS (Matrix Product State) Structure
 *============================================================================*/

typedef struct {
    CogTensor **tensors;         /* Site tensors */
    int n_sites;
    int64_t *bond_dims;          /* Bond dimensions */
    int64_t phys_dim;            /* Physical dimension */
    
    /* Canonical form */
    int center;                  /* Orthogonality center */
    int is_canonical;
    
    /* For DMRG */
    CogTensor **environments_L;
    CogTensor **environments_R;
} CogMPS;

/*============================================================================
 * MPO (Matrix Product Operator) Structure
 *============================================================================*/

typedef struct {
    CogTensor **tensors;         /* Operator tensors */
    int n_sites;
    int64_t *bond_dims;
    int64_t phys_dim_in;
    int64_t phys_dim_out;
} CogMPO;

/*============================================================================
 * Tensor Network API
 *============================================================================*/

/**
 * Create a tensor network
 */
COGINT_API CogTensorNetwork* cog_tn_create(CogContext *ctx, const char *name,
                                            CogTNType type);

/**
 * Free a tensor network
 */
COGINT_API void cog_tn_free(CogTensorNetwork *tn);

/**
 * Add a tensor node to the network
 */
COGINT_API uint32_t cog_tn_add_node(CogTensorNetwork *tn, CogTensor *tensor,
                                     const char *name);

/**
 * Add an edge (contraction) between two nodes
 */
COGINT_API uint32_t cog_tn_add_edge(CogTensorNetwork *tn, 
                                     uint32_t node_a, int axis_a,
                                     uint32_t node_b, int axis_b,
                                     const char *name);

/**
 * Mark an edge as open (external index)
 */
COGINT_API int cog_tn_mark_open(CogTensorNetwork *tn, uint32_t edge_id);

/**
 * Get node by ID
 */
COGINT_API CogTNNode* cog_tn_get_node(CogTensorNetwork *tn, uint32_t id);

/**
 * Get edge by ID
 */
COGINT_API CogTNEdge* cog_tn_get_edge(CogTensorNetwork *tn, uint32_t id);

/*============================================================================
 * Contraction API
 *============================================================================*/

/**
 * Find optimal contraction path
 */
COGINT_API CogContractionPath* cog_tn_optimize_path(CogTensorNetwork *tn,
                                                     CogContractConfig *config);

/**
 * Contract the tensor network using the computed path
 */
COGINT_API CogTensor* cog_tn_contract(CogTensorNetwork *tn,
                                       CogContractionPath *path);

/**
 * Contract two tensors along specified axes
 */
COGINT_API CogTensor* cog_tensor_contract(CogTensor *a, CogTensor *b,
                                           int *axes_a, int *axes_b, int n_axes);

/**
 * Einsum-style contraction
 */
COGINT_API CogTensor* cog_tensor_einsum(const char *subscripts,
                                         CogTensor **tensors, int n_tensors);

/**
 * Free contraction path
 */
COGINT_API void cog_path_free(CogContractionPath *path);

/*============================================================================
 * Decomposition API
 *============================================================================*/

/**
 * Decompose a tensor
 */
COGINT_API CogDecompResult* cog_tensor_decompose(CogTensor *tensor,
                                                  CogDecompConfig *config);

/**
 * SVD decomposition
 */
COGINT_API CogDecompResult* cog_tensor_svd(CogTensor *tensor, int64_t max_rank,
                                            double truncation);

/**
 * QR decomposition
 */
COGINT_API CogDecompResult* cog_tensor_qr(CogTensor *tensor);

/**
 * Tucker decomposition
 */
COGINT_API CogDecompResult* cog_tensor_tucker(CogTensor *tensor,
                                               int64_t *ranks, int n_modes);

/**
 * Tensor Train decomposition
 */
COGINT_API CogDecompResult* cog_tensor_tt(CogTensor *tensor,
                                           int64_t *ranks, int n_cores);

/**
 * Reconstruct tensor from decomposition
 */
COGINT_API CogTensor* cog_decomp_reconstruct(CogDecompResult *decomp);

/**
 * Free decomposition result
 */
COGINT_API void cog_decomp_free(CogDecompResult *decomp);

/*============================================================================
 * MPS/MPO API
 *============================================================================*/

/**
 * Create MPS from tensor
 */
COGINT_API CogMPS* cog_mps_from_tensor(CogTensor *tensor, int64_t max_bond);

/**
 * Create MPS with random initialization
 */
COGINT_API CogMPS* cog_mps_random(int n_sites, int64_t phys_dim, int64_t bond_dim);

/**
 * Contract MPS to tensor
 */
COGINT_API CogTensor* cog_mps_to_tensor(CogMPS *mps);

/**
 * Canonicalize MPS
 */
COGINT_API int cog_mps_canonicalize(CogMPS *mps, int center);

/**
 * MPS-MPS inner product
 */
COGINT_API double cog_mps_inner(CogMPS *mps1, CogMPS *mps2);

/**
 * Apply MPO to MPS
 */
COGINT_API CogMPS* cog_mpo_apply(CogMPO *mpo, CogMPS *mps);

/**
 * Compress MPS to target bond dimension
 */
COGINT_API int cog_mps_compress(CogMPS *mps, int64_t max_bond, double truncation);

/**
 * Free MPS
 */
COGINT_API void cog_mps_free(CogMPS *mps);

/**
 * Free MPO
 */
COGINT_API void cog_mpo_free(CogMPO *mpo);

/*============================================================================
 * Distributed Tensor Network API
 *============================================================================*/

/**
 * Partition tensor network for distributed computation
 */
COGINT_API int cog_tn_partition(CogTensorNetwork *tn, int n_partitions);

/**
 * Distributed contraction
 */
COGINT_API CogTensor* cog_tn_contract_distributed(CogTensorNetwork *tn,
                                                   CogContractionPath *path,
                                                   CogRuntime *rt);

/**
 * Sliced contraction (for memory-limited cases)
 */
COGINT_API CogTensor* cog_tn_contract_sliced(CogTensorNetwork *tn,
                                              CogContractionPath *path,
                                              int slice_dim, int n_slices);

/*============================================================================
 * Optimization Utilities
 *============================================================================*/

/**
 * Estimate contraction cost (FLOPs)
 */
COGINT_API double cog_tn_estimate_flops(CogTensorNetwork *tn,
                                         CogContractionPath *path);

/**
 * Estimate peak memory usage
 */
COGINT_API int64_t cog_tn_estimate_memory(CogTensorNetwork *tn,
                                           CogContractionPath *path);

/**
 * Visualize tensor network (DOT format)
 */
COGINT_API char* cog_tn_to_dot(CogTensorNetwork *tn);

/**
 * Print contraction path
 */
COGINT_API void cog_path_print(CogContractionPath *path);

#ifdef __cplusplus
}
#endif

#endif /* COG_TENSORNET_H */
