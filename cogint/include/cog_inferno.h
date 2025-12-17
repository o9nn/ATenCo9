/**
 * CogInferno - Inferno/Limbo-Style Distributed Tensor Computing
 * 
 * This header defines the distributed computing framework inspired by
 * Inferno OS and Limbo programming language concepts:
 * 
 * - Typed channels for tensor communication
 * - Styx protocol (9P variant) for resource access
 * - Dis virtual machine concepts for portable tensor operations
 * - CSP-style concurrency for parallel tensor processing
 * 
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    CogInferno Runtime                            │
 * │  ┌─────────────────────────────────────────────────────────────┐│
 * │  │                 Tensor Channel System                        ││
 * │  │  chan<Tensor>  chan<Atom>  chan<Message>                    ││
 * │  └─────────────────────────────────────────────────────────────┘│
 * │  ┌─────────────────────────────────────────────────────────────┐│
 * │  │                 Styx File System                             ││
 * │  │  /tensor/   /atom/   /cog/   /net/                          ││
 * │  └─────────────────────────────────────────────────────────────┘│
 * │  ┌─────────────────────────────────────────────────────────────┐│
 * │  │                 Distributed Scheduler                        ││
 * │  │  Worker Pool  Load Balancer  Task Queue                     ││
 * │  └─────────────────────────────────────────────────────────────┘│
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#ifndef COG_INFERNO_H
#define COG_INFERNO_H

#include "cogint.h"
#include "cog9p.h"
#include "cog_atomspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Channel Types (Limbo-style typed channels)
 *============================================================================*/

/* Channel element types */
typedef enum {
    COG_CHAN_TENSOR = 0,         /* CogTensor* */
    COG_CHAN_ATOM = 1,           /* CogAtom* */
    COG_CHAN_MESSAGE = 2,        /* CogMessage* */
    COG_CHAN_BYTES = 3,          /* Raw bytes */
    COG_CHAN_INT = 4,            /* int64_t */
    COG_CHAN_FLOAT = 5,          /* double */
    COG_CHAN_STRING = 6,         /* char* */
    COG_CHAN_TUPLE = 7,          /* CogTuple* */
} CogChanType;

/* Channel direction */
typedef enum {
    COG_CHAN_BIDIRECTIONAL = 0,
    COG_CHAN_SEND_ONLY = 1,
    COG_CHAN_RECV_ONLY = 2,
} CogChanDirection;

/* Extended channel structure */
typedef struct CogChan {
    char *name;
    CogChanType type;
    CogChanDirection direction;
    size_t elem_size;
    size_t capacity;             /* 0 = synchronous */
    
    /* Ring buffer */
    void *buffer;
    size_t head;
    size_t tail;
    size_t count;
    
    /* Synchronization */
    void *mutex;
    void *cond_not_empty;
    void *cond_not_full;
    
    /* Tensor-specific */
    CogDType tensor_dtype;
    int64_t *tensor_shape;
    int tensor_ndim;
    
    /* Network distribution */
    int distributed;
    Cog9PConn *remote_conn;
    char *remote_path;
    
    /* Statistics */
    uint64_t send_count;
    uint64_t recv_count;
    uint64_t block_count;
} CogChan;

/* Tuple for multi-value channel communication */
typedef struct {
    void **values;
    CogChanType *types;
    size_t n_values;
} CogTuple;

/*============================================================================
 * Alt Statement (Limbo-style select)
 *============================================================================*/

/* Alt case types */
typedef enum {
    COG_ALT_SEND = 0,
    COG_ALT_RECV = 1,
    COG_ALT_DEFAULT = 2,
    COG_ALT_TIMEOUT = 3,
} CogAltType;

/* Alt case structure */
typedef struct {
    CogAltType type;
    CogChan *chan;
    void *value;                 /* Value to send or receive */
    int ready;                   /* Set when case is ready */
    int selected;                /* Set when case is selected */
} CogAltCase;

/* Alt statement structure */
typedef struct {
    CogAltCase *cases;
    size_t n_cases;
    int has_default;
    int timeout_ms;              /* -1 = no timeout */
} CogAlt;

/*============================================================================
 * Distributed Worker System
 *============================================================================*/

/* Worker state */
typedef enum {
    COG_WORKER_IDLE = 0,
    COG_WORKER_BUSY = 1,
    COG_WORKER_STOPPED = 2,
    COG_WORKER_ERROR = 3,
} CogWorkerState;

/* Worker structure */
typedef struct CogWorker {
    int id;
    CogWorkerState state;
    
    /* Communication */
    CogChan *task_chan;          /* Receive tasks */
    CogChan *result_chan;        /* Send results */
    
    /* Resources */
    CogContext *ctx;
    CogAtomSpace *atomspace;
    
    /* Network */
    char *addr;
    uint16_t port;
    Cog9PConn *conn;
    
    /* Statistics */
    uint64_t tasks_completed;
    double total_time;
    
    /* Threading */
    void *thread;
    int running;
} CogWorker;

/* Worker pool */
typedef struct {
    CogWorker **workers;
    size_t n_workers;
    size_t capacity;
    
    /* Task distribution */
    CogChan *task_queue;
    CogChan *result_queue;
    
    /* Load balancing */
    int round_robin_idx;
    int use_load_balancing;
} CogWorkerPool;

/*============================================================================
 * Task System
 *============================================================================*/

/* Task types */
typedef enum {
    COG_TASK_TENSOR_OP = 0,      /* Tensor operation */
    COG_TASK_ATOM_OP = 1,        /* AtomSpace operation */
    COG_TASK_PLN_INFER = 2,      /* PLN inference */
    COG_TASK_ECAN_SPREAD = 3,    /* ECAN attention spread */
    COG_TASK_PATTERN_MINE = 4,   /* Pattern mining */
    COG_TASK_GNN_FORWARD = 5,    /* GNN forward pass */
    COG_TASK_CUSTOM = 6,         /* Custom function */
} CogTaskType;

/* Task structure */
typedef struct CogTask {
    uint64_t id;
    CogTaskType type;
    
    /* Input data */
    CogTensor **input_tensors;
    size_t n_input_tensors;
    CogAtom **input_atoms;
    size_t n_input_atoms;
    void *params;
    size_t params_size;
    
    /* Output */
    CogTensor *output_tensor;
    CogAtom **output_atoms;
    size_t n_output_atoms;
    int status;
    char *error;
    
    /* Execution */
    int (*execute)(struct CogTask *task, CogContext *ctx);
    void (*cleanup)(struct CogTask *task);
    
    /* Dependencies */
    struct CogTask **deps;
    size_t n_deps;
    int deps_completed;
    
    /* Timing */
    double start_time;
    double end_time;
} CogTask;

/* Task graph for complex workflows */
typedef struct {
    CogTask **tasks;
    size_t n_tasks;
    size_t capacity;
    
    /* Execution state */
    size_t completed;
    size_t failed;
    int running;
} CogTaskGraph;

/*============================================================================
 * Namespace Management (Plan 9 style)
 *============================================================================*/

/* Namespace entry */
typedef struct CogNsEntry {
    char *path;
    char *target;
    int flags;                   /* MREPL, MBEFORE, MAFTER */
    
    /* Mounted resource */
    Cog9PConn *conn;
    CogAtomSpace *atomspace;
    CogChan *chan;
    
    struct CogNsEntry *next;
} CogNsEntry;

/* Namespace structure */
typedef struct {
    CogNsEntry *mounts;
    char *root;
    
    /* Union directories */
    CogNsEntry **unions;
    size_t n_unions;
} CogNamespace;

/* Mount flags */
#define COG_MREPL    0x0000      /* Replace */
#define COG_MBEFORE  0x0001      /* Mount before */
#define COG_MAFTER   0x0002      /* Mount after */
#define COG_MCREATE  0x0004      /* Create if not exists */

/*============================================================================
 * Distributed Runtime
 *============================================================================*/

/* Runtime configuration */
typedef struct {
    /* Local settings */
    int n_local_workers;
    size_t task_queue_size;
    
    /* Network settings */
    char *listen_addr;
    uint16_t listen_port;
    
    /* Remote workers */
    char **remote_addrs;
    uint16_t *remote_ports;
    size_t n_remotes;
    
    /* AtomSpace settings */
    int shared_atomspace;
    char *atomspace_path;
    
    /* Channel settings */
    size_t default_chan_capacity;
} CogRuntimeConfig;

/* Distributed runtime */
typedef struct CogRuntime {
    CogContext *ctx;
    CogAtomSpace *atomspace;
    CogNamespace *ns;
    
    /* Workers */
    CogWorkerPool *local_pool;
    CogWorkerPool *remote_pool;
    
    /* Channels */
    CogChan **channels;
    size_t n_channels;
    
    /* 9P server */
    Cog9PServer *server;
    
    /* Task management */
    CogTaskGraph *current_graph;
    uint64_t next_task_id;
    
    /* State */
    int running;
    void *main_thread;
} CogRuntime;

/*============================================================================
 * Channel API
 *============================================================================*/

/**
 * Create a typed channel
 */
COGINT_API CogChan* cog_chan_create(const char *name, CogChanType type, 
                                     size_t capacity);

/**
 * Create a tensor channel with shape constraints
 */
COGINT_API CogChan* cog_chan_tensor(const char *name, CogDType dtype,
                                     int64_t *shape, int ndim, size_t capacity);

/**
 * Create a distributed channel (connected to remote)
 */
COGINT_API CogChan* cog_chan_remote(const char *name, CogChanType type,
                                     const char *addr, uint16_t port,
                                     const char *path);

/**
 * Send value to channel (blocks if full)
 */
COGINT_API int cog_chan_send(CogChan *ch, void *value);

/**
 * Send with timeout (returns -1 on timeout)
 */
COGINT_API int cog_chan_send_timeout(CogChan *ch, void *value, int timeout_ms);

/**
 * Receive value from channel (blocks if empty)
 */
COGINT_API int cog_chan_recv(CogChan *ch, void *value);

/**
 * Receive with timeout
 */
COGINT_API int cog_chan_recv_timeout(CogChan *ch, void *value, int timeout_ms);

/**
 * Try send (non-blocking)
 */
COGINT_API int cog_chan_try_send(CogChan *ch, void *value);

/**
 * Try receive (non-blocking)
 */
COGINT_API int cog_chan_try_recv(CogChan *ch, void *value);

/**
 * Close channel
 */
COGINT_API void cog_chan_close(CogChan *ch);

/**
 * Free channel
 */
COGINT_API void cog_chan_free(CogChan *ch);

/*============================================================================
 * Alt Statement API (Limbo-style select)
 *============================================================================*/

/**
 * Create alt statement
 */
COGINT_API CogAlt* cog_alt_create(void);

/**
 * Add send case to alt
 */
COGINT_API int cog_alt_add_send(CogAlt *alt, CogChan *ch, void *value);

/**
 * Add receive case to alt
 */
COGINT_API int cog_alt_add_recv(CogAlt *alt, CogChan *ch, void *value);

/**
 * Add default case
 */
COGINT_API int cog_alt_add_default(CogAlt *alt);

/**
 * Set timeout
 */
COGINT_API int cog_alt_set_timeout(CogAlt *alt, int timeout_ms);

/**
 * Execute alt statement (returns index of selected case, -1 on error)
 */
COGINT_API int cog_alt_select(CogAlt *alt);

/**
 * Free alt statement
 */
COGINT_API void cog_alt_free(CogAlt *alt);

/*============================================================================
 * Worker API
 *============================================================================*/

/**
 * Create worker
 */
COGINT_API CogWorker* cog_worker_create(int id, CogContext *ctx);

/**
 * Create remote worker connection
 */
COGINT_API CogWorker* cog_worker_remote(int id, const char *addr, uint16_t port);

/**
 * Start worker
 */
COGINT_API int cog_worker_start(CogWorker *worker);

/**
 * Stop worker
 */
COGINT_API int cog_worker_stop(CogWorker *worker);

/**
 * Free worker
 */
COGINT_API void cog_worker_free(CogWorker *worker);

/**
 * Create worker pool
 */
COGINT_API CogWorkerPool* cog_pool_create(int n_workers, CogContext *ctx);

/**
 * Add remote workers to pool
 */
COGINT_API int cog_pool_add_remote(CogWorkerPool *pool, const char *addr, 
                                    uint16_t port);

/**
 * Submit task to pool
 */
COGINT_API int cog_pool_submit(CogWorkerPool *pool, CogTask *task);

/**
 * Wait for all tasks to complete
 */
COGINT_API int cog_pool_wait(CogWorkerPool *pool);

/**
 * Free worker pool
 */
COGINT_API void cog_pool_free(CogWorkerPool *pool);

/*============================================================================
 * Task API
 *============================================================================*/

/**
 * Create tensor operation task
 */
COGINT_API CogTask* cog_task_tensor_op(uint8_t op, CogTensor **inputs, 
                                        size_t n_inputs);

/**
 * Create PLN inference task
 */
COGINT_API CogTask* cog_task_pln(CogAtomSpace *as, CogAtom *query,
                                  CogPLNConfig *config);

/**
 * Create ECAN spread task
 */
COGINT_API CogTask* cog_task_ecan(CogAtomSpace *as, CogAtom *source, int steps);

/**
 * Create GNN forward task
 */
COGINT_API CogTask* cog_task_gnn(CogAtomSpace *as, CogTensor **weights,
                                  int n_layers);

/**
 * Create custom task
 */
COGINT_API CogTask* cog_task_custom(int (*execute)(CogTask*, CogContext*),
                                     void *params, size_t params_size);

/**
 * Add dependency to task
 */
COGINT_API int cog_task_add_dep(CogTask *task, CogTask *dep);

/**
 * Execute task
 */
COGINT_API int cog_task_execute(CogTask *task, CogContext *ctx);

/**
 * Free task
 */
COGINT_API void cog_task_free(CogTask *task);

/**
 * Create task graph
 */
COGINT_API CogTaskGraph* cog_graph_create(void);

/**
 * Add task to graph
 */
COGINT_API int cog_graph_add(CogTaskGraph *graph, CogTask *task);

/**
 * Execute task graph
 */
COGINT_API int cog_graph_execute(CogTaskGraph *graph, CogWorkerPool *pool);

/**
 * Free task graph
 */
COGINT_API void cog_graph_free(CogTaskGraph *graph);

/*============================================================================
 * Namespace API
 *============================================================================*/

/**
 * Create namespace
 */
COGINT_API CogNamespace* cog_ns_create(void);

/**
 * Bind path to target
 */
COGINT_API int cog_ns_bind(CogNamespace *ns, const char *path, 
                            const char *target, int flags);

/**
 * Mount 9P connection
 */
COGINT_API int cog_ns_mount(CogNamespace *ns, const char *path,
                             Cog9PConn *conn, const char *aname, int flags);

/**
 * Mount AtomSpace
 */
COGINT_API int cog_ns_mount_atomspace(CogNamespace *ns, const char *path,
                                       CogAtomSpace *as, int flags);

/**
 * Mount channel
 */
COGINT_API int cog_ns_mount_chan(CogNamespace *ns, const char *path,
                                  CogChan *ch, int flags);

/**
 * Unmount path
 */
COGINT_API int cog_ns_unmount(CogNamespace *ns, const char *path);

/**
 * Resolve path to resource
 */
COGINT_API void* cog_ns_resolve(CogNamespace *ns, const char *path, int *type);

/**
 * Free namespace
 */
COGINT_API void cog_ns_free(CogNamespace *ns);

/*============================================================================
 * Runtime API
 *============================================================================*/

/**
 * Create distributed runtime
 */
COGINT_API CogRuntime* cog_runtime_create(CogRuntimeConfig *config);

/**
 * Start runtime
 */
COGINT_API int cog_runtime_start(CogRuntime *rt);

/**
 * Stop runtime
 */
COGINT_API int cog_runtime_stop(CogRuntime *rt);

/**
 * Submit task to runtime
 */
COGINT_API int cog_runtime_submit(CogRuntime *rt, CogTask *task);

/**
 * Create channel in runtime
 */
COGINT_API CogChan* cog_runtime_chan(CogRuntime *rt, const char *name,
                                      CogChanType type, size_t capacity);

/**
 * Get runtime AtomSpace
 */
COGINT_API CogAtomSpace* cog_runtime_atomspace(CogRuntime *rt);

/**
 * Get runtime namespace
 */
COGINT_API CogNamespace* cog_runtime_ns(CogRuntime *rt);

/**
 * Free runtime
 */
COGINT_API void cog_runtime_free(CogRuntime *rt);

/*============================================================================
 * Parallel Tensor Operations
 *============================================================================*/

/**
 * Parallel map operation on tensor
 */
COGINT_API CogTensor* cog_parallel_map(CogRuntime *rt, CogTensor *input,
                                        CogTensor* (*fn)(CogTensor*, void*),
                                        void *ctx, int n_chunks);

/**
 * Parallel reduce operation
 */
COGINT_API CogTensor* cog_parallel_reduce(CogRuntime *rt, CogTensor **inputs,
                                           size_t n_inputs,
                                           CogTensor* (*fn)(CogTensor*, CogTensor*));

/**
 * Distributed matrix multiplication
 */
COGINT_API CogTensor* cog_distributed_matmul(CogRuntime *rt, CogTensor *a,
                                              CogTensor *b);

/**
 * Distributed attention computation
 */
COGINT_API CogTensor* cog_distributed_attention(CogRuntime *rt,
                                                 CogTensor *query,
                                                 CogTensor *key,
                                                 CogTensor *value);

/**
 * Distributed GNN forward pass
 */
COGINT_API CogTensor* cog_distributed_gnn(CogRuntime *rt, CogAtomSpace *as,
                                           CogTensor **weights, int n_layers);

#ifdef __cplusplus
}
#endif

#endif /* COG_INFERNO_H */
