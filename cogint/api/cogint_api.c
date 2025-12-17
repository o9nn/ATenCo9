/**
 * CogInt Unified API - Cognitive Processing Pipeline
 * 
 * This file implements the unified API layer that integrates:
 * - ATen tensor operations
 * - 9P distributed computing
 * - OpenCog cognitive services (PLN, ECAN, URE)
 * - Neural-symbolic processing pipeline
 * 
 * The cognitive pipeline enables:
 * - Perception → Reasoning → Action cycles
 * - Attention-driven processing
 * - Symbolic-neural integration
 * - Distributed cognitive computation
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
#include "../include/cog9p.h"
#include "../include/cog_atomspace.h"
#include "../include/cog_inferno.h"

/*============================================================================
 * Global State
 *============================================================================*/

static CogError g_last_error = COG_OK;
static pthread_mutex_t g_error_lock = PTHREAD_MUTEX_INITIALIZER;

/*============================================================================
 * Error Handling
 *============================================================================*/

COGINT_API const char* cog_error_string(CogError err) {
    switch (err) {
        case COG_OK: return "Success";
        case COG_ERR_NOMEM: return "Out of memory";
        case COG_ERR_INVALID: return "Invalid argument";
        case COG_ERR_NOTFOUND: return "Not found";
        case COG_ERR_NETWORK: return "Network error";
        case COG_ERR_PROTOCOL: return "Protocol error";
        case COG_ERR_ATOMSPACE: return "AtomSpace error";
        case COG_ERR_TENSOR: return "Tensor error";
        case COG_ERR_DEVICE: return "Device error";
        default: return "Unknown error";
    }
}

COGINT_API CogError cog_last_error(void) {
    pthread_mutex_lock(&g_error_lock);
    CogError err = g_last_error;
    pthread_mutex_unlock(&g_error_lock);
    return err;
}

static void set_error(CogError err) {
    pthread_mutex_lock(&g_error_lock);
    g_last_error = err;
    pthread_mutex_unlock(&g_error_lock);
}

/*============================================================================
 * Context Management
 *============================================================================*/

COGINT_API CogContext* cogint_init(void) {
    CogContext *ctx = calloc(1, sizeof(CogContext));
    if (!ctx) {
        set_error(COG_ERR_NOMEM);
        return NULL;
    }
    
    ctx->namespace_root = strdup("/cog");
    ctx->use_cuda = 0;  /* Detect CUDA availability */
    ctx->distributed_mode = 0;
    
    return ctx;
}

COGINT_API void cogint_shutdown(CogContext *ctx) {
    if (!ctx) return;
    
    /* Free connections */
    for (size_t i = 0; i < ctx->num_connections; i++) {
        cog_9p_disconnect(ctx->connections[i]);
    }
    free(ctx->connections);
    
    /* Free channels */
    for (size_t i = 0; i < ctx->num_channels; i++) {
        cog_chan_free((CogChan*)ctx->channels[i]);
    }
    free(ctx->channels);
    
    /* Free AtomSpace */
    if (ctx->default_atomspace) {
        cog_atomspace_free(ctx->default_atomspace);
    }
    
    free(ctx->namespace_root);
    free(ctx);
}

COGINT_API const char* cogint_version(void) {
    return COGINT_VERSION_STRING;
}

/*============================================================================
 * Tensor Operations
 *============================================================================*/

/* Get element size for dtype */
static size_t dtype_size(CogDType dtype) {
    switch (dtype) {
        case COG_DTYPE_FLOAT32: return 4;
        case COG_DTYPE_FLOAT64: return 8;
        case COG_DTYPE_FLOAT16: return 2;
        case COG_DTYPE_BFLOAT16: return 2;
        case COG_DTYPE_INT8: return 1;
        case COG_DTYPE_INT16: return 2;
        case COG_DTYPE_INT32: return 4;
        case COG_DTYPE_INT64: return 8;
        case COG_DTYPE_UINT8: return 1;
        case COG_DTYPE_BOOL: return 1;
        case COG_DTYPE_COMPLEX64: return 8;
        case COG_DTYPE_COMPLEX128: return 16;
        default: return 4;
    }
}

COGINT_API CogTensor* cog_tensor_create(CogContext *ctx, int64_t *shape, 
                                         int ndim, CogDType dtype) {
    if (!shape || ndim <= 0 || ndim > 8) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    CogTensor *t = calloc(1, sizeof(CogTensor));
    if (!t) {
        set_error(COG_ERR_NOMEM);
        return NULL;
    }
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->device = COG_DEVICE_CPU;
    t->refcount = 1;
    
    /* Copy shape */
    t->shape = malloc(ndim * sizeof(int64_t));
    memcpy(t->shape, shape, ndim * sizeof(int64_t));
    
    /* Calculate strides (row-major) */
    t->strides = malloc(ndim * sizeof(int64_t));
    t->numel = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = t->numel;
        t->numel *= shape[i];
    }
    
    /* Allocate data */
    size_t elem_size = dtype_size(dtype);
    t->data = calloc(t->numel, elem_size);
    if (!t->data) {
        free(t->shape);
        free(t->strides);
        free(t);
        set_error(COG_ERR_NOMEM);
        return NULL;
    }
    
    return t;
}

COGINT_API CogTensor* cog_tensor_from_aten(void *aten_tensor) {
    /* Bridge to ATen tensor - implementation depends on ATen integration */
    /* For now, return NULL as ATen integration requires C++ */
    set_error(COG_ERR_INVALID);
    return NULL;
}

COGINT_API void* cog_tensor_to_aten(CogTensor *tensor) {
    /* Bridge to ATen tensor - implementation depends on ATen integration */
    set_error(COG_ERR_INVALID);
    return NULL;
}

COGINT_API void cog_tensor_free(CogTensor *tensor) {
    if (!tensor) return;
    
    tensor->refcount--;
    if (tensor->refcount > 0) return;
    
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor->path);
    
    if (tensor->conn) {
        /* Don't free connection, just clear reference */
        tensor->conn = NULL;
    }
    
    free(tensor);
}

COGINT_API CogTensor* cog_tensor_clone(CogTensor *tensor) {
    if (!tensor) return NULL;
    
    CogTensor *clone = cog_tensor_create(NULL, tensor->shape, tensor->ndim, 
                                          tensor->dtype);
    if (!clone) return NULL;
    
    size_t elem_size = dtype_size(tensor->dtype);
    memcpy(clone->data, tensor->data, tensor->numel * elem_size);
    
    return clone;
}

COGINT_API int cog_tensor_copy(CogTensor *dst, CogTensor *src) {
    if (!dst || !src) return COG_ERR_INVALID;
    if (dst->numel != src->numel) return COG_ERR_INVALID;
    if (dst->dtype != src->dtype) return COG_ERR_INVALID;
    
    size_t elem_size = dtype_size(src->dtype);
    memcpy(dst->data, src->data, src->numel * elem_size);
    
    return COG_OK;
}

/*============================================================================
 * Tensor Arithmetic
 *============================================================================*/

COGINT_API CogTensor* cog_tensor_add(CogTensor *a, CogTensor *b) {
    if (!a || !b) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    /* Check shapes match (simple case - no broadcasting) */
    if (a->numel != b->numel) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    CogTensor *result = cog_tensor_clone(a);
    if (!result) return NULL;
    
    /* Element-wise addition */
    float *ra = (float*)result->data;
    float *rb = (float*)b->data;
    
    for (size_t i = 0; i < result->numel; i++) {
        ra[i] += rb[i];
    }
    
    return result;
}

COGINT_API CogTensor* cog_tensor_mul(CogTensor *a, CogTensor *b) {
    if (!a || !b) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    if (a->numel != b->numel) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    CogTensor *result = cog_tensor_clone(a);
    if (!result) return NULL;
    
    float *ra = (float*)result->data;
    float *rb = (float*)b->data;
    
    for (size_t i = 0; i < result->numel; i++) {
        ra[i] *= rb[i];
    }
    
    return result;
}

COGINT_API CogTensor* cog_tensor_matmul(CogTensor *a, CogTensor *b) {
    if (!a || !b) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    /* Check dimensions for matrix multiplication */
    if (a->ndim < 2 || b->ndim < 2) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    int64_t m = a->shape[a->ndim - 2];
    int64_t k = a->shape[a->ndim - 1];
    int64_t n = b->shape[b->ndim - 1];
    
    if (k != b->shape[b->ndim - 2]) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    /* Create result tensor */
    int64_t result_shape[2] = {m, n};
    CogTensor *result = cog_tensor_create(NULL, result_shape, 2, a->dtype);
    if (!result) return NULL;
    
    float *ra = (float*)a->data;
    float *rb = (float*)b->data;
    float *rc = (float*)result->data;
    
    /* Simple matrix multiplication */
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; l++) {
                sum += ra[i * k + l] * rb[l * n + j];
            }
            rc[i * n + j] = sum;
        }
    }
    
    return result;
}

COGINT_API CogTensor* cog_tensor_softmax(CogTensor *t, int dim) {
    if (!t) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    CogTensor *result = cog_tensor_clone(t);
    if (!result) return NULL;
    
    float *data = (float*)result->data;
    
    /* Simple softmax over entire tensor */
    /* Find max for numerical stability */
    float max_val = data[0];
    for (size_t i = 1; i < result->numel; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    /* Compute exp and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < result->numel; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    
    /* Normalize */
    for (size_t i = 0; i < result->numel; i++) {
        data[i] /= sum;
    }
    
    return result;
}

/*============================================================================
 * 9P Distribution Functions
 *============================================================================*/

COGINT_API Cog9PConn* cog_9p_connect(CogContext *ctx, const char *addr, 
                                      uint16_t port) {
    Cog9PConn *conn = cog9p_dial(addr, port);
    if (!conn) {
        set_error(COG_ERR_NETWORK);
        return NULL;
    }
    
    if (cog9p_version(conn) < 0) {
        cog_9p_disconnect(conn);
        set_error(COG_ERR_PROTOCOL);
        return NULL;
    }
    
    if (cog9p_attach(conn, NULL, NULL) < 0) {
        cog_9p_disconnect(conn);
        set_error(COG_ERR_PROTOCOL);
        return NULL;
    }
    
    /* Add to context */
    if (ctx) {
        ctx->connections = realloc(ctx->connections, 
                                   (ctx->num_connections + 1) * sizeof(Cog9PConn*));
        ctx->connections[ctx->num_connections++] = conn;
    }
    
    return conn;
}

COGINT_API int cog_tensor_export(CogTensor *tensor, Cog9PConn *conn, 
                                  const char *path) {
    if (!tensor || !conn || !path) return COG_ERR_INVALID;
    
    /* Create tensor on remote */
    uint32_t fid;
    int rc = cog9p_tensor_create(conn, path, tensor->dtype, tensor->shape,
                                  tensor->ndim, &fid);
    if (rc < 0) return COG_ERR_NETWORK;
    
    /* Write tensor data */
    rc = cog9p_tensor_write(conn, fid, tensor);
    if (rc < 0) return COG_ERR_NETWORK;
    
    /* Store remote info in tensor */
    tensor->conn = conn;
    tensor->path = strdup(path);
    tensor->fid = fid;
    tensor->device = COG_DEVICE_DISTRIBUTED;
    
    return COG_OK;
}

COGINT_API CogTensor* cog_tensor_import(CogContext *ctx, Cog9PConn *conn,
                                         const char *path) {
    if (!conn || !path) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    /* Walk to tensor path */
    uint32_t fid = conn->next_fid++;
    char *names[16];
    int nnames = 0;
    
    /* Parse path */
    char *pathcopy = strdup(path);
    char *token = strtok(pathcopy, "/");
    while (token && nnames < 16) {
        names[nnames++] = token;
        token = strtok(NULL, "/");
    }
    
    Cog9PQid qids[16];
    if (cog9p_walk(conn, 0, fid, names, nnames, qids) < 0) {
        free(pathcopy);
        set_error(COG_ERR_NOTFOUND);
        return NULL;
    }
    free(pathcopy);
    
    /* Read tensor */
    CogTensor *tensor = NULL;
    if (cog9p_tensor_read(conn, fid, &tensor) < 0) {
        set_error(COG_ERR_NETWORK);
        return NULL;
    }
    
    return tensor;
}

COGINT_API int cog_atomspace_export(CogAtomSpace *as, Cog9PConn *conn,
                                     const char *path) {
    if (!as || !conn || !path) return COG_ERR_INVALID;
    
    as->export_path = strdup(path);
    as->server = conn;
    
    return COG_OK;
}

/*============================================================================
 * Channel Operations (Inferno-style)
 *============================================================================*/

COGINT_API CogChannel* cog_channel_create(CogContext *ctx, const char *name,
                                           size_t capacity) {
    CogChan *ch = cog_chan_create(name, COG_CHAN_TENSOR, capacity);
    if (!ch) {
        set_error(COG_ERR_NOMEM);
        return NULL;
    }
    
    /* Add to context */
    if (ctx) {
        ctx->channels = realloc(ctx->channels,
                                (ctx->num_channels + 1) * sizeof(CogChannel*));
        ctx->channels[ctx->num_channels++] = (CogChannel*)ch;
    }
    
    return (CogChannel*)ch;
}

COGINT_API void cog_channel_free(CogChannel *ch) {
    cog_chan_free((CogChan*)ch);
}

COGINT_API int cog_channel_send(CogChannel *ch, CogTensor *tensor) {
    return cog_chan_send((CogChan*)ch, &tensor);
}

COGINT_API CogTensor* cog_channel_recv(CogChannel *ch) {
    CogTensor *tensor = NULL;
    if (cog_chan_recv((CogChan*)ch, &tensor) < 0) {
        return NULL;
    }
    return tensor;
}

COGINT_API int cog_channel_select(CogChannel **channels, int n, int *ready) {
    if (!channels || n <= 0 || !ready) return COG_ERR_INVALID;
    
    CogAlt *alt = cog_alt_create();
    
    CogTensor *values[n];
    for (int i = 0; i < n; i++) {
        values[i] = NULL;
        cog_alt_add_recv(alt, (CogChan*)channels[i], &values[i]);
    }
    
    int selected = cog_alt_select(alt);
    
    if (selected >= 0) {
        *ready = selected;
    }
    
    cog_alt_free(alt);
    
    return (selected >= 0) ? COG_OK : COG_ERR_INVALID;
}

/*============================================================================
 * Cognitive Operations
 *============================================================================*/

COGINT_API int cog_pln_infer(CogAtomSpace *as, CogAtom *query, 
                              CogAtom **results, size_t *n) {
    if (!as || !query || !results || !n) return COG_ERR_INVALID;
    
    CogPLNConfig config = {
        .max_steps = 100,
        .confidence_threshold = 0.5,
        .max_results = 100,
        .use_tensor_similarity = 1,
        .tensor_similarity_threshold = 0.7,
    };
    
    CogAtom **found = NULL;
    size_t found_count = 0;
    
    int rc = cog_pln_inference(as, query, &config, &found, &found_count);
    if (rc != COG_OK) return rc;
    
    *results = *found;
    *n = found_count;
    
    return COG_OK;
}

COGINT_API int cog_ecan_spread(CogAtomSpace *as, CogAtom *source, int steps) {
    if (!as || !source) return COG_ERR_INVALID;
    
    for (int i = 0; i < steps; i++) {
        int rc = cog_ecan_spread_attention(as, source, 10.0);
        if (rc != COG_OK) return rc;
    }
    
    return COG_OK;
}

COGINT_API CogTensor* cog_attention_weights(CogAtomSpace *as, CogAtom **atoms,
                                             size_t n) {
    if (!as || !atoms || n == 0) {
        set_error(COG_ERR_INVALID);
        return NULL;
    }
    
    int64_t shape[] = {(int64_t)n};
    CogTensor *weights = cog_tensor_create(NULL, shape, 1, COG_DTYPE_FLOAT32);
    if (!weights) return NULL;
    
    float *data = (float*)weights->data;
    
    /* Compute attention weights from STI values */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float sti = (atoms[i]->sti + 32768.0f) / 65536.0f;
        data[i] = expf(sti * 5.0f);  /* Scale and exponentiate */
        sum += data[i];
    }
    
    /* Normalize to sum to 1 */
    for (size_t i = 0; i < n; i++) {
        data[i] /= sum;
    }
    
    return weights;
}

/*============================================================================
 * Cognitive Processing Pipeline
 *============================================================================*/

/* Pipeline stage types */
typedef enum {
    COG_STAGE_PERCEIVE = 0,
    COG_STAGE_ATTEND = 1,
    COG_STAGE_REASON = 2,
    COG_STAGE_LEARN = 3,
    COG_STAGE_ACT = 4,
} CogPipelineStage;

/* Pipeline configuration */
typedef struct {
    int use_attention;
    int use_pln;
    int use_pattern_mining;
    int max_reasoning_steps;
    double attention_threshold;
} CogPipelineConfig;

/* Pipeline state */
typedef struct {
    CogAtomSpace *atomspace;
    CogRuntime *runtime;
    CogPipelineConfig config;
    
    /* Current state */
    CogPipelineStage stage;
    CogAtom **focus_atoms;
    size_t n_focus;
    CogTensor *current_embedding;
    
    /* Channels for pipeline stages */
    CogChan *perception_chan;
    CogChan *attention_chan;
    CogChan *reasoning_chan;
    CogChan *learning_chan;
    CogChan *action_chan;
} CogPipeline;

/**
 * Create cognitive processing pipeline
 */
CogPipeline* cog_pipeline_create(CogRuntime *rt, CogPipelineConfig *config) {
    CogPipeline *pipe = calloc(1, sizeof(CogPipeline));
    if (!pipe) return NULL;
    
    pipe->runtime = rt;
    pipe->atomspace = cog_runtime_atomspace(rt);
    
    if (config) {
        pipe->config = *config;
    } else {
        pipe->config.use_attention = 1;
        pipe->config.use_pln = 1;
        pipe->config.use_pattern_mining = 0;
        pipe->config.max_reasoning_steps = 10;
        pipe->config.attention_threshold = 0.5;
    }
    
    /* Create pipeline channels */
    pipe->perception_chan = cog_chan_create("perception", COG_CHAN_TENSOR, 16);
    pipe->attention_chan = cog_chan_create("attention", COG_CHAN_ATOM, 16);
    pipe->reasoning_chan = cog_chan_create("reasoning", COG_CHAN_ATOM, 16);
    pipe->learning_chan = cog_chan_create("learning", COG_CHAN_ATOM, 16);
    pipe->action_chan = cog_chan_create("action", COG_CHAN_TENSOR, 16);
    
    return pipe;
}

/**
 * Perception stage: Convert tensor input to atoms
 */
int cog_pipeline_perceive(CogPipeline *pipe, CogTensor *input) {
    if (!pipe || !input) return COG_ERR_INVALID;
    
    pipe->stage = COG_STAGE_PERCEIVE;
    
    /* Create TensorNode for input */
    CogAtom *input_atom = cog_tensor_node_create(pipe->atomspace, input, 
                                                  "perception_input");
    if (!input_atom) return COG_ERR_ATOMSPACE;
    
    /* Set high initial attention */
    cog_atom_set_av(input_atom, 1000, 100, 10);
    
    /* Store current embedding */
    if (pipe->current_embedding) {
        cog_tensor_free(pipe->current_embedding);
    }
    pipe->current_embedding = cog_tensor_clone(input);
    
    /* Send to attention stage */
    cog_chan_send(pipe->attention_chan, &input_atom);
    
    return COG_OK;
}

/**
 * Attention stage: Focus on relevant atoms
 */
int cog_pipeline_attend(CogPipeline *pipe) {
    if (!pipe) return COG_ERR_INVALID;
    
    pipe->stage = COG_STAGE_ATTEND;
    
    if (!pipe->config.use_attention) {
        /* Skip attention, pass through */
        return COG_OK;
    }
    
    /* Get atoms in attentional focus */
    CogAtom **focus = NULL;
    size_t n_focus = 0;
    
    int rc = cog_ecan_get_focus(pipe->atomspace, &focus, &n_focus);
    if (rc != COG_OK) return rc;
    
    /* Store focus atoms */
    free(pipe->focus_atoms);
    pipe->focus_atoms = focus;
    pipe->n_focus = n_focus;
    
    /* Spread attention from focus atoms */
    for (size_t i = 0; i < n_focus; i++) {
        cog_ecan_spread_attention(pipe->atomspace, focus[i], 5.0);
    }
    
    /* Send top focus atoms to reasoning */
    for (size_t i = 0; i < n_focus && i < 10; i++) {
        cog_chan_send(pipe->reasoning_chan, &focus[i]);
    }
    
    return COG_OK;
}

/**
 * Reasoning stage: Apply PLN inference
 */
int cog_pipeline_reason(CogPipeline *pipe, CogAtom *query) {
    if (!pipe) return COG_ERR_INVALID;
    
    pipe->stage = COG_STAGE_REASON;
    
    if (!pipe->config.use_pln) {
        return COG_OK;
    }
    
    /* Run PLN inference */
    CogAtom *results = NULL;
    size_t n_results = 0;
    
    int rc = cog_pln_infer(pipe->atomspace, query, &results, &n_results);
    if (rc != COG_OK) return rc;
    
    /* Send results to learning stage */
    for (size_t i = 0; i < n_results; i++) {
        cog_chan_send(pipe->learning_chan, &results[i]);
    }
    
    free(results);
    
    return COG_OK;
}

/**
 * Learning stage: Update knowledge and patterns
 */
int cog_pipeline_learn(CogPipeline *pipe) {
    if (!pipe) return COG_ERR_INVALID;
    
    pipe->stage = COG_STAGE_LEARN;
    
    if (pipe->config.use_pattern_mining) {
        /* Run pattern mining */
        CogPatternMinerConfig config = {
            .min_support = 2,
            .max_pattern_size = 5,
            .min_confidence = 0.5,
            .max_patterns = 100,
        };
        
        CogMinedPattern *patterns = NULL;
        size_t n_patterns = 0;
        
        cog_pattern_mine(pipe->atomspace, &config, &patterns, &n_patterns);
        
        /* Patterns are now in AtomSpace */
        free(patterns);
    }
    
    return COG_OK;
}

/**
 * Action stage: Generate output tensor
 */
CogTensor* cog_pipeline_act(CogPipeline *pipe) {
    if (!pipe) return NULL;
    
    pipe->stage = COG_STAGE_ACT;
    
    /* Generate output based on focus atoms */
    if (pipe->n_focus == 0) {
        return cog_tensor_clone(pipe->current_embedding);
    }
    
    /* Compute attention-weighted output */
    CogTensor *weights = cog_attention_weights(pipe->atomspace, 
                                                pipe->focus_atoms,
                                                pipe->n_focus);
    if (!weights) {
        return cog_tensor_clone(pipe->current_embedding);
    }
    
    /* Aggregate tensor representations */
    int64_t shape[] = {128};
    CogTensor *output = cog_tensor_create(NULL, shape, 1, COG_DTYPE_FLOAT32);
    float *out_data = (float*)output->data;
    float *w_data = (float*)weights->data;
    
    for (size_t i = 0; i < pipe->n_focus; i++) {
        CogAtom *atom = pipe->focus_atoms[i];
        CogTensor *emb = cog_atomspace_to_tensor(pipe->atomspace, atom, 1);
        if (emb) {
            float *emb_data = (float*)emb->data;
            for (int j = 0; j < 128; j++) {
                out_data[j] += w_data[i] * emb_data[j];
            }
            cog_tensor_free(emb);
        }
    }
    
    cog_tensor_free(weights);
    
    return output;
}

/**
 * Run full cognitive cycle
 */
CogTensor* cog_pipeline_cycle(CogPipeline *pipe, CogTensor *input) {
    if (!pipe || !input) return NULL;
    
    /* Perception */
    if (cog_pipeline_perceive(pipe, input) != COG_OK) {
        return NULL;
    }
    
    /* Attention */
    if (cog_pipeline_attend(pipe) != COG_OK) {
        return NULL;
    }
    
    /* Reasoning (on focus atoms) */
    if (pipe->n_focus > 0) {
        cog_pipeline_reason(pipe, pipe->focus_atoms[0]);
    }
    
    /* Learning */
    cog_pipeline_learn(pipe);
    
    /* Action */
    return cog_pipeline_act(pipe);
}

/**
 * Free pipeline
 */
void cog_pipeline_free(CogPipeline *pipe) {
    if (!pipe) return;
    
    cog_chan_free(pipe->perception_chan);
    cog_chan_free(pipe->attention_chan);
    cog_chan_free(pipe->reasoning_chan);
    cog_chan_free(pipe->learning_chan);
    cog_chan_free(pipe->action_chan);
    
    free(pipe->focus_atoms);
    if (pipe->current_embedding) {
        cog_tensor_free(pipe->current_embedding);
    }
    
    free(pipe);
}
