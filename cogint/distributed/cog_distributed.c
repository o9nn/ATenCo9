/**
 * CogDistributed - Distributed Tensor Computing Implementation
 * 
 * Implements Inferno/Limbo-style distributed computing with typed channels,
 * worker pools, and namespace management for distributed AGI computing.
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#include "../include/cogint.h"
#include "../include/cog_inferno.h"

/*============================================================================
 * Utility Functions
 *============================================================================*/

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*============================================================================
 * Channel Implementation
 *============================================================================*/

COGINT_API CogChan* cog_chan_create(const char *name, CogChanType type,
                                     size_t capacity) {
    CogChan *ch = calloc(1, sizeof(CogChan));
    if (!ch) return NULL;
    
    ch->name = name ? strdup(name) : NULL;
    ch->type = type;
    ch->capacity = capacity;
    
    /* Determine element size */
    switch (type) {
        case COG_CHAN_TENSOR:
        case COG_CHAN_ATOM:
        case COG_CHAN_MESSAGE:
        case COG_CHAN_STRING:
            ch->elem_size = sizeof(void*);
            break;
        case COG_CHAN_INT:
            ch->elem_size = sizeof(int64_t);
            break;
        case COG_CHAN_FLOAT:
            ch->elem_size = sizeof(double);
            break;
        case COG_CHAN_BYTES:
            ch->elem_size = sizeof(void*);  /* Pointer to buffer */
            break;
        case COG_CHAN_TUPLE:
            ch->elem_size = sizeof(CogTuple*);
            break;
        default:
            ch->elem_size = sizeof(void*);
    }
    
    /* Allocate buffer for buffered channels */
    if (capacity > 0) {
        ch->buffer = calloc(capacity, ch->elem_size);
    }
    
    /* Initialize synchronization */
    ch->mutex = malloc(sizeof(pthread_mutex_t));
    ch->cond_not_empty = malloc(sizeof(pthread_cond_t));
    ch->cond_not_full = malloc(sizeof(pthread_cond_t));
    
    pthread_mutex_init(ch->mutex, NULL);
    pthread_cond_init(ch->cond_not_empty, NULL);
    pthread_cond_init(ch->cond_not_full, NULL);
    
    return ch;
}

COGINT_API CogChan* cog_chan_tensor(const char *name, CogDType dtype,
                                     int64_t *shape, int ndim, size_t capacity) {
    CogChan *ch = cog_chan_create(name, COG_CHAN_TENSOR, capacity);
    if (!ch) return NULL;
    
    ch->tensor_dtype = dtype;
    ch->tensor_ndim = ndim;
    if (shape && ndim > 0) {
        ch->tensor_shape = malloc(ndim * sizeof(int64_t));
        memcpy(ch->tensor_shape, shape, ndim * sizeof(int64_t));
    }
    
    return ch;
}

COGINT_API CogChan* cog_chan_remote(const char *name, CogChanType type,
                                     const char *addr, uint16_t port,
                                     const char *path) {
    CogChan *ch = cog_chan_create(name, type, 0);  /* Synchronous for remote */
    if (!ch) return NULL;
    
    ch->distributed = 1;
    ch->remote_path = path ? strdup(path) : NULL;
    
    /* Connect to remote */
    ch->remote_conn = cog9p_dial(addr, port);
    if (!ch->remote_conn) {
        cog_chan_free(ch);
        return NULL;
    }
    
    /* Negotiate protocol */
    if (cog9p_version(ch->remote_conn) < 0 ||
        cog9p_attach(ch->remote_conn, NULL, NULL) < 0) {
        cog_chan_free(ch);
        return NULL;
    }
    
    return ch;
}

COGINT_API int cog_chan_send(CogChan *ch, void *value) {
    if (!ch) return COG_ERR_INVALID;
    
    pthread_mutex_lock(ch->mutex);
    
    /* Handle distributed channel */
    if (ch->distributed && ch->remote_conn) {
        pthread_mutex_unlock(ch->mutex);
        /* Serialize and send via 9P */
        /* TODO: Implement 9P tensor write */
        return COG_OK;
    }
    
    /* Wait if buffer is full */
    while (ch->capacity > 0 && ch->count >= ch->capacity) {
        pthread_cond_wait(ch->cond_not_full, ch->mutex);
    }
    
    if (ch->capacity > 0) {
        /* Buffered send */
        char *buf = (char*)ch->buffer;
        memcpy(buf + ch->tail * ch->elem_size, value, ch->elem_size);
        ch->tail = (ch->tail + 1) % ch->capacity;
        ch->count++;
    } else {
        /* Synchronous send - wait for receiver */
        /* For simplicity, use a temporary storage */
        ch->buffer = value;
        ch->count = 1;
        pthread_cond_signal(ch->cond_not_empty);
        while (ch->count > 0) {
            pthread_cond_wait(ch->cond_not_full, ch->mutex);
        }
    }
    
    ch->send_count++;
    pthread_cond_signal(ch->cond_not_empty);
    pthread_mutex_unlock(ch->mutex);
    
    return COG_OK;
}

COGINT_API int cog_chan_send_timeout(CogChan *ch, void *value, int timeout_ms) {
    if (!ch) return COG_ERR_INVALID;
    if (timeout_ms < 0) return cog_chan_send(ch, value);
    
    pthread_mutex_lock(ch->mutex);
    
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000;
    }
    
    while (ch->capacity > 0 && ch->count >= ch->capacity) {
        int rc = pthread_cond_timedwait(ch->cond_not_full, ch->mutex, &ts);
        if (rc != 0) {
            pthread_mutex_unlock(ch->mutex);
            return -1;  /* Timeout */
        }
    }
    
    if (ch->capacity > 0) {
        char *buf = (char*)ch->buffer;
        memcpy(buf + ch->tail * ch->elem_size, value, ch->elem_size);
        ch->tail = (ch->tail + 1) % ch->capacity;
        ch->count++;
    }
    
    ch->send_count++;
    pthread_cond_signal(ch->cond_not_empty);
    pthread_mutex_unlock(ch->mutex);
    
    return COG_OK;
}

COGINT_API int cog_chan_recv(CogChan *ch, void *value) {
    if (!ch || !value) return COG_ERR_INVALID;
    
    pthread_mutex_lock(ch->mutex);
    
    /* Handle distributed channel */
    if (ch->distributed && ch->remote_conn) {
        pthread_mutex_unlock(ch->mutex);
        /* Read via 9P */
        /* TODO: Implement 9P tensor read */
        return COG_OK;
    }
    
    /* Wait if buffer is empty */
    while (ch->count == 0) {
        pthread_cond_wait(ch->cond_not_empty, ch->mutex);
    }
    
    if (ch->capacity > 0) {
        /* Buffered receive */
        char *buf = (char*)ch->buffer;
        memcpy(value, buf + ch->head * ch->elem_size, ch->elem_size);
        ch->head = (ch->head + 1) % ch->capacity;
        ch->count--;
    } else {
        /* Synchronous receive */
        memcpy(value, &ch->buffer, ch->elem_size);
        ch->count = 0;
        pthread_cond_signal(ch->cond_not_full);
    }
    
    ch->recv_count++;
    pthread_cond_signal(ch->cond_not_full);
    pthread_mutex_unlock(ch->mutex);
    
    return COG_OK;
}

COGINT_API int cog_chan_recv_timeout(CogChan *ch, void *value, int timeout_ms) {
    if (!ch || !value) return COG_ERR_INVALID;
    if (timeout_ms < 0) return cog_chan_recv(ch, value);
    
    pthread_mutex_lock(ch->mutex);
    
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000;
    }
    
    while (ch->count == 0) {
        int rc = pthread_cond_timedwait(ch->cond_not_empty, ch->mutex, &ts);
        if (rc != 0) {
            pthread_mutex_unlock(ch->mutex);
            return -1;  /* Timeout */
        }
    }
    
    if (ch->capacity > 0) {
        char *buf = (char*)ch->buffer;
        memcpy(value, buf + ch->head * ch->elem_size, ch->elem_size);
        ch->head = (ch->head + 1) % ch->capacity;
        ch->count--;
    }
    
    ch->recv_count++;
    pthread_cond_signal(ch->cond_not_full);
    pthread_mutex_unlock(ch->mutex);
    
    return COG_OK;
}

COGINT_API int cog_chan_try_send(CogChan *ch, void *value) {
    if (!ch) return COG_ERR_INVALID;
    
    pthread_mutex_lock(ch->mutex);
    
    if (ch->capacity > 0 && ch->count >= ch->capacity) {
        pthread_mutex_unlock(ch->mutex);
        return -1;  /* Would block */
    }
    
    if (ch->capacity > 0) {
        char *buf = (char*)ch->buffer;
        memcpy(buf + ch->tail * ch->elem_size, value, ch->elem_size);
        ch->tail = (ch->tail + 1) % ch->capacity;
        ch->count++;
        ch->send_count++;
        pthread_cond_signal(ch->cond_not_empty);
    }
    
    pthread_mutex_unlock(ch->mutex);
    return COG_OK;
}

COGINT_API int cog_chan_try_recv(CogChan *ch, void *value) {
    if (!ch || !value) return COG_ERR_INVALID;
    
    pthread_mutex_lock(ch->mutex);
    
    if (ch->count == 0) {
        pthread_mutex_unlock(ch->mutex);
        return -1;  /* Would block */
    }
    
    if (ch->capacity > 0) {
        char *buf = (char*)ch->buffer;
        memcpy(value, buf + ch->head * ch->elem_size, ch->elem_size);
        ch->head = (ch->head + 1) % ch->capacity;
        ch->count--;
        ch->recv_count++;
        pthread_cond_signal(ch->cond_not_full);
    }
    
    pthread_mutex_unlock(ch->mutex);
    return COG_OK;
}

COGINT_API void cog_chan_close(CogChan *ch) {
    if (!ch) return;
    
    pthread_mutex_lock(ch->mutex);
    /* Signal all waiters */
    pthread_cond_broadcast(ch->cond_not_empty);
    pthread_cond_broadcast(ch->cond_not_full);
    pthread_mutex_unlock(ch->mutex);
}

COGINT_API void cog_chan_free(CogChan *ch) {
    if (!ch) return;
    
    cog_chan_close(ch);
    
    if (ch->remote_conn) {
        cog_9p_disconnect(ch->remote_conn);
    }
    
    pthread_mutex_destroy(ch->mutex);
    pthread_cond_destroy(ch->cond_not_empty);
    pthread_cond_destroy(ch->cond_not_full);
    
    free(ch->mutex);
    free(ch->cond_not_empty);
    free(ch->cond_not_full);
    free(ch->name);
    free(ch->buffer);
    free(ch->tensor_shape);
    free(ch->remote_path);
    free(ch);
}

/*============================================================================
 * Alt Statement Implementation
 *============================================================================*/

COGINT_API CogAlt* cog_alt_create(void) {
    CogAlt *alt = calloc(1, sizeof(CogAlt));
    alt->timeout_ms = -1;
    return alt;
}

COGINT_API int cog_alt_add_send(CogAlt *alt, CogChan *ch, void *value) {
    if (!alt || !ch) return COG_ERR_INVALID;
    
    alt->cases = realloc(alt->cases, (alt->n_cases + 1) * sizeof(CogAltCase));
    CogAltCase *c = &alt->cases[alt->n_cases++];
    
    c->type = COG_ALT_SEND;
    c->chan = ch;
    c->value = value;
    c->ready = 0;
    c->selected = 0;
    
    return COG_OK;
}

COGINT_API int cog_alt_add_recv(CogAlt *alt, CogChan *ch, void *value) {
    if (!alt || !ch) return COG_ERR_INVALID;
    
    alt->cases = realloc(alt->cases, (alt->n_cases + 1) * sizeof(CogAltCase));
    CogAltCase *c = &alt->cases[alt->n_cases++];
    
    c->type = COG_ALT_RECV;
    c->chan = ch;
    c->value = value;
    c->ready = 0;
    c->selected = 0;
    
    return COG_OK;
}

COGINT_API int cog_alt_add_default(CogAlt *alt) {
    if (!alt) return COG_ERR_INVALID;
    
    alt->cases = realloc(alt->cases, (alt->n_cases + 1) * sizeof(CogAltCase));
    CogAltCase *c = &alt->cases[alt->n_cases++];
    
    c->type = COG_ALT_DEFAULT;
    c->chan = NULL;
    c->value = NULL;
    c->ready = 1;  /* Default is always ready */
    c->selected = 0;
    
    alt->has_default = 1;
    
    return COG_OK;
}

COGINT_API int cog_alt_set_timeout(CogAlt *alt, int timeout_ms) {
    if (!alt) return COG_ERR_INVALID;
    alt->timeout_ms = timeout_ms;
    return COG_OK;
}

COGINT_API int cog_alt_select(CogAlt *alt) {
    if (!alt || alt->n_cases == 0) return -1;
    
    double start = get_time_ms();
    
    while (1) {
        /* Check each case for readiness */
        int ready_count = 0;
        int ready_indices[alt->n_cases];
        
        for (size_t i = 0; i < alt->n_cases; i++) {
            CogAltCase *c = &alt->cases[i];
            c->ready = 0;
            
            if (c->type == COG_ALT_DEFAULT) {
                c->ready = 1;
            } else if (c->type == COG_ALT_SEND) {
                pthread_mutex_lock(c->chan->mutex);
                c->ready = (c->chan->capacity == 0 || 
                           c->chan->count < c->chan->capacity);
                pthread_mutex_unlock(c->chan->mutex);
            } else if (c->type == COG_ALT_RECV) {
                pthread_mutex_lock(c->chan->mutex);
                c->ready = (c->chan->count > 0);
                pthread_mutex_unlock(c->chan->mutex);
            }
            
            if (c->ready) {
                ready_indices[ready_count++] = i;
            }
        }
        
        /* If any case is ready, select one randomly */
        if (ready_count > 0) {
            int selected_idx = ready_indices[rand() % ready_count];
            CogAltCase *c = &alt->cases[selected_idx];
            c->selected = 1;
            
            /* Execute the selected case */
            if (c->type == COG_ALT_SEND) {
                cog_chan_send(c->chan, c->value);
            } else if (c->type == COG_ALT_RECV) {
                cog_chan_recv(c->chan, c->value);
            }
            
            return selected_idx;
        }
        
        /* Check timeout */
        if (alt->timeout_ms >= 0) {
            double elapsed = get_time_ms() - start;
            if (elapsed >= alt->timeout_ms) {
                return -1;  /* Timeout */
            }
        }
        
        /* If has default, return it */
        if (alt->has_default) {
            for (size_t i = 0; i < alt->n_cases; i++) {
                if (alt->cases[i].type == COG_ALT_DEFAULT) {
                    alt->cases[i].selected = 1;
                    return i;
                }
            }
        }
        
        /* Wait a bit before retrying */
        usleep(1000);  /* 1ms */
    }
}

COGINT_API void cog_alt_free(CogAlt *alt) {
    if (!alt) return;
    free(alt->cases);
    free(alt);
}

/*============================================================================
 * Worker Implementation
 *============================================================================*/

static void* worker_thread(void *arg) {
    CogWorker *worker = (CogWorker*)arg;
    
    while (worker->running) {
        CogTask *task = NULL;
        
        /* Receive task from channel */
        if (cog_chan_recv_timeout(worker->task_chan, &task, 100) < 0) {
            continue;  /* Timeout, check if still running */
        }
        
        if (!task) continue;
        
        worker->state = COG_WORKER_BUSY;
        double start = get_time_ms();
        
        /* Execute task */
        int status = cog_task_execute(task, worker->ctx);
        task->status = status;
        
        double end = get_time_ms();
        task->end_time = end;
        worker->total_time += (end - start);
        worker->tasks_completed++;
        
        /* Send result */
        cog_chan_send(worker->result_chan, &task);
        
        worker->state = COG_WORKER_IDLE;
    }
    
    worker->state = COG_WORKER_STOPPED;
    return NULL;
}

COGINT_API CogWorker* cog_worker_create(int id, CogContext *ctx) {
    CogWorker *worker = calloc(1, sizeof(CogWorker));
    if (!worker) return NULL;
    
    worker->id = id;
    worker->ctx = ctx;
    worker->state = COG_WORKER_IDLE;
    
    /* Create communication channels */
    char name[64];
    snprintf(name, sizeof(name), "worker_%d_task", id);
    worker->task_chan = cog_chan_create(name, COG_CHAN_TENSOR, 16);
    
    snprintf(name, sizeof(name), "worker_%d_result", id);
    worker->result_chan = cog_chan_create(name, COG_CHAN_TENSOR, 16);
    
    return worker;
}

COGINT_API CogWorker* cog_worker_remote(int id, const char *addr, uint16_t port) {
    CogWorker *worker = calloc(1, sizeof(CogWorker));
    if (!worker) return NULL;
    
    worker->id = id;
    worker->addr = strdup(addr);
    worker->port = port;
    worker->state = COG_WORKER_IDLE;
    
    /* Connect to remote worker */
    worker->conn = cog9p_dial(addr, port);
    if (!worker->conn) {
        free(worker->addr);
        free(worker);
        return NULL;
    }
    
    cog9p_version(worker->conn);
    cog9p_attach(worker->conn, NULL, NULL);
    
    return worker;
}

COGINT_API int cog_worker_start(CogWorker *worker) {
    if (!worker) return COG_ERR_INVALID;
    
    worker->running = 1;
    worker->thread = malloc(sizeof(pthread_t));
    pthread_create(worker->thread, NULL, worker_thread, worker);
    
    return COG_OK;
}

COGINT_API int cog_worker_stop(CogWorker *worker) {
    if (!worker) return COG_ERR_INVALID;
    
    worker->running = 0;
    
    if (worker->thread) {
        pthread_join(*(pthread_t*)worker->thread, NULL);
        free(worker->thread);
        worker->thread = NULL;
    }
    
    return COG_OK;
}

COGINT_API void cog_worker_free(CogWorker *worker) {
    if (!worker) return;
    
    cog_worker_stop(worker);
    
    if (worker->task_chan) cog_chan_free(worker->task_chan);
    if (worker->result_chan) cog_chan_free(worker->result_chan);
    if (worker->conn) cog_9p_disconnect(worker->conn);
    
    free(worker->addr);
    free(worker);
}

/*============================================================================
 * Worker Pool Implementation
 *============================================================================*/

COGINT_API CogWorkerPool* cog_pool_create(int n_workers, CogContext *ctx) {
    CogWorkerPool *pool = calloc(1, sizeof(CogWorkerPool));
    if (!pool) return NULL;
    
    pool->capacity = n_workers;
    pool->workers = calloc(n_workers, sizeof(CogWorker*));
    
    /* Create task and result queues */
    pool->task_queue = cog_chan_create("pool_tasks", COG_CHAN_TENSOR, 256);
    pool->result_queue = cog_chan_create("pool_results", COG_CHAN_TENSOR, 256);
    
    /* Create workers */
    for (int i = 0; i < n_workers; i++) {
        CogWorker *worker = cog_worker_create(i, ctx);
        if (worker) {
            /* Connect worker channels to pool queues */
            cog_chan_free(worker->task_chan);
            cog_chan_free(worker->result_chan);
            worker->task_chan = pool->task_queue;
            worker->result_chan = pool->result_queue;
            
            pool->workers[pool->n_workers++] = worker;
            cog_worker_start(worker);
        }
    }
    
    return pool;
}

COGINT_API int cog_pool_add_remote(CogWorkerPool *pool, const char *addr,
                                    uint16_t port) {
    if (!pool) return COG_ERR_INVALID;
    
    CogWorker *worker = cog_worker_remote(pool->n_workers, addr, port);
    if (!worker) return COG_ERR_NETWORK;
    
    /* Expand if needed */
    if (pool->n_workers >= pool->capacity) {
        pool->capacity *= 2;
        pool->workers = realloc(pool->workers, 
                                pool->capacity * sizeof(CogWorker*));
    }
    
    pool->workers[pool->n_workers++] = worker;
    
    return COG_OK;
}

COGINT_API int cog_pool_submit(CogWorkerPool *pool, CogTask *task) {
    if (!pool || !task) return COG_ERR_INVALID;
    
    task->start_time = get_time_ms();
    return cog_chan_send(pool->task_queue, &task);
}

COGINT_API int cog_pool_wait(CogWorkerPool *pool) {
    if (!pool) return COG_ERR_INVALID;
    
    /* Wait for task queue to be empty */
    while (pool->task_queue->count > 0) {
        usleep(10000);  /* 10ms */
    }
    
    return COG_OK;
}

COGINT_API void cog_pool_free(CogWorkerPool *pool) {
    if (!pool) return;
    
    /* Stop all workers */
    for (size_t i = 0; i < pool->n_workers; i++) {
        if (pool->workers[i]) {
            /* Restore original channels before freeing */
            pool->workers[i]->task_chan = NULL;
            pool->workers[i]->result_chan = NULL;
            cog_worker_free(pool->workers[i]);
        }
    }
    
    cog_chan_free(pool->task_queue);
    cog_chan_free(pool->result_queue);
    free(pool->workers);
    free(pool);
}

/*============================================================================
 * Task Implementation
 *============================================================================*/

COGINT_API CogTask* cog_task_tensor_op(uint8_t op, CogTensor **inputs,
                                        size_t n_inputs) {
    CogTask *task = calloc(1, sizeof(CogTask));
    if (!task) return NULL;
    
    task->type = COG_TASK_TENSOR_OP;
    task->input_tensors = malloc(n_inputs * sizeof(CogTensor*));
    memcpy(task->input_tensors, inputs, n_inputs * sizeof(CogTensor*));
    task->n_input_tensors = n_inputs;
    
    /* Store operation code in params */
    task->params = malloc(sizeof(uint8_t));
    *(uint8_t*)task->params = op;
    task->params_size = sizeof(uint8_t);
    
    return task;
}

COGINT_API CogTask* cog_task_pln(CogAtomSpace *as, CogAtom *query,
                                  CogPLNConfig *config) {
    CogTask *task = calloc(1, sizeof(CogTask));
    if (!task) return NULL;
    
    task->type = COG_TASK_PLN_INFER;
    task->input_atoms = malloc(sizeof(CogAtom*));
    task->input_atoms[0] = query;
    task->n_input_atoms = 1;
    
    if (config) {
        task->params = malloc(sizeof(CogPLNConfig));
        memcpy(task->params, config, sizeof(CogPLNConfig));
        task->params_size = sizeof(CogPLNConfig);
    }
    
    return task;
}

COGINT_API CogTask* cog_task_ecan(CogAtomSpace *as, CogAtom *source, int steps) {
    CogTask *task = calloc(1, sizeof(CogTask));
    if (!task) return NULL;
    
    task->type = COG_TASK_ECAN_SPREAD;
    task->input_atoms = malloc(sizeof(CogAtom*));
    task->input_atoms[0] = source;
    task->n_input_atoms = 1;
    
    task->params = malloc(sizeof(int));
    *(int*)task->params = steps;
    task->params_size = sizeof(int);
    
    return task;
}

COGINT_API CogTask* cog_task_gnn(CogAtomSpace *as, CogTensor **weights,
                                  int n_layers) {
    CogTask *task = calloc(1, sizeof(CogTask));
    if (!task) return NULL;
    
    task->type = COG_TASK_GNN_FORWARD;
    task->input_tensors = malloc(n_layers * sizeof(CogTensor*));
    memcpy(task->input_tensors, weights, n_layers * sizeof(CogTensor*));
    task->n_input_tensors = n_layers;
    
    return task;
}

COGINT_API CogTask* cog_task_custom(int (*execute)(CogTask*, CogContext*),
                                     void *params, size_t params_size) {
    CogTask *task = calloc(1, sizeof(CogTask));
    if (!task) return NULL;
    
    task->type = COG_TASK_CUSTOM;
    task->execute = execute;
    
    if (params && params_size > 0) {
        task->params = malloc(params_size);
        memcpy(task->params, params, params_size);
        task->params_size = params_size;
    }
    
    return task;
}

COGINT_API int cog_task_add_dep(CogTask *task, CogTask *dep) {
    if (!task || !dep) return COG_ERR_INVALID;
    
    task->deps = realloc(task->deps, (task->n_deps + 1) * sizeof(CogTask*));
    task->deps[task->n_deps++] = dep;
    
    return COG_OK;
}

COGINT_API int cog_task_execute(CogTask *task, CogContext *ctx) {
    if (!task) return COG_ERR_INVALID;
    
    /* Check dependencies */
    for (size_t i = 0; i < task->n_deps; i++) {
        if (task->deps[i]->status != COG_OK) {
            return COG_ERR_INVALID;  /* Dependency not completed */
        }
    }
    
    /* Execute based on type */
    switch (task->type) {
        case COG_TASK_TENSOR_OP: {
            uint8_t op = *(uint8_t*)task->params;
            if (task->n_input_tensors >= 2) {
                switch (op) {
                    case COG9P_TOP_ADD:
                        task->output_tensor = cog_tensor_add(
                            task->input_tensors[0], task->input_tensors[1]);
                        break;
                    case COG9P_TOP_MUL:
                        task->output_tensor = cog_tensor_mul(
                            task->input_tensors[0], task->input_tensors[1]);
                        break;
                    case COG9P_TOP_MATMUL:
                        task->output_tensor = cog_tensor_matmul(
                            task->input_tensors[0], task->input_tensors[1]);
                        break;
                }
            } else if (task->n_input_tensors >= 1) {
                switch (op) {
                    case COG9P_TOP_SOFTMAX:
                        task->output_tensor = cog_tensor_softmax(
                            task->input_tensors[0], -1);
                        break;
                }
            }
            break;
        }
        
        case COG_TASK_CUSTOM:
            if (task->execute) {
                return task->execute(task, ctx);
            }
            break;
            
        default:
            break;
    }
    
    return COG_OK;
}

COGINT_API void cog_task_free(CogTask *task) {
    if (!task) return;
    
    free(task->input_tensors);
    free(task->input_atoms);
    free(task->params);
    free(task->output_atoms);
    free(task->error);
    free(task->deps);
    
    if (task->cleanup) {
        task->cleanup(task);
    }
    
    free(task);
}

/*============================================================================
 * Task Graph Implementation
 *============================================================================*/

COGINT_API CogTaskGraph* cog_graph_create(void) {
    CogTaskGraph *graph = calloc(1, sizeof(CogTaskGraph));
    graph->capacity = 64;
    graph->tasks = calloc(graph->capacity, sizeof(CogTask*));
    return graph;
}

COGINT_API int cog_graph_add(CogTaskGraph *graph, CogTask *task) {
    if (!graph || !task) return COG_ERR_INVALID;
    
    if (graph->n_tasks >= graph->capacity) {
        graph->capacity *= 2;
        graph->tasks = realloc(graph->tasks, 
                               graph->capacity * sizeof(CogTask*));
    }
    
    graph->tasks[graph->n_tasks++] = task;
    return COG_OK;
}

COGINT_API int cog_graph_execute(CogTaskGraph *graph, CogWorkerPool *pool) {
    if (!graph || !pool) return COG_ERR_INVALID;
    
    graph->running = 1;
    graph->completed = 0;
    graph->failed = 0;
    
    /* Submit tasks with no dependencies first */
    for (size_t i = 0; i < graph->n_tasks; i++) {
        CogTask *task = graph->tasks[i];
        if (task->n_deps == 0) {
            cog_pool_submit(pool, task);
        }
    }
    
    /* Process results and submit dependent tasks */
    while (graph->completed + graph->failed < graph->n_tasks) {
        CogTask *completed_task = NULL;
        
        if (cog_chan_recv_timeout(pool->result_queue, &completed_task, 100) < 0) {
            continue;
        }
        
        if (!completed_task) continue;
        
        if (completed_task->status == COG_OK) {
            graph->completed++;
            
            /* Check if any tasks are now ready */
            for (size_t i = 0; i < graph->n_tasks; i++) {
                CogTask *task = graph->tasks[i];
                if (task->status != 0) continue;  /* Already processed */
                
                /* Check if all dependencies are complete */
                int ready = 1;
                for (size_t j = 0; j < task->n_deps; j++) {
                    if (task->deps[j]->status != COG_OK) {
                        ready = 0;
                        break;
                    }
                }
                
                if (ready && task->n_deps > 0) {
                    cog_pool_submit(pool, task);
                }
            }
        } else {
            graph->failed++;
        }
    }
    
    graph->running = 0;
    return (graph->failed == 0) ? COG_OK : COG_ERR_INVALID;
}

COGINT_API void cog_graph_free(CogTaskGraph *graph) {
    if (!graph) return;
    
    for (size_t i = 0; i < graph->n_tasks; i++) {
        cog_task_free(graph->tasks[i]);
    }
    
    free(graph->tasks);
    free(graph);
}

/*============================================================================
 * Namespace Implementation
 *============================================================================*/

COGINT_API CogNamespace* cog_ns_create(void) {
    CogNamespace *ns = calloc(1, sizeof(CogNamespace));
    ns->root = strdup("/");
    return ns;
}

COGINT_API int cog_ns_bind(CogNamespace *ns, const char *path,
                            const char *target, int flags) {
    if (!ns || !path || !target) return COG_ERR_INVALID;
    
    CogNsEntry *entry = calloc(1, sizeof(CogNsEntry));
    entry->path = strdup(path);
    entry->target = strdup(target);
    entry->flags = flags;
    
    entry->next = ns->mounts;
    ns->mounts = entry;
    
    return COG_OK;
}

COGINT_API int cog_ns_mount(CogNamespace *ns, const char *path,
                             Cog9PConn *conn, const char *aname, int flags) {
    if (!ns || !path || !conn) return COG_ERR_INVALID;
    
    CogNsEntry *entry = calloc(1, sizeof(CogNsEntry));
    entry->path = strdup(path);
    entry->conn = conn;
    entry->flags = flags;
    
    entry->next = ns->mounts;
    ns->mounts = entry;
    
    return COG_OK;
}

COGINT_API int cog_ns_mount_atomspace(CogNamespace *ns, const char *path,
                                       CogAtomSpace *as, int flags) {
    if (!ns || !path || !as) return COG_ERR_INVALID;
    
    CogNsEntry *entry = calloc(1, sizeof(CogNsEntry));
    entry->path = strdup(path);
    entry->atomspace = as;
    entry->flags = flags;
    
    entry->next = ns->mounts;
    ns->mounts = entry;
    
    return COG_OK;
}

COGINT_API int cog_ns_mount_chan(CogNamespace *ns, const char *path,
                                  CogChan *ch, int flags) {
    if (!ns || !path || !ch) return COG_ERR_INVALID;
    
    CogNsEntry *entry = calloc(1, sizeof(CogNsEntry));
    entry->path = strdup(path);
    entry->chan = ch;
    entry->flags = flags;
    
    entry->next = ns->mounts;
    ns->mounts = entry;
    
    return COG_OK;
}

COGINT_API void* cog_ns_resolve(CogNamespace *ns, const char *path, int *type) {
    if (!ns || !path) return NULL;
    
    /* Find longest matching mount */
    CogNsEntry *best = NULL;
    size_t best_len = 0;
    
    for (CogNsEntry *e = ns->mounts; e; e = e->next) {
        size_t len = strlen(e->path);
        if (strncmp(path, e->path, len) == 0 && len > best_len) {
            best = e;
            best_len = len;
        }
    }
    
    if (!best) return NULL;
    
    if (type) {
        if (best->conn) *type = 1;
        else if (best->atomspace) *type = 2;
        else if (best->chan) *type = 3;
        else *type = 0;
    }
    
    if (best->conn) return best->conn;
    if (best->atomspace) return best->atomspace;
    if (best->chan) return best->chan;
    
    return NULL;
}

COGINT_API void cog_ns_free(CogNamespace *ns) {
    if (!ns) return;
    
    CogNsEntry *e = ns->mounts;
    while (e) {
        CogNsEntry *next = e->next;
        free(e->path);
        free(e->target);
        free(e);
        e = next;
    }
    
    free(ns->unions);
    free(ns->root);
    free(ns);
}

/*============================================================================
 * Runtime Implementation
 *============================================================================*/

COGINT_API CogRuntime* cog_runtime_create(CogRuntimeConfig *config) {
    CogRuntime *rt = calloc(1, sizeof(CogRuntime));
    if (!rt) return NULL;
    
    /* Initialize context */
    rt->ctx = cogint_init();
    
    /* Create AtomSpace */
    rt->atomspace = cog_atomspace_create(rt->ctx, "runtime");
    
    /* Create namespace */
    rt->ns = cog_ns_create();
    
    /* Mount AtomSpace */
    cog_ns_mount_atomspace(rt->ns, "/atom", rt->atomspace, COG_MREPL);
    
    /* Create worker pools */
    int n_workers = config ? config->n_local_workers : 4;
    rt->local_pool = cog_pool_create(n_workers, rt->ctx);
    
    /* Add remote workers if configured */
    if (config && config->remote_addrs) {
        for (size_t i = 0; i < config->n_remotes; i++) {
            cog_pool_add_remote(rt->local_pool, 
                               config->remote_addrs[i],
                               config->remote_ports[i]);
        }
    }
    
    /* Create 9P server if configured */
    if (config && config->listen_addr) {
        Cog9PServerConfig srv_config = {
            .addr = config->listen_addr,
            .port = config->listen_port,
            .cogctx = rt->ctx,
            .atomspace = rt->atomspace,
        };
        rt->server = cog9p_server_create(&srv_config);
    }
    
    return rt;
}

COGINT_API int cog_runtime_start(CogRuntime *rt) {
    if (!rt) return COG_ERR_INVALID;
    
    rt->running = 1;
    
    if (rt->server) {
        cog9p_server_start(rt->server);
    }
    
    return COG_OK;
}

COGINT_API int cog_runtime_stop(CogRuntime *rt) {
    if (!rt) return COG_ERR_INVALID;
    
    rt->running = 0;
    
    if (rt->server) {
        cog9p_server_stop(rt->server);
    }
    
    return COG_OK;
}

COGINT_API int cog_runtime_submit(CogRuntime *rt, CogTask *task) {
    if (!rt || !task) return COG_ERR_INVALID;
    
    task->id = rt->next_task_id++;
    return cog_pool_submit(rt->local_pool, task);
}

COGINT_API CogChan* cog_runtime_chan(CogRuntime *rt, const char *name,
                                      CogChanType type, size_t capacity) {
    if (!rt) return NULL;
    
    CogChan *ch = cog_chan_create(name, type, capacity);
    if (!ch) return NULL;
    
    /* Add to runtime */
    rt->channels = realloc(rt->channels, 
                           (rt->n_channels + 1) * sizeof(CogChan*));
    rt->channels[rt->n_channels++] = ch;
    
    /* Mount in namespace */
    char path[256];
    snprintf(path, sizeof(path), "/chan/%s", name);
    cog_ns_mount_chan(rt->ns, path, ch, COG_MREPL);
    
    return ch;
}

COGINT_API CogAtomSpace* cog_runtime_atomspace(CogRuntime *rt) {
    return rt ? rt->atomspace : NULL;
}

COGINT_API CogNamespace* cog_runtime_ns(CogRuntime *rt) {
    return rt ? rt->ns : NULL;
}

COGINT_API void cog_runtime_free(CogRuntime *rt) {
    if (!rt) return;
    
    cog_runtime_stop(rt);
    
    if (rt->server) cog9p_server_free(rt->server);
    if (rt->local_pool) cog_pool_free(rt->local_pool);
    if (rt->remote_pool) cog_pool_free(rt->remote_pool);
    
    for (size_t i = 0; i < rt->n_channels; i++) {
        cog_chan_free(rt->channels[i]);
    }
    free(rt->channels);
    
    if (rt->current_graph) cog_graph_free(rt->current_graph);
    if (rt->ns) cog_ns_free(rt->ns);
    if (rt->atomspace) cog_atomspace_free(rt->atomspace);
    if (rt->ctx) cogint_shutdown(rt->ctx);
    
    free(rt);
}
