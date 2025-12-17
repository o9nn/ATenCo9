/**
 * CogInt - Cognitive Integration Layer for ATenCo9
 * 
 * This header defines the core integration architecture that bridges:
 * - ATen/PyTorch tensor computing
 * - Plan 9/Inferno distributed systems (9P protocol)
 * - OpenCog cognitive architectures (AtomSpace, PLN, ECAN)
 * 
 * Architecture Overview:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    CogInt Unified API Layer                      │
 * ├─────────────────────────────────────────────────────────────────┤
 * │  Cognitive Pipeline  │  Tensor Operations  │  Distributed Ops   │
 * ├─────────────────────────────────────────────────────────────────┤
 * │        AtomSpace Tensor Bindings (atomspace/)                   │
 * │        9P Tensor Protocol (9p/)                                  │
 * │        Inferno Styx Integration (inferno/)                       │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                    ATen Core Tensor Library                      │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#ifndef COGINT_H
#define COGINT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Version information */
#define COGINT_VERSION_MAJOR 0
#define COGINT_VERSION_MINOR 1
#define COGINT_VERSION_PATCH 0
#define COGINT_VERSION_STRING "0.1.0"

/* Export macros for shared library */
#ifdef _WIN32
  #ifdef COGINT_EXPORTS
    #define COGINT_API __declspec(dllexport)
  #else
    #define COGINT_API __declspec(dllimport)
  #endif
#else
  #define COGINT_API __attribute__((visibility("default")))
#endif

/*============================================================================
 * Core Types and Structures
 *============================================================================*/

/* Forward declarations */
typedef struct CogTensor CogTensor;
typedef struct CogAtom CogAtom;
typedef struct CogAtomSpace CogAtomSpace;
typedef struct Cog9PConn Cog9PConn;
typedef struct CogChannel CogChannel;
typedef struct CogContext CogContext;

/* Tensor data types (aligned with ATen) */
typedef enum {
    COG_DTYPE_FLOAT32 = 0,
    COG_DTYPE_FLOAT64 = 1,
    COG_DTYPE_FLOAT16 = 2,
    COG_DTYPE_BFLOAT16 = 3,
    COG_DTYPE_INT8 = 4,
    COG_DTYPE_INT16 = 5,
    COG_DTYPE_INT32 = 6,
    COG_DTYPE_INT64 = 7,
    COG_DTYPE_UINT8 = 8,
    COG_DTYPE_BOOL = 9,
    COG_DTYPE_COMPLEX64 = 10,
    COG_DTYPE_COMPLEX128 = 11,
} CogDType;

/* Device types for tensor placement */
typedef enum {
    COG_DEVICE_CPU = 0,
    COG_DEVICE_CUDA = 1,
    COG_DEVICE_DISTRIBUTED = 2,  /* Distributed across 9P network */
    COG_DEVICE_ATOMSPACE = 3,    /* Stored in AtomSpace */
} CogDeviceType;

/* Atom types for OpenCog integration - see cog_atomspace.h for full definitions */
typedef enum {
    COG_ATOM_NODE = 0,
    COG_ATOM_LINK = 1,
    COG_ATOM_CONCEPT = 4,
    COG_ATOM_PREDICATE = 5,
    COG_ATOM_EVALUATION = 6,
    COG_ATOM_INHERITANCE = 7,
    COG_ATOM_SIMILARITY = 8,
} CogAtomType;

/* 9P message types for tensor operations */
typedef enum {
    COG_9P_TENSOR_CREATE = 100,
    COG_9P_TENSOR_READ = 101,
    COG_9P_TENSOR_WRITE = 102,
    COG_9P_TENSOR_DELETE = 103,
    COG_9P_TENSOR_OP = 104,        /* Execute operation */
    COG_9P_TENSOR_SYNC = 105,      /* Synchronize distributed tensor */
    COG_9P_ATOM_QUERY = 106,       /* Query AtomSpace */
    COG_9P_ATOM_UPDATE = 107,      /* Update AtomSpace */
} Cog9PMsgType;

/* Error codes */
typedef enum {
    COG_OK = 0,
    COG_ERR_NOMEM = -1,
    COG_ERR_INVALID = -2,
    COG_ERR_NOTFOUND = -3,
    COG_ERR_NETWORK = -4,
    COG_ERR_PROTOCOL = -5,
    COG_ERR_ATOMSPACE = -6,
    COG_ERR_TENSOR = -7,
    COG_ERR_DEVICE = -8,
} CogError;

/*============================================================================
 * Tensor Structure
 *============================================================================*/

/**
 * CogTensor - Unified tensor representation
 * 
 * Extends ATen tensors with:
 * - 9P network distribution capabilities
 * - AtomSpace integration for semantic annotation
 * - Inferno channel-based communication
 */
struct CogTensor {
    void *data;                    /* Raw data pointer */
    int64_t *shape;                /* Tensor dimensions */
    int64_t *strides;              /* Memory strides */
    int ndim;                      /* Number of dimensions */
    size_t numel;                  /* Total number of elements */
    CogDType dtype;                /* Data type */
    CogDeviceType device;          /* Device placement */
    
    /* ATen integration */
    void *aten_tensor;             /* Underlying ATen tensor (if any) */
    
    /* 9P distribution */
    char *path;                    /* 9P namespace path */
    Cog9PConn *conn;               /* 9P connection for distributed tensors */
    uint32_t fid;                  /* 9P file identifier */
    
    /* AtomSpace integration */
    CogAtom *atom;                 /* Associated atom (if any) */
    uint64_t atom_handle;          /* AtomSpace handle */
    
    /* Reference counting */
    int refcount;
};

/*============================================================================
 * AtomSpace Integration
 *============================================================================*/

/**
 * CogAtom - OpenCog atom with tensor support
 */
struct CogAtom {
    CogAtomType type;
    char *name;                    /* Atom name/identifier */
    uint64_t handle;               /* AtomSpace handle */
    
    /* Truth value (for PLN) */
    double strength;               /* [0, 1] confidence in truth */
    double confidence;             /* [0, 1] certainty of strength */
    
    /* Attention value (for ECAN) */
    int16_t sti;                   /* Short-term importance */
    int16_t lti;                   /* Long-term importance */
    int16_t vlti;                  /* Very long-term importance */
    
    /* Tensor association */
    CogTensor *tensor;             /* Associated tensor (if any) */
    
    /* Links */
    CogAtom **outgoing;            /* Outgoing set for links */
    size_t arity;                  /* Number of outgoing atoms */
    
    /* Reference counting */
    int refcount;
};

/**
 * CogAtomSpace - Knowledge hypergraph with tensor support
 */
struct CogAtomSpace {
    void *impl;                    /* Implementation-specific data */
    char *name;                    /* AtomSpace name */
    
    /* 9P export */
    char *export_path;             /* Path in 9P namespace */
    Cog9PConn *server;             /* 9P server for remote access */
    
    /* Statistics */
    size_t atom_count;
    size_t tensor_atom_count;
};

/*============================================================================
 * 9P Protocol Integration
 *============================================================================*/

/**
 * Cog9PConn - 9P connection for distributed tensor operations
 */
struct Cog9PConn {
    int fd;                        /* File descriptor */
    char *addr;                    /* Server address */
    uint16_t port;                 /* Server port */
    uint32_t msize;                /* Maximum message size */
    char *version;                 /* Protocol version */
    
    /* Authentication */
    char *uname;                   /* User name */
    char *aname;                   /* Attach name */
    
    /* State */
    int connected;
    uint32_t next_tag;
    uint32_t next_fid;
};

/*============================================================================
 * Inferno/Limbo Channel Integration
 *============================================================================*/

/**
 * CogChannel - Inferno-style typed channel for tensor communication
 */
struct CogChannel {
    char *name;
    size_t elem_size;              /* Size of channel elements */
    size_t capacity;               /* Buffer capacity (0 = synchronous) */
    
    /* Buffer */
    void *buffer;
    size_t head;
    size_t tail;
    size_t count;
    
    /* Synchronization */
    void *mutex;
    void *cond_send;
    void *cond_recv;
    
    /* Tensor-specific */
    CogDType tensor_dtype;         /* Expected tensor dtype */
    int64_t *tensor_shape;         /* Expected tensor shape */
    int tensor_ndim;
};

/*============================================================================
 * Context and Initialization
 *============================================================================*/

/**
 * CogContext - Global context for CogInt operations
 */
struct CogContext {
    /* AtomSpace */
    CogAtomSpace *default_atomspace;
    
    /* 9P connections */
    Cog9PConn **connections;
    size_t num_connections;
    
    /* Channels */
    CogChannel **channels;
    size_t num_channels;
    
    /* Configuration */
    int use_cuda;
    int distributed_mode;
    char *namespace_root;          /* Root of 9P namespace */
};

/*============================================================================
 * Core API Functions
 *============================================================================*/

/* Initialization and cleanup */
COGINT_API CogContext* cogint_init(void);
COGINT_API void cogint_shutdown(CogContext *ctx);
COGINT_API const char* cogint_version(void);

/* Tensor operations */
COGINT_API CogTensor* cog_tensor_create(CogContext *ctx, int64_t *shape, int ndim, CogDType dtype);
COGINT_API CogTensor* cog_tensor_from_aten(void *aten_tensor);
COGINT_API void* cog_tensor_to_aten(CogTensor *tensor);
COGINT_API void cog_tensor_free(CogTensor *tensor);
COGINT_API CogTensor* cog_tensor_clone(CogTensor *tensor);
COGINT_API int cog_tensor_copy(CogTensor *dst, CogTensor *src);

/* Tensor arithmetic */
COGINT_API CogTensor* cog_tensor_add(CogTensor *a, CogTensor *b);
COGINT_API CogTensor* cog_tensor_mul(CogTensor *a, CogTensor *b);
COGINT_API CogTensor* cog_tensor_matmul(CogTensor *a, CogTensor *b);
COGINT_API CogTensor* cog_tensor_softmax(CogTensor *t, int dim);

/* AtomSpace integration */
COGINT_API CogAtomSpace* cog_atomspace_create(CogContext *ctx, const char *name);
COGINT_API void cog_atomspace_free(CogAtomSpace *as);
COGINT_API CogAtom* cog_atom_create(CogAtomSpace *as, CogAtomType type, const char *name);
COGINT_API CogAtom* cog_tensor_to_atom(CogAtomSpace *as, CogTensor *tensor, const char *name);
COGINT_API CogTensor* cog_atom_to_tensor(CogAtom *atom);
COGINT_API int cog_atom_set_tv(CogAtom *atom, double strength, double confidence);
COGINT_API int cog_atom_set_av(CogAtom *atom, int16_t sti, int16_t lti, int16_t vlti);

/* 9P distribution */
COGINT_API Cog9PConn* cog_9p_connect(CogContext *ctx, const char *addr, uint16_t port);
COGINT_API void cog_9p_disconnect(Cog9PConn *conn);
COGINT_API int cog_tensor_export(CogTensor *tensor, Cog9PConn *conn, const char *path);
COGINT_API CogTensor* cog_tensor_import(CogContext *ctx, Cog9PConn *conn, const char *path);
COGINT_API int cog_atomspace_export(CogAtomSpace *as, Cog9PConn *conn, const char *path);

/* Channel operations (Inferno-style) */
COGINT_API CogChannel* cog_channel_create(CogContext *ctx, const char *name, size_t capacity);
COGINT_API void cog_channel_free(CogChannel *ch);
COGINT_API int cog_channel_send(CogChannel *ch, CogTensor *tensor);
COGINT_API CogTensor* cog_channel_recv(CogChannel *ch);
COGINT_API int cog_channel_select(CogChannel **channels, int n, int *ready);

/* Cognitive operations */
COGINT_API int cog_pln_infer(CogAtomSpace *as, CogAtom *query, CogAtom **results, size_t *n);
COGINT_API int cog_ecan_spread(CogAtomSpace *as, CogAtom *source, int steps);
COGINT_API CogTensor* cog_attention_weights(CogAtomSpace *as, CogAtom **atoms, size_t n);

/* Error handling */
COGINT_API const char* cog_error_string(CogError err);
COGINT_API CogError cog_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* COGINT_H */
