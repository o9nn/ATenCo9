/**
 * Cog9P - 9P Protocol Implementation for Distributed Tensor Computing
 * 
 * Implements the 9P2000 protocol with extensions for tensor operations.
 * Based on Plan 9's file protocol, enabling network-transparent tensor access.
 * 
 * Key Features:
 * - Standard 9P2000 message types
 * - Extended tensor-specific operations (Ttensor, Rtensor)
 * - AtomSpace query protocol (Tatom, Ratom)
 * - Streaming tensor data transfer
 * 
 * Protocol Extensions:
 * - Ttensor/Rtensor: Create, read, write, and operate on tensors
 * - Tatom/Ratom: Query and update AtomSpace atoms
 * - Tcog/Rcog: Cognitive operations (PLN inference, ECAN spread)
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#ifndef COG9P_H
#define COG9P_H

#include "cogint.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * 9P Protocol Constants
 *============================================================================*/

/* Standard 9P2000 message types */
#define COG9P_TVERSION    100
#define COG9P_RVERSION    101
#define COG9P_TAUTH       102
#define COG9P_RAUTH       103
#define COG9P_TATTACH     104
#define COG9P_RATTACH     105
#define COG9P_TERROR      106   /* Illegal */
#define COG9P_RERROR      107
#define COG9P_TFLUSH      108
#define COG9P_RFLUSH      109
#define COG9P_TWALK       110
#define COG9P_RWALK       111
#define COG9P_TOPEN       112
#define COG9P_ROPEN       113
#define COG9P_TCREATE     114
#define COG9P_RCREATE     115
#define COG9P_TREAD       116
#define COG9P_RREAD       117
#define COG9P_TWRITE      118
#define COG9P_RWRITE      119
#define COG9P_TCLUNK      120
#define COG9P_RCLUNK      121
#define COG9P_TREMOVE     122
#define COG9P_RREMOVE     123
#define COG9P_TSTAT       124
#define COG9P_RSTAT       125
#define COG9P_TWSTAT      126
#define COG9P_RWSTAT      127

/* CogInt extension message types (128-255 reserved) */
#define COG9P_TTENSOR     128   /* Tensor operation request */
#define COG9P_RTENSOR     129   /* Tensor operation response */
#define COG9P_TATOM       130   /* AtomSpace operation request */
#define COG9P_RATOM       131   /* AtomSpace operation response */
#define COG9P_TCOG        132   /* Cognitive operation request */
#define COG9P_RCOG        133   /* Cognitive operation response */
#define COG9P_TSTREAM     134   /* Stream tensor data */
#define COG9P_RSTREAM     135   /* Stream acknowledgment */

/* Protocol version */
#define COG9P_VERSION     "9P2000.cog"
#define COG9P_MSIZE       (8 * 1024 * 1024)  /* 8MB max message */

/* Open modes */
#define COG9P_OREAD       0x00
#define COG9P_OWRITE      0x01
#define COG9P_ORDWR       0x02
#define COG9P_OEXEC       0x03
#define COG9P_OTRUNC      0x10
#define COG9P_ORCLOSE     0x40

/* File types */
#define COG9P_QTDIR       0x80
#define COG9P_QTAPPEND    0x40
#define COG9P_QTEXCL      0x20
#define COG9P_QTAUTH      0x08
#define COG9P_QTTENSOR    0x04  /* Extension: tensor file */
#define COG9P_QTATOM      0x02  /* Extension: atom file */
#define COG9P_QTFILE      0x00

/* Tensor operation codes */
#define COG9P_TOP_CREATE  0x01
#define COG9P_TOP_RESHAPE 0x02
#define COG9P_TOP_SLICE   0x03
#define COG9P_TOP_ADD     0x10
#define COG9P_TOP_MUL     0x11
#define COG9P_TOP_MATMUL  0x12
#define COG9P_TOP_SOFTMAX 0x13
#define COG9P_TOP_RELU    0x14
#define COG9P_TOP_CONV    0x15
#define COG9P_TOP_POOL    0x16

/* Atom operation codes */
#define COG9P_AOP_CREATE  0x01
#define COG9P_AOP_DELETE  0x02
#define COG9P_AOP_QUERY   0x03
#define COG9P_AOP_LINK    0x04
#define COG9P_AOP_SETTV   0x05
#define COG9P_AOP_SETAV   0x06

/* Cognitive operation codes */
#define COG9P_COP_PLN     0x01  /* PLN inference */
#define COG9P_COP_ECAN    0x02  /* ECAN attention spread */
#define COG9P_COP_URE     0x03  /* Unified Rule Engine */
#define COG9P_COP_MINE    0x04  /* Pattern mining */

/*============================================================================
 * 9P Message Structures
 *============================================================================*/

/* QID - Unique file identifier */
typedef struct {
    uint8_t type;
    uint32_t version;
    uint64_t path;
} Cog9PQid;

/* Stat - File information */
typedef struct {
    uint16_t size;
    uint16_t type;
    uint32_t dev;
    Cog9PQid qid;
    uint32_t mode;
    uint32_t atime;
    uint32_t mtime;
    uint64_t length;
    char *name;
    char *uid;
    char *gid;
    char *muid;
    
    /* Tensor extension */
    CogDType tensor_dtype;
    int64_t *tensor_shape;
    int tensor_ndim;
} Cog9PStat;

/* Base message header */
typedef struct {
    uint32_t size;
    uint8_t type;
    uint16_t tag;
} Cog9PMsgHdr;

/* Tversion/Rversion */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t msize;
    char *version;
} Cog9PVersion;

/* Tattach/Rattach */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t fid;
    uint32_t afid;
    char *uname;
    char *aname;
    Cog9PQid qid;  /* Response only */
} Cog9PAttach;

/* Twalk/Rwalk */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t fid;
    uint32_t newfid;
    uint16_t nwname;
    char **wname;
    uint16_t nwqid;
    Cog9PQid *wqid;
} Cog9PWalk;

/* Tread/Rread */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t fid;
    uint64_t offset;
    uint32_t count;
    uint8_t *data;  /* Response only */
} Cog9PRead;

/* Twrite/Rwrite */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t fid;
    uint64_t offset;
    uint32_t count;
    uint8_t *data;
} Cog9PWrite;

/* Rerror */
typedef struct {
    Cog9PMsgHdr hdr;
    char *ename;
} Cog9PError;

/*============================================================================
 * CogInt Extension Messages
 *============================================================================*/

/* Ttensor - Tensor operation request */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t fid;
    uint8_t op;           /* Operation code */
    CogDType dtype;
    int ndim;
    int64_t *shape;
    uint32_t data_size;
    uint8_t *data;
    
    /* For binary ops */
    uint32_t fid2;        /* Second operand fid */
    
    /* For operations with parameters */
    int param_int;
    float param_float;
} Cog9PTensor;

/* Rtensor - Tensor operation response */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t fid;         /* Result tensor fid */
    Cog9PQid qid;
    CogDType dtype;
    int ndim;
    int64_t *shape;
    uint32_t data_size;
    uint8_t *data;        /* Optional inline data */
} Cog9PRTensor;

/* Tatom - AtomSpace operation request */
typedef struct {
    Cog9PMsgHdr hdr;
    uint8_t op;           /* Operation code */
    CogAtomType type;
    char *name;
    uint64_t handle;
    
    /* Truth value */
    double strength;
    double confidence;
    
    /* Attention value */
    int16_t sti;
    int16_t lti;
    int16_t vlti;
    
    /* For links */
    uint16_t arity;
    uint64_t *outgoing;
    
    /* Associated tensor */
    uint32_t tensor_fid;
} Cog9PAtom;

/* Ratom - AtomSpace operation response */
typedef struct {
    Cog9PMsgHdr hdr;
    uint64_t handle;
    CogAtomType type;
    char *name;
    double strength;
    double confidence;
    int16_t sti;
    int16_t lti;
    int16_t vlti;
    uint16_t arity;
    uint64_t *outgoing;
    uint32_t tensor_fid;
} Cog9PRAtom;

/* Tcog - Cognitive operation request */
typedef struct {
    Cog9PMsgHdr hdr;
    uint8_t op;           /* Cognitive operation code */
    uint64_t source;      /* Source atom handle */
    int steps;            /* For iterative operations */
    char *query;          /* Query string (e.g., for pattern mining) */
    uint32_t max_results;
} Cog9PCog;

/* Rcog - Cognitive operation response */
typedef struct {
    Cog9PMsgHdr hdr;
    uint32_t n_results;
    uint64_t *results;    /* Result atom handles */
    double *confidences;  /* Result confidences */
} Cog9PRCog;

/*============================================================================
 * 9P Server/Client API
 *============================================================================*/

/* Server context */
typedef struct Cog9PServer Cog9PServer;

/* Request handler callback */
typedef int (*Cog9PHandler)(Cog9PServer *srv, void *msg, void *response);

/* Server configuration */
typedef struct {
    char *addr;
    uint16_t port;
    uint32_t msize;
    CogContext *cogctx;
    CogAtomSpace *atomspace;
    
    /* Handlers */
    Cog9PHandler on_attach;
    Cog9PHandler on_walk;
    Cog9PHandler on_read;
    Cog9PHandler on_write;
    Cog9PHandler on_tensor;
    Cog9PHandler on_atom;
    Cog9PHandler on_cog;
} Cog9PServerConfig;

/* Server API */
COGINT_API Cog9PServer* cog9p_server_create(Cog9PServerConfig *config);
COGINT_API int cog9p_server_start(Cog9PServer *srv);
COGINT_API int cog9p_server_stop(Cog9PServer *srv);
COGINT_API void cog9p_server_free(Cog9PServer *srv);

/* Client API */
COGINT_API Cog9PConn* cog9p_dial(const char *addr, uint16_t port);
COGINT_API int cog9p_version(Cog9PConn *conn);
COGINT_API int cog9p_attach(Cog9PConn *conn, const char *uname, const char *aname);
COGINT_API int cog9p_walk(Cog9PConn *conn, uint32_t fid, uint32_t newfid, 
                          char **names, int nnames, Cog9PQid *qids);
COGINT_API int cog9p_open(Cog9PConn *conn, uint32_t fid, uint8_t mode);
COGINT_API int cog9p_read(Cog9PConn *conn, uint32_t fid, uint64_t offset, 
                          uint32_t count, uint8_t *data, uint32_t *nread);
COGINT_API int cog9p_write(Cog9PConn *conn, uint32_t fid, uint64_t offset,
                           uint32_t count, uint8_t *data, uint32_t *nwritten);
COGINT_API int cog9p_clunk(Cog9PConn *conn, uint32_t fid);
COGINT_API int cog9p_stat(Cog9PConn *conn, uint32_t fid, Cog9PStat *stat);

/* Tensor extension API */
COGINT_API int cog9p_tensor_create(Cog9PConn *conn, const char *path,
                                    CogDType dtype, int64_t *shape, int ndim,
                                    uint32_t *fid);
COGINT_API int cog9p_tensor_read(Cog9PConn *conn, uint32_t fid, CogTensor **tensor);
COGINT_API int cog9p_tensor_write(Cog9PConn *conn, uint32_t fid, CogTensor *tensor);
COGINT_API int cog9p_tensor_op(Cog9PConn *conn, uint8_t op, uint32_t fid1, 
                                uint32_t fid2, uint32_t *result_fid);

/* AtomSpace extension API */
COGINT_API int cog9p_atom_create(Cog9PConn *conn, CogAtomType type, 
                                  const char *name, uint64_t *handle);
COGINT_API int cog9p_atom_query(Cog9PConn *conn, const char *pattern,
                                 uint64_t *handles, uint32_t *n);
COGINT_API int cog9p_atom_link(Cog9PConn *conn, CogAtomType type,
                                uint64_t *outgoing, uint16_t arity,
                                uint64_t *handle);

/* Cognitive extension API */
COGINT_API int cog9p_pln_infer(Cog9PConn *conn, uint64_t query, int steps,
                                uint64_t *results, double *confidences, uint32_t *n);
COGINT_API int cog9p_ecan_spread(Cog9PConn *conn, uint64_t source, int steps);

/*============================================================================
 * Message Encoding/Decoding
 *============================================================================*/

COGINT_API int cog9p_encode_msg(void *msg, uint8_t type, uint8_t *buf, size_t *len);
COGINT_API int cog9p_decode_msg(uint8_t *buf, size_t len, uint8_t *type, void **msg);
COGINT_API void cog9p_free_msg(void *msg, uint8_t type);

/* Tensor serialization */
COGINT_API int cog9p_encode_tensor(CogTensor *tensor, uint8_t *buf, size_t *len);
COGINT_API int cog9p_decode_tensor(uint8_t *buf, size_t len, CogTensor **tensor);

#ifdef __cplusplus
}
#endif

#endif /* COG9P_H */
