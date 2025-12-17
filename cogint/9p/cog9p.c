/**
 * Cog9P - 9P Protocol Implementation for Distributed Tensor Computing
 * 
 * This file implements the 9P2000.cog protocol, extending standard 9P
 * with tensor and cognitive operations for distributed AGI computing.
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>

#include "../include/cogint.h"
#include "../include/cog9p.h"

/*============================================================================
 * Internal Structures
 *============================================================================*/

/* File entry in 9P namespace */
typedef struct Cog9PFile {
    char *name;
    Cog9PQid qid;
    Cog9PStat stat;
    
    /* Content */
    uint8_t *data;
    size_t data_len;
    
    /* Tensor association */
    CogTensor *tensor;
    
    /* Atom association */
    CogAtom *atom;
    
    /* Tree structure */
    struct Cog9PFile *parent;
    struct Cog9PFile **children;
    size_t n_children;
} Cog9PFile;

/* FID mapping */
typedef struct {
    uint32_t fid;
    Cog9PFile *file;
    uint8_t mode;
    uint64_t offset;
    int open;
} Cog9PFid;

/* Server structure */
struct Cog9PServer {
    int sock;
    char *addr;
    uint16_t port;
    uint32_t msize;
    int running;
    
    /* Configuration */
    CogContext *cogctx;
    CogAtomSpace *atomspace;
    
    /* Namespace root */
    Cog9PFile *root;
    
    /* FID table */
    Cog9PFid *fids;
    size_t n_fids;
    size_t fid_capacity;
    
    /* Path counter for QIDs */
    uint64_t next_path;
    
    /* Handlers */
    Cog9PHandler on_attach;
    Cog9PHandler on_walk;
    Cog9PHandler on_read;
    Cog9PHandler on_write;
    Cog9PHandler on_tensor;
    Cog9PHandler on_atom;
    Cog9PHandler on_cog;
    
    /* Threading */
    pthread_t accept_thread;
    pthread_mutex_t lock;
};

/*============================================================================
 * Utility Functions
 *============================================================================*/

/* Little-endian encoding/decoding */
static inline void put_u8(uint8_t **p, uint8_t v) {
    *(*p)++ = v;
}

static inline void put_u16(uint8_t **p, uint16_t v) {
    *(*p)++ = v & 0xff;
    *(*p)++ = (v >> 8) & 0xff;
}

static inline void put_u32(uint8_t **p, uint32_t v) {
    *(*p)++ = v & 0xff;
    *(*p)++ = (v >> 8) & 0xff;
    *(*p)++ = (v >> 16) & 0xff;
    *(*p)++ = (v >> 24) & 0xff;
}

static inline void put_u64(uint8_t **p, uint64_t v) {
    put_u32(p, v & 0xffffffff);
    put_u32(p, (v >> 32) & 0xffffffff);
}

static inline void put_str(uint8_t **p, const char *s) {
    uint16_t len = s ? strlen(s) : 0;
    put_u16(p, len);
    if (len > 0) {
        memcpy(*p, s, len);
        *p += len;
    }
}

static inline uint8_t get_u8(uint8_t **p) {
    return *(*p)++;
}

static inline uint16_t get_u16(uint8_t **p) {
    uint16_t v = (*p)[0] | ((*p)[1] << 8);
    *p += 2;
    return v;
}

static inline uint32_t get_u32(uint8_t **p) {
    uint32_t v = (*p)[0] | ((*p)[1] << 8) | ((*p)[2] << 16) | ((*p)[3] << 24);
    *p += 4;
    return v;
}

static inline uint64_t get_u64(uint8_t **p) {
    uint64_t lo = get_u32(p);
    uint64_t hi = get_u32(p);
    return lo | (hi << 32);
}

static inline char* get_str(uint8_t **p) {
    uint16_t len = get_u16(p);
    if (len == 0) return NULL;
    char *s = malloc(len + 1);
    memcpy(s, *p, len);
    s[len] = '\0';
    *p += len;
    return s;
}

/*============================================================================
 * QID and Stat Functions
 *============================================================================*/

static void encode_qid(uint8_t **p, Cog9PQid *qid) {
    put_u8(p, qid->type);
    put_u32(p, qid->version);
    put_u64(p, qid->path);
}

static void decode_qid(uint8_t **p, Cog9PQid *qid) {
    qid->type = get_u8(p);
    qid->version = get_u32(p);
    qid->path = get_u64(p);
}

static size_t stat_size(Cog9PStat *stat) {
    size_t size = 2 + 2 + 4 + 13 + 4 + 4 + 4 + 8;  /* Fixed fields + qid */
    size += 2 + (stat->name ? strlen(stat->name) : 0);
    size += 2 + (stat->uid ? strlen(stat->uid) : 0);
    size += 2 + (stat->gid ? strlen(stat->gid) : 0);
    size += 2 + (stat->muid ? strlen(stat->muid) : 0);
    return size;
}

static void encode_stat(uint8_t **p, Cog9PStat *stat) {
    uint8_t *start = *p;
    put_u16(p, 0);  /* Placeholder for size */
    put_u16(p, stat->type);
    put_u32(p, stat->dev);
    encode_qid(p, &stat->qid);
    put_u32(p, stat->mode);
    put_u32(p, stat->atime);
    put_u32(p, stat->mtime);
    put_u64(p, stat->length);
    put_str(p, stat->name);
    put_str(p, stat->uid);
    put_str(p, stat->gid);
    put_str(p, stat->muid);
    
    /* Update size */
    uint16_t size = *p - start - 2;
    start[0] = size & 0xff;
    start[1] = (size >> 8) & 0xff;
}

/*============================================================================
 * File System Functions
 *============================================================================*/

static Cog9PFile* file_create(Cog9PServer *srv, const char *name, uint8_t qtype) {
    Cog9PFile *f = calloc(1, sizeof(Cog9PFile));
    f->name = strdup(name);
    f->qid.type = qtype;
    f->qid.version = 0;
    f->qid.path = srv->next_path++;
    
    f->stat.qid = f->qid;
    f->stat.name = strdup(name);
    f->stat.uid = strdup("cogint");
    f->stat.gid = strdup("cogint");
    f->stat.muid = strdup("cogint");
    f->stat.mode = (qtype & COG9P_QTDIR) ? 0755 | 0x80000000 : 0644;
    
    return f;
}

static void file_add_child(Cog9PFile *parent, Cog9PFile *child) {
    parent->children = realloc(parent->children, 
                               (parent->n_children + 1) * sizeof(Cog9PFile*));
    parent->children[parent->n_children++] = child;
    child->parent = parent;
}

static Cog9PFile* file_find_child(Cog9PFile *parent, const char *name) {
    for (size_t i = 0; i < parent->n_children; i++) {
        if (strcmp(parent->children[i]->name, name) == 0) {
            return parent->children[i];
        }
    }
    return NULL;
}

static void file_free(Cog9PFile *f) {
    if (!f) return;
    for (size_t i = 0; i < f->n_children; i++) {
        file_free(f->children[i]);
    }
    free(f->children);
    free(f->name);
    free(f->data);
    free(f->stat.name);
    free(f->stat.uid);
    free(f->stat.gid);
    free(f->stat.muid);
    if (f->tensor) cog_tensor_free(f->tensor);
    free(f);
}

/*============================================================================
 * FID Management
 *============================================================================*/

static Cog9PFid* fid_get(Cog9PServer *srv, uint32_t fid) {
    for (size_t i = 0; i < srv->n_fids; i++) {
        if (srv->fids[i].fid == fid) {
            return &srv->fids[i];
        }
    }
    return NULL;
}

static Cog9PFid* fid_alloc(Cog9PServer *srv, uint32_t fid) {
    /* Check if exists */
    Cog9PFid *existing = fid_get(srv, fid);
    if (existing) return existing;
    
    /* Expand if needed */
    if (srv->n_fids >= srv->fid_capacity) {
        srv->fid_capacity = srv->fid_capacity ? srv->fid_capacity * 2 : 64;
        srv->fids = realloc(srv->fids, srv->fid_capacity * sizeof(Cog9PFid));
    }
    
    Cog9PFid *f = &srv->fids[srv->n_fids++];
    memset(f, 0, sizeof(Cog9PFid));
    f->fid = fid;
    return f;
}

static void fid_free(Cog9PServer *srv, uint32_t fid) {
    for (size_t i = 0; i < srv->n_fids; i++) {
        if (srv->fids[i].fid == fid) {
            memmove(&srv->fids[i], &srv->fids[i+1], 
                    (srv->n_fids - i - 1) * sizeof(Cog9PFid));
            srv->n_fids--;
            return;
        }
    }
}

/*============================================================================
 * Message Handlers
 *============================================================================*/

static int handle_version(Cog9PServer *srv, uint8_t *buf, size_t len, 
                          uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;  /* Skip header */
    uint32_t msize = get_u32(&p);
    char *version = get_str(&p);
    
    /* Negotiate */
    uint32_t agreed_msize = (msize < srv->msize) ? msize : srv->msize;
    const char *agreed_version = COG9P_VERSION;
    
    /* Check version compatibility */
    if (strncmp(version, "9P2000", 6) != 0) {
        agreed_version = "unknown";
    }
    
    free(version);
    
    /* Build response */
    uint8_t *r = resp;
    put_u32(&r, 0);  /* Size placeholder */
    put_u8(&r, COG9P_RVERSION);
    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));  /* Tag */
    put_u32(&r, agreed_msize);
    put_str(&r, agreed_version);
    
    *resp_len = r - resp;
    resp[0] = *resp_len & 0xff;
    resp[1] = (*resp_len >> 8) & 0xff;
    resp[2] = (*resp_len >> 16) & 0xff;
    resp[3] = (*resp_len >> 24) & 0xff;
    
    return 0;
}

static int handle_attach(Cog9PServer *srv, uint8_t *buf, size_t len,
                         uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;
    uint32_t fid = get_u32(&p);
    uint32_t afid = get_u32(&p);
    char *uname = get_str(&p);
    char *aname = get_str(&p);
    
    /* Allocate FID and attach to root */
    Cog9PFid *f = fid_alloc(srv, fid);
    f->file = srv->root;
    
    free(uname);
    free(aname);
    
    /* Build response */
    uint8_t *r = resp;
    put_u32(&r, 0);
    put_u8(&r, COG9P_RATTACH);
    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
    encode_qid(&r, &srv->root->qid);
    
    *resp_len = r - resp;
    resp[0] = *resp_len & 0xff;
    resp[1] = (*resp_len >> 8) & 0xff;
    resp[2] = (*resp_len >> 16) & 0xff;
    resp[3] = (*resp_len >> 24) & 0xff;
    
    return 0;
}

static int handle_walk(Cog9PServer *srv, uint8_t *buf, size_t len,
                       uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;
    uint32_t fid = get_u32(&p);
    uint32_t newfid = get_u32(&p);
    uint16_t nwname = get_u16(&p);
    
    Cog9PFid *f = fid_get(srv, fid);
    if (!f || !f->file) {
        /* Error response */
        uint8_t *r = resp;
        put_u32(&r, 0);
        put_u8(&r, COG9P_RERROR);
        put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
        put_str(&r, "unknown fid");
        *resp_len = r - resp;
        resp[0] = *resp_len & 0xff;
        resp[1] = (*resp_len >> 8) & 0xff;
        resp[2] = (*resp_len >> 16) & 0xff;
        resp[3] = (*resp_len >> 24) & 0xff;
        return -1;
    }
    
    Cog9PFile *current = f->file;
    Cog9PQid qids[16];  /* Max walk depth */
    uint16_t nwqid = 0;
    
    for (uint16_t i = 0; i < nwname && i < 16; i++) {
        char *name = get_str(&p);
        
        if (strcmp(name, "..") == 0) {
            if (current->parent) current = current->parent;
        } else if (strcmp(name, ".") == 0) {
            /* Stay in current */
        } else {
            Cog9PFile *child = file_find_child(current, name);
            if (!child) {
                free(name);
                break;
            }
            current = child;
        }
        
        qids[nwqid++] = current->qid;
        free(name);
    }
    
    /* Allocate newfid */
    Cog9PFid *nf = fid_alloc(srv, newfid);
    nf->file = current;
    
    /* Build response */
    uint8_t *r = resp;
    put_u32(&r, 0);
    put_u8(&r, COG9P_RWALK);
    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
    put_u16(&r, nwqid);
    for (uint16_t i = 0; i < nwqid; i++) {
        encode_qid(&r, &qids[i]);
    }
    
    *resp_len = r - resp;
    resp[0] = *resp_len & 0xff;
    resp[1] = (*resp_len >> 8) & 0xff;
    resp[2] = (*resp_len >> 16) & 0xff;
    resp[3] = (*resp_len >> 24) & 0xff;
    
    return 0;
}

static int handle_open(Cog9PServer *srv, uint8_t *buf, size_t len,
                       uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;
    uint32_t fid = get_u32(&p);
    uint8_t mode = get_u8(&p);
    
    Cog9PFid *f = fid_get(srv, fid);
    if (!f || !f->file) {
        uint8_t *r = resp;
        put_u32(&r, 0);
        put_u8(&r, COG9P_RERROR);
        put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
        put_str(&r, "unknown fid");
        *resp_len = r - resp;
        resp[0] = *resp_len & 0xff;
        resp[1] = (*resp_len >> 8) & 0xff;
        resp[2] = (*resp_len >> 16) & 0xff;
        resp[3] = (*resp_len >> 24) & 0xff;
        return -1;
    }
    
    f->mode = mode;
    f->open = 1;
    f->offset = 0;
    
    uint8_t *r = resp;
    put_u32(&r, 0);
    put_u8(&r, COG9P_ROPEN);
    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
    encode_qid(&r, &f->file->qid);
    put_u32(&r, srv->msize - 24);  /* iounit */
    
    *resp_len = r - resp;
    resp[0] = *resp_len & 0xff;
    resp[1] = (*resp_len >> 8) & 0xff;
    resp[2] = (*resp_len >> 16) & 0xff;
    resp[3] = (*resp_len >> 24) & 0xff;
    
    return 0;
}

static int handle_read(Cog9PServer *srv, uint8_t *buf, size_t len,
                       uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;
    uint32_t fid = get_u32(&p);
    uint64_t offset = get_u64(&p);
    uint32_t count = get_u32(&p);
    
    Cog9PFid *f = fid_get(srv, fid);
    if (!f || !f->file) {
        uint8_t *r = resp;
        put_u32(&r, 0);
        put_u8(&r, COG9P_RERROR);
        put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
        put_str(&r, "unknown fid");
        *resp_len = r - resp;
        resp[0] = *resp_len & 0xff;
        resp[1] = (*resp_len >> 8) & 0xff;
        resp[2] = (*resp_len >> 16) & 0xff;
        resp[3] = (*resp_len >> 24) & 0xff;
        return -1;
    }
    
    uint8_t *data = NULL;
    uint32_t nread = 0;
    
    if (f->file->qid.type & COG9P_QTDIR) {
        /* Read directory */
        uint8_t dirbuf[8192];
        uint8_t *dp = dirbuf;
        
        for (size_t i = 0; i < f->file->n_children; i++) {
            Cog9PFile *child = f->file->children[i];
            size_t ss = stat_size(&child->stat);
            if (dp - dirbuf + ss > sizeof(dirbuf)) break;
            encode_stat(&dp, &child->stat);
        }
        
        size_t total = dp - dirbuf;
        if (offset < total) {
            nread = (count < total - offset) ? count : total - offset;
            data = dirbuf + offset;
        }
    } else if (f->file->tensor) {
        /* Read tensor data */
        CogTensor *t = f->file->tensor;
        size_t total = t->numel * sizeof(float);  /* Assuming float32 */
        if (offset < total) {
            nread = (count < total - offset) ? count : total - offset;
            data = (uint8_t*)t->data + offset;
        }
    } else if (f->file->data) {
        /* Read regular file data */
        if (offset < f->file->data_len) {
            nread = (count < f->file->data_len - offset) ? 
                    count : f->file->data_len - offset;
            data = f->file->data + offset;
        }
    }
    
    uint8_t *r = resp;
    put_u32(&r, 0);
    put_u8(&r, COG9P_RREAD);
    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
    put_u32(&r, nread);
    if (nread > 0 && data) {
        memcpy(r, data, nread);
        r += nread;
    }
    
    *resp_len = r - resp;
    resp[0] = *resp_len & 0xff;
    resp[1] = (*resp_len >> 8) & 0xff;
    resp[2] = (*resp_len >> 16) & 0xff;
    resp[3] = (*resp_len >> 24) & 0xff;
    
    return 0;
}

static int handle_clunk(Cog9PServer *srv, uint8_t *buf, size_t len,
                        uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;
    uint32_t fid = get_u32(&p);
    
    fid_free(srv, fid);
    
    uint8_t *r = resp;
    put_u32(&r, 0);
    put_u8(&r, COG9P_RCLUNK);
    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
    
    *resp_len = r - resp;
    resp[0] = *resp_len & 0xff;
    resp[1] = (*resp_len >> 8) & 0xff;
    resp[2] = (*resp_len >> 16) & 0xff;
    resp[3] = (*resp_len >> 24) & 0xff;
    
    return 0;
}

static int handle_stat(Cog9PServer *srv, uint8_t *buf, size_t len,
                       uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;
    uint32_t fid = get_u32(&p);
    
    Cog9PFid *f = fid_get(srv, fid);
    if (!f || !f->file) {
        uint8_t *r = resp;
        put_u32(&r, 0);
        put_u8(&r, COG9P_RERROR);
        put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
        put_str(&r, "unknown fid");
        *resp_len = r - resp;
        resp[0] = *resp_len & 0xff;
        resp[1] = (*resp_len >> 8) & 0xff;
        resp[2] = (*resp_len >> 16) & 0xff;
        resp[3] = (*resp_len >> 24) & 0xff;
        return -1;
    }
    
    /* Update length for tensors */
    if (f->file->tensor) {
        f->file->stat.length = f->file->tensor->numel * sizeof(float);
    }
    
    uint8_t *r = resp;
    put_u32(&r, 0);
    put_u8(&r, COG9P_RSTAT);
    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
    put_u16(&r, stat_size(&f->file->stat));
    encode_stat(&r, &f->file->stat);
    
    *resp_len = r - resp;
    resp[0] = *resp_len & 0xff;
    resp[1] = (*resp_len >> 8) & 0xff;
    resp[2] = (*resp_len >> 16) & 0xff;
    resp[3] = (*resp_len >> 24) & 0xff;
    
    return 0;
}

/*============================================================================
 * Tensor Extension Handlers
 *============================================================================*/

static int handle_tensor(Cog9PServer *srv, uint8_t *buf, size_t len,
                         uint8_t *resp, size_t *resp_len) {
    uint8_t *p = buf + 7;
    uint32_t fid = get_u32(&p);
    uint8_t op = get_u8(&p);
    
    switch (op) {
    case COG9P_TOP_CREATE: {
        CogDType dtype = get_u8(&p);
        int ndim = get_u8(&p);
        int64_t shape[8];
        for (int i = 0; i < ndim && i < 8; i++) {
            shape[i] = get_u64(&p);
        }
        
        /* Create tensor */
        CogTensor *t = cog_tensor_create(srv->cogctx, shape, ndim, dtype);
        if (!t) {
            uint8_t *r = resp;
            put_u32(&r, 0);
            put_u8(&r, COG9P_RERROR);
            put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
            put_str(&r, "tensor creation failed");
            *resp_len = r - resp;
            resp[0] = *resp_len & 0xff;
            resp[1] = (*resp_len >> 8) & 0xff;
            resp[2] = (*resp_len >> 16) & 0xff;
            resp[3] = (*resp_len >> 24) & 0xff;
            return -1;
        }
        
        /* Create file for tensor */
        char name[64];
        snprintf(name, sizeof(name), "tensor_%lu", srv->next_path);
        Cog9PFile *tf = file_create(srv, name, COG9P_QTTENSOR);
        tf->tensor = t;
        file_add_child(srv->root, tf);
        
        /* Allocate FID */
        Cog9PFid *f = fid_alloc(srv, fid);
        f->file = tf;
        
        /* Response */
        uint8_t *r = resp;
        put_u32(&r, 0);
        put_u8(&r, COG9P_RTENSOR);
        put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
        put_u32(&r, fid);
        encode_qid(&r, &tf->qid);
        put_u8(&r, dtype);
        put_u8(&r, ndim);
        for (int i = 0; i < ndim; i++) {
            put_u64(&r, shape[i]);
        }
        put_u32(&r, 0);  /* No inline data */
        
        *resp_len = r - resp;
        resp[0] = *resp_len & 0xff;
        resp[1] = (*resp_len >> 8) & 0xff;
        resp[2] = (*resp_len >> 16) & 0xff;
        resp[3] = (*resp_len >> 24) & 0xff;
        break;
    }
    
    case COG9P_TOP_ADD:
    case COG9P_TOP_MUL:
    case COG9P_TOP_MATMUL: {
        uint32_t fid2 = get_u32(&p);
        
        Cog9PFid *f1 = fid_get(srv, fid);
        Cog9PFid *f2 = fid_get(srv, fid2);
        
        if (!f1 || !f1->file || !f1->file->tensor ||
            !f2 || !f2->file || !f2->file->tensor) {
            uint8_t *r = resp;
            put_u32(&r, 0);
            put_u8(&r, COG9P_RERROR);
            put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
            put_str(&r, "invalid tensor fid");
            *resp_len = r - resp;
            resp[0] = *resp_len & 0xff;
            resp[1] = (*resp_len >> 8) & 0xff;
            resp[2] = (*resp_len >> 16) & 0xff;
            resp[3] = (*resp_len >> 24) & 0xff;
            return -1;
        }
        
        CogTensor *result = NULL;
        switch (op) {
        case COG9P_TOP_ADD:
            result = cog_tensor_add(f1->file->tensor, f2->file->tensor);
            break;
        case COG9P_TOP_MUL:
            result = cog_tensor_mul(f1->file->tensor, f2->file->tensor);
            break;
        case COG9P_TOP_MATMUL:
            result = cog_tensor_matmul(f1->file->tensor, f2->file->tensor);
            break;
        }
        
        if (!result) {
            uint8_t *r = resp;
            put_u32(&r, 0);
            put_u8(&r, COG9P_RERROR);
            put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
            put_str(&r, "tensor operation failed");
            *resp_len = r - resp;
            resp[0] = *resp_len & 0xff;
            resp[1] = (*resp_len >> 8) & 0xff;
            resp[2] = (*resp_len >> 16) & 0xff;
            resp[3] = (*resp_len >> 24) & 0xff;
            return -1;
        }
        
        /* Create file for result */
        char name[64];
        snprintf(name, sizeof(name), "tensor_%lu", srv->next_path);
        Cog9PFile *tf = file_create(srv, name, COG9P_QTTENSOR);
        tf->tensor = result;
        file_add_child(srv->root, tf);
        
        uint32_t result_fid = srv->next_path;
        Cog9PFid *rf = fid_alloc(srv, result_fid);
        rf->file = tf;
        
        /* Response */
        uint8_t *r = resp;
        put_u32(&r, 0);
        put_u8(&r, COG9P_RTENSOR);
        put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
        put_u32(&r, result_fid);
        encode_qid(&r, &tf->qid);
        put_u8(&r, result->dtype);
        put_u8(&r, result->ndim);
        for (int i = 0; i < result->ndim; i++) {
            put_u64(&r, result->shape[i]);
        }
        put_u32(&r, 0);
        
        *resp_len = r - resp;
        resp[0] = *resp_len & 0xff;
        resp[1] = (*resp_len >> 8) & 0xff;
        resp[2] = (*resp_len >> 16) & 0xff;
        resp[3] = (*resp_len >> 24) & 0xff;
        break;
    }
    
    default: {
        uint8_t *r = resp;
        put_u32(&r, 0);
        put_u8(&r, COG9P_RERROR);
        put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
        put_str(&r, "unknown tensor operation");
        *resp_len = r - resp;
        resp[0] = *resp_len & 0xff;
        resp[1] = (*resp_len >> 8) & 0xff;
        resp[2] = (*resp_len >> 16) & 0xff;
        resp[3] = (*resp_len >> 24) & 0xff;
        return -1;
    }
    }
    
    return 0;
}

/*============================================================================
 * Server Implementation
 *============================================================================*/

static void* server_accept_loop(void *arg) {
    Cog9PServer *srv = (Cog9PServer*)arg;
    uint8_t buf[COG9P_MSIZE];
    uint8_t resp[COG9P_MSIZE];
    
    while (srv->running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        
        int client = accept(srv->sock, (struct sockaddr*)&client_addr, &addr_len);
        if (client < 0) {
            if (srv->running) perror("accept");
            continue;
        }
        
        /* Handle client */
        while (srv->running) {
            /* Read message header */
            ssize_t n = read(client, buf, 4);
            if (n <= 0) break;
            
            uint32_t size = buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24);
            if (size > COG9P_MSIZE) {
                fprintf(stderr, "Message too large: %u\n", size);
                break;
            }
            
            /* Read rest of message */
            n = read(client, buf + 4, size - 4);
            if (n <= 0) break;
            
            uint8_t type = buf[4];
            size_t resp_len = 0;
            
            pthread_mutex_lock(&srv->lock);
            
            switch (type) {
            case COG9P_TVERSION:
                handle_version(srv, buf, size, resp, &resp_len);
                break;
            case COG9P_TATTACH:
                handle_attach(srv, buf, size, resp, &resp_len);
                break;
            case COG9P_TWALK:
                handle_walk(srv, buf, size, resp, &resp_len);
                break;
            case COG9P_TOPEN:
                handle_open(srv, buf, size, resp, &resp_len);
                break;
            case COG9P_TREAD:
                handle_read(srv, buf, size, resp, &resp_len);
                break;
            case COG9P_TCLUNK:
                handle_clunk(srv, buf, size, resp, &resp_len);
                break;
            case COG9P_TSTAT:
                handle_stat(srv, buf, size, resp, &resp_len);
                break;
            case COG9P_TTENSOR:
                handle_tensor(srv, buf, size, resp, &resp_len);
                break;
            default:
                /* Unknown message - send error */
                {
                    uint8_t *r = resp;
                    put_u32(&r, 0);
                    put_u8(&r, COG9P_RERROR);
                    put_u16(&r, get_u16(&(uint8_t*){buf + 5}));
                    put_str(&r, "unknown message type");
                    resp_len = r - resp;
                    resp[0] = resp_len & 0xff;
                    resp[1] = (resp_len >> 8) & 0xff;
                    resp[2] = (resp_len >> 16) & 0xff;
                    resp[3] = (resp_len >> 24) & 0xff;
                }
                break;
            }
            
            pthread_mutex_unlock(&srv->lock);
            
            /* Send response */
            if (resp_len > 0) {
                write(client, resp, resp_len);
            }
        }
        
        close(client);
    }
    
    return NULL;
}

/*============================================================================
 * Public API
 *============================================================================*/

COGINT_API Cog9PServer* cog9p_server_create(Cog9PServerConfig *config) {
    Cog9PServer *srv = calloc(1, sizeof(Cog9PServer));
    if (!srv) return NULL;
    
    srv->addr = strdup(config->addr ? config->addr : "0.0.0.0");
    srv->port = config->port ? config->port : 564;
    srv->msize = config->msize ? config->msize : COG9P_MSIZE;
    srv->cogctx = config->cogctx;
    srv->atomspace = config->atomspace;
    
    /* Copy handlers */
    srv->on_attach = config->on_attach;
    srv->on_walk = config->on_walk;
    srv->on_read = config->on_read;
    srv->on_write = config->on_write;
    srv->on_tensor = config->on_tensor;
    srv->on_atom = config->on_atom;
    srv->on_cog = config->on_cog;
    
    /* Create root directory */
    srv->root = file_create(srv, "/", COG9P_QTDIR);
    
    /* Create standard directories */
    Cog9PFile *tensors = file_create(srv, "tensors", COG9P_QTDIR);
    Cog9PFile *atoms = file_create(srv, "atoms", COG9P_QTDIR);
    Cog9PFile *ctl = file_create(srv, "ctl", 0);
    
    file_add_child(srv->root, tensors);
    file_add_child(srv->root, atoms);
    file_add_child(srv->root, ctl);
    
    pthread_mutex_init(&srv->lock, NULL);
    
    return srv;
}

COGINT_API int cog9p_server_start(Cog9PServer *srv) {
    /* Create socket */
    srv->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->sock < 0) {
        perror("socket");
        return -1;
    }
    
    int opt = 1;
    setsockopt(srv->sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(srv->port);
    inet_pton(AF_INET, srv->addr, &addr.sin_addr);
    
    if (bind(srv->sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(srv->sock);
        return -1;
    }
    
    if (listen(srv->sock, 10) < 0) {
        perror("listen");
        close(srv->sock);
        return -1;
    }
    
    srv->running = 1;
    pthread_create(&srv->accept_thread, NULL, server_accept_loop, srv);
    
    printf("CogInt 9P server listening on %s:%d\n", srv->addr, srv->port);
    return 0;
}

COGINT_API int cog9p_server_stop(Cog9PServer *srv) {
    srv->running = 0;
    close(srv->sock);
    pthread_join(srv->accept_thread, NULL);
    return 0;
}

COGINT_API void cog9p_server_free(Cog9PServer *srv) {
    if (!srv) return;
    
    file_free(srv->root);
    free(srv->fids);
    free(srv->addr);
    pthread_mutex_destroy(&srv->lock);
    free(srv);
}

/*============================================================================
 * Client Implementation
 *============================================================================*/

COGINT_API Cog9PConn* cog9p_dial(const char *addr, uint16_t port) {
    Cog9PConn *conn = calloc(1, sizeof(Cog9PConn));
    if (!conn) return NULL;
    
    conn->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (conn->fd < 0) {
        free(conn);
        return NULL;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, addr, &server_addr.sin_addr);
    
    if (connect(conn->fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(conn->fd);
        free(conn);
        return NULL;
    }
    
    conn->addr = strdup(addr);
    conn->port = port;
    conn->msize = COG9P_MSIZE;
    conn->connected = 1;
    conn->next_tag = 1;
    conn->next_fid = 1;
    
    return conn;
}

COGINT_API int cog9p_version(Cog9PConn *conn) {
    uint8_t buf[256];
    uint8_t *p = buf;
    
    put_u32(&p, 0);  /* Size placeholder */
    put_u8(&p, COG9P_TVERSION);
    put_u16(&p, 0xFFFF);  /* NOTAG */
    put_u32(&p, conn->msize);
    put_str(&p, COG9P_VERSION);
    
    size_t len = p - buf;
    buf[0] = len & 0xff;
    buf[1] = (len >> 8) & 0xff;
    buf[2] = (len >> 16) & 0xff;
    buf[3] = (len >> 24) & 0xff;
    
    write(conn->fd, buf, len);
    
    /* Read response */
    uint8_t resp[256];
    read(conn->fd, resp, 4);
    uint32_t rsize = resp[0] | (resp[1] << 8) | (resp[2] << 16) | (resp[3] << 24);
    read(conn->fd, resp + 4, rsize - 4);
    
    if (resp[4] == COG9P_RVERSION) {
        p = resp + 7;
        conn->msize = get_u32(&p);
        conn->version = get_str(&p);
        return 0;
    }
    
    return -1;
}

COGINT_API int cog9p_attach(Cog9PConn *conn, const char *uname, const char *aname) {
    uint8_t buf[256];
    uint8_t *p = buf;
    
    put_u32(&p, 0);
    put_u8(&p, COG9P_TATTACH);
    put_u16(&p, conn->next_tag++);
    put_u32(&p, 0);  /* root fid */
    put_u32(&p, 0xFFFFFFFF);  /* NOFID */
    put_str(&p, uname ? uname : "cogint");
    put_str(&p, aname ? aname : "");
    
    size_t len = p - buf;
    buf[0] = len & 0xff;
    buf[1] = (len >> 8) & 0xff;
    buf[2] = (len >> 16) & 0xff;
    buf[3] = (len >> 24) & 0xff;
    
    write(conn->fd, buf, len);
    
    uint8_t resp[256];
    read(conn->fd, resp, 4);
    uint32_t rsize = resp[0] | (resp[1] << 8) | (resp[2] << 16) | (resp[3] << 24);
    read(conn->fd, resp + 4, rsize - 4);
    
    return (resp[4] == COG9P_RATTACH) ? 0 : -1;
}

COGINT_API void cog_9p_disconnect(Cog9PConn *conn) {
    if (!conn) return;
    if (conn->fd >= 0) close(conn->fd);
    free(conn->addr);
    free(conn->version);
    free(conn->uname);
    free(conn->aname);
    free(conn);
}
