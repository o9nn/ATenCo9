/**
 * CogAtomSpace - OpenCog AtomSpace Implementation with Tensor Support
 * 
 * This file implements the AtomSpace hypergraph with integrated tensor
 * computing capabilities for neural-symbolic AI.
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
#include "../include/cog_atomspace.h"

/*============================================================================
 * Internal Structures
 *============================================================================*/

/* Hash table entry for name lookup */
typedef struct HashEntry {
    char *key;
    CogAtom *value;
    struct HashEntry *next;
} HashEntry;

/* Hash table for atom lookup */
typedef struct {
    HashEntry **buckets;
    size_t n_buckets;
    size_t n_entries;
} HashTable;

/* Internal AtomSpace implementation */
typedef struct {
    /* Atom storage */
    CogAtom **atoms;
    size_t n_atoms;
    size_t capacity;
    
    /* Indices */
    HashTable *name_index;
    CogAtomIndex *type_index;
    
    /* ECAN state */
    CogECANConfig ecan_config;
    int ecan_initialized;
    double sti_pool;
    double lti_pool;
    
    /* Threading */
    pthread_mutex_t lock;
    
    /* 9P export */
    void *server;
} AtomSpaceImpl;

/*============================================================================
 * Hash Table Implementation
 *============================================================================*/

static size_t hash_string(const char *str) {
    size_t hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

static HashTable* hashtable_create(size_t n_buckets) {
    HashTable *ht = calloc(1, sizeof(HashTable));
    ht->n_buckets = n_buckets;
    ht->buckets = calloc(n_buckets, sizeof(HashEntry*));
    return ht;
}

static void hashtable_put(HashTable *ht, const char *key, CogAtom *value) {
    size_t idx = hash_string(key) % ht->n_buckets;
    
    HashEntry *entry = malloc(sizeof(HashEntry));
    entry->key = strdup(key);
    entry->value = value;
    entry->next = ht->buckets[idx];
    ht->buckets[idx] = entry;
    ht->n_entries++;
}

static CogAtom* hashtable_get(HashTable *ht, const char *key) {
    size_t idx = hash_string(key) % ht->n_buckets;
    
    for (HashEntry *e = ht->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0)
            return e->value;
    }
    return NULL;
}

static void hashtable_free(HashTable *ht) {
    for (size_t i = 0; i < ht->n_buckets; i++) {
        HashEntry *e = ht->buckets[i];
        while (e) {
            HashEntry *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(ht->buckets);
    free(ht);
}

/*============================================================================
 * AtomSpace Core Implementation
 *============================================================================*/

COGINT_API CogAtomSpace* cog_atomspace_create(CogContext *ctx, const char *name) {
    CogAtomSpace *as = calloc(1, sizeof(CogAtomSpace));
    if (!as) return NULL;
    
    AtomSpaceImpl *impl = calloc(1, sizeof(AtomSpaceImpl));
    if (!impl) {
        free(as);
        return NULL;
    }
    
    as->impl = impl;
    as->name = strdup(name ? name : "default");
    
    /* Initialize storage */
    impl->capacity = 1024;
    impl->atoms = calloc(impl->capacity, sizeof(CogAtom*));
    
    /* Initialize indices */
    impl->name_index = hashtable_create(4096);
    impl->type_index = calloc(1, sizeof(CogAtomIndex));
    impl->type_index->n_types = 256;
    impl->type_index->by_type = calloc(256, sizeof(CogAtom**));
    impl->type_index->type_counts = calloc(256, sizeof(size_t));
    
    pthread_mutex_init(&impl->lock, NULL);
    
    return as;
}

COGINT_API void cog_atomspace_free(CogAtomSpace *as) {
    if (!as) return;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    
    /* Free atoms */
    for (size_t i = 0; i < impl->n_atoms; i++) {
        CogAtom *atom = impl->atoms[i];
        if (atom) {
            free(atom->name);
            free(atom->outgoing);
            if (atom->tensor) cog_tensor_free(atom->tensor);
            free(atom);
        }
    }
    free(impl->atoms);
    
    /* Free indices */
    hashtable_free(impl->name_index);
    for (size_t i = 0; i < impl->type_index->n_types; i++) {
        free(impl->type_index->by_type[i]);
    }
    free(impl->type_index->by_type);
    free(impl->type_index->type_counts);
    free(impl->type_index->tensor_atoms);
    free(impl->type_index->attention_focus);
    free(impl->type_index);
    
    pthread_mutex_destroy(&impl->lock);
    
    free(as->name);
    free(as->export_path);
    free(impl);
    free(as);
}

/* Add atom to AtomSpace */
static void atomspace_add(CogAtomSpace *as, CogAtom *atom) {
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    
    pthread_mutex_lock(&impl->lock);
    
    /* Expand if needed */
    if (impl->n_atoms >= impl->capacity) {
        impl->capacity *= 2;
        impl->atoms = realloc(impl->atoms, impl->capacity * sizeof(CogAtom*));
    }
    
    /* Add to main storage */
    atom->handle = impl->n_atoms;
    impl->atoms[impl->n_atoms++] = atom;
    as->atom_count++;
    
    /* Add to name index */
    if (atom->name) {
        hashtable_put(impl->name_index, atom->name, atom);
    }
    
    /* Add to type index */
    CogAtomIndex *idx = impl->type_index;
    size_t type = atom->type;
    if (type < idx->n_types) {
        size_t count = idx->type_counts[type];
        idx->by_type[type] = realloc(idx->by_type[type], 
                                      (count + 1) * sizeof(CogAtom*));
        idx->by_type[type][count] = atom;
        idx->type_counts[type]++;
    }
    
    /* Track tensor atoms */
    if (atom->tensor) {
        idx->tensor_atoms = realloc(idx->tensor_atoms,
                                    (idx->n_tensor_atoms + 1) * sizeof(CogAtom*));
        idx->tensor_atoms[idx->n_tensor_atoms++] = atom;
        as->tensor_atom_count++;
    }
    
    pthread_mutex_unlock(&impl->lock);
}

COGINT_API CogAtom* cog_atom_create(CogAtomSpace *as, CogAtomType type, 
                                     const char *name) {
    CogAtom *atom = calloc(1, sizeof(CogAtom));
    if (!atom) return NULL;
    
    atom->type = type;
    atom->name = name ? strdup(name) : NULL;
    atom->refcount = 1;
    
    /* Default truth value */
    atom->strength = 1.0;
    atom->confidence = 0.0;
    
    /* Default attention value */
    atom->sti = 0;
    atom->lti = 0;
    atom->vlti = 0;
    
    atomspace_add(as, atom);
    
    return atom;
}

COGINT_API int cog_atom_set_tv(CogAtom *atom, double strength, double confidence) {
    if (!atom) return COG_ERR_INVALID;
    atom->strength = strength;
    atom->confidence = confidence;
    return COG_OK;
}

COGINT_API int cog_atom_set_av(CogAtom *atom, int16_t sti, int16_t lti, int16_t vlti) {
    if (!atom) return COG_ERR_INVALID;
    atom->sti = sti;
    atom->lti = lti;
    atom->vlti = vlti;
    return COG_OK;
}

/*============================================================================
 * Tensor-Atom Binding Implementation
 *============================================================================*/

COGINT_API CogAtom* cog_tensor_node_create(CogAtomSpace *as, CogTensor *tensor,
                                            const char *name) {
    if (!as || !tensor) return NULL;
    
    CogAtom *atom = calloc(1, sizeof(CogAtom));
    if (!atom) return NULL;
    
    atom->type = COG_ATOM_TENSOR_NODE;
    atom->name = name ? strdup(name) : NULL;
    atom->tensor = cog_tensor_clone(tensor);
    atom->refcount = 1;
    
    /* Set truth value based on tensor properties */
    atom->strength = 1.0;
    atom->confidence = 1.0;  /* Tensors are concrete data */
    
    atomspace_add(as, atom);
    
    return atom;
}

COGINT_API CogTensor* cog_tensor_node_get(CogAtom *atom) {
    if (!atom || atom->type != COG_ATOM_TENSOR_NODE)
        return NULL;
    return atom->tensor;
}

COGINT_API CogAtom* cog_tensor_to_atom(CogAtomSpace *as, CogTensor *tensor,
                                        const char *name) {
    return cog_tensor_node_create(as, tensor, name);
}

COGINT_API CogTensor* cog_atom_to_tensor(CogAtom *atom) {
    return cog_tensor_node_get(atom);
}

COGINT_API CogAtom* cog_tensor_op_link_create(CogAtomSpace *as,
                                               CogTensorAtomType op_type,
                                               CogAtom **inputs, size_t n_inputs,
                                               CogAtom *output) {
    if (!as || !inputs || n_inputs == 0) return NULL;
    
    CogAtom *link = calloc(1, sizeof(CogAtom));
    if (!link) return NULL;
    
    link->type = op_type;
    link->refcount = 1;
    
    /* Build outgoing set: [output, input1, input2, ...] */
    link->arity = n_inputs + (output ? 1 : 0);
    link->outgoing = calloc(link->arity, sizeof(CogAtom*));
    
    size_t idx = 0;
    if (output) {
        link->outgoing[idx++] = output;
    }
    for (size_t i = 0; i < n_inputs; i++) {
        link->outgoing[idx++] = inputs[i];
    }
    
    /* Generate name */
    char name[256];
    const char *op_name = "op";
    switch (op_type) {
        case COG_ATOM_TENSOR_ADD_LINK: op_name = "add"; break;
        case COG_ATOM_TENSOR_MUL_LINK: op_name = "mul"; break;
        case COG_ATOM_TENSOR_MATMUL_LINK: op_name = "matmul"; break;
        case COG_ATOM_TENSOR_CONV_LINK: op_name = "conv"; break;
        case COG_ATOM_TENSOR_SOFTMAX_LINK: op_name = "softmax"; break;
        case COG_ATOM_TENSOR_RELU_LINK: op_name = "relu"; break;
        default: break;
    }
    snprintf(name, sizeof(name), "TensorOp_%s_%zu", op_name, as->atom_count);
    link->name = strdup(name);
    
    atomspace_add(as, link);
    
    return link;
}

COGINT_API CogTensor* cog_tensor_op_execute(CogAtom *link) {
    if (!link || link->arity < 1) return NULL;
    
    /* Get input tensors */
    CogTensor *inputs[8];
    size_t n_inputs = 0;
    
    for (size_t i = 1; i < link->arity && n_inputs < 8; i++) {
        CogAtom *input = link->outgoing[i];
        if (input && input->tensor) {
            inputs[n_inputs++] = input->tensor;
        }
    }
    
    if (n_inputs == 0) return NULL;
    
    CogTensor *result = NULL;
    
    switch (link->type) {
        case COG_ATOM_TENSOR_ADD_LINK:
            if (n_inputs >= 2)
                result = cog_tensor_add(inputs[0], inputs[1]);
            break;
            
        case COG_ATOM_TENSOR_MUL_LINK:
            if (n_inputs >= 2)
                result = cog_tensor_mul(inputs[0], inputs[1]);
            break;
            
        case COG_ATOM_TENSOR_MATMUL_LINK:
            if (n_inputs >= 2)
                result = cog_tensor_matmul(inputs[0], inputs[1]);
            break;
            
        case COG_ATOM_TENSOR_SOFTMAX_LINK:
            if (n_inputs >= 1)
                result = cog_tensor_softmax(inputs[0], -1);
            break;
            
        default:
            break;
    }
    
    /* Store result in output atom if present */
    if (result && link->outgoing[0]) {
        CogAtom *output = link->outgoing[0];
        if (output->tensor) {
            cog_tensor_free(output->tensor);
        }
        output->tensor = result;
    }
    
    return result;
}

COGINT_API CogAtom* cog_tensor_annotate(CogAtomSpace *as, CogAtom *tensor_atom,
                                         CogAtom *concept, CogTruthValue *tv) {
    if (!as || !tensor_atom || !concept) return NULL;
    
    CogAtom *link = calloc(1, sizeof(CogAtom));
    if (!link) return NULL;
    
    link->type = COG_ATOM_TENSOR_MEANING_LINK;
    link->arity = 2;
    link->outgoing = calloc(2, sizeof(CogAtom*));
    link->outgoing[0] = tensor_atom;
    link->outgoing[1] = concept;
    link->refcount = 1;
    
    if (tv) {
        link->strength = tv->strength;
        link->confidence = tv->confidence;
    }
    
    char name[256];
    snprintf(name, sizeof(name), "Meaning_%s_%s",
             tensor_atom->name ? tensor_atom->name : "tensor",
             concept->name ? concept->name : "concept");
    link->name = strdup(name);
    
    atomspace_add(as, link);
    
    return link;
}

/*============================================================================
 * PLN Implementation
 *============================================================================*/

COGINT_API double cog_pln_tensor_similarity(CogTensor *t1, CogTensor *t2) {
    if (!t1 || !t2) return 0.0;
    if (t1->numel != t2->numel) return 0.0;
    
    /* Cosine similarity */
    float *d1 = (float*)t1->data;
    float *d2 = (float*)t2->data;
    
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < t1->numel; i++) {
        dot += d1[i] * d2[i];
        norm1 += d1[i] * d1[i];
        norm2 += d2[i] * d2[i];
    }
    
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    
    return dot / (sqrt(norm1) * sqrt(norm2));
}

/* PLN deduction formula */
static double pln_deduction_strength(double sAB, double sBC, double sB, double sC) {
    if (sB == 0.0) return sC;
    return sAB * sBC + (1.0 - sAB) * (sC - sB * sBC) / (1.0 - sB);
}

/* PLN confidence formula */
static double pln_deduction_confidence(double cAB, double cBC, double cB, double cC,
                                        double k) {
    return cAB * cBC * cB * cC * k;
}

COGINT_API int cog_pln_inference(CogAtomSpace *as, CogAtom *query,
                                  CogPLNConfig *config,
                                  CogAtom ***results, size_t *n_results) {
    if (!as || !query || !results || !n_results) return COG_ERR_INVALID;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    
    /* Simple forward chaining implementation */
    CogAtom **found = NULL;
    size_t found_count = 0;
    size_t found_capacity = 0;
    
    pthread_mutex_lock(&impl->lock);
    
    /* Search for matching atoms */
    for (size_t i = 0; i < impl->n_atoms; i++) {
        CogAtom *atom = impl->atoms[i];
        if (!atom) continue;
        
        /* Check type match */
        if (query->type != 0 && atom->type != query->type)
            continue;
        
        /* Check confidence threshold */
        if (atom->confidence < config->confidence_threshold)
            continue;
        
        /* Check tensor similarity if enabled */
        if (config->use_tensor_similarity && query->tensor && atom->tensor) {
            double sim = cog_pln_tensor_similarity(query->tensor, atom->tensor);
            if (sim < config->tensor_similarity_threshold)
                continue;
        }
        
        /* Add to results */
        if (found_count >= found_capacity) {
            found_capacity = found_capacity ? found_capacity * 2 : 16;
            found = realloc(found, found_capacity * sizeof(CogAtom*));
        }
        found[found_count++] = atom;
        
        if (config->max_results > 0 && found_count >= (size_t)config->max_results)
            break;
    }
    
    pthread_mutex_unlock(&impl->lock);
    
    *results = found;
    *n_results = found_count;
    
    return COG_OK;
}

COGINT_API CogAtom* cog_pln_inheritance_with_tensor(CogAtomSpace *as,
                                                     CogAtom *child,
                                                     CogAtom *parent,
                                                     CogTensor *evidence) {
    if (!as || !child || !parent) return NULL;
    
    CogAtom *link = calloc(1, sizeof(CogAtom));
    if (!link) return NULL;
    
    link->type = COG_ATOM_INHERITANCE;
    link->arity = 2;
    link->outgoing = calloc(2, sizeof(CogAtom*));
    link->outgoing[0] = child;
    link->outgoing[1] = parent;
    link->refcount = 1;
    
    /* Compute truth value from tensor evidence */
    if (evidence) {
        link->tensor = cog_tensor_clone(evidence);
        
        /* Use tensor norm as strength indicator */
        float *data = (float*)evidence->data;
        double sum = 0.0;
        for (size_t i = 0; i < evidence->numel; i++) {
            sum += fabs(data[i]);
        }
        link->strength = tanh(sum / evidence->numel);  /* Normalize to [0,1] */
        link->confidence = 0.9;  /* High confidence for tensor evidence */
    } else {
        link->strength = 0.5;
        link->confidence = 0.1;
    }
    
    char name[256];
    snprintf(name, sizeof(name), "Inheritance_%s_%s",
             child->name ? child->name : "child",
             parent->name ? parent->name : "parent");
    link->name = strdup(name);
    
    atomspace_add(as, link);
    
    return link;
}

/*============================================================================
 * ECAN Implementation
 *============================================================================*/

COGINT_API int cog_ecan_init(CogAtomSpace *as, CogECANConfig *config) {
    if (!as || !config) return COG_ERR_INVALID;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    
    impl->ecan_config = *config;
    impl->sti_pool = config->sti_funds;
    impl->lti_pool = config->lti_funds;
    impl->ecan_initialized = 1;
    
    return COG_OK;
}

COGINT_API int cog_ecan_spread_attention(CogAtomSpace *as, CogAtom *source,
                                          double amount) {
    if (!as || !source) return COG_ERR_INVALID;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    if (!impl->ecan_initialized) return COG_ERR_INVALID;
    
    pthread_mutex_lock(&impl->lock);
    
    /* Spread attention to connected atoms */
    if (source->outgoing) {
        double spread_amount = amount / source->arity;
        for (size_t i = 0; i < source->arity; i++) {
            CogAtom *target = source->outgoing[i];
            if (target) {
                int16_t new_sti = target->sti + (int16_t)(spread_amount * 100);
                if (new_sti > 32767) new_sti = 32767;
                if (new_sti < -32768) new_sti = -32768;
                target->sti = new_sti;
            }
        }
    }
    
    /* Decay source attention */
    source->sti = (int16_t)(source->sti * (1.0 - impl->ecan_config.sti_rent));
    
    pthread_mutex_unlock(&impl->lock);
    
    return COG_OK;
}

COGINT_API int cog_ecan_get_focus(CogAtomSpace *as, CogAtom ***atoms,
                                   size_t *n_atoms) {
    if (!as || !atoms || !n_atoms) return COG_ERR_INVALID;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    if (!impl->ecan_initialized) return COG_ERR_INVALID;
    
    pthread_mutex_lock(&impl->lock);
    
    /* Collect atoms above AF boundary */
    CogAtom **focus = NULL;
    size_t focus_count = 0;
    size_t focus_capacity = 0;
    
    int16_t boundary = impl->ecan_config.af_boundary;
    
    for (size_t i = 0; i < impl->n_atoms; i++) {
        CogAtom *atom = impl->atoms[i];
        if (atom && atom->sti >= boundary) {
            if (focus_count >= focus_capacity) {
                focus_capacity = focus_capacity ? focus_capacity * 2 : 16;
                focus = realloc(focus, focus_capacity * sizeof(CogAtom*));
            }
            focus[focus_count++] = atom;
        }
    }
    
    pthread_mutex_unlock(&impl->lock);
    
    *atoms = focus;
    *n_atoms = focus_count;
    
    return COG_OK;
}

COGINT_API CogTensor* cog_ecan_tensor_attention(CogAtomSpace *as,
                                                 CogAtom *tensor_atom) {
    if (!as || !tensor_atom || !tensor_atom->tensor) return NULL;
    
    CogTensor *t = tensor_atom->tensor;
    
    /* Create attention weights based on STI */
    CogTensor *attention = cog_tensor_create(NULL, t->shape, t->ndim, COG_DTYPE_FLOAT32);
    if (!attention) return NULL;
    
    /* Scale attention by STI */
    double scale = (tensor_atom->sti + 32768.0) / 65536.0;  /* Normalize to [0,1] */
    
    float *data = (float*)attention->data;
    for (size_t i = 0; i < attention->numel; i++) {
        data[i] = (float)scale;
    }
    
    return attention;
}

COGINT_API CogTensor* cog_ecan_apply_attention(CogTensor *tensor,
                                                CogTensor *attention) {
    if (!tensor || !attention) return NULL;
    return cog_tensor_mul(tensor, attention);
}

/*============================================================================
 * Pattern Mining Implementation
 *============================================================================*/

COGINT_API int cog_pattern_mine(CogAtomSpace *as, CogPatternMinerConfig *config,
                                 CogMinedPattern **patterns, size_t *n_patterns) {
    if (!as || !config || !patterns || !n_patterns) return COG_ERR_INVALID;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    
    CogMinedPattern *found = NULL;
    size_t found_count = 0;
    
    pthread_mutex_lock(&impl->lock);
    
    /* Simple frequent pattern mining */
    /* Count atom type frequencies */
    size_t type_counts[256] = {0};
    for (size_t i = 0; i < impl->n_atoms; i++) {
        CogAtom *atom = impl->atoms[i];
        if (atom && atom->type < 256) {
            type_counts[atom->type]++;
        }
    }
    
    /* Find frequent types */
    for (size_t t = 0; t < 256; t++) {
        if (type_counts[t] >= (size_t)config->min_support) {
            found = realloc(found, (found_count + 1) * sizeof(CogMinedPattern));
            CogMinedPattern *p = &found[found_count++];
            memset(p, 0, sizeof(CogMinedPattern));
            
            p->support = type_counts[t];
            p->confidence = (double)type_counts[t] / impl->n_atoms;
            
            /* Create pattern atom */
            p->pattern = cog_atom_create(as, t, NULL);
            
            if (found_count >= (size_t)config->max_patterns)
                break;
        }
    }
    
    pthread_mutex_unlock(&impl->lock);
    
    *patterns = found;
    *n_patterns = found_count;
    
    return COG_OK;
}

COGINT_API int cog_pattern_find_tensor(CogAtomSpace *as, CogTensor *reference,
                                        double threshold,
                                        CogAtom ***matches, size_t *n_matches) {
    if (!as || !reference || !matches || !n_matches) return COG_ERR_INVALID;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    CogAtomIndex *idx = impl->type_index;
    
    CogAtom **found = NULL;
    size_t found_count = 0;
    size_t found_capacity = 0;
    
    pthread_mutex_lock(&impl->lock);
    
    /* Search tensor atoms */
    for (size_t i = 0; i < idx->n_tensor_atoms; i++) {
        CogAtom *atom = idx->tensor_atoms[i];
        if (!atom || !atom->tensor) continue;
        
        double sim = cog_pln_tensor_similarity(reference, atom->tensor);
        if (sim >= threshold) {
            if (found_count >= found_capacity) {
                found_capacity = found_capacity ? found_capacity * 2 : 16;
                found = realloc(found, found_capacity * sizeof(CogAtom*));
            }
            found[found_count++] = atom;
        }
    }
    
    pthread_mutex_unlock(&impl->lock);
    
    *matches = found;
    *n_matches = found_count;
    
    return COG_OK;
}

/*============================================================================
 * Neural-Symbolic Integration
 *============================================================================*/

COGINT_API CogTensor* cog_atomspace_to_tensor(CogAtomSpace *as, CogAtom *root,
                                               int depth) {
    if (!as || !root) return NULL;
    
    /* Create embedding tensor */
    int64_t shape[] = {128};  /* 128-dimensional embedding */
    CogTensor *embedding = cog_tensor_create(NULL, shape, 1, COG_DTYPE_FLOAT32);
    if (!embedding) return NULL;
    
    float *data = (float*)embedding->data;
    
    /* Encode atom type */
    data[0] = (float)root->type / 256.0f;
    
    /* Encode truth value */
    data[1] = (float)root->strength;
    data[2] = (float)root->confidence;
    
    /* Encode attention value */
    data[3] = (root->sti + 32768.0f) / 65536.0f;
    data[4] = (root->lti + 32768.0f) / 65536.0f;
    
    /* Encode name hash */
    if (root->name) {
        size_t hash = hash_string(root->name);
        for (int i = 0; i < 8; i++) {
            data[5 + i] = ((hash >> (i * 8)) & 0xFF) / 255.0f;
        }
    }
    
    /* Encode outgoing set structure */
    data[13] = (float)root->arity / 16.0f;
    
    /* Recursively encode children if depth > 0 */
    if (depth > 0 && root->outgoing) {
        for (size_t i = 0; i < root->arity && i < 4; i++) {
            CogTensor *child_emb = cog_atomspace_to_tensor(as, root->outgoing[i], 
                                                           depth - 1);
            if (child_emb) {
                float *child_data = (float*)child_emb->data;
                /* Aggregate child embedding */
                for (int j = 0; j < 16; j++) {
                    data[20 + i * 16 + j] = child_data[j];
                }
                cog_tensor_free(child_emb);
            }
        }
    }
    
    /* Include tensor data if present */
    if (root->tensor) {
        float *tensor_data = (float*)root->tensor->data;
        size_t n = root->tensor->numel < 32 ? root->tensor->numel : 32;
        for (size_t i = 0; i < n; i++) {
            data[96 + i] = tensor_data[i];
        }
    }
    
    return embedding;
}

COGINT_API CogTensor* cog_gnn_forward(CogAtomSpace *as, CogTensor **weights,
                                       int n_layers) {
    if (!as || !weights || n_layers <= 0) return NULL;
    
    AtomSpaceImpl *impl = (AtomSpaceImpl*)as->impl;
    
    /* Create node embeddings */
    int64_t shape[] = {(int64_t)impl->n_atoms, 128};
    CogTensor *embeddings = cog_tensor_create(NULL, shape, 2, COG_DTYPE_FLOAT32);
    if (!embeddings) return NULL;
    
    float *emb_data = (float*)embeddings->data;
    
    /* Initialize embeddings from atoms */
    for (size_t i = 0; i < impl->n_atoms; i++) {
        CogAtom *atom = impl->atoms[i];
        if (!atom) continue;
        
        CogTensor *atom_emb = cog_atomspace_to_tensor(as, atom, 0);
        if (atom_emb) {
            memcpy(&emb_data[i * 128], atom_emb->data, 128 * sizeof(float));
            cog_tensor_free(atom_emb);
        }
    }
    
    /* Apply GNN layers */
    for (int layer = 0; layer < n_layers; layer++) {
        CogTensor *W = weights[layer];
        if (!W) continue;
        
        /* Message passing: aggregate neighbor embeddings */
        int64_t new_shape[] = {(int64_t)impl->n_atoms, W->shape[1]};
        CogTensor *new_emb = cog_tensor_create(NULL, new_shape, 2, COG_DTYPE_FLOAT32);
        float *new_data = (float*)new_emb->data;
        
        for (size_t i = 0; i < impl->n_atoms; i++) {
            CogAtom *atom = impl->atoms[i];
            if (!atom) continue;
            
            /* Self embedding */
            float self_emb[128];
            memcpy(self_emb, &emb_data[i * 128], 128 * sizeof(float));
            
            /* Aggregate neighbor embeddings */
            float neighbor_sum[128] = {0};
            int neighbor_count = 0;
            
            if (atom->outgoing) {
                for (size_t j = 0; j < atom->arity; j++) {
                    CogAtom *neighbor = atom->outgoing[j];
                    if (neighbor && neighbor->handle < impl->n_atoms) {
                        for (int k = 0; k < 128; k++) {
                            neighbor_sum[k] += emb_data[neighbor->handle * 128 + k];
                        }
                        neighbor_count++;
                    }
                }
            }
            
            /* Combine and transform */
            float *W_data = (float*)W->data;
            for (int64_t out = 0; out < W->shape[1]; out++) {
                float sum = 0.0f;
                for (int64_t in = 0; in < W->shape[0] && in < 128; in++) {
                    float combined = self_emb[in];
                    if (neighbor_count > 0) {
                        combined += neighbor_sum[in] / neighbor_count;
                    }
                    sum += combined * W_data[in * W->shape[1] + out];
                }
                /* ReLU activation */
                new_data[i * W->shape[1] + out] = sum > 0 ? sum : 0;
            }
        }
        
        cog_tensor_free(embeddings);
        embeddings = new_emb;
        emb_data = (float*)embeddings->data;
    }
    
    return embeddings;
}
