/**
 * CogAtomSpace - OpenCog AtomSpace Integration with Tensor Computing
 * 
 * This header defines the integration between OpenCog's AtomSpace hypergraph
 * knowledge representation and ATen tensor computing. It enables:
 * 
 * - Storing tensors as atoms in the AtomSpace
 * - Representing tensor operations as links
 * - Applying PLN (Probabilistic Logic Networks) to tensor reasoning
 * - Using ECAN (Economic Attention Network) for tensor attention
 * - Pattern mining on tensor-annotated knowledge graphs
 * 
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                      AtomSpace Hypergraph                        │
 * │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
 * │  │ TensorNode  │──│ TensorLink  │──│ ConceptNode │              │
 * │  │ (data)      │  │ (operation) │  │ (semantics) │              │
 * │  └─────────────┘  └─────────────┘  └─────────────┘              │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                    Cognitive Services                            │
 * │  ┌─────┐  ┌──────┐  ┌─────┐  ┌─────────────┐                    │
 * │  │ PLN │  │ ECAN │  │ URE │  │ PatternMiner│                    │
 * │  └─────┘  └──────┘  └─────┘  └─────────────┘                    │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                    ATen Tensor Backend                           │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * Copyright (c) 2025 ATenCo9 Project
 * License: BSD-3-Clause
 */

#ifndef COG_ATOMSPACE_H
#define COG_ATOMSPACE_H

#include "cogint.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Extended Atom Types for Tensor Integration
 *============================================================================*/

/* Tensor-specific atom types */
typedef enum {
    /* Base tensor types */
    COG_ATOM_TENSOR_NODE = 100,        /* Tensor data storage */
    COG_ATOM_TENSOR_SHAPE_NODE = 101,  /* Shape specification */
    COG_ATOM_TENSOR_DTYPE_NODE = 102,  /* Data type specification */
    
    /* Tensor operation links */
    COG_ATOM_TENSOR_OP_LINK = 110,     /* Generic tensor operation */
    COG_ATOM_TENSOR_ADD_LINK = 111,    /* Addition operation */
    COG_ATOM_TENSOR_MUL_LINK = 112,    /* Multiplication operation */
    COG_ATOM_TENSOR_MATMUL_LINK = 113, /* Matrix multiplication */
    COG_ATOM_TENSOR_CONV_LINK = 114,   /* Convolution operation */
    COG_ATOM_TENSOR_SOFTMAX_LINK = 115,/* Softmax operation */
    COG_ATOM_TENSOR_RELU_LINK = 116,   /* ReLU activation */
    
    /* Semantic annotation links */
    COG_ATOM_TENSOR_MEANING_LINK = 120,/* Tensor semantic meaning */
    COG_ATOM_TENSOR_CONTEXT_LINK = 121,/* Tensor context */
    COG_ATOM_TENSOR_GRADIENT_LINK = 122,/* Gradient relationship */
    
    /* Attention and importance */
    COG_ATOM_ATTENTION_LINK = 130,     /* Attention weight relationship */
    COG_ATOM_IMPORTANCE_LINK = 131,    /* Importance annotation */
    
    /* Neural network structure */
    COG_ATOM_LAYER_NODE = 140,         /* Neural network layer */
    COG_ATOM_WEIGHT_LINK = 141,        /* Weight connection */
    COG_ATOM_ACTIVATION_LINK = 142,    /* Activation function */
} CogTensorAtomType;

/*============================================================================
 * Truth Value Types for Tensor Reasoning
 *============================================================================*/

/* Truth value types */
typedef enum {
    COG_TV_SIMPLE = 0,           /* Simple truth value (strength, confidence) */
    COG_TV_COUNT = 1,            /* Count-based truth value */
    COG_TV_INDEFINITE = 2,       /* Indefinite probability */
    COG_TV_FUZZY = 3,            /* Fuzzy truth value */
    COG_TV_PROBABILISTIC = 4,    /* Full probability distribution */
    COG_TV_TENSOR = 5,           /* Tensor-valued truth (for neural-symbolic) */
} CogTruthValueType;

/* Extended truth value with tensor support */
typedef struct {
    CogTruthValueType type;
    double strength;
    double confidence;
    double count;                /* For count-based TV */
    
    /* Tensor truth value */
    CogTensor *tensor_value;     /* For neural-symbolic integration */
} CogTruthValue;

/*============================================================================
 * Attention Value for ECAN
 *============================================================================*/

/* Attention value structure */
typedef struct {
    int16_t sti;                 /* Short-term importance [-32768, 32767] */
    int16_t lti;                 /* Long-term importance [-32768, 32767] */
    int16_t vlti;                /* Very long-term importance */
    
    /* Extended attention for tensors */
    CogTensor *attention_weights;/* Per-element attention weights */
    double focus_boundary;       /* Attentional focus threshold */
} CogAttentionValue;

/*============================================================================
 * AtomSpace Extended Structure
 *============================================================================*/

/* AtomSpace configuration */
typedef struct {
    char *name;
    size_t initial_capacity;
    int enable_indexing;
    int enable_attention;
    int enable_tensor_storage;
    
    /* 9P export settings */
    int export_9p;
    char *export_path;
    uint16_t export_port;
} CogAtomSpaceConfig;

/* Atom index for fast lookup */
typedef struct {
    /* Type index */
    CogAtom ***by_type;
    size_t *type_counts;
    size_t n_types;
    
    /* Name index (hash table) */
    void *name_index;
    
    /* Tensor index */
    CogAtom **tensor_atoms;
    size_t n_tensor_atoms;
    
    /* Attention-based index */
    CogAtom **attention_focus;
    size_t focus_size;
} CogAtomIndex;

/*============================================================================
 * Tensor-Atom Binding Functions
 *============================================================================*/

/**
 * Create a TensorNode from a CogTensor
 * 
 * @param as AtomSpace to add the atom to
 * @param tensor The tensor to store
 * @param name Optional name for the tensor atom
 * @return The created TensorNode atom
 */
COGINT_API CogAtom* cog_tensor_node_create(CogAtomSpace *as, CogTensor *tensor, 
                                            const char *name);

/**
 * Get the tensor from a TensorNode
 * 
 * @param atom The TensorNode atom
 * @return The stored tensor, or NULL if not a TensorNode
 */
COGINT_API CogTensor* cog_tensor_node_get(CogAtom *atom);

/**
 * Create a tensor operation link
 * 
 * @param as AtomSpace
 * @param op_type The operation type (add, mul, matmul, etc.)
 * @param inputs Array of input TensorNode atoms
 * @param n_inputs Number of inputs
 * @param output Output TensorNode atom
 * @return The created operation link
 */
COGINT_API CogAtom* cog_tensor_op_link_create(CogAtomSpace *as, 
                                               CogTensorAtomType op_type,
                                               CogAtom **inputs, size_t n_inputs,
                                               CogAtom *output);

/**
 * Execute a tensor operation link
 * 
 * @param link The operation link to execute
 * @return The result tensor, or NULL on error
 */
COGINT_API CogTensor* cog_tensor_op_execute(CogAtom *link);

/**
 * Annotate a tensor with semantic meaning
 * 
 * @param as AtomSpace
 * @param tensor_atom The TensorNode to annotate
 * @param concept The ConceptNode representing the meaning
 * @param tv Truth value for the annotation
 * @return The created MeaningLink
 */
COGINT_API CogAtom* cog_tensor_annotate(CogAtomSpace *as, CogAtom *tensor_atom,
                                         CogAtom *concept, CogTruthValue *tv);

/*============================================================================
 * PLN (Probabilistic Logic Networks) Integration
 *============================================================================*/

/* PLN rule types */
typedef enum {
    COG_PLN_DEDUCTION = 0,
    COG_PLN_INDUCTION = 1,
    COG_PLN_ABDUCTION = 2,
    COG_PLN_MODUS_PONENS = 3,
    COG_PLN_AND_INTRO = 4,
    COG_PLN_OR_INTRO = 5,
    COG_PLN_INHERITANCE = 6,
    COG_PLN_SIMILARITY = 7,
    
    /* Tensor-specific rules */
    COG_PLN_TENSOR_SIMILARITY = 100,  /* Tensor cosine similarity */
    COG_PLN_TENSOR_ANALOGY = 101,     /* Tensor analogy (a:b::c:d) */
    COG_PLN_TENSOR_COMPOSITION = 102, /* Tensor composition */
} CogPLNRuleType;

/* PLN inference configuration */
typedef struct {
    int max_steps;
    double confidence_threshold;
    int max_results;
    CogPLNRuleType *allowed_rules;
    size_t n_rules;
    
    /* Tensor-specific settings */
    int use_tensor_similarity;
    double tensor_similarity_threshold;
} CogPLNConfig;

/**
 * Run PLN inference on a query
 * 
 * @param as AtomSpace
 * @param query The query atom
 * @param config Inference configuration
 * @param results Output array of result atoms
 * @param n_results Number of results found
 * @return 0 on success, error code otherwise
 */
COGINT_API int cog_pln_inference(CogAtomSpace *as, CogAtom *query,
                                  CogPLNConfig *config,
                                  CogAtom ***results, size_t *n_results);

/**
 * Compute tensor-based similarity for PLN
 * 
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return Similarity score [0, 1]
 */
COGINT_API double cog_pln_tensor_similarity(CogTensor *t1, CogTensor *t2);

/**
 * Create an InheritanceLink with tensor evidence
 * 
 * @param as AtomSpace
 * @param child Child concept
 * @param parent Parent concept
 * @param evidence Tensor evidence for the inheritance
 * @return The created InheritanceLink
 */
COGINT_API CogAtom* cog_pln_inheritance_with_tensor(CogAtomSpace *as,
                                                     CogAtom *child,
                                                     CogAtom *parent,
                                                     CogTensor *evidence);

/*============================================================================
 * ECAN (Economic Attention Network) Integration
 *============================================================================*/

/* ECAN configuration */
typedef struct {
    double sti_funds;            /* Total STI funds in system */
    double lti_funds;            /* Total LTI funds in system */
    double sti_rent;             /* STI rent per cycle */
    double lti_rent;             /* LTI rent per cycle */
    double wage;                 /* Wage for useful atoms */
    
    int16_t af_boundary;         /* Attentional focus boundary */
    int max_af_size;             /* Maximum attentional focus size */
    
    /* Tensor attention settings */
    int use_tensor_attention;
    double tensor_attention_weight;
} CogECANConfig;

/**
 * Initialize ECAN for an AtomSpace
 * 
 * @param as AtomSpace
 * @param config ECAN configuration
 * @return 0 on success
 */
COGINT_API int cog_ecan_init(CogAtomSpace *as, CogECANConfig *config);

/**
 * Spread attention from a source atom
 * 
 * @param as AtomSpace
 * @param source Source atom for attention spread
 * @param amount Amount of attention to spread
 * @return 0 on success
 */
COGINT_API int cog_ecan_spread_attention(CogAtomSpace *as, CogAtom *source,
                                          double amount);

/**
 * Get atoms in attentional focus
 * 
 * @param as AtomSpace
 * @param atoms Output array of atoms in focus
 * @param n_atoms Number of atoms in focus
 * @return 0 on success
 */
COGINT_API int cog_ecan_get_focus(CogAtomSpace *as, CogAtom ***atoms, 
                                   size_t *n_atoms);

/**
 * Compute attention weights for a tensor based on AtomSpace attention
 * 
 * @param as AtomSpace
 * @param tensor_atom TensorNode to compute attention for
 * @return Attention weight tensor
 */
COGINT_API CogTensor* cog_ecan_tensor_attention(CogAtomSpace *as, 
                                                 CogAtom *tensor_atom);

/**
 * Apply attention-weighted tensor operation
 * 
 * @param tensor Input tensor
 * @param attention Attention weights
 * @return Attention-weighted tensor
 */
COGINT_API CogTensor* cog_ecan_apply_attention(CogTensor *tensor, 
                                                CogTensor *attention);

/*============================================================================
 * Pattern Mining Integration
 *============================================================================*/

/* Pattern mining configuration */
typedef struct {
    int min_support;
    int max_pattern_size;
    double min_confidence;
    int max_patterns;
    
    /* Tensor pattern settings */
    int mine_tensor_patterns;
    double tensor_similarity_threshold;
} CogPatternMinerConfig;

/* Mined pattern structure */
typedef struct {
    CogAtom *pattern;            /* The pattern atom */
    int support;                 /* Support count */
    double confidence;           /* Pattern confidence */
    
    /* Tensor pattern info */
    CogTensor *pattern_tensor;   /* Tensor representation of pattern */
    CogAtom **instances;         /* Pattern instances */
    size_t n_instances;
} CogMinedPattern;

/**
 * Mine patterns from AtomSpace
 * 
 * @param as AtomSpace
 * @param config Mining configuration
 * @param patterns Output array of mined patterns
 * @param n_patterns Number of patterns found
 * @return 0 on success
 */
COGINT_API int cog_pattern_mine(CogAtomSpace *as, CogPatternMinerConfig *config,
                                 CogMinedPattern **patterns, size_t *n_patterns);

/**
 * Find tensor patterns (similar tensor structures)
 * 
 * @param as AtomSpace
 * @param reference Reference tensor
 * @param threshold Similarity threshold
 * @param matches Output array of matching TensorNodes
 * @param n_matches Number of matches
 * @return 0 on success
 */
COGINT_API int cog_pattern_find_tensor(CogAtomSpace *as, CogTensor *reference,
                                        double threshold,
                                        CogAtom ***matches, size_t *n_matches);

/*============================================================================
 * URE (Unified Rule Engine) Integration
 *============================================================================*/

/* Rule structure */
typedef struct {
    char *name;
    CogAtom *pattern;            /* Pattern to match */
    CogAtom *rewrite;            /* Rewrite template */
    CogTruthValue tv;            /* Rule truth value */
    
    /* Tensor transformation */
    CogTensor *(*tensor_transform)(CogTensor **inputs, size_t n_inputs);
} CogRule;

/* Rule base */
typedef struct {
    CogRule *rules;
    size_t n_rules;
    size_t capacity;
} CogRuleBase;

/**
 * Create a rule base
 * 
 * @return New rule base
 */
COGINT_API CogRuleBase* cog_ure_create_rulebase(void);

/**
 * Add a rule to the rule base
 * 
 * @param rb Rule base
 * @param rule Rule to add
 * @return 0 on success
 */
COGINT_API int cog_ure_add_rule(CogRuleBase *rb, CogRule *rule);

/**
 * Run forward chaining with URE
 * 
 * @param as AtomSpace
 * @param rb Rule base
 * @param source Starting atom
 * @param max_steps Maximum inference steps
 * @param results Output results
 * @param n_results Number of results
 * @return 0 on success
 */
COGINT_API int cog_ure_forward_chain(CogAtomSpace *as, CogRuleBase *rb,
                                      CogAtom *source, int max_steps,
                                      CogAtom ***results, size_t *n_results);

/**
 * Run backward chaining with URE
 * 
 * @param as AtomSpace
 * @param rb Rule base
 * @param target Target atom to prove
 * @param max_steps Maximum inference steps
 * @param proof Output proof tree
 * @return 0 on success, -1 if unprovable
 */
COGINT_API int cog_ure_backward_chain(CogAtomSpace *as, CogRuleBase *rb,
                                       CogAtom *target, int max_steps,
                                       CogAtom **proof);

/*============================================================================
 * Neural-Symbolic Integration
 *============================================================================*/

/**
 * Convert AtomSpace subgraph to tensor representation
 * 
 * @param as AtomSpace
 * @param root Root atom of subgraph
 * @param depth Maximum depth to traverse
 * @return Tensor representation of subgraph
 */
COGINT_API CogTensor* cog_atomspace_to_tensor(CogAtomSpace *as, CogAtom *root,
                                               int depth);

/**
 * Create atoms from tensor embeddings
 * 
 * @param as AtomSpace
 * @param embeddings Tensor of embeddings
 * @param names Array of names for created atoms
 * @param n Number of embeddings
 * @return Array of created atoms
 */
COGINT_API CogAtom** cog_tensor_to_atoms(CogAtomSpace *as, CogTensor *embeddings,
                                          char **names, size_t n);

/**
 * Compute graph neural network forward pass on AtomSpace
 * 
 * @param as AtomSpace
 * @param weights GNN weight tensors
 * @param n_layers Number of GNN layers
 * @return Node embeddings tensor
 */
COGINT_API CogTensor* cog_gnn_forward(CogAtomSpace *as, CogTensor **weights,
                                       int n_layers);

/**
 * Train GNN on AtomSpace with supervision
 * 
 * @param as AtomSpace
 * @param weights GNN weight tensors (updated in place)
 * @param n_layers Number of layers
 * @param targets Target labels
 * @param learning_rate Learning rate
 * @param epochs Number of training epochs
 * @return Final loss value
 */
COGINT_API double cog_gnn_train(CogAtomSpace *as, CogTensor **weights,
                                 int n_layers, CogTensor *targets,
                                 double learning_rate, int epochs);

#ifdef __cplusplus
}
#endif

#endif /* COG_ATOMSPACE_H */
