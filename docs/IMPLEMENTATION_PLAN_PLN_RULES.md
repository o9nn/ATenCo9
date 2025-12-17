# PLN Rules Expansion Implementation Plan

## Overview

Probabilistic Logic Networks (PLN) provide the reasoning engine for CogInt's neural-symbolic integration. This plan expands the PLN rule set to enable sophisticated reasoning over tensor data, combining symbolic logic with subsymbolic neural representations for hybrid AGI systems.

## Current Status

**Existing Infrastructure**
- Basic PLN API in `cog_atomspace.h`
- Three core functions: `cog_pln_inference()`, `cog_pln_tensor_similarity()`, `cog_pln_inheritance_with_tensor()`
- Truth value structures (strength, confidence)
- AtomSpace integration for tensor nodes

**Gaps**
- Limited rule set (only basic inference)
- No forward/backward chaining
- No probabilistic reasoning
- No tensor-specific inference rules
- No learning mechanisms

## PLN Fundamentals

### Truth Value Semantics

**Simple Truth Value (STV)**:
- Strength (s): Probability estimate ∈ [0, 1]
- Confidence (c): Certainty of estimate ∈ [0, 1]
- Count (n): Evidence count (derived from confidence)

**Indefinite Truth Value (ITV)**:
- Lower bound (L): Minimum probability
- Upper bound (U): Maximum probability
- Confidence (c): Certainty of bounds

### Inference Rules

PLN rules transform truth values through logical operations while maintaining probabilistic semantics. Each rule has:
- **Premises**: Input atoms with truth values
- **Conclusion**: Output atom with derived truth value
- **Formula**: Mathematical transformation of truth values

## Architecture Design

### Rule System Structure

```
┌─────────────────────────────────────────────────────────┐
│              PLN Inference Engine                        │
├─────────────────────────────────────────────────────────┤
│  Forward Chainer  │  Backward Chainer  │  Pattern Matcher│
├─────────────────────────────────────────────────────────┤
│                    Rule Base                             │
│  ┌──────────────┬──────────────┬──────────────────┐    │
│  │ Deduction    │ Induction    │ Abduction        │    │
│  │ Revision     │ Similarity   │ Inheritance      │    │
│  │ Implication  │ Equivalence  │ Tensor Rules     │    │
│  └──────────────┴──────────────┴──────────────────┘    │
├─────────────────────────────────────────────────────────┤
│              Truth Value Computation                     │
│  STV Formulas  │  ITV Formulas  │  Tensor Similarity   │
└─────────────────────────────────────────────────────────┘
```

### Rule Categories

**Logical Rules**:
- Deduction: (A→B, B→C) ⊢ (A→C)
- Induction: (A→B, A→C) ⊢ (B→C)
- Abduction: (A→C, B→C) ⊢ (A→B)
- Revision: (A, A') ⊢ A''
- Modus Ponens: (A→B, A) ⊢ B

**Similarity Rules**:
- Similarity Introduction
- Similarity Transitivity
- Attraction Introduction

**Inheritance Rules**:
- Inheritance Deduction
- Inheritance Induction
- Inheritance Specialization

**Tensor-Specific Rules**:
- Tensor Similarity Inference
- Embedding Space Reasoning
- Neural Pattern Matching
- Gradient-Based Inference

## Implementation Phases

### Phase 1: Core Rule Engine (Week 1-2)

**Objective**: Implement forward and backward chaining infrastructure

**Tasks**:

1. **Rule representation**
   ```c
   typedef struct CogPLNRule {
       char *name;
       CogAtomType *premise_types;    // Required atom types
       size_t n_premises;
       CogAtomType conclusion_type;   // Output atom type
       
       // Rule application function
       int (*apply)(CogAtomSpace *as, CogAtom **premises,
                    CogAtom **conclusion, CogTruthValue *tv);
       
       // Precondition check
       bool (*precondition)(CogAtom **premises, size_t n);
       
       // Priority for rule selection
       double priority;
   } CogPLNRule;
   ```

2. **Forward chainer**
   ```c
   typedef struct CogForwardChainer {
       CogAtomSpace *atomspace;
       CogPLNRule **rules;
       size_t n_rules;
       
       // Configuration
       size_t max_iterations;
       double confidence_threshold;
       bool use_attention;  // Use ECAN for focus
   } CogForwardChainer;
   
   // Run forward chaining
   int cog_forward_chain(CogForwardChainer *fc, CogAtom *source,
                         CogAtom ***results, size_t *n_results);
   ```

3. **Backward chainer**
   ```c
   typedef struct CogBackwardChainer {
       CogAtomSpace *atomspace;
       CogPLNRule **rules;
       size_t n_rules;
       
       // Configuration
       size_t max_depth;
       double confidence_threshold;
   } CogBackwardChainer;
   
   // Run backward chaining
   int cog_backward_chain(CogBackwardChainer *bc, CogAtom *goal,
                          CogAtom ***proof, size_t *proof_len);
   ```

4. **Pattern matching**
   - Unification algorithm
   - Variable binding
   - Pattern templates
   - Efficient indexing

**Deliverables**:
- Forward chaining engine
- Backward chaining engine
- Pattern matcher
- Rule application framework

**Success Criteria**:
- Can apply rules iteratively
- Finds valid inference chains
- Handles variable binding
- Efficient pattern matching

### Phase 2: Logical Rules (Week 3-4)

**Objective**: Implement core logical inference rules

**Tasks**:

1. **Deduction rule**
   ```c
   // (A→B <s1, c1>) ∧ (B→C <s2, c2>) ⊢ (A→C <s3, c3>)
   // s3 = s1 * s2
   // c3 = c1 * c2 * (1 - |s1*s2 - s1*s2|)
   int pln_rule_deduction(CogAtomSpace *as, CogAtom **premises,
                          CogAtom **conclusion, CogTruthValue *tv) {
       CogAtom *AB = premises[0];  // A→B
       CogAtom *BC = premises[1];  // B→C
       
       double s1 = AB->strength;
       double c1 = AB->confidence;
       double s2 = BC->strength;
       double c2 = BC->confidence;
       
       tv->strength = s1 * s2;
       tv->confidence = c1 * c2 * (1.0 - fabs(s1*s2 - s1*s2));
       
       // Create A→C link
       *conclusion = cog_atom_create_link(as, COG_ATOM_INHERITANCE,
                                          AB->outgoing[0], BC->outgoing[1]);
       return COG_OK;
   }
   ```

2. **Induction rule**
   ```c
   // (A→B <s1, c1>) ∧ (A→C <s2, c2>) ⊢ (B→C <s3, c3>)
   // Inductive generalization
   int pln_rule_induction(CogAtomSpace *as, CogAtom **premises,
                          CogAtom **conclusion, CogTruthValue *tv);
   ```

3. **Abduction rule**
   ```c
   // (A→C <s1, c1>) ∧ (B→C <s2, c2>) ⊢ (A→B <s3, c3>)
   // Abductive hypothesis formation
   int pln_rule_abduction(CogAtomSpace *as, CogAtom **premises,
                          CogAtom **conclusion, CogTruthValue *tv);
   ```

4. **Revision rule**
   ```c
   // (A <s1, c1>) ∧ (A <s2, c2>) ⊢ (A <s3, c3>)
   // Combine evidence from multiple sources
   // s3 = (s1*c1 + s2*c2) / (c1 + c2)
   // c3 = (c1 + c2) / (1 + c1*c2)
   int pln_rule_revision(CogAtomSpace *as, CogAtom **premises,
                         CogAtom **conclusion, CogTruthValue *tv);
   ```

5. **Modus Ponens**
   ```c
   // (A→B <s1, c1>) ∧ (A <s2, c2>) ⊢ (B <s3, c3>)
   int pln_rule_modus_ponens(CogAtomSpace *as, CogAtom **premises,
                             CogAtom **conclusion, CogTruthValue *tv);
   ```

**Deliverables**:
- 10+ logical inference rules
- Truth value formulas
- Rule unit tests
- Integration tests

**Success Criteria**:
- Correct truth value computation
- Valid logical inferences
- Handles edge cases
- Numerically stable

### Phase 3: Similarity and Inheritance (Week 5)

**Objective**: Implement similarity and inheritance reasoning

**Tasks**:

1. **Similarity rules**
   ```c
   // Similarity introduction
   // (A→B <s1, c1>) ∧ (B→A <s2, c2>) ⊢ (A≈B <s3, c3>)
   int pln_rule_similarity_intro(CogAtomSpace *as, CogAtom **premises,
                                 CogAtom **conclusion, CogTruthValue *tv);
   
   // Similarity transitivity
   // (A≈B <s1, c1>) ∧ (B≈C <s2, c2>) ⊢ (A≈C <s3, c3>)
   int pln_rule_similarity_trans(CogAtomSpace *as, CogAtom **premises,
                                 CogAtom **conclusion, CogTruthValue *tv);
   ```

2. **Inheritance rules**
   ```c
   // Inheritance specialization
   // (A→B <s1, c1>) ∧ (C→A <s2, c2>) ⊢ (C→B <s3, c3>)
   int pln_rule_inheritance_spec(CogAtomSpace *as, CogAtom **premises,
                                 CogAtom **conclusion, CogTruthValue *tv);
   ```

3. **Attraction rules**
   - Intensional inheritance
   - Extensional inheritance
   - Mixed inheritance

**Deliverables**:
- Similarity reasoning rules
- Inheritance reasoning rules
- Attraction computation
- Comprehensive tests

**Success Criteria**:
- Similarity detection working
- Inheritance chains valid
- Attraction values correct

### Phase 4: Tensor-Specific Rules (Week 6-7)

**Objective**: Implement neural-symbolic integration rules

**Tasks**:

1. **Tensor similarity inference**
   ```c
   // If tensors T1 and T2 are similar (cosine > threshold),
   // infer similarity between their associated concepts
   int pln_rule_tensor_similarity(CogAtomSpace *as, CogAtom **premises,
                                  CogAtom **conclusion, CogTruthValue *tv) {
       CogAtom *concept_a = premises[0];
       CogAtom *concept_b = premises[1];
       
       // Get associated tensors
       CogTensor *t1 = cog_tensor_node_get(concept_a);
       CogTensor *t2 = cog_tensor_node_get(concept_b);
       
       // Compute cosine similarity
       double sim = cog_pln_tensor_similarity(t1, t2);
       
       // Map similarity to truth value
       tv->strength = sim;
       tv->confidence = 0.9;  // High confidence from direct computation
       
       // Create similarity link
       *conclusion = cog_atom_create_link(as, COG_ATOM_SIMILARITY,
                                          concept_a, concept_b);
       return COG_OK;
   }
   ```

2. **Embedding space reasoning**
   ```c
   // Use vector space operations for inference
   // If vec(A) + vec(R) ≈ vec(B), infer A→B
   int pln_rule_embedding_analogy(CogAtomSpace *as, CogAtom **premises,
                                  CogAtom **conclusion, CogTruthValue *tv);
   ```

3. **Neural pattern matching**
   ```c
   // Match tensor patterns for concept recognition
   int pln_rule_pattern_match(CogAtomSpace *as, CogAtom **premises,
                              CogAtom **conclusion, CogTruthValue *tv);
   ```

4. **Gradient-based inference**
   ```c
   // Use gradient information for causal reasoning
   // If ∂B/∂A is large, infer A influences B
   int pln_rule_gradient_causality(CogAtomSpace *as, CogAtom **premises,
                                   CogAtom **conclusion, CogTruthValue *tv);
   ```

5. **Attention-weighted inference**
   ```c
   // Weight inference by attention values
   // High STI atoms contribute more to conclusions
   int pln_rule_attention_weighted(CogAtomSpace *as, CogAtom **premises,
                                   CogAtom **conclusion, CogTruthValue *tv);
   ```

**Deliverables**:
- Tensor similarity rules
- Embedding space rules
- Pattern matching rules
- Gradient-based rules
- Attention integration

**Success Criteria**:
- Tensor similarity detection
- Embedding analogies working
- Pattern recognition functional
- Gradient causality valid

### Phase 5: Higher-Order Inference (Week 8-9)

**Objective**: Implement advanced reasoning capabilities

**Tasks**:

1. **Higher-order unification**
   - Unify predicates and relations
   - Handle quantifiers
   - Support lambda expressions

2. **Probabilistic inference**
   ```c
   // Bayesian network inference
   int pln_rule_bayes_inference(CogAtomSpace *as, CogAtom **premises,
                                CogAtom **conclusion, CogTruthValue *tv);
   
   // Markov logic network
   int pln_rule_mln_inference(CogAtomSpace *as, CogAtom **premises,
                              CogAtom **conclusion, CogTruthValue *tv);
   ```

3. **Temporal reasoning**
   ```c
   // Temporal deduction
   // (A→B at t1) ∧ (B→C at t2) ⊢ (A→C at t3)
   int pln_rule_temporal_deduction(CogAtomSpace *as, CogAtom **premises,
                                   CogAtom **conclusion, CogTruthValue *tv);
   ```

4. **Fuzzy logic integration**
   - Fuzzy membership functions
   - Fuzzy operators (AND, OR, NOT)
   - Defuzzification

5. **Meta-reasoning**
   - Reason about reasoning processes
   - Confidence in inference chains
   - Explanation generation

**Deliverables**:
- Higher-order unification
- Probabilistic reasoning
- Temporal reasoning
- Fuzzy logic support
- Meta-reasoning capabilities

**Success Criteria**:
- Complex queries answered
- Probabilistic inference correct
- Temporal reasoning valid
- Meta-level explanations generated

### Phase 6: Learning and Optimization (Week 10-11)

**Objective**: Enable rule learning and inference optimization

**Tasks**:

1. **Rule learning**
   ```c
   // Learn new rules from examples
   typedef struct CogPLNLearner {
       CogAtomSpace *atomspace;
       size_t min_support;      // Minimum evidence count
       double min_confidence;   // Minimum confidence threshold
   } CogPLNLearner;
   
   // Mine frequent patterns and generate rules
   int cog_pln_learn_rules(CogPLNLearner *learner,
                           CogPLNRule ***rules, size_t *n_rules);
   ```

2. **Rule optimization**
   - Prune redundant rules
   - Merge similar rules
   - Optimize rule ordering
   - Cache frequent patterns

3. **Attention-guided inference**
   - Use ECAN to focus on important atoms
   - Prioritize high-STI premises
   - Spread attention during inference

4. **Parallel inference**
   - Parallelize rule application
   - Concurrent forward chaining
   - Distributed backward chaining

5. **Incremental learning**
   - Update truth values online
   - Revise beliefs with new evidence
   - Forget low-confidence atoms

**Deliverables**:
- Rule learning system
- Inference optimization
- Attention integration
- Parallel execution
- Incremental updates

**Success Criteria**:
- New rules learned from data
- 10x inference speedup
- Attention improves relevance
- Scales to large AtomSpaces

### Phase 7: Integration and Applications (Week 12)

**Objective**: Integrate PLN with cognitive pipeline and create applications

**Tasks**:

1. **Cognitive pipeline integration**
   ```c
   // Use PLN in reasoning stage
   int cog_pipeline_reason_pln(CogPipeline *pipe, CogAtom *query) {
       // Forward chain from perception
       CogAtom **percepts = get_perception_atoms(pipe);
       CogAtom **inferences = NULL;
       size_t n_inferences = 0;
       
       cog_forward_chain(pipe->forward_chainer, percepts[0],
                         &inferences, &n_inferences);
       
       // Backward chain to goal
       CogAtom **proof = NULL;
       size_t proof_len = 0;
       cog_backward_chain(pipe->backward_chainer, query,
                          &proof, &proof_len);
       
       // Send to action stage
       send_to_action(pipe, proof, proof_len);
       
       return COG_OK;
   }
   ```

2. **Question answering**
   - Natural language query parsing
   - Knowledge base reasoning
   - Answer generation

3. **Concept formation**
   - Cluster similar concepts
   - Abstract common patterns
   - Build concept hierarchies

4. **Causal reasoning**
   - Infer causal relationships
   - Counterfactual reasoning
   - Intervention analysis

5. **Explanation generation**
   - Trace inference chains
   - Generate natural language explanations
   - Visualize reasoning graphs

**Deliverables**:
- Pipeline integration
- QA system
- Concept formation
- Causal reasoning
- Explanation system

**Success Criteria**:
- Answers complex questions
- Forms meaningful concepts
- Identifies causal relationships
- Generates understandable explanations

## Truth Value Formulas

### Deduction

```
(A→B <s1, c1>) ∧ (B→C <s2, c2>) ⊢ (A→C <s3, c3>)

s3 = s1 * s2
c3 = c1 * c2 * max(0, 1 - |s1*s2 - s1*s2|)
```

### Induction

```
(A→B <s1, c1>) ∧ (A→C <s2, c2>) ⊢ (B→C <s3, c3>)

s3 = s1 * s2 / P(A)
c3 = c1 * c2 * P(A)
```

### Revision

```
(A <s1, c1>) ∧ (A <s2, c2>) ⊢ (A <s3, c3>)

w1 = c1 / (c1 + c2)
w2 = c2 / (c1 + c2)
s3 = w1*s1 + w2*s2
c3 = (c1 + c2) / (1 + c1*c2)
```

### Similarity

```
(A≈B <s1, c1>) ∧ (B≈C <s2, c2>) ⊢ (A≈C <s3, c3>)

s3 = s1 * s2
c3 = c1 * c2
```

## Testing Strategy

### Unit Tests

**Rule Tests**:
- Test each rule in isolation
- Verify truth value formulas
- Check edge cases (s=0, s=1, c=0, c=1)

**Chainer Tests**:
- Forward chaining correctness
- Backward chaining completeness
- Termination guarantees

### Integration Tests

**Inference Chains**:
- Multi-step reasoning
- Complex query answering
- Proof verification

**Tensor Integration**:
- Neural-symbolic reasoning
- Embedding space operations
- Gradient-based inference

### Performance Tests

**Scalability**:
- Large AtomSpace (1M+ atoms)
- Deep inference chains (100+ steps)
- Parallel execution efficiency

**Accuracy**:
- Compare to ground truth
- Measure precision/recall
- Evaluate confidence calibration

## Example Applications

### Question Answering

```c
// Query: "What is the capital of France?"
CogAtom *query = parse_query("capital(France, ?X)");

// Backward chain to find answer
CogAtom **proof = NULL;
size_t proof_len = 0;
cog_backward_chain(bc, query, &proof, &proof_len);

// Extract answer: Paris
CogAtom *answer = extract_answer(proof, proof_len);
```

### Concept Learning

```c
// Learn concept "bird" from examples
CogAtom *examples[] = {
    create_atom("sparrow"),
    create_atom("eagle"),
    create_atom("penguin")
};

// Find common properties
CogAtom *bird_concept = cog_pln_learn_concept(as, examples, 3);

// Infer: "robin" is a bird (high similarity)
CogAtom *robin = create_atom("robin");
double sim = cog_pln_tensor_similarity(robin_tensor, bird_concept_tensor);
// sim = 0.92 → robin is likely a bird
```

### Causal Inference

```c
// Infer: smoking causes cancer
CogAtom *smoking = create_atom("smoking");
CogAtom *cancer = create_atom("cancer");

// Use gradient-based causality
CogTensor *smoking_t = get_tensor(smoking);
CogTensor *cancer_t = get_tensor(cancer);
double gradient = compute_gradient(cancer_t, smoking_t);

// High gradient → causal relationship
if (gradient > threshold) {
    create_link(COG_ATOM_INHERITANCE, smoking, cancer);
}
```

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 2 weeks | Core rule engine |
| Phase 2 | 2 weeks | Logical rules |
| Phase 3 | 1 week | Similarity and inheritance |
| Phase 4 | 2 weeks | Tensor-specific rules |
| Phase 5 | 2 weeks | Higher-order inference |
| Phase 6 | 2 weeks | Learning and optimization |
| Phase 7 | 1 week | Integration and applications |
| **Total** | **12 weeks** | **Complete PLN system** |

## Success Metrics

**Functionality**:
- ✅ 50+ inference rules implemented
- ✅ Forward and backward chaining working
- ✅ Tensor-symbolic integration complete
- ✅ Learning from examples functional

**Accuracy**:
- ✅ > 90% precision on benchmark queries
- ✅ > 85% recall on knowledge base
- ✅ Well-calibrated confidence values

**Performance**:
- ✅ < 100ms for simple queries
- ✅ < 10s for complex multi-step reasoning
- ✅ Scales to 1M+ atom AtomSpaces

**Integration**:
- ✅ Works with cognitive pipeline
- ✅ Integrates with ECAN
- ✅ Supports 9P remote access

This implementation plan provides a comprehensive roadmap for building a sophisticated PLN reasoning system that bridges neural and symbolic AI paradigms.
