# Contributing to ATenCo9

Thank you for your interest in contributing to ATenCo9! This document provides guidelines and information for contributors.

## Development Philosophy

ATenCo9 is built on the principle of **zero-tolerance for mock features**. All contributions must:

- Implement actual working functionality, not placeholders
- Include proper error handling and resource management
- Be robust and production-ready, not superficial or temporary
- Follow the existing code patterns and architecture

## Areas for Contribution

### High Priority

1. **ATen Integration**
   - Complete C++ bridge to core ATen library
   - Direct in-memory tensor manipulation
   - CUDA acceleration support

2. **9P Protocol**
   - Server-side logic for remote channels
   - Extended tensor operations
   - Performance optimization

3. **Cognitive Algorithms**
   - PLN rules for tensor reasoning
   - ECAN attention spreading
   - Neural-symbolic integration patterns

4. **Testing**
   - Unit tests for all modules
   - Integration tests
   - Performance benchmarks

### Medium Priority

5. **Documentation**
   - API documentation
   - Architecture guides
   - Tutorial examples

6. **Tensor Networks**
   - Additional decomposition methods
   - Distributed contraction algorithms
   - Memory optimization

7. **Distributed Computing**
   - Worker pool enhancements
   - Load balancing algorithms
   - Fault tolerance

### Low Priority

8. **Build System**
   - Package management integration
   - Cross-platform support
   - Installation scripts

## Code Style

### C Code

- Follow C11 standard
- Use clear, descriptive variable names
- Include comprehensive comments
- Maintain consistent indentation (4 spaces)

```c
/* Good example */
COGINT_API CogTensor* cog_tensor_create(CogContext *ctx, int64_t *shape,
                                         int ndim, CogDType dtype) {
    if (!ctx || !shape || ndim <= 0) {
        return NULL;
    }
    
    CogTensor *t = (CogTensor*)calloc(1, sizeof(CogTensor));
    if (!t) return NULL;
    
    // Allocate and initialize
    t->ndim = ndim;
    t->dtype = dtype;
    // ... rest of implementation
    
    return t;
}
```

### Header Files

- Use include guards
- Document all public APIs
- Group related functions
- Include usage examples in comments

```c
/**
 * Create a new tensor with specified shape and data type
 * 
 * @param ctx     CogInt context
 * @param shape   Array of dimension sizes
 * @param ndim    Number of dimensions
 * @param dtype   Data type (COG_DTYPE_FLOAT32, etc.)
 * @return        New tensor or NULL on error
 * 
 * Example:
 *   int64_t shape[] = {3, 4};
 *   CogTensor *t = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
 */
COGINT_API CogTensor* cog_tensor_create(CogContext *ctx, int64_t *shape,
                                         int ndim, CogDType dtype);
```

## Submission Process

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/ATenCo9.git
cd ATenCo9
git remote add upstream https://github.com/o9nn/ATenCo9.git
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Ensure code compiles without warnings

### 4. Test Your Changes

```bash
cd cogint/build
cmake -DCOGINT_BUILD_TESTS=ON ..
make
make test
```

### 5. Commit

```bash
git add .
git commit -m "feat: add tensor decomposition method"
# or
git commit -m "fix: resolve memory leak in cog_tensor_free"
```

Use conventional commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Build system or tooling changes

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues
- Test results
- Any breaking changes noted

## Testing Guidelines

### Unit Tests

Place tests in `cogint/tests/`:

```c
#include <cogint/cogint.h>
#include <assert.h>

int main(void) {
    CogContext *ctx = cogint_init();
    assert(ctx != NULL);
    
    // Test your feature
    int64_t shape[] = {2, 3};
    CogTensor *t = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
    assert(t != NULL);
    assert(cog_tensor_ndim(t) == 2);
    
    // Clean up
    cog_tensor_free(t);
    cogint_shutdown(ctx);
    
    return 0;
}
```

### Integration Tests

Test interactions between modules:

```c
// Test AtomSpace + Tensor integration
CogContext *ctx = cogint_init();
CogAtomSpace *as = cog_atomspace_create(ctx, "test");

int64_t shape[] = {3, 3};
CogTensor *t = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);

CogAtom *atom = cog_tensor_node_create(as, "test_tensor", t);
assert(atom != NULL);

// Verify integration
CogTensor *retrieved = cog_tensor_node_get(atom);
assert(retrieved == t);
```

## Performance Considerations

- Profile code for bottlenecks
- Minimize memory allocations
- Use appropriate data structures
- Consider cache locality
- Document complexity (time/space)

## Memory Management

- Always free allocated resources
- Use RAII patterns where possible
- Check for NULL before dereferencing
- Avoid memory leaks (use valgrind)

```c
// Good pattern
CogTensor *t = cog_tensor_create(ctx, shape, ndim, dtype);
if (!t) {
    // Handle error
    return NULL;
}

// ... use tensor ...

cog_tensor_free(t);  // Always clean up
```

## Error Handling

- Return NULL for pointer types on error
- Return negative error codes for int types
- Set appropriate error codes
- Log errors when appropriate

```c
COGINT_API int cog_operation(CogContext *ctx, CogTensor *t) {
    if (!ctx || !t) {
        return COG_ERR_INVALID;
    }
    
    if (some_condition_fails) {
        return COG_ERR_OPERATION_FAILED;
    }
    
    return COG_OK;
}
```

## Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Update README for new features
- Add architecture diagrams when helpful

## Questions?

- Open an issue for discussion
- Join our community channels
- Review existing code for patterns
- Ask for clarification in pull requests

## Code Review Process

All submissions require review:

1. Automated checks (build, tests)
2. Code style verification
3. Functionality review
4. Performance assessment
5. Documentation completeness

## License

By contributing, you agree that your contributions will be licensed under the BSD-3-Clause License.

---

Thank you for contributing to ATenCo9!
