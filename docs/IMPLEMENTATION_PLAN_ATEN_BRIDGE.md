# ATen C++ Bridge Implementation Plan

## Overview

The ATen C++ bridge provides direct, in-memory integration between CogInt's C API and PyTorch's ATen tensor library. This enables zero-copy tensor operations, native CUDA support, and access to ATen's extensive operator library while maintaining CogInt's cognitive architecture integration.

## Current Status

**Existing Infrastructure**
- C API with `CogTensor` abstraction in `cogint_api.c`
- Stub functions `cog_tensor_from_aten()` and `cog_tensor_to_aten()`
- ATen headers available in `aten/src/ATen/`
- CMake flag `COGINT_ENABLE_ATEN` for conditional compilation

**Gaps**
- No actual C++ bridge implementation
- No ATen tensor wrapping or unwrapping
- No integration with ATen's autograd system
- No access to ATen's operator library

## Architecture Design

### Layer Structure

```
┌─────────────────────────────────────────────────────────┐
│              CogInt C API (cogint_api.c)                │
├─────────────────────────────────────────────────────────┤
│          C++ Bridge Layer (cogint_aten_bridge.cpp)      │
│  - Tensor conversion (CogTensor <-> at::Tensor)         │
│  - Memory management (shared buffers)                   │
│  - Device synchronization                               │
├─────────────────────────────────────────────────────────┤
│              ATen C++ API (at::Tensor)                  │
│  - Operator library                                     │
│  - Autograd engine                                      │
│  - Device management (CPU/CUDA)                         │
└─────────────────────────────────────────────────────────┘
```

### Key Components

**cogint_aten_bridge.cpp** - C++ implementation file
- Tensor conversion functions
- Memory sharing mechanisms
- Device transfer operations
- Operator wrappers

**cogint_aten_bridge.h** - C-compatible header
- External C linkage declarations
- Opaque pointer types for C++ objects
- Error code definitions

**cogint_aten_ops.cpp** - ATen operator wrappers
- Mathematical operations
- Linear algebra
- Neural network primitives
- Reduction operations

## Implementation Phases

### Phase 1: Basic Tensor Conversion (Week 1-2)

**Objective**: Enable bidirectional conversion between `CogTensor` and `at::Tensor`

**Tasks**:

1. **Create bridge source files**
   - `cogint/aten_bridge/cogint_aten_bridge.cpp`
   - `cogint/aten_bridge/cogint_aten_bridge.h`
   - Update `cogint/CMakeLists.txt` to compile C++ sources

2. **Implement tensor wrapping**
   ```cpp
   // Wrap existing CogTensor data as at::Tensor (zero-copy)
   at::Tensor* cog_tensor_to_aten_impl(CogTensor *cog_t) {
       // Map CogDType to at::ScalarType
       // Create at::Tensor from external data pointer
       // Set proper strides and storage
       // Return wrapped tensor
   }
   ```

3. **Implement tensor unwrapping**
   ```cpp
   // Create CogTensor from at::Tensor (zero-copy when possible)
   CogTensor* cog_tensor_from_aten_impl(at::Tensor *aten_t) {
       // Extract shape, dtype, device
       // Share underlying storage
       // Create CogTensor wrapper
       // Handle reference counting
   }
   ```

4. **Memory management**
   - Implement shared ownership via reference counting
   - Handle ATen's storage lifecycle
   - Ensure proper cleanup on both sides

**Deliverables**:
- Functional `cog_tensor_from_aten()` and `cog_tensor_to_aten()`
- Unit tests for conversion
- Memory leak verification (valgrind)

**Success Criteria**:
- Zero-copy conversion for CPU tensors
- Correct shape and dtype mapping
- No memory leaks in conversion cycle

### Phase 2: Device Management (Week 3)

**Objective**: Support CPU and CUDA device transfers

**Tasks**:

1. **Device abstraction**
   ```cpp
   // Map CogDeviceType to at::Device
   at::Device cog_device_to_aten(CogDeviceType dev);
   CogDeviceType aten_device_to_cog(at::Device dev);
   ```

2. **Device transfer operations**
   ```cpp
   // Transfer tensor between devices
   CogTensor* cog_tensor_to_device(CogTensor *t, CogDeviceType target);
   ```

3. **CUDA stream integration**
   - Synchronize with ATen's CUDA streams
   - Handle asynchronous operations
   - Implement stream-aware conversions

**Deliverables**:
- Device transfer functions
- CUDA stream synchronization
- Device placement tests

**Success Criteria**:
- Seamless CPU ↔ CUDA transfers
- Proper stream synchronization
- No device memory leaks

### Phase 3: Operator Integration (Week 4-5)

**Objective**: Expose ATen operators through CogInt API

**Tasks**:

1. **Arithmetic operators**
   ```cpp
   // Implement using ATen operators
   CogTensor* cog_tensor_add_impl(CogTensor *a, CogTensor *b) {
       at::Tensor *at_a = to_aten(a);
       at::Tensor *at_b = to_aten(b);
       at::Tensor result = at_a->add(*at_b);
       return from_aten(&result);
   }
   ```

2. **Linear algebra**
   - Matrix multiplication (at::matmul)
   - Decompositions (SVD, QR, Cholesky)
   - Eigenvalue computations

3. **Neural network primitives**
   - Convolutions (at::conv2d)
   - Pooling operations
   - Activation functions (ReLU, softmax, etc.)
   - Batch normalization

4. **Reduction operations**
   - Sum, mean, std, var
   - Min, max, argmin, argmax
   - Custom reductions

**Deliverables**:
- 50+ ATen operator wrappers
- Comprehensive operator tests
- Performance benchmarks

**Success Criteria**:
- All basic operations functional
- Performance within 5% of native ATen
- Correct gradient computation (if autograd enabled)

### Phase 4: Autograd Integration (Week 6)

**Objective**: Enable automatic differentiation through ATen's autograd

**Tasks**:

1. **Gradient tracking**
   ```cpp
   // Enable gradient computation
   void cog_tensor_requires_grad(CogTensor *t, bool requires_grad);
   bool cog_tensor_is_leaf(CogTensor *t);
   ```

2. **Backward pass**
   ```cpp
   // Compute gradients
   void cog_tensor_backward(CogTensor *t, CogTensor *grad);
   CogTensor* cog_tensor_grad(CogTensor *t);
   ```

3. **Gradient accumulation**
   - Handle in-place operations
   - Manage gradient buffers
   - Support higher-order derivatives

**Deliverables**:
- Autograd-enabled tensor operations
- Gradient computation tests
- Backpropagation examples

**Success Criteria**:
- Correct gradient computation
- Support for complex computation graphs
- Memory-efficient gradient storage

### Phase 5: Advanced Features (Week 7-8)

**Objective**: Integrate advanced ATen capabilities

**Tasks**:

1. **Custom operators**
   - Register custom C++ operators with ATen
   - Enable JIT compilation
   - Support operator fusion

2. **Memory optimization**
   - Implement tensor views and slicing
   - Support in-place operations
   - Enable memory pooling

3. **Distributed tensors**
   - Integrate with ATen's distributed backend
   - Support tensor sharding
   - Enable collective operations (all-reduce, etc.)

4. **Quantization support**
   - Int8/Int16 quantized tensors
   - Dynamic quantization
   - Quantization-aware training

**Deliverables**:
- Custom operator framework
- Memory optimization utilities
- Distributed tensor support
- Quantization primitives

**Success Criteria**:
- Custom operators functional
- 30% memory reduction through views
- Multi-GPU tensor operations working

## Technical Specifications

### Data Type Mapping

| CogDType | at::ScalarType | Size |
|----------|----------------|------|
| COG_DTYPE_FLOAT32 | at::kFloat | 4 bytes |
| COG_DTYPE_FLOAT64 | at::kDouble | 8 bytes |
| COG_DTYPE_INT32 | at::kInt | 4 bytes |
| COG_DTYPE_INT64 | at::kLong | 8 bytes |
| COG_DTYPE_INT8 | at::kChar | 1 byte |
| COG_DTYPE_UINT8 | at::kByte | 1 byte |
| COG_DTYPE_BOOL | at::kBool | 1 byte |
| COG_DTYPE_FLOAT16 | at::kHalf | 2 bytes |

### Device Mapping

| CogDeviceType | at::Device |
|---------------|------------|
| COG_DEVICE_CPU | at::kCPU |
| COG_DEVICE_CUDA | at::kCUDA |
| COG_DEVICE_DISTRIBUTED | at::kCPU (with RPC) |
| COG_DEVICE_ATOMSPACE | at::kCPU (special) |

### Memory Layout

**Shared Storage Strategy**:
- Use ATen's `from_blob()` for zero-copy wrapping
- Maintain reference counting for shared ownership
- Implement custom deleters for CogTensor cleanup

**Alignment Requirements**:
- CPU: 64-byte alignment for SIMD
- CUDA: 256-byte alignment for coalesced access
- Use `posix_memalign()` or ATen's allocators

### Error Handling

**Exception Translation**:
```cpp
extern "C" CogTensor* cog_tensor_operation(...) {
    try {
        // ATen operation
        return result;
    } catch (const std::exception& e) {
        // Log error
        // Set error code
        return nullptr;
    }
}
```

**Error Codes**:
- `COG_ERR_ATEN_EXCEPTION` - ATen threw exception
- `COG_ERR_DEVICE_MISMATCH` - Incompatible devices
- `COG_ERR_DTYPE_MISMATCH` - Incompatible data types
- `COG_ERR_SHAPE_MISMATCH` - Incompatible shapes

## Build System Integration

### CMake Configuration

```cmake
# In cogint/CMakeLists.txt

if(COGINT_ENABLE_ATEN)
    # Find ATen
    find_package(Torch REQUIRED)
    
    # Add C++ sources
    set(ATEN_BRIDGE_SOURCES
        aten_bridge/cogint_aten_bridge.cpp
        aten_bridge/cogint_aten_ops.cpp
    )
    
    # Add to library
    target_sources(cogint PRIVATE ${ATEN_BRIDGE_SOURCES})
    target_link_libraries(cogint ${TORCH_LIBRARIES})
    target_compile_features(cogint PRIVATE cxx_std_17)
    
    # Set C++ standard
    set_source_files_properties(
        ${ATEN_BRIDGE_SOURCES}
        PROPERTIES COMPILE_FLAGS "-std=c++17"
    )
endif()
```

### Dependencies

**Required**:
- PyTorch/ATen (>= 1.9.0)
- C++17 compiler
- CUDA Toolkit (>= 11.0) for GPU support

**Optional**:
- cuDNN for optimized neural network operations
- NCCL for multi-GPU communication

## Testing Strategy

### Unit Tests

**Conversion Tests** (`test_aten_conversion.cpp`):
- Round-trip conversion (CogTensor → ATen → CogTensor)
- Data integrity verification
- Memory leak detection

**Operator Tests** (`test_aten_operators.cpp`):
- Numerical correctness against reference implementations
- Gradient computation verification
- Performance benchmarks

**Device Tests** (`test_aten_devices.cpp`):
- CPU ↔ CUDA transfers
- Multi-GPU operations
- Stream synchronization

### Integration Tests

**Neural Network Training** (`test_aten_training.cpp`):
- Simple MLP training
- Convolutional network inference
- Gradient descent optimization

**Distributed Operations** (`test_aten_distributed.cpp`):
- Multi-GPU data parallelism
- Tensor sharding
- Collective operations

### Performance Benchmarks

**Metrics**:
- Conversion overhead (< 1% for large tensors)
- Operator performance (within 5% of native ATen)
- Memory overhead (< 10% for wrapped tensors)
- CUDA kernel launch latency

## Migration Path

### Backward Compatibility

**Preserve existing API**:
- Keep current C API unchanged
- Add new functions with `_aten` suffix
- Provide compatibility layer

**Gradual adoption**:
- Phase 1: Optional ATen backend
- Phase 2: Default to ATen when available
- Phase 3: Deprecate pure C implementations

### Documentation

**API Documentation**:
- Doxygen comments for all functions
- Usage examples for common operations
- Performance tuning guidelines

**Migration Guide**:
- Converting existing code to use ATen backend
- Performance optimization tips
- Troubleshooting common issues

## Risk Mitigation

### Technical Risks

**ABI Compatibility**:
- Risk: ATen C++ ABI changes between versions
- Mitigation: Version checks, compatibility layer

**Memory Management**:
- Risk: Reference counting bugs, memory leaks
- Mitigation: Extensive testing, valgrind, ASAN

**Performance Overhead**:
- Risk: Conversion overhead negates ATen benefits
- Mitigation: Zero-copy design, benchmarking

### Organizational Risks

**Dependency Management**:
- Risk: Large PyTorch dependency
- Mitigation: Optional compilation, static linking option

**Maintenance Burden**:
- Risk: Keeping up with ATen API changes
- Mitigation: Version pinning, CI/CD testing

## Success Metrics

**Functionality**:
- ✅ 100% of basic tensor operations working
- ✅ Autograd support functional
- ✅ CUDA acceleration enabled

**Performance**:
- ✅ < 1% overhead for large tensors (> 1MB)
- ✅ Within 5% of native ATen performance
- ✅ Zero-copy conversion for 90% of cases

**Quality**:
- ✅ Zero memory leaks in continuous testing
- ✅ 95% code coverage in tests
- ✅ Comprehensive documentation

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 2 weeks | Basic tensor conversion |
| Phase 2 | 1 week | Device management |
| Phase 3 | 2 weeks | Operator integration |
| Phase 4 | 1 week | Autograd support |
| Phase 5 | 2 weeks | Advanced features |
| **Total** | **8 weeks** | **Complete ATen bridge** |

## Next Steps

1. Set up development environment with PyTorch
2. Create `aten_bridge/` directory structure
3. Implement Phase 1 basic conversion
4. Write comprehensive tests
5. Benchmark and optimize
6. Document and release

This implementation plan provides a clear path to full ATen integration while maintaining CogInt's architectural vision and ensuring high code quality throughout the development process.
