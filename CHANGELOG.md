# Changelog

All notable changes to the ATenCo9 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-12-19

### Fixed
- Include path configuration in tests and examples CMakeLists.txt
- Missing tensor operation function implementations (fill, reshape, ndim, size, dtype, data)
- Missing 9P protocol stub implementations (walk, tensor_create, tensor_read, tensor_write)
- Symbol visibility for product() utility function in tensor network module
- Function signature mismatch for cog9p_tensor_create()

### Added
- cog_tensor_fill() - Fill tensor with scalar value, supports all numeric types
- cog_tensor_reshape() - Reshape tensor with dimension validation
- cog_tensor_ndim() - Get number of tensor dimensions
- cog_tensor_size() - Get size of specific dimension
- cog_tensor_dtype() - Get tensor data type
- cog_tensor_data() - Get raw data pointer
- Stub implementations for 9P protocol client functions
- CURRENT_ISSUES.md - Detailed analysis of identified issues
- REPAIR_OPTIMIZATION_REPORT.md - Comprehensive repair and optimization documentation

### Improved
- Build system configuration with proper include directories
- Error handling consistency across all new functions
- Type safety in tensor operations with proper conversions
- Memory management patterns in reshape and fill operations
- Test coverage with successful execution of all test programs

## [Previous Unreleased]

### Added
- Comprehensive repository documentation (README.md, CONTRIBUTING.md)
- Analysis report documenting repository structure and issues
- Complete test suite infrastructure with 6 test programs
- Example programs demonstrating key features
- CMake package configuration template
- Build system fixes and improvements

### Fixed
- **Build System Errors**
  - Created missing `tests/` directory and CMakeLists.txt
  - Created missing `examples/` directory and CMakeLists.txt
  - Created missing `cmake/CogIntConfig.cmake.in` template
  
- **Compilation Errors**
  - Fixed duplicate `CogAtomType` enum definition conflict between cogint.h and cog_atomspace.h
  - Fixed comment syntax in cog_inferno.h (removed `/*` within comments)
  - Fixed pointer type mismatch in `cog_pln_infer` function
  - Fixed pointer type in `cog_pipeline_reason` function
  - Added forward declaration for `CogRuntime` in cog_tensornet.h
  - Fixed `CogRuntime` typedef conflict in cog_inferno.h

- **Code Quality**
  - Resolved enum conversion warnings (documented, acceptable for cross-module integration)
  - Addressed unused parameter warnings (kept for API consistency)

### Changed
- Reorganized header file dependencies for cleaner compilation
- Improved type safety in cognitive operation functions
- Enhanced build configuration with better error messages

## [0.1.0] - 2025-12-17

### Initial Release

#### Core Features

**Tensor Computing**
- Basic tensor creation and manipulation API
- Support for multiple data types (float32, float64, int32, int64)
- Multi-dimensional tensor operations
- Device abstraction (CPU, CUDA, AtomSpace, Distributed)

**9P Protocol Integration**
- 9P2000.cog protocol implementation
- Network-transparent tensor access
- File system interface for tensors
- Client-server architecture

**AtomSpace Integration**
- Tensor nodes in OpenCog AtomSpace
- Neural-symbolic reasoning support
- PLN (Probabilistic Logic Networks) scaffolding
- ECAN (Economic Attention Networks) framework
- Truth value and attention value support

**Distributed Computing**
- Inferno-style typed channels
- Worker pool implementation
- Task graph system
- Concurrent processing primitives
- CSP-style communication patterns

**Tensor Network Methods**
- SVD decomposition
- QR decomposition
- Tensor Train (TT) representation
- Tucker decomposition
- HOSVD (Higher-Order SVD)
- Distributed contraction algorithms
- Memory-efficient sliced contraction

**Cognitive Pipeline**
- Perception-reasoning-action cycle framework
- Multi-channel communication
- Attention-driven processing
- Learning integration

#### Architecture

**Modules**
- `api/` - Unified C API (cogint_api.c)
- `9p/` - 9P protocol implementation (cog9p.c)
- `atomspace/` - AtomSpace integration (cog_atomspace.c)
- `distributed/` - Distributed computing (cog_distributed.c)
- `tensornet/` - Tensor network methods (cog_tensornet.c, cog_decompose.c, cog_distributed_tn.c)
- `include/` - Public headers (cogint.h, cog9p.h, cog_atomspace.h, cog_inferno.h, cog_tensornet.h)

**Build System**
- CMake-based build configuration
- Shared and static library generation
- Optional CUDA support
- Optional ATen integration
- Test suite support
- Example programs support

#### Known Limitations

- C++ bridge to ATen not fully implemented
- Server-side 9P logic incomplete
- Limited PLN rule set
- CUDA support not yet enabled
- Some functions are stubs awaiting implementation

#### Dependencies

- C11 compiler (GCC 11+ or Clang)
- CMake 3.14+
- pthread library
- (Optional) CUDA toolkit

---

## Version History

- **v0.1.0** (2025-12-17) - Initial release with core infrastructure
- **Unreleased** - Build fixes, documentation, and test suite

## Future Roadmap

### v0.2.0 (Planned)
- Complete ATen C++ bridge
- Full 9P server implementation
- Expanded PLN rules
- CUDA acceleration
- Comprehensive test coverage

### v0.3.0 (Planned)
- Performance optimizations
- Advanced tensor network algorithms
- Distributed attention mechanisms
- Graph neural network integration

### v1.0.0 (Future)
- Production-ready release
- Full feature completeness
- Extensive documentation
- Performance benchmarks
- Stability guarantees

---

For detailed commit history, see: https://github.com/o9nn/ATenCo9/commits
