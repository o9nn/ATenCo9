# CogInt: Cognitive Integration Layer for ATenCo9

**CogInt** is a C-based integration layer designed to bridge the worlds of high-performance tensor computing, distributed systems, and cognitive architectures. It serves as the core of the **ATenCo9** project, creating a cohesive platform for developing and experimenting with Artificial General Intelligence (AGI) systems.

The library provides a unified API that integrates:

1.  **ATen/PyTorch Tensor Computing**: Leverages the power of ATen for efficient, low-level tensor operations on both CPU and CUDA devices.
2.  **Plan 9/Inferno Distributed Systems**: Implements a 9P-based protocol (`9P2000.cog`) for network-transparent access to tensors and cognitive services. It also incorporates Inferno-style typed channels for robust, concurrent programming.
3.  **OpenCog Cognitive Architectures**: Provides deep integration with OpenCog's AtomSpace, enabling tensors to be treated as first-class citizens in a symbolic knowledge hypergraph. This allows for the application of cognitive services like Probabilistic Logic Networks (PLN) and the Economic Attention Network (ECAN) directly to tensor data.

## Architecture Overview

The CogInt architecture is designed as a modular, multi-layered system that separates concerns while enabling seamless communication between components.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CogInt Unified API Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Cognitive Pipeline  │  Tensor Operations  │  Distributed Ops   │
├─────────────────────────────────────────────────────────────────┤
│        AtomSpace Tensor Bindings (atomspace/)                   │
│        9P Tensor Protocol (9p/)                                  │
│        Inferno-style Concurrency (distributed/)                  │
├─────────────────────────────────────────────────────────────────┤
│                    ATen Core Tensor Library                      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Directory | Description |
|---|---|---|
| **Unified API** | `api/` | Provides the main `cogint_api.c` and public headers for interacting with the library. |
| **9P Protocol** | `9p/` | Implements the `9P2000.cog` protocol for distributing tensors and cognitive operations across a network. |
| **AtomSpace Bindings** | `atomspace/` | Integrates `CogTensor` objects directly into the OpenCog AtomSpace, enabling neural-symbolic reasoning. |
| **Distributed Framework**| `distributed/` | Implements Inferno-style typed channels, a worker pool, and a task graph system for concurrent and distributed computation. |
| **Include Files** | `include/` | Contains all public header files (`cogint.h`, `cog9p.h`, `cog_atomspace.h`, `cog_inferno.h`). |

## Features

- **Unified Tensor Representation**: The `CogTensor` struct provides a unified representation for tensors that can exist on a CPU, a CUDA device, within the AtomSpace, or be distributed across a network.
- **Distributed Tensors via 9P**: Tensors can be exposed as files in a Plan 9-style namespace, allowing any 9P-compliant client to access and manipulate them over the network.
- **Neural-Symbolic Integration**: By embedding tensors in the AtomSpace, CogInt allows for the seamless application of symbolic reasoning (PLN) and attention allocation (ECAN) to neural network components.
- **Inferno-style Concurrency**: The library provides typed channels (`CogChan`) and a `select`-like mechanism (`CogAlt`) for building complex, concurrent data processing pipelines, inspired by the Limbo programming language.
- **Distributed Task Execution**: A built-in worker pool and task graph system enable the execution of tensor and cognitive operations across a distributed network of workers.
- **Cognitive Pipeline**: A high-level API for constructing perception-reasoning-action cycles, allowing for the creation of autonomous cognitive agents.

## Building CogInt

CogInt uses CMake for its build system. The following steps will build the library from the `cogint` directory.

### Prerequisites

- A C/C++ compiler (GCC or Clang)
- CMake (version 3.14 or later)
- `pthread` library

### Build Steps

1.  **Create a build directory:**

    ```bash
    mkdir build
    cd build
    ```

2.  **Run CMake to configure the project:**

    ```bash
    cmake ..
    ```

    You can enable optional features with the following flags:
    - `-DCOGINT_BUILD_TESTS=ON`: Build the test suite.
    - `-DCOGINT_ENABLE_CUDA=ON`: Enable CUDA support (requires CUDA toolkit).

3.  **Build the library:**

    ```bash
    make
    ```

4.  **Install the library (optional):**

    ```bash
    sudo make install
    ```

This will build the shared library (`libcogint.so`) and a static library (`libcogint.a`) and place them in the `build` directory. If you install, the libraries and header files will be placed in the appropriate system directories.

## Usage

To use CogInt in your own project, you need to include the main header and link against the library.

```c
#include <cogint/cogint.h>

int main() {
    // Initialize the CogInt context
    CogContext *ctx = cogint_init();
    if (!ctx) {
        return 1;
    }

    // Create a tensor
    int64_t shape[] = {2, 3};
    CogTensor *t = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);

    // ... perform operations ...

    // Clean up
    cog_tensor_free(t);
    cogint_shutdown(ctx);

    return 0;
}
```

Link your application with `-lcogint`.

## Future Work

- Complete the C++ bridge to the core ATen library for direct, in-memory tensor manipulation.
- Implement the server-side logic for remote channel communication.
- Expand the set of supported tensor operations in the 9P protocol.
- Add more sophisticated PLN rules for tensor reasoning.
