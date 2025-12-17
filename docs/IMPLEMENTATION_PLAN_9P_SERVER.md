# 9P Server Implementation Plan

## Overview

The 9P server implementation provides network-transparent access to tensors, cognitive operations, and AtomSpace resources through the Plan 9 file system protocol. This enables distributed tensor computing, remote cognitive services, and seamless integration with Inferno-style distributed systems.

## Current Status

**Existing Infrastructure**
- Client-side 9P protocol implementation in `cog9p.c` (1088 lines)
- Message encoding/decoding functions
- Basic file system operations (attach, walk, open, read, clunk)
- 8 public API functions for client operations

**Gaps**
- No server-side implementation
- No actual network listener
- No file system namespace implementation
- No tensor serving capabilities
- No authentication/authorization

## Architecture Design

### Protocol Stack

```
┌─────────────────────────────────────────────────────────┐
│              Client Applications                         │
├─────────────────────────────────────────────────────────┤
│              9P2000.cog Protocol                         │
│  Tversion | Tattach | Twalk | Topen | Tread | Twrite   │
├─────────────────────────────────────────────────────────┤
│              Network Transport                           │
│  TCP/IP | Unix Sockets | RDMA                           │
├─────────────────────────────────────────────────────────┤
│              CogInt 9P Server                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Virtual File System Namespace                   │   │
│  │  /tensor/  /atom/  /cog/  /net/  /ctl/         │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Resource Managers                               │   │
│  │  Tensor Manager | Atom Manager | Cog Manager    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Virtual Namespace Design

The server exposes a hierarchical namespace reflecting cognitive resources:

```
/
├── tensor/                    # Tensor storage and operations
│   ├── create                 # Create new tensor (write shape/dtype)
│   ├── list                   # List all tensors (read)
│   ├── <tensor_id>/           # Individual tensor
│   │   ├── data               # Raw tensor data
│   │   ├── shape              # Shape information
│   │   ├── dtype              # Data type
│   │   ├── device             # Device location
│   │   └── ops/               # Operations on this tensor
│   │       ├── add            # Element-wise addition
│   │       ├── mul            # Element-wise multiplication
│   │       ├── matmul         # Matrix multiplication
│   │       └── reshape        # Reshape operation
│   └── ops/                   # Global tensor operations
│       ├── matmul             # Matrix multiply two tensors
│       └── conv2d             # 2D convolution
├── atom/                      # AtomSpace integration
│   ├── create                 # Create new atom
│   ├── query                  # Query atoms
│   ├── <atom_handle>/         # Individual atom
│   │   ├── type               # Atom type
│   │   ├── name               # Atom name
│   │   ├── tv                 # Truth value (strength, confidence)
│   │   ├── av                 # Attention value (STI, LTI)
│   │   └── incoming           # Incoming links
│   └── spaces/                # AtomSpace instances
│       └── <space_name>/      # Named AtomSpace
├── cog/                       # Cognitive operations
│   ├── pln/                   # Probabilistic Logic Networks
│   │   ├── infer              # Run inference (write query, read results)
│   │   └── rules/             # PLN rules
│   ├── ecan/                  # Economic Attention Networks
│   │   ├── spread             # Spread attention
│   │   └── focus              # Get attention focus
│   └── pipeline/              # Cognitive pipeline
│       ├── perceive           # Perception stage
│       ├── reason             # Reasoning stage
│       └── act                # Action stage
├── net/                       # Network and distributed operations
│   ├── workers/               # Worker pool management
│   │   ├── local/             # Local workers
│   │   └── remote/            # Remote workers
│   ├── channels/              # Inferno-style channels
│   │   └── <channel_name>/    # Named channel
│   │       ├── send           # Send to channel
│   │       └── recv           # Receive from channel
│   └── tasks/                 # Task management
│       └── <task_id>/         # Individual task
└── ctl/                       # Control and configuration
    ├── version                # Server version
    ├── stats                  # Server statistics
    ├── config                 # Configuration
    └── shutdown               # Graceful shutdown
```

## Implementation Phases

### Phase 1: Core Server Infrastructure (Week 1-2)

**Objective**: Implement basic 9P server with network listener and message dispatcher

**Tasks**:

1. **Network listener**
   ```c
   // TCP server socket
   int cog9p_server_listen(Cog9PServer *srv, const char *addr, uint16_t port);
   
   // Unix domain socket
   int cog9p_server_listen_unix(Cog9PServer *srv, const char *path);
   
   // Accept connections
   int cog9p_server_accept(Cog9PServer *srv);
   ```

2. **Connection management**
   - Per-connection state tracking
   - File descriptor (fid) management
   - Session authentication
   - Concurrent connection handling

3. **Message dispatcher**
   ```c
   // Main server loop
   void cog9p_server_run(Cog9PServer *srv);
   
   // Message handlers
   typedef int (*Cog9PHandler)(Cog9PServer *srv, Cog9PConn *conn,
                                uint8_t *msg, size_t len);
   
   // Register handlers for each message type
   void cog9p_register_handler(Cog9PServer *srv, uint8_t type,
                                Cog9PHandler handler);
   ```

4. **Threading model**
   - Thread pool for connection handling
   - Lock-free message queues
   - Worker thread management

**Deliverables**:
- Functional TCP/Unix socket server
- Connection accept and management
- Message routing infrastructure
- Multi-threaded request handling

**Success Criteria**:
- Server accepts connections
- Handles Tversion/Rversion handshake
- Processes concurrent requests
- No resource leaks

### Phase 2: Virtual File System (Week 3-4)

**Objective**: Implement hierarchical namespace with file operations

**Tasks**:

1. **VFS data structures**
   ```c
   typedef struct Cog9PFile {
       uint64_t qid;              // Unique file ID
       char *name;                // File name
       uint32_t mode;             // Permissions
       uint64_t length;           // File size
       void *data;                // File-specific data
       Cog9PFileOps *ops;         // Operation handlers
       struct Cog9PFile *parent;  // Parent directory
       struct Cog9PFile **children; // Child files
       size_t n_children;
   } Cog9PFile;
   
   typedef struct Cog9PFileOps {
       int (*open)(Cog9PFile *f, uint8_t mode);
       int (*read)(Cog9PFile *f, uint8_t *buf, size_t count, uint64_t offset);
       int (*write)(Cog9PFile *f, const uint8_t *buf, size_t count, uint64_t offset);
       int (*close)(Cog9PFile *f);
       int (*stat)(Cog9PFile *f, Cog9PStat *stat);
   } Cog9PFileOps;
   ```

2. **Namespace construction**
   - Build initial directory tree
   - Register resource managers
   - Implement path resolution

3. **File operation handlers**
   - Twalk: Navigate directory hierarchy
   - Topen: Open files for reading/writing
   - Tread: Read file contents
   - Twrite: Write file contents
   - Tstat: Get file metadata
   - Tclunk: Close file handles

4. **Dynamic file generation**
   - Synthetic files (e.g., /ctl/stats)
   - Directory listings
   - Resource enumeration

**Deliverables**:
- Complete VFS implementation
- All 9P file operations working
- Dynamic namespace updates
- Permission checking

**Success Criteria**:
- Can navigate full namespace
- Read/write operations functional
- Directory listings accurate
- Proper error handling

### Phase 3: Tensor Resource Manager (Week 5-6)

**Objective**: Expose tensors through /tensor/ namespace

**Tasks**:

1. **Tensor registry**
   ```c
   typedef struct Cog9PTensorRegistry {
       CogTensor **tensors;
       uint64_t *tensor_ids;
       size_t n_tensors;
       pthread_rwlock_t lock;
   } Cog9PTensorRegistry;
   
   // Register tensor for serving
   uint64_t cog9p_register_tensor(Cog9PServer *srv, CogTensor *t);
   
   // Unregister tensor
   void cog9p_unregister_tensor(Cog9PServer *srv, uint64_t id);
   ```

2. **Tensor file operations**
   - `/tensor/create`: Write shape/dtype, returns tensor ID
   - `/tensor/<id>/data`: Read/write raw tensor data
   - `/tensor/<id>/shape`: Read shape as text
   - `/tensor/<id>/dtype`: Read data type
   - `/tensor/<id>/device`: Read/write device location

3. **Tensor operations**
   ```c
   // /tensor/<id>/ops/add
   // Write: tensor_id_to_add
   // Read: result_tensor_id
   int handle_tensor_add(Cog9PFile *f, const uint8_t *buf, size_t len);
   ```

4. **Data serialization**
   - Binary format for tensor data
   - Text format for metadata
   - Efficient large tensor transfers
   - Chunked read/write support

**Deliverables**:
- Tensor creation via 9P
- Tensor data access
- Tensor operations
- Efficient data transfer

**Success Criteria**:
- Create tensors remotely
- Read/write tensor data
- Perform operations
- Handle large tensors (> 1GB)

### Phase 4: AtomSpace Integration (Week 7)

**Objective**: Expose AtomSpace through /atom/ namespace

**Tasks**:

1. **Atom registry**
   - Map atom handles to file system
   - Track atom lifecycle
   - Handle atom deletion

2. **Atom file operations**
   - `/atom/create`: Create new atom
   - `/atom/<handle>/type`: Read atom type
   - `/atom/<handle>/name`: Read/write atom name
   - `/atom/<handle>/tv`: Read/write truth value
   - `/atom/<handle>/av`: Read/write attention value

3. **Query interface**
   - `/atom/query`: Write query pattern, read results
   - Pattern matching syntax
   - Result pagination

4. **AtomSpace management**
   - `/atom/spaces/<name>`: Named AtomSpace instances
   - Space creation/deletion
   - Space switching

**Deliverables**:
- Atom CRUD operations
- Query interface
- Multi-space support
- Truth/attention value access

**Success Criteria**:
- Create/read/update atoms
- Query AtomSpace
- Manage multiple spaces
- Proper synchronization

### Phase 5: Cognitive Operations (Week 8)

**Objective**: Expose PLN, ECAN, and cognitive pipeline

**Tasks**:

1. **PLN interface**
   ```c
   // /cog/pln/infer
   // Write: query atom handle
   // Read: result atom handles (one per line)
   int handle_pln_infer(Cog9PFile *f, const uint8_t *buf, size_t len);
   ```

2. **ECAN interface**
   - `/cog/ecan/spread`: Spread attention from source atom
   - `/cog/ecan/focus`: Get current attention focus

3. **Pipeline interface**
   - `/cog/pipeline/perceive`: Submit perception tensor
   - `/cog/pipeline/reason`: Trigger reasoning
   - `/cog/pipeline/act`: Read action output

4. **Asynchronous operations**
   - Long-running operations return task IDs
   - Poll task status
   - Retrieve results when complete

**Deliverables**:
- PLN inference via 9P
- ECAN operations
- Cognitive pipeline access
- Async operation support

**Success Criteria**:
- Run PLN inference remotely
- Spread attention
- Execute cognitive pipeline
- Handle long operations

### Phase 6: Distributed Computing (Week 9-10)

**Objective**: Implement /net/ namespace for distributed operations

**Tasks**:

1. **Worker management**
   - `/net/workers/local/`: List local workers
   - `/net/workers/remote/`: List remote workers
   - Worker registration/deregistration

2. **Channel implementation**
   ```c
   // /net/channels/<name>/send
   // Write: data to send
   int handle_channel_send(Cog9PFile *f, const uint8_t *buf, size_t len);
   
   // /net/channels/<name>/recv
   // Read: received data
   int handle_channel_recv(Cog9PFile *f, uint8_t *buf, size_t len);
   ```

3. **Task management**
   - Submit tasks via `/net/tasks/submit`
   - Monitor task status
   - Retrieve results

4. **Load balancing**
   - Automatic worker selection
   - Task distribution strategies
   - Failure handling

**Deliverables**:
- Worker pool management
- Channel communication
- Task submission/monitoring
- Load balancing

**Success Criteria**:
- Distribute tasks to workers
- Channel send/receive working
- Automatic failover
- Efficient load distribution

### Phase 7: Security and Performance (Week 11-12)

**Objective**: Add authentication, authorization, and optimization

**Tasks**:

1. **Authentication**
   - User/password authentication
   - Certificate-based auth (TLS)
   - Token-based auth (JWT)

2. **Authorization**
   - Per-file permissions
   - User/group access control
   - Capability-based security

3. **Performance optimization**
   - Zero-copy data transfers
   - Memory-mapped file serving
   - Connection pooling
   - Request batching

4. **Monitoring and logging**
   - Request logging
   - Performance metrics
   - Error tracking
   - Debug tracing

**Deliverables**:
- Authentication system
- Authorization framework
- Performance optimizations
- Monitoring infrastructure

**Success Criteria**:
- Secure authentication
- Fine-grained permissions
- 10x throughput improvement
- Comprehensive logging

## Technical Specifications

### Protocol Extensions

**9P2000.cog Extensions**:
- Custom message types for tensor operations
- Streaming support for large data
- Batch operation messages
- Subscription mechanism for events

### Performance Targets

| Metric | Target |
|--------|--------|
| Latency (small ops) | < 1ms |
| Throughput (data transfer) | > 1 GB/s |
| Concurrent connections | > 10,000 |
| Tensor operations/sec | > 100,000 |

### Security Model

**Authentication Levels**:
1. **None**: Open access (development only)
2. **Basic**: Username/password
3. **Certificate**: TLS client certificates
4. **Token**: JWT or OAuth2 tokens

**Permission Bits**:
- Read: Can read file contents
- Write: Can write file contents
- Execute: Can trigger operations
- Admin: Can modify server state

## Build System Integration

### CMake Configuration

```cmake
# Server-specific sources
set(COG9P_SERVER_SOURCES
    9p/cog9p_server.c
    9p/cog9p_vfs.c
    9p/cog9p_tensor_mgr.c
    9p/cog9p_atom_mgr.c
    9p/cog9p_cog_mgr.c
    9p/cog9p_net_mgr.c
)

# Server executable
add_executable(cog9p_server ${COG9P_SERVER_SOURCES})
target_link_libraries(cog9p_server cogint pthread)

# Optional TLS support
if(COGINT_9P_TLS)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(cog9p_server OpenSSL::SSL)
endif()
```

### Dependencies

**Required**:
- pthread for threading
- Standard C library

**Optional**:
- OpenSSL for TLS support
- libuv for async I/O
- jemalloc for memory efficiency

## Testing Strategy

### Unit Tests

**Protocol Tests**:
- Message encoding/decoding
- State machine transitions
- Error handling

**VFS Tests**:
- Path resolution
- File operations
- Permission checking

### Integration Tests

**Client-Server Tests**:
- Full protocol handshake
- File operations
- Concurrent access

**Resource Tests**:
- Tensor serving
- Atom operations
- Cognitive operations

### Performance Tests

**Benchmarks**:
- Latency measurements
- Throughput tests
- Scalability tests (1 to 10k connections)
- Memory usage profiling

### Stress Tests

**Reliability**:
- Connection churn
- Large data transfers
- Error injection
- Resource exhaustion

## Deployment Considerations

### Server Configuration

```ini
[server]
listen = tcp://0.0.0.0:564
unix_socket = /var/run/cog9p.sock
max_connections = 10000
thread_pool_size = 32

[auth]
method = certificate
cert_file = /etc/cog9p/server.crt
key_file = /etc/cog9p/server.key

[performance]
zero_copy = true
mmap_threshold = 1048576  # 1MB
batch_size = 100

[logging]
level = info
file = /var/log/cog9p/server.log
```

### Systemd Integration

```ini
[Unit]
Description=CogInt 9P Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/cog9p_server -c /etc/cog9p/server.conf
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Client Tools

### Command-Line Client

```bash
# Mount 9P server
cog9p mount tcp://server:564 /mnt/cog

# Create tensor
echo "3,4:float32" > /mnt/cog/tensor/create
TENSOR_ID=$(cat /mnt/cog/tensor/create)

# Write tensor data
cat data.bin > /mnt/cog/tensor/$TENSOR_ID/data

# Perform operation
echo "$TENSOR_ID" > /mnt/cog/tensor/ops/softmax
RESULT_ID=$(cat /mnt/cog/tensor/ops/softmax)

# Read result
cat /mnt/cog/tensor/$RESULT_ID/data > result.bin
```

### Python Client Library

```python
import cog9p

# Connect to server
client = cog9p.Client('tcp://server:564')

# Create tensor
tensor = client.tensor.create(shape=(3, 4), dtype='float32')
tensor.write(data)

# Perform operation
result = tensor.softmax()

# Read result
output = result.read()
```

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 2 weeks | Core server infrastructure |
| Phase 2 | 2 weeks | Virtual file system |
| Phase 3 | 2 weeks | Tensor resource manager |
| Phase 4 | 1 week | AtomSpace integration |
| Phase 5 | 1 week | Cognitive operations |
| Phase 6 | 2 weeks | Distributed computing |
| Phase 7 | 2 weeks | Security and performance |
| **Total** | **12 weeks** | **Complete 9P server** |

## Success Metrics

**Functionality**:
- ✅ All 9P protocol operations working
- ✅ Complete namespace accessible
- ✅ Tensor operations functional
- ✅ AtomSpace integration complete

**Performance**:
- ✅ < 1ms latency for small operations
- ✅ > 1 GB/s throughput for data transfer
- ✅ Support 10,000+ concurrent connections

**Security**:
- ✅ Authentication working
- ✅ Authorization enforced
- ✅ TLS encryption supported

**Reliability**:
- ✅ Zero crashes in 7-day stress test
- ✅ Graceful degradation under load
- ✅ Automatic recovery from failures

This implementation plan provides a comprehensive roadmap for building a production-ready 9P server that enables network-transparent access to CogInt's cognitive computing capabilities.
