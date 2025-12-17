# CogInt Examples

# Simple tensor example
add_executable(example_simple_tensor simple_tensor.c)
target_link_libraries(example_simple_tensor cogint)

# Cognitive pipeline example
add_executable(example_cognitive_pipeline cognitive_pipeline.c)
target_link_libraries(example_cognitive_pipeline cogint)

# Distributed tensor example
add_executable(example_distributed_tensor distributed_tensor.c)
target_link_libraries(example_distributed_tensor cogint)

# Tensor network decomposition example
add_executable(example_tensornet tensornet_decomposition.c)
target_link_libraries(example_tensornet cogint)
