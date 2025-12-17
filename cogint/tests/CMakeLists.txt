# CogInt Tests

# Test executable for basic functionality
add_executable(test_cogint_basic test_basic.c)
target_link_libraries(test_cogint_basic cogint)
add_test(NAME test_basic COMMAND test_cogint_basic)

# Test for tensor operations
add_executable(test_tensor test_tensor.c)
target_link_libraries(test_tensor cogint)
add_test(NAME test_tensor COMMAND test_tensor)

# Test for 9P protocol
add_executable(test_9p test_9p.c)
target_link_libraries(test_9p cogint)
add_test(NAME test_9p COMMAND test_9p)

# Test for AtomSpace integration
add_executable(test_atomspace test_atomspace.c)
target_link_libraries(test_atomspace cogint)
add_test(NAME test_atomspace COMMAND test_atomspace)

# Test for distributed operations
add_executable(test_distributed test_distributed.c)
target_link_libraries(test_distributed cogint)
add_test(NAME test_distributed COMMAND test_distributed)

# Test for tensor networks
add_executable(test_tensornet test_tensornet.c)
target_link_libraries(test_tensornet cogint)
add_test(NAME test_tensornet COMMAND test_tensornet)
