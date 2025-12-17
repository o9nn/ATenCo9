cmake_minimum_required(VERSION 3.14)
project(CogInt VERSION 0.1.0 LANGUAGES C CXX)

# Options
option(COGINT_BUILD_TESTS "Build CogInt tests" ON)
option(COGINT_BUILD_EXAMPLES "Build CogInt examples" ON)
option(COGINT_ENABLE_CUDA "Enable CUDA support" OFF)
option(COGINT_ENABLE_ATEN "Enable ATen integration" ON)

# C/C++ Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fvisibility=hidden")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
endif()

# Find dependencies
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src
)

# Source files
set(COGINT_SOURCES
    # Core API
    api/cogint_api.c
    
    # 9P Protocol
    9p/cog9p.c
    
    # AtomSpace
    atomspace/cog_atomspace.c
    
    # Distributed Computing
    distributed/cog_distributed.c
    
    # Tensor Networks
    tensornet/cog_tensornet.c
    tensornet/cog_decompose.c
    tensornet/cog_distributed_tn.c
)

# Create CogInt library
add_library(cogint SHARED ${COGINT_SOURCES})
add_library(cogint_static STATIC ${COGINT_SOURCES})

# Set library properties
set_target_properties(cogint PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER "include/cogint.h;include/cog9p.h;include/cog_atomspace.h;include/cog_inferno.h"
)

set_target_properties(cogint_static PROPERTIES
    OUTPUT_NAME cogint
)

# Link dependencies
target_link_libraries(cogint
    Threads::Threads
    m
)

target_link_libraries(cogint_static
    Threads::Threads
    m
)

# ATen integration
if(COGINT_ENABLE_ATEN)
    target_compile_definitions(cogint PRIVATE COGINT_ENABLE_ATEN)
    target_compile_definitions(cogint_static PRIVATE COGINT_ENABLE_ATEN)
    
    # Link ATen if available
    if(TARGET ATen)
        target_link_libraries(cogint ATen)
        target_link_libraries(cogint_static ATen)
    endif()
endif()

# CUDA support
if(COGINT_ENABLE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        target_compile_definitions(cogint PRIVATE COGINT_ENABLE_CUDA)
        target_compile_definitions(cogint_static PRIVATE COGINT_ENABLE_CUDA)
        target_link_libraries(cogint ${CUDA_LIBRARIES})
        target_link_libraries(cogint_static ${CUDA_LIBRARIES})
    endif()
endif()

# Export symbols
target_compile_definitions(cogint PRIVATE COGINT_EXPORTS)

# Installation
include(GNUInstallDirs)

install(TARGETS cogint cogint_static
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cogint
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cogint
    FILES_MATCHING PATTERN "*.h"
)

# Tests
if(COGINT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(COGINT_BUILD_EXAMPLES)
    add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CogIntConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CogIntConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CogInt
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CogIntConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CogIntConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CogIntConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CogInt
)

# Summary
message(STATUS "")
message(STATUS "CogInt Configuration Summary")
message(STATUS "============================")
message(STATUS "Version:        ${PROJECT_VERSION}")
message(STATUS "Build tests:    ${COGINT_BUILD_TESTS}")
message(STATUS "Build examples: ${COGINT_BUILD_EXAMPLES}")
message(STATUS "CUDA support:   ${COGINT_ENABLE_CUDA}")
message(STATUS "ATen support:   ${COGINT_ENABLE_ATEN}")
message(STATUS "")
