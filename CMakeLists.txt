cmake_minimum_required(VERSION 3.16)

project(acyclic_cable_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SCIP_ROOT "$ENV{SCIP_ROOT}" CACHE PATH "Path to the SCIP Optimization Suite source tree")
if(NOT SCIP_ROOT)
    set(SCIP_ROOT "/home/duweiyue25/SCIP/scipoptsuite-8.0.4")
endif()

find_path(SCIP_INCLUDE_DIR
    NAMES scip/scip.h
    PATHS "${SCIP_ROOT}/scip/src"
    NO_DEFAULT_PATH
)
find_path(SCIP_CONFIG_INCLUDE_DIR
    NAMES scip/config.h
    PATHS "${SCIP_ROOT}/build/scip"
    NO_DEFAULT_PATH
)
find_library(SCIP_LIBRARY
    NAMES scip libscip
    PATHS "${SCIP_ROOT}/build/lib"
    NO_DEFAULT_PATH
)

if(NOT SCIP_INCLUDE_DIR OR NOT SCIP_CONFIG_INCLUDE_DIR OR NOT SCIP_LIBRARY)
    message(FATAL_ERROR "SCIP headers/library not found. Set SCIP_ROOT to the SCIP 8.0.4 source tree.")
endif()

set(TORCH_ROOT "$ENV{TORCH_ROOT}" CACHE PATH "Path to the installed PyTorch package")
if(NOT TORCH_ROOT)
    set(TORCH_ROOT "/home/duweiyue25/conda/envs/rl4scip/lib/python3.11/site-packages/torch")
endif()

# The rl4scip PyTorch wheel includes CUDA runtime libraries but the server does
# not provide nvcc. Link the packaged LibTorch libraries directly so CMake does
# not require a CUDA compiler for this C++-only target.
set(TORCH_INCLUDE_DIR "${TORCH_ROOT}/include")
set(TORCH_API_INCLUDE_DIR "${TORCH_ROOT}/include/torch/csrc/api/include")
set(TORCH_LIBRARY_DIR "${TORCH_ROOT}/lib")
foreach(TORCH_COMPONENT torch torch_cpu torch_cuda c10_cuda c10)
    find_library(TORCH_${TORCH_COMPONENT}_LIBRARY
        NAMES ${TORCH_COMPONENT}
        PATHS "${TORCH_LIBRARY_DIR}"
        NO_DEFAULT_PATH
    )
    if(NOT TORCH_${TORCH_COMPONENT}_LIBRARY)
        message(FATAL_ERROR "Missing LibTorch component ${TORCH_COMPONENT} under ${TORCH_LIBRARY_DIR}")
    endif()
    list(APPEND TORCH_LIBRARIES "${TORCH_${TORCH_COMPONENT}_LIBRARY}")
endforeach()

if(NOT EXISTS "${TORCH_INCLUDE_DIR}/torch/script.h")
    message(FATAL_ERROR "LibTorch headers not found. Set TORCH_ROOT to the installed PyTorch package.")
endif()

add_executable(scip_tree
    code/scip_tree.cpp
    src/rl/rl_branchrule.cpp
    src/rl/scip_feature_extractor.cpp
    src/rl/model_runner.cpp
    src/rl/rl_mlp_branchrule.cpp
    src/rl/scip_graph_feature_extractor.cpp
    src/rl/gcnn_model_runner.cpp
    src/rl/rl_gcnn_branchrule.cpp
    src/rl/prim_bias.cpp
    src/rl/scip_profile.cpp
)
target_include_directories(scip_tree PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SCIP_INCLUDE_DIR}"
    "${SCIP_CONFIG_INCLUDE_DIR}"
    "${TORCH_INCLUDE_DIR}"
    "${TORCH_API_INCLUDE_DIR}"
)
target_link_libraries(scip_tree PRIVATE "${SCIP_LIBRARY}" "${TORCH_LIBRARIES}")
target_compile_definitions(scip_tree PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_target_properties(scip_tree PROPERTIES BUILD_RPATH "${SCIP_ROOT}/build/lib;${TORCH_LIBRARY_DIR}")
target_compile_options(scip_tree PRIVATE -Wall -Wextra)

add_executable(test_custom_branchrule
    tests/test_custom_branchrule.cpp
    src/rl/rl_branchrule.cpp
    src/rl/scip_feature_extractor.cpp
)
target_include_directories(test_custom_branchrule PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SCIP_INCLUDE_DIR}"
    "${SCIP_CONFIG_INCLUDE_DIR}"
)
target_link_libraries(test_custom_branchrule PRIVATE "${SCIP_LIBRARY}")
target_compile_options(test_custom_branchrule PRIVATE -Wall -Wextra)

add_executable(model_runner_parity
    tests/model_runner_parity.cpp
    src/rl/model_runner.cpp
)
target_include_directories(model_runner_parity PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SCIP_INCLUDE_DIR}"
    "${SCIP_CONFIG_INCLUDE_DIR}"
    "${TORCH_INCLUDE_DIR}"
    "${TORCH_API_INCLUDE_DIR}"
)
target_link_libraries(model_runner_parity PRIVATE "${TORCH_LIBRARIES}")
target_compile_definitions(model_runner_parity PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_target_properties(model_runner_parity PROPERTIES BUILD_RPATH "${TORCH_LIBRARY_DIR}")
target_compile_options(model_runner_parity PRIVATE -Wall -Wextra)

add_executable(gcnn_model_runner_parity
    tests/gcnn_model_runner_parity.cpp
    src/rl/gcnn_model_runner.cpp
)
target_include_directories(gcnn_model_runner_parity PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SCIP_INCLUDE_DIR}"
    "${SCIP_CONFIG_INCLUDE_DIR}"
    "${TORCH_INCLUDE_DIR}"
    "${TORCH_API_INCLUDE_DIR}"
)
target_link_libraries(gcnn_model_runner_parity PRIVATE "${TORCH_LIBRARIES}")
target_compile_definitions(gcnn_model_runner_parity PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_target_properties(gcnn_model_runner_parity PROPERTIES BUILD_RPATH "${TORCH_LIBRARY_DIR}")
target_compile_options(gcnn_model_runner_parity PRIVATE -Wall -Wextra)

add_executable(graph_probe
    tools/graph_probe.cpp
    src/rl/scip_profile.cpp
    src/rl/graph_size.cpp
)
target_include_directories(graph_probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SCIP_INCLUDE_DIR}"
    "${SCIP_CONFIG_INCLUDE_DIR}"
)
target_link_libraries(graph_probe PRIVATE "${SCIP_LIBRARY}")
target_compile_options(graph_probe PRIVATE -Wall -Wextra)
