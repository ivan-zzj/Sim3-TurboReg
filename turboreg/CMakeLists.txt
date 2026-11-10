# turboreg lib
add_library(turboreg SHARED
    src/turboreg.cpp
    src/rigid_transform.cpp
    src/similarity_transform.cpp
    src/core_turboreg_gpu.cpp
    src/utils_pcr.cpp
    src/model_selection.cpp
    src/scale_turboreg.cpp
    src/sim3.cpp
)
target_include_directories(turboreg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(turboreg PUBLIC "${TORCH_LIBRARIES}" Eigen3::Eigen)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(turboreg PUBLIC ${TORCH_CXX_FLAGS_LIST})

option(TURBOREG_CUDA_KERNELS "Build fused Sim3 CUDA kernels (requires a toolkit matching LibTorch)" OFF)
if(TURBOREG_CUDA_KERNELS)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(turboreg PRIVATE src/sim3_cuda.cu)
    target_compile_definitions(turboreg PRIVATE TURBOREG_WITH_CUDA_KERNELS=1)
    target_compile_options(turboreg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    set_target_properties(turboreg PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(turboreg PRIVATE CUDA::cudart)
endif()

# Standalone small-cloud CPU backend; the legacy Torch/GPU target is unchanged.
add_library(turboreg_native_core STATIC src/sim3_native.cpp)
target_include_directories(turboreg_native_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(turboreg_native_core PUBLIC Eigen3::Eigen)
target_compile_definitions(turboreg_native_core PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(turboreg_native_core PUBLIC ${TORCH_CXX_FLAGS_LIST})
