project(Kernel CUDA CXX)
find_package(CUDAToolkit REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS OpenGL)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=20012,1394")

set(header_files
    include/Macro.h
    include/ImageMaker.h
    include/KernelExports.hpp
    include/Ray.cuh
    include/RayTracer.cuh
    include/Hitable.cuh
    include/Sphere.cuh
    include/HitableList.cuh
    include/Camera.cuh
    include/Dispatcher.hpp
    include/Observer.hpp
)
set(source_files 
    src/ImageMaker.cu
    src/RayTracer.cu
    src/Sphere.cu
    src/HitableList.cu
    src/Camera.cu
    src/Dispatcher.cpp
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=20012,20014")

add_library(${PROJECT_NAME} SHARED ${source_files} ${header_files})
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart glm::glm Qt5::OpenGL)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_compile_definitions(${PROJECT_NAME} PRIVATE KERNEL_EXPORTS)

#for error ptxas fatal : Unresolved extern function
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=20012")
