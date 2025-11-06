project(Material CUDA CXX)
find_package(CUDAToolkit REQUIRED)
find_package(glm CONFIG REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=20012,1394")

set(header_files
        include/MaterialExports.hpp
        include/Material.cuh
        include/Lambertian.cuh
        include/Metal.cuh
        include/FuzzyMetalReflection.cuh
        include/Dielectric.cuh
        include/Ray.cuh
        include/HitRecord.cuh
)
set(source_files
        src/Lambertian.cu
        src/Metal.cu
        src/FuzzyMetalReflection.cu
        src/Dielectric.cu
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=20012,20014")

add_library(${PROJECT_NAME} STATIC ${source_files} ${header_files})
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart glm::glm)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_compile_definitions(${PROJECT_NAME} PRIVATE MATERIAL_EXPORTS)

#for error ptxas fatal : Unresolved extern function
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=20012")
