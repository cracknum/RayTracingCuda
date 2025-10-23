cmake_minimum_required (VERSION 3.10...3.15)

# set(CMAKE_TOOLCHAIN_FILE "F:/Workspace/github/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_TOOLCHAIN_FILE "D:/Workspace/github/vcpkg/scripts/buildsystems/vcpkg.cmake")
project ("RayTracingCuda")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_subdirectory(Render)
add_subdirectory(App)
add_subdirectory(Kernel)
