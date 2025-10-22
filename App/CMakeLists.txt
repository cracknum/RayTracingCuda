project(App)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets OpenGL)
find_package(Stb REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Core Qt5::Widgets Qt5::OpenGL Render OpenMP::OpenMP_CXX)
target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})
