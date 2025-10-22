project(Render)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets OpenGL)

set(source_files
	src/RenderWindow.cpp
	src/Image.cpp
)

set(header_files
	include/RenderWindow.hpp
	include/Image.hpp
	include/Element.hpp
	include/RenderExports.hpp
)

set(shader_files
	shader/vertexShader.vert
	shader/fragmentShader.frag
)

add_library(${PROJECT_NAME} SHARED ${source_files} ${header_files} ${shader_files})
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Core Qt5::Widgets Qt5::OpenGL)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE shader)
target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER_EXPORTS)
target_compile_definitions(${PROJECT_NAME} PRIVATE SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shader")

source_group("Shader" REGULAR_EXPRESSION  "shader/*")

