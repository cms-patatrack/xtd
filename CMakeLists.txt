cmake_minimum_required(VERSION 3.16.0)

project(xtd VERSION 0.1 LANGUAGES CXX)


# Define the header-only target
add_library(xtd INTERFACE)
target_include_directories(xtd INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Installation instructions
install(DIRECTORY include/ DESTINATION include)

install(TARGETS xtd EXPORT xtdTargets)
install(EXPORT xtdTargets
    FILE xtdConfig.cmake
    DESTINATION lib/cmake/xtd
)

