cmake_minimum_required(VERSION 3.29.7)

project(
    neural-network-module
    VERSION 1.0.0
    DESCRIPTION "Module for neural network operations."
    HOMEPAGE_URL "https://github.com/D4rkw42/python-gui-app-template"
    LANGUAGES CXX
)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)

# verifying MSVC Compiler
# pybind11 works better with this compiler and may generate undefined behavior without it

if(MSVC)
    message(STATUS "Compiler MSVC detected.")
else()
    message(FATAL_ERROR "This project only works correctly with MSVC compilers!")
endif()

# DISCLAIMERS
message(STATUS "This CMakeLists.txt was made to be integrated with Poetry Python, please be sure you are in this environment!")

# Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# linker

link_directories(${Python_LIBRARY_DIRS})

# sources

file(GLOB NEURAL_NETWORK_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/neural_network/*.cpp)

# compile options

add_compile_options(/W4)            # error handling
add_compile_options(/permissive-)   # error handling

# building python modules

pybind11_add_module(neural_network MODULE ${NEURAL_NETWORK_SOURCES})

set_target_properties(
    neural_network PROPERTIES
    PREFIX ""           # no "lib" prefix
    SUFFIX ".pyd"       # extension
    OUTPUT_NAME "neural_network"  # output file name

    # target output location
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/neural_network"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/neural_network"
)

# __init__.py template
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/neural_network.__init__.py.in
    ${CMAKE_SOURCE_DIR}/bin/neural_network/__init__.py
    COPYONLY
)
