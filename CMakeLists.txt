cmake_minimum_required(VERSION 3.5)
project(tiny_solver)


#Options
option(WITH_TESTING     "Turn on testing with gtest"                             OFF)
option(WITH_EXAMPLES    "Turn on the examples"                             ON)

set (CMAKE_CXX_STANDARD 11)



# 3rd party packages

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(src)


add_executable(main src/main.cpp src/tiny_solver.h)


if(${WITH_TESTING})
    include_directories(tests)
    include(cmake/googletest.cmake)
    fetch_googletest(
            ${PROJECT_SOURCE_DIR}/cmake
            ${PROJECT_BINARY_DIR}/googletest
    )

    enable_testing()
    add_subdirectory(tests)

endif()

if(${WITH_EXAMPLES})
    add_subdirectory(examples)

endif()