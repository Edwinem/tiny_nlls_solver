cmake_minimum_required(VERSION 3.5)
project(tiny_solver)


#Options
set (CMAKE_CXX_STANDARD 11)


# 3rd party packages

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})


include_directories(src)


add_executable(main src/main.cpp src/tiny_solver.h)