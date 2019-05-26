cmake_minimum_required(VERSION 3.5)
project(tiny_solver)


#Options
option(WITH_TESTING     "Turn on testing with gtest"                             OFF)

set (CMAKE_CXX_STANDARD 11)



# 3rd party packages

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

#if(${WITH_TESTING})
#enable_testing()
#find_package(GTest REQUIRED)
#
#endif()


include_directories(src)


add_executable(main src/main.cpp src/tiny_solver.h)