cmake_minimum_required( VERSION 2.8 )

project(useEigen )

find_package(Eigen3 REQUIRED)
set(CMAKE_BUILD_TYPE "Release" )

set(CMAKE_CXX_FLAGS "-O3" )

# 添加Eigen头文件

#include_directories( "/usr/include/eigen3" )
include_directories(${EIGEN3_INCLUDE_DIRS})
#message(${Eigen3_INCLUDE_DIRS})
#message("xxxx")
add_executable(octree octree.cc)