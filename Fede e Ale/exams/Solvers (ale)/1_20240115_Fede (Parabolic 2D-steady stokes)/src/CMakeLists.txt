cmake_minimum_required(VERSION 3.12.0)
project(exam_1 LANGUAGES CXX C)
include(../common/cmake-common.cmake)

add_executable(ex1 src/ex1.cpp src/Heat.cpp)
add_executable(ex2 src/ex2.cpp src/Stokes.cpp)
deal_ii_setup_target(ex1)
deal_ii_setup_target(ex2)