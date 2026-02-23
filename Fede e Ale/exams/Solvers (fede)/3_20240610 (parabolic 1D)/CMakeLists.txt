cmake_minimum_required(VERSION 3.12.0)
project(exam3 LANGUAGES CXX C)

include(../../../all_possibilities/common/cmake-common.cmake)

add_executable(ex1 src/ex1.cpp src/Poisson1D.cpp)
deal_ii_setup_target(ex1)
