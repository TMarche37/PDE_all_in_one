cmake_minimum_required(VERSION 3.12.0)
project(exam LANGUAGES CXX C)

include(../../../all_possibilities/common/cmake-common.cmake)

add_executable(ex1 src/lab-09.cpp src/Stokes.cpp)
deal_ii_setup_target(ex1)
