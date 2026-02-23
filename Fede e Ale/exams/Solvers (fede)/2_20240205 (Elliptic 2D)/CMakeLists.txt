cmake_minimum_required(VERSION 3.12.0)
project(exam_2 LANGUAGES CXX C)

include(../../../all_possibilities/common/cmake-common.cmake)

add_executable(ex1 src/ex1.cpp src/ADR_2D.cpp)
deal_ii_setup_target(ex1)
