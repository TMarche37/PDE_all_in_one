cmake_minimum_required(VERSION 3.12.0)
project(exam_1 LANGUAGES CXX C)
include(../common/cmake-common.cmake)

add_executable(main src/main.cpp src/Parabolic2D.cpp)
deal_ii_setup_target(ex1)