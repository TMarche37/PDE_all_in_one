cmake_minimum_required(VERSION 3.12.0)
project(lab06 LANGUAGES CXX C)

include(../../common/cmake-common.cmake)

add_executable(primo ./exercise-01.cpp ./Heat.cpp)
deal_ii_setup_target(primo)