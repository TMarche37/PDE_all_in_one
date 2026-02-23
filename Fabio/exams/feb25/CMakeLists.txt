cmake_minimum_required(VERSION 3.12.0)
project(lab02 LANGUAGES CXX C)

include(../../common/cmake-common.cmake)

add_executable(primo ./primo.cpp ./Poisson1D.cpp)
deal_ii_setup_target(primo)

