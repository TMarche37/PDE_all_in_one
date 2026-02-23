cmake_minimum_required(VERSION 3.12.0)
project(exam LANGUAGES CXX C)

include(../../common/cmake-common.cmake)

add_executable(main "src/main.cpp" "src/Stokes.cpp")
deal_ii_setup_target(main)
