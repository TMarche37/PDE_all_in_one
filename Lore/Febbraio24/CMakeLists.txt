cmake_minimum_required(VERSION 3.12.0)
project(lab02 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(exercise-02 src/exercise-02.cpp src/DiffusionReaction.cpp)
deal_ii_setup_target(exercise-02)

