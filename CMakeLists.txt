cmake_minimum_required(VERSION 3.0.0)
project(Deconvolucao VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Deconvolucao Deconvolucao.cpp)
target_link_libraries(Deconvolucao ${OpenCV_LIBS})
