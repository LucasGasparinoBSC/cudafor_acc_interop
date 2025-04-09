cmake_minimum_required(VERSION 3.10)

project(acc_cudafor LANGUAGES Fortran)

set(CMAKE_Fortran_FLAGS "-O3 -acc -cuda -gpu=cc86,lineinfo -Minfo=all")

add_subdirectory(src)