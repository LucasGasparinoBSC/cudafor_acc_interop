cmake_minimum_required(VERSION 3.10)

project(acc_cudafor LANGUAGES Fortran CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_Fortran_FLAGS "-O3 -acc -cuda -gpu=cc61,lineinfo,ptxinfo -Minfo=all")
set(CMAKE_CUDA_FLAGS "-O3 -lineinfo -res-usage")

add_subdirectory(src)
