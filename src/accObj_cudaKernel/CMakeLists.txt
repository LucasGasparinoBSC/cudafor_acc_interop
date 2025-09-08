project(accObj_cudaKernel)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.F90)

add_executable(${PROJECT_NAME} ${SRC_FILES})