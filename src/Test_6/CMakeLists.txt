project(test_6)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.F90)

add_executable(test_6 ${SRC_FILES})