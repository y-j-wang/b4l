cmake_minimum_required(VERSION 2.8)
project(liblinear)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wconversion -O3 -fPIC -fopenmp")

# look for blas
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "BLAS found")
    include_directories(${BLAS_INCLUDE_DIR})
    link_libraries(${BLAS_LIBRARIES})
endif(BLAS_FOUND)

set(linear
    linear.h
    linear.cpp
    tron.h
    tron.cpp)

add_library(liblinear SHARED ${linear})
set_target_properties(liblinear PROPERTIES LINKER_LANGUAGE CXX)
link_libraries(liblinear ${MKL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${BLAS_LIBRARIES})
add_executable(train_test train_test.cpp)