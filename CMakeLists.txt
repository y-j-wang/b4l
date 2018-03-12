cmake_minimum_required(VERSION 2.8)
project(tf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

# add a NativeRelease build type
SET( CMAKE_CXX_FLAGS_NATIVERELEASE "-O3 -DNDEBUG -march=native -ftree-vectorize" CACHE STRING
        "Flags used by the C++ compiler during native builds."
        FORCE )
SET( CMAKE_C_FLAGS_NATIVERELEASE "-O3 -DNDEBUG -march=native -ftree-vectorize" CACHE STRING
        "Flags used by the C compiler during native builds."
        FORCE )
SET( CMAKE_EXE_LINKER_FLAGS_NATIVERELEASE "" CACHE STRING
        "Flags used for linking binaries during native builds."
        FORCE )
SET( CMAKE_SHARED_LINKER_FLAGS_NATIVERELEASE "" CACHE STRING
        "Flags used by the shared libraries linker during native builds."
        FORCE )
MARK_AS_ADVANCED(
        CMAKE_CXX_FLAGS_NATIVERELEASE
        CMAKE_C_FLAGS_NATIVERELEASE
        CMAKE_EXE_LINKER_FLAGS_NATIVERELEASE
        CMAKE_SHARED_LINKER_FLAGS_NATIVERELEASE )

if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND CMAKE_CONFIGURATION_TYPES NativeRelease)
    list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
            "Add the native configuration"
            FORCE)
endif()

## default build type is native
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE NativeRelease CACHE STRING
            "Choose the type of build. Options are: None Debug NativeRelease Release RelWithDebInfo MinSizeRel."
            FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

## look for MKL
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/cmake/")
find_package(MKL REQUIRED)
if(MKL_FOUND)
    message(STATUS "MKL found " ${MKL_INCLUDE_DIRS})
    include_directories(${MKL_INCLUDE_DIRS})
    link_directories(${MKL_LIBRARIES})
endif()

## look for TBB
find_package(TBB REQUIRED)
if(TBB_FOUND)
    message(STATUS "TBB found: " ${TBB_INCLUDE_DIRS})
    include_directories(${TBB_INCLUDE_DIRS})
    link_libraries(${TBB_LIBRARIES})
endif(TBB_FOUND)

## look for boost
find_package(Boost COMPONENTS program_options system filesystem)
if(NOT ${Boost_FOUND})
    SET(BOOST_ROOT ~/local) # default
    SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
    find_package(Boost COMPONENTS program_options system filesystem)
endif(NOT ${Boost_FOUND})
if(Boost_FOUND)
    message(STATUS "Boost found")
    include_directories(${Boost_INCLUDE_DIR})
    link_libraries(${Boost_LIBRARIES})
endif(Boost_FOUND)

# look for blas
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "BLAS found")
    include_directories(${BLAS_INCLUDE_DIR})
    link_libraries(${BLAS_LIBRARIES})
endif(BLAS_FOUND)

## set include directory
include_directories(include)

## define tf library
set(libtf_SRCS)
set(libtf_HDRS
        include/tf/util/Data.h
        include/tf/util/Base.h
        include/tf/util/FileUtil.h
        include/tf/util/Monitor.h
        include/tf/util/RandomUtil.h
        include/tf/util/ThreadUtil.h
        include/tf/util/CompareUtil.h
        include/tf/util/Calculator.h
        include/tf/util/EvaluationUtil.h
        include/tf/util/Parameter.h
        include/tf/util/LRUtil.h
        include/tf/struct/Triple.h
        include/tf/struct/Tuple.h
        include/tf/struct/Weight.h
        include/tf/struct/Min_Max.h
        include/tf/alg/Optimizer.h
        include/tf/alg/THPipeline.h
#        include/tf/alg/Ensemble.h
#        include/tf/alg/Ensemble2.h
        include/tf/alg/RESCAL_RANK.h
        include/tf/alg/RESCAL.h
        include/tf/alg/HOLE.h
        include/tf/alg/Sampler.h
        include/tf/alg/TransE.h)

## look for liblinear
set(liblinear_HDRS
    include/tf/liblinear/linear.h
    include/tf/liblinear/tron.h)

set(liblinear_SRCS
    include/tf/liblinear/linear.cpp
    include/tf/liblinear/tron.cpp)

add_library(tf SHARED ${libtf_HDRS} ${libtf_SRCS} ${liblinear_HDRS} ${liblinear_SRCS})
set_target_properties(tf PROPERTIES LINKER_LANGUAGE CXX)

## executables
link_libraries(tf ${MKL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${BLAS_LIBRARIES})
add_executable(runRESCAL_RANK src/runRESCAL_RANK)
add_executable(runTransE src/runTransE.cpp)
add_executable(runRESCAL src/runRESCAL.cpp)
add_executable(runHOLE src/runHOLE.cpp)

add_executable(runRTLREnsemble src/LREnsemble/runRTLREnsemble.cpp)
add_executable(runRHLREnsemble src/LREnsemble/runRHLREnsemble.cpp)
add_executable(runHTLREnsemble src/LREnsemble/runHTLREnsemble.cpp)
add_executable(runRHTLREnsemble src/LREnsemble/runRHTLREnsemble.cpp)

add_executable(runTHPipeline src/Boosting/runTHPipeline.cpp)

add_executable(outputScore src/outputScore.cpp)
add_executable(tripleClassification src/tripleClassification.cpp)

#add_executable(runEnsemble src/JoinEnsemble/runEnsemble.cpp)
#add_executable(runEnsemble2 src/JoinEnsemble/runEnsemble2.cpp)
