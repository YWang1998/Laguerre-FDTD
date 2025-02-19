cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CXX_COMPILER "/opt/intel/oneapi/compiler/2024.2/bin/icpx")
set(CMAKE_CUDA_ARCHITECTURES 86)

project(LFDTD_PARDISO_CUDA CUDA CXX)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # to tell your code it was found via #ifdef USE_CUDA:
    add_definitions(-DUSE_CUDA)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    link_directories("/usr/local/cuda-12.6/lib64")

endif()

set(CMAKE_CXX_FLAGS " -lGLEW -lGL -lX11 -lGLU -lglfw3 -std=c++17")

set(CMAKE_PREFIX_PATH "/opt/intel/oneapi/tbb/2021.13/lib/cmake/tbb")

find_package(GLEW 2.2.0 REQUIRED)

set(BOOST_ROOT "/opt/boost_1_85_0") # change this to your local /path/to/boost folder

find_package(Boost 1.85.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

# CUDA Helper Folder
include_directories(${ProjectFileDir}../Common)

set(ENV{MKLROOT} "/opt/intel/oneapi/mkl/2024.2")

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
#message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input

add_executable(LFDTD_PARDISO_CUDA main.cpp
        global.h
        Shader.cpp
        Shader.h
        spMV.cu
        spMV.cuh
        Matrix.h
        Vector.h
        LFDTD_Coe.cpp
        LFDTD_Coe.h
        LFDTD.cpp
        LFDTD.h
)
add_definitions(-DGLEW_STATIC)

target_compile_options(LFDTD_PARDISO_CUDA PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(LFDTD_PARDISO_CUDA PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(LFDTD_PARDISO_CUDA PUBLIC $<LINK_ONLY:MKL::MKL>)
target_link_libraries(LFDTD_PARDISO_CUDA PUBLIC cusparse cublas)
target_link_libraries(LFDTD_PARDISO_CUDA PUBLIC GLEW::glew glfw3)

if(MKL_ENV)
    set_tests_properties(LFDTD_PARDISO_CUDA PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif()




