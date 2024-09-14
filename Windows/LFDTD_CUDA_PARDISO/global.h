/*

Header file that contains all the global types and parameters declaration

*/

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <map>
#include <cstdio>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <memory>
#include <future>
#include <thread>
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/multi_array.hpp>
#include "algorithm"

// Intel MKL 
#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // shared functions common to CUDA Samples
#include <helper_cuda.h>       // CUDA error checking



// Create a 3D array that is nx x ny x nz
typedef boost::multi_array<int, 4> node_array_type; // nodeNum 4D matrix
typedef boost::multi_array<double, 3> double_array_type; // double 3D matrix

enum Solver
{
    _PARDISO = 0,
    _cuSPARSE,
    _CUDA,
    _CUDA_Expanded
};

enum Precon
{
    None = 1,       // No preconditioner
    Jacobi = 2,     // Jacobi Preconditioner
    Laguerre = 3   // Proposed Preconditioner
};

extern double eps0, mu0, v0;
extern int Nnode, NNZ;
extern double R0, Rs, Rl;
extern double pi;

extern int maxit;
extern double tol;
const int warp = 32;

extern int jDirecIndex[3];
extern int probeDirecIndex[3];
