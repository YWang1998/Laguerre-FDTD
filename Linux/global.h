#ifndef GLOBAL_H
#define GLOBAL_H

#if _WIN32
#define INTEGER int
#else
#define INTEGER long long
#endif


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
#include <algorithm>

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

#include "Matrix.h"

// Create a 3D array that is nx x ny x nz
typedef boost::multi_array<INTEGER, 4> node_array_type; // nodeNum 4D matrix
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
extern INTEGER Nnode, NNZ;
extern double R0, Rs, Rl;
extern double pi;

extern INTEGER maxit;
extern double tol;
const INTEGER warp = 32;

extern INTEGER jDirecIndex[3];
extern INTEGER probeDirecIndex[3];


inline void Sparse_MatrixImport(const std::string& path, INTEGER n, INTEGER NNZ, double* a, INTEGER* ia, INTEGER* ja)
{
    std::ifstream a_file, ia_file, ja_file;
    a_file.open(path + "Sparse_Val.txt");
    ja_file.open(path + "Sparse_Col.txt");
    ia_file.open(path + "Sparse_Row.txt");

    for (INTEGER n_temp = 0; n_temp <= n; n_temp++)
    {
        ia_file >> ia[n_temp];
    }


    for (INTEGER nz_temp = 0; nz_temp < NNZ; nz_temp++)
    {
        a_file >> a[nz_temp];
        ja_file >> ja[nz_temp];
    }

    a_file.close();
    ja_file.close();
    ia_file.close();
}

inline void MatrixImport_1D(const std::string& InputFile, INTEGER n, double* a)
{
    std::ifstream InFile;

    InFile.open("./" + InputFile);

    for (INTEGER ii = 0; ii < n; ++ii)
    {
        InFile >> a[ii];
    }

    InFile.close();
}

inline void MatrixImport_3D(const std::string& InputFile, INTEGER NX, INTEGER NY, INTEGER NZ, double_array_type& a)
{
    std::ifstream InFile;

    InFile.open("./" + InputFile);

    for (INTEGER kk = 0; kk < NZ; ++kk)
    {
        for (INTEGER jj = 0; jj < NY; ++jj)
        {
            for (INTEGER ii = 0; ii < NX; ++ii)
            {
                InFile >> a[ii][jj][kk];
            }
        }
    }

    InFile.close();
}

inline void MatrixImport_4D(const std::string& InputFile, INTEGER NX, INTEGER NY, INTEGER NZ, INTEGER nnz, node_array_type& Aa)
{
    std::ifstream InFile;

    InFile.open("./" + InputFile);

    for (INTEGER ll = 0; ll < nnz; ++ll)
    {
        for (INTEGER KK = 0; KK < NZ; ++KK)
        {
            for (INTEGER JJ = 0; JJ < NY; ++JJ)
            {
                for (INTEGER II = 0; II < NX; ++II)
                {
                    InFile >> Aa[II][JJ][KK][ll];
                }
            }
        }
    }

    InFile.close();
}

inline bool cmp(const std::pair<INTEGER, INTEGER>& a, const std::pair<INTEGER, INTEGER>& b) // compare function to help sort the pair vector
{
    return a.second < b.second;
}

template <typename T> void MatrixExport_1D(const std::string& OutFile, std::vector<T> Vec)
{
    {
        std::ofstream oFile(OutFile);

        for (INTEGER i = 0; i < Vec.size(); ++i)
        {
            oFile << Vec[i] << std::endl;
        }

        oFile.close();
    }
}

template <typename T> void MatrixExport_1D(const std::string& OutFile, T* arr, INTEGER size)
{
    {
        std::ofstream oFile(OutFile);

        for (INTEGER i = 0; i < size; ++i)
        {
            oFile << arr[i] << std::endl;
        }

        oFile.close();
    }
}

template<typename Type, typename Val> constexpr void fill_3D_array(Type(&arr3D), INTEGER x, INTEGER y, INTEGER z, const Val val) noexcept
{
    std::fill_n(&arr3D[0][0][0], x * y * z, val);
    // or using std::fill
    // std::fill(&arr3D[0][0][0], &arr3D[0][0][0] + (M * N * O), val);
}

#endif
