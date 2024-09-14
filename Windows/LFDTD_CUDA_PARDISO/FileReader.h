#pragma once

#include "global.h"

inline void Sparse_MatrixImport(const std::string& path, int n, int NNZ, double* a, int* ia, int* ja)
{
    std::ifstream a_file, ia_file, ja_file;
    a_file.open(path + "Sparse_Val.txt");
    ja_file.open(path + "Sparse_Col.txt");
    ia_file.open(path + "Sparse_Row.txt");

    for (int n_temp = 0; n_temp <= n; n_temp++)
    {
        ia_file >> ia[n_temp];
    }


    for (int nz_temp = 0; nz_temp < NNZ; nz_temp++)
    {
        a_file >> a[nz_temp];
        ja_file >> ja[nz_temp];
    }

    a_file.close();
    ja_file.close();
    ia_file.close();
}

inline void MatrixImport_1D(const std::string& InputFile, int n, double* a)
{
    std::ifstream InFile;

    InFile.open("./" + InputFile);

    for (int ii = 0; ii < n; ++ii)
    {
        InFile >> a[ii];
    }

    InFile.close();
}

inline void MatrixImport_3D(const std::string& InputFile, int NX, int NY, int NZ, double_array_type& a)
{
    std::ifstream InFile;

    InFile.open("./" + InputFile);

    for (int kk = 0; kk < NZ; ++kk)
    {
        for (int jj = 0; jj < NY; ++jj)
        {
            for (int ii = 0; ii < NX; ++ii)
            {
                InFile >> a[ii][jj][kk];
            }
        }
    }

    InFile.close();
}

inline void MatrixImport_4D(const std::string& InputFile, int NX, int NY, int NZ, int nnz, node_array_type& Aa)
{
    std::ifstream InFile;

    InFile.open("./" + InputFile);

    for (int ll = 0; ll < nnz; ++ll)
    {
        for (int KK = 0; KK < NZ; ++KK)
        {
            for (int JJ = 0; JJ < NY; ++JJ)
            {
                for (int II = 0; II < NX; ++II)
                {
                    InFile >> Aa[II][JJ][KK][ll];
                }
            }
        }
    }

    InFile.close();
}

inline bool cmp(const std::pair<int, int>& a, const std::pair<int, int>& b) // compare function to help sort the pair vector
{
    return a.second < b.second;
}

template <typename T> void MatrixExport_1D(const std::string& OutFile, std::vector<T> Vec)
{
    {
        std::ofstream oFile(OutFile);

        for (int i = 0; i < Vec.size(); ++i)
        {
            oFile << Vec[i] << std::endl;
        }

        oFile.close();
    }
}

template <typename T> void MatrixExport_1D(const std::string& OutFile, T* arr, int size)
{
    {
        std::ofstream oFile(OutFile);

        for (int i = 0; i < size; ++i)
        {
            oFile << arr[i] << std::endl;
        }

        oFile.close();
    }
}

template<typename Type, typename Val> constexpr void fill_3D_array(Type(&arr3D), int x, int y, int z, const Val val) noexcept
{
    std::fill_n(&arr3D[0][0][0], x * y * z, val);
    // or using std::fill
    // std::fill(&arr3D[0][0][0], &arr3D[0][0][0] + (M * N * O), val);
}