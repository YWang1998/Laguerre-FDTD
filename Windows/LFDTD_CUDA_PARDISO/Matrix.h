//
// Created by Yifan Wang on 5/23/24.
//

#pragma once
#include "Vector.h"

namespace CBLAS
{
    template <typename Type = double, int dim = 2>
    class mat : public BLAS
    {
    public:

        typedef void (mat::* fill_ptr)(const Type&);

        explicit mat(int x, int y) : BLAS{ x,y }
        {
            Mat_Size = "2D";
            if (m == n) Mat_Type = "Square";
            else Mat_Type = "Rectangular";
            Fill_Prt = &mat::fill_M_2D;
            M.resize(boost::extents[m][n]);
        }
        explicit mat(int x, int y, int z) : BLAS{ x,y,z }
        {
            Mat_Size = "3D";
            if ((n1 == n2) && (n1 == n3)) Mat_Type = "Square";
            else Mat_Type = "Rectangular";
            Fill_Prt = &mat::fill_M_3D;
            M.resize((boost::extents[n1][n2][n3]));
        }
        explicit mat(int x, int y, int z, int w) : BLAS{ x,y,z,w }
        {
            Mat_Size = "4D";
            if ((n1 == n2) && (n1 == n3) && (n1 == n4)) Mat_Type = "Square";
            else Mat_Type = "Rectangular";
            Fill_Prt = &mat::fill_M_4D;
            M.resize((boost::extents[n1][n2][n3][n4]));
        }

        ~mat()
        {
            if (Factorization == "LU")
            {
                delete[] P;
                for (int i = 0; i < n; ++i)
                {
                    delete[] LU_Mat[i];
                }
                delete[] LU_Mat;
            }
        }

        template<typename T> friend
            mat<T> LinMat(const vec<int>& row, const vec<int>& col, const vec<T>& val); // Return a matrix based on vectors

        template<typename A> friend
            std::ostream& operator<<(std::ostream& out, const mat<A>& Mat);

        constexpr void fill_M(const Type& val)
        {
            (this->*Fill_Prt)(val);
        }

        constexpr void fill_M_2D(const Type& val)
        {
            std::fill_n(&M[0][0], m * n, val);
        }

        constexpr void fill_M_3D(const Type& val)
        {
            std::fill_n(&M[0][0][0], n1 * n2 * n3, val);
        }

        constexpr void fill_M_4D(const Type& val)
        {
            std::fill_n(&M[0][0][0][0], n1 * n2 * n3 * n4, val);
        }

        bool LU()
        {
            /* Factors "m" matrix into A=LU where L is lower triangular and U is upper
             * triangular. The matrix is overwritten by LU with the diagonal elements
             * of L (which are unity) not stored. This must be a square n x n matrix.
             * ri[n] and irow[n] are scratch vectors used by LUBackSubstitution.
             * d is returned +-1 indicating that the
             * number of row interchanges was even or odd respectively.
            */
            if ((Mat_Type != "Square") && (Mat_Size != "2D"))
            {
                std::cerr << "ERROR: Wrong matrix type for LU decomposition!" << std::endl;
                return false;
            }

            int i, j;
            double det = 1.0;
            LU_Mat = new Type * [m];
            P = new int[m]; // initialize permutation array
            P_sorted = new int[m];
            for (i = 0; i < m; ++i)
            {
                P[i] = i;
                P_sorted[i] = i;
                LU_Mat[i] = new Type[n];
                for (j = 0; j < n; ++j)
                {
                    LU_Mat[i][j] = M[i][j];
                }
            }
            // LU factorization.
            for (int p = 0; p < n - 1; ++p)
            {
                // Find pivot element.

                for (i = p + 1; i < n; ++i)
                {
                    if (std::fabs(LU_Mat[P[i]][p]) > std::fabs(LU_Mat[P[p]][p]))
                    {
                        // Switch the index for the p-1 pivot row if necessary.
                        auto t = P[p];
                        P[p] = P[i];
                        P[i] = t;
                        det = -det;
                    }
                }
                if (LU_Mat[P[p]][p] == 0) return false; // The matrix is singular.
                // Multiply the diagonal elements.
                det *= LU_Mat[P[p]][p];
                // Form multiplier.
                for (i = p + 1; i < n; ++i)
                {
                    LU_Mat[P[i]][p] /= LU_Mat[P[p]][p];
                    // Eliminate [p-1].
                    for (j = p + 1; j < n; ++j)
                        LU_Mat[P[i]][j] -= LU_Mat[P[i]][p] * LU_Mat[P[p]][j];
                }
            }
            det *= LU_Mat[P[n - 1]][n - 1];

            Factorization = "LU";

            for (int idx = 0; idx < n; ++idx)
            {
                int swapped_row = P[idx];
                int sorted_row = P_sorted[idx];
                int count{ idx };

                while (P_sorted[count] != swapped_row)
                {
                    ++count;
                }

                { // swap the row of LU matrix
                    Type* temp_ptr;
                    temp_ptr = LU_Mat[idx];
                    LU_Mat[idx] = LU_Mat[count];
                    LU_Mat[count] = temp_ptr;
                }

                P_sorted[idx] = P_sorted[count];
                P_sorted[count] = sorted_row;
            }

            std::cout << "\n LU Factorization is completed! \n";

            /*
            for (i = 0; i < m; ++i) {
                for (j = 0; j < n; ++j)
                {
                    std::cout<<LU_Mat[i][j]<<" ";
                }
                std::cout<<std::endl;
            }

            std::cout<<"The permutation array after reordering is:\n";
            for (j = 0; j < m; ++j)
            {
                std::cout<<P_sorted[j]<<std::endl;
            }*/

            return det != 0.0;
        }

        template <typename A> void
            div(const vec<A>& b, vec<A>& x)
        {
            if ((b != x) || (n != x.M()))
            {
                std::cerr << "ERROR: Unmatched vector/matrix size for solving!" << std::endl;
                exit(-1);
            }
            if (Factorization != "LU") // Perform LU decomposition if not computed before
            {
                if (!LU())
                {
                    std::cerr << "ERROR: LU factorization failed!" << std::endl;
                    exit(-1);
                }

            }

            for (int i = 0; i < n; i++)
            {
                x[i][0] = b[P[i]][0];

                for (int k = 0; k < i; k++)
                {
                    x[i][0] -= LU_Mat[i][k] * x[k][0];
                }
            }

            for (int i = n - 1; i >= 0; i--)
            {
                for (int k = i + 1; k < n; k++)
                {
                    x[i][0] -= LU_Mat[i][k] * x[k][0];
                }

                x[i][0] /= LU_Mat[i][i];
            }
        }

        /************************************************************
         *                      Operator Region                     *
        *************************************************************/

        typename boost::multi_array<Type, dim>::reference operator[](int idx1)   // overload the [] operator to return a boost multi-array reference
            // https://theboostcpplibraries.com/boost.multiarray
        {
            return M[idx1];
        }

        typename boost::multi_array<Type, dim>::const_reference operator[](int idx1) const   // overload the [] operator to return a const boost multi-array reference
            // https://theboostcpplibraries.com/boost.multiarray
        {
            return M[idx1];
        }

        /************************************************************
         *                      Level III - BLAS                    *
        *************************************************************/


    private:
        boost::multi_array<Type, dim> M;
        Type** LU_Mat; // pointer matrix for easier permutation swap
        int* P, * P_sorted; // Permutation array for LU decomposition
        fill_ptr Fill_Prt;
        std::string Factorization; // QR/LU decomposition type of M
        std::string Mat_Type; // Square/Rectangular/Complex
        std::string Mat_Size; // 2D/3D/4D
    };

    template <typename T> mat<T> LinMat(const vec<int>& row, const vec<int>& col, const vec<T>& val)
    {
        if (row == col && row == val)
        {
            int M = row.M(); int N = row.N();
            int dim = M > N ? M : N;
            int size = row(dim - 2) + 1; // Find the size of the matrix from the row vector
            mat<T> Sparse_Mat{ size,size };
            for (int i = 0; i < dim; ++i)
            {
                Sparse_Mat[row(i)][col(i)] = val(i);
            }

            return Sparse_Mat;
        }
        else
        {
            std::cerr << "ERROR:Input vectors have unmatched dimensions!!" << std::endl;
            exit(-1);
        }

    }

    template <typename A> std::ostream& operator<<(std::ostream& out, const mat<A>& Mat)
    {
        if (Mat.Mat_Size == "2D")
        {
            for (int i = 0; i < Mat.n; ++i)
            {
                for (int j = 0; j < Mat.m; ++j)
                {
                    out << Mat[i][j] << " ";
                }
                out << std::endl;
            }
        }
        return out;
    }
}

template<typename T = double> CBLAS::mat<T> zeros(int m, int n)
{
    CBLAS::mat<T> M{ m,n };

    return M;
}

template<typename T = double> CBLAS::mat<T> zeros(int n1, int n2, int n3)
{
    CBLAS::mat<T> M{ n1,n2,n3 };

    return M;
}

template<typename T = double> CBLAS::mat<T> zeros(int n1, int n2, int n3, int n4)
{
    CBLAS::mat<T> M{ n1,n2,n3,n4 };

    return M;
}

template<typename T = double> CBLAS::mat<T> ones(int m, int n)
{
    CBLAS::mat<T> M{ m,n };
    T val = 1.0;
    M.fill_M(val);
    return M;
}

template<typename T = double> CBLAS::mat<T> ones(int n1, int n2, int n3)
{
    CBLAS::mat<T> M{ n1,n2,n3 };
    T val = 1.0;
    M.fill_M(val);
    return M;
}

template<typename T = double> CBLAS::mat<T> ones(int n1, int n2, int n3, int n4)
{
    CBLAS::mat<T> M{ n1,n2,n3,n4 };
    T val = 1.0;
    M.fill_M(val);
    return M;
}