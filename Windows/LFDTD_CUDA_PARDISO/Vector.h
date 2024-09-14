//
// Created by Yifan Wang on 5/23/24.
//

#pragma once

#include <boost/multi_array.hpp>
#include <iostream>

namespace CBLAS
{
    class BLAS
    {
    public:
        BLAS(int M, int N) : m{ M }, n{ N } {} // constructor for vec and 2D matrix
        BLAS(int x, int y, int z) : n1{ x }, n2{ y }, n3{ z } {} // constructor of 3D matrix
        BLAS(int x, int y, int z, int w) : n1{ x }, n2{ y }, n3{ z }, n4{ w } {} // constructor of 4D matrix

    protected:
        int m, n; // M * N size of a 2D matrix/vec
        int n1, n2, n3, n4; // Matrix size for 3D/4D matrix construction
    };

    template <typename Type = double>
    class vec : public BLAS
    {
    public:

        vec() :BLAS{ 1,1 } // Default to 1 * 1 array
        {
            V.resize(boost::extents[m][n]);
            dim = "Row";
        }

        explicit vec(int M) : BLAS{ M,1 } // Row based vector constructor
        {
            V.resize(boost::extents[m][n]);
            dim = "Row";
        }

        explicit vec(int M, int N) : BLAS{ M,N }
        {
            if ((m != 1) && (n != 1))
            {
                std::cerr << "Invalid instantiation of vec!!" << std::endl;
                exit(-1);
            }
            V.resize(boost::extents[m][n]);
            dim = m > n ? "Row" : "Col";
        }

        constexpr void fill_v(const Type& Val) // fill in the vec with specific value
        {
            std::fill_n(&V[0][0], m * n, Val);
        }

        Type M() const // return size M
        {
            return m;
        }

        Type N() const // return size N
        {
            return n;
        }

        template <typename A> friend
            std::ostream& operator<<(std::ostream& out, const CBLAS::vec<A>& Vec);

        template<typename A>
        bool check_integrity(const vec<A>& RHS) const // check for vec-vec and vec-Matrix multiplication
        {

            if (*this == RHS)
            {
                type = "Broadcast";
                return true;
            }

            if (m == RHS.N())
            {
                type = "Dot";
                return true;
            }

            return false;

        }

        vec<Type> Transpose() const // Transpose the vec and re-assign
        {
            vec<Type> Vec_T{ n,m };
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    Vec_T[j][i] = V[i][j];
                }
            }

            return Vec_T;
        }

        vec<Type>& T() // Transpose the intrinsic vec itself
        {
            vec<Type> Vec_temp = *this;

            m = Vec_temp.n;
            n = Vec_temp.m;

            V.resize(boost::extents[m][n]);

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    V[i][j] = Vec_temp[j][i];
                }
            }

            if (dim == "Row") dim = "Col";
            else dim = "Row";

            return *this;
        }

        void append(const Type& Ele)
        {
            if (dim == "Row")
            {
                if (m == 1) --m; // edge case when vector is first initialized
                ++m;
                V.resize(boost::extents[m][n]);
                V[m - 1][0] = Ele;
            }
            else
            {
                if (n == 1) --n; // edge case when vector is first initialized
                ++n;
                V.resize(boost::extents[m][n]);
                V[0][n - 1] = Ele;
            }
        }

        void resize(const int& Range)
        {
            if (dim == "Row")
            {
                m = Range;
                V.resize(boost::extents[m][n]);
            }
            else
            {
                n = Range;
                V.resize(boost::extents[m][n]);
            }
        }

        int size()
        {
            int Dim;
            return Dim = m > n ? m : n;
        }

        /************************************************************
         *                      Operator Region                     *
        ************************************************************/

        vec<Type>& operator=(const vec<Type>& RHS) // V1 = V - same type
        {
            m = RHS.M();
            n = RHS.N();
            V.resize(boost::extents[m][n]);
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    V[i][j] = RHS[i][j];
                }
            }

            return *this;
        }

        template <typename A>
        vec<Type>& operator=(const vec<A>& RHS) // V1 = V - different type
        {
            m = RHS.M();
            n = RHS.N();
            V.resize(boost::extents[m][n]);
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    V[i][j] = RHS[i][j];
                }
            }

            return *this;
        }

        template <typename A>
        vec<Type> operator+(const vec<A>& RHS) const //  V= V1 + V2
        {

            vec<Type> Out{ m, n };

            if (check_integrity<A>(RHS))
            {
                if (type == "Broadcast")
                {
                    for (int i = 0; i < m; ++i)
                    {
                        for (int j = 0; j < n; ++j)
                        {
                            Out.V[i][j] = V[i][j] + RHS[i][j];
                        }
                    }
                    return Out;
                }
                else
                {
                    std::cerr << "ERROR:Unmatched input vec dimension for addition!" << std::endl;
                    exit(-1);
                }
            }
            else
            {
                std::cerr << "ERROR:Unmatched input vec dimensions!" << std::endl;
                exit(-1);
            }
        }

        template <typename A>
        vec<Type> operator+(const A const_val) const // V = V1 + a
        {

            vec<Type> Out{ m, n };

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    Out[i][j] = V[i][j] + const_val;
                }
            }

            return Out;
        }

        template <typename A>
        vec<Type>& operator+=(A const_val) // V1 += a
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    V[i][j] += const_val;
                }
            }

            return *this;
        }

        template <typename A>
        vec<Type> operator-(const vec<A>& RHS) const //  V= V1 - V2
        {

            vec<Type> Out{ m, n };

            if (check_integrity<A>(RHS))
            {
                if (type == "Broadcast")
                {
                    for (int i = 0; i < m; ++i)
                    {
                        for (int j = 0; j < n; ++j)
                        {
                            Out.V[i][j] = V[i][j] - RHS[i][j];
                        }
                    }
                    return Out;
                }
                else
                {
                    std::cerr << "ERROR:Unmatched input vec dimension for addition!" << std::endl;
                    exit(-1);
                }
            }
            else
            {
                std::cerr << "ERROR:Unmatched input vec dimensions!" << std::endl;
                exit(-1);
            }

        }

        template <typename A>
        vec<Type> operator-(const A const_val) const // V = V1 - a
        {

            vec<Type> Out{ m, n };

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    Out[i][j] = V[i][j] - const_val;
                }
            }

            return Out;
        }

        template <typename A>
        vec<Type>& operator-=(A const_val) // V1 -= a
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    V[i][j] -= const_val;
                }
            }

            return *this;
        }

        template <typename A>
        Type operator*(const vec<A>& RHS) const // Out = V1*V2
        {
            if (check_integrity<A>(RHS))
            {
                Type dot_product{ 0 };
                if (type == "Dot")
                {
                    for (int i = 0; i < m; ++i)
                    {
                        for (int j = 0; j < n; ++j)
                        {
                            dot_product += (V[i][j] * RHS[j][i]);
                        }
                    }
                }
                else
                {
                    std::cout << "Warning: Attempting to perform broadcasting vec multiplication - implicit vec transpose is performed." << std::endl;
                    auto Vec_T = RHS.Transpose();
                    for (int i = 0; i < m; ++i)
                    {
                        for (int j = 0; j < n; ++j)
                        {
                            dot_product += (V[i][j] * Vec_T[j][i]);
                        }
                    }
                }
                return dot_product;
            }
            else
            {
                std::cerr << "ERROR:Unmatched input vec dimensions!" << std::endl;
                exit(-1);
            }

        }


        template <typename A>
        vec<Type> operator*(const A const_val) const // V = a * V1
        {
            vec<Type> Out{ m,n };

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    Out[i][j] = V[i][j] * const_val;
                }
            }

            return Out;
        }

        template <typename A>
        vec<Type>& operator*=(A const_val) // V1 *= a
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    V[i][j] *= const_val;
                }
            }

            return *this;
        }

        typename boost::multi_array<Type, 2>::reference operator[](int idx1) // overload the [] operator to return a boost multi-array reference
            // https://theboostcpplibraries.com/boost.multiarray
        {
            return V[idx1];
        }

        typename boost::multi_array<Type, 2>::const_reference operator[](int idx1) const // overload the [] operator to return a const boost multi-array reference
            // https://theboostcpplibraries.com/boost.multiarray
        {
            return V[idx1];
        }

        Type& operator()(int idx) // return the ref of value of the idx without knowing the actual layout of the vector
        {
            if (m == 1) return V[0][idx];
            else return V[idx][0];

        }

        const Type& operator()(int idx) const // return the const ref of value of the idx without knowing the actual layout of the vector
        {
            if (m == 1) return V[0][idx];
            else return V[idx][0];
        }

        template <typename A>
        bool operator==(const vec<A>& RHS) const
        {
            if ((m == RHS.M()) && (n == RHS.N())) return true;
            else return false;
        }

        template <typename A>
        bool operator!=(const vec<A>& RHS) const
        {
            if ((m == RHS.M()) && (n == RHS.N())) return false;
            else return true;
        }
    private:
        boost::multi_array <Type, 2> V; // vec is either (m,1) or (1,n)
        std::string dim;
        mutable std::string type;   // 1. Dot - m = n dimension for dot product
                                    // 2. Broadcast - same m,n dimension for element-wise operation
    };

    template <typename Type>
    std::ostream& operator<<(std::ostream& out, const CBLAS::vec<Type>& Vec)
    {
        for (int i = 0; i < Vec.m; ++i)
        {
            for (int j = 0; j < Vec.n; ++j)
            {
                out << Vec.V[i][j] << " ";
            }
            out << std::endl;
        }
        return out;
    }

}

template<typename T = double> CBLAS::vec<T> zeros(int m)
{
    CBLAS::vec<T> V{ m,1 };

    return V;
}

template<typename T = double> CBLAS::vec<T> ones(int m)
{
    CBLAS::vec<T> V{ m,1 };
    T val = 1.0;
    V.fill_v(val);
    return V;
}
