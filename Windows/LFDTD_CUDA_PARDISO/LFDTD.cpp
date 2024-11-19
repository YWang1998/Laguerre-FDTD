#include "global.h"
#include "LFDTD.h"

using namespace CBLAS;

Solver LFDTD::_Solver;

Precon LFDTD::_M;

LFDTD::LFDTD()
{

    b = std::make_unique<double[]>(Nnode);
    x = std::make_unique<double[]>(Nnode);

    sumE = std::make_unique<double[]>(Nnode);
    PrintQ = 0;

}

void LFDTD::PrintQ_set(int i)
{
    if (i) PrintQ = 1;
}

void LFDTD::Sparse_A_Val(std::ifstream& a_file, int n, const int NNZ)
{

}

void LFDTD::SparseA_COO(const LFDTD_Coe& Coe)
{

    if (_Solver)
    {
        printf("Constructing Sparse Matrix A ...\n");

        ia_M = std::make_unique<int[]>(Nnode + 1);
        ja_M = std::make_unique<int[]>(Nnode);
        D = std::make_unique<double[]>(Nnode);
        I = std::make_unique<double[]>(Nnode);

        for (int i = 0; i < Nnode; i++)
        {
            I[i] = 1.0; // Pre-assign 1.0 to all keys for proposed preconditioner
            ia_M[i] = i;
            ja_M[i] = i;
        }

        ia_M[Nnode] = Nnode;

        // Ex equations
        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ex111 = Coe._nodeNum[i][j][k][0];

                    ex121 = Coe._nodeNum[i][j + 1][k][0];
                    ey211 = Coe._nodeNum[i + 1][j][k][1];
                    ey111 = Coe._nodeNum[i][j][k][1];

                    ex101 = Coe._nodeNum[i][j - 1][k][0];
                    ey201 = Coe._nodeNum[i + 1][j - 1][k][1];
                    ey101 = Coe._nodeNum[i][j - 1][k][1];

                    ez211 = Coe._nodeNum[i + 1][j][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];
                    ex112 = Coe._nodeNum[i][j][k + 1][0];

                    ez210 = Coe._nodeNum[i + 1][j][k - 1][2];
                    ez110 = Coe._nodeNum[i][j][k - 1][2];
                    ex110 = Coe._nodeNum[i][j][k - 1][0];

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                    VAL.emplace_back(1.0 + Coe._cey[i][j][k] * Coe._chy[i][j][k] + Coe._cey[i][j][k]
                        * Coe._chy[i][j - 1][k] + Coe._cez[i][j][k] * Coe._chz[i][j][k]
                        + Coe._cez[i][j][k] * Coe._chz[i][j][k - 1] + 2 * Coe._sigmax[i][j][k] / (Coe.s * Coe._eps[i][j][k]));
                    NNZ++;

                    D[ex111] = 1 / (1.0 + Coe._cey[i][j][k] * Coe._chy[i][j][k] + Coe._cey[i][j][k]
                        * Coe._chy[i][j - 1][k] + Coe._cez[i][j][k] * Coe._chz[i][j][k]
                        + Coe._cez[i][j][k] * Coe._chz[i][j][k - 1] + 2 * Coe._sigmax[i][j][k] / (Coe.s * Coe._eps[i][j][k]));

                    if (Coe._sigmax[i][j][k] != 0)
                    {
                        I[ex111] = (1.0 + Coe._cey[i][j][k] * Coe._chy[i][j][k] + Coe._cey[i][j][k]
                            * Coe._chy[i][j - 1][k] + Coe._cez[i][j][k] * Coe._chz[i][j][k]
                            + Coe._cez[i][j][k] * Coe._chz[i][j][k - 1])
                            / (1.0 + Coe._cey[i][j][k] * Coe._chy[i][j][k] + Coe._cey[i][j][k]
                                * Coe._chy[i][j - 1][k] + Coe._cez[i][j][k] * Coe._chz[i][j][k]
                                + Coe._cez[i][j][k] * Coe._chz[i][j][k - 1] + 2 * Coe._sigmax[i][j][k] / (Coe.s * Coe._eps[i][j][k]));

                    }

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex121);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey211);
                    VAL.emplace_back(Coe._cey[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey111);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chx[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex101);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey201);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chx[i][j - 1][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey101);
                    VAL.emplace_back(Coe._cey[i][j][k] * Coe._chx[i][j - 1][k]); NNZ++;

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez211);
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez111);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex112);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez210);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chx[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez110);
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chx[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex110);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k - 1]); NNZ++;

                }
            }
        }

        // Ey equations
        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ey111 = Coe._nodeNum[i][j][k][1];

                    ey112 = Coe._nodeNum[i][j][k + 1][1];
                    ez121 = Coe._nodeNum[i][j + 1][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];

                    ey110 = Coe._nodeNum[i][j][k - 1][1];
                    ez120 = Coe._nodeNum[i][j + 1][k - 1][2];
                    ez110 = Coe._nodeNum[i][j][k - 1][2];

                    ex121 = Coe._nodeNum[i][j + 1][k][0];
                    ex111 = Coe._nodeNum[i][j][k][0];
                    ey211 = Coe._nodeNum[i + 1][j][k][1];

                    ex021 = Coe._nodeNum[i - 1][j + 1][k][0];
                    ex011 = Coe._nodeNum[i - 1][j][k][0];
                    ey011 = Coe._nodeNum[i - 1][j][k][1];

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                    VAL.emplace_back(1.0 + Coe._cez[i][j][k] * Coe._chz[i][j][k] + Coe._cez[i][j][k]
                        * Coe._chz[i][j][k - 1] + Coe._cex[i][j][k] * Coe._chx[i][j][k]
                        + Coe._cex[i][j][k] * Coe._chx[i - 1][j][k] + 2 * Coe._sigmay[i][j][k] / (Coe.s * Coe._eps[i][j][k]));
                    NNZ++;

                    D[ey111] = 1 / (1.0 + Coe._cez[i][j][k] * Coe._chz[i][j][k] + Coe._cez[i][j][k]
                        * Coe._chz[i][j][k - 1] + Coe._cex[i][j][k] * Coe._chx[i][j][k]
                        + Coe._cex[i][j][k] * Coe._chx[i - 1][j][k] + 2 * Coe._sigmay[i][j][k] / (Coe.s * Coe._eps[i][j][k]));

                    if (Coe._sigmay[i][j][k] != 0)
                    {
                        I[ey111] = (1.0 + Coe._cez[i][j][k] * Coe._chz[i][j][k] + Coe._cez[i][j][k]
                            * Coe._chz[i][j][k - 1] + Coe._cex[i][j][k] * Coe._chx[i][j][k]
                            + Coe._cex[i][j][k] * Coe._chx[i - 1][j][k])
                            / (1.0 + Coe._cez[i][j][k] * Coe._chz[i][j][k] + Coe._cez[i][j][k]
                                * Coe._chz[i][j][k - 1] + Coe._cex[i][j][k] * Coe._chx[i][j][k]
                                + Coe._cex[i][j][k] * Coe._chx[i - 1][j][k] + 2 * Coe._sigmay[i][j][k] / (Coe.s * Coe._eps[i][j][k]));

                    }

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey112);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez121);;
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez111);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey110);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez120);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chy[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez110);
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chy[i][j][k - 1]); NNZ++;

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex121);
                    VAL.emplace_back(Coe._cex[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex111);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey211);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex021);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chy[i - 1][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex011);;
                    VAL.emplace_back(Coe._cex[i][j][k] * Coe._chy[i - 1][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey011);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                }
            }
        }

        // Ez equations
        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    if (Coe._Jz[i][j][k] == 1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];

                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        ex111 = Coe._nodeNum[i][j][k][0];

                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        ex012 = Coe._nodeNum[i - 1][j][k + 1][0];
                        ex011 = Coe._nodeNum[i - 1][j][k][0];

                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        ey102 = Coe._nodeNum[i][j - 1][k + 1][1];
                        ey101 = Coe._nodeNum[i][j - 1][k][1];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k])
                            + 2 * Coe._dze[k] / (Coe.s * Coe._eps[i][j][k] * Coe._dxh[i] * Coe._dyh[j] * Rs));
                        NNZ++;

                        D[ez111] = 1 / (1.0 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k])
                            + 2 * Coe._dze[k] / (Coe.s * Coe._eps[i][j][k] * Coe._dxh[i] * Coe._dyh[j] * Rs));


                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex112);
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex012);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex011);
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey112);
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey102);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey101);
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;

                    }
                    else if (Coe._Rz[i][j][k] == 1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];

                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        ex111 = Coe._nodeNum[i][j][k][0];

                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        ex012 = Coe._nodeNum[i - 1][j][k + 1][0];
                        ex011 = Coe._nodeNum[i - 1][j][k][0];

                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        ey102 = Coe._nodeNum[i][j - 1][k + 1][1];
                        ey101 = Coe._nodeNum[i][j - 1][k][1];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);;
                        VAL.emplace_back(1.0 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k])
                            + 2 * Coe._dze[k] / (Coe.s * Coe._eps[i][j][k] * Coe._dxh[i] * Coe._dyh[j] * Rl));
                        NNZ++;

                        D[ez111] = 1 / (1.0 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k])
                            + 2 * Coe._dze[k] / (Coe.s * Coe._eps[i][j][k] * Coe._dxh[i] * Coe._dyh[j] * Rl));


                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex112);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex111);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex012);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex011);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey112);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey111);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey102);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey101);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;

                    }
                    else
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];

                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        ex111 = Coe._nodeNum[i][j][k][0];

                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        ex012 = Coe._nodeNum[i - 1][j][k + 1][0];
                        ex011 = Coe._nodeNum[i - 1][j][k][0];

                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        ey102 = Coe._nodeNum[i][j - 1][k + 1][1];
                        ey101 = Coe._nodeNum[i][j - 1][k][1];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);;
                        VAL.emplace_back(1.0 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k]));
                        NNZ++;

                        D[ez111] = 1 / (1.0 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k]));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex112);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex111);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex012);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex011);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey112);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey111);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey102);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey101);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;

                    }
                }
            }
        }

        // Outmost ABC boundary for Ex

        for (int i = 0; i < Coe.nx; ++i)
        {

            // Edge
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((j == 0) && (k == 0)) // Case (a-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex122 = Coe._nodeNum[i][j + 1][k + 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex122);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((j == 0) && (k == Coe.nz)) // Case (a-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex120 = Coe._nodeNum[i][j + 1][k - 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex120);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((j == Coe.ny) && (k == 0)) // Case (a-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex102 = Coe._nodeNum[i][j - 1][k + 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex102);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((j == Coe.ny) && (k == Coe.nz)) // Case (a-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex100 = Coe._nodeNum[i][j - 1][k - 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex100);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            // Face
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    if (j == 0) // Case (1-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex121 = Coe._nodeNum[i][j + 1][k][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / Coe._dye[j] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex121);
                        VAL.emplace_back(-1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (j == Coe.ny) // Case (1-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex101 = Coe._nodeNum[i][j - 1][k][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex101);
                        VAL.emplace_back(-1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            for (int j = 1; j < Coe.ny; j++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (1-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / Coe._dze[k] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex112);
                        VAL.emplace_back(-1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (k == Coe.nz) // Case (1-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex110 = Coe._nodeNum[i][j][k - 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ex111] = 1 / (1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex110);
                        VAL.emplace_back(-1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }
        }

        // Outmost ABC boundary for Ey

        for (int j = 0; j < Coe.ny; ++j)
        {

            // Edge
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((i == 0) && (k == 0)) // Case (b-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey212 = Coe._nodeNum[i + 1][j][k + 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey212);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((k == 0) && (i == Coe.nx)) // Case (b-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey012 = Coe._nodeNum[i - 1][j][k + 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey012);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((k == Coe.nz) && (i == 0)) // Case (b-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey210 = Coe._nodeNum[i + 1][j][k - 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey210);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((i == Coe.nx) && (k == Coe.nz)) // Case (b-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey010 = Coe._nodeNum[i - 1][j][k - 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey010);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            // Face
            for (int i = 1; i < Coe.nx; i++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (2-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey112 = Coe._nodeNum[i][j][k + 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / Coe._dze[k] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey112);
                        VAL.emplace_back(-1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (k == Coe.nz) // Case (2-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey110 = Coe._nodeNum[i][j][k - 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey110);
                        VAL.emplace_back(-1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 1; k < Coe.nz; k++)
                {
                    if (i == 0) // Case (2-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey211 = Coe._nodeNum[i + 1][j][k][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey211);
                        VAL.emplace_back(-1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (i == Coe.nx) // Case (2-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey011 = Coe._nodeNum[i - 1][j][k][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ey111] = 1 / (1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey011);
                        VAL.emplace_back(-1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }
        }

        // Outmost ABC boundary for Ez

        for (int k = 0; k < Coe.nz; ++k) {

            // Edge

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx) {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny) {
                    if ((i == 0) && (j == 0)) // Case (c-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez221 = Coe._nodeNum[i + 1][j + 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez221);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if ((i == 0) && (j == Coe.ny)) // Case (c-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez201 = Coe._nodeNum[i + 1][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez201);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if ((i == Coe.nx) && (j == 0)) // Case (c-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez021 = Coe._nodeNum[i - 1][j + 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez021);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if ((i == Coe.nx) && (j == Coe.ny)) // Case (c-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez001 = Coe._nodeNum[i - 1][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez001);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                }
            }

            // Face
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx) {
                for (int j = 1; j < Coe.ny; j++) {
                    if (i == 0) // Case (3-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez211 = Coe._nodeNum[i + 1][j][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);
                        VAL.emplace_back(-1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if (i == Coe.nx) // Case (3-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez011 = Coe._nodeNum[i - 1][j][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0));


                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);
                        VAL.emplace_back(-1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                }
            }

            for (int i = 1; i < Coe.nx; i++) {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny) {
                    if (j == 0) // Case (3-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / Coe._dye[j] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);
                        VAL.emplace_back(-1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if (j == Coe.ny) // Case (3-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        D[ez111] = 1 / (1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0));

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);
                        VAL.emplace_back(-1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                }
            }
        }
    }
    else
    {
        printf("Constructing Sparse Matrix A ...\n");
        // Ex equations
        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ex111 = Coe._nodeNum[i][j][k][0];

                    ex121 = Coe._nodeNum[i][j + 1][k][0];
                    ey211 = Coe._nodeNum[i + 1][j][k][1];
                    ey111 = Coe._nodeNum[i][j][k][1];

                    ex101 = Coe._nodeNum[i][j - 1][k][0];
                    ey201 = Coe._nodeNum[i + 1][j - 1][k][1];
                    ey101 = Coe._nodeNum[i][j - 1][k][1];

                    ez211 = Coe._nodeNum[i + 1][j][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];
                    ex112 = Coe._nodeNum[i][j][k + 1][0];

                    ez210 = Coe._nodeNum[i + 1][j][k - 1][2];
                    ez110 = Coe._nodeNum[i][j][k - 1][2];
                    ex110 = Coe._nodeNum[i][j][k - 1][0];

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                    VAL.emplace_back(1.0 + Coe._cey[i][j][k] * Coe._chy[i][j][k] + Coe._cey[i][j][k]
                        * Coe._chy[i][j - 1][k] + Coe._cez[i][j][k] * Coe._chz[i][j][k]
                        + Coe._cez[i][j][k] * Coe._chz[i][j][k - 1] + 2 * Coe._sigmax[i][j][k] / (Coe.s * Coe._eps[i][j][k]));
                    NNZ++;

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex121);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey211);
                    VAL.emplace_back(Coe._cey[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey111);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chx[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex101);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey201);
                    VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chx[i][j - 1][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ey101);
                    VAL.emplace_back(Coe._cey[i][j][k] * Coe._chx[i][j - 1][k]); NNZ++;

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez211);
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez111);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex112);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez210);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chx[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ez110);
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chx[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex110);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k - 1]); NNZ++;

                }
            }
        }

        // Ey equations
        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ey111 = Coe._nodeNum[i][j][k][1];

                    ey112 = Coe._nodeNum[i][j][k + 1][1];
                    ez121 = Coe._nodeNum[i][j + 1][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];

                    ey110 = Coe._nodeNum[i][j][k - 1][1];
                    ez120 = Coe._nodeNum[i][j + 1][k - 1][2];
                    ez110 = Coe._nodeNum[i][j][k - 1][2];

                    ex121 = Coe._nodeNum[i][j + 1][k][0];
                    ex111 = Coe._nodeNum[i][j][k][0];
                    ey211 = Coe._nodeNum[i + 1][j][k][1];

                    ex021 = Coe._nodeNum[i - 1][j + 1][k][0];
                    ex011 = Coe._nodeNum[i - 1][j][k][0];
                    ey011 = Coe._nodeNum[i - 1][j][k][1];

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                    VAL.emplace_back(1.0 + Coe._cez[i][j][k] * Coe._chz[i][j][k] + Coe._cez[i][j][k]
                        * Coe._chz[i][j][k - 1] + Coe._cex[i][j][k] * Coe._chx[i][j][k]
                        + Coe._cex[i][j][k] * Coe._chx[i - 1][j][k] + 2 * Coe._sigmay[i][j][k] / (Coe.s * Coe._eps[i][j][k]));
                    NNZ++;

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey112);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez121);;
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez111);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey110);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chz[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez120);
                    VAL.emplace_back(-Coe._cez[i][j][k] * Coe._chy[i][j][k - 1]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ez110);
                    VAL.emplace_back(Coe._cez[i][j][k] * Coe._chy[i][j][k - 1]); NNZ++;

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex121);
                    VAL.emplace_back(Coe._cex[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex111);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chy[i][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey211);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;

                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex021);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chy[i - 1][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ex011);;
                    VAL.emplace_back(Coe._cex[i][j][k] * Coe._chy[i - 1][j][k]); NNZ++;
                    IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey011);
                    VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                }
            }
        }

        // Ez equations
        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    if (Coe._Jz[i][j][k] == 1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];

                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        ex111 = Coe._nodeNum[i][j][k][0];

                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        ex012 = Coe._nodeNum[i - 1][j][k + 1][0];
                        ex011 = Coe._nodeNum[i - 1][j][k][0];

                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        ey102 = Coe._nodeNum[i][j - 1][k + 1][1];
                        ey101 = Coe._nodeNum[i][j - 1][k][1];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k])
                            + 2 * Coe._dze[k] / (Coe.s * Coe._eps[i][j][k] * Coe._dxh[i] * Coe._dyh[j] * Rs));
                        NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex112);
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex012);
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex011);
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey112);
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey102);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey101);
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;

                    }
                    else if (Coe._Rz[i][j][k] == 1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];

                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        ex111 = Coe._nodeNum[i][j][k][0];

                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        ex012 = Coe._nodeNum[i - 1][j][k + 1][0];
                        ex011 = Coe._nodeNum[i - 1][j][k][0];

                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        ey102 = Coe._nodeNum[i][j - 1][k + 1][1];
                        ey101 = Coe._nodeNum[i][j - 1][k][1];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);;
                        VAL.emplace_back(1 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k])
                            + 2 * Coe._dze[k] / (Coe.s * Coe._eps[i][j][k] * Coe._dxh[i] * Coe._dyh[j] * Rl));
                        NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex112);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex111);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex012);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex011);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey112);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey111);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey102);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey101);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;

                    }
                    else
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];

                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        ex111 = Coe._nodeNum[i][j][k][0];

                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        ex012 = Coe._nodeNum[i - 1][j][k + 1][0];
                        ex011 = Coe._nodeNum[i - 1][j][k][0];

                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        ey102 = Coe._nodeNum[i][j - 1][k + 1][1];
                        ey101 = Coe._nodeNum[i][j - 1][k][1];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);;
                        VAL.emplace_back(1 + Coe._cex[i][j][k] * Coe._chx[i][j][k] + Coe._cex[i][j][k]
                            * Coe._chx[i - 1][j][k] + Coe._cey[i][j][k] * Coe._chy[i][j][k]
                            + Coe._cey[i][j][k] * Coe._chy[i][j - 1][k] + 2 * Coe._sigmaz[i][j][k] / (Coe.s * Coe._eps[i][j][k]));
                        NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex112);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex111);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chx[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex012);;
                        VAL.emplace_back(-Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ex011);;
                        VAL.emplace_back(Coe._cex[i][j][k] * Coe._chz[i - 1][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey112);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey111);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j][k]); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey102);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ey101);;
                        VAL.emplace_back(Coe._cey[i][j][k] * Coe._chz[i][j - 1][k]); NNZ++;
                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);;
                        VAL.emplace_back(-Coe._cey[i][j][k] * Coe._chy[i][j - 1][k]); NNZ++;

                    }
                }
            }
        }

        // Outmost ABC boundary for Ex
        for (int i = 0; i < Coe.nx; ++i)
        {

            // Edge
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((j == 0) && (k == 0)) // Case (a-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex122 = Coe._nodeNum[i][j + 1][k + 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex122);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((j == 0) && (k == Coe.nz)) // Case (a-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex120 = Coe._nodeNum[i][j + 1][k - 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex120);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((j == Coe.ny) && (k == 0)) // Case (a-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex102 = Coe._nodeNum[i][j - 1][k + 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex102);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((j == Coe.ny) && (k == Coe.nz)) // Case (a-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex100 = Coe._nodeNum[i][j - 1][k - 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex100);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dye[j - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            // Face
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    if (j == 0) // Case (1-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex121 = Coe._nodeNum[i][j + 1][k][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex121);
                        VAL.emplace_back(-1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (j == Coe.ny) // Case (1-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex101 = Coe._nodeNum[i][j - 1][k][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex101);
                        VAL.emplace_back(-1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            for (int j = 1; j < Coe.ny; j++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (1-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex112);
                        VAL.emplace_back(-1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (k == Coe.nz) // Case (1-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex110 = Coe._nodeNum[i][j][k - 1][0];

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex111);
                        VAL.emplace_back(1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ex111 }); JA.emplace_back(ex110);
                        VAL.emplace_back(-1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }
        }

        // Outmost ABC boundary for Ey
        for (int j = 0; j < Coe.ny; ++j)
        {

            // Edge
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((i == 0) && (k == 0)) // Case (b-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey212 = Coe._nodeNum[i + 1][j][k + 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey212);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((k == 0) && (i == Coe.nx)) // Case (b-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey012 = Coe._nodeNum[i - 1][j][k + 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey012);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((k == Coe.nz) && (i == 0)) // Case (b-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey210 = Coe._nodeNum[i + 1][j][k - 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey210);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if ((i == Coe.nx) && (k == Coe.nz)) // Case (b-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey010 = Coe._nodeNum[i - 1][j][k - 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey010);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dze[k - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            // Face
            for (int i = 1; i < Coe.nx; i++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (2-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey112 = Coe._nodeNum[i][j][k + 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey112);
                        VAL.emplace_back(-1.0 / Coe._dze[k] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (k == Coe.nz) // Case (2-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey110 = Coe._nodeNum[i][j][k - 1][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey110);
                        VAL.emplace_back(-1.0 / Coe._dze[k - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 1; k < Coe.nz; k++)
                {
                    if (i == 0) // Case (2-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey211 = Coe._nodeNum[i + 1][j][k][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey211);
                        VAL.emplace_back(-1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                    else if (i == Coe.nx) // Case (2-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey011 = Coe._nodeNum[i - 1][j][k][1];

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey111);
                        VAL.emplace_back(1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ey111 }); JA.emplace_back(ey011);
                        VAL.emplace_back(-1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;
                    }
                }
            }
        }

        // Outmost ABC boundary for Ez
        for (int k = 0; k < Coe.nz; ++k) {

            // Edge

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx) {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny) {
                    if ((i == 0) && (j == 0)) // Case (c-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez221 = Coe._nodeNum[i + 1][j + 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez221);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if ((i == 0) && (j == Coe.ny)) // Case (c-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez201 = Coe._nodeNum[i + 1][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez201);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if ((i == Coe.nx) && (j == 0)) // Case (c-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez021 = Coe._nodeNum[i - 1][j + 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez021);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if ((i == Coe.nx) && (j == Coe.ny)) // Case (c-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez001 = Coe._nodeNum[i - 1][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez001);
                        VAL.emplace_back(-1.0 / (sqrt(pow(Coe._dxe[i - 1], 2) + pow(Coe._dye[j - 1], 2))) + Coe.s / (4.0 * v0)); NNZ++;

                    }
                }
            }

            // Face
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx) {
                for (int j = 1; j < Coe.ny; j++) {
                    if (i == 0) // Case (3-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez211 = Coe._nodeNum[i + 1][j][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez211);
                        VAL.emplace_back(-1.0 / Coe._dxe[i] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if (i == Coe.nx) // Case (3-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez011 = Coe._nodeNum[i - 1][j][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez011);
                        VAL.emplace_back(-1.0 / Coe._dxe[i - 1] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                }
            }

            for (int i = 1; i < Coe.nx; i++) {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny) {
                    if (j == 0) // Case (3-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez121);
                        VAL.emplace_back(-1.0 / Coe._dye[j] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                    else if (j == Coe.ny) // Case (3-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez111);
                        VAL.emplace_back(1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;

                        IA.push_back({ NNZ,ez111 }); JA.emplace_back(ez101);
                        VAL.emplace_back(-1.0 / Coe._dye[j - 1] + Coe.s / (4.0 * v0)); NNZ++;

                    }
                }
            }
        }


    } // CUDA_AWARE


}

void LFDTD::Solver_Select()
{

    int Solver_select;
    int Precon_select;

    std::cout << "Please select the solver: [1] - PARDISO, [2]- cuSPARSE, [3] - CUDA RT, [4] - CUDA RT Expanded Kernel (Testing)" << std::endl;
    std::cin >> Solver_select;

    switch (Solver_select)
    {
    case 1:
        _Solver = _PARDISO;
        break;
    case 2:
        _Solver = _cuSPARSE;
        break;
    case 3:
        _Solver = _CUDA;
        break;
    case 4:
        _Solver = _CUDA_Expanded;
        break;
    default:
        std::cerr << "No valid solver matches input selection!" << std::endl;;
        std::cerr << "Program exiting..." << std::endl;;
        throw - 1;
        break;
    }

    if (_Solver)
    {
        std::cout << "Please select the preconditioner: [1] - None, [2] - Jacobi, [3] - Laguerre" << std::endl;
        std::cin >> Precon_select;
        switch (Precon_select)
        {
        case 1:
            _M = None;
            break;
        case 2:
            _M = Jacobi;
            break;
        case 3:
            _M = Laguerre;
            break;
        default:
            std::cout << "No valid preconditioner matches input selection!" << std::endl;
            std::cout << "Default to Jacobi Preconditioner for solving." << std::endl;
            _M = Jacobi;
            break;
        }
    }
}

void LFDTD::COO2CSR()
{
    ia = std::make_unique<int[]>(Nnode + 1);
    a = std::make_unique<double[]>(NNZ);
    ja = std::make_unique<int[]>(NNZ);

    std::vector<std::pair<int, int>> JA_Group;
    std::vector<int> JA_Sorted_Idx;

    std::sort(IA.begin(), IA.end(), cmp);

    if (_Solver)
    {
        for (int i = 0; i < NNZ - 1; ++i)
        {

            if (IA[i].second == IA[i + 1].second)
            {
                JA_Group.push_back({ IA[i].first,JA[IA[i].first] });
            }
            else
            {
                JA_Group.push_back({ IA[i].first, JA[IA[i].first] });
                std::sort(JA_Group.begin(), JA_Group.end(), cmp);

                for (auto& ele : JA_Group)
                {
                    JA_Sorted_Idx.push_back(ele.first);
                }

                JA_Group.clear();

            }

            ia[IA[i].second + 1]++;
        }

        // Account for the last element
        JA_Group.push_back({ IA[NNZ - 1].first, JA[IA[NNZ - 1].first] });
        std::sort(JA_Group.begin(), JA_Group.end(), cmp);

        for (auto& ele : JA_Group)
        {
            JA_Sorted_Idx.push_back(ele.first);
        }

        JA_Group.clear();
        ia[IA[NNZ - 1].second + 1]++;

        // 0-index in row column

        for (int i = 0; i < Nnode; ++i)
        {
            ia[i + 1] += ia[i];
        }

        for (int i = 0; i < NNZ; ++i)
        {
            ja[i] = JA[JA_Sorted_Idx[i]];
            a[i] = VAL[JA_Sorted_Idx[i]];
        }

        /* Create CUBLAS context */
        checkCudaErrors(cublasCreate(&cublasHandle));

        /* Create CUSPARSE context */
        checkCudaErrors(cusparseCreate(&cusparseHandle));

        /* Description of the A matrix */
        checkCudaErrors(cusparseCreateMatDescr(&descr));

        /* Allocate required memory */
        checkCudaErrors(cudaMalloc((void**)&d_col, NNZ * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_row, (Nnode + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_val, NNZ * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_x, Nnode * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_r, Nnode * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_r0, Nnode * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_p, Nnode * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_AP, Nnode * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_AS, Nnode * sizeof(double)));

        /* Wrap raw data into cuSPARSE generic API objects - Dense Vector on RHS */
        checkCudaErrors(cusparseCreateDnVec(&vecR, Nnode, d_r, CUDA_R_64F));
        checkCudaErrors(cusparseCreateDnVec(&vecP, Nnode, d_p, CUDA_R_64F));
        checkCudaErrors(cusparseCreateDnVec(&vecAP, Nnode, d_AP, CUDA_R_64F));
        checkCudaErrors(cusparseCreateDnVec(&vecAS, Nnode, d_AS, CUDA_R_64F));

        /* Initialize matrix data */
        checkCudaErrors(cudaMemcpy(
            d_col, ja.get(), NNZ * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            d_row, ia.get(), (Nnode + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            d_val, a.get(), NNZ * sizeof(double), cudaMemcpyHostToDevice));

        /* Create CSR A matrix on GPU */
        checkCudaErrors(cusparseCreateCsr(
            &matA, Nnode, Nnode, NNZ, d_row, d_col, d_val, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        /* Initialize host & device variable for kernel based CUDA implementation */

        h_nrmr0 = std::make_unique<double>();
        h_nrmr = std::make_unique<double>();
        h_APr0 = std::make_unique<double>();
        h_ASAS = std::make_unique<double>();
        h_ASsj = std::make_unique<double>();
        h_rjjr0 = std::make_unique<double>();

        checkCudaErrors(cudaMalloc((void**)&d_nrmr0, sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_nrmr, sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_APr0, sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_ASAS, sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_ASsj, sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&d_rjjr0, sizeof(double)));

        if (_M == None)
        {
            printf("Solving Ax=b without Preconditioner M\n");

            /* Allocate workspace for cuSPARSE */
            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
                vecP, &doublezero, vecAP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeAP));
            checkCudaErrors(cudaMalloc(&d_bufferSizeAP, bufferSizeAP));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
                vecR, &doublezero, vecAS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeAS));
            checkCudaErrors(cudaMalloc(&d_bufferSizeAS, bufferSizeAS));
        }
        else if (_M == Jacobi)
        {

            printf("Solving Ax=b with Jacobi Preconditioner D\n");

            /* Allocate required memory */
            checkCudaErrors(cudaMalloc((void**)&d_col_m, Nnode * sizeof(int)));
            checkCudaErrors(cudaMalloc((void**)&d_row_m, (Nnode + 1) * sizeof(int)));
            checkCudaErrors(cudaMalloc((void**)&d_val_m, Nnode * sizeof(double)));

            /* Initialize Pre-conditioner matrix data */
            checkCudaErrors(cudaMemcpy(
                d_col_m, ja_M.get(), Nnode * sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_row_m, ia_M.get(), (Nnode + 1) * sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_val_m, D.get(), Nnode * sizeof(double), cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc((void**)&d_MP, Nnode * sizeof(double)));
            checkCudaErrors(cusparseCreateDnVec(&vecMP, Nnode, d_MP, CUDA_R_64F));

            checkCudaErrors(cudaMalloc((void**)&d_MS, Nnode * sizeof(double)));
            checkCudaErrors(cusparseCreateDnVec(&vecMS, Nnode, d_MS, CUDA_R_64F));

            /* Create CSR Pre-conditioner M matrix on GPU */
            checkCudaErrors(cusparseCreateCsr(
                &matM, Nnode, Nnode, Nnode, d_row_m, d_col_m, d_val_m, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
                vecP, &doublezero, vecMP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeMP));
            checkCudaErrors(cudaMalloc(&d_bufferSizeMP, bufferSizeMP));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
                vecMP, &doublezero, vecAP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeAP));
            checkCudaErrors(cudaMalloc(&d_bufferSizeAP, bufferSizeAP));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
                vecR, &doublezero, vecMS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeMS));
            checkCudaErrors(cudaMalloc(&d_bufferSizeMS, bufferSizeMS));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
                vecMS, &doublezero, vecAS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeAS));
            checkCudaErrors(cudaMalloc(&d_bufferSizeAS, bufferSizeAS));
        }
        else if (_M == Laguerre)
        {

            printf("Solving Ax=b with Proposed Preconditioner I\n");

            /* Allocate required memory */
            checkCudaErrors(cudaMalloc((void**)&d_col_m, Nnode * sizeof(int)));
            checkCudaErrors(cudaMalloc((void**)&d_row_m, (Nnode + 1) * sizeof(int)));
            checkCudaErrors(cudaMalloc((void**)&d_val_m, Nnode * sizeof(double)));

            /* Initialize Pre-conditioner matrix data */
            checkCudaErrors(cudaMemcpy(
                d_col_m, ja_M.get(), Nnode * sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_row_m, ia_M.get(), (Nnode + 1) * sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_val_m, I.get(), Nnode * sizeof(double), cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc((void**)&d_MP, Nnode * sizeof(double)));
            checkCudaErrors(cusparseCreateDnVec(&vecMP, Nnode, d_MP, CUDA_R_64F));

            checkCudaErrors(cudaMalloc((void**)&d_MS, Nnode * sizeof(double)));
            checkCudaErrors(cusparseCreateDnVec(&vecMS, Nnode, d_MS, CUDA_R_64F));

            /* Create CSR Pre-conditioner M matrix on GPU */
            checkCudaErrors(cusparseCreateCsr(
                &matM, Nnode, Nnode, Nnode, d_row_m, d_col_m, d_val_m, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
                vecP, &doublezero, vecMP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeMP));
            checkCudaErrors(cudaMalloc(&d_bufferSizeMP, bufferSizeMP));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
                vecMP, &doublezero, vecAP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeAP));
            checkCudaErrors(cudaMalloc(&d_bufferSizeAP, bufferSizeAP));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
                vecR, &doublezero, vecMS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeMS));
            checkCudaErrors(cudaMalloc(&d_bufferSizeMS, bufferSizeMS));

            checkCudaErrors(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
                vecMS, &doublezero, vecAS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSizeAS));
            checkCudaErrors(cudaMalloc(&d_bufferSizeAS, bufferSizeAS));
        }

    }
    else
    {
        for (int i = 0; i < NNZ - 1; ++i)
        {

            if (IA[i].second == IA[i + 1].second)
            {
                JA_Group.push_back({ IA[i].first,JA[IA[i].first] });
            }
            else
            {
                JA_Group.push_back({ IA[i].first, JA[IA[i].first] });
                std::sort(JA_Group.begin(), JA_Group.end(), cmp);

                for (auto& ele : JA_Group)
                {
                    JA_Sorted_Idx.push_back(ele.first);
                }

                JA_Group.clear();

            }

            ia[IA[i].second + 1]++;
        }

        // Account for the last element
        JA_Group.push_back({ IA[NNZ - 1].first, JA[IA[NNZ - 1].first] });
        std::sort(JA_Group.begin(), JA_Group.end(), cmp);

        for (auto& ele : JA_Group)
        {
            JA_Sorted_Idx.push_back(ele.first);
        }

        JA_Group.clear();
        ia[IA[NNZ - 1].second + 1]++;

        // 1-index in row column
        ia[0] = 1;

        for (int i = 0; i < Nnode; ++i)
        {
            ia[i + 1] += ia[i];
        }

        for (int i = 0; i < NNZ; ++i)
        {
            ja[i] = JA[JA_Sorted_Idx[i]] + 1;
            a[i] = VAL[JA_Sorted_Idx[i]];
        }
    }


    IA.clear();
    JA.clear();
    VAL.clear();
    JA_Sorted_Idx.clear();

    printf("Sparse Matrix A Successfully Populated with CSR Format!\n");


}

void LFDTD::CSR_Expanded()
{
    a_expanded = std::make_unique<double[]>(Nnode * 16);
    ja_expanded = std::make_unique<int[]>(Nnode * 16);

    int count = 0;
    for (int i = 0; i < Nnode; ++i)
    {
        for (int j = ia[i]; j < ia[i + 1]; ++j)
        {
            ja_expanded[i * 16 + count] = ja[j];
            a_expanded[i * 16 + count] = a[j];
            ++count;
        }
        count = 0;
    }

}

void LFDTD::BiCGSTABL_Solver()
{


    /* Initialize r0 = b - A*x0  on GPU (Assume the initial guess x0 is zero) */
    checkCudaErrors(cudaMemcpy(d_r, b.get(), Nnode * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x.get(), Nnode * sizeof(double), cudaMemcpyHostToDevice));
    /*2: Set \tilde{r0} = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_r0, 1));
    /*2: Set p0 = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_p, 1));

    /* Calculate initial residual nrmr0 */
    checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr0));
    checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_r0, 1, d_r, 1, &rjjr0));

    iter = 0;

    while (iter <= maxit)
    {
        if (iter > 0)
        {
            /* Beta = (r_{j+1}, \tilde{r0})/(r_j, \tilde{r0}) X (alpha_j/W_j) */
            beta = (rjjr0 / rjr0) * (alpha / omega);

            /* P_j = P_j - W_j*AP_j */
            checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &nomega, d_AP, 1, d_p, 1));
            /* P_j = b_j * P_j */
            checkCudaErrors(cublasDscal(cublasHandle, Nnode, &beta, d_p, 1));
            /* P_{j+1} = r_{j+1} + P_j */
            checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &doubleone, d_r, 1, d_p, 1));

        }

        rjr0 = rjjr0;

        /* Vec_{AP_j} = A*P_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecP, &doublezero, vecAP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAP));

        /* (AP_j, r0) */
        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_AP, 1, d_r0, 1, &APr0));

        alpha = rjr0 / APr0;
        nalpha = -alpha;

        /* r_j = r_j - alpha*AP_j */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &nalpha, d_AP, 1, d_r, 1));
        /* x_{j+1} = x_j + alpha*P_j */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &alpha, d_p, 1, d_x, 1));

        // Check convergence
        checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr));
        if (nrmr / nrmr0 < tol)
        {
            break;
        }

        /* Vec_{AS_j} = A*r_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecR, &doublezero, vecAS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAS));

        /* (AS_j, S_j(r_j) ) */
        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_AS, 1, d_r, 1, &ASsj));
        /* (AS_j, AS_j ) */
        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_AS, 1, d_AS, 1, &ASAS));

        /* omega = (AS_j, S_j(r_j) )/(AS_j, AS_j ) */
        omega = ASsj / ASAS;
        nomega = -omega;

        /* x_{j+1} = x_j + omega*S_j(r_j) */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &omega, d_r, 1, d_x, 1));
        /* r_{j+1} = S_j(r_j) - omega*AS_j */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &nomega, d_AS, 1, d_r, 1));

        // Check convergence
        checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr));
        if (nrmr / nrmr0 < tol)
        {
            break;
        }

        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_r0, 1, d_r, 1, &rjjr0));

        iter++;
    }

    FLAG = (nrmr / nrmr0 <= tol) ? 0 : 1;

    if (FLAG == 0)
    {
        //printf("Number of iteration to converge is %d\n",iter);
        checkCudaErrors(cudaMemcpy(
            x.get(), d_x, Nnode * sizeof(double), cudaMemcpyDeviceToHost));
    }
    else
    {
        const char* no_convergence = "No Convergence reached!\n";
        printf("Convergence FLAG is: %d, with residual of: %e\n", FLAG, nrmr);
        throw no_convergence;
    }

}
void LFDTD::BiCGSTABL_M_Solver()
{

    /* Initialize r0 = b - A*x0  on GPU (Assume the initial guess x0 is zero) */
    checkCudaErrors(cudaMemcpy(d_r, b.get(), Nnode * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cublasDscal(cublasHandle, Nnode, &doublezero, d_x, 1));
    /*2: Set \tilde{r0} = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_r0, 1));
    /*2: Set p0 = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_p, 1));

    /* Calculate initial residual nrmr0 */
    checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr0));
    checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_r0, 1, d_r, 1, &rjjr0));

    iter = 0;

    while (iter <= maxit)
    {
        if (iter > 0)
        {
            /* Beta = (r_{j+1}, \tilde{r0})/(r_j, \tilde{r0}) X (alpha_j/W_j) */
            beta = (rjjr0 / rjr0) * (alpha / omega);

            /* P_j = P_j - W_j*AP_j */
            checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &nomega, d_AP, 1, d_p, 1));
            /* P_j = b_j * P_j */
            checkCudaErrors(cublasDscal(cublasHandle, Nnode, &beta, d_p, 1));
            /* P_{j+1} = r_{j+1} + P_j */
            checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &doubleone, d_r, 1, d_p, 1));

        }

        rjr0 = rjjr0;

        /* Vec_{MP_j} = inv(M)*P_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
            vecP, &doublezero, vecMP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeMP));

        /* Vec_{AP_j} = A*MP_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecMP, &doublezero, vecAP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAP));

        /* (AP_j, r0) */
        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_AP, 1, d_r0, 1, &APr0));

        alpha = rjr0 / APr0;
        nalpha = -alpha;

        /* r_j = r_j - alpha*AP_j */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &nalpha, d_AP, 1, d_r, 1));
        /* x_{j+1} = x_j + alpha*MP_j */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &alpha, d_MP, 1, d_x, 1));

        // Check convergence
        checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr));
        if (nrmr / nrmr0 < tol)
        {
            break;
        }

        /* Vec_{MS_j} = inv(M)*r_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
            vecR, &doublezero, vecMS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeMS));

        /* Vec_{AS_j} = A*MS_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecMS, &doublezero, vecAS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAS));

        /* (AS_j, S_j(r_j) ) */
        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_AS, 1, d_r, 1, &ASsj));
        /* (AS_j, AS_j ) */
        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_AS, 1, d_AS, 1, &ASAS));

        /* omega = (AS_j, S_j(r_j) )/(AS_j, AS_j ) */
        omega = ASsj / ASAS;
        nomega = -omega;

        /* x_{j+1} = x_j + omega*MS_j(Mr_j) */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &omega, d_MS, 1, d_x, 1));
        /* r_{j+1} = S_j(r_j) - omega*AS_j */
        checkCudaErrors(cublasDaxpy(cublasHandle, Nnode, &nomega, d_AS, 1, d_r, 1));

        // Check convergence
        checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr));
        if (nrmr / nrmr0 < tol)
        {
            break;
        }

        checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_r0, 1, d_r, 1, &rjjr0));

        iter++;
    }

    FLAG = (nrmr / nrmr0 <= tol) ? 0 : 1;

    if (FLAG == 0)
    {
        Convergence.emplace_back(iter);
        // printf("Number of iteration to converge is %d\n",iter);
        checkCudaErrors(cudaMemcpy(
            x.get(), d_x, Nnode * sizeof(double), cudaMemcpyDeviceToHost));
    }
    else
    {
        Convergence.emplace_back(iter);
        const char* no_convergence = "No Convergence reached!\n";
        printf("Convergence FLAG is: %d, with residual of: %e\n", FLAG, nrmr);
        throw no_convergence;
    }

}
/* spMV for A* b uses cuSPARSE Library - Ideal for 30 series GPU model (Test on 3080TI) */
void LFDTD::BiCGSTABL_M_Kernel_Solver()
{
    dim3 grid(Nnode / (8 * warp) + 1, 1, 1);
    dim3 block(8 * warp, 1, 1);

    /* Initialize r0 = b - A*x0  on GPU (Assume the initial guess x0 is zero) */

    checkCudaErrors(cudaMemcpy(d_r, b.get(), Nnode * sizeof(double), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_x, 0, Nnode * sizeof(double)));

    checkCudaErrors(cublasDscal(cublasHandle, Nnode, &doublezero, d_x, 1));

    /*2: Set \tilde{r0} = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_r0, 1));
    /*2: Set p0 = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_p, 1));

    /* Calculate initial residual nrmr0 */

    cuBLAS::nrm2(grid, block, d_r, h_nrmr0.get(), d_nrmr0);
    cuBLAS::dot_product(grid, block, d_r0, d_r, h_rjjr0.get(), d_rjjr0);

    iter = 0;

    while (iter < maxit)
    {
        if (iter > 0)
        {

            // Beta = (r_{j+1}, \tilde{r0})/(r_j, \tilde{r0}) X (alpha_j/W_j) 
            beta = ((*h_rjjr0) / rjr0) * (alpha / omega);

            // P_{j+1} = r_{j+1} + beta_j * (P_j - omega_j*AP_j)
            cuBLAS::p_update(grid, block, d_p, d_AP, d_r, omega, beta);

        }

        rjr0 = (*h_rjjr0);

        /* Vec_{MP_j} = inv(M)*P_j */

        cuBLAS::spMV_M(grid, block, d_val_m, d_p, d_MP);

        /*
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
            vecP, &doublezero, vecMP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeMP));
        */

        /* Vec_{AP_j} = A*MP_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecMP, &doublezero, vecAP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAP));

        /* (AP_j, r0) */
        cuBLAS::dot_product(grid, block, d_AP, d_r0, h_APr0.get(), d_APr0);

        alpha = rjr0 / (*h_APr0);
        nalpha = -alpha;

        
        // r_j = r_j - alpha*AP_j
        //cuBLAS::axpy(grid, block, d_AP, d_r, nalpha);

        // x_{j+1} = x_j + alpha*MP_j
        //cuBLAS::axpy(grid, block, d_MP, d_x, alpha);
        

        /*
            x_{j+1} = x_j + alpha*MP_j
            r_j = r_j - alpha*AP_j
        */
        cuBLAS::axpy_V2(grid, block, d_MP, d_x, d_AP, d_r, alpha);

        // Check convergence
        cuBLAS::nrm2(grid, block, d_r, h_nrmr.get(), d_nrmr);

        if ((*h_nrmr) / (*h_nrmr0) < tol)
        {
            break;
        }

        /* Vec_{MS_j} = inv(M)*r_j */

        cuBLAS::spMV_M(grid, block, d_val_m, d_r, d_MS);


        /* Vec_{AS_j} = A*MS_j */
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecMS, &doublezero, vecAS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAS));

        /* (AS_j, S_j(r_j) ) */
        //cuBLAS::dot_product(grid, block, d_AS, d_r, h_ASsj.get(), d_ASsj);
        /* (AS_j, AS_j ) */
        //cuBLAS::dot_product(grid, block, d_AS, d_AS, h_ASAS.get(), d_ASAS);

        /*
            (AS_j, AS_j)
            (AS_j, S_j(r_j))
        */
        cuBLAS::dot_product_V2(grid, block, d_AS, d_r, h_ASAS.get(), d_ASAS, h_ASsj.get(), d_ASsj);

        /* omega = (AS_j, S_j(r_j) )/(AS_j, AS_j ) */
        omega = (*h_ASsj) / (*h_ASAS);

        /*
        //x_{j+1} = x_j + omega*MS_j(Mr_j)
        cuBLAS::axpy(grid, block, d_MS, d_x, omega);
        // r_{j+1} = S_j(r_j) - omega*AS_j
        cuBLAS::axpy(grid, block, d_AS, d_r, nomega);
        */

        /*
            x_{j+1} = x_j + omega*MS_j(Mr_j)
            r_{j+1} = S_j(r_j) - omega*AS_j
        */
        cuBLAS::axpy_V2(grid, block, d_MS, d_x, d_AS, d_r, omega);

        // Check convergence
        cuBLAS::nrm2(grid, block, d_r, h_nrmr.get(), d_nrmr);

        if ((*h_nrmr) / (*h_nrmr0) < tol)
        {
            break;
        }

        cuBLAS::dot_product(grid, block, d_r0, d_r, h_rjjr0.get(), d_rjjr0);

        iter++;
    }

    FLAG = ((*h_nrmr) / (*h_nrmr0) <= tol) ? 0 : 1;

    if (FLAG == 0)
    {
        Convergence.emplace_back(iter);
        // printf("Number of iteration to converge is %d\n",iter);

        checkCudaErrors(cudaMemcpy(
            x.get(), d_x, Nnode * sizeof(double), cudaMemcpyDeviceToHost));

    }
    else
    {
        Convergence.emplace_back(iter);
        const char* no_convergence = "No Convergence reached!\n";
        printf("Convergence FLAG is: %d, with residual of: %e\n", FLAG, (*h_nrmr) / (*h_nrmr0));
        throw no_convergence;
    }

}

/* Fully expanded custom CUDA runtime kernel solver - Performs best on 40 series GPU model (Test on 4090) */
void LFDTD::BiCGSTABL_M_Expanded_Kernel_Solver()
{
    dim3 grid(Nnode / (8 * warp) + 1, 1, 1);
    dim3 block(8 * warp, 1, 1);

    dim3 grid_expanded(Nnode / (2 * 16) + 1, 1, 1);
    dim3 block_expanded(16 * warp, 1, 1);

    /* Initialize r0 = b - A*x0  on GPU (Assume the initial guess x0 is zero) */

    checkCudaErrors(cudaMemcpy(d_r, b.get(), Nnode * sizeof(double), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_x, 0, Nnode * sizeof(double)));

    checkCudaErrors(cublasDscal(cublasHandle, Nnode, &doublezero, d_x, 1));

    /*2: Set \tilde{r0} = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_r0, 1));
    /*2: Set p0 = r0 */
    checkCudaErrors(cublasDcopy(cublasHandle, Nnode, d_r, 1, d_p, 1));

    /* Calculate initial residual nrmr0 */

    cuBLAS::nrm2(grid, block, d_r, h_nrmr0.get(), d_nrmr0);
    // checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr0));
    cuBLAS::dot_product(grid, block, d_r0, d_r, h_rjjr0.get(), d_rjjr0);
    // checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_r0, 1, d_r, 1, &rjjr0));

    iter = 0;

    while (iter < maxit)
    {
        if (iter > 0)
        {
            /* Beta = (r_{j+1}, \tilde{r0})/(r_j, \tilde{r0}) X (alpha_j/W_j) */
            beta = ((*h_rjjr0) / rjr0) * (alpha / omega);

            // P_{j+1} = r_{j+1} + beta_j * (P_j - omega_j*AP_j)
            cuBLAS::p_update(grid, block, d_p, d_AP, d_r, omega, beta);

        }

        rjr0 = (*h_rjjr0);

        /* Vec_{MP_j} = inv(M)*P_j */
        cuBLAS::spMV_M(grid, block, d_val_m, d_p, d_MP);

        /*
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
            vecP, &doublezero, vecMP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeMP));
        */

        /* Vec_{AP_j} = A*MP_j */

        cuBLAS::spMV(grid_expanded, block_expanded, d_a_expanded, d_ja_expanded, d_MP, spMV_buffer, d_AP);

        /*
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecMP, &doublezero, vecAP, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAP));
        */

        /* (AP_j, r0) */
        cuBLAS::dot_product(grid, block, d_AP, d_r0, h_APr0.get(), d_APr0);
        // checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_AP, 1, d_r0, 1, &APr0));

        alpha = rjr0 / (*h_APr0);

        /*
        //r_j = r_j - alpha*AP_j
        cuBLAS::axpy(grid, block, d_AP, d_r, nalpha);

        // x_{j+1} = x_j + alpha*MP_j
        cuBLAS::axpy(grid, block, d_MP, d_x, alpha);
        */

        /*
            x_{j+1} = x_j + alpha*MP_j
            r_j = r_j - alpha*AP_j
        */
        cuBLAS::axpy_V2(grid, block, d_MP, d_x, d_AP, d_r, alpha);

        // Check convergence
        cuBLAS::nrm2(grid, block, d_r, h_nrmr.get(), d_nrmr);
        // checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr));


        if ((*h_nrmr) / (*h_nrmr0) < tol)
        {
            break;
        }

        /* Vec_{MS_j} = inv(M)*r_j */

        cuBLAS::spMV_M(grid, block, d_val_m, d_r, d_MS);

        /*
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM,
            vecR, &doublezero, vecMS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeMS));
        */

        /* Vec_{AS_j} = A*MS_j */

        cuBLAS::spMV(grid_expanded, block_expanded, d_a_expanded, d_ja_expanded, d_MS, spMV_buffer, d_AS);

        /*
        checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecMS, &doublezero, vecAS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferSizeAS));
        */

        /* (AS_j, S_j(r_j) ) */
        // cuBLAS::dot_product(grid, block, d_AS, d_r, h_ASsj.get(), d_ASsj);
        /* (AS_j, AS_j ) */
        // cuBLAS::dot_product(grid, block, d_AS, d_AS, h_ASAS.get(), d_ASAS);

        /*
            (AS_j, AS_j)
            (AS_j, S_j(r_j))
        */
        cuBLAS::dot_product_V2(grid, block, d_AS, d_r, h_ASAS.get(), d_ASAS, h_ASsj.get(), d_ASsj);

        /* omega = (AS_j, S_j(r_j) )/(AS_j, AS_j ) */
        omega = (*h_ASsj) / (*h_ASAS);

        /*
        //x_{j+1} = x_j + omega*MS_j(Mr_j)
        cuBLAS::axpy(grid, block, d_MS, d_x, omega);
        // r_{j+1} = S_j(r_j) - omega*AS_j
        cuBLAS::axpy(grid, block, d_AS, d_r, nomega);
        */

        /*
            x_{j+1} = x_j + omega*MS_j(Mr_j)
            r_{j+1} = S_j(r_j) - omega*AS_j
        */
        cuBLAS::axpy_V2(grid, block, d_MS, d_x, d_AS, d_r, omega);

        // Check convergence
        cuBLAS::nrm2(grid, block, d_r, h_nrmr.get(), d_nrmr);
        //checkCudaErrors(cublasDnrm2(cublasHandle, Nnode, d_r, 1, &nrmr));

        if ((*h_nrmr) / (*h_nrmr0) < tol)
        {
            break;
        }

        cuBLAS::dot_product(grid, block, d_r0, d_r, h_rjjr0.get(), d_rjjr0);
        // checkCudaErrors(cublasDdot(cublasHandle, Nnode, d_r0, 1, d_r, 1, &rjjr0));

        iter++;
    }

    FLAG = ((*h_nrmr) / (*h_nrmr0) <= tol) ? 0 : 1;

    if (FLAG == 0)
    {
        Convergence.emplace_back(iter);
        // printf("Number of iteration to converge is %d\n",iter);

        checkCudaErrors(cudaMemcpy(
            x.get(), d_x, Nnode * sizeof(double), cudaMemcpyDeviceToHost));

    }
    else
    {
        Convergence.emplace_back(iter);
        const char* no_convergence = "No Convergence reached!\n";
        printf("Convergence FLAG is: %d, with residual of: %e\n", FLAG, (*h_nrmr) / (*h_nrmr0));
        throw no_convergence;
    }

}

void LFDTD::Intel_PARDISO(LFDTD_Coe& Coe)
{
    PrintQ_set(1);

    SparseA_COO(Coe);
    COO2CSR();

    recordEq = std::make_unique<double[]>(Coe.num_probe * (Coe.qstop + 1));
    probe = std::make_unique<double[]>(Coe.num_probe * Coe.tStep);
    lagPoly = std::make_unique<double[]>(4 * Coe.tStep);
    lagPoly_sum = std::make_unique<double[]>(Coe.tStep);
    vtg = std::make_unique<double[]>(Coe.tStep);

    int Pos, q(0);
    double jq(0);

    auto start = std::chrono::high_resolution_clock::now();

    // PARDISO Set up
    MKL_INT mtype = 11;       /* Real unsymmetric matrix */
    // Descriptor of main sparse matrix properties
    struct matrix_descr descrA;
    // Structure with sparse matrix stored in CSR format
    sparse_matrix_t       csrA;
    /* RHS and solution vectors. */
    MKL_INT nrhs = 1;     /* Number of right hand sides. */
    /* Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    void* pt[64];
    /* Pardiso control parameters. */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    double ddum;          /* Double dummy */
    MKL_INT idum;         /* Integer dummy. */
    /* -------------------------------------------------------------------- */
    /* .. Setup Pardiso control parameters. */
    /* -------------------------------------------------------------------- */
    for (int i = 0; i < 64; i++)
    {
        iparm[i] = 0;
    }
    iparm[0] = 1;         /* No solver default */
    iparm[1] = 2;         /* Fill-in reordering from METIS */
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[6] = 0;         /* Not in use */
    iparm[7] = 2;         /* Max numbers of iterative refinement stCoe._eps */
    iparm[8] = 0;         /* Not in use */
    iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;        /* Conjugate transposed/transpose solve */
    iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[14] = 0;        /* Not in use */
    iparm[15] = 0;        /* Not in use */
    iparm[16] = 0;        /* Not in use */
    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;         /* Which factorization to use. */
    msglvl = 0;           /* Print statistical information  */
    error = 0;            /* Initialize error flag */
    /* -------------------------------------------------------------------- */
    /* .. Initialize the internal solver memory pointer. This is only */
    /* necessary for the FIRST call of the PARDISO solver. */
    /* -------------------------------------------------------------------- */
    for (int i = 0; i < 64; i++)
    {
        pt[i] = 0;
    }
    /* -------------------------------------------------------------------- */
    /* .. Reordering and Symbolic Factorization. This step also allocates */
    /* all memory that is necessary for the factorization. */
    /* -------------------------------------------------------------------- */
    phase = 11;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
        &Nnode, a.get(), ia.get(), ja.get(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

    if (error != 0)
    {
        printf("\nERROR during symbolic factorization: " IFORMAT, error);
        exit(1);
    }
    printf("\nReordering completed ... ");
    printf("\nNumber of nonzeros in factors = " IFORMAT, iparm[17]);
    printf("\nNumber of factorization MFLOPS = " IFORMAT, iparm[18]);
    /* -------------------------------------------------------------------- */
    /* .. Numerical factorization. */
    /* -------------------------------------------------------------------- */
    phase = 22;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
        &Nnode, a.get(), ia.get(), ja.get(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if (error != 0)
    {
        printf("\nERROR during numerical factorization: " IFORMAT, error);
        exit(2);
    }
    printf("\nFactorization completed ... \n");
    /* -------------------------------------------------------------------- */
    /* .. Back substitution and iterative refinement. */
    /* -------------------------------------------------------------------- */
    phase = 33;

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.mode = SPARSE_FILL_MODE_UPPER;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, Nnode, Nnode, ia.get(), ia.get() + 1, ja.get(), a.get());

    while (q <= Coe.qstop)
    {
        // Calculate Laguerre Polynomial
        if (q == 0)
        {
            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[2 * Coe.tStep + i] = 1.0;
                lagPoly[3 * Coe.tStep + i] = -Coe.s * (i + 1) * Coe.dt / 2;
            }
        }
        else if (q == 1)
        {
            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[1 * Coe.tStep + i] = lagPoly[2 * Coe.tStep + i];
                lagPoly[2 * Coe.tStep + i] = 1.0 - Coe.s * (i + 1) * Coe.dt;
            }
        }
        else
        {

            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[0 * Coe.tStep + i] = lagPoly[1 * Coe.tStep + i];
                lagPoly[1 * Coe.tStep + i] = lagPoly[2 * Coe.tStep + i];
            }

            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[2 * Coe.tStep + i] = (1.0 / q) * ((2.0 * q - 1.0 - Coe.s * (i + 1) * Coe.dt) * lagPoly[1 * Coe.tStep + i] - (q - 1) * lagPoly[0 * Coe.tStep + i]);
            }

            for (int i = 0; i < Coe.tStep; ++i)
            {
                if (lagPoly[2 * Coe.tStep + i] > 1e100) // Make sure that Laguerre polynomial does not go to infinity
                {
                    lagPoly[0 * Coe.tStep + i] = lagPoly[0 * Coe.tStep + i] * exp(-Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale);
                    lagPoly[1 * Coe.tStep + i] = lagPoly[1 * Coe.tStep + i] * exp(-Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale);
                    lagPoly[2 * Coe.tStep + i] = lagPoly[2 * Coe.tStep + i] * exp(-Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale);
                    lagPoly[3 * Coe.tStep + i] = lagPoly[3 * Coe.tStep + i] + Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale;
                }
            }

        }

        for (int i = 0; i < Coe.tStep; ++i)
        {
            lagPoly_sum[i] = lagPoly[2 * Coe.tStep + i] * exp(lagPoly[3 * Coe.tStep + i]);
        }

        // Compute Laguerre Coefficients for the source
        jq = 0;

        for (int i = 0; i < Coe.tStep; ++i)
        {
            jq += Coe._waveform[i] * (lagPoly[2 * Coe.tStep + i] * exp(lagPoly[3 * Coe.tStep + i])) * Coe.s * Coe.dt;
        }


        //printf("The jq has value of: %15.5e \n", jq);

        if (jDirecIndex[2] == 1)
        {
            for (int i = (Coe._jCellIndex[0] - 1); i < Coe._jCellIndex[1]; i++)
            {
                for (int j = (Coe._jCellIndex[2] - 1); j < Coe._jCellIndex[3]; j++)
                {
                    for (int k = (Coe._jCellIndex[4] - 1); k < Coe._jCellIndex[5]; k++)
                    {
                        Coe._Jz[i][j][k] = jq / Coe._JCount;
                    }
                }
            }
        }


        // Build b vector

        // std::fill_n(b.get(), Nnode, 0.0); // Clear b vector     

        // Ex equation except outmost PEC boundary
        // No re-assignment of b value for outmost PEC boundary

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ex111 = Coe._nodeNum[i][j][k][0];
                    b[ex111] = -2.0 * Coe._cey[i][j][k] * (Coe._sumHz[i][j][k] - Coe._sumHz[i][j - 1][k]) + 2 * Coe._cez[i][j][k] * (Coe._sumHy[i][j][k] - Coe._sumHy[i][j][k - 1])
                        - 2.0 / (Coe.s * Coe._eps[i][j][k]) * Coe._Jx[i][j][k] - 2 * sumE[ex111];
                }
            }
        }

        // Ey equation except outmost PEC boundary
        // No re-assignment of b value for outmost PEC boundary

        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ey111 = Coe._nodeNum[i][j][k][1];
                    b[ey111] = -2.0 * Coe._cez[i][j][k] * (Coe._sumHx[i][j][k] - Coe._sumHx[i][j][k - 1]) + 2 * Coe._cex[i][j][k] * (Coe._sumHz[i][j][k] - Coe._sumHz[i - 1][j][k])
                        - 2.0 / (Coe.s * Coe._eps[i][j][k]) * Coe._Jy[i][j][k] - 2 * sumE[ey111];
                }
            }
        }

        // Ez equation except outmost PEC boundary
        // No re-assignment of b value for outmost PEC boundary

        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ez111 = Coe._nodeNum[i][j][k][2];
                    b[ez111] = -2.0 * Coe._cex[i][j][k] * (Coe._sumHy[i][j][k] - Coe._sumHy[i - 1][j][k])
                        + 2.0 * Coe._cey[i][j][k] * (Coe._sumHx[i][j][k] - Coe._sumHx[i][j - 1][k])
                        - 2.0 / (Coe.s * Coe._eps[i][j][k]) * Coe._Jz[i][j][k] - 2.0 * sumE[ez111];
                }
            }
        }

        // Outmost ABC boundary for Ex

        for (int i = 0; i < Coe.nx; ++i)
        {

            // Edge
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((j == 0) && (k == 0)) // Case (a-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex122 = Coe._nodeNum[i][j + 1][k + 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex122]);
                    }
                    else if ((j == 0) && (k == Coe.nz)) // Case (a-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex120 = Coe._nodeNum[i][j + 1][k - 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex120]);
                    }
                    else if ((j == Coe.ny) && (k == 0)) // Case (a-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex102 = Coe._nodeNum[i][j - 1][k + 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex102]);
                    }
                    else if ((j == Coe.ny) && (k == Coe.nz)) // Case (a-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex100 = Coe._nodeNum[i][j - 1][k - 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex100]);
                    }
                }
            }

            // Face
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    if (j == 0) // Case (1-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex121 = Coe._nodeNum[i][j + 1][k][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex121]);
                    }
                    else if (j == Coe.ny) // Case (1-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex101 = Coe._nodeNum[i][j - 1][k][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex101]);
                    }
                }
            }

            for (int j = 1; j < Coe.ny; j++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (1-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex112]);
                    }
                    else if (k == Coe.nz) // Case (1-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex110 = Coe._nodeNum[i][j][k - 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex110]);
                    }
                }
            }
        }

        // Outmost ABC boundary for Ey

        for (int j = 0; j < Coe.ny; ++j)
        {

            // Edge
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((i == 0) && (k == 0)) // Case (b-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey212 = Coe._nodeNum[i + 1][j][k + 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey212]);
                    }
                    else if ((k == 0) && (i == Coe.nx)) // Case (b-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey012 = Coe._nodeNum[i - 1][j][k + 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey012]);
                    }
                    else if ((k == Coe.nz) && (i == 0)) // Case (b-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey210 = Coe._nodeNum[i + 1][j][k - 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey210]);
                    }
                    else if ((i == Coe.nx) && (k == Coe.nz)) // Case (b-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey010 = Coe._nodeNum[i - 1][j][k - 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey010]);
                    }
                }
            }

            // Face
            for (int i = 1; i < Coe.nx; i++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (2-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey112]);
                    }
                    else if (k == Coe.nz) // Case (2-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey110 = Coe._nodeNum[i][j][k - 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey110]);
                    }
                }
            }

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 1; k < Coe.nz; k++)
                {
                    if (i == 0) // Case (2-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey211 = Coe._nodeNum[i + 1][j][k][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey211]);
                    }
                    else if (i == Coe.nx) // Case (2-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey011 = Coe._nodeNum[i - 1][j][k][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey011]);
                    }
                }
            }
        }

        // Outmost ABC boundary for Ez

        for (int k = 0; k < Coe.nz; ++k)
        {

            // Edge

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
                {
                    if ((i == 0) && (j == 0)) // Case (c-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez221 = Coe._nodeNum[i + 1][j + 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez221]);
                    }
                    else if ((i == 0) && (j == Coe.ny)) // Case (c-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez201 = Coe._nodeNum[i + 1][j - 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez201]);
                    }
                    else if ((i == Coe.nx) && (j == 0)) // Case (c-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez021 = Coe._nodeNum[i - 1][j + 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez021]);
                    }
                    else if ((i == Coe.nx) && (j == Coe.ny)) // Case (c-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez001 = Coe._nodeNum[i - 1][j - 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez001]);
                    }
                }
            }

            // Face
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int j = 1; j < Coe.ny; j++)
                {
                    if (i == 0) // Case (3-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez211]);
                    }
                    else if (i == Coe.nx) // Case (3-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez011]);
                    }
                }
            }

            for (int i = 1; i < Coe.nx; i++)
            {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
                {
                    if (j == 0) // Case (3-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez121]);
                    }
                    else if (j == Coe.ny) // Case (3-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez101]);
                    }
                }
            }
        }

        /* Call Intel PARDISO/CUDA BiCGSTABL solver here */

        // printf("\n\nSolving system with iparm[11] = " IFORMAT " ...\n", iparm[11]);
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &Nnode, a.get(), ia.get(), ja.get(), &idum, &nrhs, iparm, &msglvl, b.get(), x.get(), &error);
        if (error != 0)
        {
            printf("\nERROR during solution: " IFORMAT, error);
            exit(3);
        }

        // Compute residual

        /*
        mkl_sparse_d_mv(transA, 1.0, csrA, descrA, x, 0.0, bs);
        res = 0.0;
        res0 = 0.0;
        for (int j = 1; j <= Nnode; j++)
        {
            res += (bs[j - 1] - b[j - 1]) * (bs[j - 1] - b[j - 1]);
            res0 += b[j - 1] * b[j - 1];
        }
        res = sqrt(res) / sqrt(res0);
        // printf("\nRelative residual = %e", res);
        // Check residual
        if (res > 1e-10)
        {
            printf("Error: residual is too high!\n");
            exit(10);
        }
        */

        // Update sumE

        for (int i = 0; i < Nnode; ++i)
        {
            sumE[i] += x[i];
        }

        // Update Hx

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ey112 = Coe._nodeNum[i][j][k + 1][1];
                    ey111 = Coe._nodeNum[i][j][k][1];
                    ez121 = Coe._nodeNum[i][j + 1][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];

                    Coe._hx[i][j][k] = Coe._chz[i][j][k] * (x[ey112] - x[ey111]) - Coe._chy[i][j][k] * (x[ez121] - x[ez111])
                        - 2.0 * Coe._sumHx[i][j][k];
                    Coe._sumHx[i][j][k] += Coe._hx[i][j][k];
                }
            }
        }

        // Update Hy

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ez211 = Coe._nodeNum[i + 1][j][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];
                    ex112 = Coe._nodeNum[i][j][k + 1][0];
                    ex111 = Coe._nodeNum[i][j][k][0];

                    Coe._hy[i][j][k] = Coe._chx[i][j][k] * (x[ez211] - x[ez111]) - Coe._chz[i][j][k] * (x[ex112] - x[ex111])
                        - 2.0 * Coe._sumHy[i][j][k];
                    Coe._sumHy[i][j][k] += Coe._hy[i][j][k];
                }
            }
        }

        // Update Hz

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ex121 = Coe._nodeNum[i][j + 1][k][0];
                    ex111 = Coe._nodeNum[i][j][k][0];
                    ey211 = Coe._nodeNum[i + 1][j][k][1];
                    ey111 = Coe._nodeNum[i][j][k][1];

                    Coe._hz[i][j][k] = Coe._chy[i][j][k] * (x[ex121] - x[ex111]) - Coe._chx[i][j][k] * (x[ey211] - x[ey111])
                        - 2.0 * Coe._sumHz[i][j][k];
                    Coe._sumHz[i][j][k] += Coe._hz[i][j][k];
                }
            }
        }

        // Print the basis coefficient for the port with the lowest x and y index

        if (PrintQ == 1)
        {
            printf("q = %5d:", q);
            for (int i = 0; i < Coe.num_probe; ++i)
            {
                if (probeDirecIndex[0] == 1)
                {
                    Pos = Coe._nodeNum[Coe._probeCell[i * 6 + 0] - 1][Coe._probeCell[i * 6 + 2] - 1][Coe._probeCell[i * 6 + 4] - 1][0];
                    printf("p%d = %15.5e;", i + 1, x[Pos]);
                }
                else if (probeDirecIndex[1] == 1)
                {
                    Pos = Coe._nodeNum[Coe._probeCell[i * 6 + 0] - 1][Coe._probeCell[i * 6 + 2] - 1][Coe._probeCell[i * 6 + 4] - 1][1];
                    printf("p%d = %15.5e;", i + 1, x[Pos]);
                }
                else if (probeDirecIndex[2] == 1)
                {
                    Pos = Coe._nodeNum[Coe._probeCell[i * 6 + 0] - 1][Coe._probeCell[i * 6 + 2] - 1][Coe._probeCell[i * 6 + 4] - 1][2];
                    printf("p%d = %15.5e;", i + 1, x[Pos]);
                }

                else
                {
                    printf("Probe printing error\n");
                    exit(0);
                }
            }
            printf("\n");
        }

        for (int n = 0; n < Coe.num_probe; ++n)
        {
            std::fill_n(vtg.get(), Coe.tStep, 0.0);

            for (int i = Coe._probeCell[n * 6 + 0] - 1; i < Coe._probeCell[n * 6 + 1]; ++i)
            {
                for (int j = Coe._probeCell[n * 6 + 2] - 1; j < Coe._probeCell[n * 6 + 3]; ++j)
                {
                    for (int k = Coe._probeCell[n * 6 + 4] - 1; k < Coe._probeCell[n * 6 + 5]; ++k)
                    {
                        Pos = Coe._nodeNum[i][j][k][2];
                        for (int l = 0; l < Coe.tStep; ++l)
                        {
                            vtg[l] += x[Pos] * lagPoly_sum[l] * (-Coe._dze[k] / (Coe._probeCell[n * 6 + 1] - Coe._probeCell[n * 6 + 0] + 1));
                            // probe[n][l] += vtg[l];
                        }
                    }
                }
            }
            for (int i = 0; i < Coe.tStep; i++)
            {
                probe[i + n * Coe.tStep] += vtg[i];
            }
            Pos = Coe._nodeNum[Coe._probeCell[n * 6 + 0] - 1][Coe._probeCell[n * 6 + 2] - 1][Coe._probeCell[n * 6 + 4] - 1][2];
            recordEq[q + n * (Coe.qstop + 1)] = x[Pos];
        }
        ++q;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Elapsed time: " << elapsed_time << " milliseconds" << std::endl;

    /* -------------------------------------------------------------------- */
    /* .. Termination and release of memory. */
    /* -------------------------------------------------------------------- */
    phase = -1;           /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &Nnode, &ddum, ia.get(), ja.get(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

    std::fill_n(sumE.get(), Nnode, 0.0);
    std::fill_n(lagPoly_sum.get(), Coe.tStep, 0.0);
}

void LFDTD::Nvidia_CUDA(LFDTD_Coe& Coe)
{
    PrintQ_set(1);

    SparseA_COO(Coe);
    COO2CSR();

    recordEq = std::make_unique<double[]>(Coe.num_probe * (Coe.qstop + 1));
    probe = std::make_unique<double[]>(Coe.num_probe * Coe.tStep);
    lagPoly = std::make_unique<double[]>(4 * Coe.tStep);
    lagPoly_sum = std::make_unique<double[]>(Coe.tStep);
    vtg = std::make_unique<double[]>(Coe.tStep);

    checkCudaErrors(cudaMallocHost((void**)&b_pinned, Nnode * sizeof(double)));
    checkCudaErrors(cudaMallocHost((void**)&x_pinned, Nnode * sizeof(double)));

    /* This will pick the best possible CUDA capable device */
    cudaDeviceProp deviceProp;
    int devID;
    checkCudaErrors(cudaGetDevice(&devID));
    printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0)
    {
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device */
    printf("> GPU device selected: %s\n", deviceProp.name);
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n",
        deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
    printf("> GPU device has Maximum %d blocks per multiprocessor\n",
        deviceProp.maxBlocksPerMultiProcessor);
    printf("> GPU device has %d 32-bit registers available per block, %d 32-bit registers available per multiprocessor\n",
        deviceProp.regsPerBlock, deviceProp.regsPerMultiprocessor);
    printf("> GPU device has Maximum %d threads per block, Maximum %d threads per multiprocessor\n",
        deviceProp.maxThreadsPerBlock, deviceProp.maxThreadsPerMultiProcessor);
    printf("> GPU device has %d Byte Shared MEM per block,%d Shared MEM per SM\n\n",
        deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerMultiprocessor);

    int Pos, q(0);
    double jq(0);

    if (_M == None)
    {
        iter_ptr = &LFDTD::BiCGSTABL_Solver;
    }
    else
    {
        if (_Solver == _cuSPARSE)
        {
            iter_ptr = &LFDTD::BiCGSTABL_M_Solver;
        }
        else if (_Solver == _CUDA)
        {
            cuBLAS::get_const_int_symbol(Nnode);
            iter_ptr = &LFDTD::BiCGSTABL_M_Kernel_Solver;
        }
        else
        {
            cuBLAS::get_const_int_symbol(Nnode);
            iter_ptr = &LFDTD::BiCGSTABL_M_Expanded_Kernel_Solver;
            CSR_Expanded(); // Only for the expanded spMV solver
            checkCudaErrors(cudaMalloc((void**)&spMV_buffer, 16 * Nnode * sizeof(double)));
            checkCudaErrors(cudaMalloc((void**)&d_a_expanded, 16 * Nnode * sizeof(double)));
            checkCudaErrors(cudaMalloc((void**)&d_ja_expanded, 16 * Nnode * sizeof(int)));
            checkCudaErrors(cudaMemcpy(d_a_expanded, a_expanded.get(), 16 * Nnode * sizeof(double), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_ja_expanded, ja_expanded.get(), 16 * Nnode * sizeof(int), cudaMemcpyHostToDevice));
        }

    }

    auto start = std::chrono::high_resolution_clock::now();

    printf("Simulation starts...\n");

    while (q <= Coe.qstop)
    {
        // Calculate Laguerre Polynomial
        if (q == 0)
        {
            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[2 * Coe.tStep + i] = 1.0;
                lagPoly[3 * Coe.tStep + i] = -Coe.s * (i + 1) * Coe.dt / 2;
            }
        }
        else if (q == 1)
        {
            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[1 * Coe.tStep + i] = lagPoly[2 * Coe.tStep + i];
                lagPoly[2 * Coe.tStep + i] = 1.0 - Coe.s * (i + 1) * Coe.dt;
            }
        }
        else
        {

            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[0 * Coe.tStep + i] = lagPoly[1 * Coe.tStep + i];
                lagPoly[1 * Coe.tStep + i] = lagPoly[2 * Coe.tStep + i];
            }

            for (int i = 0; i < Coe.tStep; ++i)
            {
                lagPoly[2 * Coe.tStep + i] = (1.0 / q) * ((2.0 * q - 1.0 - Coe.s * (i + 1) * Coe.dt) * lagPoly[1 * Coe.tStep + i] - (q - 1) * lagPoly[0 * Coe.tStep + i]);
            }

            for (int i = 0; i < Coe.tStep; ++i)
            {
                if (lagPoly[2 * Coe.tStep + i] > 1e100) // Make sure that Laguerre polynomial does not go to infinity
                {
                    lagPoly[0 * Coe.tStep + i] = lagPoly[0 * Coe.tStep + i] * exp(-Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale);
                    lagPoly[1 * Coe.tStep + i] = lagPoly[1 * Coe.tStep + i] * exp(-Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale);
                    lagPoly[2 * Coe.tStep + i] = lagPoly[2 * Coe.tStep + i] * exp(-Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale);
                    lagPoly[3 * Coe.tStep + i] = lagPoly[3 * Coe.tStep + i] + Coe.s * (i + 1) * Coe.dt / 2 * Coe.scale;
                }
            }

        }

        for (int i = 0; i < Coe.tStep; ++i)
        {
            lagPoly_sum[i] = lagPoly[2 * Coe.tStep + i] * exp(lagPoly[3 * Coe.tStep + i]);
        }

        // Compute Laguerre Coefficients for the source
        jq = 0;

        for (int i = 0; i < Coe.tStep; ++i)
        {
            jq += Coe._waveform[i] * (lagPoly[2 * Coe.tStep + i] * exp(lagPoly[3 * Coe.tStep + i])) * Coe.s * Coe.dt;
        }


        //printf("The jq has value of: %15.5e \n", jq);

        if (jDirecIndex[2] == 1)
        {
            for (int i = (Coe._jCellIndex[0] - 1); i < Coe._jCellIndex[1]; i++)
            {
                for (int j = (Coe._jCellIndex[2] - 1); j < Coe._jCellIndex[3]; j++)
                {
                    for (int k = (Coe._jCellIndex[4] - 1); k < Coe._jCellIndex[5]; k++)
                    {
                        Coe._Jz[i][j][k] = jq / Coe._JCount;
                    }
                }
            }
        }


        // Build b vector

        // std::fill_n(b_pinned, Nnode, 0.0); // Clear b vector   
        // std::fill_n(x_pinned, Nnode, 0.0); // Clear x vector  

        // Ex equation except outmost PEC boundary
        // No re-assignment of b value for outmost PEC boundary

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ex111 = Coe._nodeNum[i][j][k][0];
                    b[ex111] = -2.0 * Coe._cey[i][j][k] * (Coe._sumHz[i][j][k] - Coe._sumHz[i][j - 1][k]) + 2 * Coe._cez[i][j][k] * (Coe._sumHy[i][j][k] - Coe._sumHy[i][j][k - 1])
                        - 2.0 / (Coe.s * Coe._eps[i][j][k]) * Coe._Jx[i][j][k] - 2 * sumE[ex111];
                }
            }
        }

        // Ey equation except outmost PEC boundary
        // No re-assignment of b value for outmost PEC boundary

        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    ey111 = Coe._nodeNum[i][j][k][1];
                    b[ey111] = -2.0 * Coe._cez[i][j][k] * (Coe._sumHx[i][j][k] - Coe._sumHx[i][j][k - 1]) + 2 * Coe._cex[i][j][k] * (Coe._sumHz[i][j][k] - Coe._sumHz[i - 1][j][k])
                        - 2.0 / (Coe.s * Coe._eps[i][j][k]) * Coe._Jy[i][j][k] - 2 * sumE[ey111];
                }
            }
        }

        // Ez equation except outmost PEC boundary
        // No re-assignment of b value for outmost PEC boundary

        for (int i = 1; i < Coe.nx; ++i)
        {
            for (int j = 1; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ez111 = Coe._nodeNum[i][j][k][2];
                    b[ez111] = -2.0 * Coe._cex[i][j][k] * (Coe._sumHy[i][j][k] - Coe._sumHy[i - 1][j][k])
                        + 2.0 * Coe._cey[i][j][k] * (Coe._sumHx[i][j][k] - Coe._sumHx[i][j - 1][k])
                        - 2.0 / (Coe.s * Coe._eps[i][j][k]) * Coe._Jz[i][j][k] - 2.0 * sumE[ez111];
                }
            }
        }

        // Outmost ABC boundary for Ex

        for (int i = 0; i < Coe.nx; ++i)
        {

            // Edge
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((j == 0) && (k == 0)) // Case (a-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex122 = Coe._nodeNum[i][j + 1][k + 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex122]);
                    }
                    else if ((j == 0) && (k == Coe.nz)) // Case (a-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex120 = Coe._nodeNum[i][j + 1][k - 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex120]);
                    }
                    else if ((j == Coe.ny) && (k == 0)) // Case (a-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex102 = Coe._nodeNum[i][j - 1][k + 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex102]);
                    }
                    else if ((j == Coe.ny) && (k == Coe.nz)) // Case (a-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex100 = Coe._nodeNum[i][j - 1][k - 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex100]);
                    }
                }
            }

            // Face
            for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
            {
                for (int k = 1; k < Coe.nz; ++k)
                {
                    if (j == 0) // Case (1-1)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex121 = Coe._nodeNum[i][j + 1][k][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex121]);
                    }
                    else if (j == Coe.ny) // Case (1-2)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex101 = Coe._nodeNum[i][j - 1][k][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex101]);
                    }
                }
            }

            for (int j = 1; j < Coe.ny; j++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (1-3)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex112 = Coe._nodeNum[i][j][k + 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex112]);
                    }
                    else if (k == Coe.nz) // Case (1-4)
                    {
                        ex111 = Coe._nodeNum[i][j][k][0];
                        ex110 = Coe._nodeNum[i][j][k - 1][0];
                        b[ex111] = -Coe.s / (2.0 * v0) * (sumE[ex111] + sumE[ex110]);
                    }
                }
            }
        }

        // Outmost ABC boundary for Ey

        for (int j = 0; j < Coe.ny; ++j)
        {

            // Edge
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if ((i == 0) && (k == 0)) // Case (b-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey212 = Coe._nodeNum[i + 1][j][k + 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey212]);
                    }
                    else if ((k == 0) && (i == Coe.nx)) // Case (b-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey012 = Coe._nodeNum[i - 1][j][k + 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey012]);
                    }
                    else if ((k == Coe.nz) && (i == 0)) // Case (b-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey210 = Coe._nodeNum[i + 1][j][k - 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey210]);
                    }
                    else if ((i == Coe.nx) && (k == Coe.nz)) // Case (b-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey010 = Coe._nodeNum[i - 1][j][k - 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey010]);
                    }
                }
            }

            // Face
            for (int i = 1; i < Coe.nx; i++)
            {
                for (int k = 0; k < Coe.nz + 1; k += Coe.nz)
                {
                    if (k == 0) // Case (2-1)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey112 = Coe._nodeNum[i][j][k + 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey112]);
                    }
                    else if (k == Coe.nz) // Case (2-2)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey110 = Coe._nodeNum[i][j][k - 1][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey110]);
                    }
                }
            }

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int k = 1; k < Coe.nz; k++)
                {
                    if (i == 0) // Case (2-3)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey211 = Coe._nodeNum[i + 1][j][k][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey211]);
                    }
                    else if (i == Coe.nx) // Case (2-4)
                    {
                        ey111 = Coe._nodeNum[i][j][k][1];
                        ey011 = Coe._nodeNum[i - 1][j][k][1];
                        b[ey111] = -Coe.s / (2.0 * v0) * (sumE[ey111] + sumE[ey011]);
                    }
                }
            }
        }

        // Outmost ABC boundary for Ez

        for (int k = 0; k < Coe.nz; ++k)
        {

            // Edge

            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
                {
                    if ((i == 0) && (j == 0)) // Case (c-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez221 = Coe._nodeNum[i + 1][j + 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez221]);
                    }
                    else if ((i == 0) && (j == Coe.ny)) // Case (c-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez201 = Coe._nodeNum[i + 1][j - 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez201]);
                    }
                    else if ((i == Coe.nx) && (j == 0)) // Case (c-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez021 = Coe._nodeNum[i - 1][j + 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez021]);
                    }
                    else if ((i == Coe.nx) && (j == Coe.ny)) // Case (c-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez001 = Coe._nodeNum[i - 1][j - 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez001]);
                    }
                }
            }

            // Face
            for (int i = 0; i < Coe.nx + 1; i += Coe.nx)
            {
                for (int j = 1; j < Coe.ny; j++)
                {
                    if (i == 0) // Case (3-1)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez211 = Coe._nodeNum[i + 1][j][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez211]);
                    }
                    else if (i == Coe.nx) // Case (3-2)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez011 = Coe._nodeNum[i - 1][j][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez011]);
                    }
                }
            }

            for (int i = 1; i < Coe.nx; i++)
            {
                for (int j = 0; j < Coe.ny + 1; j += Coe.ny)
                {
                    if (j == 0) // Case (3-3)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez121 = Coe._nodeNum[i][j + 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez121]);
                    }
                    else if (j == Coe.ny) // Case (3-4)
                    {
                        ez111 = Coe._nodeNum[i][j][k][2];
                        ez101 = Coe._nodeNum[i][j - 1][k][2];
                        b[ez111] = -Coe.s / (2.0 * v0) * (sumE[ez111] + sumE[ez101]);
                    }
                }
            }
        }


        /* Call Intel CUDA BiCGSTABL solver here */

        try
        {
            (this->*iter_ptr)();
        }
        catch (const char* e)
        {
            std::cerr << e;
            break;
        }


        // Update sumE

        for (int i = 0; i < Nnode; ++i)
        {
            sumE[i] += x[i];
        }

        // Update Hx

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ey112 = Coe._nodeNum[i][j][k + 1][1];
                    ey111 = Coe._nodeNum[i][j][k][1];
                    ez121 = Coe._nodeNum[i][j + 1][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];

                    Coe._hx[i][j][k] = Coe._chz[i][j][k] * (x[ey112] - x[ey111]) - Coe._chy[i][j][k] * (x[ez121] - x[ez111])
                        - 2.0 * Coe._sumHx[i][j][k];
                    Coe._sumHx[i][j][k] += Coe._hx[i][j][k];
                }
            }
        }

        // Update Hy

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ez211 = Coe._nodeNum[i + 1][j][k][2];
                    ez111 = Coe._nodeNum[i][j][k][2];
                    ex112 = Coe._nodeNum[i][j][k + 1][0];
                    ex111 = Coe._nodeNum[i][j][k][0];

                    Coe._hy[i][j][k] = Coe._chx[i][j][k] * (x[ez211] - x[ez111]) - Coe._chz[i][j][k] * (x[ex112] - x[ex111])
                        - 2.0 * Coe._sumHy[i][j][k];
                    Coe._sumHy[i][j][k] += Coe._hy[i][j][k];
                }
            }
        }

        // Update Hz

        for (int i = 0; i < Coe.nx; ++i)
        {
            for (int j = 0; j < Coe.ny; ++j)
            {
                for (int k = 0; k < Coe.nz; ++k)
                {
                    ex121 = Coe._nodeNum[i][j + 1][k][0];
                    ex111 = Coe._nodeNum[i][j][k][0];
                    ey211 = Coe._nodeNum[i + 1][j][k][1];
                    ey111 = Coe._nodeNum[i][j][k][1];

                    Coe._hz[i][j][k] = Coe._chy[i][j][k] * (x[ex121] - x[ex111]) - Coe._chx[i][j][k] * (x[ey211] - x[ey111])
                        - 2.0 * Coe._sumHz[i][j][k];
                    Coe._sumHz[i][j][k] += Coe._hz[i][j][k];
                }
            }
        }

        // Print the basis coefficient for the port with the lowest x and y index

        if (PrintQ == 1)
        {
            printf("q = %5d:", q);
            for (int i = 0; i < Coe.num_probe; ++i)
            {
                if (probeDirecIndex[0] == 1)
                {
                    Pos = Coe._nodeNum[Coe._probeCell[i * 6 + 0] - 1][Coe._probeCell[i * 6 + 2] - 1][Coe._probeCell[i * 6 + 4] - 1][0];
                    printf("p%d = %15.5e;", i + 1, x[Pos]);
                }
                else if (probeDirecIndex[1] == 1)
                {
                    Pos = Coe._nodeNum[Coe._probeCell[i * 6 + 0] - 1][Coe._probeCell[i * 6 + 2] - 1][Coe._probeCell[i * 6 + 4] - 1][1];
                    printf("p%d = %15.5e;", i + 1, x[Pos]);
                }
                else if (probeDirecIndex[2] == 1)
                {
                    Pos = Coe._nodeNum[Coe._probeCell[i * 6 + 0] - 1][Coe._probeCell[i * 6 + 2] - 1][Coe._probeCell[i * 6 + 4] - 1][2];
                    printf("p%d = %15.5e;", i + 1, x[Pos]);
                }

                else
                {
                    printf("Probe printing error\n");
                    exit(0);
                }
            }
            printf("\n");
        }

        for (int n = 0; n < Coe.num_probe; ++n)
        {
            std::fill_n(vtg.get(), Coe.tStep, 0.0);

            for (int i = Coe._probeCell[n * 6 + 0] - 1; i < Coe._probeCell[n * 6 + 1]; ++i)
            {
                for (int j = Coe._probeCell[n * 6 + 2] - 1; j < Coe._probeCell[n * 6 + 3]; ++j)
                {
                    for (int k = Coe._probeCell[n * 6 + 4] - 1; k < Coe._probeCell[n * 6 + 5]; ++k)
                    {
                        Pos = Coe._nodeNum[i][j][k][2];
                        for (int l = 0; l < Coe.tStep; ++l)
                        {
                            vtg[l] += x[Pos] * lagPoly_sum[l] * (-Coe._dze[k] / (Coe._probeCell[n * 6 + 1] - Coe._probeCell[n * 6 + 0] + 1));
                            // probe[n][l] += vtg[l];
                        }
                    }
                }
            }
            for (int i = 0; i < Coe.tStep; i++)
            {
                probe[i + n * Coe.tStep] += vtg[i];
            }
            Pos = Coe._nodeNum[Coe._probeCell[n * 6 + 0] - 1][Coe._probeCell[n * 6 + 2] - 1][Coe._probeCell[n * 6 + 4] - 1][2];
            recordEq[q + n * (Coe.qstop + 1)] = x[Pos];
        }

        ++q;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Simulation ends with total %d orders solved!\n", Coe.qstop);
    printf("Elapsed time: %lld  milliseconds.\n", elapsed_time);

    std::fill_n(sumE.get(), Nnode, 0.0);
    std::fill_n(lagPoly_sum.get(), Coe.tStep, 0.0);
}

void LFDTD::Convergence_Profiler(const std::string& InputFile)
{
    std::ofstream myfile(InputFile);
    if (myfile.is_open())
    {
        for (int i = 0; i < Convergence.size(); ++i)
        {
            myfile << Convergence[i] << std::endl;
        }
        myfile.close();
    }

}

void LFDTD::result_write(const std::string& InputFile, const LFDTD_Coe& Coe)
{
    std::ofstream myfile(InputFile);
    if (myfile.is_open())
    {
        for (int count = 0; count < Coe.tStep; count++)
        {
            for (int i = 0; i < Coe.num_probe; i++)
            {
                myfile << probe[count + i * Coe.tStep] << " ";
            }
            myfile << std::endl;
        }
        myfile.close();
    }

}

void LFDTD::Eq_result_write(const std::string& InputFile, const LFDTD_Coe& Coe)
{
    std::ofstream myfile(InputFile);
    if (myfile.is_open())
    {
        for (int count = 0; count <= Coe.qstop; count++)
        {
            for (int i = 0; i < Coe.num_probe; i++)
            {
                myfile << recordEq[count + i * (Coe.qstop + 1)] << " ";
            }
            myfile << std::endl;
        }
        myfile.close();
    }
}

void LFDTD::result_write_app(const std::string& InputFile, const LFDTD_Coe& Coe)
{
    std::ofstream myfile(InputFile, std::ios::app);
    if (myfile.is_open())
    {
        for (int count = 0; count < Coe.tStep; count++)
        {
            for (int i = 0; i < Coe.num_probe; i++)
            {
                myfile << probe[count + i * Coe.tStep] << " ";
            }
            myfile << std::endl;
        }
        myfile.close();
    }

    std::fill_n(probe.get(), Coe.tStep, 0.0);
}

void LFDTD::Eq_result_write_app(const std::string& InputFile, const LFDTD_Coe& Coe)
{
    std::ofstream myfile(InputFile, std::ios::app);
    if (myfile.is_open())
    {
        for (int count = 0; count <= Coe.qstop; count++)
        {
            for (int i = 0; i < Coe.num_probe; i++)
            {
                myfile << recordEq[count + i * (Coe.qstop + 1)] << " ";
            }
            myfile << std::endl;
        }
        myfile.close();
    }
}
