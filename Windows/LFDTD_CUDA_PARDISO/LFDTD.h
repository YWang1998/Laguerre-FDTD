#pragma once

#include "LFDTD_Coe.h"
#include "spMv.cuh"

// Define the format to printf MKL_INT values
#if !defined(MKL_ILP64)
#define IFORMAT "%i"
#else
#define IFORMAT "%lli"
#endif

class LFDTD
{
public:

    LFDTD(); //Constructor

    typedef void (LFDTD::* BiCGSTABL_ptr)(); // BiCGSTABL solver ptr
    typedef void (LFDTD::* Sim_ptr)(LFDTD_Coe&); // PARDISO/CUDA solver ptr

    static void Solver_Select(); // Query the user to make a selection of CPU/GPU based solver

    void PrintQ_set(int); // Set whether to print out Q order on to screen

    void Sparse_A_Val(std::ifstream& a_file, int n, const int nnz); // This sets up the Sparse matrix entry value
    void SparseA_COO(const LFDTD_Coe& Coe);
    void COO2CSR();
    void CSR_Expanded();

    void BiCGSTABL_Solver();
    void BiCGSTABL_M_Solver();
    void BiCGSTABL_M_Kernel_Solver();
    void BiCGSTABL_M_Expanded_Kernel_Solver();

    void Intel_PARDISO(LFDTD_Coe& Coe);
    void Nvidia_CUDA(LFDTD_Coe& Coe);

    void Sim_start(LFDTD_Coe& Coe)
    {
        if (_Solver == Solver::_PARDISO)
        {
            sim_ptr = &LFDTD::Intel_PARDISO;
        }
        else
        {
            sim_ptr = &LFDTD::Nvidia_CUDA;
        }

        (this->*sim_ptr)(Coe);
    }

    void Convergence_Profiler(const std::string& InputFile);

    void result_write(const std::string& InputFile, const LFDTD_Coe&);
    void Eq_result_write(const std::string& InputFile, const LFDTD_Coe&);

    void result_write_app(const std::string& InputFile, const LFDTD_Coe&);
    void Eq_result_write_app(const std::string& InputFile, const LFDTD_Coe&);

    ~LFDTD()
    {
        /* Destroy contexts */
        if (static_cast<int>(_Solver))
        {

            checkCudaErrors(cudaFreeHost(x_pinned));
            checkCudaErrors(cudaFreeHost(b_pinned));

            checkCudaErrors(cusparseDestroy(cusparseHandle));
            checkCudaErrors(cublasDestroy(cublasHandle));
            if (_Solver == Solver::_CUDA_Expanded)
            {
                checkCudaErrors(cudaFree(spMV_buffer));
                checkCudaErrors(cudaFree(d_a_expanded));
                checkCudaErrors(cudaFree(d_ja_expanded));
            }

            checkCudaErrors(cudaFree(d_nrmr0));
            checkCudaErrors(cudaFree(d_nrmr));
            checkCudaErrors(cudaFree(d_APr0));
            checkCudaErrors(cudaFree(d_ASAS));
            checkCudaErrors(cudaFree(d_ASsj));
            checkCudaErrors(cudaFree(d_rjjr0));

            checkCudaErrors(cudaFree(d_col));
            checkCudaErrors(cudaFree(d_row));
            checkCudaErrors(cudaFree(d_val));
            checkCudaErrors(cudaFree(d_x));
            checkCudaErrors(cudaFree(d_AP));
            checkCudaErrors(cudaFree(d_AS));
            checkCudaErrors(cudaFree(d_r));
            checkCudaErrors(cudaFree(d_r0));
            checkCudaErrors(cudaFree(d_p));

            checkCudaErrors(cudaFree(d_bufferSizeAP));
            checkCudaErrors(cudaFree(d_bufferSizeAS));

            /* Destroy descriptors */
            checkCudaErrors(cusparseDestroySpMat(matA));
            checkCudaErrors(cusparseDestroyDnVec(vecR));
            checkCudaErrors(cusparseDestroyDnVec(vecP));
            checkCudaErrors(cusparseDestroyDnVec(vecAP));
            checkCudaErrors(cusparseDestroyDnVec(vecAS));

            checkCudaErrors(cudaFree(d_col_m));
            checkCudaErrors(cudaFree(d_row_m));
            checkCudaErrors(cudaFree(d_val_m));

            checkCudaErrors(cudaFree(d_MP));
            checkCudaErrors(cudaFree(d_MS));

            checkCudaErrors(cudaFree(d_bufferSizeMP));
            checkCudaErrors(cudaFree(d_bufferSizeMS));

            /* Destroy descriptors */
            checkCudaErrors(cusparseDestroySpMat(matM));
            checkCudaErrors(cusparseDestroyDnVec(vecMP));
            checkCudaErrors(cusparseDestroyDnVec(vecMS));
        }
    }

private:

    static Solver _Solver;
    static Precon _M;

    std::vector<std::pair<int, int>> IA;
    std::vector<int> JA;
    std::vector<double> VAL;

    std::unique_ptr<double[]> a;
    std::unique_ptr<double[]> a_expanded;
    std::unique_ptr<int[]> ja;
    std::unique_ptr<int[]> ja_expanded;
    std::unique_ptr<int[]> ia;

    std::unique_ptr<double[]> b;
    std::unique_ptr<double[]> x;

    double* b_pinned; // PINNED Memory for fast GPU-CPU data transfer
    double* x_pinned; // PINNED Memory for fast GPU-CPU data transfer

    std::unique_ptr<double[]> sumE;
    std::unique_ptr<double[]> lagPoly;
    std::unique_ptr<double[]> lagPoly_sum;
    std::unique_ptr<double[]> vtg;
    std::unique_ptr<double[]> probe;
    std::unique_ptr<double[]> recordEq;

    int PrintQ;
    int ex011, ex111, ey111, ez111, ex122, ex120, ex102, ex100, ex121, ex101, ex112, ex110, ex021, ex012;
    int ey101, ey212, ey012, ey210, ey010, ey112, ey110, ey201, ey211, ey011, ey102;
    int ez210, ez221, ez201, ez021, ez001, ez211, ez011, ez121, ez101, ez110, ez120;


    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;

    /* Description of the A matrix */
    cusparseMatDescr_t descr = 0;

    cusparseSpMatDescr_t matA = NULL;
    cusparseSpMatDescr_t matM = NULL;

    /* Wrap raw data into cuSPARSE generic API objects - Dense Vector on RHS */
    cusparseDnVecDescr_t vecR = NULL, vecP = NULL, vecAP = NULL, vecAS = NULL, vecMP = NULL, vecMS = NULL;

    std::vector<int> Convergence;

    /* Sparse Matrix Vector Multiplication Buffer Needed for NVIDIA cusparse api*/
    size_t bufferSizeAP, bufferSizeAS, bufferSizeMP, bufferSizeMS; // AS = AR
    void* d_bufferSizeAP, * d_bufferSizeAS, * d_bufferSizeMP, * d_bufferSizeMS;;

    std::unique_ptr<double> h_nrmr0;
    std::unique_ptr<double> h_nrmr;
    std::unique_ptr<double> h_APr0;
    std::unique_ptr<double> h_ASAS;
    std::unique_ptr<double> h_ASsj;
    std::unique_ptr<double> h_rjjr0;

    double* spMV_buffer, * d_a_expanded;// For expanded vector storage
    int* d_ja_expanded;
    double* d_nrmr0, * d_nrmr, * d_APr0, * d_ASAS, * d_ASsj, * d_rjjr0;
    double nrmr0, nrmr, alpha, omega, beta, nalpha, nomega, nbeta, rjjr0, rjr0;
    double APr0, ASsj, ASAS;
    const double doubleone = 1.0;
    const double doublezero = 0;
    int* d_col, * d_row, * d_col_m, * d_row_m;
    double* d_val, * d_val_m, * d_x, * d_AP, * d_AS;
    double* d_r, * d_r0, * d_p, * d_MP, * d_MS;
    int FLAG; // 0 - Converged; 1 - no convergency
    int iter;

    std::unique_ptr<int[]> ja_M;
    std::unique_ptr<int[]> ia_M;
    std::unique_ptr<double[]> D;
    std::unique_ptr<double[]> I; // Value for Jacobi Preconditioner and Propose Preconditioner

    BiCGSTABL_ptr iter_ptr;
    Sim_ptr sim_ptr;

};


