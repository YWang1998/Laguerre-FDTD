#pragma once
#include "global.h"

#define lanemask 0xFFFFFFFF

static __constant__ int d_Nnode; // length of the vector

static __constant__ double d_scale; // scale of vector-vector addition

static __constant__ double d_omega; // scale of vector-vector addition

static __constant__ double d_beta; // scale of vector-vector addition

/* Perform preconditioner matrix vector multiplication - a stream kernel with 2X improvement over cusparse APi */
template <typename T> __global__
void spMV_M_kernel(const T* d_M, const T* d_V, T* d_target);

/* Perform large global uncoalsed sparse matrix vector multiplication with summed spMV output */
template <int blockDIM, typename T> __global__
void spMV_sparse(const T* __restrict__ a_expand, const int* __restrict__ ja_expand,
	const T* __restrict__ v, T* __restrict__ v_expand, T* __restrict__ spMV);

/* spMV kernel with one thread per row - comparable with cusparseSpMV APi */
template <typename T> __global__
void spMV_thread_kernel(const int* d_ia, const int* d_ja, const T* d_val, const T* x, T* y);

/* spMV kernel with one warp (32 thread) per row - poor performance */
template <int blockDIM, typename T> __global__
void spMV_warp_kernel(const int* d_ia, const int* d_ja, const T* d_val, const T* x, T* y);

/* Fully looped dot product kernal with shared memory function - poor performance */
template <typename T> __global__
void dot_product_kernel(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ dot);

/* Fully unrolled templated dot product with shared memory function - best performance so far
	Runtime performance beats cublasDdot api on Nsight compute in release mode
	cublasDdot api performance is invariant to debug/release mode, which means the optimization is already done by Nvidia build-in api function
*/
template <int blockDIM, typename T> __global__
void dot_product_kernel_unroll(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ dot);

/*
	Two dot product operations in one kernel launch
*/
template <int blockDIM, typename T> __global__
void dot_product_kernel_V2_unroll(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ sum, T* __restrict__ dot);

/*  y = a*x + y
	similar performance of cublasDaxpy, can be used inter-changeably
*/
template <typename T> __global__
void axpy_kernal(const T* __restrict__ x, T* __restrict__ y);

/*
	Two axpy operations in one function call:
	1. y1 += scale * x1
	2. y2 -= scale * x2
*/
template <typename T> __global__
void axpy_kernal_V2(const T* __restrict__ x1, T* __restrict__ y1, const T* __restrict__ x2, T* __restrict__ y2);

/*
	This kernel combines the operation for P_{j+1} = r_{j+1} + beta_j * (P_j - omega_j*AP_j)
*/
template <typename T> __global__
void p_update_kernel(T* __restrict__ P, const T* __restrict__ AP, const T* __restrict__ r);

/*
	More than 2 - 3X improvement over cublas APi due to removal of multi-mode computation
*/
template <int blockDIM, typename T> __global__
void nrm2_kernel_unroll(const T* __restrict__ x, T* __restrict__ sum);

template <typename T> __global__ 
void Dscal_kernel(T* d_target);

template <typename T> __global__ 
void Dcopy_kernel(const T* d_source, T* d_target);

namespace cuBLAS {

	/* Return matrix X vector = d_Target vector on device */
	template <typename T>
	void spMV_M(dim3& Grid, dim3& Block, const T* d_M, const T* d_V, T* d_Target);

	template <typename T>
	void spMV(dim3& Grid, dim3& Block,
		const T* d_a_expand,
		const int* d_ja_expand,
		const T* d_v, T* d_v_expanded, T* d_spMV);

	template <typename T>
	void spMV_thread(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const T* d_a,
		const T* d_v,
		T* d_spMV);

	template <typename T>
	void spMV_warp(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const T* d_a,
		const T* d_v,
		T* d_spMV);

	/* Return the dot product of vector 1 (V_1) and vector 2 (V_2) to a host pointer - product */
	template <typename T>
	void dot_product(dim3& Grid, dim3& Block, const T* __restrict__ d_V_1, const T* __restrict__ d_V_2, T* __restrict__ product, T* __restrict__ d_product);

	/* Return the dot product of vector 1 - vectort 1 (V_1) and vector 1 - vector 2 (V_2) to a host pointer - product */
	template <typename T>
	void dot_product_V2(dim3& Grid, dim3& Block, const T* __restrict__ d_V_1, const T* __restrict__ d_V_2,
		T* __restrict__ sum, T* __restrict__ d_sum, T* __restrict__ dot, T* __restrict__ d_dot);

	/* Return the vector product of vector P_{j+1} = r_{j+1} + beta_j * (P_j - omega_j*AP_j) */
	template <typename T>
	void p_update(dim3& Grid, dim3& Block, T* __restrict__ P, const T* __restrict__ AP, const T* __restrict__ r, const T& omega, const T& beta);

	/* Return the nrm2 product of vector V to a host pointer - sum */
	template <typename T>
	void nrm2(dim3& Grid, dim3& Block, const T* __restrict__ V, T* __restrict__ sum, T* __restrict__ d_sum);

	/* Return the vector product of vector y = y + scale * x */
	template <typename T>
	void axpy(dim3& Grid, dim3& Block, const T* __restrict__ x, T* __restrict__ y, const T& scale);

	/* Return the 2 vector product of vector y1 = y1 + scale * x1 | y2 = y2 - scale * x2 */
	template <typename T>
	void axpy_V2(dim3& Grid, dim3& Block, const T* __restrict__ x1, T* __restrict__ y1, const T* __restrict__ x2, T* __restrict__ y2, const T& scale);

	template <typename T>
	void scale(dim3& Grid, dim3& Block, T* y, const T& scale);

	template <typename T>
	void copy(dim3& Grid, dim3& Block, const T* x, T* y);

	void get_const_int_symbol(const int& h_symbol);

}
