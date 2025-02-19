#pragma once
#include "global.h"

static __constant__ int d_Nnode; // length of the vector

static __constant__ double d_scale; // scale of vector-vector addition

static __constant__ double d_omega; // scale of vector-vector addition

static __constant__ double d_beta; // scale of vector-vector addition

__global__ void spMV_M_kernel(const double* d_M, const double* d_V, double* d_target);

template <int blockDIM> __global__
void spMV_sparse(const double* __restrict__ a_expand, const int* __restrict__ ja_expand,
	const double* __restrict__ v, double* __restrict__ v_expand, double* __restrict__ spMV);

__global__ void spMV_thread_kernel(const int* d_ia, const int* d_ja, const double* d_val, const double* x, double* y);

template <int blockDIM> __global__
void spMV_warp_kernel(const int* d_ia, const int* d_ja, const double* d_val, const double* x, double* y);

__global__ void dot_product_kernel(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ dot);

template <int blockDIM> __global__
void dot_product_kernel_unroll(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ dot);

template <int blockDIM> __global__
void dot_product_kernel_V2_unroll(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ sum, double* __restrict__ dot);

__global__ void dot_product_kernel_V2_Register_unroll(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ sum, double* __restrict__ dot);

__global__ void axpy_kernal(const double* __restrict__ x, double* __restrict__ y);

__global__ void axpy_kernal_V2(const double* __restrict__ x1, double* __restrict__ y1, const double* __restrict__ x2, double* __restrict__ y2);

__global__ void p_update_kernel(double* __restrict__ P, const double* __restrict__ AP, const double* __restrict__ r);

template <int blockDIM> __global__
void nrm2_kernel_unroll(const double* __restrict__ x, double* __restrict__ sum);


namespace cuBLAS {

	void spMV_M(dim3& Grid, dim3& Block, const double* d_M, const double* d_V, double* d_Target);
	void spMV(dim3& Grid, dim3& Block,
		const double* d_a_expand,
		const int* d_ja_expand,
		const double* d_v, double* d_v_expanded, double* d_spMV);
	void spMV_thread(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const double* d_a,
		const double* d_v,
		double* d_spMV);
	void spMV_warp(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const double* d_a,
		const double* d_v,
		double* d_spMV);
	void dot_product(dim3& Grid, dim3& Block, const double* __restrict__ d_V_1, const double* __restrict__ d_V_2, double* __restrict__ product, double* __restrict__ d_product);
	void dot_product_V2(dim3& Grid, dim3& Block, const double* __restrict__ d_V_1, const double* __restrict__ d_V_2,
		double* __restrict__ sum, double* __restrict__ d_sum, double* __restrict__ dot, double* __restrict__ d_dot);
	void p_update(dim3& Grid, dim3& Block, double* __restrict__ P, const double* __restrict__ AP, const double* __restrict__ r, const double& omege, const double& beta);
	void nrm2(dim3& Grid, dim3& Block, const double* __restrict__ V, double* __restrict__ sum, double* __restrict__ d_sum);
	void axpy(dim3& Grid, dim3& Block, const double* __restrict__ x, double* __restrict__ y, const double& scale);
	void axpy_V2(dim3& Grid, dim3& Block, const double* __restrict__ x1, double* __restrict__ y1, const double* __restrict__ x2, double* __restrict__ y2, const double& scale);
	void get_const_int_symbol(const int& h_symbol);

}
