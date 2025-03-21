#include "spMV.cuh"

/* Perform preconditioner matrix vector multiplication - a stream kernel with 2X improvement over cusparse APi */
template <typename T> __global__
void spMV_M_kernel(const T* d_M, const T* d_V, T* d_target)

{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < d_Nnode)
	{
		d_target[tid] = d_M[tid] * d_V[tid];
	}
}


/* Perform large global uncoalsed sparse matrix vector multiplication with summed spMV output */
template <int blockDIM, typename T> __global__
void spMV_sparse(const T* __restrict__ a_expand, const int* __restrict__ ja_expand,
	const T* __restrict__ v, T* __restrict__ v_expand, T* __restrict__ spMV)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int lane = tid & 31; // lane ID within a warp

	__shared__ volatile T cache[blockDIM];

	if (tid < d_Nnode * 16)
	{
		v_expand[tid] = v[ja_expand[tid]];
		cache[threadIdx.x] = a_expand[tid] * v_expand[tid];
		// __syncthreads(); // not necessary since the reduction is done within the warp

		// in warp reduction

		if (lane < 8 || (lane > 15 && lane < 24)) cache[threadIdx.x] += cache[threadIdx.x + 8];
		if (lane < 4 || (lane > 15 && lane < 20)) cache[threadIdx.x] += cache[threadIdx.x + 4];
		if (lane < 2 || (lane > 15 && lane < 18)) cache[threadIdx.x] += cache[threadIdx.x + 2];
		if (lane < 1 || (lane > 15 && lane < 17)) cache[threadIdx.x] += cache[threadIdx.x + 1];

		if (lane == 0 || lane == 16) spMV[tid / 16] = cache[threadIdx.x];

	}

}

/* spMV kernel with one thread per row - comparable with cusparseSpMV APi */
template <typename T>
__global__ void spMV_thread_kernel(const int* d_ia, const int* d_ja, const T* d_val, const T* x, T* y)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < d_Nnode) {
		T dot = 0;
		int row_start = d_ia[row];
		int row_end = d_ia[row + 1];
		for (int jj = row_start; jj < row_end; jj++)
			dot += d_val[jj] * x[d_ja[jj]];
		y[row] += dot;
	}
}

/* spMV kernel with one warp (32 thread) per row - poor performance */
template <int blockDIM, typename T> __global__
void spMV_warp_kernel(const int* d_ia, const int* d_ja, const T* d_val, const T* x, T* y)
{
	__shared__ T vals[blockDIM];

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index 
	int warp_id = thread_id / 32; // global warp index 
	int lane = thread_id & (32 - 1); // one warp per row 
	int row = warp_id;
	if (row < d_Nnode)
	{
		int row_start = d_ia[row];
		int row_end = d_ia[row + 1];// compute running sum per thread 
		vals[threadIdx.x] = 0;

		for (int jj = row_start + lane; jj < row_end; jj += 32)
			vals[threadIdx.x] += d_val[jj] * x[d_ja[jj]];// parallel reduction in shared memory 

		if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
		if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8];
		if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4];
		if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2];
		if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1];

		if (lane == 0) y[row] += vals[threadIdx.x]; // first thread writes the result 
	}
}

/* Fully looped dot product kernal with shared memory function - poor performance */
template <typename T> __global__
void dot_product_kernel(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ dot)
{
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ T cache[];

	cache[threadIdx.x] = x[index] * y[index];

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}


	if (threadIdx.x == 0) {
		atomicAdd(dot, cache[0]);
	}
}

/* Fully unrolled templated dot product with shared memory function - best performance so far
	Runtime performance beats cublasDdot api on Nsight compute in release mode
	cublasDdot api performance is invariant to debug/release mode, which means the optimization is already done by Nvidia build-in api function
*/
template <int blockDIM, typename T> __global__
void dot_product_kernel_unroll(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ dot)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x * 8; // block unroll factor of 8

	__shared__ volatile T cache[blockDIM]; // dynamically allocate the shared memory according to block size

	T temp = 0.0;

	if (tid + 7 * blockDim.x <= d_Nnode)
	{
		auto a1 = x[tid] * y[tid];
		auto a2 = x[tid + blockDim.x] * y[tid + blockDim.x];
		auto a3 = x[tid + 2 * blockDim.x] * y[tid + 2 * blockDim.x];
		auto a4 = x[tid + 3 * blockDim.x] * y[tid + 3 * blockDim.x];
		auto a5 = x[tid + 4 * blockDim.x] * y[tid + 4 * blockDim.x];
		auto a6 = x[tid + 5 * blockDim.x] * y[tid + 5 * blockDim.x];
		auto a7 = x[tid + 6 * blockDim.x] * y[tid + 6 * blockDim.x];
		auto a8 = x[tid + 7 * blockDim.x] * y[tid + 7 * blockDim.x];

		temp = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}
	else
	{
		unsigned int Grid_num = d_Nnode / (blockDim.x * 8); // integer number of assigned grid with unroll factor of 8

		tid = threadIdx.x + (Grid_num * blockDim.x * 8 + blockDim.x * (blockIdx.x - Grid_num)); // Remaining block will do the dot product without any unroll factor

		if (tid < d_Nnode) temp = x[tid] * y[tid];
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDIM >= 1024 && threadIdx.x < 512) cache[threadIdx.x] += cache[threadIdx.x + 512];
	__syncthreads();

	if (blockDIM >= 512 && threadIdx.x < 256) cache[threadIdx.x] += cache[threadIdx.x + 256];
	__syncthreads();

	if (blockDIM >= 256 && threadIdx.x < 128) cache[threadIdx.x] += cache[threadIdx.x + 128];
	__syncthreads();

	if (blockDIM >= 128 && threadIdx.x < 64) cache[threadIdx.x] += cache[threadIdx.x + 64];
	__syncthreads();

	// unrolling warp
	if (threadIdx.x < 32)
	{
		cache[threadIdx.x] += cache[threadIdx.x + 32];
		cache[threadIdx.x] += cache[threadIdx.x + 16];
		cache[threadIdx.x] += cache[threadIdx.x + 8];
		cache[threadIdx.x] += cache[threadIdx.x + 4];
		cache[threadIdx.x] += cache[threadIdx.x + 2];
		cache[threadIdx.x] += cache[threadIdx.x + 1];
	}

	if (threadIdx.x == 0) {
		atomicAdd(dot, cache[0]);
	}
}

/*
	Two dot product operations in one kernel launch
*/
template <int blockDIM, typename T> __global__
void dot_product_kernel_V2_unroll(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ sum, T* __restrict__ dot)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x * 8; // block unroll factor of 8

	__shared__ volatile T cache[blockDIM]; // dynamically allocate the shared memory according to block size

	T temp_sum = 0.0;
	T temp_dot = 0.0;

	// x*x dot product

	if (tid + 7 * blockDim.x <= d_Nnode)
	{
		auto a1 = x[tid] * x[tid];
		auto a11 = x[tid] * y[tid];

		auto a2 = x[tid + blockDim.x] * x[tid + blockDim.x];
		auto a22 = x[tid + blockDim.x] * y[tid + blockDim.x];

		auto a3 = x[tid + 2 * blockDim.x] * x[tid + 2 * blockDim.x];
		auto a33 = x[tid + 2 * blockDim.x] * y[tid + 2 * blockDim.x];

		auto a4 = x[tid + 3 * blockDim.x] * x[tid + 3 * blockDim.x];
		auto a44 = x[tid + 3 * blockDim.x] * y[tid + 3 * blockDim.x];

		auto a5 = x[tid + 4 * blockDim.x] * x[tid + 4 * blockDim.x];
		auto a55 = x[tid + 4 * blockDim.x] * y[tid + 4 * blockDim.x];

		auto a6 = x[tid + 5 * blockDim.x] * x[tid + 5 * blockDim.x];
		auto a66 = x[tid + 5 * blockDim.x] * y[tid + 5 * blockDim.x];

		auto a7 = x[tid + 6 * blockDim.x] * x[tid + 6 * blockDim.x];
		auto a77 = x[tid + 6 * blockDim.x] * y[tid + 6 * blockDim.x];

		auto a8 = x[tid + 7 * blockDim.x] * x[tid + 7 * blockDim.x];
		auto a88 = x[tid + 7 * blockDim.x] * y[tid + 7 * blockDim.x];

		temp_sum = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;

		temp_dot = a11 + a22 + a33 + a44 + a55 + a66 + a77 + a88;
	}
	else
	{
		unsigned int Grid_num = d_Nnode / (blockDim.x * 8); // integer number of assigned grid with unroll factor of 8

		tid = threadIdx.x + (Grid_num * blockDim.x * 8 + blockDim.x * (blockIdx.x - Grid_num)); // Remaining block will do the dot product without any unroll factor

		if (tid < d_Nnode) temp_sum = x[tid] * x[tid]; temp_dot = x[tid] * y[tid];
	}

	// x*y dot product

	cache[threadIdx.x] = temp_sum;

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDIM >= 1024 && threadIdx.x < 512) cache[threadIdx.x] += cache[threadIdx.x + 512];
	__syncthreads();

	if (blockDIM >= 512 && threadIdx.x < 256) cache[threadIdx.x] += cache[threadIdx.x + 256];
	__syncthreads();

	if (blockDIM >= 256 && threadIdx.x < 128) cache[threadIdx.x] += cache[threadIdx.x + 128];
	__syncthreads();

	if (blockDIM >= 128 && threadIdx.x < 64) cache[threadIdx.x] += cache[threadIdx.x + 64];
	__syncthreads();

	// unrolling warp
	if (threadIdx.x < 32)
	{
		cache[threadIdx.x] += cache[threadIdx.x + 32];
		cache[threadIdx.x] += cache[threadIdx.x + 16];
		cache[threadIdx.x] += cache[threadIdx.x + 8];
		cache[threadIdx.x] += cache[threadIdx.x + 4];
		cache[threadIdx.x] += cache[threadIdx.x + 2];
		cache[threadIdx.x] += cache[threadIdx.x + 1];
	}

	if (threadIdx.x == 0) {
		atomicAdd(sum, cache[0]);
	}

	cache[threadIdx.x] = temp_dot;

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDIM >= 1024 && threadIdx.x < 512) cache[threadIdx.x] += cache[threadIdx.x + 512];
	__syncthreads();

	if (blockDIM >= 512 && threadIdx.x < 256) cache[threadIdx.x] += cache[threadIdx.x + 256];
	__syncthreads();

	if (blockDIM >= 256 && threadIdx.x < 128) cache[threadIdx.x] += cache[threadIdx.x + 128];
	__syncthreads();

	if (blockDIM >= 128 && threadIdx.x < 64) cache[threadIdx.x] += cache[threadIdx.x + 64];
	__syncthreads();

	// unrolling warp
	if (threadIdx.x < 32)
	{
		cache[threadIdx.x] += cache[threadIdx.x + 32];
		cache[threadIdx.x] += cache[threadIdx.x + 16];
		cache[threadIdx.x] += cache[threadIdx.x + 8];
		cache[threadIdx.x] += cache[threadIdx.x + 4];
		cache[threadIdx.x] += cache[threadIdx.x + 2];
		cache[threadIdx.x] += cache[threadIdx.x + 1];
	}

	if (threadIdx.x == 0) {
		atomicAdd(dot, cache[0]);
	}
}

/*  y = a*x + y
	similar performance of cublasDaxpy, can be used inter-changeably
*/
template <typename T> __global__
void axpy_kernal(const T* __restrict__ x, T* __restrict__ y)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < d_Nnode)
	{
		y[tid] += d_scale * x[tid];
	}
}

/*
	Two axpy operations in one function call:
	1. y1 += scale * x1
	2. y2 -= scale * x2
*/
template <typename T> __global__
void axpy_kernal_V2(const T* __restrict__ x1, T* __restrict__ y1, const T* __restrict__ x2, T* __restrict__ y2)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < d_Nnode)
	{
		y1[tid] += d_scale * x1[tid];
		y2[tid] -= d_scale * x2[tid];
	}
}

/*
	This kernel combines the operation for P_{j+1} = r_{j+1} + beta_j * (P_j - omega_j*AP_j)
*/
template <typename T> __global__
void p_update_kernel(T* __restrict__ P, const T* __restrict__ AP, const T* __restrict__ r)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < d_Nnode)
	{
		P[tid] = r[tid] + d_beta * (P[tid] - d_omega * AP[tid]);
	}

}

/*
	More than 2 - 3X improvement over cublas APi due to removal of multi-mode computation
*/
template <int blockDIM, typename T> __global__
void nrm2_kernel_unroll(const T* __restrict__ x, T* __restrict__ sum)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x * 8; // block unroll factor of 8

	__shared__ volatile T cache[blockDIM]; // dynamically allocate the shared memory according to block size

	T temp = 0.0;

	if (tid + 7 * blockDim.x <= d_Nnode)
	{
		auto a1 = x[tid] * x[tid];
		auto a2 = x[tid + blockDim.x] * x[tid + blockDim.x];
		auto a3 = x[tid + 2 * blockDim.x] * x[tid + 2 * blockDim.x];
		auto a4 = x[tid + 3 * blockDim.x] * x[tid + 3 * blockDim.x];
		auto a5 = x[tid + 4 * blockDim.x] * x[tid + 4 * blockDim.x];
		auto a6 = x[tid + 5 * blockDim.x] * x[tid + 5 * blockDim.x];
		auto a7 = x[tid + 6 * blockDim.x] * x[tid + 6 * blockDim.x];
		auto a8 = x[tid + 7 * blockDim.x] * x[tid + 7 * blockDim.x];

		temp = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}
	else
	{
		unsigned int Grid_num = d_Nnode / (blockDim.x * 8); // integer number of assigned grid with unroll factor of 8

		tid = threadIdx.x + (Grid_num * blockDim.x * 8 + blockDim.x * (blockIdx.x - Grid_num)); // Remaining block will do the dot product without any unroll factor

		if (tid < d_Nnode) temp = x[tid] * x[tid];
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDIM >= 1024 && threadIdx.x < 512) cache[threadIdx.x] += cache[threadIdx.x + 512];
	__syncthreads();

	if (blockDIM >= 512 && threadIdx.x < 256) cache[threadIdx.x] += cache[threadIdx.x + 256];
	__syncthreads();

	if (blockDIM >= 256 && threadIdx.x < 128) cache[threadIdx.x] += cache[threadIdx.x + 128];
	__syncthreads();

	if (blockDIM >= 128 && threadIdx.x < 64) cache[threadIdx.x] += cache[threadIdx.x + 64];
	__syncthreads();

	// unrolling warp
	if (threadIdx.x < 32)
	{

		cache[threadIdx.x] += cache[threadIdx.x + 32];
		cache[threadIdx.x] += cache[threadIdx.x + 16];
		cache[threadIdx.x] += cache[threadIdx.x + 8];
		cache[threadIdx.x] += cache[threadIdx.x + 4];
		cache[threadIdx.x] += cache[threadIdx.x + 2];
		cache[threadIdx.x] += cache[threadIdx.x + 1];

	}

	if (threadIdx.x == 0) {
		atomicAdd(sum, cache[0]);
	}
}

template <typename T> __global__
void Dscal_kernel(T* d_target)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < d_Nnode)
	{
		d_target[tid] *= d_scale;
	}
}

template <typename T> __global__
void Dcopy_kernel(const T* d_source, T* d_target) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < d_Nnode)
	{
		d_target[tid] = d_source[tid];
	}
}

namespace cuBLAS
{
	template <typename T>
	void spMV_M(dim3& Grid, dim3& Block, const T* d_M, const T* d_V, T* d_Target)
	{

		spMV_M_kernel <T> << <Grid, Block >> > (d_M, d_V, d_Target);

	}

	template <typename T>
	void spMV(dim3& Grid, dim3& Block,
		const T* d_a_expand,
		const int* d_ja_expand,
		const T* d_v, T* d_v_expanded, T* d_spMV)
	{
		switch (Block.x)
		{

		case 1024:
			spMV_sparse <1024, T> << <Grid, Block >> > (d_a_expand, d_ja_expand, d_v, d_v_expanded, d_spMV);
			break;
		case 512:
			spMV_sparse <512, T> << <Grid, Block >> > (d_a_expand, d_ja_expand, d_v, d_v_expanded, d_spMV);
			break;
		case 256:
			spMV_sparse <256, T> << < Grid, Block >> > (d_a_expand, d_ja_expand, d_v, d_v_expanded, d_spMV);
			break;
		}
	}

	template <typename T>
	void spMV_thread(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const T* d_a,
		const T* d_v,
		T* d_spMV)
	{
		checkCudaErrors(cudaMemset(d_spMV, 0, Nnode * sizeof(T)));
		spMV_thread_kernel <T> << <Grid, Block >> > (d_ia_expand, d_ja_expand, d_a, d_v, d_spMV);
	}

	template <typename T>
	void spMV_warp(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const T* d_a,
		const T* d_v,
		T* d_spMV)
	{

		checkCudaErrors(cudaMemset(d_spMV, 0, Nnode * sizeof(T)));

		switch (Block.x)
		{

		case 1024:
			spMV_warp_kernel <1024, T> << <Grid, Block >> > (d_ia_expand, d_ja_expand, d_a, d_v, d_spMV);
			break;
		case 512:
			spMV_warp_kernel <512, T> << <Grid, Block >> > (d_ia_expand, d_ja_expand, d_a, d_v, d_spMV);
			break;
		case 256:
			spMV_warp_kernel <256, T> << <Grid, Block >> > (d_ia_expand, d_ja_expand, d_a, d_v, d_spMV);
			break;

		}

	}

	/* Return the dot product of vector 1 (V_1) and vector 2 (V_2) to a host pointer - product */
	template <typename T>
	void dot_product(dim3& Grid, dim3& Block, const T* __restrict__ d_V_1,
		const T* __restrict__ d_V_2, T* __restrict__ product, T* __restrict__ d_product)
	{

		int Grid_unrolled = (Grid.x - 8 * (Grid.x / 8)) + Grid.x / 8;

		checkCudaErrors(cudaMemset(d_product, 0, sizeof(T))); // Initialize to 0

		switch (Block.x)
		{

		case 1024:
			dot_product_kernel_unroll <1024, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_product);
			break;

		case 512:
			dot_product_kernel_unroll <512, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_product);
			break;

		case 256:
			dot_product_kernel_unroll <256, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_product);
			break;

		case 128:
			dot_product_kernel_unroll <128, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_product);
			break;

		}

		checkCudaErrors(cudaMemcpy(product, d_product, sizeof(T), cudaMemcpyDeviceToHost));

	}

	/* Return the dot product of vector 1 - vectort 1 (V_1) and vector 1 - vector 2 (V_2) to a host pointer - product */
	template <typename T>
	void dot_product_V2(dim3& Grid, dim3& Block, const T* __restrict__ d_V_1, const T* __restrict__ d_V_2,
		T* __restrict__ sum, T* __restrict__ d_sum, T* __restrict__ dot, T* __restrict__ d_dot)
	{

		int Grid_unrolled = (Grid.x - 8 * (Grid.x / 8)) + Grid.x / 8;

		checkCudaErrors(cudaMemset(d_sum, 0, sizeof(T))); // Initialize to 0
		checkCudaErrors(cudaMemset(d_dot, 0, sizeof(T))); // Initialize to 0

		switch (Block.x)
		{

		case 1024:
			dot_product_kernel_V2_unroll <1024, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_sum, d_dot);
			break;

		case 512:
			dot_product_kernel_V2_unroll <512, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_sum, d_dot);
			break;

		case 256:
			dot_product_kernel_V2_unroll <256, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_sum, d_dot);
			break;

		case 128:
			dot_product_kernel_V2_unroll <128, T> << <Grid_unrolled, Block >> > (d_V_1, d_V_2, d_sum, d_dot);
			break;

		}

		checkCudaErrors(cudaMemcpy(sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(dot, d_dot, sizeof(T), cudaMemcpyDeviceToHost));

	}

	/* Return the vector product of vector P_{j+1} = r_{j+1} + beta_j * (P_j - omega_j*AP_j) */
	template <typename T>
	void p_update(dim3& Grid, dim3& Block, T* __restrict__ P, const T* __restrict__ AP, const T* __restrict__ r, const T& omega, const T& beta)
	{

		if (sizeof(T) != sizeof(double))
		{
			auto domega = static_cast<double>(omega);
			auto dbeta = static_cast<double>(beta);
			checkCudaErrors(cudaMemcpyToSymbol(d_omega, &domega, sizeof(double))); // get the constant omega factor
			checkCudaErrors(cudaMemcpyToSymbol(d_beta, &dbeta, sizeof(double))); // get the constant beta factor
		}
		else
		{
			checkCudaErrors(cudaMemcpyToSymbol(d_omega, &omega, sizeof(double))); // get the constant omega factor
			checkCudaErrors(cudaMemcpyToSymbol(d_beta, &beta, sizeof(double))); // get the constant beta factor
		}

		p_update_kernel <T> << <Grid, Block >> > (P, AP, r);

	}

	/* Return the nrm2 product of vector V to a host pointer - sum */
	template <typename T>
	void nrm2(dim3& Grid, dim3& Block, const T* __restrict__ V, T* __restrict__ sum, T* __restrict__ d_sum)
	{

		int Grid_unrolled = (Grid.x - 8 * (Grid.x / 8)) + Grid.x / 8;

		checkCudaErrors(cudaMemset(d_sum, 0, sizeof(T))); // Initialize to 0

		switch (Block.x)
		{

		case 1024:
			nrm2_kernel_unroll <1024, T> << <Grid_unrolled, Block >> > (V, d_sum);
			break;

		case 512:
			nrm2_kernel_unroll <512, T> << <Grid_unrolled, Block >> > (V, d_sum);
			break;

		case 256:
			nrm2_kernel_unroll <256, T> << <Grid_unrolled, Block >> > (V, d_sum);
			break;

		case 128:
			nrm2_kernel_unroll <128, T> << <Grid_unrolled, Block >> > (V, d_sum);
			break;

		}

		checkCudaErrors(cudaMemcpy(sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost));

		sum[0] = sqrt(sum[0]); // doing the square root on the host side

	}

	/* Return the vector product of vector y = y + scale * x */
	template <typename T>
	void axpy(dim3& Grid, dim3& Block, const T* __restrict__ x, T* __restrict__ y, const T& scale)
	{
		if (sizeof(T) != sizeof(double))
		{
			auto dscale = static_cast<double>(scale);
			checkCudaErrors(cudaMemcpyToSymbol(d_scale, &dscale, sizeof(double))); // get the constant scale factor
		}
		else
		{
			checkCudaErrors(cudaMemcpyToSymbol(d_scale, &scale, sizeof(double))); // get the constant scale factor
		}

		axpy_kernal << <Grid, Block >> > (x, y);

	}

	/* Return the 2 vector product of vector y1 = y1 + scale * x1 | y2 = y2 - scale * x2 */
	template <typename T>
	void axpy_V2(dim3& Grid, dim3& Block, const T* __restrict__ x1, T* __restrict__ y1, const T* __restrict__ x2, T* __restrict__ y2, const T& scale)
	{

		if (sizeof(T) != sizeof(double))
		{
			auto dscale = static_cast<double>(scale);
			checkCudaErrors(cudaMemcpyToSymbol(d_scale, &dscale, sizeof(double))); // get the constant scale factor
		}
		else
		{
			checkCudaErrors(cudaMemcpyToSymbol(d_scale, &scale, sizeof(double))); // get the constant scale factor
		}

		axpy_kernal_V2 << <Grid, Block >> > (x1, y1, x2, y2);

	}

	/* Return the scaled vector y = scale * y */
	template <typename T>
	void scale(dim3& Grid, dim3& Block, T* y, const T& scale)
	{
		if (sizeof(T) != sizeof(double))
		{
			auto dscale = static_cast<double>(scale);
			checkCudaErrors(cudaMemcpyToSymbol(d_scale, &dscale, sizeof(double))); // get the constant scale factor
		}
		else
		{
			checkCudaErrors(cudaMemcpyToSymbol(d_scale, &scale, sizeof(double))); // get the constant scale factor
		}

		Dscal_kernel << <Grid, Block >> > (y);
	}

	/* Return the copied vector y = x */
	template <typename T>
	void copy(dim3& Grid, dim3& Block, const T* x, T* y)
	{

		Dcopy_kernel << <Grid, Block >> > (x, y);
	}

	void get_const_int_symbol(const int& h_symbol)
	{
		/*
		int* ptr;
		checkCudaErrors(cudaGetSymbolAddress((void**)&ptr, d_Nnode)); // On device side, it doesn't really care what is the type that the pointer is pointing to
																	  // It only needs a void pointer to allocate the memory space.
		*/
		checkCudaErrors(cudaMemcpyToSymbol(d_Nnode, &h_symbol, sizeof(int))); // For device variable, you can also assign a void* to any type variable (e.g int/float/float)
		// For host variable, however, you can only assign pointer to array or pointer type variable.
		// & operator is needed for host variable to be assigned by pointer
	}

	/* Forced template instantiation - Double & Float */

	template void spMV_M(dim3& Grid, dim3& Block, const double* d_M, const double* d_V, double* d_Target);
	template void spMV_M(dim3& Grid, dim3& Block, const float* d_M, const float* d_V, float* d_Target);

	template void spMV(dim3& Grid, dim3& Block, const double* d_a_expand, const int* d_ja_expand,
		const double* d_v, double* d_v_expanded, double* d_spMV);
	template void spMV(dim3& Grid, dim3& Block, const float* d_a_expand, const int* d_ja_expand,
		const float* d_v, float* d_v_expanded, float* d_spMV);

	template void spMV_thread(dim3& Grid, dim3& Block, const int* d_ia_expand, const int* d_ja_expand,
		const double* d_a, const double* d_v, double* d_spMV);
	template void spMV_thread(dim3& Grid, dim3& Block, const int* d_ia_expand, const int* d_ja_expand,
		const float* d_a, const float* d_v, float* d_spMV);

	template void spMV_warp(dim3& Grid, dim3& Block, const int* d_ia_expand, const int* d_ja_expand,
		const double* d_a, const double* d_v, double* d_spMV);
	template void spMV_warp(dim3& Grid, dim3& Block, const int* d_ia_expand, const int* d_ja_expand,
		const float* d_a, const float* d_v, float* d_spMV);

	template void dot_product(dim3& Grid, dim3& Block, const double* __restrict__ d_V_1,
		const double* __restrict__ d_V_2, double* __restrict__ product, double* __restrict__ d_product);
	template void dot_product(dim3& Grid, dim3& Block, const float* __restrict__ d_V_1,
		const float* __restrict__ d_V_2, float* __restrict__ product, float* __restrict__ d_product);

	template void dot_product_V2(dim3& Grid, dim3& Block, const double* __restrict__ d_V_1, const double* __restrict__ d_V_2,
		double* __restrict__ sum, double* __restrict__ d_sum, double* __restrict__ dot, double* __restrict__ d_dot);
	template void dot_product_V2(dim3& Grid, dim3& Block, const float* __restrict__ d_V_1, const float* __restrict__ d_V_2,
		float* __restrict__ sum, float* __restrict__ d_sum, float* __restrict__ dot, float* __restrict__ d_dot);

	template void p_update(dim3& Grid, dim3& Block, double* __restrict__ P, const double* __restrict__ AP,
		const double* __restrict__ r, const double& omega, const double& beta);
	template void p_update(dim3& Grid, dim3& Block, float* __restrict__ P, const float* __restrict__ AP,
		const float* __restrict__ r, const float& omega, const float& beta);

	template void nrm2(dim3& Grid, dim3& Block, const double* __restrict__ V, double* __restrict__ sum, double* __restrict__ d_sum);
	template void nrm2(dim3& Grid, dim3& Block, const float* __restrict__ V, float* __restrict__ sum, float* __restrict__ d_sum);

	template void axpy(dim3& Grid, dim3& Block, const double* __restrict__ x, double* __restrict__ y, const double& scale);
	template void axpy(dim3& Grid, dim3& Block, const float* __restrict__ x, float* __restrict__ y, const float& scale);

	template void axpy_V2(dim3& Grid, dim3& Block, const double* __restrict__ x1, double* __restrict__ y1,
		const double* __restrict__ x2, double* __restrict__ y2, const double& scale);
	template void axpy_V2(dim3& Grid, dim3& Block, const float* __restrict__ x1, float* __restrict__ y1,
		const float* __restrict__ x2, float* __restrict__ y2, const float& scale);

	template void scale(dim3& Grid, dim3& Block, double* y, const double& scale);
	template void scale(dim3& Grid, dim3& Block, float* y, const float& scale);

	template void copy(dim3& Grid, dim3& Block, const double* x, double* y);
	template void copy(dim3& Grid, dim3& Block, const float* x, float* y);
}
