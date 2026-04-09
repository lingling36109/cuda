#ifndef SPMM_CUH
#define SPMM_CUH

#include "common.h"
#include "CSR.hpp"

namespace spmm {

__global__ void SpMM(const size_t m, const size_t n, const size_t k,
                     float * d_A_vals, uint32_t * d_A_colinds, uint32_t * d_A_rowptrs,
                     __constant__ float * d_X, float * d_Y)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    uint32_t row_start = d_A_rowptrs[row];
    uint32_t row_end = d_A_rowptrs[row + 1];

    for (size_t j = 0; j < k; j++) {
        float sum = 0.0f;
        for (uint32_t idx = row_start; idx < row_end; idx++) {
            sum += d_A_vals[idx] * d_X[d_A_colinds[idx] * k + j];
        }
        d_Y[row * k + j] = sum;
    }
}


void SpMM_wrapper(csr_t& A, float * d_X, float * d_Y, const size_t k)
{
    uint32_t threads_per_block = 256;
    uint32_t blocks = (A.get_rows() + threads_per_block - 1) / threads_per_block;

    // Call the kernel
    SpMM<<<blocks, threads_per_block>>>(A.get_rows(), A.get_cols(), k,
                                        A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                                        d_X, d_Y);

    // Sync w/ the host
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
