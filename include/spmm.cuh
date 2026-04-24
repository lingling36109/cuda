#ifndef SPMM_CUH
#define SPMM_CUH

#include "common.h"
#include "CSR.hpp"

namespace spmm {

// One warp (32 lanes) cooperatively computes one row of Y.
// Lanes are partitioned across the k dimension: lane l owns columns
// [l*VW, l*VW + VW) of the output row. Inside the warp we walk the
// row's nonzeros in chunks of 32, broadcasting one (val, colind) per
// lane via shuffle, and each lane FMAs into VW private accumulators.
template <int VW>
__global__ void SpMM_warp_row(const size_t m,
                              const float * __restrict__ A_vals,
                              const uint32_t * __restrict__ A_colinds,
                              const uint32_t * __restrict__ A_rowptrs,
                              const float * __restrict__ X,
                              float * __restrict__ Y)
{
    constexpr int W = 32;
    constexpr int K = W * VW;

    const int lane = threadIdx.x;
    const int warp_in_block = threadIdx.y;
    const size_t row = (size_t)blockIdx.x * blockDim.y + warp_in_block;
    if (row >= m) return;

    const uint32_t row_start = A_rowptrs[row];
    const uint32_t row_end   = A_rowptrs[row + 1];

    float acc[VW];
    #pragma unroll
    for (int v = 0; v < VW; v++) acc[v] = 0.0f;

    for (uint32_t base = row_start; base < row_end; base += W)
    {
        const uint32_t idx = base + lane;
        const bool in_range = idx < row_end;
        const float    v_lane = in_range ? A_vals[idx]    : 0.0f;
        const uint32_t c_lane = in_range ? A_colinds[idx] : 0u;

        const int active = (row_end - base) < (uint32_t)W
                            ? (int)(row_end - base) : W;

        #pragma unroll
        for (int s = 0; s < W; s++)
        {
            float    val_b = __shfl_sync(0xffffffff, v_lane, s);
            uint32_t col_b = __shfl_sync(0xffffffff, c_lane, s);
            if (s >= active) continue;

            const float * xrow = X + (size_t)col_b * K + lane * VW;

            if constexpr (VW == 2) {
                float2 xv = *reinterpret_cast<const float2*>(xrow);
                acc[0] += val_b * xv.x;
                acc[1] += val_b * xv.y;
            } else if constexpr (VW == 4) {
                float4 xv = *reinterpret_cast<const float4*>(xrow);
                acc[0] += val_b * xv.x;
                acc[1] += val_b * xv.y;
                acc[2] += val_b * xv.z;
                acc[3] += val_b * xv.w;
            } else if constexpr (VW == 8) {
                float4 a = *reinterpret_cast<const float4*>(xrow);
                float4 b = *reinterpret_cast<const float4*>(xrow + 4);
                acc[0] += val_b * a.x; acc[1] += val_b * a.y;
                acc[2] += val_b * a.z; acc[3] += val_b * a.w;
                acc[4] += val_b * b.x; acc[5] += val_b * b.y;
                acc[6] += val_b * b.z; acc[7] += val_b * b.w;
            } else {
                #pragma unroll
                for (int v = 0; v < VW; v++) acc[v] += val_b * xrow[v];
            }
        }
    }

    float * yrow = Y + row * K + lane * VW;
    if constexpr (VW == 2) {
        float2 r{acc[0], acc[1]};
        *reinterpret_cast<float2*>(yrow) = r;
    } else if constexpr (VW == 4) {
        float4 r{acc[0], acc[1], acc[2], acc[3]};
        *reinterpret_cast<float4*>(yrow) = r;
    } else if constexpr (VW == 8) {
        float4 a{acc[0], acc[1], acc[2], acc[3]};
        float4 b{acc[4], acc[5], acc[6], acc[7]};
        *reinterpret_cast<float4*>(yrow)     = a;
        *reinterpret_cast<float4*>(yrow + 4) = b;
    } else {
        #pragma unroll
        for (int v = 0; v < VW; v++) yrow[v] = acc[v];
    }
}


// Generic fallback for k values not handled by the templated kernel.
__global__ void SpMM_thread_row(const size_t m, const size_t k,
                                const float * __restrict__ A_vals,
                                const uint32_t * __restrict__ A_colinds,
                                const uint32_t * __restrict__ A_rowptrs,
                                const float * __restrict__ X,
                                float * __restrict__ Y)
{
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    uint32_t row_start = A_rowptrs[row];
    uint32_t row_end   = A_rowptrs[row + 1];

    for (size_t j = 0; j < k; j++) {
        float sum = 0.0f;
        for (uint32_t idx = row_start; idx < row_end; idx++) {
            sum += A_vals[idx] * X[(size_t)A_colinds[idx] * k + j];
        }
        Y[row * k + j] = sum;
    }
}


void SpMM_wrapper(csr_t& A, float * d_X, float * d_Y, const size_t k)
{
    const size_t m = A.get_rows();
    constexpr int WARPS_PER_BLOCK = 4;

    if (k == 64 || k == 256) {
        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((m + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        if (k == 64) {
            SpMM_warp_row<2><<<grid, block>>>(m,
                A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
        } else {
            SpMM_warp_row<8><<<grid, block>>>(m,
                A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
        }
    } else {
        uint32_t threads_per_block = 256;
        uint32_t blocks = (m + threads_per_block - 1) / threads_per_block;
        SpMM_thread_row<<<blocks, threads_per_block>>>(m, k,
            A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
