#ifndef SPMM_CUH
#define SPMM_CUH

#include <vector>
#include <algorithm>
#include "common.h"
#include "CSR.hpp"

namespace spmm {

// ----------------------------------------------------------------------------
// Kernel A: warp-per-row, indexed via row_ids (medium nnz/row).
// One warp (32 lanes) cooperatively computes one row of Y. Lanes split across
// the k dimension: lane l owns columns [l*VW, l*VW + VW). The warp walks the
// row's nonzeros in chunks of 32, broadcasting (val, colind) per lane via
// shuffle, and each lane FMAs into VW private accumulators.
// ----------------------------------------------------------------------------
template <int VW>
__global__ void SpMM_warp_row(const size_t n_rows,
                              const uint32_t * __restrict__ row_ids,
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
    const size_t local_row = (size_t)blockIdx.x * blockDim.y + warp_in_block;
    if (local_row >= n_rows) return;
    const uint32_t row = row_ids[local_row];

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

    float * yrow = Y + (size_t)row * K + lane * VW;
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


// ----------------------------------------------------------------------------
// Kernel B: sub-warp-per-row (short rows like delaunay_n24).
// W_S lanes cooperate on one row, 32/W_S sub-warps per warp.
// VW = K / W_S floats per lane along k-dim.
// ----------------------------------------------------------------------------
template <int W_S, int VW>
__global__ void SpMM_short(const size_t n_rows,
                           const uint32_t * __restrict__ row_ids,
                           const float * __restrict__ A_vals,
                           const uint32_t * __restrict__ A_colinds,
                           const uint32_t * __restrict__ A_rowptrs,
                           const float * __restrict__ X,
                           float * __restrict__ Y)
{
    constexpr int K = W_S * VW;
    constexpr int SUBS_PER_WARP = 32 / W_S;

    const int tid          = threadIdx.x;
    const int warp_id      = tid >> 5;
    const int lane_in_warp = tid & 31;
    const int sub_in_warp  = lane_in_warp / W_S;
    const int lane         = lane_in_warp % W_S;
    const int sub_in_block  = warp_id * SUBS_PER_WARP + sub_in_warp;

    const size_t local_row = (size_t)blockIdx.x * (blockDim.x / W_S) + sub_in_block;
    if (local_row >= n_rows) return;
    const uint32_t row = row_ids[local_row];

    const uint32_t row_start = A_rowptrs[row];
    const uint32_t row_end   = A_rowptrs[row + 1];

    constexpr unsigned int sub_mask_local = (W_S == 32) ? 0xffffffffu : ((1u << W_S) - 1u);
    const unsigned int mask = sub_mask_local << (sub_in_warp * W_S);

    float acc[VW];
    #pragma unroll
    for (int v = 0; v < VW; v++) acc[v] = 0.0f;

    for (uint32_t base = row_start; base < row_end; base += W_S)
    {
        const uint32_t idx = base + lane;
        const bool in_range = idx < row_end;
        const float    v_lane = in_range ? A_vals[idx]    : 0.0f;
        const uint32_t c_lane = in_range ? A_colinds[idx] : 0u;

        const int active = (row_end - base) < (uint32_t)W_S
                            ? (int)(row_end - base) : W_S;

        #pragma unroll
        for (int s = 0; s < W_S; s++)
        {
            float    val_b = __shfl_sync(mask, v_lane, s, W_S);
            uint32_t col_b = __shfl_sync(mask, c_lane, s, W_S);
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

    float * yrow = Y + (size_t)row * K + lane * VW;
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


// ----------------------------------------------------------------------------
// Kernel C: multi-CTA long-row k-partition for k=256.
// Each row is split across multiple CTAs by k-slice.  Each CTA still walks the
// full CSR nnz stream for that row, but it only computes its own contiguous
// slice of output columns.  This adds launch-level parallelism without atomics.
// ----------------------------------------------------------------------------
template <int VW, int N_WARPS_PER_CTA, int CTAS_PER_ROW>
__global__ void SpMM_long_kpart_multiCTA(const size_t n_rows,
                                         const uint32_t * __restrict__ row_ids,
                                         const float * __restrict__ A_vals,
                                         const uint32_t * __restrict__ A_colinds,
                                         const uint32_t * __restrict__ A_rowptrs,
                                         const float * __restrict__ X,
                                         float * __restrict__ Y)
{
    constexpr int W = 32;
    constexpr int K_PER_CTA = W * VW * N_WARPS_PER_CTA;
    constexpr int K_TOTAL    = K_PER_CTA * CTAS_PER_ROW;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const size_t row_local = blockIdx.x / CTAS_PER_ROW;
    const int cta_in_row   = blockIdx.x % CTAS_PER_ROW;
    if (row_local >= n_rows) return;

    const uint32_t row = row_ids[row_local];

    const uint32_t row_start = A_rowptrs[row];
    const uint32_t row_end   = A_rowptrs[row + 1];

    const int col_off = cta_in_row * K_PER_CTA + warp * W * VW + lane * VW;

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

            const float * xrow = X + (size_t)col_b * K_TOTAL + col_off;

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
            } else {
                #pragma unroll
                for (int v = 0; v < VW; v++) acc[v] += val_b * xrow[v];
            }
        }
    }

    float * yrow = Y + (size_t)row * K_TOTAL + col_off;
    if constexpr (VW == 2) {
        float2 r{acc[0], acc[1]};
        *reinterpret_cast<float2*>(yrow) = r;
    } else if constexpr (VW == 4) {
        float4 r{acc[0], acc[1], acc[2], acc[3]};
        *reinterpret_cast<float4*>(yrow) = r;
    } else {
        #pragma unroll
        for (int v = 0; v < VW; v++) yrow[v] = acc[v];
    }
}


// ----------------------------------------------------------------------------
// Generic fallback for k values not handled by the templated kernels.
// ----------------------------------------------------------------------------
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


// ----------------------------------------------------------------------------
// Bucket cache: original three-bucket split restored.
// ----------------------------------------------------------------------------
struct BucketCache {
    const uint32_t * key_rowptrs = nullptr;
    size_t key_m = 0;
    size_t key_nnz = 0;
    uint32_t * d_short = nullptr;
    uint32_t * d_med   = nullptr;
    uint32_t * d_long  = nullptr;
    size_t n_short = 0;
    size_t n_med   = 0;
    size_t n_long  = 0;
};

inline BucketCache & get_bucket_cache() {
    static BucketCache cache;
    return cache;
}

inline void prepare_buckets(csr_t & A)
{
    BucketCache & bc = get_bucket_cache();
    if (bc.key_rowptrs == A.get_rowptrs()
        && bc.key_m == A.get_rows()
        && bc.key_nnz == A.get_nnz()) {
        return;
    }

    if (bc.d_short) cudaFree(bc.d_short);
    if (bc.d_med)   cudaFree(bc.d_med);
    if (bc.d_long)  cudaFree(bc.d_long);
    bc = BucketCache{};

    const size_t m = A.get_rows();
    std::vector<uint32_t> h_rowptrs(m + 1);
    CUDA_CHECK(cudaMemcpy(h_rowptrs.data(), A.get_rowptrs(),
                          sizeof(uint32_t) * (m + 1), cudaMemcpyDeviceToHost));

    constexpr uint32_t SHORT_T = 16;
    constexpr uint32_t LONG_T  = 64;

    std::vector<uint32_t> hs, hm, hl;
    hs.reserve(m); hm.reserve(m); hl.reserve(m);

    std::vector<uint32_t> row_nnz(m);
    for (uint32_t i = 0; i < (uint32_t)m; i++) {
        row_nnz[i] = h_rowptrs[i + 1] - h_rowptrs[i];
    }

    for (uint32_t i = 0; i < (uint32_t)m; i++) {
        uint32_t nnz_i = row_nnz[i];
        if (nnz_i <= SHORT_T)      hs.push_back(i);
        else if (nnz_i <= LONG_T)  hm.push_back(i);
        else                       hl.push_back(i);
    }

    auto sort_by_desc_nnz = [&](std::vector<uint32_t>& rows) {
        std::sort(rows.begin(), rows.end(),
                  [&](uint32_t a, uint32_t b) {
                      if (row_nnz[a] != row_nnz[b]) return row_nnz[a] > row_nnz[b];
                      return a < b;
                  });
    };

    sort_by_desc_nnz(hs);
    sort_by_desc_nnz(hm);
    sort_by_desc_nnz(hl);

    auto upload = [](const std::vector<uint32_t>& h, uint32_t** dp, size_t& n) {
        n = h.size();
        if (n) {
            CUDA_CHECK(cudaMalloc(dp, sizeof(uint32_t) * n));
            CUDA_CHECK(cudaMemcpy(*dp, h.data(), sizeof(uint32_t) * n,
                                  cudaMemcpyHostToDevice));
        }
    };
    upload(hs, &bc.d_short, bc.n_short);
    upload(hm, &bc.d_med,   bc.n_med);
    upload(hl, &bc.d_long,  bc.n_long);

    bc.key_rowptrs = A.get_rowptrs();
    bc.key_m       = m;
    bc.key_nnz     = A.get_nnz();
}


// ----------------------------------------------------------------------------
// Wrapper: restore bucket-specific dispatch and use multi-CTA long-row kernel.
// ----------------------------------------------------------------------------
void SpMM_wrapper(csr_t & A, float * d_X, float * d_Y, const size_t k)
{
    if (k != 64 && k != 256) {
        uint32_t threads_per_block = 256;
        uint32_t blocks = (A.get_rows() + threads_per_block - 1) / threads_per_block;
        SpMM_thread_row<<<blocks, threads_per_block>>>(A.get_rows(), k,
            A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    prepare_buckets(A);
    BucketCache & bc = get_bucket_cache();

    float    * vals = A.get_vals();
    uint32_t * cols = A.get_colinds();
    uint32_t * rowp = A.get_rowptrs();

    if (k == 64) {
        if (bc.n_short > 0) {
            constexpr int W_S = 8;
            constexpr int VW  = 8;
            constexpr int ROWS_PER_BLOCK = 256 / W_S;  // 32
            dim3 block(256);
            dim3 grid((bc.n_short + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
            SpMM_short<W_S, VW><<<grid, block>>>(
                bc.n_short, bc.d_short, vals, cols, rowp, d_X, d_Y);
        }

        if (bc.n_med > 0) {
            constexpr int VW = 2;
            constexpr int WARPS = 4;
            dim3 block(32, WARPS);
            dim3 grid((bc.n_med + WARPS - 1) / WARPS);
            SpMM_warp_row<VW><<<grid, block>>>(
                bc.n_med, bc.d_med, vals, cols, rowp, d_X, d_Y);
        }

        if (bc.n_long > 0) {
            constexpr int VW = 2;
            constexpr int WARPS = 4;
            dim3 block(32, WARPS);
            dim3 grid((bc.n_long + WARPS - 1) / WARPS);
            SpMM_warp_row<VW><<<grid, block>>>(
                bc.n_long, bc.d_long, vals, cols, rowp, d_X, d_Y);
        }
    } else {  // k == 256
        if (bc.n_short > 0) {
            constexpr int VW = 8;
            constexpr int WARPS = 4;
            dim3 block(32, WARPS);
            dim3 grid((bc.n_short + WARPS - 1) / WARPS);
            SpMM_warp_row<VW><<<grid, block>>>(
                bc.n_short, bc.d_short, vals, cols, rowp, d_X, d_Y);
        }

        if (bc.n_med > 0) {
            constexpr int VW = 8;
            constexpr int WARPS = 4;
            dim3 block(32, WARPS);
            dim3 grid((bc.n_med + WARPS - 1) / WARPS);
            SpMM_warp_row<VW><<<grid, block>>>(
                bc.n_med, bc.d_med, vals, cols, rowp, d_X, d_Y);
        }

        if (bc.n_long > 0) {
            constexpr int VW_L = 2;
            constexpr int N_WARPS_PER_CTA = 2;
            constexpr int CTAS_PER_ROW = 2;
            dim3 block(32, N_WARPS_PER_CTA);
            dim3 grid(bc.n_long * CTAS_PER_ROW);
            SpMM_long_kpart_multiCTA<VW_L, N_WARPS_PER_CTA, CTAS_PER_ROW><<<grid, block>>>(
                bc.n_long, bc.d_long, vals, cols, rowp, d_X, d_Y);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace spmm
#endif