#ifndef CUDA_COMMON_CUH_
#define CUDA_COMMON_CUH_

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>

const int threads_per_block_2d = 32;
const int threads_per_block = 512;

#define tensor_acc(T, type) (T).packed_accessor64<type, 1, torch::RestrictPtrTraits>()
#define tensor_acc_3(T, N, type) (T).packed_accessor64<type, N, torch::RestrictPtrTraits>()

/* Helpers */

/** Error handling */
inline void _cuda_check_err(const cudaError_t err, const char* file, const int line, const char* function) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Error in %s (%s:%d): %s\n", function, file, line, cudaGetErrorString(err));
        throw std::runtime_error("CUDA Error.");
    }
}
#define cuda_check_err(err) _cuda_check_err(err, __FILE__, __LINE__, __func__)
#ifdef DEBUG_SYNCHRONOUS_KERNEL_LAUNCH
#define cuda_check_kernel_launch_err() cuda_check_err(cudaGetLastError()); cuda_check_err(cudaDeviceSynchronize());
#else
#define cuda_check_kernel_launch_err() cuda_check_err(cudaGetLastError());
#endif

template <typename T>
inline void cuda_print_1d_array(const T* __restrict__ d_ary, const uint64_t length, at::cuda::CUDAStream stream) {
    T* h_ary = new T[length];
    cudaMemcpyAsync(h_ary, d_ary, length * sizeof(T), cudaMemcpyDeviceToHost, stream);
    stream.synchronize();
    for (uint64_t i = 0; i < length; i++) {
        std::cerr << h_ary[i] << " ";
    }
    delete[] h_ary;
}

/**
 * Perform a cumulative sum using CUB.
 */
template <typename T>
void cub_cumsum(T* d_ary_in, T* d_ary_out, int64_t n, at::cuda::CUDAStream stream) {
    /* Run first to determine size of auxiliary data buffer */
    void* d_temp = NULL;
    size_t temp_size = 0;
    cuda_check_err(cub::DeviceScan::InclusiveSum(d_temp, temp_size, d_ary_in, d_ary_out, n, stream));

    /* Run again to actually compute cumsum */
    cuda_check_err(cudaMalloc(&d_temp, temp_size));
    cuda_check_err(cub::DeviceScan::InclusiveSum(d_temp, temp_size, d_ary_in, d_ary_out, n, stream));
    cuda_check_err(cudaFree(d_temp));
}

/**
 * Performs a binary search on an array between i_start and i_end (inclusive).
 */
static __device__ int64_t kernel_indices_binsearch(int64_t i_start, int64_t i_end, const int64_t i_search,
                                                   const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> indices) {
    int64_t i_mid;
    while (i_start <= i_end) {
        i_mid = (i_start + i_end) / 2;
        if (indices[i_mid] < i_search) {
            i_start = i_mid + 1;
        } else if (indices[i_mid] > i_search) {
            i_end = i_mid - 1;
        } else if (indices[i_mid] == i_search) {
            return i_mid;
        }
    }
    return -1;
}

/**
 * Permutes the entries of data such that data_p[i] = data[permutation[i]].
 * Indexed on entries of data.
 */
template <typename scalar_t>
__global__ void cuda_kernel_tensor_permute(
    int length,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> data,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> data_p,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> permutation,
    const bool permute_input) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) {
        return;
    }

    if (permute_input) {
        data_p[i] = data[permutation[i]];
    } else {
        data_p[permutation[i]] = data[i];
    }
}

void lexsort_coo_ijv(torch::Tensor& Bhat_I, torch::Tensor& Bhat_J, torch::Tensor& Bhat_V);
#endif
