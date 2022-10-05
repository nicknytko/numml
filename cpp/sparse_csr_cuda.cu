#include <torch/extension.h>
#include <map>
#include <tuple>
#include <cstdint>
#include <stdexcept>
#include <iterator>
#include <tuple>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <cuco/static_map.cuh>
#include <cuco/detail/hash_functions.cuh>
#include <cuco/allocator.hpp>

#include "sparse_csr.hpp"

#define tensor_acc(T, type) (T).packed_accessor64<type, 1, torch::RestrictPtrTraits>()

/* Sparse GEMV */

/**
 * Computes the sparse matvec Ax.
 * Indexed on entries of the output.
 */
template <typename scalar_t>
__global__ void spgemv_forward_cuda_kernel_matvec(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_col_ind,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_rowptr,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> z_out) {

    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    float z_i = 0.f;
    for (int64_t i = A_rowptr[row]; i < A_rowptr[row + 1]; i++) {
        int col = A_col_ind[i];
        z_i += A_data[i] * x[col];
    }
    z_out[row] = z_i;
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spgemv_forward,
               int A_rows, int A_cols, torch::Tensor alpha,
               torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
               torch::Tensor x, torch::Tensor beta, torch::Tensor y) {

    auto options = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor Ax = torch::empty({A_rows}, options);
    const int threads_per_block = 512;
    const int threads = A_rows;
    const dim3 blocks((threads + threads_per_block - 1) / threads_per_block, 1);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_forward_cuda", ([&] {
        spgemv_forward_cuda_kernel_matvec<scalar_t><<<blocks, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols,
            tensor_acc(A_data, scalar_t), tensor_acc(A_col_ind, int64_t), tensor_acc(A_rowptr, int64_t),
            tensor_acc(x, scalar_t), tensor_acc(Ax, scalar_t));
    }));

    return {Ax, alpha * Ax + beta * y};
}

/**
 * Computes the gradient of Ax wrt A in the spgemv product.
 * Indexed on nonzeros of grad_A.
 */
template <typename scalar_t>
__global__ void spgemv_backward_cuda_kernel_grad_A(
    int A_rows, int A_cols, scalar_t alpha,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_z,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_A,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_col_ind,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_rowptr,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x) {

    /* grad_A = alpha * outer(grad_z, x) (*) mask(A) */

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    for (int64_t i = A_rowptr[row]; i < A_rowptr[row + 1]; i++) {
        int64_t col = A_col_ind[i];
        grad_A[i] = alpha * grad_z[row] * x[col];
    }
}


__device__ int64_t kernel_indices_binsearch(int64_t i_start, int64_t i_end, int64_t i_search,
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
 * Computes the gradient of Ax wrt x in the spgemv product.
 * Indexed on entries of grad_x.
 */
template <typename scalar_t>
__global__ void spgemv_backward_cuda_kernel_grad_x(
    int A_rows, int A_cols, scalar_t alpha,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_z,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_col_ind,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_rowptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_x) {

    /* Compute grad_x = alpha * A^T grad_z */

    int64_t x_row = blockIdx.x * blockDim.x + threadIdx.x; /* indexing into the output grad_x */
    if (x_row >= grad_x.size(0)) {
        return;
    }

    scalar_t xi = 0.;

    for (int64_t row = 0; row < A_rows; row++) {
        int64_t col_idx = kernel_indices_binsearch(A_rowptr[row], A_rowptr[row + 1], x_row, A_col_ind); /* TODO: optimize this */
        if (col_idx != -1) {
            xi += alpha * A_data[col_idx] * grad_z[row];
        }
    }
    grad_x[x_row] = xi;
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spgemv_backward,
               torch::Tensor grad_z, int A_rows, int A_cols, torch::Tensor alpha,
               torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
               torch::Tensor x, torch::Tensor beta, torch::Tensor y) {

    const int threads_per_block = 512;
    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Gradient wrt A */
    torch::Tensor grad_A = torch::empty_like(A_data);
    const int grad_A_threads = A_data.sizes()[0];
    const dim3 grad_A_blocks((grad_A_threads + threads_per_block - 1) / threads_per_block, 1);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_backward_cuda", ([&] {
        spgemv_backward_cuda_kernel_grad_A<scalar_t><<<grad_A_blocks, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols, alpha.item<scalar_t>(),
            tensor_acc(grad_z, scalar_t), tensor_acc(grad_A, scalar_t),
            tensor_acc(A_col_ind, int64_t), tensor_acc(A_rowptr, int64_t),
            tensor_acc(x, scalar_t));
    }));

    /* Gradient wrt x */
    torch::Tensor grad_x = torch::empty_like(x);
    const int grad_x_threads = x.sizes()[0];
    const dim3 grad_x_blocks((grad_x_threads + threads_per_block - 1) / threads_per_block, 1);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_backward_cuda", ([&] {
        spgemv_backward_cuda_kernel_grad_x<scalar_t><<<grad_x_blocks, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols, alpha.item<scalar_t>(),
            tensor_acc(grad_z, scalar_t),
            tensor_acc(A_data, scalar_t), tensor_acc(A_col_ind, int64_t), tensor_acc(A_rowptr, int64_t),
            tensor_acc(grad_x, scalar_t));
    }));

    return {grad_A, grad_x};
}

/**
 * Find the number of nonzeros for two sparse CSR matrices.
 * Indexed on rows of A.
 */
__global__ void cuda_kernel_find_nnz(
    int A_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_nnz) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= A_rows) {
        return;
    }
    A_nnz[i] = A_indptr[i+1] - A_indptr[i];
}

/**
 * Computes the maximal number of nonzero entries per row in the CSR SPGEMM product.
 * Indexed on rows of C.
 */
__global__ void cuda_kernel_find_Chat_nnz(
    int C_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_nnz,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_nnz,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_nnz) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    int64_t nnz = 0;
    int64_t k;
    for (int64_t k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        nnz += B_nnz[k];
    }
    Chat_nnz[i] = nnz;
}


/**
 * Computes the matrix expansion in the CSR SPGEMM product.
 * Indexed on rows of C.
 */
template <typename scalar_t>
__global__ void cuda_kernel_Chat_expansion(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_I,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_J) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Grab starting index for entries of Chat */
    int64_t C_row_idx = 0;
    if (i > 0) {
        C_row_idx = Chat_indptr[i - 1];
    }

    /* Multiply row of A and row of B */
    int64_t k_i, k, j_i, j;
    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        for (j_i = B_indptr[k]; j_i < B_indptr[k + 1]; j_i++) {
            j = B_indices[j_i];

            Chat_I[C_row_idx] = i;
            Chat_J[C_row_idx] = j;
            Chat_data[C_row_idx] = A_data[k_i] * B_data[j_i];

            C_row_idx ++;
        }
    }
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

/**
 * Compute the number of nonzero entries per row of (sorted) Chat by counting unique column indices.
 * Indexed on rows of C.
 */
__global__ void cuda_kernel_Chat_to_C_row_nnz(
    int C_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indices,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_nnz) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    const int64_t start_idx = (i > 0 ? Chat_indptr[i-1] : 0);
    const int64_t end_idx = Chat_indptr[i];

    int64_t cur_col = -1;
    int64_t nnz = 0;
    for (int64_t j = start_idx; j < end_idx; j++) {
        int64_t col = Chat_indices[j];
        if (col != cur_col) {
            nnz++;
        }
        cur_col = col;
    }
    C_nnz[i] = nnz;
}

/**
 * Assemble data and indices of C from Chat given indptr
 * Indexed on rows of C.
 */
template <typename scalar_t>
__global__ void cuda_kernel_assemble_C(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indices,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    const int64_t start_idx = (i > 0 ? Chat_indptr[i-1] : 0);
    const int64_t end_idx = Chat_indptr[i];
    int64_t cur_col = Chat_indices[start_idx];
    int64_t cur_C_ptr = C_indptr[i];
    scalar_t acc = 0.;

    for (int64_t j = start_idx; j <= end_idx; j++) {
        if (j == end_idx || Chat_indices[j] != cur_col) {
            C_data[cur_C_ptr] = acc;
            C_indices[cur_C_ptr] = cur_col;

            cur_C_ptr++;
            acc = 0.;

            if (j == end_idx) {
                break;
            } else {
                cur_col = Chat_indices[j];
            }
        }
        acc += Chat_data[j];
    }
}

/**
 * Assemble data and indices of C from Chat given indptr, does not affect the column indices.
 * Indexed on rows of A.
 */
template <typename scalar_t>
__global__ void cuda_kernel_assemble_C_data_only(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indices,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    const int64_t start_idx = (i > 0 ? Chat_indptr[i-1] : 0);
    const int64_t end_idx = Chat_indptr[i];
    int64_t cur_col = Chat_indices[start_idx];
    int64_t cur_C_ptr = C_indptr[i];
    scalar_t acc = 0.;

    for (int64_t j = start_idx; j <= end_idx; j++) {
        if (j == end_idx || Chat_indices[j] != cur_col) {
            C_data[cur_C_ptr] = acc;

            cur_C_ptr++;
            acc = 0.;

            if (j == end_idx) {
                break;
            } else {
                cur_col = Chat_indices[j];
            }
        }
        acc += Chat_data[j];
    }
}

/**
 * Given intermediate row, column, value COO representation,
 * sort entries first by row then by column.
 */
void lexsort_coo_ijv(torch::Tensor& Bhat_I,
                     torch::Tensor& Bhat_J,
                     torch::Tensor& Bhat_V) {

    const int threads_per_block = 512;

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Sort first by columns... */
    torch::Tensor argsort = Bhat_J.argsort(false);
    torch::Tensor i_temp = torch::empty_like(Bhat_I);
    torch::Tensor j_temp = torch::empty_like(Bhat_J);
    torch::Tensor v_temp = torch::empty_like(Bhat_V);

    const int64_t Bhat_total_nnz = Bhat_I.size(0);

    /* ...permute entries into their correct positions */
    AT_DISPATCH_FLOATING_TYPES(Bhat_V.type(), "lexsort_coo_ijv", [&] {
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(Bhat_I, int64_t), tensor_acc(i_temp, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(Bhat_J, int64_t), tensor_acc(j_temp, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(Bhat_V, scalar_t), tensor_acc(v_temp, scalar_t), tensor_acc(argsort, int64_t), true);
    });

    /* Now, stable sort on rows... */
    argsort = i_temp.argsort(true);

    /* ...and again permute entries into correct spots */
    AT_DISPATCH_FLOATING_TYPES(Bhat_V.type(), "lexsort_coo_ijv", [&] {
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(i_temp, int64_t), tensor_acc(Bhat_I, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(j_temp, int64_t), tensor_acc(Bhat_J, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(v_temp, scalar_t), tensor_acc(Bhat_V, scalar_t), tensor_acc(argsort, int64_t), true);
    });
}

/* Sparse GEMM */

/**
 * CUDA implementation of sparse csr matrix matrix product (forward pass)
 * based on the algorithm sketched here:
 * http://lukeo.cs.illinois.edu/files/2015_BeDaOl_SPMM.pdf
 *
 * I don't claim that this is a super optimized version, but it's much faster than CPU :-)
 */

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spgemm_forward,
               int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    const int C_rows = A_rows;
    const int C_cols = B_cols;
    const int threads_per_block = 512;

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Find NNZ in each matrix row */
    torch::Tensor A_nnz = torch::empty({A_rows}, int_tens_opts);
    torch::Tensor B_nnz = torch::empty({B_rows}, int_tens_opts);
    cuda_kernel_find_nnz<<<A_rows + threads_per_block - 1 / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows, tensor_acc(A_indptr, int64_t), tensor_acc(A_nnz, int64_t));
    cuda_kernel_find_nnz<<<B_rows + threads_per_block - 1 / threads_per_block, threads_per_block, 0, main_stream>>>(
        B_rows, tensor_acc(B_indptr, int64_t), tensor_acc(B_nnz, int64_t));

    /* Find NNZ in each row of \hat{C} */
    torch::Tensor Chat_nnz = torch::empty({C_rows}, int_tens_opts);
    cuda_kernel_find_Chat_nnz<<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        C_rows, tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
        tensor_acc(A_nnz, int64_t), tensor_acc(B_nnz, int64_t), tensor_acc(Chat_nnz, int64_t));

    torch::Tensor Chat_nnz_cumsum = Chat_nnz.cumsum(0);
    int64_t Chat_total_nnz = Chat_nnz_cumsum[Chat_nnz_cumsum.size(0) - 1].item<int64_t>();

    /* Compute the entries of Chat via expansion */
    torch::Tensor Chat_I = torch::empty({Chat_total_nnz}, int_tens_opts);
    torch::Tensor Chat_J = torch::empty({Chat_total_nnz}, int_tens_opts);
    torch::Tensor Chat_V = torch::empty({Chat_total_nnz}, scalar_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_forward_cuda", ([&] {
        cuda_kernel_Chat_expansion<scalar_t><<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            C_rows,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
            tensor_acc(B_data, scalar_t), tensor_acc(B_indptr, int64_t), tensor_acc(B_indices, int64_t),
            tensor_acc(Chat_nnz_cumsum, int64_t), tensor_acc(Chat_V, scalar_t), tensor_acc(Chat_I, int64_t), tensor_acc(Chat_J, int64_t));
    }));

    /* Lexicographically sort entries of Chat first by column index then by row index */
    lexsort_coo_ijv(Chat_I, Chat_J, Chat_V);

    /* Compute nonzeros in C by counting unique column indices in Chat */
    torch::Tensor C_nnz = torch::empty_like(Chat_nnz);
    cuda_kernel_Chat_to_C_row_nnz<<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        C_rows, tensor_acc(Chat_nnz_cumsum, int64_t), tensor_acc(Chat_J, int64_t), tensor_acc(C_nnz, int64_t));

    /* Get C indptr by cumulative sum */
    torch::Tensor C_indptr_t = C_nnz.cumsum(0);
    torch::Tensor C_indptr = torch::zeros({C_rows + 1}, int_tens_opts);

    C_indptr.index_put_({torch::indexing::Slice(1, torch::indexing::None, torch::indexing::None)}, C_indptr_t);

    /* Now, assemble the matrix */
    int64_t C_total_nnz = C_indptr[C_indptr.size(0) - 1].item<int64_t>();
    torch::Tensor C_data = torch::empty({C_total_nnz}, scalar_tens_opts);
    torch::Tensor C_indices = torch::empty({C_total_nnz}, int_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_forward_cuda", [&] {
        cuda_kernel_assemble_C<<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            C_rows,
            tensor_acc(Chat_V, scalar_t), tensor_acc(Chat_nnz_cumsum, int64_t), tensor_acc(Chat_J, int64_t),
            tensor_acc(C_data, scalar_t), tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t));
    });

    return {C_data, C_indices, C_indptr};
}

template <typename scalar_t>
__global__ void cuda_kernel_spmatmat_ABt_masked(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    int B_rows, int B_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    int C_rows, int C_cols,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_data.size(0)) {
        return;
    }

    /* Find row and column index */
    int64_t col = C_indices[i];
    int64_t row = 0, row_l = 0, row_h = C_rows - 1;
    while (true) {
        int64_t row_m = (row_l + row_h)/2;

        if (i >= C_indptr[row_m] && i < C_indptr[row_m+1]) {
            row = row_m;
            break;
        } else if (i < C_indptr[row_m]) {
            row_h = row_m - 1;
        } else {
            row_l = row_m + 1;
        }
    }

    /* Compute this entry as dot product of rows of A and B */
    scalar_t cij = 0.;

    int A_row_i = A_indptr[row];
    int B_row_i = B_indptr[col];

    int A_row_end = A_indptr[row + 1];
    int B_row_end = B_indptr[col + 1];

    while (A_row_i < A_row_end && B_row_i < B_row_end) {
        int A_col = A_indices[A_row_i];
        int B_col = B_indices[B_row_i];

        if (A_col < B_col) {
            A_row_i ++;
        } else if (A_col > B_col) {
            B_row_i ++;
        } else {
            cij += A_data[A_row_i] * B_data[B_row_i];
            A_row_i ++;
            B_row_i ++;
        }
    }

    C_data[i] = cij;
}

/**
 * Custom coordinate pair type for when we map i,j indices -> index in nonzeros array.
 */
struct coordinate_pair_t {
    int64_t row;
    int64_t col;

    __host__ __device__ coordinate_pair_t(int64_t _row, int64_t _col): row(_row), col(_col) {}

    __device__ bool operator==(const coordinate_pair_t& other) const {
        return (row == other.row && col == other.col);
    }
};

/**
 * Hash for the above pair type, consisting of a bitwise or between hashes of each entry.
 */
struct coordinate_pair_hash_t {
    __device__ uint32_t operator()(const coordinate_pair_t& c) {
        auto hash = cuco::detail::MurmurHash3_32<int64_t>();
        return hash(c.row) ^ hash(c.col);
    }
};

/**
 * Create a mapping between nonzero coordinates and their index in the data array.
 * Indexed on rows of the matrix.
 */
template <typename MapView>
__global__ void cuda_kernel_create_index_map(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    MapView A_index_map) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) {
        return;
    }

    for (int64_t j_i = A_indptr[i]; j_i < A_indptr[i + 1]; j_i++) {
        int64_t j = A_indices[j_i];
        A_index_map.insert({coordinate_pair_t(i, j), j_i});
    }
}

/**
 * Computes the nonzeros in each row of the matrix expansion
 * in the CSR SPGEMM product, \hat{C}=A^TB (*) mask.
 * Indexed on rows of A.
 */
template <typename MapView>
__global__ void cuda_kernel_masked_AT_Chat_expansion_nnz(
    int C_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_nnz,
    MapView output_index_map) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Multiply row of A and row of B */
    int64_t k_i, k, j_i, j;
    int64_t nnz = 0;

    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        for (j_i = B_indptr[i]; j_i < B_indptr[i + 1]; j_i++) {
            j = B_indices[j_i];

            if (output_index_map.contains(coordinate_pair_t(k, j))) {
                nnz++;
            }
        }
    }

    Chat_nnz[i] = nnz;
}

/**
 * Computes the matrix expansion in the CSR SPGEMM product, \hat{C}=A^TB (*) mask.
 * Indexed on rows of A.
 */
template <typename scalar_t, typename MapView>
__global__ void cuda_kernel_masked_AT_Chat_expansion(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_I,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_J,
    MapView output_index_map) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Grab starting index for entries of Chat */
    int64_t C_row_idx = 0;
    if (i > 0) {
        C_row_idx = Chat_indptr[i - 1];
    }

    /* Multiply row of A and row of B */
    int64_t k_i, k, j_i, j;

    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        for (j_i = B_indptr[i]; j_i < B_indptr[i + 1]; j_i++) {
            j = B_indices[j_i];

            if (output_index_map.contains(coordinate_pair_t(k, j))) {
                /* Insert the entry only if it exists in the sparsity mask */

                Chat_I[C_row_idx] = k;
                Chat_J[C_row_idx] = j;
                Chat_data[C_row_idx] = A_data[k_i] * B_data[j_i];

                C_row_idx ++;
            }
        }
    }
}

/**
 * Find the number of nonzero entries per row in a sorted COO format.
 * Indexed on the row output.
 */
__global__ void cuda_kernel_nnz_per_row_coo(
    int A_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_I,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_J,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> nnz_per_row) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) {
        return;
    }

    int64_t nnz = 0;
    for (int64_t j = 0; j < A_I.size(0); j++) {
        if (A_I[j] == i) {
            nnz++;
        }
    }

    nnz_per_row[i] = nnz;
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spgemm_backward,
               torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
               int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {

    const int C_rows = A_rows;
    const int C_cols = B_cols;
    const int threads_per_block = 512;

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    /* Compute grad_A = (grad_C B^T) (*) mask(A) */
    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor grad_A = torch::empty_like(A_data);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_backward_cuda", ([&] {
        cuda_kernel_spmatmat_ABt_masked<<<(A_data.size(0) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            C_rows, C_cols, tensor_acc(grad_C, scalar_t), tensor_acc(C_indices, int64_t), tensor_acc(C_indptr, int64_t),
            B_rows, B_cols, tensor_acc(B_data, scalar_t), tensor_acc(B_indices, int64_t), tensor_acc(B_indptr, int64_t),
            A_rows, A_cols, tensor_acc(grad_A, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));

    /* Compute grad_B = (A^T grad_C) (*) mask(B) */
    const int64_t grad_B_nnz = B_indptr[B_indptr.size(0) - 1].item<int64_t>();

    /* Create mapping of grad_B nonzeros to indices in grad_B data array */
    const float map_load_factor = 0.6f;
    cuco::static_map<coordinate_pair_t, int64_t> grad_B_idx_map(
        static_cast<size_t>(static_cast<float>(grad_B_nnz) / map_load_factor),
        cuco::sentinel::empty_key(coordinate_pair_t(-1, -1)),
        cuco::sentinel::empty_value(int64_t(-1)),
        cuco::static_map<coordinate_pair_t, int64_t>::allocator_type{},
        main_stream);

    cuda_kernel_create_index_map<<<(B_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        B_rows, B_cols, tensor_acc(B_indices, int64_t), tensor_acc(B_indptr, int64_t),
        grad_B_idx_map.get_device_mutable_view());

    /* Find NNZ in each row of \hat{grad_B}.  Note that this isn't actually a *row* in the
       output, but rather just a set of work for each thread to do. */
    torch::Tensor Bhat_nnz = torch::empty({A_rows}, int_tens_opts);
    cuda_kernel_masked_AT_Chat_expansion_nnz<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows,
        tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
        tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t),
        tensor_acc(Bhat_nnz, int64_t), grad_B_idx_map.get_device_view());

    /* Cumulative sum to find starting point for each thread to write to. */
    torch::Tensor Bhat_nnz_cumsum = Bhat_nnz.cumsum(0);
    int64_t Bhat_total_nnz = Bhat_nnz_cumsum[Bhat_nnz_cumsum.size(0) - 1].item<int64_t>();

    /* Compute the entries of Bhat via masked expansion */
    torch::Tensor Bhat_I = torch::empty({Bhat_total_nnz}, int_tens_opts);
    torch::Tensor Bhat_J = torch::empty({Bhat_total_nnz}, int_tens_opts);
    torch::Tensor Bhat_V = torch::empty({Bhat_total_nnz}, scalar_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(grad_C.type(), "spgemm_backward_cuda", ([&] {
        cuda_kernel_masked_AT_Chat_expansion<scalar_t><<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
            tensor_acc(grad_C, scalar_t), tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t),
            tensor_acc(Bhat_nnz_cumsum, int64_t), tensor_acc(Bhat_V, scalar_t), tensor_acc(Bhat_I, int64_t), tensor_acc(Bhat_J, int64_t),
            grad_B_idx_map.get_device_view());
    }));

    /* Now, lexicographically sort entries of Bhat first by column index then by row index */
    lexsort_coo_ijv(Bhat_I, Bhat_J, Bhat_V);

    /* Find the ~actual~ number of nonzeros for each row of Bhat, now that we have the output */
    Bhat_nnz.resize_(B_rows);
    cuda_kernel_nnz_per_row_coo<<<(B_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        B_rows, tensor_acc(Bhat_I, int64_t), tensor_acc(Bhat_J, int64_t), tensor_acc(Bhat_nnz, int64_t));
    Bhat_nnz_cumsum = Bhat_nnz.cumsum(0);

    /* Now, assemble the matrix */
    const int64_t B_total_nnz = B_indptr[B_indptr.size(0) - 1].item<int64_t>();
    torch::Tensor grad_B = torch::empty({B_total_nnz}, scalar_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_backward_cuda", [&] {
        cuda_kernel_assemble_C_data_only<<<(B_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            B_rows,
            tensor_acc(Bhat_V, scalar_t), tensor_acc(Bhat_nnz_cumsum, int64_t), tensor_acc(Bhat_J, int64_t),
            tensor_acc(grad_B, scalar_t), tensor_acc(B_indptr, int64_t));
    });

    return {grad_A, grad_B};
}

/**
 * Very lazy way to find nnz per column.
 * Indexed on columns of A.
 */
__global__ void cuda_kernel_csr_nnz_per_col(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_col_nnz) {

    /** TODO: think of a better way to do this...? */

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_cols) {
        return;
    }

    int64_t count = 0;
    for (int64_t j = 0; j < A_indices.size(0); j++) {
        if (A_indices[j] == i) {
            count++;
        }
    }
    A_col_nnz[i] = count;
}

template <typename scalar_t>
__global__ void cuda_kernel_csr_transpose_accumulate(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> At_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_indptr,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_to_A_idx) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_cols) {
        return;
    }
    int64_t dest_ptr = At_indptr[i];

    for (int64_t row = 0; row < A_rows; row++) {
        for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
            int64_t column = A_indices[row_i];
            if (column != i) {
                continue;
            }

            At_indices[dest_ptr] = row;
            At_data[dest_ptr] = A_data[row_i];
            At_to_A_idx[dest_ptr] = row_i;

            dest_ptr ++;
        }
    }
}

/* CSR Transpose */
FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               csr_transpose_forward,
               int A_rows, int A_columns,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    /* Based on the implementation from Scipy:
       https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L380 */

    const int64_t nnz = A_indptr[A_rows].item<int>();
    const int threads_per_block = 512;

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    torch::Tensor At_data = torch::empty(nnz, scalar_tens_opts);
    torch::Tensor At_indptr = torch::empty(A_columns + 1, int_tens_opts);
    torch::Tensor At_indices = torch::ones(nnz, int_tens_opts);

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Compute number of nonzeros per column of A */
    cuda_kernel_csr_nnz_per_col<<<(A_columns + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows, A_columns,
        tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t), tensor_acc(At_indptr, int64_t));

    /* Now, compute the cumulative sum of nnz to get starting rowptrs of A^T */
    torch::Tensor At_indptr_t = At_indptr.index({torch::indexing::Slice(0, -1)}).cumsum(0);
    At_indptr.index_put_({torch::indexing::Slice(1, torch::indexing::None, torch::indexing::None)}, At_indptr_t);
    At_indptr[0] = 0;

    /* Move data values into their correct spots */
    torch::Tensor At_to_A_idx = torch::empty(nnz, int_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_transpose_forward_cuda", ([&] {
        cuda_kernel_csr_transpose_accumulate<<<(A_columns + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows, A_columns,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
            tensor_acc(At_data, scalar_t), tensor_acc(At_indices, int64_t), tensor_acc(At_indptr, int64_t),
            tensor_acc(At_to_A_idx, int64_t));
    }));

    return {At_data, At_indices, At_indptr, At_to_A_idx};
}

FUNC_IMPL_CUDA(torch::Tensor,
               csr_transpose_backward,
               torch::Tensor grad_At, torch::Tensor At_to_A_idx) {

    const int threads_per_block = 512;

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor grad_A = torch::empty_like(grad_At);

    AT_DISPATCH_FLOATING_TYPES(grad_At.type(), "csr_transpose_backward_cuda", ([&] {
        cuda_kernel_tensor_permute<<<(grad_At.size(0) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            grad_At.size(0), tensor_acc(grad_At, scalar_t), tensor_acc(grad_A, scalar_t), tensor_acc(At_to_A_idx, int64_t), false);
    }));

    return grad_A;
}